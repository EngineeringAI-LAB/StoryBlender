"""Proxy integration checks using local servers and dummy keys only."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proxy_config import ProxySettings, configure_proxy, get_proxy_settings, get_proxy_url
from proxy_transport import genai_http_options, llm_client_args, proxy_category, requests_for


class FakeService(BaseHTTPRequestHandler):
    hits = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.endswith("/redirect-local"):
            type(self).hits.append((self.path, self.headers.get("Proxy-Authorization")))
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/redirected")
            self.end_headers()
            return
        self.respond({"ok": True})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if "generateContent" in self.path:
            self.respond({"candidates": [{"content": {"role": "model", "parts": [{"text": "gemini proxy ok"}]}, "finishReason": "STOP"}]})
        else:
            self.respond({"id": "local-test", "object": "chat.completion", "created": 0,
                          "model": "test", "choices": [{"index": 0, "finish_reason": "stop",
                          "message": {"role": "assistant", "content": "openai proxy ok"}}]})

    def respond(self, value):
        type(self).hits.append((self.path, self.headers.get("Proxy-Authorization")))
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeService)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.original = get_proxy_settings()
        configure_proxy(ProxySettings())
        FakeService.hits.clear()

    def tearDown(self):
        configure_proxy(self.original)

    def select(self, *categories, **kwargs):
        return configure_proxy(ProxySettings(port=self.server.server_port, categories=categories, **kwargs))

    def test_defaults_and_validation(self):
        self.assertEqual(ProxySettings().url, "http://127.0.0.1:7897")
        self.assertIsNone(get_proxy_url("llm"))
        for values in ({"host": "http://localhost"}, {"host": "localhost:123"},
                       {"host": "bad host"}, {"port": 0}, {"port": 65536},
                       {"port": 1.5}, {"port": None}, {"password": "secret"},
                       {"categories": ["unknown"]}, {"protocol": "FTP"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ProxySettings(**values)

    def test_auth_ipv6_and_redaction(self):
        settings = ProxySettings(host="[::1]", protocol="SOCKS5", username="a@b", password="p:/?#")
        self.assertEqual(settings.url, "socks5h://a%40b:p%3A%2F%3F%23@[::1]:7897")
        self.assertNotIn("p:/?#", repr(settings))
        self.assertNotIn("a@b", settings.summary())
        configure_proxy(settings.to_dict())
        self.assertEqual(get_proxy_settings(), settings)

    def test_requests_auth_and_loopback_bypass(self):
        self.select("model_generation", username="local-test", password="dummy")
        requests = requests_for("model_generation")
        result = requests.get("http://remote.invalid/generated.glb", timeout=5)
        self.assertTrue(result.json()["ok"])
        self.assertEqual(FakeService.hits[-1][0], "http://remote.invalid/generated.glb")
        self.assertTrue(FakeService.hits[-1][1].startswith("Basic "))
        requests.get(f"http://127.0.0.1:{self.server.server_port}/local", timeout=5)
        self.assertEqual(FakeService.hits[-1], ("/local", None))
        for url in ("http://localhost:7862", "http://127.1.2.3", "http://[::1]", "http://service.localhost"):
            self.assertIsNone(get_proxy_url("model_generation", url))

    def test_generated_model_preview_uses_generation_proxy_and_clears_old_error(self):
        from operators.concept_artist_operators import concept_artist

        self.select("model_generation")
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "asset.glb"
            model.write_bytes(b"existing generated model")
            source = {"asset_sheet": [{"asset_id": "asset", "image_prompt_path": str(model),
                                       "error": "previous failed task"}]}
            success = {"main_file_path": str(model), "thumbnail_url": "http://remote.invalid/preview.png",
                       "error": None}
            with patch.object(concept_artist, "_generate_single_3d_from_image", return_value=success):
                result = concept_artist.fetch_3d_from_image_prompt(
                    source, output_dir=directory, ai_platform="Meshy", max_concurrent=1)
            asset = result["asset_sheet"][0]
            self.assertEqual(FakeService.hits, [("http://remote.invalid/preview.png", None)])
            self.assertTrue(Path(asset["thumbnail_url"]).is_file())
            self.assertNotIn("error", asset)

    def test_parallel_categories_and_ambient_proxy_ignored(self):
        settings = self.select("model_generation")
        requests = requests_for("asset_retrieval")

        @proxy_category("model_generation")
        def generated():
            with requests.Session() as session:
                return session._proxies_for("https://remote.invalid")

        def retrieved():
            with requests.Session() as session:
                return session._proxies_for("https://remote.invalid")

        with patch.dict(os.environ, {"HTTP_PROXY": "http://unreachable.invalid:1", "NO_PROXY": "*"}):
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(generated if i % 2 else retrieved) for i in range(40)]
                results = [f.result() for f in futures]
            for i, result in enumerate(results):
                self.assertEqual(result["https"], settings.url if i % 2 else None)
            self.assertTrue(requests_for("model_generation").get("http://remote.invalid/test", timeout=5).ok)
            self.assertTrue(requests.get(f"http://127.0.0.1:{self.server.server_port}/direct", timeout=5).ok)
        self.assertIsNone(retrieved()["https"])

    def test_redirect_to_loopback_drops_proxy_and_credentials(self):
        self.select("animation", username="local-test", password="dummy")
        response = requests_for("animation").get("http://remote.invalid/redirect-local", timeout=5)
        self.assertTrue(response.ok)
        self.assertTrue(FakeService.hits[0][1].startswith("Basic "))
        self.assertEqual(FakeService.hits[-1], ("/redirected", None))

    def test_real_openai_and_gemini_llm_clients_use_proxy(self):
        from operators.llm_completion import completion
        self.select("llm")
        for provider in ("openai", "gemini"):
            response = completion(model="test", provider=provider, api_key="dummy-local-test",
                                  api_base="http://remote.invalid", messages=[{"role": "user", "content": "test"}])
            self.assertIn("proxy ok", response.choices[0].message.content)
        self.assertTrue(any("chat/completions" in path for path, _ in FakeService.hits))
        self.assertTrue(any("generateContent" in path for path, _ in FakeService.hits))

    def test_gemini_image_sync_and_async_clients(self):
        from google import genai
        self.select("image_generation")
        client = genai.Client(api_key="dummy-local-test", http_options=genai_http_options("image_generation", "http://remote.invalid"))
        try:
            self.assertIn("proxy ok", client.models.generate_content(model="test", contents="test").text)
            async def run():
                try:
                    return await client.aio.models.generate_content(model="test", contents="test")
                finally:
                    await client.aio.aclose()
            self.assertIn("proxy ok", asyncio.run(run()).text)
        finally:
            client.close()

    def test_direct_sdk_endpoint_and_socks_clients(self):
        from operators.llm_completion import completion
        self.select("llm")
        response = completion(model="test", provider="openai", api_key="dummy-local-test",
                              api_base=f"http://127.0.0.1:{self.server.server_port}",
                              messages=[{"role": "user", "content": "test"}])
        self.assertEqual(FakeService.hits[-1], ("/v1/chat/completions", None))
        self.assertIn("proxy ok", response.choices[0].message.content)
        self.select("llm", protocol="SOCKS5")
        args, owned = llm_client_args("openai", "https://remote.invalid")
        self.assertIs(args["http_client"], owned[0])
        asyncio.run(owned[0].aclose())
        session = requests_for("llm").Session()
        try:
            manager = session.get_adapter("https://").proxy_manager_for(get_proxy_url("llm"))
            self.assertEqual(manager.connection_pool_kw["_socks_options"]["rdns"], True)
        finally:
            session.close()

    def test_client_is_closed_if_provider_construction_fails(self):
        from any_llm import AnyLLM
        from operators.llm_completion import completion
        import httpx
        client = httpx.AsyncClient(trust_env=False)
        with patch("operators.llm_completion.llm_client_args", return_value=({"http_client": client}, [client])), \
             patch.object(AnyLLM, "create", side_effect=ValueError("test failure")):
            with self.assertRaisesRegex(ValueError, "test failure"):
                completion(model="test", provider="openai", messages=[])
        self.assertTrue(client.is_closed)

    def test_configuration_save_edit_and_invalid_input(self):
        import gradio as gr
        from gradio_app.config import create_config_ui, setup_config_handlers
        with gr.Blocks() as app:
            components = create_config_ui()
            setup_config_handlers(components)
        callbacks = {entry.fn.__name__: entry.fn for entry in app.fns.values()}
        save = callbacks["validate_and_save"]
        args = ["'/tmp/proxy-test-project'", "127.0.0.1", "HTTP", 7897, "", "", ["llm", "animation"]]
        updates = save(*args)
        self.assertEqual(get_proxy_settings().categories, {"llm", "animation"})
        self.assertEqual(updates[components["project_dir"]]["value"], "/tmp/proxy-test-project")
        self.assertFalse(updates[components["proxy_group"]]["visible"])
        self.assertTrue(callbacks["edit_configuration"]()[components["proxy_group"]]["visible"])
        args[3] = 70000
        warning = save(*args)[components["config_warning"]]
        self.assertIn("65535", warning["value"])
        self.assertTrue(warning["visible"])
        self.assertEqual(get_proxy_settings().port, 7897)
        self.assertEqual(components["proxy_password"].type, "password")

    def test_bridge_sends_settings_outside_project_parameters(self):
        from gradio_app.blender_client import BlenderClient
        settings = self.select("environment_assets", username="test", password="dummy")
        # Exercise the actual serializer while replacing only the local socket.
        from unittest.mock import MagicMock
        sock = MagicMock()
        sock.recv.return_value = b'{"status": "success", "result": {}}'
        with patch("gradio_app.blender_client.socket.socket", return_value=sock):
            BlenderClient()._send_command("get_scene_info", {"test": True})
        sent = json.loads(sock.sendall.call_args.args[0])
        self.assertEqual(sent["params"], {"test": True})
        self.assertEqual(sent["proxy_settings"], settings.to_dict())


if __name__ == "__main__":
    unittest.main()
