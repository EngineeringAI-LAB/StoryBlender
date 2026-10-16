"""Exercise API compatibility without generating billable images."""

import base64
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from api_endpoints import normalize_openai_base
from operators.openai_image_client import generate_openai_image
from PIL import Image


class ImageApiTests(unittest.TestCase):
    def setUp(self):
        data = BytesIO()
        Image.new("RGB", (8, 8), "blue").save(data, format="PNG")
        self.png = data.getvalue()
        self.response = Mock(ok=True, status_code=200)
        self.response.json.return_value = {"data": [{"b64_json": base64.b64encode(self.png).decode()}]}

    def test_api_bases_include_exactly_one_version(self):
        for root in ("https://api.example.test", "https://api.example.test/v1/"):
            self.assertEqual(normalize_openai_base(root), "https://api.example.test/v1")
        for root in ("https://api.openrouter.ai/v1", "https://openrouter.ai", "https://openrouter.ai/api/v1/"):
            self.assertEqual(normalize_openai_base(root), "https://openrouter.ai/api/v1")

    def test_openrouter_uses_openai_protocol_even_for_native_vendor_helpers(self):
        from any_llm import AnyLLM
        from operators.llm_completion import completion
        llm = Mock()
        llm.client = None
        with patch("operators.llm_completion.llm_client_args", return_value=({}, [])), \
             patch.object(AnyLLM, "create", return_value=llm) as create:
            completion(model="anthropic/claude-haiku-5.5", provider="gemini", messages=[],
                       api_base="https://api.openrouter.ai/v1", api_key="dummy")
        self.assertEqual(str(create.call_args.args[0]), "openai")
        self.assertEqual(create.call_args.kwargs["api_base"], "https://openrouter.ai/api/v1")
        self.assertEqual(llm.completion.call_args.kwargs["model"], "anthropic/claude-haiku-5.5")

    def test_openrouter_uses_images_api_and_decodes_base64(self):
        with patch("operators.openai_image_client.requests.post", return_value=self.response) as post:
            image = generate_openai_image("dummy", "https://openrouter.ai/api/v1", "openai/test", "test")
        self.assertEqual(image.size, (8, 8))
        self.assertEqual(post.call_args.args[0], "https://openrouter.ai/api/v1/images")
        self.assertEqual(post.call_args.kwargs["json"]["aspect_ratio"], "1:1")
        self.assertNotIn("/v1/v1/", post.call_args.args[0])

    def test_openrouter_reference_image_is_sent_as_data_url(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.png"
            path.write_bytes(self.png)
            with patch("operators.openai_image_client.requests.post", return_value=self.response) as post:
                generate_openai_image("dummy", "https://openrouter.ai/api/v1", "openai/test", "edit", path)
        reference = post.call_args.kwargs["json"]["input_references"][0]
        self.assertEqual(reference["type"], "image_url")
        self.assertTrue(reference["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_generic_provider_keeps_generation_and_edit_endpoints(self):
        with patch("operators.openai_image_client.requests.post", return_value=self.response) as post:
            generate_openai_image("dummy", "https://api.example.test/v1", "test", "test")
        self.assertEqual(post.call_args.args[0], "https://api.example.test/v1/images/generations")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.png"
            path.write_bytes(self.png)
            with patch("operators.openai_image_client.requests.post", return_value=self.response) as post:
                generate_openai_image("dummy", "https://api.example.test", "test", "edit", path)
        self.assertEqual(post.call_args.args[0], "https://api.example.test/v1/images/edits")
        self.assertIn("image", post.call_args.kwargs["files"])

    def test_url_outputs_are_downloaded_without_forwarding_api_key(self):
        self.response.json.return_value = {"data": [{"url": "https://images.example.test/result.png"}]}
        downloaded = Mock(content=self.png)
        with patch("operators.openai_image_client.requests.post", return_value=self.response), \
             patch("operators.openai_image_client.requests.get", return_value=downloaded) as get:
            self.assertEqual(generate_openai_image("dummy", "https://api.example.test", "test", "test").size, (8, 8))
        self.assertNotIn("headers", get.call_args.kwargs)

    def test_service_errors_are_reported_instead_of_returning_none(self):
        self.response.ok = False
        self.response.status_code = 402
        self.response.json.return_value = {"error": {"message": "Insufficient credits"}}
        with patch("operators.openai_image_client.requests.post", return_value=self.response), \
             self.assertRaisesRegex(RuntimeError, "402.*Insufficient credits"):
            generate_openai_image("dummy", "https://openrouter.ai/api/v1", "openai/test", "test")


if __name__ == "__main__":
    unittest.main()
