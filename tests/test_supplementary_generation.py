"""Step 4 must retain the selected cloud provider through generation and retries."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gradio_app import step_4_generate_supplementary_assets_by_set_dresser as step4

artist = importlib.import_module(step4.fetch_model.__module__)


class ResultEditor:
    def __init__(self):
        self.result = None

    def update_with_result(self, result):
        if result is not None:
            self.result = result
        return ()


class SupplementaryGenerationTests(unittest.TestCase):
    def test_gradio_actions_use_selected_provider_and_recover_cloud_failure(self):
        for action in ("generate", "retry", "regenerate"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as directory:
                project = Path(directory)
                models = project / "supplementary_assets"
                models.mkdir()
                source = {"story_summary": "test scene", "asset_sheet": [{
                    "asset_id": "lamp", "description": "A brass lamp",
                    "text_to_image_prompt": "A brass lamp on white", "tags": [],
                }], "scene_details": [{"scene_id": 1}]}
                (models / "supplementary_assets.json").write_text(json.dumps(source))
                if action != "generate":
                    (models / "fetched_supplementary_assets.json").write_text(json.dumps(source))
                editor = ResultEditor()
                args = [
                    "test-llm", "https://openrouter.ai/api/v1", "openai", "test-library",
                    "test-meshy", None, None, "unused-gemini", str(project), "Meshy",
                    "latest", None, None, "http://localhost:8000", 2, 2048, "512",
                    100000, "test-vision",
                ]
                image_args = ["OpenAI", "test-image", "https://openrouter.ai/api/v1", "openai/test-image-model"]
                if action == "generate":
                    callback = step4.create_generate_supplementary_wrapper(editor)
                    args += [True] + image_args
                elif action == "retry":
                    callback = step4.create_retry_failed_supplementary_wrapper(editor)
                    args += [True] + image_args
                else:
                    callback = step4.create_regenerate_supplementary_wrapper(editor)
                    args += [["lamp"]] + image_args

                model_path = models / "lamp.glb"
                source_image = models / "lamp.png"
                attempts = []

                def cloud_generation(**kwargs):
                    attempts.append(kwargs)
                    if len(attempts) == 1:
                        raise RuntimeError("temporary cloud failure")
                    source_image.write_bytes(b"generated source image")
                    return {}

                def download(**kwargs):
                    model_path.write_bytes(b"test model")
                    return {"model_path": str(model_path), "model_id": "test-task", "thumbnail_url": None}

                def thumbnail(url, output_dir, model_id):
                    path = Path(output_dir) / f"{model_id}.png"
                    path.write_bytes(b"model preview")
                    return str(path)

                with patch.object(artist, "text_to_image_to_3d", side_effect=cloud_generation) as generate, \
                     patch.object(artist, "download_model_from_meshy", side_effect=download), \
                     patch.object(artist, "_download_thumbnail", side_effect=thumbnail), \
                     patch.object(artist, "_fetch_single_model_retrieval_only", side_effect=AssertionError("AI-only mode searched a library")):
                    list(callback(*args))

                self.assertTrue(editor.result["success"], editor.result)
                self.assertEqual(generate.call_count, 2)
                for call in generate.call_args_list:
                    self.assertEqual(call.kwargs["image_gen_platform"], "OpenAI")
                    self.assertEqual(call.kwargs["openai_api_key"], "test-image")
                    self.assertEqual(call.kwargs["openai_api_base"], "https://openrouter.ai/api/v1")
                    self.assertEqual(call.kwargs["openai_image_model"], "openai/test-image-model")
                    self.assertEqual(call.kwargs["anyllm_provider"], "openai")
                    self.assertEqual(call.kwargs["trellis2_texture_size"], 2048)
                saved = json.loads((models / "fetched_supplementary_assets.json").read_text())
                self.assertEqual(saved["scene_details"], source["scene_details"])
                self.assertFalse(saved["asset_sheet"][0].get("error"))
                self.assertEqual(step4.find_failed_supplementary_3d_ids(str(project)), [])
                self.assertEqual(source_image.read_bytes(), b"generated source image")
                self.assertTrue(saved["asset_sheet"][0]["thumbnail_url"].endswith("lamp_preview.png"))


if __name__ == "__main__":
    unittest.main()
