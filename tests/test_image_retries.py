"""Retry recovery should persist success without re-queuing finished images."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from operators.concept_artist_operators import concept_artist


class ImageRetryTests(unittest.TestCase):
    def test_successful_base_and_variant_retries_clear_old_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "reference.png"
            image.write_bytes(b"existing reference")
            source = {"asset_sheet": [
                {"asset_id": "base", "text_to_image_prompt": "base", "image_prompt_error": "old failure"},
                {"asset_id": "variant", "text_to_image_prompt": "variant", "reference_character": "base",
                 "image_prompt_error": "old edit failure"},
                {"asset_id": "untouched", "text_to_image_prompt": "other", "image_prompt_error": "still failed"},
            ]}
            successful = {"image_prompt_path": str(image), "image_score": 9, "error": None}
            with patch.object(concept_artist, "_generate_single_image_prompt", return_value=successful) as generate, \
                 patch.object(concept_artist, "_generate_single_image_prompt_by_editing_image", return_value=successful) as edit:
                result = concept_artist.fetch_image_prompt(
                    source, "dummy", "test", save_path=directory,
                    model_id_list=["base", "variant"], max_concurrent=1)
            persisted = json.loads((Path(directory) / "image_prompt.json").read_text())
            self.assertEqual(persisted, result)
            for asset in persisted["asset_sheet"][:2]:
                self.assertNotIn("image_prompt_error", asset)
                self.assertEqual(asset["image_prompt_path"], str(image))
            self.assertEqual(persisted["asset_sheet"][2], source["asset_sheet"][2])
            self.assertEqual(source["asset_sheet"][0]["image_prompt_error"], "old failure")
            generate.assert_called_once()
            edit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
