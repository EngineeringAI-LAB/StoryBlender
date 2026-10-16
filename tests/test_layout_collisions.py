"""Layout validation must not start Blender or change its shared dependencies."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from operators.layout_artist_operators.generate_layout_description import SceneLayoutVerifier


class LayoutCollisionTests(unittest.TestCase):
    def test_collision_volume_and_tolerance_without_external_processes(self):
        for offset, expected_errors in [(0.5, 1), (0.99, 0), (2.0, 0)]:
            with self.subTest(offset=offset):
                assets = [{"asset_id": name, "location": {"x": x, "y": 0, "z": 0},
                           "rotation": {"z": 0}} for name, x in [("a", 0), ("b", offset)]]
                dimensions = {name: {"w": 1, "d": 1, "h": 1} for name in ["a", "b"]}
                verifier = SceneLayoutVerifier({"assets": assets}, dimensions, scene_id=1)
                with patch("subprocess.Popen", side_effect=AssertionError("Unexpected external process")) as popen:
                    errors = verifier._check_occlusions()
                popen.assert_not_called()
                self.assertEqual(len(errors), expected_errors)
                if errors:
                    self.assertIn("vol 0.500000m³", errors[0]["detail"])


if __name__ == "__main__":
    unittest.main()
