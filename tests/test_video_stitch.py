"""Exercise the packaged FFmpeg when no system installation is available."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import imageio_ffmpeg
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gradio_app import step_13_stich_frames_to_video as step13


class VideoStitchTests(unittest.TestCase):
    def test_packaged_binary_and_real_frame_sequences(self):
        with tempfile.TemporaryDirectory(prefix="video stitch ") as directory:
            root = Path(directory)
            # Simulate Blender's wheel extraction losing executable permissions.
            with patch.object(step13.shutil, "which", return_value=None), \
                 patch.object(imageio_ffmpeg, "get_ffmpeg_exe", side_effect=RuntimeError("not executable")), \
                 patch.object(step13.tempfile, "gettempdir", return_value=directory):
                executable = step13.get_ffmpeg_path()
            self.assertTrue(os.access(executable, os.X_OK), executable)
            self.assertTrue(str(executable).startswith(directory))
            for shot, color in ((1, "red"), (2, "blue")):
                frames = root / "renders" / f"Scene_1_Shot_{shot}" / f"cam_1_s_1_s_{shot}"
                frames.mkdir(parents=True)
                for frame in range(5, 13):
                    Image.new("RGB", (64, 64), color).save(frames / f"frame_{frame:04d}.png")
            with patch.object(step13, "get_ffmpeg_path", return_value=executable):
                result = step13.stitch_frames_to_video(str(root / "renders"), str(root / "videos"), fps=24)
            self.assertTrue(result.get("success"), result)
            self.assertIsNone(result["errors"], result)
            self.assertEqual(len(result["created_videos"]), 3)
            with patch.dict(os.environ, {"IMAGEIO_FFMPEG_EXE": executable}):
                frames, seconds = imageio_ffmpeg.count_frames_and_secs(result["all_scenes_path"])
            self.assertEqual(frames, 16)
            self.assertAlmostEqual(seconds, 16 / 24, delta=0.01)


if __name__ == "__main__":
    unittest.main()
