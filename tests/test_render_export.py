"""Render exports must not report success for missing or unplaced cameras."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gradio_app.step_12_rendering import generate_python_render_script, generate_and_save_script


class RenderExportTests(unittest.TestCase):
    def checkpoint(self, directory, wrapped=False):
        folder = Path(directory) / "camera_blocking"
        folder.mkdir()
        shots = [{"scene_id": 1, "shot_id": 1, "camera_instruction": {
            "camera_name": "cam_1_s_1_s_1", "placement_success": True,
            "start_frame": 5, "end_frame": 9,
        }, "additional_camera_instructions": [
            {"camera_name": "suggestion"},
            {"camera_name": "failed", "placement_success": False, "start_transform": {"location": [0, 0, 0]}},
            {"camera_name": "legacy", "start_transform": {"location": [0, 0, 0]}, "start_frame": 2, "end_frame": 4},
        ]}]
        (folder / "camera_blocking_v1.json").write_text(json.dumps({"shot_details": shots} if wrapped else shots))

    def test_only_placed_cameras_are_exported_and_frame_cap_is_optional(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped), tempfile.TemporaryDirectory() as directory:
                self.checkpoint(directory, wrapped)
                for cap, expected in ((False, [14, 11]), (True, [9, 4])):
                    script, jobs, error = generate_python_render_script(directory, "project.blend", 10, "", camera_frame_cap=cap)
                    self.assertIsNone(error)
                    self.assertEqual([job["camera_name"] for job in jobs], ["cam_1_s_1_s_1", "legacy"])
                    self.assertEqual([job["end_frame"] for job in jobs], expected)
                    compile(script, "render_script.py", "exec")

    def test_missing_camera_in_later_job_aborts_before_first_render(self):
        with tempfile.TemporaryDirectory() as directory:
            self.checkpoint(directory)
            script, _, _ = generate_python_render_script(directory, "project.blend", 1, "")
            scene = SimpleNamespace(objects={"cam_1_s_1_s_1": SimpleNamespace(type="CAMERA")})
            fake_bpy = SimpleNamespace(data=SimpleNamespace(scenes={"Scene_1_Shot_1": scene}), ops=SimpleNamespace(render=SimpleNamespace(render=Mock())))
            with patch.dict(sys.modules, {"bpy": fake_bpy}):
                with self.assertRaisesRegex(RuntimeError, "legacy.*not found"):
                    exec(compile(script, "render_script.py", "exec"), {"__name__": "__main__"})
            fake_bpy.ops.render.render.assert_not_called()

    def test_shell_preserves_paths_and_propagates_blender_failure(self):
        with tempfile.TemporaryDirectory(prefix="render export ") as directory:
            self.checkpoint(directory)
            executable = Path(directory) / "fake blender"
            executable.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$0.args\"\nexit 7\n")
            executable.chmod(0o755)
            blend = str(Path(directory) / "project's shot.blend")
            _, shell, _, error = generate_and_save_script(directory, blend, 1, str(executable), "")
            self.assertIsNone(error)
            result = subprocess.run(["bash", shell], capture_output=True, text=True)
            self.assertEqual(result.returncode, 7)
            self.assertNotIn("Batch render complete.", result.stdout)
            self.assertEqual(Path(str(executable) + ".args").read_text().splitlines(), [
                "-b", blend, "--python-exit-code", "1", "--python", str(Path(directory) / "renders/render_script_v1.py")
            ])


if __name__ == "__main__":
    unittest.main()
