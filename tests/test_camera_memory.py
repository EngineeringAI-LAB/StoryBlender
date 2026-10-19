"""Camera metadata must not activate material previews or evaluate shot meshes."""

import ast
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.operators.layout_artist_operators import camera_state
from src.gradio_app import step_10_camera_blocking_by_layout_artist as step10
from src.memory_budget import MemoryBudgetExceeded


class CameraMemoryTests(unittest.TestCase):
    def test_get_camera_info_never_switches_scene_or_changes_frames(self):
        curve = NS(data_path='location', array_index=0, mute=False,
                   keyframe_points=[NS(co=(1, 2)), NS(co=(73, 6))], evaluate=lambda frame: 2 if frame == 1 else 6)
        data = NS(lens=50, sensor_width=36, sensor_height=24, clip_start=0.1, clip_end=1000,
                  dof=NS(use_dof=False))
        camera = NS(name='cam', type='CAMERA', data=data, location=[99, 0, 2], rotation_euler=[0, 0, 0],
                    animation_data=NS(action=object(), action_slot=object()))
        scene = NS(objects=[camera], frame_start=1, frame_end=250,
                   frame_set=Mock(side_effect=AssertionError('must not evaluate scene')))
        # Any access to context.window would fail: only bpy.data is provided.
        tree = ast.parse((ROOT / 'src/blender_mcp.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BlenderMCPServer')
        function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'handle_get_camera_info')
        namespace = {'bpy': NS(data=NS(scenes={'shot': scene})), 'camera_transforms': camera_state.camera_transforms}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'camera_info', 'exec'), namespace)
        with patch.object(camera_state, 'action_fcurves', return_value=[curve]):
            result = namespace['handle_get_camera_info'](None, 'shot', 'cam')
        self.assertTrue(result['success'], result)
        self.assertEqual(result['start_transform']['location'], [2, 0, 2])
        self.assertEqual(result['end_transform']['location'], [6, 0, 2])
        self.assertEqual((result['start_frame'], result['end_frame']), (1, 73))
        self.assertEqual(camera.location, [99, 0, 2])
        scene.frame_set.assert_not_called()

    def test_finish_only_reads_placed_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'camera_blocking'
            folder.mkdir()
            data = [{'scene_id': 1, 'shot_id': 1,
                     'camera_instruction': {'camera_name': 'placed', 'placement_success': True},
                     'additional_camera_instructions': [
                         {'camera_name': 'suggested'},
                         {'camera_name': 'failed', 'placement_success': False, 'start_transform': {'location': [0, 0, 0]}},
                         {'camera_name': 'legacy', 'start_transform': {'location': [0, 0, 0]}},
                     ]}]
            (folder / 'camera_blocking_v1.json').write_text(json.dumps(data))
            client = Mock()
            client.ensure_server_running.return_value = (True, 'ready')
            client.get_camera_info.return_value = {'result': {'success': True}}
            result = step10.finish_camera_blocking(client, directory)
            self.assertTrue(result['success'], result)
            self.assertEqual([call.kwargs['camera_name'] for call in client.get_camera_info.call_args_list], ['placed', 'legacy'])

    def test_memory_stop_saves_completed_cameras_for_resume(self):
        tree = ast.parse((ROOT / 'src/operators/layout_artist_operators/camera_operator.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'camera_operator')
        guard = Mock(side_effect=[None, None, MemoryBudgetExceeded('budget exceeded')])
        place = Mock(return_value={'success': True})
        namespace = {'json': json, 'os': os, 'shutil': shutil, 'Optional': __import__('typing').Optional,
                     'List': list, 'CameraType': str, 'ensure_memory_budget': guard,
                     'MemoryBudgetExceeded': MemoryBudgetExceeded, 'create_and_place_camera_for_shot': place}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'camera_operator', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'camera_blocking'
            folder.mkdir()
            source = Path(directory) / 'input.json'
            source.write_text(json.dumps([{'scene_id': 1, 'shot_id': n, 'camera_instruction': {'camera_name': f'cam{n}'}} for n in (1, 2)]))
            result = namespace['camera_operator'](str(source), 'vision', 'unused', preview_image_save_dir=str(folder))
            self.assertFalse(result['success'])
            self.assertIn('budget exceeded', result['error'])
            self.assertEqual(place.call_count, 1)
            checkpoint = step10.get_latest_camera_blocking_json_path(directory)
            saved = json.loads(Path(checkpoint).read_text())
            self.assertTrue(saved[0]['camera_instruction']['placement_success'])
            self.assertNotIn('placement_success', saved[1]['camera_instruction'])


if __name__ == '__main__':
    unittest.main()
