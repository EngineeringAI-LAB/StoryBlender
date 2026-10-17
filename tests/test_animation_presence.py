"""Animation selection and generation must respect shot asset changes."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from operators.animator_operators import animator


class AnimationPresenceTests(unittest.TestCase):
    def test_selection_keeps_added_characters_and_omits_removed_character_actions(self):
        story = {
            'asset_sheet': [{'asset_id': aid, 'asset_type': 'character'} for aid in ['alice', 'bob', 'chris']],
            'scene_details': [{'scene_id': 1, 'scene_setup': {'asset_ids': ['alice', 'bob']}}],
            'shot_details': [
                {'scene_id': 1, 'shot_id': 1, 'character_actions': [{'asset_id': 'alice', 'action_description': 'waves'}],
                 'asset_modifications': [{'asset_id': 'chris', 'modification_type': 'add'}]},
                {'scene_id': 1, 'shot_id': 2, 'character_actions': [{'asset_id': 'bob', 'action_description': 'should not animate'}],
                 'asset_modifications': [{'asset_id': 'bob', 'modification_type': 'remove'}]},
                {'scene_id': 1, 'shot_id': 3, 'character_actions': []},
            ],
        }
        database = Mock(animations=[{}], categories=['Idle'])
        with patch.object(animator, 'AnimationDatabase', return_value=database), \
             patch.object(animator, 'determine_character_gender', return_value='male'), \
             patch.object(animator, '_process_single_action',
                          side_effect=lambda key, *args: (key, {'action_id': 1, 'action_name': 'Wave'})) as select:
            result = animator.generate_animation_selection(storyboard_script=story)
        self.assertEqual(select.call_count, 1)
        self.assertEqual([{a['asset_id'] for a in shot['character_actions']} for shot in result['shot_details']],
                         [{'alice', 'bob', 'chris'}, {'alice', 'chris'}, {'alice', 'bob', 'chris'}])
        self.assertEqual(story['shot_details'][1]['character_actions'][0]['asset_id'], 'bob')

    def test_legacy_selection_cannot_submit_meshy_jobs_for_removed_characters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selection.json'
            path.write_text(json.dumps({
                'asset_sheet': [{'asset_id': 'bob', 'asset_type': 'character',
                                 'main_file_path': '/unused.glb', 'rig_task_id': 'unused-rig'}],
                'shot_details': [{'scene_id': 1, 'shot_id': 1,
                                  'asset_modifications': [{'asset_id': 'bob', 'modification_type': 'remove'}],
                                  'character_actions': [{'asset_id': 'bob', 'action_id': 251, 'action_name': 'Idle_11'}]}],
            }))
            with patch.object(animator, 'ensure_valid_rig_task') as rig, \
                 patch.object(animator, 'apply_single_animation') as animate:
                result = animator.animate_rigged_model(str(path), directory, meshy_api_key='dummy')
            self.assertEqual(result['total_processed'], 0)
            rig.assert_not_called()
            animate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
