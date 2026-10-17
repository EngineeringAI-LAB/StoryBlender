"""Run in the foreground Blender after installing the extension."""

import importlib
import bpy

prefix = bpy.types.blendermcp_server.__class__.__module__.rsplit('.', 1)[0]
module = importlib.import_module(prefix + '.operators.animator_operators.import_animated_assets_to_all_shots')
original_scene = bpy.context.window.scene
# A removed character must be skipped even in older JSON with an idle action.
# The intentionally missing file makes any attempted import a test failure.
before = set(bpy.data.objects.keys())
result = module.import_animated_assets_to_shot({
    'scene_id': 999, 'shot_id': 1,
    'asset_modifications': [{'asset_id': 'removed_character', 'modification_type': 'remove'}],
    'character_actions': [{'asset_id': 'removed_character', 'animated_path': '/missing.glb'}],
}, [])
assert result.get('success'), result
assert set(bpy.data.objects.keys()) == before, 'Removed action changed Blender objects'
assert bpy.context.window.scene == original_scene, 'Removed action switched scenes'
print('PASS: animation import skips characters explicitly removed from a shot')
