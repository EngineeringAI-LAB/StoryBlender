"""Run inside the foreground Blender after installing the extension."""

import json
from pathlib import Path
import tempfile

import bpy


def check_formatting_viewport_preflight():
    server = bpy.types.blendermcp_server
    viewport_areas = [area for area in bpy.context.screen.areas if area.type == 'VIEW_3D']
    assert viewport_areas, "Run this check with a 3D Viewport open"
    original_objects = set(bpy.data.objects.keys())
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        source = folder / 'input.json'
        source.write_text(json.dumps({'asset_sheet': []}))
        output = folder / 'formatted_model.json'
        output.write_text('{"previous": "verified output"}\n')
        previous_output = output.read_bytes()
        try:
            for area in viewport_areas:
                area.type = 'CONSOLE'
            result = server.format_assets(str(source), str(folder))
            assert '3D Viewport' in result.get('error', ''), result
            assert not result.get('success'), result
            assert output.read_bytes() == previous_output, "Failed preflight overwrote existing output"
            assert set(bpy.data.objects.keys()) == original_objects, "Failed preflight changed scene objects"
        finally:
            for area in viewport_areas:
                area.type = 'VIEW_3D'
    print('PASS: missing viewport stops formatting before changing models or output JSON')


check_formatting_viewport_preflight()
