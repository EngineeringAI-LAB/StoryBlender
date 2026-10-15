"""Local Blender API regressions supplementing the Gradio Resume test."""
from pathlib import Path
import sys
import tempfile
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.operators.environment_artist_operators.import_polyhaven_textures_as_plane import (
    import_polyhaven_textures_as_plane, _enable_backface_culling,
)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    image = bpy.data.images.new('Test texture', width=2, height=2, alpha=True)
    for suffix in ('diff', 'alpha'):
        image.filepath_raw = str(root / f'test_{suffix}_1k.png')
        image.file_format = 'PNG'
        image.save()
    result = import_polyhaven_textures_as_plane(str(root))
    assert result['success'], result['message']
    material = result['material']
    assert material.surface_render_method == 'BLENDED'
    _enable_backface_culling(material)
    assert material.use_backface_culling
    assert material.surface_render_method == 'BLENDED'
    obj_path = root / 'triangle.obj'
    obj_path.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    assert bpy.ops.wm.obj_import(filepath=str(obj_path)) == {'FINISHED'}
    assert bpy.context.object.type == 'MESH'
    assert len(bpy.context.object.data.polygons) == 1
print('PASS: alpha texture import, indoor backface material, and OBJ import in Blender 5.2')
