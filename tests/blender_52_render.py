"""Run in the foreground Blender after installing the extension."""

import bpy


def check_eevee_configuration():
    server = bpy.types.blendermcp_server
    originals = [(scene, scene.render.engine, scene.eevee.taa_render_samples,
                  scene.render.use_persistent_data) for scene in bpy.data.scenes]
    spaces = [(space, space.shading.type) for screen in bpy.data.screens
              for area in screen.areas if area.type == 'VIEW_3D'
              for space in area.spaces if space.type == 'VIEW_3D']
    try:
        result = server.set_render(engine='EEVEE', samples=7, persistent_data=False)
        assert result.get('success'), result
        assert len(result['scenes']) == len(originals), result
        for scene, *_ in originals:
            assert scene.render.engine == 'BLENDER_EEVEE', scene.name
            assert scene.eevee.taa_render_samples == 7, scene.name
            assert not scene.render.use_persistent_data, scene.name
    finally:
        for scene, engine, samples, persistent in originals:
            scene.render.engine = engine
            scene.eevee.taa_render_samples = samples
            scene.render.use_persistent_data = persistent
        for space, shading in spaces:
            space.shading.type = shading
    print('PASS: Blender 5.2 EEVEE configuration applies to every scene')


check_eevee_configuration()
