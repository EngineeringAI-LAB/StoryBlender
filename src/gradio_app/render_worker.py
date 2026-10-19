"""Worker template appended to exported render settings and job data."""

import argparse
import bpy
import os
import sys
import time


def setup_render_settings(scene):
    scene.render.engine = RENDER_ENGINE
    scene.render.image_settings.file_format = 'PNG'
    if RENDER_SAMPLES is not None:
        if RENDER_ENGINE == 'CYCLES':
            scene.cycles.samples = RENDER_SAMPLES
        else:
            scene.eevee.taa_render_samples = RENDER_SAMPLES
    scene.render.use_persistent_data = PERSISTENT_DATA and RENDER_ENGINE == 'CYCLES'


def render_all():
    # Check metadata without activating scenes or evaluating their geometry.
    for scene_name, camera_name, start, end, _ in RENDER_JOBS:
        scene = bpy.data.scenes.get(scene_name)
        if scene is None:
            raise RuntimeError(f"Scene '{scene_name}' not found")
        camera = scene.objects.get(camera_name)
        if camera is None or camera.type != 'CAMERA':
            raise RuntimeError(f"Camera '{camera_name}' not found in '{scene_name}'")
        if end < start:
            raise RuntimeError(f"Invalid frame range for '{camera_name}': {start}–{end}")

    parser = argparse.ArgumentParser(description='One camera per supervised Blender process')
    parser.add_argument('--job-index', type=int, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    if not 0 <= args.job_index < len(RENDER_JOBS):
        raise RuntimeError('Render job index is out of range')
    ensure_memory_budget('Render worker')
    scene_name, camera_name, start, end, output = RENDER_JOBS[args.job_index]
    scene = bpy.data.scenes[scene_name]
    bpy.context.window.scene = scene
    scene.camera = scene.objects[camera_name]
    # Discard unused shots only in this transient worker; never save over the
    # source .blend. Otherwise unused scene data and textures remain resident.
    for other in list(bpy.data.scenes):
        if other != scene:
            bpy.data.scenes.remove(other)
    bpy.data.orphans_purge(do_recursive=True)
    setup_render_settings(scene)
    scene.frame_start, scene.frame_end = start, end
    scene.render.filepath = output
    os.makedirs(os.path.dirname(output), exist_ok=True)
    ensure_memory_budget('Render worker')
    started = time.monotonic()
    print(f'Rendering job {args.job_index + 1}/{len(RENDER_JOBS)}: {scene_name}, {camera_name}, frames {start}–{end}', flush=True)
    bpy.ops.render.render(animation=True, scene=scene.name)
    print(f'Completed job {args.job_index + 1} in {time.monotonic() - started:.1f}s', flush=True)


if __name__ == '__main__':
    render_all()
