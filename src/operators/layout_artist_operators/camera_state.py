"""Read camera action values without evaluating or displaying another scene."""

from ...blender_compat import action_fcurves


def camera_transforms(scene, camera):
    animation = camera.animation_data
    curves = action_fcurves(animation.action, animation.action_slot) if animation else ()
    times = {int(key.co[0]) for curve in curves for key in curve.keyframe_points}
    start = min(times) if times else scene.frame_start
    end = max(times) if times else scene.frame_end

    def transform(frame):
        location = list(camera.location)
        rotation = list(camera.rotation_euler)
        for curve in curves:
            if curve.mute:
                continue
            target = {"location": location, "rotation_euler": rotation}.get(curve.data_path)
            if target is not None and 0 <= curve.array_index < len(target):
                target[curve.array_index] = curve.evaluate(frame)
        return {"location": location, "rotation": rotation}

    return {
        "start_frame": start,
        "end_frame": end,
        "start_transform": transform(start),
        "end_transform": transform(end),
        "is_animated": bool(times),
    }
