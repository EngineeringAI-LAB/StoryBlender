"""Run in Blender 5.2: blender -b --factory-startup --python tests/blender_52_animation.py."""
import importlib.util
from pathlib import Path
import bpy

spec = importlib.util.spec_from_file_location('compat', Path(__file__).resolve().parents[1] / 'src/blender_compat.py')
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)

a = bpy.data.objects.new('Slot A', None)
b = bpy.data.objects.new('Slot B', None)
for obj in (a, b):
    bpy.context.scene.collection.objects.link(obj)
a.keyframe_insert(data_path='location', frame=1)
a.location.x = 2
a.keyframe_insert(data_path='location', frame=25)
action = a.animation_data.action
slot_a = a.animation_data.action_slot
b.animation_data_create()
b.animation_data.action = action
slot_b = action.slots.new(id_type='OBJECT', name=b.name)
b.animation_data.action_slot = slot_b
b.keyframe_insert(data_path='location', frame=10)
b.location.y = 5
b.keyframe_insert(data_path='location', frame=60)
curves_a = compat.action_fcurves(action, slot_a)
curves_b = compat.action_fcurves(action, slot_b)
assert len(curves_a) == len(curves_b) == 3
assert {p.co.x for fc in curves_a for p in fc.keyframe_points} == {1, 25}
assert {p.co.x for fc in curves_b for p in fc.keyframe_points} == {10, 60}
assert not set(fc.as_pointer() for fc in curves_a) & set(fc.as_pointer() for fc in curves_b)
# NLA can use a different slot from the active action.
b.animation_data.action = None
track = b.animation_data.nla_tracks.new()
strip = track.strips.new('NLA Slot A', 1, action)
strip.action_slot = slot_a
assert compat.action_fcurves(strip.action, strip.action_slot) == curves_a
assert compat.action_fcurves(action, None) == ()
assert compat.action_fcurves(None, slot_a) == ()
print('PASS: multi-slot actions, object ownership, NLA slots, and missing actions')
