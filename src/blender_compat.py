"""Helpers for the slotted animation API used by Blender 5.2 LTS."""


def action_fcurves(action, slot):
    """Return only the curves belonging to a particular action slot.

    An action can animate multiple objects. Reading all channel bags would mix
    their transforms, frame ranges, and interpolation edits. NLA callers must
    pass the strip's slot, which can differ from the object's active slot.
    """
    if action is None or slot is None:
        return ()
    return tuple(
        curve
        for layer in action.layers
        for strip in layer.strips
        if strip.type == 'KEYFRAME'
        for bag in (strip.channelbag(slot),)
        if bag is not None
        for curve in bag.fcurves
    )
