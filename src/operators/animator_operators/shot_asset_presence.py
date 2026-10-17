"""Character presence follows Step 8's shot modification rules."""


def active_character_actions(shot):
    removed = {mod.get('asset_id') for mod in shot.get('asset_modifications') or []
               if mod.get('modification_type') == 'remove'}
    return [action for action in shot.get('character_actions') or []
            if action.get('asset_id') not in removed]


def assets_present_in_shot(shot, all_shots, base_assets):
    present = set(base_assets)
    for earlier in all_shots:
        if earlier.get('scene_id') != shot.get('scene_id'):
            continue
        if earlier.get('shot_id', 0) > shot.get('shot_id', 0):
            continue
        for mod in earlier.get('asset_modifications') or []:
            if mod.get('modification_type') == 'add':
                present.add(mod.get('asset_id'))
    # Additions carry into later shots; removals affect their own shot only.
    present.difference_update(mod.get('asset_id') for mod in shot.get('asset_modifications') or []
                              if mod.get('modification_type') == 'remove')
    return present
