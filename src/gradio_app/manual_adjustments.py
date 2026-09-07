"""Preview and persist manual Blender transform adjustments.

The workflow is deliberately two-phase:

1. Capture Blender scene state and build a complete, read-only change preview.
2. Re-capture on confirmation, verify the preview is still current, write new
   JSON versions, and finally save the open blend file.

Canonical ``asset_sheet`` width/depth/height values are left untouched. Scene
layout entries store the world-space location, XYZ Euler rotation, scale, and
axis-aligned dimensions observed in Blender.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .path_utils import load_json_with_paths, make_paths_relative


logger = logging.getLogger(__name__)

DEFAULT_LINEAR_TOLERANCE = 0.01
DEFAULT_ROTATION_TOLERANCE_DEGREES = 0.1
DEFAULT_ANIMATED_CHARACTER_LOCATION_TOLERANCE = 0.3
DEFAULT_ANIMATED_CHARACTER_DIMENSIONS_TOLERANCE = 0.2
TRANSFORM_FIELDS = ("location", "rotation", "scale", "dimensions")


def _id_sort_key(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _latest_versioned_json(project_dir: str, subfolder: str, basename: str):
    directory = os.path.join(project_dir, subfolder)
    if not os.path.isdir(directory):
        return None, 0
    pattern = re.compile(rf"^{re.escape(basename)}_v(\d+)\.json$")
    candidates = []
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            candidates.append((int(match.group(1)), os.path.join(directory, filename)))
    if not candidates:
        return None, 0
    return max(candidates, key=lambda item: item[0])[1], max(v for v, _ in candidates)


def _source_info(path: str, version: int) -> dict:
    stat = os.stat(path)
    return {
        "path": path,
        "version": version,
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def _load_sources(project_dir: str) -> Tuple[dict, List[str]]:
    specs = {
        "layout_script": ("layout_script", "layout_script"),
        "supplementary_layout_script": (
            "supplementary_layout_script",
            "supplementary_layout_script",
        ),
        "animated_models": ("animated_models", "animated_models"),
    }
    sources = {}
    warnings = []
    for key, (subfolder, basename) in specs.items():
        path, version = _latest_versioned_json(project_dir, subfolder, basename)
        if path is None:
            warnings.append(f"No {basename}_v*.json found in {subfolder}/; that file family will be skipped.")
            continue
        sources[key] = {
            "data": load_json_with_paths(path, project_dir),
            "info": _source_info(path, version),
            "subfolder": subfolder,
            "basename": basename,
        }
    if "layout_script" not in sources and "supplementary_layout_script" not in sources:
        raise FileNotFoundError(
            "No versioned layout_script or supplementary_layout_script JSON was found."
        )
    return sources, warnings


def _partition_asset_ids(data: Optional[dict]) -> Tuple[List[str], List[str]]:
    object_ids = []
    character_ids = []
    if not data:
        return object_ids, character_ids
    for asset in data.get("asset_sheet", []):
        asset_id = asset.get("asset_id")
        if not asset_id:
            continue
        if asset.get("asset_type") == "character":
            character_ids.append(asset_id)
        else:
            object_ids.append(asset_id)
    return list(dict.fromkeys(object_ids)), list(dict.fromkeys(character_ids))


def _all_scene_ids(sources: dict) -> List[Any]:
    result = set()
    for source in sources.values():
        for scene in source["data"].get("scene_details", []):
            result.add(scene.get("scene_id"))
        for shot in source["data"].get("shot_details", []):
            result.add(shot.get("scene_id"))
    return sorted((value for value in result if value is not None), key=_id_sort_key)


def _all_shot_keys(sources: dict) -> List[Tuple[Any, Any]]:
    result = set()
    for source in sources.values():
        for shot in source["data"].get("shot_details", []):
            scene_id = shot.get("scene_id")
            shot_id = shot.get("shot_id")
            if scene_id is not None and shot_id is not None:
                result.add((scene_id, shot_id))
    return sorted(result, key=lambda value: (_id_sort_key(value[0]), _id_sort_key(value[1])))


def _build_manifest(
    sources: dict,
    check_character_animations: bool,
) -> Tuple[list, list, list, list, list]:
    layout_data = sources.get("layout_script", {}).get("data")
    supplementary_data = sources.get("supplementary_layout_script", {}).get("data")
    animated_data = sources.get("animated_models", {}).get("data")
    core_object_ids, character_ids = _partition_asset_ids(
        layout_data or animated_data
    )
    supplementary_ids, _supplementary_character_ids = _partition_asset_ids(
        supplementary_data
    )
    base_ids = list(dict.fromkeys(
        core_object_ids
        + (character_ids if check_character_animations else [])
        + supplementary_ids
    ))
    shot_ids = list(dict.fromkeys(core_object_ids + supplementary_ids))
    base_scenes = [
        {
            "scene_id": scene_id,
            "asset_ids": base_ids,
            # Character roots in base scenes must retain the exact layout
            # asset name. Do not mistake a Blender numeric suffix for one.
            "exact_asset_ids": character_ids if check_character_animations else [],
        }
        for scene_id in _all_scene_ids(sources)
    ]
    shots = [
        {"scene_id": scene_id, "shot_id": shot_id, "asset_ids": shot_ids}
        for scene_id, shot_id in _all_shot_keys(sources)
    ]
    return (
        base_scenes,
        shots,
        core_object_ids,
        character_ids,
        supplementary_ids,
    )


def _unwrap_blender_response(response: dict, action: str) -> dict:
    if not isinstance(response, dict):
        raise RuntimeError(f"Blender returned an invalid response while {action}.")
    if response.get("status") == "error":
        raise RuntimeError(response.get("message") or f"Blender failed while {action}.")
    result = response.get("result", {})
    if not result.get("success"):
        raise RuntimeError(result.get("error") or f"Blender failed while {action}.")
    return result


def _snapshot_maps(snapshot: dict):
    scenes = {item.get("scene_id"): item for item in snapshot.get("scenes", [])}
    shots = {
        (item.get("scene_id"), item.get("shot_id")): item
        for item in snapshot.get("shots", [])
    }
    return scenes, shots


def _clean_transform(value: Optional[dict]) -> Optional[dict]:
    if value is None:
        return None
    return {
        field: copy.deepcopy(value.get(field))
        for field in TRANSFORM_FIELDS
        if value.get(field) is not None
    }


def _number_close(
    before,
    after,
    *,
    rotation=False,
    linear_tolerance=DEFAULT_LINEAR_TOLERANCE,
    rotation_tolerance_degrees=DEFAULT_ROTATION_TOLERANCE_DEGREES,
) -> bool:
    try:
        before = float(before)
        after = float(after)
    except (TypeError, ValueError):
        return before == after
    if rotation:
        delta = abs((after - before + 180.0) % 360.0 - 180.0)
        return delta <= rotation_tolerance_degrees
    return math.isclose(before, after, rel_tol=0.0, abs_tol=linear_tolerance)


def _vector_close(
    before: Optional[dict],
    after: Optional[dict],
    *,
    rotation=False,
    linear_tolerance=DEFAULT_LINEAR_TOLERANCE,
    rotation_tolerance_degrees=DEFAULT_ROTATION_TOLERANCE_DEGREES,
) -> bool:
    if before is None or after is None:
        return before is after
    return all(
        _number_close(
            before.get(axis),
            after.get(axis),
            rotation=rotation,
            linear_tolerance=linear_tolerance,
            rotation_tolerance_degrees=rotation_tolerance_degrees,
        )
        for axis in ("x", "y", "z")
    )


def _changed_transform_fields(
    before: Optional[dict],
    after: Optional[dict],
    *,
    animated=False,
    linear_tolerance=DEFAULT_LINEAR_TOLERANCE,
    rotation_tolerance_degrees=DEFAULT_ROTATION_TOLERANCE_DEGREES,
    location_tolerance=None,
    dimensions_tolerance=None,
):
    if before is None or after is None:
        return list(TRANSFORM_FIELDS)
    fields = []
    comparison = {
        "linear_tolerance": linear_tolerance,
        "rotation_tolerance_degrees": rotation_tolerance_degrees,
    }
    location_comparison = {
        **comparison,
        "linear_tolerance": (
            linear_tolerance
            if location_tolerance is None
            else location_tolerance
        ),
    }
    dimensions_comparison = {
        **comparison,
        "linear_tolerance": (
            linear_tolerance
            if dimensions_tolerance is None
            else dimensions_tolerance
        ),
    }
    if not _vector_close(
        before.get("location"), after.get("location"), **location_comparison
    ):
        fields.append("location")
    if not _vector_close(
        before.get("rotation"), after.get("rotation"), rotation=True, **comparison
    ):
        fields.append("rotation")
    # Animated roots have pose-dependent dimensions and an import-time uniform
    # scale adjustment, so shot-level size comparisons are not stable. Their
    # canonical size is synchronized from the base scene instead.
    if not animated:
        if not _vector_close(before.get("scale"), after.get("scale"), **comparison):
            fields.append("scale")
        if not _vector_close(
            before.get("dimensions"),
            after.get("dimensions"),
            **dimensions_comparison,
        ):
            fields.append("dimensions")
    return fields


def _scene_asset_list(scene_detail: dict) -> list:
    setup = scene_detail.setdefault("scene_setup", {})
    description = setup.setdefault("layout_description", {})
    if not isinstance(description, dict):
        description = {"description": description, "assets": []}
        setup["layout_description"] = description
    return description.setdefault("assets", [])


def _sync_base_scenes(
    data: dict,
    scene_snapshots: dict,
    asset_ids: Iterable[str],
    source_name: str,
    *,
    linear_tolerance=DEFAULT_LINEAR_TOLERANCE,
    rotation_tolerance_degrees=DEFAULT_ROTATION_TOLERANCE_DEGREES,
    asset_tolerance_overrides=None,
) -> Tuple[dict, list, list]:
    updated = copy.deepcopy(data)
    changes = []
    warnings = []
    asset_ids = list(asset_ids)
    asset_tolerance_overrides = asset_tolerance_overrides or {}

    for scene_detail in updated.get("scene_details", []):
        scene_id = scene_detail.get("scene_id")
        captured_scene = scene_snapshots.get(scene_id)
        if not captured_scene or captured_scene.get("missing_scene"):
            warnings.append(f"{source_name}: Scene_{scene_id} is missing; it was left unchanged.")
            continue

        captured_assets = captured_scene.get("assets", {})
        assets = _scene_asset_list(scene_detail)
        asset_map = {
            asset.get("asset_id"): asset
            for asset in assets
            if asset.get("asset_id")
        }
        setup = scene_detail.setdefault("scene_setup", {})
        listed_ids = list(setup.get("asset_ids") or [])

        for asset_id in asset_ids:
            captured = captured_assets.get(asset_id)
            existing = asset_map.get(asset_id)
            if captured is None:
                if existing is None:
                    if asset_id in listed_ids:
                        listed_ids = [value for value in listed_ids if value != asset_id]
                        changes.append({
                            "file": source_name,
                            "scope": "base_scene",
                            "change": "deleted",
                            "scene_id": scene_id,
                            "asset_id": asset_id,
                            "object_name": None,
                            "fields": ["asset_ids"],
                            "before": {"asset_ids": True},
                            "after": None,
                        })
                    continue
                before = _clean_transform(existing)
                assets.remove(existing)
                asset_map.pop(asset_id, None)
                listed_ids = [value for value in listed_ids if value != asset_id]
                changes.append({
                    "file": source_name,
                    "scope": "base_scene",
                    "change": "deleted",
                    "scene_id": scene_id,
                    "asset_id": asset_id,
                    "object_name": None,
                    "fields": list(TRANSFORM_FIELDS),
                    "before": before,
                    "after": None,
                })
                continue

            after = _clean_transform(captured)
            if existing is None:
                new_asset = {"asset_id": asset_id, **after}
                assets.append(new_asset)
                asset_map[asset_id] = new_asset
                if asset_id not in listed_ids:
                    listed_ids.append(asset_id)
                changes.append({
                    "file": source_name,
                    "scope": "base_scene",
                    "change": "added",
                    "scene_id": scene_id,
                    "asset_id": asset_id,
                    "object_name": captured.get("actual_name"),
                    "object_kind": captured.get("object_kind"),
                    "fields": list(TRANSFORM_FIELDS),
                    "before": None,
                    "after": after,
                })
                continue

            before = _clean_transform(existing)
            tolerance_overrides = asset_tolerance_overrides.get(asset_id, {})
            fields = _changed_transform_fields(
                before,
                after,
                linear_tolerance=linear_tolerance,
                rotation_tolerance_degrees=rotation_tolerance_degrees,
                location_tolerance=tolerance_overrides.get("location"),
                dimensions_tolerance=tolerance_overrides.get("dimensions"),
            )
            membership_added = asset_id not in listed_ids
            if membership_added:
                listed_ids.append(asset_id)
                fields.append("asset_ids")
            if not fields:
                continue
            for field in fields:
                if field == "asset_ids":
                    continue
                existing[field] = copy.deepcopy(after.get(field))
            changes.append({
                "file": source_name,
                "scope": "base_scene",
                "change": "updated",
                "scene_id": scene_id,
                "asset_id": asset_id,
                "object_name": captured.get("actual_name"),
                "object_kind": captured.get("object_kind"),
                "fields": fields,
                "comparison_tolerances": {
                    "location": tolerance_overrides.get(
                        "location", linear_tolerance
                    ),
                    "rotation": rotation_tolerance_degrees,
                    "scale": linear_tolerance,
                    "dimensions": tolerance_overrides.get(
                        "dimensions", linear_tolerance
                    ),
                },
                # Show the complete captured transform even when only one
                # field crossed its tolerance. This makes it clear that scale
                # and dimensions were read successfully.
                "before": {**before, **({"asset_ids": False} if membership_added else {})},
                "after": {**after, **({"asset_ids": True} if membership_added else {})},
            })

        setup["asset_ids"] = listed_ids
        if isinstance(setup.get("layout_descriptions"), list):
            listed_set = set(listed_ids)
            setup["layout_descriptions"] = [
                description
                for description in setup["layout_descriptions"]
                if description.get("asset_id") in listed_set
            ]

    return updated, changes, warnings


def _scene_base_state(scene_snapshots: dict, scene_id, asset_ids: Iterable[str]) -> dict:
    item = scene_snapshots.get(scene_id) or {}
    if item.get("missing_scene"):
        return {}
    return {
        asset_id: _clean_transform(item.get("assets", {}).get(asset_id))
        for asset_id in asset_ids
    }


def _shot_lookup(data: dict) -> dict:
    return {
        (shot.get("scene_id"), shot.get("shot_id")): shot
        for shot in data.get("shot_details", [])
    }


def _merge_target(transform: Optional[dict], modification: dict) -> dict:
    result = copy.deepcopy(transform) if transform is not None else {}
    target_fields = {
        "target_location": "location",
        "target_rotation": "rotation",
        "target_scale": "scale",
        "target_dimensions": "dimensions",
    }
    for target_field, field in target_fields.items():
        value = modification.get(target_field)
        if value is not None:
            result[field] = copy.deepcopy(value)
    return result


def _apply_modifications_to_state(state: dict, shot: Optional[dict]):
    if not shot:
        return
    for modification in shot.get("asset_modifications") or []:
        asset_id = modification.get("asset_id")
        if not asset_id:
            continue
        modification_type = modification.get("modification_type")
        if modification_type == "remove":
            state[asset_id] = None
        elif modification_type in ("add", "transform"):
            state[asset_id] = _merge_target(state.get(asset_id), modification)


def _action_for_asset(shot: Optional[dict], asset_id: str) -> Optional[dict]:
    if not shot:
        return None
    for action in shot.get("character_actions") or []:
        if action.get("asset_id") == asset_id:
            return action
    return None


def _expected_for_shot(state_transform: Optional[dict], action: Optional[dict]):
    expected = copy.deepcopy(state_transform)
    if action is not None and expected is None:
        # Step 9 can still import an animated action when its static base object
        # is absent, particularly when explicit start coordinates are present.
        expected = {}
    if action is None:
        return expected
    if action.get("start_coords") is not None:
        expected["location"] = copy.deepcopy(action["start_coords"])
    if action.get("start_rotation") is not None:
        expected["rotation"] = copy.deepcopy(action["start_rotation"])
    return expected


def _detect_shot_changes(
    detection_data: dict,
    scene_snapshots: dict,
    shot_snapshots: dict,
    asset_ids: Iterable[str],
    *,
    linear_tolerance=DEFAULT_LINEAR_TOLERANCE,
    rotation_tolerance_degrees=DEFAULT_ROTATION_TOLERANCE_DEGREES,
) -> Tuple[list, list]:
    """Detect manual shot changes against effective Step 9 transforms.

    Base state starts from the current Blender base scenes. This prevents a
    base-scene correction from being redundantly serialized as a transform in
    every linked shot. Existing modifications and animation start overrides
    are then applied before comparing the actual shot object.
    """
    asset_ids = list(asset_ids)
    lookup = _shot_lookup(detection_data)
    changes = []
    warnings = []
    shots_by_scene = {}
    for key in shot_snapshots:
        shots_by_scene.setdefault(key[0], []).append(key[1])

    for scene_id, shot_ids in shots_by_scene.items():
        state = _scene_base_state(scene_snapshots, scene_id, asset_ids)
        for shot_id in sorted(shot_ids, key=_id_sort_key):
            captured_shot = shot_snapshots[(scene_id, shot_id)]
            if captured_shot.get("missing_scene"):
                warnings.append(
                    f"Shot scene Scene_{scene_id}_Shot_{shot_id} is missing; its JSON was left unchanged."
                )
                continue

            baseline_shot = lookup.get((scene_id, shot_id))
            _apply_modifications_to_state(state, baseline_shot)
            captured_assets = captured_shot.get("assets", {})

            for asset_id in asset_ids:
                captured = captured_assets.get(asset_id)
                underlying = state.get(asset_id)
                action = _action_for_asset(baseline_shot, asset_id)
                expected = _expected_for_shot(underlying, action)

                if captured is None and expected is None:
                    continue
                if captured is None:
                    changes.append({
                        "scope": "shot_scene",
                        "change": "deleted",
                        "scene_id": scene_id,
                        "shot_id": shot_id,
                        "asset_id": asset_id,
                        "object_name": None,
                        "fields": list(TRANSFORM_FIELDS),
                        "before": expected,
                        "after": None,
                        "modification_type": "remove",
                        "update_action": action is not None,
                    })
                    state[asset_id] = None
                    continue

                actual = _clean_transform(captured)
                if expected is None:
                    is_animated = captured.get("object_kind") == "animated"
                    persisted_actual = (
                        {
                            "location": copy.deepcopy(actual.get("location")),
                            "rotation": copy.deepcopy(actual.get("rotation")),
                        }
                        if is_animated else actual
                    )
                    changes.append({
                        "scope": "shot_scene",
                        "change": "added",
                        "scene_id": scene_id,
                        "shot_id": shot_id,
                        "asset_id": asset_id,
                        "object_name": captured.get("actual_name"),
                        "fields": ["location", "rotation"] if is_animated else list(TRANSFORM_FIELDS),
                        "before": None,
                        "after": persisted_actual,
                        "captured_transform": persisted_actual,
                        "modification_type": "add",
                        "update_action": action is not None,
                    })
                    state[asset_id] = actual
                    continue

                is_animated = captured.get("object_kind") == "animated"
                fields = _changed_transform_fields(
                    expected,
                    actual,
                    animated=is_animated,
                    linear_tolerance=linear_tolerance,
                    rotation_tolerance_degrees=rotation_tolerance_degrees,
                )
                if not fields:
                    continue
                changes.append({
                    "scope": "shot_scene",
                    "change": "updated",
                    "scene_id": scene_id,
                    "shot_id": shot_id,
                    "asset_id": asset_id,
                    "object_name": captured.get("actual_name"),
                    "object_kind": captured.get("object_kind"),
                    "fields": fields,
                    "before": expected,
                    "after": actual,
                    "captured_transform": (
                        {
                            "location": copy.deepcopy(actual.get("location")),
                            "rotation": copy.deepcopy(actual.get("rotation")),
                        }
                        if is_animated else actual
                    ),
                    "modification_type": "transform",
                    "update_action": action is not None,
                })
                state[asset_id] = _merge_target(
                    underlying,
                    {
                        "target_location": actual.get("location"),
                        "target_rotation": actual.get("rotation"),
                        "target_scale": actual.get("scale"),
                        "target_dimensions": actual.get("dimensions"),
                    },
                )

    return changes, warnings


def _get_or_create_shot(data: dict, scene_id, shot_id) -> dict:
    for shot in data.setdefault("shot_details", []):
        if shot.get("scene_id") == scene_id and shot.get("shot_id") == shot_id:
            return shot
    shot = {"scene_id": scene_id, "shot_id": shot_id, "asset_modifications": None}
    data["shot_details"].append(shot)
    return shot


def _get_or_create_modification(shot: dict, asset_id: str) -> dict:
    modifications = shot.get("asset_modifications")
    if not isinstance(modifications, list):
        modifications = []
        shot["asset_modifications"] = modifications
    for modification in modifications:
        if modification.get("asset_id") == asset_id:
            return modification
    modification = {"asset_id": asset_id}
    modifications.append(modification)
    return modification


def _apply_shot_changes(data: dict, shot_changes: list) -> dict:
    updated = copy.deepcopy(data)
    for change in shot_changes:
        shot = _get_or_create_shot(updated, change["scene_id"], change["shot_id"])
        asset_id = change["asset_id"]
        modification = _get_or_create_modification(shot, asset_id)
        modification["modification_type"] = change["modification_type"]

        if change["change"] == "deleted":
            for key in (
                "target_location", "target_rotation", "target_scale", "target_dimensions"
            ):
                modification.pop(key, None)
            actions = shot.get("character_actions")
            if isinstance(actions, list):
                shot["character_actions"] = [
                    action for action in actions if action.get("asset_id") != asset_id
                ] or None
            continue

        actual = change.get("captured_transform") or change.get("after") or {}
        target_fields = {
            "location": "target_location",
            "rotation": "target_rotation",
            "scale": "target_scale",
            "dimensions": "target_dimensions",
        }
        for field, target_field in target_fields.items():
            if field in actual and actual[field] is not None:
                modification[target_field] = copy.deepcopy(actual[field])

        if change.get("update_action"):
            action = _action_for_asset(shot, asset_id)
            if action is not None:
                if actual.get("location") is not None:
                    action["start_coords"] = copy.deepcopy(actual["location"])
                if actual.get("rotation") is not None:
                    action["start_rotation"] = copy.deepcopy(actual["rotation"])

    return updated


def _rounded_for_signature(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _rounded_for_signature(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_rounded_for_signature(item) for item in value]
    return value


def _signature(changes: list, sources: dict, options: dict) -> str:
    source_identity = {
        key: {
            "path": value["info"]["path"],
            "version": value["info"]["version"],
            "mtime_ns": value["info"]["mtime_ns"],
            "size": value["info"]["size"],
        }
        for key, value in sources.items()
    }
    payload = _rounded_for_signature({
        "changes": changes,
        "sources": source_identity,
        "options": options,
    })
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _preview_change(change: dict) -> dict:
    result = {
        key: copy.deepcopy(change.get(key))
        for key in (
            "file", "scope", "change", "scene_id", "shot_id", "asset_id",
            "object_name", "object_kind", "fields", "before", "after",
        )
        if change.get(key) is not None
    }
    return result


def _preview_value(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        value = round(value, 6)
        if value == 0:
            value = 0.0
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _format_change_values(plan: dict, change: dict) -> str:
    labels = {
        "location": "Loc",
        "rotation": "Rot",
        "scale": "Scale",
        "dimensions": "Dims",
        "asset_ids": "Membership",
    }
    before = change.get("before")
    after = change.get("after")
    is_update = change.get("change") == "updated"
    options = plan.get("options") or {}
    comparison_tolerances = change.get("comparison_tolerances") or {}
    parts = []

    for field in change.get("fields") or []:
        before_value = before.get(field) if isinstance(before, dict) else None
        after_value = after.get(field) if isinstance(after, dict) else None
        label = labels.get(field, field)

        if isinstance(before_value, dict) or isinstance(after_value, dict):
            axis_changes = []
            for axis in ("x", "y", "z"):
                old = before_value.get(axis) if isinstance(before_value, dict) else None
                new = after_value.get(axis) if isinstance(after_value, dict) else None
                if old is None and new is None:
                    continue
                if is_update and _number_close(
                    old,
                    new,
                    rotation=field == "rotation",
                    linear_tolerance=comparison_tolerances.get(
                        field,
                        options.get("linear_tolerance", DEFAULT_LINEAR_TOLERANCE),
                    ),
                    rotation_tolerance_degrees=options.get(
                        "rotation_tolerance_degrees",
                        DEFAULT_ROTATION_TOLERANCE_DEGREES,
                    ),
                ):
                    continue
                axis_changes.append(
                    f"{axis} {_preview_value(old)} → {_preview_value(new)}"
                )
            if axis_changes:
                parts.append(f"{label}: {', '.join(axis_changes)}")
            continue

        if is_update and before_value == after_value:
            continue
        if before_value is not None or after_value is not None:
            parts.append(
                f"{label}: {_preview_value(before_value)} → "
                f"{_preview_value(after_value)}"
            )

    return "; ".join(parts) or "Changed"


def format_preview_rows(plan: dict) -> list:
    """Return compact rows suitable for a Gradio Dataframe preview."""
    file_labels = {
        "layout_script": "Layout",
        "supplementary_layout_script": "Supplementary",
        "animated_models": "Animated",
        "layout_script + animated_models": "Layout + Animated",
    }
    scope_labels = {
        "base_scene": "Base",
        "shot_scene": "Shot",
    }
    rows = []
    for index, change in enumerate(plan.get("changes", []), start=1):
        rows.append([
            index,
            file_labels.get(change.get("file"), change.get("file", "")),
            scope_labels.get(change.get("scope"), change.get("scope", "")),
            change.get("scene_id", ""),
            change.get("shot_id", "—"),
            change.get("asset_id", ""),
            change.get("change", ""),
            _format_change_values(plan, change),
        ])
    return rows


def format_preview_warnings(plan: dict) -> str:
    warnings = plan.get("warnings") or []
    if not warnings:
        return ""
    return "### Warnings\n\n" + "\n".join(f"- {warning}" for warning in warnings)


def compact_pending_plan(plan: dict) -> dict:
    """Keep only data needed to validate a later confirmation click."""
    return {
        "success": bool(plan.get("success")),
        "project_dir": plan.get("project_dir"),
        "signature": plan.get("signature"),
        "options": copy.deepcopy(plan.get("options", {})),
    }


def _print_preview_debug(plan: dict) -> None:
    changes = plan.get("changes") or []
    warnings = plan.get("warnings") or []
    options = json.dumps(
        plan.get("options") or {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    print(
        f"[StoryBlender Manual Adjustments] Detected {len(changes)} edit(s) "
        f"and {len(warnings)} warning(s). Options: {options}",
        flush=True,
    )
    for index, change in enumerate(changes, start=1):
        payload = json.dumps(
            _preview_change(change),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        print(
            f"[StoryBlender Manual Adjustments] EDIT {index}/{len(changes)} {payload}",
            flush=True,
        )
    for index, warning in enumerate(warnings, start=1):
        print(
            f"[StoryBlender Manual Adjustments] WARNING {index}/{len(warnings)} {warning}",
            flush=True,
        )


def prepare_manual_adjustments(
    project_dir: str,
    blender_client,
    linear_tolerance=DEFAULT_LINEAR_TOLERANCE,
    rotation_tolerance_degrees=DEFAULT_ROTATION_TOLERANCE_DEGREES,
    check_character_animations=False,
    animated_character_location_tolerance=(
        DEFAULT_ANIMATED_CHARACTER_LOCATION_TOLERANCE
    ),
    animated_character_dimensions_tolerance=(
        DEFAULT_ANIMATED_CHARACTER_DIMENSIONS_TOLERANCE
    ),
) -> dict:
    if not project_dir or not os.path.isabs(project_dir):
        return {"success": False, "error": "Project directory must be an absolute path."}

    try:
        linear_tolerance = float(linear_tolerance)
        rotation_tolerance_degrees = float(rotation_tolerance_degrees)
        animated_character_location_tolerance = float(
            animated_character_location_tolerance
        )
        animated_character_dimensions_tolerance = float(
            animated_character_dimensions_tolerance
        )
        if any(value < 0 for value in (
            linear_tolerance,
            rotation_tolerance_degrees,
            animated_character_location_tolerance,
            animated_character_dimensions_tolerance,
        )):
            raise ValueError("Tolerance values must be zero or greater.")
        options = {
            "linear_tolerance": linear_tolerance,
            "rotation_tolerance_degrees": rotation_tolerance_degrees,
            "check_character_animations": bool(check_character_animations),
            "animated_character_location_tolerance": (
                animated_character_location_tolerance
            ),
            "animated_character_dimensions_tolerance": (
                animated_character_dimensions_tolerance
            ),
        }
        sources, warnings = _load_sources(project_dir)
        (
            base_manifest,
            shot_manifest,
            core_object_ids,
            character_ids,
            supplementary_ids,
        ) = _build_manifest(sources, options["check_character_animations"])
        success, message = blender_client.ensure_server_running()
        if not success:
            raise RuntimeError(message)
        snapshot = _unwrap_blender_response(
            blender_client.capture_manual_adjustments(base_manifest, shot_manifest),
            "capturing manual adjustments",
        )
        if not snapshot.get("blend_filepath"):
            raise RuntimeError(
                "The current Blender file has never been saved. Save it once before scanning manual adjustments."
            )
        blend_filepath = os.path.abspath(snapshot["blend_filepath"])
        try:
            common_path = os.path.commonpath(
                [blend_filepath, os.path.abspath(project_dir)]
            )
            if common_path != os.path.abspath(project_dir):
                warnings.append(
                    f"The open blend file is outside the configured project directory: {blend_filepath}"
                )
        except ValueError:
            warnings.append(
                f"The open blend file and project directory are on different drives: {blend_filepath}"
            )
        if os.path.basename(blend_filepath).lower() != "step_9.blend":
            warnings.append(
                f"The open file is named {os.path.basename(blend_filepath)!r}, not 'step_9.blend'."
            )
        scene_snapshots, shot_snapshots = _snapshot_maps(snapshot)

        updated_data = {}
        changes = []

        if "layout_script" in sources:
            data, family_changes, family_warnings = _sync_base_scenes(
                sources["layout_script"]["data"],
                scene_snapshots,
                core_object_ids,
                "layout_script",
                linear_tolerance=linear_tolerance,
                rotation_tolerance_degrees=rotation_tolerance_degrees,
            )
            updated_data["layout_script"] = data
            changes.extend(family_changes)
            warnings.extend(family_warnings)

        if "supplementary_layout_script" in sources:
            data, family_changes, family_warnings = _sync_base_scenes(
                sources["supplementary_layout_script"]["data"],
                scene_snapshots,
                supplementary_ids,
                "supplementary_layout_script",
                linear_tolerance=linear_tolerance,
                rotation_tolerance_degrees=rotation_tolerance_degrees,
            )
            updated_data["supplementary_layout_script"] = data
            changes.extend(family_changes)
            warnings.extend(family_warnings)

        # Animated models are synchronized independently so only differences
        # relative to the latest animated JSON are copied into its new version.
        if "animated_models" in sources:
            animated_asset_ids = list(dict.fromkeys(
                core_object_ids
                + (character_ids if options["check_character_animations"] else [])
            ))
            character_tolerance_overrides = (
                {
                    asset_id: {
                        "location": animated_character_location_tolerance,
                        "dimensions": animated_character_dimensions_tolerance,
                    }
                    for asset_id in character_ids
                }
                if options["check_character_animations"]
                else {}
            )
            data, family_changes, family_warnings = _sync_base_scenes(
                sources["animated_models"]["data"],
                scene_snapshots,
                animated_asset_ids,
                "animated_models",
                linear_tolerance=linear_tolerance,
                rotation_tolerance_degrees=rotation_tolerance_degrees,
                asset_tolerance_overrides=character_tolerance_overrides,
            )
            updated_data["animated_models"] = data
            changes.extend(family_changes)
            warnings.extend(family_warnings)

        detection_data = (
            sources.get("animated_models", {}).get("data")
            or sources.get("layout_script", {}).get("data")
            or {}
        )
        shot_changes, shot_warnings = _detect_shot_changes(
            detection_data,
            scene_snapshots,
            shot_snapshots,
            # Character action roots are deliberately never inspected in shot
            # scenes because their object names are action-specific.
            list(dict.fromkeys(core_object_ids + supplementary_ids)),
            linear_tolerance=linear_tolerance,
            rotation_tolerance_degrees=rotation_tolerance_degrees,
        )
        warnings.extend(shot_warnings)
        for change in shot_changes:
            change["file"] = "layout_script + animated_models"
        changes.extend(shot_changes)

        if shot_changes:
            if "layout_script" in updated_data:
                updated_data["layout_script"] = _apply_shot_changes(
                    updated_data["layout_script"], shot_changes
                )
            if "animated_models" in updated_data:
                updated_data["animated_models"] = _apply_shot_changes(
                    updated_data["animated_models"], shot_changes
                )

        change_counts = {}
        for change in changes:
            key = f"{change['scope']}:{change['change']}"
            change_counts[key] = change_counts.get(key, 0) + 1

        plan = {
            "success": True,
            "project_dir": project_dir,
            "blend_filepath": snapshot.get("blend_filepath"),
            "sources": sources,
            "updated_data": updated_data,
            "changes": changes,
            "warnings": list(dict.fromkeys(warnings)),
            "options": options,
            "summary": {
                "total_changes": len(changes),
                "by_type": change_counts,
                "source_versions": {
                    key: value["info"]["version"] for key, value in sources.items()
                },
            },
        }
        plan["signature"] = _signature(changes, sources, options)
        plan["preview_rows"] = format_preview_rows(plan)
        plan["preview_warnings"] = format_preview_warnings(plan)
        _print_preview_debug(plan)
        return plan
    except Exception as exc:
        logger.exception("Failed to prepare manual adjustment preview")
        return {"success": False, "error": str(exc)}


def _write_new_version(data: dict, source: dict, project_dir: str) -> str:
    directory = os.path.join(project_dir, source["subfolder"])
    next_version = source["info"]["version"] + 1
    while True:
        output_path = os.path.join(
            directory, f"{source['basename']}_v{next_version}.json"
        )
        if not os.path.exists(output_path):
            break
        next_version += 1

    portable = make_paths_relative(data, project_dir)
    try:
        with open(output_path, "x", encoding="utf-8") as handle:
            json.dump(portable, handle, indent=2)
    except Exception:
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise
    return output_path


def commit_manual_adjustments(
    project_dir: str,
    pending_plan: dict,
    blender_client,
    linear_tolerance=DEFAULT_LINEAR_TOLERANCE,
    rotation_tolerance_degrees=DEFAULT_ROTATION_TOLERANCE_DEGREES,
    check_character_animations=False,
    animated_character_location_tolerance=(
        DEFAULT_ANIMATED_CHARACTER_LOCATION_TOLERANCE
    ),
    animated_character_dimensions_tolerance=(
        DEFAULT_ANIMATED_CHARACTER_DIMENSIONS_TOLERANCE
    ),
) -> dict:
    if not pending_plan or not pending_plan.get("success"):
        return {"success": False, "error": "No valid manual-adjustment preview is pending."}
    if pending_plan.get("project_dir") != project_dir:
        return {
            "success": False,
            "error": "The project directory changed after preview. Scan the adjustments again.",
        }

    # Re-capture and rebuild immediately before writing. If anything changed in
    # Blender or the source JSON files, require a new preview.
    fresh = prepare_manual_adjustments(
        project_dir,
        blender_client,
        linear_tolerance=linear_tolerance,
        rotation_tolerance_degrees=rotation_tolerance_degrees,
        check_character_animations=check_character_animations,
        animated_character_location_tolerance=(
            animated_character_location_tolerance
        ),
        animated_character_dimensions_tolerance=(
            animated_character_dimensions_tolerance
        ),
    )
    if not fresh.get("success"):
        return fresh
    if fresh.get("signature") != pending_plan.get("signature"):
        return {
            "success": False,
            "stale": True,
            "error": (
                "Blender, the source JSON files, or the adjustment options changed "
                "after the preview. "
                "Nothing was saved; click Update Manual Adjustments again."
            ),
            "fresh_plan": fresh,
        }
    if not fresh.get("changes"):
        return {"success": False, "error": "No meaningful manual adjustments were detected."}

    output_paths = []
    try:
        for key in (
            "layout_script", "supplementary_layout_script", "animated_models"
        ):
            if key not in fresh["sources"] or key not in fresh["updated_data"]:
                continue
            output_paths.append(
                _write_new_version(
                    fresh["updated_data"][key], fresh["sources"][key], project_dir
                )
            )
    except Exception as exc:
        for path in output_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        logger.exception("Failed to save manual-adjustment JSON versions")
        return {"success": False, "error": f"Failed to save JSON versions: {exc}"}

    # Per the UI contract, the blend file is saved only after every JSON file.
    try:
        blend_result = _unwrap_blender_response(
            blender_client.save_current_blend_file(), "saving the blend file"
        )
    except Exception as exc:
        return {
            "success": False,
            "json_saved": True,
            "error": f"JSON versions were saved, but the blend file could not be saved: {exc}",
            "output_paths": output_paths,
        }

    return {
        "success": True,
        "output_paths": output_paths,
        "blend_filepath": blend_result.get("filepath"),
        "summary": fresh.get("summary", {}),
        "warnings": fresh.get("warnings", []),
    }
