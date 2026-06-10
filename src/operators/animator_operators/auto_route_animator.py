"""Auto Route Animator — hybrid Meshy + Uthana pipeline.

Unified pipeline with a **single planning phase** to avoid conflicts:

1. **Plan** (once) — ``animation_planner.generate_animation_plan`` plans ALL
   character animations (text_to_motion_prompt, length, coords, rotation,
   cross-shot continuity).  This is the single source of truth.
2. **Meshy match** — for each action in ``character_actions``, try to find a
   matching Meshy animation from the library (``generate_animation_selection``).
   Only actions that get ``matched=True`` use Meshy.
3. **Meshy animate** — rig + animate matched actions via Meshy API.
   Extract duration from downloaded GLBs.
4. **Uthana animate** — unmatched actions (still in ``character_actions``)
   already have ``text_to_motion_prompt`` / ``length`` from step 1.
   Generate via Uthana API.
5. **Idle** — characters present in the scene but NOT in the input's
   ``character_actions`` get a Meshy gender-based idle (Idle_11 / Idle_12).
   No Uthana cost, no extra LLM call.
6. **Merge** — combine everything into one JSON for Blender import.
"""

import glob
import json
import os
import re
import tempfile
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple


def _find_latest_animation_plan(output_dir: str) -> Optional[str]:
    """Return path to the latest ``animation_plan_v{N}.json`` in *output_dir*,
    or ``None`` if no such file exists."""
    if not os.path.isdir(output_dir):
        return None
    pattern = re.compile(r"^animation_plan_v(\d+)\.json$")
    best_v = -1
    best_path = None
    for fn in os.listdir(output_dir):
        m = pattern.match(fn)
        if m:
            v = int(m.group(1))
            if v > best_v:
                best_v = v
                best_path = os.path.join(output_dir, fn)
    return best_path


def _save_animation_plan(plan: Dict[str, Any], output_dir: str) -> str:
    """Save *plan* as the next ``animation_plan_v{N}.json`` in *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(r"^animation_plan_v(\d+)\.json$")
    max_v = 0
    for fn in os.listdir(output_dir):
        m = pattern.match(fn)
        if m:
            max_v = max(max_v, int(m.group(1)))
    next_v = max_v + 1
    out_path = os.path.join(output_dir, f"animation_plan_v{next_v}.json")
    with open(out_path, "w") as f:
        json.dump(plan, f, indent=2)
    return out_path

try:
    from .animator import (
        generate_animation_selection,
        animate_rigged_model,
        glb_duration_seconds,
        determine_character_gender,
        _is_movement_animation,
    )
    from .animation_planner import generate_animation_plan
    from .uthana_animator import uthana_animate_characters
except ImportError:
    from animator import (
        generate_animation_selection,
        animate_rigged_model,
        glb_duration_seconds,
        determine_character_gender,
        _is_movement_animation,
    )
    from animation_planner import generate_animation_plan
    from uthana_animator import uthana_animate_characters


def _add_idle_characters(
    storyboard_script: Dict[str, Any],
    anyllm_api_key: str,
    anyllm_api_base: str = None,
    anyllm_provider: str = "gemini",
    vision_model: str = "gemini-2.5-flash",
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Add Meshy idle animations for characters in the scene but NOT in
    ``character_actions``.  Uses gender detection → Idle_11 (male) / Idle_12
    (female).  Modifies *storyboard_script* in-place and returns it."""

    shot_details = storyboard_script.get("shot_details", [])
    scene_details = storyboard_script.get("scene_details", [])
    asset_sheet = storyboard_script.get("asset_sheet", [])

    # Build helper maps
    asset_types = {a["asset_id"]: a.get("asset_type", "") for a in asset_sheet}
    asset_descs = {a["asset_id"]: a.get("description", "") for a in asset_sheet}
    scene_asset_ids: Dict[int, List[str]] = {}
    for sd in scene_details:
        sid = sd.get("scene_id")
        aids = sd.get("scene_setup", {}).get("asset_ids", [])
        if sid is not None:
            scene_asset_ids[sid] = aids

    gender_cache: Dict[str, str] = {}
    idle_count = 0

    for shot in shot_details:
        scene_id = shot.get("scene_id")
        chars_in_scene = [
            aid for aid in scene_asset_ids.get(scene_id, [])
            if asset_types.get(aid) == "character"
        ]
        existing = {a.get("asset_id") for a in (shot.get("character_actions") or [])}
        idle_chars = [aid for aid in chars_in_scene if aid not in existing]

        for aid in idle_chars:
            if aid in gender_cache:
                gender = gender_cache[aid]
            else:
                gender = determine_character_gender(
                    asset_id=aid,
                    asset_description=asset_descs.get(aid, ""),
                    anyllm_api_key=anyllm_api_key,
                    anyllm_api_base=anyllm_api_base,
                    anyllm_provider=anyllm_provider,
                    vision_model=vision_model,
                    max_retries=max_retries,
                )
                gender_cache[aid] = gender

            action_id = 252 if gender == "female" else 251
            action_name = "Idle_12" if gender == "female" else "Idle_11"

            if shot.get("character_actions") is None:
                shot["character_actions"] = []
            shot["character_actions"].append({
                "asset_id": aid,
                "action_description": "idle (default action)",
                "action_id": action_id,
                "action_name": action_name,
                "matched": True,          # pre-matched to Meshy idle
                "pipeline": "meshy_idle",
            })
            idle_count += 1
            print(f"  Idle: {aid} scene {scene_id} shot {shot.get('shot_id')} → {action_name}")

    print(f"  Added {idle_count} idle animations")
    return storyboard_script


def auto_route_animate(
    storyboard_script: Dict[str, Any],
    output_dir: str,
    meshy_api_key: str,
    uthana_api_key: str,
    anyllm_api_key: str,
    anyllm_api_base: str = None,
    anyllm_provider: str = "gemini",
    vision_model: str = "gemini-2.5-flash",
    reasoning_model: str = "gemini-3.1-pro-preview",
    uthana_fps: int = 30,
    meshy_api_base: str = "https://api.meshy.ai/openapi/v1",
    max_concurrent: int = 10,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Run the full Auto Route pipeline.

    Returns::

        {"successful_animations", "failed_animations", "total_processed", "updated_json"}
    """
    os.makedirs(output_dir, exist_ok=True)

    # Keep a copy of the original character_actions asset_ids per shot
    # so we know which were in the input vs auto-added idle
    original_action_ids: Dict[str, set] = {}  # "scene::shot" → {asset_ids}
    for shot in storyboard_script.get("shot_details", []):
        key = f"{shot.get('scene_id')}::{shot.get('shot_id')}"
        original_action_ids[key] = {
            a.get("asset_id") for a in (shot.get("character_actions") or [])
        }

    # ==================================================================
    # Phase 1: Unified LLM planning (single source of truth)
    # ==================================================================
    P = "=" * 70
    print(f"\n{P}\nAUTO ROUTE — Phase 1: Unified animation planning\n{P}")

    existing_plan_path = _find_latest_animation_plan(output_dir)
    planned: Optional[Dict[str, Any]] = None
    if existing_plan_path:
        try:
            with open(existing_plan_path, "r") as f:
                planned = json.load(f)
            print(f"  Reusing existing animation plan: {existing_plan_path}")
        except Exception as e:
            print(f"  WARNING: Failed to load existing plan ({e}); regenerating.")
            planned = None

    if planned is None:
        planned = generate_animation_plan(
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            anyllm_provider=anyllm_provider,
            reasoning_model=reasoning_model,
            storyboard_script=storyboard_script,
            max_retries=max_retries,
        )
        if planned is None:
            print("ERROR: Animation planning failed")
            return {"successful_animations": [], "failed_animations": [],
                    "total_processed": 0, "updated_json": None}
        try:
            saved_path = _save_animation_plan(planned, output_dir)
            print(f"  Saved animation plan: {saved_path}")
        except Exception as e:
            print(f"  WARNING: Failed to save animation plan: {e}")

    # ==================================================================
    # Phase 2: Meshy animation matching (only for original actions)
    # ==================================================================
    print(f"\n{P}\nAUTO ROUTE — Phase 2: Meshy animation matching\n{P}")

    # generate_animation_selection works on character_actions that exist.
    # It will try to match each action + add idle chars.
    # We feed it the PLANNED script so it sees the original character_actions.
    selection = generate_animation_selection(
        anyllm_api_key=anyllm_api_key,
        anyllm_api_base=anyllm_api_base,
        anyllm_provider=anyllm_provider,
        vision_model=vision_model,
        storyboard_script=planned,
        num_candidates=3,
        max_retries=max_retries,
        max_concurrent=max_concurrent,
    )
    if selection is None:
        print("WARNING: Meshy selection failed — all actions will go to Uthana")
        selection = planned

    # Classify actions
    def _is_original_action(shot, action):
        key = f"{shot.get('scene_id')}::{shot.get('shot_id')}"
        return action.get("asset_id") in original_action_ids.get(key, set())

    meshy_matched = 0
    uthana_needed = 0
    idle_count = 0
    for shot in selection.get("shot_details", []):
        for action in shot.get("character_actions", []):
            if not _is_original_action(shot, action):
                idle_count += 1    # auto-added idle — handled in Phase 5
            elif action.get("matched"):
                meshy_matched += 1
            else:
                uthana_needed += 1

    print(f"  Meshy matched (original): {meshy_matched}")
    print(f"  Uthana needed (original): {uthana_needed}")
    print(f"  Auto-added idle:          {idle_count}")

    # ==================================================================
    # Phase 3: Meshy rig + animate (matched original + auto-added idle)
    # ==================================================================
    print(f"\n{P}\nAUTO ROUTE — Phase 3: Meshy rig & animate\n{P}")

    meshy_script = deepcopy(selection)
    for shot in meshy_script.get("shot_details", []):
        shot["character_actions"] = [
            a for a in shot.get("character_actions", [])
            if a.get("matched") or not _is_original_action(shot, a)
        ]

    meshy_input = os.path.join(output_dir, "_auto_route_meshy.json")
    with open(meshy_input, "w") as f:
        json.dump(meshy_script, f, indent=2)

    meshy_result = animate_rigged_model(
        path_to_input_json=meshy_input,
        output_dir=output_dir,
        meshy_api_key=meshy_api_key,
        meshy_api_base=meshy_api_base,
        max_concurrent=max_concurrent,
    )
    meshy_updated = meshy_result.get("updated_json")

    # Extract durations from downloaded GLBs
    if meshy_updated:
        print("  Extracting animation durations from Meshy GLBs...")
        for shot in meshy_updated.get("shot_details", []):
            for a in shot.get("character_actions", []):
                ap = a.get("animated_path")
                if ap and os.path.exists(ap):
                    dur = glb_duration_seconds(ap)
                    if dur is not None:
                        a["duration"] = round(dur, 3)

    # Build Meshy lookup
    meshy_map: Dict[str, dict] = {}
    if meshy_updated:
        for shot in meshy_updated.get("shot_details", []):
            for a in shot.get("character_actions", []):
                if a.get("animated_path"):
                    meshy_map[f"{a.get('asset_id')}::{a.get('action_id')}"] = a

    print(f"  Meshy animated: {len(meshy_map)} unique (asset, action) pairs")

    # ==================================================================
    # Phase 4: Uthana animate (unmatched original actions)
    # ==================================================================
    uthana_map: Dict[str, dict] = {}

    if uthana_needed > 0:
        print(f"\n{P}\nAUTO ROUTE — Phase 4: Uthana animate (unmatched)\n{P}")

        # Build script with only unmatched original actions.
        # These already have text_to_motion_prompt + length + coords from Phase 1.
        uthana_script = deepcopy(selection)
        for shot in uthana_script.get("shot_details", []):
            shot["character_actions"] = [
                a for a in shot.get("character_actions", [])
                if _is_original_action(shot, a) and not a.get("matched")
            ]
            # Inject plan data (prompt, length, coords) from Phase 1
            planned_lookup = {}
            for ps in planned.get("shot_details", []):
                if ps.get("scene_id") == shot.get("scene_id") and ps.get("shot_id") == shot.get("shot_id"):
                    for pa in ps.get("character_actions", []):
                        planned_lookup[pa.get("asset_id")] = pa
            for a in shot["character_actions"]:
                src = planned_lookup.get(a.get("asset_id"), {})
                for field in ("text_to_motion_prompt", "length", "start_coords",
                              "end_coords", "start_rotation"):
                    if field in src and field not in a:
                        a[field] = src[field]

        uthana_input = os.path.join(output_dir, "_auto_route_uthana.json")
        with open(uthana_input, "w") as f:
            json.dump(uthana_script, f, indent=2)

        uthana_result = uthana_animate_characters(
            path_to_input_json=uthana_input,
            output_dir=output_dir,
            uthana_api_key=uthana_api_key,
            fps=uthana_fps,
            motion_model="text-to-motion-bucmd",
            max_retries=max_retries,
        )
        uthana_updated = uthana_result.get("updated_json")
        if uthana_updated:
            for shot in uthana_updated.get("shot_details", []):
                sid, shid = shot.get("scene_id"), shot.get("shot_id")
                for a in shot.get("character_actions", []):
                    if a.get("animated_path"):
                        uthana_map[f"{a.get('asset_id')}::s{sid}sh{shid}"] = a
        print(f"  Uthana animated: {len(uthana_map)} actions")

    # ==================================================================
    # Phase 5: Merge into a single output JSON
    # ==================================================================
    print(f"\n{P}\nAUTO ROUTE — Phase 5: Merging results\n{P}")

    merged = deepcopy(selection)
    successful = list(meshy_result.get("successful_animations", []))
    failed = list(meshy_result.get("failed_animations", []))
    total = meshy_result.get("total_processed", 0)

    # Build a lookup for plan data (coords/rotation from Phase 1)
    plan_data: Dict[str, dict] = {}  # "scene::shot::asset" → plan fields
    for shot in planned.get("shot_details", []):
        sid, shid = shot.get("scene_id"), shot.get("shot_id")
        for a in shot.get("character_actions", []):
            plan_data[f"{sid}::{shid}::{a.get('asset_id')}"] = a

    for shot in merged.get("shot_details", []):
        sid = shot.get("scene_id")
        shid = shot.get("shot_id")
        for action in shot.get("character_actions", []):
            aid = action.get("asset_id")
            pkey = f"{sid}::{shid}::{aid}"
            plan = plan_data.get(pkey, {})

            if not _is_original_action(shot, action):
                # Auto-added idle → Meshy
                mkey = f"{aid}::{action.get('action_id')}"
                src = meshy_map.get(mkey)
                if src:
                    action["animated_path"] = src.get("animated_path")
                    action["duration"] = src.get("duration")
                action["pipeline"] = "meshy_idle"

            elif action.get("matched"):
                # Original action matched by Meshy
                mkey = f"{aid}::{action.get('action_id')}"
                src = meshy_map.get(mkey)
                if src:
                    action["animated_path"] = src.get("animated_path")
                    action["duration"] = src.get("duration")
                # Coords from the unified plan
                for f in ("start_coords", "end_coords", "start_rotation"):
                    if plan.get(f) is not None:
                        action[f] = plan[f]
                action["pipeline"] = "meshy"

            else:
                # Original action unmatched → Uthana
                ukey = f"{aid}::s{sid}sh{shid}"
                src = uthana_map.get(ukey)
                if src:
                    action["animated_path"] = src.get("animated_path")
                # Prompt + coords from the unified plan
                for f in ("text_to_motion_prompt", "length",
                          "start_coords", "end_coords", "start_rotation"):
                    if plan.get(f) is not None:
                        action[f] = plan[f]
                action["pipeline"] = "uthana"
                total += 1

    animated = sum(
        1 for s in merged.get("shot_details", [])
        for a in s.get("character_actions", [])
        if a.get("animated_path")
    )
    print(f"\n  Total animated: {animated}")
    print(f"  Meshy (matched + idle): {meshy_matched + idle_count}")
    print(f"  Uthana (unmatched): {uthana_needed}")

    # Cleanup
    for tmp in ("_auto_route_meshy.json", "_auto_route_uthana.json"):
        p = os.path.join(output_dir, tmp)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    return {
        "successful_animations": successful,
        "failed_animations": failed,
        "total_processed": total,
        "updated_json": merged,
    }


def _find_latest_animated_models(output_dir: str) -> Tuple[Optional[str], int]:
    """Return ``(path, version)`` of the latest ``animated_models_v{N}.json``."""
    if not os.path.isdir(output_dir):
        return None, 0
    pattern = re.compile(r"^animated_models_v(\d+)\.json$")
    best_v = 0
    best_path = None
    for fn in os.listdir(output_dir):
        m = pattern.match(fn)
        if m:
            v = int(m.group(1))
            if v > best_v:
                best_v = v
                best_path = os.path.join(output_dir, fn)
    return best_path, best_v


def retry_failed_animations(
    storyboard_data: Dict[str, Any],
    output_dir: str,
    meshy_api_key: str,
    uthana_api_key: str,
    uthana_fps: int = 30,
    meshy_api_base: str = "https://api.meshy.ai/openapi/v1",
    max_concurrent: int = 10,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Retry animation tasks whose ``animated_path`` is missing.

    Groups failed actions by ``pipeline`` field (``meshy`` / ``meshy_idle`` go
    through the Meshy API; ``uthana`` goes through Uthana) and re-runs only
    those.  Returns the updated storyboard plus retry counts.
    """
    asset_sheet = storyboard_data.get("asset_sheet", [])
    shot_details = storyboard_data.get("shot_details", [])

    # Identify failed actions per pipeline. An action is considered "failed"
    # if its ``animated_path`` is missing OR points to a file that no longer
    # exists on disk (stale path from an earlier run whose output was deleted
    # / moved). For the latter case we clear ``animated_path`` in-place so the
    # downstream deepcopy + merge correctly re-populates it with the new path.
    failed_meshy: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    failed_uthana: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    n_stale_disk_missing = 0
    for shot in shot_details:
        for action in shot.get("character_actions", []) or []:
            animated_path = action.get("animated_path")
            if animated_path and os.path.exists(animated_path):
                continue
            if animated_path:
                # Path recorded but GLB no longer on disk — treat as failed.
                print(f"  ⚠️  Stale animated_path (file missing): {animated_path}")
                action["animated_path"] = None
                n_stale_disk_missing += 1
            pipeline = action.get("pipeline")
            if pipeline in ("meshy", "meshy_idle"):
                failed_meshy.append((shot, action))
            elif pipeline == "uthana":
                failed_uthana.append((shot, action))

    n_meshy = len(failed_meshy)
    n_uthana = len(failed_uthana)
    P = "=" * 70
    print(f"\n{P}\nRETRY FAILED ANIMATIONS\n{P}")
    print(f"  Failed meshy/meshy_idle: {n_meshy}")
    print(f"  Failed uthana:           {n_uthana}")
    if n_stale_disk_missing:
        print(f"  (of which {n_stale_disk_missing} had stale paths whose GLB was missing on disk)")

    new_meshy_paths: Dict[Tuple[Any, int], str] = {}
    new_uthana_paths: Dict[Tuple[str, str, float], str] = {}
    # Refreshed rig info captured if `animate_rigged_model` re-rigged any
    # character (missing/expired rig_task_id). Maps asset_id -> dict of rig
    # fields to merge back into the final asset_sheet.
    refreshed_rig_info: Dict[str, Dict[str, Any]] = {}

    os.makedirs(output_dir, exist_ok=True)

    # ---- Meshy retry ----
    if n_meshy > 0:
        retry_shots: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        for shot, action in failed_meshy:
            key = (shot.get("scene_id"), shot.get("shot_id"))
            if key not in retry_shots:
                retry_shots[key] = {
                    "scene_id": shot.get("scene_id"),
                    "shot_id": shot.get("shot_id"),
                    "character_actions": [],
                }
            retry_shots[key]["character_actions"].append(deepcopy(action))
        retry_storyboard = {
            "asset_sheet": asset_sheet,
            "shot_details": list(retry_shots.values()),
        }
        fd, retry_path = tempfile.mkstemp(
            prefix="_retry_meshy_", suffix=".json", dir=output_dir
        )
        os.close(fd)
        try:
            with open(retry_path, "w") as f:
                json.dump(retry_storyboard, f, indent=2)
            meshy_res = animate_rigged_model(
                path_to_input_json=retry_path,
                output_dir=output_dir,
                meshy_api_key=meshy_api_key,
                meshy_api_base=meshy_api_base,
                max_concurrent=max_concurrent,
            )
            for entry in meshy_res.get("successful_animations", []):
                aid = entry.get("asset_id")
                act_id = entry.get("action_id")
                ap = entry.get("animated_path")
                if aid is not None and act_id is not None and ap:
                    new_meshy_paths[(aid, int(act_id))] = ap
            # `animate_rigged_model` may have re-rigged characters whose
            # rig_task_id was missing/expired and persisted refreshed fields
            # back to the temp JSON. Read them so we can propagate to the
            # final saved storyboard JSON.
            try:
                with open(retry_path, "r") as _rf:
                    refreshed_storyboard = json.load(_rf)
                for refreshed_asset in refreshed_storyboard.get("asset_sheet", []) or []:
                    refreshed_id = refreshed_asset.get("asset_id")
                    if not refreshed_id:
                        continue
                    # Find the original asset_sheet entry to detect changes
                    original = next(
                        (a for a in asset_sheet if a.get("asset_id") == refreshed_id),
                        None,
                    )
                    if original is None:
                        continue
                    if (
                        refreshed_asset.get("rig_task_id") != original.get("rig_task_id")
                        or refreshed_asset.get("rig_expires_at") != original.get("rig_expires_at")
                        or refreshed_asset.get("rigged_file_path") != original.get("rigged_file_path")
                        or refreshed_asset.get("rigged_running_file_path") != original.get("rigged_running_file_path")
                    ):
                        refreshed_rig_info[refreshed_id] = {
                            "rig_task_id": refreshed_asset.get("rig_task_id"),
                            "rig_expires_at": refreshed_asset.get("rig_expires_at"),
                            "rigged_file_path": refreshed_asset.get("rigged_file_path"),
                            "rigged_running_file_path": refreshed_asset.get("rigged_running_file_path"),
                        }
            except Exception as _e:
                print(f"Warning: could not read refreshed asset_sheet from {retry_path}: {_e}")
        finally:
            try:
                os.remove(retry_path)
            except OSError:
                pass

    # ---- Uthana retry ----
    if n_uthana > 0:
        retry_shots = {}
        for shot, action in failed_uthana:
            key = (shot.get("scene_id"), shot.get("shot_id"))
            if key not in retry_shots:
                retry_shots[key] = {
                    "scene_id": shot.get("scene_id"),
                    "shot_id": shot.get("shot_id"),
                    "character_actions": [],
                }
            retry_shots[key]["character_actions"].append(deepcopy(action))
        retry_storyboard = {
            "asset_sheet": asset_sheet,
            "shot_details": list(retry_shots.values()),
        }
        fd, retry_path = tempfile.mkstemp(
            prefix="_retry_uthana_", suffix=".json", dir=output_dir
        )
        os.close(fd)
        try:
            with open(retry_path, "w") as f:
                json.dump(retry_storyboard, f, indent=2)
            uth_res = uthana_animate_characters(
                path_to_input_json=retry_path,
                output_dir=output_dir,
                uthana_api_key=uthana_api_key,
                fps=uthana_fps,
                motion_model="text-to-motion-bucmd",
                max_retries=max_retries,
            )
            for entry in uth_res.get("successful_animations", []):
                aid = entry.get("asset_id", "")
                prompt = entry.get("text_to_motion_prompt")
                length = entry.get("length", 2.0)
                ap = entry.get("animated_path")
                if prompt and ap:
                    new_uthana_paths[(aid, prompt, length)] = ap
        finally:
            try:
                os.remove(retry_path)
            except OSError:
                pass

    # ---- Merge results back into a fresh copy ----
    updated = deepcopy(storyboard_data)
    # Propagate any refreshed rig info (re-rigged characters) into the
    # final asset_sheet so subsequent runs reuse the new rig_task_id.
    if refreshed_rig_info:
        for asset in updated.get("asset_sheet", []) or []:
            aid = asset.get("asset_id")
            if aid in refreshed_rig_info:
                for k, v in refreshed_rig_info[aid].items():
                    if v is not None:
                        asset[k] = v
    retried_meshy_ok = 0
    retried_uthana_ok = 0
    still_failed = 0
    for shot in updated.get("shot_details", []):
        for action in shot.get("character_actions", []) or []:
            if action.get("animated_path"):
                continue
            pipeline = action.get("pipeline")
            if pipeline in ("meshy", "meshy_idle"):
                aid = action.get("asset_id")
                act_id = action.get("action_id")
                if aid is not None and act_id is not None:
                    p = new_meshy_paths.get((aid, int(act_id)))
                    if p:
                        action["animated_path"] = p
                        retried_meshy_ok += 1
                        continue
                still_failed += 1
            elif pipeline == "uthana":
                aid = action.get("asset_id", "")
                prompt = action.get("text_to_motion_prompt")
                length = action.get("length", 2.0)
                if prompt:
                    p = new_uthana_paths.get((aid, prompt, length))
                    if p:
                        action["animated_path"] = p
                        retried_uthana_ok += 1
                        continue
                still_failed += 1

    print(f"\n  Retried meshy/meshy_idle ok: {retried_meshy_ok}/{n_meshy}")
    print(f"  Retried uthana ok:           {retried_uthana_ok}/{n_uthana}")
    print(f"  Still failed:                {still_failed}")

    return {
        "updated_json": updated,
        "retried_meshy": retried_meshy_ok,
        "retried_uthana": retried_uthana_ok,
        "n_meshy_failed_before": n_meshy,
        "n_uthana_failed_before": n_uthana,
        "still_failed": still_failed,
    }
