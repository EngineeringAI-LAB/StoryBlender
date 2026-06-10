"""Animation planner using LLM to generate text-to-motion prompts for the Uthana pipeline.

For each character in each shot, this module produces:
  - text_to_motion_prompt: a short, generic motion description (no character names)
  - length: animation duration in seconds (0.5–5.0)
  - start_coords: nullable starting coordinates (default null = use layout position)
  - end_coords: nullable ending coordinates (default null = no movement)
  - start_rotation: nullable rotation override {x, y, z} in degrees

Processing is done scene-by-scene to avoid LLM context overload.
After the initial LLM generation, a geometric verifier checks movement
directions and facing consistency, and feeds errors back to the LLM for
correction (similar to generate_layout_description.py).
"""

import math
import os
import json
import time
import gc
from typing import Any, Dict, List, Optional
from copy import deepcopy

from pydantic import BaseModel, Field

try:
    from ..llm_completion import completion
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from llm_completion import completion

import nest_asyncio
nest_asyncio.apply()

# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------

class Coords(BaseModel):
    x: float
    y: float
    z: float


class Rotation(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class CharacterAnimationPlan(BaseModel):
    asset_id: str
    text_to_motion_prompt: str = Field(
        ...,
        description="Short generic motion description, no character names, under 20 words."
    )
    length: float = Field(
        ...,
        ge=0.5,
        le=5.0,
        description="Animation length in seconds (0.5–5.0)."
    )
    start_coords: Optional[Coords] = Field(
        None,
        description="Starting position override. null = use layout position."
    )
    end_coords: Optional[Coords] = Field(
        None,
        description="Ending position. null = character stays in place."
    )
    start_rotation: Optional[Rotation] = Field(
        None,
        description="Rotation override (degrees). null = use layout rotation. "
                    "Set when the character must face a different direction than "
                    "the layout default, e.g. to face the movement direction or "
                    "face another character."
    )


class ShotAnimationPlan(BaseModel):
    shot_id: int
    character_plans: List[CharacterAnimationPlan]


class SceneAnimationPlan(BaseModel):
    shots: List[ShotAnimationPlan]


# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an Animation Planner for a 3D storyboard system. Your task is to plan character animations for each shot in a scene.

**OUTPUT REQUIREMENTS (per character per shot):**
1. `text_to_motion_prompt` – A short, physical motion description (≤20 words).
   - Use ONLY generic terms: "person", "man", "woman", "boy", "girl".
   - NEVER use character names.
   - Describe physical body motion only — no emotions, no camera directions, no dialogue.
   - Examples: "a person walks forward 2 meters and stops", "a woman stands still looking to the left", "a man raises both hands to shoulder height slowly".
2. `length` – Duration in seconds, range [0.5, 5.0].
   - Default to 1.0 for simple actions such as idle/looking.
   - Use 1.5–3.5 for walking/moving actions (based on distance).
   - Use up to 5.0 only for complex multi-step actions.
3. `start_coords` – Set to null unless the character's starting position differs from their layout position. Almost always null.
4. `end_coords` – Set to null if the character stays in place. If the character walks/moves, compute the end position from the layout coordinates plus the movement vector.  Use the coordinate system where +X=right, +Y=back, +Z=up.
5. `start_rotation` – Rotation override in degrees {x, y, z}. Set to **null** to keep the layout rotation.
   Set it when the character's layout rotation does NOT match the required facing:
   - **Moving characters:** The character MUST face the direction of travel.
     If end_coords is set, compute `rotation_z = atan2(end_x - start_x, -(end_y - start_y))` (in degrees) so the character's front faces the movement vector. Only set `z`; keep `x` and `y` at 0.
   - **Interacting characters:** Two characters talking or looking at each other should face each other. Use the same formula with the other character's position as the target.
   - **Idle characters with correct layout rotation:** Leave null.

**CROSS-SHOT CONTINUITY (CRITICAL):**
- Shots in a scene are sequential. If a character moves to `end_coords` in Shot N, then in Shot N+1 that character starts at those coordinates, NOT at the original layout position.
- Therefore: for Shot 2+, compute each character's effective start position = previous shot's `end_coords` (if set) or previous shot's start position (if stationary).
- When setting `start_coords` for a shot: set it explicitly to the previous shot's end position. Only set to null for **Shot 1** (uses layout position) or if the character didn't move in any prior shot.
- Rotation continuity: if a character's `start_rotation` was set in a previous shot, subsequent shots should keep that rotation unless the action requires a different facing.

**CRITICAL RULES:**
- You MUST plan for ALL characters present in the scene for EVERY shot, even if a character has no explicit action in a shot. For characters without explicit actions, generate an appropriate idle animation (e.g., "a person stands still", "a woman stands idle looking ahead").
- The character list for the scene is provided in the "Characters in this scene" section. Every character listed there must appear in every shot's plan.
- Keep motion prompts simple and physically descriptive.
- For walking actions with distance mentioned, set `end_coords` based on the character's **effective start position** (which may be the previous shot's end_coords) plus the movement vector.
- When `end_coords` is set, you MUST also set `start_rotation` so the character faces the movement direction, unless the current rotation already points that way (within ±45°).
- Output ONLY the JSON. No explanations.

**CHARACTER BREATHING ROOM & PATH COLLISION (CRITICAL):**
Characters need extra XY clearance because their animated silhouettes extend beyond their rest-pose bounding box (swinging arms, walking stride, gestures). Treat every character as if its footprint is padded by 0.5 m on every horizontal side (+X, -X, +Y, -Y).

- **Stationary characters:** keep their centres at least `(padded_half_A + padded_half_B)` apart from every other character in the same shot, i.e. roughly `max(width, depth)_A / 2 + 0.5 + max(width, depth)_B / 2 + 0.5` metres.
- **Moving characters:** conceptually sweep the padded footprint along the straight line from `start_coords` (or the effective start position) to `end_coords`. This sweep defines an axis-aligned box occupied by the character during the shot. Two characters must NOT have overlapping swept boxes in the same shot — otherwise their animations will collide.
- If a collision is unavoidable along the direct line, **route one character around the other** by splitting the motion across shots, or choose `end_coords` that clears the other character's swept box by at least 0.5 m on every side.
- This applies to every pair of characters active in the same shot, including stationary ones (treat stationary as a zero-length sweep at their effective position).

**COORDINATE SYSTEM:**
- X-Axis: Positive = Right, Negative = Left
- Y-Axis: Positive = Back (away from camera), Negative = Front (toward camera)
- Z-Axis: Positive = Up
- Default view is from -Y toward +Y.
- A character with rotation_z = 0 faces toward -Y (toward camera).
- A character with rotation_z = 90 faces toward +X (right).
- A character with rotation_z = 180 faces toward +Y (away from camera).
- A character with rotation_z = -90 faces toward -X (left).
- Forward vector at rotation_z = R is: (sin(R), -cos(R)).
- "Walking forward" for a character means moving in their facing direction.
"""


def _get_scene_outline(storyboard_outline: List[dict], scene_id: int) -> Optional[dict]:
    """Extract the storyboard outline entry for a given scene_id."""
    for entry in storyboard_outline:
        if entry.get("scene_id") == scene_id:
            return entry
    return None


def _get_scene_detail(scene_details: List[dict], scene_id: int) -> Optional[dict]:
    """Extract the scene_details entry for a given scene_id."""
    for entry in scene_details:
        if entry.get("scene_id") == scene_id:
            return entry
    return None


def _get_shots_for_scene(shot_details: List[dict], scene_id: int) -> List[dict]:
    """Extract all shot_details entries for a given scene_id, sorted by shot_id."""
    shots = [s for s in shot_details if s.get("scene_id") == scene_id]
    shots.sort(key=lambda s: s.get("shot_id", 0))
    return shots


def _get_characters_in_scene(
    scene_detail: dict,
    asset_sheet: List[dict],
) -> List[dict]:
    """Return asset_sheet entries for characters present in this scene."""
    asset_ids = scene_detail.get("scene_setup", {}).get("asset_ids", [])
    asset_map = {a["asset_id"]: a for a in asset_sheet}
    characters = []
    for aid in asset_ids:
        asset = asset_map.get(aid)
        if asset and asset.get("asset_type") == "character":
            characters.append(asset)
    return characters


def _get_layout_coords(
    scene_detail: dict,
    asset_sheet: Optional[List[dict]] = None,
) -> Dict[str, dict]:
    """Extract {asset_id: {x, y, z, rotation_z, w, d, h, asset_type}} from layout_description.

    If *asset_sheet* is provided, per-asset dimensions and asset_type are
    merged into the returned dict so downstream verifiers can reason about
    character footprints.
    """
    layout = scene_detail.get("scene_setup", {}).get("layout_description", {})
    assets = layout.get("assets", [])
    dims_lookup: Dict[str, dict] = {}
    if asset_sheet:
        for a in asset_sheet:
            aid = a.get("asset_id")
            if aid:
                dims_lookup[aid] = {
                    "w": a.get("width") or 0.0,
                    "d": a.get("depth") or 0.0,
                    "h": a.get("height") or 0.0,
                    "asset_type": a.get("asset_type"),
                }
    coords = {}
    for a in assets:
        loc = a.get("location", {})
        rot = a.get("rotation", {})
        aid = a["asset_id"]
        entry = {
            "x": loc.get("x", 0.0),
            "y": loc.get("y", 0.0),
            "z": loc.get("z", 0.0),
            "rotation_z": rot.get("z", 0),
        }
        entry.update(dims_lookup.get(aid, {}))
        coords[aid] = entry
    return coords


def _build_user_prompt(
    scene_outline: dict,
    scene_detail: dict,
    scene_shots: List[dict],
    characters: List[dict],
    layout_coords: Dict[str, dict],
) -> str:
    """Build the user prompt for a single scene."""
    parts = []

    # Scene description
    parts.append(f"## Scene {scene_outline['scene_id']}")
    parts.append(f"**Scene Description:** {scene_outline.get('scene_description', '')}")
    parts.append("")

    # Characters in this scene
    parts.append("## Characters in this scene")
    for ch in characters:
        aid = ch["asset_id"]
        desc = ch.get("description", "")
        coords = layout_coords.get(aid, {})
        w = ch.get("width")
        d = ch.get("depth")
        h = ch.get("height")
        dim_str = (
            f"width={w if w is not None else 'N/A'}, "
            f"depth={d if d is not None else 'N/A'}, "
            f"height={h if h is not None else 'N/A'}"
        )
        parts.append(
            f"- **{aid}**: {desc}\n"
            f"  Layout position: x={coords.get('x', 0)}, y={coords.get('y', 0)}, z={coords.get('z', 0)}, "
            f"rotation_z={coords.get('rotation_z', 0)}\n"
            f"  Dimensions (m): {dim_str}"
        )
    parts.append("")

    # Shots
    parts.append("## Shots")
    for shot in scene_shots:
        shot_id = shot.get("shot_id")
        # Find matching shot description from outline
        shot_desc = ""
        for s in scene_outline.get("shots", []):
            if s.get("shot_id") == shot_id:
                shot_desc = s.get("shot_description", "")
                break

        parts.append(f"### Shot {shot_id}")
        parts.append(f"**Shot Description:** {shot_desc}")

        # Character actions
        actions = shot.get("character_actions", [])
        if actions:
            parts.append("**Character Actions:**")
            for a in actions:
                parts.append(f"- {a['asset_id']}: {a.get('action_description', '')}")
        else:
            parts.append("**Character Actions:** None specified (all characters idle).")

        # Asset modifications (may affect positions)
        mods = shot.get("asset_modifications")
        if mods:
            parts.append("**Asset Modifications:**")
            for m in mods:
                parts.append(
                    f"- {m.get('asset_id')}: {m.get('modification_type')} — {m.get('description', '')}"
                )
        parts.append("")

    parts.append(
        "Now generate the animation plan for ALL characters listed above in EVERY shot. "
        "Output only the JSON matching the schema."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Geometric verifier (movement direction ↔ facing consistency)
# ---------------------------------------------------------------------------

_ANGLE_TOLERANCE_DEG = 45.0  # cone half-angle considered "facing"

# Extra breathing room (meters) added on every horizontal side of a
# character's footprint to account for animation reach (arms swinging,
# walking stride, gestures, etc.).  Matches the layout-stage value in
# ``generate_layout_description.SceneLayoutVerifier.CHARACTER_XY_PADDING``.
CHARACTER_XY_PADDING = 0.5

# Fallback side length (meters) used when a character has no dimensions on
# its asset_sheet entry.  A typical stylised human footprint is ~0.6 m.
DEFAULT_CHARACTER_FOOTPRINT = 0.6


def _character_footprint_half(lc: dict) -> float:
    """Half-extent of a character's padded square footprint (XY)."""
    w = lc.get("w") or 0.0
    d = lc.get("d") or 0.0
    side = max(w, d)
    if side <= 0:
        side = DEFAULT_CHARACTER_FOOTPRINT
    return side / 2.0 + CHARACTER_XY_PADDING


def _swept_aabb(
    sx: float, sy: float,
    ex: float, ey: float,
    half: float,
) -> Dict[str, float]:
    """Axis-aligned bounding box of a character's padded footprint swept
    along the straight line from (sx, sy) to (ex, ey).

    Returns dict with ``min_x``, ``max_x``, ``min_y``, ``max_y``.
    """
    return {
        "min_x": min(sx, ex) - half,
        "max_x": max(sx, ex) + half,
        "min_y": min(sy, ey) - half,
        "max_y": max(sy, ey) + half,
    }


def _aabb_overlap(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """Return True if two axis-aligned 2D bounding boxes overlap."""
    return (
        a["max_x"] > b["min_x"] and b["max_x"] > a["min_x"]
        and a["max_y"] > b["min_y"] and b["max_y"] > a["min_y"]
    )


def _normalize_angle(deg: float) -> float:
    """Normalize an angle to (-180, 180]."""
    deg = deg % 360
    if deg > 180:
        deg -= 360
    return deg


def _angle_between(a_deg: float, b_deg: float) -> float:
    """Smallest unsigned angle between two headings in degrees."""
    diff = _normalize_angle(a_deg - b_deg)
    return abs(diff)


def _heading_from_to(sx: float, sy: float, ex: float, ey: float) -> float:
    """Compute rotation_z (degrees) so that the forward vector (sin R, -cos R)
    points from (sx, sy) toward (ex, ey).
    Returns degrees in (-180, 180]."""
    dx = ex - sx
    dy = ey - sy
    return math.degrees(math.atan2(dx, -dy))


def _verify_animation_plan(
    plan: dict,
    layout_coords: Dict[str, dict],
    scene_id: int,
) -> List[Dict[str, Any]]:
    """Check every character plan for direction / facing errors AND
    cross-shot continuity errors.

    Returns a list of error dicts, each with keys:
        shot_id, asset_id, error_type, detail, fix
    """
    errors: List[Dict[str, Any]] = []

    # Sort shots for sequential continuity checking
    sorted_shots = sorted(plan.get("shots", []), key=lambda s: s["shot_id"])

    # Track each character's effective end position across shots
    # asset_id → {"x": ..., "y": ...}
    prev_end: Dict[str, dict] = {}

    for shot_plan in sorted_shots:
        shot_id = shot_plan["shot_id"]

        # Per-character swept AABB collected for this shot so we can compare
        # all character pairs once at the end of the shot loop.
        shot_swept: Dict[str, Dict[str, Any]] = {}

        for cp in shot_plan.get("character_plans", []):
            aid = cp["asset_id"]
            lc = layout_coords.get(aid, {})

            # --- Resolve effective start position ---
            sc = cp.get("start_coords")
            if sc:
                sx, sy = sc["x"], sc["y"]
            elif aid in prev_end:
                # Continuity: should start where previous shot ended
                sx, sy = prev_end[aid]["x"], prev_end[aid]["y"]
            else:
                sx, sy = lc.get("x", 0), lc.get("y", 0)

            # --- Continuity check: if character moved in prev shot,
            #     this shot's start_coords should match prev end ---
            if aid in prev_end:
                pe = prev_end[aid]
                expected_x, expected_y = pe["x"], pe["y"]

                actual_start_x = sc["x"] if sc else lc.get("x", 0)
                actual_start_y = sc["y"] if sc else lc.get("y", 0)
                gap = math.hypot(actual_start_x - expected_x, actual_start_y - expected_y)

                if gap > 0.1 and sc is None:
                    # start_coords is null but character moved in prev shot
                    errors.append({
                        "shot_id": shot_id,
                        "asset_id": aid,
                        "error_type": "continuity_break",
                        "detail": (
                            f"Character ended at ({expected_x:.2f},{expected_y:.2f}) in previous shot "
                            f"but start_coords is null (would default to layout position "
                            f"({lc.get('x',0):.2f},{lc.get('y',0):.2f}), gap={gap:.2f})."
                        ),
                        "fix": (
                            f"Set start_coords to {{\"x\": {expected_x:.2f}, \"y\": {expected_y:.2f}, \"z\": 0}} "
                            f"to maintain continuity from the previous shot."
                        ),
                    })

            # --- Resolve effective facing (rotation_z) ---
            sr = cp.get("start_rotation")
            if sr:
                facing_z = sr.get("z", 0)
            else:
                facing_z = lc.get("rotation_z", 0)

            # --- Movement direction check ---
            ec = cp.get("end_coords")
            if ec is not None:
                ex, ey = ec["x"], ec["y"]
                dist = math.hypot(ex - sx, ey - sy)
                if dist > 0.05:
                    required_z = _heading_from_to(sx, sy, ex, ey)
                    deviation = _angle_between(facing_z, required_z)
                    if deviation > _ANGLE_TOLERANCE_DEG:
                        errors.append({
                            "shot_id": shot_id,
                            "asset_id": aid,
                            "error_type": "movement_direction_mismatch",
                            "detail": (
                                f"Character moves from ({sx:.2f},{sy:.2f}) to ({ex:.2f},{ey:.2f}) "
                                f"(heading {required_z:.1f}°) but faces {facing_z:.1f}° "
                                f"(deviation {deviation:.1f}° > {_ANGLE_TOLERANCE_DEG}°)."
                            ),
                            "fix": (
                                f"Set start_rotation to {{\"x\": 0, \"y\": 0, \"z\": {required_z:.1f}}} "
                                f"so the character faces the direction of travel."
                            ),
                        })

            # --- Update prev_end for next shot ---
            if ec:
                prev_end[aid] = {"x": ec["x"], "y": ec["y"]}
            else:
                # Stationary: end = start
                prev_end[aid] = {"x": sx, "y": sy}

            # --- Record swept AABB for path-collision check (characters only) ---
            if lc.get("asset_type") == "character":
                ex_eff, ey_eff = (ec["x"], ec["y"]) if ec is not None else (sx, sy)
                half = _character_footprint_half(lc)
                shot_swept[aid] = {
                    "aabb": _swept_aabb(sx, sy, ex_eff, ey_eff, half),
                    "start": (sx, sy),
                    "end": (ex_eff, ey_eff),
                    "half": half,
                    "moved": ec is not None and math.hypot(ex_eff - sx, ey_eff - sy) > 0.05,
                }

        # --- Pairwise character path-collision check for this shot ---
        swept_items = list(shot_swept.items())
        for i in range(len(swept_items)):
            for j in range(i + 1, len(swept_items)):
                aid_a, info_a = swept_items[i]
                aid_b, info_b = swept_items[j]
                if not _aabb_overlap(info_a["aabb"], info_b["aabb"]):
                    continue
                sa, ea = info_a["start"], info_a["end"]
                sb, eb = info_b["start"], info_b["end"]
                # Required minimum centre-to-centre distance at every point
                # along both swept paths so the padded footprints don't touch.
                min_clearance = info_a["half"] + info_b["half"]
                if info_a["moved"] and info_b["moved"]:
                    detail = (
                        f"Movement paths of '{aid_a}' "
                        f"({sa[0]:.2f},{sa[1]:.2f})→({ea[0]:.2f},{ea[1]:.2f}) and "
                        f"'{aid_b}' ({sb[0]:.2f},{sb[1]:.2f})→({eb[0]:.2f},{eb[1]:.2f}) "
                        f"overlap within the {CHARACTER_XY_PADDING:.2f}m breathing-room corridor."
                    )
                elif info_a["moved"]:
                    detail = (
                        f"Movement path of '{aid_a}' "
                        f"({sa[0]:.2f},{sa[1]:.2f})→({ea[0]:.2f},{ea[1]:.2f}) sweeps through "
                        f"the padded footprint of stationary '{aid_b}' at ({sb[0]:.2f},{sb[1]:.2f})."
                    )
                elif info_b["moved"]:
                    detail = (
                        f"Movement path of '{aid_b}' "
                        f"({sb[0]:.2f},{sb[1]:.2f})→({eb[0]:.2f},{eb[1]:.2f}) sweeps through "
                        f"the padded footprint of stationary '{aid_a}' at ({sa[0]:.2f},{sa[1]:.2f})."
                    )
                else:
                    detail = (
                        f"Stationary characters '{aid_a}' ({sa[0]:.2f},{sa[1]:.2f}) and "
                        f"'{aid_b}' ({sb[0]:.2f},{sb[1]:.2f}) are within the "
                        f"{CHARACTER_XY_PADDING:.2f}m breathing-room corridor of each other."
                    )
                errors.append({
                    "shot_id": shot_id,
                    "asset_id": f"{aid_a} <-> {aid_b}",
                    "error_type": "character_path_collision",
                    "detail": detail,
                    "fix": (
                        f"Adjust start_coords/end_coords of '{aid_a}' or '{aid_b}' so that "
                        f"their padded bounding boxes (each character's footprint expanded by "
                        f"{CHARACTER_XY_PADDING:.2f}m on every horizontal side) do not overlap "
                        f"at any point along their movement. Keep their centres at least "
                        f"{min_clearance:.2f}m apart throughout the animation, or route one "
                        f"character around the other."
                    ),
                })

    return errors


def _format_plan_errors_for_llm(errors: List[Dict[str, Any]]) -> str:
    """Format verification errors into a correction prompt."""
    if not errors:
        return ""
    lines = [
        "# Animation Plan Verification Errors",
        "",
        "Your previous plan has geometric errors. Fix ALL of them in your "
        "next response. Re-output the **full corrected JSON** only.",
        "",
        "**Reminder (character breathing room):** every character's footprint "
        "is treated as padded by 0.5 m on each horizontal side for animation. "
        "In every shot, the axis-aligned box that a character sweeps from its "
        "effective start position to its end_coords must not overlap the "
        "swept box of any other character in that shot. For "
        "`character_path_collision` errors, move the start/end coords of one "
        "of the characters, or route it around the other, so their swept "
        "boxes no longer overlap.",
        "",
    ]
    for e in errors:
        lines.append(
            f"- **Shot {e['shot_id']} / {e['asset_id']}** "
            f"[{e['error_type']}]: {e['detail']}"
        )
        if e.get("fix"):
            lines.append(f"  **FIX**: {e['fix']}")
    lines.append("")
    lines.append("Output only the corrected JSON.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core generation logic (with verify → correct loop)
# ---------------------------------------------------------------------------

def _generate_single_scene_plan(
    anyllm_api_key: str,
    anyllm_api_base: Optional[str],
    anyllm_provider: str,
    reasoning_model: str,
    scene_outline: dict,
    scene_detail: dict,
    scene_shots: List[dict],
    characters: List[dict],
    layout_coords: Dict[str, dict],
    max_retries: int = 3,
    max_improvement_turns: int = 3,
) -> Optional[dict]:
    """Call the LLM to generate an animation plan for one scene,
    then run a verify → correct loop to fix geometric errors.

    Returns the parsed dict or None on failure.
    """
    scene_id = scene_outline["scene_id"]
    user_prompt = _build_user_prompt(
        scene_outline, scene_detail, scene_shots, characters, layout_coords
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]

    character_ids = {ch["asset_id"] for ch in characters}
    shot_ids = {s.get("shot_id") for s in scene_shots}

    # ---- Helper: single LLM call with retry ----
    def _call_llm(label: str) -> Optional[dict]:
        for attempt in range(max_retries):
            try:
                response = completion(
                    api_key=anyllm_api_key,
                    api_base=anyllm_api_base,
                    provider=anyllm_provider,
                    model=reasoning_model,
                    messages=messages,
                    response_format=SceneAnimationPlan,
                )
                gc.collect()
                result = json.loads(response.choices[0].message.content)
            except Exception as e:
                print(
                    f"Error in animation plan for scene {scene_id} "
                    f"({label}, attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 * (2 ** attempt))
                    continue
                return None

            # Schema validation
            try:
                SceneAnimationPlan.model_validate(result)
            except Exception as e:
                print(f"Schema error scene {scene_id} ({label}, attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    continue
                return None

            # Completeness check
            result_shot_ids = {s["shot_id"] for s in result.get("shots", [])}
            if result_shot_ids != shot_ids:
                print(f"Missing shots {shot_ids - result_shot_ids} ({label}, attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    continue
                return None

            ok = True
            for sp in result.get("shots", []):
                ids = {cp["asset_id"] for cp in sp.get("character_plans", [])}
                if not character_ids.issubset(ids):
                    print(f"Missing chars {character_ids - ids} shot {sp['shot_id']} ({label}, attempt {attempt + 1})")
                    ok = False
                    break
            if not ok:
                if attempt < max_retries - 1:
                    continue
                print(f"Warning: accepting partial result for scene {scene_id}")

            return result
        return None

    # ---- Phase 1: Initial generation ----
    result = _call_llm("initial")
    if result is None:
        return None
    print(f"  Generated initial animation plan for scene {scene_id}")

    # ---- Phase 2: Verify → correct loop ----
    if max_improvement_turns <= 0:
        return result

    messages.append({
        "role": "assistant",
        "content": json.dumps(result, ensure_ascii=False),
    })

    best_result = result
    best_error_count = float("inf")

    for turn in range(1, max_improvement_turns + 1):
        errors = _verify_animation_plan(result, layout_coords, scene_id)
        error_count = len(errors)
        print(f"  Scene {scene_id} verification turn {turn}: {error_count} error(s)")

        if error_count < best_error_count:
            best_error_count = error_count
            best_result = result

        if error_count == 0:
            print(f"  Scene {scene_id}: all checks passed after {turn} turn(s)")
            return result

        correction = _format_plan_errors_for_llm(errors)
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": correction}],
        })

        corrected = _call_llm(f"correction turn {turn}")
        if corrected is None:
            print(f"  LLM correction failed at turn {turn}, returning best ({best_error_count} errors)")
            return best_result

        messages.append({
            "role": "assistant",
            "content": json.dumps(corrected, ensure_ascii=False),
        })
        result = corrected

    # Final check
    final_errors = _verify_animation_plan(result, layout_coords, scene_id)
    if len(final_errors) < best_error_count:
        best_result = result
        best_error_count = len(final_errors)

    print(f"  Scene {scene_id}: {max_improvement_turns} turns done, returning best ({best_error_count} errors)")
    return best_result


def generate_animation_plan(
    anyllm_api_key: str = None,
    anyllm_api_base: str = None,
    anyllm_provider: str = "gemini",
    reasoning_model: str = "gemini-3.1-pro-preview",
    storyboard_script: Dict[str, Any] = None,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """Generate animation plans for all characters in all shots.

    Processes scene-by-scene. Returns an updated deep copy of *storyboard_script*
    with ``text_to_motion_prompt``, ``length``, ``start_coords``, and ``end_coords``
    added to each ``character_action``.  Idle characters (present in scene but
    without an explicit action) are auto-added.

    Args:
        anyllm_api_key: API key for the LLM provider.
        anyllm_api_base: Optional base URL for the LLM API.
        anyllm_provider: LLM provider name (default ``"gemini"``).
        reasoning_model: Reasoning model name to use for animation planning.
        storyboard_script: The full layout script dict.
        max_retries: Max retry attempts per scene.

    Returns:
        Updated storyboard script dict, or ``None`` on failure.
    """
    if not isinstance(storyboard_script, dict):
        print("Error: storyboard_script must be a dict")
        return None

    storyboard_outline = storyboard_script.get("storyboard_outline", [])
    scene_details = storyboard_script.get("scene_details", [])
    shot_details = storyboard_script.get("shot_details", [])
    asset_sheet = storyboard_script.get("asset_sheet", [])

    if not scene_details:
        print("Error: No scene_details found")
        return None

    # Collect scene IDs
    scene_ids = [sd.get("scene_id") for sd in scene_details if sd.get("scene_id") is not None]
    if not scene_ids:
        print("Error: No valid scene_ids found")
        return None

    print(f"Animation Planner: processing {len(scene_ids)} scenes: {scene_ids}")

    # Deep copy for output
    result = deepcopy(storyboard_script)
    result_shot_details = result.get("shot_details", [])

    # Process each scene
    for scene_id in scene_ids:
        print(f"\n--- Animation planning for scene {scene_id} ---")

        scene_outline = _get_scene_outline(storyboard_outline, scene_id)
        scene_detail = _get_scene_detail(scene_details, scene_id)
        scene_shots = _get_shots_for_scene(shot_details, scene_id)

        if not scene_outline or not scene_detail:
            print(f"Warning: missing outline/detail for scene {scene_id}, skipping")
            continue

        characters = _get_characters_in_scene(scene_detail, asset_sheet)
        if not characters:
            print(f"No characters in scene {scene_id}, skipping")
            continue

        layout_coords = _get_layout_coords(scene_detail, asset_sheet=asset_sheet)

        plan = _generate_single_scene_plan(
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            anyllm_provider=anyllm_provider,
            reasoning_model=reasoning_model,
            scene_outline=scene_outline,
            scene_detail=scene_detail,
            scene_shots=scene_shots,
            characters=characters,
            layout_coords=layout_coords,
            max_retries=max_retries,
        )

        if plan is None:
            print(f"Failed to generate animation plan for scene {scene_id}")
            return None

        # Merge plan into result_shot_details
        _merge_plan_into_shots(
            plan, result_shot_details, scene_id, characters, layout_coords
        )

    print(f"\nAnimation planning complete for all {len(scene_ids)} scenes")
    return result


def _merge_plan_into_shots(
    plan: dict,
    result_shot_details: List[dict],
    scene_id: int,
    characters: List[dict],
    layout_coords: Dict[str, dict],
):
    """Merge the LLM-generated plan into the result shot_details."""
    character_ids = {ch["asset_id"] for ch in characters}

    # Build a lookup: shot_id -> {asset_id -> CharacterAnimationPlan}
    plan_lookup: Dict[int, Dict[str, dict]] = {}
    for shot_plan in plan.get("shots", []):
        sid = shot_plan["shot_id"]
        plan_lookup[sid] = {}
        for cp in shot_plan.get("character_plans", []):
            plan_lookup[sid][cp["asset_id"]] = cp

    for shot in result_shot_details:
        if shot.get("scene_id") != scene_id:
            continue
        shot_id = shot.get("shot_id")
        shot_plan_map = plan_lookup.get(shot_id, {})

        # Ensure character_actions list exists
        if shot.get("character_actions") is None:
            shot["character_actions"] = []

        existing_asset_ids = {a.get("asset_id") for a in shot["character_actions"]}

        # Update existing character_actions with plan fields
        for action in shot["character_actions"]:
            aid = action.get("asset_id")
            if aid in shot_plan_map:
                cp = shot_plan_map[aid]
                action["text_to_motion_prompt"] = cp["text_to_motion_prompt"]
                action["length"] = cp["length"]
                action["start_coords"] = cp.get("start_coords")
                action["end_coords"] = cp.get("end_coords")
                action["start_rotation"] = cp.get("start_rotation")

        # Add idle characters not in character_actions
        for aid in character_ids:
            if aid in existing_asset_ids:
                continue
            cp = shot_plan_map.get(aid)
            if cp:
                idle_action = {
                    "asset_id": aid,
                    "action_description": "idle (default action)",
                    "text_to_motion_prompt": cp["text_to_motion_prompt"],
                    "length": cp["length"],
                    "start_coords": cp.get("start_coords"),
                    "end_coords": cp.get("end_coords"),
                    "start_rotation": cp.get("start_rotation"),
                }
            else:
                # Fallback if LLM didn't include this character
                idle_action = {
                    "asset_id": aid,
                    "action_description": "idle (default action)",
                    "text_to_motion_prompt": "a person stands still",
                    "length": 1.0,
                    "start_coords": None,
                    "end_coords": None,
                    "start_rotation": None,
                }
            shot["character_actions"].append(idle_action)
            print(
                f"  Added idle plan for '{aid}' in scene {scene_id}, shot {shot_id}"
            )


# ---------------------------------------------------------------------------
# Main — standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = os.path.join(
        SCRIPT_DIR, "example_input", "layout_script_v4.json"
    )
    OUTPUT_PATH = os.path.join(
        SCRIPT_DIR, "example_output", "animation_plan_v1.json"
    )

    # Load API keys
    from api_keys import anyllm_api_key, anyllm_api_base

    # Load input
    with open(INPUT_PATH, "r") as f:
        storyboard_script = json.load(f)

    print("=" * 80)
    print("Animation Planner — Standalone Test")
    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 80)

    result = generate_animation_plan(
        anyllm_api_key=anyllm_api_key,
        anyllm_api_base=anyllm_api_base,
        anyllm_provider="gemini",
        reasoning_model="gemini-3.1-pro-preview",
        storyboard_script=storyboard_script,
        max_retries=3,
    )

    if result is None:
        print("\n❌ Animation planning FAILED")
        exit(1)

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Animation plan saved to {OUTPUT_PATH}")

    # Print summary
    shot_details = result.get("shot_details", [])
    total_plans = 0
    for shot in shot_details:
        actions = shot.get("character_actions", [])
        for a in actions:
            if "text_to_motion_prompt" in a:
                total_plans += 1
                rot = a.get("start_rotation")
                rot_str = f"rot_z={rot['z']:.0f}" if rot else "rot=layout"
                ec = a.get("end_coords")
                ec_str = f"→({ec['x']:.1f},{ec['y']:.1f})" if ec else ""
                print(
                    f"  Scene {shot['scene_id']} Shot {shot['shot_id']} | "
                    f"{a['asset_id']:20s} | "
                    f"len={a.get('length', '?'):<4} | "
                    f"{rot_str:12s} | "
                    f"{a['text_to_motion_prompt']} {ec_str}"
                )
    print(f"\nTotal animation plans: {total_plans}")
