"""Reflection step that audits resized 3D model dimensions and re-estimates abnormal ones.

After ``resize_assets`` has produced a ``resized_model.json`` (or
``resized_supplementary_model.json``), the dimensions of every asset are
populated (``width``, ``depth``, ``height`` in Blender X/Y/Z axes).  Because
the original single-dimension estimate may be wrong (e.g. an arrow whose
"length" is actually its height after orientation correction), this module
re-checks each asset with a vision LLM that sees the front view of the
oriented model along with the current dimensions, and asks whether any
dimension is unreasonable.  When abnormal, the LLM returns a single
corrected dimension that the caller can feed back into ``resize_assets``
to re-resize that specific model.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import json
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from ..llm_completion import completion
    from .asset_dimension_estimator import process_url_or_path
except ImportError:  # pragma: no cover - fallback for direct execution
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from llm_completion import completion  # type: ignore
    from concept_artist_operators.asset_dimension_estimator import process_url_or_path  # type: ignore


class DimensionRefinementResponse(BaseModel):
    """Schema returned by the per-asset audit LLM call.

    The fix is encoded as a single coupled (axis, value) pair so the
    model cannot accidentally output "abnormal" without choosing one.

    - ``correct=True``  -> ``abnormal_axis`` and ``corrected_value_in_meters`` MUST both be null.
    - ``correct=False`` -> BOTH ``abnormal_axis`` (one of "width"/"depth"/"height")
      AND ``corrected_value_in_meters`` (positive float, meters) MUST be set.
    """

    correct: bool = Field(
        description="True if the current resized dimensions are reasonable. False ONLY if you can confidently name one axis whose value is clearly wrong and give a corrected meter value."
    )
    abnormal_axis: Optional[Literal["width", "depth", "height"]] = Field(
        None,
        description="The single Blender axis whose current value is wrong. Required iff correct=false; must be null iff correct=true.",
    )
    corrected_value_in_meters: Optional[float] = Field(
        None,
        description="Corrected real-world extent along abnormal_axis, in meters, rounded to 2 decimals. Must be > 0. Required iff correct=false; must be null iff correct=true.",
    )


REFINE_SYSTEM_INSTRUCTION = """\
You audit the size of ONE 3D asset at a time. You see:
- description,
- current Blender axis-aligned extents in meters: width (X), depth (Y), height (Z),
- a FRONT view image: horizontal = width, vertical = height; depth is the unseen axis.

GOAL
Judge whether width/depth/height are physically plausible for the real-world object. Be CONSERVATIVE: only flag something clearly wrong (off by ~2x or more from a plausible range).

IMPORTANT
width/depth/height are AXIS-ALIGNED extents, not semantic measurements. A vertically-placed arrow's "length" appears as HEIGHT, not depth.

OUTPUT — STRICT JSON, two shapes only:

  Reasonable:
    {"correct": true,  "abnormal_axis": null, "corrected_value_in_meters": null}

  Clearly wrong:
    {"correct": false, "abnormal_axis": "<width|depth|height>", "corrected_value_in_meters": <positive number>}

RULES
- If correct=false, you MUST fill BOTH abnormal_axis AND corrected_value_in_meters (positive, 2 decimals, in meters). Never leave them null when correct=false.
- If you cannot decide which axis or what value, set correct=true (it is the safe default).
- Pick the ONE axis you are most confident about; the system uniformly rescales the model from that single dimension, preserving aspect ratio.
- Output ONLY the JSON, no prose, no code fences.

WORKED EXAMPLE
Description: "An arrow protruding from the earth at an angle."
Current: width=2.32, depth=4.25, height=0.6
Reasoning: A real arrow is ~0.7-1.0 m long. The largest extent here is depth=4.25 m, so the arrow's length is along the depth axis. 4.25 m is far too long.
Answer: {"correct": false, "abnormal_axis": "depth", "corrected_value_in_meters": 0.85}
"""


_RETRY_USER_INSTRUCTION = (
    "Your previous JSON was invalid. When correct=false you MUST set "
    "abnormal_axis to one of \"width\", \"depth\", \"height\" AND set "
    "corrected_value_in_meters to a positive number (meters, 2 decimals). "
    "If you cannot confidently choose both, return correct=true with both "
    "other fields null. Re-answer with ONLY the JSON object."
)


def _build_user_contents(asset: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the multimodal user message for one asset."""
    asset_id = asset.get("asset_id", "unknown")
    description = asset.get("description", "")
    width = asset.get("width")
    depth = asset.get("depth")
    height = asset.get("height")
    front_view_url = asset.get("front_view_url")

    text = (
        f"Asset ID: {asset_id}\n"
        f"Description: {description}\n\n"
        f"Current resized dimensions (meters, Blender axis-aligned):\n"
        f"- width  (X): {width}\n"
        f"- depth  (Y): {depth}\n"
        f"- height (Z): {height}\n\n"
        f"Below is the FRONT view of the model in its current Blender orientation "
        f"(horizontal=width, vertical=height). Decide whether these dimensions are "
        f"physically reasonable for the described object."
    )

    contents: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    if front_view_url:
        try:
            contents.append({
                "type": "image_url",
                "image_url": {"url": process_url_or_path(front_view_url)},
            })
        except Exception as exc:
            print(f"Warning: failed to load front view for {asset_id}: {exc}")
    return contents


def _normalize_refinement(
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Validate and normalize a raw LLM result into the canonical shape.

    Accepts the new schema ``{correct, abnormal_axis, corrected_value_in_meters}``
    and converts it into the downstream-friendly shape
    ``{correct, width, depth, height}`` (with exactly one positive number
    when ``correct=False``).

    Returns ``None`` for malformed payloads so the caller can retry:
      - ``correct`` is not a bool
      - ``correct=True`` but either fix field is non-null
      - ``correct=False`` but ``abnormal_axis`` is not one of width/depth/height
        OR ``corrected_value_in_meters`` is not a positive number.
    """
    if not isinstance(result, dict):
        return None
    correct = result.get("correct")
    if not isinstance(correct, bool):
        return None

    axis = result.get("abnormal_axis")
    raw_val = result.get("corrected_value_in_meters")

    if correct:
        if axis is not None or raw_val is not None:
            return None
        return {"correct": True, "width": None, "depth": None, "height": None}

    # correct == False: BOTH fix fields must be valid.
    if axis not in ("width", "depth", "height"):
        return None
    try:
        val = float(raw_val) if raw_val is not None else None
    except (TypeError, ValueError):
        return None
    if val is None or val <= 0:
        return None
    val = round(val, 2)

    return {
        "correct": False,
        "width": val if axis == "width" else None,
        "depth": val if axis == "depth" else None,
        "height": val if axis == "height" else None,
    }


def refine_single_asset_dimension(
    asset: Dict[str, Any],
    anyllm_api_key: Optional[str] = None,
    anyllm_api_base: Optional[str] = None,
    anyllm_provider: str = "gemini",
    reasoning_model: str = "gemini-3.1-pro-preview",
    reasoning_effort: str = "medium",
    max_attempts: int = 4,
) -> Dict[str, Any]:
    """Audit a single asset's resized dimensions with a vision LLM call.

    Retries up to ``max_attempts`` times on API errors and on malformed
    responses (e.g. ``correct=False`` with no dimension provided). The
    retry appends a corrective user message reminding the model of the
    output contract. Only if all attempts fail does the function fall
    back to ``correct=True`` (a safe no-op) so the surrounding batch can
    still proceed.

    Returns a dict with keys ``asset_id``, ``correct``, ``width``,
    ``depth``, ``height``, plus optional ``error`` / ``attempts`` for
    diagnostics.
    """
    asset_id = asset.get("asset_id", "unknown")
    base_contents = _build_user_contents(asset)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": REFINE_SYSTEM_INSTRUCTION},
        {"role": "user", "content": base_contents},
    ]

    last_error: Optional[str] = None
    last_raw: Optional[Dict[str, Any]] = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = completion(
                api_key=anyllm_api_key,
                api_base=anyllm_api_base,
                provider=anyllm_provider,
                model=reasoning_model,
                reasoning_effort=reasoning_effort,
                messages=messages,
                response_format=DimensionRefinementResponse,
            )
            gc.collect()
            raw_text = response.choices[0].message.content
            raw_result = json.loads(raw_text)
            # Schema-level validation (loose: all-null is allowed by Pydantic)
            DimensionRefinementResponse.model_validate(raw_result)
        except Exception as exc:
            last_error = f"attempt {attempt}: {exc}"
            print(f"Refinement call failed for {asset_id} ({last_error})")
            # Append a brief assistant ack + retry instruction so the model
            # has context that its previous attempt failed.
            messages.append({"role": "user", "content": _RETRY_USER_INSTRUCTION})
            continue

        normalized = _normalize_refinement(raw_result)
        if normalized is not None:
            normalized["asset_id"] = asset_id
            if attempt > 1:
                normalized["attempts"] = attempt
            return normalized

        # Malformed answer (e.g. correct=False with missing axis/value). Retry.
        last_raw = raw_result
        last_error = (
            f"attempt {attempt}: malformed response "
            f"correct={raw_result.get('correct')!r}, "
            f"abnormal_axis={raw_result.get('abnormal_axis')!r}, "
            f"corrected_value_in_meters={raw_result.get('corrected_value_in_meters')!r}"
        )
        print(f"Refinement malformed for {asset_id} ({last_error}); retrying...")
        # Feed the bad answer back so the model can see what it did wrong.
        messages.append({
            "role": "assistant",
            "content": json.dumps(raw_result),
        })
        messages.append({
            "role": "user",
            "content": _RETRY_USER_INSTRUCTION,
        })

    # All attempts exhausted — fall back to a safe no-op.
    print(
        f"Refinement gave up for {asset_id} after {max_attempts} attempts; "
        f"last error: {last_error}. Treating as correct."
    )
    return {
        "asset_id": asset_id,
        "correct": True,
        "width": None,
        "depth": None,
        "height": None,
        "error": last_error or "unknown",
        "attempts": max_attempts,
        "last_raw": last_raw,
    }


def refine_all_asset_dimensions_parallel(
    asset_sheet: List[Dict[str, Any]],
    anyllm_api_key: Optional[str] = None,
    anyllm_api_base: Optional[str] = None,
    anyllm_provider: str = "gemini",
    reasoning_model: str = "gemini-3.1-pro-preview",
    reasoning_effort: str = "medium",
    max_workers: int = 8,
) -> List[Dict[str, Any]]:
    """Run :func:`refine_single_asset_dimension` for every asset in parallel.

    Returns a list of refinement dicts in the same order as ``asset_sheet``.
    """
    results: List[Optional[Dict[str, Any]]] = [None] * len(asset_sheet)

    def _job(idx: int, asset: Dict[str, Any]) -> None:
        results[idx] = refine_single_asset_dimension(
            asset,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            anyllm_provider=anyllm_provider,
            reasoning_model=reasoning_model,
            reasoning_effort=reasoning_effort,
        )

    if not asset_sheet:
        return []

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(asset_sheet)))) as ex:
        futures = [ex.submit(_job, i, a) for i, a in enumerate(asset_sheet)]
        for f in as_completed(futures):
            # Surface unexpected exceptions but never abort the batch.
            try:
                f.result()
            except Exception as exc:
                print(f"Unexpected refinement worker error: {exc}")

    # Replace any None entries (from worker exceptions before assignment)
    # with safe defaults.
    for i, a in enumerate(asset_sheet):
        if results[i] is None:
            results[i] = {
                "asset_id": a.get("asset_id", "unknown"),
                "correct": True,
                "width": None,
                "depth": None,
                "height": None,
                "error": "worker failed",
            }
    return results  # type: ignore[return-value]


def collect_assets_needing_resize(
    refinements: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Return a mapping ``asset_id -> refinement`` for entries flagged incorrect."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in refinements:
        if not r.get("correct", True) and any(
            r.get(k) is not None for k in ("width", "depth", "height")
        ):
            aid = r.get("asset_id")
            if aid:
                out[aid] = r
    return out
