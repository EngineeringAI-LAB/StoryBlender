"""Uthana AI text-to-motion animation pipeline.

Parallel to the Meshy animator pipeline. This module handles the full Uthana
workflow:

1. Check model size (<30 MB) — compress textures to JPEG if needed.
2. Upload character model + auto-rig  →  character_id.
3. Generate text-to-motion animation  →  motion_id.
4. Download animated GLB with root motion (Blender import retargets to ``end_coords``).

Works with the output of ``animation_planner.py``.
"""

import io
import os
import json
import struct
import time
import requests
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UTHANA_GRAPHQL_URL = "https://uthana.com/graphql"
UTHANA_MOTION_BASE_URL = "https://uthana.com/motion/file/motion_viewer"
MAX_UPLOAD_SIZE_MB = 30


class UthanaAPIError(RuntimeError):
    """Raised when the Uthana API returns an error or a request fails."""


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _auth(api_key: str) -> Tuple[str, str]:
    """Basic-auth tuple accepted by ``requests``."""
    return (api_key, "")


# ---------------------------------------------------------------------------
# GLB compression (no Blender / bpy required)
# ---------------------------------------------------------------------------

def _compress_glb_textures(input_path: str, output_path: str, jpeg_quality: int = 85) -> str:
    """Re-pack a GLB file, converting embedded PNG textures to JPEG.

    This mirrors the key size-reduction trick in ``compress_glb.py``
    (``export_image_format='JPEG'``) but works outside Blender by
    manipulating the GLB binary directly via *pygltflib*.

    Returns *output_path* on success, *input_path* if compression fails or
    doesn't help.
    """
    try:
        from pygltflib import GLTF2
        from PIL import Image
    except ImportError as exc:
        print(f"  Warning: compression skipped ({exc})")
        return input_path

    try:
        gltf = GLTF2.load(input_path)
    except Exception as exc:
        print(f"  Warning: failed to parse GLB for compression ({exc})")
        return input_path

    # Collect all image data replacements
    replacements = {}  # bufferView_index -> jpeg_bytes
    
    for image in gltf.images or []:
        bv_index = image.bufferView
        if bv_index is None:
            continue
        mime = (image.mimeType or "").lower()
        if "png" not in mime:
            continue  # already JPEG or unknown — leave alone

        bv = gltf.bufferViews[bv_index]
        buf = gltf.binary_blob()
        if buf is None:
            continue
        png_bytes = buf[bv.byteOffset : bv.byteOffset + bv.byteLength]

        # Convert PNG → JPEG
        try:
            img = Image.open(io.BytesIO(png_bytes))
            
            # Check if image has meaningful alpha (transparency)
            has_alpha = img.mode == "RGBA"
            if has_alpha:
                alpha = img.split()[3]
                # Check if alpha has any non-opaque pixels
                alpha_min, alpha_max = alpha.getextrema()
                if alpha_min < 255:
                    # Has transparency - keep as PNG to preserve texture
                    continue
                # Flatten alpha onto white background for JPEG
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")
            
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=jpeg_quality)
            jpeg_bytes = out.getvalue()
        except Exception as exc:
            print(f"  Warning: texture conversion failed ({exc})")
            continue

        if len(jpeg_bytes) >= len(png_bytes):
            continue  # JPEG not smaller — skip

        replacements[bv_index] = jpeg_bytes
        image.mimeType = "image/jpeg"

    if not replacements:
        return input_path

    # Rebuild the binary blob with replacements
    try:
        buf = gltf.binary_blob()
        if buf is None:
            return input_path
        
        blob = bytearray(buf)
        
        # Process replacements in reverse order of byteOffset to avoid offset shifts
        sorted_bvs = sorted(replacements.keys(), 
                           key=lambda x: gltf.bufferViews[x].byteOffset, 
                           reverse=True)
        
        for bv_index in sorted_bvs:
            jpeg_bytes = replacements[bv_index]
            bv = gltf.bufferViews[bv_index]
            png_bytes = blob[bv.byteOffset : bv.byteOffset + bv.byteLength]
            
            # Replace in place (same size or smaller with padding)
            blob[bv.byteOffset : bv.byteOffset + len(png_bytes)] = jpeg_bytes + b'\x00' * (len(png_bytes) - len(jpeg_bytes))
            bv.byteLength = len(jpeg_bytes)
        
        gltf.set_binary_blob(bytes(blob))
    except Exception as exc:
        print(f"  Warning: buffer rebuild failed ({exc})")
        return input_path

    # Save the modified GLB
    try:
        gltf.save(output_path)
        return output_path
    except Exception as exc:
        print(f"  Warning: saving compressed GLB failed ({exc})")
        return input_path


def check_and_prepare_model(
    glb_path: str,
    max_mb: float = MAX_UPLOAD_SIZE_MB,
    jpeg_quality: int = 85,
) -> str:
    """Return a path to a GLB ready for Uthana upload (<*max_mb* MB).

    1. If the file is already small enough, return it unchanged.
    2. If a ``*_compressed.glb`` sibling exists and is small enough, use it.
    3. Otherwise, attempt PNG→JPEG texture compression via *pygltflib*.
    4. If still too large, raise :class:`UthanaAPIError`.
    """
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"Model file not found: {glb_path}")

    size_mb = os.path.getsize(glb_path) / (1024 * 1024)
    if size_mb <= max_mb:
        print(f"  File size: {size_mb:.1f} MB — OK")
        return glb_path

    # Check for pre-compressed sibling
    base, ext = os.path.splitext(glb_path)
    compressed_sibling = f"{base}_compressed{ext}"
    if os.path.exists(compressed_sibling):
        cs = os.path.getsize(compressed_sibling) / (1024 * 1024)
        if cs <= max_mb:
            print(f"  Using pre-compressed file: {compressed_sibling} ({cs:.1f} MB)")
            return compressed_sibling

    # Attempt JPEG texture compression
    print(f"  File is {size_mb:.1f} MB (>{max_mb} MB), compressing textures to JPEG …")
    compressed_path = f"{base}_uthana_upload{ext}"
    result = _compress_glb_textures(glb_path, compressed_path, jpeg_quality)

    if result != glb_path and os.path.exists(result):
        cs = os.path.getsize(result) / (1024 * 1024)
        if cs <= max_mb:
            print(f"  Compressed to {cs:.1f} MB — OK")
            return result
        print(f"  Compressed to {cs:.1f} MB — still too large")

    raise UthanaAPIError(
        f"Model {glb_path} is {size_mb:.1f} MB (limit {max_mb} MB). "
        f"Automatic compression was insufficient. "
        f"Please compress with compress_glb.py in Blender first."
    )


# ---------------------------------------------------------------------------
# Uthana GraphQL API wrappers
# ---------------------------------------------------------------------------

def create_character(
    api_key: str,
    glb_path: str,
    name: str,
    auto_rig: bool = True,
    auto_rig_front_facing: bool = True,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Upload a 3D model and create a character with auto-rigging.

    Returns ``{"character_id": str, "name": str, "auto_rig_confidence": float}``.
    """
    mutation = (
        "mutation ($file: Upload!, $name: String!, "
        "$auto_rig: Boolean, $auto_rig_front_facing: Boolean) { "
        "create_character(file: $file, name: $name, "
        "auto_rig: $auto_rig, auto_rig_front_facing: $auto_rig_front_facing) { "
        "character { id name } auto_rig_confidence } }"
    )
    operations = json.dumps({
        "query": mutation,
        "variables": {
            "file": None,
            "name": name,
            "auto_rig": auto_rig,
            "auto_rig_front_facing": auto_rig_front_facing,
        },
    })
    file_map = json.dumps({"0": ["variables.file"]})

    for attempt in range(max_retries):
        try:
            with open(glb_path, "rb") as f:
                resp = requests.post(
                    UTHANA_GRAPHQL_URL,
                    auth=_auth(api_key),
                    files={
                        "operations": (None, operations, "application/json"),
                        "map": (None, file_map, "application/json"),
                        "0": (os.path.basename(glb_path), f, "application/octet-stream"),
                    },
                    timeout=300,
                )
            resp.raise_for_status()
            data = resp.json()

            if "errors" in data:
                raise UthanaAPIError(f"GraphQL errors: {data['errors']}")

            result = data["data"]["create_character"]
            char = result["character"]
            confidence = result.get("auto_rig_confidence")
            print(
                f"  ✓ Uploaded & rigged '{char['name']}' "
                f"(id={char['id']}, confidence={confidence})"
            )
            return {
                "character_id": char["id"],
                "name": char["name"],
                "auto_rig_confidence": confidence,
            }

        except UthanaAPIError:
            raise
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"  Attempt {attempt + 1}/{max_retries} failed: {exc}. Retrying in {wait}s …")
                time.sleep(wait)
            else:
                raise UthanaAPIError(
                    f"create_character failed after {max_retries} attempts: {exc}"
                )


def create_text_to_motion(
    api_key: str,
    prompt: str,
    length: float = 2.0,
    model: str = "text-to-motion-bucmd",
    foot_ik: bool = True,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Generate a motion animation from a text prompt.

    *model* can be ``"text-to-motion-bucmd"`` (default — supports explicit
    *length* control and is what the storyboard pipeline relies on for
    accurate quota usage) or ``"text-to-motion"`` (legacy fixed-duration
    model; *length* is silently ignored by the API in that mode).

    Returns ``{"motion_id": str, "name": str}``.
    """
    variables: Dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "foot_ik": foot_ik,
    }
    # The bucmd model supports length and other advanced params
    if model == "text-to-motion-bucmd":
        variables["length"] = min(max(length, 0.25), 10.0)
        variables["retargeting_ik"] = True

    query = """mutation CreateTextToMotion(
        $prompt: String!, $model: String, $foot_ik: Boolean,
        $length: Float, $retargeting_ik: Boolean
    ) {
        create_text_to_motion(
            prompt: $prompt, model: $model, foot_ik: $foot_ik,
            length: $length, retargeting_ik: $retargeting_ik
        ) { motion { id name } }
    }"""

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                UTHANA_GRAPHQL_URL,
                auth=_auth(api_key),
                json={"query": query, "variables": variables},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            if "errors" in data:
                raise UthanaAPIError(f"GraphQL errors: {data['errors']}")

            motion = data["data"]["create_text_to_motion"]["motion"]
            print(f"  ✓ Motion '{motion['name']}' (id={motion['id']})")
            return {"motion_id": motion["id"], "name": motion["name"]}

        except UthanaAPIError:
            raise
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"  Attempt {attempt + 1}/{max_retries} failed: {exc}. Retrying in {wait}s …")
                time.sleep(wait)
            else:
                raise UthanaAPIError(
                    f"create_text_to_motion failed after {max_retries} attempts: {exc}"
                )


def download_motion(
    api_key: str,
    character_id: str,
    motion_id: str,
    output_path: str,
    fps: int = 30,
    in_place: bool = False,
    max_retries: int = 3,
) -> str:
    """Download an animated GLB.

    Returns *output_path*.
    """
    filename = os.path.basename(output_path)
    url = f"{UTHANA_MOTION_BASE_URL}/{character_id}/{motion_id}/glb/{filename}"
    params: Dict[str, Any] = {"fps": fps}
    if in_place:
        params["in_place"] = "true"

    for attempt in range(max_retries):
        try:
            resp = requests.get(
                url,
                auth=_auth(api_key),
                params=params,
                timeout=300,
                allow_redirects=True,
            )
            resp.raise_for_status()

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(resp.content)

            size_mb = len(resp.content) / (1024 * 1024)
            print(f"  ✓ Downloaded {output_path} ({size_mb:.1f} MB)")
            return output_path

        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"  Download attempt {attempt + 1}/{max_retries} failed: {exc}. Retrying in {wait}s …")
                time.sleep(wait)
            else:
                raise UthanaAPIError(
                    f"download_motion failed after {max_retries} attempts: {exc}"
                )


# ---------------------------------------------------------------------------
# Single-character convenience function
# ---------------------------------------------------------------------------

def animate_single_character(
    api_key: str,
    glb_path: str,
    asset_id: str,
    text_to_motion_prompt: str,
    length: float,
    output_dir: str,
    fps: int = 30,
    motion_model: str = "text-to-motion-bucmd",
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Full Uthana pipeline for **one** character / one animation.

    Steps: check size → upload+rig → generate motion → download GLB.

    Returns a dict with ``character_id``, ``motion_id``, ``animated_path``
    on success, or ``animation_error`` on failure.
    """
    print(f"\nAnimating '{asset_id}': \"{text_to_motion_prompt}\" ({length}s)")

    try:
        upload_path = check_and_prepare_model(glb_path)

        char = create_character(api_key, upload_path, name=asset_id, max_retries=max_retries)
        character_id = char["character_id"]

        motion = create_text_to_motion(
            api_key, text_to_motion_prompt, length=length,
            model=motion_model, max_retries=max_retries,
        )
        motion_id = motion["motion_id"]

        output_filename = f"{asset_id}_{motion_id}.glb"
        output_path = os.path.join(output_dir, output_filename)

        download_motion(
            api_key, character_id, motion_id, output_path,
            fps=fps, max_retries=max_retries,
        )

        return {
            "asset_id": asset_id,
            "character_id": character_id,
            "motion_id": motion_id,
            "animated_path": output_path,
            "text_to_motion_prompt": text_to_motion_prompt,
            "length": length,
        }

    except Exception as exc:
        print(f"  ✗ Failed: {exc}")
        return {"asset_id": asset_id, "animation_error": str(exc)}


# ---------------------------------------------------------------------------
# Batch pipeline (parallel to animator.animate_rigged_model)
# ---------------------------------------------------------------------------

def uthana_animate_characters(
    path_to_input_json: str,
    output_dir: str,
    uthana_api_key: str,
    fps: int = 30,
    motion_model: str = "text-to-motion-bucmd",
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Apply Uthana text-to-motion animations to all characters.

    Reads the animation-plan JSON produced by :func:`animation_planner.generate_animation_plan`,
    uploads each unique character model once, generates motions, downloads
    animated GLBs, and returns the updated JSON with ``animated_path`` added.

    Args:
        path_to_input_json: Path to the animation-plan JSON.
        output_dir: Directory for downloaded animated GLB files.
        uthana_api_key: Uthana API key.
        fps: Frames per second for downloaded animations.
        motion_model: ``"text-to-motion"`` or ``"text-to-motion-bucmd"``.
        max_retries: Retry attempts per API call.

    Returns:
        Dict with *successful_animations*, *failed_animations*,
        *total_processed*, *updated_json*.
    """
    with open(path_to_input_json, "r") as f:
        input_data = json.load(f)

    shot_details = input_data.get("shot_details", [])
    asset_sheet = input_data.get("asset_sheet", [])

    # asset_id → model path
    model_paths: Dict[str, str] = {}
    for asset in asset_sheet:
        aid = asset.get("asset_id")
        path = asset.get("main_file_path")
        if aid and path:
            model_paths[aid] = path

    # Collect unique (asset_id, prompt, length)
    tasks: List[Tuple[str, str, float]] = []
    seen: set = set()
    for shot in shot_details:
        for action in shot.get("character_actions", []):
            prompt = action.get("text_to_motion_prompt")
            if not prompt:
                continue
            aid = action.get("asset_id", "")
            length = action.get("length", 2.0)
            key = (aid, prompt, length)
            if key not in seen:
                seen.add(key)
                tasks.append(key)

    if not tasks:
        print("No animation tasks found (missing text_to_motion_prompt).")
        return {
            "successful_animations": [],
            "failed_animations": [],
            "total_processed": 0,
        }

    print(f"Found {len(tasks)} unique animation tasks")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Phase 1: Upload unique characters ----
    unique_aids = sorted({t[0] for t in tasks})
    character_ids: Dict[str, str] = {}

    print(f"\n=== Phase 1: Upload & rig {len(unique_aids)} characters ===")
    for aid in unique_aids:
        model_path = model_paths.get(aid)
        if not model_path or not os.path.exists(model_path):
            print(f"  ⚠ {aid}: model not found ({model_path})")
            continue
        try:
            upload_path = check_and_prepare_model(model_path)
            result = create_character(
                uthana_api_key, upload_path, name=aid, max_retries=max_retries,
            )
            character_ids[aid] = result["character_id"]
        except Exception as exc:
            print(f"  ✗ {aid}: upload failed — {exc}")

    # ---- Phase 2: Generate motions + download ----
    print(f"\n=== Phase 2: Generate & download {len(tasks)} motions ===")
    successful: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    path_map: Dict[Tuple[str, str, float], str] = {}

    for aid, prompt, length in tasks:
        char_id = character_ids.get(aid)
        if not char_id:
            failed.append({
                "asset_id": aid,
                "text_to_motion_prompt": prompt,
                "animation_error": "Character upload failed or model not found",
            })
            continue

        try:
            motion = create_text_to_motion(
                uthana_api_key, prompt, length=length,
                model=motion_model, max_retries=max_retries,
            )
            mid = motion["motion_id"]

            out_name = f"{aid}_{mid}.glb"
            out_path = os.path.join(output_dir, out_name)

            download_motion(
                uthana_api_key, char_id, mid, out_path,
                fps=fps, max_retries=max_retries,
            )

            entry = {
                "asset_id": aid,
                "character_id": char_id,
                "motion_id": mid,
                "animated_path": out_path,
                "text_to_motion_prompt": prompt,
                "length": length,
            }
            successful.append(entry)
            path_map[(aid, prompt, length)] = out_path

        except Exception as exc:
            print(f"  ✗ {aid} \"{prompt}\": {exc}")
            failed.append({
                "asset_id": aid,
                "text_to_motion_prompt": prompt,
                "animation_error": str(exc),
            })

    # ---- Phase 3: Merge animated_path into JSON ----
    print(f"\n=== Phase 3: Merge results ===")
    updated_json = deepcopy(input_data)
    applied = 0
    for shot in updated_json.get("shot_details", []):
        for action in shot.get("character_actions", []):
            prompt = action.get("text_to_motion_prompt")
            if not prompt:
                continue
            key = (action.get("asset_id", ""), prompt, action.get("length", 2.0))
            if key in path_map:
                action["animated_path"] = path_map[key]
                applied += 1

    print(f"\n{'=' * 60}")
    print(
        f"Done: {len(successful)}/{len(tasks)} succeeded, "
        f"{len(failed)} failed, {applied} paths applied"
    )
    print(f"{'=' * 60}")

    return {
        "successful_animations": successful,
        "failed_animations": failed,
        "total_processed": len(tasks),
        "updated_json": updated_json,
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    from api_keys import uthana_api_key

    # Find a test model: use animation_plan output → first character with a valid path
    plan_path = os.path.join(SCRIPT_DIR, "example_output", "animation_plan_v1.json")
    fallback_path = os.path.join(SCRIPT_DIR, "example_input", "layout_script_v4.json")

    data_path = plan_path if os.path.exists(plan_path) else fallback_path
    with open(data_path, "r") as f:
        data = json.load(f)

    asset_sheet = data.get("asset_sheet", [])
    model_map = {a["asset_id"]: a.get("main_file_path") for a in asset_sheet}

    # Find first character with usable model
    test_asset = test_model = test_prompt = None
    test_length = 2.0

    for shot in data.get("shot_details", []):
        for action in shot.get("character_actions", []):
            aid = action.get("asset_id")
            mp = model_map.get(aid)
            prompt = action.get("text_to_motion_prompt", "a person walks forward")
            if not mp:
                continue
            # Prefer a file already under 30 MB (or a compressed sibling)
            candidates = [mp]
            base, ext = os.path.splitext(mp)
            candidates.append(f"{base}_compressed{ext}")
            for c in candidates:
                if os.path.exists(c):
                    sz = os.path.getsize(c) / (1024 * 1024)
                    if sz <= MAX_UPLOAD_SIZE_MB:
                        test_asset = aid
                        test_model = c
                        test_prompt = prompt
                        test_length = action.get("length", 2.0)
                        break
            if test_asset:
                break
        if test_asset:
            break

    if not test_asset:
        print("❌ No character model under 30 MB found. Available models:")
        for a in asset_sheet:
            if a.get("asset_type") == "character":
                p = a.get("main_file_path", "N/A")
                exists = os.path.exists(p) if p else False
                sz = f"{os.path.getsize(p) / (1024**2):.1f}MB" if exists else "N/A"
                print(f"  {a['asset_id']}: {p}  ({sz}, exists={exists})")
        exit(1)

    output_dir = os.path.join(SCRIPT_DIR, "example_output", "uthana_test")

    print("=" * 60)
    print("Uthana Animator — Single Animation Test")
    print(f"  Asset:  {test_asset}")
    print(f"  Model:  {test_model} ({os.path.getsize(test_model) / (1024**2):.1f} MB)")
    print(f"  Prompt: {test_prompt}")
    print(f"  Length: {test_length}s")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    result = animate_single_character(
        api_key=uthana_api_key,
        glb_path=test_model,
        asset_id=test_asset,
        text_to_motion_prompt=test_prompt,
        length=test_length,
        output_dir=output_dir,
        fps=30,
        motion_model="text-to-motion-bucmd",
    )

    print("\n" + "=" * 60)
    if result.get("animation_error"):
        print(f"❌ FAILED: {result['animation_error']}")
    else:
        print("✅ SUCCESS")
        print(f"  Character ID: {result['character_id']}")
        print(f"  Motion ID:    {result['motion_id']}")
        print(f"  Output file:  {result['animated_path']}")
        if os.path.exists(result["animated_path"]):
            sz = os.path.getsize(result["animated_path"]) / (1024 * 1024)
            print(f"  Output size:  {sz:.1f} MB")
    print("=" * 60)
