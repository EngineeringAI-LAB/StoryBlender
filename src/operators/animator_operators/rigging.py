import os
import io
import time
import base64
import mimetypes
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import requests
from requests.exceptions import RequestException, ProxyError, ConnectionError
from pygltflib import GLTF2
from PIL import Image

from .restore_texture import restore_textures


class MeshyRiggingAPIError(RuntimeError):
    """Raised when the Meshy Rigging API returns an error or the task fails."""


def _read_file_as_uri(filepath: str) -> str:
    """
    Read a local file and convert it to a data URI.
    
    Args:
        filepath: Path to the local file
        
    Returns:
        Data URI as string
    """
    with open(filepath, 'rb') as file:
        file_data = file.read()
    
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    
    encoded_data = base64.b64encode(file_data).decode('utf-8')
    data_uri = f"data:{mime_type};base64,{encoded_data}"
    
    return data_uri


def _extract_textures(filepath: str, output_folder: str) -> List[str]:
    """
    Extract textures from a GLB file and save them to the output folder.
    
    Args:
        filepath: Path to the GLB model file
        output_folder: Folder to save extracted textures
        
    Returns:
        List of paths to the extracted texture files
    """
    gltf = GLTF2.load(filepath)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    binary_blob = gltf.binary_blob()
    extracted_paths = []

    if gltf.images:
        for i, image in enumerate(gltf.images):
            ext = ".png" if image.mimeType == "image/png" else ".jpg"
            image_name = image.name if image.name else f"texture_{i}"
            file_path = os.path.join(output_folder, f"{image_name}{ext}")

            if image.bufferView is not None:
                bv = gltf.bufferViews[image.bufferView]
                start = bv.byteOffset
                end = start + bv.byteLength
                image_data = binary_blob[start:end]

                img = Image.open(io.BytesIO(image_data))
                img.save(file_path)
                print(f"Extracted texture: {file_path}")
                extracted_paths.append(file_path)

    return extracted_paths


def rig_model(
    filepath: str,
    *,
    height: float = 1.7,
    texture_image_url: Optional[str] = None,
    poll_interval: float = 5.0,
    timeout: float = 20 * 60.0,
    meshy_api_key: Optional[str] = None,
    meshy_api_base: Optional[str] = "https://api.meshy.ai/openapi/v1",
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    Create a Rigging task on Meshy and block until it completes.

    This function creates a rigging task for a given 3D model (GLB format).
    Upon successful completion, it provides a rigged character in standard formats
    and optionally basic walking/running animations.

    Args:
        filepath: Path to the local GLB model file.
        height: The approximate height of the character model in meters.
            This aids in scaling and rigging accuracy. Default: 1.7.
        texture_image_url: Model's base color texture image. Publicly accessible URL
            or Data URI. Supports .png formats.
        poll_interval: Seconds to wait between polling attempts.
        timeout: Maximum time to wait in seconds before giving up.
        meshy_api_key: Meshy API key. If not provided, read from env var MESHY_API_KEY.
        meshy_api_base: Base URL for Meshy API. Default: "https://api.meshy.ai/openapi/v1"
        session: Optional requests.Session to reuse connections.

    Returns:
        The task JSON response when status is SUCCEEDED, containing:
            - id: Task ID
            - status: "SUCCEEDED"
            - result: Object with rigged_character_fbx_url, rigged_character_glb_url,
              and optionally basic_animations with walking/running animation URLs.

    Raises:
        MeshyRiggingAPIError: If API key is missing, task fails/cancels, or timeout occurs.
        requests.HTTPError: For non-2xx HTTP responses.
    """
    key = meshy_api_key or os.getenv("MESHY_API_KEY")
    if not key:
        raise MeshyRiggingAPIError(
            "Missing Meshy API key. Set MESHY_API_KEY env var or pass meshy_api_key argument."
        )

    sess = session or requests.Session()
    headers = {"Authorization": f"Bearer {key}"}

    # Extract textures from the GLB model
    model_dir = os.path.dirname(filepath)
    model_name = os.path.splitext(os.path.basename(filepath))[0]
    texture_folder = os.path.join(model_dir, f"{model_name}_texture")
    extracted_textures = _extract_textures(filepath, texture_folder)

    # If no texture_image_url provided, use the base color texture (no suffix like _normal, _metallic, _roughness)
    if texture_image_url is None and extracted_textures:
        # Filter out normal, metallic, roughness textures
        pbr_suffixes = ("_normal", "_metallic", "_roughness")
        base_textures = [t for t in extracted_textures if not any(s in os.path.basename(t).lower() for s in pbr_suffixes)]
        selected_texture = base_textures[0] if base_textures else extracted_textures[0]
        texture_image_url = _read_file_as_uri(selected_texture)
        print(f"Using extracted texture: {selected_texture}")

    # Convert local GLB file to data URI
    model_url = _read_file_as_uri(filepath)

    # Create task payload
    task_payload: Dict[str, Any] = {
        "model_url": model_url,
        "height": height,
    }

    # Add optional parameters only if provided
    if texture_image_url is not None:
        task_payload["texture_image_url"] = texture_image_url

    # Create task with retry logic
    max_retries = 10
    retry_delay = 4.0
    
    for attempt in range(max_retries):
        try:
            create_resp = sess.post(
                f"{meshy_api_base}/rigging",
                headers=headers,
                json=task_payload,
                timeout=300
            )
            create_resp.raise_for_status()
            create_data = create_resp.json()
            break
        except (RequestException, ProxyError, ConnectionError) as e:
            if attempt == max_retries - 1:
                raise MeshyRiggingAPIError(f"Failed to create rigging task after {max_retries} attempts: {e}")
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

    task_id = create_data.get("result")
    if not task_id:
        raise MeshyRiggingAPIError("Create task response missing 'result'. Response: %r" % (create_data,))

    # Poll for task completion
    # Add small initial delay to allow task to be registered in the system
    time.sleep(2.0)
    
    start_time = time.time()
    task_url = f"{meshy_api_base}/rigging/{task_id}"

    terminal_statuses = {"SUCCEEDED", "FAILED", "CANCELED"}

    while True:
        if time.time() - start_time > timeout:
            raise MeshyRiggingAPIError(
                f"Timed out waiting for Meshy rigging task {task_id} after {timeout} seconds"
            )

        # Retry logic for polling requests
        poll_success = False
        for poll_attempt in range(5):
            try:
                resp = sess.get(task_url, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                poll_success = True
                break
            except requests.HTTPError as e:
                # If we get 404, the task might not be ready yet - wait and retry
                if resp.status_code == 404:
                    time.sleep(poll_interval)
                    break
                if poll_attempt == 4:
                    raise
                print(f"Rigging poll attempt {poll_attempt + 1}/5 failed: {e}. Retrying in 4s...")
                time.sleep(4)
            except (RequestException, ProxyError, ConnectionError) as e:
                if poll_attempt == 4:
                    raise MeshyRiggingAPIError(f"Failed to poll rigging task status after 5 attempts: {e}")
                print(f"Rigging poll attempt {poll_attempt + 1}/5 failed: {e}. Retrying in 4s...")
                time.sleep(4)
        
        if not poll_success:
            time.sleep(poll_interval)
            continue

        status = data.get("status")
        if status in terminal_statuses:
            if status == "SUCCEEDED":
                return data
            # Attach detailed error if present
            task_error = (data.get("task_error") or {}).get("message")
            raise MeshyRiggingAPIError(
                f"Meshy rigging task {task_id} ended with status {status}. Error: {task_error}"
            )

        time.sleep(poll_interval)


def _download_file(url: str, output_path: str, session: Optional[requests.Session] = None) -> str:
    """
    Download a file from URL to local path.
    
    Args:
        url: URL to download from
        output_path: Local path to save the file
        session: Optional requests.Session to reuse connections
        
    Returns:
        Path to the downloaded file
    """
    sess = session or requests.Session()
    response = sess.get(url, timeout=300)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(response.content)
    
    return output_path


def _rig_single_model(
    asset_id: str,
    asset_data: Dict[str, Any],
    output_dir: str,
    meshy_api_key: str,
    meshy_api_base: str,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    Rig a single model and download the results.
    
    Args:
        asset_id: The asset ID
        asset_data: Asset data from asset_sheet
        output_dir: Directory to save downloaded files
        meshy_api_key: Meshy API key
        meshy_api_base: Meshy API base URL
        session: Optional requests.Session
        
    Returns:
        Updated asset data with rigging info
    """
    filepath = asset_data.get("main_file_path")
    height = asset_data.get("height", 1.7)
    
    print(f"Rigging model: {asset_id} ({filepath})")
    
    try:
        # Rig the model
        result = rig_model(
            filepath,
            height=height,
            meshy_api_key=meshy_api_key,
            meshy_api_base=meshy_api_base,
            session=session,
        )
        
        # Extract result URLs
        task_id = result.get("id")
        expires_at = result.get("expires_at")
        rig_result = result.get("result", {})
        rigged_glb_url = rig_result.get("rigged_character_glb_url")
        basic_animations = rig_result.get("basic_animations", {})
        running_glb_url = basic_animations.get("running_glb_url")
        
        # Get texture folder path (created during rig_model)
        model_dir = os.path.dirname(filepath)
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        texture_folder = os.path.join(model_dir, f"{model_name}_texture")
        
        # Download rigged character GLB and restore textures
        rigged_file_path = None
        if rigged_glb_url:
            raw_rigged_path = os.path.join(output_dir, f"{asset_id}_raw.glb")
            _download_file(rigged_glb_url, raw_rigged_path, session)
            print(f"Downloaded rigged model: {raw_rigged_path}")
            
            # Restore textures to the rigged model
            if os.path.exists(texture_folder):
                rigged_file_path = os.path.join(output_dir, f"{asset_id}.glb")
                restore_textures(
                    glb_path=raw_rigged_path,
                    texture_dir=texture_folder,
                    output_path=rigged_file_path,
                )
                print(f"Restored textures to rigged model: {rigged_file_path}")
            else:
                # If no texture folder, use raw file
                rigged_file_path = raw_rigged_path
                print(f"No texture folder found, using raw rigged model: {rigged_file_path}")
        
        # Download running animation GLB and restore textures
        rigged_running_file_path = None
        if running_glb_url:
            raw_running_path = os.path.join(output_dir, f"{asset_id}_running_raw.glb")
            _download_file(running_glb_url, raw_running_path, session)
            print(f"Downloaded running animation: {raw_running_path}")
            
            # Restore textures to the running animation model
            if os.path.exists(texture_folder):
                rigged_running_file_path = os.path.join(output_dir, f"{asset_id}_running.glb")
                restore_textures(
                    glb_path=raw_running_path,
                    texture_dir=texture_folder,
                    output_path=rigged_running_file_path,
                )
                print(f"Restored textures to running animation: {rigged_running_file_path}")
            else:
                # If no texture folder, use raw file
                rigged_running_file_path = raw_running_path
                print(f"No texture folder found, using raw running animation: {rigged_running_file_path}")
        
        # Update asset data
        updated_asset = {
            **asset_data,
            "rig_task_id": task_id,
            "rig_expires_at": expires_at,
            "rigged_file_path": rigged_file_path,
            "rigged_running_file_path": rigged_running_file_path,
        }
        
        print(f"✓ Successfully rigged {asset_id}")
        return updated_asset
        
    except Exception as e:
        print(f"✗ Failed to rig {asset_id}: {e}")
        return {
            **asset_data,
            "rig_error": str(e),
        }


def is_rig_task_valid(
    rig_task_id: Optional[str],
    *,
    meshy_api_key: Optional[str] = None,
    meshy_api_base: str = "https://api.meshy.ai/openapi/v1",
    session: Optional[requests.Session] = None,
) -> bool:
    """Check whether a Meshy rigging task is still valid and usable.

    A task is considered invalid if:
      - rig_task_id is empty/None
      - GET /rigging/{id} returns 404 (task not found / expired)
      - status is not SUCCEEDED
      - expires_at (if present) is in the past

    Per Meshy docs (https://docs.meshy.ai/en/api/rigging), rigging task
    results can expire; callers should re-rig in that case.
    """
    if not rig_task_id:
        return False

    key = meshy_api_key or os.getenv("MESHY_API_KEY")
    if not key:
        return False

    sess = session or requests.Session()
    headers = {"Authorization": f"Bearer {key}"}

    try:
        resp = sess.get(
            f"{meshy_api_base}/rigging/{rig_task_id}",
            headers=headers,
            timeout=30,
        )
    except (RequestException, ProxyError, ConnectionError) as e:
        # Treat network failure as "unknown" — be conservative and say invalid
        # so callers can re-rig safely. (They could also retry later.)
        print(f"Rig task validity check failed for {rig_task_id}: {e}")
        return False

    if resp.status_code == 404:
        return False
    if not resp.ok:
        return False

    try:
        data = resp.json()
    except ValueError:
        return False

    if data.get("status") != "SUCCEEDED":
        return False

    expires_at = data.get("expires_at")
    if expires_at:
        try:
            exp = int(expires_at)
            now_s = int(time.time())
            now_ms = now_s * 1000
            # Meshy uses unix-ms; tolerate seconds-encoded values too.
            expired = (exp < now_ms) if exp > 10**12 else (exp < now_s)
            if expired:
                return False
        except (TypeError, ValueError):
            pass

    # Ensure the rigged_character_glb_url is still resolvable; if Meshy keeps
    # the task record but URLs have expired, the result section will be empty.
    rig_result = data.get("result") or {}
    if not rig_result.get("rigged_character_glb_url"):
        return False

    return True


def ensure_valid_rig_task(
    asset_data: Dict[str, Any],
    output_dir: str,
    *,
    meshy_api_key: Optional[str] = None,
    meshy_api_base: str = "https://api.meshy.ai/openapi/v1",
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Ensure the given character asset has a valid Meshy rig task.

    If the asset's `rig_task_id` is missing, expired, or otherwise invalid,
    this re-runs rigging via `_rig_single_model` and returns the updated
    asset dict (with refreshed rig_task_id, rig_expires_at, rigged_file_path,
    rigged_running_file_path). Otherwise the original asset_data is returned
    unchanged.

    Raises MeshyRiggingAPIError if no API key is available when re-rigging
    is required.
    """
    rig_task_id = asset_data.get("rig_task_id")
    if is_rig_task_valid(
        rig_task_id,
        meshy_api_key=meshy_api_key,
        meshy_api_base=meshy_api_base,
        session=session,
    ):
        return asset_data

    asset_id = asset_data.get("asset_id", "<unknown>")
    if rig_task_id:
        print(f"Rig task {rig_task_id} for asset {asset_id} is missing or expired. Re-rigging...")
    else:
        print(f"No rig_task_id for asset {asset_id}. Running rigging now...")

    key = meshy_api_key or os.getenv("MESHY_API_KEY")
    if not key:
        raise MeshyRiggingAPIError(
            "Missing Meshy API key. Set MESHY_API_KEY env var or pass meshy_api_key argument."
        )

    os.makedirs(output_dir, exist_ok=True)
    return _rig_single_model(
        asset_id=asset_id,
        asset_data=asset_data,
        output_dir=output_dir,
        meshy_api_key=key,
        meshy_api_base=meshy_api_base,
        session=session,
    )


def rig_models(
    path_to_input_json: str,
    output_dir: str,
    model_id_list: Optional[List[str]] = None,
    meshy_api_key: Optional[str] = None,
    meshy_api_base: str = "https://api.meshy.ai/openapi/v1",
    max_concurrent: int = 4,
) -> Dict[str, Any]:
    """
    Rig multiple character models from an asset sheet in parallel.
    
    This function reads a JSON file containing an asset_sheet, filters for assets
    with asset_type == "character", rigs each model using Meshy API, downloads
    the rigged models and animations, and updates the asset_sheet with rigging info.
    
    Args:
        path_to_input_json: Path to the input JSON file containing asset_sheet
        output_dir: Directory to save downloaded rigged models and animations
        model_id_list: Optional list of asset IDs to rig. If None, rigs all characters.
        meshy_api_key: Meshy API key. If not provided, read from env var MESHY_API_KEY.
        meshy_api_base: Base URL for Meshy API. Default: "https://api.meshy.ai/openapi/v1"
        max_concurrent: Maximum number of concurrent rigging tasks
        
    Returns:
        Updated JSON object with the asset_sheet containing rigging information:
            - rig_task_id: The task ID from the API result
            - rig_expires_at: The expires_at timestamp from the API result
            - rigged_file_path: Path to the downloaded rigged character GLB
            - rigged_running_file_path: Path to the downloaded running animation GLB
            
    Example:
        >>> result = rig_models(
        ...     "./assets_sheet.json",
        ...     "./rigged_models",
        ...     meshy_api_key="your_api_key"
        ... )
        >>> print(result["asset_sheet"][0]["rigged_file_path"])
        "./rigged_models/character_1.glb"
    """
    key = meshy_api_key or os.getenv("MESHY_API_KEY")
    if not key:
        raise MeshyRiggingAPIError(
            "Missing Meshy API key. Set MESHY_API_KEY env var or pass meshy_api_key argument."
        )
    
    # Load input JSON
    with open(path_to_input_json, "r") as f:
        input_data = json.load(f)
    
    asset_sheet = input_data.get("asset_sheet", [])
    
    # Build a mapping of asset_id to asset data for characters only
    character_assets = {}
    for asset in asset_sheet:
        asset_id = asset.get("asset_id")
        asset_type = asset.get("asset_type")
        
        if asset_type == "character":
            # Filter by model_id_list if provided
            if model_id_list is None or asset_id in model_id_list:
                character_assets[asset_id] = asset
    
    if not character_assets:
        print("No character assets found to rig.")
        return input_data
    
    print(f"Found {len(character_assets)} character(s) to rig: {list(character_assets.keys())}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Rig models in parallel
    results = {}
    session = requests.Session()
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_asset = {
            executor.submit(
                _rig_single_model,
                asset_id,
                asset_data,
                output_dir,
                key,
                meshy_api_base,
                session,
            ): asset_id
            for asset_id, asset_data in character_assets.items()
        }
        
        for future in as_completed(future_to_asset):
            asset_id = future_to_asset[future]
            try:
                result = future.result()
                results[asset_id] = result
            except Exception as e:
                print(f"Unexpected error for {asset_id}: {e}")
                results[asset_id] = {
                    **character_assets[asset_id],
                    "rig_error": f"Unexpected error: {str(e)}",
                }
    
    # Update asset_sheet with rigging results
    updated_asset_sheet = []
    for asset in asset_sheet:
        asset_id = asset.get("asset_id")
        if asset_id in results:
            updated_asset_sheet.append(results[asset_id])
        else:
            updated_asset_sheet.append(asset)
    
    # Final report
    successful = [aid for aid, data in results.items() if not data.get("rig_error")]
    failed = [aid for aid, data in results.items() if data.get("rig_error")]
    
    print(f"\n{'='*80}")
    print(f"Rigging Results: {len(successful)}/{len(results)} models rigged successfully")
    if failed:
        print(f"Failed models ({len(failed)}):")
        for asset_id in failed:
            print(f"  - {asset_id}: {results[asset_id].get('rig_error', 'Unknown error')}")
    print(f"{'='*80}\n")
    
    # Update and return the input data
    input_data["asset_sheet"] = updated_asset_sheet
    return input_data
