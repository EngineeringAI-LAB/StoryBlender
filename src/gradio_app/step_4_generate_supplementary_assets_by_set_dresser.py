import os
import json
import logging
import shutil
import tempfile
import gradio as gr
from ..operators.set_dresser_operators.generate_supplementary_asset_info import (
    generate_supplementary_asset_info,
)
from ..operators.concept_artist_operators.concept_artist import fetch_model
from ..operators.concept_artist_operators.asset_dimension_estimator import (
    create_supplementary_dimension_estimation_prompt,
    generate_asset_dimension_estimation,
    validate_output,
    merge_estimation,
)
from .json_editor import JSONEditorComponent
from .path_utils import make_paths_absolute, make_paths_relative

logger = logging.getLogger(__name__)


def get_cache_busted_file_path(original_path, cache_subdir="file_cache"):
    """Create a cache-busted copy of a file to force Gradio to reload it.
    
    Gradio caches files by path, so when the file content changes but
    the path stays the same, it serves stale cached content. This function
    copies the file to a temp location with a unique name based on the file's
    modification time to force a fresh load.
    
    Args:
        original_path: Path to the original file
        cache_subdir: Subdirectory name for this cache type (e.g., "image_cache", "model_cache")
        
    Returns:
        Path to the cache-busted copy, or original_path if copy fails
    """
    if not original_path or not os.path.exists(original_path):
        return original_path
    
    try:
        # Get file modification time for cache busting
        mtime = os.path.getmtime(original_path)
        mtime_str = str(int(mtime * 1000))  # millisecond precision
        
        # Create a unique temp filename
        _, ext = os.path.splitext(original_path)
        base_name = os.path.basename(original_path).replace(ext, '')
        temp_name = f"{base_name}_{mtime_str}{ext}"
        
        # Use a dedicated temp directory for cache-busted files
        cache_dir = os.path.join(tempfile.gettempdir(), f"storyblender_{cache_subdir}")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_path = os.path.join(cache_dir, temp_name)
        
        # Prune older cache-busted copies of this same source file to avoid
        # temp-dir bloat (each edit/regeneration would otherwise leave a stale copy).
        try:
            prefix = f"{base_name}_"
            for old in os.listdir(cache_dir):
                if old == temp_name:
                    continue
                if old.startswith(prefix) and old.endswith(ext):
                    try:
                        os.remove(os.path.join(cache_dir, old))
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Only copy if the cache file doesn't exist or is older
        if not os.path.exists(cache_path):
            shutil.copy2(original_path, cache_path)
        
        return cache_path
    except Exception:
        # Fall back to original path if anything fails
        return original_path


def get_latest_layout_script_path(project_dir):
    """Get the path to the latest layout_script_v{num}.json file.
    
    Args:
        project_dir: The project directory path
        
    Returns:
        str: Path to the latest layout script JSON, or None if not found
    """
    layout_dir = os.path.join(project_dir, "layout_script")
    
    if not os.path.exists(layout_dir):
        return None
    
    latest_version = 0
    latest_path = None
    
    for filename in os.listdir(layout_dir):
        if filename.startswith("layout_script_v") and filename.endswith(".json"):
            try:
                version_str = filename[len("layout_script_v"):-5]
                version = int(version_str)
                if version > latest_version:
                    latest_version = version
                    latest_path = os.path.join(layout_dir, filename)
            except ValueError:
                continue
    
    return latest_path


# ============================================================================
# Step 4.1: Design Supplementary Assets with Set Dresser
# ============================================================================

def design_supplementary_assets(
    anyllm_api_key,
    anyllm_api_base,
    reasoning_model,
    project_dir,
    editor_component
):
    """Design supplementary assets using the Set Dresser.
    
    Args:
        anyllm_api_key: API key for any-llm service
        anyllm_api_base: Base URL for any-llm API
        reasoning_model: Reasoning model to use
        project_dir: Project directory path
        editor_component: JSONEditorComponent to display results
        
    Returns:
        dict: Result with success/error and data
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path"
        }
    
    # Get latest layout script JSON path
    layout_script_path = get_latest_layout_script_path(project_dir)
    if not layout_script_path:
        return {
            "error": "⚠️ No layout script found. Please complete Step 3 first (layout_script/layout_script_v{num}.json)."
        }
    
    # Load layout script
    try:
        with open(layout_script_path, 'r') as f:
            storyboard_script = json.load(f)
        try:
            storyboard_script = make_paths_absolute(storyboard_script, project_dir)
        except Exception as e:
            logger.warning("Step 4: path conversion failed for layout script: %s", e)
    except Exception as e:
        return {
            "error": f"⚠️ Failed to load layout script: {str(e)}"
        }
    
    # Set API base to None if empty string
    api_base = anyllm_api_base if anyllm_api_base and anyllm_api_base.strip() else None
    
    # Create output directory
    output_dir = os.path.join(project_dir, "supplementary_assets")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Call generate_supplementary_asset_info
        result = generate_supplementary_asset_info(
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=api_base,
            reasoning_model=reasoning_model,
            storyboard_script=storyboard_script,
            reasoning_effort="high"
        )
        
        if not result.get("success"):
            return {
                "error": f"⚠️ Failed to generate supplementary assets: {result.get('error', 'Unknown error')}"
            }
        
        # Save the result
        output_path = os.path.join(output_dir, "supplementary_assets.json")
        try:
            save_data = make_paths_relative(result["data"], project_dir)
        except Exception as e:
            logger.warning("Step 4: path conversion failed on save: %s", e)
            save_data = result["data"]
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)
        
        return {
            "success": True,
            "data": result["data"],
            "output_path": output_path,
            "layout_script_used": layout_script_path,
        }
    except Exception as e:
        return {
            "error": f"⚠️ Failed to design supplementary assets: {str(e)}"
        }


def show_loading_and_design_supplementary_assets(
    editor_component,
    anyllm_api_key,
    anyllm_api_base,
    reasoning_model,
    project_dir
):
    """Show loading indicator and design supplementary assets."""
    # Build initial loading state
    loading_outputs = editor_component.update_with_result(None)
    loading_state = (
        gr.update(value="🔄 **Designing supplementary assets...** This may take several minutes. Please wait.", visible=True),
        gr.update(visible=False),  # Hide design button
    )
    
    yield loading_outputs + loading_state
    
    # Design supplementary assets
    result = design_supplementary_assets(
        anyllm_api_key,
        anyllm_api_base,
        reasoning_model,
        project_dir,
        editor_component
    )
    
    # Return final result
    final_outputs = editor_component.update_with_result(result)
    
    if result.get("success"):
        success_msg = "✅ **Supplementary assets designed successfully!** Review the JSON below and proceed to Step 4.2 to generate the 3D models."
    else:
        success_msg = ""
    
    final_state = (
        gr.update(value=success_msg, visible=bool(success_msg)),
        gr.update(visible=True),  # Show design button
    )
    
    yield final_outputs + final_state


def create_design_wrapper(editor_component):
    """Factory function to create a design wrapper bound to a specific editor component."""
    def design_wrapper(anyllm_api_key, anyllm_api_base, reasoning_model, project_dir):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_design_supplementary_assets(
            editor_component,
            anyllm_api_key,
            anyllm_api_base,
            reasoning_model,
            project_dir
        ):
            yield result
    return design_wrapper


# ============================================================================
# Step 4.2: Fetch Supplementary Assets with Concept Artist
# ============================================================================

def generate_supplementary_3d_assets(
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
    sketchfab_api_key,
    meshy_api_key,
    gemini_api_key,
    gemini_api_base,
    gemini_image_model,
    project_dir,
    ai_platform="Hunyuan3D",
    meshy_model="latest",
    tencent_secret_id=None,
    tencent_secret_key=None,
    trellis2_api_base=None,
    trellis2_max_concurrent=1,
    trellis2_texture_size=4096,
    trellis2_resolution="1024",
    trellis2_decimation_target=1000000,
    vision_model="gemini/gemini-2.5-flash",
    model_id_list=None,
    force_genai=False
):
    """Generate 3D supplementary assets using fetch_model.
    
    Args:
        anyllm_api_key: API key for any-llm service
        anyllm_api_base: Base URL for any-llm API
        sketchfab_api_key: Sketchfab API key
        meshy_api_key: Meshy API key
        gemini_api_key: Gemini API key
        gemini_image_model: Gemini model for image generation
        project_dir: Project directory path
        ai_platform: AI platform for 3D generation ("Hunyuan3D", "Meshy", or "TRELLIS2")
        meshy_model: Meshy AI model version
        tencent_secret_id: Tencent Cloud Secret ID for Hunyuan3D
        tencent_secret_key: Tencent Cloud Secret Key for Hunyuan3D
        vision_model: Vision model for reranking
        model_id_list: Optional list of model IDs to regenerate (None = all)
        force_genai: If True, skip polyhaven and sketchfab, use genai directly
        
    Returns:
        dict: Result with success/error and data
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path"
        }
    
    # Get supplementary assets JSON path
    supplementary_dir = os.path.join(project_dir, "supplementary_assets")
    
    # For regeneration, prefer fetched_supplementary_assets.json if it exists
    fetched_json_path = os.path.join(supplementary_dir, "fetched_supplementary_assets.json")
    supplementary_json_path = os.path.join(supplementary_dir, "supplementary_assets.json")
    
    if model_id_list is not None and os.path.exists(fetched_json_path):
        input_json_path = fetched_json_path
    elif os.path.exists(supplementary_json_path):
        input_json_path = supplementary_json_path
    else:
        return {
            "error": "⚠️ No supplementary assets found. Please complete Step 4.1 first."
        }
    
    # Set API base to None if empty string
    api_base = anyllm_api_base if anyllm_api_base and anyllm_api_base.strip() else None
    
    try:
        # Call fetch_model directly
        result = fetch_model(
            path_to_input_json=input_json_path,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=api_base,
            sketchfab_api_key=sketchfab_api_key,
            meshy_api_key=meshy_api_key,
            gemini_api_key=gemini_api_key,
            gemini_api_base=gemini_api_base,
            gemini_image_model=gemini_image_model,
            output_dir=supplementary_dir,
            ai_platform=ai_platform,
            max_concurrent=10,
            meshy_ai_model=meshy_model,
            tencent_secret_id=tencent_secret_id,
            tencent_secret_key=tencent_secret_key,
            trellis2_api_base=trellis2_api_base,
            trellis2_max_concurrent=trellis2_max_concurrent,
            trellis2_texture_size=trellis2_texture_size,
            trellis2_resolution=trellis2_resolution,
            trellis2_decimation_target=trellis2_decimation_target,
            vision_model=vision_model,
            anyllm_provider=anyllm_provider,
            model_id_list=model_id_list,
            force_genai=force_genai,
        )
        
        # If regenerating specific models, merge results into original file
        output_path = os.path.join(supplementary_dir, "fetched_supplementary_assets.json")
        
        if model_id_list is not None and os.path.exists(fetched_json_path):
            # Load original file to preserve non-regenerated assets and scene_details
            with open(fetched_json_path, "r") as f:
                original_data = json.load(f)
            try:
                original_data = make_paths_absolute(original_data, project_dir)
            except Exception as e:
                logger.warning("Step 4: path conversion failed for fetched_supplementary_assets.json: %s", e)
            
            # Build a map of regenerated asset_ids to their new data
            regenerated_assets = {
                asset.get("asset_id"): asset 
                for asset in result.get("asset_sheet", [])
            }
            
            # Update only the regenerated assets in the original asset_sheet
            updated_asset_sheet = []
            for asset in original_data.get("asset_sheet", []):
                asset_id = asset.get("asset_id")
                if asset_id in regenerated_assets:
                    # Replace with regenerated asset data
                    updated_asset_sheet.append(regenerated_assets[asset_id])
                else:
                    # Keep original asset unchanged
                    updated_asset_sheet.append(asset)
            
            # Create merged result with updated asset_sheet and original scene_details
            merged_result = {
                "asset_sheet": updated_asset_sheet,
                "scene_details": original_data.get("scene_details", []),
            }
            
            # Save the merged result
            try:
                save_data = make_paths_relative(merged_result, project_dir)
            except Exception as e:
                logger.warning("Step 4: path conversion failed on save: %s", e)
                save_data = merged_result
            with open(output_path, "w") as f:
                json.dump(save_data, f, indent=2)
            
            return {
                "success": True,
                "data": merged_result,
                "output_path": output_path,
                "input_json_used": input_json_path,
                "regenerated_assets": list(regenerated_assets.keys()),
            }
        else:
            # Full generation - save result directly
            try:
                save_data = make_paths_relative(result, project_dir)
            except Exception as e:
                logger.warning("Step 4: path conversion failed on save: %s", e)
                save_data = result
            with open(output_path, "w") as f:
                json.dump(save_data, f, indent=2)
            
            return {
                "success": True,
                "data": result,
                "output_path": output_path,
                "input_json_used": input_json_path,
            }
    except Exception as e:
        return {
            "error": f"⚠️ Failed to generate 3D assets: {str(e)}"
        }


def show_loading_and_generate_supplementary_3d(
    editor_component,
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
    sketchfab_api_key,
    meshy_api_key,
    gemini_api_key,
    gemini_api_base,
    gemini_image_model,
    project_dir,
    ai_platform="Hunyuan3D",
    meshy_model="latest",
    tencent_secret_id=None,
    tencent_secret_key=None,
    trellis2_api_base=None,
    trellis2_max_concurrent=1,
    trellis2_texture_size=4096,
    trellis2_resolution="1024",
    trellis2_decimation_target=1000000,
    vision_model="gemini/gemini-2.5-flash",
    model_id_list=None,
    force_genai=False
):
    """Show loading indicator and generate supplementary 3D assets."""
    # Build initial loading state
    loading_outputs = editor_component.update_with_result(None)
    
    if model_id_list:
        loading_msg = f"🔄 **Regenerating {len(model_id_list)} 3D model(s)...** This may take 10-20 minutes. Please wait."
    else:
        loading_msg = "🔄 **Generating supplementary 3D assets...** This may take 10-20 minutes. Please wait."
    
    loading_state = (
        gr.update(value=loading_msg, visible=True),
        gr.update(visible=False),  # Hide generate button
    )
    
    yield loading_outputs + loading_state
    
    # Generate 3D assets
    result = generate_supplementary_3d_assets(
        anyllm_api_key,
        anyllm_api_base,
        anyllm_provider,
        sketchfab_api_key,
        meshy_api_key,
        gemini_api_key,
        gemini_api_base,
        gemini_image_model,
        project_dir,
        ai_platform=ai_platform,
        meshy_model=meshy_model,
        tencent_secret_id=tencent_secret_id,
        tencent_secret_key=tencent_secret_key,
        trellis2_api_base=trellis2_api_base,
        trellis2_max_concurrent=trellis2_max_concurrent,
        trellis2_texture_size=trellis2_texture_size,
        trellis2_resolution=trellis2_resolution,
        trellis2_decimation_target=trellis2_decimation_target,
        vision_model=vision_model,
        model_id_list=model_id_list,
        force_genai=force_genai
    )
    
    # Return final result
    final_outputs = editor_component.update_with_result(result)
    
    if result.get("success"):
        # Verify which assets actually have a valid main_file_path on disk.
        failed_ids = find_failed_supplementary_3d_ids(project_dir)
        if failed_ids:
            failed_list = ", ".join(f"`{aid}`" for aid in failed_ids)
            success_msg = (
                f"⚠️ **Supplementary 3D generation finished, but {len(failed_ids)} model(s) failed** "
                f"(missing `main_file_path` on disk): {failed_list}. "
                "Click **♻️ Retry Failed 3D Models** to retry only these."
            )
        else:
            success_msg = "✅ **Supplementary 3D assets generated successfully!** Use the 3D Model Viewer below to preview the models."
    else:
        success_msg = ""
    
    final_state = (
        gr.update(value=success_msg, visible=bool(success_msg)),
        gr.update(visible=True),  # Show generate button
    )
    
    yield final_outputs + final_state


def create_generate_supplementary_wrapper(editor_component):
    """Factory function to create a generate wrapper bound to a specific editor component."""
    def generate_wrapper(
        anyllm_api_key, anyllm_api_base, anyllm_provider, sketchfab_api_key, meshy_api_key, 
        gemini_api_key, gemini_api_base, gemini_image_model, project_dir, ai_platform, 
        meshy_model, tencent_secret_id, tencent_secret_key, trellis2_api_base,
        trellis2_max_concurrent, trellis2_texture_size, trellis2_resolution,
        trellis2_decimation_target, vision_model
    ):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_generate_supplementary_3d(
            editor_component,
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
            sketchfab_api_key,
            meshy_api_key,
            gemini_api_key,
            gemini_api_base,
            gemini_image_model,
            project_dir,
            ai_platform=ai_platform,
            meshy_model=meshy_model,
            tencent_secret_id=tencent_secret_id,
            tencent_secret_key=tencent_secret_key,
            trellis2_api_base=trellis2_api_base,
            trellis2_max_concurrent=trellis2_max_concurrent,
            trellis2_texture_size=trellis2_texture_size,
            trellis2_resolution=trellis2_resolution,
            trellis2_decimation_target=trellis2_decimation_target,
            vision_model=vision_model,
            model_id_list=None,  # Generate all
            force_genai=False
        ):
            yield result
    return generate_wrapper


def find_failed_supplementary_3d_ids(project_dir):
    """Return list of asset_ids in
    ``supplementary_assets/fetched_supplementary_assets.json`` whose
    ``main_file_path`` is missing or does not exist on disk.
    """
    if not project_dir or not os.path.isabs(project_dir):
        return []
    json_path = os.path.join(
        project_dir, "supplementary_assets", "fetched_supplementary_assets.json"
    )
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning(
                "Step 4: path conversion failed for fetched_supplementary_assets.json: %s",
                e,
            )
    except Exception as e:
        logger.warning("Step 4: failed to read fetched_supplementary_assets.json: %s", e)
        return []

    failed = []
    for asset in data.get("asset_sheet", []) or []:
        aid = asset.get("asset_id")
        if not aid:
            continue
        path = asset.get("main_file_path")
        if (not path) or (not os.path.exists(path)):
            failed.append(aid)
    return failed


def create_retry_failed_supplementary_wrapper(editor_component):
    """Factory: retry only assets in ``fetched_supplementary_assets.json``
    whose ``main_file_path`` is missing or does not exist on disk."""
    def retry_wrapper(
        anyllm_api_key, anyllm_api_base, anyllm_provider, sketchfab_api_key, meshy_api_key,
        gemini_api_key, gemini_api_base, gemini_image_model, project_dir, ai_platform,
        meshy_model, tencent_secret_id, tencent_secret_key, trellis2_api_base,
        trellis2_max_concurrent, trellis2_texture_size, trellis2_resolution,
        trellis2_decimation_target, vision_model,
    ):
        failed_ids = find_failed_supplementary_3d_ids(project_dir)
        if not failed_ids:
            msg = "✅ No failed supplementary 3D models found in `fetched_supplementary_assets.json`. Nothing to retry."
            yield editor_component.update_with_result(None) + (
                gr.update(value=msg, visible=True),
                gr.update(visible=True),
            )
            return

        print(f"Step 4: retrying {len(failed_ids)} failed supplementary 3D model(s): {failed_ids}")
        for result in show_loading_and_generate_supplementary_3d(
            editor_component,
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
            sketchfab_api_key,
            meshy_api_key,
            gemini_api_key,
            gemini_api_base,
            gemini_image_model,
            project_dir,
            ai_platform=ai_platform,
            meshy_model=meshy_model,
            tencent_secret_id=tencent_secret_id,
            tencent_secret_key=tencent_secret_key,
            trellis2_api_base=trellis2_api_base,
            trellis2_max_concurrent=trellis2_max_concurrent,
            trellis2_texture_size=trellis2_texture_size,
            trellis2_resolution=trellis2_resolution,
            trellis2_decimation_target=trellis2_decimation_target,
            vision_model=vision_model,
            model_id_list=failed_ids,
            force_genai=False,  # retry the full pipeline (polyhaven/sketchfab/genai)
        ):
            yield result
    return retry_wrapper


def create_regenerate_supplementary_wrapper(editor_component):
    """Factory function to create a regenerate wrapper bound to a specific editor component."""
    def regenerate_wrapper(
        anyllm_api_key, anyllm_api_base, anyllm_provider, sketchfab_api_key, meshy_api_key, 
        gemini_api_key, gemini_api_base, gemini_image_model, project_dir, ai_platform, 
        meshy_model, tencent_secret_id, tencent_secret_key, trellis2_api_base,
        trellis2_max_concurrent, trellis2_texture_size, trellis2_resolution,
        trellis2_decimation_target, vision_model,
        model_id_list
    ):
        """Wrapper to properly yield from the generator."""
        if not model_id_list:
            # Return early if no models selected
            yield editor_component.update_with_result({"error": "⚠️ Please select at least one model to regenerate."}) + (
                gr.update(value="⚠️ Please select at least one model to regenerate.", visible=True),
                gr.update(visible=True),
            )
            return
        
        for result in show_loading_and_generate_supplementary_3d(
            editor_component,
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
            sketchfab_api_key,
            meshy_api_key,
            gemini_api_key,
            gemini_api_base,
            gemini_image_model,
            project_dir,
            ai_platform=ai_platform,
            meshy_model=meshy_model,
            tencent_secret_id=tencent_secret_id,
            tencent_secret_key=tencent_secret_key,
            trellis2_api_base=trellis2_api_base,
            trellis2_max_concurrent=trellis2_max_concurrent,
            trellis2_texture_size=trellis2_texture_size,
            trellis2_resolution=trellis2_resolution,
            trellis2_decimation_target=trellis2_decimation_target,
            vision_model=vision_model,
            model_id_list=model_id_list,
            force_genai=True  # Force genai for regeneration
        ):
            yield result
    return regenerate_wrapper


# ============================================================================
# 3D Model Viewer Functions
# ============================================================================

def load_supplementary_models_from_json(project_dir, model_source="raw"):
    """Load supplementary model data from JSON file.
    
    Args:
        project_dir: Project directory path
        model_source: "raw" for fetched_supplementary_assets.json, "formatted" for formatted_supplementary_assets.json, "resized" for resized_supplementary_model.json
        
    Returns:
        tuple: (list of (model_id, file_path) tuples, error message or None)
    """
    if not project_dir or not os.path.isabs(project_dir):
        return [], "⚠️ Please set a valid project directory first."
    
    if model_source == "resized":
        json_path = os.path.join(project_dir, "resized_supplementary_assets", "resized_supplementary_model.json")
        source_name = "resized_supplementary_model.json"
        not_found_msg = "⚠️ No resized supplementary models found. Please run Step 5.1 first."
    elif model_source == "formatted":
        json_path = os.path.join(project_dir, "formatted_supplementary_assets", "formatted_supplementary_assets.json")
        source_name = "formatted_supplementary_assets.json"
        not_found_msg = "⚠️ No formatted supplementary models found. Please run Step 4.3 first."
    else:
        json_path = os.path.join(project_dir, "supplementary_assets", "fetched_supplementary_assets.json")
        source_name = "fetched_supplementary_assets.json"
        not_found_msg = "⚠️ No supplementary 3D models found. Please run Step 4.2 first."
    
    if not os.path.exists(json_path):
        return [], f"{not_found_msg} ({source_name} not found at {json_path})"
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning("Step 4: path conversion failed for %s: %s", json_path, e)
        
        models = []
        asset_sheet = data.get("asset_sheet", [])
        for asset in asset_sheet:
            model_id = asset.get("asset_id")
            # For formatted models, use formatted_file_path if available, else main_file_path
            if model_source == "formatted":
                file_path = asset.get("formatted_file_path") or asset.get("main_file_path")
            else:
                file_path = asset.get("main_file_path")
            
            if model_id and file_path and os.path.exists(file_path):
                # Use cache-busted path to force Gradio to reload
                cache_busted_path = get_cache_busted_file_path(file_path, "model_cache")
                models.append((model_id, cache_busted_path))
        
        if not models:
            return [], f"⚠️ No valid 3D model files found in {source_name}."
        
        return models, None
    except Exception as e:
        return [], f"⚠️ Failed to load models: {str(e)}"


def display_supplementary_3d_models(project_dir, model_source="raw"):
    """Load and display available supplementary 3D models.
    
    Args:
        project_dir: Project directory path
        model_source: "Raw Models", "Formatted Models", or "Resized Models"
    
    Returns updates for: model_viewer, model_status, model_buttons, model_viewer_container, models_state
    """
    # Convert radio button value to internal source type
    if "Resized" in model_source:
        source_type = "resized"
        source_label = "Resized"
    elif "Formatted" in model_source:
        source_type = "formatted"
        source_label = "Formatted"
    else:
        source_type = "raw"
        source_label = "Raw"
    
    models, error = load_supplementary_models_from_json(project_dir, source_type)
    
    if error:
        gr.Info(error)
        return (
            gr.update(value=None),  # model_viewer
            gr.update(value=error, visible=True),  # model_status
            gr.update(samples=[], visible=False),  # model_buttons (Dataset)
            gr.update(visible=False),  # model_viewer_container
            []  # models_state
        )
    
    # Show the first model by default
    first_model_path = models[0][1] if models else None
    model_ids = [[m[0]] for m in models]  # Format for Dataset: list of lists
    
    return (
        gr.update(value=first_model_path),  # model_viewer
        gr.update(value=f"[{source_label}] Showing: **{models[0][0]}** ({len(models)} models available)", visible=True),  # model_status
        gr.update(samples=model_ids, visible=True),  # model_buttons (Dataset)
        gr.update(visible=True),  # model_viewer_container
        models  # models_state: list of (id, path) tuples
    )


def select_supplementary_model(evt: gr.SelectData, models_state):
    """Handle model selection from the dataset buttons.
    
    Returns updates for: model_viewer, model_status
    """
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not models_state or idx >= len(models_state):
        return gr.update(), gr.update()
    
    selected_model = models_state[idx]
    model_id, model_path = selected_model
    
    return (
        gr.update(value=model_path),  # model_viewer
        gr.update(value=f"Showing: **{model_id}** ({len(models_state)} models available)")  # model_status
    )


# ============================================================================
# Step 4.4: Estimate Supplementary Assets Dimensions (after formatting)
# ============================================================================

def estimate_supplementary_dimensions(
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
    reasoning_model,
    project_dir,
    editor_component
):
    """Estimate dimensions for supplementary assets.
    
    Args:
        anyllm_api_key: API key for any-llm service
        anyllm_api_base: Base URL for any-llm API
        reasoning_model: Reasoning model to use
        project_dir: Project directory path
        editor_component: JSONEditorComponent to display results
        
    Returns:
        dict: Result with success/error and data
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path"
        }
    
    # Get latest layout script JSON path
    layout_script_path = get_latest_layout_script_path(project_dir)
    if not layout_script_path:
        return {
            "error": "⚠️ No layout script found. Please complete Step 3 first (layout_script/layout_script_v{num}.json)."
        }
    
    # Get formatted supplementary assets JSON path (now reads from formatted output)
    formatted_json_path = os.path.join(project_dir, "formatted_supplementary_assets", "formatted_supplementary_assets.json")
    if not os.path.exists(formatted_json_path):
        return {
            "error": "⚠️ No formatted supplementary assets found. Please complete Step 4.3 first."
        }
    
    # Load layout script
    try:
        with open(layout_script_path, 'r') as f:
            layout_script = json.load(f)
        try:
            layout_script = make_paths_absolute(layout_script, project_dir)
        except Exception as e:
            logger.warning("Step 4: path conversion failed for layout script: %s", e)
    except Exception as e:
        return {
            "error": f"⚠️ Failed to load layout script: {str(e)}"
        }
    
    # Load formatted supplementary assets
    try:
        with open(formatted_json_path, 'r') as f:
            supplementary_assets = json.load(f)
        try:
            supplementary_assets = make_paths_absolute(supplementary_assets, project_dir)
        except Exception as e:
            logger.warning("Step 4: path conversion failed for formatted_supplementary_assets.json: %s", e)
    except Exception as e:
        return {
            "error": f"⚠️ Failed to load formatted supplementary assets: {str(e)}"
        }
    
    # Set API base to None if empty string
    api_base = anyllm_api_base if anyllm_api_base and anyllm_api_base.strip() else None
    
    # Create dimension estimation prompt
    try:
        prompt_contents = create_supplementary_dimension_estimation_prompt(
            layout_script=layout_script,
            supplementary_assets=supplementary_assets
        )
    except Exception as e:
        return {
            "error": f"⚠️ Failed to create dimension estimation prompt: {str(e)}"
        }
    
    # Generate dimension estimation
    try:
        _provider = anyllm_provider.strip() if anyllm_provider else "gemini"
        if not _provider:
            _provider = "gemini"
        estimation_result = generate_asset_dimension_estimation(
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=api_base,
            anyllm_provider=_provider,
            reasoning_model=reasoning_model,
            contents=prompt_contents,
            concept_data=supplementary_assets,
            reasoning_effort="high",
            estimation_type="supplementary_assets"
        )
        
        if estimation_result is None:
            return {
                "error": "⚠️ Dimension estimation failed. Please check logs for details."
            }
    except Exception as e:
        return {
            "error": f"⚠️ Failed to generate dimension estimation: {str(e)}"
        }
    
    # Validate output
    if not validate_output(supplementary_assets, estimation_result):
        return {
            "error": "⚠️ Dimension estimation output validation failed. Asset IDs don't match."
        }
    
    # Merge estimation into supplementary assets
    try:
        merged_data = merge_estimation(supplementary_assets, estimation_result)
    except Exception as e:
        return {
            "error": f"⚠️ Failed to merge dimension estimation: {str(e)}"
        }
    
    # Save the result (now saves to formatted_supplementary_assets folder)
    output_dir = os.path.join(project_dir, "formatted_supplementary_assets")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dimension_estimation.json")
    
    try:
        try:
            save_data = make_paths_relative(merged_data, project_dir)
        except Exception as e:
            logger.warning("Step 4: path conversion failed on dimension estimation save: %s", e)
            save_data = merged_data
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)
    except Exception as e:
        return {
            "error": f"⚠️ Failed to save dimension estimation: {str(e)}"
        }
    
    return {
        "success": True,
        "data": merged_data,
        "output_path": output_path,
        "layout_script_used": layout_script_path,
        "formatted_assets_used": formatted_json_path,
    }


def show_loading_and_estimate_supplementary_dimensions(
    editor_component,
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
    reasoning_model,
    project_dir
):
    """Show loading indicator and estimate supplementary dimensions."""
    # Build initial loading state
    loading_outputs = editor_component.update_with_result(None)
    loading_state = (
        gr.update(value="🔄 **Estimating supplementary asset dimensions...** This may take a few minutes. Please wait.", visible=True),
        gr.update(visible=False),  # Hide estimate button
    )
    
    yield loading_outputs + loading_state
    
    # Estimate dimensions
    result = estimate_supplementary_dimensions(
        anyllm_api_key,
        anyllm_api_base,
        anyllm_provider,
        reasoning_model,
        project_dir,
        editor_component
    )
    
    # Return final result
    final_outputs = editor_component.update_with_result(result)
    
    if result.get("success"):
        success_msg = "✅ **Supplementary asset dimensions estimated successfully!**"
    else:
        success_msg = ""
    
    final_state = (
        gr.update(value=success_msg, visible=bool(success_msg)),
        gr.update(visible=True),  # Show estimate button
    )
    
    yield final_outputs + final_state


def create_estimate_supplementary_wrapper(editor_component):
    """Factory function to create an estimate wrapper bound to a specific editor component."""
    def estimate_wrapper(anyllm_api_key, anyllm_api_base, anyllm_provider, reasoning_model, project_dir):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_estimate_supplementary_dimensions(
            editor_component,
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
            reasoning_model,
            project_dir
        ):
            yield result
    return estimate_wrapper


# ============================================================================
# Step 4.3: Format Supplementary Assets (Orientation Correction)
# ============================================================================

def format_supplementary_models(
    blender_client,
    project_dir,
    editor_component,
    anyllm_api_key=None,
    vision_model="gemini/gemini-3-flash-preview",
    anyllm_api_base=None,
    anyllm_provider=None,
    model_id_list=None
):
    """Format supplementary 3D models using BlenderMCPServer (orientation correction only).
    
    Args:
        blender_client: BlenderClient instance for communicating with Blender
        project_dir: Project directory path
        editor_component: JSONEditorComponent to display results
        anyllm_api_key: API key for LLM service (for rotation correction)
        vision_model: LLM model identifier for vision tasks
        anyllm_api_base: Optional API base URL for LLM service
        model_id_list: Optional list of asset_ids to format. If None, format all.
        
    Returns:
        dict: Result with success/error and data
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path"
        }
    
    # Check if fetched_supplementary_assets.json exists (format now happens before dimension estimation)
    fetched_json_path = os.path.join(project_dir, "supplementary_assets", "fetched_supplementary_assets.json")
    if not os.path.exists(fetched_json_path):
        return {
            "error": "⚠️ fetched_supplementary_assets.json not found. Please run Step 4.2 first."
        }
    
    # Create output directory
    formatted_output_dir = os.path.join(project_dir, "formatted_supplementary_assets")
    os.makedirs(formatted_output_dir, exist_ok=True)
    
    # Ensure MCP server is running (will start it if not)
    success, message = blender_client.ensure_server_running()
    if not success:
        return {
            "error": f"⚠️ {message}"
        }
    
    # Load JSON and convert relative paths to absolute for Blender
    # (JSON on disk stores portable relative paths like "supplementary_assets/file.glb")
    temp_input_path = None
    try:
        with open(fetched_json_path, 'r') as f:
            input_data = json.load(f)
        input_data = make_paths_absolute(input_data, project_dir)
        
        temp_input_path = fetched_json_path + ".tmp"
        with open(temp_input_path, 'w') as f:
            json.dump(input_data, f, indent=2)
    except Exception as e:
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        return {
            "error": f"⚠️ Failed to prepare input for Blender: {str(e)}"
        }
    
    # Call format_assets via BlenderClient
    try:
        _provider = anyllm_provider.strip() if anyllm_provider else "gemini"
        if not _provider:
            _provider = "gemini"
        response = blender_client.format_assets(
            path_to_script=temp_input_path,
            model_output_dir=formatted_output_dir,
            anyllm_api_key=anyllm_api_key,
            vision_model=vision_model,
            anyllm_api_base=anyllm_api_base,
            anyllm_provider=_provider,
            model_id_list=model_id_list,
            output_json_filename="formatted_supplementary_assets.json"
        )
        
        if response.get("status") == "error":
            return {
                "error": f"⚠️ Blender error: {response.get('message', 'Unknown error')}"
            }
        
        result = response.get("result", {})
        
        if "error" in result:
            return {
                "error": f"⚠️ {result['error']}"
            }
        
        # Check for the output file
        output_json_path = os.path.join(formatted_output_dir, "formatted_supplementary_assets.json")
        if not os.path.exists(output_json_path):
            return {
                "error": "⚠️ Output JSON was not created. Check Blender console for errors."
            }
        
        # Load the result for display
        with open(output_json_path, 'r') as f:
            formatted_data = json.load(f)
        try:
            formatted_data = make_paths_absolute(formatted_data, project_dir)
        except Exception as e:
            logger.warning("Step 4: path conversion failed for formatted_supplementary_assets.json: %s", e)
        
        # Set editor save path and return success
        editor_component.set_save_path(formatted_output_dir)
        
        return {
            "success": True,
            "data": formatted_data,
            "output_path": output_json_path,
            "formatted_count": result.get("formatted_count", 0),
            "total_models": result.get("total_models", 0),
            "errors": result.get("errors", []),
        }
        
    except Exception as e:
        return {
            "error": f"⚠️ Failed to format supplementary models: {str(e)}"
        }
    finally:
        # Clean up temporary input file
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except OSError:
                pass


def show_loading_and_format_supplementary_models(
    editor_component,
    blender_client,
    project_dir,
    anyllm_api_key=None,
    vision_model="gemini/gemini-3-flash-preview",
    anyllm_api_base=None,
    anyllm_provider=None,
    model_id_list=None
):
    """Show loading indicator and format supplementary models."""
    # Build initial loading state
    loading_outputs = editor_component.update_with_result(None)
    loading_msg = "🔄 **Formatting supplementary models in Blender...** This may take several minutes. Please wait."
    if model_id_list:
        loading_msg = f"🔄 **Re-formatting {len(model_id_list)} supplementary model(s) in Blender...** This may take several minutes. Please wait."
    loading_state = (
        gr.update(value=loading_msg, visible=True),
        gr.update(visible=False),  # Hide format button during processing
    )
    
    yield loading_outputs + loading_state
    
    # format assets
    result = format_supplementary_models(
        blender_client,
        project_dir,
        editor_component,
        anyllm_api_key=anyllm_api_key,
        vision_model=vision_model,
        anyllm_api_base=anyllm_api_base,
        anyllm_provider=anyllm_provider,
        model_id_list=model_id_list
    )
    
    # Return final result
    final_outputs = editor_component.update_with_result(result)
    
    # Show success message with tip if formatting succeeded
    if result.get("success"):
        success_msg = "✅ **Supplementary models formatted successfully!** You can use the **3D Model Viewer** below to verify the models."
    else:
        success_msg = ""  # Error message is shown in the editor
    
    final_state = (
        gr.update(value=success_msg, visible=bool(success_msg)),  # Show success tip
        gr.update(visible=True),   # Show format button
    )
    
    yield final_outputs + final_state


def create_format_supplementary_wrapper(editor_component, blender_client):
    """Factory function to create a format wrapper bound to a specific editor component and blender client."""
    def format_wrapper(project_dir, anyllm_api_key, vision_model, anyllm_api_base, anyllm_provider, model_id_list=None):
        """Wrapper to properly yield from the generator."""
        # Convert empty list to None (format all)
        if model_id_list is not None and len(model_id_list) == 0:
            model_id_list = None
        for result in show_loading_and_format_supplementary_models(
            editor_component,
            blender_client,
            project_dir,
            anyllm_api_key=anyllm_api_key,
            vision_model=vision_model,
            anyllm_api_base=anyllm_api_base,
            anyllm_provider=anyllm_provider,
            model_id_list=model_id_list
        ):
            yield result
    return format_wrapper


def load_format_supplementary_model_choices(project_dir):
    """Load model IDs from supplementary_assets/fetched_supplementary_assets.json for format selection.
    
    Args:
        project_dir: Project directory path
        
    Returns:
        list: List of asset_id strings
    """
    if not project_dir or not os.path.isabs(project_dir):
        return []
    
    fetched_json_path = os.path.join(project_dir, "supplementary_assets", "fetched_supplementary_assets.json")
    
    if not os.path.exists(fetched_json_path):
        return []
    
    try:
        with open(fetched_json_path, 'r') as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning("Step 4: path conversion failed for fetched_supplementary_assets.json: %s", e)
        
        asset_sheet = data.get("asset_sheet", [])
        model_ids = []
        for asset in asset_sheet:
            asset_id = asset.get("asset_id")
            if asset_id:
                model_ids.append(asset_id)
        
        return model_ids
    except Exception:
        return []


# ============================================================================
# Main UI Creation
# ============================================================================

def create_supplementary_assets_ui(
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
    reasoning_model,
    sketchfab_api_key,
    meshy_api_key,
    meshy_model,
    gemini_api_key,
    gemini_api_base,
    gemini_image_model,
    project_dir,
    vision_model,
    ai_platform,
    tencent_secret_id,
    tencent_secret_key,
    trellis2_api_base,
    trellis2_max_concurrent,
    trellis2_texture_size,
    trellis2_resolution,
    trellis2_decimation_target,
    blender_client
):
    """Create the Step 4: Design and Fetch Supplementary Assets by Set Dresser and Concept Artist UI section.
    
    Args:
        anyllm_api_key: Gradio component for any-llm API key
        anyllm_api_base: Gradio component for any-llm API base URL
        reasoning_model: Gradio component for reasoning model
        sketchfab_api_key: Gradio component for Sketchfab API key
        meshy_api_key: Gradio component for Meshy API key
        meshy_model: Gradio component for Meshy model version
        gemini_api_key: Gradio component for Gemini API key
        gemini_image_model: Gradio component for Gemini image model
        project_dir: Gradio component for project directory
        vision_model: Gradio component for vision model
        ai_platform: Gradio component for AI platform selector ("Hunyuan3D", "Meshy", or "TRELLIS2")
        tencent_secret_id: Gradio component for Tencent Cloud Secret ID
        tencent_secret_key: Gradio component for Tencent Cloud Secret Key
        blender_client: BlenderClient instance for communicating with Blender
        
    Returns:
        dict with UI components that may be needed by other parts of the app
    """
    gr.Markdown("## Step 4: Design and Fetch Supplementary Assets by Set Dresser and Concept Artist")
    gr.Markdown("Design and fetch decorative, atmosphere-enhancing assets to populate scenes with background objects.")
    
    # =========================================================================
    # Step 4.1: Design Supplementary Assets with Set Dresser
    # =========================================================================
    gr.Markdown("### Step 4.1: Design Supplementary Assets with Set Dresser")
    gr.Markdown("Analyze scenes and design supplementary decorative assets that enhance the atmosphere without being plot-related.")
    
    design_btn = gr.Button("🎨 Design Supplementary Assets", variant="primary", size="lg")
    
    # Loading status indicator (hidden by default)
    design_loading_status = gr.Markdown(value="", visible=False)
    
    # Create JSON editor component for displaying design results
    design_editor = JSONEditorComponent(
        label="Supplementary Assets Design",
        visible_initially=False,
        file_basename="supplementary_assets",
        use_version_control=False,  # No version control
        json_root_keys_list=["asset_sheet"],
        title="Step 4.1"
    )
    
    # Wire up the Resume button with project_dir input
    design_editor.setup_resume_with_project_dir(project_dir, subfolder="supplementary_assets")
    
    # Create wrapper function for the design generator
    design_wrapper = create_design_wrapper(design_editor)
    
    # Design button click handler
    design_btn.click(
        fn=design_wrapper,
        inputs=[
            anyllm_api_key,
            anyllm_api_base,
            reasoning_model,
            project_dir
        ],
        outputs=design_editor.get_output_components() + [design_loading_status, design_btn],
    )
    
    # =========================================================================
    # Step 4.2: Fetch Supplementary Assets with Concept Artist
    # =========================================================================
    gr.Markdown("### Step 4.2: Fetch Supplementary Assets with Concept Artist")
    gr.Markdown("Generate 3D models for all designed supplementary assets. This directly generates 3D models without image preview.")
    
    with gr.Row():
        generate_btn = gr.Button("🧊 Generate 3D Assets", variant="primary", size="lg", scale=2)
        retry_failed_supplementary_btn = gr.Button(
            "♻️ Retry Failed 3D Models", variant="secondary", size="lg", scale=1,
        )
    
    # Loading status indicator (hidden by default)
    generate_loading_status = gr.Markdown(value="", visible=False)
    
    # Create JSON editor component for displaying generation results
    generate_editor = JSONEditorComponent(
        label="Fetched Supplementary Assets",
        visible_initially=False,
        file_basename="fetched_supplementary_assets",
        use_version_control=False,  # No version control
        json_root_keys_list=["asset_sheet"],
        title="Step 4.2"
    )
    
    # Wire up the Resume button with project_dir input
    generate_editor.setup_resume_with_project_dir(project_dir, subfolder="supplementary_assets")
    
    # Create wrapper function for the generate generator
    generate_wrapper = create_generate_supplementary_wrapper(generate_editor)
    retry_failed_supplementary_wrapper = create_retry_failed_supplementary_wrapper(generate_editor)
    
    _generate_inputs = [
        anyllm_api_key,
        anyllm_api_base,
        anyllm_provider,
        sketchfab_api_key,
        meshy_api_key,
        gemini_api_key,
        gemini_api_base,
        gemini_image_model,
        project_dir,
        ai_platform,
        meshy_model,
        tencent_secret_id,
        tencent_secret_key,
        trellis2_api_base,
        trellis2_max_concurrent,
        trellis2_texture_size,
        trellis2_resolution,
        trellis2_decimation_target,
        vision_model,
    ]

    # Generate button click handler
    generate_btn.click(
        fn=generate_wrapper,
        inputs=_generate_inputs,
        outputs=generate_editor.get_output_components() + [generate_loading_status, generate_btn],
    )
    retry_failed_supplementary_btn.click(
        fn=retry_failed_supplementary_wrapper,
        inputs=_generate_inputs,
        outputs=generate_editor.get_output_components() + [generate_loading_status, retry_failed_supplementary_btn],
    )
    
    # =========================================================================
    # 3D Model Viewer Section
    # =========================================================================
    gr.Markdown("### 3D Model Viewer")
    
    with gr.Row():
        model_source_radio = gr.Radio(
            choices=["Raw Models", "Formatted Models", "Resized Models"],
            value="Raw Models",
            label="Model Source",
            info="Choose which models to display, then click 🔍 Display 3D Models to apply",
            scale=2
        )
        display_models_btn = gr.Button("🔍 Display 3D Models", variant="secondary", size="lg", scale=1)
        toggle_viewer_btn = gr.Button("👁 Hide/Show 3D Viewer", variant="secondary", size="lg", scale=1)
    
    # Status message for model viewer
    model_status = gr.Markdown(value="", visible=False)
    
    # State to store the list of models (id, path)
    models_state = gr.State([])
    
    # State to track viewer visibility
    viewer_visible = gr.State(False)
    
    # Container for 3D model viewer (hidden initially)
    with gr.Column(visible=False) as model_viewer_container:
        # Model selection buttons using Dataset
        model_buttons = gr.Dataset(
            components=[gr.Textbox(visible=False)],
            label="Select Model",
            samples=[],
            samples_per_page=20,
            visible=False
        )
        
        # 3D Model viewer
        model_viewer = gr.Model3D(
            label="3D Model Preview",
            clear_color=(0.9, 0.9, 0.9, 1.0),
            height=500
        )
        
        # Regeneration section for 3D models
        gr.Markdown("#### Regenerate Selected Assets")
        gr.Markdown("Select assets you want to regenerate using AI generation (force_genai=True), then click 'Regenerate Selected'.")
        
        # Regeneration selection using CheckboxGroup
        model_regen_selection = gr.CheckboxGroup(
            choices=[],
            label="Select assets to regenerate",
            info="Check the assets you want to regenerate"
        )
        
        model_regen_status = gr.Markdown(value="No assets selected for regeneration.", visible=True)
        regenerate_btn = gr.Button("🔄 Regenerate Selected Assets", variant="secondary", size="lg")
    
    # Create regenerate wrapper
    regenerate_wrapper = create_regenerate_supplementary_wrapper(generate_editor)
    
    def toggle_viewer(is_visible):
        """Toggle the visibility of the 3D viewer and status."""
        new_visible = not is_visible
        return gr.update(visible=new_visible), gr.update(visible=new_visible), new_visible
    
    def display_and_show(project_dir, model_source):
        """Display 3D models and set visibility to True, also update regen choices."""
        result = display_supplementary_3d_models(project_dir, model_source)
        # result: model_viewer, model_status, model_buttons, model_viewer_container, models_state
        models = result[4]  # models_state
        # Build choices for regeneration checkbox
        choices = [m[0] for m in models] if models else []
        # Append True for visibility state and regen_selection update
        return result + (True, gr.update(choices=choices, value=[]))
    
    # Display button click handler
    display_models_btn.click(
        fn=display_and_show,
        inputs=[project_dir, model_source_radio],
        outputs=[model_viewer, model_status, model_buttons, model_viewer_container, models_state, viewer_visible, model_regen_selection]
    )
    
    # Toggle button click handler
    toggle_viewer_btn.click(
        fn=toggle_viewer,
        inputs=[viewer_visible],
        outputs=[model_viewer_container, model_status, viewer_visible]
    )
    
    # Model selection handler
    model_buttons.select(
        fn=select_supplementary_model,
        inputs=[models_state],
        outputs=[model_viewer, model_status]
    )
    
    # Model regeneration selection change handler
    def update_model_regen_status(selected):
        if not selected:
            return gr.update(value="No assets selected for regeneration.")
        return gr.update(value=f"**Selected for regeneration ({len(selected)}):** {', '.join(selected)}")
    
    model_regen_selection.change(
        fn=update_model_regen_status,
        inputs=[model_regen_selection],
        outputs=[model_regen_status]
    )
    
    # Regenerate button click handler
    regenerate_btn.click(
        fn=regenerate_wrapper,
        inputs=[
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
            sketchfab_api_key,
            meshy_api_key,
            gemini_api_key,
            gemini_api_base,
            gemini_image_model,
            project_dir,
            ai_platform,
            meshy_model,
            tencent_secret_id,
            tencent_secret_key,
            trellis2_api_base,
            trellis2_max_concurrent,
            trellis2_texture_size,
            trellis2_resolution,
            trellis2_decimation_target,
            vision_model,
            model_regen_selection
        ],
        outputs=generate_editor.get_output_components() + [generate_loading_status, generate_btn]
    )
    
    # =========================================================================
    # Step 4.3: Format Supplementary Assets (Orientation Correction)
    # =========================================================================
    gr.Markdown("### Step 4.3: Format Supplementary Assets (Orientation Correction)")
    gr.Markdown("Correct model orientation in Blender using AI vision analysis. **Requires Blender with BlenderMCPServer running.**")
    
    gr.Markdown("#### Select Models to Format")
    gr.Markdown("Select specific models to re-format (useful when some models failed or orientation is incorrect). Leave empty or click 'Select All' to format all models.")
    
    with gr.Row():
        format_model_selection = gr.CheckboxGroup(
            choices=[],
            label="Select supplementary models to format",
            info="Check the models you want to format. If none selected, all models will be formatted.",
            scale=3
        )
        with gr.Column(scale=1):
            load_format_choices_btn = gr.Button("🔄 Load Models", variant="secondary", size="sm")
            select_all_format_btn = gr.Button("☑️ Select All", variant="secondary", size="sm")
    
    format_selection_status = gr.Markdown(value="No models loaded. Click 'Load Models' to load available models.", visible=True)
    
    format_btn = gr.Button("🛠️ Format Supplementary Assets", variant="primary", size="lg")
    
    # Loading status indicator for format assets (hidden by default)
    format_loading_status = gr.Markdown(value="", visible=False)
    
    # Create JSON editor component for formatted models results
    formatted_editor = JSONEditorComponent(
        label="Formatted Supplementary Assets Result",
        visible_initially=False,
        file_basename="formatted_supplementary_assets",
        use_version_control=False,  # Save as formatted_supplementary_assets.json (overwrites)
        json_root_keys_list=["asset_sheet"],
        title="Step 4.3"
    )
    
    # Wire up the Resume button with project_dir input
    formatted_editor.setup_resume_with_project_dir(project_dir, subfolder="formatted_supplementary_assets")
    
    # Create wrapper function for format assets
    format_wrapper = create_format_supplementary_wrapper(formatted_editor, blender_client)
    
    # Handler for Load Models button
    def load_format_choices_handler(project_dir_val):
        choices = load_format_supplementary_model_choices(project_dir_val)
        if choices:
            status_msg = f"**{len(choices)} model(s) available.** Select models to re-format, or leave empty to format all."
        else:
            status_msg = "⚠️ No models found. Please complete Step 4.2 (Fetch Supplementary Assets) first."
        return gr.update(choices=choices, value=[]), gr.update(value=status_msg)
    
    load_format_choices_btn.click(
        fn=load_format_choices_handler,
        inputs=[project_dir],
        outputs=[format_model_selection, format_selection_status]
    )
    
    # Handler for Select All button
    def select_all_format_handler(project_dir_val):
        choices = load_format_supplementary_model_choices(project_dir_val)
        if choices:
            status_msg = f"**All {len(choices)} model(s) selected.**"
            return gr.update(choices=choices, value=choices), gr.update(value=status_msg)
        else:
            return gr.update(choices=[], value=[]), gr.update(value="⚠️ No models found.")
    
    select_all_format_btn.click(
        fn=select_all_format_handler,
        inputs=[project_dir],
        outputs=[format_model_selection, format_selection_status]
    )
    
    # Handler for selection change
    def update_format_selection_status(selected):
        if not selected:
            return gr.update(value="No models selected. All models will be formatted.")
        return gr.update(value=f"**Selected for formatting ({len(selected)}):** {', '.join(selected)}")
    
    format_model_selection.change(
        fn=update_format_selection_status,
        inputs=[format_model_selection],
        outputs=[format_selection_status]
    )
    
    # Format button click handler - directly runs formatting with model_id_list
    format_btn.click(
        fn=format_wrapper,
        inputs=[project_dir, anyllm_api_key, vision_model, anyllm_api_base, anyllm_provider, format_model_selection],
        outputs=formatted_editor.get_output_components() + [format_loading_status, format_btn],
    )
    
    # =========================================================================
    # Step 4.4: Estimate Supplementary Assets Dimensions
    # =========================================================================
    gr.Markdown("### Step 4.4: Estimate Supplementary Assets Dimensions")
    gr.Markdown("Estimate real-world dimensions for each supplementary 3D asset using AI analysis. Requires formatted_supplementary_assets.json from Step 4.3.")
    
    estimate_btn = gr.Button("📏 Estimate Dimensions", variant="primary", size="lg")
    
    # Loading status indicator (hidden by default)
    estimate_loading_status = gr.Markdown(value="", visible=False)
    
    # Create JSON editor component for displaying estimation results
    dimension_editor = JSONEditorComponent(
        label="Dimension Estimation Result",
        visible_initially=False,
        file_basename="dimension_estimation",
        use_version_control=False,  # No version control
        json_root_keys_list=["asset_sheet"],
        title="Step 4.4"
    )
    
    # Wire up the Resume button with project_dir input
    dimension_editor.setup_resume_with_project_dir(project_dir, subfolder="formatted_supplementary_assets")
    
    # Create wrapper function for the estimate generator
    estimate_wrapper = create_estimate_supplementary_wrapper(dimension_editor)
    
    # Estimate button click handler
    estimate_btn.click(
        fn=estimate_wrapper,
        inputs=[
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
            reasoning_model,
            project_dir
        ],
        outputs=dimension_editor.get_output_components() + [estimate_loading_status, estimate_btn],
    )
    
    return {
        "design_editor": design_editor,
        "generate_editor": generate_editor,
        "dimension_editor": dimension_editor,
        "formatted_editor": formatted_editor,
        "model_viewer": model_viewer,
    }
