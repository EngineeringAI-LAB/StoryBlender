import os
import json
import logging
import shutil
import tempfile
import gradio as gr
from ..operators.concept_artist_operators.concept_artist import (
    fetch_model,
    fetch_image_prompt,
    fetch_3d_from_image_prompt,
)
from ..operators.concept_artist_operators.asset_dimension_estimator import (
    create_dimension_estimation_prompt,
    generate_asset_dimension_estimation,
    validate_output,
    merge_estimation,
)
from .json_editor import JSONEditorComponent
from .path_utils import make_paths_absolute, make_paths_relative

logger = logging.getLogger(__name__)


def get_latest_director_json_path(project_dir):
    """Get the path to the latest director_v{num}.json file.
    
    Args:
        project_dir: The project directory path
        
    Returns:
        str: Path to the latest director JSON, or None if not found
    """
    director_dir = os.path.join(project_dir, "director")
    
    if not os.path.exists(director_dir):
        return None
    
    latest_version = 0
    latest_path = None
    
    for filename in os.listdir(director_dir):
        if filename.startswith("director_v") and filename.endswith(".json"):
            try:
                version_str = filename[len("director_v"):-5]
                version = int(version_str)
                if version > latest_version:
                    latest_version = version
                    latest_path = os.path.join(director_dir, filename)
            except ValueError:
                continue
    
    return latest_path


def generate_3d_assets(
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
    sketchfab_api_key,
    meshy_api_key,
    gemini_api_key,
    gemini_image_model,
    project_dir,
    editor_component,
    ai_platform="Hunyuan3D",
    meshy_model="latest",
    tencent_secret_id=None,
    tencent_secret_key=None,
    trellis2_api_base=None,
    trellis2_max_concurrent=1,
    trellis2_texture_size=4096,
    trellis2_resolution="1024",
    trellis2_decimation_target=1000000,
    vision_model="gemini/gemini-2.5-flash"
):
    """Generate 3D assets by calling fetch_model.
    
    Args:
        anyllm_api_key: API key for any-llm service
        anyllm_api_base: Base URL for any-llm API
        sketchfab_api_key: Sketchfab API key
        meshy_api_key: Meshy API key
        gemini_api_key: Gemini API key
        gemini_image_model: Gemini model for image generation
        project_dir: Project directory path
        editor_component: JSONEditorComponent to display results
        ai_platform: AI platform for 3D generation ("Hunyuan3D", "Meshy", or "TRELLIS2")
        meshy_model: Meshy AI model version
        tencent_secret_id: Tencent Cloud Secret ID for Hunyuan3D
        tencent_secret_key: Tencent Cloud Secret Key for Hunyuan3D
        vision_model: Vision model for reranking
        
    Returns:
        dict: Result from fetch_model or error dict
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path"
        }
    
    # Get latest director JSON path
    director_json_path = get_latest_director_json_path(project_dir)
    if not director_json_path:
        return {
            "error": "⚠️ No director JSON found. Please generate a storyboard first (Step 1)."
        }
    
    # Set output directory
    output_dir = os.path.join(project_dir, "models")
    
    try:
        # Call fetch_model
        result = fetch_model(
            path_to_input_json=director_json_path,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            sketchfab_api_key=sketchfab_api_key,
            meshy_api_key=meshy_api_key,
            gemini_api_key=gemini_api_key,
            gemini_image_model=gemini_image_model,
            output_dir=output_dir,
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
        )
        
        # Save the result using the editor component
        editor_component.set_save_path(output_dir)
        output_path = editor_component.save_json_data(result)
        
        return {
            "success": True,
            "data": result,
            "output_path": output_path,
            "director_json_used": director_json_path,
        }
    except Exception as e:
        return {
            "error": f"⚠️ Failed to generate 3D assets: {str(e)}"
        }


def show_loading_and_generate_3d_assets(
    editor_component,
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
    sketchfab_api_key,
    meshy_api_key,
    gemini_api_key,
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
    vision_model="gemini/gemini-2.5-flash"
):
    """Show loading indicator and generate 3D assets."""
    # Build initial loading state
    loading_outputs = editor_component.update_with_result(None)
    loading_state = (
        gr.update(value="🔄 **Generating 3D assets...** This may take 10 to 20 minutes. Please wait. If you leave, your API credits will be wasted.", visible=True),
        gr.update(visible=False),  # Hide generate button
    )
    
    yield loading_outputs + loading_state
    
    # Generate the 3D assets
    result = generate_3d_assets(
        anyllm_api_key,
        anyllm_api_base,
        anyllm_provider,
        sketchfab_api_key,
        meshy_api_key,
        gemini_api_key,
        gemini_image_model,
        project_dir,
        editor_component,
        ai_platform=ai_platform,
        meshy_model=meshy_model,
        tencent_secret_id=tencent_secret_id,
        tencent_secret_key=tencent_secret_key,
        trellis2_api_base=trellis2_api_base,
        trellis2_max_concurrent=trellis2_max_concurrent,
        trellis2_texture_size=trellis2_texture_size,
        trellis2_resolution=trellis2_resolution,
        trellis2_decimation_target=trellis2_decimation_target,
        vision_model=vision_model
    )
    
    # Return final result
    final_outputs = editor_component.update_with_result(result)
    
    # Show success message with tip if generation succeeded
    if result.get("success"):
        success_msg = "✅ **3D assets generated successfully!** You can use the **3D Model Viewer** below to preview the models and check if you're satisfied with them."
    else:
        success_msg = ""  # Error message is shown in the editor
    
    final_state = (
        gr.update(value=success_msg, visible=bool(success_msg)),  # Show success tip
        gr.update(visible=True),   # Show generate button
    )
    
    yield final_outputs + final_state


def create_generate_wrapper(editor_component):
    """Factory function to create a generate wrapper bound to a specific editor component."""
    def generate_wrapper(
        anyllm_api_key, anyllm_api_base, anyllm_provider, sketchfab_api_key, meshy_api_key,
        gemini_api_key, gemini_image_model, project_dir, ai_platform, meshy_model,
        tencent_secret_id, tencent_secret_key, trellis2_api_base, trellis2_max_concurrent,
        trellis2_texture_size, trellis2_resolution, trellis2_decimation_target, vision_model
    ):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_generate_3d_assets(
            editor_component,
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
            sketchfab_api_key,
            meshy_api_key,
            gemini_api_key,
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
            vision_model=vision_model
        ):
            yield result
    return generate_wrapper


# ============================================================================
# Step 2.1: Generate Image Prompt for 3D Assets
# ============================================================================

def generate_image_prompts(
    gemini_api_key,
    gemini_api_base,
    gemini_image_model,
    project_dir,
    model_id_list=None,
    anyllm_api_key=None,
    anyllm_api_base=None,
    vision_model="gemini/gemini-2.5-flash-preview",
    image_gen_platform="Gemini",
    openai_api_key=None,
    openai_api_base=None,
    openai_image_model="gpt-image-2",
):
    """Generate image prompts for 3D assets.
    
    Args:
        gemini_api_key: Gemini API key
        gemini_api_base: Gemini API base URL (can be empty)
        gemini_image_model: Gemini model for image generation
        project_dir: Project directory path
        model_id_list: Optional list of model IDs to regenerate (None = all)
        anyllm_api_key: API key for any-llm service (for quality checking)
        anyllm_api_base: Base URL for any-llm API (for quality checking)
        vision_model: Vision model for quality checking
        
    Returns:
        dict: Result with success/error and data
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path"
        }
    
    # Get latest director JSON path
    director_json_path = get_latest_director_json_path(project_dir)
    if not director_json_path:
        return {
            "error": "⚠️ No director JSON found. Please generate a storyboard first (Step 1)."
        }
    
    # Load director JSON
    try:
        with open(director_json_path, 'r') as f:
            director_result = json.load(f)
        try:
            director_result = make_paths_absolute(director_result, project_dir)
        except Exception as e:
            logger.warning("Step 2: path conversion failed for director JSON: %s", e)
    except Exception as e:
        return {
            "error": f"⚠️ Failed to load director JSON: {str(e)}"
        }
    
    # If regenerating, load existing image_prompt.json and merge
    output_dir = os.path.join(project_dir, "models")
    image_prompt_json_path = os.path.join(output_dir, "image_prompt.json")
    
    if model_id_list is not None and os.path.exists(image_prompt_json_path):
        # Load existing data to preserve already generated images
        try:
            with open(image_prompt_json_path, 'r') as f:
                existing_data = json.load(f)
            try:
                existing_data = make_paths_absolute(existing_data, project_dir)
            except Exception as e:
                logger.warning("Step 2: path conversion failed for image_prompt.json: %s", e)
            # Use existing data as base (preserves image_prompt_path for non-regenerated models)
            director_result = existing_data
        except Exception:
            pass  # Use director_result if loading fails
    
    # Set API base to None if empty string
    gemini_base = gemini_api_base if gemini_api_base and gemini_api_base.strip() else None
    
    # Set OpenAI API base to None if empty string
    openai_base = openai_api_base if openai_api_base and openai_api_base.strip() else None
    
    try:
        result = fetch_image_prompt(
            director_result=director_result,
            gemini_api_key=gemini_api_key,
            gemini_image_model=gemini_image_model,
            gemini_api_base=gemini_base,
            save_path=output_dir,
            model_id_list=model_id_list,
            max_concurrent=10,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            vision_model=vision_model,
            image_gen_platform=image_gen_platform,
            openai_api_key=openai_api_key,
            openai_api_base=openai_base,
            openai_image_model=openai_image_model,
        )
        
        return {
            "success": True,
            "data": result,
            "output_path": image_prompt_json_path,
        }
    except Exception as e:
        return {
            "error": f"⚠️ Failed to generate image prompts: {str(e)}"
        }


def show_loading_and_generate_image_prompts(
    gemini_api_key,
    gemini_api_base,
    gemini_image_model,
    project_dir,
    model_id_list=None,
    anyllm_api_key=None,
    anyllm_api_base=None,
    vision_model="gemini/gemini-2.5-flash-preview",
    image_gen_platform="Gemini",
    openai_api_key=None,
    openai_api_base=None,
    openai_image_model="gpt-image-2",
):
    """Show loading indicator and generate image prompts.
    
    Returns tuple of 3 elements:
        (loading_status, generate_btn, regenerate_btn)
    """
    # Initial loading state
    loading_msg = "🔄 **Generating image prompts...** This may take a few minutes."
    if model_id_list:
        loading_msg = f"🔄 **Regenerating {len(model_id_list)} image(s)...** This may take a few minutes."
    
    yield (
        gr.update(value=loading_msg, visible=True),  # loading_status
        gr.update(visible=False),  # generate_btn
        gr.update(visible=False),  # regenerate_btn
    )
    
    # Generate image prompts
    result = generate_image_prompts(
        gemini_api_key,
        gemini_api_base,
        gemini_image_model,
        project_dir,
        model_id_list,
        anyllm_api_key,
        anyllm_api_base,
        vision_model,
        image_gen_platform,
        openai_api_key,
        openai_api_base,
        openai_image_model,
    )
    
    # Final state
    if result.get("success"):
        success_msg = "✅ **Image prompts generated successfully!** Review the images below and regenerate any that don't look good."
    else:
        success_msg = result.get("error", "")
    
    yield (
        gr.update(value=success_msg, visible=True),  # loading_status
        gr.update(visible=True),   # generate_btn
        gr.update(visible=True),   # regenerate_btn
    )


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


def load_image_prompts_from_json(project_dir):
    """Load image prompts from image_prompt.json.
    
    Args:
        project_dir: Project directory path
        
    Returns:
        tuple: (list of (model_id, image_path, has_error) tuples, error message or None)
               image_path is cache-busted to force Gradio to reload from disk
    """
    if not project_dir or not os.path.isabs(project_dir):
        return [], "⚠️ Please set a valid project directory first."
    
    json_path = os.path.join(project_dir, "models", "image_prompt.json")
    
    if not os.path.exists(json_path):
        return [], "⚠️ No image prompts found. Please generate image prompts first."
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning("Step 2: path conversion failed for image_prompt.json: %s", e)
        
        images = []
        asset_sheet = data.get("asset_sheet", [])
        for asset in asset_sheet:
            model_id = asset.get("asset_id")
            image_path = asset.get("image_prompt_path")
            has_error = bool(asset.get("image_prompt_error"))
            
            if model_id:
                # Include all models, even those without images yet
                if image_path and os.path.exists(image_path):
                    # Use cache-busted path to force Gradio to reload
                    cache_busted_path = get_cache_busted_file_path(image_path, "image_cache")
                    images.append((model_id, cache_busted_path, has_error))
                else:
                    images.append((model_id, None, True))
        
        if not images:
            return [], "⚠️ No assets found in image_prompt.json."
        
        return images, None
    except Exception as e:
        return [], f"⚠️ Failed to load image prompts: {str(e)}"


def display_image_prompts(project_dir):
    """Load and display image prompts.
    
    Returns updates for: image_viewer, image_status, image_buttons, image_viewer_container, 
                        images_state, selected_for_regen
    """
    images, error = load_image_prompts_from_json(project_dir)
    
    if error:
        return (
            gr.update(value=None),  # image_viewer
            gr.update(value=error, visible=True),  # image_status
            gr.update(samples=[], visible=False),  # image_buttons (Dataset)
            gr.update(visible=False),  # image_viewer_container
            [],  # images_state
            [],  # selected_for_regen
        )
    
    # Show the first image by default
    first_image_path = None
    first_model_id = None
    for model_id, path, _ in images:
        if path:
            first_image_path = path
            first_model_id = model_id
            break
    
    if not first_image_path and images:
        first_model_id = images[0][0]
    
    # Format for Dataset: list of lists with model_id
    model_ids = [[m[0]] for m in images]
    
    status_text = f"Showing: **{first_model_id}** ({len(images)} images available)" if first_model_id else "No images available"
    
    return (
        gr.update(value=first_image_path),  # image_viewer
        gr.update(value=status_text, visible=True),  # image_status
        gr.update(samples=model_ids, visible=True),  # image_buttons (Dataset)
        gr.update(visible=True),  # image_viewer_container
        images,  # images_state: list of (id, path, has_error) tuples
        [],  # selected_for_regen: initially empty
    )


def select_image(evt: gr.SelectData, images_state):
    """Handle image selection from the dataset buttons.
    
    Returns updates for: image_viewer, image_status
    """
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not images_state or idx >= len(images_state):
        return gr.update(), gr.update()
    
    selected_image = images_state[idx]
    model_id, image_path, has_error = selected_image
    
    status_suffix = " ⚠️ (has error)" if has_error else ""
    
    return (
        gr.update(value=image_path),  # image_viewer
        gr.update(value=f"Showing: **{model_id}**{status_suffix} ({len(images_state)} images available)")  # image_status
    )


def toggle_regen_selection(model_id, current_selection):
    """Toggle a model's selection for regeneration.
    
    Returns: updated selection list
    """
    if model_id in current_selection:
        return [m for m in current_selection if m != model_id]
    else:
        return current_selection + [model_id]


def update_regen_selection_display(selected_ids, images_state):
    """Update the display to show which images are selected for regeneration.
    
    Returns: status text showing selection
    """
    if not selected_ids:
        return gr.update(value="No images selected for regeneration. Click on image buttons to select.")
    
    return gr.update(value=f"**Selected for regeneration ({len(selected_ids)}):** {', '.join(selected_ids)}")


# ============================================================================
# Step 2.2: Generate 3D Assets from Image Prompts
# ============================================================================

def generate_3d_from_images(
    meshy_api_key,
    meshy_model,
    project_dir,
    editor_component,
    ai_platform="Hunyuan3D",
    tencent_secret_id=None,
    tencent_secret_key=None,
    trellis2_api_base=None,
    trellis2_max_concurrent=1,
    trellis2_texture_size=4096,
    trellis2_resolution="1024",
    trellis2_decimation_target=1000000,
    model_id_list=None
):
    """Generate 3D assets from image prompts.
    
    Args:
        meshy_api_key: Meshy API key
        meshy_model: Meshy model version
        project_dir: Project directory path
        editor_component: JSONEditorComponent to display results
        ai_platform: AI platform for 3D generation ("Hunyuan3D", "Meshy", or "TRELLIS2")
        tencent_secret_id: Tencent Cloud Secret ID for Hunyuan3D
        tencent_secret_key: Tencent Cloud Secret Key for Hunyuan3D
        model_id_list: Optional list of model IDs to regenerate (None = all)
        
    Returns:
        dict: Result with success/error and data
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path"
        }
    
    # Load image_prompt.json or concept_artist.json (for regeneration)
    output_dir = os.path.join(project_dir, "models")
    concept_json_path = os.path.join(output_dir, "concept_artist.json")
    image_prompt_json_path = os.path.join(output_dir, "image_prompt.json")
    
    # For regeneration, prefer concept_artist.json if it exists (to preserve existing 3D models)
    if model_id_list is not None and os.path.exists(concept_json_path):
        try:
            with open(concept_json_path, 'r') as f:
                image_prompt_data = json.load(f)
            try:
                image_prompt_data = make_paths_absolute(image_prompt_data, project_dir)
            except Exception as e:
                logger.warning("Step 2: path conversion failed for concept_artist.json: %s", e)
        except Exception as e:
            return {
                "error": f"⚠️ Failed to load concept_artist.json: {str(e)}"
            }
    elif os.path.exists(image_prompt_json_path):
        try:
            with open(image_prompt_json_path, 'r') as f:
                image_prompt_data = json.load(f)
            try:
                image_prompt_data = make_paths_absolute(image_prompt_data, project_dir)
            except Exception as e:
                logger.warning("Step 2: path conversion failed for image_prompt.json: %s", e)
        except Exception as e:
            return {
                "error": f"⚠️ Failed to load image_prompt.json: {str(e)}"
            }
    else:
        return {
            "error": "⚠️ image_prompt.json not found. Please generate image prompts first (Step 2.1)."
        }
    
    # Check if there are valid image prompts
    asset_sheet = image_prompt_data.get("asset_sheet", [])
    valid_images = [a for a in asset_sheet if a.get("image_prompt_path") and os.path.exists(a.get("image_prompt_path", ""))]
    
    if not valid_images:
        return {
            "error": "⚠️ No valid image prompts found. Please generate or regenerate image prompts first."
        }
    
    try:
        result = fetch_3d_from_image_prompt(
            image_prompt_result=image_prompt_data,
            output_dir=output_dir,
            ai_platform=ai_platform,
            meshy_api_key=meshy_api_key,
            meshy_ai_model=meshy_model,
            tencent_secret_id=tencent_secret_id,
            tencent_secret_key=tencent_secret_key,
            trellis2_api_base=trellis2_api_base,
            trellis2_max_concurrent=trellis2_max_concurrent,
            trellis2_texture_size=trellis2_texture_size,
            trellis2_resolution=trellis2_resolution,
            trellis2_decimation_target=trellis2_decimation_target,
            max_concurrent=10,
            model_id_list=model_id_list,
        )
        
        # Save the result using the editor component
        editor_component.set_save_path(output_dir)
        output_path = editor_component.save_json_data(result)
        
        return {
            "success": True,
            "data": result,
            "output_path": output_path,
        }
    except Exception as e:
        return {
            "error": f"⚠️ Failed to generate 3D assets: {str(e)}"
        }


def show_loading_and_generate_3d_from_images(
    editor_component,
    meshy_api_key,
    meshy_model,
    project_dir,
    ai_platform="Hunyuan3D",
    tencent_secret_id=None,
    tencent_secret_key=None,
    trellis2_api_base=None,
    trellis2_max_concurrent=1,
    trellis2_texture_size=4096,
    trellis2_resolution="1024",
    trellis2_decimation_target=1000000,
    model_id_list=None
):
    """Show loading indicator and generate 3D assets from images."""
    # Build initial loading state
    loading_outputs = editor_component.update_with_result(None)
    
    if model_id_list:
        loading_msg = f"🔄 **Regenerating {len(model_id_list)} 3D model(s)...** This may take 10 to 20 minutes. Please wait."
    else:
        loading_msg = "🔄 **Generating 3D assets from images...** This may take 10 to 20 minutes. Please wait."
    
    loading_state = (
        gr.update(value=loading_msg, visible=True),
        gr.update(visible=False),  # Hide generate button
    )
    
    yield loading_outputs + loading_state
    
    # Generate the 3D assets
    result = generate_3d_from_images(
        meshy_api_key,
        meshy_model,
        project_dir,
        editor_component,
        ai_platform=ai_platform,
        tencent_secret_id=tencent_secret_id,
        tencent_secret_key=tencent_secret_key,
        trellis2_api_base=trellis2_api_base,
        trellis2_max_concurrent=trellis2_max_concurrent,
        trellis2_texture_size=trellis2_texture_size,
        trellis2_resolution=trellis2_resolution,
        trellis2_decimation_target=trellis2_decimation_target,
        model_id_list=model_id_list
    )
    
    # Return final result
    final_outputs = editor_component.update_with_result(result)
    
    # Show success message with tip if generation succeeded
    if result.get("success"):
        # Verify which assets actually have a valid main_file_path on disk.
        failed_ids = find_failed_3d_model_ids(project_dir)
        if failed_ids:
            failed_list = ", ".join(f"`{aid}`" for aid in failed_ids)
            success_msg = (
                f"⚠️ **3D generation finished, but {len(failed_ids)} model(s) failed** "
                f"(missing `main_file_path` on disk): {failed_list}. "
                "Click **♻️ Retry Failed 3D Models** to retry only these."
            )
        else:
            success_msg = "✅ **3D assets generated successfully!** You can use the **3D Model Viewer** below to preview the models and check if you're satisfied with them."
    else:
        success_msg = ""  # Error message is shown in the editor
    
    final_state = (
        gr.update(value=success_msg, visible=bool(success_msg)),  # Show success tip
        gr.update(visible=True),   # Show generate button
    )
    
    yield final_outputs + final_state


def create_generate_3d_wrapper(editor_component):
    """Factory function to create a generate 3D wrapper bound to a specific editor component."""
    def generate_wrapper(
        meshy_api_key, meshy_model, project_dir, ai_platform, tencent_secret_id,
        tencent_secret_key, trellis2_api_base, trellis2_max_concurrent,
        trellis2_texture_size, trellis2_resolution, trellis2_decimation_target
    ):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_generate_3d_from_images(
            editor_component,
            meshy_api_key,
            meshy_model,
            project_dir,
            ai_platform=ai_platform,
            tencent_secret_id=tencent_secret_id,
            tencent_secret_key=tencent_secret_key,
            trellis2_api_base=trellis2_api_base,
            trellis2_max_concurrent=trellis2_max_concurrent,
            trellis2_texture_size=trellis2_texture_size,
            trellis2_resolution=trellis2_resolution,
            trellis2_decimation_target=trellis2_decimation_target,
            model_id_list=None  # Generate all
        ):
            yield result
    return generate_wrapper


def find_failed_image_prompt_ids(project_dir):
    """Return list of asset_ids in ``models/image_prompt.json`` whose
    ``image_prompt_path`` is missing/invalid or that carry ``image_prompt_error``.
    """
    if not project_dir or not os.path.isabs(project_dir):
        return []
    json_path = os.path.join(project_dir, "models", "image_prompt.json")
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning("Step 2: path conversion failed for image_prompt.json: %s", e)
    except Exception as e:
        logger.warning("Step 2: failed to read image_prompt.json: %s", e)
        return []

    failed = []
    for asset in data.get("asset_sheet", []) or []:
        aid = asset.get("asset_id")
        if not aid:
            continue
        path = asset.get("image_prompt_path")
        if (not path) or (not os.path.exists(path)) or asset.get("image_prompt_error"):
            failed.append(aid)
    return failed


def find_failed_3d_model_ids(project_dir):
    """Return list of asset_ids in ``models/concept_artist.json`` whose
    ``main_file_path`` is missing or does not exist on disk.
    """
    if not project_dir or not os.path.isabs(project_dir):
        return []
    json_path = os.path.join(project_dir, "models", "concept_artist.json")
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning("Step 2: path conversion failed for concept_artist.json: %s", e)
    except Exception as e:
        logger.warning("Step 2: failed to read concept_artist.json: %s", e)
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


def create_regenerate_3d_wrapper(editor_component):
    """Factory function to create a regenerate 3D wrapper bound to a specific editor component."""
    def regenerate_wrapper(
        meshy_api_key, meshy_model, project_dir, ai_platform, tencent_secret_id,
        tencent_secret_key, trellis2_api_base, trellis2_max_concurrent,
        trellis2_texture_size, trellis2_resolution, trellis2_decimation_target,
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
        
        for result in show_loading_and_generate_3d_from_images(
            editor_component,
            meshy_api_key,
            meshy_model,
            project_dir,
            ai_platform=ai_platform,
            tencent_secret_id=tencent_secret_id,
            tencent_secret_key=tencent_secret_key,
            trellis2_api_base=trellis2_api_base,
            trellis2_max_concurrent=trellis2_max_concurrent,
            trellis2_texture_size=trellis2_texture_size,
            trellis2_resolution=trellis2_resolution,
            trellis2_decimation_target=trellis2_decimation_target,
            model_id_list=model_id_list
        ):
            yield result
    return regenerate_wrapper


def create_retry_failed_3d_wrapper(editor_component):
    """Factory: retry only assets in ``concept_artist.json`` whose
    ``main_file_path`` is missing or does not exist on disk."""
    def retry_wrapper(meshy_api_key, meshy_model, project_dir, ai_platform,
                      tencent_secret_id, tencent_secret_key, trellis2_api_base,
                      trellis2_max_concurrent, trellis2_texture_size,
                      trellis2_resolution, trellis2_decimation_target):
        failed_ids = find_failed_3d_model_ids(project_dir)
        if not failed_ids:
            msg = "✅ No failed 3D models found in `concept_artist.json`. Nothing to retry."
            yield editor_component.update_with_result(None) + (
                gr.update(value=msg, visible=True),
                gr.update(visible=True),
            )
            return

        print(f"Step 2: retrying {len(failed_ids)} failed 3D model(s): {failed_ids}")
        for result in show_loading_and_generate_3d_from_images(
            editor_component,
            meshy_api_key,
            meshy_model,
            project_dir,
            ai_platform=ai_platform,
            tencent_secret_id=tencent_secret_id,
            tencent_secret_key=tencent_secret_key,
            trellis2_api_base=trellis2_api_base,
            trellis2_max_concurrent=trellis2_max_concurrent,
            trellis2_texture_size=trellis2_texture_size,
            trellis2_resolution=trellis2_resolution,
            trellis2_decimation_target=trellis2_decimation_target,
            model_id_list=failed_ids,
        ):
            yield result
    return retry_wrapper


def estimate_asset_dimensions(
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
    reasoning_model,
    project_dir,
    editor_component
):
    """Estimate asset dimensions using the dimension estimator.
    
    Args:
        anyllm_api_key: API key for any-llm
        anyllm_api_base: API base URL for any-llm (can be empty)
        reasoning_model: Model name for reasoning, e.g. "gemini/gemini-2.5-flash"
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
    
    # Load formatted_model.json (now dimension estimation happens after formatting)
    formatted_json_path = os.path.join(project_dir, "formatted_model", "formatted_model.json")
    if not os.path.exists(formatted_json_path):
        return {
            "error": f"⚠️ formatted_model.json not found. Please format assets first (Step 2.3)."
        }
    
    try:
        with open(formatted_json_path, 'r') as f:
            concept_data = json.load(f)
        try:
            concept_data = make_paths_absolute(concept_data, project_dir)
        except Exception as e:
            logger.warning("Step 2: path conversion failed for formatted_model.json: %s", e)
    except Exception as e:
        return {
            "error": f"⚠️ Failed to load formatted_model.json: {str(e)}"
        }
    
    # Create prompt for dimension estimation
    try:
        prompt_contents = create_dimension_estimation_prompt(concept_data)
    except Exception as e:
        return {
            "error": f"⚠️ Failed to create dimension estimation prompt: {str(e)}"
        }
    
    # Set API base to None if empty string
    api_base = anyllm_api_base if anyllm_api_base and anyllm_api_base.strip() else None
    
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
            concept_data=concept_data,
            reasoning_effort="high"
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
    if not validate_output(concept_data, estimation_result):
        return {
            "error": "⚠️ Dimension estimation output validation failed. Asset IDs don't match."
        }
    
    # Merge estimation into concept data
    try:
        merged_data = merge_estimation(concept_data, estimation_result)
    except Exception as e:
        return {
            "error": f"⚠️ Failed to merge dimension estimation: {str(e)}"
        }
    
    # Save the result using the editor component (now saves to formatted_model folder)
    output_dir = os.path.join(project_dir, "formatted_model")
    editor_component.set_save_path(output_dir)
    output_path = editor_component.save_json_data(merged_data)
    
    if output_path:
        return {
            "success": True,
            "data": merged_data,
            "output_path": output_path,
            "formatted_json_used": formatted_json_path,
        }
    else:
        return {
            "error": "⚠️ Failed to save dimension estimation result."
        }


def show_loading_and_estimate_dimensions(
    editor_component,
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
    reasoning_model,
    project_dir
):
    """Show loading indicator and estimate asset dimensions."""
    # Build initial loading state
    loading_outputs = editor_component.update_with_result(None)
    loading_state = (
        gr.update(value="🔄 **Estimating asset dimensions...** This may take a minute. Please wait.", visible=True),
        gr.update(visible=False),  # Hide estimate button
    )
    
    yield loading_outputs + loading_state
    
    # Estimate dimensions
    result = estimate_asset_dimensions(
        anyllm_api_key,
        anyllm_api_base,
        anyllm_provider,
        reasoning_model,
        project_dir,
        editor_component
    )
    
    # Return final result
    final_outputs = editor_component.update_with_result(result)
    
    # Show success message with tip if estimation succeeded
    if result.get("success"):
        success_msg = "✅ **Asset dimensions estimated successfully!** You can proceed to the next step."
    else:
        success_msg = ""  # Error message is shown in the editor
    
    final_state = (
        gr.update(value=success_msg, visible=bool(success_msg)),  # Show success tip
        gr.update(visible=True),   # Show estimate button
    )
    
    yield final_outputs + final_state


def create_estimate_wrapper(editor_component):
    """Factory function to create an estimate wrapper bound to a specific editor component."""
    def estimate_wrapper(anyllm_api_key, anyllm_api_base, anyllm_provider, reasoning_model, project_dir):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_estimate_dimensions(
            editor_component,
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
            reasoning_model,
            project_dir
        ):
            yield result
    return estimate_wrapper


def format_3d_models(
    blender_client,
    project_dir,
    editor_component,
    anyllm_api_key=None,
    vision_model="gemini/gemini-3-flash-preview",
    anyllm_api_base=None,
    anyllm_provider=None,
    model_id_list=None
):
    """Format 3D models using BlenderMCPServer.
    
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
    
    # Check if concept_artist.json exists
    concept_json_path = os.path.join(project_dir, "models", "concept_artist.json")
    if not os.path.exists(concept_json_path):
        return {
            "error": "⚠️ concept_artist.json not found. Please run Step 2.2 first."
        }
    
    # Create output directory
    formatted_output_dir = os.path.join(project_dir, "formatted_model")
    os.makedirs(formatted_output_dir, exist_ok=True)
    
    # Ensure MCP server is running (will start it if not)
    success, message = blender_client.ensure_server_running()
    if not success:
        return {
            "error": f"⚠️ {message}"
        }
    
    # Load JSON and convert relative paths to absolute for Blender
    # (JSON on disk stores portable relative paths like "models/file.glb")
    temp_input_path = None
    try:
        with open(concept_json_path, 'r') as f:
            input_data = json.load(f)
        input_data = make_paths_absolute(input_data, project_dir)
        
        temp_input_path = concept_json_path + ".tmp"
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
            model_id_list=model_id_list
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
        output_json_path = os.path.join(formatted_output_dir, "formatted_model.json")
        if not os.path.exists(output_json_path):
            return {
                "error": "⚠️ formatted_model.json was not created. Check Blender console for errors."
            }
        
        # Load the result for display
        with open(output_json_path, 'r') as f:
            formatted_data = json.load(f)
        try:
            formatted_data = make_paths_absolute(formatted_data, project_dir)
        except Exception as e:
            logger.warning("Step 2: path conversion failed for formatted_model.json: %s", e)
        
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
            "error": f"⚠️ Failed to format assets: {str(e)}"
        }
    finally:
        # Clean up temporary input file
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except OSError:
                pass


def show_loading_and_format_models(
    editor_component,
    blender_client,
    project_dir,
    anyllm_api_key=None,
    vision_model="gemini/gemini-3-flash-preview",
    anyllm_api_base=None,
    anyllm_provider=None,
    model_id_list=None
):
    """Show loading indicator and format assets."""
    # Build initial loading state
    loading_outputs = editor_component.update_with_result(None)
    loading_msg = "🔄 **Formatting models in Blender...** This may take several minutes. Please wait."
    if model_id_list:
        loading_msg = f"🔄 **Re-formatting {len(model_id_list)} model(s) in Blender...** This may take several minutes. Please wait."
    loading_state = (
        gr.update(value=loading_msg, visible=True),
        gr.update(visible=False),  # Hide format button during processing
    )
    
    yield loading_outputs + loading_state
    
    # Format assets
    result = format_3d_models(
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
        success_msg = "✅ **Models formatted successfully!** You can use the **3D Model Viewer** below (select 'Formatted Models') to verify the models are still correct after formatting."
    else:
        success_msg = ""  # Error message is shown in the editor
    
    final_state = (
        gr.update(value=success_msg, visible=bool(success_msg)),  # Show success tip
        gr.update(visible=True),   # Show format button
    )
    
    yield final_outputs + final_state


def create_format_wrapper(editor_component, blender_client):
    """Factory function to create a format wrapper bound to a specific editor component and blender client."""
    def format_wrapper(project_dir, anyllm_api_key, vision_model, anyllm_api_base, anyllm_provider, model_id_list=None):
        """Wrapper to properly yield from the generator."""
        # Convert empty list to None (format all)
        if model_id_list is not None and len(model_id_list) == 0:
            model_id_list = None
        for result in show_loading_and_format_models(
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


def load_format_model_choices(project_dir):
    """Load model IDs from concept_artist.json for format selection.
    
    Args:
        project_dir: Project directory path
        
    Returns:
        list: List of asset_id strings
    """
    if not project_dir or not os.path.isabs(project_dir):
        return []
    
    concept_json_path = os.path.join(project_dir, "models", "concept_artist.json")
    
    if not os.path.exists(concept_json_path):
        return []
    
    try:
        with open(concept_json_path, 'r') as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning("Step 2: path conversion failed for concept_artist.json: %s", e)
        
        asset_sheet = data.get("asset_sheet", [])
        model_ids = []
        for asset in asset_sheet:
            asset_id = asset.get("asset_id")
            if asset_id:
                model_ids.append(asset_id)
        
        return model_ids
    except Exception:
        return []


def load_models_from_json(project_dir, model_source="raw"):
    """Load model data from JSON file.
    
    Args:
        project_dir: Project directory path
        model_source: "raw" for concept_artist.json, "formatted" for formatted_model.json, "resized" for resized_model.json
        
    Returns:
        tuple: (list of (model_id, file_path) tuples, error message or None)
    """
    if not project_dir or not os.path.isabs(project_dir):
        return [], "⚠️ Please set a valid project directory first."
    
    if model_source == "resized":
        json_path = os.path.join(project_dir, "resized_model", "resized_model.json")
        source_name = "resized_model.json"
        not_found_msg = "⚠️ No resized models found. Please run Step 3.1 Resize Assets first."
    elif model_source == "formatted":
        json_path = os.path.join(project_dir, "formatted_model", "formatted_model.json")
        source_name = "formatted_model.json"
        not_found_msg = "⚠️ No formatted models found. Please run Step 2.3 Format Assets first."
    else:
        json_path = os.path.join(project_dir, "models", "concept_artist.json")
        source_name = "concept_artist.json"
        not_found_msg = "⚠️ No 3D models found. Please generate 3D assets first."
    
    if not os.path.exists(json_path):
        return [], f"{not_found_msg} ({source_name} not found at {json_path})"
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning("Step 2: path conversion failed for %s: %s", json_path, e)
        
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


def display_3d_models(project_dir, model_source="raw"):
    """Load and display available 3D models.
    
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
    
    models, error = load_models_from_json(project_dir, source_type)
    
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


def select_model(evt: gr.SelectData, models_state):
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


def create_3d_assets_ui(
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
    sketchfab_api_key,
    meshy_api_key,
    meshy_model,
    gemini_api_key,
    gemini_api_base,
    reasoning_model,
    gemini_image_model,
    project_dir,
    blender_client,
    vision_model,
    ai_platform,
    tencent_secret_id,
    tencent_secret_key,
    trellis2_api_base,
    trellis2_max_concurrent,
    trellis2_texture_size,
    trellis2_resolution,
    trellis2_decimation_target,
    image_gen_platform=None,
    openai_api_key=None,
    openai_api_base=None,
    openai_image_model=None,
):
    """Create the Step 2: Fetch and Format Core 3D Assets by Concept Artist UI section.
    
    Args:
        anyllm_api_key: Gradio component for any-llm API key
        anyllm_api_base: Gradio component for any-llm API base URL
        sketchfab_api_key: Gradio component for Sketchfab API key
        meshy_api_key: Gradio component for Meshy API key
        meshy_model: Gradio component for Meshy model version
        gemini_api_key: Gradio component for Gemini API key
        gemini_api_base: Gradio component for Gemini API base URL
        reasoning_model: Gradio component for reasoning model (used for dimension estimation)
        gemini_image_model: Gradio component for Gemini image model
        project_dir: Gradio component for project directory
        blender_client: BlenderClient instance for communicating with Blender
        vision_model: Gradio component for vision model
        ai_platform: Gradio component for AI platform selector ("Hunyuan3D", "Meshy", or "TRELLIS2")
        tencent_secret_id: Gradio component for Tencent Cloud Secret ID
        tencent_secret_key: Gradio component for Tencent Cloud Secret Key
        
    Returns:
        dict with UI components that may be needed by other parts of the app
    """
    gr.Markdown("## Step 2: Fetch and Format Core 3D Assets by Concept Artist")
    gr.Markdown("Generate 3D models for all assets defined in the storyboard using image generation and the selected AI 3D provider.")
    
    # =========================================================================
    # Step 2.1: Generate Image Prompt for 3D Assets
    # =========================================================================
    gr.Markdown("### Step 2.1: Generate Image Prompt for 3D Assets")
    gr.Markdown("Generate preview images for all assets. Review the images and regenerate any that don't look good before proceeding to 3D generation.")
    
    with gr.Row():
        generate_image_btn = gr.Button("🎨 Generate Image Preview", variant="primary", size="lg", scale=2)
        retry_failed_images_btn = gr.Button("♻️ Retry Failed Images", variant="secondary", size="lg", scale=1)
        display_images_btn = gr.Button("🔍 Display Images", variant="secondary", size="lg", scale=1)
        toggle_image_viewer_btn = gr.Button("👁 Hide/Show Image Viewer", variant="secondary", size="lg", scale=1)
    
    # Loading status indicator for image generation
    image_gen_loading_status = gr.Markdown(value="", visible=False)
    
    # Image preview section
    image_preview_status = gr.Markdown(value="", visible=False)
    
    # State for images and regeneration selection
    images_state = gr.State([])
    selected_for_regen = gr.State([])
    image_viewer_visible = gr.State(False)
    
    # Container for image viewer
    with gr.Column(visible=False) as image_viewer_container:
        # Image selection buttons using Dataset
        image_buttons = gr.Dataset(
            components=[gr.Textbox(visible=False)],
            label="Select Image to View",
            samples=[],
            samples_per_page=20,
            visible=False
        )
        
        # Image viewer
        image_viewer = gr.Image(
            label="Image Preview",
            height=400,
            type="filepath"
        )
        
        # Regeneration section
        gr.Markdown("#### Regenerate Selected Images")
        gr.Markdown("Use the checkboxes below to select images you want to regenerate, then click 'Regenerate Selected'.")
        
        # Regeneration selection using CheckboxGroup
        regen_selection = gr.CheckboxGroup(
            choices=[],
            label="Select images to regenerate",
            info="Check the images you want to regenerate"
        )
        
        regen_selection_status = gr.Markdown(value="No images selected for regeneration.", visible=True)
        regenerate_btn = gr.Button("🔄 Regenerate Selected", variant="secondary", size="lg")
    
    # Helper to refresh image viewer while preserving selected image
    def refresh_image_viewer_with_selection(project_dir_val, current_images_state):
        """Refresh image viewer and try to preserve the currently viewed image.
        
        Returns tuple of 8 elements matching display_and_update_regen_choices output:
            (image_viewer, image_preview_status, image_buttons, image_viewer_container,
             images_state, selected_for_regen, regen_selection, image_viewer_visible)
        """
        result = display_image_prompts(project_dir_val)
        # result: image_viewer, image_status, image_buttons, image_viewer_container, images_state, selected_for_regen
        new_images = result[4]  # images_state
        
        # Build choices for regeneration checkbox
        choices = [img[0] for img in new_images] if new_images else []
        
        # Try to find and restore the previously selected image
        # We check if any image from current_images_state matches by model_id
        restored_viewer = result[0]  # default: first image
        restored_status = result[1]  # default: first image status
        
        if current_images_state and new_images:
            # Find the first image that was being displayed (we don't track exact selection,
            # so we'll just keep the first image as default after refresh)
            # The user's selection will be preserved if they had one visible
            pass  # Use defaults from display_image_prompts
        
        return (
            restored_viewer,  # image_viewer
            restored_status,  # image_preview_status
            result[2],  # image_buttons
            result[3],  # image_viewer_container
            new_images,  # images_state
            result[5],  # selected_for_regen
            gr.update(choices=choices, value=[]),  # regen_selection
            True,  # image_viewer_visible
        )
    
    # Generate image prompts button handler with auto-refresh
    def generate_image_prompts_handler(gemini_api_key, gemini_api_base, gemini_image_model, project_dir_val, anyllm_api_key, anyllm_api_base, vision_model, image_gen_platform_val, openai_api_key, openai_api_base, openai_image_model_val, current_images_state):
        # Initial loading - hide viewer updates during loading
        yield (
            gr.update(value="🔄 **Generating image prompts...** This may take a few minutes.", visible=True),  # loading_status
            gr.update(visible=False),  # generate_btn
            gr.update(visible=False),  # regenerate_btn
            gr.update(),  # image_viewer - no change
            gr.update(),  # image_preview_status - no change
            gr.update(),  # image_buttons - no change
            gr.update(),  # image_viewer_container - no change
            current_images_state,  # images_state - preserve
            [],  # selected_for_regen
            gr.update(),  # regen_selection - no change
            gr.update(),  # image_viewer_visible - no change
        )
        
        # Generate image prompts
        result = generate_image_prompts(
            gemini_api_key,
            gemini_api_base,
            gemini_image_model,
            project_dir_val,
            None,  # Generate all
            anyllm_api_key,
            anyllm_api_base,
            vision_model,
            image_gen_platform_val,
            openai_api_key,
            openai_api_base,
            openai_image_model_val,
        )
        
        # Final state with auto-refresh of image viewer
        if result.get("success"):
            success_msg = "✅ **Image prompts generated successfully!** Review the images below and regenerate any that don't look good."
        else:
            success_msg = result.get("error", "")
        
        # Refresh image viewer
        viewer_refresh = refresh_image_viewer_with_selection(project_dir_val, current_images_state)
        
        yield (
            gr.update(value=success_msg, visible=True),  # loading_status
            gr.update(visible=True),   # generate_btn
            gr.update(visible=True),   # regenerate_btn
        ) + viewer_refresh
    
    generate_image_btn.click(
        fn=generate_image_prompts_handler,
        inputs=[gemini_api_key, gemini_api_base, gemini_image_model, project_dir, anyllm_api_key, anyllm_api_base, vision_model, image_gen_platform, openai_api_key, openai_api_base, openai_image_model, images_state],
        outputs=[image_gen_loading_status, generate_image_btn, regenerate_btn,
                 image_viewer, image_preview_status, image_buttons, image_viewer_container,
                 images_state, selected_for_regen, regen_selection, image_viewer_visible]
    )
    
    # Display images button handler
    def display_and_update_regen_choices(project_dir):
        result = display_image_prompts(project_dir)
        # result: image_viewer, image_status, image_buttons, image_viewer_container, images_state, selected_for_regen
        images = result[4]  # images_state
        # Build choices for regeneration checkbox
        choices = [img[0] for img in images] if images else []
        # Return result + regen_selection update + visibility state (True when displaying)
        return result + (gr.update(choices=choices, value=[]), True)
    
    display_images_btn.click(
        fn=display_and_update_regen_choices,
        inputs=[project_dir],
        outputs=[image_viewer, image_preview_status, image_buttons, image_viewer_container, 
                 images_state, selected_for_regen, regen_selection, image_viewer_visible]
    )
    
    # Toggle image viewer visibility
    def toggle_image_viewer(is_visible):
        new_visible = not is_visible
        return gr.update(visible=new_visible), gr.update(visible=new_visible), new_visible
    
    toggle_image_viewer_btn.click(
        fn=toggle_image_viewer,
        inputs=[image_viewer_visible],
        outputs=[image_viewer_container, image_preview_status, image_viewer_visible]
    )
    
    # Image selection handler
    image_buttons.select(
        fn=select_image,
        inputs=[images_state],
        outputs=[image_viewer, image_preview_status]
    )
    
    # Regeneration selection change handler
    def update_regen_status(selected):
        if not selected:
            return gr.update(value="No images selected for regeneration.")
        return gr.update(value=f"**Selected for regeneration ({len(selected)}):** {', '.join(selected)}")
    
    regen_selection.change(
        fn=update_regen_status,
        inputs=[regen_selection],
        outputs=[regen_selection_status]
    )
    
    # Regenerate selected button handler with auto-refresh
    def regenerate_selected_handler(gemini_api_key, gemini_api_base, gemini_image_model, project_dir_val, selected_ids, anyllm_api_key, anyllm_api_base, vision_model, image_gen_platform_val, openai_api_key, openai_api_base, openai_image_model_val, current_images_state):
        # Build empty viewer updates for error case
        empty_viewer_updates = (
            gr.update(),  # image_viewer
            gr.update(),  # image_preview_status
            gr.update(),  # image_buttons
            gr.update(),  # image_viewer_container
            current_images_state,  # images_state - preserve
            [],  # selected_for_regen
            gr.update(),  # regen_selection
            gr.update(),  # image_viewer_visible
        )
        
        if not selected_ids:
            yield (
                gr.update(value="⚠️ Please select at least one image to regenerate.", visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            ) + empty_viewer_updates
            return
        
        # Initial loading state
        yield (
            gr.update(value=f"🔄 **Regenerating {len(selected_ids)} image(s)...** This may take a few minutes.", visible=True),
            gr.update(visible=False),  # generate_btn
            gr.update(visible=False),  # regenerate_btn
        ) + empty_viewer_updates
        
        # Generate image prompts for selected IDs
        result = generate_image_prompts(
            gemini_api_key,
            gemini_api_base,
            gemini_image_model,
            project_dir_val,
            selected_ids,  # Only regenerate selected
            anyllm_api_key,
            anyllm_api_base,
            vision_model,
            image_gen_platform_val,
            openai_api_key,
            openai_api_base,
            openai_image_model_val,
        )
        
        # Final state with auto-refresh
        if result.get("success"):
            success_msg = "✅ **Image prompts regenerated successfully!** Review the images below."
        else:
            success_msg = result.get("error", "")
        
        # Refresh image viewer and try to show a regenerated image
        viewer_refresh = refresh_image_viewer_with_selection(project_dir_val, current_images_state)
        
        # Try to show one of the regenerated images if possible
        new_images = viewer_refresh[4]  # images_state from refresh
        if selected_ids and new_images:
            # Find the first regenerated image in new_images
            for idx, (model_id, image_path, has_error) in enumerate(new_images):
                if model_id in selected_ids and image_path:
                    # Update viewer to show this regenerated image
                    status_suffix = " ⚠️ (has error)" if has_error else ""
                    viewer_refresh = (
                        gr.update(value=image_path),  # image_viewer - show regenerated
                        gr.update(value=f"Showing: **{model_id}**{status_suffix} ({len(new_images)} images available)"),
                        viewer_refresh[2],  # image_buttons
                        viewer_refresh[3],  # image_viewer_container
                        new_images,  # images_state
                        viewer_refresh[5],  # selected_for_regen
                        viewer_refresh[6],  # regen_selection
                        viewer_refresh[7],  # image_viewer_visible
                    )
                    break
        
        yield (
            gr.update(value=success_msg, visible=True),
            gr.update(visible=True),   # generate_btn
            gr.update(visible=True),   # regenerate_btn
        ) + viewer_refresh
    
    regenerate_btn.click(
        fn=regenerate_selected_handler,
        inputs=[gemini_api_key, gemini_api_base, gemini_image_model, project_dir, regen_selection, anyllm_api_key, anyllm_api_base, vision_model, image_gen_platform, openai_api_key, openai_api_base, openai_image_model, images_state],
        outputs=[image_gen_loading_status, generate_image_btn, regenerate_btn,
                 image_viewer, image_preview_status, image_buttons, image_viewer_container,
                 images_state, selected_for_regen, regen_selection, image_viewer_visible]
    )

    # Retry Failed Images: auto-discover assets in image_prompt.json that have
    # no image_prompt_path / missing file / image_prompt_error and re-run only them.
    def retry_failed_images_handler(gemini_api_key_v, gemini_api_base_v, gemini_image_model_v,
                                    project_dir_val, anyllm_api_key_v, anyllm_api_base_v,
                                    vision_model_v, image_gen_platform_val,
                                    openai_api_key_v, openai_api_base_v, openai_image_model_val,
                                    current_images_state):
        empty_viewer_updates = (
            gr.update(),  # image_viewer
            gr.update(),  # image_preview_status
            gr.update(),  # image_buttons
            gr.update(),  # image_viewer_container
            current_images_state,
            [],
            gr.update(),
            gr.update(),
        )

        failed_ids = find_failed_image_prompt_ids(project_dir_val)
        if not failed_ids:
            yield (
                gr.update(value="✅ No failed images found in `image_prompt.json`. Nothing to retry.", visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            ) + empty_viewer_updates
            return

        # Initial loading
        yield (
            gr.update(value=f"🔄 **Retrying {len(failed_ids)} failed image(s)...** This may take a few minutes.", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        ) + empty_viewer_updates

        result = generate_image_prompts(
            gemini_api_key_v,
            gemini_api_base_v,
            gemini_image_model_v,
            project_dir_val,
            failed_ids,
            anyllm_api_key_v,
            anyllm_api_base_v,
            vision_model_v,
            image_gen_platform_val,
            openai_api_key_v,
            openai_api_base_v,
            openai_image_model_val,
        )

        if result.get("success"):
            success_msg = (
                f"✅ **Retried {len(failed_ids)} failed image(s).** "
                "Review below; some may still have failed if the API errored again."
            )
        else:
            success_msg = result.get("error", "")

        viewer_refresh = refresh_image_viewer_with_selection(project_dir_val, current_images_state)

        # If possible, focus viewer on the first retried image
        new_images = viewer_refresh[4]
        if new_images:
            for model_id, image_path, has_error in new_images:
                if model_id in failed_ids and image_path:
                    status_suffix = " ⚠️ (has error)" if has_error else ""
                    viewer_refresh = (
                        gr.update(value=image_path),
                        gr.update(value=f"Showing: **{model_id}**{status_suffix} ({len(new_images)} images available)"),
                        viewer_refresh[2],
                        viewer_refresh[3],
                        new_images,
                        viewer_refresh[5],
                        viewer_refresh[6],
                        viewer_refresh[7],
                    )
                    break

        yield (
            gr.update(value=success_msg, visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        ) + viewer_refresh

    retry_failed_images_btn.click(
        fn=retry_failed_images_handler,
        inputs=[gemini_api_key, gemini_api_base, gemini_image_model, project_dir,
                anyllm_api_key, anyllm_api_base, vision_model, image_gen_platform,
                openai_api_key, openai_api_base, openai_image_model, images_state],
        outputs=[image_gen_loading_status, generate_image_btn, regenerate_btn,
                 image_viewer, image_preview_status, image_buttons, image_viewer_container,
                 images_state, selected_for_regen, regen_selection, image_viewer_visible]
    )

    # =========================================================================
    # Step 2.2: Generate 3D Assets
    # =========================================================================
    gr.Markdown("### Step 2.2: Generate 3D Assets")
    gr.Markdown("Generate 3D models from the image prompts using the selected 3D provider. This step may take 10-20 minutes.")
    
    with gr.Row():
        generate_3d_btn = gr.Button("🧊 Generate 3D Assets", variant="primary", size="lg", scale=2)
        retry_failed_3d_btn = gr.Button("♻️ Retry Failed 3D Models", variant="secondary", size="lg", scale=1)
    
    # Loading status indicator (hidden by default)
    loading_status_3d = gr.Markdown(value="", visible=False)
    
    # Create JSON editor component for displaying results
    assets_editor = JSONEditorComponent(
        label="Concept Artist Result",
        visible_initially=False,
        file_basename="concept_artist",
        use_version_control=False,  # Save as concept_artist.json (overwrites)
        json_root_keys_list=["asset_sheet"],
        title="Step 2.2"
    )
    
    # Wire up the Resume button with project_dir input
    assets_editor.setup_resume_with_project_dir(project_dir, subfolder="models")
    
    # Create wrapper function for the 3D generator
    generate_3d_wrapper = create_generate_3d_wrapper(assets_editor)
    retry_failed_3d_wrapper = create_retry_failed_3d_wrapper(assets_editor)
    
    # Generate 3D button click handler
    generate_3d_btn.click(
        fn=generate_3d_wrapper,
        inputs=[
            meshy_api_key, meshy_model, project_dir, ai_platform, tencent_secret_id,
            tencent_secret_key, trellis2_api_base, trellis2_max_concurrent,
            trellis2_texture_size, trellis2_resolution, trellis2_decimation_target,
        ],
        outputs=assets_editor.get_output_components() + [loading_status_3d, generate_3d_btn],
    )
    retry_failed_3d_btn.click(
        fn=retry_failed_3d_wrapper,
        inputs=[
            meshy_api_key, meshy_model, project_dir, ai_platform, tencent_secret_id,
            tencent_secret_key, trellis2_api_base, trellis2_max_concurrent,
            trellis2_texture_size, trellis2_resolution, trellis2_decimation_target,
        ],
        outputs=assets_editor.get_output_components() + [loading_status_3d, retry_failed_3d_btn],
    )
    
    # =========================================================================
    # Step 2.3: Format Assets (Orientation Correction)
    # =========================================================================
    gr.Markdown("### Step 2.3: Format Assets (Orientation Correction)")
    gr.Markdown("Correct model orientation in Blender using AI vision analysis. **Requires Blender with BlenderMCPServer running.**")
    
    gr.Markdown("#### Select Models to Format")
    gr.Markdown("Select specific models to re-format (useful when some models failed or orientation is incorrect). Leave empty or click 'Select All' to format all models.")
    
    with gr.Row():
        format_model_selection = gr.CheckboxGroup(
            choices=[],
            label="Select models to format",
            info="Check the models you want to format. If none selected, all models will be formatted.",
            scale=3
        )
        with gr.Column(scale=1):
            load_format_choices_btn = gr.Button("🔄 Load Models", variant="secondary", size="sm")
            select_all_format_btn = gr.Button("☑️ Select All", variant="secondary", size="sm")
    
    format_selection_status = gr.Markdown(value="No models loaded. Click 'Load Models' to load available models.", visible=True)
    
    format_btn = gr.Button("🛠️ Format Assets", variant="primary", size="lg")
    
    # Loading status indicator for format assets (hidden by default)
    format_loading_status = gr.Markdown(value="", visible=False)
    
    # Create JSON editor component for formatted models results
    formatted_editor = JSONEditorComponent(
        label="Formatted Models Result",
        visible_initially=False,
        file_basename="formatted_model",
        use_version_control=False,  # Save as formatted_model.json (overwrites)
        json_root_keys_list=["asset_sheet"],
        title="Step 2.3"
    )
    
    # Wire up the Resume button with project_dir input
    formatted_editor.setup_resume_with_project_dir(project_dir, subfolder="formatted_model")
    
    # Create wrapper function for format assets
    format_wrapper = create_format_wrapper(formatted_editor, blender_client)
    
    # Handler for Load Models button
    def load_format_choices_handler(project_dir_val):
        choices = load_format_model_choices(project_dir_val)
        if choices:
            status_msg = f"**{len(choices)} model(s) available.** Select models to re-format, or leave empty to format all."
        else:
            status_msg = "⚠️ No models found. Please complete Step 2.2 (Generate 3D Assets) first."
        return gr.update(choices=choices, value=[]), gr.update(value=status_msg)
    
    load_format_choices_btn.click(
        fn=load_format_choices_handler,
        inputs=[project_dir],
        outputs=[format_model_selection, format_selection_status]
    )
    
    # Handler for Select All button
    def select_all_format_handler(project_dir_val):
        choices = load_format_model_choices(project_dir_val)
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
    # Step 2.4: Estimate Asset Dimensions
    # =========================================================================
    gr.Markdown("### Step 2.4: Estimate Asset Dimensions")
    gr.Markdown("Estimate real-world dimensions for each 3D asset using AI analysis. Requires formatted_model.json from Step 2.3.")
    
    estimate_btn = gr.Button("📏 Estimate Asset Dimensions", variant="primary", size="lg")
    
    # Loading status indicator for dimension estimation (hidden by default)
    dimension_loading_status = gr.Markdown(value="", visible=False)
    
    # Create JSON editor component for dimension estimation results
    dimension_editor = JSONEditorComponent(
        label="Dimension Estimation Result",
        visible_initially=False,
        file_basename="dimension_estimation",
        use_version_control=False,  # Save as dimension_estimation.json (overwrites)
        json_root_keys_list=["asset_sheet"],
        title="Step 2.4"
    )
    
    # Wire up the Resume button with project_dir input
    dimension_editor.setup_resume_with_project_dir(project_dir, subfolder="formatted_model")
    
    # Create wrapper function for dimension estimation
    estimate_wrapper = create_estimate_wrapper(dimension_editor)
    
    # Estimate button click handler
    estimate_btn.click(
        fn=estimate_wrapper,
        inputs=[
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
            reasoning_model,
            project_dir,
        ],
        outputs=dimension_editor.get_output_components() + [dimension_loading_status, estimate_btn],
    )
    
    # --- 3D Model Viewer Section ---
    gr.Markdown("### 3D Model Viewer")
    
    with gr.Row():
        model_source_radio = gr.Radio(
            choices=["Raw Models", "Formatted Models", "Resized Models"],
            value="Raw Models",
            label="Model Source",
            info="Choose which models to display, then click 🔍 Display 3D Models to apply",
            scale=2
        )
        display_btn = gr.Button("🔍 Display 3D Models", variant="secondary", size="lg", scale=1)
        toggle_btn = gr.Button("👁 Hide/Show 3D Viewer", variant="secondary", size="lg", scale=1)
    
    # Status message for model viewer
    model_status = gr.Markdown(value="", visible=False)
    
    # State to store the list of models (id, path)
    models_state = gr.State([])
    
    # State to track viewer visibility
    viewer_visible = gr.State(False)
    
    # Container for 3D model viewer (hidden initially)
    with gr.Column(visible=False) as model_viewer_container:
        # Model selection buttons using Dataset (similar to Examples)
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
        gr.Markdown("#### Regenerate Selected 3D Models")
        gr.Markdown("Select models you want to regenerate from scratch using their image prompts, then click 'Regenerate Selected'. Afterwards, check Step 2.2 for progress.")
        
        # Regeneration selection using CheckboxGroup
        model_regen_selection = gr.CheckboxGroup(
            choices=[],
            label="Select 3D models to regenerate",
            info="Check the 3D models you want to regenerate"
        )
        
        model_regen_status = gr.Markdown(value="No models selected for regeneration.", visible=True)
        regenerate_3d_btn = gr.Button("🔄 Regenerate Selected 3D Models", variant="secondary", size="lg")
    
    # Create regenerate 3D wrapper
    regenerate_3d_wrapper = create_regenerate_3d_wrapper(assets_editor)
    
    def toggle_viewer(is_visible):
        """Toggle the visibility of the 3D viewer and status."""
        new_visible = not is_visible
        return gr.update(visible=new_visible), gr.update(visible=new_visible), new_visible
    
    def display_and_show(project_dir, model_source):
        """Display 3D models and set visibility to True, also update regen choices."""
        result = display_3d_models(project_dir, model_source)
        # result: model_viewer, model_status, model_buttons, model_viewer_container, models_state
        models = result[4]  # models_state
        # Build choices for regeneration checkbox
        choices = [m[0] for m in models] if models else []
        # Append True for visibility state and regen_selection update
        return result + (True, gr.update(choices=choices, value=[]))
    
    # Display button click handler
    display_btn.click(
        fn=display_and_show,
        inputs=[project_dir, model_source_radio],
        outputs=[model_viewer, model_status, model_buttons, model_viewer_container, models_state, viewer_visible, model_regen_selection]
    )
    
    # Toggle button click handler
    toggle_btn.click(
        fn=toggle_viewer,
        inputs=[viewer_visible],
        outputs=[model_viewer_container, model_status, viewer_visible]
    )
    
    # Model selection handler
    model_buttons.select(
        fn=select_model,
        inputs=[models_state],
        outputs=[model_viewer, model_status]
    )
    
    # Model regeneration selection change handler
    def update_model_regen_status(selected):
        if not selected:
            return gr.update(value="No models selected for regeneration.")
        return gr.update(value=f"**Selected for regeneration ({len(selected)}):** {', '.join(selected)}")
    
    model_regen_selection.change(
        fn=update_model_regen_status,
        inputs=[model_regen_selection],
        outputs=[model_regen_status]
    )
    
    # Regenerate 3D models button handler
    regenerate_3d_btn.click(
        fn=regenerate_3d_wrapper,
        inputs=[
            meshy_api_key, meshy_model, project_dir, ai_platform, tencent_secret_id,
            tencent_secret_key, trellis2_api_base, trellis2_max_concurrent,
            trellis2_texture_size, trellis2_resolution, trellis2_decimation_target,
            model_regen_selection,
        ],
        outputs=assets_editor.get_output_components() + [loading_status_3d, generate_3d_btn]
    )
    
    return {
        "assets_editor": assets_editor,
        "dimension_editor": dimension_editor,
        "formatted_editor": formatted_editor,
        "model_viewer": model_viewer,
    }
