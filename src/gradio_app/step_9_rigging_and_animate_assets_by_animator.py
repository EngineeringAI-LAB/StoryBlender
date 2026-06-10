import os
import json
import logging
import shutil
import tempfile
import gradio as gr
from copy import deepcopy
from ..operators.animator_operators.rigging import rig_models, ensure_valid_rig_task, MeshyRiggingAPIError
from ..operators.animator_operators.animator import generate_animation_selection, animate_rigged_model, apply_single_animation, plan_movement_for_meshy_animations
from ..operators.animator_operators.animation_planner import generate_animation_plan
from ..operators.animator_operators.uthana_animator import uthana_animate_characters
from ..operators.animator_operators.auto_route_animator import auto_route_animate, retry_failed_animations, _find_latest_animated_models
from .json_editor import JSONEditorComponent
from .blender_client import BlenderClient
from .path_utils import make_paths_absolute, make_paths_relative

logger = logging.getLogger(__name__)


def _log_io_paths(action: str, input_path=None, output_path=None) -> None:
    """Print input/output JSON paths for a Step 9 action to the terminal.

    Used by every long-running button handler so the user can always see
    which JSON file was read and which one was written, even after the
    task completes.
    """
    banner = "-" * 70
    print(banner, flush=True)
    print(f"[Step 9] {action}", flush=True)
    if input_path:
        print(f"  📂 Input  JSON: {input_path}", flush=True)
    if output_path:
        print(f"  💾 Output JSON: {output_path}", flush=True)
    print(banner, flush=True)


def _io_paths_md(result: dict, *, input_label: str = "Input JSON",
                 output_label: str = "Output JSON") -> str:
    """Return a Markdown snippet with input/output paths for inclusion in a
    persistent gradio status message. Returns empty string if neither path
    is available.
    """
    if not isinstance(result, dict):
        return ""
    parts = []
    ip = result.get("input_path")
    op = result.get("output_path")
    if ip:
        parts.append(f"📂 **{input_label}:** `{ip}`")
    if op:
        parts.append(f"💾 **{output_label}:** `{op}`")
    return ("\n\n" + "\n\n".join(parts)) if parts else ""


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


def validate_and_rig_characters(
    meshy_api_key,
    project_dir,
    editor_component
):
    """Validate inputs and rig character models.
    
    Args:
        meshy_api_key: The Meshy API key for rigging
        project_dir: The absolute path to the project directory
        editor_component: The JSONEditorComponent to save the result
    
    Returns:
        A dictionary containing the result of the rigging operation
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path (e.g., /Users/username/projects/my_project)"
        }
    
    # Validate API key
    if not meshy_api_key or meshy_api_key.strip() == "":
        return {
            "error": "⚠️ Please provide a valid Meshy API key"
        }
    
    # Set up output directory for rigged models
    rigged_models_dir = os.path.join(project_dir, "rigged_models")
    os.makedirs(rigged_models_dir, exist_ok=True)
    
    # Set the save path for rigged_models (for the editor component)
    editor_component.set_save_path(rigged_models_dir)
    
    # Find the latest layout_script as input
    layout_script_dir = os.path.join(project_dir, "layout_script")
    if not os.path.exists(layout_script_dir):
        return {
            "error": "⚠️ No layout_script folder found. Please complete previous steps first."
        }
    
    # Find the latest layout_script_v*.json file
    latest_version = 0
    input_path = None
    for filename in os.listdir(layout_script_dir):
        if filename.startswith("layout_script_v") and filename.endswith(".json"):
            try:
                version_str = filename[len("layout_script_v"):-5]
                version = int(version_str)
                if version > latest_version:
                    latest_version = version
                    input_path = os.path.join(layout_script_dir, filename)
            except ValueError:
                continue
    
    if input_path is None:
        return {
            "error": "⚠️ No layout_script found. Please complete previous steps first."
        }
    
    _log_io_paths("Step 9.1 Rigging — starting", input_path=input_path)
    try:
        # Call the rig_models function
        result_data = rig_models(
            path_to_input_json=input_path,
            output_dir=rigged_models_dir,
            meshy_api_key=meshy_api_key,
        )
        
        # Save the result to rigged_models/rigged_model.json (no version control)
        output_path = editor_component.save_json_data(result_data)
        
        if output_path:
            _log_io_paths("Step 9.1 Rigging — done", input_path=input_path, output_path=output_path)
            return {
                "success": True,
                "data": result_data,
                "output_path": output_path,
                "input_path": input_path,
            }
        else:
            return {
                "error": "⚠️ Failed to save JSON file"
            }
        
    except Exception as e:
        return {
            "error": f"⚠️ Rigging failed: {str(e)}"
        }


def show_loading_and_rig(editor_component, meshy_api_key, project_dir):
    """Show loading indicator and perform rigging."""
    # Build initial loading state - all editor components hidden
    loading_outputs = editor_component.update_with_result(None)
    loading_state = (
        gr.update(value="🔄 **Rigging characters...** This may take several minutes depending on the number of characters. Please wait.", visible=True),  # Show loading
        gr.update(visible=False),  # Hide rig_btn
    )
    
    yield loading_outputs + loading_state
    
    # Perform the rigging
    result = validate_and_rig_characters(
        meshy_api_key, project_dir, editor_component
    )
    
    # Handle error case
    if "error" in result:
        error_outputs = editor_component.update_with_result(None)
        error_state = (
            gr.update(value=result["error"], visible=True),  # Show error
            gr.update(visible=True),  # Show rig_btn
        )
        yield error_outputs + error_state
        return
    
    # Return final result with editor component updated
    final_outputs = editor_component.update_with_result(result)
    
    # Show success message if rigging succeeded
    success_msg = "✅ **Character rigging completed successfully!** You can proceed to the next step." + _io_paths_md(result)
    final_state = (
        gr.update(value=success_msg, visible=True),  # Show success message
        gr.update(visible=True),   # Show rig_btn for re-run
    )
    
    yield final_outputs + final_state


def create_rig_wrapper(editor_component):
    """Factory function to create a rig wrapper bound to a specific editor component."""
    def rig_wrapper(meshy_api_key, project_dir):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_rig(editor_component, meshy_api_key, project_dir):
            yield result
    return rig_wrapper


def load_rigged_models_from_json(project_dir):
    """Load rigged model data from rigged_models/rigged_model.json.
    
    Args:
        project_dir: Project directory path
        
    Returns:
        tuple: (list of (asset_id, rigged_running_file_path) tuples, error message or None)
    """
    if not project_dir or not os.path.isabs(project_dir):
        return [], "⚠️ Please set a valid project directory first."
    
    rigged_model_path = os.path.join(project_dir, "rigged_models", "rigged_model.json")
    
    if not os.path.exists(rigged_model_path):
        return [], "⚠️ No rigged_model.json found. Please complete rigging first."
    
    try:
        with open(rigged_model_path, 'r') as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning("Step 9: path conversion failed for rigged_model.json: %s", e)
        
        models = []
        asset_sheet = data.get("asset_sheet", [])
        for asset in asset_sheet:
            asset_id = asset.get("asset_id")
            rigged_running_path = asset.get("rigged_running_file_path")
            
            # Only include assets that have rigged_running_file_path
            if asset_id and rigged_running_path and os.path.exists(rigged_running_path):
                # Use 323 path to force Gradio to reload
                cache_busted_path = get_cache_busted_file_path(rigged_running_path, "model_cache")
                models.append((asset_id, cache_busted_path))
        
        if not models:
            return [], "⚠️ No rigged models with animations found. Please run rigging first."
        
        return models, None
    except Exception as e:
        return [], f"⚠️ Failed to load rigged models: {str(e)}"


def display_rigged_models(project_dir):
    """Load and display available rigged models.
    
    Args:
        project_dir: Project directory path
    
    Returns updates for: model_viewer, model_status, model_buttons, model_viewer_container, models_state
    """
    models, error = load_rigged_models_from_json(project_dir)
    
    if error:
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
        gr.update(value=f"Showing: **{models[0][0]}** ({len(models)} rigged models available)", visible=True),  # model_status
        gr.update(samples=model_ids, visible=True),  # model_buttons (Dataset)
        gr.update(visible=True),  # model_viewer_container
        models  # models_state: list of (id, path) tuples
    )


def select_rigged_model(evt: gr.SelectData, models_state):
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
        gr.update(value=f"Showing: **{model_id}** ({len(models_state)} rigged models available)")  # model_status
    )


def validate_and_select_animations(
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
    vision_model,
    project_dir,
    editor_component
):
    """Validate inputs and generate animation selections.
    
    Args:
        anyllm_api_key: The any-llm API key
        anyllm_api_base: The any-llm API base URL
        anyllm_provider: The LLM provider (e.g. 'gemini', 'openai')
        vision_model: The vision model to use
        project_dir: The absolute path to the project directory
        editor_component: The JSONEditorComponent to save the result
    
    Returns:
        A dictionary containing the result of the animation selection
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path (e.g., /Users/username/projects/my_project)"
        }
    
    # Validate API key
    if not anyllm_api_key or anyllm_api_key.strip() == "":
        return {
            "error": "⚠️ Please provide a valid any-llm API key"
        }
    
    # Check input file exists
    input_path = os.path.join(project_dir, "rigged_models", "rigged_model.json")
    if not os.path.exists(input_path):
        return {
            "error": "⚠️ No rigged_model.json found. Please complete Step 9.1 (Rigging) first."
        }
    
    # Set up output directory for animated models
    animated_models_dir = os.path.join(project_dir, "animated_models")
    os.makedirs(animated_models_dir, exist_ok=True)
    
    # Set the save path for the editor component
    editor_component.set_save_path(animated_models_dir)
    
    try:
        # Load input JSON
        with open(input_path, 'r') as f:
            storyboard_script = json.load(f)
        try:
            storyboard_script = make_paths_absolute(storyboard_script, project_dir)
        except Exception as e:
            logger.warning("Step 9: path conversion failed for input JSON: %s", e)
        
        # Call the generate_animation_selection function
        # Set provider to default if empty
        _provider = anyllm_provider.strip() if anyllm_provider else "gemini"
        if not _provider:
            _provider = "gemini"

        result_data = generate_animation_selection(
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base if anyllm_api_base else None,
            anyllm_provider=_provider,
            vision_model=vision_model if vision_model else "gemini/gemini-3-flash-preview",
            storyboard_script=storyboard_script,
            num_candidates=3,
            max_retries=3,
            max_concurrent=10
        )
        
        if result_data is None:
            return {
                "error": "⚠️ Animation selection failed. Check console for details."
            }
        
        # Save the result as a new version
        output_path = editor_component.save_json_data(result_data)
        
        if output_path:
            _log_io_paths("Step 9.2 Animation Selection — done", input_path=input_path, output_path=output_path)
            return {
                "success": True,
                "data": result_data,
                "output_path": output_path,
                "input_path": input_path,
            }
        else:
            return {
                "error": "⚠️ Failed to save JSON file"
            }
        
    except Exception as e:
        return {
            "error": f"⚠️ Animation selection failed: {str(e)}"
        }


def show_loading_and_select_animations(editor_component, anyllm_api_key, anyllm_api_base, anyllm_provider, vision_model, project_dir):
    """Show loading indicator and perform animation selection."""
    # Build initial loading state - all editor components hidden
    loading_outputs = editor_component.update_with_result(None)
    loading_state = (
        gr.update(value="🔄 **Selecting animations...** This may take several minutes depending on the number of character actions. Please wait.", visible=True),
        gr.update(visible=False),  # Hide select_btn
    )
    
    yield loading_outputs + loading_state
    
    # Perform the animation selection
    result = validate_and_select_animations(
        anyllm_api_key, anyllm_api_base, anyllm_provider, vision_model, project_dir, editor_component
    )
    
    # Handle error case
    if "error" in result:
        error_outputs = editor_component.update_with_result(None)
        error_state = (
            gr.update(value=result["error"], visible=True),
            gr.update(visible=True),  # Show select_btn
        )
        yield error_outputs + error_state
        return
    
    # Return final result with editor component updated
    final_outputs = editor_component.update_with_result(result)
    
    # Show success message if selection succeeded
    success_msg = "✅ **Animation selection completed successfully!** You can review the selected animations and proceed to the next step." + _io_paths_md(result)
    final_state = (
        gr.update(value=success_msg, visible=True),  # Show success message
        gr.update(visible=True),   # Show select_btn for re-run
    )
    
    yield final_outputs + final_state


def create_select_animations_wrapper(editor_component):
    """Factory function to create an animation selection wrapper bound to a specific editor component."""
    def select_wrapper(anyllm_api_key, anyllm_api_base, anyllm_provider, vision_model, project_dir):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_select_animations(
            editor_component, anyllm_api_key, anyllm_api_base, anyllm_provider, vision_model, project_dir
        ):
            yield result
    return select_wrapper


def get_latest_versioned_json(directory, basename):
    """Get the path to the latest versioned JSON file.
    
    Args:
        directory: Directory to search in
        basename: Base name of the file (e.g., 'selected_animation')
        
    Returns:
        tuple: (path to latest file, version number) or (None, 0) if not found
    """
    if not os.path.exists(directory):
        return None, 0
    
    latest_version = 0
    latest_path = None
    
    for filename in os.listdir(directory):
        if filename.startswith(f"{basename}_v") and filename.endswith(".json"):
            try:
                version_str = filename[len(f"{basename}_v"):-5]
                version = int(version_str)
                if version > latest_version:
                    latest_version = version
                    latest_path = os.path.join(directory, filename)
            except ValueError:
                continue
    
    return latest_path, latest_version


def merge_rigging_task_info(project_dir):
    """Merge rigging task information from rigged_model.json into selected_animation and animated_models JSONs.
    
    Reads rig info (rig_task_id, rig_expires_at, rigged_file_path, rigged_running_file_path)
    from rigged_model.json and merges it into each character asset in both selected_animation
    and animated_models (if exists).
    
    Args:
        project_dir: The absolute path to the project directory
    
    Returns:
        tuple: (success message or None, error message or None, output_paths list or None)
    """
    if not project_dir or not os.path.isabs(project_dir):
        return None, "⚠️ Project directory must be an absolute path", None
    
    # Read rigged_model.json
    rigged_model_path = os.path.join(project_dir, "rigged_models", "rigged_model.json")
    if not os.path.exists(rigged_model_path):
        return None, "⚠️ No rigged_model.json found. Please complete Step 9.1 (Rigging) first.", None
    
    try:
        with open(rigged_model_path, 'r') as f:
            rigged_data = json.load(f)
        try:
            rigged_data = make_paths_absolute(rigged_data, project_dir)
        except Exception as e:
            logger.warning("Step 9: path conversion failed for rigged_model.json: %s", e)
    except Exception as e:
        return None, f"⚠️ Failed to read rigged_model.json: {str(e)}", None
    
    # Build a mapping of asset_id -> rig info from rigged_model.json
    rig_info_map = {}
    for asset in rigged_data.get("asset_sheet", []):
        asset_id = asset.get("asset_id")
        if asset_id and asset.get("asset_type") == "character":
            rig_info_map[asset_id] = {
                "rig_task_id": asset.get("rig_task_id"),
                "rig_expires_at": asset.get("rig_expires_at"),
                "rigged_file_path": asset.get("rigged_file_path"),
                "rigged_running_file_path": asset.get("rigged_running_file_path"),
            }
    
    if not rig_info_map:
        return None, "⚠️ No character assets with rig info found in rigged_model.json", None
    
    animated_models_dir = os.path.join(project_dir, "animated_models")
    output_paths = []
    total_updated = 0
    
    # Helper function to merge rig info into a JSON data structure
    def merge_rig_info_into_data(data):
        count = 0
        for asset in data.get("asset_sheet", []):
            asset_id = asset.get("asset_id")
            if asset_id and asset_id in rig_info_map:
                rig_info = rig_info_map[asset_id]
                asset["rig_task_id"] = rig_info["rig_task_id"]
                asset["rig_expires_at"] = rig_info["rig_expires_at"]
                asset["rigged_file_path"] = rig_info["rigged_file_path"]
                asset["rigged_running_file_path"] = rig_info["rigged_running_file_path"]
                count += 1
        return count
    
    # Process selected_animation JSON
    selected_anim_path, selected_version = get_latest_versioned_json(animated_models_dir, "selected_animation")
    if selected_anim_path:
        try:
            with open(selected_anim_path, 'r') as f:
                selected_data = json.load(f)
            try:
                selected_data = make_paths_absolute(selected_data, project_dir)
            except Exception as e:
                logger.warning("Step 9: path conversion failed for selected_animation: %s", e)
            
            updated_count = merge_rig_info_into_data(selected_data)
            if updated_count > 0:
                new_version = selected_version + 1
                output_path = os.path.join(animated_models_dir, f"selected_animation_v{new_version}.json")
                try:
                    save_data = make_paths_relative(selected_data, project_dir)
                except Exception as e:
                    logger.warning("Step 9: path conversion failed on save: %s", e)
                    save_data = selected_data
                with open(output_path, 'w') as f:
                    json.dump(save_data, f, indent=2)
                output_paths.append(output_path)
                total_updated += updated_count
        except Exception as e:
            return None, f"⚠️ Failed to process selected_animation: {str(e)}", None
    
    # Process animated_models JSON (if exists)
    animated_path, animated_version = get_latest_versioned_json(animated_models_dir, "animated_models")
    if animated_path:
        try:
            with open(animated_path, 'r') as f:
                animated_data = json.load(f)
            try:
                animated_data = make_paths_absolute(animated_data, project_dir)
            except Exception as e:
                logger.warning("Step 9: path conversion failed for animated_models: %s", e)
            
            updated_count = merge_rig_info_into_data(animated_data)
            if updated_count > 0:
                new_version = animated_version + 1
                output_path = os.path.join(animated_models_dir, f"animated_models_v{new_version}.json")
                try:
                    save_data = make_paths_relative(animated_data, project_dir)
                except Exception as e:
                    logger.warning("Step 9: path conversion failed on save: %s", e)
                    save_data = animated_data
                with open(output_path, 'w') as f:
                    json.dump(save_data, f, indent=2)
                output_paths.append(output_path)
                total_updated += updated_count
        except Exception as e:
            return None, f"⚠️ Failed to process animated_models: {str(e)}", None
    
    if not output_paths:
        return None, "⚠️ No selected_animation or animated_models JSON found to update.", None
    
    if total_updated == 0:
        return None, "⚠️ No matching character assets found to update", None
    
    # Build success message
    files_msg = "\n".join([f"  - `{p}`" for p in output_paths])
    success_msg = f"✅ Merged rig info for character assets. Saved to:\n{files_msg}"
    return success_msg, None, output_paths


def validate_and_animate_all(
    meshy_api_key,
    project_dir,
    editor_component,
    anyllm_api_key=None,
    anyllm_api_base=None,
    anyllm_provider=None,
    vision_model=None,
):
    """Validate inputs, animate all rigged models, then plan movement for
    walking/running animations.
    
    Args:
        meshy_api_key: The Meshy API key
        project_dir: The absolute path to the project directory
        editor_component: The JSONEditorComponent to save the result
        anyllm_api_key: LLM API key for movement planning
        anyllm_api_base: LLM API base URL
        vision_model: LLM vision model name
    
    Returns:
        A dictionary containing the result of the animation operation
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path"
        }
    
    # Validate API key
    if not meshy_api_key or meshy_api_key.strip() == "":
        return {
            "error": "⚠️ Please provide a valid Meshy API key"
        }
    
    # Find the latest selected_animation JSON
    animated_models_dir = os.path.join(project_dir, "animated_models")
    input_path, _ = get_latest_versioned_json(animated_models_dir, "selected_animation")
    
    if not input_path:
        return {
            "error": "⚠️ No selected_animation JSON found. Please complete Step 9.2 first."
        }
    
    # Set up output directory
    os.makedirs(animated_models_dir, exist_ok=True)
    editor_component.set_save_path(animated_models_dir)
    
    _log_io_paths("Step 9.3 Animate All — starting", input_path=input_path)
    try:
        # Call the animate_rigged_model function
        result = animate_rigged_model(
            path_to_input_json=input_path,
            output_dir=animated_models_dir,
            meshy_api_key=meshy_api_key,
            max_concurrent=10
        )
        
        updated_json = result.get("updated_json")
        if updated_json is None:
            return {
                "error": "⚠️ Animation failed. No animations were successful."
            }
        
        # Plan movement (start/end coords + rotation) for walking/running anims
        if anyllm_api_key:
            try:
                _provider = anyllm_provider.strip() if anyllm_provider else "gemini"
                if not _provider:
                    _provider = "gemini"
                updated_json = plan_movement_for_meshy_animations(
                    storyboard_script=updated_json,
                    anyllm_api_key=anyllm_api_key,
                    anyllm_api_base=anyllm_api_base if anyllm_api_base else None,
                    anyllm_provider=_provider,
                    vision_model=vision_model if vision_model else "gemini/gemini-2.5-flash",
                )
            except Exception as e:
                logger.warning("Movement planning failed (non-fatal): %s", e)
        
        # Save the result as a new version
        output_path = editor_component.save_json_data(updated_json)
        
        if output_path:
            _log_io_paths("Step 9.3 Animate All — done", input_path=input_path, output_path=output_path)
            return {
                "success": True,
                "data": updated_json,
                "output_path": output_path,
                "input_path": input_path,
                "successful_count": len(result.get("successful_animations", [])),
                "failed_count": len(result.get("failed_animations", [])),
                "total_processed": result.get("total_processed", 0),
            }
        else:
            return {
                "error": "⚠️ Failed to save JSON file"
            }
        
    except Exception as e:
        return {
            "error": f"⚠️ Animation failed: {str(e)}"
        }


def show_loading_and_animate_all(editor_component, meshy_api_key, project_dir, anyllm_api_key=None, anyllm_api_base=None, anyllm_provider=None, vision_model=None):
    """Show loading indicator and perform animation for all models."""
    loading_outputs = editor_component.update_with_result(None)
    loading_state = (
        gr.update(value="🔄 **Animating all characters...** This may take a long time depending on the number of unique animations. Please wait.", visible=True),
        gr.update(visible=False),  # Hide animate_btn
    )
    
    yield loading_outputs + loading_state
    
    result = validate_and_animate_all(
        meshy_api_key, project_dir, editor_component,
        anyllm_api_key=anyllm_api_key, anyllm_api_base=anyllm_api_base,
        anyllm_provider=anyllm_provider, vision_model=vision_model,
    )
    
    if "error" in result:
        error_outputs = editor_component.update_with_result(None)
        error_state = (
            gr.update(value=result["error"], visible=True),
            gr.update(visible=True),
        )
        yield error_outputs + error_state
        return
    
    final_outputs = editor_component.update_with_result(result)
    success_msg = f"✅ Animation complete! {result.get('successful_count', 0)}/{result.get('total_processed', 0)} animations successful."
    if result.get('failed_count', 0) > 0:
        success_msg += f" ({result.get('failed_count')} failed)"
    success_msg += _io_paths_md(result)
    final_state = (
        gr.update(value=success_msg, visible=True),
        gr.update(visible=True),
    )
    
    yield final_outputs + final_state


def create_animate_all_wrapper(editor_component):
    """Factory function to create an animate all wrapper."""
    def animate_wrapper(meshy_api_key, project_dir, anyllm_api_key=None, anyllm_api_base=None, anyllm_provider=None, vision_model=None):
        for result in show_loading_and_animate_all(
            editor_component, meshy_api_key, project_dir,
            anyllm_api_key=anyllm_api_key, anyllm_api_base=anyllm_api_base,
            anyllm_provider=anyllm_provider, vision_model=vision_model,
        ):
            yield result
    return animate_wrapper


def load_scene_shot_data(project_dir):
    """Load scene and shot data from the latest animated_models or selected_animation JSON.
    
    Returns:
        tuple: (data dict, error message or None)
    """
    if not project_dir or not os.path.isabs(project_dir):
        return None, "⚠️ Please set a valid project directory first."
    
    animated_models_dir = os.path.join(project_dir, "animated_models")
    
    # Try to load animated_models first, then fall back to selected_animation
    data_path, _ = get_latest_versioned_json(animated_models_dir, "animated_models")
    if not data_path:
        data_path, _ = get_latest_versioned_json(animated_models_dir, "selected_animation")
    
    if not data_path:
        return None, "⚠️ No animation data found. Please complete Step 9.2 or 9.3 first."
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning("Step 9: path conversion failed for animation data: %s", e)
        return data, None
    except Exception as e:
        return None, f"⚠️ Failed to load data: {str(e)}"


def get_scene_choices(data):
    """Extract unique scene IDs from shot_details."""
    if not data:
        return []
    
    shot_details = data.get("shot_details", [])
    scenes = set()
    for shot in shot_details:
        scene_id = shot.get("scene_id")
        if scene_id is not None:
            scenes.add(scene_id)
    
    return sorted(list(scenes))


def get_shot_choices(data, scene_id):
    """Get shot IDs for a specific scene."""
    if not data or scene_id is None:
        return []
    
    shot_details = data.get("shot_details", [])
    shots = []
    for shot in shot_details:
        if shot.get("scene_id") == scene_id:
            shot_id = shot.get("shot_id")
            if shot_id is not None:
                shots.append(shot_id)
    
    return sorted(list(set(shots)))


def get_asset_choices(data, scene_id, shot_id):
    """Get asset IDs from character_actions for a specific scene and shot."""
    if not data or scene_id is None or shot_id is None:
        return []
    
    shot_details = data.get("shot_details", [])
    assets = []
    for shot in shot_details:
        if shot.get("scene_id") == scene_id and shot.get("shot_id") == shot_id:
            character_actions = shot.get("character_actions", [])
            for action in character_actions:
                asset_id = action.get("asset_id")
                if asset_id:
                    assets.append(asset_id)
    
    return list(set(assets))


def apply_single_animation_and_update(
    meshy_api_key,
    project_dir,
    scene_id,
    shot_id,
    asset_id,
    action_id,
    action_name
):
    """Apply a single animation and update the JSON file.
    
    Returns:
        tuple: (result dict, error message or None)
    """
    if not meshy_api_key or meshy_api_key.strip() == "":
        return None, "⚠️ Please provide a valid Meshy API key"
    
    if not all([scene_id is not None, shot_id is not None, asset_id, action_id, action_name]):
        return None, "⚠️ Please fill in all fields (Scene, Shot, Asset, Action ID, Action Name)"
    
    animated_models_dir = os.path.join(project_dir, "animated_models")
    
    # Load the latest animated_models or selected_animation JSON
    data_path, _ = get_latest_versioned_json(animated_models_dir, "animated_models")
    if not data_path:
        data_path, _ = get_latest_versioned_json(animated_models_dir, "selected_animation")
    
    if not data_path:
        return None, "⚠️ No animation data found. Please complete Step 9.2 or 9.3 first."
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning("Step 9: path conversion failed for animation data: %s", e)
    except Exception as e:
        return None, f"⚠️ Failed to load data: {str(e)}"
    
    # Find the asset in asset_sheet to get rig_task_id and texture_folder
    asset_sheet = data.get("asset_sheet", [])
    rig_task_id = None
    texture_folder = None
    asset_index = None
    asset_entry = None
    
    for idx, asset in enumerate(asset_sheet):
        if asset.get("asset_id") == asset_id:
            asset_index = idx
            asset_entry = asset
            rig_task_id = asset.get("rig_task_id")
            main_file_path = asset.get("main_file_path")
            if main_file_path:
                model_dir = os.path.dirname(main_file_path)
                model_name = os.path.splitext(os.path.basename(main_file_path))[0]
                texture_folder = os.path.join(model_dir, f"{model_name}_texture")
            break
    
    if asset_entry is None:
        return None, f"⚠️ Asset {asset_id} not found in asset_sheet."
    
    # Validate / refresh rigging — re-rig when missing or expired (per Meshy API)
    rigged_models_dir = os.path.join(project_dir, "rigged_models")
    try:
        refreshed_asset = ensure_valid_rig_task(
            asset_entry,
            output_dir=rigged_models_dir,
            meshy_api_key=meshy_api_key,
            meshy_api_base="https://api.meshy.ai/openapi/v1",
        )
    except MeshyRiggingAPIError as e:
        return None, f"⚠️ Failed to (re-)rig asset {asset_id}: {e}"
    except Exception as e:
        return None, f"⚠️ Unexpected error while validating rig for {asset_id}: {e}"
    
    if refreshed_asset is not asset_entry:
        # Re-rigging happened — update local data and persist back to disk so
        # subsequent calls reuse the new rig_task_id.
        if refreshed_asset.get("rig_error"):
            return None, f"⚠️ Re-rigging failed for {asset_id}: {refreshed_asset.get('rig_error')}"
        asset_sheet[asset_index] = refreshed_asset
        data["asset_sheet"] = asset_sheet
        rig_task_id = refreshed_asset.get("rig_task_id")
        try:
            save_data = make_paths_relative(data, project_dir)
        except Exception as _e:
            logger.warning("Step 9: path conversion failed when persisting refreshed rig info: %s", _e)
            save_data = data
        try:
            with open(data_path, "w") as _f:
                json.dump(save_data, _f, indent=2)
            logger.info("Persisted refreshed rigging info back to %s", data_path)
        except Exception as _e:
            logger.warning("Failed to persist refreshed rigging info to %s: %s", data_path, _e)
    
    if not rig_task_id:
        return None, f"⚠️ No rig_task_id available for asset: {asset_id} (rigging did not produce a task id)."
    
    # Apply the single animation
    try:
        action_id_int = int(action_id)
    except ValueError:
        return None, "⚠️ Action ID must be a number"
    
    try:
        result = apply_single_animation(
            rig_task_id=rig_task_id,
            action_id=action_id_int,
            asset_id=asset_id,
            action_name=action_name,
            output_dir=animated_models_dir,
            meshy_api_key=meshy_api_key,
            meshy_api_base="https://api.meshy.ai/openapi/v1",
            texture_folder=texture_folder,
        )
        
        if result.get("animation_error"):
            return None, f"⚠️ Animation failed: {result.get('animation_error')}"
        
        animated_path = result.get("animated_path")
        if not animated_path:
            return None, "⚠️ Animation succeeded but no file path returned"
        
    except Exception as e:
        return None, f"⚠️ Animation failed: {str(e)}"
    
    # Update the JSON with new animation info
    updated_data = deepcopy(data)
    shot_details = updated_data.get("shot_details", [])
    
    # Convert scene_id and shot_id to int for comparison
    try:
        scene_id_int = int(scene_id)
        shot_id_int = int(shot_id)
    except (ValueError, TypeError):
        scene_id_int = scene_id
        shot_id_int = shot_id
    
    updated = False
    for shot in shot_details:
        shot_scene = shot.get("scene_id")
        shot_shot = shot.get("shot_id")
        
        # Handle both int and string comparisons
        scene_match = (shot_scene == scene_id_int or shot_scene == scene_id or 
                       str(shot_scene) == str(scene_id))
        shot_match = (shot_shot == shot_id_int or shot_shot == shot_id or 
                      str(shot_shot) == str(shot_id))
        
        if scene_match and shot_match:
            character_actions = shot.get("character_actions", [])
            for action in character_actions:
                if action.get("asset_id") == asset_id:
                    action["action_id"] = action_id_int
                    action["action_name"] = action_name
                    action["animated_path"] = animated_path
                    updated = True
    
    if not updated:
        return None, f"⚠️ Could not find the character action to update (Scene {scene_id}, Shot {shot_id}, Asset {asset_id})"
    
    # Save as new version
    _, current_version = get_latest_versioned_json(animated_models_dir, "animated_models")
    new_version = current_version + 1
    output_path = os.path.join(animated_models_dir, f"animated_models_v{new_version}.json")
    
    try:
        try:
            save_data = make_paths_relative(updated_data, project_dir)
        except Exception as e:
            logger.warning("Step 9: path conversion failed on save: %s", e)
            save_data = updated_data
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
    except Exception as e:
        return None, f"⚠️ Failed to save updated JSON: {str(e)}"
    
    _log_io_paths(
        f"Apply Single Animation — done (asset={asset_id}, action={action_name})",
        input_path=data_path,
        output_path=output_path,
    )
    return {
        "success": True,
        "data": updated_data,
        "input_path": data_path,
        "output_path": output_path,
        "animated_path": animated_path,
    }, None


def get_character_action_details(data, scene_id, shot_id, asset_id):
    """Get character action details for a specific scene/shot/asset.
    
    Returns:
        dict with action_id, action_name, action_description, animated_path or None
    """
    if not data or scene_id is None or shot_id is None or not asset_id:
        return None
    
    # Convert to int for comparison
    try:
        scene_id_int = int(scene_id)
        shot_id_int = int(shot_id)
    except (ValueError, TypeError):
        scene_id_int = scene_id
        shot_id_int = shot_id
    
    shot_details = data.get("shot_details", [])
    for shot in shot_details:
        shot_scene = shot.get("scene_id")
        shot_shot = shot.get("shot_id")
        
        scene_match = (shot_scene == scene_id_int or shot_scene == scene_id or 
                       str(shot_scene) == str(scene_id))
        shot_match = (shot_shot == shot_id_int or shot_shot == shot_id or 
                      str(shot_shot) == str(shot_id))
        
        if scene_match and shot_match:
            character_actions = shot.get("character_actions", [])
            for action in character_actions:
                if action.get("asset_id") == asset_id:
                    # Prefer animated_path (from single animation) over animated_path (from batch)
                    animated_path = action.get("animated_path") or action.get("animated_path")
                    return {
                        "action_id": action.get("action_id"),
                        "action_name": action.get("action_name"),
                        "action_description": action.get("action_description"),
                        "animated_path": animated_path,
                    }
    
    return None


# =========================================================================
# Auto Route pipeline helper functions
# =========================================================================

def validate_and_auto_route(
    meshy_api_key, uthana_api_key, uthana_fps,
    anyllm_api_key, anyllm_api_base, anyllm_provider, vision_model, reasoning_model,
    project_dir, editor_component,
):
    """Run the Auto Route hybrid pipeline (Meshy + Uthana fallback)."""
    if not project_dir or not os.path.isabs(project_dir):
        return {"error": "⚠️ Project directory must be an absolute path"}
    if not meshy_api_key or meshy_api_key.strip() == "":
        return {"error": "⚠️ Please provide a valid Meshy API key"}
    if not uthana_api_key or uthana_api_key.strip() == "":
        return {"error": "⚠️ Please provide a valid Uthana API key"}
    if not anyllm_api_key or anyllm_api_key.strip() == "":
        return {"error": "⚠️ Please provide a valid AnyLLM API key"}

    # Find input: rigged_model.json — auto-rig if it doesn't exist yet
    rigged_models_dir = os.path.join(project_dir, "rigged_models")
    input_path = os.path.join(rigged_models_dir, "rigged_model.json")
    if not os.path.exists(input_path):
        print("Auto Route: rigged_model.json not found — running rigging first…")
        # Find the latest layout_script
        layout_script_dir = os.path.join(project_dir, "layout_script")
        if not os.path.exists(layout_script_dir):
            return {"error": "⚠️ No layout_script folder found. Please complete previous steps first."}
        latest_version = 0
        layout_path = None
        for fn in os.listdir(layout_script_dir):
            if fn.startswith("layout_script_v") and fn.endswith(".json"):
                try:
                    v = int(fn[len("layout_script_v"):-5])
                    if v > latest_version:
                        latest_version = v
                        layout_path = os.path.join(layout_script_dir, fn)
                except ValueError:
                    continue
        if layout_path is None:
            return {"error": "⚠️ No layout_script found. Please complete previous steps first."}
        try:
            os.makedirs(rigged_models_dir, exist_ok=True)
            rig_result = rig_models(
                path_to_input_json=layout_path,
                output_dir=rigged_models_dir,
                meshy_api_key=meshy_api_key,
            )
            with open(input_path, "w") as f:
                json.dump(rig_result, f, indent=2)
            print(f"Auto Route: rigging complete → {input_path}")
        except Exception as e:
            return {"error": f"⚠️ Auto-rigging failed: {str(e)}"}

    animated_models_dir = os.path.join(project_dir, "animated_models")
    os.makedirs(animated_models_dir, exist_ok=True)
    editor_component.set_save_path(animated_models_dir)

    _log_io_paths("Auto Route — starting", input_path=input_path)
    try:
        with open(input_path, "r") as f:
            storyboard_script = json.load(f)
        try:
            storyboard_script = make_paths_absolute(storyboard_script, project_dir)
        except Exception as e:
            logger.warning("Auto route: path conversion failed: %s", e)

        fps = int(uthana_fps) if uthana_fps else 30
        _provider = anyllm_provider.strip() if anyllm_provider else "gemini"
        if not _provider:
            _provider = "gemini"
        result = auto_route_animate(
            storyboard_script=storyboard_script,
            output_dir=animated_models_dir,
            meshy_api_key=meshy_api_key,
            uthana_api_key=uthana_api_key,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base if anyllm_api_base else None,
            anyllm_provider=_provider,
            vision_model=vision_model if vision_model else "gemini/gemini-2.5-flash",
            reasoning_model=reasoning_model if reasoning_model else "gemini-3.1-pro-preview",
            uthana_fps=fps,
        )

        updated_json = result.get("updated_json")
        if updated_json is None:
            return {"error": "⚠️ Auto Route failed. No animations produced."}

        output_path = editor_component.save_json_data(updated_json)
        if output_path:
            _log_io_paths("Auto Route — done", input_path=input_path, output_path=output_path)
            return {
                "success": True,
                "data": updated_json,
                "output_path": output_path,
                "input_path": input_path,
                "successful_count": len(result.get("successful_animations", [])),
                "failed_count": len(result.get("failed_animations", [])),
                "total_processed": result.get("total_processed", 0),
            }
        return {"error": "⚠️ Failed to save JSON file"}
    except Exception as e:
        return {"error": f"⚠️ Auto Route failed: {str(e)}"}


def show_loading_and_auto_route(editor_component, meshy_api_key, uthana_api_key, uthana_fps,
                                anyllm_api_key, anyllm_api_base, anyllm_provider, vision_model, reasoning_model, project_dir):
    loading_outputs = editor_component.update_with_result(None)
    yield loading_outputs + (
        gr.update(value=(
            "🔄 **Auto Route running…**\n\n"
            "0️⃣ Rigging characters (if not already done)…\n"
            "1️⃣ Selecting from Meshy library…\n"
            "2️⃣ Animating matched actions with Meshy…\n"
            "3️⃣ Planning & animating unmatched actions with Uthana…\n\n"
            "This may take a long time. Please wait."
        ), visible=True),
        gr.update(visible=False),
    )

    result = validate_and_auto_route(
        meshy_api_key, uthana_api_key, uthana_fps,
        anyllm_api_key, anyllm_api_base, anyllm_provider, vision_model, reasoning_model,
        project_dir, editor_component,
    )

    if "error" in result:
        yield editor_component.update_with_result(None) + (
            gr.update(value=result["error"], visible=True),
            gr.update(visible=True),
        )
        return

    msg = (f"✅ Auto Route complete! "
           f"{result.get('successful_count', 0)} Meshy + Uthana fallback animations. "
           f"{result.get('failed_count', 0)} failed.") + _io_paths_md(result)
    yield editor_component.update_with_result(result) + (
        gr.update(value=msg, visible=True),
        gr.update(visible=True),
    )


def create_auto_route_wrapper(editor_component):
    def wrapper(meshy_api_key, uthana_api_key, uthana_fps,
                anyllm_api_key, anyllm_api_base, anyllm_provider, vision_model, reasoning_model, project_dir):
        for r in show_loading_and_auto_route(
            editor_component, meshy_api_key, uthana_api_key, uthana_fps,
            anyllm_api_key, anyllm_api_base, anyllm_provider, vision_model, reasoning_model, project_dir,
        ):
            yield r
    return wrapper


def validate_and_retry_failed_animations(
    meshy_api_key, uthana_api_key, uthana_fps, project_dir, editor_component,
):
    """Retry animation tasks (meshy / meshy_idle / uthana) whose
    ``animated_path`` is missing in the latest ``animated_models_v{N}.json``.
    """
    if not project_dir or not os.path.isabs(project_dir):
        return {"error": "⚠️ Project directory must be an absolute path"}
    if not meshy_api_key or meshy_api_key.strip() == "":
        return {"error": "⚠️ Please provide a valid Meshy API key"}
    # Uthana key only required if there are uthana failures, but request anyway
    # since we cannot know without first scanning.

    animated_models_dir = os.path.join(project_dir, "animated_models")
    latest_path, latest_v = _find_latest_animated_models(animated_models_dir)
    if not latest_path:
        return {"error": "⚠️ No animated_models_v*.json found. Run Auto Route first."}

    try:
        with open(latest_path, "r") as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning("Retry: path conversion failed: %s", e)
    except Exception as e:
        return {"error": f"⚠️ Failed to read {latest_path}: {str(e)}"}

    # Quick scan to check whether any uthana retries are needed
    needs_uthana = False
    for shot in data.get("shot_details", []):
        for a in shot.get("character_actions", []) or []:
            if not a.get("animated_path") and a.get("pipeline") == "uthana":
                needs_uthana = True
                break
        if needs_uthana:
            break
    if needs_uthana and (not uthana_api_key or uthana_api_key.strip() == ""):
        return {"error": "⚠️ Uthana API key required to retry uthana animations"}

    os.makedirs(animated_models_dir, exist_ok=True)
    editor_component.set_save_path(animated_models_dir)

    _log_io_paths("Retry Failed Animations — starting", input_path=latest_path)
    try:
        fps = int(uthana_fps) if uthana_fps else 30
        result = retry_failed_animations(
            storyboard_data=data,
            output_dir=animated_models_dir,
            meshy_api_key=meshy_api_key,
            uthana_api_key=uthana_api_key or "",
            uthana_fps=fps,
        )
    except Exception as e:
        return {"error": f"⚠️ Retry failed: {str(e)}"}

    n_before = result.get("n_meshy_failed_before", 0) + result.get("n_uthana_failed_before", 0)
    if n_before == 0:
        return {"error": "✅ No failed animations found in the latest animated_models JSON. Nothing to retry."}

    updated_json = result.get("updated_json")
    if updated_json is None:
        return {"error": "⚠️ Retry produced no updated JSON"}

    output_path = editor_component.save_json_data(updated_json)
    if not output_path:
        return {"error": "⚠️ Failed to save retry result JSON"}

    _log_io_paths("Retry Failed Animations — done", input_path=latest_path, output_path=output_path)
    return {
        "success": True,
        "data": updated_json,
        "output_path": output_path,
        "input_path": latest_path,
        "retried_meshy": result.get("retried_meshy", 0),
        "retried_uthana": result.get("retried_uthana", 0),
        "n_meshy_failed_before": result.get("n_meshy_failed_before", 0),
        "n_uthana_failed_before": result.get("n_uthana_failed_before", 0),
        "still_failed": result.get("still_failed", 0),
    }


def show_loading_and_retry_failed(editor_component, meshy_api_key, uthana_api_key,
                                  uthana_fps, project_dir):
    loading_outputs = editor_component.update_with_result(None)
    yield loading_outputs + (
        gr.update(value=(
            "🔄 **Retrying failed animations…**\n\n"
            "Scanning latest `animated_models_v{N}.json` for actions without "
            "`animated_path` and re-running them through their original pipeline."
        ), visible=True),
        gr.update(visible=False),
    )

    result = validate_and_retry_failed_animations(
        meshy_api_key, uthana_api_key, uthana_fps, project_dir, editor_component,
    )

    if "error" in result:
        yield editor_component.update_with_result(None) + (
            gr.update(value=result["error"], visible=True),
            gr.update(visible=True),
        )
        return

    msg = (
        f"✅ Retry complete! "
        f"Meshy/idle: {result.get('retried_meshy', 0)}/"
        f"{result.get('n_meshy_failed_before', 0)} recovered. "
        f"Uthana: {result.get('retried_uthana', 0)}/"
        f"{result.get('n_uthana_failed_before', 0)} recovered. "
        f"{result.get('still_failed', 0)} still failed."
    ) + _io_paths_md(result)
    yield editor_component.update_with_result(result) + (
        gr.update(value=msg, visible=True),
        gr.update(visible=True),
    )


def create_retry_failed_wrapper(editor_component):
    def wrapper(meshy_api_key, uthana_api_key, uthana_fps, project_dir):
        for r in show_loading_and_retry_failed(
            editor_component, meshy_api_key, uthana_api_key, uthana_fps, project_dir,
        ):
            yield r
    return wrapper


# =========================================================================
# Uthana pipeline helper functions
# =========================================================================

def validate_and_plan_animations(
    anyllm_api_key, anyllm_api_base, anyllm_provider, reasoning_model, project_dir, editor_component
):
    """Validate inputs and generate animation plans for the Uthana pipeline."""
    if not project_dir or not os.path.isabs(project_dir):
        return {"error": "⚠️ Project directory must be an absolute path"}
    if not anyllm_api_key or anyllm_api_key.strip() == "":
        return {"error": "⚠️ Please provide a valid AnyLLM API key"}

    # Find the latest layout_script
    layout_script_dir = os.path.join(project_dir, "layout_script")
    if not os.path.exists(layout_script_dir):
        return {"error": "⚠️ No layout_script folder found. Please complete previous steps first."}

    latest_version = 0
    input_path = None
    for fn in os.listdir(layout_script_dir):
        if fn.startswith("layout_script_v") and fn.endswith(".json"):
            try:
                v = int(fn[len("layout_script_v"):-5])
                if v > latest_version:
                    latest_version = v
                    input_path = os.path.join(layout_script_dir, fn)
            except ValueError:
                continue

    if input_path is None:
        return {"error": "⚠️ No layout_script found. Please complete previous steps first."}

    animated_models_dir = os.path.join(project_dir, "animated_models")
    os.makedirs(animated_models_dir, exist_ok=True)
    editor_component.set_save_path(animated_models_dir)

    # Skip planning if a plan already exists from a previous run
    existing_plan_path, _ = get_latest_versioned_json(animated_models_dir, "animation_plan")
    if existing_plan_path and os.path.exists(existing_plan_path):
        try:
            with open(existing_plan_path, 'r') as f:
                existing_plan = json.load(f)
            try:
                existing_plan = make_paths_absolute(existing_plan, project_dir)
            except Exception as e:
                logger.warning("Step 9 Uthana: path conversion failed for existing plan: %s", e)
            logger.info("Step 9 Uthana: reusing existing animation plan %s", existing_plan_path)
            return {
                "success": True,
                "data": existing_plan,
                "output_path": existing_plan_path,
                "reused": True,
            }
        except Exception as e:
            logger.warning("Step 9 Uthana: failed to load existing plan (%s); regenerating", e)

    try:
        with open(input_path, 'r') as f:
            storyboard_script = json.load(f)
        try:
            storyboard_script = make_paths_absolute(storyboard_script, project_dir)
        except Exception as e:
            logger.warning("Step 9 Uthana: path conversion failed: %s", e)

        _provider = anyllm_provider.strip() if anyllm_provider else "gemini"
        if not _provider:
            _provider = "gemini"
        result_data = generate_animation_plan(
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base if anyllm_api_base else None,
            anyllm_provider=_provider,
            reasoning_model=reasoning_model if reasoning_model else "gemini-3.1-pro-preview",
            storyboard_script=storyboard_script,
            max_retries=3,
        )

        if result_data is None:
            return {"error": "⚠️ Animation planning failed. Check console for details."}

        output_path = editor_component.save_json_data(result_data)
        if output_path:
            return {"success": True, "data": result_data, "output_path": output_path}
        return {"error": "⚠️ Failed to save JSON file"}
    except Exception as e:
        return {"error": f"⚠️ Animation planning failed: {str(e)}"}


def show_loading_and_plan_animations(editor_component, anyllm_api_key, anyllm_api_base, anyllm_provider, reasoning_model, project_dir):
    """Show loading indicator and run animation planning."""
    loading_outputs = editor_component.update_with_result(None)
    yield loading_outputs + (
        gr.update(value="🔄 **Planning animations…** This may take a few minutes. Please wait.", visible=True),
        gr.update(visible=False),
    )

    result = validate_and_plan_animations(anyllm_api_key, anyllm_api_base, anyllm_provider, reasoning_model, project_dir, editor_component)

    if "error" in result:
        yield editor_component.update_with_result(None) + (
            gr.update(value=result["error"], visible=True),
            gr.update(visible=True),
        )
        return

    if result.get("reused"):
        success_msg = (
            "✅ **Reusing existing animation plan from a previous run.** "
            f"Loaded `{os.path.basename(result.get('output_path', ''))}`. "
            "Delete the file in `animated_models/` if you want to regenerate."
        )
    else:
        success_msg = "✅ **Animation planning completed successfully!** You can proceed to the animation step."
    success_msg += _io_paths_md(result)
    yield editor_component.update_with_result(result) + (
        gr.update(value=success_msg, visible=True),
        gr.update(visible=True),
    )


def create_plan_animations_wrapper(editor_component):
    def wrapper(anyllm_api_key, anyllm_api_base, anyllm_provider, reasoning_model, project_dir):
        for r in show_loading_and_plan_animations(editor_component, anyllm_api_key, anyllm_api_base, anyllm_provider, reasoning_model, project_dir):
            yield r
    return wrapper


def validate_and_uthana_animate_all(
    uthana_api_key, uthana_fps, project_dir, editor_component
):
    """Validate inputs and animate all characters via Uthana."""
    if not project_dir or not os.path.isabs(project_dir):
        return {"error": "⚠️ Project directory must be an absolute path"}
    if not uthana_api_key or uthana_api_key.strip() == "":
        return {"error": "⚠️ Please provide a valid Uthana API key"}

    animated_models_dir = os.path.join(project_dir, "animated_models")

    # Find the latest animation_plan JSON (output of the planning step)
    input_path, _ = get_latest_versioned_json(animated_models_dir, "animation_plan")
    if not input_path:
        return {"error": "⚠️ No animation_plan JSON found. Please complete the 'Plan Animations' step first."}

    os.makedirs(animated_models_dir, exist_ok=True)
    editor_component.set_save_path(animated_models_dir)

    _log_io_paths("Step 9.B Uthana Animate All — starting", input_path=input_path)
    try:
        fps = int(uthana_fps) if uthana_fps else 30
        result = uthana_animate_characters(
            path_to_input_json=input_path,
            output_dir=animated_models_dir,
            uthana_api_key=uthana_api_key,
            fps=fps,
            motion_model="text-to-motion-bucmd",
            max_retries=3,
        )

        updated_json = result.get("updated_json")
        if updated_json is None:
            return {"error": "⚠️ Uthana animation failed. No animations were successful."}

        output_path = editor_component.save_json_data(updated_json)
        if output_path:
            _log_io_paths("Step 9.B Uthana Animate All — done", input_path=input_path, output_path=output_path)
            return {
                "success": True,
                "data": updated_json,
                "output_path": output_path,
                "input_path": input_path,
                "successful_count": len(result.get("successful_animations", [])),
                "failed_count": len(result.get("failed_animations", [])),
                "total_processed": result.get("total_processed", 0),
            }
        return {"error": "⚠️ Failed to save JSON file"}
    except Exception as e:
        return {"error": f"⚠️ Uthana animation failed: {str(e)}"}


def show_loading_and_uthana_animate_all(editor_component, uthana_api_key, uthana_fps, project_dir):
    """Show loading indicator and animate all characters with Uthana."""
    loading_outputs = editor_component.update_with_result(None)
    yield loading_outputs + (
        gr.update(value="🔄 **Animating with Uthana…** Uploading models, generating motions, and downloading results. This may take a while.", visible=True),
        gr.update(visible=False),
    )

    result = validate_and_uthana_animate_all(uthana_api_key, uthana_fps, project_dir, editor_component)

    if "error" in result:
        yield editor_component.update_with_result(None) + (
            gr.update(value=result["error"], visible=True),
            gr.update(visible=True),
        )
        return

    msg = f"✅ Uthana animation complete! {result.get('successful_count', 0)}/{result.get('total_processed', 0)} successful."
    if result.get("failed_count", 0) > 0:
        msg += f" ({result['failed_count']} failed)"
    msg += _io_paths_md(result)
    yield editor_component.update_with_result(result) + (
        gr.update(value=msg, visible=True),
        gr.update(visible=True),
    )


def create_uthana_animate_all_wrapper(editor_component):
    def wrapper(uthana_api_key, uthana_fps, project_dir):
        for r in show_loading_and_uthana_animate_all(editor_component, uthana_api_key, uthana_fps, project_dir):
            yield r
    return wrapper


# =========================================================================
# Main UI builder
# =========================================================================

def create_animator_rigging_ui(meshy_api_key, project_dir, anyllm_api_key, anyllm_api_base, anyllm_provider, vision_model, reasoning_model, blender_client, uthana_api_key=None, uthana_fps=None):
    """Create the Step 9: Rigging and Animate Assets by Animator UI section.
    
    Args:
        meshy_api_key: Gradio component for Meshy API key
        project_dir: Gradio component for project directory
        anyllm_api_key: Gradio component for any-llm API key
        anyllm_api_base: Gradio component for any-llm API base URL
        vision_model: Gradio component for vision model
        blender_client: BlenderClient instance for Blender operations
    
    Returns:
        dict with UI components that may be needed by other parts of the app
    """
    gr.Markdown("## Step 9: Rigging and Animate Assets by Animator")
    gr.Markdown("Choose a pipeline: **Auto Route** (recommended — Meshy + Uthana fallback), **Meshy** only, or **Uthana** only. The **Import & Preview** tab works with all.")

    with gr.Tabs():
      # ==================================================================
      # TAB 0 — Auto Route Pipeline (Meshy + Uthana)
      # ==================================================================
      with gr.Tab("Auto Route Pipeline"):
        gr.Markdown(
            "### Auto Route: Best of Both Worlds\n\n"
            "1. **Meshy first** — selects from the animation library. Well-matched actions use Meshy.\n"
            "2. **Uthana fallback** — unmatched actions get open-vocabulary text-to-motion via Uthana.\n"
            "3. **Movement planning** — walking/running animations get `end_coords` + `start_rotation`.\n\n"
            "Requires: Meshy API key, Uthana API key, AnyLLM API key, and **Rigging (Step 9.1)** completed first."
        )

        with gr.Row():
            auto_route_btn = gr.Button("🚀 Run Auto Route Pipeline", variant="primary", size="lg")
            retry_failed_btn = gr.Button("♻️ Retry Failed Animation Tasks", variant="secondary", size="lg")
        auto_route_status = gr.Markdown(value="", visible=False)

        auto_route_editor = JSONEditorComponent(
            label="Auto Route Result JSON",
            visible_initially=False,
            file_basename="animated_models",
            use_version_control=True,
            json_root_keys_list=["asset_sheet"],
            title="Auto Route",
        )
        auto_route_editor.setup_resume_with_project_dir(project_dir, subfolder="animated_models")

        auto_route_wrapper_fn = create_auto_route_wrapper(auto_route_editor)
        retry_failed_wrapper_fn = create_retry_failed_wrapper(auto_route_editor)

        if uthana_api_key is not None and uthana_fps is not None:
            auto_route_btn.click(
                fn=auto_route_wrapper_fn,
                inputs=[meshy_api_key, uthana_api_key, uthana_fps,
                        anyllm_api_key, anyllm_api_base, anyllm_provider, vision_model, reasoning_model, project_dir],
                outputs=auto_route_editor.get_output_components() + [auto_route_status, auto_route_btn],
            )
            retry_failed_btn.click(
                fn=retry_failed_wrapper_fn,
                inputs=[meshy_api_key, uthana_api_key, uthana_fps, project_dir],
                outputs=auto_route_editor.get_output_components() + [auto_route_status, retry_failed_btn],
            )
        else:
            auto_route_btn.interactive = False
            retry_failed_btn.interactive = False

      # ==================================================================
      # TAB 1 — Meshy Pipeline
      # ==================================================================
      with gr.Tab("Meshy Pipeline"):
        gr.Markdown("### Step 9.1: Rigging Characters with Animator")
        gr.Markdown("This step rigs all character models using the Meshy Rigging API. The rigged models will be saved in the `rigged_models` folder.")
    
        # Action buttons
        rig_btn = gr.Button("🦴 Rigging Characters", variant="primary", size="lg")
        loading_status = gr.Markdown(value="", visible=False)

        rigging_editor = JSONEditorComponent(
            label="Rigging Result JSON",
            visible_initially=False,
            file_basename="rigged_model",
            use_version_control=False,
            json_root_keys_list=["asset_sheet"],
            title="Step 9.1"
        )
        rigging_editor.setup_resume_with_project_dir(project_dir, subfolder="rigged_models")
        rig_wrapper = create_rig_wrapper(rigging_editor)
        rig_btn.click(
            fn=rig_wrapper,
            inputs=[meshy_api_key, project_dir],
            outputs=rigging_editor.get_output_components() + [loading_status, rig_btn],
        )

        # --- Rigged Model Viewer ---
        gr.Markdown("### Rigged Model Viewer")
        gr.Markdown("Preview rigged character models with running animations.")
        with gr.Row():
            display_models_btn = gr.Button("🔍 Display Rigged Models", variant="secondary", size="lg", scale=1)
            toggle_viewer_btn = gr.Button("👁 Hide/Show 3D Viewer", variant="secondary", size="lg", scale=1)
        model_status = gr.Markdown(value="", visible=False)
        models_state = gr.State([])
        viewer_visible = gr.State(False)
        with gr.Column(visible=False) as model_viewer_container:
            model_buttons = gr.Dataset(
                components=[gr.Textbox(visible=False)],
                label="Select Rigged Model",
                samples=[], samples_per_page=20, visible=False
            )
            model_viewer = gr.Model3D(label="Rigged Model Preview (Running Animation)", clear_color=(0.9, 0.9, 0.9, 1.0), height=500)

        def toggle_viewer(is_visible):
            new_visible = not is_visible
            return gr.update(visible=new_visible), gr.update(visible=new_visible), new_visible
        def display_and_show(project_dir):
            return display_rigged_models(project_dir) + (True,)
        display_models_btn.click(fn=display_and_show, inputs=[project_dir], outputs=[model_viewer, model_status, model_buttons, model_viewer_container, models_state, viewer_visible])
        toggle_viewer_btn.click(fn=toggle_viewer, inputs=[viewer_visible], outputs=[model_viewer_container, model_status, viewer_visible])
        model_buttons.select(fn=select_rigged_model, inputs=[models_state], outputs=[model_viewer, model_status])

        # --- Merge Rigging Task Information ---
        gr.Markdown("### Merge Rigging Task Information")
        gr.Markdown("Only necessary if you need to re-rig assets (rig tasks expire). Merges rig info from `rigged_model.json` into `selected_animation` JSON.")
        merge_rig_btn = gr.Button("🔗 Merge Rigging Task Information", variant="secondary", size="lg")
        merge_rig_status = gr.Markdown(value="", visible=False)
        def handle_merge_rig_info(project_dir):
            success_msg, error_msg, output_path = merge_rigging_task_info(project_dir)
            if error_msg:
                return gr.update(value=error_msg, visible=True)
            return gr.update(value=success_msg, visible=True)
        merge_rig_btn.click(fn=handle_merge_rig_info, inputs=[project_dir], outputs=[merge_rig_status])

        # ---- Step 9.2: Select Animation ----
        gr.Markdown("### Step 9.2: Select Animation with Animator")
        gr.Markdown("Analyzes character actions and selects appropriate animations from the animation library using AI vision analysis.")
        select_anim_btn = gr.Button("🎬 Select Animations", variant="primary", size="lg")
        anim_loading_status = gr.Markdown(value="", visible=False)
        animation_editor = JSONEditorComponent(
            label="Animation Selection Result JSON",
            visible_initially=False,
            file_basename="selected_animation",
            use_version_control=True,
            json_root_keys_list=["asset_sheet"],
            title="Step 9.2"
        )
        animation_editor.setup_resume_with_project_dir(project_dir, subfolder="animated_models")
        select_wrapper = create_select_animations_wrapper(animation_editor)
        select_anim_btn.click(
            fn=select_wrapper,
            inputs=[anyllm_api_key, anyllm_api_base, anyllm_provider, vision_model, project_dir],
            outputs=animation_editor.get_output_components() + [anim_loading_status, select_anim_btn],
        )

        # ---- Step 9.3: Animate Rigged Model ----
        gr.Markdown("### Step 9.3: Animate Rigged Model with Animators")
        gr.Markdown("Apply animations to rigged models using the Meshy Animation API.")
        gr.Markdown("#### Animate All Characters")
        animate_all_btn = gr.Button("🎭 Animate All Characters in All Shots", variant="primary", size="lg")
        animate_loading_status = gr.Markdown(value="", visible=False)
        animated_models_editor = JSONEditorComponent(
            label="Animated Models Result JSON",
            visible_initially=False,
            file_basename="animated_models",
            use_version_control=True,
            json_root_keys_list=["asset_sheet"],
            title="Step 9.3"
        )
        animated_models_editor.setup_resume_with_project_dir(project_dir, subfolder="animated_models")
        animate_all_wrapper = create_animate_all_wrapper(animated_models_editor)
        animate_all_btn.click(
            fn=animate_all_wrapper,
            inputs=[meshy_api_key, project_dir, anyllm_api_key, anyllm_api_base, anyllm_provider, vision_model],
            outputs=animated_models_editor.get_output_components() + [animate_loading_status, animate_all_btn],
        )

        # --- Edit Animation for Specific Character ---
        gr.Markdown("#### Edit Animation for Specific Character in a Shot")
        gr.Markdown("Select a specific character action to apply a custom animation.")
        edit_data_state = gr.State(None)
        load_edit_data_btn = gr.Button("🔄 Load Animation Data", variant="secondary", size="lg")
        edit_status = gr.Markdown(value="", visible=False)
        with gr.Column(visible=False) as edit_container:
            gr.Markdown("**Step 1: Select Scene**")
            scene_dropdown = gr.Dropdown(choices=[], label="Scene", interactive=True)
            gr.Markdown("**Step 2: Select Shot**")
            shot_dropdown = gr.Dropdown(choices=[], label="Shot", interactive=True)
            gr.Markdown("**Step 3: Select Character (Asset ID)**")
            asset_dropdown = gr.Dropdown(choices=[], label="Asset ID", interactive=True)
            gr.Markdown("**Step 4: Enter Animation Details**")
            with gr.Row():
                action_id_input = gr.Textbox(label="Action ID", placeholder="Enter action ID (number)", scale=1)
                action_name_input = gr.Textbox(label="Action Name", placeholder="Enter action name", scale=2)
            apply_single_btn = gr.Button("✨ Apply Animation for this Character in this Shot", variant="primary", size="lg")
            single_anim_status = gr.Markdown(value="", visible=False)
            hide_edit_btn = gr.Button("🙈 Hide Animation Editor", variant="secondary", size="lg")

        def hide_edit_container():
            return (
                gr.update(value="", visible=False), gr.update(visible=False), None,
                gr.update(choices=[], value=None), gr.update(choices=[], value=None),
                gr.update(choices=[], value=None), gr.update(value="", visible=False),
            )
        hide_edit_btn.click(fn=hide_edit_container, inputs=[], outputs=[edit_status, edit_container, edit_data_state, scene_dropdown, shot_dropdown, asset_dropdown, single_anim_status])

        def load_edit_data_handler(project_dir):
            data, error = load_scene_shot_data(project_dir)
            if error:
                return (gr.update(value=error, visible=True), gr.update(visible=False), None, gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(choices=[], value=None))
            scenes = get_scene_choices(data)
            return (gr.update(value="✅ Data loaded. Select a scene to begin.", visible=True), gr.update(visible=True), data, gr.update(choices=[str(s) for s in scenes], value=None), gr.update(choices=[], value=None), gr.update(choices=[], value=None))
        load_edit_data_btn.click(fn=load_edit_data_handler, inputs=[project_dir], outputs=[edit_status, edit_container, edit_data_state, scene_dropdown, shot_dropdown, asset_dropdown])

        def on_scene_change(scene_id, data):
            if not scene_id or not data:
                return gr.update(choices=[], value=None), gr.update(choices=[], value=None)
            try: scene_id_int = int(scene_id)
            except ValueError: scene_id_int = scene_id
            return gr.update(choices=[str(s) for s in get_shot_choices(data, scene_id_int)], value=None), gr.update(choices=[], value=None)
        scene_dropdown.change(fn=on_scene_change, inputs=[scene_dropdown, edit_data_state], outputs=[shot_dropdown, asset_dropdown])

        def on_shot_change(scene_id, shot_id, data):
            if not scene_id or not shot_id or not data:
                return gr.update(choices=[], value=None)
            try: scene_id_int, shot_id_int = int(scene_id), int(shot_id)
            except ValueError: scene_id_int, shot_id_int = scene_id, shot_id
            return gr.update(choices=get_asset_choices(data, scene_id_int, shot_id_int), value=None)
        shot_dropdown.change(fn=on_shot_change, inputs=[scene_dropdown, shot_dropdown, edit_data_state], outputs=[asset_dropdown])

        def apply_single_handler_with_reload(meshy_api_key, project_dir, scene_id, shot_id, asset_id, action_id, action_name):
            editor_loading = animated_models_editor.update_with_result(None)
            yield editor_loading + (gr.update(value="🔄 **Applying animation...** This may take several minutes.", visible=True), gr.update(visible=False))
            result, error = apply_single_animation_and_update(meshy_api_key, project_dir, scene_id, shot_id, asset_id, action_id, action_name)
            if error:
                yield animated_models_editor.update_with_result(None) + (gr.update(value=error, visible=True), gr.update(visible=True))
                return
            editor_outputs = animated_models_editor.update_with_result(result)
            yield editor_outputs + (gr.update(value=(
                "✅ **Animation applied!**"
                f"\n\n📂 **Input JSON:** `{result.get('input_path')}`"
                f"\n\n💾 **Output JSON:** `{result.get('output_path')}`"
                f"\n\n🎬 **Animated GLB:** `{result.get('animated_path')}`"
            ), visible=True), gr.update(visible=True))
        apply_single_btn.click(
            fn=apply_single_handler_with_reload,
            inputs=[meshy_api_key, project_dir, scene_dropdown, shot_dropdown, asset_dropdown, action_id_input, action_name_input],
            outputs=animated_models_editor.get_output_components() + [single_anim_status, apply_single_btn]
        )

      # ==================================================================
      # TAB 2 — Uthana Pipeline
      # ==================================================================
      with gr.Tab("Uthana Pipeline"):
        gr.Markdown(
            "Uthana supports **open-vocabulary text-to-motion** animation — describe any motion in natural language. "
            "Requires an Uthana API key (set in Configuration). "
            "Models must be under 30 MB; larger files are automatically compressed."
        )

        # ---- Step 9.A: Plan Animations (LLM) ----
        gr.Markdown("### Step 9.A: Plan Animations (LLM)")
        gr.Markdown("Use an LLM to generate text-to-motion prompts, durations, and movement coordinates for every character in every shot.")
        plan_btn = gr.Button("📝 Plan Animations", variant="primary", size="lg")
        plan_status = gr.Markdown(value="", visible=False)
        plan_editor = JSONEditorComponent(
            label="Animation Plan JSON",
            visible_initially=False,
            file_basename="animation_plan",
            use_version_control=True,
            json_root_keys_list=["shot_details"],
            title="Step 9.A",
        )
        plan_editor.setup_resume_with_project_dir(project_dir, subfolder="animated_models")
        plan_wrapper = create_plan_animations_wrapper(plan_editor)
        plan_btn.click(
            fn=plan_wrapper,
            inputs=[anyllm_api_key, anyllm_api_base, anyllm_provider, reasoning_model, project_dir],
            outputs=plan_editor.get_output_components() + [plan_status, plan_btn],
        )

        # ---- Step 9.B: Rig & Animate Characters (Uthana API) ----
        gr.Markdown("### Step 9.B: Rig & Animate Characters (Uthana)")
        gr.Markdown("Upload character models to Uthana for auto-rigging, generate text-to-motion animations, and download animated GLBs.")
        uthana_animate_btn = gr.Button("🎭 Animate All Characters (Uthana)", variant="primary", size="lg")
        uthana_animate_status = gr.Markdown(value="", visible=False)
        uthana_animated_editor = JSONEditorComponent(
            label="Uthana Animated Models JSON",
            visible_initially=False,
            file_basename="animated_models",
            use_version_control=True,
            json_root_keys_list=["asset_sheet"],
            title="Step 9.B",
        )
        uthana_animated_editor.setup_resume_with_project_dir(project_dir, subfolder="animated_models")
        uthana_animate_wrapper = create_uthana_animate_all_wrapper(uthana_animated_editor)
        if uthana_api_key is not None and uthana_fps is not None:
            uthana_animate_btn.click(
                fn=uthana_animate_wrapper,
                inputs=[uthana_api_key, uthana_fps, project_dir],
                outputs=uthana_animated_editor.get_output_components() + [uthana_animate_status, uthana_animate_btn],
            )
        else:
            uthana_animate_btn.interactive = False

      # ==================================================================
      # TAB 3 — Import & Preview
      # ==================================================================
      with gr.Tab("Import & Preview"):
        gr.Markdown("### Import Animated Characters to All Shots")
        gr.Markdown("Import animated character models into Blender scenes for each shot. Works with both **Meshy** and **Uthana** animated models.")
        with gr.Row():
            import_to_shots_btn = gr.Button("📥 Import Animated Characters to All Shots", variant="primary", size="lg", scale=2)
            delete_all_shots_btn = gr.Button("🗑️ Delete All Shots", variant="stop", size="lg", scale=1)
        import_status = gr.Markdown(value="", visible=False)

        def import_animated_characters_handler(project_dir):
            if not project_dir or not os.path.isabs(project_dir):
                return gr.update(value="⚠️ Please set a valid project directory.", visible=True)
            animated_models_dir = os.path.join(project_dir, "animated_models")
            json_path, version = get_latest_versioned_json(animated_models_dir, "animated_models")
            if not json_path:
                return gr.update(value="⚠️ No animated_models JSON found. Please complete animation steps first.", visible=True)
            _log_io_paths("Import Animated Characters to All Shots — starting", input_path=json_path)
            yield gr.update(value=f"🔄 **Importing animated characters from:** `{json_path}`\n\nThis may take a while...", visible=True)
            temp_input_path = None
            try:
                with open(json_path, 'r') as f:
                    input_data = json.load(f)
                input_data = make_paths_absolute(input_data, project_dir)
                temp_input_path = json_path + ".tmp"
                with open(temp_input_path, 'w') as f:
                    json.dump(input_data, f, indent=2)
            except Exception as e:
                if temp_input_path and os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                yield gr.update(value=f"⚠️ Failed to prepare input for Blender: {str(e)}", visible=True)
                return
            try:
                response = blender_client.import_animated_assets_to_all_shots_json_input(json_filepath=temp_input_path)
                if response.get("status") == "error":
                    yield gr.update(value=f"⚠️ Import failed: {response.get('message', 'Unknown error')}", visible=True)
                    return
                result = response.get("result", response)
                if result.get("success"):
                    successful_shots = result.get("successful_shots", [])
                    shots_processed = len(successful_shots) if isinstance(successful_shots, list) else result.get("shots_processed", 0)
                    message = f"✅ **Import complete!**\n- Processed {shots_processed} shots\n\n📂 **Source JSON:** `{json_path}`"
                    if result.get("errors"):
                        message += f"\n- Warnings: {len(result.get('errors'))} errors occurred"
                    _log_io_paths("Import Animated Characters to All Shots — done", input_path=json_path)
                    yield gr.update(value=message, visible=True)
                else:
                    # Surface per-shot failure details when present
                    err_msg = result.get("error")
                    if not err_msg:
                        failed_shots = result.get("failed_shots") or []
                        if failed_shots:
                            lines = []
                            for fs in failed_shots[:10]:
                                sid = fs.get("scene_id")
                                shid = fs.get("shot_id")
                                shot_err = fs.get("error")
                                fi = fs.get("failed_imports") or []
                                if shot_err:
                                    lines.append(f"  - scene {sid} shot {shid}: {shot_err}")
                                elif fi:
                                    for item in fi:
                                        lines.append(
                                            f"  - scene {sid} shot {shid} asset `{item.get('asset_id')}`: {item.get('error')}"
                                        )
                                else:
                                    lines.append(f"  - scene {sid} shot {shid}: (no detail)")
                            err_msg = "Some shots failed:\n" + "\n".join(lines)
                            successful_shots = result.get("successful_shots") or []
                            if successful_shots:
                                err_msg = (
                                    f"Succeeded: {len(successful_shots)} shot(s); "
                                    f"Failed: {len(failed_shots)} shot(s).\n" + err_msg
                                )
                        else:
                            err_msg = "Unknown error (no details returned)"
                    yield gr.update(value=f"⚠️ Import failed: {err_msg}", visible=True)
            except Exception as e:
                yield gr.update(value=f"⚠️ Import failed: {str(e)}", visible=True)
            finally:
                if temp_input_path and os.path.exists(temp_input_path):
                    try: os.remove(temp_input_path)
                    except OSError: pass

        import_to_shots_btn.click(fn=import_animated_characters_handler, inputs=[project_dir], outputs=[import_status])

        def delete_all_shots_handler():
            yield gr.update(value="🔄 **Deleting all shot scenes...**", visible=True)
            try:
                response = blender_client.delete_all_shots()
                if response.get("status") == "error":
                    yield gr.update(value=f"⚠️ Delete failed: {response.get('message', 'Unknown error')}", visible=True)
                    return
                result = response.get("result", response)
                if result.get("success"):
                    message = result.get("message", f"Deleted {len(result.get('deleted_scenes', []))} shot scenes.")
                    yield gr.update(value=f"✅ {message}", visible=True)
                else:
                    yield gr.update(value=f"⚠️ Delete failed: {result.get('error', 'Unknown error')}", visible=True)
            except Exception as e:
                yield gr.update(value=f"⚠️ Delete failed: {str(e)}", visible=True)
        delete_all_shots_btn.click(fn=delete_all_shots_handler, inputs=[], outputs=[import_status])

        # --- Animated Model Viewer ---
        gr.Markdown("### Animated Model Viewer")
        gr.Markdown("Preview animated character models by selecting Scene, Shot, and Asset.")
        viewer_data_state = gr.State(None)
        load_viewer_btn = gr.Button("🔄 Load Animated Models", variant="secondary", size="lg")
        viewer_status = gr.Markdown(value="", visible=False)
        with gr.Column(visible=False) as viewer_container:
            with gr.Row():
                viewer_scene_dropdown = gr.Dropdown(choices=[], label="Scene", interactive=True, scale=1)
                viewer_shot_dropdown = gr.Dropdown(choices=[], label="Shot", interactive=True, scale=1)
                viewer_asset_dropdown = gr.Dropdown(choices=[], label="Asset ID", interactive=True, scale=1)
            animation_info = gr.Markdown(value="", visible=False)
            animated_model_viewer = gr.Model3D(label="Animated Model Preview", clear_color=(0.9, 0.9, 0.9, 1.0), height=500)
            hide_viewer_btn = gr.Button("🙈 Hide Animated Model Viewer", variant="secondary", size="lg")

        def hide_viewer_container():
            return (gr.update(value="", visible=False), gr.update(visible=False), None, gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(value="", visible=False), gr.update(value=None))
        hide_viewer_btn.click(fn=hide_viewer_container, inputs=[], outputs=[viewer_status, viewer_container, viewer_data_state, viewer_scene_dropdown, viewer_shot_dropdown, viewer_asset_dropdown, animation_info, animated_model_viewer])

        def load_viewer_data_handler(project_dir):
            data, error = load_scene_shot_data(project_dir)
            if error:
                return (gr.update(value=error, visible=True), gr.update(visible=False), None, gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(value="", visible=False), gr.update(value=None))
            scenes = get_scene_choices(data)
            return (gr.update(value="✅ Data loaded. Select Scene → Shot → Asset to preview.", visible=True), gr.update(visible=True), data, gr.update(choices=[str(s) for s in scenes], value=None), gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(value="", visible=False), gr.update(value=None))
        load_viewer_btn.click(fn=load_viewer_data_handler, inputs=[project_dir], outputs=[viewer_status, viewer_container, viewer_data_state, viewer_scene_dropdown, viewer_shot_dropdown, viewer_asset_dropdown, animation_info, animated_model_viewer])

        def on_viewer_scene_change(scene_id, data):
            if not scene_id or not data:
                return (gr.update(choices=[], value=None), gr.update(choices=[], value=None), gr.update(value="", visible=False), gr.update(value=None))
            try: sid = int(scene_id)
            except ValueError: sid = scene_id
            return (gr.update(choices=[str(s) for s in get_shot_choices(data, sid)], value=None), gr.update(choices=[], value=None), gr.update(value="", visible=False), gr.update(value=None))
        viewer_scene_dropdown.change(fn=on_viewer_scene_change, inputs=[viewer_scene_dropdown, viewer_data_state], outputs=[viewer_shot_dropdown, viewer_asset_dropdown, animation_info, animated_model_viewer])

        def on_viewer_shot_change(scene_id, shot_id, data):
            if not scene_id or not shot_id or not data:
                return (gr.update(choices=[], value=None), gr.update(value="", visible=False), gr.update(value=None))
            try: sid, shid = int(scene_id), int(shot_id)
            except ValueError: sid, shid = scene_id, shot_id
            return (gr.update(choices=get_asset_choices(data, sid, shid), value=None), gr.update(value="", visible=False), gr.update(value=None))
        viewer_shot_dropdown.change(fn=on_viewer_shot_change, inputs=[viewer_scene_dropdown, viewer_shot_dropdown, viewer_data_state], outputs=[viewer_asset_dropdown, animation_info, animated_model_viewer])

        def on_viewer_asset_change(scene_id, shot_id, asset_id, data):
            if not scene_id or not shot_id or not asset_id or not data:
                return (gr.update(value="", visible=False), gr.update(value=None))
            details = get_character_action_details(data, scene_id, shot_id, asset_id)
            if not details:
                return (gr.update(value="⚠️ No action found for this character.", visible=True), gr.update(value=None))
            info_text = f"**Animation Details:**\n- **Action ID:** {details.get('action_id', 'N/A')}\n- **Action Name:** {details.get('action_name', 'N/A')}\n- **Description:** {details.get('action_description', 'N/A')}\n"
            animated_path = details.get("animated_path")
            if animated_path and os.path.exists(animated_path):
                info_text += f"- **File:** {animated_path}"
                return (gr.update(value=info_text, visible=True), gr.update(value=animated_path))
            info_text += "- **File:** ⚠️ Not found or not yet created"
            return (gr.update(value=info_text, visible=True), gr.update(value=None))
        viewer_asset_dropdown.change(fn=on_viewer_asset_change, inputs=[viewer_scene_dropdown, viewer_shot_dropdown, viewer_asset_dropdown, viewer_data_state], outputs=[animation_info, animated_model_viewer])

    return {
        "rigging_editor": rigging_editor,
        "animation_editor": animation_editor,
        "animated_models_editor": animated_models_editor,
        "auto_route_editor": auto_route_editor,
    }
