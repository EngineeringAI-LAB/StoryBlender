import os
import json
import logging
import gradio as gr
from ..operators.layout_artist_operators.generate_layout_description import (
    generate_layout_description,
    merge_layout,
)
from ..operators.concept_artist_operators.refine_dimension_estimation import (
    refine_all_asset_dimensions_parallel,
    collect_assets_needing_resize,
)
from .json_editor import JSONEditorComponent
from .blender_client import BlenderClient
from .path_utils import make_paths_absolute, make_paths_relative

logger = logging.getLogger(__name__)


# ============================================================================
# Step 3.1: Resize Assets (moved from Step 2.5)
# ============================================================================

def load_resize_model_choices(project_dir):
    """Load model IDs from dimension_estimation.json for resize selection.
    
    Args:
        project_dir: Project directory path
        
    Returns:
        list: List of asset_id strings
    """
    if not project_dir or not os.path.isabs(project_dir):
        return []
    
    dimension_json_path = os.path.join(project_dir, "formatted_model", "dimension_estimation.json")
    
    if not os.path.exists(dimension_json_path):
        return []
    
    try:
        with open(dimension_json_path, 'r') as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning("Step 3: path conversion failed for dimension_estimation.json: %s", e)
        
        asset_sheet = data.get("asset_sheet", [])
        model_ids = []
        for asset in asset_sheet:
            asset_id = asset.get("asset_id")
            if asset_id:
                model_ids.append(asset_id)
        
        return model_ids
    except Exception:
        return []


def resize_3d_models(
    blender_client,
    project_dir,
    editor_component,
    model_id_list=None
):
    """Resize 3D models using BlenderMCPServer.
    
    Args:
        blender_client: BlenderClient instance for communicating with Blender
        project_dir: Project directory path
        editor_component: JSONEditorComponent to display results
        model_id_list: Optional list of asset_ids to resize. If None, resize all.
        
    Returns:
        dict: Result with success/error and data
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path"
        }
    
    # Check if dimension_estimation.json exists
    dimension_json_path = os.path.join(project_dir, "formatted_model", "dimension_estimation.json")
    if not os.path.exists(dimension_json_path):
        return {
            "error": "⚠️ dimension_estimation.json not found. Please run Step 2.4 first."
        }
    
    # Create output directory
    resized_output_dir = os.path.join(project_dir, "resized_model")
    os.makedirs(resized_output_dir, exist_ok=True)
    
    # Ensure MCP server is running (will start it if not)
    success, message = blender_client.ensure_server_running()
    if not success:
        return {
            "error": f"⚠️ {message}"
        }
    
    # Load JSON and convert relative paths to absolute for Blender
    # (JSON on disk stores portable relative paths like "formatted_model/file.glb")
    temp_input_path = None
    try:
        with open(dimension_json_path, 'r') as f:
            input_data = json.load(f)
        input_data = make_paths_absolute(input_data, project_dir)
        
        temp_input_path = dimension_json_path + ".tmp"
        with open(temp_input_path, 'w') as f:
            json.dump(input_data, f, indent=2)
    except Exception as e:
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        return {
            "error": f"⚠️ Failed to prepare input for Blender: {str(e)}"
        }
    
    # Call resize_assets via BlenderClient
    try:
        response = blender_client.resize_assets(
            path_to_script=temp_input_path,
            model_output_dir=resized_output_dir,
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
        output_json_path = os.path.join(resized_output_dir, "resized_model.json")
        if not os.path.exists(output_json_path):
            return {
                "error": "⚠️ resized_model.json was not created. Check Blender console for errors."
            }
        
        # Load the result for display
        with open(output_json_path, 'r') as f:
            resized_data = json.load(f)
        try:
            resized_data = make_paths_absolute(resized_data, project_dir)
        except Exception as e:
            logger.warning("Step 3: path conversion failed for resized_model.json: %s", e)
        
        # Set editor save path and return success
        editor_component.set_save_path(resized_output_dir)
        
        return {
            "success": True,
            "data": resized_data,
            "output_path": output_json_path,
            "resized_count": result.get("resized_count", 0),
            "total_models": result.get("total_models", 0),
            "errors": result.get("errors", []),
        }
        
    except Exception as e:
        return {
            "error": f"⚠️ Failed to resize assets: {str(e)}"
        }
    finally:
        # Clean up temporary input file
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except OSError:
                pass


def show_loading_and_resize_models(
    editor_component,
    blender_client,
    project_dir,
    model_id_list=None
):
    """Show loading indicator and resize assets."""
    # Build initial loading state
    loading_outputs = editor_component.update_with_result(None)
    loading_msg = "🔄 **Resizing models in Blender...** This may take several minutes. Please wait."
    if model_id_list:
        loading_msg = f"🔄 **Re-resizing {len(model_id_list)} model(s) in Blender...** This may take several minutes. Please wait."
    loading_state = (
        gr.update(value=loading_msg, visible=True),
        gr.update(visible=False),  # Hide resize button during processing
    )
    
    yield loading_outputs + loading_state
    
    # Resize assets
    result = resize_3d_models(
        blender_client,
        project_dir,
        editor_component,
        model_id_list=model_id_list
    )
    
    # Return final result
    final_outputs = editor_component.update_with_result(result)
    
    # Show success message with tip if resizing succeeded
    if result.get("success"):
        success_msg = "✅ **Models resized successfully!** You can use the **3D Model Viewer** in Step 2 to verify the models are correctly sized."
    else:
        success_msg = ""  # Error message is shown in the editor
    
    final_state = (
        gr.update(value=success_msg, visible=bool(success_msg)),  # Show success tip
        gr.update(visible=True),   # Show resize button
    )
    
    yield final_outputs + final_state


def create_resize_wrapper(editor_component, blender_client):
    """Factory function to create a resize wrapper bound to a specific editor component and blender client."""
    def resize_wrapper(project_dir, model_id_list=None):
        """Wrapper to properly yield from the generator."""
        # Convert empty list to None (resize all)
        if model_id_list is not None and len(model_id_list) == 0:
            model_id_list = None
        for result in show_loading_and_resize_models(
            editor_component,
            blender_client,
            project_dir,
            model_id_list=model_id_list
        ):
            yield result
    return resize_wrapper


def refine_size_estimation(
    blender_client,
    project_dir,
    editor_component,
    reasoning_model,
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
    *,
    dim_json_path,
    resized_dir,
    resized_json_path,
    output_json_filename,
):
    """Audit resized dimensions with a vision LLM and re-resize abnormal ones.

    For each asset in the existing resized JSON, ask the LLM whether the
    Blender axis-aligned dimensions look reasonable for that real-world
    object given the front view. If any are flagged, rewrite the
    corresponding entries in the original ``dimension_estimation.json``
    (using the LLM's single corrected dimension) and call
    ``resize_assets`` with ``model_id_list`` restricted to those assets.
    """
    if not project_dir or not os.path.isabs(project_dir):
        return {"error": "⚠️ Project directory must be an absolute path"}

    if not os.path.exists(resized_json_path):
        return {"error": f"⚠️ {os.path.basename(resized_json_path)} not found. Please run resize first."}
    if not os.path.exists(dim_json_path):
        return {"error": f"⚠️ {os.path.basename(dim_json_path)} not found. Please run dimension estimation first."}

    # Load existing resized data (this already contains width/depth/height
    # populated for every asset, plus front_view_url).
    try:
        with open(resized_json_path, 'r', encoding='utf-8') as f:
            resized_data = json.load(f)
        resized_data = make_paths_absolute(resized_data, project_dir)
    except Exception as e:
        return {"error": f"⚠️ Failed to load resized JSON: {e}"}

    asset_sheet = resized_data.get("asset_sheet", []) or []
    if not asset_sheet:
        return {"error": "⚠️ resized JSON has no asset_sheet"}

    # Normalize provider/api_base
    provider = (anyllm_provider or "").strip() or "gemini"
    api_base = anyllm_api_base if (anyllm_api_base and anyllm_api_base.strip()) else None

    # Run per-asset audit in parallel
    refinements = refine_all_asset_dimensions_parallel(
        asset_sheet=asset_sheet,
        anyllm_api_key=anyllm_api_key,
        anyllm_api_base=api_base,
        anyllm_provider=provider,
        reasoning_model=reasoning_model,
        reasoning_effort="medium",
    )

    fix_map = collect_assets_needing_resize(refinements)

    if not fix_map:
        editor_component.set_save_path(resized_dir)
        return {
            "success": True,
            "data": resized_data,
            "output_path": resized_json_path,
            "refined_count": 0,
            "total_audited": len(asset_sheet),
            "refinements": refinements,
            "message": (
                f"✅ All {len(asset_sheet)} asset dimensions look reasonable. "
                f"No re-resizing performed."
            ),
        }

    # Build a temp script JSON: clone original dimension_estimation.json and
    # overwrite width/depth/height for the assets that need correction with
    # the LLM's single proposed dimension.
    try:
        with open(dim_json_path, 'r', encoding='utf-8') as f:
            dim_data = json.load(f)
        dim_data = make_paths_absolute(dim_data, project_dir)
    except Exception as e:
        return {"error": f"⚠️ Failed to load dimension_estimation.json: {e}"}

    for asset in dim_data.get("asset_sheet", []) or []:
        aid = asset.get("asset_id")
        if aid in fix_map:
            r = fix_map[aid]
            asset["width"] = r.get("width")
            asset["depth"] = r.get("depth")
            asset["height"] = r.get("height")

    # Ensure MCP server up
    ok, msg = blender_client.ensure_server_running()
    if not ok:
        return {"error": f"⚠️ {msg}"}

    os.makedirs(resized_dir, exist_ok=True)
    temp_input_path = dim_json_path + ".refine.tmp"
    try:
        with open(temp_input_path, 'w', encoding='utf-8') as f:
            json.dump(dim_data, f, indent=2)

        response = blender_client.resize_assets(
            path_to_script=temp_input_path,
            model_output_dir=resized_dir,
            model_id_list=list(fix_map.keys()),
            output_json_filename=output_json_filename,
        )

        if response.get("status") == "error":
            return {"error": f"⚠️ Blender error: {response.get('message', 'Unknown error')}"}

        result = response.get("result", {})
        if "error" in result:
            return {"error": f"⚠️ {result['error']}"}

        # Reload the merged resized JSON for display
        try:
            with open(resized_json_path, 'r', encoding='utf-8') as f:
                new_data = json.load(f)
            new_data = make_paths_absolute(new_data, project_dir)
        except Exception as e:
            logger.warning("refine: path conversion failed for resized JSON: %s", e)
            new_data = resized_data

        editor_component.set_save_path(resized_dir)

        return {
            "success": True,
            "data": new_data,
            "output_path": resized_json_path,
            "refined_count": len(fix_map),
            "total_audited": len(asset_sheet),
            "refinements": refinements,
            "errors": result.get("errors", []),
        }
    except Exception as e:
        return {"error": f"⚠️ Failed to refine and resize: {e}"}
    finally:
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except OSError:
                pass


def show_loading_and_refine_models(
    editor_component,
    blender_client,
    project_dir,
    reasoning_model,
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
):
    """Show loading indicator and run refine size estimation for core assets."""
    loading_outputs = editor_component.update_with_result(None)
    loading_state = (
        gr.update(
            value=(
                "🪞 **Auditing resized dimensions with the LLM and re-resizing "
                "any abnormal models...** This may take a few minutes."
            ),
            visible=True,
        ),
        gr.update(visible=False),  # Hide refine button
        gr.update(visible=False),  # Hide resize button
    )
    yield loading_outputs + loading_state

    result = refine_size_estimation(
        blender_client=blender_client,
        project_dir=project_dir,
        editor_component=editor_component,
        reasoning_model=reasoning_model,
        anyllm_api_key=anyllm_api_key,
        anyllm_api_base=anyllm_api_base,
        anyllm_provider=anyllm_provider,
        dim_json_path=os.path.join(project_dir, "formatted_model", "dimension_estimation.json")
            if project_dir and os.path.isabs(project_dir) else "",
        resized_dir=os.path.join(project_dir, "resized_model")
            if project_dir and os.path.isabs(project_dir) else "",
        resized_json_path=os.path.join(project_dir, "resized_model", "resized_model.json")
            if project_dir and os.path.isabs(project_dir) else "",
        output_json_filename="resized_model.json",
    )

    final_outputs = editor_component.update_with_result(result)

    if result.get("success"):
        if result.get("refined_count", 0) > 0:
            success_msg = (
                f"✅ **Refinement complete.** Re-resized "
                f"{result['refined_count']} of {result.get('total_audited', '?')} "
                f"model(s) flagged as abnormal."
            )
        else:
            success_msg = result.get(
                "message",
                "✅ All dimensions look reasonable. No re-resize performed.",
            )
    else:
        success_msg = result.get("error", "")

    final_state = (
        gr.update(value=success_msg, visible=bool(success_msg)),
        gr.update(visible=True),  # Show refine button
        gr.update(visible=True),  # Show resize button
    )
    yield final_outputs + final_state


def create_refine_wrapper(editor_component, blender_client):
    """Factory function to create a refine wrapper bound to a specific editor and client."""
    def refine_wrapper(project_dir, reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider):
        for result in show_loading_and_refine_models(
            editor_component,
            blender_client,
            project_dir,
            reasoning_model,
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
        ):
            yield result
    return refine_wrapper


def load_resized_model(project_dir):
    """Load resized model JSON from project_dir/resized_model/resized_model.json."""
    if not project_dir or not os.path.isabs(project_dir):
        return None
    resized_model_path = os.path.join(project_dir, "resized_model", "resized_model.json")
    if os.path.exists(resized_model_path):
        try:
            with open(resized_model_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            try:
                data = make_paths_absolute(data, project_dir)
            except Exception as e:
                logger.warning("Step 3: path conversion failed for resized_model.json: %s", e)
            return data
        except Exception as e:
            print(f"Error loading resized model: {e}")
            return None
    return None


def validate_and_generate_layout(
    reasoning_model,
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
    project_dir,
    editor_component
):
    """Validate inputs and generate layout using the layout artist operator.
    
    Args:
        reasoning_model: The reasoning model to use for generation
        anyllm_api_key: The API key for authentication
        anyllm_api_base: The API base URL for any-llm (optional)
        anyllm_provider: The LLM provider (e.g. 'gemini', 'openai')
        project_dir: The absolute path to the project directory
        editor_component: The JSONEditorComponent to save the result
    
    Returns:
        A dictionary containing the result
    """
    
    # Validate project directory
    if not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path"
        }
    
    # Load resized model data
    resized_model_data = load_resized_model(project_dir)
    if resized_model_data is None:
        return {
            "error": "⚠️ Could not load resized model. Please ensure project_dir/resized_model/resized_model.json exists (complete Step 3.1)."
        }
    
    # Set API base to None if empty string
    anyllm_api_base = anyllm_api_base if anyllm_api_base.strip() else None
    
    # Set provider to default if empty
    anyllm_provider = anyllm_provider.strip() if anyllm_provider else "gemini"
    if not anyllm_provider:
        anyllm_provider = "gemini"

    # Generate layout description
    result = generate_layout_description(
        anyllm_api_key=anyllm_api_key,
        anyllm_api_base=anyllm_api_base,
        anyllm_provider=anyllm_provider,
        reasoning_model=reasoning_model,
        storyboard_script=resized_model_data,
        reasoning_effort="high"
    )
    
    # Unpack the tuple (layout_result, run_stats) returned by generate_layout_description
    if isinstance(result, tuple):
        layout_result, run_stats = result
    else:
        layout_result = result
    
    if layout_result is None:
        return {
            "error": "⚠️ Failed to generate layout description. Please check the API settings and try again."
        }
    
    # Merge layout with resized model data
    merged_data = merge_layout(resized_model_data, layout_result)
    
    # Set the save path for layout files
    layout_save_path = os.path.join(project_dir, "layout_script")
    editor_component.set_save_path(layout_save_path)
    
    # Save the JSON data
    output_path = editor_component.save_json_data(merged_data)
    if output_path:
        return {
            "success": True,
            "data": merged_data,
            "output_path": output_path
        }
    else:
        return {
            "error": "⚠️ Failed to save layout JSON file"
        }


def show_loading_and_generate(editor_component, reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider, project_dir):
    """Show loading indicator and generate layout."""
    # Build initial loading state - all editor components hidden
    loading_outputs = editor_component.update_with_result(None)
    loading_state = (
        gr.update(value="🔄 **Generating scene layout...** This may take 3-5 minutes. Please wait.", visible=True),  # Show loading
    )
    
    yield loading_outputs + loading_state
    
    # Generate the layout (pass editor_component for saving)
    result = validate_and_generate_layout(
        reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider, project_dir, editor_component
    )
    
    # Return final result with editor component updated
    final_outputs = editor_component.update_with_result(result)
    
    # Show success message if generation succeeded
    if result.get("success"):
        success_msg = "✅ **Scene layout generated successfully!** You can review the layout in the JSON editor and proceed to the next step."
    else:
        success_msg = result.get("error", "")
    
    final_state = (
        gr.update(value=success_msg, visible=bool(success_msg)),  # Show success or error
    )
    
    yield final_outputs + final_state


def create_generate_wrapper(editor_component):
    """Factory function to create a generate wrapper bound to a specific editor component."""
    def generate_wrapper(reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider, project_dir):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_generate(editor_component, reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider, project_dir):
            yield result
    return generate_wrapper


def import_all_assets_to_all_scenes(
    blender_client,
    project_dir,
    layout_editor
):
    """Import all assets to all scenes using the latest layout_script JSON.
    
    Args:
        blender_client: BlenderClient instance for communicating with Blender
        project_dir: Project directory path
        layout_editor: JSONEditorComponent for the layout script
        
    Returns:
        dict: Result with success/error and data
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path"
        }
    
    # Set the save path for the layout editor
    layout_save_path = os.path.join(project_dir, "layout_script")
    layout_editor.set_save_path(layout_save_path)
    
    # Get the latest layout_script JSON path
    json_filepath = layout_editor.get_path_to_latest_json()
    
    if not json_filepath:
        return {
            "error": "⚠️ No layout_script JSON found. Please generate a scene layout first (Step 3.2)."
        }
    
    # Ensure MCP server is running
    success, message = blender_client.ensure_server_running()
    if not success:
        return {
            "error": f"⚠️ {message}"
        }
    
    # Load JSON and convert relative paths to absolute for Blender
    temp_input_path = None
    try:
        with open(json_filepath, 'r') as f:
            input_data = json.load(f)
        input_data = make_paths_absolute(input_data, project_dir)
        
        temp_input_path = json_filepath + ".tmp"
        with open(temp_input_path, 'w') as f:
            json.dump(input_data, f, indent=2)
    except Exception as e:
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        return {
            "error": f"⚠️ Failed to prepare input for Blender: {str(e)}"
        }
    
    # Call import_all_assets_to_all_scenes_json_input via BlenderClient
    try:
        response = blender_client.import_all_assets_to_all_scenes_json_input(
            json_filepath=temp_input_path
        )
        
        if response.get("status") == "error":
            return {
                "error": f"⚠️ Blender error: {response.get('message', 'Unknown error')}"
            }
        
        result = response.get("result", {})
        
        if result.get("success"):
            return {
                "success": True,
                "message": "✅ All assets have been imported to all scenes successfully!",
                "json_filepath": json_filepath,
            }
        else:
            # Check for failed objects
            failed_objects = result.get("failed_objects", [])
            error_msg = result.get("error", "")
            
            if failed_objects:
                error_lines = ["⚠️ Some assets failed to import:"]
                for failed in failed_objects:
                    scene_id = failed.get("scene_id", "unknown")
                    object_id = failed.get("object_id", "unknown")
                    err = failed.get("error", "Unknown error")
                    error_lines.append(f"  - Scene {scene_id}, Asset {object_id}: {err}")
                return {
                    "error": "\n".join(error_lines),
                    "failed_objects": failed_objects,
                }
            elif error_msg:
                return {
                    "error": f"⚠️ {error_msg}"
                }
            else:
                return {
                    "error": "⚠️ Import failed with unknown error"
                }
                
    except Exception as e:
        return {
            "error": f"⚠️ Failed to import assets: {str(e)}"
        }
    finally:
        # Clean up temporary input file
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except OSError:
                pass


def finish_basic_layout_formulation(
    blender_client,
    project_dir,
    layout_editor
):
    """Read transforms from Blender and update the layout JSON with actual values.
    
    Args:
        blender_client: BlenderClient instance for communicating with Blender
        project_dir: Project directory path
        layout_editor: JSONEditorComponent for the layout script
        
    Returns:
        dict: Result with success/error and updated data
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path"
        }
    
    # Set the save path for the layout editor
    layout_save_path = os.path.join(project_dir, "layout_script")
    layout_editor.set_save_path(layout_save_path)
    
    # Get the latest layout_script JSON path
    json_filepath = layout_editor.get_path_to_latest_json()
    
    if not json_filepath:
        return {
            "error": "⚠️ No layout_script JSON found. Please generate a scene layout first (Step 3.2)."
        }
    
    # Load the JSON data
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
        try:
            layout_data = make_paths_absolute(layout_data, project_dir)
        except Exception as e:
            logger.warning("Step 3: path conversion failed for layout JSON: %s", e)
    except Exception as e:
        return {
            "error": f"⚠️ Failed to load layout JSON: {str(e)}"
        }
    
    # Ensure MCP server is running
    success, message = blender_client.ensure_server_running()
    if not success:
        return {
            "error": f"⚠️ {message}"
        }
    
    # Process each scene and update transforms
    scene_details = layout_data.get("scene_details", [])
    update_errors = []
    updated_count = 0
    
    for scene_detail in scene_details:
        scene_id = scene_detail.get("scene_id")
        scene_name = f"Scene_{scene_id}"
        
        # Switch to the scene
        switch_response = blender_client.switch_or_create_scene(scene_name=scene_name)
        if switch_response.get("status") == "error":
            update_errors.append(f"Scene {scene_id}: Failed to switch - {switch_response.get('message', 'Unknown error')}")
            continue
        
        switch_result = switch_response.get("result", {})
        if not switch_result.get("success", True):  # Default to True if success not in result
            error = switch_result.get("error", "Unknown error")
            update_errors.append(f"Scene {scene_id}: Failed to switch - {error}")
            continue
        
        # Get assets from scene_setup.layout_description.assets
        scene_setup = scene_detail.get("scene_setup", {})
        layout_description = scene_setup.get("layout_description", {})
        assets = layout_description.get("assets", [])
        
        for asset_info in assets:
            asset_id = asset_info.get("asset_id")
            
            # Get transform from Blender using asset_id as model_name
            transform_response = blender_client.get_asset_transform(model_name=asset_id)
            
            if transform_response.get("status") == "error":
                update_errors.append(f"Scene {scene_id}, Asset {asset_id}: {transform_response.get('message', 'Unknown error')}")
                continue
            
            transform_result = transform_response.get("result", {})
            
            if not transform_result.get("success"):
                error = transform_result.get("error", "Unknown error")
                update_errors.append(f"Scene {scene_id}, Asset {asset_id}: {error}")
                continue
            
            # Update asset_info with transform data
            asset_info["location"] = transform_result.get("location")
            asset_info["rotation"] = transform_result.get("rotation")
            asset_info["scale"] = transform_result.get("scale")
            asset_info["dimensions"] = transform_result.get("dimensions")
            updated_count += 1
    
    # Save the updated JSON as a new version
    output_path = layout_editor.save_json_data(layout_data)
    
    if not output_path:
        return {
            "error": "⚠️ Failed to save updated layout JSON"
        }
    
    # Build result message
    if update_errors:
        error_summary = "\n".join([f"  - {e}" for e in update_errors[:10]])  # Limit to first 10 errors
        if len(update_errors) > 10:
            error_summary += f"\n  - ... and {len(update_errors) - 10} more errors"
        
        return {
            "success": True,
            "data": layout_data,
            "output_path": output_path,
            "warning": f"⚠️ Updated {updated_count} assets, but some errors occurred:\n{error_summary}",
            "updated_count": updated_count,
            "error_count": len(update_errors),
        }
    else:
        return {
            "success": True,
            "data": layout_data,
            "output_path": output_path,
            "updated_count": updated_count,
        }


def show_loading_and_import_assets(blender_client, layout_editor, project_dir):
    """Show loading indicator and import all assets."""
    # Initial loading state
    loading_state = (
        gr.update(value="🔄 **Importing all assets to all scenes...** This may take a few minutes. Please wait.", visible=True),
        gr.update(visible=False),  # Hide import button
        gr.update(visible=False),  # Hide finish button
        gr.update(visible=False),  # Hide delete button
    )
    
    yield loading_state
    
    # Import assets
    result = import_all_assets_to_all_scenes(
        blender_client,
        project_dir,
        layout_editor
    )
    
    # Final state
    if result.get("success"):
        success_msg = result.get("message", "✅ All assets imported successfully!")
        success_msg += "\n\n📝 **Next step:** Check each scene in Blender and modify the assets transformations if needed. Then click the **'✅ Finish Basic Layout Formulation'** button when you're done."
        yield (
            gr.update(value=success_msg, visible=True),
            gr.update(visible=True),   # Show import button
            gr.update(visible=True),   # Show finish button
            gr.update(visible=True),   # Show delete button
        )
    else:
        error_msg = result.get("error", "⚠️ Unknown error occurred")
        yield (
            gr.update(value=error_msg, visible=True),
            gr.update(visible=True),   # Show import button
            gr.update(visible=True),   # Show finish button
            gr.update(visible=True),   # Show delete button
        )


def show_loading_and_finish_layout(blender_client, layout_editor, project_dir):
    """Show loading indicator and finish layout formulation."""
    # Initial loading state - hide editor while processing
    loading_outputs = layout_editor.update_with_result(None)
    loading_state = (
        gr.update(value="🔄 **Reading transforms from Blender and updating layout...** Please wait.", visible=True),
        gr.update(visible=False),  # Hide import button
        gr.update(visible=False),  # Hide finish button
        gr.update(visible=False),  # Hide delete button
    )
    
    yield loading_outputs + loading_state
    
    # Finish layout formulation
    result = finish_basic_layout_formulation(
        blender_client,
        project_dir,
        layout_editor
    )
    
    # Final state
    final_outputs = layout_editor.update_with_result(result)
    
    if result.get("success"):
        updated_count = result.get("updated_count", 0)
        warning = result.get("warning", "")
        
        if warning:
            success_msg = warning
        else:
            success_msg = f"✅ **Layout formulation complete!** Updated {updated_count} asset transforms from Blender.\n\nThe updated layout has been saved as a new version."
        
        yield final_outputs + (
            gr.update(value=success_msg, visible=True),
            gr.update(visible=True),   # Show import button
            gr.update(visible=True),   # Show finish button
            gr.update(visible=True),   # Show delete button
        )
    else:
        error_msg = result.get("error", "⚠️ Unknown error occurred")
        yield final_outputs + (
            gr.update(value=error_msg, visible=True),
            gr.update(visible=True),   # Show import button
            gr.update(visible=True),   # Show finish button
            gr.update(visible=True),   # Show delete button
        )


def create_import_wrapper(blender_client, layout_editor):
    """Factory function to create an import wrapper bound to specific components."""
    def import_wrapper(project_dir):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_import_assets(blender_client, layout_editor, project_dir):
            yield result
    return import_wrapper


def create_finish_wrapper(blender_client, layout_editor):
    """Factory function to create a finish wrapper bound to specific components."""
    def finish_wrapper(project_dir):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_finish_layout(blender_client, layout_editor, project_dir):
            yield result
    return finish_wrapper


def delete_all_scenes_and_assets(blender_client):
    """Delete all scenes and assets in Blender.
    
    Args:
        blender_client: BlenderClient instance for communicating with Blender
        
    Returns:
        dict: Result with success/error
    """
    # Ensure MCP server is running
    success, message = blender_client.ensure_server_running()
    if not success:
        return {
            "error": f"⚠️ {message}"
        }
    
    try:
        response = blender_client.delete_all_scenes_and_assets()
        
        if response.get("status") == "error":
            return {
                "error": f"⚠️ Blender error: {response.get('message', 'Unknown error')}"
            }
        
        result = response.get("result", {})
        
        if result.get("success"):
            deleted_scenes = result.get("deleted_scenes", [])
            return {
                "success": True,
                "message": f"🗑️ Successfully deleted all scenes and assets. Deleted scenes: {', '.join(deleted_scenes) if deleted_scenes else 'None'}",
            }
        else:
            error = result.get("error", "Unknown error")
            return {
                "error": f"⚠️ {error}"
            }
    except Exception as e:
        return {
            "error": f"⚠️ Failed to delete scenes and assets: {str(e)}"
        }


def show_loading_and_delete_all(blender_client):
    """Show loading indicator and delete all scenes and assets."""
    # Initial loading state
    loading_state = (
        gr.update(value="🔄 **Deleting all scenes and assets...** Please wait.", visible=True),
        gr.update(visible=False),  # Hide import button
        gr.update(visible=False),  # Hide finish button
        gr.update(visible=False),  # Hide delete button
    )
    
    yield loading_state
    
    # Delete all
    result = delete_all_scenes_and_assets(blender_client)
    
    # Final state
    if result.get("success"):
        success_msg = result.get("message", "🗑️ All scenes and assets deleted successfully!")
        yield (
            gr.update(value=success_msg, visible=True),
            gr.update(visible=True),   # Show import button
            gr.update(visible=True),   # Show finish button
            gr.update(visible=True),   # Show delete button
        )
    else:
        error_msg = result.get("error", "⚠️ Unknown error occurred")
        yield (
            gr.update(value=error_msg, visible=True),
            gr.update(visible=True),   # Show import button
            gr.update(visible=True),   # Show finish button
            gr.update(visible=True),   # Show delete button
        )


def create_delete_wrapper(blender_client):
    """Factory function to create a delete wrapper bound to specific components."""
    def delete_wrapper():
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_delete_all(blender_client):
            yield result
    return delete_wrapper


def create_layout_ui(reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider, project_dir, blender_client):
    """Create the Step 3: Formulate Core Scene Spatial Layout by Layout Artist UI section.
    
    Args:
        reasoning_model: Gradio component for reasoning model selection
        anyllm_api_key: Gradio component for any-llm API key
        anyllm_api_base: Gradio component for any-llm API base URL
        project_dir: Gradio component for project directory
        blender_client: BlenderClient instance for communicating with Blender
    
    Returns:
        dict with UI components that may be needed by other parts of the app
    """
    gr.Markdown("## Step 3: Formulate Core Scene Spatial Layout by Layout Artist")
    
    # =========================================================================
    # Step 3.1: Resize Assets (moved from Step 2.5)
    # =========================================================================
    gr.Markdown("### Step 3.1: Resize Assets")
    gr.Markdown("Resize 3D models in Blender based on estimated dimensions. **Requires Blender with BlenderMCPServer running.**")
    
    gr.Markdown("#### Select Models to Resize")
    gr.Markdown("Select specific models to re-resize (useful when some models failed). Leave empty or click 'Select All' to resize all models.")
    
    with gr.Row():
        resize_model_selection = gr.CheckboxGroup(
            choices=[],
            label="Select models to resize",
            info="Check the models you want to resize. If none selected, all models will be resized.",
            scale=3
        )
        with gr.Column(scale=1):
            load_resize_choices_btn = gr.Button("🔄 Load Models", variant="secondary", size="sm")
            select_all_resize_btn = gr.Button("☑️ Select All", variant="secondary", size="sm")
    
    resize_selection_status = gr.Markdown(value="No models loaded. Click 'Load Models' to load available models.", visible=True)
    
    with gr.Row():
        resize_btn = gr.Button("📐 Resize Assets", variant="primary", size="lg", scale=1)
        refine_btn = gr.Button("🪞 Refine Size Estimation", variant="secondary", size="lg", scale=1)

    gr.Markdown(
        "ℹ️ **Refine Size Estimation** audits the current resized dimensions of every "
        "core asset with a vision LLM (one model at a time, in parallel). For models "
        "with clearly abnormal width/depth/height (e.g. a vertically-oriented arrow "
        "whose `length` ended up as `height`), it re-estimates a single corrected "
        "dimension and re-resizes those models in Blender."
    )

    # Loading status indicator for resize assets (hidden by default)
    resize_loading_status = gr.Markdown(value="", visible=False)
    
    # Create JSON editor component for resized models results
    resized_editor = JSONEditorComponent(
        label="Resized Models Result",
        visible_initially=False,
        file_basename="resized_model",
        use_version_control=False,  # Save as resized_model.json (overwrites)
        json_root_keys_list=["asset_sheet"],
        title="Step 3.1"
    )
    
    # Wire up the Resume button with project_dir input
    resized_editor.setup_resume_with_project_dir(project_dir, subfolder="resized_model")
    
    # Create wrapper function for resize assets
    resize_wrapper = create_resize_wrapper(resized_editor, blender_client)
    
    # Handler for Load Models button (for resize)
    def load_resize_choices_handler(project_dir_val):
        choices = load_resize_model_choices(project_dir_val)
        if choices:
            status_msg = f"**{len(choices)} model(s) available.** Select models to re-resize, or leave empty to resize all."
        else:
            status_msg = "⚠️ No models found. Please complete Step 2.4 (Estimate Dimensions) first."
        return gr.update(choices=choices, value=[]), gr.update(value=status_msg)
    
    load_resize_choices_btn.click(
        fn=load_resize_choices_handler,
        inputs=[project_dir],
        outputs=[resize_model_selection, resize_selection_status]
    )
    
    # Handler for Select All button (for resize)
    def select_all_resize_handler(project_dir_val):
        choices = load_resize_model_choices(project_dir_val)
        if choices:
            status_msg = f"**All {len(choices)} model(s) selected.**"
            return gr.update(choices=choices, value=choices), gr.update(value=status_msg)
        else:
            return gr.update(choices=[], value=[]), gr.update(value="⚠️ No models found.")
    
    select_all_resize_btn.click(
        fn=select_all_resize_handler,
        inputs=[project_dir],
        outputs=[resize_model_selection, resize_selection_status]
    )
    
    # Handler for selection change (for resize)
    def update_resize_selection_status(selected):
        if not selected:
            return gr.update(value="No models selected. All models will be resized.")
        return gr.update(value=f"**Selected for resizing ({len(selected)}):** {', '.join(selected)}")
    
    resize_model_selection.change(
        fn=update_resize_selection_status,
        inputs=[resize_model_selection],
        outputs=[resize_selection_status]
    )
    
    # Resize button click handler
    resize_btn.click(
        fn=resize_wrapper,
        inputs=[project_dir, resize_model_selection],
        outputs=resized_editor.get_output_components() + [resize_loading_status, resize_btn],
    )

    # Refine button click handler
    refine_wrapper = create_refine_wrapper(resized_editor, blender_client)
    refine_btn.click(
        fn=refine_wrapper,
        inputs=[project_dir, reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider],
        outputs=resized_editor.get_output_components() + [resize_loading_status, refine_btn, resize_btn],
    )

    # =========================================================================
    # Step 3.2: Generate Spatial Layout for Each Scene (was Step 3.1)
    # =========================================================================
    gr.Markdown("### Step 3.2: Generate Spatial Layout for Each Scene")
    gr.Markdown("Generate 3D spatial layout for scene assets based on the resized model data.")
    
    # Input info
    gr.Markdown("**Input:** `project_dir/resized_model/resized_model.json`")
    gr.Markdown("**Output:** `project_dir/layout_script/layout_script_v{N}.json`")
    
    # Generate button
    generate_layout_btn = gr.Button("🎬 Formulate Scene Layout", variant="primary", size="lg")
    
    # Loading/status indicator (hidden by default)
    layout_status = gr.Markdown(value="", visible=False)
    
    # Create JSON editor component
    # save_path will be set dynamically when generating (project_dir/layout_script)
    layout_editor = JSONEditorComponent(
        label="Scene Layout JSON",
        visible_initially=False,
        file_basename="layout_script",  # Files will be saved as layout_script_v1.json, layout_script_v2.json, etc.
        json_root_keys_list=["scene_details"],
        title="Step 3.2"
    )
    
    # Wire up the Resume button with project_dir input
    layout_editor.setup_resume_with_project_dir(project_dir, subfolder="layout_script")
    
    # Create wrapper function for the generator
    generate_wrapper = create_generate_wrapper(layout_editor)
    
    # Generate button click handler
    generate_layout_btn.click(
        fn=generate_wrapper,
        inputs=[
            reasoning_model,
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
            project_dir,
        ],
        outputs=layout_editor.get_output_components() + [layout_status],
        concurrency_limit=None,
    )
    
    # ============================================================================
    # Step 3.3: Organize Assets Layout for Each Scene (was Step 3.2)
    # ============================================================================
    gr.Markdown("### Step 3.3: Organize Assets Layout for Each Scene")
    gr.Markdown("Import all assets to scenes in Blender, make sure to check is there any modification needed after the import, then adjust their positions manually in Blender. Finally, save the final layout to the layout_script_v{N}.json file by clicking the '✅ Finish Basic Layout Formulation' button. You can use the '🗑️ Delete All Scenes and Assets' button to restart.")
    gr.Markdown("After the import, there will be scenes named 'Scene_{id}' in Blender. There is also a scene named 'Scene', which is used as a drafting playground.")

    # Status indicator for Step 3.2
    organize_status = gr.Markdown(value="", visible=False)
    
    with gr.Row():
        # Import all assets button
        import_assets_btn = gr.Button(
            "🚀 Import All Core Assets to All Scenes",
            variant="primary",
            size="lg"
        )
        
        # Finish layout button
        finish_layout_btn = gr.Button(
            "✅ Finish Basic Layout Formulation",
            variant="secondary",
            size="lg",
            visible=True  # Always visible, but will only work after import
        )
        
        # Delete all scenes and assets button
        delete_all_btn = gr.Button(
            "🗑️ Delete All Scenes and Assets",
            variant="stop",
            size="lg"
        )
    
    # Create wrapper functions for the buttons
    import_wrapper = create_import_wrapper(blender_client, layout_editor)
    finish_wrapper = create_finish_wrapper(blender_client, layout_editor)
    delete_wrapper = create_delete_wrapper(blender_client)
    
    # Import assets button click handler
    import_assets_btn.click(
        fn=import_wrapper,
        inputs=[project_dir],
        outputs=[
            organize_status,
            import_assets_btn,
            finish_layout_btn,
            delete_all_btn,
        ],
    )
    
    # Finish layout button click handler
    finish_layout_btn.click(
        fn=finish_wrapper,
        inputs=[project_dir],
        outputs=layout_editor.get_output_components() + [
            organize_status,
            import_assets_btn,
            finish_layout_btn,
            delete_all_btn,
        ],
    )
    
    # Delete all button click handler
    delete_all_btn.click(
        fn=delete_wrapper,
        inputs=[],
        outputs=[
            organize_status,
            import_assets_btn,
            finish_layout_btn,
            delete_all_btn,
        ],
    )
    
    return {
        "layout_editor": layout_editor,
        "resized_editor": resized_editor,
    }
