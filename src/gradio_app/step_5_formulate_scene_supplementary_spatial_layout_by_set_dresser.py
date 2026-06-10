import os
import json
import logging
import gradio as gr
from ..operators.set_dresser_operators.generate_supplementary_layout_description import (
    generate_supplementary_layout_description,
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
# Step 5.1: Resize Supplementary Assets
# ============================================================================

def load_resize_supplementary_model_choices(project_dir):
    """Load model IDs from formatted_supplementary_assets/dimension_estimation.json for resize selection.
    
    Args:
        project_dir: Project directory path
        
    Returns:
        list: List of asset_id strings
    """
    if not project_dir or not os.path.isabs(project_dir):
        return []
    
    dimension_json_path = os.path.join(project_dir, "formatted_supplementary_assets", "dimension_estimation.json")
    
    if not os.path.exists(dimension_json_path):
        return []
    
    try:
        with open(dimension_json_path, 'r') as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning("Step 5: path conversion failed for dimension_estimation.json: %s", e)
        
        asset_sheet = data.get("asset_sheet", [])
        model_ids = []
        for asset in asset_sheet:
            asset_id = asset.get("asset_id")
            if asset_id:
                model_ids.append(asset_id)
        
        return model_ids
    except Exception:
        return []


def resize_supplementary_models(
    blender_client,
    project_dir,
    editor_component,
    model_id_list=None
):
    """Resize supplementary 3D models using BlenderMCPServer.
    
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
    
    # Check if dimension_estimation.json exists in formatted_supplementary_assets
    dimension_json_path = os.path.join(project_dir, "formatted_supplementary_assets", "dimension_estimation.json")
    if not os.path.exists(dimension_json_path):
        return {
            "error": "⚠️ dimension_estimation.json not found. Please run Step 4.4 first."
        }
    
    # Create output directory
    resized_output_dir = os.path.join(project_dir, "resized_supplementary_assets")
    os.makedirs(resized_output_dir, exist_ok=True)
    
    # Ensure MCP server is running (will start it if not)
    success, message = blender_client.ensure_server_running()
    if not success:
        return {
            "error": f"⚠️ {message}"
        }
    
    # Load JSON and convert relative paths to absolute for Blender
    # (JSON on disk stores portable relative paths like "formatted_supplementary_assets/file.glb")
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
            model_id_list=model_id_list,
            output_json_filename="resized_supplementary_model.json"
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
        output_json_path = os.path.join(resized_output_dir, "resized_supplementary_model.json")
        if not os.path.exists(output_json_path):
            return {
                "error": "⚠️ resized_supplementary_model.json was not created. Check Blender console for errors."
            }
        
        # Load the result for display
        with open(output_json_path, 'r') as f:
            resized_data = json.load(f)
        try:
            resized_data = make_paths_absolute(resized_data, project_dir)
        except Exception as e:
            logger.warning("Step 5: path conversion failed for resized_supplementary_model.json: %s", e)
        
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
            "error": f"⚠️ Failed to resize supplementary models: {str(e)}"
        }
    finally:
        # Clean up temporary input file
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except OSError:
                pass


def show_loading_and_resize_supplementary_models(
    editor_component,
    blender_client,
    project_dir,
    model_id_list=None
):
    """Show loading indicator and resize supplementary models."""
    # Build initial loading state
    loading_outputs = editor_component.update_with_result(None)
    loading_msg = "🔄 **Resizing supplementary models in Blender...** This may take several minutes. Please wait."
    if model_id_list:
        loading_msg = f"🔄 **Re-resizing {len(model_id_list)} supplementary model(s) in Blender...** This may take several minutes. Please wait."
    loading_state = (
        gr.update(value=loading_msg, visible=True),
        gr.update(visible=False),  # Hide resize button during processing
    )
    
    yield loading_outputs + loading_state
    
    # Resize assets
    result = resize_supplementary_models(
        blender_client,
        project_dir,
        editor_component,
        model_id_list=model_id_list
    )
    
    # Return final result
    final_outputs = editor_component.update_with_result(result)
    
    # Show success message with tip if resizing succeeded
    if result.get("success"):
        success_msg = "✅ **Supplementary models resized successfully!** You can use the **3D Model Viewer** in Step 4 to verify the models are correctly sized."
    else:
        success_msg = ""  # Error message is shown in the editor
    
    final_state = (
        gr.update(value=success_msg, visible=bool(success_msg)),  # Show success tip
        gr.update(visible=True),   # Show resize button
    )
    
    yield final_outputs + final_state


def create_resize_supplementary_wrapper(editor_component, blender_client):
    """Factory function to create a resize wrapper bound to a specific editor component and blender client."""
    def resize_wrapper(project_dir, model_id_list=None):
        """Wrapper to properly yield from the generator."""
        # Convert empty list to None (resize all)
        if model_id_list is not None and len(model_id_list) == 0:
            model_id_list = None
        for result in show_loading_and_resize_supplementary_models(
            editor_component,
            blender_client,
            project_dir,
            model_id_list=model_id_list
        ):
            yield result
    return resize_wrapper


def refine_supplementary_size_estimation(
    blender_client,
    project_dir,
    editor_component,
    reasoning_model,
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
):
    """Audit resized supplementary dimensions with a vision LLM and re-resize abnormal ones."""
    if not project_dir or not os.path.isabs(project_dir):
        return {"error": "⚠️ Project directory must be an absolute path"}

    dim_json_path = os.path.join(project_dir, "formatted_supplementary_assets", "dimension_estimation.json")
    resized_dir = os.path.join(project_dir, "resized_supplementary_assets")
    resized_json_path = os.path.join(resized_dir, "resized_supplementary_model.json")
    output_json_filename = "resized_supplementary_model.json"

    if not os.path.exists(resized_json_path):
        return {"error": f"⚠️ {output_json_filename} not found. Please run Step 5.1 (Resize) first."}
    if not os.path.exists(dim_json_path):
        return {"error": "⚠️ formatted_supplementary_assets/dimension_estimation.json not found."}

    try:
        with open(resized_json_path, 'r', encoding='utf-8') as f:
            resized_data = json.load(f)
        resized_data = make_paths_absolute(resized_data, project_dir)
    except Exception as e:
        return {"error": f"⚠️ Failed to load resized JSON: {e}"}

    asset_sheet = resized_data.get("asset_sheet", []) or []
    if not asset_sheet:
        return {"error": "⚠️ resized supplementary JSON has no asset_sheet"}

    provider = (anyllm_provider or "").strip() or "gemini"
    api_base = anyllm_api_base if (anyllm_api_base and anyllm_api_base.strip()) else None

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
                f"✅ All {len(asset_sheet)} supplementary asset dimensions look "
                f"reasonable. No re-resizing performed."
            ),
        }

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

        try:
            with open(resized_json_path, 'r', encoding='utf-8') as f:
                new_data = json.load(f)
            new_data = make_paths_absolute(new_data, project_dir)
        except Exception as e:
            logger.warning("refine: path conversion failed for resized supplementary JSON: %s", e)
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


def show_loading_and_refine_supplementary_models(
    editor_component,
    blender_client,
    project_dir,
    reasoning_model,
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
):
    """Show loading indicator and run refine for supplementary assets."""
    loading_outputs = editor_component.update_with_result(None)
    loading_state = (
        gr.update(
            value=(
                "🪞 **Auditing resized supplementary dimensions with the LLM and "
                "re-resizing any abnormal models...** This may take a few minutes."
            ),
            visible=True,
        ),
        gr.update(visible=False),  # Hide refine button
        gr.update(visible=False),  # Hide resize button
    )
    yield loading_outputs + loading_state

    result = refine_supplementary_size_estimation(
        blender_client=blender_client,
        project_dir=project_dir,
        editor_component=editor_component,
        reasoning_model=reasoning_model,
        anyllm_api_key=anyllm_api_key,
        anyllm_api_base=anyllm_api_base,
        anyllm_provider=anyllm_provider,
    )

    final_outputs = editor_component.update_with_result(result)

    if result.get("success"):
        if result.get("refined_count", 0) > 0:
            success_msg = (
                f"✅ **Refinement complete.** Re-resized "
                f"{result['refined_count']} of {result.get('total_audited', '?')} "
                f"supplementary model(s) flagged as abnormal."
            )
        else:
            success_msg = result.get(
                "message",
                "✅ All supplementary dimensions look reasonable. No re-resize performed.",
            )
    else:
        success_msg = result.get("error", "")

    final_state = (
        gr.update(value=success_msg, visible=bool(success_msg)),
        gr.update(visible=True),  # Show refine button
        gr.update(visible=True),  # Show resize button
    )
    yield final_outputs + final_state


def create_refine_supplementary_wrapper(editor_component, blender_client):
    """Factory function to create a refine wrapper for supplementary assets."""
    def refine_wrapper(project_dir, reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider):
        for result in show_loading_and_refine_supplementary_models(
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


def load_layout_script(project_dir):
    """Load the latest layout_script JSON from project_dir/layout_script/."""
    if not project_dir or not os.path.isabs(project_dir):
        return None, None
    
    layout_script_dir = os.path.join(project_dir, "layout_script")
    if not os.path.exists(layout_script_dir):
        return None, None
    
    # Find the latest version
    latest_version = 0
    latest_path = None
    
    for filename in os.listdir(layout_script_dir):
        if filename.startswith("layout_script_v") and filename.endswith(".json"):
            try:
                version_str = filename[len("layout_script_v"):-5]
                version = int(version_str)
                if version > latest_version:
                    latest_version = version
                    latest_path = os.path.join(layout_script_dir, filename)
            except ValueError:
                continue
    
    if latest_path:
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            try:
                data = make_paths_absolute(data, project_dir)
            except Exception as e:
                logger.warning("Step 5: path conversion failed for layout script: %s", e)
            return data, latest_path
        except Exception as e:
            print(f"Error loading layout script: {e}")
            return None, None
    return None, None


def load_resized_supplementary_assets(project_dir):
    """Load resized supplementary assets JSON from project_dir/resized_supplementary_assets/."""
    if not project_dir or not os.path.isabs(project_dir):
        return None, None
    
    assets_path = os.path.join(project_dir, "resized_supplementary_assets", "resized_supplementary_model.json")
    if os.path.exists(assets_path):
        try:
            with open(assets_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            try:
                data = make_paths_absolute(data, project_dir)
            except Exception as e:
                logger.warning("Step 5: path conversion failed for resized_supplementary_model.json: %s", e)
            return data, assets_path
        except Exception as e:
            print(f"Error loading resized supplementary assets: {e}")
            return None, None
    return None, None


def validate_and_generate_supplementary_layout(
    reasoning_model,
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
    project_dir,
    editor_component
):
    """Validate inputs and generate supplementary layout using the set dresser operator.
    
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
    
    # Load layout script (latest version)
    layout_script, layout_script_path = load_layout_script(project_dir)
    if layout_script is None:
        return {
            "error": "⚠️ Could not load layout script. Please ensure project_dir/layout_script/layout_script_v{N}.json exists."
        }
    
    # Load resized supplementary assets
    resized_supplementary_assets, assets_path = load_resized_supplementary_assets(project_dir)
    if resized_supplementary_assets is None:
        return {
            "error": "⚠️ Could not load resized supplementary assets. Please ensure project_dir/resized_supplementary_assets/resized_supplementary_model.json exists (complete Step 5.1)."
        }
    
    # Set API base to None if empty string
    anyllm_api_base = anyllm_api_base if anyllm_api_base.strip() else None
    
    # Set provider to default if empty
    anyllm_provider = anyllm_provider.strip() if anyllm_provider else "gemini"
    if not anyllm_provider:
        anyllm_provider = "gemini"

    # Generate supplementary layout description
    result = generate_supplementary_layout_description(
        anyllm_api_key=anyllm_api_key,
        anyllm_api_base=anyllm_api_base,
        anyllm_provider=anyllm_provider,
        reasoning_model=reasoning_model,
        layout_script=layout_script,
        formatted_supplementary_assets=resized_supplementary_assets,
        reasoning_effort="high"
    )
    
    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        return {
            "error": f"⚠️ Failed to generate supplementary layout: {error_msg}"
        }
    
    # Get the generated data
    generated_data = result.get("data")
    
    # Set the save path for supplementary layout files
    supplementary_layout_save_path = os.path.join(project_dir, "supplementary_layout_script")
    editor_component.set_save_path(supplementary_layout_save_path)
    
    # Save the JSON data
    output_path = editor_component.save_json_data(generated_data)
    if output_path:
        return {
            "success": True,
            "data": generated_data,
            "output_path": output_path
        }
    else:
        return {
            "error": "⚠️ Failed to save supplementary layout JSON file"
        }


def show_loading_and_generate(editor_component, reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider, project_dir):
    """Show loading indicator and generate supplementary layout."""
    # Build initial loading state - all editor components hidden
    loading_outputs = editor_component.update_with_result(None)
    loading_state = (
        gr.update(value="🔄 **Generating supplementary spatial layout...** This may take several minutes as it processes each scene. Please wait.", visible=True),
    )
    
    yield loading_outputs + loading_state
    
    # Generate the supplementary layout (pass editor_component for saving)
    result = validate_and_generate_supplementary_layout(
        reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider, project_dir, editor_component
    )
    
    # Return final result with editor component updated
    final_outputs = editor_component.update_with_result(result)
    
    # Show success message if generation succeeded
    if result.get("success"):
        success_msg = "✅ **Supplementary spatial layout generated successfully!** You can review the layout in the JSON editor and proceed to the next step."
    else:
        success_msg = result.get("error", "")
    
    final_state = (
        gr.update(value=success_msg, visible=bool(success_msg)),
    )
    
    yield final_outputs + final_state


def create_generate_wrapper(editor_component):
    """Factory function to create a generate wrapper bound to a specific editor component."""
    def generate_wrapper(reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider, project_dir):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_generate(editor_component, reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider, project_dir):
            yield result
    return generate_wrapper


def import_supplementary_assets_to_all_scenes(
    blender_client,
    project_dir,
    supplementary_layout_editor
):
    """Import supplementary assets to all scenes using the latest supplementary_layout_script JSON.
    
    Args:
        blender_client: BlenderClient instance for communicating with Blender
        project_dir: Project directory path
        supplementary_layout_editor: JSONEditorComponent for the supplementary layout script
        
    Returns:
        dict: Result with success/error and data
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path"
        }
    
    # Set the save path for the supplementary layout editor
    supplementary_layout_save_path = os.path.join(project_dir, "supplementary_layout_script")
    supplementary_layout_editor.set_save_path(supplementary_layout_save_path)
    
    # Get the latest supplementary_layout_script JSON path
    json_filepath = supplementary_layout_editor.get_path_to_latest_json()
    
    if not json_filepath:
        return {
            "error": "⚠️ No supplementary_layout_script JSON found. Please generate a supplementary layout first (Step 5.2)."
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
    
    # Call import_supplementary_assets_to_all_scenes_json_input via BlenderClient
    try:
        response = blender_client.import_supplementary_assets_to_all_scenes_json_input(
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
                "message": "✅ All supplementary assets have been imported to all scenes successfully!",
                "json_filepath": json_filepath,
            }
        else:
            # Check for failed objects
            failed_objects = result.get("failed_objects", [])
            error_msg = result.get("error", "")
            
            if failed_objects:
                error_lines = ["⚠️ Some supplementary assets failed to import:"]
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
            "error": f"⚠️ Failed to import supplementary assets: {str(e)}"
        }
    finally:
        # Clean up temporary input file
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except OSError:
                pass


def finish_supplementary_layout_formulation(
    blender_client,
    project_dir,
    supplementary_layout_editor
):
    """Read transforms from Blender and update the supplementary layout JSON with actual values.
    
    Args:
        blender_client: BlenderClient instance for communicating with Blender
        project_dir: Project directory path
        supplementary_layout_editor: JSONEditorComponent for the supplementary layout script
        
    Returns:
        dict: Result with success/error and updated data
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path"
        }
    
    # Set the save path for the supplementary layout editor
    supplementary_layout_save_path = os.path.join(project_dir, "supplementary_layout_script")
    supplementary_layout_editor.set_save_path(supplementary_layout_save_path)
    
    # Get the latest supplementary_layout_script JSON path
    json_filepath = supplementary_layout_editor.get_path_to_latest_json()
    
    if not json_filepath:
        return {
            "error": "⚠️ No supplementary_layout_script JSON found. Please generate a supplementary layout first (Step 5.2)."
        }
    
    # Load the JSON data
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
        try:
            layout_data = make_paths_absolute(layout_data, project_dir)
        except Exception as e:
            logger.warning("Step 5: path conversion failed for supplementary layout JSON: %s", e)
    except Exception as e:
        return {
            "error": f"⚠️ Failed to load supplementary layout JSON: {str(e)}"
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
        if not switch_result.get("success", True):
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
    output_path = supplementary_layout_editor.save_json_data(layout_data)
    
    if not output_path:
        return {
            "error": "⚠️ Failed to save updated supplementary layout JSON"
        }
    
    # Build result message
    if update_errors:
        error_summary = "\n".join([f"  - {e}" for e in update_errors[:10]])
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


def show_loading_and_import_supplementary_assets(blender_client, supplementary_layout_editor, project_dir):
    """Show loading indicator and import all supplementary assets."""
    # Initial loading state
    loading_state = (
        gr.update(value="🔄 **Importing all supplementary assets to all scenes...** This may take a few minutes. Please wait.", visible=True),
        gr.update(visible=False),  # Hide import button
        gr.update(visible=False),  # Hide finish button
        gr.update(visible=False),  # Hide delete button
    )
    
    yield loading_state
    
    # Import assets
    result = import_supplementary_assets_to_all_scenes(
        blender_client,
        project_dir,
        supplementary_layout_editor
    )
    
    # Final state
    if result.get("success"):
        success_msg = result.get("message", "✅ All supplementary assets imported successfully!")
        success_msg += "\n\n📝 **Next step:** Check each scene in Blender and modify the supplementary assets transformations if needed. Then click the **'✅ Finish Supplementary Layout Formulation'** button when you're done."
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


def show_loading_and_finish_supplementary_layout(blender_client, supplementary_layout_editor, project_dir):
    """Show loading indicator and finish supplementary layout formulation."""
    # Initial loading state - hide editor while processing
    loading_outputs = supplementary_layout_editor.update_with_result(None)
    loading_state = (
        gr.update(value="🔄 **Reading transforms from Blender and updating supplementary layout...** Please wait.", visible=True),
        gr.update(visible=False),  # Hide import button
        gr.update(visible=False),  # Hide finish button
        gr.update(visible=False),  # Hide delete button
    )
    
    yield loading_outputs + loading_state
    
    # Finish supplementary layout formulation
    result = finish_supplementary_layout_formulation(
        blender_client,
        project_dir,
        supplementary_layout_editor
    )
    
    # Final state
    final_outputs = supplementary_layout_editor.update_with_result(result)
    
    if result.get("success"):
        updated_count = result.get("updated_count", 0)
        warning = result.get("warning", "")
        
        if warning:
            success_msg = warning
        else:
            success_msg = f"✅ **Supplementary layout formulation complete!** Updated {updated_count} asset transforms from Blender.\n\nThe updated supplementary layout has been saved as a new version."
        
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


def create_import_wrapper(blender_client, supplementary_layout_editor):
    """Factory function to create an import wrapper bound to specific components."""
    def import_wrapper(project_dir):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_import_supplementary_assets(blender_client, supplementary_layout_editor, project_dir):
            yield result
    return import_wrapper


def create_finish_wrapper(blender_client, supplementary_layout_editor):
    """Factory function to create a finish wrapper bound to specific components."""
    def finish_wrapper(project_dir):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_finish_supplementary_layout(blender_client, supplementary_layout_editor, project_dir):
            yield result
    return finish_wrapper


def delete_supplementary_assets_from_all_scenes(blender_client, project_dir, supplementary_layout_editor):
    """Delete supplementary assets from all scenes in Blender.
    
    Args:
        blender_client: BlenderClient instance for communicating with Blender
        project_dir: Project directory path
        supplementary_layout_editor: JSONEditorComponent for the supplementary layout script
        
    Returns:
        dict: Result with success/error
    """
    # Ensure MCP server is running
    success, message = blender_client.ensure_server_running()
    if not success:
        return {
            "error": f"⚠️ {message}"
        }
    
    # Set the save path and get the latest supplementary layout JSON
    supplementary_layout_save_path = os.path.join(project_dir, "supplementary_layout_script")
    supplementary_layout_editor.set_save_path(supplementary_layout_save_path)
    json_filepath = supplementary_layout_editor.get_path_to_latest_json()
    
    if not json_filepath:
        return {
            "error": "⚠️ No supplementary_layout_script JSON found. Nothing to delete."
        }
    
    # Load the JSON to get asset IDs
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
        try:
            layout_data = make_paths_absolute(layout_data, project_dir)
        except Exception as e:
            logger.warning("Step 5: path conversion failed for supplementary layout JSON: %s", e)
    except Exception as e:
        return {
            "error": f"⚠️ Failed to load supplementary layout JSON: {str(e)}"
        }
    
    deleted_count = 0
    delete_errors = []
    
    # Process each scene and delete supplementary assets
    scene_details = layout_data.get("scene_details", [])
    for scene_detail in scene_details:
        scene_id = scene_detail.get("scene_id")
        scene_name = f"Scene_{scene_id}"
        
        # Switch to the scene
        switch_response = blender_client.switch_or_create_scene(scene_name=scene_name)
        if switch_response.get("status") == "error":
            delete_errors.append(f"Scene {scene_id}: Failed to switch - {switch_response.get('message', 'Unknown error')}")
            continue
        
        # Get assets from scene_setup.layout_description.assets
        scene_setup = scene_detail.get("scene_setup", {})
        layout_description = scene_setup.get("layout_description", {})
        assets = layout_description.get("assets", [])
        
        for asset_info in assets:
            asset_id = asset_info.get("asset_id")
            
            try:
                # Delete the asset using BlenderClient
                response = blender_client.delete_asset(model_name=asset_id)
                
                if response.get("status") == "error":
                    delete_errors.append(f"Scene {scene_id}, Asset {asset_id}: {response.get('message', 'Unknown error')}")
                else:
                    result = response.get("result", {})
                    if result.get("success"):
                        deleted_count += 1
                    else:
                        error = result.get("error", "Unknown error")
                        delete_errors.append(f"Scene {scene_id}, Asset {asset_id}: {error}")
            except Exception as e:
                delete_errors.append(f"Scene {scene_id}, Asset {asset_id}: {str(e)}")
    
    if delete_errors:
        error_summary = "\n".join([f"  - {e}" for e in delete_errors[:10]])
        if len(delete_errors) > 10:
            error_summary += f"\n  - ... and {len(delete_errors) - 10} more errors"
        return {
            "success": True,
            "message": f"🗑️ Deleted {deleted_count} supplementary assets with some errors:\n{error_summary}",
        }
    else:
        return {
            "success": True,
            "message": f"🗑️ Successfully deleted {deleted_count} supplementary assets from all scenes.",
        }


def show_loading_and_delete_supplementary(blender_client, supplementary_layout_editor, project_dir):
    """Show loading indicator and delete supplementary assets."""
    # Initial loading state
    loading_state = (
        gr.update(value="🔄 **Deleting supplementary assets from all scenes...** Please wait.", visible=True),
        gr.update(visible=False),  # Hide import button
        gr.update(visible=False),  # Hide finish button
        gr.update(visible=False),  # Hide delete button
    )
    
    yield loading_state
    
    # Delete supplementary assets
    result = delete_supplementary_assets_from_all_scenes(blender_client, project_dir, supplementary_layout_editor)
    
    # Final state
    if result.get("success"):
        success_msg = result.get("message", "🗑️ Supplementary assets deleted successfully!")
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


def create_delete_wrapper(blender_client, supplementary_layout_editor):
    """Factory function to create a delete wrapper bound to specific components."""
    def delete_wrapper(project_dir):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_delete_supplementary(blender_client, supplementary_layout_editor, project_dir):
            yield result
    return delete_wrapper


def create_supplementary_layout_ui(reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider, project_dir, blender_client):
    """Create the Step 5: Formulate Supplementary Scene Spatial Layout by Set Dresser UI section.
    
    Args:
        reasoning_model: Gradio component for reasoning model selection
        anyllm_api_key: Gradio component for any-llm API key
        anyllm_api_base: Gradio component for any-llm API base URL
        project_dir: Gradio component for project directory
        blender_client: BlenderClient instance for communicating with Blender
    
    Returns:
        dict with UI components that may be needed by other parts of the app
    """
    gr.Markdown("## Step 5: Formulate Supplementary Scene Spatial Layout by Set Dresser")
    
    # =========================================================================
    # Step 5.1: Resize Supplementary Assets (moved from Step 4.5)
    # =========================================================================
    gr.Markdown("### Step 5.1: Resize Supplementary Assets")
    gr.Markdown("Resize supplementary 3D models in Blender based on estimated dimensions. **Requires Blender with BlenderMCPServer running.**")
    
    gr.Markdown("#### Select Models to Resize")
    gr.Markdown("Select specific models to re-resize (useful when some models failed). Leave empty or click 'Select All' to resize all models.")
    
    with gr.Row():
        resize_model_selection = gr.CheckboxGroup(
            choices=[],
            label="Select supplementary models to resize",
            info="Check the models you want to resize. If none selected, all models will be resized.",
            scale=3
        )
        with gr.Column(scale=1):
            load_resize_choices_btn = gr.Button("🔄 Load Models", variant="secondary", size="sm")
            select_all_resize_btn = gr.Button("☑️ Select All", variant="secondary", size="sm")
    
    resize_selection_status = gr.Markdown(value="No models loaded. Click 'Load Models' to load available models.", visible=True)
    
    with gr.Row():
        resize_btn = gr.Button("📐 Resize Supplementary Assets", variant="primary", size="lg", scale=1)
        refine_btn = gr.Button("🪞 Refine Size Estimation", variant="secondary", size="lg", scale=1)

    gr.Markdown(
        "ℹ️ **Refine Size Estimation** audits the current resized dimensions of every "
        "supplementary asset with a vision LLM (one model at a time, in parallel). "
        "Models with clearly abnormal width/depth/height are re-estimated and "
        "re-resized in Blender."
    )

    # Loading status indicator for resize assets (hidden by default)
    resize_loading_status = gr.Markdown(value="", visible=False)
    
    # Create JSON editor component for resized models results
    resized_editor = JSONEditorComponent(
        label="Resized Supplementary Assets Result",
        visible_initially=False,
        file_basename="resized_supplementary_model",
        use_version_control=False,  # Save as resized_supplementary_model.json (overwrites)
        json_root_keys_list=["asset_sheet"],
        title="Step 5.1"
    )
    
    # Wire up the Resume button with project_dir input
    resized_editor.setup_resume_with_project_dir(project_dir, subfolder="resized_supplementary_assets")
    
    # Create wrapper function for resize assets
    resize_wrapper = create_resize_supplementary_wrapper(resized_editor, blender_client)
    
    # Handler for Load Models button (for resize)
    def load_resize_choices_handler(project_dir_val):
        choices = load_resize_supplementary_model_choices(project_dir_val)
        if choices:
            status_msg = f"**{len(choices)} model(s) available.** Select models to re-resize, or leave empty to resize all."
        else:
            status_msg = "⚠️ No models found. Please complete Step 4.4 (Estimate Dimensions) first."
        return gr.update(choices=choices, value=[]), gr.update(value=status_msg)
    
    load_resize_choices_btn.click(
        fn=load_resize_choices_handler,
        inputs=[project_dir],
        outputs=[resize_model_selection, resize_selection_status]
    )
    
    # Handler for Select All button (for resize)
    def select_all_resize_handler(project_dir_val):
        choices = load_resize_supplementary_model_choices(project_dir_val)
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
    refine_supplementary_wrapper = create_refine_supplementary_wrapper(resized_editor, blender_client)
    refine_btn.click(
        fn=refine_supplementary_wrapper,
        inputs=[project_dir, reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider],
        outputs=resized_editor.get_output_components() + [resize_loading_status, refine_btn, resize_btn],
    )

    # =========================================================================
    # Step 5.2: Generate Supplementary Spatial Layout for Each Scene (was Step 5.1)
    # =========================================================================
    gr.Markdown("### Step 5.2: Generate Supplementary Spatial Layout for Each Scene")
    gr.Markdown("Generate 3D spatial layout for supplementary decorative assets based on the existing scene layout and resized supplementary assets.")
    
    # Input info
    gr.Markdown("**Input:** `project_dir/layout_script/layout_script_v{N}.json` (latest), `project_dir/resized_supplementary_assets/resized_supplementary_model.json`")
    gr.Markdown("**Output:** `project_dir/supplementary_layout_script/supplementary_layout_script_v{N}.json`")
    
    # Generate button
    generate_supplementary_layout_btn = gr.Button("🎨 Generate Supplementary Layout", variant="primary", size="lg")
    
    # Loading/status indicator (hidden by default)
    supplementary_layout_status = gr.Markdown(value="", visible=False)
    
    # Create JSON editor component
    supplementary_layout_editor = JSONEditorComponent(
        label="Supplementary Layout JSON",
        visible_initially=False,
        file_basename="supplementary_layout_script",
        json_root_keys_list=["scene_details"],
        title="Step 5.2"
    )
    
    # Wire up the Resume button with project_dir input
    supplementary_layout_editor.setup_resume_with_project_dir(project_dir, subfolder="supplementary_layout_script")
    
    # Create wrapper function for the generator
    generate_wrapper = create_generate_wrapper(supplementary_layout_editor)
    
    # Generate button click handler
    generate_supplementary_layout_btn.click(
        fn=generate_wrapper,
        inputs=[
            reasoning_model,
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
            project_dir,
        ],
        outputs=supplementary_layout_editor.get_output_components() + [supplementary_layout_status],
        concurrency_limit=None,
    )
    
    # ============================================================================
    # Step 5.3: Organize Supplementary Assets Layout for Each Scene (was Step 5.2)
    # ============================================================================
    gr.Markdown("### Step 5.3: Organize Supplementary Assets Layout for Each Scene")
    gr.Markdown("Import supplementary assets to scenes in Blender, adjust their positions manually if needed. Then save the final layout to supplementary_layout_script_v{N}.json by clicking the '✅ Finish Supplementary Layout Formulation' button. Use the '🗑️ Delete Supplementary Assets' button to remove them and restart.")

    # Status indicator for Step 5.2
    organize_supplementary_status = gr.Markdown(value="", visible=False)
    
    with gr.Row():
        # Import supplementary assets button
        import_supplementary_btn = gr.Button(
            "🚀 Import Supplementary Assets to All Scenes",
            variant="primary",
            size="lg"
        )
        
        # Finish supplementary layout button
        finish_supplementary_btn = gr.Button(
            "✅ Finish Supplementary Layout Formulation",
            variant="secondary",
            size="lg",
            visible=True
        )
        
        # Delete supplementary assets button
        delete_supplementary_btn = gr.Button(
            "🗑️ Delete Supplementary Assets",
            variant="stop",
            size="lg"
        )
    
    # Create wrapper functions for the buttons
    import_wrapper = create_import_wrapper(blender_client, supplementary_layout_editor)
    finish_wrapper = create_finish_wrapper(blender_client, supplementary_layout_editor)
    delete_wrapper = create_delete_wrapper(blender_client, supplementary_layout_editor)
    
    # Import supplementary assets button click handler
    import_supplementary_btn.click(
        fn=import_wrapper,
        inputs=[project_dir],
        outputs=[
            organize_supplementary_status,
            import_supplementary_btn,
            finish_supplementary_btn,
            delete_supplementary_btn,
        ],
    )
    
    # Finish supplementary layout button click handler
    finish_supplementary_btn.click(
        fn=finish_wrapper,
        inputs=[project_dir],
        outputs=supplementary_layout_editor.get_output_components() + [
            organize_supplementary_status,
            import_supplementary_btn,
            finish_supplementary_btn,
            delete_supplementary_btn,
        ],
    )
    
    # Delete supplementary assets button click handler
    delete_supplementary_btn.click(
        fn=delete_wrapper,
        inputs=[project_dir],
        outputs=[
            organize_supplementary_status,
            import_supplementary_btn,
            finish_supplementary_btn,
            delete_supplementary_btn,
        ],
    )
    
    return {
        "supplementary_layout_editor": supplementary_layout_editor,
        "resized_editor": resized_editor,
    }
