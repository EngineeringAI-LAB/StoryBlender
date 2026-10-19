import os
import json
import logging
import gradio as gr
from .json_editor import JSONEditorComponent
from .blender_client import BlenderClient
from ..operators.layout_artist_operators.generate_additional_camera_instruction import generate_additional_camera_instruction
from .path_utils import make_paths_absolute, make_paths_relative

logger = logging.getLogger(__name__)


def get_latest_animated_models_json_path(project_dir):
    """Get the path to the latest animated_models_v{num}.json file.
    
    Args:
        project_dir: The project directory path
        
    Returns:
        str: Path to the latest animated_models JSON, or None if not found
    """
    animated_models_dir = os.path.join(project_dir, "animated_models")
    
    if not os.path.exists(animated_models_dir):
        return None
    
    latest_version = 0
    latest_path = None
    
    for filename in os.listdir(animated_models_dir):
        if filename.startswith("animated_models_v") and filename.endswith(".json"):
            try:
                version_str = filename[len("animated_models_v"):-5]
                version = int(version_str)
                if version > latest_version:
                    latest_version = version
                    latest_path = os.path.join(animated_models_dir, filename)
            except ValueError:
                continue
    
    return latest_path


def get_latest_camera_instructions_json_path(project_dir):
    """Get the path to the latest camera_instructions_v{num}.json file.
    
    Args:
        project_dir: The project directory path
        
    Returns:
        str: Path to the latest camera_instructions JSON, or None if not found
    """
    camera_blocking_dir = os.path.join(project_dir, "camera_blocking")
    
    if not os.path.exists(camera_blocking_dir):
        return None
    
    latest_version = 0
    latest_path = None
    
    for filename in os.listdir(camera_blocking_dir):
        if filename.startswith("camera_instructions_v") and filename.endswith(".json"):
            try:
                version_str = filename[len("camera_instructions_v"):-5]
                version = int(version_str)
                if version > latest_version:
                    latest_version = version
                    latest_path = os.path.join(camera_blocking_dir, filename)
            except ValueError:
                continue
    
    return latest_path


def get_next_camera_instructions_version(project_dir):
    """Get the next version number for camera_instructions JSON.
    
    Args:
        project_dir: The project directory path
        
    Returns:
        int: Next version number
    """
    camera_blocking_dir = os.path.join(project_dir, "camera_blocking")
    
    if not os.path.exists(camera_blocking_dir):
        return 1
    
    latest_version = 0
    
    for filename in os.listdir(camera_blocking_dir):
        if filename.startswith("camera_instructions_v") and filename.endswith(".json"):
            try:
                version_str = filename[len("camera_instructions_v"):-5]
                version = int(version_str)
                if version > latest_version:
                    latest_version = version
            except ValueError:
                continue
    
    return latest_version + 1


def get_latest_camera_blocking_json_path(project_dir):
    """Get the path to the latest camera_blocking_v{num}.json file.
    
    Args:
        project_dir: The project directory path
        
    Returns:
        str: Path to the latest camera_blocking JSON, or None if not found
    """
    camera_blocking_dir = os.path.join(project_dir, "camera_blocking")
    
    if not os.path.exists(camera_blocking_dir):
        return None
    
    latest_version = 0
    latest_path = None
    
    for filename in os.listdir(camera_blocking_dir):
        if filename.startswith("camera_blocking_v") and filename.endswith(".json"):
            try:
                version_str = filename[len("camera_blocking_v"):-5]
                version = int(version_str)
                if version > latest_version:
                    latest_version = version
                    latest_path = os.path.join(camera_blocking_dir, filename)
            except ValueError:
                continue
    
    checkpoint = os.path.join(camera_blocking_dir, "camera_blocking_in_progress.json")
    if os.path.isfile(checkpoint) and (
        latest_path is None or os.path.getmtime(checkpoint) > os.path.getmtime(latest_path)
    ):
        return checkpoint
    return latest_path


def get_next_camera_blocking_version(project_dir):
    """Get the next version number for camera_blocking JSON.
    
    Args:
        project_dir: The project directory path
        
    Returns:
        int: Next version number
    """
    camera_blocking_dir = os.path.join(project_dir, "camera_blocking")
    
    if not os.path.exists(camera_blocking_dir):
        return 1
    
    latest_version = 0
    
    for filename in os.listdir(camera_blocking_dir):
        if filename.startswith("camera_blocking_v") and filename.endswith(".json"):
            try:
                version_str = filename[len("camera_blocking_v"):-5]
                version = int(version_str)
                if version > latest_version:
                    latest_version = version
            except ValueError:
                continue
    
    return latest_version + 1


def load_camera_names_from_json(project_dir):
    """Load camera names from the latest camera_blocking JSON, falling back to camera_instructions.
    
    Args:
        project_dir: Project directory path
        
    Returns:
        list: List of camera names
    """
    if not project_dir or not os.path.isabs(project_dir):
        return []
    
    # Try camera_blocking first, then fall back to camera_instructions
    json_path = get_latest_camera_blocking_json_path(project_dir)
    if not json_path or not os.path.exists(json_path):
        json_path = get_latest_camera_instructions_json_path(project_dir)
    
    if not json_path or not os.path.exists(json_path):
        return []
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning("Step 10: path conversion failed for camera JSON: %s", e)
        
        camera_names = []
        shot_details = data if isinstance(data, list) else data.get("shot_details", [])
        
        for shot in shot_details:
            # Main camera
            main_camera = shot.get("camera_instruction", {})
            if main_camera.get("camera_name"):
                camera_names.append(main_camera["camera_name"])
            
            # Additional cameras
            additional_cameras = shot.get("additional_camera_instructions", [])
            for cam in additional_cameras:
                if cam.get("camera_name"):
                    camera_names.append(cam["camera_name"])
        
        return camera_names
    except Exception:
        return []


def load_camera_previews_from_json(project_dir):
    """Load camera names and their preview image paths from latest camera_blocking_v{num}.json.
    
    Args:
        project_dir: Project directory path
        
    Returns:
        tuple: (list of (camera_name, preview_path) tuples, error message or None)
    """
    if not project_dir or not os.path.isabs(project_dir):
        return [], "⚠️ Please set a valid project directory first."
    
    json_path = get_latest_camera_blocking_json_path(project_dir)
    
    if not json_path or not os.path.exists(json_path):
        return [], "⚠️ No camera_blocking JSON found. Please perform camera blocking first."
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning("Step 10: path conversion failed for camera_blocking JSON: %s", e)
        
        cameras = []
        shot_details = data if isinstance(data, list) else data.get("shot_details", [])
        
        for shot in shot_details:
            # Main camera
            main_camera = shot.get("camera_instruction", {})
            if main_camera.get("camera_name"):
                camera_name = main_camera["camera_name"]
                preview_path = main_camera.get("camera_preview_image")
                cameras.append((camera_name, preview_path))
            
            # Additional cameras
            additional_cameras = shot.get("additional_camera_instructions", [])
            for cam in additional_cameras:
                if cam.get("camera_name"):
                    camera_name = cam["camera_name"]
                    preview_path = cam.get("camera_preview_image")
                    cameras.append((camera_name, preview_path))
        
        if not cameras:
            return [], "⚠️ No cameras found in camera_blocking.json."
        
        return cameras, None
    except Exception as e:
        return [], f"⚠️ Failed to load camera_blocking.json: {str(e)}"


def display_camera_previews(project_dir):
    """Load and display camera previews.
    
    Returns updates for: camera_viewer, camera_status, camera_buttons, camera_viewer_container, 
                        cameras_state, selected_cameras
    """
    cameras, error = load_camera_previews_from_json(project_dir)
    
    if error:
        return (
            gr.update(value=None),  # camera_viewer
            gr.update(value=error, visible=True),  # camera_status
            gr.update(samples=[], visible=False),  # camera_buttons (Dataset)
            gr.update(visible=False),  # camera_viewer_container
            [],  # cameras_state
            [],  # selected_cameras
        )
    
    # Show the first camera by default
    first_preview_path = None
    first_camera_name = None
    for camera_name, path in cameras:
        if path and os.path.exists(path):
            first_preview_path = path
            first_camera_name = camera_name
            break
    
    if not first_preview_path and cameras:
        first_camera_name = cameras[0][0]
    
    # Format for Dataset: list of lists with camera_name
    camera_ids = [[c[0]] for c in cameras]
    
    status_text = f"Showing: **{first_camera_name}** ({len(cameras)} cameras available)" if first_camera_name else "No cameras available"
    
    return (
        gr.update(value=first_preview_path),  # camera_viewer
        gr.update(value=status_text, visible=True),  # camera_status
        gr.update(samples=camera_ids, visible=True),  # camera_buttons (Dataset)
        gr.update(visible=True),  # camera_viewer_container
        cameras,  # cameras_state: list of (name, path) tuples
        [],  # selected_cameras: initially empty
    )


def select_camera(evt: gr.SelectData, cameras_state):
    """Handle camera selection from the dataset buttons.
    
    Returns updates for: camera_viewer, camera_status
    """
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not cameras_state or idx >= len(cameras_state):
        return gr.update(), gr.update()
    
    selected_camera = cameras_state[idx]
    camera_name, preview_path = selected_camera
    
    return (
        gr.update(value=preview_path if preview_path and os.path.exists(preview_path) else None),  # camera_viewer
        gr.update(value=f"Showing: **{camera_name}** ({len(cameras_state)} cameras available)")  # camera_status
    )


def toggle_camera_selection(camera_name, current_selection):
    """Toggle a camera's selection for reperform/resume.
    
    Returns: updated selection list
    """
    if camera_name in current_selection:
        return [c for c in current_selection if c != camera_name]
    else:
        return current_selection + [camera_name]


def update_camera_selection_display(selected_cameras, cameras_state):
    """Update the display to show which cameras are selected.
    
    Returns: status text showing selection
    """
    if not selected_cameras:
        return "No cameras selected for reperform/resume."
    return f"**Selected cameras ({len(selected_cameras)}):** {', '.join(selected_cameras)}"


def generate_camera_instructions(
    project_dir: str,
    anyllm_api_key: str,
    anyllm_api_base: str,
    reasoning_model: str,
):
    """Generate additional camera instructions from animated_models JSON.
    
    Args:
        project_dir: Project directory path
        anyllm_api_key: API key for any-llm
        anyllm_api_base: API base URL for any-llm
        reasoning_model: Reasoning model for generation
        
    Returns:
        dict: Result with success/error and data
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {"error": "⚠️ Project directory must be an absolute path"}
    
    # Get latest animated_models JSON path
    input_json_path = get_latest_animated_models_json_path(project_dir)
    if not input_json_path:
        return {"error": "⚠️ No animated_models JSON found. Please complete animation step first (Step 9)."}
    
    # Create output directory
    output_dir = os.path.join(project_dir, "camera_blocking")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set API base to None if empty string
    api_base = anyllm_api_base if anyllm_api_base and anyllm_api_base.strip() else None
    
    try:
        # Load the storyboard script
        with open(input_json_path, 'r') as f:
            storyboard_script = json.load(f)
        try:
            storyboard_script = make_paths_absolute(storyboard_script, project_dir)
        except Exception as e:
            logger.warning("Step 10: path conversion failed for animated_models JSON: %s", e)
        
        # Generate additional camera instructions
        result = generate_additional_camera_instruction(
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=api_base,
            storyboard_script=storyboard_script,
            reasoning_model=reasoning_model,
            reasoning_effort="high",
            max_retries=3
        )
        
        if result is None:
            return {"error": "⚠️ Failed to generate additional camera instructions"}
        
        # Save to versioned file
        version = get_next_camera_instructions_version(project_dir)
        output_filename = f"camera_instructions_v{version}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            save_data = make_paths_relative(result, project_dir)
        except Exception as e:
            logger.warning("Step 10: path conversion failed on save: %s", e)
            save_data = result
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)
        
        return {
            "success": True,
            "data": result,
            "output_path": output_path,
            "version": version,
        }
    except Exception as e:
        return {"error": f"⚠️ Failed to generate camera instructions: {str(e)}"}


def perform_camera_blocking(
    blender_client: BlenderClient,
    project_dir: str,
    camera_type: str,
    max_additional_cameras: int,
    start_frame: int,
    end_frame: int,
    max_adjustment_rounds: int,
    vision_model: str,
    anyllm_api_key: str,
    anyllm_api_base: str,
    anyllm_provider: str = "gemini",
    camera_name_filter: list = None,
):
    """Perform camera blocking by calling camera_operator in Blender.
    
    Args:
        blender_client: BlenderClient instance
        project_dir: Project directory path
        camera_type: 'director', 'additional', or 'all'
        max_additional_cameras: Maximum additional cameras per shot
        start_frame: Start frame for camera animation
        end_frame: End frame for camera animation
        max_adjustment_rounds: Maximum LLM adjustment rounds
        vision_model: Vision model for LLM
        anyllm_api_key: API key for any-llm
        anyllm_api_base: API base URL for any-llm
        camera_name_filter: List of camera names to place (None = all)
        
    Returns:
        dict: Result with success/error and data
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {"error": "⚠️ Project directory must be an absolute path"}
    
    # Get latest camera_blocking JSON, fall back to camera_instructions
    input_json_path = get_latest_camera_blocking_json_path(project_dir)
    if not input_json_path:
        input_json_path = get_latest_camera_instructions_json_path(project_dir)
    if not input_json_path:
        return {"error": "⚠️ No camera_blocking or camera_instructions JSON found. Please complete Step 10.1 first."}
    
    # Create preview save directory
    preview_save_dir = os.path.join(project_dir, "camera_blocking")
    os.makedirs(preview_save_dir, exist_ok=True)
    
    # Set API base to None if empty string
    api_base = anyllm_api_base if anyllm_api_base and anyllm_api_base.strip() else None
    
    # Load JSON and convert relative paths to absolute for Blender
    temp_input_path = None
    try:
        with open(input_json_path, 'r') as f:
            input_data = json.load(f)
        input_data = make_paths_absolute(input_data, project_dir)
        
        temp_input_path = input_json_path + ".tmp"
        with open(temp_input_path, 'w') as f:
            json.dump(input_data, f, indent=2)
    except Exception as e:
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        return {"error": f"⚠️ Failed to prepare input for Blender: {str(e)}"}
    
    try:
        # Call camera_operator via client method
        response = blender_client.camera_operator(
            path_to_input_json=temp_input_path,
            vision_model=vision_model,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=api_base,
            anyllm_provider=anyllm_provider,
            camera_type=camera_type,
            max_additional_cameras=max_additional_cameras,
            camera_name_filter=camera_name_filter,
            start_frame=start_frame,
            end_frame=end_frame,
            max_adjustment_rounds=max_adjustment_rounds,
            preview_image_save_dir=preview_save_dir,
        )
        
        if response.get("status") == "error":
            return {"error": f"⚠️ Camera blocking failed: {response.get('message', 'Unknown error')}"}
        
        result = response.get("result", response)
        
        if result.get("success"):
            # Save the result to versioned camera_blocking file
            version = get_next_camera_blocking_version(project_dir)
            output_filename = f"camera_blocking_v{version}.json"
            output_path = os.path.join(preview_save_dir, output_filename)
            shot_details = result.get("shot_details", [])
            
            try:
                save_data = make_paths_relative(shot_details, project_dir)
            except Exception as e:
                logger.warning("Step 10: path conversion failed on save: %s", e)
                save_data = shot_details
            with open(output_path, "w") as f:
                json.dump(save_data, f, indent=2)
            
            return {
                "success": True,
                "data": shot_details,
                "output_path": output_path,
                "cameras_placed": result.get("cameras_placed", []),
                "cameras_failed": result.get("cameras_failed", []),
            }
        else:
            error_msg = result.get("error", "Unknown error during camera blocking")
            return {"error": f"⚠️ Camera blocking failed: {error_msg}"}
    except Exception as e:
        return {"error": f"⚠️ Camera blocking failed: {str(e)}"}
    finally:
        # Clean up temporary input file
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except OSError:
                pass


def resume_camera_blocking(
    blender_client: BlenderClient,
    project_dir: str,
    camera_name_filter: list = None,
):
    """Resume camera blocking by calling resume_camera_operator in Blender.
    
    Args:
        blender_client: BlenderClient instance
        project_dir: Project directory path
        camera_name_filter: List of camera names to resume (None = all)
        
    Returns:
        dict: Result with success/error
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {"error": "⚠️ Project directory must be an absolute path"}
    
    # Get latest camera_blocking JSON (output from perform_camera_blocking)
    input_json_path = get_latest_camera_blocking_json_path(project_dir)
    if not input_json_path or not os.path.exists(input_json_path):
        return {"error": "⚠️ No camera_blocking JSON found. Please complete Step 10.2 first."}
    
    # Load JSON and convert relative paths to absolute for Blender
    temp_input_path = None
    try:
        with open(input_json_path, 'r') as f:
            input_data = json.load(f)
        input_data = make_paths_absolute(input_data, project_dir)
        
        temp_input_path = input_json_path + ".tmp"
        with open(temp_input_path, 'w') as f:
            json.dump(input_data, f, indent=2)
    except Exception as e:
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        return {"error": f"⚠️ Failed to prepare input for Blender: {str(e)}"}
    
    try:
        # Call resume_camera_operator via client method
        response = blender_client.resume_camera_operator(
            path_to_input_json=temp_input_path,
            camera_name_filter=camera_name_filter,
        )
        
        if response.get("status") == "error":
            return {"error": f"⚠️ Camera resume failed: {response.get('message', 'Unknown error')}"}
        
        result = response.get("result", response)
        
        if result.get("success"):
            return {
                "success": True,
                "cameras_resumed": result.get("cameras_resumed", []),
                "cameras_failed": result.get("cameras_failed", []),
            }
        else:
            error_msg = result.get("error", "Unknown error during camera resume")
            return {"error": f"⚠️ Camera resume failed: {error_msg}"}
    except Exception as e:
        return {"error": f"⚠️ Camera resume failed: {str(e)}"}
    finally:
        # Clean up temporary input file
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except OSError:
                pass


def finish_camera_blocking(
    blender_client: BlenderClient,
    project_dir: str,
):
    """Read camera info from Blender and update the JSON with actual values.
    
    For each camera in the JSON, reads transform and parameters from Blender
    and updates the JSON, then saves a new version.
    
    Args:
        blender_client: BlenderClient instance
        project_dir: Project directory path
        
    Returns:
        dict: Result with success/error and updated data
    """
    # Validate project directory
    if not project_dir or not os.path.isabs(project_dir):
        return {"error": "⚠️ Project directory must be an absolute path"}
    
    # Get latest camera_blocking JSON (output from perform_camera_blocking)
    input_json_path = get_latest_camera_blocking_json_path(project_dir)
    if not input_json_path or not os.path.exists(input_json_path):
        return {"error": "⚠️ No camera_blocking JSON found. Please complete Step 10.2 first."}
    
    # Load the JSON data
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {"error": f"⚠️ Failed to load JSON: {str(e)}"}
    
    # Ensure MCP server is running
    success, message = blender_client.ensure_server_running()
    if not success:
        return {"error": f"⚠️ {message}"}
    
    # Get shot_details from the data
    shot_details = data.get("shot_details", data) if isinstance(data, dict) else data
    if not isinstance(shot_details, list):
        shot_details = [shot_details]
    
    update_errors = []
    updated_count = 0
    skipped_count = 0
    
    for shot in shot_details:
        scene_id = shot.get("scene_id")
        shot_id = shot.get("shot_id")
        scene_name = f"Scene_{scene_id}_Shot_{shot_id}"
        
        # Process main camera instruction
        camera_instruction = shot.get("camera_instruction", {})
        if camera_instruction:
            camera_name = camera_instruction.get("camera_name")
            is_placed = camera_instruction.get("placement_success") is not False and (
                camera_instruction.get("placement_success") or camera_instruction.get("start_transform")
            )
            if camera_name and is_placed:
                response = blender_client.get_camera_info(
                    scene_name=scene_name,
                    camera_name=camera_name
                )
                
                if response.get("status") == "error":
                    update_errors.append(f"{camera_name}: {response.get('message', 'Unknown error')}")
                else:
                    result = response.get("result", response)
                    if result.get("success"):
                        # Update camera instruction with info from Blender
                        camera_instruction["camera_parameters"] = result.get("camera_parameters")
                        camera_instruction["start_transform"] = result.get("start_transform")
                        camera_instruction["end_transform"] = result.get("end_transform")
                        camera_instruction["start_frame"] = result.get("start_frame")
                        camera_instruction["end_frame"] = result.get("end_frame")
                        camera_instruction["is_animated"] = result.get("is_animated")
                        camera_instruction["dof_applied"] = result.get("dof_applied")
                        camera_instruction["focus_distance"] = result.get("focus_distance")
                        updated_count += 1
                    else:
                        # Camera not found in scene - might not have been placed yet
                        skipped_count += 1
        
        # Process additional camera instructions
        additional_cameras = shot.get("additional_camera_instructions", [])
        for cam_instruction in additional_cameras:
            camera_name = cam_instruction.get("camera_name")
            is_placed = cam_instruction.get("placement_success") is not False and (
                cam_instruction.get("placement_success") or cam_instruction.get("start_transform")
            )
            if camera_name and is_placed:
                response = blender_client.get_camera_info(
                    scene_name=scene_name,
                    camera_name=camera_name
                )
                
                if response.get("status") == "error":
                    update_errors.append(f"{camera_name}: {response.get('message', 'Unknown error')}")
                else:
                    result = response.get("result", response)
                    if result.get("success"):
                        # Update camera instruction with info from Blender
                        cam_instruction["camera_parameters"] = result.get("camera_parameters")
                        cam_instruction["start_transform"] = result.get("start_transform")
                        cam_instruction["end_transform"] = result.get("end_transform")
                        cam_instruction["start_frame"] = result.get("start_frame")
                        cam_instruction["end_frame"] = result.get("end_frame")
                        cam_instruction["is_animated"] = result.get("is_animated")
                        cam_instruction["dof_applied"] = result.get("dof_applied")
                        cam_instruction["focus_distance"] = result.get("focus_distance")
                        updated_count += 1
                    else:
                        # Camera not found - might not have been placed yet
                        skipped_count += 1
    
    # Save the updated JSON as a new version
    output_dir = os.path.join(project_dir, "camera_blocking")
    os.makedirs(output_dir, exist_ok=True)
    version = get_next_camera_blocking_version(project_dir)
    output_filename = f"camera_blocking_v{version}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # Preserve original structure
    if isinstance(data, dict) and "shot_details" in data:
        data["shot_details"] = shot_details
        output_data = data
    else:
        output_data = shot_details
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
    except Exception as e:
        return {"error": f"⚠️ Failed to save JSON: {str(e)}"}
    
    # Build result
    if update_errors:
        error_summary = "\n".join([f"  - {e}" for e in update_errors[:10]])
        if len(update_errors) > 10:
            error_summary += f"\n  - ... and {len(update_errors) - 10} more errors"
        
        return {
            "success": True,
            "data": output_data,
            "output_path": output_path,
            "version": version,
            "warning": f"⚠️ Updated {updated_count} cameras, skipped {skipped_count}, errors:\n{error_summary}",
            "updated_count": updated_count,
            "skipped_count": skipped_count,
            "error_count": len(update_errors),
        }
    
    return {
        "success": True,
        "data": output_data,
        "output_path": output_path,
        "version": version,
        "updated_count": updated_count,
        "skipped_count": skipped_count,
    }


def show_loading_and_finish_camera_blocking(
    blender_client,
    project_dir,
):
    """Show loading indicator and finish camera blocking."""
    loading_msg = "🔄 **Reading camera info from Blender...** Please wait."
    
    yield (
        gr.update(value=loading_msg, visible=True),  # loading_status
        gr.update(visible=False),  # finish_btn
    )
    
    # Finish camera blocking
    result = finish_camera_blocking(blender_client, project_dir)
    
    # Final state
    if result.get("success"):
        version = result.get("version", "?")
        updated = result.get("updated_count", 0)
        skipped = result.get("skipped_count", 0)
        warning = result.get("warning", "")
        if warning:
            success_msg = warning
        else:
            success_msg = f"✅ **Camera blocking finished!** Updated {updated} cameras, saved to camera_instructions_v{version}.json"
            if skipped:
                success_msg += f" ({skipped} skipped - not in scene)"
    else:
        success_msg = result.get("error", "")
    
    yield (
        gr.update(value=success_msg, visible=True),  # loading_status
        gr.update(visible=True),  # finish_btn
    )


def show_loading_and_perform_camera_blocking(
    blender_client,
    project_dir,
    camera_type,
    max_additional_cameras,
    start_frame,
    end_frame,
    max_adjustment_rounds,
    vision_model,
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider="gemini",
    camera_name_filter=None,
):
    """Show loading indicator and perform camera blocking."""
    # Initial loading state
    if camera_name_filter:
        loading_msg = f"🔄 **Reperforming camera blocking for {len(camera_name_filter)} camera(s)...** This may take several minutes."
    else:
        loading_msg = "🔄 **Performing camera blocking...** This may take several minutes per camera."
    
    yield (
        gr.update(value=loading_msg, visible=True),  # loading_status
        gr.update(visible=False),  # perform_btn
        gr.update(visible=False),  # reperform_btn
    )
    
    # Perform camera blocking
    result = perform_camera_blocking(
        blender_client,
        project_dir,
        camera_type,
        int(max_additional_cameras),
        int(start_frame),
        int(end_frame),
        int(max_adjustment_rounds),
        vision_model,
        anyllm_api_key,
        anyllm_api_base,
        anyllm_provider,
        camera_name_filter,
    )
    
    # Final state
    if result.get("success"):
        placed = result.get("cameras_placed", [])
        failed = result.get("cameras_failed", [])
        success_msg = f"✅ **Camera blocking complete!** {len(placed)} camera(s) placed."
        if failed:
            success_msg += f" {len(failed)} failed."
    else:
        success_msg = result.get("error", "")
    
    yield (
        gr.update(value=success_msg, visible=True),  # loading_status
        gr.update(visible=True),  # perform_btn
        gr.update(visible=True),  # reperform_btn
    )


def show_loading_and_resume_cameras(
    blender_client,
    project_dir,
    camera_name_filter=None,
):
    """Show loading indicator and resume cameras."""
    # Initial loading state
    if camera_name_filter:
        loading_msg = f"🔄 **Resuming {len(camera_name_filter)} camera(s)...** Please wait."
    else:
        loading_msg = "🔄 **Resuming all cameras...** Please wait."
    
    yield (
        gr.update(value=loading_msg, visible=True),  # loading_status
        gr.update(visible=False),  # resume_btn
    )
    
    # Resume cameras
    result = resume_camera_blocking(
        blender_client,
        project_dir,
        camera_name_filter,
    )
    
    # Final state
    if result.get("success"):
        resumed = result.get("cameras_resumed", [])
        failed = result.get("cameras_failed", [])
        success_msg = f"✅ **Cameras resumed!** {len(resumed)} camera(s) recreated."
        if failed:
            success_msg += f" {len(failed)} failed."
    else:
        success_msg = result.get("error", "")
    
    yield (
        gr.update(value=success_msg, visible=True),  # loading_status
        gr.update(visible=True),  # resume_btn
    )


def show_loading_and_generate_instructions(
    project_dir,
    anyllm_api_key,
    anyllm_api_base,
    reasoning_model,
):
    """Show loading indicator and generate camera instructions."""
    # Initial loading state
    loading_msg = "🔄 **Generating additional camera instructions...** This may take a few minutes."
    
    yield (
        gr.update(value=loading_msg, visible=True),  # loading_status
        gr.update(visible=False),  # generate_btn
    )
    
    # Generate camera instructions
    result = generate_camera_instructions(
        project_dir,
        anyllm_api_key,
        anyllm_api_base,
        reasoning_model,
    )
    
    # Final state
    if result.get("success"):
        version = result.get("version", "?")
        success_msg = f"✅ **Camera instructions generated!** Saved to camera_instructions_v{version}.json"
    else:
        success_msg = result.get("error", "")
    
    yield (
        gr.update(value=success_msg, visible=True),  # loading_status
        gr.update(visible=True),  # generate_btn
    )


def create_camera_blocking_ui(
    vision_model,
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
    project_dir,
    blender_client: BlenderClient,
    reasoning_model=None
):
    """Create the camera blocking UI step."""
    
    gr.Markdown("## Step 10: Camera Blocking by Layout Artist")
    
    # ============================================================================
    # Step 10.1: Generate Additional Camera Instructions
    # ============================================================================
    gr.Markdown("### Step 10.1: Generate Additional Camera Instructions")
    gr.Markdown("Generate additional camera angles for each shot based on the storyboard.")
    
    # Step 10.1 Loading status
    step_10_1_status = gr.Markdown(value="", visible=False)
    
    # Generate button
    generate_instructions_btn = gr.Button("🎥 Generate Camera Instructions", variant="primary")
    
    # JSON Editor for Step 10.1 results
    step_10_1_editor = JSONEditorComponent(
        label="Camera Instructions",
        visible_initially=False,
        file_basename="camera_instructions",
        use_version_control=True,
        json_root_keys_list=["shot_details"],
        title="Step 10.1"
    )
    step_10_1_editor.setup_resume_with_project_dir(project_dir, subfolder="camera_blocking")
    
    # Step 10.1 event handler
    def generate_wrapper(proj_dir, api_key, api_base, r_model):
        for result in show_loading_and_generate_instructions(proj_dir, api_key, api_base, r_model):
            yield result
    
    generate_instructions_btn.click(
        fn=generate_wrapper,
        inputs=[project_dir, anyllm_api_key, anyllm_api_base, reasoning_model],
        outputs=[step_10_1_status, generate_instructions_btn]
    ).then(
        fn=step_10_1_editor._handle_resume,
        inputs=[project_dir],
        outputs=step_10_1_editor._get_resume_outputs()
    )
    
    # ============================================================================
    # Step 10.2: Camera Blocking
    # ============================================================================
    gr.Markdown("### Step 10.2: Camera Blocking")
    gr.Markdown("Place and animate cameras in Blender scenes based on camera instructions. Click ✅ Finish Camera Blocking when finished.")
    
    # Parameters row
    with gr.Row():
        camera_type_dropdown = gr.Dropdown(
            choices=["director", "additional", "all"],
            value="director",
            label="Camera Type"
        )
        max_additional_cameras_input = gr.Number(
            value=1,
            label="Max Additional Cameras",
            precision=0,
            minimum=1,
            maximum=10
        )
        start_frame_input = gr.Number(
            value=1,
            label="Start Frame",
            precision=0,
            minimum=1
        )
        end_frame_input = gr.Number(
            value=73,
            label="End Frame",
            precision=0,
            minimum=1
        )
        max_adjustment_rounds_input = gr.Number(
            value=10,
            label="Max Adjustment Rounds",
            precision=0,
            minimum=1,
            maximum=20
        )
    
    # Loading status
    loading_status = gr.Markdown(value="", visible=False)
    
    # Perform Camera Blocking button
    perform_btn = gr.Button("🎬 Perform Camera Blocking", variant="primary")
    
    # JSON Editor for results (camera_blocking_v{num}.json)
    step_10_2_editor = JSONEditorComponent(
        label="Camera Blocking Result",
        visible_initially=False,
        file_basename="camera_blocking",
        use_version_control=True,
        json_root_keys_list=["shot_details"],
        title="Step 10.2"
    )
    step_10_2_editor.setup_resume_with_project_dir(project_dir, subfolder="camera_blocking")
    
    # Camera Preview Section
    gr.Markdown("### Camera Previews")
    
    with gr.Column(visible=False) as camera_viewer_container:
        camera_status = gr.Markdown(value="", visible=False)
        camera_viewer = gr.Image(label="Camera Preview", height=400)
        camera_buttons = gr.Dataset(
            components=[gr.Textbox(visible=False)],
            samples=[],
            label="Click to view camera",
            samples_per_page=20,
        )
    
    # Camera selection for reperform/resume
    gr.Markdown("### Select Cameras for Reperform/Resume")
    camera_selection_status = gr.Markdown(value="No cameras selected.", visible=True)
    camera_checkboxes = gr.CheckboxGroup(
        choices=[],
        label="Select cameras",
        visible=False
    )
    
    # State variables
    cameras_state = gr.State([])
    selected_cameras = gr.State([])
    
    # Action buttons row
    with gr.Row():
        load_cameras_btn = gr.Button("📂 Load Cameras", variant="secondary")
        reperform_btn = gr.Button("🔄 Reperform Camera Blocking", variant="secondary")
        resume_btn = gr.Button("▶️ Resume Camera Blocking", variant="secondary")
        finish_btn = gr.Button("✅ Finish Camera Blocking", variant="secondary")
    
    # Event handlers
    def perform_wrapper(camera_type, max_additional, start_frame, end_frame, max_rounds, proj_dir, v_model, api_key, api_base, provider):
        for result in show_loading_and_perform_camera_blocking(
            blender_client, proj_dir, camera_type, max_additional,
            start_frame, end_frame, max_rounds, v_model, api_key, api_base,
            anyllm_provider=provider,
            camera_name_filter=None
        ):
            yield result
    
    perform_btn.click(
        fn=perform_wrapper,
        inputs=[
            camera_type_dropdown, max_additional_cameras_input,
            start_frame_input, end_frame_input, max_adjustment_rounds_input,
            project_dir, vision_model, anyllm_api_key, anyllm_api_base, anyllm_provider
        ],
        outputs=[loading_status, perform_btn, reperform_btn]
    ).then(
        fn=step_10_2_editor._handle_resume,
        inputs=[project_dir],
        outputs=step_10_2_editor._get_resume_outputs()
    ).then(
        fn=display_camera_previews,
        inputs=[project_dir],
        outputs=[camera_viewer, camera_status, camera_buttons, camera_viewer_container, cameras_state, selected_cameras]
    ).then(
        fn=lambda proj_dir: gr.update(choices=load_camera_names_from_json(proj_dir), value=[], visible=bool(load_camera_names_from_json(proj_dir))),
        inputs=[project_dir],
        outputs=[camera_checkboxes]
    )
    
    # Reperform with selected cameras
    def reperform_wrapper(camera_type, max_additional, start_frame, end_frame, max_rounds, proj_dir, v_model, api_key, api_base, provider, selected):
        if not selected:
            yield (
                gr.update(value="⚠️ Please select at least one camera to reperform.", visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )
            return
        for result in show_loading_and_perform_camera_blocking(
            blender_client, proj_dir, camera_type, max_additional,
            start_frame, end_frame, max_rounds, v_model, api_key, api_base,
            anyllm_provider=provider,
            camera_name_filter=selected
        ):
            yield result
    
    reperform_btn.click(
        fn=reperform_wrapper,
        inputs=[
            camera_type_dropdown, max_additional_cameras_input,
            start_frame_input, end_frame_input, max_adjustment_rounds_input,
            project_dir, vision_model, anyllm_api_key, anyllm_api_base,
            anyllm_provider, camera_checkboxes
        ],
        outputs=[loading_status, perform_btn, reperform_btn]
    ).then(
        fn=step_10_2_editor._handle_resume,
        inputs=[project_dir],
        outputs=step_10_2_editor._get_resume_outputs()
    ).then(
        fn=display_camera_previews,
        inputs=[project_dir],
        outputs=[camera_viewer, camera_status, camera_buttons, camera_viewer_container, cameras_state, selected_cameras]
    ).then(
        fn=lambda proj_dir: gr.update(choices=load_camera_names_from_json(proj_dir), value=[], visible=bool(load_camera_names_from_json(proj_dir))),
        inputs=[project_dir],
        outputs=[camera_checkboxes]
    )
    
    # Load cameras button
    load_cameras_btn.click(
        fn=step_10_2_editor._handle_resume,
        inputs=[project_dir],
        outputs=step_10_2_editor._get_resume_outputs()
    ).then(
        fn=display_camera_previews,
        inputs=[project_dir],
        outputs=[camera_viewer, camera_status, camera_buttons, camera_viewer_container, cameras_state, selected_cameras]
    ).then(
        fn=lambda proj_dir: gr.update(choices=load_camera_names_from_json(proj_dir), value=[], visible=bool(load_camera_names_from_json(proj_dir))),
        inputs=[project_dir],
        outputs=[camera_checkboxes]
    )
    
    # Resume cameras
    def resume_wrapper(proj_dir, selected):
        filter_list = selected if selected else None
        for result in show_loading_and_resume_cameras(blender_client, proj_dir, filter_list):
            yield result
    
    resume_btn.click(
        fn=resume_wrapper,
        inputs=[project_dir, camera_checkboxes],
        outputs=[loading_status, resume_btn]
    )
    
    # Finish camera blocking - read info from Blender and update JSON
    def finish_wrapper(proj_dir):
        for result in show_loading_and_finish_camera_blocking(blender_client, proj_dir):
            yield result
    
    finish_btn.click(
        fn=finish_wrapper,
        inputs=[project_dir],
        outputs=[loading_status, finish_btn]
    ).then(
        fn=step_10_2_editor._handle_resume,
        inputs=[project_dir],
        outputs=step_10_2_editor._get_resume_outputs()
    )
    
    # Camera selection handler
    camera_buttons.select(
        fn=select_camera,
        inputs=[cameras_state],
        outputs=[camera_viewer, camera_status]
    )
    
    # Update selection display when checkboxes change
    camera_checkboxes.change(
        fn=lambda selected: f"**Selected cameras ({len(selected)}):** {', '.join(selected)}" if selected else "No cameras selected.",
        inputs=[camera_checkboxes],
        outputs=[camera_selection_status]
    )

    return {
        "camera_type": camera_type_dropdown,
        "max_additional_cameras": max_additional_cameras_input,
        "start_frame": start_frame_input,
        "end_frame": end_frame_input,
        "max_adjustment_rounds": max_adjustment_rounds_input,
        "perform_btn": perform_btn,
        "reperform_btn": reperform_btn,
        "resume_btn": resume_btn,
        "load_cameras_btn": load_cameras_btn,
        "finish_btn": finish_btn,
    }
