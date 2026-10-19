import json
import os
import shutil
from typing import Optional, List, Literal

from .camera_helpers import create_and_place_camera_for_shot, resume_camera
from ...memory_budget import ensure_memory_budget, MemoryBudgetExceeded


CameraType = Literal['director', 'additional', 'all']


def camera_operator(
    path_to_input_json: str,
    vision_model: str,
    anyllm_api_key: str,
    anyllm_api_base: Optional[str] = None,
    anyllm_provider: str = "gemini",
    camera_type: CameraType = 'director',
    max_additional_cameras: int = 1,
    camera_name_filter: Optional[List[str]] = None,
    start_frame: int = 1,
    end_frame: int = 73,
    max_adjustment_rounds: int = 10,
    preview_image_save_dir: Optional[str] = None,
) -> dict:
    """
    Create cameras in Blender for each shot based on camera instructions from a JSON file.
    
    This function reads shot details from a JSON file and uses create_and_place_camera_for_shot
    to place cameras for each shot. After placement, camera information is updated in the JSON
    for reproducibility.
    
    Parameters:
    - path_to_input_json: File path to the JSON file containing shot_details with camera instructions
    - vision_model: LLM model identifier for vision analysis
    - anyllm_api_key: API key for the LLM service
    - anyllm_api_base: Optional API base URL
    - camera_type: Which cameras to place:
        - 'director': Only place the main camera_instruction for each shot
        - 'additional': Only place cameras from additional_camera_instructions
        - 'all': Place all cameras (main + additional)
    - max_additional_cameras: Maximum number of additional cameras to place per shot (default: 1)
    - camera_name_filter: List of camera names to place. Only cameras with names in this list will be placed.
                          If None, place all cameras (default: None)
    - start_frame: Starting frame for camera placement (default: 1)
    - end_frame: Ending frame for camera placement (default: 73)
    - max_adjustment_rounds: Maximum rounds for camera adjustment (default: 10)
    - preview_image_save_dir: Directory to save camera preview images. If None, no images are saved.
                              Preview image paths are added to camera instructions as 'camera_preview_image'
    
    Returns:
    - Dictionary with:
        - 'success': Boolean indicating overall success
        - 'shot_details': Updated shot_details list with camera placement info
        - 'cameras_placed': List of camera names that were successfully placed
        - 'cameras_failed': List of camera names that failed to place
        - 'error': Error message if failed (optional)
    """
    try:
        # Load the JSON file
        with open(path_to_input_json, 'r') as f:
            input_data = json.load(f)
        
        # Get shot_details - could be at root level or nested
        if 'shot_details' in input_data:
            shot_details = input_data['shot_details']
        elif isinstance(input_data, list):
            shot_details = input_data
        else:
            return {
                "success": False,
                "error": "Could not find 'shot_details' in the JSON file"
            }
        
        cameras_placed = []
        cameras_failed = []
        last_successful_camera = None

        def checkpoint():
            if not preview_image_save_dir:
                return None
            os.makedirs(preview_image_save_dir, exist_ok=True)
            path = os.path.join(preview_image_save_dir, 'camera_blocking_in_progress.json')
            with open(path + '.tmp', 'w') as stream:
                json.dump(shot_details, stream, indent=2)
            os.replace(path + '.tmp', path)
            return path
        
        # First pass: count total cameras to be placed (considering filter)
        total_cameras_to_place = 0
        for shot in shot_details:
            scene_id = shot.get('scene_id')
            shot_id = shot.get('shot_id')
            if scene_id is None or shot_id is None:
                continue
            
            if camera_type in ['director', 'all']:
                main_camera = shot.get('camera_instruction')
                if main_camera:
                    cam_name = main_camera.get('camera_name', 'Camera')
                    if camera_name_filter is None or cam_name in camera_name_filter:
                        total_cameras_to_place += 1
            
            if camera_type in ['additional', 'all']:
                additional_cameras = shot.get('additional_camera_instructions', [])
                for idx, add_cam in enumerate(additional_cameras[:max_additional_cameras]):
                    cam_name = add_cam.get('camera_name', 'Camera')
                    if camera_name_filter is None or cam_name in camera_name_filter:
                        total_cameras_to_place += 1
        
        print(f"\n{'='*60}")
        print(f"CAMERA OPERATOR: Starting camera placement")
        print(f"Total cameras to place: {total_cameras_to_place}")
        if camera_name_filter:
            print(f"Camera name filter: {camera_name_filter}")
        print(f"{'='*60}\n")
        
        # Process each shot
        for shot in shot_details:
            scene_id = shot.get('scene_id')
            shot_id = shot.get('shot_id')
            
            if scene_id is None or shot_id is None:
                continue
            
            # Build scene name
            scene_name = f"Scene_{scene_id}_Shot_{shot_id}"
            
            # Collect camera instructions to process based on camera_type
            camera_instructions_to_place = []
            
            # Main camera instruction
            if camera_type in ['director', 'all']:
                main_camera = shot.get('camera_instruction')
                if main_camera:
                    camera_instructions_to_place.append({
                        'instruction': main_camera,
                        'key': 'camera_instruction',
                        'index': None
                    })
            
            # Additional camera instructions
            if camera_type in ['additional', 'all']:
                additional_cameras = shot.get('additional_camera_instructions', [])
                if additional_cameras:
                    # Slice to max_additional_cameras
                    for idx, add_cam in enumerate(additional_cameras[:max_additional_cameras]):
                        camera_instructions_to_place.append({
                            'instruction': add_cam,
                            'key': 'additional_camera_instructions',
                            'index': idx
                        })
            
            # Place each camera
            for cam_info in camera_instructions_to_place:
                camera_instruction = cam_info['instruction']
                camera_name = camera_instruction.get('camera_name', 'Camera')
                
                # Check camera name filter
                if camera_name_filter is not None and camera_name not in camera_name_filter:
                    continue
                
                try:
                    ensure_memory_budget("Camera blocking")
                    # Place the camera
                    result = create_and_place_camera_for_shot(
                        camera_instruction=camera_instruction,
                        vision_model=vision_model,
                        anyllm_api_key=anyllm_api_key,
                        anyllm_api_base=anyllm_api_base,
                        anyllm_provider=anyllm_provider,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        max_adjustment_rounds=max_adjustment_rounds,
                        scene_name=scene_name,
                        preview_image_save_dir=preview_image_save_dir
                    )
                    
                    if result.get('success'):
                        cameras_placed.append(camera_name)
                        last_successful_camera = camera_name
                        print("-"*20)
                        print(f"[{len(cameras_placed)}/{total_cameras_to_place}] Successfully placed: {camera_name}")
                        print("-"*20)
                        
                        # Handle preview image
                        camera_preview_image = None
                        final_image_path = result.get('final_image_path')
                        if preview_image_save_dir and final_image_path and os.path.exists(final_image_path):
                            # Create save directory if needed
                            os.makedirs(preview_image_save_dir, exist_ok=True)
                            # Build destination filename
                            ext = os.path.splitext(final_image_path)[1]
                            dest_filename = f"{camera_name}_preview{ext}"
                            dest_path = os.path.join(preview_image_save_dir, dest_filename)
                            # Copy the image
                            shutil.copy2(final_image_path, dest_path)
                            camera_preview_image = dest_path
                        elif final_image_path:
                            # Use original path if no save dir specified
                            camera_preview_image = final_image_path
                        
                        # Update camera instruction with placement info
                        placement_info = {
                            'camera_parameters': result.get('camera_parameters'),
                            'start_transform': result.get('start_transform'),
                            'end_transform': result.get('end_transform'),
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'is_animated': result.get('is_animated', False),
                            'movement_operation': result.get('movement_operation'),
                            'movement_steps': result.get('movement_steps'),
                            'dof_applied': result.get('dof_applied', False),
                            'focus_distance': result.get('focus_distance'),
                            'camera_preview_image': camera_preview_image,
                            'placement_success': True
                        }
                        
                        # Update the instruction in place
                        camera_instruction.update(placement_info)
                        checkpoint()
                    else:
                        cameras_failed.append({
                            'camera_name': camera_name,
                            'error': result.get('error', 'Unknown error')
                        })
                        camera_instruction['placement_success'] = False
                        camera_instruction['placement_error'] = result.get('error', 'Unknown error')
                        print(f"[{len(cameras_placed)}/{total_cameras_to_place}] FAILED: {camera_name} - {result.get('error', 'Unknown error')}")

                    ensure_memory_budget("Camera blocking")
                except MemoryBudgetExceeded as exc:
                    path = checkpoint()
                    return {
                        "success": False, "error": f"{exc} Checkpoint: {path}",
                        "shot_details": shot_details, "cameras_placed": cameras_placed,
                        "cameras_failed": cameras_failed,
                    }
                        
                except Exception as e:
                    cameras_failed.append({
                        'camera_name': camera_name,
                        'error': str(e)
                    })
                    camera_instruction['placement_success'] = False
                    camera_instruction['placement_error'] = str(e)
                    print(f"[{len(cameras_placed)}/{total_cameras_to_place}] FAILED: {camera_name} - {str(e)}")
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"CAMERA OPERATOR: Placement complete")
        print(f"Total cameras to place: {total_cameras_to_place}")
        print(f"Successfully placed: {len(cameras_placed)}")
        print(f"Failed: {len(cameras_failed)}")
        if last_successful_camera:
            print(f"Last successful camera: {last_successful_camera}")
        else:
            print(f"Last successful camera: None")
        print(f"{'='*60}\n")
        
        # Update the original data structure
        if 'shot_details' in input_data:
            input_data['shot_details'] = shot_details
        else:
            input_data = shot_details
        
        return {
            "success": len(cameras_failed) == 0,
            "shot_details": shot_details,
            "cameras_placed": cameras_placed,
            "cameras_failed": cameras_failed,
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def resume_camera_operator(
    path_to_input_json: str,
    camera_name_filter: Optional[List[str]] = None,
) -> dict:
    """
    Resume/recreate all cameras from a previously saved JSON file without LLM.
    
    This function reads the JSON file that was output by camera_operator (with placement info)
    and recreates all cameras in their respective Blender scenes using the saved parameters.
    
    Parameters:
    - path_to_input_json: File path to the JSON file containing shot_details with placement info
    - camera_name_filter: List of camera names to resume. Only cameras with names in this list will be recreated.
                          If None, resume all cameras (default: None)
    
    Returns:
    - Dictionary with:
        - 'success': Boolean indicating overall success
        - 'cameras_resumed': List of camera names that were successfully recreated
        - 'cameras_failed': List of camera names that failed to recreate
        - 'error': Error message if failed (optional)
    """
    try:
        # Load the JSON file
        with open(path_to_input_json, 'r') as f:
            input_data = json.load(f)
        
        # Get shot_details - could be at root level or nested
        if 'shot_details' in input_data:
            shot_details = input_data['shot_details']
        elif isinstance(input_data, list):
            shot_details = input_data
        else:
            return {
                "success": False,
                "error": "Could not find 'shot_details' in the JSON file"
            }
        
        cameras_resumed = []
        cameras_failed = []
        
        print(f"\n{'='*60}")
        print(f"RESUME CAMERA OPERATOR: Starting camera resume")
        print(f"Input JSON: {path_to_input_json}")
        print(f"Camera name filter: {camera_name_filter}")
        print(f"Total shots in file: {len(shot_details)}")
        print(f"{'='*60}\n")
        
        # Process each shot
        for shot in shot_details:
            scene_id = shot.get('scene_id')
            shot_id = shot.get('shot_id')
            
            if scene_id is None or shot_id is None:
                continue
            
            # Build scene name
            scene_name = f"Scene_{scene_id}_Shot_{shot_id}"
            
            # Collect all camera instructions that have placement info
            camera_instructions_to_resume = []
            
            # Main camera instruction
            main_camera = shot.get('camera_instruction')
            if main_camera and main_camera.get('placement_success'):
                camera_instructions_to_resume.append(main_camera)
            
            # Additional camera instructions
            additional_cameras = shot.get('additional_camera_instructions', [])
            for add_cam in additional_cameras:
                if add_cam.get('placement_success'):
                    camera_instructions_to_resume.append(add_cam)
            
            # Resume each camera
            for camera_instruction in camera_instructions_to_resume:
                camera_name = camera_instruction.get('camera_name', 'Camera')
                
                # Check camera name filter
                if camera_name_filter is not None and camera_name not in camera_name_filter:
                    continue
                
                try:
                    ensure_memory_budget("Camera resume")
                    print(f"\n--- Resuming camera: {camera_name} in {scene_name} ---")
                    print(f"  placement_success: {camera_instruction.get('placement_success')}")
                    print(f"  camera_parameters: {camera_instruction.get('camera_parameters')}")
                    print(f"  start_transform: {camera_instruction.get('start_transform')}")
                    print(f"  end_transform: {camera_instruction.get('end_transform')}")
                    print(f"  is_animated: {camera_instruction.get('is_animated')}")
                    
                    result = resume_camera(
                        camera_instruction=camera_instruction,
                        scene_name=scene_name
                    )
                    
                    print(f"  result: {result}")
                    
                    if result.get('success'):
                        cameras_resumed.append(camera_name)
                        print(f"  ✓ Successfully resumed: {camera_name}")
                    else:
                        cameras_failed.append({
                            'camera_name': camera_name,
                            'error': result.get('error', 'Unknown error')
                        })
                        print(f"  ✗ Failed: {camera_name} - {result.get('error', 'Unknown error')}")
                        
                except MemoryBudgetExceeded as exc:
                    return {"success": False, "error": str(exc),
                            "cameras_resumed": cameras_resumed, "cameras_failed": cameras_failed}
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    cameras_failed.append({
                        'camera_name': camera_name,
                        'error': str(e)
                    })
                    print(f"  ✗ Exception: {camera_name} - {str(e)}")
        
        return {
            "success": len(cameras_failed) == 0,
            "cameras_resumed": cameras_resumed,
            "cameras_failed": cameras_failed,
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
