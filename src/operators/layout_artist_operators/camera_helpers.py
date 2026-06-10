import bpy
import tempfile
import os
import shutil
import time
import gc
import warnings
import json

warnings.filterwarnings("ignore", message="Unclosed client session")
warnings.filterwarnings("ignore", message="Unclosed connector")
warnings.filterwarnings("ignore")
import base64
import mimetypes
from typing import Optional, List
import math
import mathutils
from PIL import Image, ImageDraw
from bpy_extras.object_utils import world_to_camera_view

from pydantic import BaseModel
try:
    from ..llm_completion import completion
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from llm_completion import completion


def _switch_to_scene(scene_name: str) -> Optional[bpy.types.Scene]:
    """Switch to a scene with the given scene_name.

    Args:
        scene_name: The name of the scene to switch to.

    Returns:
        The scene that was switched to, or None if not found.
    """
    if scene_name in bpy.data.scenes:
        scene = bpy.data.scenes[scene_name]
        bpy.context.window.scene = scene
        return scene
    return None


def _find_3d_viewport():
    """Find the first 3D viewport area, space, and region.
    
    Returns:
        Tuple of (area, space, region) or (None, None, None) if not found.
    """
    for a in bpy.context.screen.areas:
        if a.type == 'VIEW_3D':
            space = None
            region = None
            for s in a.spaces:
                if s.type == 'VIEW_3D':
                    space = s
                    break
            for r in a.regions:
                if r.type == 'WINDOW':
                    region = r
                    break
            return a, space, region
    return None, None, None


def _compute_diagonal_rotation_map() -> dict:
    """Compute diagonal viewport rotation quaternions for 8-direction turnaround.
    
    Returns:
        Dict mapping direction names to mathutils.Quaternion values for
        'front_right', 'back_right', 'back_left', 'front_left'.
    """
    front_quat = mathutils.Quaternion((0.7071068, 0.7071068, 0.0, 0.0))
    diagonal_rotation_map = {}
    for name, z_angle_deg in [
        ('front_right', 45),
        ('back_right', 135),
        ('back_left', 225),
        ('front_left', 315),
    ]:
        z_rot = mathutils.Quaternion((0, 0, 1), math.radians(z_angle_deg))
        diagonal_rotation_map[name] = z_rot @ front_quat
    return diagonal_rotation_map


def _call_llm_with_retry(
    messages: list,
    response_format,
    vision_model: str,
    anyllm_api_key: str,
    anyllm_api_base: Optional[str] = None,
    anyllm_provider: str = "gemini",
    reasoning_effort: str = "low",
    max_retries: int = 3,
    retry_delays: list = None,
):
    """Call the LLM with retry logic.
    
    Args:
        messages: Chat messages to send.
        response_format: Pydantic model for structured response.
        vision_model: Model identifier.
        anyllm_api_key: API key.
        anyllm_api_base: Optional API base URL.
        anyllm_provider: Provider name (default: "gemini").
        reasoning_effort: Reasoning effort level (default: "low").
        max_retries: Number of retry attempts (default: 3).
        retry_delays: Delay between retries in seconds (default: [2, 4, 8]).
    
    Returns:
        The LLM response object, or None if all retries failed.
    
    Thread-safety note:
        completion() spawns background async cleanup tasks (via create_task in
        llm_completion._close_client).  Per Blender's Python API docs, Python
        threads / async tasks that outlive the calling script cause random
        crashes.  We therefore:
        - Do NOT call gc.collect() immediately after completion(); the async
          tasks may still hold references to objects being finalized, leading to
          fatal _PyObject_GC_NewVar / _PyTuple_FromArray crashes.
        - Sleep briefly after completion() returns so background async cleanup
          can finish before the caller resumes bpy API operations.
    """
    if retry_delays is None:
        retry_delays = [2, 4, 8]
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = completion(
                api_key=anyllm_api_key,
                api_base=anyllm_api_base,
                provider=anyllm_provider,
                model=vision_model,
                messages=messages,
                response_format=response_format,
                reasoning_effort=reasoning_effort,
            )
            # THREAD-SAFETY: Do NOT gc.collect() here.  completion() may leave
            # background async tasks (aiohttp session close) still running.
            # Give them time to finish before returning to bpy API callers.
            time.sleep(0.5)
            return response
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delays[attempt])
                continue
    
    print(f"LLM call failed after {max_retries} retries: {last_error}")
    return None


def _get_mesh_children(obj: bpy.types.Object) -> List[bpy.types.Object]:
    """
    Recursively find all MESH type children of an object.
    
    Args:
        obj: The parent object to search from.
    
    Returns:
        List of MESH type objects found in the hierarchy.
    """
    mesh_children = []
    
    def collect_meshes(parent):
        for child in parent.children:
            if child.type == 'MESH':
                mesh_children.append(child)
            collect_meshes(child)
    
    collect_meshes(obj)
    return mesh_children


def select_objects_for_outline(
    asset_id_list: List[str],
    scene_name: Optional[str] = None,
) -> dict:
    """
    Select objects by asset_id for outline visibility in viewport screenshots.
    
    If the root object is not a MESH type (e.g., EMPTY for character rigs),
    this function will select its MESH children instead, as only MESH objects
    show selection outlines in the viewport.
    
    Args:
        asset_id_list: List of asset IDs to select.
        scene_name: Optional scene name to operate on.
    
    Returns:
        A dict with:
            - 'success': Boolean indicating if objects were selected
            - 'selected_objects': List of selected object names (may include mesh children)
            - 'failed_objects': List of asset_ids that could not be found
    """
    try:
        if isinstance(asset_id_list, str):
            asset_id_list = [asset_id_list]
        
        if scene_name is not None:
            scene = _switch_to_scene(scene_name)
            if scene is None:
                return {
                    "success": False,
                    "selected_objects": [],
                    "failed_objects": asset_id_list,
                    "error": f"Scene '{scene_name}' not found.",
                }
        else:
            scene = bpy.context.scene
        
        bpy.ops.object.select_all(action='DESELECT')

        time.sleep(0.1)
        
        selected_objects = []
        failed_objects = []
        
        for asset_id in asset_id_list:
            obj = _find_object_by_asset_id(asset_id, scene)
            if obj is not None:
                # If the object is a MESH, select it directly
                if obj.type == 'MESH':
                    obj.select_set(True)
                    selected_objects.append(obj.name)
                else:
                    # For non-MESH objects (EMPTY, ARMATURE, etc.), find and select mesh children
                    mesh_children = _get_mesh_children(obj)
                    if mesh_children:
                        for mesh_obj in mesh_children:
                            mesh_obj.select_set(True)
                            selected_objects.append(mesh_obj.name)
                    else:
                        # No mesh children found, select the object itself as fallback
                        obj.select_set(True)
                        selected_objects.append(obj.name)
            else:
                failed_objects.append(asset_id)
        
        # Set first selected object as active
        if selected_objects:
            bpy.context.view_layer.objects.active = bpy.data.objects[selected_objects[0]]
        
        return {
            "success": len(selected_objects) > 0,
            "selected_objects": selected_objects,
            "failed_objects": failed_objects,
        }
    
    except Exception as e:
        return {
            "success": False,
            "selected_objects": [],
            "failed_objects": asset_id_list if isinstance(asset_id_list, list) else [asset_id_list],
            "error": str(e),
        }


def _find_object_by_asset_id(asset_id: str, scene: bpy.types.Scene) -> Optional[bpy.types.Object]:
    """
    Find an object in the scene matching the asset_id pattern.
    
    Matches objects named exactly as asset_id, or with Blender's auto-generated
    suffixes (e.g., "snow_white.001", "snow_white.002"), or with underscore
    suffixes (e.g., "snow_white_sleeping").
    
    Args:
        asset_id: The base asset ID to search for.
        scene: The Blender scene to search in.
    
    Returns:
        The matching object, or None if not found.
    """
    scene_objects = scene.objects
    
    # First, try exact match
    if asset_id in scene_objects:
        return scene_objects[asset_id]
    
    # Search for objects whose name starts with asset_id followed by a dot or underscore
    # Priority: exact match > dot suffix (.001) > underscore suffix (_sleeping)
    dot_matches = []
    underscore_matches = []
    
    for obj in scene_objects:
        if obj.name == asset_id:
            return obj
        elif obj.name.startswith(f"{asset_id}."):
            dot_matches.append(obj)
        elif obj.name.startswith(f"{asset_id}_"):
            underscore_matches.append(obj)
    
    # Prefer dot suffix matches first (Blender's standard naming)
    if dot_matches:
        return dot_matches[0]
    
    # Fall back to underscore suffix matches
    if underscore_matches:
        return underscore_matches[0]
    
    return None


def select_object_from_scene(
    asset_id_list: List[str],
    scene_name: Optional[str] = None,
) -> dict:
    """
    Select objects by asset_id in a scene and set them as active.
    
    Args:
        asset_id_list: List of asset IDs to select. Can be a single string or list of strings.
        scene_name: Optional scene name to operate on. If provided, will switch to that scene
                   first. If None, operates on the current scene.
    
    Returns:
        A dict with:
            - 'success': Boolean indicating if all objects were selected successfully
            - 'selected_objects': List of successfully selected object names
            - 'failed_objects': List of asset_ids that could not be found
            - 'error': Error message if any failures occurred
    """
    try:
        # Handle single string input
        if isinstance(asset_id_list, str):
            asset_id_list = [asset_id_list]
        
        # Switch to target scene if specified
        if scene_name is not None:
            scene = _switch_to_scene(scene_name)
            if scene is None:
                return {
                    "success": False,
                    "selected_objects": [],
                    "failed_objects": asset_id_list,
                    "error": f"Scene '{scene_name}' not found.",
                }
        else:
            scene = bpy.context.scene
        
        # Deselect all objects first
        bpy.ops.object.select_all(action='DESELECT')
        
        time.sleep(0.1)
        
        selected_objects = []
        failed_objects = []
        
        # Find and select each object
        for asset_id in asset_id_list:
            obj = _find_object_by_asset_id(asset_id, scene)
            if obj is not None:
                obj.select_set(True)
                selected_objects.append(obj.name)
            else:
                failed_objects.append(asset_id)
        
        # Set the first selected object as active
        if selected_objects:
            first_obj_name = selected_objects[0]
            bpy.context.view_layer.objects.active = bpy.data.objects[first_obj_name]
        
        if failed_objects:
            return {
                "success": False,
                "selected_objects": selected_objects,
                "failed_objects": failed_objects,
                "error": f"Could not find objects: {', '.join(failed_objects)}",
            }
        
        return {
            "success": True,
            "selected_objects": selected_objects,
            "failed_objects": [],
        }
    
    except Exception as e:
        return {
            "success": False,
            "selected_objects": [],
            "failed_objects": asset_id_list if isinstance(asset_id_list, list) else [asset_id_list],
            "error": str(e),
        }


def _capture_viewport_screenshot_with_outline(
    max_size: int,
    direction: str,
    outline_enabled: bool = True,
) -> dict:
    """
    Capture a screenshot of the current 3D viewport with optional object outline.
    Copied from BlenderMCPServer.get_viewport_screenshot which is confirmed working.
    
    Parameters:
    - max_size: Maximum size in pixels for the largest dimension
    - direction: Direction name for the filename
    - outline_enabled: Whether to enable object outline during capture
    
    Returns:
    - Dictionary with success status and filepath or error
    """
    # Generate filepath in temp directory
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, f"turnaround_{direction}.png")
    
    try:
        # Store original viewport settings and set up for clean screenshots
        original_viewport_settings = {}
        area = None
        space = None
        try:
            for a in bpy.context.screen.areas:
                if a.type == 'VIEW_3D':
                    area = a
                    for s in area.spaces:
                        if s.type == 'VIEW_3D':
                            space = s
                            # Store original UI settings
                            original_viewport_settings['show_region_header'] = space.show_region_header
                            original_viewport_settings['show_region_toolbar'] = space.show_region_toolbar
                            original_viewport_settings['show_region_ui'] = space.show_region_ui
                            original_viewport_settings['show_gizmo'] = space.show_gizmo
                            original_viewport_settings['shading_type'] = space.shading.type
                            if hasattr(space, 'overlay'):
                                original_viewport_settings['show_overlays'] = space.overlay.show_overlays
                            
                            # Hide UI elements for clean screenshots
                            space.show_region_header = False
                            space.show_region_toolbar = False
                            space.show_region_ui = False
                            space.show_gizmo = False
                            
                            # Handle overlays based on outline_enabled parameter
                            if hasattr(space, 'overlay'):
                                overlay = space.overlay
                                if outline_enabled:
                                    overlay.show_overlays = True
                                    if hasattr(overlay, 'show_outline_selected'):
                                        overlay.show_outline_selected = True
                                    for attr in [
                                        'show_cursor', 'show_floor', 'show_axis_x', 'show_axis_y', 'show_axis_z',
                                        'show_object_origins', 'show_stats', 'show_text', 'show_extras', 'show_bones',
                                        'show_relationship_lines', 'show_motion_paths', 'show_wireframes',
                                        'show_face_orientation', 'show_ortho_grid'
                                    ]:
                                        if hasattr(overlay, attr):
                                            setattr(overlay, attr, False)
                                else:
                                    overlay.show_overlays = False
                            
                            # Set viewport shading to MATERIAL
                            space.shading.type = 'MATERIAL'
                            break
                    break
            
            # Store original background/gradient settings and set to white
            gradients = bpy.context.preferences.themes[0].view_3d.space.gradients
            original_viewport_settings['background_type'] = gradients.background_type
            original_viewport_settings['high_gradient'] = tuple(gradients.high_gradient)
            
            # Set background to white single color
            gradients.background_type = 'SINGLE_COLOR'
            gradients.high_gradient = (1.0, 1.0, 1.0)
            
            # Increase outline width for thicker, more visible outlines
            theme_view3d = bpy.context.preferences.themes[0].view_3d
            if hasattr(theme_view3d, 'outline_width'):
                original_viewport_settings['outline_width'] = theme_view3d.outline_width
                theme_view3d.outline_width = 5
            
            # Set outline color to red for active and selected objects
            if hasattr(theme_view3d, 'object_active'):
                original_viewport_settings['object_active'] = tuple(theme_view3d.object_active)
                theme_view3d.object_active = (1.0, 0.0, 0.0)
            if hasattr(theme_view3d, 'object_selected'):
                original_viewport_settings['object_selected'] = tuple(theme_view3d.object_selected)
                theme_view3d.object_selected = (1.0, 0.0, 0.0)
            
            # Force UI update
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            
        except Exception as e:
            print(f"Warning: Could not fully configure viewport settings: {e}")

        if not area:
            return {"success": False, "error": "No 3D viewport found"}

        # Store original render settings
        scene = bpy.context.scene
        original_filepath = scene.render.filepath
        original_format = scene.render.image_settings.file_format
        original_res_x = scene.render.resolution_x
        original_res_y = scene.render.resolution_y
        original_res_percentage = scene.render.resolution_percentage

        try:
            # Set up render settings
            scene.render.filepath = filepath
            scene.render.image_settings.file_format = 'PNG'
            
            # Get viewport dimensions and scale to max_size
            region = None
            for r in area.regions:
                if r.type == 'WINDOW':
                    region = r
                    break
            
            if region:
                # Render at 3x resolution for supersampling, then downsample for sharper results
                supersample_factor = 3
                vp_width, vp_height = region.width, region.height
                if max(vp_width, vp_height) > max_size:
                    scale_factor = max_size / max(vp_width, vp_height)
                    target_width = int(vp_width * scale_factor)
                    target_height = int(vp_height * scale_factor)
                else:
                    target_width = vp_width
                    target_height = vp_height
                
                # Render at 3x the target size
                scene.render.resolution_x = target_width * supersample_factor
                scene.render.resolution_y = target_height * supersample_factor
                scene.render.resolution_percentage = 100
            
            # THREAD-SAFETY: Disable GC during OpenGL render.  Gradio's
            # background asyncio thread can trigger gc.collect() at any moment;
            # if that happens while Blender's render pipeline holds transient
            # C-level references the result is a fatal crash.  We bracket the
            # render with gc.disable/enable and sleep afterwards so any pending
            # finalizers spawned by background threads settle before we touch
            # the rendered image or any bpy state.
            gc.disable()
            try:
                with bpy.context.temp_override(
                    window=bpy.context.window,
                    screen=bpy.context.screen,
                    area=area,
                    region=region,
                    space_data=space,
                ):
                    bpy.ops.render.opengl(write_still=True, view_context=True)
            finally:
                gc.enable()
            
            # Brief sleep to let background threads / async tasks settle
            time.sleep(0.2)
            
            # Downsample to target size with high-quality LANCZOS filter
            if region:
                img = Image.open(filepath)
                img_downsampled = img.resize((target_width, target_height), Image.LANCZOS)
                img_downsampled.save(filepath)
                img.close()
                img_downsampled.close()
                width = target_width
                height = target_height
            else:
                width = scene.render.resolution_x
                height = scene.render.resolution_y
            
        finally:
            # Restore original render settings
            scene.render.filepath = original_filepath
            scene.render.image_settings.file_format = original_format
            scene.render.resolution_x = original_res_x
            scene.render.resolution_y = original_res_y
            scene.render.resolution_percentage = original_res_percentage
            
            # Restore viewport to original state
            try:
                if space:
                    # Restore UI regions to original or default values
                    space.show_region_header = original_viewport_settings.get('show_region_header', True)
                    space.show_region_toolbar = original_viewport_settings.get('show_region_toolbar', True)
                    space.show_region_ui = original_viewport_settings.get('show_region_ui', True)
                    space.show_gizmo = original_viewport_settings.get('show_gizmo', True)
                    
                    # Restore overlays
                    if hasattr(space, 'overlay'):
                        space.overlay.show_overlays = original_viewport_settings.get('show_overlays', True)
                        space.overlay.show_floor = True
                        space.overlay.show_axis_x = True
                        space.overlay.show_axis_y = True
                        space.overlay.show_cursor = True
                    
                    # Restore shading type
                    space.shading.type = original_viewport_settings.get('shading_type', 'MATERIAL')
                
                # Restore original background/gradient settings
                gradients = bpy.context.preferences.themes[0].view_3d.space.gradients
                if 'background_type' in original_viewport_settings:
                    gradients.background_type = original_viewport_settings['background_type']
                if 'high_gradient' in original_viewport_settings:
                    gradients.high_gradient = original_viewport_settings['high_gradient']
                
                # Restore original outline width and colors
                theme_view3d = bpy.context.preferences.themes[0].view_3d
                if 'outline_width' in original_viewport_settings and hasattr(theme_view3d, 'outline_width'):
                    theme_view3d.outline_width = original_viewport_settings['outline_width']
                if 'object_active' in original_viewport_settings and hasattr(theme_view3d, 'object_active'):
                    theme_view3d.object_active = original_viewport_settings['object_active']
                if 'object_selected' in original_viewport_settings and hasattr(theme_view3d, 'object_selected'):
                    theme_view3d.object_selected = original_viewport_settings['object_selected']
                
                # Force UI update
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                
            except Exception as e:
                print(f"Warning: Could not fully restore viewport settings: {e}")

        return {
            "success": True,
            "width": width,
            "height": height,
            "filepath": filepath
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def get_objects_turnaround_images(max_size: int = 512) -> list:
    """
    Capture front, back, left, right, and diagonal images of the active object(s) in the Blender scene.
    
    This function captures turnaround images from eight directions (4 cardinal + 4 diagonal)
    with object outline enabled for better visibility. After capture, the viewport settings
    are restored to their original state.
    
    Parameters:
    - max_size: Maximum size in pixels for the largest dimension of each image (default: 512)
    
    Returns:
    - List of dictionaries, each with keys:
        - 'direction': The direction name ('front', 'back', 'left', 'right',
                       'front_left', 'front_right', 'back_left', 'back_right')
        - 'filepath': Path to the saved image
        - 'success': Boolean indicating success
        - 'error': Error message if failed (optional)
    """
    # Cardinal directions map to Blender's view axis types
    cardinal_direction_map = {
        'front': 'FRONT',
        'back': 'BACK',
        'left': 'LEFT',
        'right': 'RIGHT',
    }
    
    diagonal_rotation_map = _compute_diagonal_rotation_map()
    
    directions = [
        'front', 'front_right', 'right', 'back_right',
        'back', 'back_left', 'left', 'front_left'
    ]
    
    # Get the active object
    active_object = bpy.context.active_object
    if active_object is None:
        return [{"direction": d, "success": False, "error": "No object is currently selected/active"} for d in directions]
    
    # Get all selected objects
    selected_objects = [obj for obj in bpy.context.selected_objects]
    if not selected_objects:
        # Make sure the active object is selected
        active_object.select_set(True)
        selected_objects = [active_object]
    
    results = []
    
    area, space, region = _find_3d_viewport()
    
    if not area or not space:
        return [{"direction": d, "success": False, "error": "No 3D viewport found"} for d in directions]
    
    if not region:
        return [{"direction": d, "success": False, "error": "No WINDOW region found"} for d in directions]
    
    for direction in directions:
        direction_lower = direction.lower()
        
        with bpy.context.temp_override(
            window=bpy.context.window,
            screen=bpy.context.screen,
            area=area,
            region=region,
            space_data=space
        ):
            if direction_lower in cardinal_direction_map:
                # Cardinal direction: use Blender's built-in view_axis
                bpy.ops.view3d.view_axis(type=cardinal_direction_map[direction_lower], align_active=False)
            else:
                # Diagonal direction: manually set viewport rotation quaternion
                region_3d = space.region_3d
                region_3d.view_rotation = diagonal_rotation_map[direction_lower]
                region_3d.view_perspective = 'ORTHO'
            
            # Frame the selected object(s)
            bpy.ops.view3d.view_selected()
            
            # Zoom out a bit for better framing
            for _ in range(1):
                bpy.ops.view3d.zoom(delta=-1)
        
        # Force viewport update after view operations
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        
        # Take screenshot with outline enabled
        # _capture_viewport_screenshot_with_outline handles all viewport settings
        screenshot_result = _capture_viewport_screenshot_with_outline(
            max_size=max_size,
            direction=direction_lower,
            outline_enabled=True
        )
        
        # Sleep after each capture to prevent memory corruption
        time.sleep(0.1)
        
        if screenshot_result.get("success"):
            results.append({
                "direction": direction,
                "filepath": screenshot_result["filepath"],
                "success": True
            })
        else:
            results.append({
                "direction": direction,
                "success": False,
                "error": screenshot_result.get("error", "Unknown error")
            })
    
    return results

def create_camera(
    name="Camera",
    location=(0.0, -10.0, 10.0),
    rotation=(math.radians(45), 0.0, 0.0),
    
    # --- Lens vs Angle ---
    focal_length=50.0,      
    fov_angle=None,         
    
    type='PERSP',           
    ortho_scale=6.0,        
    
    # --- Sensor ---
    sensor_width=36.0,      
    sensor_height=24.0,     
    sensor_fit='AUTO',      
    
    # --- Clipping ---
    clip_start=0.1,
    clip_end=1000.0,
    
    # --- Depth of Field (DoF) ---
    use_dof=False,
    dof_focus_object=None,      # If provided, calculates distance to surface
    dof_focus_distance=10.0,    # Fallback if no object provided
    dof_fstop=2.8,              
    
    # --- Scene ---
    collection=None
):
    
    # Delete existing camera with the same name if it exists
    if name in bpy.data.objects:
        existing_obj = bpy.data.objects[name]
        if existing_obj.type == 'CAMERA':
            # Also remove the camera data block
            existing_cam_data = existing_obj.data
            bpy.data.objects.remove(existing_obj, do_unlink=True)
            if existing_cam_data and existing_cam_data.users == 0:
                bpy.data.cameras.remove(existing_cam_data)
    
    # 1. Create Data and Object
    cam_data = bpy.data.cameras.new(name=name)
    cam_obj = bpy.data.objects.new(name=name, object_data=cam_data)
    
    # 2. Set Basic Camera Parameters
    cam_data.type = type
    cam_data.ortho_scale = ortho_scale
    
    if fov_angle is not None:
        cam_data.angle = fov_angle
    else:
        cam_data.lens = focal_length

    cam_data.sensor_width = sensor_width
    cam_data.sensor_height = sensor_height
    cam_data.sensor_fit = sensor_fit
    cam_data.clip_start = clip_start
    cam_data.clip_end = clip_end
    
    # 3. Transform & Link (CRITICAL: Must be done before Depsgraph calc)
    cam_obj.location = location
    cam_obj.rotation_euler = rotation
    
    if collection:
        collection.objects.link(cam_obj)
    else:
        bpy.context.collection.objects.link(cam_obj)
    
    # Force a scene update so matrix_world is valid for calculation
    bpy.context.view_layer.update()

    # 4. Handle Depth of Field with Surface Logic
    if use_dof:
        cam_data.dof.use_dof = True
        cam_data.dof.aperture_fstop = dof_fstop
        
        # --- LOGIC REPLACEMENT START ---
        if dof_focus_object and dof_focus_object.type == 'MESH':
            try:
                # A. Get Evaluated Object (Apply Modifiers)
                depsgraph = bpy.context.evaluated_depsgraph_get()
                obj_eval = dof_focus_object.evaluated_get(depsgraph)
                
                # B. Transform Camera to Object Local Space
                cam_world_loc = cam_obj.matrix_world.translation
                cam_local_loc = obj_eval.matrix_world.inverted() @ cam_world_loc
                
                # C. Find Closest Point on Surface
                success, closest_loc_local, normal, index = obj_eval.closest_point_on_mesh(cam_local_loc)
                
                if success:
                    # D. Convert back to World Space & Calculate Distance
                    closest_loc_world = obj_eval.matrix_world @ closest_loc_local
                    distance = (closest_loc_world - cam_world_loc).length
                    
                    # E. Apply Calculated Distance
                    cam_data.dof.focus_distance = distance
                    cam_data.dof.focus_object = None # Ensure manual distance is used
                    print(f"Surface Focus Success: {distance:.4f}m on '{dof_focus_object.name}'")
                else:
                    print(f"Warning: Could not find surface point on '{dof_focus_object.name}'. Using default distance.")
                    cam_data.dof.focus_distance = dof_focus_distance
                    
            except Exception as e:
                print(f"Error calculating surface focus: {e}")
                cam_data.dof.focus_distance = dof_focus_distance
        
        # Fallback if no object or object is not a mesh
        elif dof_focus_object: 
             print(f"Warning: Focus object '{dof_focus_object.name}' is not a MESH. Using default distance.")
             cam_data.dof.focus_distance = dof_focus_distance
        else:
            # Standard manual distance
            cam_data.dof.focus_distance = dof_focus_distance
        # --- LOGIC REPLACEMENT END ---

    # 5. Set Active
    bpy.context.view_layer.objects.active = cam_obj
    cam_obj.select_set(True)
    
    return cam_obj


def auto_focus(
    camera_name: str,
    camera_parameters: dict,
    scene_name: Optional[str] = None,
) -> dict:
    """
    Set depth of field parameters for an existing camera.
    
    Args:
        camera_name: Name of the camera in the current Blender scene.
        camera_parameters: Dictionary containing camera parameters:
            - 'focal_length': Optional focal length to set
            - 'use_dof': Whether to enable depth of field
            - 'dof_focus_object': Asset ID of the object to focus on
            - 'dof_fstop': F-stop value for depth of field
        scene_name: Optional scene name to operate on.
    
    Returns:
        A dict with:
            - 'success': Boolean indicating success
            - 'focus_distance': The calculated focus distance (if successful)
            - 'error': Error message if failed
    """
    try:
        # Switch to target scene if specified
        if scene_name is not None:
            scene = _switch_to_scene(scene_name)
            if scene is None:
                return {
                    "success": False,
                    "error": f"Scene '{scene_name}' not found.",
                }
        else:
            scene = bpy.context.scene
        
        # Find the camera
        if camera_name not in bpy.data.objects:
            return {
                "success": False,
                "error": f"Camera '{camera_name}' not found.",
            }
        
        cam_obj = bpy.data.objects[camera_name]
        if cam_obj.type != 'CAMERA':
            return {
                "success": False,
                "error": f"Object '{camera_name}' is not a camera.",
            }
        
        cam_data = cam_obj.data
        
        # Set focal length if provided
        focal_length = camera_parameters.get('focal_length')
        if focal_length is not None:
            cam_data.lens = focal_length
        
        # Handle depth of field
        use_dof = camera_parameters.get('use_dof', False)
        dof_focus_object_id = camera_parameters.get('dof_focus_object')
        dof_fstop = camera_parameters.get('dof_fstop', 2.8)
        
        if use_dof:
            cam_data.dof.use_dof = True
            cam_data.dof.aperture_fstop = dof_fstop
            
            if dof_focus_object_id:
                # Find the focus object
                focus_obj = _find_object_by_asset_id(dof_focus_object_id, scene)
                if focus_obj is None:
                    return {
                        "success": False,
                        "error": f"Focus object '{dof_focus_object_id}' not found.",
                    }
                
                # Get mesh object for distance calculation
                mesh_obj = None
                if focus_obj.type == 'MESH':
                    mesh_obj = focus_obj
                else:
                    # Find mesh children for non-MESH objects (EMPTY, ARMATURE, etc.)
                    mesh_children = _get_mesh_children(focus_obj)
                    if mesh_children:
                        mesh_obj = mesh_children[0]
                
                if mesh_obj is None:
                    return {
                        "success": False,
                        "error": f"Could not find mesh for focus object '{dof_focus_object_id}'.",
                    }
                
                # Calculate distance to mesh surface (same logic as create_camera)
                bpy.context.view_layer.update()
                
                depsgraph = bpy.context.evaluated_depsgraph_get()
                obj_eval = mesh_obj.evaluated_get(depsgraph)
                
                cam_world_loc = cam_obj.matrix_world.translation
                cam_local_loc = obj_eval.matrix_world.inverted() @ cam_world_loc
                
                success, closest_loc_local, normal, index = obj_eval.closest_point_on_mesh(cam_local_loc)
                
                if success:
                    closest_loc_world = obj_eval.matrix_world @ closest_loc_local
                    distance = (closest_loc_world - cam_world_loc).length
                    
                    cam_data.dof.focus_distance = distance
                    cam_data.dof.focus_object = None  # Ensure manual distance is used
                    
                    return {
                        "success": True,
                        "focus_distance": distance,
                        "focus_object": dof_focus_object_id,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Could not find surface point on '{mesh_obj.name}'.",
                    }
            else:
                # No focus object specified but DOF enabled
                return {
                    "success": True,
                    "use_dof": True,
                    "message": "DOF enabled without focus object, using existing focus distance.",
                }
        else:
            cam_data.dof.use_dof = False
            return {
                "success": True,
                "use_dof": False,
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def viewport_navigation(operations: list) -> dict:
    """
    Perform viewport navigation operations in the 3D viewport.
    
    Args:
        operations: A list of operation dicts, executed in order. Each dict has:
                   - 'operation': Operation name (string)
                   - 'steps': Number of steps/times to repeat (int)
                   
                   Supported operations:
                   - orbit_left, orbit_right, orbit_up, orbit_down
                   - roll_left, roll_right
                   - pan_left, pan_right, pan_up, pan_down
                   - zoom_in, zoom_out
                   
                   Example: [{"operation": "orbit_left", "steps": 3}, {"operation": "zoom_in", "steps": 2}]
    
    Returns:
        A dict with:
            - 'success': Boolean indicating if all operations completed
            - 'executed': List of operations that were executed with their step counts
            - 'error': Error message if failed (optional)
    """
    area, space, region = _find_3d_viewport()
    
    if not area or not space or not region:
        return {
            "success": False,
            "executed": {},
            "error": "No 3D viewport found"
        }
    
    # Define operation mappings
    # Each operation maps to (operator_function, kwargs)
    # Pan operations use 'manual' to indicate direct view_location manipulation
    operation_map = {
        # Orbit operations (rotate view around the pivot point)
        'orbit_left': ('view3d.view_orbit', {'type': 'ORBITLEFT'}),
        'orbit_right': ('view3d.view_orbit', {'type': 'ORBITRIGHT'}),
        'orbit_up': ('view3d.view_orbit', {'type': 'ORBITUP'}),
        'orbit_down': ('view3d.view_orbit', {'type': 'ORBITDOWN'}),
        
        # Roll operations (orbit the view)
        'roll_left': ('view3d.view_roll', {'angle': math.radians(5)}),
        'roll_right': ('view3d.view_roll', {'angle': math.radians(-5)}),
        
        # Pan operations (move the view) - handled manually via view_location
        # X-axis = horizontal screen direction, Y-axis = vertical screen direction
        'pan_left': ('manual_pan', {'direction': mathutils.Vector((-1, 0, 0))}),
        'pan_right': ('manual_pan', {'direction': mathutils.Vector((1, 0, 0))}),
        'pan_up': ('manual_pan', {'direction': mathutils.Vector((0, 1, 0))}),
        'pan_down': ('manual_pan', {'direction': mathutils.Vector((0, -1, 0))}),
        
        # Zoom operations - dolly camera along its forward axis
        'zoom_in': ('manual_zoom', {'step': 1}),
        'zoom_out': ('manual_zoom', {'step': -1}),
    }
    
    # Pan step size (in Blender units)
    pan_step = 0.5
    
    executed = []
    errors = []
    
    # Get region_3d for manual pan operations
    region_3d = space.region_3d
    
    try:
        with bpy.context.temp_override(
            window=bpy.context.window,
            screen=bpy.context.screen,
            area=area,
            region=region,
            space_data=space
        ):
            for op_dict in operations:
                op_name = op_dict.get('operation', '')
                steps = op_dict.get('steps', 1)
                op_name_lower = op_name.lower()
                
                if op_name_lower not in operation_map:
                    errors.append(f"Unknown operation: {op_name}")
                    continue
                
                op_path, op_kwargs = operation_map[op_name_lower]
                steps = int(steps)
                
                if op_path == 'manual_pan':
                    direction = op_kwargs['direction']
                    # Transform direction by view rotation to get camera-relative movement
                    view_rotation = region_3d.view_rotation
                    world_direction = view_rotation @ direction
                    offset = world_direction * pan_step * abs(steps)
                    
                    # Check if we're in camera view mode
                    if region_3d.view_perspective == 'CAMERA':
                        # Move the camera that is currently being viewed through
                        # space.camera is the camera for this specific viewport (can be overridden)
                        # Falls back to scene.camera if not set
                        camera = space.camera if space.camera else bpy.context.scene.camera
                        if camera:
                            camera.location += offset
                    else:
                        # Standard viewport pan
                        region_3d.view_location += offset
                elif op_path == 'manual_zoom':
                    # Dolly the camera along its local forward axis (-Z)
                    camera = space.camera if space.camera else bpy.context.scene.camera
                    if camera:
                        forward = camera.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))
                        dolly_step = op_kwargs['step']
                        offset = forward * dolly_step * abs(steps)
                        camera.location += offset
                    else:
                        # Fallback for free viewport
                        region_3d.view_distance *= (0.8 if op_kwargs['step'] > 0 else 1.25)
                else:
                    # Use operator for other operations
                    op_module, op_func = op_path.split('.')
                    operator = getattr(getattr(bpy.ops, op_module), op_func)
                    
                    for _ in range(abs(steps)):
                        operator(**op_kwargs)
                
                executed.append({"operation": op_name, "steps": steps})
        
        # Sleep after viewport operations to prevent memory corruption
        time.sleep(0.1)
        
        if errors:
            return {
                "success": False,
                "executed": executed,
                "error": "; ".join(errors)
            }
        
        return {
            "success": True,
            "executed": executed
        }
    
    except Exception as e:
        return {
            "success": False,
            "executed": executed,
            "error": str(e)
        }


# === Optimal Initial Angle Selection ===

class OptimalAngleResponse(BaseModel):
    """Response schema for optimal initial angle selection."""
    optimal_angle: str  # front, front_right, right, back_right, back, back_left, left, or front_left


class AngleVerificationResponse(BaseModel):
    """Response schema for verifying/scoring a selected camera angle."""
    score: int  # 1-10 score for the selected angle
    issues: str  # Description of any issues with the selected angle
    is_acceptable: bool  # Whether the angle is acceptable (score >= 7)
    better_alternative: Optional[str] = None  # Suggested better angle if score < 7

OPTIMAL_ANGLE_SYSTEM_PROMPT = """**Role:**
You are a Camera Angle Selection Expert for 3D scene composition. Your job is to select the optimal initial camera angle from eight images (front, front_right, right, back_right, back, back_left, left, front_left) based on camera instructions.

**Context:**
You are given eight images showing the same 3D scene from different angles: four cardinal directions (front, back, left, right) and four diagonal directions (front_left, front_right, back_left, back_right). The objects with **red highlighted borders** are the focus objects that the camera should prioritize.

**Your Task:**
Based on the camera instruction provided, select which of the eight angles would be the best starting point for further camera adjustments.

**Selection Criteria:**
1. **Visibility:** The focus objects (with red borders) should be clearly visible and not obstructed by other objects.
2. **Camera Description Alignment:** The angle should align with the described shot type (low angle, medium shot, etc.).
3. **Front/Side Preference:** Unless the description explicitly requires a back view, prefer front or side views as they typically show the most important features of characters and objects. Diagonal views (e.g., front_left, front_right) can provide more dynamic compositions.
4. **Obstruction Avoidance:** Choose an angle where other objects do not block the view of the focus objects.

**Important Notes:**
- Only choose back-facing angles ("back", "back_left", "back_right") if the description explicitly requires it, or if all other views have significant obstructions.
- Consider the natural orientation of characters (their faces should typically be visible).
- Diagonal angles often provide more visually interesting compositions than straight-on cardinal angles.
- If there are more than one focus objects, front and back angle often lead to objects blocking each other, **do not select "front" or "back" when there are two or more focus objects**.
- Think about which angle would require the least adjustment to achieve the described shot.
- Select views that shows the front face of the characters or objects as much as possible (front, front_right, front_left, left, right), unless the description explicitly requires a back view (e.g. "over the shoulder").
"""


OPTIMAL_ANGLE_USER_PROMPT_TEMPLATE = """**Camera Instruction:**
- Focus Objects: {focus_on_ids}
- Angle: {angle}
- Distance: {distance}
- Movement: {movement}
- Direction: {direction}
- Description: {description}

**Images:**
You are provided with eight images from different viewing angles:
1. **Front** - Looking from the front
2. **Front Right** - Looking from the front-right diagonal (45°)
3. **Right** - Looking from the right side
4. **Back Right** - Looking from the back-right diagonal (135°)
5. **Back** - Looking from the back
6. **Back Left** - Looking from the back-left diagonal (225°)
7. **Left** - Looking from the left side
8. **Front Left** - Looking from the front-left diagonal (315°)

The objects with **red highlighted borders** are the focus objects ({focus_on_ids}).

Return your answer with:
- optimal_angle: One of "front", "front_right", "right", "back_right", "back", "back_left", "left", or "front_left"
"""

ANGLE_VERIFICATION_SYSTEM_PROMPT = """**Role:**
You are a Camera Composition Critic for 3D cinematography. Your job is to critically evaluate a selected camera angle and score its quality for the given shot requirements.

**Your Task:**
Given the camera instruction, the selected angle, ALL eight reference images, and the selected angle highlighted, evaluate whether this angle choice is optimal for achieving the desired composition. You can compare the selected angle against all other available angles to make an informed judgement.

**Scoring Criteria (1-10):**
- **10:** Perfect angle - focus objects clearly visible, great composition, matches shot description exactly
- **8-9:** Excellent - minor improvements possible but overall very good choice
- **7:** Good - acceptable starting point, some adjustment needed
- **5-6:** Fair - noticeable issues but workable
- **3-4:** Poor - significant problems with visibility or composition
- **1-2:** Unacceptable - focus objects obscured, wrong direction, or completely misaligned with description

**Evaluation Points:**
1. Are ALL focus objects clearly visible and not obscured?
2. Does the angle match the shot description (e.g., if "low angle" is requested, is this achievable from this view)?
3. Are character faces visible (unless back view is explicitly requested)?
4. Is the composition dynamic and visually interesting?
5. Would this angle require minimal adjustment to achieve the described shot?
6. Compare with the other provided angles — is there a clearly better option?
"""

ANGLE_VERIFICATION_USER_PROMPT_TEMPLATE = """**Camera Instruction:**
- Focus Objects: {focus_on_ids}
- Angle: {angle}
- Distance: {distance}
- Movement: {movement}
- Direction: {direction}
- Description: {description}

**Selected Angle:** {selected_angle}

**All Eight Views:**
Below are all eight viewing angles. The currently selected angle ("{selected_angle}") is marked with **(SELECTED)**. Objects with red borders are the focus objects.

**Task:** Evaluate the selected angle and compare it against the other views. Provide:
1. `score`: Integer 1-10 rating
2. `issues`: Description of any problems (empty string if none)
3. `is_acceptable`: True if score >= 7, False otherwise
4. `better_alternative`: If score < 7, suggest a better angle from the eight views based on what you can see (null if acceptable)
"""


def _image_path_to_data_url(image_path: str) -> str:
    """Convert a local image file path to a base64 data URL.
    
    Args:
        image_path: Path to a local image file.
        
    Returns:
        A base64 data URL suitable for API calls.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = 'image/png'
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    base64_data = base64.b64encode(image_data).decode('utf-8')
    return f"data:{mime_type};base64,{base64_data}"


def select_optimal_initial_angle(
    camera_instruction: dict,
    vision_model: str,
    anyllm_api_key: str,
    anyllm_api_base: Optional[str] = None,
    anyllm_provider: str = "gemini",
    scene_name: Optional[str] = None,
    max_image_size: int = 512
) -> dict:
    """
    Select the optimal initial camera angle based on camera instructions and scene composition.
    
    This function captures viewport images of the focus objects from eight directions
    (front, front_right, right, back_right, back, back_left, left, front_left),
    then uses an LLM to determine which angle would be the best starting point for the camera.
    
    Parameters:
    - camera_instruction: Dictionary containing camera setup information:
        - focus_on_ids: List of asset IDs to focus on
        - angle: Camera angle description (e.g., "low angle")
        - distance: Shot distance (e.g., "medium shot")
        - movement: Camera movement type (e.g., "static")
        - direction: Movement direction if applicable
        - description: Natural language description of the shot
        - id: Shot ID (optional)
        - camera_name: Camera name (optional)
    - vision_model: LLM model identifier for vision analysis
    - anyllm_api_key: API key for the LLM service
    - anyllm_api_base: Optional API base URL
    - scene_name: Optional scene name to operate on
    - max_image_size: Maximum size in pixels for captured images (default: 512)
    
    Returns:
    - Dictionary with keys:
        - 'success': Boolean indicating if the operation succeeded
        - 'optimal_angle': The selected angle ('front', 'front_right', 'right', 'back_right', 'back', 'back_left', 'left', or 'front_left')
        - 'image_paths': Dictionary mapping directions to captured image file paths
        - 'verification_history': List of verification iterations with scores and feedback
        - 'error': Error message if failed (optional)
        - 'failed_objects': List of object IDs that could not be found in scene (optional)
    """
    try:
        # Extract focus_on_ids from camera_instruction
        focus_on_ids = camera_instruction.get("focus_on_ids", [])
        if isinstance(focus_on_ids, str):
            focus_on_ids = [focus_on_ids]
        
        if not focus_on_ids:
            return {
                "success": False,
                "error": "No focus_on_ids provided in camera_instruction"
            }
        
        # Step 1: Select the focus objects for outline visibility
        # Uses select_objects_for_outline which selects mesh children for non-mesh objects
        # (e.g., character rigs) to ensure selection outlines are visible in screenshots
        selection_result = select_objects_for_outline(
            asset_id_list=focus_on_ids,
            scene_name=scene_name
        )
        
        if not selection_result.get("success"):
            # Some objects could not be found
            if selection_result.get("selected_objects"):
                # Partial success - some objects were found
                pass
            else:
                # Complete failure - no objects found
                return {
                    "success": False,
                    "error": selection_result.get("error", "Failed to select objects"),
                    "failed_objects": selection_result.get("failed_objects", [])
                }
        
        failed_objects = selection_result.get("failed_objects", [])
        
        # Step 2: Capture turnaround images (8 directions: cardinal + diagonal)
        turnaround_results = get_objects_turnaround_images(max_size=max_image_size)
        
        # Build image paths dictionary and check for failures
        image_paths = {}
        capture_errors = []
        
        for result in turnaround_results:
            direction = result.get("direction")
            if result.get("success"):
                image_paths[direction] = result.get("filepath")
            else:
                capture_errors.append(f"{direction}: {result.get('error', 'Unknown error')}")
        
        if len(image_paths) < 8:
            return {
                "success": False,
                "error": f"Failed to capture all viewport images. Errors: {'; '.join(capture_errors)}",
                "image_paths": image_paths
            }
        
        # Step 3: Build LLM request with all eight images
        user_content = [
            {
                "type": "text",
                "text": OPTIMAL_ANGLE_USER_PROMPT_TEMPLATE.format(
                    focus_on_ids=", ".join(focus_on_ids),
                    angle=camera_instruction.get("angle", "not specified"),
                    distance=camera_instruction.get("distance", "not specified"),
                    movement=camera_instruction.get("movement", "not specified"),
                    direction=camera_instruction.get("direction", "not specified"),
                    description=camera_instruction.get("description", "not specified")
                )
            }
        ]
        
        # Add images in order: front, front_right, right, back_right, back, back_left, left, front_left
        for direction in ["front", "front_right", "right", "back_right", "back", "back_left", "left", "front_left"]:
            if direction in image_paths:
                # Format direction label: "front_right" -> "Front Right"
                direction_label = direction.replace('_', ' ').title()
                user_content.append({
                    "type": "text",
                    "text": f"**{direction_label} View:**"
                })
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": _image_path_to_data_url(image_paths[direction])
                    }
                })
        
        # Step 4: Call LLM for angle selection
        response = _call_llm_with_retry(
            messages=[
                {"role": "system", "content": OPTIMAL_ANGLE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format=OptimalAngleResponse,
            vision_model=vision_model,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            anyllm_provider=anyllm_provider,
            reasoning_effort="high",
        )
        
        # If all retries failed, fallback to "front_left" view
        if response is None:
            result = {
                "success": True,
                "optimal_angle": "front_left",
                "image_paths": image_paths,
                "warning": "LLM failed after retries, using fallback 'front_left' view."
            }
            if failed_objects:
                result["failed_objects"] = failed_objects
            return result
        
        # Step 5: Parse response
        response_content = response.choices[0].message.content
        parsed = json.loads(response_content)
        
        optimal_angle = parsed.get("optimal_angle", "front").lower()
        
        # Validate the angle
        valid_angles = ["front", "front_right", "right", "back_right", "back", "back_left", "left", "front_left"]
        if optimal_angle not in valid_angles:
            optimal_angle = "front"  # Default fallback
        
        # Step 6: Reflection loop - verify and score the selected angle
        max_reflection_iterations = 2
        verification_history = []
        final_angle = optimal_angle
        
        for reflection_iter in range(max_reflection_iterations):
            # Build verification request
            verification_content = [
                {
                    "type": "text",
                    "text": ANGLE_VERIFICATION_USER_PROMPT_TEMPLATE.format(
                        focus_on_ids=", ".join(focus_on_ids),
                        angle=camera_instruction.get("angle", "not specified"),
                        distance=camera_instruction.get("distance", "not specified"),
                        movement=camera_instruction.get("movement", "not specified"),
                        direction=camera_instruction.get("direction", "not specified"),
                        description=camera_instruction.get("description", "not specified"),
                        selected_angle=final_angle
                    )
                }
            ]
            
            # Add all 8 images, marking the selected angle
            for direction in ["front", "front_right", "right", "back_right", "back", "back_left", "left", "front_left"]:
                if direction in image_paths:
                    direction_label = direction.replace('_', ' ').title()
                    selected_marker = " **(SELECTED)**" if direction == final_angle else ""
                    verification_content.append({
                        "type": "text",
                        "text": f"**{direction_label} View{selected_marker}:**"
                    })
                    verification_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": _image_path_to_data_url(image_paths[direction])
                        }
                    })
            
            # Call LLM for verification
            verification_response = _call_llm_with_retry(
                messages=[
                    {"role": "system", "content": ANGLE_VERIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": verification_content}
                ],
                response_format=AngleVerificationResponse,
                vision_model=vision_model,
                anyllm_api_key=anyllm_api_key,
                anyllm_api_base=anyllm_api_base,
                anyllm_provider=anyllm_provider,
                reasoning_effort="high",
            )
            
            if verification_response is None:
                # Verification failed, proceed with current angle
                print(f"Verification LLM call failed, proceeding with angle: {final_angle}")
                break
            
            # Parse verification response
            try:
                verification_content_str = verification_response.choices[0].message.content
                verification_parsed = json.loads(verification_content_str)
                
                score = verification_parsed.get("score", 7)
                is_acceptable = verification_parsed.get("is_acceptable", True)
                issues = verification_parsed.get("issues", "")
                better_alternative = verification_parsed.get("better_alternative")
                
                verification_history.append({
                    "iteration": reflection_iter + 1,
                    "angle": final_angle,
                    "score": score,
                    "issues": issues,
                    "is_acceptable": is_acceptable
                })
                
                if is_acceptable or score >= 7:
                    # Angle is acceptable, exit reflection loop
                    print(f"Angle '{final_angle}' verified with score {score}/10")
                    break
                else:
                    # Angle not acceptable, try the suggested alternative
                    if better_alternative and better_alternative.lower() in valid_angles:
                        # Avoid revisiting already tried angles
                        tried_angles = [h["angle"] for h in verification_history]
                        if better_alternative.lower() not in tried_angles:
                            print(f"Angle '{final_angle}' scored {score}/10. Issues: {issues}. Trying alternative: {better_alternative}")
                            final_angle = better_alternative.lower()
                        else:
                            # Already tried this alternative, stop
                            print(f"Alternative '{better_alternative}' already tried. Keeping current angle: {final_angle}")
                            break
                    else:
                        # No valid alternative suggested, keep current
                        print(f"Angle '{final_angle}' scored {score}/10 but no valid alternative. Keeping current angle.")
                        break
                        
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Failed to parse verification response: {e}. Proceeding with angle: {final_angle}")
                break
        
        result = {
            "success": True,
            "optimal_angle": final_angle,
            "image_paths": image_paths,
            "verification_history": verification_history
        }
        
        if failed_objects:
            result["failed_objects"] = failed_objects
            result["warning"] = f"Some objects could not be found: {', '.join(failed_objects)}"
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# === Camera Parameters Design ===

class CameraParametersResponse(BaseModel):
    """Response schema for camera parameters design."""
    focal_length: float  # in mm
    use_dof: bool
    dof_focus_object: Optional[str] = None  # asset_id of the focus object
    dof_fstop: Optional[float] = None


CAMERA_PARAMETERS_SYSTEM_PROMPT = """**Role:** You are a 3D Cinematography Expert. Design optimal camera parameters based on the shot description and viewport image.

**Focal Length Guide:**
- Close-up: 85-135mm (Intimate/Emotional, portrait compression, subject isolation)
- Medium: 50-75mm (Dialogue/Interaction, natural perspective)
- Wide/Establishing: 24-35mm (Context/Environment, broad coverage)

**Depth of Field (DoF) & Aperture Guide:**
- **Strategy:** Enable `use_dof` for subject separation; disable for action or total environment focus.
- **f/1.2 - 2.0:** Strong blur (Emotional/Portrait).
- **f/2.8 - 4.0:** Moderate blur (Dialogue).
- **f/5.6 - 16:** Deep focus (Action/Wide/No DoF).

**Constraints:**
- If `use_dof` is True, you MUST select exactly ONE `dof_focus_object` from the provided candidate list.
"""

CAMERA_PARAMETERS_USER_PROMPT_TEMPLATE = """**Shot Instructions:**
- Targets: {focus_on_ids}
- Specs: {angle}, {distance}, {movement}, {direction}
- Desc: {description}

**Candidate IDs for DoF:**
{focus_objects_list}

**Visual Context:**
The attached image shows the scene. Objects with red borders are the focus targets.

**Task:**
Analyze the image and instructions to return a JSON response with:
1. `focal_length`: (float) mm based on shot type.
2. `use_dof`: (bool) True if artistic separation is needed.
3. `dof_focus_object`: (str) The specific ID from candidates (null if use_dof is False).
4. `dof_fstop`: (float) Aperture value (null if use_dof is False).
"""


def design_camera_parameters(
    camera_instruction: dict,
    image_path: str,
    vision_model: str,
    anyllm_api_key: str,
    anyllm_api_base: Optional[str] = None,
    anyllm_provider: str = "gemini"
) -> dict:
    """
    Design camera parameters based on camera instruction and scene thumbnail.
    
    This function uses an LLM to analyze the camera instruction and viewport image
    to determine optimal camera parameters including focal length and depth of field.
    
    Parameters:
    - camera_instruction: Dictionary containing camera setup information:
        - focus_on_ids: List of asset IDs to focus on
        - angle: Camera angle description (e.g., "low angle")
        - distance: Shot distance (e.g., "medium shot")
        - movement: Camera movement type (e.g., "static")
        - direction: Movement direction if applicable
        - description: Natural language description of the shot
    - image_path: Path to the viewport image (typically the optimal angle image)
    - vision_model: LLM model identifier for vision analysis
    - anyllm_api_key: API key for the LLM service
    - anyllm_api_base: Optional API base URL
    
    Returns:
    - Dictionary with keys:
        - 'success': Boolean indicating if the operation succeeded
        - 'focal_length': The designed focal length in mm
        - 'use_dof': Whether depth of field should be used
        - 'dof_focus_object': The asset_id of the DoF focus object (if use_dof is True)
        - 'dof_fstop': The f-stop value for DoF (if use_dof is True)
        - 'error': Error message if failed (optional)
    """
    try:
        # Validate image path
        if not os.path.isfile(image_path):
            return {
                "success": False,
                "error": f"Image file not found: {image_path}"
            }
        
        # Extract focus_on_ids from camera_instruction
        focus_on_ids = camera_instruction.get("focus_on_ids", [])
        if isinstance(focus_on_ids, str):
            focus_on_ids = [focus_on_ids]
        
        if not focus_on_ids:
            return {
                "success": False,
                "error": "No focus_on_ids provided in camera_instruction"
            }
        
        # Build focus objects list for the prompt
        focus_objects_list = "\n".join([f"- {obj_id}" for obj_id in focus_on_ids])
        
        # Build LLM request
        user_content = [
            {
                "type": "text",
                "text": CAMERA_PARAMETERS_USER_PROMPT_TEMPLATE.format(
                    focus_on_ids=", ".join(focus_on_ids),
                    angle=camera_instruction.get("angle", "not specified"),
                    distance=camera_instruction.get("distance", "not specified"),
                    movement=camera_instruction.get("movement", "not specified"),
                    direction=camera_instruction.get("direction", "not specified"),
                    description=camera_instruction.get("description", "not specified"),
                    focus_objects_list=focus_objects_list
                )
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_path_to_data_url(image_path)
                }
            }
        ]
        
        # Call LLM for camera parameters design
        response = _call_llm_with_retry(
            messages=[
                {"role": "system", "content": CAMERA_PARAMETERS_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format=CameraParametersResponse,
            vision_model=vision_model,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            anyllm_provider=anyllm_provider,
            reasoning_effort="low",
        )
        
        # If all retries failed, return default parameters
        if response is None:
            return {
                "success": True,
                "focal_length": 50.0,
                "use_dof": False,
                "dof_focus_object": None,
                "dof_fstop": None,
                "warning": "LLM failed after retries, using default parameters."
            }
        
        # Parse response
        response_content = response.choices[0].message.content
        parsed = json.loads(response_content)
        
        focal_length = parsed.get("focal_length", 50.0)
        use_dof = parsed.get("use_dof", False)
        dof_focus_object = parsed.get("dof_focus_object")
        dof_fstop = parsed.get("dof_fstop")
        
        # Validate focal_length range
        if focal_length < 10:
            focal_length = 10.0
        elif focal_length > 300:
            focal_length = 300.0
        
        # Validate DoF parameters
        if use_dof:
            # Ensure dof_focus_object is valid
            if dof_focus_object and dof_focus_object not in focus_on_ids:
                # Try to find a partial match
                matched = False
                for obj_id in focus_on_ids:
                    if obj_id in dof_focus_object or dof_focus_object in obj_id:
                        dof_focus_object = obj_id
                        matched = True
                        break
                if not matched:
                    # Default to first focus object
                    dof_focus_object = focus_on_ids[0]
            elif not dof_focus_object:
                # Default to first focus object if not provided
                dof_focus_object = focus_on_ids[0]
            
            # Validate dof_fstop range
            if dof_fstop is None:
                dof_fstop = 2.8  # Default
            elif dof_fstop < 1.0:
                dof_fstop = 1.2
            elif dof_fstop > 22:
                dof_fstop = 16.0
        else:
            # Clear DoF parameters if not using DoF
            dof_focus_object = None
            dof_fstop = None
        
        result = {
            "success": True,
            "focal_length": focal_length,
            "use_dof": use_dof,
            "dof_focus_object": dof_focus_object,
            "dof_fstop": dof_fstop
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# === Initial Camera Placement ===

def initial_camera_placement(
    optimal_angle: str,
    camera_parameters: dict,
    camera_instruction: dict,
    scene_name: Optional[str] = None
) -> dict:
    """
    Create and place a camera at the optimal viewing angle aligned to the viewport.
    
    This function:
    1. Selects the focus objects from camera_instruction
    2. Sets the viewport lens to match the camera focal length
    3. Sets the viewport to the optimal_angle direction and frames the selected objects
    4. Creates a camera with the specified parameters
    5. Aligns the camera to the current viewport view (camera_to_view)
    6. Restores the viewport lens
    7. Fits the selected objects into the camera frame (camera_to_view_selected)
    
    Parameters:
    - optimal_angle: The viewing direction ('front', 'front_right', 'right', 'back_right',
                      'back', 'back_left', 'left', or 'front_left')
    - camera_parameters: Dictionary from design_camera_parameters containing:
        - focal_length: The focal length in mm
    - camera_instruction: Dictionary containing:
        - focus_on_ids: List of asset IDs to focus on
        - camera_name: Name for the created camera
        - angle, distance, movement, direction, description: Shot metadata
    - scene_name: Optional scene name to operate on
    
    Returns:
    - Dictionary with keys:
        - 'success': Boolean indicating if the operation succeeded
        - 'camera_name': Name of the created camera
        - 'camera_object': The created camera object (bpy.types.Object)
        - 'optimal_angle': The angle used for placement
        - 'focal_length': The focal length applied
        - 'viewport_info': Dict with viewport state for navigation adjustments
        - 'error': Error message if failed (optional)
    """
    try:
        # Validate optimal_angle
        cardinal_direction_map = {
            'front': 'FRONT',
            'back': 'BACK',
            'left': 'LEFT',
            'right': 'RIGHT',
        }
        
        diagonal_rotation_map = _compute_diagonal_rotation_map()
        
        valid_angles = list(cardinal_direction_map.keys()) + list(diagonal_rotation_map.keys())
        
        optimal_angle_lower = optimal_angle.lower()
        print(f"[initial_camera_placement] optimal_angle={optimal_angle_lower}, camera_name={camera_instruction.get('camera_name', 'Camera')}")
        if optimal_angle_lower not in valid_angles:
            return {
                "success": False,
                "error": f"Invalid optimal_angle: {optimal_angle}. Must be one of: {', '.join(valid_angles)}"
            }
        
        # Extract parameters from camera_instruction
        focus_on_ids = camera_instruction.get("focus_on_ids", [])
        if isinstance(focus_on_ids, str):
            focus_on_ids = [focus_on_ids]
        
        camera_name = camera_instruction.get("camera_name", "Camera")
        
        if not focus_on_ids:
            return {
                "success": False,
                "error": "No focus_on_ids provided in camera_instruction"
            }
        
        # Extract camera parameters
        focal_length = camera_parameters.get("focal_length", 50.0)
        print(f"[initial_camera_placement] requested focal_length={focal_length}, focus_on_ids={focus_on_ids}")
        
        # Step 1: Select the focus objects
        selection_result = select_object_from_scene(
            asset_id_list=focus_on_ids,
            scene_name=scene_name
        )
        
        print(f"[initial_camera_placement] Step 1 select_object_from_scene: success={selection_result.get('success')}, selected={selection_result.get('selected_objects')}, failed={selection_result.get('failed_objects')}")
        if not selection_result.get("success") and not selection_result.get("selected_objects"):
            return {
                "success": False,
                "error": selection_result.get("error", "Failed to select focus objects"),
                "failed_objects": selection_result.get("failed_objects", [])
            }
        
        # Step 2: Find 3D viewport
        area, space, region = _find_3d_viewport()
        
        if not area or not space or not region:
            return {
                "success": False,
                "error": "No 3D viewport found"
            }
        
        # Step 3: Set viewport lens to match camera focal length, then set direction and frame.
        # This ensures view_selected frames with the same FOV as the camera, so that
        # camera_to_view produces a placement where objects already fit without cropping.
        original_lens = space.lens
        print(f"[initial_camera_placement] Step 3 viewport original_lens={original_lens}, setting to focal_length={focal_length}")
        print(f"[initial_camera_placement] viewport size: {area.width}x{area.height}")
        print(f"[initial_camera_placement] render resolution: {bpy.context.scene.render.resolution_x}x{bpy.context.scene.render.resolution_y}")
        space.lens = focal_length
        with bpy.context.temp_override(
            window=bpy.context.window,
            screen=bpy.context.screen,
            area=area,
            region=region,
            space_data=space
        ):
            # Set view to the optimal direction (world-space)
            if optimal_angle_lower in cardinal_direction_map:
                # Cardinal direction: use Blender's built-in view_axis
                ret_axis = bpy.ops.view3d.view_axis(type=cardinal_direction_map[optimal_angle_lower], align_active=False)
                print(f"[initial_camera_placement] view_axis({cardinal_direction_map[optimal_angle_lower]}) returned: {ret_axis}")
            else:
                # Diagonal direction: manually set viewport rotation quaternion
                region_3d = space.region_3d
                region_3d.view_rotation = diagonal_rotation_map[optimal_angle_lower]
                region_3d.view_perspective = 'ORTHO'
                print(f"[initial_camera_placement] set diagonal rotation for {optimal_angle_lower}")
            
            # Frame the selected objects using the camera's FOV
            ret_selected = bpy.ops.view3d.view_selected()
            print(f"[initial_camera_placement] view_selected() returned: {ret_selected}")
            print(f"[initial_camera_placement] after view_selected: view_distance={space.region_3d.view_distance}, view_location={tuple(space.region_3d.view_location)}, view_perspective={space.region_3d.view_perspective}")
            print(f"[initial_camera_placement] after view_selected: viewport lens={space.lens}")
        
        # Force viewport update
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        
        # Step 4: Create the camera with specified parameters
        print(f"[initial_camera_placement] Step 4 creating camera '{camera_name}' with focal_length={focal_length}")
        cam_obj = create_camera(
            name=camera_name,
            focal_length=focal_length
        )
        print(f"[initial_camera_placement] camera created: name={cam_obj.name}, focal_length={cam_obj.data.lens}, location={tuple(cam_obj.location)}")
        
        # Step 5: Set the camera as active and selected, and as scene camera
        bpy.ops.object.select_all(action='DESELECT')
        time.sleep(0.1)
        cam_obj.select_set(True)
        bpy.context.view_layer.objects.active = cam_obj
        bpy.context.scene.camera = cam_obj  # Required for camera_to_view operator
        
        # Step 6: Align camera to the current viewport view
        print(f"[initial_camera_placement] Step 6 calling camera_to_view, viewport lens={space.lens}")
        with bpy.context.temp_override(
            window=bpy.context.window,
            screen=bpy.context.screen,
            area=area,
            region=region,
            space_data=space
        ):
            ret_ctv = bpy.ops.view3d.camera_to_view()
        print(f"[initial_camera_placement] camera_to_view() returned: {ret_ctv}")
        print(f"[initial_camera_placement] after camera_to_view: cam location={tuple(cam_obj.location)}, rotation={tuple(cam_obj.rotation_euler)}, cam focal_length={cam_obj.data.lens}")
        
        # Restore viewport lens to original value
        space.lens = original_lens
        print(f"[initial_camera_placement] restored viewport lens to {original_lens}")
        
        # Force update after camera alignment
        bpy.context.view_layer.update()
        
        # Step 7: Fit focus objects into the camera frame by computing the
        # required distance from the combined bounding box and the camera's FOV.
        # (camera_to_view_selected is unreliable here — it returns CANCELLED
        # because the viewport is left in ORTHO state after view_axis.)
        reselect_result = select_object_from_scene(
            asset_id_list=focus_on_ids,
            scene_name=scene_name
        )
        print(f"[initial_camera_placement] Step 7 re-select focus objects: success={reselect_result.get('success')}, selected={reselect_result.get('selected_objects')}")
        
        # Compute combined world-space bounding box of focus objects (including mesh children)
        min_co = mathutils.Vector((float('inf'), float('inf'), float('inf')))
        max_co = mathutils.Vector((float('-inf'), float('-inf'), float('-inf')))
        for obj in bpy.context.selected_objects:
            objects_to_check = [obj] + list(obj.children_recursive)
            for o in objects_to_check:
                if hasattr(o, 'bound_box') and len(o.bound_box) > 0:
                    for corner in o.bound_box:
                        world_co = o.matrix_world @ mathutils.Vector(corner)
                        min_co.x = min(min_co.x, world_co.x)
                        min_co.y = min(min_co.y, world_co.y)
                        min_co.z = min(min_co.z, world_co.z)
                        max_co.x = max(max_co.x, world_co.x)
                        max_co.y = max(max_co.y, world_co.y)
                        max_co.z = max(max_co.z, world_co.z)
        
        bbox_center = (min_co + max_co) / 2
        bbox_size = max_co - min_co
        bbox_radius = bbox_size.length / 2
        print(f"[initial_camera_placement] bbox center={tuple(bbox_center)}, size={tuple(bbox_size)}, radius={bbox_radius}")
        
        # Camera FOV from focal length and sensor
        sensor_width = cam_obj.data.sensor_width
        render = bpy.context.scene.render
        aspect_ratio = render.resolution_x / render.resolution_y
        hfov = 2 * math.atan(sensor_width / (2 * focal_length))
        vfov = 2 * math.atan((sensor_width / aspect_ratio) / (2 * focal_length))
        # Use the narrower FOV to guarantee objects fit in both dimensions
        fit_fov = min(hfov, vfov)
        required_distance = bbox_radius / math.tan(fit_fov / 2)
        # Add 20% margin so objects aren't at the frame edge
        # required_distance *= 1.2
        print(f"[initial_camera_placement] sensor_width={sensor_width}, hfov={math.degrees(hfov):.1f}°, vfov={math.degrees(vfov):.1f}°, fit_fov={math.degrees(fit_fov):.1f}°")
        print(f"[initial_camera_placement] required_distance={required_distance:.3f} (with 20% margin)")
        
        # Reposition camera: keep rotation, move along its backward axis to required_distance from bbox center
        forward = cam_obj.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))
        cam_obj.location = bbox_center - forward * required_distance
        print(f"[initial_camera_placement] repositioned cam location={tuple(cam_obj.location)}")
        bpy.context.view_layer.update()
        
        bpy.context.view_layer.update()
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        
        # Step 8: Collect viewport info for navigation adjustments
        region_3d = space.region_3d
        viewport_info = {
            "view_location": tuple(region_3d.view_location),
            "view_rotation": tuple(region_3d.view_rotation),
            "view_distance": region_3d.view_distance,
            "view_perspective": region_3d.view_perspective,
        }
        
        # Collect camera transform for reference
        camera_transform = {
            "location": tuple(cam_obj.location),
            "rotation_euler": tuple(cam_obj.rotation_euler),
        }
        print(f"[initial_camera_placement] FINAL camera transform: location={camera_transform['location']}, rotation={camera_transform['rotation_euler']}")
        print(f"[initial_camera_placement] FINAL camera focal_length={cam_obj.data.lens}")
        
        return {
            "success": True,
            "camera_name": cam_obj.name,
            "camera_object": cam_obj,
            "optimal_angle": optimal_angle_lower,
            "focal_length": focal_length,
            "viewport_info": viewport_info,
            "camera_transform": camera_transform,
            "focus_on_ids": focus_on_ids,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def _check_focus_objects_in_frame(
    camera_obj: bpy.types.Object,
    focus_on_ids: List[str],
    scene: bpy.types.Scene,
) -> dict:
    """
    Check if focus objects' bounding boxes and geometric centers are within the camera frame.

    Uses bpy_extras.object_utils.world_to_camera_view to project 3D world points
    to camera normalized device coordinates (NDC). A point is in frame when
    x in [0, 1], y in [0, 1], and z > 0 (in front of camera).

    Parameters:
    - camera_obj: The camera object to check against
    - focus_on_ids: List of asset IDs to check
    - scene: The Blender scene

    Returns:
    - Dictionary with:
        - 'all_centers_in_frame': True if every focus object's geometric center is in frame
        - 'all_bboxes_in_frame': True if every focus object's full bounding box is in frame
        - 'objects_status': List of per-object status dicts containing:
            - 'asset_id', 'object_name'
            - 'center_in_frame': bool
            - 'bbox_in_frame': bool (all corners in frame)
            - 'bbox_fraction_in_frame': float (fraction of corners in frame)
            - 'center_ndc': (x, y, z) normalized device coordinates of the center
    """
    results = {
        'all_centers_in_frame': True,
        'all_bboxes_in_frame': True,
        'objects_status': []
    }

    for asset_id in focus_on_ids:
        obj = _find_object_by_asset_id(asset_id, scene)
        if obj is None:
            continue

        # Get world-space bounding box corners
        if obj.type == 'MESH':
            bbox_world = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        else:
            # For non-mesh objects (rigs, empties), compute combined bbox from mesh children
            mesh_children = _get_mesh_children(obj)
            if not mesh_children:
                bbox_world = [obj.matrix_world.translation.copy()]
            else:
                bbox_world = []
                for child in mesh_children:
                    bbox_world.extend(
                        child.matrix_world @ mathutils.Vector(corner)
                        for corner in child.bound_box
                    )

        # Compute geometric center of bounding box
        if bbox_world:
            center = sum(bbox_world, mathutils.Vector()) / len(bbox_world)
        else:
            center = obj.matrix_world.translation.copy()

        # Project center to camera NDC
        center_ndc = world_to_camera_view(scene, camera_obj, center)
        center_in_frame = (0 <= center_ndc.x <= 1 and 0 <= center_ndc.y <= 1 and center_ndc.z > 0)

        # Project bbox corners to camera NDC
        corners_in_frame = 0
        for corner in bbox_world:
            ndc = world_to_camera_view(scene, camera_obj, corner)
            if 0 <= ndc.x <= 1 and 0 <= ndc.y <= 1 and ndc.z > 0:
                corners_in_frame += 1

        total_corners = len(bbox_world)
        bbox_fully_in_frame = (corners_in_frame == total_corners) if total_corners > 0 else True
        bbox_fraction = corners_in_frame / total_corners if total_corners > 0 else 1.0

        if not center_in_frame:
            results['all_centers_in_frame'] = False
        if not bbox_fully_in_frame:
            results['all_bboxes_in_frame'] = False

        results['objects_status'].append({
            'asset_id': asset_id,
            'object_name': obj.name,
            'center_in_frame': center_in_frame,
            'bbox_in_frame': bbox_fully_in_frame,
            'bbox_fraction_in_frame': bbox_fraction,
            'center_ndc': tuple(center_ndc),
        })

    return results


def capture_camera_preview(
    camera_name: str,
    focus_on_ids: List[str],
    max_size: int = 512,
    scene_name: Optional[str] = None,
    extend_ratio: float = 1.0
) -> dict:
    """
    Capture a viewport render image from a camera's perspective with object outlines visible.
    
    This function renders what the camera sees using OpenGL viewport render, with
    selected object outlines overlaid. When extend_ratio > 1.0, the rendered image
    includes surrounding context beyond the camera frame. The actual camera frame
    boundary is drawn as a black rectangle border with the area outside slightly
    dimmed, helping viewers understand what is inside vs. outside the camera frame.
    
    Parameters:
    - camera_name: Name of the camera to render from
    - focus_on_ids: List of asset IDs to highlight with outlines
    - max_size: Maximum size in pixels for the largest dimension (default: 512)
    - scene_name: Optional scene name to operate on
    - extend_ratio: Factor to widen the field of view beyond the camera frame
        (default: 1.0, no extension). A value of 1.5 means the rendered view
        covers 1.5x the angular extent of the original camera in each dimension,
        showing surrounding context with a black border marking the actual frame.
    
    Returns:
    - Dictionary with keys:
        - 'success': Boolean indicating if the operation succeeded
        - 'filepath': Path to the saved image
        - 'width': Image width in pixels
        - 'height': Image height in pixels
        - 'camera_name': Name of the camera used
        - 'error': Error message if failed (optional)
    """
    try:
        # Step 1: Select focus objects for outline visibility
        selection_result = select_objects_for_outline(
            asset_id_list=focus_on_ids,
            scene_name=scene_name
        )
        
        # Continue even if some objects weren't found
        if not selection_result.get("selected_objects"):
            return {
                "success": False,
                "error": selection_result.get("error", "No objects could be selected for outline")
            }
        
        # Step 2: Get scene and find camera
        scene = bpy.context.scene if scene_name is None else bpy.data.scenes.get(scene_name)
        if not scene:
            return {
                "success": False,
                "error": f"Scene '{scene_name}' not found"
            }
        
        camera_obj = None
        if camera_name in bpy.data.objects:
            camera_obj = bpy.data.objects[camera_name]
        
        if not camera_obj or camera_obj.type != 'CAMERA':
            return {
                "success": False,
                "error": f"Camera '{camera_name}' not found or is not a camera object"
            }
        
        # Step 3: Find 3D viewport
        area, space, region = _find_3d_viewport()
        
        if not area or not space:
            return {
                "success": False,
                "error": "No 3D viewport found"
            }
        
        # Generate filepath
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, f"camera_preview_{camera_name}.png")
        
        # Store original settings
        original_scene_camera = scene.camera
        original_filepath = scene.render.filepath
        original_format = scene.render.image_settings.file_format
        original_res_x = scene.render.resolution_x
        original_res_y = scene.render.resolution_y
        original_res_percentage = scene.render.resolution_percentage
        original_shading_type = space.shading.type
        original_lens = camera_obj.data.lens
        
        original_overlay_settings = {}
        if hasattr(space, 'overlay'):
            overlay = space.overlay
            original_overlay_settings['show_overlays'] = overlay.show_overlays
            if hasattr(overlay, 'show_outline_selected'):
                original_overlay_settings['show_outline_selected'] = overlay.show_outline_selected
            # Save all overlay attributes that will be modified
            for attr in [
                'show_cursor', 'show_floor', 'show_axis_x', 'show_axis_y', 'show_axis_z',
                'show_object_origins', 'show_stats', 'show_text', 'show_extras', 'show_bones',
                'show_relationship_lines', 'show_motion_paths', 'show_wireframes',
                'show_face_orientation', 'show_ortho_grid'
            ]:
                if hasattr(overlay, attr):
                    original_overlay_settings[attr] = getattr(overlay, attr)
        
        # Store viewport camera settings
        original_space_camera = space.camera
        original_view_perspective = space.region_3d.view_perspective
        
        try:
            # Set the scene camera to our target camera
            scene.camera = camera_obj
            
            # Set viewport to look through the camera (required for view_context=True)
            space.camera = camera_obj
            space.region_3d.view_perspective = 'CAMERA'
            
            # Temporarily widen FOV for extended view
            if extend_ratio > 1.0:
                camera_obj.data.lens = original_lens / extend_ratio
            
            # Configure viewport shading to MATERIAL preview
            space.shading.type = 'MATERIAL'
            
            # Configure overlays for object outline
            if hasattr(space, 'overlay'):
                overlay = space.overlay
                overlay.show_overlays = True
                if hasattr(overlay, 'show_outline_selected'):
                    overlay.show_outline_selected = True
                # Hide other overlay elements
                for attr in [
                    'show_cursor', 'show_floor', 'show_axis_x', 'show_axis_y', 'show_axis_z',
                    'show_object_origins', 'show_stats', 'show_text', 'show_extras', 'show_bones',
                    'show_relationship_lines', 'show_motion_paths', 'show_wireframes',
                    'show_face_orientation', 'show_ortho_grid'
                ]:
                    if hasattr(overlay, attr):
                        setattr(overlay, attr, False)
            
            # Calculate render resolution maintaining camera aspect ratio
            # Use scene render resolution as base, scale to max_size
            base_width = scene.render.resolution_x
            base_height = scene.render.resolution_y
            aspect_ratio = base_width / base_height
            
            if base_width >= base_height:
                target_width = min(max_size, base_width)
                target_height = int(target_width / aspect_ratio)
            else:
                target_height = min(max_size, base_height)
                target_width = int(target_height * aspect_ratio)
            
            # Apply supersampling for better quality
            supersample_factor = 2
            scene.render.resolution_x = target_width * supersample_factor
            scene.render.resolution_y = target_height * supersample_factor
            scene.render.resolution_percentage = 100
            scene.render.filepath = filepath
            scene.render.image_settings.file_format = 'PNG'
            
            # Force UI update to ensure shading change takes effect
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            
            # THREAD-SAFETY: Same gc guard as _capture_viewport_screenshot_with_outline.
            # See that function's comment for full rationale.
            gc.disable()
            try:
                with bpy.context.temp_override(
                    window=bpy.context.window,
                    screen=bpy.context.screen,
                    area=area,
                    region=region,
                    space_data=space,
                ):
                    bpy.ops.render.opengl(write_still=True, view_context=True)
            finally:
                gc.enable()
            
            # Brief sleep to let background threads / async tasks settle
            time.sleep(0.2)
            
            # Downsample to target size with high-quality filter
            img = Image.open(filepath)
            img_downsampled = img.resize((target_width, target_height), Image.LANCZOS)
            img.close()
            
            # Draw extended view border and dim outside area if applicable
            if extend_ratio > 1.0:
                margin_x = int(target_width * (1 - 1.0 / extend_ratio) / 2)
                margin_y = int(target_height * (1 - 1.0 / extend_ratio) / 2)
                
                # Create semi-transparent overlay to dim area outside camera frame
                overlay = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))
                draw_overlay = ImageDraw.Draw(overlay)
                dim_color = (0, 0, 0, 80)
                # Top strip
                draw_overlay.rectangle([0, 0, target_width, margin_y], fill=dim_color)
                # Bottom strip
                draw_overlay.rectangle([0, target_height - margin_y, target_width, target_height], fill=dim_color)
                # Left strip
                draw_overlay.rectangle([0, margin_y, margin_x, target_height - margin_y], fill=dim_color)
                # Right strip
                draw_overlay.rectangle([target_width - margin_x, margin_y, target_width, target_height - margin_y], fill=dim_color)
                
                img_downsampled = Image.alpha_composite(
                    img_downsampled.convert('RGBA'), overlay
                ).convert('RGB')
                
                # Draw black border around the actual camera frame boundary
                draw = ImageDraw.Draw(img_downsampled)
                border_width = 4
                for i in range(border_width):
                    draw.rectangle(
                        [margin_x - i, margin_y - i,
                         target_width - margin_x + i - 1, target_height - margin_y + i - 1],
                        outline=(0, 0, 0)
                    )
            
            img_downsampled.save(filepath)
            img_downsampled.close()
            
            width = target_width
            height = target_height
            
        finally:
            # Restore all original settings
            scene.camera = original_scene_camera
            scene.render.filepath = original_filepath
            scene.render.image_settings.file_format = original_format
            scene.render.resolution_x = original_res_x
            scene.render.resolution_y = original_res_y
            scene.render.resolution_percentage = original_res_percentage
            space.shading.type = original_shading_type
            camera_obj.data.lens = original_lens
            
            # Restore viewport camera settings
            space.camera = original_space_camera
            space.region_3d.view_perspective = original_view_perspective
            
            # Restore overlay settings
            if hasattr(space, 'overlay'):
                overlay = space.overlay
                for attr, val in original_overlay_settings.items():
                    if hasattr(overlay, attr):
                        setattr(overlay, attr, val)
            
            # Force UI update
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        
        return {
            "success": True,
            "filepath": filepath,
            "width": width,
            "height": height,
            "camera_name": camera_name
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# === Camera Adjustment with LLM ===

OPPOSITE_OPERATIONS = {
    'orbit_left': 'orbit_right',
    'orbit_right': 'orbit_left',
    'orbit_up': 'orbit_down',
    'orbit_down': 'orbit_up',
    'pan_left': 'pan_right',
    'pan_right': 'pan_left',
    'pan_up': 'pan_down',
    'pan_down': 'pan_up',
    'zoom_in': 'zoom_out',
    'zoom_out': 'zoom_in',
    'roll_left': 'roll_right',
    'roll_right': 'roll_left',
}


class CameraAdjustmentResponse(BaseModel):
    """Response schema for camera adjustment decisions."""
    satisfied: bool
    operation: Optional[str] = None


CAMERA_ADJUSTMENT_STATIC_SYSTEM_PROMPT = """**Role:**
You are a Cinematography Expert helping to compose shots for a 3D animated film. Your job is to adjust camera placement to achieve the desired shot composition.

**Context:**
You are viewing a camera preview with focus objects highlighted with **red borders**. You will iteratively adjust the camera until the composition matches the shot description.

**Preview Image:**
The preview image shows an **extended view** beyond the actual camera frame. The actual camera frame boundary is indicated by a **black rectangle border**, and the area outside the border is slightly dimmed. Content **inside** the black border is what the camera captures; content **outside** shows the surrounding context. Use the surrounding context to guide your navigation decisions. For example, if you see a desired object partially visible outside the left border, use `pan_left` to bring it into the camera frame.

**Composition Techniques:**
| Technique | Description | Primary Effect & Usage |
| :--- | :--- | :--- |
| **Rule of Thirds** | Frame divided into a 3x3 grid; subject placed on intersection points. | Creates a natural, balanced, and aesthetically pleasing image; standard for dialogue. |
| **Quadrant System** | Frame divided into 4 equal corners (2x2); subject occupies one corner. | Generates tension, isolation, or imbalance; emphasizes negative space between characters. |
| **Center Framing** | Subject placed exactly in the middle of the frame. | Creates symmetry, rigidity, or intense focus; implies artificiality, authority, or uneasiness. |

**Available Operations:**
- `orbit_left`, `orbit_right`, `orbit_up`, `orbit_down` - Orbit the camera around the center of the frame
- `pan_left`, `pan_right`, `pan_up`, `pan_down` - Shift camera position horizontally/vertically
- `zoom_in`, `zoom_out` - Move camera closer/farther from the center of the frame
- `roll_left`, `roll_right` - Roll/tilt the camera along its view axis

**Orbit Operation Rules (IMPORTANT):**
- Use orbit if the instruction explicitly requires an angle (high angle, low angle) OR to avoid obstacles blocking the view, especially for front or back view for more than one focus objects as they block each other.
- Use **at most TWO orbit per direction** across ALL adjustment rounds (e.g., one `orbit_left` + one `orbit_left` max), except when reverting a bad operation.
- **Keeping subjects visible is more important than matching the exact angle.** Do not sacrifice visibility for angle requirements.

**Character Framing Rules:**
- For character subjects, use `pan` to **position the character's face at the center of the frame BEFORE applying any `zoom_in`**.
- When refining composition, prefer `pan_up` to ensure character faces are centered in the shot.
- Faces are the most important part of character subjects - prioritize face visibility and centering.

**Operation Order:**
You can skip some of the following operation if the composition is already align with the requirements.
1. **Orbit (if needed)** - Only if instruction requires specific angle or to avoid obstacles
2. **Pan** - Position subjects within the frame, **center character faces before zooming**. For example, if the character is at the bottom left of the frame, use `pan_left` and `pan_up` to move the character to the center of the frame.
3. **Zoom** - Adjust frame scale to match the shot's `distance` (close-up, medium, long shot)
4. **Roll** - Apply Dutch angle if specified, usually skipped
5. **Pan to refine** - Fine-tune the final composition, use `pan_up` to center faces
6. **Zoom to refine** - Finally, if the subjects are not in the frame, try zoom out one or more times to include them.

**Recovery from Mistakes:**
If an operation moves subjects out of frame or produces unintended results, immediately use the **opposite operation** to revert (e.g., `orbit_left` to undo `orbit_right`, `zoom_out` to undo `zoom_in`). Then try a different approach or smaller adjustment.

**Guidelines:**
1. Focus objects (red borders) must remain visible in frame at all times - this is the TOP priority.
2. For characters, ensure faces are centered before zooming in.
3. Each operation is a small step; prefer cautious adjustments over aggressive ones.
4. If subjects are near the frame border, use only one operation at a time.
5. Angle requirements are secondary to subject visibility.
6. For satisfied standard: is the composition matches the description? are the focus objects or the characters at the center of the frame? can you see the face of the characters? If not, adjust the camera.

**Response:**
- Set `satisfied` to true when the composition matches the description.
- If not satisfied, provide exactly **one operation** to execute per turn.
- Take small, cautious steps - you will have multiple turns to refine the composition."""


CAMERA_ADJUSTMENT_DYNAMIC_SYSTEM_PROMPT = """**Role:**
You are a Cinematography Expert helping to compose shots for a 3D animated film. Your job is to adjust camera placement to achieve the desired shot composition, considering camera movement.

**Context:**
You are viewing a camera preview with focus objects highlighted with **red borders**. You will iteratively adjust the camera until the composition matches the shot description. This shot involves camera movement, so you must apply **preventive composition**.

**Preview Image:**
The preview image shows an **extended view** beyond the actual camera frame. The actual camera frame boundary is indicated by a **black rectangle border**, and the area outside the border is slightly dimmed. Content **inside** the black border is what the camera captures; content **outside** shows the surrounding context. Use the surrounding context to guide your navigation decisions. For example, if you see a desired object partially visible outside the left border, use `pan_left` to bring it into the camera frame.

**Composition Techniques:**
| Technique | Description | Primary Effect & Usage |
| :--- | :--- | :--- |
| **Rule of Thirds** | Frame divided into a 3x3 grid; subject placed on intersection points. | Creates a natural, balanced, and aesthetically pleasing image; standard for dialogue. |
| **Quadrant System** | Frame divided into 4 equal corners (2x2); subject occupies one corner. | Generates tension, isolation, or imbalance; emphasizes negative space between characters. |
| **Center Framing** | Subject placed exactly in the middle of the frame. | Creates symmetry, rigidity, or intense focus; implies artificiality, authority, or uneasiness. |

**Preventive Composition for Camera Movement:**
Since this shot has camera movement, you must position subjects to account for where they'll be AFTER the movement:
- **Pan left**: Place subject slightly LEFT of ideal position, so after panning left they remain in frame
- **Pan right**: Place subject slightly RIGHT of ideal position
- **Orbit left/right**: Similar to pan, offset in the direction of movement
- **Push in / Zoom in**: Place subject at CENTER of frame so they remain centered after zoom
- **Push out / Zoom out**: Ensure subject is centered; more of the scene will be visible after
- **Tracking**: Ensure subject has leading space in the direction of their movement

**Available Operations:**
- `orbit_left`, `orbit_right`, `orbit_up`, `orbit_down` - Orbit the camera around the center of the frame
- `pan_left`, `pan_right`, `pan_up`, `pan_down` - Shift camera position horizontally/vertically
- `zoom_in`, `zoom_out` - Move camera closer/farther from the center of the frame
- `roll_left`, `roll_right` - Roll/tilt the camera along its view axis.

**Orbit Operation Rules (IMPORTANT):**
- Use orbit if the instruction explicitly requires an angle (high angle, low angle) OR to avoid obstacles blocking the view, especially for front or back view for more than one focus objects as they block each other.
- Use **at most TWO orbit per direction** across ALL adjustment rounds (e.g., one `orbit_left` + one `orbit_left` max), except when reverting a bad operation.
- **Keeping subjects visible is more important than matching the exact angle.** Do not sacrifice visibility for angle requirements.

**Character Framing Rules:**
- For character subjects, use `pan` to **position the character's face at the center of the frame BEFORE applying any `zoom_in`**.
- When refining composition, prefer `pan_up` to ensure character faces are centered in the shot.
- Faces are the most important part of character subjects - prioritize face visibility and centering.

**Operation Order:**
You can skip some of the following operation if the composition is already align with the requirements.
1. **Orbit (if needed)** - Only if instruction requires specific angle or to avoid obstacles
2. **Pan** - Position subjects within the frame, **center character faces before zooming**. For example, if the character is at the bottom left of the frame, use `pan_left` and `pan_up` to move the character to the center of the frame.
3. **Zoom** - Adjust frame scale to match the shot's `distance` (close-up, medium, long shot), only use zoom_in if you have the objects at the center of the frame with previous pan operation(s)
4. **Roll** - Apply Dutch angle if specified, usually skipped
5. **Pan to refine** - Fine-tune the final composition, use `pan_up` to center faces
6. **Zoom to refine** - Finally, if the subjects are not in the frame, try zoom out one or more times to include them.

**Recovery from Mistakes:**
If an operation moves subjects out of frame or produces unintended results, immediately use the **opposite operation** to revert (e.g., `orbit_left` to undo `orbit_right`, `zoom_out` to undo `zoom_in`). Then try a different approach or smaller adjustment.

**Guidelines:**
1. Focus objects (red borders) must remain visible in frame at all times - this is the TOP priority.
2. For characters, ensure faces are centered before zooming in.
3. Apply preventive composition offset based on the `movement` and `direction` specified.
4. Each operation is a small step; prefer cautious adjustments over aggressive ones.
5. If subjects are near the frame border, use only one operation at a time.
6. Angle requirements are secondary to subject visibility.
7. For satisfied standard: is the composition matches the description? are the focus objects or the characters at the center of the frame? can you see the face of the characters? If not, adjust the camera.

**Response:**
- Set `satisfied` to true when the composition matches the description with proper preventive offset.
- If not satisfied, provide exactly **one operation** to execute per turn.
- Take small, cautious steps - you will have multiple turns to refine the composition."""


CAMERA_ADJUSTMENT_USER_PROMPT_STATIC = """**Shot Requirements:**
- Focus Objects: {focus_on_ids}
- Angle: {angle}
- Distance: {distance}
- Description: {description}

**Current Camera Preview:**
The image shows an extended camera view. The **black rectangle border** indicates the actual camera frame boundary — only content inside this border will appear in the final shot. The area outside the border is dimmed surrounding context to help you decide navigation. Objects with **red borders** are the focus objects.

Analyze the composition (based on what is inside the black border) and either:
1. Set `satisfied: true` if the shot matches the requirements
2. Provide exactly **one operation** to adjust the camera"""


CAMERA_ADJUSTMENT_USER_PROMPT_DYNAMIC = """**Shot Requirements:**
- Focus Objects: {focus_on_ids}
- Angle: {angle}
- Distance: {distance}
- Movement: {movement}
- Direction: {direction}
- Description: {description}

**Current Camera Preview:**
The image shows an extended camera view. The **black rectangle border** indicates the actual camera frame boundary — only content inside this border will appear in the final shot. The area outside the border is dimmed surrounding context to help you decide navigation. Objects with **red borders** are the focus objects.

Remember to apply preventive composition for the `{movement}` movement (direction: {direction}).

Analyze the composition (based on what is inside the black border) and either:
1. Set `satisfied: true` if the shot is properly composed with preventive offset for the movement
2. Provide exactly **one operation** to adjust the camera"""


def adjust_camera_placement(
    camera_instruction: dict,
    camera_parameters: dict,
    initial_angle_image_path: str,
    vision_model: str,
    anyllm_api_key: str,
    anyllm_api_base: Optional[str] = None,
    anyllm_provider: str = "gemini",
    max_rounds: int = 10,
    scene_name: Optional[str] = None,
    preview_image_save_dir: Optional[str] = None
) -> dict:
    """
    Iteratively adjust camera placement using LLM guidance until the desired composition is achieved.
    
    This function uses a multi-turn conversation with an LLM to analyze camera previews and
    suggest adjustments until the shot matches the description in camera_instruction.
    
    Parameters:
    - camera_instruction: Dictionary containing:
        - focus_on_ids: List of asset IDs to focus on
        - angle: Camera angle (eye-level, high angle, low angle)
        - distance: Shot distance (close-up, medium shot, long shot)
        - movement: Camera movement type (static, pan, orbit, push in, etc.)
        - direction: Movement direction if applicable (left, right, up, down)
        - description: Natural language description of the shot
        - camera_name: Name of the camera to adjust
    - camera_parameters: Dictionary with focal_length, use_dof, etc.
    - initial_angle_image_path: Path to the initial camera preview image
    - vision_model: LLM model identifier for vision analysis
    - anyllm_api_key: API key for the LLM service
    - anyllm_api_base: Optional API base URL
    - max_rounds: Maximum number of adjustment rounds (default: 5)
    - scene_name: Optional scene name to operate on
    - preview_image_save_dir: Optional directory to save per-step preview images with unique names
    
    Returns:
    - Dictionary with keys:
        - 'success': Boolean indicating if adjustment completed successfully
        - 'satisfied': Boolean indicating if LLM was satisfied with final result
        - 'rounds_used': Number of adjustment rounds used
        - 'camera_name': Name of the camera
        - 'camera_object': The camera Blender object
        - 'camera_transform': Dict with location and rotation_euler
        - 'final_image_path': Path to the final camera preview
        - 'operations_executed': List of all operations executed
        - 'error': Error message if failed (optional)
    """
    try:
        # Extract instruction fields
        focus_on_ids = camera_instruction.get('focus_on_ids', [])
        angle = camera_instruction.get('angle', 'eye-level')
        distance = camera_instruction.get('distance', 'medium shot')
        movement = camera_instruction.get('movement', 'static')
        direction = camera_instruction.get('direction')
        description = camera_instruction.get('description', '')
        camera_name = camera_instruction.get('camera_name', 'Camera')
        
        # Get the camera object
        if camera_name not in bpy.data.objects:
            return {
                "success": False,
                "error": f"Camera '{camera_name}' not found."
            }
        
        cam_obj = bpy.data.objects[camera_name]
        if cam_obj.type != 'CAMERA':
            return {
                "success": False,
                "error": f"Object '{camera_name}' is not a camera."
            }
        
        # Determine if static or dynamic movement
        is_static = movement.lower() == 'static'
        
        # Select system prompt and user prompt template
        if is_static:
            system_prompt = CAMERA_ADJUSTMENT_STATIC_SYSTEM_PROMPT
            user_prompt_template = CAMERA_ADJUSTMENT_USER_PROMPT_STATIC
        else:
            system_prompt = CAMERA_ADJUSTMENT_DYNAMIC_SYSTEM_PROMPT
            user_prompt_template = CAMERA_ADJUSTMENT_USER_PROMPT_DYNAMIC
        
        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Track execution
        all_operations_executed = []
        current_image_path = initial_angle_image_path
        satisfied = False
        rounds_used = 0
        reflection_message = ""  # Prepended to next user prompt after revert or bbox warning
        
        # Orbit count enforcement: max 2 per direction (excluding reverts)
        MAX_ORBIT_PER_DIRECTION = 2
        orbit_counts = {
            'orbit_left': 0, 'orbit_right': 0,
            'orbit_up': 0, 'orbit_down': 0,
        }
        
        # Create a temp subdir for per-step preview images
        step_preview_dir = os.path.join(tempfile.gettempdir(), f"camera_steps_{camera_name}")
        os.makedirs(step_preview_dir, exist_ok=True)
        print(f"[adjust_camera_placement] Step previews will be saved to: {step_preview_dir}")
        
        # Save the initial image before any adjustments
        if os.path.exists(initial_angle_image_path):
            shutil.copy2(initial_angle_image_path, os.path.join(step_preview_dir, f"{camera_name}_step_0_initial.png"))
            if preview_image_save_dir:
                os.makedirs(preview_image_save_dir, exist_ok=True)
                shutil.copy2(initial_angle_image_path, os.path.join(preview_image_save_dir, f"{camera_name}_step_0_initial.png"))
        
        # Find 3D viewport for camera alignment
        area, space, region = _find_3d_viewport()
        
        if not area or not space or not region:
            return {
                "success": False,
                "error": "No 3D viewport found."
            }
        
        # Set viewport to look through the camera
        original_space_camera = space.camera
        original_view_perspective = space.region_3d.view_perspective
        original_lock_camera = space.lock_camera
        
        space.camera = cam_obj
        space.region_3d.view_perspective = 'CAMERA'
        space.lock_camera = True  # Lock camera to view so navigation moves the camera
        bpy.context.scene.camera = cam_obj
        
        try:
            for round_num in range(max_rounds):
                rounds_used = round_num + 1
                
                # Format user prompt
                if is_static:
                    user_prompt = user_prompt_template.format(
                        focus_on_ids=focus_on_ids,
                        angle=angle,
                        distance=distance,
                        description=description
                    )
                else:
                    user_prompt = user_prompt_template.format(
                        focus_on_ids=focus_on_ids,
                        angle=angle,
                        distance=distance,
                        movement=movement,
                        direction=direction or 'N/A',
                        description=description
                    )

                if round_num == max_rounds - 1:
                    user_prompt += (
                        "\n\nIMPORTANT: This is the last adjustment round. Do not take radical operations. "
                        "Keeping the target object(s) in the frame is more important than the aesthetic of composition."
                    )
                
                # Prepend reflection message from previous round (revert notice or bbox warning)
                if reflection_message:
                    user_prompt = reflection_message + "\n\n" + user_prompt
                    reflection_message = ""
                
                # Build user message with image
                user_content = [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_path_to_data_url(current_image_path)}
                    }
                ]
                
                # Prune old images from conversation to reduce context size.
                # Keep images only in the most recent user message (the one we're about to add).
                # Text content (operation history, reflection messages) is preserved.
                for msg in messages:
                    if msg["role"] == "user" and isinstance(msg.get("content"), list):
                        msg["content"] = [
                            item for item in msg["content"]
                            if item.get("type") != "image_url"
                        ]
                
                messages.append({"role": "user", "content": user_content})
                
                # Call LLM
                llm_response = _call_llm_with_retry(
                    messages=messages,
                    response_format=CameraAdjustmentResponse,
                    vision_model=vision_model,
                    anyllm_api_key=anyllm_api_key,
                    anyllm_api_base=anyllm_api_base,
                    anyllm_provider=anyllm_provider,
                    reasoning_effort="low",
                )
                
                # If all retries failed, break the adjustment loop and use current position
                if llm_response is None:
                    # Remove the last user message since we didn't get a response
                    messages.pop()
                    break
                
                # Parse response
                response_content = llm_response.choices[0].message.content
                parsed = json.loads(response_content)
                
                # Add assistant response to conversation
                messages.append({"role": "assistant", "content": response_content})
                
                satisfied = parsed.get('satisfied', False)
                
                if satisfied:
                    break
                
                # Collect operation (single operation per turn)
                operations = []
                op_value = parsed.get('operation')
                if op_value and op_value.strip():
                    operations.append({
                        "operation": op_value.strip(),
                        "steps": 1
                    })
                
                if not operations:
                    # No operations but not satisfied - end loop
                    break
                
                # Enforce orbit count limit
                op_name_lower = operations[0]["operation"].lower()
                if op_name_lower in orbit_counts and orbit_counts[op_name_lower] >= MAX_ORBIT_PER_DIRECTION:
                    reflection_message = (
                        f"**BLOCKED OPERATION:** `{op_name_lower}` has already been used "
                        f"{MAX_ORBIT_PER_DIRECTION} times (the maximum allowed). "
                        f"Choose a different operation (e.g., pan or zoom instead)."
                    )
                    print(f"[adjust_camera_placement] Blocked '{op_name_lower}' (orbit limit reached: {orbit_counts[op_name_lower]})")
                    # Remove the assistant message that suggested the blocked op
                    # so the LLM doesn't see it as executed
                    messages.pop()
                    continue
                
                # Execute operations via viewport_navigation
                nav_result = viewport_navigation(operations)
                
                if nav_result.get('success') or nav_result.get('executed'):
                    all_operations_executed.extend(nav_result.get('executed', []))
                    # Track orbit counts for successfully executed operations
                    if op_name_lower in orbit_counts:
                        orbit_counts[op_name_lower] += 1
                
                # Camera is automatically moved because lock_camera is True
                # Force update to ensure camera transform is applied
                bpy.context.view_layer.update()
                
                # --- Reflection: validate focus objects are still in camera frame ---
                scene_for_check = bpy.context.scene if scene_name is None else bpy.data.scenes.get(scene_name, bpy.context.scene)
                frame_check = _check_focus_objects_in_frame(cam_obj, focus_on_ids, scene_for_check)
                op_label = op_value.strip()
                
                # Check if any geometric center moved out of frame -> revert
                centers_out = [s for s in frame_check['objects_status'] if not s['center_in_frame']]
                if centers_out:
                    # Revert the operation
                    reverse_op = OPPOSITE_OPERATIONS.get(op_label.lower())
                    if reverse_op:
                        print(f"[adjust_camera_placement] Reverting '{op_label}' -> '{reverse_op}' (center out of frame)")
                        viewport_navigation([{"operation": reverse_op, "steps": 1}])
                        bpy.context.view_layer.update()
                        # Undo the orbit count increment since the operation was reverted
                        if op_name_lower in orbit_counts and orbit_counts[op_name_lower] > 0:
                            orbit_counts[op_name_lower] -= 1
                    
                    obj_names = [s['object_name'] for s in centers_out]
                    reflection_message = (
                        f"**INVALID OPERATION (auto-reverted):** Your last operation `{op_label}` caused "
                        f"{', '.join(obj_names)}'s geometric center to move out of the camera frame. "
                        f"The operation has been reverted. Try other operations instead "
                        f"(e.g., if `{op_label}` failed, consider a different direction or use zoom)."
                    )
                    print('reflection_message', reflection_message)
                    # Don't capture new preview — reuse the previous image
                    continue
                
                # Check if bounding box is partially out of frame -> warning
                bboxes_partial = [s for s in frame_check['objects_status']
                                  if not s['bbox_in_frame'] and s['center_in_frame']]
                if bboxes_partial:
                    dist_lower = distance.lower()
                    is_closeup_or_medium = any(k in dist_lower for k in ['close', 'medium'])
                    
                    warning_parts = []
                    for s in bboxes_partial:
                        pct = int(s['bbox_fraction_in_frame'] * 100)
                        warning_parts.append(f"{s['object_name']} ({pct}% of bounding box visible)")
                    
                    if is_closeup_or_medium:
                        reflection_message = (
                            f"**NOTE:** {'; '.join(warning_parts)} — parts of the object extend beyond the "
                            f"camera frame. This may be acceptable for a {distance} shot. "
                            f"Proceed if the composition looks right, or consider `zoom_out` or `pan` if too much is cropped."
                        )
                    else:
                        reflection_message = (
                            f"**WARNING:** {'; '.join(warning_parts)} — parts of the object are outside the "
                            f"camera frame. Consider reverting with `{OPPOSITE_OPERATIONS.get(op_label.lower(), 'the opposite operation')}` "
                            f"unless the object is very large or cropping is intentional."
                        )
                    print('reflection_message', reflection_message)
                
                # Capture new preview with extended view for LLM context
                preview_result = capture_camera_preview(
                    camera_name=camera_name,
                    focus_on_ids=focus_on_ids,
                    max_size=512,
                    scene_name=scene_name,
                    extend_ratio=1.5
                )
                
                if preview_result.get('success'):
                    current_image_path = preview_result['filepath']
                    op_suffix = ("_" + op_value.strip().replace(" ", "_")) if op_value and op_value.strip() else ""
                    step_filename = f"{camera_name}_step_{round_num + 1}{op_suffix}.png"
                    step_dest = os.path.join(step_preview_dir, step_filename)
                    shutil.copy2(current_image_path, step_dest)
                    if preview_image_save_dir:
                        os.makedirs(preview_image_save_dir, exist_ok=True)
                        shutil.copy2(current_image_path, os.path.join(preview_image_save_dir, step_filename))
                else:
                    # Continue with old image if capture failed
                    pass
            
            # Safety check: ensure camera is not below ground (z < 0)
            # if cam_obj.location.z < 0:
            #     cam_obj.location.z = 0
            #     bpy.context.view_layer.update()
            
            # Capture final preview
            preview_result = capture_camera_preview(
                camera_name=camera_name,
                focus_on_ids=focus_on_ids,
                max_size=512,
                scene_name=scene_name
            )
            if preview_result.get('success'):
                current_image_path = preview_result['filepath']
                final_step_filename = f"{camera_name}_step_final.png"
                shutil.copy2(current_image_path, os.path.join(step_preview_dir, final_step_filename))
                if preview_image_save_dir:
                    os.makedirs(preview_image_save_dir, exist_ok=True)
                    shutil.copy2(current_image_path, os.path.join(preview_image_save_dir, final_step_filename))
            
            # Get final camera transform
            camera_transform = {
                "location": tuple(cam_obj.location),
                "rotation_euler": tuple(cam_obj.rotation_euler),
            }
            
            return {
                "success": True,
                "satisfied": satisfied,
                "rounds_used": rounds_used,
                "camera_name": cam_obj.name,
                "camera_object": cam_obj,
                "camera_parameters": camera_parameters,
                "camera_transform": camera_transform,
                "final_image_path": current_image_path,
                "operations_executed": all_operations_executed,
            }
            
        finally:
            # Restore viewport settings
            space.lock_camera = original_lock_camera
            space.camera = original_space_camera
            space.region_3d.view_perspective = original_view_perspective
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def create_and_place_camera_for_start_of_a_shot(
    camera_instruction: dict,
    vision_model: str,
    anyllm_api_key: str,
    anyllm_api_base: Optional[str] = None,
    anyllm_provider: str = "gemini",
    max_adjustment_rounds: int = 10,
    scene_name: Optional[str] = None,
    preview_image_save_dir: Optional[str] = None
) -> dict:
    """
    Create and place a camera for the start of a shot using LLM-guided workflow.
    
    This function orchestrates the complete camera setup process:
    1. Select optimal initial angle (front/front_right/right/back_right/back/back_left/left/front_left)
    2. Design camera parameters (focal length, DoF settings)
    3. Initial camera placement at the optimal angle
    4. Iterative adjustment using LLM guidance
    5. Apply auto focus if DoF is enabled
    
    Parameters:
    - camera_instruction: Dictionary containing:
        - focus_on_ids: List of asset IDs to focus on
        - angle: Camera angle (eye-level, high angle, low angle)
        - distance: Shot distance (close-up, medium shot, long shot)
        - movement: Camera movement type (static, pan, orbit, etc.)
        - direction: Movement direction if applicable
        - description: Natural language description of the shot
        - camera_name: Name for the camera to create
    - vision_model: LLM model identifier for vision analysis
    - anyllm_api_key: API key for the LLM service
    - anyllm_api_base: Optional API base URL
    - max_adjustment_rounds: Maximum rounds for camera adjustment (default: 5)
    - scene_name: Optional scene name to operate on
    
    Returns:
    - Dictionary with keys:
        - 'success': Boolean indicating overall success
        - 'camera_name': Name of the created camera
        - 'camera_object': The camera Blender object
        - 'camera_parameters': Designed camera parameters
        - 'camera_transform': Final camera location and rotation
        - 'optimal_angle': The initial angle selected
        - 'adjustment_satisfied': Whether LLM was satisfied with final placement
        - 'adjustment_rounds': Number of adjustment rounds used
        - 'final_image_path': Path to final camera preview
        - 'dof_applied': Whether DoF was applied
        - 'focus_distance': Focus distance if DoF was applied
        - 'error': Error message if failed (optional)
        - 'step_failed': Which step failed if error occurred (optional)
    """
    try:
        focus_on_ids = camera_instruction.get('focus_on_ids', [])
        camera_name = camera_instruction.get('camera_name', 'Camera')
        
        # Step 1: Select optimal initial angle
        angle_result = select_optimal_initial_angle(
            camera_instruction=camera_instruction,
            vision_model=vision_model,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            anyllm_provider=anyllm_provider,
            scene_name=scene_name
        )
        
        if not angle_result.get('success'):
            return {
                "success": False,
                "step_failed": "select_optimal_initial_angle",
                "error": angle_result.get('error', 'Failed to select optimal angle')
            }
        
        optimal_angle = angle_result['optimal_angle']
        angle_image_paths = angle_result.get('image_paths', {})
        initial_image_path = angle_image_paths.get(optimal_angle)
        
        # Save turnaround images to the step preview dir
        step_preview_dir = os.path.join(tempfile.gettempdir(), f"camera_steps_{camera_name}")
        os.makedirs(step_preview_dir, exist_ok=True)
        for direction, img_path in angle_image_paths.items():
            if img_path and os.path.exists(img_path):
                dest = os.path.join(step_preview_dir, f"{camera_name}_turnaround_{direction}.png")
                shutil.copy2(img_path, dest)
                if preview_image_save_dir:
                    os.makedirs(preview_image_save_dir, exist_ok=True)
                    shutil.copy2(img_path, os.path.join(preview_image_save_dir, f"{camera_name}_turnaround_{direction}.png"))
        
        # Step 2: Design camera parameters
        if not initial_image_path:
            return {
                "success": False,
                "step_failed": "design_camera_parameters",
                "error": f"No image available for optimal angle '{optimal_angle}'"
            }
        
        params_result = design_camera_parameters(
            camera_instruction=camera_instruction,
            image_path=initial_image_path,
            vision_model=vision_model,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            anyllm_provider=anyllm_provider
        )
        
        if not params_result.get('success'):
            return {
                "success": False,
                "step_failed": "design_camera_parameters",
                "error": params_result.get('error', 'Failed to design camera parameters')
            }
        
        camera_parameters = {
            'focal_length': params_result.get('focal_length'),
            'use_dof': params_result.get('use_dof', False),
            'dof_focus_object': params_result.get('dof_focus_object'),
            'dof_fstop': params_result.get('dof_fstop')
        }
        
        # Step 3: Initial camera placement
        placement_result = initial_camera_placement(
            optimal_angle=optimal_angle,
            camera_parameters=camera_parameters,
            camera_instruction=camera_instruction,
            scene_name=scene_name
        )
        
        if not placement_result.get('success'):
            return {
                "success": False,
                "step_failed": "initial_camera_placement",
                "error": placement_result.get('error', 'Failed to place camera initially')
            }
        
        cam_obj = placement_result.get('camera_object')
        
        # Capture initial preview for adjustment with extended view for LLM context
        preview_result = capture_camera_preview(
            camera_name=camera_name,
            focus_on_ids=focus_on_ids,
            max_size=512,
            scene_name=scene_name,
            extend_ratio=1.5
        )
        
        if not preview_result.get('success'):
            return {
                "success": False,
                "step_failed": "capture_camera_preview",
                "error": preview_result.get('error', 'Failed to capture initial preview')
            }
        
        initial_preview_path = preview_result['filepath']
        
        # Step 4: Adjust camera placement
        adjustment_result = adjust_camera_placement(
            camera_instruction=camera_instruction,
            camera_parameters=camera_parameters,
            initial_angle_image_path=initial_preview_path,
            vision_model=vision_model,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            anyllm_provider=anyllm_provider,
            max_rounds=max_adjustment_rounds,
            scene_name=scene_name,
            preview_image_save_dir=preview_image_save_dir
        )
        
        print(adjustment_result)
        if not adjustment_result.get('success'):
            return {
                "success": False,
                "step_failed": "adjust_camera_placement",
                "error": adjustment_result.get('error', 'Failed to adjust camera placement')
            }
        
        # Step 5: Apply auto focus if DoF is enabled
        dof_applied = False
        focus_distance = None
        
        if camera_parameters.get('use_dof') and camera_parameters.get('dof_focus_object'):
            focus_result = auto_focus(
                camera_name=camera_name,
                camera_parameters=camera_parameters,
                scene_name=scene_name
            )
            
            if focus_result.get('success'):
                dof_applied = True
                focus_distance = focus_result.get('focus_distance')
        
        # Get final camera transform
        camera_transform = {
            "location": tuple(cam_obj.location),
            "rotation_euler": tuple(cam_obj.rotation_euler),
        }
        
        return {
            "success": True,
            "camera_name": cam_obj.name,
            "camera_object": cam_obj,
            "camera_parameters": camera_parameters,
            "camera_transform": camera_transform,
            "optimal_angle": optimal_angle,
            "adjustment_satisfied": adjustment_result.get('satisfied', False),
            "adjustment_rounds": adjustment_result.get('rounds_used', 0),
            "final_image_path": adjustment_result.get('final_image_path'),
            "dof_applied": dof_applied,
            "focus_distance": focus_distance,
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def create_and_place_camera_for_shot(
    camera_instruction: dict,
    vision_model: str,
    anyllm_api_key: str,
    anyllm_api_base: Optional[str] = None,
    anyllm_provider: str = "gemini",
    start_frame: int = 1,
    end_frame: int = 73,
    max_adjustment_rounds: int = 10,
    scene_name: Optional[str] = None,
    preview_image_save_dir: Optional[str] = None
) -> dict:
    """
    Create and place a camera for a shot with animation support for dynamic movements.
    
    This function:
    1. Goes to start_frame and places camera using create_and_place_camera_for_start_of_a_shot
    2. If movement is static, done
    3. If movement is dynamic (pan, orbit, zoom in, zoom out):
       - Inserts keyframe at start_frame
       - Goes to end_frame
       - Executes movement operations (2 steps for pan/zoom, 1 step for orbit)
       - Inserts keyframe at end_frame with smooth interpolation
       - Applies auto_focus at end frame if DoF is enabled
    
    Parameters:
    - camera_instruction: Dictionary containing camera setup info (see create_and_place_camera_for_start_of_a_shot)
    - vision_model: LLM model identifier for vision analysis
    - anyllm_api_key: API key for the LLM service
    - anyllm_api_base: Optional API base URL
    - start_frame: Starting frame for the shot (default: 1)
    - end_frame: Ending frame for the shot (default: 73)
    - max_adjustment_rounds: Maximum rounds for camera adjustment (default: 5)
    - scene_name: Optional scene name to operate on
    
    Returns:
    - Dictionary with keys:
        - 'success': Boolean indicating overall success
        - 'camera_name': Name of the created camera
        - 'camera_object': The camera Blender object
        - 'camera_parameters': Designed camera parameters
        - 'start_transform': Camera transform at start_frame
        - 'end_transform': Camera transform at end_frame (if animated)
        - 'is_animated': Whether camera has animation
        - 'movement': The movement type
        - 'dof_applied': Whether DoF was applied
        - 'error': Error message if failed (optional)
    """
    try:
        movement = camera_instruction.get('movement', 'static')
        direction = camera_instruction.get('direction')
        camera_name = camera_instruction.get('camera_name', 'Camera')
        
        # Switch to scene first if scene_name is provided
        if scene_name:
            scene = _switch_to_scene(scene_name)
            if not scene:
                return {"success": False, "error": f"Scene '{scene_name}' not found."}
        else:
            scene = bpy.context.scene
        
        # Step 1: Go to start_frame
        scene.frame_set(start_frame)
        
        # Step 2: Place camera at start position
        start_result = create_and_place_camera_for_start_of_a_shot(
            camera_instruction=camera_instruction,
            vision_model=vision_model,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            anyllm_provider=anyllm_provider,
            max_adjustment_rounds=max_adjustment_rounds,
            scene_name=scene_name,
            preview_image_save_dir=preview_image_save_dir
        )
        
        if not start_result.get('success'):
            return start_result
        
        cam_obj = start_result['camera_object']
        camera_parameters = start_result['camera_parameters']
        start_transform = start_result['camera_transform']
        
        # Step 3: If static, we're done
        # Also skip animation if placement was not satisfied (even for dynamic movements)
        if movement.lower() == 'static' or not start_result.get('adjustment_satisfied', True):
            return {
                "success": True,
                "camera_name": cam_obj.name,
                "camera_object": cam_obj,
                "camera_parameters": camera_parameters,
                "start_transform": start_transform,
                "end_transform": None,
                "is_animated": False,
                "movement": movement,
                "dof_applied": start_result.get('dof_applied', False),
                "focus_distance": start_result.get('focus_distance'),
                "final_image_path": start_result.get('final_image_path'),
                "placement_satisfied": start_result.get('satisfied', True),
            }
        
        # Step 4: For dynamic movements, create animation
        # Insert keyframe at start_frame
        cam_obj.keyframe_insert(data_path="location", frame=start_frame)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=start_frame)
        
        # Determine operation and steps based on movement type
        movement_lower = movement.lower()
        if movement_lower == 'pan':
            # Pan: shift camera position
            if direction and direction.lower() in ['left', 'right', 'up', 'down']:
                op_name = f"pan_{direction.lower()}"
            else:
                op_name = "pan_left"  # Default
            steps = 2
        elif movement_lower == 'orbit':
            # Orbit: rotate around focus
            if direction and direction.lower() in ['left', 'right', 'up', 'down']:
                op_name = f"orbit_{direction.lower()}"
            else:
                op_name = "orbit_left"  # Default
            steps = 1
        elif movement_lower == 'zoom in':
            op_name = "zoom_in"
            steps = 2
        elif movement_lower == 'zoom out':
            op_name = "zoom_out"
            steps = 2
        else:
            # Unknown movement, treat as static
            return {
                "success": True,
                "camera_name": cam_obj.name,
                "camera_object": cam_obj,
                "camera_parameters": camera_parameters,
                "start_transform": start_transform,
                "end_transform": None,
                "is_animated": False,
                "movement": movement,
                "dof_applied": start_result.get('dof_applied', False),
                "focus_distance": start_result.get('focus_distance'),
                "final_image_path": start_result.get('final_image_path'),
            }
        
        # Step 5: Go to end_frame
        scene.frame_set(end_frame)
        
        # Find 3D viewport for camera operations
        area, space, region = _find_3d_viewport()
        
        if not area or not space or not region:
            return {"success": False, "error": "No 3D viewport found."}
        
        # Set viewport to camera view with lock
        original_space_camera = space.camera
        original_view_perspective = space.region_3d.view_perspective
        original_lock_camera = space.lock_camera
        
        space.camera = cam_obj
        space.region_3d.view_perspective = 'CAMERA'
        space.lock_camera = True
        bpy.context.scene.camera = cam_obj
        
        try:
            # Execute the movement operation
            operations = [{"operation": op_name, "steps": steps}]
            nav_result = viewport_navigation(operations)
            
            # Force update
            bpy.context.view_layer.update()
            
            # Get end transform
            end_transform = {
                "location": tuple(cam_obj.location),
                "rotation_euler": tuple(cam_obj.rotation_euler),
            }
            
            # Insert keyframe at end_frame
            cam_obj.keyframe_insert(data_path="location", frame=end_frame)
            cam_obj.keyframe_insert(data_path="rotation_euler", frame=end_frame)
            
            # Set smooth interpolation (Bezier) for natural camera movement
            if cam_obj.animation_data and cam_obj.animation_data.action:
                for fcurve in cam_obj.animation_data.action.fcurves:
                    for keyframe in fcurve.keyframe_points:
                        keyframe.interpolation = 'BEZIER'
                        keyframe.handle_left_type = 'AUTO_CLAMPED'
                        keyframe.handle_right_type = 'AUTO_CLAMPED'
            
            # Step 6: Apply auto_focus at end frame if DoF is enabled
            dof_applied_end = False
            focus_distance_end = None
            
            if camera_parameters.get('use_dof') and camera_parameters.get('dof_focus_object'):
                focus_result = auto_focus(
                    camera_name=camera_name,
                    camera_parameters=camera_parameters,
                    scene_name=scene_name
                )
                
                if focus_result.get('success'):
                    dof_applied_end = True
                    focus_distance_end = focus_result.get('focus_distance')
            
            return {
                "success": True,
                "camera_name": cam_obj.name,
                "camera_object": cam_obj,
                "camera_parameters": camera_parameters,
                "start_transform": start_transform,
                "end_transform": end_transform,
                "is_animated": True,
                "movement": movement,
                "movement_operation": op_name,
                "movement_steps": steps,
                "dof_applied": dof_applied_end or start_result.get('dof_applied', False),
                "focus_distance": focus_distance_end,
                "final_image_path": start_result.get('final_image_path'),
            }
            
        finally:
            # Restore viewport settings
            space.lock_camera = original_lock_camera
            space.camera = original_space_camera
            space.region_3d.view_perspective = original_view_perspective
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def resume_camera(
    camera_instruction: dict,
    scene_name: Optional[str] = None
) -> dict:
    """
    Resume/recreate a camera from saved placement info without LLM.
    
    This function recreates a camera that was previously placed by create_and_place_camera_for_shot,
    using the saved camera_parameters, transforms, and animation info.
    
    Parameters:
    - camera_instruction: Dictionary containing camera info with placement data:
        - camera_name: Name for the camera
        - camera_parameters: Dict with focal_length, use_dof, dof_focus_object, dof_fstop
        - start_transform: Dict with location and rotation_euler at start frame
        - end_transform: Dict with location and rotation_euler at end frame (if animated)
        - start_frame: Starting frame
        - end_frame: Ending frame
        - is_animated: Whether camera has animation
        - movement: Movement type (static, pan, orbit, zoom in, zoom out)
        - dof_applied: Whether to apply DoF
        - focus_distance: Focus distance if DoF applied
    - scene_name: Optional scene name to operate on
    
    Returns:
    - Dictionary with:
        - 'success': Boolean indicating success
        - 'camera_name': Name of the created camera
        - 'camera_object': The camera Blender object
        - 'error': Error message if failed (optional)
    """
    try:
        print(f"[resume_camera] Starting...")
        camera_name = camera_instruction.get('camera_name', 'Camera')
        camera_parameters = camera_instruction.get('camera_parameters', {})
        start_transform = camera_instruction.get('start_transform', {})
        end_transform = camera_instruction.get('end_transform')
        start_frame = camera_instruction.get('start_frame', 1)
        end_frame = camera_instruction.get('end_frame', 73)
        is_animated = camera_instruction.get('is_animated', False)
        dof_applied = camera_instruction.get('dof_applied', False)
        focus_distance = camera_instruction.get('focus_distance')
        
        print(f"[resume_camera] Parsed values:")
        print(f"  camera_name: {camera_name}")
        print(f"  camera_parameters: {camera_parameters}")
        print(f"  start_transform: {start_transform}")
        print(f"  start_frame: {start_frame}, end_frame: {end_frame}")
        print(f"  is_animated: {is_animated}, dof_applied: {dof_applied}")
        
        # Get scene
        if scene_name:
            print(f"[resume_camera] Looking for scene: {scene_name}")
            scene = bpy.data.scenes.get(scene_name)
            if not scene:
                print(f"[resume_camera] ERROR: Scene '{scene_name}' not found!")
                print(f"[resume_camera] Available scenes: {[s.name for s in bpy.data.scenes]}")
                return {"success": False, "error": f"Scene '{scene_name}' not found."}
            # Switch to the scene
            print(f"[resume_camera] Switching to scene: {scene.name}")
            bpy.context.window.scene = scene
        else:
            scene = bpy.context.scene
            print(f"[resume_camera] Using current scene: {scene.name}")
        
        # Extract transform values - support both 'rotation' and 'rotation_euler' keys
        start_location = start_transform.get('location', (0, 0, 0))
        start_rotation = start_transform.get('rotation_euler') or start_transform.get('rotation', (0, 0, 0))
        
        print(f"[resume_camera] Creating camera with:")
        print(f"  location: {start_location}")
        print(f"  rotation: {start_rotation}")
        print(f"  focal_length: {camera_parameters.get('focal_length', 50.0)}")
        
        # Resolve dof_focus_object from asset_id string to Blender object
        dof_focus_obj = None
        dof_focus_id = camera_parameters.get('dof_focus_object')
        if dof_focus_id and isinstance(dof_focus_id, str):
            dof_focus_obj = _find_object_by_asset_id(dof_focus_id, scene)
            if dof_focus_obj and dof_focus_obj.type != 'MESH':
                mesh_children = _get_mesh_children(dof_focus_obj)
                dof_focus_obj = mesh_children[0] if mesh_children else None
        elif dof_focus_id and hasattr(dof_focus_id, 'type'):
            dof_focus_obj = dof_focus_id  # Already a Blender object
        
        # Create the camera using create_camera (returns camera object directly)
        cam_obj = create_camera(
            name=camera_name,
            location=start_location,
            rotation=start_rotation,
            focal_length=camera_parameters.get('focal_length', 50.0),
            use_dof=camera_parameters.get('use_dof', False),
            dof_focus_object=dof_focus_obj,
            dof_fstop=camera_parameters.get('dof_fstop', 2.8),
        )
        
        print(f"[resume_camera] create_camera result: {cam_obj}")
        
        if not cam_obj:
            print(f"[resume_camera] create_camera failed!")
            return {"success": False, "error": "Failed to create camera object"}
        
        # If animated, set up keyframes
        if is_animated and end_transform:
            end_location = end_transform.get('location', start_location)
            end_rotation = end_transform.get('rotation_euler') or end_transform.get('rotation', start_rotation)
            
            # Go to start frame and set keyframe
            scene.frame_set(start_frame)
            cam_obj.location = start_location
            cam_obj.rotation_euler = start_rotation
            cam_obj.keyframe_insert(data_path="location", frame=start_frame)
            cam_obj.keyframe_insert(data_path="rotation_euler", frame=start_frame)
            
            # Go to end frame and set keyframe
            scene.frame_set(end_frame)
            cam_obj.location = end_location
            cam_obj.rotation_euler = end_rotation
            cam_obj.keyframe_insert(data_path="location", frame=end_frame)
            cam_obj.keyframe_insert(data_path="rotation_euler", frame=end_frame)
            
            # Set smooth interpolation (Bezier) for natural camera movement
            if cam_obj.animation_data and cam_obj.animation_data.action:
                for fcurve in cam_obj.animation_data.action.fcurves:
                    for keyframe in fcurve.keyframe_points:
                        keyframe.interpolation = 'BEZIER'
                        keyframe.handle_left_type = 'AUTO_CLAMPED'
                        keyframe.handle_right_type = 'AUTO_CLAMPED'
        
        # Apply DoF focus distance if specified
        if dof_applied and focus_distance is not None:
            cam_obj.data.dof.use_dof = True
            cam_obj.data.dof.focus_distance = focus_distance
        
        print(f"[resume_camera] SUCCESS: Camera '{cam_obj.name}' created")
        return {
            "success": True,
            "camera_name": cam_obj.name,
            "camera_object": cam_obj,
        }
    
    except Exception as e:
        import traceback
        print(f"[resume_camera] EXCEPTION: {str(e)}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }
