import bpy
import mathutils
import os
import gc
import time
from typing import Optional


def _switch_or_create_scene(scene_name: str) -> bpy.types.Scene:
    """Switch to a scene with the given scene_name, or create it if it doesn't exist.

    Args:
        scene_name: The name of the scene to switch to or create.

    Returns:
        The scene that was switched to (existing or newly created).
    """
    # Check if scene with the given scene_name already exists
    if scene_name in bpy.data.scenes:
        scene = bpy.data.scenes[scene_name]
        bpy.context.window.scene = scene
        return scene

    # Scene doesn't exist, create a new empty one
    bpy.ops.scene.new(type="EMPTY")
    new_scene = bpy.context.window.scene
    new_scene.name = scene_name
    return new_scene


def _delete_scene_and_its_objects(scene_name: str) -> None:
    """Delete a scene and objects that belong ONLY to it (not shared with other scenes).

    Args:
        scene_name: The name of the scene to delete.
    """
    if scene_name not in bpy.data.scenes:
        return

    scene = bpy.data.scenes[scene_name]

    # Collect all objects that belong to this scene
    objects_in_temp_scene = set(scene.collection.all_objects)
    
    # Collect objects that exist in other scenes
    objects_in_other_scenes = set()
    for other_scene in bpy.data.scenes:
        if other_scene.name != scene_name:
            objects_in_other_scenes.update(other_scene.collection.all_objects)
    
    # Only delete objects that are exclusive to the temporary scene
    objects_to_delete = objects_in_temp_scene - objects_in_other_scenes

    # Delete objects exclusive to this scene
    for obj in objects_to_delete:
        bpy.data.objects.remove(obj, do_unlink=True)

    # Delete the scene
    bpy.data.scenes.remove(scene, do_unlink=True)


def resize_asset(
    model_info: dict,
    export_dir: str = None
):
    """
    Resize a 3D model by importing, resizing based on dimensions, setting origin to ground center,
    renaming, and exporting.
    
    This function assumes the model has already been orientation-corrected (e.g., by format_asset).
    It only handles resizing based on provided dimensions.
    
    Parameters:
    - model_info: Dictionary containing model information with keys:
        - 'asset_id': Identifier for the model
        - 'main_file_path': Absolute path to the model file (GLB/GLTF/FBX/OBJ)
        - 'width': Target width in meters (provide only one of width, depth, height)
        - 'depth': Target depth in meters (provide only one of width, depth, height)
        - 'height': Target height in meters (provide only one of width, depth, height)
    - export_dir: Directory to export the resized model
    
    Returns:
    - Dictionary with:
        - 'export_path': Path to the exported model
        - 'dimensions': Dictionary with 'X', 'Y', 'Z' dimensions in meters
        - 'thumbnail_url': Path to the thumbnail image (if captured)
    
    Raises:
    - ValueError: If model_info is missing required fields or has invalid data
    - FileNotFoundError: If the model file is not found
    """
    # Disable GC and add sync points for stability with Gradio
    gc.disable()
    try:
        # Brief pause before starting
        time.sleep(1)
        
        result = _resize_asset_core(
            model_info=model_info,
            export_dir=export_dir
        )
        
        # Brief pause after completion
        time.sleep(1)
        
        return result
    finally:
        gc.enable()


def _resize_asset_core(
    model_info: dict,
    export_dir: str = None
):
    """Core implementation of resize_asset. Called with extended GIL hold time."""
    
    # === Step 1: Switch to temporary scene ===
    original_scene = bpy.context.window.scene
    original_scene_name = original_scene.name
    
    _switch_or_create_scene("Temporary_resize_scene")
    
    # Extract model_id for error messages
    model_id = model_info.get("asset_id", "unknown")
    
    # Get the model file path
    main_file_path = model_info.get("main_file_path")
    if not main_file_path:
        raise ValueError(f"{model_id}: No main_file_path specified")
    
    path_to_model = main_file_path
    
    # Check if file exists
    if not os.path.exists(path_to_model):
        raise FileNotFoundError(f"{model_id}: Model file not found at {path_to_model}")
    
    # Determine which dimension is provided (width=X, depth=Y, height=Z)
    width = model_info.get("width")
    depth = model_info.get("depth")
    height = model_info.get("height")
    
    # Map to X, Y, Z
    X = width
    Y = depth
    Z = height
    
    # Validate exactly one dimension is provided
    dimensions_provided = sum([X is not None, Y is not None, Z is not None])
    if dimensions_provided != 1:
        raise ValueError(f"{model_id}: Exactly one dimension (width, depth, or height) must be provided")
    
    # Extract base filename without extension for naming
    original_filename = os.path.basename(path_to_model)
    base_name = os.path.splitext(original_filename)[0]
    
    # === Step 2: Empty the current scene only ===
    current_scene = bpy.context.scene
    for obj in list(current_scene.collection.all_objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # Delete only child collections of the current scene
    for coll in list(current_scene.collection.children):
        try:
            bpy.data.collections.remove(coll)
        except Exception:
            pass
    
    # === Step 3: Import the model ===
    bpy.ops.object.select_all(action='DESELECT')
    
    ext = os.path.splitext(path_to_model)[1].lower()
    if ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=path_to_model)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=path_to_model)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=path_to_model)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # === Step 4: Merge geometry ===
    geom_types = {"MESH", "CURVE", "SURFACE", "META", "FONT", "GPENCIL"}
    mesh_objs = [o for o in bpy.context.scene.objects if o.type in geom_types]
    
    if not mesh_objs:
        raise RuntimeError("No geometry objects found after import")
    
    # Unhide all geometry objects
    for o in mesh_objs:
        o.hide_set(False)
        o.hide_viewport = False
        if hasattr(o, 'hide_select'):
            o.hide_select = False
    
    # Convert non-meshes to mesh
    for o in mesh_objs:
        if o.type != 'MESH':
            bpy.context.view_layer.objects.active = o
            o.select_set(True)
            try:
                bpy.ops.object.convert(target='MESH', keep_original=False)
            except Exception:
                pass
            o.select_set(False)
    
    # Get all mesh objects and join if multiple
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    
    if len(mesh_objs) > 1:
        for o in bpy.context.selected_objects:
            o.select_set(False)
        for o in mesh_objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objs[0]
        bpy.ops.object.join()
        merged_obj = bpy.context.view_layer.objects.active
    else:
        merged_obj = mesh_objs[0]
    
    # Unparent the merged object from any hierarchy
    if merged_obj.parent is not None:
        world_matrix = merged_obj.matrix_world.copy()
        merged_obj.parent = None
        merged_obj.matrix_world = world_matrix
    
    # Update scene after merge
    bpy.context.view_layer.update()
    
    # === Step 5: Resize model with aspect ratio preserved ===
    bpy.context.view_layer.update()
    
    # Compute current world-space bounding box
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    for corner in merged_obj.bound_box:
        co = merged_obj.matrix_world @ mathutils.Vector(corner)
        min_x = min(min_x, co.x); max_x = max(max_x, co.x)
        min_y = min(min_y, co.y); max_y = max(max_y, co.y)
        min_z = min(min_z, co.z); max_z = max(max_z, co.z)
    
    current_x = max_x - min_x
    current_y = max_y - min_y
    current_z = max_z - min_z
    
    # Determine scale factor based on provided dimension
    if X is not None:
        scale_factor = X / current_x if current_x > 0 else 1.0
    elif Y is not None:
        scale_factor = Y / current_y if current_y > 0 else 1.0
    else:  # Z is not None
        scale_factor = Z / current_z if current_z > 0 else 1.0
    
    # Apply uniform scale to the merged object
    merged_obj.scale = tuple(s * scale_factor for s in merged_obj.scale)
    
    # Update scene after scaling
    bpy.context.view_layer.update()
    
    # === Step 6: Use merged object as final object ===
    final_obj = merged_obj
    
    # === Step 7: Set origin point to bottom center ===
    for o in bpy.context.selected_objects:
        o.select_set(False)
    final_obj.select_set(True)
    bpy.context.view_layer.objects.active = final_obj
    
    # Apply transforms to bake into geometry before origin adjustment
    try:
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    except Exception:
        pass
    
    bpy.context.view_layer.update()
    
    # Compute bounding box directly from mesh vertices
    mesh = final_obj.data
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    for vert in mesh.vertices:
        co = final_obj.matrix_world @ vert.co
        min_x = min(min_x, co.x); max_x = max(max_x, co.x)
        min_y = min(min_y, co.y); max_y = max(max_y, co.y)
        min_z = min(min_z, co.z); max_z = max(max_z, co.z)
    
    # Bottom-center of bounding box in world space
    bottom_center_world = mathutils.Vector(((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, min_z))
    
    # Convert to object local space
    offset_local = final_obj.matrix_world.inverted() @ bottom_center_world
    
    # Manually offset all mesh vertices
    for vert in mesh.vertices:
        vert.co -= offset_local
    
    mesh.update()
    
    # Adjust object location
    final_obj.location = bottom_center_world
    bpy.context.view_layer.update()
    
    # Set location to (0,0,0)
    final_obj.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    
    # === Step 8: Calculate final dimensions ===
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    for corner in final_obj.bound_box:
        co = final_obj.matrix_world @ mathutils.Vector(corner)
        min_x = min(min_x, co.x); max_x = max(max_x, co.x)
        min_y = min(min_y, co.y); max_y = max(max_y, co.y)
        min_z = min(min_z, co.z); max_z = max(max_z, co.z)
    
    final_dimensions = {
        'X': max_x - min_x,
        'Y': max_y - min_y,
        'Z': max_z - min_z
    }
    
    # === Step 9: Rename object, mesh, and material ===
    obj_name = base_name
    mesh_name = f"{base_name}_mesh"
    mat_name = f"{base_name}_material"
    
    final_obj.name = obj_name
    if final_obj.data:
        final_obj.data.name = mesh_name
    
    # Handle materials
    mats = [m for m in final_obj.data.materials if m is not None]
    unique_mats = []
    seen = set()
    for m in mats:
        if m and m.name_full not in seen:
            unique_mats.append(m)
            seen.add(m.name_full)
    
    if len(unique_mats) == 0:
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            mat = bpy.data.materials.new(name=mat_name)
        if len(final_obj.data.materials) == 0:
            final_obj.data.materials.append(mat)
        else:
            final_obj.data.materials[0] = mat

    
    # === Step 10: Reset Transform ===
    # Use final_obj directly. Do NOT look up by obj_name via bpy.data.objects,
    # because bpy.data.objects is global across all scenes: if an unrelated
    # object with the same name exists (e.g., a root Empty from the imported
    # GLB hierarchy like dead_bush.glb), Blender silently renames our mesh to
    # "<name>.001" and the lookup returns the wrong object (which may have
    # data=None, breaking the vertex iteration below).
    target_obj = final_obj
    
    # Unparent target_obj if it has a parent
    if target_obj.parent is not None:
        world_matrix = target_obj.matrix_world.copy()
        target_obj.parent = None
        target_obj.matrix_world = world_matrix
    
    # Delete all other objects in the CURRENT scene only
    current_scene = bpy.context.scene
    for obj in list(current_scene.collection.all_objects):
        if obj != target_obj:
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Select only the target object
    for o in bpy.context.selected_objects:
        o.select_set(False)
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj
    
    # Apply rotation and scale
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    
    # Reset location to origin
    target_obj.location = (0, 0, 0)
    
    # === Step 11: Re-verify and fix origin at bottom center ===
    bpy.context.view_layer.update()
    
    mesh = target_obj.data
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    for vert in mesh.vertices:
        co = target_obj.matrix_world @ vert.co
        min_x = min(min_x, co.x); max_x = max(max_x, co.x)
        min_y = min(min_y, co.y); max_y = max(max_y, co.y)
        min_z = min(min_z, co.z); max_z = max(max_z, co.z)
    
    # Check if origin is at bottom center
    expected_origin = mathutils.Vector(((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, min_z))
    current_origin = target_obj.matrix_world.translation
    
    if (expected_origin - current_origin).length > 0.0001:
        offset_local = target_obj.matrix_world.inverted() @ expected_origin
        for vert in mesh.vertices:
            vert.co -= offset_local
        mesh.update()
        target_obj.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
    
    # === Step 12: Export ===
    if export_dir is None:
        export_dir = os.path.dirname(path_to_model)
    
    os.makedirs(export_dir, exist_ok=True)
    export_filename = f"{base_name}.glb"
    export_path = os.path.join(export_dir, export_filename)
    
    # Select only the final object
    for o in bpy.context.selected_objects:
        o.select_set(False)
    final_obj.select_set(True)
    bpy.context.view_layer.objects.active = final_obj
    
    # Ensure the object is only in the scene's root collection
    current_scene = bpy.context.scene
    
    for coll in list(final_obj.users_collection):
        coll.objects.unlink(final_obj)
    current_scene.collection.objects.link(final_obj)
    
    # Delete ALL child collections from the scene
    def remove_all_child_collections(parent_collection):
        for child_coll in list(parent_collection.children):
            remove_all_child_collections(child_coll)
            try:
                bpy.data.collections.remove(child_coll)
            except Exception:
                pass
    
    remove_all_child_collections(current_scene.collection)
    
    # Export as GLB
    bpy.ops.export_scene.gltf(
        filepath=export_path,
        export_format='GLB',
        use_selection=True,
        use_active_scene=True,
        export_apply=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_materials='EXPORT',
        export_cameras=False,
        export_lights=False,
        export_extras=False,
        check_existing=False,
    )
    
    # === Step 14: Cleanup and switch back to original scene ===
    if original_scene_name in bpy.data.scenes:
        bpy.context.window.scene = bpy.data.scenes[original_scene_name]
    
    _delete_scene_and_its_objects("Temporary_resize_scene")
    
    # Purge orphan data
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    
    result = {
        'export_path': export_path,
        'dimensions': final_dimensions
    }
    
    return result
