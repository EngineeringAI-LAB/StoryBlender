import bpy
import os
import json
import struct
import tempfile
import shutil
from typing import Optional
from typing import Literal
import math

def _fix_gltf_json(gltf_data: dict) -> bool:
    """
    Fix common issues in glTF JSON data that cause import failures.
    
    Specifically handles: scene.nodes is null instead of an empty array.
    Does NOT merge scenes - that causes cross-contamination of meshes.
    
    Returns True if any fixes were applied, False otherwise.
    """
    modified = False
    
    if "scenes" in gltf_data and gltf_data["scenes"]:
        for scene in gltf_data["scenes"]:
            # Only fix null nodes, don't merge scenes
            if "nodes" in scene and scene["nodes"] is None:
                scene["nodes"] = []
                modified = True
    
    return modified


def _preprocess_gltf_file(filepath: str) -> str:
    """
    Preprocess a glTF/GLB file to fix common issues before import.
    
    Returns the path to the file to import (original if no fix needed,
    or a temp file if fixes were applied).
    Does NOT modify the original file.
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == ".gltf":
        # JSON-based glTF - read and check if fix is needed
        with open(filepath, "r", encoding="utf-8") as f:
            gltf_data = json.load(f)
        
        if _fix_gltf_json(gltf_data):
            # Create a temp file with the fix
            temp_dir = tempfile.mkdtemp()
            temp_filepath = os.path.join(temp_dir, os.path.basename(filepath))
            
            # Copy any associated files (.bin, textures) to temp dir
            src_dir = os.path.dirname(filepath)
            for item in os.listdir(src_dir):
                src_item = os.path.join(src_dir, item)
                if os.path.isfile(src_item) and item != os.path.basename(filepath):
                    shutil.copy2(src_item, temp_dir)
            
            # Write fixed gltf
            with open(temp_filepath, "w", encoding="utf-8") as f:
                json.dump(gltf_data, f)
            
            return temp_filepath
        
        return filepath
    
    elif ext == ".glb":
        # Binary glTF - need to extract JSON chunk and check
        with open(filepath, "rb") as f:
            # Read GLB header (12 bytes)
            magic = f.read(4)
            if magic != b"glTF":
                return filepath  # Not a valid GLB, let Blender handle the error
            
            version = struct.unpack("<I", f.read(4))[0]
            total_length = struct.unpack("<I", f.read(4))[0]
            
            # Read JSON chunk header
            json_chunk_length = struct.unpack("<I", f.read(4))[0]
            json_chunk_type = f.read(4)
            
            if json_chunk_type != b"JSON":
                return filepath  # Unexpected format
            
            # Read JSON data
            json_bytes = f.read(json_chunk_length)
            json_str = json_bytes.decode("utf-8").rstrip("\x00")  # Remove padding
            gltf_data = json.loads(json_str)
            
            # Check if fix is needed
            if not _fix_gltf_json(gltf_data):
                return filepath  # No fix needed
            
            # Read remaining chunks (binary data)
            remaining_data = f.read()
        
        # Create temp file with the fix
        temp_fd, temp_filepath = tempfile.mkstemp(suffix=".glb")
        os.close(temp_fd)
        
        # Repack the GLB
        fixed_json_str = json.dumps(gltf_data, separators=(",", ":"))
        # Pad JSON to 4-byte alignment
        while len(fixed_json_str) % 4 != 0:
            fixed_json_str += " "
        fixed_json_bytes = fixed_json_str.encode("utf-8")
        
        new_json_chunk_length = len(fixed_json_bytes)
        new_total_length = 12 + 8 + new_json_chunk_length + len(remaining_data)
        
        with open(temp_filepath, "wb") as f:
            # Write header
            f.write(b"glTF")
            f.write(struct.pack("<I", version))
            f.write(struct.pack("<I", new_total_length))
            # Write JSON chunk
            f.write(struct.pack("<I", new_json_chunk_length))
            f.write(b"JSON")
            f.write(fixed_json_bytes)
            # Write remaining chunks
            f.write(remaining_data)
        
        return temp_filepath
    
    return filepath


def import_asset(filepath: str) -> dict:
    """
    Import a GLB model into the current Blender scene.

    Args:
        filepath: The path to the GLB file.

    Returns:
        A dict containing success message and the name of the imported model.
    """
    original_filepath = filepath
    preprocessed_cleanup = None
    try:
        # Preprocess the file to fix common glTF issues
        filepath = _preprocess_gltf_file(filepath)
        if filepath != original_filepath:
            preprocessed_cleanup = filepath
        
        # Store existing objects and scenes before import
        existing_objects = set(bpy.data.objects.keys())
        existing_scenes = set(bpy.data.scenes.keys())
        existing_images = set(bpy.data.images.keys())
        target_scene = bpy.context.scene

        # Import the GLB file
        bpy.ops.import_scene.gltf(
            filepath=filepath,
            bone_heuristic='TEMPERANCE'
        )

        # Pack any images newly referenced from the preprocessed temp location
        # so they persist after the temp file/dir is cleaned up.
        if preprocessed_cleanup is not None:
            try:
                temp_dir_abs = os.path.abspath(os.path.dirname(preprocessed_cleanup))
                for img_name in set(bpy.data.images.keys()) - existing_images:
                    img = bpy.data.images.get(img_name)
                    if img is None or img.packed_file is not None:
                        continue
                    try:
                        img_abs = os.path.abspath(bpy.path.abspath(img.filepath)) if img.filepath else ""
                    except Exception:
                        img_abs = ""
                    if img_abs and img_abs.startswith(temp_dir_abs):
                        try:
                            img.pack()
                        except Exception:
                            pass
            except Exception:
                pass

        # Find newly imported objects BEFORE cleaning up scenes
        new_objects = set(bpy.data.objects.keys()) - existing_objects
        
        # Ensure new objects are linked to the target scene's collection
        for obj_name in new_objects:
            obj = bpy.data.objects.get(obj_name)
            if obj:
                # Link to target scene if not already linked
                if obj.name not in target_scene.collection.all_objects:
                    target_scene.collection.objects.link(obj)

        # Clean up any newly created scenes (glTF importer may create them)
        new_scenes = set(bpy.data.scenes.keys()) - existing_scenes
        for scene_name in new_scenes:
            scene = bpy.data.scenes.get(scene_name)
            if scene:
                # Unlink objects from this scene before removing it
                # (objects are already linked to target_scene)
                for obj in list(scene.collection.all_objects):
                    try:
                        scene.collection.objects.unlink(obj)
                    except RuntimeError:
                        pass  # Object might be in a nested collection
                bpy.data.scenes.remove(scene, do_unlink=True)
        
        # Ensure we're still in the target scene
        if bpy.context.scene != target_scene and target_scene.name in bpy.data.scenes:
            bpy.context.window.scene = target_scene
        
        # Unlink new objects from any other scenes they might have been added to
        for obj_name in new_objects:
            obj = bpy.data.objects.get(obj_name)
            if obj:
                for scene in bpy.data.scenes:
                    if scene != target_scene:
                        if obj.name in scene.collection.all_objects:
                            try:
                                scene.collection.objects.unlink(obj)
                            except RuntimeError:
                                pass  # Object might be in a nested collection

        # Get the name of the imported model (typically the root object)
        model_name = list(new_objects)[0] if new_objects else os.path.splitext(os.path.basename(filepath))[0]

        return {
            "success": True,
            "model_name": model_name,
            "new_objects": list(new_objects)
        }
    except Exception as e:
        return {
            "success": False,
            "model_name": None,
            "error": str(e)
        }
    finally:
        # Clean up the temp file/dir created by _preprocess_gltf_file.
        if preprocessed_cleanup is not None:
            try:
                parent = os.path.dirname(preprocessed_cleanup)
                # mkstemp (.glb) places the file directly under the system tempdir;
                # mkdtemp (.gltf) creates a dedicated subdir. Handle both safely.
                if os.path.abspath(parent) == os.path.abspath(tempfile.gettempdir()):
                    os.remove(preprocessed_cleanup)
                else:
                    shutil.rmtree(parent, ignore_errors=True)
            except Exception:
                pass


def get_asset_transform(model_name: str) -> dict:
    """
    Get all transform properties of a model in the CURRENT Blender scene.

    This function searches for the object in the current scene's objects collection,
    not the global bpy.data.objects. This ensures that when the same asset exists
    in multiple scenes (e.g., "snow_white" in Scene_1, "snow_white.001" in Scene_2),
    we get the correct transform for the current scene.

    Args:
        model_name: Name of an existing model in the scene. Can be the base name
                    (e.g., "snow_white") and will match objects with Blender's
                    auto-generated suffixes (e.g., "snow_white.001").

    Returns:
        A dict with 'success' (bool) and transform data:
            - location: {"x": float, "y": float, "z": float}
            - rotation: {"x": float, "y": float, "z": float} (in degrees)
            - scale: {"x": float, "y": float, "z": float}
            - dimensions: {"x": float, "y": float, "z": float}
            - actual_name: The actual object name found in Blender
    """
    try:
        current_scene = bpy.context.scene
        scene_objects = current_scene.objects
        
        # First, try exact match in current scene
        obj = None
        if model_name in scene_objects:
            obj = scene_objects[model_name]
        else:
            # If not found, search for objects whose name starts with model_name
            # This handles Blender's auto-naming like "snow_white.001", "snow_white.002"
            matching_objects = [
                o for o in scene_objects 
                if o.name == model_name or o.name.startswith(f"{model_name}.")
            ]
            
            if matching_objects:
                # Use the first matching object in this scene
                obj = matching_objects[0]
        
        if obj is None:
            return {
                "success": False,
                "error": f"Model '{model_name}' not found in scene '{current_scene.name}'. "
                         f"Available objects: {[o.name for o in scene_objects]}",
            }

        return {
            "success": True,
            "actual_name": obj.name,
            "location": {"x": obj.location.x, "y": obj.location.y, "z": obj.location.z},
            "rotation": {
                "x": math.degrees(obj.rotation_euler.x),
                "y": math.degrees(obj.rotation_euler.y),
                "z": math.degrees(obj.rotation_euler.z),
            },
            "scale": {"x": obj.scale.x, "y": obj.scale.y, "z": obj.scale.z},
            "dimensions": {"x": obj.dimensions.x, "y": obj.dimensions.y, "z": obj.dimensions.z},
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def transform_asset(
    model_name: str,
    location: Optional[dict] = None,
    rotation: Optional[dict] = None,
    scale: Optional[dict] = None,
    dimensions: Optional[dict] = None,
) -> dict:
    """
    Transform a model in Blender by setting its location, rotation, scale, and dimensions.

    Args:
        model_name: Name of an existing model in the scene to transform.
        location: Dict with keys 'x', 'y', 'z' for coordinates. Can be None or have None values.
        rotation: Dict with keys 'x', 'y', 'z' for rotation angles in XYZ Euler (degrees).
                  Can be None or have None values.
        scale: Dict with keys 'x', 'y', 'z' for scale factors. Can be None or have None values.
        dimensions: Dict with keys 'x', 'y', 'z' for dimensions. Can be None or have None values.

    Returns:
        A dict with 'success' (bool) and 'message' (str) keys.
    """
    try:
        # Check if model exists in the scene
        if model_name not in bpy.data.objects:
            return {
                "success": False,
                "message": f"Model '{model_name}' not found in the scene.",
            }

        obj = bpy.data.objects[model_name]

        # Set location
        if location is not None:
            if location.get("x") is not None:
                obj.location.x = location["x"]
            if location.get("y") is not None:
                obj.location.y = location["y"]
            if location.get("z") is not None:
                obj.location.z = location["z"]

        # Set rotation (XYZ Euler) - convert from degrees to radians
        if rotation is not None:
            # Ensure rotation mode is set to XYZ Euler
            obj.rotation_mode = "XYZ"
            if rotation.get("x") is not None:
                obj.rotation_euler.x = math.radians(rotation["x"])
            if rotation.get("y") is not None:
                obj.rotation_euler.y = math.radians(rotation["y"])
            if rotation.get("z") is not None:
                obj.rotation_euler.z = math.radians(rotation["z"])

        # Set scale
        if scale is not None:
            if scale.get("x") is not None:
                obj.scale.x = scale["x"]
            if scale.get("y") is not None:
                obj.scale.y = scale["y"]
            if scale.get("z") is not None:
                obj.scale.z = scale["z"]

        # Set dimensions
        if dimensions is not None:
            if dimensions.get("x") is not None:
                obj.dimensions.x = dimensions["x"]
            if dimensions.get("y") is not None:
                obj.dimensions.y = dimensions["y"]
            if dimensions.get("z") is not None:
                obj.dimensions.z = dimensions["z"]

        return {
            "success": True,
            "message": f"Successfully transformed model '{model_name}'.",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error transforming model '{model_name}': {str(e)}",
        }


def create_empty_scene(
    scene_name: str,
    switch_to_created_scene: bool = True,
    type: Literal["NEW", "EMPTY", "LINK_COPY", "FULL_COPY"] = "EMPTY",
) -> bpy.types.Scene:
    """Create a new scene in Blender with the given scene_name.

    Args:
        scene_name: The name of the scene to create.
        switch_to_created_scene: Whether to switch to the new scene after creation.
            Defaults to True.
        type: The type of scene to create. Options are:
            - 'NEW': Add a new, empty scene with default objects.
            - 'EMPTY': Add a new, completely empty scene.
            - 'LINK_COPY': Link objects from the current scene to the new one.
            - 'FULL_COPY': Make a full copy of the current scene.
            Defaults to 'EMPTY'.

    Returns:
        The newly created scene.
    """
    # Store the current scene before creating a new one
    original_scene = bpy.context.window.scene

    # Create the new scene
    bpy.ops.scene.new(type=type)

    # The newly created scene becomes the active scene
    new_scene = bpy.context.window.scene
    new_scene.name = scene_name

    # Switch back to the original scene if requested
    if not switch_to_created_scene:
        bpy.context.window.scene = original_scene

    return new_scene


def switch_or_create_scene(scene_name: str) -> bpy.types.Scene:
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
    return create_empty_scene(scene_name=scene_name, switch_to_created_scene=True, type="EMPTY")


def delete_scene_and_assets(scene_name: str) -> dict:
    """Delete a scene and all its assets.

    Args:
        scene_name: The name of the scene to delete.

    Returns:
        A dict with 'success' (bool) and 'message' or 'error'.
    """
    try:
        # Check if scene exists
        if scene_name not in bpy.data.scenes:
            return {
                "success": False,
                "error": f"Scene '{scene_name}' not found.",
            }

        scene = bpy.data.scenes[scene_name]

        # Collect all objects that belong to this scene
        objects_to_delete = [obj for obj in scene.collection.all_objects]

        # Delete all objects in the scene
        for obj in objects_to_delete:
            bpy.data.objects.remove(obj, do_unlink=True)

        # Delete the scene
        bpy.data.scenes.remove(scene, do_unlink=True)

        return {
            "success": True,
            "message": f"Successfully deleted scene '{scene_name}' and {len(objects_to_delete)} objects.",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def delete_all_scenes_and_assets() -> dict:
    """Delete all scenes and assets, leaving only an empty scene named 'Scene'.

    Returns:
        A dict with 'success' (bool) and 'message' or 'error'.
    """
    try:
        # First, create or switch to a scene named "Scene"
        switch_or_create_scene("Scene")

        # Get all scene names except "Scene"
        scenes_to_delete = [scene.name for scene in bpy.data.scenes if scene.name != "Scene"]

        deleted_scenes = []
        errors = []

        # Delete all other scenes
        for scene_name in scenes_to_delete:
            result = delete_scene_and_assets(scene_name)
            if result.get("success"):
                deleted_scenes.append(scene_name)
            else:
                errors.append(f"{scene_name}: {result.get('error', 'Unknown error')}")

        # Also remove any orphan objects not linked to any scene
        orphan_objects = [obj for obj in bpy.data.objects if obj.users == 0]
        for obj in orphan_objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        # Clean up the "Scene" to make sure it's empty
        scene = bpy.data.scenes["Scene"]
        objects_in_scene = [obj for obj in scene.collection.all_objects]
        for obj in objects_in_scene:
            bpy.data.objects.remove(obj, do_unlink=True)

        if errors:
            return {
                "success": False,
                "message": f"Deleted {len(deleted_scenes)} scenes, but some errors occurred.",
                "deleted_scenes": deleted_scenes,
                "errors": errors,
            }

        return {
            "success": True,
            "message": f"Successfully deleted {len(deleted_scenes)} scenes. Blender now has only an empty 'Scene'.",
            "deleted_scenes": deleted_scenes,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def import_asset_to_scene(
    filepath: str,
    scene_name: str,
    transform_parameters: Optional[dict] = None,
) -> dict:
    """
    Import a GLB asset to a specific scene and apply transforms.

    Args:
        filepath: File path to the GLB file.
        scene_name: The name of the scene to import the model to (created if not exists).
        transform_parameters: Dict with transform parameters. Possible keys:
            - location: {"x": float, "y": float, "z": float}
            - rotation: {"x": float, "y": float, "z": float}
            - scale: {"x": float, "y": float, "z": float}
            - dimensions: {"x": float, "y": float, "z": float}
            Each key can be None or omitted.

    Returns:
        A dict with 'success' (bool) and additional info.
    """
    try:
        # Switch to or create the target scene
        switch_or_create_scene(scene_name)

        # Import the asset
        import_result = import_asset(filepath)
        if not import_result.get("success"):
            return {
                "success": False,
                "error": import_result.get("error", "Failed to import asset"),
            }

        model_name = import_result.get("model_name")

        # Apply transforms if provided
        if transform_parameters:
            transform_result = transform_asset(
                model_name=model_name,
                location=transform_parameters.get("location"),
                rotation=transform_parameters.get("rotation"),
                scale=transform_parameters.get("scale"),
                dimensions=transform_parameters.get("dimensions"),
            )
            if not transform_result.get("success"):
                return {
                    "success": False,
                    "model_name": model_name,
                    "error": transform_result.get("message", "Failed to transform asset"),
                }

        return {
            "success": True,
            "model_name": model_name,
            "scene_name": scene_name,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def import_all_assets_to_scene(
    asset_sheet: list,
    scene_detail: dict,
) -> dict:
    """
    Import all assets from asset_sheet to a single scene based on scene_detail.

    Args:
        asset_sheet: A list of asset dictionaries, each with 'id' and 'main_file_path'.
        scene_detail: A dictionary containing scene_id and scene_setup with layout info.

    Returns:
        A dict with 'success' (bool) and 'failed_objects' list if any failures.
    """
    try:
        scene_id = scene_detail.get("scene_id")
        scene_name = f"Scene_{scene_id}"

        # Build a lookup dict from asset_sheet by id
        asset_lookup = {asset["asset_id"]: asset for asset in asset_sheet}

        # Get assets from scene_setup.layout_description.assets
        scene_setup = scene_detail.get("scene_setup", {})
        layout_description = scene_setup.get("layout_description", {})
        assets = layout_description.get("assets", [])

        failed_objects = []

        for asset_info in assets:
            asset_id = asset_info.get("asset_id")

            # Find the asset in asset_sheet
            asset_data = asset_lookup.get(asset_id)
            if not asset_data:
                failed_objects.append({
                    "scene_id": scene_id,
                    "object_id": asset_id,
                    "error": f"Asset '{asset_id}' not found in asset_sheet",
                })
                continue

            filepath = asset_data.get("main_file_path")
            if not filepath:
                failed_objects.append({
                    "scene_id": scene_id,
                    "object_id": asset_id,
                    "error": f"No 'main_file_path' for asset '{asset_id}'",
                })
                continue

            # Build transform_parameters from asset_info
            transform_parameters = {
                "location": asset_info.get("location"),
                "rotation": asset_info.get("rotation"),
                "scale": asset_info.get("scale"),
                "dimensions": asset_info.get("dimensions"),
            }

            # Import the asset to the scene
            result = import_asset_to_scene(
                filepath=filepath,
                scene_name=scene_name,
                transform_parameters=transform_parameters,
            )

            if not result.get("success"):
                failed_objects.append({
                    "scene_id": scene_id,
                    "object_id": asset_id,
                    "error": result.get("error", "Unknown error"),
                })

        if failed_objects:
            return {
                "success": False,
                "failed_objects": failed_objects,
            }

        return {"success": True}

    except Exception as e:
        print(f"Error importing assets to scene '{scene_name}': {str(e)}", flush=True)
        return {
            "success": False,
            "error": str(e),
        }


def import_all_assets_to_all_scenes(
    asset_sheet: list,
    scene_details: list,
) -> dict:
    """
    Import all assets to all scenes based on scene_details.

    Args:
        asset_sheet: A list of asset dictionaries, each with 'id' and 'main_file_path'.
        scene_details: A list of scene_detail dictionaries.

    Returns:
        A dict with 'success' (bool) and 'failed_objects' list if any failures.
    """
    try:
        all_failed_objects = []

        for scene_detail in scene_details:
            result = import_all_assets_to_scene(
                asset_sheet=asset_sheet,
                scene_detail=scene_detail,
            )

            if not result.get("success"):
                failed_objects = result.get("failed_objects", [])
                if failed_objects:
                    all_failed_objects.extend(failed_objects)
                else:
                    # General error for this scene
                    scene_id = scene_detail.get("scene_id")
                    all_failed_objects.append({
                        "scene_id": scene_id,
                        "object_id": None,
                        "error": result.get("error", "Unknown error"),
                    })

        if all_failed_objects:
            return {
                "success": False,
                "failed_objects": all_failed_objects,
            }

        return {"success": True}

    except Exception as e:
        print(f"Error importing assets to all scenes: {str(e)}", flush=True)
        return {
            "success": False,
            "error": str(e),
        }


def import_all_assets_to_all_scenes_json_input(json_filepath: str) -> dict:
    """
    Import all assets to all scenes from a JSON file.

    Args:
        json_filepath: Path to the JSON file containing asset_sheet and scene_details.

    Returns:
        A dict with 'success' (bool) and 'failed_objects' list if any failures.
    """
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        asset_sheet = data.get("asset_sheet", [])
        scene_details = data.get("scene_details", [])

        if not asset_sheet:
            return {
                "success": False,
                "error": "No 'asset_sheet' found in JSON file",
            }

        if not scene_details:
            return {
                "success": False,
                "error": "No 'scene_details' found in JSON file",
            }

        return import_all_assets_to_all_scenes(
            asset_sheet=asset_sheet,
            scene_details=scene_details,
        )

    except FileNotFoundError:
        return {
            "success": False,
            "error": f"JSON file not found: {json_filepath}",
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid JSON format: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def import_supplementary_assets_to_scene(
    asset_sheet: list,
    scene_detail: dict,
) -> dict:
    """
    Import supplementary assets from asset_sheet to a single scene based on scene_detail.

    Args:
        asset_sheet: A list of supplementary asset dictionaries, each with 'asset_id' and 'main_file_path'.
        scene_detail: A dictionary containing scene_id and scene_setup with layout_description.

    Returns:
        A dict with 'success' (bool) and 'failed_objects' list if any failures.
    """
    try:
        scene_id = scene_detail.get("scene_id")
        scene_name = f"Scene_{scene_id}"

        # Build a lookup dict from asset_sheet by asset_id
        asset_lookup = {asset["asset_id"]: asset for asset in asset_sheet}

        # Get assets from scene_setup.layout_description.assets
        scene_setup = scene_detail.get("scene_setup", {})
        layout_description = scene_setup.get("layout_description", {})
        assets = layout_description.get("assets", [])

        failed_objects = []

        for asset_info in assets:
            asset_id = asset_info.get("asset_id")

            # Find the asset in asset_sheet
            asset_data = asset_lookup.get(asset_id)
            if not asset_data:
                failed_objects.append({
                    "scene_id": scene_id,
                    "object_id": asset_id,
                    "error": f"Supplementary asset '{asset_id}' not found in asset_sheet",
                })
                continue

            filepath = asset_data.get("main_file_path")
            if not filepath:
                failed_objects.append({
                    "scene_id": scene_id,
                    "object_id": asset_id,
                    "error": f"No 'main_file_path' for supplementary asset '{asset_id}'",
                })
                continue

            # Build transform_parameters from asset_info
            transform_parameters = {
                "location": asset_info.get("location"),
                "rotation": asset_info.get("rotation"),
                "scale": asset_info.get("scale"),
                "dimensions": asset_info.get("dimensions"),
            }

            # Import the asset to the scene
            result = import_asset_to_scene(
                filepath=filepath,
                scene_name=scene_name,
                transform_parameters=transform_parameters,
            )

            if not result.get("success"):
                failed_objects.append({
                    "scene_id": scene_id,
                    "object_id": asset_id,
                    "error": result.get("error", "Unknown error"),
                })

        if failed_objects:
            return {
                "success": False,
                "failed_objects": failed_objects,
            }

        return {"success": True}

    except Exception as e:
        print(f"Error importing supplementary assets to scene '{scene_name}': {str(e)}", flush=True)
        return {
            "success": False,
            "error": str(e),
        }


def import_supplementary_assets_to_all_scenes(
    asset_sheet: list,
    scene_details: list,
) -> dict:
    """
    Import supplementary assets to all scenes based on scene_details.

    Args:
        asset_sheet: A list of supplementary asset dictionaries, each with 'asset_id' and 'main_file_path'.
        scene_details: A list of scene_detail dictionaries.

    Returns:
        A dict with 'success' (bool) and 'failed_objects' list if any failures.
    """
    try:
        all_failed_objects = []

        for scene_detail in scene_details:
            result = import_supplementary_assets_to_scene(
                asset_sheet=asset_sheet,
                scene_detail=scene_detail,
            )

            if not result.get("success"):
                failed_objects = result.get("failed_objects", [])
                if failed_objects:
                    all_failed_objects.extend(failed_objects)
                else:
                    # General error for this scene
                    scene_id = scene_detail.get("scene_id")
                    all_failed_objects.append({
                        "scene_id": scene_id,
                        "object_id": None,
                        "error": result.get("error", "Unknown error"),
                    })

        if all_failed_objects:
            return {
                "success": False,
                "failed_objects": all_failed_objects,
            }

        return {"success": True}

    except Exception as e:
        print(f"Error importing supplementary assets to all scenes: {str(e)}", flush=True)
        return {
            "success": False,
            "error": str(e),
        }


def import_supplementary_assets_to_all_scenes_json_input(json_filepath: str) -> dict:
    """
    Import supplementary assets to all scenes from a JSON file.

    Args:
        json_filepath: Path to the JSON file containing asset_sheet and scene_details
                      for supplementary assets.

    Returns:
        A dict with 'success' (bool) and 'failed_objects' list if any failures.
    """
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        asset_sheet = data.get("asset_sheet", [])
        scene_details = data.get("scene_details", [])

        if not asset_sheet:
            return {
                "success": False,
                "error": "No 'asset_sheet' found in supplementary layout JSON file",
            }

        if not scene_details:
            return {
                "success": False,
                "error": "No 'scene_details' found in supplementary layout JSON file",
            }

        return import_supplementary_assets_to_all_scenes(
            asset_sheet=asset_sheet,
            scene_details=scene_details,
        )

    except FileNotFoundError:
        return {
            "success": False,
            "error": f"Supplementary layout JSON file not found: {json_filepath}",
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid JSON format in supplementary layout file: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
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


def _make_object_local_for_scene(obj: bpy.types.Object, scene: bpy.types.Scene) -> bpy.types.Object:
    """
    Make a linked object local to the current scene only.
    
    Since shot scenes are created using "Linked Copy", objects are shared across
    scenes. To modify an object only in a specific shot scene, we need to make
    a local copy that is unique to this scene.
    
    Args:
        obj: The object to make local.
        scene: The scene where the local copy should exist.
    
    Returns:
        The local copy of the object (or the original if it was already local).
    """
    # Switch to the target scene
    bpy.context.window.scene = scene
    
    # Deselect all objects first
    bpy.ops.object.select_all(action='DESELECT')
    
    # Select the object
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Check if this object is used in multiple scenes (linked copy scenario)
    # In linked copies, the same object instance is shared across scenes,
    # so modifying obj.location affects all scenes.
    # We need to check if the object appears in more than one scene.
    scenes_using_obj = [s for s in bpy.data.scenes if obj.name in s.objects]
    
    if len(scenes_using_obj) > 1:
        # Make the object single-user for this scene only
        bpy.ops.object.make_single_user(
            object=True,
            obdata=True,
            material=False,
            animation=False,
            obdata_animation=False
        )
        # The active object is now the local copy
        return bpy.context.view_layer.objects.active
    
    return obj


def _create_linked_copy_scene(source_scene_name: str, new_scene_name: str) -> dict:
    """
    Create a linked copy of a scene.
    
    Args:
        source_scene_name: Name of the source scene to copy from.
        new_scene_name: Name for the new scene.
    
    Returns:
        Dict with success status and the new scene or error.
    """
    try:
        # Check if source scene exists
        if source_scene_name not in bpy.data.scenes:
            return {
                "success": False,
                "error": f"Source scene '{source_scene_name}' not found.",
            }
        
        # Check if new scene already exists
        if new_scene_name in bpy.data.scenes:
            return {
                "success": True,
                "scene": bpy.data.scenes[new_scene_name],
                "already_exists": True,
            }
        
        # Switch to source scene
        source_scene = bpy.data.scenes[source_scene_name]
        bpy.context.window.scene = source_scene
        
        # Create linked copy
        bpy.ops.scene.new(type='LINK_COPY')
        
        # Rename the new scene
        new_scene = bpy.context.window.scene
        new_scene.name = new_scene_name
        
        return {
            "success": True,
            "scene": new_scene,
            "already_exists": False,
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def apply_asset_modifications(shot_details: list, asset_sheet: list = None) -> dict:
    """
    Apply asset modifications to shot scenes based on shot_details.
    
    This function:
    1. FIRST creates all shot scenes as linked copies of their parent Scene_{scene_id}
    2. THEN applies asset_modifications (add/remove/transform) to each shot scene
    
    This order is important because 'add' operations need all shot scenes to exist
    before they can add assets to the current shot and all subsequent shots.
    
    Since shot scenes are created using "Linked Copy", this function makes
    objects local before modifying them to avoid affecting other scenes.
    
    Args:
        shot_details: A list of shot detail dictionaries, each containing:
            - scene_id: The scene ID
            - shot_id: The shot ID
            - asset_modifications: List of modifications (can be None), each with:
                - asset_id: The asset to modify
                - modification_type: 'add', 'remove', or 'transform'
                - target_location: {"x": float, "y": float, "z": float} (optional)
                - target_rotation: {"x": float, "y": float, "z": float} in degrees (optional)
        asset_sheet: A list of asset dictionaries with 'asset_id' and 'main_file_path'.
                    Required for 'add' modification type.
    
    Returns:
        A dict with 'success' (bool), 'modified_count', 'scenes_created', and 'errors' list if any failures.
    """
    try:
        modified_count = 0
        scenes_created = 0
        errors = []
        
        # Build asset lookup for 'add' operations
        asset_lookup = {}
        if asset_sheet:
            asset_lookup = {asset["asset_id"]: asset for asset in asset_sheet}
        
        # =========================================================================
        # PHASE 1: Create all shot scenes as linked copies of their parent scenes
        # =========================================================================
        for shot in shot_details:
            scene_id = shot.get("scene_id")
            shot_id = shot.get("shot_id")
            
            source_scene_name = f"Scene_{scene_id}"
            shot_scene_name = f"Scene_{scene_id}_Shot_{shot_id}"
            
            # Create the shot scene as a linked copy of the parent scene
            create_result = _create_linked_copy_scene(source_scene_name, shot_scene_name)
            
            if not create_result.get("success"):
                errors.append({
                    "scene_id": scene_id,
                    "shot_id": shot_id,
                    "error": f"Failed to create shot scene: {create_result.get('error', 'Unknown error')}",
                })
                continue
            
            if not create_result.get("already_exists"):
                scenes_created += 1
        
        # =========================================================================
        # PHASE 2: Apply asset modifications to each shot scene
        # =========================================================================
        for shot in shot_details:
            asset_modifications = shot.get("asset_modifications")
            
            # Skip shots without modifications
            if not asset_modifications:
                continue
            
            scene_id = shot.get("scene_id")
            shot_id = shot.get("shot_id")
            shot_scene_name = f"Scene_{scene_id}_Shot_{shot_id}"
            
            # Check if the shot scene exists
            if shot_scene_name not in bpy.data.scenes:
                errors.append({
                    "scene_id": scene_id,
                    "shot_id": shot_id,
                    "error": f"Shot scene '{shot_scene_name}' not found",
                })
                continue
            
            shot_scene = bpy.data.scenes[shot_scene_name]
            
            # Switch to the shot scene
            bpy.context.window.scene = shot_scene
            
            # Track removed assets to skip subsequent operations on them
            removed_assets_in_shot = set()
            
            for modification in asset_modifications:
                asset_id = modification.get("asset_id")
                modification_type = modification.get("modification_type")
                
                # Skip if this asset was already removed in this shot
                if asset_id in removed_assets_in_shot:
                    continue
                
                # Find the object in the shot scene
                obj = _find_object_by_asset_id(asset_id, shot_scene)
                
                if obj is None and modification_type != "add":
                    errors.append({
                        "scene_id": scene_id,
                        "shot_id": shot_id,
                        "asset_id": asset_id,
                        "error": f"Object '{asset_id}' not found in scene '{shot_scene_name}'",
                    })
                    continue
                
                if modification_type == "remove":
                    # Remove the object from this scene only
                    print(f"[INFO] {shot_scene_name}: remove '{asset_id}'")
                    # First, unlink from scene's collection
                    try:
                        shot_scene.collection.objects.unlink(obj)
                        modified_count += 1
                        removed_assets_in_shot.add(asset_id)
                        print(f"[INFO] {shot_scene_name}: remove '{asset_id}' - SUCCESS")
                    except RuntimeError:
                        # Object might be in a nested collection
                        # Try to find and remove from all collections in the scene
                        removed = False
                        for collection in shot_scene.collection.children_recursive:
                            if obj.name in collection.objects:
                                collection.objects.unlink(obj)
                                removed = True
                                break
                        if removed:
                            modified_count += 1
                            removed_assets_in_shot.add(asset_id)
                            print(f"[INFO] {shot_scene_name}: remove '{asset_id}' - SUCCESS")
                        else:
                            errors.append({
                                "scene_id": scene_id,
                                "shot_id": shot_id,
                                "asset_id": asset_id,
                                "error": f"Failed to remove object '{obj.name}' from scene",
                            })
                            print(f"[INFO] {shot_scene_name}: remove '{asset_id}' - FAILED")
                elif modification_type == "add":
                    # Import a new asset to this shot scene AND all subsequent shots in the same scene
                    target_location = modification.get("target_location")
                    target_rotation = modification.get("target_rotation")
                    
                    # Build info string for logging
                    loc_str = f"location={target_location}" if target_location else ""
                    rot_str = f"rotation={target_rotation}" if target_rotation else ""
                    transform_info = ", ".join(filter(None, [loc_str, rot_str]))
                    info_suffix = f" - {transform_info}" if transform_info else ""
                    
                    print(f"[INFO] {shot_scene_name}: add '{asset_id}'{info_suffix}")
                    
                    asset_data = asset_lookup.get(asset_id)
                    if not asset_data:
                        errors.append({
                            "scene_id": scene_id,
                            "shot_id": shot_id,
                            "asset_id": asset_id,
                            "error": f"Asset '{asset_id}' not found in asset_sheet for 'add' operation",
                        })
                        print(f"[INFO] {shot_scene_name}: add '{asset_id}' - FAILED (not in asset_sheet)")
                        continue
                    
                    filepath = asset_data.get("main_file_path")
                    if not filepath:
                        errors.append({
                            "scene_id": scene_id,
                            "shot_id": shot_id,
                            "asset_id": asset_id,
                            "error": f"No 'main_file_path' for asset '{asset_id}'",
                        })
                        print(f"[INFO] {shot_scene_name}: add '{asset_id}' - FAILED (no file path)")
                        continue
                    
                    # Build transform_parameters from modification
                    transform_parameters = {
                        "location": target_location,
                        "rotation": target_rotation,
                    }
                    
                    # Find all shots in the same scene with shot_id >= current shot_id
                    target_shot_scenes = []
                    for other_shot in shot_details:
                        if other_shot.get("scene_id") == scene_id and other_shot.get("shot_id") >= shot_id:
                            target_scene_name = f"Scene_{scene_id}_Shot_{other_shot.get('shot_id')}"
                            if target_scene_name in bpy.data.scenes:
                                target_shot_scenes.append(target_scene_name)
                    
                    # Import the asset to all target shot scenes
                    success_count = 0
                    for target_scene_name in target_shot_scenes:
                        result = import_asset_to_scene(
                            filepath=filepath,
                            scene_name=target_scene_name,
                            transform_parameters=transform_parameters,
                        )
                        
                        if not result.get("success"):
                            errors.append({
                                "scene_id": scene_id,
                                "shot_id": shot_id,
                                "asset_id": asset_id,
                                "error": f"Failed to import to {target_scene_name}: {result.get('error', 'Unknown error')}",
                            })
                        else:
                            modified_count += 1
                            success_count += 1
                    
                    if success_count == len(target_shot_scenes):
                        print(f"[INFO] {shot_scene_name}: add '{asset_id}' - SUCCESS (added to {success_count} shots)")
                    else:
                        print(f"[INFO] {shot_scene_name}: add '{asset_id}' - PARTIAL ({success_count}/{len(target_shot_scenes)} shots)")
                else:
                    # For 'transform' type: Remove from current+subsequent shots, then re-add with new transform
                    # This avoids the complexity of linked objects from parent Scene_X
                    
                    target_location = modification.get("target_location")
                    target_rotation = modification.get("target_rotation")
                    
                    # Build info string for logging
                    loc_str = f"location={target_location}" if target_location else ""
                    rot_str = f"rotation={target_rotation}" if target_rotation else ""
                    transform_info = ", ".join(filter(None, [loc_str, rot_str]))
                    
                    print(f"[INFO] {shot_scene_name}: transform '{asset_id}' - {transform_info}")
                    
                    # Find the object in current shot to get its current data (mesh, materials, etc.)
                    current_obj = _find_object_by_asset_id(asset_id, shot_scene)
                    if current_obj is None:
                        errors.append({
                            "scene_id": scene_id,
                            "shot_id": shot_id,
                            "asset_id": asset_id,
                            "error": f"Object '{asset_id}' not found in scene '{shot_scene_name}' for transform",
                        })
                        print(f"[INFO] {shot_scene_name}: transform '{asset_id}' - FAILED (object not found)")
                        continue
                    
                    # Find all shots in same scene with shot_id >= current shot_id
                    target_shot_scene_names = []
                    for other_shot in shot_details:
                        if other_shot.get("scene_id") == scene_id and other_shot.get("shot_id") >= shot_id:
                            target_scene_name = f"Scene_{scene_id}_Shot_{other_shot.get('shot_id')}"
                            if target_scene_name in bpy.data.scenes:
                                target_shot_scene_names.append(target_scene_name)
                    
                    # STEP 1: Remove the object from current shot and all subsequent shots
                    for target_scene_name in target_shot_scene_names:
                        target_scene = bpy.data.scenes[target_scene_name]
                        target_obj = _find_object_by_asset_id(asset_id, target_scene)
                        if target_obj is not None:
                            # Unlink from scene's collection
                            try:
                                target_scene.collection.objects.unlink(target_obj)
                            except RuntimeError:
                                # Object might be in a nested collection
                                for collection in target_scene.collection.children_recursive:
                                    if target_obj.name in collection.objects:
                                        collection.objects.unlink(target_obj)
                                        break
                    
                    # STEP 2: Create a NEW object copy with the new transform
                    new_obj = current_obj.copy()
                    if current_obj.data:
                        new_obj.data = current_obj.data.copy()
                    
                    # Apply the new transform
                    if target_location:
                        if target_location.get("x") is not None:
                            new_obj.location.x = target_location["x"]
                        if target_location.get("y") is not None:
                            new_obj.location.y = target_location["y"]
                        if target_location.get("z") is not None:
                            new_obj.location.z = target_location["z"]
                    
                    if target_rotation:
                        new_obj.rotation_mode = "XYZ"
                        if target_rotation.get("x") is not None:
                            new_obj.rotation_euler.x = math.radians(target_rotation["x"])
                        if target_rotation.get("y") is not None:
                            new_obj.rotation_euler.y = math.radians(target_rotation["y"])
                        if target_rotation.get("z") is not None:
                            new_obj.rotation_euler.z = math.radians(target_rotation["z"])
                    
                    # Preserve the asset_id name
                    new_obj.name = asset_id
                    
                    # STEP 3: Link the new object to current shot and all subsequent shots
                    for target_scene_name in target_shot_scene_names:
                        target_scene = bpy.data.scenes[target_scene_name]
                        target_scene.collection.objects.link(new_obj)
                    
                    print(f"[INFO] {shot_scene_name}: transform '{asset_id}' - SUCCESS (applied to {len(target_shot_scene_names)} shots)")
                    modified_count += 1
        
        if errors:
            return {
                "success": False,
                "modified_count": modified_count,
                "scenes_created": scenes_created,
                "errors": errors,
            }
        
        return {
            "success": True,
            "modified_count": modified_count,
            "scenes_created": scenes_created,
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def apply_asset_modifications_json_input(json_filepath: str) -> dict:
    """
    Apply asset modifications to shot scenes from a JSON file.
    
    This function reads the shot_details from a JSON file and applies any
    asset_modifications defined for each shot.
    
    Args:
        json_filepath: Path to the JSON file containing shot_details with
                      asset_modifications.
    
    Returns:
        A dict with 'success' (bool), 'modified_count', and 'errors' list if any failures.
    """
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        shot_details = data.get("shot_details", [])
        asset_sheet = data.get("asset_sheet", [])
        
        if not shot_details:
            return {
                "success": False,
                "error": "No 'shot_details' found in JSON file",
            }
        
        return apply_asset_modifications(shot_details=shot_details, asset_sheet=asset_sheet)
    
    except FileNotFoundError:
        return {
            "success": False,
            "error": f"JSON file not found: {json_filepath}",
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Invalid JSON format: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }