import bpy
import mathutils
import csv
import os
import json
import struct
import tempfile
import shutil
import math
import re
from typing import Optional, List, Dict, Any

# Movement categories — Walking/Running animations have natural root motion
_MOVEMENT_CATEGORIES = {"Walking", "Running", "CrouchWalking"}
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ANIMATION_CSV = os.path.join(_SCRIPT_DIR, "Meshy_animation.csv")


def _is_movement_animation(action_name: str) -> bool:
    """Return True if *action_name* belongs to a walking/running category."""
    try:
        with open(_ANIMATION_CSV, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("Name") == action_name:
                    return row.get("Category", "") in _MOVEMENT_CATEGORIES
    except Exception:
        pass
    lower = action_name.lower()
    return any(kw in lower for kw in ("walk", "run", "jog", "sprint"))


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
        with open(filepath, "r", encoding="utf-8") as f:
            gltf_data = json.load(f)
        
        if _fix_gltf_json(gltf_data):
            temp_dir = tempfile.mkdtemp()
            temp_filepath = os.path.join(temp_dir, os.path.basename(filepath))
            
            src_dir = os.path.dirname(filepath)
            for item in os.listdir(src_dir):
                src_item = os.path.join(src_dir, item)
                if os.path.isfile(src_item) and item != os.path.basename(filepath):
                    shutil.copy2(src_item, temp_dir)
            
            with open(temp_filepath, "w", encoding="utf-8") as f:
                json.dump(gltf_data, f)
            
            return temp_filepath
        
        return filepath
    
    elif ext == ".glb":
        with open(filepath, "rb") as f:
            magic = f.read(4)
            if magic != b"glTF":
                return filepath
            
            version = struct.unpack("<I", f.read(4))[0]
            total_length = struct.unpack("<I", f.read(4))[0]
            
            json_chunk_length = struct.unpack("<I", f.read(4))[0]
            json_chunk_type = f.read(4)
            
            if json_chunk_type != b"JSON":
                return filepath
            
            json_bytes = f.read(json_chunk_length)
            json_str = json_bytes.decode("utf-8").rstrip("\x00")
            gltf_data = json.loads(json_str)
            
            if not _fix_gltf_json(gltf_data):
                return filepath
            
            remaining_data = f.read()
        
        temp_fd, temp_filepath = tempfile.mkstemp(suffix=".glb")
        os.close(temp_fd)
        
        fixed_json_str = json.dumps(gltf_data, separators=(",", ":"))
        while len(fixed_json_str) % 4 != 0:
            fixed_json_str += " "
        fixed_json_bytes = fixed_json_str.encode("utf-8")
        
        new_json_chunk_length = len(fixed_json_bytes)
        new_total_length = 12 + 8 + new_json_chunk_length + len(remaining_data)
        
        with open(temp_filepath, "wb") as f:
            f.write(b"glTF")
            f.write(struct.pack("<I", version))
            f.write(struct.pack("<I", new_total_length))
            f.write(struct.pack("<I", new_json_chunk_length))
            f.write(b"JSON")
            f.write(fixed_json_bytes)
            f.write(remaining_data)
        
        return temp_filepath
    
    return filepath


def set_origin_to_bottom_center(obj: bpy.types.Object) -> dict:
    """
    Set the origin of an object to its bottom center based on the deformed/animated mesh.
    
    For animated models with NLA that may have non-standing poses (e.g., laying down),
    this ensures the origin is at the bottom center of the actual deformed mesh,
    so the object sits on the ground when z=0.
    
    Args:
        obj: The Blender object to adjust.
    
    Returns:
        Dict with success status.
    """
    try:
        bpy.context.view_layer.update()
        
        # Get the dependency graph to evaluate deformed meshes (with armature/NLA applied)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # Calculate bounding box in world space from DEFORMED vertices
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        def update_bounds_from_evaluated_mesh(mesh_obj):
            """Calculate bounds from the evaluated (deformed) mesh vertices."""
            nonlocal min_x, min_y, min_z, max_x, max_y, max_z
            
            # Get the evaluated object with all modifiers/armature deformations applied
            eval_obj = mesh_obj.evaluated_get(depsgraph)
            
            # Get the evaluated mesh data
            eval_mesh = eval_obj.to_mesh()
            if eval_mesh is None:
                return
            
            try:
                # Iterate through actual deformed vertex positions
                for vertex in eval_mesh.vertices:
                    # Transform vertex to world space
                    co = mesh_obj.matrix_world @ vertex.co
                    min_x = min(min_x, co.x); max_x = max(max_x, co.x)
                    min_y = min(min_y, co.y); max_y = max(max_y, co.y)
                    min_z = min(min_z, co.z); max_z = max(max_z, co.z)
            finally:
                # Clean up the temporary mesh
                eval_obj.to_mesh_clear()
        
        # For objects with mesh data, evaluate the deformed mesh
        if obj.type == 'MESH':
            update_bounds_from_evaluated_mesh(obj)
        elif obj.type == 'ARMATURE':
            # For armatures, iterate through all child meshes and evaluate each
            for child in obj.children_recursive:
                if child.type == 'MESH':
                    update_bounds_from_evaluated_mesh(child)
        else:
            # Fallback: use object's bound_box if available (non-mesh objects)
            if hasattr(obj, 'bound_box') and obj.bound_box:
                for corner in obj.bound_box:
                    co = obj.matrix_world @ mathutils.Vector(corner)
                    min_x = min(min_x, co.x); max_x = max(max_x, co.x)
                    min_y = min(min_y, co.y); max_y = max(max_y, co.y)
                    min_z = min(min_z, co.z); max_z = max(max_z, co.z)
        
        # Check if we found valid bounds
        if min_z == float('inf'):
            return {
                "success": False,
                "error": "Could not calculate bounding box",
            }
        
        # Calculate bottom center in world space
        bottom_center_x = (min_x + max_x) * 0.5
        bottom_center_y = (min_y + max_y) * 0.5
        bottom_center_z = min_z
        
        # Calculate offset needed to move bottom center to origin
        # We need to adjust the object location so that the bottom center is at (0, 0, 0)
        # But we want to preserve the intended location, so we offset relative to current position
        current_loc = obj.location.copy()
        
        # The offset is how much the bottom center is away from the object's current origin
        offset_x = bottom_center_x - current_loc.x
        offset_y = bottom_center_y - current_loc.y
        offset_z = bottom_center_z - current_loc.z
        
        # Adjust location so that when we set location to the target, bottom center is on ground
        # Store the offset as a property or apply it
        # For animated models, we simply adjust the z location to account for the bottom offset
        obj.location.z = current_loc.z - offset_z
        
        bpy.context.view_layer.update()
        
        return {
            "success": True,
            "offset_applied": {"x": 0, "y": 0, "z": -offset_z},
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def find_object_by_asset_id(scene: bpy.types.Scene, asset_id: str) -> Optional[bpy.types.Object]:
    """
    Find an object in the scene by asset_id, handling Blender's auto-rename pattern (e.g., asset_id.001).
    
    Args:
        scene: The Blender scene to search in.
        asset_id: The base asset_id to search for.
    
    Returns:
        The matching object, or None if not found.
    """
    scene_objects = scene.objects
    
    # First, try exact match
    if asset_id in scene_objects:
        return scene_objects[asset_id]
    
    # Search for objects with Blender's auto-rename pattern (e.g., asset_id.001)
    pattern = re.compile(rf"^{re.escape(asset_id)}(\.\d{{3}})?$")
    for obj in scene_objects:
        if pattern.match(obj.name):
            return obj
    
    return None


def find_character_objects_in_shot(
    scene: bpy.types.Scene,
    asset_id: str,
    scene_id: Optional[int] = None,
    shot_id: Optional[int] = None,
) -> List[bpy.types.Object]:
    """
    Find all top-level objects in *scene* that represent the character ``asset_id``.

    This matches:
    - exact name ``asset_id``
    - Blender auto-rename ``asset_id.NNN``
    - previously-animated names ``asset_id_s{scene_id}sh{shot_id}_*`` produced by
      a prior run of this step (so re-runs replace cleanly instead of stacking
      duplicates).

    Returns a possibly empty list of objects.
    """
    patterns = [re.compile(rf"^{re.escape(asset_id)}(\.\d{{3}})?$")]
    if scene_id is not None and shot_id is not None:
        patterns.append(
            re.compile(rf"^{re.escape(asset_id)}_s{scene_id}sh{shot_id}_.*$")
        )

    matches: List[bpy.types.Object] = []
    seen = set()
    for obj in scene.objects:
        if obj.name in seen:
            continue
        for pat in patterns:
            if pat.match(obj.name):
                matches.append(obj)
                seen.add(obj.name)
                break
    return matches


def get_asset_transform_from_scene(scene: bpy.types.Scene, asset_id: str) -> Optional[dict]:
    """
    Get the transform of an asset in the given scene.
    
    Args:
        scene: The Blender scene to search in.
        asset_id: The asset_id to get transform for.
    
    Returns:
        Dict with location, rotation, scale, dimensions, or None if not found.
    """
    obj = find_object_by_asset_id(scene, asset_id)
    if obj is None:
        return None
    
    return {
        "location": {"x": obj.location.x, "y": obj.location.y, "z": obj.location.z},
        "rotation": {
            "x": math.degrees(obj.rotation_euler.x),
            "y": math.degrees(obj.rotation_euler.y),
            "z": math.degrees(obj.rotation_euler.z),
        },
        "scale": {"x": obj.scale.x, "y": obj.scale.y, "z": obj.scale.z},
        "dimensions": {"x": obj.dimensions.x, "y": obj.dimensions.y, "z": obj.dimensions.z},
    }


def get_asset_transform_from_layout(scene_details: List[dict], scene_id: int, asset_id: str) -> Optional[dict]:
    """
    Get the transform of an asset from layout_description in scene_details.
    
    Args:
        scene_details: List of scene_detail dictionaries.
        scene_id: The scene_id to search in.
        asset_id: The asset_id to get transform for.
    
    Returns:
        Dict with location, rotation, scale, dimensions, or None if not found.
    """
    for scene_detail in scene_details:
        if scene_detail.get("scene_id") == scene_id:
            scene_setup = scene_detail.get("scene_setup", {})
            layout_description = scene_setup.get("layout_description", {})
            assets = layout_description.get("assets", [])
            
            for asset in assets:
                if asset.get("asset_id") == asset_id:
                    return {
                        "location": asset.get("location"),
                        "rotation": asset.get("rotation"),
                        "scale": asset.get("scale"),
                        "dimensions": asset.get("dimensions"),
                    }
    
    return None


def create_linked_copy_scene(source_scene_name: str, new_scene_name: str) -> dict:
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
                "message": f"Scene '{new_scene_name}' already exists.",
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
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def unlink_character_from_scene(
    scene: bpy.types.Scene,
    asset_id: str,
    scene_id: Optional[int] = None,
    shot_id: Optional[int] = None,
) -> dict:
    """
    Unlink character objects (static or previously-animated) from *scene* so a
    fresh animated GLB can take their place.

    Matches the static ``asset_id`` (and ``.NNN`` auto-rename variants) and, when
    ``scene_id``/``shot_id`` are provided, any prior animated version named
    ``{asset_id}_s{scene_id}sh{shot_id}_*`` left over from an earlier run. Each
    matched root and all of its descendants are unlinked from every collection
    in the scene.

    Args:
        scene: The Blender scene.
        asset_id: The asset_id of the character to unlink.
        scene_id: Scene id of the shot (used to match prior animated names).
        shot_id: Shot id of the shot (used to match prior animated names).

    Returns:
        Dict with success status and list of unlinked object names.
    """
    try:
        roots = find_character_objects_in_shot(scene, asset_id, scene_id, shot_id)
        if not roots:
            return {
                "success": False,
                "error": f"Object '{asset_id}' not found in scene '{scene.name}'.",
            }

        # Collect every collection in the scene (root + nested) once
        all_collections = [scene.collection] + list(scene.collection.children_recursive)

        # Build the full set of objects to unlink (roots + their descendants).
        # Children are included so armature/mesh hierarchies fully detach,
        # avoiding stale meshes that share the static character name.
        targets: List[bpy.types.Object] = []
        seen_names = set()
        for root in roots:
            for o in [root] + list(root.children_recursive):
                if o.name not in seen_names:
                    targets.append(o)
                    seen_names.add(o.name)

        unlinked: List[str] = []
        for obj in targets:
            removed_any = False
            for collection in all_collections:
                if obj.name in collection.objects:
                    try:
                        collection.objects.unlink(obj)
                        removed_any = True
                    except RuntimeError:
                        pass
            if removed_any:
                unlinked.append(obj.name)

        return {
            "success": True,
            "unlinked_objects": unlinked,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def import_animated_asset(
    filepath: str,
    target_scene: bpy.types.Scene,
    new_object_name: str,
    transform: Optional[dict] = None,
) -> dict:
    """
    Import an animated GLB model into the target scene.
    
    Args:
        filepath: Path to the animated GLB file.
        target_scene: The Blender scene to import into.
        new_object_name: The name to assign to the imported object.
        transform: Optional transform parameters (location, rotation, scale, dimensions).
    
    Returns:
        Dict with success status and imported object info.
    """
    original_filepath = filepath
    preprocessed_cleanup = None
    try:
        # Preprocess the file
        filepath = _preprocess_gltf_file(filepath)
        if filepath != original_filepath:
            preprocessed_cleanup = filepath
        
        # Store existing objects before import
        existing_objects = set(bpy.data.objects.keys())
        existing_scenes = set(bpy.data.scenes.keys())
        existing_images = set(bpy.data.images.keys())
        
        # Switch to target scene
        bpy.context.window.scene = target_scene
        
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
        
        # Find newly imported objects
        new_objects = set(bpy.data.objects.keys()) - existing_objects
        
        # Ensure new objects are linked to the target scene's collection
        for obj_name in new_objects:
            obj = bpy.data.objects.get(obj_name)
            if obj:
                if obj.name not in target_scene.collection.all_objects:
                    target_scene.collection.objects.link(obj)
        
        # Clean up any newly created scenes
        new_scenes = set(bpy.data.scenes.keys()) - existing_scenes
        for scene_name in new_scenes:
            scene = bpy.data.scenes.get(scene_name)
            if scene:
                for obj in list(scene.collection.all_objects):
                    try:
                        scene.collection.objects.unlink(obj)
                    except RuntimeError:
                        pass
                bpy.data.scenes.remove(scene, do_unlink=True)
        
        # Ensure we're still in the target scene
        if bpy.context.scene != target_scene and target_scene.name in bpy.data.scenes:
            bpy.context.window.scene = target_scene
        
        # Unlink new objects from any other scenes
        for obj_name in new_objects:
            obj = bpy.data.objects.get(obj_name)
            if obj:
                for scene in bpy.data.scenes:
                    if scene != target_scene:
                        if obj.name in scene.collection.all_objects:
                            try:
                                scene.collection.objects.unlink(obj)
                            except RuntimeError:
                                pass
        
        # Get the root object (typically the first one or the armature parent)
        root_obj = None
        for obj_name in new_objects:
            obj = bpy.data.objects.get(obj_name)
            if obj and obj.parent is None:
                root_obj = obj
                break
        
        if root_obj is None and new_objects:
            root_obj = bpy.data.objects.get(list(new_objects)[0])
        
        if root_obj is None:
            return {
                "success": False,
                "error": "No objects imported from file.",
            }
        
        # Rename the root object
        original_name = root_obj.name
        root_obj.name = new_object_name
        
        # Apply transform if provided
        if transform:
            location = transform.get("location")
            rotation = transform.get("rotation")
            scale = transform.get("scale")
            dimensions = transform.get("dimensions")
            
            if location:
                if location.get("x") is not None:
                    root_obj.location.x = location["x"]
                if location.get("y") is not None:
                    root_obj.location.y = location["y"]
                if location.get("z") is not None:
                    root_obj.location.z = location["z"]
            
            if rotation:
                root_obj.rotation_mode = "XYZ"
                if rotation.get("x") is not None:
                    root_obj.rotation_euler.x = math.radians(rotation["x"])
                if rotation.get("y") is not None:
                    root_obj.rotation_euler.y = math.radians(rotation["y"])
                if rotation.get("z") is not None:
                    root_obj.rotation_euler.z = math.radians(rotation["z"])
            
            if scale:
                if scale.get("x") is not None:
                    root_obj.scale.x = scale["x"]
                if scale.get("y") is not None:
                    root_obj.scale.y = scale["y"]
                if scale.get("z") is not None:
                    root_obj.scale.z = scale["z"]
            
            if dimensions:
                # For animated models that may not be standing (e.g., sleeping),
                # use uniform scaling based on longest dimension to maintain aspect ratio.
                # Find the longest dimension of the imported model
                current_dims = [root_obj.dimensions.x, root_obj.dimensions.y, root_obj.dimensions.z]
                current_longest = max(current_dims)
                
                # Find the largest value in the target dimensions
                target_dims = [
                    dimensions.get("x") if dimensions.get("x") is not None else 0,
                    dimensions.get("y") if dimensions.get("y") is not None else 0,
                    dimensions.get("z") if dimensions.get("z") is not None else 0,
                ]
                target_largest = max(target_dims)
                
                # Apply uniform scaling by setting dimensions directly
                if current_longest > 0 and target_largest > 0:
                    scale_factor = target_largest / current_longest
                    root_obj.dimensions.x = root_obj.dimensions.x * scale_factor
                    root_obj.dimensions.y = root_obj.dimensions.y * scale_factor
                    root_obj.dimensions.z = root_obj.dimensions.z * scale_factor
        
        # Set origin to bottom center so model sits on ground when z=0
        # This is important for animated models with non-standing poses (e.g., sleeping)
        origin_result = set_origin_to_bottom_center(root_obj)
        if not origin_result.get("success"):
            print(f"Warning: Could not set origin to bottom center: {origin_result.get('error')}", flush=True)
        
        return {
            "success": True,
            "object_name": root_obj.name,
            "new_objects": list(new_objects),
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
    finally:
        # Clean up the temp file/dir created by _preprocess_gltf_file.
        if preprocessed_cleanup is not None:
            try:
                parent = os.path.dirname(preprocessed_cleanup)
                if os.path.abspath(parent) == os.path.abspath(tempfile.gettempdir()):
                    os.remove(preprocessed_cleanup)
                else:
                    shutil.rmtree(parent, ignore_errors=True)
            except Exception:
                pass


def _get_animation_frame_range(obj: bpy.types.Object) -> tuple:
    """
    Get the start and end frame of the animation on *obj* or its children.

    Searches NLA tracks first (imported GLBs typically use NLA), then
    falls back to the scene frame range.

    Returns:
        (frame_start, frame_end) as ints.
    """
    candidates = [obj] + list(obj.children_recursive)
    min_f = float('inf')
    max_f = float('-inf')

    for o in candidates:
        ad = o.animation_data
        if ad is None:
            continue
        for track in (ad.nla_tracks or []):
            for strip in track.strips:
                min_f = min(min_f, strip.frame_start)
                max_f = max(max_f, strip.frame_end)
        if ad.action:
            for fcurve in ad.action.fcurves:
                for kp in fcurve.keyframe_points:
                    min_f = min(min_f, kp.co[0])
                    max_f = max(max_f, kp.co[0])

    if min_f < float('inf'):
        return int(min_f), int(max_f)

    return bpy.context.scene.frame_start, bpy.context.scene.frame_end


def _find_root_motion_info(obj: bpy.types.Object) -> Optional[dict]:
    """Find location F-Curves that carry root motion for *obj*.

    For Uthana/Mixamo models the root motion lives on the **root bone**
    (e.g. ``mixamorig:Hips``) as ``pose.bones["…"].location``.

    Returns a dict with keys:
        ``fcurves``  – ``{axis_index: FCurve}``
        ``bone_name`` – str or None (None = object-level motion)
        ``armature_obj`` – the Blender object that owns the animation_data
        ``action_name`` – name of the action for logging
    or ``None`` if nothing was found.
    """
    candidates = [obj] + list(obj.children_recursive)

    # Root bone names to prefer (in priority order)
    _ROOT_BONE_NAMES = {"Hips", "mixamorig:Hips", "hip", "Root", "root",
                        "Spine", "pelvis", "Pelvis"}

    def _search_action(action, owner):
        if action is None:
            return None
        # Collect ALL bone location FCurves, grouped by bone name
        bones: Dict[str, Dict[int, Any]] = {}  # bone_name → {axis: fc}
        for fc in action.fcurves:
            if fc.data_path.endswith(".location") and "pose.bones" in fc.data_path:
                try:
                    bname = fc.data_path.split('"')[1]
                except (IndexError, KeyError):
                    bname = fc.data_path
                bones.setdefault(bname, {})[fc.array_index] = fc

        if bones:
            # Prefer known root bone names
            chosen = None
            for preferred in _ROOT_BONE_NAMES:
                if preferred in bones:
                    chosen = preferred
                    break
            # Fallback: pick the bone with largest displacement across axes
            if chosen is None:
                best_disp = -1.0
                for bname, fcs in bones.items():
                    disp = 0.0
                    for fc in fcs.values():
                        if len(fc.keyframe_points) >= 2:
                            disp += abs(fc.keyframe_points[-1].co[1] - fc.keyframe_points[0].co[1])
                    if disp > best_disp:
                        best_disp = disp
                        chosen = bname
            if chosen:
                return {
                    "fcurves": bones[chosen],
                    "bone_name": chosen,
                    "armature_obj": owner,
                    "action_name": action.name,
                }

        # Priority 2: object-level location
        obj_loc = {}
        for fc in action.fcurves:
            if fc.data_path == "location":
                obj_loc[fc.array_index] = fc
        if obj_loc:
            return {
                "fcurves": obj_loc,
                "bone_name": None,
                "armature_obj": owner,
                "action_name": action.name,
            }
        return None

    for cand in candidates:
        ad = cand.animation_data
        if ad is None:
            continue
        for track in (ad.nla_tracks or []):
            for strip in track.strips:
                result = _search_action(strip.action, cand)
                if result:
                    return result
        result = _search_action(ad.action, cand)
        if result:
            return result
    return None


def _retarget_root_motion(
    obj: bpy.types.Object,
    desired_end_world: mathutils.Vector,
    frame_start: int,
    frame_end: int,
) -> bool:
    """Retarget character movement so the character ends at *desired_end_world* XY.

    Strategy (rig-agnostic, animation-preserving):

    1. Measure the root bone's natural world-space XY displacement during the
       animation (``natural_xy``).
    2. Compute the extra XY displacement needed to end at
       ``desired_end_world``:  ``extra_xy = desired_xy - natural_xy``.
    3. Apply ``extra_xy`` as a **linear object-level XY tween**, keeping Z at
       the object's current ground-level origin.

    The original bone F-Curves are **never modified**, so vertical motion
    (fall, jump, crouch, stand-up, walking bob) and bone-local axis alignment
    quirks cannot distort the animation.  The only visual artefact is minor
    foot-sliding proportional to ``|extra_xy|``, which is an accepted
    trade-off without IK-based motion retargeting.

    Returns ``True`` in all cases (the tween always succeeds; zero extra
    displacement is a valid outcome meaning no adjustment was necessary).
    """
    P = lambda msg: print(f"    [retarget] {msg}", flush=True)

    # Try to locate the root bone (used only to measure natural displacement;
    # we do NOT modify its F-Curves).
    info = _find_root_motion_info(obj)
    bone_name = info["bone_name"] if info else None
    armature_obj = info["armature_obj"] if info else obj
    if info:
        P(f"Root bone='{bone_name}' on '{armature_obj.name}' "
          f"(action='{info['action_name']}')")
    else:
        P("No root-motion F-Curves found — will measure object-level displacement")

    # --- Evaluate root world position at first and last frame ---
    scene = bpy.context.scene
    orig_frame = scene.frame_current

    def _eval_root_world(frame_nr):
        scene.frame_set(frame_nr)
        bpy.context.view_layer.update()
        if bone_name and armature_obj.type == 'ARMATURE':
            pb = armature_obj.pose.bones.get(bone_name)
            if pb:
                return (armature_obj.matrix_world @ pb.matrix).translation.copy()
        return obj.matrix_world.translation.copy()

    root_start = _eval_root_world(frame_start)
    root_end = _eval_root_world(frame_end)
    scene.frame_set(orig_frame)

    natural_dx = root_end.x - root_start.x
    natural_dy = root_end.y - root_start.y
    desired_dx = desired_end_world.x - root_start.x
    desired_dy = desired_end_world.y - root_start.y
    extra_dx = desired_dx - natural_dx
    extra_dy = desired_dy - natural_dy
    extra_len = math.hypot(extra_dx, extra_dy)

    P(f"Root world start   : ({root_start.x:.3f}, {root_start.y:.3f}, {root_start.z:.3f})")
    P(f"Root world end     : ({root_end.x:.3f}, {root_end.y:.3f}, {root_end.z:.3f})")
    P(f"Natural  XY disp   : ({natural_dx:.3f}, {natural_dy:.3f}) len={math.hypot(natural_dx, natural_dy):.3f}")
    P(f"Desired  XY disp   : ({desired_dx:.3f}, {desired_dy:.3f}) len={math.hypot(desired_dx, desired_dy):.3f}")
    P(f"Extra    XY disp   : ({extra_dx:.3f}, {extra_dy:.3f}) len={extra_len:.3f}")

    # Threshold: ≤5 cm of residual is visually indistinguishable from target.
    if extra_len < 0.05:
        P("Natural animation already reaches target within 0.05 — no tween added")
        return True

    # --- Apply linear object-level XY tween (preserve Z) ---
    obj_start = obj.location.copy()
    obj_end = mathutils.Vector((
        obj_start.x + extra_dx,
        obj_start.y + extra_dy,
        obj_start.z,
    ))
    obj.location = obj_start
    obj.keyframe_insert(data_path="location", frame=frame_start)
    obj.location = obj_end
    obj.keyframe_insert(data_path="location", frame=frame_end)

    # Force linear interpolation on the two tween keyframes we just inserted.
    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            if fc.data_path == "location":
                for kp in fc.keyframe_points:
                    if kp.co[0] in (frame_start, frame_end):
                        kp.interpolation = 'LINEAR'

    P(f"Applied object XY tween: "
      f"({obj_start.x:.2f},{obj_start.y:.2f},z={obj_start.z:.2f}) → "
      f"({obj_end.x:.2f},{obj_end.y:.2f},z={obj_end.z:.2f}) "
      f"over frames {frame_start}–{frame_end}")

    # --- Verify final world XY ---
    verify_end = _eval_root_world(frame_end)
    scene.frame_set(orig_frame)
    residual = math.hypot(desired_end_world.x - verify_end.x,
                          desired_end_world.y - verify_end.y)
    P(f"Verify: root now ends at ({verify_end.x:.3f}, {verify_end.y:.3f}), residual={residual:.3f}")

    return True


def import_animated_assets_to_shot(
    shot_detail: dict,
    scene_details: List[dict],
) -> dict:
    """
    Import animated assets to a single shot.
    
    This function assumes Scene_{scene_id}_Shot_{shot_id} already exists (created by apply_asset_modifications).
    It:
    1. Reads the transform of character from the existing shot scene
    2. Unlinks the static character from the shot scene
    3. Imports animated models with the same transform
    
    Supports both **Meshy** and **Uthana** pipelines:
    - Meshy actions have ``action_id``/``action_name`` and use Meshy-rigged GLBs.
    - Uthana actions have ``text_to_motion_prompt`` and use root-motion GLBs.
      If ``start_coords`` is set, it overrides the scene location.
      If ``start_rotation`` is set, it overrides the scene rotation.
      If ``end_coords`` is set, the animation's root-motion F-Curves are
      scaled so the character ends at the desired world position.
    
    Args:
        shot_detail: Dict containing scene_id, shot_id, and character_actions.
        scene_details: List of scene_detail dictionaries for transform lookup fallback.
    
    Returns:
        Dict with success status and details.
    """
    try:
        scene_id = shot_detail.get("scene_id")
        shot_id = shot_detail.get("shot_id")
        character_actions = shot_detail.get("character_actions", [])
        
        if not character_actions:
            return {
                "success": True,
                "message": f"No character_actions for scene {scene_id} shot {shot_id}. Skipping.",
            }
        
        shot_scene_name = f"Scene_{scene_id}_Shot_{shot_id}"
        
        # Get the existing shot scene (should already be created by apply_asset_modifications)
        if shot_scene_name not in bpy.data.scenes:
            return {
                "success": False,
                "scene_id": scene_id,
                "shot_id": shot_id,
                "error": f"Shot scene '{shot_scene_name}' not found. It should be created by apply_asset_modifications first.",
            }
        
        shot_scene = bpy.data.scenes[shot_scene_name]
        bpy.context.window.scene = shot_scene
        
        failed_imports = []
        
        # For each character action, get transform, unlink static, and import animated
        for action in character_actions:
            asset_id = action.get("asset_id")
            animated_path = action.get("animated_path")
            
            if not animated_path:
                failed_imports.append({
                    "asset_id": asset_id,
                    "error": "No animated_path provided",
                })
                continue
            
            if not os.path.exists(animated_path):
                failed_imports.append({
                    "asset_id": asset_id,
                    "error": f"Animated file not found: {animated_path}",
                })
                continue
            
            # Detect pipeline.  The "pipeline" field (set by auto_route) is the
            # most reliable indicator.  Fall back to checking for
            # text_to_motion_prompt ONLY when there is no action_id (pure
            # Uthana-only pipeline without auto_route).
            pipeline = action.get("pipeline", "")
            if pipeline:
                is_uthana = pipeline == "uthana"
            else:
                # Legacy / standalone Uthana pipeline: has prompt but no action_id
                is_uthana = ("text_to_motion_prompt" in action
                             and action.get("action_id") is None)
            
            # Build object name based on pipeline (include shot context for uniqueness)
            if is_uthana:
                motion_id = action.get("motion_id", "uthana")
                new_object_name = f"{asset_id}_s{scene_id}sh{shot_id}_{motion_id}"
            else:
                action_id = str(action.get("action_id", ""))
                action_name = action.get("action_name", "")
                new_object_name = f"{asset_id}_s{scene_id}sh{shot_id}_{action_id}_{action_name}"
            
            # Get transform from the character in the shot scene BEFORE unlinking
            transform = get_asset_transform_from_scene(shot_scene, asset_id)
            
            # If not found in shot scene, try layout_description as fallback
            if transform is None:
                transform = get_asset_transform_from_layout(scene_details, scene_id, asset_id)
            
            # Override location/rotation with start_coords/start_rotation
            # (used by both Uthana and Meshy walking/running animations)
            start_coords = action.get("start_coords")
            end_coords = action.get("end_coords")
            start_rotation = action.get("start_rotation")
            
            # Record the intended start z before import (for z-offset calc later)
            intended_start_z = 0.0
            if start_coords:
                intended_start_z = start_coords.get("z", 0.0)
                # Override the location portion of the transform
                override_loc = {
                    "x": start_coords.get("x", 0.0),
                    "y": start_coords.get("y", 0.0),
                    "z": start_coords.get("z", 0.0),
                }
                if transform is None:
                    transform = {"location": override_loc}
                else:
                    transform = {**transform, "location": override_loc}
            elif transform and transform.get("location"):
                intended_start_z = transform["location"].get("z", 0.0)
            
            # Override rotation with start_rotation if provided (both pipelines)
            if start_rotation:
                override_rot = {
                    "x": start_rotation.get("x", 0.0),
                    "y": start_rotation.get("y", 0.0),
                    "z": start_rotation.get("z", 0.0),
                }
                if transform is None:
                    transform = {"rotation": override_rot}
                else:
                    transform = {**transform, "rotation": override_rot}

            # For Uthana models: only apply Z rotation (facing direction) and
            # let the glTF importer's X/Y rotation (Y-up → Z-up correction)
            # remain untouched.  Without this, setting x=0 would remove the
            # correction on models that need it.
            if is_uthana and transform and transform.get("rotation"):
                z_only = transform["rotation"].get("z")
                transform = {**transform, "rotation": {"z": z_only} if z_only is not None else {}}

            # Unlink the static character (and any prior animated version
            # from a previous run) from the shot scene.
            unlink_result = unlink_character_from_scene(
                shot_scene, asset_id, scene_id=scene_id, shot_id=shot_id
            )
            if not unlink_result.get("success"):
                print(f"Warning: Could not unlink '{asset_id}' from scene: {unlink_result.get('error')}", flush=True)
            
            # Import the animated asset
            import_result = import_animated_asset(
                filepath=animated_path,
                target_scene=shot_scene,
                new_object_name=new_object_name,
                transform=transform,
            )
            
            if not import_result.get("success"):
                failed_imports.append({
                    "asset_id": asset_id,
                    "error": import_result.get("error", "Unknown import error"),
                })
                continue
            
            # Retarget root motion so the character ends at end_coords.
            # Applies to both Uthana and Meshy whenever end_coords is set.
            # _retarget_root_motion handles both cases:
            #   - Has root motion FCurves → scale them
            #   - In-place animation (zero displacement) → tween object location
            if end_coords and import_result.get("success"):
                root_obj = bpy.data.objects.get(import_result.get("object_name"))
                if root_obj:
                    current_loc = root_obj.location.copy()
                    z_offset = current_loc.z - intended_start_z
                    desired_end = mathutils.Vector((
                        end_coords.get("x", current_loc.x),
                        end_coords.get("y", current_loc.y),
                        end_coords.get("z", 0.0) + z_offset,
                    ))
                    frame_start, frame_end = _get_animation_frame_range(root_obj)
                    ok = _retarget_root_motion(
                        root_obj, desired_end, frame_start, frame_end,
                    )
                    if ok:
                        print(
                            f"  Retargeted root motion for '{asset_id}' → "
                            f"end ({desired_end.x:.2f}, {desired_end.y:.2f}) "
                            f"over frames {frame_start}–{frame_end}",
                            flush=True,
                        )
                    else:
                        print(
                            f"  Warning: no root-motion F-Curves found for '{asset_id}'; "
                            f"end_coords ignored",
                            flush=True,
                        )
        
        if failed_imports:
            return {
                "success": False,
                "scene_id": scene_id,
                "shot_id": shot_id,
                "scene_name": shot_scene_name,
                "failed_imports": failed_imports,
            }
        
        return {
            "success": True,
            "scene_id": scene_id,
            "shot_id": shot_id,
            "scene_name": shot_scene_name,
        }
    
    except Exception as e:
        return {
            "success": False,
            "scene_id": shot_detail.get("scene_id"),
            "shot_id": shot_detail.get("shot_id"),
            "error": str(e),
        }


def import_animated_assets_to_all_shots(
    shot_details: List[dict],
    scene_details: List[dict],
) -> dict:
    """
    Import animated assets to all shots.

    Processes shots **in order** within each scene so that cross-shot
    continuity is maintained: if a character moved to ``end_coords`` in
    shot N, the import for shot N+1 will place that character at those
    coordinates (unless an explicit ``start_coords`` overrides it).
    
    Args:
        shot_details: List of shot_detail dictionaries.
        scene_details: List of scene_detail dictionaries for transform lookup.
    
    Returns:
        Dict with success status and details.
    """
    try:
        all_failed = []
        successful_shots = []

        # --- Build per-scene, sorted shot order ---
        # Continuity tracking: (scene_id, asset_id) → last known end state
        # end state = {"location": {x,y,z}, "rotation": {x,y,z}}
        continuity: Dict[str, dict] = {}

        # Sort shots by (scene_id, shot_id) for deterministic ordering
        ordered = sorted(shot_details, key=lambda s: (s.get("scene_id", 0), s.get("shot_id", 0)))

        for shot_detail in ordered:
            scene_id = shot_detail.get("scene_id")

            # Inject continuity state into character_actions before import
            for action in shot_detail.get("character_actions", []):
                aid = action.get("asset_id")
                ckey = f"{scene_id}::{aid}"
                prev = continuity.get(ckey)

                # If no explicit start_coords, inherit from previous shot's end
                if action.get("start_coords") is None and prev:
                    action["start_coords"] = prev.get("location")
                    print(
                        f"  Continuity: {aid} scene {scene_id} shot {shot_detail.get('shot_id')} "
                        f"inherits start from previous end {prev.get('location')}",
                        flush=True,
                    )

                # If no explicit start_rotation, inherit from previous shot's end rotation
                if action.get("start_rotation") is None and prev and prev.get("rotation"):
                    action["start_rotation"] = prev.get("rotation")

            result = import_animated_assets_to_shot(
                shot_detail=shot_detail,
                scene_details=scene_details,
            )

            # Update continuity state from this shot's actions
            for action in shot_detail.get("character_actions", []):
                aid = action.get("asset_id")
                ckey = f"{scene_id}::{aid}"
                ec = action.get("end_coords")
                sr = action.get("start_rotation")
                if ec:
                    continuity[ckey] = {
                        "location": ec,
                        "rotation": sr,  # may be None — that's fine
                    }
                elif action.get("start_coords"):
                    # Character was repositioned but no end_coords → stays at start
                    continuity[ckey] = {
                        "location": action["start_coords"],
                        "rotation": sr,
                    }
            
            if result.get("success"):
                successful_shots.append({
                    "scene_id": result.get("scene_id"),
                    "shot_id": result.get("shot_id"),
                    "scene_name": result.get("scene_name"),
                })
            else:
                all_failed.append(result)
        
        if all_failed:
            return {
                "success": False,
                "successful_shots": successful_shots,
                "failed_shots": all_failed,
            }
        
        return {
            "success": True,
            "successful_shots": successful_shots,
        }
    
    except Exception as e:
        print(f"Error importing animated assets to all shots: {str(e)}", flush=True)
        return {
            "success": False,
            "error": str(e),
        }


def import_animated_assets_to_all_shots_json_input(json_filepath: str) -> dict:
    """
    Import animated assets to all shots from a JSON file.
    
    This function reads a story script JSON file and:
    1. Loops through shot_details
    2. For each shot, uses the existing Scene_{scene_id}_Shot_{shot_id} (created by apply_asset_modifications)
    3. Reads character transforms from the shot scene, then unlinks character actors
    4. Imports animated models with the same transforms
    5. Names imported objects as {asset_id}_{action_id}_{action_name}
    
    Args:
        json_filepath: Path to the JSON file containing shot_details and scene_details.
    
    Returns:
        Dict with success status and details.
    """
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        shot_details = data.get("shot_details", [])
        scene_details = data.get("scene_details", [])
        
        if not shot_details:
            return {
                "success": False,
                "error": "No 'shot_details' found in JSON file",
            }
        
        if not scene_details:
            return {
                "success": False,
                "error": "No 'scene_details' found in JSON file",
            }
        
        return import_animated_assets_to_all_shots(
            shot_details=shot_details,
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


def delete_shot(scene_id: int, shot_id: int) -> dict:
    """
    Delete a specific shot scene.
    
    Args:
        scene_id: The scene_id of the shot.
        shot_id: The shot_id of the shot.
    
    Returns:
        Dict with success status and details.
    """
    try:
        scene_name = f"Scene_{scene_id}_Shot_{shot_id}"
        
        if scene_name not in bpy.data.scenes:
            return {
                "success": False,
                "error": f"Scene '{scene_name}' not found.",
            }
        
        scene = bpy.data.scenes[scene_name]
        
        # Collect all objects unique to this scene (not linked to other scenes)
        objects_to_delete = []
        for obj in scene.collection.all_objects:
            # Check if this object is only in this scene
            in_other_scenes = False
            for other_scene in bpy.data.scenes:
                if other_scene != scene:
                    if obj.name in other_scene.collection.all_objects:
                        in_other_scenes = True
                        break
            if not in_other_scenes:
                objects_to_delete.append(obj)
        
        # Delete objects unique to this scene
        for obj in objects_to_delete:
            bpy.data.objects.remove(obj, do_unlink=True)
        
        # Delete the scene
        bpy.data.scenes.remove(scene, do_unlink=True)
        
        return {
            "success": True,
            "message": f"Deleted scene '{scene_name}' and {len(objects_to_delete)} unique objects.",
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def delete_all_shots_of_scene(scene_id: int) -> dict:
    """
    Delete all shot scenes for a given scene_id.
    
    Deletes all scenes matching pattern Scene_{scene_id}_Shot_{shot_id},
    but does NOT delete the original scene Scene_{scene_id}.
    
    Args:
        scene_id: The scene_id to delete shots for.
    
    Returns:
        Dict with success status and details.
    """
    try:
        # Pattern to match: Scene_{scene_id}_Shot_{any_shot_id}
        pattern = re.compile(rf"^Scene_{scene_id}_Shot_(\d+)$")
        
        scenes_to_delete = []
        for scene in bpy.data.scenes:
            if pattern.match(scene.name):
                scenes_to_delete.append(scene.name)
        
        if not scenes_to_delete:
            return {
                "success": True,
                "message": f"No shot scenes found for scene_id {scene_id}.",
                "deleted_scenes": [],
            }
        
        deleted_scenes = []
        errors = []
        
        for scene_name in scenes_to_delete:
            # Extract shot_id from scene name
            match = pattern.match(scene_name)
            if match:
                shot_id = int(match.group(1))
                result = delete_shot(scene_id, shot_id)
                if result.get("success"):
                    deleted_scenes.append(scene_name)
                else:
                    errors.append({
                        "scene_name": scene_name,
                        "error": result.get("error"),
                    })
        
        if errors:
            return {
                "success": False,
                "deleted_scenes": deleted_scenes,
                "errors": errors,
            }
        
        return {
            "success": True,
            "message": f"Deleted {len(deleted_scenes)} shot scenes for scene_id {scene_id}.",
            "deleted_scenes": deleted_scenes,
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def delete_all_shots() -> dict:
    """
    Delete all shot scenes in Blender.
    
    Deletes all scenes matching pattern Scene_{scene_id}_Shot_{shot_id},
    but leaves all other scenes intact (including original scenes like Scene_{scene_id}).
    
    Returns:
        Dict with success status and details.
    """
    try:
        # Pattern to match: Scene_{any_scene_id}_Shot_{any_shot_id}
        pattern = re.compile(r"^Scene_(\d+)_Shot_(\d+)$")
        
        scenes_to_delete = []
        for scene in bpy.data.scenes:
            if pattern.match(scene.name):
                scenes_to_delete.append(scene.name)
        
        if not scenes_to_delete:
            return {
                "success": True,
                "message": "No shot scenes found.",
                "deleted_scenes": [],
            }
        
        deleted_scenes = []
        errors = []
        
        for scene_name in scenes_to_delete:
            match = pattern.match(scene_name)
            if match:
                scene_id = int(match.group(1))
                shot_id = int(match.group(2))
                result = delete_shot(scene_id, shot_id)
                if result.get("success"):
                    deleted_scenes.append(scene_name)
                else:
                    errors.append({
                        "scene_name": scene_name,
                        "error": result.get("error"),
                    })
        
        if errors:
            return {
                "success": False,
                "deleted_scenes": deleted_scenes,
                "errors": errors,
            }
        
        return {
            "success": True,
            "message": f"Deleted {len(deleted_scenes)} shot scenes.",
            "deleted_scenes": deleted_scenes,
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
