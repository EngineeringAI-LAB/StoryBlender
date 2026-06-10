import bpy
import os

def _switch_or_create_scene(scene_name: str) -> bpy.types.Scene:
    if scene_name in bpy.data.scenes:
        scene = bpy.data.scenes[scene_name]
        bpy.context.window.scene = scene
        return scene
    bpy.ops.scene.new(type="EMPTY")
    new_scene = bpy.context.window.scene
    new_scene.name = scene_name
    return new_scene

def _delete_scene_and_its_objects(scene_name: str) -> None:
    if scene_name not in bpy.data.scenes:
        return
    scene = bpy.data.scenes[scene_name]
    objects_in_temp_scene = set(scene.collection.all_objects)
    objects_in_other_scenes = set()
    for other_scene in bpy.data.scenes:
        if other_scene.name != scene_name:
            objects_in_other_scenes.update(other_scene.collection.all_objects)
    
    objects_to_delete = objects_in_temp_scene - objects_in_other_scenes
    for obj in objects_to_delete:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.scenes.remove(scene, do_unlink=True)

def sanitize_materials_for_compatibility():
    """
    Blender 4.0+ triggers KHR_materials_specular and KHR_materials_ior 
    which instantly breaks certain strict native OS viewers. This resets the specific 
    sliders that cause the exporter to flag these extensions, while keeping the textures intact.
    """
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                # Forcing these values to exact defaults prevents strict extensions from generating
                if 'Specular IOR Level' in node.inputs and not node.inputs['Specular IOR Level'].is_linked:
                    node.inputs['Specular IOR Level'].default_value = 0.5
                if 'Transmission Weight' in node.inputs and not node.inputs['Transmission Weight'].is_linked:
                    node.inputs['Transmission Weight'].default_value = 0.0
                if 'Coat Weight' in node.inputs and not node.inputs['Coat Weight'].is_linked:
                    node.inputs['Coat Weight'].default_value = 0.0

def compress_glb(input_path: str, export_path: str):
    """
    Imports and exports a GLB specifically tuned for highly strict native viewers.
    (Draco is forced OFF to guarantee compatibility).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    original_scene = bpy.context.window.scene
    original_scene_name = original_scene.name
    temp_scene_name = "Temp_GLB_Compression"
    _switch_or_create_scene(temp_scene_name)
    current_scene = bpy.context.scene

    try:
        # Clear temp scene
        for obj in list(current_scene.collection.all_objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for coll in list(current_scene.collection.children):
            try: bpy.data.collections.remove(coll)
            except: pass

        # Import the model
        bpy.ops.object.select_all(action='DESELECT')
        ext = os.path.splitext(input_path)[1].lower()
        if ext in ['.glb', '.gltf']: bpy.ops.import_scene.gltf(filepath=input_path)
        elif ext == '.fbx': bpy.ops.import_scene.fbx(filepath=input_path)
        elif ext == '.obj': bpy.ops.import_scene.obj(filepath=input_path)

        # Flatten Collections
        imported_objects = list(current_scene.collection.all_objects)
        for obj in imported_objects:
            for coll in list(obj.users_collection):
                coll.objects.unlink(obj)
            current_scene.collection.objects.link(obj)
            obj.select_set(True)
            
        def remove_all_child_collections(parent_collection):
            for child_coll in list(parent_collection.children):
                remove_all_child_collections(child_coll) 
                try: bpy.data.collections.remove(child_coll)
                except: pass
        remove_all_child_collections(current_scene.collection)

        if imported_objects:
            bpy.context.view_layer.objects.active = imported_objects[0]

        sanitize_materials_for_compatibility()

        # Export Parameters
        os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
        
        bpy.ops.export_scene.gltf(
            filepath=export_path,
            export_format='GLB',
            
            # Core
            use_selection=True,          
            use_active_scene=True,       
            export_apply=True,
            export_yup=True,
            
            # Geometry & Materials
            export_texcoords=True,
            export_normals=True,
            export_tangents=True,
            export_materials='EXPORT',
            export_image_format='JPEG', # JPEG is safe and shrinks size 
            
            # COMPATIBILITY FIXES: Explicitly turn off anything that can cause empty arrays
            export_draco_mesh_compression_enable=False, # Draco MUST be false for strict native viewers
            export_cameras=False,
            export_lights=False,
            export_extras=False,
            export_animations=False, # Empty animation tracks break strict parsers
            export_morph=False,
            export_skins=False,
            check_existing=False,
        )
        print(f"Successfully processed highly compatible GLB: {export_path}")

    except Exception as e:
        print(f"Failed to process GLB: {e}")

    finally:
        if original_scene_name in bpy.data.scenes:
            bpy.context.window.scene = bpy.data.scenes[original_scene_name]
        _delete_scene_and_its_objects(temp_scene_name)
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

# --- Example Usage ---
# compress_glb("input_model.glb", "output_model.glb")