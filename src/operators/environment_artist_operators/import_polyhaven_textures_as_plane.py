"""
Poly Haven Texture Importer for Blender
Automatically imports and sets up PBR textures from Poly Haven
"""

import bpy
import os
from pathlib import Path


def import_polyhaven_textures_as_plane(
    textures_path,
    side_length=2,
    scale=0.1,
    midlevel=0.5,
    plane_name="Polyhaven_Plane"
):
    """
    Import Poly Haven textures and create a plane with complete PBR material setup as the plane of the scene.
    
    Parameters:
    -----------
    textures_path : str
        Path to the directory containing Poly Haven texture files
    side_length : float, optional
        Side length of the square plane (default: 2)
    scale : float, optional
        Scale value for displacement (default: 0.1)
    midlevel : float, optional
        Midlevel value for displacement (default: 0.5)
    plane_name : str, optional
        Name of the created plane (default: "Polyhaven_Plane")
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'success': bool - Whether the operation succeeded
        - 'message': str - Status message
        - 'plane': bpy.types.Object or None - Created plane object
        - 'material': bpy.types.Material or None - Created material
        - 'textures_found': dict - Which textures were found
    
    Texture Naming Convention:
    --------------------------
    The function looks for files with these patterns:
    
    Color & Visibility:
    - Base Color: *_diff_* (diffuse/albedo) - sRGB
    - Alpha: *_alpha_* (transparency mask) - Non-Color
    - Emission: *_emission_* (emissive color) - sRGB
    
    Surface Details:
    - Normal: *_nor_gl_* (OpenGL format, preferred) or *_nor_dx_* (DirectX) - Non-Color
    - Displacement: *_disp_* (height map) - Non-Color
    
    Surface Properties:
    - Roughness: *_rough_* - Non-Color
    - Metallic: *_metal_* - Non-Color
    - Ambient Occlusion: *_ao_* - Non-Color
    
    Packed Maps:
    - ARM: *_arm_* (AO in R, Roughness in G, Metallic in B) - Non-Color
      Note: Individual ao/rough/metal maps take priority over ARM channels
    - Mask: *_mask_* (custom masking) - Non-Color
    
    Supported formats: .jpg, .jpeg, .png, .exr, .tif, .tiff
    """
    
    # Validate input path
    if not textures_path:
        return {
            'success': False,
            'message': 'Error: textures_path cannot be empty',
            'plane': None,
            'material': None,
            'textures_found': {}
        }
    
    texture_dir = Path(textures_path)
    
    if not texture_dir.exists():
        return {
            'success': False,
            'message': f'Error: Directory does not exist: {textures_path}',
            'plane': None,
            'material': None,
            'textures_found': {}
        }
    
    if not texture_dir.is_dir():
        return {
            'success': False,
            'message': f'Error: Path is not a directory: {textures_path}',
            'plane': None,
            'material': None,
            'textures_found': {}
        }
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.exr', '.tif', '.tiff'}
    
    # Find texture files
    # Primary patterns for each texture type
    texture_patterns = {
        'base_color': '_diff_',
        'alpha': '_alpha_',
        'emission': '_emission_',
        'roughness': '_rough_',
        'metallic': '_metal_',
        'normal_gl': '_nor_gl_',
        'normal_dx': '_nor_dx_',
        'displacement': '_disp_',
        'ao': '_ao_',
        'arm': '_arm_',
        'mask': '_mask_'
    }
    
    found_textures = {}
    textures_found_status = {}
    
    print("="*60)
    print("Searching for texture files...")
    print("="*60)
    
    for tex_type, pattern in texture_patterns.items():
        found = False
        for file_path in texture_dir.iterdir():
            if file_path.is_file():
                if pattern in file_path.name.lower() and file_path.suffix.lower() in image_extensions:
                    found_textures[tex_type] = str(file_path)
                    textures_found_status[tex_type] = True
                    found = True
                    print(f"✓ Found {tex_type}: {file_path.name}")
                    break
        
        if not found:
            textures_found_status[tex_type] = False
            print(f"✗ Missing {tex_type} (pattern: *{pattern}*)")
    
    # Check if we have at least base color
    if 'base_color' not in found_textures:
        return {
            'success': False,
            'message': 'Error: Base color texture is required but not found',
            'plane': None,
            'material': None,
            'textures_found': textures_found_status
        }
    
    # Resolve normal map: prefer OpenGL (nor_gl) over DirectX (nor_dx)
    if 'normal_gl' in found_textures:
        found_textures['normal'] = found_textures['normal_gl']
        print(f"  → Using OpenGL normal map (nor_gl)")
    elif 'normal_dx' in found_textures:
        found_textures['normal'] = found_textures['normal_dx']
        print(f"  → Using DirectX normal map (nor_dx) - Green channel inverted from Blender standard")
    
    # Resolve ARM packed map vs individual maps
    # Individual maps take priority over ARM channels
    use_arm_for_ao = False
    use_arm_for_roughness = False
    use_arm_for_metallic = False
    
    if 'arm' in found_textures:
        if 'ao' not in found_textures:
            use_arm_for_ao = True
            print(f"  → Using ARM Red channel for Ambient Occlusion")
        if 'roughness' not in found_textures:
            use_arm_for_roughness = True
            print(f"  → Using ARM Green channel for Roughness")
        if 'metallic' not in found_textures:
            use_arm_for_metallic = True
            print(f"  → Using ARM Blue channel for Metallic")
    
    print("="*60)
    
    # Extract material name from texture files
    base_name = Path(found_textures['base_color']).name
    # Remove the pattern and extension to get base name
    material_name = base_name.replace('_diff_', '_').rsplit('.', 1)[0]
    # Clean up the name
    material_name = material_name.replace('_4k', '').replace('_2k', '').replace('_1k', '')
    
    try:
        # Create plane
        bpy.ops.mesh.primitive_plane_add(size=side_length, location=(0, 0, 0))
        plane = bpy.context.active_object
        plane.name = plane_name
        
        print(f"✓ Created plane: {plane.name}")
        print(f"  Side length: {side_length}")
        
        # Create material
        mat = bpy.data.materials.new(name=material_name)
        mat.use_nodes = True
        plane.data.materials.append(mat)
        
        # Get node tree
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Create shader nodes
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        output_node.location = (306, 144.5)
        
        bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf_node.location = (-79.5, 294.3)
        
        tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
        tex_coord_node.location = (29.5, -35.7)
        
        mapping_node = nodes.new(type='ShaderNodeMapping')
        mapping_node.location = (229.5, -35.7)
        
        reroute_node = nodes.new(type='NodeReroute')
        reroute_node.location = (34.4, -590.8)
        
        # Connect texture coordinate chain
        links.new(tex_coord_node.outputs['UV'], mapping_node.inputs['Vector'])
        links.new(mapping_node.outputs['Vector'], reroute_node.inputs[0])
        
        # Create and connect texture nodes
        y_offset = -36.0
        y_spacing = 280
        current_y = y_offset
        
        # Track ARM texture node if needed (created once, used for multiple channels)
        arm_tex_node = None
        arm_separate_node = None
        
        # Create ARM texture node if any channel will use it
        if use_arm_for_ao or use_arm_for_roughness or use_arm_for_metallic:
            arm_tex_node = nodes.new(type='ShaderNodeTexImage')
            arm_tex_node.location = (84.4, current_y - y_spacing * 8)
            arm_tex_node.label = "ARM (AO/Rough/Metal)"
            arm_tex_node.image = bpy.data.images.load(found_textures['arm'])
            arm_tex_node.image.colorspace_settings.name = 'Non-Color'
            links.new(reroute_node.outputs[0], arm_tex_node.inputs['Vector'])
            
            # Create Separate RGB node for ARM
            arm_separate_node = nodes.new(type='ShaderNodeSeparateRGB')
            arm_separate_node.location = (384.4, current_y - y_spacing * 8)
            arm_separate_node.label = "Separate ARM"
            links.new(arm_tex_node.outputs['Color'], arm_separate_node.inputs['Image'])
            print(f"✓ Loaded ARM texture for channel separation")
        
        # Base Color
        if 'base_color' in found_textures:
            base_tex = nodes.new(type='ShaderNodeTexImage')
            base_tex.location = (84.4, current_y)
            base_tex.label = "Base Color"
            base_tex.image = bpy.data.images.load(found_textures['base_color'])
            base_tex.image.colorspace_settings.name = 'sRGB'
            links.new(reroute_node.outputs[0], base_tex.inputs['Vector'])
            
            # If we have AO, multiply it with base color
            if 'ao' in found_textures or use_arm_for_ao:
                mix_ao_node = nodes.new(type='ShaderNodeMixRGB')
                mix_ao_node.location = (-279.5, 294.3)
                mix_ao_node.blend_type = 'MULTIPLY'
                mix_ao_node.inputs['Fac'].default_value = 1.0
                mix_ao_node.label = "AO Mix"
                links.new(base_tex.outputs['Color'], mix_ao_node.inputs['Color1'])
                links.new(mix_ao_node.outputs['Color'], bsdf_node.inputs['Base Color'])
                # AO connection will be made below
            else:
                links.new(base_tex.outputs['Color'], bsdf_node.inputs['Base Color'])
            print(f"✓ Connected Base Color texture")
            current_y -= y_spacing
        
        # Alpha
        if 'alpha' in found_textures:
            alpha_tex = nodes.new(type='ShaderNodeTexImage')
            alpha_tex.location = (84.4, current_y)
            alpha_tex.label = "Alpha"
            alpha_tex.image = bpy.data.images.load(found_textures['alpha'])
            alpha_tex.image.colorspace_settings.name = 'Non-Color'
            links.new(reroute_node.outputs[0], alpha_tex.inputs['Vector'])
            links.new(alpha_tex.outputs['Color'], bsdf_node.inputs['Alpha'])
            # Enable alpha blend mode
            mat.surface_render_method = 'BLENDED'
            print(f"✓ Connected Alpha texture")
            current_y -= y_spacing
        
        # Emission
        if 'emission' in found_textures:
            emission_tex = nodes.new(type='ShaderNodeTexImage')
            emission_tex.location = (84.4, current_y)
            emission_tex.label = "Emission"
            emission_tex.image = bpy.data.images.load(found_textures['emission'])
            emission_tex.image.colorspace_settings.name = 'sRGB'
            links.new(reroute_node.outputs[0], emission_tex.inputs['Vector'])
            links.new(emission_tex.outputs['Color'], bsdf_node.inputs['Emission Color'])
            bsdf_node.inputs['Emission Strength'].default_value = 1.0
            print(f"✓ Connected Emission texture")
            current_y -= y_spacing
        
        # Roughness (individual map or ARM Green channel)
        if 'roughness' in found_textures:
            rough_tex = nodes.new(type='ShaderNodeTexImage')
            rough_tex.location = (84.4, current_y)
            rough_tex.label = "Roughness"
            rough_tex.image = bpy.data.images.load(found_textures['roughness'])
            rough_tex.image.colorspace_settings.name = 'Non-Color'
            links.new(reroute_node.outputs[0], rough_tex.inputs['Vector'])
            links.new(rough_tex.outputs['Color'], bsdf_node.inputs['Roughness'])
            print(f"✓ Connected Roughness texture")
            current_y -= y_spacing
        elif use_arm_for_roughness and arm_separate_node:
            links.new(arm_separate_node.outputs['G'], bsdf_node.inputs['Roughness'])
            print(f"✓ Connected Roughness from ARM Green channel")
        
        # Metallic (individual map or ARM Blue channel)
        if 'metallic' in found_textures:
            metal_tex = nodes.new(type='ShaderNodeTexImage')
            metal_tex.location = (84.4, current_y)
            metal_tex.label = "Metallic"
            metal_tex.image = bpy.data.images.load(found_textures['metallic'])
            metal_tex.image.colorspace_settings.name = 'Non-Color'
            links.new(reroute_node.outputs[0], metal_tex.inputs['Vector'])
            links.new(metal_tex.outputs['Color'], bsdf_node.inputs['Metallic'])
            print(f"✓ Connected Metallic texture")
            current_y -= y_spacing
        elif use_arm_for_metallic and arm_separate_node:
            links.new(arm_separate_node.outputs['B'], bsdf_node.inputs['Metallic'])
            print(f"✓ Connected Metallic from ARM Blue channel")
        
        # Ambient Occlusion (individual map or ARM Red channel)
        # AO is mixed with Base Color above, here we just connect the source
        if 'ao' in found_textures:
            ao_tex = nodes.new(type='ShaderNodeTexImage')
            ao_tex.location = (84.4, current_y)
            ao_tex.label = "Ambient Occlusion"
            ao_tex.image = bpy.data.images.load(found_textures['ao'])
            ao_tex.image.colorspace_settings.name = 'Non-Color'
            links.new(reroute_node.outputs[0], ao_tex.inputs['Vector'])
            # Connect to mix node if it exists
            for node in nodes:
                if node.label == "AO Mix":
                    links.new(ao_tex.outputs['Color'], node.inputs['Color2'])
                    break
            print(f"✓ Connected Ambient Occlusion texture")
            current_y -= y_spacing
        elif use_arm_for_ao and arm_separate_node:
            # Connect ARM Red channel to AO mix node
            for node in nodes:
                if node.label == "AO Mix":
                    links.new(arm_separate_node.outputs['R'], node.inputs['Color2'])
                    break
            print(f"✓ Connected Ambient Occlusion from ARM Red channel")
        
        # Normal
        if 'normal' in found_textures:
            normal_tex = nodes.new(type='ShaderNodeTexImage')
            normal_tex.location = (84.4, current_y)
            normal_tex.label = "Normal"
            normal_tex.image = bpy.data.images.load(found_textures['normal'])
            normal_tex.image.colorspace_settings.name = 'Non-Color'
            
            normal_map_node = nodes.new(type='ShaderNodeNormalMap')
            normal_map_node.location = (-329.5, -65.7)
            
            links.new(reroute_node.outputs[0], normal_tex.inputs['Vector'])
            links.new(normal_tex.outputs['Color'], normal_map_node.inputs['Color'])
            links.new(normal_map_node.outputs['Normal'], bsdf_node.inputs['Normal'])
            print(f"✓ Connected Normal texture")
            current_y -= y_spacing
        
        # Displacement
        if 'displacement' in found_textures:
            disp_tex = nodes.new(type='ShaderNodeTexImage')
            disp_tex.location = (84.4, current_y)
            disp_tex.label = "Displacement"
            disp_tex.image = bpy.data.images.load(found_textures['displacement'])
            disp_tex.image.colorspace_settings.name = 'Non-Color'
            
            disp_node = nodes.new(type='ShaderNodeDisplacement')
            disp_node.location = (20.5, -405.7)
            disp_node.inputs['Scale'].default_value = scale
            disp_node.inputs['Midlevel'].default_value = midlevel
            
            links.new(reroute_node.outputs[0], disp_tex.inputs['Vector'])
            links.new(disp_tex.outputs['Color'], disp_node.inputs['Height'])
            links.new(disp_node.outputs['Displacement'], output_node.inputs['Displacement'])
            print(f"✓ Connected Displacement texture (scale: {scale}, midlevel: {midlevel})")
            current_y -= y_spacing
        
        # Mask (for custom use - stored as a value node, not connected to BSDF)
        if 'mask' in found_textures:
            mask_tex = nodes.new(type='ShaderNodeTexImage')
            mask_tex.location = (84.4, current_y)
            mask_tex.label = "Mask"
            mask_tex.image = bpy.data.images.load(found_textures['mask'])
            mask_tex.image.colorspace_settings.name = 'Non-Color'
            links.new(reroute_node.outputs[0], mask_tex.inputs['Vector'])
            print(f"✓ Loaded Mask texture (available for custom use)")
            current_y -= y_spacing
        
        # Connect BSDF to output
        links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
        
        # Add organizational frames
        frame_mapping = nodes.new(type='NodeFrame')
        frame_mapping.label = "Mapping"
        tex_coord_node.parent = frame_mapping
        mapping_node.parent = frame_mapping
        
        frame_textures = nodes.new(type='NodeFrame')
        frame_textures.label = "Textures"
        for node in nodes:
            if node.type == 'TEX_IMAGE':
                node.parent = frame_textures
        
        # Set up displacement
        mat.displacement_method = 'DISPLACEMENT'
        
        # Add subdivision surface modifier if displacement exists
        if 'displacement' in found_textures:
            subsurf = plane.modifiers.new(name="Subdivision", type='SUBSURF')
            subsurf.levels = 2  # Viewport: lower for performance
            subsurf.render_levels = 5  # Render: higher for quality
            subsurf.subdivision_type = 'SIMPLE'
            print(f"✓ Added Subdivision Surface modifier (viewport: 2, render: 5, SIMPLE)")
        
        
        print("="*60)
        
        # Build success message
        # Count actual textures used (excluding intermediate keys like normal_gl/normal_dx)
        used_textures = [k for k in found_textures.keys() if k not in ('normal_gl', 'normal_dx')]
        texture_count = len(used_textures)
        
        # Core texture types that are commonly expected
        core_types = ['base_color', 'roughness', 'normal', 'displacement']
        optional_types = ['alpha', 'emission', 'metallic', 'ao', 'arm', 'mask']
        
        missing_core = [t for t in core_types if t not in found_textures]
        found_optional = [t for t in optional_types if t in found_textures]
        
        success_msg = f"✅ Successfully created plane with {texture_count} textures\n"
        success_msg += f"   Material: {material_name}\n"
        success_msg += f"   Plane: {plane.name}\n"
        success_msg += f"   Used textures: {', '.join(used_textures)}\n"
        
        if use_arm_for_ao or use_arm_for_roughness or use_arm_for_metallic:
            arm_channels = []
            if use_arm_for_ao:
                arm_channels.append('AO')
            if use_arm_for_roughness:
                arm_channels.append('Roughness')
            if use_arm_for_metallic:
                arm_channels.append('Metallic')
            success_msg += f"   ARM channels used: {', '.join(arm_channels)}\n"
        
        if missing_core:
            success_msg += f"   Note: Missing core textures: {', '.join(missing_core)}"
        
        return {
            'success': True,
            'message': success_msg,
            'plane': plane,
            'material': mat,
            'textures_found': textures_found_status
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f'Error during setup: {str(e)}',
            'plane': None,
            'material': None,
            'textures_found': textures_found_status
        }


def create_repetitive_plane(
    plane_name,
    x_times,
    y_times,
    output_plane="output_plane"
):
    """
    Create a new plane with repeated/tiled textures from an existing textured plane.
    
    This function duplicates the source plane, scales it to x_times * y_times the original size,
    and tiles the textures accordingly with anti-tiling techniques to avoid visible seams.
    
    Parameters:
    -----------
    plane_name : str
        Name of the source plane with textures to repeat
    x_times : int
        Number of repetitions in the X axis
    y_times : int
        Number of repetitions in the Y axis
    output_plane : str, optional
        Name for the output plane (default: "output_plane")
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'success': bool - Whether the operation succeeded
        - 'message': str - Status message
        - 'plane': bpy.types.Object or None - Created plane object
        - 'material': bpy.types.Material or None - Modified material
    """
    
    # Validate inputs
    if x_times < 1 or y_times < 1:
        return {
            'success': False,
            'message': 'Error: x_times and y_times must be >= 1',
            'plane': None,
            'material': None
        }
    
    # Find source plane
    source_plane = bpy.data.objects.get(plane_name)
    if source_plane is None:
        return {
            'success': False,
            'message': f'Error: Plane "{plane_name}" not found',
            'plane': None,
            'material': None
        }
    
    if source_plane.type != 'MESH':
        return {
            'success': False,
            'message': f'Error: Object "{plane_name}" is not a mesh',
            'plane': None,
            'material': None
        }
    
    # Check if source has materials
    if len(source_plane.data.materials) == 0:
        return {
            'success': False,
            'message': f'Error: Plane "{plane_name}" has no materials',
            'plane': None,
            'material': None
        }
    
    source_material = source_plane.data.materials[0]
    if source_material is None or not source_material.use_nodes:
        return {
            'success': False,
            'message': f'Error: Source material does not use nodes',
            'plane': None,
            'material': None
        }
    
    try:
        # Get source plane dimensions
        source_dimensions = source_plane.dimensions.copy()
        source_x = source_dimensions.x
        source_y = source_dimensions.y
        
        # Calculate new plane size
        new_x = source_x * x_times
        new_y = source_y * y_times
        
        # Duplicate the plane
        bpy.ops.object.select_all(action='DESELECT')
        source_plane.select_set(True)
        bpy.context.view_layer.objects.active = source_plane
        bpy.ops.object.duplicate()
        
        new_plane = bpy.context.active_object
        new_plane.name = output_plane
        
        # Scale the plane to new dimensions
        scale_x = x_times
        scale_y = y_times
        new_plane.scale.x *= scale_x
        new_plane.scale.y *= scale_y
        
        # Apply scale
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        # Create a copy of the material for the new plane
        new_material = source_material.copy()
        new_material.name = f"{source_material.name}_tiled_{x_times}x{y_times}"
        
        # Replace material on new plane
        new_plane.data.materials.clear()
        new_plane.data.materials.append(new_material)
        
        # Modify the material's node tree for tiling with anti-tiling
        nodes = new_material.node_tree.nodes
        links = new_material.node_tree.links
        
        # Find the mapping node
        mapping_node = None
        tex_coord_node = None
        reroute_node = None
        
        for node in nodes:
            if node.type == 'MAPPING':
                mapping_node = node
            elif node.type == 'TEX_COORD':
                tex_coord_node = node
            elif node.type == 'REROUTE':
                reroute_node = node
        
        if mapping_node is None or tex_coord_node is None:
            return {
                'success': False,
                'message': 'Error: Could not find Mapping or Texture Coordinate node in material',
                'plane': None,
                'material': None
            }
        
        # Set up tiling scale in the mapping node
        mapping_node.inputs['Scale'].default_value[0] = x_times
        mapping_node.inputs['Scale'].default_value[1] = y_times
        
        # Create anti-tiling node group to reduce visible seams
        # This uses noise to slightly offset UV coordinates per tile
        
        # Store mapping node output location for reconnection
        mapping_output_location = mapping_node.location
        
        # Create nodes for anti-tiling
        # Noise texture for randomization
        noise_tex = nodes.new(type='ShaderNodeTexNoise')
        noise_tex.location = (mapping_node.location.x + 200, mapping_node.location.y - 200)
        noise_tex.label = "Tile Noise"
        noise_tex.inputs['Scale'].default_value = 1.0  # One noise cell per tile
        noise_tex.inputs['Detail'].default_value = 0.0
        noise_tex.inputs['Roughness'].default_value = 0.0
        
        # Separate the noise into components
        separate_xyz = nodes.new(type='ShaderNodeSeparateXYZ')
        separate_xyz.location = (noise_tex.location.x + 180, noise_tex.location.y)
        separate_xyz.label = "Separate Noise"
        
        # Create floor nodes to get tile index
        floor_x = nodes.new(type='ShaderNodeMath')
        floor_x.location = (mapping_node.location.x + 200, mapping_node.location.y + 100)
        floor_x.operation = 'FLOOR'
        floor_x.label = "Floor X"
        
        floor_y = nodes.new(type='ShaderNodeMath')
        floor_y.location = (floor_x.location.x, floor_x.location.y - 80)
        floor_y.operation = 'FLOOR'
        floor_y.label = "Floor Y"
        
        # Combine floor values to create tile coordinate
        combine_floor = nodes.new(type='ShaderNodeCombineXYZ')
        combine_floor.location = (floor_x.location.x + 150, floor_x.location.y - 40)
        combine_floor.label = "Tile Index"
        
        # Separate scaled UV
        separate_uv = nodes.new(type='ShaderNodeSeparateXYZ')
        separate_uv.location = (mapping_node.location.x + 200, mapping_node.location.y)
        separate_uv.label = "Separate UV"
        
        # Offset value (small random offset per tile to break repetition)
        offset_scale = nodes.new(type='ShaderNodeValue')
        offset_scale.location = (noise_tex.location.x, noise_tex.location.y - 100)
        offset_scale.outputs[0].default_value = 0.02  # Small offset to avoid obvious artifacts
        offset_scale.label = "Offset Amount"
        
        # Multiply noise by offset scale
        mult_offset_x = nodes.new(type='ShaderNodeMath')
        mult_offset_x.location = (separate_xyz.location.x + 150, separate_xyz.location.y + 40)
        mult_offset_x.operation = 'MULTIPLY'
        mult_offset_x.label = "Scale Offset X"
        
        mult_offset_y = nodes.new(type='ShaderNodeMath')
        mult_offset_y.location = (mult_offset_x.location.x, mult_offset_x.location.y - 80)
        mult_offset_y.operation = 'MULTIPLY'
        mult_offset_y.label = "Scale Offset Y"
        
        # Subtract 0.5 to center the noise around 0
        sub_half_x = nodes.new(type='ShaderNodeMath')
        sub_half_x.location = (mult_offset_x.location.x + 150, mult_offset_x.location.y)
        sub_half_x.operation = 'SUBTRACT'
        sub_half_x.inputs[1].default_value = 0.01  # Half of offset
        sub_half_x.label = "Center X"
        
        sub_half_y = nodes.new(type='ShaderNodeMath')
        sub_half_y.location = (sub_half_x.location.x, sub_half_x.location.y - 80)
        sub_half_y.operation = 'SUBTRACT'
        sub_half_y.inputs[1].default_value = 0.01
        sub_half_y.label = "Center Y"
        
        # Add offset to UV
        add_x = nodes.new(type='ShaderNodeMath')
        add_x.location = (sub_half_x.location.x + 150, sub_half_x.location.y)
        add_x.operation = 'ADD'
        add_x.label = "Offset UV X"
        
        add_y = nodes.new(type='ShaderNodeMath')
        add_y.location = (add_x.location.x, add_x.location.y - 80)
        add_y.operation = 'ADD'
        add_y.label = "Offset UV Y"
        
        # Combine back to vector
        combine_uv = nodes.new(type='ShaderNodeCombineXYZ')
        combine_uv.location = (add_x.location.x + 150, add_x.location.y - 40)
        combine_uv.label = "Tiled UV"
        
        # Connect the anti-tiling network
        links.new(mapping_node.outputs['Vector'], separate_uv.inputs['Vector'])
        links.new(separate_uv.outputs['X'], floor_x.inputs[0])
        links.new(separate_uv.outputs['Y'], floor_y.inputs[0])
        links.new(floor_x.outputs['Value'], combine_floor.inputs['X'])
        links.new(floor_y.outputs['Value'], combine_floor.inputs['Y'])
        
        # Use tile index as noise input
        links.new(combine_floor.outputs['Vector'], noise_tex.inputs['Vector'])
        links.new(noise_tex.outputs['Color'], separate_xyz.inputs['Vector'])
        
        # Scale the noise offset
        links.new(separate_xyz.outputs['X'], mult_offset_x.inputs[0])
        links.new(offset_scale.outputs[0], mult_offset_x.inputs[1])
        links.new(separate_xyz.outputs['Y'], mult_offset_y.inputs[0])
        links.new(offset_scale.outputs[0], mult_offset_y.inputs[1])
        
        # Center around 0
        links.new(mult_offset_x.outputs['Value'], sub_half_x.inputs[0])
        links.new(mult_offset_y.outputs['Value'], sub_half_y.inputs[0])
        
        # Add offset to original UV
        links.new(separate_uv.outputs['X'], add_x.inputs[0])
        links.new(sub_half_x.outputs['Value'], add_x.inputs[1])
        links.new(separate_uv.outputs['Y'], add_y.inputs[0])
        links.new(sub_half_y.outputs['Value'], add_y.inputs[1])
        
        # Combine to final UV
        links.new(add_x.outputs['Value'], combine_uv.inputs['X'])
        links.new(add_y.outputs['Value'], combine_uv.inputs['Y'])
        links.new(separate_uv.outputs['Z'], combine_uv.inputs['Z'])
        
        # Reconnect all texture nodes to use the new tiled UV
        if reroute_node:
            # Find all links from reroute to texture nodes
            links_to_update = []
            for link in new_material.node_tree.links:
                if link.from_node == reroute_node:
                    links_to_update.append(link.to_socket)
            
            # Remove the old connection from mapping to reroute
            for link in list(new_material.node_tree.links):
                if link.to_node == reroute_node and link.from_node == mapping_node:
                    new_material.node_tree.links.remove(link)
                    break
            
            # Connect combine_uv to reroute
            links.new(combine_uv.outputs['Vector'], reroute_node.inputs[0])
        else:
            # Direct connection to texture nodes
            for node in nodes:
                if node.type == 'TEX_IMAGE':
                    for link in list(new_material.node_tree.links):
                        if link.to_node == node and link.to_socket.name == 'Vector':
                            if link.from_node == mapping_node:
                                new_material.node_tree.links.remove(link)
                                links.new(combine_uv.outputs['Vector'], node.inputs['Vector'])
                                break
        
        # Create a frame for anti-tiling nodes
        frame_antitile = nodes.new(type='NodeFrame')
        frame_antitile.label = "Anti-Tiling"
        for node in [noise_tex, separate_xyz, floor_x, floor_y, combine_floor,
                     separate_uv, offset_scale, mult_offset_x, mult_offset_y,
                     sub_half_x, sub_half_y, add_x, add_y, combine_uv]:
            node.parent = frame_antitile
        
        print("="*60)
        print(f"✓ Created repetitive plane: {output_plane}")
        print(f"  Source plane: {plane_name}")
        print(f"  Repetitions: {x_times}x (X) × {y_times}x (Y)")
        print(f"  New dimensions: {new_x:.2f} × {new_y:.2f}")
        print(f"  Material: {new_material.name}")
        print(f"  Anti-tiling: Enabled (noise-based UV offset)")
        print("="*60)
        
        return {
            'success': True,
            'message': f"✅ Successfully created repetitive plane '{output_plane}' with {x_times}×{y_times} tiling",
            'plane': new_plane,
            'material': new_material
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'message': f'Error during repetitive plane creation: {str(e)}',
            'plane': None,
            'material': None
        }


def create_repetitive_plane_from_polyhaven(
    textures_path,
    width,
    depth,
    side_length=2.0,
    scale=0.1,
    midlevel=0.5,
    output_plane="output_plane",
    enable_backface_culling=True
):
    """
    Create a tiled plane with Poly Haven textures covering a specified area.
    
    This function combines import_polyhaven_textures_as_plane and create_repetitive_plane
    to create a textured plane that covers the specified width and depth dimensions.
    
    Parameters:
    -----------
    textures_path : str
        Path to the directory containing Poly Haven texture files
    width : float
        Desired width of the output plane in meters (X axis)
    depth : float
        Desired depth of the output plane in meters (Y axis)
    side_length : float, optional
        Side length of a single texture tile in meters (default: 2.0)
    scale : float, optional
        Scale value for displacement (default: 0.1)
    midlevel : float, optional
        Midlevel value for displacement (default: 0.5)
    output_plane : str, optional
        Name for the output plane (default: "output_plane")
    enable_backface_culling : bool, optional
        If True, enable backface culling so the plane is invisible from below
        but visible from above (default: True)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'success': bool - Whether the operation succeeded
        - 'message': str - Status message
        - 'plane': bpy.types.Object or None - Created plane object
        - 'material': bpy.types.Material or None - Created material
        - 'x_times': int - Number of tiles in X direction
        - 'y_times': int - Number of tiles in Y direction
    """
    import math
    
    # Validate inputs
    if width <= 0 or depth <= 0:
        return {
            'success': False,
            'message': 'Error: width and depth must be positive values',
            'plane': None,
            'material': None,
            'x_times': 0,
            'y_times': 0
        }
    
    if side_length <= 0:
        return {
            'success': False,
            'message': 'Error: side_length must be a positive value',
            'plane': None,
            'material': None,
            'x_times': 0,
            'y_times': 0
        }
    
    # Calculate number of tiles needed (ceiling to ensure coverage)
    x_times = math.ceil(width / side_length)
    y_times = math.ceil(depth / side_length)
    
    # Ensure at least 1 tile
    x_times = max(1, x_times)
    y_times = max(1, y_times)
    
    print("="*60)
    print(f"Creating tiled plane from Poly Haven textures")
    print(f"  Target dimensions: {width}m × {depth}m")
    print(f"  Tile size: {side_length}m × {side_length}m")
    print(f"  Tiles needed: {x_times} × {y_times}")
    print(f"  Actual dimensions: {x_times * side_length}m × {y_times * side_length}m")
    print("="*60)
    
    # Step 1: Create the base plane with Poly Haven textures
    base_result = import_polyhaven_textures_as_plane(
        textures_path=textures_path,
        side_length=side_length,
        scale=scale,
        midlevel=midlevel,
        plane_name="Polyhaven_Plane"
    )
    
    if not base_result['success']:
        return {
            'success': False,
            'message': f"Failed to create base plane: {base_result['message']}",
            'plane': None,
            'material': None,
            'x_times': x_times,
            'y_times': y_times
        }
    
    # If only 1×1 tiling needed, just rename the base plane
    if x_times == 1 and y_times == 1:
        base_plane = base_result['plane']
        base_plane.name = output_plane
        
        # Enable backface culling if requested
        if enable_backface_culling and base_plane.data.materials:
            _enable_backface_culling(base_plane.data.materials[0])
        
        return {
            'success': True,
            'message': f"✅ Created plane '{output_plane}' (no tiling needed, 1×1)",
            'plane': base_plane,
            'material': base_result['material'],
            'x_times': 1,
            'y_times': 1
        }
    
    # Step 2: Create the repetitive plane
    tile_result = create_repetitive_plane(
        plane_name=base_result['plane'].name,
        x_times=x_times,
        y_times=y_times,
        output_plane=output_plane
    )
    
    if not tile_result['success']:
        return {
            'success': False,
            'message': f"Failed to create tiled plane: {tile_result['message']}",
            'plane': None,
            'material': None,
            'x_times': x_times,
            'y_times': y_times
        }
    
    # Delete the base plane and clean up its data to reduce file size
    base_plane = base_result['plane']
    base_material = base_result['material']
    base_mesh = base_plane.data
    
    # Remove the base plane object
    bpy.data.objects.remove(base_plane, do_unlink=True)
    
    # Remove the base mesh data if it has no users
    if base_mesh and base_mesh.users == 0:
        bpy.data.meshes.remove(base_mesh)
    
    # Remove the base material if it has no users (tiled material uses same textures)
    if base_material and base_material.users == 0:
        bpy.data.materials.remove(base_material)
    
    print(f"✓ Cleaned up base plane 'Polyhaven_Plane' and unused data")
    
    # Switch to Cycles renderer and enable GPU
    bpy.context.scene.render.engine = 'CYCLES'
    
    # Enable GPU rendering
    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
    # Try to find available GPU device types
    for compute_device_type in ['METAL', 'OPTIX', 'HIP', 'CUDA', 'ONEAPI']:
        try:
            cycles_prefs.compute_device_type = compute_device_type
            cycles_prefs.get_devices()
            # Check if we have any devices of this type
            devices = cycles_prefs.devices
            if any(d.type == compute_device_type for d in devices):
                # Enable all GPU devices
                for device in devices:
                    device.use = True
                print(f"✓ Enabled {compute_device_type} GPU rendering")
                break
        except:
            continue
    
    bpy.context.scene.cycles.device = 'GPU'
    
    actual_width = x_times * side_length
    actual_depth = y_times * side_length
    
    # Enable backface culling if requested
    tiled_plane = tile_result['plane']
    if enable_backface_culling and tiled_plane.data.materials:
        _enable_backface_culling(tiled_plane.data.materials[0])
    
    return {
        'success': True,
        'message': f"✅ Created tiled plane '{output_plane}' ({x_times}×{y_times} tiles, {actual_width}m × {actual_depth}m)",
        'plane': tiled_plane,
        'material': tile_result['material'],
        'x_times': x_times,
        'y_times': y_times
    }


def create_indoor_walls(
    textures_path,
    x=5,
    x_negative=-5,
    y=5,
    y_negative=-5,
    z=4,
    side_length=2.0,
    scale=0,
    midlevel=0.5
):
    """
    Create indoor walls and roof forming a room, with backface culling enabled
    so walls are invisible when viewed from outside.
    
    Parameters:
    -----------
    textures_path : str
        Path to the directory containing Poly Haven texture files
    x : float, optional
        Positive X boundary of the room (default: 5)
    x_negative : float, optional
        Negative X boundary of the room (default: -5)
    y : float, optional
        Positive Y boundary of the room (default: 5)
    y_negative : float, optional
        Negative Y boundary of the room (default: -5)
    z : float, optional
        Height of the walls in meters (default: 4)
    side_length : float, optional
        Side length of a single texture tile in meters (default: 2.0)
    scale : float, optional
        Scale value for displacement (default: 0)
    midlevel : float, optional
        Midlevel value for displacement (default: 0.5)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'success': bool - Whether the operation succeeded
        - 'message': str - Status message
        - 'walls': dict - Dictionary of wall objects
        - 'roof': bpy.types.Object or None - Roof object
    """
    import math
    
    # Calculate room dimensions
    room_width = x - x_negative  # X dimension
    room_depth = y - y_negative  # Y dimension
    room_height = z
    
    print("="*60)
    print(f"Creating indoor walls and roof")
    print(f"  Room boundaries: X=[{x_negative}, {x}], Y=[{y_negative}, {y}], Z=[0, {z}]")
    print(f"  Room dimensions: {room_width}m × {room_depth}m × {room_height}m")
    print("="*60)
    
    walls = {}
    created_objects = []
    
    # Wall configurations: name, width, height, position, rotation
    wall_configs = [
        # wall_y: at y position, facing -Y (inward), width = room_width, height = z
        ("wall_y", room_width, room_height, (x_negative + room_width/2, y, z/2), (math.pi/2, 0, 0)),
        # wall_y_negative: at y_negative position, facing +Y (inward), width = room_width, height = z
        ("wall_y_negative", room_width, room_height, (x_negative + room_width/2, y_negative, z/2), (math.pi/2, 0, math.pi)),
        # wall_x: at x position, facing -X (inward), width = room_depth, height = z
        ("wall_x", room_depth, room_height, (x, y_negative + room_depth/2, z/2), (math.pi/2, 0, -math.pi/2)),
        # wall_x_negative: at x_negative position, facing +X (inward), width = room_depth, height = z
        ("wall_x_negative", room_depth, room_height, (x_negative, y_negative + room_depth/2, z/2), (math.pi/2, 0, math.pi/2)),
    ]
    
    # Roof configuration
    roof_config = ("roof", room_width, room_depth, (x_negative + room_width/2, y_negative + room_depth/2, z), (0, 0, 0))
    
    try:
        # Create each wall
        for wall_name, width, height, position, rotation in wall_configs:
            print(f"\nCreating {wall_name}...")
            
            # Create the wall plane
            result = create_repetitive_plane_from_polyhaven(
                textures_path=textures_path,
                width=width,
                depth=height,
                side_length=side_length,
                scale=scale,
                midlevel=midlevel,
                output_plane=f"_temp_{wall_name}"
            )
            
            if not result['success']:
                # Clean up created objects
                for obj in created_objects:
                    if obj and obj.name in bpy.data.objects:
                        bpy.data.objects.remove(obj, do_unlink=True)
                return {
                    'success': False,
                    'message': f"Failed to create {wall_name}: {result['message']}",
                    'walls': {},
                    'roof': None
                }
            
            wall_plane = result['plane']
            created_objects.append(wall_plane)
            
            # Calculate actual dimensions created (may be larger due to tiling)
            actual_width = result['x_times'] * side_length
            actual_height = result['y_times'] * side_length
            
            # Crop to desired size if needed
            _crop_plane_to_size(wall_plane, width, height, actual_width, actual_height)
            
            # Position and rotate the wall
            wall_plane.location = position
            wall_plane.rotation_euler = rotation
            
            # Apply a small scale to eliminate gaps between walls and floor
            wall_plane.scale = (1.02, 1.02, 1.02)
            
            # Apply transforms
            bpy.ops.object.select_all(action='DESELECT')
            wall_plane.select_set(True)
            bpy.context.view_layer.objects.active = wall_plane
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
            
            # Rename to final name
            wall_plane.name = wall_name
            
            # Enable backface culling for this wall's material
            if wall_plane.data.materials:
                _enable_backface_culling(wall_plane.data.materials[0])
            
            # Disable shadow ray visibility for Cycles
            wall_plane.visible_shadow = False
            
            walls[wall_name] = wall_plane
            print(f"  ✓ {wall_name} created at {position}")
        
        # Create roof
        print(f"\nCreating roof...")
        roof_name, roof_width, roof_depth, roof_position, roof_rotation = roof_config
        
        result = create_repetitive_plane_from_polyhaven(
            textures_path=textures_path,
            width=roof_width,
            depth=roof_depth,
            side_length=side_length,
            scale=scale,
            midlevel=midlevel,
            output_plane=f"_temp_{roof_name}"
        )
        
        if not result['success']:
            # Clean up created objects
            for obj in created_objects:
                if obj and obj.name in bpy.data.objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
            return {
                'success': False,
                'message': f"Failed to create roof: {result['message']}",
                'walls': {},
                'roof': None
            }
        
        roof_plane = result['plane']
        created_objects.append(roof_plane)
        
        # Calculate actual dimensions created
        actual_width = result['x_times'] * side_length
        actual_depth = result['y_times'] * side_length
        
        # Crop roof to desired size
        _crop_plane_to_size(roof_plane, roof_width, roof_depth, actual_width, actual_depth)
        
        # Position the roof (flip it so it faces downward into the room)
        roof_plane.location = roof_position
        roof_plane.rotation_euler = (math.pi, 0, 0)  # Flip to face downward
        
        # Apply transforms
        bpy.ops.object.select_all(action='DESELECT')
        roof_plane.select_set(True)
        bpy.context.view_layer.objects.active = roof_plane
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        
        # Rename to final name
        roof_plane.name = roof_name
        
        # Enable backface culling for roof material
        if roof_plane.data.materials:
            _enable_backface_culling(roof_plane.data.materials[0])
        
        # Disable shadow ray visibility for Cycles
        roof_plane.visible_shadow = False
        
        print(f"  ✓ roof created at {roof_position}")
        
        print("\n" + "="*60)
        print(f"✅ Successfully created indoor walls and roof")
        print(f"   Walls: wall_x, wall_x_negative, wall_y, wall_y_negative")
        print(f"   Roof: roof")
        print(f"   Backface culling: Enabled (transparent from outside)")
        print("="*60)
        
        return {
            'success': True,
            'message': f"✅ Created indoor walls and roof with backface culling",
            'walls': walls,
            'roof': roof_plane
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Clean up created objects on error
        for obj in created_objects:
            if obj and obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        return {
            'success': False,
            'message': f'Error during indoor walls creation: {str(e)}',
            'walls': {},
            'roof': None
        }


def _crop_plane_to_size(plane, target_width, target_depth, actual_width, actual_depth):
    """
    Crop a plane to the target dimensions by modifying its mesh vertices.
    
    Parameters:
    -----------
    plane : bpy.types.Object
        The plane object to crop
    target_width : float
        Desired width (X dimension)
    target_depth : float
        Desired depth (Y dimension)
    actual_width : float
        Current width of the plane
    actual_depth : float
        Current depth of the plane
    """
    import bmesh
    
    # Only crop if actual dimensions are larger than target
    if actual_width <= target_width and actual_depth <= target_depth:
        return
    
    # Get the mesh data
    mesh = plane.data
    
    # Create a bmesh to work with
    bm = bmesh.new()
    bm.from_mesh(mesh)
    
    # Calculate the crop boundaries
    # Plane is centered at origin, so bounds are -half to +half
    half_target_width = target_width / 2
    half_target_depth = target_depth / 2
    
    # Move vertices that are outside the target bounds to the bounds
    for vert in bm.verts:
        if vert.co.x > half_target_width:
            vert.co.x = half_target_width
        elif vert.co.x < -half_target_width:
            vert.co.x = -half_target_width
            
        if vert.co.y > half_target_depth:
            vert.co.y = half_target_depth
        elif vert.co.y < -half_target_depth:
            vert.co.y = -half_target_depth
    
    # Update the mesh
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def _enable_backface_culling(material):
    """
    Enable backface culling for a material by adding a Geometry node
    and Mix Shader that makes backfaces transparent.
    
    This works in both Eevee viewport and Cycles render.
    
    Parameters:
    -----------
    material : bpy.types.Material
        The material to modify
    """
    if not material or not material.use_nodes:
        return
    
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    
    # Find the Material Output node and the current surface shader
    output_node = None
    current_shader = None
    current_shader_link = None
    
    for node in nodes:
        if node.type == 'OUTPUT_MATERIAL':
            output_node = node
            break
    
    if not output_node:
        return
    
    # Find what's connected to the Surface input
    for link in links:
        if link.to_node == output_node and link.to_socket.name == 'Surface':
            current_shader = link.from_node
            current_shader_link = link
            break
    
    if not current_shader:
        return
    
    # Remove the current link to Surface
    links.remove(current_shader_link)
    
    # Create Geometry node
    geometry_node = nodes.new(type='ShaderNodeNewGeometry')
    geometry_node.location = (output_node.location.x - 600, output_node.location.y + 200)
    geometry_node.label = "Backface Detection"
    
    # Create Transparent BSDF
    transparent_node = nodes.new(type='ShaderNodeBsdfTransparent')
    transparent_node.location = (output_node.location.x - 400, output_node.location.y - 100)
    transparent_node.label = "Backface Transparent"
    
    # Create Mix Shader
    mix_shader_node = nodes.new(type='ShaderNodeMixShader')
    mix_shader_node.location = (output_node.location.x - 200, output_node.location.y)
    mix_shader_node.label = "Backface Culling Mix"
    
    # Connect: Geometry.Backfacing -> Mix Shader.Fac
    links.new(geometry_node.outputs['Backfacing'], mix_shader_node.inputs['Fac'])
    
    # Connect: Current Shader -> Mix Shader.Shader (first slot, shown when Fac=0)
    links.new(current_shader.outputs[0], mix_shader_node.inputs[1])
    
    # Connect: Transparent BSDF -> Mix Shader.Shader (second slot, shown when Fac=1)
    links.new(transparent_node.outputs['BSDF'], mix_shader_node.inputs[2])
    
    # Connect: Mix Shader -> Material Output.Surface
    links.new(mix_shader_node.outputs['Shader'], output_node.inputs['Surface'])
    
    # Enable blend mode for transparency in Eevee
    material.surface_render_method = 'BLENDED'
    
    # Enable backface culling in viewport settings as well
    material.use_backface_culling = True
