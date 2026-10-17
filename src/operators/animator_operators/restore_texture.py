#!/usr/bin/env python3
"""
Restore textures (normal, metallic, roughness) to a rigged GLB model.
Uses pygltflib to repackage textures into the GLB file.
"""

import os
from pathlib import Path
from pygltflib import GLTF2, Image, Texture, TextureInfo, NormalMaterialTexture, Material
import base64
import json


def load_image_as_data_uri(image_path: str) -> str:
    """Load an image file and convert it to a data URI."""
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # Determine MIME type based on extension
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    mime_type = mime_types.get(ext, "image/png")
    
    encoded = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def add_image_to_gltf(gltf: GLTF2, image_path: str, name: str) -> int:
    """Add an image to the GLTF and return its index."""
    data_uri = load_image_as_data_uri(image_path)
    
    image = Image()
    image.uri = data_uri
    image.name = name
    
    if gltf.images is None:
        gltf.images = []
    
    image_index = len(gltf.images)
    gltf.images.append(image)
    return image_index


def add_texture_to_gltf(gltf: GLTF2, image_index: int, name: str) -> int:
    """Add a texture referencing an image and return its index."""
    texture = Texture()
    texture.source = image_index
    texture.name = name
    
    if gltf.textures is None:
        gltf.textures = []
    
    texture_index = len(gltf.textures)
    gltf.textures.append(texture)
    return texture_index


def restore_textures(
    glb_path: str,
    texture_dir: str,
    output_path: str = None,
    base_color_file: str = None,
    normal_file: str = None,
    metallic_roughness_file: str = None,
):
    """
    Restore textures to a rigged GLB model.
    
    Args:
        glb_path: Path to the input GLB file
        texture_dir: Directory containing texture files
        output_path: Path to save the output GLB (defaults to input with _textured suffix)
        base_color_file: Filename of base color texture (auto-detected if None)
        normal_file: Filename of normal map texture (auto-detected if None)
        metallic_roughness_file: Filename of metallic-roughness texture (auto-detected if None)
    """
    # Load the GLB file
    gltf = GLTF2().load(glb_path)
    
    # Auto-detect texture files if not specified
    texture_files = os.listdir(texture_dir)
    manifest_path = Path(texture_dir) / "texture_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    base_color_file = base_color_file or manifest.get("base_color")
    normal_file = normal_file or manifest.get("normal")
    metallic_roughness_file = metallic_roughness_file or manifest.get("metallic_roughness")
    
    if base_color_file is None and not manifest_path.exists():
        # Find base color (typically doesn't have _normal or _metallic/_roughness suffix)
        for f in texture_files:
            if f.endswith((".png", ".jpg", ".jpeg")):
                lower = f.lower()
                if "normal" not in lower and "metallic" not in lower and "roughness" not in lower:
                    base_color_file = f
                    break
    
    if normal_file is None and not manifest_path.exists():
        for f in texture_files:
            if "normal" in f.lower() and f.endswith((".png", ".jpg", ".jpeg")):
                normal_file = f
                break
    
    if metallic_roughness_file is None and not manifest_path.exists():
        for f in texture_files:
            lower = f.lower()
            if ("metallic" in lower or "roughness" in lower) and f.endswith((".png", ".jpg", ".jpeg")):
                metallic_roughness_file = f
                break
    
    print(f"Base color texture: {base_color_file}")
    print(f"Normal texture: {normal_file}")
    print(f"Metallic-roughness texture: {metallic_roughness_file}")
    
    # Add images and textures
    texture_indices = {}
    
    if base_color_file:
        base_color_path = os.path.join(texture_dir, base_color_file)
        if os.path.exists(base_color_path):
            img_idx = add_image_to_gltf(gltf, base_color_path, "baseColorTexture")
            tex_idx = add_texture_to_gltf(gltf, img_idx, "baseColorTexture")
            texture_indices["baseColor"] = tex_idx
            print(f"Added base color texture at index {tex_idx}")
    
    if normal_file:
        normal_path = os.path.join(texture_dir, normal_file)
        if os.path.exists(normal_path):
            img_idx = add_image_to_gltf(gltf, normal_path, "normalTexture")
            tex_idx = add_texture_to_gltf(gltf, img_idx, "normalTexture")
            texture_indices["normal"] = tex_idx
            print(f"Added normal texture at index {tex_idx}")
    
    if metallic_roughness_file:
        mr_path = os.path.join(texture_dir, metallic_roughness_file)
        if os.path.exists(mr_path):
            img_idx = add_image_to_gltf(gltf, mr_path, "metallicRoughnessTexture")
            tex_idx = add_texture_to_gltf(gltf, img_idx, "metallicRoughnessTexture")
            texture_indices["metallicRoughness"] = tex_idx
            print(f"Added metallic-roughness texture at index {tex_idx}")
    
    # Update materials to use the textures
    if gltf.materials:
        for i, material in enumerate(gltf.materials):
            print(f"Updating material {i}: {material.name}")
            
            # Ensure pbrMetallicRoughness exists
            if material.pbrMetallicRoughness is None:
                from pygltflib import PbrMetallicRoughness
                material.pbrMetallicRoughness = PbrMetallicRoughness()
            
            pbr = material.pbrMetallicRoughness
            
            # Explicit source material roles replace any maps changed by rigging.
            # Legacy folders without a manifest only fill missing texture slots.
            if "baseColor" in texture_indices:
                if pbr.baseColorTexture is None or "base_color" in manifest:
                    pbr.baseColorTexture = TextureInfo(index=texture_indices["baseColor"])
                    print(f"  Set baseColorTexture to index {texture_indices['baseColor']}")
                else:
                    print(f"  baseColorTexture already exists at index {pbr.baseColorTexture.index}, skipping")
            
            # Restore the source metallic/roughness map when its role is known.
            if "metallicRoughness" in texture_indices:
                if pbr.metallicRoughnessTexture is None or "metallic_roughness" in manifest:
                    pbr.metallicRoughnessTexture = TextureInfo(index=texture_indices["metallicRoughness"])
                    # When using a texture, typically set factors to 1.0 so texture controls the values
                    pbr.metallicFactor = 1.0
                    pbr.roughnessFactor = 1.0
                    print(f"  Set metallicRoughnessTexture to index {texture_indices['metallicRoughness']}")
                else:
                    print(f"  metallicRoughnessTexture already exists at index {pbr.metallicRoughnessTexture.index}, skipping")
            
            # Restore the source normal map when its role is known.
            if "normal" in texture_indices:
                if material.normalTexture is None or "normal" in manifest:
                    material.normalTexture = NormalMaterialTexture(index=texture_indices["normal"])
                    print(f"  Set normalTexture to index {texture_indices['normal']}")
                else:
                    print(f"  normalTexture already exists at index {material.normalTexture.index}, skipping")
    else:
        print("Warning: No materials found in the GLB file")
    
    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(glb_path)
        output_path = f"{base}_textured{ext}"
    
    # Save the modified GLB
    gltf.save(output_path)
    print(f"Saved textured model to: {output_path}")
    
    return output_path


def main():
    """Main entry point with default paths for the prince model."""
    script_dir = Path(__file__).parent
    
    glb_path = script_dir / "prince_rigged.glb"
    texture_dir = script_dir / "prince_texture"
    output_path = script_dir / "prince_rigged_textured.glb"
    
    if not glb_path.exists():
        print(f"Error: GLB file not found: {glb_path}")
        return
    
    if not texture_dir.exists():
        print(f"Error: Texture directory not found: {texture_dir}")
        return
    
    restore_textures(
        glb_path=str(glb_path),
        texture_dir=str(texture_dir),
        output_path=str(output_path),
    )


if __name__ == "__main__":
    main()
