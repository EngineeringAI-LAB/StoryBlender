"""Test GLB texture compression.

Tests the PNG-to-JPEG texture compression logic from uthana_animator.py
on the example input file.
"""

import io
import os
from typing import Optional

def _compress_glb_textures(input_path: str, output_path: str, jpeg_quality: int = 85) -> str:
    """Re-pack a GLB file, converting embedded PNG textures to JPEG.

    This mirrors the key size-reduction trick in compress_glb.py
    (export_image_format='JPEG') but works outside Blender by
    manipulating the GLB binary directly via pygltflib.

    Returns *output_path* on success, *input_path* if compression fails or
    doesn't help.
    """
    try:
        from pygltflib import GLTF2
        from PIL import Image
    except ImportError as exc:
        print(f"  Warning: compression skipped ({exc})")
        return input_path

    try:
        gltf = GLTF2.load(input_path)
    except Exception as exc:
        print(f"  Warning: failed to parse GLB for compression ({exc})")
        return input_path

    # Collect all image data replacements
    replacements = {}  # bufferView_index -> jpeg_bytes
    
    for image in gltf.images or []:
        bv_index = image.bufferView
        if bv_index is None:
            continue
        mime = (image.mimeType or "").lower()
        if "png" not in mime:
            continue  # already JPEG or unknown — leave alone

        bv = gltf.bufferViews[bv_index]
        buf = gltf.binary_blob()
        if buf is None:
            continue
        png_bytes = buf[bv.byteOffset : bv.byteOffset + bv.byteLength]

        # Convert PNG → JPEG (to achieve <30 MB target)
        try:
            img = Image.open(io.BytesIO(png_bytes))
            print(f"  Texture: mode={img.mode}, size={img.size}")
            
            # Check if image has meaningful alpha (transparency)
            has_alpha = img.mode == "RGBA"
            if has_alpha:
                alpha = img.split()[3]
                # Check if alpha has any non-opaque pixels
                alpha_min, alpha_max = alpha.getextrema()
                print(f"  Alpha channel: min={alpha_min}, max={alpha_max}")
                if alpha_min < 255:
                    # Has transparency - keep as PNG to preserve texture
                    print(f"  Skipping texture with alpha channel (preserving PNG)")
                    continue
                else:
                    print(f"  Alpha is fully opaque, converting to JPEG")
            
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=jpeg_quality)
            jpeg_bytes = out.getvalue()
            print(f"  Converted to JPEG: {len(png_bytes)} → {len(jpeg_bytes)} bytes")
        except Exception as exc:
            print(f"  Warning: texture conversion failed ({exc})")
            continue

        if len(jpeg_bytes) >= len(png_bytes):
            print(f"  JPEG not smaller, skipping")
            continue

        replacements[bv_index] = jpeg_bytes
        image.mimeType = "image/jpeg"
        print(f"  Queued buffer view {bv_index} for JPEG replacement")

    if not replacements:
        return input_path

    # Rebuild the binary blob with replacements
    try:
        buf = gltf.binary_blob()
        if buf is None:
            return input_path
        
        blob = bytearray(buf)
        
        # Process replacements in reverse order of byteOffset to avoid offset shifts
        sorted_bvs = sorted(replacements.keys(), 
                           key=lambda x: gltf.bufferViews[x].byteOffset, 
                           reverse=True)
        
        for bv_index in sorted_bvs:
            jpeg_bytes = replacements[bv_index]
            bv = gltf.bufferViews[bv_index]
            png_bytes = blob[bv.byteOffset : bv.byteOffset + bv.byteLength]
            
            # Replace in place (same size or smaller with padding)
            blob[bv.byteOffset : bv.byteOffset + len(png_bytes)] = jpeg_bytes + b'\x00' * (len(png_bytes) - len(jpeg_bytes))
            bv.byteLength = len(jpeg_bytes)
            print(f"  Replaced buffer view {bv_index}")
        
        gltf.set_binary_blob(bytes(blob))
    except Exception as exc:
        print(f"  Warning: buffer rebuild failed ({exc})")
        return input_path

    # Save the modified GLB
    try:
        gltf.save(output_path)
        return output_path
    except Exception as exc:
        print(f"  Warning: saving compressed GLB failed ({exc})")
        return input_path


def verify_glb_textures(glb_path: str) -> bool:
    """Verify that a GLB file has valid textures and material references."""
    try:
        from pygltflib import GLTF2
        from PIL import Image
    except ImportError as exc:
        print(f"  Warning: cannot verify ({exc})")
        return False

    try:
        gltf = GLTF2.load(glb_path)
    except Exception as exc:
        print(f"  ❌ Failed to load GLB: {exc}")
        return False

    if not gltf.images:
        print(f"  ❌ No images found in GLB")
        return False

    print(f"\nVerifying {len(gltf.images)} textures:")
    valid_count = 0
    for i, image in enumerate(gltf.images):
        bv_index = image.bufferView
        if bv_index is None:
            print(f"  Image {i}: no buffer view")
            continue
        
        bv = gltf.bufferViews[bv_index]
        buf = gltf.binary_blob()
        if buf is None:
            print(f"  Image {i}: no binary blob")
            continue
        
        img_bytes = buf[bv.byteOffset : bv.byteOffset + bv.byteLength]
        mime = image.mimeType or "unknown"
        
        try:
            img = Image.open(io.BytesIO(img_bytes))
            print(f"  Image {i}: {mime}, {img.mode}, {img.size} - ✅ valid")
            valid_count += 1
        except Exception as exc:
            print(f"  Image {i}: {mime} - ❌ failed to load: {exc}")

    # Check material/texture references
    if gltf.textures:
        print(f"\nVerifying {len(gltf.textures)} texture references:")
        for i, texture in enumerate(gltf.textures):
            if texture.source is not None and texture.source < len(gltf.images):
                print(f"  Texture {i}: references image {texture.source} - ✅ valid")
            else:
                print(f"  Texture {i}: invalid image reference {texture.source} - ❌")
    else:
        print(f"\nNo texture references found (might be material-based)")

    if gltf.materials:
        print(f"\nFound {len(gltf.materials)} materials")

    return valid_count == len(gltf.images)


def main():
    """Test compression on sarah.glb."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "example_input", "sarah.glb")
    
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return
    
    # Get original size
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Original file: {input_path}")
    print(f"Original size: {original_size:.2f} MB")
    
    # Verify original file
    print(f"\nVerifying original file textures...")
    verify_glb_textures(input_path)
    
    # Generate output path
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_compressed{ext}"
    
    print(f"\nCompressing textures (PNG → JPEG)...")
    result = _compress_glb_textures(input_path, output_path, jpeg_quality=85)
    
    if result == input_path:
        print("\n❌ Compression failed or no PNG textures found")
        print("File unchanged")
    else:
        compressed_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = original_size - compressed_size
        reduction_pct = (reduction / original_size) * 100
        
        print(f"\n✅ Compression successful")
        print(f"Output file: {output_path}")
        print(f"Compressed size: {compressed_size:.2f} MB")
        print(f"Size reduction: {reduction:.2f} MB ({reduction_pct:.1f}%)")
        
        # Verify compressed file
        print(f"\nVerifying compressed file textures...")
        if verify_glb_textures(output_path):
            print(f"\n✅ All textures valid in compressed file")
        else:
            print(f"\n❌ Some textures invalid in compressed file")


if __name__ == "__main__":
    main()
