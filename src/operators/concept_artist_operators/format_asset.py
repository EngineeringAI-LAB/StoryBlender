import bpy
import mathutils
import math
import os
import tempfile
import base64
import mimetypes
import time
import gc
import warnings
from typing import Optional

from PIL import Image

from pydantic import BaseModel
try:
    from ..llm_completion import completion
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from llm_completion import completion

import logging

# Pre-import pandas to prevent Gradio's lazy import from crashing Blender's OpenGL render
# Gradio's compute_analytics_summary imports pandas on-demand in a background thread,
# which can cause segfaults when concurrent with bpy.ops.render.opengl()
try:
    import pandas  # noqa: F401
except ImportError:
    pass


# === LLM Rotation Classification ===

class RotationClassificationResponse(BaseModel):
    """Response schema for LLM rotation classification."""
    natural_top_view_id: str  # A, B, C, D, E, F
    natural_front_view_id: str  # A, B, C, D, E, F, or "Ambiguous"


class RotationVerificationResponse(BaseModel):
    """Response schema for LLM rotation verification."""
    is_orientation_correct: bool


# Mapping from Image Labels to Blender viewport directions
# Image A: Camera at +X → sees face pointing +X → 'right' view in Blender
# Image B: Camera at -X → sees face pointing -X → 'left' view in Blender
# Image C: Camera at +Y → sees face pointing +Y → 'back' view in Blender
# Image D: Camera at -Y → sees face pointing -Y → 'front' view in Blender
# Image E: Camera at +Z → sees face pointing +Z → 'top' view in Blender
# Image F: Camera at -Z → sees face pointing -Z → 'bottom' view in Blender
IMAGE_LABEL_TO_DIRECTION = {
    "A": "right",
    "B": "left",
    "C": "back",
    "D": "front",
    "E": "top",
    "F": "bottom"
}

# Mapping from Image Labels to vectors (the face normal visible in that image)
IMAGE_LABEL_TO_VECTOR = {
    "A": mathutils.Vector((1, 0, 0)),   # +X
    "B": mathutils.Vector((-1, 0, 0)),  # -X
    "C": mathutils.Vector((0, 1, 0)),   # +Y
    "D": mathutils.Vector((0, -1, 0)),  # -Y
    "E": mathutils.Vector((0, 0, 1)),   # +Z
    "F": mathutils.Vector((0, 0, -1))   # -Z
}

ROTATION_CLASSIFICATION_SYSTEM_PROMPT = """**Role:**
You are a 3D Spatial Reasoning Expert. Your job is to analyze 6 orthographic views of a 3D asset and determine its correct semantic orientation in a standard coordinate system (Z-Up, Y-Forward).

**Input Format:**
1. **Object Description:** A text label describing the object (e.g., "Rug", "Dining Table", "Car").
2. **Images:** 6 images labeled A, B, C, D, E, F. These represent the object viewed from the 6 cardinal directions relative to its *current* local axes.

For the original asset, the face normals are:
    - A: right (+X face)
    - B: left (-X face)
    - C: back (+Y face)
    - D: front (-Y face)
    - E: top (+Z face)
    - F: bottom (-Z face)
But sometimes the original asset is not in the correct orientation, so you need to rotate it to make it correct.

**Your Task:**
Identify two key semantic axes based on the visual evidence and common sense:
1.  **Natural Top:** Which image shows the surface that should be facing **Up** (towards the sky/ceiling)?
    * *Example:* For a rug, the patterned side is Top. For a table, the flat surface is Top. For a car, the roof is Top.
2.  **Natural Front:** Which image shows the side that should be facing **Forward** (towards the viewer/camera)?
    * *Example:* Human character (The side featuring the face and chest), chair (The open edge you approach to sit down, opposite the backrest), mirror (The reflective glass surface intended for viewing reflection), sofa (The long, open side containing the seating cushions, facing outward into the room and opposite the backrest), camera (The side housing the lens or aperture, pointing toward the subject being captured), etc.
    * *Exception:* If the object has rotational symmetry (e.g., a round vase, a round table, a pole) or no clear front, mark this as "Ambiguous".

**Constraints:**
* Do not guess rotation degrees. Only select the Image ID.
* If the object is already correct, the Natural Top might be Image E (+Z) or F (-Z) depending on camera setup, but rely purely on visual content.
* "Ambiguous" is a valid and preferred answer for Front view if the object is symmetric (cylindrical/round)."""

ROTATION_CLASSIFICATION_USER_PROMPT_TEMPLATE = """**Object Description:**
{object_description}

**Images:**
[Image A] [Image B] [Image C]
[Image D] [Image E] [Image F]

**Instructions:**
Analyze the images above.
1. Identify the Image ID (A-F) that shows the **Natural Top** face of the object.
2. Identify the Image ID (A-F) that shows the **Natural Front** face of the object. If the object is rotationally symmetric (like a round table or lamp) or the front is indistinguishable, explicitly state "Ambiguous".

Return your answer in the specified JSON format."""


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


def classify_object_rotation(
    object_description: str,
    image_paths: dict,
    anyllm_api_key: str,
    anyllm_api_base: Optional[str] = None,
    anyllm_provider: str = "gemini",
    vision_model: str = "gemini-3-flash-preview",
    attempted_classify_results: Optional[list] = None
) -> dict:
    """
    Use LLM to classify the correct rotation of a 3D object based on 6 directional images.
    
    Parameters:
    - object_description: Text description of the object (e.g., "A patterned floor rug")
    - image_paths: Dictionary mapping image labels to file paths.
                   Keys must be "A", "B", "C", "D", "E", "F".
                   Example: {"A": "/tmp/viewport_right.png", "B": "/tmp/viewport_left.png", ...}
    - anyllm_api_key: API key for the LLM service
    - anyllm_api_base: Optional API base URL
    - anyllm_provider: LLM provider (default: "gemini")
    - vision_model: LLM model identifier (default: gemini-3-flash-preview)
    
    Returns:
    - Dictionary with keys:
        - 'natural_top_view_id': Image label (A-F) showing the natural top
        - 'natural_front_view_id': Image label (A-F) or "Ambiguous"
        - 'natural_top_vector': mathutils.Vector for the natural top direction
        - 'natural_front_vector': mathutils.Vector for the natural front direction (or None if Ambiguous)
        - 'success': Boolean indicating success
        - 'error': Error message if failed (optional)
    """
    try:
        # Validate image_paths
        required_labels = {"A", "B", "C", "D", "E", "F"}
        if set(image_paths.keys()) != required_labels:
            return {
                "success": False,
                "error": f"image_paths must contain exactly labels A-F. Got: {list(image_paths.keys())}"
            }
        
        # Build user message content with images in order A, B, C, D, E, F
        user_content = []
        
        # Add initial text prompt
        base_prompt = ROTATION_CLASSIFICATION_USER_PROMPT_TEMPLATE.format(
            object_description=object_description
        )
        
        # If there are previous failed attempts, append them to the prompt
        if attempted_classify_results and len(attempted_classify_results) > 0:
            failed_attempts_text = "\n\n**IMPORTANT - Previous Failed Attempts:**\n"
            failed_attempts_text += "The following classifications were already tried and verified to be INCORRECT. Do NOT repeat these answers:\n"
            for i, prev_result in enumerate(attempted_classify_results, 1):
                failed_attempts_text += f"- Attempt {i}: Top={prev_result['top']}, Front={prev_result['front']} (WRONG)\n"
            failed_attempts_text += "\nPlease carefully re-analyze the images and provide a DIFFERENT classification."
            base_prompt += failed_attempts_text
        
        user_content.append({
            "type": "text",
            "text": base_prompt
        })
        
        # Add images with labels
        for label in ["A", "B", "C", "D", "E", "F"]:
            image_path = image_paths[label]
            user_content.append({
                "type": "text",
                "text": f"[Image {label}]"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": _image_path_to_data_url(image_path)
                }
            })
        
        # Call LLM
        response = completion(
            api_key=anyllm_api_key,
            api_base=anyllm_api_base,
            provider=anyllm_provider,
            model=vision_model,
            reasoning_effort="low",
            messages=[
                {
                    "role": "system",
                    "content": ROTATION_CLASSIFICATION_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            response_format=RotationClassificationResponse
        )
        gc.collect()
        
        # Handle generator response in threaded context
        if hasattr(response, '__iter__') and not hasattr(response, 'choices'):
            chunks = list(response)
            if chunks:
                response = chunks[-1]
        
        # Parse response
        response_content = response.choices[0].message.content

        print("Classification Response: ", response_content)
        
        # Parse JSON from response
        import json
        parsed = json.loads(response_content)
        
        natural_top_id = parsed.get("natural_top_view_id", "").upper()
        natural_front_id = parsed.get("natural_front_view_id", "")
        
        # Validate top view ID
        if natural_top_id not in IMAGE_LABEL_TO_VECTOR:
            return {
                "success": False,
                "error": f"Invalid natural_top_view_id: {natural_top_id}"
            }
        
        # Get vectors
        natural_top_vector = IMAGE_LABEL_TO_VECTOR[natural_top_id].copy()
        
        # Handle front view (can be "Ambiguous")
        natural_front_vector = None
        if natural_front_id.upper() in IMAGE_LABEL_TO_VECTOR:
            natural_front_vector = IMAGE_LABEL_TO_VECTOR[natural_front_id.upper()].copy()
            natural_front_id = natural_front_id.upper()
        elif natural_front_id.lower() == "ambiguous":
            natural_front_id = "Ambiguous"
        else:
            return {
                "success": False,
                "error": f"Invalid natural_front_view_id: {natural_front_id}"
            }
        
        return {
            "success": True,
            "natural_top_view_id": natural_top_id,
            "natural_front_view_id": natural_front_id,
            "natural_top_vector": natural_top_vector,
            "natural_front_vector": natural_front_vector
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_rotation_classification_images(max_size: int = 384) -> dict:
    """
    Capture all 6 directional images needed for rotation classification.
    
    Returns a dictionary mapping image labels (A-F) to file paths:
    - A: right (+X face)
    - B: left (-X face)
    - C: back (+Y face)
    - D: front (-Y face)
    - E: top (+Z face)
    - F: bottom (-Z face)
    
    Parameters:
    - max_size: Maximum size in pixels for the largest dimension (default: 384)
    
    Returns:
    - Dictionary with keys "A" through "F" mapping to image file paths
    - Or {"success": False, "error": "..."} if failed
    """
    # Map labels to Blender viewport directions
    label_to_direction = {
        "A": "right",
        "B": "left", 
        "C": "back",
        "D": "front",
        "E": "top",
        "F": "bottom"
    }
    
    # Get all directions
    directions = list(label_to_direction.values())
    results = get_object_images_by_viewport_directions(directions, max_size=max_size)
    
    # Build result dictionary
    image_paths = {}
    errors = []
    
    for label, direction in label_to_direction.items():
        # Find the result for this direction
        for result in results:
            if result["direction"].lower() == direction:
                if result.get("success"):
                    image_paths[label] = result["filepath"]
                else:
                    errors.append(f"{label} ({direction}): {result.get('error', 'Unknown error')}")
                break
    
    if errors:
        return {
            "success": False,
            "error": "; ".join(errors)
        }
    
    return image_paths


def apply_rotation_correction(
    natural_top_vector: mathutils.Vector,
    natural_front_vector: Optional[mathutils.Vector],
    target_objects: list
) -> dict:
    """
    Apply rotation correction to align objects based on LLM classification results.
    
    This function computes and applies rotations to align:
    1. The natural top of the object to Global +Z (up)
    2. The natural front of the object to Global -Y (forward/front in Blender)
    
    If natural_front_vector is None (Ambiguous), applies the "Long Edge Rule":
    After aligning top, if bounding box Y > X, rotate 90° on Z to make longer edge horizontal.
    
    Parameters:
    - natural_top_vector: Vector indicating current direction of the natural top face
    - natural_front_vector: Vector indicating current direction of the natural front face, or None if ambiguous
    - target_objects: List of Blender objects to apply the rotation to (typically root objects)
    
    Returns:
    - Dictionary with:
        - 'success': Boolean indicating success
        - 'rotation_applied': Description of the rotation applied
        - 'error': Error message if failed (optional)
    """
    try:
        if not target_objects:
            return {"success": False, "error": "No target objects provided"}
        
        # Target vectors
        global_up = mathutils.Vector((0, 0, 1))      # +Z
        global_front = mathutils.Vector((0, -1, 0))  # -Y (front in Blender)
        
        rotation_descriptions = []
        
        # === Step 1: Align natural top to Global +Z ===
        # Calculate rotation from natural_top_vector to global_up
        if natural_top_vector.normalized() != global_up:
            # Get rotation quaternion from natural_top to global_up
            rotation_to_up = natural_top_vector.rotation_difference(global_up)
            
            # Apply rotation to all target objects
            for obj in target_objects:
                # Convert current rotation to quaternion, apply new rotation, convert back
                obj.rotation_mode = 'QUATERNION'
                obj.rotation_quaternion = rotation_to_up @ obj.rotation_quaternion
            
            rotation_descriptions.append(f"Aligned top ({natural_top_vector}) to +Z")
        else:
            rotation_descriptions.append("Top already aligned to +Z")
        
        # Update scene to apply first rotation
        bpy.context.view_layer.update()
        
        # === Step 2: Align natural front to Global -Y ===
        if natural_front_vector is not None:
            # Transform the front vector by the first rotation
            rotation_to_up = natural_top_vector.rotation_difference(global_up)
            transformed_front = rotation_to_up @ natural_front_vector
            
            # Project both vectors onto XY plane (we only rotate around Z now)
            # Use 2D vectors for angle_signed (Blender's angle_signed only works on 2D vectors)
            transformed_front_2d = mathutils.Vector((transformed_front.x, transformed_front.y))
            global_front_2d = mathutils.Vector((global_front.x, global_front.y))
            
            if transformed_front_2d.length > 0.001:  # Avoid zero-length vector
                transformed_front_2d.normalize()
                global_front_2d.normalize()
                # Calculate signed angle between 2D vectors (positive = counter-clockwise)
                angle = transformed_front_2d.angle_signed(global_front_2d)
                
                if abs(angle) > 0.01:  # Only rotate if angle is significant
                    # Create rotation around Z axis
                    z_rotation = mathutils.Quaternion((0, 0, 1), angle)
                    
                    for obj in target_objects:
                        obj.rotation_quaternion = z_rotation @ obj.rotation_quaternion
                    
                    rotation_descriptions.append(f"Aligned front to -Y (rotated {math.degrees(angle):.1f}° around Z)")
                else:
                    rotation_descriptions.append("Front already aligned to -Y")
            else:
                rotation_descriptions.append("Front vector perpendicular to XY plane, skipped front alignment")
        else:
            # === Ambiguous front: Apply Long Edge Rule ===
            # After aligning top, check bounding box dimensions
            # If Y > X, rotate 90° around Z to make longer edge horizontal (along X)
            
            bpy.context.view_layer.update()
            
            # Compute combined bounding box of all target objects
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')
            
            for obj in target_objects:
                if obj.type == 'MESH':
                    for corner in obj.bound_box:
                        co = obj.matrix_world @ mathutils.Vector(corner)
                        min_x = min(min_x, co.x); max_x = max(max_x, co.x)
                        min_y = min(min_y, co.y); max_y = max(max_y, co.y)
            
            size_x = max_x - min_x
            size_y = max_y - min_y
            
            if size_y > size_x:
                # Rotate 90° around Z to make longer edge along X
                z_rotation = mathutils.Quaternion((0, 0, 1), math.radians(90))
                
                for obj in target_objects:
                    obj.rotation_quaternion = z_rotation @ obj.rotation_quaternion
                
                rotation_descriptions.append(f"Long Edge Rule: rotated 90° around Z (Y={size_y:.3f} > X={size_x:.3f})")
            else:
                rotation_descriptions.append(f"Long Edge Rule: no rotation needed (X={size_x:.3f} >= Y={size_y:.3f})")
        
        # Update scene
        bpy.context.view_layer.update()
        
        return {
            "success": True,
            "rotation_applied": "; ".join(rotation_descriptions)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


ROTATION_VERIFICATION_SYSTEM_PROMPT = """**Role:**
You are a 3D Orientation Verification Expert. Your job is to verify whether a 3D model has the correct semantic orientation in a standard coordinate system (Z-Up, Y-Forward).

**Context:**
You are given a 3/4 view (three-quarter perspective) image of a 3D object that has been rotated to what is believed to be the correct orientation.

**Correct Orientation Criteria:**
1. **Up Direction (+Z):** The natural "top" of the object should be facing upward (towards the sky/ceiling).
   - For furniture: the usable surface should be up (table top, seat surface, etc.)
   - For characters/animals: the head should be up
   - For vehicles: the roof/top should be up
   - For rugs/mats: the patterned/decorated side should be up
   - For books: the cover should be up

2. **Ground Placement:** The object should appear to be resting naturally on a ground plane.
   - Objects should not appear to be floating upside-down or sideways

**Your Task:**
Examine the 3/4 view image and determine if the object appears to have the correct orientation based on the criteria above.

**Important Notes:**
- Focus on the up/down orientation (Z-axis alignment)
- For cuboid objects, since many of them have ambiguous front, we define the front of the object as the side with the longest dimension between width and depth. For example, the side of the car is defined as the front for a car model, the long edge of the table is defined as the front for a table model.
- For objects that have a clear distinction between the front and back, make sure you can see its actual front. For example, human character (The side featuring the face and chest), chair (The open edge you approach to sit down, opposite the backrest), mirror (The reflective glass surface intended for viewing reflection), sofa (The long, open side containing the seating cushions, facing outward into the room and opposite the backrest), camera (The side housing the lens or aperture, pointing toward the subject being captured), etc.
- If the object appears naturally oriented (top facing up, front facing forward, resting on ground), mark it as correct"""

ROTATION_VERIFICATION_USER_PROMPT_TEMPLATE = """**Object Description:**
{object_description}

**Image:**
A 3/4 view (three-quarter perspective) of the 3D model after rotation correction.

**Instructions:**
Examine the image and determine:
1. Is the natural "top" of the object facing upward (towards the sky)?
2. Does the object appear to be resting naturally on the ground?

If both conditions are met, the orientation is correct. If the object appears upside-down, sideways, or otherwise incorrectly oriented, it is not correct.

Return your answer in the specified JSON format with:
- is_orientation_correct: true if the orientation is correct, false otherwise"""


def verify_rotation_orientation(
    object_description: str,
    image_path: str,
    anyllm_api_key: str,
    anyllm_api_base: Optional[str] = None,
    anyllm_provider: str = "gemini",
    vision_model: str = "gemini-2.5-flash-preview-05-20",
) -> dict:
    """
    Use LLM to verify if a 3D object's orientation is correct after rotation correction.
    
    Parameters:
    - object_description: Text description of the object (e.g., "A patterned floor rug")
    - image_path: Path to the 3/4 view image of the object
    - anyllm_api_key: API key for the LLM service
    - anyllm_api_base: Optional API base URL
    - anyllm_provider: LLM provider (default: "gemini")
    - vision_model: LLM model identifier (default: gemini-2.5-flash-preview-05-20)
    
    Returns:
    - Dictionary with keys:
        - 'is_orientation_correct': Boolean indicating if orientation is correct
        - 'success': Boolean indicating API call success
        - 'error': Error message if failed (optional)
    """
    try:
        if not os.path.isfile(image_path):
            return {
                "success": False,
                "error": f"Image file not found: {image_path}"
            }
        
        # Build user message content with image
        user_content = [
            {
                "type": "text",
                "text": ROTATION_VERIFICATION_USER_PROMPT_TEMPLATE.format(
                    object_description=object_description
                )
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_path_to_data_url(image_path)
                }
            }
        ]
        
        # Call LLM
        response = completion(
            api_key=anyllm_api_key,
            api_base=anyllm_api_base,
            provider=anyllm_provider,
            model=vision_model,
            reasoning_effort="low",
            messages=[
                {
                    "role": "system",
                    "content": ROTATION_VERIFICATION_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            response_format=RotationVerificationResponse
        )
        gc.collect()
        
        # Handle generator response in threaded context
        if hasattr(response, '__iter__') and not hasattr(response, 'choices'):
            chunks = list(response)
            if chunks:
                response = chunks[-1]
        
        # Parse response
        response_content = response.choices[0].message.content

        print("Verification Response: ", response_content)
        
        import json
        parsed = json.loads(response_content)
        
        is_correct = parsed.get("is_orientation_correct", False)

        return {
            "success": True,
            "is_orientation_correct": is_correct,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_object_images_by_viewport_directions(directions: list, max_size: int = 384) -> list:
    """
    Capture images of the active object from specified viewport directions (world-space).
    
    Unlike get_object_image which aligns view to object's local axes, this function
    uses world-space directions (e.g., front = -Y direction in world space).
    
    Parameters:
    - directions: List of directions to capture. Valid options: 'front', 'back', 'top', 'bottom', 'left', 'right'
    - max_size: Maximum size in pixels for the largest dimension of the image (default: 384)
    
    Returns:
    - List of dictionaries with keys:
        - 'direction': The direction name
        - 'filepath': Path to the saved image
        - 'success': Boolean indicating success
        - 'error': Error message if failed (optional)
    """
    
    # Dictionary to map direction strings to Blender's view axis types
    direction_map = {
        'front': 'FRONT',
        'back': 'BACK',
        'top': 'TOP',
        'bottom': 'BOTTOM',
        'left': 'LEFT',
        'right': 'RIGHT'
    }
    
    # Get the active object
    active_object = bpy.context.active_object
    if active_object is None:
        return [{"direction": d, "success": False, "error": "No object is currently selected/active"} for d in directions]
    
    # Make sure the object is selected for framing
    if not active_object.select_get():
        active_object.select_set(True)
    
    results = []
    
    for direction in directions:
        direction_lower = direction.lower()
        
        # Validate direction
        if direction_lower not in direction_map:
            results.append({
                "direction": direction,
                "success": False,
                "error": f"Invalid direction '{direction}'. Valid options are: {', '.join(direction_map.keys())}"
            })
            continue
        
        # Find 3D viewport and set the view
        viewport_set = False
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        with bpy.context.temp_override(
                            window=bpy.context.window,
                            screen=bpy.context.screen,
                            area=area,
                            region=region,
                            space_data=area.spaces.active
                        ):
                            # Set view to world-space direction (align_active=False means world space)
                            bpy.ops.view3d.view_axis(type=direction_map[direction_lower], align_active=False)
                            
                            # Frame the selected object
                            bpy.ops.view3d.view_selected()
                            
                            # Zoom out a bit for better framing
                            # for _ in range(2):
                            #     bpy.ops.view3d.zoom(delta=-1)
                        
                        viewport_set = True
                        break
                if viewport_set:
                    break
        
        if not viewport_set:
            results.append({
                "direction": direction,
                "success": False,
                "error": "No 3D viewport found"
            })
            continue
        
        # Take screenshot
        screenshot_result = _capture_viewport_screenshot(max_size=max_size, direction=direction_lower)
        
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


def capture_thumbnail_three_quarter_view(filepath: str, max_size: int = 512) -> dict:
    """
    Capture a 3/4 view (three-quarter view) thumbnail of the active object.
    
    Parameters:
    - filepath: Absolute path where to save the thumbnail (e.g., /path/to/model_id.png)
    - max_size: Maximum size in pixels for the largest dimension (default: 512)
    
    Returns:
    - Dictionary with success status and filepath or error
    """
    try:
        # Find the active 3D viewport
        area = None
        space = None
        for a in bpy.context.screen.areas:
            if a.type == 'VIEW_3D':
                area = a
                for s in area.spaces:
                    if s.type == 'VIEW_3D':
                        space = s
                        break
                break
        
        if not area or not space:
            return {"success": False, "error": "No 3D viewport found"}
        
        # Set material shading (UI hiding is done once in format_assets)
        try:
            space.shading.type = 'MATERIAL'
        except Exception:
            pass
        
        # Find WINDOW region
        region = None
        for r in area.regions:
            if r.type == 'WINDOW':
                region = r
                break
        
        if not region:
            return {"success": False, "error": "No WINDOW region found"}
        
        # Set up 3/4 view by directly setting view rotation
        # This is more reliable than view_orbit which can produce unexpected results
        region_3d = space.region_3d
        
        # Create rotation for 3/4 view: looking from front-right and slightly above
        # Euler angles: X=75° (orbit down), Z=25° (rotate right)
        view_rotation = mathutils.Euler((math.radians(75), 0, math.radians(25)), 'XYZ').to_quaternion()
        region_3d.view_rotation = view_rotation
        region_3d.view_perspective = 'PERSP'
        
        # Use full context override including window/screen for timer context compatibility
        with bpy.context.temp_override(
            window=bpy.context.window,
            screen=bpy.context.screen,
            area=area,
            region=region,
            space_data=space
        ):
            # Frame the selected object
            bpy.ops.view3d.view_selected()
        
        # Store original render settings
        scene = bpy.context.scene
        original_filepath = scene.render.filepath
        original_format = scene.render.image_settings.file_format
        original_res_x = scene.render.resolution_x
        original_res_y = scene.render.resolution_y
        original_res_percentage = scene.render.resolution_percentage
        original_film_transparent = scene.render.film_transparent
        
        try:
            # Set up render settings
            scene.render.filepath = filepath
            scene.render.image_settings.file_format = 'PNG'
            scene.render.film_transparent = False  # Solid background
            
            # Render at 2x resolution for supersampling, then downsample for sharper results
            supersample_factor = 3
            vp_width, vp_height = region.width, region.height
            if max(vp_width, vp_height) > max_size:
                scale_factor = max_size / max(vp_width, vp_height)
                target_width = int(vp_width * scale_factor)
                target_height = int(vp_height * scale_factor)
            else:
                target_width = vp_width
                target_height = vp_height
            
            # Render at 2x the target size
            scene.render.resolution_x = target_width * supersample_factor
            scene.render.resolution_y = target_height * supersample_factor
            scene.render.resolution_percentage = 100
            
            # Disable GC during OpenGL render to prevent race condition with Gradio's asyncio thread
            # Only re-enable if we were the ones who disabled it (avoid premature re-enable when called from format_asset)
            gc_was_enabled = gc.isenabled()
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
                if gc_was_enabled:
                    gc.enable()
            
            # Allow Blender to stabilize
            bpy.context.view_layer.update()
            time.sleep(0.3)
            
            # Downsample to target size with high-quality LANCZOS filter
            img = Image.open(filepath)
            img_downsampled = img.resize((target_width, target_height), Image.LANCZOS)
            img_downsampled.save(filepath)
            img.close()
            img_downsampled.close()
            
        finally:
            # Restore render settings
            scene.render.filepath = original_filepath
            scene.render.image_settings.file_format = original_format
            scene.render.resolution_x = original_res_x
            scene.render.resolution_y = original_res_y
            scene.render.resolution_percentage = original_res_percentage
            scene.render.film_transparent = original_film_transparent
        
        return {
            "success": True,
            "filepath": filepath
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def _capture_viewport_screenshot(max_size: int, direction: str) -> dict:
    """
    Capture a screenshot of the current 3D viewport.
    
    Parameters:
    - max_size: Maximum size in pixels for the largest dimension
    - direction: Direction name for the filename
    
    Returns:
    - Dictionary with success status and filepath or error
    """
    try:
        # Generate filepath in temp directory
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, f"viewport_{direction}.png")
        
        # Find the active 3D viewport (UI hiding is done once in format_assets)
        area = None
        for a in bpy.context.screen.areas:
            if a.type == 'VIEW_3D':
                area = a
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        # Set material shading
                        try:
                            space.shading.type = 'MATERIAL'
                        except Exception:
                            pass
                        break
                break
        
        if not area:
            return {"success": False, "error": "No 3D viewport found"}
        
        # Store original render settings
        scene = bpy.context.scene
        original_filepath = scene.render.filepath
        original_format = scene.render.image_settings.file_format
        original_res_x = scene.render.resolution_x
        original_res_y = scene.render.resolution_y
        original_res_percentage = scene.render.resolution_percentage
        original_engine = scene.render.engine
        
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
            
            if region is None:
                return {"success": False, "error": "No WINDOW region found in 3D viewport"}
            
            # Render at 2x resolution for supersampling, then downsample for sharper results
            supersample_factor = 3
            vp_width, vp_height = region.width, region.height
            if max(vp_width, vp_height) > max_size:
                scale_factor = max_size / max(vp_width, vp_height)
                target_width = int(vp_width * scale_factor)
                target_height = int(vp_height * scale_factor)
            else:
                target_width = vp_width
                target_height = vp_height
            
            # Render at 2x the target size
            scene.render.resolution_x = target_width * supersample_factor
            scene.render.resolution_y = target_height * supersample_factor
            scene.render.resolution_percentage = 100
            
            # Disable GC during OpenGL render to prevent race condition with Gradio's asyncio thread
            # Only re-enable if we were the ones who disabled it (avoid premature re-enable when called from format_asset)
            gc_was_enabled = gc.isenabled()
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
                if gc_was_enabled:
                    gc.enable()
            
            # Allow Blender to stabilize after OpenGL render to prevent crashes
            # when multiple renders are done in quick succession
            bpy.context.view_layer.update()
            time.sleep(0.3)
            
            # Downsample to target size with high-quality LANCZOS filter
            img = Image.open(filepath)
            img_downsampled = img.resize((target_width, target_height), Image.LANCZOS)
            img_downsampled.save(filepath)
            img.close()
            img_downsampled.close()
            
            width = target_width
            height = target_height
            
        finally:
            # Restore original render settings
            scene.render.filepath = original_filepath
            scene.render.image_settings.file_format = original_format
            scene.render.resolution_x = original_res_x
            scene.render.resolution_y = original_res_y
            scene.render.resolution_percentage = original_res_percentage
            scene.render.engine = original_engine
        
        return {
            "success": True,
            "width": width,
            "height": height,
            "filepath": filepath
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}


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


def format_asset(
    model_info: dict,
    export_dir: str = None,
    correct_rotation: bool = True,
    anyllm_api_key: str = None,
    anyllm_api_base: str = None,
    anyllm_provider: str = "gemini",
    vision_model: str = "gemini-3-flash-preview",
):
    """
    Format a 3D model by importing, correcting rotation, setting origin to ground center,
    renaming, and exporting. This function does NOT resize the model.
    
    Parameters:
    - model_info: Dictionary containing model information with keys:
        - 'asset_id': Identifier for the model
        - 'description': Text description of the model (used for rotation classification)
        - 'main_file_path': Absolute path to the model file (GLB/GLTF/FBX/OBJ)
        - 'tags': List of tags indicating model source (polyhaven, sketchfab, hunyuan3d, meshy)
    - export_dir: Directory to export the formatted model
    - correct_rotation: Whether to use LLM to correct model rotation alignment (default: True)
    - anyllm_api_key: API key for LLM service (required if correct_rotation=True)
    - anyllm_api_base: Optional API base URL for LLM service
    - anyllm_provider: LLM provider (default: "gemini")
    - vision_model: LLM model identifier for vision tasks (default: gemini-3-flash-preview)
    
    Returns:
    - Dictionary with:
        - 'export_path': Path to the exported model
        - 'rotation_correction': Description of rotation correction applied (if correct_rotation=True)
    
    Raises:
    - ValueError: If model_info is missing required fields or has invalid data
    - FileNotFoundError: If the model file is not found
    """
    # Disable GC and add sync points for stability with Gradio
    gc.disable()
    try:
        # Brief pause before starting to let any pending asyncio operations complete
        # time.sleep(3.5)
        
        result = _format_asset_core(
            model_info=model_info,
            export_dir=export_dir,
            correct_rotation=correct_rotation,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            anyllm_provider=anyllm_provider,
            vision_model=vision_model,
        )
        
        # Brief pause after completion to let asyncio stabilize before response
        # time.sleep(3.5)
        
        return result
    finally:
        gc.enable()


def _format_asset_core(
    model_info: dict,
    export_dir: str = None,
    correct_rotation: bool = True,
    anyllm_api_key: str = None,
    anyllm_api_base: str = None,
    anyllm_provider: str = "gemini",
    vision_model: str = "gemini-3-flash-preview",
):
    """Core implementation of format_asset. Called with extended GIL hold time."""
    
    # === Step 1: Switch to temporary scene ===
    # Save the original scene to switch back later
    original_scene = bpy.context.window.scene
    original_scene_name = original_scene.name
    
    # Switch to or create a temporary scene for formatting
    _switch_or_create_scene("Temporary_scene")
    
    # Extract model_id for error messages
    model_id = model_info.get("asset_id", "unknown")
    
    # Get the model file path
    main_file_path = model_info.get("main_file_path")
    if not main_file_path:
        raise ValueError(f"{model_id}: No main_file_path specified")
    
    # Validate tags
    model_tags = model_info.get("tags")
    if not model_tags:
        raise ValueError(f"{model_id}: No tags specified")
    
    # Determine model source from tags
    model_source = None
    if "polyhaven" in model_tags:
        model_source = "polyhaven"
    if "sketchfab" in model_tags:
        model_source = "sketchfab"
    if "hunyuan3d" in model_tags:
        model_source = "hunyuan3d"
    if "meshy" in model_tags:
        model_source = "meshy"
    if not model_source:
        raise ValueError(f"{model_id}: No valid model source found in tags")
    
    path_to_original_model = main_file_path
    
    # Check if file exists
    if not os.path.exists(path_to_original_model):
        raise FileNotFoundError(f"{model_id}: Model file not found at {path_to_original_model}")
    
    # Extract base filename without extension for naming
    original_filename = os.path.basename(path_to_original_model)
    base_name = os.path.splitext(original_filename)[0]
    
    # === Step 2: Empty the current scene only ===
    # Delete all objects in the current scene
    current_scene = bpy.context.scene
    for obj in list(current_scene.collection.all_objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # Delete only child collections of the current scene (not global collections)
    for coll in list(current_scene.collection.children):
        try:
            bpy.data.collections.remove(coll)
        except Exception:
            pass
    
    # === Step 3: Import the original model ===
    if not os.path.exists(path_to_original_model):
        raise FileNotFoundError(f"Model file not found: {path_to_original_model}")
    
    time.sleep(0.2)
    bpy.ops.object.select_all(action='DESELECT')
    time.sleep(0.2)
    
    # Import based on file extension
    ext = os.path.splitext(path_to_original_model)[1].lower()
    if ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=path_to_original_model)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=path_to_original_model)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=path_to_original_model)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # === Step 4: Merge geometry FIRST (before rotation correction) ===
    # This ensures multi-part models rotate as a single unit, not each part separately
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
    
    # Unparent the merged object from any hierarchy (critical for deeply nested models like Sketchfab)
    # This must happen BEFORE rotation correction to prevent crashes when accessing matrix_world
    if merged_obj.parent is not None:
        world_matrix = merged_obj.matrix_world.copy()
        merged_obj.parent = None
        merged_obj.matrix_world = world_matrix
    
    # Update scene after merge
    bpy.context.view_layer.update()
    
    # === Step 5: Correct rotation alignment (LLM-based or Long Edge Rule for polyhaven) ===
    # Now operates on the single merged object, ensuring all parts rotate together
    rotation_correction_result = None
    if correct_rotation:
        if model_source == "polyhaven":
            # Polyhaven models are always correctly rotated (up/down is correct)
            # but front is ambiguous, so we apply Long Edge Rule only
            # print(f"Polyhaven model detected - skipping LLM rotation, applying Long Edge Rule...")
            
            # Compute current bounding box to determine X and Y dimensions
            bpy.context.view_layer.update()
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')
            
            for corner in merged_obj.bound_box:
                co = merged_obj.matrix_world @ mathutils.Vector(corner)
                min_x = min(min_x, co.x); max_x = max(max_x, co.x)
                min_y = min(min_y, co.y); max_y = max(max_y, co.y)
            
            dim_x = max_x - min_x
            dim_y = max_y - min_y
            
            # Long Edge Rule: If Y > X, rotate 90° around Z to make longer edge horizontal (along X)
            if dim_y > dim_x:
                import math
                # Rotate 90 degrees around Z axis
                rotation_angle = math.radians(90)
                rotation_matrix = mathutils.Matrix.Rotation(rotation_angle, 4, 'Z')
                merged_obj.matrix_world = rotation_matrix @ merged_obj.matrix_world
                bpy.context.view_layer.update()
                rotation_correction_result = "Long Edge Rule: Rotated 90° around Z (Y was longer than X)"
                # print(f"Applied Long Edge Rule: Rotated 90° around Z (Y={dim_y:.3f} > X={dim_x:.3f})")
            else:
                rotation_correction_result = "Long Edge Rule: No rotation needed (X >= Y)"
                # print(f"Long Edge Rule: No rotation needed (X={dim_x:.3f} >= Y={dim_y:.3f})")
        elif not anyllm_api_key:
            print(f"Warning: correct_rotation=True but no anyllm_api_key provided. Skipping rotation correction.")
        else:
            # Get object description from model_info
            object_description = model_info.get("description", model_info.get("asset_id", "3D object"))
            
            # Select the merged object for framing in screenshots
            time.sleep(0.2)
            bpy.ops.object.select_all(action='DESELECT')
            time.sleep(0.2)
            merged_obj.select_set(True)
            bpy.context.view_layer.objects.active = merged_obj
            
            # Capture 6 directional images
            # print(f"Capturing 6 directional images for rotation classification...")
            image_paths = get_rotation_classification_images(max_size=384)
            
            if isinstance(image_paths, dict) and image_paths.get("success") == False:
                print(f"Warning: Failed to capture images for rotation classification: {image_paths.get('error')}")
            else:
                # === Initial orientation check: verify if model is already correctly oriented ===
                print(f"Checking if model is already correctly oriented...")
                initial_check_image_path = os.path.join(tempfile.gettempdir(), "rotation_initial_check.png")
                initial_thumbnail_result = capture_thumbnail_three_quarter_view(
                    filepath=initial_check_image_path,
                    max_size=768
                )
                
                skip_rotation_correction = False
                if initial_thumbnail_result.get("success"):
                    initial_verification_result = verify_rotation_orientation(
                        object_description=object_description,
                        image_path=initial_check_image_path,
                        anyllm_api_key=anyllm_api_key,
                        anyllm_api_base=anyllm_api_base,
                        anyllm_provider=anyllm_provider,
                        vision_model=vision_model,
                    )
                    
                    if initial_verification_result.get("success") and initial_verification_result.get("is_orientation_correct"):
                        print(f"Model is already correctly oriented. Skipping rotation correction.")
                        rotation_correction_result = "Already correctly oriented (verified by initial check)"
                        skip_rotation_correction = True
                    else:
                        print(f"Model orientation needs correction. Proceeding with rotation classification...")
                else:
                    print(f"Warning: Failed to capture initial check thumbnail: {initial_thumbnail_result.get('error')}")
                    print(f"Proceeding with rotation classification...")
                
                if skip_rotation_correction:
                    pass  # Skip the rotation correction loop
                else:
                    # Rotation correction with verification loop
                    max_rotation_attempts = 5
                    rotation_verified = False
                    attempted_classify_results = []  # Track previous failed classification results
                    
                    for attempt in range(max_rotation_attempts):
                        print(f"Rotation correction attempt {attempt + 1}/{max_rotation_attempts}")
                        
                        # Recapture images for each attempt (except first which already has them)
                        if attempt > 0:
                            # print(f"Re-capturing 6 directional images for rotation classification...")
                            image_paths = get_rotation_classification_images(max_size=512)
                            if isinstance(image_paths, dict) and image_paths.get("success") == False:
                                print(f"Warning: Failed to capture images: {image_paths.get('error')}")
                                break
                        
                        # Call LLM to classify rotation
                        print(f"Calling LLM to classify rotation for: {object_description}")
                        classification_result = classify_object_rotation(
                            object_description=object_description,
                            image_paths=image_paths,
                            anyllm_api_key=anyllm_api_key,
                            anyllm_api_base=anyllm_api_base,
                            anyllm_provider=anyllm_provider,
                            vision_model=vision_model,
                            attempted_classify_results=attempted_classify_results
                        )
                        
                        if not classification_result.get("success"):
                            print(f"Warning: LLM classification failed: {classification_result.get('error')}")
                            break
                        
                        # print(f"LLM classification: top={classification_result['natural_top_view_id']}, front={classification_result['natural_front_view_id']}")
                        
                        # Save current rotation state before applying correction (for potential revert)
                        merged_obj.rotation_mode = 'QUATERNION'
                        saved_rotation_quaternion = merged_obj.rotation_quaternion.copy()
                        
                        # Apply rotation correction to the single merged object
                        rotation_result = apply_rotation_correction(
                            natural_top_vector=classification_result["natural_top_vector"],
                            natural_front_vector=classification_result["natural_front_vector"],
                            target_objects=[merged_obj]
                        )
                        
                        if not rotation_result.get("success"):
                            print(f"Warning: Failed to apply rotation correction: {rotation_result.get('error')}")
                            break
                        
                        rotation_correction_result = rotation_result["rotation_applied"]
                        # print(f"Rotation correction applied: {rotation_correction_result}")
                        
                        # Update scene after rotation
                        bpy.context.view_layer.update()
                        
                        # Verify rotation with 3/4 view image
                        # print(f"Verifying rotation orientation...")
                        verification_image_path = os.path.join(tempfile.gettempdir(), f"rotation_verify_{attempt}.png")
                        thumbnail_result = capture_thumbnail_three_quarter_view(
                            filepath=verification_image_path,
                            max_size=768
                        )
                        
                        if not thumbnail_result.get("success"):
                            print(f"Warning: Failed to capture verification thumbnail: {thumbnail_result.get('error')}")
                            # Continue anyway, consider it verified
                            rotation_verified = True
                            break
                        
                        verification_result = verify_rotation_orientation(
                            object_description=object_description,
                            image_path=verification_image_path,
                            anyllm_api_key=anyllm_api_key,
                            anyllm_api_base=anyllm_api_base,
                            anyllm_provider=anyllm_provider,
                            vision_model=vision_model,
                        )
                        
                        if not verification_result.get("success"):
                            print(f"Warning: Verification API call failed: {verification_result.get('error')}")
                            # Continue anyway, consider it verified
                            rotation_verified = True
                            break
                        
                        if verification_result.get("is_orientation_correct"):
                            # print(f"Rotation verified as correct. Reasoning: {verification_result.get('reasoning', '')}")
                            rotation_verified = True
                            break
                        else:
                            print("Rotation verification failed.")
                            # Record this failed classification result
                            attempted_classify_results.append({
                                "top": classification_result['natural_top_view_id'],
                                "front": classification_result['natural_front_view_id']
                            })
                            if attempt < max_rotation_attempts - 1:
                                # Revert the rotation before retrying
                                # print(f"Reverting rotation and retrying...")
                                merged_obj.rotation_mode = 'QUATERNION'
                                merged_obj.rotation_quaternion = saved_rotation_quaternion
                                bpy.context.view_layer.update()
                            else:
                                print(f"Max rotation attempts reached. Proceeding with current orientation.")
                    
                    if not rotation_verified:
                        print(f"Rotation verification loop completed without confirmation.")
    
    # === Step 6: Use merged object as final object (no resize) ===
    final_obj = merged_obj
    
    # === Step 7: Set origin point to bottom center ===
    # Make sure object is selected and active
    for o in bpy.context.selected_objects:
        o.select_set(False)
    final_obj.select_set(True)
    bpy.context.view_layer.objects.active = final_obj
    
    # Apply transforms to bake into geometry before origin adjustment
    try:
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    except Exception:
        pass
    
    # Update scene to ensure transformations are applied
    bpy.context.view_layer.update()
    
    # Compute bounding box directly from mesh vertices (more reliable than bound_box for complex models)
    mesh = final_obj.data
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    for vert in mesh.vertices:
        # Transform vertex to world space
        co = final_obj.matrix_world @ vert.co
        min_x = min(min_x, co.x); max_x = max(max_x, co.x)
        min_y = min(min_y, co.y); max_y = max(max_y, co.y)
        min_z = min(min_z, co.z); max_z = max(max_z, co.z)
    
    # Bottom-center of bounding box in world space
    bottom_center_world = mathutils.Vector(((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, min_z))
    
    # Convert to object local space - this is the offset we need to apply to vertices
    offset_local = final_obj.matrix_world.inverted() @ bottom_center_world
    
    # Manually offset all mesh vertices to move origin to bottom-center
    for vert in mesh.vertices:
        vert.co -= offset_local
    
    # Update mesh
    mesh.update()
    
    # Now adjust object location to compensate and place at world origin
    final_obj.location = bottom_center_world
    
    # Update scene
    bpy.context.view_layer.update()
    
    # Now set location to (0,0,0) - the bottom-center should be at world origin
    final_obj.location = (0.0, 0.0, 0.0)
    
    # Final update
    bpy.context.view_layer.update()
    
    # === Step 8: Rename object, mesh, and material ===
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
        # Create new material
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            mat = bpy.data.materials.new(name=mat_name)
        if len(final_obj.data.materials) == 0:
            final_obj.data.materials.append(mat)
        else:
            final_obj.data.materials[0] = mat

    
    # === Step 9: Reset Transform ===
    # Prefer the direct reference to the just-renamed object. Looking up by name
    # via bpy.data.objects.get(obj_name) is unsafe because if another object with
    # the same base name already exists in bpy.data (e.g. leftover in another
    # scene), Blender silently appends ".001" to final_obj, and the name lookup
    # would return the unrelated object that may not be in the current view layer
    # (causing "can't be selected because it is not in View Layer" errors).
    target_obj = final_obj
    # Sanity check: ensure target_obj is actually linked to the current view layer.
    view_layer_objs = set(bpy.context.view_layer.objects)
    if target_obj not in view_layer_objs:
        # Fallback: try name lookup, but only accept it if it's in the view layer.
        candidate = bpy.data.objects.get(obj_name)
        if candidate is not None and candidate in view_layer_objs:
            target_obj = candidate
        else:
            # Last-resort: link target_obj into the current scene collection.
            current_scene = bpy.context.scene
            if target_obj.name not in current_scene.collection.objects:
                try:
                    current_scene.collection.objects.link(target_obj)
                except RuntimeError:
                    pass
            bpy.context.view_layer.update()
    
    # Unparent target_obj if it has a parent (keep transform)
    if target_obj.parent is not None:
        # Store world matrix before unparenting
        world_matrix = target_obj.matrix_world.copy()
        target_obj.parent = None
        target_obj.matrix_world = world_matrix
    
    # Delete all other objects in the CURRENT scene only (not globally)
    current_scene = bpy.context.scene
    for obj in list(current_scene.collection.all_objects):
        if obj != target_obj:
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Select only the target object
    for o in bpy.context.selected_objects:
        o.select_set(False)
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj
    
    # Apply rotation and scale to mesh data (keeps visual appearance)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    
    # Reset location to origin
    target_obj.location = (0, 0, 0)
    
    # Rotation and scale are now (0,0,0) and (1,1,1) after apply
    
    # === Step 10: Re-verify and fix origin at bottom center ===
    # After all transforms applied, ensure origin is truly at bottom center
    # This is crucial for complex hierarchical models where earlier origin setting may drift
    bpy.context.view_layer.update()
    
    mesh = target_obj.data
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    for vert in mesh.vertices:
        co = target_obj.matrix_world @ vert.co
        min_x = min(min_x, co.x); max_x = max(max_x, co.x)
        min_y = min(min_y, co.y); max_y = max(max_y, co.y)
        min_z = min(min_z, co.z); max_z = max(max_z, co.z)
    
    # Check if origin is at bottom center (within tolerance)
    expected_origin = mathutils.Vector(((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, min_z))
    current_origin = target_obj.matrix_world.translation
    
    if (expected_origin - current_origin).length > 0.0001:
        # Origin drifted, fix it
        offset_local = target_obj.matrix_world.inverted() @ expected_origin
        for vert in mesh.vertices:
            vert.co -= offset_local
        mesh.update()
        target_obj.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
    
    # === Step 11: Capture 3/4 view thumbnail ===
    if export_dir is None:
        export_dir = os.path.dirname(path_to_original_model)
    
    os.makedirs(export_dir, exist_ok=True)
    
    # Capture thumbnail before export
    model_id = model_info.get("asset_id", base_name)
    thumbnail_filename = f"{model_id}.png"
    thumbnail_path = os.path.join(export_dir, thumbnail_filename)
    
    thumbnail_result = capture_thumbnail_three_quarter_view(filepath=thumbnail_path, max_size=1024)
    thumbnail_url = None
    if thumbnail_result.get("success"):
        thumbnail_url = os.path.abspath(thumbnail_path)
        # print(f"Thumbnail saved: {thumbnail_url}")
    else:
        print(f"Warning: Failed to capture thumbnail: {thumbnail_result.get('error')}")
    
    # === Step 11b: Capture directional views (front, top, left) ===
    directional_view_urls = {}
    directions_to_capture = ["front", "top", "left"]
    
    # print(f"Capturing directional views: {directions_to_capture}")
    view_results = get_object_images_by_viewport_directions(directions_to_capture, max_size=384)
    
    for view_result in view_results:
        direction = view_result.get("direction", "").lower()
        if view_result.get("success") and direction in directions_to_capture:
            # Copy from temp to export_dir with proper naming
            temp_path = view_result["filepath"]
            final_filename = f"{model_id}_{direction}_view.png"
            final_path = os.path.join(export_dir, final_filename)
            try:
                import shutil
                shutil.copy2(temp_path, final_path)
                directional_view_urls[f"{direction}_view_url"] = os.path.abspath(final_path)
                # print(f"{direction.capitalize()} view saved: {final_path}")
            except Exception as e:
                print(f"Warning: Failed to save {direction} view: {e}")
        else:
            print(f"Warning: Failed to capture {direction} view: {view_result.get('error', 'Unknown error')}")
    
    # === Step 12: Export ===
    # Export with original filename (keeping .glb extension)
    export_filename = f"{base_name}.glb"
    export_path = os.path.join(export_dir, export_filename)
    
    # Select only the final object
    for o in bpy.context.selected_objects:
        o.select_set(False)
    final_obj.select_set(True)
    bpy.context.view_layer.objects.active = final_obj
    
    # Ensure the object is only in the scene's root collection (no nested collections)
    current_scene = bpy.context.scene
    
    # Unlink from all collections
    for coll in list(final_obj.users_collection):
        coll.objects.unlink(final_obj)
    # Link directly to scene's root collection
    current_scene.collection.objects.link(final_obj)
    
    # Delete ALL child collections from the scene (recursively) to avoid collection hierarchy in export
    def remove_all_child_collections(parent_collection):
        for child_coll in list(parent_collection.children):
            remove_all_child_collections(child_coll)  # Recursively remove nested collections
            try:
                bpy.data.collections.remove(child_coll)
            except Exception:
                pass
    
    remove_all_child_collections(current_scene.collection)
    
    # Export as GLB - only selected object from active scene
    bpy.ops.export_scene.gltf(
        filepath=export_path,
        export_format='GLB',
        use_selection=True,          # Export only selected objects
        use_active_scene=True,       # Export only active scene (not all scenes)
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
    
    # === Step 13: Cleanup and switch back to original scene ===
    # Switch back to the original scene
    if original_scene_name in bpy.data.scenes:
        bpy.context.window.scene = bpy.data.scenes[original_scene_name]
    
    # Delete the temporary scene and its objects
    _delete_scene_and_its_objects("Temporary_scene")
    
    # Purge orphan data
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    
    result = {
        'export_path': export_path
    }
    
    if thumbnail_url is not None:
        result['thumbnail_url'] = thumbnail_url
    
    if rotation_correction_result is not None:
        result['rotation_correction'] = rotation_correction_result
    
    # Add directional view URLs
    result.update(directional_view_urls)
    
    # print(result)
    return result