"""Generate layout descriptions for supplementary assets to be added to existing scenes."""

import json
import time
import gc
import warnings
from typing import Any, Dict, List, Literal, Optional
from copy import deepcopy

try:
    from ..llm_completion import completion
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from llm_completion import completion

# Import SceneLayoutVerifier from the layout artist operators package
try:
    from ..layout_artist_operators.generate_layout_description import SceneLayoutVerifier
    _HAS_VERIFIER = True
except Exception:
    _HAS_VERIFIER = False
    print("Warning: SceneLayoutVerifier not available; verification loop disabled.")

warnings.filterwarnings("ignore", message="Unclosed client session")
warnings.filterwarnings("ignore", message="Unclosed connector")
from pydantic import BaseModel


# Schema definitions matching LayoutObject from generate_layout_description.py
class Location(BaseModel):
    x: float
    y: float
    z: float


class Rotation(BaseModel):
    x: int
    y: int
    z: int


Relationship = Literal[
    "on_top_of",
    "on_the_left_of",
    "on_the_right_of",
    "in_front_of",
    "behind",
]


class LayoutObject(BaseModel):
    """Layout object for a single asset, matching the schema in generate_layout_description.py."""
    asset_id: str
    location: Location
    rotation: Rotation
    relationship: Optional[Relationship] = None
    anchor_asset_id: Optional[str] = None
    contact: Optional[bool] = None


class SupplementaryLayoutDescriptionOutput(BaseModel):
    """Output schema for supplementary asset layout descriptions for a single scene."""
    assets: List[LayoutObject]


def _format_dimension(value: Any) -> str:
    """Return a string representation for optional numeric values.
    
    Formats numbers to 3 decimal places, or as integers if no fractional part.
    """
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if value == int(value):
            return str(int(value))
        return f"{value:.3f}"
    return f"{value}"


def get_reference_supplementary_layout(generated_layouts: Dict[str, Any], reference_scene_id: int) -> Optional[Dict[str, Any]]:
    """Get the supplementary asset layout of a reference scene from already-generated layouts.
    
    This retrieves the supplementary layout (not core asset layout) from a previously
    processed scene, allowing scenes that share a location to have consistent
    supplementary asset placement.
    
    Args:
        generated_layouts: The accumulated result data containing scene_details with
            generated layout_description for previously processed scenes.
        reference_scene_id: The scene_id of the reference scene.
        
    Returns:
        The supplementary layout_description dict of the reference scene, or None if not found.
    """
    if "scene_details" not in generated_layouts:
        return None
    
    for scene in generated_layouts["scene_details"]:
        if scene.get("scene_id") == reference_scene_id:
            scene_setup = scene.get("scene_setup", {})
            layout_desc = scene_setup.get("layout_description")
            if isinstance(layout_desc, dict):
                return layout_desc
    return None


def create_supplementary_layout_prompt_for_scene(
    layout_script: Dict[str, Any],
    formatted_supplementary_assets: Dict[str, Any],
    scene_id: int,
    generated_layouts: Optional[Dict[str, Any]] = None
) -> str:
    """Create a markdown prompt for generating supplementary asset layout descriptions for a single scene.
    
    Args:
        layout_script: The complete layout script with storyboard_outline, asset_sheet, scene_details, shot_details.
        formatted_supplementary_assets: The supplementary assets data with asset_sheet and scene_details.
        scene_id: The scene ID to generate layout descriptions for.
        generated_layouts: The accumulated result data containing already-generated supplementary
            layouts for previous scenes. Used to provide reference scene supplementary layout
            when a scene has a reference_scene_id.
    
    Returns:
        A markdown-formatted string prompt for the LLM.
    """
    lines = []
    
    # Find the target scene's data from layout_script
    target_scene_detail = None
    target_asset_ids = set()
    reference_scene_id = None
    if "scene_details" in layout_script:
        for scene in layout_script["scene_details"]:
            if scene.get("scene_id") == scene_id:
                target_scene_detail = scene
                scene_setup = scene.get("scene_setup", {})
                target_asset_ids = set(scene_setup.get("asset_ids", []))
                reference_scene_id = scene_setup.get("reference_scene_id")
                break
    
    # Find the supplementary assets for this scene
    supplementary_scene_detail = None
    supplementary_asset_ids = set()
    if "scene_details" in formatted_supplementary_assets:
        for scene in formatted_supplementary_assets["scene_details"]:
            if scene.get("scene_id") == scene_id:
                supplementary_scene_detail = scene
                scene_setup = scene.get("scene_setup", {})
                supplementary_asset_ids = set(scene_setup.get("asset_ids", []))
                break
    
    # ===== EXISTING SCENE INFORMATION =====
    lines.append("# Existing Scene Information\n")
    lines.append("The following describes an existing scene in Blender with its current assets and layout.\n")
    
    # 1. Storyboard outline (only target scene: scene_description, shots)
    lines.append("## Storyboard Outline\n")
    if "storyboard_outline" in layout_script:
        for scene in layout_script["storyboard_outline"]:
            if scene.get('scene_id') == scene_id:
                lines.append(f"**Scene Description**: {scene.get('scene_description', '')}\n")
                
                if "shots" in scene:
                    lines.append("### Shots")
                    for shot in scene["shots"]:
                        shot_id = shot.get('shot_id', 'N/A')
                        lines.append(f"- **Shot {shot_id}**: {shot.get('shot_description', '')}")
                    lines.append("")
                break
    
    # 2. Asset sheet (only for assets in this scene: description, thumbnail_url, width, depth, height)
    lines.append("\n## Existing Asset Sheet\n")
    lines.append("These are the assets already in this scene:\n")
    if "asset_sheet" in layout_script:
        for asset in layout_script["asset_sheet"]:
            asset_id = asset.get("asset_id", "N/A")
            if asset_id not in target_asset_ids:
                continue
            
            description = asset.get("description", "")
            thumbnail_url = asset.get("thumbnail_url", "")
            width = asset.get("width")
            depth = asset.get("depth")
            height = asset.get("height")
            
            lines.append(f"### {asset_id}")
            lines.append(f"- **Description**: {description}")
            if thumbnail_url:
                lines.append(f"- **Thumbnail**: {thumbnail_url}")
            lines.append(f"- **Dimensions**: width={_format_dimension(width)}, depth={_format_dimension(depth)}, height={_format_dimension(height)}")
            lines.append("")
    
    # 3. Scene details (scene_setup, scene_type, layout_description content)
    lines.append("\n## Scene Details\n")
    if target_scene_detail:
        scene_setup = target_scene_detail.get("scene_setup", {})
        
        scene_type = scene_setup.get("scene_type", "")
        if scene_type:
            lines.append(f"**Scene Type**: {scene_type}\n")
        
        layout_desc = scene_setup.get("layout_description")
        if layout_desc:
            if isinstance(layout_desc, dict):
                lines.append(f"**Layout Description**: {layout_desc.get('description', '')}\n")
                
                # Add scene_size
                scene_size = layout_desc.get("scene_size")
                if scene_size:
                    lines.append(f"**Scene Size (Bounding Box in meters)**:")
                    lines.append(f"  - X range: {_format_dimension(scene_size.get('x_negative'))} to {_format_dimension(scene_size.get('x'))}")
                    lines.append(f"  - Y range: {_format_dimension(scene_size.get('y_negative'))} to {_format_dimension(scene_size.get('y'))}")
                    lines.append("")
                
                # Add existing asset placements
                assets = layout_desc.get("assets", [])
                if assets:
                    lines.append("**Existing Asset Placements**:")
                    for asset in assets:
                        asset_id = asset.get("asset_id", "N/A")
                        location = asset.get("location", {})
                        rotation = asset.get("rotation", {})
                        relationship = asset.get("relationship")
                        anchor_asset_id = asset.get("anchor_asset_id")
                        dimensions = asset.get("dimensions", {})
                        
                        lines.append(f"- **{asset_id}**:")
                        lines.append(f"  - Location: x={_format_dimension(location.get('x', 0))}, y={_format_dimension(location.get('y', 0))}, z={_format_dimension(location.get('z', 0))}")
                        lines.append(f"  - Rotation: x={_format_dimension(rotation.get('x', 0))}, y={_format_dimension(rotation.get('y', 0))}, z={_format_dimension(rotation.get('z', 0))}")
                        lines.append(f"  - Dimensions: x={_format_dimension(dimensions.get('x'))}, y={_format_dimension(dimensions.get('y'))}, z={_format_dimension(dimensions.get('z'))}")
                        if relationship:
                            lines.append(f"  - Relationship: {relationship} {anchor_asset_id or ''}")
                    lines.append("")
            else:
                lines.append(f"**Layout Description**: {layout_desc}\n")
        
        lighting_desc = scene_setup.get("lighting_description", "")
        if lighting_desc:
            lines.append(f"**Lighting**: {lighting_desc}")
        
        ground_desc = scene_setup.get("ground_description", "")
        if ground_desc:
            lines.append(f"**Ground**: {ground_desc}")
        
        lines.append("")
    
    # 4. Shot details (only for shots in this scene: shot_id, asset_modifications, character_actions)
    lines.append("\n## Shot Details\n")
    if "shot_details" in layout_script:
        for shot in layout_script["shot_details"]:
            if shot.get("scene_id") != scene_id:
                continue
            
            shot_id = shot.get("shot_id")
            lines.append(f"### Shot {shot_id}")
            
            asset_modifications = shot.get("asset_modifications")
            if asset_modifications:
                lines.append("**Asset Modifications**:")
                for mod in asset_modifications:
                    aid = mod.get("asset_id", "")
                    mod_type = mod.get("modification_type", "")
                    desc = mod.get("description", "")
                    lines.append(f"- {aid} ({mod_type}): {desc}")
            
            character_actions = shot.get("character_actions")
            if character_actions:
                lines.append("**Character Actions**:")
                for action in character_actions:
                    aid = action.get("asset_id", "")
                    desc = action.get("action_description", "")
                    lines.append(f"- {aid}: {desc}")
            
            if not asset_modifications and not character_actions:
                lines.append("No modifications or actions in this shot.")
            
            lines.append("")
    
    # ===== SUPPLEMENTARY ASSETS TO BE ADDED =====
    lines.append("\n# Supplementary Assets to be Added to the Scene\n")
    lines.append("The following decorative assets need to be placed in the scene. Generate precise layout descriptions for each.\n")
    
    # Supplementary asset sheet (only for assets in this scene)
    lines.append("## Supplementary Asset Sheet\n")
    if "asset_sheet" in formatted_supplementary_assets:
        for asset in formatted_supplementary_assets["asset_sheet"]:
            asset_id = asset.get("asset_id", "N/A")
            if asset_id not in supplementary_asset_ids:
                continue
            
            description = asset.get("description", "")
            thumbnail_url = asset.get("thumbnail_url", "")
            width = asset.get("width")
            depth = asset.get("depth")
            height = asset.get("height")
            
            lines.append(f"### {asset_id}")
            lines.append(f"- **Description**: {description}")
            if thumbnail_url:
                lines.append(f"- **Thumbnail**: {thumbnail_url}")
            lines.append(f"- **Dimensions**: width={_format_dimension(width)}, depth={_format_dimension(depth)}, height={_format_dimension(height)}")
            lines.append("")
    
    # Scene setup suggestions for supplementary assets
    lines.append("\n# Scene Setup Suggestions for Supplementary Assets\n")
    if supplementary_scene_detail:
        scene_setup = supplementary_scene_detail.get("scene_setup", {})
        
        layout_descriptions = scene_setup.get("layout_descriptions", [])
        if layout_descriptions:
            lines.append("**Suggested Placements**:")
            for layout in layout_descriptions:
                asset_id = layout.get("asset_id", "N/A")
                anchor_asset_id = layout.get("anchor_asset_id", "")
                relationship = layout.get("relationship", "")
                distance = layout.get("distance", 0)
                description = layout.get("description", "")
                
                lines.append(f"- **{asset_id}**: {relationship} {anchor_asset_id} (distance: {_format_dimension(distance)}m)")
                lines.append(f"  - {description}")
            lines.append("")
    
    # Reference scene SUPPLEMENTARY layout information (if this scene has a reference_scene_id)
    if reference_scene_id is not None and generated_layouts is not None:
        ref_supplementary_layout = get_reference_supplementary_layout(generated_layouts, reference_scene_id)
        if ref_supplementary_layout:
            lines.append("\n# Reference Scene Supplementary Asset Layout (Scene {})\n".format(reference_scene_id))
            lines.append("**IMPORTANT**: This scene shares a similar location with Scene {}. ".format(reference_scene_id))
            lines.append("The supplementary assets in the reference scene have already been placed. ")
            lines.append("You MUST place the supplementary assets for this scene in the **SAME positions** as the reference scene ")
            lines.append("to maintain visual consistency. Use the exact same coordinates for matching asset_ids.\n")
            
            # Add reference scene supplementary asset positions
            ref_assets = ref_supplementary_layout.get("assets", [])
            if ref_assets:
                lines.append("## Reference Scene Supplementary Asset Positions (COPY THESE)")
                for asset in ref_assets:
                    asset_id = asset.get("asset_id", "N/A")
                    location = asset.get("location", {})
                    rotation = asset.get("rotation", {})
                    relationship = asset.get("relationship")
                    anchor_asset_id = asset.get("anchor_asset_id")
                    contact = asset.get("contact")
                    
                    lines.append(f"- **{asset_id}**:")
                    lines.append(f"  - Location: x={_format_dimension(location.get('x', 0))}, y={_format_dimension(location.get('y', 0))}, z={_format_dimension(location.get('z', 0))}")
                    lines.append(f"  - Rotation: x={rotation.get('x', 0)}, y={rotation.get('y', 0)}, z={rotation.get('z', 0)}")
                    if relationship:
                        lines.append(f"  - Relationship: {relationship} {anchor_asset_id or ''}")
                    if contact is not None:
                        lines.append(f"  - Contact: {contact}")
                lines.append("")
    
    return "\n".join(lines)


def _get_system_prompt() -> str:
    """Return the system prompt for generating supplementary asset layout descriptions."""
    return """
You are a specialist AI 3D Scene Layout Planner for Blender. Your task is to generate precise layout descriptions for supplementary decorative assets that will be added to an existing scene.

### Your Goal
For each supplementary asset, generate a layout object that specifies:
1. `asset_id`: The ID of the supplementary asset to place
2. `location`: Precise x, y, z coordinates in meters
3. `rotation`: Rotation angles in degrees (x, y, z) as integers
4. `relationship`: The spatial relationship to the anchor asset (optional)
5. `anchor_asset_id`: An existing asset to use as a spatial reference (optional)
6. `contact`: Whether the asset is in direct contact with the anchor (optional)

### Coordinate System
* **Viewpoint:** Assume a default view from -Y to +Y (front view).
* `X-Axis`: Positive = Right, Negative = Left.
* `Y-Axis`: Positive = Back (away from view), Negative = Front (towards view).
* `Z-Axis`: Positive = Up, Negative = Down.
* **Object Origin:** All assets have their origin point at their **bottom center**.
* **Ground Plane:** An object with `location` (0, 0, 0) will be at the center of the scene, sitting on the ground.
* **Z-Location:** The `z` location for all assets should be 0, *unless* an object is on top of another object or floating/hanging on the wall.

### Relationship Types
Valid relationships are:
- `on_top_of`: Asset is placed on top of the anchor
- `on_the_left_of`: Asset is to the left of the anchor (negative X direction from default view)
- `on_the_right_of`: Asset is to the right of the anchor (positive X direction from default view)
- `in_front_of`: Asset is in front of the anchor (negative Y direction from default view)
- `behind`: Asset is behind the anchor (positive Y direction from default view)

### Location Calculation Rules

* **Contact `on_top_of`:** If 'Object A' is on 'Object B', set `A_location_z = B_dimensions_z`. (Assuming B is on the ground, its `z` is 0).
* **Contact `on_the_right_of`:** If 'Object A' is in direct geometry contact to the right of 'Object B', their Y and Z locations might be similar, but their X locations are calculated as:
    `A_location_x = B_location_x + (B_dimensions_x / 2) + (A_dimensions_x / 2)`
* **Contact `on_the_left_of`:** If 'Object A' is in direct geometry contact to the left of 'Object B':
    `A_location_x = B_location_x - (B_dimensions_x / 2) - (A_dimensions_x / 2)`
* **Contact `behind`:** If 'Object A' is in direct geometry contact behind 'Object B':
    `A_location_y = B_location_y + (B_dimensions_y / 2) + (A_dimensions_y / 2)`
* **Contact `in_front_of`:** If 'Object A' is in direct geometry contact in front of 'Object B':
    `A_location_y = B_location_y - (B_dimensions_y / 2) - (A_dimensions_y / 2)`
* **Proximity (No Contact):** If assets are *close* but not touching, the distance between them must be greater than the calculated contact distance. For example, if A is "to the right of" B but not touching:
    `A_location_x > B_location_x + (B_dimensions_x / 2) + (A_dimensions_x / 2)`
* For assets with no direct contact, you need to determine the distance between the two assets based on the calculation, plot, and common knowledge. For example, you can set a close distance for two characters when they are talking, and reserve a longer distance if one charater is walking towards a object in the later shots of the scene.
* If you need to stick or hang an asset to the wall for an indoor scene, first determine which wall the asset should be stuck to, then calculate the location based on the size limit, which is the location of the wall. Then add or minus the depth/2 (dimensions_y/2) of the asset to the location of the wall based on the wall's direction. For example, if the asset is stuck to the right wall on the X direction, the location of the asset should be `location_x = wall_location_x - (dimensions_y/2)`, and the rotation of the asset should be `rotation_z = -90`.
* For 3D models that are approximately cuboid in shape and where the definition of the 'front' may be ambiguous (e.g., tables, houses, carpets, where the front and back may appear identical), the dimensions are defined as follows: width denotes the long side, depth the short side, and height the vertical height. By default, the front of the object is the side with the longest dimension. You can rotate the object to achieve the desired orientation in the layout if needed.
* The number MUST be accurate to three decimal places (e.g., 1.700, 0.454) for location.

### Rotation Rules
* **Default State (0, 0, 0):** The object's "front" faces the viewport (negative Y direction).
* `Z: 0` = Faces front, `Z: 90` = Faces right, `Z: -90` = Faces left, `Z: 180` = Faces back.
* Keep `X` and `Y` at 0 unless the object needs tilting.
* Values: integers from -180 to 180.

### Rules

1. **Use existing assets as anchors**: Reference any asset from the existing scene, or chain placements with other supplementary assets.

2. **Respect scene boundaries**: All assets must fit within the scene's bounding box.

3. **Avoid collisions**: Use dimensions to calculate safe positions. You also need to pay attention to the "Character Actions" for each characters in the later shots in the scene, especially the actions such as "walk" and "run", to avoid collisions, do not place the assets on the path of the characters.

4. **For indoor wall placements**: If sticking an asset to a wall, calculate based on scene size limits.

5. **Be flexible**: You don't need to strictly follow the "Scene Setup Suggestions". If one suggestion may lead to collision, you can place the asset elsewhere.

### Output Format

Generate a JSON object with:
```json
{
    "assets": [
        {
            "asset_id": "supplementary_asset_id",
            "location": {"x": 1.500, "y": 2.000, "z": 0.000},
            "rotation": {"x": 0, "y": 0, "z": 90},
            "relationship": "on_the_left_of",
            "anchor_asset_id": "existing_asset_id",
            "contact": false
        }
    ]
}
```

### Tolerance & Verification

Your layout will be verified geometrically against the existing scene. The verification uses tolerances:

* **Relationship (position):** The valid bearing sector spans 90° (e.g., "behind" = bearing 45°–135° from the anchor). Position the asset naturally within this sector.
* **Contact:** Surfaces must be within 0.05 m of each other to count as "touching".
* **Occlusion:** Bounding-box penetration depth must be < 0.02 m unless `on_top_of`.

Treat `relationship` as **general guidance**, not a rigid constraint. Choose positions that look natural for the scene rather than aiming for mathematically perfect alignment.

Analyze the existing scene layout and generate precise layout objects for all supplementary assets.
"""


def generate_json_response(
    api_key=None,
    base_url=None,
    provider="gemini",
    model="gemini-3.1-pro-preview",
    contents=None,
    system_instruction=None,
    response_schema=None,
    reasoning_effort="high",
    max_retries=3,
    retry_delay=5
) -> dict:
    """Generate JSON response using any-llm with reasoning capability and retry logic."""
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": contents})
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                provider=provider,
                reasoning_effort=reasoning_effort,
                messages=messages,
                response_format=response_schema,
                api_key=api_key,
                api_base=base_url,
            )
            gc.collect()

            print("Output tokens:", response.usage.completion_tokens)
            
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                backoff_time = 2 * (2 ** attempt)
                print(f"Attempt {attempt + 1} failed: {str(e)[:100]}...")
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
            else:
                raise last_error


def validate_layout_descriptions(
    formatted_supplementary_assets: Dict[str, Any],
    layout_result: Dict[str, Any],
    scene_id: int
) -> bool:
    """Validate that all supplementary assets for a scene have layout descriptions.
    
    Args:
        formatted_supplementary_assets: The supplementary assets data.
        layout_result: The generated layout descriptions.
        scene_id: The scene ID being validated.
        
    Returns:
        True if validation passes, False otherwise.
    """
    try:
        # Get expected supplementary asset IDs for this scene
        expected_asset_ids = set()
        for scene in formatted_supplementary_assets.get("scene_details", []):
            if scene.get("scene_id") == scene_id:
                scene_setup = scene.get("scene_setup", {})
                expected_asset_ids = set(scene_setup.get("asset_ids", []))
                break
        
        # Get produced asset IDs from assets list
        produced_asset_ids = set()
        assets = layout_result.get("assets", [])
        for asset in assets:
            asset_id = asset.get("asset_id")
            if asset_id:
                produced_asset_ids.add(asset_id)
        
        # Check if all expected assets have layout descriptions
        if expected_asset_ids != produced_asset_ids:
            missing = expected_asset_ids - produced_asset_ids
            extra = produced_asset_ids - expected_asset_ids
            if missing:
                print(f"Missing layout descriptions for: {missing}")
            if extra:
                print(f"Unexpected layout descriptions for: {extra}")
            return False
        
        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def _verify_supplementary_assets(
    existing_assets: List[Dict[str, Any]],
    supplementary_assets: List[Dict[str, Any]],
    asset_dims: Dict[str, Dict[str, float]],
    supplementary_ids: set,
    scene_id: Any,
) -> List[Dict[str, Any]]:
    """Run geometric verification on supplementary assets against existing scene.

    Combines existing + supplementary assets into one scene, runs the
    SceneLayoutVerifier, and filters errors to only those involving at
    least one supplementary asset.
    """
    if not _HAS_VERIFIER:
        return []

    # Ensure every asset dict has the fields the verifier expects
    combined: List[Dict[str, Any]] = []
    for a in existing_assets:
        entry = dict(a)
        entry.setdefault("direction", None)
        entry.setdefault("anchor_asset_id", None)
        entry.setdefault("relationship", None)
        entry.setdefault("contact", None)
        combined.append(entry)
    for a in supplementary_assets:
        entry = dict(a)
        entry.setdefault("direction", None)
        entry.setdefault("anchor_asset_id", None)
        entry.setdefault("relationship", None)
        entry.setdefault("contact", None)
        combined.append(entry)

    scene_json = {"assets": combined}
    verifier = SceneLayoutVerifier(scene_json, asset_dims, scene_id=scene_id)
    all_errors = verifier.verify_scene()

    # Keep only errors that involve at least one supplementary asset
    filtered: List[Dict[str, Any]] = []
    for err in all_errors:
        aid = err.get("asset_id", "")
        if " <-> " in aid:
            a, b = aid.split(" <-> ")
            if a in supplementary_ids or b in supplementary_ids:
                filtered.append(err)
        elif aid in supplementary_ids:
            filtered.append(err)
    return filtered


def _format_supplementary_errors_for_llm(errors: List[Dict[str, Any]]) -> str:
    """Format verification errors into a structured correction prompt."""
    if not errors:
        return ""

    lines = [
        "# Layout Verification Errors",
        "",
        "Your previous supplementary asset layout has the following geometric "
        "verification errors. Please fix ALL of them in your next response while "
        "keeping the layout natural. The fix suggestions provide exact values for "
        "reference, but you do NOT need to use them verbatim — any value within "
        "the stated tolerance range is acceptable. Only change the values that "
        "need fixing; keep everything else the same.",
        "",
        f"## Errors ({len(errors)})",
        "",
    ]
    for e in errors:
        lines.append(f"- **{e['asset_id']}** [{e['error_type']}]: {e['detail']}")
        fix = e.get("fix")
        if fix:
            lines.append(f"  **FIX**: {fix}")
    lines.append("")
    lines.append(
        "Apply fixes for all errors above. Use the suggested exact values as "
        "guidance — you may adjust them slightly to maintain a natural layout "
        "as long as the values stay within the stated tolerance ranges. "
        "Re-output the full corrected JSON. Do not explain—just output the JSON."
    )
    return "\n".join(lines)


def generate_supplementary_layout_description(
    anyllm_api_key: Optional[str] = None,
    anyllm_api_base: Optional[str] = None,
    anyllm_provider: str = "gemini",
    reasoning_model: str = "gemini-3.1-pro-preview",
    layout_script: Optional[Dict[str, Any]] = None,
    formatted_supplementary_assets: Optional[Dict[str, Any]] = None,
    reasoning_effort: str = "high",
    max_retries: int = 3,
    max_improvement_turns: int = 5
) -> Dict[str, Any]:
    """Generate layout descriptions for supplementary assets for all scenes.
    
    This function processes each scene sequentially, generating precise layout
    descriptions for supplementary decorative assets. After initial generation,
    a verification-correction loop checks for geometric errors (occlusions,
    relationship violations, contact issues) and asks the LLM to fix them.
    When a scene has a reference_scene_id, it provides the already-generated
    supplementary layout from the reference scene as context.
    
    Args:
        anyllm_api_key: API key for authentication.
        anyllm_api_base: Base URL for the API.
        anyllm_provider: LLM provider (default: "gemini").
        reasoning_model: The model name to use (e.g., "gemini-3.1-pro-preview").
        layout_script: The complete layout script dict containing
            storyboard_outline, asset_sheet, scene_details, and shot_details.
        formatted_supplementary_assets: The supplementary assets data containing
            asset_sheet and scene_details with initial layout suggestions.
        reasoning_effort: Reasoning effort level ("low", "medium", "high").
        max_retries: Maximum number of retry attempts per scene.
        max_improvement_turns: Maximum rounds of verify→correct per scene.
            Defaults to 5. Set to 0 to skip verification entirely.
    
    Returns:
        dict: Dictionary with 'success' boolean, 'data' (on success), or 'error' (on failure).
            On success, 'data' contains the same structure as formatted_supplementary_assets
            but with updated layout_descriptions for each scene.
    """
    if layout_script is None:
        return {
            "success": False,
            "error": "layout_script is required",
            "error_type": "missing_input"
        }
    
    if formatted_supplementary_assets is None:
        return {
            "success": False,
            "error": "formatted_supplementary_assets is required",
            "error_type": "missing_input"
        }
    
    system_prompt = _get_system_prompt()
    
    # Deep copy the formatted_supplementary_assets to update with generated layouts
    result_data = deepcopy(formatted_supplementary_assets)
    
    # Get all scene IDs from formatted_supplementary_assets
    scene_ids = [scene.get("scene_id") for scene in formatted_supplementary_assets.get("scene_details", [])]
    
    # Build asset_dims from both asset sheets (existing + supplementary)
    asset_dims: Dict[str, Dict[str, float]] = {}
    for a in layout_script.get("asset_sheet", []):
        asset_dims[a["asset_id"]] = {
            "w": a.get("width", 0), "d": a.get("depth", 0), "h": a.get("height", 0)
        }
    for a in formatted_supplementary_assets.get("asset_sheet", []):
        asset_dims[a["asset_id"]] = {
            "w": a.get("width", 0), "d": a.get("depth", 0), "h": a.get("height", 0)
        }
    
    try:
        for scene_id in scene_ids:
            print("=" * 60)
            print(f"Processing Scene {scene_id}...")
            print("=" * 60)
            
            # Create prompt for this scene, passing already-generated layouts for reference
            prompt = create_supplementary_layout_prompt_for_scene(
                layout_script, formatted_supplementary_assets, scene_id,
                generated_layouts=result_data
            )
            
            # Get supplementary asset IDs for this scene (for error filtering)
            supplementary_ids: set = set()
            for scene in formatted_supplementary_assets.get("scene_details", []):
                if scene.get("scene_id") == scene_id:
                    supplementary_ids = set(scene.get("scene_setup", {}).get("asset_ids", []))
                    break
            
            # Get existing assets from layout_script for verification
            existing_assets: List[Dict[str, Any]] = []
            for scene in layout_script.get("scene_details", []):
                if scene.get("scene_id") == scene_id:
                    layout_desc = scene.get("scene_setup", {}).get("layout_description")
                    if isinstance(layout_desc, dict):
                        existing_assets = deepcopy(layout_desc.get("assets", []))
                    break
            
            # Build messages for multi-turn conversation
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Helper: call LLM with retry and validation
            def _call_llm(attempt_label: str) -> Optional[Dict[str, Any]]:
                for attempt in range(max_retries):
                    try:
                        response = completion(
                            model=reasoning_model,
                            provider=anyllm_provider,
                            reasoning_effort=reasoning_effort,
                            messages=messages,
                            response_format=SupplementaryLayoutDescriptionOutput,
                            api_key=anyllm_api_key,
                            api_base=anyllm_api_base,
                        )
                        gc.collect()
                        result = json.loads(response.choices[0].message.content)
                    except Exception as e:
                        print(f"Error for scene {scene_id} "
                              f"({attempt_label}, attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            backoff_time = 2 * (2 ** attempt)
                            print(f"Retrying in {backoff_time} seconds...")
                            time.sleep(backoff_time)
                            continue
                        return None
                    
                    # Validate schema
                    try:
                        SupplementaryLayoutDescriptionOutput(**result)
                    except Exception as e:
                        print(f"Schema validation error for scene {scene_id} "
                              f"({attempt_label}, attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            continue
                        return None
                    
                    # Validate asset IDs
                    if not validate_layout_descriptions(formatted_supplementary_assets, result, scene_id):
                        print(f"Asset ID validation failed for scene {scene_id} "
                              f"({attempt_label}, attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            continue
                        return None
                    
                    return result
                return None
            
            # ------------------------------------------------------------------
            # Phase 1: Initial generation
            # ------------------------------------------------------------------
            layout_result = _call_llm("initial")
            if layout_result is None:
                return {
                    "success": False,
                    "error": f"Scene {scene_id} generation failed after {max_retries} attempts",
                    "error_type": "generation_error"
                }
            
            print(f"Successfully generated initial supplementary layout for scene {scene_id}")
            
            # ------------------------------------------------------------------
            # Phase 2: Iterative verification → correction loop
            # ------------------------------------------------------------------
            reflection_log = {
                "num_turns": 0,
                "converged": False,
                "turns": [],
            }

            if max_improvement_turns > 0 and _HAS_VERIFIER:
                # Append assistant response for multi-turn conversation
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(layout_result, ensure_ascii=False)
                })
                
                best_result = layout_result
                best_error_count = float("inf")
                
                for turn in range(1, max_improvement_turns + 1):
                    supp_assets = layout_result.get("assets", [])
                    errors = _verify_supplementary_assets(
                        existing_assets, supp_assets,
                        asset_dims, supplementary_ids, scene_id
                    )
                    error_count = len(errors)

                    # Collect per-turn stats
                    by_type: Dict[str, int] = {}
                    for e in errors:
                        by_type[e["error_type"]] = by_type.get(e["error_type"], 0) + 1
                    reflection_log["turns"].append({
                        "turn": turn,
                        "total_errors": error_count,
                        "errors_by_type": by_type,
                        "error_details": [
                            {k: v for k, v in e.items() if k != "fix"}
                            for e in errors
                        ],
                    })
                    reflection_log["num_turns"] = turn
                    
                    print(f"  Scene {scene_id} improvement turn {turn}: {error_count} error(s)")
                    
                    if error_count < best_error_count:
                        best_error_count = error_count
                        best_result = layout_result
                    
                    if error_count == 0:
                        reflection_log["converged"] = True
                        print(f"  Scene {scene_id}: all verification checks passed after {turn} turn(s)")
                        break
                    
                    # Build correction prompt and append as a new user message
                    correction_text = _format_supplementary_errors_for_llm(errors)
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": correction_text}]
                    })
                    
                    corrected = _call_llm(f"improvement turn {turn}")
                    if corrected is None:
                        print(f"  Scene {scene_id}: LLM failed at turn {turn}, "
                              f"using best result ({best_error_count} errors)")
                        layout_result = best_result
                        break
                    
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps(corrected, ensure_ascii=False)
                    })
                    layout_result = corrected
                else:
                    # Exhausted turns — final verification
                    supp_assets = layout_result.get("assets", [])
                    final_errors = _verify_supplementary_assets(
                        existing_assets, supp_assets,
                        asset_dims, supplementary_ids, scene_id
                    )
                    final_by_type: Dict[str, int] = {}
                    for e in final_errors:
                        final_by_type[e["error_type"]] = final_by_type.get(e["error_type"], 0) + 1
                    reflection_log["turns"].append({
                        "turn": max_improvement_turns + 1,
                        "total_errors": len(final_errors),
                        "errors_by_type": final_by_type,
                        "error_details": [
                            {k: v for k, v in e.items() if k != "fix"}
                            for e in final_errors
                        ],
                    })
                    if len(final_errors) < best_error_count:
                        best_error_count = len(final_errors)
                        best_result = layout_result
                    if len(final_errors) == 0:
                        reflection_log["converged"] = True
                    
                    layout_result = best_result
                    print(f"  Scene {scene_id}: reached max turns ({max_improvement_turns}), "
                          f"using best result ({best_error_count} errors)")
            
            # Update the result data with generated assets and reflection_log
            for scene in result_data.get("scene_details", []):
                if scene.get("scene_id") == scene_id:
                    scene["scene_setup"]["layout_description"] = {
                        "assets": layout_result["assets"],
                        "reflection_log": reflection_log,
                    }
                    break
            
            print(f"✓ Scene {scene_id} completed with {len(layout_result['assets'])} layout objects")
        
        print("\n" + "=" * 60)
        print("All scenes processed successfully!")
        print("=" * 60)
        
        return {
            "success": True,
            "data": result_data
        }
        
    except Exception as e:
        error_msg = f"Unexpected error in generate_supplementary_layout_description: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": error_msg,
            "error_type": "unexpected_error"
        }


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    with open(file_path, "r") as f:
        return json.load(f)


def save_json_file(file_path: str, data: Dict[str, Any]) -> None:
    """Save data to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
