"""Generate supplementary decorative assets to enhance scene atmosphere.

This module processes all scenes in a single LLM call for better context and performance.
When a scene has a reference_scene_id, it reuses supplementary assets from the reference
scene (same asset_ids) and only generates additional new assets if needed.
"""

import json
import time
import gc
import warnings
from typing import Any, Dict, List, Optional
from copy import deepcopy

try:
    from ..llm_completion import completion
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from llm_completion import completion

try:
    from .set_dresser_schema import (
        SupplementarySceneOutput,
        SupplementaryTextToImagePromptSheet,
        SupplementaryAssetWithPrompt,
        SupplementarySceneDetail,
        SupplementarySceneSetup,
        AllScenesSupplementaryOutput,
    )
except ImportError:
    from set_dresser_schema import (
        SupplementarySceneOutput,
        SupplementaryTextToImagePromptSheet,
        SupplementaryAssetWithPrompt,
        SupplementarySceneDetail,
        SupplementarySceneSetup,
        AllScenesSupplementaryOutput,
    )


def _format_dimension(value: Any) -> str:
    """Return a string representation for optional numeric values."""
    if value is None:
        return "N/A"
    return f"{value}"


def get_scene_reference_map(storyboard_script: Dict[str, Any]) -> Dict[int, Optional[int]]:
    """Build a mapping of scene_id to reference_scene_id.
    
    Args:
        storyboard_script: The complete storyboard script with all data.
        
    Returns:
        Dict mapping scene_id to its reference_scene_id (or None if no reference).
    """
    reference_map = {}
    for scene in storyboard_script.get("scene_details", []):
        scene_id = scene.get("scene_id")
        scene_setup = scene.get("scene_setup", {})
        reference_map[scene_id] = scene_setup.get("reference_scene_id")
    return reference_map


def create_supplementary_asset_prompt_all_scenes(
    storyboard_script: Dict[str, Any]
) -> str:
    """Create a markdown prompt for generating supplementary assets for all scenes at once.
    
    When a scene has a reference_scene_id, the prompt instructs the model to reuse
    supplementary assets from the reference scene (same asset_ids) and only generate
    additional new assets if needed.
    
    Args:
        storyboard_script: The complete storyboard script with all data.
    
    Returns:
        A markdown-formatted string prompt for the LLM containing all scenes.
    """
    lines = []
    
    # Build reference map
    reference_map = get_scene_reference_map(storyboard_script)
    
    # Get all scene IDs
    scene_ids = [scene.get("scene_id") for scene in storyboard_script.get("scene_details", [])]
    
    # Header with reference relationships
    lines.append("# Scene Reference Relationships\n")
    lines.append("The following shows which scenes share locations and should reuse supplementary assets:\n")
    for scene_id in scene_ids:
        ref_id = reference_map.get(scene_id)
        if ref_id is not None:
            lines.append(f"- **Scene {scene_id}** references **Scene {ref_id}** (MUST reuse supplementary assets from Scene {ref_id})")
        else:
            lines.append(f"- **Scene {scene_id}** has no reference (generate new supplementary assets)")
    lines.append("")
    
    # Process each scene
    for scene_id in scene_ids:
        lines.append(f"\n{'='*60}")
        lines.append(f"# SCENE {scene_id}")
        lines.append(f"{'='*60}\n")
        
        # Find scene data
        target_scene_detail = None
        target_asset_ids = set()
        for scene in storyboard_script.get("scene_details", []):
            if scene.get("scene_id") == scene_id:
                target_scene_detail = scene
                scene_setup = scene.get("scene_setup", {})
                target_asset_ids = set(scene_setup.get("asset_ids", []))
                break
        
        ref_id = reference_map.get(scene_id)
        if ref_id is not None:
            lines.append(f"**⚠️ IMPORTANT: This scene references Scene {ref_id}.**")
            lines.append(f"**You MUST reuse the same supplementary asset_ids from Scene {ref_id}.**")
            lines.append(f"**Only add NEW supplementary assets if the scene requires additional decoration.**\n")
        
        # Storyboard outline for this scene
        lines.append("## Storyboard Outline\n")
        if "storyboard_outline" in storyboard_script:
            for scene in storyboard_script["storyboard_outline"]:
                if scene.get('scene_id') == scene_id:
                    lines.append(f"{scene.get('scene_description', '')}\n")
                    if "shots" in scene:
                        lines.append("### Shots")
                        for shot in scene["shots"]:
                            shot_id = shot.get('shot_id', 'N/A')
                            lines.append(f"- **Shot {shot_id}**: {shot.get('shot_description', '')}")
                        lines.append("")
                    break
        
        # Existing assets in this scene
        lines.append("\n## Existing Plot Assets\n")
        if "asset_sheet" in storyboard_script:
            for asset in storyboard_script["asset_sheet"]:
                asset_id = asset.get("asset_id", "N/A")
                if asset_id not in target_asset_ids:
                    continue
                description = asset.get("description", "")
                lines.append(f"- **{asset_id}**: {description}")
            lines.append("")
        
        # Scene details
        lines.append("\n## Scene Details\n")
        if target_scene_detail:
            scene_setup = target_scene_detail.get("scene_setup", {})
            
            scene_type = scene_setup.get("scene_type", "")
            if scene_type:
                lines.append(f"**Scene Type**: {scene_type}")
            
            layout_desc = scene_setup.get("layout_description")
            if layout_desc and isinstance(layout_desc, dict):
                lines.append(f"**Layout Description**: {layout_desc.get('description', '')}")
                scene_size = layout_desc.get("scene_size")
                if scene_size:
                    lines.append(f"**Scene Size (meters)**:")
                    lines.append(f"  - X: {scene_size.get('x_negative', 'N/A')} to {scene_size.get('x', 'N/A')}")
                    lines.append(f"  - Y: {scene_size.get('y_negative', 'N/A')} to {scene_size.get('y', 'N/A')}")
            
            lighting_desc = scene_setup.get("lighting_description", "")
            if lighting_desc:
                lines.append(f"**Lighting**: {lighting_desc}")
            
            ground_desc = scene_setup.get("ground_description", "")
            if ground_desc:
                lines.append(f"**Ground**: {ground_desc}")
            
            lines.append("")
    
    return "\n".join(lines)


def _get_system_prompt_step1() -> str:
    """Return the system prompt for generating supplementary assets for all scenes.
    
    The prompt instructs the model to:
    - Reuse asset_ids from reference scenes when reference_scene_id is set
    - Only generate new supplementary assets when needed
    - Process all scenes in a single response
    """
    return """
You are a Set Dresser AI specializing in 3D scene decoration. Your task is to enhance ALL scenes with supplementary, decorative, plot-unrelated objects that fit the atmosphere and visual style of the story.

### CRITICAL: Asset Reuse Rules

**When a scene has a reference_scene_id:**
1. You MUST reuse the EXACT SAME supplementary asset_ids from the reference scene
2. These reused assets represent the same physical objects in the same location
3. Only add NEW supplementary assets if the scene requires additional decoration not present in the reference scene
4. Mark reused assets with `is_reused: true` and new assets with `is_reused: false`

**When a scene has NO reference_scene_id:**
1. Generate 5-10 new supplementary assets for that scene
2. All assets should have `is_reused: false`

### Your Goal
Add static, decorative objects to populate scenes and enhance atmosphere. These objects should:
1. **NOT be plot-related** - They are background decoration only
2. **Fit the story setting** - Match the time period, location, and visual style
3. **Enhance atmosphere** - Create a more immersive and realistic environment
4. **Be practical for 3D** - Solid objects that can be easily created as 3D models

### Types of Supplementary Objects to Consider

**For Indoor Scenes:**
- Furniture: chairs, shelves, cabinets, rugs, curtains
- Decorations: paintings, vases, candles, clocks, tapestries
- Small objects: books, bottles, baskets, cushions

**For Outdoor Scenes:**
- Nature: trees, bushes, rocks, flowers, logs
- Man-made: fences, barrels, crates, lanterns, signs
- Atmosphere: fallen leaves, mushrooms, stumps

### Rules

1. **Reuse assets for referenced scenes** - If scene X references scene Y, scene X MUST include the same asset_ids as scene Y
2. **Generate 5-10 supplementary assets for non-referenced scenes**
3. **Use unique snake_case asset_ids** - descriptive and specific (e.g., "oak_tree", "stone_bench_2")
4. **Provide a brief visual description** for each asset
5. **Use existing assets as anchors** - describe new asset positions relative to existing assets
6. **Specify relationship and distance** - use "on_top_of", "on_the_left_of", "on_the_right_of", "in_front_of", or "behind"
7. **Respect scene bounding box** - All supplementary assets must be placed within the Scene Size
8. **Asset characteristics** - Each asset should be a single solid object that can be easily created as a 3D model

### Output Format

Generate a JSON object with:
1. `scenes`: Array of scene outputs, each containing:
   - `scene_id`: The scene ID
   - `asset_sheet`: Object with `assets` array (each asset has: asset_id, description, is_reused)
   - `scene_detail`: Layout information with asset_id, anchor_asset_id, relationship, distance, description

Process ALL scenes in the input and output them in the `scenes` array.
"""


def _get_system_prompt_step2() -> str:
    """Return the system prompt for step 2: generating text_to_image prompts."""
    return """
You are an expert prompt engineer for 3D asset generation. Generate detailed text_to_image_prompt for each supplementary decorative asset.

### Guidelines

1. **Opening**: Start with "A single detailed stylized 3D model of [object name], [view], whole object, wide angle shot, centered composition, white background."

2. **View Selection**: Use front view for most objects, side view if more informative (e.g., benches, vehicles)

3. **Material Description**: Describe materials specifically:
   - "Rough-hewn oak wood" instead of "brown wood"
   - "Weathered iron" instead of "metal"
   - "Polished marble" instead of "stone"

4. **Style Consistency**: Match the story's visual style (medieval fantasy for fairy tales, etc.)

5. **Lighting**: "Studio lighting," "flat lighting," "evenly lit," "no shadows"

6. **Avoid**: Smoke, fire, magic effects, motion blur, transparent elements

### Output Format

For each asset, output:
- `asset_id`: Same as input
- `text_to_image_prompt`: Detailed prompt (2-3 sentences)

Generate prompts for all provided supplementary assets.
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


def _format_new_assets_for_prompt_generation(new_assets: List[dict], storyboard_script: dict) -> str:
    """Format NEW supplementary assets (is_reused=False) for prompt generation.
    
    Args:
        new_assets: List of new asset dicts (only assets with is_reused=False).
        storyboard_script: The complete storyboard script for context.
    
    Returns:
        A markdown-formatted string prompt for the LLM.
    """
    lines = []
    
    # Add story context
    lines.append("# Story Context\n")
    if "storyboard_outline" in storyboard_script:
        for scene in storyboard_script["storyboard_outline"]:
            lines.append(f"- Scene {scene.get('scene_id')}: {scene.get('scene_description', '')}")
    lines.append("")
    
    # Add supplementary assets
    lines.append("\n# Supplementary Assets to Generate Prompts For\n")
    lines.append("Generate text_to_image_prompt for each of these NEW assets:\n")
    
    for asset in new_assets:
        lines.append(f"## {asset.get('asset_id', 'N/A')}")
        lines.append(f"- **Description**: {asset.get('description', '')}")
        lines.append("")
    
    return "\n".join(lines)


def _merge_scene_output_with_prompts(scene_output: dict, prompt_sheet: dict) -> dict:
    """Merge scene output with text_to_image prompts."""
    prompt_map = {p["asset_id"]: p["text_to_image_prompt"] for p in prompt_sheet.get("prompts", [])}
    
    merged_assets = []
    for asset in scene_output.get("asset_sheet", {}).get("assets", []):
        merged_asset = asset.copy()
        merged_asset["text_to_image_prompt"] = prompt_map.get(
            asset["asset_id"], 
            asset.get("description", "")
        )
        merged_assets.append(merged_asset)
    
    return {
        "scene_id": scene_output.get("scene_id"),
        "assets": merged_assets,
        "scene_detail": scene_output.get("scene_detail")
    }


def generate_supplementary_asset_info(
    anyllm_api_key=None,
    anyllm_api_base=None,
    anyllm_provider="gemini",
    reasoning_model="gemini-3.1-pro-preview",
    storyboard_script=None,
    reasoning_effort="high"
) -> Dict[str, Any]:
    """Generate supplementary decorative assets for all scenes in a single LLM call.
    
    This function processes all scenes at once for better context and performance.
    When a scene has a reference_scene_id, it instructs the model to reuse 
    supplementary assets from the reference scene (same asset_ids) and only 
    generate additional new assets if needed.
    
    Args:
        anyllm_api_key: API key for authentication.
        anyllm_api_base: Base URL for the API.
        anyllm_provider: LLM provider (default: "gemini").
        reasoning_model: The model name to use (e.g., "gemini-3.1-pro-preview").
        storyboard_script: The complete storyboard script dict containing
            storyboard_outline, asset_sheet, scene_details, and shot_details.
        reasoning_effort: Reasoning effort level ("low", "medium", "high").
    
    Returns:
        dict: Dictionary with 'success' boolean, 'data' (on success), or 'error' (on failure).
            On success, 'data' contains:
            - asset_sheet: List of all new decorative assets with prompts (is_reused=False only)
            - scene_details: List of scene details for supplementary assets
    """
    if storyboard_script is None:
        return {
            "success": False,
            "error": "storyboard_script is required",
            "error_type": "missing_input"
        }
    
    system_prompt_step1 = _get_system_prompt_step1()
    system_prompt_step2 = _get_system_prompt_step2()
    
    # Get all scene IDs from scene_details
    scene_ids = [scene.get("scene_id") for scene in storyboard_script.get("scene_details", [])]
    
    try:
        # ===== STEP 1: Generate supplementary assets for ALL scenes (without prompts) =====
        print("=" * 60)
        print(f"Step 1: Generating supplementary assets for ALL {len(scene_ids)} scenes...")
        print("=" * 60)
        
        prompt = create_supplementary_asset_prompt_all_scenes(storyboard_script)
        
        all_scenes_output = generate_json_response(
            api_key=anyllm_api_key,
            base_url=anyllm_api_base,
            provider=anyllm_provider,
            model=reasoning_model,
            contents=prompt,
            system_instruction=system_prompt_step1,
            response_schema=AllScenesSupplementaryOutput,
            reasoning_effort=reasoning_effort
        )
        
        # Validate Step 1
        try:
            AllScenesSupplementaryOutput(**all_scenes_output)
            print("✓ Step 1 schema validation successful!")
        except Exception as validation_error:
            error_msg = f"Step 1 validation failed: {str(validation_error)}"
            print(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "error_type": "step1_validation_error"
            }
        
        # Collect unique NEW assets (is_reused=False) for prompt generation
        new_assets_for_prompts = []
        seen_asset_ids = set()
        for scene_output in all_scenes_output.get("scenes", []):
            for asset in scene_output.get("asset_sheet", {}).get("assets", []):
                asset_id = asset.get("asset_id")
                is_reused = asset.get("is_reused", False)
                if not is_reused and asset_id not in seen_asset_ids:
                    new_assets_for_prompts.append(asset)
                    seen_asset_ids.add(asset_id)
        
        print(f"  Found {len(new_assets_for_prompts)} unique NEW assets requiring prompts")
        
        # ===== STEP 2: Generate text_to_image prompts for NEW assets only =====
        print("\n" + "=" * 60)
        print("Step 2: Generating text_to_image prompts for new assets...")
        print("=" * 60)
        
        step2_input = _format_new_assets_for_prompt_generation(new_assets_for_prompts, storyboard_script)
        
        prompt_sheet = generate_json_response(
            api_key=anyllm_api_key,
            base_url=anyllm_api_base,
            provider=anyllm_provider,
            model=reasoning_model,
            contents=step2_input,
            system_instruction=system_prompt_step2,
            response_schema=SupplementaryTextToImagePromptSheet,
            reasoning_effort=reasoning_effort
        )
        
        # Validate Step 2
        try:
            SupplementaryTextToImagePromptSheet(**prompt_sheet)
            print("✓ Step 2 schema validation successful!")
        except Exception as validation_error:
            error_msg = f"Step 2 validation failed: {str(validation_error)}"
            print(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "error_type": "step2_validation_error"
            }
        
        # ===== MERGE: Combine all scene outputs with prompts =====
        prompt_map = {p["asset_id"]: p["text_to_image_prompt"] for p in prompt_sheet.get("prompts", [])}
        
        # Build final asset sheet (only NEW assets, not reused)
        all_supplementary_assets = []
        for asset in new_assets_for_prompts:
            merged_asset = {
                "asset_id": asset["asset_id"],
                "description": asset["description"],
                "text_to_image_prompt": prompt_map.get(asset["asset_id"], asset.get("description", "")),
                "is_reused": False
            }
            all_supplementary_assets.append(merged_asset)
        
        # Build scene details
        all_scene_details = []
        for scene_output in all_scenes_output.get("scenes", []):
            scene_id = scene_output.get("scene_id")
            scene_detail = scene_output.get("scene_detail", {})
            
            all_scene_details.append({
                "scene_id": scene_id,
                "scene_setup": scene_detail
            })
            
            # Count assets for this scene
            scene_assets = scene_output.get("asset_sheet", {}).get("assets", [])
            reused_count = sum(1 for a in scene_assets if a.get("is_reused", False))
            new_count = len(scene_assets) - reused_count
            print(f"  Scene {scene_id}: {len(scene_assets)} total ({reused_count} reused, {new_count} new)")
        
        # Validate final output
        final_result = {
            "asset_sheet": all_supplementary_assets,
            "scene_details": all_scene_details
        }
        
        print("\n" + "=" * 60)
        print("All scenes processed successfully!")
        print(f"Total unique NEW supplementary assets: {len(all_supplementary_assets)}")
        print("=" * 60)
        
        return {
            "success": True,
            "data": final_result
        }
        
    except Exception as e:
        error_msg = f"Unexpected error in generate_supplementary_asset_info: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": error_msg,
            "error_type": "unexpected_error"
        }
