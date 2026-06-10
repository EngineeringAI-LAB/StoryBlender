import os
try:
    from ..llm_completion import completion
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from llm_completion import completion
from pydantic import BaseModel
import gc
import time
import warnings
from typing import Optional
from pprint import pprint
import json

try:
    from .director_schema import Storyboard, StoryboardWithoutPrompts, TextToImagePromptSheet
except ImportError:
    from director_schema import Storyboard, StoryboardWithoutPrompts, TextToImagePromptSheet


def generate_json_response(
    anyllm_api_key=None,
    anyllm_api_base=None,
    model="gemini-3.1-pro-preview",
    anyllm_provider="gemini",
    contents=None,
    system_instruction=None,
    response_schema=None,
    reasoning_effort="high"
):
    """Generate JSON response using any-llm with reasoning capability.
    
    Args:
        anyllm_api_key: API key for authentication
        anyllm_api_base: Base URL for the API
        model: Model name to use (default: "gemini-3.1-pro-preview")
        anyllm_provider: LLM provider (default: "gemini")
        contents: Input prompt/contents
        system_instruction: System instruction for the model
        response_schema: Pydantic model for response schema
        reasoning_effort: Reasoning effort level ("low", "medium", "high")
    
    Returns:
        dict: Generated JSON response as dictionary
    """
    # Build messages
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": contents})
    
    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            # Generate content using any-llm
            response = completion(
                model=model,
                provider=anyllm_provider,
                reasoning_effort=reasoning_effort,
                messages=messages,
                response_format=response_schema,
                api_key=anyllm_api_key,
                api_base=anyllm_api_base,
            )
            gc.collect()

            print("Output tokens:", response.usage.completion_tokens)
            
            # Return as dict - use attribute access for any-llm response
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                backoff_time = 2 * (2 ** attempt)
                print(f"Attempt {attempt + 1} failed: {str(e)[:100]}...")
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                continue
            raise last_error


def _merge_storyboard_with_prompts(storyboard_without_prompts: dict, prompt_sheet: dict) -> dict:
    """Merge storyboard without prompts with text_to_image_prompt sheet.
    
    Args:
        storyboard_without_prompts: Storyboard dict without text_to_image_prompt in assets
        prompt_sheet: Dict containing prompts list with asset_id and text_to_image_prompt
    
    Returns:
        dict: Complete storyboard with text_to_image_prompt added to each asset
    """
    # Create a mapping from asset_id to text_to_image_prompt
    prompt_map = {p["asset_id"]: p["text_to_image_prompt"] for p in prompt_sheet["prompts"]}
    
    # Create the merged storyboard
    merged = storyboard_without_prompts.copy()
    
    # Add text_to_image_prompt to each asset in asset_sheet
    merged_assets = []
    for asset in storyboard_without_prompts["asset_sheet"]:
        merged_asset = asset.copy()
        if asset["asset_id"] in prompt_map:
            merged_asset["text_to_image_prompt"] = prompt_map[asset["asset_id"]]
        else:
            # Fallback: use description as prompt if not found
            merged_asset["text_to_image_prompt"] = asset["description"]
        merged_assets.append(merged_asset)
    
    merged["asset_sheet"] = merged_assets
    return merged


def _format_storyboard_for_prompt_generation(storyboard_without_prompts: dict) -> str:
    """Format storyboard data as input for the second step (prompt generation).
    
    Args:
        storyboard_without_prompts: Storyboard dict without text_to_image_prompt
    
    Returns:
        str: Formatted string describing the storyboard and assets
    """
    lines = []
    
    # Add story summary
    if "story_summary" in storyboard_without_prompts:
        lines.append(f"## Story Summary\n{storyboard_without_prompts['story_summary']}\n")
    
    # Add storyboard outline
    lines.append("## Storyboard Outline\n")
    for scene in storyboard_without_prompts["storyboard_outline"]:
        lines.append(f"### Scene {scene['scene_id']}: {scene['scene_description']}")
        for shot in scene["shots"]:
            lines.append(f"  - Shot {shot['shot_id']}: {shot['shot_description']}")
        lines.append("")
    
    # Add asset sheet
    lines.append("\n## Asset Sheet\n")
    lines.append("Generate text_to_image_prompt for each of the following assets:\n")
    for asset in storyboard_without_prompts["asset_sheet"]:
        lines.append(f"- **asset_id**: `{asset['asset_id']}`")
        lines.append(f"  - **asset_type**: {asset['asset_type']}")
        lines.append(f"  - **description**: {asset['description']}")
        lines.append("")
    
    return "\n".join(lines)


def generate_script_by_director(
    anyllm_api_key=None,
    anyllm_api_base=None,
    model="gemini-3.1-pro-preview",
    anyllm_provider="gemini",
    contents=None,
    reasoning_effort="high"
):
    """Generate storyboard JSON using a two-step LLM generation process.
    
    Step 1: Generate storyboard structure without text_to_image_prompt
    Step 2: Generate text_to_image_prompt for each asset
    
    Args:
        anyllm_api_key: API key for authentication
        anyllm_api_base: Base URL for the API
        model: Model name to use (default: "gemini-3.1-pro-preview")
        anyllm_provider: LLM provider (default: "gemini")
        contents: Input prompt/contents (story description)
        reasoning_effort: Reasoning effort level ("low", "medium", "high")
    
    Returns:
        dict: Dictionary with 'success' boolean, 'data' (on success), or 'error' (on failure)
    """
    # Step 1 system prompt: Generate storyboard without text_to_image_prompt
    system_prompt_step1 = """
You are a meticulous storyboard script writer. Your primary function is to transform a user's story concept into a comprehensive, structured guide for creating a storyboard in a 3D engine. Your output must be a single, complete JSON object that strictly adheres to the provided schema. You are to follow all rules and principles to ensure the generated storyboard is visually coherent, purposeful, and simple to execute.

### Core Storyboarding Principles

You MUST adhere to the following principles when generating the storyboard:

1. Purposeful Scenes: Every scene must serve a clear narrative purpose and advance the story. Do not create a new scene unless there is a reason.
2. Directed Camera: Use camera movements deliberately to guide the audience's attention. Default to "static" shots for simplicity unless movement is required for storytelling.
3. Avoid Jump-Cuts: When cutting to a new shot in a scene, ensure the camera's new position is not dramatically closer, further away, or at a significantly different angle to create a smooth visual transition.
4. Effective Composition: Utilize negative space effectively. Avoid placing a small character in a vast frame unless it is for a specific narrative or emotional effect (e.g., showing isolation).
5. Simplicity is Key: Keep scenes visually uncluttered. Do not overpopulate the scene with unnecessary objects or characters. Ensure interactions are simple and clear.
6. Maintain Continuity: Within a single scene, all elements (characters, objects, lighting, environment) must remain consistent unless explicitly modified as part of the action.
7. Action and Posing: Every humanoid character in a shot must have a clear pose or action (e.g., "talking", "walking with phone", "idle").
8. Consistent Lighting: Clearly define the lighting for each scene (e.g., "Bright sunny afternoon", "Dimly lit room at night"). This lighting must remain constant throughout a sequence of shots within a single scene unless the plot explicitly requires a change.
9. Subtle Sound Design: Use sound effects sparingly only to enhance the illusion of reality (e.g., "car horns in the distance", "birds singing"). Sound is not required for every shot.

### Glossary

* Scene: A continuous sequence of action, composed of one or more shots. Scenes are numbered sequentially (1, 2, 3...). The overall environment and lighting remain constant within a scene. Each scene represents a major plot point in the story (e.g., "The Queen visits Snow White."). For convenience, each scene in the plot corresponds to a 3D scene in the 3D engine.
* Shots: A single storyboard frame or panel within a scene, representing a specific moment or a minor action. A scene is made up of a sequence of shots (e.g., "The Queen walks to Snow White."). Within each scene, shots are also numbered sequentially (1, 2, 3...). Each scene has at least one shot. All shots in the same scene are based on the initial scene setup or the previous shot.

### Generation Workflow

You must follow this process sequentially:

1. Generate the Story Summary
    * Write a `story_summary` string to describe the entire story, describing the vibe, background, era, etc., in max three sentences.

2. Generate the Storyboard Outline
    * First, based on the user's input, break the story into logical **Scenes**. For each scene, write a single-sentence `scene_description` summarizing its core plot point.
    * Next, for each scene, break it down further into one or more **Shots**. For each shot, write a single-sentence `shot_description` detailing the specific action or moment.
    * Usually, create 1 to 10 scenes for a short story, and ensure each scene has 1 to 5 shots, unless the story is very complex or the user specifies the number of scenes and shots. For scenes and shots, rather be concise than overcomplicated.
    * Rules:
       * When crafting the outline, keep in mind that the scene and plot should be easily executable within a 3D engine. The character cannot have direct geometry interaction with other characters or objects in a way that requires complex physics or animation involving multiple assets simultaneously.
       * If the plot requires too many complex actions or interactions that are difficult to represent in 3D, simplify or restructure it. You SHOULD modify the story to make it more suitable for 3D representation. For example, change "Snow White is eating the apple" to "Snow White is looking at the apple", or change "playing the piano" to "standing by the piano". The only allowed interaction between a character and an object is "on top of", such as "standing on a table" or "lying on the bed"; all other interactions are forbidden, e.g., pushing, pulling, sitting, or holding.

3. Generate the Asset Sheet
    * Identify all unique assets (characters and objects) from your outline. Include objects and characters that are necessary for the story; other background elements that contribute to the atmosphere will be added to the scene later by another agent, which you do not need to concern yourself with. Only identify assets that can be easily manipulated by the 3D engine. Do not include assets that are too large or too small compared to the other assets, such as mountains, a pebble, or a single leaf.
    * Assign a unique, descriptive, snake_case `asset_id` to each (e.g., "snow_white", "red_apple").
    * Define the `asset_type` of asset: either "character" or "object". Only humanoid assets should be defined as "character"; all other assets should be defined as "object".
    * Provide a brief visual `description` for the appearance of each asset; this is used for the general visual concept.
    * Characters and objects can be reused in different shots and scenes.
    * If the same character has multiple outfits, such as Cinderella's maid and princess outfits, we need to create different asset_ids for each looks, and describe each asset separately. The first appearance is the reference_character, and its asset_id needs to be mentioned in other assets of the same character. However, if a character has completely different appearances, like Anakin Skywalker and Darth Vader, there is no need to mention the reference_character.

4. Generate the Scene Details
    * For each `scene_id` from your outline, create a `scene_setup`.
    * Optionally, set `reference_scene_id` to the ID of a previous scene that shares a similar layout. This is useful when different scenes occur in the same place (e.g., a cafe in the morning vs. the same cafe at night). Only reference scenes that have already been defined (i.e., with a lower scene_id). Leave this field as `null` if the scene is entirely new or does not share its layout with any previous scene.
    * List the `asset_ids` of all characters and objects present at the very start of the scene as an array. Usually, a scene needs 3 to 6 assets to make the scene more substantial. Each asset is a single 3D character or object.
    * Define the `scene_type` for each scene, which can either be "indoor" or "outdoor". Do not create too many "indoor" scenes, as they are harder to handle compare to "outdoor" scenes.
    * Write a `layout_description` detailing the initial spatial positions and relationships of these assets, with the following rules:
        1. An asset cannot be inside another asset, for example, "a dagger inside a box" is forbidden.
        2. Always assume each asset is a solid 3D bounding box when two assets have geometry interactions. Be careful with interactions such as "on top of" or "holding". "Prince on top of horse" and "snow_white holding the apple" are NOT feasible because these interactions are too complex to handle. However, "bottle on top of table" is feasible because the top of the table is flat. For spatial relationships that are not feasible, you should modify them to feasible ones; for example, change "prince on top of the horse" to "prince standing next to the horse".
        3. Use distance and angle to describe the relative positions of assets. For example, "prince stands 2 meters away left from the horse".
    * Define the `lighting_description` with a concise description (time of day, weather, environment).
    * Define the `ground_description` with a concise description (e.g., "green grass", "snowy ground", "concrete covered with leaves").
    * Define the `wall_description` with a concise description, for indoor scenes only. (e.g., "polished stone walls", "red brick", "beige wall").

5. Generate the Shot Details
    * For each `shot` in your outline, create a corresponding detailed description.
    * Assign the correct `scene_id` and `shot_id`.
    * Detail any `assets_modifications` from the previous shot (or from the initial `scene_setup` if it's the first shot of a scene).
    * Assets Modification Rules:
        * A modification is for a character or object. For each asset you need to modify, provide `asset_id` and `modification_type`, which can be either "add", "transform", or "remove".
        * If `modification_type` is "add" or "transform", provide a `description` to describe where you want to place the (new) asset and its rotation, for example, "rotate the horse to face the prince". If you added a character in a shot, also remember to add its action in the `character_actions` field.
        * You **CANNOT** modify a part of an object (e.g., "open the door of the house" is forbidden). You can only move, add, or remove the entire object.
        * If there are no changes for objects or characters in a shot, the `assets_modifications` field should be `null`.
        * You can change one or more assets in a shot, but do not change multiple assets in a shot unless necessary.
        * For characters, use the below `character_actions` instead of transform to describe their actions, for example, use "lie down" as an action, instead of "rotate snow_white to lie flat on the ground" as a transform, because actions are more immersive than transforms. The modificaion for character can only be add or remove.
    * Provide a list for how you want the characters to act in each shot with `character_actions`:
        * Only humanoid characters can have actions (e.g., "starts talking", "walks towards snow_white"). Animals and inanimate objects cannot perform actions.
        * Actions are physical activities on limbs level, this does not include emotions or expressions.
        * Actions are not allowed to have direct physical contact with objects (e.g., "sits on the chair" or "holds the apple" are forbidden). Indirect relationships such as "stand next to" or "look at" are allowed. You can modify the original plot to avoid direct physical contact.
        * When an action involves the location change of an asset, for example, "running" or "walking", specify the distance and angle, e.g., "run 2 meters to the left" or "walk 3 meters to the right".
    * Detail any `lighting_modification` from the previous shot if the lighting changes. Only use this if you want to reflect the change of time of day or weather; default to `null`.
        * `new_lighting_description` is a string; describe the new lighting of the scene.
    * Add an optional `sound_effect` if necessary, such as "car horns in the distance", "birds singing", etc.
    * Provide a precise `camera_instruction`, specifying the `focus_on_ids`, `angle`, `distance`, `movement`, `direction`, and a clear `description`.
        * `focus_on_ids` is an array of asset IDs you want to include in the shot.
        * `angle` can be one of: "eye-level", "high-angle", or "low-angle".
        * `distance` can be one of: "close-up", "medium-shot", or "long-shot".
        * `movement` can be one of: "static", "pan", "orbit", "zoom-in", "zoom-out".
        * `direction` defaults to `null`. It is only used when `movement` is "pan" or "orbit"; available options are "left", "right", "up", or "down".
        * Finally, use the `description` field to provide a clear, human-readable summary that synthesizes all the preceding camera parameters into a single instruction, for example, "An eye-level close-up of snow_white that slowly zooms out."

Now, based on the user's input, generate the complete storyboard in the specified JSON format.
"""

    # Step 2 system prompt: Generate text_to_image_prompt for each asset
    system_prompt_step2 = """
You are an expert prompt engineer specializing in creating text-to-image prompts for 3D asset generation. Your task is to generate detailed `text_to_image_prompt` for each asset in the provided storyboard.

You will receive:
1. The story summary
2. The storyboard outline (scenes and shots)
3. The asset sheet with `asset_id`, `asset_type`, and `description` for each asset

For each asset, create a detailed `text_to_image_prompt` that will be used to generate an image, which will then be converted to a 3D asset using image-to-3D generation.

### Guidelines for Writing text_to_image_prompt

1. The "Clean geometry" Introduction: Every prompt must strictly begin with the phrase: "A single detailed textured 3D model of a [realistic or stylized] [asset name] [from story title (if applicable)], [front or side view] (always use front view unless the side view is much more informative than the front view, for example, a bike), full body with detailed facial features (if it is a character), whole object (if it is an object), wide angle shot, centered composition, white background."

2. The "Visible Limb" Doctrine (Crucial for Rigging): You must engineer the description to ensure physical separation of limbs.
    * Mandatory Pose: For humanoid characters, explicitly state: "in a T-Pose".
    * Clothing Modification: You are authorized to alter character designs to remove geometry-hiding elements.
        * Forbidden: Long cloaks, trench coats, floor-length gowns, long hair draped over shoulders.
        * Required Substitutes: Change gowns to "short tunics" or "biker suits"; change coats to "bomber jackets" or "tactical vests"; tie long hair back into a "high ponytail" or "bun."
    * Leg Definition: Explicitly mention footwear and pants/legs to force the AI to generate two distinct leg volumes (e.g., "wearing knee-high boots and tight trousers").

3. Lighting and Texture Neutrality: You must enforce a "Neutral Albedo" look to prevent lighting artifacts from being baked into the 3D texture.
    * Keywords to Add: "Studio lighting," "flat lighting," "soft omnidirectional light," "no shadows," "evenly lit."
    * Keywords to Avoid: "Cinematic lighting," "dramatic shadows," "noir," "rim light," "sunlight," "darkness."

4. Material and Surface Definition: Instead of generic colors, use material descriptors to help the AI predict surface behavior (roughness/metallicity).
    * Bad: "A gray suit."
    * Good: "A polished steel suit," "matte cotton fabric," "worn leather armor," "rough granite skin."

5. Stylistic Consistency: Enforce a specific visual style to ensure the assets look like they belong in the same universe. Unless otherwise specified, default to "Stylized PBR" (Physically Based Rendering) or "Stylized 3D Character" (similar to Overwatch or Fortnite).
    * Keywords: "Stylized," "realistic," "clean topology."

6. Negative Constraints (Physics & Effects): Strictly avoid describing non-solid elements.
    * Forbidden: "Smoke," "fire," "magic spells," "glowing auras," "dust particles," "motion blur."
    * Exception: Solid glowing parts (e.g., "a glowing LED panel on the chest") are acceptable if they are part of the physical mesh.

7. The "3-5 Detail" Limit: Focus on 3 to 5 distinct visual anchors.
    * Focus Areas: Headgear/Hair, Torso/Clothing, Hand/Weapon (sheathed), Footwear.

8. IP Recognition and Anchoring: If the character is from a well-known IP (e.g., Marvel, Nintendo, Disney), use their name and title of the story to anchor the AI's prior knowledge, but immediately follow it with the specific visual overrides from Rule 2 (e.g., "Princess Peach from Mario, wearing a white biker racing suit instead of a dress").

9. Facial Neutrality: Do not describe complex emotions (e.g., "screaming in rage").

10. Paragraph Format: Construct the prompt as a single, cohesive paragraph of 2-3 sentences. This narrative structure helps advanced models understand the relationship between the items (e.g., "The belt is over the tunic" rather than just "belt, tunic").

11. Be clear about the race, gender, and age if the asset is a character (e.g., "a young asian boy"). Be clear about the era if necessary (e.g., "a Victorian-era house").

12. Examples of good prompts:
    * "A single detailed textured 3D model of a realistic Sci-Fi Hoverbike vehicle, side view from right, white background. The vehicle features a sleek, aerodynamic crimson chassis with exposed mechanical engine parts and glowing blue anti-gravity thruster pads underneath. It has a black leather rider's seat and futuristic handlebars. The design is rendered in a high-gloss, stylized cyberpunk aesthetic with clear separation between the chassis and the hovering elements."
    * "A single detailed textured 3D model of a realistic fantasy ranger, front view, full body, white background. The ranger is in a T-Pose, dressed in a short leather tunic that ends at the waist and tight wool trousers tucked into medieval knee-high leather boots. The design avoids loose cloaks, focusing on the matte texture of the fabric and worn leather, lit by even, omnidirectional light."
    * "A single detailed textured 3D model of a realistic dwarven battle axe, side view from right, white background. The weapon features a double-headed blade made of chipped iron attached to a thick oak wood handle wrapped in leather strips. The visual style emphasizes the contrast between the metallic blade and organic wood grain, lit evenly to ensure a clean texture map."

### Output Format

For each asset in the asset sheet, output a JSON object with:
- `asset_id`: The same asset_id from the input
- `text_to_image_prompt`: The detailed prompt you generated

Now, based on the provided storyboard and asset sheet, generate text_to_image_prompt for each asset.
"""

    try:
        # ===== STEP 1: Generate storyboard without text_to_image_prompt =====
        print("=" * 60)
        print("STEP 1: Generating storyboard structure...")
        print("=" * 60)
        
        storyboard_without_prompts = generate_json_response(
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            model=model,
            anyllm_provider=anyllm_provider,
            contents=contents,
            system_instruction=system_prompt_step1,
            response_schema=StoryboardWithoutPrompts,
            reasoning_effort=reasoning_effort
        )
        
        # Validate Step 1 schema
        try:
            validated_step1 = StoryboardWithoutPrompts(**storyboard_without_prompts)
            print("✓ Step 1 schema validation successful!")
        except Exception as validation_error:
            error_msg = f"Step 1 schema validation failed: {str(validation_error)}"
            print(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "error_type": "step1_validation_error"
            }
        
        # ===== STEP 2: Generate text_to_image_prompt for each asset =====
        print("\n" + "=" * 60)
        print("STEP 2: Generating text_to_image_prompt for each asset...")
        print("=" * 60)
        
        # Format the storyboard as input for step 2
        step2_input = _format_storyboard_for_prompt_generation(storyboard_without_prompts)
        
        prompt_sheet = generate_json_response(
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            model=model,
            anyllm_provider=anyllm_provider,
            contents=step2_input,
            system_instruction=system_prompt_step2,
            response_schema=TextToImagePromptSheet,
            reasoning_effort=reasoning_effort
        )
        
        # Validate Step 2 schema
        try:
            validated_step2 = TextToImagePromptSheet(**prompt_sheet)
            print("✓ Step 2 schema validation successful!")
        except Exception as validation_error:
            error_msg = f"Step 2 schema validation failed: {str(validation_error)}"
            print(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "error_type": "step2_validation_error"
            }
        
        # ===== MERGE: Combine storyboard with prompts =====
        print("\n" + "=" * 60)
        print("MERGING: Combining storyboard with text_to_image_prompts...")
        print("=" * 60)
        
        merged_result = _merge_storyboard_with_prompts(storyboard_without_prompts, prompt_sheet)
        
        # Validate final merged schema
        try:
            validated_final = Storyboard(**merged_result)
            print("✓ Final schema validation successful!")
        except Exception as validation_error:
            error_msg = f"Final schema validation failed: {str(validation_error)}"
            print(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "error_type": "final_validation_error"
            }
        
        # Add default tags to each asset in asset_sheet
        if "asset_sheet" in merged_result:
            for asset in merged_result["asset_sheet"]:
                asset["tags"] = ["no_polyhaven", "no_sketchfab"]
        
        return {
            "success": True,
            "data": merged_result,
        }
        
    except Exception as e:
        error_msg = f"Unexpected error in generate_script_by_director: {str(e)}"
        print(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "error_type": "unexpected_error"
        }
