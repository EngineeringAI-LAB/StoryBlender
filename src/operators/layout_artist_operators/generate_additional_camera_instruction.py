"""Generate additional camera instructions for each shot to enhance storytelling."""

from typing import Any, Dict, List
from copy import deepcopy
import json
import gc
import warnings
import time
import os

try:
    from ..llm_completion import completion
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from llm_completion import completion
from pydantic import BaseModel
from typing import Literal, Optional

# Type definitions for camera properties
Angle = Literal['eye-level', 'high angle', 'low angle']
Distance = Literal['close-up', 'medium shot', 'long shot']
Movement = Literal['static', 'pan', 'orbit', 'zoom in', 'zoom out']
Direction = Literal['left', 'right', 'up', 'down']
Mode = Literal['perspective', 'orthographic']
LookAt = Literal['lock', 'track']

class CameraInstruction(BaseModel):
    """Defines the camera's properties and behavior for a single shot."""
    focus_on_ids: List[str]
    angle: Angle
    distance: Distance
    movement: Movement
    direction: Optional[Direction] = None
    description: str

class ShotAdditionalCameraInstructions(BaseModel):
    """Contains additional camera instructions for a specific shot."""
    scene_id: int
    shot_id: int
    additional_camera_instructions: List[CameraInstruction]


class AdditionalCameraInstructionsOutput(BaseModel):
    """Output schema for generating additional camera instructions for all shots."""
    shots: List[ShotAdditionalCameraInstructions]


def extract_prompt_data(storyboard_script: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the relevant data from storyboard script for the LLM prompt.
    
    Extracts:
    - The entire storyboard_outline
    - shot_details: scene_id, shot_id, character_actions (only non-idle actions)
    - camera_instruction as "director_camera_instruction"
    
    Args:
        storyboard_script: The full storyboard script.
        
    Returns:
        A dict containing the extracted data for the prompt.
    """
    result = {}
    
    # Extract the entire storyboard_outline
    if "storyboard_outline" in storyboard_script:
        result["storyboard_outline"] = storyboard_script["storyboard_outline"]
    
    # Extract shot_details with filtered character_actions and camera_instruction
    if "shot_details" in storyboard_script:
        extracted_shots = []
        for shot in storyboard_script["shot_details"]:
            extracted_shot = {
                "scene_id": shot.get("scene_id"),
                "shot_id": shot.get("shot_id"),
            }
            
            # Filter character_actions - only include if "idle" is not in action_description
            character_actions = shot.get("character_actions", [])
            if character_actions:
                filtered_actions = []
                for action in character_actions:
                    action_desc = action.get("action_description", "").lower()
                    if "idle" not in action_desc:
                        filtered_actions.append({
                            "asset_id": action.get("asset_id"),
                            "action_description": action.get("action_description"),
                        })
                if filtered_actions:
                    extracted_shot["character_actions"] = filtered_actions
            
            # Add camera_instruction as director_camera_instruction
            if "camera_instruction" in shot:
                extracted_shot["director_camera_instruction"] = shot["camera_instruction"]
            
            extracted_shots.append(extracted_shot)
        
        result["shot_details"] = extracted_shots
    
    return result


def create_additional_camera_prompt(prompt_data: Dict[str, Any]) -> str:
    """Convert extracted storyboard data into a Markdown formatted prompt.
    
    Args:
        prompt_data: The extracted data containing storyboard_outline and shot_details.
        
    Returns:
        A Markdown formatted string for the user prompt.
    """
    md_parts = []
    
    # 1. Storyboard Outline
    if "storyboard_outline" in prompt_data:
        md_text = "# Storyboard Outline\n\n"
        for scene in prompt_data["storyboard_outline"]:
            scene_id = scene.get("scene_id", "N/A")
            md_text += f"## Scene {scene_id}\n\n"
            md_text += f"{scene.get('scene_description', '')}\n\n"
            
            if "shots" in scene:
                md_text += "### Shots\n\n"
                for shot in scene["shots"]:
                    shot_id = shot.get("shot_id", "N/A")
                    md_text += f"- **Shot {shot_id}**: {shot.get('shot_description', '')}\n"
                md_text += "\n"
        
        md_parts.append(md_text)
    
    # 2. Shot Details with character actions and director camera instructions
    if "shot_details" in prompt_data:
        md_text = "# Shot Details\n\n"
        
        for shot in prompt_data["shot_details"]:
            scene_id = shot.get("scene_id", "N/A")
            shot_id = shot.get("shot_id", "N/A")
            md_text += f"## Scene {scene_id}, Shot {shot_id}\n\n"
            
            # Character actions
            if "character_actions" in shot and shot["character_actions"]:
                md_text += "### Character Actions\n\n"
                for idx, action in enumerate(shot["character_actions"], start=1):
                    asset_id = action.get("asset_id", "")
                    action_desc = action.get("action_description", "")
                    md_text += f"{idx}. **{asset_id}**: {action_desc}\n"
                md_text += "\n"
            
            # Director camera instruction
            if "director_camera_instruction" in shot:
                cam = shot["director_camera_instruction"]
                md_text += "### Director Camera Instruction\n\n"
                md_text += f"- **Focus on**: {', '.join(cam.get('focus_on_ids', []))}\n"
                md_text += f"- **Angle**: {cam.get('angle', 'N/A')}\n"
                md_text += f"- **Distance**: {cam.get('distance', 'N/A')}\n"
                md_text += f"- **Movement**: {cam.get('movement', 'N/A')}\n"
                if cam.get("direction"):
                    md_text += f"- **Direction**: {cam.get('direction')}\n"
                md_text += f"- **Description**: {cam.get('description', '')}\n"
                md_text += "\n"
        
        md_parts.append(md_text)
    
    return "".join(md_parts)


def validate_output(
    storyboard_script: Dict[str, Any],
    output: Dict[str, Any]
) -> bool:
    """Validate the generated additional camera instructions output.
    
    Args:
        storyboard_script: The original storyboard script.
        output: The generated output containing additional camera instructions.
        
    Returns:
        True if validation passes, False otherwise.
    """
    try:
        if not isinstance(output, dict):
            return False
        
        shots = output.get("shots", [])
        if not isinstance(shots, list):
            return False
        
        # Build expected (scene_id, shot_id) pairs from storyboard_script
        expected_pairs = set()
        shot_details = storyboard_script.get("shot_details", [])
        for shot in shot_details:
            scene_id = shot.get("scene_id")
            shot_id = shot.get("shot_id")
            if scene_id is not None and shot_id is not None:
                expected_pairs.add((scene_id, shot_id))
        
        # Check output covers all expected pairs
        output_pairs = set()
        for shot in shots:
            scene_id = shot.get("scene_id")
            shot_id = shot.get("shot_id")
            if scene_id is not None and shot_id is not None:
                output_pairs.add((scene_id, shot_id))
            
            # Validate additional_camera_instructions exists and is a list
            additional_cams = shot.get("additional_camera_instructions", [])
            if not isinstance(additional_cams, list):
                return False
            
            # Validate each has at most 3 additional camera instructions
            if len(additional_cams) > 3:
                return False
        
        # Ensure all expected shots are covered
        if expected_pairs != output_pairs:
            return False
        
        return True
    except Exception:
        return False


def add_camera_ids(merged_script: Dict[str, Any]) -> Dict[str, Any]:
    """Add id and camera_name fields to all camera instructions.
    
    For each shot:
    - Director's camera_instruction gets id=1
    - Additional camera instructions get id=2, 3, 4...
    - camera_name follows pattern: cam_{id}_s_{scene_id}_s_{shot_id}
    
    Args:
        merged_script: The storyboard script with merged camera instructions.
        
    Returns:
        The script with id and camera_name added to all camera instructions.
    """
    result = deepcopy(merged_script)
    
    shot_details = result.get("shot_details", [])
    for shot in shot_details:
        scene_id = shot.get("scene_id")
        shot_id = shot.get("shot_id")
        
        # Add id and camera_name to director's camera_instruction (id=1)
        if "camera_instruction" in shot and shot["camera_instruction"]:
            shot["camera_instruction"]["id"] = 1
            shot["camera_instruction"]["camera_name"] = f"cam_1_s_{scene_id}_s_{shot_id}"
        
        # Add id and camera_name to additional camera instructions (id=2, 3, 4...)
        additional_cams = shot.get("additional_camera_instructions", [])
        for idx, cam in enumerate(additional_cams, start=2):
            cam["id"] = idx
            cam["camera_name"] = f"cam_{idx}_s_{scene_id}_s_{shot_id}"
    
    return result


def merge_camera_instructions(
    storyboard_script: Dict[str, Any],
    additional_output: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge additional camera instructions into the storyboard script.
    
    The director camera instruction is kept as "camera_instruction" (singular),
    and additional camera instructions are added as "additional_camera_instructions".
    Also adds id and camera_name fields to all camera instructions.
    
    Args:
        storyboard_script: The original storyboard script.
        additional_output: The generated additional camera instructions.
        
    Returns:
        The merged storyboard script with additional camera instructions and IDs.
    """
    merged = deepcopy(storyboard_script)
    
    # Build lookup for additional camera instructions by (scene_id, shot_id)
    additional_lookup: Dict[tuple, List[Dict[str, Any]]] = {}
    for shot in additional_output.get("shots", []):
        scene_id = shot.get("scene_id")
        shot_id = shot.get("shot_id")
        additional_cams = shot.get("additional_camera_instructions", [])
        if scene_id is not None and shot_id is not None:
            additional_lookup[(scene_id, shot_id)] = additional_cams
    
    # Merge into shot_details
    shot_details = merged.get("shot_details", [])
    for shot in shot_details:
        scene_id = shot.get("scene_id")
        shot_id = shot.get("shot_id")
        key = (scene_id, shot_id)
        
        if key in additional_lookup:
            shot["additional_camera_instructions"] = additional_lookup[key]
    
    # Add id and camera_name to all camera instructions
    merged = add_camera_ids(merged)
    
    return merged


SYSTEM_PROMPT = """You are an expert cinematographer and film director AI. Your task is to analyze a storyboard script and generate additional camera instructions for each shot to enhance the storytelling.

**YOUR GOAL:**
For each shot in the storyboard, generate 0-3 additional camera instructions that complement the director's main camera instruction. These additional cameras should:
- Focus on character actions, emotions, or important objects
- Provide alternative angles that enhance the narrative
- Capture details that the director's main shot might miss
- Add visual variety and cinematic depth to the scene

**INPUTS YOU WILL RECEIVE:**
1. **Storyboard Outline**: The high-level scene and shot descriptions showing the story flow.
2. **Shot Details**: For each shot, you'll see:
   - Character actions (only non-idle actions are shown)
   - The director's camera instruction (this is the primary shot for narrative/storytelling)

**CAMERA INSTRUCTION PARAMETERS:**
Each camera instruction must include:
- `focus_on_ids`: List of asset IDs (characters/objects) the camera focuses on
- `angle`: One of "eye-level", "high angle", "low angle"
- `distance`: One of "close-up", "medium shot", "long shot"
- `movement`: One of "static", "pan", "orbit", "push in", "push out", "zoom in", "zoom out", "tracking"
- `direction`: Optional, one of "left", "right", "up", "down" (only needed for pan, orbit movements)
- `description`: A brief description of the shot purpose

**GUIDELINES:**
1. Generate 0-3 additional camera instructions per shot (0 if the director's shot is sufficient)
2. Do NOT duplicate the director's camera instruction
3. Focus additional cameras on:
   - Reaction shots of other characters
   - Close-ups of important objects or actions
   - Establishing shots or cutaways
   - Character expressions during emotional moments
4. Use asset IDs exactly as provided in the input
5. Vary the angles, distances, and movements to create visual interest

**OUTPUT FORMAT:**
Output a JSON object with the following structure:
```json
{
  "shots": [
    {
      "scene_id": 1,
      "shot_id": 1,
      "additional_camera_instructions": [
        {
          "focus_on_ids": ["asset_id"],
          "angle": "eye-level",
          "distance": "close-up",
          "movement": "static",
          "direction": null,
          "description": "Close-up on character's reaction"
        }
      ]
    }
  ]
}
```

Generate additional camera instructions for ALL shots in the input. Your output must be valid JSON only."""


def generate_additional_camera_instruction(
    anyllm_api_key=None,
    anyllm_api_base=None,
    anyllm_provider="gemini",
    storyboard_script=None,
    reasoning_model="gemini-3.1-pro-preview",
    reasoning_effort="high",
    max_retries=3
) -> Dict[str, Any]:
    """Generate additional camera instructions for each shot in the storyboard.
    
    Args:
        anyllm_api_key: API key for authentication.
        anyllm_api_base: Optional base URL for the API service.
        anyllm_provider: LLM provider (default: "gemini").
        storyboard_script: The storyboard script containing all scenes and shots.
        reasoning_model: The model name to invoke.
        reasoning_effort: Controls thinking capability ("low", "medium", "high").
        max_retries: Maximum number of retry attempts on failure.
        
    Returns:
        The storyboard script with additional camera instructions merged in,
        or None if generation fails.
    """
    if not isinstance(storyboard_script, dict):
        print("Error: storyboard_script must be a dict")
        return None
    
    # Extract relevant data for the prompt
    prompt_data = extract_prompt_data(storyboard_script)
    
    # Create the Markdown formatted user prompt
    user_prompt = create_additional_camera_prompt(prompt_data)
    
    # Build messages for any-llm
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    
    for attempt in range(max_retries):
        try:
            response = completion(
                api_key=anyllm_api_key,
                api_base=anyllm_api_base,
                provider=anyllm_provider,
                model=reasoning_model,
                reasoning_effort=reasoning_effort,
                messages=messages,
                response_format=AdditionalCameraInstructionsOutput,
            )
            gc.collect()
            result = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error generating content (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                backoff_time = 2 * (2 ** attempt)
                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                continue
            return None
        
        # Validate schema
        try:
            AdditionalCameraInstructionsOutput.model_validate(result)
        except Exception as e:
            print(f"Error validating response schema (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                continue
            return None
        
        # Validate output
        if not validate_output(storyboard_script, result):
            print(f"Error: Output validation failed (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                continue
            return None
        
        # Merge additional camera instructions into storyboard
        print("Successfully generated additional camera instructions")
        merged_result = merge_camera_instructions(storyboard_script, result)
        return merged_result
    
    return None
