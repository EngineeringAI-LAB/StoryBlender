"""Animation selection using LLM to match story actions to animations from a database."""

import math
import os
import io
import time
import base64
import mimetypes
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import base64
import tempfile
import urllib.request
import uuid
import warnings
import threading
import time

import requests
from requests.exceptions import RequestException, ProxyError, ConnectionError
from pydantic import BaseModel
try:
    from ..llm_completion import completion
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from llm_completion import completion

import nest_asyncio
nest_asyncio.apply()

import gc

try:
    from .restore_texture import restore_textures
except ImportError:
    from restore_texture import restore_textures

try:
    from .rigging import ensure_valid_rig_task, is_rig_task_valid, MeshyRiggingAPIError
except ImportError:
    from rigging import ensure_valid_rig_task, is_rig_task_valid, MeshyRiggingAPIError

warnings.filterwarnings("ignore", category=UserWarning, module="moviepy")

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ANIMATION_CSV = os.path.join(SCRIPT_DIR, "Meshy_animation.csv")


class MeshyAnimationAPIError(RuntimeError):
    """Raised when the Meshy Animation API returns an error or the task fails."""


class CategorySelection(BaseModel):
    """Schema for category selection output."""
    category: str


class CandidateSelection(BaseModel):
    """Schema for candidate animation selection output."""
    candidates: List[str]


class FinalAnimationSelection(BaseModel):
    """Schema for final animation selection output."""
    selected_animation: str
    matched: bool


class GenderSelection(BaseModel):
    """Schema for character gender selection output."""
    gender: str  # "male" or "female"


class AnimationDatabase:
    """Manages the animation database loaded from CSV."""
    
    def __init__(self, csv_path: str = DEFAULT_ANIMATION_CSV):
        self.csv_path = csv_path
        self.animations: List[Dict[str, str]] = []
        self.categories: List[str] = []
        self.animations_by_category: Dict[str, List[Dict[str, str]]] = {}
        self.animation_by_name: Dict[str, Dict[str, str]] = {}
        self._load_database()
    
    def _load_database(self):
        """Load and parse the animation CSV file."""
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.animations.append(row)
                category = row.get('Category', '')
                name = row.get('Name', '')
                
                if category not in self.animations_by_category:
                    self.animations_by_category[category] = []
                self.animations_by_category[category].append(row)
                
                self.animation_by_name[name] = row
        
        self.categories = sorted(list(self.animations_by_category.keys()))
    
    def get_categories(self) -> List[str]:
        """Return all available categories."""
        return self.categories
    
    def get_animations_in_category(self, category: str) -> List[Dict[str, str]]:
        """Return all animations in a given category."""
        return self.animations_by_category.get(category, [])
    
    def get_animation_by_name(self, name: str) -> Optional[Dict[str, str]]:
        """Return animation data by name."""
        return self.animation_by_name.get(name)
    
    def is_valid_category(self, category: str) -> bool:
        """Check if a category exists."""
        return category in self.animations_by_category
    
    def is_valid_animation_name(self, name: str) -> bool:
        """Check if an animation name exists."""
        return name in self.animation_by_name


def gif_to_mp4(url_or_path: str) -> str:
    """Convert a GIF file to MP4 and return the base64 data URL.
    
    Args:
        url_or_path: Either a web URL or a local path to a GIF file.
        
    Returns:
        Base64 encoded data URL of the converted MP4 file.
    """
    from moviepy import VideoFileClip
    
    unique_id = uuid.uuid4().hex
    temp_dir = tempfile.gettempdir()
    temp_gif_path = os.path.join(temp_dir, f"temp_gif_{unique_id}.gif")
    temp_mp4_path = os.path.join(temp_dir, f"temp_mp4_{unique_id}.mp4")
    
    try:
        # Download or copy the GIF to a temporary location
        if url_or_path.startswith(('http://', 'https://')):
            # Download with retry logic
            max_retries = 3
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    urllib.request.urlretrieve(url_or_path, temp_gif_path)
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise  # Last attempt, re-raise the exception
                    print(f"Download attempt {attempt + 1}/{max_retries} failed: {e}")
                    time.sleep(retry_delay)
        else:
            if not os.path.isfile(url_or_path):
                raise FileNotFoundError(f"File not found: {url_or_path}")
            with open(url_or_path, 'rb') as src, open(temp_gif_path, 'wb') as dst:
                dst.write(src.read())
        
        # Convert GIF to MP4 using moviepy
        clip = VideoFileClip(temp_gif_path)
        clip.write_videofile(temp_mp4_path, codec='libx264', audio=False, logger=None)
        clip.close()
        
        # Read the MP4 file and encode to base64
        with open(temp_mp4_path, 'rb') as f:
            mp4_data = f.read()
        
        base64_data = base64.b64encode(mp4_data).decode('utf-8')
        return f"data:video/mp4;base64,{base64_data}"
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_gif_path):
            os.remove(temp_gif_path)
        if os.path.exists(temp_mp4_path):
            os.remove(temp_mp4_path)


def select_category(
    action_description: str,
    categories: List[str],
    asset_id: str = "",
    asset_description: str = "",
    anyllm_api_key: str = None,
    anyllm_api_base: str = None,
    anyllm_provider: str = "gemini",
    vision_model: str = "gemini-2.5-flash",
    max_retries: int = 3
) -> Optional[str]:
    """Select the most appropriate animation category for an action.
    
    Args:
        action_description: The action to find an animation for.
        categories: List of available category names.
        asset_id: The character/asset identifier (can indicate gender, character type, etc.).
        asset_description: Description of the character's appearance and traits.
        anyllm_api_key: API key for the LLM.
        anyllm_api_base: Optional base URL for the API.
        anyllm_provider: LLM provider (default: "gemini").
        vision_model: Model to use for selection.
        max_retries: Maximum retry attempts.
        
    Returns:
        Selected category name, or None if failed.
    """
    system_prompt = """You are an expert at matching character actions to animation categories.
Given a character's identity, description, an action description, and a list of available animation categories, select the SINGLE most appropriate category.
Consider the character's gender, role, and appearance when selecting an animation style.
Your output must be valid JSON with the selected category name exactly as it appears in the list."""

    # Representative examples for each category
    category_examples = {
        "Idle": ["Idle", "Chair_Sit_Idle_F", "Mirror_Viewing"],
        "Walking": ["Casual_Walk", "Confident_Walk", "Sneaky_Walk"],
        "LookingAround": ["Alert", "Long_Breathe_and_Look_Around", "Walk_Slowly_and_Look_Around"],
        "AttackingwithWeapon": ["Archery_Shot", "Sword_Judgment", "Run_and_Shoot"],
        "Running": ["Run_02", "RunFast", "Hello_Run"],
        "GettingHit": ["Slap_Reaction", "Face_Punch_Reaction", "Gunshot_Reaction"],
        "Dying": ["Dead", "Shot_and_Fall_Backward", "Knock_Down"],
        "Transitioning": ["Stand_to_Sit_Transition_M", "Stand_Dodge", "Stand_To_Side_Lying"],
        "Acting": ["Victory_Cheer", "Happy_Sway_Standing", "Finger_Wag_No"],
        "Dancing": ["FunnyDancing_01", "Hip_Hop_Dance", "ymca_dance"],
        "Interacting": ["Big_Wave_Hello", "Talk_with_Left_Hand_Raised", "Stand_Talking_Angry"],
        "Punching": ["Boxing_Practice", "Kung_Fu_Punch", "Roundhouse_Kick"],
        "CastingSpell": ["Charged_Spell_Cast", "Charged_Ground_Slam", "mage_soell_cast"],
        "Blocking": ["Block1", "Sword_Parry", "Two_Handed_Parry"],
        "Pushing": ["Step_Forward_and_Push", "Push_Forward_and_Stop", "Push_and_Walk_Forward"],
        "Sleeping": ["Sleep_on_Desk", "Sleep_Normally", "Toss_and_Turn"],
        "PickingUpItem": ["Male_Bend_Over_Pick_Up", "Collect_Object", "Pull_Radish"],
        "WorkingOut": ["air_squat", "push_up", "jumping_jacks"],
        "Drinking": ["Stand_and_Drink", "Sit_and_Drink"],
        "VaultingOverObstacle": ["Parkour_Vault", "Roll_Behind_Cover", "Unarmed_Vault"],
        "Climbing": ["Fast_Ladder_Climb", "Climb_Stairs", "climbing_up_wall"],
        "PerformingStunt": ["Backflip", "Wall_Flip", "One_Arm_Handstand"],
        "Jumping": ["Regular_Jump", "Back_Jump", "Run_and_Jump"],
        "HangingfromLedge": ["Rope_Hang_Idle", "Bar_Hang_Idle", "Fall_from_Bar"],
        "FallingFreely": ["Leap_of_Faith", "Fall1", "Dive_Down_and_Land"],
        "CrouchWalking": ["Crouch_Walk_with_Torch", "Cautious_Crouch_Walk_Forward", "Walk_Left_with_Gun"],
        "Swimming": ["Swim_Idle", "Swim_Forward", "swimming_to_edge"],
        "TurningAround": ["Idle_Turn_Left", "Walk_Turn_Right", "Run_Turn_Left"],
    }
    
    # Build categories list with examples
    categories_lines = []
    for cat in categories:
        examples = category_examples.get(cat, [])
        if examples:
            examples_str = ", ".join(examples[:3])
            categories_lines.append(f"- {cat} (e.g., {examples_str})")
        else:
            categories_lines.append(f"- {cat}")
    categories_list = "\n".join(categories_lines)
    
    user_prompt = f"""Character ID: "{asset_id}"
Character description: "{asset_description}"
Action: "{action_description}"

Available animation categories:
{categories_list}

Select the most appropriate category for this character and action. The category must be EXACTLY one from the list above (without the examples)."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    for attempt in range(max_retries):
        try:
            response = completion(
                api_key=anyllm_api_key,
                api_base=anyllm_api_base,
                provider=anyllm_provider,
                model=vision_model,
                messages=messages,
                response_format=CategorySelection,
            )
            gc.collect()
            # Handle generator response in threaded context
            if hasattr(response, '__iter__') and not hasattr(response, 'choices'):
                chunks = list(response)
                if chunks:
                    response = chunks[-1]
            result = json.loads(response.choices[0].message.content)
            selected_category = result.get("category", "")
            
            # Validate the category
            if selected_category in categories:
                return selected_category
            else:
                print(f"Invalid category '{selected_category}' (attempt {attempt + 1}/{max_retries})")
                
        except Exception as e:
            print(f"Error selecting category (attempt {attempt + 1}/{max_retries}): {e}")
    
    return None


def select_candidates(
    action_description: str,
    animations: List[Dict[str, str]],
    asset_id: str = "",
    asset_description: str = "",
    num_candidates: int = 3,
    anyllm_api_key: str = None,
    anyllm_api_base: str = None,
    anyllm_provider: str = "gemini",
    vision_model: str = "gemini-2.5-flash",
    max_retries: int = 3
) -> Optional[List[str]]:
    """Select candidate animations from a category based on their names.
    
    Args:
        action_description: The action to find animations for.
        animations: List of animation dicts with 'Name' key.
        asset_id: The character/asset identifier (can indicate gender, character type, etc.).
        asset_description: Description of the character's appearance and traits.
        num_candidates: Number of candidates to select.
        anyllm_api_key: API key for the LLM.
        anyllm_api_base: Optional base URL for the API.
        anyllm_provider: LLM provider (default: "gemini").
        vision_model: Model to use for selection.
        max_retries: Maximum retry attempts.
        
    Returns:
        List of candidate animation names, or None if failed.
    """
    system_prompt = f"""You are an expert at selecting animations for character actions.
Given a character's identity, description, an action description, and a list of animation names, select the {num_candidates} most appropriate animations.
Consider the character's gender, role, and appearance when selecting animations. Prefer simple, concise actions than complex actions. Then actions you selected should fit the character and the plot.
Your output must be valid JSON with the animation names exactly as they appear in the list."""

    animation_names = [anim.get('Name', '') for anim in animations]
    animations_list = "\n".join(f"- {name}" for name in animation_names)
    
    user_prompt = f"""Character ID: "{asset_id}"
Character description: "{asset_description}"
Action: "{action_description}"

Available animations in this category:
{animations_list}

Select the {num_candidates} most appropriate animations for this character and action. 
The animation names must be EXACTLY as they appear in the list above.
If there are fewer than {num_candidates} animations available, select all of them."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    for attempt in range(max_retries):
        try:
            response = completion(
                api_key=anyllm_api_key,
                api_base=anyllm_api_base,
                provider=anyllm_provider,
                model=vision_model,
                messages=messages,
                response_format=CandidateSelection,
            )
            gc.collect()
            # Handle generator response in threaded context
            if hasattr(response, '__iter__') and not hasattr(response, 'choices'):
                chunks = list(response)
                if chunks:
                    response = chunks[-1]
            result = json.loads(response.choices[0].message.content)
            candidates = result.get("candidates", [])
            
            # Validate all candidates
            valid_candidates = [c for c in candidates if c in animation_names]
            
            if len(valid_candidates) >= 1:
                return valid_candidates[:num_candidates]
            else:
                print(f"No valid candidates found (attempt {attempt + 1}/{max_retries})")
                
        except Exception as e:
            print(f"Error selecting candidates (attempt {attempt + 1}/{max_retries}): {e}")
    
    return None


def select_final_animation(
    action_description: str,
    candidates: List[Dict[str, str]],
    asset_id: str = "",
    asset_description: str = "",
    anyllm_api_key: str = None,
    anyllm_api_base: str = None,
    anyllm_provider: str = "gemini",
    vision_model: str = "gemini-2.5-flash",
    max_retries: int = 3
) -> Optional[Tuple[str, bool]]:
    """Select the final animation by analyzing GIF previews.
    
    Args:
        action_description: The action to find an animation for.
        candidates: List of candidate animation dicts with 'Name' and 'URL'.
        asset_id: The character/asset identifier (can indicate gender, character type, etc.).
        asset_description: Description of the character's appearance and traits.
        anyllm_api_key: API key for the LLM.
        anyllm_api_base: Optional base URL for the API.
        anyllm_provider: LLM provider (default: "gemini").
        vision_model: Model to use for selection.
        max_retries: Maximum retry attempts.
        
    Returns:
        Tuple of (selected animation name, matched), or None if failed.
        matched is True if the animation genuinely fits the action, False if it is a fallback (e.g. Idle).
    """
    system_prompt = """You are an expert at analyzing animations for character actions.
You will see preview videos of candidate animations. Select the ONE that best matches the character and action description.
Consider the character's gender, role, and appearance when selecting the animation.
Prefer simple, concise actions than complex actions. Then actions you selected should fit the character and the plot.

Special Rules (even they are not in the candidate animations):
- For walking actions, always choose "walking_2".
- For running actions, always choose "Run_03".
- If there are no suitable animations, use "Idle_12" for female character, "Idle_11" for male character as the selected animation.

For the `matched` field:
- Set matched=true if the selected animation genuinely matches the action description.
- Set matched=false if you are falling back to an Idle animation (Idle_11 or Idle_12) because no suitable animation was found.
Your output must be valid JSON with the selected animation name exactly as provided."""

    # Build multimodal content with GIF videos
    content_parts = [
        {
            "type": "text",
            "text": f'Character ID: "{asset_id}"\nCharacter description: "{asset_description}"\nAction: "{action_description}"\n\nHere are the candidate animations:'
        }
    ]
    
    candidate_names = []
    for i, anim in enumerate(candidates, 1):
        name = anim.get('Name', '')
        url = anim.get('URL', '')
        candidate_names.append(name)
        
        content_parts.append({
            "type": "text",
            "text": f"\n{i}. Animation name: \"{name}\""
        })
        
        if url:
            try:
                mp4_data_url = gif_to_mp4(url)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": mp4_data_url}
                })
            except Exception as e:
                print(f"Warning: Failed to load GIF for {name}: {e}")
                content_parts.append({
                    "type": "text",
                    "text": f"(Preview unavailable for {name})"
                })
    
    content_parts.append({
        "type": "text",
        "text": f"\n\nSelect the animation that best matches the character '{asset_id}' ({asset_description}) performing the action: \"{action_description}\". The name must be exactly one of: {candidate_names}"
    })
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_parts}
    ]
    
    for attempt in range(max_retries):
        try:
            response = completion(
                api_key=anyllm_api_key,
                api_base=anyllm_api_base,
                provider=anyllm_provider,
                model=vision_model,
                messages=messages,
                response_format=FinalAnimationSelection,
            )
            gc.collect()
            # Handle generator response in threaded context
            if hasattr(response, '__iter__') and not hasattr(response, 'choices'):
                chunks = list(response)
                if chunks:
                    response = chunks[-1]
            result = json.loads(response.choices[0].message.content)
            selected = result.get("selected_animation", "")
            matched = bool(result.get("matched", True))
            
            # Validate the selection (includes special-rule overrides from system prompt)
            allowed_overrides = {"Idle_11", "Idle_12", "walking_2", "Run_03"}
            if selected in candidate_names or selected in allowed_overrides:
                return selected, matched
            else:
                print(f"Invalid selection '{selected}' (attempt {attempt + 1}/{max_retries})")
                
        except Exception as e:
            print(f"Error selecting final animation (attempt {attempt + 1}/{max_retries}): {e}")
    
    return None


def select_animation_for_action(
    action_description: str,
    database: AnimationDatabase,
    asset_id: str = "",
    asset_description: str = "",
    anyllm_api_key: str = None,
    anyllm_api_base: str = None,
    anyllm_provider: str = "gemini",
    vision_model: str = "gemini-2.5-flash",
    num_candidates: int = 3,
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """Complete pipeline to select an animation for an action description.
    
    Args:
        action_description: The action to find an animation for.
        database: AnimationDatabase instance.
        asset_id: The character/asset identifier (can indicate gender, character type, etc.).
        asset_description: Description of the character's appearance and traits.
        anyllm_api_key: API key for the LLM.
        anyllm_api_base: Optional base URL for the API.
        anyllm_provider: LLM provider (default: "gemini").
        vision_model: Model to use for selection.
        num_candidates: Number of candidates for final selection.
        max_retries: Maximum retry attempts per step.
        
    Returns:
        Dict with 'action_id' and 'action_name', or None if failed.
    """    
    # Step 1: Select category
    category = select_category(
        action_description=action_description,
        categories=database.get_categories(),
        asset_id=asset_id,
        asset_description=asset_description,
        anyllm_api_key=anyllm_api_key,
        anyllm_api_base=anyllm_api_base,
        anyllm_provider=anyllm_provider,
        vision_model=vision_model,
        max_retries=max_retries
    )
    
    if not category:
        print(f"Failed to select category for: {action_description}")
        return None
        
    # Step 2: Get animations in category and select candidates
    animations_in_category = database.get_animations_in_category(category)
    
    if not animations_in_category:
        print(f"No animations found in category: {category}")
        return None
    
    # If only 1 animation, use it directly
    if len(animations_in_category) == 1:
        anim = animations_in_category[0]
        return {
            "action_id": anim.get("ID"),
            "action_name": anim.get("Name"),
            "matched": True
        }
    
    candidate_names = select_candidates(
        action_description=action_description,
        animations=animations_in_category,
        asset_id=asset_id,
        asset_description=asset_description,
        num_candidates=num_candidates,
        anyllm_api_key=anyllm_api_key,
        anyllm_api_base=anyllm_api_base,
        anyllm_provider=anyllm_provider,
        vision_model=vision_model,
        max_retries=max_retries
    )
    
    if not candidate_names:
        print(f"Failed to select candidates for: {action_description}")
        return None    
    # If only 1 candidate, use it directly
    if len(candidate_names) == 1:
        anim = database.get_animation_by_name(candidate_names[0])
        if anim:
            return {
                "action_id": anim.get("ID"),
                "action_name": anim.get("Name"),
                "matched": True
            }
        return None
    
    # Step 3: Final selection using GIF preview
    candidate_animations = [
        database.get_animation_by_name(name) 
        for name in candidate_names 
        if database.get_animation_by_name(name)
    ]
    
    final_result = select_final_animation(
        action_description=action_description,
        candidates=candidate_animations,
        asset_id=asset_id,
        asset_description=asset_description,
        anyllm_api_key=anyllm_api_key,
        anyllm_api_base=anyllm_api_base,
        anyllm_provider=anyllm_provider,
        vision_model=vision_model,
        max_retries=max_retries
    )
    
    if final_result:
        final_name, matched = final_result
    else:
        # Fallback to first candidate if visual selection fails
        print("Visual selection failed, using first candidate as fallback")
        final_name = candidate_names[0]
        matched = final_name not in {"Idle_11", "Idle_12"}
    
    final_anim = database.get_animation_by_name(final_name)
    if final_anim:
        return {
            "action_id": final_anim.get("ID"),
            "action_name": final_anim.get("Name"),
            "matched": matched
        }
    
    return None


def determine_character_gender(
    asset_id: str,
    asset_description: str,
    anyllm_api_key: str = None,
    anyllm_api_base: str = None,
    anyllm_provider: str = "gemini",
    vision_model: str = "gemini-2.5-flash",
    max_retries: int = 3
) -> str:
    """Determine if a character is male or female based on name and description.
    
    Args:
        asset_id: The character/asset identifier (name).
        asset_description: Description of the character's appearance and traits.
        anyllm_api_key: API key for the LLM.
        anyllm_api_base: Optional base URL for the API.
        anyllm_provider: LLM provider (default: "gemini").
        vision_model: Model to use for selection.
        max_retries: Maximum retry attempts.
        
    Returns:
        "male" or "female" - must choose one.
    """
    system_prompt = """You are an expert at analyzing character descriptions to determine gender.
Given a character's name and description, determine if the character is male or female.
You MUST choose one - either "male" or "female". There is no neutral option.
If the description is ambiguous, make your best guess based on the name and any available context.
Your output must be valid JSON with the gender field set to exactly "male" or "female"."""

    user_prompt = f"""Character name: "{asset_id}"
Character description: "{asset_description}"

Based on the name and description above, is this character male or female?
You must choose exactly one: "male" or "female"."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    for attempt in range(max_retries):
        try:
            response = completion(
                api_key=anyllm_api_key,
                api_base=anyllm_api_base,
                provider=anyllm_provider,
                model=vision_model,
                messages=messages,
                response_format=GenderSelection,
            )
            gc.collect()
            # Handle generator response in threaded context
            if hasattr(response, '__iter__') and not hasattr(response, 'choices'):
                chunks = list(response)
                if chunks:
                    response = chunks[-1]
            result = json.loads(response.choices[0].message.content)
            gender = result.get("gender", "").lower().strip()
            
            # Validate the gender
            if gender in ["male", "female"]:
                return gender
            else:
                print(f"Invalid gender '{gender}' (attempt {attempt + 1}/{max_retries}), defaulting to 'male'")
                
        except Exception as e:
            print(f"Error determining gender (attempt {attempt + 1}/{max_retries}): {e}")
    
    # Default to male if all retries fail
    print(f"Failed to determine gender for {asset_id}, defaulting to 'male'")
    return "male"


def _process_single_action(
    cache_key: str,
    action_description: str,
    asset_id: str,
    asset_description: str,
    database: AnimationDatabase,
    anyllm_api_key: str,
    anyllm_api_base: str,
    anyllm_provider: str,
    vision_model: str,
    num_candidates: int,
    max_retries: int,
    progress_lock: threading.Lock,
    progress_counter: List[int],
    total_unique: int
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Process a single action for a character (used for parallel execution).
    
    Args:
        cache_key: Unique key combining asset_id and action_description.
        action_description: The action to find an animation for.
        asset_id: The character/asset identifier (can indicate gender, character type, etc.).
        asset_description: Description of the character's appearance and traits.
        database: AnimationDatabase instance.
        anyllm_api_key: API key for the LLM.
        anyllm_api_base: Optional base URL for the API.
        anyllm_provider: LLM provider (default: "gemini").
        vision_model: Model to use for selection.
        num_candidates: Number of candidates for final selection.
        max_retries: Maximum retry attempts per step.
        progress_lock: Thread lock for progress counter.
        progress_counter: Shared progress counter.
        total_unique: Total number of unique actions to process.
    
    Returns:
        Tuple of (cache_key, selection_result)
    """
    with progress_lock:
        progress_counter[0] += 1
        current = progress_counter[0]
    
    selection = select_animation_for_action(
        action_description=action_description,
        database=database,
        asset_id=asset_id,
        asset_description=asset_description,
        anyllm_api_key=anyllm_api_key,
        anyllm_api_base=anyllm_api_base,
        anyllm_provider=anyllm_provider,
        vision_model=vision_model,
        num_candidates=num_candidates,
        max_retries=max_retries
    )
    
    return cache_key, selection


def generate_animation_selection(
    anyllm_api_key: str = None,
    anyllm_api_base: str = None,
    anyllm_provider: str = "gemini",
    vision_model: str = "gemini-3-flash-preview",
    storyboard_script: Dict[str, Any] = None,
    animation_csv_path: str = DEFAULT_ANIMATION_CSV,
    num_candidates: int = 3,
    max_retries: int = 3,
    max_concurrent: int = 10
) -> Optional[Dict[str, Any]]:
    """Generate animation selections for all character actions in a storyboard script.
    
    For each character_action in shot_details, this function:
    1. Selects the best animation category based on action_description
    2. Selects candidate animations from that category
    3. Uses visual analysis of GIF previews to select the final animation
    4. Updates the character_action with action_id and action_name
    
    Args:
        anyllm_api_key: API key for the LLM.
        anyllm_api_base: Optional base URL for the API.
        anyllm_provider: LLM provider (default: "gemini").
        vision_model: Model to use for selection.
        storyboard_script: The full storyboard script containing shot_details.
        animation_csv_path: Path to the animation database CSV.
        num_candidates: Number of candidates for final selection.
        max_retries: Maximum retry attempts per step.
        max_concurrent: Maximum number of concurrent LLM calls.
        
    Returns:
        Updated storyboard_script with action_id and action_name added to
        each character_action, or None on failure.
    """
    if not isinstance(storyboard_script, dict):
        print("Error: storyboard_script must be a dict")
        return None
    
    shot_details = storyboard_script.get("shot_details", [])
    if not isinstance(shot_details, list) or not shot_details:
        print("Error: No shot_details found in storyboard_script")
        return None
    
    # Load animation database
    try:
        database = AnimationDatabase(animation_csv_path)
        print(f"Loaded animation database with {len(database.animations)} animations in {len(database.categories)} categories")
    except Exception as e:
        print(f"Error loading animation database: {e}")
        return None
    
    # Create a deep copy to avoid modifying the original
    result = deepcopy(storyboard_script)
    result_shot_details = result.get("shot_details", [])
    
    # Build mapping from asset_id to description and asset_type from asset_sheet
    asset_descriptions: Dict[str, str] = {}
    asset_types: Dict[str, str] = {}
    asset_sheet = storyboard_script.get("asset_sheet", [])
    for asset in asset_sheet:
        aid = asset.get("asset_id", "")
        desc = asset.get("description", "")
        asset_type = asset.get("asset_type", "")
        if aid:
            asset_descriptions[aid] = desc
            asset_types[aid] = asset_type
    
    # Build mapping from scene_id to list of asset_ids in that scene
    scene_details = storyboard_script.get("scene_details", [])
    scene_asset_ids: Dict[int, List[str]] = {}
    for scene_detail in scene_details:
        scene_id = scene_detail.get("scene_id")
        setup_data = scene_detail.get("scene_setup", {})
        asset_ids = setup_data.get("asset_ids", [])
        if scene_id is not None:
            scene_asset_ids[scene_id] = asset_ids
    
    # Collect all unique (asset_id, action_description) pairs first
    # cache_key -> (asset_id, asset_description, action_description)
    unique_actions: Dict[str, Tuple[str, str, str]] = {}
    for shot in result_shot_details:
        character_actions = shot.get("character_actions", [])
        if not character_actions:
            continue
        for action in character_actions:
            action_description = action.get("action_description", "")
            asset_id = action.get("asset_id", "")
            if action_description:
                # Cache key includes both asset_id and action_description
                # since same action for different characters may need different animations
                cache_key = f"{asset_id}::{action_description.lower().strip()}"
                if cache_key not in unique_actions:
                    asset_desc = asset_descriptions.get(asset_id, "")
                    unique_actions[cache_key] = (asset_id, asset_desc, action_description)
    
    total_unique = len(unique_actions)
    print(f"\nProcessing {total_unique} unique (asset_id, action) pairs in parallel (max_concurrent={max_concurrent})...")
    
    # Process unique actions in parallel
    action_cache: Dict[str, Dict[str, Any]] = {}
    progress_lock = threading.Lock()
    progress_counter = [0]  # Use list to allow mutation in closure
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(
                _process_single_action,
                cache_key,
                action_desc,
                asset_id,
                asset_desc,
                database,
                anyllm_api_key,
                anyllm_api_base,
                anyllm_provider,
                vision_model,
                num_candidates,
                max_retries,
                progress_lock,
                progress_counter,
                total_unique
            ): cache_key
            for cache_key, (asset_id, asset_desc, action_desc) in unique_actions.items()
        }
        
        for future in as_completed(futures):
            try:
                cache_key, selection = future.result()
                if selection:
                    action_cache[cache_key] = selection
                else:
                    print(f"Warning: Failed to select animation for cache_key: {cache_key}")
            except Exception as e:
                cache_key = futures[future]
                print(f"Error processing action {cache_key}: {e}")
    
    # Apply results to all character_actions
    applied_count = 0
    for shot in result_shot_details:
        character_actions = shot.get("character_actions", [])
        if not character_actions:
            continue
        for action in character_actions:
            action_description = action.get("action_description", "")
            asset_id = action.get("asset_id", "")
            if not action_description:
                continue
            
            cache_key = f"{asset_id}::{action_description.lower().strip()}"
            if cache_key in action_cache:
                action["action_id"] = action_cache[cache_key]["action_id"]
                action["action_name"] = action_cache[cache_key]["action_name"]
                action["matched"] = action_cache[cache_key].get("matched", True)
                applied_count += 1
            else:
                action["action_id"] = None
                action["action_name"] = None
                action["matched"] = False
    
    print(f"\nCompleted animation selection: {len(action_cache)}/{total_unique} unique (asset_id, action) pairs resolved, applied to {applied_count} character_actions")
    
    # Handle idle characters - characters in scene but not in character_actions
    # Cache gender results to avoid redundant LLM calls
    gender_cache: Dict[str, str] = {}
    idle_count = 0
    
    print("\nProcessing idle characters (characters in scene without actions)...")
    
    for shot in result_shot_details:
        scene_id = shot.get("scene_id")
        
        # Get all asset_ids in this scene
        all_scene_assets = scene_asset_ids.get(scene_id, [])
        
        # Get all character asset_ids in this scene (filter by asset_type == "character")
        characters_in_scene = [
            aid for aid in all_scene_assets 
            if asset_types.get(aid) == "character"
        ]
        
        # Get character asset_ids that already have actions in this shot
        character_actions = shot.get("character_actions", [])
        if character_actions is None:
            character_actions = []
        characters_with_actions = {
            action.get("asset_id") for action in character_actions
        }
        
        # Find idle characters (in scene but no action)
        idle_characters = [
            aid for aid in characters_in_scene 
            if aid not in characters_with_actions
        ]
        
        # Process each idle character
        for asset_id in idle_characters:
            asset_desc = asset_descriptions.get(asset_id, "")
            
            # Determine gender (use cache if available)
            if asset_id in gender_cache:
                gender = gender_cache[asset_id]
            else:
                gender = determine_character_gender(
                    asset_id=asset_id,
                    asset_description=asset_desc,
                    anyllm_api_key=anyllm_api_key,
                    anyllm_api_base=anyllm_api_base,
                    anyllm_provider=anyllm_provider,
                    vision_model=vision_model,
                    max_retries=max_retries
                )
                gender_cache[asset_id] = gender
                print(f"  Determined gender for '{asset_id}': {gender}")
            
            # Apply idle animation based on gender
            # Male: Idle_11 (id 251), Female: Idle_12 (id 252)
            if gender == "female":
                action_id = 252
                action_name = "Idle_12"
            else:
                action_id = 251
                action_name = "Idle_11"
            
            # Create idle action entry
            idle_action = {
                "asset_id": asset_id,
                "action_description": "idle (default action)",
                "action_id": action_id,
                "action_name": action_name,
                "matched": False,
            }
            
            # Add to character_actions
            if shot.get("character_actions") is None:
                shot["character_actions"] = []
            shot["character_actions"].append(idle_action)
            idle_count += 1
            print(f"  Added idle animation for '{asset_id}' in scene {scene_id}, shot {shot.get('shot_id')}: {action_name} (id: {action_id})")
    
    print(f"\nAdded {idle_count} idle animations for characters without actions")
    
    return result


def merge_animation_selection(
    storyboard_script: Dict[str, Any],
    animation_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge animation selection results back into the storyboard script.
    
    This function is provided for compatibility but generate_animation_selection
    already returns the merged result.
    
    Args:
        storyboard_script: The original storyboard script.
        animation_result: The result from generate_animation_selection.
        
    Returns:
        The animation_result if valid, otherwise the original script.
    """
    if animation_result is None:
        print("Animation selection failed, using original storyboard script")
        return storyboard_script
    return animation_result


def _download_file(url: str, output_path: str, session: Optional[requests.Session] = None) -> str:
    """
    Download a file from URL to local path.
    
    Args:
        url: URL to download from
        output_path: Local path to save the file
        session: Optional requests.Session to reuse connections
        
    Returns:
        Path to the downloaded file
    """
    sess = session or requests.Session()
    response = sess.get(url, timeout=300)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(response.content)
    
    return output_path


def apply_single_animation(
    rig_task_id: str,
    action_id: int,
    asset_id: str,
    action_name: str,
    output_dir: str,
    meshy_api_key: str,
    meshy_api_base: str,
    texture_folder: Optional[str] = None,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    Apply animation to a single rigged model and download the result.
    
    Args:
        rig_task_id: The rigging task ID from Meshy
        action_id: The animation action ID
        asset_id: The asset ID for naming
        action_name: The action name for naming
        output_dir: Directory to save downloaded files
        meshy_api_key: Meshy API key
        meshy_api_base: Meshy API base URL
        texture_folder: Optional path to texture folder for restoration
        session: Optional requests.Session
        
    Returns:
        Dict with animation info or error
    """
    print(f"Applying animation: {asset_id} + {action_name} (action_id: {action_id})")
    
    try:
        sess = session or requests.Session()
        headers = {"Authorization": f"Bearer {meshy_api_key}"}
        
        # Create animation task
        task_payload = {
            "rig_task_id": rig_task_id,
            "action_id": action_id,
        }
        
        # Create task with retry logic
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                create_resp = sess.post(
                    f"{meshy_api_base}/animations",
                    headers=headers,
                    json=task_payload,
                    timeout=300
                )
                create_resp.raise_for_status()
                create_data = create_resp.json()
                break
            except (RequestException, ProxyError, ConnectionError) as e:
                if attempt == max_retries - 1:
                    raise MeshyAnimationAPIError(f"Failed to create animation task after {max_retries} attempts: {e}")
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        task_id = create_data.get("result")
        if not task_id:
            raise MeshyAnimationAPIError("Create animation task response missing 'result'. Response: %r" % (create_data,))
        
        print(f"Created animation task: {task_id}")
        
        # Poll for task completion
        time.sleep(2.0)  # Initial delay
        
        start_time = time.time()
        task_url = f"{meshy_api_base}/animations/{task_id}"
        poll_interval = 5.0
        timeout = 20 * 60.0  # 20 minutes
        
        terminal_statuses = {"SUCCEEDED", "FAILED", "CANCELED"}
        
        while True:
            if time.time() - start_time > timeout:
                raise MeshyAnimationAPIError(
                    f"Timed out waiting for Meshy animation task {task_id} after {timeout} seconds"
                )
            
            # Poll task status
            poll_success = False
            for poll_attempt in range(3):
                try:
                    resp = sess.get(task_url, headers=headers, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    poll_success = True
                    break
                except requests.HTTPError as e:
                    if resp.status_code == 404:
                        time.sleep(poll_interval)
                        break
                    if poll_attempt == 2:
                        raise
                    print(f"Animation poll attempt {poll_attempt + 1}/3 failed: {e}. Retrying in 2s...")
                    time.sleep(2)
                except (RequestException, ProxyError, ConnectionError) as e:
                    if poll_attempt == 2:
                        raise MeshyAnimationAPIError(f"Failed to poll animation task status after 3 attempts: {e}")
                    print(f"Animation poll attempt {poll_attempt + 1}/3 failed: {e}. Retrying in 2s...")
                    time.sleep(2)
            
            if not poll_success:
                time.sleep(poll_interval)
                continue
            
            status = data.get("status")
            progress = data.get("progress", 0)
            print(f"Animation task {task_id} status: {status} ({progress}%)")
            
            if status in terminal_statuses:
                if status == "SUCCEEDED":
                    # Download the animation file
                    result = data.get("result", {})
                    animation_glb_url = result.get("animation_glb_url")
                    
                    if animation_glb_url:
                        output_filename = f"{asset_id}_{action_id}_{action_name}.glb"
                        raw_output_path = os.path.join(output_dir, f"raw_{output_filename}")
                        final_output_path = os.path.join(output_dir, output_filename)
                        
                        # Download the raw animation file
                        _download_file(animation_glb_url, raw_output_path, session)
                        print(f"Downloaded raw animation: {raw_output_path}")
                        
                        # Restore textures if texture folder is available
                        if texture_folder and os.path.exists(texture_folder):
                            try:
                                restore_textures(
                                    glb_path=raw_output_path,
                                    texture_dir=texture_folder,
                                    output_path=final_output_path,
                                )
                                print(f"Restored textures to animation: {final_output_path}")
                                animated_path = final_output_path
                            except Exception as e:
                                print(f"Warning: Failed to restore textures for {asset_id} + {action_name}: {e}")
                                print(f"Using raw animation file: {raw_output_path}")
                                animated_path = raw_output_path
                        else:
                            # If no texture folder, use raw file
                            print(f"No texture folder found, using raw animation: {raw_output_path}")
                            animated_path = raw_output_path
                        
                        return {
                            "asset_id": asset_id,
                            "action_id": action_id,
                            "action_name": action_name,
                            "animation_task_id": task_id,
                            "animated_path": animated_path,
                            "animation_url": animation_glb_url,
                        }
                    else:
                        raise MeshyAnimationAPIError("Animation task succeeded but no animation_glb_url found")
                
                # Handle failed/canceled tasks
                task_error = (data.get("task_error") or {}).get("message")
                raise MeshyAnimationAPIError(
                    f"Meshy animation task {task_id} ended with status {status}. Error: {task_error}"
                )
            
            time.sleep(poll_interval)
    
    except Exception as e:
        print(f"✗ Failed to apply animation for {asset_id} + {action_name}: {e}")
        return {
            "asset_id": asset_id,
            "action_id": action_id,
            "action_name": action_name,
            "animation_error": str(e),
        }


def animate_rigged_model(
    path_to_input_json: str,
    output_dir: str,
    meshy_api_key: Optional[str] = None,
    meshy_api_base: str = "https://api.meshy.ai/openapi/v1",
    max_concurrent: int = 10,
) -> Dict[str, Any]:
    """
    Apply animations to rigged models using Meshy's animation API.
    
    This function reads a JSON file containing shot_details and asset_sheet,
    extracts character actions with their action_id and action_name, looks up
    the rig_task_id for each character from the asset_sheet, and applies
    animations using the Meshy API for each combination of rig_task_id and action_id.
    
    Args:
        path_to_input_json: Path to the JSON file containing the story script with shot_details and asset_sheet
        output_dir: Directory to save animated GLB files. Files will be named as {asset_id}_{action_id}_{action_name}.glb
        meshy_api_key: Meshy API key. If not provided, read from env var MESHY_API_KEY.
        meshy_api_base: Base URL for Meshy API. Default: "https://api.meshy.ai/openapi/v1"
        max_concurrent: Maximum number of concurrent animation tasks
        
    Returns:
        Dict containing:
        - successful_animations: List of successful animation results
        - failed_animations: List of failed animation results
        - total_processed: Total number of animation combinations processed
        - updated_json: The updated JSON data with animated_model_path added to each character_action
        
    Example:
        >>> result = animate_rigged_model(
        ...     "./selected_animation_v1.json",
        ...     "./animated_models",
        ...     meshy_api_key="your_api_key"
        ... )
        >>> print(f"Successfully animated: {len(result['successful_animations'])} models")
        >>> # Save the updated JSON with animated paths
        >>> with open("animated_script.json", "w") as f:
        ...     json.dump(result["updated_json"], f, indent=2)
    """
    key = meshy_api_key or os.getenv("MESHY_API_KEY")
    if not key:
        raise MeshyAnimationAPIError(
            "Missing Meshy API key. Set MESHY_API_KEY env var or pass meshy_api_key argument."
        )
    
    # Load input JSON
    with open(path_to_input_json, "r") as f:
        input_data = json.load(f)
    
    shot_details = input_data.get("shot_details", [])
    asset_sheet = input_data.get("asset_sheet", [])
    
    # Determine which character assets are referenced by any character_action
    referenced_character_asset_ids = set()
    for shot in shot_details:
        for action in shot.get("character_actions", []) or []:
            aid = action.get("asset_id")
            if aid:
                referenced_character_asset_ids.add(aid)
    
    # Validate / refresh rig tasks for referenced character assets BEFORE animating.
    # If a rig_task_id is missing or expired (per Meshy API), re-rig the model
    # so animation calls don't fail. The rigged_models output_dir is derived
    # from the input json's directory to keep artifacts grouped.
    rigged_models_output_dir = os.path.join(os.path.dirname(path_to_input_json), "rigged_models")
    rigging_session = requests.Session()
    asset_sheet_changed = False
    for idx, asset in enumerate(asset_sheet):
        asset_id = asset.get("asset_id")
        if not asset_id or asset_id not in referenced_character_asset_ids:
            continue
        if asset.get("asset_type") and asset.get("asset_type") != "character":
            continue
        if not asset.get("main_file_path"):
            # Cannot re-rig without a source model; skip silently — it will
            # surface later as a missing rig_task_id warning.
            continue
        try:
            refreshed = ensure_valid_rig_task(
                asset,
                output_dir=rigged_models_output_dir,
                meshy_api_key=key,
                meshy_api_base=meshy_api_base,
                session=rigging_session,
            )
        except MeshyRiggingAPIError as e:
            print(f"Warning: failed to (re-)rig {asset_id}: {e}")
            continue
        if refreshed is not asset:
            asset_sheet[idx] = refreshed
            asset_sheet_changed = True
    
    if asset_sheet_changed:
        input_data["asset_sheet"] = asset_sheet
        try:
            with open(path_to_input_json, "w") as _f:
                json.dump(input_data, _f, indent=2)
            print(f"Persisted refreshed rigging info back to {path_to_input_json}")
        except Exception as _e:
            print(f"Warning: failed to persist refreshed rigging info to {path_to_input_json}: {_e}")
    
    # Build asset_id to rig_task_id mapping from asset_sheet
    rig_task_map = {}
    texture_folder_map = {}
    for asset in asset_sheet:
        asset_id = asset.get("asset_id")
        rig_task_id = asset.get("rig_task_id")
        main_file_path = asset.get("main_file_path")
        
        if asset_id and rig_task_id:
            rig_task_map[asset_id] = rig_task_id
        
        # Build texture folder mapping
        if asset_id and main_file_path:
            model_dir = os.path.dirname(main_file_path)
            model_name = os.path.splitext(os.path.basename(main_file_path))[0]
            texture_folder = os.path.join(model_dir, f"{model_name}_texture")
            texture_folder_map[asset_id] = texture_folder
    
    # Collect all (asset_id, action_id, action_name) combinations from shot_details
    animation_combinations = []
    for shot in shot_details:
        character_actions = shot.get("character_actions", [])
        for action in character_actions:
            asset_id = action.get("asset_id")
            action_id = action.get("action_id")
            action_name = action.get("action_name")
            
            if asset_id and action_id is not None and action_name:
                # Check if this asset has a rig_task_id
                if asset_id in rig_task_map:
                    combination = (asset_id, int(action_id), action_name)
                    animation_combinations.append(combination)
                else:
                    print(f"Warning: No rig_task_id found for asset_id: {asset_id}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_combinations = []
    for combo in animation_combinations:
        key_tuple = (combo[0], combo[1])  # asset_id, action_id
        if key_tuple not in seen:
            seen.add(key_tuple)
            unique_combinations.append(combo)
    
    if not unique_combinations:
        print("No animation combinations found to process.")
        return {
            "successful_animations": [],
            "failed_animations": [],
            "total_processed": 0,
        }
    
    print(f"Found {len(unique_combinations)} unique animation combinations to process:")
    for asset_id, action_id, action_name in unique_combinations:
        rig_task_id = rig_task_map.get(asset_id, "unknown")
        print(f"  - {asset_id} + {action_name} (action_id: {action_id}, rig_task_id: {rig_task_id})")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process animations in parallel
    successful_animations = []
    failed_animations = []
    session = requests.Session()
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_combo = {
            executor.submit(
                apply_single_animation,
                rig_task_map[asset_id],  # rig_task_id
                action_id,
                asset_id,
                action_name,
                output_dir,
                key,
                meshy_api_base,
                texture_folder_map.get(asset_id),  # texture_folder
                session,
            ): (asset_id, action_id, action_name)
            for asset_id, action_id, action_name in unique_combinations
        }
        
        for future in as_completed(future_to_combo):
            asset_id, action_id, action_name = future_to_combo[future]
            try:
                result = future.result()
                if result.get("animation_error"):
                    failed_animations.append(result)
                else:
                    successful_animations.append(result)
                    print(f"✓ Successfully animated: {asset_id} + {action_name}")
            except Exception as e:
                error_result = {
                    "asset_id": asset_id,
                    "action_id": action_id,
                    "action_name": action_name,
                    "animation_error": f"Unexpected error: {str(e)}",
                }
                failed_animations.append(error_result)
                print(f"✗ Unexpected error for {asset_id} + {action_name}: {e}")
    
    # Final report
    total_processed = len(unique_combinations)
    print(f"\n{'='*80}")
    print(f"Animation Results: {len(successful_animations)}/{total_processed} animations completed successfully")
    if failed_animations:
        print(f"Failed animations ({len(failed_animations)}):")
        for failed in failed_animations:
            asset_id = failed.get("asset_id")
            action_name = failed.get("action_name")
            error = failed.get("animation_error", "Unknown error")
            print(f"  - {asset_id} + {action_name}: {error}")
    print(f"{'='*80}\n")
    
    # Create updated JSON with animated paths
    updated_json = None
    if successful_animations:
        print("Creating updated JSON with animated model paths...")
        
        # Create a mapping for easy lookup
        animation_path_map = {}
        for anim in successful_animations:
            asset_id = anim.get("asset_id")
            action_id = anim.get("action_id")
            animated_path = anim.get("animated_path")
            if asset_id is not None and action_id is not None and animated_path:
                key = (asset_id, int(action_id))
                animation_path_map[key] = animated_path
        
        # Update shot_details with animated paths
        updated_shot_details = []
        for shot in shot_details:
            updated_shot = deepcopy(shot)
            character_actions = updated_shot.get("character_actions", [])
            
            updated_character_actions = []
            for action in character_actions:
                updated_action = deepcopy(action)
                asset_id = updated_action.get("asset_id")
                action_id = updated_action.get("action_id")
                
                if asset_id is not None and action_id is not None:
                    key = (asset_id, int(action_id))
                    if key in animation_path_map:
                        updated_action["animated_path"] = animation_path_map[key]
                
                updated_character_actions.append(updated_action)
            
            updated_shot["character_actions"] = updated_character_actions
            updated_shot_details.append(updated_shot)
        
        # Create the updated JSON data
        updated_json = deepcopy(input_data)
        updated_json["shot_details"] = updated_shot_details
        print(f"Created updated JSON with {len(successful_animations)} animated paths")
    
    return {
        "successful_animations": successful_animations,
        "failed_animations": failed_animations,
        "total_processed": total_processed,
        "updated_json": updated_json,
    }


# ---------------------------------------------------------------------------
# GLB duration extraction
# ---------------------------------------------------------------------------

def glb_duration_seconds(glb_path: str) -> Optional[float]:
    """Extract the animation duration in seconds from a GLB file.

    Reads the glTF animation sampler input accessors (timestamps) and returns
    the maximum timestamp across all samplers.  Returns ``None`` on error.
    """
    try:
        import numpy as np
        from pygltflib import GLTF2

        gltf = GLTF2.load(glb_path)
        blob = gltf.binary_blob()
        if blob is None or not gltf.animations:
            return None

        max_t = 0.0
        for anim in gltf.animations:
            for sampler in anim.samplers:
                acc = gltf.accessors[sampler.input]
                bv = gltf.bufferViews[acc.bufferView]
                ts = np.frombuffer(
                    blob[bv.byteOffset : bv.byteOffset + bv.byteLength],
                    dtype=np.float32,
                )
                if len(ts) > 0:
                    max_t = max(max_t, float(ts[-1]))
        return max_t if max_t > 0 else None
    except Exception as e:
        print(f"Warning: could not read duration from {glb_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Movement categories (Walking / Running animations involve root motion)
# ---------------------------------------------------------------------------

_MOVEMENT_CATEGORIES = {"Walking", "Running", "CrouchWalking"}


def _is_movement_animation(action_name: str, csv_path: str = DEFAULT_ANIMATION_CSV) -> bool:
    """Return True if *action_name* belongs to a movement category."""
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Name") == action_name:
                    return row.get("Category", "") in _MOVEMENT_CATEGORIES
    except Exception:
        pass
    # Fallback: check by name substring
    lower = action_name.lower()
    return any(kw in lower for kw in ("walk", "run", "jog", "sprint", "crouch_walk"))


# ---------------------------------------------------------------------------
# LLM-based movement planner for Meshy animations
# ---------------------------------------------------------------------------

class _Coords3(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

class _Rot3(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

class _MeshyMovementCoords(BaseModel):
    """Schema for LLM output: start/end coords + rotation for one character."""
    asset_id: str
    start_coords: Optional[_Coords3] = None
    end_coords: Optional[_Coords3] = None
    start_rotation: Optional[_Rot3] = None


class _ShotMovementPlan(BaseModel):
    shot_id: int
    movements: List[_MeshyMovementCoords]


class _SceneMovementPlan(BaseModel):
    shots: List[_ShotMovementPlan]


_MOVEMENT_SYSTEM_PROMPT = """You are a Movement Planner for a 3D storyboard system.

You are given character layout positions and their animation details (animation name, duration).
For characters with **walking or running animations**, compute where they should end up.

**OUTPUT per moving character per shot:**
- `start_coords`: {x, y, z} or null.
  - For Shot 1: null means use layout position.
  - For Shot 2+: if the character moved (had end_coords) in a previous shot, set start_coords to that previous end position to maintain continuity. Only use null if the character has not moved from its layout position in any prior shot.
- `end_coords`: {x, y, z} — the world position the character reaches after walking/running.
  Compute from the **effective start position** + movement direction based on action_description.
  Use the coordinate system: +X=right, +Y=back, +Z=up.
- `start_rotation`: {x, y, z} degrees — set rotation_z so the character faces the movement direction.
  Formula: rotation_z = atan2(end_x - start_x, -(end_y - start_y)) in degrees.
  Keep x=0, y=0. Set to null only if the current rotation already faces the movement direction (within ±45°).

**CROSS-SHOT CONTINUITY:**
- Shots are sequential. If a character walks to end_coords in Shot N, Shot N+1 must set start_coords to that position.
- Idle characters that didn't move in prior shots: start_coords = null (uses layout).

**RULES:**
- Only output entries for characters that have walking/running/crouching animations.
- Characters with idle or non-locomotion animations should NOT appear in the output.
- Forward vector at rotation_z = R is: (sin(R), -cos(R)).
- rotation_z = 0 → faces -Y (toward camera), 90 → faces +X, 180 → faces +Y, -90 → faces -X.
- Output ONLY the JSON. No explanations.
"""


def _heading_from_to(sx: float, sy: float, ex: float, ey: float) -> float:
    dx = ex - sx
    dy = ey - sy
    return math.degrees(math.atan2(dx, -dy))


def _angle_between(a: float, b: float) -> float:
    d = (a - b) % 360
    if d > 180:
        d -= 360
    return abs(d)


def _verify_meshy_movement(plan: dict, layout_coords: Dict[str, dict]) -> List[Dict[str, Any]]:
    """Geometric check: movement direction must match facing + cross-shot continuity."""
    errors = []
    sorted_shots = sorted(plan.get("shots", []), key=lambda s: s["shot_id"])
    prev_end: Dict[str, dict] = {}  # asset_id → {"x", "y"}

    for sp in sorted_shots:
        sid = sp["shot_id"]
        for mv in sp.get("movements", []):
            aid = mv["asset_id"]
            lc = layout_coords.get(aid, {})
            ec = mv.get("end_coords")
            sc = mv.get("start_coords")

            # Resolve effective start
            if sc:
                sx, sy = sc["x"], sc["y"]
            elif aid in prev_end:
                sx, sy = prev_end[aid]["x"], prev_end[aid]["y"]
            else:
                sx, sy = lc.get("x", 0), lc.get("y", 0)

            # Continuity check
            if aid in prev_end and sc is None:
                pe = prev_end[aid]
                gap = math.hypot(lc.get("x", 0) - pe["x"], lc.get("y", 0) - pe["y"])
                if gap > 0.1:
                    errors.append({
                        "shot_id": sid, "asset_id": aid,
                        "error_type": "continuity_break",
                        "detail": f"Ended at ({pe['x']:.1f},{pe['y']:.1f}) in prev shot but start_coords is null (layout={lc.get('x',0):.1f},{lc.get('y',0):.1f}, gap={gap:.1f})",
                        "fix": f"Set start_coords to {{\"x\":{pe['x']:.2f},\"y\":{pe['y']:.2f},\"z\":0}}",
                    })

            # Direction check
            if ec:
                ex, ey = ec["x"], ec["y"]
                dist = math.hypot(ex - sx, ey - sy)
                if dist > 0.05:
                    required_z = _heading_from_to(sx, sy, ex, ey)
                    sr = mv.get("start_rotation")
                    facing_z = sr["z"] if sr else lc.get("rotation_z", 0)
                    dev = _angle_between(facing_z, required_z)
                    if dev > 45:
                        errors.append({
                            "shot_id": sid, "asset_id": aid,
                            "error_type": "movement_direction_mismatch",
                            "detail": f"Moves ({sx:.1f},{sy:.1f})→({ex:.1f},{ey:.1f}) heading={required_z:.0f}° but faces {facing_z:.0f}° (dev={dev:.0f}°)",
                            "fix": f"Set start_rotation to {{\"x\":0,\"y\":0,\"z\":{required_z:.1f}}}",
                        })

            # Track end for continuity
            if ec:
                prev_end[aid] = {"x": ec["x"], "y": ec["y"]}
            else:
                prev_end[aid] = {"x": sx, "y": sy}

    return errors


def _format_movement_errors(errors):
    if not errors:
        return ""
    lines = ["# Movement Verification Errors", "",
             "Fix ALL errors. Re-output the full corrected JSON.", ""]
    for e in errors:
        lines.append(f"- **Shot {e['shot_id']} / {e['asset_id']}** [{e['error_type']}]: {e['detail']}")
        if e.get("fix"):
            lines.append(f"  **FIX**: {e['fix']}")
    lines.append("\nOutput only the corrected JSON.")
    return "\n".join(lines)


def plan_movement_for_meshy_animations(
    storyboard_script: Dict[str, Any],
    anyllm_api_key: str = None,
    anyllm_api_base: str = None,
    anyllm_provider: str = "gemini",
    vision_model: str = "gemini-3-flash-preview",
    max_retries: int = 3,
    max_improvement_turns: int = 3,
) -> Dict[str, Any]:
    """Plan start_coords / end_coords / start_rotation for Meshy walking/running
    animations and also extract animation duration from downloaded GLBs.

    This function should be called after ``animate_rigged_model`` has populated
    ``animated_path`` on each character_action.  It:

    1. Reads each ``animated_path`` GLB to extract the animation duration.
    2. Identifies walking/running animations via the Meshy CSV category.
    3. For shots containing movement animations, calls the LLM to plan
       ``end_coords`` and ``start_rotation``, then verifies geometrically.
    4. Returns an updated deep copy of *storyboard_script*.
    """
    result = deepcopy(storyboard_script)
    shot_details = result.get("shot_details", [])
    scene_details = result.get("scene_details", [])

    # --- Phase 1: Extract durations from animated GLBs ---
    print("\n=== Meshy movement planner: extracting animation durations ===")
    for shot in shot_details:
        for action in shot.get("character_actions", []):
            apath = action.get("animated_path")
            if apath and os.path.exists(apath):
                dur = glb_duration_seconds(apath)
                if dur is not None:
                    action["duration"] = round(dur, 3)
                    print(f"  {action.get('asset_id')} / {action.get('action_name')}: {dur:.3f}s")

    # --- Build layout_coords per scene ---
    layout_by_scene: Dict[int, Dict[str, dict]] = {}
    for sd in scene_details:
        sid = sd.get("scene_id")
        layout = sd.get("scene_setup", {}).get("layout_description", {})
        coords = {}
        for a in layout.get("assets", []):
            loc = a.get("location", {})
            rot = a.get("rotation", {})
            coords[a["asset_id"]] = {
                "x": loc.get("x", 0), "y": loc.get("y", 0), "z": loc.get("z", 0),
                "rotation_z": rot.get("z", 0),
            }
        layout_by_scene[sid] = coords

    # --- Phase 2: Identify movement actions, group by scene ---
    scenes_needing_plan: Dict[int, List[dict]] = {}  # scene_id → list of shot dicts
    for shot in shot_details:
        sid = shot.get("scene_id")
        has_movement = False
        for action in shot.get("character_actions", []):
            aname = action.get("action_name", "")
            if aname and _is_movement_animation(aname):
                action["_is_movement"] = True
                has_movement = True
        if has_movement:
            scenes_needing_plan.setdefault(sid, []).append(shot)

    if not scenes_needing_plan:
        print("No walking/running animations found — skipping movement planning")
        return result

    # --- Phase 3: LLM call per scene (with verify → correct loop) ---
    print(f"\n=== Planning movement for {sum(len(v) for v in scenes_needing_plan.values())} shots across {len(scenes_needing_plan)} scenes ===")

    for scene_id, shots in scenes_needing_plan.items():
        layout_coords = layout_by_scene.get(scene_id, {})
        storyboard_outline = storyboard_script.get("storyboard_outline", [])
        scene_desc = ""
        for so in storyboard_outline:
            if so.get("scene_id") == scene_id:
                scene_desc = so.get("scene_description", "")
                break

        # Build user prompt
        prompt_parts = [f"## Scene {scene_id}", f"**Description:** {scene_desc}", ""]
        prompt_parts.append("## Character Positions (from layout)")
        for aid, lc in layout_coords.items():
            prompt_parts.append(f"- **{aid}**: x={lc['x']}, y={lc['y']}, z={lc['z']}, rotation_z={lc['rotation_z']}")
        prompt_parts.append("")
        prompt_parts.append("## Shots with Walking/Running Animations")
        for shot in shots:
            shot_id = shot.get("shot_id")
            prompt_parts.append(f"### Shot {shot_id}")
            for action in shot.get("character_actions", []):
                if action.get("_is_movement"):
                    dur = action.get("duration", "?")
                    prompt_parts.append(
                        f"- **{action['asset_id']}**: animation=\"{action.get('action_name')}\", "
                        f"duration={dur}s, action_description=\"{action.get('action_description', '')}\""
                    )
            prompt_parts.append("")
        prompt_parts.append("Plan end_coords and start_rotation for ONLY the walking/running characters listed above.")

        messages = [
            {"role": "system", "content": _MOVEMENT_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": "\n".join(prompt_parts)}]},
        ]

        # LLM call with retry
        def _call(label):
            for attempt in range(max_retries):
                try:
                    resp = completion(
                        api_key=anyllm_api_key, api_base=anyllm_api_base,
                        provider=anyllm_provider, model=vision_model,
                        messages=messages, response_format=_SceneMovementPlan,
                    )
                    gc.collect()
                    return json.loads(resp.choices[0].message.content)
                except Exception as e:
                    print(f"  Movement plan scene {scene_id} ({label}, attempt {attempt+1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 * (2 ** attempt))
            return None

        plan = _call("initial")
        if plan is None:
            print(f"  Failed to plan movement for scene {scene_id}")
            continue
        print(f"  Generated initial movement plan for scene {scene_id}")

        # Verify → correct loop
        messages.append({"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)})
        best, best_err = plan, float("inf")
        for turn in range(1, max_improvement_turns + 1):
            errs = _verify_meshy_movement(plan, layout_coords)
            print(f"  Scene {scene_id} verify turn {turn}: {len(errs)} error(s)")
            if len(errs) < best_err:
                best_err = len(errs)
                best = plan
            if not errs:
                break
            messages.append({"role": "user", "content": [{"type": "text", "text": _format_movement_errors(errs)}]})
            corrected = _call(f"correction {turn}")
            if corrected is None:
                break
            messages.append({"role": "assistant", "content": json.dumps(corrected, ensure_ascii=False)})
            plan = corrected
        else:
            final_errs = _verify_meshy_movement(plan, layout_coords)
            if len(final_errs) < best_err:
                best = plan

        # --- Merge into shot_details ---
        plan_lookup: Dict[int, Dict[str, dict]] = {}
        for sp in best.get("shots", []):
            sid = sp["shot_id"]
            plan_lookup[sid] = {mv["asset_id"]: mv for mv in sp.get("movements", [])}

        for shot in shots:
            shot_id = shot.get("shot_id")
            mvs = plan_lookup.get(shot_id, {})
            for action in shot.get("character_actions", []):
                aid = action.get("asset_id")
                mv = mvs.get(aid)
                if mv:
                    action["start_coords"] = mv.get("start_coords")
                    action["end_coords"] = mv.get("end_coords")
                    action["start_rotation"] = mv.get("start_rotation")
                    ec = mv.get("end_coords")
                    sr = mv.get("start_rotation")
                    print(f"  Shot {shot_id} / {aid}: end_coords={ec}, start_rotation={sr}")

    # --- Cleanup temporary flags ---
    for shot in shot_details:
        for action in shot.get("character_actions", []):
            action.pop("_is_movement", None)

    print("\nMeshy movement planning complete")
    return result
