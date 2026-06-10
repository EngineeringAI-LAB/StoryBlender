"""
metadata structure:
    key(str): the id of the model, each key should be the name of the model ({key}.glb). The corresponding value is the metadata of the model, which is also a dictionary.
    
metadata information:
    prompt(str): the prompt of the model.
    tags(list): the tags of the model.
        "object": the model is an object.
        "character": the model is a character.
        "no_polyhaven": do not use fetch_model_from_polyhaven to download the model.
        "polyhaven": the model is downloaded from polyhaven.
        "no_sketchfab": do not use fetch_model_from_sketchfab to download the model.
        "sketchfab": the model is downloaded from sketchfab.
        "no_genai": do not use text_to_image_to_3d to generate the model with AI
        "meshy": the model is generated with Meshy.
        "hunyuan3d": the model is generated with Hunyuan3D.
    main_file_path(str): the path of the main file of the model.
    thumbnail_url(str): the url of the thumbnail of the model.
    polyhaven_id(str): the id of the model in polyhaven.
    polyhaven_name(str): the name of the model in polyhaven.
    polyhaven_tags(str): the tags of the model in polyhaven.
    sketchfab_uid(str): the uid of the model in sketchfab.
    sketchfab_tags(str): the tags of the model in sketchfab.
    sketchfab_name(str): the name of the model in sketchfab.
    meshy_model_id(str): the id of the model in meshy.
    error(str): the error of the model.
"""

import os
from typing import Optional, Union, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import requests
import re
try:
    from ..llm_completion import completion
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from llm_completion import completion
from google import genai
import asyncio
import time
import base64
from io import BytesIO
from requests.exceptions import RequestException, ProxyError, ConnectionError
import tempfile
import shutil
import zipfile
import mimetypes
from gltflib import GLTF

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
    TencentCloudSDKException,
)
from tencentcloud.ai3d.v20250513 import ai3d_client, models

import nest_asyncio
nest_asyncio.apply()

import gc
import warnings
warnings.filterwarnings("ignore", message="Unclosed client session")
warnings.filterwarnings("ignore", message="Unclosed connector")

try:
    from .retrieve_polyhaven_asset.search_polyhaven_assets import search_polyhaven_assets
except ImportError:
    import importlib
    search_polyhaven_assets = importlib.import_module('retrieve_polyhaven_asset.search_polyhaven_assets').search_polyhaven_assets

def create_assets_base_metadata(director_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build base metadata dict from director result's asset_sheet.

    Args:
        director_result: Dict validated against Storyboard schema with key 'asset_sheet'.

    Returns:
        Dict[str, Dict[str, Any]] mapping asset_id -> {"prompt", "description", "tags"}.
    """
    metadata: Dict[str, Dict[str, Any]] = {}
    asset_sheet = director_result.get("asset_sheet", [])

    if not isinstance(asset_sheet, list):
        return metadata

    for asset in asset_sheet:
        if not isinstance(asset, dict):
            continue
        asset_id = asset.get("asset_id")
        if not asset_id:
            continue
        prompt = asset.get("text_to_image_prompt", "")
        description = asset.get("description", "")
        try:
            tags = asset.get("tags", [])
        except Exception:
            tags = []
        metadata[asset_id] = {
            "prompt": prompt,
            "description": description,
            "tags": tags
        }

    return metadata

def convert_metadata_to_asset_sheet(
    metadata: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Convert metadata dict into an asset sheet list.

    Args:
        metadata: Dict keyed by asset id with fields like prompt/description/tags.

    Returns:
        List of asset dicts, each containing at least: id, text_to_image_prompt, description.
        Optional fields from metadata (tags, main_file_path, thumbnail_url, meshy_model_id,
        sketchfab_uid, sketchfab_name, sketchfab_tags, error) are carried over if present.
    """
    asset_sheet: List[Dict[str, Any]] = []
    for asset_id, info in (metadata or {}).items():
        if not isinstance(info, dict):
            continue
        item: Dict[str, Any] = {
            "asset_id": asset_id,
            "text_to_image_prompt": info.get("prompt", ""),
            "description": info.get("description", ""),
        }
        # TODO fix keys
        for key in [
            "tags",
            "main_file_path",
            "thumbnail_url",
            "thumbnail_web_url",
            "meshy_model_id",
            "sketchfab_uid",
            "sketchfab_name",
            "sketchfab_tags",
            "polyhaven_id",
            "polyhaven_name",
            "polyhaven_tags",
            "hunyuan3d_job_id",
            "error",
            "reflection_log",
        ]:
            if key in info:
                item[key] = info.get(key)
        asset_sheet.append(item)
    return asset_sheet

def replace_asset_sheet_with_new_asset_sheet(
    director_result: Dict[str, Any],
    new_asset_sheet: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Replace the 'asset_sheet' in the director result with the provided new asset sheet list.

    Args:
        director_result: Original storyboard dict from director.
        new_asset_sheet: List of asset dicts to set as the new asset sheet.

    Returns:
        New dict with 'asset_sheet' replaced by the new list.
    """
    result = dict(director_result)  # shallow copy is sufficient
    result["asset_sheet"] = new_asset_sheet
    return result

# Sketchfab
def extract_query(
    model_description: str,
    anyllm_api_key: str,
    anyllm_api_base: str,
    anyllm_provider: str = "openai",
    number_of_tries: int = 10,
    vision_model: str = "gpt-4o-mini",
) -> str:
    """
    Extract 1-2 keywords from a model description using an LLM for Sketchfab search.
    
    Args:
        model_description: Detailed description of the desired 3D model
        anyllm_api_key: API key for the any-llm service
        anyllm_api_base: Base URL for the any-llm API
        anyllm_provider: LLM provider (default: "openai")
        number_of_tries: Total attempts to try the LLM if parsing fails (default 10)
        vision_model: Name of the LLM model to use (default: "gpt-4o-mini")
        
    Returns:
        str: Extracted keywords suitable for Sketchfab search.

    Raises:
        RuntimeError: If all attempts fail to produce parsable keywords from the LLM response.
    """

    last_error: Exception | None = None

    for attempt in range(1, max(1, number_of_tries) + 1):
        try:
            # Create a prompt with Chain of Thought reasoning
            prompt = f"""You are an expert at extracting search keywords for 3D model databases like Sketchfab.

Given a model description, extract 1-2 essential keywords that would be most effective for searching 3D models.

Think step by step:
1. Identify the main object or subject in the description
2. Consider what secondary attribute (style, material, etc.) would be most important for search, for example, cute, realistic, cartoon, etc.
3. Choose keywords that are commonly used in 3D model naming and tagging
4. Avoid overly specific details that might limit search results
5. Prefer simple, clear terms over complex phrases

Please follow this format:
Reasoning: [Your step-by-step analysis]
Keywords: [1-2 keywords separated by space, ONLY provide the keywords, do not explain anything else at here]

Example:
Model description: "A medieval knight in shining armor holding a sword"
Reasoning: The main subject is "knight" which is the primary object. "Medieval" is important for the historical context and style. "Armor" and "sword" are secondary but "knight" already implies these. Choose "medieval knight" as the most effective search terms.
Keywords: medieval knight

Model description: "{model_description}"
"""

            messages = [{"content": prompt, "role": "user"}]
            response = completion(
                model=vision_model,
                provider=anyllm_provider,
                api_key=anyllm_api_key,
                api_base=anyllm_api_base,
                messages=messages
            )
            gc.collect()

            response_text = response.choices[0].message.content

            # Extract keywords from the response using regex
            keywords_match = re.search(r"Keywords:\s*(.+?)(?:\n|$)", response_text, re.IGNORECASE)

            if keywords_match:
                keywords = keywords_match.group(1).strip()
                # Clean up the keywords - remove extra whitespace and limit to reasonable length
                keywords = re.sub(r'\s+', ' ', keywords).strip()
                if keywords and len(keywords) <= 100:  # Reasonable length limit
                    return keywords

            # If parse failed, set error and retry
            last_error = RuntimeError("Failed to parse keywords from LLM response")
        except Exception as e:
            last_error = e
            time.sleep(1)

    # Exhausted all attempts
    raise RuntimeError("extract_query failed after retries") from last_error


def search_sketchfab_models(
    query: str,
    sketchfab_api_key: str,
    categories: Optional[Union[str, List[str]]] = None,
    count: int = 10,
    downloadable: bool = True,
    animated: bool = False,
    sound: bool = False,
    rigged: bool = False,
) -> Union[List[Dict[str, str]], Dict[str, Any]]:
    """
    Search for models on Sketchfab based on query and optional filters.

    Args:
        query: Text to search for.
        sketchfab_api_key: Sketchfab API key (Token-based).
        categories: Optional category or list of categories to filter results.
        count: Number of results to return (default 20).
        downloadable: Whether to restrict to downloadable models (default True).
        animated: Whether to restrict to animated models (default False).
        sound: Whether to restrict to models with sound (default False).
        rigged: Whether to restrict to rigged models (default False).

    Returns:
        list | dict: On success, returns a list of dicts in the format
        [{"uid": uid, "name": name, "tags": tags_csv, "thumbnail_url": thumbnail_url}] extracted
        from the Sketchfab response (first thumbnail URL per result). On failure,
        returns a dict in the form {"error": "..."}.
    """
    if not sketchfab_api_key:
        return {"error": "Sketchfab API key is required"}

    params: Dict[str, Any] = {
        "type": "models",
        "q": query,
        "count": count,
        "downloadable": downloadable,
        "animated": animated,
        "sound": sound,
        "rigged": rigged,
        "archives_flavours": False,  # Mirroring addon.py behavior
    }

    if categories:
        params["categories"] = categories

    headers = {
        "Authorization": f"Token {sketchfab_api_key}",
    }

    try:
        response = requests.get(
            "https://api.sketchfab.com/v3/search",
            headers=headers,
            params=params,
            timeout=30,
        )
        if response.status_code == 401:
            return {"error": "Authentication failed (401). Check your API key."}

        if response.status_code != 200:
            return {"error": f"API request failed with status code {response.status_code}"}

        # Convert response to JSON
        data = response.json()

        if data is None:
            return {"error": "Received empty response from Sketchfab API"}

        # Validate expected structure
        results = data.get("results", [])

        if not isinstance(results, list):
            return {"error": f"Unexpected response format from Sketchfab API: {data}"}

        # Extract uid, name, tags (comma-joined), and first thumbnail url for each result
        uid_name_tags_thumbnail_list: List[Dict[str, str]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            uid = item.get("uid") or ""
            name = item.get("name") or ""
            # Extract tag names and join by comma
            tags_list = item.get("tags") or []
            if isinstance(tags_list, list):
                tag_names = [t.get("name") for t in tags_list if isinstance(t, dict) and t.get("name")]
            else:
                tag_names = []
            tags_csv = ", ".join(tag_names)
            thumbnails = item.get("thumbnails") or {}
            images = thumbnails.get("images") if isinstance(thumbnails, dict) else None
            thumbnail_url = ""
            if isinstance(images, list) and len(images) > 0 and isinstance(images[0], dict):
                thumbnail_url = images[0].get("url") or ""

            uid_name_tags_thumbnail_list.append({
                "uid": uid,
                "name": name,
                "tags": tags_csv,
                "thumbnail_url": thumbnail_url,
            })

        # Return only the extracted list as requested
        return uid_name_tags_thumbnail_list

    except requests.exceptions.Timeout:
        print("Request timed out. Check your internet connection.")
        return {"error": "Request timed out. Check your internet connection."}
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response from Sketchfab API: {str(e)}")
        return {"error": f"Invalid JSON response from Sketchfab API: {str(e)}"}
    except Exception as e:
        print(str(e))
        return {"error": str(e)}


def check_match_model_sketchfab(
    description: str,
    uid_name_thumbnail_list: Union[List[Dict[str, str]], Dict[str, Any]],
    anyllm_api_key: str,
    anyllm_api_base: str,
    anyllm_provider: str = "openai",
    vision_model: str = "gpt-4o-mini",
    threshold_llm: int = 7,
) -> Optional[str]:
    """
    Use an LLM to score and select the best matching model from uid_name_thumbnail_list.

    Args:
        description: Detailed description of the desired 3D model.
        uid_name_thumbnail_list: Output of search_sketchfab_models() or an error dict.
        anyllm_api_key: API key for the any-llm service.
        anyllm_api_base: Base URL for the any-llm API.
        anyllm_provider: LLM provider (default: "openai").
        vision_model: Name of the vision-capable LLM model to use (default: "gpt-4o-mini").
        threshold_llm: Minimum score (1-10) required for a match (default: 7).

    Returns:
        tuple: (best_match_or_none, all_scored_results)
            best_match_or_none: (dict) {"uid": uid, "name": name, "tags": tags_csv, "thumbnail_url": thumbnail_url, "llm_score": score, "llm_explanation": explanation}
                for the highest scoring model if score >= threshold_llm; otherwise None.
            all_scored_results: list of dicts with {"uid", "name", "tags", "thumbnail_url", "llm_score", "llm_explanation"}
                for all evaluated candidates.
    """
    # If upstream returned an error or invalid type, bail out
    if not isinstance(uid_name_thumbnail_list, list):
        return None, []

    if len(uid_name_thumbnail_list) == 0:
        return None, []

    # Prompt to guide the LLM scoring
    instruction_text = (
        "You are evaluating a 3D model asset for use in a scene.\n\n"
        "User's Query:\n"
        "- Desired Description: {description}\n\n"
        "3D Model Asset Information:\n"
        "- Name: {name}\n"
        "- Tags: {tags}\n\n"
        "Please evaluate this 3D model based on how well it matches the user's query. Consider:\n"
        "- Visual appearance and style match, includes color, shape, material, age, condition, etc.\n"
        "- Functional purpose alignment\n"
        "- Scale and proportion appropriateness\n"
        "- Material and texture relevance\n"
        "- Overall suitability for the described use case\n"
        "- The thumbnail must show a complete model of the desired description, otherwise assign a score of 1. For example, if the thumbnail shows only a saddle, and the desciption is white horse wit a saddle, assign score of 1.\n"
        "- Special rule 1: if the thumbnail of the model is plural (have more than one instance of the model), "
        "for example, more than one tree, more than one chair, then the model is not a good match for the query, give it a score of 1.\n"
        "- Special rule 2: if the thumbnail of the model is in exploded view which you can see the parts of the model, "
        "then the model is not a good match for the query, give it a score of 1.\n"
        "- Special rule 3: if there are unnecessary components or the model is in unfitting style, "
        "it should receive a low score.\n"
        "- Your evaluation should consider both the provided image and the information of the asset.\n\n"
        "First, provide a brief explanation (1-2 sentences) of your reasoning.\n"
        "Then, provide a score on a scale of 1-10. A score of 10 means the asset is the perfect match for the query, "
        "7-9 means it is a good match, 6 means it is a mediocre match, below 6 means it is a poor match.\n\n"
        "Format your response as:\n"
        "Explanation: [your reasoning]\n"
        "Score: [number from 1-10]"
    )

    scored_results = []

    # Iterate through candidates, ask the model for a score
    for item in uid_name_thumbnail_list:
        try:
            uid = item.get("uid") if isinstance(item, dict) else None
            name = item.get("name") if isinstance(item, dict) else None
            tags_csv = item.get("tags") if isinstance(item, dict) else ""
            thumbnail_url = item.get("thumbnail_url") if isinstance(item, dict) else None

            if not uid or not name or not thumbnail_url:
                continue

            # Format the prompt
            prompt_text = instruction_text.format(
                description=description,
                name=name,
                tags=tags_csv,
            )

            # Compose the user content for multimodal input
            user_content = [
                {
                    "type": "text",
                    "text": prompt_text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": thumbnail_url,
                        "format": "image/jpeg",
                    },
                },
            ]

            resp = completion(
                model=vision_model,
                provider=anyllm_provider,
                api_key=anyllm_api_key,
                api_base=anyllm_api_base,
                messages=[{"role": "user", "content": user_content}]
            )
            gc.collect()

            answer_text = resp.choices[0].message.content

            if not isinstance(answer_text, str):
                answer_text = str(answer_text)

            # Extract score from response
            score_match = re.search(r"Score:\s*([1-9]|10)\b", answer_text, re.IGNORECASE)
            if not score_match:
                score_match = re.search(r"\b([1-9]|10)\b", answer_text)

            if score_match:
                score = int(score_match.group(1))
            else:
                score = 5  # Default score if extraction fails

            # Extract explanation
            explanation_match = re.search(
                r"Explanation:\s*(.+?)(?=Score:|$)", answer_text, re.IGNORECASE | re.DOTALL
            )
            explanation = explanation_match.group(1).strip() if explanation_match else None

            scored_results.append({
                "uid": uid,
                "name": name,
                "tags": tags_csv,
                "thumbnail_url": thumbnail_url,
                "llm_score": score,
                "llm_explanation": explanation,
            })

            # Early termination if we find a high score match
            if score >= 9:
                break

        except Exception as e:
            # On any exception for this candidate, continue to the next
            continue

    # If no candidates were scored
    if not scored_results:
        return None, []

    # Sort by score descending and get the best match
    scored_results.sort(key=lambda x: x.get("llm_score", 0), reverse=True)
    best_match = scored_results[0]
    # Return the best match only if it meets the threshold
    if best_match.get("llm_score", 0) >= threshold_llm:
        return best_match, scored_results

    return None, scored_results


def check_match_image_prompt(
    image_path: str,
    description: str,
    anyllm_api_key: str,
    anyllm_api_base: str,
    anyllm_provider: str = "gemini",
    vision_model: str = "gemini-3-flash-preview",
    story_summary: str = "",
) -> int:
    """
    Use an LLM to score how well a generated image matches the model description.

    Args:
        image_path: Path to the generated image file.
        description: Description of the desired 3D model.
        anyllm_api_key: API key for the any-llm service.
        anyllm_api_base: Base URL for the any-llm API.
        anyllm_provider: LLM provider (default: "gemini").
        vision_model: Name of the vision-capable LLM model to use (default: "gemini-3-flash-preview").
        story_summary: Summary of the story for additional context (default: "").

    Returns:
        int: Score from 1-10 indicating how well the image matches the description.
    """

    # Read image and convert to base64
    try:
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # Determine mime type
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        image_data_url = f"data:{mime_type};base64,{img_base64}"
    except Exception as e:
        print(f"Failed to read image {image_path}: {e}")
        return 1

    # Build story context if provided
    story_context = ""
    if story_summary:
        story_context = f"\n\nStory Context: {story_summary}\n"

    # Prompt to guide the LLM scoring
    instruction_text = f"""Evaluate this image as a 3D model reference for: {description}{story_context}
CRITICAL RULES (Violations = Score 1):
1. NO CROPPING: The object MUST be fully visible with clear margins on all sides. ANY cropping of the subject = Score 1.
2. SINGLE INSTANCE: There must be EXACTLY ONE instance of the object. Multiple instances or views = Score 1.
3. CORRECT ANATOMY/PROPORTIONS: No distorted body ratios for character models (e.g. large heads/chibi style) unless explicitly requested.
4. MINIMAL EXPRESSION: No facial expressions or body language for character models unless explicitly requested, a character should be in either A pose or T pose.
5. COMPLETE: Full object must be shown (e.g. full body for characters, not just bust).
6. CLEAN BACKGROUND: The background must be white or light gray, no UI elements or other objects.
7. COLORED: The object must be colored, no wireframes or black and white.

Scoring Criteria:
- Score 1: IF ANY CRITICAL RULE IS VIOLATED (Cropped, Multiple, Bad Proportions, Not Colored, Not Clean Background).
- Score 9-10: Perfect, single, full-view object with correct proportions and high fidelity to description.
- Score 2-8: Usable but has minor quality/style issues.

Be strict. If in doubt, Score below 6.

Response Format:
Reasoning: [Concise explanation of why it passed or failed]
Score: [Integer 1-10]"""

    try:
        # Compose the user content for multimodal input
        user_content = [
            {
                "type": "text",
                "text": instruction_text,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_url,
                },
            },
        ]

        resp = completion(
            model=vision_model,
            provider=anyllm_provider,
            api_key=anyllm_api_key,
            api_base=anyllm_api_base,
            messages=[{"role": "user", "content": user_content}]
        )
        gc.collect()

        answer_text = resp.choices[0].message.content

        if not isinstance(answer_text, str):
            answer_text = str(answer_text)

        # Extract score from response
        score_match = re.search(r"Score:\s*([1-9]|10)\b", answer_text, re.IGNORECASE)
        if not score_match:
            score_match = re.search(r"\b([1-9]|10)\b", answer_text)

        if score_match:
            score = int(score_match.group(1))
        else:
            score = 1  # Default score if extraction fails

        return score

    except Exception as e:
        print(f"Error evaluating image {image_path}: {e}")
        return 1


def download_sketchfab_model(sketchfab_api_key, uid, name, output_path="./model.glb"):
    """Download a GLB model from Sketchfab by its UID and save to specified path
    
    Args:
        sketchfab_api_key: Sketchfab API key
        uid: Model UID to download
        name: Model name (used for logging/error messages)
        output_path: Full path including filename and extension (default: './model.glb')
        
    Returns:
        str: Path to the GLB file on success
        dict: Error dictionary on failure
    """
    try:
        if not sketchfab_api_key:
            return {"error": "Sketchfab API key is not configured"}

        # Use proper authorization header for API key auth
        headers = {
            "Authorization": f"Token {sketchfab_api_key}"
        }

        # Request download URL using the exact endpoint from the documentation
        download_endpoint = f"https://api.sketchfab.com/v3/models/{uid}/download"

        response = requests.get(
            download_endpoint,
            headers=headers,
            timeout=30  # Add timeout of 30 seconds
        )

        if response.status_code == 401:
            return {"error": "Authentication failed (401). Check your API key."}

        if response.status_code != 200:
            return {"error": f"Download request failed with status code {response.status_code}"}

        data = response.json()

        # Safety check for None data
        if data is None:
            return {"error": "Received empty response from Sketchfab API for download request"}

        # Extract GLB download URL with safety checks
        glb_data = data.get("glb")
        if not glb_data:
            return {"error": "No GLB download URL available for this model. Response: " + str(data)}

        download_url = glb_data.get("url")
        if not download_url:
            return {"error": "No download URL available for this model. Make sure the model is downloadable and you have access."}

        # Download the GLB model with retry logic for transient SSL/network errors
        import time
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        download_headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        max_retries = 5
        model_response = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                session = requests.Session()
                retry_strategy = Retry(
                    total=2,
                    backoff_factor=1,
                    status_forcelist=[500, 502, 503, 504],
                    allowed_methods=["GET"],
                    raise_on_status=False,
                )
                adapter = HTTPAdapter(max_retries=retry_strategy)
                session.mount("https://", adapter)
                session.mount("http://", adapter)
                
                model_response = session.get(
                    download_url, 
                    timeout=180,
                    headers=download_headers
                )
                
                if model_response.status_code == 200:
                    break
                else:
                    last_error = f"Status code {model_response.status_code}"
                    
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                last_error = str(e)
                print(f"[Sketchfab] Download attempt {attempt + 1}/{max_retries} failed: {last_error}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
                continue
        
        if model_response is None or model_response.status_code != 200:
            return {"error": f"Model download failed after {max_retries} retries. Last error: {last_error}"}

        # Create parent directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create if there's a directory path
            os.makedirs(output_dir, exist_ok=True)
        
        # Save GLB file directly (no unzipping needed)
        with open(output_path, "wb") as f:
            f.write(model_response.content)

        return output_path

    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Check your internet connection and try again with a simpler model."}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON response from Sketchfab API: {str(e)}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to download model: {str(e)}"}


# Polyhaven
REQ_HEADERS = requests.utils.default_headers()
REQ_HEADERS.update({"User-Agent": "blender-mcp"})


def fetch_model_from_polyhaven(
    model_description: str,
    anyllm_api_key: str,
    anyllm_api_base: str,
    anyllm_provider: str = "openai",
    output_path: str = "./model.glb",
    returned_count: int = 5,
    threshold_score: float = 0.6,
    threshold_llm: int = 7,
    resolution: str = "2k",
    vision_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Fetch a model from Polyhaven given a natural language description.

    Pipeline:
    1) Search for models using `search_polyhaven_assets()` with semantic search and LLM reranking.
    2) If results found, download the best match model (gltf format) with all dependencies.
    3) Convert the downloaded gltf file to glb format using gltflib and save to output_path.

    Args:
        model_description: The detailed text description of the desired 3D model.
        anyllm_api_key: API key for the LLM (used in search reranking).
        anyllm_api_base: Base URL for the LLM API.
        anyllm_provider: LLM provider (default: "openai").
        output_path: Full path including filename and extension (default: './model.glb').
        returned_count: Number of results to request from search.
        threshold_score: Minimum combined_score threshold for search results.
        threshold_llm: Minimum LLM score threshold for reranking.
        resolution: Resolution for model download (default: '2k').
        vision_model: Name of the vision-capable LLM model to use for reranking (default: "gpt-4o-mini").

    Returns:
        dict: {
            "main_file_path": str | None,
            "polyhaven_id": str | None,
            "polyhaven_name": str | None,
            "polyhaven_tags": str | None,
            "thumbnail_url": str | None,
            "error": str | None,
        }
    """
    result_payload: Dict[str, Any] = {
        "main_file_path": None,
        "polyhaven_id": None,
        "polyhaven_name": None,
        "polyhaven_tags": None,
        "thumbnail_url": None,
        "error": None,
        "reflection_log": {},
    }

    try:
        # 1) Search for models using semantic search + LLM reranking
        print(f"[Polyhaven DEBUG] Searching for: {model_description[:100]}...")
        search_results = search_polyhaven_assets(
            asset_type="models",
            description=model_description,
            returned_count=returned_count,
            threshold_score=threshold_score,
            rerank_with_llm=True,
            threshold_llm=threshold_llm,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            anyllm_provider=anyllm_provider,
            vision_model=vision_model,
        )
        print(f"[Polyhaven DEBUG] Search returned {len(search_results) if search_results else 0} results")

        # Record polyhaven rerank scores for reflection_log
        if search_results:
            result_payload["reflection_log"]["polyhaven_rerank_scores"] = [
                {"id": r.get("id"), "name": r.get("name"), "score": r.get("llm_score")}
                for r in search_results if r.get("llm_score") is not None
            ]

        if not search_results or len(search_results) == 0:
            result_payload["error"] = "No matching model found from Polyhaven search"
            return result_payload

        # Get the best match (first result after reranking)
        best_match = search_results[0]
        asset_id = best_match.get("id")
        asset_name = best_match.get("name", "")
        asset_tags = best_match.get("tags", [])
        thumbnail_url = best_match.get("thumbnail_url", "")

        if not asset_id:
            result_payload["error"] = "Search result missing asset ID"
            return result_payload

        # 2) Get the files information from Polyhaven API
        files_response = requests.get(
            f"https://api.polyhaven.com/files/{asset_id}",
            headers=REQ_HEADERS,
            timeout=30,
        )
        if files_response.status_code != 200:
            result_payload["error"] = f"Failed to get asset files: {files_response.status_code}"
            return result_payload

        files_data = files_response.json()

        # 3) Download the gltf model
        file_format = "gltf"
        if file_format not in files_data or resolution not in files_data[file_format]:
            result_payload["error"] = f"Requested format ({file_format}) or resolution ({resolution}) not available for this model"
            return result_payload

        file_info = files_data[file_format][resolution][file_format]
        file_url = file_info["url"]

        # Create a temporary directory to store the model and its dependencies
        temp_dir = tempfile.mkdtemp()

        try:
            # Download the main model file
            main_file_name = file_url.split("/")[-1]
            temp_main_file_path = os.path.join(temp_dir, main_file_name)

            response = requests.get(file_url, headers=REQ_HEADERS, timeout=180)
            if response.status_code != 200:
                result_payload["error"] = f"Failed to download model: {response.status_code}"
                return result_payload

            with open(temp_main_file_path, "wb") as f:
                f.write(response.content)

            # Check for included files (textures, etc.) and download them
            if "include" in file_info and file_info["include"]:
                for include_path, include_info in file_info["include"].items():
                    include_url = include_info["url"]
                    include_file_path = os.path.join(temp_dir, include_path)
                    os.makedirs(os.path.dirname(include_file_path), exist_ok=True)

                    include_response = requests.get(include_url, headers=REQ_HEADERS, timeout=60)
                    if include_response.status_code == 200:
                        with open(include_file_path, "wb") as f:
                            f.write(include_response.content)

            # 4) Convert gltf to glb and save to output_path
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Use gltflib to convert gltf to glb (embeds all external resources)
            gltf = GLTF.load(temp_main_file_path)
            gltf.export(output_path)

            result_payload["main_file_path"] = os.path.abspath(output_path)
            result_payload["polyhaven_id"] = asset_id
            result_payload["polyhaven_name"] = asset_name
            result_payload["polyhaven_tags"] = ", ".join(asset_tags) if isinstance(asset_tags, list) else str(asset_tags)
            result_payload["thumbnail_url"] = thumbnail_url
            return result_payload

        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

    except Exception as e:
        result_payload["error"] = f"fetch_model_from_polyhaven failed: {str(e)}"
        print(result_payload)
        return result_payload


def fetch_model_from_sketchfab(
    model_description: str,
    anyllm_api_key: str,
    anyllm_api_base: str,
    anyllm_provider: str = "openai",
    sketchfab_api_key: str = None,
    output_path: str = "./model.glb",
    categories: Optional[Union[str, List[str]]] = None,
    count: int = 10,
    downloadable: bool = True,
    number_of_tries: int = 3,
    vision_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    End-to-end helper to fetch a model from Sketchfab given a natural language description.

    Pipeline:
    1) Use `extract_query()` to get concise keywords from `model_description` via LLM.
    2) Call `search_sketchfab_models()` with the keywords.
    3) Select the best match using `check_match_model_sketchfab()` (LLM + thumbnail).
    4) Download the matched model via `download_sketchfab_model()` and return the GLB file path.

    Args:
        model_description: The detailed text description of the desired 3D model.
        anyllm_api_key: API key for the LLM (used in extract and match steps).
        anyllm_api_base: Base URL for the LLM API.
        anyllm_provider: LLM provider (default: "openai").
        sketchfab_api_key: Sketchfab API key (Token).
        output_path: Full path including filename and extension (default: './model.glb').
        categories: Optional category filter(s) for search.
        count: Number of results to request from search.
        downloadable: Restrict search to downloadable models.
        number_of_tries: Number of attempts for keyword extraction.
        vision_model: Name of the vision-capable LLM model to use (default: "gpt-4o-mini").

    Returns:
        dict: {
            "main_file_path": str | None,
            "keywords": str | None,
            "error": str | None,
        }
    """
    result_payload: Dict[str, Any] = {
        "main_file_path": None,
        "keywords": None,
        "error": None,
        "reflection_log": {},
    }

    try:
        # 1) Extract concise keywords for search
        keywords = extract_query(model_description, anyllm_api_key, anyllm_api_base, anyllm_provider, number_of_tries, vision_model)
        result_payload["keywords"] = keywords
    except Exception as e:
        result_payload["error"] = f"extract_query failed: {e}"
        return result_payload
    
    # 2) Search Sketchfab
    search_res = search_sketchfab_models(
        query=keywords,
        sketchfab_api_key=sketchfab_api_key,
        categories=categories,
        count=count,
        downloadable=downloadable,
    )
    if isinstance(search_res, dict) and search_res.get("error"):
        result_payload["error"] = f"search_sketchfab_models error: {search_res.get('error')}"
        return result_payload
    
    # 3) Choose a matching model using LLM+thumbnail
    match, all_scored = check_match_model_sketchfab(model_description, search_res, anyllm_api_key, anyllm_api_base, anyllm_provider, vision_model)
    # Record all candidate scores for reflection_log
    if all_scored:
        result_payload["reflection_log"]["sketchfab_thumbnail_scores"] = [
            {"uid": s.get("uid"), "name": s.get("name"), "score": s.get("llm_score")}
            for s in all_scored
        ]
    if not match:
        result_payload["error"] = "No matching model found from search results"
        return result_payload

    uid = match.get("uid")
    name = match.get("name")
    tags = match.get("tags")
    thumbnail_url = match.get("thumbnail_url")

    # 4) Download the model
    main_file_path = download_sketchfab_model(sketchfab_api_key=sketchfab_api_key, uid=uid, name=name, output_path=output_path)
    if isinstance(main_file_path, dict) and main_file_path.get("error"):
        result_payload["error"] = f"download_sketchfab_model error: {main_file_path.get('error')}"
        return result_payload

    result_payload["main_file_path"] = main_file_path
    result_payload["sketchfab_uid"] = uid
    result_payload["sketchfab_name"] = name
    result_payload["sketchfab_tags"] = tags
    result_payload["thumbnail_url"] = thumbnail_url
    return result_payload


# Meshy API
class MeshyAPIError(RuntimeError):
    """Raised when the Meshy API returns an error or the task fails."""

def _close_genai_client(client):
    """Close a genai.Client's underlying aiohttp/httpx sessions to suppress
    'Task was destroyed but it is pending' warnings from aiohttp connectors."""
    # 1. Close sync transport
    for sync_close in [
        lambda: client._api_client._httpx_client.close(),
        lambda: client.close(),
    ]:
        try:
            sync_close()
        except Exception:
            pass

    # 2. Collect async close callables from known attribute paths
    _aclose_fns = []
    # Path A: client.aio._api_client._httpx_async_client.aclose
    try:
        _aclose_fns.append(client.aio._api_client._httpx_async_client.aclose)
    except (AttributeError, TypeError):
        pass
    # Path B: client.aio.aclose  (some versions expose this directly)
    try:
        _aclose_fns.append(client.aio.aclose)
    except (AttributeError, TypeError):
        pass
    # Path C: client._api_client.aclose
    try:
        _aclose_fns.append(client._api_client.aclose)
    except (AttributeError, TypeError):
        pass

    if not _aclose_fns:
        return

    # 3. Run async closes in a fresh, short-lived event loop (thread-safe)
    try:
        loop = asyncio.new_event_loop()
        try:
            for fn in _aclose_fns:
                try:
                    loop.run_until_complete(fn())
                except Exception:
                    pass
        finally:
            loop.close()
    except Exception:
        pass


def generate_image_gemini(gemini_api_key, gemini_api_base, model, prompt):
    """
    Generate an image using Google Gemini API.
    
    Args:
        gemini_api_key: Gemini API key for authentication
        gemini_api_base: Base URL for third-party provider (None for official Gemini API)
        model: Model name to use (e.g., "gemini-2.5-flash-image")
        prompt: Text prompt describing the image to generate
    
    Returns:
        PIL.Image: Generated image object, or None if no image was generated
    """
    # Create client based on whether custom base URL is provided
    if gemini_api_base:
        client = genai.Client(
            api_key=gemini_api_key,
            http_options={'base_url': gemini_api_base}
        )
    else:
        client = genai.Client(api_key=gemini_api_key)
    
    try:
        # Generate content
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        
        # Extract and return the image
        for part in response.parts:
            if part.inline_data is not None:
                return part.as_image()
        
        return None
    finally:
        _close_genai_client(client)


def generate_image_openai(openai_api_key, openai_api_base, model, prompt):
    """
    Generate an image using OpenAI API (OpenAI-compatible images/generations endpoint).
    
    Args:
        openai_api_key: OpenAI API key for authentication
        openai_api_base: Base URL for the OpenAI API
        model: Model name to use (e.g., "gpt-image-2")
        prompt: Text prompt describing the image to generate
    
    Returns:
        PIL.Image: Generated image object, or None if no image was generated
    """
    from PIL import Image
    
    url = f"{openai_api_base}/v1/images/generations"
    
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "size": "960x960"
    })
    headers = {
        'Authorization': f'Bearer {openai_api_key}',
        'Content-Type': 'application/json'
    }
    
    response = requests.post(url, headers=headers, data=payload, timeout=120)
    response_data = response.json()
    
    if 'data' in response_data and len(response_data['data']) > 0:
        image_url = response_data['data'][0]['url']
        img_response = requests.get(image_url, timeout=120)
        img = Image.open(BytesIO(img_response.content))
        return img
    
    return None


def generate_image_openai_edit(openai_api_key, openai_api_base, model, prompt, image_path):
    """
    Edit an image using OpenAI API (OpenAI-compatible images/edits endpoint).
    
    Args:
        openai_api_key: OpenAI API key for authentication
        openai_api_base: Base URL for the OpenAI API
        model: Model name to use (e.g., "gpt-image-2")
        prompt: Text prompt describing the edit to apply
        image_path: Path to the reference image file to edit
    
    Returns:
        PIL.Image: Edited image object, or None if editing failed
    """
    from PIL import Image
    
    url = f"{openai_api_base}/v1/images/edits"
    
    headers = {
        'Authorization': f'Bearer {openai_api_key}',
    }
    
    with open(image_path, 'rb') as img_file:
        files = {
            'image': (os.path.basename(image_path), img_file, 'image/png'),
        }
        data = {
            'model': model,
            'prompt': prompt,
            'n': '1',
        }
        response = requests.post(url, headers=headers, files=files, data=data, timeout=120)
    
    response_data = response.json()
    
    if 'data' in response_data and len(response_data['data']) > 0:
        image_url = response_data['data'][0]['url']
        img_response = requests.get(image_url, timeout=120)
        img = Image.open(BytesIO(img_response.content))
        return img
    
    return None


def generate_image(
    prompt,
    image_gen_platform="Gemini",
    gemini_api_key=None,
    gemini_api_base=None,
    gemini_image_model="gemini-3.1-flash-image-preview",
    openai_api_key=None,
    openai_api_base=None,
    openai_image_model="gpt-image-2",
):
    """
    Generate an image using either Gemini or OpenAI, based on image_gen_platform.
    
    Args:
        prompt: Text prompt describing the image to generate
        image_gen_platform: "Gemini" or "OpenAI"
        gemini_api_key: Gemini API key (required if platform is Gemini)
        gemini_api_base: Base URL for Gemini API (optional)
        gemini_image_model: Gemini model name
        openai_api_key: OpenAI API key (required if platform is OpenAI)
        openai_api_base: Base URL for OpenAI API (required if platform is OpenAI)
        openai_image_model: OpenAI model name
    
    Returns:
        PIL.Image: Generated image object, or None if no image was generated
    """
    if image_gen_platform == "OpenAI":
        return generate_image_openai(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            model=openai_image_model,
            prompt=prompt,
        )
    else:
        return generate_image_gemini(
            gemini_api_key=gemini_api_key,
            gemini_api_base=gemini_api_base,
            model=gemini_image_model,
            prompt=prompt,
        )


def text_to_3d_meshy(
    prompt: str,
    *,
    art_style: Optional[str] = None,
    seed: Optional[int] = None,
    meshy_ai_model: Optional[str] = None,
    topology: Optional[str] = None,
    target_polycount: Optional[int] = None,
    should_remesh: Optional[bool] = None,
    symmetry_mode: Optional[str] = None,
    is_a_t_pose: Optional[bool] = None,
    enable_pbr: bool = True,
    texture_prompt: Optional[str] = None,
    texture_image_url: Optional[str] = None,
    moderation: Optional[bool] = None,
    poll_interval: float = 5.0,
    timeout: float = 20 * 60.0,
    meshy_api_key: Optional[str] = None,
    meshy_api_base: Optional[str] = "https://api.meshy.ai/openapi/v2",
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    Create a Text-to-3D task on Meshy (preview + refine with textures) and block until it completes.

    This function creates a preview task first, then automatically creates a refine task to add
    textures to the model. The final result includes both the model and textures.

    Args:
        prompt: Describe what kind of object the 3D model is. Max 600 characters.
        art_style: Desired art style. Options: "realistic", "sculpture". Default: "realistic".
            Note: enable_pbr should be False when using "sculpture" style.
        seed: The seed for the task. Same prompt and seed generates same result in most cases.
        meshy_ai_model: Model ID to use. Options: "meshy-4", "meshy-5", "latest" (Meshy 6 Preview). Default: "latest".
        topology: Topology of generated model. Options: "quad", "triangle". Default: "triangle".
        target_polycount: Target number of polygons (100-300,000 inclusive). Default: 30,000.
        should_remesh: Enable remesh phase. If False, returns unprocessed triangular mesh. Default: True.
        symmetry_mode: Symmetry behavior. Options: "off", "auto", "on". Default: "auto".
        is_a_t_pose: Whether to generate the model in an A/T pose. Default: False.
        enable_pbr: Generate PBR Maps (metallic, roughness, normal) in addition to base color. Default: True.
            Note: Should be False when using "sculpture" art_style.
        texture_prompt: Text prompt to guide texturing. If None, uses the same prompt. Max 600 characters.
            Note: Only one of texture_prompt or texture_image_url should be used.
        texture_image_url: URL or data URI of image to guide texturing. Supports .jpg, .jpeg, .png.
            Note: Only one of texture_prompt or texture_image_url should be used.
        moderation: Screen prompt/image for harmful content. Default: False.
        poll_interval: Seconds to wait between polling attempts.
        timeout: Maximum time to wait in seconds before giving up (applies to each phase).
        meshy_api_key: Meshy API key. If not provided, read from env var MESHY_API_KEY.
        meshy_api_base: Base URL for Meshy API. Default: "https://api.meshy.ai/openapi/v2"
        session: Optional requests.Session to reuse connections.

    Returns:
        The final refine task JSON response when status is SUCCEEDED.

    Raises:
        MeshyAPIError: If API key is missing, task fails/cancels, or timeout occurs.
        requests.HTTPError: For non-2xx HTTP responses.
    """
    key = meshy_api_key or os.getenv("MESHY_API_KEY")
    if not key:
        raise MeshyAPIError(
            "Missing Meshy API key. Set MESHY_API_KEY env var or pass api_key argument."
        )

    sess = session or requests.Session()
    headers = {"Authorization": f"Bearer {key}"}

    # 1) Create preview task
    preview_payload: Dict[str, Any] = {
        "mode": "preview",
        "prompt": prompt,
    }

    # Add optional parameters only if provided
    if art_style is not None:
        preview_payload["art_style"] = art_style
    if seed is not None:
        preview_payload["seed"] = seed
    if meshy_ai_model is not None:
        preview_payload["ai_model"] = meshy_ai_model
    if topology is not None:
        preview_payload["topology"] = topology
    if target_polycount is not None:
        preview_payload["target_polycount"] = target_polycount
    if should_remesh is not None:
        preview_payload["should_remesh"] = should_remesh
    if symmetry_mode is not None:
        preview_payload["symmetry_mode"] = symmetry_mode
    if is_a_t_pose is not None:
        preview_payload["is_a_t_pose"] = is_a_t_pose
    if moderation is not None:
        preview_payload["moderation"] = moderation

    # Create preview task with retry logic
    max_retries = 5
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            create_resp = sess.post(f"{meshy_api_base}/text-to-3d", headers=headers, json=preview_payload, timeout=30)
            create_resp.raise_for_status()
            create_data = create_resp.json()
            break
        except (RequestException, ProxyError, ConnectionError) as e:
            if attempt == max_retries - 1:
                raise MeshyAPIError(f"Failed to create preview task after {max_retries} attempts: {e}")
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay *= 2

    preview_task_id = create_data.get("result")
    if not preview_task_id:
        raise MeshyAPIError("Create preview task response missing 'result'. Response: %r" % (create_data,))

    # 2) Poll for preview task completion
    # Add small initial delay to allow task to be registered in the system
    time.sleep(2.0)
    
    start_time = time.time()
    task_url = f"{meshy_api_base}/text-to-3d/{preview_task_id}"

    terminal_statuses = {"SUCCEEDED", "FAILED", "CANCELED"}

    while True:
        if time.time() - start_time > timeout:
            raise MeshyAPIError(
                f"Timed out waiting for Meshy preview task {preview_task_id} after {timeout} seconds"
            )

        # Retry logic for polling requests
        poll_success = False
        for poll_attempt in range(3):
            try:
                resp = sess.get(task_url, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                poll_success = True
                break
            except requests.HTTPError as e:
                # If we get 404, the task might not be ready yet - wait and retry
                if resp.status_code == 404:
                    time.sleep(poll_interval)
                    break
                if poll_attempt == 2:
                    raise
                print(f"Preview poll attempt {poll_attempt + 1}/3 failed: {e}. Retrying in 2s...")
                time.sleep(2)
            except (RequestException, ProxyError, ConnectionError) as e:
                if poll_attempt == 2:
                    raise MeshyAPIError(f"Failed to poll preview task status after 3 attempts: {e}")
                print(f"Preview poll attempt {poll_attempt + 1}/3 failed: {e}. Retrying in 2s...")
                time.sleep(2)
        
        if not poll_success:
            time.sleep(poll_interval)
            continue

        status = data.get("status")
        if status in terminal_statuses:
            if status == "SUCCEEDED":
                break  # Continue to refine task
            # Attach detailed error if present
            task_error = (data.get("task_error") or {}).get("message")
            raise MeshyAPIError(
                f"Meshy preview task {preview_task_id} ended with status {status}. Error: {task_error}"
            )

        time.sleep(poll_interval)

    # 3) Create refine task for texturing
    refine_payload: Dict[str, Any] = {
        "mode": "refine",
        "preview_task_id": preview_task_id,
        "enable_pbr": enable_pbr,
    }

    # Add texture guidance - use texture_prompt if provided, otherwise use original prompt
    if texture_image_url is not None:
        refine_payload["texture_image_url"] = texture_image_url
    else:
        # Use texture_prompt if provided, otherwise fall back to original prompt
        refine_payload["texture_prompt"] = texture_prompt if texture_prompt is not None else prompt

    # Add optional refine parameters
    if meshy_ai_model is not None:
        refine_payload["ai_model"] = meshy_ai_model
    if moderation is not None:
        refine_payload["moderation"] = moderation

    # Create refine task with retry logic
    retry_delay = 2.0
    for attempt in range(max_retries):
        try:
            refine_resp = sess.post(f"{meshy_api_base}/text-to-3d", headers=headers, json=refine_payload, timeout=30)
            refine_resp.raise_for_status()
            refine_data = refine_resp.json()
            break
        except (RequestException, ProxyError, ConnectionError) as e:
            if attempt == max_retries - 1:
                raise MeshyAPIError(f"Failed to create refine task after {max_retries} attempts: {e}")
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay *= 2

    refine_task_id = refine_data.get("result")
    if not refine_task_id:
        raise MeshyAPIError("Create refine task response missing 'result'. Response: %r" % (refine_data,))

    # 4) Poll for refine task completion
    # Add small initial delay to allow task to be registered in the system
    time.sleep(2.0)
    
    start_time = time.time()
    refine_task_url = f"{meshy_api_base}/text-to-3d/{refine_task_id}"

    while True:
        if time.time() - start_time > timeout:
            raise MeshyAPIError(
                f"Timed out waiting for Meshy refine task {refine_task_id} after {timeout} seconds"
            )

        # Retry logic for polling refine requests
        poll_success = False
        for poll_attempt in range(3):
            try:
                resp = sess.get(refine_task_url, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                poll_success = True
                break
            except requests.HTTPError as e:
                # If we get 404, the task might not be ready yet - wait and retry
                if resp.status_code == 404:
                    time.sleep(poll_interval)
                    break
                if poll_attempt == 2:
                    raise
                print(f"Refine poll attempt {poll_attempt + 1}/3 failed: {e}. Retrying in 2s...")
                time.sleep(2)
            except (RequestException, ProxyError, ConnectionError) as e:
                if poll_attempt == 2:
                    raise MeshyAPIError(f"Failed to poll refine task status after 3 attempts: {e}")
                print(f"Refine poll attempt {poll_attempt + 1}/3 failed: {e}. Retrying in 2s...")
                time.sleep(2)
        
        if not poll_success:
            time.sleep(poll_interval)
            continue

        status = data.get("status")
        if status in terminal_statuses:
            if status == "SUCCEEDED":
                return data
            # Attach detailed error if present
            task_error = (data.get("task_error") or {}).get("message")
            raise MeshyAPIError(
                f"Meshy refine task {refine_task_id} ended with status {status}. Error: {task_error}"
            )

        time.sleep(poll_interval)


def image_to_3d_meshy(
    image_url: str,
    *,
    meshy_ai_model: Optional[str] = None,
    topology: Optional[str] = None,
    target_polycount: int = 100000,
    should_remesh: Optional[bool] = None,
    symmetry_mode: Optional[str] = None,
    is_a_t_pose: Optional[bool] = None,
    should_texture: bool = True,
    enable_pbr: bool = True,
    texture_prompt: Optional[str] = None,
    texture_image_url: Optional[str] = None,
    moderation: Optional[bool] = None,
    poll_interval: float = 5.0,
    timeout: float = 20 * 60.0,
    meshy_api_key: Optional[str] = None,
    meshy_api_base: Optional[str] = "https://api.meshy.ai/openapi/v1",
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    Create an Image-to-3D task on Meshy and block until it completes.

    This function creates a single task that generates a 3D model from an input image.
    By default, it sets should_texture=True and texture_image_url to the same as the input image.

    Args:
        image_url: Publicly accessible URL or base64-encoded data URI of the image.
            Supports .jpg, .jpeg, and .png formats.
            Example data URI: "data:image/jpeg;base64,<your base64-encoded image data>"
        meshy_ai_model: Model ID to use. Options: "meshy-4", "meshy-5", "latest" (Meshy 6 Preview). Default: "latest".
        topology: Topology of generated model. Options: "quad", "triangle". Default: "triangle".
        target_polycount: Target number of polygons (100-300,000 inclusive). Default: 100000.
        should_remesh: Enable remesh phase. If False, returns unprocessed triangular mesh. Default: True.
        symmetry_mode: Symmetry behavior. Options: "off", "auto", "on". Default: "auto".
        is_a_t_pose: Whether to generate the model in an A/T pose. Default: False.
        should_texture: Whether to generate textures. Default: True.
        enable_pbr: Generate PBR Maps (metallic, roughness, normal) in addition to base color. Default: True.
        texture_prompt: Text prompt to guide texturing. Max 600 characters.
            Note: Only one of texture_prompt or texture_image_url should be used.
        texture_image_url: URL or data URI of image to guide texturing. Supports .jpg, .jpeg, .png.
            If not provided and should_texture=True, defaults to the input image_url.
            Note: Only one of texture_prompt or texture_image_url should be used.
        moderation: Screen prompt/image for harmful content. Default: False.
        poll_interval: Seconds to wait between polling attempts.
        timeout: Maximum time to wait in seconds before giving up.
        meshy_api_key: Meshy API key. If not provided, read from env var MESHY_API_KEY.
        meshy_api_base: Base URL for Meshy API. Default: "https://api.meshy.ai/openapi/v1"
        session: Optional requests.Session to reuse connections.

    Returns:
        The task JSON response when status is SUCCEEDED.

    Raises:
        MeshyAPIError: If API key is missing, task fails/cancels, or timeout occurs.
        requests.HTTPError: For non-2xx HTTP responses.
    """
    key = meshy_api_key or os.getenv("MESHY_API_KEY")
    if not key:
        raise MeshyAPIError(
            "Missing Meshy API key. Set MESHY_API_KEY env var or pass api_key argument."
        )

    sess = session or requests.Session()
    headers = {"Authorization": f"Bearer {key}"}

    # Create task payload
    task_payload: Dict[str, Any] = {
        "image_url": image_url,
    }

    # Add optional parameters only if provided
    if meshy_ai_model is not None:
        task_payload["ai_model"] = meshy_ai_model
    if topology is not None:
        task_payload["topology"] = topology
    if target_polycount is not None:
        task_payload["target_polycount"] = target_polycount
    if should_remesh is not None:
        task_payload["should_remesh"] = should_remesh
    if symmetry_mode is not None:
        task_payload["symmetry_mode"] = symmetry_mode
    if is_a_t_pose is not None:
        task_payload["is_a_t_pose"] = is_a_t_pose
    if should_texture is not None:
        task_payload["should_texture"] = should_texture
    if enable_pbr is not None:
        task_payload["enable_pbr"] = enable_pbr
    if moderation is not None:
        task_payload["moderation"] = moderation

    # Handle texture guidance
    # If should_texture is True and no texture guidance is provided, use the input image
    if should_texture:
        if texture_prompt is not None:
            task_payload["texture_prompt"] = texture_prompt
        elif texture_image_url is not None:
            task_payload["texture_image_url"] = texture_image_url
        else:
            # Default: use the input image for texturing
            task_payload["texture_image_url"] = image_url

    # Create task with retry logic
    max_retries = 3
    retry_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            create_resp = sess.post(f"{meshy_api_base}/image-to-3d", headers=headers, json=task_payload, timeout=30)
            create_resp.raise_for_status()
            create_data = create_resp.json()
            break
        except (RequestException, ProxyError, ConnectionError) as e:
            if attempt == max_retries - 1:
                raise MeshyAPIError(f"Failed to create image-to-3d task after {max_retries} attempts: {e}")
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

    task_id = create_data.get("result")
    if not task_id:
        raise MeshyAPIError("Create task response missing 'result'. Response: %r" % (create_data,))

    # Poll for task completion
    # Add small initial delay to allow task to be registered in the system
    time.sleep(2.0)
    
    start_time = time.time()
    task_url = f"{meshy_api_base}/image-to-3d/{task_id}"

    terminal_statuses = {"SUCCEEDED", "FAILED", "CANCELED"}

    while True:
        if time.time() - start_time > timeout:
            raise MeshyAPIError(
                f"Timed out waiting for Meshy image-to-3d task {task_id} after {timeout} seconds"
            )

        # Retry logic for polling requests
        poll_success = False
        for poll_attempt in range(3):
            try:
                resp = sess.get(task_url, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                poll_success = True
                break
            except requests.HTTPError as e:
                # If we get 404, the task might not be ready yet - wait and retry
                if resp.status_code == 404:
                    time.sleep(poll_interval)
                    break
                if poll_attempt == 2:
                    raise
                print(f"Image-to-3D poll attempt {poll_attempt + 1}/3 failed: {e}. Retrying in 2s...")
                time.sleep(2)
            except (RequestException, ProxyError, ConnectionError) as e:
                if poll_attempt == 2:
                    raise MeshyAPIError(f"Failed to poll image-to-3d task status after 3 attempts: {e}")
                print(f"Image-to-3D poll attempt {poll_attempt + 1}/3 failed: {e}. Retrying in 2s...")
                time.sleep(2)
        
        if not poll_success:
            time.sleep(poll_interval)
            continue

        status = data.get("status")
        if status in terminal_statuses:
            if status == "SUCCEEDED":
                return data
            # Attach detailed error if present
            task_error = (data.get("task_error") or {}).get("message")
            raise MeshyAPIError(
                f"Meshy image-to-3d task {task_id} ended with status {status}. Error: {task_error}"
            )

        time.sleep(poll_interval)


class Hunyuan3DAPIError(RuntimeError):
    """Raised when the Hunyuan3D API returns an error or the task fails."""


def image_to_3d_hunyuan3d(
    tencent_secret_id: str,
    tencent_secret_key: str,
    image_path: str,
    enable_pbr: bool = True,
    *,
    face_count: int = 300000,
    poll_interval: float = 5.0,
    timeout: float = 20 * 60.0,
) -> Dict[str, Any]:
    """
    Create an Image-to-3D task on Tencent Hunyuan3D and block until it completes.

    This function creates a task that generates a 3D model from an input image using
    Tencent's Hunyuan3D Pro API.

    Args:
        tencent_secret_id: Tencent Cloud Secret ID for authentication.
        tencent_secret_key: Tencent Cloud Secret Key for authentication.
        image_path: Local path to the image file. Supports .jpg, .jpeg, and .png formats.
        enable_pbr: Generate PBR Maps (metallic, roughness, normal) in addition to base color. Default: True.
        face_count: Target number of faces for the generated model (up to 300,000). Default: 300000.
        poll_interval: Seconds to wait between polling attempts.
        timeout: Maximum time to wait in seconds before giving up.

    Returns:
        Dict containing:
            - job_id: The Hunyuan3D job ID
            - glb_url: URL to download the GLB model
            - preview_image_url: URL of the preview image
            - obj_url: URL to download the OBJ model (optional)

    Raises:
        Hunyuan3DAPIError: If API credentials are missing, task fails, or timeout occurs.
        TencentCloudSDKException: For Tencent Cloud API errors.
    """
    if not tencent_secret_id or not tencent_secret_key:
        raise Hunyuan3DAPIError(
            "Missing Tencent Cloud credentials. Provide tencent_secret_id and tencent_secret_key."
        )

    if not os.path.exists(image_path):
        raise Hunyuan3DAPIError(f"Image file not found: {image_path}")

    # Read image file and convert to data URI
    with open(image_path, "rb") as file:
        file_data = file.read()
    
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    
    encoded_data = base64.b64encode(file_data).decode("utf-8")
    image_data_uri = f"data:{mime_type};base64,{encoded_data}"

    try:
        # Initialize Tencent Cloud client
        cred = credential.Credential(tencent_secret_id, tencent_secret_key)
        
        httpProfile = HttpProfile()
        httpProfile.endpoint = "ai3d.tencentcloudapi.com"
        
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        
        client = ai3d_client.Ai3dClient(cred, "ap-guangzhou", clientProfile)

        # Submit job request
        req = models.SubmitHunyuanTo3DProJobRequest()
        params = {
            "EnablePBR": enable_pbr,
            "FaceCount": face_count,
            "ImageBase64": image_data_uri,
        }
        req.from_json_string(json.dumps(params))

        # Retry indefinitely on concurrent-job limit errors (not a real failure,
        # just means too many users are hitting the API). Do not count these as
        # retries in any outer retry loop.
        while True:
            try:
                resp = client.SubmitHunyuanTo3DProJob(req)
                break
            except TencentCloudSDKException as err:
                err_code = getattr(err, "code", "") or ""
                if "RequestLimitExceeded" in err_code or "JobNumExceed" in err_code:
                    print(
                        f"[Hunyuan3D] Concurrent job limit reached ({err_code}); "
                        f"waiting 30s and retrying submit..."
                    )
                    time.sleep(30)
                    continue
                raise
        resp_data = json.loads(resp.to_json_string())
        
        job_id = resp_data.get("JobId")
        if not job_id:
            raise Hunyuan3DAPIError(f"Submit job response missing 'JobId'. Response: {resp_data}")

        # Poll for job completion
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                raise Hunyuan3DAPIError(
                    f"Timed out waiting for Hunyuan3D job {job_id} after {timeout} seconds"
                )

            # Query job status
            query_req = models.QueryHunyuanTo3DProJobRequest()
            query_params = {"JobId": job_id}
            query_req.from_json_string(json.dumps(query_params))
            
            query_resp = client.QueryHunyuanTo3DProJob(query_req)
            query_data = json.loads(query_resp.to_json_string())
            
            status = query_data.get("Status", "")
            
            if status == "DONE":
                # Job completed successfully
                result_files = query_data.get("ResultFile3Ds", [])
                
                result = {
                    "job_id": job_id,
                    "glb_url": None,
                    "obj_url": None,
                    "preview_image_url": None,
                }
                
                for file_info in result_files:
                    file_type = file_info.get("Type", "")
                    if file_type == "GLB":
                        result["glb_url"] = file_info.get("Url")
                        result["preview_image_url"] = file_info.get("PreviewImageUrl")
                    elif file_type == "OBJ":
                        result["obj_url"] = file_info.get("Url")
                
                if not result["glb_url"]:
                    raise Hunyuan3DAPIError(f"No GLB file found in result. Response: {query_data}")
                
                return result
            
            elif status == "FAIL" or query_data.get("ErrorCode") or query_data.get("ErrorMessage"):
                error_code = query_data.get("ErrorCode", "")
                error_message = query_data.get("ErrorMessage", "Unknown error")
                raise Hunyuan3DAPIError(
                    f"Hunyuan3D job {job_id} failed. ErrorCode: {error_code}, ErrorMessage: {error_message}"
                )
            
            # Status is still "RUN" or other, keep polling
            time.sleep(poll_interval)

    except TencentCloudSDKException as err:
        raise Hunyuan3DAPIError(f"Tencent Cloud SDK error: {err}")


def _download_chunk(
    url: str,
    start: int,
    end: int,
    chunk_index: int,
    max_retries: int = 3,
) -> tuple:
    """
    Download a single chunk of a file using HTTP Range requests.
    
    Returns:
        Tuple of (chunk_index, data) for ordering.
    """
    headers = {"Range": f"bytes={start}-{end}"}
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            return (chunk_index, response.content)
        except (RequestException, ProxyError, ConnectionError) as e:
            if attempt == max_retries - 1:
                raise Hunyuan3DAPIError(f"Failed to download chunk {chunk_index} after {max_retries} attempts: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2


def download_model_from_hunyuan3d(
    task_result: Dict[str, Any],
    output_path: str = "./model.glb",
    session: Optional[requests.Session] = None,
    num_threads: int = 8,
    chunk_size: int = 2 * 1024 * 1024,
) -> Dict[str, Any]:
    """
    Download a GLB model file from a Hunyuan3D task result to a local file.
    Uses parallel chunk downloading for faster speeds.

    Args:
        task_result: The task result dictionary returned by image_to_3d_hunyuan3d.
        output_path: Path where the file should be saved. Default: "./model.glb"
        session: Optional requests.Session to reuse connections.
        num_threads: Number of parallel download threads. Default: 8
        chunk_size: Size of each chunk in bytes. Default: 2MB

    Returns:
        Dict with model_path, model_id, and thumbnail_url.

    Raises:
        Hunyuan3DAPIError: If the task result doesn't contain a GLB URL.
        requests.HTTPError: For non-2xx HTTP responses during download.
        IOError: If there are issues writing the file.
    """
    glb_url = task_result.get("glb_url")
    if not glb_url:
        raise Hunyuan3DAPIError("Task result does not contain a GLB URL.")
    
    model_id = task_result.get("job_id")
    thumbnail_url = task_result.get("preview_image_url")
    
    # Create parent directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get file size with HEAD request
    max_retries = 3
    retry_delay = 2.0
    file_size = None
    supports_range = False
    
    for attempt in range(max_retries):
        try:
            head_response = requests.head(glb_url, timeout=30, allow_redirects=True)
            head_response.raise_for_status()
            file_size = int(head_response.headers.get("Content-Length", 0))
            accept_ranges = head_response.headers.get("Accept-Ranges", "")
            supports_range = accept_ranges.lower() == "bytes" and file_size > 0
            break
        except (RequestException, ProxyError, ConnectionError) as e:
            if attempt == max_retries - 1:
                file_size = 0
                supports_range = False
            else:
                time.sleep(retry_delay)
                retry_delay *= 2
    
    # Use parallel download if server supports Range requests and file is large enough
    if supports_range and file_size > chunk_size:
        # Calculate chunks
        chunks = []
        for i, start in enumerate(range(0, file_size, chunk_size)):
            end = min(start + chunk_size - 1, file_size - 1)
            chunks.append((i, start, end))
        
        # Download chunks in parallel
        downloaded_chunks = {}
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(_download_chunk, glb_url, start, end, idx): idx
                for idx, start, end in chunks
            }
            for future in as_completed(futures):
                chunk_idx, data = future.result()
                downloaded_chunks[chunk_idx] = data
        
        # Write chunks in order
        with open(output_path, "wb") as f:
            for i in range(len(chunks)):
                f.write(downloaded_chunks[i])
    else:
        # Fallback to sequential download with larger chunk size
        sess = session or requests.Session()
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                response = sess.get(glb_url, stream=True, timeout=180)
                response.raise_for_status()
                break
            except (RequestException, ProxyError, ConnectionError) as e:
                if attempt == max_retries - 1:
                    raise Hunyuan3DAPIError(f"Failed to download model after {max_retries} attempts: {e}")
                print(f"Download attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    
    return {
        "model_path": output_path,
        "model_id": model_id,
        "thumbnail_url": thumbnail_url,
    }


def text_to_image_to_3d(
    prompt: str,
    *,
    model_id: str,
    description: str = "",
    output_dir: str = "./models",
    ai_platform: str = "Hunyuan3D",
    image_gen_platform: str = "Gemini",
    gemini_api_key: Optional[str] = None,
    gemini_api_base: Optional[str] = None,
    gemini_image_model: str = "gemini-3-pro-image-preview",
    openai_api_key: Optional[str] = None,
    openai_api_base: Optional[str] = None,
    openai_image_model: str = "gpt-image-2",
    meshy_ai_model: Optional[str] = None,
    topology: Optional[str] = None,
    target_polycount: Optional[int] = None,
    should_remesh: Optional[bool] = None,
    symmetry_mode: Optional[str] = None,
    is_a_t_pose: Optional[bool] = None,
    should_texture: bool = True,
    enable_pbr: Optional[bool] = None,
    moderation: Optional[bool] = None,
    poll_interval: float = 5.0,
    timeout: float = 20 * 60.0,
    meshy_api_key: Optional[str] = None,
    meshy_api_base: Optional[str] = "https://api.meshy.ai/openapi/v1",
    tencent_secret_id: Optional[str] = None,
    tencent_secret_key: Optional[str] = None,
    session: Optional[requests.Session] = None,
    anyllm_api_key: Optional[str] = None,
    anyllm_api_base: Optional[str] = None,
    anyllm_provider: str = "gemini",
    vision_model: str = "gemini-2.5-flash-preview",
    max_retries: int = 5,
    threshold_llm: int = 8,
    story_summary: str = "",
) -> Dict[str, Any]:
    """
    Generate a 3D model from a text prompt by first generating an image, then converting to 3D.
    
    This function combines text-to-image (via Gemini) and image-to-3D (via Meshy or Hunyuan3D) into a single workflow.
    It mocks the interface of text_to_3d_meshy but uses a two-step process internally.
    
    Args:
        prompt: Text prompt describing the desired 3D model.
        model_id: Identifier for the model (used for naming the saved image file).
        description: Description of the 3D model for quality checking.
        output_dir: Directory to save the generated image (default: "./models").
        ai_platform: AI platform for 3D generation. Options: "Hunyuan3D", "Meshy". Default: "Hunyuan3D".
        gemini_api_key: Gemini API key for image generation. If not provided, read from env var GEMINI_API_KEY.
        gemini_api_base: Base URL for Gemini API (None for official API).
        gemini_image_model: Gemini model to use for image generation (default: "gemini-3-pro-image-preview").
        meshy_ai_model: Meshy model ID. Options: "meshy-4", "meshy-5", "latest". Default: "latest".
        topology: Topology of generated model. Options: "quad", "triangle". Default: "triangle".
        target_polycount: Target number of polygons (100-300,000 inclusive). Default: 30,000.
        should_remesh: Enable remesh phase. If False, returns unprocessed triangular mesh. Default: True.
        symmetry_mode: Symmetry behavior. Options: "off", "auto", "on". Default: "auto".
        is_a_t_pose: Whether to generate the model in an A/T pose. Default: False.
        should_texture: Whether to generate textures. Default: True.
        enable_pbr: Generate PBR Maps (metallic, roughness, normal). Default: False.
        moderation: Screen content for harmful material. Default: False.
        poll_interval: Seconds to wait between polling attempts.
        timeout: Maximum time to wait in seconds before giving up.
        meshy_api_key: Meshy API key. If not provided, read from env var MESHY_API_KEY.
        meshy_api_base: Base URL for Meshy API. Default: "https://api.meshy.ai/openapi/v1"
        tencent_secret_id: Tencent Cloud Secret ID for Hunyuan3D.
        tencent_secret_key: Tencent Cloud Secret Key for Hunyuan3D.
        session: Optional requests.Session to reuse connections.
        anyllm_api_key: API key for any-llm service (for quality checking).
        anyllm_api_base: Base URL for any-llm API (for quality checking), can be None.
        anyllm_provider: LLM provider (default: "gemini").
        vision_model: Vision model for quality checking (default: "gemini-2.5-flash-preview").
        max_retries: Maximum number of image generation attempts (default: 5).
        threshold_llm: Minimum acceptable score (default: 8).
        story_summary: Summary of the story for additional context (default: "").
    
    Returns:
        The task JSON response when status is SUCCEEDED.
    
    Raises:
        MeshyAPIError/Hunyuan3DAPIError: If API key is missing, task fails/cancels, or timeout occurs.
        requests.HTTPError: For non-2xx HTTP responses.
        RuntimeError: If image generation fails.
        ValueError: If ai_platform is not one of "Hunyuan3D" or "Meshy".
    """
    # Validate ai_platform
    if ai_platform not in ("Hunyuan3D", "Meshy"):
        raise ValueError(f"ai_platform must be one of 'Hunyuan3D' or 'Meshy', got '{ai_platform}'")
    
    # Validate image generation credentials
    if image_gen_platform == "OpenAI":
        if not openai_api_key:
            raise RuntimeError("Missing OpenAI API key. Pass openai_api_key argument.")
        if not openai_api_base:
            raise RuntimeError("Missing OpenAI API base. Pass openai_api_base argument.")
        gemini_key = gemini_api_key  # may be None, not needed for OpenAI
    else:
        gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise RuntimeError(
                "Missing Gemini API key. Set GEMINI_API_KEY env var or pass gemini_api_key argument."
            )
    
    os.makedirs(output_dir, exist_ok=True)
    final_image_path = os.path.join(output_dir, f"{model_id}.png")
    
    # Use description for quality checking, fall back to prompt if description is empty
    check_description = description if description else prompt
    
    # Check if quality checking is enabled (anyllm credentials provided)
    enable_quality_check = anyllm_api_key
    
    best_score = 0
    best_image = None
    last_error = None
    image_prompt_scores = []
    
    # Step 1: Generate image from prompt with quality checking and retry logic
    for attempt in range(max_retries):
        try:
            generated_image = generate_image(
                prompt=prompt,
                image_gen_platform=image_gen_platform,
                gemini_api_key=gemini_key,
                gemini_api_base=gemini_api_base,
                gemini_image_model=gemini_image_model,
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
                openai_image_model=openai_image_model,
            )
            
            if generated_image is None:
                last_error = f"Failed to generate image for {model_id} (attempt {attempt + 1})"
                print(f"  {model_id}: Attempt {attempt + 1}/{max_retries}: Generation failed")
                continue
            
            # Save temporary image for quality check
            temp_image_path = os.path.join(output_dir, f"{model_id}_temp_{attempt}.png")
            generated_image.save(temp_image_path)
            
            if enable_quality_check:
                # Check image quality
                score = check_match_image_prompt(
                    image_path=temp_image_path,
                    description=check_description,
                    anyllm_api_key=anyllm_api_key,
                    anyllm_api_base=anyllm_api_base,
                    vision_model=vision_model,
                    story_summary=story_summary,
                )
                
                print(f"  {model_id}: Attempt {attempt + 1}/{max_retries}: Score = {score}")
                image_prompt_scores.append(score)
                
                # Keep track of the best image
                if score > best_score:
                    # Remove previous best temp image if exists
                    if best_image and os.path.exists(best_image) and best_image != temp_image_path:
                        try:
                            os.remove(best_image)
                        except Exception:
                            pass
                    best_score = score
                    best_image = temp_image_path
                else:
                    # Remove this temp image as it's not better
                    try:
                        os.remove(temp_image_path)
                    except Exception:
                        pass
                
                # Early termination if we get a perfect or very good score
                if score >= 9:
                    break
            else:
                # No quality check, just use the first successful generation
                best_image = temp_image_path
                best_score = 0  # Unknown score
                break
                
        except Exception as e:
            last_error = f"Error generating image for {model_id} (attempt {attempt + 1}): {str(e)}"
            print(f"  {model_id}: Attempt {attempt + 1}/{max_retries}: Error - {str(e)}")
            continue
    
    # Step 2: Finalize the best image
    if best_image and os.path.exists(best_image):
        # Rename best image to final path
        if best_image != final_image_path:
            shutil.move(best_image, final_image_path)
        image_path = final_image_path
    else:
        raise RuntimeError(last_error or f"Failed to generate image from prompt after {max_retries} attempts: {prompt}")
    
    # Clean up any remaining temp files
    for attempt in range(max_retries):
        temp_path = os.path.join(output_dir, f"{model_id}_temp_{attempt}.png")
        if os.path.exists(temp_path) and temp_path != final_image_path:
            try:
                os.remove(temp_path)
            except Exception:
                pass
    
    # Step 3: Convert image to 3D using the selected platform
    if ai_platform == "Hunyuan3D":
        # Use Hunyuan3D
        result = image_to_3d_hunyuan3d(
            tencent_secret_id=tencent_secret_id,
            tencent_secret_key=tencent_secret_key,
            image_path=image_path,
            enable_pbr=enable_pbr if enable_pbr is not None else True,
            poll_interval=poll_interval,
            timeout=timeout,
        )
    else:
        # Use Meshy - convert image to base64 data URI
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
        img_str = base64.b64encode(img_bytes).decode()
        image_data_uri = f"data:image/png;base64,{img_str}"
        
        result = image_to_3d_meshy(
            image_url=image_data_uri,
            meshy_ai_model=meshy_ai_model,
            topology=topology,
            target_polycount=target_polycount,
            should_remesh=should_remesh,
            symmetry_mode=symmetry_mode,
            is_a_t_pose=is_a_t_pose,
            should_texture=should_texture,
            enable_pbr=enable_pbr,
            moderation=moderation,
            poll_interval=poll_interval,
            timeout=timeout,
            meshy_api_key=meshy_api_key,
            meshy_api_base=meshy_api_base,
            session=session,
        )
    
    # Attach image_prompt_scores to result for reflection_log propagation
    if image_prompt_scores:
        result["_image_prompt_scores"] = image_prompt_scores
    
    return result


def download_model_from_meshy(
    task_result: Dict[str, Any],
    output_path: str = "./model.glb",
    format: str = "glb",
    session: Optional[requests.Session] = None,
) -> str:
    """
    Download a model file from a Meshy task result to a local file.

    Args:
        task_result: The task result dictionary returned by text_to_3d_meshy or image_to_3d_meshy.
        output_path: Path where the file should be saved. Default: "./model.glb"
        format: Model format to download. Options: "glb", "fbx", "obj", "mtl", "usdz". Default: "glb"
        session: Optional requests.Session to reuse connections.

    Returns:
        The output path where the model was saved.

    Raises:
        MeshyAPIError: If the task result doesn't contain the requested format URL.
        requests.HTTPError: For non-2xx HTTP responses during download.
        IOError: If there are issues writing the file.
    """
    # Extract model URL for the requested format
    model_urls = task_result.get("model_urls", {})
    model_url = model_urls.get(format)

    model_id = task_result.get("asset_id")
    thumbnail_url = task_result.get("thumbnail_url")
    
    if not model_url:
        raise MeshyAPIError(
            f"Task result does not contain a '{format}' model URL. Available formats: {list(model_urls.keys())}"
        )
    
    # Download the model with retry logic
    sess = session or requests.Session()
    max_retries = 3
    retry_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            response = sess.get(model_url, stream=True, timeout=180)
            response.raise_for_status()
            break
        except (RequestException, ProxyError, ConnectionError) as e:
            if attempt == max_retries - 1:
                raise MeshyAPIError(f"Failed to download model after {max_retries} attempts: {e}")
            print(f"Download attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay *= 2
    
    # Write to file
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    model_info = {
        "model_path": output_path,
        "model_id": model_id,
        "thumbnail_url": thumbnail_url,
    }
    
    return model_info


# Image prompt generation functions
def _generate_single_image_prompt(
    model_id: str,
    prompt: str,
    description: str,
    gemini_api_key: str,
    gemini_api_base: Optional[str],
    gemini_image_model: str,
    save_path: str,
    anyllm_api_key: Optional[str] = None,
    anyllm_api_base: Optional[str] = None,
    vision_model: str = "gemini/gemini-2.5-flash-preview",
    max_retries: int = 5,
    threshold_llm: int = 9,
    story_summary: str = "",
    image_gen_platform: str = "Gemini",
    openai_api_key: Optional[str] = None,
    openai_api_base: Optional[str] = None,
    openai_image_model: str = "gpt-image-2",
) -> Dict[str, Any]:
    """
    Generate a single image prompt for a model with quality checking and retry logic.
    
    Args:
        model_id: The identifier for the model
        prompt: Text prompt for image generation
        description: Description of the 3D model for quality checking
        gemini_api_key: Gemini API key
        gemini_api_base: Gemini API base URL (optional)
        gemini_image_model: Gemini model name for image generation
        save_path: Directory to save the generated image
        anyllm_api_key: API key for any-llm service (for quality checking)
        anyllm_api_base: Base URL for any-llm API (for quality checking), can be None, will use built-in default value
        vision_model: Vision model for quality checking (default: "gemini/gemini-2.5-flash-preview")
        max_retries: Maximum number of generation attempts (default: 5)
        threshold_llm: Minimum acceptable score (default: 9)
        story_summary: Summary of the story for additional context (default: "")
        
    Returns:
        Dict with model_id, image_prompt_path, image_score, and error (if any)
    """
    result = {
        "model_id": model_id,
        "image_prompt_path": None,
        "image_score": None,
        "image_prompt_scores": [],
        "error": None,
    }
    
    os.makedirs(save_path, exist_ok=True)
    final_image_path = os.path.join(save_path, f"{model_id}_image_prompt.png")
    
    best_score = 0
    best_image = None
    last_error = None
    image_prompt_scores = []
    
    # Use description for quality checking, fall back to prompt if description is empty
    check_description = description if description else prompt
    
    # Check if quality checking is enabled (anyllm credentials provided)
    enable_quality_check = anyllm_api_key
    
    for attempt in range(max_retries):
        try:
            # Generate image
            generated_image = generate_image(
                prompt=prompt,
                image_gen_platform=image_gen_platform,
                gemini_api_key=gemini_api_key,
                gemini_api_base=gemini_api_base,
                gemini_image_model=gemini_image_model,
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base,
                openai_image_model=openai_image_model,
            )
            
            if generated_image is None:
                last_error = f"Failed to generate image for {model_id} (attempt {attempt + 1})"
                print(f"  Attempt {attempt + 1}/{max_retries}: Generation failed")
                continue
            
            # Save temporary image for quality check
            temp_image_path = os.path.join(save_path, f"{model_id}_image_prompt_temp_{attempt}.png")
            generated_image.save(temp_image_path)
            
            if enable_quality_check:
                # Check image quality
                score = check_match_image_prompt(
                    image_path=temp_image_path,
                    description=check_description,
                    anyllm_api_key=anyllm_api_key,
                    anyllm_api_base=anyllm_api_base,
                    vision_model=vision_model,
                    story_summary=story_summary,
                )
                
                print(f"  {model_id}: Attempt {attempt + 1}/{max_retries}: Score = {score}")
                image_prompt_scores.append(score)
                
                # Keep track of the best image
                if score > best_score:
                    # Remove previous best temp image if exists
                    if best_image and os.path.exists(best_image) and best_image != temp_image_path:
                        try:
                            os.remove(best_image)
                        except Exception:
                            pass
                    best_score = score
                    best_image = temp_image_path
                else:
                    # Remove this temp image as it's not better
                    try:
                        os.remove(temp_image_path)
                    except Exception:
                        pass
                
                # Early termination if we get a perfect or very good score
                if score >= 9:
                    break
            else:
                # No quality check, just use the first successful generation
                best_image = temp_image_path
                best_score = 0  # Unknown score
                break
                
        except Exception as e:
            last_error = f"Error generating image for {model_id} (attempt {attempt + 1}): {str(e)}"
            print(f"  {model_id}: Attempt {attempt + 1}/{max_retries}: Error - {str(e)}")
            continue
    
    # Finalize result
    result["image_prompt_scores"] = image_prompt_scores
    if best_image and os.path.exists(best_image):
        # Rename best image to final path
        try:
            if best_image != final_image_path:
                shutil.move(best_image, final_image_path)
            result["image_prompt_path"] = os.path.abspath(final_image_path)
            result["image_score"] = best_score if enable_quality_check else None
        except Exception as e:
            result["error"] = f"Error saving final image: {str(e)}"
    else:
        result["error"] = last_error or f"Failed to generate image for {model_id} after {max_retries} attempts"
    
    # Clean up any remaining temp files
    for attempt in range(max_retries):
        temp_path = os.path.join(save_path, f"{model_id}_image_prompt_temp_{attempt}.png")
        if os.path.exists(temp_path) and temp_path != final_image_path:
            try:
                os.remove(temp_path)
            except Exception:
                pass
    
    return result


def _generate_single_image_prompt_by_editing_image(
    model_id: str,
    prompt: str,
    description: str,
    reference_image_path: str,
    gemini_api_key: str,
    gemini_api_base: Optional[str],
    gemini_image_model: str,
    save_path: str,
    anyllm_api_key: Optional[str] = None,
    anyllm_api_base: Optional[str] = None,
    vision_model: str = "gemini/gemini-2.5-flash-preview",
    max_retries: int = 5,
    threshold_llm: int = 8,
    story_summary: str = "",
    image_gen_platform: str = "Gemini",
    openai_api_key: Optional[str] = None,
    openai_api_base: Optional[str] = None,
    openai_image_model: str = "gpt-image-2",
) -> Dict[str, Any]:
    """
    Generate an image prompt for a character variant by editing a reference character image.
    
    This function uses image editing to create a new character look while preserving
    facial features and height from the reference character. It changes clothing and
    accessories according to the new description.
    
    Args:
        model_id: The identifier for the model
        prompt: Text prompt for image generation (describes the new look)
        description: Description of the character variant for quality checking
        reference_image_path: Path to the reference character's image prompt
        gemini_api_key: Gemini API key
        gemini_api_base: Gemini API base URL (optional)
        gemini_image_model: Gemini model name for image generation
        save_path: Directory to save the generated image
        anyllm_api_key: API key for any-llm service (for quality checking)
        anyllm_api_base: Base URL for any-llm API (for quality checking)
        vision_model: Vision model for quality checking (default: "gemini/gemini-2.5-flash-preview")
        max_retries: Maximum number of generation attempts (default: 5)
        threshold_llm: Minimum acceptable score (default: 8)
        story_summary: Summary of the story for additional context (default: "")
        
    Returns:
        Dict with model_id, image_prompt_path, image_score, and error (if any)
    """
    from PIL import Image
    
    result = {
        "model_id": model_id,
        "image_prompt_path": None,
        "image_score": None,
        "image_prompt_scores": [],
        "error": None,
    }
    
    os.makedirs(save_path, exist_ok=True)
    final_image_path = os.path.join(save_path, f"{model_id}_image_prompt.png")
    
    best_score = 0
    best_image = None
    last_error = None
    image_prompt_scores = []
    
    # Use description for quality checking, fall back to prompt if description is empty
    check_description = description if description else prompt
    
    # Check if quality checking is enabled (anyllm credentials provided)
    enable_quality_check = anyllm_api_key
    
    # Load the reference image (validate it exists)
    try:
        reference_image = Image.open(reference_image_path)
    except Exception as e:
        result["error"] = f"Failed to load reference image: {str(e)}"
        return result
    
    # Create the editing prompt that preserves facial features and height
    editing_prompt = f"""Edit this image of a 3D character to create a new look. 
Keep the facial features and height from this reference character exactly the same.
Unless the character has change in age, for example, from a boy to a teenager, in this case you can change the face and body.
Change the character's clothing and accessories according to this description:

{prompt}

Important: 
- Maintain the same facial structure, body proportions (unless age change is explicitly mentioned), and overall pose (T-Pose or A-Pose)
- Modify makeup or cleanliness of the face or body as needed, but the facial structure and body shape should remain the same.
- Keep the same art style and rendering quality
- Keep the white background
- Show full body, front view"""
    
    # Setup platform-specific client for Gemini
    client = None
    if image_gen_platform != "OpenAI":
        if gemini_api_base:
            client = genai.Client(
                api_key=gemini_api_key,
                http_options={'base_url': gemini_api_base}
            )
        else:
            client = genai.Client(api_key=gemini_api_key)
    
    try:
        for attempt in range(max_retries):
            try:
                if image_gen_platform == "OpenAI":
                    # Generate image using OpenAI image editing API
                    generated_image = generate_image_openai_edit(
                        openai_api_key=openai_api_key,
                        openai_api_base=openai_api_base,
                        model=openai_image_model,
                        prompt=editing_prompt,
                        image_path=reference_image_path,
                    )
                else:
                    # Generate image using Gemini image editing
                    response = client.models.generate_content(
                        model=gemini_image_model,
                        contents=[editing_prompt, reference_image],
                    )
                    
                    # Extract the generated image
                    generated_image = None
                    for part in response.parts:
                        if part.inline_data is not None:
                            generated_image = part.as_image()
                            break
                
                if generated_image is None:
                    last_error = f"Failed to generate image for {model_id} (attempt {attempt + 1})"
                    print(f"  Attempt {attempt + 1}/{max_retries}: Generation failed")
                    continue
                
                # Save temporary image for quality check
                temp_image_path = os.path.join(save_path, f"{model_id}_image_prompt_temp_{attempt}.png")
                generated_image.save(temp_image_path)
                
                if enable_quality_check:
                    # Check image quality
                    score = check_match_image_prompt(
                        image_path=temp_image_path,
                        description=check_description,
                        anyllm_api_key=anyllm_api_key,
                        anyllm_api_base=anyllm_api_base,
                        vision_model=vision_model,
                        story_summary=story_summary,
                    )
                    
                    print(f"  {model_id}: Attempt {attempt + 1}/{max_retries}: Score = {score}")
                    image_prompt_scores.append(score)
                    
                    # Keep track of the best image
                    if score > best_score:
                        # Remove previous best temp image if exists
                        if best_image and os.path.exists(best_image) and best_image != temp_image_path:
                            try:
                                os.remove(best_image)
                            except Exception:
                                pass
                        best_score = score
                        best_image = temp_image_path
                    else:
                        # Remove this temp image as it's not better
                        try:
                            os.remove(temp_image_path)
                        except Exception:
                            pass
                    
                    # Early termination if we get a perfect or very good score
                    if score >= 9:
                        break
                else:
                    # No quality check, just use the first successful generation
                    best_image = temp_image_path
                    best_score = 0  # Unknown score
                    break
                    
            except Exception as e:
                last_error = f"Error generating image for {model_id} (attempt {attempt + 1}): {str(e)}"
                print(f"  {model_id}: Attempt {attempt + 1}/{max_retries}: Error - {str(e)}")
                continue
    finally:
        if client is not None:
            _close_genai_client(client)
    
    # Finalize result
    result["image_prompt_scores"] = image_prompt_scores
    if best_image and os.path.exists(best_image):
        # Rename best image to final path
        try:
            if best_image != final_image_path:
                shutil.move(best_image, final_image_path)
            result["image_prompt_path"] = os.path.abspath(final_image_path)
            result["image_score"] = best_score if enable_quality_check else None
        except Exception as e:
            result["error"] = f"Error saving final image: {str(e)}"
    else:
        result["error"] = last_error or f"Failed to generate image for {model_id} after {max_retries} attempts"
    
    # Clean up any remaining temp files
    for attempt in range(max_retries):
        temp_path = os.path.join(save_path, f"{model_id}_image_prompt_temp_{attempt}.png")
        if os.path.exists(temp_path) and temp_path != final_image_path:
            try:
                os.remove(temp_path)
            except Exception:
                pass
    
    return result


def fetch_image_prompt(
    director_result: Dict[str, Any],
    gemini_api_key: str,
    gemini_image_model: str = "gemini-2.0-flash-preview-image-generation",
    gemini_api_base: Optional[str] = None,
    save_path: str = "./models",
    model_id_list: Optional[List[str]] = None,
    max_concurrent: int = 10,
    anyllm_api_key: Optional[str] = None,
    anyllm_api_base: Optional[str] = None,
    vision_model: str = "gemini/gemini-2.5-flash-preview",
    max_retries: int = 5,
    threshold_llm: int = 8,
    image_gen_platform: str = "Gemini",
    openai_api_key: Optional[str] = None,
    openai_api_base: Optional[str] = None,
    openai_image_model: str = "gpt-image-2",
) -> Dict[str, Any]:
    """
    Generate image prompts for 3D assets in parallel.
    
    This function generates preview images for all assets (or specific ones) in the asset_sheet
    using Gemini image generation. The images can be reviewed before proceeding to 3D generation.
    
    For character assets with a reference_character field, the function uses image editing to
    maintain facial features and height consistency while changing clothing and accessories.
    These assets are processed after all base character assets (with null reference_character)
    have been generated.
    
    Args:
        director_result: Dictionary containing asset_sheet with assets to process
        gemini_api_key: Gemini API key for image generation
        gemini_image_model: Gemini model to use for image generation
        gemini_api_base: Base URL for Gemini API (optional)
        save_path: Directory to save generated images and JSON
        model_id_list: Optional list of model IDs to generate. If None, generates all.
        max_concurrent: Maximum number of concurrent image generations
        anyllm_api_key: API key for any-llm service (for quality checking)
        anyllm_api_base: Base URL for any-llm API (for quality checking)
        vision_model: Vision model for quality checking (default: "gemini/gemini-2.5-flash-preview")
        max_retries: Maximum number of generation attempts per asset (default: 5)
        threshold_llm: Minimum acceptable score for quality check (default: 8)
        
    Returns:
        Updated director_result with image_prompt_path added to each processed asset
    """
    # Make a deep copy to avoid modifying the original
    result = json.loads(json.dumps(director_result))
    asset_sheet = result.get("asset_sheet", [])
    story_summary = result.get("story_summary", "")
    
    if not isinstance(asset_sheet, list):
        print(f"Invalid asset_sheet: {asset_sheet}")
        return result
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Build mapping of model_id to asset, prompt and description
    # Separate assets into two groups: base assets (no reference) and variant assets (with reference)
    base_assets_to_process = []  # Assets with reference_character = null
    variant_assets_to_process = []  # Assets with reference_character != null
    
    for asset in asset_sheet:
        if not isinstance(asset, dict):
            continue
        asset_id = asset.get("asset_id")
        if not asset_id:
            continue
        # Filter by model_id_list if provided
        if model_id_list is not None and asset_id not in model_id_list:
            continue
        prompt = asset.get("text_to_image_prompt", "")
        description = asset.get("description", "")
        reference_character = asset.get("reference_character")
        
        if prompt or description:
            if reference_character:
                # This is a character variant that needs image editing
                variant_assets_to_process.append((asset_id, prompt, description, reference_character, asset))
            else:
                # This is a base asset that can be generated directly
                base_assets_to_process.append((asset_id, prompt, description, asset))
    
    if not base_assets_to_process and not variant_assets_to_process:
        print("No assets to process")
        return result
    
    image_results = {}
    
    # Phase 1: Generate base assets (those without reference_character) in parallel
    if base_assets_to_process:
        print(f"\n{'='*80}")
        print(f"Phase 1: Generating {len(base_assets_to_process)} base asset image prompts...")
        if image_gen_platform == "Gemini":
            print(f"Using Gemini image model: {gemini_image_model}")
        elif image_gen_platform == "OpenAI":
            print(f"Using OpenAI image model: {openai_image_model}")
        print(f"{'='*80}")
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_model = {
                executor.submit(
                    _generate_single_image_prompt,
                    asset_id,
                    prompt,
                    description,
                    gemini_api_key,
                    gemini_api_base,
                    gemini_image_model,
                    save_path,
                    anyllm_api_key,
                    anyllm_api_base,
                    vision_model,
                    max_retries,
                    threshold_llm,
                    story_summary,
                    image_gen_platform,
                    openai_api_key,
                    openai_api_base,
                    openai_image_model,
                ): asset_id
                for asset_id, prompt, description, _ in base_assets_to_process
            }
            
            for future in as_completed(future_to_model):
                model_id = future_to_model[future]
                try:
                    gen_result = future.result()
                    image_results[model_id] = gen_result
                    if gen_result.get("error"):
                        print(f"✗ {model_id}: {gen_result['error']}")
                    else:
                        score_info = f" (score: {gen_result.get('image_score')})" if gen_result.get('image_score') else ""
                        print(f"✓ {model_id}: Image generated successfully{score_info}")
                except Exception as e:
                    image_results[model_id] = {
                        "model_id": model_id,
                        "image_prompt_path": None,
                        "error": f"Unexpected error: {str(e)}",
                    }
                    print(f"✗ {model_id}: Unexpected error: {str(e)}")
    
    # Update asset_sheet with base asset image paths (needed for variant assets)
    for asset in asset_sheet:
        asset_id = asset.get("asset_id")
        if asset_id in image_results:
            gen_result = image_results[asset_id]
            if gen_result.get("image_prompt_path"):
                asset["image_prompt_path"] = gen_result["image_prompt_path"]
            if gen_result.get("image_score") is not None:
                asset["image_prompt_score"] = gen_result["image_score"]
            if gen_result.get("error"):
                asset["image_prompt_error"] = gen_result["error"]
            # Propagate image_prompt_scores into reflection_log
            if gen_result.get("image_prompt_scores"):
                reflection_log = asset.get("reflection_log", {})
                reflection_log["image_prompt_scores"] = gen_result["image_prompt_scores"]
                asset["reflection_log"] = reflection_log
    
    # Phase 2: Generate variant assets (those with reference_character) using image editing
    if variant_assets_to_process:
        print(f"\n{'='*80}")
        print(f"Phase 2: Generating {len(variant_assets_to_process)} character variant image prompts using image editing...")
        print(f"{'='*80}")
        
        # Build a mapping of asset_id to image_prompt_path for quick lookup
        asset_id_to_image_path = {}
        for asset in asset_sheet:
            aid = asset.get("asset_id")
            img_path = asset.get("image_prompt_path")
            if aid and img_path:
                asset_id_to_image_path[aid] = img_path
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_model = {}
            
            for asset_id, prompt, description, reference_character, _ in variant_assets_to_process:
                # Get the reference character's image path
                reference_image_path = asset_id_to_image_path.get(reference_character)
                
                if not reference_image_path:
                    # Reference character image not found, skip or report error
                    image_results[asset_id] = {
                        "model_id": asset_id,
                        "image_prompt_path": None,
                        "error": f"Reference character '{reference_character}' image not found",
                    }
                    print(f"✗ {asset_id}: Reference character '{reference_character}' image not found")
                    continue
                
                future = executor.submit(
                    _generate_single_image_prompt_by_editing_image,
                    asset_id,
                    prompt,
                    description,
                    reference_image_path,
                    gemini_api_key,
                    gemini_api_base,
                    gemini_image_model,
                    save_path,
                    anyllm_api_key,
                    anyllm_api_base,
                    vision_model,
                    max_retries,
                    threshold_llm,
                    story_summary,
                    image_gen_platform,
                    openai_api_key,
                    openai_api_base,
                    openai_image_model,
                )
                future_to_model[future] = asset_id
            
            for future in as_completed(future_to_model):
                model_id = future_to_model[future]
                try:
                    gen_result = future.result()
                    image_results[model_id] = gen_result
                    if gen_result.get("error"):
                        print(f"✗ {model_id}: {gen_result['error']}")
                    else:
                        score_info = f" (score: {gen_result.get('image_score')})" if gen_result.get('image_score') else ""
                        print(f"✓ {model_id}: Image generated via editing successfully{score_info}")
                except Exception as e:
                    image_results[model_id] = {
                        "model_id": model_id,
                        "image_prompt_path": None,
                        "error": f"Unexpected error: {str(e)}",
                    }
                    print(f"✗ {model_id}: Unexpected error: {str(e)}")
    
    # Update asset_sheet with image paths
    for asset in asset_sheet:
        asset_id = asset.get("asset_id")
        if asset_id in image_results:
            gen_result = image_results[asset_id]
            if gen_result.get("image_prompt_path"):
                asset["image_prompt_path"] = gen_result["image_prompt_path"]
            if gen_result.get("image_score") is not None:
                asset["image_prompt_score"] = gen_result["image_score"]
            if gen_result.get("error"):
                asset["image_prompt_error"] = gen_result["error"]
            # Propagate image_prompt_scores into reflection_log
            if gen_result.get("image_prompt_scores"):
                reflection_log = asset.get("reflection_log", {})
                reflection_log["image_prompt_scores"] = gen_result["image_prompt_scores"]
                asset["reflection_log"] = reflection_log
    
    # Save the result as image_prompt.json
    output_json_path = os.path.join(save_path, "image_prompt.json")
    with open(output_json_path, "w") as f:
        json.dump(result, f, indent=4)
    
    # Report results
    successful = sum(1 for r in image_results.values() if r.get("image_prompt_path"))
    failed = sum(1 for r in image_results.values() if r.get("error"))
    print(f"\n{'='*80}")
    print(f"Image Prompt Generation: {successful}/{len(image_results)} images generated successfully")
    if failed:
        print(f"Failed: {failed}")
    print(f"Saved to: {output_json_path}")
    print(f"{'='*80}\n")
    
    return result


def _generate_single_3d_from_image(
    model_id: str,
    image_path: str,
    output_dir: str,
    ai_platform: str = "Hunyuan3D",
    meshy_api_key: Optional[str] = None,
    meshy_ai_model: str = "latest",
    tencent_secret_id: Optional[str] = None,
    tencent_secret_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a single 3D model from an image prompt.
    
    Args:
        model_id: The identifier for the model
        image_path: Path to the image file
        output_dir: Directory to save the 3D model
        ai_platform: AI platform for 3D generation. Options: "Hunyuan3D", "Meshy". Default: "Hunyuan3D".
        meshy_api_key: Meshy API key (required if ai_platform is "Meshy")
        meshy_ai_model: Meshy AI model version
        tencent_secret_id: Tencent Cloud Secret ID (required if ai_platform is "Hunyuan3D")
        tencent_secret_key: Tencent Cloud Secret Key (required if ai_platform is "Hunyuan3D")
        
    Returns:
        Dict with model_id, main_file_path, model_id (from API), thumbnail_url, and error
    """
    result = {
        "model_id": model_id,
        "main_file_path": None,
        "meshy_model_id": None,
        "hunyuan3d_job_id": None,
        "thumbnail_url": None,
        "error": None,
    }
    
    try:
        output_path = os.path.join(output_dir, f"{model_id}.glb")
        
        if ai_platform == "Hunyuan3D":
            # Use Hunyuan3D
            task_result = image_to_3d_hunyuan3d(
                tencent_secret_id=tencent_secret_id,
                tencent_secret_key=tencent_secret_key,
                image_path=image_path,
                enable_pbr=True,
                poll_interval=5,
                timeout=15 * 60,
            )
            
            # Download the model
            model_info = download_model_from_hunyuan3d(
                task_result=task_result,
                output_path=output_path,
            )
            
            result["main_file_path"] = os.path.abspath(model_info["model_path"])
            result["hunyuan3d_job_id"] = model_info["model_id"]
            result["thumbnail_url"] = model_info["thumbnail_url"]
        else:
            # Use Meshy - read image and convert to base64 data URI
            with open(image_path, "rb") as img_file:
                img_bytes = img_file.read()
            img_str = base64.b64encode(img_bytes).decode()
            image_data_uri = f"data:image/png;base64,{img_str}"
            
            # Convert image to 3D
            task_result = image_to_3d_meshy(
                image_url=image_data_uri,
                meshy_ai_model=meshy_ai_model,
                should_remesh=True,
                poll_interval=5,
                timeout=15 * 60,
                meshy_api_key=meshy_api_key,
            )
            
            # Download the model
            model_info = download_model_from_meshy(
                task_result=task_result,
                output_path=output_path,
            )
            
            result["main_file_path"] = os.path.abspath(model_info["model_path"])
            result["meshy_model_id"] = model_info["model_id"]
            result["thumbnail_url"] = model_info["thumbnail_url"]
        
        return result
        
    except Exception as e:
        result["error"] = f"Error generating 3D for {model_id}: {str(e)}"
        return result


def fetch_3d_from_image_prompt(
    image_prompt_result: Dict[str, Any],
    output_dir: str = "./models",
    ai_platform: str = "Hunyuan3D",
    meshy_api_key: Optional[str] = None,
    meshy_ai_model: str = "latest",
    tencent_secret_id: Optional[str] = None,
    tencent_secret_key: Optional[str] = None,
    max_concurrent: int = 10,
    model_id_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate 3D models from image prompts in parallel.
    
    This function takes the result of fetch_image_prompt (which contains image_prompt_path for each asset)
    and generates 3D models using Meshy's or Hunyuan3D's image-to-3D API.
    
    Args:
        image_prompt_result: Dictionary containing asset_sheet with image_prompt_path for each asset
        output_dir: Directory to save 3D models
        ai_platform: AI platform for 3D generation. Options: "Hunyuan3D", "Meshy". Default: "Hunyuan3D".
        meshy_api_key: Meshy API key for 3D generation (required if ai_platform is "Meshy")
        meshy_ai_model: Meshy AI model version (default: "latest")
        tencent_secret_id: Tencent Cloud Secret ID (required if ai_platform is "Hunyuan3D")
        tencent_secret_key: Tencent Cloud Secret Key (required if ai_platform is "Hunyuan3D")
        max_concurrent: Maximum number of concurrent 3D generations
        model_id_list: Optional list of model IDs to generate. If None, generates all.
        
    Returns:
        Updated result with main_file_path, model IDs, thumbnail_url added to each asset
    """
    # Make a deep copy to avoid modifying the original
    result = json.loads(json.dumps(image_prompt_result))
    asset_sheet = result.get("asset_sheet", [])
    
    if not isinstance(asset_sheet, list):
        return result
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build list of assets to process (those with valid image_prompt_path)
    assets_to_process = []
    for asset in asset_sheet:
        if not isinstance(asset, dict):
            continue
        asset_id = asset.get("asset_id")
        # Filter by model_id_list if provided
        if model_id_list is not None and asset_id not in model_id_list:
            continue
        image_path = asset.get("image_prompt_path")
        if asset_id and image_path and os.path.exists(image_path):
            assets_to_process.append((asset_id, image_path, asset))
    
    if not assets_to_process:
        print("No valid image prompts found to process")
        return result
    
    # Use ThreadPoolExecutor for parallel 3D generation
    model_results = {}
    max_retries = 5
    
    if ai_platform == "Hunyuan3D" and max_concurrent > 3:
        max_concurrent = 3
    if ai_platform == "Meshy" and max_concurrent > 10:
        max_concurrent = 10

    # Initial attempt
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_model = {
            executor.submit(
                _generate_single_3d_from_image,
                asset_id,
                image_path,
                output_dir,
                ai_platform,
                meshy_api_key,
                meshy_ai_model,
                tencent_secret_id,
                tencent_secret_key,
            ): asset_id
            for asset_id, image_path, _ in assets_to_process
        }
        
        for future in as_completed(future_to_model):
            model_id = future_to_model[future]
            try:
                gen_result = future.result()
                model_results[model_id] = gen_result
                if gen_result.get("error"):
                    print(f"✗ {model_id}: {gen_result['error']}")
                else:
                    print(f"✓ {model_id}: 3D model generated successfully")
            except Exception as e:
                model_results[model_id] = {
                    "model_id": model_id,
                    "main_file_path": None,
                    "error": f"Unexpected error: {str(e)}",
                }
                print(f"✗ {model_id}: Unexpected error: {str(e)}")
    
    # Retry failed models
    for retry_attempt in range(1, max_retries + 1):
        failed_models = [
            (asset_id, image_path, asset)
            for asset_id, image_path, asset in assets_to_process
            if model_results.get(asset_id, {}).get("error") and not model_results.get(asset_id, {}).get("main_file_path")
        ]
        
        if not failed_models:
            break
        
        print(f"\n{'='*80}")
        print(f"Retry attempt {retry_attempt}/{max_retries} for {len(failed_models)} failed model(s)")
        print(f"{'='*80}\n")
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_model = {
                executor.submit(
                    _generate_single_3d_from_image,
                    asset_id,
                    image_path,
                    output_dir,
                    ai_platform,
                    meshy_api_key,
                    meshy_ai_model,
                    tencent_secret_id,
                    tencent_secret_key,
                ): asset_id
                for asset_id, image_path, _ in failed_models
            }
            
            for future in as_completed(future_to_model):
                model_id = future_to_model[future]
                try:
                    gen_result = future.result()
                    model_results[model_id] = gen_result
                    if not gen_result.get("error"):
                        print(f"✓ {model_id}: Successfully generated on retry {retry_attempt}")
                except Exception as e:
                    model_results[model_id] = {
                        "model_id": model_id,
                        "main_file_path": None,
                        "error": f"Retry {retry_attempt} failed: {str(e)}",
                    }
    
    # Update asset_sheet with 3D model info
    for asset in asset_sheet:
        asset_id = asset.get("asset_id")
        if asset_id in model_results:
            gen_result = model_results[asset_id]
            if gen_result.get("main_file_path"):
                asset["main_file_path"] = gen_result["main_file_path"]
                # Add platform tag
                if "tags" not in asset:
                    asset["tags"] = []
                if ai_platform == "Hunyuan3D":
                    if "hunyuan3d" not in asset["tags"]:
                        asset["tags"].append("hunyuan3d")
                else:
                    if "meshy" not in asset["tags"]:
                        asset["tags"].append("meshy")
            if gen_result.get("meshy_model_id"):
                asset["meshy_model_id"] = gen_result["meshy_model_id"]
            if gen_result.get("hunyuan3d_job_id"):
                asset["hunyuan3d_job_id"] = gen_result["hunyuan3d_job_id"]
            if gen_result.get("thumbnail_url"):
                # Download thumbnail and update URLs
                thumbnail_url = gen_result["thumbnail_url"]
                asset["thumbnail_web_url"] = thumbnail_url
                downloaded_path = _download_thumbnail(thumbnail_url, output_dir, asset_id)
                asset["thumbnail_url"] = downloaded_path if downloaded_path else thumbnail_url
            if gen_result.get("error"):
                asset["error"] = gen_result["error"]
    
    # Report results
    successful = sum(1 for r in model_results.values() if r.get("main_file_path"))
    failed = sum(1 for r in model_results.values() if r.get("error") and not r.get("main_file_path"))
    print(f"\n{'='*80}")
    print(f"3D Model Generation: {successful}/{len(model_results)} models generated successfully")
    if failed:
        print(f"Failed: {failed}")
    print(f"{'='*80}\n")
    
    return result


def _download_thumbnail(thumbnail_url: str, output_dir: str, model_id: str) -> Optional[str]:
    """
    Download a thumbnail image to output_dir with model_id as filename.
    
    Args:
        thumbnail_url: URL of the thumbnail to download
        output_dir: Directory to save the thumbnail
        model_id: The model identifier used as the filename
        
    Returns:
        Absolute path to the downloaded thumbnail, or None if failed
    """
    if not thumbnail_url:
        return None
    try:
        response = requests.get(thumbnail_url, timeout=30)
        response.raise_for_status()
        
        # Determine extension from Content-Type or URL
        content_type = response.headers.get('Content-Type', '')
        if 'png' in content_type or thumbnail_url.lower().endswith('.png'):
            ext = '.png'
        elif 'gif' in content_type or thumbnail_url.lower().endswith('.gif'):
            ext = '.gif'
        elif 'webp' in content_type or thumbnail_url.lower().endswith('.webp'):
            ext = '.webp'
        else:
            ext = '.jpg'  # Default to jpg
        
        thumbnail_path = os.path.join(output_dir, f"{model_id}{ext}")
        with open(thumbnail_path, 'wb') as f:
            f.write(response.content)
        return os.path.abspath(thumbnail_path)
    except Exception as e:
        print(f"Failed to download thumbnail for {model_id}: {e}")
        return None


# Fetch a single model using only retrieval sources (Polyhaven, Sketchfab).
def _fetch_single_model_retrieval_only(
    model_id: str,
    model_metadata: Dict[str, Any],
    anyllm_api_key: str,
    anyllm_api_base: str,
    sketchfab_api_key: str,
    output_dir: str = "./models",
    vision_model: str = "openai/gpt-5-mini",
    anyllm_provider: str = "openai",
) -> Dict[str, Any]:
    """
    Fetch a single model using only retrieval sources (Polyhaven, Sketchfab).
    
    This function is optimized for fast parallel retrieval. It does NOT perform
    AI generation (Hunyuan3D/Meshy). Use _fetch_single_model_generation_only for
    models that fail retrieval.
    
    Args:
        model_id: The identifier for the model (used for output filename)
        model_metadata: Dictionary containing prompt, tags, main_file_path, error
        anyllm_api_key: API key for LLM service
        anyllm_api_base: Base URL for LLM API
        sketchfab_api_key: Sketchfab API key
        output_dir: Directory to save downloaded models
        vision_model: Name of the vision-capable LLM model to use for Polyhaven reranking
        anyllm_provider: LLM provider name (default: "openai")
        
    Returns:
        Updated model_metadata dictionary with retrieval result or error indicating
        retrieval failure (which signals the need for generation fallback)
    """
    # Create a copy to avoid modifying the original
    result = model_metadata.copy()
    result["tags"] = list(model_metadata.get("tags", []))
    reflection_log = {}
    
    description = model_metadata.get("description", "")
    tags = result["tags"]
    output_path = os.path.join(output_dir, f"{model_id}.glb")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we should skip Polyhaven
    skip_polyhaven = "no_polyhaven" in tags
    
    if not skip_polyhaven:
        # Try Polyhaven first
        try:
            polyhaven_result = fetch_model_from_polyhaven(
                model_description=description,
                anyllm_api_key=anyllm_api_key,
                anyllm_api_base=anyllm_api_base,
                anyllm_provider=anyllm_provider,
                output_path=output_path,
                vision_model=vision_model,
            )
            
            # Merge polyhaven reflection_log
            if polyhaven_result.get("reflection_log"):
                reflection_log.update(polyhaven_result["reflection_log"])
            
            if polyhaven_result.get("main_file_path") and not polyhaven_result.get("error"):
                # Success with Polyhaven
                result["main_file_path"] = os.path.abspath(polyhaven_result["main_file_path"])
                if "polyhaven" not in tags:
                    result["tags"].append("polyhaven")
                result["polyhaven_id"] = polyhaven_result["polyhaven_id"]
                result["polyhaven_name"] = polyhaven_result["polyhaven_name"]
                result["polyhaven_tags"] = polyhaven_result["polyhaven_tags"]
                # Download thumbnail and update URLs
                thumbnail_url = polyhaven_result["thumbnail_url"]
                result["thumbnail_web_url"] = thumbnail_url
                downloaded_path = _download_thumbnail(thumbnail_url, output_dir, model_id)
                result["thumbnail_url"] = downloaded_path if downloaded_path else thumbnail_url
                result["error"] = None
                result["reflection_log"] = reflection_log
                return result
            else:
                # Polyhaven failed, fall through to Sketchfab
                print(f"Polyhaven failed for model {model_id}: {polyhaven_result.get('error', 'No error message')}")
        except Exception as e:
            # Polyhaven failed with exception, fall through to Sketchfab
            print(f"Polyhaven failed with exception for model {model_id}: {str(e)}")
            pass
    
    # Check if we should skip Sketchfab
    skip_sketchfab = "no_sketchfab" in tags
    
    if not skip_sketchfab:
        # Try Sketchfab
        try:
            sketchfab_result = fetch_model_from_sketchfab(
                model_description=description,
                anyllm_api_key=anyllm_api_key,
                anyllm_api_base=anyllm_api_base,
                anyllm_provider=anyllm_provider,
                sketchfab_api_key=sketchfab_api_key,
                output_path=output_path,
                vision_model=vision_model,
            )
            
            # Merge sketchfab reflection_log
            if sketchfab_result.get("reflection_log"):
                reflection_log.update(sketchfab_result["reflection_log"])
            
            if sketchfab_result.get("main_file_path") and not sketchfab_result.get("error"):
                # Success with Sketchfab
                result["main_file_path"] = os.path.abspath(sketchfab_result["main_file_path"])
                if "sketchfab" not in tags:
                    result["tags"].append("sketchfab")
                result["sketchfab_uid"] = sketchfab_result["sketchfab_uid"]
                result["sketchfab_name"] = sketchfab_result["sketchfab_name"]
                result["sketchfab_tags"] = sketchfab_result["sketchfab_tags"]
                # Download thumbnail and update URLs
                thumbnail_url = sketchfab_result["thumbnail_url"]
                result["thumbnail_web_url"] = thumbnail_url
                downloaded_path = _download_thumbnail(thumbnail_url, output_dir, model_id)
                result["thumbnail_url"] = downloaded_path if downloaded_path else thumbnail_url
                result["error"] = None
                result["reflection_log"] = reflection_log
                return result
            else:
                # Sketchfab failed
                print(f"Sketchfab failed for model {model_id}: {sketchfab_result.get('error', 'No error message')}")
        except Exception as e:
            # Sketchfab failed with exception
            print(f"Sketchfab failed with exception for model {model_id}: {str(e)}")
            pass
    
    # Retrieval failed - mark for generation fallback
    result["error"] = f"Retrieval failed for {model_id} from Polyhaven and Sketchfab"
    result["main_file_path"] = None
    result["reflection_log"] = reflection_log
    return result


# Fetch a single model using only AI generation (Hunyuan3D/Meshy).
def _fetch_single_model_generation_only(
    model_id: str,
    model_metadata: Dict[str, Any],
    anyllm_api_key: str,
    anyllm_api_base: str,
    meshy_api_key: str,
    gemini_api_key: str,
    gemini_api_base: Optional[str] = None,
    output_dir: str = "./models",
    ai_platform: str = "Hunyuan3D",
    gemini_image_model: str = "gemini-3-pro-image-preview",
    meshy_ai_model: str = "latest",
    tencent_secret_id: Optional[str] = None,
    tencent_secret_key: Optional[str] = None,
    vision_model: str = "openai/gpt-5-mini",
    story_summary: str = "",
) -> Dict[str, Any]:
    """
    Fetch a single model using only AI generation (Hunyuan3D or Meshy).
    
    This function is used as a fallback for models that failed retrieval.
    It performs text-to-image-to-3D generation which is slower than retrieval.
    
    Args:
        model_id: The identifier for the model (used for output filename)
        model_metadata: Dictionary containing prompt, tags, main_file_path, error
        anyllm_api_key: API key for LLM service
        anyllm_api_base: Base URL for LLM API
        meshy_api_key: Meshy API key
        gemini_api_key: Gemini API key for image generation
        gemini_api_base: Base URL for Gemini API (optional)
        output_dir: Directory to save downloaded models
        ai_platform: AI platform for 3D generation ("Hunyuan3D" or "Meshy")
        gemini_image_model: Gemini model for image generation
        meshy_ai_model: Meshy AI model version
        tencent_secret_id: Tencent Cloud Secret ID (required for Hunyuan3D)
        tencent_secret_key: Tencent Cloud Secret Key (required for Hunyuan3D)
        vision_model: Name of the vision-capable LLM model
        story_summary: Summary of the story for additional context (default: "")
        
    Returns:
        Updated model_metadata dictionary with generation result
    """
    # Create a copy to avoid modifying the original
    result = model_metadata.copy()
    result["tags"] = list(model_metadata.get("tags", []))
    # Preserve any existing reflection_log (e.g. from prior retrieval attempt)
    reflection_log = dict(model_metadata.get("reflection_log", {}))
    
    prompt = model_metadata.get("prompt", "")
    description = model_metadata.get("description", "")
    output_path = os.path.join(output_dir, f"{model_id}.glb")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create text-to-image-to-3d task and wait for completion
        task_result = text_to_image_to_3d(
            prompt=prompt,
            model_id=model_id,
            description=description,
            output_dir=output_dir,
            ai_platform=ai_platform,
            gemini_api_key=gemini_api_key,
            gemini_api_base=gemini_api_base,
            gemini_image_model=gemini_image_model,
            meshy_ai_model=meshy_ai_model,
            should_remesh=True,
            poll_interval=5,
            timeout=15 * 60,
            meshy_api_key=meshy_api_key,
            tencent_secret_id=tencent_secret_id,
            tencent_secret_key=tencent_secret_key,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            vision_model=vision_model,
            story_summary=story_summary,
        )
    
        # Download the model based on platform
        # Remove old source tags when regenerating with a new source
        source_tags = ["sketchfab", "polyhaven", "hunyuan3d", "meshy"]
        result["tags"] = [t for t in result["tags"] if t not in source_tags]
        
        if ai_platform == "Hunyuan3D":
            model_info = download_model_from_hunyuan3d(
                task_result=task_result,
                output_path=output_path,
            )
            result["main_file_path"] = os.path.abspath(model_info["model_path"])
            result["tags"].append("hunyuan3d")
            result["hunyuan3d_job_id"] = model_info["model_id"]
        if ai_platform == "Meshy":
            model_info = download_model_from_meshy(
                task_result=task_result,
                output_path=output_path,
            )
            result["main_file_path"] = os.path.abspath(model_info["model_path"])
            result["tags"].append("meshy")
            result["meshy_model_id"] = model_info["model_id"]
        
        result["error"] = None
        # Download thumbnail and update URLs
        thumbnail_url = model_info["thumbnail_url"]
        result["thumbnail_web_url"] = thumbnail_url
        downloaded_path = _download_thumbnail(thumbnail_url, output_dir, model_id)
        result["thumbnail_url"] = downloaded_path if downloaded_path else thumbnail_url

        # Propagate image_prompt_scores from text_to_image_to_3d into reflection_log
        if task_result.get("_image_prompt_scores"):
            reflection_log["image_prompt_scores"] = task_result["_image_prompt_scores"]
        result["reflection_log"] = reflection_log

        return result
        
    except Exception as e:
        # AI generation failed
        result["error"] = f"Failed to generate model {model_id}: {str(e)}"
        result["main_file_path"] = None
        result["reflection_log"] = reflection_log
        return result


# Fetch a single model using Polyhaven, Sketchfab, or AI (Hunyuan3D/Meshy).
# NOTE: This function is kept for backward compatibility. The fetch_model() function
# now uses a two-phase approach with _fetch_single_model_retrieval_only and
# _fetch_single_model_generation_only for better throughput optimization.
def _fetch_single_model(
    model_id: str,
    model_metadata: Dict[str, Any],
    anyllm_api_key: str,
    anyllm_api_base: str,
    sketchfab_api_key: str,
    meshy_api_key: str,
    gemini_api_key: str,
    gemini_api_base: Optional[str] = None,
    output_dir: str = "./models",
    ai_platform: str = "Hunyuan3D",
    gemini_image_model: str = "gemini-3-pro-image-preview",
    meshy_ai_model: str = "latest",
    tencent_secret_id: Optional[str] = None,
    tencent_secret_key: Optional[str] = None,
    vision_model: str = "openai/gpt-5-mini",
    anyllm_provider: str = "openai",
    force_genai: bool = False,
    skip_genai: bool = False,
    story_summary: str = "",
) -> Dict[str, Any]:
    """
    Fetch a single model using Polyhaven, Sketchfab, or AI (Hunyuan3D/Meshy).
    
    NOTE: This function combines retrieval and generation in sequence for a single model.
    For batch processing, fetch_model() now uses a two-phase approach that decouples
    retrieval (_fetch_single_model_retrieval_only) from generation 
    (_fetch_single_model_generation_only) for better throughput. This function is kept
    for backward compatibility and edge cases where combined processing is preferred.
    
    Args:
        model_id: The identifier for the model (used for output filename)
        model_metadata: Dictionary containing prompt, tags, main_file_path, error
        anyllm_api_key: API key for LLM service
        anyllm_api_base: Base URL for LLM API
        sketchfab_api_key: Sketchfab API key
        meshy_api_key: Meshy API key
        gemini_api_key: Gemini API key for image generation
        output_dir: Directory to save downloaded models
        ai_platform: AI platform for 3D generation. Options: "Hunyuan3D", "Meshy". Default: "Hunyuan3D".
        gemini_image_model: Gemini model to use for image generation (default: "gemini-3-pro-image-preview")
        meshy_ai_model: Meshy AI model version to use for 3D generation (default: "latest")
        tencent_secret_id: Tencent Cloud Secret ID (required if ai_platform is "Hunyuan3D")
        tencent_secret_key: Tencent Cloud Secret Key (required if ai_platform is "Hunyuan3D")
        vision_model: Name of the vision-capable LLM model to use for Polyhaven reranking (default: "openai/gpt-5-mini")
        force_genai: If True, skip polyhaven and sketchfab, use genai directly (default: False)
        skip_genai: If True, skip AI generation (Hunyuan3D/Meshy), overrides tags (default: False)
        story_summary: Summary of the story for additional context (default: "")
        
    Returns:
        Updated model_metadata dictionary
    """
    # Create a copy to avoid modifying the original
    result = model_metadata.copy()
    result["tags"] = list(model_metadata.get("tags", []))
    reflection_log = {}
    
    model_name = re.sub(r'[_0-9]', '', model_id)
    prompt = model_metadata.get("prompt", "")
    description = model_metadata.get("description", "")
    tags = result["tags"]
    output_path = os.path.join(output_dir, f"{model_id}.glb")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we should skip Polyhaven (force_genai overrides tags)
    skip_polyhaven = force_genai or "no_polyhaven" in tags
    
    if not skip_polyhaven:
        # Try Polyhaven first
        try:
            polyhaven_result = fetch_model_from_polyhaven(
                model_description=description,
                anyllm_api_key=anyllm_api_key,
                anyllm_api_base=anyllm_api_base,
                anyllm_provider=anyllm_provider,
                output_path=output_path,
                vision_model=vision_model,
            )
            
            # Merge polyhaven reflection_log
            if polyhaven_result.get("reflection_log"):
                reflection_log.update(polyhaven_result["reflection_log"])
            
            if polyhaven_result.get("main_file_path") and not polyhaven_result.get("error"):
                # Success with Polyhaven
                result["main_file_path"] = os.path.abspath(polyhaven_result["main_file_path"])
                if "polyhaven" not in tags:
                    result["tags"].append("polyhaven")
                result["polyhaven_id"] = polyhaven_result["polyhaven_id"]
                result["polyhaven_name"] = polyhaven_result["polyhaven_name"]
                result["polyhaven_tags"] = polyhaven_result["polyhaven_tags"]
                # Download thumbnail and update URLs
                thumbnail_url = polyhaven_result["thumbnail_url"]
                result["thumbnail_web_url"] = thumbnail_url
                downloaded_path = _download_thumbnail(thumbnail_url, output_dir, model_id)
                result["thumbnail_url"] = downloaded_path if downloaded_path else thumbnail_url
                result["error"] = None
                result["reflection_log"] = reflection_log
                return result
            else:
                # Polyhaven failed, fall through to Sketchfab
                print(f"Polyhaven failed for model {model_id}: {polyhaven_result.get('error', 'No error message')}")
        except Exception as e:
            # Polyhaven failed with exception, fall through to Sketchfab
            print(f"Polyhaven failed with exception for model {model_id}: {str(e)}")
            pass
    
    # Check if we should skip Sketchfab (force_genai overrides tags)
    skip_sketchfab = force_genai or "no_sketchfab" in tags
    
    if not skip_sketchfab:
        # Try Sketchfab first
        try:
            sketchfab_result = fetch_model_from_sketchfab(
                model_description=description,
                anyllm_api_key=anyllm_api_key,
                anyllm_api_base=anyllm_api_base,
                anyllm_provider=anyllm_provider,
                sketchfab_api_key=sketchfab_api_key,
                output_path=output_path,
                vision_model=vision_model,
            )
            
            # Merge sketchfab reflection_log
            if sketchfab_result.get("reflection_log"):
                reflection_log.update(sketchfab_result["reflection_log"])
            
            if sketchfab_result.get("main_file_path") and not sketchfab_result.get("error"):
                # Success with Sketchfab
                result["main_file_path"] = os.path.abspath(sketchfab_result["main_file_path"])
                if "sketchfab" not in tags:
                    result["tags"].append("sketchfab")
                result["sketchfab_uid"] = sketchfab_result["sketchfab_uid"]
                result["sketchfab_name"] = sketchfab_result["sketchfab_name"]
                result["sketchfab_tags"] = sketchfab_result["sketchfab_tags"]
                # Download thumbnail and update URLs
                thumbnail_url = sketchfab_result["thumbnail_url"]
                result["thumbnail_web_url"] = thumbnail_url
                downloaded_path = _download_thumbnail(thumbnail_url, output_dir, model_id)
                result["thumbnail_url"] = downloaded_path if downloaded_path else thumbnail_url
                result["error"] = None
                result["reflection_log"] = reflection_log
                return result
            else:
                # Sketchfab failed, fall through to Meshy
                print(f"Sketchfab failed for model {model_id}: {sketchfab_result.get('error', 'No error message')}")
        except Exception as e:
            # Sketchfab failed with exception, fall through to Meshy
            print(f"Sketchfab failed with exception for model {model_id}: {str(e)}")
            pass
    
    # check if we should skip AI generation (skip_genai parameter overrides tags)
    should_skip_genai = skip_genai or "no_genai" in tags
    
    if not should_skip_genai:
        # Use AI (Hunyuan3D or Meshy) as fallback or if sketchfab was skipped
        try:
            # Create text-to-image-to-3d task and wait for completion
            task_result = text_to_image_to_3d(
                prompt=prompt,
                model_id=model_id,
                description=description,
                output_dir=output_dir,
                ai_platform=ai_platform,
                gemini_api_key=gemini_api_key,
                gemini_api_base=gemini_api_base,
                gemini_image_model=gemini_image_model,
                meshy_ai_model=meshy_ai_model,
                should_remesh=True,
                poll_interval=5,
                timeout=15 * 60,
                meshy_api_key=meshy_api_key,
                tencent_secret_id=tencent_secret_id,
                tencent_secret_key=tencent_secret_key,
                anyllm_api_key=anyllm_api_key,
                anyllm_api_base=anyllm_api_base,
                vision_model=vision_model,
                story_summary=story_summary,
            )
        
            # Download the model based on platform
            # Remove old source tags when regenerating with a new source
            source_tags = ["sketchfab", "polyhaven", "hunyuan3d", "meshy"]
            result["tags"] = [t for t in result["tags"] if t not in source_tags]
            
            if ai_platform == "Hunyuan3D":
                model_info = download_model_from_hunyuan3d(
                    task_result=task_result,
                    output_path=output_path,
                )
                result["main_file_path"] = os.path.abspath(model_info["model_path"])
                result["tags"].append("hunyuan3d")
                result["hunyuan3d_job_id"] = model_info["model_id"]
            if ai_platform == "Meshy":
                model_info = download_model_from_meshy(
                    task_result=task_result,
                    output_path=output_path,
                )
                result["main_file_path"] = os.path.abspath(model_info["model_path"])
                result["tags"].append("meshy")
                result["meshy_model_id"] = model_info["model_id"]
            
            result["error"] = None
            # Download thumbnail and update URLs
            thumbnail_url = model_info["thumbnail_url"]
            result["thumbnail_web_url"] = thumbnail_url
            downloaded_path = _download_thumbnail(thumbnail_url, output_dir, model_id)
            result["thumbnail_url"] = downloaded_path if downloaded_path else thumbnail_url

            # Propagate image_prompt_scores from text_to_image_to_3d into reflection_log
            if task_result.get("_image_prompt_scores"):
                reflection_log["image_prompt_scores"] = task_result["_image_prompt_scores"]
            result["reflection_log"] = reflection_log

            return result
            
        except Exception as e:
            # AI generation failed
            result["error"] = f"Failed to fetch model {model_id}: {str(e)}"
            result["main_file_path"] = None
            result["reflection_log"] = reflection_log
            return result
    
    # All methods were either skipped or failed - return original metadata with error
    result["error"] = f"No matching model found for {model_id} from any source"
    result["main_file_path"] = None
    result["reflection_log"] = reflection_log
    return result


def fetch_model(
    path_to_input_json: str,
    anyllm_api_key: str,
    anyllm_api_base: str,
    sketchfab_api_key: str,
    meshy_api_key: str,
    gemini_api_key: str,
    gemini_api_base: Optional[str] = None,
    gemini_image_model: str = "gemini-3-pro-image-preview",
    output_dir: str = "./models",
    ai_platform: str = "Hunyuan3D",
    max_concurrent: int = 10,
    meshy_ai_model: str = "latest",
    tencent_secret_id: Optional[str] = None,
    tencent_secret_key: Optional[str] = None,
    vision_model: str = "openai/gpt-5-mini",
    anyllm_provider: str = "openai",
    model_id_list: Optional[List[str]] = None,
    force_genai: bool = False,
    skip_genai: bool = False,
) -> Dict[str, Any]:
    """
    Fetch multiple models using a two-phase approach: retrieval first, then generation.
    
    This function decouples retrieval (Polyhaven, Sketchfab) from generation (Hunyuan3D, Meshy)
    to optimize throughput. Since retrieval is much faster than generation, we:
    
    1. **Phase 1 (Retrieval)**: Run all models through Polyhaven/Sketchfab in parallel with
       high concurrency. This phase completes quickly.
    2. **Phase 2 (Generation)**: After all retrieval attempts complete, models that failed
       retrieval are processed through AI generation in parallel (with lower concurrency
       for Hunyuan3D due to API limits).
    
    This approach prevents slow generation tasks from blocking fast retrieval tasks that
    would otherwise be batched together.
    
    Args:
        path_to_input_json: Path to the input JSON file containing asset_sheet
        anyllm_api_key: API key for LLM service (used by Sketchfab)
        anyllm_api_base: Base URL for LLM API (used by Sketchfab)
        sketchfab_api_key: Sketchfab API key
        meshy_api_key: Meshy API key
        gemini_api_key: Gemini API key for image generation
        gemini_image_model: Gemini model to use for image generation (default: "gemini-3-pro-image-preview")
        output_dir: Directory to save downloaded models (default: "./models")
        ai_platform: AI platform for 3D generation. Options: "Hunyuan3D", "Meshy". Default: "Hunyuan3D".
        max_concurrent: Maximum number of concurrent downloads (default: 10)
        meshy_ai_model: Meshy AI model version to use for 3D generation (default: "latest")
        tencent_secret_id: Tencent Cloud Secret ID (required if ai_platform is "Hunyuan3D")
        tencent_secret_key: Tencent Cloud Secret Key (required if ai_platform is "Hunyuan3D")
        vision_model: Name of the vision-capable LLM model to use for Polyhaven reranking (default: "openai/gpt-5-mini")
        model_id_list: Optional list of model IDs to generate. If None, generates all.
        force_genai: If True, skip polyhaven and sketchfab for all assets, use genai directly. Overrides asset tags. (default: False)
        skip_genai: If True, skip AI generation (Hunyuan3D/Meshy) for all assets, overrides tags. (default: False)
        
    Returns:
        Updated storyboard dictionary with the new asset_sheet containing fetched model information
        
    Example:
        >>> result = fetch_model(
        ...     "./assets_sheet.json",
        ...     llm_key, llm_base, sf_key, meshy_key, gemini_key
        ... )
        >>> print(result["asset_sheet"][0]["main_file_path"])
        "./models/table_1.glb"
    """
    # Load director JSON and create metadata
    with open(path_to_input_json, "r") as f:
        director_result = json.load(f)
    metadata = create_assets_base_metadata(director_result)
    story_summary = director_result.get("story_summary", "")
    
    # Filter by model_id_list if provided
    if model_id_list is not None:
        metadata = {k: v for k, v in metadata.items() if k in model_id_list}
    
    results = {}
    max_generation_retries = 5
    
    # Determine concurrency limits
    # Retrieval can use high concurrency since Polyhaven/Sketchfab are fast
    max_retrieval_concurrent = max_concurrent
    # Generation needs lower concurrency for Hunyuan3D due to API limits
    max_generation_concurrent = 3 if ai_platform == "Hunyuan3D" else max_concurrent
    
    # Separate models that need force_genai (skip retrieval) from others
    models_for_retrieval = {}
    models_for_direct_generation = {}
    
    for model_id, model_data in metadata.items():
        tags = model_data.get("tags", [])
        # Check if this model should skip retrieval entirely
        if force_genai or ("no_polyhaven" in tags and "no_sketchfab" in tags):
            models_for_direct_generation[model_id] = model_data
        else:
            models_for_retrieval[model_id] = model_data
    
    # ============================================================
    # PHASE 1: Parallel Retrieval (Polyhaven, Sketchfab)
    # ============================================================
    if models_for_retrieval:
        print(f"\n{'='*80}")
        print(f"Phase 1: Retrieval - Processing {len(models_for_retrieval)} model(s) via Polyhaven/Sketchfab")
        print(f"{'='*80}\n")
        
        with ThreadPoolExecutor(max_workers=max_retrieval_concurrent) as executor:
            future_to_model = {
                executor.submit(
                    _fetch_single_model_retrieval_only,
                    model_id,
                    model_data,
                    anyllm_api_key,
                    anyllm_api_base,
                    sketchfab_api_key,
                    output_dir,
                    vision_model,
                    anyllm_provider,
                ): model_id
                for model_id, model_data in models_for_retrieval.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_id = future_to_model[future]
                try:
                    result = future.result()
                    results[model_id] = result
                    if not result.get("error"):
                        print(f"✓ Retrieved {model_id} successfully")
                except Exception as e:
                    results[model_id] = {
                        **metadata[model_id],
                        "error": f"Retrieval error: {str(e)}",
                        "main_file_path": None,
                    }
        
        # Clean up aiohttp sessions
        gc.collect()
    
    # Identify models that failed retrieval and need generation
    # Use results[model_id] to preserve reflection_log from retrieval phase
    models_needing_generation = {}
    for model_id, model_data in results.items():
        if model_data.get("error") and not model_data.get("main_file_path"):
            # Check if genai is allowed for this model
            tags = metadata[model_id].get("tags", [])
            if not skip_genai and "no_genai" not in tags:
                # Merge reflection_log from retrieval into original metadata for generation
                gen_metadata = metadata[model_id].copy()
                if model_data.get("reflection_log"):
                    gen_metadata["reflection_log"] = model_data["reflection_log"]
                models_needing_generation[model_id] = gen_metadata
    
    # Add models that were marked for direct generation
    for model_id, model_data in models_for_direct_generation.items():
        tags = model_data.get("tags", [])
        if not skip_genai and "no_genai" not in tags:
            models_needing_generation[model_id] = model_data
        else:
            # Mark as failed if no generation is allowed
            results[model_id] = {
                **model_data,
                "error": f"No matching model found for {model_id} (retrieval skipped, generation disabled)",
                "main_file_path": None,
            }
    
    # ============================================================
    # PHASE 2: Parallel Generation (Hunyuan3D/Meshy)
    # ============================================================
    if models_needing_generation:
        print(f"\n{'='*80}")
        print(f"Phase 2: Generation - Processing {len(models_needing_generation)} model(s) via {ai_platform}")
        print(f"Models: {', '.join(models_needing_generation.keys())}")
        print(f"{'='*80}\n")
        
        with ThreadPoolExecutor(max_workers=max_generation_concurrent) as executor:
            future_to_model = {
                executor.submit(
                    _fetch_single_model_generation_only,
                    model_id,
                    model_data,
                    anyllm_api_key,
                    anyllm_api_base,
                    meshy_api_key,
                    gemini_api_key,
                    gemini_api_base,
                    output_dir,
                    ai_platform,
                    gemini_image_model,
                    meshy_ai_model,
                    tencent_secret_id,
                    tencent_secret_key,
                    vision_model,
                    story_summary,
                ): model_id
                for model_id, model_data in models_needing_generation.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_id = future_to_model[future]
                try:
                    result = future.result()
                    results[model_id] = result
                    if not result.get("error"):
                        print(f"✓ Generated {model_id} successfully")
                except Exception as e:
                    results[model_id] = {
                        **metadata[model_id],
                        "error": f"Generation error: {str(e)}",
                        "main_file_path": None,
                    }
        
        # Clean up aiohttp sessions
        gc.collect()
    
    # ============================================================
    # PHASE 3: Retry failed generation (up to max_generation_retries)
    # ============================================================
    for retry_attempt in range(1, max_generation_retries + 1):
        # Identify models that failed generation (not retrieval failures that were skipped)
        # Preserve reflection_log from prior attempts
        failed_generation_models = {}
        for model_id, model_data in results.items():
            if (model_data is not None 
                and model_data.get("error") 
                and not model_data.get("main_file_path")
                and "Retrieval failed" not in str(model_data.get("error", ""))
                and "No matching model found" not in str(model_data.get("error", ""))
                and not skip_genai
                and "no_genai" not in metadata[model_id].get("tags", [])):
                retry_metadata = metadata[model_id].copy()
                if model_data.get("reflection_log"):
                    retry_metadata["reflection_log"] = model_data["reflection_log"]
                failed_generation_models[model_id] = retry_metadata
        
        if not failed_generation_models:
            break  # No failed models, we're done
        
        print(f"\n{'='*80}")
        print(f"Retry attempt {retry_attempt}/{max_generation_retries} for {len(failed_generation_models)} failed generation(s):")
        for model_id in failed_generation_models.keys():
            print(f"  - {model_id}: {results[model_id].get('error', 'Unknown error')}")
        print(f"{'='*80}\n")
        
        # Retry failed models in parallel
        with ThreadPoolExecutor(max_workers=max_generation_concurrent) as executor:
            future_to_model = {
                executor.submit(
                    _fetch_single_model_generation_only,
                    model_id,
                    model_data,
                    anyllm_api_key,
                    anyllm_api_base,
                    meshy_api_key,
                    gemini_api_key,
                    gemini_api_base,
                    output_dir,
                    ai_platform,
                    gemini_image_model,
                    meshy_ai_model,
                    tencent_secret_id,
                    tencent_secret_key,
                    vision_model,
                    story_summary,
                ): model_id
                for model_id, model_data in failed_generation_models.items()
            }
            
            # Collect retry results
            for future in as_completed(future_to_model):
                model_id = future_to_model[future]
                try:
                    result = future.result()
                    results[model_id] = result
                    if not result.get("error"):
                        print(f"✓ Successfully generated {model_id} on retry {retry_attempt}")
                except Exception as e:
                    results[model_id] = {
                        **metadata[model_id],
                        "error": f"Retry {retry_attempt} failed: {str(e)}",
                        "main_file_path": None,
                    }
        
        # Clean up aiohttp sessions after each retry batch
        gc.collect()
    
    # Final report
    successful = [mid for mid, data in results.items() if data is not None and not data.get("error")]
    failed = [mid for mid, data in results.items() if data is not None and data.get("error")]
    
    print(f"\n{'='*80}")
    print(f"Final Results: {len(successful)}/{len(results)} models fetched successfully")
    if failed:
        print(f"Failed models ({len(failed)}):")
        for model_id in failed:
            print(f"  - {model_id}: {results[model_id].get('error', 'Unknown error')}")
    print(f"{'='*80}\n")
    
    # Convert results to asset sheet and update director result
    new_asset_sheet = convert_metadata_to_asset_sheet(results)
    concept_storyboard = replace_asset_sheet_with_new_asset_sheet(director_result, new_asset_sheet)
    
    return concept_storyboard
