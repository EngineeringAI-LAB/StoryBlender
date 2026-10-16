import os
try:
    from ..llm_completion import completion
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from llm_completion import completion
try:
    from ...proxy_transport import requests_for
except ImportError:
    from proxy_transport import requests_for
requests = requests_for("environment_assets")
import gc
import warnings
from pathlib import Path
import random

# Chat with image
def chat_with_image(
    anyllm_api_key: str,
    anyllm_api_base: str,
    anyllm_provider: str = "openai",
    prompt: str = "",
    image_url: str = "",
    model: str = "gpt-4o-mini"
):
    """
    Send a chat completion request with an image using any-llm.
    
    Args:
        anyllm_api_key: API key for authentication
        anyllm_api_base: Base URL for the API endpoint
        anyllm_provider: LLM provider (default: "openai")
        prompt: Text prompt to send with the image
        image_url: URL of the image to analyze
        model: Model to use for completion
    
    Returns:
        The completion response from the API
    """
    response = completion(
        model=model,
        provider=anyllm_provider,
        api_key=anyllm_api_key,
        api_base=anyllm_api_base,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]
    )
    gc.collect()

    # check if response is valid
    if not response.choices[0].message.content:
        raise ValueError("Invalid response from API")
    
    return response.choices[0].message.content


def categories_selection(
    scene_description,
    anyllm_api_key=None,
    anyllm_api_base=None,
    anyllm_provider="openai",
    model="gemini-2.5-flash"
):
    """
    Select appropriate categories for a given scene description using an AI model.
    
    Args:
        scene_description (str): Description of the scene lighting (e.g., "afternoon woods")
        anyllm_api_key (str, optional): API key for the model service
        anyllm_api_base (str, optional): API base URL for the model service
        anyllm_provider (str, optional): LLM provider (default: "openai")
        model (str, optional): Model to use for completion
    
    Returns:
        str: Comma-separated string of selected categories, one from each category group
    
    Raises:
        ValueError: If scene_description is empty or None
        Exception: If API call fails or response parsing fails
    """
    # Input validation
    if not scene_description or not isinstance(scene_description, str):
        raise ValueError("scene_description must be a non-empty string")
    
    scene_description = scene_description.strip()
    if not scene_description:
        raise ValueError("scene_description cannot be empty or whitespace only")

    # Categories definition
    categories = {
        "Source": "natural, man made",
        "Location & Context": "outdoor, indoor, aerial, industrial, road, floor",
        "Material": "terrain, brick, concrete, plaster-concrete, plaster, metal, cobblestone, asphalt, tiles, rock, sand, sandstone, gravel, snow"
    }

    # Create the prompt for category selection
    categories_text = ""
    for category, options in categories.items():
        categories_text += f"{category}: {options}\n"
    
    prompt = f"""You are a professional 3D artist for selecting terrain materials. You need to select some tags in order to find terrains textures. Given the scene description: "{scene_description}"

Please select exactly ONE option from each of the following category groups:

{categories_text}

Rules:
1. Select exactly one option from each category group
2. Base your selection on the scene description provided
3. Return only the selected options as a comma-separated string
4. Use the exact text as provided in the options (case-sensitive)
5. Do not include category names, only the selected values
6. Order: Source, Location & Context, Material
7. For Material, please select "terrain" unless the scene description mentions material that directly match with the provided categories.

Example format: natural, outdoor, cobblestone

Your selection:"""

    try:
        # Make the API call
        messages = [{"content": prompt, "role": "user"}]
        response = completion(
            model=model,
            provider=anyllm_provider,
            api_key=anyllm_api_key,
            api_base=anyllm_api_base,
            messages=messages
        )
        gc.collect()
        
        # Extract the response content
        if not response or not response.choices:
            raise Exception("Invalid response structure from API")
        
        message_content = response.choices[0].message.content.strip()
        
        # Clean and validate the response
        selected_categories = message_content.strip()
        
        # Remove any extra text and extract just the comma-separated values
        # Look for a line that contains comma-separated values
        lines = selected_categories.split('\n')
        for line in lines:
            line = line.strip()
            if ',' in line and not line.startswith(('Based', 'Given', 'For', 'The', 'Here')):
                # This looks like our comma-separated result
                selected_categories = line
                break
        
        # Validate the format (should have exactly 4 commas for 5 categories)
        parts = [part.strip() for part in selected_categories.split(',')]
        if len(parts) != 3:
            raise Exception(f"Expected 3 categories, got {len(parts)}: {selected_categories}")
        
        # Validate each selection against available options
        category_keys = list(categories.keys())
        for i, selected in enumerate(parts):
            if i < len(category_keys):
                available_options = [opt.strip() for opt in categories[category_keys[i]].split(',')]
                if selected not in available_options:
                    # Try to find a close match (case-insensitive)
                    selected_lower = selected.lower()
                    for option in available_options:
                        if option.lower() == selected_lower:
                            parts[i] = option  # Use the correct case
                            break
                    else:
                        raise Exception(f"Invalid selection '{selected}' for category '{category_keys[i]}'. Available options: {available_options}")
        
        return ','.join(parts)
        
    except Exception as e:
        # Fallback to default categories if AI selection fails
        print(f"AI category selection failed: {e}, using default categories")
        return "outdoor, natural, cobblestone"

def search_polyhaven_textures(categories=None, headers=None, random_order=True, max_number_of_results=10):
    """
    Search for texture assets from Polyhaven with optional filtering.
    
    Args:
        categories (str, optional): Comma-separated string of categories to filter by
        headers (dict, optional): HTTP headers for the API request
        random_order (bool, optional): If True, return a random subset of results. If False, return the first N results. Defaults to True.
        max_number_of_results (int, optional): Maximum number of assets to return. Defaults to 20.
    
    Returns:
        dict: Response containing texture assets and metadata with the following schema:
            {
                'assets': {
                    '<asset_id>': {
                        'name': str,  # Display name of the asset
                        'description': str,  # Detailed description of the texture
                        'authors': dict,  # Author names mapped to their contribution (e.g., {'Greg Zaal': 'All'})
                        'categories': list[str],  # List of category tags
                        'tags': list[str],  # Additional descriptive tags
                        'thumbnail_url': str,  # URL to thumbnail image
                        'type': int,  # Asset type (1 for texture)
                        'date_published': int,  # Unix timestamp of publication date
                        'date_taken': int,  # Unix timestamp when photo was taken
                        'download_count': int,  # Number of downloads
                        'evs_cap': int,  # EV range captured
                        'max_resolution': list[int],  # [width, height] in pixels
                        'files_hash': str,  # Hash of the asset files
                        'sponsors': list,  # List of sponsor IDs or dicts with 'name' and 'url'
                        'coords': list[float],  # Optional: [latitude, longitude]
                        'backplates': bool,  # Optional: Whether backplates are available
                        'whitebalance': int  # Optional: White balance in Kelvin
                    },
                    ...
                },
                'total_count': int,  # Total number of assets found
                'returned_count': int  # Number of assets returned (limited to max_number_of_results)
            }
            
            On error:
            {
                'error': str  # Error message describing what went wrong
            }
    
    Example:
        >>> result = search_polyhaven_textures(categories="outdoor,natural,cobblestone")
        >>> print(result['returned_count'])  # Number of assets returned
        >>> for asset_id, asset_data in result['assets'].items():
        ...     print(f"{asset_id}: {asset_data['name']}")
    """
    try:
        url = "https://api.polyhaven.com/assets"
        params = {"type": "textures"}

        if categories:
            params["categories"] = categories

        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            # Limit the response size to avoid overwhelming Blender
            assets = response.json()
            # Select a subset of assets based on parameters
            items_list = list(assets.items())
            k = min(max_number_of_results, len(items_list))
            if random_order:
                selected_items = items_list if k == len(items_list) else random.sample(items_list, k)
            else:
                selected_items = items_list[:k]
            limited_assets = dict(selected_items)

            return {"assets": limited_assets, "total_count": len(assets), "returned_count": len(limited_assets)}
        else:
            return {"error": f"API request failed with status code {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def rank_terrain_texture(
    search_result,
    scene_description,
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider="openai",
    model="gpt-4o-mini"
):
    """
    Rank terrain textures from Polyhaven search results based on how well they fit the scene description.

    Args:
        search_result: Dictionary returned from search_polyhaven_textures containing an 'assets' dict.
        scene_description: String describing the scene's plot and appearance (e.g., "afternoon woods, wet cobblestone").
        anyllm_api_key: API key for authentication.
        anyllm_api_base: Base URL for the API endpoint.
        anyllm_provider: LLM provider (default: "openai").
        model: Model to use for completion.

    Returns:
        str: The asset_id of the best matching terrain texture.
    """
    import re

    # Check if search_result has error
    if "error" in search_result:
        raise ValueError(f"Search result contains error: {search_result['error']}")

    # Get assets from search result
    assets = search_result.get("assets", {})

    if not assets:
        raise ValueError("No assets found in search result")

    # If only one texture, return it directly
    if len(assets) == 1:
        return list(assets.keys())[0]

    # Score each texture
    scored_textures = []

    for asset_id, asset_data in assets.items():
        name = asset_data.get("name", "")
        description = asset_data.get("description", "")
        thumbnail_url = asset_data.get("thumbnail_url", "")

        # Upgrade thumbnail resolution from 256 to 1024
        if thumbnail_url and "256" in thumbnail_url:
            thumbnail_url = thumbnail_url.replace("256", "1024")

        # Create prompt for scoring (terrain texture focus)
        prompt = f"""You are evaluating a PBR terrain texture for use as ground material in a 3D scene.

Scene Description: {scene_description}

Texture Information:
- Name: {name}
- Description: {description}

Evaluate how well this texture fits the scene description. This texture will be used as ground material in the 3D scene. Consider the following factors:
- Material match (e.g., cobblestone, gravel, sand, rock, snow, asphalt, tiles)
- Pattern scale believability for human-scale scenes
- Tiling artifacts visibility and uniformity
- Surface condition (dry/wet, rough/smooth, worn/new, moss/dirt presence)
- Color and value harmony with the setting
- Context suitability (indoor/outdoor, road/floor/path, natural/man-made)
- Overall visual fit for the scene description, pay close attention to the preview image

First, provide a brief explanation (1-2 sentences) of your reasoning.
Then, provide a score on a scale of 1-10. For each mismatched factor, deduct 1 point. On a perfect match, give 10 points.

Format your response as:
Explanation: [your reasoning]
Score: [number from 1-10]"""

        try:
            # Use chat_with_image to get the score
            response = chat_with_image(
                anyllm_api_key=anyllm_api_key,
                anyllm_api_base=anyllm_api_base,
                anyllm_provider=anyllm_provider,
                prompt=prompt,
                image_url=thumbnail_url,
                model=model
            )

            # Extract score from response using regex
            # First try to find "Score: X" pattern
            score_match = re.search(r'Score:\s*([1-9]|10)\b', response, re.IGNORECASE)

            # If that fails, look for any number between 1-10 in the response
            if not score_match:
                score_match = re.search(r'\b([1-9]|10)\b', response)

            if score_match:
                score = int(score_match.group(1))
            else:
                # If no valid score found, default to 5
                score = 5

            scored_textures.append({
                "asset_id": asset_id,
                "score": score,
                "name": name
            })

            print({
                "asset_id": asset_id,
                "score": score,
                "name": name
            })

            # If we find a texture with score == 10, use it immediately
            if score == 10:
                return asset_id

        except Exception as e:
            # Assign a default score if scoring fails
            scored_textures.append({
                "asset_id": asset_id,
                "score": 5,
                "name": name
            })

    # If no texture scored == 10, return the one with highest score
    if scored_textures:
        print(scored_textures)
        best_tex = max(scored_textures, key=lambda x: x["score"])
        return best_tex["asset_id"]

    # Fallback: return first asset if all scoring failed
    return list(assets.keys())[0]
