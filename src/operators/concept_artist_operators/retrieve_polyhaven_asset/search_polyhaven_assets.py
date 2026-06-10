"""
Search Polyhaven assets using semantic similarity.

This module provides functionality to search for assets from Polyhaven
using pre-computed embeddings for semantic matching on name and description.
Optionally supports LLM-based reranking for improved relevance.
"""

import os
import sys
import shutil
import tempfile
import threading
from pprint import pprint

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
from pathlib import Path

_MODEL_SOURCE_DIR = Path(__file__).resolve().parent / "bge_small_en_v1.5_model"
_MODEL_CACHE_LOCK = threading.Lock()


def _get_model_cache_dir() -> str:
    """Return a usable model cache directory.

    On Windows the HuggingFace-hub cache structure nested inside the Blender
    extension path easily exceeds MAX_PATH (260 chars).  Work around this by
    copying the bundled model to a short temporary directory and patching the
    ``files_metadata.json`` so recorded sizes match the files on disk (git may
    convert LF → CRLF in text files, inflating their size).

    Thread-safe: the setup is guarded by a process-wide lock so concurrent
    callers (e.g. ThreadPoolExecutor in the concept artist) don't race on
    creating the cache directory.
    """
    if sys.platform != "win32":
        return str(_MODEL_SOURCE_DIR)

    short_dir = Path(tempfile.gettempdir()) / "sb_bge_cache"
    hf_subdir = "models--qdrant--bge-small-en-v1.5-onnx-q"
    marker = short_dir / ".cache_ok"

    if marker.exists():
        return str(short_dir)

    with _MODEL_CACHE_LOCK:
        # Re-check after acquiring the lock: another thread may have finished.
        if marker.exists():
            return str(short_dir)

        # (Re-)create the short cache from the bundled source. Wipe any
        # partial leftovers from a previous failed/interrupted run.
        if short_dir.exists():
            shutil.rmtree(short_dir, ignore_errors=True)
        short_dir.mkdir(parents=True, exist_ok=True)

        # Use the \\?\ extended-length-path prefix so shutil can read
        # source paths that exceed 260 characters.
        src = "\\\\?\\" + str((_MODEL_SOURCE_DIR / hf_subdir).resolve())
        dst = str(short_dir / hf_subdir)
        # dirs_exist_ok handles the (now unlikely) case where the dst was
        # created between the rmtree above and this call.
        shutil.copytree(src, dst, dirs_exist_ok=True)

        # Fix files_metadata.json: update recorded sizes to match the
        # actual (possibly CRLF-inflated) files in the copy.
        _fix_hf_cache_metadata(str(short_dir))
        marker.touch()

    return str(short_dir)


def _fix_hf_cache_metadata(cache_dir: str) -> None:
    """Fix files_metadata.json so recorded sizes match actual files on disk."""
    cache_path = Path(cache_dir)
    for metadata_file in cache_path.glob("*/files_metadata.json"):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            changed = False
            repo_dir = metadata_file.parent
            for rel_path, info in metadata.items():
                full_path = repo_dir / rel_path
                if full_path.exists():
                    actual_size = full_path.stat().st_size
                    if info.get("size") != actual_size:
                        info["size"] = actual_size
                        changed = True

            if changed:
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f)
        except Exception:
            pass


from typing import Dict, List, Any, Optional, Literal

import ssl
import requests
import numpy as np
try:
    from fastembed import TextEmbedding
except Exception as _fastembed_err:
    TextEmbedding = None
    print(f"WARNING [StoryBlender]: fastembed not available: {_fastembed_err}")
    print("Polyhaven semantic search will fall back to basic matching.")

try:
    from ...llm_completion import completion
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from llm_completion import completion

import nest_asyncio
nest_asyncio.apply()

import gc


def get_embeddings_dir() -> Path:
    """Get the directory containing embeddings."""
    return Path(__file__).parent / "polyhaven_embeddings"


def _get_ssl_ca_bundle():
    """
    Return a valid CA bundle for requests.verify.

    Inside Blender's extension Python environment, certifi.where() can point to
    a path that does not exist on disk, causing TLS errors.  This function
    resolves a usable bundle in the following order:
    1. certifi.where() — only if the file actually exists.
    2. ssl.get_default_verify_paths().cafile — system CA file.
    3. ssl.get_default_verify_paths().capath — system CA directory.
    4. False — disables verification as a last resort (logged as a warning).
    """
    try:
        import certifi
        ca = certifi.where()
        if os.path.isfile(ca):
            return ca
    except Exception:
        pass

    paths = ssl.get_default_verify_paths()
    if paths.cafile and os.path.isfile(paths.cafile):
        return paths.cafile
    if paths.capath and os.path.isdir(paths.capath):
        return paths.capath

    print("[StoryBlender] WARNING: No valid TLS CA bundle found; disabling SSL verification.")
    return False


def verify_asset_availability(asset_id: str, timeout: float = 5.0) -> bool:
    """
    Verify if an asset is still available on Polyhaven.

    Args:
        asset_id: The ID of the asset to verify
        timeout: Request timeout in seconds

    Returns:
        True if asset is available (200 OK), False if not found (404) or error
    """
    try:
        response = requests.get(
            f"https://api.polyhaven.com/info/{asset_id}",
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://polyhaven.com/",
            },
            timeout=timeout,
            verify=_get_ssl_ca_bundle(),
        )
        return response.status_code == 200
    except Exception as e:
        print(f"[polyhaven] verify_asset_availability request failed for {asset_id}: {e}")
        # On network error, assume asset is available to avoid false negatives
        return True


def filter_available_assets(
    results: List[Dict[str, Any]], timeout: float = 5.0
) -> List[Dict[str, Any]]:
    """
    Filter search results to only include assets that are still available on Polyhaven.

    Args:
        results: List of asset dictionaries from search
        timeout: Request timeout in seconds for each verification

    Returns:
        Filtered list containing only available assets
    """
    available_results = []
    for asset in results:
        asset_id = asset.get("id")
        if asset_id and verify_asset_availability(asset_id, timeout):
            available_results.append(asset)
    return available_results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Similarity score between 0 and 1
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_batch(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between a query and all embeddings efficiently.

    Args:
        query: Query embedding vector (1D)
        embeddings: Matrix of embeddings (2D: num_items x embedding_dim)

    Returns:
        Array of similarity scores
    """
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(embeddings.shape[0])

    embeddings_norm = np.linalg.norm(embeddings, axis=1)
    # Avoid division by zero
    embeddings_norm = np.where(embeddings_norm == 0, 1e-10, embeddings_norm)

    similarities = np.dot(embeddings, query) / (embeddings_norm * query_norm)
    return similarities


def _chat_with_image(
    api_key: str,
    api_base: str,
    provider: str = "gemini",
    prompt: str = "",
    image_url: str = "",
    vision_model: str = "gemini-3-flash-preview",
) -> str:
    """
    Send a chat completion request with an image using any-llm.

    Args:
        api_key: API key for authentication
        api_base: Base URL for the API endpoint
        provider: LLM provider (default: "gemini")
        prompt: Text prompt to send with the image
        image_url: URL of the image to analyze
        vision_model: Name of the vision-capable LLM model to use (default: "gemini-3-flash-preview")

    Returns:
        The completion response content from the API
    """
    import types
    
    response = completion(
        model=vision_model,
        provider=provider,
        api_key=api_key,
        api_base=api_base,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
    )
    gc.collect()

    if not response.choices[0].message.content:
        raise ValueError("Invalid response from API")
    
    return response.choices[0].message.content


def _extract_score_from_response(response: str) -> int:
    """
    Extract a score (1-10) from the LLM response.

    Args:
        response: The LLM response text

    Returns:
        Extracted score as integer (defaults to 5 if extraction fails)
    """
    # First try to find "Score: X" pattern
    score_match = re.search(r"Score:\s*([1-9]|10)\b", response, re.IGNORECASE)

    # If that fails, look for any number between 1-10 in the response
    if not score_match:
        score_match = re.search(r"\b([1-9]|10)\b", response)

    if score_match:
        return int(score_match.group(1))

    # Default score if extraction fails
    return 5


def rerank_assets_with_llm(
    results: List[Dict[str, Any]],
    asset_type: Literal["hdris", "models", "textures"],
    query_name: Optional[str],
    query_description: Optional[str],
    api_key: str,
    api_base: str,
    provider: str = "gemini",
    vision_model: str = "gemini-3-flash-preview",
) -> List[Dict[str, Any]]:
    """
    Rerank search results using LLM-based scoring.

    Args:
        results: List of asset dictionaries from initial search
        asset_type: Type of asset (hdris, models, textures)
        query_name: Original name query
        query_description: Original description query
        api_key: API key for authentication
        api_base: Base URL for the API endpoint
        provider: LLM provider (default: "gemini")
        vision_model: Name of the vision-capable LLM model to use (default: "gemini-3-flash-preview")

    Returns:
        List of asset dictionaries reranked by LLM score, with added llm_score field
    """
    if not results:
        print("No results to rerank")
        return results

    # If only one result, return it directly
    if len(results) == 1:
        results[0]["llm_score"] = None
        results[0]["llm_explanation"] = None
        return results

    rerank_prompt_templates = {
        "hdris": """You are evaluating an HDRI (360-degree environment image) for use in a 3D scene.

User's Query:
- Desired Name: {query_name}
- Desired Description: {query_description}

HDRI Asset Information:
- Name: {asset_name}
- Description: {asset_description}
- Categories: {asset_categories}
- Tags: {asset_tags}

Please evaluate this HDRI based on how well it matches the user's query. Consider:
- The lighting mood and atmosphere
- Time of day compatibility
- Environment type (indoor/outdoor, urban/nature, etc.)
- Weather conditions
- Overall aesthetic match with the desired description
- Your evaluation should consider both the provided image and the information of the asset.

First, provide a brief explanation (1-2 sentences) of your reasoning.
Then, provide a score on a scale of 1-10.  A score of 10 means the asset is the perfect match for the query, 7-9 means it is a good match, 6 means it is a mediocre match, below 6 means it is a poor match.

Format your response as:
Explanation: [your reasoning]
Score: [number from 1-10]""",
        "models": """You are evaluating a 3D model asset for use in a scene.

User's Query:
- Desired Name: {query_name}
- Desired Description: {query_description}

3D Model Asset Information:
- Name: {asset_name}
- Description: {asset_description}
- Categories: {asset_categories}
- Tags: {asset_tags}
- Dimensions (mm): {asset_dimensions}

Please evaluate this 3D model based on how well it matches the user's query. Consider:
- Visual appearance and style match
- Functional purpose alignment
- Scale and proportion appropriateness
- Material and texture relevance
- Overall suitability for the described use case
- Special rule 1: if the thumbnail of the model is plural (have more than one instance of the model), for example, more than one tree, more than one chair, then the model is not a good match for the query, give it a score of 1.
- Special rule 2: if the thumbnail of the model is in exploded view which you can see the parts of the model, then the model is not a good match for the query, give it a score of 1.
- Your evaluation should consider both the provided image and the information of the asset.

First, provide a brief explanation (1-2 sentences) of your reasoning.
Then, provide a score on a scale of 1-10. A score of 10 means the asset is the perfect match for the query, 7-9 means it is a good match, 6 means it is a mediocre match, below 6 means it is a poor match.

Format your response as:
Explanation: [your reasoning]
Score: [number from 1-10]""",
        "textures": """You are evaluating a texture/material asset for use in 3D rendering.

User's Query:
- Desired Name: {query_name}
- Desired Description: {query_description}

Texture Asset Information:
- Name: {asset_name}
- Description: {asset_description}
- Categories: {asset_categories}
- Tags: {asset_tags}
- Dimensions (mm): {asset_dimensions}

Please evaluate this texture based on how well it matches the user's query. Consider:
- Surface material type match
- Visual appearance and pattern
- Color and tone appropriateness
- Intended use case alignment (terrain, floor, wall, object, etc.)
- Overall aesthetic compatibility
- Your evaluation should consider both the provided image and the information of the asset.

First, provide a brief explanation (1-2 sentences) of your reasoning.
Then, provide a score on a scale of 1-10. A score of 10 means the asset is the perfect match for the query, 7-9 means it is a good match, 6 means it is a mediocre match, below 6 means it is a poor match.

Format your response as:
Explanation: [your reasoning]
Score: [number from 1-10]""",
    }

    prompt_template = rerank_prompt_templates.get(asset_type)
    if not prompt_template:
        raise ValueError(f"No prompt template for asset type: {asset_type}")

    scored_results = []

    for asset in results:
        asset_id = asset.get("asset_id", "unknown")
        asset_name = asset.get("name", "")
        asset_description = asset.get("description", "")
        asset_categories = ", ".join(asset.get("categories", []))
        asset_tags = ", ".join(asset.get("tags", []))
        asset_dimensions = asset.get("dimensions", "N/A")
        thumbnail_url = asset.get("thumbnail_url", "")

        # Upgrade thumbnail resolution for better analysis
        if thumbnail_url and "768" in thumbnail_url:
            thumbnail_url = thumbnail_url.replace("768", "1024")

        # Format the prompt
        prompt = prompt_template.format(
            query_name=query_name or "Not specified",
            query_description=query_description or "Not specified",
            asset_name=asset_name,
            asset_description=asset_description,
            asset_categories=asset_categories,
            asset_tags=asset_tags,
            asset_dimensions=asset_dimensions,
        )

        try:
            # Use chat_with_image to get the score
            response = _chat_with_image(
                api_key=api_key,
                api_base=api_base,
                provider=provider,
                prompt=prompt,
                image_url=thumbnail_url,
                vision_model=vision_model
            )
            score = _extract_score_from_response(response)

            # Extract explanation
            explanation_match = re.search(
                r"Explanation:\s*(.+?)(?=Score:|$)", response, re.IGNORECASE | re.DOTALL
            )
            explanation = (
                explanation_match.group(1).strip() if explanation_match else None
            )

            asset_copy = asset.copy()
            asset_copy["llm_score"] = score
            asset_copy["llm_explanation"] = explanation
            scored_results.append(asset_copy)

            # Early termination if we find a perfect match
            if score >= 9:
                # Add remaining assets with None scores
                for remaining_asset in results:
                    if remaining_asset.get("asset_id") != asset_id and remaining_asset.get(
                        "id"
                    ) not in [r.get("asset_id") for r in scored_results]:
                        remaining_copy = remaining_asset.copy()
                        remaining_copy["llm_score"] = None
                        remaining_copy["llm_explanation"] = None
                        scored_results.append(remaining_copy)
                break

        except Exception as e:
            # Assign default score if scoring fails
            asset_copy = asset.copy()
            asset_copy["llm_score"] = 5
            asset_copy["llm_explanation"] = f"Scoring failed: {str(e)}"
            print("Scoring failed: ", str(e))
            scored_results.append(asset_copy)

    # Sort by LLM score descending (None scores go to the end)
    scored_results.sort(
        key=lambda x: (x.get("llm_score") is not None, x.get("llm_score") or 0),
        reverse=True,
    )

    return scored_results


class PolyhavenAssetSearcher:
    """
    Searcher for Polyhaven assets using semantic similarity.

    This class loads pre-computed embeddings and provides search functionality
    based on name and/or description similarity.
    """

    def __init__(self, lazy_load: bool = True):
        """
        Initialize the searcher.

        Args:
            lazy_load: If True, load embeddings on first search. If False, load all immediately.
        """
        self._embeddings_dir = get_embeddings_dir()
        self._embedding_model: Optional[TextEmbedding] = None
        self._loaded_data: Dict[str, Dict[str, Any]] = {}

        if not lazy_load:
            self._ensure_model_loaded()
            for asset_type in ("hdris", "models", "textures"):
                self._load_asset_type(asset_type)

    def _ensure_model_loaded(self):
        """Ensure the embedding model is loaded and return it."""
        if self._embedding_model is None:
            if TextEmbedding is None:
                raise RuntimeError(
                    "fastembed is not available (onnxruntime failed to load). "
                    "Polyhaven semantic search is unavailable."
                )
            cache_dir = _get_model_cache_dir()
            self._embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=cache_dir, local_files_only=True)
        return self._embedding_model

    def _load_asset_type(self, asset_type: str) -> Dict[str, Any]:
        """
        Load embeddings and metadata for a specific asset type.

        Args:
            asset_type: Type of asset (hdris, models, textures)

        Returns:
            Dictionary containing loaded data
        """
        if asset_type in self._loaded_data:
            return self._loaded_data[asset_type]

        embeddings_dir = self._embeddings_dir

        # Load name embeddings
        name_emb_path = embeddings_dir / f"{asset_type}_name_embeddings.npy"
        if not name_emb_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found for {asset_type}. "
                f"Please run process_polyhaven_assets_info.py first."
            )
        name_embeddings = np.load(name_emb_path)

        # Load description embeddings
        desc_emb_path = embeddings_dir / f"{asset_type}_description_embeddings.npy"
        description_embeddings = np.load(desc_emb_path)

        # Load index to ID mapping
        index_to_id_path = embeddings_dir / f"{asset_type}_index_to_id.json"
        with open(index_to_id_path, "r", encoding="utf-8") as f:
            index_to_id = json.load(f)

        # Load metadata
        metadata_path = embeddings_dir / f"{asset_type}_metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self._loaded_data[asset_type] = {
            "name_embeddings": name_embeddings,
            "description_embeddings": description_embeddings,
            "index_to_id": index_to_id,
            "metadata": metadata,
        }

        return self._loaded_data[asset_type]

    def search(
        self,
        asset_type: Literal["hdris", "models", "textures"],
        name: Optional[str] = None,
        description: Optional[str] = None,
        returned_count: int = 10,
        categories_limitation: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for assets based on name and/or description similarity.

        Args:
            asset_type: Type of asset to search ("hdris", "models", or "textures")
            name: Optional name query (e.g., "Wooden Table")
            description: Optional description query (e.g., "Simple wooden table with warm brown grain")
            returned_count: Number of results to return
            categories_limitation: Optional list of category strings. If provided, only assets
                that contain ALL specified categories will be returned.

        Returns:
            List of asset dictionaries with similarity scores, sorted by combined score descending.
            Each dict contains:
                - id: Asset ID
                - All original metadata fields
                - name_similarity_score: float or None
                - description_similarity_score: float or None
                - combined_score: float

        Raises:
            ValueError: If asset_type is invalid or both name and description are None
            FileNotFoundError: If embeddings have not been generated
        """
        if name is None and description is None:
            raise ValueError("At least one of 'name' or 'description' must be provided")

        if returned_count <= 0:
            raise ValueError("returned_count must be positive")

        # Load data for this asset type
        data = self._load_asset_type(asset_type)
        model = self._ensure_model_loaded()

        name_embeddings = data["name_embeddings"]
        description_embeddings = data["description_embeddings"]
        index_to_id = data["index_to_id"]
        metadata = data["metadata"]

        num_assets = len(index_to_id)

        # Calculate similarity scores
        name_scores: Optional[np.ndarray] = None
        description_scores: Optional[np.ndarray] = None

        if name is not None:
            query_name_embedding = list(model.embed([name]))[0]
            name_scores = cosine_similarity_batch(query_name_embedding, name_embeddings)

        if description is not None:
            query_desc_embedding = list(model.embed([description]))[0]
            description_scores = cosine_similarity_batch(
                query_desc_embedding, description_embeddings
            )

        # Calculate combined scores
        combined_scores = np.zeros(num_assets)

        if name_scores is not None and description_scores is not None:
            # Both provided: 0.3 weight for name, 0.7 weight for description
            combined_scores = 0.3 * name_scores + 0.7 * description_scores
        elif name_scores is not None:
            # Only name provided: 1.0 weight
            combined_scores = name_scores
        else:
            # Only description provided: 1.0 weight
            combined_scores = description_scores

        # Get all indices sorted by combined score descending
        sorted_indices = np.argsort(combined_scores)[::-1]

        # Build results with optional category filtering
        results = []
        for idx in sorted_indices:
            if len(results) >= returned_count:
                break

            idx = int(idx)
            asset_id = index_to_id[idx]
            asset_metadata = metadata[asset_id].copy()

            # Apply categories_limitation filter if provided
            if categories_limitation is not None:
                asset_categories = asset_metadata.get("categories", [])
                # Check if all required categories are present in the asset's categories
                if not all(cat in asset_categories for cat in categories_limitation):
                    continue

            # Add id field
            asset_metadata["id"] = asset_id

            # Add similarity scores
            asset_metadata["name_similarity_score"] = (
                float(name_scores[idx]) if name_scores is not None else None
            )
            asset_metadata["description_similarity_score"] = (
                float(description_scores[idx])
                if description_scores is not None
                else None
            )
            asset_metadata["combined_score"] = float(combined_scores[idx])

            results.append(asset_metadata)

        return results


def search_polyhaven_assets(
    asset_type: Literal["hdris", "models", "textures"],
    name: Optional[str] = None,
    description: Optional[str] = None,
    returned_count: int = 10,
    threshold_score: float = 0.6,
    rerank_with_llm: bool = True,
    threshold_llm: int = 6,
    anyllm_api_key: Optional[str] = None,
    anyllm_api_base: Optional[str] = None,
    anyllm_provider: str = "gemini",
    vision_model: str = "gemini-3-flash-preview",
    categories_limitation: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search for Polyhaven assets based on name and/or description similarity.

    This function performs semantic search using pre-computed embeddings, with optional
    LLM-based reranking for improved relevance. The search pipeline is:
    1. Semantic search using embeddings (weighted: 0.3 name + 0.7 description)
    2. Filter by combined_score threshold
    3. Verify asset availability via Polyhaven API (removes 404 assets)
    4. Optional LLM reranking with thumbnail analysis
    5. Filter by LLM score threshold (if reranking enabled)

    Args:
        asset_type: Type of asset to search ("hdris", "models", or "textures")
        name: Optional name query (e.g., "Wooden Table")
        description: Optional description query (e.g., "Simple wooden table with warm brown grain")
        returned_count: Number of initial results from semantic search
        threshold_score: Minimum combined_score threshold (default 0.6). Assets below this are removed.
        rerank_with_llm: If True, use LLM to rerank results based on thumbnail and metadata
        threshold_llm: Minimum LLM score threshold (default 6). Assets with llm_score below this
            are removed after reranking. Assets with None llm_score are also removed.
        anyllm_api_key: API key for any-llm authentication (required if rerank_with_llm=True)
        anyllm_api_base: Base URL for the any-llm API endpoint (required if rerank_with_llm=True)
        anyllm_provider: LLM provider (default: "gemini")
        vision_model: Name of the vision-capable LLM model to use for reranking (default: "gemini-3-flash-preview")
        categories_limitation: Optional list of category strings. If provided, only assets
            that contain ALL specified categories will be returned (e.g., ["sunrise-sunset", "pure skies"]).

    Returns:
        List of asset dictionaries sorted by relevance. May be empty if all assets are
        filtered out by thresholds or availability checks. Each dict contains:
            - id: Asset ID
            - name, description, categories, tags, thumbnail_url, etc. (original metadata)
            - name_similarity_score: float or None
            - description_similarity_score: float or None
            - combined_score: float (0.3 * name + 0.7 * description similarity)
            - llm_score: int (1-10) (only if rerank_with_llm=True)
            - llm_explanation: str (only if rerank_with_llm=True)

    Raises:
        ValueError: If asset_type is invalid, both name and description are None,
                   or rerank_with_llm=True but API credentials are missing
        FileNotFoundError: If embeddings have not been generated

    Example:
        >>> # Basic search without reranking
        >>> results = search_polyhaven_assets(
        ...     asset_type="models",
        ...     name="Wooden Table",
        ...     description="Simple wooden table with warm brown grain",
        ...     returned_count=5
        ... )
        >>> for r in results:
        ...     print(f"{r['id']}: {r['combined_score']:.4f}")

        >>> # Search with LLM reranking
        >>> results = search_polyhaven_assets(
        ...     asset_type="models",
        ...     name="Wooden Table",
        ...     description="Simple wooden table with warm brown grain",
        ...     returned_count=5,
        ...     rerank_with_llm=True,
        ...     anyllm_api_key="your-api-key",
        ...     anyllm_api_base="https://api.example.com/v1",
        ...     anyllm_provider="gemini"
        ... )
        >>> for r in results:
        ...     print(f"{r['id']}: LLM={r['llm_score']}, Combined={r['combined_score']:.4f}")
    """
    # Validate reranking parameters
    if rerank_with_llm:
        if not anyllm_api_key:
            raise ValueError("anyllm_api_key is required when rerank_with_llm=True")

    # Lazy initialization of searcher (cached as function attribute)
    if not hasattr(search_polyhaven_assets, "_searcher"):
        search_polyhaven_assets._searcher = PolyhavenAssetSearcher(lazy_load=True)

    results = search_polyhaven_assets._searcher.search(
        asset_type=asset_type,
        name=name,
        description=description,
        returned_count=returned_count,
        categories_limitation=categories_limitation,
    )

    # Filter by combined_score threshold
    if results:
        results = [r for r in results if r.get("combined_score", 0) >= threshold_score]

    # Filter out assets that are no longer available on Polyhaven
    if results:
        results = filter_available_assets(results)
    
    # Apply LLM reranking if requested
    if rerank_with_llm and results:
        results = rerank_assets_with_llm(
            results=results,
            asset_type=asset_type,
            query_name=name,
            query_description=description,
            api_key=anyllm_api_key,
            api_base=anyllm_api_base,
            provider=anyllm_provider,
            vision_model=vision_model,
        )
        # Filter by LLM score threshold (remove None and scores below threshold)
        results = [
            r for r in results
            if r.get("llm_score") is not None and r.get("llm_score", 0) >= threshold_llm
        ]
    
    return results
