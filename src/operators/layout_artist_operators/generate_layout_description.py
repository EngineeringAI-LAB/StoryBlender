"""Helpers for building prompts and calling Gemini to generate 3D layout
descriptions for production designers."""

from typing import Any, Dict, List, Optional, Literal, Tuple
from collections import Counter

from copy import deepcopy

from pydantic import BaseModel

import json
import base64
import math
import mimetypes
import os
import gc
import warnings
import time

import numpy as np
import trimesh

try:
    from ..llm_completion import completion
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from llm_completion import completion

class SceneSize(BaseModel):
    x: float
    x_negative: float
    y: float
    y_negative: float

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


Direction = Literal[
    "facing",
    "facing_away",
    "left_side_facing",
    "right_side_facing",
]


class LayoutObject(BaseModel):
    """Schema for a single asset's layout in a scene.
    
    Attributes:
        asset_id: Unique identifier for the asset.
        location: The 3D position (x, y, z) of the asset.
        rotation: The Euler rotation (x, y, z) in degrees.
        anchor_asset_id: The asset_id of the anchor object for spatial relationship.
        relationship: Spatial relationship to the anchor asset (e.g., on_top_of, behind).
        contact: Whether this asset is in direct geometry contact with the anchor.
        direction: The orientation of this asset relative to its anchor_asset_id.
            - "facing": The front of this asset faces the anchor (e.g., character looking at another).
            - "facing_away": The back of this asset faces the anchor (e.g., turned away from someone).
            - "left_side_facing": The left side of this asset faces the anchor.
            - "right_side_facing": The right side of this asset faces the anchor.
            - null: Not applicable or no specific directional orientation.
            Recommended for assets with distinct views from different angles (characters,
            vehicles, pianos, etc.). Optional for symmetric objects.
    """
    asset_id: str
    location: Location
    rotation: Rotation
    anchor_asset_id: Optional[str] = None
    relationship: Optional[Relationship] = None
    contact: Optional[bool] = None
    direction: Optional[Direction] = None


class SceneLayout(BaseModel):
    scene_id: str
    scene_size: SceneSize
    assets: List[LayoutObject]


class LayoutDescriptionOutput(BaseModel):
    scenes: List[SceneLayout]


class AssetModificationTransform(BaseModel):
    """Schema for target transform of an asset modification.
    
    Attributes:
        asset_id: Unique identifier for the asset being modified.
        target_location: The target 3D position (x, y, z) for the asset in this shot.
        target_rotation: The target Euler rotation (x, y, z) in degrees for this shot.
        anchor_asset_id: The asset_id of the anchor object for spatial relationship in this shot.
        relationship: Spatial relationship to the anchor asset during this shot.
        contact: Whether this asset is in direct geometry contact with the anchor in this shot.
        direction: The orientation of this asset relative to its anchor_asset_id in this shot.
            Same values as LayoutObject.direction: facing, facing_away, left_side_facing,
            right_side_facing, or null.
    """
    asset_id: str
    target_location: Location
    target_rotation: Rotation
    anchor_asset_id: Optional[str] = None
    relationship: Optional[Relationship] = None
    contact: Optional[bool] = None
    direction: Optional[Direction] = None


class ShotAssetModifications(BaseModel):
    """Schema for asset modifications in a specific shot."""
    shot_id: int
    asset_modifications: List[AssetModificationTransform]


class SingleSceneLayout(BaseModel):
    """Schema for single scene layout without scene_id (used for per-scene generation)."""
    scene_size: SceneSize
    assets: List[LayoutObject]
    shot_asset_modifications: Optional[List[ShotAssetModifications]] = None


class SingleSceneLayoutOutput(BaseModel):
    """Schema for single scene layout output."""
    scene: SingleSceneLayout


def process_url_or_path(url_or_path: str) -> str:
    """Process a URL or local file path and return a URL suitable for API calls.
    
    Args:
        url_or_path: Either a web URL or a path to a local image file.
        
    Returns:
        The URL as-is if it's a web URL, or a base64 data URL if it's a local path.
    """
    if url_or_path.startswith(('http://', 'https://')):
        return url_or_path
    
    # It's a local file path, convert to base64 data URL
    if not os.path.isfile(url_or_path):
        raise FileNotFoundError(f"File not found: {url_or_path}")
    
    mime_type, _ = mimetypes.guess_type(url_or_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    
    with open(url_or_path, 'rb') as f:
        image_data = f.read()
    
    base64_data = base64.b64encode(image_data).decode('utf-8')
    return f"data:{mime_type};base64,{base64_data}"


def _format_dimension(value: Any) -> str:
    """Return a string representation for optional numeric values."""
    if value is None:
        return "N/A"
    return f"{value}"


def get_reference_scene_layout(storyboard_script: Dict[str, Any], reference_scene_id: int) -> Optional[Dict[str, Any]]:
    """Get the layout_description of a reference scene.
    
    Args:
        storyboard_script: The full storyboard script containing all scenes.
        reference_scene_id: The scene_id of the reference scene.
        
    Returns:
        The layout_description dict of the reference scene, or None if not found.
    """
    if "scene_details" not in storyboard_script:
        return None
    
    for scene in storyboard_script["scene_details"]:
        if scene.get("scene_id") == reference_scene_id:
            scene_setup = scene.get("scene_setup", {})
            layout_desc = scene_setup.get("layout_description")
            if isinstance(layout_desc, dict):
                return layout_desc
    return None


def extract_single_scene_data(storyboard_script: Dict[str, Any], scene_id: int) -> Dict[str, Any]:
    """Extract data for a single scene from the full storyboard script.
    
    Args:
        storyboard_script: The full storyboard script containing all scenes.
        scene_id: The scene_id to extract.
        
    Returns:
        A dict containing only the data for the specified scene, with the same
        structure as storyboard_script but filtered to one scene.
    """
    result = {}
    
    # Extract the single scene from storyboard_outline
    if "storyboard_outline" in storyboard_script:
        for scene in storyboard_script["storyboard_outline"]:
            if scene.get("scene_id") == scene_id:
                result["storyboard_outline"] = [scene]
                break
    
    # Extract the single scene from scene_details
    scene_detail = None
    if "scene_details" in storyboard_script:
        for scene in storyboard_script["scene_details"]:
            if scene.get("scene_id") == scene_id:
                scene_detail = scene
                result["scene_details"] = [scene]
                break
    
    # Extract shot_details for this scene
    if "shot_details" in storyboard_script:
        scene_shots = [
            shot for shot in storyboard_script["shot_details"]
            if shot.get("scene_id") == scene_id
        ]
        if scene_shots:
            result["shot_details"] = scene_shots
    
    # Extract only the assets that appear in this scene
    if "asset_sheet" in storyboard_script and scene_detail:
        scene_setup = scene_detail.get("scene_setup", {})
        asset_ids_in_scene = set(scene_setup.get("asset_ids", []))
        # Also include assets introduced by shot "add" modifications
        for shot in result.get("shot_details", []):
            for mod in (shot.get("asset_modifications") or []):
                if isinstance(mod, dict) and mod.get("modification_type") == "add":
                    aid = mod.get("asset_id")
                    if aid:
                        asset_ids_in_scene.add(aid)
        scene_assets = [
            asset for asset in storyboard_script["asset_sheet"]
            if asset.get("asset_id") in asset_ids_in_scene
        ]
        result["asset_sheet"] = scene_assets
    
    return result


def create_layout_description_prompt(
    storyboard_script: Dict[str, Any],
    reference_scene_layout: Optional[Dict[str, Any]] = None,
    reference_scene_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Convert storyboard and asset-sheet data into a multimodal prompt.

    The ``storyboard_script`` dict is expected to contain:

    - ``storyboard_outline``: High-level scene and shot descriptions.
    - ``scene_details``: Scene setups including ``asset_ids`` and natural
      language layout descriptions.
    - ``shot_details``: Shot-level details including ``asset_modifications``
      that specify transforms needed for specific shots. When present, these
      are included in the prompt to request target_location and target_rotation
      for modified assets.
    - ``asset_sheet``: Asset metadata including dimensions and optional
      thumbnail URLs.

    Args:
        storyboard_script: The storyboard data dict.
        reference_scene_layout: Optional layout_description from a reference scene
            to provide as context for maintaining visual consistency.
        reference_scene_id: Optional ID of the reference scene.

    Returns:
        A list of dicts in any-llm format with "type" and "text"/"image_url" keys
        suitable for ``generate_layout_description``.
    """

    prompt_parts: List[Dict[str, Any]] = []
    
    # Add reference scene layout if provided
    if reference_scene_layout and reference_scene_id is not None:
        ref_text = f"# Reference Scene Layout (Scene {reference_scene_id})\n\n"
        ref_text += "**IMPORTANT**: This scene shares a similar location with the reference scene below. "
        ref_text += "You should maintain visual consistency by reusing similar scene_size and keeping "
        ref_text += "static objects (furniture, decorations, etc.) in similar positions where applicable.\n\n"
        
        # Add scene_size
        scene_size = reference_scene_layout.get("scene_size")
        if scene_size:
            ref_text += "## Scene Size\n"
            ref_text += f"- x: {scene_size.get('x', 'N/A')}\n"
            ref_text += f"- x_negative: {scene_size.get('x_negative', 'N/A')}\n"
            ref_text += f"- y: {scene_size.get('y', 'N/A')}\n"
            ref_text += f"- y_negative: {scene_size.get('y_negative', 'N/A')}\n\n"
        
        # Add assets
        assets = reference_scene_layout.get("assets", [])
        if assets:
            ref_text += "## Asset Positions in Reference Scene\n"
            for asset in assets:
                asset_id = asset.get("asset_id", "N/A")
                location = asset.get("location", {})
                rotation = asset.get("rotation", {})
                ref_text += f"- **{asset_id}**: "
                ref_text += f"location=({location.get('x', 0)}, {location.get('y', 0)}, {location.get('z', 0)}), "
                ref_text += f"rotation=({rotation.get('x', 0)}, {rotation.get('y', 0)}, {rotation.get('z', 0)})\n"
            ref_text += "\n"
        
        prompt_parts.append({"type": "text", "text": ref_text})

    # Build a lookup dictionary for shot_details by (scene_id, shot_id)
    shot_details_lookup = {}
    if "shot_details" in storyboard_script:
        for shot_detail in storyboard_script["shot_details"]:
            key = (shot_detail.get("scene_id"), shot_detail.get("shot_id"))
            shot_details_lookup[key] = shot_detail

    # 1. Storyboard outline
    if "storyboard_outline" in storyboard_script:
        md_text = "# Storyboard Outline\n\n"
        for scene in storyboard_script["storyboard_outline"]:
            scene_id = scene.get('scene_id', 'N/A')
            md_text += f"## Scene {scene_id}\n\n"
            md_text += f"{scene.get('scene_description', '')}\n\n"

            if "shots" in scene:
                md_text += "### Shots\n\n"
                for shot in scene["shots"]:
                    shot_id = shot.get('shot_id', 'N/A')
                    md_text += (
                        f"- **Shot {shot_id}**: "
                        f"{shot.get('shot_description', '')}\n"
                    )
                    
                    # Add character_actions as sub-bullets if available
                    shot_detail = shot_details_lookup.get((scene_id, shot_id))
                    if shot_detail and "character_actions" in shot_detail:
                        char_actions = shot_detail["character_actions"]
                        if char_actions:
                            for idx, action in enumerate(char_actions, start=1):
                                asset_id = action.get("asset_id", "")
                                action_desc = action.get("action_description", "")
                                md_text += f"  - Action {idx}: {asset_id} {action_desc}\n"
                
                md_text += "\n"

        prompt_parts.append({"type": "text", "text": md_text})

    # 2. Scene details (scene_id, asset_ids, layout_description)
    if "scene_details" in storyboard_script:
        md_text = "# Scene Details\n\n"
        for scene in storyboard_script["scene_details"]:
            scene_id = scene.get('scene_id', 'N/A')
            md_text += f"## Scene {scene_id}\n\n"

            scene_setup = scene.get("scene_setup", {})

            scene_type = scene_setup.get("scene_type")
            if scene_type:
                md_text += f"**Scene Type**: {scene_type}\n\n"

            asset_ids = scene_setup.get("asset_ids")
            if asset_ids:
                md_text += f"**Asset IDs**: {', '.join(asset_ids)}\n\n"

            layout_description = scene_setup.get("layout_description")
            if layout_description:
                md_text += f"**Layout Description**: {layout_description}\n\n"

        prompt_parts.append({"type": "text", "text": md_text})

    # 2.5. Asset modifications from shot_details (if any)
    if "shot_details" in storyboard_script:
        modifications_text = ""
        for shot in storyboard_script["shot_details"]:
            asset_modifications = shot.get("asset_modifications")
            if asset_modifications:
                shot_id = shot.get("shot_id", "N/A")
                scene_id = shot.get("scene_id", "N/A")
                if not modifications_text:
                    modifications_text = "# Asset Modifications by Shot\n\n"
                    modifications_text += "For the following shots, you MUST generate target transforms (target_location and target_rotation) for the specified assets if the modification type is 'add' or 'transform'. These transforms describe where and how the assets should be placed/repositioned/rotated during that specific shot. Only when the modification type is 'remove', there is no need to provide the target transforms for that modification.\n\n"
                modifications_text += f"## Scene {scene_id}, Shot {shot_id}\n\n"
                for mod in asset_modifications:
                    asset_id = mod.get("asset_id", "N/A")
                    modification_type = mod.get("modification_type", "N/A")
                    description = mod.get("description", "")
                    modifications_text += f"- **{asset_id}** ({modification_type}): {description}\n"
                modifications_text += "\n"
        
        if modifications_text:
            prompt_parts.append({"type": "text", "text": modifications_text})

    # 3. Asset sheet with dimensions before thumbnails
    if "asset_sheet" in storyboard_script:
        md_text = "# Asset Sheet\n\n"

        for asset in storyboard_script["asset_sheet"]:
            asset_id = asset.get("asset_id", "N/A")
            asset_type = asset.get("asset_type")
            description = asset.get("description", "")
            width = asset.get("width")
            depth = asset.get("depth")
            height = asset.get("height")
            thumbnail_url = asset.get("thumbnail_url", "")

            md_text += f"## {asset_id}\n\n"
            if asset_type:
                md_text += f"asset_type: {asset_type}\n"
            md_text += f"{description}\n\n"
            md_text += f"width: {_format_dimension(width)}\n"
            md_text += f"depth: {_format_dimension(depth)}\n"
            md_text += f"height: {_format_dimension(height)}\n\n"

            prompt_parts.append({"type": "text", "text": md_text})
            md_text = ""

            if thumbnail_url:
                try:
                    prompt_parts.append({
                        "type": "image_url",
                        "image_url": {"url": process_url_or_path(thumbnail_url)}
                    })
                except Exception as exc:  # pragma: no cover - best effort logging
                    print(f"Warning: Failed to load image for {asset_id}: {exc}")

    return prompt_parts

def validate_single_scene_output(
    single_scene_script: Dict[str, Any],
    layout_result: Dict[str, Any],
    scene_id: int
) -> bool:
    """Validate the layout output for a single scene.
    
    Args:
        single_scene_script: The storyboard script filtered for one scene.
        layout_result: The layout result containing a single scene.
        scene_id: The expected scene_id.
        
    Returns:
        True if validation passes, False otherwise.
    """
    try:
        if not isinstance(single_scene_script, dict) or not isinstance(layout_result, dict):
            return False

        scene_details = single_scene_script.get("scene_details", [])
        result_scene = layout_result.get("scene")

        if not isinstance(scene_details, list) or not isinstance(result_scene, dict):
            return False

        # Get expected asset IDs from scene_details
        expected_asset_ids: List[str] = []
        for scene in scene_details:
            if not isinstance(scene, dict):
                continue
            if str(scene.get("scene_id")) != str(scene_id):
                continue
            scene_setup = scene.get("scene_setup", {})
            asset_ids = scene_setup.get("asset_ids", [])
            if isinstance(asset_ids, list):
                for asset_id in asset_ids:
                    if isinstance(asset_id, str):
                        expected_asset_ids.append(asset_id)

        # Get produced asset IDs from layout_result
        produced_asset_ids: List[str] = []
        assets = result_scene.get("assets", [])
        if isinstance(assets, list):
            for obj in assets:
                if isinstance(obj, dict):
                    asset_id = obj.get("asset_id")
                    if isinstance(asset_id, str):
                        produced_asset_ids.append(asset_id)

        # Ensure asset IDs match exactly (names and counts)
        if Counter(expected_asset_ids) != Counter(produced_asset_ids):
            return False

        # Validate shot_asset_modifications: when modification_type is 'add' or 'transform',
        # the generated layout must have target_location and target_rotation for that asset
        shot_details = single_scene_script.get("shot_details", [])
        shot_asset_modifications = result_scene.get("shot_asset_modifications", []) or []
        
        # Build lookup for generated asset modifications by (shot_id, asset_id)
        generated_mods_lookup: Dict[tuple, Dict[str, Any]] = {}
        for shot_mod in shot_asset_modifications:
            if not isinstance(shot_mod, dict):
                continue
            shot_id = shot_mod.get("shot_id")
            asset_mods = shot_mod.get("asset_modifications", [])
            if isinstance(asset_mods, list):
                for asset_mod in asset_mods:
                    if isinstance(asset_mod, dict):
                        asset_id = asset_mod.get("asset_id")
                        if shot_id is not None and asset_id:
                            generated_mods_lookup[(str(shot_id), asset_id)] = asset_mod
        
        # Check each input asset modification with 'add' or 'transform' type
        for shot in shot_details:
            if not isinstance(shot, dict):
                continue
            shot_id = shot.get("shot_id")
            input_asset_mods = shot.get("asset_modifications", [])
            if not isinstance(input_asset_mods, list):
                continue
            for input_mod in input_asset_mods:
                if not isinstance(input_mod, dict):
                    continue
                modification_type = input_mod.get("modification_type")
                if modification_type in ("add", "transform"):
                    asset_id = input_mod.get("asset_id")
                    if not asset_id:
                        continue
                    # Check if generated output has target_location and target_rotation
                    generated_mod = generated_mods_lookup.get((str(shot_id), asset_id))
                    if not generated_mod:
                        return False
                    if "target_location" not in generated_mod or "target_rotation" not in generated_mod:
                        return False

        return True
    except Exception:
        return False


def validate_output(storyboard_script: Dict[str, Any], layout_result: Dict[str, Any]) -> bool:
    """Validate the merged layout output for all scenes."""
    try:
        if not isinstance(storyboard_script, dict) or not isinstance(layout_result, dict):
            return False

        scene_details = storyboard_script.get("scene_details", [])
        result_scenes = layout_result.get("scenes", [])

        if not isinstance(scene_details, list) or not isinstance(result_scenes, list):
            return False

        # Compare scene IDs (normalize to str to allow int vs str differences)
        expected_scene_ids: List[str] = []
        for scene in scene_details:
            if not isinstance(scene, dict):
                continue
            sid = scene.get("scene_id")
            if sid is None:
                continue
            expected_scene_ids.append(str(sid))

        produced_scene_ids: List[str] = []
        for scene in result_scenes:
            if not isinstance(scene, dict):
                continue
            sid = scene.get("scene_id")
            if sid is None:
                continue
            produced_scene_ids.append(str(sid))

        if Counter(expected_scene_ids) != Counter(produced_scene_ids):
            return False

        # Build expected asset IDs per scene from storyboard_script
        expected_assets_by_scene: Dict[str, List[str]] = {}
        for scene in scene_details:
            if not isinstance(scene, dict):
                continue
            sid = scene.get("scene_id")
            if sid is None:
                continue
            key = str(sid)
            scene_setup = scene.get("scene_setup", {})
            asset_ids = scene_setup.get("asset_ids", [])
            if not isinstance(asset_ids, list):
                continue
            expected_assets_by_scene.setdefault(key, [])
            for asset_id in asset_ids:
                if isinstance(asset_id, str):
                    expected_assets_by_scene[key].append(asset_id)

        # Build produced asset IDs per scene from layout_result
        produced_assets_by_scene: Dict[str, List[str]] = {}
        for scene in result_scenes:
            if not isinstance(scene, dict):
                continue
            sid = scene.get("scene_id")
            if sid is None:
                continue
            key = str(sid)
            assets = scene.get("assets", [])
            if not isinstance(assets, list):
                continue
            produced_assets_by_scene.setdefault(key, [])
            for obj in assets:
                if not isinstance(obj, dict):
                    continue
                asset_id = obj.get("asset_id")
                if isinstance(asset_id, str):
                    produced_assets_by_scene[key].append(asset_id)

        # For each scene, ensure the asset IDs match exactly (names and counts)
        for scene_id, expected_ids in expected_assets_by_scene.items():
            produced_ids = produced_assets_by_scene.get(scene_id, [])
            if Counter(expected_ids) != Counter(produced_ids):
                return False

        return True
    except Exception:
        return False

class SceneLayoutVerifier:
    """Verifies spatial relationships, contacts, directions, and occlusions
    for assets in a 3D scene layout using trimesh.

    Coordinate system:
        +Y = Behind/Back, -Y = Front
        +X = Right,       -X = Left
        +Z = Up,          -Z = Down

    Rotation convention (Z-axis Euler):
        0°   -> faces -Y (front)
        90°  -> faces +X (right)
        180° -> faces +Y (back)
        -90° -> faces -X (left)

    Local forward vector at rot=0 is (0, -1, 0).
    """

    CONTACT_TOL = 0.05
    OCCLUSION_TOL = 0.02
    DIRECTION_CONE_DEG = 45.0
    ON_TOP_Z_TOL = 0.05
    # Extra breathing room (meters) added on each XY side of a character's
    # bounding box to reserve space for animations that may extend beyond the
    # rest pose (arms swinging, walking, etc.).
    CHARACTER_XY_PADDING = 0.5

    def __init__(
        self,
        scene_json: Dict[str, Any],
        asset_dims: Dict[str, Dict[str, float]],
        scene_id: Any,
    ):
        self.scene_id = scene_id
        self.assets_data: List[Dict[str, Any]] = scene_json.get("assets", [])
        self.asset_dims = asset_dims
        self._asset_lookup: Dict[str, Dict[str, Any]] = {
            a["asset_id"]: a for a in self.assets_data
        }
        self._meshes: Dict[str, trimesh.Trimesh] = {}
        self._manager = trimesh.collision.CollisionManager()
        self._build_meshes()

    def _build_meshes(self):
        for asset in self.assets_data:
            aid = asset["asset_id"]
            dims = self.asset_dims.get(aid)
            if dims is None:
                continue
            w, d, h = dims.get("w"), dims.get("d"), dims.get("h")
            if w is None or d is None or h is None:
                continue
            if w <= 0 or d <= 0 or h <= 0:
                continue
            # Inflate XY bounding box for characters so animations
            # (arm swings, walking, etc.) don't collide with neighbours.
            if dims.get("type") == "character":
                w = w + 2 * self.CHARACTER_XY_PADDING
                d = d + 2 * self.CHARACTER_XY_PADDING
            box = trimesh.creation.box(extents=[w, d, h])
            loc = asset["location"]
            rot_z_rad = math.radians(asset["rotation"].get("z", 0))
            T = np.eye(4)
            cz, sz = math.cos(rot_z_rad), math.sin(rot_z_rad)
            T[0, 0] = cz;  T[0, 1] = -sz
            T[1, 0] = sz;  T[1, 1] = cz
            T[0, 3] = loc["x"]
            T[1, 3] = loc["y"]
            T[2, 3] = loc["z"] + h / 2.0
            box.apply_transform(T)
            self._meshes[aid] = box
            self._manager.add_object(aid, box)

    @staticmethod
    def _angle_between_vectors_2d(v1: np.ndarray, v2: np.ndarray) -> float:
        dot = np.clip(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12),
            -1, 1,
        )
        return math.degrees(math.acos(dot))

    @staticmethod
    def _bearing_deg(dx: float, dy: float) -> float:
        return math.degrees(math.atan2(dy, dx)) % 360

    def _get_pos(self, aid: str) -> np.ndarray:
        loc = self._asset_lookup[aid]["location"]
        return np.array([loc["x"], loc["y"], loc["z"]])

    def _get_forward(self, aid: str) -> np.ndarray:
        rot_z = math.radians(self._asset_lookup[aid]["rotation"].get("z", 0))
        fx = 0 * math.cos(rot_z) - (-1) * math.sin(rot_z)
        fy = 0 * math.sin(rot_z) + (-1) * math.cos(rot_z)
        return np.array([fx, fy, 0.0])

    def _get_bbox_z(self, aid: str) -> Tuple[float, float]:
        mesh = self._meshes.get(aid)
        if mesh is None:
            return (0.0, 0.0)
        return (mesh.bounds[0][2], mesh.bounds[1][2])

    def _get_bbox_xy(self, aid: str) -> Tuple[float, float, float, float]:
        mesh = self._meshes.get(aid)
        if mesh is None:
            return (0, 0, 0, 0)
        return (mesh.bounds[0][0], mesh.bounds[1][0],
                mesh.bounds[0][1], mesh.bounds[1][1])

    def _is_declared_contact_pair(self, a: str, b: str) -> bool:
        """Return True if either asset declares the other as its anchor with contact=true."""
        for x, y in ((a, b), (b, a)):
            info = self._asset_lookup.get(x)
            if info and info.get("anchor_asset_id") == y and info.get("contact") is True:
                return True
        return False

    def _check_occlusions(self) -> List[Dict[str, Any]]:
        errors = []
        aids = [a["asset_id"] for a in self.assets_data if a["asset_id"] in self._meshes]
        for i in range(len(aids)):
            for j in range(i + 1, len(aids)):
                a, b = aids[i], aids[j]
                # Declared contact pairs are allowed to overlap — especially
                # important for characters whose padded bounding box may stick
                # into an anchor they are touching on purpose.
                if self._is_declared_contact_pair(a, b):
                    continue
                mgr = trimesh.collision.CollisionManager()
                mgr.add_object("a", self._meshes[a])
                mgr.add_object("b", self._meshes[b])
                if mgr.in_collision_internal():
                    try:
                        intersection = self._meshes[a].intersection(self._meshes[b])
                        if intersection is not None and hasattr(intersection, 'volume') and intersection.volume > 0:
                            d_a = self.asset_dims.get(a, {"w": 1, "d": 1, "h": 1})
                            d_b = self.asset_dims.get(b, {"w": 1, "d": 1, "h": 1})
                            min_area = min(
                                d_a["w"] * d_a["d"], d_a["w"] * d_a["h"], d_a["d"] * d_a["h"],
                                d_b["w"] * d_b["d"], d_b["w"] * d_b["h"], d_b["d"] * d_b["h"],
                            )
                            est_depth = intersection.volume / max(min_area, 1e-6)
                            if est_depth > self.OCCLUSION_TOL:
                                # Compute separation direction
                                pos_a = self._get_pos(a)
                                pos_b = self._get_pos(b)
                                diff = pos_b[:2] - pos_a[:2]
                                dist_xy = np.linalg.norm(diff)
                                if dist_xy > 1e-6:
                                    sep_dir = diff / dist_xy
                                    move_desc_a = self._vec_to_move_instruction(-sep_dir, est_depth)
                                    move_desc_b = self._vec_to_move_instruction(sep_dir, est_depth)
                                    fix_msg = (f"Move '{a}' {move_desc_a} OR move '{b}' {move_desc_b} "
                                               f"to separate them by at least {est_depth:.3f}m.")
                                else:
                                    fix_msg = (f"'{a}' and '{b}' overlap at the same XY position. "
                                               f"Move one of them along the X or Y axis by at least "
                                               f"{est_depth:.3f}m to eliminate the intersection.")
                                errors.append({
                                    "scene_id": self.scene_id, "asset_id": f"{a} <-> {b}",
                                    "error_type": "occlusion",
                                    "detail": f"Intersection (est. depth {est_depth:.4f}m, vol {intersection.volume:.6f}m³)",
                                    "fix": fix_msg,
                                })
                    except Exception:
                        pos_a = self._get_pos(a)
                        pos_b = self._get_pos(b)
                        fix_msg = (f"'{a}' and '{b}' are physically overlapping. "
                                   f"Increase the distance between them. '{a}' is at "
                                   f"({pos_a[0]:.3f}, {pos_a[1]:.3f}), '{b}' is at "
                                   f"({pos_b[0]:.3f}, {pos_b[1]:.3f}). Move them further apart.")
                        errors.append({
                            "scene_id": self.scene_id, "asset_id": f"{a} <-> {b}",
                            "error_type": "occlusion",
                            "detail": "Collision detected (could not compute penetration depth)",
                            "fix": fix_msg,
                        })
        return errors

    @staticmethod
    def _vec_to_move_instruction(direction: np.ndarray, amount: float) -> str:
        """Convert a 2D direction vector to a human-readable move instruction."""
        dx, dy = direction[0], direction[1]
        parts = []
        if abs(dx) > 0.3:
            parts.append(f"along the {'positive' if dx > 0 else 'negative'} X axis")
        if abs(dy) > 0.3:
            parts.append(f"along the {'positive' if dy > 0 else 'negative'} Y axis")
        if not parts:
            parts.append("away from the other asset")
        return f"{' and '.join(parts)} by at least {amount:.3f}m"

    # Mapping from relationship to the axis and sign the asset should be
    # displaced along relative to anchor.
    _RELATIONSHIP_MOVE_HINTS = {
        "on_the_right_of": ("increase", "x", "+X"),
        "on_the_left_of":  ("decrease", "x", "-X"),
        "behind":          ("increase", "y", "+Y"),
        "in_front_of":     ("decrease", "y", "-Y"),
    }

    def _check_relationship(self, asset: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        aid = asset["asset_id"]
        anchor_id = asset.get("anchor_asset_id")
        relationship = asset.get("relationship")
        if anchor_id is None or relationship is None:
            return None
        if anchor_id not in self._asset_lookup:
            return {"scene_id": self.scene_id, "asset_id": aid, "error_type": "relationship",
                    "detail": f"anchor_asset_id '{anchor_id}' not found in scene",
                    "fix": f"Ensure '{anchor_id}' exists in the scene assets list, or change anchor_asset_id."}

        if relationship == "on_top_of":
            return self._check_on_top_of(aid, anchor_id)

        anchor_pos = self._get_pos(anchor_id)
        asset_pos = self._get_pos(aid)
        dx = asset_pos[0] - anchor_pos[0]
        dy = asset_pos[1] - anchor_pos[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            hint = self._RELATIONSHIP_MOVE_HINTS.get(relationship)
            if hint:
                verb, axis, label = hint
                fix_msg = (f"'{aid}' and anchor '{anchor_id}' are at the same XY position. "
                           f"To satisfy '{relationship}', {verb} the {axis} value of '{aid}' "
                           f"so it is in the {label} direction from '{anchor_id}'.")
            else:
                fix_msg = f"Move '{aid}' away from '{anchor_id}' along X or Y axis."
            return {"scene_id": self.scene_id, "asset_id": aid, "error_type": "relationship",
                    "detail": f"Asset and anchor '{anchor_id}' at same XY position",
                    "fix": fix_msg}

        bearing = self._bearing_deg(dx, dy)
        sectors = {
            "on_the_right_of": (315, 45),
            "behind":          (45, 135),
            "on_the_left_of":  (135, 225),
            "in_front_of":     (225, 315),
        }
        lo, hi = sectors.get(relationship, (0, 0))
        in_sector = (bearing >= lo or bearing < hi) if lo > hi else (lo <= bearing < hi)
        if not in_sector:
            hint = self._RELATIONSHIP_MOVE_HINTS.get(relationship)
            anchor_loc = self._asset_lookup[anchor_id]["location"]
            asset_loc = self._asset_lookup[aid]["location"]
            dist = math.sqrt(dx**2 + dy**2)
            # Compute ideal position at the exact center of the valid sector
            sector_centers = {
                "on_the_right_of": 0,    # +X direction
                "behind":          90,   # +Y direction
                "on_the_left_of":  180,  # -X direction
                "in_front_of":     270,  # -Y direction
            }
            center_deg = sector_centers.get(relationship, 0)
            center_rad = math.radians(center_deg)
            ideal_x = anchor_loc["x"] + dist * math.cos(center_rad)
            ideal_y = anchor_loc["y"] + dist * math.sin(center_rad)
            if hint:
                verb, axis, label = hint
                fix_msg = (f"'{aid}' should be in the {label} direction from '{anchor_id}' "
                           f"(anchor at x={anchor_loc['x']:.3f}, y={anchor_loc['y']:.3f}). "
                           f"Exact position for dead-center '{relationship}' at current distance "
                           f"({dist:.3f}m): x={ideal_x:.3f}, y={ideal_y:.3f}. "
                           f"Any position within the [{lo}°,{hi}°) bearing sector is acceptable. "
                           f"Current: x={asset_loc['x']:.3f}, y={asset_loc['y']:.3f}, bearing={bearing:.1f}°.")
            else:
                fix_msg = f"Adjust the position of '{aid}' relative to '{anchor_id}'."
            return {"scene_id": self.scene_id, "asset_id": aid, "error_type": "relationship",
                    "detail": f"Expected '{relationship}' relative to '{anchor_id}' "
                              f"(sector [{lo}°, {hi}°)), but bearing is {bearing:.1f}°",
                    "fix": fix_msg}
        return None

    def _check_on_top_of(self, aid: str, anchor_id: str) -> Optional[Dict[str, Any]]:
        ax_min, ax_max, ay_min, ay_max = self._get_bbox_xy(aid)
        bx_min, bx_max, by_min, by_max = self._get_bbox_xy(anchor_id)
        if not (ax_min < bx_max and ax_max > bx_min and ay_min < by_max and ay_max > by_min):
            anchor_loc = self._asset_lookup[anchor_id]["location"]
            fix_msg = (f"Set the x and y location of '{aid}' closer to the anchor '{anchor_id}' "
                       f"(anchor at x={anchor_loc['x']:.3f}, y={anchor_loc['y']:.3f}) "
                       f"so that their XY bounding boxes overlap.")
            return {"scene_id": self.scene_id, "asset_id": aid, "error_type": "relationship",
                    "detail": f"on_top_of '{anchor_id}': XY bounding boxes do not overlap",
                    "fix": fix_msg}
        asset_z_min, _ = self._get_bbox_z(aid)
        _, anchor_z_max = self._get_bbox_z(anchor_id)
        if asset_z_min < anchor_z_max - self.ON_TOP_Z_TOL:
            anchor_dims = self.asset_dims.get(anchor_id, {})
            correct_z = anchor_dims.get("h", 0)
            fix_msg = (f"Set the z location of '{aid}' to {correct_z:.3f} "
                       f"(= height of '{anchor_id}') so it sits on top of the anchor.")
            return {"scene_id": self.scene_id, "asset_id": aid, "error_type": "relationship",
                    "detail": f"on_top_of '{anchor_id}': asset bottom z ({asset_z_min:.3f}) "
                              f"< anchor top z ({anchor_z_max:.3f})",
                    "fix": fix_msg}
        return None

    def _check_contact(self, asset: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        aid = asset["asset_id"]
        anchor_id = asset.get("anchor_asset_id")
        contact = asset.get("contact")
        if anchor_id is None or contact is None:
            return None
        if aid not in self._meshes or anchor_id not in self._meshes:
            return None
        mgr = trimesh.collision.CollisionManager()
        mgr.add_object("a", self._meshes[aid])
        mgr.add_object("b", self._meshes[anchor_id])
        min_dist = mgr.min_distance_internal()
        is_colliding = mgr.in_collision_internal()
        geom_contact = (min_dist <= self.CONTACT_TOL) or is_colliding
        if contact and not geom_contact:
            relationship = asset.get("relationship")
            anchor_loc = self._asset_lookup[anchor_id]["location"]
            asset_loc = asset["location"]
            a_dims = self.asset_dims.get(aid, {})
            b_dims = self.asset_dims.get(anchor_id, {})
            if relationship == "on_the_right_of":
                correct_x = anchor_loc["x"] + b_dims.get("w", 0) / 2 + a_dims.get("w", 0) / 2
                fix_msg = (f"Set '{aid}' location.x = {correct_x:.3f} "
                           f"(= {anchor_id}.x + {anchor_id}.width/2 + {aid}.width/2) for contact.")
            elif relationship == "on_the_left_of":
                correct_x = anchor_loc["x"] - b_dims.get("w", 0) / 2 - a_dims.get("w", 0) / 2
                fix_msg = (f"Set '{aid}' location.x = {correct_x:.3f} "
                           f"(= {anchor_id}.x - {anchor_id}.width/2 - {aid}.width/2) for contact.")
            elif relationship == "behind":
                correct_y = anchor_loc["y"] + b_dims.get("d", 0) / 2 + a_dims.get("d", 0) / 2
                fix_msg = (f"Set '{aid}' location.y = {correct_y:.3f} "
                           f"(= {anchor_id}.y + {anchor_id}.depth/2 + {aid}.depth/2) for contact.")
            elif relationship == "in_front_of":
                correct_y = anchor_loc["y"] - b_dims.get("d", 0) / 2 - a_dims.get("d", 0) / 2
                fix_msg = (f"Set '{aid}' location.y = {correct_y:.3f} "
                           f"(= {anchor_id}.y - {anchor_id}.depth/2 - {aid}.depth/2) for contact.")
            elif relationship == "on_top_of":
                correct_z = b_dims.get("h", 0)
                fix_msg = (f"Set '{aid}' location.z = {correct_z:.3f} "
                           f"(= height of '{anchor_id}') for contact on top.")
            else:
                fix_msg = (f"Move '{aid}' closer to '{anchor_id}' so the gap ({min_dist:.4f}m) "
                           f"is within {self.CONTACT_TOL}m.")
            return {"scene_id": self.scene_id, "asset_id": aid, "error_type": "contact",
                    "detail": f"contact=true but min distance to '{anchor_id}' is {min_dist:.4f}m "
                              f"(tolerance {self.CONTACT_TOL}m)",
                    "fix": fix_msg}
        return None

    @staticmethod
    def _compute_required_rotation_z(asset_pos_2d: np.ndarray, anchor_pos_2d: np.ndarray,
                                      direction: str) -> int:
        """Compute the rotation_z value that satisfies the given direction.

        Returns the ideal integer rotation_z in [-180, 180].
        """
        to_anchor = anchor_pos_2d - asset_pos_2d
        # Angle of the vector from asset to anchor, measured from +X axis CCW
        angle_to_anchor = math.degrees(math.atan2(to_anchor[1], to_anchor[0]))

        if direction == "facing":
            # Forward vector (sin(rot_z), -cos(rot_z)) should point toward anchor.
            # sin(rot_z) = cos(angle_to_anchor-like), -cos(rot_z) = sin(angle_to_anchor-like)
            # rot_z = angle_to_anchor + 90
            rot_z = angle_to_anchor + 90
        elif direction == "facing_away":
            rot_z = angle_to_anchor + 90 + 180
        elif direction == "left_side_facing":
            # Left vector at rot_z is (-cos(rot_z), -sin(rot_z)) ... simplified:
            # left = rotate forward by -90. We need left to point toward anchor.
            rot_z = angle_to_anchor + 90 + 90  # = angle_to_anchor + 180
        elif direction == "right_side_facing":
            rot_z = angle_to_anchor + 90 - 90  # = angle_to_anchor
        else:
            return 0

        # Normalise to [-180, 180]
        rot_z = rot_z % 360
        if rot_z > 180:
            rot_z -= 360
        return int(round(rot_z))

    def _check_direction(self, asset: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        aid = asset["asset_id"]
        anchor_id = asset.get("anchor_asset_id")
        direction = asset.get("direction")
        if anchor_id is None or direction is None:
            return None
        if anchor_id not in self._asset_lookup:
            return None

        asset_pos = self._get_pos(aid)
        anchor_pos = self._get_pos(anchor_id)
        target_vec = anchor_pos[:2] - asset_pos[:2]
        if np.linalg.norm(target_vec) < 1e-6:
            return None

        forward = self._get_forward(aid)[:2]
        cone = self.DIRECTION_CONE_DEG
        current_rot_z = asset["rotation"].get("z", 0)
        correct_rot_z = self._compute_required_rotation_z(
            asset_pos[:2], anchor_pos[:2], direction)

        if direction == "facing":
            angle = self._angle_between_vectors_2d(forward, target_vec)
            if angle > cone:
                fix_msg = (f"Set '{aid}' rotation.z ≈ {correct_rot_z} "
                           f"(currently {current_rot_z}) so that its front faces '{anchor_id}'. "
                           f"Exact rotation_z for directly facing '{anchor_id}' = {correct_rot_z}°; "
                           f"any rotation_z within ±{int(cone)}° of that is acceptable. "
                           f"Current angular deviation: {angle:.1f}°. "
                           f"Ref: rotation_z 0→-Y, 90→+X, -90→-X, 180→+Y.")
                return {"scene_id": self.scene_id, "asset_id": aid, "error_type": "direction",
                        "detail": f"direction='facing' but angle to '{anchor_id}' is {angle:.1f}° "
                                  f"(max {cone}°, exact rotation_z={correct_rot_z}°)",
                        "fix": fix_msg}

        elif direction == "facing_away":
            angle = self._angle_between_vectors_2d(forward, target_vec)
            if angle < (180 - cone):
                fix_msg = (f"Set '{aid}' rotation.z ≈ {correct_rot_z} "
                           f"(currently {current_rot_z}) so that its back faces '{anchor_id}'. "
                           f"Exact rotation_z for directly facing away from '{anchor_id}' = {correct_rot_z}°; "
                           f"any rotation_z within ±{int(cone)}° of that is acceptable. "
                           f"Current angular deviation from ideal: {abs(180 - angle):.1f}°.")
                return {"scene_id": self.scene_id, "asset_id": aid, "error_type": "direction",
                        "detail": f"direction='facing_away' but angle to '{anchor_id}' is {angle:.1f}° "
                                  f"(need >{180 - cone}°, exact rotation_z={correct_rot_z}°)",
                        "fix": fix_msg}

        elif direction == "left_side_facing":
            left_vec = np.array([forward[1], -forward[0]])
            angle = self._angle_between_vectors_2d(left_vec, target_vec)
            if angle > cone:
                fix_msg = (f"Set '{aid}' rotation.z ≈ {correct_rot_z} "
                           f"(currently {current_rot_z}) so that its left side faces '{anchor_id}'. "
                           f"Exact rotation_z for left side directly facing '{anchor_id}' = {correct_rot_z}°; "
                           f"any rotation_z within ±{int(cone)}° of that is acceptable. "
                           f"Current angular deviation: {angle:.1f}°.")
                return {"scene_id": self.scene_id, "asset_id": aid, "error_type": "direction",
                        "detail": f"direction='left_side_facing' but angle to '{anchor_id}' is {angle:.1f}° "
                                  f"(max {cone}°, exact rotation_z={correct_rot_z}°)",
                        "fix": fix_msg}

        elif direction == "right_side_facing":
            right_vec = np.array([-forward[1], forward[0]])
            angle = self._angle_between_vectors_2d(right_vec, target_vec)
            if angle > cone:
                fix_msg = (f"Set '{aid}' rotation.z ≈ {correct_rot_z} "
                           f"(currently {current_rot_z}) so that its right side faces '{anchor_id}'. "
                           f"Exact rotation_z for right side directly facing '{anchor_id}' = {correct_rot_z}°; "
                           f"any rotation_z within ±{int(cone)}° of that is acceptable. "
                           f"Current angular deviation: {angle:.1f}°.")
                return {"scene_id": self.scene_id, "asset_id": aid, "error_type": "direction",
                        "detail": f"direction='right_side_facing' but angle to '{anchor_id}' is {angle:.1f}° "
                                  f"(max {cone}°, exact rotation_z={correct_rot_z}°)",
                        "fix": fix_msg}

        return None

    def verify_scene(self) -> List[Dict[str, Any]]:
        """Run all verifications and return a list of error dicts.

        Each error dict has keys: scene_id, asset_id, error_type, detail.
        An empty list means all checks passed.
        """
        errors: List[Dict[str, Any]] = []
        errors.extend(self._check_occlusions())
        for asset in self.assets_data:
            if asset.get("anchor_asset_id") is None:
                continue
            err = self._check_relationship(asset)
            if err:
                errors.append(err)
            err = self._check_contact(asset)
            if err:
                errors.append(err)
            err = self._check_direction(asset)
            if err:
                errors.append(err)
        print("Scene verification errors:", errors)
        return errors

    # ------------------------------------------------------------------
    # Shot-level verification
    # ------------------------------------------------------------------
    @staticmethod
    def _build_shot_scene_state(
        base_assets: List[Dict[str, Any]],
        shot_details: List[Dict[str, Any]],
        target_scene_id: Any,
        up_to_shot_id: int,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Apply cumulative modifications up to *up_to_shot_id* and return
        (updated_assets_list, list_of_modified_asset_ids_in_target_shot).

        Modifications are applied in shot order.  ``add`` inserts a new
        asset, ``remove`` deletes one, ``transform`` updates location /
        rotation / relationship fields.
        """
        from copy import deepcopy
        assets_by_id: Dict[str, Dict[str, Any]] = {
            a["asset_id"]: deepcopy(a) for a in base_assets
        }
        modified_ids: List[str] = []

        # Collect shots for this scene, sorted by shot_id
        scene_shots = sorted(
            [s for s in shot_details
             if s.get("scene_id") == target_scene_id],
            key=lambda s: s.get("shot_id", 0),
        )

        for shot in scene_shots:
            shot_id = shot.get("shot_id")
            mods = shot.get("asset_modifications")
            if not mods:
                if shot_id == up_to_shot_id:
                    break
                continue

            for mod in mods:
                aid = mod.get("asset_id")
                mod_type = mod.get("modification_type")
                if not aid or not mod_type:
                    continue

                if mod_type == "add":
                    new_asset: Dict[str, Any] = {
                        "asset_id": aid,
                        "location": mod.get("target_location", {"x": 0, "y": 0, "z": 0}),
                        "rotation": mod.get("target_rotation", {"x": 0, "y": 0, "z": 0}),
                        "anchor_asset_id": mod.get("anchor_asset_id"),
                        "relationship": mod.get("relationship"),
                        "contact": mod.get("contact"),
                        "direction": mod.get("direction"),
                    }
                    # Normalise location/rotation if they are dicts with extra keys
                    if isinstance(new_asset["location"], dict):
                        new_asset["location"] = {
                            "x": new_asset["location"].get("x", 0),
                            "y": new_asset["location"].get("y", 0),
                            "z": new_asset["location"].get("z", 0),
                        }
                    if isinstance(new_asset["rotation"], dict):
                        new_asset["rotation"] = {
                            "x": new_asset["rotation"].get("x", 0),
                            "y": new_asset["rotation"].get("y", 0),
                            "z": new_asset["rotation"].get("z", 0),
                        }
                    assets_by_id[aid] = new_asset
                    if shot_id == up_to_shot_id:
                        modified_ids.append(aid)

                elif mod_type == "remove":
                    assets_by_id.pop(aid, None)
                    # No need to verify removed assets

                elif mod_type == "transform":
                    existing = assets_by_id.get(aid)
                    if existing is None:
                        continue
                    tl = mod.get("target_location")
                    if tl:
                        existing["location"] = {
                            "x": tl.get("x", 0), "y": tl.get("y", 0), "z": tl.get("z", 0),
                        }
                    tr = mod.get("target_rotation")
                    if tr:
                        existing["rotation"] = {
                            "x": tr.get("x", 0), "y": tr.get("y", 0), "z": tr.get("z", 0),
                        }
                    # Update relationship fields if provided
                    for field in ("anchor_asset_id", "relationship", "contact", "direction"):
                        if field in mod and mod[field] is not None:
                            existing[field] = mod[field]
                    if shot_id == up_to_shot_id:
                        modified_ids.append(aid)

            if shot_id == up_to_shot_id:
                break

        return list(assets_by_id.values()), modified_ids

    def verify_scene_shots(
        self,
        shot_details: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Verify each shot that has asset_modifications for this scene.

        For each shot with modifications, cumulative state is built from the
        base scene layout, a fresh ``SceneLayoutVerifier`` is constructed,
        and only the assets modified *in that shot* are checked (relationship,
        contact, direction).  Occlusion is checked across all assets.

        Args:
            shot_details: The full ``shot_details`` list from the storyboard.

        Returns:
            List of error dicts, each with an extra ``shot_id`` key.
        """
        errors: List[Dict[str, Any]] = []

        scene_shots = sorted(
            [s for s in shot_details if s.get("scene_id") == self.scene_id],
            key=lambda s: s.get("shot_id", 0),
        )

        for shot in scene_shots:
            shot_id = shot.get("shot_id")
            mods = shot.get("asset_modifications")
            if not mods:
                continue

            # Check if there are any non-remove modifications
            has_verifiable = any(
                m.get("modification_type") in ("add", "transform")
                for m in mods if isinstance(m, dict)
            )
            if not has_verifiable:
                continue

            # Build cumulative state up to this shot
            shot_assets, modified_ids = self._build_shot_scene_state(
                self.assets_data, shot_details, self.scene_id, shot_id,
            )

            if not modified_ids:
                continue

            # Build a temporary verifier for the shot state
            shot_scene_json = {"assets": shot_assets}
            shot_verifier = SceneLayoutVerifier(
                shot_scene_json, self.asset_dims,
                scene_id=self.scene_id,
            )

            # Occlusion check (all pairs — a new asset might collide with existing)
            for occ_err in shot_verifier._check_occlusions():
                occ_err["shot_id"] = shot_id
                errors.append(occ_err)

            # Per-modified-asset checks
            modified_set = set(modified_ids)
            for asset in shot_assets:
                if asset["asset_id"] not in modified_set:
                    continue
                if asset.get("anchor_asset_id") is None:
                    continue
                for check_fn in (shot_verifier._check_relationship,
                                 shot_verifier._check_contact,
                                 shot_verifier._check_direction):
                    err = check_fn(asset)
                    if err:
                        err["shot_id"] = shot_id
                        errors.append(err)
        print("Shot verification errors:", errors)
        return errors


def verify_scene_layout(
    scene_layout: Dict[str, Any],
    asset_sheet: List[Dict[str, Any]],
    scene_id: Any,
    shot_details: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Convenience wrapper to verify a single scene's layout and its shots.

    Args:
        scene_layout: Dict with at least an ``assets`` list (as produced by
            the layout generator and stored under
            ``scene_setup.layout_description``).
        asset_sheet: The ``asset_sheet`` list from the storyboard script.
        scene_id: Identifier of the scene being verified.
        shot_details: Optional full ``shot_details`` list.  When provided the
            shots belonging to *scene_id* are verified as well.

    Returns:
        A list of error dicts.  Empty means all checks passed.
    """
    asset_dims: Dict[str, Dict[str, Any]] = {}
    for a in asset_sheet:
        asset_dims[a["asset_id"]] = {
            "w": a.get("width", 0),
            "d": a.get("depth", 0),
            "h": a.get("height", 0),
            "type": a.get("asset_type"),
        }
    verifier = SceneLayoutVerifier(scene_layout, asset_dims, scene_id=scene_id)
    errors = verifier.verify_scene()
    if shot_details:
        errors.extend(verifier.verify_scene_shots(shot_details))
    return errors


def merge_layout(
    storyboard_script: Dict[str, Any],
    layout_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge generated layout results back into the storyboard script.
    
    This function merges:
    1. Scene layout data (scene_size, assets) into scene_details[].scene_setup.layout_description
    2. Shot asset modifications (target_location, target_rotation) into shot_details[].asset_modifications
    
    Args:
        storyboard_script: The original storyboard script.
        layout_result: The generated layout result containing scenes with layouts
            and optional shot_asset_modifications.
    
    Returns:
        The merged storyboard script with layout data integrated.
    """
    if not isinstance(storyboard_script, dict) or not isinstance(layout_result, dict):
        return storyboard_script

    if not validate_output(storyboard_script, layout_result):
        return storyboard_script

    merged = deepcopy(storyboard_script)

    scene_details = merged.get("scene_details", [])
    result_scenes = layout_result.get("scenes", [])

    if not isinstance(scene_details, list) or not isinstance(result_scenes, list):
        return merged

    scene_setup_by_id: Dict[str, Dict[str, Any]] = {}
    for scene in scene_details:
        if not isinstance(scene, dict):
            continue
        sid = scene.get("scene_id")
        if sid is None:
            continue
        scene_setup = scene.get("scene_setup")
        if not isinstance(scene_setup, dict):
            continue
        scene_setup_by_id[str(sid)] = scene_setup

    # Build a lookup for shot_details by (scene_id, shot_id)
    shot_details_lookup: Dict[tuple, Dict[str, Any]] = {}
    shot_details = merged.get("shot_details", [])
    if isinstance(shot_details, list):
        for shot in shot_details:
            if isinstance(shot, dict):
                scene_id = shot.get("scene_id")
                shot_id = shot.get("shot_id")
                if scene_id is not None and shot_id is not None:
                    shot_details_lookup[(str(scene_id), str(shot_id))] = shot

    for scene_layout in result_scenes:
        if not isinstance(scene_layout, dict):
            continue
        sid = scene_layout.get("scene_id")
        if sid is None:
            continue
        scene_setup = scene_setup_by_id.get(str(sid))
        if scene_setup is None:
            continue

        original_layout = scene_setup.get("layout_description")

        # Build new layout, excluding shot_asset_modifications (handled separately)
        new_layout: Dict[str, Any] = {"description": original_layout}
        for key, value in scene_layout.items():
            if key != "shot_asset_modifications":
                new_layout[key] = value

        scene_setup["layout_description"] = new_layout

        # Merge shot_asset_modifications into shot_details
        shot_asset_modifications = scene_layout.get("shot_asset_modifications")
        if shot_asset_modifications and isinstance(shot_asset_modifications, list):
            for shot_mod in shot_asset_modifications:
                if not isinstance(shot_mod, dict):
                    continue
                shot_id = shot_mod.get("shot_id")
                if shot_id is None:
                    continue
                
                # Find the corresponding shot_detail
                shot_detail = shot_details_lookup.get((str(sid), str(shot_id)))
                if shot_detail is None:
                    continue
                
                # Get the asset_modifications list in the shot_detail
                original_asset_mods = shot_detail.get("asset_modifications")
                if not isinstance(original_asset_mods, list):
                    continue
                
                # Build a lookup for generated transforms by asset_id
                generated_transforms = shot_mod.get("asset_modifications", [])
                transform_by_asset: Dict[str, Dict[str, Any]] = {}
                if isinstance(generated_transforms, list):
                    for transform in generated_transforms:
                        if isinstance(transform, dict):
                            asset_id = transform.get("asset_id")
                            if asset_id:
                                transform_by_asset[asset_id] = transform
                
                # Merge generated fields into each asset modification
                for asset_mod in original_asset_mods:
                    if not isinstance(asset_mod, dict):
                        continue
                    asset_id = asset_mod.get("asset_id")
                    if asset_id and asset_id in transform_by_asset:
                        generated = transform_by_asset[asset_id]
                        if "target_location" in generated:
                            asset_mod["target_location"] = generated["target_location"]
                        if "target_rotation" in generated:
                            asset_mod["target_rotation"] = generated["target_rotation"]
                        if "anchor_asset_id" in generated:
                            asset_mod["anchor_asset_id"] = generated["anchor_asset_id"]
                        if "relationship" in generated:
                            asset_mod["relationship"] = generated["relationship"]
                        if "contact" in generated:
                            asset_mod["contact"] = generated["contact"]
                        if "direction" in generated:
                            asset_mod["direction"] = generated["direction"]

    return merged

def _format_verification_errors_for_llm(errors: List[Dict[str, Any]]) -> str:
    """Format verification errors into a structured correction prompt for the LLM.

    Groups errors by type and includes the predefined fix instructions.
    """
    if not errors:
        return ""

    # Separate scene-level and shot-level errors
    scene_errors = [e for e in errors if "shot_id" not in e]
    shot_errors = [e for e in errors if "shot_id" in e]

    lines = [
        "# Layout Verification Errors",
        "",
        "Your previous layout has the following geometric verification errors. "
        "Please fix ALL of them in your next response while keeping the layout "
        "natural and cinematically compelling. The fix suggestions provide exact "
        "values for reference, but you do NOT need to use them verbatim — any "
        "value within the stated tolerance range is acceptable. Prioritise a "
        "layout that serves the story and looks good on camera over mechanical "
        "precision. Only change the values that need fixing; keep everything "
        "else the same. For scenes with a reference scene, do NOT move static "
        "objects (furniture, props) unless absolutely necessary to resolve an error. "
        "**Reminder:** every asset whose `asset_type` is `character` must keep at "
        "least 0.5 m of clearance on each horizontal side (+X, -X, +Y, -Y) from "
        "any other asset's bounding box (except an anchor it is in declared "
        "`contact: true` with). Occlusion errors involving a character usually "
        "mean you need to move the character or its neighbour further apart so "
        "that this 0.5 m breathing room is preserved for animation.",
        "",
    ]

    if scene_errors:
        lines.append(f"## BASE SCENE LAYOUT errors ({len(scene_errors)})")
        lines.append("Fix these in the `scene.assets` array:")
        lines.append("")
        for e in scene_errors:
            lines.append(f"- **{e['asset_id']}** [{e['error_type']}]: {e['detail']}")
            fix = e.get("fix")
            if fix:
                lines.append(f"  **FIX**: {fix}")
        lines.append("")

    if shot_errors:
        # Group by shot_id
        by_shot: Dict[Any, List[Dict[str, Any]]] = {}
        for e in shot_errors:
            by_shot.setdefault(e["shot_id"], []).append(e)
        for shot_id in sorted(by_shot):
            errs = by_shot[shot_id]
            lines.append(f"## SHOT {shot_id} MODIFICATION errors ({len(errs)})")
            lines.append(f"Fix these in `scene.shot_asset_modifications` for shot_id={shot_id}:")
            lines.append("")
            for e in errs:
                lines.append(f"- **{e['asset_id']}** [{e['error_type']}]: {e['detail']}")
                fix = e.get("fix")
                if fix:
                    lines.append(f"  **FIX**: {fix}")
            lines.append("")

    lines.append(
        "Apply fixes for all errors above. Use the suggested exact values as "
        "guidance — you may adjust them slightly to maintain a natural, "
        "cinematically pleasing layout as long as the values stay within "
        "the stated tolerance ranges. Re-output the full corrected JSON. "
        "Do not explain—just output the JSON."
    )
    lines.append(
        "**Double-check your changes for consistency with the reference scene (if provided) and ensure that your changes do not introduce new errors.**"
    )
    return "\n".join(lines)


def generate_single_scene_layout(
    anyllm_api_key=None,
    anyllm_api_base=None,
    anyllm_provider="gemini",
    reasoning_model="gemini-3.1-pro-preview",
    contents=None,
    single_scene_script=None,
    scene_id=None,
    response_schema=SingleSceneLayoutOutput,
    reasoning_effort="high",
    max_retries=3,
    max_improvement_turns=5
):
    """Generate structured 3D layout description for a single scene with retry logic
    and an iterative verification-correction loop.

    After a valid layout is produced, it is verified geometrically (occlusions,
    relationships, contacts, directions) for both the base scene and its shots.
    If errors are found, the errors and predefined fix instructions are fed back
    to the LLM as a follow-up message, and the LLM is asked to output a
    corrected layout.  This repeats for up to *max_improvement_turns*.

    Args:
        anyllm_api_key: Optional API key used to authenticate the client.
        anyllm_api_base: Optional base URL for the API service.
        anyllm_provider: LLM provider (default: "gemini").
        reasoning_model: The model name to invoke.
        contents: The input message contents for this scene.
        single_scene_script: The storyboard script filtered for this scene.
        scene_id: The scene_id being processed.
        response_schema: A Pydantic model describing the expected JSON shape.
        reasoning_effort: Controls thinking capability. Defaults to "high".
        max_retries: Maximum number of retry attempts on validation failure.
        max_improvement_turns: Maximum rounds of verify→correct conversation.
            Defaults to 5.  Set to 0 to skip verification entirely.

    Returns:
        dict: A JSON-compatible object with the scene layout, or None on failure.
    """
    
    system_instruction = """
You are a specialist AI 3D Scene Layout Planner for Blender. Your primary task is to interpret a multimodal storyboard script and an asset sheet to generate precise 3D layout data. You will read a story's plot, scene descriptions, and object dimensions, then output the exact `Transform` parameters (`Location` and `Rotation`) and spatial relationships for every object in every scene.

Your output must be a single, valid JSON object that strictly adheres to the provided schema. Do not provide any text, apologies, or explanations outside of the final JSON.

**INPUTS YOU WILL RECEIVE:**

1.  **Storyboard Script:** Contains the story outline, plot descriptions, and a list of scenes. For each "Scene", this describe the plot of the story in each scene and its corresponding "Shots". For each "Shot", there are "Action" which describe the actions of the characters and objects in the shot, use the above information to determain the layout of the scene. Note: reserve distance if the characters have moving actions such as running or walking in the later shots of a scene.

2.  **Scene Details:** Have a list of "Asset IDs" that appear in the scene. And "Layout Description", a natural language description of the desired spatial layout and the relationships between assets.

3.  **Asset Sheet:** Contains visual descriptions, thumbnails, and the crucial dimensions for each `asset_id` in meters (width, depth, height, corresponding to Blender's X, Y, Z axes).

**CORE TASK & LOGIC:**

For each `scene_id` in the script, you must iterate through every `asset_id` in its `scene_setup` and determine its final `location`, `rotation`, and `relationship` data.

You must meticulously follow these rules:

---

**1. GENERAL LAYOUT PRINCIPLES**

* **Scene Focus:** Identify the most important object or character for the scene's plot and place it at or near the world origin (0, 0, 0).
* **Scene Size:** Define the size of the scene based on the plot and available assets. The size of a scene should be large enough to accommodate all assets and provide enough space for movement. For indoor scenes, the size should be not larger than 10 meters (20*20) in any direction, usually -10 to 10 in X and Y directions is large enough. Walls will be placed at the border of an interior scene at the Y(front), -Y(back), X(right), and -X(left) borders, symmetrical to each other with respect to the origin (0, 0).
* **Object Origin:** All assets have their origin point at their **bottom center**.
* **Ground Plane:** An object with `location` (0, 0, 0) will be at the center of the scene, sitting on the ground (the X-Y plane).
* **Z-Location:** The `z` location for all assets should be 0, *unless* an object is described as floating or is explicitly placed on top of another object.
* **Humanoid Characters:** Humanoid characters must remain upright, regardless of their actions in the scene. Set rotation_x and rotation_y to 0; however, you may adjust rotation_z to change the character's facing direction. Animation is handled in a later process. For example, even if a character is meant to be lying down in later shots, rotation_x and rotation_y must remain at 0.
* **Object Orientation:** All objects should be oriented in the direction of their intended use.

**2. LOCATION COORDINATE SYSTEM & CALCULATION**

You MUST use the dimensions (width=X, depth=Y, height=Z) from the "Asset Sheet" to calculate precise locations.

* **Viewpoint:** Assume a default view from -Y to +Y (front view).
* `X-Axis`: Positive = Right, Negative = Left.
* `Y-Axis`: Positive = Back (away from view), Negative = Front (towards view).
* `Z-Axis`: Positive = Up, Negative = Down.
* All assets should be placed within the scene size limit.

**CRITICAL CALCULATION RULES:**

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

**3. RELATIONSHIP FIELDS**

You must set these fields for each asset. They are set on the "anchored" asset (the current object).

* Anchor/Child Logic: Use the plot and common sense. A book on a table -> book is the child, table is the anchor. A character standing next to a horse -> the horse might be the "anchor" object for the relationship, or vice versa, depending on which one is more static or central.
* `anchor_asset_id`: The `asset_id` of the anchor object in the relationship, for example, if a person is playing piano, the anchor for the piano is the person; if two persons are talking to each other, the anchors for the two persons can be each other. If no anchor, set to `null`.
* `relationship`: Set for the 'anchored' object. Valid values are `on_top_of`, `on_the_left_of`, `on_the_right_of`, `in_front_of`, `behind`.
    * This is based on the object's **scene-relative position** viewed from the default front view, not its rotation.
    * If the object is a anchor, or has no defined relationship, set to `null`.
* `contact`: Set to `true` if assets have direct geometry contact (use the calculations above). Set to `false` if they are in proximity but not touching. If no relationship, set to `null`.
* `direction`: Describes how this asset is oriented relative to its `anchor_asset_id`. This is about the **object's own facing direction**, independent of the global coordinate system. Valid values:
    * `facing`: The front of this asset faces toward the anchor. Example: Character A looking at Character B → A's direction is `facing`.
    * `facing_away`: The back of this asset faces the anchor. Example: Character B has turned away from A during a conversation → B's direction is `facing_away`.
    * `left_side_facing`: The left side of this asset faces the anchor. Example: A pianist sitting at a piano with their left profile toward another character.
    * `right_side_facing`: The right side of this asset faces the anchor. Example: A car parked with its passenger side toward a building.
    * `null`: Not applicable, or the asset has no distinct directional orientation.
    * **When to use:** Recommended for assets with visually distinct orientations—characters, vehicles (cars, horses), musical instruments with a clear front (piano, guitar), furniture with a defined front (TV, desk). Optional for symmetric objects (tables, spheres, trees).
    * **Important:** The `direction` field describes the object's orientation toward its anchor, NOT its position in the scene. It should be consistent with the `rotation` values you set.
* The relationship is based on the scene view, NOT the asset's internal rotation (As if we see the scene as a 2D image as a director, from -Y to +Y. e.g., if B is to the left of A in the scene, the relationship is `on_the_left_of`, even if A is facing away from B). The most common relationships are `on_the_left_of` and `on_the_right_of` when two assets share a similar location in y axis but not x axis, for example, character A and character B are talking to each other face to face, with their side facing the camera (A_location_y ≈ B_location_y, A_location_x < B_location_x, A is `on_the_left_of` B). For `in_front_of` and `behind`, they usually happens when an asset blocks the view of another asset, for example, character A standing in front of house B, the character A will block part of the view of the house B, and they share a similar location in x axis but not y axis (A_location_x ≈ B_location_x, A_location_y < B_location_y, A is `in_front_of` B).

**4. ROTATION COORDINATE SYSTEM (XYZ EULER)**

* **Default State (0, 0, 0):** The object's "front" faces the viewport (If we view the model as a 3D vector, the front of the model is pointing at the negative Y direction, the viewport is looking from the positive Y direction).
* `X-Axis (Tilting)`: Positive = Tilts forward (top moves toward -Y), Negative = Tilts backward.
* `Y-Axis (Turning)`: Positive = Turns left (front moves toward +X), Negative = Turns right.
* `Z-Axis (Spinning)`: Positive = Spins counter-clockwise, Negative = Spins clockwise.
* **Primary Control:** You will mostly use the **Z-Axis** to orient assets.
    * `Z: 0` = Faces front (default, negative Y direction).
    * `Z: 90` = Faces right (positive X direction).
    * `Z: -90` = Faces left (negative X direction).
    * `Z: 180` = Faces back (positive Y direction).
* **Default:** Keep `X` and `Y` at 0 unless the `layout_description` or plot specifically describes tilting or turning (e.g., "knocked over").
* **Values:** Output rotation values as numbers in degrees, integer, from -180 to 180.

**4a. DIRECTION ↔ ROTATION_Z VERIFICATION (CRITICAL)**

When you set `direction` for an asset, you MUST ensure its `rotation.z` is consistent. The forward vector of an asset at rotation_z = R is: `forward = (sin(R), -cos(R))`. Use this to verify:

* **`facing` anchor:** The forward vector must point toward the anchor (angle < 45°).
    Formula: `rotation_z = atan2(anchor.x - asset.x, -(anchor.y - asset.y))` (in degrees).
    Quick reference for common cases:
    - Asset is to the LEFT of anchor (anchor at +X): rotation_z ≈ 90
    - Asset is to the RIGHT of anchor (anchor at -X): rotation_z ≈ -90
    - Asset is IN FRONT of anchor (anchor at +Y): rotation_z ≈ 0
    - Asset is BEHIND anchor (anchor at -Y): rotation_z ≈ 180
* **`facing_away` anchor:** The forward vector must point away from anchor (angle > 135°). rotation_z = facing_rotation_z + 180.
* **`left_side_facing` anchor:** The asset's left side faces the anchor. rotation_z = facing_rotation_z + 90.
* **`right_side_facing` anchor:** The asset's right side faces the anchor. rotation_z = facing_rotation_z - 90.

**COMMON MISTAKES TO AVOID:**
* Confusing `rotation_z = 90` (faces +X/right) with `rotation_z = -90` (faces -X/left).
* Setting `direction: "facing"` but using a rotation that points the front AWAY from the anchor.
* Two characters "facing each other": they need DIFFERENT rotation_z values, not the same. If A is at y=-1 and B is at y=1, A should face +Y (rotation_z=0) and B should face -Y (rotation_z=180).
* Assets that are `on_the_right_of` an anchor must have asset.x > anchor.x. Assets `behind` must have asset.y > anchor.y. The relationship is based on scene coordinates, NOT asset rotation.
* Ensure no two assets physically overlap (bounding boxes must not intersect) unless one is `on_top_of` the other.

**4a-bis. CHARACTER BREATHING ROOM (CRITICAL)**

Characters (assets whose `asset_type` is `character`) will be animated later (walking, swinging arms, gesturing, turning, etc.), so their animated silhouette extends well beyond their rest-pose bounding box. When placing any asset whose `asset_type` is `character`:

* Leave **at least 0.5 m of empty clearance** on every horizontal side (+X, -X, +Y, -Y) of the character's bounding box, measured from the edge of the character's bounding box to the nearest edge of any other asset's bounding box.
* Equivalently, when separating a character from any neighbour asset along an axis, add an extra **0.5 m** on top of the usual `(character_dimension/2 + neighbour_dimension/2)` contact distance — and another 0.5 m if the neighbour is also a character (so two characters need ~1.0 m of clearance between their bounding boxes).
* This applies even when `contact` is `true` (e.g. a character "next to" a chair): use the geometric contact formula for the non-character side but keep the 0.5 m breathing room on the character's other sides relative to unrelated assets.
* The breathing room does **not** apply to the vertical axis (Z) — characters can still stand on the ground (`z=0`) normally.
* Exception: the character's declared anchor in a direct-contact relationship (e.g. a character leaning on a door with `contact: true`) may touch the character's bounding box. All other unrelated assets must respect the 0.5 m clearance.

**4b. TOLERANCE & CREATIVE FREEDOM**

Your layout will be verified geometrically, but the verification uses generous tolerances. Treat the `direction` and `relationship` fields as **general cinematic guidance**, not rigid constraints:

* **Direction (rotation_z):** A ±45° cone from the ideal facing angle is acceptable. You do NOT need pixel-perfect alignment — choose a rotation that looks natural for the story context. For example, two characters in conversation may angle slightly toward the camera rather than staring directly at each other.
* **Relationship (position):** The valid bearing sector spans 90° (e.g., "behind" = bearing 45°–135° from the anchor). Position the asset naturally within this sector — you do not need to hit the dead center.
* **Contact:** Surfaces must be within 0.05 m of each other to count as "touching".
* **Occlusion:** Bounding-box penetration depth must be < 0.02 m unless the objects are meant to overlap (e.g., `on_top_of`).

When planning a layout, consider:
* The **story plot** — how characters interact emotionally and physically in the current scene.
* **Cinematic staging rules** — sight lines, the 180° rule, composition, depth, and visual balance.
* **Natural-looking arrangements** — real people don't stand at mathematically precise angles; slight asymmetry often looks more realistic.

**4c. REFERENCE SCENE CONSISTENCY**

When the input provides a reference scene layout (from a previous scene), you MUST keep the new layout consistent with it:

* **Do NOT move static objects** (furniture, props, set dressing) unless the new scene's plot explicitly requires it.
* **Only reposition characters** whose actions in the new scene demand a different location or orientation.
* **Maintain the same `scene_size`** as the reference scene.
* Use the reference layout as a starting point and only apply the changes needed for the new scene's narrative.

**5. SHOT ASSET MODIFICATIONS**

If the input contains an "Asset Modifications by Shot" section, you MUST also generate `shot_asset_modifications` in your output. This field contains target transforms for assets that need to change position/rotation during specific shots.

* For each shot that has asset modifications, create an entry with `shot_id` and a list of `asset_modifications`.
* Each asset modification contains:
    * `asset_id`: The ID of the asset to modify (must match an asset in the scene).
    * `target_location`: The new location {x, y, z} for the asset during this shot.
    * `target_rotation`: The new rotation {x, y, z} for the asset during this shot.
    * `anchor_asset_id`: (Optional) The asset_id of the anchor object for the spatial relationship in this shot. This may differ from the initial layout if the asset's relationship changes during the shot.
    * `relationship`: (Optional) The spatial relationship to the anchor asset during this shot. Same values as the main layout: `on_top_of`, `on_the_left_of`, `on_the_right_of`, `in_front_of`, `behind`, or `null`.
    * `contact`: (Optional) Whether this asset is in direct geometry contact with the anchor in this shot. Set to `true`, `false`, or `null`.
    * `direction`: (Optional) The orientation of this asset relative to its anchor_asset_id in this shot. Same values as the main layout: `facing`, `facing_away`, `left_side_facing`, `right_side_facing`, or `null`. Use this when the asset's facing direction changes during the shot (e.g., a character turns away from another).
* Use the same coordinate system and calculation rules as for the main asset layout.
* The target transforms describe where the asset should end up during that shot (e.g., if a character falls to the ground, the target_location.z might be 0 with target_rotation indicating lying down).
* If no asset modifications are specified in the input, set `shot_asset_modifications` to `null`.

---

**REQUIRED OUTPUT JSON SCHEMA:**

Your entire response must be ONLY the provided JSON schema, populated with your calculations. You are processing a single scene, so output the layout for that one scene only. Use the exact same `asset_id` as the input. Do NOT include `scene_id` in your output.

**EXAMPLE OUTPUT (single scene with shot asset modifications):**

This example demonstrates a prince at the center, slightly turned, and his horse in contact with him on his right side, angled away. The prince is facing toward the horse, while the horse has turned away. Shot 2 has an asset modification where the prince falls to the ground.

```json
{
    "scene": {
        "scene_size": {
            "x": 10,
            "x_negative": -10,
            "y": 10,
            "y_negative": -10
        },
        "assets": [
            {
                "asset_id": "prince",
                "location": {
                    "x": 0,
                    "y": 0,
                    "z": 0
                },
                "rotation": {
                    "x": 0,
                    "y": 0,
                    "z": 30
                },
                "relationship": null,
                "anchor_asset_id": "princes_horse",
                "contact": null,
                "direction": "facing"
            },
            {
                "asset_id": "princes_horse",
                "location": {
                    "x": 2.5,
                    "y": 0.75,
                    "z": 0
                },
                "rotation": {
                    "x": 0,
                    "y": 0,
                    "z": -60
                },
                "relationship": "on_the_right_of",
                "anchor_asset_id": "prince",
                "contact": true,
                "direction": "facing_away"
            }
        ],
        "shot_asset_modifications": [
            {
                "shot_id": 2,
                "asset_modifications": [
                    {
                        "asset_id": "prince",
                        "target_location": {
                            "x": 0.5,
                            "y": 0.2,
                            "z": 0.0
                        },
                        "target_rotation": {
                            "x": -90,
                            "y": 0,
                            "z": 30
                        },
                        "anchor_asset_id": "princes_horse",
                        "relationship": "in_front_of",
                        "contact": false,
                        "direction": "facing_away"
                    }
                ]
            }
        ]
    }
}
```

Begin processing the inputs. Your output must be the complete JSON only.
"""
    
    # Build messages for any-llm (mutable — the reflection loop appends to this)
    messages = [
        {
            "role": "system",
            "content": system_instruction
        },
        {
            "role": "user",
            "content": contents
        }
    ]
    

    # Build asset_dims once for verification
    asset_sheet = single_scene_script.get("asset_sheet", [])
    asset_dims: Dict[str, Dict[str, Any]] = {}
    for a in asset_sheet:
        asset_dims[a["asset_id"]] = {
            "w": a.get("width", 0),
            "d": a.get("depth", 0),
            "h": a.get("height", 0),
            "type": a.get("asset_type"),
        }
    shot_details = single_scene_script.get("shot_details", [])

    # Helper: call LLM and return parsed result (or None on failure)
    def _call_llm(attempt_label: str) -> Optional[Dict[str, Any]]:
        for attempt in range(max_retries):
            try:
                response = completion(
                    api_key=anyllm_api_key,
                    api_base=anyllm_api_base,
                    provider=anyllm_provider,
                    model=reasoning_model,
                    reasoning_effort=reasoning_effort,
                    messages=messages,
                    response_format=response_schema,
                )
                gc.collect()
                result = json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"Error generating content for scene {scene_id} "
                      f"({attempt_label}, attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    backoff_time = 2 * (2 ** attempt)
                    print(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    continue
                return None

            # Validate schema
            try:
                response_schema.model_validate(result)
            except Exception as e:
                print(f"Schema validation error for scene {scene_id} "
                      f"({attempt_label}, attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
                return None

            # Validate output asset IDs
            if not validate_single_scene_output(single_scene_script, result, scene_id):
                print(f"Output mismatch for scene {scene_id} "
                      f"({attempt_label}, attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    continue
                return None

            return result
        return None

    # Statistics collection for this scene
    scene_stats: Dict[str, Any] = {
        "scene_id": scene_id,
        "turns": [],
        "num_turns": 0,
        "converged": False,
    }

    def _collect_turn_stats(turn_num: int, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a stats dict for one verification turn."""
        scene_errs = [e for e in errors if "shot_id" not in e]
        shot_errs = [e for e in errors if "shot_id" in e]
        by_type: Dict[str, int] = {}
        for e in errors:
            by_type[e["error_type"]] = by_type.get(e["error_type"], 0) + 1
        return {
            "turn": turn_num,
            "total_errors": len(errors),
            "scene_errors": len(scene_errs),
            "shot_errors": len(shot_errs),
            "errors_by_type": by_type,
            "error_details": [
                {k: v for k, v in e.items() if k != "fix"}
                for e in errors
            ],
        }

    def _verify_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run full verification (scene + shots) and return error list."""
        scene_data = result.get("scene", {})
        scene_layout = {"assets": scene_data.get("assets", [])}
        verifier = SceneLayoutVerifier(scene_layout, asset_dims, scene_id=scene_id)
        errors = verifier.verify_scene()

        shot_mods = scene_data.get("shot_asset_modifications")
        if shot_mods:
            merged_shot_details = deepcopy(shot_details)
            shot_lookup: Dict[tuple, Dict[str, Any]] = {}
            for sd in merged_shot_details:
                if isinstance(sd, dict):
                    sid = sd.get("scene_id")
                    shid = sd.get("shot_id")
                    if sid is not None and shid is not None:
                        shot_lookup[(str(sid), str(shid))] = sd

            for sm in shot_mods:
                if not isinstance(sm, dict):
                    continue
                shot_id_val = sm.get("shot_id")
                if shot_id_val is None:
                    continue
                sd = shot_lookup.get((str(scene_id), str(shot_id_val)))
                if sd is None:
                    continue
                original_mods = sd.get("asset_modifications")
                if not isinstance(original_mods, list):
                    continue
                gen_transforms = sm.get("asset_modifications", [])
                transform_by_asset: Dict[str, Dict[str, Any]] = {}
                if isinstance(gen_transforms, list):
                    for t in gen_transforms:
                        if isinstance(t, dict):
                            aid = t.get("asset_id")
                            if aid:
                                transform_by_asset[aid] = t
                for am in original_mods:
                    if not isinstance(am, dict):
                        continue
                    aid = am.get("asset_id")
                    if aid and aid in transform_by_asset:
                        gen = transform_by_asset[aid]
                        for field in ("target_location", "target_rotation",
                                      "anchor_asset_id", "relationship",
                                      "contact", "direction"):
                            if field in gen:
                                am[field] = gen[field]

            shot_errors = verifier.verify_scene_shots(merged_shot_details)
            errors.extend(shot_errors)
        return errors

    # ------------------------------------------------------------------
    # Phase 1: Initial generation
    # ------------------------------------------------------------------
    result = _call_llm("initial")
    if result is None:
        return None, scene_stats
    print(f"Successfully generated initial layout for scene {scene_id}")

    # ------------------------------------------------------------------
    # Phase 2: Iterative verification → correction loop
    # ------------------------------------------------------------------
    if max_improvement_turns <= 0:
        return result, scene_stats

    # Append assistant response to conversation history for multi-turn
    messages.append({
        "role": "assistant",
        "content": json.dumps(result, ensure_ascii=False)
    })

    best_result = result
    best_error_count = float("inf")

    for turn in range(1, max_improvement_turns + 1):
        errors = _verify_result(result)
        error_count = len(errors)

        # Record stats for this turn
        turn_stats = _collect_turn_stats(turn, errors)
        scene_stats["turns"].append(turn_stats)
        scene_stats["num_turns"] = turn

        print(f"  Scene {scene_id} improvement turn {turn}: {error_count} error(s)")

        # Track best result
        if error_count < best_error_count:
            best_error_count = error_count
            best_result = result

        # No errors — we're done
        if error_count == 0:
            scene_stats["converged"] = True
            print(f"  Scene {scene_id}: all verification checks passed after {turn} turn(s)")
            return result, scene_stats

        # Build correction prompt and append as a new user message
        correction_text = _format_verification_errors_for_llm(errors)
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": correction_text}]
        })

        # Ask LLM to correct
        corrected = _call_llm(f"improvement turn {turn}")
        if corrected is None:
            print(f"  Scene {scene_id}: LLM failed to produce correction at turn {turn}, "
                  f"returning best result ({best_error_count} errors)")
            return best_result, scene_stats

        # Append the corrected assistant response for next turn
        messages.append({
            "role": "assistant",
            "content": json.dumps(corrected, ensure_ascii=False)
        })
        result = corrected

    # Final verification of the last corrected result
    final_errors = _verify_result(result)
    final_turn_stats = _collect_turn_stats(max_improvement_turns + 1, final_errors)
    scene_stats["turns"].append(final_turn_stats)
    if len(final_errors) < best_error_count:
        best_error_count = len(final_errors)
        best_result = result
    if len(final_errors) == 0:
        scene_stats["converged"] = True

    # Exhausted improvement turns — return best
    print(f"  Scene {scene_id}: reached max improvement turns ({max_improvement_turns}), "
          f"returning best result ({best_error_count} errors)")
    return best_result, scene_stats


def generate_layout_description(
    anyllm_api_key=None,
    anyllm_api_base=None,
    anyllm_provider="gemini",
    reasoning_model="gemini-3.1-pro-preview",
    storyboard_script=None,
    reasoning_effort="high",
    max_retries_per_scene=3,
    max_improvement_turns=5
):
    """Generate structured 3D layout descriptions for all scenes, processing one scene at a time.

    This function processes each scene individually, validates each result, retries on
    failure, runs a verification-correction loop, and merges all successful scene
    layouts into a single output.

    Args:
        anyllm_api_key: Optional API key used to authenticate the client.
        anyllm_api_base: Optional base URL for the API service.
        anyllm_provider: LLM provider (default: "gemini").
        reasoning_model: The model name to invoke, e.g. "gemini-3.1-pro-preview".
        storyboard_script: The full storyboard script containing all scenes.
        reasoning_effort: Controls thinking capability. Options are "low",
            "medium", or "high". Defaults to "high".
        max_retries_per_scene: Maximum retry attempts per scene. Defaults to 3.
        max_improvement_turns: Maximum verification-correction rounds per scene.
            Defaults to 5.  Set to 0 to skip verification.

    Returns:
        dict: A JSON-compatible object with all scene layouts merged,
        or None if any scene fails after all retries.
    """
    if not isinstance(storyboard_script, dict):
        print("Error: storyboard_script must be a dict")
        return None
    
    # Extract all scene IDs from scene_details
    scene_details = storyboard_script.get("scene_details", [])
    if not isinstance(scene_details, list) or not scene_details:
        print("Error: No scene_details found in storyboard_script")
        return None
    
    scene_ids = []
    for scene in scene_details:
        if isinstance(scene, dict):
            sid = scene.get("scene_id")
            if sid is not None:
                scene_ids.append(sid)
    
    if not scene_ids:
        print("Error: No valid scene_ids found in scene_details")
        return None
    
    print(f"Processing {len(scene_ids)} scenes: {scene_ids}")
    
    # Run-level statistics
    run_stats: Dict[str, Any] = {
        "scenes": [],
        "summary": {},
    }

    # Process each scene one at a time
    all_scene_layouts = []
    
    for scene_id in scene_ids:
        print(f"\n--- Processing scene {scene_id} ---")
        
        # Extract single scene data
        single_scene_script = extract_single_scene_data(storyboard_script, scene_id)
        
        # Check if this scene has a reference_scene_id
        reference_scene_layout = None
        reference_scene_id = None
        for scene in storyboard_script.get("scene_details", []):
            if scene.get("scene_id") == scene_id:
                scene_setup = scene.get("scene_setup", {})
                ref_id = scene_setup.get("reference_scene_id")
                if ref_id is not None:
                    reference_scene_id = ref_id
                    reference_scene_layout = get_reference_scene_layout(storyboard_script, ref_id)
                    if reference_scene_layout:
                        print(f"  Using reference scene {ref_id} for visual consistency")
                break
        
        # Build prompt for this scene (with reference scene context if available)
        scene_prompt = create_layout_description_prompt(
            single_scene_script,
            reference_scene_layout=reference_scene_layout,
            reference_scene_id=reference_scene_id
        )
        
        # Generate layout for this scene with retry and verification loop
        scene_result, scene_stats = generate_single_scene_layout(
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            anyllm_provider=anyllm_provider,
            reasoning_model=reasoning_model,
            contents=scene_prompt,
            single_scene_script=single_scene_script,
            scene_id=scene_id,
            response_schema=SingleSceneLayoutOutput,
            reasoning_effort=reasoning_effort,
            max_retries=max_retries_per_scene,
            max_improvement_turns=max_improvement_turns
        )
        run_stats["scenes"].append(scene_stats)
        
        if scene_result is None:
            print(f"Failed to generate layout for scene {scene_id} after {max_retries_per_scene} attempts")
            return None, run_stats
        
        # Extract the scene layout, add scene_id, and add to list
        scene_layout = scene_result.get("scene")
        if scene_layout:
            # Build reflection_log from scene_stats
            reflection_log = {
                "num_turns": scene_stats.get("num_turns", 0),
                "converged": scene_stats.get("converged", False),
                "turns": scene_stats.get("turns", []),
            }
            # Add scene_id back since we didn't require it in the output
            scene_layout_with_id = {
                "scene_id": str(scene_id),
                **scene_layout,
                "reflection_log": reflection_log,
            }
            all_scene_layouts.append(scene_layout_with_id)
    
    # Merge all scene layouts into the final output format
    merged_result = {"scenes": all_scene_layouts}
    
    # Final validation of merged result
    if not validate_output(storyboard_script, merged_result):
        print("Error: Final merged output validation failed")
        return None, run_stats
    
    # ---- Compute run-level summary statistics ----
    total_scenes = len(run_stats["scenes"])
    initial_errors_all = []
    final_errors_all = []
    initial_by_type: Dict[str, int] = {}
    final_by_type: Dict[str, int] = {}
    converged_count = 0
    turns_to_converge = []

    for ss in run_stats["scenes"]:
        turns = ss.get("turns", [])
        if turns:
            init = turns[0]
            final = turns[-1]
            initial_errors_all.append(init["total_errors"])
            final_errors_all.append(final["total_errors"])
            for etype, cnt in init["errors_by_type"].items():
                initial_by_type[etype] = initial_by_type.get(etype, 0) + cnt
            for etype, cnt in final["errors_by_type"].items():
                final_by_type[etype] = final_by_type.get(etype, 0) + cnt
        else:
            initial_errors_all.append(0)
            final_errors_all.append(0)

        if ss.get("converged"):
            converged_count += 1
            turns_to_converge.append(ss["num_turns"])

    run_stats["summary"] = {
        "total_scenes": total_scenes,
        "total_initial_errors": sum(initial_errors_all),
        "total_final_errors": sum(final_errors_all),
        "avg_initial_errors_per_scene": sum(initial_errors_all) / max(total_scenes, 1),
        "avg_final_errors_per_scene": sum(final_errors_all) / max(total_scenes, 1),
        "convergence_rate": converged_count / max(total_scenes, 1),
        "converged_scenes": converged_count,
        "avg_turns_to_converge": (
            sum(turns_to_converge) / len(turns_to_converge)
            if turns_to_converge else None
        ),
        "max_improvement_turns": max_improvement_turns,
        "initial_errors_by_type": initial_by_type,
        "final_errors_by_type": final_by_type,
    }

    print(f"\nSuccessfully generated layouts for all {len(scene_ids)} scenes")
    return merged_result, run_stats
