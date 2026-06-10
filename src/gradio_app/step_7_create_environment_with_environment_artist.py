import os
import json
import logging
import gradio as gr
from .blender_client import BlenderClient
from .json_editor import JSONEditorComponent
from .path_utils import make_paths_absolute, make_paths_relative

logger = logging.getLogger(__name__)


def load_layout_script(project_dir):
    """Load the latest layout_script JSON from project_dir/layout_script/."""
    if not project_dir or not os.path.isabs(project_dir):
        return None, None
    
    layout_script_dir = os.path.join(project_dir, "layout_script")
    if not os.path.exists(layout_script_dir):
        return None, None
    
    # Find the latest version
    latest_version = 0
    latest_path = None
    
    for filename in os.listdir(layout_script_dir):
        if filename.startswith("layout_script_v") and filename.endswith(".json"):
            try:
                version_str = filename[len("layout_script_v"):-5]
                version = int(version_str)
                if version > latest_version:
                    latest_version = version
                    latest_path = os.path.join(layout_script_dir, filename)
            except ValueError:
                continue
    
    if latest_path:
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            try:
                data = make_paths_absolute(data, project_dir)
            except Exception as e:
                logger.warning("Step 7: path conversion failed for layout script: %s", e)
            return data, latest_path
        except Exception as e:
            print(f"Error loading layout script: {e}")
            return None, None
    return None, None


def get_next_version_path(save_dir, basename):
    """Get the next version file path for saving."""
    os.makedirs(save_dir, exist_ok=True)
    
    version = 1
    while True:
        filename = f"{basename}_v{version}.json"
        filepath = os.path.join(save_dir, filename)
        if not os.path.exists(filepath):
            return filepath, version
        version += 1


def initialize_environment_artist(project_dir):
    """Initialize the environment artist by loading layout script and creating scene buttons."""
    if not project_dir or not os.path.isabs(project_dir):
        return (
            gr.update(value="⚠️ Please set a valid absolute project directory path.", visible=True),
            gr.update(visible=False),  # scene_buttons_row
            None,  # layout_data state
            None,  # current_filepath state
            {},    # environment_applied state
        )
    
    layout_data, filepath = load_layout_script(project_dir)
    
    if layout_data is None:
        return (
            gr.update(value="⚠️ Could not load layout script. Please ensure project_dir/layout_script/layout_script_v{N}.json exists.", visible=True),
            gr.update(visible=False),
            None,
            None,
            {},
        )
    
    # Get scene count
    scene_details = layout_data.get("scene_details", [])
    num_scenes = len(scene_details)
    
    if num_scenes == 0:
        return (
            gr.update(value="⚠️ No scenes found in layout script.", visible=True),
            gr.update(visible=False),
            None,
            None,
            {},
        )
    
    return (
        gr.update(value=f"✅ Loaded layout script: `{filepath}` with {num_scenes} scene(s). Click a scene button to configure environment.", visible=True),
        gr.update(visible=True),
        layout_data,
        filepath,
        {},  # Reset environment_applied state
    )


def get_scene_setup_by_id(layout_data, scene_id):
    """Get scene_setup from layout_data by scene_id."""
    if not layout_data:
        return None
    
    scene_details = layout_data.get("scene_details", [])
    for scene_detail in scene_details:
        if scene_detail.get("scene_id") == scene_id:
            return scene_detail.get("scene_setup", {})
    return None


def select_scene(blender_client, layout_data, scene_id):
    """Handle scene button click - switch to scene and populate fields."""
    # Default empty return for wall fields
    empty_wall_return = ("", "", "", "", "", "", "", gr.update(visible=False))
    
    if not layout_data:
        return (
            gr.update(value="⚠️ Layout data not loaded. Please initialize first.", visible=True),
            gr.update(visible=False),  # environment_config_row
            "",  # ground_description
            "",  # asset_id
            "",  # categories_limitation
            "",  # width
            "",  # depth
            "",  # multiply_factor
            None,  # current_scene_id
        ) + empty_wall_return
    
    # Ensure MCP server is running
    success, message = blender_client.ensure_server_running()
    if not success:
        return (
            gr.update(value=f"⚠️ {message}", visible=True),
            gr.update(visible=False),
            "",
            "",
            "",
            "",
            "",
            "",
            None,
        ) + empty_wall_return
    
    # Switch to the scene
    scene_name = f"Scene_{scene_id}"
    response = blender_client.switch_or_create_scene(scene_name=scene_name)
    
    if response.get("status") == "error":
        return (
            gr.update(value=f"⚠️ Failed to switch to scene: {response.get('message', 'Unknown error')}", visible=True),
            gr.update(visible=False),
            "",
            "",
            "",
            "",
            "",
            "",
            None,
        ) + empty_wall_return
    
    # Get scene setup
    scene_setup = get_scene_setup_by_id(layout_data, scene_id)
    if not scene_setup:
        return (
            gr.update(value=f"⚠️ Could not find scene_setup for scene {scene_id}.", visible=True),
            gr.update(visible=False),
            "",
            "",
            "",
            "",
            "",
            "",
            None,
        ) + empty_wall_return
    
    # Extract values
    scene_type = scene_setup.get("scene_type", "outdoor")
    ground_description = scene_setup.get("ground_description", "")
    ground_asset_id = scene_setup.get("ground_asset_id", "")
    
    # Determine categories limitation based on scene_type
    if scene_type == "indoor":
        categories_limitation = '["floor"]'
    else:  # outdoor
        categories_limitation = '["terrain"]'
    
    # Get scene_size and calculate width/depth
    layout_description = scene_setup.get("layout_description", {})
    scene_size = layout_description.get("scene_size", {})
    
    x = scene_size.get("x", 10)
    x_negative = scene_size.get("x_negative", -10)
    y = scene_size.get("y", 10)
    y_negative = scene_size.get("y_negative", -10)
    z = scene_size.get("z", 4)
    
    width = x - x_negative
    depth = y - y_negative
    
    # Default multiply factor
    if scene_type == "indoor":
        multiply_factor = "2"
    else:  # outdoor
        multiply_factor = "3"
    
    # Extract wall info for indoor scenes
    wall_description = scene_setup.get("wall_description", "")
    wall_asset_id = scene_setup.get("wall_asset_id", "")
    
    # Show wall config row only for indoor scenes
    is_indoor = scene_type == "indoor"
    wall_row_visible = gr.update(visible=is_indoor)
    
    # Wall dimension strings (only for indoor)
    if is_indoor:
        wall_x_str = str(x)
        wall_x_neg_str = str(x_negative)
        wall_y_str = str(y)
        wall_y_neg_str = str(y_negative)
        wall_z_str = str(z)
        wall_categories = '["wall"]'
    else:
        wall_x_str = ""
        wall_x_neg_str = ""
        wall_y_str = ""
        wall_y_neg_str = ""
        wall_z_str = ""
        wall_categories = ""
    
    return (
        gr.update(value=f"✅ Switched to **Scene {scene_id}** ({scene_type}). Configure environment below and click 'Create Environment'.", visible=True),
        gr.update(visible=True),
        ground_description,
        ground_asset_id if ground_asset_id else "",
        categories_limitation,
        str(width),
        str(depth),
        multiply_factor,
        scene_id,
        # Wall fields
        wall_description,
        wall_asset_id if wall_asset_id else "",
        wall_categories,
        wall_x_str,
        wall_x_neg_str,
        wall_y_str,
        wall_y_neg_str,
        wall_z_str,
        wall_row_visible,
    )


def apply_environment(blender_client, layout_data, current_scene_id, ground_description, asset_id, categories_limitation, 
                       width_str, depth_str, multiply_factor_str, environment_applied,
                       wall_description=None, wall_asset_id=None, wall_categories_limitation=None,
                       wall_x_str=None, wall_x_neg_str=None, wall_y_str=None, wall_y_neg_str=None, wall_z_str=None,
                       anyllm_api_key=None, anyllm_api_base=None, anyllm_provider="gemini", vision_model="gemini-3-flash-preview"):
    """Apply environment to the current scene using environment_artist."""
    if not current_scene_id:
        return (
            gr.update(value="⚠️ No scene selected. Please click a scene button first.", visible=True),
            environment_applied,
            "",  # applied_asset_id
        )
    
    # Ensure MCP server is running
    success, message = blender_client.ensure_server_running()
    if not success:
        return (
            gr.update(value=f"⚠️ {message}", visible=True),
            environment_applied,
            "",
        )
    
    # Parse categories_limitation from string to list
    try:
        if categories_limitation and categories_limitation.strip():
            categories_list = json.loads(categories_limitation)
            if not isinstance(categories_list, list):
                categories_list = [categories_list]
        else:
            categories_list = None
    except json.JSONDecodeError:
        # Try to handle as simple string
        categories_list = [categories_limitation.strip()] if categories_limitation.strip() else None
    
    # Parse width and depth
    try:
        width = float(width_str) if width_str else 10.0
    except ValueError:
        width = 10.0
    
    try:
        depth = float(depth_str) if depth_str else 10.0
    except ValueError:
        depth = 10.0
    
    # Parse multiply factor and apply
    try:
        multiply_factor = float(multiply_factor_str) if multiply_factor_str else 3.0
    except ValueError:
        multiply_factor = 3.0
    
    if multiply_factor and multiply_factor > 0:
        width = width * multiply_factor
        depth = depth * multiply_factor
    
    # Prepare parameters for environment_artist
    # If asset_id is provided and not empty, use it directly
    asset_id_param = asset_id.strip() if asset_id and asset_id.strip() else None
    ground_desc_param = ground_description.strip() if ground_description and ground_description.strip() else None
    
    # Parse wall parameters
    wall_desc_param = wall_description.strip() if wall_description and wall_description.strip() else None
    wall_asset_id_param = wall_asset_id.strip() if wall_asset_id and wall_asset_id.strip() else None
    
    # Parse wall categories
    try:
        if wall_categories_limitation and wall_categories_limitation.strip():
            wall_categories_list = json.loads(wall_categories_limitation)
            if not isinstance(wall_categories_list, list):
                wall_categories_list = [wall_categories_list]
        else:
            wall_categories_list = None
    except json.JSONDecodeError:
        wall_categories_list = [wall_categories_limitation.strip()] if wall_categories_limitation and wall_categories_limitation.strip() else None
    
    # Parse wall dimensions
    def parse_float_or_none(s):
        if s and s.strip():
            try:
                return float(s.strip())
            except ValueError:
                return None
        return None
    
    wall_x = parse_float_or_none(wall_x_str)
    wall_x_negative = parse_float_or_none(wall_x_neg_str)
    wall_y = parse_float_or_none(wall_y_str)
    wall_y_negative = parse_float_or_none(wall_y_neg_str)
    wall_z = parse_float_or_none(wall_z_str)
    
    # Call environment_artist with all parameters
    response = blender_client.environment_artist(
        ground_description=ground_desc_param,
        asset_id=asset_id_param,
        categories_limitation=categories_list,
        width=width,
        depth=depth,
        wall_description=wall_desc_param,
        wall_asset_id=wall_asset_id_param,
        wall_categories_limitation=wall_categories_list,
        wall_x=wall_x,
        wall_x_negative=wall_x_negative,
        wall_y=wall_y,
        wall_y_negative=wall_y_negative,
        wall_z=wall_z,
        anyllm_api_key=anyllm_api_key,
        anyllm_api_base=anyllm_api_base,
        anyllm_provider=anyllm_provider,
        vision_model=vision_model,
    )
    
    if response.get("status") == "error":
        return (
            gr.update(value=f"⚠️ Environment artist error: {response.get('message', 'Unknown error')}", visible=True),
            environment_applied,
            "",
        )
    
    result = response.get("result", {})
    
    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        return (
            gr.update(value=f"⚠️ Environment artist failed: {error_msg}", visible=True),
            environment_applied,
            "",
        )
    
    # Get the applied asset_id from the result
    applied_asset_id = result.get("ground_asset_id", "")
    texture_name = result.get("ground_texture_name", applied_asset_id)
    plane_name = result.get("plane_name", "ground_plane")
    
    # Get wall info if created
    wall_asset_id_result = result.get("wall_asset_id", "")
    walls_created = result.get("walls_created", [])
    
    # Update environment_applied state (store both ground and wall asset IDs)
    new_environment_applied = environment_applied.copy() if environment_applied else {}
    new_environment_applied[current_scene_id] = {
        "ground_asset_id": applied_asset_id,
        "wall_asset_id": wall_asset_id_result if wall_asset_id_result else None
    }
    
    # Build status message
    status_msg = f"✅ Environment created for Scene {current_scene_id}!\n\n"
    status_msg += f"**Ground Texture:** {texture_name}\n"
    status_msg += f"**Ground Asset ID:** `{applied_asset_id}`\n"
    status_msg += f"**Plane:** {plane_name}\n"
    status_msg += f"**Size:** {width:.1f}m x {depth:.1f}m"
    
    if wall_asset_id_result:
        status_msg += f"\n\n**Wall Texture:** {wall_asset_id_result}\n"
        if walls_created:
            status_msg += f"**Walls Created:** {', '.join(walls_created)}"
    
    return (
        gr.update(value=status_msg, visible=True),
        new_environment_applied,
        applied_asset_id,
    )


def finish_environment_creation(project_dir, layout_data, current_filepath, environment_applied, layout_script_editor):
    """Finish environment creation - save updated layout_script with ground_asset_id values.
    
    Returns:
        dict: Result with success/error/warning info
    """
    if not layout_data:
        return {
            "error": "⚠️ Layout data not loaded. Please initialize first."
        }
    
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Please set a valid absolute project directory path."
        }
    
    # Check which scenes don't have environment applied
    scene_details = layout_data.get("scene_details", [])
    scenes_without_environment = []
    
    for scene_detail in scene_details:
        scene_id = scene_detail.get("scene_id")
        if scene_id not in (environment_applied or {}):
            # Check if ground_asset_id already exists in the JSON
            scene_setup = scene_detail.get("scene_setup", {})
            if not scene_setup.get("ground_asset_id"):
                scenes_without_environment.append(scene_id)
    
    # Update layout_data with ground_asset_id and wall_asset_id values
    for scene_detail in scene_details:
        scene_id = scene_detail.get("scene_id")
        if scene_id in (environment_applied or {}):
            if "scene_setup" not in scene_detail:
                scene_detail["scene_setup"] = {}
            env_data = environment_applied[scene_id]
            # Handle both old format (string) and new format (dict)
            if isinstance(env_data, dict):
                scene_detail["scene_setup"]["ground_asset_id"] = env_data.get("ground_asset_id", "")
                if env_data.get("wall_asset_id"):
                    scene_detail["scene_setup"]["wall_asset_id"] = env_data["wall_asset_id"]
            else:
                # Backward compatibility: if it's a string, treat it as ground_asset_id
                scene_detail["scene_setup"]["ground_asset_id"] = env_data
    
    # Save using the layout_script_editor
    layout_script_dir = os.path.join(project_dir, "layout_script")
    layout_script_editor.set_save_path(layout_script_dir)
    output_path = layout_script_editor.save_json_data(layout_data)
    
    if not output_path:
        return {
            "error": "⚠️ Failed to save layout script."
        }
    
    # Build result
    if scenes_without_environment:
        scenes_str = ", ".join([str(s) for s in scenes_without_environment])
        warning = f"⚠️ **Warning:** Scenes {scenes_str} do not have environment applied."
    else:
        warning = None
    
    return {
        "success": True,
        "data": layout_data,
        "output_path": output_path,
        "warning": warning,
    }


def apply_environment_all_scenes_ai(blender_client, layout_data, environment_applied, layout_script_editor, project_dir, anyllm_api_key=None, anyllm_api_base=None, anyllm_provider="gemini", vision_model="gemini-3-flash-preview", progress=gr.Progress(track_tqdm=False)):
    """Apply AI-selected environment to ALL scenes in sequence, then auto-finish.
    
    Yields intermediate status updates as each scene is processed.
    Final yield includes all outputs needed to save and reinitialize.
    """
    noop_editor = layout_script_editor.update_with_result(None)
    
    if not layout_data:
        yield (
            gr.update(value="\u26a0\ufe0f Layout data not loaded. Please initialize first.", visible=True),
            environment_applied,
            gr.update(),  # scene_buttons_row
            gr.update(),  # environment_config_row
            gr.update(),  # apply_all_ai_btn
        ) + noop_editor
        return
    
    # Ensure MCP server is running
    success, message = blender_client.ensure_server_running()
    if not success:
        yield (
            gr.update(value=f"\u26a0\ufe0f {message}", visible=True),
            environment_applied,
            gr.update(),
            gr.update(),
            gr.update(),
        ) + noop_editor
        return
    
    scene_details = layout_data.get("scene_details", [])
    num_scenes = len(scene_details)
    new_environment_applied = environment_applied.copy() if environment_applied else {}
    
    for idx, scene_detail in enumerate(scene_details):
        scene_id = scene_detail.get("scene_id")
        scene_setup = scene_detail.get("scene_setup", {})
        scene_type = scene_setup.get("scene_type", "outdoor")
        ground_description = scene_setup.get("ground_description", "")
        is_indoor = scene_type == "indoor"
        categories_list = ["floor"] if is_indoor else ["terrain"]
        
        # Compute width/depth from scene_size
        layout_description = scene_setup.get("layout_description", {})
        scene_size = layout_description.get("scene_size", {})
        x = scene_size.get("x", 10)
        x_negative = scene_size.get("x_negative", -10)
        y = scene_size.get("y", 10)
        y_negative = scene_size.get("y_negative", -10)
        z = scene_size.get("z", 4)
        
        width = (x - x_negative) * (2.0 if is_indoor else 3.0)
        depth = (y - y_negative) * (2.0 if is_indoor else 3.0)
        
        # Wall parameters for indoor scenes
        wall_description = scene_setup.get("wall_description", "") if is_indoor else None
        wall_categories_list = ["wall"] if is_indoor else None
        wall_x = float(x) if is_indoor else None
        wall_x_negative = float(x_negative) if is_indoor else None
        wall_y = float(y) if is_indoor else None
        wall_y_negative = float(y_negative) if is_indoor else None
        wall_z = float(z) if is_indoor else None
        
        progress((idx, num_scenes), desc=f"Scene {scene_id}")
        
        # Yield progress status
        yield (
            gr.update(value=f"\u23f3 Creating environment for **Scene {scene_id}** ({idx + 1}/{num_scenes})...", visible=True),
            new_environment_applied,
            gr.update(),
            gr.update(),
            gr.update(),
        ) + noop_editor
        
        # Switch to the scene
        scene_name = f"Scene_{scene_id}"
        response = blender_client.switch_or_create_scene(scene_name=scene_name)
        if response.get("status") == "error":
            yield (
                gr.update(value=f"\u26a0\ufe0f Failed to switch to Scene {scene_id}: {response.get('message', 'Unknown error')}. Skipping.", visible=True),
                new_environment_applied,
                gr.update(),
                gr.update(),
                gr.update(),
            ) + noop_editor
            continue
        
        # Apply environment with AI (no manual asset_id)
        response = blender_client.environment_artist(
            ground_description=ground_description.strip() if ground_description else None,
            asset_id=None,
            categories_limitation=categories_list,
            width=width,
            depth=depth,
            wall_description=wall_description.strip() if wall_description else None,
            wall_asset_id=None,
            wall_categories_limitation=wall_categories_list,
            wall_x=wall_x,
            wall_x_negative=wall_x_negative,
            wall_y=wall_y,
            wall_y_negative=wall_y_negative,
            wall_z=wall_z,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            anyllm_provider=anyllm_provider,
            vision_model=vision_model,
        )
        
        if response.get("status") == "error":
            yield (
                gr.update(value=f"\u26a0\ufe0f Environment artist error for Scene {scene_id}: {response.get('message', 'Unknown error')}. Skipping.", visible=True),
                new_environment_applied,
                gr.update(),
                gr.update(),
                gr.update(),
            ) + noop_editor
            continue
        
        result = response.get("result", {})
        if not result.get("success"):
            yield (
                gr.update(value=f"\u26a0\ufe0f Environment failed for Scene {scene_id}: {result.get('error', 'Unknown')}. Skipping.", visible=True),
                new_environment_applied,
                gr.update(),
                gr.update(),
                gr.update(),
            ) + noop_editor
            continue
        
        applied_asset_id = result.get("ground_asset_id", "")
        wall_asset_id_result = result.get("wall_asset_id", "")
        new_environment_applied[scene_id] = {
            "ground_asset_id": applied_asset_id,
            "wall_asset_id": wall_asset_id_result if wall_asset_id_result else None,
        }
    
    progress((num_scenes, num_scenes), desc="Saving")
    
    # Auto-finish: save JSON and reinitialize
    finish_result = finish_environment_creation(project_dir, layout_data, None, new_environment_applied, layout_script_editor)
    
    if finish_result.get("error"):
        yield (
            gr.update(value=finish_result["error"], visible=True),
            new_environment_applied,
            gr.update(),
            gr.update(),
            gr.update(),
        ) + noop_editor
        return
    
    # Build summary of applied environments
    summary_lines = []
    for scene_detail in scene_details:
        sid = scene_detail.get("scene_id")
        env = new_environment_applied.get(sid)
        if env and isinstance(env, dict):
            ground = env.get("ground_asset_id", "\u274c not applied")
            wall = env.get("wall_asset_id")
            line = f"- **Scene {sid}:** ground=`{ground}`"
            if wall:
                line += f", wall=`{wall}`"
            summary_lines.append(line)
        else:
            summary_lines.append(f"- **Scene {sid}:** \u274c not applied")
    summary = "\n".join(summary_lines)
    
    warning = finish_result.get("warning", "")
    if warning:
        status_msg = f"{warning}\n\n{summary}\n\n\u2705 Layout script saved to: `{finish_result['output_path']}`"
    else:
        status_msg = f"\u2705 **Environment created for all {num_scenes} scene(s) with AI!**\n\n{summary}\n\nLayout script saved to: `{finish_result['output_path']}`"
    
    yield (
        gr.update(value=status_msg, visible=True),
        new_environment_applied,
        gr.update(visible=False),  # scene_buttons_row - hide
        gr.update(visible=False),  # environment_config_row - hide
        gr.update(visible=False),  # apply_all_ai_btn - hide
    ) + layout_script_editor.update_with_result(finish_result)


def resume_environment_all_scenes(blender_client, layout_data, environment_applied, layout_script_editor, project_dir, progress=gr.Progress(track_tqdm=False)):
    """Resume environment creation for ALL scenes using `ground_asset_id` (and
    `wall_asset_id` for indoor scenes) already stored in the layout script.
    Bypasses AI selection — same path as the manual Asset ID fields, applied to
    every scene with one click.

    Yields the same output tuple shape as `apply_environment_all_scenes_ai`.
    """
    noop_editor = layout_script_editor.update_with_result(None)

    if not layout_data:
        yield (
            gr.update(value="\u26a0\ufe0f Layout data not loaded. Please initialize first.", visible=True),
            environment_applied,
            gr.update(), gr.update(), gr.update(),
        ) + noop_editor
        return

    success, message = blender_client.ensure_server_running()
    if not success:
        yield (
            gr.update(value=f"\u26a0\ufe0f {message}", visible=True),
            environment_applied,
            gr.update(), gr.update(), gr.update(),
        ) + noop_editor
        return

    scene_details = layout_data.get("scene_details", [])
    num_scenes = len(scene_details)
    new_environment_applied = environment_applied.copy() if environment_applied else {}
    skipped = []

    for idx, scene_detail in enumerate(scene_details):
        scene_id = scene_detail.get("scene_id")
        scene_setup = scene_detail.get("scene_setup", {})
        scene_type = scene_setup.get("scene_type", "outdoor")
        is_indoor = scene_type == "indoor"
        ground_asset_id = (scene_setup.get("ground_asset_id") or "").strip()
        wall_asset_id = (scene_setup.get("wall_asset_id") or "").strip() if is_indoor else ""
        categories_list = ["floor"] if is_indoor else ["terrain"]
        wall_categories_list = ["wall"] if is_indoor else None

        # Compute width/depth from scene_size
        layout_description = scene_setup.get("layout_description", {})
        scene_size = layout_description.get("scene_size", {})
        x = scene_size.get("x", 10)
        x_negative = scene_size.get("x_negative", -10)
        y = scene_size.get("y", 10)
        y_negative = scene_size.get("y_negative", -10)
        z = scene_size.get("z", 4)

        width = (x - x_negative) * (2.0 if is_indoor else 3.0)
        depth = (y - y_negative) * (2.0 if is_indoor else 3.0)

        wall_x = float(x) if is_indoor else None
        wall_x_negative = float(x_negative) if is_indoor else None
        wall_y = float(y) if is_indoor else None
        wall_y_negative = float(y_negative) if is_indoor else None
        wall_z = float(z) if is_indoor else None

        progress((idx, num_scenes), desc=f"Scene {scene_id}")

        if not ground_asset_id:
            skipped.append(scene_id)
            yield (
                gr.update(value=f"\u26a0\ufe0f Scene {scene_id} has no `ground_asset_id` in layout JSON. Skipping.", visible=True),
                new_environment_applied,
                gr.update(), gr.update(), gr.update(),
            ) + noop_editor
            continue

        yield (
            gr.update(value=f"\u23f3 Resuming environment for **Scene {scene_id}** ({idx + 1}/{num_scenes}) using ground=`{ground_asset_id}`" + (f", wall=`{wall_asset_id}`" if wall_asset_id else "") + "...", visible=True),
            new_environment_applied,
            gr.update(), gr.update(), gr.update(),
        ) + noop_editor

        scene_name = f"Scene_{scene_id}"
        response = blender_client.switch_or_create_scene(scene_name=scene_name)
        if response.get("status") == "error":
            yield (
                gr.update(value=f"\u26a0\ufe0f Failed to switch to Scene {scene_id}: {response.get('message', 'Unknown error')}. Skipping.", visible=True),
                new_environment_applied,
                gr.update(), gr.update(), gr.update(),
            ) + noop_editor
            continue

        # Apply environment with explicit asset IDs (bypasses AI selection)
        response = blender_client.environment_artist(
            ground_description=None,
            asset_id=ground_asset_id,
            categories_limitation=categories_list,
            width=width,
            depth=depth,
            wall_description=None,
            wall_asset_id=wall_asset_id if wall_asset_id else None,
            wall_categories_limitation=wall_categories_list,
            wall_x=wall_x,
            wall_x_negative=wall_x_negative,
            wall_y=wall_y,
            wall_y_negative=wall_y_negative,
            wall_z=wall_z,
        )

        if response.get("status") == "error":
            yield (
                gr.update(value=f"\u26a0\ufe0f Environment artist error for Scene {scene_id}: {response.get('message', 'Unknown error')}. Skipping.", visible=True),
                new_environment_applied,
                gr.update(), gr.update(), gr.update(),
            ) + noop_editor
            continue

        result = response.get("result", {})
        if not result.get("success"):
            yield (
                gr.update(value=f"\u26a0\ufe0f Environment failed for Scene {scene_id}: {result.get('error', 'Unknown')}. Skipping.", visible=True),
                new_environment_applied,
                gr.update(), gr.update(), gr.update(),
            ) + noop_editor
            continue

        applied_ground = result.get("ground_asset_id", ground_asset_id)
        applied_wall = result.get("wall_asset_id", wall_asset_id) or None
        new_environment_applied[scene_id] = {
            "ground_asset_id": applied_ground,
            "wall_asset_id": applied_wall,
        }

    progress((num_scenes, num_scenes), desc="Saving")

    finish_result = finish_environment_creation(project_dir, layout_data, None, new_environment_applied, layout_script_editor)

    if finish_result.get("error"):
        yield (
            gr.update(value=finish_result["error"], visible=True),
            new_environment_applied,
            gr.update(), gr.update(), gr.update(),
        ) + noop_editor
        return

    summary_lines = []
    for scene_detail in scene_details:
        sid = scene_detail.get("scene_id")
        env = new_environment_applied.get(sid)
        if env and isinstance(env, dict):
            ground = env.get("ground_asset_id", "\u274c not applied")
            wall = env.get("wall_asset_id")
            line = f"- **Scene {sid}:** ground=`{ground}`"
            if wall:
                line += f", wall=`{wall}`"
            summary_lines.append(line)
        else:
            summary_lines.append(f"- **Scene {sid}:** \u274c not applied")
    summary = "\n".join(summary_lines)

    header = f"\u2705 **Resumed environment for {num_scenes - len(skipped)}/{num_scenes} scene(s) from layout JSON.**"
    if skipped:
        header += f"\n\n\u26a0\ufe0f Skipped scenes (no `ground_asset_id`): {', '.join(str(s) for s in skipped)}"
    warning = finish_result.get("warning", "")
    extra = f"\n\n{warning}" if warning else ""
    status_msg = f"{header}{extra}\n\n{summary}\n\nLayout script saved to: `{finish_result['output_path']}`"

    yield (
        gr.update(value=status_msg, visible=True),
        new_environment_applied,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    ) + layout_script_editor.update_with_result(finish_result)


def create_scene_button_click_handler(blender_client, scene_id):
    """Create a click handler for a specific scene button."""
    def handler(layout_data):
        return select_scene(blender_client, layout_data, scene_id)
    return handler


def create_environment_artist_ui(project_dir, blender_client, anyllm_api_key=None, anyllm_api_base=None, anyllm_provider=None, vision_model=None):
    """Create the Step 7: Create Environment with Environment Artist UI section.
    
    Args:
        project_dir: Gradio component for project directory
        blender_client: BlenderClient instance for communicating with Blender
        anyllm_api_key: Gradio component for AnyLLM API key
        anyllm_api_base: Gradio component for AnyLLM API base URL
        anyllm_provider: Gradio component for AnyLLM provider
        vision_model: Gradio component for vision model
    
    Returns:
        dict with UI components that may be needed by other parts of the app
    """
    gr.Markdown("## Step 7: Create Environment with Environment Artist")
    gr.Markdown("Create ground planes using AI-powered texture selection or manual asset ID specification.")
    
    # Input/Output info
    gr.Markdown("**Input:** `project_dir/layout_script/layout_script_v{N}.json` (latest)")
    gr.Markdown("**Output:** `project_dir/layout_script/layout_script_v{N+1}.json` (with ground_asset_id)")
    
    # State variables
    layout_data_state = gr.State(value=None)
    current_filepath_state = gr.State(value=None)
    current_scene_id_state = gr.State(value=None)
    environment_applied_state = gr.State(value={})
    
    # Initialize button
    initialize_btn = gr.Button("🌍 Initialize Environment Artist", variant="primary", size="lg")
    
    # JSON Editor for viewing/editing layout_script
    layout_script_editor = JSONEditorComponent(
        label="Layout Script JSON",
        visible_initially=False,
        file_basename="layout_script",
        json_root_keys_list=["scene_details"],
        title="Step 7"
    )
    
    # Wire up the Resume button with project_dir input
    layout_script_editor.setup_resume_with_project_dir(project_dir, subfolder="layout_script")
    
    # Status indicator
    environment_status = gr.Markdown(value="", visible=False)
    
    # Batch buttons (hidden initially)
    with gr.Row(visible=False) as batch_buttons_row:
        apply_all_ai_btn = gr.Button(
            "🤖 Create Environment for All Scenes with AI",
            variant="primary", size="lg",
        )
        resume_all_btn = gr.Button(
            "♻️ Resume Environment for All Scenes",
            variant="secondary", size="lg",
        )
    
    gr.Markdown("*Or select a scene below to manually configure environment:*", visible=True)
    
    # Scene buttons row (hidden initially)
    with gr.Row(visible=False) as scene_buttons_row:
        # Create buttons for up to 10 scenes (can be extended if needed)
        scene_buttons = []
        for i in range(1, 11):
            btn = gr.Button(f"Scene {i}", variant="secondary", visible=(i <= 3))  # Show first 3 by default
            scene_buttons.append(btn)
    
    # Environment configuration row (hidden initially)
    with gr.Column(visible=False) as environment_config_row:
        gr.Markdown("### Ground Configuration")
        with gr.Row():
            ground_description_input = gr.Textbox(
                label="Ground Description",
                placeholder="e.g., Polished stone floor, Dirt path with scattered leaves.",
                lines=2
            )
            asset_id_input = gr.Textbox(
                label="Asset ID (Optional - bypasses AI selection)",
                placeholder="e.g., cobblestone_floor (leave empty for AI selection)",
                lines=1
            )
        with gr.Row():
            categories_limitation_input = gr.Textbox(
                label="Categories Limitation",
                placeholder='e.g., ["floor"] or ["terrain"]',
                lines=1
            )
            width_input = gr.Textbox(
                label="Width (meters)",
                placeholder="e.g., 20",
                lines=1
            )
            depth_input = gr.Textbox(
                label="Depth (meters)",
                placeholder="e.g., 20",
                lines=1
            )
            multiply_factor_input = gr.Textbox(
                label="Multiply Factor",
                placeholder="e.g., 2 (multiplies width and depth)",
                value="2.5",
                lines=1
            )
        
        # Wall configuration row (hidden by default, shown for indoor scenes)
        with gr.Column(visible=False) as wall_config_row:
            gr.Markdown("### Wall Configuration (Indoor Only)")
            with gr.Row():
                wall_description_input = gr.Textbox(
                    label="Wall Description",
                    placeholder="e.g., White painted plaster walls, Exposed brick walls",
                    lines=2
                )
                wall_asset_id_input = gr.Textbox(
                    label="Wall Asset ID (Optional)",
                    placeholder="e.g., white_plaster (leave empty for AI selection)",
                    lines=1
                )
            with gr.Row():
                wall_categories_input = gr.Textbox(
                    label="Wall Categories",
                    placeholder='e.g., ["plaster", "concrete", "brick"]',
                    lines=1
                )
                wall_x_input = gr.Textbox(
                    label="Wall X",
                    placeholder="e.g., 5",
                    lines=1
                )
                wall_x_neg_input = gr.Textbox(
                    label="Wall X Negative",
                    placeholder="e.g., -5",
                    lines=1
                )
            with gr.Row():
                wall_y_input = gr.Textbox(
                    label="Wall Y",
                    placeholder="e.g., 5",
                    lines=1
                )
                wall_y_neg_input = gr.Textbox(
                    label="Wall Y Negative",
                    placeholder="e.g., -5",
                    lines=1
                )
                wall_z_input = gr.Textbox(
                    label="Wall Height (Z)",
                    placeholder="e.g., 4",
                    lines=1
                )
        
        applied_asset_id_display = gr.Textbox(
            label="Applied Asset ID (from last apply)",
            interactive=False,
            visible=True
        )
        
        with gr.Row():
            apply_environment_btn = gr.Button("🏞️ Create Environment with Environment Artist", variant="primary", size="lg")
            finish_btn = gr.Button("✅ Finish Environment Creation", variant="secondary", size="lg")
    
    # Initialize button handler
    def init_handler(proj_dir):
        result = initialize_environment_artist(proj_dir)
        status_update, scene_row_update, layout_data, filepath, environment_applied = result
        
        # Update scene button visibility based on number of scenes
        button_updates = []
        show_ai_btn = False
        if layout_data:
            num_scenes = len(layout_data.get("scene_details", []))
            show_ai_btn = num_scenes > 0
            for i in range(10):
                button_updates.append(gr.update(visible=(i < num_scenes)))
        else:
            button_updates = [gr.update(visible=False) for _ in range(10)]
        
        return (status_update, scene_row_update, layout_data, filepath, environment_applied, gr.update(visible=show_ai_btn)) + tuple(button_updates)
    
    initialize_btn.click(
        fn=init_handler,
        inputs=[project_dir],
        outputs=[
            environment_status,
            scene_buttons_row,
            layout_data_state,
            current_filepath_state,
            environment_applied_state,
            batch_buttons_row,
        ] + scene_buttons,
        concurrency_limit=None,
        show_progress="hidden",
    )
    
    # Scene button handlers
    for i, btn in enumerate(scene_buttons):
        scene_id = i + 1
        handler = create_scene_button_click_handler(blender_client, scene_id)
        btn.click(
            fn=handler,
            inputs=[layout_data_state],
            outputs=[
                environment_status,
                environment_config_row,
                ground_description_input,
                asset_id_input,
                categories_limitation_input,
                width_input,
                depth_input,
                multiply_factor_input,
                current_scene_id_state,
                # Wall fields
                wall_description_input,
                wall_asset_id_input,
                wall_categories_input,
                wall_x_input,
                wall_x_neg_input,
                wall_y_input,
                wall_y_neg_input,
                wall_z_input,
                wall_config_row,
            ],
            concurrency_limit=None,
            show_progress="hidden",
        )
    
    # Apply environment button handler
    def apply_handler(layout_data, current_scene_id, ground_desc, asset_id, categories, width, depth, multiply_factor, environment_applied,
                      wall_desc, wall_asset, wall_cats, wall_x, wall_x_neg, wall_y, wall_y_neg, wall_z,
                      api_key, api_base, provider, v_model):
        status, new_environment_applied, applied_id = apply_environment(
            blender_client, layout_data, current_scene_id, ground_desc, asset_id, categories, width, depth, multiply_factor, environment_applied,
            wall_description=wall_desc, wall_asset_id=wall_asset, wall_categories_limitation=wall_cats,
            wall_x_str=wall_x, wall_x_neg_str=wall_x_neg, wall_y_str=wall_y, wall_y_neg_str=wall_y_neg, wall_z_str=wall_z,
            anyllm_api_key=api_key, anyllm_api_base=api_base, anyllm_provider=provider, vision_model=v_model,
        )
        return status, new_environment_applied, applied_id
    
    apply_environment_btn.click(
        fn=apply_handler,
        inputs=[
            layout_data_state,
            current_scene_id_state,
            ground_description_input,
            asset_id_input,
            categories_limitation_input,
            width_input,
            depth_input,
            multiply_factor_input,
            environment_applied_state,
            # Wall inputs
            wall_description_input,
            wall_asset_id_input,
            wall_categories_input,
            wall_x_input,
            wall_x_neg_input,
            wall_y_input,
            wall_y_neg_input,
            wall_z_input,
            # Config inputs
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
            vision_model,
        ],
        outputs=[
            environment_status,
            environment_applied_state,
            applied_asset_id_display,
        ],
        concurrency_limit=None,
        show_progress="full",
    )
    
    # Finish button handler
    def finish_handler(proj_dir, layout_data, current_filepath, environment_applied):
        """Handle finish button click - save JSON, hide UI elements, display saved file."""
        result = finish_environment_creation(proj_dir, layout_data, current_filepath, environment_applied, layout_script_editor)
        
        if result.get("error"):
            # On error, keep UI visible and show error message
            return (
                gr.update(value=result["error"], visible=True),  # environment_status
                gr.update(),  # scene_buttons_row - no change
                gr.update(),  # environment_config_row - no change
                gr.update(),  # apply_all_ai_btn - no change
            ) + layout_script_editor.update_with_result(None)
        
        # Success - hide scene buttons and config, show JSON editor with saved file
        warning = result.get("warning", "")
        if warning:
            status_msg = f"{warning}\n\n✅ Layout script saved to: `{result['output_path']}`"
        else:
            status_msg = f"✅ **Environment creation complete!** All scenes have environment applied.\n\nLayout script saved to: `{result['output_path']}`"
        
        return (
            gr.update(value=status_msg, visible=True),  # environment_status
            gr.update(visible=False),  # scene_buttons_row - hide
            gr.update(visible=False),  # environment_config_row - hide
            gr.update(visible=False),  # apply_all_ai_btn - hide
        ) + layout_script_editor.update_with_result(result)
    
    finish_btn.click(
        fn=finish_handler,
        inputs=[
            project_dir,
            layout_data_state,
            current_filepath_state,
            environment_applied_state,
        ],
        outputs=[
            environment_status,
            scene_buttons_row,
            environment_config_row,
            batch_buttons_row,
        ] + layout_script_editor.get_output_components(),
    )
    
    # Apply All Scenes with AI button handler
    def apply_all_ai_handler(layout_data, environment_applied, proj_dir, api_key, api_base, provider, v_model, progress=gr.Progress(track_tqdm=False)):
        yield from apply_environment_all_scenes_ai(
            blender_client, layout_data, environment_applied, layout_script_editor, proj_dir,
            anyllm_api_key=api_key,
            anyllm_api_base=api_base,
            anyllm_provider=provider,
            vision_model=v_model,
            progress=progress,
        )
    
    apply_all_ai_btn.click(
        fn=apply_all_ai_handler,
        inputs=[
            layout_data_state,
            environment_applied_state,
            project_dir,
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
            vision_model,
        ],
        outputs=[
            environment_status,
            environment_applied_state,
            scene_buttons_row,
            environment_config_row,
            batch_buttons_row,
        ] + layout_script_editor.get_output_components(),
        concurrency_limit=None,
        show_progress="full",
    )
    
    # Resume Environment for All Scenes button handler
    def resume_all_handler(layout_data, environment_applied, proj_dir, progress=gr.Progress(track_tqdm=False)):
        yield from resume_environment_all_scenes(
            blender_client, layout_data, environment_applied, layout_script_editor, proj_dir,
            progress=progress,
        )
    
    resume_all_btn.click(
        fn=resume_all_handler,
        inputs=[
            layout_data_state,
            environment_applied_state,
            project_dir,
        ],
        outputs=[
            environment_status,
            environment_applied_state,
            scene_buttons_row,
            environment_config_row,
            batch_buttons_row,
        ] + layout_script_editor.get_output_components(),
        concurrency_limit=None,
        show_progress="full",
    )
    
    return {
        "environment_status": environment_status,
        "layout_data_state": layout_data_state,
        "layout_script_editor": layout_script_editor,
    }
