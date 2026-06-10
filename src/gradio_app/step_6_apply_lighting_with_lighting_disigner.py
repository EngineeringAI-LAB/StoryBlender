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
                logger.warning("Step 6: path conversion failed for layout script: %s", e)
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


def initialize_lighting_designer(project_dir):
    """Initialize the lighting designer by loading layout script and creating scene buttons."""
    if not project_dir or not os.path.isabs(project_dir):
        return (
            gr.update(value="⚠️ Please set a valid absolute project directory path.", visible=True),
            gr.update(visible=False),  # scene_buttons_row
            None,  # layout_data state
            None,  # current_filepath state
            {},    # lighting_applied state
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
        gr.update(value=f"✅ Loaded layout script: `{filepath}` with {num_scenes} scene(s). Click a scene button to configure lighting.", visible=True),
        gr.update(visible=True),
        layout_data,
        filepath,
        {},  # Reset lighting_applied state
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
    if not layout_data:
        return (
            gr.update(value="⚠️ Layout data not loaded. Please initialize first.", visible=True),
            gr.update(visible=False),  # lighting_config_row
            "",  # lighting_description
            "",  # asset_id
            "",  # categories_limitation
            None,  # current_scene_id
        )
    
    # Ensure MCP server is running
    success, message = blender_client.ensure_server_running()
    if not success:
        return (
            gr.update(value=f"⚠️ {message}", visible=True),
            gr.update(visible=False),
            "",
            "",
            "",
            None,
        )
    
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
            None,
        )
    
    # Get scene setup
    scene_setup = get_scene_setup_by_id(layout_data, scene_id)
    if not scene_setup:
        return (
            gr.update(value=f"⚠️ Could not find scene_setup for scene {scene_id}.", visible=True),
            gr.update(visible=False),
            "",
            "",
            "",
            None,
        )
    
    # Extract values
    scene_type = scene_setup.get("scene_type", "outdoor")
    lighting_description = scene_setup.get("lighting_description", "")
    lighting_asset_id = scene_setup.get("lighting_asset_id", "")
    
    # Determine categories limitation based on scene_type
    if scene_type == "indoor":
        categories_limitation = '["indoor"]'
    else:  # outdoor
        categories_limitation = '["pure skies"]'
    
    return (
        gr.update(value=f"✅ Switched to **Scene {scene_id}** ({scene_type}). Configure lighting below and click 'Apply Lighting'.", visible=True),
        gr.update(visible=True),
        lighting_description,
        lighting_asset_id if lighting_asset_id else "",
        categories_limitation,
        scene_id,
    )


def apply_lighting(blender_client, layout_data, current_scene_id, lighting_description, asset_id, categories_limitation, lighting_applied, anyllm_api_key=None, anyllm_api_base=None, anyllm_provider="gemini", vision_model="gemini-3-flash-preview"):
    """Apply lighting to the current scene using lighting_designer."""
    if not current_scene_id:
        return (
            gr.update(value="⚠️ No scene selected. Please click a scene button first.", visible=True),
            lighting_applied,
            "",  # applied_asset_id
        )
    
    # Ensure MCP server is running
    success, message = blender_client.ensure_server_running()
    if not success:
        return (
            gr.update(value=f"⚠️ {message}", visible=True),
            lighting_applied,
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
    
    # Prepare parameters for lighting_designer
    # If asset_id is provided and not empty, use it directly
    asset_id_param = asset_id.strip() if asset_id and asset_id.strip() else None
    scene_desc_param = lighting_description.strip() if lighting_description and lighting_description.strip() else None
    
    # Call lighting_designer
    response = blender_client.lighting_designer(
        scene_description=scene_desc_param,
        asset_id=asset_id_param,
        categories_limitation=categories_list,
        anyllm_api_key=anyllm_api_key,
        anyllm_api_base=anyllm_api_base,
        anyllm_provider=anyllm_provider,
        vision_model=vision_model,
    )
    
    if response.get("status") == "error":
        return (
            gr.update(value=f"⚠️ Lighting designer error: {response.get('message', 'Unknown error')}", visible=True),
            lighting_applied,
            "",
        )
    
    result = response.get("result", {})
    
    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        return (
            gr.update(value=f"⚠️ Lighting designer failed: {error_msg}", visible=True),
            lighting_applied,
            "",
        )
    
    # Get the applied asset_id from the result
    applied_asset_id = result.get("asset_id", "")
    hdri_name = result.get("hdri_name", applied_asset_id)
    
    # Update lighting_applied state
    new_lighting_applied = lighting_applied.copy() if lighting_applied else {}
    new_lighting_applied[current_scene_id] = applied_asset_id
    
    return (
        gr.update(value=f"✅ Lighting applied to Scene {current_scene_id}!\n\n**HDRI:** {hdri_name}\n**Asset ID:** `{applied_asset_id}`", visible=True),
        new_lighting_applied,
        applied_asset_id,
    )


def finish_lighting_design(project_dir, layout_data, current_filepath, lighting_applied, layout_script_editor):
    """Finish lighting design - save updated layout_script with lighting_asset_id values.
    
    Returns:
        tuple: (status_update, scene_buttons_row_update, lighting_config_row_update, 
                layout_data, output_path, warning_message)
    """
    if not layout_data:
        return {
            "error": "⚠️ Layout data not loaded. Please initialize first."
        }
    
    if not project_dir or not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Please set a valid absolute project directory path."
        }
    
    # Check which scenes don't have lighting applied
    scene_details = layout_data.get("scene_details", [])
    scenes_without_lighting = []
    
    for scene_detail in scene_details:
        scene_id = scene_detail.get("scene_id")
        if scene_id not in (lighting_applied or {}):
            # Check if lighting_asset_id already exists in the JSON
            scene_setup = scene_detail.get("scene_setup", {})
            if not scene_setup.get("lighting_asset_id"):
                scenes_without_lighting.append(scene_id)
    
    # Update layout_data with lighting_asset_id values
    for scene_detail in scene_details:
        scene_id = scene_detail.get("scene_id")
        if scene_id in (lighting_applied or {}):
            if "scene_setup" not in scene_detail:
                scene_detail["scene_setup"] = {}
            scene_detail["scene_setup"]["lighting_asset_id"] = lighting_applied[scene_id]
    
    # Save using the layout_script_editor
    layout_script_dir = os.path.join(project_dir, "layout_script")
    layout_script_editor.set_save_path(layout_script_dir)
    output_path = layout_script_editor.save_json_data(layout_data)
    
    if not output_path:
        return {
            "error": "⚠️ Failed to save layout script."
        }
    
    # Build result
    if scenes_without_lighting:
        scenes_str = ", ".join([str(s) for s in scenes_without_lighting])
        warning = f"⚠️ **Warning:** Scenes {scenes_str} do not have lighting applied."
    else:
        warning = None
    
    return {
        "success": True,
        "data": layout_data,
        "output_path": output_path,
        "warning": warning,
    }


def apply_lighting_all_scenes_ai(blender_client, layout_data, lighting_applied, layout_script_editor, project_dir, anyllm_api_key=None, anyllm_api_base=None, anyllm_provider="gemini", vision_model="gemini-3-flash-preview", progress=gr.Progress(track_tqdm=False)):
    """Apply AI-selected lighting to ALL scenes in sequence, then auto-finish.
    
    Yields intermediate status updates as each scene is processed.
    Final yield includes all outputs needed to save and reinitialize.
    """
    if not layout_data:
        yield (
            gr.update(value="⚠️ Layout data not loaded. Please initialize first.", visible=True),
            lighting_applied,
            gr.update(),  # scene_buttons_row
            gr.update(),  # lighting_config_row
            gr.update(),  # apply_all_ai_btn
        ) + layout_script_editor.update_with_result(None)
        return
    
    # Ensure MCP server is running
    success, message = blender_client.ensure_server_running()
    if not success:
        yield (
            gr.update(value=f"⚠️ {message}", visible=True),
            lighting_applied,
            gr.update(),
            gr.update(),
            gr.update(),
        ) + layout_script_editor.update_with_result(None)
        return
    
    scene_details = layout_data.get("scene_details", [])
    num_scenes = len(scene_details)
    new_lighting_applied = lighting_applied.copy() if lighting_applied else {}
    
    for idx, scene_detail in enumerate(scene_details):
        scene_id = scene_detail.get("scene_id")
        scene_setup = scene_detail.get("scene_setup", {})
        scene_type = scene_setup.get("scene_type", "outdoor")
        lighting_description = scene_setup.get("lighting_description", "")
        categories_list = ["indoor"] if scene_type == "indoor" else ["pure skies"]
        
        progress((idx, num_scenes), desc=f"Scene {scene_id}")
        
        # Yield progress status
        yield (
            gr.update(value=f"⏳ Applying lighting to **Scene {scene_id}** ({idx + 1}/{num_scenes})...", visible=True),
            new_lighting_applied,
            gr.update(),
            gr.update(),
            gr.update(),
        ) + layout_script_editor.update_with_result(None)
        
        # Switch to the scene
        scene_name = f"Scene_{scene_id}"
        response = blender_client.switch_or_create_scene(scene_name=scene_name)
        if response.get("status") == "error":
            yield (
                gr.update(value=f"⚠️ Failed to switch to Scene {scene_id}: {response.get('message', 'Unknown error')}. Skipping.", visible=True),
                new_lighting_applied,
                gr.update(),
                gr.update(),
                gr.update(),
            ) + layout_script_editor.update_with_result(None)
            continue
        
        # Apply lighting with AI (no manual asset_id)
        response = blender_client.lighting_designer(
            scene_description=lighting_description.strip() if lighting_description else None,
            asset_id=None,
            categories_limitation=categories_list,
            anyllm_api_key=anyllm_api_key,
            anyllm_api_base=anyllm_api_base,
            anyllm_provider=anyllm_provider,
            vision_model=vision_model,
        )
        
        if response.get("status") == "error":
            yield (
                gr.update(value=f"⚠️ Lighting designer error for Scene {scene_id}: {response.get('message', 'Unknown error')}. Skipping.", visible=True),
                new_lighting_applied,
                gr.update(),
                gr.update(),
                gr.update(),
            ) + layout_script_editor.update_with_result(None)
            continue
        
        result = response.get("result", {})
        if not result.get("success"):
            yield (
                gr.update(value=f"⚠️ Lighting failed for Scene {scene_id}: {result.get('error', 'Unknown')}. Skipping.", visible=True),
                new_lighting_applied,
                gr.update(),
                gr.update(),
                gr.update(),
            ) + layout_script_editor.update_with_result(None)
            continue
        
        applied_asset_id = result.get("asset_id", "")
        new_lighting_applied[scene_id] = applied_asset_id
    
    progress((num_scenes, num_scenes), desc="Saving")
    
    # Auto-finish: save JSON and reinitialize
    finish_result = finish_lighting_design(project_dir, layout_data, None, new_lighting_applied, layout_script_editor)
    
    if finish_result.get("error"):
        yield (
            gr.update(value=finish_result["error"], visible=True),
            new_lighting_applied,
            gr.update(),
            gr.update(),
            gr.update(),
        ) + layout_script_editor.update_with_result(None)
        return
    
    # Build summary of applied lighting
    summary_lines = []
    for scene_detail in scene_details:
        sid = scene_detail.get("scene_id")
        aid = new_lighting_applied.get(sid, "❌ not applied")
        summary_lines.append(f"- **Scene {sid}:** `{aid}`")
    summary = "\n".join(summary_lines)
    
    warning = finish_result.get("warning", "")
    if warning:
        status_msg = f"{warning}\n\n{summary}\n\n✅ Layout script saved to: `{finish_result['output_path']}`"
    else:
        status_msg = f"✅ **Lighting applied to all {num_scenes} scene(s) with AI!**\n\n{summary}\n\nLayout script saved to: `{finish_result['output_path']}`"
    
    yield (
        gr.update(value=status_msg, visible=True),
        new_lighting_applied,
        gr.update(visible=False),  # scene_buttons_row - hide
        gr.update(visible=False),  # lighting_config_row - hide
        gr.update(visible=False),  # apply_all_ai_btn - hide
    ) + layout_script_editor.update_with_result(finish_result)


def resume_lighting_all_scenes(blender_client, layout_data, lighting_applied, layout_script_editor, project_dir, progress=gr.Progress(track_tqdm=False)):
    """Resume lighting for ALL scenes using `lighting_asset_id` already stored in the
    layout script (bypasses AI selection). Effectively runs the same path as the
    manual 'Asset ID' field, but iterates over every scene with one click.

    Yields the same output tuple shape as `apply_lighting_all_scenes_ai`.
    """
    if not layout_data:
        yield (
            gr.update(value="⚠️ Layout data not loaded. Please initialize first.", visible=True),
            lighting_applied,
            gr.update(), gr.update(), gr.update(),
        ) + layout_script_editor.update_with_result(None)
        return

    success, message = blender_client.ensure_server_running()
    if not success:
        yield (
            gr.update(value=f"⚠️ {message}", visible=True),
            lighting_applied,
            gr.update(), gr.update(), gr.update(),
        ) + layout_script_editor.update_with_result(None)
        return

    scene_details = layout_data.get("scene_details", [])
    num_scenes = len(scene_details)
    new_lighting_applied = lighting_applied.copy() if lighting_applied else {}
    skipped = []

    for idx, scene_detail in enumerate(scene_details):
        scene_id = scene_detail.get("scene_id")
        scene_setup = scene_detail.get("scene_setup", {})
        scene_type = scene_setup.get("scene_type", "outdoor")
        lighting_asset_id = (scene_setup.get("lighting_asset_id") or "").strip()
        categories_list = ["indoor"] if scene_type == "indoor" else ["pure skies"]

        progress((idx, num_scenes), desc=f"Scene {scene_id}")

        if not lighting_asset_id:
            skipped.append(scene_id)
            yield (
                gr.update(value=f"⚠️ Scene {scene_id} has no `lighting_asset_id` in layout JSON. Skipping.", visible=True),
                new_lighting_applied,
                gr.update(), gr.update(), gr.update(),
            ) + layout_script_editor.update_with_result(None)
            continue

        yield (
            gr.update(value=f"⏳ Resuming lighting for **Scene {scene_id}** ({idx + 1}/{num_scenes}) using asset `{lighting_asset_id}`...", visible=True),
            new_lighting_applied,
            gr.update(), gr.update(), gr.update(),
        ) + layout_script_editor.update_with_result(None)

        scene_name = f"Scene_{scene_id}"
        response = blender_client.switch_or_create_scene(scene_name=scene_name)
        if response.get("status") == "error":
            yield (
                gr.update(value=f"⚠️ Failed to switch to Scene {scene_id}: {response.get('message', 'Unknown error')}. Skipping.", visible=True),
                new_lighting_applied,
                gr.update(), gr.update(), gr.update(),
            ) + layout_script_editor.update_with_result(None)
            continue

        # Apply with explicit asset_id (bypasses AI selection)
        response = blender_client.lighting_designer(
            scene_description=None,
            asset_id=lighting_asset_id,
            categories_limitation=categories_list,
        )

        if response.get("status") == "error":
            yield (
                gr.update(value=f"⚠️ Lighting designer error for Scene {scene_id}: {response.get('message', 'Unknown error')}. Skipping.", visible=True),
                new_lighting_applied,
                gr.update(), gr.update(), gr.update(),
            ) + layout_script_editor.update_with_result(None)
            continue

        result = response.get("result", {})
        if not result.get("success"):
            yield (
                gr.update(value=f"⚠️ Lighting failed for Scene {scene_id}: {result.get('error', 'Unknown')}. Skipping.", visible=True),
                new_lighting_applied,
                gr.update(), gr.update(), gr.update(),
            ) + layout_script_editor.update_with_result(None)
            continue

        applied_asset_id = result.get("asset_id", lighting_asset_id)
        new_lighting_applied[scene_id] = applied_asset_id

    progress((num_scenes, num_scenes), desc="Saving")

    finish_result = finish_lighting_design(project_dir, layout_data, None, new_lighting_applied, layout_script_editor)

    if finish_result.get("error"):
        yield (
            gr.update(value=finish_result["error"], visible=True),
            new_lighting_applied,
            gr.update(), gr.update(), gr.update(),
        ) + layout_script_editor.update_with_result(None)
        return

    summary_lines = []
    for scene_detail in scene_details:
        sid = scene_detail.get("scene_id")
        aid = new_lighting_applied.get(sid, "❌ not applied")
        summary_lines.append(f"- **Scene {sid}:** `{aid}`")
    summary = "\n".join(summary_lines)

    header = f"✅ **Resumed lighting for {num_scenes - len(skipped)}/{num_scenes} scene(s) from layout JSON.**"
    if skipped:
        header += f"\n\n⚠️ Skipped scenes (no `lighting_asset_id`): {', '.join(str(s) for s in skipped)}"
    warning = finish_result.get("warning", "")
    extra = f"\n\n{warning}" if warning else ""
    status_msg = f"{header}{extra}\n\n{summary}\n\nLayout script saved to: `{finish_result['output_path']}`"

    yield (
        gr.update(value=status_msg, visible=True),
        new_lighting_applied,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    ) + layout_script_editor.update_with_result(finish_result)


def create_scene_button_click_handler(blender_client, scene_id):
    """Create a click handler for a specific scene button."""
    def handler(layout_data):
        return select_scene(blender_client, layout_data, scene_id)
    return handler


def create_lighting_designer_ui(project_dir, blender_client, anyllm_api_key=None, anyllm_api_base=None, anyllm_provider=None, vision_model=None):
    """Create the Step 6: Apply Lighting with Lighting Designer UI section.
    
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
    gr.Markdown("## Step 6: Apply Lighting with Lighting Designer")
    gr.Markdown("Apply HDRI lighting to each scene using AI-powered HDRI selection or manual asset ID specification.")
    
    # Input/Output info
    gr.Markdown("**Input:** `project_dir/layout_script/layout_script_v{N}.json` (latest)")
    gr.Markdown("**Output:** `project_dir/layout_script/layout_script_v{N+1}.json` (with lighting_asset_id)")
    
    # State variables
    layout_data_state = gr.State(value=None)
    current_filepath_state = gr.State(value=None)
    current_scene_id_state = gr.State(value=None)
    lighting_applied_state = gr.State(value={})
    
    # Initialize button
    initialize_btn = gr.Button("🔦 Initialize Lighting Designer", variant="primary", size="lg")
    
    # JSON Editor for viewing/editing layout_script
    layout_script_editor = JSONEditorComponent(
        label="Layout Script JSON",
        visible_initially=False,
        file_basename="layout_script",
        json_root_keys_list=["scene_details"],
        title="Step 6"
    )
    
    # Wire up the Resume button with project_dir input
    layout_script_editor.setup_resume_with_project_dir(project_dir, subfolder="layout_script")
    
    # Status indicator
    lighting_status = gr.Markdown(value="", visible=False)
    
    # Batch buttons (hidden initially)
    with gr.Row(visible=False) as batch_buttons_row:
        apply_all_ai_btn = gr.Button(
            "🤖 Apply Lighting to All Scenes with AI",
            variant="primary", size="lg",
        )
        resume_all_btn = gr.Button(
            "♻️ Resume Lighting for All Scenes",
            variant="secondary", size="lg",
        )
    
    gr.Markdown("*Or select a scene below to manually configure lighting:*", visible=True)
    
    # Scene buttons row (hidden initially)
    with gr.Row(visible=False) as scene_buttons_row:
        # Create buttons for up to 10 scenes (can be extended if needed)
        scene_buttons = []
        for i in range(1, 11):
            btn = gr.Button(f"Scene {i}", variant="secondary", visible=(i <= 3))  # Show first 3 by default
            scene_buttons.append(btn)
    
    # Lighting configuration row (hidden initially)
    with gr.Column(visible=False) as lighting_config_row:
        gr.Markdown("### Lighting Configuration")
        with gr.Row():
            lighting_description_input = gr.Textbox(
                label="Lighting Description",
                placeholder="e.g., Dim, moody castle lighting with a spotlight on the mirror.",
                lines=2
            )
            asset_id_input = gr.Textbox(
                label="Asset ID (Optional - bypasses AI selection)",
                placeholder="e.g., basement_boxing_ring (leave empty for AI selection)",
                lines=1
            )
            categories_limitation_input = gr.Textbox(
                label="Categories Limitation",
                placeholder='e.g., ["indoor"] or ["pure skies"]',
                lines=1
            )
        
        applied_asset_id_display = gr.Textbox(
            label="Applied Asset ID (from last apply)",
            interactive=False,
            visible=True
        )
        
        with gr.Row():
            apply_lighting_btn = gr.Button("🌅 Apply Lighting with Lighting Designer", variant="primary", size="lg")
            finish_btn = gr.Button("✅ Finish Lighting Design", variant="secondary", size="lg")
    
    # Initialize button handler
    def init_handler(proj_dir):
        result = initialize_lighting_designer(proj_dir)
        status_update, scene_row_update, layout_data, filepath, lighting_applied = result
        
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
        
        return (status_update, scene_row_update, layout_data, filepath, lighting_applied, gr.update(visible=show_ai_btn)) + tuple(button_updates)
    
    initialize_btn.click(
        fn=init_handler,
        inputs=[project_dir],
        outputs=[
            lighting_status,
            scene_buttons_row,
            layout_data_state,
            current_filepath_state,
            lighting_applied_state,
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
                lighting_status,
                lighting_config_row,
                lighting_description_input,
                asset_id_input,
                categories_limitation_input,
                current_scene_id_state,
            ],
            concurrency_limit=None,
            show_progress="hidden",
        )
    
    # Apply lighting button handler
    def apply_handler(layout_data, current_scene_id, lighting_desc, asset_id, categories, lighting_applied, api_key, api_base, provider, v_model):
        status, new_lighting_applied, applied_id = apply_lighting(
            blender_client, layout_data, current_scene_id, lighting_desc, asset_id, categories, lighting_applied,
            anyllm_api_key=api_key,
            anyllm_api_base=api_base,
            anyllm_provider=provider,
            vision_model=v_model,
        )
        return status, new_lighting_applied, applied_id
    
    apply_lighting_btn.click(
        fn=apply_handler,
        inputs=[
            layout_data_state,
            current_scene_id_state,
            lighting_description_input,
            asset_id_input,
            categories_limitation_input,
            lighting_applied_state,
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
            vision_model,
        ],
        outputs=[
            lighting_status,
            lighting_applied_state,
            applied_asset_id_display,
        ],
        concurrency_limit=None,
        show_progress="full",
    )
    
    # Finish button handler
    def finish_handler(proj_dir, layout_data, current_filepath, lighting_applied):
        """Handle finish button click - save JSON, hide UI elements, display saved file."""
        result = finish_lighting_design(proj_dir, layout_data, current_filepath, lighting_applied, layout_script_editor)
        
        if result.get("error"):
            # On error, keep UI visible and show error message
            return (
                gr.update(value=result["error"], visible=True),  # lighting_status
                gr.update(),  # scene_buttons_row - no change
                gr.update(),  # lighting_config_row - no change
                gr.update(),  # apply_all_ai_btn - no change
            ) + layout_script_editor.update_with_result(None)
        
        # Success - hide scene buttons and config, show JSON editor with saved file
        warning = result.get("warning", "")
        if warning:
            status_msg = f"{warning}\n\n✅ Layout script saved to: `{result['output_path']}`"
        else:
            status_msg = f"✅ **Lighting design complete!** All scenes have lighting applied.\n\nLayout script saved to: `{result['output_path']}`"
        
        return (
            gr.update(value=status_msg, visible=True),  # lighting_status
            gr.update(visible=False),  # scene_buttons_row - hide
            gr.update(visible=False),  # lighting_config_row - hide
            gr.update(visible=False),  # apply_all_ai_btn - hide
        ) + layout_script_editor.update_with_result(result)
    
    finish_btn.click(
        fn=finish_handler,
        inputs=[
            project_dir,
            layout_data_state,
            current_filepath_state,
            lighting_applied_state,
        ],
        outputs=[
            lighting_status,
            scene_buttons_row,
            lighting_config_row,
            batch_buttons_row,
        ] + layout_script_editor.get_output_components(),
    )
    
    # Apply All Scenes with AI button handler
    def apply_all_ai_handler(layout_data, lighting_applied, proj_dir, api_key, api_base, provider, v_model, progress=gr.Progress(track_tqdm=False)):
        yield from apply_lighting_all_scenes_ai(
            blender_client, layout_data, lighting_applied, layout_script_editor, proj_dir,
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
            lighting_applied_state,
            project_dir,
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
            vision_model,
        ],
        outputs=[
            lighting_status,
            lighting_applied_state,
            scene_buttons_row,
            lighting_config_row,
            batch_buttons_row,
        ] + layout_script_editor.get_output_components(),
        concurrency_limit=None,
        show_progress="full",
    )
    
    # Resume Lighting for All Scenes button handler
    def resume_all_handler(layout_data, lighting_applied, proj_dir, progress=gr.Progress(track_tqdm=False)):
        yield from resume_lighting_all_scenes(
            blender_client, layout_data, lighting_applied, layout_script_editor, proj_dir,
            progress=progress,
        )
    
    resume_all_btn.click(
        fn=resume_all_handler,
        inputs=[
            layout_data_state,
            lighting_applied_state,
            project_dir,
        ],
        outputs=[
            lighting_status,
            lighting_applied_state,
            scene_buttons_row,
            lighting_config_row,
            batch_buttons_row,
        ] + layout_script_editor.get_output_components(),
        concurrency_limit=None,
        show_progress="full",
    )
    
    return {
        "lighting_status": lighting_status,
        "layout_data_state": layout_data_state,
        "layout_script_editor": layout_script_editor,
    }
