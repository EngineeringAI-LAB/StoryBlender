import os
import sys

# Allow running this file directly: python storyblender_app.py
if __name__ == "__main__" and not __package__:
    # Add the grandparent directory to sys.path so relative imports work.
    # Derive __package__ from actual directory names so this works both in the
    # repo (src/gradio_app) and as an installed Blender extension (storyblender/gradio_app).
    import types
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.dirname(_this_dir)
    _project_root = os.path.dirname(_parent_dir)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    _parent_name = os.path.basename(_parent_dir)
    _this_name = os.path.basename(_this_dir)
    # Register a lightweight parent package in sys.modules so Python does NOT
    # execute the real __init__.py (which imports bpy, only available inside
    # Blender).  The fake package just provides __path__ so sub-package
    # resolution (e.g. from ..operators...) works normally.
    _pkg = types.ModuleType(_parent_name)
    _pkg.__path__ = [_parent_dir]
    _pkg.__package__ = _parent_name
    sys.modules[_parent_name] = _pkg
    __package__ = f"{_parent_name}.{_this_name}"

import gradio as gr
from .step_1_generate_base_storyboard_script_by_director import create_director_ui
from .step_2_generate_and_format_core_3d_assets_by_concept_artist import create_3d_assets_ui
from .step_3_formulate_scene_core_spatial_layout_by_layout_artist import create_layout_ui
from .step_4_generate_supplementary_assets_by_set_dresser import create_supplementary_assets_ui
from .step_5_formulate_scene_supplementary_spatial_layout_by_set_dresser import create_supplementary_layout_ui
from .step_6_apply_lighting_with_lighting_disigner import create_lighting_designer_ui
from .step_7_create_environment_with_environment_artist import create_environment_artist_ui
from .step_8_apply_asset_modifications_in_all_shots import create_apply_asset_modifications_ui
from .step_9_rigging_and_animate_assets_by_animator import create_animator_rigging_ui
from .step_10_camera_blocking_by_layout_artist import create_camera_blocking_ui
from .step_11_post_process import create_post_process_ui
from .step_12_rendering import create_rendering_ui
from .step_13_stich_frames_to_video import create_stitch_frames_ui
from .config import create_config_ui, setup_config_handlers
from .blender_client import BlenderClient


COMPACT_CSS = """
#StoryBlenderAPP {
  max-width: 900px;
  margin-left: auto;
  margin-right: auto;
}

#StoryBlenderAPP .gradio-container {
  padding: 1rem !important;
}
"""


# Initialize BlenderClient for communicating with Blender
# This will be used by later steps that require Blender operations
# Port is dynamically retrieved from Blender's bpy.context.scene.blendermcp_port
blender_client = BlenderClient(host='localhost')

# In Gradio 6, css moved from Blocks() to launch(); see launch() call below.
with gr.Blocks(elem_id="StoryBlenderAPP") as StoryBlenderAPP:
    gr.Markdown("# StoryBlender - AI Storyboard Generator")
    gr.Markdown("Configure your API settings and generate storyboard scripts from your story ideas.")
    
    # Create configuration UI
    config = create_config_ui()
    image_gen_platform = config["image_gen_platform"]
    gemini_image_model = config["gemini_image_model"]
    gemini_api_key = config["gemini_api_key"]
    gemini_api_base = config["gemini_api_base"]
    openai_api_key = config["openai_api_key"]
    openai_api_base = config["openai_api_base"]
    openai_image_model = config["openai_image_model"]
    reasoning_model = config["reasoning_model"]
    vision_model = config["vision_model"]
    anyllm_api_key = config["anyllm_api_key"]
    anyllm_api_base = config["anyllm_api_base"]
    anyllm_provider = config["anyllm_provider"]
    sketchfab_api_key = config["sketchfab_api_key"]
    meshy_api_key = config["meshy_api_key"]
    meshy_model = config["meshy_model"]
    uthana_api_key = config["uthana_api_key"]
    uthana_fps = config["uthana_fps"]
    tencent_secret_id = config["tencent_secret_id"]
    tencent_secret_key = config["tencent_secret_key"]
    ai_platform = config["ai_platform"]
    project_dir = config["project_dir"]
    
    gr.Markdown("---")
    
    # Step 1: Generate Base Storyboard Script by Director
    director_ui = create_director_ui(reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider, project_dir)
    
    gr.Markdown("---")
    
    # Step 2: Fetch and Format Core 3D Assets by Concept Artist
    assets_ui = create_3d_assets_ui(
        anyllm_api_key,
        anyllm_api_base,
        anyllm_provider,
        sketchfab_api_key,
        meshy_api_key,
        meshy_model,
        gemini_api_key,
        gemini_api_base,
        reasoning_model,
        gemini_image_model,
        project_dir,
        blender_client,
        vision_model,
        ai_platform,
        tencent_secret_id,
        tencent_secret_key,
        image_gen_platform,
        openai_api_key,
        openai_api_base,
        openai_image_model
    )
    
    gr.Markdown("---")
    
    # Step 3: Formulate Scene Spatial Layout by Production Designer
    layout_ui = create_layout_ui(
        reasoning_model,
        anyllm_api_key,
        anyllm_api_base,
        anyllm_provider,
        project_dir,
        blender_client
    )
    
    gr.Markdown("---")
    
    # Step 4: Design and Fetch Supplementary Assets by Set Dresser and Concept Artist
    supplementary_ui = create_supplementary_assets_ui(
        anyllm_api_key,
        anyllm_api_base,
        anyllm_provider,
        reasoning_model,
        sketchfab_api_key,
        meshy_api_key,
        meshy_model,
        gemini_api_key,
        gemini_api_base,
        gemini_image_model,
        project_dir,
        vision_model,
        ai_platform,
        tencent_secret_id,
        tencent_secret_key,
        blender_client
    )
    
    gr.Markdown("---")
    
    # Step 5: Formulate Scene Supplementary Spatial Layout by Set Dresser
    supplementary_layout_ui = create_supplementary_layout_ui(
        reasoning_model,
        anyllm_api_key,
        anyllm_api_base,
        anyllm_provider,
        project_dir,
        blender_client
    )
    
    gr.Markdown("---")
    
    # Step 6: Apply Lighting with Lighting Designer
    lighting_designer_ui = create_lighting_designer_ui(
        project_dir,
        blender_client,
        anyllm_api_key,
        anyllm_api_base,
        anyllm_provider,
        vision_model,
    )
    
    gr.Markdown("---")
    
    # Step 7: Create Environment with Environment Artist
    environment_artist_ui = create_environment_artist_ui(
        project_dir,
        blender_client,
        anyllm_api_key,
        anyllm_api_base,
        anyllm_provider,
        vision_model,
    )
    
    gr.Markdown("---")
    
    # Step 8: Apply Asset Modifications in All Shots
    apply_asset_modifications_ui = create_apply_asset_modifications_ui(
        project_dir,
        blender_client
    )
    
    gr.Markdown("---")
    
    # Step 9: Rigging and Animate Assets by Animator
    animator_rigging_ui = create_animator_rigging_ui(
        meshy_api_key,
        project_dir,
        anyllm_api_key,
        anyllm_api_base,
        anyllm_provider,
        vision_model,
        reasoning_model,
        blender_client,
        uthana_api_key=uthana_api_key,
        uthana_fps=uthana_fps,
    )
    
    gr.Markdown("---")
    
    # Step 10: Camera Blocking by Layout Artist
    camera_blocking_ui = create_camera_blocking_ui(
        vision_model,
        anyllm_api_key,
        anyllm_api_base,
        anyllm_provider,
        project_dir,
        blender_client,
        reasoning_model
    )
    
    gr.Markdown("---")
    
    # Step 11: Post Process
    post_process_ui = create_post_process_ui(
        project_dir,
        blender_client
    )
    
    gr.Markdown("---")
    
    # Step 12: Rendering
    rendering_ui = create_rendering_ui(
        project_dir
    )
    
    gr.Markdown("---")
    
    # Step 13: Stitch Frames to Video
    stitch_frames_ui = create_stitch_frames_ui(
        project_dir
    )
    
    gr.Markdown("---")
    
    # Setup configuration button handlers (must be at end after all UI is created)
    setup_config_handlers(config)

# This app does NO local ML inference — all heavy work is delegated to Blender via
# socket.  Setting default_concurrency_limit=None removes the per-function queue
# serialisation bottleneck so lightweight UI handlers (config save, visibility toggles,
# JSON reads) are never blocked behind a slow Blender socket call.
StoryBlenderAPP.queue(
    api_open=False,
    default_concurrency_limit=None,
)

if __name__ == "__main__":
    # Allow loading 3D models from various directories
    # Add the parent src directory to cover most use cases
    import os
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Accept port from environment variable (set by Blender addon subprocess launcher)
    port = int(os.environ.get('GRADIO_SERVER_PORT', '7860'))

    # Disable analytics and API docs to reduce background thread activity
    # Note: project_dir is added dynamically via gr.set_static_paths() when user saves config
    StoryBlenderAPP.launch(
        server_port=port,
        server_name="127.0.0.1",
        allowed_paths=[src_dir],
        css=COMPACT_CSS,
        max_threads=40,
    )
