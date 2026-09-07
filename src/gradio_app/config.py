"""Configuration UI components and handlers for StoryBlender."""

import os
import gradio as gr

try:
    from ..env_config import get_env, get_int_env
except ImportError:
    from env_config import get_env, get_int_env

from .manual_adjustments import (
    compact_pending_plan,
    commit_manual_adjustments,
    format_preview_rows,
    format_preview_warnings,
    prepare_manual_adjustments,
)


def _choice_env(name, default, choices):
    value = get_env(name, default)
    return value if value in choices else default


def create_config_ui():
    """Create configuration UI components.
    
    Returns:
        A dictionary containing all configuration components.
    """
    md_title = gr.Markdown("## Configuration")
    
    md_image_gen = gr.Markdown("### Image Generation")
    with gr.Row():
        image_gen_platform = gr.Dropdown(
            label="Image Generation Platform",
            choices=["Gemini", "OpenAI"],
            value=_choice_env("STORYBLENDER_IMAGE_GEN_PLATFORM", "OpenAI", ["Gemini", "OpenAI"]),
            info="Choose which platform to use for text-to-image generation",
            visible=True
        )
    
    with gr.Row():
        gemini_image_model = gr.Textbox(
            label="Gemini Image Model",
            value=get_env("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview"),
            info="The Gemini image model used during text to image to 3D",
            visible=True
        )
        gemini_api_key = gr.Textbox(
            label="Gemini API Key",
            value=get_env("GEMINI_API_KEY"),
            type="password",
            info="Your Gemini API key for authentication",
            visible=True
        )
        gemini_api_base = gr.Textbox(
            label="Gemini API Base",
            value=get_env("GEMINI_API_BASE"),
            info="Custom API base URL for Gemini (leave empty for default)",
            visible=True
        )
    
    with gr.Row():
        openai_image_model = gr.Textbox(
            label="OpenAI Image Model",
            value=get_env("OPENAI_IMAGE_MODEL", "grok-imagine-image"),
            info="The OpenAI model used for image generation (accept OpenAI compatible models, e.g., grok-imagine-image)",
            visible=True
        )
        openai_api_key = gr.Textbox(
            label="OpenAI API Key",
            value=get_env("OPENAI_API_KEY"),
            type="password",
            info="Your OpenAI API key for image generation (required if using OpenAI)",
            visible=True
        )
        openai_api_base = gr.Textbox(
            label="OpenAI API Base",
            value=get_env("OPENAI_API_BASE"),
            info="API base URL for OpenAI (required if using OpenAI)",
            visible=True
        )

    md_sep1 = gr.Markdown("---")

    md_reasoning = gr.Markdown("### Reasoning & Vision Models")
    with gr.Row():
        # Reasoning Model
        reasoning_model = gr.Textbox(
            label="Reasoning Model",
            value=get_env("STORYBLENDER_REASONING_MODEL", "gemini-3.1-pro-preview"),
            info="The primary reasoning model for complex reasoning tasks",
            visible=True
        )
        vision_model = gr.Textbox(
            label="Vision Model",
            value=get_env("STORYBLENDER_VISION_MODEL", "gemini-3-flash-preview"),
            info="A lighter model for fast multi-model inference",
            visible=True
        )
        anyllm_api_key = gr.Textbox(
            label="AnyLLM API Key",
            value=get_env("ANYLLM_API_KEY"),
            type="password",
            info="Your AnyLLM API key",
            visible=True
        )
        anyllm_api_base = gr.Textbox(
            label="AnyLLM API Base",
            value=get_env("ANYLLM_API_BASE"),
            info="API base URL for AnyLLM",
            visible=True
        )
        anyllm_provider = gr.Textbox(
            label="AnyLLM Provider",
            value=get_env("ANYLLM_PROVIDER", "gemini"),
            info="LLM provider (default: gemini)",
            visible=True
        )
    
    md_sep2 = gr.Markdown("---")

    md_3d_services = gr.Markdown("### 3D Model Services")
    with gr.Row():
        sketchfab_api_key = gr.Textbox(
            label="Sketchfab API Key",
            value=get_env("SKETCHFAB_API_KEY"),
            type="password",
            info="Your Sketchfab API key for 3D model retrieval",
            visible=True
        )
    
    with gr.Row():
        meshy_api_key = gr.Textbox(
            label="Meshy API Key",
            value=get_env("MESHY_API_KEY"),
            type="password",
            info="Your Meshy API key for 3D model generation",
            visible=True
        )
        meshy_model = gr.Textbox(
                label="Meshy Model",
                value=get_env("MESHY_MODEL", "latest"),
                info="Meshy AI model version to use for 3D generation (default: 'latest')",
                visible=True
        )
    
    with gr.Row():
        tencent_secret_id = gr.Textbox(
            label="Tencent Cloud Secret ID",
            value=get_env("TENCENT_SECRET_ID"),
            type="password",
            info="Your Tencent Cloud Secret ID for Hunyuan3D",
            visible=True
        )
        tencent_secret_key = gr.Textbox(
            label="Tencent Cloud Secret Key",
            value=get_env("TENCENT_SECRET_KEY"),
            type="password",
            info="Your Tencent Cloud Secret Key for Hunyuan3D",
            visible=True
        )
        
    with gr.Row():
        trellis2_api_base = gr.Textbox(
            label="TRELLIS2 API Base",
            value=get_env("TRELLIS2_API_BASE", "http://127.0.0.1:7862/openapi/v1"),
            info="Local TRELLIS2 Meshy-like API base URL",
            visible=True
        )
        trellis2_max_concurrent = gr.Number(
            label="TRELLIS2 Max Concurrent",
            value=get_int_env("TRELLIS2_MAX_CONCURRENT", 1),
            precision=0,
            info="Maximum concurrent TRELLIS2 generations (default: 1)",
            visible=True
        )
        trellis2_texture_size = gr.Number(
            label="TRELLIS2 Texture Size",
            value=get_int_env("TRELLIS2_TEXTURE_SIZE", 4096),
            precision=0,
            info="Texture size for TRELLIS2 GLB export (default: 4096)",
            visible=True
        )

    with gr.Row():
        trellis2_resolution = gr.Dropdown(
            label="TRELLIS2 Resolution",
            choices=["512", "1024", "1536"],
            value=_choice_env("TRELLIS2_RESOLUTION", "1024", ["512", "1024", "1536"]),
            info="TRELLIS2 generation resolution",
            visible=True
        )
        trellis2_decimation_target = gr.Number(
            label="TRELLIS2 Decimation Target",
            value=get_int_env("TRELLIS2_DECIMATION_TARGET", 1000000),
            precision=0,
            info="Target face count for TRELLIS2 GLB export",
            visible=True
        )

    with gr.Row():
        ai_platform = gr.Dropdown(
            label="AI Platform",
            choices=["Hunyuan3D", "Meshy", "TRELLIS2"],
            value=_choice_env("STORYBLENDER_AI_PLATFORM", "TRELLIS2", ["Hunyuan3D", "Meshy", "TRELLIS2"]),
            info="AI platform for 3D model generation",
            visible=True
        )
    
    with gr.Row():
        uthana_api_key = gr.Textbox(
            label="Uthana API Key",
            value=get_env("UTHANA_API_KEY"),
            type="password",
            info="Your Uthana API key for text-to-motion animation (alternative to Meshy)",
            visible=True
        )
        uthana_fps = gr.Number(
            label="Uthana FPS",
            value=get_int_env("UTHANA_FPS", 24),
            precision=0,
            info="Frames per second for Uthana animations (24, 30, or 60)",
            visible=True
        )
    
    md_sep3 = gr.Markdown("---")
    
    md_project = gr.Markdown("### Project Configuration")
    
    with gr.Row():
        project_dir = gr.Textbox(
            label="Project Absolute Directory",
            value=get_env("STORYBLENDER_PROJECT_DIR"),
            placeholder="/Users/username/projects/my_project",
            info="⚠️ Must be an absolute path to a directory where the generated files will be saved",
            visible=True
        )
    
    with gr.Row():
        save_config_btn = gr.Button("💾 Save Configuration", variant="secondary")
        edit_config_btn = gr.Button("⚙️ Edit Configuration", variant="secondary", visible=False)
    config_warning = gr.Markdown("", visible=False)

    with gr.Group():
        gr.Markdown("### Update Manual Adjustments")
        gr.Markdown(
            "Use this after you have finished the automated pipeline through Step 9 "
            "and want to **hand-edit the open Blender file** (move, rotate, scale, "
            "resize, add, or delete assets in base or shot scenes) without "
            "re-running earlier steps.\n\n"
            "StoryBlender keeps layout and animation state in versioned JSON "
            "(`layout_script`, `supplementary_layout_script`, and "
            "`animated_models`). This section **reads the current Blender scene "
            "transforms** (world-space location, XYZ Euler rotation, scale, and "
            "axis-aligned dimensions) and proposes matching JSON updates. "
            "Canonical `asset_sheet` width/depth/height values are left unchanged; "
            "only scene/shot layout entries are updated.\n\n"
            "**Workflow**\n"
            "1. Open `step_9.blend` (recommended) and make your edits in Blender.\n"
            "2. Set comparison tolerances so tiny float noise is ignored. Optionally "
            "enable **Check Character Animations** to sync exact-name character "
            "roots from base scenes into `animated_models` (shot-scene character "
            "action objects are always ignored).\n"
            "3. Click **Update Manual Adjustments** to scan Blender and show a "
            "read-only preview of every detected change—nothing is written yet.\n"
            "4. Review the table (and any warnings). Click **Proceed and Save** to "
            "write new JSON versions and then save the open `.blend` file, or "
            "**Cancel** to discard the preview.\n\n"
            "If Blender, the source JSON, or the tolerance options change between "
            "preview and confirm, the save is rejected and you must scan again."
        )
        with gr.Row():
            manual_linear_tolerance = gr.Number(
                label="Linear Tolerance",
                value=0.01,
                minimum=0,
                info="Tolerance for location, scale, and dimensions.",
            )
            manual_rotation_tolerance = gr.Number(
                label="Rotation Tolerance Degrees",
                value=0.1,
                minimum=0,
                info="Angular tolerance in degrees.",
            )
            check_character_animations = gr.Checkbox(
                label="Check Character Animations",
                value=False,
                info=(
                    "Update exact-name character roots in animated_models from "
                    "base scenes; character objects in shot scenes are always ignored."
                ),
            )
        with gr.Row():
            animated_character_location_tolerance = gr.Number(
                label="Animated Character Linear Tolerance (Location)",
                value=0.3,
                minimum=0,
                info=(
                    "Used for character locations only when Check Character "
                    "Animations is enabled."
                ),
            )
            animated_character_dimensions_tolerance = gr.Number(
                label="Animated Character Linear Tolerance (Dimensions)",
                value=0.2,
                minimum=0,
                info=(
                    "Used for character dimensions only when Check Character "
                    "Animations is enabled."
                ),
            )
        update_manual_adjustments_btn = gr.Button(
            "🔄 Update Manual Adjustments", variant="primary"
        )
        manual_adjustments_status = gr.Markdown("", visible=False)
        manual_adjustments_preview = gr.Dataframe(
            headers=[
                "#",
                "File",
                "Scope",
                "Scene",
                "Shot",
                "Asset",
                "Change",
                "Changes",
            ],
            value=[],
            interactive=False,
            wrap=True,
            visible=False,
        )
        manual_adjustments_warnings = gr.Markdown("", visible=False)
        pending_manual_adjustments = gr.State(value=None)
        with gr.Row():
            proceed_manual_adjustments_btn = gr.Button(
                "✅ Proceed and Save", variant="primary", visible=False
            )
            cancel_manual_adjustments_btn = gr.Button(
                "Cancel", variant="secondary", visible=False
            )
    
    return {
        "image_gen_platform": image_gen_platform,
        "gemini_image_model": gemini_image_model,
        "gemini_api_key": gemini_api_key,
        "gemini_api_base": gemini_api_base,
        "openai_api_key": openai_api_key,
        "openai_api_base": openai_api_base,
        "openai_image_model": openai_image_model,
        "reasoning_model": reasoning_model,
        "vision_model": vision_model,
        "anyllm_api_key": anyllm_api_key,
        "anyllm_api_base": anyllm_api_base,
        "anyllm_provider": anyllm_provider,
        "sketchfab_api_key": sketchfab_api_key,
        "meshy_api_key": meshy_api_key,
        "meshy_model": meshy_model,
        "uthana_api_key": uthana_api_key,
        "uthana_fps": uthana_fps,
        "tencent_secret_id": tencent_secret_id,
        "tencent_secret_key": tencent_secret_key,
        "trellis2_api_base": trellis2_api_base,
        "trellis2_max_concurrent": trellis2_max_concurrent,
        "trellis2_texture_size": trellis2_texture_size,
        "trellis2_resolution": trellis2_resolution,
        "trellis2_decimation_target": trellis2_decimation_target,
        "ai_platform": ai_platform,
        "project_dir": project_dir,
        "save_config_btn": save_config_btn,
        "edit_config_btn": edit_config_btn,
        "config_warning": config_warning,
        "manual_linear_tolerance": manual_linear_tolerance,
        "manual_rotation_tolerance": manual_rotation_tolerance,
        "check_character_animations": check_character_animations,
        "animated_character_location_tolerance": (
            animated_character_location_tolerance
        ),
        "animated_character_dimensions_tolerance": (
            animated_character_dimensions_tolerance
        ),
        "update_manual_adjustments_btn": update_manual_adjustments_btn,
        "manual_adjustments_status": manual_adjustments_status,
        "manual_adjustments_preview": manual_adjustments_preview,
        "manual_adjustments_warnings": manual_adjustments_warnings,
        "pending_manual_adjustments": pending_manual_adjustments,
        "proceed_manual_adjustments_btn": proceed_manual_adjustments_btn,
        "cancel_manual_adjustments_btn": cancel_manual_adjustments_btn,
        "md_title": md_title,
        "md_image_gen": md_image_gen,
        "md_sep1": md_sep1,
        "md_reasoning": md_reasoning,
        "md_sep2": md_sep2,
        "md_3d_services": md_3d_services,
        "md_sep3": md_sep3,
        "md_project": md_project,
    }


def setup_config_handlers(config_components, blender_client=None):
    """Setup click handlers for configuration buttons.
    
    Args:
        config_components: Dictionary of config components from create_config_ui()
    """
    image_gen_platform = config_components["image_gen_platform"]
    gemini_image_model = config_components["gemini_image_model"]
    gemini_api_key = config_components["gemini_api_key"]
    gemini_api_base = config_components["gemini_api_base"]
    openai_api_key = config_components["openai_api_key"]
    openai_api_base = config_components["openai_api_base"]
    openai_image_model = config_components["openai_image_model"]
    reasoning_model = config_components["reasoning_model"]
    vision_model = config_components["vision_model"]
    anyllm_api_key = config_components["anyllm_api_key"]
    anyllm_api_base = config_components["anyllm_api_base"]
    anyllm_provider = config_components["anyllm_provider"]
    sketchfab_api_key = config_components["sketchfab_api_key"]
    meshy_api_key = config_components["meshy_api_key"]
    meshy_model = config_components["meshy_model"]
    uthana_api_key = config_components["uthana_api_key"]
    uthana_fps = config_components["uthana_fps"]
    tencent_secret_id = config_components["tencent_secret_id"]
    tencent_secret_key = config_components["tencent_secret_key"]
    trellis2_api_base = config_components["trellis2_api_base"]
    trellis2_max_concurrent = config_components["trellis2_max_concurrent"]
    trellis2_texture_size = config_components["trellis2_texture_size"]
    trellis2_resolution = config_components["trellis2_resolution"]
    trellis2_decimation_target = config_components["trellis2_decimation_target"]
    ai_platform = config_components["ai_platform"]
    project_dir = config_components["project_dir"]
    save_config_btn = config_components["save_config_btn"]
    edit_config_btn = config_components["edit_config_btn"]
    config_warning = config_components["config_warning"]
    manual_linear_tolerance = config_components["manual_linear_tolerance"]
    manual_rotation_tolerance = config_components["manual_rotation_tolerance"]
    check_character_animations = config_components["check_character_animations"]
    animated_character_location_tolerance = config_components[
        "animated_character_location_tolerance"
    ]
    animated_character_dimensions_tolerance = config_components[
        "animated_character_dimensions_tolerance"
    ]
    update_manual_adjustments_btn = config_components["update_manual_adjustments_btn"]
    manual_adjustments_status = config_components["manual_adjustments_status"]
    manual_adjustments_preview = config_components["manual_adjustments_preview"]
    manual_adjustments_warnings = config_components["manual_adjustments_warnings"]
    pending_manual_adjustments = config_components["pending_manual_adjustments"]
    proceed_manual_adjustments_btn = config_components["proceed_manual_adjustments_btn"]
    cancel_manual_adjustments_btn = config_components["cancel_manual_adjustments_btn"]
    md_title = config_components["md_title"]
    md_image_gen = config_components["md_image_gen"]
    md_sep1 = config_components["md_sep1"]
    md_reasoning = config_components["md_reasoning"]
    md_sep2 = config_components["md_sep2"]
    md_3d_services = config_components["md_3d_services"]
    md_sep3 = config_components["md_sep3"]
    md_project = config_components["md_project"]
    
    def validate_and_save(project_dir_value):
        """Validate project_dir and save configuration if valid."""
        # Strip wrapping single quotes (macOS), backticks, or double quotes (Windows) from path
        if project_dir_value:
            project_dir_value = project_dir_value.strip()
            if (project_dir_value.startswith("'") and project_dir_value.endswith("'")) or \
               (project_dir_value.startswith('`') and project_dir_value.endswith('`')) or \
               (project_dir_value.startswith('"') and project_dir_value.endswith('"')):
                project_dir_value = project_dir_value[1:-1]
        # Check if project_dir is empty or not an absolute path
        if not project_dir_value or not project_dir_value.strip():
            return (
                gr.update(),  # image_gen_platform
                gr.update(),  # gemini_image_model
                gr.update(),  # reasoning_model
                gr.update(),  # vision_model
                gr.update(),  # gemini_api_key
                gr.update(),  # gemini_api_base
                gr.update(),  # openai_api_key
                gr.update(),  # openai_api_base
                gr.update(),  # openai_image_model
                gr.update(),  # anyllm_api_key
                gr.update(),  # anyllm_api_base
                gr.update(),  # anyllm_provider
                gr.update(),  # sketchfab_api_key
                gr.update(),  # meshy_api_key
                gr.update(),  # meshy_model
                gr.update(),  # uthana_api_key
                gr.update(),  # uthana_fps
                gr.update(),  # tencent_secret_id
                gr.update(),  # tencent_secret_key
                gr.update(),  # trellis2_api_base
                gr.update(),  # trellis2_max_concurrent
                gr.update(),  # trellis2_texture_size
                gr.update(),  # trellis2_resolution
                gr.update(),  # trellis2_decimation_target
                gr.update(),  # ai_platform
                gr.update(),  # project_dir
                gr.update(),  # save_config_btn
                gr.update(),  # edit_config_btn
                gr.update(value="⚠️ **Warning:** Project Directory cannot be empty. Please provide a valid absolute path.", visible=True),  # config_warning
                gr.update(),  # md_title
                gr.update(),  # md_image_gen
                gr.update(),  # md_sep1
                gr.update(),  # md_reasoning
                gr.update(),  # md_sep2
                gr.update(),  # md_3d_services
                gr.update(),  # md_sep3
                gr.update(),  # md_project
            )
        
        if not os.path.isabs(project_dir_value.strip()):
            return (
                gr.update(),  # image_gen_platform
                gr.update(),  # gemini_image_model
                gr.update(),  # reasoning_model
                gr.update(),  # vision_model
                gr.update(),  # gemini_api_key
                gr.update(),  # gemini_api_base
                gr.update(),  # openai_api_key
                gr.update(),  # openai_api_base
                gr.update(),  # openai_image_model
                gr.update(),  # anyllm_api_key
                gr.update(),  # anyllm_api_base
                gr.update(),  # anyllm_provider
                gr.update(),  # sketchfab_api_key
                gr.update(),  # meshy_api_key
                gr.update(),  # meshy_model
                gr.update(),  # uthana_api_key
                gr.update(),  # uthana_fps
                gr.update(),  # tencent_secret_id
                gr.update(),  # tencent_secret_key
                gr.update(),  # trellis2_api_base
                gr.update(),  # trellis2_max_concurrent
                gr.update(),  # trellis2_texture_size
                gr.update(),  # trellis2_resolution
                gr.update(),  # trellis2_decimation_target
                gr.update(),  # ai_platform
                gr.update(),  # project_dir
                gr.update(),  # save_config_btn
                gr.update(),  # edit_config_btn
                gr.update(value=f"⚠️ **Warning:** '{project_dir_value}' is not an absolute path. Please provide a path starting with '/'.", visible=True),  # config_warning
                gr.update(),  # md_title
                gr.update(),  # md_image_gen
                gr.update(),  # md_sep1
                gr.update(),  # md_reasoning
                gr.update(),  # md_sep2
                gr.update(),  # md_3d_services
                gr.update(),  # md_sep3
                gr.update(),  # md_project
            )
        
        # Add project_dir to Gradio's static paths so files can be served
        # This allows the app to serve files from the working directory after launch
        project_path = project_dir_value.strip()
        gr.set_static_paths(paths=[project_path])
        
        # Validation passed, proceed with saving
        return (
            gr.update(visible=False),  # image_gen_platform
            gr.update(visible=False),  # gemini_image_model
            gr.update(visible=False),  # reasoning_model
            gr.update(visible=False),  # vision_model
            gr.update(visible=False),  # gemini_api_key
            gr.update(visible=False),  # gemini_api_base
            gr.update(visible=False),  # openai_api_key
            gr.update(visible=False),  # openai_api_base
            gr.update(visible=False),  # openai_image_model
            gr.update(visible=False),  # anyllm_api_key
            gr.update(visible=False),  # anyllm_api_base
            gr.update(visible=False),  # anyllm_provider
            gr.update(visible=False),  # sketchfab_api_key
            gr.update(visible=False),  # meshy_api_key
            gr.update(visible=False),  # meshy_model
            gr.update(visible=False),  # uthana_api_key
            gr.update(visible=False),  # uthana_fps
            gr.update(visible=False),  # tencent_secret_id
            gr.update(visible=False),  # tencent_secret_key
            gr.update(visible=False),  # trellis2_api_base
            gr.update(visible=False),  # trellis2_max_concurrent
            gr.update(visible=False),  # trellis2_texture_size
            gr.update(visible=False),  # trellis2_resolution
            gr.update(visible=False),  # trellis2_decimation_target
            gr.update(visible=False),  # ai_platform
            gr.update(value=project_path, visible=False),  # project_dir
            gr.update(visible=False),  # save_config_btn
            gr.update(visible=True),   # edit_config_btn
            gr.update(visible=False),  # config_warning - hide warning on success
            gr.update(visible=False),  # md_title
            gr.update(visible=False),  # md_image_gen
            gr.update(visible=False),  # md_sep1
            gr.update(visible=False),  # md_reasoning
            gr.update(visible=False),  # md_sep2
            gr.update(visible=False),  # md_3d_services
            gr.update(visible=False),  # md_sep3
            gr.update(visible=False),  # md_project
        )
    
    # Save Configuration button click handler
    save_config_btn.click(
        fn=validate_and_save,
        inputs=[project_dir],
        outputs=[
            image_gen_platform,
            gemini_image_model, reasoning_model, vision_model,
            gemini_api_key, gemini_api_base,
            openai_api_key, openai_api_base, openai_image_model,
            anyllm_api_key, anyllm_api_base, anyllm_provider,
            sketchfab_api_key, meshy_api_key, meshy_model,
            uthana_api_key, uthana_fps,
            tencent_secret_id, tencent_secret_key,
            trellis2_api_base, trellis2_max_concurrent, trellis2_texture_size,
            trellis2_resolution, trellis2_decimation_target,
            ai_platform,
            project_dir, save_config_btn, edit_config_btn, config_warning,
            md_title, md_image_gen, md_sep1, md_reasoning,
            md_sep2, md_3d_services, md_sep3, md_project
        ],
        concurrency_limit=None,
        show_progress="hidden",
    )
    
    # Edit Configuration button click handler
    edit_config_btn.click(
        fn=lambda: (
            gr.update(visible=True),   # image_gen_platform
            gr.update(visible=True),   # gemini_image_model
            gr.update(visible=True),   # reasoning_model
            gr.update(visible=True),   # vision_model
            gr.update(visible=True),   # gemini_api_key
            gr.update(visible=True),   # gemini_api_base
            gr.update(visible=True),   # openai_api_key
            gr.update(visible=True),   # openai_api_base
            gr.update(visible=True),   # openai_image_model
            gr.update(visible=True),   # anyllm_api_key
            gr.update(visible=True),   # anyllm_api_base
            gr.update(visible=True),   # anyllm_provider
            gr.update(visible=True),   # sketchfab_api_key
            gr.update(visible=True),   # meshy_api_key
            gr.update(visible=True),   # meshy_model
            gr.update(visible=True),   # uthana_api_key
            gr.update(visible=True),   # uthana_fps
            gr.update(visible=True),   # tencent_secret_id
            gr.update(visible=True),   # tencent_secret_key
            gr.update(visible=True),   # trellis2_api_base
            gr.update(visible=True),   # trellis2_max_concurrent
            gr.update(visible=True),   # trellis2_texture_size
            gr.update(visible=True),   # trellis2_resolution
            gr.update(visible=True),   # trellis2_decimation_target
            gr.update(visible=True),   # ai_platform
            gr.update(visible=True),   # project_dir
            gr.update(visible=True),   # save_config_btn
            gr.update(visible=False),  # edit_config_btn
            gr.update(visible=False),  # config_warning - hide warning when editing
            gr.update(visible=True),   # md_title
            gr.update(visible=True),   # md_image_gen
            gr.update(visible=True),   # md_sep1
            gr.update(visible=True),   # md_reasoning
            gr.update(visible=True),   # md_sep2
            gr.update(visible=True),   # md_3d_services
            gr.update(visible=True),   # md_sep3
            gr.update(visible=True),   # md_project
        ),
        inputs=[],
        outputs=[
            image_gen_platform,
            gemini_image_model, reasoning_model, vision_model,
            gemini_api_key, gemini_api_base,
            openai_api_key, openai_api_base, openai_image_model,
            anyllm_api_key, anyllm_api_base, anyllm_provider,
            sketchfab_api_key, meshy_api_key, meshy_model,
            uthana_api_key, uthana_fps,
            tencent_secret_id, tencent_secret_key,
            trellis2_api_base, trellis2_max_concurrent, trellis2_texture_size,
            trellis2_resolution, trellis2_decimation_target,
            ai_platform,
            project_dir, save_config_btn, edit_config_btn, config_warning,
            md_title, md_image_gen, md_sep1, md_reasoning,
            md_sep2, md_3d_services, md_sep3, md_project
        ],
        concurrency_limit=None,
        show_progress="hidden",
    )

    def scan_manual_adjustments(
        project_dir_value,
        linear_tolerance,
        rotation_tolerance_degrees,
        check_characters,
        character_location_tolerance,
        character_dimensions_tolerance,
    ):
        if blender_client is None:
            return (
                gr.update(value="⚠️ Blender client is unavailable.", visible=True),
                gr.update(visible=False),
                gr.update(value="", visible=False),
                None,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
            )
        plan = prepare_manual_adjustments(
            project_dir_value,
            blender_client,
            linear_tolerance=linear_tolerance,
            rotation_tolerance_degrees=rotation_tolerance_degrees,
            check_character_animations=check_characters,
            animated_character_location_tolerance=character_location_tolerance,
            animated_character_dimensions_tolerance=character_dimensions_tolerance,
        )
        if not plan.get("success"):
            return (
                gr.update(value=f"⚠️ {plan.get('error', 'Scan failed.')}", visible=True),
                gr.update(visible=False),
                gr.update(value="", visible=False),
                None,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
            )

        count = plan.get("summary", {}).get("total_changes", 0)
        warning_count = len(plan.get("warnings", []))
        if count:
            message = (
                f"🔎 **Found {count} manual adjustment(s).** Review every item below, "
                "then click **Proceed and Save**. No files have been written yet."
            )
            if warning_count:
                message += f" The preview also contains {warning_count} warning(s)."
        else:
            message = (
                "✅ **No meaningful manual adjustments found.** No files were written."
            )
            if warning_count:
                message += f" Review the {warning_count} warning(s) below."
        return (
            gr.update(value=message, visible=True),
            gr.update(value=plan["preview_rows"], visible=bool(count)),
            gr.update(
                value=plan["preview_warnings"],
                visible=bool(warning_count),
            ),
            compact_pending_plan(plan),
            gr.update(visible=bool(count)),
            gr.update(visible=True),
            gr.update(visible=True),
        )

    def proceed_manual_adjustments(
        project_dir_value,
        pending_plan,
        linear_tolerance,
        rotation_tolerance_degrees,
        check_characters,
        character_location_tolerance,
        character_dimensions_tolerance,
    ):
        if blender_client is None:
            return (
                gr.update(value="⚠️ Blender client is unavailable.", visible=True),
                gr.update(),
                gr.update(),
                pending_plan,
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        result = commit_manual_adjustments(
            project_dir_value,
            pending_plan,
            blender_client,
            linear_tolerance=linear_tolerance,
            rotation_tolerance_degrees=rotation_tolerance_degrees,
            check_character_animations=check_characters,
            animated_character_location_tolerance=character_location_tolerance,
            animated_character_dimensions_tolerance=character_dimensions_tolerance,
        )
        if result.get("stale"):
            fresh_plan = result.get("fresh_plan")
            return (
                gr.update(value=f"⚠️ {result['error']}", visible=True),
                gr.update(
                    value=format_preview_rows(fresh_plan),
                    visible=bool(fresh_plan and fresh_plan.get("changes")),
                ) if fresh_plan else gr.update(),
                gr.update(
                    value=format_preview_warnings(fresh_plan),
                    visible=bool(fresh_plan and fresh_plan.get("warnings")),
                ) if fresh_plan else gr.update(),
                compact_pending_plan(fresh_plan) if fresh_plan else pending_plan,
                gr.update(visible=bool(fresh_plan and fresh_plan.get("changes"))),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        if not result.get("success"):
            saved_paths = result.get("output_paths", [])
            suffix = ""
            if saved_paths:
                suffix = "\n\nSaved JSON files:\n" + "\n".join(
                    f"- `{path}`" for path in saved_paths
                )
            return (
                gr.update(
                    value=f"⚠️ {result.get('error', 'Update failed.')}{suffix}",
                    visible=True,
                ),
                gr.update(),
                gr.update(),
                None if result.get("json_saved") else pending_plan,
                gr.update(visible=not result.get("json_saved", False)),
                gr.update(visible=not result.get("json_saved", False)),
                gr.update(visible=True),
            )

        paths = "\n".join(f"- `{path}`" for path in result.get("output_paths", []))
        warning_text = ""
        if result.get("warnings"):
            warning_text = f"\n\nCompleted with {len(result['warnings'])} preview warning(s)."
        message = (
            f"✅ **Manual adjustments saved successfully.**\n\n"
            f"New JSON versions:\n{paths}\n\n"
            f"Blend file saved last: `{result.get('blend_filepath')}`{warning_text}"
        )
        return (
            gr.update(value=message, visible=True),
            gr.update(),
            gr.update(),
            None,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    def cancel_manual_adjustments():
        return (
            gr.update(value="Manual adjustment update cancelled. No files were written.", visible=True),
            gr.update(value=[], visible=False),
            gr.update(value="", visible=False),
            None,
            gr.update(visible=False),
            gr.update(visible=False),
        )

    update_manual_adjustments_btn.click(
        fn=scan_manual_adjustments,
        inputs=[
            project_dir,
            manual_linear_tolerance,
            manual_rotation_tolerance,
            check_character_animations,
            animated_character_location_tolerance,
            animated_character_dimensions_tolerance,
        ],
        outputs=[
            manual_adjustments_status,
            manual_adjustments_preview,
            manual_adjustments_warnings,
            pending_manual_adjustments,
            proceed_manual_adjustments_btn,
            cancel_manual_adjustments_btn,
            update_manual_adjustments_btn,
        ],
        concurrency_limit=1,
    )

    proceed_manual_adjustments_btn.click(
        fn=proceed_manual_adjustments,
        inputs=[
            project_dir,
            pending_manual_adjustments,
            manual_linear_tolerance,
            manual_rotation_tolerance,
            check_character_animations,
            animated_character_location_tolerance,
            animated_character_dimensions_tolerance,
        ],
        outputs=[
            manual_adjustments_status,
            manual_adjustments_preview,
            manual_adjustments_warnings,
            pending_manual_adjustments,
            proceed_manual_adjustments_btn,
            cancel_manual_adjustments_btn,
            update_manual_adjustments_btn,
        ],
        concurrency_limit=1,
    )

    cancel_manual_adjustments_btn.click(
        fn=cancel_manual_adjustments,
        inputs=[],
        outputs=[
            manual_adjustments_status,
            manual_adjustments_preview,
            manual_adjustments_warnings,
            pending_manual_adjustments,
            proceed_manual_adjustments_btn,
            cancel_manual_adjustments_btn,
        ],
        queue=False,
        concurrency_limit=None,
        show_progress="hidden",
    )
