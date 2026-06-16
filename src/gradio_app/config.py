"""Configuration UI components and handlers for StoryBlender."""

import os
import gradio as gr

try:
    from ..env_config import get_env, get_int_env
except ImportError:
    from env_config import get_env, get_int_env


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
    
    save_config_btn = gr.Button("💾 Save Configuration", variant="secondary")
    edit_config_btn = gr.Button("⚙️ Edit Configuration", variant="secondary", visible=False)
    config_warning = gr.Markdown("", visible=False)
    
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
        "md_title": md_title,
        "md_image_gen": md_image_gen,
        "md_sep1": md_sep1,
        "md_reasoning": md_reasoning,
        "md_sep2": md_sep2,
        "md_3d_services": md_3d_services,
        "md_sep3": md_sep3,
        "md_project": md_project,
    }


def setup_config_handlers(config_components):
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
