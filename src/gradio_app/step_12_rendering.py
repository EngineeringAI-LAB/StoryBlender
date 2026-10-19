import os
import json
import logging
import shlex
from pathlib import Path
import gradio as gr
from .path_utils import make_paths_absolute, make_paths_relative

logger = logging.getLogger(__name__)


def get_path_to_latest_camera_blocking(project_dir):
    """
    Get the absolute path to the latest version of camera_blocking_v{num}.json.
    
    Args:
        project_dir: Project directory path
    
    Returns:
        str: Absolute path to the latest version, or None if no file exists
    """
    camera_blocking_dir = os.path.join(project_dir, "camera_blocking")
    
    if not os.path.exists(camera_blocking_dir):
        return None
    
    latest_version = 0
    latest_path = None
    
    for filename in os.listdir(camera_blocking_dir):
        if filename.startswith("camera_blocking_v") and filename.endswith(".json"):
            try:
                version_str = filename[len("camera_blocking_v"):-5]  # Remove prefix and .json
                version = int(version_str)
                if version > latest_version:
                    latest_version = version
                    latest_path = os.path.join(camera_blocking_dir, filename)
            except ValueError:
                continue
    
    return latest_path


def load_camera_blocking_data(project_dir):
    """Load camera blocking JSON data from the latest version file."""
    latest_path = get_path_to_latest_camera_blocking(project_dir)
    if not latest_path:
        return None, "No camera_blocking_v*.json found in project."
    
    try:
        with open(latest_path, 'r') as f:
            data = json.load(f)
        try:
            data = make_paths_absolute(data, project_dir)
        except Exception as e:
            logger.warning("Step 12: path conversion failed for camera_blocking JSON: %s", e)
        return data, latest_path
    except Exception as e:
        return None, f"Error loading camera blocking file: {str(e)}"


def generate_python_render_script(
    project_dir,
    blend_file_path,
    number_of_frames,
    output_base_path,
    render_engine="Cycles",
    render_samples=None,
    persistent_data=True,
    camera_frame_cap=False
):
    """
    Generate a worker script that renders one selected camera per process.
    
    Args:
        number_of_frames: Number of frames to render per camera
        render_engine: "Cycles" or "EEVEE"
        render_samples: Number of samples (None to use file defaults)
        persistent_data: Enable persistent data for faster re-renders
        camera_frame_cap: If True, cap render_end to JSON end_frame; if False, render exactly number_of_frames
    
    Returns:
        tuple: (python_script_content, render_jobs_list, error_message)
    """
    # Load camera blocking data
    camera_data, result_info = load_camera_blocking_data(project_dir)
    if camera_data is None:
        return None, [], result_info
    
    # Default output_base_path
    if not output_base_path or output_base_path.strip() == "":
        output_base_path = os.path.join(project_dir, "renders")
    
    # Build render jobs list
    render_jobs = []
    
    shots = camera_data.get("shot_details", []) if isinstance(camera_data, dict) else camera_data
    for shot in shots:
        scene_id = shot.get("scene_id")
        shot_id = shot.get("shot_id")
        
        if scene_id is None or shot_id is None:
            continue
        
        scene_name = f"Scene_{scene_id}_Shot_{shot_id}"
        
        cameras = [shot.get("camera_instruction") or {}]
        cameras.extend(shot.get("additional_camera_instructions") or [])
        for camera in cameras:
            # Suggestions share this JSON with placed cameras. Legacy checkpoints
            # may omit the success flag but still contain the saved transform.
            if camera.get("placement_success") is False:
                continue
            if not (camera.get("placement_success") or camera.get("start_transform")):
                continue
            camera_name = camera.get("camera_name")
            json_start = camera.get("start_frame", 1)
            json_end = camera.get("end_frame", json_start)
            
            if camera_name:
                render_start = json_start
                if number_of_frames <= 1:
                    render_end = json_start
                elif camera_frame_cap:
                    render_end = min(json_start + number_of_frames - 1, json_end)
                else:
                    render_end = json_start + number_of_frames - 1
                output_path = os.path.join(output_base_path, scene_name, camera_name, "frame_####")
                
                render_jobs.append({
                    "scene_name": scene_name,
                    "camera_name": camera_name,
                    "start_frame": render_start,
                    "end_frame": render_end,
                    "output_path": output_path
                })
    
    if not render_jobs:
        return None, [], "No placed cameras found. Complete camera blocking before rendering."

    # Export standalone sources: workers need no installed StoryBlender package.
    source_dir = Path(__file__).resolve().parent
    guard = (source_dir.parent / "memory_budget.py").read_text()
    worker = (source_dir / "render_worker.py").read_text()
    job_tuples = [(job["scene_name"], job["camera_name"], job["start_frame"],
                   job["end_frame"], job["output_path"].replace("\\", "/")) for job in render_jobs]
    config = (f"\n# Generated from {result_info}\n"
              f"RENDER_ENGINE = {'CYCLES' if render_engine == 'Cycles' else 'BLENDER_EEVEE'!r}\n"
              f"RENDER_SAMPLES = {render_samples or None!r}\n"
              f"PERSISTENT_DATA = {bool(persistent_data)!r}\n"
              f"RENDER_JOBS = {job_tuples!r}\n")
    python_script = guard + config + "\n" + worker
    return python_script, render_jobs, None


def generate_all_render_files(
    project_dir,
    blend_file_path,
    number_of_frames,
    blender_executable,
    output_base_path,
    target_os="Mac/Linux",
    render_engine="Cycles",
    render_samples=None,
    persistent_data=True,
    camera_frame_cap=False
):
    """
    Generate a worker script and sequential, memory-supervised batch launcher.
    
    Args:
        number_of_frames: Number of frames to render per camera
        target_os: "Windows" or "Mac/Linux"
        render_engine: "Cycles" or "EEVEE"
        render_samples: Number of samples (None to use file defaults)
        persistent_data: Enable persistent data for faster re-renders
        camera_frame_cap: If True, cap render_end to JSON end_frame
    
    Returns:
        tuple: (shell_script_content, python_script_content, job_count, error_message)
    """
    # Validate inputs
    if not project_dir or not os.path.isabs(project_dir):
        return None, None, 0, "Project directory must be an absolute path."
    
    if not blend_file_path or blend_file_path.strip() == "":
        return None, None, 0, "Please provide a .blend file path."
    
    # Generate the Python render script
    python_script, render_jobs, error = generate_python_render_script(
        project_dir=project_dir,
        blend_file_path=blend_file_path,
        number_of_frames=number_of_frames,
        output_base_path=output_base_path,
        render_engine=render_engine,
        render_samples=render_samples,
        persistent_data=persistent_data,
        camera_frame_cap=camera_frame_cap
    )
    
    if error:
        return None, None, 0, error
    
    # Default output path for display
    if not output_base_path or output_base_path.strip() == "":
        output_base_path = os.path.join(project_dir, "renders")
    
    # Return scripts without path references - paths will be set during save
    return python_script, render_jobs, len(render_jobs), None


def generate_and_save_script(
    project_dir,
    blend_file_path,
    number_of_frames,
    blender_executable,
    output_base_path,
    target_os="Mac/Linux",
    render_engine="Cycles",
    render_samples=None,
    persistent_data=True,
    camera_frame_cap=False
):
    """Generate render scripts and save to files."""
    # Generate the Python render script
    python_script, render_jobs, job_count, error = generate_all_render_files(
        project_dir=project_dir,
        blend_file_path=blend_file_path,
        number_of_frames=int(number_of_frames),
        blender_executable=blender_executable,
        output_base_path=output_base_path,
        target_os=target_os,
        render_engine=render_engine,
        render_samples=render_samples,
        persistent_data=persistent_data,
        camera_frame_cap=camera_frame_cap
    )
    
    if error:
        return None, None, 0, error
    
    # Save the script files
    renders_dir = os.path.join(project_dir, "renders")
    os.makedirs(renders_dir, exist_ok=True)
    
    # Find next version number (shared between shell and Python scripts)
    version = 1
    while True:
        shell_ext = ".bat" if target_os == "Windows" else ".sh"
        shell_script_path = os.path.join(renders_dir, f"render_v{version}{shell_ext}")
        python_script_path = os.path.join(renders_dir, f"render_script_v{version}.py")
        runner_path = os.path.join(renders_dir, f"render_runner_v{version}.py")
        if not os.path.exists(shell_script_path) and not os.path.exists(python_script_path) and not os.path.exists(runner_path):
            break
        version += 1
    
    # Save Python script with version number
    with open(python_script_path, "w", newline='\n') as f:
        f.write(python_script)
    
    source_dir = Path(__file__).resolve().parent
    runner = ((source_dir.parent / "memory_budget.py").read_text() + "\n"
              + (source_dir / "render_batch_runner.py").read_text()
              + "\nif __name__ == '__main__':\n    main(ensure_memory_budget)\n")
    Path(runner_path).write_text(runner)

    # Default output path for display
    if not output_base_path or output_base_path.strip() == "":
        output_base_path = os.path.join(project_dir, "renders")
    
    # Generate shell script that references the versioned Python script
    if target_os == "Windows":
        blend_file_path_os = blend_file_path.replace('/', '\\\\')
        python_script_path_os = python_script_path.replace('/', '\\\\')
        shell_script = f"""@echo off
REM Supervised batch: one camera per fresh Blender worker
REM Renders {job_count} camera(s) sequentially
REM Output directory: {output_base_path}

echo Starting batch render ({job_count} cameras)...
echo.

"{blender_executable}" -b --factory-startup --python-exit-code 1 --python "{runner_path}" -- --blender "{blender_executable}" --blend "{blend_file_path_os}" --worker "{python_script_path_os}"
if errorlevel 1 exit /b %errorlevel%

echo.
echo Batch render complete.
pause
"""
    else:
        shell_script = f"""#!/bin/bash
set -e
# Supervised batch: one camera per fresh Blender worker
# Renders {job_count} camera(s) sequentially
# Output directory: {output_base_path}

echo "Starting batch render ({job_count} cameras)..."
echo

{shlex.quote(blender_executable)} -b --factory-startup --python-exit-code 1 --python {shlex.quote(runner_path)} -- --blender {shlex.quote(blender_executable)} --blend {shlex.quote(blend_file_path)} --worker {shlex.quote(python_script_path)}

echo
echo "Batch render complete."
"""
    
    # Write the shell script
    with open(shell_script_path, "w", newline='\n' if target_os != "Windows" else '\r\n') as f:
        f.write(shell_script)
    
    # Make executable on Unix systems
    if target_os != "Windows":
        try:
            os.chmod(shell_script_path, 0o755)
            os.chmod(python_script_path, 0o755)
        except:
            pass
    
    return shell_script, shell_script_path, job_count, None


def create_rendering_ui(project_dir):
    """Create the Step 12: Rendering UI section.
    
    Args:
        project_dir: Gradio component for project directory
    
    Returns:
        dict with UI components that may be needed by other parts of the app
    """
    gr.Markdown("## Step 12: Rendering")
    gr.Markdown("Generate headless render commands. Close Blender’s UI before running them; each camera uses a fresh supervised worker.")
    
    # Input fields
    with gr.Row():
        blend_file_path = gr.Textbox(
            label="Blend File Path",
            placeholder="/path/to/your/project.blend",
            info="Required: Full path to your .blend file",
        )
    
    with gr.Row():
        number_of_frames = gr.Number(
            label="Number of Frames",
            value=1,
            precision=0,
            minimum=1,
            info="Frames per camera, starting at its first frame. Enable Camera Frame Cap to stop at its saved end frame."
        )
    
    with gr.Row():
        blender_executable = gr.Textbox(
            label="Blender Executable",
            value="blender",
            info="Path to Blender executable or 'blender' if in PATH"
        )
        output_base_path = gr.Textbox(
            label="Output Path",
            value="",
            placeholder="{project_dir}/renders",
            info="Base output path for renders (leave empty to use {project_dir}/renders)"
        )
    
    with gr.Row():
        target_os = gr.Dropdown(
            label="Target OS",
            choices=["Mac/Linux", "Windows"],
            value="Windows",
            info="Select the operating system where you will run the render script"
        )
    
    with gr.Row():
        render_engine = gr.Dropdown(
            label="Render Engine",
            choices=["Cycles", "EEVEE"],
            value="Cycles",
            info="Render engine to use"
        )
        render_samples = gr.Number(
            label="Render Samples (optional)",
            value=64,
            precision=0,
            info="EEVEE default: 64, Cycles default: 64. Leave empty to use file settings."
        )
        persistent_data = gr.Checkbox(
            label="Persistent Data",
            value=True,
            info="Enable for faster re-renders (keeps data in memory)"
        )
    
    with gr.Row():
        camera_frame_cap = gr.Checkbox(
            label="Camera Frame Cap",
            value=False,
            info="If enabled, cap frames to JSON end_frame; if disabled, render exactly number_of_frames"
        )
    
    # Generate button
    generate_btn = gr.Button("🎬 Generate Render Script", variant="primary", size="lg")
    
    # Status display
    status_display = gr.Markdown(value="", visible=False)
    
    gr.Markdown("### Generated Render Script")
    gr.Markdown("*Copy and paste the script below into your terminal to render all shots.*")
    
    # Script display - Code editor with shell language
    script_display = gr.Code(
        label="Render Script",
        language="shell",
        visible=False,
        lines=20,
        interactive=False
    )
    
    # File path display
    file_path_display = gr.Markdown(value="", visible=False)
    
    # Copy button
    copy_btn = gr.Button("📋 Copy to Clipboard", variant="secondary", visible=False)
    
    def handle_generate(proj_dir, blend_path, num_frames, blender_exe, output_path, os_choice, engine, samples, persistent, frame_cap):
        """Handle generate button click."""
        # Strip wrapping backticks (macOS) or double quotes (Windows) from paths
        def _unwrap_path(p):
            if not p:
                return p
            p = p.strip()
            if (p.startswith('`') and p.endswith('`')) or \
               (p.startswith('"') and p.endswith('"')):
                p = p[1:-1]
            return p
        blend_path = _unwrap_path(blend_path)
        blender_exe = _unwrap_path(blender_exe)
        output_path = _unwrap_path(output_path)
        # Convert samples to int or None
        samples_int = int(samples) if samples and samples > 0 else None
        
        script_content, script_path, cmd_count, error = generate_and_save_script(
            project_dir=proj_dir,
            blend_file_path=blend_path,
            number_of_frames=num_frames,
            blender_executable=blender_exe,
            output_base_path=output_path,
            target_os=os_choice,
            render_engine=engine,
            render_samples=samples_int,
            persistent_data=persistent,
            camera_frame_cap=frame_cap
        )
        
        if error:
            return (
                gr.update(value=f"❌ **Error:** {error}", visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        return (
            gr.update(value=f"✅ **Generated {cmd_count} render commands.** Saved to: `{script_path}`", visible=True),
            gr.update(value=script_content, visible=True),
            gr.update(value=f"**Script saved to:** `{script_path}`", visible=True),
            gr.update(visible=True)
        )
    
    # Generate button click handler
    generate_btn.click(
        fn=handle_generate,
        inputs=[
            project_dir,
            blend_file_path,
            number_of_frames,
            blender_executable,
            output_base_path,
            target_os,
            render_engine,
            render_samples,
            persistent_data,
            camera_frame_cap
        ],
        outputs=[status_display, script_display, file_path_display, copy_btn],
    )
    
    # Copy button handler - uses JavaScript to copy content to clipboard
    copy_btn.click(
        fn=None,
        inputs=[script_display],
        outputs=None,
        js="(text) => { navigator.clipboard.writeText(text); return []; }"
    )
    
    return {
        "blend_file_path": blend_file_path,
        "number_of_frames": number_of_frames,
        "blender_executable": blender_executable,
        "output_base_path": output_base_path,
        "target_os": target_os,
        "render_engine": render_engine,
        "render_samples": render_samples,
        "persistent_data": persistent_data,
        "script_display": script_display,
    }
