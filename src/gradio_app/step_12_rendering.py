import os
import json
import logging
import shlex
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
    Generate a Python script that renders all cameras in a single Blender session.
    
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
    
    # Generate the Python script
    engine_code = "'CYCLES'" if render_engine == "Cycles" else "'BLENDER_EEVEE'"
    
    # Calculate total frames for ETA
    total_frames = sum(job['end_frame'] - job['start_frame'] + 1 for job in render_jobs)
    
    script_lines = [
        "import bpy",
        "import os",
        "import time",
        "",
        f"# Auto-generated render script from: {result_info}",
        f"# Total render jobs: {len(render_jobs)}",
        f"# Total frames to render: {total_frames}",
        "",
        "# Render configuration",
        f"RENDER_ENGINE = {engine_code}",
        f"RENDER_SAMPLES = {render_samples if render_samples else 'None'}",
        f"PERSISTENT_DATA = {persistent_data}",
        "",
        "# Render jobs: (scene_name, camera_name, start_frame, end_frame, output_path)",
        "RENDER_JOBS = [",
    ]
    
    for job in render_jobs:
        output_path_for_py = job["output_path"].replace("\\", "/")
        script_lines.append(
            f"    ({json.dumps(job['scene_name'])}, {json.dumps(job['camera_name'])}, {job['start_frame']}, {job['end_frame']}, {json.dumps(output_path_for_py)}),"
        )
    
    script_lines.extend([
        "]",
        "",
        "def format_time(seconds):",
        "    \"\"\"Format seconds into human-readable time string.\"\"\"",
        "    if seconds < 60:",
        "        return f\"{seconds:.1f}s\"",
        "    elif seconds < 3600:",
        "        mins = int(seconds // 60)",
        "        secs = int(seconds % 60)",
        "        return f\"{mins}m {secs}s\"",
        "    else:",
        "        hours = int(seconds // 3600)",
        "        mins = int((seconds % 3600) // 60)",
        "        return f\"{hours}h {mins}m\"",
        "",
        "def setup_render_settings(scene):",
        "    \"\"\"Configure render settings for the scene.\"\"\"",
        "    scene.render.engine = RENDER_ENGINE",
        "    scene.render.image_settings.file_format = 'PNG'",
        "    ",
        "    if RENDER_SAMPLES is not None:",
        "        if RENDER_ENGINE == 'CYCLES':",
        "            scene.cycles.samples = RENDER_SAMPLES",
        "        else:",
        "            scene.eevee.taa_render_samples = RENDER_SAMPLES",
        "    ",
        "    scene.render.use_persistent_data = PERSISTENT_DATA and RENDER_ENGINE == 'CYCLES'",
        "",
        "def render_all():",
        "    \"\"\"Render all jobs in sequence.\"\"\"",
        "    total_jobs = len(RENDER_JOBS)",
        "    # Validate every job before spending time on any render.",
        "    for scene_name, camera_name, start, end, _ in RENDER_JOBS:",
        "        scene = bpy.data.scenes.get(scene_name)",
        "        if scene is None:",
        "            raise RuntimeError(f\"Scene '{scene_name}' not found\")",
        "        camera = scene.objects.get(camera_name)",
        "        if camera is None or camera.type != 'CAMERA':",
        "            raise RuntimeError(f\"Camera '{camera_name}' not found in '{scene_name}'\")",
        "        if end < start:",
        "            raise RuntimeError(f\"Invalid frame range for '{camera_name}': {start}–{end}\")",
        "    ",
        "    # Calculate total frames for ETA",
        "    total_frames = sum(end - start + 1 for _, _, start, end, _ in RENDER_JOBS)",
        "    frames_rendered = 0",
        "    total_render_time = 0",
        "    render_start_time = time.time()",
        "    ",
        "    for idx, (scene_name, camera_name, start_frame, end_frame, output_path) in enumerate(RENDER_JOBS, 1):",
        "        job_frames = end_frame - start_frame + 1",
        "        print(f\"\\n{'='*60}\")",
        "        print(f\"Rendering job {idx}/{total_jobs}: {scene_name} - {camera_name}\")",
        "        print(f\"Frames: {start_frame} to {end_frame} ({job_frames} frames)\")",
        "        print(f\"Output: {output_path}\")",
        "        if frames_rendered > 0:",
        "            avg_time_per_frame = total_render_time / frames_rendered",
        "            remaining_frames = total_frames - frames_rendered",
        "            eta_seconds = avg_time_per_frame * remaining_frames",
        "            print(f\"Avg: {avg_time_per_frame:.2f}s/frame | Remaining: {remaining_frames} frames | ETA: {format_time(eta_seconds)}\")",
        "        print(f\"{'='*60}\\n\")",
        "        ",
        "        # Switch to the scene",
        "        scene = bpy.data.scenes[scene_name]",
        "        bpy.context.window.scene = scene",
        "        ",
        "        # Set the camera",
        "        scene.camera = scene.objects[camera_name]",
        "        ",
        "        # Setup render settings",
        "        setup_render_settings(scene)",
        "        ",
        "        # Set output path and frame range",
        "        scene.render.filepath = output_path",
        "        scene.frame_start = start_frame",
        "        scene.frame_end = end_frame",
        "        ",
        "        # Create output directory",
        "        output_dir = os.path.dirname(output_path)",
        "        os.makedirs(output_dir, exist_ok=True)",
        "        ",
        "        # Render with timing",
        "        job_start_time = time.time()",
        "        bpy.ops.render.render(animation=True)",
        "        job_elapsed = time.time() - job_start_time",
        "        ",
        "        # Update stats",
        "        frames_rendered += job_frames",
        "        total_render_time += job_elapsed",
        "        avg_time_per_frame = total_render_time / frames_rendered",
        "        ",
        "        print(f\"\\nCompleted: {scene_name} - {camera_name}\")",
        "        print(f\"Job time: {format_time(job_elapsed)} ({job_elapsed/job_frames:.2f}s/frame)\")",
        "        print(f\"Progress: {frames_rendered}/{total_frames} frames ({100*frames_rendered/total_frames:.1f}%)\")",
        "        ",
        "        remaining_frames = total_frames - frames_rendered",
        "        if remaining_frames > 0:",
        "            eta_seconds = avg_time_per_frame * remaining_frames",
        "            print(f\"ETA for remaining {remaining_frames} frames: {format_time(eta_seconds)}\")",
        "    ",
        "    total_elapsed = time.time() - render_start_time",
        "    print(f\"\\n{'='*60}\")",
        "    print(f\"All {total_jobs} render jobs completed!\")",
        "    print(f\"Total time: {format_time(total_elapsed)}\")",
        "    print(f\"Average: {total_elapsed/total_frames:.2f}s/frame\")",
        "    print(f\"{'='*60}\")",
        "",
        "# Run the render",
        "if __name__ == '__main__':",
        "    render_all()",
    ])
    
    python_script = "\n".join(script_lines)
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
    Generate render files: a Python script for Blender and a shell script to run it.
    
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
        if not os.path.exists(shell_script_path) and not os.path.exists(python_script_path):
            break
        version += 1
    
    # Save Python script with version number
    with open(python_script_path, "w", newline='\n') as f:
        f.write(python_script)
    
    # Default output path for display
    if not output_base_path or output_base_path.strip() == "":
        output_base_path = os.path.join(project_dir, "renders")
    
    # Generate shell script that references the versioned Python script
    if target_os == "Windows":
        blend_file_path_os = blend_file_path.replace('/', '\\\\')
        python_script_path_os = python_script_path.replace('/', '\\\\')
        shell_script = f"""@echo off
REM Single-session batch render script
REM Renders {job_count} camera(s) in one Blender session
REM Output directory: {output_base_path}

echo Starting batch render ({job_count} cameras)...
echo.

"{blender_executable}" -b "{blend_file_path_os}" --python-exit-code 1 --python "{python_script_path_os}"
if errorlevel 1 exit /b %errorlevel%

echo.
echo Batch render complete.
pause
"""
    else:
        shell_script = f"""#!/bin/bash
set -e
# Single-session batch render script
# Renders {job_count} camera(s) in one Blender session
# Output directory: {output_base_path}

echo "Starting batch render ({job_count} cameras)..."
echo

{shlex.quote(blender_executable)} -b {shlex.quote(blend_file_path)} --python-exit-code 1 --python {shlex.quote(python_script_path)}

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
    gr.Markdown("Generate Blender render commands for all cameras in your camera blocking.")
    
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
