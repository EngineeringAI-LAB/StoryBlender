"""Step 13: Stitch rendered frames to video using ffmpeg."""

import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
import gradio as gr


# Detect Windows platform
IS_WINDOWS = platform.system() == "Windows"


def get_ffmpeg_path():
    """Get the path to ffmpeg executable."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    # Common Windows locations
    if IS_WINDOWS:
        common_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
    # imageio-ffmpeg is already shipped with the extension on every platform.
    try:
        import imageio_ffmpeg
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except RuntimeError:
            # Blender's wheel extraction can discard Unix executable bits. Keep
            # site-packages untouched and run a user-executable temporary copy.
            binaries = list((Path(imageio_ffmpeg.__file__).parent / "binaries").glob("ffmpeg-*"))
            if len(binaries) == 1 and binaries[0].is_file():
                cache = Path(tempfile.gettempdir()) / "storyblender-ffmpeg"
                cache.mkdir(mode=0o700, parents=True, exist_ok=True)
                executable = cache / binaries[0].name
                shutil.copyfile(binaries[0], executable)
                executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
                return str(executable)
    except (ImportError, OSError):
        pass
    return "ffmpeg"  # Fall back to hoping it's in PATH


def get_renders_folder(project_dir_value):
    """Get the default renders folder path."""
    if project_dir_value and project_dir_value.strip():
        return os.path.join(project_dir_value.strip(), "renders")
    return ""


def get_videos_folder(project_dir_value):
    """Get the default videos folder path."""
    if project_dir_value and project_dir_value.strip():
        return os.path.join(project_dir_value.strip(), "videos")
    return ""


def parse_scene_shot_folder(folder_name):
    """Parse scene and shot IDs from folder name like 'Scene_1_Shot_1'.
    
    Returns:
        tuple: (scene_id, shot_id) or (None, None) if parsing fails
    """
    match = re.match(r'Scene_(\d+)_Shot_(\d+)', folder_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def parse_camera_folder(folder_name):
    """Parse camera ID from folder name like 'cam_1_s_1_s_1'.
    
    Format: cam_{camera_id}_s_{scene_id}_s_{shot_id}
    
    Returns:
        tuple: (camera_id, scene_id, shot_id) or (None, None, None) if parsing fails
    """
    match = re.match(r'cam_(\d+)_s_(\d+)_s_(\d+)', folder_name)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None


def stitch_frames_to_video(renders_folder, output_folder, fps=24):
    """Stitch rendered frames to videos using ffmpeg.
    
    Args:
        renders_folder: Path to the renders folder containing scene/shot subfolders
        output_folder: Path to save output videos
        fps: Frames per second for output videos
        
    Returns:
        dict: Result with success/error and details
    """
    if not renders_folder or not renders_folder.strip():
        return {"error": "⚠️ Renders folder path is required."}
    
    if not output_folder or not output_folder.strip():
        return {"error": "⚠️ Output folder path is required."}
    
    renders_folder = renders_folder.strip()
    if (renders_folder.startswith('`') and renders_folder.endswith('`')) or \
       (renders_folder.startswith('"') and renders_folder.endswith('"')):
        renders_folder = renders_folder[1:-1]
    output_folder = output_folder.strip()
    if (output_folder.startswith('`') and output_folder.endswith('`')) or \
       (output_folder.startswith('"') and output_folder.endswith('"')):
        output_folder = output_folder[1:-1]
    
    if not os.path.exists(renders_folder):
        return {"error": f"⚠️ Renders folder does not exist: {renders_folder}"}
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Collect all video paths for final concatenation
    video_paths = []
    created_videos = []
    errors = []
    
    # Scan renders folder for scene/shot folders
    scene_shot_folders = sorted([
        f for f in os.listdir(renders_folder)
        if os.path.isdir(os.path.join(renders_folder, f)) and f.startswith("Scene_")
    ])
    
    if not scene_shot_folders:
        return {"error": f"⚠️ No scene folders found in: {renders_folder}"}

    ffmpeg_path = get_ffmpeg_path()
    
    for scene_shot_folder in scene_shot_folders:
        scene_shot_path = os.path.join(renders_folder, scene_shot_folder)
        scene_id, shot_id = parse_scene_shot_folder(scene_shot_folder)
        
        if scene_id is None:
            errors.append(f"Could not parse scene/shot from: {scene_shot_folder}")
            continue
        
        # Scan for camera folders
        camera_folders = sorted([
            f for f in os.listdir(scene_shot_path)
            if os.path.isdir(os.path.join(scene_shot_path, f)) and f.startswith("cam_")
        ])
        
        if not camera_folders:
            errors.append(f"No camera folders found in: {scene_shot_folder}")
            continue
        
        for camera_folder in camera_folders:
            camera_path = os.path.join(scene_shot_path, camera_folder)
            cam_id, cam_scene_id, cam_shot_id = parse_camera_folder(camera_folder)
            
            if cam_id is None:
                errors.append(f"Could not parse camera info from: {camera_folder}")
                continue
            
            # Check for frame files
            frame_files = sorted([
                f for f in os.listdir(camera_path)
                if f.startswith("frame_") and f.endswith(".png")
            ])
            
            if not frame_files:
                errors.append(f"No frame files found in: {camera_folder}")
                continue
            
            # Determine frame pattern
            # Frames are named like frame_0001.png, frame_0002.png, etc.
            # ffmpeg pattern: frame_%04d.png
            frame_pattern = os.path.join(camera_path, "frame_%04d.png")
            
            # Get the starting frame number
            first_frame = frame_files[0]
            start_match = re.match(r'frame_(\d+)\.png', first_frame)
            start_number = int(start_match.group(1)) if start_match else 1
            
            # Output video filename: scene_{scene_id}_shot_{shot_id}_cam_{cam_id}.mp4
            output_filename = f"scene_{scene_id}_shot_{shot_id}_cam_{cam_id}.mp4"
            output_path = os.path.join(output_folder, output_filename)
            
            # Build ffmpeg command
            ffmpeg_cmd = [
                ffmpeg_path,
                "-y",  # Overwrite output file if exists
                "-framerate", str(fps),
                "-start_number", str(start_number),
                "-i", frame_pattern,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",  # High quality
                output_path
            ]
            
            try:
                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout per video
                    shell=IS_WINDOWS  # Use shell on Windows for proper path handling
                )
                
                if result.returncode != 0:
                    errors.append(f"ffmpeg error for {output_filename}: {result.stderr[:200]}")
                else:
                    created_videos.append(output_filename)
                    video_paths.append(output_path)
                    
            except subprocess.TimeoutExpired:
                errors.append(f"Timeout creating video: {output_filename}")
            except Exception as e:
                errors.append(f"Error creating {output_filename}: {str(e)}")
    
    # Create concatenated video if we have multiple videos
    all_scenes_path = None
    if len(video_paths) > 0:
        all_scenes_path = os.path.join(output_folder, "all_scenes.mp4")
        
        # Create a temporary file list for ffmpeg concat
        concat_list_path = os.path.join(output_folder, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for video_path in video_paths:
                # Use relative paths for the concat file
                f.write(f"file '{os.path.basename(video_path)}'\n")
        
        # Concatenate all videos
        concat_cmd = [
            ffmpeg_path,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            all_scenes_path
        ]
        
        try:
            result = subprocess.run(
                concat_cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for concatenation
                cwd=output_folder,  # Run in output folder for relative paths
                shell=IS_WINDOWS  # Use shell on Windows for proper path handling
            )
            
            if result.returncode != 0:
                errors.append(f"ffmpeg concat error: {result.stderr[:200]}")
                all_scenes_path = None
            else:
                created_videos.append("all_scenes.mp4")
                
        except subprocess.TimeoutExpired:
            errors.append("Timeout creating all_scenes.mp4")
            all_scenes_path = None
        except Exception as e:
            errors.append(f"Error creating all_scenes.mp4: {str(e)}")
            all_scenes_path = None
        finally:
            # Clean up concat list file
            if os.path.exists(concat_list_path):
                os.remove(concat_list_path)
    
    # Build result
    if created_videos:
        return {
            "success": True,
            "created_videos": created_videos,
            "output_folder": output_folder,
            "all_scenes_path": all_scenes_path,
            "errors": errors if errors else None
        }
    else:
        return {
            "error": "⚠️ No videos were created.",
            "details": errors
        }


def show_loading_and_stitch(renders_folder, output_folder, fps):
    """Show loading state while stitching frames."""
    # Loading state
    loading_state = (
        gr.update(value="🔄 **Stitching frames to video...** This may take a while.", visible=True),
        gr.update(visible=False),  # Hide button
    )
    
    yield loading_state
    
    # Stitch frames
    result = stitch_frames_to_video(renders_folder, output_folder, int(fps))
    
    # Final state
    if result.get("success"):
        created_videos = result.get("created_videos", [])
        output_folder = result.get("output_folder", "")
        errors = result.get("errors")
        
        # Build success message
        video_list = "\n".join([f"- {v}" for v in created_videos])
        msg = f"✅ **Successfully created {len(created_videos)} video(s)!**\n\n"
        msg += f"**Videos created:**\n{video_list}\n\n"
        msg += f"**Output folder:** `{output_folder}`"
        
        if errors:
            error_list = "\n".join([f"- {e}" for e in errors])
            msg += f"\n\n⚠️ **Warnings:**\n{error_list}"
        
        yield (
            gr.update(value=msg, visible=True),
            gr.update(visible=True),  # Show button
        )
    else:
        error_msg = result.get("error", "⚠️ Unknown error occurred")
        details = result.get("details")
        
        if details:
            detail_list = "\n".join([f"- {d}" for d in details])
            error_msg += f"\n\n**Details:**\n{detail_list}"
        
        yield (
            gr.update(value=error_msg, visible=True),
            gr.update(visible=True),  # Show button
        )


def create_stitch_wrapper():
    """Factory function to create a stitch wrapper."""
    def stitch_wrapper(renders_folder, output_folder, fps):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_stitch(renders_folder, output_folder, fps):
            yield result
    return stitch_wrapper


def create_stitch_frames_ui(project_dir):
    """Create the Step 13: Stitch Frames to Video UI section.
    
    Args:
        project_dir: Gradio component for project directory
    
    Returns:
        dict with UI components that may be needed by other parts of the app
    """
    gr.Markdown("## Step 13: Stitch Frames to Video")
    gr.Markdown("Convert rendered frames from Blender into videos using ffmpeg.")
    
    with gr.Row():
        renders_folder = gr.Textbox(
            label="Path to `renders` Folder",
            value="",
            placeholder="{project_dir}/renders",
            info="Folder containing rendered frames (e.g., renders/Scene_1_Shot_1/cam_1_s_1_s_1/frame_0001.png)",
            interactive=True,
        )
        output_folder = gr.Textbox(
            label="Path to Output `videos` Folder",
            value="",
            placeholder="{project_dir}/videos",
            info="Folder to save output videos",
            interactive=True,
        )
    
    with gr.Row():
        fps = gr.Number(
            label="Frames Per Second (FPS)",
            value=24,
            precision=0,
            interactive=True,
            info="Frame rate for output videos"
        )
    
    # Status indicator
    stitch_status = gr.Markdown(value="", visible=False)
    
    # Stitch button
    stitch_btn = gr.Button(
        "🎬 Stitch Frames to Video",
        variant="primary",
        size="lg"
    )
    
    # Update default paths when project_dir changes
    def update_default_paths(proj_dir):
        renders_path = get_renders_folder(proj_dir)
        videos_path = get_videos_folder(proj_dir)
        return gr.update(value=renders_path), gr.update(value=videos_path)
    
    project_dir.change(
        fn=update_default_paths,
        inputs=[project_dir],
        outputs=[renders_folder, output_folder],
    )
    
    # Create wrapper function for the button
    stitch_wrapper = create_stitch_wrapper()
    
    # Stitch button click handler
    stitch_btn.click(
        fn=stitch_wrapper,
        inputs=[renders_folder, output_folder, fps],
        outputs=[
            stitch_status,
            stitch_btn,
        ],
    )
    
    return {
        "renders_folder": renders_folder,
        "output_folder": output_folder,
        "fps": fps,
        "stitch_status": stitch_status,
        "stitch_btn": stitch_btn,
    }
