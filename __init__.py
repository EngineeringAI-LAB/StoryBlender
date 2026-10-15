# Code created by Siddharth Ahuja: www.github.com/ahujasid © 2025

import bpy
import webbrowser
import subprocess
import socket
import os
import sys
import atexit
from bpy.props import IntProperty, BoolProperty, StringProperty

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
os.environ.setdefault('DISABLE_TELEMETRY', '1')


# Import the BlenderMCP functionality from blender_mcp.py
try:
    from .src.blender_mcp import BlenderMCPServer
except Exception as _blender_mcp_err:
    BlenderMCPServer = None
    print(f"ERROR [StoryBlender]: Failed to import blender_mcp: {_blender_mcp_err}")
    import traceback; traceback.print_exc()

def _tag_redraw_view3d():
    """Force redraw of all VIEW_3D areas so the sidebar panel updates."""
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass


def check_server_actually_running(port):
    """Check if a server is actually running by trying to connect to the socket.
    
    Args:
        port: The port to check
        
    Returns:
        bool: True if server is actually listening on the port
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except:
        return False


# Alias for backward compatibility
check_mcp_server_actually_running = check_server_actually_running


def _ensure_mcp_server(scene):
    """Start our own bridge, avoiding ports used by other Blender add-ons."""
    if BlenderMCPServer is None:
        raise RuntimeError("StoryBlender core module failed to load. Check the console.")
    server = getattr(bpy.types, "blendermcp_server", None)
    if server is None or not server.running:
        port = scene.blendermcp_port
        # A listening socket alone does not identify StoryBlender's protocol.
        # In particular, the separate MCP extension also defaults to port 9876.
        while check_server_actually_running(port):
            if port == 65535:
                raise RuntimeError("No free StoryBlender bridge port available")
            port += 1
        if port != scene.blendermcp_port:
            print(f"StoryBlender bridge port {scene.blendermcp_port} is occupied; using {port}.")
        server = BlenderMCPServer(port=port)
        server.start()
        if not server.running:
            raise RuntimeError(f"Could not start StoryBlender bridge on port {port}")
        bpy.types.blendermcp_server = server
    scene.blendermcp_port = server.port
    scene.blendermcp_server_running = True
    return server.port


def _find_python_executable():
    """Find the Python executable for subprocess use.

    Inside Blender, sys.executable points to the Blender binary.
    This function locates Blender's bundled Python interpreter.
    """
    basename = os.path.basename(sys.executable).lower()
    if 'python' in basename:
        return sys.executable

    # Blender's embedded Python is under sys.prefix
    if sys.platform == 'win32':
        candidate = os.path.join(sys.prefix, 'python.exe')
    else:
        candidate = os.path.join(
            sys.prefix, 'bin',
            f'python{sys.version_info.major}.{sys.version_info.minor}'
        )

    if os.path.isfile(candidate):
        return candidate

    # Fallback: try common names
    for name in ('python3', 'python'):
        p = os.path.join(sys.prefix, 'bin', name)
        if os.path.isfile(p):
            return p

    return sys.executable


def _kill_port_occupant(port):
    """Kill any process listening on the given port (handles stale/orphan processes)."""
    try:
        if sys.platform == 'win32':
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    try:
                        pid = int(parts[-1])
                        subprocess.run(
                            ['taskkill', '/F', '/PID', str(pid)],
                            capture_output=True, timeout=5
                        )
                    except (ValueError, subprocess.SubprocessError):
                        pass
        else:
            import signal
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'],
                capture_output=True, text=True, timeout=5
            )
            for pid_str in result.stdout.strip().splitlines():
                pid = int(pid_str)
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
    except Exception:
        pass


def _kill_gradio_process():
    """Kill the Gradio subprocess if it exists."""
    proc = getattr(bpy.types, 'gradio_process', None)
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


# Ensure Gradio subprocess is cleaned up on Blender exit
atexit.register(_kill_gradio_process)


# Blender UI Panel
class BLENDERMCP_PT_Panel(bpy.types.Panel):
    bl_label = "StoryBlender"
    bl_idname = "BLENDERMCP_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'StoryBlender'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "blendermcp_port")

        if not scene.blendermcp_server_running:
            layout.operator("blendermcp.start_server", text="Connect to MCP server")
        else:
            layout.operator("blendermcp.stop_server", text="Disconnect from MCP server")
            layout.label(text=f"Running on port {scene.blendermcp_port}")
        
        # Gradio UI section
        layout.separator()
        layout.label(text="Gradio UI:")
        layout.prop(scene, "gradio_port")
        
        if scene.gradio_launching:
            layout.label(text="Launching Gradio, please wait...", icon='SORTTIME')
        elif not scene.gradio_server_running:
            layout.operator("blendermcp.launch_gradio", text="Launch Gradio")
        else:
            layout.label(text="Gradio Running")
            if scene.gradio_url:
                layout.label(text=scene.gradio_url)
            layout.operator("blendermcp.stop_gradio", text="Stop Gradio")

# Operator to start the server
class BLENDERMCP_OT_StartServer(bpy.types.Operator):
    bl_idname = "blendermcp.start_server"
    bl_label = "Connect to MCP"
    bl_description = "Start the MCP server"

    def execute(self, context):
        scene = context.scene

        try:
            _ensure_mcp_server(scene)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        return {'FINISHED'}

# Operator to stop the server
class BLENDERMCP_OT_StopServer(bpy.types.Operator):
    bl_idname = "blendermcp.stop_server"
    bl_label = "Stop MCP connection"
    bl_description = "Stop the connection to MCP"

    def execute(self, context):
        scene = context.scene

        # Stop the server if it exists
        if hasattr(bpy.types, "blendermcp_server") and bpy.types.blendermcp_server:
            bpy.types.blendermcp_server.stop()
            del bpy.types.blendermcp_server

        scene.blendermcp_server_running = False

        return {'FINISHED'}

# Operator to launch Gradio in a subprocess (thread-safe for Blender)
class BLENDERMCP_OT_LaunchGradio(bpy.types.Operator):
    bl_idname = "blendermcp.launch_gradio"
    bl_label = "Launch Gradio UI"
    bl_description = "Launch the Gradio web interface"

    def execute(self, context):
        scene = context.scene
        port = scene.gradio_port
        try:
            mcp_port = _ensure_mcp_server(scene)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        # Generate the URL
        url = f"http://127.0.0.1:{port}"
        scene.gradio_url = url

        # Path to storyblender_app.py
        src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
        app_script = os.path.join(src_dir, "gradio_app", "storyblender_app.py")
        python_exe = _find_python_executable()

        # Set up environment for the subprocess
        env = os.environ.copy()
        env['GRADIO_ANALYTICS_ENABLED'] = 'False'
        env['DISABLE_TELEMETRY'] = '1'
        env['GRADIO_SERVER_PORT'] = str(port)
        env['BLENDERMCP_PORT'] = str(mcp_port)
        # Pass Blender's sys.path so the subprocess can find packages
        # installed via blender_manifest.toml wheels (e.g. gradio, etc.)
        env['PYTHONPATH'] = os.pathsep.join(sys.path)

        # Kill any stale process occupying the target port (e.g. from a previous crash)
        _kill_gradio_process()
        _kill_port_occupant(port)

        try:
            # Launch Gradio as a subprocess — its threads live in a separate
            # process and cannot interfere with Blender's main thread.
            proc = subprocess.Popen(
                [python_exe, '-u', app_script],
                env=env,
            )
            bpy.types.gradio_process = proc

            # Poll until Gradio is actually ready, then mark running and open the browser
            scene.gradio_launching = True
            _poll_state = {"attempts": 0, "max_attempts": 120}  # ~120s timeout

            def _poll_gradio_ready():
                _poll_state["attempts"] += 1
                # Check if the process died
                if proc.poll() is not None:
                    scene.gradio_launching = False
                    scene.gradio_server_running = False
                    _tag_redraw_view3d()
                    return None
                if check_server_actually_running(port):
                    scene.gradio_server_running = True
                    _tag_redraw_view3d()
                    # Delay 1s for Gradio to finish preparing HTML content
                    def _open_after_delay():
                        scene.gradio_launching = False
                        _tag_redraw_view3d()
                        webbrowser.open(url)
                        return None
                    bpy.app.timers.register(_open_after_delay, first_interval=1.0)
                    return None
                if _poll_state["attempts"] >= _poll_state["max_attempts"]:
                    # Timeout – open anyway and let the user see the page
                    scene.gradio_server_running = True
                    scene.gradio_launching = False
                    _tag_redraw_view3d()
                    webbrowser.open(url)
                    return None
                return 1.0  # retry in 1 second

            bpy.app.timers.register(_poll_gradio_ready, first_interval=1.0)

            self.report({'INFO'}, f"Gradio launched at {url} (subprocess pid={proc.pid})")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to launch Gradio: {e}")

        return {'FINISHED'}

# Operator to stop Gradio subprocess
class BLENDERMCP_OT_StopGradio(bpy.types.Operator):
    bl_idname = "blendermcp.stop_gradio"
    bl_label = "Stop Gradio UI"
    bl_description = "Stop the Gradio web interface"

    def execute(self, context):
        scene = context.scene

        _kill_gradio_process()
        if hasattr(bpy.types, 'gradio_process'):
            del bpy.types.gradio_process

        scene.gradio_server_running = False
        scene.gradio_url = ""

        self.report({'INFO'}, "Gradio stopped")
        return {'FINISHED'}

# Registration functions
def register():
    bpy.types.Scene.blendermcp_port = IntProperty(
        name="Port",
        description="Port for the BlenderMCP server",
        default=9876,
        min=1024,
        max=65535
    )

    bpy.types.Scene.blendermcp_server_running = BoolProperty(
        name="Server Running",
        default=False
    )
    
    # Gradio properties
    bpy.types.Scene.gradio_port = IntProperty(
        name="Gradio Port",
        description="Port for the Gradio web interface",
        default=7860,
        min=1024,
        max=65535
    )
    
    bpy.types.Scene.gradio_server_running = BoolProperty(
        name="Gradio Running",
        default=False
    )

    bpy.types.Scene.gradio_launching = BoolProperty(
        name="Gradio Launching",
        default=False
    )
    
    bpy.types.Scene.gradio_url = StringProperty(
        name="Gradio URL",
        default=""
    )

    bpy.utils.register_class(BLENDERMCP_PT_Panel)
    bpy.utils.register_class(BLENDERMCP_OT_StartServer)
    bpy.utils.register_class(BLENDERMCP_OT_StopServer)
    bpy.utils.register_class(BLENDERMCP_OT_LaunchGradio)
    bpy.utils.register_class(BLENDERMCP_OT_StopGradio)

    print("BlenderMCP addon registered")
    
    # Schedule a check to verify MCP and Gradio server status on startup
    # This fixes the case where Blender was closed without stopping the servers
    def verify_server_status():
        try:
            scene = bpy.context.scene
            
            # Check MCP server status
            mcp_port = scene.blendermcp_port
            if scene.blendermcp_server_running:
                server = getattr(bpy.types, "blendermcp_server", None)
                if not server or not server.running:
                    # Server is not actually running, reset the status
                    print(f"MCP server status was 'running' but port {mcp_port} is not listening. Resetting status.")
                    scene.blendermcp_server_running = False
                    # Also clean up any stale server reference
                    if hasattr(bpy.types, "blendermcp_server"):
                        try:
                            del bpy.types.blendermcp_server
                        except:
                            pass
            
            # Check Gradio server status
            gradio_port = scene.gradio_port
            if scene.gradio_server_running:
                if not check_server_actually_running(gradio_port):
                    # Gradio is not actually running, reset the status
                    print(f"Gradio status was 'running' but port {gradio_port} is not listening. Resetting status.")
                    scene.gradio_server_running = False
                    scene.gradio_url = ""
                    
        except Exception as e:
            print(f"Error verifying server status on startup: {e}")
        return None  # Don't repeat
    
    # Schedule the check after Blender has fully initialized
    bpy.app.timers.register(verify_server_status, first_interval=1.0)

    def auto_start_mcp_server():
        if BlenderMCPServer is None:
            return None
        try:
            scene = bpy.context.scene
            _ensure_mcp_server(scene)
        except Exception as e:
            print(f"Error auto-starting MCP server: {e}")
        return None

    bpy.app.timers.register(auto_start_mcp_server, first_interval=1.5)

def unregister():
    # Stop the server if it's running
    if hasattr(bpy.types, "blendermcp_server") and bpy.types.blendermcp_server:
        bpy.types.blendermcp_server.stop()
        del bpy.types.blendermcp_server
    
    # Stop Gradio subprocess if running
    _kill_gradio_process()
    if hasattr(bpy.types, 'gradio_process'):
        try:
            del bpy.types.gradio_process
        except Exception:
            pass

    bpy.utils.unregister_class(BLENDERMCP_PT_Panel)
    bpy.utils.unregister_class(BLENDERMCP_OT_StartServer)
    bpy.utils.unregister_class(BLENDERMCP_OT_StopServer)
    bpy.utils.unregister_class(BLENDERMCP_OT_LaunchGradio)
    bpy.utils.unregister_class(BLENDERMCP_OT_StopGradio)

    del bpy.types.Scene.blendermcp_port
    del bpy.types.Scene.blendermcp_server_running
    del bpy.types.Scene.gradio_port
    del bpy.types.Scene.gradio_server_running
    del bpy.types.Scene.gradio_launching
    del bpy.types.Scene.gradio_url

    print("BlenderMCP addon unregistered")

if __name__ == "__main__":
    register()
