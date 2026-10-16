"""Simple client to communicate with BlenderMCPServer running inside Blender."""

import os
import socket
import json
import time
try:
    from ..proxy_config import get_proxy_settings
except ImportError:
    from proxy_config import get_proxy_settings

# Try to import bpy - will only work when running inside Blender
try:
    import bpy
    HAS_BPY = True
except ImportError:
    HAS_BPY = False


def get_blender_mcp_port():
    """Get the MCP server port from Blender settings.
    
    Returns:
        int: The port number, or 9876 as default
    """
    if HAS_BPY:
        try:
            return bpy.context.scene.blendermcp_port
        except:
            pass
    # Fallback to environment variable (for subprocess mode)
    env_port = os.environ.get('BLENDERMCP_PORT')
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            pass
    return 9876


def is_mcp_server_running():
    """Check if the BlenderMCPServer is running according to Blender's state.
    
    Returns:
        bool: True if server is marked as running in Blender
    """
    if HAS_BPY:
        try:
            return bpy.context.scene.blendermcp_server_running
        except:
            pass
    return False


def stop_mcp_server():
    """Stop the BlenderMCPServer if running.
    
    This schedules the server stop in Blender's main thread by calling
    the registered Blender operator.
    
    Returns:
        tuple: (success: bool, message: str)
    """
    if not HAS_BPY:
        return False, "Not running inside Blender"
    
    try:
        server_stopped = [False]
        
        def stop_server_main_thread():
            try:
                result = bpy.ops.blendermcp.stop_server()
                if result == {'FINISHED'}:
                    server_stopped[0] = True
                    print("BlenderMCP server stopped via operator")
            except Exception as e:
                print(f"Error stopping server via operator: {e}")
            return None  # Don't repeat
        
        # Schedule in main thread
        bpy.app.timers.register(stop_server_main_thread, first_interval=0.0)
        
        # Wait for stop to complete
        for _ in range(5):
            time.sleep(0.2)
            if server_stopped[0]:
                return True, "Server stopped"
        
        return True, "Server stop requested"
        
    except Exception as e:
        return False, f"Failed to stop server: {str(e)}"


def start_mcp_server(force_restart=False):
    """Start the BlenderMCPServer if not already running.
    
    This schedules the server start in Blender's main thread by calling
    the registered Blender operator.
    
    Args:
        force_restart: If True, stop the server first before starting
    
    Returns:
        tuple: (success: bool, message: str)
    """
    if not HAS_BPY:
        return False, "Not running inside Blender"
    
    try:
        # Force restart if requested
        if force_restart:
            print("Force restarting MCP server...")
            stop_mcp_server()
            time.sleep(0.5)
            # Reset the running flag
            try:
                bpy.context.scene.blendermcp_server_running = False
            except:
                pass
        
        # Check if already running
        try:
            if bpy.context.scene.blendermcp_server_running and not force_restart:
                return True, "Server already running"
            port = bpy.context.scene.blendermcp_port
        except Exception:
            port = 9876
        
        # Track if server started successfully
        server_started = [False]
        server_error = [None]
        
        # Define function to start server in main thread using the operator
        def start_server_main_thread():
            try:
                # Call the registered operator - this is the same code path as clicking the button
                result = bpy.ops.blendermcp.start_server()
                if result == {'FINISHED'}:
                    server_started[0] = True
                    print(f"BlenderMCP server started via operator on port {port}")
                else:
                    server_error[0] = f"Operator returned: {result}"
            except Exception as e:
                server_error[0] = str(e)
                print(f"Error starting server via operator: {e}")
            return None  # Don't repeat
        
        # Schedule in main thread
        bpy.app.timers.register(start_server_main_thread, first_interval=0.0)
        
        # Wait for the server to start (check multiple times)
        for _ in range(10):
            time.sleep(0.3)
            try:
                if bpy.context.scene.blendermcp_server_running:
                    return True, f"Server started on port {port}"
            except Exception:
                pass
            if server_started[0]:
                return True, f"Server started on port {port}"
            if server_error[0]:
                return False, f"Failed to start server: {server_error[0]}"
        
        return False, "Server start timed out. Please start manually from Blender's panel."
        
    except Exception as e:
        return False, f"Failed to start server: {str(e)}"


class BlenderClient:
    """Client for sending commands to BlenderMCPServer.
    
    This client connects to the BlenderMCPServer running inside Blender
    and sends JSON commands directly without using the MCP protocol.
    """
    
    def __init__(self, host='localhost', port=None):
        """Initialize the client.
        
        Args:
            host: Host where BlenderMCPServer is running
            port: Port where BlenderMCPServer is listening. If None, will use Blender's configured port.
        """
        self.host = host
        self._port = port
    
    @property
    def port(self):
        """Get the port, dynamically from Blender if not explicitly set."""
        if self._port is not None:
            return self._port
        return get_blender_mcp_port()
    
    @port.setter
    def port(self, value):
        self._port = value
    
    def _send_command(self, command_type: str, params: dict = None) -> dict:
        """Send a command to the Blender server and return the response.
        
        Args:
            command_type: The type of command to execute
            params: Parameters for the command
            
        Returns:
            dict: Response from the server
        """
        command = {
            "type": command_type,
            "params": params or {},
            "proxy_settings": get_proxy_settings().to_dict()
        }
        
        try:
            # Create socket and connect
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3600)  # 1 hour timeout for long operations
            sock.connect((self.host, self.port))
            
            # Send command
            command_json = json.dumps(command)
            sock.sendall(command_json.encode('utf-8'))
            
            # Receive response
            response_data = b''
            while True:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                response_data += chunk
                # Try to parse - if successful, we have complete response
                try:
                    response = json.loads(response_data.decode('utf-8'))
                    break
                except json.JSONDecodeError:
                    # Incomplete data, keep receiving
                    continue
            
            sock.close()
            return response
            
        except socket.timeout:
            return {"status": "error", "message": "Connection timed out. Is Blender running with the server started?"}
        except ConnectionRefusedError:
            return {"status": "error", "message": "Connection refused. Please start the BlenderMCPServer in Blender first."}
        except Exception as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}
    
    def is_connected(self) -> bool:
        """Check if we can connect to the Blender server.
        
        Returns:
            bool: True if connection is successful
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((self.host, self.port))
            sock.close()
            return True
        except:
            return False
    
    def ensure_server_running(self) -> tuple:
        """Ensure the MCP server is running, starting it if necessary.
        
        If the server status says running but connection fails, it will
        force restart the server to fix stale state from previous sessions.
        
        Returns:
            tuple: (success: bool, message: str)
        """
        # First check if we can connect
        if self.is_connected():
            return True, "Server is running"
        
        # Check if status says running but connection failed - this means stale state
        status_says_running = is_mcp_server_running()
        
        # Try to start the server (force restart if status was stale)
        success, message = start_mcp_server(force_restart=status_says_running)
        if not success:
            return False, message
        
        # Wait for server to fully initialize before checking connection
        # The server is started asynchronously via bpy.app.timers, so we need
        # to give it time to create the socket and start listening
        time.sleep(1.0)  # Initial delay for server socket to be created
        
        # Retry connection with more attempts and longer delays
        for attempt in range(10):
            if self.is_connected():
                return True, "Server started successfully"
            time.sleep(0.5)
        
        # If still failing, try one more force restart
        if not self.is_connected():
            print("Connection still failing after start, attempting force restart...")
            success, message = start_mcp_server(force_restart=True)
            if success:
                time.sleep(3)
                for attempt in range(5):
                    if self.is_connected():
                        return True, "Server restarted successfully"
                    time.sleep(0.5)
        
        return False, "Server started but connection failed. Please try again."
    
    def get_scene_info(self) -> dict:
        """Get information about the current Blender scene.
        
        Returns:
            dict: Scene information or error
        """
        return self._send_command("get_scene_info")

    def capture_manual_adjustments(self, base_scenes: list, shots: list) -> dict:
        """Capture known asset transforms from base and shot scenes.

        This operation is read-only. Saving the blend file is deliberately a
        separate command so the Gradio UI can preview and confirm changes first.
        """
        return self._send_command("capture_manual_adjustments", {
            "base_scenes": base_scenes,
            "shots": shots,
        })

    def save_current_blend_file(self) -> dict:
        """Save the currently open blend file to its existing filepath."""
        return self._send_command("save_current_blend_file", {})
    
    def format_assets(self, path_to_script: str, model_output_dir: str, anyllm_api_key: str = None, vision_model: str = "gemini-3-flash-preview", anyllm_api_base: str = None, anyllm_provider: str = "gemini", model_id_list: list = None, output_json_filename: str = "formatted_model.json") -> dict:
        """Format all models in a storyboard script.
        
        Args:
            path_to_script: Path to the storyboard script JSON file
            model_output_dir: Directory to save formatted models
            anyllm_api_key: API key for LLM service (for rotation correction)
            vision_model: LLM model identifier for vision tasks
            anyllm_api_base: Optional API base URL for LLM service
            anyllm_provider: LLM provider (default: "gemini")
            model_id_list: Optional list of asset_ids to format. If None, format all.
                          If provided but output json doesn't exist, format all.
            output_json_filename: Filename for the output JSON (default: formatted_model.json)
            
        Returns:
            dict: Result of the formatting process
        """
        return self._send_command("format_assets", {
            "path_to_script": path_to_script,
            "model_output_dir": model_output_dir,
            "anyllm_api_key": anyllm_api_key,
            "vision_model": vision_model,
            "anyllm_api_base": anyllm_api_base,
            "anyllm_provider": anyllm_provider,
            "model_id_list": model_id_list,
            "output_json_filename": output_json_filename
        })
    
    def lighting_designer(self, scene_description: str = None, asset_id: str = None, categories_limitation: list = None, anyllm_api_key: str = None, anyllm_api_base: str = None, anyllm_provider: str = "gemini", vision_model: str = "gemini-3-flash-preview") -> dict:
        """Run the lighting designer to set up scene lighting.
        
        Uses semantic search with pre-computed embeddings for efficient HDRI matching,
        with LLM-based reranking for improved relevance.
        
        Args:
            scene_description: Description of the scene lighting needs
                              (e.g., "afternoon woods", "urban sunset", "studio lighting").
                              Ignored if asset_id is provided.
            asset_id: Optional Polyhaven HDRI asset ID to use directly,
                     bypassing AI selection.
            categories_limitation: Optional list of category strings. If provided, only HDRIs
                                  containing ALL specified categories will be returned
                                  (e.g., ["sunrise-sunset", "pure skies"]).
            anyllm_api_key: API key for LLM service (for reranking)
            anyllm_api_base: API base URL for LLM service
            anyllm_provider: LLM provider (default: "gemini")
            vision_model: Vision model for LLM reranking (default: "gemini-3-flash-preview")
            
        Returns:
            dict: Result of the lighting setup with asset_id, hdri_name, scores
        """
        params = {}
        if asset_id:
            params["asset_id"] = asset_id
        elif scene_description:
            params["scene_description"] = scene_description
        if categories_limitation:
            params["categories_limitation"] = categories_limitation
        params["anyllm_api_key"] = anyllm_api_key
        params["anyllm_api_base"] = anyllm_api_base
        params["anyllm_provider"] = anyllm_provider
        params["vision_model"] = vision_model
        return self._send_command("lighting_designer", params)
    
    def execute_code(self, code: str) -> dict:
        """Execute arbitrary Python code in Blender.
        
        Args:
            code: Python code to execute
            
        Returns:
            dict: Result of code execution
        """
        return self._send_command("execute_code", {"code": code})
    
    def import_all_assets_to_all_scenes_json_input(self, json_filepath: str) -> dict:
        """Import all assets to all scenes from a JSON file.
        
        Args:
            json_filepath: Path to the JSON file containing asset_sheet and scene_details
            
        Returns:
            dict: Result with 'success' (bool) and 'failed_objects' list if any failures
        """
        return self._send_command("import_all_assets_to_all_scenes_json_input", {
            "json_filepath": json_filepath
        })
    
    def get_asset_transform(self, model_name: str) -> dict:
        """Get all transform properties of a model in Blender.
        
        Args:
            model_name: Name of an existing model in the scene
            
        Returns:
            dict: Transform data including location, rotation, scale, dimensions
        """
        return self._send_command("get_asset_transform", {
            "model_name": model_name
        })
    
    def switch_or_create_scene(self, scene_name: str) -> dict:
        """Switch to a scene with the given name, or create it if it doesn't exist.
        
        Args:
            scene_name: The name of the scene to switch to or create
            
        Returns:
            dict: Result with 'success' (bool)
        """
        return self._send_command("switch_or_create_scene", {
            "scene_name": scene_name
        })
    
    def delete_all_scenes_and_assets(self) -> dict:
        """Delete all scenes and assets, leaving only an empty scene named 'Scene'.
        
        This is useful for resetting Blender to a clean state before importing new assets.
        
        Returns:
            dict: Result with success status, message, and list of deleted scenes
        """
        return self._send_command("delete_all_scenes_and_assets", {})
    
    def import_supplementary_assets_to_all_scenes_json_input(self, json_filepath: str) -> dict:
        """Import supplementary assets to all scenes from a JSON file.
        
        Args:
            json_filepath: Path to the JSON file containing asset_sheet and scene_details
                          for supplementary assets
            
        Returns:
            dict: Result with 'success' (bool) and 'failed_objects' list if any failures
        """
        return self._send_command("import_supplementary_assets_to_all_scenes_json_input", {
            "json_filepath": json_filepath
        })
    
    def apply_asset_modifications_json_input(self, json_filepath: str) -> dict:
        """Apply asset modifications to shot scenes from a JSON file.
        
        Args:
            json_filepath: Path to the JSON file containing shot_details with
                          asset_modifications
            
        Returns:
            dict: Result with 'success' (bool), 'modified_count', and 'errors' list if any failures
        """
        return self._send_command("apply_asset_modifications_json_input", {
            "json_filepath": json_filepath
        })
    
    def import_animated_assets_to_all_shots_json_input(self, json_filepath: str) -> dict:
        """Import animated assets to all shots from a JSON file.
        
        This function reads a story script JSON file and imports animated models
        into shot scenes for each shot in shot_details.
        
        Args:
            json_filepath: Path to the JSON file containing shot_details and scene_details
            
        Returns:
            dict: Result with 'success' (bool) and details about the import
        """
        return self._send_command("import_animated_assets_to_all_shots_json_input", {
            "json_filepath": json_filepath
        })
    
    def delete_all_shots(self) -> dict:
        """Delete all shot scenes in Blender.
        
        Deletes all scenes matching pattern Scene_{scene_id}_Shot_{shot_id},
        but leaves all other scenes intact (including original scenes like Scene_{scene_id}).
        
        Returns:
            dict: Result with success status and list of deleted scenes
        """
        return self._send_command("delete_all_shots", {})
    
    def environment_artist(
        self,
        ground_description: str = None,
        asset_id: str = None,
        categories_limitation: list = None,
        width: float = 10.0,
        depth: float = 10.0,
        wall_description: str = None,
        wall_asset_id: str = None,
        wall_categories_limitation: list = None,
        wall_x: float = None,
        wall_x_negative: float = None,
        wall_y: float = None,
        wall_y_negative: float = None,
        wall_z: float = None,
        anyllm_api_key: str = None,
        anyllm_api_base: str = None,
        anyllm_provider: str = "gemini",
        vision_model: str = "gemini-3-flash-preview",
    ) -> dict:
        """Create a ground plane and optionally indoor walls using textures from Polyhaven.
        
        Uses semantic search with pre-computed embeddings for efficient texture matching,
        with LLM-based reranking for improved relevance.
        
        Args:
            ground_description: Description of the ground/terrain
                               (e.g., "cobblestone floor", "grassy meadow").
                               Ignored if asset_id is provided.
            asset_id: Optional Polyhaven texture asset ID to use directly,
                     bypassing AI selection.
            categories_limitation: Optional list of category strings. If provided, only textures
                                  containing ALL specified categories will be returned
                                  (e.g., ["floor", "cobblestone"]).
            width: Width of the ground plane in meters (default: 10.0)
            depth: Depth of the ground plane in meters (default: 10.0)
            wall_description: Description of wall texture for indoor scenes.
            wall_asset_id: Optional Polyhaven texture asset ID for walls.
            wall_categories_limitation: Optional category filter for wall textures.
            wall_x: Positive X boundary for walls.
            wall_x_negative: Negative X boundary for walls.
            wall_y: Positive Y boundary for walls.
            wall_y_negative: Negative Y boundary for walls.
            wall_z: Height of walls in meters.
            anyllm_api_key: API key for LLM service (for reranking)
            anyllm_api_base: API base URL for LLM service
            anyllm_provider: LLM provider (default: "gemini")
            vision_model: Vision model for LLM reranking (default: "gemini-3-flash-preview")
            
        Returns:
            dict: Result of the environment creation with asset_id, texture_name,
                  plane_name, dimensions, tile configuration, and scores
        """
        params = {
            "width": width,
            "depth": depth
        }
        if asset_id:
            params["asset_id"] = asset_id
        elif ground_description:
            params["ground_description"] = ground_description
        if categories_limitation:
            params["categories_limitation"] = categories_limitation
        
        # Wall parameters
        if wall_description:
            params["wall_description"] = wall_description
        if wall_asset_id:
            params["wall_asset_id"] = wall_asset_id
        if wall_categories_limitation:
            params["wall_categories_limitation"] = wall_categories_limitation
        if wall_x is not None:
            params["wall_x"] = wall_x
        if wall_x_negative is not None:
            params["wall_x_negative"] = wall_x_negative
        if wall_y is not None:
            params["wall_y"] = wall_y
        if wall_y_negative is not None:
            params["wall_y_negative"] = wall_y_negative
        if wall_z is not None:
            params["wall_z"] = wall_z
        
        params["anyllm_api_key"] = anyllm_api_key
        params["anyllm_api_base"] = anyllm_api_base
        params["anyllm_provider"] = anyllm_provider
        params["vision_model"] = vision_model
        
        return self._send_command("environment_artist", params)
    
    def delete_asset(self, model_name: str) -> dict:
        """Delete an asset (object) from the current scene by its name.
        
        Args:
            model_name: The name of the object to delete.
            
        Returns:
            dict: Result with 'success' (bool) and 'message' or 'error'.
        """
        return self._send_command("delete_asset", {
            "model_name": model_name
        })
    
    def set_render(self, engine: str = "EEVEE", samples: int = None, persistent_data: bool = True) -> dict:
        """Configure render settings for all scenes in the Blender file.
        
        Args:
            engine: Render engine to use. Options: "EEVEE" or "Cycles" (case-insensitive).
            samples: Number of render samples (optional).
                    For EEVEE: default is 64
                    For Cycles: default is 512 (render), viewport uses 128
            persistent_data: Whether to enable persistent data for faster re-renders. Default: True.
            
        Returns:
            dict: Result with 'success' (bool), 'message', and 'scenes' list with details
        """
        params = {"engine": engine, "persistent_data": persistent_data}
        if samples is not None:
            params["samples"] = samples
        return self._send_command("set_render", params)
    
    def resize_assets(self, path_to_script: str, model_output_dir: str, model_id_list: list = None, output_json_filename: str = "resized_model.json") -> dict:
        """Resize all models in a script file based on their estimated dimensions.
        
        Args:
            path_to_script: Path to the JSON file containing asset_sheet with dimension estimates
            model_output_dir: Directory to save resized models
            model_id_list: Optional list of asset_ids to resize. If None, resize all.
                          If provided but output json doesn't exist, resize all.
            output_json_filename: Filename for the output JSON (default: resized_model.json)
            
        Returns:
            dict: Result of the resizing process
        """
        return self._send_command("resize_assets", {
            "path_to_script": path_to_script,
            "model_output_dir": model_output_dir,
            "model_id_list": model_id_list,
            "output_json_filename": output_json_filename
        })
    
    def get_camera_info(self, scene_name: str, camera_name: str) -> dict:
        """Get camera information from Blender by scene name and camera name.
        
        Reads camera transform, focal length, DoF settings, and keyframe data.
        
        Args:
            scene_name: Name of the scene containing the camera
            camera_name: Name of the camera object
            
        Returns:
            dict: Camera info including transforms, parameters, and animation data
        """
        return self._send_command("get_camera_info", {
            "scene_name": scene_name,
            "camera_name": camera_name,
        })
    
    def camera_operator(
        self,
        path_to_input_json: str,
        vision_model: str = "gemini-2.5-flash",
        anyllm_api_key: str = None,
        anyllm_api_base: str = None,
        anyllm_provider: str = "gemini",
        camera_type: str = "director",
        max_additional_cameras: int = 1,
        camera_name_filter: list = None,
        start_frame: int = 1,
        end_frame: int = 73,
        max_adjustment_rounds: int = 5,
        preview_image_save_dir: str = None,
    ) -> dict:
        """Place cameras in Blender scenes based on storyboard instructions.
        
        Args:
            path_to_input_json: Path to JSON file with shot_details
            vision_model: Vision model for LLM (default: gemini-2.5-flash)
            anyllm_api_key: API key for any-llm
            anyllm_api_base: API base URL for any-llm (optional)
            anyllm_provider: LLM provider (default: "gemini")
            camera_type: 'director', 'additional', or 'all' (default: director)
            max_additional_cameras: Maximum additional cameras per shot (default: 1)
            camera_name_filter: List of camera names to place (None = all)
            start_frame: Start frame for camera animation (default: 1)
            end_frame: End frame for camera animation (default: 73)
            max_adjustment_rounds: Maximum LLM adjustment rounds (default: 5)
            preview_image_save_dir: Directory to save preview images (optional)
            
        Returns:
            dict: Result with shot_details, cameras_placed, cameras_failed
        """
        return self._send_command("camera_operator", {
            "path_to_input_json": path_to_input_json,
            "vision_model": vision_model,
            "anyllm_api_key": anyllm_api_key,
            "anyllm_api_base": anyllm_api_base,
            "anyllm_provider": anyllm_provider,
            "camera_type": camera_type,
            "max_additional_cameras": max_additional_cameras,
            "camera_name_filter": camera_name_filter,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "max_adjustment_rounds": max_adjustment_rounds,
            "preview_image_save_dir": preview_image_save_dir,
        })
    
    def resume_camera_operator(
        self,
        path_to_input_json: str,
        camera_name_filter: list = None,
    ) -> dict:
        """Resume/recreate cameras from a previously saved JSON file.
        
        Args:
            path_to_input_json: Path to JSON file containing camera placement info
            camera_name_filter: List of camera names to resume (None = all)
            
        Returns:
            dict: Result with cameras_resumed, cameras_failed
        """
        return self._send_command("resume_camera_operator", {
            "path_to_input_json": path_to_input_json,
            "camera_name_filter": camera_name_filter,
        })
