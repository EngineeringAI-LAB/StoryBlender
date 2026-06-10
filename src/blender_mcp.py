# Code created by Siddharth Ahuja: www.github.com/ahujasid © 2025

import bpy
import mathutils
import json
import threading
import socket
import time
import tempfile
import traceback
import os
import shutil
import zipfile
import io
import gc
from contextlib import redirect_stdout, suppress
from pathlib import Path

# Wrap third-party imports in try/except so the addon can still register its
# UI on platforms where wheels were not installed correctly (e.g. Windows).
try:
    import requests
except ImportError:
    requests = None
    print("WARNING [StoryBlender]: 'requests' package not available. Some features will not work.")

try:
    from PIL import Image
except ImportError:
    Image = None
    print("WARNING [StoryBlender]: 'PIL' (Pillow) package not available. Some features will not work.")

# Defer heavy operator imports so that a failure in native extensions
# (fastembed, onnxruntime, tokenizers, pydantic, etc.) does not prevent the
# entire addon from loading.  The handler methods already have try/except
# blocks, so a missing name will surface as a clear runtime error.
try:
    from .operators.concept_artist_operators.retrieve_polyhaven_asset.search_polyhaven_assets import search_polyhaven_assets

    from .operators.environment_artist_operators.import_polyhaven_textures_as_plane import create_repetitive_plane_from_polyhaven, create_indoor_walls

    from .operators.concept_artist_operators.format_asset import format_asset
    from .operators.layout_artist_operators.resize_asset import resize_asset

    from .operators.director_operators.import_all_assets_to_all_scenes import (
        import_asset_to_scene,
        get_asset_transform,
        transform_asset,
        import_all_assets_to_all_scenes_json_input,
        import_supplementary_assets_to_all_scenes_json_input,
        apply_asset_modifications_json_input,
        switch_or_create_scene,
        delete_all_scenes_and_assets,
    )

    from .operators.animator_operators.import_animated_assets_to_all_shots import (
        import_animated_assets_to_all_shots_json_input,
        delete_all_shots,
    )

    from .operators.layout_artist_operators.camera_operator import (
        camera_operator,
        resume_camera_operator,
    )
except Exception as _op_err:
    traceback.print_exc()
    print(f"WARNING [StoryBlender]: Operator imports failed: {_op_err}")
    print("The addon UI will load, but some features may not work until dependencies are resolved.")

# Add User-Agent as required by Poly Haven API
if requests is not None:
    REQ_HEADERS = requests.utils.default_headers()
    REQ_HEADERS.update({"User-Agent": "blender-mcp"})
else:
    REQ_HEADERS = {"User-Agent": "blender-mcp"}


class BlenderMCPServer:
    def __init__(self, host='localhost', port=9876):
        self.host = host
        self.port = port
        self.running = False
        self.socket = None
        self.server_thread = None

    def start(self):
        if self.running:
            print("Server is already running")
            return

        self.running = True

        try:
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)

            # Start server thread
            self.server_thread = threading.Thread(target=self._server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()

            print(f"BlenderMCP server started on {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to start server: {str(e)}")
            self.stop()

    def stop(self):
        self.running = False

        # Close socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None

        # Wait for thread to finish
        if self.server_thread:
            try:
                if self.server_thread.is_alive():
                    self.server_thread.join(timeout=1.0)
            except:
                pass
            self.server_thread = None

        print("BlenderMCP server stopped")

    def _server_loop(self):
        """Main server loop in a separate thread"""
        print("Server thread started")
        self.socket.settimeout(1.0)  # Timeout to allow for stopping

        while self.running:
            try:
                # Accept new connection
                try:
                    client, address = self.socket.accept()
                    print(f"Connected to client: {address}")

                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client,)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except socket.timeout:
                    # Just check running condition
                    continue
                except Exception as e:
                    print(f"Error accepting connection: {str(e)}")
                    time.sleep(3)
            except Exception as e:
                print(f"Error in server loop: {str(e)}")
                if not self.running:
                    break
                time.sleep(3)

        print("Server thread stopped")

    def _handle_client(self, client):
        """Handle connected client"""
        print("Client handler started")
        client.settimeout(3600)  # 1 hour timeout
        
        # Enable TCP keepalive to prevent connection timeout during long operations
        client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        # On macOS/Linux, set keepalive parameters (send keepalive every 60s after 60s idle)
        try:
            # TCP_KEEPIDLE: time before sending keepalive probes (Linux)
            # TCP_KEEPALIVE: equivalent on macOS
            if hasattr(socket, 'TCP_KEEPIDLE'):
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
            elif hasattr(socket, 'TCP_KEEPALIVE'):
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 60)
            # TCP_KEEPINTVL: interval between keepalive probes
            if hasattr(socket, 'TCP_KEEPINTVL'):
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60)
            # TCP_KEEPCNT: number of failed probes before connection is considered broken
            if hasattr(socket, 'TCP_KEEPCNT'):
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 10)
        except (OSError, AttributeError) as e:
            print(f"Warning: Could not set TCP keepalive parameters: {e}")
        
        buffer = b''

        try:
            while self.running:
                # Receive data
                try:
                    data = client.recv(8192)
                    if not data:
                        print("Client disconnected")
                        break

                    buffer += data
                    try:
                        # Try to parse command
                        command = json.loads(buffer.decode('utf-8'))
                        buffer = b''

                        # Execute command in Blender's main thread
                        def execute_wrapper():
                            try:
                                response = self.execute_command(command)
                                response_json = json.dumps(response)
                                try:
                                    client.sendall(response_json.encode('utf-8'))
                                except:
                                    print("Failed to send response - client disconnected")
                            except Exception as e:
                                print(f"Error executing command: {str(e)}")
                                traceback.print_exc()
                                try:
                                    error_response = {
                                        "status": "error",
                                        "message": str(e)
                                    }
                                    client.sendall(json.dumps(error_response).encode('utf-8'))
                                except:
                                    pass
                            return None

                        # Schedule execution in main thread
                        bpy.app.timers.register(execute_wrapper, first_interval=0.0)
                    except json.JSONDecodeError:
                        # Incomplete data, wait for more
                        pass
                except Exception as e:
                    print(f"Error receiving data: {str(e)}")
                    break
        except Exception as e:
            print(f"Error in client handler: {str(e)}")
        finally:
            try:
                client.close()
            except:
                pass
            print("Client handler stopped")

    def execute_command(self, command):
        """Execute a command in the main Blender thread"""
        try:
            return self._execute_command_internal(command)

        except Exception as e:
            print(f"Error executing command: {str(e)}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _execute_command_internal(self, command):
        """Internal command execution with proper context"""
        cmd_type = command.get("type")
        params = command.get("params", {})

        # Base handlers that are always available
        handlers = {
            "get_scene_info": self.get_scene_info,
            "get_object_info": self.get_object_info,
            "get_viewport_screenshot": self.get_viewport_screenshot,
            "execute_code": self.execute_code,
            "get_object_image": self.get_object_image,
            "download_polyhaven_asset": self.download_polyhaven_asset,
            "lighting_designer": self.lighting_designer,
            "environment_artist": self.environment_artist,
            "format_assets": self.format_assets,
            "resize_assets": self.resize_assets,
            "import_asset_to_scene": self.handle_import_asset_to_scene,
            "get_asset_transform": self.handle_get_asset_transform,
            "transform_asset": self.handle_transform_asset,
            "import_all_assets_to_all_scenes_json_input": self.handle_import_all_assets_to_all_scenes_json_input,
            "import_supplementary_assets_to_all_scenes_json_input": self.handle_import_supplementary_assets_to_all_scenes_json_input,
            "apply_asset_modifications_json_input": self.handle_apply_asset_modifications_json_input,
            "switch_or_create_scene": self.handle_switch_or_create_scene,
            "delete_all_scenes_and_assets": self.handle_delete_all_scenes_and_assets,
            "import_animated_assets_to_all_shots_json_input": self.handle_import_animated_assets_to_all_shots_json_input,
            "delete_all_shots": self.handle_delete_all_shots,
            "delete_asset": self.handle_delete_asset,
            "set_render": self.set_render,
            "camera_operator": self.handle_camera_operator,
            "resume_camera_operator": self.handle_resume_camera_operator,
            "get_camera_info": self.handle_get_camera_info,
        }


        handler = handlers.get(cmd_type)
        if handler:
            try:
                print(f"Executing handler for {cmd_type}")
                result = handler(**params)
                print(f"Handler execution complete")
                return {"status": "success", "result": result}
            except Exception as e:
                print(f"Error in handler: {str(e)}")
                traceback.print_exc()
                return {"status": "error", "message": str(e)}
        else:
            return {"status": "error", "message": f"Unknown command type: {cmd_type}"}



    def get_scene_info(self):
        """Get information about the current Blender scene"""
        try:
            print("Getting scene info...")
            # Simplify the scene info to reduce data size
            scene_info = {
                "name": bpy.context.scene.name,
                "object_count": len(bpy.context.scene.objects),
                "objects": [],
                "materials_count": len(bpy.data.materials),
            }

            # Collect minimal object information (limit to first 10 objects)
            for i, obj in enumerate(bpy.context.scene.objects):
                if i >= 10:  # Reduced from 20 to 10
                    break

                obj_info = {
                    "name": obj.name,
                    "type": obj.type,
                    # Only include basic location data
                    "location": [round(float(obj.location.x), 2),
                                round(float(obj.location.y), 2),
                                round(float(obj.location.z), 2)],
                }
                scene_info["objects"].append(obj_info)

            print(f"Scene info collected: {len(scene_info['objects'])} objects")
            return scene_info
        except Exception as e:
            print(f"Error in get_scene_info: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}

    @staticmethod
    def _get_aabb(obj):
        """ Returns the world-space axis-aligned bounding box (AABB) of an object. """
        if obj.type != 'MESH':
            raise TypeError("Object must be a mesh")

        # Get the bounding box corners in local space
        local_bbox_corners = [mathutils.Vector(corner) for corner in obj.bound_box]

        # Convert to world coordinates
        world_bbox_corners = [obj.matrix_world @ corner for corner in local_bbox_corners]

        # Compute axis-aligned min/max coordinates
        min_corner = mathutils.Vector(map(min, zip(*world_bbox_corners)))
        max_corner = mathutils.Vector(map(max, zip(*world_bbox_corners)))

        return [
            [*min_corner], [*max_corner]
        ]



    def get_object_info(self, name):
        """Get detailed information about a specific object"""
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object not found: {name}")

        # Basic object info
        obj_info = {
            "name": obj.name,
            "type": obj.type,
            "location": [obj.location.x, obj.location.y, obj.location.z],
            "rotation": [obj.rotation_euler.x, obj.rotation_euler.y, obj.rotation_euler.z],
            "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
            "visible": obj.visible_get(),
            "materials": [],
        }

        if obj.type == "MESH":
            bounding_box = self._get_aabb(obj)
            obj_info["world_bounding_box"] = bounding_box

        # Add material slots
        for slot in obj.material_slots:
            if slot.material:
                obj_info["materials"].append(slot.material.name)

        # Add mesh data if applicable
        if obj.type == 'MESH' and obj.data:
            mesh = obj.data
            obj_info["mesh"] = {
                "vertices": len(mesh.vertices),
                "edges": len(mesh.edges),
                "polygons": len(mesh.polygons),
            }

        return obj_info
    
    def in_5_seconds(self):
        return None

    def get_viewport_screenshot(self, max_size=800, filepath=None, format="png", selected_object_outline=True, shading="material"):
        """
        Capture a screenshot of the current 3D viewport and save it to the specified path.

        Parameters:
        - max_size: Maximum size in pixels for the largest dimension of the image
        - filepath: Path where to save the screenshot file
        - format: Image format (png, jpg, etc.)
        - selected_object_outline: Whether to show selection outlines (default: True)
        - shading: Viewport shading mode ('material' or 'rendered', default: 'material')

        Returns success/error status
        """
        try:
            if not filepath:
                return {"error": "No filepath provided"}

            # Store original viewport settings and set up for clean screenshots
            original_viewport_settings = {}
            area = None
            space = None
            try:
                for a in bpy.context.screen.areas:
                    if a.type == 'VIEW_3D':
                        area = a
                        for s in area.spaces:
                            if s.type == 'VIEW_3D':
                                space = s
                                # Store original UI settings
                                original_viewport_settings['show_region_header'] = space.show_region_header
                                original_viewport_settings['show_region_toolbar'] = space.show_region_toolbar
                                original_viewport_settings['show_region_ui'] = space.show_region_ui
                                original_viewport_settings['show_gizmo'] = space.show_gizmo
                                original_viewport_settings['shading_type'] = space.shading.type
                                if hasattr(space, 'overlay'):
                                    original_viewport_settings['show_overlays'] = space.overlay.show_overlays
                                
                                # Hide UI elements for clean screenshots
                                space.show_region_header = False
                                space.show_region_toolbar = False
                                space.show_region_ui = False
                                space.show_gizmo = False
                                
                                # Handle overlays based on selected_object_outline parameter
                                if hasattr(space, 'overlay'):
                                    overlay = space.overlay
                                    if selected_object_outline:
                                        overlay.show_overlays = True
                                        if hasattr(overlay, 'show_outline_selected'):
                                            overlay.show_outline_selected = True
                                        for attr in [
                                            'show_cursor', 'show_floor', 'show_axis_x', 'show_axis_y', 'show_axis_z',
                                            'show_object_origins', 'show_stats', 'show_text', 'show_extras', 'show_bones',
                                            'show_relationship_lines', 'show_motion_paths', 'show_wireframes',
                                            'show_face_orientation', 'show_ortho_grid'
                                        ]:
                                            if hasattr(overlay, attr):
                                                setattr(overlay, attr, False)
                                    else:
                                        overlay.show_overlays = False
                                
                                # Set viewport shading based on parameter
                                target_shading = 'RENDERED' if shading.lower() == 'rendered' else 'MATERIAL'
                                space.shading.type = target_shading
                                break
                        break
                
                # Store original background/gradient settings and set to white
                gradients = bpy.context.preferences.themes[0].view_3d.space.gradients
                original_viewport_settings['background_type'] = gradients.background_type
                original_viewport_settings['high_gradient'] = tuple(gradients.high_gradient)
                
                # Set background to white single color
                gradients.background_type = 'SINGLE_COLOR'
                gradients.high_gradient = (1.0, 1.0, 1.0)
                
                # Force UI update
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                
            except Exception as e:
                print(f"Warning: Could not fully configure viewport settings: {e}")

            if not area:
                return {"error": "No 3D viewport found"}

            # Store original render settings
            scene = bpy.context.scene
            original_filepath = scene.render.filepath
            original_format = scene.render.image_settings.file_format
            original_res_x = scene.render.resolution_x
            original_res_y = scene.render.resolution_y
            original_res_percentage = scene.render.resolution_percentage
            original_engine = scene.render.engine

            try:
                # Set up render settings
                scene.render.filepath = filepath
                scene.render.image_settings.file_format = format.upper()
                
                # Get viewport dimensions and scale to max_size
                region = None
                for r in area.regions:
                    if r.type == 'WINDOW':
                        region = r
                        break
                
                if region:
                    # Render at 2x resolution for supersampling, then downsample for sharper results
                    supersample_factor = 3
                    vp_width, vp_height = region.width, region.height
                    if max(vp_width, vp_height) > max_size:
                        scale_factor = max_size / max(vp_width, vp_height)
                        target_width = int(vp_width * scale_factor)
                        target_height = int(vp_height * scale_factor)
                    else:
                        target_width = vp_width
                        target_height = vp_height
                    
                    # Render at 2x the target size
                    scene.render.resolution_x = target_width * supersample_factor
                    scene.render.resolution_y = target_height * supersample_factor
                    scene.render.resolution_percentage = 100
                
                # For 'rendered' shading, temporarily switch to EEVEE which renders instantly
                # This allows using viewport capture (preserves view angle and outlines)
                # without Cycles progressive rendering issues
                if shading.lower() == 'rendered':
                    scene.render.engine = 'BLENDER_EEVEE_NEXT'
                
                # Disable GC during OpenGL render to prevent race condition with Gradio's asyncio thread
                gc.disable()
                try:
                    with bpy.context.temp_override(
                        window=bpy.context.window,
                        screen=bpy.context.screen,
                        area=area,
                        region=region,
                        space_data=space,
                    ):
                        bpy.ops.render.opengl(write_still=True, view_context=True)
                finally:
                    gc.enable()
                
                # Downsample to target size with high-quality LANCZOS filter
                if region:
                    img = Image.open(filepath)
                    img_downsampled = img.resize((target_width, target_height), Image.LANCZOS)
                    img_downsampled.save(filepath)
                    img.close()
                    img_downsampled.close()
                    width = target_width
                    height = target_height
                else:
                    width = scene.render.resolution_x
                    height = scene.render.resolution_y
                
            finally:
                # Restore original render settings
                scene.render.filepath = original_filepath
                scene.render.image_settings.file_format = original_format
                scene.render.resolution_x = original_res_x
                scene.render.resolution_y = original_res_y
                scene.render.resolution_percentage = original_res_percentage
                scene.render.engine = original_engine
                
                # Restore viewport to original state
                try:
                    if space:
                        # Restore UI regions to original or default values
                        space.show_region_header = original_viewport_settings.get('show_region_header', True)
                        space.show_region_toolbar = original_viewport_settings.get('show_region_toolbar', True)
                        space.show_region_ui = original_viewport_settings.get('show_region_ui', True)
                        space.show_gizmo = original_viewport_settings.get('show_gizmo', True)
                        
                        # Restore overlays
                        if hasattr(space, 'overlay'):
                            space.overlay.show_overlays = original_viewport_settings.get('show_overlays', True)
                            space.overlay.show_floor = True
                            space.overlay.show_axis_x = True
                            space.overlay.show_axis_y = True
                            space.overlay.show_cursor = True
                        
                        # Restore shading type
                        space.shading.type = original_viewport_settings.get('shading_type', 'MATERIAL')
                    
                    # Restore original background/gradient settings
                    gradients = bpy.context.preferences.themes[0].view_3d.space.gradients
                    if 'background_type' in original_viewport_settings:
                        gradients.background_type = original_viewport_settings['background_type']
                    if 'high_gradient' in original_viewport_settings:
                        gradients.high_gradient = original_viewport_settings['high_gradient']
                    
                    # Force UI update
                    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                    
                except Exception as e:
                    print(f"Warning: Could not fully restore viewport settings: {e}")

            return {
                "success": True,
                "width": width,
                "height": height,
                "filepath": filepath
            }

        except Exception as e:
            return {"error": str(e)}



    def execute_code(self, code):
        """Execute arbitrary Blender Python code"""
        # This is powerful but potentially dangerous - use with caution
        try:
            # Create a local namespace for execution
            namespace = {"bpy": bpy}

            # Capture stdout during execution, and return it as result
            capture_buffer = io.StringIO()
            with redirect_stdout(capture_buffer):
                exec(code, namespace)

            captured_output = capture_buffer.getvalue()
            return {"executed": True, "result": captured_output}
        except Exception as e:
            raise Exception(f"Code execution error: {str(e)}")

    def get_object_image(self, direction='front', view_distance='close', max_size=800, filepath=None, format="png", selected_object_outline=True, shading="material"):
        """
        Align the 3D viewport to show the specified direction of the currently active object,
        then frame it, adjust the zoom distance, and take a screenshot.
        
        Parameters:
        direction (str): The direction to align to. Options: 'front', 'back', 'top', 'bottom', 'left', 'right' (default: 'front')
        view_distance (str): The zoom distance after framing. Options: 'close', 'medium', 'far' (default: 'close')
                            - 'close': zoom out 1 step (close-up view)
                            - 'medium': zoom out 3 steps (see surrounding objects that are close)
                            - 'far': zoom out 6 steps (see even more surrounding objects)
        max_size (int): Maximum size in pixels for the largest dimension of the image (default: 800)
        filepath (str): Path where to save the screenshot file
        format (str): Image format (png, jpg, etc.) (default: "png")
        selected_object_outline (bool): Whether to show selection outlines (default: True)
        shading (str): Viewport shading mode ('material' or 'rendered', default: 'material')
        
        Returns:
        dict: Success/error status with screenshot information
        """
        
        # Dictionary to map direction strings to Blender's view axis types
        direction_map = {
            'front': 'FRONT',
            'back': 'BACK',
            'top': 'TOP',
            'bottom': 'BOTTOM',
            'left': 'LEFT',
            'right': 'RIGHT'
        }
        
        # Dictionary to map view distance to zoom steps
        distance_map = {
            'close': 1,
            'medium': 3,
            'far': 6
        }
        
        # Convert inputs to lowercase for case-insensitive input
        direction = direction.lower()
        view_distance = view_distance.lower()
        
        # Check if direction is valid
        if direction not in direction_map:
            return {"error": f"Invalid direction '{direction}'. Valid options are: {', '.join(direction_map.keys())}"}
        
        # Check if view_distance is valid
        if view_distance not in distance_map:
            return {"error": f"Invalid view_distance '{view_distance}'. Valid options are: {', '.join(distance_map.keys())}"}
        
        # Get the active object
        active_object = bpy.context.active_object
        
        if active_object is None:
            return {"error": "No object is currently selected/active"}
        
        # Make sure the object is selected for framing
        if not active_object.select_get():
            active_object.select_set(True)
        
        # Get all 3D viewports and perform the operations
        success = False
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        # Get the 3D viewport context
                        override = bpy.context.copy()
                        override['area'] = area
                        override['region'] = region
                        override['space_data'] = area.spaces.active
                        
                        with bpy.context.temp_override(**override):
                            # Step 1: Align view to specified direction of active object
                            bpy.ops.view3d.view_axis(type=direction_map[direction], align_active=True)
                            
                            # Step 2: Frame the selected object (equivalent to Numpad .)
                            bpy.ops.view3d.view_selected()
                            
                            # Step 3: Zoom out based on view_distance
                            zoom_steps = distance_map[view_distance]
                            for _ in range(zoom_steps):
                                bpy.ops.view3d.zoom(delta=-1)
                        
                        success = True
                        break
        
        if not success:
            return {"error": "No 3D viewport found"}
        
        # Now take a screenshot using the existing get_viewport_screenshot method
        screenshot_result = self.get_viewport_screenshot(
            max_size=max_size,
            filepath=filepath,
            format=format,
            selected_object_outline=selected_object_outline,
            shading=shading
        )
        
        # Check if screenshot was successful
        if "error" in screenshot_result:
            return {"error": f"Failed to capture screenshot: {screenshot_result['error']}"}
        
        # Combine alignment success message with screenshot result
        zoom_info = f" and zoomed out {distance_map[view_distance]} steps" if distance_map[view_distance] > 0 else ""
        alignment_message = f"Successfully aligned view to {direction} of '{active_object.name}', framed object{zoom_info} ({view_distance} distance)"
        
        # Return combined result
        return {
            "success": True,
            "message": f"{alignment_message} and captured screenshot",
            "alignment": alignment_message,
            "screenshot": screenshot_result,
            "width": screenshot_result.get("width"),
            "height": screenshot_result.get("height"),
            "filepath": screenshot_result.get("filepath")
        }


    def download_polyhaven_asset(self, asset_id, asset_type, resolution="2k", file_format=None):
        try:
            # First get the files information
            files_response = requests.get(f"https://api.polyhaven.com/files/{asset_id}", headers=REQ_HEADERS)
            if files_response.status_code != 200:
                return {"error": f"Failed to get asset files: {files_response.status_code}"}

            files_data = files_response.json()

            # Handle different asset types
            if asset_type == "hdris":
                # For HDRIs, download the .hdr or .exr file
                if not file_format:
                    file_format = "hdr"  # Default format for HDRIs

                if "hdri" in files_data and resolution in files_data["hdri"] and file_format in files_data["hdri"][resolution]:
                    file_info = files_data["hdri"][resolution][file_format]
                    file_url = file_info["url"]

                    # For HDRIs, we need to save to a temporary file first
                    # since Blender can't properly load HDR data directly from memory
                    with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as tmp_file:
                        # Download the file
                        response = requests.get(file_url, headers=REQ_HEADERS)
                        if response.status_code != 200:
                            return {"error": f"Failed to download HDRI: {response.status_code}"}

                        tmp_file.write(response.content)
                        tmp_path = tmp_file.name

                    try:
                        # Create a new unique world for this scene to allow different HDRIs per scene
                        scene = bpy.context.scene
                        world_name = f"World_{scene.name}"
                        
                        # Check if a world with this name already exists, remove it to avoid duplicates
                        if world_name in bpy.data.worlds:
                            bpy.data.worlds.remove(bpy.data.worlds[world_name])
                        
                        # Create a new world with the unique name
                        world = bpy.data.worlds.new(world_name)
                        world.use_nodes = True
                        node_tree = world.node_tree

                        # Clear existing nodes (new world should be empty, but just in case)
                        for node in node_tree.nodes:
                            node_tree.nodes.remove(node)

                        # Create nodes
                        tex_coord = node_tree.nodes.new(type='ShaderNodeTexCoord')
                        tex_coord.location = (-800, 0)

                        mapping = node_tree.nodes.new(type='ShaderNodeMapping')
                        mapping.location = (-600, 0)

                        # Load the image from the temporary file
                        env_tex = node_tree.nodes.new(type='ShaderNodeTexEnvironment')
                        env_tex.location = (-400, 0)
                        env_tex.image = bpy.data.images.load(tmp_path)

                        # Use a color space that exists in all Blender versions
                        if file_format.lower() == 'exr':
                            # Try to use Linear color space for EXR files
                            try:
                                env_tex.image.colorspace_settings.name = 'Linear'
                            except:
                                # Fallback to Non-Color if Linear isn't available
                                env_tex.image.colorspace_settings.name = 'Non-Color'
                        else:  # hdr
                            # For HDR files, try these options in order
                            for color_space in ['Linear', 'Linear Rec.709', 'Non-Color']:
                                try:
                                    env_tex.image.colorspace_settings.name = color_space
                                    break  # Stop if we successfully set a color space
                                except:
                                    continue

                        background = node_tree.nodes.new(type='ShaderNodeBackground')
                        background.location = (-200, 0)

                        output = node_tree.nodes.new(type='ShaderNodeOutputWorld')
                        output.location = (0, 0)

                        # Connect nodes
                        node_tree.links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
                        node_tree.links.new(mapping.outputs['Vector'], env_tex.inputs['Vector'])
                        node_tree.links.new(env_tex.outputs['Color'], background.inputs['Color'])
                        node_tree.links.new(background.outputs['Background'], output.inputs['Surface'])

                        # Set as active world
                        bpy.context.scene.world = world

                        # Pack the HDRI image into the blend file so it persists after temp cleanup
                        try:
                            if env_tex.image.packed_file is None:
                                env_tex.image.pack()
                                print(f"Packed HDRI image: {env_tex.image.name}")
                        except Exception as pack_error:
                            print(f"Warning: Failed to pack HDRI image: {pack_error}")

                        # Clean up temporary file
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass

                        return {
                            "success": True,
                            "message": f"HDRI {asset_id} imported successfully for scene '{scene.name}'",
                            "image_name": env_tex.image.name,
                            "world_name": world_name,
                            "scene_name": scene.name
                        }
                    except Exception as e:
                        return {"error": f"Failed to set up HDRI in Blender: {str(e)}"}
                else:
                    return {"error": f"Requested resolution or format not available for this HDRI"}

            elif asset_type == "textures":
                if not file_format:
                    file_format = "jpg"  # Default format for textures

                downloaded_maps = {}

                try:
                    for map_type in files_data:
                        if map_type not in ["blend", "gltf"]:  # Skip non-texture files
                            if resolution in files_data[map_type] and file_format in files_data[map_type][resolution]:
                                file_info = files_data[map_type][resolution][file_format]
                                file_url = file_info["url"]

                                # Use NamedTemporaryFile like we do for HDRIs
                                with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as tmp_file:
                                    # Download the file
                                    response = requests.get(file_url, headers=REQ_HEADERS)
                                    if response.status_code == 200:
                                        tmp_file.write(response.content)
                                        tmp_path = tmp_file.name

                                        # Load image from temporary file
                                        image = bpy.data.images.load(tmp_path)
                                        image.name = f"{asset_id}_{map_type}.{file_format}"

                                        # Pack the image into .blend file
                                        image.pack()

                                        # Set color space based on map type
                                        if map_type in ['color', 'diffuse', 'albedo']:
                                            try:
                                                image.colorspace_settings.name = 'sRGB'
                                            except:
                                                pass
                                        else:
                                            try:
                                                image.colorspace_settings.name = 'Non-Color'
                                            except:
                                                pass

                                        downloaded_maps[map_type] = image

                                        # Clean up temporary file
                                        try:
                                            os.unlink(tmp_path)
                                        except:
                                            pass

                    if not downloaded_maps:
                        return {"error": f"No texture maps found for the requested resolution and format"}

                    # Create a new material with the downloaded textures
                    mat = bpy.data.materials.new(name=asset_id)
                    mat.use_nodes = True
                    nodes = mat.node_tree.nodes
                    links = mat.node_tree.links

                    # Clear default nodes
                    for node in nodes:
                        nodes.remove(node)

                    # Create output node
                    output = nodes.new(type='ShaderNodeOutputMaterial')
                    output.location = (300, 0)

                    # Create principled BSDF node
                    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
                    principled.location = (0, 0)
                    links.new(principled.outputs[0], output.inputs[0])

                    # Add texture nodes based on available maps
                    tex_coord = nodes.new(type='ShaderNodeTexCoord')
                    tex_coord.location = (-800, 0)

                    mapping = nodes.new(type='ShaderNodeMapping')
                    mapping.location = (-600, 0)
                    mapping.vector_type = 'TEXTURE'  # Changed from default 'POINT' to 'TEXTURE'
                    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])

                    # Position offset for texture nodes
                    x_pos = -400
                    y_pos = 300

                    # Connect different texture maps
                    for map_type, image in downloaded_maps.items():
                        tex_node = nodes.new(type='ShaderNodeTexImage')
                        tex_node.location = (x_pos, y_pos)
                        tex_node.image = image

                        # Set color space based on map type
                        if map_type.lower() in ['color', 'diffuse', 'albedo']:
                            try:
                                tex_node.image.colorspace_settings.name = 'sRGB'
                            except:
                                pass  # Use default if sRGB not available
                        else:
                            try:
                                tex_node.image.colorspace_settings.name = 'Non-Color'
                            except:
                                pass  # Use default if Non-Color not available

                        links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])

                        # Connect to appropriate input on Principled BSDF
                        if map_type.lower() in ['color', 'diffuse', 'albedo']:
                            links.new(tex_node.outputs['Color'], principled.inputs['Base Color'])
                        elif map_type.lower() in ['roughness', 'rough']:
                            links.new(tex_node.outputs['Color'], principled.inputs['Roughness'])
                        elif map_type.lower() in ['metallic', 'metalness', 'metal']:
                            links.new(tex_node.outputs['Color'], principled.inputs['Metallic'])
                        elif map_type.lower() in ['normal', 'nor']:
                            # Add normal map node
                            normal_map = nodes.new(type='ShaderNodeNormalMap')
                            normal_map.location = (x_pos + 200, y_pos)
                            links.new(tex_node.outputs['Color'], normal_map.inputs['Color'])
                            links.new(normal_map.outputs['Normal'], principled.inputs['Normal'])
                        elif map_type in ['displacement', 'disp', 'height']:
                            # Add displacement node
                            disp_node = nodes.new(type='ShaderNodeDisplacement')
                            disp_node.location = (x_pos + 200, y_pos - 200)
                            links.new(tex_node.outputs['Color'], disp_node.inputs['Height'])
                            links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])

                        y_pos -= 250

                    return {
                        "success": True,
                        "message": f"Texture {asset_id} imported as material",
                        "material": mat.name,
                        "maps": list(downloaded_maps.keys())
                    }

                except Exception as e:
                    return {"error": f"Failed to process textures: {str(e)}"}

            elif asset_type == "models":
                # For models, prefer glTF format if available
                if not file_format:
                    file_format = "gltf"  # Default format for models

                if file_format in files_data and resolution in files_data[file_format]:
                    file_info = files_data[file_format][resolution][file_format]
                    file_url = file_info["url"]

                    # Create a temporary directory to store the model and its dependencies
                    temp_dir = tempfile.mkdtemp()
                    main_file_path = ""

                    try:
                        # Download the main model file
                        main_file_name = file_url.split("/")[-1]
                        main_file_path = os.path.join(temp_dir, main_file_name)

                        response = requests.get(file_url, headers=REQ_HEADERS)
                        if response.status_code != 200:
                            return {"error": f"Failed to download model: {response.status_code}"}

                        with open(main_file_path, "wb") as f:
                            f.write(response.content)

                        # Check for included files and download them
                        if "include" in file_info and file_info["include"]:
                            for include_path, include_info in file_info["include"].items():
                                # Get the URL for the included file - this is the fix
                                include_url = include_info["url"]

                                # Create the directory structure for the included file
                                include_file_path = os.path.join(temp_dir, include_path)
                                os.makedirs(os.path.dirname(include_file_path), exist_ok=True)

                                # Download the included file
                                include_response = requests.get(include_url, headers=REQ_HEADERS)
                                if include_response.status_code == 200:
                                    with open(include_file_path, "wb") as f:
                                        f.write(include_response.content)
                                else:
                                    print(f"Failed to download included file: {include_path}")

                        # Import the model into Blender
                        if file_format == "gltf" or file_format == "glb":
                            bpy.ops.import_scene.gltf(filepath=main_file_path)
                        elif file_format == "fbx":
                            bpy.ops.import_scene.fbx(filepath=main_file_path)
                        elif file_format == "obj":
                            bpy.ops.import_scene.obj(filepath=main_file_path)
                        elif file_format == "blend":
                            # For blend files, we need to append or link
                            with bpy.data.libraries.load(main_file_path, link=False) as (data_from, data_to):
                                data_to.objects = data_from.objects

                            # Link the objects to the scene
                            for obj in data_to.objects:
                                if obj is not None:
                                    bpy.context.collection.objects.link(obj)
                        else:
                            return {"error": f"Unsupported model format: {file_format}"}

                        # Get the names of imported objects
                        imported_objects = [obj.name for obj in bpy.context.selected_objects]

                        return {
                            "success": True,
                            "message": f"Model {asset_id} imported successfully",
                            "imported_objects": imported_objects
                        }
                    except Exception as e:
                        return {"error": f"Failed to import model: {str(e)}"}
                    finally:
                        # Clean up temporary directory
                        with suppress(Exception):
                            shutil.rmtree(temp_dir)
                else:
                    return {"error": f"Requested format or resolution not available for this model"}

            else:
                return {"error": f"Unsupported asset type: {asset_type}"}

        except Exception as e:
            return {"error": f"Failed to download asset: {str(e)}"}


    def lighting_designer(self, scene_description=None, asset_id=None, categories_limitation=None, anyllm_api_key=None, anyllm_api_base=None, anyllm_provider="gemini", vision_model="gemini-3-flash-preview"):
        """
        Design lighting for a scene using AI-selected HDRIs from Polyhaven.
        
        Uses semantic search with pre-computed embeddings for efficient HDRI matching,
        with optional LLM-based reranking for improved relevance.
        
        Args:
            scene_description (str, optional): Description of the scene lighting needs.
                                               Ignored if asset_id is provided.
            asset_id (str, optional): Polyhaven HDRI asset ID to use directly,
                                      bypassing AI selection.
            categories_limitation (list, optional): List of category strings. If provided,
                                                    only HDRIs containing ALL specified categories
                                                    will be returned (e.g., ["sunrise-sunset", "pure skies"]).
            anyllm_api_key (str, optional): API key for LLM service (for reranking)
            anyllm_api_base (str, optional): API base URL for LLM service
            anyllm_provider (str): LLM provider (default: "gemini")
            vision_model (str): Vision model for LLM reranking (default: "gemini-3-flash-preview")
            
        Returns:
            dict: Result of the lighting design process info
        """
        try:
            # If asset_id is provided, use it directly
            if asset_id:
                print(f"Using provided asset_id: {asset_id}")
                best_hdri = {"id": asset_id, "name": asset_id}
            else:
                # Validate scene_description is provided
                if not scene_description:
                    return {"error": "Either scene_description or asset_id must be provided"}
                
                print(f"Starting lighting_designer with scene description: {scene_description}")
                if categories_limitation:
                    print(f"Categories limitation: {categories_limitation}")
                
                # Search for HDRIs using semantic search with LLM reranking
                try:
                    search_results = search_polyhaven_assets(
                        asset_type="hdris",
                        description=scene_description,
                        returned_count=5,
                        threshold_score=0.4,
                        rerank_with_llm=True,
                        threshold_llm=1,
                        anyllm_api_key=anyllm_api_key,
                        anyllm_api_base=anyllm_api_base,
                        anyllm_provider=anyllm_provider,
                        vision_model=vision_model,
                        categories_limitation=categories_limitation,
                    )
                    
                    if not search_results:
                        # Fallback: try without LLM reranking with lower threshold
                        print("No results with LLM reranking, trying without...")
                        search_results = search_polyhaven_assets(
                            asset_type="hdris",
                            description=scene_description,
                            returned_count=5,
                            threshold_score=0.4,
                            rerank_with_llm=False,
                            categories_limitation=categories_limitation,
                        )
                    
                    if not search_results:
                        return {"error": "No suitable HDRIs found for the given scene description"}
                    
                    # Get the best matching HDRI (first result after reranking)
                    best_hdri = search_results[0]
                    
                    print(f"Selected HDRI: {best_hdri.get('id')} (score: {best_hdri.get('combined_score', 'N/A')}, llm_score: {best_hdri.get('llm_score', 'N/A')})")
                    
                except Exception as e:
                    print(f"Error in search_polyhaven_assets: {e}")
                    traceback.print_exc()
                    return {"error": f"Failed to search HDRIs: {str(e)}"}
            
            # Download and import the HDRI using the existing download_polyhaven_asset method
            try:
                hdri_asset_id = best_hdri.get("id")
                # Use 1k for indoor scenes, 2k for outdoor scenes
                is_indoor = categories_limitation and "indoor" in categories_limitation
                hdri_resolution = "1k" if is_indoor else "2k"
                result = self.download_polyhaven_asset(hdri_asset_id, "hdris", resolution=hdri_resolution, file_format="hdr")
                if "error" in result:
                    return {"error": f"Failed to download and import HDRI: {result['error']}"}
                
                print(f"Successfully imported HDRI: {hdri_asset_id} (resolution: {hdri_resolution})")
                
                # Set film transparency based on indoor/outdoor scene (current scene only)
                bpy.context.scene.render.film_transparent = is_indoor
                print(f"Set film_transparent to {is_indoor} ({'indoor' if is_indoor else 'outdoor'} scene)")
                
                return {
                    "success": True,
                    "message": "lighting design completed successfully.",
                    "asset_id": hdri_asset_id,
                    "hdri_name": best_hdri.get("name", ""),
                    "combined_score": best_hdri.get("combined_score"),
                    "llm_score": best_hdri.get("llm_score"),
                }
                
            except Exception as e:
                print(f"Error downloading and importing HDRI: {e}")
                traceback.print_exc()
                return {"error": f"Failed to download and import HDRI: {str(e)}"}
                
        except Exception as e:
            print(f"Error in lighting_designer: {e}")
            traceback.print_exc()
            return {"error": f"lighting designer failed: {str(e)}"}


    def set_render(self, engine="EEVEE", samples=None, persistent_data=True):
        """
        Configure render settings for all scenes in the Blender file.
        
        Args:
            engine (str): Render engine to use. Options: "EEVEE" or "Cycles" (case-insensitive).
            samples (int, optional): Number of render samples.
                - For EEVEE: default is 64
                - For Cycles: default is 512 (render), viewport uses 128
            persistent_data (bool): Whether to enable persistent data for faster re-renders. Default: True.
        
        Returns:
            dict: Result of the render configuration
        """
        try:
            engine = engine.upper()
            
            if engine not in ["EEVEE", "CYCLES"]:
                return {"error": f"Invalid engine '{engine}'. Valid options are: EEVEE, Cycles"}
            
            configured_scenes = []
            
            for scene in bpy.data.scenes:
                if engine == "EEVEE":
                    # Set render samples (default 64)
                    render_samples = samples if samples is not None else 64
                    
                    # Set render engine to EEVEE
                    scene.render.engine = 'BLENDER_EEVEE_NEXT'
                    
                    # Set EEVEE samples
                    scene.eevee.taa_render_samples = render_samples
                    
                    # Set persistent data
                    scene.render.use_persistent_data = persistent_data
                    
                    configured_scenes.append({
                        "scene": scene.name,
                        "engine": "EEVEE",
                        "render_samples": render_samples,
                        "persistent_data": persistent_data,
                    })
                    
                elif engine == "CYCLES":
                    # Set render samples (default 512 for render, 128 for viewport)
                    render_samples = samples if samples is not None else 512
                    viewport_samples = 128
                    
                    # Set render engine to Cycles
                    scene.render.engine = 'CYCLES'
                    
                    # Set feature set to experimental
                    scene.cycles.feature_set = 'EXPERIMENTAL'
                    
                    # Try to set device to GPU
                    # First, enable GPU compute devices in preferences
                    prefs = bpy.context.preferences.addons['cycles'].preferences
                    
                    # Try different GPU compute device types
                    gpu_device_type = None
                    for device_type in ['METAL', 'CUDA', 'OPTIX', 'HIP', 'ONEAPI']:
                        try:
                            prefs.compute_device_type = device_type
                            prefs.get_devices()
                            # Check if any devices are available
                            if prefs.devices:
                                # Enable all available GPU devices
                                for device in prefs.devices:
                                    device.use = True
                                gpu_device_type = device_type
                                break
                        except:
                            continue
                    
                    if gpu_device_type:
                        scene.cycles.device = 'GPU'
                    else:
                        scene.cycles.device = 'CPU'
                    
                    # Viewport settings: Max samples 128, Denoise, Noise Threshold 0.1
                    scene.cycles.preview_samples = viewport_samples
                    scene.cycles.use_preview_denoising = True
                    scene.cycles.preview_denoising_input_passes = 'RGB_ALBEDO_NORMAL'
                    scene.cycles.use_preview_adaptive_sampling = True
                    scene.cycles.preview_adaptive_threshold = 0.1
                    
                    # Render settings: Max samples, Denoise, Noise Threshold 0.05
                    scene.cycles.samples = render_samples
                    scene.cycles.use_denoising = True
                    scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
                    scene.cycles.use_adaptive_sampling = True
                    scene.cycles.adaptive_threshold = 0.05
                    
                    # Set persistent data
                    scene.render.use_persistent_data = persistent_data
                    
                    configured_scenes.append({
                        "scene": scene.name,
                        "engine": "Cycles",
                        "device": gpu_device_type if gpu_device_type else "CPU",
                        "feature_set": "EXPERIMENTAL",
                        "render_samples": render_samples,
                        "viewport_samples": viewport_samples,
                        "render_noise_threshold": 0.05,
                        "viewport_noise_threshold": 0.1,
                        "persistent_data": persistent_data,
                    })
            
            # Set viewport shading to Rendered for all 3D viewports
            for screen in bpy.data.screens:
                for area in screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                space.shading.type = 'RENDERED'
            
            return {
                "success": True,
                "message": f"Render settings configured for {len(configured_scenes)} scene(s)",
                "scenes": configured_scenes,
            }
            
        except Exception as e:
            print(f"Error in set_render: {e}")
            traceback.print_exc()
            return {"error": f"Failed to set render: {str(e)}"}


    def environment_artist(self, ground_description=None, asset_id=None, categories_limitation=None, width=10.0, depth=10.0,
                           wall_description=None, wall_asset_id=None, wall_categories_limitation=None,
                           wall_x=None, wall_x_negative=None, wall_y=None, wall_y_negative=None, wall_z=None,
                           anyllm_api_key=None, anyllm_api_base=None, anyllm_provider="gemini", vision_model="gemini-3-flash-preview"):
        """
        Create a ground plane and optionally indoor walls using textures from Polyhaven.
        
        Uses semantic search with pre-computed embeddings for efficient texture matching,
        with optional LLM-based reranking for improved relevance.
        
        Args:
            ground_description (str, optional): Description of the ground/terrain.
                                                Ignored if asset_id is provided.
            asset_id (str, optional): Polyhaven texture asset ID to use directly,
                                      bypassing AI selection.
            categories_limitation (list, optional): List of category strings. If provided,
                                                    only textures containing ALL specified categories
                                                    will be returned (e.g., ["floor", "cobblestone"]).
            width (float): Width of the ground plane in meters (default: 10.0)
            depth (float): Depth of the ground plane in meters (default: 10.0)
            wall_description (str, optional): Description of the wall texture for indoor scenes.
                                              Ignored if wall_asset_id is provided.
            wall_asset_id (str, optional): Polyhaven texture asset ID for walls,
                                           bypassing AI selection.
            wall_categories_limitation (list, optional): Category filter for wall textures.
            wall_x (float, optional): Positive X boundary for walls.
            wall_x_negative (float, optional): Negative X boundary for walls.
            wall_y (float, optional): Positive Y boundary for walls.
            wall_y_negative (float, optional): Negative Y boundary for walls.
            wall_z (float, optional): Height of walls in meters.
            anyllm_api_key (str, optional): API key for LLM service (for reranking)
            anyllm_api_base (str, optional): API base URL for LLM service
            anyllm_provider (str): LLM provider (default: "gemini")
            vision_model (str): Vision model for LLM reranking (default: "gemini-3-flash-preview")
            
        Returns:
            dict: Result of the environment creation process
        """
        try:
            # Remove existing environment objects from the current scene first
            # Objects may have .001, .002 suffixes if same name exists in different scenes
            current_scene = bpy.context.scene
            env_object_base_names = ["ground_plane", "wall_x", "wall_x_negative", "wall_y", "wall_y_negative", "roof"]
            
            objects_to_remove = []
            for obj in current_scene.objects:
                # Check if object name matches any base name or has a suffix like .001
                obj_base_name = obj.name.split('.')[0] if '.' in obj.name else obj.name
                if obj_base_name in env_object_base_names:
                    objects_to_remove.append(obj)
            
            # Remove the objects
            for obj in objects_to_remove:
                print(f"Removing existing environment object from scene '{current_scene.name}': {obj.name}")
                bpy.data.objects.remove(obj, do_unlink=True)
            
            if objects_to_remove:
                print(f"Removed {len(objects_to_remove)} existing environment objects from scene '{current_scene.name}'")
            
            # If asset_id is provided, use it directly
            if asset_id:
                print(f"Using provided asset_id: {asset_id}")
                best_texture = {"id": asset_id, "name": asset_id, "dimensions": [2000, 2000]}
            else:
                # Validate ground_description is provided
                if not ground_description:
                    return {"error": "Either ground_description or asset_id must be provided"}
                
                print(f"Starting environment_artist with ground description: {ground_description}")
                if categories_limitation:
                    print(f"Categories limitation: {categories_limitation}")
                
                # Search for textures using semantic search with LLM reranking
                try:
                    search_results = search_polyhaven_assets(
                        asset_type="textures",
                        description=ground_description,
                        returned_count=5,
                        threshold_score=0.4,
                        rerank_with_llm=True,
                        threshold_llm=1,
                        anyllm_api_key=anyllm_api_key,
                        anyllm_api_base=anyllm_api_base,
                        anyllm_provider=anyllm_provider,
                        vision_model=vision_model,
                        categories_limitation=categories_limitation,
                    )
                    
                    if not search_results:
                        # Fallback: try without LLM reranking with lower threshold
                        print("No results with LLM reranking, trying without...")
                        search_results = search_polyhaven_assets(
                            asset_type="textures",
                            description=ground_description,
                            returned_count=5,
                            threshold_score=0.4,
                            rerank_with_llm=False,
                            categories_limitation=categories_limitation,
                        )
                    
                    if not search_results:
                        return {"error": "No suitable textures found for the given ground description"}
                    
                    # Get the best matching texture (first result after reranking)
                    best_texture = search_results[0]
                    
                    print(f"Selected texture: {best_texture.get('id')} (score: {best_texture.get('combined_score', 'N/A')}, llm_score: {best_texture.get('llm_score', 'N/A')})")
                    
                except Exception as e:
                    print(f"Error in search_polyhaven_assets: {e}")
                    traceback.print_exc()
                    return {"error": f"Failed to search textures: {str(e)}"}
            
            # Extract asset info
            texture_asset_id = best_texture.get("id")
            dimensions = best_texture.get("dimensions", [2000, 2000])
            
            # Convert dimensions to meters (divide by 1000, use first value since square)
            side_length = dimensions[0] / 1000.0
            print(f"Texture dimensions: {dimensions}mm, side_length: {side_length}m")
            
            # Download individual texture files (diff, rough, nor_gl, disp)
            try:
                # Get the files information
                files_response = requests.get(f"https://api.polyhaven.com/files/{texture_asset_id}", headers=REQ_HEADERS)
                if files_response.status_code != 200:
                    return {"error": f"Failed to get asset files: {files_response.status_code}"}
                
                files_data = files_response.json()
                
                # Preferred resolution order
                resolution = "2k"
                resolution_fallbacks = ["2k", "1k", "4k", "8k"]
                
                # Map types to download with their preferred formats
                # Keys are the API map type names (case-sensitive as in API response)
                # Formats must match what works in Blender: diff=jpg, rough=exr, nor_gl=exr, disp=png
                texture_map_configs = {
                    "Diffuse": {"formats": ["jpg", "png"], "output_name": "diff"},
                    "Rough": {"formats": ["exr", "png", "jpg"], "output_name": "rough"},
                    "nor_gl": {"formats": ["exr", "png", "jpg"], "output_name": "nor_gl"},
                    "Displacement": {"formats": ["png", "exr", "jpg"], "output_name": "disp"},
                }
                
                # Create a temporary directory for textures
                temp_dir = tempfile.mkdtemp()
                textures_path = temp_dir
                
                try:
                    downloaded_files = []
                    
                    for map_type, config in texture_map_configs.items():
                        if map_type not in files_data:
                            print(f"Warning: {map_type} not found in texture files")
                            continue
                        
                        # Find available resolution
                        found_resolution = None
                        for res in resolution_fallbacks:
                            if res in files_data[map_type]:
                                found_resolution = res
                                break
                        
                        if not found_resolution:
                            print(f"Warning: No resolution found for {map_type}")
                            continue
                        
                        # Find available format
                        found_format = None
                        file_url = None
                        for fmt in config["formats"]:
                            if fmt in files_data[map_type][found_resolution]:
                                found_format = fmt
                                file_url = files_data[map_type][found_resolution][fmt]["url"]
                                break
                        
                        if not file_url:
                            print(f"Warning: No suitable format found for {map_type}")
                            continue
                        
                        # Download the file
                        print(f"Downloading {map_type} ({found_resolution}, {found_format}): {file_url}")
                        response = requests.get(file_url, headers=REQ_HEADERS)
                        if response.status_code != 200:
                            print(f"Warning: Failed to download {map_type}: {response.status_code}")
                            continue
                        
                        # Save the file with standardized naming: {asset_id}_{map_type}_{resolution}.{format}
                        output_filename = f"{texture_asset_id}_{config['output_name']}_{found_resolution}.{found_format}"
                        output_path = os.path.join(textures_path, output_filename)
                        with open(output_path, 'wb') as f:
                            f.write(response.content)
                        
                        downloaded_files.append(output_filename)
                        print(f"Saved: {output_filename}")
                    
                    if not downloaded_files:
                        return {"error": f"Failed to download any texture files for {texture_asset_id}"}
                    
                    print(f"Found textures at: {textures_path}")
                    
                    # Use create_repetitive_plane_from_polyhaven to create the ground plane
                    result = create_repetitive_plane_from_polyhaven(
                        textures_path=textures_path,
                        width=width,
                        depth=depth,
                        side_length=side_length,
                        output_plane="ground_plane"
                    )
                    
                    if not result['success']:
                        return {"error": f"Failed to create ground plane: {result['message']}"}
                    
                    print(f"Successfully created ground plane with texture: {texture_asset_id}")
                    
                    # Pack all images into the blend file so textures persist after temp cleanup
                    # Use os.path.normpath for consistent path comparison
                    normalized_temp_dir = os.path.normpath(temp_dir)
                    packed_count = 0
                    for img in bpy.data.images:
                        if img.filepath:
                            # Normalize the image path for comparison
                            # Handle Blender's // relative paths
                            img_path = bpy.path.abspath(img.filepath)
                            normalized_img_path = os.path.normpath(img_path)
                            
                            if normalized_temp_dir in normalized_img_path or texture_asset_id in img.name:
                                try:
                                    # Ensure image data is loaded before packing
                                    if img.packed_file is None:
                                        img.pack()
                                        packed_count += 1
                                        print(f"Packed image: {img.name}")
                                except Exception as pack_error:
                                    print(f"Warning: Failed to pack image {img.name}: {pack_error}")
                    
                    print(f"Total images packed: {packed_count}")
                    
                    # Prepare ground result
                    ground_result = {
                        "ground_asset_id": texture_asset_id,
                        "ground_texture_name": best_texture.get("name", ""),
                        "ground_combined_score": best_texture.get("combined_score"),
                        "ground_llm_score": best_texture.get("llm_score"),
                        "plane_name": result.get('plane').name if result.get('plane') else "ground_plane",
                        "x_times": result.get('x_times'),
                        "y_times": result.get('y_times'),
                        "side_length": side_length,
                        "width": width,
                        "depth": depth,
                    }
                    
                    # Check if wall parameters are provided
                    wall_result = {}
                    has_walls = wall_description or wall_asset_id
                    
                    if has_walls:
                        print("\n" + "="*60)
                        print("Creating indoor walls...")
                        print("="*60)
                        
                        # Search or use provided wall asset
                        if wall_asset_id:
                            print(f"Using provided wall_asset_id: {wall_asset_id}")
                            wall_texture = {"id": wall_asset_id, "name": wall_asset_id, "dimensions": [2000, 2000]}
                        else:
                            try:
                                wall_search_results = search_polyhaven_assets(
                                    asset_type="textures",
                                    description=wall_description,
                                    returned_count=5,
                                    threshold_score=0.4,
                                    rerank_with_llm=True,
                                    threshold_llm=1,
                                    anyllm_api_key=anyllm_api_key,
                                    anyllm_api_base=anyllm_api_base,
                                    anyllm_provider=anyllm_provider,
                                    vision_model=vision_model,
                                    categories_limitation=wall_categories_limitation,
                                )
                                
                                if not wall_search_results:
                                    wall_search_results = search_polyhaven_assets(
                                        asset_type="textures",
                                        description=wall_description,
                                        returned_count=5,
                                        threshold_score=0.4,
                                        rerank_with_llm=False,
                                        categories_limitation=wall_categories_limitation,
                                    )
                                
                                if not wall_search_results:
                                    print("Warning: No suitable wall textures found")
                                    wall_texture = None
                                else:
                                    wall_texture = wall_search_results[0]
                                    print(f"Selected wall texture: {wall_texture.get('id')}")
                            except Exception as e:
                                print(f"Error searching wall textures: {e}")
                                wall_texture = None
                        
                        if wall_texture:
                            wall_asset_id_final = wall_texture.get("id")
                            wall_dimensions = wall_texture.get("dimensions", [2000, 2000])
                            wall_side_length = wall_dimensions[0] / 1000.0
                            
                            # Download wall texture files
                            wall_files_response = requests.get(f"https://api.polyhaven.com/files/{wall_asset_id_final}", headers=REQ_HEADERS)
                            if wall_files_response.status_code == 200:
                                wall_files_data = wall_files_response.json()
                                wall_temp_dir = tempfile.mkdtemp()
                                
                                try:
                                    wall_downloaded = []
                                    for map_type, config in texture_map_configs.items():
                                        if map_type not in wall_files_data:
                                            continue
                                        found_res = None
                                        for res in resolution_fallbacks:
                                            if res in wall_files_data[map_type]:
                                                found_res = res
                                                break
                                        if not found_res:
                                            continue
                                        found_fmt = None
                                        file_url = None
                                        for fmt in config["formats"]:
                                            if fmt in wall_files_data[map_type][found_res]:
                                                found_fmt = fmt
                                                file_url = wall_files_data[map_type][found_res][fmt]["url"]
                                                break
                                        if file_url:
                                            resp = requests.get(file_url, headers=REQ_HEADERS)
                                            if resp.status_code == 200:
                                                fname = f"{wall_asset_id_final}_{config['output_name']}_{found_res}.{found_fmt}"
                                                with open(os.path.join(wall_temp_dir, fname), 'wb') as f:
                                                    f.write(resp.content)
                                                wall_downloaded.append(fname)
                                    
                                    if wall_downloaded:
                                        # Create walls with provided or default dimensions
                                        wall_params = {
                                            "textures_path": wall_temp_dir,
                                            "side_length": wall_side_length,
                                        }
                                        if wall_x is not None:
                                            wall_params["x"] = wall_x
                                        if wall_x_negative is not None:
                                            wall_params["x_negative"] = wall_x_negative
                                        if wall_y is not None:
                                            wall_params["y"] = wall_y
                                        if wall_y_negative is not None:
                                            wall_params["y_negative"] = wall_y_negative
                                        if wall_z is not None:
                                            wall_params["z"] = wall_z
                                        
                                        walls_result = create_indoor_walls(**wall_params)
                                        
                                        if walls_result['success']:
                                            # Pack wall images
                                            normalized_wall_dir = os.path.normpath(wall_temp_dir)
                                            for img in bpy.data.images:
                                                if img.filepath:
                                                    img_path = bpy.path.abspath(img.filepath)
                                                    normalized_img_path = os.path.normpath(img_path)
                                                    if normalized_wall_dir in normalized_img_path or wall_asset_id_final in img.name:
                                                        if img.packed_file is None:
                                                            try:
                                                                img.pack()
                                                                print(f"Packed wall image: {img.name}")
                                                            except:
                                                                pass
                                            
                                            wall_result = {
                                                "wall_asset_id": wall_asset_id_final,
                                                "wall_texture_name": wall_texture.get("name", ""),
                                                "walls_created": list(walls_result.get('walls', {}).keys()),
                                                "roof_created": walls_result.get('roof') is not None,
                                            }
                                        else:
                                            print(f"Warning: Failed to create walls: {walls_result.get('message')}")
                                finally:
                                    try:
                                        shutil.rmtree(wall_temp_dir)
                                    except:
                                        pass
                    
                    # Combine results
                    final_result = {
                        "success": True,
                        "message": "Environment design completed successfully.",
                        **ground_result,
                        **wall_result,
                    }
                    
                    return final_result
                    
                finally:
                    # Clean up temporary directory after packing textures
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as cleanup_error:
                        print(f"Warning: Failed to clean up temp directory: {cleanup_error}")
                
            except Exception as e:
                print(f"Error downloading and processing texture: {e}")
                traceback.print_exc()
                return {"error": f"Failed to download and process texture: {str(e)}"}
                
        except Exception as e:
            print(f"Error in environment_artist: {e}")
            traceback.print_exc()
            return {"error": f"Environment artist failed: {str(e)}"}


    def format_assets(self, path_to_script, model_output_dir, anyllm_api_key=None, vision_model="gemini-3-flash-preview", anyllm_api_base=None, anyllm_provider="gemini", model_id_list=None, output_json_filename="formatted_model.json"):
        """
        Format all models in a storyboard script by calling format_asset for each asset.
        
        Args:
            path_to_script (str): Path to the storyboard script JSON file
            model_output_dir (str): Directory to save formatted models
            anyllm_api_key (str): API key for LLM service (for rotation correction)
            vision_model (str): LLM model identifier for vision tasks
            anyllm_api_base (str): Optional API base URL for LLM service
            anyllm_provider (str): LLM provider (default: "gemini")
            model_id_list (list): Optional list of asset_ids to format. If None, format all.
                                  If provided but output json doesn't exist, format all.
            output_json_filename (str): Filename for the output JSON (default: formatted_model.json)
            
        Returns:
            dict: Result of the formatting process with success status and details
        """
        try:
            # Read the JSON file
            if not os.path.exists(path_to_script):
                return {"error": f"Script file not found: {path_to_script}"}
            
            with open(path_to_script, 'r') as f:
                script_data = json.load(f)
            
            # Get the asset_sheet
            if "asset_sheet" not in script_data:
                return {"error": "No asset_sheet found in script"}
            
            asset_sheet = script_data["asset_sheet"]
            
            # Create output directory if it doesn't exist
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Check if output json exists for merge logic
            output_json_path = os.path.join(model_output_dir, output_json_filename)
            existing_formatted_data = None
            
            if model_id_list is not None and os.path.exists(output_json_path):
                # Load existing formatted data for merging
                try:
                    with open(output_json_path, 'r') as f:
                        existing_formatted_data = json.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load existing {output_json_filename}: {e}")
                    # If we can't load existing data, format all models
                    model_id_list = None
            elif model_id_list is not None and not os.path.exists(output_json_path):
                # If model_id_list is provided but json doesn't exist, format all
                print(f"{output_json_filename} doesn't exist, formatting all models")
                model_id_list = None
            
            # Process each model in the asset_sheet
            formatted_models = []
            errors = []
            
            # Store original viewport settings and set up for format operations
            original_viewport_settings = {}
            try:
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                # Store original UI settings
                                original_viewport_settings['show_region_header'] = space.show_region_header
                                original_viewport_settings['show_region_toolbar'] = space.show_region_toolbar
                                original_viewport_settings['show_region_ui'] = space.show_region_ui
                                original_viewport_settings['show_gizmo'] = space.show_gizmo
                                original_viewport_settings['shading_type'] = space.shading.type
                                if hasattr(space, 'overlay'):
                                    original_viewport_settings['show_overlays'] = space.overlay.show_overlays
                                
                                # Hide UI elements for clean screenshots
                                space.show_region_header = False
                                space.show_region_toolbar = False
                                space.show_region_ui = False
                                space.show_gizmo = False
                                if hasattr(space, 'overlay'):
                                    space.overlay.show_overlays = False
                                space.shading.type = 'MATERIAL'
                                break
                        break
                
                # Store original background/gradient settings and set to white
                gradients = bpy.context.preferences.themes[0].view_3d.space.gradients
                original_viewport_settings['background_type'] = gradients.background_type
                original_viewport_settings['high_gradient'] = tuple(gradients.high_gradient)
                
                # Set background to white single color
                gradients.background_type = 'SINGLE_COLOR'
                gradients.high_gradient = (1.0, 1.0, 1.0)
                
                # Force UI update
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                
            except Exception as e:
                print(f"Warning: Failed to set up viewport for format operations: {e}")
            
            # Calculate total assets to process
            if model_id_list is not None:
                assets_to_process = [m for m in asset_sheet if m.get("asset_id") in model_id_list]
            else:
                assets_to_process = asset_sheet
            total_assets = len(assets_to_process)
            current_asset_num = 0
            
            for model_info in asset_sheet:
                model_id = model_info.get("asset_id", "unknown")
                
                # Skip if model_id_list is provided and this model is not in it
                if model_id_list is not None and model_id not in model_id_list:
                    continue
                
                current_asset_num += 1
                print(f"\n{'='*60}")
                print(f"Processing asset {current_asset_num}/{total_assets}: {model_id}")
                print(f"{'='*60}")
                
                try:
                    # Call format_asset with model_info dict
                    result = format_asset(
                        model_info=model_info,
                        export_dir=model_output_dir,
                        anyllm_api_key=anyllm_api_key,
                        vision_model=vision_model,
                        anyllm_api_base=anyllm_api_base,
                        anyllm_provider=anyllm_provider
                    )
                    
                    # Update the main_file_path to the new export path
                    model_info["main_file_path"] = os.path.abspath(result["export_path"])
                    
                    # Add thumbnail_url if available
                    if result.get("thumbnail_url"):
                        model_info["thumbnail_url"] = result["thumbnail_url"]
                    
                    # Add directional view URLs if available
                    if result.get("front_view_url"):
                        model_info["front_view_url"] = result["front_view_url"]
                    if result.get("top_view_url"):
                        model_info["top_view_url"] = result["top_view_url"]
                    if result.get("left_view_url"):
                        model_info["left_view_url"] = result["left_view_url"]
                    
                    formatted_models.append(model_id)
                    print(f"Successfully formatted model: {model_id}")
                    
                    # Add delay between assets to let Gradio's asyncio event loop stabilize
                    # This helps prevent crashes from thread conflicts during heavy Blender operations
                    # time.sleep(3)
                    
                except Exception as e:
                    error_msg = f"{model_id}: {str(e)}"
                    errors.append(error_msg)
                    print(f"Error formatting model {model_id}: {str(e)}")
                    traceback.print_exc()
                    continue
            
            # Merge with existing formatted data if we're doing selective re-format
            if existing_formatted_data is not None and model_id_list is not None:
                # Build a map of asset_id to updated model_info for quick lookup
                updated_assets = {m.get("asset_id"): m for m in asset_sheet if m.get("asset_id") in formatted_models}
                
                # Update existing_formatted_data's asset_sheet with newly formatted models
                existing_asset_sheet = existing_formatted_data.get("asset_sheet", [])
                for i, existing_asset in enumerate(existing_asset_sheet):
                    asset_id = existing_asset.get("asset_id")
                    if asset_id in updated_assets:
                        # Replace with the newly formatted data
                        existing_asset_sheet[i] = updated_assets[asset_id]
                
                # Use existing_formatted_data as the output
                script_data = existing_formatted_data
            
            # Save the updated JSON to model_output_dir
            output_path = os.path.join(model_output_dir, output_json_filename)
            
            with open(output_path, 'w') as f:
                json.dump(script_data, f, indent=4)
            
            # Prepare result
            result = {
                "success": True,
                "formatted_models": formatted_models,
                "total_models": len(asset_sheet),
                "formatted_count": len(formatted_models),
                "output_script_path": output_path
            }
            
            if errors:
                result["errors"] = errors
                result["error_count"] = len(errors)
            
            # Restore viewport to original state
            try:
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                # Restore UI regions to original or default values
                                space.show_region_header = original_viewport_settings.get('show_region_header', True)
                                space.show_region_toolbar = original_viewport_settings.get('show_region_toolbar', True)
                                space.show_region_ui = original_viewport_settings.get('show_region_ui', True)
                                space.show_gizmo = original_viewport_settings.get('show_gizmo', True)
                                
                                # Restore overlays
                                if hasattr(space, 'overlay'):
                                    space.overlay.show_overlays = original_viewport_settings.get('show_overlays', True)
                                    space.overlay.show_floor = True
                                    space.overlay.show_axis_x = True
                                    space.overlay.show_axis_y = True
                                    space.overlay.show_cursor = True
                                
                                # Restore shading type
                                space.shading.type = original_viewport_settings.get('shading_type', 'MATERIAL')
                                
                                break
                        break
                
                # Restore original background/gradient settings
                gradients = bpy.context.preferences.themes[0].view_3d.space.gradients
                if 'background_type' in original_viewport_settings:
                    gradients.background_type = original_viewport_settings['background_type']
                if 'high_gradient' in original_viewport_settings:
                    gradients.high_gradient = original_viewport_settings['high_gradient']
                
                # Force UI update
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                
            except Exception as e:
                print(f"Warning: Failed to restore viewport defaults: {e}")
            
            # Final delay to let Gradio's asyncio event loop stabilize before returning
            # This helps prevent crashes during the response phase
            # time.sleep(3)
            # gc.collect()  # Force garbage collection at a safe point
            # time.sleep(3)
            
            return result
            
        except Exception as e:
            print(f"Error in format_assets: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to format assets: {str(e)}"}

    def resize_assets(self, path_to_script, model_output_dir, model_id_list=None, output_json_filename="resized_model.json"):
        """
        Resize all models in a script file based on their estimated dimensions.
        
        Args:
            path_to_script (str): Path to the JSON file containing asset_sheet with dimension estimates
            model_output_dir (str): Directory to save resized models
            model_id_list (list): Optional list of asset_ids to resize. If None, resize all.
                                  If provided but output json doesn't exist, resize all.
            output_json_filename (str): Filename for the output JSON (default: resized_model.json)
            
        Returns:
            dict: Result of the resizing process with success status and details
        """
        try:
            # Read the JSON file
            if not os.path.exists(path_to_script):
                return {"error": f"Script file not found: {path_to_script}"}
            
            with open(path_to_script, 'r') as f:
                script_data = json.load(f)
            
            # Get the asset_sheet
            if "asset_sheet" not in script_data:
                return {"error": "No asset_sheet found in script"}
            
            asset_sheet = script_data["asset_sheet"]
            
            # Create output directory if it doesn't exist
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Check if output json exists for merge logic
            output_json_path = os.path.join(model_output_dir, output_json_filename)
            existing_resized_data = None
            
            if model_id_list is not None and os.path.exists(output_json_path):
                # Load existing resized data for merging
                try:
                    with open(output_json_path, 'r') as f:
                        existing_resized_data = json.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load existing {output_json_filename}: {e}")
                    model_id_list = None
            elif model_id_list is not None and not os.path.exists(output_json_path):
                print(f"{output_json_filename} doesn't exist, resizing all models")
                model_id_list = None
            
            # Process each model in the asset_sheet
            resized_models = []
            errors = []
            
            # Store original viewport settings and set up for resize operations
            original_viewport_settings = {}
            try:
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                original_viewport_settings['show_region_header'] = space.show_region_header
                                original_viewport_settings['show_region_toolbar'] = space.show_region_toolbar
                                original_viewport_settings['show_region_ui'] = space.show_region_ui
                                original_viewport_settings['show_gizmo'] = space.show_gizmo
                                original_viewport_settings['shading_type'] = space.shading.type
                                if hasattr(space, 'overlay'):
                                    original_viewport_settings['show_overlays'] = space.overlay.show_overlays
                                
                                space.show_region_header = False
                                space.show_region_toolbar = False
                                space.show_region_ui = False
                                space.show_gizmo = False
                                if hasattr(space, 'overlay'):
                                    space.overlay.show_overlays = False
                                space.shading.type = 'MATERIAL'
                                break
                        break
                
                gradients = bpy.context.preferences.themes[0].view_3d.space.gradients
                original_viewport_settings['background_type'] = gradients.background_type
                original_viewport_settings['high_gradient'] = tuple(gradients.high_gradient)
                
                gradients.background_type = 'SINGLE_COLOR'
                gradients.high_gradient = (1.0, 1.0, 1.0)
                
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                
            except Exception as e:
                print(f"Warning: Failed to set up viewport for resize operations: {e}")
            
            # Calculate total assets to process
            if model_id_list is not None:
                assets_to_process = [m for m in asset_sheet if m.get("asset_id") in model_id_list]
            else:
                assets_to_process = asset_sheet
            total_assets = len(assets_to_process)
            current_asset_num = 0
            
            for model_info in asset_sheet:
                model_id = model_info.get("asset_id", "unknown")
                
                # Skip if model_id_list is provided and this model is not in it
                if model_id_list is not None and model_id not in model_id_list:
                    continue
                
                current_asset_num += 1
                print(f"\n{'='*60}")
                print(f"Processing asset {current_asset_num}/{total_assets}: {model_id}")
                print(f"{'='*60}")
                
                try:
                    # Call resize_asset with model_info dict
                    result = resize_asset(
                        model_info=model_info,
                        export_dir=model_output_dir
                    )
                    
                    # Update the model with returned dimensions (rounded to 2 decimal places)
                    model_info["width"] = round(result["dimensions"]["X"], 2)
                    model_info["depth"] = round(result["dimensions"]["Y"], 2)
                    model_info["height"] = round(result["dimensions"]["Z"], 2)
                    
                    # Update the main_file_path to the new export path
                    model_info["main_file_path"] = os.path.abspath(result["export_path"])
                    
                    # Add thumbnail_url if available
                    if result.get("thumbnail_url"):
                        model_info["thumbnail_url"] = result["thumbnail_url"]
                    
                    # Add directional view URLs if available
                    if result.get("front_view_url"):
                        model_info["front_view_url"] = result["front_view_url"]
                    if result.get("top_view_url"):
                        model_info["top_view_url"] = result["top_view_url"]
                    if result.get("left_view_url"):
                        model_info["left_view_url"] = result["left_view_url"]
                    
                    resized_models.append(model_id)
                    print(f"Successfully resized model: {model_id}")
                    
                    # time.sleep(3)
                    
                except Exception as e:
                    error_msg = f"{model_id}: {str(e)}"
                    errors.append(error_msg)
                    print(f"Error resizing model {model_id}: {str(e)}")
                    traceback.print_exc()
                    continue
            
            # Merge with existing resized data if we're doing selective re-resize
            if existing_resized_data is not None and model_id_list is not None:
                updated_assets = {m.get("asset_id"): m for m in asset_sheet if m.get("asset_id") in resized_models}
                
                existing_asset_sheet = existing_resized_data.get("asset_sheet", [])
                for i, existing_asset in enumerate(existing_asset_sheet):
                    asset_id = existing_asset.get("asset_id")
                    if asset_id in updated_assets:
                        existing_asset_sheet[i] = updated_assets[asset_id]
                
                script_data = existing_resized_data
            
            # Save the updated JSON to model_output_dir
            output_path = os.path.join(model_output_dir, output_json_filename)
            
            with open(output_path, 'w') as f:
                json.dump(script_data, f, indent=4)
            
            # Prepare result
            result = {
                "success": True,
                "resized_models": resized_models,
                "total_models": len(asset_sheet),
                "resized_count": len(resized_models),
                "output_script_path": output_path
            }
            
            if errors:
                result["errors"] = errors
                result["error_count"] = len(errors)
            
            # Restore viewport to original state
            try:
                for area in bpy.context.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                space.show_region_header = original_viewport_settings.get('show_region_header', True)
                                space.show_region_toolbar = original_viewport_settings.get('show_region_toolbar', True)
                                space.show_region_ui = original_viewport_settings.get('show_region_ui', True)
                                space.show_gizmo = original_viewport_settings.get('show_gizmo', True)
                                
                                if hasattr(space, 'overlay'):
                                    space.overlay.show_overlays = original_viewport_settings.get('show_overlays', True)
                                    space.overlay.show_floor = True
                                    space.overlay.show_axis_x = True
                                    space.overlay.show_axis_y = True
                                    space.overlay.show_cursor = True
                                
                                space.shading.type = original_viewport_settings.get('shading_type', 'MATERIAL')
                                
                                break
                        break
                
                gradients = bpy.context.preferences.themes[0].view_3d.space.gradients
                if 'background_type' in original_viewport_settings:
                    gradients.background_type = original_viewport_settings['background_type']
                if 'high_gradient' in original_viewport_settings:
                    gradients.high_gradient = original_viewport_settings['high_gradient']
                
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                
            except Exception as e:
                print(f"Warning: Failed to restore viewport defaults: {e}")
            
            # time.sleep(3)
            # gc.collect()
            # time.sleep(3)
            
            return result
            
        except Exception as e:
            print(f"Error in resize_assets: {str(e)}")
            traceback.print_exc()
            return {"error": f"Failed to resize assets: {str(e)}"}

    @staticmethod
    def _clean_imported_glb(filepath, mesh_name=None):
        # Get the set of existing objects before import
        existing_objects = set(bpy.data.objects)

        # Import the GLB file
        bpy.ops.import_scene.gltf(filepath=filepath)

        # Ensure the context is updated
        bpy.context.view_layer.update()

        # Get all imported objects
        imported_objects = list(set(bpy.data.objects) - existing_objects)
        # imported_objects = [obj for obj in bpy.context.view_layer.objects if obj.select_get()]

        if not imported_objects:
            print("Error: No objects were imported.")
            return

        # Identify the mesh object
        mesh_obj = None

        if len(imported_objects) == 1 and imported_objects[0].type == 'MESH':
            mesh_obj = imported_objects[0]
            print("Single mesh imported, no cleanup needed.")
        else:
            if len(imported_objects) == 2:
                empty_objs = [i for i in imported_objects if i.type == "EMPTY"]
                if len(empty_objs) != 1:
                    print("Error: Expected an empty node with one mesh child or a single mesh object.")
                    return
                parent_obj = empty_objs.pop()
                if len(parent_obj.children) == 1:
                    potential_mesh = parent_obj.children[0]
                    if potential_mesh.type == 'MESH':
                        print("GLB structure confirmed: Empty node with one mesh child.")

                        # Unparent the mesh from the empty node
                        potential_mesh.parent = None

                        # Remove the empty node
                        bpy.data.objects.remove(parent_obj)
                        print("Removed empty node, keeping only the mesh.")

                        mesh_obj = potential_mesh
                    else:
                        print("Error: Child is not a mesh object.")
                        return
                else:
                    print("Error: Expected an empty node with one mesh child or a single mesh object.")
                    return
            else:
                print("Error: Expected an empty node with one mesh child or a single mesh object.")
                return

        # Rename the mesh if needed
        try:
            if mesh_obj and mesh_obj.name is not None and mesh_name:
                mesh_obj.name = mesh_name
                if mesh_obj.data.name is not None:
                    mesh_obj.data.name = mesh_name
                print(f"Mesh renamed to: {mesh_name}")
        except Exception as e:
            print("Having issue with renaming, give up renaming.")

        return mesh_obj


    def handle_import_asset_to_scene(self, filepath, scene_name, transform_parameters=None):
        """
        Import a GLB asset to a specific scene and apply transforms.
        
        Args:
            filepath: File path to the GLB file.
            scene_name: The name of the scene to import the model to (created if not exists).
            transform_parameters: Dict with transform parameters (location, rotation, scale, dimensions).
            
        Returns:
            A dict with 'success' (bool) and additional info.
        """
        try:
            result = import_asset_to_scene(
                filepath=filepath,
                scene_name=scene_name,
                transform_parameters=transform_parameters,
            )
            return result
        except Exception as e:
            print(f"Error in handle_import_asset_to_scene: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def handle_get_asset_transform(self, model_name):
        """
        Get all transform properties of a model in Blender.
        
        Args:
            model_name: Name of an existing model in the scene.
            
        Returns:
            A dict with 'success' (bool) and transform data.
        """
        try:
            result = get_asset_transform(model_name=model_name)
            return result
        except Exception as e:
            print(f"Error in handle_get_asset_transform: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def handle_transform_asset(self, model_name, location=None, rotation=None, scale=None, dimensions=None):
        """
        Transform a model in Blender by setting its location, rotation, scale, and dimensions.
        
        Args:
            model_name: Name of an existing model in the scene to transform.
            location: Dict with keys 'x', 'y', 'z' for coordinates.
            rotation: Dict with keys 'x', 'y', 'z' for rotation angles in XYZ Euler (radians).
            scale: Dict with keys 'x', 'y', 'z' for scale factors.
            dimensions: Dict with keys 'x', 'y', 'z' for dimensions.
            
        Returns:
            A dict with 'success' (bool) and 'message' (str) keys.
        """
        try:
            result = transform_asset(
                model_name=model_name,
                location=location,
                rotation=rotation,
                scale=scale,
                dimensions=dimensions,
            )
            return result
        except Exception as e:
            print(f"Error in handle_transform_asset: {str(e)}")
            traceback.print_exc()
            return {"success": False, "message": str(e)}

    def handle_import_all_assets_to_all_scenes_json_input(self, json_filepath):
        """
        Import all assets to all scenes from a JSON file.
        
        Args:
            json_filepath: Path to the JSON file containing asset_sheet and scene_details.
            
        Returns:
            A dict with 'success' (bool) and 'failed_objects' list if any failures.
        """
        try:
            result = import_all_assets_to_all_scenes_json_input(json_filepath=json_filepath)
            return result
        except Exception as e:
            print(f"Error in handle_import_all_assets_to_all_scenes_json_input: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def handle_import_supplementary_assets_to_all_scenes_json_input(self, json_filepath):
        """
        Import supplementary assets to all scenes from a JSON file.
        
        Args:
            json_filepath: Path to the JSON file containing asset_sheet and scene_details
                          for supplementary assets.
            
        Returns:
            A dict with 'success' (bool) and 'failed_objects' list if any failures.
        """
        try:
            result = import_supplementary_assets_to_all_scenes_json_input(json_filepath=json_filepath)
            return result
        except Exception as e:
            print(f"Error in handle_import_supplementary_assets_to_all_scenes_json_input: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def handle_apply_asset_modifications_json_input(self, json_filepath):
        """
        Apply asset modifications to shot scenes from a JSON file.
        
        Args:
            json_filepath: Path to the JSON file containing shot_details with
                          asset_modifications.
            
        Returns:
            A dict with 'success' (bool), 'modified_count', and 'errors' list if any failures.
        """
        try:
            result = apply_asset_modifications_json_input(json_filepath=json_filepath)
            return result
        except Exception as e:
            print(f"Error in handle_apply_asset_modifications_json_input: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def handle_switch_or_create_scene(self, scene_name):
        """
        Switch to a scene with the given scene_name, or create it if it doesn't exist.
        
        Args:
            scene_name: The name of the scene to switch to or create.
            
        Returns:
            A dict with 'success' (bool) and 'error' if failed.
        """
        try:
            switch_or_create_scene(scene_name=scene_name)
            return {"success": True}
        except Exception as e:
            print(f"Error in handle_switch_or_create_scene: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def handle_delete_all_scenes_and_assets(self):
        """
        Delete all scenes and assets, leaving only an empty scene named 'Scene'.
        
        Returns:
            A dict with 'success' (bool) and 'message' or 'error'.
        """
        try:
            result = delete_all_scenes_and_assets()
            return result
        except Exception as e:
            print(f"Error in handle_delete_all_scenes_and_assets: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def handle_import_animated_assets_to_all_shots_json_input(self, json_filepath):
        """
        Import animated assets to all shots from a JSON file.
        
        Args:
            json_filepath: Path to the JSON file containing shot_details and scene_details.
            
        Returns:
            A dict with 'success' (bool) and details about the import.
        """
        try:
            result = import_animated_assets_to_all_shots_json_input(json_filepath=json_filepath)
            return result
        except Exception as e:
            print(f"Error in handle_import_animated_assets_to_all_shots_json_input: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def handle_delete_all_shots(self):
        """
        Delete all shot scenes in Blender.
        
        Returns:
            A dict with 'success' (bool) and details about deleted scenes.
        """
        try:
            result = delete_all_shots()
            return result
        except Exception as e:
            print(f"Error in handle_delete_all_shots: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def handle_delete_asset(self, model_name):
        """
        Delete an asset (object) from the current scene by its name.
        
        If the exact name is not found, this will try to find objects matching
        the Blender duplicate naming pattern (e.g., {model_name}.001, {model_name}.002).
        
        Args:
            model_name: The name of the object to delete.
            
        Returns:
            A dict with 'success' (bool) and 'message' or 'error'.
        """
        try:
            # Check if object exists with exact name
            obj = bpy.data.objects.get(model_name)
            
            # If not found, try to find objects with Blender's duplicate suffix pattern
            if obj is None:
                import re
                pattern = re.compile(rf'^{re.escape(model_name)}\.(\d{{3}})$')
                matching_objects = []
                for obj_candidate in bpy.data.objects:
                    if pattern.match(obj_candidate.name):
                        matching_objects.append(obj_candidate)
                
                if matching_objects:
                    # Sort by suffix number and take the first one (lowest number)
                    matching_objects.sort(key=lambda x: x.name)
                    obj = matching_objects[0]
            
            if obj is None:
                return {
                    "success": False,
                    "error": f"Object '{model_name}' (or variants like {model_name}.001) not found in the scene"
                }
            
            # Store the name for the success message
            deleted_name = obj.name
            
            # Remove the object from all collections first
            for collection in obj.users_collection:
                collection.objects.unlink(obj)
            
            # Delete the object data (mesh, etc.) if it has no other users
            obj_data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            
            # Clean up orphan data if the mesh/data has no users
            if obj_data is not None and obj_data.users == 0:
                if hasattr(bpy.data, obj_data.rna_type.identifier.lower() + 's'):
                    data_collection = getattr(bpy.data, obj_data.rna_type.identifier.lower() + 's', None)
                    if data_collection is not None:
                        try:
                            data_collection.remove(obj_data)
                        except:
                            pass
            
            return {
                "success": True,
                "message": f"Successfully deleted object '{deleted_name}'"
            }
        except Exception as e:
            print(f"Error in handle_delete_asset: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def handle_camera_operator(
        self,
        path_to_input_json,
        vision_model="gemini-2.5-flash",
        anyllm_api_key=None,
        anyllm_api_base=None,
        anyllm_provider="gemini",
        camera_type="director",
        max_additional_cameras=1,
        camera_name_filter=None,
        start_frame=1,
        end_frame=73,
        max_adjustment_rounds=5,
        preview_image_save_dir=None,
    ):
        """
        Place cameras in Blender scenes based on storyboard instructions.
        
        Args:
            path_to_input_json: Path to JSON file with shot_details
            vision_model: Vision model for LLM
            anyllm_api_key: API key for any-llm
            anyllm_api_base: API base URL for any-llm
            anyllm_provider: LLM provider (default: "gemini")
            camera_type: 'director', 'additional', or 'all'
            max_additional_cameras: Maximum additional cameras per shot
            camera_name_filter: List of camera names to place (None = all)
            start_frame: Start frame for camera animation
            end_frame: End frame for camera animation
            max_adjustment_rounds: Maximum LLM adjustment rounds
            preview_image_save_dir: Directory to save preview images
            
        Returns:
            dict: Result with shot_details, cameras_placed, cameras_failed
        """
        try:
            result = camera_operator(
                path_to_input_json=path_to_input_json,
                vision_model=vision_model,
                anyllm_api_key=anyllm_api_key,
                anyllm_api_base=anyllm_api_base,
                anyllm_provider=anyllm_provider,
                camera_type=camera_type,
                max_additional_cameras=max_additional_cameras,
                camera_name_filter=camera_name_filter,
                start_frame=start_frame,
                end_frame=end_frame,
                max_adjustment_rounds=max_adjustment_rounds,
                preview_image_save_dir=preview_image_save_dir,
            )
            return result
        except Exception as e:
            print(f"Error in handle_camera_operator: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def handle_resume_camera_operator(
        self,
        path_to_input_json,
        camera_name_filter=None,
    ):
        """
        Resume/recreate cameras from a previously saved JSON file.
        
        Args:
            path_to_input_json: Path to JSON file with camera placement info
            camera_name_filter: List of camera names to resume (None = all)
            
        Returns:
            dict: Result with cameras_resumed, cameras_failed
        """
        try:
            result = resume_camera_operator(
                path_to_input_json=path_to_input_json,
                camera_name_filter=camera_name_filter,
            )
            return result
        except Exception as e:
            print(f"Error in handle_resume_camera_operator: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def handle_get_camera_info(self, scene_name, camera_name):
        """
        Get camera information from Blender by scene name and camera name.
        
        Reads camera transform, focal length, DoF settings, and keyframe data.
        
        Args:
            scene_name: Name of the scene containing the camera
            camera_name: Name of the camera object
            
        Returns:
            dict: Camera info including transforms, parameters, and animation data
        """
        try:
            # Switch to the scene
            if scene_name not in bpy.data.scenes:
                return {"success": False, "error": f"Scene '{scene_name}' not found"}
            
            scene = bpy.data.scenes[scene_name]
            bpy.context.window.scene = scene
            
            # Find the camera
            camera_obj = None
            for obj in scene.objects:
                if obj.name == camera_name and obj.type == 'CAMERA':
                    camera_obj = obj
                    break
            
            if camera_obj is None:
                return {"success": False, "error": f"Camera '{camera_name}' not found in scene '{scene_name}'"}
            
            camera_data = camera_obj.data
            
            # Get camera parameters
            camera_parameters = {
                "focal_length": camera_data.lens,
                "sensor_width": camera_data.sensor_width,
                "sensor_height": camera_data.sensor_height,
                "clip_start": camera_data.clip_start,
                "clip_end": camera_data.clip_end,
            }
            
            # Get DoF settings
            dof_applied = camera_data.dof.use_dof
            focus_distance = None
            if dof_applied:
                if camera_data.dof.focus_object:
                    # Calculate distance to focus object
                    focus_obj = camera_data.dof.focus_object
                    focus_distance = (camera_obj.location - focus_obj.location).length
                else:
                    focus_distance = camera_data.dof.focus_distance
            
            # Get current transform
            location = list(camera_obj.location)
            rotation = list(camera_obj.rotation_euler)
            
            # Check for animation and get keyframe transforms
            is_animated = False
            start_transform = None
            end_transform = None
            start_frame = scene.frame_start
            end_frame = scene.frame_end
            
            if camera_obj.animation_data and camera_obj.animation_data.action:
                action = camera_obj.animation_data.action
                fcurves = action.fcurves
                
                if fcurves:
                    is_animated = True
                    
                    # Find keyframe times
                    keyframe_times = set()
                    for fcurve in fcurves:
                        for keyframe in fcurve.keyframe_points:
                            keyframe_times.add(int(keyframe.co[0]))
                    
                    if keyframe_times:
                        keyframe_times = sorted(keyframe_times)
                        start_frame = keyframe_times[0]
                        end_frame = keyframe_times[-1] if len(keyframe_times) > 1 else keyframe_times[0]
                        
                        # Get transform at start frame
                        scene.frame_set(start_frame)
                        start_transform = {
                            "location": list(camera_obj.location),
                            "rotation": list(camera_obj.rotation_euler),
                        }
                        
                        # Get transform at end frame
                        scene.frame_set(end_frame)
                        end_transform = {
                            "location": list(camera_obj.location),
                            "rotation": list(camera_obj.rotation_euler),
                        }
                        
                        # Reset to start frame
                        scene.frame_set(start_frame)
            
            # If not animated, use current transform for both
            if not is_animated:
                start_transform = {
                    "location": location,
                    "rotation": rotation,
                }
                end_transform = start_transform
            
            return {
                "success": True,
                "camera_name": camera_name,
                "scene_name": scene_name,
                "camera_parameters": camera_parameters,
                "start_transform": start_transform,
                "end_transform": end_transform,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "is_animated": is_animated,
                "dof_applied": dof_applied,
                "focus_distance": focus_distance,
            }
            
        except Exception as e:
            print(f"Error in handle_get_camera_info: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
