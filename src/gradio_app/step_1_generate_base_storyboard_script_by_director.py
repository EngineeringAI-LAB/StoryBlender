import os
import gradio as gr
from ..operators.director_operators.generate_script_by_director import generate_script_by_director
from ..operators.director_operators.director_schema import Storyboard
from .json_editor import JSONEditorComponent


def save_story_to_file(project_dir, story_text):
    """Save story text to project_dir/story.txt (overwrites existing file)."""
    if not project_dir or not os.path.isabs(project_dir):
        return False
    try:
        os.makedirs(project_dir, exist_ok=True)
        story_path = os.path.join(project_dir, "story.txt")
        with open(story_path, "w", encoding="utf-8") as f:
            f.write(story_text)
        print(f"Story saved to: {story_path}")
        return True
    except Exception as e:
        print(f"Error saving story: {e}")
        return False


def load_story_from_file(project_dir):
    """Load story text from project_dir/story.txt."""
    if not project_dir or not os.path.isabs(project_dir):
        return None
    story_path = os.path.join(project_dir, "story.txt")
    if os.path.exists(story_path):
        try:
            with open(story_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error loading story: {e}")
            return None
    return None


def validate_and_generate_director_script(
    reasoning_model,
    anyllm_api_key,
    anyllm_api_base,
    anyllm_provider,
    project_dir,
    story_input,
    editor_component
):
    """Validate inputs and generate script using the director function. The json script will be saved to the project directory.
    
    Args:
        reasoning_model: The reasoning model to use for generation
        anyllm_api_key: The API key for authentication
        anyllm_api_base: The API base URL for any-llm (optional)
        anyllm_provider: The LLM provider (default: gemini)
        project_dir: The absolute path to the project directory
        story_input: The story input to generate the script from
        editor_component: The JSONEditorComponent to save the result
    
    Returns:
        A dictionary containing the result of the director function call
    """
    
    # Validate project directory
    if not os.path.isabs(project_dir):
        return {
            "error": "⚠️ Project directory must be an absolute path (e.g., /Users/username/projects/my_project)"
        }
    
    # Validate story input
    if not story_input or story_input.strip() == "":
        return {
            "error": "⚠️ Please enter a story before submitting"
        }
    
    # Save the story to file
    save_story_to_file(project_dir, story_input)
    
    # Set API base to None if empty string
    api_base = anyllm_api_base if anyllm_api_base.strip() else None
    
    # Call the director function (no longer saves file)
    result = generate_script_by_director(
        anyllm_api_key=anyllm_api_key,
        anyllm_api_base=api_base,
        model=reasoning_model,
        anyllm_provider=anyllm_provider,
        contents=story_input,
        reasoning_effort="high"
    )
    
    # If successful, save the JSON using the editor component
    if result.get("success") and "data" in result:
        # Set the save path for director files
        director_save_path = os.path.join(project_dir, "director")
        editor_component.set_save_path(director_save_path)
        
        # Save the JSON data
        output_path = editor_component.save_json_data(result["data"])
        if output_path:
            result["output_path"] = output_path
        else:
            result["error"] = "Failed to save JSON file"
            result["success"] = False
    
    return result


def show_loading_and_generate(editor_component, reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider, project_dir, story_input):
    """Show loading indicator and generate script."""
    # Build initial loading state - all editor components hidden
    loading_outputs = editor_component.update_with_result(None)
    loading_state = (
        gr.update(value="🔄 **Generating storyboard...** This may take 3-5 minutes. Please wait.", visible=True),  # Show loading
        gr.update(visible=False),  # Hide story input
        gr.update(visible=False),  # Hide story_resume_btn
        gr.update(visible=True),   # Show story_toggle_btn
        gr.update(visible=False),  # Hide submit_btn
        gr.update(visible=False),  # Hide save_story_btn
        False,                     # story_visible state
    )
    
    yield loading_outputs + loading_state
    
    # Generate the script (pass editor_component for saving)
    result = validate_and_generate_director_script(
        reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider, project_dir, story_input, editor_component
    )
    
    # Return final result with editor component updated
    final_outputs = editor_component.update_with_result(result)
    
    # Show success message if generation succeeded
    if result.get("success"):
        success_msg = "✅ **Storyboard generated successfully!** You can review the JSON below and proceed to the next step."
    else:
        success_msg = ""  # Error message is shown in the editor
    
    final_state = (
        gr.update(value=success_msg, visible=bool(success_msg)),  # Show success tip
        gr.update(visible=False),  # story_input hidden
        gr.update(visible=False),  # Hide story_resume_btn
        gr.update(visible=True),   # Show story_toggle_btn
        gr.update(visible=False),  # Hide submit_btn
        gr.update(visible=False),  # Hide save_story_btn
        False,                     # story_visible state
    )
    
    yield final_outputs + final_state


def create_generate_wrapper(editor_component):
    """Factory function to create a generate wrapper bound to a specific editor component."""
    def generate_wrapper(reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider, project_dir, story_input):
        """Wrapper to properly yield from the generator."""
        for result in show_loading_and_generate(editor_component, reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider, project_dir, story_input):
            yield result
    return generate_wrapper


def handle_save_story(proj_dir, story_text):
    """Save Story button click handler - only saves story.txt without generation."""
    if not proj_dir or not os.path.isabs(proj_dir):
        return gr.update(value="**Error:** Please set a valid absolute project directory first.", visible=True)
    if not story_text or story_text.strip() == "":
        return gr.update(value="**Error:** Please enter a story before saving.", visible=True)
    
    success = save_story_to_file(proj_dir, story_text)
    if success:
        return gr.update(value=f"**Story saved to:** `{os.path.join(proj_dir, 'story.txt')}`", visible=True)
    else:
        return gr.update(value="**Error:** Failed to save story.", visible=True)


def handle_story_resume(proj_dir):
    """Resume Story button click handler."""
    story_text = load_story_from_file(proj_dir)
    if story_text:
        return (
            gr.update(value=story_text, visible=True),  # Load and show story_input
            gr.update(visible=False),                   # Hide resume button
            gr.update(visible=True),                    # Show toggle button
            gr.update(value=f"**Loaded from:** `{os.path.join(proj_dir, 'story.txt')}`", visible=True),  # Show status
            gr.update(visible=True),                    # Show submit_btn
            gr.update(visible=True),                    # Show save_story_btn
            True,                                       # story_visible state
        )
    else:
        return (
            gr.update(),                                # Keep story_input as is
            gr.update(visible=True),                    # Keep resume button visible
            gr.update(visible=False),                   # Keep toggle hidden
            gr.update(value="**No story.txt found.** Enter a new story.", visible=True),  # Show error
            gr.update(),                                # Keep submit_btn as is
            gr.update(),                                # Keep save_story_btn as is
            True,                                       # story_visible state
        )


def toggle_story_visibility(is_visible):
    """Toggle Story visibility button handler."""
    new_visibility = not is_visible
    return (
        gr.update(visible=new_visibility),  # story_input
        gr.update(visible=new_visibility),  # submit_btn
        gr.update(visible=new_visibility),  # save_story_btn
        new_visibility,                     # story_visible state
    )


def create_director_ui(reasoning_model, anyllm_api_key, anyllm_api_base, anyllm_provider, project_dir):
    """Create the Step 1: Generate Base Storyboard Script by Director UI section.
    
    Args:
        reasoning_model: Gradio component for reasoning model selection
        anyllm_api_key: Gradio component for any-llm API key
        anyllm_api_base: Gradio component for any-llm API base URL
        anyllm_provider: Gradio component for LLM provider
        project_dir: Gradio component for project directory
    
    Returns:
        dict with UI components that may be needed by other parts of the app
    """
    gr.Markdown("## Step 1: Generate Base Storyboard Script by Director")
    gr.Markdown("### Step 1.1: Story Input")
    
    # Story input section buttons
    story_resume_btn = gr.Button("📂 Resume Story", variant="secondary", visible=True)
    story_status = gr.Markdown(value="", visible=False)
    
    story_input = gr.Textbox(
        label="Your Story",
        placeholder="Enter your story script here...\n\nExample:\n### Scene 1\n**INT. QUEEN'S CHAMBER - NIGHT**\nA severe, dark room...",
        lines=15,
        info="Enter your story script or narrative",
        visible=True
    )
    
    submit_btn = gr.Button("Generate Storyboard", variant="primary", size="lg")
    save_story_btn = gr.Button("💾 Save Story", variant="secondary", visible=True)
    story_toggle_btn = gr.Button("👁️ Hide/Show Story Editor", variant="secondary", visible=True)
    
    # State to track story input visibility
    story_visible = gr.State(value=True)
    
    gr.Markdown("### Step 1.2: Generated Storyboard from Director")
    
    # Loading status indicator (hidden by default)
    loading_status = gr.Markdown(value="", visible=False)
    
    # Create JSON editor component (reusable for future steps)
    # save_path will be set dynamically when generating (project_dir/director)
    director_editor = JSONEditorComponent(
        label="Storyboard JSON",
        visible_initially=False,
        file_basename="director",  # Files will be saved as director_v1.json, director_v2.json, etc.
        title="Step 1"
    )
    
    # Wire up the Resume button with project_dir input
    director_editor.setup_resume_with_project_dir(project_dir, subfolder="director")
    
    # Create wrapper function for the generator
    generate_wrapper = create_generate_wrapper(director_editor)
    
    # Submit button click handler
    submit_btn.click(
        fn=generate_wrapper,
        inputs=[
            reasoning_model,
            anyllm_api_key,
            anyllm_api_base,
            anyllm_provider,
            project_dir,
            story_input
        ],
        outputs=director_editor.get_output_components() + [loading_status, story_input, story_resume_btn, story_toggle_btn, submit_btn, save_story_btn, story_visible],
    )
    
    # Save Story button click handler
    save_story_btn.click(
        fn=handle_save_story,
        inputs=[project_dir, story_input],
        outputs=[story_status],
    )
    
    # Resume Story button click handler
    story_resume_btn.click(
        fn=handle_story_resume,
        inputs=[project_dir],
        outputs=[story_input, story_resume_btn, story_toggle_btn, story_status, submit_btn, save_story_btn, story_visible],
    )
    
    # Toggle Story visibility button handler
    story_toggle_btn.click(
        fn=toggle_story_visibility,
        inputs=[story_visible],
        outputs=[story_input, submit_btn, save_story_btn, story_visible],
    )
    
    return {
        "director_editor": director_editor,
        "story_input": story_input,
        "story_visible": story_visible,
    }
