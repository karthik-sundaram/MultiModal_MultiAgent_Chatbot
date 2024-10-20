import os
import time
from src.utils import save_image
from src.workflow import app as workflow_app
import gradio as gr
from langsmith import Client
from langchain.schema import HumanMessage
from config import LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT

# Set environment variables from config.py
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

client = Client()

# Initialize variables
feedback_score = None
feedback_comment = None
# context and current_run_id are managed in workflow.py

def gradio_interface(text, image):
    """Handles the user interaction with text input and image upload."""
    # No need to declare context as global here since it's managed in workflow.py
    response_content = ""
    generated_image_path = None
    generated_audio_path = None

    print("[DEBUG] Received text:", text)
    print("[DEBUG] Received image:", image)

    try:
        if image:
            image_filename = f"uploaded_image_{int(time.time())}.png"
            image_path = save_image(image, image_filename)
            print(f"[DEBUG] Image saved at: {image_path}")

            print(f"[DEBUG] Sending image path to workflow: {image_path}")
            final_state = workflow_app.invoke({"messages": [HumanMessage(content=f"Image uploaded: {image_path}")]})

            response_content = final_state["messages"][-1].content
            print(f"[DEBUG] Image description: {response_content}")

            # context is managed in workflow.py

            return f"Image uploaded successfully! Description: {response_content}", image_path, None

        elif text:
            print(f"[DEBUG] Sending text query to workflow: {text}")
            final_state = workflow_app.invoke({"messages": [HumanMessage(content=text)]})

            response_content = final_state["messages"][-1].content
            print(f"[DEBUG] Assistant response: {response_content}")

            # context is managed in workflow.py

            if ".wav" in response_content.lower():
                generated_audio_path = response_content
                print(f"[DEBUG] Detected generated audio at: {generated_audio_path}")
                return response_content, None, generated_audio_path

            elif "images/" in response_content.lower():
                generated_image_path = response_content
                print(f"[DEBUG] Detected generated image at: {generated_image_path}")
                return response_content, generated_image_path, None

            return response_content, None, None

        else:
            print("[DEBUG] No text or image received.")
            return "No input provided.", None, None

    except Exception as e:
        print(f"[DEBUG] Error occurred: {str(e)}")
        return f"Error occurred: {str(e)}", None, None

def submit_feedback(feedback_score_input=None, feedback_comment_input=None):
    """Function to log user feedback without triggering chatbot."""
    global feedback_score, feedback_comment
    feedback_score = feedback_score_input
    feedback_comment = feedback_comment_input or "No comment"
    log_feedback()
    return "Feedback submitted successfully."

def log_feedback():
    # Assuming current_run_id is accessible here
    global current_run_id, feedback_score, feedback_comment
    if current_run_id is not None and feedback_score is not None:
        try:
            client.create_feedback(
                run_id=current_run_id,
                key="user-feedback",
                score=feedback_score,
                comment=feedback_comment
            )
            print(f"Feedback logged successfully for run_id: {current_run_id}")
        except Exception as e:
            print(f"Error logging feedback: {str(e)}")
    else:
        print("No run_id or feedback score available for logging feedback.")

    feedback_score = None
    feedback_comment = None

# Gradio interface
gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(lines=2, placeholder="Type your question here", label="Text Input"),
        gr.Image(type="filepath", label="Image Upload")
    ],
    outputs=[
        gr.Textbox(lines=5, label="Response"),
        gr.Image(type="filepath", label="Generated or Uploaded Image"),
        gr.Audio(type="filepath", label="Generated Audio")
    ],
    title="Multimodal Chatbot with Live Audio",
    description="Upload an image or ask a question to interact with the chatbot, including audio responses.",
    examples=None,
    allow_flagging="never"
).launch(debug=True)

# Feedback submission interface
gr.Interface(
    fn=submit_feedback,
    inputs=[
        gr.Slider(minimum=-1, maximum=1, step=1, label="Feedback Score"),
        gr.Textbox(lines=1, placeholder="Optional Feedback Comment")
    ],
    outputs="text",
    title="Submit Feedback",
    description="Submit feedback without re-triggering the chatbot."
).launch()
