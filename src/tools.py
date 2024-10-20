import os
import time
import soundfile as sf
from PIL import Image
from transformers.tools import Tool  # Corrected import
from src.models import (
    wiki_wrapper,
    blip_processor,
    blip_model,
    clip_processor,
    clip_model,
    client_sd,
    llm_engine,
    generate_audio,
    SAMPLE_RATE
)
from src.utils import save_image
from config import AUDIO_DIR, IMAGE_DIR  # Import from config.py

# Define generated_image_paths at module level
generated_image_paths = []

class wiki_tool__(Tool):
    name = "wiki_search"
    description = "Search Wikipedia for relevant information."

    inputs = {
        "query": {
            "type": "string",
            "description": "The search query for Wikipedia."
        }
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        try:
            return wiki_wrapper.run(query)
        except Exception as e:
            return f"Error in Wikipedia search: {str(e)}"

wiki_tool = wiki_tool__()

class gpt_text_response__(Tool):
    name = "gpt_text_response"
    description = "Generates a text response using the LLM."

    inputs = {
        "query": {
            "type": "string",
            "description": "The user's query."
        },
        "context": {
            "type": "string",
            "description": "Additional context to assist in generating the response.",
            "default": ""
        }
    }
    output_type = "string"

    def forward(self, query: str, context: str = "") -> str:
        """Generate a text response using the LLM."""
        try:
            print(f"[DEBUG] gpt_text_response received query: {query}")
            print(f"[DEBUG] gpt_text_response received context: {context}")

            messages = [{"role": "user", "content": f"{query} {context}"}]
            print(f"[DEBUG] Combined input for LLM: {messages}")

            response = llm_engine(messages)
            print(f"[DEBUG] gpt_text_response raw response: {response}")

            response_content = response.strip()
            print(f"[DEBUG] gpt_text_response processed content: {response_content}")

            return response_content
        except Exception as e:
            print(f"[DEBUG] Error in gpt_text_response: {str(e)}")
            return f"Error: {str(e)}"

gpt_text_response = gpt_text_response__()

class blip_image_caption__(Tool):
    name = "blip_image_caption"
    description = "Generates a caption or description for an image uploaded by user - using the BLIP model."

    inputs = {
        "image_path": {
            "type": "string",
            "description": "Path to the image for captioning."
        },
        "context": {
            "type": "string",
            "description": "Optional context for generating the caption.",
            "default": ""
        }
    }
    output_type = "string"

    def forward(self, image_path: str, context: str = "") -> str:
        """Generate a caption for or describe an image using the BLIP model."""
        try:
            image = Image.open(image_path)
            inputs = blip_processor(image, return_tensors="pt")
            out = blip_model.generate(**inputs)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            return f"Error in generating caption: {str(e)}"

blip_image_caption = blip_image_caption__()

class generate_image__(Tool):
    name = "generate_image"
    description = "Generates an image based on a text prompt using Stable Diffusion."

    inputs = {
        "prompt": {
            "type": "string",
            "description": "The text prompt to generate the image."
        },
        "context": {
            "type": "string",
            "description": "Optional context to refine the image generation.",
            "default": ""
        }
    }
    output_type = "string"

    def forward(self, prompt: str, context: str = "") -> str:
        global generated_image_paths
        try:
            image_response = client_sd.text_to_image(prompt + " " + context)
            image = image_response  # Assuming it's a PIL Image

            filename = f"generated_image_{int(time.time())}.png"
            saved_path = save_image(image, filename)

            generated_image_paths.append(saved_path)
            print(f"[DEBUG] Appended image path: {saved_path}")
            print(f"[DEBUG] Current generated_image_paths: {generated_image_paths}")

            return saved_path  # Returning the path so Gradio can display it
        except Exception as e:
            return f"Error in generating image: {str(e)}"

generate_image = generate_image__()

class compare_image_to_text__(Tool):
    name = "compare_image_to_text"
    description = "Compares a user-uploaded image to a user input text description using the CLIP model and returns a similarity score."

    inputs = {
        "image_path": {
            "type": "string",
            "description": "Path to the image for comparison."
        },
        "description": {
            "type": "string",
            "description": "Text description to compare with the image."
        },
        "context": {
            "type": "string",
            "description": "Optional context to add to the description.",
            "default": ""
        }
    }
    output_type = "string"

    def forward(self, image_path: str, description: str, context: str = "") -> str:
        """Compare an image to a text description using the CLIP model."""
        try:
            image = Image.open(image_path)
            inputs = clip_processor(
                text=[description + " " + context],
                images=image,
                return_tensors="pt",
                padding=True
            )
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            similarity = logits_per_image.softmax(dim=1)
            score = similarity[0][0].item()
            if score > 0.5:
                return f"The image is very similar to the description '{description}'."
            else:
                return f"The image is not similar to the description '{description}'."
        except Exception as e:
            return f"Error in comparing image to text: {str(e)}"

compare_image_to_text = compare_image_to_text__()

class generate_audio_from_text__(Tool):
    name = "generate_audio_from_text"
    description = "Generates audio from a text prompt using the Bark model."

    inputs = {
        "prompt": {
            "type": "string",
            "description": "The text prompt to generate audio from."
        }
    }
    output_type = "audio"

    def forward(self, prompt: str) -> str:
        try:
            audio_array = generate_audio(prompt)

            audio_filename = f"generated_audio_{int(time.time())}.wav"
            audio_path = os.path.join(AUDIO_DIR, audio_filename)

            sf.write(audio_path, audio_array, SAMPLE_RATE)

            print(f"Audio generated successfully at: {audio_path}")
            return audio_path
        except Exception as e:
            print(f"Error in generating audio: {str(e)}")
            return f"Error in generating audio: {str(e)}"

generate_audio_from_text = generate_audio_from_text__()

tools = [
    gpt_text_response,
    blip_image_caption,
    generate_image,
    compare_image_to_text,
    generate_audio_from_text,
    wiki_tool
]
