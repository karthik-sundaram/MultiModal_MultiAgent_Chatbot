import os
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from huggingface_hub import InferenceClient
from transformers.agents import HfApiEngine
from langchain_community.utilities import WikipediaAPIWrapper
from bark import generate_audio, SAMPLE_RATE  # Correct placement at the top
from config import HF_TOKEN  # Import HF_TOKEN from config.py

# Initialize models
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

client_sd = InferenceClient(
    model="stabilityai/stable-diffusion-xl-base-1.0",
    token=HF_TOKEN  # Use HF_TOKEN from config.py
)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

client_audio = InferenceClient(
    model="suno/bark",
    token=HF_TOKEN  # Use HF_TOKEN from config.py
)

llm_engine = HfApiEngine(model="meta-llama/Meta-Llama-3-8B-Instruct")

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
