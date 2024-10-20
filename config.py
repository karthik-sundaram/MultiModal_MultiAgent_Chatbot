import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
LANGCHAIN_KEY = os.getenv('LANGCHAIN_KEY')
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_API_KEY = LANGCHAIN_KEY
LANGCHAIN_PROJECT = "langgraph_multiM"

IMAGE_DIR = os.path.join(os.getcwd(), 'images')
AUDIO_DIR = os.path.join(os.getcwd(), 'audio')

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
