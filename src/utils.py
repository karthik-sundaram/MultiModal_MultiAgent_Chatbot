import os
from PIL import Image
from config import IMAGE_DIR  # Import from config.py

def save_image(image, filename):
    """Saves the given image to the specified filename in the fixed directory."""
    try:
        path = os.path.join(IMAGE_DIR, filename)
        if isinstance(image, str):
            image = Image.open(image)
        image.save(path)
        return path
    except Exception as e:
        return f"Error saving image: {str(e)}"

def retrieve_image_path(filename):
    """Retrieves the path of the specified image from the fixed directory."""
    return os.path.join(IMAGE_DIR, filename)
