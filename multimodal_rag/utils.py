"""Utility functions for the Multimodal RAG system."""
import base64
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from langchain_core.documents import Document


def encode_image(image_path: Union[str, Path]) -> str:
    """Encode an image file as a base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def is_base64(s: str) -> bool:
    """Check if a string is base64 encoded."""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", s) is not None


def is_image_data(b64data: str) -> bool:
    """Check if the base64 data is an image."""
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89PNG\r\n\x1a\n": "png",
        b"GIF8": "gif",
        b"RIFF": "webp",
    }
    
    try:
        header = base64.b64decode(b64data)[:8]
        return any(header.startswith(sig) for sig in image_signatures.keys())
    except Exception:
        return False


def split_image_text_types(docs: List[Document]) -> Dict[str, List]:
    """Split documents into images and text.
    
    Args:
        docs: List of Document objects or strings
        
    Returns:
        Dictionary with 'images' and 'texts' keys
    """
    b64_images = []
    texts = []
    
    for doc in docs:
        content = doc.page_content if isinstance(doc, Document) else doc
        
        if is_base64(content) and is_image_data(content):
            b64_images.append(content)
        else:
            texts.append(content)
            
    return {"images": b64_images, "texts": texts}


def display_image(b64_string: str) -> None:
    """Display a base64 encoded image in Jupyter notebook."""
    from IPython.display import display as ip_display, Image as IPImage
    ip_display(IPImage(data=base64.b64decode(b64_string)))


def resize_image(image_path: Union[str, Path], max_size: int = 1024) -> Image.Image:
    """Resize an image while maintaining aspect ratio."""
    img = Image.open(image_path)
    width, height = img.size
    
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
            
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
    return img


def extract_text_from_image(b64_image: str) -> str:
    """Extract text from an image using OCR."""
    try:
        import pytesseract
        from PIL import Image
        import io
        
        # Decode base64 and open as image
        image_data = base64.b64decode(b64_image)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale for better OCR
        image = image.convert('L')
        
        # Use Tesseract to do OCR on the image
        text = pytesseract.image_to_string(image)
        return text.strip()
    except ImportError:
        raise ImportError("pytesseract is required for OCR. Install with: pip install pytesseract")
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""
