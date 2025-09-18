"""Utility functions for the Multimodal RAG system."""
import base64
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

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


def is_base64(s) -> bool:
    """Check if a string is base64 encoded.
    
    Args:
        s: Input to check. Can be string, bytes, or any other type.
        
    Returns:
        bool: True if input is a valid base64 string, False otherwise
    """
    if not isinstance(s, (str, bytes, bytearray)):
        return False
        
    if isinstance(s, str):
        # Remove data URL prefix if present
        if s.startswith('data:'):
            s = s.split(',', 1)[-1]
        # Check if the string contains only valid base64 characters
        try:
            # Try to decode the string to verify it's valid base64
            base64.b64decode(s, validate=True)
            return True
        except Exception:
            return False
    return False


def is_image_data(b64data) -> bool:
    """Check if the input is base64-encoded image data.
    
    Args:
        b64data: Input data to check. Can be string, bytes, or any other type.
        
    Returns:
        bool: True if input is a valid base64-encoded image, False otherwise
    """
    if not b64data or not isinstance(b64data, (str, bytes, bytearray)):
        return False
        
    # Common image signatures
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89PNG\r\n\x1a\n": "png",
        b"GIF8": "gif",
        b"RIFF": "webp",
        b"\x00\x00\x01\x00": "ico",  # ICO format
        b"BM": "bmp",                # BMP format
    }
    
    try:
        if isinstance(b64data, str):
            # Handle data URLs if present
            if b64data.startswith('data:image/'):
                b64data = b64data.split(',', 1)[-1]
            # Decode the base64 data
            decoded = base64.b64decode(b64data, validate=True)
        else:
            # Already bytes or bytearray
            decoded = base64.b64decode(b64data, validate=True)
            
        # Check against known image signatures
        header = decoded[:16]  # Check first 16 bytes for signature
        return any(header.startswith(sig) for sig in image_signatures.keys())
        
    except (ValueError, TypeError, IndexError):
        return False


def split_image_text_types(docs: List[Union[Document, str, dict]]) -> Dict[str, List]:
    """Split documents into images and text.
    
    Args:
        docs: List of Document objects, strings, or dictionaries with 'page_content' and 'type' keys
        
    Returns:
        Dictionary with 'images' and 'texts' keys
    """
    b64_images = []
    texts = []
    
    for doc in docs:
        # Handle different input types
        if isinstance(doc, Document):
            content = doc.page_content
            doc_type = doc.metadata.get('type', 'text') if hasattr(doc, 'metadata') else 'text'
        elif isinstance(doc, dict):
            content = doc.get('page_content', '') if 'page_content' in doc else str(doc)
            doc_type = doc.get('type', 'text')
        else:
            content = str(doc)
            doc_type = 'text'
        
        # Skip empty content
        if not content:
            continue
            
        # Check if content is an image (either by type or by content analysis)
        if doc_type == 'image' or (is_base64(content) and is_image_data(content)):
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


def extract_text_from_image(b64_image: str, service_account_path: str = None) -> str:
    """Extract text from an image using Google's Gemini 2.0 Flash model.
    
    Args:
        b64_image: Base64 encoded image data
        service_account_path: Path to service account JSON file. If not provided,
                           will use GOOGLE_APPLICATION_CREDENTIALS environment variable.
    
    Returns:
        Extracted text from the image
        
    Raises:
        ImportError: If required packages are not installed
        FileNotFoundError: If service account file is not found
        Exception: For any errors during text extraction
    """
    try:
        from google.cloud import aiplatform
        from google.oauth2 import service_account
        from vertexai.generative_models import GenerativeModel, Part
        from PIL import Image
        import base64
        import io
        import os
        import json
        
        # Get service account path from environment if not provided
        service_account_path = service_account_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not service_account_path or not os.path.exists(service_account_path):
            raise FileNotFoundError(
                "Service account JSON file is required. Either pass the path as an argument "
                "or set GOOGLE_APPLICATION_CREDENTIALS environment variable to point to the file."
            )
        
        # Load service account credentials
        credentials = service_account.Credentials.from_service_account_file(
            service_account_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Initialize Vertex AI
        project_id = json.load(open(service_account_path)).get('project_id')
        location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
        aiplatform.init(project=project_id, location=location, credentials=credentials)
        
        # Import here to avoid circular imports
        from vertexai.generative_models import GenerativeModel
        
        # Initialize the model
        model = GenerativeModel('gemini-2.0-flash')
        
        # Prepare the image data
        image_data = base64.b64decode(b64_image)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save to bytes buffer
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        
        # Generate content with proper format
        from vertexai.generative_models import Part
        
        # Create the prompt and image parts
        prompt = """Extract all text from this image. Return only the extracted text, 
        without any additional commentary or formatting."""
        
        # Create the content with proper format
        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(img_byte_arr.getvalue()).decode()
                        }
                    }
                ]
            }
        ]
        
        # Generate the response with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(contents)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Retry {attempt + 1}/{max_retries}...")
        
        # Extract text from the response
        try:
            # Try to get text directly
            if hasattr(response, 'text'):
                return response.text.strip()
                
            # Try to get text from parts
            if hasattr(response, 'parts'):
                return '\n'.join(part.text for part in response.parts if hasattr(part, 'text') and part.text).strip()
                
            # Try to get text from candidates
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    return '\n'.join(part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text).strip()
                    
            # If we get here, try to convert the response to string
            return str(response).strip()
            
        except Exception as e:
            print(f"Error processing API response: {e}")
            raise ValueError("Could not extract text from API response")
        
    except ImportError as ie:
        print(f"Error: {ie}")
        print("Falling back to pytesseract...")
    except Exception as e:
        print(f"Error extracting text with Gemini 2.0 Flash: {e}")
        print("Falling back to pytesseract...")
    
    # Fallback to pytesseract
    try:
        import pytesseract
        from PIL import Image
        import io
        import base64
        
        print("Attempting to use pytesseract for OCR...")
        image_data = base64.b64decode(b64_image)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale for better OCR results
        if image.mode != 'L':
            image = image.convert('L')
            
        # Use pytesseract to do OCR on the image
        text = pytesseract.image_to_string(image).strip()
        if not text:
            print("Warning: Pytesseract returned empty text")
        return text
        
    except ImportError as ie:
        print("Error: Pytesseract is not installed. Install with: pip install pytesseract and install tesseract-ocr")
        print("On macOS: brew install tesseract tesseract-lang")
        print("On Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        raise Exception("No OCR method available. Please install either:"
                      "\n1. google-generativeai (recommended): pip install google-generativeai"
                      "\n2. pytesseract: pip install pytesseract and install tesseract-ocr")
    except Exception as fallback_error:
        print(f"Fallback OCR failed: {fallback_error}")
        return ""
