"""Utility functions for handling different file types."""
from pathlib import Path
from typing import Optional, Tuple, Union
import mimetypes


def get_file_type(file_path: Union[str, Path]) -> Tuple[str, str]:
    """Get the file type and extension from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (file_type, extension)
        file_type: One of 'pdf', 'docx', 'pptx', 'xlsx', 'image', 'text', or 'unknown'
        extension: File extension in lowercase
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    # Common document types
    if extension in ('.pdf', '.docx', '.pptx', '.xlsx'):
        return extension[1:], extension  # Remove the dot
    
    # Image types
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    if extension in image_extensions:
        return 'image', extension
    
    # Fallback to mimetype detection
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        if mime_type.startswith('image/'):
            return 'image', extension
        if mime_type in ('application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'):
            return mime_type.split('/')[-1], extension
    
    # Text files
    if extension in ('.txt', '.md', '.markdown'):
        return 'text', extension
    
    return 'unknown', extension


def is_supported_file(file_path: Union[str, Path]) -> bool:
    """Check if a file type is supported for processing."""
    supported_types = ('pdf', 'docx', 'pptx', 'xlsx', 'image', 'text')
    file_type, _ = get_file_type(file_path)
    return file_type in supported_types
