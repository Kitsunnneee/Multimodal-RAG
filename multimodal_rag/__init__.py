"""
Multimodal RAG System with Google's Gemini and Vertex AI

This package provides tools for building and querying a multimodal retrieval-augmented
generation system that can process and reason across text, tables, and images.
Supports document parsing via LlamaParse for improved extraction of complex documents.
"""

__version__ = "0.2.0"

# Expose the main components
from .rag_system import MultimodalRAG
from .document_processor import DocumentProcessor
from .llama_parse_loader import LlamaParseLoader

__all__ = [
    "MultimodalRAG",
    "DocumentProcessor",
    "LlamaParseLoader",
]
