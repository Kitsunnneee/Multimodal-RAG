"""Embedding management for multimodal documents using Google Gemini."""
from typing import Dict, List, Optional, Union, Any
import numpy as np
from google.generativeai import configure, GenerativeModel
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_core.documents import Document
from PIL import Image
import os

class LocalEmbeddings:
    """Local embeddings using Gemini as a fallback.
    
    This class is maintained for backward compatibility with existing code.
    New code should use GeminiEmbeddings or EmbeddingManager directly.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with Gemini embeddings."""
        self.embeddings = GeminiEmbeddings()
        self.dimension = self.embeddings.dimension
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate text embeddings."""
        return self.embeddings.embed_text(text)
    
    def embed_image(self, image: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """Generate image embeddings."""
        return self.embeddings.embed_image(image)
    
    def embed_video(self, video_path: str) -> np.ndarray:
        """Generate video embeddings."""
        return self.embeddings.embed_video(video_path)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.
        
        This method is maintained for backward compatibility with LangChain's interface.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embeddings, one for each document
        """
        embeddings = self.embed_text(texts)
        return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

def safe_b64decode(data: str) -> bytes:
    """Safely decode base64 string, handling padding issues."""
    try:
        # Add padding if needed
        padding = len(data) % 4
        if padding:
            data += '=' * (4 - padding)
        return base64.b64decode(data)
    except Exception as e:
        raise ValueError(f"Failed to decode base64: {e}")

class GeminiEmbeddings:
    """Handles embedding generation using Google's Vertex AI Text Embeddings with fallback to local model."""
    
    def __init__(self, api_key: str = None, model_name: str = "text-embedding-004"):
        """Initialize the embeddings model with fallback to local model.
        
        Args:
            api_key: Google Cloud API key. If None, will use application default credentials.
            model_name: Name of the embeddings model to use (default: text-embedding-004)
        """
        self.model_name = model_name
        self.api_key = api_key
        self.use_local = False
        self.local_model = None
        self.dimension = 768  # Standard dimension for text-embedding-004
        
        # Try to initialize Vertex AI with proper error handling
        try:
            # First try with explicit API key if provided
            if self.api_key:
                os.environ["GOOGLE_API_KEY"] = self.api_key
                
            # Check if we have any authentication method available
            if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ and 'GOOGLE_API_KEY' not in os.environ:
                raise ValueError("No Google Cloud credentials found. Please set GOOGLE_APPLICATION_CREDENTIALS or provide an API key.")
            
            # Initialize Vertex AI
            import vertexai
            from google.cloud import aiplatform
            
            # Try to initialize with explicit project/location if available
            project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
            location = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
            
            if project_id:
                vertexai.init(project=project_id, location=location)
            else:
                vertexai.init()
            
            # Initialize the embeddings model
            from vertexai.language_models import TextEmbeddingModel
            self.model = TextEmbeddingModel.from_pretrained(self.model_name)
            
            # Test the API with a simple call
            try:
                self.model.get_embeddings(["test"])
                print(f"Successfully initialized Vertex AI Embeddings with model: {self.model_name}")
                return  # Success, no need for fallback
            except Exception as e:
                print(f"Warning: Vertex AI API test failed: {e}")
                
        except ImportError as e:
            print(f"Warning: Could not import Vertex AI dependencies: {e}")
        except Exception as e:
            print(f"Warning: Could not initialize Vertex AI: {e}")
        
        # If we get here, Vertex AI initialization failed
        print("Falling back to local embeddings model...")
        self._init_local_model()
    
    def _init_local_model(self):
        """Initialize a local model for embeddings with multiple fallback options."""
        # List of models to try in order of preference
        model_options = [
            'all-MiniLM-L6-v2',  # Good balance of speed and quality
            'paraphrase-MiniLM-L6-v2',  # Alternative if above fails
            'all-mpnet-base-v2',  # Higher quality but larger
        ]
        
        # Try each model until one works
        for model_name in model_options:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Attempting to load local embedding model: {model_name}")
                self.local_model = SentenceTransformer(model_name)
                self.dimension = self.local_model.get_sentence_embedding_dimension()
                self.use_local = True
                print(f"Successfully initialized local embedding model: {model_name}")
                return  # Success, exit after first working model
            except Exception as e:
                print(f"Warning: Could not load {model_name}: {e}")
                continue
        
        # If no model worked, try a basic fallback
        try:
            print("Falling back to basic universal-sentence-encoder")
            import tensorflow_hub as hub
            import tensorflow as tf
            # Suppress TensorFlow logging
            tf.get_logger().setLevel('ERROR')
            self.local_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            self.dimension = 512  # USEv4 uses 512 dimensions
            self.use_local = True
            print("Initialized basic universal-sentence-encoder")
        except Exception as e:
            print(f"Warning: Could not initialize any local embeddings model: {e}")
            print("Using random embeddings as fallback (not recommended for production)")
            self.use_local = False
            self.dimension = 384  # Standard dimension for smaller models
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            numpy.ndarray: Text embeddings
        """
        if self.use_local:
            return self.local_model.encode(text, convert_to_numpy=True)
            
        if isinstance(text, str):
            text = [text]
            
        try:
            if not hasattr(self, 'model') or self.model is None:
                raise ValueError("Model not initialized")
                
            # Use Vertex AI's TextEmbeddingModel
            embeddings = []
            for i in range(0, len(text), 5):  # Batch process to avoid rate limits
                batch = text[i:i+5]
                try:
                    results = self.model.get_embeddings(batch)
                    batch_embeddings = [np.array(result.values) for result in results]
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    print(f"Error in batch {i//5}: {e}")
                    # For failed batches, try one by one
                    for single_text in batch:
                        try:
                            result = self.model.get_embeddings([single_text])[0]
                            embeddings.append(np.array(result.values))
                        except Exception as e2:
                            print(f"Error embedding text: {e2}")
                            # Use zeros as fallback for failed embeddings
                            embeddings.append(np.zeros(self.dimension))
            
            if not embeddings:
                raise ValueError("No embeddings were generated")
                
            return np.array(embeddings)
            
        except Exception as e:
            print(f"Error in embed_text: {e}")
            if not hasattr(self, '_local_fallback_warned'):
                print("Falling back to local model for text embeddings")
                self._local_fallback_warned = True
            self._init_local_model()
            return self.local_model.encode(text, convert_to_numpy=True)
    
    def embed_image(self, image: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """Generate embeddings for images.
        
        Args:
            image: PIL Image or list of PIL Images
            
        Returns:
            numpy.ndarray: Image embeddings
        """
        if self.use_local:
            if not hasattr(self, '_image_warning_shown'):
                print("Warning: Using text embeddings for images (local mode)")
                self._image_warning_shown = True
            # Convert images to text descriptions (simplified)
            if isinstance(image, list):
                text_descriptions = [f"An image with shape {img.size}" for img in image]
            else:
                text_descriptions = f"An image with shape {image.size}"
            return self.embed_text(text_descriptions)
            
        if not isinstance(image, list):
            image = [image]
            
        embeddings = []
        for img in image:
            try:
                response = self.model.embed_content(
                    content=img,
                    task_type="retrieval_document",
                    title="Image"
                )
                if 'embedding' in response:
                    embeddings.append(response['embedding'])
                else:
                    # Try different response format
                    embeddings.append(response['values'] if 'values' in response else response)
            except Exception as e:
                print(f"Error generating image embedding: {e}")
                # Fall back to text description
                text_desc = f"An image with shape {img.size}"
                return self.embed_text([text_desc] * len(image))
                
        return np.array(embeddings)
    
    def embed_video(self, video_path: str) -> np.ndarray:
        """Generate embeddings for video by sampling frames.
        
        Args:
            video_path: Path to video file
            
        Returns:
            numpy.ndarray: Video embeddings (average of frame embeddings)
        """
        # This is a simplified implementation
        # In production, you'd want to sample frames and average their embeddings
        raise NotImplementedError("Video embedding not implemented in GeminiEmbeddings")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text documents.
        
        This method is maintained for compatibility with LangChain's interface.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embeddings, one for each document
        """
        embeddings = self.embed_text(texts)
        return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings


class EmbeddingManager:
    """Manages embedding generation with Gemini."""
    
    def __init__(
        self, 
        api_key: str = None,
        model_name: str = "gemini-pro",
        use_vision: bool = False
    ):
        """Initialize the embedding manager.
        
        Args:
            api_key: Google AI API key. If None, will look for GOOGLE_API_KEY in environment.
            model_name: Base name of the Gemini model to use.
            use_vision: Whether to use the vision model for image processing.
        """
        if use_vision:
            model_name = "gemini-pro-vision"
            
        self.embeddings = GeminiEmbeddings(api_key=api_key, model_name=model_name)
        self.dimension = self.embeddings.dimension
        self.vector_store = None
        self._documents = []
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate text embeddings."""
        return self.embeddings.embed_text(text)
    
    def embed_image(self, image: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """Generate image embeddings."""
        return self.embeddings.embed_image(image)
    
    def embed_video(self, video_path: str) -> np.ndarray:
        """Generate video embeddings."""
        return self.embeddings.embed_video(video_path)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.dimension
        
    def get_relevant_documents(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """Get documents most relevant to the query using simple in-memory storage.
        
        Args:
            query: The query string
            k: Number of documents to return
            **kwargs: Additional arguments (ignored)
            
        Returns:
            List of relevant documents
        """
        try:
            # Use local similarity search which works with in-memory storage
            return self._local_similarity_search(query, k)
            
        except Exception as e:
            print(f"Error in get_relevant_documents: {e}")
            return self._get_fallback_documents(k)
    
    def _ensure_document_list(self, items) -> List[Document]:
        """Ensure the result is a list of Document objects."""
        if not items:
            return []
            
        results = []
        for item in items:
            if isinstance(item, Document):
                results.append(item)
            elif isinstance(item, (str, bytes)):
                results.append(Document(page_content=str(item)))
            elif hasattr(item, 'page_content'):
                # Handle LangChain-like objects
                results.append(Document(
                    page_content=str(getattr(item, 'page_content', '')),
                    metadata=dict(getattr(item, 'metadata', {}))
                ))
            else:
                # Convert any other type to string
                results.append(Document(page_content=str(item)))
        return results
    
    def _local_similarity_search(self, query: str, k: int) -> List[Document]:
        """Perform similarity search using local embeddings."""
        try:
            # Ensure query is a string
            query = str(query) if query else ""
            
            # Get query embedding
            try:
                query_embedding = self.embeddings.embed_text(query)
                if isinstance(query_embedding, list):
                    query_embedding = np.array(query_embedding)
                elif hasattr(query_embedding, 'numpy'):
                    query_embedding = query_embedding.numpy()
                query_flat = query_embedding.flatten().astype(float)
            except Exception as e:
                print(f"Error getting query embedding: {e}")
                return self._get_fallback_documents(k)
            
            # Calculate similarity scores (cosine similarity)
            scores = []
            for i, doc in enumerate(self._documents):
                try:
                    # Safely get document content
                    doc_content = doc.get('page_content', '')
                    if not isinstance(doc_content, (str, bytes)):
                        doc_content = str(doc_content) if doc_content is not None else ""
                    
                    # Skip empty documents
                    if not doc_content.strip():
                        continue
                        
                    # Get document embedding
                    doc_embedding = self.embeddings.embed_text(doc_content)
                    
                    # Convert to numpy array if needed
                    if isinstance(doc_embedding, list):
                        doc_embedding = np.array(doc_embedding, dtype=float)
                    elif hasattr(doc_embedding, 'numpy'):
                        doc_embedding = doc_embedding.numpy().astype(float)
                    
                    # Ensure proper shape and type
                    doc_flat = doc_embedding.flatten().astype(float)
                    
                    # Calculate cosine similarity
                    norm_q = np.linalg.norm(query_flat) + 1e-10
                    norm_d = np.linalg.norm(doc_flat) + 1e-10
                    similarity = np.dot(query_flat, doc_flat) / (norm_q * norm_d)
                    
                    scores.append((float(similarity), i))
                except Exception as e:
                    print(f"Error processing document {i}: {e}")
                    continue
            
            # Sort by score and return top k
            scores.sort(reverse=True, key=lambda x: x[0])
            top_indices = [i for _, i in scores[:k] if 0 <= i < len(self._documents)]
            
            # Return Document objects with metadata
            results = []
            for j, idx in enumerate(top_indices):
                doc = self._documents[idx]
                results.append(Document(
                    page_content=str(doc.get('page_content', '')),
                    metadata=dict(doc.get('metadata', {}), score=float(scores[j][0]) if j < len(scores) else 0.0)
                ))
            return results
            
        except Exception as e:
            print(f"Error in local similarity search: {e}")
            return self._get_fallback_documents(k)
    
    def initialize_vector_store(self, persist_directory: str = None, **kwargs):
        """Initialize a simple in-memory vector store.
        
        Args:
            persist_directory: Not used, kept for compatibility
            **kwargs: Additional arguments (ignored)
        """
        print("Initializing in-memory document store...")
        self.vector_store = None
        self._documents = []
    
    def _get_fallback_documents(self, k: int) -> List[Document]:
        """Get fallback documents when search fails."""
        if not hasattr(self, '_documents') or not self._documents:
            return []
            
        return [
            Document(
                page_content=str(doc.get('page_content', '')),
                metadata=dict(doc.get('metadata', {}))
            ) for doc in self._documents[:k] if doc.get('page_content')
        ]
    
    def add_documents(self, documents: Union[Document, List[Document], Dict, List[Dict]], 
                             summaries: Optional[List[str]] = None, **kwargs):
        """Add documents to in-memory storage.
        
        Args:
            documents: Single or list of Document objects or dicts to add
            summaries: Optional list of summaries corresponding to the documents
            **kwargs: Additional arguments (ignored)
            
        Returns:
            List of document indices
        """
        if not documents:
            return []
            
        # Initialize documents list if it doesn't exist
        if not hasattr(self, '_documents'):
            self._documents = []
            
        # Convert single document to list
        if isinstance(documents, (Document, dict)):
            documents = [documents]
            
        start_idx = len(self._documents)
            
        # Convert dictionaries to Document objects if needed
        for i, doc in enumerate(documents):
            if isinstance(doc, dict):
                # Handle dict input
                page_content = doc.get('page_content', '')
                metadata = doc.get('metadata', {})
                
                # Add summary to metadata if provided
                if summaries and i < len(summaries):
                    metadata['summary'] = summaries[i]
                
                # Store raw metadata as string for reference
                if 'raw_metadata' not in metadata:
                    metadata['raw_metadata'] = str(metadata)
                    
                self._documents.append({
                    'page_content': page_content,
                    'metadata': metadata
                })
            elif isinstance(doc, Document):
                # Handle Document object
                metadata = dict(doc.metadata)
                if summaries and i < len(summaries):
                    metadata['summary'] = str(summaries[i])
                
                # Store raw metadata as string for reference
                if 'raw_metadata' not in metadata:
                    metadata['raw_metadata'] = str(doc.metadata)
                    
                self._documents.append({
                    'page_content': doc.page_content,
                    'metadata': metadata
                })
            else:
                print(f"Warning: Unsupported document type: {type(doc)}")
                continue
        
        print(f"Stored {len(self._documents) - start_idx}/{len(documents)} documents in memory")
        
        # Return list of indices as document IDs for in-memory storage
        return list(range(start_idx, len(self._documents)))
