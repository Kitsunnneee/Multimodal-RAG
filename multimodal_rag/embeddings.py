"""Embedding management for the Multimodal RAG system."""
import os
import io
import base64
import logging
from typing import List, Optional, Union, Dict, Any
import numpy as np
from PIL import Image

from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GeminiEmbeddings(Embeddings):
    """Wrapper around Gemini embedding models."""
    
    def __init__(self, model_name: str = "text-embedding-004", **kwargs):
        """Initialize the Gemini embeddings.
        
        Args:
            model_name: Name of the Gemini model to use.
            **kwargs: Additional arguments to pass to the model.
        """
        self.model_name = model_name
        self.model = None
        self.initialize_model()
    def initialize_model(self):
        """Initialize the Gemini model with API key authentication."""
        try:
            import google.generativeai as genai
            
            # Try to get API key from multiple possible environment variables
            api_key = (
                os.getenv('GEMINI_API_KEY') or 
                os.getenv('GOOGLE_API_KEY') or
                os.getenv('GEMINI_KEY')
            )
            
            if not api_key:
                error_msg = """
                Gemini API key not found. Please set one of these environment variables:
                - GEMINI_API_KEY (recommended)
                - GOOGLE_API_KEY
                - GEMINI_KEY
                
                You can set it in your terminal using:
                export GEMINI_API_KEY='your-api-key-here'
                
                Or create a .env file in your project root with:
                GEMINI_API_KEY=your-api-key-here
                """
                raise ValueError(error_msg)
            
            # Configure the client with the API key
            genai.configure(api_key=api_key)
            self.client = genai
            
            # Test the connection with a simple request
            try:
                models = genai.list_models()
                if not models:
                    logger.warning("No models found. The API key may not have access to any models.")
                else:
                    logger.info("Successfully connected to Gemini API with API key")
                    
            except Exception as e:
                logger.error(f"Failed to connect to Gemini API: {e}")
                raise ValueError(
                    "Failed to connect to Gemini API. "
                    "Please check your GEMINI_API_KEY and ensure it's valid."
                ) from e
                
        except ImportError as e:
            raise ImportError(
                "Could not import google.generativeai package. "
                "Please install it with: pip install google-generativeai"
            ) from e
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Gemini.
        
        Args:
            texts: List of text documents to embed.
            
        Returns:
            List of embeddings, one for each document.
            
        Raises:
            ValueError: If there's an error embedding the documents.
        """
        if not hasattr(self, 'client') or self.client is None:
            self.initialize_model()
        
        try:
            embeddings = []
            for i, text in enumerate(texts):
                try:
                    # Use the correct method for text embeddings
                    response = self.client.embed_content(
                        model=f"models/{self.model_name}",
                        content={"parts": [{"text": text}]},
                        task_type="retrieval_document"
                    )
                    
                    if 'embedding' in response:
                        embeddings.append(response['embedding'])
                    else:
                        raise ValueError("Unexpected response format from embedding model")
                        
                except Exception as e:
                    logger.error(f"Error embedding document {i + 1}/{len(texts)}: {str(e)}")
                    # Re-raise the exception to be handled by the caller
                    raise
                    
            return embeddings
                
        except Exception as e:
            logger.error(f"Failed to embed documents: {str(e)}")
            raise ValueError(f"Error embedding documents: {str(e)}")
            
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using Gemini.
        
        Args:
            text: The query text to embed.
            
        Returns:
            The embedding for the query.
        """
        return self.embed_documents([text])[0]
    
    def _resize_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """Resize image to reduce its size while maintaining aspect ratio.
        
        Args:
            image: PIL Image to resize
            max_size: Maximum width or height of the resized image
            
        Returns:
            Resized PIL Image
        """
        width, height = image.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image
        
    def _compress_image(self, image: Image.Image, quality: int = 75, max_size: int = 768) -> bytes:
        """Compress image to reduce its file size while maintaining acceptable quality.
        
        Args:
            image: PIL Image to compress
            quality: Initial JPEG quality (1-100)
            max_size: Maximum dimension for the image
            
        Returns:
            Compressed image as bytes
        """
        img_byte_arr = io.BytesIO()
        
        # Convert to RGB if needed
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if needed
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Try different quality levels if needed
        for q in [quality, 65, 50, 40]:
            img_byte_arr = io.BytesIO()
            image.save(
                img_byte_arr, 
                format='JPEG', 
                quality=q,
                optimize=True,
                progressive=True,
                subsampling='4:4:4' if q > 70 else '4:2:0'
            )
            
            if len(img_byte_arr.getvalue()) < 34000:
                break
                
        return img_byte_arr.getvalue()
    
    def embed_image(self, image: Union[Image.Image, List[Image.Image]], max_retries: int = 3) -> List[List[float]]:
        """Embed an image or list of images using Gemini with retry logic.
        
        Args:
            image: PIL Image or list of PIL Images to embed.
            max_retries: Maximum number of retry attempts for each image
            
        Returns:
            List of embeddings, one for each image.
            
        Raises:
            ValueError: If there's an error embedding the images after all retries.
        """
        if not hasattr(self, 'client') or self.client is None:
            self.initialize_model()
            
        if isinstance(image, Image.Image):
            images = [image]
            return_single = True
        else:
            images = image
            return_single = False
            
        embeddings = []
        
        for img_idx, img in enumerate(images):
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # Make a copy to avoid modifying the original
                    img_copy = img.copy()
                    
                    # Resize and compress the image more aggressively with each attempt
                    target_size = 1024 // (attempt + 1)  # Reduce size with each retry
                    img_copy = self._resize_image(img_copy, max_size=target_size)
                    
                    # Try with lower quality on retries
                    quality = max(30, 80 - (attempt * 15))  # Start at 80%, decrease by 15% each retry
                    img_data = self._compress_image(img_copy, quality=quality, max_size=target_size)
                    
                    # Log image size for debugging
                    img_size_kb = len(img_data) / 1024
                    logger.info(f"Attempt {attempt + 1}: Image size: {img_size_kb:.1f}KB, Quality: {quality}%")
                    
                    # Generate embedding using the client
                    response = self.client.embed_content(
                        model=f"models/{self.model_name}",
                        content={
                            "parts": [{
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": base64.b64encode(img_data).decode('utf-8')
                                }
                            }]
                        },
                        task_type="retrieval_document"
                    )
                    
                    if 'embedding' in response:
                        embeddings.append(response['embedding'])
                        break  # Success, move to next image
                    else:
                        last_error = ValueError("Unexpected response format from embedding model")
                        
                except Exception as e:
                    last_error = e
                    if attempt == max_retries - 1:  # Last attempt
                        logger.error(f"Failed to embed image after {max_retries} attempts: {e}")
                        raise ValueError(f"Error embedding image {img_idx + 1}/{len(images)}: {str(e)}") from e
                    
                    # Exponential backoff before retry
                    import time
                    time.sleep(1 * (2 ** attempt))
            
            else:  # No break occurred in the for loop
                if last_error:
                    raise last_error
        
        return embeddings[0] if return_single else embeddings


class EmbeddingManager:
    """Manages embedding generation for the Multimodal RAG system."""
    
    def __init__(
        self, 
        model_name: str = "text-embedding-004",
        use_vision: bool = False,
        **kwargs
    ):
        """Initialize the embedding manager.
        
        Args:
            model_name: Name of the embedding model to use.
            use_vision: Whether to enable vision capabilities.
            **kwargs: Additional arguments to pass to the embedding model.
        """
        self.model_name = model_name
        self.use_vision = use_vision
        self.embedding_model = GeminiEmbeddings(model_name=model_name, **kwargs)
        self.dimension = self._get_embedding_dimension()
    
    def _get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        # Default dimensions for common models
        if "text-embedding-004" in self.model_name:
            return 768
        elif "multimodalembedding" in self.model_name.lower():
            return 1024
        else:
            # Default dimension
            return 768
    
    def embed_text(self, text: Union[str, List[str]]) -> List[List[float]]:
        """Embed text or a list of texts.
        
        Args:
            text: Text or list of texts to embed.
            
        Returns:
            List of embeddings, one for each input text.
        """
        if not text:
            return []
            
        if isinstance(text, str):
            return [self.embedding_model.embed_query(text)]
        else:
            return self.embedding_model.embed_documents(text)
    
    def embed_image(self, image: Union[Image.Image, List[Image.Image]]) -> List[List[float]]:
        """Embed an image or list of images.
        
        Args:
            image: PIL Image or list of PIL Images to embed.
            
        Returns:
            List of embeddings, one for each image.
            
        Raises:
            ValueError: If vision is not enabled or if there's an error embedding the images.
        """
        if not self.use_vision:
            raise ValueError("Vision capabilities are not enabled. Set use_vision=True when initializing EmbeddingManager.")
            
        if not isinstance(image, list):
            image = [image]
            
        return self.embedding_model.embed_image(image)
    
    def embed_video(self, video_path: str) -> List[float]:
        """Embed a video by sampling frames.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            Average embedding of the video frames.
            
        Raises:
            ValueError: If vision is not enabled or if there's an error processing the video.
        """
        if not self.use_vision:
            raise ValueError("Vision capabilities are not enabled. Set use_vision=True when initializing EmbeddingManager.")
            
        try:
            import cv2
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
                
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Sample frames (e.g., 1 frame per second)
            frame_indices = [int(i * fps) for i in range(0, int(frame_count / fps))]
            frames = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
            
            cap.release()
            
            if not frames:
                raise ValueError("No frames could be extracted from the video.")
                
            # Get embeddings for all frames
            frame_embeddings = self.embed_image(frames)
            
            # Return the average embedding
            return np.mean(frame_embeddings, axis=0).tolist()
            
        except ImportError:
            raise ImportError("OpenCV (cv2) is required for video processing. Install with: pip install opencv-python")
        except Exception as e:
            raise ValueError(f"Error processing video: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.dimension
        
    def initialize_vector_store(
        self,
        vector_store_type: str = 'local',
        persist_directory: str = None,
        pinecone_api_key: str = None,
        pinecone_index_name: str = 'multimodal-rag',
        pinecone_environment: str = 'gcp-starter',
        project_id: str = None,
        location: str = 'us-central1',
        index_id: str = None,
        endpoint_id: str = None,
        **kwargs
    ) -> None:
        """Initialize the vector store for storing and retrieving embeddings.
        
        Args:
            vector_store_type: Type of vector store to use ('local', 'pinecone', or 'gcs')
            persist_directory: Directory to persist the vector store (for 'local' type)
            pinecone_api_key: API key for Pinecone (required for 'pinecone' type)
            pinecone_index_name: Name of the Pinecone index
            pinecone_environment: Pinecone environment
            project_id: Google Cloud project ID (for 'gcs' type)
            location: Google Cloud region (for 'gcs' type)
            index_id: Custom index ID (for 'gcs' type)
            endpoint_id: Custom endpoint ID (for 'gcs' type)
            **kwargs: Additional arguments for the vector store
        """
        vector_store_type = vector_store_type.lower()
        
        if vector_store_type == 'pinecone':
            self._initialize_pinecone(
                api_key=pinecone_api_key,
                index_name=pinecone_index_name,
                environment=pinecone_environment,
                **kwargs
            )
        elif vector_store_type == 'gcs':
            self._initialize_gcs(
                project_id=project_id,
                location=location,
                index_id=index_id,
                endpoint_id=endpoint_id,
                **kwargs
            )
        else:  # Default to local
            self._initialize_local(
                persist_directory=persist_directory,
                **kwargs
            )
    
    def _initialize_pinecone(
        self,
        api_key: str,
        index_name: str = 'multimodal-rag',
        environment: str = 'gcp-starter',
        **kwargs
    ) -> None:
        """Initialize Pinecone vector store.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            environment: Pinecone environment (not used in newer versions)
            **kwargs: Additional arguments for Pinecone
        """
        try:
            from pinecone import Pinecone, ServerlessSpec
            from langchain_pinecone import PineconeVectorStore
            
            # Initialize Pinecone client
            pc = Pinecone(api_key=api_key)
            
            # Check if index exists
            existing_indexes = [index.name for index in pc.list_indexes()]
            
            # Create index if it doesn't exist
            if index_name not in existing_indexes:
                # Use ServerlessSpec for serverless indexes
                spec = ServerlessSpec(
                    cloud='aws',  # or 'gcp' or 'azure'
                    region='us-west-2'  # or other supported region
                )
                
                # Create the index
                pc.create_index(
                    name=index_name,
                    dimension=self.dimension,
                    metric='cosine',
                    spec=spec
                )
                
                print(f"Created new Pinecone index: {index_name}")
            
            # Get the index
            index = pc.Index(index_name)
            
            # Create the vector store using the index directly
            self.vector_store = PineconeVectorStore(
                index=index,
                embedding=self.embedding_model,
                text_key=kwargs.get('text_key', 'text'),
                namespace=kwargs.get('namespace')
            )
            
            print(f"Pinecone vector store initialized with index '{index_name}'")
            
        except ImportError as e:
            raise ImportError(
                "Could not import required Pinecone packages. "
                "Please install with `pip install pinecone-client langchain-community`. "
                f"Error: {str(e)}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Pinecone: {str(e)}\n"
                "Please check your Pinecone configuration and API key.\n"
                "Make sure you're using a compatible version of pinecone-client.\n"
                "Try: pip install pinecone-client==2.2.2"
            ) from e
    
    def _initialize_gcs(
        self,
        project_id: str,
        location: str = 'us-central1',
        index_id: str = None,
        endpoint_id: str = None,
        **kwargs
    ) -> None:
        """Initialize GCS vector store.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
            index_id: Custom index ID
            endpoint_id: Custom endpoint ID
            **kwargs: Additional arguments for GCS
        """
        try:
            from google.cloud import aiplatform
            from langchain_google_vertexai import VectorSearchVectorStore
            
            # Initialize Vertex AI
            aiplatform.init(project=project_id, location=location)
            
            # Create or connect to the index
            if index_id and endpoint_id:
                # Connect to existing index and endpoint
                self.vector_store = VectorSearchVectorStore.from_components(
                    project_id=project_id,
                    location=location,
                    index_id=index_id,
                    endpoint_id=endpoint_id,
                    **kwargs
                )
            else:
                # Create new index and endpoint
                self.vector_store = VectorSearchVectorStore(
                    project_id=project_id,
                    location=location,
                    embedding=self.embedding_model,
                    **kwargs
                )
            
            print("GCS vector store initialized successfully")
            
        except ImportError:
            raise ImportError(
                "Could not import google-cloud-aiplatform or langchain_google_vertexai. "
                "Please install with `pip install google-cloud-aiplatform langchain-google-vertexai`."
            )
    
    def _initialize_local(
        self,
        persist_directory: str = None,
        **kwargs
    ) -> None:
        """Initialize local vector store.
        
        Args:
            persist_directory: Directory to persist the vector store
            **kwargs: Additional arguments for the vector store
        """
        try:
            from langchain_community.vectorstores import Chroma
            
            if persist_directory:
                os.makedirs(persist_directory, exist_ok=True)
            
            self.vector_store = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=persist_directory,
                **kwargs
            )
            
            print(f"Local vector store initialized at {persist_directory or 'memory'}")
            
        except ImportError as e:
            raise ImportError(
                "Could not import required packages for local vector store. "
                "Please install with `pip install chromadb`. "
                f"Error: {str(e)}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize local vector store: {str(e)}") from e
    
    def get_relevant_documents(self, query: str, k: int = 5, use_hybrid: bool = True, **kwargs) -> List[Document]:
        """Retrieve relevant documents from the vector store using multimodal search.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            use_hybrid: Whether to use hybrid search (combine text and image search results)
            **kwargs: Additional arguments to pass to the vector store
            
        Returns:
            List of relevant documents
            
        Raises:
            RuntimeError: If the vector store is not initialized
        """
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize_vector_store() first.")
            
        try:
            # First, try to get results using text search
            text_docs = self.vector_store.similarity_search(
                query=query,
                k=k,
                **kwargs
            )
            
            # If hybrid search is enabled and we have embeddings, try image search
            if use_hybrid and hasattr(self, 'embeddings') and hasattr(self.embeddings, 'embed_query'):
                try:
                    # Get query embedding
                    query_embedding = self.embeddings.embed_query(query)
                    
                    # Search using vector similarity
                    vector_docs = self.vector_store.similarity_search_by_vector(
                        embedding=query_embedding,
                        k=k,
                        **kwargs
                    )
                    
                    # Combine and deduplicate results
                    combined_docs = {}
                    for doc in text_docs + vector_docs:
                        doc_id = doc.metadata.get('source', '') + str(doc.metadata.get('page', ''))
                        combined_docs[doc_id] = doc
                    
                    return list(combined_docs.values())[:k]
                    
                except Exception as e:
                    logger.warning(f"Error in hybrid search: {e}. Falling back to text search.")
                    return text_docs
            
            return text_docs
            
        except Exception as e:
            raise RuntimeError(f"Error retrieving documents: {str(e)}") from e
            
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            **kwargs: Additional arguments for the vector store
            
        Returns:
            List of document IDs
            
        Raises:
            RuntimeError: If the vector store is not initialized
        """
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize_vector_store() first.")
            
        try:
            # Add documents to the vector store
            doc_ids = self.vector_store.add_documents(documents, **kwargs)
            return doc_ids
            
        except Exception as e:
            raise RuntimeError(f"Error adding documents: {str(e)}") from e
            
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, **kwargs) -> List[str]:
        """Add texts to the vector store.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries (one per text)
            **kwargs: Additional arguments for the vector store
            
        Returns:
            List of document IDs
            
        Raises:
            RuntimeError: If the vector store is not initialized
        """
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize_vector_store() first.")
            
        try:
            # Add texts to the vector store
            doc_ids = self.vector_store.add_texts(texts, metadatas=metadatas, **kwargs)
            return doc_ids
            
        except Exception as e:
            raise RuntimeError(f"Error adding texts: {str(e)}") from e
