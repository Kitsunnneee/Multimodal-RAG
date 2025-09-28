"""Vector store implementation using Qdrant with multimodal support."""
import base64
import io
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, BinaryIO

from qdrant_client import models

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictionServiceClient
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.documents import Document
from PIL import Image
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny

logger = logging.getLogger(__name__)

class MultimodalEmbeddings:
    """Handles multimodal embeddings for text and images."""
    
    def __init__(self, project_id: Optional[str] = None, location: str = "us-central1"):
        """Initialize the multimodal embeddings.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        
        # Initialize Vertex AI
        aiplatform.init(project=self.project_id, location=self.location)
        
        # Text embedding model
        self.text_embedding_model = VertexAIEmbeddings(
            model_name="text-embedding-004",
            project=self.project_id,
            location=location
        )
        
        # Multimodal embedding model
        self.multimodal_endpoint = f"projects/{self.project_id}/locations/{self.location}/publishers/google/models/multimodalembedding@001"
        self.client_options = {
            "api_endpoint": f"{self.location}-aiplatform.googleapis.com"
        }
        self.client = PredictionServiceClient(client_options=self.client_options)
        
        # Store the text embedding model's embed_query and embed_documents methods
        self.embed_query = self.textembedding_model.embed_query
        self.embed_documents = self.textembedding_model.embed_documents
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Vertex AI's text embedding model."""
        try:
            # First try using the multimodal endpoint
            instance = {
                "text": text
            }
            instances = [json_format.ParseDict(instance, Value())]
            
            # Get the prediction
            response = self.client.predict(
                endpoint=self.multimodal_endpoint,
                instances=instances
            )
            
            # Extract the text embedding
            embedding = response.predictions[0]['text_embedding']
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating text embedding with multimodal model: {str(e)}")
            # Fallback to the text-only model if multimodal fails
            try:
                # Use embed_query for single text
                result = self.embed_query(text)
                if isinstance(result, list):
                    return result
                # Handle case where embed_query returns a numpy array or other type
                return result.tolist() if hasattr(result, 'tolist') else list(result)
            except Exception as e2:
                logger.error(f"Error with text embedding fallback: {str(e2)}")
                raise
    
    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        if not texts:
            return []
            
        try:
            # Try getting embeddings in a batch
            if len(texts) == 1:
                # For single text, use get_text_embedding for consistency
                return [self.get_text_embedding(texts[0])]
                
            # Use embed_documents for multiple texts
            results = self.text_embedding_model.embed_documents(texts)
            
            # Ensure we return a list of lists
            if results and len(results) > 0 and not isinstance(results[0], list):
                return [result.tolist() if hasattr(result, 'tolist') else list(result) 
                       for result in results]
            return results
            
        except Exception as e:
            logger.error(f"Error getting embeddings for {len(texts)} texts: {str(e)}")
            logger.info("Falling back to one-by-one processing...")
            # Fall back to one-by-one processing if batch fails
            return [self.get_text_embedding(text) for text in texts]
    
    def _prepare_image(self, image_input: Union[str, Image.Image, bytes]) -> bytes:
        """Prepare image for embedding generation."""
        if isinstance(image_input, str):
            with open(image_input, "rb") as f:
                return f.read()
        elif isinstance(image_input, Image.Image):
            img_byte_arr = io.BytesIO()
            image_input.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()
        elif isinstance(image_input, bytes):
            return image_input
        else:
            raise ValueError("Unsupported image input type. Expected path (str), PIL Image, or bytes.")

    def get_image_embedding(self, image_input: Union[str, Image.Image, bytes]) -> List[float]:
        """Get embedding for an image using Google's Multimodal Embeddings.
        
        Args:
            image_input: Path to image, PIL Image object, or image bytes
            
        Returns:
            List of floats representing the image embedding
        """
        try:
            # Prepare image bytes
            image_bytes = self._prepare_image(image_input)
            
            # Create the instance with the image
            instance = {
                "image": {
                    "bytesBase64Encoded": base64.b64encode(image_bytes).decode("utf-8")
                }
            }
            instances = [json_format.ParseDict(instance, Value())]
            
            # Get the prediction
            response = self.client.predict(
                endpoint=self.multimodal_endpoint,
                instances=instances
            )
            
            # Extract the image embedding
            embedding = response.predictions[0]['image_embedding']
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating image embedding: {str(e)}")
            raise
    
    def encode_image_to_base64(self, image_path: Union[str, Image.Image]) -> str:
        """Encode image to base64 string."""
        if isinstance(image_path, str):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        elif isinstance(image_path, Image.Image):
            buffered = io.BytesIO()
            image_path.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        raise ValueError("Unsupported image format")

class QdrantVectorStore:
    """Vector store implementation using Qdrant with multimodal support."""
    
    def __init__(
        self,
        collection_name: str = "multimodal_rag",
        embedding_model: Optional[Any] = None,
        location: str = "us-central1",
        project_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize the Qdrant vector store with multimodal support.
        
        Args:
            collection_name: Name of the collection to use or create.
            embedding_model: The embedding model to use for generating embeddings.
                            If None, will create a default MultimodalEmbeddings instance.
            location: Google Cloud region (for multimodal embeddings).
            project_id: Google Cloud project ID (for multimodal embeddings).
        """
        self.collection_name = collection_name
        self.location = location
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        
        # Initialize embedding model
        if embedding_model is None:
            self.embedding_model = MultimodalEmbeddings(
                project_id=self.project_id,
                location=self.location
            )
        else:
            self.embedding_model = embedding_model
        
        # Initialize Qdrant Cloud client with proper timeout and retry handling
        self.client = None
        self._initialize_qdrant_client(**kwargs)
        
        # Initialize or get collection with multimodal support
        self._initialize_collection()
    
    def _initialize_qdrant_client(self, **kwargs):
        """Initialize Qdrant client with fallback options.
        
        Reads configuration from environment variables:
        - QDRANT_URL: The URL of the Qdrant instance (e.g., 'your-instance.aws.cloud.qdrant.io')
        - QDRANT_API_KEY: The API key for authentication
        - QDRANT_PORT: (Optional) Port to use (default: 6333 for REST, 6334 for gRPC)
        """
        connection_successful = False
        
        # Get configuration from environment variables
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            error_msg = (
                "QDRANT_URL and QDRANT_API_KEY environment variables must be set. "
                "Please set these environment variables with your Qdrant Cloud credentials."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Remove any credentials from kwargs to prevent passing them to QdrantClient
        kwargs.pop('credentials', None)
        
        # Try REST API first
        try:
            logger.info("Attempting to connect to Qdrant Cloud using REST API...")
            self.client = QdrantClient(
                url=qdrant_url,
                port=int(os.getenv("QDRANT_PORT", "6333")),
                api_key=qdrant_api_key,
                prefer_grpc=False,
                timeout=30.0,
                **kwargs
            )
            # Test the connection
            collections = self.client.get_collections()
            logger.info(f"✅ Successfully connected to Qdrant Cloud using REST API. Found {len(collections.collections)} collections.")
            connection_successful = True
            
        except Exception as e:
            logger.warning(f"REST API connection failed: {str(e)}")
            logger.info("Falling back to gRPC...")
            
            # If REST fails, try gRPC
            try:
                self.client = QdrantClient(
                    url=qdrant_url,
                    port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),  # Default gRPC port
                    api_key=qdrant_api_key,
                    prefer_grpc=True,
                    timeout=30.0,
                    **kwargs
                )
                # Test the connection
                collections = self.client.get_collections()
                logger.info(f"✅ Successfully connected to Qdrant Cloud using gRPC. Found {len(collections.collections)} collections.")
                connection_successful = True
                
            except Exception as grpc_e:
                logger.error(f"gRPC connection also failed: {str(grpc_e)}")
        
        # If both REST and gRPC fail, fall back to in-memory
        if not connection_successful:
            logger.warning("❌ All connection attempts to Qdrant Cloud failed")
            logger.info("Falling back to local in-memory Qdrant instance")
            self.client = QdrantClient(":memory:")
    
    def _initialize_collection(self):
        """Initialize or get the Qdrant collection with multimodal support."""
        from qdrant_client.http import models
        from qdrant_client.http.models import Distance, VectorParams
        
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            
            if self.collection_name not in collection_names:
                # Define vectors configuration for multimodal support
                vectors_config = {
                    "text": VectorParams(
                        size=768,  # Adjust based on your text embedding model
                        distance=Distance.COSINE
                    ),
                    "image": VectorParams(
                        size=512,  # Adjust based on your image embedding model
                        distance=Distance.COSINE
                    )
                }
                
                # Create new collection with the correct configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config
                )
                logger.info(f"✅ Created new multimodal collection: {self.collection_name}")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            logger.info("Falling back to in-memory collection")
            # Fallback to in-memory collection with default config
            self.client = QdrantClient(":memory:")
            try:
                vectors_config = VectorParams(
                    size=768,  # Default size
                    distance=Distance.COSINE
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config
                )
                logger.info(f"Created in-memory collection: {self.collection_name}")
            except Exception as e2:
                logger.error(f"Failed to create in-memory collection: {e2}")
                raise
    
    async def add_documents(
        self, 
        documents: Union[Document, List[Document]], 
        **kwargs
    ) -> List[str]:
        """Add documents to the vector store with support for multimodal data.
        
        Args:
            documents: Single Document or list of Document objects to add.
                      Each document can contain text, image, or both.
            
        Returns:
            List of document IDs that were added.
            
        Raises:
            ValueError: If no documents are provided or if documents are invalid.
        """
        if not documents:
            logger.warning("No documents provided to add_documents")
            return []
            
        # Convert single document to list for uniform processing
        if isinstance(documents, Document):
            documents = [documents]
            
        try:
            points = []
            
            for idx, doc in enumerate(documents):
                if not isinstance(doc, Document):
                    logger.warning(f"Skipping non-Document object at index {idx}")
                    continue
                    
                # Extract content and metadata
                content = doc.page_content
                metadata = doc.metadata or {}
                
                # Add required IDs to metadata
                metadata.update({
                    'user_id': metadata.get('user_id', 'default_user'),
                    'agent_id': metadata.get('agent_id', 'default_agent'),
                    'run_id': metadata.get('run_id', str(uuid.uuid4()))
                })
                
                # Generate a unique ID for the document if not exists
                doc_id = metadata.get("id", str(uuid.uuid4()))
                metadata['id'] = doc_id
                
                # Initialize vectors dictionary
                vectors = {}
                
                # Handle text content
                if content and isinstance(content, str) and content.strip():
                    try:
                        # Use the embed_query method directly
                        text_embedding = self.embedding_model.embed_query(content)
                        if not isinstance(text_embedding, list):
                            text_embedding = text_embedding.tolist() if hasattr(text_embedding, 'tolist') else list(text_embedding)
                        vectors["text"] = text_embedding
                    except Exception as e:
                        logger.error(f"Error generating text embedding for document {idx}: {e}")
                        logger.exception("Full traceback:")
                
                # Handle image content if present in metadata
                image_embedding = None
                if "image_path" in metadata:
                    try:
                        image_path = metadata["image_path"]
                        if isinstance(image_path, (str, Path)) and os.path.exists(image_path):
                            # Encode image to base64 for storage
                            image_base64 = self.embedding_model.encode_image_to_base64(image_path)
                            metadata["image_base64"] = image_base64
                            
                            # Generate image embedding
                            image_embedding = self.embedding_model.get_image_embedding(image_path)
                            vectors["image"] = image_embedding
                    except Exception as e:
                        logger.error(f"Error processing image for document {idx}: {e}")
                
                # Only add the point if we have at least one valid vector
                if vectors:
                    point = {
                        "id": idx,
                        "vector": {},
                        "payload": {
                            "text": doc.page_content,
                            "metadata": doc.metadata
                        }
                    }
                    
                    # Add text vector if available
                    if hasattr(doc, 'embedding') and doc.embedding is not None:
                        point["vector"]["text"] = doc.embedding
                    
                    # Add image vector if available in metadata
                    if "imageembedding" in doc.metadata:
                        point["vector"]["image"] = doc.metadata["imageembedding"]
                    
                    # Convert to PointStruct
                    points.append(models.PointStruct(
                        id=point["id"],
                        vector=point["vector"],
                        payload=point["payload"]
                    ))
            # Use upsert to handle both new and existing documents
            # Remove any kwargs that aren't valid for upsert
            upsert_kwargs = {k: v for k, v in kwargs.items() 
                           if k in ['wait', 'max_retries', 'timeout', 'shard_key_selector']}
            
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                **upsert_kwargs
            )
            
            logger.info(f"Successfully added/updated {len(points)} documents in Qdrant")
            return [str(point.id) for point in points]
                
        except Exception as e:
            logger.error(f"Error in add_documents: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    async def similarity_search(
        self,
        query: Union[str, Dict[str, Any], Image.Image],
        k: int = 4,
        filter_by: Optional[Dict] = None,
        search_in: str = "text",
        **kwargs
    ) -> List[Document]:
        """Search for similar documents using text or image query.
        
        Args:
            query: The query string (for text) or a dictionary with 'image_path' (for image).
            k: Number of results to return.
            filter_by: Optional filter to apply to the search.
            search_in: Which vector space to search in ('text' or 'image').
            
        Returns:
            List of Document objects.
            
        Raises:
            ValueError: If the search_in parameter is invalid or query is invalid.
        """
        if not self.client:
            logger.error("Qdrant client is not initialized")
            return []
            
        try:
            # Generate query vector based on input type
            query_vector = None
            
            if search_in == "text" and isinstance(query, str):
                # Text search
                logger.debug(f"Performing text search for query: {query[:100]}...")
                try:
                    if hasattr(self.embedding_model, 'get_text_embedding'):
                        query_embedding = self.embedding_model.get_text_embedding(query)
                    else:
                        query_embedding = self.embedding_model.embed_query(query)
                    
                    if not isinstance(query_embedding, (list, tuple)) or not all(isinstance(x, (int, float)) for x in query_embedding):
                        logger.error(f"Invalid embedding format: {type(query_embedding)}")
                        return []
                        
                    query_vector = ("text", query_embedding)
                    logger.debug("Generated text query vector successfully")
                    
                except Exception as e:
                    logger.error(f"Error generating text embedding: {str(e)}")
                    return []
                
            elif search_in == "image":
                # Image search - handle different input types
                logger.debug("Performing image similarity search")
                try:
                    if isinstance(query, dict) and "image_path" in query:
                        image_path = query["image_path"]
                        if isinstance(image_path, (str, Path)) and os.path.exists(image_path):
                            query_embedding = self.embedding_model.get_image_embedding(image_path)
                            query_vector = ("image", query_embedding)
                    elif isinstance(query, Image.Image):
                        query_embedding = self.embedding_model.get_image_embedding(query)
                        query_vector = ("image", query_embedding)
                    
                    if not query_vector:
                        logger.warning("Invalid image query provided")
                        return []
                        
                    logger.debug("Generated image query vector successfully")
                    
                except Exception as e:
                    logger.error(f"Error generating image embedding: {str(e)}")
                    return []
            else:
                logger.error(f"Unsupported search_in value: {search_in}")
                return []
            
            if not query_vector:
                logger.error("Failed to generate query vector")
                return []
            
            # Extract vector name and embedding
            vector_name, embedding = query_vector
            
            # Build filter if provided
            qdrant_filter = None
            if filter_by:
                must_conditions = []
                for key, value in filter_by.items():
                    if isinstance(value, (list, tuple, set)):
                        must_conditions.append(
                            models.FieldCondition(
                                key=f"metadata.{key}",
                                match=models.MatchAny(any=list(value))
                            )
                        )
                    else:
                        must_conditions.append(
                            models.FieldCondition(
                                key=f"metadata.{key}",
                                match=models.MatchValue(value=value)
                            )
                        )
                
                if must_conditions:
                    qdrant_filter = models.Filter(must=must_conditions)
            
            # Prepare query parameters
            query_params = {
                'collection_name': self.collection_name,
                'limit': k,
                'with_payload': True,
                'with_vectors': False
            }
            
            # Always specify the vector name explicitly
            # The collection requires either 'image' or 'text' as the vector name
            vector_to_use = vector_name if vector_name in ["image", "text"] else "text"
            
            # Create a named vector query
            # For Qdrant client, we need to use 'query' parameter with a tuple of (vector_name, vector)
            query_params['query'] = (vector_to_use, embedding)
            
            # Add filter if it exists
            if qdrant_filter:
                query_params['query_filter'] = qdrant_filter
            
            # Add additional parameters if they exist in kwargs
            for param in ['score_threshold', 'offset', 'consistency', 'shard_key_selector']:
                if param in kwargs:
                    query_params[param] = kwargs[param]
            
            # Execute the search using query_points
            search_results = self.client.query_points(**query_params)
            
            logger.info(f"✅ Found {len(search_results)} results in collection {self.collection_name}")
            
            # Convert results to Document objects
            documents = []
            for result in search_results:
                try:
                    payload = result.payload or {}
                    metadata = payload.get("metadata", {})
                    
                    # Add score to metadata
                    metadata["score"] = result.score
                    
                    # Handle different payload structures
                    if "text" in payload:
                        page_content = payload["text"]
                    elif "page_content" in payload:
                        page_content = payload["page_content"]
                    else:
                        page_content = str(payload)
                    
                    # Create document
                    doc = Document(
                        page_content=page_content,
                        metadata=metadata
                    )
                    documents.append(doc)
                    
                except Exception as e:
                    logger.error(f"Error processing search result: {str(e)}")
                    continue
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in similarity_search: {str(e)}", exc_info=True)
            return []
            
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.
        
        Returns:
            Dictionary with collection information.
        """
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            return {
                'name': collection_info.name,
                'status': collection_info.status,
                'vectors_count': collection_info.vectors_count,
                'points_count': collection_info.points_count,
                'config': collection_info.config.dict() if hasattr(collection_info.config, 'dict') else str(collection_info.config)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """Delete the collection.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Successfully deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize the vector store with multimodal support
        vector_store = QdrantVectorStore(
            collection_name="multimodal_demo",
            location="us-central1"
        )
        
        # Example 1: Add text documents
        text_docs = [
            Document(
                page_content="A beautiful sunset over the mountains",
                metadata={"type": "text", "source": "example"}
            ),
            Document(
                page_content="A group of people hiking in the forest",
                metadata={"type": "text", "source": "example"}
            )
        ]
        
        # Add text documents
        text_ids = await vector_store.add_documents(text_docs)
        print(f"Added {len(text_ids)} text documents")
        
        # Example 2: Add image documents (requires actual image files)
        # Uncomment and modify paths as needed
        """
        image_docs = [
            Document(
                page_content="A scenic mountain landscape",
                metadata={
                    "type": "image",
                    "image_path": "path/to/mountain.jpg"
                }
            ),
            Document(
                page_content="People enjoying nature",
                metadata={
                    "type": "image",
                    "image_path": "path/to/hiking.jpg"
                }
            )
        ]
        image_ids = await vector_store.add_documents(image_docs)
        print(f"Added {len(image_ids)} image documents")
        """
        
        # Example 3: Text search
        print("\nText search results:")
        text_results = await vector_store.similarity_search(
            "scenic views",
            k=2,
            search_in="text"
        )
        
        for i, doc in enumerate(text_results, 1):
            print(f"{i}. {doc.page_content[:100]}... (Score: {doc.metadata.get('score', 0):.3f})")
        
        # Example 4: Image search (requires actual image files)
        # Uncomment and modify as needed
        """
        print("\nImage search results:")
        image_results = await vector_store.similarity_search(
            {"image_path": "path/to/query_image.jpg"},
            k=2,
            search_in="image"
        )
        
        for i, doc in enumerate(image_results, 1):
            print(f"{i}. {doc.page_content[:100]}... (Score: {doc.metadata.get('score', 0):.3f})")
        """
    
    # Run the async main function
    asyncio.run(main())
