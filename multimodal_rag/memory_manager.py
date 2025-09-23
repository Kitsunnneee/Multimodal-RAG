"""Memory management for the Multimodal RAG system using Mem0 with Google's Gemini."""
from typing import Dict, List, Optional, Any, Callable
import os
import vertexai
from vertexai.language_models import TextEmbeddingModel
from mem0 import Memory, MemoryClient
from mem0.retrievers import VectorRetriever
from mem0.memory import MemoryConfig
from google.cloud import aiplatform
import logging

from .config import EMBEDDING_MODEL_NAME, VERTEX_AI_PROJECT, VERTEX_AI_LOCATION, MODEL_NAME

# Set up logging
logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages long-term and short-term memory for the RAG system using Mem0."""
    
    def __init__(self, collection_name: str = "multimodal_rag", use_hosted: bool = False):
        """Initialize the MemoryManager.
        
        Args:
            collection_name: Name of the memory collection to use.
            use_hosted: If True, use the hosted Mem0 service. If False, use local mode.
        """
        self.collection_name = collection_name
        self.use_hosted = use_hosted
        self._initialize_memory()
    
    def _get_embedding_function(self) -> Callable[[List[str]], List[List[float]]]:
        """Get the appropriate embedding function based on the configuration."""
        if self.use_hosted:
            # For hosted mode, we'll use the default embedding function from Mem0
            logger.info("Using hosted Mem0 with default embeddings")
            return None  # Let Mem0 handle embeddings
            
        # For local mode, use Vertex AI's embeddings
        logger.info("Initializing local Mem0 with Vertex AI embeddings")
        
        # Initialize Vertex AI
        if not VERTEX_AI_PROJECT or not VERTEX_AI_LOCATION:
            raise ValueError("VERTEX_AI_PROJECT and VERTEX_AI_LOCATION must be set for local mode")
            
        vertexai.init(project=VERTEX_AI_PROJECT, location=VERTEX_AI_LOCATION)
        
        # Initialize the embedding model
        embeddings = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        
        def embed_texts(texts: List[str]) -> List[List[float]]:
            """Embed a list of texts using Vertex AI's text embedding model."""
            try:
                logger.debug(f"Generating embeddings for {len(texts)} texts")
                embeddings_result = embeddings.get_embeddings(texts)
                return [embedding.values for embedding in embeddings_result]
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                raise
                
        return embed_texts
    
    def _initialize_memory(self):
        """Initialize the Mem0 memory system."""
        if self.use_hosted:
            # Initialize hosted Mem0 client
            self._initialize_hosted_memory()
        else:
            # Initialize local Mem0 with Vertex AI
            self._initialize_local_memory()
    
    def _initialize_hosted_memory(self):
        """Initialize the hosted Mem0 client."""
        mem0_api_key = os.getenv("MEM0_API_KEY")
        if not mem0_api_key:
            raise ValueError("MEM0_API_KEY environment variable must be set for hosted mode")
            
        # Initialize the hosted client
        self.client = MemoryClient(api_key=mem0_api_key)
        logger.info("Initialized hosted Mem0 client")
        
        # For hosted mode, we'll use the client directly
        self.memory = self.client
    
    def _initialize_local_memory(self):
        """Initialize the local Mem0 with Vertex AI embeddings."""
        embedding_function = self._get_embedding_function()
        
        # Configure memory
        config = MemoryConfig(
            collection_name=self.collection_name,
            embedding_function=embedding_function,
            retriever=VectorRetriever(
                top_k=5,
                distance_metric="cosine"
            ),
            persist_directory=os.path.join(os.getcwd(), "memory_store"),
            use_async=True  # Enable async operations for better performance
        )
        
        # Initialize memory
        self.memory = Memory(config=config)
        logger.info("Initialized local Mem0 with Vertex AI embeddings")
    
    def add_memory(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> str:
        """Add a memory to the memory store.
        
        Args:
            content: The content to remember.
            metadata: Optional metadata associated with the memory.
            document_id: Optional ID of the document this memory is associated with.
            
        Returns:
            The ID of the created memory.
        """
        if metadata is None:
            metadata = {}
            
        if document_id:
            metadata["document_id"] = document_id
            
        memory_id = self.memory.add(
            content=content,
            metadata=metadata
        )
        
        return memory_id
    
    def search_memories(
        self, 
        query: str, 
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant memories.
        
        Args:
            query: The search query.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filters to apply to the search.
            
        Returns:
            List of relevant memories with their scores.
        """
        results = self.memory.search(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        return [
            {
                "content": r["content"],
                "metadata": r["metadata"],
                "score": r["score"]
            }
            for r in results
        ]
    
    def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the conversation history for a specific conversation.
        
        Args:
            conversation_id: The ID of the conversation.
            limit: Maximum number of messages to return.
            
        Returns:
            List of conversation messages.
        """
        return self.search_memories(
            query="conversation history",
            filter_metadata={"conversation_id": conversation_id},
            top_k=limit
        )
    
    def clear_memories(self, filter_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Clear memories matching the filter criteria.
        
        Args:
            filter_metadata: Optional metadata filters to select which memories to clear.
            
        Returns:
            True if operation was successful.
        """
        if filter_metadata is None:
            # Clear all memories if no filter is provided
            self.memory.clear()
        else:
            # Find and delete specific memories
            memories = self.search_memories("*", top_k=1000, filter_metadata=filter_metadata)
            for mem in memories:
                self.memory.delete(mem["metadata"]["id"])
                
        return True
