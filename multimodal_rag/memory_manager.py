"""Memory management for the Multimodal RAG system using Mem0."""
from typing import Dict, List, Optional, Any, Callable
import os
import logging
from mem0 import Memory, MemoryClient
import base64
from io import BytesIO
from PIL import Image
import streamlit as st

# Set up logging
logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages long-term and short-term memory for the RAG system using Mem0."""
    
    def __init__(self, collection_name: str = "multimodal_rag", use_hosted: bool = False, user_id: Optional[str] = None):
        """Initialize the MemoryManager.
        
        Args:
            collection_name: Name of the memory collection to use.
            use_hosted: If True, use the hosted Mem0 service. If False, use local mode.
            user_id: Optional user ID for multi-tenant scenarios.
        """
        self.collection_name = collection_name
        self.use_hosted = use_hosted
        self.user_id = user_id or "default_user"
        self.client = self._initialize_client()
    
    def _get_memory_key(self, memory_type: str) -> str:
        """Generate a unique key for a memory."""
        return f"{self.collection_name}_{memory_type}"
    
    def _build_local_config(self) -> Dict[str, Any]:
        """Build configuration for Mem0 with Qdrant as the vector store."""
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": self.collection_name,
                    "host": os.getenv("QDRANT_URL", "localhost"),
                    "port": int(os.getenv("QDRANT_PORT", "6333")),
                    "api_key": os.getenv("QDRANT_API_KEY"),
                    "embedding_model_dims": 768,  # Adjust based on your embedding model
                    "on_disk": False
                }
            },
            "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": os.getenv("GOOGLE_API_KEY"),
        }
    },
            "llm": {
                "provider": "gemini",
                "config": {
                    "model": "gemini-2.0-flash-001",
                    "api_key": os.getenv("GOOGLE_API_KEY"),
                    "temperature": 0.2,
                    "max_tokens": 2000,
                    "top_p": 1.0
                }
            }
        }
    
    def _initialize_client(self):
        """Initialize the Mem0 client based on configuration."""
        try:
            if self.use_hosted:
                api_key = os.getenv("MEM0_API_KEY")
                if not api_key:
                    raise ValueError("MEM0_API_KEY environment variable must be set for hosted mode")
                logger.info("Initializing hosted Mem0 client")
                return MemoryClient(api_key=api_key)
            
            logger.info("Initializing local Mem0 instance with Qdrant")
            
            # Ensure Google Cloud credentials are set
            google_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not google_creds_path or not os.path.exists(google_creds_path):
                raise ValueError(
                    "GOOGLE_APPLICATION_CREDENTIALS environment variable must point to a valid service account key file"
                )
            
            # Set the project ID for Vertex AI
            os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("GOOGLE_CLOUD_PROJECT", "elite-thunder-461308")
            
            # Build and validate the configuration
            config = self._build_local_config()
            
            # Initialize the client with the configuration
            return Memory.from_config(config)
                
        except Exception as e:
            logger.error(f"Failed to initialize Mem0: {str(e)}")
            raise

    def add_memory(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> str:
        """Add a memory to the memory store.
        
        Args:
            content: The content to remember.
            metadata: Optional metadata associated with the memory.
            document_id: Optional ID of the document this memory is associated with.
            user_id: Optional user ID for multi-tenant scenarios.
            agent_id: Optional agent ID for multi-agent scenarios.
            run_id: Optional run ID for tracking specific execution runs.
            
        Returns:
            The ID of the created memory.
            
        Raises:
            ValueError: If required IDs are missing.
        """
        try:
            if not self.client:
                raise RuntimeError("Memory client not initialized")
            
            # Use instance user_id if not provided
            user_id = user_id or self.user_id
            
            # Ensure at least one ID is provided
            if not any([user_id, agent_id, run_id]):
                raise ValueError(
                    "At least one of 'user_id', 'agent_id', or 'run_id' must be provided. "
                    f"Current values - user_id: {user_id}, agent_id: {agent_id}, run_id: {run_id}"
                )
            
            # Prepare metadata
            memory_metadata = metadata or {}
            if document_id:
                memory_metadata['document_id'] = document_id
            
            # Add IDs to metadata if provided
            if user_id:
                memory_metadata['user_id'] = user_id
            if agent_id:
                memory_metadata['agent_id'] = agent_id
            if run_id:
                memory_metadata['run_id'] = run_id
            
            # Prepare the message according to Mem0 v1.0.0 API
            message = {
                'role': 'user',
                'content': content
            }
            
            # Prepare the add arguments with required IDs
            add_kwargs = {
                'messages': [message],
                'metadata': memory_metadata,
            }
            
            # Add required IDs to the request
            if user_id:
                add_kwargs['user_id'] = user_id
            if agent_id:
                add_kwargs['agent_id'] = agent_id
            if run_id:
                add_kwargs['run_id'] = run_id
            
            logger.debug(f"Adding memory with kwargs: {add_kwargs}")
            
            # Add the memory
            result = self.client.add(**add_kwargs)
            
            # Handle the response format for Mem0 v1.0.0
            if result and isinstance(result, dict):
                res_list = result.get("results", [])
                if res_list and isinstance(res_list, list) and len(res_list) > 0:
                    mem_id = res_list[0].get("id")
                    if mem_id is not None:
                        memory_id = str(mem_id)
                        logger.info(f"Successfully added memory with ID: {memory_id}")
                        return memory_id
            
            logger.error(f"Failed to add memory. Response: {result}")
            return ""
            
        except Exception as e:
            logger.error(f"Error adding memory: {str(e)}")
            raise

    def search_memories(
        self, 
        query: str = "",
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant memories.
        
        Args:
            query: The search query string.
            top_k: Maximum number of results to return.
            filter_metadata: Optional metadata filters to apply to the search.
            
        Returns:
            List of relevant memories with their scores.
        """
        try:
            if not self.client:
                logger.warning("Memory client not initialized. Returning empty results.")
                return []
            
            # Prepare search parameters
            search_params = {
                'query': query,
                'limit': top_k,
                'metadata': filter_metadata or {},
                'user_id': self.user_id
            }
            
            # Execute search
            results = self.client.search(**search_params)
            
            # Format results
            memories = []
            for result in results:
                if isinstance(result, dict):
                    memories.append({
                        'id': result.get('id'),
                        'content': result.get('content', ''),
                        'metadata': result.get('metadata', {}),
                        'score': result.get('score', 0.0)
                    })
            
            return memories
            
        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}")
            return []

def display_image(image_data: str, width: int = 400) -> None:
    """Display an image from base64 data or file path in Streamlit.
    
    Args:
        image_data: Base64 encoded image data or file path
        width: Width to display the image (height will be scaled proportionally)
    """
    try:
        # If it's a file path
        if os.path.isfile(image_data):
            image = Image.open(image_data)
        # If it's base64 data
        elif image_data.startswith('data:image'):
            # Extract the base64 data
            header, encoded = image_data.split(',', 1)
            # Decode the base64 data
            image = Image.open(BytesIO(base64.b64decode(encoded)))
        else:
            # Try to decode as base64 directly
            try:
                image = Image.open(BytesIO(base64.b64decode(image_data)))
            except:
                st.error("Invalid image data format")
                return
        
        # Display the image
        st.image(image, width=width, use_column_width=False)
        
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

    def clear_memories(self, filter_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Clear memories matching the filter criteria.
        
        Args:
            filter_metadata: Optional metadata filters to select which memories to clear.
                           If None, clears all memories for the user (use with caution).
            
        Returns:
            bool: True if operation was successful, False otherwise.
        """
        try:
            if not self.client:
                logger.warning("Memory client not initialized. Cannot clear memories.")
                return False
            
            if filter_metadata is None:
                # Clear all memories for the user (use with caution)
                if self.user_id:
                    # If user_id is set, only clear memories for that user
                    self.client.clear(user_id=self.user_id)
                else:
                    # Otherwise clear everything (admin only)
                    self.client.clear()
                logger.info("Cleared all memories" + (f" for user {self.user_id}" if self.user_id else ""))
            else:
                # Find and delete specific memories
                memories = self.search_memories(
                    query="",  # Empty query to match all
                    top_k=1000,  # Adjust based on expected number of memories
                    filter_metadata=filter_metadata
                )
                
                # Delete each matching memory
                for memory in memories:
                    try:
                        if 'id' in memory:
                            self.client.delete(memory['id'])
                    except Exception as delete_error:
                        logger.warning(f"Failed to delete memory {memory.get('id')}: {str(delete_error)}")
                        continue
                
                logger.info(f"Cleared {len(memories)} memories matching filters")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing memories: {str(e)}")
            return False
