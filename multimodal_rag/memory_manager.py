"""Memory management for the Multimodal RAG system using Mem0."""
from typing import Dict, List, Optional, Any, Callable
import os
import logging
from mem0 import Memory, MemoryClient

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
        self.user_id = user_id
        self.client = self._initialize_client()
    
    def _build_local_config(self) -> Dict[str, Any]:
        """Build configuration for local Mem0 instance with Google multimodal embedding."""
        config = {
            "vector_store": {
                "provider": "faiss",
                "config": {
                    "collection_name": self.collection_name,
                    "path": "./local_vector_store",
                    "distance_strategy": "cosine"
                }
            },
            "embedder": {
                "provider": "vertexai",
                "config": {
                    "model": "text-embedding-004",
                    "memory_add_embedding_type": "RETRIEVAL_DOCUMENT",
                    "memory_update_embedding_type": "RETRIEVAL_DOCUMENT",
                    "memory_search_embedding_type": "RETRIEVAL_QUERY"
                }
            },
            "llm": {
                # Note: 'gemini' provider uses Vertex AI infrastructure for Gemini models
                "provider": "gemini",
                "config": {
                    "model": "gemini-2.0-flash-001",
                    "temperature": 0.2,
                    "max_tokens": 1000,
                    "top_p": 1.0
                }
            }
        }
        return config
    
    def _initialize_client(self):
        """Initialize the Mem0 client based on configuration."""
        try:
            if self.use_hosted:
                api_key = os.getenv("MEM0_API_KEY")
                if not api_key:
                    raise ValueError("MEM0_API_KEY environment variable must be set for hosted mode")
                logger.info("Initializing hosted Mem0 client")
                return MemoryClient(api_key=api_key)
            else:
                logger.info("Initializing local Mem0 instance")
                config = self._build_local_config()
                return Memory.from_config(config)
                
        except Exception as e:
            logger.error(f"Failed to initialize Mem0: {str(e)}")
            raise
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
        try:
            if not self.client:
                raise RuntimeError("Memory client not initialized")
            
            # Prepare metadata
            memory_metadata = metadata or {}
            if document_id:
                memory_metadata['document_id'] = document_id
            
            # Prepare the message according to Mem0 v1.0.0 API
            message = {
                'role': 'user',
                'content': content
            }
            
            # Prepare the add arguments
            if not self.user_id:
                raise ValueError("user_id must be set in MemoryManager configuration")
                
            add_kwargs = {
                'messages': [message],  # Must be a list of message objects
                'metadata': memory_metadata,
                'infer': True,  # Let Mem0 infer the facts
                'user_id': self.user_id  # Required by Mem0 API
            }
            
            # Add the memory
            result = self.client.add(**add_kwargs)
            
            # Handle the response format for Mem0 v1.0.0
            if result and isinstance(result, dict):
                res_list = result.get("results", [])
                if res_list and isinstance(res_list, list) and len(res_list) > 0:
                    mem_id = res_list[0].get("id")
                    if mem_id is not None:
                        memory_id = str(mem_id)
                        logger.debug(f"Added memory with ID: {memory_id}")
                        return memory_id
            
            logger.error("Failed to add memory: Invalid or empty response from Mem0")
            return ""
            
        except Exception as e:
            logger.error(f"Error adding memory: {str(e)}")
            # Don't raise the exception, just log it to keep the system working
            return ""
            
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
            search_kwargs = {
                'query': query,
                'limit': top_k,
                'metadata': filter_metadata or {}
            }
            
            if self.user_id:
                search_kwargs['user_id'] = self.user_id
            
            # Execute search
            results = self.client.search(**search_kwargs)
            
            # Format results in a consistent way
            memories = []
            for result in results:
                memory_data = {
                    'content': getattr(result, 'content', ''),
                    'metadata': getattr(result, 'metadata', {}),
                    'score': float(getattr(result, 'score', 0.0)),
                    'id': str(getattr(result, 'id', ''))
                }
                memories.append(memory_data)
            
            # Sort by score in descending order
            memories.sort(key=lambda x: x['score'], reverse=True)
            
            return memories
            
        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}")
            return []
    
    def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the conversation history for a specific conversation.
        
        Args:
            conversation_id: The ID of the conversation.
            limit: Maximum number of messages to return.
            
        Returns:
            List of conversation messages with metadata.
        """
        try:
            if not self.client:
                logger.warning("Memory client not initialized. Returning empty history.")
                return []
            
            # Prepare search parameters
            search_kwargs = {
                'metadata': {'conversation_id': conversation_id},
                'limit': limit,
                'sort': 'timestamp'  # Assuming timestamps are stored in metadata
            }
            
            if self.user_id:
                search_kwargs['user_id'] = self.user_id
            
            # Execute search
            results = self.client.search(**search_kwargs)
            
            # Format results
            messages = []
            for result in results:
                metadata = getattr(result, 'metadata', {})
                messages.append({
                    'content': getattr(result, 'content', ''),
                    'metadata': metadata,
                    'timestamp': metadata.get('timestamp'),
                    'id': str(getattr(result, 'id', ''))
                })
            
            # Sort by timestamp if available
            if all(msg['timestamp'] is not None for msg in messages):
                messages.sort(key=lambda x: x['timestamp'])
            
            return messages
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
    
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
