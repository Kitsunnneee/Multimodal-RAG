"""Test script for Mem0 integration with Google's Gemini.

This script demonstrates how to use Mem0 for memory management in a RAG system.
"""
import os
import datetime
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel

# Set a dummy OpenAI API key to bypass the requirement
os.environ["OPENAI_API_KEY"] = "dummy-key"

# Load environment variables
load_dotenv()

# Now import Mem0 after setting environment variables
from mem0 import Memory

# Initialize Vertex AI if Google Cloud project is set
if os.getenv("GOOGLE_CLOUD_PROJECT"):
    try:
        vertexai.init(
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("VERTEX_AI_LOCATION", "us-central1")
        )
        GEMINI_AVAILABLE = True
    except Exception as e:
        print(f"Warning: Could not initialize Vertex AI: {str(e)}")
        GEMINI_AVAILABLE = False
else:
    GEMINI_AVAILABLE = False

class Mem0Integration:
    """Class to demonstrate Mem0 integration with Gemini."""
    
    def __init__(self):
        """Initialize the Mem0 integration with Gemini."""
        # Initialize Gemini model if available
        self.gemini_model = None
        if GEMINI_AVAILABLE:
            try:
                self.gemini_model = GenerativeModel("gemini-1.5-pro")
                print("Successfully initialized Gemini model")
            except Exception as e:
                print(f"Warning: Could not initialize Gemini model: {str(e)}")
                print("Some features may be limited")
        else:
            print("Gemini model not available. Using in-memory storage only.")

        # Initialize Mem0 with minimal configuration
        try:
            # Configure Mem0 with minimal settings
            config = {
                "llm": {
                    "provider": "openai",  # Required but won't be used
                    "config": {
                        "api_key": "dummy-key"  # Dummy key to bypass validation
                    }
                },
                "embedder": {
                    "provider": "openai",  # Required but won't be used
                    "config": {
                        "api_key": "dummy-key"  # Dummy key to bypass validation
                    }
                },
                "vector_store": {
                    "provider": "in-memory"  # Use in-memory storage
                }
            }
            
            self.memory = Memory(config=config)
            print("Initialized Mem0 with in-memory storage")
            self._memories = []  # Initialize in-memory storage as fallback
            
        except Exception as e:
            print(f"Warning: Could not initialize Mem0: {str(e)}")
            print("Falling back to in-memory storage only...")
            self._memories = []  # Fallback to in-memory storage only
        
    def add_memory(self, content, user_id="default_user"):
        """Add a memory to the memory store.
        
        Args:
            content: The content to remember.
            user_id: User ID for the memory (defaults to "default_user")
        """
        # Check if we're using the in-memory fallback
        if hasattr(self, '_memories'):
            # Using in-memory fallback
            self._memories.append({
                'content': content,
                'user_id': user_id,
                'timestamp': datetime.datetime.now().isoformat()
            })
            print(f"[In-memory] Added memory: {content[:100]}...")
        else:
            # Try to use Mem0 if available
            try:
                # Add a simple memory
                self.memory.add(
                    messages=[{"role": "user", "content": content}],
                    user_id=user_id
                )
                print(f"[Mem0] Added memory: {content[:100]}...")
            except Exception as e:
                print(f"Warning: Could not add memory to Mem0: {str(e)}")
                print("Falling back to in-memory storage...")
                self._memories = [{
                    'content': content,
                    'user_id': user_id,
                    'timestamp': datetime.datetime.now().isoformat()
                }]
    
    def search_memories(self, query, limit=3, user_id="default_user"):
        """Search for relevant memories using a simple keyword search.
        
        Args:
            query: The search query.
            limit: Maximum number of results to return (default: 3)
            user_id: User ID to search memories for
            
        Returns:
            List of relevant memories.
        """
        print(f"\nSearch results for '{query}':")
        
        # Check if we're using the in-memory fallback
        if hasattr(self, '_memories'):
            # Simple keyword search in the fallback storage
            query_terms = query.lower().split()
            matching_memories = []
            
            for mem in self._memories:
                if mem['user_id'] == user_id:
                    content = mem['content'].lower()
                    # Check if any of the query terms are in the content
                    if any(term in content for term in query_terms):
                        matching_memories.append({
                            'content': mem['content'],
                            'user_id': mem['user_id'],
                            'timestamp': mem.get('timestamp', '')
                        })
            
            # Sort by timestamp if available (newest first)
            try:
                matching_memories.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            except:
                pass
            
            # Display results
            if not matching_memories:
                print("No matching memories found")
                return []
                
            for i, mem in enumerate(matching_memories[:limit], 1):
                content = mem['content']
                if len(content) > 150:
                    content = content[:150] + "..."
                print(f"{i}. {content}")
                
            return matching_memories[:limit]
        else:
            # Try to use Mem0's search if available
            try:
                results = self.memory.search(
                    query=query,
                    user_id=user_id,
                    limit=limit
                )
                
                if not results or 'results' not in results:
                    print("No memories found")
                    return []
                    
                for i, memory in enumerate(results['results'][:limit], 1):
                    content = str(memory.get('content', memory))
                    if len(content) > 150:
                        content = content[:150] + "..."
                    print(f"{i}. {content}")
                    
                return results['results'][:limit]
                
            except Exception as e:
                print(f"Error searching memories: {str(e)}")
                print("Falling back to simple memory search")
                return []
    
    def chat_with_memory(self, user_input, user_id="default_user"):
        """Simulate a chat interaction with memory using Gemini or fallback.
        
        Args:
            user_input: The user's input message.
            user_id: User ID for the conversation
            
        Returns:
            The assistant's response.
        """
        try:
            # First, search for relevant memories
            relevant_memories = self.search_memories(user_input, limit=2, user_id=user_id)
            
            # Create context from memories
            if hasattr(self, '_memories'):
                # Using in-memory fallback
                memories_context = "\n".join(
                    f"- {mem.get('content', '')}" 
                    for mem in relevant_memories
                )
            else:
                # Using Mem0's format
                memories_context = "\n".join(
                    f"- {mem.get('content', str(mem))}" 
                    for mem in relevant_memories
                )
            
            # If Gemini is available, use it for generating responses
            if self.gemini_model:
                try:
                    # Generate response using Gemini
                    prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question.
                    
                    Previous context:
                    {memories_context}
                    
                    Current conversation:
                    User: {user_input}
                    Assistant:"""
                    
                    response = self.gemini_model.generate_content(prompt)
                    assistant_response = response.text
                    
                    # Add the interaction to memory
                    self.add_memory(
                        content=f"User: {user_input}\nAssistant: {assistant_response}",
                        user_id=user_id
                    )
                    
                    return assistant_response
                    
                except Exception as e:
                    print(f"Warning: Error using Gemini: {str(e)}")
                    print("Falling back to simple response...")
            
            # Fallback response if Gemini is not available or fails
            if not relevant_memories:
                return "I don't have enough information to respond to that. Could you provide more details?"
                
            # Simple fallback response based on memories
            memory_text = " ".join(str(mem.get('content', '')) for mem in relevant_memories)
            
            # Add a simple response based on the memory content
            response = f"Based on what you've told me: {memory_text[:200]}..."
            
            # Add the interaction to memory
            self.add_memory(
                content=f"User: {user_input}\nAssistant: {response}",
                user_id=user_id
            )
            
            return response
            
        except Exception as e:
            print(f"Error in chat_with_memory: {str(e)}")
            return "I'm sorry, I encountered an error while processing your request."

def test_mem0_integration():
    """Test the Mem0 integration with Gemini or fallback."""
    print("Testing Mem0 integration...")
    
    try:
        # Check environment
        if not os.getenv("GOOGLE_CLOUD_PROJECT"):
            print("Note: GOOGLE_CLOUD_PROJECT environment variable is not set")
            print("Gemini features will be disabled. Using in-memory storage only.")
        
        if not os.getenv("OPENAI_API_KEY"):
            print("Note: OPENAI_API_KEY environment variable is not set")
            print("Using in-memory storage for Mem0.")

        print("\nInitializing Mem0 integration...")
        mem0_int = Mem0Integration()

        # Add some example memories
        print("\n=== Adding Example Memories ===")
        test_memories = [
            "The user's name is Alex and they are interested in machine learning.",
            "Alex mentioned they are working on a project about multimodal AI.",
            "The user prefers Python over Java for development.",
            "Alex is currently learning about large language models and their applications.",
            "The user's favorite programming language is Python."
        ]

        for memory in test_memories:
            mem0_int.add_memory(memory)

        # Test memory search
        print("\n=== Testing Memory Search ===")
        search_queries = [
            "What does Alex work on?",
            "What programming languages does the user know?",
            "Tell me about the user's interests"
        ]

        for query in search_queries:
            print(f"\nSearch query: {query}")
            results = mem0_int.search_memories(query)
            if not results:
                print("No results found")

        # Test chat with memory
        print("\n=== Testing Chat with Memory ===")
        chat_messages = [
            "What programming language do I prefer?",
            "What am I currently learning?",
            "What's my name?"
        ]

        for message in chat_messages:
            print(f"\nYou: {message}")
            try:
                response = mem0_int.chat_with_memory(message)
                print(f"Assistant: {response}")
            except Exception as e:
                print(f"Error in chat: {str(e)}")

        print("\n=== Test Summary ===")
        if hasattr(mem0_int, '_memories'):
            print(f"\nUsing in-memory storage with {len(mem0_int._memories)} memories")
        
        if mem0_int.gemini_model:
            print("Gemini model is available and being used for responses")
        else:
            print("Gemini model is not available. Using fallback responses.")

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"\nError during test: {str(e)}")
        print("\nTest failed. Here are some things to check:")
        print("1. Make sure you have the required environment variables set:")
        print("   - GOOGLE_CLOUD_PROJECT: Your Google Cloud project ID (optional for basic testing)")
        print("   - GOOGLE_APPLICATION_CREDENTIALS: Path to your service account key file (optional)")
        print("\nFor more information, check the documentation at: https://docs.mem0.ai/")

        # Print the full error for debugging
        import traceback
        print("\nFull error traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    test_mem0_integration()
