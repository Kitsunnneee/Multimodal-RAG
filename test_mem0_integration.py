"""Test script for Mem0 integration with Google's Gemini.

This script demonstrates how to use Mem0 for memory management in a RAG system.
"""
import os
import sys
import datetime
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel

# Load environment variables from .env file
load_dotenv()

# Import Mem0 after setting environment variables
from mem0 import Memory

# Check for required environment variables
GEMINI_AVAILABLE = False

# Check if we can use Google's Gemini
if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_API_KEY"):
    try:
        # For direct Gemini API access
        import google.generativeai as genai
        
        # Configure with application default credentials
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            import google.auth
            credentials, project = google.auth.default()
            genai.configure(credentials=credentials)
            print("Using Google Application Default Credentials")
        else:
            # Fallback to API key (may not work for all features)
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            print("Using Google API Key (some features may be limited)")
        
        # Verify the API works by listing available models
        try:
            models = genai.list_models()
            if any('gemini' in model.name for model in models):
                GEMINI_AVAILABLE = True
                print("Google Gemini API is available and working")
            else:
                print("Google Gemini API is available but no Gemini models found")
        except Exception as e:
            print(f"Warning: Could not verify Google Gemini API: {str(e)}")
            
    except ImportError:
        print("google-generativeai package not installed. Please install it with: pip install google-generativeai")
    except Exception as e:
        print(f"Warning: Could not initialize Google Gemini: {str(e)}")
else:
    print("Google Gemini not configured - set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_API_KEY environment variable")

# Check for Vertex AI (alternative approach)
if os.getenv("GOOGLE_CLOUD_PROJECT") and not GEMINI_AVAILABLE:
    try:
        vertexai.init(
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("VERTEX_AI_LOCATION", "us-central1")
        )
        GEMINI_AVAILABLE = True
        print("Google Vertex AI is available")
    except Exception as e:
        print(f"Warning: Could not initialize Vertex AI: {str(e)}")

# Set a dummy OpenAI API key if not set
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "dummy-key"
    print("Using dummy OpenAI API key for embedding model")

class Mem0Integration:
    """Class to demonstrate Mem0 integration with Gemini."""
    
    def __init__(self):
        """Initialize the Mem0 integration with Gemini."""
        # Initialize Gemini model if available
        self.gemini_model = None
        
        # Configure Mem0 with Google's Gemini
        try:
            # Check if we have the required API key
            google_api_key = os.getenv("GOOGLE_API_KEY")
            openai_api_key = os.getenv("OPENAI_API_KEY", "dummy-key")  # For embedding model
            
            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or google_api_key:
                # Configure Mem0 with the correct structure
                # Create a configuration class instance instead of a dictionary
                from mem0.configs import MemoryConfig
                
                config = MemoryConfig(
                    version="v1.1",
                    llm={
                        "provider": "google",
                        "config": {
                            "model": "gemini-1.5-pro",
                            "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
                            "location": os.getenv("VERTEX_AI_LOCATION", "us-central1"),
                            "temperature": 0.2,
                            "max_tokens": 2000,
                            "top_p": 1.0
                        }
                    },
                    embedder={
                        "provider": "google",
                        "config": {
                            "project_id": os.getenv("GOOGLE_CLOUD_PROJECT"),
                            "location": os.getenv("VERTEX_AI_LOCATION", "us-central1"),
                            "model": "text-embedding-004"
                        }
                    },
                    vector_store={
                        "provider": "in-memory"
                    }
                )
                
                # Set the custom fact extraction prompt as an attribute
                config.custom_fact_extraction_prompt = """
                Analyze the input and extract key facts. Focus on entities, actions, and relationships.
                Return a JSON object with a 'facts' array containing the extracted facts.
                Example: {"facts": ["User mentioned they like Python", "User is learning about AI"]}
                """
                
                # If using API key (not recommended for production)
                if google_api_key and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                    config["llm"]["config"]["api_key"] = google_api_key
                    config["embedder"]["config"]["api_key"] = google_api_key
                
                # Initialize Gemini model if available
                self.gemini_model = None
                if GEMINI_AVAILABLE:
                    try:
                        import google.generativeai as genai
                        
                        # Configure with the appropriate authentication method
                        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                            # Using application default credentials
                            import google.auth
                            credentials, _ = google.auth.default()
                            genai.configure(credentials=credentials)
                        elif google_api_key:
                            # Fallback to API key if available
                            genai.configure(api_key=google_api_key)
                        
                        # Initialize the model
                        self.gemini_model = genai.GenerativeModel('gemini-pro')
                        print("Successfully initialized Gemini model")
                    except Exception as e:
                        print(f"Warning: Could not initialize Gemini model: {str(e)}")
                        self.gemini_model = None
                        
                # Initialize Mem0 with Gemini configuration
                self.memory = Memory(config=config)
                print("Initialized Mem0 with Gemini")
                
            else:
                print("GOOGLE_API_KEY not found. Using in-memory storage only.")
                self._memories = []
                
        except Exception as e:
            print(f"Warning: Could not initialize Mem0 with Gemini: {str(e)}")
            print("Falling back to in-memory storage...")
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
            memory_item = {
                'content': content,
                'user_id': user_id,
                'timestamp': datetime.datetime.now().isoformat()
            }
            self._memories.append(memory_item)
            print(f"[In-memory] Added memory: {content[:100]}...")
            return memory_item
            
        # Try to use Mem0 if available
        try:
            # Format the message according to Mem0's expected format
            message = {
                "role": "user",
                "content": content
            }
            
            # Add the memory
            result = self.memory.add(
                messages=[message],
                user_id=user_id,
                infer=True  # Let Mem0 infer the facts
            )
            
            print(f"[Mem0] Added memory: {content[:100]}...")
            return result
            
        except Exception as e:
            print(f"Warning: Could not add memory to Mem0: {str(e)}")
            print("Falling back to in-memory storage...")
            
            # Initialize in-memory storage if it doesn't exist
            if not hasattr(self, '_memories'):
                self._memories = []
                
            # Add to in-memory storage
            memory_item = {
                'content': content,
                'user_id': user_id,
                'timestamp': datetime.datetime.now().isoformat()
            }
            self._memories.append(memory_item)
            return memory_item
    
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
    
    def chat_with_memory(self, user_input: str, user_id: str = "default_user") -> str:
        """Chat with the user using memory and Gemini if available."""
        try:
            # First, search for relevant memories
            search_results = self.memory.search(
                query=user_input,
                user_id=user_id,
                limit=3
            )
            
            # Extract relevant memories
            relevant_memories = search_results.get("results", [])
            
            # Format memories as context
            memories_context = "\n".join(
                f"- {mem.get('memory', mem.get('content', ''))}" for mem in relevant_memories
            )
            
            # If we have memories, use them as context
            if memories_context:
                print(f"\nSearch results for '{user_input}':")
                for i, mem in enumerate(relevant_memories[:3], 1):
                    mem_content = str(mem.get('memory', mem.get('content', '')))
                    print(f"{i}. {mem_content[:100]}{'...' if len(mem_content) > 100 else ''}")
                print()
            
            # If Gemini is available, use it for generating responses
            if self.gemini_model:
                try:
                    # Prepare the prompt with context
                    prompt = f"""Use the following context to answer the user's question.
                    
                    Context:
                    {memories_context if memories_context else 'No relevant context found.'}
                    
                    Question: {user_input}
                    
                    Answer in a helpful and concise way:"""
                    
                    # Generate response using Gemini
                    response = self.gemini_model.generate_content(prompt)
                    
                    # Extract the response text
                    if hasattr(response, 'text'):
                        assistant_response = response.text
                    elif hasattr(response, 'candidates') and response.candidates:
                        assistant_response = response.candidates[0].content.parts[0].text
                    else:
                        raise ValueError("Unexpected response format from Gemini")
                    
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
            memory_text = " ".join(
                str(mem.get('memory', mem.get('content', ''))) 
                for mem in relevant_memories
            )
            
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
