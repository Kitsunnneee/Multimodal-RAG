"""Main RAG system implementation for multimodal retrieval and generation."""
import asyncio
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

from .config import (
    PROJECT_ID,
    LOCATION,
    MODEL_NAME,
    TOKEN_LIMIT,
    USE_GCS,
)

from .document_processor import DocumentProcessor
from .vector_store import QdrantVectorStore
from .memory_manager import MemoryManager, display_image


class MultimodalRAG:
    """Multimodal RAG system for document search and question answering."""
    
    def __init__(
        self,
        project_id: str = PROJECT_ID,
        location: str = LOCATION,
        model_name: str = MODEL_NAME,
        token_limit: int = TOKEN_LIMIT,
        use_gcs: bool = False,  # Not used with Qdrant
    ):
        """Initialize the Multimodal RAG system.
        
        Args:
            project_id: Google Cloud project ID (for LLM)
            location: Google Cloud region (for LLM)
            model_name: Name of the model to use for generation
            token_limit: Maximum number of tokens for model responses
            use_gcs: Kept for backward compatibility, not used with Qdrant
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.token_limit = token_limit
        self.use_gcs = use_gcs
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.llm = None
        self.chain = None
        self.embedding_model = None
        self.vector_store = None
    
    def initialize(
        self,
        index_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
    ) -> None:
        """Initialize the RAG system components.
        Args:
            index_id: Optional custom index ID (only used with GCS)
            endpoint_id: Optional custom endpoint ID (only used with GCS)
        """
        # Initialize the embedding model
        print("Initializing embedding model...")
        try:
            import os
            from google.oauth2 import service_account
            
            # Path to your service account key file
            credentials_path = os.path.join(os.path.dirname(__file__), "..", "elite-thunder-461308-f7-cc85c56bb209.json")
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            
            # Initialize the embedding model with explicit credentials
            self.embedding_model = VertexAIEmbeddings(
                model_name="text-embedding-005",
                project=self.project_id,
                location=self.location,
                credentials=credentials
            )
            print("Embedding model initialized successfully")
            
            # Initialize the vector store with the embedding model
            self.vector_store = QdrantVectorStore(
                collection_name="multimodal_rag",
                embedding_model=self.embedding_model,
                location=self.location
            )
            print("Vector store initialized successfully")
            
        except Exception as e:
            print(f"Error initializing embedding manager: {e}")
            print("Please ensure you have set up your Google Cloud credentials correctly.")
            print("You can set the GOOGLE_APPLICATION_CREDENTIALS environment variable to point to your service account key file.")
            raise
        
        # Initialize the LLM
        try:
            print(f"Initializing LLM with model: {self.model_name}")
            # Use the same credentials for the LLM
            self.llm = ChatVertexAI(
                model_name=self.model_name,
                project=self.project_id,
                location=self.location,
                max_output_tokens=self.token_limit,
                temperature=0.0,
                credentials=credentials  # Use the same credentials as above
            )
            print("LLM initialized successfully")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            raise
        # Create the RAG chain
        print("Creating RAG chain...")
        self._create_rag_chain()
        print("RAG system initialization complete")
    
    async def retrieve_documents(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: The query string
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call initialize() first.")
            
        try:
            # Get relevant documents from the vector store
            docs = await self.vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    async def _arun_retrieve_documents(self, inputs):
        """Async wrapper for retrieve_documents to be used in the chain."""
        question = inputs.get("question", "")
        k = inputs.get("k", 4)
        try:
            # Directly await the async retrieve_documents method
            return await self.retrieve_documents(question, k=k)
        except Exception as e:
            print(f"Error in _arun_retrieve_documents: {e}")
            return []
    
    def _create_rag_chain(self) -> None:
        """Create the RAG chain with prompt template and model."""
        from langchain_core.runnables import RunnableLambda
        from langchain_core.runnables.passthrough import RunnablePassthrough
        
        # Create a simple wrapper that will be used in the chain
        async def retrieve_docs_wrapper(inputs):
            try:
                # Directly await the async retrieval
                return await self._arun_retrieve_documents(inputs)
            except Exception as e:
                import traceback
                print(f"Error in retrieve_docs_wrapper: {e}")
                print(traceback.format_exc())
                return []
        
        # Create a runnable that properly handles the async operation
        retrieval_chain = RunnableLambda(
            lambda x: asyncio.run(retrieve_docs_wrapper(x)),
            name="retrieve_documents"
        )
        
        # Define the prompt template
        prompt_template = """You are a helpful assistant that answers questions based on the provided context.
        
Context:
{context}
        
Chat History:
{chat_history}
        
Question: {question}
        
Please provide a helpful response based on the context above."""
        
        # Create the prompt template
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "chat_history"]
        )
        
        # Define the RAG chain with proper async handling
        self.chain = (
            {
                "context": RunnablePassthrough() | retrieval_chain,
                "question": lambda x: x["question"],
                "chat_history": lambda x: x.get("chat_history", []),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    async def _format_prompt(self, data: Dict[str, Any]) -> str:
        """Format the prompt with context and question.
        
        Returns:
            Formatted prompt string
        """
        try:
            # Extract context and question from the input data
            context = data.get("context", [])
            
            # If context is a list of documents, extract their content
            if isinstance(context, list) and all(isinstance(doc, Document) for doc in context):
                context = "\n\n".join([doc.page_content for doc in context if doc.page_content])
            
            question = data.get("question", "")
            chat_history = data.get("chat_history", [])
            
            # Format the context for the prompt
            context_str = ""
            if context:
                if isinstance(context, list):
                    context_str = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in context])
                else:
                    context_str = str(context)
            
            # Format chat history
            chat_history_str = ""
            for msg in chat_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    chat_history_str += f"\nUser: {content}"
                elif role == "assistant":
                    chat_history_str += f"\nAssistant: {content}"
            
            # Create the prompt with the context and chat history
            prompt = f"""You are a helpful assistant that answers questions based on the provided context.
            
Context:
{context_str}
            
{chat_history_str}
            
Question: {question}
            
Please provide a helpful response based on the context above."""
            
            return prompt.strip()
            
        except Exception as e:
            print(f"Error in _format_prompt: {e}")
            import traceback
            traceback.print_exc()
            # Return a basic prompt if formatting fails
            return f"You are a helpful assistant.\n\nQuestion: {question}"
    
    def add_documents(self, file_paths: Union[str, Path, List[Union[str, Path]]], **kwargs) -> Dict[str, int]:
        """Add documents to the vector store.
        
        Args:
{{ ... }}
            file_paths: Single file path or list of file paths to process and add
            **kwargs: Additional arguments for document processing
            
        Returns:
            Dictionary with counts of added documents by type
        """
        import asyncio
        
        async def _add_docs_async():
            nonlocal file_paths
            
            # Convert single file path to list if needed
            if isinstance(file_paths, (str, Path)):
                file_paths = [file_paths]
                
            print(f"Starting to process {len(file_paths)} files...")
            
            # Process documents
            all_elements = {
                'texts': [],
                'tables': [],
                'images': []
            }
            
            for file_path in file_paths:
                elements = self.document_processor.process_file(file_path, **kwargs)
                all_elements['texts'].extend(elements.get('texts', []))
                all_elements['tables'].extend(elements.get('tables', []))
                all_elements['images'].extend(elements.get('images', []))
            
            print(f"Processed elements - texts: {len(all_elements['texts'])}, "
                  f"tables: {len(all_elements['tables'])}, "
                  f"images: {len(all_elements['images'])}")
            
            elements = all_elements
            
            # Initialize counters
            texts_added = 0
            tables_added = 0
            images_added = 0
            
            # Extract IDs from kwargs or use defaults
            user_id = kwargs.get('user_id', 'default_user')
            agent_id = kwargs.get('agent_id', 'default_agent')
            run_id = kwargs.get('run_id', str(uuid.uuid4()))
            
            # Add required IDs to all documents
            for doc_type in ["texts", "tables", "images"]:
                if elements.get(doc_type):
                    for doc in elements[doc_type]:
                        if not hasattr(doc, 'metadata') or not doc.metadata:
                            doc.metadata = {}
                        doc.metadata.update({
                            'user_id': user_id,
                            'agent_id': agent_id,
                            'run_id': run_id,
                            'doc_type': doc_type.rstrip('s')  # 'texts' -> 'text', 'tables' -> 'table', etc.
                        })
            
            # Add text documents
            if elements.get("texts"):
                print("Adding text documents to vector store...")
                try:
                    await self.vector_store.add_documents(elements["texts"])
                    texts_added = len(elements["texts"])
                    print(f"Successfully added {texts_added} text documents")
                except Exception as e:
                    print(f"Error adding text documents: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            # Add tables
            if elements.get("tables"):
                print("Adding tables to vector store...")
                try:
                    await self.vector_store.add_documents(elements["tables"])
                    tables_added = len(elements["tables"])
                    print(f"Successfully added {tables_added} tables")
                except Exception as e:
                    print(f"Error adding tables: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            # Add images (store as text with metadata)
            if elements.get("images"):
                print("Adding images to vector store...")
                try:
                    await self.vector_store.add_documents(elements["images"])
                    images_added = len(elements["images"])
                    print(f"Successfully added {images_added} images")
                except Exception as e:
                    print(f"Error adding images: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                
            result = {
                "texts": texts_added,
                "tables": tables_added,
                "images": images_added,
                "total": texts_added + tables_added + images_added
            }
            print(f"Document addition complete. Result: {result}")
            return result
            
        try:
            # Run the async function in the event loop
            return asyncio.run(_add_docs_async())
            
        except Exception as e:
            print(f"Error in add_documents: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return empty result with error
            return {"texts": 0, "tables": 0, "images": 0, "total": 0, "error": str(e)}
    
    async def _generate_summaries(self, texts: List[str]) -> List[str]:
        """Generate summaries for a list of texts."""
        from .document_processor import generate_summaries
        return await generate_summaries(texts, model_name=self.model_name, use_qdrant=True)
    async def _generate_image_summaries(self, image_docs: List[Union[Document, str]]) -> List[str]:
        """Generate summaries for a list of image documents.
        
        Args:
            image_docs: List of Document objects containing image data in metadata
            
        Returns:
            List of image summaries
        """
        if not image_docs:
            return []
        
        # Create a temporary LLM for image summarization
        llm = ChatVertexAI(
            model_name=self.model_name,
            project=self.project_id,
            location=self.location,
            max_output_tokens=self.token_limit,
            temperature=0.0,
        )
        
        # Generate summaries
        summaries = []
        for doc in image_docs:
            try:
                # Get image data from document metadata
                if not hasattr(doc, 'metadata') or 'image_data' not in doc.metadata:
                    print("Document missing image data in metadata")
                    summaries.append("Image data not found")
                    continue
                
                img_data = doc.metadata['image_data']
                
                # Handle case where image data is already base64 encoded
                if not isinstance(img_data, str):
                    print("Image data is not a string, skipping...")
                    summaries.append("Invalid image data")
                    continue
                
                # Clean up the base64 string
                img_data = img_data.strip()
                
                # Remove data URL prefix if present
                if img_data.startswith('data:image/'):
                    img_data = img_data.split(',', 1)[-1]
                
                # Remove any non-base64 characters
                import re
                img_data = re.sub(r'[^a-zA-Z0-9+/=]', '', img_data)
                
                # Ensure proper length (must be multiple of 4)
                padding = len(img_data) % 4
                if padding:
                    img_data += '=' * (4 - padding)
                
                # Try to validate base64
                import base64
                try:
                    # Decode and re-encode to ensure valid base64
                    decoded = base64.b64decode(img_data, validate=True)
                    # Use the re-encoded version to ensure consistency
                    img_data = base64.b64encode(decoded).decode('ascii')
                except Exception as e:
                    print(f"Invalid base64 image data: {str(e)[:100]}")
                    print(f"Image data length: {len(img_data)}")
                    print(f"First 50 chars: {img_data[:50]}...")
                    summaries.append("Invalid image data")
                    continue
                
                # Create the message with proper formatting
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe this image in detail for retrieval."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}},
                    ]
                )
                
                # Get the response
                response = llm.invoke([message])
                summaries.append(response.content)
                
            except Exception as e:
                import traceback
                print(f"Error generating summary for image: {str(e)}")
    async def query(
        self,
        question: str,
        chat_history: Optional[List[dict[str, str]]] = None,
        return_context: bool = False,
        **kwargs
    ) -> dict[str, Any]:
        """Query the RAG system with a question.
        
        Args:
            question: The user's question
            chat_history: List of previous chat messages in the format [{"role": "user"|"assistant", "content": "message"}]
            return_context: Whether to include the retrieved context in the response
            **kwargs: Additional arguments for the query
            
        Returns:
            Dictionary containing the answer and optionally the context
        """
        if not self.chain:
            raise ValueError("RAG system not initialized. Call initialize() first.")
            
        # Prepare the input data
        input_data = {"question": question}
        if chat_history:
            input_data["chat_history"] = chat_history
        
        try:
            # Get the answer from the chain using the async method
            answer = await self.chain.ainvoke(input_data)
            
            # If requested, get the context as well
            context = []
            if return_context:
                docs = await self.retrieve_documents(question)
                context = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in docs
                ]
            
            # Format the response
            response = {
                "answer": answer if isinstance(answer, str) else str(answer),
                "sources": []
            }
            
            if return_context:
                response["context"] = context
                
            return response
            
        except Exception as e:
            import traceback
            print(f"Error in query: {e}")
            print(traceback.format_exc())
            return {
                "answer": "Sorry, I encountered an error while processing your request.",
                "sources": []
            }
