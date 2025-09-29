"""Main RAG system implementation for multimodal retrieval and generation."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_vertexai import ChatVertexAI

from .config import MODEL_NAME, TOKEN_LIMIT, PROJECT_ID, LOCATION, USE_GCS
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingManager
from .utils import split_image_text_types, display_image


class MultimodalRAG:
    """Multimodal RAG system for document search and question answering."""
    
    def __init__(
        self,
        project_id: str = PROJECT_ID,
        location: str = LOCATION,
        model_name: str = MODEL_NAME,
        token_limit: int = TOKEN_LIMIT,
        use_gcs: bool = USE_GCS,
        vector_store_type: str = 'local',  # 'local', 'gcs', or 'pinecone'
        pinecone_api_key: str = None,
        pinecone_index_name: str = 'multimodal-rag',
        pinecone_environment: str = 'gcp-starter',
    ):
        """Initialize the Multimodal RAG system.
        
        Args:
            project_id: Google Cloud project ID (used for GCS and Vertex AI)
            location: Google Cloud region (used for GCS and Vertex AI)
            model_name: Name of the model to use for generation
            token_limit: Maximum number of tokens for model responses
            use_gcs: Whether to use Google Cloud Storage (legacy, prefer vector_store_type)
            vector_store_type: Type of vector store to use ('local', 'gcs', or 'pinecone')
            pinecone_api_key: Pinecone API key (required if vector_store_type is 'pinecone')
            pinecone_index_name: Name of the Pinecone index
            pinecone_environment: Pinecone environment
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.token_limit = token_limit
        self.use_gcs = use_gcs
        self.vector_store_type = vector_store_type.lower()
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_index_name = pinecone_index_name
        self.pinecone_environment = pinecone_environment
        
        # Initialize components
        self.embedding_manager = None
        self.llm = None
        self.chain = None
        
        # Initialize document processor with our embedding manager
        self.document_processor = DocumentProcessor(embedding_manager=self.embedding_manager)
        
        # Reassign embedding manager after initialization to ensure it's the same instance
        self.embedding_manager = self.document_processor.embedding_manager
        
        # Initialize the RAG system
        try:
            self.initialize()
            print("Multimodal RAG system initialized successfully")
        except Exception as e:
            print(f"Error initializing Multimodal RAG system: {str(e)}")
            raise
    
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
        print("Initializing embedding manager...")
        # Initialize embedding manager with appropriate model
        try:
            self.embedding_manager = EmbeddingManager(
                model_name="text-embedding-004",  # Use the correct embedding model
                use_vision=True,  # Enable vision capabilities
                api_key=os.getenv('GOOGLE_API_KEY'),
                vector_store_type=self.vector_store_type,
                pinecone_api_key=self.pinecone_api_key,
                pinecone_index_name=self.pinecone_index_name,
                pinecone_environment=self.pinecone_environment
            )
            
            # Initialize vector store based on storage type
            if self.vector_store_type == 'pinecone':
                print(f"Initializing Pinecone vector store with index '{self.pinecone_index_name}'...")
                self.embedding_manager.initialize_vector_store(
                    vector_store_type='pinecone',
                    pinecone_api_key=self.pinecone_api_key or os.getenv('PINECONE_API_KEY'),
                    pinecone_index_name=self.pinecone_index_name,
                    pinecone_environment=self.pinecone_environment
                )
            elif self.use_gcs or self.vector_store_type == 'gcs':
                print("Initializing GCS vector store...")
                # Initialize vector store with existing index and endpoint for GCS
                self.embedding_manager.initialize_vector_store(
                    vector_store_type='gcs',
                    project_id=self.project_id,
                    location=self.location,
                    index_id=index_id,
                    endpoint_id=endpoint_id
                )
            else:
                print("Initializing local vector store...")
                # Initialize local vector store
                self.embedding_manager.initialize_vector_store(
                    vector_store_type='local',
                    persist_directory=str(Path(__file__).parent.parent / "data" / "vector_store")
                )
                
            print("Vector store initialized successfully")
            
        except Exception as e:
            print(f"Error initializing embedding manager: {e}")
            raise
        
        # Initialize the LLM
        try:
            print(f"Initializing LLM with model: {self.model_name}")
            self.llm = ChatVertexAI(
                model_name=self.model_name,
                project=self.project_id,
                location=self.location,
                max_output_tokens=self.token_limit,
                temperature=0.0,
            )
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            raise
        
        # Create the RAG chain
        print("Creating RAG chain...")
        self._create_rag_chain()
        print("RAG system initialization complete")
    
    def _create_rag_chain(self) -> None:
        """Create the RAG chain with prompt template and model."""
        # Define the prompt template
        prompt_template = """You are a helpful assistant that answers questions based on the provided context.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions based on the provided context."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        
        # Define the RAG chain
        self.chain = (
            {
                "context": self.retrieve_documents,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: x.get("chat_history", []),
            }
            | RunnableLambda(self._format_prompt)
            | self.llm
            | StrOutputParser()
        )
    
    def _format_prompt(self, data: Dict[str, Any]) -> List[Union[HumanMessage, str]]:
        """Format the prompt with context and question."""
        context = data["context"]
        question = data["question"]
        chat_history = data.get("chat_history", [])
        
        # Format context
        formatted_context = []
        if context["texts"]:
            formatted_context.append("Text and tables:")
            formatted_context.extend([f"- {text}" for text in context["texts"]])
        
        # Add image information if present
        if context["images"]:
            formatted_context.append("\nImages:")
            for i, img in enumerate(context["images"], 1):
                formatted_context.append(f"- Image {i} (see below)")
        
        # Combine all context
        full_context = "\n".join(formatted_context) if formatted_context else "No relevant context found."
        
        # Create messages
        messages = [
            ("system", "You are a helpful assistant that answers questions based on the provided context."),
            *chat_history,
        ]
        
        # Create the human message with text content
        human_message = [
            ("human", f"Context:\n{full_context}\n\nQuestion: {question}")
        ]
        
        # Add images if any
        if context["images"]:
            # Convert the human message to a list of message parts
            message_parts = [{"type": "text", "text": f"Context:\n{full_context}\n\nQuestion: {question}"}]
            
            # Add image parts
            for img_doc in context["images"]:
                # Handle both Document objects and direct base64 strings
                if hasattr(img_doc, 'metadata') and 'image_data' in img_doc.metadata:
                    img_data = img_doc.metadata['image_data']
                elif isinstance(img_doc, str):
                    img_data = img_doc
                else:
                    print("Skipping invalid image format in context")
                    continue
                
                # Clean up the image data if needed
                if img_data.startswith('data:image/'):
                    img_data = img_data.split(',', 1)[-1]
                
                # Add the image part
                try:
                    message_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
                    })
                except Exception as e:
                    print(f"Error adding image to message: {e}")
            
            # Only update the message if we have valid image parts
            if len(message_parts) > 1:  # More than just the text part
                human_message = [("human", message_parts)]
            else:
                # Fallback to text-only if no valid images were added
                human_message = [("human", f"Context:\n{full_context}\n\nQuestion: {question}")]
        
        # Add the human message to the messages list
        messages.extend(human_message)
        
        return messages
    
    def retrieve_documents(self, query: str) -> Dict[str, List]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: The query string
            
        Returns:
            Dictionary with 'texts' and 'images' keys
        """
        if self.embedding_manager is None:
            raise ValueError("Embedding manager not initialized. Call initialize() first.")
        
        # Get relevant documents
        docs = self.embedding_manager.get_relevant_documents(query, k=5)
        
        # Split into text and images
        return split_image_text_types(docs)
    
    def query(
        self,
        question: str,
        chat_history: Optional[List] = None,
        return_context: bool = False,
    ) -> Dict[str, Any]:
        """Query the RAG system with a question.
        
        Args:
            question: The question to ask
            chat_history: Optional list of previous messages in the conversation
            return_context: Whether to include the retrieved context in the response
            
        Returns:
            Dictionary with the answer and optionally the context
        """
        if self.chain is None:
            self.initialize()
        
        # Prepare input
        input_data = {
            "question": question,
            "chat_history": chat_history or [],
        }
        
        # Get the answer and context
        answer = self.chain.invoke(input_data)
        
        # Always retrieve context for citations, but only include in response if requested
        context = self.retrieve_documents(question)
        
        # Prepare citations from context
        citations = []
        
        # Process text documents
        for doc in context.get('texts', []):
            if hasattr(doc, 'metadata'):
                source = doc.metadata.get('source', 'Unknown source')
                page = doc.metadata.get('page', '')
                citation_text = f"Source: {Path(source).name}"
                if page:
                    citation_text += f", Page: {page}"
                
                # Add a snippet of the content
                content = doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                
                citations.append({
                    'type': 'text',
                    'source': source,
                    'page': page,
                    'content': content,
                    'display_text': citation_text
                })
        
        # Process images
        for doc in context.get('images', []):
            if hasattr(doc, 'metadata'):
                source = doc.metadata.get('source', 'Unknown source')
                citations.append({
                    'type': 'image',
                    'source': source,
                    'image_path': doc.metadata.get('image_path', ''),
                    'display_text': f"Image from: {Path(source).name}"
                })
        
        # Prepare response
        response = {
            "answer": answer,
            "citations": citations
        }
        
        # Include full context if requested
        if return_context:
            response["context"] = context
        
        return response
    
    def add_documents(
        self, 
        file_paths: Union[str, List[str]], 
        use_llama_parse: bool = False,
        **kwargs
    ) -> Dict[str, int]:
        """Add documents to the vector store.
        
        Args:
            file_paths: Path or list of paths to documents to add
            use_llama_parse: Whether to use LlamaParse for document processing
            **kwargs: Additional arguments to pass to the document processor and vector store
            
        Returns:
            Dictionary with counts of added documents by type
        """
        if not file_paths:
            return {"texts": 0, "tables": 0, "images": 0, "total": 0}
            
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        total_added = 0
        text_count = 0
        table_count = 0
        image_count = 0
        
        for file_path in file_paths:
            try:
                print(f"Processing document: {file_path}")
                # Process the document
                processed_docs = self.document_processor.process_file(
                    file_path, 
                    use_llama_parse=use_llama_parse,
                    **{k: v for k, v in kwargs.items() if k not in ['pinecone_metadata', 'pinecone_namespace']}
                )
                
                if not processed_docs:
                    print(f"No content extracted from {file_path}")
                    continue
                
                # Prepare metadata for Pinecone if using Pinecone
                pinecone_metadata = kwargs.get('pinecone_metadata', {})
                pinecone_namespace = kwargs.get('pinecone_namespace')
                
                # Add file-specific metadata
                file_metadata = {
                    'source': file_path,
                    'file_name': os.path.basename(file_path),
                    **pinecone_metadata
                }
                
                # Process documents by type (texts, tables, images)
                documents = []
                for doc_type in ['texts', 'tables', 'images']:
                    if doc_type in processed_docs and processed_docs[doc_type]:
                        for doc in processed_docs[doc_type]:
                            # Ensure we have a Document object
                            if isinstance(doc, str):
                                doc = Document(
                                    page_content=doc,
                                    metadata={
                                        'type': doc_type.rstrip('s'),  # 'texts' -> 'text', 'tables' -> 'table', etc.
                                        **file_metadata
                                    }
                                )
                            elif hasattr(doc, 'metadata'):
                                doc.metadata.update(file_metadata)
                                doc.metadata.setdefault('type', doc_type.rstrip('s'))
                            
                            documents.append(doc)
                            
                            # Update counts
                            if doc_type == 'tables':
                                table_count += 1
                            elif doc_type == 'images':
                                image_count += 1
                            else:
                                text_count += 1
                
                # Add documents to the vector store
                add_kwargs = {}
                if self.vector_store_type == 'pinecone' and pinecone_namespace:
                    add_kwargs['namespace'] = pinecone_namespace
                
                doc_ids = self.embedding_manager.add_documents(documents, **add_kwargs)
                added_count = len(doc_ids)
                total_added += added_count
                print(f"Added {added_count} documents from {file_path}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        return {
            "texts": text_count,
            "tables": table_count,
            "images": image_count,
            "total": total_added
        }
    
    def _generate_summaries(self, texts: List[str]) -> List[str]:
        """Generate summaries for a list of texts."""
        from .document_processor import generate_summaries
        return generate_summaries(texts, model_name=self.model_name)
    
    def _generate_image_summaries(self, image_docs: List[Union[Document, str]]) -> List[str]:
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
                print(traceback.format_exc())
                summaries.append("Image description not available")
                
        return summaries
