"""Main RAG system implementation for multimodal retrieval and generation."""
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
    ):
        """Initialize the Multimodal RAG system.
        
        Args:
            project_id: Google Cloud project ID (only used if use_gcs=True)
            location: Google Cloud region (only used if use_gcs=True)
            model_name: Name of the model to use for generation
            token_limit: Maximum number of tokens for model responses
            use_gcs: Whether to use Google Cloud Storage (False for local storage)
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.token_limit = token_limit
        self.use_gcs = use_gcs
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_manager = None
        self.llm = None
        self.chain = None
    
    def initialize(
        self,
        index_id: Optional[str] = None,
        endpoint_id: Optional[str] = None,
    ) -> None:
        """Initialize the RAG system components.
        
        Args:
            index_id: Optional custom index ID
            endpoint_id: Optional custom endpoint ID
        """
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(
            project_id=self.project_id,
            location=self.location,
            use_gcs=self.use_gcs,
        )
        
        # Initialize vector store with existing index and endpoint
        self.embedding_manager.initialize_vector_store(
            index_id=index_id if self.use_gcs else None,
            endpoint_id=endpoint_id if self.use_gcs else None,
        )
        

        self.llm = ChatVertexAI(
                model_name=self.model_name,
                project=self.project_id,
                location=self.location,
                max_output_tokens=self.token_limit,
                temperature=0.0,
            )

        
        # Create the RAG chain
        self._create_rag_chain()
    
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
            ("human", f"Context:\n{full_context}\n\nQuestion: {question}"),
        ]
        
        # Add images if any
        if context["images"]:
            messages[-1][1].extend([
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
                for img in context["images"]
            ])
        
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
        
        # Get the answer
        answer = self.chain.invoke(input_data)
        
        # Get the context if requested
        context = None
        if return_context:
            context = self.retrieve_documents(question)
        
        # Prepare response
        response = {"answer": answer}
        if context is not None:
            response["context"] = context
        
        return response
    
    def add_documents(
        self,
        file_path: str,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
    ) -> Dict[str, List[str]]:
        """Add documents to the RAG system.
        
        Args:
            file_path: Path to the document file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Dictionary with counts of added documents by type
        """
        if self.embedding_manager is None:
            self.initialize()
        
        # Process the document
        elements = self.document_processor.process_file(file_path)
        
        # Generate summaries for text and tables
        text_summaries = []
        if elements["texts"]:
            text_summaries = self._generate_summaries([doc.page_content for doc in elements["texts"]])
        
        table_summaries = []
        if elements["tables"]:
            table_summaries = self._generate_summaries([doc.page_content for doc in elements["tables"]])
        
        # Generate summaries for images
        image_summaries = []
        if elements["images"]:
            image_summaries = self._generate_image_summaries(
                [doc.page_content for doc in elements["images"]]
            )
        
        # Combine all documents and summaries
        all_docs = elements["texts"] + elements["tables"] + elements["images"]
        all_summaries = text_summaries + table_summaries + image_summaries
        
        # Add to vector store
        doc_ids = self.embedding_manager.add_documents(
            documents=all_docs,
            summaries=all_summaries,
        )
        
        # Return counts
        return {
            "texts": len(elements["texts"]),
            "tables": len(elements["tables"]),
            "images": len(elements["images"]),
            "total": len(doc_ids),
        }
    
    def _generate_summaries(self, texts: List[str]) -> List[str]:
        """Generate summaries for a list of texts."""
        from .document_processor import generate_summaries
        return generate_summaries(texts, model_name=self.model_name)
    
    def _generate_image_summaries(self, b64_images: List[str]) -> List[str]:
        """Generate summaries for a list of base64-encoded images."""
        if not b64_images:
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
        for img in b64_images:
            try:
                response = llm.invoke([
                    HumanMessage(
                        content=[
                            {"type": "text", "text": "Describe this image in detail for retrieval."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
                        ]
                    )
                ])
                summaries.append(response.content)
            except Exception as e:
                print(f"Error generating summary for image: {e}")
                summaries.append("Image content")
        
        return summaries
