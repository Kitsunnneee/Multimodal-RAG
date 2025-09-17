"""Embedding generation and management for the Multimodal RAG system."""
from typing import Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai.vectorstores import VectorSearchVectorStore
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

from .config import (
    PROJECT_ID,
    LOCATION,
    GCS_BUCKET,
    EMBEDDING_MODEL_NAME,
    VECTOR_SEARCH_INDEX_NAME,
    DEPLOYED_INDEX_ID,
)
from .utils import split_image_text_types


class EmbeddingManager:
    """Manages document embeddings and vector search."""
    
    def __init__(
        self,
        project_id: str = PROJECT_ID,
        location: str = LOCATION,
        gcs_bucket: str = GCS_BUCKET,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        index_name: str = VECTOR_SEARCH_INDEX_NAME,
        endpoint_id: str = DEPLOYED_INDEX_ID,
    ):
        """Initialize the embedding manager.
        
        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
            gcs_bucket: GCS bucket for vector search
            embedding_model_name: Name of the embedding model
            index_name: Name of the vector search index
            endpoint_id: ID of the deployed index endpoint
        """
        self.project_id = project_id
        self.location = location
        self.gcs_bucket = gcs_bucket
        self.embedding_model_name = embedding_model_name
        self.index_name = index_name
        self.endpoint_id = endpoint_id
        
        # Initialize embedding model
        self.embeddings = VertexAIEmbeddings(
            model_name=embedding_model_name,
            project=project_id,
            location=location,
        )
        
        # Initialize vector store and retriever
        self.vector_store = None
        self.retriever = None
    
    def initialize_vector_store(self, index_id: Optional[str] = None, endpoint_id: Optional[str] = None):
        """Initialize the vector store with an existing index and endpoint.
        
        Args:
            index_id: ID of the existing index (defaults to class attribute)
            endpoint_id: ID of the existing endpoint (defaults to class attribute)
        """
        index_id = index_id or self.index_name
        endpoint_id = endpoint_id or self.endpoint_id
        
        self.vector_store = VectorSearchVectorStore.from_components(
            project_id=self.project_id,
            region=self.location,
            gcs_bucket_name=self.gcs_bucket,
            index_id=index_id,
            endpoint_id=endpoint_id,
            embedding=self.embeddings,
            stream_update=True,
        )
    
    def create_retriever(self, id_key: str = "doc_id") -> MultiVectorRetriever:
        """Create a multi-vector retriever.
        
        Args:
            id_key: Key to use for document IDs
            
        Returns:
            Configured MultiVectorRetriever instance
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call initialize_vector_store() first.")
        
        # Create document store
        docstore = InMemoryStore()
        
        # Create retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vector_store,
            docstore=docstore,
            id_key=id_key,
        )
        
        return self.retriever
    
    def add_documents(
        self,
        documents: List[Document],
        summaries: Optional[List[Union[str, Document]]] = None,
        id_key: str = "doc_id",
    ) -> List[str]:
        """Add documents to the vector store and document store.
        
        Args:
            documents: List of Document objects to add
            summaries: Optional list of summaries or Documents with summaries
            id_key: Key to use for document IDs
            
        Returns:
            List of document IDs
        """
        if self.retriever is None:
            self.create_retriever(id_key=id_key)
        
        # Generate IDs for documents
        doc_ids = [str(i) for i in range(len(documents))]
        
        # Add documents to docstore
        self.retriever.docstore.mset(zip(doc_ids, documents))
        
        # Prepare summary documents
        if summaries is None:
            # Use document contents as summaries if none provided
            summary_docs = [
                Document(
                    page_content=doc.page_content,
                    metadata={id_key: doc_id, "source": doc.metadata.get("source", "")}
                )
                for doc_id, doc in zip(doc_ids, documents)
            ]
        else:
            # Use provided summaries
            summary_docs = []
            for i, (doc_id, summary) in enumerate(zip(doc_ids, summaries)):
                if isinstance(summary, Document):
                    summary_doc = summary
                    summary_doc.metadata[id_key] = doc_id
                else:
                    summary_doc = Document(
                        page_content=summary,
                        metadata={
                            id_key: doc_id,
                            "source": documents[i].metadata.get("source", ""),
                        },
                    )
                summary_docs.append(summary_doc)
        
        # Add summary documents to vector store
        self.vector_store.add_documents(summary_docs)
        
        return doc_ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict] = None,
    ) -> List[Document]:
        """Perform a similarity search.
        
        Args:
            query: Query string
            k: Number of results to return
            filter_dict: Optional filter dictionary
            
        Returns:
            List of matching Document objects
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call initialize_vector_store() first.")
        
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict,
        )
    
    def get_relevant_documents(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict] = None,
    ) -> List[Document]:
        """Get relevant documents for a query.
        
        Args:
            query: Query string
            k: Number of results to return
            filter_dict: Optional filter dictionary
            
        Returns:
            List of relevant Document objects
        """
        if self.retriever is None:
            self.create_retriever()
        
        # Get relevant document IDs
        summary_docs = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict,
        )
        
        # Get full documents from docstore
        doc_ids = [doc.metadata[self.retriever.id_key] for doc in summary_docs]
        documents = [self.retriever.docstore.mget([doc_id])[0] for doc_id in doc_ids]
        
        return documents
