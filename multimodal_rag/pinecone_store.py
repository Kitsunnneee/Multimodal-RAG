"""Pinecone vector store implementation for the Multimodal RAG system."""
from typing import List, Optional, Dict, Any, Union, Iterable
import os
import numpy as np
import logging
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

class PineconeStore(VectorStore):
    """Pinecone vector store implementation."""

    def __init__(
        self,
        api_key: str,
        index_name: str = "multimodal-rag",
        environment: str = "gcp-starter",
        embedding_dimension: int = 768,
        metric: str = "cosine",
        **kwargs
    ):
        """Initialize with Pinecone client."""
        super().__init__()
        self.index_name = index_name
        self.environment = environment
        self.embedding_dimension = embedding_dimension
        self.metric = metric
        
        logger.info("\n" + "="*50)
        logger.info("Initializing Pinecone Store")
        logger.info("="*50)
        logger.info(f"API Key: {'*' * 8}{api_key[-4:] if api_key else 'Not set'}")
        logger.info(f"Environment: {environment}")
        logger.info(f"Index Name: {index_name}")
        logger.info(f"Embedding Dimension: {embedding_dimension}")
        logger.info("="*50)
        
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=api_key)
            logger.info("✓ Pinecone client initialized successfully")
            
            # List existing indexes
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            logger.info(f"Existing indexes: {existing_indexes}")
            
            # Create index if it doesn't exist
            if index_name not in existing_indexes:
                logger.info(f"Creating new index: {index_name}")
                # Using AWS us-east-1 (N. Virginia) which is supported in free tier
                serverless_spec = ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # N. Virginia region (supported in free tier)
                )
                
                self.pc.create_index(
                    name=index_name,
                    dimension=embedding_dimension,
                    metric=metric,
                    spec=serverless_spec
                )
                logger.info(f"✓ Index '{index_name}' created successfully")
            else:
                logger.info(f"Using existing index: {index_name}")
                
            # Connect to the index
            self.index = self.pc.Index(index_name)
            logger.info("✓ Pinecone index is ready")
            logger.info("="*50 + "\n")
            
        except Exception as e:
            logger.error("\n" + "!"*50)
            logger.error("Error initializing Pinecone:")
            logger.error(str(e), exc_info=True)
            logger.error("!"*50 + "\n")
            raise

    def add_documents(
        self,
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        **kwargs
    ) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            embedding: Embedding model to use for generating document embeddings
            **kwargs: Additional arguments to pass to the embedding model
            
        Returns:
            List of document IDs that were added
        """
        if not documents:
            return []
            
        # Generate embeddings if not provided
        texts = []
        valid_docs = []
        
        # Filter out invalid documents and collect texts
        for doc in documents:
            if not doc.page_content:
                logger.warning("Skipping document with empty page_content")
                continue
                
            # Ensure metadata is a dict
            if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                doc.metadata = {}
                
            # Add source to metadata if not present
            if 'source' not in doc.metadata and hasattr(doc, 'metadata') and 'source' in doc.metadata:
                doc.metadata['source'] = doc.metadata['source']
            
            texts.append(doc.page_content)
            valid_docs.append(doc)
        
        if not valid_docs:
            return []
        
        # Generate embeddings
        try:
            if embedding is not None:
                # Try embed_documents first, fall back to embed_query if needed
                if hasattr(embedding, 'embed_documents'):
                    embeddings = embedding.embed_documents(texts)
                elif hasattr(embedding, 'embed_query'):
                    # Fallback to using embed_query for each document
                    embeddings = [embedding.embed_query(text) for text in texts]
                else:
                    raise ValueError(
                        "Embedding model must have either embed_documents or embed_query method"
                    )
            else:
                # Try to get embeddings from document metadata
                embeddings = [doc.metadata.get('embedding') for doc in valid_docs]
                if any(emb is None for emb in embeddings):
                    raise ValueError(
                        "No embedding provided and documents don't have 'embedding' in metadata"
                    )
            
            # Convert numpy arrays to lists if needed, handling float values
            processed_embeddings = []
            for emb in embeddings:
                if emb is None:
                    raise ValueError("Received None value in embeddings")
                if isinstance(emb, (int, float)):
                    # If it's a single float, wrap it in a list
                    processed_embeddings.append([float(emb)])
                elif hasattr(emb, 'tolist'):
                    # Convert numpy arrays to lists
                    processed_embeddings.append(emb.tolist())
                elif isinstance(emb, (list, tuple, np.ndarray)):
                    # Convert to list if it's a sequence
                    processed_embeddings.append(list(emb))
                else:
                    raise ValueError(f"Unexpected embedding type: {type(emb)}")
            
            embeddings = processed_embeddings
            
            # Prepare records for upsert
            records = []
            for i, (doc, emb) in enumerate(zip(valid_docs, embeddings)):
                if not isinstance(emb, (list, np.ndarray)):
                    logger.warning(f"Skipping document {i} due to invalid embedding type: {type(emb)}")
                    continue
                    
                # Generate a unique ID if not provided
                record_id = doc.metadata.get('id', f"doc_{i}_{hash(doc.page_content[:100])}")
                
                # Prepare metadata
                metadata = {
                    'text': doc.page_content,
                    **{k: v for k, v in doc.metadata.items() 
                       if k != 'embedding' and v is not None}
                }
                
                # Clean up metadata values
                for k, v in list(metadata.items()):
                    if isinstance(v, (list, dict, set)):
                        # Convert complex types to strings
                        metadata[k] = str(v)
                
                records.append((record_id, emb, metadata))
            
            if not records:
                logger.warning("No valid records to upsert")
                return []
            
            # Upsert in batches of 100 (Pinecone's limit)
            batch_size = 100
            doc_ids = []
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                try:
                    # Convert batch to list of dicts for upsert
                    vectors = [
                        {"id": idx, "values": emb, "metadata": meta}
                        for idx, emb, meta in batch
                    ]
                    
                    # Upsert the batch
                    result = self.index.upsert(vectors=vectors)
                    doc_ids.extend([vec[0] for vec in batch])
                    logger.info(f"Upserted batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1}")
                    
                except Exception as e:
                    logger.error(f"Error upserting batch {i//batch_size + 1}: {str(e)}")
                    # Continue with next batch instead of failing completely
                    continue
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error in add_documents: {str(e)}", exc_info=True)
            # Return empty list instead of raising to prevent breaking the application
            return []
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None,
        embedding: Optional[Embeddings] = None,
        **kwargs
    ) -> List[Document]:
        """Return documents most similar to the query.
        
        Args:
            query: Query text
            k: Number of documents to return
            filter: Optional filter to apply to the search
            embedding: Embedding model to use for generating query embedding
            **kwargs: Additional arguments to pass to the embedding model
            
        Returns:
            List of documents most similar to the query
        """
        if embedding is None:
            raise ValueError("Embedding model must be provided for similarity search")
            
        try:
            # Try to generate query embedding using embed_query
            if hasattr(embedding, 'embed_query'):
                query_embedding = embedding.embed_query(query)
            # Fallback to using embed_documents with a single query
            elif hasattr(embedding, 'embed_documents'):
                query_embedding = embedding.embed_documents([query])[0]
            else:
                raise ValueError("Embedding model must have either embed_query or embed_documents method")
            
            # Ensure query_embedding is a list of floats
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            elif not isinstance(query_embedding, list):
                query_embedding = list(query_embedding)
                
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                filter=filter,
                include_metadata=True,
                **kwargs
            )
            
            # Convert to Document objects
            documents = []
            if hasattr(results, 'matches'):
                for match in results.matches:
                    metadata = getattr(match, 'metadata', {}) or {}
                    # Handle different ways the text might be stored in metadata
                    text = metadata.get('text', '')
                    if not text and hasattr(match, 'metadata') and hasattr(match.metadata, 'get'):
                        text = match.metadata.get('page_content', '')
                    
                    # Create document with proper metadata
                    doc_metadata = {
                        'id': getattr(match, 'id', str(hash(text))),
                        'score': getattr(match, 'score', 0.0),
                        **metadata
                    }
                    
                    # Remove any None values from metadata
                    doc_metadata = {k: v for k, v in doc_metadata.items() if v is not None}
                    
                    doc = Document(
                        page_content=text,
                        metadata=doc_metadata
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in similarity_search: {str(e)}", exc_info=True)
            # Return empty list instead of raising to prevent breaking the application
            return []
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        api_key: Optional[str] = None,
        **kwargs
    ) -> 'PineconeStore':
        """Create a PineconeStore from a list of documents.
        
        Args:
            documents: List of Document objects
            embedding: Embedding model to use
            api_key: Optional Pinecone API key. If not provided, will look for PINECONE_API_KEY in environment.
            **kwargs: Additional arguments to pass to PineconeStore initialization
            
        Returns:
            PineconeStore instance with the documents added
        """
        # Get API key from kwargs or environment
        api_key = api_key or os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("Pinecone API key must be provided or set in environment as PINECONE_API_KEY")
        
        # Extract index-specific parameters from kwargs
        index_name = kwargs.pop('index_name', 'multimodal-rag')
        environment = kwargs.pop('environment', 'gcp-starter')
        embedding_dimension = kwargs.pop('embedding_dimension', 768)
        
        # Initialize the store with explicit parameters
        store = cls(
            api_key=api_key,
            index_name=index_name,
            environment=environment,
            embedding_dimension=embedding_dimension,
            **kwargs
        )
        
        # Add documents with error handling
        if documents:
            try:
                added_ids = store.add_documents(documents, embedding=embedding)
                logger.info(f"Successfully added {len(added_ids)} documents to Pinecone index '{index_name}'")
                
                # Verify the documents were added
                if hasattr(store, 'index') and hasattr(store.index, 'describe_index_stats'):
                    try:
                        stats = store.index.describe_index_stats()
                        logger.info(f"Index stats: {stats}")
                    except Exception as e:
                        logger.warning(f"Could not get index stats: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Error adding documents to Pinecone: {str(e)}")
                # Re-raise the exception to ensure the user knows something went wrong
                raise
        
        return store
            
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs
    ) -> 'PineconeStore':
        """Create a PineconeStore from a list of texts.
        
        Args:
            texts: List of texts to add to the vector store
            embedding: Embedding model to use
            metadatas: Optional list of metadata dicts (one per text)
            **kwargs: Additional arguments to pass to PineconeStore initialization
            
        Returns:
            PineconeStore instance with the texts added
            
        Raises:
            ValueError: If no valid texts are provided or if there's an error adding documents
        """
        from langchain_core.documents import Document
        
        if not texts or not all(isinstance(t, str) for t in texts):
            raise ValueError("texts must be a non-empty list of strings")
            
        # Prepare documents with proper metadata
        documents = []
        for i, text in enumerate(texts):
            if not text.strip():
                logger.warning(f"Skipping empty text at index {i}")
                continue
                
            # Get metadata for this document if available
            metadata = {}
            if metadatas and i < len(metadatas) and isinstance(metadatas[i], dict):
                metadata = metadatas[i].copy()  # Don't modify the original
                
            # Add source information if not present
            if 'source' not in metadata:
                metadata['source'] = f"text_{i}"
                
            # Create document with text and metadata
            documents.append(Document(
                page_content=text,
                metadata=metadata
            ))
        
        if not documents:
            raise ValueError("No valid documents to add to Pinecone store")
            
        logger.info(f"Converted {len(documents)} texts to Document objects")
        
        # Extract Pinecone-specific parameters from kwargs
        api_key = kwargs.pop('api_key', None)
        
        # Use from_documents to handle the rest
        return cls.from_documents(
            documents=documents,
            embedding=embedding,
            api_key=api_key,
            **kwargs
        )
