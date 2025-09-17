"""
Example usage of the Multimodal RAG system.

This script demonstrates how to:
1. Initialize the RAG system with either GCS or local storage
2. Add documents
3. Query the system
4. Display results
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the MultimodalRAG class
from multimodal_rag.rag_system import MultimodalRAG

def process_document(rag, doc_path):
    """Process a document with the given RAG system."""
    if not os.path.exists(doc_path):
        print(f"Document not found at {doc_path}. Please check the path and try again.")
        return False
    
    print(f"\nProcessing document: {doc_path}")
    try:
        result = rag.add_documents(doc_path)
        print(f"Added {result['total']} elements ({result['texts']} texts, {result['tables']} tables, {result['images']} images)")
        return True
    except Exception as e:
        print(f"Error processing document: {e}")
        return False

def query_rag(rag, queries):
    """Run queries against the RAG system."""
    for query in queries:
        print(f"\nQuery: {query}")
        try:
            response = rag.query(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error running query: {e}")

def main():
    # Choose storage type (GCS or local)
    use_gcs = os.getenv("USE_GCS", "false").lower() == "true"
    
    if use_gcs:
        print("Initializing Multimodal RAG system with Google Cloud Storage...")
        # Initialize with GCS
        rag = MultimodalRAG(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION"),
            use_gcs=True
        )
        rag.initialize(
            index_id=os.getenv("VECTOR_SEARCH_INDEX_NAME"),
            endpoint_id=os.getenv("VECTOR_SEARCH_ENDPOINT_ID"),
        )
    else:
        print("Initializing Multimodal RAG system with local storage...")
        # Initialize with local storage
        rag = MultimodalRAG(use_gcs=False)
        rag.initialize()
    
    # Example document path (update this to your document)
    doc_path = "data/your_document.pdf"
    
    # Process the document
    if process_document(rag, doc_path):
        # Example queries
        queries = [
            "What are the key points from this document?",
            "Can you summarize the tables in this document?",
            "What images are included in this document?",
        ]
        
        # Run the queries
        query_rag(rag, queries)
    else:
        print("Skipping queries due to document processing error.")

if __name__ == "__main__":
    main()
