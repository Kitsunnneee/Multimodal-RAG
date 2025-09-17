"""
Example usage of the Multimodal RAG system.

This script demonstrates how to:
1. Initialize the RAG system
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

def main():
    # Initialize the RAG system
    print("Initializing Multimodal RAG system...")
    rag = MultimodalRAG(
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION"),
    )
    
    # Initialize with existing index and endpoint (if any)
    rag.initialize(
        index_id=os.getenv("VECTOR_SEARCH_INDEX_NAME"),
        endpoint_id=os.getenv("VECTOR_SEARCH_ENDPOINT_ID"),
    )
    
    # Path to your document
    doc_path = "data/your_document.pdf"  # Update this path
    
    if not os.path.exists(doc_path):
        print(f"Document not found at {doc_path}. Please update the path and try again.")
        return
    
    # Add document to the RAG system
    print(f"Processing document: {doc_path}")
    result = rag.add_documents(doc_path)
    print(f"Added {result['total']} elements ({result['texts']} texts, {result['tables']} tables, {result['images']} images)")
    
    # Example queries
    queries = [
        "What are the key points from this document?",
        "Can you summarize the tables in this document?",
        "What images are included in this document?",
    ]
    
    # Process each query
    for query in queries:
        print(f"\nQuery: {query}")
        response = rag.query(query, return_context=True)
        print(f"Answer: {response['answer']}\n")
        
        # Display context if available
        if 'context' in response:
            context = response['context']
            if context['images']:
                print(f"Found {len(context['images'])} relevant images.")
                for i, img in enumerate(context['images'][:2]):  # Show first 2 images
                    print(f"Displaying image {i+1}...")
                    try:
                        from IPython.display import display, Image as IPImage
                        display(IPImage(data=base64.b64decode(img)))
                    except ImportError:
                        print("IPython not available. Cannot display images.")
            
            if context['texts']:
                print(f"\nRelevant text snippets:")
                for i, text in enumerate(context['texts'][:3]):  # Show first 3 text snippets
                    print(f"{i+1}. {text[:200]}...")  # Truncate long text

if __name__ == "__main__":
    main()
