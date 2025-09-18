"""Test script to add a document to the RAG system with enhanced error handling and embedding tests."""
import sys
import os
from pathlib import Path
from pprint import pprint
import shutil
from typing import Dict, Any, List, Optional

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_add_document(pdf_path: str, cleanup: bool = True):
    """Test adding a document to the RAG system with embedding tests.
    
    Args:
        pdf_path: Path to the PDF file to test
        cleanup: Whether to clean up test artifacts after completion
    """
    print(f"\n{'='*50}\nTesting document addition: {pdf_path}\n{'='*50}")
    
    # Convert to Path object
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        return
    
    try:
        # Import here to catch any import errors
        from multimodal_rag.rag_system import MultimodalRAG
        from multimodal_rag.config import DATA_DIR, VECTOR_STORE_DIR
        from multimodal_rag.embeddings import LocalEmbeddings
        
        # Create test output directory
        test_output_dir = DATA_DIR / "test_output"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up previous test artifacts if they exist
        if cleanup and VECTOR_STORE_DIR.exists():
            print("\nCleaning up previous test artifacts...")
            shutil.rmtree(VECTOR_STORE_DIR, ignore_errors=True)
        
        # Initialize RAG system with local storage
        print("\n[1/4] Initializing RAG system...")
        rag = MultimodalRAG(use_gcs=False)
        rag.initialize()
        
        # Test 1: Add the document
        print(f"\n[2/4] Adding document: {pdf_path}")
        result = rag.add_documents(
            file_path=str(pdf_path),
            chunk_size=2000,
            chunk_overlap=200
        )
        
        print("\nDocument added successfully!")
        print("\nResults:")
        pprint(result)
        
        # Test 2: Test text embeddings
        print("\n[3/4] Testing text embeddings...")
        test_text_embeddings(rag)
        
        # Test 3: Test retrieval
        print("\n[4/4] Testing retrieval...")
        test_retrieval(rag)
        
        return result
        
    except Exception as e:
        print(f"\nError during testing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def test_text_embeddings(rag) -> None:
    """Test text embedding functionality."""
    try:
        # Get the embedding model
        if not hasattr(rag, 'embedding_manager') or not hasattr(rag.embedding_manager, 'embeddings'):
            print("Warning: No embedding manager found, skipping text embedding test")
            return
            
        embeddings = rag.embedding_manager.embeddings
        
        # Test text embedding
        test_text = "This is a test sentence for embedding generation."
        print(f"\nTesting text embedding for: '{test_text[:50]}...'")
        
        # Test single document
        single_embedding = embeddings.embed_documents([test_text])
        print(f"Single embedding dimension: {len(single_embedding[0])}")
        
        # Test multiple documents
        multi_texts = [test_text, "Another test sentence"]
        multi_embeddings = embeddings.embed_documents(multi_texts)
        print(f"Multiple embeddings: {len(multi_embeddings)} embeddings generated")
        
        # Verify dimensions
        assert len(single_embedding) == 1, "Should return one embedding for single text"
        assert len(multi_embeddings) == 2, "Should return two embeddings for two texts"
        assert len(single_embedding[0]) > 100, "Embedding should have reasonable dimension"
        
        print("✅ Text embedding test passed")
        
    except Exception as e:
        print(f"⚠️ Text embedding test failed: {e}")


def test_retrieval(rag) -> None:
    """Test document retrieval functionality."""
    try:
        # Test with a general query
        query = "What is this document about?"
        print(f"\nTesting retrieval with query: '{query}'")
        
        response = rag.query(query, return_context=True)
        
        # Print response
        print("\nResponse:")
        print(response.get("answer", "No answer found"))
        
        # Check context
        context = response.get("context", {})
        if context:
            print("\nContext used:")
            for doc_type, docs in context.items():
                if not isinstance(docs, (list, tuple)):
                    print(f"\n{doc_type.capitalize()}: (Not in expected format)")
                    continue
                    
                print(f"\n{doc_type.capitalize()} (showing {min(2, len(docs))} of {len(docs)}):")
                for i, doc in enumerate(docs[:2]):  # Show first 2 of each type
                    try:
                        # Handle both Document objects and strings
                        if hasattr(doc, 'page_content'):
                            content = doc.page_content
                        else:
                            content = str(doc)
                            
                        if len(content) > 200:
                            content = content[:200] + "..."
                        
                        print(f"{i+1}. {content}")
                    except Exception as e:
                        print(f"  Error processing document {i+1}: {e}")
        
        # Test with a more specific query if we have context
        if context and any(docs for docs in context.values() if isinstance(docs, (list, tuple)) and len(docs) > 0):
            print("\nTesting with a more specific query...")
            specific_query = "What are the main topics discussed?"
            print(f"Query: '{specific_query}'")
            
            specific_response = rag.query(specific_query, return_context=False)
            print("\nResponse:")
            print(specific_response.get("answer", "No answer found"))
        
        print("✅ Retrieval test completed")
        
    except Exception as e:
        print(f"⚠️ Retrieval test failed: {e}")


def main():
    """Main function to run the test."""
    if len(sys.argv) < 2:
        print("Usage: python test_add_document.py <path_to_pdf> [--no-cleanup]")
        print("\nOptions:")
        print("  --no-cleanup  Keep test artifacts after completion")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    cleanup = "--no-cleanup" not in sys.argv
    
    test_add_document(pdf_path, cleanup=cleanup)


if __name__ == "__main__":
    main()
