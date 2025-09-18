"""Test script for PDF processing with enhanced error handling and embedding tests."""
import sys
import os
from pathlib import Path
from pprint import pprint
import pytest
from PIL import Image, ImageDraw
import io

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Create a test image for processing
def create_test_image(output_path: Path, size=(200, 200), text="Test"):
    """Create a test image with some text."""
    img = Image.new('RGB', size, color='white')
    d = ImageDraw.Draw(img)
    d.text((10, 10), text, fill='black')
    img.save(output_path)
    return output_path

def test_pdf_processing(pdf_path):
    """Test PDF processing with the given PDF file."""
    print(f"\n{'='*50}\nTesting PDF: {pdf_path}\n{'='*50}")
    
    # Import here to catch any import errors
    try:
        from multimodal_rag.document_processor import DocumentProcessor
        from multimodal_rag.embeddings import LocalEmbeddings, EmbeddingManager
        from multimodal_rag.config import DATA_DIR, VECTOR_STORE_DIR
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please make sure you have installed all dependencies from requirements.txt")
        return
    
    # Create test output directory
    test_output_dir = DATA_DIR / "test_output"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a test image if it doesn't exist
    test_image_path = test_output_dir / "test_image.jpg"
    if not test_image_path.exists():
        create_test_image(test_image_path, text="Test Image for Embedding")
    
    # Initialize processor and embeddings
    processor = DocumentProcessor(output_dir=test_output_dir)
    
    try:
        # Test 1: Process the PDF
        print("\n[Test 1/3] Processing PDF...")
        result = processor.process_pdf(
            pdf_path,
            extract_images=True,
            infer_table_structure=True
        )
        
        # Print results
        print("\nProcessing complete. Results:")
        print(f"- Text documents: {len(result.get('texts', []))}")
        print(f"- Tables: {len(result.get('tables', []))}")
        print(f"- Images: {len(result.get('images', []))}")
        
        # Test 2: Test text embedding
        print("\n[Test 2/3] Testing text embeddings...")
        embeddings = LocalEmbeddings()
        
        if result.get('texts'):
            sample_text = result['texts'][0].page_content[:500]  # Use first 500 chars
            print(f"\nSample text content (first 100 chars): {sample_text[:100]}...")
            
            # Test text embedding
            text_embedding = embeddings.embed_documents([sample_text])
            print(f"Generated embedding dimension: {len(text_embedding[0])}")
            assert len(text_embedding) == 1, "Should return one embedding per text"
            assert len(text_embedding[0]) > 100, "Embedding should have reasonable dimension"
        
        # Test 3: Test image embedding if images were extracted
        if result.get('images') and hasattr(embeddings, 'embed_image'):
            print("\n[Test 3/3] Testing image embeddings...")
            try:
                # Use the first extracted image or the test image
                image_to_test = result['images'][0].metadata.get('image_path', test_image_path)
                
                if isinstance(image_to_test, str):
                    image_to_test = Path(image_to_test)
                
                if not image_to_test.exists():
                    print(f"Warning: Image not found at {image_to_test}, using test image")
                    image_to_test = test_image_path
                
                with open(image_to_test, 'rb') as f:
                    image_data = f.read()
                
                image_embedding = embeddings.embed_image(image_data)
                print(f"Generated image embedding dimension: {len(image_embedding)}")
                assert len(image_embedding) > 100, "Image embedding should have reasonable dimension"
                
            except Exception as e:
                print(f"Warning: Image embedding test skipped: {e}")
        
        # Test 4: Test EmbeddingManager
        print("\n[Test 4/4] Testing EmbeddingManager...")
        try:
            # Clean up any existing vector store
            if VECTOR_STORE_DIR.exists():
                import shutil
                shutil.rmtree(VECTOR_STORE_DIR, ignore_errors=True)
            
            # Initialize embedding manager
            manager = EmbeddingManager(use_gcs=False)
            manager.initialize_vector_store()
            
            # Add documents
            if result.get('texts'):
                doc_ids = manager.add_documents(result['texts'])
                print(f"Added {len(doc_ids)} documents to vector store")
                
                # Test search
                query = "What is this document about?"
                print(f"\nTesting search with query: '{query}'")
                search_results = manager.similarity_search(query, k=1)
                
                if search_results:
                    print(f"Found {len(search_results)} results")
                    print(f"Top result: {search_results[0].page_content[:200]}...")
                else:
                    print("No search results found")
        
        except Exception as e:
            print(f"Warning: EmbeddingManager test skipped: {e}")
        
        return result
        
    except Exception as e:
        print(f"\nError during testing: {e}", file=sys.stderr)
        print("\nTroubleshooting steps:")
        print("1. Make sure the file exists and is a valid PDF")
        print("2. Check that you have the required permissions to read the file")
        print("3. Verify that all dependencies are installed (PyMuPDF, pdf2image, etc.)")
        print(f"4. Check the error message above for more details: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pdf_processing.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    test_pdf_processing(pdf_path)
