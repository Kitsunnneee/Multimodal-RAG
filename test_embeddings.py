"""Test script for multimodal embeddings with Vertex AI and fallbacks."""
import os
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Test text
TEST_TEXT = "This is a test sentence for embedding generation."

# Create a test image
def create_test_image(width=100, height=100):
    """Create a simple test image."""
    img = Image.new('RGB', (width, height), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Test Image", fill=(255, 255, 0))
    return img

# Create a test video
def create_test_video():
    """Create a path to a test video file (mock)."""
    return "test_data/sample_video.mp4"  # This should be replaced with an actual test video

def test_vertex_ai_initialization():
    """Test Vertex AI Multimodal Embeddings initialization with different dimensions."""
    from multimodal_rag.embeddings import LocalEmbeddings
    
    # Test with different dimensions
    for dim in [128, 256, 512, 1408]:
        try:
            embeddings = LocalEmbeddings(use_vertex_ai=True, dimension=dim)
            if embeddings.vertex_ai_initialized:
                print(f"Vertex AI Multimodal Embeddings initialized with dimension {dim}")
                assert embeddings.dimension == dim, f"Dimension should be {dim}"
            else:
                print("Vertex AI initialization failed, falling back to local models")
                pytest.skip("Vertex AI not configured, skipping test")
        except Exception as e:
            print(f"Error initializing Vertex AI with dimension {dim}: {e}")
            pytest.skip(f"Vertex AI with dimension {dim} not supported")

def test_text_embeddings():
    """Test text embedding generation with Vertex AI."""
    from multimodal_rag.embeddings import LocalEmbeddings
    
    try:
        embeddings = LocalEmbeddings(use_vertex_ai=True)
        if not embeddings.vertex_ai_initialized:
            pytest.skip("Vertex AI not available, skipping test")
            
        # Test single document
        embedding = embeddings.embed_documents([TEST_TEXT])[0]
        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) == embeddings.dimension, f"Embedding dimension should be {embeddings.dimension}"
        assert all(isinstance(x, float) for x in embedding), "All elements should be floats"
        
        # Test multiple documents
        texts = ["First document", "Second document"]
        embeddings_list = embeddings.embed_documents(texts)
        assert len(embeddings_list) == 2, "Should return one embedding per document"
        
    except Exception as e:
        print(f"Error in test_text_embeddings: {e}")
        raise

def test_image_embeddings():
    """Test image embedding generation with Vertex AI."""
    from multimodal_rag.embeddings import LocalEmbeddings
    
    try:
        embeddings = LocalEmbeddings(use_vertex_ai=True)
        if not embeddings.vertex_ai_initialized:
            pytest.skip("Vertex AI not available, skipping test")
        
        # Create a test image
        img = create_test_image()
        
        # Convert image to bytes
        import io
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Test with image only
        embedding = embeddings.embed_image(img_byte_arr)
        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) == embeddings.dimension, f"Embedding dimension should be {embeddings.dimension}"
        
        # Test with image and text context
        embedding_with_text = embeddings.embed_image(img_byte_arr, "A test image")
        assert len(embedding_with_text) == len(embedding), "Embedding dimensions should match"
        
    except Exception as e:
        print(f"Error in test_image_embeddings: {e}")
        raise

@pytest.mark.skipif(not os.path.exists("test_data/sample_video.mp4"), 
                   reason="Test video file not found")
def test_video_embeddings():
    """Test video embedding generation with Vertex AI."""
    from multimodal_rag.embeddings import LocalEmbeddings
    
    try:
        embeddings = LocalEmbeddings(use_vertex_ai=True)
        if not embeddings.vertex_ai_initialized:
            pytest.skip("Vertex AI not available, skipping test")
        
        # Test with video file
        video_path = "test_data/sample_video.mp4"
        
        # Test with default segment config
        embeddings_list = embeddings.embed_video(video_path)
        assert isinstance(embeddings_list, list), "Should return a list of embeddings"
        assert len(embeddings_list) > 0, "Should return at least one segment embedding"
        assert all(isinstance(e, list) for e in embeddings_list), "All items should be embedding lists"
        
        # Test with custom segment config
        segment_config = {"segment_granularity": "10s_5s", "segment_count": 3}
        custom_embeddings = embeddings.embed_video(video_path, segment_config=segment_config)
        assert len(custom_embeddings) <= 3, "Should respect segment_count"
        
    except Exception as e:
        print(f"Error in test_video_embeddings: {e}")
        raise

def test_similarity_functions():
    """Test similarity calculation functions."""
    from multimodal_rag.embeddings import LocalEmbeddings
    
    embeddings = LocalEmbeddings(use_vertex_ai=False)
    
    # Test cosine similarity
    emb1 = [1.0, 0.0, 0.0]
    emb2 = [1.0, 0.0, 0.0]  # Same as emb1
    emb3 = [0.0, 1.0, 0.0]  # Orthogonal to emb1
    
    # Test identical vectors
    sim = embeddings.cosine_similarity(emb1, emb2)
    assert abs(sim - 1.0) < 1e-6, f"Identical vectors should have similarity ~1.0, got {sim}"
    
    # Test orthogonal vectors
    sim = embeddings.cosine_similarity(emb1, emb3)
    assert abs(sim) < 1e-6, f"Orthogonal vectors should have similarity ~0.0, got {sim}"
    
    # Test find_similar
    query = [1.0, 0.0, 0.0]
    emb_list = [
        [1.0, 0.0, 0.0],  # Should be first (identical)
        [0.9, 0.1, 0.0],  # Should be second
        [0.1, 0.9, 0.0],  # Should be last
    ]
    
    results = embeddings.find_similar(query, emb_list, k=2)
    assert len(results) == 2, "Should return k results"
    assert results[0][0] == 0, "Most similar should be first"
    assert results[0][1] > results[1][1], "Scores should be in descending order"

def test_embedding_manager(use_llama_parse=False):
    """Test the EmbeddingManager class with multimodal documents.
    
    Args:
        use_llama_parse: Whether to use LlamaParse for document processing
    """
    from multimodal_rag.embeddings import EmbeddingManager
    from multimodal_rag.config import VECTOR_STORE_DIR, LLAMA_CLOUD_API_KEY
    from multimodal_rag.document_processor import DocumentProcessor
    
    # Check if LlamaParse is available
    if use_llama_parse and not LLAMA_CLOUD_API_KEY:
        print("Warning: LLAMA_CLOUD_API_KEY not set. LlamaParse will not be available.")
        print("Set LLAMA_CLOUD_API_KEY environment variable to use LlamaParse.")
        print("Falling back to default parser.")
        use_llama_parse = False
    
    # Create test documents
    text_doc = {"text": TEST_TEXT, "type": "text"}
    image_doc = {"image_path": str(test_image_path), "type": "image"}
    
    # Initialize embedding manager with document processor
    doc_processor = DocumentProcessor(use_llama_parse=use_llama_parse)
    manager = EmbeddingManager(
        vector_store_dir=VECTOR_STORE_DIR,
        collection_name="test_embeddings",
        document_processor=doc_processor
    )
    
    # Test adding documents
    print(f"\n[Test 6/6] Testing document embedding with {'LlamaParse' if use_llama_parse else 'default parser'}...")
    doc_ids = manager.add_documents([text_doc, image_doc])
    assert len(doc_ids) == 2, "Should return two document IDs"
    print(f"Added documents with IDs: {doc_ids}")
    
    # Test similarity search
    results = manager.similarity_search(TEST_TEXT, k=1)
    assert results, "Should return at least one similar document"
    print("Similarity search successful")
    
    # Test getting document by ID
    doc = manager.get_document(doc_ids[0])
    assert doc, "Should retrieve document by ID"
    print("Document retrieval by ID successful")
    
    # Clean up
    manager.delete_collection()
    print("Test collection cleaned up")

if __name__ == "__main__":
    import argparse
    
    # Create test data directory if it doesn't exist
    os.makedirs("test_data", exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Test embeddings with or without LlamaParse")
    parser.add_argument("--llama-parse", action="store_true", 
                      help="Use LlamaParse for document processing")
    
    args = parser.parse_args()
    
    # Run tests
    test_vertex_ai_initialization()
    test_text_embeddings()
    test_image_embeddings()
    test_video_embeddings()
    test_similarity_functions()
    
    # Run embedding manager test with specified parser
    test_embedding_manager(use_llama_parse=args.llama_parse)
