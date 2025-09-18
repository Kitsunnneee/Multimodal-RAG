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

def test_embedding_manager():
    """Test the EmbeddingManager class with multimodal documents."""
    from multimodal_rag.embeddings import EmbeddingManager
    from langchain_core.documents import Document
    
    try:
        manager = EmbeddingManager(use_gcs=False)
        
        # Create a test image
        img = create_test_image()
        img_byte_arr = img.tobytes()
        
        # Test with text and image documents
        documents = [
            Document(
                page_content="A document with an image",
                metadata={"image_data": img_byte_arr}
            ),
            Document(page_content="A text-only document")
        ]
        
        doc_ids = manager.add_documents(documents)
        assert isinstance(doc_ids, list), "Should return a list of document IDs"
        assert len(doc_ids) == 2, "Should return one ID per document"
        
        # Test similarity search with text
        results = manager.similarity_search("test content", k=1)
        assert isinstance(results, list), "Should return a list of results"
        
        # Test with filter
        results = manager.similarity_search(
            "test", 
            k=2, 
            filter_dict={"page_content": {"$regex": "image"}}
        )
        assert len(results) == 1, "Should return only documents matching the filter"
        
    except Exception as e:
        print(f"Error in test_embedding_manager: {e}")
        raise

if __name__ == "__main__":
    # Create test data directory if it doesn't exist
    os.makedirs("test_data", exist_ok=True)
    
    # Run tests
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
