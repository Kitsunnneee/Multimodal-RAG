"""Configuration settings for the Multimodal RAG system."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_CACHE_DIR = BASE_DIR / ".cache"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Storage Configuration
USE_GCS = os.getenv("USE_GCS", "false").lower() == "true"

# Google Cloud Configuration (only used if USE_GCS is True)
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")

# Model configuration  
MODEL_NAME = "gemini-1.5-pro"  # Use a more stable model
EMBEDDING_MODEL_NAME = "text-embedding-005"
MULTIMODAL_EMBEDDING_MODEL = "multimodalembedding"
TOKEN_LIMIT = 8192  # Max tokens for Gemini model

# Vector Search Configuration
VECTOR_SEARCH_INDEX_NAME = "mm_rag_langchain_index"
DEPLOYED_INDEX_ID = "mm_rag_langchain_index_endpoint"
DIMENSIONS = 768  # Dimensions for text-embedding-005
MULTIMODAL_DIMENSIONS = 1408  # Dimensions for multimodal embeddings

# Vertex AI Configuration
VERTEX_AI_LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")
VERTEX_AI_PROJECT = os.getenv("VERTEX_AI_PROJECT", PROJECT_ID)
VERTEX_AI_ENDPOINT = os.getenv(
    "VERTEX_AI_ENDPOINT",
    f"projects/{VERTEX_AI_PROJECT}/locations/{VERTEX_AI_LOCATION}/publishers/google/models/{MULTIMODAL_EMBEDDING_MODEL}"
)

# Document Processing
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200
MAX_CHARACTERS = 4000
NEW_AFTER_N_CHARS = 3800
COMBINE_TEXT_UNDER_N_CHARS = 2000

# LlamaParse Configuration
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
LLAMA_PARSE_RESULT_TYPE = "markdown"  # or "text"
LLAMA_PARSE_NUM_WORKERS = 4
LLAMA_PARSE_VERBOSE = True
LLAMA_PARSE_TIMEOUT = 300  # seconds

# Ensure directories exist
for directory in [DATA_DIR, MODEL_CACHE_DIR, VECTOR_STORE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class ConfigError(Exception):
    """Raised when there's an error in configuration."""
    pass

def validate_config():
    """Validate that all required configuration is set."""
    if USE_GCS and not all([PROJECT_ID, GCS_BUCKET]):
        raise ConfigError(
            "Google Cloud Storage is enabled but required configuration is missing. "
            "Please set GOOGLE_CLOUD_PROJECT and GCS_BUCKET environment variables."
        )
    
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and USE_GCS:
        raise ConfigError(
            "GOOGLE_APPLICATION_CREDENTIALS environment variable is required when using GCS."
        )
    
    # Check LlamaParse API key if using LlamaParse
    if os.getenv("USE_LLAMA_PARSE", "false").lower() == "true" and not LLAMA_CLOUD_API_KEY:
        print("Warning: LLAMA_CLOUD_API_KEY environment variable is not set. "
              "LlamaParse will not be available.")

# Validate configuration on import
try:
    validate_config()
except ConfigError as e:
    print(f"Warning: {e} Some features may not work correctly.")
    if not USE_GCS:
        print("Continuing with local storage...")
