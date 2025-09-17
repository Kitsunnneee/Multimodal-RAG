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

# Google Cloud Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GCS_BUCKET = os.getenv("GCS_BUCKET", "your-bucket-name")

# Model Configuration
MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL_NAME = "text-embedding-005"
TOKEN_LIMIT = 8192  # Max tokens for Gemini model

# Vector Search Configuration
VECTOR_SEARCH_INDEX_NAME = "mm_rag_langchain_index"
DEPLOYED_INDEX_ID = "mm_rag_langchain_index_endpoint"
DIMENSIONS = 768  # Dimensions for text-embedding-005

# Document Processing
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200
MAX_CHARACTERS = 4000
NEW_AFTER_N_CHARS = 3800
COMBINE_TEXT_UNDER_N_CHARS = 2000

# Ensure directories exist
for directory in [DATA_DIR, MODEL_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class ConfigError(Exception):
    """Raised when there's an error in configuration."""
    pass

def validate_config():
    """Validate that all required configuration is set."""
    required_vars = ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        raise ConfigError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please set these in your .env file or environment variables."
        )

# Validate configuration on import
try:
    validate_config()
except ConfigError as e:
    import warnings
    warnings.warn(str(e))
