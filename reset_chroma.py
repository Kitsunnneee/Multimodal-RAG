"""Script to reset the Chroma database."""
import os
import shutil
from pathlib import Path

def reset_chroma():
    # Define the vector store directory
    vector_store_dir = Path(__file__).parent / "vector_store"
    
    if vector_store_dir.exists():
        print(f"Removing existing vector store at: {vector_store_dir}")
        shutil.rmtree(vector_store_dir)
    
    # Create a new empty directory
    os.makedirs(vector_store_dir, exist_ok=True)
    print(f"Created new empty vector store at: {vector_store_dir}")
    
    # Create a .gitkeep file to keep the directory in git
    (vector_store_dir / ".gitkeep").touch()
    print("Reset complete. You can now restart your RAG system.")

if __name__ == "__main__":
    reset_chroma()
