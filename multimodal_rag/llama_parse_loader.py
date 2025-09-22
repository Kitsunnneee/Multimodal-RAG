"""Document loader using LlamaParse for advanced document parsing."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from llama_cloud_services import LlamaParse
from unstructured.partition.pdf import partition_pdf

from .config import DATA_DIR
from .utils import is_image_data, split_image_text_types


class LlamaParseLoader:
    """Document loader that uses LlamaParse for advanced document parsing."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        result_type: str = "markdown",
        num_workers: int = 4,
        verbose: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the LlamaParse loader.
        
        Args:
            api_key: LlamaParse API key. If None, will use LLAMA_CLOUD_API_KEY env var.
            result_type: Type of output to return ("markdown" or "text")
            num_workers: Number of parallel workers for processing
            verbose: Whether to print verbose output
            output_dir: Directory to save extracted images (if any)
        """
        self.parser = LlamaParse(
            api_key=api_key or os.getenv("LLAMA_CLOUD_API_KEY"),
            result_type=result_type,
            num_workers=num_workers,
            verbose=verbose,
        )
        self.output_dir = Path(output_dir) if output_dir else DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_file(
        self,
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Load and parse a document using LlamaParse.
        
        Args:
            file_path: Path to the file to load
            file_type: File type (e.g., 'pdf', 'docx'). If None, inferred from file extension.
            extra_info: Additional metadata to include in the documents
            
        Returns:
            List of Document objects with parsed content and metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Set default file type if not provided
        if file_type is None:
            file_type = file_path.suffix.lower().lstrip('.')
        
        # Prepare metadata
        metadata = {
            "source": str(file_path),
            "file_name": file_path.name,
            "file_type": file_type,
            **(extra_info or {})
        }
        
        try:
            # Parse the document using LlamaParse
            parsed_result = self.parser.parse(file_path, extra_info=metadata)
            
            # Convert to LangChain documents
            documents = []
            if isinstance(parsed_result, list):
                for i, result in enumerate(parsed_result):
                    doc_meta = {**metadata, "chunk": i}
                    documents.append(Document(page_content=result.text, metadata=doc_meta))
            else:
                documents.append(Document(page_content=parsed_result.text, metadata=metadata))
            
            return documents
            
        except Exception as e:
            # Fallback to unstructured if LlamaParse fails
            print(f"LlamaParse failed: {e}. Falling back to unstructured...")
            elements = partition_pdf(
                filename=str(file_path),
                extract_images_in_pdf=False,  # We'll handle images separately
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=1000,
                new_after_n_chars=900,
                combine_text_under_n_chars=500,
            )
            
            # Convert elements to documents
            documents = []
            for i, element in enumerate(elements):
                doc_meta = {
                    **metadata,
                    "element_type": str(type(element).__name__),
                    "chunk": i,
                }
                documents.append(Document(page_content=str(element), metadata=doc_meta))
            
            return documents
    
    def load_directory(
        self,
        directory: Union[str, Path],
        glob_pattern: str = "**/*.pdf",
        recursive: bool = True,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Load and parse all matching files in a directory.
        
        Args:
            directory: Directory containing files to load
            glob_pattern: Pattern to match files against
            recursive: Whether to search subdirectories
            extra_info: Additional metadata to include in all documents
            
        Returns:
            List of Document objects from all matching files
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")
        
        # Find all matching files
        if recursive:
            file_paths = list(directory.glob(glob_pattern))
        else:
            file_paths = list(directory.glob(glob_pattern))
        
        # Process all files in parallel
        documents = []
        for file_path in file_paths:
            try:
                file_docs = self.load_file(file_path, extra_info=extra_info)
                documents.extend(file_docs)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        return documents
