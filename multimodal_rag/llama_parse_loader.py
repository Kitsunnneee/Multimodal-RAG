"""Document loader using LlamaParse for advanced document parsing."""
import asyncio
import logging
import nest_asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

from langchain_core.documents import Document
from llama_cloud_services import LlamaParse
from unstructured.partition.pdf import partition_pdf

from .config import DATA_DIR

logger = logging.getLogger(__name__)

class LlamaParseLoader:
    """Document loader that uses LlamaParse for advanced document parsing."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        result_type: str = "markdown",
        num_workers: int = 2,  # Reduced from 4 to 2 to avoid rate limiting
        verbose: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        timeout: int = 120,  # 2 minutes timeout
    ):
        """Initialize the LlamaParse loader.
        
        Args:
            api_key: LlamaParse API key. If None, will use LLAMA_CLOUD_API_KEY env var.
            result_type: Type of output to return ("markdown" or "text")
            num_workers: Number of parallel workers for processing (recommended: 1-2)
            verbose: Whether to print verbose output
            output_dir: Directory to save extracted images (if any)
            timeout: Timeout in seconds for API requests
        """
        api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError(
                "LlamaParse API key not provided. "
                "Please set LLAMA_CLOUD_API_KEY environment variable or pass api_key."
            )
            
        self.parser = LlamaParse(
            api_key=api_key,
            result_type=result_type,
            num_workers=min(num_workers, 2),  # Cap at 2 workers to avoid rate limiting
            verbose=verbose,
            timeout=timeout,
        )
        self.output_dir = Path(output_dir) if output_dir else DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
    
    async def _aload_file(self, file_path: Path, file_type: str, metadata: Dict[str, Any]) -> List[Document]:
        """Async implementation of file loading with LlamaParse."""
        try:
            # Ensure file exists and is accessible
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"No read permissions for file: {file_path}")
                
            # Read file content first to ensure it's accessible
            try:
                with open(file_path, 'rb') as f:
                    # Just check if we can read the first few bytes
                    f.read(1024)
            except Exception as e:
                raise IOError(f"Error reading file {file_path}: {str(e)}")
            
            # Use aparser for async parsing
            logger.info(f"Processing file: {file_path}")
            try:
                parsed_result = await self.parser.aparse(str(file_path), extra_info=metadata)
                return self._handle_parsed_result(parsed_result, metadata)
            except Exception as e:
                logger.error(f"Error in LlamaParse for {file_path}: {str(e)}", exc_info=True)
                return [
                    Document(
                        page_content=f"Error processing file with LlamaParse: {str(e)}",
                        metadata={"error": str(e), **metadata}
                    )
                ]
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}", exc_info=True)
            return [
                Document(
                    page_content=f"Error loading file: {str(e)}",
                    metadata={"error": str(e), "file_path": str(file_path), **metadata}
                )
            ]
    
    def _handle_parsed_result(self, parsed_result: Any, metadata: Dict[str, Any]) -> List[Document]:
        """Handle different types of parsed results from LlamaParse."""
        documents = []
        
        # Handle JobResult object
        if hasattr(parsed_result, 'parsed'):
            content = parsed_result.parsed
            if hasattr(content, 'text'):
                documents.append(Document(
                    page_content=content.text,
                    metadata=metadata
                ))
        
        # Handle direct text attribute
        elif hasattr(parsed_result, 'text'):
            documents.append(Document(
                page_content=parsed_result.text,
                metadata=metadata
            ))
        
        # Handle batch results
        elif hasattr(parsed_result, 'results') and isinstance(parsed_result.results, list):
            for i, result in enumerate(parsed_result.results):
                content = getattr(result, 'parsed', result)
                if hasattr(content, 'text'):
                    doc_meta = {**metadata, "chunk": i}
                    documents.append(Document(
                        page_content=content.text,
                        metadata=doc_meta
                    ))
        
        # Handle list of results
        elif isinstance(parsed_result, list):
            for i, result in enumerate(parsed_result):
                content = getattr(result, 'parsed', result)
                if hasattr(content, 'text'):
                    doc_meta = {**metadata, "chunk": i}
                    documents.append(Document(
                        page_content=content.text,
                        metadata=doc_meta
                    ))
        
        # Handle dictionary response
        elif isinstance(parsed_result, dict):
            if 'text' in parsed_result:
                documents.append(Document(
                    page_content=parsed_result['text'],
                    metadata=metadata
                ))
            elif 'parsed' in parsed_result and hasattr(parsed_result['parsed'], 'text'):
                documents.append(Document(
                    page_content=parsed_result['parsed'].text,
                    metadata=metadata
                ))
        
        # If we still don't have documents, try to convert the entire result to string
        if not documents and parsed_result is not None:
            try:
                documents.append(Document(
                    page_content=str(parsed_result),
                    metadata={**metadata, "fallback_conversion": True}
                ))
            except Exception as e:
                logger.warning(f"Could not convert parsed result to string: {str(e)}")
                documents.append(Document(
                    page_content=f"Error processing document: {str(e)}\n\nRaw result: {str(parsed_result)[:1000]}",
                    metadata={**metadata, "error": str(e)}
                ))
        
        return documents
        
    def load_file(
        self,
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Load and parse a single file.
        
        Args:
            file_path: Path to the file to load
            file_type: Optional file type/extension
            extra_info: Additional metadata to include in the document
            
        Returns:
            List of Document objects from the file
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists() or not file_path.is_file():
                raise FileNotFoundError(f"File not found or is not a file: {file_path}")
                
            # Use provided file_type or extract from path
            file_ext = file_type or file_path.suffix.lower().lstrip('.')
            if not file_ext:
                raise ValueError(f"Could not determine file type for {file_path}")
                
            metadata = extra_info or {}
            metadata.update({
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_ext
            })
            
            # Ensure the file is readable
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"No read permissions for file: {file_path}")
            
            # Run the async load operation in an event loop
            loop = asyncio.get_event_loop()
            documents = loop.run_until_complete(
                self._aload_file(file_path, file_ext, metadata)
            )
            
            if not documents:
                logger.warning(f"No documents were extracted from {file_path}")
                return [
                    Document(
                        page_content="No content extracted from file",
                        metadata=metadata
                    )
                ]
                
            return documents
            
        except asyncio.CancelledError:
            logger.error(f"File processing was cancelled: {file_path}")
            raise
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}", exc_info=True)
            return [
                Document(
                    page_content=f"Error loading file: {str(e)}",
                    metadata={"error": str(e), **metadata} if 'metadata' in locals() else {"error": str(e), "file_path": str(file_path)}
                )
            ]
    
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
