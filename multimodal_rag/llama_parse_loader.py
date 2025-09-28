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
            
            # Debug: Print the type and available attributes of parsed_result
            print(f"Parsed result type: {type(parsed_result)}")
            if hasattr(parsed_result, '__dict__'):
                print(f"Available attributes: {vars(parsed_result).keys()}")
            
            # Handle different response formats from LlamaParse
            try:
                # Try to access as a JobResult object first
                if hasattr(parsed_result, 'parsed'):
                    # Handle JobResult.parsed which might contain the actual content
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
                        content = getattr(result, 'parsed', result)  # Try to get parsed content or use result
                        if hasattr(content, 'text'):
                            doc_meta = {**metadata, "chunk": i}
                            documents.append(Document(
                                page_content=content.text,
                                metadata=doc_meta
                            ))
                
                # Handle list of results
                elif isinstance(parsed_result, list):
                    for i, result in enumerate(parsed_result):
                        content = getattr(result, 'parsed', result)  # Try to get parsed content or use result
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
                    documents.append(Document(
                        page_content=str(parsed_result),
                        metadata={**metadata, "fallback_conversion": True}
                    ))
                    
            except Exception as parse_error:
                print(f"Error processing LlamaParse result: {parse_error}")
                # If we can't process the result, include it as a string for debugging
                documents.append(Document(
                    page_content=f"Error processing document: {str(parse_error)}\n\nRaw result: {str(parsed_result)[:1000]}",
                    metadata={**metadata, "error": str(parse_error)}
                ))
            
            if not documents:
                raise ValueError("No valid document content found in LlamaParse response")
                
            return documents
            
        except Exception as e:
            print(f"LlamaParse failed: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            
            if not str(file_path).lower().endswith('.pdf'):
                # For non-PDF files, re-raise the exception to try other loaders
                print(f"Non-PDF file, re-raising error to try other loaders")
                raise
                
            print("Falling back to unstructured for PDF...")
            try:
                # Try with standard PDF parsing first
                elements = partition_pdf(
                    filename=str(file_path),
                    extract_images_in_pdf=False,
                    infer_table_structure=True,
                    chunking_strategy="by_title",
                    max_characters=1000,
                    new_after_n_chars=900,
                    combine_text_under_n_chars=500,
                )
                
                # If we got elements, use them
                if elements:
                    documents = [
                        Document(
                            page_content=str(element),
                            metadata={
                                **metadata,
                                "element_type": str(type(element).__name__),
                                "chunk": i,
                            }
                        )
                        for i, element in enumerate(elements)
                    ]
                    return documents
                
            except Exception as e:
                print(f"Standard PDF parsing failed: {e}")
                
            # Fallback to simpler parsing if standard parsing fails
            try:
                print("Trying fallback PDF parsing...")
                elements = partition_pdf(
                    filename=str(file_path),
                    extract_images_in_pdf=False,
                    infer_table_structure=False,
                    chunking_strategy="basic",
                )
                
                documents = [
                    Document(
                        page_content=str(element),
                        metadata={
                            **metadata,
                            "element_type": str(type(element).__name__),
                            "chunk": i,
                            "fallback_parser": True
                        }
                    )
                    for i, element in enumerate(elements)
                ]
                return documents
                
            except Exception as e:
                print(f"Fallback PDF parsing failed: {e}")
                raise ValueError(f"Failed to parse PDF with any available method: {e}")
            
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
