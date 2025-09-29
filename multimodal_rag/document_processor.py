"""Document processing utilities for the Multimodal RAG system."""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Set up logging
logger = logging.getLogger(__name__)

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from pptx import Presentation
from unstructured.partition.pdf import partition_pdf

from .llama_parse_loader import LlamaParseLoader
from .file_utils import get_file_type, is_supported_file
from langchain_community.document_loaders import (
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    TextLoader,
    UnstructuredFileLoader
)

from .config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_CHARACTERS,
    NEW_AFTER_N_CHARS,
    COMBINE_TEXT_UNDER_N_CHARS,
)
from .utils import encode_image, is_image_data, split_image_text_types
from .embeddings import EmbeddingManager
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Processes different types of documents and extracts their content with support for multimodal data."""
    
    def __init__(
        self, 
        embedding_manager: Optional[EmbeddingManager] = None,
        use_llama_parse: bool = False,
        output_dir: Optional[str] = None,
        llama_parse_kwargs: Optional[Dict[str, Any]] = None,
        **embedding_kwargs
    ):
        """Initialize the document processor with support for multimodal content.
        
        Args:
            embedding_manager: Optional EmbeddingManager instance for generating embeddings.
                If not provided, will create one with default settings.
            use_llama_parse: Whether to use LlamaParse for document parsing (default: False)
            output_dir: Directory to store temporary files (required if use_llama_parse is True)
            llama_parse_kwargs: Additional arguments to pass to LlamaParse
            **embedding_kwargs: Additional arguments to pass to EmbeddingManager if creating a new instance
        """
        if embedding_manager is None:
            # Create a new EmbeddingManager with vision capabilities enabled by default
            self.embedding_manager = EmbeddingManager(
                use_vision=True,
                **embedding_kwargs
            )
        else:
            self.embedding_manager = embedding_manager
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        self.supported_image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
        self.supported_video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
        self.use_llama_parse = use_llama_parse
        self.output_dir = output_dir or os.getcwd()
        self.llama_loader = None
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if self.use_llama_parse:
            if not output_dir:
                raise ValueError("output_dir must be provided when use_llama_parse is True")
                
            try:
                from llama_parse import LlamaParse
                self.llama_loader = LlamaParse(
                    result_type="markdown",
                    **(llama_parse_kwargs or {})
                )
                logger.info("LlamaParse initialized successfully")
            except ImportError:
                logger.warning("llama-parse package not installed. Install with: pip install llama-parse")
                self.use_llama_parse = False
            except Exception as e:
                logger.error(f"Failed to initialize LlamaParse: {e}")
                logger.info("Falling back to default parser")
                self.use_llama_parse = False
    
    def process_file(
        self,
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
        extract_images: bool = True,
        infer_table_structure: bool = True,
        use_llama_parse: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, List[Document]]:
        """Process a file and extract text, tables, and images.
        
        Args:
            file_path: Path to the file to process
            file_type: Optional file type (e.g., 'pdf', 'txt', 'image', 'video'). 
                     If None, will be inferred from file extension.
            extract_images: Whether to extract images from the file (for PDFs)
            infer_table_structure: Whether to infer table structure (for PDFs)
            use_llama_parse: Override for using LlamaParse. If None, uses class default.
            **kwargs: Additional arguments to pass to the loader
            
        Returns:
            Dictionary containing 'texts', 'tables', and 'images' as lists of Documents
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If the file type is not supported
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type if not provided
        if file_type is None:
            file_type = file_path.suffix.lower()
        
        # Initialize result dictionary
        result = {
            "texts": [],
            "tables": [],
            "images": []
        }
        
        # Handle metadata
        metadata = kwargs.get('metadata', {})
        
        # Use LlamaParse if enabled and available for this file type
        use_llama = use_llama_parse if use_llama_parse is not None else self.use_llama_parse
        if use_llama and self.llama_loader and file_type.lstrip('.') in ['pdf', 'docx', 'pptx', 'xlsx']:
            try:
                # Remove the dot for the file_type parameter since LlamaParse expects it without dot
                documents = self.llama_loader.load_file(file_path, file_type=file_type.lstrip('.'), **kwargs)
                result["texts"] = documents
                return result
            except Exception as e:
                print(f"Warning: LlamaParse failed with error: {e}. Falling back to default parser.")
                if use_llama_parse:  # Only if explicitly requested, otherwise just warn
                    raise
        
        # Process the file based on its type
        if file_type == '.pdf':
            return self._process_pdf(
                file_path, 
                extract_images=extract_images,
                infer_table_structure=infer_table_structure
            )
        elif file_type in ['.pptx', '.ppt']:
            return self._process_pptx(file_path, metadata)
        elif file_type in ['.xlsx', '.xls']:
            return self._process_excel(file_path, metadata)
        elif file_type == '.csv':
            return self._process_csv(file_path)
        elif file_type in ['.txt', '.md', '.markdown']:
            return self._process_text(file_path)
        elif file_type in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
            return self._process_image(file_path, metadata)
        else:
            # Try with unstructured loader as fallback
            try:
                loader = UnstructuredFileLoader(str(file_path))
                docs = loader.load()
                return {"texts": docs, "tables": [], "images": []}
            except Exception as e:
                raise ValueError(
                    f"Unsupported file type: {file_type}. "
                    "Supported types: .pdf, .pptx, .ppt, .xlsx, .xls, .csv, .txt, "
                    ".png, .jpg, .jpeg, .tiff, .bmp, .gif"
                )
    
    def _process_image(self, file_path: Union[str, Path], metadata: Optional[Dict] = None) -> Dict[str, List[Document]]:
        """Process an image file to extract text and generate embeddings.
        
        Args:
            file_path: Path to the image file
            metadata: Optional metadata to include with the document
            
        Returns:
            Dictionary containing 'texts' with the processed image document
        """
        from PIL import Image
        import base64
        import io
        
        try:
            # Read and encode image
            with open(file_path, 'rb') as img_file:
                b64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Generate image description using Gemini
            image_description = self._generate_image_description(file_path)
            
            # Extract text using OCR
            extracted_text = self._extract_text_with_ocr(file_path)
            
            # Combine description and OCR text
            combined_text = f"""Image Description: {image_description}
            
            Extracted Text:
            {extracted_text}"""
            
            # Generate embeddings
            try:
                # Generate image embedding
                image = Image.open(file_path)
                image_embedding = self.embedding_manager.embeddings.embed_image(image)
                
                # Generate text embedding
                text_embedding = self.embedding_manager.embeddings.embed_documents([combined_text])[0]
                
                # Create document with all metadata
                doc_metadata = {
                    "source": str(file_path),
                    "type": "image",
                    "description": image_description,
                    "extracted_text": extracted_text,
                    "image_embedding": image_embedding,
                    "text_embedding": text_embedding,
                    "has_image": True,
                    "image_data": b64_image  # Store base64 for display
                }
                
                if metadata:
                    doc_metadata.update(metadata)
                
                doc = Document(
                    page_content=combined_text,
                    metadata=doc_metadata
                )
                
                return {"texts": [doc], "tables": [], "images": [doc]}
                
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                # Return document without embeddings
                doc = Document(
                    page_content=combined_text,
                    metadata={
                        "source": str(file_path),
                        "type": "image",
                        "description": image_description,
                        "extracted_text": extracted_text,
                        "error": str(e),
                        "has_image": True,
                        "image_data": b64_image
                    }
                )
                return {"texts": [doc], "tables": [], "images": [doc]}
                
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            return {"texts": [], "tables": [], "images": []}
    
    def _generate_image_description(self, image_path: Union[str, Path]) -> str:
        """Generate a description of the image using Gemini."""
        try:
            import google.generativeai as genai
            
            # Initialize Gemini
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            model = genai.GenerativeModel('gemini-pro-vision')
            
            # Load image
            img = Image.open(image_path)
            
            # Generate description
            response = model.generate_content(["Describe this image in detail, including any text, objects, and their relationships.", img])
            
            return response.text
            
        except Exception as e:
            logger.warning(f"Error generating image description: {e}")
            return "[No description available]"
    
    def _extract_text_with_ocr(self, image_path: Union[str, Path]) -> str:
        """Extract text from image using Tesseract OCR."""
        try:
            import pytesseract
            from PIL import Image
            
            # Open image
            img = Image.open(image_path)
            
            # Extract text
            text = pytesseract.image_to_string(img)
            
            return text.strip() if text else "[No text could be extracted]"
            
        except Exception as e:
            logger.warning(f"Error extracting text with OCR: {e}")
            return "[Text extraction failed]"
            
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}", exc_info=True)
            # Return empty results on error
            return {"texts": [], "tables": [], "images": []}
    
    def _process_pptx(self, file_path: Path) -> Dict[str, List[Document]]:
        """Process a PowerPoint file and extract text and slides.
        
        Args:
            file_path: Path to the PowerPoint file
            
        Returns:
            Dictionary containing 'texts' and empty 'tables' and 'images' lists
        """
        try:
            loader = UnstructuredPowerPointLoader(str(file_path))
            docs = loader.load()
            return {"texts": docs, "tables": [], "images": []}
        except Exception as e:
            print(f"Error processing PowerPoint file {file_path}: {e}")
            return {"texts": [], "tables": [], "images": []}
    
    def _process_excel(self, file_path: Path) -> Dict[str, List[Document]]:
        """Process an Excel file and extract data from all sheets.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary containing 'texts' (for sheet data) and 'tables' (for structured data)
        """
        try:
            # First try with UnstructuredExcelLoader
            loader = UnstructuredExcelLoader(str(file_path), mode="elements")
            docs = loader.load()
            
            # Separate text and tables
            texts = []
            tables = []
            
            for doc in docs:
                if doc.metadata.get("category") == "Table":
                    tables.append(doc)
                else:
                    texts.append(doc)
            
            return {"texts": texts, "tables": tables, "images": []}
            
        except Exception as e:
            print(f"Error processing Excel file {file_path}: {e}")
            return {"texts": [], "tables": [], "images": []}
    
    def _process_csv(self, file_path: Path) -> Dict[str, List[Document]]:
        """Process a CSV file and extract data.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary containing 'texts' with CSV data
        """
        try:
            # First try with pandas for better handling of various CSV formats
            try:
                import pandas as pd
                df = pd.read_csv(file_path, nrows=1000)  # Read first 1000 rows
                text = df.to_string()
                doc = Document(
                    page_content=text,
                    metadata={"source": str(file_path), "type": "csv"}
                )
                return {"texts": [doc], "tables": [], "images": []}
            except Exception as pd_error:
                # Fall back to CSVLoader if pandas fails
                loader = CSVLoader(str(file_path))
                docs = loader.load()
                return {"texts": docs, "tables": [], "images": []}
                
        except Exception as e:
            print(f"Error processing CSV file {file_path}: {e}")
            return {"texts": [], "tables": [], "images": []}
    
    def _process_text(self, file_path: Path) -> Dict[str, List[Document]]:
        """Process a plain text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dictionary containing 'texts' with file content
        """
        try:
            loader = TextLoader(str(file_path))
            docs = loader.load()
            return {"texts": docs, "tables": [], "images": []}
        except Exception as e:
            print(f"Error processing text file {file_path}: {e}")
            return {"texts": [], "tables": [], "images": []}
    
    def _process_pdf(
        self,
        file_path: Path,
        extract_images: bool = True,
        infer_table_structure: bool = True,
    ) -> Dict[str, List[Document]]:
        """Process a PDF file and extract text, tables, and images."""
        from PyPDF2 import PdfReader
        from pdf2image import convert_from_path
        import tempfile
        import os
        
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # First, extract text using PyPDF2
            text = ""
            try:
                with open(file_path, 'rb') as f:
                    pdf_reader = PdfReader(f)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
            except Exception as e:
                logger.warning(f"Error extracting text with PyPDF2: {e}")
            
            # Process images if needed
            image_docs = []
            if extract_images:
                try:
                    # Convert PDF pages to images
                    images = convert_from_path(file_path)
                    temp_dir = tempfile.mkdtemp()
                    
                    for i, image in enumerate(images):
                        img_path = os.path.join(temp_dir, f"page_{i+1}.jpg")
                        image.save(img_path, 'JPEG')
                        
                        # Process each image
                        try:
                            img_doc = self._process_image(img_path, {"source": str(file_path), "page": i+1})
                            if img_doc and "texts" in img_doc and img_doc["texts"]:
                                image_docs.extend(img_doc["texts"])
                        except Exception as e:
                            logger.warning(f"Error processing image from PDF page {i+1}: {e}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Error extracting images from PDF: {e}")
            
            # Create text document if we have text
            text_docs = []
            if text.strip():
                text_docs = [Document(
                    page_content=text,
                    metadata={"source": str(file_path), "type": "pdf_text"}
                )]
            
            return {
                "texts": text_docs,
                "tables": [],  # Tables are included in text extraction
                "images": image_docs
            }
            
        except Exception as e:
            # Clean up any temporary files on error
            if 'elements' in locals():
                del elements
            raise Exception(f"Error processing PDF {file_path.name}: {str(e)}") from e
    
    def _process_pptx(self, file_path: Union[str, Path], extra_info: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Extract text from a PowerPoint presentation.
        
        Args:
            file_path: Path to the PowerPoint file
            extra_info: Additional metadata to include in the documents
            
        Returns:
            List of Document objects, one per slide
        """
        prs = Presentation(file_path)
        documents = []
        
        for i, slide in enumerate(prs.slides):
            text = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
            
            if text:
                content = "\n".join(filter(None, text))
                metadata = {
                    "source": str(file_path),
                    "page": i + 1,
                    "file_type": "pptx",
                    "total_slides": len(prs.slides),
                    "slide_number": i + 1,
                    **(extra_info or {})
                }
                documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def _process_xlsx(self, file_path: Union[str, Path], extra_info: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Extract data from an Excel file.
        
        Args:
            file_path: Path to the Excel file
            extra_info: Additional metadata to include in the documents
            
        Returns:
            List of Document objects, one per sheet
        """
        xls = pd.ExcelFile(file_path)
        documents = []
        
        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                
                # Convert DataFrame to markdown for better formatting
                content = f"# {sheet_name}\n\n{df.to_markdown()}"
                
                metadata = {
                    "source": str(file_path),
                    "file_type": "xlsx",
                    "sheet_name": sheet_name,
                    "shape": f"{df.shape[0]} rows x {df.shape[1]} columns",
                    **(extra_info or {})
                }
                documents.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                print(f"Error processing sheet '{sheet_name}': {e}")
        
        return documents

    def _get_file_type(self, file_path: str) -> str:
        """Get the file type from the file extension."""
        return os.path.splitext(file_path)[1].lower()
        
    def _is_image_file(self, file_path: str) -> bool:
        """Check if the file is an image based on its extension."""
        file_ext = self._get_file_type(file_path)
        return file_ext in self.supported_image_extensions
            
    def _extract_text_from_image(self, image_path: Union[str, Path]) -> str:
        """Extract text from an image using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text from the image
            
        Raises:
            ImportError: If required OCR libraries are not installed
            RuntimeError: If text extraction fails
        """
        try:
            # Try using pytesseract if available
            try:
                import pytesseract
                from PIL import Image
                
                # Open the image
                img = Image.open(image_path)
                
                # Convert to grayscale for better OCR results
                img = img.convert('L')
                
                # Use pytesseract to extract text
                text = pytesseract.image_to_string(img)
                
                # Clean up the extracted text
                if text:
                    text = ' '.join(line.strip() for line in text.split('\n') if line.strip())
                
                return text if text else ""
                
            except ImportError:
                logger.warning("pytesseract not available, trying easyocr...")
                
            # Fall back to easyocr if available
            try:
                import easyocr
                
                # Initialize the OCR reader
                reader = easyocr.Reader(['en'])
                
                # Read the image
                result = reader.readtext(str(image_path))
                
                # Extract and join the text
                text = ' '.join([item[1] for item in result])
                
                return text if text else ""
                
            except ImportError:
                logger.warning("easyocr not available, no OCR will be performed")
                return ""
                
        except Exception as e:
            logger.warning(f"Error extracting text from image: {str(e)}")
            return ""
    
    def _get_content_type(self, file_path: str, file_ext: str) -> str:
        """Determine the content type of the file."""
        if self._is_image_file(file_path):
            return 'image'
        elif self._is_video_file(file_path):
            return 'video'
        elif file_ext == '.pdf':
            return 'document'
        elif file_ext in ['.docx', '.doc']:
            return 'document'
        elif file_ext in ['.pptx', '.ppt']:
            return 'presentation'
        elif file_ext in ['.xlsx', '.xls', '.csv']:
            return 'spreadsheet'
        else:
            return 'text'
            
    def _process_image(self, file_path: Union[str, Path], metadata: Dict[str, Any]) -> List[Document]:
        """Process an image file and extract text and visual features with multimodal embeddings.
        
        Args:
            file_path: Path to the image file
            metadata: Metadata to include with the document
            
        Returns:
            List containing a single Document with image content and embeddings
            
        Raises:
            ValueError: If the image cannot be processed
        """
        try:
            from PIL import Image
            import base64
            from io import BytesIO
            
            # Load and preprocess the image
            try:
                image = Image.open(file_path).convert('RGB')
                # Resize if needed to reduce memory usage
                max_size = (1024, 1024)
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            except Exception as e:
                raise ValueError(f"Failed to load image {file_path}: {str(e)}")
            
            # Generate image embedding
            try:
                image_embedding = self.embedding_manager.embed_image(image)[0]
            except Exception as e:
                logger.warning(f"Could not generate image embedding: {str(e)}")
                image_embedding = None
            
            # Extract text from image using OCR if needed
            text = ""
            try:
                text = self._extract_text_from_image(file_path)
            except Exception as e:
                logger.warning(f"Could not extract text from image: {str(e)}")
            
            # Prepare image data for storage
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Create document with both text and image data
            doc_metadata = {
                **metadata,
                'embedding': image_embedding,
                'content_type': 'image',
                'has_image': True,
                'image_data': img_str,
                'image_format': 'jpeg',
                'image_size': os.path.getsize(file_path),
                'image_dimensions': f"{image.width}x{image.height}"
            }
            
            # Add text if available
            if text.strip():
                doc_metadata['extracted_text'] = text
            
            doc = Document(
                page_content=text or "[Image content]" + (f"\nDetected text: {text}" if text else ""),
                metadata=doc_metadata
            )
            
            return [doc]
                
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to process image {file_path}: {str(e)}")


def generate_summaries(
    texts: List[str],
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.0,
    max_tokens: int = 1000,
) -> List[str]:
    """Generate summaries for a list of text chunks.
    
    Args:
        texts: List of text chunks to summarize
        model_name: Name of the model to use for summarization
        temperature: Temperature for text generation
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        List of summaries
    """
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    if not texts:
        return []
    
    prompt_template = """You are an assistant tasked with summarizing text for retrieval. 
    These summaries will be embedded and used to retrieve the raw text. 
    Give a concise summary of the text that is well optimized for retrieval.
    
    Text: {text}
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    try:
        # First try using VertexAI
        from langchain_google_vertexai import VertexAI
        
        llm = VertexAI(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        chain = prompt | llm
        return chain.batch([{"text": text} for text in texts], {"max_concurrency": 5})
        
    except Exception as e:
        print(f"Warning: Could not use VertexAI: {e}")
        print("Falling back to a simple text truncation for summarization...")
        
        # Fallback: Just return the first few sentences as a simple summary
        def simple_summary(text: str, max_sentences: int = 3) -> str:
            import re
            # Split into sentences (very basic)
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return ' '.join(sentences[:max_sentences])
            
        return [simple_summary(text) for text in texts]
