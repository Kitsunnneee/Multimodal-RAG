"""Document processing utilities for the Multimodal RAG system."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

from .config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_CHARACTERS,
    NEW_AFTER_N_CHARS,
    COMBINE_TEXT_UNDER_N_CHARS,
    DATA_DIR,
)
from .utils import encode_image, is_image_data, split_image_text_types


class DocumentProcessor:
    """Process documents and extract text, tables, and images."""
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """Initialize the document processor.
        
        Args:
            output_dir: Directory to save extracted images. If None, uses DATA_DIR.
        """
        self.output_dir = Path(output_dir) if output_dir else DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_pdf(
        self,
        file_path: Union[str, Path],
        extract_images: bool = True,
        infer_table_structure: bool = True,
    ) -> Dict[str, List[Document]]:
        """Process a PDF file and extract text, tables, and images.
        
        Args:
            file_path: Path to the PDF file
            extract_images: Whether to extract images from the PDF
            infer_table_structure: Whether to infer table structure
            
        Returns:
            Dictionary containing 'texts' and 'tables' as lists of Documents
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract elements from PDF
        elements = partition_pdf(
            filename=str(file_path),
            extract_images_in_pdf=extract_images,
            infer_table_structure=infer_table_structure,
            chunking_strategy="by_title",
            max_characters=MAX_CHARACTERS,
            new_after_n_chars=NEW_AFTER_N_CHARS,
            combine_text_under_n_chars=COMBINE_TEXT_UNDER_N_CHARS,
            image_output_dir_path=str(self.output_dir),
        )
        
        # Categorize elements
        texts = []
        tables = []
        
        for element in elements:
            element_str = str(element)
            if "unstructured.documents.elements.Table" in str(type(element)):
                tables.append(element_str)
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                texts.append(element_str)
        
        # Convert to Documents
        text_docs = [Document(page_content=text) for text in texts]
        table_docs = [Document(page_content=table) for table in tables]
        
        # Process images if any were extracted
        image_docs = []
        if extract_images:
            image_files = list(self.output_dir.glob("*.png")) + list(self.output_dir.glob("*.jpg"))
            for img_path in image_files:
                try:
                    b64_image = encode_image(img_path)
                    image_docs.append(Document(page_content=b64_image, metadata={"source": str(img_path.name)}))
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
        
        return {
            "texts": text_docs,
            "tables": table_docs,
            "images": image_docs,
        }
    
    def chunk_documents(
        self,
        documents: List[Document],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> List[Document]:
        """Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects to chunk
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunked Document objects
        """
        if not documents:
            return []
            
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        # Extract text from documents
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Split the texts
        chunks = []
        for i, text in enumerate(texts):
            chunks.extend(
                Document(
                    page_content=chunk,
                    metadata={
                        **metadatas[i],
                        "chunk": j,
                        "total_chunks": len(text_splitter.split_text(text)),
                    },
                )
                for j, chunk in enumerate(text_splitter.split_text(text))
            )
            
        return chunks


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
    from langchain_google_vertexai import VertexAI
    
    if not texts:
        return []
    
    prompt_template = """You are an assistant tasked with summarizing text for retrieval. 
    These summaries will be embedded and used to retrieve the raw text. 
    Give a concise summary of the text that is well optimized for retrieval.
    
    Text: {text}
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    llm = VertexAI(
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    
    chain = prompt | llm
    return chain.batch([{"text": text} for text in texts], {"max_concurrency": 5})
