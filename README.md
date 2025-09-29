# Multimodal RAG with Gemini and Pinecone

A Python implementation of a Multimodal Retrieval Augmented Generation (RAG) system using Google's Gemini model for embeddings and Pinecone as the vector store. This system can process and retrieve information from various document types including PDFs, images, and more.

## Features

- **Multimodal Processing**: Handle text, images, and documents in a unified pipeline
- **Advanced OCR**: Extract text from images using Tesseract and EasyOCR
- **Vector Embeddings**: Generate rich embeddings for both text and images
- **Efficient Retrieval**: Fast similarity search using Pinecone vector database
- **Flexible Document Types**: Support for PDFs, Word, Excel, PowerPoint, and images
- **Configurable**: Easy to customize for different use cases and document types

## Prerequisites

- Python 3.9+
- Google Cloud account with Gemini API access
- Pinecone account (free tier available)
- Tesseract OCR (for local text extraction from images)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd multimodal-rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   For OCR support, install Tesseract:
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## Configuration

Create a `.env` file in the project root with the following variables:

```
# Google Cloud
GOOGLE_API_KEY=your-google-api-key

# Pinecone
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment
PINECONE_INDEX=multimodal-rag

# Optional: Tesseract path (if not in system PATH)
TESSERACT_CMD=/usr/local/bin/tesseract  # Update this path as needed
```

## Usage

### Basic Example

```python
from multimodal_rag.document_processor import DocumentProcessor
from multimodal_rag.embeddings import EmbeddingManager
import os

# Initialize the embedding manager
embedding_manager = EmbeddingManager(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model_name="gemini-pro-vision",
    vector_store_type="pinecone",
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
    pinecone_index_name="multimodal-rag"
)

# Initialize the document processor
processor = DocumentProcessor(embedding_manager=embedding_manager)

# Process a document
documents = processor.process_file("path/to/your/document.pdf")

# Print the results
for doc in documents:
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")
```

### Process Images

```python
# Process an image
documents = processor.process_file("path/to/your/image.jpg")

# The image will be processed with OCR (if Tesseract is installed)
# and embedded using Gemini's multimodal capabilities
for doc in documents:
    print(f"Extracted text: {doc.page_content}")
    print(f"Image dimensions: {doc.metadata.get('image_dimensions')}")
    print(f"Embedding length: {len(doc.metadata.get('embedding', []))}")
```

### Search for Similar Content

```python
from multimodal_rag.embeddings import GeminiEmbeddings

# Initialize the embedding model
embeddings = GeminiEmbeddings(api_key=os.getenv("GOOGLE_API_KEY"))

# Get embedding for a query
query = "a picture of a cat"
query_embedding = embeddings.embed_query(query)

# Search in Pinecone (pseudo-code, adjust based on your Pinecone client)
results = pinecone_index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

for match in results.matches:
    print(f"Score: {match.score}")
    print(f"Content: {match.metadata.get('text', '')}")
```

## Advanced Usage

### Custom Document Processing

You can customize how different file types are processed by extending the `DocumentProcessor` class:

```python
class CustomDocumentProcessor(DocumentProcessor):
    def _process_custom_format(self, file_path: str, metadata: dict) -> List[Document]:
        # Your custom processing logic here
        pass
```

### Batch Processing

For processing multiple files in parallel:

```python
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_file(file_path):
    try:
        return processor.process_file(file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

files_to_process = ["file1.pdf", "image1.jpg", "document.docx"]
all_documents = []

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(tqdm(
        executor.map(process_file, files_to_process),
        total=len(files_to_process)
    ))
    
for docs in results:
    all_documents.extend(docs)
```

## Command Line Interface

You can also use the provided CLI to process documents:

```bash
# Process a single file
python -m multimodal_rag.cli process path/to/your/file.pdf

# Process all files in a directory
python -m multimodal_rag.cli process-dir path/to/documents/

# Search for similar content
python -m multimodal_rag.cli search "your query here"
```

## Performance Considerations

- **Image Processing**: Large images are automatically resized to 1024x1024 pixels to reduce memory usage
- **Batch Processing**: Process multiple files in parallel for better performance
- **Caching**: Embeddings are cached to avoid redundant computations
- **Error Handling**: The system is designed to be resilient to failures in individual document processing

## Troubleshooting

### Common Issues

1. **Tesseract Not Found**:
   - Ensure Tesseract is installed and in your system PATH
   - Or set the `TESSERACT_CMD` environment variable to the full path of the Tesseract executable

2. **API Rate Limits**:
   - The Gemini API has rate limits. If you hit them, consider adding delays between requests
   - Implement retry logic for transient failures

3. **Memory Issues with Large Files**:
   - For very large documents, consider splitting them into smaller chunks before processing
   - Increase the chunk size in the `DocumentProcessor` if needed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Gemini for the powerful multimodal embeddings
- Pinecone for the vector database
- The open-source community for the various libraries used in this project
   
   processor = DocumentProcessor()
   elements = processor.process_pdf("path/to/your/document.pdf")
   ```

2. **Initialize RAG System**:
   ```python
   from multimodal_rag.rag_system import MultimodalRAG
   
   rag = MultimodalRAG()
   rag.initialize()
   ```

3. **Query the System**:
   ```python
   query = "What are the key financial metrics from the document?"
   response = rag.query(query)
   print(response)
   ```

## Project Structure

```
multimodal_rag/
├── examples/                  # Example usage scripts and notebooks
│   └── example_usage.py       # Demo script showing how to use the RAG system
│
├── multimodal_rag/           # Main package source code
│   ├── __init__.py           # Package initialization
│   ├── cli.py                # Command-line interface
│   ├── config.py             # Configuration settings and environment variables
│   ├── document_processor.py # Handles document parsing and processing
│   ├── embeddings.py         # Manages text embeddings (local and cloud)
│   ├── rag_system.py         # Core RAG system implementation
│   └── utils.py              # Utility functions
│
├── .env.example              # Template for environment variables
├── .gitignore               # Specifies intentionally untracked files
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
└── setup.py                # Package installation script
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Cloud Platform
- LangChain
- Vertex AI
