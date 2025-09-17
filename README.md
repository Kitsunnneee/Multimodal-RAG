# Multimodal RAG with Gemini and Vertex AI

A Python implementation of a Multimodal Retrieval Augmented Generation (RAG) system using Google's Gemini model and Vertex AI Vector Search.

## Features

- Extract and process text, tables, and images from PDF documents
- Generate embeddings for multimodal content
- Perform semantic search across different content types
- Generate responses using Google's Gemini model
- Support for both text and image-based queries

## Prerequisites

- Python 3.9+
- Google Cloud account with Vertex AI and Vector Search enabled
- Service account with appropriate permissions

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

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your Google Cloud credentials and project details
   ```

## Configuration

Create a `.env` file in the project root with the following variables:

```
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=your-region
GCS_BUCKET=your-bucket-name
```

## Usage

1. **Process Documents**:
   ```python
   from multimodal_rag.document_processor import DocumentProcessor
   
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
