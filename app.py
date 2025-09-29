"""
Streamlit UI for Multimodal RAG System with Mem0 Memory
"""
import os
import time
from dotenv import load_dotenv
load_dotenv()
import tempfile
import logging
import sys

# Configure logging to show debug output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional

from multimodal_rag.rag_system import MultimodalRAG
from multimodal_rag.memory_manager import MemoryManager
from multimodal_rag.config import (
    DATA_DIR, MODEL_NAME, TOKEN_LIMIT, CHUNK_SIZE, CHUNK_OVERLAP
)

# Set page config
st.set_page_config(
    page_title="Multimodal RAG Chat",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'memory_manager' not in st.session_state:
    st.session_state.memory_manager = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Model settings
    st.subheader("Model Settings")
    model_name = st.selectbox(
        "Model",
        ["gemini-2.0-flash", "gemini-1.5-pro"],
        index=0
    )
    
    # RAG settings
    st.subheader("RAG Settings")
    chunk_size = st.slider("Chunk Size", 500, 4000, CHUNK_SIZE, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 1000, CHUNK_OVERLAP, 50)
    
    # Memory settings
    st.subheader("Memory Settings")
    use_memory = st.toggle("Enable Memory", value=True)
    memory_collection = st.text_input("Memory Collection", "multimodal_rag_chat")
    
    # Initialize/Reset button
    if st.button("Initialize/Reset System"):
        with st.spinner("Initializing RAG system..."):
            try:
                # Initialize RAG system with Pinecone
                st.session_state.rag_system = MultimodalRAG(
                    model_name=model_name,
                    token_limit=TOKEN_LIMIT,
                    vector_store_type='pinecone',
                    pinecone_api_key=os.getenv('PINECONE_API_KEY'),
                    pinecone_index_name='multimodal-rag',
                    pinecone_environment='gcp-starter'
                )
                
                # Initialize memory manager if enabled
                if use_memory:
                    # Get the MEM0_API_KEY from environment
                    mem0_api_key = os.environ.get("MEM0_API_KEY")
                    if not mem0_api_key:
                        st.warning("MEM0_API_KEY environment variable not set. Memory features will be disabled.")
                        use_memory = False
                    else:
                        try:
                            # Import MemoryClient
                            from mem0 import MemoryClient
                            
                            # Initialize MemoryClient with the API key
                            st.session_state.memory_manager = MemoryClient(api_key=mem0_api_key)
                            
                            # Store user ID in session state if not already set
                            if "user_id" not in st.session_state:
                                st.session_state.user_id = "default_user"
                            
                            # Test the connection with a simple operation
                            test_result = st.session_state.memory_manager.add(
                                messages=[{"role": "system", "content": "Memory initialized successfully"}],
                                user_id=st.session_state.user_id
                            )
                            
                            # Check if test_result is not None and has a 'results' key
                            if test_result is not None and 'results' in test_result:
                                st.session_state.memory_initialized = True
                                logger.info("Memory system initialized successfully")
                                logger.debug(f"Memory test result: {test_result}")
                            else:
                                # If we get here, the test failed
                                logger.warning(f"Memory test returned unexpected response: {test_result}")
                                # Still continue with memory enabled, as the API might work even if test is flaky
                                st.session_state.memory_initialized = True
                                st.warning("Memory test was inconclusive, but continuing with memory enabled")
                            
                        except ImportError as ie:
                            st.error(f"Failed to import mem0 package: {str(ie)}")
                            logger.error(f"Failed to import mem0 package: {str(ie)}")
                            use_memory = False
                        except Exception as e:
                            st.error(f"Failed to initialize memory: {str(e)}")
                            logger.error(f"Memory initialization failed: {str(e)}", exc_info=True)
                            use_memory = False
                
                st.success("RAG system initialized successfully!")
                
            except Exception as e:
                st.error(f"Failed to initialize RAG system: {str(e)}")
                logger.error(f"RAG system initialization failed: {str(e)}", exc_info=True)

# Main content
st.title("🤖 Multimodal RAG Chat")
st.caption("Upload documents and chat with them using the power of multimodal RAG!")

# File uploader
st.subheader("📂 Upload Documents")
uploaded_files = st.file_uploader(
    "Upload documents, presentations, spreadsheets, or images",
    type=[
        # Documents
        "pdf", "txt", "md", "html", "rtf",
        # Office Documents
        "docx", "doc", "odt", "ott",
        # Spreadsheets
        "xlsx", "xls", "xlsm", "xlsb", "ods", "ots",
        # Presentations
        "pptx", "ppt", "odp", "otp",
        # Images
        "png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp",
        # Data
        "csv", "tsv", "json", "xml", "yaml", "yml"
    ],
    accept_multiple_files=True,
    help="""
    Supported formats:
    - Documents: PDF, TXT, MD, HTML, DOCX, DOC, ODT
    - Spreadsheets: XLSX, XLS, XLSM, ODS
    - Presentations: PPTX, PPT, ODP
    - Images: PNG, JPG, JPEG, TIFF, BMP, GIF, WEBP
    - Data: CSV, TSV, JSON, XML, YAML
    """
)

# Process uploaded files
if uploaded_files and st.button("Process Documents"):
    if not st.session_state.rag_system:
        st.warning("Please initialize the RAG system first using the sidebar.")
    else:
        with st.spinner("Processing documents..."):
            temp_dir = Path(tempfile.mkdtemp())
            processed_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    file_path = temp_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Update status
                    status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Add the document to the RAG system
                    result = st.session_state.rag_system.add_documents(
                        str(file_path),
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    # Store uploaded file info with processing results
                    st.session_state.uploaded_files.append({
                        "name": uploaded_file.name,
                        "type": uploaded_file.type,
                        "size": len(uploaded_file.getvalue()),
                        "chunks": result.get("texts", 0) + result.get("tables", 0) + result.get("images", 0)
                    })
                    
                    processed_count += 1
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name if 'uploaded_file' in locals() else 'file'}: {str(e)}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.empty()
            progress_bar.empty()
            
            if processed_count > 0:
                st.success(f"Successfully processed {processed_count} out of {len(uploaded_files)} documents!")
            else:
                st.warning("No documents were processed successfully. Please check the error messages above.")

# Display uploaded files
if st.session_state.uploaded_files:
    st.subheader("📝 Uploaded Documents")
    
    # Create a table for better organization
    cols = st.columns([4, 2, 2, 2])
    with cols[0]: st.markdown("**File Name**")
    with cols[1]: st.markdown("**Type**")
    with cols[2]: st.markdown("**Size**")
    with cols[3]: st.markdown("**Chunks**")
    
    for file_info in st.session_state.uploaded_files:
        cols = st.columns([4, 2, 2, 2])
        with cols[0]: st.text(file_info['name'])
        with cols[1]: st.text(file_info['type'].split('/')[-1].upper())
        with cols[2]: st.text(f"{file_info['size']/1024:.1f} KB")
        with cols[3]: st.text(file_info.get('chunks', 'N/A'))
    
    # Add a button to clear all uploaded files
    if st.button("Clear All Documents", type="secondary"):
        st.session_state.uploaded_files = []
        st.session_state.chat_history = []
        st.rerun()

# Chat interface
st.subheader("💬 Chat with your documents")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Show citations if available
        if message.get("citations"):
            with st.expander(f"📚 Sources ({len(message['citations'])})", expanded=False):
                for i, citation in enumerate(message["citations"], 1):
                    with st.container():
                        st.markdown(f"### Source {i}")
                        
                        # Display source information
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            # Show icon based on source type
                            if citation.get('type') == 'image':
                                st.markdown("🖼️ **Image Source**")
                                if 'image_path' in citation and citation['image_path']:
                                    try:
                                        st.image(citation['image_path'], use_column_width=True)
                                    except Exception as e:
                                        st.warning("Could not load image")
                            else:
                                st.markdown("📄 **Text Source**")
                        
                        with col2:
                            # Display source metadata
                            st.markdown(f"**{citation.get('display_text', 'Source')}**")
                            
                            # Show content preview for text sources
                            if citation.get('type') == 'text' and 'content' in citation:
                                with st.expander("View content"):
                                    st.markdown(citation['content'])
                            
                            # Add a download/view button for the source
                            if 'source' in citation and citation['source']:
                                source_path = Path(citation['source'])
                                if source_path.exists():
                                    with open(source_path, "rb") as f:
                                        # Create a unique key for each download button using file name, index, and timestamp
                                        import time
                                        timestamp = int(time.time() * 1000)  # Current time in milliseconds
                                        button_key = f"download_{source_path.stem}_{i}_{timestamp}"
                                        st.download_button(
                                            label="View Source",
                                            data=f,
                                            file_name=source_path.name,
                                            key=button_key,
                                            mime="application/octet-stream",
                                            use_container_width=True
                                        )
                        
                        st.markdown("---")  # Divider between citations

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.rag_system:
        st.warning("Please initialize the RAG system first using the sidebar.")
    elif not st.session_state.uploaded_files:
        st.warning("Please upload and process some documents first.")
    else:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from RAG system
                    response = st.session_state.rag_system.query(
                        question=prompt,
                        chat_history=st.session_state.chat_history,
                        return_context=True
                    )
                    
                    # Create a placeholder for the response
                    response_placeholder = st.empty()
                    
                    # Format citations for display
                    formatted_citations = []
                    for citation in response.get("citations", []):
                        formatted_citation = {
                            "type": citation.get("type", "text"),
                            "display_text": citation.get("display_text", ""),
                            "source": citation.get("source", ""),
                            "page": citation.get("page", ""),
                            "content": citation.get("content", ""),
                            "image_path": citation.get("image_path", "")
                        }
                        formatted_citations.append(formatted_citation)
                    
                    # Add to chat history with formatted citations
                    assistant_message = {
                        "role": "assistant",
                        "content": response["answer"],
                        "citations": formatted_citations,
                        "timestamp": time.time()
                    }
                    st.session_state.chat_history.append(assistant_message)
                    
                    # Display the response with citations
                    with response_placeholder.container():
                        st.write(response["answer"])
                        
                        # Display citations if available
                        if formatted_citations:
                            with st.expander(f"📚 Sources ({len(formatted_citations)})", expanded=True):
                                for i, citation in enumerate(formatted_citations, 1):
                                    with st.container():
                                        st.markdown(f"### Source {i}")
                                        
                                        # Display source information
                                        col1, col2 = st.columns([1, 3])
                                        
                                        with col1:
                                            # Show icon based on source type
                                            if citation.get('type') == 'image':
                                                st.markdown("🖼️ **Image Source")
                                                if citation.get('image_path'):
                                                    try:
                                                        st.image(citation['image_path'], use_column_width=True)
                                                    except Exception as e:
                                                        st.warning("Could not load image")
                                            else:
                                                st.markdown("📄 **Text Source")
                                        
                                        with col2:
                                            # Display source metadata
                                            if citation.get('display_text'):
                                                st.markdown(f"**{citation['display_text']}**")
                                            
                                            # Show content preview for text sources
                                            if citation.get('content'):
                                                with st.expander("View content"):
                                                    st.markdown(citation['content'])
                                            
                                            # Add a download/view button for the source
                                            if citation.get('source'):
                                                source_path = Path(citation['source'])
                                                if source_path.exists():
                                                    with open(source_path, "rb") as f:
                                                        button_key = f"download_{source_path.stem}_{i}_{int(time.time()*1000)}"
                                                        st.download_button(
                                                            label="View Source",
                                                            data=f,
                                                            file_name=source_path.name,
                                                            key=button_key,
                                                            use_container_width=True
                                                        )
                    
                    # Update memory if enabled
                    if use_memory and hasattr(st.session_state, 'memory_manager') and st.session_state.memory_manager:
                        try:
                            # Prepare messages for Mem0
                            messages = [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": response["answer"]}
                            ]
                            
                            # Prepare metadata
                            memory_metadata = {
                                "source": "chat",
                                "user_input": prompt,
                                "ai_response": response["answer"]
                            }
                            
                            # Add citations to metadata if they exist
                            if "citations" in response and response["citations"]:
                                try:
                                    # Convert citations to a serializable format if needed
                                    memory_metadata["citations"] = response["citations"]
                                except Exception as e:
                                    logger.warning(f"Could not add citations to memory: {str(e)}")
                            
                            # Get user ID
                            user_id = st.session_state.get("user_id", "default_user")
                            
                            # Add to memory using the MemoryClient
                            try:
                                result = st.session_state.memory_manager.add(
                                    messages=messages,
                                    user_id=user_id,
                                    metadata=memory_metadata,
                                    infer=True,  # Let Mem0 infer and extract entities
                                    output_format='v1.1'  # Use the latest version
                                )
                                
                                # In v1.1, a successful addition returns results with the memory ID
                                if result and 'results' in result and isinstance(result['results'], list):
                                    logger.info(f"Successfully added to memory. Result: {result}")
                                else:
                                    logger.warning(f"Memory addition returned unexpected response: {result}")
                                    
                            except Exception as e:
                                logger.error(f"Error adding to memory: {str(e)}", exc_info=True)
                                
                        except Exception as e:
                            error_msg = f"Failed to handle memory: {str(e)}"
                            logger.error(error_msg, exc_info=True)
                            st.warning("Failed to save conversation to memory")
                
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    st.error("An error occurred while generating the response. Please try again.")

# Debug Pinecone environment
logger.info("="*50)
logger.info("Starting Multimodal RAG Application")
logger.info("="*50)
logger.info(f"Python version: {sys.version}")
logger.info(f"Pinecone API Key: {'*' * 8}{os.getenv('PINECONE_API_KEY', '')[-4:] if os.getenv('PINECONE_API_KEY') else 'Not set'}")
logger.info(f"Pinecone Environment: {os.getenv('PINECONE_ENVIRONMENT', 'Not set')}")
logger.info("="*50 + "\n")

# Add some CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stSidebar {
        padding: 2rem 1rem;
    }
    /* Style for citations */
    .citation {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4a90e2;
    }
    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .citation-content {
        font-size: 0.9rem;
        color: #495057;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)