"""
Streamlit UI for Multimodal RAG System with Mem0 Memory
"""
import os
import tempfile
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
            st.session_state.rag_system = MultimodalRAG(
                model_name=model_name,
                token_limit=TOKEN_LIMIT
            )
            
            if use_memory:
                st.session_state.memory_manager = MemoryManager(
                    collection_name=memory_collection,
                    use_hosted=False  # Set to True if using hosted Mem0
                )
            
            st.success("RAG system initialized successfully!")

# Main content
st.title("🤖 Multimodal RAG Chat")
st.caption("Upload documents and chat with them using the power of multimodal RAG!")

# File uploader
st.subheader("📂 Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDFs, images, or text files",
    type=["pdf", "png", "jpg", "jpeg", "txt", "csv", "xlsx"],
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files and st.button("Process Documents"):
    if not st.session_state.rag_system:
        st.warning("Please initialize the RAG system first using the sidebar.")
    else:
        with st.spinner("Processing documents..."):
            temp_dir = Path(tempfile.mkdtemp())
            
            for uploaded_file in uploaded_files:
                file_path = temp_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
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
                        "size": len(uploaded_file.getvalue())
                    })
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            st.success(f"Processed {len(uploaded_files)} documents!")

# Display uploaded files
if st.session_state.uploaded_files:
    st.subheader("📝 Uploaded Documents")
    for file_info in st.session_state.uploaded_files:
        st.write(f"- {file_info['name']} ({file_info['type']}, {file_info['size']/1024:.1f} KB)")

# Chat interface
st.subheader("💬 Chat with your documents")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Show citations if available
        if message.get("citations"):
            with st.expander("View Citations"):
                for i, citation in enumerate(message["citations"]):
                    st.write(f"**Citation {i+1}**")
                    if "image" in citation:
                        st.image(citation["image"])
                    if "text" in citation:
                        st.text(citation["text"][:500] + "...")

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
                    
                    # Display response
                    st.write(response["answer"])
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "citations": response.get("citations", [])
                    })
                    
                    # Update memory if enabled
                    if use_memory and st.session_state.memory_manager:
                        # Combine user input and AI response for memory
                        memory_content = f"User: {prompt}\nAssistant: {response['answer']}"
                        st.session_state.memory_manager.add_memory(
                            content=memory_content,
                            metadata={
                                "source": "chat",
                                "citations": response.get("citations", []),
                                "user_input": prompt,
                                "ai_response": response["answer"]
                            }
                        )
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

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
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)
