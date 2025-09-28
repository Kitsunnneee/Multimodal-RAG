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
from multimodal_rag.audio_processor import initialize_components, process_audio_query
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
if 'audio_transcriptions' not in st.session_state:
    st.session_state.audio_transcriptions = {}
if 'audio_processor_initialized' not in st.session_state:
    st.session_state.audio_processor_initialized = False

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
    user_id = st.text_input("User ID", "streamlit_user", help="Unique identifier for your session")
    
    # Initialize/Reset button
    if st.button("Initialize/Reset System"):
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = MultimodalRAG(
                model_name=model_name,
                token_limit=TOKEN_LIMIT
            )
            
            if use_memory:
                try:
                    st.session_state.memory_manager = MemoryManager(
                        collection_name=memory_collection,
                        use_hosted=False,  # Set to True if using hosted Mem0
                        user_id=user_id  # Use the user-provided user_id
                    )
                    st.success("✓ Memory manager initialized!")
                except Exception as e:
                    st.warning(f"⚠ Memory manager initialization failed: {e}")
                    st.info("Continuing without memory functionality.")
                    st.session_state.memory_manager = None
            
            # Initialize audio processor components
            if not st.session_state.audio_processor_initialized:
                with st.spinner("Initializing audio processor..."):
                    try:
                        initialize_components()
                        st.session_state.audio_processor_initialized = True
                        st.success("✓ Audio processor initialized!")
                    except Exception as e:
                        st.warning(f"⚠ Audio processor initialization failed: {e}")
                        st.info("Audio files will still be uploaded but transcription may not work.")
            
            st.success("RAG system initialized successfully!")

# Main content
st.title("🤖 Multimodal RAG Chat")
st.caption("Upload documents and chat with them using the power of multimodal RAG!")

# File uploader
st.subheader("📂 Upload Documents & Audio Files")
uploaded_files = st.file_uploader(
    "Upload PDFs, images, text files, or audio files",
    type=["pdf", "png", "jpg", "jpeg", "txt", "csv", "xlsx", "wav", "mp3", "m4a", "flac", "ogg"],
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files and st.button("Process Documents & Audio"):
    if not st.session_state.rag_system:
        st.warning("Please initialize the RAG system first using the sidebar.")
    else:
        with st.spinner("Processing files..."):
            temp_dir = Path(tempfile.mkdtemp())
            processed_docs = 0
            processed_audio = 0
            
            for uploaded_file in uploaded_files:
                file_path = temp_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Check if file is audio
                audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
                is_audio = any(uploaded_file.name.lower().endswith(ext) for ext in audio_extensions)
                
                if is_audio:
                    # Process audio file
                    st.write(f"🎧 Processing audio file: {uploaded_file.name}")
                    try:
                        if st.session_state.audio_processor_initialized:
                            with st.spinner(f"Transcribing {uploaded_file.name}..."):
                                # Process audio with a generic query to get transcription
                                st.write(f"🔍 Calling process_audio_query for {uploaded_file.name}")
                                result = process_audio_query(str(file_path), "What is the content of this audio?")
                                transcription = result.get('transcription', '')
                                st.write(f"📝 Got transcription: {len(transcription)} characters")
                                
                                # Store transcription
                                st.session_state.audio_transcriptions[uploaded_file.name] = {
                                    'transcription': transcription,
                                    'file_type': uploaded_file.type,
                                    'file_size': len(uploaded_file.getvalue())
                                }
                                st.write(f"💾 Stored transcription for {uploaded_file.name}")
                                
                                # Add transcription as a document to the RAG system
                                # Create a temporary text file with the transcription
                                transcript_path = temp_dir / f"{uploaded_file.name}_transcript.txt"
                                with open(transcript_path, "w") as tf:
                                    tf.write(f"Audio transcription from {uploaded_file.name}:\n\n{transcription}")
                                
                                st.session_state.rag_system.add_documents(
                                    str(transcript_path),
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap
                                )
                                
                                processed_audio += 1
                                st.success(f"✓ Audio transcribed: {uploaded_file.name}")
                        else:
                            st.warning(f"⚠ Audio processor not initialized. Skipping {uploaded_file.name}")
                    
                    except Exception as e:
                        st.error(f"Error processing audio {uploaded_file.name}: {str(e)}")
                
                else:
                    # Process as document
                    try:
                        # Add the document to the RAG system
                        result = st.session_state.rag_system.add_documents(
                            str(file_path),
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        processed_docs += 1
                        
                    except Exception as e:
                        st.error(f"Error processing document {uploaded_file.name}: {str(e)}")
                
                # Store uploaded file info
                st.session_state.uploaded_files.append({
                    "name": uploaded_file.name,
                    "type": uploaded_file.type,
                    "size": len(uploaded_file.getvalue()),
                    "is_audio": is_audio
                })
            
            # Show results
            if processed_docs > 0 or processed_audio > 0:
                st.success(f"Processed {processed_docs} documents and {processed_audio} audio files!")
            else:
                st.warning("No files were processed successfully.")

# Debug section (temporary)
with st.expander("🔍 Debug Information"):
    st.write(f"**Uploaded files:** {len(st.session_state.uploaded_files)}")
    for f in st.session_state.uploaded_files:
        st.write(f"- {f['name']} ({f['type']}) - Audio: {f.get('is_audio', False)}")
    
    st.write(f"**Audio transcriptions:** {len(st.session_state.audio_transcriptions)}")
    for fname, data in st.session_state.audio_transcriptions.items():
        st.write(f"- {fname}: '{data.get('transcription', '')[:100]}...'")

# Display uploaded files
if st.session_state.uploaded_files:
    st.subheader("📝 Uploaded Files")
    
    # Separate documents and audio files
    docs = [f for f in st.session_state.uploaded_files if not f.get('is_audio', False)]
    audio_files = [f for f in st.session_state.uploaded_files if f.get('is_audio', False)]
    
    if docs:
        st.write("📄 **Documents:**")
        for file_info in docs:
            st.write(f"- {file_info['name']} ({file_info['type']}, {file_info['size']/1024:.1f} KB)")
    
    if audio_files:
        st.write("🎧 **Audio Files:**")
        for file_info in audio_files:
            st.write(f"- {file_info['name']} ({file_info['type']}, {file_info['size']/1024:.1f} KB)")
            
            # Show transcription if available
            if file_info['name'] in st.session_state.audio_transcriptions:
                transcription = st.session_state.audio_transcriptions[file_info['name']]['transcription']
                with st.expander(f"📝 View transcription of {file_info['name']}"):
                    st.text_area(
                        "Transcription:",
                        value=transcription,
                        height=100,
                        disabled=True,
                        key=f"transcript_{file_info['name']}"
                    )

# Audio Query Section
if st.session_state.audio_transcriptions:
    st.subheader("🎧 Audio Query Section")
    
    # Add info about the audio content
    with st.expander("ℹ️ Audio Content Summary"):
        st.write(f"🎧 **{len(st.session_state.audio_transcriptions)} audio file(s) processed**")
        for name, data in st.session_state.audio_transcriptions.items():
            word_count = len(data['transcription'].split())
            st.write(f"- **{name}**: ~{word_count} words transcribed")
    
    st.write("Ask specific questions about your uploaded audio files:")
    
    # Audio file selector
    audio_file_names = list(st.session_state.audio_transcriptions.keys())
    selected_audio = st.selectbox(
        "Select an audio file to query:",
        options=["All audio files"] + audio_file_names,
        key="audio_selector"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        audio_query = st.text_input(
            "Ask about the audio content:",
            placeholder="What are the main topics discussed in this audio?",
            key="audio_query_input"
        )
    with col2:
        if st.button("🎤 Query Audio", key="query_audio_btn"):
            if audio_query:
                if selected_audio == "All audio files":
                    # Query all transcriptions
                    combined_transcription = "\n\n".join([
                        f"From {name}: {data['transcription']}"
                        for name, data in st.session_state.audio_transcriptions.items()
                    ])
                    context_query = f"Based on these audio transcriptions: {combined_transcription}. {audio_query}"
                else:
                    # Query specific audio file
                    transcription = st.session_state.audio_transcriptions[selected_audio]['transcription']
                    context_query = f"Based on this audio transcription from {selected_audio}: '{transcription}'. {audio_query}"
                
                with st.spinner("Analyzing audio content..."):
                    try:
                        response = st.session_state.rag_system.query(
                            question=context_query,
                            chat_history=[],
                            return_context=True
                        )
                        st.success("✓ Audio analysis complete!")
                        st.write("💬 **Response:**")
                        st.write(response["answer"])
                    except Exception as e:
                        st.error(f"Error analyzing audio: {e}")
            else:
                st.warning("Please enter a query about the audio.")
    
    st.divider()

# Chat interface
st.subheader("💬 Chat with your documents & audio")

# Show helpful examples if audio files are available
if st.session_state.audio_transcriptions:
    with st.expander("💡 Example questions for audio content"):
        st.write("Try asking:")
        st.write("- What is the summary of the audio file?")
        st.write("- What are the main topics mentioned in the audio?")
        st.write("- What food items are mentioned in the audio?")
        st.write("- Analyze the content of the audio recording")

# Display chat history
for msg_idx, message in enumerate(st.session_state.chat_history):
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
                                with st.expander("View content", expanded=False):
                                    st.markdown(citation['content'])
                            
                            # Add a download/view button for the source
                            if 'source' in citation and citation['source']:
                                source_path = Path(citation['source'])
                                if source_path.exists():
                                    with open(source_path, "rb") as f:
                                        st.download_button(
                                            label="View Source",
                                            data=f,
                                            file_name=source_path.name,
                                            mime="application/octet-stream",
                                            use_container_width=True,
                                            key=f"download_btn_{msg_idx}_{i}_{source_path.name}"
                                        )
                        
                        st.markdown("---")  # Divider between citations

# Chat input
chat_placeholder = "Ask a question about your documents"
if st.session_state.audio_transcriptions:
    chat_placeholder += " or audio files"
chat_placeholder += "..."

if prompt := st.chat_input(chat_placeholder):
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
            # Show indicator if audio context is being used
            if st.session_state.audio_transcriptions:
                st.info(f"🎧 Including context from {len(st.session_state.audio_transcriptions)} audio file(s)")
            
            with st.spinner("Thinking..."):
                try:
                    # Debug: Check what audio transcriptions are available
                    st.write(f"🔍 Debug: Found {len(st.session_state.audio_transcriptions)} audio transcriptions")
                    for fname, data in st.session_state.audio_transcriptions.items():
                        st.write(f"📁 {fname}: {len(data.get('transcription', ''))} characters")
                    
                    # Enhance the prompt with audio context if available
                    enhanced_prompt = prompt
                    if st.session_state.audio_transcriptions:
                        # Add audio context to the prompt
                        audio_context = "\n\n--- Audio Content Available ---\n"
                        for filename, data in st.session_state.audio_transcriptions.items():
                            transcription = data.get('transcription', '')
                            audio_context += f"Audio file '{filename}': {transcription}\n\n"
                        audio_context += "--- End Audio Content ---\n\n"
                        enhanced_prompt = audio_context + "User question: " + prompt
                        st.write(f"📝 Enhanced prompt preview: {enhanced_prompt[:200]}...")
                    
                    # Get response from RAG system
                    response = st.session_state.rag_system.query(
                        question=enhanced_prompt,
                        chat_history=st.session_state.chat_history,
                        return_context=True
                    )
                    
                    # Display response
                    st.write(response["answer"])
                    
                    # Format citations for display
                    formatted_citations = []
                    for citation in response.get("citations", []):
                        formatted_citation = {
                            "type": citation.get("type", "text"),
                            "display_text": citation.get("display_text", ""),
                            "source": citation.get("source", ""),
                            "page": citation.get("page", "")
                        }
                        
                        # Add type-specific fields
                        if formatted_citation["type"] == "image":
                            formatted_citation["image_path"] = citation.get("image_path", "")
                        else:
                            formatted_citation["content"] = citation.get("content", "")
                        
                        formatted_citations.append(formatted_citation)
                    
                    # Add to chat history with formatted citations
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "citations": formatted_citations
                    })
                    
                    # Update memory if enabled
                    if use_memory and st.session_state.memory_manager:
                        try:
                            # Combine user input and AI response for memory
                            memory_content = f"User: {prompt}\nAssistant: {response['answer']}"
                            result = st.session_state.memory_manager.add_memory(
                                content=memory_content,
                                metadata={
                                    "source": "chat",
                                    "citations": response.get("citations", []),
                                    "user_input": prompt,
                                    "ai_response": response["answer"]
                                }
                            )
                            if result:
                                st.success("✓ Conversation saved to memory", icon="🧠")
                        except Exception as mem_error:
                            st.warning(f"⚠ Could not save to memory: {str(mem_error)[:100]}...")
                    
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
