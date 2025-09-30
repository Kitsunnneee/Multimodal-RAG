"""
Streamlit UI for Multimodal RAG System with Mem0 Memory
"""
import os
import tempfile
import logging

# Set up logging
logger = logging.getLogger(__name__)
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional

from multimodal_rag.rag_system import MultimodalRAG
from multimodal_rag.memory_manager import MemoryManager
from multimodal_rag.audio_processor import initialize_components, process_audio_query
from multimodal_rag.video_processor import initialize_video_components, process_video_query
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
if 'video_analyses' not in st.session_state:
    st.session_state.video_analyses = {}
if 'video_processor_initialized' not in st.session_state:
    st.session_state.video_processor_initialized = False

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
            try:
                # Initialize RAG system
                st.session_state.rag_system = MultimodalRAG(
                    model_name=model_name,
                    token_limit=TOKEN_LIMIT
                )
                
                # Initialize memory manager if enabled
                if use_memory:
                    # Try MEM0 first if API key is available
                    mem0_api_key = os.environ.get("MEM0_API_KEY")
                    if mem0_api_key:
                        try:
                            from mem0 import MemoryClient
                            st.session_state.memory_manager = MemoryClient(api_key=mem0_api_key)
                            
                            # Store user ID in session state
                            st.session_state.user_id = user_id or "default_user"
                            
                            # Test the connection
                            test_result = st.session_state.memory_manager.add(
                                messages=[{"role": "system", "content": "Memory initialized successfully"}],
                                user_id=st.session_state.user_id
                            )
                            
                            if test_result is not None and 'results' in test_result:
                                st.session_state.memory_initialized = True
                                logger.info("Memory system initialized successfully with MEM0")
                                st.success("✓ Memory system initialized (MEM0)")
                            else:
                                logger.warning("MEM0 test returned unexpected response, falling back to local memory")
                                st.session_state.memory_manager = MemoryManager(
                                    collection_name=memory_collection,
                                    use_hosted=False,
                                    user_id=user_id
                                )
                                st.success("✓ Memory system initialized (Local)")
                                
                        except ImportError:
                            logger.warning("MEM0 package not available, falling back to local memory")
                            st.session_state.memory_manager = MemoryManager(
                                collection_name=memory_collection,
                                use_hosted=False,
                                user_id=user_id
                            )
                            st.success("✓ Memory system initialized (Local)")
                    else:
                        # Use local memory manager if MEM0_API_KEY not set
                        st.session_state.memory_manager = MemoryManager(
                            collection_name=memory_collection,
                            use_hosted=False,
                            user_id=user_id
                        )
                        st.success("✓ Memory system initialized (Local)")
                
                # Initialize audio processor components
                if not st.session_state.audio_processor_initialized:
                    with st.spinner("Initializing audio processor..."):
                        try:
                            initialize_components()
                            st.session_state.audio_processor_initialized = True
                            st.success("✓ Audio processor initialized")
                            logger.info("Audio processor initialized successfully")
                        except Exception as e:
                            st.warning(f"⚠ Audio processor initialization failed: {e}")
                            logger.warning(f"Audio processor initialization failed: {e}")
                
                # Initialize video processor components
                if not st.session_state.video_processor_initialized:
                    with st.spinner("Initializing video processor..."):
                        try:
                            initialize_video_components()
                            st.session_state.video_processor_initialized = True
                            st.success("✓ Video processor initialized")
                            logger.info("Video processor initialized successfully")
                        except Exception as e:
                            st.warning(f"⚠ Video processor initialization failed: {e}")
                            logger.warning(f"Video processor initialization failed: {e}")
                
                st.success("✅ RAG system initialized successfully!")
                
            except Exception as e:
                st.error(f"Failed to initialize RAG system: {str(e)}")
                logger.error(f"RAG system initialization failed: {str(e)}", exc_info=True)

# Main content
st.title("🤖 Multimodal RAG Chat")
st.caption("Upload documents and chat with them using the power of multimodal RAG!")

# File uploader
st.subheader("📂 Upload Documents, Audio & Video Files")
uploaded_files = st.file_uploader(
    "Upload documents, presentations, spreadsheets, images, audio, or video files",
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
        # Audio
        "wav", "mp3", "m4a", "flac", "ogg",
        # Video
        "mp4", "avi", "mov", "mkv", "wmv", "flv",
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
    - Audio: WAV, MP3, M4A, FLAC, OGG
    - Video: MP4, AVI, MOV, MKV, WMV, FLV
    - Data: CSV, TSV, JSON, XML, YAML
    """
)

# Process uploaded files
if uploaded_files and st.button("Process Documents, Audio & Video"):
    if not st.session_state.rag_system:
        st.warning("Please initialize the RAG system first using the sidebar.")
    else:
        with st.spinner("Processing files..."):
            temp_dir = Path(tempfile.mkdtemp())
            processed_docs = 0
            processed_audio = 0
            processed_video = 0
            
            for uploaded_file in uploaded_files:
                file_path = temp_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Check file type
                audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
                is_audio = any(uploaded_file.name.lower().endswith(ext) for ext in audio_extensions)
                is_video = any(uploaded_file.name.lower().endswith(ext) for ext in video_extensions)
                
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
                
                elif is_video:
                    # Process video file
                    st.write(f"🎬 Processing video file: {uploaded_file.name}")
                    try:
                        if st.session_state.video_processor_initialized:
                            with st.spinner(f"Processing video {uploaded_file.name}..."):
                                st.write(f"🔍 Calling process_video_query for {uploaded_file.name}")
                                result = process_video_query(str(file_path), "Analyze this video content")
                                
                                # Store video analysis
                                st.session_state.video_analyses[uploaded_file.name] = {
                                    'analysis': result.get('answer', ''),
                                    'visual_analysis': result.get('visual_analysis', ''),
                                    'audio_transcription': result.get('audio_transcription', ''),
                                    'metadata': result.get('metadata', {}),
                                    'frames': result.get('frames', []),
                                    'file_type': uploaded_file.type,
                                    'file_size': len(uploaded_file.getvalue())
                                }
                                st.write(f"💾 Stored video analysis for {uploaded_file.name}")
                                
                                # Add video analysis as a document to the RAG system
                                video_content = f"Video analysis from {uploaded_file.name}:\n\n{result.get('combined_analysis', '')}"
                                video_text_path = temp_dir / f"{uploaded_file.name}_analysis.txt"
                                with open(video_text_path, "w") as vf:
                                    vf.write(video_content)
                                
                                st.session_state.rag_system.add_documents(
                                    str(video_text_path),
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap
                                )
                                
                                processed_video += 1
                                st.success(f"✓ Video processed: {uploaded_file.name}")
                        else:
                            st.warning(f"⚠ Video processor not initialized. Skipping {uploaded_file.name}")
                    
                    except Exception as e:
                        st.error(f"Error processing video {uploaded_file.name}: {str(e)}")
                
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
                    "is_audio": is_audio,
                    "is_video": is_video
                })
            
            # Show results
            total_processed = processed_docs + processed_audio + processed_video
            if total_processed > 0:
                result_parts = []
                if processed_docs > 0:
                    result_parts.append(f"{processed_docs} document{'s' if processed_docs != 1 else ''}")
                if processed_audio > 0:
                    result_parts.append(f"{processed_audio} audio file{'s' if processed_audio != 1 else ''}")
                if processed_video > 0:
                    result_parts.append(f"{processed_video} video file{'s' if processed_video != 1 else ''}")
                
                st.success(f"Processed {', '.join(result_parts)}!")
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
    
    st.write(f"**Video analyses:** {len(st.session_state.video_analyses)}")
    for fname, data in st.session_state.video_analyses.items():
        st.write(f"- {fname}: '{data.get('analysis', '')[:100]}...'")

# Display uploaded files
if st.session_state.uploaded_files:
    st.subheader("📝 Uploaded Files")
    
    # Separate documents, audio, and video files
    docs = [f for f in st.session_state.uploaded_files if not f.get('is_audio', False) and not f.get('is_video', False)]
    audio_files = [f for f in st.session_state.uploaded_files if f.get('is_audio', False)]
    video_files = [f for f in st.session_state.uploaded_files if f.get('is_video', False)]
    
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
    
    if video_files:
        st.write("🎬 **Video Files:**")
        for file_info in video_files:
            st.write(f"- {file_info['name']} ({file_info['type']}, {file_info['size']/1024:.1f} KB)")
            
            # Show video analysis if available
            if file_info['name'] in st.session_state.video_analyses:
                analysis_data = st.session_state.video_analyses[file_info['name']]
                
                with st.expander(f"🎬 View analysis of {file_info['name']}"):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write("**Video Analysis:**")
                        st.text_area(
                            "Analysis:",
                            value=analysis_data.get('analysis', 'No analysis available'),
                            height=150,
                            disabled=True,
                            key=f"video_analysis_{file_info['name']}"
                        )
                        
                        # Show metadata
                        if analysis_data.get('metadata'):
                            metadata = analysis_data['metadata']
                            st.write("**Video Info:**")
                            duration = metadata.get('duration', 0)
                            size = metadata.get('size', (0, 0))
                            fps = metadata.get('fps', 0)
                            st.write(f"Duration: {duration:.1f}s")
                            st.write(f"Resolution: {size[0]}x{size[1]}")
                            st.write(f"FPS: {fps:.1f}")
                    
                    with col2:
                        # Show audio transcription if available
                        if analysis_data.get('audio_transcription'):
                            st.write("**Audio from Video:**")
                            st.text_area(
                                "Audio Transcription:",
                                value=analysis_data['audio_transcription'],
                                height=100,
                                disabled=True,
                                key=f"video_audio_{file_info['name']}"
                            )
                        
                        # Show frame count
                        frames_count = len(analysis_data.get('frames', []))
                        if frames_count > 0:
                            st.write(f"**Extracted {frames_count} frames for analysis**")

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

# Video Query Section
if st.session_state.video_analyses:
    st.subheader("🎬 Video Query Section")
    
    # Add info about the video content
    with st.expander("ℹ️ Video Content Summary"):
        st.write(f"🎬 **{len(st.session_state.video_analyses)} video file(s) processed**")
        for name, data in st.session_state.video_analyses.items():
            metadata = data.get('metadata', {})
            duration = metadata.get('duration', 0)
            frames_count = len(data.get('frames', []))
            st.write(f"- **{name}**: {duration:.1f}s, {frames_count} frames analyzed")
    
    st.write("Ask specific questions about your uploaded video files:")
    
    # Video file selector
    video_file_names = list(st.session_state.video_analyses.keys())
    selected_video = st.selectbox(
        "Select a video file to query:",
        options=["All video files"] + video_file_names,
        key="video_selector"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        video_query = st.text_input(
            "Ask about the video content:",
            placeholder="What happens in this video? Who are the people? What objects are visible?",
            key="video_query_input"
        )
    with col2:
        if st.button("🎬 Query Video", key="query_video_btn"):
            if video_query:
                if selected_video == "All video files":
                    # Query all video analyses
                    combined_analysis = "\n\n".join([
                        f"From {name}: {data.get('analysis', '')}"
                        for name, data in st.session_state.video_analyses.items()
                    ])
                    context_query = f"Based on these video analyses: {combined_analysis}. {video_query}"
                else:
                    # Query specific video file
                    analysis = st.session_state.video_analyses[selected_video].get('analysis', '')
                    context_query = f"Based on this video analysis from {selected_video}: '{analysis}'. {video_query}"
                
                with st.spinner("Analyzing video content..."):
                    try:
                        response = st.session_state.rag_system.query(
                            question=context_query,
                            chat_history=[],
                            return_context=True
                        )
                        st.success("✓ Video analysis complete!")
                        st.write("💬 **Response:**")
                        st.write(response["answer"])
                        
                        # Store the analysis in session state
                        if "video_analyses" not in st.session_state:
                            st.session_state.video_analyses = {}
                        st.session_state.video_analyses[selected_video] = {
                            "analysis": response["answer"],
                            "file_type": st.session_state.video_analyses[selected_video]["file_type"],
                            "timestamp": datetime.now().isoformat()
                        }
                        
                    except Exception as e:
                        st.error(f"Error analyzing video: {e}")
            else:
                st.warning("Please enter a query about the video.")
            
            st.divider()
            
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
        # Clear audio and video states as well
        if 'audio_transcriptions' in st.session_state:
            st.session_state.audio_transcriptions = {}
        if 'video_analyses' in st.session_state:
            st.session_state.video_analyses = {}
        st.rerun()

# Chat interface
st.subheader("💬 Chat with your documents, audio & video")

# Show helpful examples for multimedia content
if st.session_state.audio_transcriptions or st.session_state.video_analyses:
    with st.expander("💡 Example questions for multimedia content"):
        st.write("Try asking:")
        
        if st.session_state.audio_transcriptions:
            st.write("**Audio queries:**")
            st.write("- What is the summary of the audio file?")
            st.write("- What are the main topics mentioned in the audio?")
            st.write("- Analyze the content of the audio recording")
        
        if st.session_state.video_analyses:
            st.write("**Video queries:**")
            st.write("- What happens in the video?")
            st.write("- Who are the people in the video?")
            st.write("- What objects or scenes are visible?")
            st.write("- Summarize the video content")
        
        if st.session_state.audio_transcriptions and st.session_state.video_analyses:
            st.write("**Combined queries:**")
            st.write("- Compare the audio and video content")
            st.write("- What information is available from all media files?")

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
                                            use_container_width=True,
                                            key=f"download_btn_{msg_idx}_{i}_{source_path.name}"
                                        )
                        
                        st.markdown("---")  # Divider between citations

# Chat input
chat_placeholder = "Ask a question about your documents"
if st.session_state.audio_transcriptions:
    chat_placeholder += " or audio files"
if st.session_state.video_analyses:
    if st.session_state.audio_transcriptions:
        chat_placeholder += " or video files"
    else:
        chat_placeholder += " or video files"
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
            # Show indicator if multimedia context is being used
            context_indicators = []
            if st.session_state.audio_transcriptions:
                context_indicators.append(f"🎧 {len(st.session_state.audio_transcriptions)} audio file(s)")
            if st.session_state.video_analyses:
                context_indicators.append(f"🎬 {len(st.session_state.video_analyses)} video file(s)")
            
            if context_indicators:
                st.info(f"Including context from: {', '.join(context_indicators)}")
            
            with st.spinner("Thinking..."):
                try:
                    # Debug: Check what multimedia content is available
                    st.write(f"🔍 Debug: Found {len(st.session_state.audio_transcriptions)} audio transcriptions and {len(st.session_state.video_analyses)} video analyses")
                    for fname, data in st.session_state.audio_transcriptions.items():
                        st.write(f"🎧 {fname}: {len(data.get('transcription', ''))} characters")
                    for fname, data in st.session_state.video_analyses.items():
                        st.write(f"🎬 {fname}: Analysis available")
                    
                    # Enhance the prompt with multimedia context if available
                    enhanced_prompt = prompt
                    multimedia_context = ""
                    
                    # Add audio context
                    if st.session_state.audio_transcriptions:
                        multimedia_context += "\n\n--- Audio Content Available ---\n"
                        for filename, data in st.session_state.audio_transcriptions.items():
                            transcription = data.get('transcription', '')
                            multimedia_context += f"Audio file '{filename}': {transcription}\n\n"
                        multimedia_context += "--- End Audio Content ---\n\n"
                    
                    # Add video context
                    if st.session_state.video_analyses:
                        multimedia_context += "\n\n--- Video Content Available ---\n"
                        for filename, data in st.session_state.video_analyses.items():
                            analysis = data.get('analysis', '')
                            multimedia_context += f"Video file '{filename}': {analysis}\n\n"
                        multimedia_context += "--- End Video Content ---\n\n"
                    
                    if multimedia_context:
                        enhanced_prompt = multimedia_context + "User question: " + prompt
                        st.write(f"📝 Enhanced prompt preview: {enhanced_prompt[:200]}...")
                    
                    # Get response from RAG system
                    response = st.session_state.rag_system.query(
                        question=enhanced_prompt,
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
                    if use_memory and st.session_state.memory_manager:
                        try:
                            # Combine user input and AI response for memory
                            memory_content = f"User: {prompt}\nAssistant: {response['answer']}"
                            
                            # Try MEM0 first if available
                            if hasattr(st.session_state.memory_manager, 'add'):
                                result = st.session_state.memory_manager.add(
                                    messages=[
                                        {"role": "user", "content": prompt},
                                        {"role": "assistant", "content": response['answer']}
                                    ],
                                    user_id=st.session_state.user_id,
                                    metadata={
                                        "source": "chat",
                                        "citations": response.get("citations", [])
                                    }
                                )
                            else:
                                # Fallback to local memory manager
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
                                # Log successful addition
                                if hasattr(st.session_state.memory_manager, 'add'):  # MEM0
                                    if isinstance(result, dict) and 'results' in result and isinstance(result['results'], list):
                                        logger.info(f"Successfully added to memory. Result: {len(result['results'])} items")
                                    else:
                                        logger.warning(f"MEM0 returned unexpected response format: {result}")
                                else:  # Local memory manager
                                    logger.info("Successfully added to local memory")
                        except Exception as mem_error:
                            error_msg = str(mem_error)
                            logger.error(f"Error saving to memory: {error_msg}", exc_info=True)
                            st.warning(f"⚠ Could not save to memory: {error_msg[:100]}...")
                        except Exception as e:
                            error_msg = f"Failed to handle memory: {str(e)}"
                            logger.error(error_msg, exc_info=True)
                            st.warning("Failed to save conversation to memory")
                
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    st.error("An error occurred while generating the response. Please try again.")

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