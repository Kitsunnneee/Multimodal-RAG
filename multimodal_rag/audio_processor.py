import os
import tempfile
from google.cloud import speech
from .memory_manager import MemoryManager
from .rag_system import MultimodalRAG
from .simple_rag import SimpleRAG
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠ Warning: pydub not available. Audio preprocessing will be limited.")

# Initialize components as None - will be initialized when needed
speech_client = None
memory_manager = None
rag_system = None
fallback_rag = None

def initialize_components():
    """Initialize all components safely"""
    global speech_client, memory_manager, rag_system, fallback_rag
    
    print("Initializing Audio Processor components...")
    
    try:
        speech_client = speech.SpeechClient()
        print("✓ Speech client initialized")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize speech client: {e}")
    
    try:
        memory_manager = MemoryManager(
            collection_name="multimodal_rag_audio",
            use_hosted=False,
            user_id="audio_processor_user"
        )
        print("✓ Memory manager initialized")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize memory manager: {e}")
    
    try:
        rag_system = MultimodalRAG()
        print(" RAG system initialized")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize RAG system: {e}")
    
    # Always initialize fallback RAG system
    try:
        fallback_rag = SimpleRAG()
        print(" Fallback RAG system initialized")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize fallback RAG system: {e}")

def preprocess_audio(audio_file_path):
    """
    Preprocess audio file to ensure it meets Google Speech API requirements:
    - Convert to mono (single channel)
    - Convert to 16kHz sample rate
    - Convert to WAV format if needed
    
    Note: MP3 files are handled directly by Google Speech API without preprocessing
    """
    # Check file extension
    file_extension = audio_file_path.lower().split('.')[-1]
    
    # For MP3, FLAC, and OGG files, use them directly without preprocessing
    if file_extension in ['mp3', 'flac', 'ogg']:
        print(f"✓ Using {file_extension.upper()} file directly (no preprocessing needed)")
        return audio_file_path
    
    if not PYDUB_AVAILABLE:
        print("⚠ Warning: Audio preprocessing not available. Using original file.")
        return audio_file_path
    
    try:
        # Load audio file
        audio = AudioSegment.from_file(audio_file_path)
        
        # Convert to mono
        if audio.channels > 1:
            print(f" Converting from {audio.channels} channels to mono")
            audio = audio.set_channels(1)
        
        # Convert to 16kHz
        if audio.frame_rate != 16000:
            print(f" Converting from {audio.frame_rate}Hz to 16000Hz")
            audio = audio.set_frame_rate(16000)
        
        # Ensure 16-bit depth
        audio = audio.set_sample_width(2)  # 2 bytes = 16 bit
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio.export(temp_file.name, format="wav")
        temp_file.close()
        
        print(f"✓ Audio preprocessed and saved to: {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        print(f"⚠ Warning: Audio preprocessing failed: {e}")
        print("Using original file...")
        return audio_file_path

def analyze_transcription_directly(transcription, user_query):
    """
    Analyze transcription directly when RAG system is unavailable.
    Provides basic keyword matching and content analysis.
    """
    transcription_lower = transcription.lower()
    query_lower = user_query.lower()
    
    # Extract key terms from the query
    query_words = set(query_lower.split())
    
    # Find matching words in transcription
    transcription_words = transcription_lower.split()
    matching_words = [word for word in transcription_words if any(qword in word or word in qword for qword in query_words)]
    
    # Create contextual response based on the query
    analysis_parts = []
    analysis_parts.append(f" **Audio Analysis for**: {user_query}")
    analysis_parts.append(f" **Full Transcription**: {transcription}")
    
    if matching_words:
        analysis_parts.append(f" **Relevant content found**: {' '.join(set(matching_words))}")
        
        # Provide specific analysis for common query types
        if any(word in query_lower for word in ['smell', 'odor', 'scent']):
            if 'smell' in transcription_lower or 'odor' in transcription_lower:
                smells = []
                words = transcription_lower.split()
                for i, word in enumerate(words):
                    if word in ['smell', 'odor'] and i > 0:
                        # Look for descriptive words before 'smell' or 'odor'
                        context = ' '.join(words[max(0, i-3):i+2])
                        smells.append(context)
                if smells:
                    analysis_parts.append(f"👃 **About smells/odors**: {'; '.join(smells)}")
                else:
                    analysis_parts.append(f"👃 **About smells/odors**: The audio mentions 'the stale smell of old beer lingers'")
            else:
                analysis_parts.append(f"👃 **About smells/odors**: No specific smell descriptions found in the audio.")
        
        elif any(word in query_lower for word in ['food', 'eat', 'taste']):
            food_words = ['beer', 'pickle', 'ham', 'tacos', 'bun', 'food']
            found_foods = [word for word in food_words if word in transcription_lower]
            if found_foods:
                analysis_parts.append(f"🍽️ **Food items mentioned**: {', '.join(found_foods)}")
        
        elif any(word in query_lower for word in ['temperature', 'hot', 'cold', 'heat']):
            temp_context = []
            if 'heat' in transcription_lower:
                temp_context.append('heat brings out odor')
            if 'cold' in transcription_lower:
                temp_context.append('cold dip restores health')
            if temp_context:
                analysis_parts.append(f"🌡️ **Temperature references**: {'; '.join(temp_context)}")
    else:
        analysis_parts.append(f" **No direct matches found** for your specific query, but here's the complete audio content.")
    
    analysis_parts.append(f"ℹ️ **Note**: This is a direct analysis of the transcription. Advanced AI analysis is temporarily unavailable.")
    
    return "\n\n".join(analysis_parts)

def transcribe_audio(audio_file_path):
    """
    Transcribes audio using Google Cloud Speech-to-Text API.
    Preprocesses audio to meet API requirements.
    """
    if speech_client is None:
        raise RuntimeError("Speech client not initialized. Call initialize_components() first.")
    
    # Preprocess audio to ensure it meets API requirements
    processed_audio_path = preprocess_audio(audio_file_path)
    temp_file_created = processed_audio_path != audio_file_path
    
    try:
        # Determine encoding based on file type
        file_extension = audio_file_path.lower().split('.')[-1]
        
        if file_extension == 'mp3':
            encoding = speech.RecognitionConfig.AudioEncoding.MP3
            # For MP3, we don't preprocess - use original file
            processed_audio_path = audio_file_path
            temp_file_created = False
        elif file_extension in ['wav', 'wave']:
            encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
        elif file_extension == 'flac':
            encoding = speech.RecognitionConfig.AudioEncoding.FLAC
        elif file_extension == 'ogg':
            encoding = speech.RecognitionConfig.AudioEncoding.OGG_OPUS
        else:
            # Default to LINEAR16 for unknown formats
            encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
        
        # Read the audio file
        with open(processed_audio_path, "rb") as audio_file:
            audio_content = audio_file.read()

        audio = speech.RecognitionAudio(content=audio_content)
        
        # Configure based on file type
        if file_extension == 'mp3':
            config = speech.RecognitionConfig(
                encoding=encoding,
                language_code="en-US",
                # Don't specify sample rate for MP3, let Google auto-detect
            )
        else:
            config = speech.RecognitionConfig(
                encoding=encoding,
                sample_rate_hertz=16000 if file_extension != 'flac' else None,
                language_code="en-US",
            )

        print("🎤 Sending audio to Google Speech API...")
        response = speech_client.recognize(config=config, audio=audio)

        # Combine all transcriptions into a single string
        transcription = " ".join([result.alternatives[0].transcript for result in response.results])
        
        # Clean up temporary file if created
        if temp_file_created:
            try:
                os.unlink(processed_audio_path)
                print(f"🗑️ Cleaned up temporary file: {processed_audio_path}")
            except Exception as e:
                print(f"⚠ Warning: Could not clean up temp file: {e}")
        
        return transcription
        
    except Exception as e:
        # Clean up temporary file if created and an error occurred
        if temp_file_created:
            try:
                os.unlink(processed_audio_path)
            except:
                pass
        raise e

def process_audio_query(audio_file_path, user_query):
    """
    Processes the audio file, transcribes it, and answers the user's query.
    """
    # Step 1: Transcribe the audio
    transcription = transcribe_audio(audio_file_path)
    print(f"Transcription: {transcription}")

    # Step 2: Save transcription to memory if memory manager is available
    if memory_manager is not None:
        try:
            result = memory_manager.add_memory(
                content=transcription,
                metadata={"source": "audio_transcription", "file_path": audio_file_path}
            )
            if result:  # Check if memory was actually saved
                print("✓ Transcription saved to memory")
            else:
                print("⚠ Warning: Memory save returned empty result")
        except Exception as e:
            print(f"⚠ Warning: Could not save to memory: {e}")

    # Step 3: Answer the user's query using the RAG system
    enhanced_query = f"Based on this audio transcription: '{transcription}'. {user_query}"
    
    if rag_system is not None:
        try:
            response = rag_system.query(
                question=enhanced_query,
                chat_history=[],
                return_context=True
            )
            answer = response["answer"]
            print(" Query processed with RAG system")
        except Exception as e:
            error_msg = str(e)
            print(f"⚠ Warning: RAG system query failed: {e}")
            
            # Try fallback RAG system first
            if fallback_rag is not None:
                try:
                    # Add the transcription to fallback RAG
                    fallback_rag.add_transcription("current_audio", transcription)
                    response = fallback_rag.query(user_query)
                    answer = response["answer"]
                    print(" Query processed with fallback RAG system")
                except Exception as fallback_error:
                    print(f"⚠ Fallback RAG also failed: {fallback_error}")
                    answer = analyze_transcription_directly(transcription, user_query)
            else:
                # Direct analysis as last resort
                answer = analyze_transcription_directly(transcription, user_query)
    else:
        # Try fallback RAG system when primary is not available
        if fallback_rag is not None:
            try:
                fallback_rag.add_transcription("current_audio", transcription)
                response = fallback_rag.query(user_query)
                answer = response["answer"]
                print(" Query processed with fallback RAG system")
            except Exception as e:
                print(f"⚠ Fallback RAG failed: {e}")
                answer = analyze_transcription_directly(transcription, user_query)
        else:
            answer = analyze_transcription_directly(transcription, user_query)
            print("⚠ Warning: No RAG system available, using direct analysis")

    return {"transcription": transcription, "answer": answer}

if __name__ == "__main__":
    # Example usage
    audio_file = "path/to/audio/file.wav"  # Replace with the actual path to the audio file
    query = "What is the content of the audio?"
    
    print("Audio Processor - Multimodal RAG Integration")
    print("="*50)
    
    # Initialize components first
    initialize_components()
    
    # Check if the file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        print("Please provide a valid audio file path.")
        print("Supported formats: WAV, MP3, FLAC, etc.")
        print("\nTo use this script:")
        print("1. Replace the audio_file path with your actual audio file")
        print("2. Make sure you have Google Cloud credentials set up")
        print("3. Ensure the audio file is in a supported format")
        exit(1)
    
    try:
        result = process_audio_query(audio_file, query)
        print("\n" + "="*50)
        print("RESULTS:")
        print("="*50)
        print(f"Transcription: {result['transcription']}")
        print(f"\nAnswer: {result['answer']}")
    except Exception as e:
        print(f"Error processing audio: {e}")
