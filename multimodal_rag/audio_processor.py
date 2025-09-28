import os
import tempfile
from google.cloud import speech
from .memory_manager import MemoryManager
from .rag_system import MultimodalRAG
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

def initialize_components():
    """Initialize all components safely"""
    global speech_client, memory_manager, rag_system
    
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
        print("✓ RAG system initialized")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize RAG system: {e}")

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
            print(f"🔄 Converting from {audio.channels} channels to mono")
            audio = audio.set_channels(1)
        
        # Convert to 16kHz
        if audio.frame_rate != 16000:
            print(f"🔄 Converting from {audio.frame_rate}Hz to 16000Hz")
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
    # For now, we'll use the transcription as context for the query
    enhanced_query = f"Based on this audio transcription: '{transcription}'. {user_query}"
    
    if rag_system is not None:
        try:
            response = rag_system.query(
                question=enhanced_query,
                chat_history=[],
                return_context=True
            )
            answer = response["answer"]
            print("✓ Query processed with RAG system")
        except Exception as e:
            # Fallback: return transcription with query context
            answer = f"Audio transcription: {transcription}\n\nRegarding your query '{user_query}': The transcribed audio content is shown above."
            print(f"⚠ Warning: RAG system query failed: {e}")
    else:
        # Fallback: return transcription with query context
        answer = f"Audio transcription: {transcription}\n\nRegarding your query '{user_query}': The transcribed audio content is shown above."
        print("⚠ Warning: RAG system not available, using fallback")

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


def process_audio_query(audio_file_path, user_query):
    """
    Processes the audio file, transcribes it, and answers the user's query.
    """
    # Step 1: Transcribe the audio
    transcription = transcribe_audio(audio_file_path)
    print(f"Transcription: {transcription}")

    # Step 2: Save transcription to memory if memory manager is available
    try:
        memory_manager.add_memory(
            content=transcription,
            metadata={"source": "audio_transcription", "file_path": audio_file_path}
        )
    except Exception as e:
        print(f"Warning: Could not save to memory: {e}")

    # Step 3: Answer the user's query using the RAG system
    # For now, we'll use the transcription as context for the query
    enhanced_query = f"Based on this audio transcription: '{transcription}'. {user_query}"
    
    try:
        response = rag_system.query(
            question=enhanced_query,
            chat_history=[],
            return_context=True
        )
        answer = response["answer"]
    except Exception as e:
        # Fallback: return transcription with query context
        answer = f"Audio transcription: {transcription}\n\nRegarding your query '{user_query}': The transcribed audio content is shown above."
        print(f"Warning: RAG system query failed: {e}")

    return {"transcription": transcription, "answer": answer}

if __name__ == "__main__":
    # Example usage
    audio_file = "/Users/adityaarya/Documents/new_chatbot_ab/Multimodal-RAG/harvard.wav"  # Replace with the actual path to the audio file
    query = "What is the content of the audio?"
    
    print("Audio Processor - Multimodal RAG Integration")
    print("="*50)
    
    # Check if the file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        print("Please provide a valid audio file path.")
        print("Supported formats: WAV, MP3, FLAC, etc.")
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
