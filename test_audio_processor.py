#!/usr/bin/env python3
"""
Test script for the audio processor module.
Usage: python test_audio_processor.py [audio_file_path] [query]
"""
import sys
import os
from multimodal_rag.audio_processor import initialize_components, process_audio_query

def main():
    if len(sys.argv) < 2:
        print("Audio Processor Test Script")
        print("="*50)
        print("Usage: python test_audio_processor.py <audio_file_path> [query]")
        print("")
        print("Examples:")
        print("  python test_audio_processor.py ./sample.wav")
        print("  python test_audio_processor.py ./meeting.mp3 'What were the main topics discussed?'")
        print("")
        print("Supported audio formats: WAV, MP3, FLAC, M4A, etc.")
        print("Make sure you have Google Cloud credentials configured.")
        return
    
    audio_file = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "What is the content of this audio?"
    
    print("Audio Processor - Multimodal RAG Integration")
    print("="*50)
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file '{audio_file}' not found.")
        return
    
    print(f"🎵 Processing audio file: {audio_file}")
    print(f"❓ Query: {query}")
    print("")
    
    # Initialize components
    print("Initializing components...")
    initialize_components()
    print("")
    
    try:
        # Process the audio
        result = process_audio_query(audio_file, query)
        
        print("="*50)
        print("RESULTS:")
        print("="*50)
        print(f"📝 Transcription: {result['transcription']}")
        print("")
        print(f"💬 Answer: {result['answer']}")
        print("")
        print("✅ Audio processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        print("\nPossible solutions:")
        print("1. Check that your Google Cloud credentials are properly set up")
        print("2. Verify the audio file format is supported")
        print("3. Ensure you have the necessary permissions for Google Cloud Speech API")

if __name__ == "__main__":
    main()