#!/usr/bin/env python3
"""
Test script for the video processor module.
Usage: python test_video_processor.py [video_file_path] [query]
"""
import sys
import os
from multimodal_rag.video_processor import initialize_video_components, process_video_query

def main():
    if len(sys.argv) < 2:
        print("Video Processor Test Script")
        print("="*50)
        print("Usage: python test_video_processor.py <video_file_path> [query]")
        print("")
        print("Examples:")
        print("  python test_video_processor.py ./sample.mp4")
        print("  python test_video_processor.py ./demo.mov 'What are the main actions in this video?'")
        print("  python test_video_processor.py ./presentation.avi 'Summarize the key points from this video'")
        print("")
        print("Supported video formats: MP4, AVI, MOV, MKV, FLV, WMV, etc.")
        print("Make sure you have Google Cloud credentials configured.")
        return
    
    video_file = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "What happens in this video?"
    
    print("Video Processor - Multimodal RAG Integration")
    print("="*50)
    
    # Check if file exists
    if not os.path.exists(video_file):
        print(f"❌ Error: Video file '{video_file}' not found.")
        return
    
    print(f"🎬 Processing video file: {video_file}")
    print(f"❓ Query: {query}")
    print("")
    
    # Initialize components
    print("Initializing components...")
    initialize_video_components()
    print("")
    
    try:
        # Process the video
        result = process_video_query(video_file, query)
        
        print("="*50)
        print("RESULTS:")
        print("="*50)
        
        # Main answer
        print(f"📝 Analysis: {result.get('answer', 'No analysis available')}")
        print("")
        
        # Visual analysis
        if result.get('visual_analysis'):
            print(f"👁️ Visual Analysis: {result['visual_analysis'][:300]}...")
            print("")
        
        # Audio transcription
        if result.get('audio_transcription'):
            print(f"🎵 Audio Transcription: {result['audio_transcription']}")
            print("")
        
        # Metadata
        if result.get('metadata'):
            metadata = result['metadata']
            duration = metadata.get('duration', 0)
            size = metadata.get('size', (0, 0))
            fps = metadata.get('fps', 0)
            has_audio = metadata.get('has_audio', False)
            
            print("📊 Video Information:")
            print(f"   Duration: {duration:.1f} seconds")
            print(f"   Resolution: {size[0]}x{size[1]}")
            print(f"   FPS: {fps:.1f}")
            print(f"   Audio: {'Yes' if has_audio else 'No'}")
            print("")
        
        # Frames extracted
        frames_count = len(result.get('frames', []))
        if frames_count > 0:
            print(f"🖼️ Extracted {frames_count} frames for analysis")
        
        if result.get('error'):
            print(f"⚠️ Errors encountered: {result['error']}")
        
        print("")
        print("✅ Video processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        print("\nPossible solutions:")
        print("1. Check that your Google Cloud credentials are properly set up")
        print("2. Verify the video file format is supported")
        print("3. Ensure you have the necessary permissions for Google Cloud services")
        print("4. Make sure OpenCV and MoviePy are installed correctly")

if __name__ == "__main__":
    main()