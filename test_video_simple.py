#!/usr/bin/env python3
"""
Simple test script for video processing functionality.
This script tests basic video processing without complex Gemini integration.
"""

import os
from pathlib import Path
from multimodal_rag.video_processor import VideoProcessor

def test_basic_video_processing():
    """Test basic video processing components"""
    print("🎬 Testing Basic Video Processing")
    print("=" * 50)
    
    # Initialize video processor
    processor = VideoProcessor()
    
    # Look for a test video file
    test_files = [
        "/tmp/file_example_MP4_480_1_5MG.mp4",  # If uploaded to temp
        "file_example_MP4_480_1_5MG.mp4",       # If in current directory
        "test_video.mp4",                       # Generic test file
    ]
    
    video_file = None
    for test_file in test_files:
        if os.path.exists(test_file):
            video_file = test_file
            break
    
    if not video_file:
        print("❌ No test video file found. Please provide a video file for testing.")
        print("Expected locations:")
        for f in test_files:
            print(f"  - {f}")
        return False
    
    print(f"📁 Using video file: {video_file}")
    
    try:
        # Test 1: Extract metadata
        print("\n🔍 Test 1: Extract metadata...")
        metadata = processor.extract_video_metadata(video_file)
        if metadata:
            print(f"✅ Metadata extraction successful:")
            print(f"   Duration: {metadata.get('duration', 0):.2f}s")
            print(f"   Resolution: {metadata.get('size', (0, 0))}")
            print(f"   FPS: {metadata.get('fps', 0)}")
            print(f"   Has audio: {metadata.get('has_audio', False)}")
        else:
            print("❌ Metadata extraction failed")
            return False
        
        # Test 2: Extract frames
        print("\n🖼️ Test 2: Extract frames...")
        frames = processor.extract_frames(video_file, max_frames=5)
        if frames:
            print(f"✅ Frame extraction successful: {len(frames)} frames extracted")
            print(f"   Average frame size: {sum(len(f) for f in frames) / len(frames) / 1024:.1f} KB")
        else:
            print("❌ Frame extraction failed")
            return False
        
        # Test 3: Extract audio (optional - may fail)
        print("\n🎵 Test 3: Extract audio...")
        audio_path = processor.extract_audio_from_video(video_file)
        if audio_path and os.path.exists(audio_path):
            print(f"✅ Audio extraction successful: {audio_path}")
            # Clean up
            os.unlink(audio_path)
        else:
            print("⚠️ Audio extraction failed or no audio track (this is okay)")
        
        # Test 4: Simple analysis (without Gemini)
        print("\n📊 Test 4: Basic analysis...")
        basic_analysis = f"""
        Video Analysis Summary:
        - File: {Path(video_file).name}
        - Duration: {metadata.get('duration', 0):.2f} seconds
        - Resolution: {metadata.get('size', (0, 0))[0]}x{metadata.get('size', (0, 0))[1]}
        - Frame Rate: {metadata.get('fps', 0):.1f} FPS
        - Total Frames: {metadata.get('total_frames', 0)}
        - Frames Extracted: {len(frames)}
        - Has Audio: {metadata.get('has_audio', False)}
        
        Basic video processing completed successfully.
        Note: Advanced visual analysis requires proper Google Cloud authentication.
        """
        
        print("✅ Basic analysis generated:")
        print(basic_analysis)
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_video_processing()
    if success:
        print("\n✅ Video processing basic functionality is working!")
        print("You can now test with the full Streamlit app.")
    else:
        print("\n❌ Basic video processing tests failed.")
        print("Please check the errors above and fix them before using the full app.")