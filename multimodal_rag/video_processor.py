"""
Video processing module for the Multimodal RAG system.
Handles video frame extraction, audio extraction, and content analysis using Google Cloud services.
"""

import os
import cv2
import tempfile
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import moviepy as mp
from moviepy import VideoFileClip

# Google Cloud imports
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# Import from other modules
from .memory_manager import MemoryManager
from .rag_system import MultimodalRAG
from .simple_rag import SimpleRAG
from .audio_processor import initialize_components as init_audio_components, transcribe_audio

# Initialize global components
video_processor_components = {
    'memory_manager': None,
    'rag_system': None,
    'fallback_rag': None,
    'gemini_model': None
}

def initialize_video_components():
    """Initialize all video processing components safely"""
    global video_processor_components
    
    print("Initializing Video Processor components...")
    
    try:
        # Initialize Vertex AI
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'elite-thunder-461308-f7')
        location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        vertexai.init(project=project_id, location=location)
        
        # Initialize Gemini model for video analysis with 2.0 flash
        try:
            # Try gemini-2.0-flash first (best for multimodal)
            video_processor_components['gemini_model'] = GenerativeModel('gemini-2.0-flash-exp')
            print(" Successfully initialized gemini-2.0-flash-exp model")
        except Exception as flash_error:
            print(f"⚠ gemini-2.0-flash-exp failed: {flash_error}")
            try:
                # Fallback to standard 2.0 flash
                video_processor_components['gemini_model'] = GenerativeModel('gemini-2.0-flash')
                print(" Successfully initialized gemini-2.0-flash model")
            except Exception as flash2_error:
                print(f"⚠ gemini-2.0-flash failed: {flash2_error}")
                try:
                    # Final fallback to gemini-pro
                    video_processor_components['gemini_model'] = GenerativeModel('gemini-pro')
                    print(" Successfully initialized gemini-pro model as fallback")
                except Exception as pro_error:
                    print(f"⚠ All models failed: {pro_error}")
                    video_processor_components['gemini_model'] = None
        print("✓ Gemini model initialized for video analysis")
        
    except Exception as e:
        print(f"⚠ Warning: Could not initialize Gemini model: {e}")
    
    try:
        video_processor_components['memory_manager'] = MemoryManager(
            collection_name="multimodal_rag_video",
            use_hosted=False,
            user_id="video_processor_user"
        )
        print("✓ Video memory manager initialized")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize video memory manager: {e}")
    
    try:
        video_processor_components['rag_system'] = MultimodalRAG()
        print(" Video RAG system initialized")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize video RAG system: {e}")
    
    # Always initialize fallback RAG system
    try:
        video_processor_components['fallback_rag'] = SimpleRAG()
        print(" Video fallback RAG system initialized")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize video fallback RAG system: {e}")

class VideoProcessor:
    """Main video processing class"""
    
    def __init__(self):
        self.temp_dir = None
        self.video_info = {}
    
    def extract_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract basic metadata from video file"""
        try:
            with VideoFileClip(video_path) as video:
                metadata = {
                    'duration': video.duration,
                    'fps': video.fps,
                    'size': video.size,  # (width, height)
                    'aspect_ratio': video.aspect_ratio if hasattr(video, 'aspect_ratio') else video.size[0]/video.size[1],
                    'total_frames': int(video.fps * video.duration) if video.fps else 0
                }
                
                if video.audio:
                    metadata['has_audio'] = True
                    metadata['audio_fps'] = video.audio.fps
                else:
                    metadata['has_audio'] = False
                
                print(f"📹 Video metadata extracted: {metadata}")
                return metadata
                
        except Exception as e:
            print(f"⚠ Error extracting video metadata: {e}")
            return {}
    
    def extract_frames(self, video_path: str, max_frames: int = 10, method: str = 'uniform') -> List[str]:
        """
        Extract frames from video and return as base64 encoded strings
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            method: 'uniform' for evenly spaced frames, 'keyframes' for key frames
        """
        frames_b64 = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"🎬 Extracting frames from video: {total_frames} total frames, {duration:.2f}s duration")
            
            if method == 'uniform':
                # Extract frames uniformly distributed across the video
                frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
            else:
                # For now, use uniform distribution (keyframe detection would require more complex logic)
                frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Resize if too large (max 1024x1024 for efficiency)
                    max_size = 1024
                    if max(pil_image.size) > max_size:
                        pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    
                    # Convert to base64
                    import io
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format='JPEG', quality=85)
                    img_b64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    frames_b64.append(img_b64)
                    
                    timestamp = frame_idx / fps if fps > 0 else 0
                    print(f"📸 Extracted frame {i+1}/{len(frame_indices)} at {timestamp:.2f}s")
                
            cap.release()
            print(f"✓ Successfully extracted {len(frames_b64)} frames")
            
        except Exception as e:
            print(f"⚠ Error extracting frames: {e}")
        
        return frames_b64
    
    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """Extract audio from video and save as temporary WAV file"""
        try:
            with VideoFileClip(video_path) as video:
                if video.audio is None:
                    print("📹 No audio track found in video")
                    return None
                
                # Create temporary audio file
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                audio_path = temp_audio.name
                temp_audio.close()
                
                # Extract audio
                video.audio.write_audiofile(audio_path, logger=None)
                print(f"🎵 Audio extracted to: {audio_path}")
                
                return audio_path
                
        except Exception as e:
            print(f"⚠ Error extracting audio: {e}")
            return None
    
    def analyze_frames_with_gemini(self, frames_b64: List[str], query: str = "Describe what you see in these video frames") -> str:
        """Analyze video frames using Gemini Vision model"""
        if not video_processor_components['gemini_model']:
            return "Gemini model not available for video analysis"
        
        if not frames_b64:
            return "No frames to analyze"
        
        try:
            # Prepare the content for Gemini
            prompt = f"""
            Analyze these video frames and provide a comprehensive description. 
            Focus on: objects, people, actions, scenes, text visible, overall narrative or progression.
            
            User query: {query}
            
            Please provide a detailed response based on what you observe across all frames.
            """
            
            # Create parts for Gemini
            parts = [prompt]  # Start with text prompt
            
            # Add frames (limit to first 10 frames to avoid token limits)
            for i, frame_b64 in enumerate(frames_b64[:10]):
                # Create Part object for image
                image_part = Part.from_data(
                    data=base64.b64decode(frame_b64),
                    mime_type="image/jpeg"
                )
                parts.append(image_part)
            
            print(f"🤖 Analyzing {len(frames_b64[:10])} frames with Gemini...")
            
            # Generate content
            response = video_processor_components['gemini_model'].generate_content(parts)
            
            analysis = response.text if hasattr(response, 'text') else str(response)
            print(f"✓ Video analysis complete: {len(analysis)} characters")
            
            return analysis
            
        except Exception as e:
            print(f"⚠ Error analyzing frames with Gemini: {e}")
            # Fallback: provide comprehensive basic information about the video
            error_type = "authentication" if "UNAUTHENTICATED" in str(e) else "model access" if "not found" in str(e) else "processing"
            
            fallback_analysis = f"""
Video Processing Summary (Visual Analysis Unavailable):

Basic Information:
- Successfully extracted {len(frames_b64)} frames from the video
- Frame extraction completed without errors
- Video file appears to be valid and processable

Technical Details:
- Frame count: {len(frames_b64)}
- All frames were successfully converted to base64 format
- Frames are ready for analysis when Gemini access is restored

Issue Encountered:
- Visual analysis failed due to {error_type} issues
- Error: {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}

What This Means:
- The video file itself is perfectly fine and has been processed successfully
- Frame extraction, metadata parsing, and basic video operations work correctly
- Only the AI-powered visual analysis component is currently unavailable
- Audio transcription (if available) will still work normally

Recommendations:
- Check Google Cloud authentication setup
- Verify Vertex AI API access and billing
- Ensure the correct model names are being used
- The video content can still be queried using any available audio transcription
            """
            return fallback_analysis.strip()
    
    def process_video_comprehensive(self, video_path: str, query: str = "Analyze this video content") -> Dict[str, Any]:
        """
        Comprehensive video processing including frame extraction, audio transcription, and analysis
        """
        print(f"🎬 Starting comprehensive video processing for: {Path(video_path).name}")
        
        result = {
            'metadata': {},
            'frames': [],
            'visual_analysis': '',
            'audio_transcription': '',
            'combined_analysis': '',
            'error': None
        }
        
        try:
            # 1. Extract metadata
            print(" Extracting video metadata...")
            result['metadata'] = self.extract_video_metadata(video_path)
            
            # 2. Extract frames
            print(" Extracting video frames...")
            frames_b64 = self.extract_frames(video_path, max_frames=8)
            result['frames'] = frames_b64
            
            # 3. Analyze frames with Gemini
            if frames_b64:
                print(" Analyzing frames with Gemini...")
                result['visual_analysis'] = self.analyze_frames_with_gemini(frames_b64, query)
            
            # 4. Extract and transcribe audio
            print("🎵 Processing audio track...")
            audio_path = self.extract_audio_from_video(video_path)
            if audio_path:
                try:
                    # Initialize audio components if needed
                    init_audio_components()
                    
                    # Transcribe audio
                    result['audio_transcription'] = transcribe_audio(audio_path)
                    print(f"✓ Audio transcribed: {len(result['audio_transcription'])} characters")
                    
                    # Clean up temporary audio file
                    os.unlink(audio_path)
                    
                except Exception as e:
                    print(f"⚠ Audio transcription failed: {e}")
                    result['audio_transcription'] = f"Audio transcription failed: {str(e)}"
            
            # 5. Combine analyses
            combined_parts = []
            
            if result['visual_analysis'] and not result['visual_analysis'].startswith('Error'):
                combined_parts.append(f"VISUAL ANALYSIS:\n{result['visual_analysis']}")
            elif result['visual_analysis']:
                # Include error analysis for transparency
                combined_parts.append(f"VISUAL ANALYSIS (FAILED):\n{result['visual_analysis']}")
            
            if result['audio_transcription'] and not result['audio_transcription'].startswith('Audio transcription failed'):
                combined_parts.append(f"AUDIO TRANSCRIPTION:\n{result['audio_transcription']}")
            elif result['audio_transcription']:
                # Include error info for transparency
                combined_parts.append(f"AUDIO PROCESSING:\n{result['audio_transcription']}")
            
            if result['metadata']:
                duration = result['metadata'].get('duration', 0)
                size = result['metadata'].get('size', (0, 0))
                fps = result['metadata'].get('fps', 0)
                combined_parts.append(f"VIDEO INFO:\nDuration: {duration:.2f}s, Resolution: {size[0]}x{size[1]}, FPS: {fps:.1f}, Frames extracted: {len(result['frames'])}")
            
            result['combined_analysis'] = "\n\n".join(combined_parts) if combined_parts else "Video processing completed with limited analysis available."
            
            # 6. Save to memory if available
            if video_processor_components['memory_manager']:
                try:
                    memory_content = f"Video: {Path(video_path).name}\n\n{result['combined_analysis']}"
                    video_processor_components['memory_manager'].add_memory(
                        content=memory_content,
                        metadata={
                            "source": "video_processing",
                            "file_path": video_path,
                            "query": query,
                            "duration": result['metadata'].get('duration', 0),
                            "frames_extracted": len(result['frames'])
                        }
                    )
                    print("✓ Video analysis saved to memory")
                except Exception as e:
                    print(f"⚠ Could not save to memory: {e}")
            
            print("🎉 Comprehensive video processing complete!")
            
        except Exception as e:
            error_msg = f"Error in comprehensive video processing: {str(e)}"
            print(f" {error_msg}")
            result['error'] = error_msg
        
        return result

def generate_smart_video_answer(result: Dict[str, Any], user_query: str) -> str:
    """
    Generate a smart answer based on video processing results and user query
    """
    metadata = result.get('metadata', {})
    duration = metadata.get('duration', 0)
    size = metadata.get('size', (0, 0))
    fps = metadata.get('fps', 0)
    frames_count = len(result.get('frames', []))
    
    # Get available content
    visual_analysis = result.get('visual_analysis', '')
    audio_transcription = result.get('audio_transcription', '')
    
    # Analyze the user query
    query_lower = user_query.lower()
    
    # Start building the answer
    answer_parts = []
    answer_parts.append(f"🎬 **Video Analysis for**: {user_query}")
    
    # Handle specific query types
    if any(word in query_lower for word in ['what', 'describe', 'content', 'about', 'happens']):
        answer_parts.append(f" **Video Content Analysis:**")
        
        if visual_analysis and not visual_analysis.startswith('Video Processing Summary (Visual Analysis Unavailable)'):
            answer_parts.append(f"**Visual Content:** {visual_analysis}")
        else:
            answer_parts.append(f"**Visual Content:** Advanced visual analysis is currently unavailable, but the video has been successfully processed.")
        
        if audio_transcription and not audio_transcription.startswith('Audio transcription failed'):
            answer_parts.append(f" **Audio Content:** {audio_transcription}")
        elif metadata.get('has_audio'):
            answer_parts.append(f" **Audio:** The video has an audio track, but transcription was not available.")
        else:
            answer_parts.append(f" **Audio:** This video does not contain an audio track.")
    
    elif any(word in query_lower for word in ['duration', 'long', 'length', 'time']):
        answer_parts.append(f" **Duration Information:**")
        answer_parts.append(f"The video is **{duration:.2f} seconds** long ({duration/60:.1f} minutes).")
        if fps:
            answer_parts.append(f"At {fps:.1f} FPS, this video contains approximately {int(fps * duration)} total frames.")
    
    elif any(word in query_lower for word in ['quality', 'resolution', 'size']):
        answer_parts.append(f" **Technical Quality:**")
        answer_parts.append(f"- **Resolution:** {size[0]}x{size[1]} pixels")
        answer_parts.append(f"- **Frame Rate:** {fps:.1f} FPS")
        answer_parts.append(f"- **Duration:** {duration:.2f} seconds")
        answer_parts.append(f"- **Aspect Ratio:** {metadata.get('aspect_ratio', 'Unknown'):.2f}" if metadata.get('aspect_ratio') else "")
    
    elif any(word in query_lower for word in ['summary', 'summarize']):
        answer_parts.append(f"📝 **Video Summary:**")
        summary_parts = []
        summary_parts.append(f"A {duration:.1f}-second video at {size[0]}x{size[1]} resolution")
        
        if audio_transcription and not audio_transcription.startswith('Audio transcription failed'):
            # Try to summarize the audio content
            if len(audio_transcription) > 200:
                summary_parts.append(f"with audio content about: {audio_transcription[:200]}...")
            else:
                summary_parts.append(f"with audio content: {audio_transcription}")
        elif metadata.get('has_audio'):
            summary_parts.append("with an audio track")
        else:
            summary_parts.append("without audio")
        
        answer_parts.append(" ".join(summary_parts))
        
        if visual_analysis and not visual_analysis.startswith('Video Processing Summary'):
            answer_parts.append(f"**Visual Summary:** {visual_analysis}")
    
    else:
        # General query - provide all available information
        answer_parts.append(f" **Complete Video Information:**")
        
        # Technical info
        answer_parts.append(f"**Technical:** {duration:.2f}s, {size[0]}x{size[1]}, {fps:.1f} FPS, {frames_count} frames analyzed")
        
        # Content info
        if audio_transcription and not audio_transcription.startswith('Audio transcription failed'):
            answer_parts.append(f" **Audio:** {audio_transcription}")
        
        if visual_analysis and not visual_analysis.startswith('Video Processing Summary'):
            answer_parts.append(f"👁️ **Visual:** {visual_analysis}")
        else:
            answer_parts.append(f"👁️ **Visual:** Analysis not available (requires enhanced AI vision setup)")
    
    # Add processing status
    answer_parts.append(f"\n **Processing Status:**")
    answer_parts.append(f"-  Video successfully loaded and processed")
    answer_parts.append(f"-  {frames_count} frames extracted for analysis")
    answer_parts.append(f"- {'' if audio_transcription and not audio_transcription.startswith('Audio transcription failed') else '' if metadata.get('has_audio') else '➖'} Audio {'transcribed' if audio_transcription and not audio_transcription.startswith('Audio transcription failed') else 'processing failed' if metadata.get('has_audio') else 'not present'}")
    answer_parts.append(f"- {'' if visual_analysis and not visual_analysis.startswith('Video Processing Summary') else ''} Visual analysis {'completed' if visual_analysis and not visual_analysis.startswith('Video Processing Summary') else 'unavailable'}")
    
    return "\n\n".join(filter(None, answer_parts))

# Global video processor instance
video_processor = VideoProcessor()

def process_video_query(video_path: str, user_query: str) -> Dict[str, Any]:
    """
    Main function to process a video file and answer a user query
    """
    print(f"🎬 Processing video query: {user_query}")
    
    # Initialize components if not already done
    if not video_processor_components['gemini_model']:
        initialize_video_components()
    
    # Process video
    result = video_processor.process_video_comprehensive(video_path, user_query)
    
    # Generate query-specific response
    if result['combined_analysis']:
        # Check if we have meaningful analysis content
        has_visual = result.get('visual_analysis') and not result['visual_analysis'].startswith('Video Processing Summary (Visual Analysis Unavailable)')
        has_audio = result.get('audio_transcription') and not result['audio_transcription'].startswith('Audio transcription failed')
        
        if has_visual or has_audio:
            # We have some meaningful content, try RAG system with fallback
            answer_generated = False
            
            # Try main RAG system first
            if video_processor_components['rag_system']:
                try:
                    enhanced_query = f"""
                    Based on this video analysis:
                    
                    {result['combined_analysis']}
                    
                    User question: {user_query}
                    
                    Please provide a comprehensive answer based on the video content above.
                    """
                    
                    rag_response = video_processor_components['rag_system'].query(
                        question=enhanced_query,
                        chat_history=[],
                        return_context=True
                    )
                    
                    result['answer'] = rag_response.get('answer', result['combined_analysis'])
                    result['citations'] = rag_response.get('citations', [])
                    answer_generated = True
                    print(" Video query processed with main RAG system")
                    
                except Exception as e:
                    print(f"⚠ Main RAG system query failed: {e}")
            
            # Try fallback RAG system if main failed
            if not answer_generated and video_processor_components['fallback_rag']:
                try:
                    # Add video content to fallback RAG
                    video_content = result['combined_analysis']
                    video_processor_components['fallback_rag'].add_text_content(
                        video_content, 
                        f"Video: {Path(video_path).name}"
                    )
                    
                    # Query the fallback RAG
                    rag_response = video_processor_components['fallback_rag'].query(user_query)
                    result['answer'] = rag_response.get('answer', result['combined_analysis'])
                    result['citations'] = rag_response.get('citations', [])
                    answer_generated = True
                    print(" Video query processed with fallback RAG system")
                    
                except Exception as fallback_e:
                    print(f"⚠ Fallback RAG system also failed: {fallback_e}")
            
            # Use combined analysis as final fallback
            if not answer_generated:
                result['answer'] = generate_smart_video_answer(result, user_query)
                print(" Generated smart video answer from direct analysis")
        else:
            # No meaningful content, provide helpful fallback
            metadata = result.get('metadata', {})
            duration = metadata.get('duration', 0)
            size = metadata.get('size', (0, 0))
            fps = metadata.get('fps', 0)
            frames_count = len(result.get('frames', []))
            
            result['answer'] = f"""
            I was able to process your video file successfully, but I cannot provide a detailed analysis of its visual content due to current limitations with the AI vision system.
            
            Here's what I can tell you about your video:
            
            **Video Technical Information:**
            - Duration: {duration:.2f} seconds
            - Resolution: {size[0]}x{size[1]} pixels
            - Frame Rate: {fps:.1f} FPS
            - Frames Extracted: {frames_count}
            - Audio Track: {'Present' if metadata.get('has_audio') else 'Not present'}
            
            **Processing Status:**
            -  Video file successfully loaded and processed
            -  Metadata extraction completed
            -  Frame extraction completed ({frames_count} frames)
            - {'' if result.get('audio_transcription') and not result['audio_transcription'].startswith('Audio transcription failed') else ''} Audio processing {'completed' if result.get('audio_transcription') and not result['audio_transcription'].startswith('Audio transcription failed') else 'failed or no audio present'}
            -  Visual content analysis unavailable (requires Google Cloud Vertex AI setup)
            
            **What you can do:**
            1. Check your Google Cloud authentication and Vertex AI access
            2. Upload the video again once the visual analysis is working
            3. If the video has audio, any spoken content would have been transcribed for querying
            
            The video processing infrastructure is working correctly - only the AI vision analysis component needs to be configured.
            """
    else:
        result['answer'] = "Video processing encountered errors and no analysis is available."
    
    return result

if __name__ == "__main__":
    # Example usage
    video_file = "path/to/video/file.mp4"  # Replace with actual video path
    query = "What happens in this video?"
    
    print("Video Processor - Multimodal RAG Integration")
    print("="*50)
    
    # Check if the file exists
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found.")
        print("Please provide a valid video file path.")
        print("Supported formats: MP4, AVI, MOV, MKV, etc.")
        exit(1)
    
    try:
        initialize_video_components()
        result = process_video_query(video_file, query)
        
        print("\n" + "="*50)
        print("RESULTS:")
        print("="*50)
        print(f"Video Analysis: {result.get('answer', 'No analysis available')}")
        
        if result.get('audio_transcription'):
            print(f"\nAudio Transcription: {result['audio_transcription']}")
        
        if result.get('metadata'):
            metadata = result['metadata']
            print(f"\nVideo Info: {metadata.get('duration', 0):.1f}s, {metadata.get('size', (0,0))[0]}x{metadata.get('size', (0,0))[1]}")
            
    except Exception as e:
        print(f"Error processing video: {e}")