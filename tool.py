import os
import tempfile
import json
import subprocess
from typing import Dict, List, Any, Callable
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import yt_dlp
import librosa
import soundfile as sf
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import lyricsgenius

class KaraokeGenerator:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4",
            temperature=0.3
        )
        # self.genius = lyricsgenius.Genius("YOUR_GENIUS_API_KEY")
    
    def get_clarifying_questions(self, song_name: str) -> List[str]:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a music expert helping to identify the exact song for karaoke generation.
            Given a song name, generate 2-3 specific questions to clarify which exact version/recording the user wants.
            Focus on: artist/performer, specific version, language, or notable covers.
            Keep questions simple and practical."""),
            HumanMessage(content=f"Song name: {song_name}")
        ])
        response = self.llm.invoke(prompt.format_messages())
        
        questions = []
        lines = response.content.strip().split('\n')
        for line in lines:
            if line.strip() and ('?' in line):
                question = line.strip().lstrip('123.-* ')
                questions.append(question)
        return questions[:3]

    def get_song_info(self, song_name: str, answers: Dict[str, str]) -> Dict[str, Any]:
        answers_text = "\n".join([f"- {answer}" for answer in answers.values() if answer])
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a music database expert. Based on the song name and additional details,
            provide a JSON object with: title, artist, album, year, genre, duration (mm:ss), language, and a popularity score (1-10).
            Return ONLY the JSON object."""),
            HumanMessage(content=f"Song: {song_name}\nAdditional details:\n{answers_text}")
        ])
        response = self.llm.invoke(prompt.format_messages())
        
        try:
            return json.loads(response.content)
        except:
            return {
                "title": song_name, "artist": "Unknown", "album": "Unknown", "year": "Unknown",
                "genre": "Unknown", "duration": "3:30", "language": "English", "popularity": 5
            }

    def get_lyrics(self, song_info: Dict[str, Any]) -> str:
        try:
            if hasattr(self, 'genius'):
                song = self.genius.search_song(song_info['title'], song_info['artist'])
                if song:
                    return song.lyrics
        except:
            pass
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a lyrics database. Provide the complete lyrics for the requested song with proper verse/chorus structure."),
            HumanMessage(content=f"Song: {song_info['title']} by {song_info['artist']}")
        ])
        response = self.llm.invoke(prompt.format_messages())
        return response.content

    def download_audio(self, song_info: Dict[str, Any]) -> str:
        search_query = f"{song_info['title']} {song_info['artist']} official audio"
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': tempfile.mktemp(suffix='.%(ext)s'),
            'noplaylist': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_results = ydl.extract_info(f"ytsearch1:{search_query}", download=False)
                if search_results['entries']:
                    video_url = search_results['entries'][0]['webpage_url']
                    info = ydl.extract_info(video_url, download=True)
                    return ydl.prepare_filename(info)
                else:
                    raise Exception("No search results found for the song on YouTube.")
        except Exception as e:
            raise Exception(f"Failed to download audio: {str(e)}")

    def separate_vocals(self, audio_path: str) -> str:
        output_dir = tempfile.mkdtemp()
        instrumental_path = os.path.join(output_dir, 'instrumental.wav')
        try:
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 10240:
                raise ValueError("Downloaded audio file is missing, empty, or too small.")

            # Fixed: Use librosa.load with explicit sr parameter to avoid deprecation warning
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            
            # Use harmonic-percussive separation to create instrumental
            y_harmonic, _ = librosa.effects.hpss(y)
            
            # Fixed: Use soundfile.write instead of deprecated librosa.output.write_wav
            sf.write(instrumental_path, y_harmonic, sr)
            
            return instrumental_path
        except Exception as e:
            print(f"A detailed error occurred in audio processing: {e}")
            raise Exception(f"Could not process the audio file. Ensure FFmpeg is installed and accessible in your system's PATH. Error: {str(e)}")

    def create_background_frames(self, duration: float, fps: int = 24) -> str:
        """Create background video frames using OpenCV"""
        width, height = 1920, 1080
        total_frames = int(duration * fps)
        frames_dir = tempfile.mkdtemp()
        
        for frame_num in range(total_frames):
            t = frame_num / fps
            
            # Create gradient background
            gradient = np.zeros((height, width, 3), dtype=np.uint8)
            hue = int((t * 20) % 180)  # OpenCV uses 0-180 for hue
            
            # Create HSV gradient
            for y in range(height):
                ratio = y / height
                h = int(hue + ratio * 30) % 180
                s = 255
                v = int(100 + ratio * 50)
                gradient[y, :] = [h, s, v]
            
            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(gradient, cv2.COLOR_HSV2BGR)
            
            frame_path = os.path.join(frames_dir, f"frame_{frame_num:06d}.png")
            cv2.imwrite(frame_path, frame_bgr)
        
        return frames_dir

    def create_text_overlay_frames(self, timed_lyrics: List[tuple], duration: float, fps: int = 24) -> str:
        """Create text overlay frames using PIL"""
        width, height = 1920, 1080
        total_frames = int(duration * fps)
        overlay_dir = tempfile.mkdtemp()
        
        # Try to load a font, fallback to default if not available
        try:
            # Try different font paths for different operating systems
            font_paths = [
                "arial.ttf",  # Windows
                "/System/Library/Fonts/Arial.ttf",  # macOS
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",  # Linux
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux alternative
            ]
            
            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, 50)
                    break
                except (OSError, IOError):
                    continue
            
            if font is None:
                font = ImageFont.load_default()
                
        except Exception:
            font = ImageFont.load_default()
        
        for frame_num in range(total_frames):
            t = frame_num / fps
            
            # Create transparent overlay
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Find current lyrics
            current_text = ""
            for start_time, end_time, text in timed_lyrics:
                if start_time <= t < end_time:
                    current_text = text
                    break
            
            if current_text:
                # Get text dimensions
                try:
                    bbox = draw.textbbox((0, 0), current_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError:
                    # Fallback for older PIL versions
                    text_width, text_height = draw.textsize(current_text, font=font)
                
                # Center the text
                x = (width - text_width) // 2
                y = (height - text_height) // 2 + height // 4  # Lower third
                
                # Draw text with outline for better visibility
                outline_width = 2
                for dx in range(-outline_width, outline_width + 1):
                    for dy in range(-outline_width, outline_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((x + dx, y + dy), current_text, fill='black', font=font)
                
                # Draw main text
                draw.text((x, y), current_text, fill='white', font=font)
            
            frame_path = os.path.join(overlay_dir, f"overlay_{frame_num:06d}.png")
            overlay.save(frame_path)
        
        return overlay_dir

    def combine_video_with_ffmpeg(self, background_dir: str, overlay_dir: str, audio_path: str, output_path: str, fps: int = 24) -> str:
        """Use FFmpeg to combine background, text overlay, and audio"""
        try:
            # Check if FFmpeg is available
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise Exception("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
            
            # FFmpeg command to combine everything
            cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-framerate', str(fps),
                '-i', os.path.join(background_dir, 'frame_%06d.png'),  # Background frames
                '-framerate', str(fps),
                '-i', os.path.join(overlay_dir, 'overlay_%06d.png'),   # Text overlay frames
                '-i', audio_path,  # Audio
                '-filter_complex', '[0:v][1:v]overlay=0:0[v]',  # Overlay text on background
                '-map', '[v]',  # Use combined video
                '-map', '2:a',  # Use audio from third input
                '-c:v', 'libx264',  # Video codec
                '-c:a', 'aac',      # Audio codec
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                '-shortest',  # End when shortest stream ends
                '-loglevel', 'error',  # Reduce FFmpeg output
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"FFmpeg error: {e.stderr}")

    def create_karaoke_video(self, instrumental_path: str, lyrics: str, song_info: Dict[str, Any]) -> str:
        """Create karaoke video using OpenCV, PIL, and FFmpeg"""
        
        try:
            # Get audio duration using librosa
            y, sr = librosa.load(instrumental_path, sr=None)
            duration = len(y) / sr
            
            fps = 24
            timed_lyrics = self.time_lyrics(lyrics, duration)
            
            # Create background frames
            background_dir = self.create_background_frames(duration, fps)
            
            # Create text overlay frames
            overlay_dir = self.create_text_overlay_frames(timed_lyrics, duration, fps)
            
            # Combine everything with FFmpeg
            output_path = tempfile.mktemp(suffix='.mp4')
            final_path = self.combine_video_with_ffmpeg(background_dir, overlay_dir, instrumental_path, output_path, fps)
            
            # Clean up temporary directories
            try:
                import shutil
                shutil.rmtree(background_dir)
                shutil.rmtree(overlay_dir)
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up temporary files: {cleanup_error}")
            
            return final_path
            
        except Exception as e:
            raise Exception(f"Error creating karaoke video: {str(e)}")

    def time_lyrics(self, lyrics: str, duration: float) -> List[tuple]:
        """Create timing for lyrics based on duration"""
        lines = [line.strip() for line in lyrics.split('\n') if line.strip() and not line.startswith('[')]
        if not lines:
            return [(0, duration, "Instrumental")]
        
        time_per_line = duration / len(lines)
        timed_lyrics = []
        for i, line in enumerate(lines):
            start_time = i * time_per_line
            end_time = (i + 1) * time_per_line
            timed_lyrics.append((start_time, end_time, line))
        
        return timed_lyrics

    def generate_karaoke(self, song_name: str, answers: Dict[str, str], progress_callback: Callable = None) -> Dict[str, Any]:
        """Main method to generate karaoke video"""
        try:
            if progress_callback: 
                progress_callback("Getting song information...", 0.1)
            song_info = self.get_song_info(song_name, answers)
            
            if progress_callback: 
                progress_callback("Fetching lyrics...", 0.2)
            lyrics = self.get_lyrics(song_info)
            
            if progress_callback: 
                progress_callback("Downloading audio...", 0.4)
            audio_path = self.download_audio(song_info)
            
            if progress_callback: 
                progress_callback("Creating instrumental track...", 0.6)
            instrumental_path = self.separate_vocals(audio_path)
            
            if progress_callback: 
                progress_callback("Generating karaoke video...", 0.8)
            video_path = self.create_karaoke_video(instrumental_path, lyrics, song_info)
            
            if progress_callback: 
                progress_callback("Complete!", 1.0)
            
            return {
                'song_info': song_info, 
                'lyrics': lyrics,
                'instrumental_path': instrumental_path, 
                'video_path': video_path,
                'success': True
            }
            
        except Exception as e:
            error_msg = f"Error generating karaoke: {str(e)}"
            print(error_msg)
            return {'success': False, 'error': error_msg}


# Additional utility functions for installation requirements
def check_dependencies():
    """Check if required dependencies are installed"""
    dependencies = {
        'ffmpeg': 'FFmpeg is required for video processing',
        'librosa': 'librosa is required for audio processing', 
        'soundfile': 'soundfile is required for audio I/O',
        'opencv-python': 'opencv-python (cv2) is required for video processing',
        'Pillow': 'Pillow (PIL) is required for image processing',
        'yt-dlp': 'yt-dlp is required for downloading audio',
        'langchain-community': 'langchain-community is required for LLM integration'
    }
    
    missing = []
    
    # Check FFmpeg separately as it's a system dependency
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append('ffmpeg (system dependency)')
    
    # Check Python packages
    python_packages = ['librosa', 'soundfile', 'cv2', 'PIL', 'yt_dlp']
    for package in python_packages:
        try:
            __import__(package)
        except ImportError:
            if package == 'cv2':
                missing.append('opencv-python')
            elif package == 'PIL':
                missing.append('Pillow')
            elif package == 'yt_dlp':
                missing.append('yt-dlp')
            else:
                missing.append(package)
    
    return missing


def install_requirements():
    """Print installation instructions for missing dependencies"""
    missing = check_dependencies()
    
    if not missing:
        print("✅ All dependencies are installed!")
        return True
    
    print("❌ Missing dependencies:")
    for dep in missing:
        print(f"   - {dep}")
    
    print("\n📦 Installation commands:")
    
    python_deps = [dep for dep in missing if dep != 'ffmpeg (system dependency)']
    if python_deps:
        print(f"pip install {' '.join(python_deps)}")
    
    if 'ffmpeg (system dependency)' in missing:
        print("\nFFmpeg installation:")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        print("  macOS: brew install ffmpeg")
        print("  Linux: sudo apt install ffmpeg (Ubuntu/Debian) or sudo yum install ffmpeg (RHEL/CentOS)")
    
    return False


if __name__ == "__main__":
    # Check dependencies on import
    install_requirements()