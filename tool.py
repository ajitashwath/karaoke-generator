import os
import tempfile
import json
import subprocess
from typing import Dict, List, Any, Callable, Tuple
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import yt_dlp
import librosa
import soundfile as sf

class KaraokeGenerator:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4o-mini",
            temperature=0.3
        )
    
    def get_clarifying_questions(self, song_name: str) -> List[str]:
        """Generate clarifying questions for song identification"""
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
        """Get detailed song information based on name and clarifying answers"""
        answers_text = "\n".join([f"- {answer}" for answer in answers.values() if answer])
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a music database expert. Based on the song name and additional details,
            provide a JSON object with: title, artist, album, year, genre, duration (mm:ss), language, and a popularity score (1-10).
            Return ONLY the JSON object, no additional text."""),
            HumanMessage(content=f"Song: {song_name}\nAdditional details:\n{answers_text}")
        ])
        response = self.llm.invoke(prompt.format_messages())
        
        try:
            return json.loads(response.content.strip())
        except:
            return {
                "title": song_name, 
                "artist": "Unknown", 
                "album": "Unknown", 
                "year": "Unknown",
                "genre": "Unknown", 
                "duration": "3:30", 
                "language": "English", 
                "popularity": 5
            }

    def get_lyrics(self, song_info: Dict[str, Any]) -> str:
        """Get song lyrics using LLM (for demonstration purposes)"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are helping create a karaoke system. Generate sample lyrics structure 
            for the requested song with proper verse/chorus format. Create a representative structure with 
            placeholder text like [Verse 1], [Chorus], etc. Do not reproduce actual copyrighted lyrics."""),
            HumanMessage(content=f"Song: {song_info['title']} by {song_info['artist']}")
        ])
        response = self.llm.invoke(prompt.format_messages())
        return response.content

    def download_audio(self, song_info: Dict[str, Any]) -> str:
        """Download audio from YouTube"""
        search_query = f"{song_info['title']} {song_info['artist']} official audio"
        temp_path = tempfile.mktemp(suffix='.%(ext)s')
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_path,
            'noplaylist': True,
            'extract_flat': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_results = ydl.extract_info(f"ytsearch1:{search_query}", download=False)
                if not search_results['entries']:
                    raise Exception("No search results found for the song on YouTube.")
                
                video_url = search_results['entries'][0]['webpage_url']
                info = ydl.extract_info(video_url, download=True)
                return ydl.prepare_filename(info)
                
        except Exception as e:
            raise Exception(f"Failed to download audio: {str(e)}")

    def separate_vocals(self, audio_path: str) -> str:
        """Create instrumental version by separating vocals"""
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 10240:
            raise ValueError("Downloaded audio file is missing, empty, or too small.")
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050, mono=False)
            
            # If stereo, convert to mono for processing
            if y.ndim > 1:
                y_mono = librosa.to_mono(y)
            else:
                y_mono = y
            
            # Use harmonic-percussive separation to create instrumental
            y_harmonic, y_percussive = librosa.effects.hpss(y_mono)
            
            # Combine harmonic and percussive parts (reduces vocals)
            y_instrumental = y_harmonic + (y_percussive * 0.3)
            
            # Save instrumental
            output_path = tempfile.mktemp(suffix='.wav')
            sf.write(output_path, y_instrumental, sr)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Could not process the audio file: {str(e)}")

    def create_lyrics_timing(self, lyrics: str, duration_str: str) -> List[Tuple[float, float, str]]:
        """Create timing for lyrics based on song duration"""
        # Parse duration string (mm:ss format)
        try:
            if ':' in duration_str:
                minutes, seconds = map(int, duration_str.split(':'))
                total_duration = minutes * 60 + seconds
            else:
                total_duration = 210  # Default 3:30
        except:
            total_duration = 210  # Default fallback
        
        # Clean and split lyrics into lines
        lines = [line.strip() for line in lyrics.split('\n') 
                if line.strip() and not line.startswith('[') and len(line.strip()) > 3]
        
        if not lines:
            return [(0, total_duration, "🎵 Instrumental Track 🎵")]
        
        # Create timing for each line
        timed_lyrics = []
        
        # Add intro pause
        intro_duration = 5.0
        timed_lyrics.append((0, intro_duration, "🎵 Get ready to sing! 🎵"))
        
        # Distribute remaining time among lyrics
        remaining_time = total_duration - intro_duration - 5.0  # Leave 5s outro
        time_per_line = remaining_time / len(lines)
        
        current_time = intro_duration
        for line in lines:
            end_time = current_time + time_per_line
            timed_lyrics.append((current_time, end_time, line))
            current_time = end_time
        
        # Add outro
        timed_lyrics.append((current_time, total_duration, "🎵 Thank you for singing! 🎵"))
        
        return timed_lyrics

    def generate_karaoke(self, song_name: str, answers: Dict[str, str], progress_callback: Callable = None) -> Dict[str, Any]:
        """Main method to generate karaoke with instrumental and timed lyrics"""
        try:
            if progress_callback: 
                progress_callback("Getting song information...", 0.1)
            song_info = self.get_song_info(song_name, answers)
            
            if progress_callback: 
                progress_callback("Fetching lyrics structure...", 0.2)
            lyrics = self.get_lyrics(song_info)
            
            if progress_callback: 
                progress_callback("Downloading audio...", 0.4)
            audio_path = self.download_audio(song_info)
            
            if progress_callback: 
                progress_callback("Creating instrumental track...", 0.7)
            instrumental_path = self.separate_vocals(audio_path)
            
            if progress_callback: 
                progress_callback("Creating lyrics timing...", 0.9)
            timed_lyrics = self.create_lyrics_timing(lyrics, song_info['duration'])
            
            if progress_callback: 
                progress_callback("Complete!", 1.0)
            
            # Clean up original audio file
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass
            
            return {
                'song_info': song_info, 
                'lyrics': lyrics,
                'timed_lyrics': timed_lyrics,
                'instrumental_path': instrumental_path,
                'success': True
            }
            
        except Exception as e:
            error_msg = f"Error generating karaoke: {str(e)}"
            print(error_msg)
            return {'success': False, 'error': error_msg}


def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    # Check Python packages
    required_packages = {
        'librosa': 'librosa',
        'soundfile': 'soundfile', 
        'yt_dlp': 'yt-dlp',
        'langchain_community': 'langchain-community'
    }
    
    for import_name, package_name in required_packages.items():
        try:
            if import_name == 'langchain_community':
                from langchain_community.chat_models import ChatOpenAI
            else:
                __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
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
    
    print(f"\n📦 Installation command:")
    print(f"pip install {' '.join(missing)}")
    
    return False


if __name__ == "__main__":
    # Check dependencies when run directly
    install_requirements()