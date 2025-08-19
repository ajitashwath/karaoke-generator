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
import numpy as np
from scipy.signal import butter, sosfilt
import requests
import warnings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
            provide a JSON object with EXACTLY these fields:
            - title: string (song title)
            - artist: string (artist name)
            - album: string (album name, or "Single" if unknown)
            - year: string (year or "Unknown")
            - genre: string (music genre)
            - duration: string (format: "mm:ss", estimate if unknown)
            - language: string (primary language of the song)
            - popularity: number (integer from 1-10)
            
            Return ONLY the JSON object, no additional text. Ensure all fields are included."""),
            HumanMessage(content=f"Song: {song_name}\nAdditional details:\n{answers_text}")
        ])
        response = self.llm.invoke(prompt.format_messages())
        
        # Default values
        default_info = {
            "title": song_name, 
            "artist": "Unknown Artist", 
            "album": "Unknown Album", 
            "year": "Unknown",
            "genre": "Pop", 
            "duration": "3:30", 
            "language": "English", 
            "popularity": 5
        }
        
        try:
            # Try to parse the JSON response
            content = response.content.strip()
            # Remove markdown code blocks if present
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            content = content.strip()
            
            parsed_info = json.loads(content)
            
            # Merge with defaults to ensure all fields are present
            for key, default_value in default_info.items():
                if key not in parsed_info:
                    parsed_info[key] = default_value
                # Ensure popularity is an integer
                if key == 'popularity':
                    try:
                        parsed_info[key] = min(10, max(1, int(parsed_info[key])))
                    except:
                        parsed_info[key] = 5
            
            return parsed_info
            
        except Exception as e:
            print(f"Error parsing song info JSON: {e}")
            # Return default info with the song name
            default_info["title"] = song_name
            return default_info

    def get_lyrics(self, song_info: Dict[str, Any]) -> str:
        """Get song lyrics structure in the correct language"""
        detected_language = song_info.get('language', 'English')
        
        # Language-specific prompts
        language_instructions = {
            'Hindi': 'Generate Hindi lyrics structure using Devanagari script. Include typical Bollywood song elements like antara, mukhda.',
            'Tamil': 'Generate Tamil lyrics structure using Tamil script. Include typical elements like pallavi, charanam.',
            'Telugu': 'Generate Telugu lyrics structure using Telugu script.',
            'Malayalam': 'Generate Malayalam lyrics structure using Malayalam script.',
            'Kannada': 'Generate Kannada lyrics structure using Kannada script.',
            'Bengali': 'Generate Bengali lyrics structure using Bengali script.',
            'Marathi': 'Generate Marathi lyrics structure using Devanagari script.',
            'Gujarati': 'Generate Gujarati lyrics structure using Gujarati script.',
            'Punjabi': 'Generate Punjabi lyrics structure using Gurmukhi script.',
            'Urdu': 'Generate Urdu lyrics structure using Urdu script.',
            'Spanish': 'Generate Spanish lyrics structure with proper Spanish verses and chorus.',
            'French': 'Generate French lyrics structure with proper French verses and chorus.',
            'German': 'Generate German lyrics structure with proper German verses and chorus.',
            'Italian': 'Generate Italian lyrics structure with proper Italian verses and chorus.',
            'Portuguese': 'Generate Portuguese lyrics structure with proper Portuguese verses and chorus.',
            'Japanese': 'Generate Japanese lyrics structure using Japanese script (Hiragana/Katakana/Kanji).',
            'Korean': 'Generate Korean lyrics structure using Hangul script.',
            'Chinese': 'Generate Chinese lyrics structure using Chinese characters (Simplified or Traditional).',
            'Arabic': 'Generate Arabic lyrics structure using Arabic script.',
            'Russian': 'Generate Russian lyrics structure using Cyrillic script.',
            'Default': 'Generate lyrics structure in the appropriate language for this song.'
        }
        
        instruction = language_instructions.get(detected_language, language_instructions['Default'])
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are a multilingual karaoke lyrics generator. {instruction}
            
            IMPORTANT: 
            - Generate lyrics in the SAME LANGUAGE as the original song
            - Use proper script/alphabet for the language
            - Create a representative structure with actual sample lyrics (not just placeholders)
            - Make it appropriate for karaoke singing
            - Include verse/chorus structure typical for that language/culture
            - Do NOT use English if the song is in another language
            
            Format the output as clean, singable lyrics without extra formatting."""),
            HumanMessage(content=f"Song: {song_info.get('title', 'Unknown')} by {song_info.get('artist', 'Unknown')} (Language: {detected_language})")
        ])
        response = self.llm.invoke(prompt.format_messages())
        return response.content

    def download_audio(self, song_info: Dict[str, Any]) -> str:
        """Download audio from YouTube with better quality options"""
        search_query = f"{song_info.get('title', '')} {song_info.get('artist', '')} official audio"
        temp_path = tempfile.mktemp(suffix='.%(ext)s')
        
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio/best',
            'outtmpl': temp_path,
            'noplaylist': True,
            'extract_flat': False,
            'prefer_ffmpeg': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_results = ydl.extract_info(f"ytsearch1:{search_query}", download=False)
                if not search_results.get('entries'):
                    raise Exception("No search results found for the song on YouTube.")
                
                video_url = search_results['entries'][0]['webpage_url']
                info = ydl.extract_info(video_url, download=True)
                
                # Get the actual output filename (with .wav extension)
                base_path = temp_path.replace('.%(ext)s', '')
                wav_path = f"{base_path}.wav"
                
                if os.path.exists(wav_path):
                    return wav_path
                else:
                    # Fallback: find any audio file with the base name
                    for ext in ['.wav', '.mp3', '.m4a', '.webm']:
                        test_path = f"{base_path}{ext}"
                        if os.path.exists(test_path):
                            return test_path
                    
                    raise Exception("Downloaded audio file not found.")
                
        except Exception as e:
            raise Exception(f"Failed to download audio: {str(e)}")

    def separate_vocals_advanced(self, audio_path: str) -> str:
        """Advanced vocal separation using multiple techniques"""
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 10240:
            raise ValueError("Downloaded audio file is missing, empty, or too small.")
        
        try:
            # Load audio with error handling for different formats
            try:
                y, sr = librosa.load(audio_path, sr=44100, mono=False, res_type='kaiser_fast')
            except Exception:
                # Fallback loading method
                y, sr = librosa.load(audio_path, sr=22050, mono=False)
            
            # Ensure we have stereo audio for better separation
            if y.ndim == 1:
                # If mono, create pseudo-stereo
                y_stereo = np.array([y, y])
            else:
                y_stereo = y
            
            # Multiple vocal separation techniques
            
            # Method 1: Center channel extraction (vocal isolation removal)
            if y_stereo.shape[0] == 2:
                # Subtract right channel from left channel
                vocal_isolated = y_stereo[0] - y_stereo[1]
                # Keep the center (instrumental + some vocals)
                instrumental_basic = (y_stereo[0] + y_stereo[1]) / 2
                # Reduce vocal frequencies further
                instrumental_basic = instrumental_basic - (vocal_isolated * 0.3)
            else:
                instrumental_basic = librosa.to_mono(y_stereo)
            
            # Method 2: Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(instrumental_basic, margin=8)
            
            # Method 3: Spectral subtraction for vocal frequencies
            S = librosa.stft(instrumental_basic, n_fft=4096, hop_length=1024)
            magnitude, phase = np.abs(S), np.angle(S)
            
            # Reduce vocal frequency range (roughly 80Hz to 1000Hz for human vocals)
            vocal_freq_bins = int(1000 * S.shape[0] / (sr / 2))
            vocal_start_bin = int(80 * S.shape[0] / (sr / 2))
            
            # Apply gentle reduction to vocal frequencies
            magnitude[vocal_start_bin:vocal_freq_bins, :] *= 0.4
            
            # Reconstruct audio
            S_modified = magnitude * np.exp(1j * phase)
            y_vocal_reduced = librosa.istft(S_modified, hop_length=1024)
            
            # Method 4: Combine techniques
            # Weight the different methods
            y_instrumental = (
                y_harmonic * 0.5 +           # Harmonic component
                y_percussive * 0.8 +         # Keep drums/percussion
                y_vocal_reduced * 0.4        # Spectrally processed
            )
            
            # Method 5: Final EQ adjustments
            # High-pass filter to remove some low-end vocal rumble
            def butter_highpass(cutoff, fs, order=5):
                nyquist = 0.5 * fs
                normal_cutoff = cutoff / nyquist
                sos = butter(order, normal_cutoff, btype='high', output='sos')
                return sos
            
            sos = butter_highpass(60, sr, order=3)  # Remove below 60Hz
            y_instrumental = sosfilt(sos, y_instrumental)
            
            # Normalize audio to prevent clipping
            y_instrumental = y_instrumental / (np.max(np.abs(y_instrumental)) + 1e-6) * 0.95
            
            # Save instrumental
            output_path = tempfile.mktemp(suffix='.wav')
            sf.write(output_path, y_instrumental, sr, format='WAV', subtype='PCM_16')
            
            return output_path
            
        except Exception as e:
            # Fallback to basic method
            try:
                y, sr = librosa.load(audio_path, sr=22050, mono=True)
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                y_instrumental = y_harmonic + (y_percussive * 0.7)
                
                output_path = tempfile.mktemp(suffix='.wav')
                sf.write(output_path, y_instrumental, sr)
                return output_path
            except Exception as fallback_error:
                raise Exception(f"Could not process the audio file: {str(e)}. Fallback also failed: {str(fallback_error)}")

    def create_lyrics_timing(self, lyrics: str, duration_str: str) -> List[Tuple[float, float, str]]:
        """Create timing for lyrics based on song duration"""
        # Parse duration string (mm:ss format)
        try:
            if ':' in duration_str:
                parts = duration_str.split(':')
                if len(parts) == 2:
                    minutes, seconds = map(int, parts)
                    total_duration = minutes * 60 + seconds
                else:
                    total_duration = 210  # Default 3:30
            else:
                total_duration = 210  # Default 3:30
        except:
            total_duration = 210  # Default fallback
        
        # Clean and split lyrics into lines
        lines = []
        for line in lyrics.split('\n'):
            line = line.strip()
            if line and not line.startswith('[') and not line.startswith('(') and len(line) > 2:
                # Remove common structural markers
                if not any(marker in line.lower() for marker in ['verse', 'chorus', 'bridge', 'intro', 'outro']):
                    lines.append(line)
        
        if not lines:
            return [(0, total_duration, "🎵 Instrumental Track 🎵")]
        
        # Create timing for each line
        timed_lyrics = []
        
        # Add intro pause
        intro_duration = 8.0
        timed_lyrics.append((0, intro_duration, "🎵 Get ready to sing! 🎵"))
        
        # Distribute remaining time among lyrics
        remaining_time = total_duration - intro_duration - 8.0  # Leave 8s outro
        
        # Calculate time per line with some variation
        base_time_per_line = remaining_time / len(lines) if lines else 5.0
        
        current_time = intro_duration
        for i, line in enumerate(lines):
            # Vary duration based on line length
            line_duration = base_time_per_line * (0.8 + 0.4 * min(len(line) / 50, 2))
            end_time = min(current_time + line_duration, total_duration - 8)
            
            timed_lyrics.append((current_time, end_time, line))
            current_time = end_time + 0.5  # Small pause between lines
        
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
                progress_callback("Fetching lyrics in correct language...", 0.2)
            lyrics = self.get_lyrics(song_info)
            
            if progress_callback: 
                progress_callback("Downloading high-quality audio...", 0.4)
            audio_path = self.download_audio(song_info)
            
            if progress_callback: 
                progress_callback("Creating instrumental track (advanced processing)...", 0.7)
            instrumental_path = self.separate_vocals_advanced(audio_path)
            
            if progress_callback: 
                progress_callback("Creating synchronized lyrics timing...", 0.9)
            timed_lyrics = self.create_lyrics_timing(lyrics, song_info.get('duration', '3:30'))
            
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
        'langchain_community': 'langchain-community',
        'scipy': 'scipy',
        'numpy': 'numpy'
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