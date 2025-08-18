import os
import tempfile
import requests
import json
from typing import Dict, List, Any, Callable
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import yt_dlp
import librosa
import soundfile as sf
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
from spleeter.separator import Separator
import genius
import lyricsgenius

class KaraokeGenerator:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4o-mini",
            temperature=0.3
        )
        
        self.separator = Separator('spleeter:2stems-16khz')
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
                question = line.strip()
                if question.startswith(('1.', '2.', '3.', '-', '*')):
                    question = question[2:].strip()
                questions.append(question)
        
        return questions[:3] 
    
    def get_song_info(self, song_name: str, answers: Dict[str, str]) -> Dict[str, Any]:
        """Get detailed song information using LLM."""
        
        answers_text = "\n".join([f"- {answer}" for answer in answers.values() if answer])
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a music database expert. Based on the song name and additional details,
            provide structured information about the song. Return ONLY a JSON object with these fields:
            {
                "title": "exact song title",
                "artist": "primary artist/performer",
                "album": "album name",
                "year": "release year",
                "genre": "music genre",
                "duration": "approximate duration (mm:ss)",
                "language": "primary language",
                "popularity": "popularity score 1-10"
            }"""),
            HumanMessage(content=f"Song: {song_name}\nAdditional details:\n{answers_text}")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        
        try:
            return json.loads(response.content)
        except:
            # Fallback if JSON parsing fails
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
        """Get song lyrics using multiple methods."""
        
        # Method 1: Try Genius API (if available)
        try:
            if hasattr(self, 'genius'):
                song = self.genius.search_song(song_info['title'], song_info['artist'])
                if song:
                    return song.lyrics
        except:
            pass
        
        # Method 2: Use LLM to generate/recall lyrics
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a lyrics database. Provide the complete lyrics for the requested song.
            If you don't know the exact lyrics, clearly state that and provide a similar structure or explain limitations.
            Format lyrics with proper verse/chorus structure."""),
            HumanMessage(content=f"Song: {song_info['title']} by {song_info['artist']}")
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        return response.content
    
    def download_audio(self, song_info: Dict[str, Any]) -> str:
        """Download audio using yt-dlp."""
        
        search_query = f"{song_info['title']} {song_info['artist']} official audio"
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': tempfile.mktemp(suffix='.%(ext)s'),
            'noplaylist': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Search for the song
                search_results = ydl.extract_info(
                    f"ytsearch1:{search_query}",
                    download=False
                )
                
                if search_results['entries']:
                    video_url = search_results['entries'][0]['webpage_url']
                    info = ydl.extract_info(video_url, download=True)
                    return ydl.prepare_filename(info)
        except Exception as e:
            raise Exception(f"Failed to download audio: {str(e)}")
    
    def separate_vocals(self, audio_path: str) -> str:
        """Separate vocals from music using Spleeter."""
        
        output_dir = tempfile.mkdtemp()
        
        try:
            # Use Spleeter to separate
            waveform, _ = librosa.load(audio_path, sr=22050)
            prediction = self.separator.separate(waveform)
            
            # Save instrumental (without vocals)
            instrumental = prediction['accompaniment']
            instrumental_path = os.path.join(output_dir, 'instrumental.wav')
            sf.write(instrumental_path, instrumental, 22050)
            
            return instrumental_path
            
        except Exception as e:
            # Fallback: return original audio if separation fails
            return audio_path
    
    def create_karaoke_video(self, instrumental_path: str, lyrics: str, song_info: Dict[str, Any]) -> str:
        """Create karaoke video with lyrics and instrumental."""
        
        # Load instrumental audio
        audio_clip = AudioFileClip(instrumental_path)
        duration = audio_clip.duration
        
        # Create background video (simple gradient or solid color)
        background = self.create_background_video(duration)
        
        # Parse and time lyrics
        timed_lyrics = self.time_lyrics(lyrics, duration)
        
        # Create text clips for lyrics
        text_clips = []
        for start_time, end_time, text in timed_lyrics:
            text_clip = TextClip(
                text,
                fontsize=50,
                color='white',
                font='Arial-Bold',
                stroke_color='black',
                stroke_width=2
            ).set_position('center').set_start(start_time).set_end(end_time)
            
            text_clips.append(text_clip)
        
        # Compose final video
        final_video = CompositeVideoClip([background] + text_clips)
        final_video = final_video.set_audio(audio_clip)
        
        # Export video
        output_path = tempfile.mktemp(suffix='.mp4')
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=tempfile.mktemp(suffix='.m4a'),
            remove_temp=True
        )
        
        return output_path
    
    def create_background_video(self, duration: float) -> VideoFileClip:
        """Create a simple background video."""
        
        # Create a simple gradient background
        width, height = 1920, 1080
        
        # Generate gradient frames
        def make_frame(t):
            # Create a moving gradient
            gradient = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Calculate color based on time
            hue = (t * 20) % 360
            color1 = np.array([hue, 255, 100])  # HSV
            color2 = np.array([(hue + 60) % 360, 255, 150])  # HSV
            
            # Convert to RGB and create gradient
            for y in range(height):
                ratio = y / height
                color = color1 * (1 - ratio) + color2 * ratio
                gradient[y, :] = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0, 0]
            
            return gradient
        
        from moviepy.video.VideoClip import VideoClip
        background_clip = VideoClip(make_frame, duration=duration)
        return background_clip
    
    def time_lyrics(self, lyrics: str, duration: float) -> List[tuple]:
        """Time lyrics to fit the song duration."""
        
        # Split lyrics into lines
        lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
        
        # Remove empty lines and section headers
        filtered_lines = []
        for line in lines:
            if line and not line.startswith('[') and not line.endswith(']'):
                filtered_lines.append(line)
        
        if not filtered_lines:
            return [(0, duration, "♪ Instrumental ♪")]
        
        # Calculate timing
        time_per_line = duration / len(filtered_lines)
        timed_lyrics = []
        
        for i, line in enumerate(filtered_lines):
            start_time = i * time_per_line
            end_time = (i + 1) * time_per_line
            timed_lyrics.append((start_time, end_time, line))
        
        return timed_lyrics
    
    def generate_karaoke(self, song_name: str, answers: Dict[str, str], 
                        progress_callback: Callable = None) -> Dict[str, Any]:
        """Main method to generate complete karaoke."""
        
        try:
            # Step 1: Get song information
            if progress_callback:
                progress_callback("Getting song information...", 0.1)
            
            song_info = self.get_song_info(song_name, answers)
            
            # Step 2: Get lyrics
            if progress_callback:
                progress_callback("Fetching lyrics...", 0.2)
            
            lyrics = self.get_lyrics(song_info)
            
            # Step 3: Download audio
            if progress_callback:
                progress_callback("Downloading audio...", 0.4)
            
            audio_path = self.download_audio(song_info)
            
            # Step 4: Separate vocals
            if progress_callback:
                progress_callback("Creating instrumental track...", 0.6)
            
            instrumental_path = self.separate_vocals(audio_path)
            
            # Step 5: Create karaoke video
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
            return {
                'success': False,
                'error': str(e)
            }
