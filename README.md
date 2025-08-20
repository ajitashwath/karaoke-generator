# 🎤 AI Karaoke Generator
An AI karaoke application that generates high-quality instrumental tracks with synchronized multilingual lyrics display. Built with Streamlit, OpenAI GPT, and advanced audio processing techniques.

## ✨ Features

- **🎵 Real Lyrics**: Fetches accurate lyrics via Genius API
- **🔑 Secure API Key Handling**: Environment-based configuration
- **🌏 Multilingual Support**: Works with songs in multiple languages
- **🎧 High-Quality Audio**: Advanced vocal separation algorithms
- **⏱️ LLM-Powered Timing**: AI-generated synchronized lyrics timing
- **🎤 Real-Time Display**: Live karaoke lyrics synchronization
- **📱 User-Friendly Interface**: Intuitive Streamlit web application

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- FFmpeg (for audio processing)
- OpenAI API key
- Genius API access token

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ajitashwath/karaoke-generator.git
cd karaoke-generator
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```env
OPENAI_API_KEY="your-openai-api-key-here"
GENIUS_ACCESS_TOKEN="your-genius-api-token-here"
```

4. **Install FFmpeg**

**Windows:**
- Download from [FFmpeg official site](https://ffmpeg.org/download.html)
- Add to system PATH

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

5. **Run the application**
```bash
streamlit run app.py
```

## 🔧 API Keys Setup

### OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account or sign in
3. Navigate to API Keys section
4. Create a new secret key
5. Copy the key to your `.env` file

### Genius API Token

1. Visit [Genius API](https://genius.com/api-clients)
2. Create a new API client
3. Generate an access token
4. Copy the token to your `.env` file

## 📖 How It Works

### Step 1: Song Search
- Enter any song name (e.g., "Shape of You", "Despacito", "Tum Hi Ho")
- The system uses AI to generate clarifying questions
- Answer questions to identify the exact song version

### Step 2: Content Generation
1. **Song Information**: AI extracts detailed metadata
2. **Lyrics Retrieval**: Fetches accurate lyrics from Genius API
3. **Audio Download**: Downloads high-quality audio from YouTube
4. **Vocal Separation**: Advanced algorithms remove vocals
5. **Timing Synchronization**: AI generates precise timing data

### Step 3: Karaoke Experience
- Play the instrumental track
- Follow synchronized lyrics display
- Real-time highlighting of current lyrics
- Preview of upcoming lines

## 🏗️ Architecture

```
├── app.py                 # Streamlit web application
├── tool.py               # Core karaoke generation logic
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables (create this)
└── README.md          
```

### Key Components
- **Audio Processing**: Uses librosa and scipy for vocal separation
- **AI Integration**: OpenAI GPT for intelligent song analysis
- **Lyrics API**: Genius API for accurate lyrics retrieval
- **UI Framework**: Streamlit for interactive web interface

## 🎯 Usage Examples

### Popular Song Examples
- **English**: "Blinding Lights", "Shape of You", "Bohemian Rhapsody"
- **Hindi**: "Kesariya", "Channa Mereya", "Tera Hone Laga Hoon", "Kun Faya Kun"
- **Other Languages**: The system supports songs in various languages

### Advanced Features

**Custom Timing Adjustments**
- AI analyzes song structure (verses, chorus, bridges)
- Accounts for instrumental intros and outros
- Adapts timing based on song tempo and style

**Audio Quality Optimization**
- 44.1kHz sampling rate for CD-quality output
- Advanced harmonic-percussive separation
- Dynamic range optimization

## 🛠️ Technical Details

### Audio Processing Pipeline

1. **Download**: High-quality audio extraction via yt-dlp
2. **Analysis**: Librosa-based audio feature extraction
3. **Separation**: Multi-technique vocal isolation:
   - Stereo channel subtraction for simple cases
   - Harmonic-percussive separation for complex audio
   - Spectral masking for vocal removal
4. **Output**: WAV format with 16-bit PCM encoding

### AI-Powered Features

- **Song Identification**: GPT-4 analyzes user input for precise matching
- **Lyrics Timing**: Intelligent synchronization based on song structure
- **Language Detection**: Automatic language identification
- **Quality Assessment**: AI validates lyrics accuracy


## 🔍 Troubleshooting

### Common Issues

**"Missing API Keys" Error**
- Ensure `.env` file exists in project root
- Verify API keys are correctly formatted
- Check for extra spaces or quotes

**Audio Download Failures**
- Verify FFmpeg installation and PATH configuration
- Check internet connection
- Try alternative song search terms

**Poor Vocal Separation Quality**
- Some songs may have vocals mixed in mono
- Try different versions of the same song
- Instrumental versions work best as input

**Lyrics Not Found**
- Verify song title and artist spelling
- Try popular or well-known songs first
- Check Genius.com for lyrics availability

### Performance Tips
- **Close other applications** to free up system resources
- **Use wired internet connection** for faster downloads
- **Choose popular songs** for better lyrics accuracy
- **Allow processing time** for complex audio separation

## 🤝 Contributing
Contributions are welcome! Here are some areas for improvement:
- **Enhanced vocal separation algorithms**
- **Support for additional lyrics sources**
- **Offline mode capabilities**
- **Mobile-responsive design improvements**
- **Batch processing features**

## 📄 License
This project is for educational and personal use. Ensure compliance with:
- YouTube Terms of Service for audio downloading
- Genius API Terms of Service for lyrics
- OpenAI API Usage Policies
- Local copyright laws for audio processing

## 🆘 Support
For issues and questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure API keys have sufficient credits/quota
4. Test with simple, popular songs first

## 🎵 Enjoy Your Karaoke Experience!

Transform any song into a professional karaoke track with the power of AI. Perfect for parties, practice sessions, or just having fun with your favorite music!
