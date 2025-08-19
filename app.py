import streamlit as st
import os
from tool import KaraokeGenerator
import tempfile
import time

def main():
    st.set_page_config(
        page_title="AI Karaoke Generator",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 AI Karaoke Generator")
    st.markdown("Generate instrumental tracks with synchronized lyrics display!")
    
    with st.sidebar:
        st.header("🔑 Configuration")
        openai_api_key = st.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            help="Get your API key from https://platform.openai.com/account/api-keys"
        )
        
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.success("✅ API Key configured!")
        else:
            st.warning("⚠️ Please enter your OpenAI API Key to continue")
        
        st.header("ℹ️ How it works")
        st.markdown("""
        **Step 1:** Enter a song name
        
        **Step 2:** Answer clarifying questions
        
        **Step 3:** Get:
        - 🎵 Instrumental track
        - 📝 Synchronized lyrics display
        
        **Features:**
        - Real-time lyrics synchronization
        - Clean instrumental separation
        - Easy playback controls
        """)
    
    if not openai_api_key:
        st.info("👈 Please enter your OpenAI API Key in the sidebar to get started")
        return
    
    try:
        karaoke_gen = KaraokeGenerator(openai_api_key)
        
        st.header("🎵 Song Request")
        
        # Initialize session state
        if 'stage' not in st.session_state:
            st.session_state.stage = 'input'
            st.session_state.song_info = {}
        
        if st.session_state.stage == 'input':
            song_name = st.text_input(
                "🎶 Enter the song name:",
                placeholder="e.g., Shape of You, Bohemian Rhapsody, Despacito..."
            )
            
            if song_name and st.button("🔍 Search Song", type="primary"):
                with st.spinner("Searching for song information..."):
                    questions = karaoke_gen.get_clarifying_questions(song_name)
                    st.session_state.questions = questions
                    st.session_state.song_name = song_name
                    st.session_state.stage = 'clarify'
                    st.rerun()
        
        elif st.session_state.stage == 'clarify':
            st.info(f"🎵 Song: **{st.session_state.song_name}**")
            st.subheader("📋 Please provide additional details:")
            answers = {}
            for i, question in enumerate(st.session_state.questions):
                answers[f"q{i}"] = st.text_input(question, key=f"question_{i}")
            
            col_back, col_generate = st.columns([1, 2])
            
            with col_back:
                if st.button("⬅️ Back"):
                    st.session_state.stage = 'input'
                    st.rerun()
            
            with col_generate:
                if st.button("🎤 Generate Karaoke", type="primary"):
                    if all(answers.values()):
                        st.session_state.answers = answers
                        st.session_state.stage = 'generate'
                        st.rerun()
                    else:
                        st.error("Please answer all questions before proceeding.")
        
        elif st.session_state.stage == 'generate':
            st.info(f"🎵 Generating karaoke for: **{st.session_state.song_name}**")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                result = karaoke_gen.generate_karaoke(
                    st.session_state.song_name,
                    st.session_state.answers,
                    progress_callback=lambda step, progress: (
                        progress_bar.progress(progress),
                        status_text.text(step)
                    )
                )
                st.session_state.result = result
                st.session_state.stage = 'result'
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating karaoke: {str(e)}")
                if st.button("🔄 Try Again"):
                    st.session_state.stage = 'clarify'
                    st.rerun()
        
        elif st.session_state.stage == 'result':
            result = st.session_state.result
            
            if result.get('success', False):
                st.success("✅ Karaoke generated successfully!")
                
                # Song information
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Title:** {result['song_info']['title']}")
                    st.write(f"**Artist:** {result['song_info']['artist']}")
                with col2:
                    st.write(f"**Genre:** {result['song_info']['genre']}")
                    st.write(f"**Duration:** {result['song_info']['duration']}")
                
                st.divider()
                
                # Audio player
                st.subheader("🎵 Instrumental Track")
                if result.get('instrumental_path'):
                    with open(result['instrumental_path'], 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/wav')
                        
                        # Download button for instrumental
                        st.download_button(
                            label="⬇️ Download Instrumental Track",
                            data=audio_bytes,
                            file_name=f"{st.session_state.song_name}_instrumental.wav",
                            mime="audio/wav"
                        )
                
                st.divider()
                
                # Synchronized lyrics display
                st.subheader("🎤 Synchronized Lyrics")
                
                # Initialize lyrics timing if not already done
                if 'lyrics_timing' not in st.session_state:
                    st.session_state.lyrics_timing = result.get('timed_lyrics', [])
                
                # Lyrics display container
                lyrics_container = st.container()
                
                # Control buttons
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("▶️ Start Lyrics Sync"):
                        st.session_state.lyrics_started = True
                        st.session_state.start_time = time.time()
                        st.rerun()
                
                with col2:
                    if st.button("⏹️ Stop"):
                        st.session_state.lyrics_started = False
                        st.rerun()
                
                with col3:
                    if st.button("🔄 Reset"):
                        st.session_state.lyrics_started = False
                        if 'start_time' in st.session_state:
                            del st.session_state.start_time
                        st.rerun()
                
                # Display lyrics
                with lyrics_container:
                    if st.session_state.get('lyrics_started', False) and 'start_time' in st.session_state:
                        # Calculate current time
                        current_time = time.time() - st.session_state.start_time
                        
                        # Find current lyric
                        current_lyric = "🎵 Music playing..."
                        next_lyric = ""
                        
                        for i, (start_time, end_time, text) in enumerate(st.session_state.lyrics_timing):
                            if start_time <= current_time < end_time:
                                current_lyric = text
                                # Get next lyric for preview
                                if i + 1 < len(st.session_state.lyrics_timing):
                                    next_lyric = st.session_state.lyrics_timing[i + 1][2]
                                break
                        
                        # Display current lyric prominently
                        st.markdown(f"""
                        <div style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 30px; border-radius: 15px; margin: 20px 0;'>
                            <h2 style='color: white; margin: 0; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                                {current_lyric}
                            </h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show next lyric preview
                        if next_lyric:
                            st.markdown(f"""
                            <div style='text-align: center; background: rgba(255,255,255,0.1); 
                                        padding: 15px; border-radius: 10px; margin: 10px 0;'>
                                <p style='color: #666; margin: 0; font-size: 1.2em; font-style: italic;'>
                                    Next: {next_lyric}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Auto-refresh for real-time updates
                        time.sleep(0.5)
                        st.rerun()
                        
                    else:
                        # Show all lyrics when not playing
                        st.markdown("**Full Lyrics:**")
                        lyrics_text = "\n".join([text for _, _, text in st.session_state.lyrics_timing])
                        st.text_area("", lyrics_text, height=300, disabled=True)
                
                st.divider()
                
                if st.button("🎵 Generate Another Song"):
                    # Clear all session state
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.session_state.stage = 'input'
                    st.rerun()
            
            else:
                st.error(f"❌ Error generating karaoke: {result.get('error', 'Unknown error')}")
                if st.button("🔄 Try Again"):
                    st.session_state.stage = 'clarify'
                    st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error initializing application: {str(e)}")

if __name__ == "__main__":
    main()