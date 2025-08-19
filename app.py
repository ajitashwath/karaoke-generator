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
    st.markdown("Generate high-quality instrumental tracks with synchronized multilingual lyrics display!")
    
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
        
        st.header("🆕 What's New")
        st.markdown("""
        **✨ Enhanced Features:**
        - 🎵 **Advanced vocal separation** for cleaner instrumentals
        - 🌏 **Multilingual lyrics support** (Hindi, Tamil, Spanish, etc.)
        - 🎧 **Better audio quality** with improved processing
        - ⏱️ **Smart timing synchronization**
        
        **🔧 Recent Fixes:**
        - Fixed instrumental quality issues
        - Added proper language detection
        - Improved audio processing stability
        - Enhanced lyrics timing accuracy
        """)
        
        st.header("ℹ️ How it works")
        st.markdown("""
        **Step 1:** Enter a song name
        
        **Step 2:** Answer clarifying questions
        
        **Step 3:** Get:
        - 🎵 High-quality instrumental track
        - 📝 Synchronized lyrics in original language
        - 🎤 Real-time karaoke display
        
        **Supported Languages:**
        - English, Hindi, Tamil, Telugu, Malayalam
        - Spanish, French, German, Italian
        - Japanese, Korean, Chinese, Arabic
        - And many more!
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
            col1, col2 = st.columns([3, 1])
            
            with col1:
                song_name = st.text_input(
                    "🎶 Enter the song name:",
                    placeholder="e.g., Shape of You, Tum Hi Ho, Despacito, Gangnam Style...",
                    label_visibility="visible"
                )
            
            with col2:
                st.write("")  # Add spacing
                st.write("")  # Add spacing
                search_button = st.button("🔍 Search Song", type="primary", use_container_width=True)
            
            # Examples section
            st.markdown("**Popular Examples:**")
            example_cols = st.columns(4)
            examples = [
                "Tum Hi Ho", "Shape of You", "Despacito", "Gangnam Style",
                "Vande Mataram", "Perfect", "Let It Go", "Believer"
            ]
            
            for i, example in enumerate(examples):
                with example_cols[i % 4]:
                    if st.button(example, key=f"example_{i}", use_container_width=True):
                        st.session_state.example_song = example
                        st.rerun()
            
            # Handle example selection
            if hasattr(st.session_state, 'example_song'):
                song_name = st.session_state.example_song
                del st.session_state.example_song
                search_button = True
            
            if song_name and search_button:
                with st.spinner("🔍 Searching for song information..."):
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
                answers[f"q{i}"] = st.text_input(
                    question, 
                    key=f"question_{i}",
                    label_visibility="visible"
                )
            
            col_back, col_generate = st.columns([1, 2])
            
            with col_back:
                if st.button("⬅️ Back to Search"):
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
            
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Detailed progress steps
            progress_container = st.container()
            with progress_container:
                step_cols = st.columns(5)
                step_indicators = []
                step_names = ["Song Info", "Lyrics", "Download", "Process", "Timing"]
                
                for i, (col, name) in enumerate(zip(step_cols, step_names)):
                    with col:
                        indicator = st.empty()
                        step_indicators.append((indicator, name))
            
            def enhanced_progress_callback(step, progress):
                progress_bar.progress(progress)
                status_text.text(f"🔄 {step}")
                
                # Update step indicators
                current_step = int(progress * 5)
                for i, (indicator, name) in enumerate(step_indicators):
                    if i < current_step:
                        indicator.markdown(f"✅ **{name}**")
                    elif i == current_step:
                        indicator.markdown(f"🔄 **{name}**")
                    else:
                        indicator.markdown(f"⏳ {name}")
            
            try:
                result = karaoke_gen.generate_karaoke(
                    st.session_state.song_name,
                    st.session_state.answers,
                    progress_callback=enhanced_progress_callback
                )
                st.session_state.result = result
                st.session_state.stage = 'result'
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating karaoke: {str(e)}")
                
                # Error details in expander
                with st.expander("🔍 Error Details"):
                    st.code(str(e))
                    st.markdown("**Possible solutions:**")
                    st.markdown("- Check your internet connection")
                    st.markdown("- Try a different song")
                    st.markdown("- Ensure the song exists on YouTube")
                
                if st.button("🔄 Try Again"):
                    st.session_state.stage = 'clarify'
                    st.rerun()
        
        elif st.session_state.stage == 'result':
            result = st.session_state.result
            
            if result.get('success', False):
                st.success("✅ Karaoke generated successfully!")
                
                # Song information in a nice card
                with st.container():
                    st.markdown("### 🎵 Song Information")
                    info_cols = st.columns(2)
                    
                    with info_cols[0]:
                        st.markdown(f"**🎤 Title:** {result['song_info']['title']}")
                        st.markdown(f"**👨‍🎤 Artist:** {result['song_info']['artist']}")
                        st.markdown(f"**🎼 Genre:** {result['song_info']['genre']}")
                    
                    with info_cols[1]:
                        st.markdown(f"**⏰ Duration:** {result['song_info'].get('duration', 'Unknown')}")
                        st.markdown(f"**🌏 Language:** {result['song_info'].get('language', 'Unknown')}")
                        popularity = result['song_info'].get('popularity', 5)
                        st.markdown(f"**⭐ Popularity:** {'⭐' * popularity}")
                
                st.divider()
                
                # Audio player with enhanced controls
                st.subheader("🎵 Instrumental Track")
                if result.get('instrumental_path'):
                    # Audio quality info
                    st.info("🎧 High-quality instrumental track with advanced vocal separation")
                    
                    with open(result['instrumental_path'], 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        
                        # Audio player
                        st.audio(audio_bytes, format='audio/wav')
                        
                        # Download section
                        download_cols = st.columns([2, 1])
                        with download_cols[0]:
                            st.download_button(
                                label="⬇️ Download Instrumental Track",
                                data=audio_bytes,
                                file_name=f"{st.session_state.song_name}_instrumental.wav",
                                mime="audio/wav",
                                use_container_width=True
                            )
                        with download_cols[1]:
                            file_size = len(audio_bytes) / (1024 * 1024)  # MB
                            st.metric("File Size", f"{file_size:.1f} MB")
                
                st.divider()
                
                # Enhanced synchronized lyrics display
                st.subheader("🎤 Synchronized Lyrics")
                
                # Initialize lyrics timing if not already done
                if 'lyrics_timing' not in st.session_state:
                    st.session_state.lyrics_timing = result.get('timed_lyrics', [])
                
                # Lyrics control panel
                control_cols = st.columns([1, 1, 1, 1])
                
                with control_cols[0]:
                    start_button = st.button("▶️ Start Sync", use_container_width=True)
                    if start_button:
                        st.session_state.lyrics_started = True
                        st.session_state.start_time = time.time()
                        st.rerun()
                
                with control_cols[1]:
                    if st.button("⏸️ Pause", use_container_width=True):
                        if st.session_state.get('lyrics_started', False):
                            st.session_state.paused_time = time.time() - st.session_state.get('start_time', 0)
                            st.session_state.lyrics_started = False
                        st.rerun()
                
                with control_cols[2]:
                    if st.button("▶️ Resume", use_container_width=True):
                        if st.session_state.get('paused_time', 0) > 0:
                            st.session_state.start_time = time.time() - st.session_state.paused_time
                            st.session_state.lyrics_started = True
                            st.session_state.paused_time = 0
                        st.rerun()
                
                with control_cols[3]:
                    if st.button("🔄 Reset", use_container_width=True):
                        st.session_state.lyrics_started = False
                        st.session_state.paused_time = 0
                        if 'start_time' in st.session_state:
                            del st.session_state.start_time
                        if 'paused_time' in st.session_state:
                            del st.session_state.paused_time
                        st.rerun()
                
                # Lyrics display container
                lyrics_container = st.container()
                
                # Display lyrics with highlighting
                if st.session_state.get('lyrics_started', False) and 'start_time' in st.session_state:
                    current_time = time.time() - st.session_state.start_time
                    
                    # Auto-refresh for real-time updates
                    if current_time < 300:  # Limit to 5 minutes to prevent infinite refresh
                        time.sleep(0.1)
                        st.rerun()
                elif st.session_state.get('paused_time', 0) > 0:
                    current_time = st.session_state.paused_time
                else:
                    current_time = 0
                
                with lyrics_container:
                    lyrics_display = st.empty()
                    
                    if st.session_state.lyrics_timing:
                        # Find current and next lyrics
                        current_lyric = ""
                        next_lyric = ""
                        
                        # Fixed: Unpacking 3 values (start_time, end_time, text) instead of 2
                        for i, (start_time, end_time, text) in enumerate(st.session_state.lyrics_timing):
                            if start_time <= current_time < end_time:
                                current_lyric = text
                                if i + 1 < len(st.session_state.lyrics_timing):
                                    next_lyric = st.session_state.lyrics_timing[i + 1][2]  # Get text from next tuple
                                break
                            elif current_time < start_time:
                                if not next_lyric:
                                    next_lyric = text
                                break
                        
                        # Display with enhanced styling
                        if current_lyric or next_lyric:
                            lyrics_html = f"""
                            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin: 10px 0;">
                                <div style="color: white; font-size: 28px; font-weight: bold; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
                                    🎤 {current_lyric if current_lyric else "♪ ♫ ♪"}
                                </div>
                                {f'<div style="color: #e0e0e0; font-size: 18px; opacity: 0.8;">Next: {next_lyric}</div>' if next_lyric else ''}
                                <div style="margin-top: 15px;">
                                    <div style="color: #ffeb3b; font-size: 16px;">⏱️ {current_time:.1f}s</div>
                                </div>
                            </div>
                            """
                            lyrics_display.markdown(lyrics_html, unsafe_allow_html=True)
                        else:
                            lyrics_display.markdown("""
                            <div style="text-align: center; padding: 40px; background: #f5f5f5; border-radius: 10px; margin: 10px 0;">
                                <div style="font-size: 24px; color: #666;">
                                    🎵 Press Start Sync to begin karaoke mode 🎵
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        lyrics_display.info("📝 Lyrics timing data not available")
                
                # Static lyrics display option
                st.subheader("📝 Full Lyrics")
                if result.get('lyrics'):
                    with st.expander("View Full Lyrics", expanded=False):
                        st.text_area(
                            "Lyrics:",
                            value=result['lyrics'],
                            height=300,
                            disabled=True
                        )
                        
                        # Download lyrics option
                        st.download_button(
                            label="⬇️ Download Lyrics",
                            data=result['lyrics'],
                            file_name=f"{st.session_state.song_name}_lyrics.txt",
                            mime="text/plain"
                        )
                
                # Action buttons
                st.divider()
                action_cols = st.columns([1, 1, 1])
                
                with action_cols[0]:
                    if st.button("🎵 Generate Another Song", use_container_width=True):
                        # Clear session state
                        keys_to_keep = []  # Keep API key related session state
                        keys_to_clear = [k for k in st.session_state.keys() if k not in keys_to_keep]
                        for key in keys_to_clear:
                            del st.session_state[key]
                        st.rerun()
                
                with action_cols[1]:
                    if st.button("📤 Share Results", use_container_width=True):
                        share_text = f"🎤 Just generated karaoke for '{st.session_state.song_name}' using AI Karaoke Generator!"
                        st.success("Copy this text to share:")
                        st.code(share_text)
                
                with action_cols[2]:
                    if st.button("⭐ Rate This Song", use_container_width=True):
                        rating = st.select_slider(
                            "How was the karaoke quality?",
                            options=[1, 2, 3, 4, 5],
                            value=5,
                            format_func=lambda x: "⭐" * x
                        )
                        if st.button("Submit Rating"):
                            st.success(f"Thank you for rating: {'⭐' * rating}")
            
            else:
                st.error("❌ Failed to generate karaoke. Please try again.")
                if st.button("🔄 Try Again"):
                    st.session_state.stage = 'clarify'
                    st.rerun()
    
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.info("Please refresh the page and try again.")
        
        with st.expander("🔍 Technical Details"):
            st.code(str(e))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        🎤 <strong>AI Karaoke Generator</strong> - Turn any song into a karaoke experience!<br>
        Made with ❤️ using Streamlit | Support: multilingual lyrics & advanced vocal separation
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()