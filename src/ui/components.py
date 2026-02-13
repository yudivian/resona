import streamlit as st
import os
from src.ui.session import SessionManager
from src.models import SourceType

SUPPORTED_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Chinese": "zh", "Japanese": "ja",
    "Korean": "ko", "Russian": "ru"
}

def init_page():
    """
    Configures the global Streamlit page settings.
    """
    st.set_page_config(
        page_title="Resona CAD",
        page_icon="🎛️",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    st.markdown("""
        <style>
        .main { background-color: #0e1117; }
        .stMetric { background-color: #1e2130; padding: 10px; border-radius: 5px; }
        .stTextArea textarea { font-size: 16px; }
        div.stButton > button:first-child { border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)

def render_debug_panel(session: SessionManager):
    """
    Renders the debug information in the sidebar.

    Args:
        session (SessionManager): The active session manager.
    """
    with st.sidebar:
        st.header("🛠️ System Status")
        if hasattr(session, 'config'):
            st.caption(f"Device: {session.config.system.compute.tts_device}")
            st.caption(f"DB: {os.path.basename(session.config.paths.db_file)}")
        
        if st.button("Reset Session", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

def _save_temp_file(uploaded_file, prefix: str) -> str:
    """
    Helper to save uploaded or recorded files to a temp directory.

    Args:
        uploaded_file: Streamlit file object.
        prefix (str): Filename prefix.

    Returns:
        str: Absolute path to the saved file.
    """
    os.makedirs("temp", exist_ok=True)
    filename = getattr(uploaded_file, 'name', 'audio.wav')
    file_path = os.path.join("temp", f"{prefix}_{filename}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def _mark_track_dirty(track_id: str):
    """
    Callback to invalidate track readiness.

    Args:
        track_id (str): 'A' or 'B'.
    """
    st.session_state[f"track_{track_id}_ready"] = False

def _ensure_track_identity(session: SessionManager, track_id: str):
    """
    Generates the identity (Vector) only if not ready.

    Args:
        session (SessionManager): Active session.
        track_id (str): 'A' or 'B'.
    """
    if st.session_state.get(f"track_{track_id}_ready", False):
        return

    mode_key = f"mode_{track_id}"
    if mode_key not in st.session_state: return
    current_mode = st.session_state[mode_key]

    if current_mode == SourceType.DESIGN.value:
        prompt = st.session_state.get(f"p_{track_id}")
        if not prompt: raise ValueError(f"Track {track_id}: Design prompt required.")
        session.design_voice(track_id, prompt, seed=None)

    elif current_mode == SourceType.CLONE.value:
        method = st.session_state.get(f"m_{track_id}")
        audio_file = st.session_state.get(f"up_{track_id}") if method == "Upload" else st.session_state.get(f"mic_{track_id}")
        ref_text = st.session_state.get(f"ref_{track_id}")
        
        if not audio_file: raise ValueError(f"Track {track_id}: No audio provided.")
        if not ref_text: raise ValueError(f"Track {track_id}: Reference text required.")
        
        path = _save_temp_file(audio_file, f"clone_{track_id}")
        session.clone_voice(track_id, path, transcript=ref_text)

    elif current_mode == "library":
        voice_id = st.session_state.get(f"lib_{track_id}")
        if not voice_id: raise ValueError(f"Track {track_id}: No voice selected.")
        session.load_voice_from_library(track_id, voice_id)

    st.session_state[f"track_{track_id}_ready"] = True

def _render_track_config(session: SessionManager, track_id: str):
    """
    Renders the configuration interface for a single track.

    Args:
        session (SessionManager): Active session.
        track_id (str): 'A' or 'B'.
    """
    st.markdown(f"### 🎚️ Track {track_id}")
    
    mode_key = f"mode_{track_id}"
    options = [SourceType.DESIGN.value, SourceType.CLONE.value, "library"]
    
    st.selectbox(
        f"Source Type ({track_id})", options, key=mode_key,
        format_func=lambda x: x.upper(),
        on_change=_mark_track_dirty, args=(track_id,)
    )
    current_mode = st.session_state[mode_key]

    if current_mode == SourceType.DESIGN.value:
        st.text_area("Design Prompt", key=f"p_{track_id}", height=100, on_change=_mark_track_dirty, args=(track_id,))

    elif current_mode == SourceType.CLONE.value:
        method = st.radio(f"Input ({track_id})", ["Upload", "Mic"], key=f"m_{track_id}", horizontal=True, on_change=_mark_track_dirty, args=(track_id,))
        if method == "Upload":
            st.file_uploader("Audio", type=["wav", "mp3"], key=f"up_{track_id}", on_change=_mark_track_dirty, args=(track_id,))
        else:
            st.audio_input("Record", key=f"mic_{track_id}", on_change=_mark_track_dirty, args=(track_id,))
        st.text_area("Reference Text", key=f"ref_{track_id}", height=80, help="Mandatory.", on_change=_mark_track_dirty, args=(track_id,))

    elif current_mode == "library":
        voices = session.list_library_voices()
        st.selectbox("Select Voice", list(voices.keys()) if voices else [], key=f"lib_{track_id}", on_change=_mark_track_dirty, args=(track_id,))

    st.caption(f"🔊 Test Track {track_id}")
    c1, c2 = st.columns([3, 1])
    with c1:
        phrase = st.text_input("Test Phrase", value="Hello check.", key=f"preview_txt_{track_id}", label_visibility="collapsed")
    with c2:
        if st.button("Play", key=f"btn_prev_{track_id}"):
            if phrase:
                with st.spinner("Generating..."):
                    try:
                        _ensure_track_identity(session, track_id)
                        alpha = 0.0 if track_id == "A" else 1.0
                        path = session.preview_voice(phrase, blend_alpha=alpha)
                        st.audio(path, format="audio/wav")
                    except Exception as e:
                        st.error(f"Error: {e}")
    st.markdown("---")

def render_studio(session: SessionManager):
    """
    Main Studio Interface.

    Args:
        session (SessionManager): Active session.
    """
    st.subheader("🌍 Project Settings")
    lang_name = st.selectbox("Language", list(SUPPORTED_LANGUAGES.keys()))
    lang_code = SUPPORTED_LANGUAGES[lang_name]
    if hasattr(session, 'set_language'): session.set_language(lang_code)

    st.markdown("---")
    _render_track_config(session, "A")
    
    enable_b = st.checkbox("🔗 Mix with Track B", value=False)
    alpha = 0.0
    if enable_b:
        _render_track_config(session, "B")
        st.markdown("### 🎛️ Blending")
        alpha = st.slider("Mix Balance", 0.0, 1.0, 0.5, 0.1)

    st.markdown("### 🗣️ Final Generation")
    final_text = st.text_area("Final Text", height=150, key="tts_final")
    
    if st.button("🚀 Generate Final Audio", type="primary", use_container_width=True):
        if final_text:
            with st.spinner("Synthesizing..."):
                try:
                    _ensure_track_identity(session, "A")
                    if enable_b: _ensure_track_identity(session, "B")
                    path = session.preview_voice(final_text, blend_alpha=alpha if enable_b else None)
                    st.audio(path, format="audio/wav")
                except Exception as e: st.error(f"Error: {e}")

    with st.expander("💾 Save Voice"):
        name = st.text_input("Name")
        tags = st.text_input("Tags")
        if st.button("Save"):
            if name:
                try:
                    src = SourceType.DESIGN
                    if enable_b: src = SourceType.BLEND
                    elif st.session_state.get("mode_A") == SourceType.CLONE.value: src = SourceType.CLONE
                    
                    meta = {
                        "seed": session.get_current_seed(alpha if enable_b else None),
                        "blend": alpha if enable_b else None,
                        "source_type": src,
                        "language": lang_code
                    }
                    session.save_session_voice(name, "", tags.split(",") if tags else [], meta)
                    st.success(f"Saved {name}")
                except Exception as e: st.error(f"Error: {e}")