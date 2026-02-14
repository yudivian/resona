import streamlit as st
import os
import logging
from src.ui.session import SessionManager
from src.models import SourceType

logger = logging.getLogger(__name__)

# Standard ISO language mapping for the TTS engines
SUPPORTED_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Chinese": "zh", "Japanese": "ja",
    "Korean": "ko", "Russian": "ru"
}

def init_page():
    """
    Sets up the global Streamlit page configuration and executes the deferred 
    UI reset mechanism.
    
    This must be the first function called in the entry point script. The reset 
    logic is placed here to ensure that session state keys are deleted BEFORE 
    any widget is instantiated in the current rerun, preventing 
    StreamlitAPIExceptions.
    """
    st.set_page_config(
        page_title="Resona CAD",
        page_icon="🎛️",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Deferred reset execution to ensure clean widget instantiation
    if st.session_state.get("reset_pending", False):
        _perform_actual_reset()
        st.session_state["reset_pending"] = False

    st.markdown("""
        <style>
        .main { background-color: #0e1117; }
        .stMetric { background-color: #1e2130; padding: 10px; border-radius: 5px; }
        .stTextArea textarea { font-size: 16px; }
        div.stButton > button:first-child { border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)

def _perform_actual_reset():
    """
    Physically deletes specific keys from the Streamlit session state.
    
    By removing these keys before the script reaches the widget definitions, 
    Streamlit is forced to render them with their default values, achieving 
    a "blank" UI state.
    """
    keys_to_reset = [
        "track_A_ready", "track_B_ready", "p_A", "p_B", "ref_A", "ref_B",
        "up_A", "up_B", "mic_A", "mic_B", "m_A", "m_B", "lib_A", "lib_B",
        "tts_final", "preview_txt_A", "preview_txt_B", "save_name", "save_desc", "save_tags",
        "mix_checkbox", "blend_slider"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    # Re-initialize generation flags
    st.session_state["track_A_ready"] = False
    st.session_state["track_B_ready"] = False

def _trigger_ui_reset():
    """
    Sets the reset flag to 'True' to trigger a full UI cleanup on the next 
    script iteration.
    """
    st.session_state["reset_pending"] = True

@st.dialog("🔄 Reset Workflow")
def _dialog_reset_workflow(session: SessionManager):
    """
    Displays a confirmation dialog for resetting the entire design workflow.
    
    Args:
        session (SessionManager): The active session manager to perform cleanup.
    """
    st.warning("All unsaved progress and temporary files will be permanently lost.")
    if st.button("Confirm Reset", type="primary", use_container_width=True):
        session.reset_workflow()
        _trigger_ui_reset()
        st.rerun()

@st.dialog("☢️ Purge Storage")
def _dialog_purge_storage(session: SessionManager):
    """
    Displays a high-risk confirmation dialog to wipe all temporary files 
    stored on the server's disk.
    
    Args:
        session (SessionManager): The active session manager to perform the purge.
    """
    st.error("DANGER: This action deletes ALL files in the temporary directory.")
    if st.button("Purge Everything", type="primary", use_container_width=True):
        session.purge_temp_storage()
        _trigger_ui_reset()
        st.rerun()

def render_debug_panel(session: SessionManager):
    """
    Renders the sidebar with system health metrics and session maintenance tools.
    
    Args:
        session (SessionManager): The session controller for system state access.
    """
    with st.sidebar:
        st.header("🛠️ System Status")
        if hasattr(session, 'config'):
            st.caption(f"Device: {session.config.system.compute.tts_device}")
            st.caption(f"DB Path: {os.path.basename(session.config.paths.db_file)}")
        st.markdown("---")
        
        if st.button("🔄 Reset Workflow", use_container_width=True):
            _dialog_reset_workflow(session)
            
        st.markdown("### Maintenance")
        if st.button("🗑️ Purge Temp Storage", use_container_width=True):
            _dialog_purge_storage(session)

def _save_temp_file(session: SessionManager, uploaded_file, prefix: str) -> str:
    """
    Saves an uploaded file to the session's temp directory and registers it.
    
    Args:
        session (SessionManager): The session to register the file for cleanup.
        uploaded_file: The Streamlit UploadedFile object.
        prefix (str): Prefix to avoid filename collisions (e.g., track ID).
        
    Returns:
        str: Absolute path to the saved file.
    """
    temp_dir = session.config.paths.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    filename = getattr(uploaded_file, 'name', 'input_audio.wav')
    file_path = os.path.join(temp_dir, f"{prefix}_{filename}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    session.register_temp_file(file_path)
    return file_path

def _mark_track_dirty(track_id: str):
    """
    Callback to set the track status as 'not ready' when inputs change.
    
    Args:
        track_id (str): The identifier ('A' or 'B') of the track.
    """
    st.session_state[f"track_{track_id}_ready"] = False

def _ensure_track_identity(session: SessionManager, track_id: str) -> bool:
    """
    Ensures that the specific track has a materialized voice identity.
    Triggers Just-In-Time generation based on the selected mode.
    
    Args:
        session (SessionManager): The controller for generation engines.
        track_id (str): The identifier ('A' or 'B') of the track.
        
    Returns:
        bool: True if identity is successfully generated or already exists.
    """
    if st.session_state.get(f"track_{track_id}_ready", False):
        return True

    mode_key = f"mode_{track_id}"
    mode = st.session_state.get(mode_key, SourceType.DESIGN.value)

    try:
        if mode == SourceType.DESIGN.value:
            prompt = st.session_state.get(f"p_{track_id}")
            if not prompt:
                st.warning(f"Track {track_id}: Missing design prompt.")
                return False
            with st.spinner(f"Designing Identity {track_id}..."):
                session.design_voice(track_id, prompt)

        elif mode == SourceType.CLONE.value:
            method = st.session_state.get(f"m_{track_id}")
            audio = st.session_state.get(f"up_{track_id}") if method == "Upload" else st.session_state.get(f"mic_{track_id}")
            ref_text = st.session_state.get(f"ref_{track_id}")
            if not audio or not ref_text:
                st.warning(f"Track {track_id}: Audio and transcript required for cloning.")
                return False
            path = _save_temp_file(session, audio, f"clone_{track_id}")
            with st.spinner(f"Extracting Identity {track_id}..."):
                session.clone_voice(track_id, path, transcript=ref_text)

        elif mode == "library":
            voice_id = st.session_state.get(f"lib_{track_id}")
            if not voice_id:
                st.warning(f"Track {track_id}: Please select a voice from library.")
                return False
            session.load_voice_from_library(track_id, voice_id)

        st.session_state[f"track_{track_id}_ready"] = True
        return True
    except Exception as e:
        st.error(f"Track {track_id} generation error: {e}")
        return False

def _render_track_config(session: SessionManager, track_id: str):
    """
    Renders the configuration interface for a specific voice track.
    
    Args:
        session (SessionManager): The controller to fetch library data.
        track_id (str): The identifier ('A' or 'B') of the track.
    """
    st.markdown(f"### 🎚️ Track {track_id}")
    mode_key = f"mode_{track_id}"
    options = [SourceType.DESIGN.value, SourceType.CLONE.value, "library"]
    st.selectbox(
        f"Source Type ({track_id})", 
        options, 
        key=mode_key, 
        format_func=lambda x: x.upper(), 
        on_change=_mark_track_dirty, 
        args=(track_id,)
    )
    
    current_mode = st.session_state[mode_key]
    
    if current_mode == SourceType.DESIGN.value:
        st.text_area("Prompt", key=f"p_{track_id}", height=100, on_change=_mark_track_dirty, args=(track_id,))
        
    elif current_mode == SourceType.CLONE.value:
        m = st.radio(f"Input ({track_id})", ["Upload", "Mic"], key=f"m_{track_id}", horizontal=True, on_change=_mark_track_dirty, args=(track_id,))
        if m == "Upload":
            st.file_uploader("Audio File", type=["wav", "mp3"], key=f"up_{track_id}", on_change=_mark_track_dirty, args=(track_id,))
        else:
            st.audio_input("Record Voice", key=f"mic_{track_id}", on_change=_mark_track_dirty, args=(track_id,))
        st.text_area("Reference Text", key=f"ref_{track_id}", height=80, on_change=_mark_track_dirty, args=(track_id,))
        
    elif current_mode == "library":
        voices = session.list_library_voices() # Expected to return Dict[name, id]
        st.selectbox("Select Voice", list(voices.keys()) if voices else [], key=f"lib_{track_id}", on_change=_mark_track_dirty, args=(track_id,))
    
    st.caption(f"🔊 Test Track {track_id}")
    phrase = st.text_input("Test Phrase", value="Testing this voice identity.", key=f"preview_txt_{track_id}")
    if st.button(f"Play Track {track_id}", key=f"btn_prev_{track_id}", use_container_width=True):
        if phrase and _ensure_track_identity(session, track_id):
            alpha = 0.0 if track_id == "A" else 1.0
            with st.spinner("Synthesizing..."):
                try:
                    path = session.preview_voice(phrase, blend_alpha=alpha)
                    st.audio(path, format="audio/wav")
                except Exception as e:
                    st.error(f"Preview Error: {e}")
    st.markdown("---")

def render_studio(session: SessionManager):
    """
    Main Studio Interface. Orchestrates track configuration, blending 
    operations, and final persistence to the library.
    
    Args:
        session (SessionManager): The central controller for the application.
    """
    st.subheader("🌍 Project Settings")
    lang_name = st.selectbox("Global Language", list(SUPPORTED_LANGUAGES.keys()), key="project_lang")
    lang_code = SUPPORTED_LANGUAGES[lang_name]
    session.set_language(lang_code)
    st.markdown("---")
    
    _render_track_config(session, "A")
    
    is_mix = st.checkbox("🔗 Mix with Track B", value=False, key="mix_checkbox")
    alpha = 0.5
    if is_mix:
        _render_track_config(session, "B")
        st.markdown("### 🎛️ Blending Control")
        alpha = st.slider("Mix Balance (A to B)", 0.0, 1.0, 0.5, 0.1, key="blend_slider")

    st.markdown("### 🗣️ Final Performance")
    final_text = st.text_area("Synthesis Text", height=150, key="tts_final")
    if st.button("🚀 Generate Final Audio", type="primary", use_container_width=True):
        if not final_text:
            st.warning("Please enter text to synthesize.")
        else:
            ready_a = _ensure_track_identity(session, "A")
            ready_b = _ensure_track_identity(session, "B") if is_mix else True
            if ready_a and ready_b:
                with st.spinner("Synthesizing final performance..."):
                    try:
                        path = session.preview_voice(final_text, blend_alpha=alpha if is_mix else None)
                        st.audio(path, format="audio/wav")
                    except Exception as e:
                        st.error(f"Synthesis Error: {e}")

    # --- PERSISTENCE SECTION ---
    with st.expander("💾 Save Voice to Library"):
        name = st.text_input("Voice Profile Name", key="save_name")
        description = st.text_area("Visible Notes", key="save_desc")
        tags_raw = st.text_input("Tags (comma separated)", key="save_tags")
        
        if st.button("Confirm Save", use_container_width=True):
            if not name:
                st.warning("A profile name is required for the library.")
            else:
                # Ensure identities are materialized (JIT)
                ready_a = _ensure_track_identity(session, "A")
                ready_b = _ensure_track_identity(session, "B") if is_mix else True
                
                if ready_a and ready_b:
                    try:
                        # 1. Build Rich Semantic Context for Vector Indexing
                        # Consolidates human notes + technical prompts for semantic search
                        context = [f"Name: {name}", f"Note: {description}", f"Tags: {tags_raw}"]
                        if st.session_state.get("mode_A") == SourceType.DESIGN.value:
                            context.append(f"Prompt A: {st.session_state.get('p_A')}")
                        if is_mix and st.session_state.get("mode_B") == SourceType.DESIGN.value:
                            context.append(f"Prompt B: {st.session_state.get('p_B')}")
                        
                        semantic_index_text = ". ".join(context)

                        meta = {
                            "blend": alpha if is_mix else None,
                            "source_type": SourceType.BLEND if is_mix else SourceType.DESIGN,
                            "language": lang_code,
                            "semantic_index": semantic_index_text
                        }
                        
                        # 2. Call Session Manager for Persistence Orchestration
                        session.save_session_voice(
                            name=name,
                            description=description,
                            tags=[t.strip() for t in tags_raw.split(",")] if tags_raw else [],
                            metadata=meta
                        )
                        
                        st.success(f"Voice '{name}' successfully persisted and indexed.")
                        
                        # 3. Clean UI and Reset Workflow
                        _trigger_ui_reset()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Persistence Error: {e}")