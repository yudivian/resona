import streamlit as st
import os
import logging
import random
from src.ui.session import SessionManager
from src.models import SourceType

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "Spanish": "es", "English": "en",  "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Chinese": "zh", "Japanese": "ja",
    "Korean": "ko", "Russian": "ru"
}

def init_page():
    """
    Configures the global Streamlit page settings and orchestrates the application's
    lifecycle management, including deferred resets and global notification handling.

    This function serves as the entry point for the UI rendering pipeline. It must be
    invoked before any other Streamlit widget instantiation. Its responsibilities include:
    1. Setting page metadata.
    2. Processing the 'reset_pending' state pattern to cleanly wipe the interface inputs.
    3. Intercepting and displaying global toast notifications triggered by modal dialogs
       after a script rerun. This centralization ensures that ephemeral messages are 
       not lost during the context switch between a dialog closing and the main page reloading.
    """
    st.set_page_config(
        page_title="Resona CAD",
        page_icon="🎛️",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    if st.session_state.get("toast_save_success", False):
        st.toast("Voice successfully saved to library.", icon="💾")
        st.session_state["toast_save_success"] = False
        
    if st.session_state.get("toast_preview_success", False):
        st.toast("Preview generated successfully.", icon="🔊")
        st.session_state["toast_preview_success"] = False

    if st.session_state.get("toast_final_success", False):
        st.toast("Final audio generated successfully.", icon="🚀")
        st.session_state["toast_final_success"] = False

    if st.session_state.get("reset_pending", False):
        _perform_actual_reset()
        st.session_state["reset_pending"] = False

    st.markdown("""
        <style>
        .main { background-color: #0e1117; }
        .stMetric { background-color: #1e2130; padding: 10px; border-radius: 5px; }
        div.stButton > button:first-child { border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)

def _perform_actual_reset():
    """
    Executes a hard reset of the user session by physically removing specific keys
    from the Streamlit session state dictionary.

    This method is the enforcement mechanism for the 'Reset Workflow' feature. By
    deleting keys related to track readiness, prompts, uploaded files, and generated
    audio paths, it forces Streamlit to re-instantiate the corresponding widgets
    with their default values on the subsequent render pass.
    """
    keys_to_reset = [
        "track_A_ready", "track_B_ready", "p_A", "p_B", "ref_A", "ref_B",
        "up_A", "mic_A", "up_B", "mic_B", "m_A", "m_B", "lib_A", "lib_B",
        "save_name", "save_desc", "save_tags", "mix_checkbox", "blend_slider",
        "tts_final", "preview_txt_A", "preview_txt_B", 
        "seed_A", "seed_B",
        "last_audio_A", "last_audio_B", "last_final_audio"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state["track_A_ready"] = False
    st.session_state["track_B_ready"] = False

@st.dialog("🔄 Reset Workflow")
def _dialog_reset_workflow(session: SessionManager):
    """
    Displays a modal confirmation dialog to safeguard the destructive 'Reset' action.

    This dialog acts as a UX barrier. It clarifies that this action resets the
    current session state and clears session-specific temporary files, but does
    not affect the persistent database or other system files.
    """
    st.warning("All unsaved progress will be lost. This cannot be undone.")
    if st.button("Confirm Reset", type="primary", use_container_width=True):
        session.reset_workflow()
        st.session_state["reset_pending"] = True
        st.rerun()

@st.dialog("☢️ Purge Storage")
def _dialog_purge_storage(session: SessionManager):
    """
    Displays a high-stakes modal confirmation dialog for purging physical storage.

    This action is strictly limited to temporary files (audio artifacts, caches)
    located on the disk. It explicitly DOES NOT affect the BeaverDB database file,
    ensuring that saved voice profiles remain safe. It triggers a session reset
    after purging to prevent state inconsistencies.
    """
    st.error("DANGER: This will permanently delete ALL temporary files on disk.")
    if st.button("Purge Everything", type="primary", use_container_width=True):
        session.purge_temp_storage()
        st.session_state["reset_pending"] = True
        st.rerun()

@st.dialog("🔊 Preview Generation")
def _dialog_preview_execution(session: SessionManager, track_id: str, phrase: str):
    """
    Orchestrates the voice preview synthesis process within a blocking modal dialog.

    This function implements a 'Process-and-Return' architecture. It performs the
    heavy lifting (inference) inside the modal to block UI interaction, but
    delegates the result rendering to the main page via session state persistence.
    This prevents the audio player from vanishing when the dialog closes.
    """
    st.markdown(f"**Track {track_id}** processing...")
    
    try:
        with st.status("Synthesizing...", expanded=True) as status:
            st.write("Verifying voice identity...")
            if _ensure_track_identity(session, track_id):
                st.write("Rendering audio waveform...")
                alpha = 0.0 if track_id == "A" else 1.0
                output_path = session.preview_voice(phrase, blend_alpha=alpha)
                
                st.session_state[f"last_audio_{track_id}"] = output_path
                st.session_state["toast_preview_success"] = True
                
                status.update(label="Complete!", state="complete", expanded=False)
                st.rerun()
            else:
                status.update(label="Failed", state="error")
                st.error("Identity generation failed.")
    except Exception as e:
        st.error(f"Preview Failed: {e}")

@st.dialog("🚀 Final Audio Generation")
def _dialog_final_execution(session: SessionManager, text: str, is_mix: bool, alpha: float):
    """
    Manages the high-fidelity synthesis of the final output within a blocking modal.

    It ensures validation of both tracks (if mixing is enabled) before proceeding.
    The result path is stored in 'last_final_audio' and the main interface is 
    reloaded to display the result persistently.
    """
    st.markdown("Generating high-quality output...")
    
    try:
        with st.status("Processing Workflow...", expanded=True) as status:
            st.write("Validating Track A...")
            ready_a = _ensure_track_identity(session, "A")
            
            ready_b = True
            if is_mix:
                st.write("Validating Track B...")
                ready_b = _ensure_track_identity(session, "B")
            
            if ready_a and ready_b:
                st.write("Synthesizing final performance...")
                output_path = session.preview_voice(text, blend_alpha=alpha if is_mix else None)
                
                st.session_state["last_final_audio"] = output_path
                st.session_state["toast_final_success"] = True
                
                status.update(label="Done!", state="complete", expanded=False)
                st.rerun()
            else:
                status.update(label="Identity Validation Failed", state="error")
                st.error("Could not materialize one or more voice identities.")
    except Exception as e:
        st.error(f"Synthesis Error: {e}")

@st.dialog("💾 Save to Library")
def _dialog_save_execution(session: SessionManager, name: str, desc: str, tags: list, meta: dict):
    """
    Executes the database persistence transaction within a secure, blocking modal.

    This function performs a critical re-verification of the engine state via
    `_ensure_track_identity` to prevent 'empty identity' errors that can occur
    if the session was reset prior to saving. Upon success, it flags the workspace
    for a reset and triggers a notification toast.
    """
    st.markdown(f"Saving **{name}**...")
    
    try:
        with st.status("Persisting Data...", expanded=True) as status:
            st.write("Materializing identities...")
            ready_a = _ensure_track_identity(session, "A")
            
            is_mix = meta.get("blend") is not None
            ready_b = _ensure_track_identity(session, "B") if is_mix else True
            
            if ready_a and ready_b:
                st.write("Generating semantic indices...")
                st.write("Writing to BeaverDB...")
                session.save_session_voice(name, desc, tags, meta)
                
                status.update(label="Saved!", state="complete", expanded=False)
                
                st.session_state["reset_pending"] = True
                st.session_state["toast_save_success"] = True
                st.rerun()
            else:
                status.update(label="Validation Failed", state="error")
                st.error("Cannot save: Voice identities are not ready.")
    except Exception as e:
        st.error(f"Persistence Error: {e}")

def render_debug_panel(session: SessionManager):
    """
    Renders the collapsible sidebar panel containing system health metrics and
    maintenance controls.
    """
    with st.sidebar:
        st.header("🛠️ System Status")
        if hasattr(session, 'config'):
            st.caption(f"Device: {session.config.system.compute.tts_device}")
            st.caption(f"DB: {os.path.basename(session.config.paths.db_file)}")
        st.markdown("---")
        
        if st.button("🔄 Reset Workflow", use_container_width=True):
            _dialog_reset_workflow(session)
            
        if st.button("🗑️ Purge Temp Files", use_container_width=True):
            _dialog_purge_storage(session)

def _save_temp_file(session: SessionManager, uploaded_file, prefix: str) -> str:
    """
    Persists a raw Streamlit UploadedFile object to the local filesystem and
    registers it for session-based cleanup.
    """
    temp_dir = session.config.paths.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    filename = getattr(uploaded_file, 'name', 'audio.wav')
    file_path = os.path.join(temp_dir, f"{prefix}_{filename}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    session.register_temp_file(file_path)
    return file_path

def _mark_track_dirty(track_id: str):
    """
    Callback function to invalidate the readiness state of a track upon user input.
    Also clears any stale audio results associated with the modified track.
    """
    st.session_state[f"track_{track_id}_ready"] = False
    
    key_audio = f"last_audio_{track_id}"
    if key_audio in st.session_state:
        del st.session_state[key_audio]

def _ensure_track_identity(session: SessionManager, track_id: str) -> bool:
    """
    Guarantees the materialization of a valid voice identity for the specified track,
    handling state recovery and JIT generation.

    This function implements a paranoid dual-check strategy. It verifies not just
    the UI 'ready' flag, but also the physical memory state of the backend engine.
    If the engine is empty (e.g. after a reset), it forces a silent regeneration
    using the stored parameters. This robustly prevents 'Target engine has no active
    identity' errors.
    """
    engine = session.engine_a if track_id == "A" else session.engine_b
    is_hydrated = engine.active_identity is not None
    is_flagged_ready = st.session_state.get(f"track_{track_id}_ready", False)

    if is_flagged_ready and is_hydrated:
        return True

    mode_key = f"mode_{track_id}"
    mode = st.session_state.get(mode_key, SourceType.DESIGN.value)

    try:
        if mode == SourceType.DESIGN.value:
            prompt = st.session_state.get(f"p_{track_id}")
            if not prompt:
                st.error(f"Track {track_id}: Missing design prompt.")
                return False
            
            seed_key = f"seed_{track_id}"
            if seed_key not in st.session_state:
                st.session_state[seed_key] = random.randint(0, 999999)
            current_seed = st.session_state[seed_key]

            st.write(f"Designing Track {track_id} (Seed: {current_seed})...")
            session.design_voice(track_id, prompt, seed=current_seed)

        elif mode == SourceType.CLONE.value:
            method = st.session_state.get(f"m_{track_id}")
            audio = st.session_state.get(f"up_{track_id}") if method == "Upload" else st.session_state.get(f"mic_{track_id}")
            ref = st.session_state.get(f"ref_{track_id}")
            
            if not audio or not ref:
                st.error(f"Track {track_id}: Missing audio or transcript.")
                return False
                
            path = _save_temp_file(session, audio, f"clone_{track_id}")
            st.write(f"Cloning identity for Track {track_id}...")
            session.clone_voice(track_id, path, ref)
        
        elif mode == "library":
            selected_name = st.session_state.get(f"lib_{track_id}")
            if not selected_name:
                st.error(f"Track {track_id}: No voice selected.")
                return False
            
            voices_map = session.list_library_voices()
            voice_id = voices_map.get(selected_name)
            
            if not voice_id:
                st.error(f"Voice '{selected_name}' not found in registry.")
                return False
                
            st.write(f"Loading '{selected_name}' from library...")
            session.load_voice_from_library(track_id, voice_id)
        
        st.session_state[f"track_{track_id}_ready"] = True
        return True

    except Exception as e:
        st.error(f"Error restoring Track {track_id}: {e}")
        return False

def _render_track_config(session: SessionManager, track_id: str):
    """
    Renders the configuration widget block for a single voice track (A or B).

    This function implements Progressive Disclosure logic for the 'Library' source.
    If the database is empty, the 'Library' option is dynamically removed from
    the source dropdown. It also performs state sanitization: if the user was 
    previously in 'Library' mode but the library is now empty, it automatically 
    reverts the state to 'Design' to prevent UI crashes.
    """
    st.markdown(f"##### 🎚️ Track {track_id}")
    
    with st.container(border=True):
        mode_key = f"mode_{track_id}"
        
        voices = session.list_library_voices()
        
        options = [SourceType.DESIGN.value, SourceType.CLONE.value]
        if voices:
            options.append("library")
        
        current_mode = st.session_state.get(mode_key, SourceType.DESIGN.value)
        
        if current_mode == "library" and not voices:
            current_mode = SourceType.DESIGN.value
            st.session_state[mode_key] = current_mode

        st.selectbox(
            f"Source ({track_id})", 
            options, 
            key=mode_key, 
            format_func=lambda x: x.upper(), 
            on_change=_mark_track_dirty, 
            args=(track_id,)
        )
        
        if current_mode == SourceType.DESIGN.value:
            col_text, col_btn = st.columns([0.85, 0.15])
            with col_text:
                st.text_area("Prompt", key=f"p_{track_id}", height=100, on_change=_mark_track_dirty, args=(track_id,))
            with col_btn:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("🎲", key=f"reroll_{track_id}", help="New Variation (Change Seed)"):
                    st.session_state[f"seed_{track_id}"] = random.randint(0, 999999)
                    _mark_track_dirty(track_id)
                    st.toast(f"New seed set for Track {track_id}")

        elif current_mode == SourceType.CLONE.value:
            m = st.radio(f"Input ({track_id})", ["Upload", "Mic"], key=f"m_{track_id}", horizontal=True, on_change=_mark_track_dirty, args=(track_id,))
            if m == "Upload":
                st.file_uploader("Audio", type=["wav", "mp3"], key=f"up_{track_id}", on_change=_mark_track_dirty, args=(track_id,))
            else:
                st.audio_input("Record", key=f"mic_{track_id}", on_change=_mark_track_dirty, args=(track_id,))
            st.text_area("Transcript", key=f"ref_{track_id}", height=80, on_change=_mark_track_dirty, args=(track_id,))

        elif current_mode == "library":
            st.selectbox(
                "Select Voice", 
                list(voices.keys()) if voices else [], 
                key=f"lib_{track_id}", 
                on_change=_mark_track_dirty, 
                args=(track_id,)
            )
        
        st.caption(f"🔊 Preview Track {track_id}")
        phrase = st.text_input("Test Phrase", value="Testing voice.", key=f"preview_txt_{track_id}")
        
        if st.button(f"Play {track_id}", key=f"btn_{track_id}", use_container_width=True):
            if phrase:
                _dialog_preview_execution(session, track_id, phrase)
            else:
                st.warning("Please enter a test phrase.")

        last_audio_key = f"last_audio_{track_id}"
        if last_audio_key in st.session_state:
            st.audio(st.session_state[last_audio_key])



def render_studio(session: SessionManager):
    """
    Constructs the main application layout, aggregating all sub-components.

    It orchestrates the rendering of global settings, track configurations,
    blending controls, final generation output, and the voice saving interface.
    It checks the session state for persistent results (final audio) and renders
    the player if available.
    """
    lang_name = st.selectbox("Language", list(SUPPORTED_LANGUAGES.keys()), key="project_lang")
    session.set_language(SUPPORTED_LANGUAGES[lang_name])
    st.markdown("---")
    
    _render_track_config(session, "A")
    
    st.markdown("---")
    is_mix = st.checkbox("🔗 Mix with Track B", value=False, key="mix_checkbox")
    alpha = 0.5
    if is_mix:
        _render_track_config(session, "B")
        st.markdown("##### 🎛️ Blending")
        alpha = st.slider("Mix Balance", 0.0, 1.0, 0.5, 0.1, key="blend_slider")
    st.markdown("---")
    
    st.markdown("##### 🗣️ Final Audio Synthesis Test")
    final_text = st.text_area("Synthesis Text", height=150, key="tts_final")
    
    if st.button("🚀 Generate Audio", type="primary", use_container_width=True):
        if final_text:
            _dialog_final_execution(session, final_text, is_mix, alpha)
        else:
            st.warning("Please enter text to synthesize.")
    
    if "last_final_audio" in st.session_state:
        st.audio(st.session_state["last_final_audio"])

    with st.expander("💾 Save Voice to Library"):
        name = st.text_input("Voice Name", key="save_name")
        desc = st.text_area("Description", key="save_desc")
        tags_raw = st.text_input("Tags (comma separated)", key="save_tags")
        
        if st.button("Confirm Save", use_container_width=True):
            if name:
                meta = {
                    "blend": alpha if is_mix else None,
                    "source_type": SourceType.BLEND if is_mix else SourceType.DESIGN,
                }
                
                if st.session_state.get("mode_A") == SourceType.DESIGN.value:
                    meta["design_prompt_A"] = st.session_state.get("p_A", "")
                
                if is_mix and st.session_state.get("mode_B") == SourceType.DESIGN.value:
                    meta["design_prompt_B"] = st.session_state.get("p_B", "")
                
                _dialog_save_execution(
                    session, 
                    name, 
                    desc, 
                    [t.strip() for t in tags_raw.split(",")] if tags_raw else [], 
                    meta
                )
            else:
                st.warning("Please provide a name for the voice.")