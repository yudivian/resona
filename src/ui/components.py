import streamlit as st
import os
import logging
import random
from src.ui.session import SessionManager
from src.models import SourceType

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Chinese": "zh", "Japanese": "ja",
    "Korean": "ko", "Russian": "ru"
}

def init_page():
    """
    Configures page settings and handles the deferred UI reset mechanism.
    """
    st.set_page_config(
        page_title="Resona CAD",
        page_icon="🎛️",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
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
    Clears session state keys including seeds and inputs.
    """
    keys_to_reset = [
        "track_A_ready", "track_B_ready", "p_A", "p_B", "ref_A", "ref_B",
        "up_A", "mic_A", "up_B", "mic_B", "m_A", "m_B", "lib_A", "lib_B",
        "save_name", "save_desc", "save_tags", "mix_checkbox", "blend_slider",
        "tts_final", "preview_txt_A", "preview_txt_B", 
        "seed_A", "seed_B" # Reset seeds as well
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state["track_A_ready"] = False
    st.session_state["track_B_ready"] = False

def render_debug_panel(session: SessionManager):
    """
    Renders sidebar maintenance tools.
    """
    with st.sidebar:
        st.header("🛠️ System Status")
        if hasattr(session, 'config'):
            st.caption(f"Device: {session.config.system.compute.tts_device}")
            st.caption(f"DB: {os.path.basename(session.config.paths.db_file)}")
        st.markdown("---")
        if st.button("🔄 Reset Workflow", use_container_width=True):
            session.reset_workflow()
            st.session_state["reset_pending"] = True
            st.rerun()
        if st.button("🗑️ Purge Temp Files", use_container_width=True):
            session.purge_temp_storage()
            st.session_state["reset_pending"] = True
            st.rerun()

def _save_temp_file(session: SessionManager, uploaded_file, prefix: str) -> str:
    """
    Saves uploaded file to temp directory.
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
    """Invalidates track readiness."""
    st.session_state[f"track_{track_id}_ready"] = False

def _ensure_track_identity(session: SessionManager, track_id: str) -> bool:
    """
    Triggers identity generation. Handles Seed logic for Design mode.
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
            
            # --- Seed Management ---
            seed_key = f"seed_{track_id}"
            if seed_key not in st.session_state:
                st.session_state[seed_key] = random.randint(0, 999999)
            current_seed = st.session_state[seed_key]

            with st.spinner(f"Designing Track {track_id} (Seed: {current_seed})..."):
                # Pass the seed explicitly to the session
                session.design_voice(track_id, prompt, seed=current_seed)

        elif mode == SourceType.CLONE.value:
            method = st.session_state.get(f"m_{track_id}")
            audio = st.session_state.get(f"up_{track_id}") if method == "Upload" else st.session_state.get(f"mic_{track_id}")
            ref = st.session_state.get(f"ref_{track_id}")
            if not audio or not ref:
                st.warning(f"Track {track_id}: Missing audio or transcript.")
                return False
            path = _save_temp_file(session, audio, f"clone_{track_id}")
            with st.spinner(f"Cloning Track {track_id}..."):
                session.clone_voice(track_id, path, ref)
        
        st.session_state[f"track_{track_id}_ready"] = True
        return True
    except Exception as e:
        st.error(f"Error on Track {track_id}: {e}")
        return False

def _render_track_config(session: SessionManager, track_id: str):
    """
    Renders track config. Includes Reroll Button (Dice) for Design mode.
    """
    st.markdown(f"### 🎚️ Track {track_id}")
    mode_key = f"mode_{track_id}"
    st.selectbox(
        f"Source ({track_id})", 
        [SourceType.DESIGN.value, SourceType.CLONE.value], 
        key=mode_key, 
        format_func=lambda x: x.upper(), 
        on_change=_mark_track_dirty, 
        args=(track_id,)
    )
    
    current_mode = st.session_state[mode_key]
    
    if current_mode == SourceType.DESIGN.value:
        # Layout: Prompt area + Reroll button side-by-side
        col_text, col_btn = st.columns([0.85, 0.15])
        with col_text:
            st.text_area("Prompt", key=f"p_{track_id}", height=100, on_change=_mark_track_dirty, args=(track_id,))
        with col_btn:
            st.markdown("<br><br>", unsafe_allow_html=True) # Vertical alignment hack
            # Reroll Button
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
    
    st.caption(f"🔊 Preview Track {track_id}")
    phrase = st.text_input("Test Phrase", value="Testing voice.", key=f"preview_txt_{track_id}")
    if st.button(f"Play {track_id}", key=f"btn_{track_id}", use_container_width=True):
        if phrase and _ensure_track_identity(session, track_id):
            alpha = 0.0 if track_id == "A" else 1.0
            with st.spinner("Synthesizing..."):
                try:
                    path = session.preview_voice(phrase, blend_alpha=alpha)
                    st.audio(path)
                except Exception as e:
                    st.error(f"Preview Error: {e}")
    st.markdown("---")

def render_studio(session: SessionManager):
    """
    Main Studio Interface.
    """
    st.subheader("🌍 Project Settings")
    lang_name = st.selectbox("Language", list(SUPPORTED_LANGUAGES.keys()), key="project_lang")
    session.set_language(SUPPORTED_LANGUAGES[lang_name])
    st.markdown("---")
    
    _render_track_config(session, "A")
    
    is_mix = st.checkbox("🔗 Mix with Track B", value=False, key="mix_checkbox")
    alpha = 0.5
    if is_mix:
        _render_track_config(session, "B")
        st.markdown("### 🎛️ Blending")
        alpha = st.slider("Mix Balance", 0.0, 1.0, 0.5, 0.1, key="blend_slider")

    st.markdown("### 🗣️ Final Generation")
    final_text = st.text_area("Final Text", height=150, key="tts_final")
    if st.button("🚀 Generate Final Audio", type="primary", use_container_width=True):
        if final_text:
            ready_a = _ensure_track_identity(session, "A")
            ready_b = _ensure_track_identity(session, "B") if is_mix else True
            if ready_a and ready_b:
                with st.spinner("Synthesizing..."):
                    try:
                        path = session.preview_voice(final_text, blend_alpha=alpha if is_mix else None)
                        st.audio(path)
                    except Exception as e:
                        st.error(f"Synthesis Error: {e}")

    with st.expander("💾 Save Voice to Library"):
        name = st.text_input("Voice Name", key="save_name")
        desc = st.text_area("Description", key="save_desc")
        tags_raw = st.text_input("Tags (comma separated)", key="save_tags")
        
        if st.button("Confirm Save", use_container_width=True):
            if name:
                ready_a = _ensure_track_identity(session, "A")
                ready_b = _ensure_track_identity(session, "B") if is_mix else True
                
                if ready_a and ready_b:
                    try:
                        context = [f"Name: {name}", f"Note: {desc}", f"Tags: {tags_raw}"]
                        if st.session_state.get("mode_A") == SourceType.DESIGN.value:
                            context.append(f"Prompt A: {st.session_state.get('p_A')}")
                        if is_mix and st.session_state.get("mode_B") == SourceType.DESIGN.value:
                            context.append(f"Prompt B: {st.session_state.get('p_B')}")
                        
                        meta = {
                            "blend": alpha if is_mix else None,
                            "source_type": SourceType.BLEND if is_mix else SourceType.DESIGN,
                            "semantic_index": ". ".join(context)
                        }
                        
                        session.save_session_voice(
                            name=name,
                            description=desc,
                            tags=[t.strip() for t in tags_raw.split(",")] if tags_raw else [],
                            metadata=meta
                        )
                        st.success(f"Voice '{name}' saved.")
                        st.session_state["reset_pending"] = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Persistence Error: {e}")