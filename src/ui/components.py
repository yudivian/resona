import streamlit as st
import os
import logging
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
    Configures the global Streamlit page settings and handles deferred resets.
    """
    st.set_page_config(
        page_title="Resona CAD",
        page_icon="🎛️",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Handle deferred UI reset to avoid State modification errors
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
    Internal helper to clear session state keys before widgets are rendered.
    """
    keys_to_reset = [
        "track_A_ready", "track_B_ready", "p_A", "p_B", "ref_A", "ref_B", 
        "tts_final", "preview_txt_A", "preview_txt_B", "save_name", "save_desc", "save_tags"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            st.session_state[key] = ""
            
    # Reset specific non-string defaults
    st.session_state["track_A_ready"] = False
    st.session_state["track_B_ready"] = False
    st.session_state["project_lang"] = "English"
    st.session_state["mix_checkbox"] = False
    st.session_state["blend_slider"] = 0.5
    
    # Clear file uploaders
    for key in ["up_A", "up_B", "mic_A", "mic_B"]:
        if key in st.session_state:
            del st.session_state[key]

def _trigger_ui_reset():
    """
    Sets a flag to reset the UI on the next script rerun.
    """
    st.session_state["reset_pending"] = True

@st.dialog("🔄 Reset Workflow")
def _dialog_reset_workflow(session: SessionManager):
    st.warning("You are about to clear the current session inputs and history.")
    st.markdown("Are you sure you want to start over?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Reset Session", type="primary", use_container_width=True):
            session.reset_workflow()
            _trigger_ui_reset()
            st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

# ... (Previous helper functions _save_temp_file, _mark_track_dirty, _ensure_track_identity remain unchanged)

def render_studio(session: SessionManager):
    """
    Main Studio Interface with reactive saving logic.
    """
    # (Previous rendering logic for Tracks A and B remains unchanged)
    
    # --- SAVE SECTION ---
    with st.expander("💾 Save Voice to Library"):
        name = st.text_input("Voice Name", key="save_name")
        description = st.text_area("Description (Human Readable)", key="save_desc")
        tags_raw = st.text_input("Tags", key="save_tags")
        
        if st.button("Confirm Save", use_container_width=True):
            if not name:
                st.warning("Voice Name is required.")
            else:
                ready_a = _ensure_track_identity(session, "A")
                if ready_a:
                    try:
                        # Construct indexing context
                        context_parts = [f"Name: {name}", f"Note: {description}"]
                        if st.session_state.get("mode_A") == SourceType.DESIGN.value:
                            context_parts.append(f"Prompt: {st.session_state.get('p_A', '')}")
                        
                        semantic_text = ". ".join(context_parts)

                        meta = {
                            "blend": st.session_state.get("blend_slider") if st.session_state.get("mix_checkbox") else None,
                            "source_type": SourceType.DESIGN, 
                            "language": SUPPORTED_LANGUAGES[st.session_state.get("project_lang", "English")],
                            "semantic_index": semantic_text 
                        }
                        
                        session.save_session_voice(
                            name=name,
                            description=description,
                            tags=[t.strip() for t in tags_raw.split(",")] if tags_raw else [],
                            metadata=meta
                        )
                        st.success(f"Saved '{name}' successfully.")
                        _trigger_ui_reset()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Save Error: {e}")