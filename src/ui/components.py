import streamlit as st
from src.ui.session import SessionManager

def init_page():
    """
    Configures the global Streamlit page settings and injects custom CSS.
    """
    st.set_page_config(
        page_title="Resona CAD",
        page_icon="🎛️",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Standard styles for the project
    st.markdown("""
        <style>
        .main { background-color: #0e1117; }
        .stMetric { background-color: #1e2130; padding: 10px; border-radius: 5px; }
        </style>
    """, unsafe_allow_html=True)

def render_debug_panel(session: SessionManager):
    """
    Displays system status and configuration for debugging purposes.
    """
    with st.sidebar:
        st.divider()
        st.subheader("🛠️ System Status")
        st.write(f"**Database:** `{session.config.paths.db_file}`")
        st.write(f"**VRAM Device:** `{session.config.system.compute.tts_device}`")
        
        if st.button("Clear Session Cache"):
            st.cache_resource.clear()
            st.rerun()
import streamlit as st
import torch
from src.ui.session import SessionManager

def _render_track_config(session: SessionManager, track_id: str):
    st.subheader(f"Track {track_id}")
    engine = session.engine_a if track_id == "A" else session.engine_b
    
    with st.container(border=True):
        source_type = st.selectbox(
            "Source Type", 
            ["Text Design", "Library", "Audio File", "Microphone"], 
            key=f"cad_src_type_{track_id}"
        )
        
        if source_type == "Text Design":
            st.text_area("Design Prompt", placeholder="Describe la voz...", key=f"cad_p_{track_id}")
        elif source_type == "Library":
            voices = session.store.get_all()
            st.selectbox("Select Voice", [v.name for v in voices], key=f"cad_lib_{track_id}")
        elif source_type == "Audio File":
            st.file_uploader("Upload Audio", type=["wav", "mp3"], key=f"cad_file_{track_id}")
        elif source_type == "Microphone":
            st.audio_input("Record Voice", key=f"cad_mic_{track_id}")

        st.divider()
        # TEXTO PARA PREVIEW (INDISPENSABLE)
        preview_text = st.text_input("Preview Text", value=f"Testing track {track_id}", key=f"cad_txt_{track_id}")

        if st.button(f"Preview Track {track_id}", key=f"cad_btn_{track_id}", use_container_width=True):
            with st.spinner(f"Configurando Track {track_id}..."):
                try:
                    # Sincronizamos idioma antes de procesar
                    engine.lang = session.engine_a.lang

                    if source_type == "Text Design":
                        prompt = st.session_state[f"cad_p_{track_id}"]
                        # USAR EL MÉTODO DE SESSION QUE LLAMA A design_identity
                        session.create_design_on_track(track_id, prompt, engine.lang)
                    
                    elif source_type == "Library":
                        v_name = st.session_state[f"cad_lib_{track_id}"]
                        voice = next((v for v in session.store.get_all() if v.name == v_name), None)
                        if voice:
                            # Carga directa del tensor
                            engine.set_identity(voice.identity_embedding, voice.seed)

                    # Una vez cargado el active_identity, generamos audio
                    path = engine.generate_preview(preview_text)
                    st.audio(path)
                except Exception as e:
                    st.error(f"Track Preview Error: {e}")

def render_studio(session: SessionManager):
    st.title("🎛️ Design Studio")

    # 1. Idioma Global (Requerido por InferenceEngine)
    langs = {
        "English": "en", "Spanish": "es", "French": "fr", "German": "de", 
        "Italian": "it", "Portuguese": "pt", "Chinese": "zh", 
        "Japanese": "ja", "Korean": "ko", "Russian": "ru"
    }
    sel_lang = st.selectbox("Global Language", list(langs.keys()), key="cad_global_lang")
    session.engine_a.lang = langs[sel_lang]
    session.engine_b.lang = langs[sel_lang]

    # 2. Track A
    _render_track_config(session, "A")

    # 3. Track B Opcional
    enable_b = st.checkbox("Enable Track B (Blending)", key="cad_show_b")
    if enable_b:
        _render_track_config(session, "B")
        st.subheader("🧬 Blending Parameters")
        blend_rate = st.slider("Blend Rate (A vs B)", 0.0, 1.0, 0.5, key="cad_blend_rate")

    st.divider()

    # 4. Refinamiento y Preview Final
    st.subheader("✨ Final Polish")
    refinement = st.text_area("Refinement Prompt", key="cad_ref_prompt")
    final_txt = st.text_input("Final Synthesis Text", value="Resona CAD check.", key="cad_final_txt")

    if st.button("Preview Final Design", type="primary", use_container_width=True):
        try:
            if enable_b:
                # Mezclamos A y B usando tu utilidad VoiceBlender
                from src.backend.engine import VoiceBlender
                alpha = st.session_state.cad_blend_rate
                mixed_vec = VoiceBlender.blend(session.engine_a, session.engine_b, alpha)
                # Cargamos el mix en el motor A para el preview final
                session.engine_a.active_identity = torch.tensor(mixed_vec).to(session.engine_a.device)
            
            path = session.engine_a.generate_preview(final_txt, refinement=refinement)
            st.audio(path)
        except Exception as e:
            st.error(f"Final Preview Error: {e}")

    if st.button("Save Voice to Library", use_container_width=True):
        st.info("Identity ready for persistence.")