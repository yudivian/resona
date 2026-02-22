import streamlit as st
import os
from src.ui.session import SessionManager
from src.models import VoiceProfile
from src.ui.explorer import _get_lang_flag

@st.dialog("🔬 Voice Synthesis Test Bench")
def _dialog_test_bench(session: SessionManager, voice: VoiceProfile, text: str):
    """
    Renders a dedicated synthesis test environment for a specific voice profile.
    
    This dialog replicates the core preview functionality from the explorer,
    ensuring that search results can be auditioned without leaving the search view.
    It loads the selected voice into Engine A for the duration of the test.

    Args:
        session (SessionManager): Global session manager instance.
        voice (VoiceProfile): The voice profile to be synthesized.
        text (str): The input text to render.
    """
    preview_key = f"search_preview_{voice.id}"
    
    with st.spinner(f"Synthesizing with {voice.name}..."):
        if preview_key not in st.session_state:
            try:
                session.load_voice_from_library("A", voice.id)
                output_path = session.engine_a.render(text)
                session._temp_files_registry.append(output_path)
                st.session_state[preview_key] = output_path
            except Exception as e:
                st.error(f"Synthesis failed: {e}")
                return

    if preview_key in st.session_state:
        st.audio(st.session_state[preview_key])
    
    if st.button("Close", use_container_width=True):
        if preview_key in st.session_state:
            del st.session_state[preview_key]
        st.rerun()

def _render_result_card(session: SessionManager, voice: VoiceProfile):
    """
    Renders a comprehensive, full-width voice profile card for search results.
    
    This component mirrors the visual hierarchy and feature set of the explorer 
    library but omits management actions such as Edit and Delete. It focuses 
    on auditioning and exporting discovered voices.

    Structure:
    - Header: Name, Language Flag, and Tags.
    - Info Section: Description and metadata.
    - Anchor Test: Reference audio playback and download.
    - Synthesis Test: Live text-to-speech preview.
    - Export: Portable .rnb bundle generation.

    Args:
        session (SessionManager): Active session controller.
        voice (VoiceProfile): Voice profile data object.
    """
    flag = _get_lang_flag(voice.language)
    
    tags_html = ""
    if voice.tags:
        tags_clean = ", ".join(voice.tags)
        tags_html = f"<span style='color: #888; font-size: 0.9em; margin-right: 10px;'>Tags: {tags_clean}</span>"

    with st.container(border=True):
        c_head_left, c_head_right = st.columns([0.5, 0.5])
        
        with c_head_left:
            st.markdown(f"### 🔊 {voice.name}")
            
        with c_head_right:
            st.markdown(
                f"<div style='text-align: right;'>{tags_html}<span style='font-size: 1.5em;'>{flag}</span></div>", 
                unsafe_allow_html=True
            )
        
        with st.expander("Details, Testing & Export", expanded=False):
            if voice.description:
                st.markdown(f"*{voice.description}*")
            
            st.caption(f"UUID: {voice.id} • Seed: {voice.seed} • Source: {voice.source_type}")

            st.markdown("---")

            st.markdown("**Anchor audio test:**")
            c_anc_play, c_anc_dl = st.columns([0.85, 0.15])
            
            with c_anc_play:
                if voice.anchor_audio_path:
                    path = os.path.join(session.config.paths.assets_dir, voice.anchor_audio_path)
                    if os.path.exists(path):
                        st.audio(path)
                    else:
                        st.warning("Anchor file missing.")
                else:
                    st.info("No anchor audio available.")
            
            with c_anc_dl:
                if voice.anchor_audio_path:
                    path = os.path.join(session.config.paths.assets_dir, voice.anchor_audio_path)
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            st.download_button(
                                label="📥",
                                data=f,
                                file_name=f"{voice.name}_anchor.wav",
                                mime="audio/wav",
                                key=f"s_anc_dl_{voice.id}",
                                use_container_width=True
                            )

            st.markdown("") 

            # Section: Audio Synthesis Test
            st.markdown("**Audio synthesis test:**")
            c_syn_in, c_syn_btn = st.columns([0.85, 0.15])
            
            with c_syn_in:
                prompt = st.text_input(
                    "Test Text", 
                    placeholder="Type to synthesize with this voice...", 
                    label_visibility="collapsed",
                    key=f"s_syn_in_{voice.id}"
                )
            
            with c_syn_btn:
                if st.button("▶️", key=f"s_syn_btn_{voice.id}", use_container_width=True):
                    if prompt:
                        _dialog_test_bench(session, voice, prompt)
                    else:
                        st.toast("Enter text to synthesize", icon="✍️")

            st.markdown("---")

            # Section: Export Action
            try:
                bundle = session.export_voice(voice.id)
                st.download_button(
                    label="📦 Export Voice Bundle (.rnb)",
                    data=bundle,
                    file_name=f"{voice.name.replace(' ', '_')}.rnb",
                    mime="application/octet-stream",
                    key=f"s_exp_btn_{voice.id}",
                    use_container_width=True
                )
            except Exception:
                st.error("Failed to generate export bundle.")

def render_search_view(session: SessionManager):
    """
    Renders the unified Semantic Search interface.
    
    Allows users to query the library using natural language and provides 
    immediate access to high-fidelity testing and export tools.
    """
    
    with st.expander("⚙️ Search Configuration", expanded=False):
        limit = st.slider(
            "Maximum results", 
            min_value=1, 
            max_value=50, 
            value=5,
            key="search_limit_slider"
        )

    st.markdown("Describe the voice personality or usage (e.g. 'deep gravelly voice for narrating').")
    
    query = st.text_input(
        "Search Prompt", 
        placeholder="Enter search description...", 
        key="search_query_input",
        label_visibility="collapsed"
    )



    if query:
        with st.spinner("Querying vector database..."):
            results = session.search_by_text(query, limit=limit)

        if results:
            st.divider()
            st.caption(f"Found {len(results)} matching profiles")
            for profile in results:
                _render_result_card(session, profile)
        else:
            st.divider()
            st.info("No matching voices found in the library.")