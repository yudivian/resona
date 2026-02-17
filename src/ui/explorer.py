import streamlit as st
import os
import math
from typing import List, Optional
from src.ui.session import SessionManager
from src.models import VoiceProfile

ITEMS_PER_PAGE = 10

def _get_lang_flag(lang_code: str) -> str:
    """
    Retrieves the emoji flag associated with a given ISO language code.
    
    Args:
        lang_code (str): Two-letter ISO 639-1 language code.
        
    Returns:
        str: Emoji flag or globe icon if unknown.
    """
    if not lang_code:
        return "🌐"
    
    mapping = {
        "en": "🇬🇧", "es": "🇪🇸", "fr": "🇫🇷", "de": "🇩🇪",
        "it": "🇮🇹", "pt": "🇵🇹", "zh": "🇨🇳", "ja": "🇯🇵",
        "ko": "🇰🇷", "ru": "🇷🇺"
    }
    return mapping.get(lang_code.lower(), "🌐")

def render_explorer(session: SessionManager):
    """
    Renders the Voice Library Explorer with a structured, labeled card layout.

    This interface manages the display, filtering, and manipulation of the voice 
    database. It implements a hierarchical card design where functional areas 
    (Anchor Test, Synthesis Test, Management) are clearly labeled and segmented 
    for improved usability.

    Features:
    - Server-side pagination.
    - Advanced meta-data filtering.
    - Modal workflows for Editing, Importing, and Deleting.
    - Explicitly labeled test sections within each voice card.

    Args:
        session (SessionManager): The active session orchestrator.
    """
    st.header("📚 Voice Library")
    
    all_voices = session.store.get_all()
    available_langs = sorted(list({v.language for v in all_voices if v.language}))

    col_search, col_lang, col_import = st.columns([0.5, 0.3, 0.2])
    
    with col_search:
        search_query = st.text_input(
            "Search", 
            placeholder="Search voice...", 
            label_visibility="collapsed"
        ).lower()
    
    with col_lang:
        selected_langs = st.multiselect(
            "Language",
            options=available_langs,
            placeholder="Language",
            label_visibility="collapsed"
        )

    with col_import:
        if st.button("📥 Import", use_container_width=True):
            _dialog_import_bundle(session)

    filtered_voices = [
        v for v in all_voices 
        if (not search_query or search_query in v.name.lower() or search_query in (v.description or "").lower())
        and (not selected_langs or v.language in selected_langs)
    ]

    if not filtered_voices:
        st.info("No voices found.")
        return

    total_items = len(filtered_voices)
    total_pages = math.ceil(total_items / ITEMS_PER_PAGE)
    
    if "exp_page" not in st.session_state:
        st.session_state.exp_page = 1
    if st.session_state.exp_page > total_pages:
        st.session_state.exp_page = max(1, total_pages)
        
    current_page = st.session_state.exp_page
    start_idx = (current_page - 1) * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    current_batch = filtered_voices[start_idx:end_idx]

    st.caption(f"Showing {start_idx + 1}-{min(end_idx, total_items)} of {total_items} voices")

    for voice in current_batch:
        _render_card_layout(session, voice)

    if total_pages > 1:
        c1, c2, c3, c4, c5 = st.columns([0.1, 0.2, 0.4, 0.2, 0.1])
        with c2:
            if st.button("< Previous", disabled=(current_page == 1), use_container_width=True):
                st.session_state.exp_page -= 1
                st.rerun()
        with c4:
            if st.button("Next >", disabled=(current_page == total_pages), use_container_width=True):
                st.session_state.exp_page += 1
                st.rerun()

def _render_card_layout(session: SessionManager, voice: VoiceProfile):
    """
    Renders a voice profile card with explicit section headers.

    Structure:
    1. Header Bar: Name (Left) | Tags + Flag (Right).
    2. Expander Body:
       - Metadata Block.
       - "Anchor audio test" Section: Player + Download.
       - "Audio synthesis test" Section: Input + Play.
       - Management Section: Edit, Bundle, Delete.

    Args:
        session (SessionManager): Active session.
        voice (VoiceProfile): Voice data object.
    """
    name_raw = voice.name.strip()
    flag = _get_lang_flag(voice.language)
    
    tags_html = ""
    if voice.tags:
        tags_clean = ", ".join(voice.tags[:3])
        tags_html = f"<span style='color: #888; font-size: 0.9em; margin-right: 10px;'>Tags: {tags_clean}</span>"

    with st.container(border=True):
        c_left, c_right = st.columns([0.4, 0.6])
        
        with c_left:
            st.markdown(f"**🔊 {name_raw}**")
            
        with c_right:
            st.markdown(
                f"<div style='text-align: right;'>{tags_html}<span style='font-size: 1.2em;'>{flag}</span></div>", 
                unsafe_allow_html=True
            )
        
        with st.expander("Details & Actions", expanded=False):
            
            if voice.description:
                st.markdown(f"_{voice.description}_")
            
            source_str = str(voice.source_type.value if hasattr(voice.source_type, "value") else voice.source_type).upper()
            st.caption(f"UUID: {voice.id} • Seed: {voice.seed} • Source: {source_str} ")

            st.markdown("---")

            # --- Section 1: Anchor Audio ---
            st.markdown("**Anchor audio test:**")
            c_audio, c_dl = st.columns([0.85, 0.15])
            
            with c_audio:
                if voice.anchor_audio_path:
                    file_path = os.path.join(session.config.paths.assets_dir, voice.anchor_audio_path)
                    if os.path.exists(file_path):
                        st.audio(file_path)
                    else:
                        st.warning("File missing")
                else:
                    st.info("No audio reference")
            
            with c_dl:
                if voice.anchor_audio_path and os.path.exists(os.path.join(session.config.paths.assets_dir, voice.anchor_audio_path)):
                    with open(os.path.join(session.config.paths.assets_dir, voice.anchor_audio_path), "rb") as f:
                        st.download_button(
                            label="📥",
                            data=f,
                            file_name=f"{voice.name}_anchor.wav",
                            mime="audio/wav",
                            key=f"dl_wav_{voice.id}",
                            use_container_width=True,
                            help="Download Anchor WAV"
                        )
                else:
                    st.button("🚫", disabled=True, key=f"no_wav_{voice.id}")

            st.markdown("") 

            # --- Section 2: Synthesis Test ---
            st.markdown("**Audio synthesis test:**")
            c_input, c_play = st.columns([0.85, 0.15])
            
            with c_input:
                test_text = st.text_input(
                    "Test Prompt", 
                    placeholder="Type to synthesize...", 
                    label_visibility="collapsed",
                    key=f"txt_{voice.id}"
                )
            with c_play:
                if st.button("▶️", key=f"btn_{voice.id}", use_container_width=True, help="Run Synthesis"):
                    if test_text:
                        _dialog_test_bench(session, voice, test_text)
                    else:
                        st.toast("Enter text first", icon="✍️")

            st.markdown("---")

            # --- Section 3: Management ---
            b_edit, b_export, b_del = st.columns(3)
            
            with b_edit:
                if st.button("✏️ Edit", key=f"edt_{voice.id}", use_container_width=True):
                    _dialog_edit_voice(session, voice)
            
            with b_export:
                try:
                    bundle_data = session.export_voice(voice.id)
                    st.download_button(
                        label="📦 Bundle",
                        data=bundle_data,
                        file_name=f"{voice.name.replace(' ', '_')}.rnb",
                        mime="application/octet-stream",
                        key=f"dl_rnb_{voice.id}",
                        use_container_width=True
                    )
                except:
                    st.button("🚫 Error", disabled=True, key=f"err_b_{voice.id}")

            with b_del:
                if st.button("🗑️ Delete", key=f"del_{voice.id}", type="secondary", use_container_width=True):
                    _dialog_delete_confirm(session, voice.id, voice.name)

@st.dialog("✏️ Edit Voice Metadata")
def _dialog_edit_voice(session: SessionManager, voice: VoiceProfile):
    """
    Modal interface for updating voice metadata.

    Allows user modification of Name, Tags, and Description.
    Triggers backend semantic re-indexing upon save.
    """
    st.caption(f"Editing ID: {voice.id}")
    
    new_name = st.text_input("Name", value=voice.name)
    
    current_tags_str = ", ".join(voice.tags) if voice.tags else ""
    new_tags_str = st.text_input("Tags (comma separated)", value=current_tags_str)
    
    new_desc = st.text_area("Description", value=voice.description or "")

    if st.button("Save Changes", type="primary", use_container_width=True):
        try:
            clean_tags = [t.strip() for t in new_tags_str.split(",") if t.strip()]
            
            # Requires session.update_voice_metadata to be implemented in SessionManager
            session.update_voice_metadata(
                voice_id=voice.id,
                new_name=new_name,
                new_desc=new_desc,
                new_tags=clean_tags
            )
            st.toast("Voice updated successfully", icon="✅")
            st.rerun()
        except Exception as e:
            st.error(f"Update failed: {e}")

@st.dialog("🧪 Test Bench")
def _dialog_test_bench(session: SessionManager, voice: VoiceProfile, text: str):
    """
    Isolated synthesis modal. Bypasses global studio state.
    """
    st.markdown(f"**Testing:** {voice.name}")
    preview_key = f"exp_prev_{voice.id}"
    
    if preview_key not in st.session_state:
        with st.spinner("Synthesizing..."):
            try:
                session.load_voice_from_library("A", voice.id)
                output_path = session.engine_a.render(text)
                session._temp_files_registry.append(output_path)
                st.session_state[preview_key] = output_path
            except Exception as e:
                st.error(f"Error: {e}")
                return

    if preview_key in st.session_state:
        st.audio(st.session_state[preview_key])
    
    if st.button("Close", use_container_width=True):
        if preview_key in st.session_state:
            del st.session_state[preview_key]
        st.rerun()

@st.dialog("📥 Import Voice")
def _dialog_import_bundle(session: SessionManager):
    """Import Modal."""
    st.markdown("Upload `.rnb` file.")
    f = st.file_uploader("File", type=["rnb"])
    if f and st.button("Import", type="primary", use_container_width=True):
        try:
            session.import_voice(f.getvalue())
            st.toast("Imported!", icon="✅")
            st.rerun()
        except Exception as e:
            st.error(str(e))

@st.dialog("⚠️ Delete Confirmation")
def _dialog_delete_confirm(session: SessionManager, voice_id: str, name: str):
    """Deletion Modal."""
    st.write(f"Permanently delete **{name}**?")
    if st.button("Confirm Delete", type="primary", use_container_width=True):
        try:
            session.delete_voice(voice_id)
            st.toast("Deleted")
            st.rerun()
        except Exception as e:
            st.error(str(e))