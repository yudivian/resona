import streamlit as st
import os
import uuid
import math
import difflib
import unicodedata
from typing import Dict, Any, Optional, List, Tuple

from src.models import AppConfig
from src.config import ResonaConfig
from src.emotions.manager import EmotionManager
from src.backend.engine import TTSModelProvider, InferenceEngine
from src.backend.store import VoiceStore

SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ru"]
ITEMS_PER_PAGE = 10


@st.cache_resource
def load_system_dependencies() -> Tuple[AppConfig, EmotionManager, VoiceStore, TTSModelProvider]:
    """
    Initializes and caches the core backend components of the application.
    Caching TTSModelProvider prevents memory leaks during repetitive testing.

    Returns:
        Tuple[AppConfig, EmotionManager, VoiceStore, TTSModelProvider]: The cached instances.
    """
    config = ResonaConfig.load()
    manager = EmotionManager(config)
    store = VoiceStore(config)
    provider = TTSModelProvider(config)
    return config, manager, store, provider


def initialize_session_state() -> None:
    """
    Initializes required variables within the Streamlit session state.
    """
    if "emotion_page" not in st.session_state:
        st.session_state.emotion_page = 1
    if "intensifier_page" not in st.session_state:
        st.session_state.intensifier_page = 1
    if "active_test_audio" not in st.session_state:
        st.session_state.active_test_audio = None


def is_fuzzy_match(query: str, target: str, threshold: float = 0.65) -> bool:
    """
    Evaluates if a search query fuzzily matches a target string using sequence similarity.
    Ignores accents, casing, and tolerates minor typographical errors.

    Args:
        query (str): The search input string.
        target (str): The string to evaluate against.
        threshold (float): Minimum similarity ratio required for a match.

    Returns:
        bool: True if the strings are deemed a match, False otherwise.
    """
    if not query:
        return True
        
    def normalize_string(s: str) -> str:
        return "".join(c for c in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(c)).lower()
        
    normalized_query = normalize_string(query)
    normalized_target = normalize_string(target)
    
    if normalized_query in normalized_target:
        return True
        
    if difflib.SequenceMatcher(None, normalized_query, normalized_target).ratio() >= threshold:
        return True
        
    for word in normalized_target.split():
        if difflib.SequenceMatcher(None, normalized_query, word).ratio() >= threshold:
            return True
            
    return False


@st.dialog("Confirm Deletion")
def delete_item_modal(manager: EmotionManager, item_type: str, item_id: str) -> None:
    """
    Renders a modal dialogue to confirm the permanent removal of a catalog item.

    Args:
        manager (EmotionManager): The instance controlling the emotion catalog.
        item_type (str): The category of the item ('emotion' or 'intensifier').
        item_id (str): The unique canonical identifier of the item to delete.
    """
    st.warning(f"Are you sure you want to permanently delete the {item_type} '{item_id}'?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("Delete", type="primary", use_container_width=True):
            if item_type == "emotion":
                manager.delete_emotion(item_id)
            else:
                manager.delete_intensifier(item_id)
            st.rerun()


@st.dialog("Confirm Session Cleanup")
def confirm_clear_session_modal(manager: EmotionManager) -> None:
    """
    Renders a confirmation modal for clearing session-specific audio files.

    Args:
        manager (EmotionManager): The controller handling the workspace directory.
    """
    st.warning("Are you sure you want to clear all audio files generated during this session?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("Clear Session", type="primary", use_container_width=True):
            manager.clear_session_audios()
            st.session_state.active_test_audio = None
            st.rerun()


@st.dialog("Confirm Workspace Wipe")
def confirm_wipe_all_modal(manager: EmotionManager) -> None:
    """
    Renders a confirmation modal for wiping all temporary audio files in the workspace.

    Args:
        manager (EmotionManager): The controller handling the workspace directory.
    """
    st.warning("Are you sure you want to permanently delete ALL temporary audio files in the emotions workspace?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("Wipe Workspace", type="primary", use_container_width=True):
            manager.clear_all_temp_audios()
            st.session_state.active_test_audio = None
            st.rerun()


@st.dialog("Synthesis Result")
def synthesis_result_modal(audio_path: str, emotion_name: str, intensifier_name: str, params: Dict[str, float]) -> None:
    """
    Renders a modal dialogue displaying the results of a standalone synthesis test.

    Args:
        audio_path (str): The file system path to the generated audio.
        emotion_name (str): The display name of the applied emotion.
        intensifier_name (str): The display name of the applied intensifier.
        params (Dict[str, float]): The exact hyperparameters passed to the inference engine.
    """
    st.success("Synthesis completed successfully.")
    st.audio(audio_path)
    st.divider()
    st.markdown(f"**Emotion:** {emotion_name} | **Intensity:** {intensifier_name}")
    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.metric("Temperature", f"{params['temp']:.2f}")
    p_col2.metric("Penalty", f"{params['penalty']:.2f}")
    p_col3.metric("Top P", f"{params['top_p']:.2f}")


def execute_synthesis(config: AppConfig, manager: EmotionManager, store: VoiceStore, 
                      provider: TTSModelProvider, voice_id: str, lang: str, text: str, 
                      params: Dict[str, float]) -> Optional[str]:
    """
    Centralized execution pipeline for audio synthesis.

    Args:
        config (AppConfig): Application configuration.
        manager (EmotionManager): Emotion catalog controller.
        store (VoiceStore): Voice identity controller.
        provider (TTSModelProvider): Cached neural network provider.
        voice_id (str): Selected voice profile ID.
        lang (str): Selected generation language.
        text (str): Dialogue script to synthesize.
        params (Dict[str, float]): Clamped prosody parameters.

    Returns:
        Optional[str]: The path to the generated file, or None if failed.
    """
    with st.spinner("Synthesizing..."):
        try:
            engine = InferenceEngine(config, provider, lang=lang)
            profile = store.get_profile(voice_id)
            anchor = os.path.join(config.paths.assets_dir, profile.anchor_audio_path) if profile.anchor_audio_path else None
            engine.load_identity_from_state(profile.identity_embedding, profile.seed, anchor)
            
            output_name = f"preview_{uuid.uuid4().hex[:8]}.wav"
            output_path = os.path.join(manager.temp_emotions_dir, output_name)
            final_path = engine.render_with_emotion(text, params, output_path)
            manager.register_session_audio(final_path)
            return final_path
        except Exception as e:
            st.error(f"Synthesis failed: {e}")
            return None


@st.dialog("Quick Synthesis Test")
def quick_test_modal(manager: EmotionManager, store: VoiceStore, config: AppConfig, provider: TTSModelProvider,
                     ui_language: str = "en", fixed_emotion: Optional[str] = None, fixed_intensifier: Optional[str] = None) -> None:
    """
    Renders a modal dialogue for rapid audio testing directly from the explorer lists.

    Args:
        manager (EmotionManager): The instance controlling the emotion catalog.
        store (VoiceStore): The instance providing access to voice profiles.
        config (AppConfig): The global application configuration.
        provider (TTSModelProvider): The cached model provider.
        ui_language (str): The inherited UI language for rendering localized vocabulary.
        fixed_emotion (Optional[str]): A canonical ID to lock the emotion selection.
        fixed_intensifier (Optional[str]): A canonical ID to lock the intensifier selection.
    """
    profiles = store.get_all()
    if not profiles:
        st.warning("No voice profiles available in the database.")
        return

    vocab = manager.get_localized_vocab(lang=ui_language)
    voice_options = {p.id: p.name for p in profiles}
    voice_ids = sorted(list(voice_options.keys()), key=lambda x: voice_options[x].lower())
    
    emotion_ids = sorted(list(manager.catalog.get("emotions", {}).keys()), key=lambda x: vocab["emotions"].get(x, x).lower())
    int_ids = [None] + sorted(list(manager.catalog.get("intensifiers", {}).keys()), key=lambda x: vocab["intensifiers"].get(x, x).lower())

    default_e_idx = emotion_ids.index(fixed_emotion) if fixed_emotion in emotion_ids else 0
    default_i_idx = int_ids.index(fixed_intensifier) if fixed_intensifier in int_ids else 0
    default_lang_idx = SUPPORTED_LANGUAGES.index(ui_language) if ui_language in SUPPORTED_LANGUAGES else 1

    col1, col2 = st.columns(2)
    with col1:
        selected_voice = st.selectbox("Voice", options=voice_ids, format_func=lambda x: voice_options[x])
        selected_emotion = st.selectbox("Emotion", options=emotion_ids, index=default_e_idx, 
                                        format_func=lambda x: vocab["emotions"].get(x, x), disabled=(fixed_emotion is not None))
    with col2:
        selected_lang = st.selectbox("Generation Lang", options=SUPPORTED_LANGUAGES, index=default_lang_idx)
        selected_intensifier = st.selectbox("Intensity", options=int_ids, index=default_i_idx, 
                                            format_func=lambda x: vocab["intensifiers"].get(x, "None") if x else "None", disabled=(fixed_intensifier is not None))

    test_text = st.text_area("Test Script", placeholder="Type the dialogue to render...")

    if st.button("Generate Audio", type="primary", use_container_width=True):
        if not test_text:
            st.error("Please provide a test script.")
            return
            
        calculated = manager.calculate_parameters(emotion_id=selected_emotion, intensifier_id=selected_intensifier)
        final_path = execute_synthesis(config, manager, store, provider, selected_voice, selected_lang, test_text, calculated)
        
        if final_path:
            st.success("Synthesis completed successfully.")
            st.audio(final_path)
            st.caption(f"Applied Math -> Temp: {calculated['temp']:.2f} | Penalty: {calculated['penalty']:.2f} | Top P: {calculated['top_p']:.2f}")


@st.dialog("Emotion Editor")
def emotion_form_modal(manager: EmotionManager, store: VoiceStore, config: AppConfig, provider: TTSModelProvider,
                       ui_language: str = "en", emotion_id: Optional[str] = None) -> None:
    """
    Renders a modal dialogue for creating or updating an emotion entry, including an embedded tester.

    Args:
        manager (EmotionManager): The instance controlling the emotion catalog.
        store (VoiceStore): The instance providing access to voice profiles.
        config (AppConfig): The global application configuration.
        provider (TTSModelProvider): The cached model provider.
        ui_language (str): The inherited UI language for rendering localized vocabulary.
        emotion_id (Optional[str]): The ID of the emotion to edit, or None for a new one.
    """
    is_new = emotion_id is None
    current_data = manager.catalog.get("emotions", {}).get(emotion_id, {}) if not is_new else {}
    new_id = st.text_input("Canonical ID", value=emotion_id if emotion_id else "", disabled=not is_new)
    
    m_col1, m_col2, m_col3 = st.columns(3)
    temp = m_col1.slider("Temp", -1.0, 1.0, float(current_data.get("modifiers", {}).get("temp", 0.0)))
    penalty = m_col2.slider("Penalty", -0.5, 0.5, float(current_data.get("modifiers", {}).get("penalty", 0.0)))
    top_p = m_col3.slider("Top P", -0.5, 0.5, float(current_data.get("modifiers", {}).get("top_p", 0.0)))
    
    with st.expander("🌐 Localized Translations (Empty = ID Fallback)"):
        local_names = {}
        t_col1, t_col2 = st.columns(2)
        for i, lcode in enumerate(SUPPORTED_LANGUAGES):
            target_col = t_col1 if i % 2 == 0 else t_col2
            local_names[lcode] = target_col.text_input(f"{lcode.upper()}", value=current_data.get("localized_names", {}).get(lcode, ""))

    with st.expander("🧪 Mini-Tester"):
        vocab = manager.get_localized_vocab(lang=ui_language)
        profiles = store.get_all()
        voice_options = {p.id: p.name for p in profiles}
        voice_ids = sorted(list(voice_options.keys()), key=lambda x: voice_options[x].lower())
        default_lang_idx = SUPPORTED_LANGUAGES.index(ui_language) if ui_language in SUPPORTED_LANGUAGES else 1
        
        it_col1, it_col2, it_col3 = st.columns(3)
        test_voice = it_col1.selectbox("Voice", options=voice_ids, format_func=lambda x: voice_options[x], key="mt_e_voice")
        test_lang = it_col2.selectbox("Gen Lang", options=SUPPORTED_LANGUAGES, index=default_lang_idx, key="mt_e_lang")
        
        int_ids = [None] + sorted(list(vocab["intensifiers"].keys()), key=lambda x: vocab["intensifiers"][x].lower())
        test_int = it_col3.selectbox("Intensity", options=int_ids, format_func=lambda x: vocab["intensifiers"].get(x, "None") if x else "None/Normal", key="mt_e_int")
        
        test_text = st.text_input("Script", placeholder="Type a sentence...", key="mt_e_script")
            
        if st.button("Generate Preview", use_container_width=True, key="mt_e_btn"):
            if not test_text:
                st.error("Please provide a test script.")
            else:
                calc_params = manager.calculate_parameters(emotion_override={"temp": temp, "penalty": penalty, "top_p": top_p}, intensifier_id=test_int)
                final_path = execute_synthesis(config, manager, store, provider, test_voice, test_lang, test_text, calc_params)
                if final_path:
                    st.audio(final_path)
                    st.caption(f"Preview -> Temp: {calc_params['temp']:.2f} | Penalty: {calc_params['penalty']:.2f} | Top P: {calc_params['top_p']:.2f}")

    if st.button("Save Emotion", type="primary", use_container_width=True):
        is_valid, msg = manager.validate_entry(new_id, "emotions", local_names)
        if not is_valid:
            st.error(msg)
        else:
            clean_id = new_id.strip().lower().replace(" ", "_")
            manager.save_emotion(clean_id, {"temp": temp, "penalty": penalty, "top_p": top_p}, local_names)
            st.rerun()


@st.dialog("Intensifier Editor")
def intensifier_form_modal(manager: EmotionManager, store: VoiceStore, config: AppConfig, provider: TTSModelProvider,
                           ui_language: str = "en", intensifier_id: Optional[str] = None) -> None:
    """
    Renders a modal dialogue for creating or updating an intensifier entry, including an embedded tester.

    Args:
        manager (EmotionManager): The instance controlling the emotion catalog.
        store (VoiceStore): The instance providing access to voice profiles.
        config (AppConfig): The global application configuration.
        provider (TTSModelProvider): The cached model provider.
        ui_language (str): The inherited UI language for rendering localized vocabulary.
        intensifier_id (Optional[str]): The ID of the intensifier to edit, or None for a new one.
    """
    is_new = intensifier_id is None
    current_data = manager.catalog.get("intensifiers", {}).get(intensifier_id, {}) if not is_new else {}
    new_id = st.text_input("Canonical ID", value=intensifier_id if intensifier_id else "", disabled=not is_new)
    multiplier = st.number_input("Multiplier", value=float(current_data.get("multiplier", 1.0)), step=0.1)
    
    with st.expander("🌐 Localized Translations (Empty = ID Fallback)"):
        local_names = {}
        t_col1, t_col2 = st.columns(2)
        for i, lcode in enumerate(SUPPORTED_LANGUAGES):
            target_col = t_col1 if i % 2 == 0 else t_col2
            local_names[lcode] = target_col.text_input(f"{lcode.upper()}", value=current_data.get("localized_names", {}).get(lcode, ""))

    with st.expander("🧪 Mini-Tester"):
        vocab = manager.get_localized_vocab(lang=ui_language)
        profiles = store.get_all()
        voice_options = {p.id: p.name for p in profiles}
        voice_ids = sorted(list(voice_options.keys()), key=lambda x: voice_options[x].lower())
        default_lang_idx = SUPPORTED_LANGUAGES.index(ui_language) if ui_language in SUPPORTED_LANGUAGES else 1
        
        it_col1, it_col2, it_col3 = st.columns(3)
        test_voice = it_col1.selectbox("Voice", options=voice_ids, format_func=lambda x: voice_options[x], key="mt_i_voice")
        test_lang = it_col2.selectbox("Gen Lang", options=SUPPORTED_LANGUAGES, index=default_lang_idx, key="mt_i_lang")
        
        emotion_ids = sorted(list(vocab["emotions"].keys()), key=lambda x: vocab["emotions"][x].lower())
        test_emo = it_col3.selectbox("Base Emotion", options=emotion_ids, format_func=lambda x: vocab["emotions"].get(x, x), key="mt_i_emo")
        
        test_text = st.text_input("Script", placeholder="Type a sentence...", key="mt_i_script")
            
        if st.button("Generate Preview", use_container_width=True, key="mt_i_btn"):
            if not test_text:
                st.error("Please provide a test script.")
            else:
                calc_params = manager.calculate_parameters(emotion_id=test_emo, intensifier_override=multiplier)
                final_path = execute_synthesis(config, manager, store, provider, test_voice, test_lang, test_text, calc_params)
                if final_path:
                    st.audio(final_path)
                    st.caption(f"Preview -> Temp: {calc_params['temp']:.2f} | Penalty: {calc_params['penalty']:.2f} | Top P: {calc_params['top_p']:.2f}")

    if st.button("Save Intensifier", type="primary", use_container_width=True):
        is_valid, msg = manager.validate_entry(new_id, "intensifiers", local_names)
        if not is_valid:
            st.error(msg)
        else:
            clean_id = new_id.strip().lower().replace(" ", "_")
            manager.save_intensifier(clean_id, multiplier, local_names)
            st.rerun()


def render_explorer_tab(manager: EmotionManager, store: VoiceStore, config: AppConfig, provider: TTSModelProvider, itype: str) -> None:
    """
    Handles the rendering of the explorer tabs for either emotions or intensifiers, 
    including pagination controls and fuzzy searching.

    Args:
        manager (EmotionManager): The instance controlling the emotion catalog.
        store (VoiceStore): The instance providing access to voice profiles.
        config (AppConfig): The global application configuration.
        provider (TTSModelProvider): The cached model provider.
        itype (str): The category of items to render ('emotion' or 'intensifier').
    """
    st.subheader(f"{itype.capitalize()} Catalog Explorer")
    col_search, col_lang, col_add = st.columns([3, 1, 1])
    search_query = col_search.text_input(f"Search {itype}s...", key=f"search_{itype}")
    filter_lang = col_lang.selectbox("Display Language", SUPPORTED_LANGUAGES, index=0, key=f"lang_{itype}")
    
    with col_add:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("➕ Add New", use_container_width=True, key=f"add_{itype}"):
            if itype == "emotion":
                emotion_form_modal(manager, store, config, provider, ui_language=filter_lang)
            else:
                intensifier_form_modal(manager, store, config, provider, ui_language=filter_lang)
            
    vocab = manager.get_localized_vocab(lang=filter_lang)[f"{itype}s"]
    
    items = []
    for iid, data in manager.catalog.get(f"{itype}s", {}).items():
        display_name = vocab.get(iid, "")
        if is_fuzzy_match(search_query, iid) or is_fuzzy_match(search_query, display_name):
            items.append((iid, data))
            
    items.sort(key=lambda x: vocab.get(x[0], x[0]).lower())
            
    page_key = f"{itype}_page"
    total_pages = max(1, math.ceil(len(items) / ITEMS_PER_PAGE))
    
    if st.session_state[page_key] > total_pages:
        st.session_state[page_key] = max(1, total_pages)
        
    start_idx = (st.session_state[page_key] - 1) * ITEMS_PER_PAGE
    page_items = items[start_idx:start_idx + ITEMS_PER_PAGE]
    
    for iid, data in page_items:
        display_name = vocab.get(iid, iid)
        with st.expander(f"🎭 {display_name} (`{iid}`)", expanded=False):
            if itype == "emotion":
                mods = data.get("modifiers", {})
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Temp Delta", f"{mods.get('temp', 0.0):+.2f}")
                m_col2.metric("Penalty Delta", f"{mods.get('penalty', 0.0):+.2f}")
                m_col3.metric("Top P Delta", f"{mods.get('top_p', 0.0):+.2f}")
            else:
                st.metric("Multiplier", f"{data.get('multiplier', 1.0):.2f}x")
            
            with st.container(border=True):
                st.markdown("**🌐 Translations**")
                cols = st.columns(5)
                for idx, lang in enumerate(SUPPORTED_LANGUAGES):
                    cols[idx % 5].caption(f"**{lang.upper()}**: {data.get('localized_names', {}).get(lang, '—')}")
            
            _, b1, b2, b3 = st.columns([5, 1, 1, 1])
            if b1.button("Edit", key=f"edit_{itype}_{iid}", use_container_width=True): 
                if itype == "emotion":
                    emotion_form_modal(manager, store, config, provider, ui_language=filter_lang, emotion_id=iid)
                else:
                    intensifier_form_modal(manager, store, config, provider, ui_language=filter_lang, intensifier_id=iid)
            if b2.button("Delete", key=f"del_{itype}_{iid}", use_container_width=True): 
                delete_item_modal(manager, itype, iid)
            if b3.button("Test", key=f"test_{itype}_{iid}", use_container_width=True): 
                quick_test_modal(manager, store, config, provider, ui_language=filter_lang, fixed_emotion=iid if itype == "emotion" else None, 
                                 fixed_intensifier=iid if itype == "intensifier" else None)

    if total_pages > 1:
        st.divider()
        col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
        if col_p1.button("⬅️ Previous", disabled=(st.session_state[page_key] <= 1), key=f"prev_{itype}", use_container_width=True):
            st.session_state[page_key] -= 1
            st.rerun()
            
        col_p2.markdown(f"<div style='text-align: center; padding-top: 10px;'>Page {st.session_state[page_key]} of {total_pages}</div>", unsafe_allow_html=True)
        
        if col_p3.button("Next ➡️", disabled=(st.session_state[page_key] >= total_pages), key=f"next_{itype}", use_container_width=True):
            st.session_state[page_key] += 1
            st.rerun()


def render_test_tab(manager: EmotionManager, store: VoiceStore, config: AppConfig, provider: TTSModelProvider) -> None:
    """
    Renders the manual synthesis playground.

    Args:
        manager (EmotionManager): The instance controlling the emotion catalog.
        store (VoiceStore): The instance providing access to voice profiles.
        config (AppConfig): The global application configuration.
        provider (TTSModelProvider): The cached model provider.
    """
    st.subheader("Synthesis Playground")
    current_lang = st.selectbox("Display Language", SUPPORTED_LANGUAGES, index=0, key="test_meta_lang")
    vocab = manager.get_localized_vocab(lang=current_lang)
    default_lang_idx = SUPPORTED_LANGUAGES.index(current_lang) if current_lang in SUPPORTED_LANGUAGES else 1
    
    col1, col2, col3, col4 = st.columns(4)
    profiles = store.get_all()
    voice_options = {p.id: p.name for p in profiles}
    voice_ids = sorted(list(voice_options.keys()), key=lambda x: voice_options[x].lower())
    
    emotion_ids = sorted(list(vocab["emotions"].keys()), key=lambda x: vocab["emotions"][x].lower())
    int_ids = [None] + sorted(list(vocab["intensifiers"].keys()), key=lambda x: vocab["intensifiers"][x].lower())

    selected_voice = col1.selectbox("Voice", options=voice_ids, format_func=lambda x: voice_options[x], key="tp_voice")
    selected_emotion = col2.selectbox("Emotion", options=emotion_ids, format_func=lambda x: vocab["emotions"][x], key="tp_emo")
    selected_intensifier = col3.selectbox("Intensity", options=int_ids, format_func=lambda x: vocab["intensifiers"][x] if x else "None/Normal", key="tp_int")
    gen_lang = col4.selectbox("Generation Language", SUPPORTED_LANGUAGES, index=default_lang_idx, key="tp_lang")
    
    text = st.text_area("Test Script", placeholder="Enter dialogue to render...", key="tp_script")
    
    if st.button("Generate Synthesis", type="primary", use_container_width=True, key="tp_btn"):
        if not text:
            st.error("Please provide a test script.")
            return
            
        calculated = manager.calculate_parameters(emotion_id=selected_emotion, intensifier_id=selected_intensifier)
        emo_name = vocab["emotions"].get(selected_emotion, selected_emotion)
        int_name = vocab["intensifiers"].get(selected_intensifier, "None") if selected_intensifier else "None"
        
        final_path = execute_synthesis(config, manager, store, provider, selected_voice, gen_lang, text, calculated)
        
        if final_path:
            synthesis_result_modal(final_path, emo_name, int_name, calculated)


def main() -> None:
    """
    Main entry point for the Resona Emotion CAD application. Orchestrates 
    dependencies, session state, and sidebar maintenance tools.
    """
    st.set_page_config(page_title="Resona Emotion CAD", page_icon="🎭", layout="wide")
    initialize_session_state()
    config, manager, store, provider = load_system_dependencies()
    
    st.sidebar.title("🛠️ Maintenance")
    if st.sidebar.button("Clear Session Audios", use_container_width=True): confirm_clear_session_modal(manager)
    if st.sidebar.button("Wipe All Temp Audios", use_container_width=True): confirm_wipe_all_modal(manager)
    
    st.title("🎭 Resona Emotion CAD")
    tab_e, tab_i, tab_t = st.tabs(["📚 Emotions", "📈 Intensifiers", "🧪 Test"])
    
    with tab_e: render_explorer_tab(manager, store, config, provider, "emotion")
    with tab_i: render_explorer_tab(manager, store, config, provider, "intensifier")
    with tab_t: render_test_tab(manager, store, config, provider)


if __name__ == "__main__":
    main()