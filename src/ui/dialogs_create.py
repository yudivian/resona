import streamlit as st
import time
from typing import Callable, Optional, List

from src.config import settings
from src.models import (
    DialogProject, DialogScript, DialogLine, LineState, 
    ProjectSource, ProjectStatus, LineStatus
)
from src.backend.store import VoiceStore
from src.emotions.manager import EmotionManager

def _persist_project(
    name: str, 
    language: str, 
    description: str, 
    tags: List[str],
    lines_data: list
) -> str:
    """
    Constructs and persists the DialogProject to BeaverDB.
    Hardcodes ProjectSource.UI and maps all fields from models.py.
    """
    db = st.session_state.db
    projects_dict = db.dict("dialog_projects")
    
    script_lines = []
    for i, data in enumerate(lines_data):
        script_lines.append(DialogLine(
            index=i,
            voice_id=data["voice_id"],
            text=data["text"].strip(),
            language=data["language"],
            emotion=data["emotion"] if data["emotion"] else None,
            intensity=data["intensity"] if data["emotion"] and data["intensity"] else None
        ))
    
    script = DialogScript(
        name=name,
        description=description,
        tags=tags,
        default_language=language,
        script=script_lines
    )
    
    states = [
        LineState(
            line_id=line.id,
            index=line.index,
            status=LineStatus.PENDING,
            emotion_id=line.emotion,
            intensity_id=line.intensity
        )
        for line in script.script
    ]
    
    project = DialogProject(
        source=ProjectSource.UI,
        definition=script,
        states=states,
        status=ProjectStatus.IDLE,
        project_path=f"{settings.paths.dialogspace_dir}/{script.id}"
    )
    
    projects_dict[project.id] = project.model_dump()
    return project.id

def render_create(navigate_to: Callable[[str, Optional[str]], None]) -> None:
    """
    Optimized single-column creation view with metadata in one row and strict validation.
    """
    from src.app_dialogs import MAX_DIALOG_LINES

    if "voice_store" not in st.session_state:
        st.session_state.voice_store = VoiceStore(settings)
    if "emotion_manager" not in st.session_state:
        st.session_state.emotion_manager = EmotionManager(settings)

    if st.button("← Back to Dashboard"):
        navigate_to("dashboard", None)

    st.title("✨ Create New Project")
    
    LANG_OPTIONS = ["es", "en", "fr", "zh", "ja", "ko"]

    if "temp_lines" not in st.session_state:
        st.session_state.temp_lines = [{"voice_id": "", "text": "", "language": "es", "emotion": "", "intensity": ""}]

    # 1. Metadata Container - Optimized space
    with st.container(border=True):
        st.subheader("Project Configuration")
        # Condensed row for Name, Language and Tags
        c_meta = st.columns([2, 1, 1.5])
        with c_meta[0]:
            p_name = st.text_input("Project Name", placeholder="e.g. Documentary V1")
        with c_meta[1]:
            p_lang = st.selectbox("Base Lang", options=LANG_OPTIONS)
        with c_meta[2]:
            p_tags_raw = st.text_input("Tags", help="Comma-separated (e.g. ad, male)")
        
        p_desc = st.text_area("Description (Optional)", height=68, placeholder="Context...")
        p_tags = [t.strip() for t in p_tags_raw.split(",")] if p_tags_raw else []

    # Assets
    voices = st.session_state.voice_store.get_all()
    voice_map = {v.id: v.name for v in voices}
    emotions_cat = st.session_state.emotion_manager.catalog.get("emotions", {})
    intensities_cat = st.session_state.emotion_manager.catalog.get("intensifiers", {})

    st.subheader(f"Composition ({len(st.session_state.temp_lines)}/{MAX_DIALOG_LINES})")
    
    updated_lines = []
    for i, line in enumerate(st.session_state.temp_lines):
        with st.container(border=True):
            # Control Row: Voice, Lang, Emotion, Intensity, Delete
            row = st.columns([1.5, 0.7, 1.5, 1.2, 0.4])
            
            with row[0]:
                v_id = st.selectbox("Voice", options=list(voice_map.keys()), format_func=lambda x: voice_map.get(x, x), key=f"v_{i}")
            with row[1]:
                l_lang = st.selectbox("Lang", options=LANG_OPTIONS, index=LANG_OPTIONS.index(line.get("language", p_lang)), key=f"l_{i}")
            with row[2]:
                emo_opts = {"": "None (Neutral)"}
                for eid, edata in emotions_cat.items():
                    loc = edata.get("localized_names", {}).get(l_lang, edata.get("localized_names", {}).get("en", eid))
                    emo_opts[eid] = loc
                selected_emo = st.selectbox("Emotion", options=list(emo_opts.keys()), format_func=lambda x: emo_opts[x], key=f"e_{i}")
            with row[3]:
                int_opts = {"": "Default"}
                for iid, idata in intensities_cat.items():
                    loc_i = idata.get("localized_names", {}).get(l_lang, idata.get("localized_names", {}).get("en", iid))
                    int_opts[iid] = loc_i
                
                is_emo_set = selected_emo != ""
                selected_int = st.selectbox(
                    "Intensity", 
                    options=list(int_opts.keys()), 
                    format_func=lambda x: int_opts[x], 
                    key=f"i_{i}",
                    disabled=not is_emo_set,
                    index=0 if not is_emo_set else list(int_opts.keys()).index(line["intensity"]) if line["intensity"] in int_opts else 0
                )
            with row[4]:
                # Precision Alignment: Structural padding for the delete button
                st.markdown("<div style='padding-top: 28.5px;'></div>", unsafe_allow_html=True)
                if st.button("🗑️", key=f"del_{i}", help="Delete this dialogue line", use_container_width=True):
                    if len(st.session_state.temp_lines) > 1:
                        st.session_state.temp_lines.pop(i)
                        st.rerun()
            
            t_val = st.text_area("Text", value=line["text"], key=f"t_{i}", height=90, placeholder="Text to generate...")

            updated_lines.append({
                "voice_id": v_id, 
                "text": t_val, 
                "language": l_lang, 
                "emotion": selected_emo, 
                "intensity": selected_int if is_emo_set else ""
            })

    st.session_state.temp_lines = updated_lines

    if len(st.session_state.temp_lines) < MAX_DIALOG_LINES:
        if st.button("➕ Add Line", use_container_width=True):
            st.session_state.temp_lines.append({
                "voice_id": list(voice_map.keys())[0] if voice_map else "", 
                "text": "", "language": p_lang, "emotion": "", "intensity": ""
            })
            st.rerun()

    st.divider()

    # Final Validation
    if st.button("🚀 Initialize Project", type="primary", use_container_width=True):
        valid_lines = [l for l in st.session_state.temp_lines if l["voice_id"] and l["text"].strip()]
        
        if not p_name.strip():
            st.error("Project name is required.")
        elif not st.session_state.temp_lines:
            st.error("At least one line is required.")
        elif len(valid_lines) != len(st.session_state.temp_lines):
            st.error("Every line must have a voice selected and valid text.")
        else:
            pid = _persist_project(p_name, p_lang, p_desc, p_tags, st.session_state.temp_lines)
            st.success("Project ready.")
            del st.session_state.temp_lines
            time.sleep(0.5)
            navigate_to("editor", pid)