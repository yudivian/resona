import streamlit as st
import time
from typing import Callable, Optional, List

from src.config import settings
from src.models import (
    DialogProject,
    DialogScript,
    DialogLine,
    LineState,
    ProjectSource,
    ProjectStatus,
    LineStatus,
    MasteringConfig,
)
from src.backend.store import VoiceStore
from src.emotions.manager import EmotionManager


def _persist_project(
    name: str,
    language: str,
    description: str,
    tags: List[str],
    mastering_data: dict,
    lines_data: list,
) -> str:
    """
    Constructs and persists a new DialogProject instance to the database.

    Args:
        name (str): The display name of the project.
        language (str): The primary ISO language code.
        description (str): Detailed context or summary of the script.
        tags (List[str]): Categorization labels for searching.
        nastering_data (List): Parameter for final mastering audio merge
        lines_data (list): Raw dictionary list containing line parameters.

    Returns:
        str: The unique identifier of the created project.
    """
    db = st.session_state.db
    projects_dict = db.dict("dialog_projects")

    script_lines = []
    for i, data in enumerate(lines_data):
        script_lines.append(
            DialogLine(
                index=i,
                voice_id=data["voice_id"],
                text=data["text"].strip(),
                language=data["language"],
                emotion=data["emotion"] if data["emotion"] else None,
                intensity=(
                    data["intensity"] if data["emotion"] and data["intensity"] else None
                ),
                scene=data.get("scene") if data.get("scene") else None,
                scene_location=(
                    data.get("scene_location") if data.get("scene_location") else None
                ),
                post_delay_ms=data.get("post_delay_ms", 400),
                fade_in_ms=data.get("fade_in_ms", 50),
                fade_out_ms=data.get("fade_out_ms", 50),
                room_tone_level=data.get("room_tone_level", 0.0001),
                pan=data.get("pan", 0.0),
                gain_db=data.get("gain_db", 0.0),
                depth=data.get("depth", 0.0),
            )
        )

    script = DialogScript(
        name=name,
        description=description,
        tags=tags,
        default_language=language,
        mastering=MasteringConfig(**mastering_data),
        script=script_lines,
    )

    states = [
        LineState(
            line_id=line.id,
            index=line.index,
            status=LineStatus.PENDING,
            emotion_id=line.emotion,
            intensity_id=line.intensity,
        )
        for line in script.script
    ]

    project = DialogProject(
        source=ProjectSource.UI,
        definition=script,
        states=states,
        status=ProjectStatus.IDLE,
        project_path=f"{settings.paths.dialogspace_dir}/{script.id}",
        merged_audio_path=None,
    )

    projects_dict[project.id] = project.model_dump()
    return project.id


def render_create(navigate_to: Callable[[str, Optional[str]], None]) -> None:
    """
    Renders the multi-step project creation form.

    Integrates voice selection, emotional mapping, and advanced acoustic pacing
    controls for high-fidelity dialog synthesis.

    Args:
        navigate_to (Callable[[str, Optional[str]], None]): Navigation router function.
    """
    from src.app_dialogs import MAX_DIALOG_LINES

    if "voice_store" not in st.session_state:
        st.session_state.voice_store = VoiceStore(settings)
    if "emotion_manager" not in st.session_state:
        st.session_state.emotion_manager = EmotionManager(settings)

    st.markdown(f"### **Create New Project**")

    LANG_OPTIONS = ["es", "en", "fr", "zh", "ja", "ko"]

    if "temp_mastering" not in st.session_state:
        st.session_state.temp_mastering = {
            "target_lufs": -14.0,
            "compressor_ratio": 3.0,
            "compressor_threshold": -20.0,
        }

    if "temp_lines" not in st.session_state:
        st.session_state.temp_lines = [
            {
                "voice_id": "",
                "text": "",
                "language": "es",
                "emotion": "",
                "intensity": "",
                "scene": "",
                "scene_location": "",
                "post_delay_ms": 400,
                "fade_in_ms": 50,
                "fade_out_ms": 50,
                "room_tone_level": 0.0001,
                "pan": 0.0,
                "gain_db": 0.0,
                "depth": 0.0,
            }
        ]

    with st.container(border=True):
        st.subheader("Project Configuration")
        c_meta = st.columns([2, 1, 1.5])
        with c_meta[0]:
            p_name = st.text_input("Project Name", placeholder="e.g. Documentary V1")
        with c_meta[1]:
            p_lang = st.selectbox("Base Lang", options=LANG_OPTIONS)
        with c_meta[2]:
            p_tags_raw = st.text_input("Tags", help="Comma-separated (e.g. ad, male)")

        p_desc = st.text_area(
            "Description (Optional)", height=68, placeholder="Context..."
        )
        p_tags = [t.strip() for t in p_tags_raw.split(",")] if p_tags_raw else []

    with st.container(border=True):
        st.markdown("**Global Mastering Settings**")
        with st.expander(
            "Configure final mix dynamics (LUFS & Compression)", expanded=False
        ):
            m_cols = st.columns(3)
            with m_cols[0]:
                m_lufs = st.number_input(
                    "Target LUFS",
                    value=st.session_state.temp_mastering["target_lufs"],
                    min_value=-24.0,
                    max_value=-6.0,
                    step=0.5,
                )
            with m_cols[1]:
                m_ratio = st.number_input(
                    "Comp. Ratio",
                    value=st.session_state.temp_mastering["compressor_ratio"],
                    min_value=1.0,
                    max_value=20.0,
                    step=0.5,
                )
            with m_cols[2]:
                m_thresh = st.number_input(
                    "Comp. Threshold (dB)",
                    value=st.session_state.temp_mastering["compressor_threshold"],
                    min_value=-60.0,
                    max_value=0.0,
                    step=1.0,
                )

        st.session_state.temp_mastering = {
            "target_lufs": m_lufs,
            "compressor_ratio": m_ratio,
            "compressor_threshold": m_thresh,
        }

    voices = st.session_state.voice_store.get_all()
    voice_map = {v.id: v.name for v in voices}
    emotions_cat = st.session_state.emotion_manager.catalog.get("emotions", {})
    intensities_cat = st.session_state.emotion_manager.catalog.get("intensifiers", {})

    st.subheader(f"Composition ({len(st.session_state.temp_lines)}/{MAX_DIALOG_LINES})")

    updated_lines = []
    for i, line in enumerate(st.session_state.temp_lines):
        with st.container(border=True):
            row = st.columns([1.5, 0.7, 1.5, 1.2, 0.4])

            with row[0]:
                v_id = st.selectbox(
                    "Voice",
                    options=list(voice_map.keys()),
                    format_func=lambda x: voice_map.get(x, x),
                    key=f"v_{i}",
                )
            with row[1]:
                l_lang = st.selectbox(
                    "Lang",
                    options=LANG_OPTIONS,
                    index=LANG_OPTIONS.index(line.get("language", p_lang)),
                    key=f"l_{i}",
                )
            with row[2]:
                emo_opts = {"": "None (Neutral)"}
                for eid, edata in emotions_cat.items():
                    loc = edata.get("localized_names", {}).get(
                        l_lang, edata.get("localized_names", {}).get("en", eid)
                    )
                    emo_opts[eid] = loc
                selected_emo = st.selectbox(
                    "Emotion",
                    options=list(emo_opts.keys()),
                    format_func=lambda x: emo_opts[x],
                    key=f"e_{i}",
                )
            with row[3]:
                int_opts = {"": "Default"}
                for iid, idata in intensities_cat.items():
                    loc_i = idata.get("localized_names", {}).get(
                        l_lang, idata.get("localized_names", {}).get("en", iid)
                    )
                    int_opts[iid] = loc_i

                is_emo_set = selected_emo != ""
                selected_int = st.selectbox(
                    "Intensity",
                    options=list(int_opts.keys()),
                    format_func=lambda x: int_opts[x],
                    key=f"i_{i}",
                    disabled=not is_emo_set,
                    index=(
                        0
                        if not is_emo_set
                        else (
                            list(int_opts.keys()).index(line["intensity"])
                            if line["intensity"] in int_opts
                            else 0
                        )
                    ),
                )
            with row[4]:
                st.markdown(
                    "<div style='padding-top: 28.5px;'></div>", unsafe_allow_html=True
                )
                if st.button(
                    "🗑️",
                    key=f"del_{i}",
                    help="Delete this dialogue line",
                    use_container_width=True,
                ):
                    if len(st.session_state.temp_lines) > 1:
                        st.session_state.temp_lines.pop(i)
                        st.rerun()

            with st.expander("🛠️ Acoustic & Context Settings", expanded=False):
                ctx_cols = st.columns(2)
                with ctx_cols[0]:
                    l_scene = st.text_input(
                        "Scene Name",
                        value=line.get("scene", ""),
                        key=f"sc_{i}",
                        placeholder="e.g. Scene 1",
                    )
                with ctx_cols[1]:
                    l_loc = st.text_input(
                        "Location",
                        value=line.get("scene_location", ""),
                        key=f"sl_{i}",
                        placeholder="e.g. INT. OFFICE",
                    )

                aco_cols = st.columns(4)
                with aco_cols[0]:
                    l_delay = st.number_input(
                        "Post Delay (ms)",
                        value=line.get("post_delay_ms", 400),
                        step=50,
                        key=f"pd_{i}",
                    )
                with aco_cols[1]:
                    l_fin = st.number_input(
                        "Fade In (ms)",
                        value=line.get("fade_in_ms", 50),
                        step=10,
                        key=f"fi_{i}",
                    )
                with aco_cols[2]:
                    l_fout = st.number_input(
                        "Fade Out (ms)",
                        value=line.get("fade_out_ms", 50),
                        step=10,
                        key=f"fo_{i}",
                    )
                with aco_cols[3]:
                    l_room = st.number_input(
                        "Room Tone",
                        value=line.get("room_tone_level", 0.0001),
                        format="%.5f",
                        step=0.00005,
                        key=f"rt_{i}",
                    )
                mix_cols = st.columns(3)
                with mix_cols[0]:
                    l_gain = st.slider(
                        "Gain (dB)",
                        min_value=-12.0,
                        max_value=12.0,
                        value=float(line.get("gain_db", 0.0)),
                        step=0.5,
                        key=f"gain_{i}",
                    )
                with mix_cols[1]:
                    l_pan = st.slider(
                        "Pan (L/R)",
                        min_value=-1.0,
                        max_value=1.0,
                        value=float(line.get("pan", 0.0)),
                        step=0.1,
                        key=f"pan_{i}",
                    )
                with mix_cols[2]:
                    l_depth = st.slider(
                        "Depth",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(line.get("depth", 0.0)),
                        step=0.05,
                        key=f"depth_{i}",
                    )

            t_val = st.text_area(
                "Text",
                value=line["text"],
                key=f"t_{i}",
                height=90,
                placeholder="Text to generate...",
            )

            updated_lines.append(
                {
                    "voice_id": v_id,
                    "text": t_val,
                    "language": l_lang,
                    "emotion": selected_emo,
                    "intensity": selected_int if is_emo_set else "",
                    "scene": l_scene,
                    "scene_location": l_loc,
                    "post_delay_ms": l_delay,
                    "fade_in_ms": l_fin,
                    "fade_out_ms": l_fout,
                    "room_tone_level": l_room,
                    "pan": l_pan,
                    "gain_db": l_gain,
                    "depth": l_depth,
                }
            )

    st.session_state.temp_lines = updated_lines

    if len(st.session_state.temp_lines) < MAX_DIALOG_LINES:
        if st.button("➕ Add Line", use_container_width=True):
            st.session_state.temp_lines.append(
                {
                    "voice_id": list(voice_map.keys())[0] if voice_map else "",
                    "text": "",
                    "language": p_lang,
                    "emotion": "",
                    "intensity": "",
                    "scene": "",
                    "scene_location": "",
                    "post_delay_ms": 400,
                    "fade_in_ms": 50,
                    "fade_out_ms": 50,
                    "room_tone_level": 0.0001,
                }
            )
            st.rerun()

    st.divider()

    if st.button(
        "🚀 Initialize Project and Go to Monitor",
        type="primary",
        use_container_width=True,
    ):
        valid_lines = [
            l
            for l in st.session_state.temp_lines
            if l["voice_id"] and l["text"].strip()
        ]

        if not p_name.strip():
            st.error("Project name is required.")
        elif not st.session_state.temp_lines:
            st.error("At least one line is required.")
        elif len(valid_lines) != len(st.session_state.temp_lines):
            st.error("Every line must have a voice selected and valid text.")
        else:
            pid = _persist_project(
                p_name,
                p_lang,
                p_desc,
                p_tags,
                st.session_state.temp_mastering,
                st.session_state.temp_lines,   
            )    
            st.success("Project ready.")
            del st.session_state.temp_lines
            time.sleep(0.5)
            navigate_to("monitor", pid)
