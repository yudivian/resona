import streamlit as st
import json
import uuid
from typing import Callable, Optional

from src.config import settings
from src.models import DialogProject, ProjectStatus, LineStatus
from src.backend.store import VoiceStore
from src.emotions.manager import EmotionManager


def _apply_changes(state: dict, project_id: str) -> bool:
    """
    Executes a comprehensive project update by validating data integrity, resetting
    execution states, and purging physical assets.

    This function invalidates any existing master audio merge by resetting the
    merged_audio_path to ensure the filesystem remains synchronized with the script.

    Args:
        state (dict): The in-memory buffer containing modified project data.
        project_id (str): The unique identifier of the project to update.

    Returns:
        bool: True if the operation succeeded, False if aborted due to active generation.
    """
    db = st.session_state.db
    current_db_state = db.dict("dialog_projects").get(project_id)

    if current_db_state and current_db_state.get("status") in [
        ProjectStatus.STARTING.value,
        ProjectStatus.GENERATING.value,
    ]:
        st.error(
            "Operation aborted: The project is actively generating in another session."
        )
        return False

    try:
        state["status"] = ProjectStatus.IDLE.value
        state["pid"] = None
        state["error"] = None
        state["merged_audio_path"] = None

        script = state["definition"].get("script", [])
        state["states"] = [
            {
                "line_id": line.get("id", str(uuid.uuid4())),
                "index": i,
                "status": LineStatus.PENDING.value,
                "emotion_id": line.get("emotion"),
                "intensity_id": line.get("intensity"),
                "audio_path": None,
            }
            for i, line in enumerate(script)
        ]

        project = DialogProject(**state)

        st.session_state.orchestrator.purge_project_assets(project_id)

        db.dict("dialog_projects")[project_id] = project.model_dump()
        st.session_state.editor_state = db.dict("dialog_projects").get(project_id)

        return True
    except Exception as e:
        st.error(f"Validation Error: {e}")
        return False


def render_editor(navigate_to: Callable[[str, Optional[str]], None]) -> None:
    """
    Renders the unified script editor interface.

    Provides a dual-mode editing experience (Visual/JSON) allowing for granular control
    over dialog parameters, emotional intensity, and acoustic pacing.

    Args:
        navigate_to (Callable[[str, Optional[str]], None]): The routing function used
                                                            to transition between views.
    """
    from src.app_dialogs import MAX_DIALOG_LINES

    project_id = st.session_state.get("selected_project_id")
    if not project_id:
        st.error("No project selected.")
        return

    if (
        "editor_state" not in st.session_state
        or st.session_state.editor_state.get("id") != project_id
    ):
        db = st.session_state.db
        raw_data = db.dict("dialog_projects").get(project_id)
        if not raw_data:
            st.error("Project not found in database.")
            return
        st.session_state.editor_state = raw_data

    state = st.session_state.editor_state
    is_locked = state.get("status") in [
        ProjectStatus.STARTING.value,
        ProjectStatus.GENERATING.value,
    ]
    line_count = len(state["definition"].get("script", []))

    if "voice_store" not in st.session_state:
        st.session_state.voice_store = VoiceStore(settings)
    if "emotion_manager" not in st.session_state:
        st.session_state.emotion_manager = EmotionManager(settings)

    voices = st.session_state.voice_store.get_all()
    voice_map = {v.id: v.name for v in voices}
    emotions_cat = st.session_state.emotion_manager.catalog.get("emotions", {})
    intensities_cat = st.session_state.emotion_manager.catalog.get("intensifiers", {})
    lang_options = ["es", "en", "fr", "zh", "ja", "ko"]

    st.markdown(f"### **Editor**: *{state['definition'].get('name', 'Unknown')}*")

    if is_locked:
        st.warning(f"Project is {state.get('status')}. Structural edits are disabled.")

    if line_count <= MAX_DIALOG_LINES:
        edit_mode = st.radio(
            "Editing Mode",
            ["Visual Editor", "JSON Editor"],
            horizontal=True,
            disabled=is_locked,
        )
    else:
        st.info(
            f"Visual Editor disabled for massive projects ({line_count} lines). Limit is {MAX_DIALOG_LINES}."
        )
        edit_mode = "JSON Editor"

    st.divider()

    if edit_mode == "JSON Editor":
        with st.container(border=True):
            st.subheader("Bulk JSON Update")
            st.write("Upload a complete JSON project definition (metadata and script).")
            up_file = st.file_uploader(
                "Upload Script JSON", type=["json"], disabled=is_locked
            )
            if up_file and not is_locked:
                try:
                    new_data = json.load(up_file)
                    new_def = (
                        new_data["definition"] if "definition" in new_data else new_data
                    )

                    script_list = new_def.get("script", [])
                    for l in script_list:
                        if "id" not in l:
                            l["id"] = str(uuid.uuid4())

                    state["definition"] = new_def
                    st.success(
                        "JSON loaded into memory. Click 'Save Changes' below to apply."
                    )
                    with st.expander("Preview Loaded Definition"):
                        st.json(state["definition"])
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
    else:
        with st.container(border=True):
            st.subheader("Project Configuration")
            c_meta = st.columns([2, 1, 1.5])
            with c_meta[0]:
                state["definition"]["name"] = st.text_input(
                    "Project Name",
                    value=state["definition"].get("name", ""),
                    disabled=is_locked,
                )
            with c_meta[1]:
                state["definition"]["default_language"] = st.selectbox(
                    "Base Lang",
                    options=lang_options,
                    index=lang_options.index(
                        state["definition"].get("default_language", "es")
                    ),
                    disabled=is_locked,
                )
            with c_meta[2]:
                tags_val = ", ".join(state["definition"].get("tags", []))
                p_tags_raw = st.text_input("Tags", value=tags_val, disabled=is_locked)
                state["definition"]["tags"] = (
                    [t.strip() for t in p_tags_raw.split(",")] if p_tags_raw else []
                )

            state["definition"]["description"] = st.text_area(
                "Description",
                value=state["definition"].get("description") or "",
                height=68,
                disabled=is_locked,
            )

        with st.container(border=True):
            st.markdown("**Global Mastering Settings**")
            with st.expander(
                "Configure final mix dynamics (LUFS & Compression)", expanded=False
            ):
                if "mastering" not in state["definition"]:
                    state["definition"]["mastering"] = {
                        "target_lufs": -14.0,
                        "compressor_ratio": 3.0,
                        "compressor_threshold": -20.0,
                    }

                m_dict = state["definition"]["mastering"]
                m_cols = st.columns(3)
                with m_cols[0]:
                    m_dict["target_lufs"] = st.number_input(
                        "Target LUFS",
                        value=float(m_dict.get("target_lufs", -14.0)),
                        min_value=-24.0,
                        max_value=-6.0,
                        step=0.5,
                        disabled=is_locked,
                    )
                with m_cols[1]:
                    m_dict["compressor_ratio"] = st.number_input(
                        "Comp. Ratio",
                        value=float(m_dict.get("compressor_ratio", 3.0)),
                        min_value=1.0,
                        max_value=20.0,
                        step=0.5,
                        disabled=is_locked,
                    )
                with m_cols[2]:
                    m_dict["compressor_threshold"] = st.number_input(
                        "Comp. Threshold (dB)",
                        value=float(m_dict.get("compressor_threshold", -20.0)),
                        min_value=-60.0,
                        max_value=0.0,
                        step=1.0,
                        disabled=is_locked,
                    )

        st.subheader(f"Script Composition ({line_count}/{MAX_DIALOG_LINES})")
        for i, line in enumerate(state["definition"].get("script", [])):
            with st.container(border=True):
                row = st.columns([1.5, 0.7, 1.5, 1.2, 0.4])
                with row[0]:
                    line["voice_id"] = st.selectbox(
                        "Voice",
                        options=list(voice_map.keys()),
                        format_func=lambda x: voice_map.get(x, x),
                        index=(
                            list(voice_map.keys()).index(line["voice_id"])
                            if line["voice_id"] in voice_map
                            else 0
                        ),
                        key=f"v_{i}",
                        disabled=is_locked,
                    )
                with row[1]:
                    line["language"] = st.selectbox(
                        "Lang",
                        options=lang_options,
                        index=lang_options.index(
                            line.get(
                                "language",
                                state["definition"].get("default_language", "es"),
                            )
                        ),
                        key=f"l_{i}",
                        disabled=is_locked,
                    )
                with row[2]:
                    emo_opts = {"": "None (Neutral)"}
                    for eid, edata in emotions_cat.items():
                        loc = edata.get("localized_names", {}).get(
                            line.get("language", "en"),
                            edata.get("localized_names", {}).get("en", eid),
                        )
                        emo_opts[eid] = loc
                    line["emotion"] = st.selectbox(
                        "Emotion",
                        options=list(emo_opts.keys()),
                        format_func=lambda x: emo_opts[x],
                        index=(
                            list(emo_opts.keys()).index(line.get("emotion"))
                            if line.get("emotion") in emo_opts
                            else 0
                        ),
                        key=f"e_{i}",
                        disabled=is_locked,
                    )
                with row[3]:
                    int_opts = {"": "Default"}
                    for iid, idata in intensities_cat.items():
                        loc_i = idata.get("localized_names", {}).get(
                            line.get("language", "en"),
                            idata.get("localized_names", {}).get("en", iid),
                        )
                        int_opts[iid] = loc_i
                    has_emo = bool(line.get("emotion"))
                    line["intensity"] = st.selectbox(
                        "Intensity",
                        options=list(int_opts.keys()),
                        format_func=lambda x: int_opts[x],
                        index=(
                            list(int_opts.keys()).index(line.get("intensity"))
                            if has_emo and line.get("intensity") in int_opts
                            else 0
                        ),
                        key=f"i_{i}",
                        disabled=not has_emo or is_locked,
                    )
                with row[4]:
                    st.markdown(
                        "<div style='padding-top: 28.5px;'></div>",
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        "🗑️",
                        key=f"del_{i}",
                        disabled=is_locked or line_count <= 1,
                        use_container_width=True,
                    ):
                        state["definition"]["script"].pop(i)
                        st.rerun()

                with st.expander("🛠️ Acoustic & Context Settings", expanded=False):
                    ctx_cols = st.columns(2)
                    with ctx_cols[0]:
                        line["scene"] = st.text_input(
                            "Scene Name",
                            value=line.get("scene", ""),
                            key=f"sc_{i}",
                            placeholder="e.g. Scene 1",
                            disabled=is_locked,
                        )
                    with ctx_cols[1]:
                        line["scene_location"] = st.text_input(
                            "Location",
                            value=line.get("scene_location", ""),
                            key=f"sl_{i}",
                            placeholder="e.g. INT. OFFICE",
                            disabled=is_locked,
                        )

                    aco_cols = st.columns(4)
                    with aco_cols[0]:
                        line["post_delay_ms"] = st.number_input(
                            "Post Delay (ms)",
                            value=line.get("post_delay_ms", 400),
                            step=50,
                            key=f"pd_{i}",
                            disabled=is_locked,
                        )
                    with aco_cols[1]:
                        line["fade_in_ms"] = st.number_input(
                            "Fade In (ms)",
                            value=line.get("fade_in_ms", 50),
                            step=10,
                            key=f"fi_{i}",
                            disabled=is_locked,
                        )
                    with aco_cols[2]:
                        line["fade_out_ms"] = st.number_input(
                            "Fade Out (ms)",
                            value=line.get("fade_out_ms", 50),
                            step=10,
                            key=f"fo_{i}",
                            disabled=is_locked,
                        )
                    with aco_cols[3]:
                        line["room_tone_level"] = st.number_input(
                            "Room Tone",
                            value=line.get("room_tone_level", 0.0001),
                            format="%.5f",
                            step=0.00005,
                            key=f"rt_{i}",
                            disabled=is_locked,
                        )

                    mix_cols = st.columns(3)
                    with mix_cols[0]:
                        line["gain_db"] = st.slider(
                            "Gain (dB)",
                            min_value=-12.0,
                            max_value=12.0,
                            value=float(line.get("gain_db", 0.0)),
                            step=0.5,
                            key=f"gain_{i}",
                            disabled=is_locked,
                        )
                    with mix_cols[1]:
                        line["pan"] = st.slider(
                            "Pan (L/R)",
                            min_value=-1.0,
                            max_value=1.0,
                            value=float(line.get("pan", 0.0)),
                            step=0.1,
                            key=f"pan_{i}",
                            disabled=is_locked,
                        )
                    with mix_cols[2]:
                        line["depth"] = st.slider(
                            "Depth",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(line.get("depth", 0.0)),
                            step=0.05,
                            key=f"depth_{i}",
                            disabled=is_locked,
                        )

                line["text"] = st.text_area(
                    "Text",
                    value=line.get("text", ""),
                    key=f"t_{i}",
                    height=90,
                    disabled=is_locked,
                )

        if line_count < MAX_DIALOG_LINES:
            if st.button("➕ Add Line", use_container_width=True, disabled=is_locked):
                new_line = {
                    "id": str(uuid.uuid4()),
                    "index": line_count,
                    "voice_id": list(voice_map.keys())[0],
                    "text": "",
                    "language": state["definition"].get("default_language", "es"),
                    "emotion": None,
                    "intensity": None,
                    "scene": "",
                    "scene_location": "",
                    "post_delay_ms": 400,
                    "fade_in_ms": 50,
                    "fade_out_ms": 50,
                    "room_tone_level": 0.0001,
                    "gain_db": 0.0,
                    "depth": 0.0,
                    "post_delay_ms": 400,
                }
                state["definition"]["script"].append(new_line)
                st.rerun()

    st.divider()
    f_col1, f_col2 = st.columns(2)

    validation_passed = True
    val_error = ""

    if not state["definition"].get("name", "").strip():
        validation_passed = False
        val_error = "Project name is required."
    elif any(
        not l.get("text", "").strip() for l in state["definition"].get("script", [])
    ):
        validation_passed = False
        val_error = "All lines must contain text before saving."

    with f_col1:
        if st.button("💾 Save Changes", use_container_width=True, disabled=is_locked):
            if validation_passed:
                if _apply_changes(state, project_id):
                    st.success("Project updated and reset successfully.")
            else:
                st.error(val_error)
    with f_col2:
        if st.button(
            "🚀 Save and Go to Monitor", type="primary", use_container_width=True
        ):
            if validation_passed:
                if _apply_changes(state, project_id):
                    navigate_to("monitor", project_id)
            else:
                st.error(val_error)
