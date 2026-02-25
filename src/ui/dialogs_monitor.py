"""
UI component for monitoring and controlling the execution of dialog generation projects.
Includes integrated master audio synthesis and playback capabilities.
"""

import streamlit as st
import time
import json
import zipfile
import io
import psutil
from pathlib import Path
from typing import Callable, Optional

from src.config import settings
from src.models import DialogProject, ProjectStatus, LineStatus
from src.backend.store import VoiceStore
from src.emotions.manager import EmotionManager

@st.dialog("Confirm Project Deletion")
def delete_project_dialog(project_id: str, navigate_to: Callable[[str, Optional[str]], None]) -> None:
    """
    Displays a safety modal to confirm the permanent destruction of the project 
    from both the database and the filesystem.

    Args:
        project_id (str): The unique identifier of the project to be deleted.
        navigate_to (Callable[[str, Optional[str]], None]): The routing function used 
                                                            to redirect the user.
    """
    st.warning("Are you sure you want to permanently delete this project?")
    st.write("All generated audio files and script definitions will be lost.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Delete", type="primary", use_container_width=True):
            st.session_state.orchestrator.delete_project(project_id)
            st.success("Project deleted.")
            time.sleep(0.5)
            navigate_to("dashboard", None)
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

def render_monitor(navigate_to: Callable[[str, Optional[str]], None]) -> None:
    """
    Renders the execution monitor interface for a specific dialog project.

    This component utilizes strict Enum comparisons to evaluate execution states and 
    renders comprehensive metadata. It implements a static, non-interactive layout 
    for acoustic and context settings to maintain UI consistency with the editor 
    while optimizing for read-only monitoring.

    Args:
        navigate_to (Callable[[str, Optional[str]], None]): The core routing function used 
                                                            to transition between views.
    """
    from src.app_dialogs import LINES_PER_PAGE, POLLING_INTERVAL_SEC

    project_id = st.session_state.get("selected_project_id")
    if not project_id:
        st.error("No project selected.")
        return
    
    st.session_state.orchestrator.sync_status(project_id)
    
    db = st.session_state.db
    raw_data = db.dict("dialog_projects").get(project_id)
    
    if not raw_data:
        st.error("Project not found.")
        return

    project = DialogProject(**raw_data)
    
    is_active = project.status in [ProjectStatus.STARTING, ProjectStatus.GENERATING]
    is_completed = project.status == ProjectStatus.COMPLETED
    is_idle = project.status == ProjectStatus.IDLE
    
    if "voice_store" not in st.session_state:
        st.session_state.voice_store = VoiceStore(settings)
    if "emotion_manager" not in st.session_state:
        st.session_state.emotion_manager = EmotionManager(settings)

    voice_map = {v.id: v.name for v in st.session_state.voice_store.get_all()}

    total_lines = len(project.definition.script)
    completed_lines = sum(1 for s in project.states if s.status == LineStatus.COMPLETED)
    progress_pct = completed_lines / total_lines if total_lines > 0 else 0.0

    st.markdown(f"### **Monitor**: *{project.definition.name}* ")

    with st.container(border=True):
        status_colors = {
            ProjectStatus.IDLE: "gray", 
            ProjectStatus.COMPLETED: "green",
            ProjectStatus.GENERATING: "blue", 
            ProjectStatus.FAILED: "red",
            ProjectStatus.PAUSED: "orange", 
            ProjectStatus.STARTING: "blue"
        }
        color = status_colors.get(project.status, "gray")
        
        st.markdown(f"**Status:** :{color}[{project.status.value.upper()}]")
        st.progress(progress_pct, text=f"Progress: {completed_lines} / {total_lines} lines completed ({int(progress_pct * 100)}%)")
        
        error_msg = getattr(project, "error", raw_data.get("error"))
        if project.status == ProjectStatus.FAILED and error_msg:
            st.error(f"Execution Error: {error_msg}")

    with st.container(border=True):
        st.markdown("**Project Details & Metadata**")
        
        unique_scenes = len(set(l.scene for l in project.definition.script))
        unique_locations = len(set(l.scene_location for l in project.definition.script))
        unique_voices = len(set(l.voice_id for l in project.definition.script))
       
        m_cc1, m_cc2 = st.columns(2) 
        with m_cc1:
            st.write(f"**Name:** {project.definition.name}")
        with m_cc2:
            st.write(f"**ID:** {project.id}")
        st.divider()    
        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            st.write(f"**Base Language:** {project.definition.default_language.upper()}")
            st.write(f"**Voices:** {unique_voices}")
        with m_c2:
            st.write(f"**Tags:** {', '.join(project.definition.tags) if project.definition.tags else 'None'}")
            st.write(f"**Scenes:** {unique_scenes}")
        with m_c3:
            st.write(f"**Created At:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(project.definition.created_at))}")
            st.write(f"**Locations:** {unique_locations}")
            
        st.markdown(f"> **Description:** {project.definition.description if project.definition.description else 'No description provided.'}")

    st.markdown("**Global Controls & Export**")
    ctrl_1, ctrl_2, ctrl_3, ctrl_4, ctrl_5, ctrl_6, ctrl_7, ctrl_8 = st.columns(8)
    
    with ctrl_1:
        if st.button("▶️", help="Start", key="btn_start", use_container_width=True, disabled=is_active or is_completed):
            st.session_state.orchestrator.start_generation(project.id)
            st.rerun()
    with ctrl_2:
        if st.button("⏸️", help="Pause", key="btn_pause", use_container_width=True, disabled=not is_active):
            st.session_state.orchestrator.pause_generation(project.id)
            st.rerun()
    with ctrl_3:
        if st.button("⏹️", help="Cancel", key="btn_cancel", use_container_width=True, disabled=is_idle and not is_completed):
            st.session_state.orchestrator.cancel_generation(project.id)
            st.rerun()
    with ctrl_4:
        if st.button("🎬", help="Merge Audios", key="btn_merge", use_container_width=True, disabled=not is_completed or is_active):
            with st.spinner("Synthesizing timeline..."):
                st.session_state.orchestrator.merge_project_audio(project.id)
                st.rerun()
    with ctrl_5:
        script_json = project.definition.export_as_template()
        st.download_button(label="📄", help="Download JSON", data=script_json, file_name=f"{project.definition.name.replace(' ', '_')}_script.json", mime="application/json", use_container_width=True)
    with ctrl_6:
        if is_completed:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for state in project.states:
                    if state.status == LineStatus.COMPLETED and state.audio_path:
                        audio_p = Path(project.project_path) / state.audio_path
                        if audio_p.exists():
                            zip_file.write(audio_p, arcname=f"{state.index + 1:03d}_{state.line_id}.wav")
            st.download_button(label="📦", help="Download all audios", data=zip_buffer.getvalue(), file_name=f"{project.definition.name.replace(' ', '_')}_audio.zip", mime="application/zip", use_container_width=True)
        else:
            st.button("📦", help="Download all audios", use_container_width=True, disabled=True)
            
    with ctrl_7:
        if st.button("✏️", help="Edit", use_container_width=True, disabled=is_active):
            navigate_to("editor", project.id)

    with ctrl_8:
        if st.button("🗑️", help="Delete", type="secondary", use_container_width=True, disabled=is_active):
            delete_project_dialog(project.id, navigate_to)
            
    st.divider()

    master_path_str = raw_data.get("merged_audio_path")
    if master_path_str:
        master_full_path = Path(project.project_path) / master_path_str
        if master_full_path.exists():
            with st.container(border=True):
                st.markdown("### 🎙️ Master Timeline Output")
                mc1, mc2 = st.columns([5, 1], vertical_alignment="center")
                with mc1:
                    st.audio(str(master_full_path))
                with mc2:
                    with open(master_full_path, "rb") as f: master_bytes = f.read()
                    st.download_button(label="⬇️ Master WAV", data=master_bytes, file_name=f"master_{project.definition.name.replace(' ', '_')}.wav", mime="audio/wav", key=f"dl_master_{project.id}", use_container_width=True)

    

    st.markdown("**Search & Filter**")
    unique_v_ids = list(set(l.voice_id for l in project.definition.script))
    unique_langs = list(set(l.language or project.definition.default_language for l in project.definition.script))
    unique_emos = list(set(l.emotion for l in project.definition.script if l.emotion))
    
    r1_c1, r1_c2 = st.columns([2, 1])
    with r1_c1:
        search_q = st.text_input("Fuzzy Search", placeholder="Search text content, scenes or locations...").lower()
    with r1_c2:
        sel_voices = st.multiselect("Voice", options=unique_v_ids, format_func=lambda x: voice_map.get(x, x))
        
    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1:
        sel_langs = st.multiselect("Language", options=unique_langs)
    with r2_c2:
        sel_emos = st.multiselect("Emotion", options=unique_emos)
    with r2_c3:
        sel_statuses = st.multiselect("Status", options=list(LineStatus), format_func=lambda x: x.name)

    state_map = {s.line_id: s for s in project.states}
    filtered_lines = []
    
    for line in project.definition.script:
        l_state = state_map.get(line.id)
        if not l_state: continue
            
        m_search = not search_q or search_q in line.text.lower() or \
                   search_q in (line.scene or "").lower() or \
                   search_q in (line.scene_location or "").lower()

        m_voice = not sel_voices or line.voice_id in sel_voices
        m_lang = not sel_langs or (line.language or project.definition.default_language) in sel_langs
        m_emo = not sel_emos or line.emotion in sel_emos
        m_status = not sel_statuses or l_state.status in sel_statuses
        
        if m_search and m_voice and m_status and m_lang and m_emo:
            filtered_lines.append((line, l_state))

    total_filtered = len(filtered_lines)
    pages = (total_filtered - 1) // LINES_PER_PAGE + 1 if total_filtered > 0 else 1
    
    if "monitor_page" not in st.session_state: st.session_state.monitor_page = 1
    if st.session_state.monitor_page > pages: st.session_state.monitor_page = pages or 1

    start_idx = (st.session_state.monitor_page - 1) * LINES_PER_PAGE
    page_data = filtered_lines[start_idx:start_idx + LINES_PER_PAGE]

    st.write(f"Showing {len(page_data)} of {total_filtered} lines")

    for line, l_state in page_data:
        with st.container(border=True):
            r_head, r_stat = st.columns([5, 1], vertical_alignment="center")
            v_name = voice_map.get(line.voice_id, line.voice_id)
            lang = line.language or project.definition.default_language
            emo = line.emotion or "Neutral"
            intensity = line.intensity or "Default"
            
            with r_head:
                metadata_html = f"<span style='font-size: 0.85em; color: gray;'><b>#{l_state.index + 1}</b> | <b>Voice:</b> {v_name} | <b>Lang:</b> {lang} | <b>Emotion:</b> {emo} | <b>Intensity:</b> {intensity}</span>"
                st.markdown(metadata_html, unsafe_allow_html=True)
            with r_stat:
                if l_state.status == LineStatus.COMPLETED: 
                    st.markdown("<div style='background-color: rgba(25, 135, 84, 0.15); color: #198754; padding: 4px 8px; border-radius: 4px; text-align: center; font-size: 0.8em; font-weight: 600;'>COMPLETED</div>", unsafe_allow_html=True)
                elif l_state.status == LineStatus.FAILED: 
                    st.markdown("<div style='background-color: rgba(220, 53, 69, 0.15); color: #dc3545; padding: 4px 8px; border-radius: 4px; text-align: center; font-size: 0.8em; font-weight: 600;'>FAILED</div>", unsafe_allow_html=True)
                else: 
                    st.markdown(f"<div style='background-color: rgba(13, 202, 240, 0.15); color: #0dcaf0; padding: 4px 8px; border-radius: 4px; text-align: center; font-size: 0.8em; font-weight: 600;'>{l_state.status.value.upper()}</div>", unsafe_allow_html=True)

            with st.expander("📝 Script Text", expanded=False):
                st.write(f"*{line.text}*")
            
            with st.expander("🛠️ Acoustic & Context Settings", expanded=False):
                ctx_cols = st.columns(2)
                ctx_cols[0].markdown(f"**Scene Name:**\n{line.scene or 'None'}")
                ctx_cols[1].markdown(f"**Location:**\n{line.scene_location or 'None'}")
                
                aco_cols = st.columns(4)
                aco_cols[0].markdown(f"**Post Delay:**\n{line.post_delay_ms} ms")
                aco_cols[1].markdown(f"**Fade In:**\n{line.fade_in_ms} ms")
                aco_cols[2].markdown(f"**Fade Out:**\n{line.fade_out_ms} ms")
                aco_cols[3].markdown(f"**Room Tone:**\n{line.room_tone_level:.5f}")
                    
            if l_state.status == LineStatus.COMPLETED and l_state.audio_path:
                audio_file_path = Path(project.project_path) / l_state.audio_path
                if audio_file_path.exists():
                    with st.expander("🔊 Generated Audio", expanded=True):
                        ac1, ac2 = st.columns([5, 1], vertical_alignment="center")
                        with ac1:
                            st.audio(str(audio_file_path))
                        with ac2:
                            with open(audio_file_path, "rb") as f: audio_bytes = f.read()
                            st.download_button(label="⬇️ WAV", data=audio_bytes, file_name=f"line_{l_state.index + 1}.wav", mime="audio/wav", key=f"dl_{line.id}", use_container_width=True)

    if pages > 1:
        st.divider()
        p_c1, p_c2, p_c3 = st.columns([1, 2, 1])
        with p_c1:
            if st.button("Previous", disabled=st.session_state.monitor_page <= 1, use_container_width=True):
                st.session_state.monitor_page -= 1
                st.rerun()
        with p_c2: 
            st.markdown(f"<div style='text-align: center'>Page {st.session_state.monitor_page} of {pages}</div>", unsafe_allow_html=True)
        with p_c3:
            if st.button("Next", disabled=st.session_state.monitor_page >= pages, use_container_width=True):
                st.session_state.monitor_page += 1
                st.rerun()

    if is_active:
        time.sleep(POLLING_INTERVAL_SEC)
        st.rerun()