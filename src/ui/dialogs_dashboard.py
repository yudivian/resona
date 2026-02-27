import streamlit as st
import time
from pathlib import Path
from typing import Callable, Optional, List
from src.models import DialogProject, ProjectStatus, LineStatus

@st.dialog("Confirm Project Deletion")
def delete_confirmation_dialog(project_id: str, project_name: str) -> None:
    """
    Displays a safety modal to confirm the permanent removal of a project and its records.

    Args:
        project_id (str): The unique identifier of the project to delete.
        project_name (str): The display name of the project to be shown in the warning.
    """
    st.warning(f"Confirm permanent deletion of: {project_name}")
    st.write("This action will purge all project records from the database. This cannot be undone.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Yes, Delete", type="primary", use_container_width=True):
            st.session_state.orchestrator.delete_project(project_id)
            st.success("Project purged.")
            time.sleep(0.5)
            st.rerun()
    with col_b:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

def render_dashboard(navigate_to: Callable[[str, Optional[str]], None]) -> None:
    """
    Renders the centralized project management dashboard. 

    Includes deep fuzzy search across script lines and metadata, advanced filtering 
    by language and execution state, and comprehensive aggregate statistics evaluating 
    unique spatial, chronological, and vocal script components distributed in a 3-column layout.
    Provides direct access to script JSON and master audio exports when available.

    Args:
        navigate_to (Callable[[str, Optional[str]], None]): The routing function used 
                                                            to transition between views.
    """
    from src.app_dialogs import PROJECTS_PER_PAGE
    
    db = st.session_state.db
    projects_dict = db.dict("dialog_projects")
    
    if not projects_dict:
        st.info("No projects found in the workspace. Use the sidebar to create or import one.")
        return

    all_projects = []
    for p_id, p_data in projects_dict.items():
        if p_data:
            all_projects.append(DialogProject(**p_data))

    with st.container(border=True):
        st.markdown("**Filters & Search**")
        f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
        
        with f_col1:
            query = st.text_input("Fuzzy Search", placeholder="Name, tags, description, or scenes...").lower()
        with f_col2:
            langs = sorted(list(set(p.definition.default_language for p in all_projects)))
            sel_lang = st.multiselect("Base Language", options=langs)
        with f_col3:
            stats = [s.value for s in ProjectStatus]
            sel_status = st.multiselect("Status", options=stats)

    filtered = []
    for p in all_projects:
        m_search = not query or query in p.definition.name.lower() or \
                   query in (p.definition.description or "").lower() or \
                   any(query in t.lower() for t in p.definition.tags) or \
                   any(query in (l.scene or "").lower() for l in p.definition.script if l.scene) or \
                   any(query in (l.scene_location or "").lower() for l in p.definition.script if l.scene_location)
                   
        m_lang = not sel_lang or p.definition.default_language in sel_lang
        m_status = not sel_status or p.status in sel_status
        
        if m_search and m_lang and m_status:
            filtered.append(p)

    filtered.sort(key=lambda x: x.definition.created_at, reverse=True)

    total = len(filtered)
    pages = (total - 1) // PROJECTS_PER_PAGE + 1 if total > 0 else 1
    if "dash_page" not in st.session_state:
        st.session_state.dash_page = 1
    
    start = (st.session_state.dash_page - 1) * PROJECTS_PER_PAGE
    page_projects = filtered[start:start + PROJECTS_PER_PAGE]

    st.write(f"Showing {len(page_projects)} of {total} projects")

    for project in page_projects:
        with st.container(border=True):
            total_l = len(project.definition.script)
            done_l = len([s for s in project.states if s.status == LineStatus.COMPLETED])
            progress = (done_l / total_l * 100) if total_l > 0 else 0
            
            color = {
                ProjectStatus.IDLE: "gray", ProjectStatus.COMPLETED: "green",
                ProjectStatus.GENERATING: "blue", ProjectStatus.FAILED: "red",
                ProjectStatus.PAUSED: "orange"
            }.get(project.status, "gray")

            h_cols = st.columns([5, 2, 2])
            with h_cols[0]: 
                st.markdown(f"##### {project.definition.name}")
            with h_cols[1]: 
                st.markdown(f"**Status:** :{color}[{project.status.value.upper()}]")
            with h_cols[2]: 
                st.markdown(f"**Progress:** {progress:.1f}%")

            with st.expander("Details & Management"):
                st.caption(f"Technical ID: {project.id}")
                
                unique_scenes = len(set(l.scene for l in project.definition.script))
                unique_locations = len(set(l.scene_location for l in project.definition.script))
                unique_voices = len(set(l.voice_id for l in project.definition.script))
                
                m_c1, m_c2, m_c3 = st.columns(3)
                with m_c1: 
                    tags_str = ', '.join(project.definition.tags) if project.definition.tags else "None"
                    st.write(f"**Tags:** {tags_str}")                    
                    st.write(f"**Voices:** {unique_voices}")
                with m_c2: 
                    st.write(f"**Base Language:** {project.definition.default_language.upper()}")
                    st.write(f"**Scenes:** {unique_scenes}")
                with m_c3:
                    st.write(f"**Lines:** {done_l} / {total_l}")
                    st.write(f"**Locations:** {unique_locations}")
                
                if project.definition.description:
                    st.markdown(f"> **Description:** {project.definition.description}")
                
                st.divider()
                
                b_cols = st.columns(6)
                is_active = project.status in [ProjectStatus.STARTING, ProjectStatus.GENERATING]
                can_modify = not is_active

                with b_cols[0]:
                    if st.button("👁️", key=f"mon_{project.id}", use_container_width=True,help="Monitor Project"):
                        navigate_to("monitor", project.id)
                
                
                with b_cols[1]:
                    script_json = project.definition.export_as_template()
                    st.download_button(
                        label="📄 JSON", 
                        data=script_json, 
                        file_name=f"{project.definition.name.replace(' ', '_')}.json", 
                        mime="application/json", 
                        use_container_width=True,
                        help="JSON Script",
                        key=f"json_{project.id}"
                    )
                
                with b_cols[2]:
                    master_path_str = getattr(project, "merged_audio_path", None)
                    show_audio = False
                    if master_path_str:
                        master_file = Path(project.project_path) / master_path_str
                        if master_file.exists():
                            show_audio = True
                    
                    if show_audio:
                        with open(master_file, "rb") as f:
                            audio_bytes = f.read()
                        st.download_button(
                            label="🔊 WAV",
                            data=audio_bytes,
                            file_name=f"master_{project.definition.name.replace(' ', '_')}.wav",
                            mime="audio/wav",
                            use_container_width=True,
                            help="WAV Master File",
                            key=f"wav_{project.id}"
                        )
                    else:
                        st.button("🔊 Audio", use_container_width=True, disabled=True, help="Full master audio not yet generated.",key=f"wav_{project.id}")
                
                with b_cols[3]:
                    mp3_path_str = getattr(project, "merged_mp3_path", None)
                    mp3_audio = False
                    if mp3_path_str:
                        mp3_file = Path(project.project_path) / mp3_path_str
                        if mp3_file.exists():
                            mp3_audio = True
                    
                    if mp3_audio:
                        with open(mp3_file, "rb") as f:
                            mp3_bytes = f.read()
                        st.download_button(
                            label="🔊 MP3",
                            data=mp3_bytes,
                            file_name=f"master_{project.definition.name.replace(' ', '_')}.mp3",
                            mime="audio/mp3g",
                            use_container_width=True,
                            help="MP3 Master File",
                            key=f"mp3_{project.id}"
                        )
                    else:
                        st.button("🔊 MP3", use_container_width=True, disabled=True, help="Full master audio not yet generated.",key=f"mp3_{project.id}")
                
                with b_cols[4]:
                    if st.button("✏️", key=f"ed_{project.id}", use_container_width=True,help="Edit Project", disabled=not can_modify):
                        navigate_to("editor", project.id)                  
                
                with b_cols[5]:
                    if st.button("🗑️", key=f"del_{project.id}", use_container_width=True,help="Delete Project", disabled=not can_modify):
                        delete_confirmation_dialog(project.id, project.definition.name)

    if pages > 1:
        st.divider()
        p_c1, p_c2, p_c3 = st.columns([1, 2, 1])
        with p_c1:
            if st.button("Previous", disabled=st.session_state.dash_page <= 1, use_container_width=True):
                st.session_state.dash_page -= 1
                st.rerun()
        with p_c2: 
            st.markdown(f"<div style='text-align: center'>Page {st.session_state.dash_page} of {pages}</div>", unsafe_allow_html=True)
        with p_c3:
            if st.button("Next", disabled=st.session_state.dash_page >= pages, use_container_width=True):
                st.session_state.dash_page += 1
                st.rerun()