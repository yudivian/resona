import streamlit as st
import time
from typing import Callable, Optional, List
from src.models import DialogProject, ProjectStatus, LineStatus

@st.dialog("Confirm Project Deletion")
def delete_confirmation_dialog(project_id: str, project_name: str) -> None:
    """
    Displays a safety modal to confirm the permanent removal of a project and its records.

    Args:
        project_id: The unique identifier of the project to delete.
        project_name: The display name of the project.
    """
    st.warning(f"Confirm permanent deletion of: {project_name}")
    st.write("This action will purge all project records from the database. This cannot be undone.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Yes, Delete", type="primary", use_container_width=True):
            del st.session_state.db.dict("dialog_projects")[project_id]
            st.success("Project purged.")
            time.sleep(0.5)
            st.rerun()
    with col_b:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

def render_dashboard(navigate_to: Callable[[str, Optional[str]], None]) -> None:
    """
    Renders the centralized project management dashboard. 
    Includes fuzzy search, advanced filtering, and state-aware project management.

    Args:
        navigate_to: Callback for application-level navigation.
    """
    from src.app_dialogs import PROJECTS_PER_PAGE
    
    db = st.session_state.db
    projects_dict = db.dict("dialog_projects")
    
    if not projects_dict:
        st.info("No projects found in the workspace. Use the sidebar to create or import one.")
        return

    all_projects = [DialogProject(**p) for p in projects_dict.values()]

    with st.container(border=True):
        st.subheader("Filters & Search")
        f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
        
        with f_col1:
            query = st.text_input("Fuzzy Search", placeholder="Name, tags, or description...").lower()
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
                   any(query in t.lower() for t in p.definition.tags)
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

            h_cols = st.columns([4, 2, 2])
            with h_cols[0]: 
                st.markdown(f"### {project.definition.name}")
            with h_cols[1]: 
                st.markdown(f"**Status:** :{color}[{project.status.upper()}]")
            with h_cols[2]: 
                st.markdown(f"**Progress:** {progress:.1f}%")

            with st.expander("Details & Management"):
                st.caption(f"Technical ID: {project.id}")
                
                m_c1, m_c2 = st.columns(2)
                with m_c1: 
                    st.write(f"**Base Language:** {project.definition.default_language.upper()}")
                    if project.definition.tags:
                        st.write(f"**Tags:** {', '.join(project.definition.tags)}")
                with m_c2: 
                    st.write(f"**Lines:** {done_l} / {total_l} completed lines")
                if project.definition.description:
                    st.write(f"**Description:** {project.definition.description}")
                
                st.divider()
                
                b_cols = st.columns(3)
                is_active = project.status in [ProjectStatus.STARTING, ProjectStatus.GENERATING]
                can_modify = not is_active

                with b_cols[0]:
                    if st.button("👁️ Monitor", key=f"mon_{project.id}", use_container_width=True):
                        navigate_to("monitor", project.id)
                with b_cols[1]:
                    if st.button("✏️ Edit Script", key=f"ed_{project.id}", use_container_width=True, disabled=not can_modify):
                        navigate_to("editor", project.id)
                with b_cols[2]:
                    if st.button("🗑️ Delete", key=f"del_{project.id}", use_container_width=True, disabled=not can_modify):
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