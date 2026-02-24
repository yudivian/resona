import streamlit as st
import json
import time
import uuid
from typing import Callable, Optional

from src.config import settings
from beaver import BeaverDB
from src.dialogs.orchestrator import DialogOrchestrator
from src.models import DialogProject, DialogScript, ProjectStatus, ProjectSource, LineState, LineStatus

from src.ui.dialogs_dashboard import render_dashboard
from src.ui.dialogs_create import render_create
from src.ui.dialogs_editor import render_editor
from src.ui.dialogs_monitor import render_monitor

MAX_DIALOG_LINES = 5
POLLING_INTERVAL_SEC = 10.0
PROJECTS_PER_PAGE = 8
LINES_PER_PAGE = 10

@st.dialog("Import Project from JSON")
def import_dialog() -> None:
    """
    Displays a modal for importing an external JSON script definition.
    
    It validates the schema, initializes technical states for each line, 
    and saves the new DialogProject instance to the database.
    """
    st.write("Upload a JSON file to initialize a new project workspace.")
    imp_file = st.file_uploader("JSON Script File", type=['json'], key="global_import_uploader")
    
    if imp_file:
        try:
            data = json.load(imp_file)
            script_data = data["definition"] if "definition" in data else data
            script = DialogScript(**script_data)
            
            db = st.session_state.db
            projects_dict = db.dict("dialog_projects")
            
            project = DialogProject(
                source=ProjectSource.UI,
                definition=script,
                states=[
                    LineState(line_id=l.id, index=i, status=LineStatus.PENDING) 
                    for i, l in enumerate(script.script)
                ],
                status=ProjectStatus.IDLE,
                project_path=f"{settings.paths.dialogspace_dir}/{script.id}"
            )
            
            projects_dict[project.id] = project.model_dump()
            st.success("Project imported and initialized.")
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.error(f"Import failed: {e}")

def init_session_state() -> None:
    """
    Bootstraps the Streamlit session state with required engine components,
    database connections, and navigation variables.
    """
    if "current_view" not in st.session_state:
        st.session_state.current_view = "dashboard"
    if "selected_project_id" not in st.session_state:
        st.session_state.selected_project_id = None
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = DialogOrchestrator()
    if "db" not in st.session_state:
        st.session_state.db = BeaverDB(settings.paths.db_file)
    if "settings" not in st.session_state:
        st.session_state.settings = settings

def navigate_to(view_name: str, project_id: Optional[str] = None) -> None:
    """
    Triggers a state-based view transition across the application.
    """
    st.session_state.current_view = view_name
    st.session_state.selected_project_id = project_id
    st.rerun()

def main() -> None:
    """
    Main application controller. Configures the centered page layout, 
    manages the global sidebar navigation, and routes to the active view.
    """
    st.set_page_config(
        page_title="Resona DialogSpace", 
        page_icon="🎙️", 
        layout="centered", 
        initial_sidebar_state="expanded"
    )
    init_session_state()

    with st.sidebar:
        st.title("🎙️ DialogSpace")
        st.divider()
        
        st.subheader("Navigation")
        if st.button("🏠 Dashboard", use_container_width=True):
            navigate_to("dashboard")
            
        st.divider()
        st.subheader("Global Actions")
        if st.button("➕ New Project", use_container_width=True, type="primary"):
            navigate_to("create")
            
        if st.button("📥 Import Project", use_container_width=True):
            import_dialog()

    view = st.session_state.current_view
    if view == "dashboard":
        render_dashboard(navigate_to)
    elif view == "create":
        render_create(navigate_to)
    elif view == "editor":
        render_editor(navigate_to)
    elif view == "monitor":
        render_monitor(navigate_to)
    else:
        navigate_to("dashboard")

if __name__ == "__main__":
    main()