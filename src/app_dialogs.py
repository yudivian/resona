import streamlit as st
import json
import time
import uuid
from typing import Callable, Optional

from src.config import settings
from beaver import BeaverDB
from pydantic import ValidationError
from src.dialogs.orchestrator import DialogOrchestrator
from src.models import DialogProject, DialogScript, ProjectStatus, ProjectSource, LineState, LineStatus

from src.ui.dialogs_dashboard import render_dashboard
from src.ui.dialogs_create import render_create
from src.ui.dialogs_editor import render_editor
from src.ui.dialogs_monitor import render_monitor

from src.ui.dialogs_books_dashboard import render_books_dashboard
from src.ui.dialogs_books_create import render_books_create
from src.ui.dialogs_books_editor import render_books_editor
from src.ui.dialogs_books_monitor import render_books_monitor


MAX_DIALOG_LINES = 50
POLLING_INTERVAL_SEC = 10.0
PROJECTS_PER_PAGE = 3
LINES_PER_PAGE = 10

@st.dialog("Import Project from JSON")
def import_dialog() -> None:
    """
    Displays a modal for importing an external JSON script definition.
    
    It validates the schema using the new import_template factory, 
    initializes technical states for each line, and saves the new 
    DialogProject instance to the database.
    """
    st.write("Upload a JSON file to initialize a new project workspace.")
    imp_file = st.file_uploader("JSON Script File", type=['json'], key="global_import_uploader")
    
    if imp_file:
        try:
            json_string = imp_file.getvalue().decode("utf-8")
            
            script = DialogScript.import_template(json_string)
            
            db = st.session_state.db
            projects_dict = db.dict("dialog_projects")
            
            from pathlib import Path
            workspace_dir = str(Path(settings.paths.dialogspace_dir) / script.id)
            
            project = DialogProject(
                source=ProjectSource.UI,
                definition=script,
                states=[LineState(line_id=line.id, index=line.index, status=LineStatus.PENDING) for line in script.script],
                project_path=workspace_dir,
                status=ProjectStatus.IDLE
            )
            
            projects_dict[project.id] = project.model_dump()
            st.success(f"Project '{script.name}' imported successfully!")
            time.sleep(1)
            navigate_to("monitor", project.id)
            
        except ValueError as e:
            st.error(f"Validation Error: {e}")
            
        except ValidationError as e:
            st.error("Schema Error: The JSON structure is invalid or missing required fields.")
            with st.expander("View schema details"):
                st.text(str(e))
                
        except Exception as e:
            st.error(f"Unexpected error: {e}")

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
        
        st.subheader("🎬 Scripts & Scenes")
        if st.button("🏠 Scripts Dashboard", width="stretch"):
            navigate_to("dashboard")
        if st.button("➕ New Script", width="stretch", type="primary"):
            navigate_to("create")
        if st.button("📥 Import Script", width="stretch"):
            import_dialog()
            
        st.divider()
        
        st.subheader("📚 AudioBooks")
        if st.button("📚 Bookshelf", width="stretch"):
            navigate_to("books_dashboard")
        if st.button("➕ New AudioBook", width="stretch", type="primary"):
            navigate_to("books_create")

    view = st.session_state.current_view
    if view == "dashboard":
        render_dashboard(navigate_to)
    elif view == "create":
        render_create(navigate_to)
    elif view == "editor":
        render_editor(navigate_to)
    elif view == "monitor":
        render_monitor(navigate_to)
    elif view == "books_dashboard":
        render_books_dashboard(navigate_to)
    elif view == "books_create":
        render_books_create(navigate_to)
    elif view == "books_editor":
        render_books_editor(navigate_to)
    elif view == "books_monitor":
        render_books_monitor(navigate_to)
    else:
        navigate_to("dashboard")

if __name__ == "__main__":
    main()