import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings
from src.ui.session import SessionManager
from src.ui.components import init_page, render_debug_panel, render_studio

init_page()

@st.cache_resource
def get_session():
    """
    Initializes and caches the SessionManager instance.
    """
    return SessionManager(settings)

try:
    session = get_session()
except Exception as e:
    st.error(f"Critical Error: Failed to initialize backend. {e}")
    st.stop()

st.title("🎛️ Resona Voice CAD")

render_debug_panel(session)

tab_studio, tab_library, tab_search = st.tabs(["🎙️ Studio", "📚 Library", "🔍 Search"])

with tab_studio: 
    render_studio(session)

with tab_library:
    st.header("Voice Library")
    st.info("🚧 Library browser coming in Phase 5...")

with tab_search:
    st.header("Semantic Search")
    st.info("🚧 Semantic search coming in Phase 5...")

st.markdown("---")
st.caption("Resona CAD System - Modular Voice Design Interface")