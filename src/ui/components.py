import streamlit as st
from src.ui.session import SessionManager

def init_page():
    """
    Configures the global Streamlit page settings and injects custom CSS for styling tabs and expanders.
    """
    st.set_page_config(
        page_title="Resona CAD",
        page_icon="🎛️",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        div[data-testid="stExpander"] div[role="button"] p {
            font-size: 1.1rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

def render_debug_panel(session: SessionManager):
    """
    Renders a sidebar panel displaying the current technical status of the engines and the active session state.
    """
    st.sidebar.header("🛠️ System Status")
    
    st.sidebar.markdown("### Engines")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Track A", "Ready" if session.engine_a else "Error")
    with col2:
        st.metric("Track B", "Ready" if session.engine_b else "Error")
    
    st.sidebar.markdown("### Session State")
    if session.active_mix is not None:
        st.sidebar.success(f"Mix Active (Seed: {session.active_seed})")
    else:
        st.sidebar.info("No Mix Active")
        
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Config: {session.config.system.name} v{session.config.system.version}")