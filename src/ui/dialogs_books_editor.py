import streamlit as st
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from src.models import (
    AudioBookProject, 
    AudioBookDefinition, 
    AudioBookChapter, 
    AudioBookStatus,
    DialogProject
)

def _apply_changes(book_id: str, state_data: dict, available_scripts: dict) -> bool:
    """
    Validates and saves the edited audiobook back to the database.
    Dynamically recalculates the status upon saving.
    """
    db = st.session_state.db
    books_dict = db.dict("audiobooks")
    book_data = books_dict.get(book_id)
    
    if not book_data:
        st.error("AudioBook not found in database.")
        return False
        
    try:
        book = AudioBookProject(**book_data)
        
        new_chapters = []
        for c_data in state_data["chapters"]:
            new_chapters.append(AudioBookChapter(
                script_id=c_data["script_id"],
                title=c_data["title"].strip(),
                author=c_data["author"].strip() if c_data["author"].strip() else None
            ))
            
        book.definition.title = state_data["title"].strip()
        book.definition.author = state_data["author"].strip()
        book.definition.description = state_data["description"].strip()
        book.definition.copyright = state_data["copyright"].strip()
        book.definition.chapters = new_chapters
        
        new_status = AudioBookStatus.READY
        for chap in new_chapters:
            script = available_scripts.get(chap.script_id)
            if not script:
                new_status = AudioBookStatus.INVALID
                break
            elif script.status != "completed" or not script.merged_audio_path:
                new_status = AudioBookStatus.INCOMPLETE
                
        book.status = new_status
        book.touch() 
        
        books_dict[book_id] = book.model_dump(mode="json")
        return True
        
    except Exception as e:
        st.error(f"Error saving book: {str(e)}")
        return False


def render_books_editor(navigate_to: Callable[[str, Optional[str]], None]) -> None:
    """Renders the AudioBook editor and repair clinic."""
    
    book_id = st.session_state.selected_project_id
    if not book_id:
        st.error("No AudioBook selected.")
        if st.button("Back to Dashboard"):
            navigate_to("books_dashboard")
        return

    db = st.session_state.db
    books_dict = db.dict("audiobooks")
    book_data = books_dict.get(book_id)

    if not book_data:
        st.error("AudioBook not found.")
        return

    book = AudioBookProject(**book_data)

    scripts_raw = db.dict("dialog_projects")
    available_scripts = {}
    for pid, pdata in scripts_raw.items():
        try:
            available_scripts[pid] = DialogProject(**pdata)
        except Exception:
            continue

    if "book_edit_state" not in st.session_state or st.session_state.get("book_edit_id") != book_id:
        st.session_state.book_edit_id = book_id
        st.session_state.book_edit_state = {
            "title": book.definition.title,
            "author": book.definition.author,
            "description": book.definition.description or "",
            "copyright": book.definition.copyright or "",
            "chapters": [
                {
                    "_ui_id": str(uuid.uuid4()),
                    "script_id": c.script_id, 
                    "title": c.title, 
                    "author": c.author or ""
                } for c in book.definition.chapters
            ]
        }

    state = st.session_state.book_edit_state

    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.header(f"✏️ Editing: {book.definition.title}")
    with col_h2:
        if st.button("Cancel & Return", width="stretch"):
            del st.session_state.book_edit_state
            navigate_to("books_dashboard")
            return

    mastering_configs = set()
    for chap in state["chapters"]:
        sc = available_scripts.get(chap["script_id"])
        if sc:
            target_lufs = sc.definition.mastering.target_lufs
            comp_ratio = sc.definition.mastering.compressor_ratio
            
            cfg_str = f"LUFS:{target_lufs}_Comp:{comp_ratio}"
            mastering_configs.add(cfg_str)
            
    if len(mastering_configs) > 1:
        st.warning("🎛️ **Mastering Discrepancy Detected:** The scripts in this book have different mastering configurations (LUFS/Compression). The final audiobook might sound inconsistent in volume.")

    with st.container(border=True):
        st.markdown("**Global Metadata**")
        c1, c2 = st.columns(2)
        with c1:
            state["title"] = st.text_input("AudioBook Title", value=state["title"])
            state["author"] = st.text_input("Author / Compiler", value=state["author"])
            state["copyright"] = st.text_input("Copyright", value=state["copyright"])
        with c2:
            state["description"] = st.text_area("Synopsis (Description)", value=state["description"], height=120)

    st.markdown("**Chapters**")
    
    if not state["chapters"]:
        st.info("No chapters. Add one below.")

    for i, chap in enumerate(state["chapters"]):
        
        if "_ui_id" not in chap:
            chap["_ui_id"] = str(uuid.uuid4())
        ui_key = chap["_ui_id"]
        
        is_ghost = chap["script_id"] not in available_scripts
        is_incomplete = False
        if not is_ghost:
            sc = available_scripts[chap["script_id"]]
            is_incomplete = (sc.status != "completed" or not sc.merged_audio_path)

        border_color = "🔴 " if is_ghost else "🟡 " if is_incomplete else ""
        
        with st.container(border=True):
            h_cols = st.columns([7, 1, 1, 1])
            h_cols[0].markdown(f"**{border_color}Chapter {i+1}**")
            
            if h_cols[1].button("⬆️", key=f"up_{ui_key}", disabled=(i == 0)):
                state["chapters"][i], state["chapters"][i-1] = state["chapters"][i-1], state["chapters"][i]
                st.rerun()
            if h_cols[2].button("⬇️", key=f"down_{ui_key}", disabled=(i == len(state["chapters"])-1)):
                state["chapters"][i], state["chapters"][i+1] = state["chapters"][i+1], state["chapters"][i]
                st.rerun()
            if h_cols[3].button("🗑️", key=f"del_{ui_key}"):
                state["chapters"].pop(i)
                st.rerun()

            if is_ghost:
                st.error("⚠️ **Broken Link:** The original script was deleted. You must replace it or delete this chapter.")
            elif is_incomplete:
                st.warning("⏳ **Incomplete Asset:** This script has not been fully generated or masterized yet.")

            options_dict = {pid: f"{p.definition.name} ({p.status})" for pid, p in available_scripts.items()}
            if is_ghost:
                options_dict[chap["script_id"]] = "⚠️ SCRIPT MISSING"

            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                current_idx = list(options_dict.keys()).index(chap["script_id"]) if chap["script_id"] in options_dict else 0
                selected_id = st.selectbox(
                    "Linked Script",
                    options=list(options_dict.keys()),
                    format_func=lambda x: options_dict[x],
                    index=current_idx,
                    key=f"sc_{ui_key}"
                )
                
                if selected_id != chap["script_id"]:
                    chap["script_id"] = selected_id
                    new_title = available_scripts[selected_id].definition.name if selected_id in available_scripts else ""
                    chap["title"] = new_title
                    st.session_state[f"tit_{ui_key}"] = new_title
                    st.rerun()

            with c2:
                chap["title"] = st.text_input("Chapter Title", value=chap["title"], key=f"tit_{ui_key}")
            with c3:
                chap["author"] = st.text_input("Specific Author", value=chap["author"], key=f"aut_{ui_key}")

    if st.button("➕ Add Chapter", width="stretch"):
        if available_scripts:
            first_script_id = list(available_scripts.keys())[0]
            state["chapters"].append({
                "_ui_id": str(uuid.uuid4()),
                "script_id": first_script_id,
                "title": available_scripts[first_script_id].definition.name,
                "author": ""
            })
            st.rerun()
        else:
            st.error("No scripts available to add.")

    st.divider()

    valid = True
    val_msg = ""
    if not state["title"].strip():
        valid, val_msg = False, "Title is required."
    elif not state["chapters"]:
        valid, val_msg = False, "At least one chapter is required."
    elif any(not c["title"].strip() for c in state["chapters"]):
        valid, val_msg = False, "All chapters must have a title."
    elif any(c["script_id"] not in available_scripts for c in state["chapters"]):
        valid, val_msg = False, "Cannot save with broken links (Ghost scripts). Remove or replace them."

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        if st.button("💾 Save Changes", width="stretch"):
            if valid:
                if _apply_changes(book_id, state, available_scripts):
                    st.success("AudioBook updated successfully.")
                    time.sleep(0.5)
                    st.rerun()
            else:
                st.error(val_msg)
                
    with f_col2:
        if st.button("🚀 Save & Go to Monitor", type="primary", width="stretch"):
            if valid:
                if _apply_changes(book_id, state, available_scripts):
                    del st.session_state.book_edit_state
                    navigate_to("books_monitor", book_id)
            else:
                st.error(val_msg)