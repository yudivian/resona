import streamlit as st
import uuid
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, Optional

from src.config import settings
from src.models import (
    AudioBookProject, 
    AudioBookDefinition, 
    AudioBookChapter, 
    AudioBookStatus,
    DialogProject
)

def _persist_audiobook(
    title: str, 
    author: str, 
    description: str, 
    copyright_txt: str, 
    cover_file, 
    chapters_data: list
) -> str:
    """
    Constructs and persists a new AudioBookProject to the database.
    """
    db = st.session_state.db
    
    books_dict = db.dict("audiobooks")

    book_id = str(uuid.uuid4())
    
    project_path = Path(settings.paths.bookspace_dir) / book_id
    project_path.mkdir(parents=True, exist_ok=True)

    cover_path_str = None
    if cover_file:
        cover_path = project_path / cover_file.name
        with open(cover_path, "wb") as f:
            f.write(cover_file.getvalue())
        cover_path_str = cover_file.name

    chapters = []
    for data in chapters_data:
        chapters.append(AudioBookChapter(
            script_id=data["script_id"],
            title=data["title"].strip(),
            author=data["author"].strip() if data["author"].strip() else None
        ))

    definition = AudioBookDefinition(
        title=title,
        author=author,
        description=description,
        copyright=copyright_txt,
        cover_path=cover_path_str,
        chapters=chapters
    )

    book = AudioBookProject(
        id=book_id,
        project_path=str(project_path),
        definition=definition,
        status=AudioBookStatus.INCOMPLETE 
    )

    books_dict[book_id] = book.model_dump(mode="json")
    return book_id


def render_books_create(navigate_to: Callable[[str, Optional[str]], None]) -> None:
    """
    Renders the AudioBook creation view.
    """
    st.header("➕ Create New AudioBook")
    st.markdown("Assemble your generated scripts into a cohesive `.m4b` audiobook container.")

    db = st.session_state.db
    all_scripts_raw = db.dict("dialog_projects")
    
    available_scripts = {}
    for pid, pdata in all_scripts_raw.items():
        try:
            available_scripts[pid] = DialogProject(**pdata)
        except Exception:
            continue

    if not available_scripts:
        st.warning("You don't have any dialog scripts yet. Create some scripts before building an AudioBook.")
        if st.button("Go to Scripts", type="primary"):
            navigate_to("dashboard")
        return

    if "temp_book_chapters" not in st.session_state:
        st.session_state.temp_book_chapters = []

    with st.container(border=True):
        st.markdown("**Global Metadata**")
        col1, col2 = st.columns(2)
        with col1:
            b_title = st.text_input("AudioBook Title", placeholder="e.g. The Martian Chronicles")
            b_author = st.text_input("Author / Compiler", value="Various Authors")
            b_copy = st.text_input("Copyright", placeholder="e.g. © 2026 Resona")
        with col2:
            b_desc = st.text_area("Synopsis (Description)", height=120)
            b_cover = st.file_uploader("Cover Art (Optional)", type=["jpg", "jpeg", "png"])

    mastering_configs = set()
    for chap in st.session_state.temp_book_chapters:
        sc = available_scripts.get(chap["script_id"])
        if sc:
            target_lufs = sc.definition.mastering.target_lufs
            comp_ratio = sc.definition.mastering.compressor_ratio
            mastering_configs.add(f"LUFS:{target_lufs}_Comp:{comp_ratio}")
            
    if len(mastering_configs) > 1:
        st.warning("🎛️ **Mastering Discrepancy Detected:** The selected scripts have different mastering configurations (LUFS/Compression). The final audiobook might sound inconsistent in volume.")

    st.markdown("**Chapters List**")
    
    if not st.session_state.temp_book_chapters:
        st.info("No chapters added yet. Click below to add your first chapter.")

    options_dict = {pid: f"{p.definition.name} ({p.status})" for pid, p in available_scripts.items()}

    for i, chap in enumerate(st.session_state.temp_book_chapters):
        if "_ui_id" not in chap:
            chap["_ui_id"] = str(uuid.uuid4())
        ui_key = chap["_ui_id"]

        tit_key = f"tit_{ui_key}"
        aut_key = f"aut_{ui_key}"
        
        if tit_key not in st.session_state:
            st.session_state[tit_key] = chap.get("title", "")
        if aut_key not in st.session_state:
            st.session_state[aut_key] = chap.get("author", "")

        is_incomplete = False
        sc = available_scripts.get(chap["script_id"])
        if sc:
            if sc.status != "completed" or not sc.merged_audio_path:
                is_incomplete = True

        border_color = "🟡 " if is_incomplete else ""

        with st.container(border=True):
            header_cols = st.columns([8, 1, 1, 1])
            header_cols[0].markdown(f"**{border_color}Chapter {i+1}**")
            
            if header_cols[1].button("⬆️", key=f"up_{ui_key}", disabled=(i == 0)):
                st.session_state.temp_book_chapters[i], st.session_state.temp_book_chapters[i-1] = st.session_state.temp_book_chapters[i-1], st.session_state.temp_book_chapters[i]
                st.rerun()
            if header_cols[2].button("⬇️", key=f"down_{ui_key}", disabled=(i == len(st.session_state.temp_book_chapters)-1)):
                st.session_state.temp_book_chapters[i], st.session_state.temp_book_chapters[i+1] = st.session_state.temp_book_chapters[i+1], st.session_state.temp_book_chapters[i]
                st.rerun()
            if header_cols[3].button("🗑️", key=f"del_{ui_key}"):
                st.session_state.temp_book_chapters.pop(i)
                st.rerun()

            if is_incomplete:
                st.warning("⏳ **Incomplete Asset:** This script has not been fully generated or masterized yet. You can link it, but compilation will be blocked later.")

            c1, c2, c3 = st.columns([2, 2, 1])
            
            with c1:
                current_idx = list(options_dict.keys()).index(chap["script_id"]) if chap["script_id"] in options_dict else 0
                selected_id = st.selectbox(
                    "Link to Script", 
                    options=list(options_dict.keys()), 
                    format_func=lambda x: options_dict[x],
                    key=f"sc_{ui_key}",
                    index=current_idx
                )
                
                if selected_id != chap["script_id"]:
                    chap["script_id"] = selected_id
                    new_title = available_scripts[selected_id].definition.name if selected_id in available_scripts else ""
                    st.session_state[tit_key] = new_title
                    st.rerun()
                
            with c2:
                st.text_input("Chapter Title", key=tit_key)
                chap["title"] = st.session_state[tit_key]  
                
            with c3:
                st.text_input("Specific Author", placeholder="Optional", key=aut_key)
                chap["author"] = st.session_state[aut_key]

    if st.button("➕ Add Chapter", width="stretch"):
        first_script_id = list(available_scripts.keys())[0]
        st.session_state.temp_book_chapters.append({
            "_ui_id": str(uuid.uuid4()),
            "script_id": first_script_id,
            "title": available_scripts[first_script_id].definition.name,
            "author": ""
        })
        st.rerun()

    st.divider()

    if st.button("🚀 Create AudioBook & Go to Monitor", type="primary", width="stretch"):
        if not b_title.strip():
            st.error("AudioBook title is required.")
        elif not st.session_state.temp_book_chapters:
            st.error("At least one chapter is required.")
        elif any(not c["title"].strip() for c in st.session_state.temp_book_chapters):
            st.error("All chapters must have a title.")
        else:
            book_id = _persist_audiobook(
                title=b_title, author=b_author, description=b_desc,
                copyright_txt=b_copy, cover_file=b_cover, chapters_data=st.session_state.temp_book_chapters
            )
            st.success("AudioBook created successfully!")
            
            del st.session_state.temp_book_chapters
            time.sleep(0.5)
            
            navigate_to("books_monitor", book_id)