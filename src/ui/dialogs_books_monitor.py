import streamlit as st
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, Optional

from src.config import settings
from src.models import AudioBookProject, AudioBookStatus, DialogProject
from src.audiobook.builder import AudiobookBuilder
from src.audiobook.schemas import AudiobookManifest, Chapter

@st.dialog("Confirm AudioBook Deletion")
def delete_book_from_monitor(book_id: str, book_title: str, navigate_to: Callable) -> None:
    """Modal to safely delete an audiobook project directly from the monitor."""
    st.warning(f"Confirm permanent deletion of: {book_title}")
    st.write("This will delete the compiled `.m4b` and the book definition.")
    st.write("⚠️ **Note:** Your original dialog scripts will NOT be deleted.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Yes, Delete", type="primary", width="stretch"):
            db = st.session_state.db
            books_dict = db.dict("audiobooks")
            if book_id in books_dict:
                del books_dict[book_id]
            st.success("AudioBook deleted.")
            time.sleep(0.5)
            navigate_to("books_dashboard")
    with col_b:
        if st.button("Cancel", width="stretch"):
            st.rerun()


def render_books_monitor(navigate_to: Callable[[str, Optional[str]], None]) -> None:
    """Renders the compilation and export room for the AudioBook."""
    
    book_id = st.session_state.get("selected_project_id")
    if not book_id:
        st.error("No AudioBook selected.")
        if st.button("Back to Dashboard"):
            navigate_to("books_dashboard")
        return

    db = st.session_state.db
    books_dict = db.dict("audiobooks")
    scripts_dict = db.dict("dialog_projects")
    book_data = books_dict.get(book_id)

    if not book_data:
        st.error("AudioBook not found.")
        return

    book = AudioBookProject(**book_data)

    col1, col2, col3, col4 = st.columns([5, 1, 1, 1])
    with col1:
        st.header(f"🎧 Monitor: {book.definition.title}")
    with col2:
        if st.button("⬅️", help="Go back", width="stretch"):
            navigate_to("books_dashboard")
    with col3:
        if st.button("✏️", help="Edit book", width="stretch"):
            navigate_to("books_editor", book.id)
    with col4:
        if st.button("🗑️",help="Delete book", width="stretch"):
            delete_book_from_monitor(book.id, book.definition.title, navigate_to)

    st.divider()
    
    with st.expander("**Global Metadata**",expanded=False):
        col_img, col_meta = st.columns([1, 3])
        
        with col_img:
            if book.definition.cover_path:
                img_path = Path(book.project_path) / book.definition.cover_path
                if img_path.exists():
                    st.image(str(img_path), width="stretch")
                else:
                    st.warning("Cover image file is missing from disk.")
            else:
                st.info("No cover art provided.")
                
        with col_meta:
            st.markdown(f"**Author / Compiler:** {book.definition.author}")
            
            if book.definition.copyright:
                st.markdown(f"**Copyright:** {book.definition.copyright}")
                
            if book.definition.description:
                st.markdown("**Synopsis:**")
                st.write(book.definition.description)

    with st.expander("**Chapters & Health Check**",expanded=False):
    
        can_compile = True
        manifest_chapters = []
        
        for i, chap in enumerate(book.definition.chapters):
            script_data = scripts_dict.get(chap.script_id)
            
            with st.container(border=True):
                c_info, c_status, c_action = st.columns([4, 2, 2], vertical_alignment="center")
                
                with c_info:
                    st.markdown(f"**{i+1}. {chap.title}**")
                    if script_data:
                        sc_name = script_data.get("definition", {}).get("name", "Unknown Script")
                        st.caption(f"🔗 Linked to: *{sc_name}*")
                    else:
                        st.caption("🔗 Linked to: ⚠️ DELETED SCRIPT")
                
                with c_status:
                    if not script_data:
                        st.error("🔴 Broken Link")
                        can_compile = False
                    else:
                        sc_status = script_data.get("status")
                        audio_rel_path = script_data.get("merged_audio_path") or script_data.get("master_path")
                        
                        if sc_status == "completed" and audio_rel_path:
                            st.success("🟢 Audio Ready")
                            
                            abs_audio_path = Path(script_data.get("project_path")) / audio_rel_path
                            manifest_chapters.append(
                                Chapter(
                                    title=chap.title,
                                    author=chap.author,
                                    audio_path=abs_audio_path
                                )
                            )
                        elif sc_status == "completed" and not audio_rel_path:
                            st.warning("🟡 Missing Master Audio")
                            can_compile = False
                        else:
                            st.info(f"⏳ Status: {sc_status.capitalize()}")
                            can_compile = False
                            
                with c_action:
                    if script_data:
                        if st.button("➡️ Go to Script", key=f"go_sc_{i}", width="stretch"):
                            navigate_to("monitor", chap.script_id)
                    else:
                        if st.button("✏️", help="Edit book", key=f"fix_{i}", width="stretch"):
                            navigate_to("books_editor", book.id)

        if book.status == AudioBookStatus.OUTDATED:
            st.warning("🟠 **Update Recommended:** Some scripts have been modified since the last compilation.")

    st.markdown("**Assembly Engine**")
    
    if st.button("🎬 Compile AudioBook (.m4b)", type="primary", width="stretch", disabled=not can_compile):
        
        cover_abs_path = Path(book.project_path) / book.definition.cover_path if book.definition.cover_path else None
        
        manifest = AudiobookManifest(
            title=book.definition.title,
            author=book.definition.author,
            description=book.definition.description,
            copyright=book.definition.copyright,
            cover_path=cover_abs_path,
            chapters=manifest_chapters
        )
        
        with st.spinner("Assembling M4B container and writing metadata..."):
            result = AudiobookBuilder.compile(
                manifest=manifest, 
                output_dir=Path(book.project_path)
            )
            
        if result.success:
            book.output_m4b_path = result.output_path.name
            book.last_compiled_at = datetime.now(timezone.utc)
            book.status = AudioBookStatus.READY
            
            books_dict[book.id] = book.model_dump(mode="json")
            
            st.success(f"✅ Audiobook compiled successfully in {result.processing_time_sec:.2f} seconds!")
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"❌ Compilation failed: {result.message}")

    st.divider()

    if book.output_m4b_path:
        st.markdown("**Final Output**")
        
        m4b_full_path = Path(book.project_path) / book.output_m4b_path
        
        if m4b_full_path.exists():
            with st.container(border=True):
                col_img, col_audio = st.columns([1, 4], vertical_alignment="center")
                
                with col_img:
                    if book.definition.cover_path:
                        img_path = Path(book.project_path) / book.definition.cover_path
                        if img_path.exists():
                            st.image(str(img_path), width="stretch")
                        else:
                            st.markdown("📄")
                    else:
                        st.markdown("📄")
                
                with col_audio:
                    with open(m4b_full_path, "rb") as f:
                        m4b_bytes = f.read()
                        
                    st.audio(m4b_bytes, format="audio/mp4")
                    
                    st.download_button(
                        label="⬇️ Download .m4b AudioBook", 
                        data=m4b_bytes, 
                        file_name=book.output_m4b_path, 
                        mime="audio/mp4", 
                        key=f"dl_m4b_{book.id}", 
                        width="stretch"
                    )
                    
                    st.caption(f"Last compiled: {book.last_compiled_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        else:
            st.warning("The compiled file is missing from the disk. Please re-compile.")
            book.output_m4b_path = None
            book.status = AudioBookStatus.INCOMPLETE
            books_dict[book.id] = book.model_dump(mode="json")