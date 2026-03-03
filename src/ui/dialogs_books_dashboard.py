import streamlit as st
import time
from datetime import datetime
from typing import Callable, Optional

from src.models import AudioBookProject, AudioBookStatus, DialogProject

def _refresh_book_status(book: AudioBookProject, scripts_db: dict) -> AudioBookStatus:
    """
    Dynamically evaluates the health of the soft links.
    Returns the true current status of the AudioBook.
    """
    if not book.definition.chapters:
        return AudioBookStatus.INCOMPLETE

    is_outdated = False

    for chap in book.definition.chapters:
        script_data = scripts_db.get(chap.script_id)
        
        if not script_data:
            return AudioBookStatus.INVALID
            
        if script_data.get("status") != "completed" or not script_data.get("merged_audio_path"):
            return AudioBookStatus.INCOMPLETE
            
        if book.last_compiled_at:
            script_updated_raw = script_data.get("updated_at")
            if script_updated_raw:
                try:
                    from datetime import timezone 
                    
                    if isinstance(script_updated_raw, datetime):
                        script_updated = script_updated_raw
                    elif isinstance(script_updated_raw, (int, float)):
                        script_updated = datetime.fromtimestamp(script_updated_raw, tz=timezone.utc)
                    else:
                        clean_str = str(script_updated_raw).replace("Z", "+00:00")
                        script_updated = datetime.fromisoformat(clean_str)
                    
                    if script_updated.tzinfo is None:
                        script_updated = script_updated.replace(tzinfo=timezone.utc)
                    
                    book_compiled_at = book.last_compiled_at
                    if book_compiled_at.tzinfo is None:
                        book_compiled_at = book_compiled_at.replace(tzinfo=timezone.utc)

                    if script_updated > book_compiled_at:
                        is_outdated = True
                except Exception:
                    pass

    if is_outdated:
        return AudioBookStatus.OUTDATED
    
    if not book.output_m4b_path:
        return AudioBookStatus.READY

    return AudioBookStatus.READY


@st.dialog("Confirm AudioBook Deletion")
def delete_book_dialog(book_id: str, book_title: str) -> None:
    """Modal to safely delete an audiobook project."""
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
            st.rerun()
    with col_b:
        if st.button("Cancel", width="stretch"):
            st.rerun()


def render_books_dashboard(navigate_to: Callable[[str, Optional[str]], None]) -> None:
    """Renders the library of AudioBooks with dynamic health checks and pagination."""
    from src.app_dialogs import PROJECTS_PER_PAGE
    
    st.header("📚 AudioBooks Library")
    
    db = st.session_state.db
    books_dict = db.dict("audiobooks")
    scripts_dict = db.dict("dialog_projects")
    
    if not books_dict:
        st.info("You haven't created any AudioBooks yet. Assemble your scripts into a book to get started!")
        if st.button("➕ Create AudioBook", type="primary"):
            navigate_to("books_create")
        return

    search_query = st.text_input("🔍 Search AudioBooks by title or author...", "").lower()
    st.divider()

    filtered_books = []
    for book_id, book_data in books_dict.items():
        try:
            book = AudioBookProject(**book_data)
        except Exception:
            continue
            
        if search_query and search_query not in book.definition.title.lower() and search_query not in book.definition.author.lower():
            continue
            
        filtered_books.append(book)

    filtered_books.sort(key=lambda x: x.created_at, reverse=True)

    total_books = len(filtered_books)
    pages = (total_books - 1) // PROJECTS_PER_PAGE + 1 if total_books > 0 else 1
    
    if "books_dash_page" not in st.session_state:
        st.session_state.books_dash_page = 1
        
    if st.session_state.books_dash_page > pages:
        st.session_state.books_dash_page = max(1, pages)
        
    start_idx = (st.session_state.books_dash_page - 1) * PROJECTS_PER_PAGE
    page_books = filtered_books[start_idx : start_idx + PROJECTS_PER_PAGE]

    st.write(f"Showing {len(page_books)} of {total_books} AudioBooks")

    for book in page_books:
        current_status = _refresh_book_status(book, scripts_dict)
        if current_status != book.status:
            book.status = current_status
            books_dict[book.id] = book.model_dump(mode="json")

        with st.container(border=True):
            cols = st.columns([4, 2, 3])
            
            with cols[0]:
                st.markdown(f"### {book.definition.title}")
                st.markdown(f"**Author:** {book.definition.author} | **Chapters:** {len(book.definition.chapters)}")
                
            with cols[1]:
                if current_status == AudioBookStatus.READY:
                    st.success("🟢 Ready")
                elif current_status == AudioBookStatus.OUTDATED:
                    st.warning("🟠 Outdated (Needs Rebuild)")
                elif current_status == AudioBookStatus.INCOMPLETE:
                    st.info("🟡 Missing Audios")
                elif current_status == AudioBookStatus.INVALID:
                    st.error("🔴 Broken Links (Script Missing)")

            with cols[2]:
                b_cols = st.columns(3)
                
                with b_cols[0]:
                    can_compile = current_status in [AudioBookStatus.READY, AudioBookStatus.OUTDATED]
                    if st.button("🎬",help="View book", key=f"mon_{book.id}", width="stretch", disabled=not can_compile):
                        navigate_to("books_monitor", book.id)
                        
                with b_cols[1]:
                    if st.button("✏️",help="Edit book", key=f"ed_{book.id}", width="stretch"):
                        navigate_to("books_editor", book.id)
                        
                with b_cols[2]:
                    if st.button("🗑️",help="Delete book", key=f"del_{book.id}", width="stretch"):
                        delete_book_dialog(book.id, book.definition.title)

    if pages > 1:
        st.divider()
        p_c1, p_c2, p_c3 = st.columns([1, 2, 1])
        with p_c1:
            if st.button("< Previous", key="prev_books_page", disabled=st.session_state.books_dash_page <= 1, width="stretch"):
                st.session_state.books_dash_page -= 1
                st.rerun()
        with p_c2: 
            st.markdown(f"<div style='text-align: center'>Page {st.session_state.books_dash_page} of {pages}</div>", unsafe_allow_html=True)
        with p_c3:
            if st.button("Next >", key="next_books_page", disabled=st.session_state.books_dash_page >= pages, width="stretch"):
                st.session_state.books_dash_page += 1
                st.rerun()