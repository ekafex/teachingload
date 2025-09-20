# -*- coding: utf-8 -*-
import os
from pathlib import Path
import streamlit as st

#from app.core.io import DEFAULT_DB_REL, drop_cached_connection, make_working_copy, snapshot_bytes, save_as
from core.io import DEFAULT_DB_REL, drop_cached_connection, make_working_copy, snapshot_bytes, save_as

st.set_page_config(page_title="TeachLoad", page_icon="🎓", layout="wide")

APP_DIR = Path(__file__).resolve().parent

def init_state():
    if "base_db_path" not in st.session_state:
        # Default DB relative to app/ now
        st.session_state.base_db_path = str(APP_DIR / DEFAULT_DB_REL)
    if "work_db_path" not in st.session_state:
        base = st.session_state.base_db_path
        st.session_state.work_db_path = make_working_copy(base) if os.path.exists(base) else ""

init_state()

st.sidebar.header("Database")
default_display = st.session_state.get("base_db_path", "")
base_choice = st.sidebar.text_input(
    "Base DB path",
    value=default_display,
    help="Set this to your int-PK DB path under app/data, or upload a .db below."
)
uploaded = st.sidebar.file_uploader("...or upload a .db file", type=["db","sqlite","sqlite3"], accept_multiple_files=False)
if uploaded is not None:
    tmp_dir = APP_DIR / ".uploaded"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / uploaded.name
    with open(tmp_path, "wb") as f:
        f.write(uploaded.read())
    base_choice = str(tmp_path)

colA, colB, colC = st.sidebar.columns([1,1,1])
if colA.button("Use base → working"):
    st.session_state.base_db_path = base_choice
    drop_cached_connection()
    if os.path.exists(st.session_state.base_db_path):
        st.session_state.work_db_path = make_working_copy(st.session_state.base_db_path)
        st.success("Working copy created from base DB.")
    else:
        st.error("Base DB not found. Please set a valid path or upload a .db.")
    st.rerun()

if st.session_state.get("work_db_path"):
    try:
        snap = snapshot_bytes(st.session_state["work_db_path"])
        st.sidebar.download_button(
            "⬇️ Download working DB snapshot",
            data=snap,
            file_name=Path(st.session_state["work_db_path"]).name,
            mime="application/octet-stream"
        )
    except Exception:
        pass

if colB.button("Save As…"):
    st.session_state["save_as_mode"] = True

if colC.button("Discard working"):
    if st.session_state.get("work_db_path"):
        try:
            os.remove(st.session_state["work_db_path"])
        except Exception:
            pass
        st.session_state["work_db_path"] = ""
        drop_cached_connection()
        st.success("Working copy discarded.")
        st.rerun()

if st.session_state.get("save_as_mode"):
    with st.popover("Save working copy as…"):
        out = st.text_input("Destination path", value=str(APP_DIR / "data" / "teaching_load_saved.db"))
        if st.button("Confirm Save"):
            if st.session_state.get("work_db_path"):
                save_as(st.session_state["work_db_path"], out)
                st.success(f"Saved to {out}")
            st.session_state["save_as_mode"] = False

if st.session_state.get("work_db_path"):
    st.info(
        f"Editing working copy: `{st.session_state['work_db_path']}`  \n"
        f"Base: `{st.session_state['base_db_path']}`"
    )

st.title("TeachLoad")
st.write("Use the **Pages** menu (left) to access DB Editing and other features.")

