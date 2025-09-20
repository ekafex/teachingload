# -*- coding: utf-8 -*-
import os
import shutil
import sqlite3
from pathlib import Path
from contextlib import contextmanager
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]  # .../app/

# Default int-PK DB name, now relative to app/
DEFAULT_DB_REL = "data/teaching_load.db"

@st.cache_resource(show_spinner=False)
def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
    except sqlite3.OperationalError:
        pass
    return conn

def drop_cached_connection():
    get_connection.clear()

@contextmanager
def transaction(conn: sqlite3.Connection):
    cur = conn.cursor()
    try:
        cur.execute("BEGIN;")
        yield cur
        cur.execute("COMMIT;")
    except Exception as e:
        cur.execute("ROLLBACK;")
        raise e
    finally:
        cur.close()

def ensure_working_dir() -> Path:
    work = APP_DIR / ".working"
    work.mkdir(exist_ok=True)
    return work

def make_working_copy(source_db: str) -> str:
    work = ensure_working_dir()
    dest = work / (Path(source_db).stem + "_working.db")
    shutil.copy2(source_db, dest)
    return str(dest)

def snapshot_bytes(db_path: str) -> bytes:
    import tempfile
    conn = get_connection(db_path)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        tmp = sqlite3.connect(tf.name)
        conn.backup(tmp); tmp.close()
        data = open(tf.name, "rb").read()
    try:
        os.unlink(tf.name)
    except Exception:
        pass
    return data

def save_as(work_db_path: str, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(work_db_path, out_path)

