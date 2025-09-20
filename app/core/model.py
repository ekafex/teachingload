# core/model.py
# -*- coding: utf-8 -*-
import sqlite3
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

# ---------- Core schema utilities ----------
def list_tables(conn: sqlite3.Connection) -> List[str]:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
    return [r[0] for r in conn.execute(q).fetchall()]

def table_schema(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f"PRAGMA table_info({table});", conn)

def table_has_rowid(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name=? AND sql LIKE '%WITHOUT ROWID%';", (table,))
    return cur.fetchone()[0] == 0

def read_table(conn: sqlite3.Connection, table: str, include_rowid: bool = True) -> pd.DataFrame:
    if include_rowid and table_has_rowid(conn, table):
        return pd.read_sql_query(f"SELECT rowid as __rowid__, * FROM {table};", conn)
    return pd.read_sql_query(f"SELECT * FROM {table};", conn)

def fk_references(conn: sqlite3.Connection, table: str):
    refs = []
    for t in list_tables(conn):
        try:
            fk = pd.read_sql_query(f"PRAGMA foreign_key_list({t});", conn)
        except Exception:
            continue
        for _, row in fk.iterrows():
            if str(row["table"]).lower() == table.lower():
                refs.append((t, row["from"], row["to"]))
    return refs

def count_impacts(conn: sqlite3.Connection, table: str, key_col: str, ids: List[Any]) -> List[Tuple[str, int]]:
    impacts = []
    for (child, child_col, _parent_col) in fk_references(conn, table):
        try:
            q = f"SELECT COUNT(*) FROM {child} WHERE {child_col} IN ({','.join(['?']*len(ids))});"
            n = conn.execute(q, ids).fetchone()[0]
            if n:
                impacts.append((child, n))
        except Exception:
            pass
    return impacts

def insert_row(conn: sqlite3.Connection, table: str, row: Dict[str, Any]):
    cols = ", ".join(row.keys())
    qmarks = ", ".join(["?"] * len(row))
    vals = list(row.values())
    conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({qmarks});", vals)

def delete_by_rowid(conn: sqlite3.Connection, table: str, rowids: List[int]):
    if not rowids: return
    q = f"DELETE FROM {table} WHERE rowid IN ({','.join(['?']*len(rowids))});"
    conn.execute(q, rowids)

def update_by_rowid(conn: sqlite3.Connection, table: str, rowid: int, values: Dict[str, Any]):
    sets = ", ".join([f"{k}=?" for k in values.keys()])
    params = list(values.values()) + [rowid]
    conn.execute(f"UPDATE {table} SET {sets} WHERE rowid=?;", params)

# ---------- Introspection helpers ----------
def pk_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    info = pd.read_sql_query(f"PRAGMA table_info({table});", conn)
    return [str(r["name"]) for _, r in info.iterrows() if int(r["pk"]) == 1]

def pk_of(conn: sqlite3.Connection, table: str) -> str:
    pks = pk_columns(conn, table)
    if pks:
        return pks[0]
    info = table_schema(conn, table)
    names = [c for c in info["name"].astype(str).tolist()]
    return "id" if "id" in names else (f"{table.rstrip('s')}_id" if f"{table.rstrip('s')}_id" in names else names[0])

def guess_id_column(conn: sqlite3.Connection, table: str) -> Optional[str]:
    pks = pk_columns(conn, table)
    if pks: return pks[0]
    info = table_schema(conn, table)
    candidates = [str(r["name"]) for _, r in info.iterrows() if str(r["name"]).lower().endswith("_id")]
    base = table.rstrip("s").lower() + "_id"
    for c in candidates:
        if c.lower() == base:
            return c
    return candidates[0] if candidates else None

def col_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    info = table_schema(conn, table)
    return col in info["name"].astype(str).tolist()

# ---------- Name resolution (singular/plural) ----------
def resolve_table(conn: sqlite3.Connection, preferred: str, alternate: str) -> str:
    names = set(n.lower() for n in list_tables(conn))
    return preferred if preferred.lower() in names else (alternate if alternate.lower() in names else preferred)

def assignments_table(conn: sqlite3.Connection) -> str:
    # new DB uses 'assignment'
    return resolve_table(conn, "assignments", "assignment")

def people_qual_table(conn: sqlite3.Connection) -> str:
    return resolve_table(conn, "people_qualifications", "people_qualification")

def people_unavail_table(conn: sqlite3.Connection) -> str:
    # keep the DB's current misspelling for compatibility
    return resolve_table(conn, "people_unavailability", "people_unavalability")

# ---------- Label column guesses ----------
def label_column_guess(conn: sqlite3.Connection, table: str) -> str:
    preferred = ["name", "title", "type", "label", "code"]
    info = table_schema(conn, table)
    cols = [str(r["name"]) for _, r in info.iterrows()]
    types = {str(r["name"]): str(r["type"]) for _, r in info.iterrows()}
    for p in preferred:
        if p in cols:
            return p
    # any TEXT-like column
    for c in cols:
        if "CHAR" in types.get(c, "").upper() or "TEXT" in types.get(c, "").upper():
            return c
    return cols[0] if cols else "rowid"

# ---------- Friendly label DataFrames ----------
def df_programs(conn: sqlite3.Connection) -> pd.DataFrame:
    pk = pk_of(conn, "programs")
    label = "name" if col_exists(conn, "programs", "name") else pk
    return pd.read_sql_query(
        f"SELECT {pk} AS __id__, COALESCE({label}, '') AS __label__ FROM programs ORDER BY __label__, __id__;",
        conn
    )

def df_courses(conn: sqlite3.Connection, program_id: int | None = None) -> pd.DataFrame:
    c_pk = pk_of(conn, "courses")
    p_pk = pk_of(conn, "programs")
    # FK from courses->programs
    prog_fk = "program_id" if col_exists(conn, "courses", "program_id") else p_pk
    label_c = "title" if col_exists(conn, "courses", "title") else ("name" if col_exists(conn, "courses", "name") else c_pk)
    label_p = "name" if col_exists(conn, "programs", "name") else p_pk
    base = f"""
    SELECT c.{c_pk} AS __id__,
           (COALESCE(p.{label_p},'') || ' · ' || COALESCE(c.{label_c},'')) AS __label__,
           c.{prog_fk} AS program_fk
    FROM courses c
    LEFT JOIN programs p ON c.{prog_fk} = p.{p_pk}
    """
    where, params = "", ()
    if program_id is not None:
        where = f"WHERE p.{p_pk} = ?"
        params = (int(program_id),)
    q = base + ((" " + where) if where else "") + " ORDER BY __label__, __id__;"
    return pd.read_sql_query(q, conn, params=params)

def df_types(conn: sqlite3.Connection) -> pd.DataFrame:
    pk = pk_of(conn, "types")
    label = "type" if col_exists(conn, "types", "type") else ("name" if col_exists(conn, "types", "name") else pk)
    return pd.read_sql_query(
        f"SELECT {pk} AS __id__, COALESCE({label}, '') AS __label__ FROM types ORDER BY __label__, __id__;",
        conn
    )

def df_people(conn: sqlite3.Connection) -> pd.DataFrame:
    pk = pk_of(conn, "people")
    label = "name" if col_exists(conn, "people", "name") else pk
    return pd.read_sql_query(
        f"SELECT {pk} AS __id__, COALESCE({label}, '') AS __label__ FROM people ORDER BY __label__, __id__;",
        conn
    )

def df_timeslots_joined(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Return canonical timeslots as: id, weekday (text), slot (text)
    by joining weekdays + timeslots through schedule when necessary.
    """
    # Simple cartesian isn't intended; we just shape what's already defined
    # "schedule" links day_id and timeslot_id; we still want the master lists:
    # weekdays(id, day), timeslots(id, slot)
    # Build all existing (weekday, slot) pairs appearing in schedule OR list all base to let user choose any:
    base = pd.read_sql_query("SELECT id AS ts_id, slot FROM timeslots ORDER BY slot;", conn)
    days = pd.read_sql_query("SELECT id AS day_id, day FROM weekdays ORDER BY id;", conn)
    # Produce a combined table for UI (not persisted): each row is a single slot label, we keep their own ids.
    # For picking we need separate ids; pages will search by (day, slot) to locate schedule rows.
    # To keep previous API, we *return repeated rows per day* so UI can index per day.
    base["key"] = 1; days["key"] = 1
    full = days.merge(base, on="key").drop(columns=["key"])
    full.rename(columns={"day": "weekday", "slot": "slot"}, inplace=True)
    # Provide a synthetic id per pair if needed (but UI later resolves by day_id+ts_id anyway)
    full["id"] = (full["day_id"].astype(str) + "-" + full["ts_id"].astype(str))
    # Reorder to match previous function contract: id, weekday, slot
    return full[["id", "weekday", "slot", "day_id", "ts_id"]]

def df_sections_with_label(conn: sqlite3.Connection,
                           program_id: int | None = None,
                           course_id: int | None = None,
                           type_id: int | None = None) -> pd.DataFrame:
    s_pk = pk_of(conn, "sections")
    c_pk = pk_of(conn, "courses")
    p_pk = pk_of(conn, "programs")
    t_pk = pk_of(conn, "types")

    # FK names (static in new DB)
    s_course_fk = "course_id"
    s_type_fk   = "type_id"
    c_prog_fk   = "program_id"

    label_p = "name" if col_exists(conn, "programs", "name") else p_pk
    label_c = "title" if col_exists(conn, "courses", "title") else ("name" if col_exists(conn, "courses", "name") else c_pk)
    label_t = "type"  if col_exists(conn, "types", "type")   else ("name" if col_exists(conn, "types", "name") else t_pk)

    base = f"""
    SELECT s.{s_pk} AS __id__,
           (COALESCE(p.{label_p},'') || ' · ' ||
            COALESCE(c.{label_c},'') || ' · ' ||
            COALESCE(t.{label_t},'') ||
            CASE WHEN s.split IS NOT NULL AND TRIM(s.split) <> '' THEN (' · Split ' || s.split) ELSE '' END
           ) AS __label__,
           c.{c_prog_fk} AS program_id, s.{s_course_fk} AS course_id, s.{s_type_fk} AS type_id,
           s.split, s.semester
    FROM sections s
    LEFT JOIN courses c ON s.{s_course_fk} = c.{c_pk}
    LEFT JOIN programs p ON c.{c_prog_fk} = p.{p_pk}
    LEFT JOIN types t ON s.{s_type_fk} = t.{t_pk}
    """
    where, params = [], []
    if program_id is not None:
        where.append(f"p.{p_pk} = ?"); params.append(int(program_id))
    if course_id is not None:
        where.append(f"c.{c_pk} = ?"); params.append(int(course_id))
    if type_id is not None:
        where.append(f"t.{t_pk} = ?"); params.append(int(type_id))
    if where:
        base += " WHERE " + " AND ".join(where)
    base += " ORDER BY __label__, __id__;"
    return pd.read_sql_query(base, conn, params=tuple(params))

# ---------- Friendlify ----------
def fk_map(conn: sqlite3.Connection, table: str) -> list[dict]:
    try:
        fk = pd.read_sql_query(f"PRAGMA foreign_key_list({table});", conn)
    except Exception:
        return []
    out = []
    for _, r in fk.iterrows():
        out.append({"from": str(r["from"]), "table": str(r["table"]), "to": str(r["to"])})
    return out

def friendlify_df(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> pd.DataFrame:
    """Replace FK ids with human labels where possible."""
    out = df.copy()
    for f in fk_map(conn, table):
        local_col, parent_table, parent_pk = f["from"], f["table"], f["to"]
        if local_col not in out.columns: continue
        # choose a label column on parent
        lab = label_column_guess(conn, parent_table)
        try:
            parent = pd.read_sql_query(
                f"SELECT {parent_pk} AS __id__, COALESCE({lab}, '') AS __label__ FROM {parent_table};", conn
            )
            m = parent.set_index("__id__")["__label__"].to_dict()
            out[f"{local_col}__label__"] = out[local_col].map(m)
        except Exception:
            pass
    return out

# --- Compatibility helpers expected by pages ---

def fk_options(conn: sqlite3.Connection,
               parent_table: str,
               parent_pk: str | None = None,
               label_col: str | None = None) -> pd.DataFrame:
    """
    Return id/label pairs for a parent table to populate FK dropdowns.

    Compatible with both call styles:
      - fk_options(conn, "programs")
      - fk_options(conn, "programs", "id", "name")
    If parent_pk/label_col are None, they are inferred from the schema.
    """
    parent_table = str(parent_table)
    # Infer PK if not provided
    if parent_pk is None:
        parent_pk = pk_of(conn, parent_table)
    # Infer label column if not provided
    if label_col is None:
        label_col = label_column_guess(conn, parent_table)

    # Safety: ensure columns exist (in case callers passed outdated names)
    cols = table_schema(conn, parent_table)["name"].astype(str).tolist()
    if parent_pk not in cols:
        parent_pk = pk_of(conn, parent_table)  # fall back
    if label_col not in cols:
        label_col = label_column_guess(conn, parent_table)

    q = f"""
        SELECT {parent_pk} AS __id__,
               COALESCE({label_col}, '') AS __label__
        FROM {parent_table}
        ORDER BY __label__, __id__;
    """
    try:
        return pd.read_sql_query(q, conn)
    except Exception:
        # Robust empty fallback
        return pd.DataFrame(columns=["__id__", "__label__"])


def df_timeslots(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Provide a simple timeslots listing for editors/selectors.
    New schema splits weekday into a separate table; this function returns
    the base slots only (id, slot). Pages that need (day, time) should join
    on 'weekdays' themselves.
    """
    try:
        return pd.read_sql_query("SELECT id AS __id__, slot AS __label__ FROM timeslots ORDER BY slot;", conn)
    except Exception:
        return pd.DataFrame(columns=["__id__", "__label__"])

def df_qualifications(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    List of qualifications, if the table exists. Falls back to empty if not present.
    """
    try:
        return fk_options(conn, "qualifications")
    except Exception:
        return pd.DataFrame(columns=["__id__", "__label__"])

# --- Compatibility helpers for pages expecting these symbols ---

def is_integer_affinity(sql_type: str) -> bool:
    """Return True if the declared SQL type looks integer-like (e.g., INT, INTEGER)."""
    return "INT" in str(sql_type).upper()

def next_int_id(conn: sqlite3.Connection, table: str, id_col: str) -> int:
    """
    Best-effort next integer id for manual PKs. Works even if id_col is TEXT-typed but stores ints.
    """
    try:
        row = conn.execute(f"SELECT COALESCE(MAX({id_col}), 0) + 1 FROM {table};").fetchone()
        if row and row[0] is not None:
            return int(row[0])
    except Exception:
        row = conn.execute(f"SELECT COALESCE(MAX(CAST({id_col} AS INTEGER)), 0) + 1 FROM {table};").fetchone()
        if row and row[0] is not None:
            return int(row[0])
    return 1

def guess_id_column(conn: sqlite3.Connection, table: str) -> str:
    """
    Guess the PK/ID column name for a table, preferring an explicit PK, then 'id',
    then a legacy '<singular>_id' fallback.
    """
    pks = pk_columns(conn, table)
    if pks:
        return pks[0]
    info = table_schema(conn, table)
    cols = [str(r["name"]) for _, r in info.iterrows()]
    if "id" in cols:
        return "id"
    legacy = table.rstrip("s").lower() + "_id"
    return legacy if legacy in cols else cols[0]

def friendlify_df(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace FK id columns in DataFrame with human-friendly '__label__' columns for display.
    Keeps original id columns (the page may hide them via UI).
    """
    out = df.copy()
    try:
        fk = pd.read_sql_query(f"PRAGMA foreign_key_list({table});", conn)
    except Exception:
        fk = pd.DataFrame(columns=["from","table","to"])
    for _, r in fk.iterrows():
        local_col  = str(r.get("from"))
        parent_tbl = str(r.get("table"))
        parent_pk  = str(r.get("to"))
        if local_col not in out.columns:
            continue
        label_col = label_column_guess(conn, parent_tbl)
        try:
            parent = pd.read_sql_query(
                f"SELECT {parent_pk} AS __id__, COALESCE({label_col}, '') AS __label__ FROM {parent_tbl};",
                conn
            )
            mapping = parent.set_index("__id__")["__label__"].to_dict()
            out[f"{local_col}__label__"] = out[local_col].map(mapping)
        except Exception:
            # If anything fails, just skip that FK
            pass
    return out

