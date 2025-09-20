# app/pages/7_Results.py
# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

from core.io import get_connection
from core.model import (
    table_schema,
    df_programs, df_courses, df_types, df_sections_with_label, df_people,
    col_exists,
    assignments_table,
)

st.set_page_config(page_title="Results", page_icon="📊", layout="wide")

# =========================
# Weekday normalization
# =========================
AL_ORDER = ["Hene", "Marte", "Merkure", "Enjte", "Premte", "Shtune", "Diel"]
AL_RANK = {w: i for i, w in enumerate(AL_ORDER)}
_WD_ALIASES = {
    "mon":"Hene","monday":"Hene","tue":"Marte","tues":"Marte","tuesday":"Marte","wed":"Merkure","wednesday":"Merkure",
    "thu":"Enjte","thur":"Enjte","thurs":"Enjte","thursday":"Enjte","fri":"Premte","friday":"Premte",
    "sat":"Shtune","saturday":"Shtune","sun":"Diel","sunday":"Diel",
    "hene":"Hene","hënë":"Hene","e hene":"Hene","e hënë":"Hene","marte":"Marte","martë":"Marte","e marte":"Marte","e martë":"Marte",
    "merkure":"Merkure","mërkurë":"Merkure","e merkure":"Merkure","e mërkurë":"Merkure","enjte":"Enjte","e enjte":"Enjte",
    "premte":"Premte","e premte":"Premte","shtune":"Shtune","shtunë":"Shtune","e shtune":"Shtune","e shtunë":"Shtune",
    "diel":"Diel","e diel":"Diel",
    "1":"Hene","2":"Marte","3":"Merkure","4":"Enjte","5":"Premte","6":"Shtune","7":"Diel",
}
def _lc(s): return "" if s is None else str(s).strip().lower()
def canonical_weekday(s) -> str:
    k = _lc(s)
    if k in _WD_ALIASES: return _WD_ALIASES[k]
    if len(k)>=3 and k[:3] in _WD_ALIASES: return _WD_ALIASES[k[:3]]
    raw = str(s).strip()
    return raw if raw in AL_ORDER else raw
def weekday_sort_key(s):
    c = canonical_weekday(s)
    return (AL_RANK.get(c, 999), c)
def norm_slot(x): return str(x).replace(" ", "").strip() if x is not None else ""

# =========================
# DB helpers (same robustness as 3/4)
# =========================
def get_conn_from_state():
    db = st.session_state.get("work_db_path")
    if not db:
        st.warning("No working DB selected. Open the main page to create/select one.")
        st.stop()
    return get_connection(db)

def has_table(conn, name: str) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND lower(name)=lower(?)", (name,)
    ).fetchone())

def choose_labels(conn):
    prog_lbl   = "p.name"  if col_exists(conn, "programs", "name") else "CAST(p.id AS TEXT)"
    course_lbl = "c.title" if col_exists(conn, "courses", "title") else ("c.name" if col_exists(conn, "courses", "name") else "CAST(c.id AS TEXT)")
    type_lbl   = "t.type"  if col_exists(conn, "types", "type")   else ("t.name" if col_exists(conn, "types", "name") else "CAST(t.id AS TEXT)")
    return prog_lbl, course_lbl, type_lbl

def schema_flags(conn):
    has_weekdays_tbl = has_table(conn, "weekdays") and col_exists(conn, "weekdays", "id") and col_exists(conn, "weekdays", "day")
    timeslots_has_weekday = col_exists(conn, "timeslots", "weekday")
    schedule_has_day = col_exists(conn, "schedule", "day_id")
    return {
        "has_weekdays_tbl": has_weekdays_tbl,
        "timeslots_has_weekday": timeslots_has_weekday,
        "schedule_has_day": schedule_has_day
    }

def df_timeslot_pairs(conn) -> pd.DataFrame:
    flags = schema_flags(conn)
    slots = pd.read_sql_query("SELECT id AS ts_id, slot FROM timeslots ORDER BY slot;", conn)

    if flags["timeslots_has_weekday"] and not flags["schedule_has_day"]:
        ts = pd.read_sql_query("SELECT id AS ts_id, weekday, slot FROM timeslots ORDER BY weekday, slot;", conn)
        ts["weekday_label"] = ts["weekday"].apply(canonical_weekday)
        ts["pair_id"] = ts["ts_id"].astype(str)
        ts["day_id"] = pd.NA
        return ts[["pair_id","day_id","ts_id","weekday_label","slot"]]

    if flags["has_weekdays_tbl"]:
        days = pd.read_sql_query("SELECT id AS day_id, day FROM weekdays ORDER BY id;", conn)
        days["key"]=1; slots["key"]=1
        pairs = days.merge(slots, on="key").drop(columns=["key"])
        pairs["weekday_label"] = pairs["day"].apply(canonical_weekday)
        pairs["pair_id"] = pairs["day_id"].astype(str) + "-" + pairs["ts_id"].astype(str)
        return pairs[["pair_id","day_id","ts_id","weekday_label","slot"]]

    synth = pd.DataFrame({"day_id":[1,2,3,4,5], "day":["Monday","Tuesday","Wednesday","Thursday","Friday"]})
    synth["key"]=1; slots["key"]=1
    pairs = synth.merge(slots, on="key").drop(columns=["key"])
    pairs["weekday_label"] = pairs["day"].apply(canonical_weekday)
    pairs["pair_id"] = pairs["day_id"].astype(str) + "-" + pairs["ts_id"].astype(str)
    return pairs[["pair_id","day_id","ts_id","weekday_label","slot"]]

# =========================
# Load computation
# =========================
def person_weekly_hours(conn, program_id: int | None, semester) -> pd.DataFrame:
    """
    Returns DataFrame: instructor, hours (float), details columns
    Priority:
      1) sum sections.hours_week / weekly_hours / hours × assignment.fraction
      2) else count scheduled entries for that person's sections × fraction
      3) else number of assignments × fraction
    All filtered by Program (via courses.program_id) and optional Semester.
    """
    tbl_assign = assignments_table(conn)
    sec_pk = "id" if col_exists(conn, "sections", "id") else "section_id"
    # try to pick an hours column
    sec_cols = {c.lower(): c for c in table_schema(conn, "sections")["name"].tolist()}
    hours_col = None
    for cand in ("hours_week", "weekly_hours", "hours"):
        if cand in sec_cols:
            hours_col = sec_cols[cand]
            break

    where_prog = " AND c.program_id = ?" if program_id is not None else ""
    where_sem  = " AND s.semester = ?"   if semester is not None else ""
    params = []
    if program_id is not None: params.append(program_id)
    if semester is not None: params.append(semester)

    # Strategy 1: section hours
    if hours_col:
        q1 = f"""
            SELECT pe.name AS instructor, SUM(COALESCE(s.{hours_col},0.0)*COALESCE(a.fraction,1.0)) AS hours
            FROM {tbl_assign} a
            JOIN sections s ON s.{sec_pk} = a.section_id
            JOIN courses  c ON c.id = s.course_id
            LEFT JOIN people pe ON pe.id = a.person_id
            WHERE 1=1 {where_prog} {where_sem}
            GROUP BY pe.name
            ORDER BY hours DESC, instructor ASC;
        """
        try:
            df = pd.read_sql_query(q1, conn, params=tuple(params))
            if not df.empty:
                return df
        except Exception:
            pass

    # Strategy 2: schedule count × fraction
    flags = schema_flags(conn)
    if flags["schedule_has_day"]:
        sch_join = "JOIN schedule sch ON sch.section_id = a.section_id"
    else:
        sch_join = "JOIN schedule sch ON sch.section_id = a.section_id"

    q2 = f"""
        SELECT pe.name AS instructor, SUM(1.0*COALESCE(a.fraction,1.0)) AS hours
        FROM {tbl_assign} a
        {sch_join}
        JOIN sections s ON s.{sec_pk} = a.section_id
        JOIN courses  c ON c.id = s.course_id
        LEFT JOIN people pe ON pe.id = a.person_id
        WHERE 1=1 {where_prog} {where_sem}
        GROUP BY pe.name
        ORDER BY hours DESC, instructor ASC;
    """
    try:
        df = pd.read_sql_query(q2, conn, params=tuple(params))
        if not df.empty:
            return df
    except Exception:
        pass

    # Strategy 3: assignment count × fraction
    q3 = f"""
        SELECT pe.name AS instructor, SUM(1.0*COALESCE(a.fraction,1.0)) AS hours
        FROM {tbl_assign} a
        JOIN sections s ON s.{sec_pk} = a.section_id
        JOIN courses  c ON c.id = s.course_id
        LEFT JOIN people pe ON pe.id = a.person_id
        WHERE 1=1 {where_prog} {where_sem}
        GROUP BY pe.name
        ORDER BY hours DESC, instructor ASC;
    """
    df = pd.read_sql_query(q3, conn, params=tuple(params))
    return df

# =========================
# Page
# =========================
conn = get_conn_from_state()
tbl_assign = assignments_table(conn)
flags = schema_flags(conn)
PROG_LBL, COURSE_LBL, TYPE_LBL = choose_labels(conn)

st.title("📊 Results")

tab_cal, tab_loads = st.tabs(["🗓️ Calendar (read-only)", "⏱️ Instructor Loads"])

# ------------------------------------------------------------------
# Tab 1: Calendar (read-only, per Program + Semester)
# ------------------------------------------------------------------
with tab_cal:
    col1, col2 = st.columns([1,1])
    with col1:
        progs = df_programs(conn)
        if progs.empty:
            st.warning("No programs defined."); st.stop()
        sel_prog_label = st.selectbox("Program", progs["__label__"].tolist(), index=0, key="res_prog")
        sel_prog_id = int(progs.loc[progs["__label__"] == sel_prog_label, "__id__"].iloc[0])
    with col2:
        sems_df = pd.read_sql_query("""
            SELECT DISTINCT s.semester
            FROM sections s
            JOIN courses c ON c.id = s.course_id
            WHERE c.program_id = ?
            ORDER BY s.semester
        """, conn, params=(sel_prog_id,))
        sem_raw = sems_df["semester"].tolist()
        sem_labels = ["(Të gjitha)"] + [str(v) for v in sem_raw]
        sel_sem_label = st.selectbox("Semester", sem_labels, index=0, key="res_sem")
        sel_sem_val = None if sel_sem_label=="(Të gjitha)" else sem_raw[sem_labels.index(sel_sem_label)-1]

    # axes from normalized pairs
    pairs = df_timeslot_pairs(conn)
    pairs["weekday_label"] = pairs["weekday_label"].apply(canonical_weekday)
    col_order = [w for w in AL_ORDER if w in pairs["weekday_label"].unique().tolist()]
    if not col_order:
        col_order = sorted(pairs["weekday_label"].dropna().unique().tolist(), key=weekday_sort_key)
    slot_order = sorted(pairs["slot"].dropna().astype(str).map(norm_slot).unique().tolist())

    # board data
    if flags["schedule_has_day"]:
        base_sql = f"""
            SELECT s.id AS section_id, w.day AS weekday, tl.slot AS slot,
                   ({PROG_LBL} || ' · ' || {COURSE_LBL} || ' · ' || {TYPE_LBL}
                    || CASE WHEN s.split IS NOT NULL AND TRIM(s.split)<>'' THEN (' · Split '||s.split) ELSE '' END
                   ) AS section_label
            FROM schedule sch
            JOIN sections s   ON s.id = sch.section_id
            JOIN courses  c   ON c.id = s.course_id
            JOIN programs p   ON p.id = c.program_id
            JOIN types    t   ON t.id = s.type_id
            JOIN weekdays w   ON w.id = sch.day_id
            JOIN timeslots tl ON tl.id = sch.timeslot_id
            WHERE p.id = ?
        """
    else:
        base_sql = f"""
            SELECT s.id AS section_id, tl.weekday AS weekday, tl.slot AS slot,
                   ({PROG_LBL} || ' · ' || {COURSE_LBL} || ' · ' || {TYPE_LBL}
                    || CASE WHEN s.split IS NOT NULL AND TRIM(s.split)<>'' THEN (' · Split '||s.split) ELSE '' END
                   ) AS section_label
            FROM schedule sch
            JOIN sections s   ON s.id = sch.section_id
            JOIN courses  c   ON c.id = s.course_id
            JOIN programs p   ON p.id = c.program_id
            JOIN types    t   ON t.id = s.type_id
            JOIN timeslots tl ON tl.id = sch.timeslot_id
            WHERE p.id = ?
        """
    params = [sel_prog_id]
    if sel_sem_val is not None:
        base_sql += " AND s.semester = ?"
        params.append(sel_sem_val)
    base_sql += " ORDER BY 3, 2, section_label;"

    board = pd.read_sql_query(base_sql, conn, params=tuple(params))
    if not board.empty:
        board["weekday_al"] = board["weekday"].apply(canonical_weekday)
        board["slot_norm"] = board["slot"].map(norm_slot)

    # render read-only grid as a table (one cell potentially multi-line)
    grid = pd.DataFrame(index=slot_order, columns=col_order, data="")
    grid.index.name = "Slot"
    if not board.empty:
        for _, r in board.iterrows():
            wd = r["weekday_al"]; sl = r["slot_norm"]
            if wd in grid.columns and sl in grid.index:
                cell = grid.at[sl, wd]
                grid.at[sl, wd] = (cell + "\n" if cell else "") + r["section_label"]

    # simple color state (green if cell has any instructor)
    inst = pd.read_sql_query(f"""
        SELECT a.section_id, GROUP_CONCAT(pe.name, ', ') AS instructors
        FROM {tbl_assign} a
        LEFT JOIN people pe ON pe.id = a.person_id
        GROUP BY a.section_id
    """, conn)
    inst_map = {int(r["section_id"]): (r["instructors"] or "") for _, r in inst.iterrows()}
    state = pd.DataFrame(index=slot_order, columns=col_order, data="")
    if not board.empty:
        for _, r in board.iterrows():
            wd = r["weekday_al"]; sl = r["slot_norm"]
            if wd in state.columns and sl in state.index:
                has = bool(inst_map.get(int(r["section_id"]), ""))
                prev = state.at[sl, wd]
                this_state = "green" if has else "red"
                if prev == "": state.at[sl, wd] = this_state
                elif prev == "green" and this_state == "red": state.at[sl, wd] = "red"

    def style_func(df):
        styled = df.copy()
        for r in styled.index:
            for c in styled.columns:
                s = state.at[r, c]
                if s == "green":
                    styled.at[r, c] = "background-color: #d9f5d3; white-space: pre-line;"
                elif s == "red":
                    styled.at[r, c] = "background-color: #f9d2d0; white-space: pre-line;"
                else:
                    styled.at[r, c] = "white-space: pre-line;"
        return styled

    st.dataframe(grid.style.apply(style_func, axis=None), width="stretch")

# ------------------------------------------------------------------
# Tab 2: Loads (hours/week per instructor)
# ------------------------------------------------------------------
with tab_loads:
    colL, colR = st.columns([1,1])
    with colL:
        progs = df_programs(conn)
        if progs.empty:
            st.warning("No programs."); st.stop()
        sel_prog_label2 = st.selectbox("Program", progs["__label__"].tolist(), index=0, key="res_load_prog")
        sel_prog_id2 = int(progs.loc[progs["__label__"] == sel_prog_label2, "__id__"].iloc[0])
    with colR:
        sems_df2 = pd.read_sql_query("""
            SELECT DISTINCT s.semester
            FROM sections s
            JOIN courses c ON c.id = s.course_id
            WHERE c.program_id = ?
            ORDER BY s.semester
        """, conn, params=(sel_prog_id2,))
        sem_raw2 = sems_df2["semester"].tolist()
        sem_labels2 = ["(Të gjitha)"] + [str(v) for v in sem_raw2]
        sel_sem_label2 = st.selectbox("Semester", sem_labels2, index=0, key="res_load_sem")
        sel_sem_val2 = None if sel_sem_label2=="(Të gjitha)" else sem_raw2[sem_labels2.index(sel_sem_label2)-1]

    loads = person_weekly_hours(conn, sel_prog_id2, sel_sem_val2)
    if loads.empty:
        st.info("No data to show (no assignments yet?).")
    else:
        loads = loads.fillna({"instructor":"(Unnamed)"}).sort_values(by=["hours","instructor"], ascending=[False, True])
        loads["hours"] = loads["hours"].astype(float).round(2)
        st.subheader("Table")
        st.dataframe(loads.rename(columns={"instructor":"Instructor","hours":"Hours/week"}),
                     width="stretch", hide_index=True)

        st.subheader("Bar chart")
        # For Streamlit built-in charts, keep it simple:
        chart_df = loads.rename(columns={"instructor":"Instructor","hours":"Hours/week"})
        st.bar_chart(chart_df, x="Instructor", y="Hours/week", height=360)

