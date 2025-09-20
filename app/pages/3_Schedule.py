# app/pages/3_Schedule.py
# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

from core.io import get_connection, transaction
from core.model import (
    table_schema,
    df_programs, df_courses, df_types, df_sections_with_label,
    col_exists,
    assignments_table,
)

st.set_page_config(page_title="Schedule", page_icon="🗓️", layout="wide")

# ---------------- Utilities ----------------
def get_conn_from_state():
    work_db = st.session_state.get("work_db_path")
    if not work_db:
        st.warning("No working DB selected. Open the main page and create a working copy.")
        st.stop()
    return get_connection(work_db)

def has_table(conn, name: str) -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND lower(name)=lower(?)",
            (name,)
        ).fetchone()
        return bool(row)
    except Exception:
        return False

def pk_of(conn, table: str) -> str:
    info = table_schema(conn, table)
    pks = [str(r["name"]) for _, r in info.iterrows() if int(r["pk"]) == 1]
    if pks:
        return pks[0]
    cols = [str(c) for c in info["name"].tolist()]
    return "id" if "id" in cols else (table.rstrip("s") + "_id" if (table.rstrip("s") + "_id") in cols else cols[0])

def choose_labels(conn):
    """Return SQL snippets for program, course, type labels based on real columns."""
    prog_lbl   = "p.name"  if col_exists(conn, "programs", "name") else "CAST(p.id AS TEXT)"
    course_lbl = "c.title" if col_exists(conn, "courses", "title") else ("c.name" if col_exists(conn, "courses", "name") else "CAST(c.id AS TEXT)")
    type_lbl   = "t.type"  if col_exists(conn, "types", "type")   else ("t.name" if col_exists(conn, "types", "name") else "CAST(t.id AS TEXT)")
    return prog_lbl, course_lbl, type_lbl


# ----- Weekday normalization & ordering (Albanian canonical) -----
AL_ORDER = ["Hene", "Marte", "Merkure", "Enjte", "Premte", "Shtune", "Diel"]
AL_RANK = {w: i for i, w in enumerate(AL_ORDER)}

# very liberal alias map (lowercased keys)
_WD_ALIASES = {
    # English
    "mon": "Hene", "monday": "Hene",
    "tue": "Marte", "tues": "Marte", "tuesday": "Marte",
    "wed": "Merkure", "wednesday": "Merkure",
    "thu": "Enjte", "thur": "Enjte", "thurs": "Enjte", "thursday": "Enjte",
    "fri": "Premte", "friday": "Premte",
    "sat": "Shtune", "saturday": "Shtune",
    "sun": "Diel", "sunday": "Diel",
    # Albanian common spellings
    "hene": "Hene", "hënë": "Hene", "e hene": "Hene", "e hënë": "Hene",
    "marte": "Marte", "martë": "Marte", "e marte": "Marte", "e martë": "Marte",
    "merkure": "Merkure", "mërkurë": "Merkure", "e merkure": "Merkure", "e mërkurë": "Merkure",
    "enjte": "Enjte", "e enjte": "Enjte",
    "premte": "Premte", "e premte": "Premte",
    "shtune": "Shtune", "shtunë": "Shtune", "e shtune": "Shtune", "e shtunë": "Shtune",
    "diel": "Diel", "e diel": "Diel",
    # numbers (strings)
    "1": "Hene", "2": "Marte", "3": "Merkure", "4": "Enjte", "5": "Premte", "6": "Shtune", "7": "Diel",
}

def _lc(s): 
    return "" if s is None else str(s).strip().lower()

def canonical_weekday(s) -> str:
    """Map anything (English/Albanian/number/variants) to canonical Albanian weekday."""
    key = _lc(s)
    if key in _WD_ALIASES:
        return _WD_ALIASES[key]
    # try first 3 letters (eng) as a last resort
    if len(key) >= 3 and key[:3] in _WD_ALIASES:
        return _WD_ALIASES[key[:3]]
    # if already canonical, keep; else return raw (will be sorted last)
    raw = str(s).strip()
    return raw if raw in AL_ORDER else raw

def weekday_sort_key(s):
    c = canonical_weekday(s)
    return (AL_RANK.get(c, 999), c)


# ---------- Schema detection ----------
def schema_flags(conn):
    has_weekdays_tbl = has_table(conn, "weekdays") and col_exists(conn, "weekdays", "id") and col_exists(conn, "weekdays", "day")
    timeslots_has_weekday = col_exists(conn, "timeslots", "weekday")
    schedule_has_day = col_exists(conn, "schedule", "day_id")
    return {
        "has_weekdays_tbl": has_weekdays_tbl,
        "timeslots_has_weekday": timeslots_has_weekday,
        "schedule_has_day": schedule_has_day
    }

# ---------- Pairs (weekday, slot) unified view ----------
def df_timeslot_pairs(conn) -> pd.DataFrame:
    """
    Return a normalized dataframe with columns:
      pair_id (str), day_id (Int64 or NA), ts_id (int), weekday_label (str), slot (str)

    Works for both:
      - Schema A: weekdays + timeslots + schedule(day_id, timeslot_id)
      - Schema B: timeslots has weekday; schedule(timeslot_id)
    """
    flags = schema_flags(conn)

    # Always load base slots (no weekday here)
    slots = pd.read_sql_query("SELECT id AS ts_id, slot FROM timeslots ORDER BY slot;", conn)

    if flags["timeslots_has_weekday"] and not flags["schedule_has_day"]:
        # Schema B (legacy): timeslots has weekday
        ts = pd.read_sql_query("SELECT id AS ts_id, weekday, slot FROM timeslots ORDER BY weekday, slot;", conn)
        ts["weekday_label"] = ts["weekday"].apply(canonical_weekday)
        ts["pair_id"] = ts["ts_id"].astype(str)     # only timeslot id needed
        ts["day_id"] = pd.NA
        return ts[["pair_id","day_id","ts_id","weekday_label","slot"]]

    # Schema A (new): weekdays table + schedule.day_id
    if flags["has_weekdays_tbl"]:
        days = pd.read_sql_query("SELECT id AS day_id, day FROM weekdays ORDER BY id;", conn)
        days["key"]=1; slots["key"]=1
        pairs = days.merge(slots, on="key").drop(columns=["key"])
        pairs["weekday_label"] = pairs["day"].apply(canonical_weekday)
        pairs["weekday_label"] = pairs["weekday_label"].apply(canonical_weekday)
        pairs["pair_id"] = pairs["day_id"].astype(str) + "-" + pairs["ts_id"].astype(str)
        return pairs[["pair_id","day_id","ts_id","weekday_label","slot"]]

    # Fallback (very rare): synthesize Mon-Fri with whatever slots exist
    synth_days = pd.DataFrame({"day_id":[1,2,3,4,5], "day":["Monday","Tuesday","Wednesday","Thursday","Friday"]})
    synth_days["key"]=1; slots["key"]=1
    pairs = synth_days.merge(slots, on="key").drop(columns=["key"])
    pairs["weekday_label"] = pairs["day"].apply(canonical_weekday)
    pairs["pair_id"] = pairs["day_id"].astype(str) + "-" + pairs["ts_id"].astype(str)
    return pairs[["pair_id","day_id","ts_id","weekday_label","slot"]]

def used_pairs(conn, program_id: int, semester):
    """Return used pairs: for Schema A => set of (day_id, ts_id); for Schema B => set of ts_id."""
    flags = schema_flags(conn)
    if flags["schedule_has_day"]:
        if semester is None:
            rows = conn.execute("""
                SELECT DISTINCT sch.day_id, sch.timeslot_id
                FROM schedule sch
                JOIN sections s ON s.id = sch.section_id
                JOIN courses  c ON c.id = s.course_id
                WHERE c.program_id = ?
            """, (int(program_id),)).fetchall()
        else:
            rows = conn.execute("""
                SELECT DISTINCT sch.day_id, sch.timeslot_id
                FROM schedule sch
                JOIN sections s ON s.id = sch.section_id
                JOIN courses  c ON c.id = s.course_id
                WHERE c.program_id = ? AND s.semester = ?
            """, (int(program_id), semester)).fetchall()
        return {(int(r[0]), int(r[1])) for r in rows if r and r[0] is not None and r[1] is not None}
    else:
        if semester is None:
            rows = conn.execute("""
                SELECT DISTINCT sch.timeslot_id
                FROM schedule sch
                JOIN sections s ON s.id = sch.section_id
                JOIN courses  c ON c.id = s.course_id
                WHERE c.program_id = ?
            """, (int(program_id),)).fetchall()
        else:
            rows = conn.execute("""
                SELECT DISTINCT sch.timeslot_id
                FROM schedule sch
                JOIN sections s ON s.id = sch.section_id
                JOIN courses  c ON c.id = s.course_id
                WHERE c.program_id = ? AND s.semester = ?
            """, (int(program_id), semester)).fetchall()
        return {int(r[0]) for r in rows if r and r[0] is not None}

def available_pairs(conn, program_id: int, semester, include_occupied: bool) -> pd.DataFrame:
    pairs = df_timeslot_pairs(conn)
    if pairs.empty or include_occupied:
        return pairs
    used = used_pairs(conn, program_id, semester)
    if not used:
        return pairs

    flags = schema_flags(conn)
    if flags["schedule_has_day"]:
        pairs["key"] = list(zip(pairs["day_id"].astype(int), pairs["ts_id"].astype(int)))
        return pairs.loc[~pairs["key"].isin(used)].drop(columns=["key"]).reset_index(drop=True)
    else:
        return pairs.loc[~pairs["ts_id"].isin(list(used))].reset_index(drop=True)

def occupants_for_section(conn, tbl_assign: str, section_id: int) -> list[str]:
    rows = conn.execute(f"""
        SELECT COALESCE(pe.name, 'Person '||a.person_id) AS name,
               COALESCE(a.fraction, 1.0) AS frac
        FROM {tbl_assign} a
        LEFT JOIN people pe ON pe.id = a.person_id
        WHERE a.section_id = ?
        ORDER BY name
    """, (int(section_id),)).fetchall()
    out = []
    for nm, frac in rows:
        out.append(f"{nm} (x{float(frac):g})" if float(frac) != 1.0 else str(nm))
    return out

def section_context_ids(conn, section_id: int) -> dict:
    row = conn.execute("""
        SELECT p.id AS program_id, c.id AS course_id, s.semester
        FROM sections s
        JOIN courses  c ON c.id = s.course_id
        JOIN programs p ON p.id = c.program_id
        WHERE s.id = ?
    """, (int(section_id),)).fetchone()
    return {"program_id": row[0], "course_id": row[1], "semester": row[2]} if row else {}

def distinct_splits(conn, program_id: int, course_id: int, type_id: int) -> list[str]:
    q = """
        SELECT DISTINCT TRIM(COALESCE(s.split,'')) AS split_val
        FROM sections s
        WHERE s.course_id = ? AND s.type_id = ?
          AND EXISTS (SELECT 1 FROM courses c WHERE c.id = s.course_id AND c.program_id = ?)
        ORDER BY
          CASE WHEN split_val GLOB '[0-9]*' THEN CAST(split_val AS INTEGER) END,
          split_val
    """
    rows = conn.execute(q, (int(course_id), int(type_id), int(program_id))).fetchall()
    vals = [r[0] for r in rows]
    display = []
    for v in vals:
        display.append("(None)" if (v is None or str(v).strip() == "") else str(v))
    out = ["(All)"] + (sorted(set(display), key=lambda x: (x=="(None)", x)) if display else [])
    return out

def warn_unassigned_unscheduled(conn, tbl_assign: str, program_id: int, semester):
    PROG_LBL, COURSE_LBL, TYPE_LBL = choose_labels(conn)
    where_sem = "AND s.semester = ?" if semester is not None else ""
    params = (program_id,) + ((semester,) if semester is not None else tuple())

    unassigned = pd.read_sql_query(f"""
        SELECT s.id AS section_id,
               ({PROG_LBL} || ' · ' || {COURSE_LBL} || ' · ' || {TYPE_LBL}
                || CASE WHEN s.split IS NOT NULL AND TRIM(s.split)<>'' THEN (' · Split '||s.split) ELSE '' END
               ) AS section_label
        FROM sections s
        LEFT JOIN courses  c ON c.id = s.course_id
        LEFT JOIN programs p ON p.id = c.program_id
        LEFT JOIN types    t ON t.id = s.type_id
        LEFT JOIN {tbl_assign} a ON a.section_id = s.id
        WHERE p.id = ? {where_sem}
        GROUP BY s.id
        HAVING COUNT(a.section_id) = 0
        ORDER BY section_label;
    """, conn, params=params)

    unscheduled = pd.read_sql_query(f"""
        SELECT s.id AS section_id,
               ({PROG_LBL} || ' · ' || {COURSE_LBL} || ' · ' || {TYPE_LBL}
                || CASE WHEN s.split IS NOT NULL AND TRIM(s.split)<>'' THEN (' · Split '||s.split) ELSE '' END
               ) AS section_label
        FROM sections s
        LEFT JOIN courses  c ON c.id = s.course_id
        LEFT JOIN programs p ON p.id = c.program_id
        LEFT JOIN types    t ON t.id = s.type_id
        LEFT JOIN schedule sch ON sch.section_id = s.id
        WHERE p.id = ? {where_sem}
        GROUP BY s.id
        HAVING COUNT(sch.section_id) = 0
        ORDER BY section_label;
    """, conn, params=params)

    cols = st.columns(2)
    with cols[0]:
        st.caption("Unassigned sections")
        if unassigned.empty: st.success("None 🎉")
        else:
            for _, r in unassigned.iterrows(): st.write(f"- {r['section_label']}")
    with cols[1]:
        st.caption("Unscheduled sections")
        if unscheduled.empty: st.success("None 🎉")
        else:
            for _, r in unscheduled.iterrows(): st.write(f"- {r['section_label']}")

# ---------------- Page ----------------
conn = get_conn_from_state()
tbl_assign = assignments_table(conn)  # 'assignment' or 'assignments'
st.title("🗓️ Schedule")

tab_build, tab_overview = st.tabs(["➕ Build schedule", "📊 Overview"])

# =========================================================
# TAB 1: Build Schedule
# =========================================================
with tab_build:
    left, right = st.columns([1.4, 1.0])

    with left:
        st.subheader("1) Choose a Section")
        pick_mode = st.radio(
            "Section selection mode",
            ["Program → Course → Type → Split → Section", "Direct Section"],
            horizontal=True,
            key="sched_pick_mode_build"
        )

        selected_section = None
        ctx = {}

        if pick_mode.startswith("Program"):
            progs = df_programs(conn)
            if progs.empty:
                st.warning("No programs available.")
            else:
                idx_p = st.selectbox(
                    "Program",
                    list(range(len(progs))),
                    format_func=lambda i: progs.iloc[i]["__label__"],
                    key="sched_prog_build_select"
                )
                sel_prog = int(progs.iloc[idx_p]["__id__"])

                courses = df_courses(conn, sel_prog)
                if courses.empty:
                    st.warning("No courses in this program.")
                else:
                    idx_c = st.selectbox(
                        "Course",
                        list(range(len(courses))),
                        format_func=lambda i: courses.iloc[i]["__label__"],
                        key="sched_course_build_select"
                    )
                    sel_course = int(courses.iloc[idx_c]["__id__"])

                    types = df_types(conn)
                    if types.empty:
                        st.warning("No types available.")
                    else:
                        idx_t = st.selectbox(
                            "Type",
                            list(range(len(types))),
                            format_func=lambda i: types.iloc[i]["__label__"],
                            key="sched_type_build_select"
                        )
                        sel_type = int(types.iloc[idx_t]["__id__"])

                        # Split dropdown (dependent on P/C/T)
                        split_opts = distinct_splits(conn, sel_prog, sel_course, sel_type)
                        split_choice = st.selectbox("Split", split_opts, index=0, key="sched_split_build_select")

                        secs = df_sections_with_label(conn, program_id=sel_prog, course_id=sel_course, type_id=sel_type)
                        if split_choice != "(All)":
                            if split_choice == "(None)":
                                secs = secs[(secs["split"].isna()) | (secs["split"].astype(str).str.strip() == "")]
                            else:
                                secs = secs[secs["split"].astype(str) == split_choice]

                        if secs.empty:
                            st.warning("No sections for this Program/Course/Type/Split.")
                        else:
                            def sec_label(i):
                                sid = int(secs.iloc[i]["__id__"])
                                who = occupants_for_section(conn, tbl_assign, sid)
                                tag = ", ".join(who) if who else "unassigned"
                                return f"{secs.iloc[i]['__label__']}  ·  ({tag})"
                            idx_s = st.selectbox(
                                "Section",
                                list(range(len(secs))),
                                format_func=sec_label,
                                key="sched_section_build_select"
                            )
                            selected_section = int(secs.iloc[idx_s]["__id__"])
                            ctx = section_context_ids(conn, selected_section)
        else:
            # Direct section with optional Split filter
            secs = df_sections_with_label(conn)
            if secs.empty:
                st.warning("No sections available.")
            else:
                raw_splits = secs["split"].fillna("").astype(str).map(str.strip)
                has_empty = (raw_splits == "").any()
                uniq = sorted(v for v in raw_splits.unique() if v != "")
                split_opts = ["(All)"] + (["(None)"] if has_empty else []) + uniq
                split_choice = st.selectbox("Filter by Split (optional)", split_opts, index=0, key="sched_direct_split_filter")

                if split_choice != "(All)":
                    if split_choice == "(None)":
                        secs = secs[(secs["split"].isna()) | (secs["split"].astype(str).str.strip() == "")]
                    else:
                        secs = secs[secs["split"].astype(str) == split_choice]

                def sec_label(i):
                    sid = int(secs.iloc[i]["__id__"])
                    who = occupants_for_section(conn, tbl_assign, sid)
                    tag = ", ".join(who) if who else "unassigned"
                    return f"{secs.iloc[i]['__label__']}  ·  ({tag})"
                idx_s = st.selectbox(
                    "Section",
                    list(range(len(secs))),
                    format_func=sec_label,
                    key="sched_section_direct_select"
                )
                selected_section = int(secs.iloc[idx_s]["__id__"])
                ctx = section_context_ids(conn, selected_section)

        st.subheader("2) Choose a Timeslot (Weekday → Time)")
        include_occupied = st.toggle(
            "Include occupied slots (Program + Semester)",
            value=False,
            key="sched_include_occupied_toggle",
            help="When OFF, shows only free pairs in this Program & Semester."
        )

        flags = schema_flags(conn)
        if selected_section:
            program_id = ctx.get("program_id")
            semester   = ctx.get("semester")
            pairs = available_pairs(conn, program_id, semester, include_occupied)
            if pairs.empty:
                st.warning("No day/time pairs defined / available under current filters.")
                selected_wd = selected_slot = day_id = ts_id = None
            else:
                # Build pickers from pairs only (no direct timeslots.weekday usage)
                ordered = AL_ORDER[:]
                wd_unique = sorted(pairs["weekday_label"].dropna().unique().tolist(), key=weekday_sort_key)
                present_wd = [w for w in AL_ORDER if w in wd_unique] or wd_unique  # prefer canonical order, fallback to whatever exists
                selected_wd = st.selectbox("Weekday", present_wd, index=0, key="sched_weekday_build_select")

                slots_for_day = sorted(pairs.loc[pairs["weekday_label"] == selected_wd, "slot"].dropna().astype(str).unique().tolist())
                if not slots_for_day:
                    st.info("No time slots for the selected weekday under current filters.")
                    selected_slot = None; day_id = ts_id = None
                else:
                    selected_slot = st.selectbox("Time slot", slots_for_day, index=0, key="sched_slot_build_select")
                    row = pairs.loc[
                        (pairs["weekday_label"] == selected_wd) &
                        (pairs["slot"].astype(str) == str(selected_slot))
                    ].iloc[0]
                    day_id = None if pd.isna(row["day_id"]) else int(row["day_id"])
                    ts_id  = int(row["ts_id"])
        else:
            st.info("Select a section first to filter available timeslots.")
            selected_wd = selected_slot = None
            day_id = ts_id = None

        st.markdown("---")
        st.subheader("3) Save")
        if selected_section is not None:
            # Show existing times for this section using only joins compatible with both schemas
            if flags["schedule_has_day"]:
                existing = pd.read_sql_query("""
                    SELECT w.day AS weekday, t.slot AS slot
                    FROM schedule sch
                    LEFT JOIN weekdays w ON w.id = sch.day_id
                    LEFT JOIN timeslots t ON t.id = sch.timeslot_id
                    WHERE sch.section_id = ?
                    ORDER BY w.id, t.slot
                """, conn, params=(selected_section,))
                existing["weekday"] = existing["weekday"].apply(canonical_weekday)
            else:
                existing = pd.read_sql_query("""
                    SELECT t.weekday AS weekday, t.slot AS slot
                    FROM schedule sch
                    LEFT JOIN timeslots t ON t.id = sch.timeslot_id
                    WHERE sch.section_id = ?
                    ORDER BY t.weekday, t.slot
                """, conn, params=(selected_section,))
                existing["weekday"] = existing["weekday"].apply(canonical_weekday)

            if existing.empty:
                st.info("This section has no scheduled timeslots yet.")
            else:
                existing["timeslot"] = existing["weekday"].astype(str) + " · " + existing["slot"].astype(str)
                st.caption("Existing timeslots for this section")
                st.dataframe(existing[["timeslot"]], width='stretch', hide_index=True)
                existing["weekday"] = existing["weekday"].apply(canonical_weekday)


        can_save = (selected_section is not None) and (ts_id is not None) and ((day_id is not None) if flags["schedule_has_day"] else True)
        if st.button("➕ Add to schedule", type="primary", disabled=not can_save, key="sched_add_btn"):
            try:
                if flags["schedule_has_day"]:
                    dup = conn.execute(
                        "SELECT 1 FROM schedule WHERE section_id=? AND day_id=? AND timeslot_id=? LIMIT 1;",
                        (selected_section, day_id, ts_id)
                    ).fetchone()
                    if dup:
                        st.warning("This section is already scheduled on that day/time.")
                    else:
                        with transaction(conn) as cur:
                            cur.execute(
                                "INSERT INTO schedule(section_id, day_id, timeslot_id) VALUES (?, ?, ?);",
                                (int(selected_section), int(day_id), int(ts_id))
                            )
                        st.success("Scheduled ✅"); st.balloons(); st.rerun()
                else:
                    dup = conn.execute(
                        "SELECT 1 FROM schedule WHERE section_id=? AND timeslot_id=? LIMIT 1;",
                        (selected_section, ts_id)
                    ).fetchone()
                    if dup:
                        st.warning("This section is already scheduled in that timeslot.")
                    else:
                        with transaction(conn) as cur:
                            cur.execute(
                                "INSERT INTO schedule(section_id, timeslot_id) VALUES (?, ?);",
                                (int(selected_section), int(ts_id))
                            )
                        st.success("Scheduled ✅"); st.balloons(); st.rerun()
            except Exception as e:
                st.error(f"Insert failed: {e}")

    with right:
        st.subheader("Context")
        if selected_section:
            PROG_LBL, COURSE_LBL, TYPE_LBL = choose_labels(conn)
            info = pd.read_sql_query(f"""
                SELECT ({PROG_LBL} || ' · ' || {COURSE_LBL} || ' · ' || {TYPE_LBL}
                        || CASE WHEN s.split IS NOT NULL AND TRIM(s.split)<>'' THEN (' · Split '||s.split) ELSE '' END
                       ) AS label
                FROM sections s
                LEFT JOIN courses  c ON c.id = s.course_id
                LEFT JOIN programs p ON p.id = c.program_id
                LEFT JOIN types    t ON t.id = s.type_id
                WHERE s.id = ?
            """, conn, params=(selected_section,))
            lab = info.iloc[0]["label"] if not info.empty else f"Section {selected_section}"
            who = occupants_for_section(conn, tbl_assign, selected_section)
            st.write(lab + (" · " + ", ".join(who) if who else " · — unassigned —"))
            if ctx:
                st.caption(f"Program ID: {ctx.get('program_id')} · Course ID: {ctx.get('course_id')} · Semester: {ctx.get('semester')}")
        st.markdown("---")
        st.caption("Tip: Use the toggle to include occupied slots if parallel sessions are allowed.")

# =========================================================
# TAB 2: Overview (Program + optional Semester)
# =========================================================
with tab_overview:
    st.subheader("📋 Visual schedule")

    # program picker
    progs = df_programs(conn)
    if progs.empty:
        st.warning("No programs found."); st.stop()
    sel_prog_label = st.selectbox("Program", progs["__label__"].tolist(), index=0, key="sched_overview_prog_select")
    sel_prog_id = int(progs.loc[progs["__label__"] == sel_prog_label, "__id__"].iloc[0])

    # semester picker
    sems_df = pd.read_sql_query("""
        SELECT DISTINCT s.semester
        FROM sections s
        JOIN courses c ON c.id = s.course_id
        WHERE c.program_id = ?
        ORDER BY s.semester
    """, conn, params=(sel_prog_id,))
    sem_raw_values = sems_df["semester"].tolist()
    sem_labels = ["(Të gjitha)"] + [str(v) for v in sem_raw_values]
    sel_sem_label = st.selectbox("Semester", sem_labels, index=0, key="sched_overview_semester_select")
    sel_sem_raw = None if sel_sem_label == "(Të gjitha)" else sem_raw_values[sem_labels.index(sel_sem_label) - 1]

    # instructors per section (for green vs red)
    inst = pd.read_sql_query(f"""
        SELECT a.section_id, GROUP_CONCAT(pe.name, ', ') AS instructors
        FROM {assignments_table(conn)} a
        LEFT JOIN people pe ON pe.id = a.person_id
        GROUP BY a.section_id
    """, conn)
    inst_map = {int(r["section_id"]): (r["instructors"] or "") for _, r in inst.iterrows()}

    flags = schema_flags(conn)

    # Build axes from the unified pairs view (no direct weekday column usage)
    pairs_all = df_timeslot_pairs(conn)
    def norm_slot(x): return str(x).replace(" ", "").strip() if x is not None else ""
    pairs_all = df_timeslot_pairs(conn)
    pairs_all["weekday_label"] = pairs_all["weekday_label"].apply(canonical_weekday)
    col_order = [w for w in AL_ORDER if w in pairs_all["weekday_label"].unique().tolist()]
    if not col_order:
        # fallback to whatever exists, in canonical-ish order
        col_order = sorted(pairs_all["weekday_label"].dropna().unique().tolist(), key=weekday_sort_key)
    slots = sorted(pairs_all["slot"].dropna().astype(str).map(lambda x: x.replace(" ", "").strip()).unique().tolist())


    PROG_LBL, COURSE_LBL, TYPE_LBL = choose_labels(conn)

    def render_board(filter_semester_raw):
        if flags["schedule_has_day"]:
            base_sql = f"""
                SELECT
                  p.id   AS program_id,
                  s.id   AS section_id,
                  s.semester,
                  w.day     AS weekday,
                  tl.slot   AS slot,
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
                SELECT
                  p.id   AS program_id,
                  s.id   AS section_id,
                  s.semester,
                  tl.weekday AS weekday,
                  tl.slot    AS slot,
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
        if filter_semester_raw is not None:
            base_sql += " AND s.semester = ?"
            params.append(filter_semester_raw)
        base_sql += " ORDER BY 5, 4, section_label;"

        full = pd.read_sql_query(base_sql, conn, params=tuple(params))
        
        if not full.empty:
            full["weekday_al"] = full["weekday"].apply(canonical_weekday)
            full["slot_norm"] = full["slot"].map(lambda x: str(x).replace(" ", "").strip() if x is not None else "")

        grid  = pd.DataFrame(index=slots, columns=col_order, data="")
        state = pd.DataFrame(index=slots, columns=col_order, data="")  # "", "green", "red"
        grid.index.name = "Orari (slot)"

        if not full.empty:
            for _, r in full.iterrows():
                wd_al = r["weekday_al"]
                sl    = r["slot_norm"]
                if (sl in grid.index) and (wd_al in grid.columns):
                    names = (inst_map.get(int(r["section_id"]), "") or "").strip()
                    label = r["section_label"] + (f" · {names}" if names else "")
                    cell = grid.at[sl, wd_al]
                    grid.at[sl, wd_al] = (cell + "\n" if cell else "") + label
                    prev = state.at[sl, wd_al]
                    this_state = "green" if names else "red"
                    if prev == "":
                        state.at[sl, wd_al] = this_state
                    elif prev == "green" and this_state == "red":
                        state.at[sl, wd_al] = "red"

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

        st.dataframe(grid.style.apply(style_func, axis=None), width='stretch')

        colL, colR = st.columns([1,1])
        with colL:
            st.markdown(
                "<div style='display:flex;gap:12px;align-items:center;'>"
                "<div style='width:14px;height:14px;background:#d9f5d3;border:1px solid #b7e8af;'></div>"
                "<span>Scheduled + instructor(s)</span>"
                "<div style='width:14px;height:14px;background:#f9d2d0;border:1px solid #f0a8a3;margin-left:14px;'></div>"
                "<span>Scheduled but unassigned</span>"
                "</div>",
                unsafe_allow_html=True
            )
        with colR:
            with st.expander("⚠️ Unassigned / Unscheduled", expanded=False):
                warn_unassigned_unscheduled(conn, assignments_table(conn), program_id=sel_prog_id, semester=filter_semester_raw)

    if sel_sem_raw is None:
        for sem_raw in sem_raw_values:
            st.markdown(f"### Semester: **{sem_raw}**")
            render_board(sem_raw)
    else:
        st.markdown(f"### Semester: **{sel_sem_raw}**")
        render_board(sel_sem_raw)

