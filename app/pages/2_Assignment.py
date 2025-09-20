# app/pages/2_Assignment.py
# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

from core.io import get_connection, transaction
from core.model import (
    table_schema,
    col_exists,
    # rich label helpers
    df_people, df_programs, df_courses, df_types, df_sections_with_label,
    # plural/singular resolver
    assignments_table,
)

st.set_page_config(page_title="Assignments", page_icon="🧩", layout="wide")

# ---------------- Utilities ----------------
def get_conn_from_state():
    work_db = st.session_state.get("work_db_path")
    if not work_db:
        st.warning("No working DB selected. Go to the main app page and create a working copy from your base DB.")
        st.stop()
    return get_connection(work_db)

def pk_of(conn, table: str) -> str:
    info = table_schema(conn, table)
    pks = [str(r["name"]) for _, r in info.iterrows() if int(r["pk"]) == 1]
    if pks:
        return pks[0]
    names = [str(n) for n in info["name"].tolist()]
    if "id" in names:
        return "id"
    legacy = table.rstrip("s") + "_id"
    return legacy if legacy in names else names[0]

def choose_labels(conn):
    """Return SQL snippets for program, course, type labels based on real columns."""
    prog_lbl = "p.name" if col_exists(conn, "programs", "name") else "CAST(p.id AS TEXT)"
    course_lbl = "c.title" if col_exists(conn, "courses", "title") else ("c.name" if col_exists(conn, "courses", "name") else "CAST(c.id AS TEXT)")
    type_lbl = "t.type" if col_exists(conn, "types", "type") else ("t.name" if col_exists(conn, "types", "name") else "CAST(t.id AS TEXT)")
    return prog_lbl, course_lbl, type_lbl

def person_weekly_hours(conn, tbl_assign: str) -> dict[int, float]:
    """
    Best-effort weekly load per person:
      1) sum sections.hours_week OR sections.weekly_hours per assigned person
      2) else sum schedule entries (count) if schedule exists
      3) else count assignments
    """
    sec_pk = pk_of(conn, "sections")
    sec_cols = {c.lower() for c in table_schema(conn, "sections")["name"].tolist()}
    hours_col = None
    for cand in ("hours_week", "weekly_hours", "hours"):
        if cand in sec_cols:
            hours_col = cand
            break

    if hours_col:
        q = f"""
            SELECT a.person_id, COALESCE(SUM(s.{hours_col}), 0.0)
            FROM {tbl_assign} a
            JOIN sections s ON s.{sec_pk} = a.section_id
            GROUP BY a.person_id
        """
        try:
            return {int(pid): float(h) for (pid, h) in conn.execute(q).fetchall()}
        except Exception:
            pass

    # schedule-based count (treat each scheduled slot as 1.0 h unit if no durations are defined)
    try:
        q = f"""
            SELECT a.person_id, COUNT(*)*1.0
            FROM {tbl_assign} a
            JOIN schedule sch ON sch.section_id = a.section_id
            GROUP BY a.person_id
        """
        return {int(pid): float(h) for (pid, h) in conn.execute(q).fetchall()}
    except Exception:
        pass

    # plain count of assignments
    try:
        q = f"SELECT person_id, COUNT(*)*1.0 FROM {tbl_assign} GROUP BY person_id"
        return {int(pid): float(h) for (pid, h) in conn.execute(q).fetchall()}
    except Exception:
        return {}

def section_occupants(conn, tbl_assign: str, section_id: int) -> list[tuple[int, str, float]]:
    """Return [(person_id, person_name, fraction)] currently assigned to the given section."""
    rows = conn.execute(f"""
        SELECT a.person_id, COALESCE(pe.name, 'Person '||a.person_id) AS name, COALESCE(a.fraction, 1.0) AS frac
        FROM {tbl_assign} a
        LEFT JOIN people pe ON pe.id = a.person_id
        WHERE a.section_id = ?
        ORDER BY name;
    """, (int(section_id),)).fetchall()
    return [(int(pid), str(name), float(frac)) for (pid, name, frac) in rows]

def course_sections_with_occupancy(conn, tbl_assign: str,
                                   program_id: int | None, course_id: int | None, type_id: int | None,
                                   split_value: str | None = None) -> pd.DataFrame:
    """Sections with extra 'occupied_by' string; optional split filter."""
    secs = df_sections_with_label(conn, program_id=program_id, course_id=course_id, type_id=type_id).copy()
    if split_value not in (None, "(All)"):
        # Normalize empty/None
        if split_value == "(None)":
            secs = secs[(secs["split"].isna()) | (secs["split"].astype(str).str.strip() == "")]
        else:
            secs = secs[secs["split"].astype(str) == str(split_value)]
    secs["occupied_by"] = ""
    for i in range(len(secs)):
        sid = int(secs.iloc[i]["__id__"])
        occ = section_occupants(conn, tbl_assign, sid)
        if occ:
            names = ", ".join([f"{n} (x{frac:g})" if frac != 1.0 else n for _, n, frac in occ])
            secs.at[i, "occupied_by"] = names
    return secs

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
    # Normalize: show "(None)" when empty string appears
    display = []
    for v in vals:
        if v is None or v == "":
            display.append("(None)")
        else:
            display.append(v)
    # Always provide "(All)" at top
    out = ["(All)"] + (sorted(set(display), key=lambda x: (x=="(None)", x)) if display else [])
    return out

# ---------------- Page ----------------
conn = get_conn_from_state()
tbl_assign = assignments_table(conn)  # 'assignment' or 'assignments'
st.title("🧩 Assignments")

tab_assign, tab_overview = st.tabs(["➕ Assign", "📊 Overview"])

# =========================================================
# TAB 1: Add Assignment (smart wizard)
# =========================================================
with tab_assign:
    left, mid, right = st.columns([1.0, 1.3, 0.9])

    # ---- Instructor ----
    with left:
        st.subheader("1) Instructor")
        loads = person_weekly_hours(conn, tbl_assign)
        ppl = df_people(conn)
        if ppl.empty:
            st.warning("No people available in 'people'.")
            st.stop()

        def person_label(i):
            pid = int(ppl.iloc[i]["__id__"])
            h = loads.get(pid, 0.0)
            return f"{ppl.iloc[i]['__label__']}  ·  {h:g} h/w"

        idx_pe = st.selectbox("Select instructor", list(range(len(ppl))), format_func=person_label, key="assign_instr_select")
        person_id = int(ppl.iloc[idx_pe]["__id__"])

        # Current assignments for this instructor
        PROG_LBL, COURSE_LBL, TYPE_LBL = choose_labels(conn)
        mine_sql = f"""
            SELECT
              a.section_id,
              COALESCE(a.fraction, 1.0) AS fraction,
              ({PROG_LBL} || ' · ' || {COURSE_LBL} || ' · ' || {TYPE_LBL}
               || CASE WHEN s.split IS NOT NULL AND TRIM(s.split)<>'' THEN (' · Split '||s.split) ELSE '' END
              ) AS section_label
            FROM {tbl_assign} a
            LEFT JOIN sections  s ON s.id = a.section_id
            LEFT JOIN courses   c ON c.id = s.course_id
            LEFT JOIN programs  p ON p.id = c.program_id
            LEFT JOIN types     t ON t.id = s.type_id
            WHERE a.person_id = ?
            ORDER BY section_label;
        """
        mine = pd.read_sql_query(mine_sql, conn, params=(person_id,))
        with st.expander("Current assignments for this instructor", expanded=False):
            if mine.empty:
                st.info("No assignments yet.")
            else:
                st.dataframe(mine, width='stretch', hide_index=True)

    # ---- Program / Course / Type / Split / Section OR Direct Section ----
    with mid:
        st.subheader("2) Pick a Section")
        mode = st.radio("Selection mode", ["Program → Course → Type → Split → Section", "Direct Section"], horizontal=True, key="assign_mode_radio")

        selected_section = None
        derived_info = None

        if mode.startswith("Program"):
            # Program → Course → Type → Split → Section
            progs = df_programs(conn)
            if progs.empty:
                st.warning("No programs available. Add a program first.")
            else:
                idx_p = st.selectbox("Program", list(range(len(progs))), format_func=lambda i: progs.iloc[i]["__label__"], key="assign_prog_select")
                sel_prog = int(progs.iloc[idx_p]["__id__"])

                courses = df_courses(conn, sel_prog)
                if courses.empty:
                    st.warning("No courses under this program.")
                else:
                    idx_c = st.selectbox("Course", list(range(len(courses))), format_func=lambda i: courses.iloc[i]["__label__"], key="assign_course_select")
                    sel_course = int(courses.iloc[idx_c]["__id__"])

                    types = df_types(conn)
                    if types.empty:
                        st.warning("No types available.")
                    else:
                        idx_t = st.selectbox("Type", list(range(len(types))), format_func=lambda i: types.iloc[i]["__label__"], key="assign_type_select")
                        sel_type = int(types.iloc[idx_t]["__id__"])

                        # NEW: Split dropdown (dependent on P/C/T)
                        split_opts = distinct_splits(conn, sel_prog, sel_course, sel_type)
                        split_choice = st.selectbox("Split", split_opts, index=0, key="assign_split_select")

                        secs = course_sections_with_occupancy(conn, tbl_assign,
                                                              program_id=sel_prog, course_id=sel_course, type_id=sel_type,
                                                              split_value=split_choice if split_choice != "(All)" else None)
                        if secs.empty:
                            st.warning("No sections for this Program/Course/Type/Split.")
                        else:
                            def sec_label(i):
                                lab = secs.iloc[i]["__label__"]
                                occ = secs.iloc[i]["occupied_by"]
                                return f"{lab}  ·  ({'occupied: ' + occ if occ else 'free'})"
                            idx_s = st.selectbox("Section", list(range(len(secs))), format_func=sec_label, key="assign_section_select")
                            selected_section = int(secs.iloc[idx_s]["__id__"])

        else:
            # Direct Section -> Optional Split filter across all sections
            secs_all = df_sections_with_label(conn)
            if secs_all.empty:
                st.warning("No sections available. Add sections first.")
            else:
                # Build global split list
                raw_splits = secs_all["split"].fillna("").astype(str).map(str.strip)
                has_empty = (raw_splits == "").any()
                uniq = sorted(v for v in raw_splits.unique() if v != "")
                split_opts = ["(All)"] + (["(None)"] if has_empty else []) + uniq
                split_choice = st.selectbox("Filter by Split (optional)", split_opts, index=0, key="assign_direct_split_filter")

                # Filter by split
                secs = secs_all.copy()
                if split_choice != "(All)":
                    if split_choice == "(None)":
                        secs = secs[(secs["split"].isna()) | (secs["split"].astype(str).str.strip() == "")]
                    else:
                        secs = secs[secs["split"].astype(str) == split_choice]

                def sec_label(i):
                    sid = int(secs.iloc[i]["__id__"])
                    occ = section_occupants(conn, tbl_assign, sid)
                    tag = (", ".join([n for _, n, _ in occ])) if occ else "free"
                    return f"{secs.iloc[i]['__label__']}  ·  ({'occupied: ' + tag if occ else tag})"
                idx_s = st.selectbox("Section", list(range(len(secs))), format_func=sec_label, key="assign_section_direct_select")
                selected_section = int(secs.iloc[idx_s]["__id__"])

        # Inline warning if section already assigned
        if selected_section is not None:
            occ = section_occupants(conn, tbl_assign, selected_section)
            if occ:
                names = ", ".join([f"{n} (x{frac:g})" if frac != 1.0 else n for _, n, frac in occ])
                st.warning(f"This section already has: {names}")

    # ---- Fraction + Save ----
    with right:
        st.subheader("3) Fraction & Save")
        fraction = st.number_input("Fraction", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="assign_fraction_input")
        can_save = (selected_section is not None) and (person_id is not None)

        # Prevent exact duplicate (same person & section)
        dup = pd.read_sql_query(
            f"SELECT 1 FROM {tbl_assign} WHERE person_id=? AND section_id=? LIMIT 1;",
            conn, params=(person_id, selected_section if selected_section is not None else -1)
        )
        if not dup.empty:
            st.info("This instructor already has an assignment for this section (you can add another row for partial loads).")

        if st.button("➕ Add assignment", type="primary", disabled=not can_save, key="assign_add_btn"):
            try:
                with transaction(conn) as cur:
                    cur.execute(
                        f"INSERT INTO {tbl_assign}(person_id, section_id, fraction) VALUES (?, ?, ?);",
                        (int(person_id), int(selected_section), float(fraction))
                    )
                st.success("Assignment saved.")
                st.balloons()
                st.rerun()
            except Exception as e:
                st.error(f"Insert failed: {e}")

# =========================================================
# TAB 2: Overview (smart view)
# =========================================================
with tab_overview:
    st.subheader("📋 All Assignments (smart view)")

    # Filters
    colf1, colf2, colf3, colf4 = st.columns([1,1,1,1])
    progs = df_programs(conn)
    ppl = df_people(conn)

    with colf1:
        prog_filter = st.selectbox("Program", ["(All)"] + progs["__label__"].tolist(), index=0, key="assign_over_prog")

    with colf2:
        sel_prog_id = None
        if prog_filter != "(All)":
            sel_prog_id = int(progs.loc[progs["__label__"] == prog_filter, "__id__"].iloc[0])
        courses = df_courses(conn, sel_prog_id) if sel_prog_id else df_courses(conn, None)
        course_filter = st.selectbox("Course", ["(All)"] + courses["__label__"].tolist(), index=0, key="assign_over_course")

    with colf3:
        types = df_types(conn)
        type_filter = st.selectbox("Type", ["(All)"] + types["__label__"].tolist(), index=0, key="assign_over_type")

    with colf4:
        person_filter = st.selectbox("Instructor", ["(All)"] + ppl["__label__"].tolist(), index=0, key="assign_over_person")

    # Build dynamic label pieces once
    PROG_LBL, COURSE_LBL, TYPE_LBL = choose_labels(conn)

    # Base view
    overview_sql = f"""
        SELECT
          pe.name  AS instructor,
          {PROG_LBL}   AS program,
          {COURSE_LBL} AS course,
          {TYPE_LBL}   AS section_type,
          s.split      AS split,
          s.id         AS section,
          COALESCE(a.fraction, 1.0) AS fraction
        FROM {tbl_assign} a
        LEFT JOIN sections  s ON s.id = a.section_id
        LEFT JOIN courses   c ON c.id = s.course_id
        LEFT JOIN people    pe ON pe.id = a.person_id
        LEFT JOIN programs  p  ON p.id = c.program_id
        LEFT JOIN types     t  ON t.id = s.type_id
        ORDER BY {PROG_LBL}, {COURSE_LBL}, {TYPE_LBL}, s.split, s.id, pe.name;
    """
    view = pd.read_sql_query(overview_sql, conn)

    # Apply filters
    if prog_filter != "(All)":
        view = view.loc[view["program"] == prog_filter]
    if course_filter != "(All)":
        view = view.loc[view["course"] == course_filter]
    if type_filter != "(All)":
        view = view.loc[view["section_type"] == type_filter]
    if person_filter != "(All)":
        view = view.loc[view["instructor"] == person_filter]

    st.dataframe(view, width='stretch', hide_index=True)

