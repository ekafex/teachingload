# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

from core.io import get_connection, transaction
from core.model import (
    list_tables, read_table, table_schema, table_has_rowid,
    fk_references, count_impacts, insert_row, delete_by_rowid, update_by_rowid,
    # friendly helpers
    pk_columns, fk_map, label_column_guess, fk_options,
    # rich label sources
    df_programs, df_courses, df_types, df_sections_with_label,
    df_people, df_timeslots, df_qualifications
)

st.set_page_config(page_title="DB Edit", page_icon="🗄️", layout="wide")

# ---------------- Utilities specific to this page ----------------
def get_conn_from_state():
    work_db = st.session_state.get("work_db_path")
    if not work_db:
        st.warning("No working DB selected. Go to the main app page and create a working copy from your base DB.")
        st.stop()
    return get_connection(work_db)

def pk_of(conn, table: str) -> str:
    """Return the first PK column name of table (or 'id' fallback)."""
    pks = pk_columns(conn, table)
    if pks:
        return pks[0]
    info = table_schema(conn, table)
    names = [str(c) for c in info["name"].tolist()]
    if "id" in names:
        return "id"
    # legacy-style fallback (e.g., section_id)
    t = table.rstrip("s").lower() + "_id"
    return t if t in names else names[0]

def get_person_weekly_hours(conn) -> dict[int, float]:
    """
    Best-effort compute weekly hours per person from assignments:
    1) If sections has 'hours' or 'weekly_hours', sum that per assigned person.
    2) Else, if schedule + timeslots exist, count scheduled slots (assume 1.0 per slot unless a duration column exists: minutes/duration/mins).
    3) Else, fallback to counting number of assigned sections.
    """
    sec_pk = pk_of(conn, "sections")
    # 1) sections.hours or sections.weekly_hours
    sec_cols = {c.lower() for c in table_schema(conn, "sections")["name"].tolist()}
    hours_col = "hours" if "hours" in sec_cols else ("weekly_hours" if "weekly_hours" in sec_cols else None)
    if hours_col:
        q = f"""
        SELECT a.person_id, COALESCE(SUM(s.{hours_col}), 0.0) AS weekly_hours
        FROM assignments a
        JOIN sections s ON a.section_id = s.{sec_pk}
        GROUP BY a.person_id
        """
        rows = conn.execute(q).fetchall()
        return {int(pid): float(h) for (pid, h) in rows}

    # 2) schedule + timeslots duration
    try:
        # find timeslot duration if any
        ts_cols = {c.lower() for c in table_schema(conn, "timeslots")["name"].tolist()}
        dur_col = None
        for candidate in ("minutes", "duration_minutes", "duration", "mins"):
            if candidate in ts_cols:
                dur_col = candidate
                break
        if dur_col:
            q = f"""
            SELECT a.person_id, COALESCE(SUM(t.{dur_col})/60.0, 0.0) AS weekly_hours
            FROM assignments a
            JOIN schedule sch ON sch.section_id = a.section_id
            JOIN timeslots t ON t.id = sch.timeslot_id
            GROUP BY a.person_id
            """
            rows = conn.execute(q).fetchall()
            return {int(pid): float(h) for (pid, h) in rows}
        else:
            # treat each scheduled slot as 1 hour
            q = """
            SELECT a.person_id, COUNT(*)*1.0 AS weekly_hours
            FROM assignments a
            JOIN schedule sch ON sch.section_id = a.section_id
            GROUP BY a.person_id
            """
            rows = conn.execute(q).fetchall()
            return {int(pid): float(h) for (pid, h) in rows}
    except Exception:
        pass

    # 3) fallback: number of assigned sections
    q = "SELECT person_id, COUNT(*)*1.0 AS weekly_hours FROM assignments GROUP BY person_id"
    try:
        rows = conn.execute(q).fetchall()
        return {int(pid): float(h) for (pid, h) in rows}
    except Exception:
        return {}

def section_occupants(conn, section_id: int) -> list[tuple[int, str]]:
    """Return list of (person_id, label) currently assigned to the given section."""
    sec_pk = pk_of(conn, "sections")
    # Identify people currently assigned to this section
    rows = conn.execute("SELECT person_id FROM assignments WHERE section_id = ?", (int(section_id),)).fetchall()
    if not rows:
        return []
    ppl_df = df_people(conn).set_index("__id__")
    out = []
    for (pid,) in rows:
        if pid in ppl_df.index:
            out.append((int(pid), str(ppl_df.loc[pid]["__label__"])))
        else:
            out.append((int(pid), f"Person {pid}"))
    return out

def course_sections_with_occupancy(conn, program_id: int | None, course_id: int | None, type_id: int | None):
    """Return sections DataFrame with an extra 'occupied_by' label column."""
    df_secs = df_sections_with_label(conn, program_id=program_id, course_id=course_id, type_id=type_id).copy()
    df_secs["occupied_by"] = ""
    for i in range(len(df_secs)):
        sid = int(df_secs.iloc[i]["__id__"])
        occ = section_occupants(conn, sid)
        if occ:
            names = ", ".join([o[1] for o in occ])
            df_secs.at[i, "occupied_by"] = names
    return df_secs

# ---------------- App main ----------------
conn = get_conn_from_state()

st.title("🗄️ Database Loader & Editor")

tables = list_tables(conn)
if not tables:
    st.warning("No user tables found."); st.stop()

# Put commonly-edited tables first (pluralized assignments)
preferred = [
    "people","ranks","programs","courses","types","sections",
    # both spellings:
    "assignments","assignment",
    "preferences","exclusions",
    "timeslots","weekdays","schedule",
    "qualifications", "people_qualifications", "people_qualification",
    "section_required_qualifications",
    "people_availability", "people_unavailability", "people_unavalability",
    "students","course_semesters","section_preferences"
]

present = [t for t in preferred if t in tables]
others = [t for t in tables if t not in present]
ordered_tables = present + others

entity = st.selectbox("Entity (table)", ordered_tables, index=ordered_tables.index("sections") if "sections" in ordered_tables else 0)
action = st.selectbox("Action", ["View/Edit grid", "Add", "Delete", "Export CSV", "Import CSV"], index=0,
                      help="Add/Delete are guided with validations and confirmations.")
show_raw_ids = st.checkbox("Show raw ID columns", value=False,
                           help="Toggle to reveal numeric foreign keys and IDs for power users.")
st.markdown("---")

# ---------------- View/Edit grid ----------------
if action == "View/Edit grid":
    from core.model import friendlify_df, fk_map
    has_rowid = table_has_rowid(conn, entity)

    if not has_rowid:
        st.info("This table was created WITHOUT ROWID; showing read-only view.")
        df = read_table(conn, entity, include_rowid=False)
        display_df = friendlify_df(conn, entity, df)
        if not show_raw_ids:
            for f in fk_map(conn, entity):
                col = f["from"]
                if col in display_df.columns:
                    display_df.drop(columns=[col], inplace=True)
        # Hide first column visually (even though no __rowid__ here)
        st.markdown("""
        <style>
        [data-testid="stDataEditor"] thead tr th:first-child,
        [data-testid="stDataEditor"] tbody tr td:first-child { display: none !important; }
        </style>
        """, unsafe_allow_html=True)
        st.dataframe(display_df, width='stretch', hide_index=True)

    else:
        df = read_table(conn, entity, include_rowid=True)
        display_df = friendlify_df(conn, entity, df)
        if not show_raw_ids:
            from core.model import fk_map
            for f in fk_map(conn, entity):
                col = f["from"]
                if col in display_df.columns:
                    display_df.drop(columns=[col], inplace=True)

        # Ensure __rowid__ is first for internal mapping, but we'll hide it via CSS
        cols = display_df.columns.tolist()
        if "__rowid__" in cols:
            cols.remove("__rowid__")
            display_df = display_df[["__rowid__"] + cols]

        # Hide first visible column (our __rowid__) in the UI
        st.markdown("""
        <style>
        [data-testid="stDataEditor"] thead tr th:first-child,
        [data-testid="stDataEditor"] tbody tr td:first-child { display: none !important; }
        </style>
        """, unsafe_allow_html=True)

        # Disable editing of PK columns if present
        pk_cols = pk_columns(conn, entity)
        column_config = {"__rowid__": st.column_config.NumberColumn("__rowid__", help="Internal ROWID (hidden)", disabled=True)}
        for pk in pk_cols:
            if pk in display_df.columns:
                column_config[pk] = st.column_config.TextColumn(pk, disabled=True, help="Primary key")

        edited_df = st.data_editor(
            display_df,
            num_rows="dynamic",
            width='stretch',
            hide_index=True,
            column_config=column_config,
        )

        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("💾 Save changes", type="primary"):
                try:
                    original_df = df.copy()
                    editable_cols = [c for c in original_df.columns if c != "__rowid__"]

                    orig_ids = set(pd.to_numeric(original_df["__rowid__"], errors="coerce").dropna().astype(int))
                    new_ids  = set(pd.to_numeric(edited_df.get("__rowid__", pd.Series([], dtype=float)), errors="coerce").dropna().astype(int))

                    to_delete = sorted(orig_ids - new_ids)

                    to_update = []
                    ed_indexed = edited_df.set_index("__rowid__", drop=False)
                    for rid in sorted(orig_ids & new_ids):
                        a = original_df.loc[original_df["__rowid__"] == rid, editable_cols]
                        b = ed_indexed.loc[rid, editable_cols] if rid in ed_indexed.index else None
                        if a.empty or b is None: continue
                        if not a.fillna(pd.NA).reset_index(drop=True).equals(pd.DataFrame([b]).fillna(pd.NA).reset_index(drop=True)):
                            to_update.append((rid, dict(b)))

                    ins_rows = edited_df[edited_df.get("__rowid__", pd.Series([float("nan")] * len(edited_df))).isna()]
                    ins_rows = ins_rows[[c for c in editable_cols if c in ins_rows.columns]].dropna(how="all")
                    inserts = [{c: (None if pd.isna(row.get(c, None)) else row.get(c, None)) for c in editable_cols if c in ins_rows.columns} for _, row in ins_rows.iterrows()]

                    with transaction(conn) as cur:
                        delete_by_rowid(conn, entity, [int(r) for r in to_delete])
                        if to_update:
                            for rid, vals in to_update:
                                update_by_rowid(conn, entity, int(rid), {k: vals.get(k) for k in editable_cols})
                        for row in inserts:
                            if row: insert_row(conn, entity, row)
                    st.success("Changes saved."); st.rerun()
                except Exception as e:
                    st.error(f"Save failed: {e}")
        with c2:
            if st.button("↩️ Revert (reload)"):
                st.rerun()

# ---------------- Add ----------------
elif action == "Add":
    schema = table_schema(conn, entity)
    from core.model import (
        guess_id_column, is_integer_affinity, next_int_id, friendlify_df
    )
    id_col = guess_id_column(conn, entity)
    fk_list = fk_map(conn, entity)
    fk_by_local = {f["from"]: f for f in fk_list}

    # Build base field list excluding id_col
    form_cols = []
    types = {}
    for _, r in schema.iterrows():
        col = str(r["name"]); typ = str(r["type"])
        types[col] = typ
        if id_col and col == id_col:
            continue
        form_cols.append((col, typ, int(r["notnull"])))

    st.subheader(f"Add row to `{entity}`")
    values = {}
    skip_cols = set()
    entity_l = entity.lower()

    with st.form("add_form"):
        # ---------------- ASSIGNMENTS enhanced wizard ----------------
        if entity_l == "assignments":
            # Person dropdown with current weekly hours as badges
            loads = get_person_weekly_hours(conn)
            ppl = df_people(conn)
            if len(ppl) == 0:
                st.warning("No people available. Add a person first.")
            else:
                def person_label(i):
                    pid = int(ppl.iloc[i]["__id__"])
                    h = loads.get(pid, 0.0)
                    return f"{ppl.iloc[i]['__label__']}  ·  {h:g} h/w"
                idx_pe = st.selectbox("Person", list(range(len(ppl))), format_func=person_label, index=0)
                values["person_id"] = int(ppl.iloc[idx_pe]["__id__"]); skip_cols.add("person_id")

            st.markdown("**Section selection mode**")
            mode = st.radio("Choose how to pick a section:", ["Program → Course → Type → Section", "Direct Section"], horizontal=False, index=0)

            # helper: backfill P/C/T from a selected section
            def derive_pct_from_section(section_id: int):
                s_pk = pk_of(conn, "sections")
                q = f"""
                SELECT p.id AS program_id, c.id AS course_id, t.id AS type_id
                FROM sections s
                LEFT JOIN courses c ON s.course_id = c.id
                LEFT JOIN programs p ON c.program_id = p.id
                LEFT JOIN types t ON s.type_id = t.id
                WHERE s.{s_pk} = ?
                """
                try:
                    row = conn.execute(q, (int(section_id),)).fetchone()
                    if row:
                        return dict(program_id=row[0], course_id=row[1], type_id=row[2])
                except Exception:
                    pass
                return {}

            selected_program = None
            selected_course  = None
            selected_type    = None
            selected_section = None

            if mode.startswith("Program"):
                # Program -> Course -> Type -> Section (with occupancy hints)
                prog_df = df_programs(conn)
                if len(prog_df) == 0:
                    st.warning("No programs available. Add a program first.")
                else:
                    idx_p = st.selectbox("Program", list(range(len(prog_df))), format_func=lambda i: prog_df.iloc[i]["__label__"], index=0, key="ass_p")
                    selected_program = int(prog_df.iloc[idx_p]["__id__"])
                    crs_df = df_courses(conn, selected_program)
                    if len(crs_df) == 0:
                        st.warning("No courses under this program. Add a course first.")
                    else:
                        idx_c = st.selectbox("Course", list(range(len(crs_df))), format_func=lambda i: crs_df.iloc[i]["__label__"], index=0, key="ass_c")
                        selected_course = int(crs_df.iloc[idx_c]["__id__"])

                        # Occupancy note at course level (show all sections under course)
                        secs_course = course_sections_with_occupancy(conn, program_id=selected_program, course_id=selected_course, type_id=None)
                        occ_any = secs_course[secs_course["occupied_by"] != ""]
                        if not occ_any.empty:
                            with st.expander("⚠️ Occupied sections for this course"):
                                for _, r in occ_any.iterrows():
                                    st.write(f"- {r['__label__']} — assigned to: {r['occupied_by']}")

                        typ_df = df_types(conn)
                        if len(typ_df) == 0:
                            st.warning("No types available. Add a type first.")
                        else:
                            idx_t = st.selectbox("Type", list(range(len(typ_df))), format_func=lambda i: typ_df.iloc[i]["__label__"], index=0, key="ass_t")
                            selected_type = int(typ_df.iloc[idx_t]["__id__"])

                            secs = course_sections_with_occupancy(conn, program_id=selected_program, course_id=selected_course, type_id=selected_type)
                            if len(secs) == 0:
                                st.warning("No sections for this Program/Course/Type.")
                            else:
                                def sec_label(i):
                                    lab = secs.iloc[i]["__label__"]
                                    occ = secs.iloc[i]["occupied_by"]
                                    return f"{lab}  ·  ({'occupied: ' + occ if occ else 'free'})"
                                idx_s = st.selectbox("Section", list(range(len(secs))), format_func=sec_label, index=0, key="ass_s")
                                selected_section = int(secs.iloc[idx_s]["__id__"])
                                values["section_id"] = selected_section; skip_cols.add("section_id")

            else:
                # Direct Section -> backfill Program/Course/Type automatically
                secs = df_sections_with_label(conn)
                if len(secs) == 0:
                    st.warning("No sections available. Add a section first.")
                else:
                    def sec_label(i):
                        sid = int(secs.iloc[i]["__id__"])
                        occ = section_occupants(conn, sid)
                        tag = (", ".join([o[1] for o in occ])) if occ else "free"
                        return f"{secs.iloc[i]['__label__']}  ·  ({'occupied: ' + tag if occ else tag})"
                    idx_s = st.selectbox("Section", list(range(len(secs))), format_func=sec_label, index=0, key="ass_s_direct")
                    selected_section = int(secs.iloc[idx_s]["__id__"])
                    values["section_id"] = selected_section; skip_cols.add("section_id")

                    # Backfill P/C/T for clarity (not saved, just info)
                    back = derive_pct_from_section(selected_section)
                    if back:
                        selected_program = back.get("program_id")
                        selected_course  = back.get("course_id")
                        selected_type    = back.get("type_id")
                        st.info(f"Derived: Program={selected_program}, Course={selected_course}, Type={selected_type}")

            # Quick inline alert if chosen section already has assignees
            if selected_section is not None:
                occ = section_occupants(conn, selected_section)
                if occ:
                    names = ", ".join([o[1] for o in occ])
                    st.warning(f"Selected section is already assigned to: {names}")

            # Continue to render remaining non-FK fields (e.g., fraction) below
            # and then submit as usual.
            # Mark managed FKs:
            skip_cols.update({"person_id","section_id"})

        # ---------------- Sections guided picker (unchanged) ----------------
        elif entity_l == "sections":
            prog_df = df_programs(conn)
            if len(prog_df) == 0:
                st.warning("No programs available. Add a program first.")
            else:
                idx_p = st.selectbox("Program", list(range(len(prog_df))), format_func=lambda i: prog_df.iloc[i]["__label__"], index=0)
                sel_prog = int(prog_df.iloc[idx_p]["__id__"])
                crs_df = df_courses(conn, sel_prog)
                if len(crs_df) == 0:
                    st.warning("No courses under this program. Add a course first.")
                else:
                    idx_c = st.selectbox("Course", list(range(len(crs_df))), format_func=lambda i: crs_df.iloc[i]["__label__"], index=0)
                    sel_course = int(crs_df.iloc[idx_c]["__id__"])
                    typ_df = df_types(conn)
                    if len(typ_df) == 0:
                        st.warning("No types available. Add a type first.")
                    else:
                        idx_t = st.selectbox("Type", list(range(len(typ_df))), format_func=lambda i: typ_df.iloc[i]["__label__"], index=0)
                        sel_type = int(typ_df.iloc[idx_t]["__id__"])
                        values["course_id"] = sel_course
                        values["type_id"] = sel_type
                        skip_cols.update({"course_id","type_id"})

        # ---------------- Other entity-specific shortcuts ----------------
        elif entity_l in {"exclusions","section_preferences"}:
            # Person + Section via P->C->T chain
            ppl = df_people(conn)
            if len(ppl) == 0:
                st.warning("No people available. Add a person first.")
            else:
                idx_pe = st.selectbox("Person", list(range(len(ppl))), format_func=lambda i: ppl.iloc[i]["__label__"], index=0)
                values["person_id"] = int(ppl.iloc[idx_pe]["__id__"]); skip_cols.add("person_id")
                prog_df = df_programs(conn)
                if len(prog_df) == 0:
                    st.warning("No programs available. Add a program first.")
                else:
                    idx_p = st.selectbox("Program", list(range(len(prog_df))), format_func=lambda i: prog_df.iloc[i]["__label__"], index=0)
                    sel_prog = int(prog_df.iloc[idx_p]["__id__"])
                    crs_df = df_courses(conn, sel_prog)
                    if len(crs_df) == 0:
                        st.warning("No courses under this program. Add a course first.")
                    else:
                        idx_c = st.selectbox("Course", list(range(len(crs_df))), format_func=lambda i: crs_df.iloc[i]["__label__"], index=0)
                        sel_course = int(crs_df.iloc[idx_c]["__id__"])
                        typ_df = df_types(conn)
                        if len(typ_df) == 0:
                            st.warning("No types available. Add a type first.")
                        else:
                            idx_t = st.selectbox("Type", list(range(len(typ_df))), format_func=lambda i: typ_df.iloc[i]["__label__"], index=0)
                            sel_type = int(typ_df.iloc[idx_t]["__id__"])
                            sec_df = df_sections_with_label(conn, program_id=sel_prog, course_id=sel_course, type_id=sel_type)
                            if len(sec_df) == 0:
                                st.warning("No sections for this Program/Course/Type.")
                            else:
                                idx_s = st.selectbox("Section", list(range(len(sec_df))), format_func=lambda i: sec_df.iloc[i]["__label__"], index=0)
                                values["section_id"] = int(sec_df.iloc[idx_s]["__id__"]); skip_cols.add("section_id")

        elif entity_l == "people_availability":
            ppl = df_people(conn)
            if len(ppl) == 0:
                st.warning("No people available. Add a person first.")
            else:
                idx_pe = st.selectbox("Person", list(range(len(ppl))), format_func=lambda i: ppl.iloc[i]["__label__"], index=0)
                values["person_id"] = int(ppl.iloc[idx_pe]["__id__"]); skip_cols.add("person_id")
                tsl = df_timeslots(conn)
                if len(tsl) == 0:
                    st.warning("No timeslots available. Add timeslots first.")
                else:
                    idx_t = st.selectbox("Timeslot", list(range(len(tsl))), format_func=lambda i: tsl.iloc[i]["__label__"], index=0)
                    values["timeslot_id"] = int(tsl.iloc[idx_t]["__id__"]); skip_cols.add("timeslot_id")

        elif entity_l == "schedule":
            prog_df = df_programs(conn)
            if len(prog_df) == 0:
                st.warning("No programs available. Add a program first.")
            else:
                idx_p = st.selectbox("Program", list(range(len(prog_df))), format_func=lambda i: prog_df.iloc[i]["__label__"], index=0)
                sel_prog = int(prog_df.iloc[idx_p]["__id__"])
                crs_df = df_courses(conn, sel_prog)
                if len(crs_df) == 0:
                    st.warning("No courses under this program. Add a course first.")
                else:
                    idx_c = st.selectbox("Course", list(range(len(crs_df))), format_func=lambda i: crs_df.iloc[i]["__label__"], index=0)
                    sel_course = int(crs_df.iloc[idx_c]["__id__"])
                    typ_df = df_types(conn)
                    if len(typ_df) == 0:
                        st.warning("No types available. Add a type first.")
                    else:
                        idx_t = st.selectbox("Type", list(range(len(typ_df))), format_func=lambda i: typ_df.iloc[i]["__label__"], index=0)
                        sel_type = int(typ_df.iloc[idx_t]["__id__"])
                        sec_df = df_sections_with_label(conn, program_id=sel_prog, course_id=sel_course, type_id=sel_type)
                        if len(sec_df) == 0:
                            st.warning("No sections for this Program/Course/Type.")
                        else:
                            idx_s = st.selectbox("Section", list(range(len(sec_df))), format_func=lambda i: sec_df.iloc[i]["__label__"], index=0)
                            values["section_id"] = int(sec_df.iloc[idx_s]["__id__"]); skip_cols.add("section_id")
                            tsl = df_timeslots(conn)
                            if len(tsl) == 0:
                                st.warning("No timeslots available. Add timeslots first.")
                            else:
                                idx_tt = st.selectbox("Timeslot", list(range(len(tsl))), format_func=lambda i: tsl.iloc[i]["__label__"], index=0)
                                values["timeslot_id"] = int(tsl.iloc[idx_tt]["__id__"]); skip_cols.add("timeslot_id")

        elif entity_l == "section_required_qualifications":
            prog_df = df_programs(conn)
            if len(prog_df) == 0:
                st.warning("No programs available. Add a program first.")
            else:
                idx_p = st.selectbox("Program", list(range(len(prog_df))), format_func=lambda i: prog_df.iloc[i]["__label__"], index=0)
                sel_prog = int(prog_df.iloc[idx_p]["__id__"])
                crs_df = df_courses(conn, sel_prog)
                if len(crs_df) == 0:
                    st.warning("No courses under this program. Add a course first.")
                else:
                    idx_c = st.selectbox("Course", list(range(len(crs_df))), format_func=lambda i: crs_df.iloc[i]["__label__"], index=0)
                    sel_course = int(crs_df.iloc[idx_c]["__id__"])
                    typ_df = df_types(conn)
                    if len(typ_df) == 0:
                        st.warning("No types available. Add a type first.")
                    else:
                        idx_t = st.selectbox("Type", list(range(len(typ_df))), format_func=lambda i: typ_df.iloc[i]["__label__"], index=0)
                        sel_type = int(typ_df.iloc[idx_t]["__id__"])
                        sec_df = df_sections_with_label(conn, program_id=sel_prog, course_id=sel_course, type_id=sel_type)
                        if len(sec_df) == 0:
                            st.warning("No sections for this Program/Course/Type.")
                        else:
                            idx_s = st.selectbox("Section", list(range(len(sec_df))), format_func=lambda i: sec_df.iloc[i]["__label__"], index=0)
                            values["section_id"] = int(sec_df.iloc[idx_s]["__id__"]); skip_cols.add("section_id")
                            qd = df_qualifications(conn)
                            if len(qd) == 0:
                                st.warning("No qualifications available. Add qualifications first.")
                            else:
                                idx_q = st.selectbox("Qualification", list(range(len(qd))), format_func=lambda i: qd.iloc[i]["__label__"], index=0)
                                values["qualification_id"] = int(qd.iloc[idx_q]["__id__"]); skip_cols.add("qualification_id")

        # ---------- Render remaining inputs (generic for any table) ----------
        for name, typ, notnull in form_cols:
            if name in skip_cols:
                continue
            if name in fk_by_local:
                parent = fk_by_local[name]["table"]
                parent_pk = fk_by_local[name]["to"]
                label_col = label_column_guess(conn, parent)
                opts = fk_options(conn, parent, parent_pk, label_col)
                if opts.empty:
                    st.warning(f"No options in parent table '{parent}'. Cannot set foreign key '{name}'.")
                    sel = None
                else:
                    sel_label = st.selectbox(
                        f"{name} → {parent}",
                        list(range(len(opts))),
                        format_func=lambda i: f"{opts.iloc[i]['__label__']}  ·  [{opts.iloc[i]['__id__']}]",
                        index=0
                    )
                    sel = int(opts.iloc[sel_label]["__id__"])
                values[name] = sel
            else:
                if is_integer_affinity(typ):
                    v = st.number_input(name, value=0, step=1, format="%d")
                    values[name] = int(v)
                elif "REAL" in typ.upper() or "FLOA" in typ.upper() or "DOUB" in typ.upper():
                    v = st.number_input(name, value=0.0, step=0.1)
                    values[name] = float(v)
                else:
                    v = st.text_input(name, value="" if notnull else "", placeholder=f"{typ}")
                    values[name] = (v if v != "" else None)

        submitted = st.form_submit_button("➕ Add row", type="primary")

    if submitted:
        try:
            if id_col:
                if is_integer_affinity(types.get(id_col, "")):
                    values[id_col] = next_int_id(conn, entity, id_col)
                else:
                    try:
                        values[id_col] = str(next_int_id(conn, entity, id_col))
                    except Exception:
                        n = conn.execute(f"SELECT COUNT(*) FROM {entity};").fetchone()[0]
                        values[id_col] = str(n + 1)
            with transaction(conn) as cur:
                insert_row(conn, entity, values)
            st.success(f"Row added to '{entity}'."); st.rerun()
        except Exception as e:
            st.error(f"Insert failed: {e}")

# ---------------- Delete ----------------
elif action == "Delete":
    has_rowid = table_has_rowid(conn, entity)
    if not has_rowid:
        st.warning("This table has no ROWID; use a key-based delete flow (not implemented).")
    else:
        df = read_table(conn, entity, include_rowid=True)
        if df.empty:
            st.info("Table is empty.")
        else:
            from core.model import friendlify_df
            display_df = friendlify_df(conn, entity, df)
            if not show_raw_ids:
                for f in fk_map(conn, entity):
                    col = f["from"]
                    if col in display_df.columns:
                        display_df.drop(columns=[col], inplace=True)
            display_df["__delete__"] = False

            # Hide first column via CSS
            st.markdown("""
            <style>
            [data-testid="stDataEditor"] thead tr th:first-child,
            [data-testid="stDataEditor"] tbody tr td:first-child { display: none !important; }
            </style>
            """, unsafe_allow_html=True)

            cols_order = ["__delete__"] + [c for c in display_df.columns if c != "__delete__"]
            edited = st.data_editor(
                display_df[cols_order], width='stretch', hide_index=True,
                column_config={
                    "__rowid__": st.column_config.NumberColumn("__rowid__", help="Internal ROWID", disabled=True),
                    "__delete__": st.column_config.CheckboxColumn("Delete?"),
                },
            )
            to_del = edited.loc[edited.get("__delete__", False) == True, "__rowid__"].dropna().astype(int).tolist()
            if to_del:
                schema = table_schema(conn, entity)
                pk_cols = [r["name"] for _, r in schema.iterrows() if int(r["pk"]) == 1]
                key_col = pk_cols[0] if pk_cols else None
                ids = df.loc[df["__rowid__"].isin(to_del), key_col].tolist() if key_col and key_col in df.columns else []
                impacts = count_impacts(conn, entity, key_col, ids) if (ids and key_col) else []
                with st.expander("🔎 Deletion impact preview", expanded=True):
                    if not ids or not impacts:
                        st.write("No referencing rows detected, or unable to determine PK; deletion should be local.")
                    else:
                        st.write("Child tables referencing these rows:")
                        for child, n in impacts:
                            st.write(f"- **{child}**: {n} row(s)")
                col1, col2 = st.columns([1,1])
                with col1:
                    confirm = st.text_input("Type DELETE to confirm:", value="")
                with col2:
                    if st.button("🗑️ Delete selected"):
                        if confirm.strip() != "DELETE":
                            st.error("Confirmation text does not match. Type DELETE to proceed.")
                        else:
                            try:
                                with transaction(conn) as cur:
                                    delete_by_rowid(conn, entity, to_del)
                                st.success(f"Deleted {len(to_del)} row(s)."); st.rerun()
                            except Exception as e:
                                st.error(f"Delete failed: {e}")
            else:
                st.info("Tick rows to delete.")

# ---------------- Export / Import ----------------
elif action == "Export CSV":
    df = read_table(conn, entity, include_rowid=False)
    st.download_button(f"⬇️ Download {entity}.csv", df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{entity}.csv", mime="text/csv")
elif action == "Import CSV":
    up = st.file_uploader("Upload CSV to append into this table", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            with transaction(conn) as cur:
                df.to_sql(entity, conn, if_exists="append", index=False)
            st.success(f"Imported {len(df)} rows into '{entity}'.")
        except Exception as e:
            st.error(f"Import failed: {e}")

