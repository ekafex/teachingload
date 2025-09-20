# app/pages/4_Calendar.py
# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

from core.io import get_connection, transaction
from core.model import (
    table_schema,
    df_programs, df_courses, df_types, df_sections_with_label, df_people,
    col_exists,
    assignments_table,
)

st.set_page_config(page_title="Calendar", page_icon="📆", layout="wide")

# =========================
# Weekday normalization
# =========================
AL_ORDER = ["Hene", "Marte", "Merkure", "Enjte", "Premte", "Shtune", "Diel"]
AL_RANK = {w: i for i, w in enumerate(AL_ORDER)}
_WD_ALIASES = {
    # English
    "mon":"Hene", "monday":"Hene",
    "tue":"Marte", "tues":"Marte", "tuesday":"Marte",
    "wed":"Merkure", "wednesday":"Merkure",
    "thu":"Enjte", "thur":"Enjte", "thurs":"Enjte", "thursday":"Enjte",
    "fri":"Premte", "friday":"Premte",
    "sat":"Shtune", "saturday":"Shtune",
    "sun":"Diel", "sunday":"Diel",
    # Albanian spellings/phrases
    "hene":"Hene", "hënë":"Hene", "e hene":"Hene", "e hënë":"Hene",
    "marte":"Marte", "martë":"Marte", "e marte":"Marte", "e martë":"Marte",
    "merkure":"Merkure", "mërkurë":"Merkure", "e merkure":"Merkure", "e mërkurë":"Merkure",
    "enjte":"Enjte", "e enjte":"Enjte",
    "premte":"Premte", "e premte":"Premte",
    "shtune":"Shtune", "shtunë":"Shtune", "e shtune":"Shtune", "e shtunë":"Shtune",
    "diel":"Diel", "e diel":"Diel",
    # numeric strings
    "1":"Hene", "2":"Marte", "3":"Merkure", "4":"Enjte", "5":"Premte", "6":"Shtune", "7":"Diel",
}
def _lc(s): return "" if s is None else str(s).strip().lower()
def canonical_weekday(s) -> str:
    k = _lc(s)
    if k in _WD_ALIASES: return _WD_ALIASES[k]
    if len(k) >= 3 and k[:3] in _WD_ALIASES: return _WD_ALIASES[k[:3]]
    raw = str(s).strip()
    return raw if raw in AL_ORDER else raw
def weekday_sort_key(s): 
    c = canonical_weekday(s)
    return (AL_RANK.get(c, 999), c)

# =========================
# DB utilities
# =========================
def get_conn_from_state():
    db = st.session_state.get("work_db_path")
    if not db:
        st.warning("No working DB selected. Open the main page to create/select one.")
        st.stop()
    return get_connection(db)

def has_table(conn, name: str) -> bool:
    try:
        return bool(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND lower(name)=lower(?)", (name,)
        ).fetchone())
    except Exception:
        return False

def pk_of(conn, table: str) -> str:
    info = table_schema(conn, table)
    pks = [str(r["name"]) for _, r in info.iterrows() if int(r["pk"]) == 1]
    if pks: return pks[0]
    names = [str(n) for n in info["name"].tolist()]
    if "id" in names: return "id"
    legacy = table.rstrip("s") + "_id"
    return legacy if legacy in names else names[0]

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

# Normalized pairs (weekday, slot) for UI & lookups
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

    # fallback: synthesize Mon–Fri
    synth = pd.DataFrame({"day_id":[1,2,3,4,5], "day":["Monday","Tuesday","Wednesday","Thursday","Friday"]})
    synth["key"]=1; slots["key"]=1
    pairs = synth.merge(slots, on="key").drop(columns=["key"])
    pairs["weekday_label"] = pairs["day"].apply(canonical_weekday)
    pairs["pair_id"] = pairs["day_id"].astype(str) + "-" + pairs["ts_id"].astype(str)
    return pairs[["pair_id","day_id","ts_id","weekday_label","slot"]]

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
    disp = ["(None)" if (v is None or str(v).strip()=="") else str(v) for v in vals]
    return ["(All)"] + (sorted(set(disp), key=lambda x:(x=="(None)", x)) if disp else [])

def occupants(conn, tbl_assign: str, section_id: int) -> list[str]:
    rows = conn.execute(f"""
        SELECT COALESCE(pe.name, 'Person '||a.person_id) AS name,
               COALESCE(a.fraction, 1.0) AS frac
        FROM {tbl_assign} a
        LEFT JOIN people pe ON pe.id = a.person_id
        WHERE a.section_id = ?
        ORDER BY name
    """, (int(section_id),)).fetchall()
    out = []
    for nm, fr in rows:
        out.append(f"{nm} (x{float(fr):g})" if float(fr) != 1.0 else str(nm))
    return out

def label_for_section(conn, section_id: int) -> str:
    PROG_LBL, COURSE_LBL, TYPE_LBL = choose_labels(conn)
    row = conn.execute(f"""
        SELECT ({PROG_LBL} || ' · ' || {COURSE_LBL} || ' · ' || {TYPE_LBL}
                || CASE WHEN s.split IS NOT NULL AND TRIM(s.split)<>'' THEN (' · Split '||s.split) ELSE '' END
               ) AS lbl
        FROM sections s
        JOIN courses  c ON c.id = s.course_id
        JOIN programs p ON p.id = c.program_id
        JOIN types    t ON t.id = s.type_id
        WHERE s.id = ?
    """, (int(section_id),)).fetchone()
    return row[0] if row and row[0] else f"Section {section_id}"

# =========================
# Page
# =========================
conn = get_conn_from_state()
tbl_assign = assignments_table(conn)
flags = schema_flags(conn)

st.title("📆 Calendar (Interactive)")

# Filters controlling the board and the “New entry” picker
colF1, colF2 = st.columns([1,1])
with colF1:
    progs = df_programs(conn)
    if progs.empty:
        st.warning("No programs defined."); st.stop()
    sel_prog_label = st.selectbox("Program", progs["__label__"].tolist(), index=0, key="cal_prog")
    sel_prog_id = int(progs.loc[progs["__label__"] == sel_prog_label, "__id__"].iloc[0])

with colF2:
    # Show separate semester choice so the board can be filtered
    sems = pd.read_sql_query("""
        SELECT DISTINCT s.semester
        FROM sections s
        JOIN courses c ON c.id = s.course_id
        WHERE c.program_id = ?
        ORDER BY s.semester
    """, conn, params=(sel_prog_id,))
    sem_vals = sems["semester"].tolist()
    sem_labels = ["(Të gjitha)"] + [str(v) for v in sem_vals]
    sel_sem_label = st.selectbox("Semester", sem_labels, index=0, key="cal_sem")
    sel_sem_raw = None if sel_sem_label=="(Të gjitha)" else sem_vals[sem_labels.index(sel_sem_label)-1]

# Build axes from normalized pairs
pairs_all = df_timeslot_pairs(conn)
pairs_all["weekday_label"] = pairs_all["weekday_label"].apply(canonical_weekday)
# columns -> canonical order if present, otherwise whatever exists
col_order = [w for w in AL_ORDER if w in pairs_all["weekday_label"].unique().tolist()]
if not col_order:
    col_order = sorted(pairs_all["weekday_label"].dropna().unique().tolist(), key=weekday_sort_key)
# rows -> slot labels (normalized spacing)
def norm_slot(x): return str(x).replace(" ", "").strip() if x is not None else ""
slot_order = sorted(pairs_all["slot"].dropna().astype(str).map(norm_slot).unique().tolist())

# Map (weekday, slot) -> ids for scheduling
#  - Schema A: -> (day_id, ts_id)
#  - Schema B: -> timeslot_id (encoded via pair_id == ts_id)
pair_lookup = {}
for _, r in pairs_all.iterrows():
    wd = canonical_weekday(r["weekday_label"])
    sl = norm_slot(r["slot"])
    if flags["schedule_has_day"]:
        pair_lookup[(wd, sl)] = ("A", int(r["day_id"]), int(r["ts_id"]))
    else:
        pair_lookup[(wd, sl)] = ("B", None, int(r["ts_id"]))

# Current scheduled items for the board (with labels and instructors)
PROG_LBL, COURSE_LBL, TYPE_LBL = choose_labels(conn)
if flags["schedule_has_day"]:
    qry_board = f"""
        SELECT
          s.id AS section_id,
          w.day   AS weekday,
          tl.slot AS slot,
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
        { " AND s.semester = ?" if sel_sem_raw is not None else "" }
        ORDER BY w.id, tl.slot, section_label;
    """
else:
    qry_board = f"""
        SELECT
          s.id AS section_id,
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
        { " AND s.semester = ?" if sel_sem_raw is not None else "" }
        ORDER BY tl.weekday, tl.slot, section_label;
    """

params = [sel_prog_id] + ([sel_sem_raw] if sel_sem_raw is not None else [])
board = pd.read_sql_query(qry_board, conn, params=tuple(params))
if not board.empty:
    board["weekday_label"] = board["weekday"].apply(canonical_weekday)
    board["slot_norm"] = board["slot"].map(norm_slot)

# precompute instructors per section
inst = pd.read_sql_query(f"""
    SELECT a.section_id, GROUP_CONCAT(pe.name, ', ') AS instructors
    FROM {tbl_assign} a
    LEFT JOIN people pe ON pe.id = a.person_id
    GROUP BY a.section_id
""", conn)
inst_map = {int(r["section_id"]): (r["instructors"] or "") for _, r in inst.iterrows()}

# =========================
# Interactive Grid (buttons)
# =========================
st.subheader("Javore (Mon–Sun) × Kohë")
grid_cols = len(col_order)
for slot in slot_order:
    row_cols = st.columns(grid_cols)
    for j, wd in enumerate(col_order):
        key = f"cell_{wd}_{slot}"
        # Compose cell content
        cell_items = []
        if not board.empty:
            sub = board[(board["weekday_label"] == wd) & (board["slot_norm"] == slot)]
            for _, rr in sub.iterrows():
                names = inst_map.get(int(rr["section_id"]), "")
                label = rr["section_label"] + (f" · {names}" if names else "")
                cell_items.append(label)
        display = "—" if not cell_items else "\n".join(cell_items)
        # prefer green if any instructor else red
        state = "red"
        if not board.empty:
            sub2 = board[(board["weekday_label"] == wd) & (board["slot_norm"] == slot)]
            has_inst = any(bool(inst_map.get(int(rid), "")) for rid in sub2["section_id"].tolist())
            state = "green" if has_inst else ("red" if len(sub2) else "empty")
        bg = "#ffffff"
        if state == "green": bg = "#d9f5d3"
        elif state == "red": bg = "#f9d2d0"
        # Render as button with styled label
        html = f"""
        <div style="white-space:pre-line;min-height:48px;border:1px solid #e5e7eb;background:{bg};
                    border-radius:10px;padding:8px;font-size:12px;">
            <b>{wd}</b> · <i>{slot}</i>
            <div>{display}</div>
        </div>
        """
        if row_cols[j].button("", key=key, help=f"{wd} · {slot}"):
            st.session_state["cal_selected_cell"] = (wd, slot)
        row_cols[j].markdown(html, unsafe_allow_html=True)

st.markdown("---")

# =========================
# Inline form after clicking a cell
# =========================
sel_cell = st.session_state.get("cal_selected_cell")
left, right = st.columns([1.4, 1.0])

with left:
    st.subheader("➕ Add / Update at selected cell")
    if not sel_cell:
        st.info("Click any cell above to add a schedule/assignment there.")
    else:
        wd, sl = sel_cell
        st.success(f"Selected: {wd} · {sl}")
        mapping = pair_lookup.get((wd, sl))
        if not mapping:
            st.error("This day/time is not recognized in the current pairs mapping."); st.stop()
        mode_schema, day_id, ts_id = mapping
        # --- Section picker with Split ---
        choose_mode = st.radio(
            "Section selection mode",
            ["Program → Course → Type → Split → Section", "Direct Section"],
            horizontal=True, key="cal_pick_mode"
        )
        selected_section = None

        if choose_mode.startswith("Program"):
            progs2 = df_programs(conn)
            if progs2.empty:
                st.warning("No programs."); st.stop()
            idx_p = st.selectbox("Program", list(range(len(progs2))),
                                 format_func=lambda i: progs2.iloc[i]["__label__"], key="cal_prog_pick")
            sel_prog2 = int(progs2.iloc[idx_p]["__id__"])

            courses2 = df_courses(conn, sel_prog2)
            if courses2.empty:
                st.warning("No courses for this program.")
            else:
                idx_c = st.selectbox("Course", list(range(len(courses2))),
                                     format_func=lambda i: courses2.iloc[i]["__label__"], key="cal_course_pick")
                sel_course2 = int(courses2.iloc[idx_c]["__id__"])

                types2 = df_types(conn)
                if types2.empty:
                    st.warning("No types.")
                else:
                    idx_t = st.selectbox("Type", list(range(len(types2))),
                                         format_func=lambda i: types2.iloc[i]["__label__"], key="cal_type_pick")
                    sel_type2 = int(types2.iloc[idx_t]["__id__"])

                    split_opts = distinct_splits(conn, sel_prog2, sel_course2, sel_type2)
                    split_choice = st.selectbox("Split", split_opts, index=0, key="cal_split_pick")

                    secs = df_sections_with_label(conn, program_id=sel_prog2, course_id=sel_course2, type_id=sel_type2)
                    if split_choice != "(All)":
                        if split_choice == "(None)":
                            secs = secs[(secs["split"].isna()) | (secs["split"].astype(str).str.strip()=="")]
                        else:
                            secs = secs[secs["split"].astype(str) == split_choice]
                    if secs.empty:
                        st.warning("No sections match Program/Course/Type/Split.")
                    else:
                        def sec_lab(i):
                            sid = int(secs.iloc[i]["__id__"])
                            occ = occupants(conn, tbl_assign, sid)
                            tag = ", ".join(occ) if occ else "free"
                            return f"{secs.iloc[i]['__label__']} · ({tag})"
                        idx_s = st.selectbox("Section", list(range(len(secs))),
                                             format_func=sec_lab, key="cal_section_pick")
                        selected_section = int(secs.iloc[idx_s]["__id__"])
        else:
            # Direct + optional Split filter
            secs = df_sections_with_label(conn)
            if secs.empty:
                st.warning("No sections."); 
            else:
                raw_splits = secs["split"].fillna("").astype(str).map(str.strip)
                has_none = (raw_splits=="").any()
                uniq = sorted([v for v in raw_splits.unique() if v!=""])
                split_opts = ["(All)"] + (["(None)"] if has_none else []) + uniq
                split_choice = st.selectbox("Filter by Split (optional)", split_opts, index=0, key="cal_split_direct")
                if split_choice != "(All)":
                    if split_choice == "(None)":
                        secs = secs[(secs["split"].isna()) | (secs["split"].astype(str).str.strip()=="")]
                    else:
                        secs = secs[secs["split"].astype(str) == split_choice]
                def sec_lab(i):
                    sid = int(secs.iloc[i]["__id__"])
                    occ = occupants(conn, tbl_assign, sid)
                    tag = ", ".join(occ) if occ else "free"
                    return f"{secs.iloc[i]['__label__']} · ({tag})"
                idx_s = st.selectbox("Section", list(range(len(secs))),
                                     format_func=sec_lab, key="cal_section_direct")
                selected_section = int(secs.iloc[idx_s]["__id__"])

        # --- Optional instructor ---
        ppl = df_people(conn)
        inst_names = ["(None)"] + (ppl["__label__"].tolist() if not ppl.empty else [])
        sel_inst_label = st.selectbox("Instructor (optional)", inst_names, index=0, key="cal_inst")
        sel_inst_id = None
        if sel_inst_label != "(None)" and not ppl.empty:
            sel_inst_id = int(ppl.loc[ppl["__label__"] == sel_inst_label, "__id__"].iloc[0])
        fraction = st.number_input("Fraction", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="cal_fraction")

        # --- Save button ---
        can_save = selected_section is not None and ((ts_id is not None) and ((day_id is not None) if flags["schedule_has_day"] else True))
        if st.button("💾 Save to this cell", type="primary", disabled=not can_save, key="cal_save_btn"):
            try:
                with transaction(conn) as cur:
                    if flags["schedule_has_day"]:
                        # prevent duplicates
                        dup = conn.execute(
                            "SELECT 1 FROM schedule WHERE section_id=? AND day_id=? AND timeslot_id=? LIMIT 1;",
                            (selected_section, day_id, ts_id)
                        ).fetchone()
                        if not dup:
                            cur.execute(
                                "INSERT INTO schedule(section_id, day_id, timeslot_id) VALUES (?, ?, ?);",
                                (int(selected_section), int(day_id), int(ts_id))
                            )
                    else:
                        dup = conn.execute(
                            "SELECT 1 FROM schedule WHERE section_id=? AND timeslot_id=? LIMIT 1;",
                            (selected_section, ts_id)
                        ).fetchone()
                        if not dup:
                            cur.execute(
                                "INSERT INTO schedule(section_id, timeslot_id) VALUES (?, ?);",
                                (int(selected_section), int(ts_id))
                            )
                    # optional assignment
                    if sel_inst_id is not None:
                        # avoid exact duplicate for same section/person
                        dup2 = conn.execute(
                            f"SELECT 1 FROM {tbl_assign} WHERE person_id=? AND section_id=? LIMIT 1;",
                            (sel_inst_id, selected_section)
                        ).fetchone()
                        if not dup2:
                            cur.execute(
                                f"INSERT INTO {tbl_assign}(person_id, section_id, fraction) VALUES (?, ?, ?);",
                                (int(sel_inst_id), int(selected_section), float(fraction))
                            )
                st.success("Saved. ✅")
                st.balloons()
                # refresh page data
                st.session_state.pop("cal_selected_cell", None)
                st.rerun()
            except Exception as e:
                st.error(f"Save failed: {e}")

with right:
    st.subheader("ℹ️ Context")
    if sel_cell:
        wd, sl = sel_cell
        st.write(f"Selected cell: **{wd} · {sl}**")
    st.markdown("**Legend**")
    st.markdown(
        "<div style='display:flex;gap:12px;align-items:center;'>"
        "<div style='width:14px;height:14px;background:#d9f5d3;border:1px solid #b7e8af;'></div>"
        "<span>Scheduled + instructor(s)</span>"
        "<div style='width:14px;height:14px;background:#f9d2d0;border:1px solid #f0a8a3;margin-left:14px;'></div>"
        "<span>Scheduled but unassigned</span>"
        "</div>",
        unsafe_allow_html=True
    )

# Optional: compact table of what’s scheduled now (for quick scan)
with st.expander("📋 Current schedule (read-only)"):
    if board.empty:
        st.info("No entries.")
    else:
        show = board.copy()
        show["weekday"] = show["weekday_label"]
        show["slot"] = show["slot_norm"]
        show = show[["weekday","slot","section_label"]].sort_values(
            by=["weekday","slot","section_label"], key=lambda s: s.map(lambda x: weekday_sort_key(x) if s.name=="weekday" else x)
        )
        st.dataframe(show, width="stretch", hide_index=True)

