# Teaching Load Management App (teachingload)

A Streamlit-based web application for managing **programs, courses, sections, schedules, and teaching loads** in a university department.  
It provides a simple interface for non-technical staff to edit the teaching database, assign instructors, schedule classes, and review teaching loads.


## Features

- **Database editing** (Programs, Courses, Sections, People, etc.)
- **Assignments**: link instructors ↔ sections, with fractional loads
- **Scheduling**: timetable slots (Program → Course → Type → Split → Section)
- **Interactive Calendar**: click cells to assign both schedule + instructor
- **Results Dashboard**:
  - Read-only Calendar view (per Program + Semester)
  - Instructor weekly load (table + bar chart)
- **CSV import/export** for backups or bulk editing



## Requirements

- **Python 3.11** or newer (works on Windows, Linux, macOS)  
- Required libraries are listed in [`requirements.txt`](requirements.txt):

```txt
streamlit>=1.35
pandas>=2.2
numpy>=1.26
sqlalchemy>=2.0
altair>=5.0
```


## Installation

1. **Install Python**

   * Download Python (downlod directly from [python.org](https://www.python.org/downloads/ or from Anaconda etc.)
   * If manual installation, use the **64-bit installer**, and tick **“Add Python to PATH”**, and install.
   * Verify in Command Prompt:

     ```bat
     python --version
     ```

2. **Get the app**

   * Clone this repo or download the `teachload` folder.
   * Example in windows location: `C:\Users\YourName\Documents\teachload`.

3. **Install requirements**
   Open Command Prompt, navigate into the app folder, and install dependencies:

   ```bat
   cd C:\Users\YourName\Documents\teachload\app
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```bat
   streamlit run app.py
   ```

   Your browser will open [http://localhost:8501](http://localhost:8501).



## Usage Workflow

1. **DB Edit** → Add or update Programs, Courses, Sections, People.
2. **Assignment** → Assign instructors to sections (with fractions).
3. **Schedule** → Place sections into timeslots.
4. **Calendar** → Interactive weekly grid for quick scheduling.
5. **Results** →

   * Calendar overview per Program + Semester
   * Instructor loads (table + bar chart)

> Pages 5️⃣ Constraints and 6️⃣ Optimize are placeholders for future features.



## Database Preparation

The app works with an SQLite database called `teaching_load.db`.
A template DB is provided, but you can also create your own.

### Main tables

* **programs** → degree programs (e.g., *Bachelor Physics*, *Master Physics*)
* **courses** → courses linked to programs (`course_id`, `program_id`, `title`/`name`, `semester`, `total_hours`, `exam_type`)
* **types** → type of teaching (e.g., *Lecture*, *Lab*, *Seminar*)
* **sections** → sub-groups of a course (`split` = group number/name)
* **people** → instructors
* **assignments** → links people ↔ sections (with `fraction`)
* **timeslots / weekdays** → defines available slots (day × time)
* **schedule** → links sections ↔ timeslots (and `day_id` if separate weekdays table is used)

### Starting from CSVs

You can prepare simple CSVs in Excel and import them via **DB Edit → Import CSV**.
Examples:

#### `programs.csv`

```csv
id,name
1,Bachelor Physics
2,Master Physics
```

#### `courses.csv`

```csv
id,program_id,name,semester,total_hours,exam_type
1,1,Mechanics,1,60,Written
2,1,Electromagnetism,2,60,Oral
```

#### `types.csv`

```csv
id,name
1,Lecture
2,Laboratory
```

#### `sections.csv`

```csv
id,course_id,type_id,semester,split,hours
1,1,1,1,A,2
2,1,2,1,A,2
3,1,2,1,B,2
```

#### `people.csv`

```csv
id,name
1,Dr. Smith
2,Dr. Jones
```

Once these basics are in place, you can assign instructors, schedule timeslots, and review loads.



## Screenshots

*(to be added)*

* **Assignments page**
* **Schedule overview**
* **Interactive calendar**
* **Results dashboard**


##  Notes

* The app always works on a *working copy* of the DB (`teaching_load.db`), so you can experiment safely.
* Export/import CSVs regularly to keep backups.
* Works fully offline — no external server required.


## Contributing

Pull requests and suggestions are welcome! Please open an issue to discuss major changes.


## License

MIT License © 2025 Your Department
