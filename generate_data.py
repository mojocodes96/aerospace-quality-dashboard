"""
generate_data.py
================
Phase 1: Generate realistic mock data for an aerospace hardware
Quality Analytics dashboard.

LEARNING NOTES
--------------
This script teaches you three things:
  1. How to use Python's `random` and `numpy` libraries to create
     realistic-looking data (not just random noise).
  2. How to build Python dictionaries and lists — the core data
     structures you'll use constantly in Pandas later.
  3. How to write that data into a SQLite database using Python's
     built-in `sqlite3` module.

Run this script once to create the file: aerospace_quality.db
Every other script in this project reads from that file.
"""

import sqlite3          # Built into Python — no install needed. Lets us create/query a database file.
import random           # Built into Python — generates random numbers and picks random items from lists.
import numpy as np      # NumPy: the foundation of scientific Python. We use it for realistic number distributions.
from datetime import datetime, timedelta  # Built into Python — for working with dates.
import os

# =============================================================================
# SECTION 1: CONSTANTS — the "universe" our fake factory lives in
# =============================================================================
# These lists define the domain. In a real system these would come from your
# ERP/MES master data tables. Here we hard-code them to keep things simple.

PART_NUMBERS = [
    "AS-7740-FRAME",      # Airframe structural bracket
    "AS-7741-FRAME",      # Airframe rib assembly
    "GD-2201-SERVO",      # Guidance servo actuator
    "GD-2202-SERVO",      # Guidance servo housing
    "PR-5501-NOZZLE",     # Propulsion nozzle insert
    "PR-5502-NOZZLE",     # Propulsion nozzle throat
    "AV-3301-PCB",        # Avionics control board
    "AV-3302-PCB",        # Avionics power distribution board
    "ST-8801-FASTENER",   # Structural titanium fastener (high volume)
    "ST-8802-FASTENER",   # Structural fastener, flanged
]

PRODUCTION_LINES = ["Line-Alpha", "Line-Bravo", "Line-Charlie", "Line-Delta"]

# Shift names reflect real aerospace manufacturing schedules
SHIFTS = ["Day", "Swing", "Night"]

# Operator IDs — anonymized as is typical in real systems
OPERATOR_IDS = [f"OP-{str(i).zfill(3)}" for i in range(1, 26)]
# ^ LEARNING NOTE: f-strings let you embed variables inside strings using {}.
#   str(i).zfill(3) pads a number with leading zeros: 1 → "001", 12 → "012"
#   So this creates: ["OP-001", "OP-002", ..., "OP-025"]

SUPPLIERS = [
    ("SUP-001", "Apex Precision Machining",    "Tier-1", "USA",     True),
    ("SUP-002", "Orbital Composites Ltd",       "Tier-1", "USA",     True),
    ("SUP-003", "StratoMet Components",         "Tier-2", "UK",      True),
    ("SUP-004", "Helios Fastener Systems",      "Tier-2", "Germany", True),
    ("SUP-005", "NovaTech Avionics Supply",     "Tier-1", "USA",     True),
    ("SUP-006", "Pacific Rim Castings",         "Tier-2", "Japan",   True),
    ("SUP-007", "Delta Forge Industries",       "Tier-2", "USA",     True),
    ("SUP-008", "Meridian Surface Treatments",  "Tier-3", "Mexico",  False),  # Not approved — creates interesting data!
    ("SUP-009", "Quantum Seal Technologies",    "Tier-2", "USA",     True),
    ("SUP-010", "Atlas Substrate Corp",         "Tier-3", "India",   True),
]
# ^ Each tuple is: (id, name, tier, country, approved_status)
#   A tuple is like a list but immutable (can't be changed after creation).
#   We use tuples here because this data is fixed — it won't change.

INSPECTION_TYPES = ["IQC", "In-Process", "Final", "FAI", "Receiving"]
# IQC = Incoming Quality Control (parts arriving from supplier)
# FAI = First Article Inspection (first part off a new production run)
# These are real aerospace quality terms the hiring manager will recognize!

DEFECT_TYPES = [
    "Dimensional Out-of-Tolerance",
    "Surface Finish Non-Conformance",
    "Porosity / Void",
    "Improper Torque",
    "Wrong Material Cert",
    "Corrosion / Oxidation",
    "Weld Defect",
    "Missing Feature",
    "Contamination",
    "Marking / Labeling Error",
]

DEFECT_SEVERITIES = ["Critical", "Major", "Minor"]

PART_LOCATIONS = ["Flange", "Bore", "Weld Zone", "Surface", "Thread", "Seal Face", "Datum", "Edge Break"]

DISPOSITIONS = ["Scrap", "Rework", "Use-As-Is", "Return to Supplier", "MRB Hold"]
# MRB = Material Review Board — a real aerospace quality process

ROOT_CAUSES = [
    "Worn tooling",
    "Operator error",
    "Incorrect setup",
    "Supplier non-conformance",
    "Environmental contamination",
    "Design tolerance stack-up",
    "Machine calibration drift",
    "Incoming material defect",
    "Process parameter drift",
    "Inadequate inspection",
]

ACTIONS_TAKEN = [
    "Tool replacement and re-run",
    "Operator retraining issued",
    "Process SOP updated",
    "Supplier corrective action request (SCAR) issued",
    "Cleanroom protocol enforced",
    "Engineering review of tolerance stack",
    "Machine recalibrated and verified",
    "Supplier lot quarantined and returned",
    "Process control limits tightened",
    "Inspection frequency increased",
]


# =============================================================================
# SECTION 2: DATABASE SETUP — creating the tables
# =============================================================================

def create_database(db_path: str) -> sqlite3.Connection:
    """
    Creates the SQLite database file and all five tables.

    LEARNING NOTE: A function is defined with `def name(parameters):`.
    The `-> sqlite3.Connection` part is a "type hint" — it tells other
    developers (and your IDE) what type of value this function returns.
    It doesn't change how the code runs; it's just documentation.

    sqlite3.connect() creates the file if it doesn't exist, or opens it
    if it does. The connection object `conn` is your handle to the database.
    A `cursor` is how you send SQL commands to the database.
    """
    if os.path.exists(db_path):
        os.remove(db_path)  # Start fresh every time we run this script
        print(f"  Removed existing database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # LEARNING NOTE: cursor.execute() sends one SQL statement to the database.
    # The triple-quoted strings (""" ... """) let us write multi-line strings
    # in Python — very handy for SQL which often spans multiple lines.

    # --- Table 1: suppliers ---
    cursor.execute("""
        CREATE TABLE suppliers (
            supplier_id     TEXT PRIMARY KEY,
            supplier_name   TEXT NOT NULL,
            tier            TEXT NOT NULL,
            country         TEXT NOT NULL,
            approved_status INTEGER NOT NULL  -- SQLite stores booleans as 0/1
        )
    """)

    # --- Table 2: production_runs ---
    cursor.execute("""
        CREATE TABLE production_runs (
            run_id          TEXT PRIMARY KEY,
            part_number     TEXT NOT NULL,
            production_line TEXT NOT NULL,
            run_date        TEXT NOT NULL,    -- Stored as ISO string: "2024-03-15"
            shift           TEXT NOT NULL,
            operator_id     TEXT NOT NULL,
            planned_qty     INTEGER NOT NULL,
            actual_qty      INTEGER NOT NULL
        )
    """)

    # --- Table 3: inspections ---
    cursor.execute("""
        CREATE TABLE inspections (
            inspection_id   TEXT PRIMARY KEY,
            run_id          TEXT NOT NULL,
            supplier_id     TEXT,             -- NULL allowed: not all inspections have a supplier
            inspection_type TEXT NOT NULL,
            inspection_date TEXT NOT NULL,
            inspector_id    TEXT NOT NULL,
            result          TEXT NOT NULL,    -- "Pass" or "Fail"
            dimension_error REAL,             -- How far out of tolerance, in mm. NULL if passed.
            visual_score    INTEGER,          -- 1-10 visual quality score
            FOREIGN KEY (run_id)      REFERENCES production_runs(run_id),
            FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
        )
    """)
    # LEARNING NOTE: FOREIGN KEY constraints enforce referential integrity —
    # they prevent you from inserting an inspection with a run_id that doesn't
    # exist in production_runs. This is fundamental relational database design.

    # --- Table 4: defects ---
    cursor.execute("""
        CREATE TABLE defects (
            defect_id           TEXT PRIMARY KEY,
            inspection_id       TEXT NOT NULL,
            defect_type         TEXT NOT NULL,
            severity            TEXT NOT NULL,
            location_on_part    TEXT NOT NULL,
            disposition         TEXT NOT NULL,
            cost_impact         REAL NOT NULL,  -- Estimated cost in USD
            FOREIGN KEY (inspection_id) REFERENCES inspections(inspection_id)
        )
    """)

    # --- Table 5: corrective_actions ---
    cursor.execute("""
        CREATE TABLE corrective_actions (
            ca_id           TEXT PRIMARY KEY,
            defect_id       TEXT NOT NULL,
            root_cause      TEXT NOT NULL,
            action_taken    TEXT NOT NULL,
            date_opened     TEXT NOT NULL,
            date_closed     TEXT,             -- NULL means still open
            effectiveness   TEXT,             -- "Effective", "Partially Effective", "Ineffective"
            FOREIGN KEY (defect_id) REFERENCES defects(defect_id)
        )
    """)

    conn.commit()  # "Commit" saves the schema changes to the file.
    print("  Database schema created: 5 tables")
    return conn


# =============================================================================
# SECTION 3: DATA GENERATION — the interesting part
# =============================================================================

def generate_production_runs(n: int = 500) -> list[dict]:
    """
    Generates n production run records.

    LEARNING NOTE: `list[dict]` means this function returns a list of
    dictionaries. A dictionary is Python's key-value data structure:
        {"run_id": "RUN-001", "part_number": "AS-7740-FRAME", ...}
    This is exactly what a row of data looks like before we put it in SQL.

    We generate 18 months of data to give the ML models enough history
    to find patterns in.
    """
    runs = []
    start_date = datetime(2023, 1, 1)

    for i in range(n):
        run_date = start_date + timedelta(days=random.randint(0, 548))  # ~18 months

        # LEARNING NOTE: np.random.randint vs random.randint
        # Both give random integers, but numpy's version is faster for large
        # arrays. Here we use Python's built-in random since we're making
        # one record at a time. We'll use numpy more in the ML phase.
        planned = random.randint(10, 200)

        # Actual quantity is almost always slightly less than planned
        # np.random.normal(mean, std_dev) gives a normally distributed number
        # — most values cluster near `planned`, with occasional big shortfalls
        yield_factor = np.random.normal(0.96, 0.04)   # ~96% yield on average
        yield_factor = max(0.70, min(1.0, yield_factor))  # clamp between 70-100%
        actual = int(planned * yield_factor)

        runs.append({
            "run_id":          f"RUN-{str(i+1).zfill(5)}",
            "part_number":     random.choice(PART_NUMBERS),
            "production_line": random.choice(PRODUCTION_LINES),
            "run_date":        run_date.strftime("%Y-%m-%d"),
            # ^ .strftime() formats a datetime object as a string.
            #   "%Y-%m-%d" means: 4-digit year, 2-digit month, 2-digit day.
            "shift":           random.choice(SHIFTS),
            "operator_id":     random.choice(OPERATOR_IDS),
            "planned_qty":     planned,
            "actual_qty":      actual,
        })

    return runs


def generate_inspections(runs: list[dict], supplier_ids: list[str]) -> list[dict]:
    """
    For each production run, generate 1-3 inspection records.

    KEY DESIGN DECISION: We deliberately make Night shift and certain
    production lines have higher failure rates. This is the "signal"
    that the ML model will learn to detect. Real manufacturing data
    almost always has patterns like this (worn tooling on a specific
    line, a less experienced night crew, etc.)
    """
    inspections = []
    insp_counter = 0

    # These dictionaries define our "baked-in" failure patterns.
    # The ML model should discover these automatically later!
    line_failure_boost = {
        "Line-Alpha":   0.0,   # Best performing line
        "Line-Bravo":   0.05,  # Slightly elevated
        "Line-Charlie": 0.20,  # Struggling line
        "Line-Delta":   0.08,  # Above average issues
    }
    shift_failure_boost = {
        "Day":   0.0,    # Best shift
        "Swing": 0.05,
        "Night": 0.20,   # Night shift has the most issues
    }

    for run in runs:
        # Most runs get 1 inspection, some get 2-3
        num_inspections = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
        # ^ random.choices() picks from a list WITH weights.
        #   weights=[60, 30, 10] means: 60% chance of 1, 30% chance of 2, 10% chance of 3.
        #   It returns a list, so [0] gets the first (and only) element.

        for j in range(num_inspections):
            insp_counter += 1

            # Base failure rate depends on inspection type
            insp_type = random.choice(INSPECTION_TYPES)
            base_fail_rate = {
                "IQC":        0.12,  # 12% of incoming parts fail
                "In-Process": 0.08,
                "Final":      0.05,  # Tighter — catching escapes here is bad
                "FAI":        0.15,  # First articles often need adjustment
                "Receiving":  0.10,
            }[insp_type]

            # Add the shift and line failure boosts
            total_fail_rate = (
                base_fail_rate
                + line_failure_boost[run["production_line"]]
                + shift_failure_boost[run["shift"]]
            )

            # Did this inspection pass or fail?
            result = "Fail" if random.random() < total_fail_rate else "Pass"
            # ^ random.random() returns a float between 0.0 and 1.0.
            #   If it's less than our failure rate, the inspection fails.

            # Dimension error is only recorded for failed inspections
            dim_error = None
            if result == "Fail":
                # np.random.exponential gives values that are mostly small
                # but occasionally large — realistic for out-of-tolerance errors
                dim_error = round(np.random.exponential(0.05), 4)

            # Visual score: Pass → 7-10, Fail → 1-6
            visual_score = random.randint(7, 10) if result == "Pass" else random.randint(1, 6)

            # Supplier is recorded for IQC and Receiving inspections
            supplier_id = random.choice(supplier_ids) if insp_type in ("IQC", "Receiving") else None

            # Inspection date is on or after the run date
            run_dt = datetime.strptime(run["run_date"], "%Y-%m-%d")
            insp_date = run_dt + timedelta(days=random.randint(0, 3))

            inspections.append({
                "inspection_id":   f"INS-{str(insp_counter).zfill(6)}",
                "run_id":          run["run_id"],
                "supplier_id":     supplier_id,
                "inspection_type": insp_type,
                "inspection_date": insp_date.strftime("%Y-%m-%d"),
                "inspector_id":    f"INSP-{str(random.randint(1, 8)).zfill(2)}",
                "result":          result,
                "dimension_error": dim_error,
                "visual_score":    visual_score,
            })

    return inspections


def generate_defects(inspections: list[dict]) -> list[dict]:
    """
    For every failed inspection, generate 1-3 defect records.

    LEARNING NOTE: This function uses list comprehension — a compact
    way to filter a list. [x for x in my_list if condition] is
    equivalent to:
        result = []
        for x in my_list:
            if condition:
                result.append(x)
    """
    failed_inspections = [i for i in inspections if i["result"] == "Fail"]
    defects = []
    defect_counter = 0

    # Map severity to a realistic cost range (USD)
    severity_cost = {
        "Critical": (5000, 50000),
        "Major":    (500,  5000),
        "Minor":    (50,   500),
    }

    for insp in failed_inspections:
        num_defects = random.choices([1, 2, 3], weights=[70, 22, 8])[0]

        for _ in range(num_defects):
            # The underscore _ is Python convention for "I don't need this variable"
            defect_counter += 1
            severity = random.choices(
                DEFECT_SEVERITIES,
                weights=[5, 25, 70]  # Most defects are Minor, few are Critical
            )[0]

            cost_min, cost_max = severity_cost[severity]
            cost = round(random.uniform(cost_min, cost_max), 2)

            defects.append({
                "defect_id":        f"DEF-{str(defect_counter).zfill(6)}",
                "inspection_id":    insp["inspection_id"],
                "defect_type":      random.choice(DEFECT_TYPES),
                "severity":         severity,
                "location_on_part": random.choice(PART_LOCATIONS),
                "disposition":      random.choice(DISPOSITIONS),
                "cost_impact":      cost,
            })

    return defects


def generate_corrective_actions(defects: list[dict]) -> list[dict]:
    """
    Generates corrective actions for Critical and Major defects.
    Minor defects often don't get a formal CA — this is realistic.
    """
    cas = []
    ca_counter = 0

    # Only critical and major defects get corrective actions
    qualifying_defects = [d for d in defects if d["severity"] in ("Critical", "Major")]

    for defect in qualifying_defects:
        ca_counter += 1

        # Pick a root cause, then pick the matching action
        # zip() pairs two lists together: [(cause1, action1), (cause2, action2), ...]
        rc_idx = random.randint(0, len(ROOT_CAUSES) - 1)
        root_cause = ROOT_CAUSES[rc_idx]
        action_taken = ACTIONS_TAKEN[rc_idx]  # Aligned index = consistent pairing

        # Date opened is within 3 days of the defect being found
        # We need the inspection date — look it up via the defect's inspection_id
        date_opened = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 548))

        # 75% of CAs are closed, 25% are still open (backlog)
        is_closed = random.random() < 0.75
        date_closed = None
        effectiveness = None
        if is_closed:
            days_to_close = random.randint(3, 45)
            date_closed = (date_opened + timedelta(days=days_to_close)).strftime("%Y-%m-%d")
            effectiveness = random.choices(
                ["Effective", "Partially Effective", "Ineffective"],
                weights=[65, 25, 10]
            )[0]

        cas.append({
            "ca_id":         f"CA-{str(ca_counter).zfill(5)}",
            "defect_id":     defect["defect_id"],
            "root_cause":    root_cause,
            "action_taken":  action_taken,
            "date_opened":   date_opened.strftime("%Y-%m-%d"),
            "date_closed":   date_closed,
            "effectiveness": effectiveness,
        })

    return cas


# =============================================================================
# SECTION 4: WRITING TO THE DATABASE
# =============================================================================

def insert_data(conn: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    """
    Inserts a list of dictionaries into a database table.

    LEARNING NOTE: This is a generic, reusable function — it works for
    ANY table and ANY list of dicts, as long as the dict keys match
    the table column names.

    rows[0].keys() gets the column names from the first row's dictionary.
    ", ".join(columns) turns ["a", "b", "c"] into "a, b, c".
    The ? placeholders prevent SQL injection attacks — never build SQL
    strings by concatenating user input directly!
    """
    if not rows:
        print(f"  Warning: no rows to insert for {table}")
        return

    columns = list(rows[0].keys())
    placeholders = ", ".join(["?" for _ in columns])
    col_names = ", ".join(columns)
    sql = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})"

    # Convert list of dicts to list of tuples (what sqlite3 expects)
    values = [tuple(row[col] for col in columns) for row in rows]
    # ^ This is a list comprehension inside another list comprehension.
    #   For each row dict, build a tuple of values in column order.

    cursor = conn.cursor()
    cursor.executemany(sql, values)
    # ^ executemany() is much faster than calling execute() in a loop.
    #   It sends all rows in one batch operation.
    conn.commit()
    print(f"  Inserted {len(rows):,} rows into {table}")
    # ^ {len(rows):,} — the :, format spec adds comma separators: 1234 → "1,234"


# =============================================================================
# SECTION 5: MAIN — the entry point
# =============================================================================

def main():
    db_path = "aerospace_quality.db"
    print(f"\n{'='*55}")
    print("  Aerospace Quality Analytics — Data Generator")
    print(f"{'='*55}\n")

    # Step 1: Create the database and tables
    print("Step 1: Creating database schema...")
    conn = create_database(db_path)

    # Step 2: Generate suppliers (fixed — not random, defined above)
    print("\nStep 2: Inserting supplier master data...")
    supplier_rows = [
        {
            "supplier_id":     s[0],
            "supplier_name":   s[1],
            "tier":            s[2],
            "country":         s[3],
            "approved_status": int(s[4]),  # Convert True/False to 1/0 for SQLite
        }
        for s in SUPPLIERS
    ]
    insert_data(conn, "suppliers", supplier_rows)

    # Step 3: Generate production runs
    print("\nStep 3: Generating production runs (18 months)...")
    runs = generate_production_runs(n=500)
    insert_data(conn, "production_runs", runs)

    # Step 4: Generate inspections
    print("\nStep 4: Generating inspection records...")
    supplier_ids = [s[0] for s in SUPPLIERS]
    inspections = generate_inspections(runs, supplier_ids)
    insert_data(conn, "inspections", inspections)

    # Step 5: Generate defects
    print("\nStep 5: Generating defect records...")
    defects = generate_defects(inspections)
    insert_data(conn, "defects", defects)

    # Step 6: Generate corrective actions
    print("\nStep 6: Generating corrective actions...")
    cas = generate_corrective_actions(defects)
    insert_data(conn, "corrective_actions", cas)

    # Step 7: Print a summary of what we created
    print(f"\n{'='*55}")
    print("  Data generation complete!")
    print(f"{'='*55}")
    cursor = conn.cursor()
    for table in ["suppliers", "production_runs", "inspections", "defects", "corrective_actions"]:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table:<25} {count:>6,} rows")

    print(f"\n  Database file: {os.path.abspath(db_path)}")
    print("  Ready for Phase 2: SQL queries + Pandas pipeline\n")
    conn.close()


# =============================================================================
# LEARNING NOTE: This pattern — `if __name__ == "__main__"` — is Python's
# standard way to say "only run main() if this file is run directly".
# If another script imports this file (e.g. `import generate_data`),
# main() will NOT run automatically. This makes files safe to import.
# =============================================================================
if __name__ == "__main__":
    main()