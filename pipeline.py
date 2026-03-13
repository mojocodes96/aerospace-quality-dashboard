"""
pipeline.py
===========
Phase 2: Load data from SQLite, run analytical SQL queries,
and build clean Pandas DataFrames for the dashboard and ML models.

WHAT YOU'LL LEARN IN THIS FILE
--------------------------------
1. pd.read_sql()      — how to run a SQL query and get a DataFrame back
2. DataFrame basics   — rows, columns, dtypes, .head(), .info()
3. .groupby()         — the Pandas equivalent of SQL GROUP BY
4. .merge()           — the Pandas equivalent of SQL JOIN
5. Feature engineering — creating new columns from existing ones
6. Saving DataFrames  — writing to CSV so other scripts can load them fast
"""

import sqlite3
import pandas as pd       # The core library. "pd" is the universal alias — everyone uses it.
import numpy as np
from datetime import datetime
import os

# =============================================================================
# SECTION 1: CONNECTION HELPER
# =============================================================================

DB_PATH = "aerospace_quality.db"

def get_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite database."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Run generate_data.py first."
        )
    return sqlite3.connect(DB_PATH)


# =============================================================================
# SECTION 2: LOADING DATA WITH pd.read_sql()
# =============================================================================
# LEARNING NOTE: pd.read_sql(query, connection) is the key bridge between
# SQL and Pandas. It runs the SQL query against your database and returns
# the results as a DataFrame — a table with named columns.
#
# A DataFrame is the central object in Pandas. Think of it like a
# spreadsheet in memory: rows are records, columns are fields, and you
# can do almost any transformation you'd do in Excel — but in code.

def load_raw_tables(conn: sqlite3.Connection) -> dict:
    """
    Loads all five tables into DataFrames.
    Returns a dictionary so we can access them by name: tables["defects"]
    """
    print("Loading raw tables from database...")

    tables = {}

    # The simplest possible pd.read_sql call — just "give me the whole table"
    tables["suppliers"]          = pd.read_sql("SELECT * FROM suppliers",          conn)
    tables["production_runs"]    = pd.read_sql("SELECT * FROM production_runs",    conn)
    tables["inspections"]        = pd.read_sql("SELECT * FROM inspections",        conn)
    tables["defects"]            = pd.read_sql("SELECT * FROM defects",            conn)
    tables["corrective_actions"] = pd.read_sql("SELECT * FROM corrective_actions", conn)

    # LEARNING NOTE: Let's explore what a DataFrame looks like.
    # .shape returns (number_of_rows, number_of_columns) as a tuple.
    for name, df in tables.items():
        print(f"  {name:<25} {df.shape[0]:>5} rows × {df.shape[1]} columns")

    return tables


# =============================================================================
# SECTION 3: DATE PARSING
# =============================================================================
# LEARNING NOTE: When Pandas reads dates from SQLite, it treats them as
# plain strings like "2024-03-15". We need to convert them to proper
# datetime objects so we can do date math (e.g. "group by month").
# pd.to_datetime() handles this conversion.

def parse_dates(tables: dict) -> dict:
    """Converts date string columns to proper datetime objects."""
    print("\nParsing date columns...")

    tables["production_runs"]["run_date"] = pd.to_datetime(
        tables["production_runs"]["run_date"]
    )
    tables["inspections"]["inspection_date"] = pd.to_datetime(
        tables["inspections"]["inspection_date"]
    )
    tables["corrective_actions"]["date_opened"] = pd.to_datetime(
        tables["corrective_actions"]["date_opened"]
    )
    tables["corrective_actions"]["date_closed"] = pd.to_datetime(
        tables["corrective_actions"]["date_closed"]   # NaT (Not a Time) where NULL
    )

    # Now we can extract the month/year for time-series grouping.
    # dt.to_period("M") converts a date to its month period: "2024-03"
    tables["production_runs"]["month"] = (
        tables["production_runs"]["run_date"].dt.to_period("M")
    )
    tables["inspections"]["month"] = (
        tables["inspections"]["inspection_date"].dt.to_period("M")
    )

    print("  Date columns parsed and month periods added.")
    return tables


# =============================================================================
# SECTION 4: KPI DATAFRAME
# =============================================================================
# LEARNING NOTE: These are the "headline numbers" for the dashboard —
# overall pass rate, total defects, total cost impact, open CAs.
# We build this using a mix of SQL (for the complex joins) and Pandas
# (for simple aggregations on already-loaded DataFrames).

def build_kpi_dataframe(conn: sqlite3.Connection, tables: dict) -> pd.DataFrame:
    """
    Builds a single-row DataFrame of top-level KPIs.
    A single row might seem odd, but it makes the dashboard code clean —
    you just do kpis["overall_pass_rate"][0] to get any metric.
    """
    print("\nBuilding KPI dataframe...")

    insp = tables["inspections"]
    defects = tables["defects"]
    runs = tables["production_runs"]
    cas = tables["corrective_actions"]

    # --- KPI 1: Overall pass rate ---
    # LEARNING NOTE: Boolean indexing — (insp["result"] == "Pass") creates a
    # Series of True/False values. .sum() counts the Trues (True = 1, False = 0).
    total_inspections = len(insp)
    total_passes      = (insp["result"] == "Pass").sum()
    overall_pass_rate = round(100 * total_passes / total_inspections, 1)

    # --- KPI 2: Total defect cost ---
    total_defect_cost = round(defects["cost_impact"].sum(), 2)
    # ^ .sum() adds up an entire column. One of the most-used Pandas methods.

    # --- KPI 3: Average cost per defect ---
    avg_cost_per_defect = round(defects["cost_impact"].mean(), 2)
    # ^ .mean() gives the average. Also: .median(), .std(), .min(), .max()

    # --- KPI 4: Critical defect count ---
    critical_count = (defects["severity"] == "Critical").sum()

    # --- KPI 5: Open corrective actions ---
    open_cas = cas["date_closed"].isna().sum()
    # ^ .isna() returns True where a value is null/NaN/NaT. The opposite is .notna()

    # --- KPI 6: Overall yield (actual vs planned) ---
    total_planned = runs["planned_qty"].sum()
    total_actual  = runs["actual_qty"].sum()
    overall_yield = round(100 * total_actual / total_planned, 1)

    # --- KPI 7: Supplier defect rate (via SQL JOIN) ---
    # This one is easier in SQL because it joins three tables.
    # We pass a multi-line SQL string directly to pd.read_sql().
    supplier_defect_query = """
        SELECT
            COUNT(DISTINCT d.defect_id) AS supplier_defects,
            COUNT(DISTINCT i.inspection_id) AS supplier_inspections
        FROM inspections i
        LEFT JOIN defects d ON i.inspection_id = d.inspection_id
        WHERE i.supplier_id IS NOT NULL
    """
    # LEARNING NOTE: LEFT JOIN returns all rows from the left table (inspections)
    # and matching rows from the right table (defects). Where there's no match,
    # you get NULL. This means we keep inspections that had no defects too.
    supplier_stats = pd.read_sql(supplier_defect_query, conn)
    supplier_defect_rate = round(
        100 * supplier_stats["supplier_defects"][0] / supplier_stats["supplier_inspections"][0], 1
    )

    # Build the single-row DataFrame using a dictionary
    # LEARNING NOTE: pd.DataFrame([{...}]) — wrapping the dict in a list [ ]
    # creates a one-row DataFrame. Without the list it would fail.
    df_kpis = pd.DataFrame([{
        "overall_pass_rate":    overall_pass_rate,
        "total_defect_cost":    total_defect_cost,
        "avg_cost_per_defect":  avg_cost_per_defect,
        "critical_defect_count": critical_count,
        "open_corrective_actions": open_cas,
        "overall_yield_pct":    overall_yield,
        "supplier_defect_rate": supplier_defect_rate,
        "total_inspections":    total_inspections,
        "total_defects":        len(defects),
        "total_production_runs": len(runs),
    }])

    print(f"  Overall pass rate:     {overall_pass_rate}%")
    print(f"  Total defect cost:    ${total_defect_cost:,.2f}")
    print(f"  Critical defects:      {critical_count}")
    print(f"  Open CAs:              {open_cas}")
    print(f"  Overall yield:         {overall_yield}%")

    return df_kpis


# =============================================================================
# SECTION 5: DEFECT ANALYSIS DATAFRAME
# =============================================================================

def build_defect_dataframe(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Builds a rich defect analysis DataFrame by joining all relevant tables.
    This is the workhorse query — it pulls together everything we need
    to analyze defects: what they are, where they came from, what they cost.
    """
    print("\nBuilding defect analysis dataframe...")

    # LEARNING NOTE: This is a complex multi-table JOIN — the kind of SQL
    # the job description specifically mentioned. Let's break it down:
    #
    # We want one row per defect, enriched with context from other tables.
    # The chain is:  defects → inspections → production_runs
    #                                      → suppliers (optional)
    #
    # STRFTIME('%Y-%m', date) extracts the year-month from a date string.
    # This is SQLite's date function — other databases use different syntax.
    # COALESCE(x, 'Unknown') returns x if it's not null, otherwise 'Unknown'.

    query = """
        SELECT
            d.defect_id,
            d.defect_type,
            d.severity,
            d.location_on_part,
            d.disposition,
            d.cost_impact,

            i.inspection_id,
            i.inspection_type,
            i.inspection_date,
            i.result             AS inspection_result,
            i.dimension_error,
            i.visual_score,

            pr.run_id,
            pr.part_number,
            pr.production_line,
            pr.run_date,
            pr.shift,
            pr.operator_id,
            STRFTIME('%Y-%m', pr.run_date)  AS year_month,
            STRFTIME('%Y',    pr.run_date)  AS year,

            COALESCE(s.supplier_name, 'Internal') AS supplier_name,
            COALESCE(s.tier, 'N/A')               AS supplier_tier,
            COALESCE(s.country, 'N/A')            AS supplier_country

        FROM defects d
        JOIN inspections    i  ON d.inspection_id  = i.inspection_id
        JOIN production_runs pr ON i.run_id         = pr.run_id
        LEFT JOIN suppliers  s  ON i.supplier_id    = s.supplier_id

        ORDER BY i.inspection_date DESC
    """
    # NOTE: We use JOIN (inner join) for tables that must have a match,
    # and LEFT JOIN for suppliers because not every inspection has a supplier.

    df_defects = pd.read_sql(query, conn)

    # Convert date column now that we have it
    df_defects["inspection_date"] = pd.to_datetime(df_defects["inspection_date"])
    df_defects["run_date"]        = pd.to_datetime(df_defects["run_date"])

    print(f"  {len(df_defects)} defect records loaded with full context")
    print(f"  Columns: {list(df_defects.columns)}")

    return df_defects


# =============================================================================
# SECTION 6: MONTHLY TREND DATAFRAME
# =============================================================================

def build_trend_dataframe(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Builds a monthly time-series DataFrame for trend charts.
    This is what powers the "quality over time" line chart on the dashboard.
    """
    print("\nBuilding monthly trend dataframe...")

    query = """
        SELECT
            STRFTIME('%Y-%m', i.inspection_date)        AS year_month,
            COUNT(i.inspection_id)                       AS total_inspections,
            SUM(CASE WHEN i.result = 'Pass' THEN 1 ELSE 0 END) AS passes,
            SUM(CASE WHEN i.result = 'Fail' THEN 1 ELSE 0 END) AS failures,
            ROUND(
                100.0 * SUM(CASE WHEN i.result = 'Pass' THEN 1 ELSE 0 END)
                / COUNT(i.inspection_id), 1
            )                                            AS pass_rate_pct,
            COUNT(DISTINCT d.defect_id)                  AS defect_count,
            ROUND(COALESCE(SUM(d.cost_impact), 0), 2)   AS defect_cost
        FROM inspections i
        LEFT JOIN defects d ON i.inspection_id = d.inspection_id
        GROUP BY year_month
        ORDER BY year_month
    """
    # LEARNING NOTE: COALESCE(SUM(d.cost_impact), 0) — if a month has no
    # defects, SUM returns NULL. COALESCE converts that NULL to 0 so we
    # don't get gaps in our chart data.

    df_trend = pd.read_sql(query, conn)

    # Convert year_month string to a proper datetime for charting
    df_trend["year_month_dt"] = pd.to_datetime(df_trend["year_month"] + "-01")
    # ^ We append "-01" to make it a full date: "2024-03" → "2024-03-01"
    # Plotly needs a full date to draw a proper time axis.

    # Add a rolling 3-month average of pass rate — useful for spotting trends
    # LEARNING NOTE: .rolling(window=3).mean() calculates a moving average.
    # Each value becomes the average of itself and the 2 preceding values.
    # The first 2 values will be NaN because there aren't enough prior points.
    df_trend["pass_rate_rolling_3m"] = (
        df_trend["pass_rate_pct"].rolling(window=3, min_periods=1).mean().round(1)
    )

    print(f"  {len(df_trend)} months of trend data")
    print(f"  Date range: {df_trend['year_month'].iloc[0]} to {df_trend['year_month'].iloc[-1]}")
    # ^ .iloc[0] = first row, .iloc[-1] = last row. Index-based access.

    return df_trend


# =============================================================================
# SECTION 7: SUPPLIER SCORECARD DATAFRAME
# =============================================================================

def build_supplier_dataframe(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Builds a supplier quality scorecard.
    This is classic Pareto analysis — which suppliers are causing the most pain?
    """
    print("\nBuilding supplier scorecard dataframe...")

    query = """
        SELECT
            s.supplier_id,
            s.supplier_name,
            s.tier,
            s.country,
            s.approved_status,
            COUNT(DISTINCT i.inspection_id)              AS total_inspections,
            SUM(CASE WHEN i.result = 'Fail' THEN 1 ELSE 0 END) AS failed_inspections,
            ROUND(
                100.0 * SUM(CASE WHEN i.result = 'Fail' THEN 1 ELSE 0 END)
                / COUNT(DISTINCT i.inspection_id), 1
            )                                            AS fail_rate_pct,
            COUNT(DISTINCT d.defect_id)                  AS total_defects,
            ROUND(COALESCE(SUM(d.cost_impact), 0), 2)   AS total_defect_cost,
            ROUND(
                COALESCE(AVG(i.visual_score), 0), 1
            )                                            AS avg_visual_score
        FROM suppliers s
        LEFT JOIN inspections i ON s.supplier_id = i.supplier_id
        LEFT JOIN defects d     ON i.inspection_id = d.inspection_id
        GROUP BY s.supplier_id
        ORDER BY fail_rate_pct DESC
    """

    df_supplier = pd.read_sql(query, conn)

    # Convert approved_status from 0/1 integer back to a readable label
    # LEARNING NOTE: .map() applies a function or dictionary to every value
    # in a column. {1: "Approved", 0: "Not Approved"} is the mapping dict.
    df_supplier["approved_status"] = df_supplier["approved_status"].map(
        {1: "Approved", 0: "Not Approved"}
    )

    print(f"  {len(df_supplier)} suppliers scored")

    return df_supplier


# =============================================================================
# SECTION 8: FAILURE RATE BY DIMENSION DATAFRAME
# =============================================================================

def build_failure_breakdown(conn: sqlite3.Connection) -> dict:
    """
    Builds several small breakdown DataFrames used for bar charts.
    Returns a dict of DataFrames, one per dimension.
    """
    print("\nBuilding failure breakdown dataframes...")

    breakdowns = {}

    # --- By production line ---
    breakdowns["by_line"] = pd.read_sql("""
        SELECT
            pr.production_line,
            COUNT(i.inspection_id)  AS total,
            SUM(CASE WHEN i.result = 'Fail' THEN 1 ELSE 0 END) AS failures,
            ROUND(100.0 * SUM(CASE WHEN i.result = 'Fail' THEN 1 ELSE 0 END)
                  / COUNT(i.inspection_id), 1) AS fail_rate_pct
        FROM inspections i
        JOIN production_runs pr ON i.run_id = pr.run_id
        GROUP BY pr.production_line
        ORDER BY fail_rate_pct DESC
    """, conn)

    # --- By shift ---
    breakdowns["by_shift"] = pd.read_sql("""
        SELECT
            pr.shift,
            COUNT(i.inspection_id)  AS total,
            SUM(CASE WHEN i.result = 'Fail' THEN 1 ELSE 0 END) AS failures,
            ROUND(100.0 * SUM(CASE WHEN i.result = 'Fail' THEN 1 ELSE 0 END)
                  / COUNT(i.inspection_id), 1) AS fail_rate_pct
        FROM inspections i
        JOIN production_runs pr ON i.run_id = pr.run_id
        GROUP BY pr.shift
        ORDER BY fail_rate_pct DESC
    """, conn)

    # --- By inspector ---
    breakdowns["by_inspector"] = pd.read_sql("""
        SELECT
            i.inspector_id,
            COUNT(i.inspection_id)  AS total,
            SUM(CASE WHEN i.result = 'Fail' THEN 1 ELSE 0 END) AS failures,
            ROUND(100.0 * SUM(CASE WHEN i.result = 'Fail' THEN 1 ELSE 0 END)
                  / COUNT(i.inspection_id), 1) AS fail_rate_pct
        FROM inspections i
        JOIN production_runs pr ON i.run_id = pr.run_id
        GROUP BY i.inspector_id
        ORDER BY fail_rate_pct DESC
    """, conn)

    # --- By defect type (Pareto) ---
    breakdowns["pareto_defect"] = pd.read_sql("""
        SELECT
            defect_type,
            COUNT(*)                    AS count,
            ROUND(SUM(cost_impact), 2)  AS total_cost,
            severity
        FROM defects
        GROUP BY defect_type
        ORDER BY count DESC
    """, conn)

    # LEARNING NOTE: Pareto analysis — adding a cumulative percentage column.
    # This is the "80/20 rule" calculation: which defect types account for
    # 80% of all defects? This is a core quality engineering concept.
    df_pareto = breakdowns["pareto_defect"].copy()
    # ^ .copy() creates an independent copy so changes don't affect the original

    df_pareto["cumulative_pct"] = (
        df_pareto["count"].cumsum()          # Running total: [5, 12, 18, ...]
        / df_pareto["count"].sum()           # Divide by grand total
        * 100                                # Convert to percentage
    ).round(1)
    # ^ .cumsum() = cumulative sum. This is how you calculate Pareto in Pandas.

    breakdowns["pareto_defect"] = df_pareto

    # --- By part number ---
    breakdowns["by_part"] = pd.read_sql("""
        SELECT
            pr.part_number,
            COUNT(DISTINCT i.inspection_id)  AS total_inspections,
            COUNT(DISTINCT d.defect_id)      AS total_defects,
            ROUND(COALESCE(SUM(d.cost_impact), 0), 2) AS total_cost
        FROM production_runs pr
        JOIN inspections i ON pr.run_id = pr.run_id
        LEFT JOIN defects d ON i.inspection_id = d.inspection_id
        GROUP BY pr.part_number
        ORDER BY total_defects DESC
    """, conn)

    for name, df in breakdowns.items():
        print(f"  {name:<20} {len(df)} rows")

    return breakdowns


# =============================================================================
# SECTION 9: ML FEATURE ENGINEERING
# =============================================================================
# This is the most important section for Phase 3.
# Feature engineering = creating the input columns the ML model will learn from.
# The better your features, the better your model's predictions.

def build_ml_features(conn: sqlite3.Connection, tables: dict) -> pd.DataFrame:
    """
    Builds the ML-ready feature DataFrame.

    LEARNING NOTE: Machine learning models can't work with raw text like
    "Line-Charlie" or "Night". We need to convert categorical variables
    into numbers. There are two main approaches:

    1. Label encoding: Night=0, Day=1, Swing=2 (arbitrary numbers)
    2. One-hot encoding: creates a new True/False column for each category
       e.g. shift_Night, shift_Day, shift_Swing

    We'll use one-hot encoding (pd.get_dummies) because it doesn't imply
    any ordering between categories — Night isn't "less than" Day.

    The TARGET variable (what we're predicting) is whether an inspection
    will fail: result = "Fail" → 1, "Pass" → 0.
    """
    print("\nBuilding ML feature dataframe...")

    # Start with the joined inspections + production_runs data
    query = """
        SELECT
            i.inspection_id,
            i.result,
            i.inspection_type,
            i.visual_score,
            i.dimension_error,
            pr.production_line,
            pr.shift,
            pr.part_number,
            pr.operator_id,
            CAST(STRFTIME('%m', i.inspection_date) AS INTEGER) AS month_num,
            CAST(STRFTIME('%H', i.inspection_date) AS INTEGER) AS hour_num,
            pr.planned_qty,
            pr.actual_qty,
            ROUND(100.0 * pr.actual_qty / pr.planned_qty, 1)  AS yield_pct
        FROM inspections i
        JOIN production_runs pr ON i.run_id = pr.run_id
    """
    # LEARNING NOTE: CAST(... AS INTEGER) converts a string to an integer.
    # STRFTIME('%m', date) gives the month number as a string "03",
    # so we CAST it to get the integer 3. Month number is a useful feature —
    # there might be seasonal patterns in the data.

    df = pd.read_sql(query, conn)

    # --- Target variable: binary encode the result column ---
    # LEARNING NOTE: .map() replaces values using a dictionary.
    # "Fail" → 1 (the "positive" class we're trying to detect)
    # "Pass" → 0
    df["target_fail"] = df["result"].map({"Fail": 1, "Pass": 0})

    # --- Fill nulls before encoding ---
    # dimension_error is NULL for passed inspections. Fill with 0.
    # visual_score should always exist, but fill just in case.
    df["dimension_error"] = df["dimension_error"].fillna(0)
    df["visual_score"]    = df["visual_score"].fillna(df["visual_score"].median())
    # ^ .fillna(value) replaces NaN with the given value.
    # Using the median for visual_score is better than 0 — it's a neutral fill.

    # --- One-hot encode categorical columns ---
    # pd.get_dummies() creates new binary columns for each unique value.
    # prefix= names them clearly: "line_Line-Alpha", "shift_Night", etc.
    # drop_first=True drops one category per group to avoid multicollinearity
    # (a statistical issue where one column can be predicted from the others).
    categorical_cols = ["production_line", "shift", "inspection_type", "part_number"]
    df_encoded = pd.get_dummies(
        df,
        columns=categorical_cols,
        prefix=["line", "shift", "insp_type", "part"],
        drop_first=True,
        dtype=int   # Store as 0/1 integers instead of True/False booleans
    )

    # Drop columns that aren't features (IDs, raw strings we've already encoded)
    cols_to_drop = ["inspection_id", "result", "operator_id", 
                "dimension_error", "visual_score",
                "planned_qty", "actual_qty", "yield_pct"]
    df_encoded = df_encoded.drop(columns=cols_to_drop)

    # Report the shape and a preview of the feature columns
    feature_cols = [c for c in df_encoded.columns if c != "target_fail"]
    print(f"  {len(df_encoded)} rows × {len(feature_cols)} features + 1 target")
    print(f"  Failure rate in dataset: "
          f"{round(100 * df_encoded['target_fail'].mean(), 1)}%")
    print(f"  Feature columns: {feature_cols[:8]}... (+{max(0,len(feature_cols)-8)} more)")

    return df_encoded


# =============================================================================
# SECTION 10: SAVE OUTPUTS
# =============================================================================

def save_outputs(
    df_kpis: pd.DataFrame,
    df_defects: pd.DataFrame,
    df_trend: pd.DataFrame,
    df_supplier: pd.DataFrame,
    breakdowns: dict,
    df_ml_features: pd.DataFrame,
) -> None:
    """
    Saves all DataFrames to CSV files in a /data subfolder.

    LEARNING NOTE: We save to CSV (not pickle or parquet) because:
    - CSVs are human-readable — you can open them in Excel to sanity check
    - They're universally compatible
    - For our data size (~1000 rows), the performance difference is negligible

    In production systems you'd use Parquet for large datasets — it's
    much faster and smaller. But CSV is perfect for learning and portfolios.
    """
    os.makedirs("data", exist_ok=True)
    # ^ os.makedirs creates the folder if it doesn't exist.
    # exist_ok=True means "don't throw an error if it already exists"

    outputs = {
        "data/kpis.csv":             df_kpis,
        "data/defects.csv":          df_defects,
        "data/trend.csv":            df_trend,
        "data/suppliers.csv":        df_supplier,
        "data/by_line.csv":          breakdowns["by_line"],
        "data/by_shift.csv":         breakdowns["by_shift"],
        "data/pareto_defect.csv":    breakdowns["pareto_defect"],
        "data/by_part.csv":          breakdowns["by_part"],
        "data/ml_features.csv":      df_ml_features,
        "data/by_inspector.csv":     breakdowns["by_inspector"],
    }

    print("\nSaving outputs to /data folder...")
    for path, df in outputs.items():
        df.to_csv(path, index=False)
        # ^ index=False stops Pandas from writing the row numbers (0,1,2,3...)
        # as an extra column in the CSV. You almost always want this.
        print(f"  Saved {path:<35} ({len(df):,} rows × {df.shape[1]} cols)")


# =============================================================================
# SECTION 11: MAIN
# =============================================================================

def main():
    print(f"\n{'='*55}")
    print("  Aerospace Quality Analytics — Data Pipeline")
    print(f"{'='*55}\n")

    conn = get_connection()

    # Load and prepare raw tables
    tables = load_raw_tables(conn)
    tables = parse_dates(tables)

    # Build each analytical DataFrame
    df_kpis         = build_kpi_dataframe(conn, tables)
    df_defects      = build_defect_dataframe(conn)
    df_trend        = build_trend_dataframe(conn)
    df_supplier     = build_supplier_dataframe(conn)
    breakdowns      = build_failure_breakdown(conn)
    df_ml_features  = build_ml_features(conn, tables)

    # Save everything to CSV
    save_outputs(df_kpis, df_defects, df_trend, df_supplier, breakdowns, df_ml_features)

    conn.close()

    print(f"\n{'='*55}")
    print("  Pipeline complete! Ready for Phase 3: ML models")
    print(f"{'='*55}\n")

    # STRETCH EXERCISE (try this on your own before Phase 3):
    # ----------------------------------------------------------
    # Open data/defects.csv in Excel and look at it.
    # Then try adding ONE new breakdown to build_failure_breakdown():
    # a "by_inspector" breakdown showing fail rate per inspector_id.
    # Hint: the pattern is identical to by_shift — just change the
    # GROUP BY column from pr.shift to i.inspector_id.
    # ----------------------------------------------------------


if __name__ == "__main__":
    main()