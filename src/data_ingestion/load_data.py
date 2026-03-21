"""
load_data.py - Phase 1: Data Ingestion
=======================================
WHAT THIS SCRIPT DOES:
    1. Drops any existing tables (clean slate every run)
    2. Reads the 5 CSV files from data/raw/
    3. Cleans data types (dates, booleans, nulls)
    4. Creates properly structured tables in PostgreSQL
    5. Loads all data with foreign key relationships
    6. Verifies everything loaded correctly

WHY WE LOAD INTO POSTGRESQL:
    In a production MLOps pipeline, your data should live in a
    database - not in CSV files. A database gives you:
    - Data integrity (primary/foreign keys prevent broken references)
    - Consistency (every script reads from the same source)
    - Speed (SQL queries are faster than pandas on CSVs)
    - Scalability (databases handle millions of rows)

HOW TO RUN:
    From your project root directory (with venv activated):
    python -m src.data_ingestion.load_data
"""

import pandas as pd
from sqlalchemy import (
    MetaData, Table, Column,
    String, Integer, Float, Boolean, Date, DateTime, Text,
    ForeignKey, inspect, text
)
from src.utils.config import Config
from src.utils.db import engine, test_connection


# ══════════════════════════════════════════════════════════════════════
# STEP 1: DROP existing tables (guarantees clean slate)
# ══════════════════════════════════════════════════════════════════════
# WHY: If you ran this script before (even if it failed halfway),
# old data might be stuck in the tables. Dropping removes EVERYTHING
# - the table structure AND the data. We recreate fresh every time.
# This makes the script safely re-runnable no matter what.

def drop_all_tables():
    """
    Drop all project tables if they exist.
    
    Uses engine.begin() which AUTO-COMMITS when the block ends.
    This guarantees the DROP actually takes effect (no silent rollbacks).
    
    We drop child tables before parent tables because PostgreSQL
    won't let you drop a parent table that children still reference.
    """
    print("\n🧹 Dropping existing tables (clean slate)...")

    # engine.begin() = auto-commit context manager
    # When this block ends successfully, changes are committed automatically
    # If an error occurs, changes are rolled back automatically
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS feature_usage CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS support_tickets CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS churn_events CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS subscriptions CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS accounts CASCADE"))

    print("   ✅ All tables dropped")


# ══════════════════════════════════════════════════════════════════════
# STEP 2: CREATE fresh table schemas
# ══════════════════════════════════════════════════════════════════════
# WHY: We explicitly define column types, primary keys (PK), and
# foreign keys (FK) instead of letting pandas guess.
#
# PRIMARY KEY (PK) = uniquely identifies each row (no duplicates)
# FOREIGN KEY (FK) = a column that must match a PK in another table
#   Example: subscriptions.account_id must exist in accounts.account_id

def create_all_tables():
    """Create all 5 tables with proper schema and relationships."""
    print("\n🔨 Creating database tables...")

    metadata = MetaData()

    # ── accounts (parent table - everyone references this) ───────────
    Table(
        "accounts", metadata,
        Column("account_id", String(20), primary_key=True),
        Column("account_name", String(100)),
        Column("industry", String(50)),
        Column("country", String(10)),
        Column("signup_date", Date),
        Column("referral_source", String(50)),
        Column("plan_tier", String(20)),
        Column("seats", Integer),
        Column("is_trial", Boolean),
        Column("churn_flag", Boolean),
    )

    # ── subscriptions (child of accounts) ────────────────────────────
    Table(
        "subscriptions", metadata,
        Column("subscription_id", String(20), primary_key=True),
        Column("account_id", String(20), ForeignKey("accounts.account_id")),
        Column("start_date", Date),
        Column("end_date", Date, nullable=True),
        Column("plan_tier", String(20)),
        Column("seats", Integer),
        Column("mrr_amount", Float),
        Column("arr_amount", Float),
        Column("is_trial", Boolean),
        Column("upgrade_flag", Boolean),
        Column("downgrade_flag", Boolean),
        Column("churn_flag", Boolean),
        Column("billing_frequency", String(20)),
        Column("auto_renew_flag", Boolean),
    )

    # ── feature_usage (child of subscriptions) ───────────────────────
    Table(
        "feature_usage", metadata,
        Column("usage_id", String(20), primary_key=True),
        Column("subscription_id", String(20), ForeignKey("subscriptions.subscription_id")),
        Column("usage_date", Date),
        Column("feature_name", String(50)),
        Column("usage_count", Integer),
        Column("usage_duration_secs", Integer),
        Column("error_count", Integer),
        Column("is_beta_feature", Boolean),
    )

    # ── support_tickets (child of accounts) ──────────────────────────
    Table(
        "support_tickets", metadata,
        Column("ticket_id", String(20), primary_key=True),
        Column("account_id", String(20), ForeignKey("accounts.account_id")),
        Column("submitted_at", DateTime),
        Column("closed_at", DateTime, nullable=True),
        Column("resolution_time_hours", Float),
        Column("priority", String(20)),
        Column("first_response_time_minutes", Integer),
        Column("satisfaction_score", Float, nullable=True),
        Column("escalation_flag", Boolean),
    )

    # ── churn_events (child of accounts) ─────────────────────────────
    Table(
        "churn_events", metadata,
        Column("churn_event_id", String(20), primary_key=True),
        Column("account_id", String(20), ForeignKey("accounts.account_id")),
        Column("churn_date", Date),
        Column("reason_code", String(50)),
        Column("refund_amount_usd", Float),
        Column("preceding_upgrade_flag", Boolean),
        Column("preceding_downgrade_flag", Boolean),
        Column("is_reactivation", Boolean),
        Column("feedback_text", Text, nullable=True),
    )

    # Actually create them in PostgreSQL
    metadata.create_all(engine)

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"   ✅ Tables created: {tables}")


# ══════════════════════════════════════════════════════════════════════
# STEP 3: Read and clean each CSV file
# ══════════════════════════════════════════════════════════════════════
# WHY CLEAN? Raw CSVs have messy types - dates as strings, booleans
# as "True"/"False" text, etc. We fix them BEFORE inserting so
# PostgreSQL stores them with the correct types.

def load_accounts():
    """Load and clean accounts.csv"""
    print("\n📂 Loading accounts.csv...")
    df = pd.read_csv(Config.RAW_DATA_DIR / "ravenstack_accounts.csv")
    df["signup_date"] = pd.to_datetime(df["signup_date"]).dt.date
    df["is_trial"] = df["is_trial"].astype(bool)
    df["churn_flag"] = df["churn_flag"].astype(bool)
    print(f"   ✅ {len(df)} rows | {list(df.columns)}")
    return df


def load_subscriptions():
    """Load and clean subscriptions.csv"""
    print("\n📂 Loading subscriptions.csv...")
    df = pd.read_csv(Config.RAW_DATA_DIR / "ravenstack_subscriptions.csv")
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["end_date"] = df["end_date"].apply(lambda x: x.date() if pd.notna(x) else None)
    for col in ["is_trial", "upgrade_flag", "downgrade_flag",
                "churn_flag", "auto_renew_flag"]:
        df[col] = df[col].astype(bool)
    df["mrr_amount"] = df["mrr_amount"].astype(float)
    df["arr_amount"] = df["arr_amount"].astype(float)
    df["seats"] = df["seats"].astype(int)
    print(f"   ✅ {len(df)} rows | {list(df.columns)}")
    return df


def load_feature_usage():
    """Load and clean feature_usage.csv"""
    print("\n📂 Loading feature_usage.csv...")
    df = pd.read_csv(Config.RAW_DATA_DIR / "ravenstack_feature_usage.csv")
    df["usage_date"] = pd.to_datetime(df["usage_date"]).dt.date
    df["is_beta_feature"] = df["is_beta_feature"].astype(bool)
    df["usage_count"] = df["usage_count"].astype(int)
    df["usage_duration_secs"] = df["usage_duration_secs"].astype(int)
    df["error_count"] = df["error_count"].astype(int)

    # ── Handle duplicate usage_ids in the raw CSV ────────────────────
    # The dataset generator created 21 rows with duplicate usage_id
    # values. Since usage_id is our primary key (must be unique),
    # we keep the first occurrence and drop the rest.
    # This is a common data cleaning step - real-world data is messy!
    dupes = df.duplicated(subset=["usage_id"], keep="first").sum()
    if dupes > 0:
        print(f"   ⚠️  Found {dupes} duplicate usage_ids in CSV - removing them")
        df = df.drop_duplicates(subset=["usage_id"], keep="first")
    print(f"   ✅ {len(df)} rows | {list(df.columns)}")
    return df


def load_support_tickets():
    """Load and clean support_tickets.csv"""
    print("\n📂 Loading support_tickets.csv...")
    df = pd.read_csv(Config.RAW_DATA_DIR / "ravenstack_support_tickets.csv")
    df["submitted_at"] = pd.to_datetime(df["submitted_at"])
    df["closed_at"] = pd.to_datetime(df["closed_at"], errors="coerce")
    df["escalation_flag"] = df["escalation_flag"].astype(bool)
    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")
    print(f"   ✅ {len(df)} rows | {list(df.columns)}")
    return df


def load_churn_events():
    """Load and clean churn_events.csv"""
    print("\n📂 Loading churn_events.csv...")
    df = pd.read_csv(Config.RAW_DATA_DIR / "ravenstack_churn_events.csv")
    df["churn_date"] = pd.to_datetime(df["churn_date"]).dt.date
    df["refund_amount_usd"] = df["refund_amount_usd"].astype(float)
    for col in ["preceding_upgrade_flag", "preceding_downgrade_flag",
                "is_reactivation"]:
        df[col] = df[col].astype(bool)
    print(f"   ✅ {len(df)} rows | {list(df.columns)}")
    return df


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Insert data using raw SQL (most reliable method)
# ══════════════════════════════════════════════════════════════════════
# WHY RAW SQL INSTEAD OF pandas to_sql()?
#   pandas to_sql() with the default method tries to build a massive
#   parameterized INSERT that can exceed PostgreSQL's parameter limit.
#   Even with chunksize, some pandas/SQLAlchemy version combos still
#   pack too many parameters.
#
#   Instead, we use executemany() directly on the SQLAlchemy connection.
#   This sends one row at a time to the database driver, which batches
#   them efficiently under the hood using psycopg2's fast executemany.
#   It NEVER hits parameter limits because each INSERT has exactly
#   one row of parameters.

def insert_data(df, table_name):
    """
    Insert a DataFrame into PostgreSQL using raw executemany.

    This approach:
    - Builds a simple INSERT INTO ... VALUES (:col1, :col2, ...) statement
    - Passes all rows as a list of dicts to executemany()
    - psycopg2 handles batching internally (very efficient)
    - NEVER exceeds parameter limits
    """
    if len(df) == 0:
        print(f"   ⚠️ No data to insert into '{table_name}'")
        return

    # Build column list and placeholder list from DataFrame columns
    columns = df.columns.tolist()
    col_str = ", ".join(columns)
    val_str = ", ".join([f":{col}" for col in columns])
    sql = text(f"INSERT INTO {table_name} ({col_str}) VALUES ({val_str})")

    # Convert DataFrame to list of dicts (one dict per row)
    # .where(pd.notnull(df), None) replaces NaN with None (= SQL NULL)
    records = df.where(pd.notnull(df), None).to_dict(orient="records")

    # Insert in chunks of 500 for progress feedback
    chunk_size = 500
    total = len(records)

    with engine.begin() as conn:
        for i in range(0, total, chunk_size):
            chunk = records[i:i + chunk_size]
            conn.execute(sql, chunk)
            loaded = min(i + chunk_size, total)
            print(f"   ... {loaded}/{total} rows inserted", end="\r")

    print(f"   ✅ Inserted {total} rows into '{table_name}'       ")


# ══════════════════════════════════════════════════════════════════════
# STEP 5: Verify the load
# ══════════════════════════════════════════════════════════════════════

def verify_load():
    """Check that all tables have the expected row counts."""
    print("\n🔍 Verifying data load...")

    expected = {
        "accounts": 500,
        "subscriptions": 5000,
        "feature_usage": 24979,  # 25000 minus 21 duplicate usage_ids in raw CSV
        "support_tickets": 2000,
        "churn_events": 600,
    }

    all_good = True
    with engine.connect() as conn:
        for table_name, expected_count in expected.items():
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            actual_count = result.scalar()
            status = "✅" if actual_count == expected_count else "⚠️"
            if actual_count != expected_count:
                all_good = False
            print(f"   {status} {table_name}: {actual_count} rows (expected {expected_count})")

    return all_good


# ══════════════════════════════════════════════════════════════════════
# STEP 6: Run the full pipeline
# ══════════════════════════════════════════════════════════════════════

def run_ingestion():
    """
    Execute the complete data ingestion pipeline.

    Pipeline order:
    1. Test connection (fail fast if DB is unreachable)
    2. Drop all existing tables (clean slate)
    3. Create fresh tables with proper schema
    4. Read and clean CSV files
    5. Insert data (parents before children due to FK constraints)
    6. Verify row counts
    """
    print("=" * 60)
    print("🚀 PHASE 1: DATA INGESTION PIPELINE")
    print("=" * 60)

    # ── 1. Test database connection ──────────────────────────────────
    print("\n📡 Testing database connection...")
    if not test_connection():
        print("\n❌ Cannot proceed without database connection. Exiting.")
        return False

    # ── 2. Drop all existing tables ──────────────────────────────────
    drop_all_tables()

    # ── 3. Create fresh table schemas ────────────────────────────────
    create_all_tables()

    # ── 4. Read and clean CSV files ──────────────────────────────────
    print("\n" + "=" * 60)
    print("📥 LOADING CSV FILES")
    print("=" * 60)

    df_accounts = load_accounts()
    df_subscriptions = load_subscriptions()
    df_feature_usage = load_feature_usage()
    df_support_tickets = load_support_tickets()
    df_churn_events = load_churn_events()

    # ── 5. Insert into PostgreSQL ────────────────────────────────────
    # ORDER MATTERS! Parent tables first, then children.
    # accounts must exist before subscriptions (which references it)
    # subscriptions must exist before feature_usage (which references it)
    print("\n" + "=" * 60)
    print("💾 INSERTING INTO POSTGRESQL")
    print("=" * 60)

    print("\n🔄 accounts (parent table)...")
    insert_data(df_accounts, "accounts")

    print("\n🔄 subscriptions (references accounts)...")
    insert_data(df_subscriptions, "subscriptions")

    print("\n🔄 feature_usage (references subscriptions)...")
    insert_data(df_feature_usage, "feature_usage")

    print("\n🔄 support_tickets (references accounts)...")
    insert_data(df_support_tickets, "support_tickets")

    print("\n🔄 churn_events (references accounts)...")
    insert_data(df_churn_events, "churn_events")

    # ── 6. Verify ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ VERIFICATION")
    print("=" * 60)

    success = verify_load()

    if success:
        print("\n" + "=" * 60)
        print("🎉 DATA INGESTION COMPLETE!")
        print("=" * 60)
        print("All 5 tables loaded into PostgreSQL successfully.")
        print("You can now query them in pgAdmin or from Python.")
        print("\nNext step: Phase 2 - Feature Engineering")
    else:
        print("\n⚠️ Some tables have unexpected row counts.")

    return success


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    run_ingestion()