"""
load_data.py - Phase 1: Data Ingestion
 
Reads raw CSV files, cleans data types, and loads them into
PostgreSQL with proper schema (PKs, FKs, column types).
 
The script is idempotent: it drops and recreates tables on every run,
ensuring a consistent state regardless of previous executions.

HOW TO RUN:
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
# STEP 1: DROP existing tables 
# ══════════════════════════════════════════════════════════════════════

def drop_all_tables():
    """
    We drop child tables before parent tables because PostgreSQL
    won't let you drop a parent table that children still reference.
    """
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS feature_usage CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS support_tickets CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS churn_events CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS subscriptions CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS accounts CASCADE"))


# ══════════════════════════════════════════════════════════════════════
# STEP 2: CREATE fresh table schemas
# ══════════════════════════════════════════════════════════════════════

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

    metadata.create_all(engine)

# ══════════════════════════════════════════════════════════════════════
# STEP 3: Read and clean each CSV file
# ══════════════════════════════════════════════════════════════════════

def load_accounts():
    df = pd.read_csv(Config.RAW_DATA_DIR / "ravenstack_accounts.csv")
    df["signup_date"] = pd.to_datetime(df["signup_date"]).dt.date
    df["is_trial"] = df["is_trial"].astype(bool)
    df["churn_flag"] = df["churn_flag"].astype(bool)
    return df


def load_subscriptions():
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
    return df


def load_feature_usage():
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
        print(f"Found {dupes} duplicate usage_ids in CSV - removing them")
        df = df.drop_duplicates(subset=["usage_id"], keep="first")
    return df


def load_support_tickets():
    df = pd.read_csv(Config.RAW_DATA_DIR / "ravenstack_support_tickets.csv")
    df["submitted_at"] = pd.to_datetime(df["submitted_at"])
    df["closed_at"] = pd.to_datetime(df["closed_at"], errors="coerce")
    df["escalation_flag"] = df["escalation_flag"].astype(bool)
    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")
    return df


def load_churn_events():
    df = pd.read_csv(Config.RAW_DATA_DIR / "ravenstack_churn_events.csv")
    df["churn_date"] = pd.to_datetime(df["churn_date"]).dt.date
    df["refund_amount_usd"] = df["refund_amount_usd"].astype(float)
    for col in ["preceding_upgrade_flag", "preceding_downgrade_flag",
                "is_reactivation"]:
        df[col] = df[col].astype(bool)
    return df


# ══════════════════════════════════════════════════════════════════════
# STEP 4: Insert data using raw SQL(Database insertion)
# ══════════════════════════════════════════════════════════════════════

def insert_data(df, table_name):
    if len(df) == 0:
        print(f"No data to insert into '{table_name}'")
        return

    columns = df.columns.tolist()
    col_str = ", ".join(columns)
    val_str = ", ".join([f":{col}" for col in columns])
    sql = text(f"INSERT INTO {table_name} ({col_str}) VALUES ({val_str})")

    # Replace NaN with None so PostgreSQL receives proper NULLs
    records = df.where(pd.notnull(df), None).to_dict(orient="records")

    # Insert in chunks of 500 for progress feedback
    chunk_size = 500
    total = len(records)

    with engine.begin() as conn:
        for i in range(0, total, chunk_size):
            chunk = records[i:i + chunk_size]
            conn.execute(sql, chunk)
            loaded = min(i + chunk_size, total)

# ══════════════════════════════════════════════════════════════════════
# STEP 5: Verify the load
# ══════════════════════════════════════════════════════════════════════

EXPECTED_COUNTS = {
    "accounts": 500,
    "subscriptions": 5000,
    "feature_usage": 24979,   # 25000 minus 21 duplicate usage_ids
    "support_tickets": 2000,
    "churn_events": 600,
}
 
 
def verify_load():
    """Return True if all tables have the expected row counts."""
    with engine.connect() as conn:
        for table_name, expected in EXPECTED_COUNTS.items():
            actual = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
            if actual != expected:
                return False
    return True


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

    # ── 1. Test database connection ──────────────────────────────────
    if not test_connection():
        raise ConnectionError("Database is not reachable. Check PostgreSQL and .env settings.")

    # ── 2. Drop all existing tables ──────────────────────────────────
    drop_all_tables()

    # ── 3. Create fresh table schemas ────────────────────────────────
    create_all_tables()

    # ── 4. Read and clean CSV files ──────────────────────────────────

    df_accounts = load_accounts()
    df_subscriptions = load_subscriptions()
    df_feature_usage = load_feature_usage()
    df_support_tickets = load_support_tickets()
    df_churn_events = load_churn_events()

    # ── 5. Insert into PostgreSQL ────────────────────────────────────
    # ORDER MATTERS! Parent tables first, then children.
    # accounts must exist before subscriptions (which references it)
    # subscriptions must exist before feature_usage (which references it)

    insert_data(df_accounts, "accounts")
    insert_data(df_subscriptions, "subscriptions")
    insert_data(df_feature_usage, "feature_usage")
    insert_data(df_support_tickets, "support_tickets")
    insert_data(df_churn_events, "churn_events")

    # ── 6. Verify ────────────────────────────────────────────────────
    success = verify_load()
    if success:
        print("All 5 tables loaded into PostgreSQL successfully.")
       
    else:
        print("\n Some tables have unexpected row counts.")

    return success


if __name__ == "__main__":
    run_ingestion()