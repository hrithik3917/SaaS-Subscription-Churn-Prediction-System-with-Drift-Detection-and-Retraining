"""
build_features.py - Phase 2: Feature Engineering

Reads the 5 raw tables from PostgreSQL and builds a single feature
table (one row per account) for churn prediction. Features are grouped
into four categories matching the project architecture:

    1. Account features      — tenure, plan tier, seats, industry
    2. Subscription features — MRR, billing patterns, upgrades/downgrades
    3. Usage features        — product engagement, errors, beta adoption
    4. Support features      — ticket volume, resolution time, satisfaction

The resulting table is saved to:
    - PostgreSQL: `churn_features` table
    - CSV snapshot: data/processed/churn_features_v1.csv

Usage:
    python -m src.feature_engineering.build_features
"""

import pandas as pd
import numpy as np
from sqlalchemy import text
from src.utils.config import Config
from src.utils.db import engine


# ---------------------------------------------------------------------------
# Data loading — read raw tables from PostgreSQL
# ---------------------------------------------------------------------------

def load_raw_tables():
    """
    Read raw tables needed for feature engineering from PostgreSQL.

    Note: churn_events is intentionally excluded — it contains post-churn
    data (reason codes, refunds) that would cause data leakage if used
    as model features. The churn_flag target comes from the accounts table.
    """
    with engine.connect() as conn:
        accounts = pd.read_sql("SELECT * FROM accounts", conn)
        subscriptions = pd.read_sql("SELECT * FROM subscriptions", conn)
        feature_usage = pd.read_sql("SELECT * FROM feature_usage", conn)
        support_tickets = pd.read_sql("SELECT * FROM support_tickets", conn)

    return accounts, subscriptions, feature_usage, support_tickets


# ---------------------------------------------------------------------------
# Feature builders — one function per category
# ---------------------------------------------------------------------------

def build_account_features(accounts):
    """
    Account-level features derived from the accounts table.

    Features:
        - tenure_days: days since signup (how long they've been a customer)
        - seats: licensed user count
        - is_trial: whether currently on trial
        - plan_tier: encoded as numeric (Basic=0, Pro=1, Enterprise=2)
        - industry: one-hot encoded (creates a column per industry)
        - referral_source: one-hot encoded
    """
    df = accounts[["account_id", "seats", "is_trial", "plan_tier",
                    "industry", "referral_source", "signup_date", "churn_flag"]].copy()

    # Tenure: days between signup and the latest date in the dataset
    reference_date = pd.Timestamp("2024-12-31")
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["tenure_days"] = (reference_date - df["signup_date"]).dt.days
    df = df.drop(columns=["signup_date"])

    # Encode plan_tier as ordinal (reflects plan hierarchy)
    plan_map = {"Basic": 0, "Pro": 1, "Enterprise": 2}
    df["plan_tier_encoded"] = df["plan_tier"].map(plan_map)
    df = df.drop(columns=["plan_tier"])

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=["industry", "referral_source"],
                        prefix=["industry", "referral"], dtype=int)

    # Convert boolean to int for model compatibility
    df["is_trial"] = df["is_trial"].astype(int)
    df["churn_flag"] = df["churn_flag"].astype(int)

    return df


def build_subscription_features(subscriptions):
    """
    Billing and subscription behavior features.

    Features:
        - total_subscriptions: number of subscription records per account
        - active_subscriptions: subscriptions with no end_date
        - avg_mrr: average monthly recurring revenue
        - total_arr: total annual recurring revenue
        - avg_seats_per_sub: average seats across subscriptions
        - has_upgrade: whether any subscription was upgraded
        - has_downgrade: whether any subscription was downgraded
        - trial_ratio: fraction of subscriptions that were trials
        - auto_renew_ratio: fraction with auto-renew enabled
        - monthly_billing_ratio: fraction billed monthly vs annually
    """
    df = subscriptions.copy()

    agg = df.groupby("account_id").agg(
        total_subscriptions=("subscription_id", "count"),
        active_subscriptions=("end_date", lambda x: x.isna().sum()),
        avg_mrr=("mrr_amount", "mean"),
        total_arr=("arr_amount", "sum"),
        avg_seats_per_sub=("seats", "mean"),
        has_upgrade=("upgrade_flag", "max"),
        has_downgrade=("downgrade_flag", "max"),
        trial_ratio=("is_trial", "mean"),
        auto_renew_ratio=("auto_renew_flag", "mean"),
    ).reset_index()

    # Monthly vs annual billing ratio
    billing = df.groupby("account_id")["billing_frequency"].apply(
        lambda x: (x == "monthly").mean()
    ).reset_index(name="monthly_billing_ratio")

    agg = agg.merge(billing, on="account_id", how="left")

    # Convert boolean aggregates to int
    agg["has_upgrade"] = agg["has_upgrade"].astype(int)
    agg["has_downgrade"] = agg["has_downgrade"].astype(int)

    return agg


def build_usage_features(feature_usage, subscriptions):
    """
    Product engagement features from usage logs.

    We first map usage records to account_id via the subscriptions table,
    then aggregate per account.

    Features:
        - total_usage_events: number of usage log entries
        - total_usage_minutes: total time spent using the product
        - avg_usage_count: average feature interactions per event
        - unique_features_used: number of distinct features used
        - total_errors: total error count across all usage
        - error_rate: errors per usage event
        - beta_feature_ratio: fraction of usage on beta features
        - avg_daily_usage_mins: average usage minutes per active day
    """
    # Map subscription_id to account_id
    sub_account = subscriptions[["subscription_id", "account_id"]].drop_duplicates()
    df = feature_usage.merge(sub_account, on="subscription_id", how="left")

    agg = df.groupby("account_id").agg(
        total_usage_events=("usage_id", "count"),
        total_usage_secs=("usage_duration_secs", "sum"),
        avg_usage_count=("usage_count", "mean"),
        unique_features_used=("feature_name", "nunique"),
        total_errors=("error_count", "sum"),
        beta_feature_ratio=("is_beta_feature", "mean"),
        active_days=("usage_date", "nunique"),
    ).reset_index()

    # Convert seconds to minutes for readability
    agg["total_usage_minutes"] = (agg["total_usage_secs"] / 60).round(2)
    agg = agg.drop(columns=["total_usage_secs"])

    # Error rate: errors per usage event (0 if no events)
    agg["error_rate"] = (
        agg["total_errors"] / agg["total_usage_events"]
    ).fillna(0).round(4)

    # Average daily usage in minutes
    agg["avg_daily_usage_mins"] = (
        agg["total_usage_minutes"] / agg["active_days"]
    ).fillna(0).round(2)

    agg = agg.drop(columns=["active_days"])

    return agg


def build_support_features(support_tickets):
    """
    Customer support interaction features.

    Features:
        - ticket_count: total support tickets filed
        - avg_resolution_hours: average time to resolve a ticket
        - avg_first_response_mins: average time to first response
        - avg_satisfaction: average satisfaction score (1-5)
        - escalation_count: number of escalated tickets
        - escalation_rate: fraction of tickets that were escalated
        - high_priority_ratio: fraction of tickets marked high/urgent
    """
    df = support_tickets.copy()

    agg = df.groupby("account_id").agg(
        ticket_count=("ticket_id", "count"),
        avg_resolution_hours=("resolution_time_hours", "mean"),
        avg_first_response_mins=("first_response_time_minutes", "mean"),
        avg_satisfaction=("satisfaction_score", "mean"),
        escalation_count=("escalation_flag", "sum"),
    ).reset_index()

    # Escalation rate
    agg["escalation_rate"] = (
        agg["escalation_count"] / agg["ticket_count"]
    ).fillna(0).round(4)

    # High/urgent priority ratio
    priority_high = df[df["priority"].isin(["high", "urgent"])].groupby(
        "account_id"
    )["ticket_id"].count().reset_index(name="high_priority_tickets")

    agg = agg.merge(priority_high, on="account_id", how="left")
    agg["high_priority_tickets"] = agg["high_priority_tickets"].fillna(0).astype(int)
    agg["high_priority_ratio"] = (
        agg["high_priority_tickets"] / agg["ticket_count"]
    ).fillna(0).round(4)

    agg = agg.drop(columns=["high_priority_tickets"])

    # Round float columns
    agg["avg_resolution_hours"] = agg["avg_resolution_hours"].round(2)
    agg["avg_first_response_mins"] = agg["avg_first_response_mins"].round(2)
    agg["avg_satisfaction"] = agg["avg_satisfaction"].round(2)

    return agg


# ---------------------------------------------------------------------------
# Feature assembly — merge all feature groups into one table
# ---------------------------------------------------------------------------

def assemble_features(account_ft, subscription_ft, usage_ft, support_ft):
    """
    Merge all feature groups into a single DataFrame with one row per account.

    Left joins ensure we keep all accounts even if they have no subscriptions,
    usage, or support tickets (those columns get filled with 0).
    """
    df = account_ft.copy()
    df = df.merge(subscription_ft, on="account_id", how="left")
    df = df.merge(usage_ft, on="account_id", how="left")
    df = df.merge(support_ft, on="account_id", how="left")

    # Fill NaN with 0 for accounts that had no activity in a category
    # (e.g., no support tickets → ticket_count should be 0, not NaN)
    cols_to_fill = df.columns.difference(["account_id"])
    df[cols_to_fill] = df[cols_to_fill].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Storage — save to PostgreSQL and CSV
# ---------------------------------------------------------------------------

def save_features(df):
    """
    Persist the feature table to:
      1. PostgreSQL `churn_features` table (replaces if exists)
      2. CSV snapshot in data/processed/ for version tracking
    """
    params = Config.load_params()
    version = params["features"]["version"]

    # Save to PostgreSQL
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS churn_features"))
    df.to_sql("churn_features", engine, if_exists="replace", index=False)

    # Save CSV snapshot
    output_dir = Config.PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"churn_features_{version}.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_feature_engineering():
    """
    Execute the full feature engineering pipeline:
      1. Load raw tables from PostgreSQL
      2. Build features per category
      3. Merge into single table
      4. Save to PostgreSQL and CSV
    """
    # Load raw data (churn_events excluded — see load_raw_tables docstring)
    accounts, subscriptions, feature_usage, support_tickets = load_raw_tables()

    # Build feature groups
    account_ft = build_account_features(accounts)
    subscription_ft = build_subscription_features(subscriptions)
    usage_ft = build_usage_features(feature_usage, subscriptions)
    support_ft = build_support_features(support_tickets)

    # Assemble into one table
    features = assemble_features(account_ft, subscription_ft, usage_ft, support_ft)

    # Save outputs
    save_features(features)

    return features


if __name__ == "__main__":
    run_feature_engineering()