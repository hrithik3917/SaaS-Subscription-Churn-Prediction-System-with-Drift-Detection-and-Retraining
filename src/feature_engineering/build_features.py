"""
build_features.py - Phase 2: Feature Engineering (v2 — Time-Windowed)

Improved version that builds time-aware features using observation and
prediction windows from params.yaml. Recent behavior (last 30/90 days)
is far more predictive of churn than all-time aggregates.

Key improvements over v1:
    - Time-windowed aggregations (recent vs older periods)
    - Trend features (is usage increasing or decreasing?)
    - Recency features (days since last activity)
    - Interaction features (combinations that amplify signal)
    - Subscription lifecycle features (velocity of changes)

Usage:
    python -m src.feature_engineering.build_features
"""

import pandas as pd
import numpy as np
from sqlalchemy import text
from src.utils.config import Config
from src.utils.db import engine


# ---------------------------------------------------------------------------
# Reference date — the "today" for all time calculations
# ---------------------------------------------------------------------------
# Using the max date in the dataset as our reference point.
# In production, this would be datetime.now().
REFERENCE_DATE = pd.Timestamp("2024-12-31")


# ---------------------------------------------------------------------------
# Data loading
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
# Account features
# ---------------------------------------------------------------------------

def build_account_features(accounts):
    """
    Customer identity and tenure features.

    Tenure is one of the strongest churn predictors — new customers
    churn at much higher rates than established ones.
    """
    df = accounts[["account_id", "seats", "is_trial", "plan_tier",
                    "industry", "referral_source", "signup_date", "churn_flag"]].copy()

    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["tenure_days"] = (REFERENCE_DATE - df["signup_date"]).dt.days

    # Tenure buckets — captures non-linear relationship with churn
    # (very new and very old customers behave differently)
    df["is_new_customer"] = (df["tenure_days"] <= 90).astype(int)
    df["is_established"] = (df["tenure_days"] >= 365).astype(int)

    df = df.drop(columns=["signup_date"])

    # Ordinal encoding for plan tier
    plan_map = {"Basic": 0, "Pro": 1, "Enterprise": 2}
    df["plan_tier_encoded"] = df["plan_tier"].map(plan_map)
    df = df.drop(columns=["plan_tier"])

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=["industry", "referral_source"],
                        prefix=["industry", "referral"], dtype=int)

    df["is_trial"] = df["is_trial"].astype(int)
    df["churn_flag"] = df["churn_flag"].astype(int)

    return df


# ---------------------------------------------------------------------------
# Subscription features (time-windowed)
# ---------------------------------------------------------------------------

def build_subscription_features(subscriptions):
    """
    Billing behavior features with time awareness.

    New in v2:
        - recent_mrr vs older_mrr (MRR trend)
        - subscription_velocity (how fast they add/drop subscriptions)
        - latest_plan_tier (current plan, not historical average)
        - days_since_last_subscription
        - mrr_change_ratio (recent vs older MRR — declining = churn risk)
    """
    params = Config.load_params()
    window = params["data"]["observation_window_days"]  # 90 days

    df = subscriptions.copy()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])

    cutoff_recent = REFERENCE_DATE - pd.Timedelta(days=window)

    # Split into recent (last 90 days) and older subscriptions
    recent = df[df["start_date"] >= cutoff_recent]
    older = df[df["start_date"] < cutoff_recent]

    # --- All-time aggregations ---
    all_time = df.groupby("account_id").agg(
        total_subscriptions=("subscription_id", "count"),
        active_subscriptions=("end_date", lambda x: x.isna().sum()),
        avg_mrr=("mrr_amount", "mean"),
        max_mrr=("mrr_amount", "max"),
        total_arr=("arr_amount", "sum"),
        avg_seats_per_sub=("seats", "mean"),
        has_upgrade=("upgrade_flag", "max"),
        has_downgrade=("downgrade_flag", "max"),
        trial_ratio=("is_trial", "mean"),
        auto_renew_ratio=("auto_renew_flag", "mean"),
    ).reset_index()

    # --- Recent window aggregations ---
    recent_agg = recent.groupby("account_id").agg(
        recent_sub_count=("subscription_id", "count"),
        recent_avg_mrr=("mrr_amount", "mean"),
        recent_has_upgrade=("upgrade_flag", "max"),
        recent_has_downgrade=("downgrade_flag", "max"),
    ).reset_index()

    # --- Older window aggregations (for trend comparison) ---
    older_agg = older.groupby("account_id").agg(
        older_avg_mrr=("mrr_amount", "mean"),
    ).reset_index()

    # --- Latest subscription info per account ---
    latest_sub = df.sort_values("start_date").groupby("account_id").last().reset_index()
    latest_info = latest_sub[["account_id", "start_date", "plan_tier", "billing_frequency"]].copy()
    latest_info["days_since_last_sub"] = (REFERENCE_DATE - latest_info["start_date"]).dt.days

    # Encode latest plan tier
    plan_map = {"Basic": 0, "Pro": 1, "Enterprise": 2}
    latest_info["latest_plan_tier"] = latest_info["plan_tier"].map(plan_map)
    latest_info["is_monthly_billing"] = (latest_info["billing_frequency"] == "monthly").astype(int)
    latest_info = latest_info[["account_id", "days_since_last_sub", "latest_plan_tier", "is_monthly_billing"]]

    # --- Merge everything ---
    result = all_time.merge(recent_agg, on="account_id", how="left")
    result = result.merge(older_agg, on="account_id", how="left")
    result = result.merge(latest_info, on="account_id", how="left")

    # --- Derived features ---
    # MRR trend: ratio of recent MRR to older MRR (< 1 means declining)
    result["mrr_change_ratio"] = (
        result["recent_avg_mrr"] / result["older_avg_mrr"].replace(0, np.nan)
    ).fillna(1.0)

    # Cap extreme values
    result["mrr_change_ratio"] = result["mrr_change_ratio"].clip(0, 5)

    # Subscription velocity: recent subs as fraction of total
    result["sub_velocity"] = (
        result["recent_sub_count"] / result["total_subscriptions"]
    ).fillna(0)

    # Convert boolean aggregates
    result["has_upgrade"] = result["has_upgrade"].astype(int)
    result["has_downgrade"] = result["has_downgrade"].astype(int)
    result["recent_has_upgrade"] = result["recent_has_upgrade"].fillna(0).astype(int)
    result["recent_has_downgrade"] = result["recent_has_downgrade"].fillna(0).astype(int)

    # Drop intermediate columns
    result = result.drop(columns=["recent_avg_mrr", "older_avg_mrr", "recent_sub_count"])

    return result


# ---------------------------------------------------------------------------
# Usage features (time-windowed with trends)
# ---------------------------------------------------------------------------

def build_usage_features(feature_usage, subscriptions):
    """
    Product engagement features with time awareness and trends.

    New in v2:
        - recent vs older usage comparison (trend detection)
        - usage_trend_ratio (declining usage = strong churn signal)
        - days_since_last_usage (recency)
        - recent_error_rate (recent errors matter more)
        - feature_diversity_trend (exploring fewer features = disengaging)
    """
    params = Config.load_params()
    window = params["data"]["observation_window_days"]  # 90 days

    # Map usage to accounts via subscriptions
    sub_account = subscriptions[["subscription_id", "account_id"]].drop_duplicates()
    df = feature_usage.merge(sub_account, on="subscription_id", how="left")
    df["usage_date"] = pd.to_datetime(df["usage_date"])

    cutoff_recent = REFERENCE_DATE - pd.Timedelta(days=window)
    cutoff_older = REFERENCE_DATE - pd.Timedelta(days=window * 2)

    recent = df[df["usage_date"] >= cutoff_recent]
    older = df[(df["usage_date"] >= cutoff_older) & (df["usage_date"] < cutoff_recent)]

    # --- All-time aggregations ---
    all_time = df.groupby("account_id").agg(
        total_usage_events=("usage_id", "count"),
        total_usage_secs=("usage_duration_secs", "sum"),
        avg_usage_count=("usage_count", "mean"),
        unique_features_used=("feature_name", "nunique"),
        total_errors=("error_count", "sum"),
        beta_feature_ratio=("is_beta_feature", "mean"),
        total_active_days=("usage_date", "nunique"),
    ).reset_index()

    all_time["total_usage_minutes"] = (all_time["total_usage_secs"] / 60).round(2)
    all_time["error_rate"] = (
        all_time["total_errors"] / all_time["total_usage_events"]
    ).fillna(0).round(4)
    all_time["avg_daily_usage_mins"] = (
        all_time["total_usage_minutes"] / all_time["total_active_days"]
    ).fillna(0).round(2)
    all_time = all_time.drop(columns=["total_usage_secs", "total_active_days"])

    # --- Recent window ---
    recent_agg = recent.groupby("account_id").agg(
        recent_usage_events=("usage_id", "count"),
        recent_usage_secs=("usage_duration_secs", "sum"),
        recent_errors=("error_count", "sum"),
        recent_features_used=("feature_name", "nunique"),
        recent_active_days=("usage_date", "nunique"),
    ).reset_index()

    recent_agg["recent_usage_minutes"] = (recent_agg["recent_usage_secs"] / 60).round(2)
    recent_agg["recent_error_rate"] = (
        recent_agg["recent_errors"] / recent_agg["recent_usage_events"]
    ).fillna(0).round(4)
    recent_agg["recent_avg_daily_mins"] = (
        recent_agg["recent_usage_minutes"] / recent_agg["recent_active_days"]
    ).fillna(0).round(2)
    recent_agg = recent_agg.drop(columns=["recent_usage_secs", "recent_active_days"])

    # --- Older window (for trend) ---
    older_agg = older.groupby("account_id").agg(
        older_usage_minutes=("usage_duration_secs", lambda x: (x.sum() / 60)),
        older_features_used=("feature_name", "nunique"),
    ).reset_index()

    # --- Recency: days since last usage ---
    last_usage = df.groupby("account_id")["usage_date"].max().reset_index()
    last_usage["days_since_last_usage"] = (REFERENCE_DATE - last_usage["usage_date"]).dt.days
    last_usage = last_usage[["account_id", "days_since_last_usage"]]

    # --- Merge ---
    result = all_time.merge(recent_agg, on="account_id", how="left")
    result = result.merge(older_agg, on="account_id", how="left")
    result = result.merge(last_usage, on="account_id", how="left")

    # --- Trend features ---
    # Usage trend: recent minutes / older minutes (< 1 = declining)
    result["usage_trend_ratio"] = (
        result["recent_usage_minutes"] / result["older_usage_minutes"].replace(0, np.nan)
    ).fillna(1.0).clip(0, 5)

    # Feature diversity trend
    result["feature_diversity_trend"] = (
        result["recent_features_used"] / result["older_features_used"].replace(0, np.nan)
    ).fillna(1.0).clip(0, 5)

    # Drop intermediate columns
    result = result.drop(columns=["older_usage_minutes", "older_features_used",
                                   "recent_errors"])

    return result


# ---------------------------------------------------------------------------
# Support features (time-windowed)
# ---------------------------------------------------------------------------

def build_support_features(support_tickets):
    """
    Support interaction features with recency and trend awareness.

    New in v2:
        - recent_ticket_count (tickets in last 90 days)
        - ticket_trend_ratio (increasing tickets = growing frustration)
        - days_since_last_ticket
        - recent_escalation_rate
        - recent_satisfaction (recent experience matters most)
        - unresolved_ratio (fraction of tickets still open)
    """
    params = Config.load_params()
    window = params["data"]["observation_window_days"]

    df = support_tickets.copy()
    df["submitted_at"] = pd.to_datetime(df["submitted_at"])
    df["closed_at"] = pd.to_datetime(df["closed_at"])

    cutoff_recent = REFERENCE_DATE - pd.Timedelta(days=window)
    cutoff_older = REFERENCE_DATE - pd.Timedelta(days=window * 2)

    recent = df[df["submitted_at"] >= cutoff_recent]
    older = df[(df["submitted_at"] >= cutoff_older) & (df["submitted_at"] < cutoff_recent)]

    # --- All-time ---
    all_time = df.groupby("account_id").agg(
        ticket_count=("ticket_id", "count"),
        avg_resolution_hours=("resolution_time_hours", "mean"),
        avg_first_response_mins=("first_response_time_minutes", "mean"),
        avg_satisfaction=("satisfaction_score", "mean"),
        escalation_count=("escalation_flag", "sum"),
    ).reset_index()

    all_time["escalation_rate"] = (
        all_time["escalation_count"] / all_time["ticket_count"]
    ).fillna(0).round(4)

    # High priority ratio
    high_priority = df[df["priority"].isin(["high", "urgent"])].groupby(
        "account_id"
    )["ticket_id"].count().reset_index(name="high_priority_tickets")

    all_time = all_time.merge(high_priority, on="account_id", how="left")
    all_time["high_priority_tickets"] = all_time["high_priority_tickets"].fillna(0)
    all_time["high_priority_ratio"] = (
        all_time["high_priority_tickets"] / all_time["ticket_count"]
    ).fillna(0).round(4)
    all_time = all_time.drop(columns=["high_priority_tickets"])

    # Unresolved tickets
    unresolved = df[df["closed_at"].isna()].groupby("account_id")["ticket_id"].count().reset_index(
        name="unresolved_tickets"
    )
    all_time = all_time.merge(unresolved, on="account_id", how="left")
    all_time["unresolved_tickets"] = all_time["unresolved_tickets"].fillna(0).astype(int)
    all_time["unresolved_ratio"] = (
        all_time["unresolved_tickets"] / all_time["ticket_count"]
    ).fillna(0).round(4)

    # --- Recent window ---
    recent_agg = recent.groupby("account_id").agg(
        recent_ticket_count=("ticket_id", "count"),
        recent_avg_resolution_hours=("resolution_time_hours", "mean"),
        recent_avg_satisfaction=("satisfaction_score", "mean"),
        recent_escalation_count=("escalation_flag", "sum"),
    ).reset_index()

    recent_agg["recent_escalation_rate"] = (
        recent_agg["recent_escalation_count"] / recent_agg["recent_ticket_count"]
    ).fillna(0).round(4)
    recent_agg = recent_agg.drop(columns=["recent_escalation_count"])

    # --- Older window (for trend) ---
    older_agg = older.groupby("account_id").agg(
        older_ticket_count=("ticket_id", "count"),
    ).reset_index()

    # --- Recency ---
    last_ticket = df.groupby("account_id")["submitted_at"].max().reset_index()
    last_ticket["days_since_last_ticket"] = (REFERENCE_DATE - last_ticket["submitted_at"]).dt.days
    last_ticket = last_ticket[["account_id", "days_since_last_ticket"]]

    # --- Merge ---
    result = all_time.merge(recent_agg, on="account_id", how="left")
    result = result.merge(older_agg, on="account_id", how="left")
    result = result.merge(last_ticket, on="account_id", how="left")

    # --- Trend ---
    result["ticket_trend_ratio"] = (
        result["recent_ticket_count"] / result["older_ticket_count"].replace(0, np.nan)
    ).fillna(1.0).clip(0, 10)

    result = result.drop(columns=["older_ticket_count"])

    # Round float columns
    for col in ["avg_resolution_hours", "avg_first_response_mins", "avg_satisfaction",
                "recent_avg_resolution_hours", "recent_avg_satisfaction"]:
        if col in result.columns:
            result[col] = result[col].round(2)

    return result


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

def build_interaction_features(df):
    """
    Cross-category features that combine signals from multiple domains.

    These capture compound effects that individual features miss.
    For example: low usage ALONE might be okay (light user), and
    high ticket count ALONE might be okay (power user with questions).
    But low usage + high tickets = frustrated customer about to leave.
    """
    # Low usage + high tickets = frustrated and disengaging
    if "total_usage_minutes" in df.columns and "ticket_count" in df.columns:
        usage_norm = df["total_usage_minutes"] / (df["total_usage_minutes"].max() + 1)
        ticket_norm = df["ticket_count"] / (df["ticket_count"].max() + 1)
        df["frustration_score"] = ((1 - usage_norm) * ticket_norm).round(4)

    # Revenue at risk: high MRR customers with declining usage
    if "avg_mrr" in df.columns and "usage_trend_ratio" in df.columns:
        mrr_norm = df["avg_mrr"] / (df["avg_mrr"].max() + 1)
        declining = (1 - df["usage_trend_ratio"].clip(0, 1))
        df["revenue_risk_score"] = (mrr_norm * declining).round(4)

    # Support burden: tickets per tenure day (normalized complaint rate)
    if "ticket_count" in df.columns and "tenure_days" in df.columns:
        df["tickets_per_tenure_day"] = (
            df["ticket_count"] / df["tenure_days"].replace(0, 1)
        ).round(6)

    # Engagement depth: usage minutes per seat (are all seats active?)
    if "total_usage_minutes" in df.columns and "seats" in df.columns:
        df["usage_per_seat"] = (
            df["total_usage_minutes"] / df["seats"].replace(0, 1)
        ).round(2)

    return df


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def assemble_features(account_ft, subscription_ft, usage_ft, support_ft):
    """Merge all feature groups into a single table with one row per account."""
    df = account_ft.copy()
    df = df.merge(subscription_ft, on="account_id", how="left")
    df = df.merge(usage_ft, on="account_id", how="left")
    df = df.merge(support_ft, on="account_id", how="left")

    # Fill NaN with 0 for accounts with no activity in a category
    cols_to_fill = df.columns.difference(["account_id"])
    df[cols_to_fill] = df[cols_to_fill].fillna(0)

    # Add interaction features (needs columns from multiple groups)
    df = build_interaction_features(df)

    return df


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def save_features(df):
    """Save feature table to PostgreSQL and CSV snapshot."""
    params = Config.load_params()
    version = params["features"]["version"]

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS churn_features"))
    df.to_sql("churn_features", engine, if_exists="replace", index=False)

    output_dir = Config.PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"churn_features_{version}.csv", index=False)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_feature_engineering():
    """
    Execute the full feature engineering pipeline:
      1. Load raw tables from PostgreSQL
      2. Build features per category (with time windows)
      3. Merge into single table
      4. Add interaction features
      5. Save to PostgreSQL and CSV
    """
    accounts, subscriptions, feature_usage, support_tickets = load_raw_tables()

    account_ft = build_account_features(accounts)
    subscription_ft = build_subscription_features(subscriptions)
    usage_ft = build_usage_features(feature_usage, subscriptions)
    support_ft = build_support_features(support_tickets)

    features = assemble_features(account_ft, subscription_ft, usage_ft, support_ft)

    save_features(features)

    return features


if __name__ == "__main__":
    run_feature_engineering()