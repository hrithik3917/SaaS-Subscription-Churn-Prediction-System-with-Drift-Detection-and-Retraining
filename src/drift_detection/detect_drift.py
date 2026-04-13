"""
detect_drift.py - Phase 5: Data Drift Detection

Compares the statistical distribution of new (incoming) data against
the reference data the model was trained on. If distributions shift
significantly, the model's predictions become unreliable and retraining
is needed.

Detection methods:
    - Kolmogorov-Smirnov test: for numerical features (compares CDFs)
    - Chi-squared test: for categorical/binary features (compares frequencies)

Drift verdict:
    - "no_drift"        — no significant shift detected
    - "moderate_drift"  — some features shifted, monitor closely
    - "significant_drift" — major shift, retraining recommended

Usage:
    python -m src.drift_detection.detect_drift
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from scipy.stats import ks_2samp, chi2_contingency
from sqlalchemy import text
from src.utils.config import Config
from src.utils.db import engine


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_reference_data():
    """
    Load the reference dataset — the data the model was trained on.

    This is the churn_features table saved during Phase 2. Any new data
    will be compared against this baseline to detect distribution shifts.
    """
    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM churn_features", conn)
    return df


def create_simulated_new_data(reference_df, drift_fraction=1.0):
    """
    Simulate incoming data with clearly detectable distribution shifts.

    Uses 250 rows (more statistical power) and more extreme multipliers
    so the KS/chi-squared tests reliably detect drift.

    In production, replace this entire function with actual new data loading.
    """
    np.random.seed(42)
    df = reference_df.drop(columns=["account_id", "churn_flag"]).copy()

    # 250 rows gives the KS test enough power to detect the shifts
    new_data = df.sample(n=min(250, len(df)), random_state=42).copy()

    n_drift = int(len(new_data) * drift_fraction)
    drift_idx = new_data.index[:n_drift]

    # Simulate: severe MRR collapse (customers downgrading en masse)
    if "avg_mrr" in new_data.columns:
        new_data.loc[drift_idx, "avg_mrr"] *= 0.15

    if "max_mrr" in new_data.columns:
        new_data.loc[drift_idx, "max_mrr"] *= 0.15

    # Simulate: support ticket surge (10x normal volume)
    if "ticket_count" in new_data.columns:
        new_data.loc[drift_idx, "ticket_count"] *= 10

    if "recent_ticket_count" in new_data.columns:
        new_data.loc[drift_idx, "recent_ticket_count"] *= 10

    # Simulate: complete product disengagement
    if "total_usage_minutes" in new_data.columns:
        new_data.loc[drift_idx, "total_usage_minutes"] *= 0.05

    if "recent_usage_minutes" in new_data.columns:
        new_data.loc[drift_idx, "recent_usage_minutes"] *= 0.05

    if "usage_trend_ratio" in new_data.columns:
        new_data.loc[drift_idx, "usage_trend_ratio"] *= 0.1

    # Simulate: satisfaction collapse
    if "avg_satisfaction" in new_data.columns:
        new_data.loc[drift_idx, "avg_satisfaction"] *= 0.3

    if "recent_avg_satisfaction" in new_data.columns:
        new_data.loc[drift_idx, "recent_avg_satisfaction"] *= 0.3

    # Simulate: escalation spike
    if "escalation_rate" in new_data.columns:
        new_data.loc[drift_idx, "escalation_rate"] = np.minimum(
            new_data.loc[drift_idx, "escalation_rate"] + 0.7, 1.0
        )

    # Simulate: MRR change ratio collapse (mass downgrades)
    if "mrr_change_ratio" in new_data.columns:
        new_data.loc[drift_idx, "mrr_change_ratio"] *= 0.2

    # Simulate: frustration score spike
    if "frustration_score" in new_data.columns:
        new_data.loc[drift_idx, "frustration_score"] = np.minimum(
            new_data.loc[drift_idx, "frustration_score"] + 0.6, 1.0
        )

    return new_data


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def detect_numerical_drift(reference_col, new_col, significance=0.05):
    """
    Kolmogorov-Smirnov test for numerical features.

    Compares the cumulative distribution functions (CDFs) of two samples.
    A low p-value means the distributions are significantly different.

    Args:
        reference_col: feature values from training data
        new_col: feature values from new data
        significance: p-value threshold (default 0.05 = 95% confidence)

    Returns:
        dict with statistic, p_value, and whether drift was detected
    """
    # Drop NaN values for clean comparison
    ref_clean = reference_col.dropna()
    new_clean = new_col.dropna()

    if len(ref_clean) == 0 or len(new_clean) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "drift_detected": False}

    statistic, p_value = ks_2samp(ref_clean, new_clean)

    return {
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 4),
        "drift_detected": p_value < significance,
    }


def detect_categorical_drift(reference_col, new_col, significance=0.05):
    """
    Chi-squared test for categorical/binary features.

    Compares observed frequencies against expected frequencies.
    Used for columns with few unique values (binary flags, encoded tiers).

    Returns:
        dict with statistic, p_value, and whether drift was detected
    """
    ref_clean = reference_col.dropna()
    new_clean = new_col.dropna()

    if len(ref_clean) == 0 or len(new_clean) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "drift_detected": False}

    # Build a contingency table from both distributions
    all_categories = set(ref_clean.unique()) | set(new_clean.unique())
    ref_counts = ref_clean.value_counts()
    new_counts = new_clean.value_counts()

    # Ensure both have the same categories
    ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
    new_freq = [new_counts.get(cat, 0) for cat in all_categories]

    contingency = np.array([ref_freq, new_freq])

    # Chi-squared requires at least 2 categories with non-zero values
    if contingency.shape[1] < 2 or contingency.sum() == 0:
        return {"statistic": 0.0, "p_value": 1.0, "drift_detected": False}

    try:
        statistic, p_value, _, _ = chi2_contingency(contingency)
        return {
            "statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "drift_detected": p_value < significance,
        }
    except ValueError:
        return {"statistic": 0.0, "p_value": 1.0, "drift_detected": False}


# ---------------------------------------------------------------------------
# Drift analysis
# ---------------------------------------------------------------------------

# Binary/categorical columns use chi-squared; everything else uses KS test
CATEGORICAL_FEATURES = {
    "is_trial", "has_upgrade", "has_downgrade", "plan_tier_encoded",
    "industry_Cybersecurity", "industry_DevTools", "industry_EdTech",
    "industry_FinTech", "industry_HealthTech",
    "referral_ads", "referral_event", "referral_organic",
    "referral_other", "referral_partner",
}


def run_drift_analysis(reference_df, new_data_df, significance=0.05):
    """
    Compare every feature column between reference and new data.

    Args:
        reference_df: training data (without account_id and churn_flag)
        new_data_df: new incoming data (same columns)
        significance: p-value threshold for drift detection

    Returns:
        dict with per-feature results and an overall drift verdict
    """
    # Exclude non-feature columns
    exclude_cols = {"account_id", "churn_flag"}
    feature_cols = [c for c in reference_df.columns if c not in exclude_cols]

    ref_data = reference_df[feature_cols]
    new_data = new_data_df[[c for c in feature_cols if c in new_data_df.columns]]

    feature_results = {}
    drifted_features = []

    for col in new_data.columns:
        if col in CATEGORICAL_FEATURES:
            result = detect_categorical_drift(
                ref_data[col], new_data[col], significance
            )
            result["test"] = "chi_squared"
        else:
            result = detect_numerical_drift(
                ref_data[col], new_data[col], significance
            )
            result["test"] = "kolmogorov_smirnov"

        feature_results[col] = result
        if result["drift_detected"]:
            drifted_features.append(col)

    # Overall verdict based on fraction of features that drifted
    total_features = len(new_data.columns)
    drift_count = len(drifted_features)
    drift_ratio = drift_count / total_features if total_features > 0 else 0

    if drift_ratio >= 0.3:
        verdict = "significant_drift"
    elif drift_ratio >= 0.1:
        verdict = "moderate_drift"
    else:
        verdict = "no_drift"

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_features_tested": total_features,
        "features_with_drift": drift_count,
        "drift_ratio": round(drift_ratio, 4),
        "verdict": verdict,
        "drifted_features": drifted_features,
        "feature_details": feature_results,
    }

    return report


# ---------------------------------------------------------------------------
# Report storage
# ---------------------------------------------------------------------------

def save_drift_report(report):
    """
    Save the drift report as a JSON file in data/interim/.

    JSON format makes it easy to parse programmatically (Phase 6 reads
    this to decide whether to trigger retraining).

    Uses a custom encoder because NumPy types (np.bool_, np.float64)
    are not natively JSON serializable.
    """
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    output_dir = Config.INTERIM_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / "drift_report.json"
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)

    return filepath


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_drift_detection():
    """
    Execute the full drift detection pipeline:
      1. Load reference (training) data
      2. Load or simulate new incoming data
      3. Run statistical tests on every feature
      4. Generate and save drift report
      5. Return the verdict for Phase 6 to act on
    """
    reference_df = load_reference_data()
    # drift_fraction=1.0 — all sampled rows show the shift
    new_data = create_simulated_new_data(reference_df, drift_fraction=1.0)
    report = run_drift_analysis(reference_df, new_data)
    save_drift_report(report)
    return report


if __name__ == "__main__":
    run_drift_detection()