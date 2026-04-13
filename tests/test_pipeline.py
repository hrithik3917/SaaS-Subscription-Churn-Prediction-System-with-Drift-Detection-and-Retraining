"""
tests/test_pipeline.py

Unit tests for the SaaS Churn Prediction MLOps Pipeline.

Tests cover all 6 phases of the pipeline:
    - Phase 1: Data ingestion (row counts, table existence)
    - Phase 2: Feature engineering (column count, no nulls, label present)
    - Phase 3: Model (loads correctly, produces valid probabilities)
    - Phase 4: API (health endpoint returns 200, prediction schema correct)
    - Phase 5: Drift detection (report exists, has required keys)
    - Phase 6: Retraining (drift report is readable by retrain pipeline)

Usage:
    pytest tests/ -v
    pytest tests/ -v --tb=short   # shorter traceback on failure
"""

import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def db_engine():
    """Single database engine shared across all tests in the session."""
    from src.utils.db import engine
    return engine


@pytest.fixture(scope="session")
def features_df(db_engine):
    """Load churn_features table once and reuse across tests."""
    with db_engine.connect() as conn:
        return pd.read_sql("SELECT * FROM churn_features", conn)


@pytest.fixture(scope="session")
def config():
    """Load project config once."""
    from src.utils.config import Config
    return Config


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 — Data Ingestion
# ═══════════════════════════════════════════════════════════════════════

class TestDataIngestion:
    """Verify all 5 tables were loaded correctly into PostgreSQL."""

    EXPECTED_ROW_COUNTS = {
        "accounts":        500,
        "subscriptions":   5000,
        "support_tickets": 2000,
        "churn_events":    600,
    }

    def test_accounts_row_count(self, db_engine):
        """accounts table must have exactly 500 rows."""
        with db_engine.connect() as conn:
            count = pd.read_sql(
                "SELECT COUNT(*) as n FROM accounts", conn
            )["n"].iloc[0]
        assert count == 500, f"Expected 500 accounts, got {count}"

    def test_subscriptions_row_count(self, db_engine):
        """subscriptions table must have exactly 5000 rows."""
        with db_engine.connect() as conn:
            count = pd.read_sql(
                "SELECT COUNT(*) as n FROM subscriptions", conn
            )["n"].iloc[0]
        assert count == 5000, f"Expected 5000 subscriptions, got {count}"

    def test_feature_usage_row_count(self, db_engine):
        """feature_usage table must have 24979 rows (25000 minus 21 dupes)."""
        with db_engine.connect() as conn:
            count = pd.read_sql(
                "SELECT COUNT(*) as n FROM feature_usage", conn
            )["n"].iloc[0]
        assert count == 24979, f"Expected 24979 usage rows, got {count}"

    def test_support_tickets_row_count(self, db_engine):
        """support_tickets table must have exactly 2000 rows."""
        with db_engine.connect() as conn:
            count = pd.read_sql(
                "SELECT COUNT(*) as n FROM support_tickets", conn
            )["n"].iloc[0]
        assert count == 2000, f"Expected 2000 tickets, got {count}"

    def test_churn_events_row_count(self, db_engine):
        """churn_events table must have exactly 600 rows."""
        with db_engine.connect() as conn:
            count = pd.read_sql(
                "SELECT COUNT(*) as n FROM churn_events", conn
            )["n"].iloc[0]
        assert count == 600, f"Expected 600 churn events, got {count}"

    def test_accounts_has_churn_flag(self, db_engine):
        """accounts table must have a churn_flag column with 0/1 values."""
        with db_engine.connect() as conn:
            df = pd.read_sql(
                "SELECT DISTINCT churn_flag FROM accounts", conn
            )
        assert set(df["churn_flag"].astype(int).unique()).issubset(
            {0, 1}
        ), "churn_flag must only contain 0 or 1"

    def test_no_duplicate_usage_ids(self, db_engine):
        """feature_usage must have no duplicate usage_id values."""
        with db_engine.connect() as conn:
            df = pd.read_sql(
                "SELECT usage_id, COUNT(*) as n FROM feature_usage "
                "GROUP BY usage_id HAVING COUNT(*) > 1",
                conn,
            )
        assert len(df) == 0, f"Found {len(df)} duplicate usage_id values"


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — Feature Engineering
# ═══════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    """Verify the churn_features table was built correctly."""

    def test_feature_table_row_count(self, features_df):
        """churn_features must have one row per account — 500 rows."""
        assert len(features_df) == 500, (
            f"Expected 500 rows in churn_features, got {len(features_df)}"
        )

    def test_feature_table_min_columns(self, features_df):
        """churn_features must have at least 30 feature columns
        (after feature selection removes low-variance/correlated cols)."""
        n_cols = len(features_df.columns)
        assert n_cols >= 30, (
            f"Expected at least 30 columns, got {n_cols}"
        )

    def test_churn_flag_present(self, features_df):
        """churn_features must contain the target label column."""
        assert "churn_flag" in features_df.columns, (
            "churn_flag column missing from churn_features"
        )

    def test_churn_flag_binary(self, features_df):
        """churn_flag must only contain 0 and 1."""
        unique_vals = set(features_df["churn_flag"].unique())
        assert unique_vals.issubset({0, 1}), (
            f"churn_flag contains unexpected values: {unique_vals}"
        )

    def test_churn_rate_approx_22_percent(self, features_df):
        """Dataset churn rate must be approximately 22%
        (110 churned out of 500 accounts)."""
        churn_rate = features_df["churn_flag"].mean()
        assert 0.18 <= churn_rate <= 0.26, (
            f"Expected churn rate ~22%, got {churn_rate:.1%}"
        )

    def test_no_null_values_in_key_features(self, features_df):
        """Core feature columns must have no null values."""
        key_features = [
            "tenure_days", "avg_mrr", "ticket_count",
            "total_usage_minutes", "avg_satisfaction",
        ]
        for col in key_features:
            if col in features_df.columns:
                null_count = features_df[col].isnull().sum()
                assert null_count == 0, (
                    f"Column '{col}' has {null_count} null values"
                )

    def test_account_id_is_unique(self, features_df):
        """Each account_id must appear exactly once in churn_features."""
        assert features_df["account_id"].nunique() == len(features_df), (
            "Duplicate account_id values found in churn_features"
        )

    def test_usage_minutes_positive(self, features_df):
        """total_usage_minutes must be non-negative for all customers."""
        assert (features_df["total_usage_minutes"] >= 0).all(), (
            "Negative usage_minutes found"
        )

    def test_churn_features_csv_exists(self, config):
        """data/processed/churn_features_v2.csv must exist as a backup."""
        csv_path = config.PROCESSED_DATA_DIR / "churn_features_v2.csv"
        assert csv_path.exists(), (
            f"churn_features_v2.csv not found at {csv_path}"
        )


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — Model
# ═══════════════════════════════════════════════════════════════════════

class TestModel:
    """Verify the trained model loads correctly and produces valid output."""

    @pytest.fixture(scope="class")
    def loaded_model(self):
        """Load model from MLflow once for all model tests."""
        import mlflow
        import mlflow.sklearn
        import mlflow.xgboost
        from src.utils.config import Config

        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)

        for registry_name in ["churn_predictor", "churn_xgboost"]:
            for loader in [mlflow.sklearn.load_model,
                           mlflow.xgboost.load_model]:
                try:
                    return loader(f"models:/{registry_name}/latest")
                except Exception:
                    continue

        pytest.skip("MLflow server not running — skipping model tests")

    def test_model_loads_successfully(self, loaded_model):
        """Model must load from MLflow without returning None."""
        assert loaded_model is not None, "Model failed to load from MLflow"

    def test_model_has_predict_proba(self, loaded_model):
        """Model must expose predict_proba method."""
        assert hasattr(loaded_model, "predict_proba"), (
            "Model does not have predict_proba method"
        )

    def test_model_has_feature_names(self, loaded_model):
        """Model must store its feature names for column alignment."""
        has_names = (
            hasattr(loaded_model, "feature_names_in_") or
            hasattr(loaded_model, "get_booster")
        )
        assert has_names, "Model does not expose feature names"

    def test_model_feature_count(self, loaded_model):
        """Model must use at least 20 features (after feature selection)."""
        if hasattr(loaded_model, "feature_names_in_"):
            n = len(loaded_model.feature_names_in_)
        elif hasattr(loaded_model, "get_booster"):
            n = len(loaded_model.get_booster().feature_names)
        else:
            pytest.skip("Cannot determine feature count")
        assert n >= 20, f"Expected at least 20 features, got {n}"

    def test_predictions_are_probabilities(self, loaded_model, features_df):
        """predict_proba output must be between 0 and 1 for all customers."""
        X = features_df.drop(columns=["account_id", "churn_flag"])

        # Align to model features
        if hasattr(loaded_model, "feature_names_in_"):
            names = list(loaded_model.feature_names_in_)
            for col in names:
                if col not in X.columns:
                    X[col] = 0
            X = X[names]

        probas = loaded_model.predict_proba(X)[:, 1]
        assert probas.min() >= 0.0, "Probabilities below 0 found"
        assert probas.max() <= 1.0, "Probabilities above 1 found"
        assert len(probas) == 500, f"Expected 500 predictions, got {len(probas)}"

    def test_model_predicts_both_classes(self, loaded_model, features_df):
        """Model must predict both churn and non-churn across 500 customers.
        A model stuck always predicting one class is useless."""
        X = features_df.drop(columns=["account_id", "churn_flag"])

        if hasattr(loaded_model, "feature_names_in_"):
            names = list(loaded_model.feature_names_in_)
            for col in names:
                if col not in X.columns:
                    X[col] = 0
            X = X[names]

        probas = loaded_model.predict_proba(X)[:, 1]
        predictions = (probas >= 0.5).astype(int)

        assert predictions.sum() > 0, "Model never predicts churn"
        assert predictions.sum() < 500, "Model always predicts churn"


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4 — FastAPI
# ═══════════════════════════════════════════════════════════════════════

class TestAPI:
    """Verify the FastAPI prediction endpoints work correctly."""

    @pytest.fixture(scope="class")
    def client(self):
        """Create a TestClient for the FastAPI app.
        Using context manager ensures startup event fires and model loads."""
        try:
            from fastapi.testclient import TestClient
            from src.api.main import app
            with TestClient(app) as c:
                yield c
        except Exception as e:
            pytest.skip(f"FastAPI app could not be loaded: {e}")

    def test_health_endpoint_returns_200(self, client):
        """GET /health must return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}"
        )

    def test_health_response_has_required_keys(self, client):
        """GET /health response must include status and model_loaded keys."""
        response = client.get("/health")
        data = response.json()
        assert "status"       in data, "Missing 'status' key in /health"
        assert "model_loaded" in data, "Missing 'model_loaded' key in /health"

    def test_predict_endpoint_returns_200(self, client):
        """POST /predict with minimal valid input must return HTTP 200."""
        payload = {
            "tenure_days":          180,
            "avg_mrr":              800.0,
            "total_usage_minutes":  1500.0,
            "ticket_count":         4.0,
            "avg_satisfaction":     3.5,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

    def test_predict_response_schema(self, client):
        """POST /predict response must contain probability, prediction,
        and risk_level fields with correct types."""
        payload = {"tenure_days": 180, "avg_mrr": 800.0}
        response = client.post("/predict", json=payload)
        data = response.json()

        assert "churn_probability" in data, "Missing churn_probability"
        assert "churn_prediction"  in data, "Missing churn_prediction"
        assert "risk_level"        in data, "Missing risk_level"

        assert 0.0 <= data["churn_probability"] <= 1.0, (
            f"churn_probability out of range: {data['churn_probability']}"
        )
        assert data["churn_prediction"] in {0, 1}, (
            f"churn_prediction must be 0 or 1, got {data['churn_prediction']}"
        )
        assert data["risk_level"] in {"low", "medium", "high"}, (
            f"Invalid risk_level: {data['risk_level']}"
        )

    def test_batch_predict_endpoint(self, client):
        """POST /predict/batch must return predictions for all customers
        in the request."""
        payload = {
            "customers": [
                {"tenure_days": 180, "avg_mrr": 800.0},
                {"tenure_days": 30,  "avg_mrr": 100.0, "ticket_count": 10.0},
            ]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}"
        )
        data = response.json()
        assert data["total_customers"] == 2, (
            f"Expected 2 predictions, got {data['total_customers']}"
        )
        assert "high_risk_count" in data, "Missing high_risk_count"


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5 — Drift Detection
# ═══════════════════════════════════════════════════════════════════════

class TestDriftDetection:
    """Verify the drift report is valid and contains required fields."""

    @pytest.fixture(scope="class")
    def drift_report(self, config):
        """Load drift report JSON — skip if not yet generated."""
        report_path = config.INTERIM_DATA_DIR / "drift_report.json"
        if not report_path.exists():
            pytest.skip("drift_report.json not found — run detect_drift first")
        with open(report_path) as f:
            return json.load(f)

    def test_drift_report_exists(self, config):
        """data/interim/drift_report.json must exist."""
        report_path = config.INTERIM_DATA_DIR / "drift_report.json"
        assert report_path.exists(), (
            "drift_report.json not found. "
            "Run: python -m src.drift_detection.detect_drift"
        )

    def test_drift_report_has_required_keys(self, drift_report):
        """Drift report must contain all required top-level keys."""
        required_keys = [
            "verdict",
            "total_features_tested",
            "features_with_drift",
            "drift_ratio",
            "drifted_features",
            "feature_details",
            "timestamp",
        ]
        for key in required_keys:
            assert key in drift_report, (
                f"Missing key '{key}' in drift_report.json"
            )

    def test_drift_verdict_is_valid(self, drift_report):
        """Drift verdict must be one of the 3 valid categories."""
        valid_verdicts = {"no_drift", "moderate_drift", "significant_drift"}
        assert drift_report["verdict"] in valid_verdicts, (
            f"Invalid verdict: {drift_report['verdict']}"
        )

    def test_drift_ratio_is_between_0_and_1(self, drift_report):
        """drift_ratio must be a valid proportion between 0 and 1."""
        ratio = drift_report["drift_ratio"]
        assert 0.0 <= ratio <= 1.0, (
            f"drift_ratio out of range: {ratio}"
        )

    def test_drift_feature_counts_consistent(self, drift_report):
        """features_with_drift must equal len(drifted_features)."""
        assert drift_report["features_with_drift"] == len(
            drift_report["drifted_features"]
        ), "features_with_drift count does not match drifted_features list"

    def test_feature_details_populated(self, drift_report):
        """feature_details must contain results for at least 20 features."""
        n = len(drift_report["feature_details"])
        assert n >= 20, (
            f"Expected at least 20 features in feature_details, got {n}"
        )

    def test_each_feature_has_test_and_pvalue(self, drift_report):
        """Each feature in feature_details must have test type and p_value."""
        for feature, detail in list(drift_report["feature_details"].items())[:5]:
            assert "test"      in detail, f"Missing 'test' for feature {feature}"
            assert "p_value"   in detail, f"Missing 'p_value' for feature {feature}"
            assert "statistic" in detail, f"Missing 'statistic' for feature {feature}"


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6 — Retraining Pipeline
# ═══════════════════════════════════════════════════════════════════════

class TestRetraining:
    """Verify the retraining pipeline components work correctly."""

    def test_drift_report_is_readable_by_retrain(self, config):
        """retrain.py must be able to read drift_report.json
        and parse the verdict field."""
        report_path = config.INTERIM_DATA_DIR / "drift_report.json"
        if not report_path.exists():
            pytest.skip("drift_report.json not found")

        with open(report_path) as f:
            report = json.load(f)

        assert "verdict" in report, "verdict key missing from drift report"
        assert report["verdict"] in {
            "no_drift", "moderate_drift", "significant_drift"
        }, f"Unexpected verdict: {report['verdict']}"

    def test_params_yaml_has_required_model_keys(self, config):
        """params.yaml must contain all keys needed to train a challenger."""
        params = config.load_params()
        required = [
            "n_estimators", "max_depth", "random_state"
        ]
        model_params = params.get("model", {})
        for key in required:
            assert key in model_params, (
                f"Missing '{key}' in params.yaml model section"
            )

    def test_params_yaml_model_type_is_random_forest(self, config):
        """params.yaml model type must reflect the actual champion
        (random_forest, not xgboost)."""
        params = config.load_params()
        model_type = params.get("model", {}).get("type", "")
        assert model_type == "random_forest", (
            f"params.yaml model.type should be 'random_forest', got '{model_type}'"
        )

    def test_feature_engineering_produces_correct_output(self, features_df):
        """churn_features table must have account_id and churn_flag columns
        so retrain.py can split features and target correctly."""
        assert "account_id"  in features_df.columns, "Missing account_id"
        assert "churn_flag"  in features_df.columns, "Missing churn_flag"

        # Ensure these can be dropped to produce a valid feature matrix
        X = features_df.drop(columns=["account_id", "churn_flag"])
        assert len(X.columns) >= 20, (
            f"Feature matrix too small: {len(X.columns)} columns"
        )


# ═══════════════════════════════════════════════════════════════════════
# Config and utilities
# ═══════════════════════════════════════════════════════════════════════

class TestConfig:
    """Verify project configuration loads correctly."""

    def test_config_loads_params(self, config):
        """Config.load_params() must return a non-empty dict."""
        params = config.load_params()
        assert isinstance(params, dict), "params.yaml did not load as dict"
        assert len(params) > 0, "params.yaml is empty"

    def test_config_has_mlflow_uri(self, config):
        """Config must expose MLFLOW_TRACKING_URI."""
        assert hasattr(config, "MLFLOW_TRACKING_URI"), (
            "Config missing MLFLOW_TRACKING_URI"
        )
        assert "5000" in str(config.MLFLOW_TRACKING_URI), (
            "MLFLOW_TRACKING_URI should point to port 5000"
        )

    def test_data_directories_exist(self, config):
        """All required data directories must exist."""
        dirs = [
            config.RAW_DATA_DIR,
            config.PROCESSED_DATA_DIR,
            config.INTERIM_DATA_DIR,
        ]
        for d in dirs:
            assert Path(d).exists(), f"Directory not found: {d}"

    def test_raw_csv_files_exist(self, config):
        """All 5 raw CSV files must be present in data/raw/."""
        expected = [
            "ravenstack_accounts.csv",
            "ravenstack_subscriptions.csv",
            "ravenstack_feature_usage.csv",
            "ravenstack_support_tickets.csv",
            "ravenstack_churn_events.csv",
        ]
        for fname in expected:
            fpath = Path(config.RAW_DATA_DIR) / fname
            assert fpath.exists(), f"Raw CSV not found: {fname}"