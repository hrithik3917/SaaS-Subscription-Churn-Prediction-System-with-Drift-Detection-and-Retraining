"""
main.py - Phase 4: Model Deployment (FastAPI)

REST API that loads the trained model from MLflow and serves
churn predictions. Tries sklearn first (Random Forest, Logistic Regression),
then falls back to xgboost for backwards compatibility.

Endpoints:
    GET  /health          — API and model health check
    POST /predict         — Single customer churn prediction
    POST /predict/batch   — Batch predictions for multiple customers

Usage:
    1. Start MLflow server:  mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
    2. Start API server:     python -m uvicorn src.api.main:app --reload --port 8000
    3. Open docs:            http://127.0.0.1:8000/docs
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.utils.config import Config
from contextlib import asynccontextmanager




# Loaded once at startup, reused across all requests
model = None
model_feature_names = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """
    Load the latest registered model from MLflow.

    Tries churn_predictor (current registry name) first.
    Falls back to churn_xgboost for backward compatibility during
    any transition period. Within each registry name, tries sklearn
    first (handles Random Forest and Logistic Regression), then xgboost.
    """
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)

    registry_names = ["churn_predictor", "churn_xgboost"]
    loaders = [mlflow.sklearn.load_model, mlflow.xgboost.load_model]

    native_model = None
    for registry_name in registry_names:
        for loader in loaders:
            try:
                native_model = loader(f"models:/{registry_name}/latest")
                break
            except Exception:
                continue
        if native_model is not None:
            break

    if native_model is None:
        raise RuntimeError(
            "Could not load model from MLflow. "
            "Ensure the MLflow server is running and a model is registered "
            "under 'churn_predictor'."
        )

    # Extract the ordered list of feature names the model was trained on.
    # This is used at inference time to reorder incoming request columns.
    feature_names = None
    if hasattr(native_model, "feature_names_in_"):
        feature_names = list(native_model.feature_names_in_)
    elif hasattr(native_model, "get_booster"):
        feature_names = native_model.get_booster().feature_names

    return native_model, feature_names


# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model into memory on startup, clean up on shutdown."""
    global model, model_feature_names
    model, model_feature_names = load_model()
    yield 

    
app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn probability using the production ML model",
    version="3.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    """Input schema: all possible customer features. All fields have safe
    defaults so callers can send only the features they have."""

    # Account features
    seats: int = Field(default=0)
    is_trial: int = Field(default=0)
    plan_tier_encoded: int = Field(default=0)
    tenure_days: int = Field(default=0)
    is_new_customer: int = Field(default=0)
    is_established: int = Field(default=0)

    # Subscription features
    total_subscriptions: float = Field(default=0)
    active_subscriptions: float = Field(default=0)
    avg_mrr: float = Field(default=0)
    max_mrr: float = Field(default=0)
    total_arr: float = Field(default=0)
    avg_seats_per_sub: float = Field(default=0)
    has_upgrade: int = Field(default=0)
    has_downgrade: int = Field(default=0)
    trial_ratio: float = Field(default=0)
    auto_renew_ratio: float = Field(default=0)
    days_since_last_sub: float = Field(default=0)
    latest_plan_tier: float = Field(default=0)
    is_monthly_billing: int = Field(default=0)
    mrr_change_ratio: float = Field(default=1.0)
    sub_velocity: float = Field(default=0)
    recent_has_upgrade: int = Field(default=0)
    recent_has_downgrade: int = Field(default=0)

    # Usage features
    total_usage_events: float = Field(default=0)
    avg_usage_count: float = Field(default=0)
    unique_features_used: float = Field(default=0)
    total_errors: float = Field(default=0)
    beta_feature_ratio: float = Field(default=0)
    total_usage_minutes: float = Field(default=0)
    error_rate: float = Field(default=0)
    avg_daily_usage_mins: float = Field(default=0)
    recent_usage_events: float = Field(default=0)
    recent_usage_minutes: float = Field(default=0)
    recent_error_rate: float = Field(default=0)
    recent_avg_daily_mins: float = Field(default=0)
    recent_features_used: float = Field(default=0)
    days_since_last_usage: float = Field(default=0)
    usage_trend_ratio: float = Field(default=1.0)
    feature_diversity_trend: float = Field(default=1.0)

    # Support features
    ticket_count: float = Field(default=0)
    avg_resolution_hours: float = Field(default=0)
    avg_first_response_mins: float = Field(default=0)
    avg_satisfaction: float = Field(default=0)
    escalation_count: float = Field(default=0)
    escalation_rate: float = Field(default=0)
    high_priority_ratio: float = Field(default=0)
    unresolved_tickets: float = Field(default=0)
    unresolved_ratio: float = Field(default=0)
    recent_ticket_count: float = Field(default=0)
    recent_avg_resolution_hours: float = Field(default=0)
    recent_avg_satisfaction: float = Field(default=0)
    recent_escalation_rate: float = Field(default=0)
    days_since_last_ticket: float = Field(default=0)
    ticket_trend_ratio: float = Field(default=1.0)

    # One-hot: industry
    industry_Cybersecurity: int = Field(default=0)
    industry_DevTools: int = Field(default=0)
    industry_EdTech: int = Field(default=0)
    industry_FinTech: int = Field(default=0)
    industry_HealthTech: int = Field(default=0)

    # One-hot: referral source
    referral_ads: int = Field(default=0)
    referral_event: int = Field(default=0)
    referral_organic: int = Field(default=0)
    referral_other: int = Field(default=0)
    referral_partner: int = Field(default=0)

    # Interaction / derived features
    frustration_score: float = Field(default=0)
    revenue_risk_score: float = Field(default=0)
    tickets_per_tenure_day: float = Field(default=0)
    usage_per_seat: float = Field(default=0)

    model_config = {"json_schema_extra": {
        "examples": [{
            "seats": 12,
            "is_trial": 0,
            "plan_tier_encoded": 1,
            "tenure_days": 300,
            "avg_mrr": 850.0,
            "total_usage_minutes": 1500.0,
            "ticket_count": 4,
            "avg_satisfaction": 3.2,
            "industry_FinTech": 1,
            "referral_organic": 1,
        }]
    }}


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_customers: int
    high_risk_count: int


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def classify_risk(probability: float) -> str:
    """Map churn probability to a human-readable risk level."""
    if probability >= 0.7:
        return "high"
    elif probability >= 0.4:
        return "medium"
    return "low"


def predict_single(features: CustomerFeatures) -> PredictionResponse:
    """
    Run prediction for a single customer.

    Converts the Pydantic model to a DataFrame, aligns columns to
    match the exact feature order the model was trained on (critical
    for tree-based models), then calls predict_proba.
    """
    df = pd.DataFrame([features.model_dump()])

    # Align columns to the order the model expects
    if model_feature_names is not None:
        for col in model_feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[model_feature_names]

    probability = float(model.predict_proba(df)[0][1])

    return PredictionResponse(
        churn_probability=round(probability, 4),
        churn_prediction=int(probability >= 0.5),
        risk_level=classify_risk(probability),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Verify the API is running and the model is loaded."""
    return {
        "status":         "healthy",
        "model_loaded":   model is not None,
        "model_features": len(model_feature_names) if model_feature_names else 0,
        "registry_name":  "churn_predictor",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    """Predict churn probability for a single customer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predict_single(features)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    """Predict churn for multiple customers in one request."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions = [predict_single(c) for c in request.customers]
    high_risk = sum(1 for p in predictions if p.risk_level == "high")

    return BatchResponse(
        predictions=predictions,
        total_customers=len(predictions),
        high_risk_count=high_risk,
    )