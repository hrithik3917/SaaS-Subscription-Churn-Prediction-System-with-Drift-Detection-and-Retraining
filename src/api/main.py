"""
main.py - Phase 4: Model Deployment (FastAPI)

REST API that loads the trained XGBoost model from MLflow and serves
churn predictions via HTTP endpoints.

Endpoints:
    GET  /health          — API and model health check
    POST /predict         — Single customer churn prediction
    POST /predict/batch   — Batch predictions for multiple customers

Usage:
    1. Start MLflow server:  mlflow server --host 127.0.0.1 --port 5000
    2. Start API server:     python -m uvicorn src.api.main:app --reload --port 8000
    3. Open docs:            http://127.0.0.1:8000/docs
"""

import mlflow
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.utils.config import Config


# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn probability using XGBoost model",
    version="1.0.0",
)

# Loaded once at startup
model = None
model_feature_names = None  # Column names the model was trained on, in exact order


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """
    Load the latest registered model from MLflow.
    Also extract the feature names the model expects, so the API
    can reorder incoming data to match the training column order.
    """
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    loaded_model = mlflow.xgboost.load_model("models:/churn_xgboost/latest")

    # XGBoost stores the feature names it was trained with
    feature_names = loaded_model.get_booster().feature_names
    return loaded_model, feature_names


@app.on_event("startup")
def startup_event():
    """Load model into memory when the API server starts."""
    global model, model_feature_names
    model, model_feature_names = load_model()


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    """
    Input schema: customer features for prediction.

    All fields have defaults (0 or 0.0) so callers only need to send
    the fields they have data for. The API fills missing fields with
    zeros, which is safe since 0 means "no activity" for most features.
    """
    seats: int = Field(default=0, description="Licensed seat count")
    is_trial: int = Field(default=0, description="1 if trial, 0 if not")
    plan_tier_encoded: int = Field(default=0, description="0=Basic, 1=Pro, 2=Enterprise")
    tenure_days: int = Field(default=0, description="Days since signup")

    # Subscription features
    total_subscriptions: float = Field(default=0)
    active_subscriptions: float = Field(default=0)
    avg_mrr: float = Field(default=0, description="Average monthly recurring revenue")
    total_arr: float = Field(default=0, description="Total annual recurring revenue")
    avg_seats_per_sub: float = Field(default=0)
    has_upgrade: int = Field(default=0)
    has_downgrade: int = Field(default=0)
    trial_ratio: float = Field(default=0)
    auto_renew_ratio: float = Field(default=0)
    monthly_billing_ratio: float = Field(default=0)

    # Usage features
    total_usage_events: float = Field(default=0)
    avg_usage_count: float = Field(default=0)
    unique_features_used: float = Field(default=0)
    total_errors: float = Field(default=0)
    beta_feature_ratio: float = Field(default=0)
    total_usage_minutes: float = Field(default=0)
    error_rate: float = Field(default=0)
    avg_daily_usage_mins: float = Field(default=0)

    # Support features
    ticket_count: float = Field(default=0)
    avg_resolution_hours: float = Field(default=0)
    avg_first_response_mins: float = Field(default=0)
    avg_satisfaction: float = Field(default=0)
    escalation_count: float = Field(default=0)
    escalation_rate: float = Field(default=0)
    high_priority_ratio: float = Field(default=0)

    # One-hot: industry (set the one that applies to 1)
    industry_Cybersecurity: int = Field(default=0)
    industry_DevTools: int = Field(default=0)
    industry_EdTech: int = Field(default=0)
    industry_FinTech: int = Field(default=0)
    industry_HealthTech: int = Field(default=0)

    # One-hot: referral source (set the one that applies to 1)
    referral_ads: int = Field(default=0)
    referral_event: int = Field(default=0)
    referral_organic: int = Field(default=0)
    referral_other: int = Field(default=0)
    referral_partner: int = Field(default=0)

    model_config = {"json_schema_extra": {
        "examples": [{
            "seats": 12,
            "is_trial": 0,
            "plan_tier_encoded": 1,
            "tenure_days": 300,
            "total_subscriptions": 3,
            "active_subscriptions": 1,
            "avg_mrr": 850.0,
            "total_arr": 10200.0,
            "ticket_count": 4,
            "avg_resolution_hours": 28.5,
            "avg_satisfaction": 3.2,
            "total_usage_minutes": 1500.0,
            "unique_features_used": 12,
            "error_rate": 0.03,
            "industry_FinTech": 1,
            "referral_organic": 1,
        }]
    }}


class PredictionResponse(BaseModel):
    """Output schema: prediction result for one customer."""
    churn_probability: float = Field(..., description="Probability of churn (0.0 to 1.0)")
    churn_prediction: int = Field(..., description="1 = will churn, 0 = will stay")
    risk_level: str = Field(..., description="low / medium / high")


class BatchRequest(BaseModel):
    """Input schema: multiple customers for batch prediction."""
    customers: list[CustomerFeatures]


class BatchResponse(BaseModel):
    """Output schema: predictions for multiple customers."""
    predictions: list[PredictionResponse]
    total_customers: int
    high_risk_count: int


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def classify_risk(probability):
    """Map churn probability to a human-readable risk level."""
    if probability >= 0.7:
        return "high"
    elif probability >= 0.4:
        return "medium"
    return "low"


def predict_single(features: CustomerFeatures) -> PredictionResponse:
    """Run prediction for a single customer."""
    feature_dict = features.model_dump()
    df = pd.DataFrame([feature_dict])

    # Ensure columns match the exact order the model was trained with.
    # Add any missing columns as 0, drop any extra columns, reorder.
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
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Verify API is running and model is loaded."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_features": len(model_feature_names) if model_feature_names else 0,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    """
    Predict churn probability for a single customer.

    Send customer features as JSON, receive back:
    - churn_probability (0.0 to 1.0)
    - churn_prediction (0 or 1)
    - risk_level (low / medium / high)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predict_single(features)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    """
    Predict churn for multiple customers in one request.

    Returns individual predictions plus summary stats.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions = [predict_single(customer) for customer in request.customers]
    high_risk = sum(1 for p in predictions if p.risk_level == "high")

    return BatchResponse(
        predictions=predictions,
        total_customers=len(predictions),
        high_risk_count=high_risk,
    )