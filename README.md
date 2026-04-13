# SaaS Subscription Churn Prediction System with Drift Detection and Automated Retraining

[![CI Pipeline](https://github.com/hrithik3917/SaaS-Subscription-Churn-Prediction-System-with-Drift-Detection-and-Retraining/actions/workflows/ci.yml/badge.svg)](https://github.com/hrithik3917/SaaS-Subscription-Churn-Prediction-System-with-Drift-Detection-and-Retraining/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.10-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)
![Tests](https://img.shields.io/badge/Tests-42%20passing-brightgreen)

A production-grade, end-to-end MLOps platform that predicts which SaaS customers are likely to cancel their subscription. The system spans six phases — from raw data ingestion to a live interactive dashboard — and includes automated drift detection and champion-challenger model retraining.

> **Master's Capstone Project** — MS Data Science (Software & Systems Specialization)  
> University of Kentucky · May 2026  
> **Author:** Hritik Dalvi · **Advisor:** Dr. Brent Harrison

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Pipeline](#running-the-pipeline)
- [Running the Dashboard](#running-the-dashboard)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Dataset](#dataset)
- [Acknowledgements](#acknowledgements)

---

## Overview

Customer churn is one of the most costly challenges for SaaS businesses. This project builds a complete MLOps infrastructure — not just a model, but the full production pipeline:

- **Data ingestion** from a five-table relational dataset into PostgreSQL
- **Time-windowed feature engineering** producing 68 behavioral features
- **Multi-model training** with MLflow experiment tracking (AUC improved from 0.53 → 0.64)
- **REST API** via FastAPI for real-time single and batch predictions
- **Automated drift detection** using KS and chi-squared statistical tests
- **Champion-challenger retraining loop** that promotes new models only when performance improves
- **Five-page Streamlit dashboard** for live predictions, customer risk monitoring, and drift reporting

---

## System Architecture

![Pipeline Architecture](Project_architecture_drawio.png)

The system follows a six-phase pipeline with a feedback loop for automated retraining:

```
Raw CSVs → Phase 1: Ingestion → PostgreSQL
                                    ↓
                         Phase 2: Feature Engineering (68 features)
                                    ↓
                         Phase 3: Model Training (MLflow)
                                    ↓
                         Phase 4: FastAPI Prediction Endpoint
                                    ↓
                         Phase 5: Drift Detection (KS + Chi-squared)
                                    ↓
                         Phase 6: Champion-Challenger Retraining
                                    ↑ (rebuild features if drift detected)
```

---

## Model Performance

Three training iterations documented with full experiment tracking in MLflow:

| Version | Algorithm | AUC-ROC | Precision | Recall | F1 Score | Key Change |
|---------|-----------|---------|-----------|--------|----------|------------|
| v1 Baseline | XGBoost (default) | 0.5303 | 0.2000 | 0.1364 | 0.1622 | All-time features, default params |
| v2 Time-windowed | XGBoost (regularized) | 0.5385 | 0.2727 | 0.1364 | 0.1818 | 90-day observation windows, trend features |
| v3 Optimized | **Random Forest** | **0.6352** | **0.5000** | **0.4091** | **0.4500** | Feature selection, multi-model CV, threshold tuning |

**v3 champion:** Random Forest · `n_estimators=300` · `max_depth=5` · `class_weight=balanced` · `threshold=0.4758`

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.12 |
| Database | PostgreSQL 15 |
| ORM / DB | SQLAlchemy |
| ML Framework | scikit-learn, XGBoost |
| Experiment Tracking | MLflow 3.10 |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Drift Detection | scipy (KS test, chi-squared) |
| Data Processing | pandas, NumPy |
| Testing | pytest (42 tests) |
| CI/CD | GitHub Actions |
| Config | python-dotenv, PyYAML |

---

## Project Structure

```
Capstone_project/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI pipeline
├── configs/
│   └── params.yaml                 # Model hyperparameters and window config
├── data/
│   ├── raw/                        # 5 RavenStack CSV source files
│   ├── processed/                  # churn_features_v2.csv (engineered features)
│   └── interim/                    # drift_report.json
├── src/
│   ├── api/
│   │   └── main.py                 # Phase 4: FastAPI prediction endpoints
│   ├── dashboard/
│   │   └── app.py                  # 5-page Streamlit dashboard
│   ├── data_ingestion/
│   │   └── load_data.py            # Phase 1: CSV → PostgreSQL
│   ├── drift_detection/
│   │   └── detect_drift.py         # Phase 5: KS + chi-squared drift tests
│   ├── feature_engineering/
│   │   └── build_features.py       # Phase 2: 68 time-windowed features
│   ├── retraining/
│   │   └── retrain.py              # Phase 6: Champion-challenger loop
│   ├── training/
│   │   └── train.py                # Phase 3: Multi-model training + MLflow
│   └── utils/
│       ├── config.py               # Central config and params loader
│       └── db.py                   # PostgreSQL connection via SQLAlchemy
├── tests/
│   └── test_pipeline.py            # 42 unit tests across all 6 phases
├── .env                            # DB credentials + MLflow URI (gitignored)
├── .gitignore
├── requirements.txt
├── requirements-ci.txt             # CI-safe requirements (no Windows packages)
├── run_dashboard.bat               # One-click dashboard launcher (Windows)
└── README.md
```

---

## Setup and Installation

### Prerequisites

- Python 3.12
- PostgreSQL 15 running locally
- Git

### 1. Clone the repository

```bash
git clone https://github.com/hrithik3917/SaaS-Subscription-Churn-Prediction-System-with-Drift-Detection-and-Retraining.git
cd SaaS-Subscription-Churn-Prediction-System-with-Drift-Detection-and-Retraining
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=saas_churn
DB_USER=postgres
DB_PASSWORD=your_password
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

### 5. Create the PostgreSQL database

```sql
CREATE DATABASE saas_churn;
```

---

## Running the Pipeline

Run these commands in order. Each phase depends on the previous one.

**Terminal 1 — Start MLflow tracking server:**
```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

**Terminal 2 — Run the pipeline phases:**

```bash
# Set Python path (Windows)
$env:PYTHONPATH = "C:\path\to\project"

# Phase 1: Load raw CSV files into PostgreSQL
python -m src.data_ingestion.load_data

# Phase 2: Build 68 time-windowed features
python -m src.feature_engineering.build_features

# Phase 3: Train and compare models, register champion in MLflow
python -m src.training.train

# Phase 4: Start the prediction API (keep running)
python -m uvicorn src.api.main:app --reload --port 8000

# Phase 5: Run drift detection
python -m src.drift_detection.detect_drift

# Phase 6: Run automated retraining pipeline
python -m src.retraining.retrain
```

---

## Running the Dashboard

Make sure the MLflow server and FastAPI are both running, then:

```bash
# Windows
$env:PYTHONPATH = "C:\path\to\project"
streamlit run src/dashboard/app.py
```

Or double-click `run_dashboard.bat` on Windows.

Open `http://localhost:8501` in your browser.

### Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | KPI metrics, risk distribution charts, industry breakdown |
| **Customers** | Filterable table of all 500 customers scored by churn risk |
| **Predict** | Select a customer by name OR explore risk scenarios by profile |
| **Model** | AUC improvement journey, feature importances, current metrics |
| **Drift** | Drift verdict, drifted features table, run drift detection |

---

## API Reference

Base URL: `http://127.0.0.1:8000`  
Interactive docs: `http://127.0.0.1:8000/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Returns model loaded status and feature count |
| `POST` | `/predict` | Single customer churn prediction |
| `POST` | `/predict/batch` | Batch predictions for multiple customers |

**Example — single prediction:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_days": 90,
    "avg_mrr": 200.0,
    "total_usage_minutes": 150.0,
    "ticket_count": 12.0,
    "avg_satisfaction": 1.5,
    "escalation_rate": 0.8
  }'
```

**Response:**
```json
{
  "churn_probability": 0.847,
  "churn_prediction": 1,
  "risk_level": "high"
}
```

---

## Running Tests

Make sure the MLflow server is running, then:

```bash
$env:PYTHONPATH = "C:\path\to\project"
pytest tests/ -v
```

**42 tests across 7 test classes:**

| Class | Tests | What it covers |
|-------|-------|----------------|
| `TestDataIngestion` | 7 | Row counts, FK integrity, deduplication |
| `TestFeatureEngineering` | 9 | Column count, nulls, label distribution |
| `TestModel` | 6 | Model loads, valid probabilities, both classes predicted |
| `TestAPI` | 5 | All 3 endpoints return correct schema |
| `TestDriftDetection` | 7 | Report structure, verdict validity, counts |
| `TestRetraining` | 4 | params.yaml correct, pipeline readable |
| `TestConfig` | 4 | Directories exist, CSVs present, MLflow URI set |

---

## Dataset

**RavenStack** — Synthetic multi-table SaaS dataset by River @ Rivalytics

| Table | Rows | Description |
|-------|------|-------------|
| accounts | 500 | Customer profiles, plan tier, churn label |
| subscriptions | 5,000 | Billing records, MRR, upgrades/downgrades |
| feature_usage | 24,979 | Daily product usage events |
| support_tickets | 2,000 | Support interactions, resolution, satisfaction |
| churn_events | 600 | Post-churn data (excluded from features — data leakage prevention) |

**Class distribution:** 78% retained · 22% churned (110 of 500 accounts)

---

## Acknowledgements

This project was completed as part of the Master of Science in Data Science (Software & Systems Specialization) program at the **University of Kentucky**.

Special thanks to **Dr. Brent Harrison** for guidance and support throughout the capstone project.

Dataset: [RavenStack by River @ Rivalytics](https://rivalytics.com)