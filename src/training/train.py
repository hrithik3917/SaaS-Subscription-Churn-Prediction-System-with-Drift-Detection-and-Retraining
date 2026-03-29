"""
train.py - Phase 3: Model Training

Trains an XGBoost classifier on the churn_features table, evaluates
performance on a held-out test set, and logs everything to MLflow
(parameters, metrics, model artifact).

Pipeline steps:
    1. Load feature table from PostgreSQL
    2. Split into train/test sets
    3. Train XGBoost with params from params.yaml
    4. Evaluate (AUC-ROC, precision, recall, F1, accuracy)
    5. Log experiment to MLflow
    6. Register model in MLflow model registry

Usage:
    python -m src.training.train
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, classification_report
)
from xgboost import XGBClassifier
from src.utils.config import Config
from src.utils.db import engine


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_feature_table():
    """Load the churn_features table built in Phase 2."""
    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM churn_features", conn)
    return df


def prepare_data(df, test_size, random_state):
    """
    Split features and target, then create train/test sets.

    - X: all columns except account_id (identifier) and churn_flag (target)
    - y: churn_flag (what we're predicting — 0 = stayed, 1 = churned)
    - Stratified split ensures train and test have the same churn ratio
    """
    X = df.drop(columns=["account_id", "churn_flag"])
    y = df["churn_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # Preserve churn ratio in both sets
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, model_params):
    """
    Train an XGBoost classifier.

    XGBoost is a gradient-boosted tree algorithm — it builds many small
    decision trees sequentially, where each tree corrects the mistakes
    of the previous ones. It's the go-to algorithm for tabular data
    because it handles mixed feature types, missing values, and
    imbalanced classes well.

    scale_pos_weight handles class imbalance: our dataset has ~390 non-churned
    vs ~110 churned accounts. Without balancing, the model would just predict
    "not churned" for everyone and still get 78% accuracy. scale_pos_weight
    tells XGBoost to pay more attention to the minority (churned) class.
    """
    # Calculate class imbalance ratio for weighting
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    model = XGBClassifier(
        max_depth=model_params["max_depth"],
        learning_rate=model_params["learning_rate"],
        n_estimators=model_params["n_estimators"],
        random_state=model_params["random_state"],
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    Metrics:
        - AUC-ROC: how well the model ranks churners above non-churners (0.5 = random, 1.0 = perfect)
        - Precision: of all predicted churners, how many actually churned
        - Recall: of all actual churners, how many did we catch
        - F1: harmonic mean of precision and recall (balances both)
        - Accuracy: overall correct predictions
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "auc_roc": round(roc_auc_score(y_test, y_pred_proba), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
    }

    return metrics, y_pred


# ---------------------------------------------------------------------------
# MLflow experiment logging
# ---------------------------------------------------------------------------

def log_experiment(model, model_params, training_params, metrics, feature_columns):
    """
    Log the full experiment to MLflow: parameters, metrics, and model artifact.

    MLflow stores everything so you can compare experiments later:
    - Which hyperparameters produced the best AUC?
    - Did adding new features improve recall?
    - Which model version should go to production?
    """
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("churn_prediction")

    with mlflow.start_run(run_name="xgboost_baseline"):
        # Log model hyperparameters
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("max_depth", model_params["max_depth"])
        mlflow.log_param("learning_rate", model_params["learning_rate"])
        mlflow.log_param("n_estimators", model_params["n_estimators"])
        mlflow.log_param("test_size", training_params["test_size"])
        mlflow.log_param("num_features", len(feature_columns))

        # Log evaluation metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log the trained model artifact
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="churn_xgboost",
        )

        run_id = mlflow.active_run().info.run_id

    return run_id


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_training():
    """
    Execute the full training pipeline:
      1. Load features from PostgreSQL
      2. Split into train/test
      3. Train XGBoost model
      4. Evaluate on test set
      5. Log everything to MLflow
    """
    params = Config.load_params()
    model_params = params["model"]
    training_params = params["training"]

    # Load and split data
    df = load_feature_table()
    X_train, X_test, y_train, y_test = prepare_data(
        df,
        test_size=training_params["test_size"],
        random_state=model_params["random_state"],
    )

    # Train
    model = train_model(X_train, y_train, model_params)

    # Evaluate
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    # Log to MLflow
    run_id = log_experiment(
        model, model_params, training_params, metrics,
        feature_columns=X_train.columns.tolist(),
    )

    return model, metrics, run_id


if __name__ == "__main__":
    run_training()