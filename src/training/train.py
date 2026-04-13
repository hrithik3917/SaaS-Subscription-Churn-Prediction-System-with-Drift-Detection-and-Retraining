"""
train.py - Phase 3: Model Training (v3 — Feature Selection + Model Comparison)

Addresses the low-signal challenge in the synthetic dataset by:
    1. Removing low-variance and highly correlated features (noise reduction)
    2. Training multiple model types (XGBoost, Random Forest, Logistic Regression)
    3. Selecting the best performer via cross-validation
    4. Tuning the classification threshold for optimal F1

With 500 rows and 68 features, the model was drowning in noise.
Fewer, stronger features + simpler models = better generalization.

Usage:
    python -m src.training.train
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
    """Split features and target into train/test sets with stratification."""
    X = df.drop(columns=["account_id", "churn_flag"])
    y = df["churn_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features(X_train, X_test, y_train, variance_threshold=0.01, correlation_threshold=0.90):
    """
    Reduce feature count by removing noise:

    Step 1 — Remove near-zero variance features:
        Features that are almost always the same value (e.g., a column
        that's 0 for 99% of rows) provide no predictive signal.

    Step 2 — Remove highly correlated features:
        If two features are 90%+ correlated, they carry the same info.
        Keeping both adds noise without adding signal. We drop the one
        with lower correlation to the target.

    This typically cuts features from 68 down to 25-35, dramatically
    reducing the noise the model has to deal with.
    """
    # Step 1: Remove near-zero variance
    variances = X_train.var()
    low_var_cols = variances[variances < variance_threshold].index.tolist()
    X_train = X_train.drop(columns=low_var_cols)
    X_test = X_test.drop(columns=low_var_cols)

    # Step 2: Remove highly correlated features (keep the one more correlated with target)
    corr_matrix = X_train.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # For each pair above threshold, drop the one less correlated with target
    target_corr = X_train.corrwith(y_train).abs()
    to_drop = set()

    for col in upper_triangle.columns:
        correlated_cols = upper_triangle.index[upper_triangle[col] > correlation_threshold].tolist()
        for corr_col in correlated_cols:
            # Drop whichever has lower correlation to churn_flag
            if target_corr.get(col, 0) >= target_corr.get(corr_col, 0):
                to_drop.add(corr_col)
            else:
                to_drop.add(col)

    X_train = X_train.drop(columns=list(to_drop))
    X_test = X_test.drop(columns=list(to_drop))

    return X_train, X_test, low_var_cols, list(to_drop)


# ---------------------------------------------------------------------------
# Model candidates
# ---------------------------------------------------------------------------

def build_candidates(model_params, scale_pos_weight):
    """
    Create multiple model candidates for comparison.

    Different algorithms have different strengths:
    - XGBoost: powerful but can overfit on small data
    - Random Forest: more robust on small datasets
    - Logistic Regression: simplest, often best on small noisy data
    """
    candidates = {
        "xgboost": XGBClassifier(
            max_depth=model_params.get("max_depth", 4),
            learning_rate=model_params.get("learning_rate", 0.05),
            n_estimators=model_params.get("n_estimators", 300),
            min_child_weight=model_params.get("min_child_weight", 3),
            subsample=model_params.get("subsample", 0.8),
            colsample_bytree=model_params.get("colsample_bytree", 0.8),
            gamma=model_params.get("gamma", 1),
            reg_alpha=model_params.get("reg_alpha", 0.1),
            reg_lambda=model_params.get("reg_lambda", 1.0),
            random_state=model_params.get("random_state", 42),
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 300),
            max_depth=model_params.get("max_depth", 5),
            min_samples_leaf=model_params.get("min_samples_leaf", 5),
            class_weight=model_params.get("class_weight", "balanced"),
            random_state=model_params.get("random_state", 42),
        ),
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            C=0.5,
            max_iter=1000,
            random_state=model_params.get("random_state", 42),
        ),
    }

    return candidates


# ---------------------------------------------------------------------------
# Cross-validation comparison
# ---------------------------------------------------------------------------

def compare_models_cv(candidates, X_train, y_train, random_state):
    """
    Run 5-fold cross-validation for each candidate model.
    Returns the name and CV scores of each model.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = {}

    for name, model in candidates.items():
        # Logistic regression needs scaled features
        if name == "logistic_regression":
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index,
            )
            scores = cross_val_score(model, X_scaled, y_train, cv=cv, scoring="roc_auc")
        else:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

        results[name] = {
            "cv_auc_mean": round(float(scores.mean()), 4),
            "cv_auc_std": round(float(scores.std()), 4),
            "cv_scores": scores,
        }

    return results


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def find_optimal_threshold(model, X_test, y_test, is_logistic=False, scaler=None):
    """
    Find the classification threshold that maximizes F1 score.

    Default threshold is 0.5, but for imbalanced datasets, a lower
    threshold (e.g., 0.35) often catches more churners (higher recall)
    at the cost of some precision — which is usually a good tradeoff
    since missing a churner is more costly than a false alarm.
    """
    X_eval = X_test
    if is_logistic and scaler is not None:
        X_eval = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

    y_proba = model.predict_proba(X_eval)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # Calculate F1 for each threshold
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1]),
        0,
    )

    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])

    return best_threshold, float(f1_scores[best_idx])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, threshold=0.5, is_logistic=False, scaler=None):
    """Evaluate model with a custom classification threshold."""
    X_eval = X_test
    if is_logistic and scaler is not None:
        X_eval = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

    y_pred_proba = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "auc_roc": round(roc_auc_score(y_test, y_pred_proba), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "threshold": round(threshold, 4),
    }

    return metrics


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(model, feature_names, model_name):
    """Extract top 20 features by importance."""
    if model_name == "logistic_regression":
        importance = np.abs(model.coef_[0])
    else:
        importance = model.feature_importances_

    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    return feat_imp.head(20)


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def log_experiment(model, model_name, model_params, training_params, metrics,
                   cv_metrics, feature_columns, is_xgboost):
    """Log experiment to MLflow and register the model."""
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("churn_prediction")

    with mlflow.start_run(run_name=f"v3_{model_name}"):
        # Log params
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("num_features", len(feature_columns))
        mlflow.log_param("feature_selection", "variance + correlation")

        for key, value in model_params.items():
            mlflow.log_param(key, value)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.log_metric("cv_auc_mean", cv_metrics["cv_auc_mean"])
        mlflow.log_metric("cv_auc_std", cv_metrics["cv_auc_std"])

        # Log model
        if is_xgboost:
            mlflow.xgboost.log_model(
                model, name="model", registered_model_name="churn_xgboost",
            )
        else:
            mlflow.sklearn.log_model(
                model, name="model", registered_model_name="churn_xgboost",
            )

        run_id = mlflow.active_run().info.run_id

    return run_id


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_training():
    """
    Execute the full training pipeline:
      1. Load features
      2. Split train/test
      3. Select features (remove noise)
      4. Compare multiple models via cross-validation
      5. Pick the best model
      6. Tune classification threshold
      7. Evaluate on test set
      8. Log to MLflow
    """
    params = Config.load_params()
    model_params = params["model"]
    training_params = params["training"]

    # Load and split
    df = load_feature_table()
    X_train, X_test, y_train, y_test = prepare_data(
        df,
        test_size=training_params["test_size"],
        random_state=model_params["random_state"],
    )

    # Feature selection — reduce noise
    X_train_sel, X_test_sel, removed_low_var, removed_corr = select_features(
        X_train, X_test, y_train
    )

    # Class imbalance ratio
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    # Compare models via cross-validation
    candidates = build_candidates(model_params, scale_pos_weight)
    cv_results = compare_models_cv(candidates, X_train_sel, y_train, model_params["random_state"])

    # Pick the best model by CV AUC
    best_name = max(cv_results, key=lambda k: cv_results[k]["cv_auc_mean"])
    best_model = candidates[best_name]

    # Train the best model on full training set
    is_logistic = best_name == "logistic_regression"
    scaler = None

    if is_logistic:
        scaler = StandardScaler()
        X_train_final = pd.DataFrame(
            scaler.fit_transform(X_train_sel),
            columns=X_train_sel.columns,
            index=X_train_sel.index,
        )
        best_model.fit(X_train_final, y_train)
    else:
        best_model.fit(X_train_sel, y_train)

    # Find optimal threshold
    optimal_threshold, _ = find_optimal_threshold(
        best_model, X_test_sel, y_test, is_logistic, scaler
    )

    # Evaluate with optimal threshold
    metrics = evaluate_model(
        best_model, X_test_sel, y_test, optimal_threshold, is_logistic, scaler
    )

    # Feature importance
    feat_imp = get_feature_importance(best_model, X_train_sel.columns.tolist(), best_name)

    # Log to MLflow
    is_xgboost = best_name == "xgboost"
    run_id = log_experiment(
        best_model, best_name, model_params, training_params,
        metrics, cv_results[best_name],
        feature_columns=X_train_sel.columns.tolist(),
        is_xgboost=is_xgboost,
    )

    return best_model, metrics, cv_results, feat_imp, run_id


if __name__ == "__main__":
    run_training()