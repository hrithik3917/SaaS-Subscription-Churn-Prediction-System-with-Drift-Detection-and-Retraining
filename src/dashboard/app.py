"""
app.py - Churn Prediction Dashboard (v3)

Multi-page Streamlit dashboard for the MLOps churn prediction pipeline.

Pages:
    1. Overview    — KPIs, risk distribution, charts
    2. Customers   — Browse and filter customers by risk
    3. Predict     — Cascading filters to select customer, or enter custom data
    4. Model       — Experiment history, feature importance
    5. Drift       — Drift detection report

Usage:
    set PYTHONPATH=C:\\Users\\hrith\\Desktop\\Capstone_project
    streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import plotly.express as px
import plotly.graph_objects as go
from src.utils.config import Config
from src.utils.db import engine


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Detect theme and set colors accordingly
# ---------------------------------------------------------------------------

COLORS = {
    "primary": "#6366F1",
    "high": "#EF4444",
    "medium": "#F59E0B",
    "low": "#10B981",
    "blue": "#6366F1",
    "teal": "#14B8A6",
}
RISK_COLOR_MAP = {"high": COLORS["high"], "medium": COLORS["medium"], "low": COLORS["low"]}


# ---------------------------------------------------------------------------
# Minimal CSS — works in both light and dark themes
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px;
        padding: 12px 16px;
    }

    .risk-high {
        background: rgba(239,68,68,0.15); color: #EF4444;
        padding: 4px 14px; border-radius: 12px; font-weight: 600;
        display: inline-block;
    }
    .risk-medium {
        background: rgba(245,158,11,0.15); color: #F59E0B;
        padding: 4px 14px; border-radius: 12px; font-weight: 600;
        display: inline-block;
    }
    .risk-low {
        background: rgba(16,185,129,0.15); color: #10B981;
        padding: 4px 14px; border-radius: 12px; font-weight: 600;
        display: inline-block;
    }

    .detail-label {
        font-size: 13px; opacity: 0.6; margin-bottom: 2px;
    }
    .detail-value {
        font-size: 16px; font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly theme helper — transparent backgrounds, auto text color
# ---------------------------------------------------------------------------

def chart_layout(height=320, **kwargs):
    """Standard plotly layout that works in both light and dark mode."""
    base = dict(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=40, l=40, r=20),
    )
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Model loading (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load production model from MLflow. Tries sklearn first, then xgboost."""
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    # FIX: updated registry name from churn_xgboost → churn_predictor
    try:
        return mlflow.sklearn.load_model("models:/churn_predictor/latest")
    except Exception:
        pass
    try:
        return mlflow.xgboost.load_model("models:/churn_predictor/latest")
    except Exception:
        return None


def get_feature_names(m):
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    elif hasattr(m, "get_booster"):
        return m.get_booster().feature_names
    return None


# ---------------------------------------------------------------------------
# Data loading (cached with TTL)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_features():
    with engine.connect() as conn:
        return pd.read_sql("SELECT * FROM churn_features", conn)


@st.cache_data(ttl=300)
def load_accounts():
    with engine.connect() as conn:
        return pd.read_sql("SELECT * FROM accounts", conn)


@st.cache_data(ttl=300)
def load_drift_report():
    path = Config.INTERIM_DATA_DIR / "drift_report.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def align_features(df, m):
    """Ensure DataFrame columns match model's expected features in correct order."""
    names = get_feature_names(m)
    if names is not None:
        for col in names:
            if col not in df.columns:
                df[col] = 0
        df = df[names]
    return df


def score_all(m, features_df):
    """Score all customers and return predictions DataFrame."""
    X = features_df.drop(columns=["account_id", "churn_flag"])
    X = align_features(X, m)
    probas = m.predict_proba(X)[:, 1]

    return pd.DataFrame({
        "account_id": features_df["account_id"],
        "churn_probability": np.round(probas, 4),
        "churn_prediction": (probas >= 0.5).astype(int),
        "risk_level": pd.cut(probas, bins=[-0.01, 0.4, 0.7, 1.01],
                             labels=["low", "medium", "high"]),
    })


def predict_customer(m, features_df, account_id):
    """Get prediction for one customer using their real feature data."""
    row = features_df[features_df["account_id"] == account_id]
    if row.empty:
        return None

    X = row.drop(columns=["account_id", "churn_flag"])
    X = align_features(X, m)
    prob = float(m.predict_proba(X)[0][1])
    risk = "high" if prob >= 0.7 else "medium" if prob >= 0.4 else "low"

    return {"probability": round(prob, 4), "prediction": int(prob >= 0.5), "risk": risk}


def build_custom_features(model, features_df, inputs):
    """
    Build a feature vector for custom prediction.

    Strategy:
      1. Start from the median of ONLY the features the model actually uses
         (not all 68 — avoids feeding the model features it was never trained on).
      2. Override every feature that can be derived from user inputs.
      3. Only apply overrides that exist in the model's feature set so nothing
         is silently ignored.

    This ensures extreme inputs (e.g. 0 usage, 50 tickets) produce extreme
    predictions rather than being diluted by unrelated median values.
    """
    model_features = get_feature_names(model)
    if model_features is None:
        return None

    # Step 1 — baseline: median of model features only
    feat_cols = features_df.drop(columns=["account_id", "churn_flag"])
    available = [f for f in model_features if f in feat_cols.columns]
    fd = feat_cols[available].median().to_dict()

    # Ensure every model feature exists (fill any missing with 0)
    for f in model_features:
        if f not in fd:
            fd[f] = 0.0

    # Unpack inputs
    seats        = inputs["seats"]
    plan         = inputs["plan"]
    tenure       = inputs["tenure"]
    is_trial     = inputs["is_trial"]
    avg_mrr      = inputs["avg_mrr"]
    usage_mins   = inputs["usage_mins"]
    features_used = inputs["features_used"]
    error_rate   = inputs["error_rate"]
    tickets      = inputs["tickets"]
    satisfaction = inputs["satisfaction"]
    escalation   = inputs["escalation"]
    downgraded   = inputs["downgraded"]

    plan_map = {"Basic": 0, "Pro": 1, "Enterprise": 2}

    # Derived values — computed once so they're consistent across all overrides
    # Usage trend: ratio of recent (last 30 days) to older (first 60 days) usage
    # Low usage → ratio well below 1.0 (disengagement signal)
    if usage_mins < 100:
        usage_trend = 0.15
        days_since_usage = 90
        recent_usage = usage_mins * 0.1
    elif usage_mins < 400:
        usage_trend = 0.40
        days_since_usage = 30
        recent_usage = usage_mins * 0.35
    elif usage_mins < 1000:
        usage_trend = 0.85
        days_since_usage = 10
        recent_usage = usage_mins * 0.70
    else:
        usage_trend = 1.30
        days_since_usage = 3
        recent_usage = usage_mins * 0.80

    # Ticket trend: ratio of recent to older support volume
    if tickets > 15:
        ticket_trend = 4.0
        days_since_ticket = 3
    elif tickets > 8:
        ticket_trend = 2.5
        days_since_ticket = 7
    elif tickets > 3:
        ticket_trend = 1.2
        days_since_ticket = 20
    else:
        ticket_trend = 0.5
        days_since_ticket = 60

    # Revenue risk: high when MRR is LOW *and* usage is declining
    # (1 - mrr_ratio) makes low MRR = high base risk
    # (1 - usage_trend) capped at 0 makes declining usage multiply that risk
    mrr_ratio = min(avg_mrr / 5000.0, 1.0)
    revenue_risk = (1.0 - mrr_ratio) * max(0.0, 1.0 - usage_trend)

    # Frustration: high when usage is LOW but tickets are HIGH
    frustration = (1.0 - min(usage_mins / 10000.0, 1.0)) * min(tickets / 50.0, 1.0)

    # Step 2 — build the full override dictionary
    overrides = {
        # ── Account features ──────────────────────────────────────────────
        "seats":               float(seats),
        "plan_tier_encoded":   float(plan_map[plan]),
        "tenure_days":         float(tenure),
        "is_trial":            1.0 if is_trial == "Yes" else 0.0,
        "is_new_customer":     1.0 if tenure <= 90  else 0.0,
        "is_established":      1.0 if tenure >= 365 else 0.0,

        # ── Subscription features ─────────────────────────────────────────
        "avg_mrr":             float(avg_mrr),
        "max_mrr":             float(avg_mrr * 1.2),
        "total_arr":           float(avg_mrr * 12),
        "avg_seats_per_sub":   float(seats),
        "has_upgrade":         0.0,
        "has_downgrade":       1.0 if downgraded == "Yes" else 0.0,
        "recent_has_downgrade":1.0 if downgraded == "Yes" else 0.0,
        "recent_has_upgrade":  0.0,
        # mrr_change_ratio: recent MRR / older MRR — downgrade = drop in ratio
        "mrr_change_ratio":    0.35 if downgraded == "Yes" else 1.05,
        "auto_renew_ratio":    0.15 if downgraded == "Yes" else 0.90,
        "sub_velocity":        0.05 if downgraded == "Yes" else 0.30,
        "trial_ratio":         1.0  if is_trial == "Yes" else 0.0,
        "is_monthly_billing":  1.0  if plan == "Basic" else 0.0,
        "latest_plan_tier":    float(plan_map[plan]),
        "days_since_last_sub": 30.0 if downgraded == "Yes" else 5.0,
        "active_subscriptions":0.0  if downgraded == "Yes" else 1.0,

        # ── Usage features ────────────────────────────────────────────────
        "total_usage_minutes":    float(usage_mins),
        "avg_daily_usage_mins":   float(usage_mins / 90.0),
        "total_usage_events":     float(max(usage_mins / 15.0, 0)),
        "avg_usage_count":        float(usage_mins / 90.0),
        "unique_features_used":   float(features_used),
        "error_rate":             float(error_rate),
        "total_errors":           float(error_rate * max(usage_mins / 10.0, 1)),
        "beta_feature_ratio":     0.05,
        "recent_usage_minutes":   float(recent_usage),
        "recent_usage_events":    float(max(recent_usage / 15.0, 0)),
        "recent_error_rate":      float(min(error_rate * 1.5, 1.0) if error_rate > 0.1 else error_rate),
        "recent_avg_daily_mins":  float(recent_usage / 30.0),
        "recent_features_used":   float(features_used * 0.7 if usage_trend < 0.5 else features_used),
        "usage_trend_ratio":      float(usage_trend),
        "feature_diversity_trend":0.4 if features_used < 5 else 1.0,
        "days_since_last_usage":  float(days_since_usage),

        # ── Support features ──────────────────────────────────────────────
        "ticket_count":               float(tickets),
        "avg_resolution_hours":       72.0 if satisfaction < 2.0 else (36.0 if satisfaction < 3.5 else 10.0),
        "avg_first_response_mins":    300.0 if satisfaction < 2.0 else 60.0,
        "avg_satisfaction":           float(satisfaction),
        "escalation_count":           float(escalation * tickets),
        "escalation_rate":            float(escalation),
        "high_priority_ratio":        0.85 if escalation > 0.5 else 0.15,
        "unresolved_tickets":         float(tickets * (0.65 if satisfaction < 2.5 else 0.08)),
        "unresolved_ratio":           0.65 if satisfaction < 2.5 else 0.08,
        "recent_ticket_count":        float(tickets * 0.85),
        "recent_avg_resolution_hours":72.0 if satisfaction < 2.0 else 10.0,
        "recent_avg_satisfaction":    float(satisfaction),
        "recent_escalation_rate":     float(escalation),
        "ticket_trend_ratio":         float(ticket_trend),
        "days_since_last_ticket":     float(days_since_ticket),

        # ── Interaction / derived features ────────────────────────────────
        "frustration_score":      float(frustration),
        "revenue_risk_score":     float(revenue_risk),
        "tickets_per_tenure_day": float(tickets / max(tenure, 1)),
        "usage_per_seat":         float(usage_mins / max(seats, 1)),
    }

    # Step 3 — apply overrides only for features the model actually uses
    for key, value in overrides.items():
        if key in fd:
            fd[key] = float(value)

    # Step 4 — zero out all one-hot columns (industry / referral)
    for col in list(fd.keys()):
        if col.startswith("industry_") or col.startswith("referral_"):
            fd[col] = 0.0

    return fd


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("### 📊 Churn Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation",
                         ["Overview", "Customers", "Predict", "Model", "Drift"],
                         label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("SaaS Churn MLOps Pipeline\nRavenStack Dataset")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

model = load_model()
features_df = load_features()
accounts_df = load_accounts()


# ═══════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.markdown("## Pipeline Overview")
    st.caption("Real-time snapshot of model performance and customer risk distribution.")

    if model is None:
        st.error("Model not loaded. Start MLflow server on port 5000.")
        st.stop()

    predictions = score_all(model, features_df)
    merged = accounts_df.merge(predictions, on="account_id", how="left")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", len(merged))
    c2.metric("High Risk", int((merged["risk_level"] == "high").sum()))
    c3.metric("Medium Risk", int((merged["risk_level"] == "medium").sum()))
    c4.metric("Actual Churn Rate", f"{merged['churn_flag'].mean() * 100:.1f}%")

    st.markdown("---")

    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("#### Risk Distribution")
        rc = merged["risk_level"].value_counts().reset_index()
        rc.columns = ["Risk Level", "Count"]
        fig = px.bar(rc, x="Risk Level", y="Count", color="Risk Level",
                     color_discrete_map=RISK_COLOR_MAP, text="Count")
        fig.update_layout(**chart_layout(), showlegend=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        st.markdown("#### Churn Probability Distribution")
        fig = px.histogram(merged, x="churn_probability", nbins=20,
                           color_discrete_sequence=[COLORS["blue"]])
        fig.update_layout(**chart_layout())
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("#### Risk by Industry")
    ir = merged.groupby(["industry", "risk_level"]).size().reset_index(name="count")
    fig = px.bar(ir, x="industry", y="count", color="risk_level",
                 color_discrete_map=RISK_COLOR_MAP, barmode="stack")
    fig.update_layout(**chart_layout(height=340), legend_title_text="Risk")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE 2: CUSTOMERS
# ═══════════════════════════════════════════════════════════════════════

elif page == "Customers":
    st.markdown("## Customer Risk Explorer")
    st.caption("All customers scored by churn risk. Filter and export.")

    if model is None:
        st.error("Model not loaded.")
        st.stop()

    predictions = score_all(model, features_df)
    merged = accounts_df.merge(predictions, on="account_id", how="left")
    merged = merged.merge(
        features_df[["account_id", "tenure_days", "avg_mrr", "ticket_count",
                      "total_usage_minutes"]],
        on="account_id", how="left",
    )

    f1, f2, f3 = st.columns(3)
    with f1:
        risk_f = st.multiselect("Risk Level", ["high", "medium", "low"],
                                default=["high", "medium", "low"])
    with f2:
        ind_f = st.multiselect("Industry", sorted(merged["industry"].unique()),
                               default=sorted(merged["industry"].unique()))
    with f3:
        plan_f = st.multiselect("Plan Tier", sorted(merged["plan_tier"].unique()),
                                default=sorted(merged["plan_tier"].unique()))

    filtered = merged[
        (merged["risk_level"].isin(risk_f)) &
        (merged["industry"].isin(ind_f)) &
        (merged["plan_tier"].isin(plan_f))
    ].sort_values("churn_probability", ascending=False)

    st.markdown(f"Showing **{len(filtered)}** of {len(merged)} customers")

    display = filtered[[
        "account_id", "account_name", "industry", "plan_tier",
        "churn_probability", "risk_level", "churn_flag",
        "tenure_days", "avg_mrr", "ticket_count", "total_usage_minutes",
    ]].copy()
    display.columns = ["ID", "Name", "Industry", "Plan", "Churn Prob",
                       "Risk", "Churned", "Tenure", "Avg MRR", "Tickets", "Usage Mins"]

    st.dataframe(display, use_container_width=True, height=500)
    st.download_button("Download CSV", display.to_csv(index=False),
                       "churn_predictions.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 3: PREDICT
# ═══════════════════════════════════════════════════════════════════════

elif page == "Predict":
    st.markdown("## Predict Customer Churn")

    if model is None:
        st.error("Model not loaded.")
        st.stop()

    tab1, tab2 = st.tabs(["🔍 Select Customer", "✏️ Custom Prediction"])

    # ----- Tab 1: Select existing customer with cascading filters -----
    with tab1:
        st.markdown("##### Filter and select a customer")

        sel1, sel2, sel3 = st.columns(3)

        with sel1:
            plan_options = sorted(accounts_df["plan_tier"].unique())
            sel_plan = st.selectbox("Plan Tier", plan_options, key="pred_plan")

        filtered_by_plan = accounts_df[accounts_df["plan_tier"] == sel_plan]

        with sel2:
            ind_options = sorted(filtered_by_plan["industry"].unique())
            sel_industry = st.selectbox("Industry", ind_options, key="pred_ind")

        filtered_by_ind = filtered_by_plan[filtered_by_plan["industry"] == sel_industry]

        with sel3:
            cust_options = filtered_by_ind[["account_id", "account_name"]].copy()
            cust_options["label"] = cust_options["account_name"] + " (" + cust_options["account_id"] + ")"
            sel_customer = st.selectbox("Customer", cust_options["label"].tolist(), key="pred_cust")

        selected_id = cust_options[cust_options["label"] == sel_customer]["account_id"].iloc[0]

        if st.button("Get Prediction", type="primary", key="btn_predict"):
            result = predict_customer(model, features_df, selected_id)
            cust_info = accounts_df[accounts_df["account_id"] == selected_id].iloc[0]
            cust_features = features_df[features_df["account_id"] == selected_id].iloc[0]

            if result is not None:
                st.markdown("---")

                # Customer details card
                st.markdown("##### Customer Profile")
                d1, d2, d3, d4, d5 = st.columns(5)
                d1.markdown(f"<div class='detail-label'>Company</div>"
                            f"<div class='detail-value'>{cust_info['account_name']}</div>",
                            unsafe_allow_html=True)
                d2.markdown(f"<div class='detail-label'>Industry</div>"
                            f"<div class='detail-value'>{cust_info['industry']}</div>",
                            unsafe_allow_html=True)
                d3.markdown(f"<div class='detail-label'>Plan</div>"
                            f"<div class='detail-value'>{cust_info['plan_tier']}</div>",
                            unsafe_allow_html=True)
                d4.markdown(f"<div class='detail-label'>Seats</div>"
                            f"<div class='detail-value'>{cust_info['seats']}</div>",
                            unsafe_allow_html=True)
                d5.markdown(f"<div class='detail-label'>Country</div>"
                            f"<div class='detail-value'>{cust_info['country']}</div>",
                            unsafe_allow_html=True)

                st.markdown("")

                # Key metrics row
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Tenure", f"{int(cust_features.get('tenure_days', 0))} days")
                k2.metric("Avg MRR", f"${cust_features.get('avg_mrr', 0):,.0f}")
                k3.metric("Tickets", f"{int(cust_features.get('ticket_count', 0))}")
                k4.metric("Usage", f"{cust_features.get('total_usage_minutes', 0):,.0f} mins")

                st.markdown("---")

                # Prediction result
                st.markdown("##### Prediction Result")
                r1, r2, r3 = st.columns(3)
                r1.metric("Churn Probability", f"{result['probability']:.1%}")
                r2.metric("Prediction",
                          "Will Churn" if result["prediction"] == 1 else "Will Stay")
                r3.markdown(
                    f"**Risk Level**<br><span class='risk-{result['risk']}'>"
                    f"{result['risk'].upper()}</span>",
                    unsafe_allow_html=True,
                )

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=result["probability"] * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": RISK_COLOR_MAP.get(result["risk"], "#6B7280")},
                        "steps": [
                            {"range": [0, 40], "color": "rgba(16,185,129,0.15)"},
                            {"range": [40, 70], "color": "rgba(245,158,11,0.15)"},
                            {"range": [70, 100], "color": "rgba(239,68,68,0.15)"},
                        ],
                    },
                ))
                fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)",
                                  margin=dict(t=30, b=10, l=30, r=30))
                st.plotly_chart(fig, use_container_width=True)

                # Actual churn status (ground truth from dataset — for evaluation only)
                actually_churned = bool(cust_info["churn_flag"])
                if actually_churned:
                    st.info("📌 **Ground truth (dataset):** This customer actually churned — for evaluation only.")
                else:
                    st.info("📌 **Ground truth (dataset):** This customer did not churn — for evaluation only.")

                # Expandable feature details
                with st.expander("View all feature values"):
                    feat_data = cust_features.drop(["account_id", "churn_flag"])
                    non_zero = feat_data[feat_data != 0].sort_values(ascending=False)
                    st.dataframe(
                        non_zero.reset_index().rename(columns={"index": "Feature", 0: "Value"}),
                        use_container_width=True,
                    )

    # ----- Tab 2: Custom data entry -----
    with tab2:
        st.markdown("##### Simulate a hypothetical customer")
        st.caption(
            "Adjust the sliders to describe a customer profile. "
            "All signals — usage, tickets, MRR trend, satisfaction — feed into the prediction together."
        )

        with st.form("custom_predict"):
            p1, p2, p3 = st.columns(3)

            with p1:
                st.markdown("**Account**")
                seats = st.slider("Seats", 1, 100, 10)
                plan = st.selectbox("Plan", ["Basic", "Pro", "Enterprise"], key="cp_plan")
                tenure = st.slider("Tenure (days)", 1, 730, 180)
                is_trial = st.selectbox("Trial?", ["No", "Yes"], key="cp_trial")

            with p2:
                st.markdown("**Usage & Revenue**")
                avg_mrr = st.slider("Avg MRR ($)", 0, 5000, 800)
                usage_mins = st.slider("Total Usage (minutes)", 0, 10000, 1500)
                features_used = st.slider("Unique Features Used", 0, 40, 12)
                error_rate = st.slider("Error Rate", 0.0, 0.5, 0.03, 0.01)

            with p3:
                st.markdown("**Support & Health**")
                tickets = st.slider("Support Tickets", 0, 50, 4)
                satisfaction = st.slider("Avg Satisfaction (1–5)", 1.0, 5.0, 3.5, 0.1)
                escalation = st.slider("Escalation Rate", 0.0, 1.0, 0.1, 0.05)
                downgraded = st.selectbox("Downgraded?", ["No", "Yes"], key="cp_dg")

            submitted = st.form_submit_button("Predict Churn Risk", type="primary",
                                              use_container_width=True)

        if submitted:
            inputs = dict(
                seats=seats, plan=plan, tenure=tenure, is_trial=is_trial,
                avg_mrr=avg_mrr, usage_mins=usage_mins, features_used=features_used,
                error_rate=error_rate, tickets=tickets, satisfaction=satisfaction,
                escalation=escalation, downgraded=downgraded,
            )

            fd = build_custom_features(model, features_df, inputs)

            if fd is None:
                st.error("Could not determine model features. Is the model loaded correctly?")
                st.stop()

            df = pd.DataFrame([fd])
            df = align_features(df, model)

            prob = float(model.predict_proba(df)[0][1])
            risk = "high" if prob >= 0.7 else "medium" if prob >= 0.4 else "low"

            st.markdown("---")
            st.markdown("##### Prediction Result")

            r1, r2, r3 = st.columns(3)
            r1.metric("Churn Probability", f"{prob:.1%}")
            r2.metric("Prediction", "Will Churn" if prob >= 0.5 else "Will Stay")
            r3.markdown(f"**Risk Level**<br><span class='risk-{risk}'>{risk.upper()}</span>",
                        unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": RISK_COLOR_MAP.get(risk, "#6B7280")},
                    "steps": [
                        {"range": [0, 40], "color": "rgba(16,185,129,0.15)"},
                        {"range": [40, 70], "color": "rgba(245,158,11,0.15)"},
                        {"range": [70, 100], "color": "rgba(239,68,68,0.15)"},
                    ],
                },
            ))
            fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=30, b=10, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

            # Show key signals so the user understands what drove the prediction
            with st.expander("Key signals sent to model"):
                model_features = get_feature_names(model)
                signal_features = [
                    "usage_trend_ratio", "ticket_trend_ratio", "frustration_score",
                    "revenue_risk_score", "escalation_rate", "avg_satisfaction",
                    "mrr_change_ratio", "unresolved_ratio", "days_since_last_usage",
                    "auto_renew_ratio", "ticket_count", "avg_mrr",
                ]
                visible = {k: round(fd[k], 4) for k in signal_features if k in fd}
                sig_df = pd.DataFrame(
                    visible.items(), columns=["Feature", "Value"]
                ).sort_values("Value", ascending=False)
                st.dataframe(sig_df, use_container_width=True, hide_index=True)
                st.caption(
                    f"Model uses {len(model_features)} features total. "
                    f"{sum(1 for f in model_features if f in fd and fd[f] != features_df[f].median() if f in features_df.columns)} "
                    f"were overridden from median based on your inputs."
                )


# ═══════════════════════════════════════════════════════════════════════
# PAGE 4: MODEL
# ═══════════════════════════════════════════════════════════════════════

elif page == "Model":
    st.markdown("## Model Performance")
    st.caption("Experiment history and model analysis.")

    st.markdown("#### Improvement Journey")
    journey = pd.DataFrame({
        "Version": ["v1 Baseline", "v2 Time-windowed", "v3 Optimized"],
        "AUC-ROC": [0.5303, 0.5385, 0.6352],
        "Precision": [0.2000, 0.2727, 0.5000],
        "Recall": [0.1364, 0.1364, 0.4091],
        "F1 Score": [0.1622, 0.1818, 0.4500],
        "Changes": [
            "All-time features, default XGBoost",
            "90-day windows, trend features, regularization",
            "Feature selection, model comparison, threshold tuning",
        ],
    })
    st.dataframe(journey, use_container_width=True, hide_index=True)

    fig = go.Figure()
    for metric, color in zip(
        ["AUC-ROC", "Precision", "Recall", "F1 Score"],
        [COLORS["blue"], COLORS["low"], COLORS["medium"], COLORS["primary"]],
    ):
        fig.add_trace(go.Scatter(
            x=journey["Version"], y=journey[metric],
            mode="lines+markers", name=metric,
            line=dict(color=color, width=2), marker=dict(size=8),
        ))
    fig.update_layout(
        **chart_layout(350),
        yaxis=dict(range=[0, 1], title="Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown("#### Current Model (v3 — Random Forest)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AUC-ROC", "0.6352")
    m2.metric("Precision", "0.5000")
    m3.metric("Recall", "0.4091")
    m4.metric("F1 Score", "0.4500")
    m5.metric("Threshold", "0.4758")

    st.markdown("---")

    st.markdown("#### Top Features by Importance")
    if model is not None:
        try:
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
            elif hasattr(model, "coef_"):
                imp = np.abs(model.coef_[0])
            else:
                imp = None

            names = get_feature_names(model)

            if imp is not None and names is not None:
                fi = pd.DataFrame({"Feature": names, "Importance": imp})
                fi = fi.sort_values("Importance", ascending=True).tail(15)

                fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                             color_discrete_sequence=[COLORS["blue"]])
                fig.update_layout(**chart_layout(420), yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load feature importance: {e}")
    else:
        st.warning("Model not loaded.")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 5: DRIFT
# ═══════════════════════════════════════════════════════════════════════

elif page == "Drift":
    st.markdown("## Drift Detection")
    st.caption("Monitor data distribution changes.")

    report = load_drift_report()

    if report is not None:
        verdict = report["verdict"]
        cfg = {
            "no_drift": ("✅", "No significant drift. Model is stable.", "success"),
            "moderate_drift": ("⚠️", "Moderate drift. Monitor closely.", "warning"),
            "significant_drift": ("🔴", "Significant drift. Retraining recommended.", "error"),
        }
        icon, msg, alert = cfg.get(verdict, ("ℹ️", "", "info"))
        getattr(st, alert)(f"{icon} **{verdict.replace('_', ' ').title()}** — {msg}")

        d1, d2, d3 = st.columns(3)
        d1.metric("Features Tested", report["total_features_tested"])
        d2.metric("Features Drifted", report["features_with_drift"])
        d3.metric("Drift Ratio", f"{report['drift_ratio']:.1%}")

        st.markdown("---")

        if report["drifted_features"]:
            st.markdown("#### Drifted Features")
            drifted = [{
                "Feature": f,
                "Test": report["feature_details"][f].get("test", ""),
                "Statistic": round(report["feature_details"][f].get("statistic", 0), 4),
                "P-Value": round(report["feature_details"][f].get("p_value", 0), 6),
            } for f in report["drifted_features"]]
            st.dataframe(pd.DataFrame(drifted).sort_values("P-Value"),
                         use_container_width=True, hide_index=True)

        with st.expander("View all feature details"):
            all_f = [{
                "Feature": f,
                "Test": d.get("test", ""),
                "Statistic": round(d.get("statistic", 0), 4),
                "P-Value": round(d.get("p_value", 0), 6),
                "Drifted": "Yes" if d.get("drift_detected") else "No",
            } for f, d in report["feature_details"].items()]
            st.dataframe(pd.DataFrame(all_f).sort_values("P-Value"),
                         use_container_width=True, height=400, hide_index=True)

        st.caption(f"Report generated: {report['timestamp']}")

    else:
        st.info("No drift report found. Run drift detection first:")
        st.code("python -m src.drift_detection.detect_drift", language="bash")

        if st.button("Run Drift Detection Now"):
            with st.spinner("Running..."):
                from src.drift_detection.detect_drift import run_drift_detection
                run_drift_detection()
                st.cache_data.clear()
                st.rerun()