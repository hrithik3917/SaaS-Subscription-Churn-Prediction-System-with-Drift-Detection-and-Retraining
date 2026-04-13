"""
app.py - Churn Prediction Dashboard (v3)

Multi-page Streamlit dashboard for the MLOps churn prediction pipeline.

Pages:
    1. Overview    — KPIs, risk distribution, charts
    2. Customers   — Browse and filter customers by risk
    3. Predict     — Select customer OR explore scenarios
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

COLORS = {
    "primary": "#6366F1",
    "high":    "#EF4444",
    "medium":  "#F59E0B",
    "low":     "#10B981",
    "blue":    "#6366F1",
    "teal":    "#14B8A6",
}
RISK_COLOR_MAP = {
    "high":   COLORS["high"],
    "medium": COLORS["medium"],
    "low":    COLORS["low"],
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px;
        padding: 12px 16px;
    }

    .risk-high   { background:rgba(239,68,68,0.15);  color:#EF4444;
                   padding:4px 14px; border-radius:12px; font-weight:600; display:inline-block; }
    .risk-medium { background:rgba(245,158,11,0.15); color:#F59E0B;
                   padding:4px 14px; border-radius:12px; font-weight:600; display:inline-block; }
    .risk-low    { background:rgba(16,185,129,0.15); color:#10B981;
                   padding:4px 14px; border-radius:12px; font-weight:600; display:inline-block; }

    .detail-label { font-size:13px; opacity:0.6; margin-bottom:2px; }
    .detail-value { font-size:16px; font-weight:500; }

    .customer-card {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 8px;
    }
    .card-name  { font-size:15px; font-weight:600; margin-bottom:4px; }
    .card-label { font-size:12px; opacity:0.55; }
    .card-value { font-size:14px; font-weight:500; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly helper
# ---------------------------------------------------------------------------

def chart_layout(height=320, **kwargs):
    base = dict(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=40, l=40, r=20),
    )
    base.update(kwargs)
    return base


def gauge_chart(prob, risk):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 10}},
            "bar":  {"color": RISK_COLOR_MAP.get(risk, "#6B7280"), "thickness": 0.3},
            "steps": [
                {"range": [0,  40], "color": "rgba(16,185,129,0.12)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.12)"},
                {"range": [70,100], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line":  {"color": RISK_COLOR_MAP.get(risk, "#6B7280"), "width": 3},
                "thickness": 0.8,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=10, l=20, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """
    Load production model from MLflow.
    Tries churn_predictor first (current name), falls back to churn_xgboost
    (legacy name). Within each, tries sklearn then xgboost flavor.
    """
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    for registry_name in ["churn_predictor", "churn_xgboost"]:
        for loader in [mlflow.sklearn.load_model, mlflow.xgboost.load_model]:
            try:
                return loader(f"models:/{registry_name}/latest")
            except Exception:
                continue
    return None


def get_feature_names(m):
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    elif hasattr(m, "get_booster"):
        return m.get_booster().feature_names
    return None


# ---------------------------------------------------------------------------
# Data loading
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
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def align_features(df, m):
    """Reorder / fill columns to match model's expected feature set."""
    names = get_feature_names(m)
    if names is not None:
        for col in names:
            if col not in df.columns:
                df[col] = 0
        df = df[names]
    return df


def score_all(m, features_df):
    X = features_df.drop(columns=["account_id", "churn_flag"])
    X = align_features(X, m)
    probas = m.predict_proba(X)[:, 1]
    return pd.DataFrame({
        "account_id":       features_df["account_id"],
        "churn_probability": np.round(probas, 4),
        "churn_prediction":  (probas >= 0.5).astype(int),
        "risk_level":        pd.cut(
            probas,
            bins=[-0.01, 0.4, 0.7, 1.01],
            labels=["low", "medium", "high"],
        ),
    })


def predict_customer(m, features_df, account_id):
    row = features_df[features_df["account_id"] == account_id]
    if row.empty:
        return None
    X = row.drop(columns=["account_id", "churn_flag"])
    X = align_features(X, m)
    prob = float(m.predict_proba(X)[0][1])
    risk = "high" if prob >= 0.7 else "medium" if prob >= 0.4 else "low"
    return {"probability": round(prob, 4), "prediction": int(prob >= 0.5), "risk": risk}


# ---------------------------------------------------------------------------
# Scenario Explorer helpers  (Tab 2)
# ---------------------------------------------------------------------------

# Maps plain-English levels to numeric targets used for nearest-neighbour search
# Usage and ticket HARD RANGE filters — these are applied first
# so the returned customers actually match the selected profile
USAGE_RANGES = {
    "Low (< 1900 mins)":       (0,    1922),
    "Medium (1900–3100 mins)": (1922,  3118),
    "High (> 3100 mins)":     (3118, 999999),
}
TICKET_RANGES = {
    "Low (0–3 tickets)":  (0, 3),
    "Medium (4–5 tickets)": (4, 5),
    "High (> 6 tickets)": (6, 999),
}
USAGE_LABELS  = list(USAGE_RANGES.keys())
TICKET_LABELS = list(TICKET_RANGES.keys())


def find_matching_customers(features_df, accounts_df, plan, industry,
                             usage_label, support_label, n=3):
    """
    Return the n real customers whose profile best matches the selected
    plan, industry, usage level, and support load.

    Matching strategy (usage + tickets are HARD filters — always respected)
    -----------------------------------------------------------------------
    1. Hard filter on usage range AND ticket range — these are non-negotiable.
       This guarantees returned customers actually match the selected profile.
    2. Within that pool, prefer plan_tier + industry match.
    3. If plan+industry yields < n rows, relax to plan only.
    4. If still < n, use the full usage+ticket filtered pool.
    5. Rank by churn probability descending so the most interesting
       (highest risk) customers surface first.
    """
    u_min, u_max = USAGE_RANGES[usage_label]
    t_min, t_max = TICKET_RANGES[support_label]

    # Attach account-level info
    acct_cols = ["account_id", "account_name", "plan_tier",
                 "industry", "seats", "country", "churn_flag"]
    acct_subset = accounts_df[[c for c in acct_cols if c in accounts_df.columns]]
    merged = features_df.merge(acct_subset, on="account_id", how="left",
                                suffixes=("", "_acct"))
    if "churn_flag_acct" in merged.columns:
        merged = merged.drop(columns=["churn_flag_acct"])

    # ── Step 1: Hard filter on usage AND tickets (always applied) ────
    usage_ticket_pool = merged[
        (merged["total_usage_minutes"] >= u_min) &
        (merged["total_usage_minutes"] <  u_max) &
        (merged["ticket_count"]        >= t_min) &
        (merged["ticket_count"]        <= t_max)
    ].copy()

    # Fallback: if the hard filter returns nothing, widen ticket range slightly
    if len(usage_ticket_pool) < n:
        usage_ticket_pool = merged[
            (merged["total_usage_minutes"] >= u_min) &
            (merged["total_usage_minutes"] <  u_max)
        ].copy()

    # Ultimate fallback: if still not enough, use whole dataset
    if len(usage_ticket_pool) < n:
        usage_ticket_pool = merged.copy()

    # ── Step 2: Prefer plan + industry within the usage/ticket pool ──
    strict = usage_ticket_pool[
        (usage_ticket_pool["plan_tier"] == plan) &
        (usage_ticket_pool["industry"]  == industry)
    ]

    if len(strict) >= n:
        pool = strict.copy()
    else:
        plan_only = usage_ticket_pool[usage_ticket_pool["plan_tier"] == plan]
        pool = plan_only.copy() if len(plan_only) >= n else usage_ticket_pool.copy()

    # ── Step 3: Score each candidate — predict churn probability ─────
    scored_rows = []
    for _, row in pool.iterrows():
        pred = predict_customer(model, features_df, row["account_id"])
        if pred:
            row = row.copy()
            row["_churn_prob"] = pred["probability"]
            scored_rows.append(row)

    if not scored_rows:
        return pd.DataFrame()

    scored_df = pd.DataFrame(scored_rows)

    # ── Step 4: Sort by churn probability descending ─────────────────
    # High-risk customers surface first — more informative for the user
    return scored_df.nlargest(n, "_churn_prob").reset_index(drop=True)


def render_customer_card(col, row, pred, rank):
    """Render one customer card inside a Streamlit column."""
    risk  = pred["risk"]
    prob  = pred["probability"]

    with col:
        # Header
        st.markdown(
            f"<div class='card-name'>#{rank} — {row.get('account_name', 'Unknown')}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<span class='risk-{risk}'>{risk.upper()}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        # Key metrics
        m1, m2 = st.columns(2)
        m1.metric("Churn Prob",  f"{prob:.1%}")
        m2.metric("Tenure",      f"{int(row.get('tenure_days', 0))} days")

        m3, m4 = st.columns(2)
        m3.metric("Avg MRR",     f"${row.get('avg_mrr', 0):,.0f}")
        m4.metric("Tickets",     f"{int(row.get('ticket_count', 0))}")

        m5, m6 = st.columns(2)
        m5.metric("Usage (mins)", f"{row.get('total_usage_minutes', 0):,.0f}")
        m6.metric("Satisfaction", f"{row.get('avg_satisfaction', 0):.1f} / 5")

        # Gauge
        st.plotly_chart(gauge_chart(prob, risk), use_container_width=True)

        # Ground truth badge
        churned = bool(row.get("churn_flag", 0))
        if churned:
            st.error("📌 Actually churned in dataset")
        else:
            st.success("📌 Did not churn in dataset")

        # Plan / industry info
        st.caption(
            f"Plan: {row.get('plan_tier','–')}  ·  "
            f"Industry: {row.get('industry','–')}  ·  "
            f"Seats: {int(row.get('seats', 0))}  ·  "
            f"Country: {row.get('country','–')}"
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("### 📊 Churn Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Customers", "Predict", "Model", "Drift"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("SaaS Churn MLOps Pipeline\nRavenStack Dataset")

# ---------------------------------------------------------------------------
# Load shared data
# ---------------------------------------------------------------------------

model       = load_model()
features_df = load_features()
accounts_df = load_accounts()


# ═══════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.markdown("## Pipeline Overview")
    st.caption("Real-time snapshot of model performance and customer risk distribution.")

    if model is None:
        st.error("Model not loaded. Start MLflow server:  mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db")
        st.stop()

    predictions = score_all(model, features_df)
    merged = accounts_df.merge(predictions, on="account_id", how="left")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers",  len(merged))
    c2.metric("High Risk",        int((merged["risk_level"] == "high").sum()))
    c3.metric("Medium Risk",      int((merged["risk_level"] == "medium").sum()))
    c4.metric("Actual Churn Rate",f"{merged['churn_flag'].mean() * 100:.1f}%")

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
        features_df[["account_id", "tenure_days", "avg_mrr",
                     "ticket_count", "total_usage_minutes"]],
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
        (merged["industry"].isin(ind_f))   &
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

    tab1, tab2 = st.tabs(["🔍 Select Customer", "🧪 Scenario Explorer"])

    # ───────────────────────────────────────────────────────────────────
    # TAB 1 — Select a real customer by cascading filters
    # ───────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("##### Filter and select a customer")

        sel1, sel2, sel3 = st.columns(3)
        with sel1:
            sel_plan = st.selectbox("Plan Tier",
                                    sorted(accounts_df["plan_tier"].unique()),
                                    key="pred_plan")

        filtered_by_plan = accounts_df[accounts_df["plan_tier"] == sel_plan]

        with sel2:
            sel_industry = st.selectbox("Industry",
                                        sorted(filtered_by_plan["industry"].unique()),
                                        key="pred_ind")

        filtered_by_ind = filtered_by_plan[
            filtered_by_plan["industry"] == sel_industry
        ]

        with sel3:
            cust_opts = filtered_by_ind[["account_id", "account_name"]].copy()
            cust_opts["label"] = (cust_opts["account_name"]
                                  + " (" + cust_opts["account_id"] + ")")
            sel_customer = st.selectbox("Customer",
                                        cust_opts["label"].tolist(),
                                        key="pred_cust")

        selected_id = cust_opts[
            cust_opts["label"] == sel_customer
        ]["account_id"].iloc[0]

        if st.button("Get Prediction", type="primary", key="btn_predict"):
            result       = predict_customer(model, features_df, selected_id)
            cust_info    = accounts_df[accounts_df["account_id"] == selected_id].iloc[0]
            cust_feats   = features_df[features_df["account_id"] == selected_id].iloc[0]

            if result:
                st.markdown("---")
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
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Tenure",     f"{int(cust_feats.get('tenure_days', 0))} days")
                k2.metric("Avg MRR",    f"${cust_feats.get('avg_mrr', 0):,.0f}")
                k3.metric("Tickets",    f"{int(cust_feats.get('ticket_count', 0))}")
                k4.metric("Usage",      f"{cust_feats.get('total_usage_minutes', 0):,.0f} mins")

                st.markdown("---")
                st.markdown("##### Prediction Result")
                r1, r2, r3 = st.columns(3)
                r1.metric("Churn Probability", f"{result['probability']:.1%}")
                r2.metric("Prediction",
                          "Will Churn" if result["prediction"] == 1 else "Will Stay")
                risk_level = result["risk"]
                r3.markdown(
                    f"**Risk Level**<br>"
                    f"<span class='risk-{risk_level}'>"
                    f"{risk_level.upper()}</span>",
                    unsafe_allow_html=True,
                )

                st.plotly_chart(
                    gauge_chart(result["probability"], result["risk"]),
                    use_container_width=True,
                )

                # Ground truth — for evaluation only
                if bool(cust_info["churn_flag"]):
                    st.info("📌 **Ground truth (dataset):** This customer actually churned — for evaluation only.")
                else:
                    st.info("📌 **Ground truth (dataset):** This customer did not churn — for evaluation only.")

                with st.expander("View all feature values"):
                    feat_data = cust_feats.drop(["account_id", "churn_flag"])
                    non_zero  = feat_data[feat_data != 0].sort_values(ascending=False)
                    st.dataframe(
                        non_zero.reset_index().rename(
                            columns={"index": "Feature", 0: "Value"}
                        ),
                        use_container_width=True,
                    )

    # ───────────────────────────────────────────────────────────────────
    # TAB 2 — SCENARIO EXPLORER
    # ───────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("##### Explore customer profiles by scenario")
        st.caption(
            "Pick a customer profile using the dropdowns below. "
            "The system finds the 3 real customers in the dataset "
            "whose usage and support patterns best match your description "
            "and predicts their churn risk using actual feature data — "
            "no synthetic reconstruction involved."
        )

        st.markdown("---")

        # ── Profile selectors ────────────────────────────────────────
        s1, s2, s3, s4 = st.columns(4)

        with s1:
            sc_plan = st.selectbox(
                "Plan Tier",
                sorted(accounts_df["plan_tier"].unique()),
                key="sc_plan",
            )

        with s2:
            sc_industry = st.selectbox(
                "Industry",
                sorted(accounts_df["industry"].unique()),
                key="sc_industry",
            )

        with s3:
            sc_usage = st.selectbox(
                "Usage Level",
                USAGE_LABELS,
                index=1,
                key="sc_usage",
                help="Total product usage in minutes over the observation period",
            )

        with s4:
            sc_support = st.selectbox(
                "Support Load",
                TICKET_LABELS,
                index=1,
                key="sc_support",
                help="Volume of support tickets raised",
            )

        st.markdown("")
        run_btn = st.button(
            "🔍 Find Matching Customers",
            type="primary",
            use_container_width=True,
            key="sc_run",
        )

        if run_btn:
            matches = find_matching_customers(
                features_df, accounts_df,
                sc_plan, sc_industry, sc_usage, sc_support,
                n=3,
            )

            if matches.empty:
                st.warning("No customers found. Try a different combination.")
                st.stop()

            # ── How many rows actually matched the strict filter ──────
            strict_count = len(
                features_df.merge(
                    accounts_df[accounts_df["plan_tier"] == sc_plan]
                              [accounts_df["industry"]  == sc_industry],
                    on="account_id", how="inner",
                )
            )

            st.markdown("---")
            if strict_count >= 3:
                st.markdown(
                    f"**Showing top 3 matches** — "
                    f"{sc_plan} plan · {sc_industry} industry · "
                    f"{sc_usage} · {sc_support}"
                )
            else:
                st.info(
                    f"Fewer than 3 exact matches for **{sc_plan} / {sc_industry}**. "
                    f"Showing closest profiles from similar plan tiers."
                )

            # ── Predict for each match ────────────────────────────────
            results = []
            for _, row in matches.iterrows():
                pred = predict_customer(model, features_df, row["account_id"])
                if pred:
                    results.append((row, pred))

            if not results:
                st.error("Could not generate predictions for matched customers.")
                st.stop()

            # ── Render 3 cards side by side ───────────────────────────
            cols = st.columns(len(results))
            for i, (row, pred) in enumerate(results):
                render_customer_card(cols[i], row, pred, i + 1)

            # ── Comparison table ──────────────────────────────────────
            st.markdown("---")
            st.markdown("#### Side-by-side comparison")

            comparison_rows = []
            for row, pred in results:
                comparison_rows.append({
                    "Customer":        row.get("account_name", "—"),
                    "Plan":            row.get("plan_tier", "—"),
                    "Industry":        row.get("industry", "—"),
                    "Churn Prob":      f"{pred['probability']:.1%}",
                    "Risk":            pred["risk"].upper(),
                    "Tenure (days)":   int(row.get("tenure_days", 0)),
                    "Avg MRR ($)":     f"{row.get('avg_mrr', 0):,.0f}",
                    "Usage (mins)":    f"{row.get('total_usage_minutes', 0):,.0f}",
                    "Tickets":         int(row.get("ticket_count", 0)),
                    "Satisfaction":    f"{row.get('avg_satisfaction', 0):.1f}",
                    "Escalation Rate": f"{row.get('escalation_rate', 0):.0%}",
                    "Actually Churned":("Yes" if bool(row.get("churn_flag", 0))
                                        else "No"),
                })

            comp_df = pd.DataFrame(comparison_rows).set_index("Customer")
            st.dataframe(comp_df, use_container_width=True)

            # ── Insight summary ───────────────────────────────────────
            st.markdown("---")
            st.markdown("#### What this tells you")

            high_count = sum(1 for _, p in results if p["risk"] == "high")
            avg_prob   = np.mean([p["probability"] for _, p in results])
            churned_count = sum(
                1 for r, _ in results if bool(r.get("churn_flag", 0))
            )

            ins1, ins2, ins3 = st.columns(3)
            ins1.metric("Avg Churn Probability", f"{avg_prob:.1%}")
            ins2.metric("High Risk Customers",   f"{high_count} of {len(results)}")
            ins3.metric("Actually Churned",       f"{churned_count} of {len(results)}")

            if avg_prob >= 0.6:
                st.error(
                    "⚠️ This customer profile is **high risk**. "
                    "Customers with this combination of plan, industry, "
                    "usage pattern, and support load have a high likelihood of churning. "
                    "Proactive outreach is recommended."
                )
            elif avg_prob >= 0.35:
                st.warning(
                    "🟡 This profile shows **moderate churn risk**. "
                    "Monitor these customers closely and consider targeted retention actions."
                )
            else:
                st.success(
                    "✅ This profile shows **low churn risk**. "
                    "Customers matching this description tend to stay engaged."
                )


# ═══════════════════════════════════════════════════════════════════════
# PAGE 4: MODEL
# ═══════════════════════════════════════════════════════════════════════

elif page == "Model":
    st.markdown("## Model Performance")
    st.caption("Experiment history and model analysis.")

    st.markdown("#### Improvement Journey")
    journey = pd.DataFrame({
        "Version":   ["v1 Baseline", "v2 Time-windowed", "v3 Optimized"],
        "AUC-ROC":   [0.5303, 0.5385, 0.6352],
        "Precision": [0.2000, 0.2727, 0.5000],
        "Recall":    [0.1364, 0.1364, 0.4091],
        "F1 Score":  [0.1622, 0.1818, 0.4500],
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
    m1.metric("AUC-ROC",   "0.6352")
    m2.metric("Precision", "0.5000")
    m3.metric("Recall",    "0.4091")
    m4.metric("F1 Score",  "0.4500")
    m5.metric("Threshold", "0.4758")

    st.markdown("---")
    st.markdown("#### Top Features by Importance")
    if model is not None:
        try:
            imp   = (model.feature_importances_
                     if hasattr(model, "feature_importances_")
                     else np.abs(model.coef_[0])
                     if hasattr(model, "coef_") else None)
            names = get_feature_names(model)
            if imp is not None and names is not None:
                fi = (pd.DataFrame({"Feature": names, "Importance": imp})
                        .sort_values("Importance", ascending=True)
                        .tail(15))
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
            "no_drift":         ("✅", "No significant drift. Model is stable.",          "success"),
            "moderate_drift":   ("⚠️", "Moderate drift detected. Monitor closely.",       "warning"),
            "significant_drift":("🔴", "Significant drift. Retraining recommended.",       "error"),
        }
        icon, msg, alert = cfg.get(verdict, ("ℹ️", "", "info"))
        getattr(st, alert)(
            f"{icon} **{verdict.replace('_', ' ').title()}** — {msg}"
        )

        d1, d2, d3 = st.columns(3)
        d1.metric("Features Tested", report["total_features_tested"])
        d2.metric("Features Drifted", report["features_with_drift"])
        d3.metric("Drift Ratio", f"{report['drift_ratio']:.1%}")

        st.markdown("---")

        if report["drifted_features"]:
            st.markdown("#### Drifted Features")
            drifted = [{
                "Feature":   f,
                "Test":      report["feature_details"][f].get("test", ""),
                "Statistic": round(report["feature_details"][f].get("statistic", 0), 4),
                "P-Value":   round(report["feature_details"][f].get("p_value", 0), 6),
            } for f in report["drifted_features"]]
            st.dataframe(
                pd.DataFrame(drifted).sort_values("P-Value"),
                use_container_width=True, hide_index=True,
            )

        with st.expander("View all feature details"):
            all_f = [{
                "Feature":   f,
                "Test":      d.get("test", ""),
                "Statistic": round(d.get("statistic", 0), 4),
                "P-Value":   round(d.get("p_value", 0), 6),
                "Drifted":   "Yes" if d.get("drift_detected") else "No",
            } for f, d in report["feature_details"].items()]
            st.dataframe(
                pd.DataFrame(all_f).sort_values("P-Value"),
                use_container_width=True, height=400, hide_index=True,
            )

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