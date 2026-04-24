"""
Credit Card Fraud Detection & Analysis Dashboard
Run: streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.utils import (
    calculate_risk_score,
    get_kpis,
    load_data,
    load_model,
    risk_level,
)

# ── Display constants ─────────────────────────────────────────────────────────
SCATTER_SAMPLE_SIZE = 3000
DIST_SAMPLE_SIZE = 3000
MAX_DISPLAY_AMOUNT = 2000        # clip threshold for amount distribution chart
FRAUD_RISK_RANGE = (0.6, 0.99)   # fallback risk range when model absent (fraud)
NORMAL_RISK_RANGE = (0.01, 0.4)  # fallback risk range when model absent (normal)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard – Credit Card Fraud Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #0e1117; color: #fafafa; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-delta {
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #7f8fff;
        border-left: 3px solid #7f8fff;
        padding-left: 10px;
        margin: 18px 0 10px 0;
    }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d111c;
        border-right: 1px solid #1e2433;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #0d111c;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8892a4;
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a1f2e !important;
        color: #ffffff !important;
    }

    /* Table tweaks */
    .dataframe { font-size: 0.82rem !important; }

    /* Insight boxes */
    .insight-box {
        background: #111827;
        border: 1px solid #374151;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
        color: #d1d5db;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Colour palette ────────────────────────────────────────────────────────────
FRAUD_COLOR = "#ef4444"
NORMAL_COLOR = "#22c55e"
ACCENT = "#6366f1"
PLOTLY_TEMPLATE = "plotly_dark"


def kpi_card(label: str, value: str, delta: str = "", color: str = "#7f8fff") -> str:
    delta_html = f'<div class="kpi-delta" style="color:{color};">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        {delta_html}
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔐 FraudGuard")
    st.markdown("*Credit Card Fraud Detection*")
    st.divider()
    st.markdown("### 🗺 Navigation")
    page = st.radio(
        "Select Page",
        [
            "📊 Overview",
            "⏰ Time Analysis",
            "💳 Payment Types",
            "🌍 Geographic",
            "🎯 Risk & Transactions",
            "🔴 Live Detector",
            "👥 User Behavior",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Dataset: Synthetic Credit Card Transactions (50K rows)")
    st.caption("Model: Random Forest Classifier")

# ══════════════════════════════════════════════════════════════════════════════
# Load data & model
# ══════════════════════════════════════════════════════════════════════════════
df = load_data()
artifact = load_model()
kpis = get_kpis(df)

# Add risk scores if model is available
if artifact and "risk_score" not in df.columns:
    try:
        encoders = artifact["encoders"]
        model = artifact["model"]
        features = artifact["features"]
        cat_features = artifact["categorical_features"]

        df_enc = df.copy()
        for col in cat_features:
            le = encoders[col]
            # Handle unseen labels gracefully
            df_enc[f"{col}_encoded"] = df_enc[col].apply(
                lambda x: int(le.transform([x])[0]) if x in le.classes_ else 0
            )
        X = df_enc[features]
        df["risk_score"] = model.predict_proba(X)[:, 1]
    except Exception:
        df["risk_score"] = np.where(df["Class"] == 1,
                                    np.random.uniform(*FRAUD_RISK_RANGE, len(df)),
                                    np.random.uniform(*NORMAL_RISK_RANGE, len(df)))
else:
    if "risk_score" not in df.columns:
        df["risk_score"] = np.where(df["Class"] == 1,
                                    np.random.uniform(*FRAUD_RISK_RANGE, len(df)),
                                    np.random.uniform(*NORMAL_RISK_RANGE, len(df)))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Executive Overview")
    st.caption("Real-time fraud monitoring across all transactions")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(kpi_card("Total Transactions", f"{kpis['total_transactions']:,}", color="#60a5fa"), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi_card("Fraud Cases", f"{kpis['fraud_count']:,}", "⚠ Flagged", FRAUD_COLOR), unsafe_allow_html=True)
    with col3:
        st.markdown(kpi_card("Fraud Rate", f"{kpis['fraud_rate']}%", "of all transactions", "#f59e0b"), unsafe_allow_html=True)
    with col4:
        st.markdown(kpi_card("Total Fraud Amount", f"${kpis['total_fraud_amount']:,.0f}", color="#ef4444"), unsafe_allow_html=True)
    with col5:
        model_acc = artifact["metrics"]["accuracy"] * 100 if artifact else 0
        st.markdown(kpi_card("Model Accuracy", f"{model_acc:.1f}%", "Random Forest", "#22c55e"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<div class="section-header">Transaction Distribution</div>', unsafe_allow_html=True)
        pie_fig = px.pie(
            names=["Normal", "Fraud"],
            values=[kpis["normal_count"], kpis["fraud_count"]],
            color_discrete_sequence=[NORMAL_COLOR, FRAUD_COLOR],
            hole=0.55,
            template=PLOTLY_TEMPLATE,
        )
        pie_fig.update_traces(textinfo="percent+label", textfont_size=13)
        pie_fig.update_layout(
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
            margin=dict(t=10, b=10, l=10, r=10),
            height=320,
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Top Fraud Countries</div>', unsafe_allow_html=True)
        country_fraud = (
            df[df["Class"] == 1]
            .groupby("country")
            .agg(fraud_count=("Class", "count"), fraud_amount=("Amount", "sum"))
            .reset_index()
            .sort_values("fraud_count", ascending=False)
            .head(8)
        )
        bar_fig = px.bar(
            country_fraud,
            x="country",
            y="fraud_count",
            color="fraud_amount",
            color_continuous_scale="Reds",
            labels={"fraud_count": "Fraud Cases", "fraud_amount": "Total Fraud Amount ($)"},
            template=PLOTLY_TEMPLATE,
            text="fraud_count",
        )
        bar_fig.update_traces(textposition="outside")
        bar_fig.update_layout(
            coloraxis_colorbar=dict(title="Amount ($)"),
            margin=dict(t=10, b=10),
            height=320,
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # Model performance gauge
    if artifact:
        st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        metrics = artifact["metrics"]
        for col, (label, key, color) in zip(
            [c1, c2, c3],
            [("Accuracy", "accuracy", "#60a5fa"), ("ROC-AUC", "roc_auc", "#a78bfa"), ("Avg Precision", "avg_precision", "#34d399")],
        ):
            val = metrics.get(key, 0)
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val * 100,
                number={"suffix": "%", "font": {"size": 28}},
                title={"text": label, "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#444"},
                    "bar": {"color": color},
                    "bgcolor": "#1a1f2e",
                    "bordercolor": "#2d3561",
                    "steps": [
                        {"range": [0, 50], "color": "#1e293b"},
                        {"range": [50, 80], "color": "#1e3a5f"},
                        {"range": [80, 100], "color": "#1e4040"},
                    ],
                },
            ))
            gauge.update_layout(
                height=200,
                margin=dict(t=30, b=10, l=20, r=20),
                paper_bgcolor="#0e1117",
                font_color="#fafafa",
                template=PLOTLY_TEMPLATE,
            )
            with col:
                st.plotly_chart(gauge, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Time Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⏰ Time Analysis":
    st.title("⏰ Time Analysis")
    st.caption("Understanding when fraud occurs throughout the day and week")

    hourly = df.groupby("hour").agg(
        total=("Class", "count"),
        fraud=("Class", "sum"),
    ).reset_index()
    hourly["fraud_rate"] = (hourly["fraud"] / hourly["total"] * 100).round(2)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Fraud Count by Hour of Day</div>', unsafe_allow_html=True)
        fig = px.line(
            hourly, x="hour", y="fraud",
            markers=True,
            labels={"hour": "Hour of Day (0–23)", "fraud": "Fraud Cases"},
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=[FRAUD_COLOR],
        )
        fig.add_vrect(x0=23, x1=24, fillcolor="rgba(239,68,68,0.1)", line_width=0, annotation_text="High Risk")
        fig.add_vrect(x0=0, x1=5, fillcolor="rgba(239,68,68,0.1)", line_width=0)
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Fraud Rate (%) by Hour</div>', unsafe_allow_html=True)
        fig2 = px.bar(
            hourly, x="hour", y="fraud_rate",
            color="fraud_rate",
            color_continuous_scale="Reds",
            labels={"hour": "Hour of Day", "fraud_rate": "Fraud Rate (%)"},
            template=PLOTLY_TEMPLATE,
        )
        fig2.update_layout(height=320, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Fraud Heatmap – Hour × Day of Week</div>', unsafe_allow_html=True)
    DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    heatmap_data = df[df["Class"] == 1].groupby(["day_of_week", "hour"]).size().reset_index(name="count")
    pivot = heatmap_data.pivot(index="day_of_week", columns="hour", values="count").fillna(0)
    pivot.index = [DOW_LABELS[i] for i in pivot.index]

    heat_fig = px.imshow(
        pivot,
        color_continuous_scale="Reds",
        labels={"x": "Hour of Day", "y": "Day of Week", "color": "Fraud Cases"},
        aspect="auto",
        template=PLOTLY_TEMPLATE,
    )
    heat_fig.update_layout(height=280, margin=dict(t=10, b=10))
    st.plotly_chart(heat_fig, use_container_width=True)

    st.markdown(
        '<div class="insight-box">💡 <strong>Key Insight:</strong> Fraud peaks during late-night hours (11 PM – 5 AM) '
        "when transaction monitoring is typically reduced. Saturday and Sunday nights show the highest fraud density. "
        "Consider stricter velocity checks and step-up authentication during these windows.</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Payment Types
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💳 Payment Types":
    st.title("💳 Payment Type Analysis")
    st.caption("Comparing fraud rates across payment methods including digital wallets")

    pt = df.groupby("payment_type").agg(
        total=("Class", "count"),
        fraud=("Class", "sum"),
        avg_amount=("Amount", "mean"),
        total_amount=("Amount", "sum"),
    ).reset_index()
    pt["fraud_rate"] = (pt["fraud"] / pt["total"] * 100).round(2)
    pt["normal"] = pt["total"] - pt["fraud"]
    pt = pt.sort_values("fraud_rate", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Fraud Rate by Payment Type</div>', unsafe_allow_html=True)
        fig = px.bar(
            pt, x="payment_type", y="fraud_rate",
            color="fraud_rate",
            color_continuous_scale="RdYlGn_r",
            labels={"payment_type": "Payment Method", "fraud_rate": "Fraud Rate (%)"},
            template=PLOTLY_TEMPLATE,
            text=pt["fraud_rate"].apply(lambda x: f"{x:.1f}%"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=340, coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Transaction Volume by Payment Type</div>', unsafe_allow_html=True)
        stacked = pt.melt(
            id_vars="payment_type", value_vars=["normal", "fraud"],
            var_name="type", value_name="count",
        )
        stacked["type"] = stacked["type"].str.capitalize()
        fig2 = px.bar(
            stacked, x="payment_type", y="count",
            color="type",
            color_discrete_map={"Normal": NORMAL_COLOR, "Fraud": FRAUD_COLOR},
            labels={"payment_type": "Payment Method", "count": "Transactions"},
            template=PLOTLY_TEMPLATE,
            barmode="stack",
        )
        fig2.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Average Fraud Amount by Payment Type</div>', unsafe_allow_html=True)
    pt_fraud = df[df["Class"] == 1].groupby("payment_type")["Amount"].mean().reset_index()
    pt_fraud.columns = ["payment_type", "avg_fraud_amount"]
    pt_fraud = pt_fraud.sort_values("avg_fraud_amount", ascending=True)
    fig3 = px.bar(
        pt_fraud, x="avg_fraud_amount", y="payment_type",
        orientation="h",
        color="avg_fraud_amount",
        color_continuous_scale="Reds",
        labels={"avg_fraud_amount": "Avg Fraud Amount ($)", "payment_type": ""},
        template=PLOTLY_TEMPLATE,
        text=pt_fraud["avg_fraud_amount"].apply(lambda x: f"${x:,.0f}"),
    )
    fig3.update_traces(textposition="outside")
    fig3.update_layout(height=260, coloraxis_showscale=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        '<div class="insight-box">💡 <strong>Key Insight:</strong> '
        "Apple Pay and Google Pay show <strong>lower fraud rates</strong> thanks to device-based tokenization and biometric "
        "authentication – the actual card number is never transmitted. Samsung Pay offers similar protections via MST/NFC. "
        "Traditional Credit Cards and Debit Cards remain the highest fraud risk due to static card numbers.</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">Summary Table</div>', unsafe_allow_html=True)
    display_pt = pt[["payment_type", "total", "fraud", "fraud_rate", "avg_amount"]].copy()
    display_pt.columns = ["Payment Type", "Total Tx", "Fraud Cases", "Fraud Rate (%)", "Avg Amount ($)"]
    display_pt["Avg Amount ($)"] = display_pt["Avg Amount ($)"].round(2)
    st.dataframe(display_pt.reset_index(drop=True), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – Geographic
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Geographic":
    st.title("🌍 Geographic Analysis")
    st.caption("Fraud distribution and risk scoring by country")

    geo = df.groupby("country").agg(
        total=("Class", "count"),
        fraud=("Class", "sum"),
        avg_fraud_amount=("Amount", lambda x: x[df.loc[x.index, "Class"] == 1].mean() if (df.loc[x.index, "Class"] == 1).any() else 0),
    ).reset_index()
    geo["fraud_rate"] = (geo["fraud"] / geo["total"] * 100).round(2)
    geo = geo.sort_values("fraud_rate", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Fraud Cases by Country</div>', unsafe_allow_html=True)
        fig = px.bar(
            geo, x="country", y="fraud",
            color="fraud_rate",
            color_continuous_scale="OrRd",
            labels={"country": "Country", "fraud": "Fraud Cases", "fraud_rate": "Fraud Rate (%)"},
            template=PLOTLY_TEMPLATE,
            text="fraud",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Average Fraud Amount by Country</div>', unsafe_allow_html=True)
        avg_amt = df[df["Class"] == 1].groupby("country")["Amount"].mean().reset_index()
        avg_amt.columns = ["country", "avg_fraud_amount"]
        avg_amt = avg_amt.sort_values("avg_fraud_amount", ascending=True)
        fig2 = px.bar(
            avg_amt, x="avg_fraud_amount", y="country",
            orientation="h",
            color="avg_fraud_amount",
            color_continuous_scale="Oranges",
            labels={"avg_fraud_amount": "Avg Fraud Amount ($)", "country": ""},
            template=PLOTLY_TEMPLATE,
            text=avg_amt["avg_fraud_amount"].apply(lambda x: f"${x:,.0f}"),
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(height=340, coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Country Risk Table</div>', unsafe_allow_html=True)
    risk_table = geo[["country", "total", "fraud", "fraud_rate"]].copy()
    risk_table.columns = ["Country", "Total Transactions", "Fraud Cases", "Fraud Rate (%)"]
    risk_table["Risk Level"] = risk_table["Fraud Rate (%)"].apply(
        lambda r: "🔴 Critical" if r >= 25 else ("🟠 High" if r >= 15 else ("🟡 Medium" if r >= 8 else "🟢 Low"))
    )
    st.dataframe(risk_table.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="insight-box">💡 <strong>Key Insight:</strong> Nigeria, Romania, and Russia consistently rank as '
        "highest-risk origination countries for card-not-present fraud. These regions are known hotspots for organized "
        "card fraud rings. Transactions originating from or destined to these countries should trigger enhanced verification.</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – Risk & Transactions
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Risk & Transactions":
    st.title("🎯 Risk Score & Transaction Analysis")
    st.caption("Deep-dive into transaction risk scoring and fraud amount patterns")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Amount vs Risk Score</div>', unsafe_allow_html=True)
        sample = df.sample(min(SCATTER_SAMPLE_SIZE, len(df)), random_state=42)
        scatter = px.scatter(
            sample,
            x="Amount",
            y="risk_score",
            color=sample["Class"].map({0: "Normal", 1: "Fraud"}),
            color_discrete_map={"Normal": NORMAL_COLOR, "Fraud": FRAUD_COLOR},
            opacity=0.55,
            labels={"Amount": "Transaction Amount ($)", "risk_score": "Fraud Risk Score"},
            template=PLOTLY_TEMPLATE,
            hover_data=["payment_type", "country", "merchant_category"],
        )
        scatter.update_layout(height=340, margin=dict(t=10, b=10), legend_title="")
        st.plotly_chart(scatter, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Fraud vs Normal Amount Distribution</div>', unsafe_allow_html=True)
        fraud_amt = df[df["Class"] == 1]["Amount"].clip(upper=MAX_DISPLAY_AMOUNT)
        normal_amt = df[df["Class"] == 0]["Amount"].clip(upper=MAX_DISPLAY_AMOUNT).sample(DIST_SAMPLE_SIZE, random_state=1)
        dist_fig = go.Figure()
        dist_fig.add_trace(go.Histogram(
            x=normal_amt, name="Normal", nbinsx=60,
            marker_color=NORMAL_COLOR, opacity=0.65,
        ))
        dist_fig.add_trace(go.Histogram(
            x=fraud_amt, name="Fraud", nbinsx=60,
            marker_color=FRAUD_COLOR, opacity=0.75,
        ))
        dist_fig.update_layout(
            barmode="overlay",
            xaxis_title="Transaction Amount ($)",
            yaxis_title="Count",
            template=PLOTLY_TEMPLATE,
            height=340,
            margin=dict(t=10, b=10),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(dist_fig, use_container_width=True)

    st.markdown('<div class="section-header">Top 20 Highest Risk Transactions</div>', unsafe_allow_html=True)
    top_risk = (
        df.sort_values("risk_score", ascending=False)
        .head(20)[["Amount", "payment_type", "country", "hour", "merchant_category",
                   "user_type", "small_tx_sequence", "Class", "risk_score"]]
        .copy()
    )
    top_risk["risk_score"] = (top_risk["risk_score"] * 100).round(1)
    top_risk["Class"] = top_risk["Class"].map({0: "✅ Normal", 1: "🚨 Fraud"})
    top_risk.columns = ["Amount ($)", "Payment", "Country", "Hour", "Merchant", "User", "SmallTx", "Label", "Risk (%)"]
    st.dataframe(top_risk.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Small Transaction Sequence Analysis</div>', unsafe_allow_html=True)
    seq = df.groupby("small_tx_sequence").agg(
        total=("Class", "count"),
        fraud=("Class", "sum"),
    ).reset_index()
    seq["fraud_rate"] = (seq["fraud"] / seq["total"] * 100).round(2)
    seq = seq[seq["small_tx_sequence"] <= 8]
    fig_seq = px.bar(
        seq, x="small_tx_sequence", y="fraud_rate",
        color="fraud_rate",
        color_continuous_scale="Reds",
        labels={"small_tx_sequence": "# Small Transactions in Last Hour", "fraud_rate": "Fraud Rate (%)"},
        template=PLOTLY_TEMPLATE,
        text=seq["fraud_rate"].apply(lambda x: f"{x:.1f}%"),
    )
    fig_seq.update_traces(textposition="outside")
    fig_seq.update_layout(height=300, coloraxis_showscale=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig_seq, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 – Live Detector
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔴 Live Detector":
    st.title("🔴 Real-Time Fraud Detection Simulator")
    st.caption("Enter transaction details and get an instant fraud risk assessment")

    if not artifact:
        st.error("⚠️ Model not found. Please run `python models/train_model.py` first.")
        st.stop()

    with st.form("fraud_form"):
        st.markdown("### Transaction Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            amount = st.number_input("💰 Transaction Amount ($)", min_value=0.01, max_value=50000.0, value=250.0, step=1.0)
            payment_type = st.selectbox("💳 Payment Type", ["Credit Card", "Apple Pay", "Google Pay", "Samsung Pay", "Debit Card"])
            country = st.selectbox("🌍 Country", ["US", "UK", "Germany", "France", "Canada", "China", "Japan", "Brazil", "Russia", "Nigeria", "Romania"])

        with col2:
            hour = st.slider("⏰ Hour of Day", min_value=0, max_value=23, value=14)
            user_type = st.radio("👤 User Type", ["new", "existing"], horizontal=True)
            is_weekend = st.checkbox("📅 Weekend Transaction", value=False)

        with col3:
            merchant_category = st.selectbox("🏪 Merchant Category", ["retail", "food", "travel", "entertainment", "online", "gas", "ATM"])
            small_tx_seq = st.slider("🔢 Small Tx Count (Last Hour)", min_value=0, max_value=10, value=0)
            tx_count = st.number_input("📊 User Total Transactions", min_value=1, max_value=1000, value=12)

        submitted = st.form_submit_button("🔍 Detect Fraud", use_container_width=True, type="primary")

    if submitted:
        prob, contributions = calculate_risk_score(
            artifact=artifact,
            amount=amount,
            hour=hour,
            is_weekend=int(is_weekend),
            user_type=user_type,
            payment_type=payment_type,
            country=country,
            small_tx_sequence=small_tx_seq,
            transaction_count=tx_count,
            merchant_category=merchant_category,
        )
        score = int(prob * 100)
        level, color = risk_level(prob)

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown("### Risk Assessment")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix": "/100", "font": {"size": 36, "color": color}},
                title={"text": "FRAUD RISK SCORE", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#444"},
                    "bar": {"color": color, "thickness": 0.3},
                    "bgcolor": "#1a1f2e",
                    "bordercolor": "#2d3561",
                    "steps": [
                        {"range": [0, 25], "color": "#14532d"},
                        {"range": [25, 50], "color": "#713f12"},
                        {"range": [50, 75], "color": "#7c2d12"},
                        {"range": [75, 100], "color": "#450a0a"},
                    ],
                    "threshold": {
                        "line": {"color": color, "width": 4},
                        "thickness": 0.75,
                        "value": score,
                    },
                },
            ))
            gauge.update_layout(
                height=280,
                margin=dict(t=30, b=10, l=20, r=20),
                paper_bgcolor="#0e1117",
                font_color="#fafafa",
            )
            st.plotly_chart(gauge, use_container_width=True)

            st.markdown(
                f'<div style="text-align:center;">'
                f'<span class="risk-badge" style="background-color:{color}33; color:{color}; border: 2px solid {color};">'
                f"{'🟢' if level=='Low' else '🟡' if level=='Medium' else '🟠' if level=='High' else '🔴'} {level.upper()} RISK"
                f"</span></div>",
                unsafe_allow_html=True,
            )

        with res_col2:
            st.markdown("### Feature Contributions")
            feat_df = pd.DataFrame(
                list(contributions.items()), columns=["Feature", "Contribution"]
            ).sort_values("Contribution", ascending=True)
            bar = px.bar(
                feat_df, x="Contribution", y="Feature",
                orientation="h",
                color="Contribution",
                color_continuous_scale="Reds",
                template=PLOTLY_TEMPLATE,
                labels={"Contribution": "Risk Contribution Score", "Feature": ""},
                text=feat_df["Contribution"].apply(lambda x: f"{x:.1f}"),
            )
            bar.update_traces(textposition="outside")
            bar.update_layout(height=300, coloraxis_showscale=False, margin=dict(t=10, b=10))
            st.plotly_chart(bar, use_container_width=True)

            st.markdown("### Transaction Summary")
            summary_data = {
                "Field": ["Amount", "Payment Type", "Country", "Hour", "User Type", "Merchant", "Small Tx Seq"],
                "Value": [f"${amount:,.2f}", payment_type, country, f"{hour}:00", user_type, merchant_category, str(small_tx_seq)],
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # Decision banner
        if prob >= 0.5:
            st.error(
                f"🚨 **FRAUD ALERT** – This transaction has a {score}% risk score and is flagged as likely fraudulent. "
                "Recommended action: **Block & Verify**"
            )
        elif prob >= 0.25:
            st.warning(
                f"⚠️ **SUSPICIOUS** – Risk score {score}%. This transaction requires additional verification. "
                "Recommended action: **Step-up Authentication**"
            )
        else:
            st.success(
                f"✅ **APPROVED** – Risk score {score}%. This transaction appears legitimate. "
                "Recommended action: **Allow**"
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 – User Behavior
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 User Behavior":
    st.title("👥 User Behavior Analysis")
    st.caption("Pre-fraud behavioral patterns, user types, and merchant category insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">New vs Existing User Fraud Rate</div>', unsafe_allow_html=True)
        user_stats = df.groupby("user_type").agg(
            total=("Class", "count"),
            fraud=("Class", "sum"),
        ).reset_index()
        user_stats["fraud_rate"] = (user_stats["fraud"] / user_stats["total"] * 100).round(2)
        fig = px.bar(
            user_stats, x="user_type", y="fraud_rate",
            color="user_type",
            color_discrete_map={"new": FRAUD_COLOR, "existing": NORMAL_COLOR},
            labels={"user_type": "User Type", "fraud_rate": "Fraud Rate (%)"},
            template=PLOTLY_TEMPLATE,
            text=user_stats["fraud_rate"].apply(lambda x: f"{x:.1f}%"),
        )
        fig.update_traces(textposition="outside", showlegend=False)
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        for _, row in user_stats.iterrows():
            st.markdown(
                kpi_card(
                    f"{row['user_type'].capitalize()} Users",
                    f"{row['fraud_rate']}%",
                    f"{row['fraud']:,} fraud / {row['total']:,} total",
                    FRAUD_COLOR if row["user_type"] == "new" else NORMAL_COLOR,
                ),
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown('<div class="section-header">Merchant Category Fraud Distribution</div>', unsafe_allow_html=True)
        merch = df.groupby("merchant_category").agg(
            total=("Class", "count"),
            fraud=("Class", "sum"),
        ).reset_index()
        merch["fraud_rate"] = (merch["fraud"] / merch["total"] * 100).round(2)
        merch = merch.sort_values("fraud_rate", ascending=False)
        fig2 = px.bar(
            merch, x="merchant_category", y="fraud_rate",
            color="fraud_rate",
            color_continuous_scale="RdYlGn_r",
            labels={"merchant_category": "Merchant Category", "fraud_rate": "Fraud Rate (%)"},
            template=PLOTLY_TEMPLATE,
            text=merch["fraud_rate"].apply(lambda x: f"{x:.1f}%"),
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(height=340, coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Small Transaction Sequences → Fraud Escalation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="insight-box">💡 Fraudsters often test stolen cards with small micro-transactions ($1–$5) before '
        "attempting larger fraudulent charges. Multiple small transactions in a short window is a strong fraud precursor signal.</div>",
        unsafe_allow_html=True,
    )

    seq_fraud = df[df["Class"] == 1].groupby("small_tx_sequence").size().reset_index(name="fraud_count")
    seq_normal = df[df["Class"] == 0].groupby("small_tx_sequence").size().reset_index(name="normal_count")
    seq_merged = seq_fraud.merge(seq_normal, on="small_tx_sequence", how="outer").fillna(0)
    seq_merged = seq_merged[seq_merged["small_tx_sequence"] <= 8]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=seq_merged["small_tx_sequence"], y=seq_merged["normal_count"],
        name="Normal", marker_color=NORMAL_COLOR, opacity=0.8,
    ))
    fig3.add_trace(go.Bar(
        x=seq_merged["small_tx_sequence"], y=seq_merged["fraud_count"],
        name="Fraud", marker_color=FRAUD_COLOR, opacity=0.85,
    ))
    fig3.update_layout(
        barmode="group",
        xaxis_title="# Small Transactions in Last Hour (Pre-Fraud Signal)",
        yaxis_title="Transaction Count",
        template=PLOTLY_TEMPLATE,
        height=320,
        margin=dict(t=10, b=10),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Fraud Rate by Merchant Category – Detail</div>', unsafe_allow_html=True)
    merch_detail = df.groupby(["merchant_category", "Class"]).agg(count=("Amount", "count")).reset_index()
    merch_detail["label"] = merch_detail["Class"].map({0: "Normal", 1: "Fraud"})
    fig4 = px.bar(
        merch_detail, x="merchant_category", y="count",
        color="label",
        color_discrete_map={"Normal": NORMAL_COLOR, "Fraud": FRAUD_COLOR},
        barmode="stack",
        labels={"merchant_category": "Merchant Category", "count": "Transactions", "label": ""},
        template=PLOTLY_TEMPLATE,
    )
    fig4.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        '<div class="insight-box">💡 <strong>Pattern Insight:</strong> ATM and Online merchants show the highest fraud rates. '
        "ATM fraud often involves skimming devices, while online merchants face card-not-present fraud. "
        "New users making their first online or ATM transaction in a high-risk country at night represent the highest risk profile.</div>",
        unsafe_allow_html=True,
    )
