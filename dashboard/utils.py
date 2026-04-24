"""
Utility helpers for the Fraud Dashboard.
All heavy I/O is cached with st.cache_data / st.cache_resource.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "creditcard_enriched.csv"
MODEL_PATH = ROOT / "models" / "fraud_model.pkl"


# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading transaction data …")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.info("Generating synthetic dataset – this takes ~30 seconds …")
        from data.generate_data import generate_dataset
        generate_dataset(output_path=str(DATA_PATH))
    df = pd.read_csv(DATA_PATH)
    # Ensure derived columns exist
    if "hour" not in df.columns:
        df["hour"] = (df["Time"] // 3600) % 24
    if "is_weekend" not in df.columns:
        df["is_weekend"] = 0
    return df


# ── Model ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading fraud detection model …")
def load_model() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# ── KPIs ──────────────────────────────────────────────────────────────────────

def get_kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    fraud = df["Class"].sum()
    fraud_rate = fraud / total * 100
    fraud_amount = df.loc[df["Class"] == 1, "Amount"].sum()
    normal_amount = df.loc[df["Class"] == 0, "Amount"].sum()
    avg_fraud_amount = df.loc[df["Class"] == 1, "Amount"].mean()
    return {
        "total_transactions": total,
        "fraud_count": int(fraud),
        "normal_count": int(total - fraud),
        "fraud_rate": round(fraud_rate, 2),
        "total_fraud_amount": round(fraud_amount, 2),
        "total_normal_amount": round(normal_amount, 2),
        "avg_fraud_amount": round(avg_fraud_amount, 2),
    }


# ── Single-transaction risk score ─────────────────────────────────────────────

def calculate_risk_score(
    artifact: dict,
    amount: float,
    hour: int,
    is_weekend: int,
    user_type: str,
    payment_type: str,
    country: str,
    small_tx_sequence: int,
    transaction_count: int,
    merchant_category: str,
    v_features: list[float] | None = None,
) -> tuple[float, dict]:
    """Return (probability_fraud, feature_dict) for a single transaction."""
    encoders = artifact["encoders"]
    model = artifact["model"]
    features = artifact["features"]
    categorical_features = artifact["categorical_features"]

    if v_features is None:
        v_features = [0.0] * 10

    row: dict[str, float] = {
        "Amount": amount,
        "hour": hour,
        "is_weekend": is_weekend,
        "small_tx_sequence": small_tx_sequence,
        "transaction_count": transaction_count,
        "avg_amount": amount,
    }
    for i, v in enumerate(v_features[:10], start=1):
        row[f"V{i}"] = v

    cat_values = {
        "user_type": user_type,
        "payment_type": payment_type,
        "country": country,
        "merchant_category": merchant_category,
    }
    for col in categorical_features:
        le = encoders[col]
        val = cat_values[col]
        if val in le.classes_:
            row[f"{col}_encoded"] = int(le.transform([val])[0])
        else:
            row[f"{col}_encoded"] = 0

    X = pd.DataFrame([row])[features]
    prob = float(model.predict_proba(X)[0, 1])

    # Feature contributions (mean decrease impurity proxy)
    importances = model.feature_importances_
    contributions = {
        feat: round(float(importances[i]) * prob * 100, 2)
        for i, feat in enumerate(features)
    }
    top_contributions = dict(
        sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:8]
    )
    return prob, top_contributions


def risk_level(prob: float) -> tuple[str, str]:
    """Return (label, color) for a given fraud probability."""
    if prob < 0.25:
        return "Low", "#2ecc71"
    elif prob < 0.50:
        return "Medium", "#f39c12"
    elif prob < 0.75:
        return "High", "#e67e22"
    else:
        return "Critical", "#e74c3c"
