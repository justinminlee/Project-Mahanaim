"""
Synthetic Credit Card Fraud Dataset Generator
Simulates 50,000 transactions with realistic fraud patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N = 50_000
FRAUD_BASE_RATE = 0.07  # ~7% base fraud rate for demo visibility

PAYMENT_TYPES = ["Credit Card", "Apple Pay", "Google Pay", "Samsung Pay", "Debit Card"]
COUNTRIES = ["US", "UK", "Germany", "France", "Canada", "China", "Japan", "Brazil", "Russia", "Nigeria", "Romania"]
USER_TYPES = ["new", "existing"]
MERCHANT_CATEGORIES = ["retail", "food", "travel", "entertainment", "online", "gas", "ATM"]


def _fraud_probability(hour, country, user_type, small_tx_seq, payment_type, merchant_category):
    """Compute per-transaction fraud probability using domain-based multipliers."""
    p = FRAUD_BASE_RATE

    # Night hours (23-5): 3x more fraud
    if hour >= 23 or hour <= 5:
        p *= 3.0

    # High-risk countries: 4x more fraud
    if country in ("Nigeria", "Romania", "Russia"):
        p *= 4.0
    elif country in ("Brazil", "China"):
        p *= 2.0

    # New users: 3x more fraud
    if user_type == "new":
        p *= 3.0

    # Digital wallets are slightly safer (tokenization)
    if payment_type in ("Apple Pay", "Google Pay"):
        p *= 0.6
    elif payment_type == "Samsung Pay":
        p *= 0.75

    # Small transaction sequences are a strong pre-fraud signal
    if small_tx_seq > 3:
        p *= 5.0
    elif small_tx_seq > 1:
        p *= 2.0

    # High-risk merchant categories
    if merchant_category in ("online", "ATM"):
        p *= 2.5
    elif merchant_category == "travel":
        p *= 1.5

    return min(p, 0.95)


def generate_dataset(n: int = N, output_path: str = "data/creditcard_enriched.csv") -> pd.DataFrame:
    print(f"Generating {n:,} synthetic transactions...")

    # ── Time & derived temporal features ─────────────────────────────────────
    time_seconds = np.random.uniform(0, 172_800, n).astype(int)  # 0 – 48 h
    hour = (time_seconds // 3600) % 24
    day_of_week = np.random.randint(0, 7, n)  # 0=Monday … 6=Sunday
    is_weekend = (day_of_week >= 5).astype(int)

    # ── Categorical features ──────────────────────────────────────────────────
    payment_type = np.random.choice(
        PAYMENT_TYPES,
        n,
        p=[0.35, 0.20, 0.18, 0.12, 0.15],
    )
    country = np.random.choice(
        COUNTRIES,
        n,
        p=[0.30, 0.12, 0.08, 0.07, 0.08, 0.07, 0.06, 0.06, 0.05, 0.06, 0.05],
    )
    user_type = np.random.choice(USER_TYPES, n, p=[0.25, 0.75])
    merchant_category = np.random.choice(
        MERCHANT_CATEGORIES,
        n,
        p=[0.25, 0.20, 0.10, 0.10, 0.20, 0.10, 0.05],
    )

    # ── Per-user statistics (simulate user history) ───────────────────────────
    n_users = n // 10
    user_ids = np.random.randint(0, n_users, n)
    transaction_count = np.zeros(n_users, dtype=int)
    for uid in user_ids:
        transaction_count[uid] += 1
    tx_count = transaction_count[user_ids]

    avg_amount_by_user = np.random.lognormal(mean=3.5, sigma=1.2, size=n_users)
    avg_amount = avg_amount_by_user[user_ids]

    # ── Small transaction sequence (pre-fraud signal) ─────────────────────────
    small_tx_sequence = np.zeros(n, dtype=int)
    for i in range(n):
        if np.random.random() < 0.15:
            small_tx_sequence[i] = np.random.randint(1, 8)

    # ── Assign fraud labels using per-row probabilities ───────────────────────
    fraud_probs = np.array([
        _fraud_probability(
            hour[i], country[i], user_type[i],
            small_tx_sequence[i], payment_type[i], merchant_category[i],
        )
        for i in range(n)
    ])
    labels = (np.random.random(n) < fraud_probs).astype(int)

    # ── Transaction amounts ───────────────────────────────────────────────────
    # Fraud transactions tend to cluster at extreme amounts
    fraud_mask = labels == 1
    small_probe = np.random.uniform(0.5, 5, n)
    large_charge = np.random.uniform(200, 5000, n)
    pick_large = np.random.random(n) < 0.65
    fraud_amounts = np.where(pick_large, large_charge, small_probe)
    normal_amounts = np.abs(np.random.lognormal(mean=3.5, sigma=1.5, size=n))
    amount = np.where(fraud_mask, fraud_amounts, normal_amounts)
    amount = np.round(amount, 2)

    # ── PCA-like features V1-V28 ──────────────────────────────────────────────
    # For fraud rows, shift the distribution to simulate anomalous PCA values
    pca_normal = np.random.randn(n, 28)
    shift = np.where(labels == 1, -3.5, 0)[:, None] * (np.arange(28) % 7 == 0)
    pca_features = np.round(pca_normal + shift, 6)
    pca_cols = {f"V{i+1}": pca_features[:, i] for i in range(28)}

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "Time": time_seconds,
            **pca_cols,
            "Amount": amount,
            "Class": labels,
            "payment_type": payment_type,
            "country": country,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "user_type": user_type,
            "transaction_count": tx_count,
            "avg_amount": np.round(avg_amount, 2),
            "small_tx_sequence": small_tx_sequence,
            "merchant_category": merchant_category,
        }
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    fraud_count = labels.sum()
    print(f"✅ Saved {len(df):,} rows → {out}")
    print(f"   Fraud: {fraud_count:,} ({fraud_count/len(df)*100:.2f}%)")
    print(f"   Normal: {(len(df)-fraud_count):,} ({(len(df)-fraud_count)/len(df)*100:.2f}%)")
    return df


if __name__ == "__main__":
    generate_dataset()
