"""
Train a Random Forest classifier for credit card fraud detection.
Saves the trained model and label encoders to models/fraud_model.pkl.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Ensure repo root is on path so data module is importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "creditcard_enriched.csv"
MODEL_PATH = ROOT / "models" / "fraud_model.pkl"

# Training configuration
TEST_SIZE = 0.20
RANDOM_STATE = 42
RF_N_ESTIMATORS = 150
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 5

CATEGORICAL_FEATURES = ["user_type", "payment_type", "country", "merchant_category"]
NUMERIC_FEATURES = [
    "Amount", "hour", "is_weekend", "small_tx_sequence",
    "transaction_count", "avg_amount",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
]
ALL_FEATURES = NUMERIC_FEATURES + [f"{c}_encoded" for c in CATEGORICAL_FEATURES]


def load_or_generate_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print("Dataset not found – generating synthetic data …")
        from data.generate_data import generate_dataset
        generate_dataset(output_path=str(DATA_PATH))
    return pd.read_csv(DATA_PATH)


def encode_categoricals(df: pd.DataFrame):
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def train(df: pd.DataFrame):
    df, encoders = encode_categoricals(df)

    X = df[ALL_FEATURES]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training on {len(X_train):,} samples, testing on {len(X_test):,} samples …")

    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\n── Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    print("── Confusion Matrix ──")
    print(confusion_matrix(y_test, y_pred))

    roc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    accuracy = clf.score(X_test, y_test)
    print(f"\nROC-AUC  : {roc:.4f}")
    print(f"Avg Prec : {ap:.4f}")
    print(f"Accuracy : {accuracy:.4f}")

    # ── Feature importances ───────────────────────────────────────────────────
    importances = pd.Series(clf.feature_importances_, index=ALL_FEATURES)
    print("\n── Top 10 Feature Importances ──")
    print(importances.sort_values(ascending=False).head(10).to_string())

    # ── Persist ───────────────────────────────────────────────────────────────
    artifact = {
        "model": clf,
        "encoders": encoders,
        "features": ALL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "roc_auc": round(roc, 4),
            "avg_precision": round(ap, 4),
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")
    return artifact


if __name__ == "__main__":
    df = load_or_generate_data()
    train(df)
