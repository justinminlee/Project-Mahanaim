# 🔐 FraudGuard – Credit Card Fraud Detection & Analysis Dashboard

A professional **Fraud Monitoring / Risk Dashboard** built with Python and Streamlit, inspired by real-world fraud detection systems used by financial institutions.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.13+-purple?logo=plotly)

---

## 📋 Project Overview

This dashboard analyzes credit card fraud patterns using a synthetic dataset modeled after the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It includes a trained **Random Forest** classifier, 7 interactive analysis pages, and a real-time fraud detection simulator.

### Key Features

| Feature | Description |
|---|---|
| 📊 Executive Overview | KPIs, fraud distribution, top countries, model accuracy gauges |
| ⏰ Time Analysis | Peak fraud hours, day-of-week heatmap, temporal patterns |
| 💳 Payment Types | Digital wallet vs card fraud rates, tokenization impact |
| 🌍 Geographic Analysis | Country-level risk scoring and fraud amount analysis |
| 🎯 Risk Scoring | Scatter plots, amount distributions, top-risk transactions |
| 🔴 Live Detector | Real-time fraud probability with feature contribution breakdown |
| 👥 User Behavior | New vs existing user patterns, pre-fraud micro-transaction signals |

---

## 🚀 Quick Start

### Option 1 – One-click Setup
```bash
python setup.py
streamlit run dashboard/app.py
```

### Option 2 – Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic dataset (50,000 transactions)
python data/generate_data.py

# 3. Train the fraud detection model
python models/train_model.py

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

The dashboard will be available at **http://localhost:8501**

---

## 📁 Project Structure

```
Project-Mahanaim/
├── requirements.txt            # Python dependencies
├── setup.py                    # One-click setup script
├── README.md
├── data/
│   ├── generate_data.py        # Synthetic dataset generator
│   └── creditcard_enriched.csv # Generated dataset (created on first run)
├── models/
│   ├── train_model.py          # Model training script
│   └── fraud_model.pkl         # Trained model (created on first run)
└── dashboard/
    ├── app.py                  # Main Streamlit application
    └── utils.py                # Data loading & ML utilities
```

---

## 🧠 Machine Learning Model

- **Algorithm**: Random Forest Classifier (150 trees)
- **Class balancing**: `class_weight="balanced"` for imbalanced fraud data
- **Features**: Amount, hour, weekend flag, user type, payment type, country, merchant category, small transaction sequence, V1–V10 PCA features
- **Metrics**: Accuracy, ROC-AUC, Average Precision

---

## 📊 Dataset

The synthetic dataset simulates **50,000 credit card transactions** with realistic fraud patterns:

- **Fraud rate**: ~7% (elevated from real-world ~0.17% for demo visibility)
- **Temporal**: Nighttime hours (11 PM – 5 AM) have 3× higher fraud rate
- **Geographic**: Nigeria, Romania, Russia have 4× higher fraud rate
- **User behavior**: New users have 3× higher fraud rate
- **Payment methods**: Apple Pay / Google Pay are safer (tokenization); Credit/Debit cards are higher risk
- **Pre-fraud signals**: Multiple small transactions (<$10) within an hour → 5× higher subsequent fraud rate
- **Merchant risk**: Online and ATM categories have higher fraud rates

### Enriched Columns (beyond standard Kaggle dataset)
| Column | Description |
|---|---|
| `payment_type` | Credit Card, Apple Pay, Google Pay, Samsung Pay, Debit Card |
| `country` | 11 countries with varying risk profiles |
| `hour` | Hour of day (0–23) derived from Time |
| `is_weekend` | Boolean weekend flag |
| `user_type` | new / existing |
| `transaction_count` | Simulated user transaction history |
| `small_tx_sequence` | # small transactions (<$10) in last hour (pre-fraud signal) |
| `merchant_category` | retail, food, travel, entertainment, online, gas, ATM |

---

## 🛠 Tech Stack

- **[Streamlit](https://streamlit.io/)** – Interactive web dashboard framework
- **[Pandas](https://pandas.pydata.org/)** / **[NumPy](https://numpy.org/)** – Data manipulation
- **[Scikit-learn](https://scikit-learn.org/)** – Random Forest ML model
- **[Plotly Express](https://plotly.com/python/plotly-express/)** – Interactive charts
- **[Joblib](https://joblib.readthedocs.io/)** – Model serialization

---

## 📜 License

See [LICENSE](LICENSE) for details.