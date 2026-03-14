# 📦 Retail Supply Chain Intelligence Platform
## Demand Forecasting & Stockout Prevention

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> **Career pitch:** *"I built an end-to-end supply chain analytics platform on 3 years of retail data that predicts stockouts 4 weeks in advance with 87% accuracy — the kind of early warning system that saves retailers millions in lost sales."*

---

## 🎯 Business Problem

Retailers lose **$1.75 trillion annually** to inventory distortion — products either overstocked (wasted capital) or understocked (lost sales, dissatisfied customers). This platform provides an AI-powered early warning system that identifies at-risk products **before** stockouts occur, enabling proactive replenishment decisions.

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Stockout Classification Accuracy | **87%** |
| ROC-AUC Score | **0.91** |
| Demand Forecast MAE | ~$4,200 / week |
| Prediction Horizon | **4 weeks ahead** |
| Stores Monitored | 45 |
| Features Engineered | 48 |

---

## 🏗 Architecture

```
Raw CSVs (Kaggle)
    │
    ▼
data_pipeline.py          ← ETL: load, clean, validate, save
    │
    ▼
feature_engineering.py    ← 48 features: lag, rolling, calendar,
    │                        trend, store, holiday interactions
    ▼
demand_forecaster.py      ← XGBoost regression, TimeSeriesSplit CV,
    │                        4-week iterative prediction
    ▼
stockout_classifier.py    ← XGBoost binary classifier + SMOTE,
    │                        threshold tuning, ROC-AUC evaluation
    ▼
alert_engine.py           ← Business logic: severity tiers,
    │                        recommended actions, days to act
    ▼
Streamlit Dashboard       ← 5 views: Overview · Forecast ·
                             Risk · Alerts · Product Analysis
```

---

## 📁 Project Structure

```
retail-supply-chain-intelligence/
├── data/
│   ├── raw/                    # Kaggle downloads (gitignored)
│   ├── processed/              # Cleaned + feature matrix
│   └── sample/                 # Demo data for GitHub
├── notebooks/
│   ├── 01_eda_superstore.ipynb
│   ├── 02_eda_walmart.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_demand_forecasting.ipynb
│   ├── 05_stockout_classifier.ipynb
│   └── 06_model_evaluation.ipynb
├── src/
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── demand_forecaster.py
│   ├── stockout_classifier.py
│   └── alert_engine.py
├── models/                     # Saved .pkl models
├── dashboard/
│   ├── app.py
│   └── views/
│       ├── overview.py
│       ├── demand_forecast.py
│       ├── stockout_risk.py
│       ├── alert_center.py
│       └── product_analysis.py
├── reports/figures/            # EDA + model plots
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/sayansarkar-txt/retail-supply-chain-intelligence.git
cd retail-supply-chain-intelligence
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Datasets

Download from Kaggle and place in `data/raw/`:

| Dataset | Kaggle Link |
|---------|-------------|
| Sample Superstore | [Search: "superstore dataset final"](https://www.kaggle.com/) |
| Walmart Store Sales | [Search: "walmart recruiting store sales"](https://www.kaggle.com/) |

Rename files to:
- `data/raw/superstore_sales.csv`
- `data/raw/walmart_store_sales.csv`

### 3. Run the Full Pipeline

```bash
# Step 1: Clean data
python src/data_pipeline.py

# Step 2: Feature engineering
python src/feature_engineering.py

# Step 3: Train demand forecaster
python src/demand_forecaster.py

# Step 4: Train stockout classifier
python src/stockout_classifier.py

# Step 5: Launch dashboard
cd dashboard && streamlit run app.py
```

---

## 🔧 Feature Engineering (48 Features)

| Category | Features | Purpose |
|----------|----------|---------|
| **Calendar** | year, month, week, quarter, month_sin/cos, week_sin/cos | Seasonal cycles |
| **Lag** | lag_1w to lag_52w (9 lags) | Historical demand memory |
| **Rolling** | mean, std, min, max, CV over 4/8/12/26w | Trend & volatility |
| **Trend** | diff_1w, diff_4w, pct_change, acceleration | Momentum signals |
| **Store** | avg_sales, rank, vs_avg_ratio | Relative performance |
| **Holiday** | flag, pre/post window, interactions with CPI/fuel | External shocks |

---

## 🛠 Tech Stack

| Layer | Tools |
|-------|-------|
| Data Processing | Pandas, NumPy |
| Machine Learning | XGBoost, Scikit-learn |
| Class Imbalance | imbalanced-learn (SMOTE) |
| Visualization | Plotly, Seaborn, Matplotlib |
| Dashboard | Streamlit |
| Model Persistence | Joblib |

---

## 💡 Key Design Decisions

**Why XGBoost over Linear Regression?**
Sales data has non-linear holiday interactions that tree-based models capture natively. XGBoost achieved ~30% lower MAE than a linear baseline.

**Why TimeSeriesSplit instead of random CV?**
Standard k-fold randomly shuffles time-ordered data, allowing future information to leak into training. TimeSeriesSplit preserves temporal order — essential for honest evaluation.

**Why SMOTE?**
Stockout events are rare (~20-30% of weeks). Without balancing, a model learns to always predict "no risk" and still achieves high accuracy — but fails when it actually matters.

**Why 4-week horizon?**
Typical retail replenishment lead times are 1-3 weeks. A 4-week forecast gives enough buffer for procurement teams to act before a stockout occurs.

---

## 📈 Dashboard Views

| View | Description |
|------|-------------|
| 📊 Executive Overview | KPIs, revenue trend, category breakdown, store heatmap |
| 📈 Demand Forecast | Per-store 4-week forecast with confidence band |
| ⚠️ Stockout Risk | Risk heatmap (store × month), probability distribution |
| 🔔 Alert Center | Severity-ranked alerts with recommended actions |
| 🔍 Product Analysis | Margin analysis, discount impact, top/bottom products |

---

## 👤 Author

**Sayan Sarkar**
B.Tech Computer Science & Engineering (Data Science)
Pragati Engineering College, Kakinada — Class of 2027
GitHub: [github.com/sayansarkar-txt](https://github.com/sayansarkar-txt)
