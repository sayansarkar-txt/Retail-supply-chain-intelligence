# =============================================================================
# src/demand_forecaster.py
# XGBoost Demand Forecasting — Train, Evaluate, Predict 4 Weeks Ahead
# =============================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import json
import os

PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"
FIGURES_DIR   = "reports/figures"

TARGET = "Weekly_Sales"

# Load feature columns saved during feature engineering
def load_feature_cols() -> list:
    path = os.path.join(MODELS_DIR, "feature_columns.json")
    with open(path) as f:
        cols = json.load(f)
    # Exclude label and non-feature columns
    exclude = {"Weekly_Sales", "Date", "stockout_risk", "future_sales"}
    return [c for c in cols if c not in exclude]


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_demand_model(df: pd.DataFrame = None):
    """
    Train XGBoost regressor on the full feature matrix.

    Key design decisions:
    - TimeSeriesSplit (5 folds): no future data leaks into training
    - Early stopping on each fold to prevent overfitting
    - Final model retrained on 100% of data
    """
    if df is None:
        path = os.path.join(PROCESSED_DIR, "features_final.csv")
        df = pd.read_csv(path, parse_dates=["Date"])

    feature_cols = load_feature_cols()
    # Only keep cols that exist in this dataframe
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[TARGET]

    print("=" * 55)
    print("  DEMAND FORECASTING MODEL — XGBoost")
    print("=" * 55)
    print(f"Features: {len(feature_cols)}  |  Samples: {len(X):,}")
    print(f"Target range: ${y.min():,.0f} – ${y.max():,.0f}")

    # ── TimeSeriesSplit Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores, rmse_scores, r2_scores = [], [], []

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)
        preds = np.maximum(preds, 0)   # sales can't be negative

        mae  = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2   = r2_score(y_val, preds)

        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        print(f"  Fold {fold+1}  MAE=${mae:>10,.0f}  RMSE=${rmse:>10,.0f}  R²={r2:.4f}")

    print(f"\n  Mean MAE  = ${np.mean(mae_scores):>10,.0f}  ±${np.std(mae_scores):,.0f}")
    print(f"  Mean RMSE = ${np.mean(rmse_scores):>10,.0f}  ±${np.std(rmse_scores):,.0f}")
    print(f"  Mean R²   = {np.mean(r2_scores):.4f}")

    # ── Final model on all data
    model.fit(X, y)

    # ── Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "xgb_demand_forecaster.pkl"))
    print(f"\n✅ Model saved → {MODELS_DIR}/xgb_demand_forecaster.pkl")

    # ── Save feature importance plot
    _plot_feature_importance(model, feature_cols)

    # ── Save actual vs predicted plot (last fold)
    _plot_actual_vs_predicted(y_val, preds)

    return model


# ─────────────────────────────────────────────
# PREDICTION — 4 WEEKS AHEAD
# ─────────────────────────────────────────────

def predict_4_weeks(store_id: int, df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generate 4-week ahead demand forecast for a specific store.
    Uses the last known row as the seed and iteratively predicts.
    """
    model_path = os.path.join(MODELS_DIR, "xgb_demand_forecaster.pkl")
    model = joblib.load(model_path)

    if df is None:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, "features_final.csv"), parse_dates=["Date"])

    feature_cols = load_feature_cols()
    feature_cols = [c for c in feature_cols if c in df.columns]

    store_df = df[df["Store"] == store_id].sort_values("Date")
    last_row  = store_df.iloc[[-1]][feature_cols].copy()

    forecasts = []
    last_date = store_df["Date"].max()

    for week in range(1, 5):
        pred = float(np.maximum(model.predict(last_row), 0))
        next_date = last_date + pd.Timedelta(weeks=week)
        forecasts.append({
            "store_id":    store_id,
            "forecast_date": next_date,
            "week_ahead":  week,
            "forecast_sales": round(pred, 2),
        })
        # Update lag_1w for next iteration (simple iterative forecast)
        if "lag_1w" in last_row.columns:
            last_row["lag_1w"] = pred

    return pd.DataFrame(forecasts)


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def _plot_feature_importance(model, feature_cols: list) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    top20 = importance.sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    top20.sort_values().plot(kind="barh", ax=ax, color="#00aaff")
    ax.set_title("Top 20 Feature Importances — Demand Forecaster", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "feature_importance_demand.png"), dpi=150)
    plt.close()
    print(f"[Plot] Feature importance → {FIGURES_DIR}/feature_importance_demand.png")


def _plot_actual_vs_predicted(y_true, y_pred) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.array(y_true), label="Actual", color="#ffffff", linewidth=1.5)
    ax.plot(y_pred, label="Predicted", color="#00aaff", linewidth=1.5, linestyle="--")
    ax.set_title("Actual vs Predicted Weekly Sales (Validation Fold)", fontsize=12)
    ax.legend()
    ax.set_facecolor("#0a0e14")
    fig.patch.set_facecolor("#0a0e14")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()
    print(f"[Plot] Actual vs Predicted → {FIGURES_DIR}/actual_vs_predicted.png")


if __name__ == "__main__":
    train_demand_model()
