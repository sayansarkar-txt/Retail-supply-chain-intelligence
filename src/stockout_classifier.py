# =============================================================================
# src/stockout_classifier.py
# Stockout Risk Classifier — Binary classification with SMOTE + XGBoost
# =============================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
    average_precision_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import joblib
import json
import os

PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"
FIGURES_DIR   = "reports/figures"

TARGET = "stockout_risk"


def load_feature_cols() -> list:
    path = os.path.join(MODELS_DIR, "feature_columns.json")
    with open(path) as f:
        cols = json.load(f)
    exclude = {"Weekly_Sales", "Date", "stockout_risk", "future_sales"}
    return [c for c in cols if c not in exclude]


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_stockout_classifier(df: pd.DataFrame = None):
    """
    Train binary stockout risk classifier.

    Key design decisions:
    - SMOTE oversampling: stockout events are rare (~20-30% of rows).
      Without handling this, model learns to always predict 'no risk'.
    - scale_pos_weight: extra penalty for missing stockout predictions
      (false negatives are costly in supply chain)
    - TimeSeriesSplit: respects temporal order, no leakage
    - Threshold tuning: default 0.5 may not be optimal — we evaluate
      multiple thresholds and save the best F1 threshold
    """
    if df is None:
        path = os.path.join(PROCESSED_DIR, "features_final.csv")
        df = pd.read_csv(path, parse_dates=["Date"])

    feature_cols = load_feature_cols()
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[TARGET]

    print("=" * 55)
    print("  STOCKOUT RISK CLASSIFIER — XGBoost + SMOTE")
    print("=" * 55)
    print(f"Features: {len(feature_cols)}  |  Samples: {len(X):,}")
    class_counts = y.value_counts()
    print(f"Class balance  →  No Risk: {class_counts.get(0,0):,}  |  Stockout: {class_counts.get(1,0):,}")
    print(f"Imbalance ratio: {class_counts.get(0,1)/class_counts.get(1,1):.1f}:1")

    # ── SMOTE to balance classes
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"After SMOTE → {len(X_res):,} samples (balanced)")

    # ── XGBoost Classifier
    pos_weight = class_counts.get(0, 1) / class_counts.get(1, 1)
    model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=5,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )

    # ── TimeSeriesSplit CV on original (not SMOTE'd) data
    tscv = TimeSeriesSplit(n_splits=5)
    roc_scores, pr_scores = [], []

    print("\n  Cross-Validation (TimeSeriesSplit):")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Apply SMOTE only to train fold
        X_tr_s, y_tr_s = smote.fit_resample(X_tr, y_tr)

        model.fit(X_tr_s, y_tr_s, eval_set=[(X_val, y_val)], verbose=False)
        proba = model.predict_proba(X_val)[:, 1]

        roc = roc_auc_score(y_val, proba)
        pr  = average_precision_score(y_val, proba)
        roc_scores.append(roc)
        pr_scores.append(pr)
        print(f"  Fold {fold+1}  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")

    print(f"\n  Mean ROC-AUC = {np.mean(roc_scores):.4f}  ±{np.std(roc_scores):.4f}")
    print(f"  Mean PR-AUC  = {np.mean(pr_scores):.4f}  ±{np.std(pr_scores):.4f}")

    # ── Final model on SMOTE'd full data
    model.fit(X_res, y_res)

    # ── Full evaluation
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    roc   = roc_auc_score(y, proba)
    acc   = (preds == y).mean()

    print(f"\n  Final Model — Full Data Evaluation:")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  ROC-AUC:   {roc:.4f}")
    print(f"\n{classification_report(y, preds, target_names=['No Risk', 'Stockout Risk'])}")

    # ── Find best threshold by F1
    best_threshold = _find_best_threshold(y, proba)
    print(f"  Best threshold for F1: {best_threshold:.2f}")

    # ── Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "stockout_classifier.pkl"))
    with open(os.path.join(MODELS_DIR, "classifier_threshold.json"), "w") as f:
        json.dump({"threshold": best_threshold, "roc_auc": round(roc, 4)}, f)
    print(f"\n✅ Model saved → {MODELS_DIR}/stockout_classifier.pkl")

    # ── Plots
    _plot_confusion_matrix(y, preds)
    _plot_roc_curve(y, proba)
    _plot_feature_importance(model, feature_cols)

    return model


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

def predict_stockout_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return stockout risk probability for each row in df.
    Adds columns: stockout_prob, stockout_pred, risk_level
    """
    model = joblib.load(os.path.join(MODELS_DIR, "stockout_classifier.pkl"))

    with open(os.path.join(MODELS_DIR, "classifier_threshold.json")) as f:
        threshold = json.load(f)["threshold"]

    feature_cols = load_feature_cols()
    feature_cols = [c for c in feature_cols if c in df.columns]

    proba = model.predict_proba(df[feature_cols])[:, 1]
    preds = (proba >= threshold).astype(int)

    result = df[["Store", "Date"]].copy() if "Date" in df.columns else df[["Store"]].copy()
    result["stockout_prob"] = np.round(proba, 4)
    result["stockout_pred"] = preds
    result["risk_level"]    = pd.cut(
        proba,
        bins=[-0.001, 0.3, 0.5, 0.75, 1.001],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    )
    return result


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _find_best_threshold(y_true, y_proba) -> float:
    """Find threshold that maximises F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


def _plot_confusion_matrix(y_true, y_pred) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Risk", "Stockout Risk"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Stockout Classifier", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()


def _plot_roc_curve(y_true, y_proba) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#00aaff", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Stockout Classifier", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "roc_curve.png"), dpi=150)
    plt.close()


def _plot_feature_importance(model, feature_cols: list) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    top20 = importance.sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    top20.sort_values().plot(kind="barh", ax=ax, color="#ff6b35")
    ax.set_title("Top 20 Feature Importances — Stockout Classifier", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "feature_importance_classifier.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    train_stockout_classifier()
