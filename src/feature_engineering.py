# =============================================================================
# src/feature_engineering.py
# Feature Engineering — Lag, Rolling, Calendar, Interaction, Stockout Label
# =============================================================================

import pandas as pd
import numpy as np
import json
import os

PROCESSED_DIR = "data/processed"


# ─────────────────────────────────────────────
# 1. CALENDAR FEATURES
# ─────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Extract rich calendar signals from date column.
    These help the model learn holiday cycles, year-end spikes, etc.
    """
    df = df.copy()
    d = df[date_col]

    df["year"]             = d.dt.year
    df["month"]            = d.dt.month
    df["week"]             = d.dt.isocalendar().week.astype(int)
    df["quarter"]          = d.dt.quarter
    df["day_of_week"]      = d.dt.dayofweek
    df["is_month_start"]   = d.dt.is_month_start.astype(int)
    df["is_month_end"]     = d.dt.is_month_end.astype(int)
    df["is_quarter_start"] = d.dt.is_quarter_start.astype(int)
    df["is_quarter_end"]   = d.dt.is_quarter_end.astype(int)
    df["is_year_end"]      = ((d.dt.month == 12) & (d.dt.day >= 25)).astype(int)

    # Cyclical encoding of month and week (sin/cos preserves circular nature)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"]  = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"]  = np.cos(2 * np.pi * df["week"] / 52)

    return df


# ─────────────────────────────────────────────
# 2. LAG FEATURES
# ─────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    lags: list = [1, 2, 3, 4, 6, 8, 12, 26, 52]
) -> pd.DataFrame:
    """
    Lag features: sales from N weeks ago for the same store.
    Gives the model historical context at multiple horizons.

    lags list explanation:
    - 1, 2, 3, 4  → recent short-term trend
    - 6, 8        → 6-8 weeks ago (medium term)
    - 12          → ~3 months ago
    - 26          → ~6 months ago (half-year seasonality)
    - 52          → same week last year (strong seasonal signal)
    """
    df = df.copy()
    df = df.sort_values([group_col, "Date"])

    for lag in lags:
        col_name = f"lag_{lag}w"
        df[col_name] = df.groupby(group_col)[target_col].shift(lag)

    return df


# ─────────────────────────────────────────────
# 3. ROLLING WINDOW FEATURES
# ─────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    windows: list = [4, 8, 12, 26]
) -> pd.DataFrame:
    """
    Rolling mean, std, min, max over past N weeks.
    All shifted by 1 to prevent leakage (only past data used).

    - roll_mean_Nw  → average demand level
    - roll_std_Nw   → demand volatility (high std = risky)
    - roll_min_Nw   → worst-case recent demand
    - roll_max_Nw   → peak recent demand
    - roll_cv_Nw    → coefficient of variation (volatility ratio)
    """
    df = df.copy()
    df = df.sort_values([group_col, "Date"])

    for w in windows:
        grp = df.groupby(group_col)[target_col]

        shifted = grp.transform(lambda x: x.shift(1))

        df[f"roll_mean_{w}w"] = (
            shifted.groupby(df[group_col]).transform(lambda x: x.rolling(w, min_periods=1).mean())
        )
        df[f"roll_std_{w}w"] = (
            shifted.groupby(df[group_col]).transform(lambda x: x.rolling(w, min_periods=2).std())
        )
        df[f"roll_min_{w}w"] = (
            shifted.groupby(df[group_col]).transform(lambda x: x.rolling(w, min_periods=1).min())
        )
        df[f"roll_max_{w}w"] = (
            shifted.groupby(df[group_col]).transform(lambda x: x.rolling(w, min_periods=1).max())
        )
        # Coefficient of variation = std / mean (avoids division by zero)
        df[f"roll_cv_{w}w"] = (
            df[f"roll_std_{w}w"] / (df[f"roll_mean_{w}w"] + 1e-6)
        )

    return df


# ─────────────────────────────────────────────
# 4. TREND FEATURES
# ─────────────────────────────────────────────

def add_trend_features(
    df: pd.DataFrame,
    target_col: str,
    group_col: str
) -> pd.DataFrame:
    """
    Momentum and acceleration signals.

    - sales_diff_1w   → week-over-week change (momentum)
    - sales_diff_4w   → 4-week change (medium trend)
    - sales_pct_chg_1w → percentage change (normalized momentum)
    - sales_accel     → change in change (acceleration signal)
    """
    df = df.copy()
    df = df.sort_values([group_col, "Date"])

    grp = df.groupby(group_col)[target_col]

    df["sales_diff_1w"]    = grp.transform(lambda x: x.shift(1).diff(1))
    df["sales_diff_4w"]    = grp.transform(lambda x: x.shift(1).diff(4))
    df["sales_pct_chg_1w"] = grp.transform(
        lambda x: x.shift(1).pct_change(1).replace([np.inf, -np.inf], np.nan)
    )
    df["sales_accel"]      = df.groupby(group_col)["sales_diff_1w"].transform(
        lambda x: x.diff(1)
    )

    return df


# ─────────────────────────────────────────────
# 5. STORE-LEVEL FEATURES
# ─────────────────────────────────────────────

def add_store_features(df: pd.DataFrame, target_col: str, group_col: str) -> pd.DataFrame:
    """
    Relative performance features comparing a store to the overall average.

    - store_avg_sales   → historical average for this store
    - store_sales_rank  → rank among all stores (1 = highest)
    - sales_vs_avg_ratio → this week vs store average (>1 = above avg)
    """
    df = df.copy()

    store_avg = df.groupby(group_col)[target_col].mean().rename("store_avg_sales")
    df = df.merge(store_avg, on=group_col, how="left")

    store_rank = (
        store_avg.rank(ascending=False).astype(int).rename("store_sales_rank")
    )
    df = df.merge(store_rank, on=group_col, how="left")

    df["sales_vs_avg_ratio"] = df[target_col] / (df["store_avg_sales"] + 1e-6)

    return df


# ─────────────────────────────────────────────
# 6. HOLIDAY / EXTERNAL FEATURES
# ─────────────────────────────────────────────

def add_holiday_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction terms between holiday flag and macroeconomic variables.
    These capture how holidays amplify or dampen the impact of
    external factors like fuel prices and unemployment.
    """
    df = df.copy()

    if "Holiday_Flag" in df.columns and "Fuel_Price" in df.columns:
        df["holiday_x_fuel"]         = df["Holiday_Flag"] * df["Fuel_Price"]
        df["holiday_x_unemployment"] = df["Holiday_Flag"] * df["Unemployment"]
        df["holiday_x_cpi"]          = df["Holiday_Flag"] * df["CPI"]

    # Pre/post holiday window (weeks around a holiday are also elevated)
    if "Holiday_Flag" in df.columns:
        df = df.sort_values(["Store", "Date"])
        df["pre_holiday"]  = df.groupby("Store")["Holiday_Flag"].shift(-1).fillna(0).astype(int)
        df["post_holiday"] = df.groupby("Store")["Holiday_Flag"].shift(1).fillna(0).astype(int)

    return df


# ─────────────────────────────────────────────
# 7. STOCKOUT LABEL CREATION
# ─────────────────────────────────────────────

def create_stockout_label(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    horizon_weeks: int = 4,
    threshold_pct: float = 0.25
) -> pd.DataFrame:
    """
    Binary stockout risk label (TARGET for classifier).

    Logic:
    - Look ahead `horizon_weeks` into the future for each store
    - If the sum of the next N weeks of sales falls below the
      25th percentile of that store's historical baseline → label = 1
    - This simulates "demand will crash soon" = stockout risk

    horizon_weeks=4 means we predict 4 weeks in advance.
    threshold_pct=0.25 means bottom 25% of store history = risk.
    """
    df = df.copy()
    df = df.sort_values([group_col, "Date"])

    # Future N-week rolling sum (shift by -horizon to look forward)
    df["future_sales"] = df.groupby(group_col)[target_col].transform(
        lambda x: x.shift(-horizon_weeks).rolling(horizon_weeks, min_periods=1).sum()
    )

    # Per-store threshold (25th percentile of weekly sales)
    threshold = df.groupby(group_col)[target_col].transform(
        lambda x: x.quantile(threshold_pct)
    )

    df["stockout_risk"] = (df["future_sales"] < threshold).astype(int)

    # Drop the raw future column (not a valid feature — it's leakage)
    df = df.drop(columns=["future_sales"])

    pos = df["stockout_risk"].sum()
    neg = len(df) - pos
    print(f"[Label] Stockout events: {pos:,}  |  No-risk: {neg:,}  |  Ratio: {pos/len(df):.2%}")

    return df


# ─────────────────────────────────────────────
# 8. FULL FEATURE MATRIX BUILDER
# ─────────────────────────────────────────────

FEATURE_COLS = [
    # Store ID
    "Store",
    # Calendar
    "year", "month", "week", "quarter", "day_of_week",
    "is_month_start", "is_month_end", "is_quarter_start",
    "is_quarter_end", "is_year_end",
    "month_sin", "month_cos", "week_sin", "week_cos",
    # External / macroeconomic
    "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment",
    # Holiday interactions
    "holiday_x_fuel", "holiday_x_unemployment", "holiday_x_cpi",
    "pre_holiday", "post_holiday",
    # Lag features
    "lag_1w", "lag_2w", "lag_3w", "lag_4w",
    "lag_6w", "lag_8w", "lag_12w", "lag_26w", "lag_52w",
    # Rolling features
    "roll_mean_4w", "roll_mean_8w", "roll_mean_12w", "roll_mean_26w",
    "roll_std_4w",  "roll_std_8w",  "roll_std_12w",
    "roll_min_4w",  "roll_max_4w",
    "roll_cv_4w",   "roll_cv_8w",
    # Trend features
    "sales_diff_1w", "sales_diff_4w",
    "sales_pct_chg_1w", "sales_accel",
    # Store-level
    "store_avg_sales", "store_sales_rank", "sales_vs_avg_ratio",
]


def build_feature_matrix(walmart_clean_path: str = None) -> pd.DataFrame:
    """
    Full pipeline: load cleaned Walmart data → apply all
    feature engineering → create stockout label → save.
    """
    if walmart_clean_path is None:
        walmart_clean_path = os.path.join(PROCESSED_DIR, "walmart_clean.csv")

    print("=" * 55)
    print("  FEATURE ENGINEERING PIPELINE")
    print("=" * 55)

    df = pd.read_csv(walmart_clean_path, parse_dates=["Date"])
    print(f"[FE] Input shape: {df.shape}")

    df = add_calendar_features(df)
    print("[FE] ✓ Calendar features added")

    df = add_lag_features(df, "Weekly_Sales", "Store")
    print("[FE] ✓ Lag features added")

    df = add_rolling_features(df, "Weekly_Sales", "Store")
    print("[FE] ✓ Rolling features added")

    df = add_trend_features(df, "Weekly_Sales", "Store")
    print("[FE] ✓ Trend features added")

    df = add_store_features(df, "Weekly_Sales", "Store")
    print("[FE] ✓ Store features added")

    df = add_holiday_interaction_features(df)
    print("[FE] ✓ Holiday interaction features added")

    df = create_stockout_label(df, "Weekly_Sales", "Store")
    print("[FE] ✓ Stockout label created")

    # Drop rows with NaN caused by lag/rolling lookback
    before = len(df)
    df = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns])
    print(f"[FE] Dropped {before - len(df)} rows with NaN (lag warmup period)")
    print(f"[FE] Final feature matrix: {df.shape}")

    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "features_final.csv")
    df.to_csv(out_path, index=False)
    print(f"[FE] Saved → {out_path}")

    # Save feature column list
    os.makedirs("models", exist_ok=True)
    with open("models/feature_columns.json", "w") as f:
        valid_cols = [c for c in FEATURE_COLS if c in df.columns]
        json.dump(valid_cols, f, indent=2)
    print(f"[FE] Feature list saved → models/feature_columns.json ({len(valid_cols)} features)")

    return df


if __name__ == "__main__":
    build_feature_matrix()
