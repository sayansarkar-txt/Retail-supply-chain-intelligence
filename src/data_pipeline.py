# =============================================================================
# src/data_pipeline.py
# ETL Pipeline — Load, Validate, Clean, and Save both datasets
# =============================================================================

import pandas as pd
import numpy as np
import os

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
SAMPLE_DIR    = "data/sample"


# ─────────────────────────────────────────────
# SUPERSTORE PIPELINE
# ─────────────────────────────────────────────

def load_superstore(path: str = None) -> pd.DataFrame:
    """Load raw Superstore CSV with encoding fix."""
    if path is None:
        path = os.path.join(RAW_DIR, "superstore_sales.csv")
    df = pd.read_csv(path, encoding="latin-1")
    print(f"[Superstore] Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def clean_superstore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Superstore dataset:
    - Parse dates
    - Compute lead time
    - Derive margin
    - Create stockout proxy label
    - Drop duplicates
    """
    df = df.copy()

    # ── Parse dates
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  dayfirst=True)

    # ── Derived columns
    df["lead_time"]   = (df["Ship Date"] - df["Order Date"]).dt.days
    df["margin"]      = (df["Profit"] / df["Sales"]).replace([np.inf, -np.inf], np.nan)
    df["year"]        = df["Order Date"].dt.year
    df["month"]       = df["Order Date"].dt.month
    df["quarter"]     = df["Order Date"].dt.quarter
    df["day_of_week"] = df["Order Date"].dt.dayofweek

    # ── Stockout proxy: low qty + zero discount = likely out-of-stock pressure
    qty_threshold = df["Quantity"].quantile(0.15)
    df["stockout_proxy"] = (
        (df["Quantity"] <= qty_threshold) & (df["Discount"] == 0)
    ).astype(int)

    # ── Drop duplicates on Order ID + Product
    before = len(df)
    df = df.drop_duplicates(subset=["Order ID", "Product Name"])
    print(f"[Superstore] Removed {before - len(df)} duplicate rows")

    # ── Drop rows with null Sales/Quantity
    df = df.dropna(subset=["Sales", "Quantity"])

    print(f"[Superstore] Clean shape: {df.shape}")
    return df


def save_superstore(df: pd.DataFrame) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out = os.path.join(PROCESSED_DIR, "superstore_clean.csv")
    df.to_csv(out, index=False)
    print(f"[Superstore] Saved → {out}")


# ─────────────────────────────────────────────
# WALMART PIPELINE
# ─────────────────────────────────────────────

def load_walmart(path: str = None) -> pd.DataFrame:
    """Load raw Walmart store sales CSV."""
    if path is None:
        path = os.path.join(RAW_DIR, "walmart_store_sales.csv")
    df = pd.read_csv(path)
    print(f"[Walmart] Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def clean_walmart(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Walmart dataset:
    - Parse Date
    - Sort by Store + Date
    - Clip negative sales (data errors)
    - Add calendar features
    """
    df = df.copy()

    # ── Parse date
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # ── Clip negative weekly sales (artifacts in raw data)
    neg_count = (df["Weekly_Sales"] < 0).sum()
    df["Weekly_Sales"] = df["Weekly_Sales"].clip(lower=0)
    print(f"[Walmart] Clipped {neg_count} negative sales rows to 0")

    # ── Calendar features
    df["year"]        = df["Date"].dt.year
    df["month"]       = df["Date"].dt.month
    df["week"]        = df["Date"].dt.isocalendar().week.astype(int)
    df["quarter"]     = df["Date"].dt.quarter
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_month_end"]    = df["Date"].dt.is_month_end.astype(int)
    df["is_quarter_end"]  = df["Date"].dt.is_quarter_end.astype(int)

    # ── Drop nulls
    df = df.dropna(subset=["Weekly_Sales"])

    print(f"[Walmart] Clean shape: {df.shape}")
    return df


def save_walmart(df: pd.DataFrame) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out = os.path.join(PROCESSED_DIR, "walmart_clean.csv")
    df.to_csv(out, index=False)
    print(f"[Walmart] Saved → {out}")


# ─────────────────────────────────────────────
# SAMPLE DATA GENERATOR (for GitHub demo)
# ─────────────────────────────────────────────

def generate_demo_data(n_stores: int = 5, n_weeks: int = 52) -> pd.DataFrame:
    """
    Generate synthetic weekly sales data for dashboard demo
    when real Kaggle data is not present.
    """
    np.random.seed(42)
    dates  = pd.date_range("2021-01-01", periods=n_weeks, freq="W")
    rows   = []

    for store in range(1, n_stores + 1):
        base = np.random.randint(80_000, 200_000)
        for i, date in enumerate(dates):
            seasonal = 1 + 0.25 * np.sin(2 * np.pi * i / 52)
            noise    = np.random.normal(1, 0.08)
            trend    = 1 + 0.002 * i
            holiday  = 1.20 if date.month in [11, 12] else 1.0
            sales    = max(0, base * seasonal * noise * trend * holiday)
            rows.append({
                "Store":        store,
                "Date":         date,
                "Weekly_Sales": round(sales, 2),
                "Holiday_Flag": int(date.month in [11, 12]),
                "Temperature":  round(np.random.uniform(20, 95), 1),
                "Fuel_Price":   round(np.random.uniform(2.5, 4.5), 3),
                "CPI":          round(np.random.uniform(120, 220), 3),
                "Unemployment": round(np.random.uniform(4, 10), 3),
            })

    demo_df = pd.DataFrame(rows)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    demo_df.to_csv(os.path.join(SAMPLE_DIR, "demo_data.csv"), index=False)
    print(f"[Demo] Generated {len(demo_df):,} rows → data/sample/demo_data.csv")
    return demo_df


# ─────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────

def run_pipeline():
    print("=" * 55)
    print("  RETAIL SUPPLY CHAIN — DATA PIPELINE")
    print("=" * 55)

    # Superstore
    ss_raw   = load_superstore()
    ss_clean = clean_superstore(ss_raw)
    save_superstore(ss_clean)

    # Walmart
    wm_raw   = load_walmart()
    wm_clean = clean_walmart(wm_raw)
    save_walmart(wm_clean)

    # Demo data
    generate_demo_data()

    print("\n✅ Pipeline complete. Check data/processed/")


if __name__ == "__main__":
    run_pipeline()
