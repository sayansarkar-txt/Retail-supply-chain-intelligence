# =============================================================================
# src/alert_engine.py
# Alert Engine — Convert model outputs into prioritised business alerts
# =============================================================================

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional
import json
import os

PROCESSED_DIR = "data/processed"


# ─────────────────────────────────────────────
# DATA STRUCTURE
# ─────────────────────────────────────────────

@dataclass
class StockoutAlert:
    store_id:           int
    forecast_date:      str
    week_ahead:         int
    risk_probability:   float
    forecast_sales:     float
    sales_drop_pct:     float
    severity:           str          # CRITICAL | HIGH | MEDIUM
    recommended_action: str
    days_to_act:        int


SEVERITY_CONFIG = {
    "CRITICAL": {
        "min_prob":  0.75,
        "action":    "IMMEDIATE: Contact supplier today — order emergency stock",
        "days":      1,
        "color":     "#ff3b3b",
    },
    "HIGH": {
        "min_prob":  0.50,
        "action":    "URGENT: Schedule reorder within 3 days",
        "days":      3,
        "color":     "#ff9933",
    },
    "MEDIUM": {
        "min_prob":  0.30,
        "action":    "MONITOR: Review inventory levels this week",
        "days":      7,
        "color":     "#ffdd00",
    },
}


# ─────────────────────────────────────────────
# CORE ALERT GENERATION
# ─────────────────────────────────────────────

def generate_alerts(
    forecast_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    store_baseline: Optional[pd.Series] = None,
) -> List[StockoutAlert]:
    """
    Merge demand forecasts + stockout probabilities into
    ranked, actionable alerts.

    Args:
        forecast_df:    output of predict_4_weeks() for all stores
        risk_df:        output of predict_stockout_risk()
        store_baseline: pd.Series mapping store_id → avg weekly sales
    """
    if "Date" not in forecast_df.columns and "forecast_date" in forecast_df.columns:
        forecast_df = forecast_df.rename(columns={"forecast_date": "Date"})

    merged = forecast_df.merge(
        risk_df[["Store", "Date", "stockout_prob", "risk_level"]],
        on=["Store", "Date"],
        how="left",
    )

    alerts = []
    for _, row in merged.iterrows():
        prob = row.get("stockout_prob", 0.0)
        if prob < 0.30:
            continue   # below threshold — not worth flagging

        # Calculate sales drop vs store baseline
        baseline_sales = (
            float(store_baseline.get(row["Store"], row.get("forecast_sales", 100_000)))
            if store_baseline is not None
            else row.get("forecast_sales", 100_000)
        )
        forecast_val  = row.get("forecast_sales", baseline_sales)
        drop_pct      = max(0, (baseline_sales - forecast_val) / (baseline_sales + 1e-6)) * 100

        # Assign severity
        if prob >= SEVERITY_CONFIG["CRITICAL"]["min_prob"]:
            sev = "CRITICAL"
        elif prob >= SEVERITY_CONFIG["HIGH"]["min_prob"]:
            sev = "HIGH"
        else:
            sev = "MEDIUM"

        cfg = SEVERITY_CONFIG[sev]

        alerts.append(StockoutAlert(
            store_id=int(row["Store"]),
            forecast_date=str(row["Date"])[:10],
            week_ahead=int(row.get("week_ahead", 1)),
            risk_probability=round(float(prob), 4),
            forecast_sales=round(float(forecast_val), 2),
            sales_drop_pct=round(float(drop_pct), 1),
            severity=sev,
            recommended_action=cfg["action"],
            days_to_act=cfg["days"],
        ))

    # Sort: CRITICAL first, then by probability descending
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
    alerts.sort(key=lambda a: (severity_order[a.severity], -a.risk_probability))

    print(f"[Alerts] Generated {len(alerts)} alerts | "
          f"CRITICAL: {sum(1 for a in alerts if a.severity=='CRITICAL')} | "
          f"HIGH: {sum(1 for a in alerts if a.severity=='HIGH')} | "
          f"MEDIUM: {sum(1 for a in alerts if a.severity=='MEDIUM')}")

    return alerts


def alerts_to_dataframe(alerts: List[StockoutAlert]) -> pd.DataFrame:
    """Convert list of alerts to a clean DataFrame for display."""
    if not alerts:
        return pd.DataFrame(columns=[
            "store_id", "forecast_date", "week_ahead",
            "risk_probability", "forecast_sales", "sales_drop_pct",
            "severity", "recommended_action", "days_to_act",
        ])
    return pd.DataFrame([asdict(a) for a in alerts])


# ─────────────────────────────────────────────
# DEMO ALERTS (for dashboard when models not trained)
# ─────────────────────────────────────────────

def get_demo_alerts(n: int = 15) -> pd.DataFrame:
    """
    Generate realistic-looking demo alerts for dashboard
    development without needing trained models.
    """
    np.random.seed(7)
    stores    = np.random.randint(1, 46, size=n)
    # Generate enough probs to always cover n items
    n_crit = max(1, n // 5)
    n_high = max(1, n // 4)
    n_med  = n - n_crit - n_high
    probs  = np.concatenate([
        np.random.uniform(0.75, 0.98, n_crit),
        np.random.uniform(0.50, 0.74, n_high),
        np.random.uniform(0.30, 0.49, n_med),
    ])
    probs  = np.sort(probs)[::-1]
    weeks     = np.random.randint(1, 5, size=n)
    forecasts = np.random.randint(60_000, 250_000, size=n)
    drops     = np.random.uniform(0, 45, size=n)

    rows = []
    for i in range(n):
        p = probs[i]
        if p >= 0.75:
            sev, action, days = "CRITICAL", "IMMEDIATE: Contact supplier today", 1
        elif p >= 0.50:
            sev, action, days = "HIGH",     "URGENT: Schedule reorder within 3 days", 3
        else:
            sev, action, days = "MEDIUM",   "MONITOR: Review inventory this week", 7

        rows.append({
            "store_id":           int(stores[i]),
            "forecast_date":      str(pd.Timestamp("2023-10-01") + pd.Timedelta(weeks=int(weeks[i]))),
            "week_ahead":         int(weeks[i]),
            "risk_probability":   round(float(p), 4),
            "forecast_sales":     int(forecasts[i]),
            "sales_drop_pct":     round(float(drops[i]), 1),
            "severity":           sev,
            "recommended_action": action,
            "days_to_act":        days,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    demo = get_demo_alerts()
    print(demo.to_string())
