# dashboard/views/stockout_risk.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys

BASE      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(BASE, "data", "processed")
MODELS    = os.path.join(BASE, "models")
DARK      = "plotly_dark"

PLOTLY_LAYOUT = dict(
    template=DARK,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,46,0.6)",
    font=dict(family="Inter", color="#c9d1d9"),
    margin=dict(l=20, r=20, t=40, b=20),
)

@st.cache_data
def load_data():
    for fname in ["features_final.csv", "walmart_clean.csv"]:
        p = os.path.join(PROCESSED, fname)
        if os.path.exists(p):
            return pd.read_csv(p, parse_dates=["Date"])
    return None


def simulate_risk(df):
    np.random.seed(42)
    risk = df[["Store","Date"]].copy()
    base = np.random.uniform(0.08, 0.55, size=len(df))
    high_risk = [3, 7, 14, 22, 33, 38, 41]
    mask = df["Store"].isin(high_risk)
    base[mask] = np.clip(base[mask] * 1.7, 0, 1)
    # Add seasonality
    week_num = df["Date"].dt.isocalendar().week.astype(int)
    seasonal = 0.15 * np.sin(2 * np.pi * week_num / 52)
    base = np.clip(base + seasonal, 0, 1)
    risk["stockout_prob"] = np.round(base, 4)
    risk["risk_level"] = pd.cut(
        risk["stockout_prob"],
        bins=[-0.001, 0.3, 0.5, 0.75, 1.001],
        labels=["LOW","MEDIUM","HIGH","CRITICAL"],
    )
    return risk


def render():
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:1.9rem;font-weight:700;color:#e6edf3;margin:0;">
            ⚠️ Stockout Risk Analysis
        </h1>
        <p style="color:#6e7c8c;margin-top:4px;font-size:0.88rem;">
            Store × time risk heatmap · probability distribution · drill-down per store
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    if df is None:
        st.error("❌ Data not found. Check `data/processed/` folder.")
        return

    risk_df = simulate_risk(df)

    # ── KPIs
    total    = len(risk_df)
    critical = (risk_df["risk_level"] == "CRITICAL").sum()
    high     = (risk_df["risk_level"] == "HIGH").sum()
    medium   = (risk_df["risk_level"] == "MEDIUM").sum()
    low      = (risk_df["risk_level"] == "LOW").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔴 Critical",        f"{critical}",  f"{critical/total*100:.1f}%")
    c2.metric("🟠 High Risk",       f"{high}",      f"{high/total*100:.1f}%")
    c3.metric("🟡 Medium Risk",     f"{medium}",    f"{medium/total*100:.1f}%")
    c4.metric("✅ Low Risk",        f"{low}",       f"{low/total*100:.1f}%")
    c5.metric("📊 Avg Risk Score",  f"{risk_df['stockout_prob'].mean():.2f}", "out of 1.0")

    st.markdown("---")

    # ── Heatmap
    risk_df["Month"] = risk_df["Date"].dt.to_period("M").astype(str)
    pivot = risk_df.groupby(["Store","Month"])["stockout_prob"].mean().unstack(fill_value=0)
    pivot = pivot.iloc[:, -12:]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[f"Store {s}" for s in pivot.index],
        colorscale=[
            [0.0,  "#0d2d1a"],
            [0.3,  "#166534"],
            [0.5,  "#854d0e"],
            [0.75, "#9a3412"],
            [1.0,  "#7f1d1d"],
        ],
        zmin=0, zmax=1,
        hoverongaps=False,
        colorbar=dict(title="Risk", tickfont=dict(color="#8b9ab0")),
        hovertemplate="Store %{y}<br>%{x}<br>Risk: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=680,
        title=dict(text="Stockout Risk Heatmap — Store × Month (Last 12 Months)",
                   font=dict(size=14, color="#e6edf3")),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)

    # ── Risk distribution
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=risk_df["stockout_prob"], nbinsx=40,
        marker=dict(color="#3b82f6", line=dict(color="#0f1117", width=0.5)),
        hovertemplate="Risk: %{x:.2f}<br>Count: %{y}<extra></extra>",
        name="Distribution",
    ))
    fig2.add_vline(x=0.5,  line_dash="dash", line_color="#f97316",
                   annotation_text="HIGH", annotation_font_color="#f97316")
    fig2.add_vline(x=0.75, line_dash="dash", line_color="#ef4444",
                   annotation_text="CRITICAL", annotation_font_color="#ef4444")
    fig2.update_layout(**PLOTLY_LAYOUT, height=320,
        title=dict(text="Risk Probability Distribution", font=dict(size=14, color="#e6edf3")),
        xaxis=dict(title="Stockout Probability", gridcolor="#1e2535"),
        yaxis=dict(title="Count", gridcolor="#1e2535"),
        showlegend=False,
    )
    col_l.plotly_chart(fig2, use_container_width=True)

    # ── Top risky stores
    top10 = (risk_df.groupby("Store")["stockout_prob"]
             .mean().sort_values(ascending=False).head(10).reset_index())
    top10.columns = ["Store","Risk"]
    colors = ["#ef4444" if r >= 0.75 else "#f97316" if r >= 0.5 else "#eab308"
              for r in top10["Risk"]]

    fig3 = go.Figure(go.Bar(
        x=top10["Risk"],
        y=[f"Store {s}" for s in top10["Store"]],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="Store %{y}: %{x:.2f}<extra></extra>",
    ))
    fig3.update_layout(**PLOTLY_LAYOUT, height=320,
        title=dict(text="Top 10 Highest Risk Stores", font=dict(size=14, color="#e6edf3")),
        xaxis=dict(title="Avg Risk Score", gridcolor="#1e2535", range=[0,1]),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
    )
    col_r.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── Store drill-down
    st.markdown("### 🔎 Store Risk Drill-Down")
    sel = st.selectbox("Select Store", sorted(risk_df["Store"].unique()))
    store_risk = risk_df[risk_df["Store"] == sel].sort_values("Date")

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=store_risk["Date"], y=store_risk["stockout_prob"],
        fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
        line=dict(color="#ef4444", width=2),
        name="Risk Score",
        hovertemplate="%{x|%b %d, %Y}: %{y:.3f}<extra></extra>",
    ))
    fig4.add_hrect(y0=0.75, y1=1.0,
                   fillcolor="rgba(239,68,68,0.08)", line_width=0,
                   annotation_text="CRITICAL ZONE", annotation_position="top left",
                   annotation_font_color="#ef4444", annotation_font_size=11)
    fig4.add_hrect(y0=0.5, y1=0.75,
                   fillcolor="rgba(249,115,22,0.06)", line_width=0,
                   annotation_text="HIGH ZONE", annotation_position="top left",
                   annotation_font_color="#f97316", annotation_font_size=11)
    fig4.add_hline(y=0.5,  line_dash="dash", line_color="#f97316", line_width=1)
    fig4.add_hline(y=0.75, line_dash="dash", line_color="#ef4444", line_width=1)
    fig4.update_layout(**PLOTLY_LAYOUT, height=320,
        title=dict(text=f"Store {sel} — Stockout Risk Timeline",
                   font=dict(size=14, color="#e6edf3")),
        yaxis=dict(range=[0,1], title="Risk Probability", gridcolor="#1e2535"),
        xaxis=dict(gridcolor="#1e2535"),
    )
    st.plotly_chart(fig4, use_container_width=True)
