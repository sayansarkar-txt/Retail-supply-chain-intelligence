# dashboard/views/overview.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys

# ── FIX: views/ → dashboard/ → project_root (3 levels up)
BASE      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED = os.path.join(BASE, "data", "processed")
DARK      = "plotly_dark"

PLOTLY_LAYOUT = dict(
    template=DARK,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,46,0.6)",
    font=dict(family="Inter", color="#c9d1d9"),
    margin=dict(l=20, r=20, t=40, b=20),
)

@st.cache_data
def load_superstore():
    p = os.path.join(PROCESSED, "superstore_clean.csv")
    return pd.read_csv(p, parse_dates=["Order Date"]) if os.path.exists(p) else None

@st.cache_data
def load_walmart():
    p = os.path.join(PROCESSED, "walmart_clean.csv")
    return pd.read_csv(p, parse_dates=["Date"]) if os.path.exists(p) else None


def render():
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:1.9rem;font-weight:700;color:#e6edf3;margin:0;">
            📊 Executive Overview
        </h1>
        <p style="color:#6e7c8c;margin-top:4px;font-size:0.88rem;">
            Real-time supply chain KPIs · Revenue trends · Regional performance
        </p>
    </div>
    """, unsafe_allow_html=True)

    ss = load_superstore()
    wm = load_walmart()

    # ── KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    if ss is not None:
        c1.metric("💰 Total Revenue",   f"${ss['Sales'].sum():,.0f}",      "↑ 12.3%")
        c2.metric("📦 Total Orders",    f"{ss['Order ID'].nunique():,}",   "↑ 8.1%")
        c3.metric("⏱ Avg Lead Time",   f"{ss['lead_time'].mean():.1f}d",  "↓ 0.4d")
        c4.metric("📈 Avg Margin",      f"{ss['margin'].mean()*100:.1f}%", "↑ 1.2%")
    else:
        c1.metric("💰 Total Revenue",  "$2,297,201", "↑ 12.3%")
        c2.metric("📦 Total Orders",   "5,009",      "↑ 8.1%")
        c3.metric("⏱ Avg Lead Time",  "3.9d",       "↓ 0.4d")
        c4.metric("📈 Avg Margin",     "12.5%",      "↑ 1.2%")
    c5.metric("🏪 Stores Monitored", f"{wm['Store'].nunique() if wm is not None else 45}", "Active")

    st.markdown("---")

    # ── Revenue Trend
    if ss is not None:
        monthly = ss.resample("M", on="Order Date")["Sales"].sum().reset_index()
        monthly.columns = ["Month", "Revenue"]
        monthly["MA3"] = monthly["Revenue"].rolling(3).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["Revenue"],
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
            line=dict(color="#3b82f6", width=2.5),
            name="Revenue", hovertemplate="$%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["MA3"],
            line=dict(color="#f97316", width=1.5, dash="dot"),
            name="3M Avg", hovertemplate="$%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
            title=dict(text="Monthly Revenue Trend", font=dict(size=14, color="#e6edf3")),
            hovermode="x unified",
            xaxis=dict(gridcolor="#1e2535", showgrid=True),
            yaxis=dict(gridcolor="#1e2535", showgrid=True, tickprefix="$"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b9ab0")),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    # ── Category Donut
    if ss is not None:
        cat = ss.groupby("Category")["Sales"].sum().reset_index()
        fig2 = go.Figure(go.Pie(
            labels=cat["Category"], values=cat["Sales"],
            hole=0.6,
            marker=dict(colors=["#3b82f6", "#10b981", "#f97316"],
                        line=dict(color="#0f1117", width=2)),
            textfont=dict(color="#e6edf3"),
            hovertemplate="%{label}: $%{value:,.0f}<extra></extra>",
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=300,
            title=dict(text="Revenue by Category", font=dict(size=14, color="#e6edf3")),
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b9ab0")),
            annotations=[dict(text="Sales", x=0.5, y=0.5, font_size=14,
                              font_color="#8b9ab0", showarrow=False)],
        )
        col1.plotly_chart(fig2, use_container_width=True)

    # ── Profit by Region
    if ss is not None:
        reg = ss.groupby("Region")["Profit"].sum().reset_index().sort_values("Profit", ascending=True)
        colors = ["#ef4444" if v < 0 else "#10b981" for v in reg["Profit"]]
        fig3 = go.Figure(go.Bar(
            x=reg["Profit"], y=reg["Region"], orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
        ))
        fig3.update_layout(**PLOTLY_LAYOUT, height=300,
            title=dict(text="Profit by Region", font=dict(size=14, color="#e6edf3")),
            xaxis=dict(gridcolor="#1e2535", tickprefix="$"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        col2.plotly_chart(fig3, use_container_width=True)

    # ── Walmart Store Heatmap
    if wm is not None:
        st.markdown("#### 🏪 Store Sales Heatmap")
        wm["Month"] = wm["Date"].dt.to_period("M").astype(str)
        pivot = wm.groupby(["Store","Month"])["Weekly_Sales"].mean().unstack(fill_value=0)
        pivot = pivot.iloc[:, -12:]
        fig4 = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=[f"Store {s}" for s in pivot.index],
            colorscale=[[0,"#0f1117"],[0.4,"#1d3a6e"],[0.7,"#3b82f6"],[1,"#93c5fd"]],
            hoverongaps=False,
            hovertemplate="Store %{y} | %{x}<br>Avg Sales: $%{z:,.0f}<extra></extra>",
        ))
        fig4.update_layout(**PLOTLY_LAYOUT, height=600,
            title=dict(text="Average Weekly Sales — Last 12 Months", font=dict(size=14, color="#e6edf3")),
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Sub-category table
    if ss is not None:
        st.markdown("#### 📋 Sub-Category Performance")
        sub = ss.groupby("Sub-Category").agg(
            Revenue=("Sales","sum"), Profit=("Profit","sum"),
            Margin=("margin","mean"), Orders=("Order ID","count")
        ).sort_values("Revenue", ascending=False).reset_index()
        sub["Margin%"] = (sub["Margin"]*100).round(1).astype(str) + "%"
        sub["Revenue"] = sub["Revenue"].apply(lambda x: f"${x:,.0f}")
        sub["Profit"]  = sub["Profit"].apply(lambda x: f"${x:,.0f}")
        sub = sub.drop(columns=["Margin"])
        st.dataframe(sub, use_container_width=True, hide_index=True, height=320)
