# dashboard/views/product_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, sys

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
def load_data():
    p = os.path.join(PROCESSED, "superstore_clean.csv")
    if os.path.exists(p):
        return pd.read_csv(p, parse_dates=["Order Date"])
    return None


def render():
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:1.9rem;font-weight:700;color:#e6edf3;margin:0;">
            🔍 Product Analysis
        </h1>
        <p style="color:#6e7c8c;margin-top:4px;font-size:0.88rem;">
            Margin analysis · discount impact · top & bottom performers
        </p>
    </div>
    """, unsafe_allow_html=True)

    ss = load_data()
    if ss is None:
        st.error("❌ superstore_clean.csv not found in `data/processed/`")
        return

    # ── Filters
    fc1, fc2, fc3 = st.columns(3)
    cats    = ["All"] + sorted(ss["Category"].unique())
    regions = ["All"] + sorted(ss["Region"].unique())
    years   = ["All"] + sorted(ss["year"].unique().astype(str).tolist(), reverse=True)

    sel_cat    = fc1.selectbox("📦 Category", cats)
    sel_region = fc2.selectbox("🌍 Region", regions)
    sel_year   = fc3.selectbox("📅 Year", years)

    f = ss.copy()
    if sel_cat    != "All": f = f[f["Category"] == sel_cat]
    if sel_region != "All": f = f[f["Region"]   == sel_region]
    if sel_year   != "All": f = f[f["year"]      == int(sel_year)]

    # ── KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 Revenue",       f"${f['Sales'].sum():,.0f}")
    k2.metric("📈 Profit",        f"${f['Profit'].sum():,.0f}")
    k3.metric("📊 Avg Margin",    f"{f['margin'].mean()*100:.1f}%")
    k4.metric("📦 Orders",        f"{f['Order ID'].nunique():,}")

    st.markdown("---")

    # ── Sales vs Profit scatter
    sub = f.groupby("Sub-Category").agg(
        Sales=("Sales","sum"), Profit=("Profit","sum"),
        Orders=("Order ID","count"), Margin=("margin","mean"),
    ).reset_index()
    sub["Margin_pct"] = (sub["Margin"] * 100).round(1)

    fig = px.scatter(
        sub, x="Sales", y="Profit",
        size="Orders", color="Margin_pct", text="Sub-Category",
        color_continuous_scale=[[0,"#ef4444"],[0.5,"#eab308"],[1,"#10b981"]],
        labels={"Margin_pct":"Margin %"},
    )
    fig.update_traces(
        textposition="top center",
        textfont=dict(size=10, color="#c9d1d9"),
        marker=dict(line=dict(width=1, color="#0f1117")),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#4a5568", line_width=1)
    fig.add_vline(x=f["Sales"].mean(), line_dash="dot", line_color="#4a5568", line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT, height=440,
        title=dict(text="Sub-Category: Sales vs Profit (size = order volume, color = margin%)",
                   font=dict(size=13, color="#e6edf3")),
        xaxis=dict(title="Total Sales ($)", gridcolor="#1e2535", tickprefix="$"),
        yaxis=dict(title="Total Profit ($)", gridcolor="#1e2535", tickprefix="$"),
        coloraxis_colorbar=dict(tickfont=dict(color="#8b9ab0"), title="Margin%"),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns(2)

    # ── Discount impact
    f = f.copy()
    f["disc_bucket"] = pd.cut(
        f["Discount"],
        bins=[-0.001,0,0.1,0.2,0.4,1.0],
        labels=["0%","1-10%","11-20%","21-40%",">40%"],
    )
    disc_agg = f.groupby("disc_bucket", observed=True).agg(
        Avg_Margin=("margin","mean"), Orders=("Order ID","count"),
    ).reset_index()
    disc_agg["Margin_pct"] = (disc_agg["Avg_Margin"] * 100).round(1)
    disc_colors = ["#10b981" if m >= 0 else "#ef4444" for m in disc_agg["Margin_pct"]]

    fig2 = go.Figure(go.Bar(
        x=disc_agg["disc_bucket"].astype(str), y=disc_agg["Margin_pct"],
        marker=dict(color=disc_colors, line=dict(width=0)),
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        text=disc_agg["Margin_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside", textfont=dict(color="#c9d1d9"),
    ))
    fig2.add_hline(y=0, line_color="#4a5568", line_width=1)
    fig2.update_layout(**PLOTLY_LAYOUT, height=320,
        title=dict(text="Discount Level vs Avg Profit Margin", font=dict(size=13, color="#e6edf3")),
        xaxis=dict(title="Discount Range", gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Avg Margin %", gridcolor="#1e2535"),
    )
    col_l.plotly_chart(fig2, use_container_width=True)

    # ── Top 10 products
    top10 = f.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
    fig3 = go.Figure(go.Bar(
        x=top10["Sales"],
        y=top10["Product Name"],
        orientation="h",
        marker=dict(
            color=top10["Sales"],
            colorscale=[[0,"#1d3a6e"],[1,"#3b82f6"]],
            line=dict(width=0),
        ),
        hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
    ))
    fig3.update_layout(**PLOTLY_LAYOUT, height=320,
        title=dict(text="Top 10 Products by Revenue", font=dict(size=13, color="#e6edf3")),
        xaxis=dict(gridcolor="#1e2535", tickprefix="$"),
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)",
                   tickfont=dict(size=10)),
    )
    col_r.plotly_chart(fig3, use_container_width=True)

    # ── Loss makers
    st.markdown("### ⚠️ Loss-Making Sub-Categories")
    loss = f.groupby("Sub-Category")["Profit"].sum().sort_values().head(8).reset_index()
    fig4 = go.Figure(go.Bar(
        x=loss["Sub-Category"], y=loss["Profit"],
        marker=dict(
            color=["#ef4444" if v < 0 else "#10b981" for v in loss["Profit"]],
            line=dict(width=0),
        ),
        hovertemplate="%{x}<br>Profit: $%{y:,.0f}<extra></extra>",
        text=loss["Profit"].apply(lambda x: f"${x:,.0f}"),
        textposition="outside", textfont=dict(color="#c9d1d9", size=10),
    ))
    fig4.add_hline(y=0, line_color="#4a5568", line_width=1)
    fig4.update_layout(**PLOTLY_LAYOUT, height=320,
        title=dict(text="Bottom Sub-Categories by Total Profit", font=dict(size=13, color="#e6edf3")),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#1e2535", tickprefix="$"),
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ── Detailed table
    with st.expander("📊 Full Sub-Category Table"):
        tbl = f.groupby("Sub-Category").agg(
            Revenue=("Sales","sum"), Profit=("Profit","sum"),
            Margin=("margin","mean"), Orders=("Order ID","count"),
        ).sort_values("Revenue", ascending=False).reset_index()
        tbl["Margin%"] = (tbl["Margin"]*100).round(1).astype(str)+"%"
        tbl["Revenue"] = tbl["Revenue"].apply(lambda x: f"${x:,.0f}")
        tbl["Profit"]  = tbl["Profit"].apply(lambda x: f"${x:,.0f}")
        tbl = tbl.drop(columns=["Margin"])
        st.dataframe(tbl, use_container_width=True, hide_index=True)
