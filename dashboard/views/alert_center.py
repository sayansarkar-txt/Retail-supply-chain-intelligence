# dashboard/views/alert_center.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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


def make_alerts(n=20):
    np.random.seed(42)
    n_crit = max(1, n // 5)
    n_high = max(1, n // 4)
    n_med  = n - n_crit - n_high

    probs = np.concatenate([
        np.random.uniform(0.76, 0.97, n_crit),
        np.random.uniform(0.51, 0.74, n_high),
        np.random.uniform(0.31, 0.49, n_med),
    ])
    probs = np.sort(probs)[::-1]

    stores    = np.random.randint(1, 46, size=n)
    weeks     = np.random.randint(1, 5,  size=n)
    forecasts = np.random.randint(60_000, 280_000, size=n)
    drops     = np.random.uniform(5, 48, size=n)

    rows = []
    for i in range(n):
        p = float(probs[i])
        if p >= 0.75:
            sev    = "CRITICAL"
            action = "Contact supplier immediately — emergency reorder required"
            icon   = "🚨"
            days   = 1
        elif p >= 0.50:
            sev    = "HIGH"
            action = "Schedule reorder within 3 days before stockout occurs"
            icon   = "⚠️"
            days   = 3
        else:
            sev    = "MEDIUM"
            action = "Monitor inventory levels — review reorder point this week"
            icon   = "📋"
            days   = 7

        rows.append({
            "icon":         icon,
            "severity":     sev,
            "store_id":     int(stores[i]),
            "week_ahead":   int(weeks[i]),
            "risk_pct":     round(p * 100, 1),
            "forecast":     int(forecasts[i]),
            "drop_pct":     round(float(drops[i]), 1),
            "action":       action,
            "days_to_act":  days,
        })
    return pd.DataFrame(rows)


def render():
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:1.9rem;font-weight:700;color:#e6edf3;margin:0;">
            🔔 Alert Center
        </h1>
        <p style="color:#6e7c8c;margin-top:4px;font-size:0.88rem;">
            Severity-ranked stockout alerts · CRITICAL → HIGH → MEDIUM
        </p>
    </div>
    """, unsafe_allow_html=True)

    alerts = make_alerts(20)

    # ── KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Alerts",  len(alerts))
    c2.metric("🚨 Critical",   (alerts["severity"] == "CRITICAL").sum())
    c3.metric("⚠️ High",       (alerts["severity"] == "HIGH").sum())
    c4.metric("📋 Medium",     (alerts["severity"] == "MEDIUM").sum())

    st.markdown("---")

    # ── Filters
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    sev_filter = fc1.multiselect(
        "Filter by Severity",
        ["CRITICAL","HIGH","MEDIUM"],
        default=["CRITICAL","HIGH","MEDIUM"],
    )
    week_max = fc2.slider("Max Weeks Ahead", 1, 4, 4)
    min_risk = fc3.slider("Min Risk %", 30, 90, 30)

    filtered = alerts[
        (alerts["severity"].isin(sev_filter)) &
        (alerts["week_ahead"] <= week_max) &
        (alerts["risk_pct"] >= min_risk)
    ].reset_index(drop=True)

    st.caption(f"Showing **{len(filtered)}** alerts")
    st.markdown("")

    # ── Alert cards
    SEV_COLORS = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308"}
    SEV_BG     = {
        "CRITICAL": "linear-gradient(135deg,#2d0a0a,#1a0505)",
        "HIGH":     "linear-gradient(135deg,#2d1a0a,#1a0e00)",
        "MEDIUM":   "linear-gradient(135deg,#2d2a0a,#1a1800)",
    }

    for _, row in filtered.iterrows():
        color = SEV_COLORS[row["severity"]]
        bg    = SEV_BG[row["severity"]]
        st.markdown(f"""
        <div style="
            background:{bg};
            border-left:4px solid {color};
            border-radius:0 12px 12px 0;
            padding:1rem 1.25rem;
            margin-bottom:0.6rem;
            box-shadow:0 2px 12px rgba(0,0,0,0.3);
        ">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <span style="color:{color};font-weight:700;font-size:1rem;">
                        {row['icon']} [{row['severity']}] Store {row['store_id']}
                    </span>
                    <span style="
                        background:{color}22;color:{color};
                        font-size:0.7rem;font-weight:600;
                        padding:2px 8px;border-radius:20px;
                        margin-left:0.75rem;
                    ">Act within {row['days_to_act']}d</span>
                </div>
                <div style="text-align:right;">
                    <span style="color:{color};font-size:1.2rem;font-weight:700;">{row['risk_pct']}%</span>
                    <span style="color:#6e7c8c;font-size:0.75rem;margin-left:4px;">risk</span>
                </div>
            </div>
            <div style="margin-top:0.6rem;display:flex;gap:2rem;font-size:0.82rem;color:#8b9ab0;">
                <span>📅 Week +{row['week_ahead']} ahead</span>
                <span>💰 Forecast: <b style="color:#c9d1d9;">${row['forecast']:,}</b></span>
                <span>📉 Drop: <b style="color:{color};">{row['drop_pct']}%</b></span>
            </div>
            <div style="margin-top:0.5rem;font-size:0.83rem;color:#c9d1d9;">
                📋 {row['action']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Alert volume chart
    col_l, col_r = st.columns(2)

    sev_counts = filtered["severity"].value_counts().reset_index()
    sev_counts.columns = ["Severity","Count"]
    colors_bar = [SEV_COLORS.get(s,"#3b82f6") for s in sev_counts["Severity"]]

    fig = go.Figure(go.Bar(
        x=sev_counts["Severity"], y=sev_counts["Count"],
        marker=dict(color=colors_bar, line=dict(width=0)),
        hovertemplate="%{x}: %{y} alerts<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=280,
        title=dict(text="Alert Volume by Severity", font=dict(size=14, color="#e6edf3")),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="#1e2535"),
        showlegend=False,
    )
    col_l.plotly_chart(fig, use_container_width=True)

    # ── Risk gauge
    avg_risk = filtered["risk_pct"].mean() / 100 if len(filtered) > 0 else 0
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(avg_risk * 100, 1),
        delta={"reference": 50, "increasing": {"color": "#ef4444"}, "decreasing": {"color":"#10b981"}},
        gauge={
            "axis":       {"range":[0,100], "tickcolor":"#8b9ab0"},
            "bar":        {"color": "#3b82f6"},
            "bgcolor":    "#1a1f35",
            "bordercolor":"#2a3045",
            "steps": [
                {"range":[0,30],   "color":"#0d2d1a"},
                {"range":[30,50],  "color":"#1a2a0a"},
                {"range":[50,75],  "color":"#2d1a0a"},
                {"range":[75,100], "color":"#2d0a0a"},
            ],
            "threshold":  {"value":75,"line":{"color":"#ef4444","width":2},"thickness":0.75},
        },
        number={"suffix": "%", "font":{"color":"#60a5fa","size":32}},
        title={"text":"Avg Alert Risk Score", "font":{"color":"#8b9ab0","size":13}},
    ))
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c9d1d9"), height=280,
                       margin=dict(l=20,r=20,t=40,b=20))
    col_r.plotly_chart(fig2, use_container_width=True)

    # ── Download
    with st.expander("📄 Export Alert Table"):
        export = filtered.drop(columns=["icon"])
        st.dataframe(export, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download CSV", export.to_csv(index=False).encode(),
            "stockout_alerts.csv", "text/csv",
        )
