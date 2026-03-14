# =============================================================================
# dashboard/app.py — Retail Supply Chain Intelligence Platform
# =============================================================================

import streamlit as st
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0f1117; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12151f 0%, #0d1117 100%);
    border-right: 1px solid #1e2535;
}

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #161b2e 0%, #1a1f35 100%);
    border: 1px solid #2a3045;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: all 0.2s ease;
}
[data-testid="stMetricValue"] {
    color: #60a5fa !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #8b9ab0 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    color: white; border: none; border-radius: 8px;
    padding: 0.5rem 1.5rem; font-weight: 600;
    transition: all 0.2s;
    box-shadow: 0 2px 10px rgba(59,130,246,0.3);
}
.stButton > button:hover {
    box-shadow: 0 4px 20px rgba(59,130,246,0.5);
    transform: translateY(-1px);
}

.stSelectbox > div > div {
    background: #1a1f35; border: 1px solid #2a3045;
    border-radius: 8px; color: #c9d1d9;
}

hr { border-color: #1e2535 !important; margin: 1.5rem 0 !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2a3045; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0.5rem 0.5rem;">
        <div style="font-size:1.4rem;font-weight:700;color:#e6edf3;">📦 Supply Chain</div>
        <div style="font-size:0.75rem;color:#6e7c8c;margin-top:2px;">Intelligence Platform v1.0</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    view = st.radio("", [
        "📊  Executive Overview",
        "📈  Demand Forecast",
        "⚠️  Stockout Risk",
        "🔔  Alert Center",
        "🔍  Product Analysis",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style="padding:0.5rem;font-size:0.75rem;color:#4a5568;line-height:1.8;">
        <div style="color:#8b9ab0;font-weight:600;margin-bottom:0.5rem;">BUILT BY</div>
        <div style="color:#c9d1d9;">Sayan Sarkar</div>
        <div>B.Tech CSE · Data Science</div>
        <div>Pragati Engineering College</div>
    </div>
    """, unsafe_allow_html=True)

if "Overview"  in view: from views.overview         import render
elif "Forecast" in view: from views.demand_forecast  import render
elif "Risk"     in view: from views.stockout_risk    import render
elif "Alert"    in view: from views.alert_center     import render
else:                    from views.product_analysis import render

render()
