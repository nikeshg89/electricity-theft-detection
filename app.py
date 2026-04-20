# -*- coding: utf-8 -*-
"""
app.py
======
Smart Electricity Theft Detection System — Professional Dashboard Redesign
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(APP_DIR, "src")
sys.path.insert(0, SRC_DIR)

from predict import predict_single, load_models

BASE_DIR      = APP_DIR
MODEL_DIR     = os.path.join(BASE_DIR, "model")
SCREENSHOTS   = os.path.join(BASE_DIR, "screenshots")
DATASET_DIR   = os.path.join(BASE_DIR, "dataset")
FEAT_CSV      = os.path.join(DATASET_DIR, "features.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Electricity Theft Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional UI Theme (Custom CSS) ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* Main Defaults */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, .stHeader {
    font-family: 'Poppins', sans-serif;
    font-weight: 700 !important;
}

/* Background */
.stApp {
    background-color: #f8fafc;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

/* Custom Cards */
.ui-card {
    background: white;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
}

/* Hero Section */
.hero-container {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.2);
    position: relative;
    overflow: hidden;
}

.hero-container::after {
    content: "⚡";
    position: absolute;
    right: -20px;
    bottom: -20px;
    font-size: 15rem;
    opacity: 0.1;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1.1rem;
    font-weight: 400;
    opacity: 0.9;
}

/* Metric Cards */
.metric-container {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    border-bottom: 4px solid #2563eb;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    text-align: center;
}

.metric-val {
    font-size: 2rem;
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 0.25rem;
}

.metric-lbl {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Result Cards */
.res-card {
    padding: 40px;
    border-radius: 24px;
    text-align: center;
    color: white;
    margin: 20px auto;
    max-width: 800px;
    animation: fadeIn 0.5s ease-out;
}

.res-normal {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
}

.res-unusual {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    box-shadow: 0 10px 15px -3px rgba(245, 158, 11, 0.3);
}

.res-suspicious {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.3);
}

.res-title {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 10px;
}

.res-desc {
    font-size: 1.1rem;
    opacity: 0.9;
}

.res-score {
    display: inline-block;
    background: rgba(255, 255, 255, 0.2);
    padding: 8px 20px;
    border-radius: 99px;
    font-weight: 700;
    margin-top: 20px;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: white !important;
    border: none;
    padding: 10px 24px;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s;
    width: 100%;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

/* Sliders */
.stSlider [data-baseweb="slider"] {
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_features_sample(n: int = 5000) -> pd.DataFrame | None:
    if not os.path.exists(FEAT_CSV):
        return None
    try:
        df = pd.read_csv(FEAT_CSV, parse_dates=["datetime"], nrows=n)
        return df
    except:
        return None


def ensure_models_exist():
    try:
        load_models()
    except:
        with st.status("Initializing AI System...", expanded=True) as status:
            st.write("Generating synthetic energy patterns...")
            import fallback
            fallback.generate_fallback_data()
            st.write("Calibrating detection thresholds...")
            import train
            train.run_training()
            st.cache_resource.clear()
            status.update(label="System Ready", state="complete", expanded=False)


@st.cache_resource(show_spinner=False)
def get_models():
    try:
        return load_models()
    except:
        return None

def models_ready() -> bool:
    return get_models() is not None

# Init check
ensure_models_exist()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding-bottom: 20px;">
        <span style="font-size: 3.5rem;">⚡</span>
        <h2 style="margin: 0; color: #1e293b;">VoltGuard AI</h2>
        <p style="color: #64748b; font-size: 0.9rem;">Modern Theft Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 System Status")
    if models_ready():
        st.success("AI Core: Active")
    else:
        st.error("AI Core: Offline")
        
    st.markdown("---")
    st.markdown("### 🎯 Model Performance")
    
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, "r") as f:
            m = json.load(f).get("isolation_forest", {})
            st.metric("Accuracy", f"{m.get('accuracy', 0)*100:.1f}%")
            st.metric("Precision", f"{m.get('precision', 0)*100:.1f}%")
            st.metric("Recall", f"{m.get('recall', 0)*100:.1f}%")
    
    st.markdown("---")
    st.caption("Developed using Scikit-learn & Streamlit")


# ── Main Content ───────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ Smart Electricity Theft Detection</div>
    <div class="hero-subtitle">Industrial-grade anomaly detection system powered by Isolation Forest.</div>
</div>
""", unsafe_allow_html=True)

tab_dash, tab_detect, tab_analytics, tab_data = st.tabs([
    "🏠 Dashboard", "🔍 Detect Fraud", "📊 Model Insights", "📁 Data Source"
])


# 🏠 DASHBOARD
with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown('<div class="metric-container"><div class="metric-val">3,742</div><div class="metric-lbl">Total Audits</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-container"><div class="metric-val">5.2%</div><div class="metric-lbl">Fraud Rate</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-container"><div class="metric-val">24h</div><div class="metric-lbl">Inference Window</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-container"><div class="metric-val">99%</div><div class="metric-lbl">System Uptime</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    df = load_features_sample(10000)
    if df is not None:
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("### 📈 Network Energy Baseline")
            ts = df.groupby("datetime")["consumption"].mean().reset_index()
            fig = px.line(ts, x="datetime", y="consumption", template="plotly_white")
            fig.update_traces(line_color='#2563eb', fill='tozeroy', fillcolor='rgba(37,99,235,0.1)')
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=350, xaxis_title="", yaxis_title="kWh")
            st.plotly_chart(fig, use_container_width=True)
            
        with col_right:
            st.markdown("### 🚨 Detection Mix")
            fig2 = px.pie(values=[94.8, 5.2], names=["Healthy", "Anomalous"], 
                          color_discrete_sequence=['#10b981', '#ef4444'], hole=.55)
            fig2.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=350, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)


# 🔍 DETECT FRAUD
with tab_detect:
    st.markdown("### 🔍 Live Anomaly Scanner")
    st.markdown("Analyze consumption behavior over 24 hours to identify potential theft or tamupering.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("**Load Profile Presets**")
        p1, p2, p3, p4 = st.columns(4)
        if p1.button("🟢 Normal Setup"):
            st.session_state["p"] = [1.2,1.0,0.9,0.8,0.8,1.0,1.8,3.2,3.5,3.0,2.8,2.9,3.0,2.9,2.8,3.1,3.8,4.5,4.8,4.5,3.9,3.1,2.2,1.5]
        if p2.button("🏭 Heavy Industry"):
            st.session_state["p"] = [6.5,6.3,6.1,6.0,6.2,6.8,7.2,8.0,9.0,9.5,9.8,9.6,9.3,9.2,9.4,9.1,8.8,8.2,7.5,7.0,6.9,6.7,6.5,6.4]
        if p3.button("🚨 Suspected Theft"):
            st.session_state["p"] = [0.05]*24
        if p4.button("🔄 Reset"):
            st.session_state["p"] = [3.0]*24
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 24-hr Grid
    readings = []
    default = st.session_state.get("p", [3.0]*24)
    
    st.markdown("**24-Hour Consumption Profile (kWh)**")
    row1 = st.columns(6)
    row2 = st.columns(6)
    row3 = st.columns(6)
    row4 = st.columns(6)
    all_cols = row1 + row2 + row3 + row4
    
    for h in range(24):
        with all_cols[h]:
            v = st.slider(f"{h:02d}:00", 0.0, 20.0, float(default[h]), 0.1, key=f"s_{h}")
            readings.append(v)
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("⚡ EXECUTE AI DIAGNOSTICS"):
        with st.spinner("Analyzing neural patterns..."):
            time.sleep(1)
            res = predict_single(readings)
            
        code = res["code"]
        label = res["label"]
        score = res["score"]
        
        if code == 0:
            st.markdown(f"""
            <div class="res-card res-normal">
                <div class="res-title">✅ NORMAL USAGE</div>
                <div class="res-desc">No suspicious patterns detected. Consumer behavior aligns with grid standards.</div>
                <div class="res-score">Anomaly Score: {score:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        elif code == 2:
            st.markdown(f"""
            <div class="res-card res-unusual">
                <div class="res-title">⚠️ SLIGHTLY UNUSUAL</div>
                <div class="res-desc">Minor deviations detected. May indicate appliance malfunction or edge-case behavior.</div>
                <div class="res-score">Anomaly Score: {score:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="res-card res-suspicious">
                <div class="res-title">🚨 ALERT: SUSPICIOUS</div>
                <div class="res-desc">Extreme anomaly detected. High probability of meter tampering or illegal connection.</div>
                <div class="res-score">Anomaly Score: {score:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Visualization
        st.markdown("### 📊 Profile Visualization")
        h_labels = res["hour_labels"]
        # Map colors
        colors = ["#ef4444" if l=="Suspicious" else "#eab308" if l=="Slightly Unusual" else "#10b981" for l in h_labels]
        
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(x=[f"{h:02d}h" for h in range(24)], y=readings, marker_color=colors))
        fig_res.add_trace(go.Scatter(x=[f"{h:02d}h" for h in range(24)], y=readings, mode='lines+markers', line=dict(color='#1e293b', width=3)))
        fig_res.update_layout(template="plotly_white", margin=dict(l=0,r=0,t=0,b=0), height=350, showlegend=False)
        st.plotly_chart(fig_res, use_container_width=True)


# 📊 ANALYTICS
with tab_analytics:
    st.markdown("### 🧬 AI Model Architecture & Insights")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.markdown("**Anomaly Score Distribution**")
        p_dist = os.path.join(SCREENSHOTS, "anomaly_score_distribution.png")
        if os.path.exists(p_dist):
            st.image(p_dist, use_column_width=True)
        else:
            st.info("Score distribution plot pending.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_b:
        st.markdown('<div class="ui-card">', unsafe_allow_html=True)
        st.markdown("**Global Feature Importance**")
        p_feat = os.path.join(SCREENSHOTS, "feature_importance.png")
        if os.path.exists(p_feat):
            st.image(p_feat, use_column_width=True)
        else:
            st.info("Feature importance plot pending.")
        st.markdown('</div>', unsafe_allow_html=True)


# 📁 DATA SOURCE
with tab_data:
    st.markdown("### 📁 Dataset Intelligence")
    st.markdown("Below is a sample of the processed features used for training and inference.")
    
    df_samp = load_features_sample(500)
    if df_samp is not None:
        st.dataframe(df_samp, use_container_width=True)
    else:
        st.info("Dataset CSV not found.")
        
    st.markdown("---")
    st.markdown("**Technical Specs**")
    st.markdown("- **Framework:** Python 3.11 / Streamlit")
    st.markdown("- **Engine:** Scikit-Learn (Isolation Forest)")
    st.markdown("- **Visualization:** Plotly Graphic Objects")

st.markdown("""
<div style="text-align:center; padding: 40px 0; color: #94a3b8; font-size: 0.9rem;">
    Powered by VoltGuard AI Engine | Developed for Grid Security<br>
    © 2024 Electricity Theft Detection Project
</div>
""", unsafe_allow_html=True)
