# -*- coding: utf-8 -*-
"""
app.py
======
Smart Electricity Theft Detection System - Professional Dashboard Redesign
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import paths
from predict import predict_single, load_models

# -- Page config ----------------------------------------------------------------
st.set_page_config(
page_title="Smart Electricity Theft Detection",
page_icon="V",
layout="wide",
initial_sidebar_state="expanded",
)

# -- Professional UI Theme (Custom CSS) -----------------------------------------
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

    /* Sidebar Buttons */
    .stButton>button {
        width: 100%;
            border-radius: 8px;
                height: 3em;
                    background-color: #ffffff;
                        color: #1e293b;
                            border: 1px solid #e2e8f0;
                                font-weight: 600;
                                }

                                .stButton>button:hover {
                                    border-color: #3b82f6;
                                        color: #3b82f6;
                                        }

                                        /* Metric Colors */
                                        [data-testid="stMetricValue"] {
                                            font-size: 1.8rem !important;
                                                font-weight: 800 !important;
                                                }

                                                /* Header indicator */
                                                .main .block-container::before {
                                                    content: "Professional Grid Analysis";
                                                        position: absolute;
                                                            top: -40px;
                                                                left: 0;
                                                                    font-size: 0.8rem;
                                                                        font-weight: 600;
                                                                            text-transform: uppercase;
                                                                                letter-spacing: 0.1em;
                                                                                    color: #64748b;
                                                                                    }
                                                                                    </style>
                                                                                    """, unsafe_allow_html=True)


# -- State management & Helpers -------------------------------------------------

def init_session_state():
i                f "data_loaded" not in st.session_state:
st.session_state.data_loaded = False
if "current_client" not in st.session_state:
st.session_state.current_client = None
if         "model_type" not in st.session_state:
st.session_state.model_type = "isolation_forest"
if "readings" not in st.session_state:
# Default: some realistic values
st.session_state.readings = [
1.2, 1.0, 0.9, 0.8, 0.8, 1.0, 1.8, 3.2, 3.5, 3.0, 2.8, 2.9,
3.0, 2.9, 2.8, 3.1, 3.8, 4.5, 4.8, 4.5, 3.9, 3.1, 2.2, 1.5
]

def load_system_data():
"""Load the processed dataset and models."""
try:
# Ensure directories exist
paths.MODEL_DIR.mkdir(parents=True, exist_ok=True)
paths.SCREENSHOTS.mkdir(parents=True, exist_ok=True)

# Load models (this will raise FileNotFoundError if missing)
load_models()

# Load processed data if available
if paths.PROC_CSV.exists():
df = pd.read_csv(paths.PROC_CSV, index_col=0, parse_dates=True)
st.session_state.data_loaded = True
return df
return None
except Exception as e:
st.error(f"System initialization error: {e}")
return None

def run_prediction():
"""Execute prediction based on current session readings."""
try:

res = predict_single(st.session_state.readings, st.session_state.model_type)
return res
except Exception as e:
st.error(f"Prediction failed: {e}")
return None


# -- Main Layout ----------------------------------------------------------------

def main():
init_session_state()
df_raw = load_system_data()

# Sidebar Header
with st.sidebar:
st.markdown("<h1 style='margin-bottom:0;'>Grid Guardian</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748b; font-weight:500; margin-bottom:2rem;'>Model-Driven Anomaly Detection</p>", unsafe_allow_html=True)

st.markdown("### Analysis Configuration")
model_choice = st.selectbox(
"Detection Model",
["isolation_forest", "random_forest"],
index=0 if st.session_state.model_type == "isolation_forest" else 1,
help="Choose the algorithm for anomaly scoring."
)
st.session_state.model_type = model_choice

st.markdown("---")
st.markdown("### Profile Presets")
col1, col2 = st.columns(2)

if col1.button("Residential", help="Standard household profile"):
st.session_state.readings = [0.5,0.4,0.3,0.3,0.4,0.6,1.2,2.0,1.8,1.5,1.4,1.6,1.8,1.7,1.6,1.8,2.5,3.5,3.8,3.2,2.5,1.8,1.2,0.7]
st.rerun()

if col2.button("Industrial", help="High-load heavy industrial"):
st.session_state.readings = [6.5,6.3,6.1,6.0,6.2,6.8,7.2,8.0,9.0,9.5,9.8,9.6,9.3,9.2,9.4,9.1,8.8,8.2,7.5,7.0,6.9,6.7,6.5,6.4]
st.rerun()

if st.button("Reset", help="Restore default values"):
st.session_state.readings = [3.0]*24
st.rerun()

st.markdown("---")
st.markdown("### System Status")
if st.session_state.data_loaded:
st.success("Historical Engine: ONLINE")
else:
st.warning("Historical Engine: FALLBACK")

st.info(f"Active Model: {st.session_state.model_type.replace('_',' ').title()}")

# -- Main Dashboard Area ----------------------------------------------------

st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>PowerGrid Intelligence Dashboard</h1>", unsafe_allow_html=True)

# Top Row: Metrics & Key Findings
m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
st.metric("Total daily load", f"{sum(st.session_state.readings):.1f} kWh", "+2.4%")
st.markdown("</div>", unsafe_allow_html=True)

with m_col2:
st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
peak_val = max(st.session_state.readings)
peak_hour = st.session_state.readings.index(peak_val)
st.metric("Peak Demand", f"{peak_val:.1f} kWh", f"at {peak_hour:02d}:00")
st.markdown("</div>", unsafe_allow_html=True)

with m_col3:
st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
# Run prediction
prediction = run_prediction()
if prediction:
label = prediction['label']
code = prediction['code']
color = "#ef4444" if code == 1 else ("#f59e0b" if code == 2 else "#10b981")
st.markdown(f"<p style='color:#64748b; font-size:0.8rem; font-weight:600; margin-bottom:0;'>SECURITY SCAN RESULT</p>", unsafe_allow_html=True)
st.markdown(f"<h2 style='color:{color}; margin-top:0;'>{label}</h2>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Second Row: Graph & Controls
st.markdown("### Profile Analysis: 24-Hour Consumer Profile")

g_col1, g_col2 = st.columns([3, 1])

with g_col1:
st.markdown("<div class='ui-card'>", unsafe_allow_html=True)

# Hourly labels for the chart
h_labels = prediction['hour_labels'] if prediction else ["Normal"]*24
h_colors = ["#ef4444" if l=="Suspicious" else ("#f59e0b" if l=="Slightly Unusual" else "#3b82f6") for l in h_labels]

fig = go.Figure()
fig.add_trace(go.Bar(
x=[f"{h:02d}:00" for h in range(24)],
y=st.session_state.readings,
marker_color=h_colors,
name="Consumption",
hovertemplate="Hour: %{x}<br>Usage: %{y:.2f} kWh<extra></extra>"
))
fig.update_layout(
margin=dict(l=0, r=0, t=20, b=0),
height=400,
plot_bgcolor="rgba(0,0,0,0)",
paper_bgcolor="rgba(0,0,0,0)",
xaxis=dict(gridcolor="#f1f5f9", tickfont=dict(color="#64748b")),
yaxis=dict(gridcolor="#f1f5f9", title="Usage (kWh)", tickfont=dict(color="#64748b")),
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Bar colors represent individual hourly anomaly scores (Blue=Normal, Yellow=Unusual, Red=Suspicious)")
st.markdown("</div>", unsafe_allow_html=True)

with g_col2:
st.markdown("<div class='ui-card' style='height: 485px;'>", unsafe_allow_html=True)
st.markdown("#### House Profile Adjuster")
st.markdown("<p style='font-size:0.8rem; color:#64748b;'>Manually adjust hourly consumption (kWh)</p>", unsafe_allow_html=True)

with st.expander("Expand Hourly Editor", expanded=True):

new_readings = []
for h in range(24):
val = st.number_input(f"{h:02d}:00", value=float(st.session_state.readings[h]), step=0.1, format="%.1f", key=f"h_input_{h}")
new_readings.append(val)

if new_readings != st.session_state.readings:
st.session_state.readings = new_readings
st.rerun()
st.markdown("</div>", unsafe_allow_html=True)


# Third Row: Historical Discovery
st.markdown("### Enterprise Historical Discovery")

if not st.session_state.data_loaded:
st.info("TIP: Connect a historical dataset to enable deep-dive analysis of individual client behaviors.")
# Offer to generate fallback data in UI
if st.button("Generate Intelligence Fallback"):
with st.spinner("Generating synthetic profiles..."):
from fallback import generate_fallback_data
generate_fallback_data()
st.success("Synthetic logic activated. Rerunning...")
time.sleep(1)
st.rerun()
else:
# Data discovery interface
st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
h_col1, h_col2 = st.columns([1, 2])

with h_col1:
st.subheader("Client Registry")
all_clients = df_raw.columns.tolist() if df_raw is not None else []
search = st.text_input("Filter IDs", placeholder="e.g. MT_005")
filtered = [c for c in all_clients if search.upper() in c.upper()] if search else all_clients[:20]

client_id = st.selectbox("Select Target ID", filtered)
st.session_state.current_client = client_id

if df_raw is not None and client_id:
c_data = df_raw[client_id]
st.write(f"**Records:** {len(c_data)}")
st.write(f"**Avg Consumption:** {c_data.mean():.2f} kWh")

if st.button("Load to Profile Adjuster"):
# Take the most recent 24-hour window
recent = c_data.tail(24).tolist()
if len(recent) == 24:
                    recent = c_data.tail(24).tolist()
                    if len(recent) == 24:
                                                st.session_state.readings = recent
                                                st.rerun()

        with h_col2:
                        if df_raw is not None and st.session_state.current_client:
                                            st.subheader(f"Timeline Analysis: {st.session_state.current_client}")
                                            fig_time = px.line(df_raw[st.session_state.current_client].reset_index(), x='datetime', y=st.session_state.current_client)
                                            fig_time.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                                            st.plotly_chart(fig_time, use_container_width=True)

                    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
f_col1, f_col2 = st.columns(2)
f_col1.markdown("<p style='color:#64748b; font-size:0.8rem;'>System V1.5.2 | Scaled Environment Ready</p>", unsafe_allow_html=True)
f_col2.markdown("<p style='color:#64748b; font-size:0.8rem; text-align:right;'>Powered by scikit-learn & streamlit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
main()

                        st.session_state.readings = recent
                        st.rerun()

with h_col2:
if df_raw is not None and st.session_state.current_client:
                    st.subheader(f"Timeline Analysis: {st.session_state.current_client}")
                    fig_time = px.line(df_raw[st.session_state.current_client].reset_index(), x='datetime', y=st.session_state.current_client)
                    fig_time.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_time, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
f_col1, f_col2 = st.columns(2)
f_col1.markdown("<p style='color:#64748b; font-size:0.8rem;'>System V1.5.2 | Scaled Environment Ready</p>", unsafe_allow_html=True)
f_col2.markdown("<p style='color:#64748b; font-size:0.8rem; text-align:right;'>Powered by scikit-learn & streamlit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
main()
