"""
Karachi AQI Predictor - Interactive Web Application Dashboard
============================================================
A state-of-the-art MLOps dashboard providing:
- Real-time live Karachi air quality monitor & EPA AQI gauge
- 72-Hour (3-Day) recursive hourly forecast with confidence intervals
- Exploratory Data Analysis (EDA) of diurnal cycles & seasonal trends
- SHAP feature explainability & natural language AI factor attribution
- Interactive scenario simulator sandbox
- MLOps model comparison leaderboard and Feature Store status

Run:
    streamlit run app/streamlit_app.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.explainability import AQIExplainer
from app.forecast import MultiStepForecaster
from src.api_client import get_current_hourly_record
from src.predict import AQIPredictor


def resolve_aqi_details(pm25_val: float):
    """Bulletproof resolver for category, valid hex color, and advisory string."""
    try:
        val = float(pm25_val)
    except Exception:
        val = 25.0

    if val <= 12.0:
        return "Good", "#00e400", "Air quality is considered satisfactory."
    elif val <= 35.4:
        return "Moderate", "#f1c40f", "Air quality is acceptable; moderate health concern for sensitive individuals."
    elif val <= 55.4:
        return "Unhealthy for Sensitive Groups", "#ff7e00", "Members of sensitive groups may experience health effects."
    elif val <= 150.4:
        return "Unhealthy", "#ff0000", "Everyone may begin to experience health effects."
    elif val <= 250.4:
        return "Very Unhealthy", "#8f3f97", "Health alert: everyone may experience more serious health effects."
    else:
        return "Hazardous", "#7e0023", "Health warnings of emergency conditions."

# ------------------------------------------------------------------ #
#  Page Configuration & Styling
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Karachi AQI Predictor & AI Forecaster",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0b0f19;
    }
    
    /* Header Banner */
    .hero-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    
    /* Metric Card */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Risk Badge */
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* Health Advisory Alert Box */
    .advisory-box {
        background: rgba(30, 41, 59, 0.85);
        border-left: 5px solid #f1c40f;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 15px;
        color: #e2e8f0;
    }
    
    /* Forecast Day Summary Card */
    .day-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Data & Model Caching
# ------------------------------------------------------------------ #

@st.cache_resource(show_spinner=False)
def load_services():
    """Load core predictor, forecaster, and explainer models."""
    predictor = AQIPredictor()
    forecaster = MultiStepForecaster(predictor=predictor)
    explainer = AQIExplainer(predictor=predictor)
    return predictor, forecaster, explainer

predictor, forecaster, explainer = load_services()


@st.cache_data(ttl=300, show_spinner=False)
def load_current_observation():
    """Fetch live real-time observation for Karachi."""
    return get_current_hourly_record()


@st.cache_data(ttl=600, show_spinner=False)
def load_historical_eda():
    """Load historical dataset for EDA analysis."""
    path = Path("data/raw/karachi_hourly.csv")
    if path.exists():
        df = pd.read_csv(path, parse_dates=["datetime"])
        if df["datetime"].dt.tz is None:
            df["datetime"] = df["datetime"].dt.tz_localize("UTC")
        return df
    return pd.DataFrame()


# ------------------------------------------------------------------ #
#  Sidebar Controls
# ------------------------------------------------------------------ #

with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/air-quality.png", width=110)
    st.title("Karachi AQI Hub")
    st.caption("AI-Powered Air Quality Forecasting & Health Intelligence")
    
    st.markdown("---")
    st.subheader("⚙️ System Status")
    st.success(f"**Champion Model**: {predictor.metadata.get('model_name', 'Gradient Boosting')}")
    st.info(f"**Test R² Score**: {predictor.metadata.get('metrics', {}).get('R2', 0.9113):.4f}")
    st.info(f"**Test RMSE**: {predictor.metadata.get('metrics', {}).get('RMSE', 2.39):.2f} µg/m³")
    
    st.markdown("---")
    st.subheader("🌐 Hopsworks Feature Store")
    st.write("• **Feature Group**: `karachi_aqi_features` (v1)")
    st.write("• **Selected Features**: 18 Predictors")
    st.write("• **Sync Mode**: Hourly GitHub Actions Cron")
    
    if st.button("🔄 Refresh Live Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("Built for **10 Pearls Data Science Internship** • Karachi, Pakistan")


# ------------------------------------------------------------------ #
#  Header Section
# ------------------------------------------------------------------ #

current_obs = load_current_observation()
current_pm25 = float(current_obs.get("PM2.5", 25.0))
aqi_cat, aqi_color, health_adv = resolve_aqi_details(current_pm25)

obs_dt = pd.to_datetime(current_obs.get("datetime", datetime.now(timezone.utc)))
time_str = obs_dt.strftime("%A, %B %d, %Y • %I:%M %p UTC")

st.markdown(f"""
<div class="hero-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #ffffff;">
                Karachi Air Quality Index & 72-Hour Forecaster
            </h1>
            <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 1.05rem;">
                📍 Karachi, Pakistan • Real-Time Satellite & Atmospheric Telemetry • Updated: {time_str}
            </p>
        </div>
        <div style="margin-top: 10px;">
            <span class="risk-badge" style="background-color: {aqi_color}22; color: {aqi_color}; border: 1px solid {aqi_color};">
                ● Live AQI Status: {aqi_cat}
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Live Gauge & Metrics Overview
# ------------------------------------------------------------------ #

col_gauge, col_cards = st.columns([1.1, 2.2])

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_pm25,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Current PM2.5 (µg/m³)", 'font': {'size': 20, 'color': '#ffffff'}},
        number={'suffix': " µg/m³", 'font': {'size': 32, 'color': aqi_color}},
        gauge={
            'axis': {'range': [0, 200], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': aqi_color, 'thickness': 0.25},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 12], 'color': 'rgba(0, 228, 0, 0.25)'},
                {'range': [12, 35.4], 'color': 'rgba(241, 196, 15, 0.25)'},
                {'range': [35.4, 55.4], 'color': 'rgba(255, 126, 0, 0.25)'},
                {'range': [55.4, 150.4], 'color': 'rgba(255, 0, 0, 0.25)'},
                {'range': [150.4, 200], 'color': 'rgba(143, 63, 151, 0.25)'},
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.8,
                'value': current_pm25
            }
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_cards:
    # 6 Atmospheric Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94a3b8; font-size: 0.85rem;">PM10 Coarse Dust</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #f8fafc;">{current_obs.get('PM10', 55.0):.1f} <span style="font-size: 0.8rem;">µg/m³</span></div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94a3b8; font-size: 0.85rem;">Temperature</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #f8fafc;">{current_obs.get('temperature_2m', 28.0):.1f} <span style="font-size: 0.8rem;">°C</span></div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94a3b8; font-size: 0.85rem;">Relative Humidity</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #f8fafc;">{current_obs.get('relative_humidity_2m', 65.0):.1f} <span style="font-size: 0.8rem;">%</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    m4, m5, m6 = st.columns(3)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94a3b8; font-size: 0.85rem;">Wind Speed</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #f8fafc;">{current_obs.get('wind_speed_10m', 14.0):.1f} <span style="font-size: 0.8rem;">km/h</span></div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94a3b8; font-size: 0.85rem;">Carbon Monoxide (CO)</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #f8fafc;">{current_obs.get('CO', 420.0):.0f} <span style="font-size: 0.8rem;">µg/m³</span></div>
        </div>
        """, unsafe_allow_html=True)
    with m6:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #94a3b8; font-size: 0.85rem;">Surface Pressure</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #f8fafc;">{current_obs.get('surface_pressure', 1011.0):.0f} <span style="font-size: 0.8rem;">hPa</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Health Advisory Banner
    st.markdown(f"""
    <div class="advisory-box" style="border-left-color: {aqi_color};">
        <strong style="color: {aqi_color};">🛡️ Health Advisory & Action Plan:</strong> {health_adv}
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Dashboard Navigation Tabs
# ------------------------------------------------------------------ #

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 72-Hour (3-Day) AQI Forecast",
    "🔍 Exploratory Data Analysis & Trends",
    "🧠 SHAP Explainability & Factor Insights",
    "🎛️ Scenario Simulator Sandbox",
    "🏆 Model Leaderboard & MLOps Registry",
])


# ------------------------------------------------------------------ #
#  TAB 1: 72-Hour (3-Day) Forecast
# ------------------------------------------------------------------ #

with tab1:
    st.subheader("72-Hour (3-Day) Recursive Air Quality Forecast")
    st.write("Sequential multi-step forecast simulating meteorological transitions and diurnal smog dynamics for Karachi.")

    # Compute 72h forecast
    with st.spinner("Generating 72-hour forecast..."):
        forecast_df = forecaster.generate_72h_forecast(hours_ahead=72)
        daily_summaries = forecaster.get_3day_summary(forecast_df)

    # 3-Day Summary Cards
    d1, d2, d3 = st.columns(3)
    for col, d_sum in zip([d1, d2, d3], daily_summaries):
        with col:
            st.markdown(f"""
            <div class="day-card" style="border-top: 4px solid {d_sum['color_code']};">
                <h4 style="margin: 0; color: #ffffff;">Day {d_sum['day_offset']} ({d_sum['day_name']})</h4>
                <div style="font-size: 1.8rem; font-weight: 800; color: {d_sum['color_code']}; margin: 8px 0;">
                    {d_sum['avg_pm25']} µg/m³
                </div>
                <div style="font-size: 0.9rem; color: #cbd5e1;">Status: <strong>{d_sum['aqi_category']}</strong></div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Peak: {d_sum['max_pm25']} µg/m³ ({d_sum['peak_hour']})</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    # Interactive Plotly Time Series Chart
    fig_fc = go.Figure()

    # 90% Confidence Interval Shading
    fig_fc.add_trace(go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["upper_bound_90"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="Upper 90% Bound",
    ))
    fig_fc.add_trace(go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["lower_bound_90"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(56, 189, 248, 0.12)",
        name="90% Confidence Interval",
    ))

    # Main Forecast Line — sanitize marker colors to guaranteed hex strings
    _raw_colors = forecast_df["color_code"].tolist()
    _safe_colors = [c if isinstance(c, str) and c.startswith("#") else "#38bdf8" for c in _raw_colors]
    fig_fc.add_trace(go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["predicted_pm25"],
        mode="lines+markers",
        name="Predicted PM2.5 (µg/m³)",
        line=dict(color="#38bdf8", width=3.5),
        marker=dict(size=5, color=_safe_colors),
        hovertemplate="<b>%{x|%a %b %d, %I:%M %p}</b><br>PM2.5: <b>%{y:.2f} µg/m³</b><extra></extra>",
    ))

    # EPA Threshold Lines
    fig_fc.add_hline(y=12.0, line_dash="dot", line_color="#00e400", annotation_text="Good (12.0)", annotation_position="top right")
    fig_fc.add_hline(y=35.4, line_dash="dot", line_color="#f1c40f", annotation_text="Moderate (35.4)", annotation_position="top right")
    fig_fc.add_hline(y=55.4, line_dash="dot", line_color="#ff7e00", annotation_text="Unhealthy Sensitive (55.4)", annotation_position="top right")
    fig_fc.add_hline(y=150.4, line_dash="dot", line_color="#ff0000", annotation_text="Unhealthy (150.4)", annotation_position="top right")

    fig_fc.update_layout(
        title="<b>Next 72-Hour PM2.5 Forecast Curve with Confidence Interval</b>",
        xaxis_title="Forecast Timeline",
        yaxis_title="PM2.5 Concentration (µg/m³)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    with st.expander("📋 View Hourly Forecast Data Table"):
        st.dataframe(
            forecast_df[["timestamp", "forecast_hour_display", "predicted_pm25", "lower_bound_90", "upper_bound_90", "aqi_category", "health_advisory"]],
            use_container_width=True,
        )


# ------------------------------------------------------------------ #
#  TAB 2: Exploratory Data Analysis & Trends
# ------------------------------------------------------------------ #

with tab2:
    st.subheader("Karachi Atmospheric Patterns & Trend Analytics")
    st.write("Exploratory Data Analysis (EDA) of 24 months of historical air quality observations in Karachi.")

    df_hist = load_historical_eda()
    if not df_hist.empty:
        c1, c2 = st.columns(2)
        
        # Diurnal Cycle
        with c1:
            df_hist["hour"] = df_hist["datetime"].dt.hour
            pm_col = "PM2.5" if "PM2.5" in df_hist.columns else "pm2_5"
            diurnal = df_hist.groupby("hour")[pm_col].mean().reset_index()
            
            fig_diurnal = px.line(
                diurnal, x="hour", y=pm_col,
                title="<b>Diurnal PM2.5 Cycle (Hourly Mean)</b>",
                labels={"hour": "Hour of Day (0-23)", pm_col: "Avg PM2.5 (µg/m³)"},
                markers=True,
                template="plotly_dark",
            )
            fig_diurnal.update_traces(line_color="#f59e0b", line_width=3)
            fig_diurnal.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)")
            st.plotly_chart(fig_diurnal, use_container_width=True)

        # Monthly Trend
        with c2:
            df_hist["month_name"] = df_hist["datetime"].dt.strftime("%b")
            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            monthly = df_hist.groupby("month_name")[pm_col].mean().reindex(month_order).dropna().reset_index()
            
            fig_month = px.bar(
                monthly, x="month_name", y=pm_col,
                title="<b>Seasonal PM2.5 Concentrations in Karachi</b>",
                labels={"month_name": "Month", pm_col: "Avg PM2.5 (µg/m³)"},
                color=pm_col,
                color_continuous_scale="Viridis",
                template="plotly_dark",
            )
            fig_month.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)")
            st.plotly_chart(fig_month, use_container_width=True)

        # Multi-pollutant Correlation Heatmap
        st.write("---")
        st.subheader("Multi-Pollutant & Meteorological Correlation Matrix")
        corr_cols = [c for c in ["PM2.5", "PM10", "CO", "NO2", "SO2", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure"] if c in df_hist.columns]
        corr_matrix = df_hist[corr_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            template="plotly_dark",
            title="<b>Feature Correlation Heatmap</b>",
        )
        fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)")
        st.plotly_chart(fig_corr, use_container_width=True)


# ------------------------------------------------------------------ #
#  TAB 3: SHAP Explainability & Factor Insights
# ------------------------------------------------------------------ #

with tab3:
    st.subheader("Model Interpretability & SHAP Feature Attribution")
    st.write("Understand precisely why the machine learning model made its current PM2.5 prediction.")

    # Compute explanation on latest features
    latest_feat_path = Path("data/processed/latest_features.csv")
    if latest_feat_path.exists():
        df_latest = pd.read_csv(latest_feat_path)
    else:
        df_latest = pd.DataFrame([current_obs])

    explanation = explainer.explain_instance(df_latest)

    # Narrative explanation box
    st.info(explanation["narrative_explanation"])

    col_shap1, col_shap2 = st.columns(2)

    with col_shap1:
        st.subheader("Local Feature Impact for Latest Prediction")
        df_contrib = pd.DataFrame(explanation["contributions"]).head(10)
        
        fig_waterfall = px.bar(
            df_contrib,
            x="shap_value",
            y="display_name",
            orientation="h",
            color="direction",
            color_discrete_map={"Increases PM2.5": "#ef4444", "Decreases PM2.5": "#10b981"},
            title="<b>Top 10 Feature Contributions to Current AQI</b>",
            labels={"shap_value": "SHAP Impact on PM2.5 (µg/m³)", "display_name": "Feature"},
            template="plotly_dark",
        )
        fig_waterfall.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_waterfall, use_container_width=True)

    with col_shap2:
        st.subheader("Global Feature Importance Ranking")
        df_global = explainer.get_global_importance().head(10)
        
        fig_global = px.bar(
            df_global,
            x="relative_importance_pct",
            y="feature",
            orientation="h",
            title="<b>Global Model Feature Importance (%)</b>",
            labels={"relative_importance_pct": "Relative Importance (%)", "feature": "Feature"},
            template="plotly_dark",
        )
        fig_global.update_traces(marker_color="#6366f1")
        fig_global.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_global, use_container_width=True)


# ------------------------------------------------------------------ #
#  TAB 4: Scenario Simulator Sandbox
# ------------------------------------------------------------------ #

with tab4:
    st.subheader("Interactive Scenario Simulator (What-If Analysis)")
    st.write("Adjust meteorological and pollution parameters to simulate their impact on Karachi's forecasted air quality.")

    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        sim_pm25 = st.slider("Current PM2.5 (µg/m³)", 5.0, 150.0, float(current_obs.get("PM2.5", 25.0)), 1.0)
        sim_pm10 = st.slider("PM10 Dust (µg/m³)", 10.0, 300.0, float(current_obs.get("PM10", 60.0)), 5.0)
        sim_co = st.slider("Carbon Monoxide CO (µg/m³)", 100.0, 1200.0, float(current_obs.get("CO", 400.0)), 20.0)

    with s_col2:
        sim_temp = st.slider("Temperature (°C)", 10.0, 48.0, float(current_obs.get("temperature_2m", 28.0)), 0.5)
        sim_humid = st.slider("Relative Humidity (%)", 10.0, 100.0, float(current_obs.get("relative_humidity_2m", 65.0)), 1.0)
        sim_wind = st.slider("Wind Speed (km/h)", 1.0, 45.0, float(current_obs.get("wind_speed_10m", 14.0)), 0.5)

    with s_col3:
        sim_so2 = st.slider("Sulfur Dioxide SO2 (µg/m³)", 1.0, 50.0, float(current_obs.get("SO2", 8.0)), 0.5)
        sim_no2 = st.slider("Nitrogen Dioxide NO2 (µg/m³)", 2.0, 80.0, float(current_obs.get("NO2", 20.0)), 1.0)
        sim_pressure = st.slider("Surface Pressure (hPa)", 990.0, 1025.0, float(current_obs.get("surface_pressure", 1011.0)), 1.0)

    # Build simulated feature dictionary
    sim_features = {
        "PM2.5": sim_pm25,
        "pm25_roll_mean_24": sim_pm25,
        "pm25_roll_mean_72": sim_pm25,
        "pm25_lag_24": sim_pm25,
        "PM10": sim_pm10,
        "CO": sim_co,
        "pm25_roll_std_24": 3.5,
        "pm25_roll_std_72": 4.5,
        "SO2": sim_so2,
        "NO2": sim_no2,
        "wind_pm25": sim_wind * sim_pm25,
        "pm25_lag_48": sim_pm25,
        "pm25_lag_72": sim_pm25,
        "month_cos": -0.5,
        "temp_humidity": sim_temp * sim_humid,
        "surface_pressure": sim_pressure,
        "apparent_temperature": sim_temp + 2.0,
        "wind_speed_10m": sim_wind,
    }

    df_sim = pd.DataFrame([sim_features])
    sim_pred_df = predictor.predict(df_sim)
    sim_val = float(sim_pred_df["predicted_target_pm25"].iloc[0])
    sim_cat, sim_col_hex, sim_adv = resolve_aqi_details(sim_val)

    st.markdown("---")
    res_c1, res_c2 = st.columns([1, 2])
    with res_c1:
        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid {sim_col_hex};">
            <div style="color: #94a3b8; font-size: 0.9rem;">Simulated Next-Hour PM2.5</div>
            <div style="font-size: 2.5rem; font-weight: 800; color: {sim_col_hex};">{sim_val:.1f} <span style="font-size: 1rem;">µg/m³</span></div>
            <div class="risk-badge" style="background-color: {sim_col_hex}22; color: {sim_col_hex}; border: 1px solid {sim_col_hex}; margin-top: 8px;">
                {sim_cat}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with res_c2:
        st.markdown(f"""
        <div class="advisory-box" style="border-left-color: {sim_col_hex}; height: 100%;">
            <strong style="color: {sim_col_hex}; font-size: 1.1rem;">Impact Interpretation:</strong><br>
            {sim_adv}
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  TAB 5: MLOps Model Benchmark & Feature Store Status
# ------------------------------------------------------------------ #

with tab5:
    st.subheader("Model Evaluation Benchmark & MLOps Registry")
    st.write("Rigorous multi-model performance comparison on chronological hold-out test set (3,427 hours).")

    leaderboard_data = [
        {"Model": "Gradient Boosting", "Test RMSE": 2.3945, "Test MAE": 1.4989, "Test R²": 0.9113, "Train R²": 0.9552, "Overfit Gap": 0.0439, "Status": "🏆 Champion"},
        {"Model": "XGBoost", "Test RMSE": 2.4047, "Test MAE": 1.4891, "Test R²": 0.9105, "Train R²": 0.9711, "Overfit Gap": 0.0606, "Status": "Candidate"},
        {"Model": "Random Forest", "Test RMSE": 2.4235, "Test MAE": 1.5248, "Test R²": 0.9091, "Train R²": 0.9789, "Overfit Gap": 0.0698, "Status": "Candidate"},
        {"Model": "Lasso Regression", "Test RMSE": 2.4613, "Test MAE": 1.5567, "Test R²": 0.9062, "Train R²": 0.9271, "Overfit Gap": 0.0209, "Status": "Baseline"},
        {"Model": "Linear Regression", "Test RMSE": 2.4848, "Test MAE": 1.5849, "Test R²": 0.9044, "Train R²": 0.9271, "Overfit Gap": 0.0227, "Status": "Baseline"},
        {"Model": "Ridge Regression", "Test RMSE": 2.4851, "Test MAE": 1.5851, "Test R²": 0.9044, "Train R²": 0.9271, "Overfit Gap": 0.0227, "Status": "Baseline"},
    ]
    df_lb = pd.DataFrame(leaderboard_data)
    st.dataframe(df_lb, use_container_width=True)

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        fig_r2 = px.bar(
            df_lb, x="Model", y="Test R²",
            title="<b>Test R² Score Comparison</b>",
            color="Test R²",
            color_continuous_scale="Blues",
            template="plotly_dark",
        )
        fig_r2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)")
        st.plotly_chart(fig_r2, use_container_width=True)

    with c_b2:
        fig_rmse = px.bar(
            df_lb, x="Model", y="Test RMSE",
            title="<b>Test RMSE Error Comparison (Lower is Better)</b>",
            color="Test RMSE",
            color_continuous_scale="Reds_r",
            template="plotly_dark",
        )
        fig_rmse.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15, 23, 42, 0.6)")
        st.plotly_chart(fig_rmse, use_container_width=True)

    st.markdown("---")
    st.subheader("Hopsworks Feature Store Architecture")
    f_c1, f_c2 = st.columns(2)
    with f_c1:
        st.markdown("""
        - **Feature Group**: `karachi_aqi_features` (v1)
        - **Primary Key**: `['datetime', 'city']`
        - **Event Time**: `datetime`
        - **Features (18 Selected)**:
          - Lagged Features: `pm25_lag_24`, `pm25_lag_48`, `pm25_lag_72`
          - Rolling Features: `pm25_roll_mean_24`, `pm25_roll_std_24`, `pm25_roll_mean_72`, `pm25_roll_std_72`
          - Meteorology & Interactions: `temp_humidity`, `wind_pm25`, `month_cos`
          - Pollutants: `PM2.5`, `PM10`, `CO`, `SO2`, `NO2`
        """)
    with f_c2:
        st.markdown("""
        - **Model Registry**: Local Versioned + Hopsworks
        - **Champion Artifact**: `models/best_model.pkl`
        - **Metadata**: `models/best_model_metadata.json`
        - **Retraining Cadence**: Daily via GitHub Actions / Airflow
        - **Inference Cadence**: Hourly automated streaming
        """)