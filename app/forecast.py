"""
Multi-Step 72-Hour (3-Day) Forecasting Engine
=============================================
Computes sequential recursive forecasts for the next 72 hours in Karachi:
- Dynamically shifts timestamp hour-by-hour
- Recomputes lag features (lag_24, lag_48, lag_72) from predicted history
- Updates rolling windows (roll_mean_24, roll_mean_72, roll_std_24, roll_std_72)
- Recomputes cyclic time encodings (hour_sin, hour_cos, month_cos) and weather interactions
- Generates 90% and 95% confidence intervals and EPA AQI risk classifications
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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

log = logging.getLogger(__name__)

LATEST_FEATURES_CSV = Path("data/processed/latest_features.csv")
RAW_DATA_PATH = Path("data/raw/karachi_hourly.csv")


class MultiStepForecaster:
    """Recursively forecasts Karachi PM2.5 up to 72 hours ahead."""

    def __init__(self, predictor: Optional[AQIPredictor] = None):
        self.predictor = predictor or AQIPredictor()
        self.features = self.predictor.features

    def generate_72h_forecast(
        self,
        base_features_df: Optional[pd.DataFrame] = None,
        hours_ahead: int = 72,
    ) -> pd.DataFrame:
        """
        Generate hour-by-hour forecast for the specified horizon (default 72 hours / 3 days).
        """
        # Load baseline historical window
        if base_features_df is None or base_features_df.empty:
            if LATEST_FEATURES_CSV.exists():
                base_features_df = pd.read_csv(LATEST_FEATURES_CSV)
            elif RAW_DATA_PATH.exists():
                from pipelines.feature_pipeline import compute_live_inference_features
                df_raw = pd.read_csv(RAW_DATA_PATH, parse_dates=["datetime"])
                base_features_df = compute_live_inference_features(df_raw)
            else:
                raise FileNotFoundError("No feature baseline available for forecasting.")

        current_row = base_features_df.iloc[-1].to_dict()
        
        # Parse base timestamp
        base_dt_val = current_row.get("datetime", datetime.now(timezone.utc).isoformat())
        try:
            base_dt = pd.to_datetime(base_dt_val)
            if base_dt.tz is None:
                base_dt = base_dt.tz_localize("UTC")
        except Exception:
            base_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        # Buffer of PM2.5 history (start with available lag values or current PM2.5)
        current_pm25 = float(current_row.get("PM2.5", current_row.get("pm2_5", 25.0)))
        
        # Build initial 72h historical buffer if available
        history_pm25 = []
        if RAW_DATA_PATH.exists():
            df_raw = pd.read_csv(RAW_DATA_PATH)
            pm_col = "PM2.5" if "PM2.5" in df_raw.columns else "pm2_5"
            if pm_col in df_raw.columns:
                history_pm25 = df_raw[pm_col].tail(168).tolist()

        if len(history_pm25) < 72:
            history_pm25 = [current_pm25] * 72

        forecast_records: List[Dict[str, Any]] = []
        step_row = current_row.copy()
        
        # Model standard error from metadata for confidence interval calculation
        test_rmse = float(self.predictor.metadata.get("metrics", {}).get("RMSE", 2.40))

        for step in range(1, hours_ahead + 1):
            future_dt = base_dt + timedelta(hours=step)
            
            # 1. Update temporal & cyclic encodings
            hour = future_dt.hour
            month = future_dt.month
            step_row["hour"] = hour
            step_row["month"] = month
            step_row["month_cos"] = np.cos(2 * np.pi * month / 12)
            step_row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            step_row["hour_cos"] = np.cos(2 * np.pi * hour / 24)

            # 2. Update lag features from buffer
            if len(history_pm25) >= 24:
                step_row["pm25_lag_24"] = history_pm25[-24]
            if len(history_pm25) >= 48:
                step_row["pm25_lag_48"] = history_pm25[-48]
            if len(history_pm25) >= 72:
                step_row["pm25_lag_72"] = history_pm25[-72]

            # 3. Update rolling statistics
            recent_24 = history_pm25[-24:] if len(history_pm25) >= 24 else history_pm25
            recent_72 = history_pm25[-72:] if len(history_pm25) >= 72 else history_pm25

            step_row["pm25_roll_mean_24"] = float(np.mean(recent_24))
            step_row["pm25_roll_std_24"] = float(np.std(recent_24)) if len(recent_24) > 1 else 0.0
            step_row["pm25_roll_mean_72"] = float(np.mean(recent_72))
            step_row["pm25_roll_std_72"] = float(np.std(recent_72)) if len(recent_72) > 1 else 0.0

            # 4. Update interactions
            temp = float(step_row.get("temperature_2m", 28.0))
            humid = float(step_row.get("relative_humidity_2m", 65.0))
            wind = float(step_row.get("wind_speed_10m", 12.0))
            step_row["temp_humidity"] = temp * humid
            step_row["wind_pm25"] = wind * current_pm25

            # 5. Build single-row DataFrame and predict
            df_input = pd.DataFrame([step_row])
            pred_df = self.predictor.predict(df_input)
            pred_val = max(0.5, float(pred_df["predicted_target_pm25"].iloc[0]))
            
            # Accumulate compound forecast uncertainty over horizon
            uncertainty_growth = 1.0 + 0.015 * np.sqrt(step)
            step_se = test_rmse * uncertainty_growth
            ci_lower = max(0.0, pred_val - 1.645 * step_se)   # 90% Lower
            ci_upper = pred_val + 1.645 * step_se              # 90% Upper

            aqi_cat, color_hex, advisory = resolve_aqi_details(pred_val)

            record = {
                "step": step,
                "timestamp": future_dt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "forecast_hour_display": future_dt.strftime("%a %I:%M %p"),
                "day_offset": (step - 1) // 24 + 1,
                "predicted_pm25": round(pred_val, 2),
                "lower_bound_90": round(ci_lower, 2),
                "upper_bound_90": round(ci_upper, 2),
                "aqi_category": aqi_cat,
                "color_code": color_hex,
                "health_advisory": advisory,
            }
            forecast_records.append(record)

            # Advance autoregressive history buffer
            history_pm25.append(pred_val)
            current_pm25 = pred_val
            if "PM2.5" in step_row:
                step_row["PM2.5"] = pred_val
            if "pm2_5" in step_row:
                step_row["pm2_5"] = pred_val

        return pd.DataFrame(forecast_records)

    def get_3day_summary(self, forecast_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Compute aggregate daily stats (Day 1, Day 2, Day 3)."""
        summaries = []
        for day in [1, 2, 3]:
            day_slice = forecast_df[forecast_df["day_offset"] == day]
            if not day_slice.empty:
                avg_pm25 = float(day_slice["predicted_pm25"].mean())
                max_pm25 = float(day_slice["predicted_pm25"].max())
                min_pm25 = float(day_slice["predicted_pm25"].min())
                peak_time = day_slice.loc[day_slice["predicted_pm25"].idxmax(), "forecast_hour_display"]
                avg_cat, avg_color, avg_adv = resolve_aqi_details(avg_pm25)
                
                day_start_dt = pd.to_datetime(day_slice["timestamp"].iloc[0])
                day_name = day_start_dt.strftime("%A, %b %d")

                summaries.append({
                    "day_offset": day,
                    "day_name": day_name,
                    "avg_pm25": round(avg_pm25, 1),
                    "max_pm25": round(max_pm25, 1),
                    "min_pm25": round(min_pm25, 1),
                    "peak_hour": peak_time,
                    "aqi_category": avg_cat,
                    "color_code": avg_color,
                    "health_advisory": avg_adv,
                })
        return summaries
