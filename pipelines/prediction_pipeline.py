"""
Hourly Prediction Pipeline
==========================
Runs every hour automatically via GitHub Actions / Airflow / CLI:
1. Loads the latest live feature vector (from data/processed/latest_features.csv).
2. Loads the registered champion model from models/ directory.
3. Computes the predicted next-hour PM2.5 concentration and AQI category.
4. Saves prediction to monitoring log (data/monitoring/predictions_log.csv).
5. Updates data/monitoring/latest_prediction.json for the Streamlit web dashboard.

Run:
    python pipelines/prediction_pipeline.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import AQIPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

load_dotenv()

LATEST_FEATURES_PATH = Path("data/processed/latest_features.csv")
MONITORING_DIR = Path("data/monitoring")
PREDICTIONS_LOG_PATH = MONITORING_DIR / "predictions_log.csv"
LATEST_PRED_JSON_PATH = MONITORING_DIR / "latest_prediction.json"


def run_prediction_pipeline() -> Dict[str, Any]:
    """Execute hourly automated prediction pipeline."""
    t0 = time.time()
    log.info("=========================================================")
    log.info("   KARACHI AQI PREDICTOR - HOURLY PREDICTION PIPELINE   ")
    log.info("=========================================================")

    # 1. Load latest features
    if not LATEST_FEATURES_PATH.exists():
        log.info("Latest features not found. Triggering feature pipeline first...")
        from pipelines.feature_pipeline import run_feature_pipeline
        run_feature_pipeline()

    df_latest = pd.read_csv(LATEST_FEATURES_PATH)
    log.info("Loaded latest live feature record (%s)", df_latest.get("datetime", ["Now"])[0])

    # 2. Load champion predictor
    predictor = AQIPredictor()

    # 3. Generate prediction
    pred_df = predictor.predict(df_latest)
    pred_row = pred_df.iloc[0].to_dict()

    # 4. Append to monitoring log
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    
    current_obs_pm25 = float(df_latest["PM2.5"].iloc[0]) if "PM2.5" in df_latest.columns else None
    
    log_entry = {
        "prediction_time": datetime.now(timezone.utc).isoformat() + "Z",
        "input_timestamp": str(df_latest.get("datetime", [""])[0]),
        "current_pm25": current_obs_pm25,
        "predicted_next_hour_pm25": pred_row["predicted_target_pm25"],
        "aqi_category": pred_row["aqi_category"],
        "color_code": pred_row["color_code"],
        "health_advisory": pred_row["health_advisory"],
        "champion_model": predictor.metadata.get("model_name", "Gradient Boosting"),
    }
    
    log_df = pd.DataFrame([log_entry])
    if PREDICTIONS_LOG_PATH.exists():
        log_df.to_csv(PREDICTIONS_LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(PREDICTIONS_LOG_PATH, mode="w", header=True, index=False)

    # 5. Save latest prediction JSON for Streamlit UI / FastAPI
    with open(LATEST_PRED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2)

    elapsed = time.time() - t0
    log.info("=========================================================")
    log.info("✓ Predicted Next-Hour PM2.5: %.2f µg/m³ (%s)",
             pred_row["predicted_target_pm25"], pred_row["aqi_category"])
    log.info("✓ Updated %s and %s in %.2fs", PREDICTIONS_LOG_PATH, LATEST_PRED_JSON_PATH, elapsed)
    log.info("=========================================================")

    return log_entry


if __name__ == "__main__":
    run_prediction_pipeline()
