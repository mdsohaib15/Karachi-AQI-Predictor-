"""
Hourly Feature Pipeline
=======================
Runs every hour automatically via GitHub Actions / Airflow / CLI:
1. Fetches current real-time hourly weather and air quality observations for Karachi.
2. Cleans and validates data integrity (ranges, datatypes).
3. Appends raw observation to Hopsworks Feature Store (Feature Group: karachi_aqi_raw).
4. Appends to local storage (data/raw/karachi_hourly.csv) with deduplication.
5. Computes live lag, rolling, interaction, and cyclic features for model inference.
6. Saves latest feature vector to data/processed/latest_features.csv and updates processed datasets.

Run:
    python pipelines/feature_pipeline.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api_client import get_current_hourly_record
from src.feature_store import FeatureStoreManager, sanitize_column_names

# ------------------------------------------------------------------ #
#  Logging Configuration
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

load_dotenv()

RAW_DATA_PATH = Path("data/raw/karachi_hourly.csv")
PROCESSED_FEATURES_PATH = Path("data/processed/karachi_features.csv")
SELECTED_FEATURES_PATH = Path("data/processed/karachi_selected_features.csv")
LATEST_FEATURES_PATH = Path("data/processed/latest_features.csv")
SELECTED_FEATURES_TXT = Path("data/processed/selected_features.txt")


# ------------------------------------------------------------------ #
#  Step 1: Fetch and Clean Live Observation
# ------------------------------------------------------------------ #

def fetch_and_clean_hourly_record() -> pd.DataFrame:
    """Fetch current observation and validate data ranges."""
    log.info("Step 1: Fetching current live atmospheric observations for Karachi...")
    record = get_current_hourly_record()
    df_new = pd.DataFrame([record])

    # Ensure datetime format
    df_new["datetime"] = pd.to_datetime(df_new["datetime"])
    if df_new["datetime"].dt.tz is None:
        df_new["datetime"] = df_new["datetime"].dt.tz_localize("UTC")

    # Sanity checks
    df_new["PM2.5"] = df_new["PM2.5"].clip(lower=0.0, upper=1000.0)
    df_new["PM10"] = df_new["PM10"].clip(lower=0.0, upper=1500.0)
    df_new["relative_humidity_2m"] = df_new["relative_humidity_2m"].clip(lower=0.0, upper=100.0)

    log.info("✓ Record validated for timestamp: %s", df_new["datetime"].iloc[0])
    return df_new


# ------------------------------------------------------------------ #
#  Step 2: Update Local Raw Storage
# ------------------------------------------------------------------ #

def append_to_raw_storage(df_new: pd.DataFrame) -> pd.DataFrame:
    """Append new observation to raw CSV, avoiding duplicate timestamps."""
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if RAW_DATA_PATH.exists():
        df_raw = pd.read_csv(RAW_DATA_PATH, parse_dates=["datetime"])
        if df_raw["datetime"].dt.tz is None:
            df_raw["datetime"] = df_raw["datetime"].dt.tz_localize("UTC")

        # Check duplicate
        new_dt = df_new["datetime"].iloc[0]
        if new_dt in df_raw["datetime"].values:
            log.info("Timestamp %s already exists in %s. Updating existing row.", new_dt, RAW_DATA_PATH)
            df_raw = df_raw[df_raw["datetime"] != new_dt]

        df_combined = pd.concat([df_raw, df_new], ignore_index=True)
    else:
        df_combined = df_new.copy()

    df_combined = df_combined.sort_values("datetime").reset_index(drop=True)
    df_combined.to_csv(RAW_DATA_PATH, index=False)
    log.info("✓ Updated %s (total %d rows)", RAW_DATA_PATH, len(df_combined))
    return df_combined


# ------------------------------------------------------------------ #
#  Step 3: Upload to Hopsworks Feature Store
# ------------------------------------------------------------------ #

def upload_to_hopsworks_feature_store(df_features: pd.DataFrame) -> bool:
    """Insert the new live selected feature row into the single Hopsworks Feature Group."""
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        log.warning("HOPSWORKS_API_KEY not set. Skipping Feature Store upload.")
        return False

    try:
        log.info("Step 3: Uploading live selected feature vector to Hopsworks Feature Store...")
        fsm = FeatureStoreManager()
        
        # Prepare data with compliant column names
        df_hw = df_features.copy()
        if "city" not in df_hw.columns:
            df_hw["city"] = "karachi"

        # Sanitize columns
        df_sanitized, _ = sanitize_column_names(df_hw)
        
        # Insert into the single feature group: karachi_aqi_features
        fsm.insert_features(
            df=df_sanitized,
            feature_group_name="karachi_aqi_features",
            version=1,
            wait=False,
        )
        log.info("✓ Successfully inserted record into Hopsworks 'karachi_aqi_features'")
        return True
    except Exception as exc:
        log.warning("Hopsworks Feature Store upload failed: %s", exc)
        return False


# ------------------------------------------------------------------ #
#  Step 4: Compute Live Features for Next-Hour Inference
# ------------------------------------------------------------------ #

def compute_live_inference_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute engineered features on the latest historical window to produce
    the current hour's feature vector for inference.
    """
    log.info("Step 4: Computing engineered features for latest observation...")
    df = df_raw.copy().sort_values("datetime").reset_index(drop=True)

    # Time features
    dt = df["datetime"].dt
    df["year"] = dt.year
    df["month"] = dt.month
    df["day"] = dt.day
    df["hour"] = dt.hour
    df["weekday"] = dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # Lag features
    for lag in [24, 48, 72, 168]:
        df[f"pm25_lag_{lag}"] = df["PM2.5"].shift(lag)

    # Rolling features
    for w in [24, 72]:
        df[f"pm25_roll_mean_{w}"] = df["PM2.5"].rolling(w, min_periods=max(1, w // 2)).mean()
        df[f"pm25_roll_std_{w}"] = df["PM2.5"].rolling(w, min_periods=max(1, w // 2)).std().fillna(0.0)

    # Interaction features
    df["temp_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"]
    df["wind_pm25"] = df["wind_speed_10m"] * df["PM2.5"]

    # Cyclic features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Extract the very latest row (current hour live feature vector)
    latest_feature_row = df.iloc[[-1]].copy()
    
    # Save latest features for prediction pipeline & dashboard
    LATEST_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    latest_feature_row.to_csv(LATEST_FEATURES_PATH, index=False)
    log.info("✓ Saved latest feature vector -> %s", LATEST_FEATURES_PATH)

    return latest_feature_row


# ------------------------------------------------------------------ #
#  Main Execution
# ------------------------------------------------------------------ #

def run_feature_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete hourly feature pipeline."""
    t0 = time.time()
    log.info("=========================================================")
    log.info("   KARACHI AQI PREDICTOR - HOURLY FEATURE PIPELINE      ")
    log.info("=========================================================")

    # 1. Fetch live data
    df_new = fetch_and_clean_hourly_record()

    # 2. Append to raw storage
    df_raw = append_to_raw_storage(df_new)

    # 3. Feature engineering for live inference
    latest_features = compute_live_inference_features(df_raw)

    # 4. Hopsworks upload (single feature group with selected features)
    upload_to_hopsworks_feature_store(latest_features)

    elapsed = time.time() - t0
    log.info("=========================================================")
    log.info("✓ Feature Pipeline completed successfully in %.2f seconds.", elapsed)
    log.info("=========================================================")

    return df_new, latest_features


if __name__ == "__main__":
    run_feature_pipeline()