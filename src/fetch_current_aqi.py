"""
Fetch Current AQI & Weather Data
================================
Fetches real-time hourly meteorological and air quality observations
for Karachi and saves them for monitoring, feature updating, and live inference.

Output:
    data/monitoring/current_aqi.csv
    data/monitoring/current_aqi.json

Run:
    python src/fetch_current_aqi.py
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.api_client import get_current_hourly_record

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MONITORING_DIR = Path("data/monitoring")
OUTPUT_CSV = MONITORING_DIR / "current_aqi.csv"
OUTPUT_JSON = MONITORING_DIR / "current_aqi.json"


def fetch_and_save_current_aqi() -> pd.DataFrame:
    """Fetch current observation and persist to monitoring storage."""
    log.info("Fetching real-time hourly atmospheric data for Karachi...")
    record = get_current_hourly_record()

    df = pd.DataFrame([record])
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    # Save current observation
    df.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    log.info("✓ Saved current AQI to %s and %s", OUTPUT_CSV, OUTPUT_JSON)
    log.info("Observation Summary:")
    log.info("  Timestamp       : %s", record["datetime"])
    log.info("  PM2.5           : %.1f µg/m³", record["PM2.5"])
    log.info("  PM10            : %.1f µg/m³", record["PM10"])
    log.info("  Temperature     : %.1f °C", record["temperature_2m"])
    log.info("  Relative Humidity: %.1f %%", record["relative_humidity_2m"])
    log.info("  Wind Speed      : %.1f km/h", record["wind_speed_10m"])

    return df


if __name__ == "__main__":
    fetch_and_save_current_aqi()