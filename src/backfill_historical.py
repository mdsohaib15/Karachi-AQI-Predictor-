"""
Step 2: Historical Backfill
============================
Download 24 months of historical hourly weather and air-quality
data for Karachi using the Open-Meteo APIs (free, no key required).

Sources:
    Weather     → Open-Meteo Archive API
    Air Quality → Open-Meteo Air Quality API

Output:
    data/raw/karachi_hourly.csv   (17 columns, ~17 500 rows)

Run:
    python src/backfill_historical.py
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# ------------------------------------------------------------------ #
#  Logging
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #

LATITUDE = 24.8607
LONGITUDE = 67.0011

HISTORY_MONTHS = 24          # look-back window

OUTPUT_PATH = Path("data/raw/karachi_hourly.csv")

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
]

AIR_QUALITY_PARAMS = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "carbon_monoxide",
    "ozone",
]

# Map Open-Meteo AQ names → project standard names (config.yml)
AQ_RENAME_MAP = {
    "pm2_5":             "PM2.5",
    "pm10":              "PM10",
    "nitrogen_dioxide":  "NO2",
    "sulphur_dioxide":   "SO2",
    "carbon_monoxide":   "CO",
    "ozone":             "O3",
}

# Final column order (must match docs/dataset_info.md)
REQUIRED_COLUMNS = [
    "datetime",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
]

REQUEST_TIMEOUT = 60         # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 2            # exponential base (seconds)


# ------------------------------------------------------------------ #
#  Date helpers
# ------------------------------------------------------------------ #

def compute_date_range() -> tuple[str, str]:
    """Return (start_date, end_date) strings for the past N months."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=HISTORY_MONTHS * 30)   # approximate
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    log.info("Date range: %s  →  %s  (%d months)", start_str, end_str, HISTORY_MONTHS)
    return start_str, end_str


# ------------------------------------------------------------------ #
#  HTTP helper with retry
# ------------------------------------------------------------------ #

def _get_json(url: str, params: dict) -> dict:
    """GET request with timeout and retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF ** attempt
                log.warning(
                    "HTTP %d on attempt %d/%d – retrying in %.1fs …",
                    status, attempt, MAX_RETRIES, wait,
                )
                time.sleep(wait)
            else:
                raise
        except requests.exceptions.ConnectionError:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF ** attempt
                log.warning(
                    "Connection error on attempt %d/%d – retrying in %.1fs …",
                    attempt, MAX_RETRIES, wait,
                )
                time.sleep(wait)
            else:
                raise
    # Should never reach here, but just in case
    raise RuntimeError("Exceeded max retries")


# ------------------------------------------------------------------ #
#  Fetch weather data
# ------------------------------------------------------------------ #

def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical weather from Open-Meteo Archive API."""
    log.info("Fetching weather data …")

    params = {
        "latitude":   LATITUDE,
        "longitude":  LONGITUDE,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     ",".join(WEATHER_PARAMS),
        "timezone":   "UTC",
    }

    data = _get_json(WEATHER_URL, params)
    df = parse_weather(data)

    log.info("Weather rows received: %d", len(df))
    return df


def parse_weather(raw: dict) -> pd.DataFrame:
    """Convert raw Open-Meteo weather JSON to DataFrame."""
    hourly = raw["hourly"]
    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


# ------------------------------------------------------------------ #
#  Fetch air-quality data
# ------------------------------------------------------------------ #

def fetch_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical air quality from Open-Meteo AQ API."""
    log.info("Fetching air quality data …")

    params = {
        "latitude":   LATITUDE,
        "longitude":  LONGITUDE,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     ",".join(AIR_QUALITY_PARAMS),
        "timezone":   "UTC",
    }

    data = _get_json(AIR_QUALITY_URL, params)
    df = parse_air_quality(data)

    log.info("Air quality rows received: %d", len(df))
    return df


def parse_air_quality(raw: dict) -> pd.DataFrame:
    """Convert raw Open-Meteo AQ JSON to DataFrame and rename columns."""
    hourly = raw["hourly"]
    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "datetime"})
    df = df.rename(columns=AQ_RENAME_MAP)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df


# ------------------------------------------------------------------ #
#  Merge & align
# ------------------------------------------------------------------ #

def merge_and_align(
    weather_df: pd.DataFrame,
    aq_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge weather and air-quality on datetime."""
    log.info("Merging weather (%d rows) + air quality (%d rows) …",
             len(weather_df), len(aq_df))

    # Both datasets are hourly from the same source; inner merge is safe
    df = pd.merge(weather_df, aq_df, on="datetime", how="inner")

    # Sort and deduplicate
    df = df.sort_values("datetime").reset_index(drop=True)
    before = len(df)
    df = df.drop_duplicates(subset="datetime").reset_index(drop=True)
    dupes = before - len(df)
    if dupes:
        log.warning("Removed %d duplicate timestamps", dupes)

    log.info("Merged rows: %d", len(df))
    return df


# ------------------------------------------------------------------ #
#  Validation
# ------------------------------------------------------------------ #

def validate_dataset(df: pd.DataFrame) -> None:
    """Run validation checks and log a summary report."""
    log.info("=" * 50)
    log.info("VALIDATION REPORT")
    log.info("=" * 50)

    # Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    log.info("✓ All 17 required columns present")

    # Reorder to canonical order
    df_ordered = df[REQUIRED_COLUMNS]

    # Stats
    log.info("  Rows           : %d", len(df_ordered))
    log.info("  Earliest       : %s", df_ordered["datetime"].min())
    log.info("  Latest         : %s", df_ordered["datetime"].max())

    # Missing values
    log.info("  Missing values :")
    for col in REQUIRED_COLUMNS:
        n_miss = df_ordered[col].isna().sum()
        pct = n_miss / len(df_ordered) * 100
        status = "✓" if n_miss == 0 else "⚠"
        log.info("    %s %-25s %5d  (%.1f%%)", status, col, n_miss, pct)

    log.info("=" * 50)


# ------------------------------------------------------------------ #
#  Save
# ------------------------------------------------------------------ #

def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save the validated dataset to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Enforce column order
    df = df[REQUIRED_COLUMNS]
    df.to_csv(path, index=False)

    log.info("✓ Saved to %s", path)
    log.info("  Shape: %s", df.shape)


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main() -> None:
    log.info("=" * 50)
    log.info("Karachi Historical Data Backfill")
    log.info("=" * 50)

    t0 = time.time()

    # 1. Compute date range
    start_date, end_date = compute_date_range()

    # 2. Fetch data
    weather_df = fetch_weather(start_date, end_date)
    aq_df = fetch_air_quality(start_date, end_date)

    # 3. Merge
    df = merge_and_align(weather_df, aq_df)

    # 4. Validate
    validate_dataset(df)

    # 5. Save
    save_csv(df, OUTPUT_PATH)

    elapsed = time.time() - t0
    log.info("Done in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()