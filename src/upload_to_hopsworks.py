"""
Upload Dataset to Hopsworks Feature Store
==========================================
Uploads the processed feature dataset (or raw/cleaned data)
to the Hopsworks Feature Store.

Default Feature Group:
    - Name        : karachi_aqi_features
    - Version     : 1
    - Source File : data/processed/karachi_features.csv
    - Primary Key : ["datetime", "city"]
    - Event Time  : datetime

Run:
    python src/upload_to_hopsworks.py
    python src/upload_to_hopsworks.py --file data/processed/karachi_cleaned.csv --name karachi_aqi_raw
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv

try:
    from src.feature_store import (
        FeatureStoreManager,
        get_feature_store,
        sanitize_column_names,
    )
except ImportError:
    from feature_store import (
        FeatureStoreManager,
        get_feature_store,
        sanitize_column_names,
    )

# ------------------------------------------------------------------ #
#  Logging
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

load_dotenv()

# ------------------------------------------------------------------ #
#  Default Configuration
# ------------------------------------------------------------------ #

DEFAULT_FEATURES_PATH = Path("data/processed/karachi_selected_features.csv")
DEFAULT_FULL_FEATURES_PATH = Path("data/processed/karachi_features.csv")
DEFAULT_CLEANED_PATH = Path("data/processed/karachi_cleaned.csv")
DEFAULT_RAW_PATH = Path("data/raw/karachi_hourly.csv")

DEFAULT_FEATURE_GROUP_NAME = "karachi_aqi_features"
DEFAULT_VERSION = 1


# ------------------------------------------------------------------ #
#  Upload Pipeline
# ------------------------------------------------------------------ #

def upload_features_to_hopsworks(
    file_path: Path,
    feature_group_name: str = DEFAULT_FEATURE_GROUP_NAME,
    version: int = DEFAULT_VERSION,
    description: str = "Karachi hourly weather & air quality features for AQI prediction",
    online_enabled: bool = False,
    wait: bool = True,
) -> None:
    """
    Load CSV, prepare columns, connect to Hopsworks, and insert into Feature Group.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    log.info("=" * 60)
    log.info("Hopsworks Feature Group Upload")
    log.info("=" * 60)
    log.info("  Source file    : %s", file_path)
    log.info("  Feature Group  : %s (v%d)", feature_group_name, version)

    # 1. Load Data
    log.info("Reading source dataset …")
    df = pd.read_csv(file_path)
    log.info("  Loaded %d rows × %d columns", len(df), len(df.columns))

    if "datetime" not in df.columns:
        raise ValueError("Dataset must contain a 'datetime' column for event_time.")

    # 2. Add City identifier
    city_name = os.getenv("CITY", "karachi").lower()
    df["city"] = city_name

    # 3. Ensure datetime parsing
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # 4. Preview sanitized schema
    df_sanitized, col_map = sanitize_column_names(df)
    log.info("  Sanitized columns (%d total):", len(df_sanitized.columns))
    for orig, san in list(col_map.items())[:8]:
        if orig != san:
            log.info("    • %s  →  %s", orig, san)
    if len(col_map) > 8:
        log.info("    ... and %d more columns", len(col_map) - 8)

    # 5. Connect and Upload
    t0 = time.time()
    fs_mgr = FeatureStoreManager()

    fg = fs_mgr.insert_features(
        df=df,
        feature_group_name=feature_group_name,
        version=version,
        description=description,
        primary_key=["datetime", "city"],
        event_time="datetime",
        online_enabled=online_enabled,
        wait=wait,
    )

    elapsed = time.time() - t0

    # 6. Report
    log.info("=" * 60)
    log.info("UPLOAD SUMMARY")
    log.info("=" * 60)
    log.info("✓ Feature Group '%s' (v%d) updated successfully", feature_group_name, version)
    log.info("  Rows uploaded  : %d", len(df))
    log.info("  Columns total  : %d", len(df_sanitized.columns))
    log.info("  Earliest event : %s", df["datetime"].min())
    log.info("  Latest event   : %s", df["datetime"].max())
    log.info("  Upload time    : %.2f seconds", elapsed)
    log.info("=" * 60)


# ------------------------------------------------------------------ #
#  CLI Interface
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload local feature or raw datasets to Hopsworks Feature Store."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help=f"Path to input CSV file (default: {DEFAULT_FEATURES_PATH})",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_FEATURE_GROUP_NAME,
        help=f"Feature Group name (default: {DEFAULT_FEATURE_GROUP_NAME})",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=DEFAULT_VERSION,
        help=f"Feature Group version (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Karachi hourly weather & air quality features for AQI prediction",
        help="Feature group description",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Enable online storage for real-time low-latency retrieval",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait synchronously for the ingestion job to finish",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # If default features file doesn't exist, check fallback
    target_file = args.file
    if not target_file.exists():
        if DEFAULT_CLEANED_PATH.exists():
            target_file = DEFAULT_CLEANED_PATH
            log.warning("'%s' not found. Falling back to '%s'", args.file, target_file)
        elif DEFAULT_RAW_PATH.exists():
            target_file = DEFAULT_RAW_PATH
            log.warning("'%s' not found. Falling back to '%s'", args.file, target_file)
        else:
            log.error("No dataset file found to upload. Please run backfill and feature engineering first.")
            sys.exit(1)

    upload_features_to_hopsworks(
        file_path=target_file,
        feature_group_name=args.name,
        version=args.version,
        description=args.description,
        online_enabled=args.online,
        wait=not args.no_wait,
    )


if __name__ == "__main__":
    main()