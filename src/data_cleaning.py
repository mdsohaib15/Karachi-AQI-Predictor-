"""
Data Cleaning & Standardization Pipeline
=========================================
Cleans, standardizes, and validates raw historical hourly data
before feature engineering.

Input:
    data/raw/karachi_hourly.csv          (17 columns, raw API pull)

Output:
    data/processed/karachi_cleaned.csv   (17 columns, cleaned & standardized)

Processing Steps:
    1. Schema validation (17 required columns & correct numeric types)
    2. Datetime standardization (UTC timezone, chronological sorting)
    3. Duplicate detection & removal
    4. Hourly temporal continuity & gap analysis
    5. Missing value handling (time-based interpolation for small gaps)
    6. Physical range validation & outlier clipping (e.g. non-negative concentrations)

Run:
    python src/data_cleaning.py
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

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
#  Configuration & Constants
# ------------------------------------------------------------------ #

INPUT_PATH = Path("data/raw/karachi_hourly.csv")
OUTPUT_PATH = Path("data/processed/karachi_cleaned.csv")
CONFIG_PATH = Path("config/config.yml")

DATETIME_COL = "datetime"

WEATHER_COLUMNS = [
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

AIR_QUALITY_COLUMNS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
]

REQUIRED_COLUMNS = [DATETIME_COL] + WEATHER_COLUMNS + AIR_QUALITY_COLUMNS

# Physical & plausible bounding ranges: (min_val, max_val)
PLAUSIBLE_RANGES = {
    # Weather parameters
    "temperature_2m": (-10.0, 65.0),          # °C (Karachi climate range)
    "relative_humidity_2m": (0.0, 100.0),      # %
    "dew_point_2m": (-20.0, 50.0),            # °C
    "apparent_temperature": (-15.0, 70.0),     # °C
    "precipitation": (0.0, 500.0),             # mm/h
    "rain": (0.0, 500.0),                      # mm/h
    "surface_pressure": (850.0, 1100.0),       # hPa
    "cloud_cover": (0.0, 100.0),               # %
    "wind_speed_10m": (0.0, 150.0),            # km/h or m/s non-negative
    "wind_direction_10m": (0.0, 360.0),        # degrees [0, 360]
    # Air quality concentrations (µg/m³) - must be non-negative
    "PM2.5": (0.0, 1500.0),
    "PM10": (0.0, 2500.0),
    "NO2": (0.0, 1000.0),
    "SO2": (0.0, 1000.0),
    "CO": (0.0, 50000.0),
    "O3": (0.0, 1000.0),
}


# ------------------------------------------------------------------ #
#  Config Loader
# ------------------------------------------------------------------ #

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load settings from config.yml if available."""
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            log.warning("Could not parse %s (%s); using default parameters", config_path, e)
    return {}


# ------------------------------------------------------------------ #
#  Load Data
# ------------------------------------------------------------------ #

def load_raw_data(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    log.info("Loading raw data from %s …", path)
    df = pd.read_csv(path)
    log.info("  Loaded %d rows × %d columns", len(df), len(df.columns))
    return df


# ------------------------------------------------------------------ #
#  1. Schema & Type Validation
# ------------------------------------------------------------------ #

def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required columns exist and numeric features are correctly typed."""
    df = df.copy()
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema Error: Missing required columns: {missing_cols}")

    # Convert numeric columns
    numeric_cols = WEATHER_COLUMNS + AIR_QUALITY_COLUMNS
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info("✓ Schema validation passed: All 17 required columns present & typed")
    return df


# ------------------------------------------------------------------ #
#  2. Datetime Standardization & Sorting
# ------------------------------------------------------------------ #

def standardize_datetime(
    df: pd.DataFrame,
    dt_col: str = DATETIME_COL,
    timezone_str: str = "UTC"
) -> pd.DataFrame:
    """Convert datetime column to UTC timezone-aware timestamps and sort chronologically."""
    df = df.copy()

    # Parse datetime
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")

    # Check for unparseable dates
    invalid_dt = df[dt_col].isna().sum()
    if invalid_dt > 0:
        log.warning("Dropping %d rows with unparseable datetime values", invalid_dt)
        df = df.dropna(subset=[dt_col])

    # Sort chronologically
    df = df.sort_values(dt_col).reset_index(drop=True)
    log.info("✓ Datetime standardized (UTC) & chronologically sorted")
    return df


# ------------------------------------------------------------------ #
#  3. Duplicate Handling
# ------------------------------------------------------------------ #

def handle_duplicates(df: pd.DataFrame, dt_col: str = DATETIME_COL) -> pd.DataFrame:
    """Detect and remove duplicate timestamps, retaining the first valid entry."""
    df = df.copy()
    num_dupes = df.duplicated(subset=[dt_col]).sum()

    if num_dupes > 0:
        log.warning("Found %d duplicate timestamps. Removing duplicates (keeping first) …", num_dupes)
        df = df.drop_duplicates(subset=[dt_col], keep="first").reset_index(drop=True)
    else:
        log.info("✓ No duplicate timestamps found")

    return df


# ------------------------------------------------------------------ #
#  4. Continuity & Temporal Gap Check
# ------------------------------------------------------------------ #

def check_time_continuity(df: pd.DataFrame, dt_col: str = DATETIME_COL) -> None:
    """Analyze time series for missing hourly timestamps."""
    if len(df) < 2:
        return

    time_diffs = df[dt_col].diff().dropna()
    expected_step = pd.Timedelta(hours=1)
    gaps = time_diffs[time_diffs > expected_step]

    if len(gaps) > 0:
        log.warning("Detected %d temporal gap(s) larger than 1 hour in hourly sequence", len(gaps))
        for idx, gap in gaps.head(5).items():
            gap_start = df.loc[idx - 1, dt_col]
            gap_end = df.loc[idx, dt_col]
            log.warning("    Gap: %s  →  %s  (duration: %s)", gap_start, gap_end, gap)
        if len(gaps) > 5:
            log.warning("    ... and %d more gap(s)", len(gaps) - 5)
    else:
        log.info("✓ Time-series continuity verified: Consecutive 1-hour intervals across full range")


# ------------------------------------------------------------------ #
#  5. Missing Value Handling
# ------------------------------------------------------------------ #

def handle_missing_values(
    df: pd.DataFrame,
    max_interpolate_hours: int = 3,
) -> pd.DataFrame:
    """
    Handle missing values appropriately for hourly environmental time series:
    - Small gaps (<= max_interpolate_hours) are linearly interpolated across time.
    - Remaining edge NaNs use nearest valid forward/backward fill.
    """
    df = df.copy()
    numeric_cols = WEATHER_COLUMNS + AIR_QUALITY_COLUMNS

    missing_counts_before = df[numeric_cols].isna().sum()
    total_missing_before = missing_counts_before.sum()

    if total_missing_before == 0:
        log.info("✓ No missing values detected in any feature column")
        return df

    log.info("Handling missing values (Total missing: %d) …", total_missing_before)
    for col, count in missing_counts_before[missing_counts_before > 0].items():
        log.info("    Missing in %-20s: %d (%.2f%%)", col, count, count / len(df) * 100)

    # Set datetime index temporarily for time-based interpolation
    df = df.set_index(DATETIME_COL)

    # Linear interpolation for small gaps (up to max_interpolate_hours)
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="time",
        limit=max_interpolate_hours,
        limit_direction="both"
    )

    # Fill any remaining edge missing values
    df[numeric_cols] = df[numeric_cols].bfill().ffill()

    df = df.reset_index()

    remaining_missing = df[numeric_cols].isna().sum().sum()
    log.info("✓ Missing value resolution complete. Remaining NaNs: %d", remaining_missing)
    return df


# ------------------------------------------------------------------ #
#  6. Range Validation & Outlier Boundaries
# ------------------------------------------------------------------ #

def validate_and_clean_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate physical boundaries for meteorological and pollutant variables.
    Clips physically impossible negative values (sensor artifacts) to zero,
    and bounds percentages (humidity, clouds) to [0, 100].
    """
    df = df.copy()
    clipped_report = {}

    for col, (min_val, max_val) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue

        original_series = df[col]
        clipped_series = original_series.copy()

        if min_val is not None:
            below_min = (clipped_series < min_val).sum()
            if below_min > 0:
                clipped_report[f"{col} < {min_val}"] = below_min
                clipped_series = clipped_series.clip(lower=min_val)

        if max_val is not None:
            above_max = (clipped_series > max_val).sum()
            if above_max > 0:
                clipped_report[f"{col} > {max_val}"] = above_max
                clipped_series = clipped_series.clip(upper=max_val)

        df[col] = clipped_series

    if clipped_report:
        log.info("Outlier/Range adjustments applied:")
        for condition, count in clipped_report.items():
            log.info("    Clipped %-25s: %d values", condition, count)
    else:
        log.info("✓ All numeric variables are strictly within valid physical bounds")

    return df


# ------------------------------------------------------------------ #
#  Master Clean Function
# ------------------------------------------------------------------ #

def clean_data(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Execute full data cleaning and standardization pipeline on DataFrame.
    Can be invoked directly or imported into pipelines.
    """
    # 1. Schema Validation
    df = validate_schema(df)

    # 2. Datetime standardization
    df = standardize_datetime(df)

    # 3. Duplicate handling
    df = handle_duplicates(df)

    # 4. Temporal continuity check
    check_time_continuity(df)

    # 5. Missing value resolution
    df = handle_missing_values(df)

    # 6. Range bounds & physical sanity check
    df = validate_and_clean_ranges(df)

    # 7. Order columns strictly
    df = df[REQUIRED_COLUMNS]

    return df


# ------------------------------------------------------------------ #
#  Save Cleaned Dataset
# ------------------------------------------------------------------ #

def save_cleaned_data(df: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """Save cleaned dataset to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("✓ Cleaned dataset saved to %s", output_path)
    log.info("  Shape: %s", df.shape)


# ------------------------------------------------------------------ #
#  Summary Report
# ------------------------------------------------------------------ #

def print_cleaning_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Print comprehensive summary comparing raw and cleaned datasets."""
    log.info("=" * 55)
    log.info("CLEANING & STANDARDIZATION SUMMARY REPORT")
    log.info("=" * 55)
    log.info("  Raw dataset rows        : %d", len(df_raw))
    log.info("  Cleaned dataset rows    : %d", len(df_clean))
    log.info("  Total columns           : %d", len(df_clean.columns))
    log.info("  Datetime range          : %s  →  %s",
             df_clean[DATETIME_COL].min(), df_clean[DATETIME_COL].max())
    log.info("  Remaining missing values: %d", df_clean.isna().sum().sum())
    log.info("  Infinite values         : %d",
             np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum())
    log.info("-" * 55)
    log.info("  Column Summary:")
    for col in df_clean.columns:
        if col == DATETIME_COL:
            log.info("    • %-22s : [datetime, tz-aware UTC]", col)
        else:
            log.info("    • %-22s : min=%8.2f | mean=%8.2f | max=%8.2f",
                     col, df_clean[col].min(), df_clean[col].mean(), df_clean[col].max())
    log.info("=" * 55)


# ------------------------------------------------------------------ #
#  Main Execution
# ------------------------------------------------------------------ #

def main() -> None:
    log.info("=" * 55)
    log.info("Starting Data Cleaning Pipeline")
    log.info("=" * 55)

    t0 = time.time()
    config = load_config()

    # Load raw data
    raw_df = load_raw_data(INPUT_PATH)

    # Run cleaning pipeline
    clean_df = clean_data(raw_df, config=config)

    # Print summary
    print_cleaning_summary(raw_df, clean_df)

    # Save cleaned data
    save_cleaned_data(clean_df, OUTPUT_PATH)

    elapsed = time.time() - t0
    log.info("Data cleaning completed successfully in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
