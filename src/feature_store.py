"""
Hopsworks Feature Store Wrapper & Utility Module
=================================================
Provides a clean, robust, and unified interface to connect,
write, and read from the Hopsworks Feature Store.

Features:
    - Automatic credential loading (.env validation)
    - Column sanitization (Avro / Hopsworks compliance: e.g. PM2.5 -> pm2_5)
    - Feature Group creation, insertion, and retrieval
    - Feature View creation and dataset splitting
    - Online/Offline feature reading for batch inference and training

Project: Karachi_AQI_Predictors
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import hopsworks
from hsfs.feature_group import FeatureGroup
from hsfs.feature_store import FeatureStore as HSFSFeatureStore
from hsfs.feature_view import FeatureView

# ------------------------------------------------------------------ #
#  Logging & Environment
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Load .env file
load_dotenv()


# ------------------------------------------------------------------ #
#  Column Sanitization for Hopsworks / Avro
# ------------------------------------------------------------------ #

# Map project column names to Hopsworks compliant column names
HOPSWORKS_COLUMN_MAP = {
    "PM2.5": "pm2_5",
    "PM10": "pm10",
    "NO2": "no2",
    "SO2": "so2",
    "CO": "co",
    "O3": "o3",
}

REVERSE_COLUMN_MAP = {v: k for k, v in HOPSWORKS_COLUMN_MAP.items()}


def sanitize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Sanitize dataframe columns for Hopsworks compatibility:
    - Lowercase all column names
    - Replace '.' with '_' (e.g. PM2.5 -> pm2_5)
    - Remove invalid Avro characters
    """
    df_clean = df.copy()
    mapping = {}

    for col in df_clean.columns:
        if col in HOPSWORKS_COLUMN_MAP:
            new_col = HOPSWORKS_COLUMN_MAP[col]
        else:
            # Replace special characters with underscores and lowercase
            new_col = re.sub(r"[^\w]", "_", col).strip("_").lower()
            # Collapse multiple underscores
            new_col = re.sub(r"_+", "_", new_col)
        mapping[col] = new_col

    df_clean = df_clean.rename(columns=mapping)
    return df_clean, mapping


def desanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Restore original project column names where applicable."""
    df_restored = df.copy()
    return df_restored.rename(columns=REVERSE_COLUMN_MAP)


# ------------------------------------------------------------------ #
#  Authentication & Feature Store Connection
# ------------------------------------------------------------------ #

def get_hopsworks_credentials() -> Tuple[str, str]:
    """Retrieve and validate Hopsworks credentials from environment."""
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT", "Karachi_AQI_Predictors")

    if not api_key:
        raise ValueError(
            "Missing 'HOPSWORKS_API_KEY' in environment. "
            "Please check your .env file or environment variables."
        )
    return api_key, project_name


def get_feature_store(
    project_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> HSFSFeatureStore:
    """
    Authenticate with Hopsworks and return the Feature Store handle.
    """
    env_api_key, env_project_name = get_hopsworks_credentials()
    api_key = api_key or env_api_key
    project_name = project_name or env_project_name

    log.info("Connecting to Hopsworks Project: %s …", project_name)
    project = hopsworks.login(
        project=project_name,
        api_key_value=api_key
    )
    fs = project.get_feature_store()
    log.info("✓ Connected to Hopsworks Feature Store successfully")
    return fs


# ------------------------------------------------------------------ #
#  FeatureStore Manager Class
# ------------------------------------------------------------------ #

class FeatureStoreManager:
    """High-level interface for managing Hopsworks Feature Groups & Views."""

    def __init__(
        self,
        project_name: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.api_key, self.project_name = get_hopsworks_credentials()
        if api_key:
            self.api_key = api_key
        if project_name:
            self.project_name = project_name

        self.project = hopsworks.login(
            project=self.project_name,
            api_key_value=self.api_key
        )
        self.fs: HSFSFeatureStore = self.project.get_feature_store()

    def get_or_create_feature_group(
        self,
        name: str,
        version: int = 1,
        description: str = "",
        primary_key: Optional[List[str]] = None,
        event_time: Optional[str] = "datetime",
        online_enabled: bool = False,
        time_travel_format: str = "HUDI",
    ) -> FeatureGroup:
        """
        Get an existing Feature Group or initialize a new one.
        """
        if primary_key is None:
            primary_key = ["datetime", "city"]

        try:
            fg = self.fs.get_feature_group(name=name, version=version)
            if fg is not None:
                log.info("✓ Retrieved existing Feature Group: '%s' (v%d)", name, version)
                return fg
        except Exception:
            pass

        try:
            fg = self.fs.get_or_create_feature_group(
                name=name,
                version=version,
                description=description,
                primary_key=primary_key,
                event_time=event_time,
                online_enabled=online_enabled,
                time_travel_format=time_travel_format,
            )
            log.info("✓ Initialized Feature Group: '%s' (v%d)", name, version)
            return fg
        except Exception as e:
            log.info("Creating Feature Group with create_feature_group: '%s' (v%d) …", name, version)
            fg = self.fs.create_feature_group(
                name=name,
                version=version,
                description=description,
                primary_key=primary_key,
                event_time=event_time,
                online_enabled=online_enabled,
                time_travel_format=time_travel_format,
            )
            log.info("✓ Created Feature Group: '%s' (v%d)", name, version)
            return fg

    def insert_features(
        self,
        df: pd.DataFrame,
        feature_group_name: str,
        version: int = 1,
        description: str = "Karachi AQI and Weather Features",
        primary_key: Optional[List[str]] = None,
        event_time: str = "datetime",
        online_enabled: bool = False,
        wait: bool = True,
        write_options: Optional[Dict[str, Any]] = None,
    ) -> FeatureGroup:
        """
        Sanitize, prepare, and insert features into a Hopsworks Feature Group.
        """
        df_to_upload = df.copy()

        # Add city column if not present
        if "city" not in [c.lower() for c in df_to_upload.columns]:
            city_val = os.getenv("CITY", "karachi").lower()
            df_to_upload["city"] = city_val

        # Sanitize column names for Hopsworks/Avro
        df_to_upload, _ = sanitize_column_names(df_to_upload)

        # Ensure datetime is formatted properly
        if event_time in df_to_upload.columns:
            df_to_upload[event_time] = pd.to_datetime(df_to_upload[event_time], utc=True)

        # Get or create feature group
        fg = self.get_or_create_feature_group(
            name=feature_group_name,
            version=version,
            description=description,
            primary_key=primary_key or ["datetime", "city"],
            event_time=event_time,
            online_enabled=online_enabled,
        )

        log.info(
            "Inserting %d rows × %d columns into Feature Group '%s' (v%d) …",
            len(df_to_upload), len(df_to_upload.columns), feature_group_name, version
        )

        # Insert DataFrame
        fg.insert(
            df_to_upload,
            wait=wait,
            write_options=write_options or {"wait_for_job": wait}
        )
        log.info("✓ Insert completed for Feature Group '%s' (v%d)", feature_group_name, version)
        return fg

    def read_features(
        self,
        feature_group_name: str,
        version: int = 1,
        read_options: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Read historical feature dataset from Hopsworks Feature Group.
        """
        fg = self.fs.get_feature_group(name=feature_group_name, version=version)
        log.info("Reading data from Feature Group '%s' (v%d) …", feature_group_name, version)
        df = fg.read(read_options=read_options)
        log.info("✓ Loaded %d rows × %d columns from Hopsworks", len(df), len(df.columns))
        return df

    def get_or_create_feature_view(
        self,
        name: str,
        query: Any,
        version: int = 1,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> FeatureView:
        """
        Get or create a Feature View based on a Feature Group query.
        """
        try:
            fv = self.fs.get_feature_view(name=name, version=version)
            log.info("✓ Retrieved existing Feature View: '%s' (v%d)", name, version)
            return fv
        except Exception:
            log.info("Creating new Feature View: '%s' (v%d) …", name, version)
            fv = self.fs.create_feature_view(
                name=name,
                version=version,
                query=query,
                labels=labels or ["target_pm25"],
                description=description,
            )
            log.info("✓ Initialized Feature View: '%s' (v%d)", name, version)
            return fv


# Alias for backward compatibility
FeatureStore = FeatureStoreManager