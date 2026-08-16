"""
Automated Training Pipeline
===========================
Executes the scheduled / automated training pipeline:
1. Fetches historical features and targets from Hopsworks Feature Store
   (with automated fallback to local processed dataset).
2. Performs chronological 80/20 train/test split.
3. Experiments with candidate ML models (Ridge, Random Forest, Gradient Boosting, XGBoost).
4. Evaluates performance using RMSE, MAE, and R² metrics.
5. Saves and registers the champion model in the Model Registry.

Run:
    python pipelines/training_pipeline.py
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

# Ensure root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import get_feature_store, FeatureStoreManager
from src.model_registry import LocalModelRegistry, register_hopsworks_model
from src.model_training import (
    load_data,
    time_based_train_test_split,
    train_and_compare_models,
    tune_champion_model,
    scale_features,
    evaluate_predictions,
)

# ------------------------------------------------------------------ #
#  Logging & Setup
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

load_dotenv()

LOCAL_FEATURES_CSV = Path("data/processed/karachi_selected_features.csv")
FALLBACK_FEATURES_CSV = Path("data/processed/karachi_features.csv")
MODELS_DIR = Path("models")
TARGET_COL = "target_pm25"


# ------------------------------------------------------------------ #
#  Step 1: Fetch Data from Feature Store (or Local Fallback)
# ------------------------------------------------------------------ #

def fetch_training_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Fetch features and targets from Hopsworks Feature Store if available,
    otherwise load from local processed CSV.
    """
    log.info("Step 1: Fetching training dataset...")
    
    # Try fetching from Hopsworks
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if api_key:
        try:
            log.info("Attempting to read features from Hopsworks Feature Store...")
            fsm = FeatureStoreManager()
            fg = fsm.get_feature_group("karachi_aqi_features", version=1)
            df_hw = fg.read()
            if df_hw is not None and not df_hw.empty:
                log.info("✓ Successfully fetched %d rows from Hopsworks Feature Group.", len(df_hw))
                df_hw = df_hw.sort_values("datetime").reset_index(drop=True)
                df_indexed = df_hw.set_index("datetime")
                if TARGET_COL in df_indexed.columns:
                    y = df_indexed[TARGET_COL]
                    X = df_indexed.drop(columns=[TARGET_COL])
                    return df_hw, X, y
        except Exception as exc:
            log.warning("Could not fetch from Hopsworks (%s). Falling back to local CSV.", exc)

    # Local fallback
    data_path = LOCAL_FEATURES_CSV if LOCAL_FEATURES_CSV.exists() else FALLBACK_FEATURES_CSV
    log.info("Loading from local storage: %s", data_path)
    return load_data(data_path)


# ------------------------------------------------------------------ #
#  Step 2: Train & Evaluate Multiple Candidate Models
# ------------------------------------------------------------------ #

def execute_model_experiments(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Train candidate models and output comparative performance summary.
    """
    log.info("Step 2 & 3: Training candidate ML models and evaluating metrics...")
    results_df, trained_models, champion_name = train_and_compare_models(
        X_train, X_test, y_train, y_test
    )
    return results_df, trained_models, champion_name


# ------------------------------------------------------------------ #
#  Step 3: Register Champion Model
# ------------------------------------------------------------------ #

def register_champion(
    champion_name: str,
    trained_models: Dict[str, Any],
    feature_names: List[str],
    X_test: pd.DataFrame,
) -> Path:
    """
    Save champion model locally and register with Hopsworks Model Registry.
    """
    log.info("Step 4: Registering champion model in Model Registry...")
    champ_bundle = trained_models[champion_name]
    model = champ_bundle["model"]
    scaler = champ_bundle["scaler"]
    test_metrics = champ_bundle["test_metrics"]

    # 1. Local registry
    registry = LocalModelRegistry(MODELS_DIR)
    saved_dir = registry.save_model(
        model=model,
        model_name=champion_name,
        metrics=test_metrics,
        feature_names=feature_names,
        scaler=scaler,
        is_champion=True,
    )

    # 2. Hopsworks Model Registry upload
    register_hopsworks_model(
        model=model,
        model_name=champion_name,
        metrics=test_metrics,
        input_example=X_test.head(5),
        description=f"Champion Karachi PM2.5 Model - {champion_name} (Test RMSE: {test_metrics['RMSE']}, R²: {test_metrics['R2']})",
    )

    return saved_dir


# ------------------------------------------------------------------ #
#  Main Execution Flow
# ------------------------------------------------------------------ #

def main():
    t_start = time.time()
    log.info("=" * 65)
    log.info("   KARACHI AQI PREDICTOR - DAILY TRAINING PIPELINE   ")
    log.info("=" * 65)

    # 1. Fetch data
    df, X, y = fetch_training_dataset()

    # 2. Split data chronologically
    X_train, X_test, y_train, y_test = time_based_train_test_split(X, y, test_ratio=0.20)

    # 3. Model experimentation
    results_df, trained_models, champion_name = execute_model_experiments(
        X_train, X_test, y_train, y_test
    )

    log.info("\n=== MODEL COMPARISON LEADERBOARD ===\n%s\n", results_df.to_string(index=False))

    # 4. Register champion
    saved_path = register_champion(
        champion_name=champion_name,
        trained_models=trained_models,
        feature_names=list(X.columns),
        X_test=X_test,
    )

    elapsed = time.time() - t_start
    log.info("=" * 65)
    log.info("✓ Training Pipeline completed successfully in %.2f seconds.", elapsed)
    log.info("✓ Best Model: %s -> %s", champion_name, saved_path)
    log.info("=" * 65)


if __name__ == "__main__":
    main()