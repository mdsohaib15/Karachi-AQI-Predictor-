"""
Model Registry Module
======================
Handles model saving, loading, versioning, and metadata tracking.
Supports both local file-based storage and Hopsworks Model Registry.

Functions:
    - save_model_locally
    - load_model_locally
    - register_hopsworks_model
    - save_metrics_report
    - get_latest_model_metadata
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ------------------------------------------------------------------ #
#  Logging & Environment
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

load_dotenv()

DEFAULT_MODELS_DIR = Path("models")


# ------------------------------------------------------------------ #
#  Local Model Registry
# ------------------------------------------------------------------ #

class LocalModelRegistry:
    """Manages saving, loading, and versioning models locally."""

    def __init__(self, models_dir: Union[str, Path] = DEFAULT_MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        feature_names: List[str],
        scaler: Optional[Any] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        is_champion: bool = True,
    ) -> Path:
        """
        Save model artifacts, scaler, feature list, and metadata locally.

        Args:
            model: Trained estimator object
            model_name: Model identifier (e.g. 'RandomForest', 'XGBoost')
            metrics: Dictionary of evaluation metrics (RMSE, MAE, R2)
            feature_names: List of predictor feature names
            scaler: Optional fitted StandardScaler/RobustScaler
            hyperparameters: Model configuration parameters
            version: Optional custom version string (defaults to timestamp)
            is_champion: Whether this model should be saved as default champion

        Returns:
            Path to the saved model bundle directory
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_slug = model_name.lower().replace(" ", "_")
        version_dir = self.models_dir / f"{model_slug}_v_{version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save model object
        model_file = version_dir / "model.pkl"
        joblib.dump(model, model_file)

        # 2. Save scaler if present
        if scaler is not None:
            scaler_file = version_dir / "scaler.pkl"
            joblib.dump(scaler, scaler_file)

        # 3. Save feature list
        features_file = version_dir / "features.json"
        with open(features_file, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, indent=2)

        # 4. Save metadata and metrics
        metadata = {
            "model_name": model_name,
            "version": version,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "metrics": metrics,
            "num_features": len(feature_names),
            "features": feature_names,
            "hyperparameters": hyperparameters or {},
            "is_champion": is_champion,
        }
        meta_file = version_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 5. If champion, also save/update root champions in models/
        if is_champion:
            champ_model_path = self.models_dir / "best_model.pkl"
            champ_meta_path = self.models_dir / "best_model_metadata.json"
            champ_feat_path = self.models_dir / "selected_features.json"
            
            joblib.dump(model, champ_model_path)
            with open(champ_meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            with open(champ_feat_path, "w", encoding="utf-8") as f:
                json.dump(feature_names, f, indent=2)
                
            if scaler is not None:
                champ_scaler_path = self.models_dir / "scaler.pkl"
                joblib.dump(scaler, champ_scaler_path)

        log.info("✓ Saved model [%s v%s] locally to %s", model_name, version, version_dir)
        return version_dir

    def load_model(
        self,
        model_dir_or_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Load model artifact, scaler, and metadata.
        If no path is provided, loads the default champion model.

        Returns:
            Dictionary with keys: 'model', 'scaler', 'features', 'metadata'
        """
        if model_dir_or_path is None:
            # Load default best model
            model_path = self.models_dir / "best_model.pkl"
            meta_path = self.models_dir / "best_model_metadata.json"
            scaler_path = self.models_dir / "scaler.pkl"
            feat_path = self.models_dir / "selected_features.json"
        else:
            p = Path(model_dir_or_path)
            if p.is_file():
                model_path = p
                version_dir = p.parent
            else:
                model_path = p / "model.pkl"
                version_dir = p
            meta_path = version_dir / "metadata.json"
            scaler_path = version_dir / "scaler.pkl"
            feat_path = version_dir / "features.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        
        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        features = []
        if feat_path.exists():
            with open(feat_path, "r", encoding="utf-8") as f:
                features = json.load(f)

        return {
            "model": model,
            "scaler": scaler,
            "features": features,
            "metadata": metadata,
        }


# ------------------------------------------------------------------ #
#  Hopsworks Model Registry Integration
# ------------------------------------------------------------------ #

def register_hopsworks_model(
    model: Any,
    model_name: str,
    metrics: Dict[str, float],
    input_example: Optional[pd.DataFrame] = None,
    description: Optional[str] = None,
) -> bool:
    """
    Register and upload the trained model into Hopsworks Model Registry.
    Gracefully falls back to local registry if Hopsworks is not configured.
    """
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "Karachi_AQI_Predictors")

    if not api_key:
        log.warning("HOPSWORKS_API_KEY not found. Skipping Hopsworks model registry upload.")
        return False

    try:
        import hopsworks

        log.info("Connecting to Hopsworks project '%s' for model registration...", project_name)
        project = hopsworks.login(api_key_value=api_key, project=project_name)
        mr = project.get_model_registry()

        temp_dir = Path("models/temp_hopsworks_upload")
        temp_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, temp_dir / "model.pkl")

        # Define model schema
        from hsml.schema.model_schema import ModelSchema
        from hsml.schema.schema import Schema

        model_schema = None
        if input_example is not None:
            input_schema = Schema(input_example)
            model_schema = ModelSchema(input_schema=input_schema)

        # Clean metric keys for Hopsworks (lowercase, no special chars)
        hw_metrics = {k.lower().replace("^2", "2").replace(" ", "_"): float(v) for k, v in metrics.items()}

        hw_model = mr.python.create_model(
            name=model_name.lower().replace(" ", "_"),
            metrics=hw_metrics,
            model_schema=model_schema,
            description=description or f"Karachi PM2.5 Hourly Predictor - {model_name}",
        )

        hw_model.save(str(temp_dir))
        log.info("✓ Successfully registered model to Hopsworks Model Registry (version %s)", hw_model.version)

        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        return True

    except Exception as exc:
        log.warning("Could not register model in Hopsworks Model Registry: %s", exc)
        return False


# ------------------------------------------------------------------ #
#  Convenience Helpers
# ------------------------------------------------------------------ #

def save_model(
    model: Any,
    model_name: str,
    metrics: Dict[str, float],
    feature_names: List[str],
    scaler: Optional[Any] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    output_dir: Union[str, Path] = DEFAULT_MODELS_DIR,
) -> Path:
    """Save model to local registry and attempt Hopsworks registration."""
    registry = LocalModelRegistry(output_dir)
    saved_path = registry.save_model(
        model=model,
        model_name=model_name,
        metrics=metrics,
        feature_names=feature_names,
        scaler=scaler,
        hyperparameters=hyperparameters,
        is_champion=True,
    )
    register_hopsworks_model(model=model, model_name=model_name, metrics=metrics)
    return saved_path


def load_best_model(models_dir: Union[str, Path] = DEFAULT_MODELS_DIR) -> Dict[str, Any]:
    """Load the champion model package."""
    registry = LocalModelRegistry(models_dir)
    return registry.load_model()