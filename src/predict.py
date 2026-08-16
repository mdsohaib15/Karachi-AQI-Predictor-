"""
Inference & Prediction Module
==============================
Loads the registered champion model and generates next-hour PM2.5
and AQI category predictions from single records or batch datasets.

Run:
    python src/predict.py --sample
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    from src.model_registry import LocalModelRegistry
except ImportError:
    from model_registry import LocalModelRegistry

# ------------------------------------------------------------------ #
#  Logging Configuration
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")


# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
#  AQI Category Classification Helper
# ------------------------------------------------------------------ #

class AQIInfo(tuple):
    """Tuple subclass supporting both tuple unpacking (cat, col, adv) and dict-like key access."""

    def __new__(cls, category: str, color: str, health_advisory: str):
        return super().__new__(cls, (category, color, health_advisory))

    @property
    def category(self) -> str:
        return self[0]

    @property
    def color(self) -> str:
        return self[1]

    @property
    def health_advisory(self) -> str:
        return self[2]

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in ("category", "aqi_category"):
                return self[0]
            elif item in ("color", "color_code"):
                return self[1]
            elif item == "health_advisory":
                return self[2]
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, item, default=None):
        try:
            return self[item]
        except (KeyError, IndexError):
            return default


def get_aqi_category(pm25_value: float) -> AQIInfo:
    """
    Map PM2.5 concentration (ug/m3) to EPA AQI Risk Category & Color.
    Returns AQIInfo supporting both tuple unpacking: (category, color, advisory)
    and dict access: info['category'], info['color'], info['health_advisory'].
    """
    if pm25_value <= 12.0:
        return AQIInfo("Good", "#00e400", "Air quality is considered satisfactory.")
    elif pm25_value <= 35.4:
        return AQIInfo("Moderate", "#f1c40f", "Air quality is acceptable; moderate health concern for sensitive individuals.")
    elif pm25_value <= 55.4:
        return AQIInfo("Unhealthy for Sensitive Groups", "#ff7e00", "Members of sensitive groups may experience health effects.")
    elif pm25_value <= 150.4:
        return AQIInfo("Unhealthy", "#ff0000", "Everyone may begin to experience health effects.")
    elif pm25_value <= 250.4:
        return AQIInfo("Very Unhealthy", "#8f3f97", "Health alert: everyone may experience more serious health effects.")
    else:
        return AQIInfo("Hazardous", "#7e0023", "Health warnings of emergency conditions.")


# ------------------------------------------------------------------ #
#  Predictor Class
# ------------------------------------------------------------------ #

class AQIPredictor:
    """Production predictor wrapper for Karachi PM2.5 forecasting."""

    def __init__(self, models_dir: Union[str, Path] = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.registry = LocalModelRegistry(self.models_dir)
        
        # Load model bundle
        bundle = self.registry.load_model()
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.features = bundle["features"]
        self.metadata = bundle["metadata"]

        log.info("✓ Initialized AQIPredictor with model: %s", self.metadata.get("model_name", "Unknown"))
        log.info("  Features required (%d): %s", len(self.features), self.features)

    def _prepare_features(self, input_df: pd.DataFrame) -> np.ndarray:
        """Validate input columns and scale if scaler exists."""
        missing = [f for f in self.features if f not in input_df.columns]
        if missing:
            raise ValueError(f"Input data missing required features: {missing}")

        X = input_df[self.features]
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def predict(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """
        Generate next-hour PM2.5 predictions and risk categories.

        Args:
            data: Single record dictionary or pandas DataFrame

        Returns:
            DataFrame with predictions and health advisory details
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        X_prepared = self._prepare_features(df)
        preds = self.model.predict(X_prepared)
        # Ensure non-negative predictions for physical pollutant concentrations
        preds = np.clip(preds, a_min=0.0, a_max=None)

        results = []
        for i, pred_val in enumerate(preds):
            aqi_info = get_aqi_category(pred_val)
            res_row = {
                "predicted_target_pm25": round(float(pred_val), 2),
                "aqi_category": aqi_info["category"],
                "color_code": aqi_info["color"],
                "health_advisory": aqi_info["health_advisory"],
            }
            if "datetime" in df.columns:
                res_row["timestamp"] = str(df.iloc[i]["datetime"])
            results.append(res_row)

        return pd.DataFrame(results)


# ------------------------------------------------------------------ #
#  CLI Usage
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PM2.5 predictions")
    parser.add_argument("--sample", action="store_true", help="Run sample prediction on test data")
    parser.add_argument("--file", type=str, help="Path to input features CSV file")
    args = parser.parse_args()

    predictor = AQIPredictor()

    if args.sample or args.file is None:
        sample_path = Path("data/processed/karachi_selected_features.csv")
        if sample_path.exists():
            df_sample = pd.read_csv(sample_path).tail(5)
            log.info("Running prediction on last 5 rows of %s:", sample_path)
            out = predictor.predict(df_sample)
            print("\n=== SAMPLE PREDICTIONS ===")
            print(out.to_string(index=False))
        else:
            log.warning("Sample file %s not found.", sample_path)
    elif args.file:
        df_input = pd.read_csv(args.file)
        out = predictor.predict(df_input)
        print(out.head(10).to_string(index=False))
