"""
Explainability Engine (SHAP & Feature Attribution)
===================================================
Provides interpretability for AQI predictions:
- SHAP TreeExplainer for local feature contributions on any prediction
- Global feature importance analysis
- Natural language automated interpretations explaining why AQI is rising or falling
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

from src.predict import AQIPredictor

log = logging.getLogger(__name__)

SELECTED_FEATURES_CSV = Path("data/processed/karachi_selected_features.csv")


class AQIExplainer:
    """Computes SHAP explanations and plain-English interpretations."""

    def __init__(self, predictor: Optional[AQIPredictor] = None):
        self.predictor = predictor or AQIPredictor()
        self.model = self.predictor.model
        self.features = self.predictor.features
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initialize SHAP TreeExplainer with sample background data."""
        try:
            # Tree-based models (GradientBoosting, RandomForest, XGBoost)
            if hasattr(self.model, "estimators_") or "XGB" in type(self.model).__name__:
                self.explainer = shap.TreeExplainer(self.model)
                log.info("✓ Initialized SHAP TreeExplainer")
            else:
                # Fallback to linear or KernelExplainer
                if SELECTED_FEATURES_CSV.exists():
                    bg_df = pd.read_csv(SELECTED_FEATURES_CSV)
                    X_bg = bg_df[self.features].head(100)
                    self.explainer = shap.LinearExplainer(self.model, X_bg)
                    log.info("✓ Initialized SHAP LinearExplainer")
        except Exception as e:
            log.warning("SHAP explainer initialization note: %s. Using surrogate attribution.", e)
            self.explainer = None

    def explain_instance(self, input_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute local feature attribution values for a single input record.
        """
        X = input_df[self.features].copy()
        
        # Scale if required
        if self.predictor.scaler is not None:
            X_eval = self.predictor.scaler.transform(X)
        else:
            X_eval = X

        pred_val = float(self.predictor.predict(input_df)["predicted_target_pm25"].iloc[0])
        
        feature_contributions = []
        base_value = 25.0

        if self.explainer is not None:
            try:
                shap_values = self.explainer(X_eval)
                if hasattr(shap_values, "values"):
                    sv = shap_values.values[0]
                    base_val = float(shap_values.base_values[0]) if hasattr(shap_values, "base_values") else 25.0
                else:
                    sv = shap_values[0]
                    base_val = float(self.explainer.expected_value) if hasattr(self.explainer, "expected_value") else 25.0

                base_value = base_val
                for feat_name, shap_val, feat_val in zip(self.features, sv, X.iloc[0]):
                    feature_contributions.append({
                        "feature": feat_name,
                        "value": float(feat_val),
                        "shap_value": round(float(shap_val), 3),
                        "direction": "Increases PM2.5" if shap_val > 0 else "Decreases PM2.5",
                        "abs_impact": abs(float(shap_val)),
                    })
            except Exception as exc:
                log.warning("SHAP compute failed (%s). Falling back to tree feature attribution.", exc)

        # Fallback to model feature importances if SHAP calculation had issues
        if not feature_contributions:
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                diff = pred_val - 25.0
                for feat_name, imp, feat_val in zip(self.features, importances, X.iloc[0]):
                    shap_val = imp * diff
                    feature_contributions.append({
                        "feature": feat_name,
                        "value": float(feat_val),
                        "shap_value": round(float(shap_val), 3),
                        "direction": "Increases PM2.5" if shap_val > 0 else "Decreases PM2.5",
                        "abs_impact": abs(float(shap_val)),
                    })

        # Sort by absolute impact
        feature_contributions.sort(key=lambda x: x["abs_impact"], reverse=True)

        # Human-friendly feature name mapping
        friendly_names = {
            "PM2.5": "Current PM2.5 Level",
            "pm2_5": "Current PM2.5 Level",
            "pm25_roll_mean_24": "24h Avg PM2.5",
            "pm25_roll_mean_72": "72h Avg PM2.5",
            "pm25_lag_24": "Yesterday's PM2.5 (24h lag)",
            "PM10": "PM10 Coarse Dust",
            "pm10": "PM10 Coarse Dust",
            "CO": "Carbon Monoxide (CO)",
            "co": "Carbon Monoxide (CO)",
            "pm25_roll_std_24": "24h PM2.5 Variability",
            "pm25_roll_std_72": "72h PM2.5 Variability",
            "SO2": "Sulfur Dioxide (SO2)",
            "so2": "Sulfur Dioxide (SO2)",
            "NO2": "Nitrogen Dioxide (NO2)",
            "no2": "Nitrogen Dioxide (NO2)",
            "wind_pm25": "Wind × Pollution Dispersion",
            "pm25_lag_48": "48h Lagged PM2.5",
            "pm25_lag_72": "72h Lagged PM2.5",
            "month_cos": "Seasonal Cycle (Month)",
            "temp_humidity": "Heat & Humidity Trapping",
            "surface_pressure": "Barometric Pressure",
            "apparent_temperature": "Feels-Like Temperature",
            "wind_speed_10m": "Surface Wind Speed",
        }

        for item in feature_contributions:
            item["display_name"] = friendly_names.get(item["feature"], item["feature"])

        # Generate narrative text
        narrative = self._generate_narrative(pred_val, base_value, feature_contributions)

        return {
            "predicted_pm25": pred_val,
            "base_value": round(base_value, 2),
            "contributions": feature_contributions,
            "narrative_explanation": narrative,
        }

    def _generate_narrative(
        self,
        predicted_pm25: float,
        base_value: float,
        contributions: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable AI explanation of the forecast."""
        top_positive = [c for c in contributions if c["shap_value"] > 0][:2]
        top_negative = [c for c in contributions if c["shap_value"] < 0][:2]

        lines = [
            f"The AI model predicts a PM2.5 concentration of **{predicted_pm25:.1f} µg/m³** "
            f"(baseline expectation is {base_value:.1f} µg/m³)."
        ]

        if top_positive:
            reasons_up = ", ".join([f"**{c['display_name']}** (+{c['shap_value']:.2f})" for c in top_positive])
            lines.append(f"• **Key factors elevating pollution**: {reasons_up}.")

        if top_negative:
            reasons_down = ", ".join([f"**{c['display_name']}** ({c['shap_value']:.2f})" for c in top_negative])
            lines.append(f"• **Key factors mitigating pollution**: {reasons_down}.")

        return "\n\n".join(lines)

    def get_global_importance(self) -> pd.DataFrame:
        """Return global feature importance ranking."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            df_imp = pd.DataFrame({
                "feature": self.features,
                "importance": importances,
            }).sort_values("importance", ascending=False).reset_index(drop=True)
            df_imp["relative_importance_pct"] = (df_imp["importance"] / df_imp["importance"].sum()) * 100
            return df_imp

        # If linear model
        if hasattr(self.model, "coef_"):
            coefs = np.abs(self.model.coef_)
            df_imp = pd.DataFrame({
                "feature": self.features,
                "importance": coefs,
            }).sort_values("importance", ascending=False).reset_index(drop=True)
            df_imp["relative_importance_pct"] = (df_imp["importance"] / df_imp["importance"].sum()) * 100
            return df_imp

        return pd.DataFrame({"feature": self.features, "importance": [1.0] * len(self.features)})