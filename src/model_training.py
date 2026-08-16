"""
Model Training Module
=====================
Trains and evaluates multiple regression models for next-hour PM2.5 prediction.

Candidate Models:
    1. Linear Regression
    2. Ridge Regression
    3. Lasso Regression
    4. Random Forest Regressor
    5. Gradient Boosting Regressor
    6. XGBoost Regressor

Metrics:
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - Coefficient of Determination (R²)

Run:
    python src/model_training.py
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.model_registry import LocalModelRegistry, save_model

# ------------------------------------------------------------------ #
#  Logging Configuration
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/karachi_selected_features.csv")
FALLBACK_RAW_PATH = Path("data/processed/karachi_features.csv")
MODELS_DIR = Path("models")
TARGET_COL = "target_pm25"


# ------------------------------------------------------------------ #
#  Data Loading & Preparation
# ------------------------------------------------------------------ #

def load_data(filepath: Path = DATA_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load dataset, sort by datetime, and split into predictors (X) and target (y).
    """
    if not filepath.exists():
        if FALLBACK_RAW_PATH.exists():
            log.warning("Selected features file %s not found. Using %s instead.", filepath, FALLBACK_RAW_PATH)
            filepath = FALLBACK_RAW_PATH
        else:
            raise FileNotFoundError(f"Neither {filepath} nor {FALLBACK_RAW_PATH} exists.")

    log.info("Loading training data from %s ...", filepath)
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from {filepath}")

    # Set datetime as index
    df_indexed = df.set_index("datetime")
    y = df_indexed[TARGET_COL]
    X = df_indexed.drop(columns=[TARGET_COL])

    log.info("  Rows loaded : %d", len(df))
    log.info("  Features (%d): %s", X.shape[1], list(X.columns))
    log.info("  Date range  : %s  -->  %s", df["datetime"].min(), df["datetime"].max())

    return df, X, y


# ------------------------------------------------------------------ #
#  Time-Series Train / Test Split
# ------------------------------------------------------------------ #

def time_based_train_test_split(
    X: pd.DataFrame, y: pd.Series, test_ratio: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform chronological train/test split to prevent temporal data leakage.
    """
    split_idx = int(len(X) * (1 - test_ratio))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    log.info("Chronological Split (%d%% train, %d%% test):", int((1 - test_ratio) * 100), int(test_ratio * 100))
    log.info("  Train set: %d rows (%s -> %s)", len(X_train), X_train.index.min(), X_train.index.max())
    log.info("  Test set : %d rows (%s -> %s)", len(X_test), X_test.index.min(), X_test.index.max())

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------ #
#  Feature Scaling
# ------------------------------------------------------------------ #

def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit standard scaler on training features and transform test features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ------------------------------------------------------------------ #
#  Evaluation Metrics Calculation
# ------------------------------------------------------------------ #

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE, MAE, and R² scores.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(rmse, 4),
        "R2": round(float(r2), 4),
    }


# ------------------------------------------------------------------ #
#  Model Dictionary Definition
# ------------------------------------------------------------------ #

def get_candidate_models() -> Dict[str, Dict[str, Any]]:
    """
    Returns candidate models and indicates whether feature scaling is required.
    """
    return {
        "Linear Regression": {
            "model": LinearRegression(),
            "needs_scaling": True,
        },
        "Ridge Regression": {
            "model": Ridge(alpha=1.0, random_state=42),
            "needs_scaling": True,
        },
        "Lasso Regression": {
            "model": Lasso(alpha=0.01, random_state=42, max_iter=2000),
            "needs_scaling": True,
        },
        "Random Forest": {
            "model": RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_split=4, random_state=42, n_jobs=-1),
            "needs_scaling": False,
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42),
            "needs_scaling": False,
        },
        "XGBoost": {
            "model": XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, subsample=0.85, random_state=42, n_jobs=-1),
            "needs_scaling": False,
        },
    }


# ------------------------------------------------------------------ #
#  Train & Compare Multiple Models
# ------------------------------------------------------------------ #

def train_and_compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Train all candidate models and compile evaluation summary.
    """
    log.info("Starting candidate model experimentation...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    models_dict = get_candidate_models()
    results = []
    trained_models = {}

    for name, config in models_dict.items():
        t0 = time.time()
        model = config["model"]
        use_scaled = config["needs_scaling"]

        # Select data representation
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        # Fit model
        model.fit(X_tr, y_train)
        elapsed = time.time() - t0

        # Predict
        y_train_pred = model.predict(X_tr)
        y_test_pred = model.predict(X_te)

        train_metrics = evaluate_predictions(y_train.values, y_train_pred)
        test_metrics = evaluate_predictions(y_test.values, y_test_pred)

        results.append({
            "Model": name,
            "Train_RMSE": train_metrics["RMSE"],
            "Test_RMSE": test_metrics["RMSE"],
            "Train_MAE": train_metrics["MAE"],
            "Test_MAE": test_metrics["MAE"],
            "Train_R2": train_metrics["R2"],
            "Test_R2": test_metrics["R2"],
            "Overfit_Gap_R2": round(train_metrics["R2"] - test_metrics["R2"], 4),
            "Train_Time_Sec": round(elapsed, 2),
            "Needs_Scaling": use_scaled,
        })
        trained_models[name] = {
            "model": model,
            "scaler": scaler if use_scaled else None,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }

        log.info("  ✓ %-20s | Test RMSE: %6.2f | Test MAE: %6.2f | Test R²: %.4f | Time: %.2fs",
                 name, test_metrics["RMSE"], test_metrics["MAE"], test_metrics["R2"], elapsed)

    results_df = pd.DataFrame(results).sort_values(by="Test_RMSE", ascending=True).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]
    log.info("Champion Model Selected: %s (Test RMSE: %.4f, Test R²: %.4f)",
             best_model_name, results_df.iloc[0]["Test_RMSE"], results_df.iloc[0]["Test_R2"])

    return results_df, trained_models, best_model_name


# ------------------------------------------------------------------ #
#  Hyperparameter Tuning (TimeSeriesSplit CV)
# ------------------------------------------------------------------ #

def tune_champion_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 15,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform TimeSeriesSplit cross-validated hyperparameter optimization.
    """
    log.info("Running hyperparameter tuning for %s...", model_name)
    tscv = TimeSeriesSplit(n_splits=4)

    if model_name == "XGBoost":
        base_estimator = XGBRegressor(random_state=42, n_jobs=-1)
        param_dist = {
            "n_estimators": [100, 150, 200, 250],
            "max_depth": [4, 5, 6, 8],
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "subsample": [0.75, 0.85, 0.95],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [1.0, 3.0, 5.0],
        }
    elif model_name == "Random Forest":
        base_estimator = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_dist = {
            "n_estimators": [100, 150, 200],
            "max_depth": [10, 14, 18, None],
            "min_samples_split": [2, 4, 8],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.8, 1.0],
        }
    else:
        # For linear models, tune alpha
        base_estimator = Ridge(random_state=42)
        param_dist = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    log.info("  ✓ Best CV RMSE: %.4f", -search.best_score_)
    log.info("  ✓ Best Parameters: %s", search.best_params_)

    return search.best_estimator_, search.best_params_


# ------------------------------------------------------------------ #
#  Complete Training Pipeline Flow
# ------------------------------------------------------------------ #

def run_training_pipeline(
    data_path: Path = DATA_PATH,
    tune: bool = False,
    save_champion: bool = True,
) -> Tuple[pd.DataFrame, str, Dict[str, Any]]:
    """
    Orchestrates the entire training workflow:
    1. Load data
    2. Temporal split
    3. Candidate models training & evaluation
    4. (Optional) Fine-tuning
    5. Model registration
    """
    df, X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = time_based_train_test_split(X, y, test_ratio=0.20)

    # Train and compare
    results_df, trained_models, champion_name = train_and_compare_models(X_train, X_test, y_train, y_test)

    champ_bundle = trained_models[champion_name]
    final_model = champ_bundle["model"]
    final_scaler = champ_bundle["scaler"]
    final_metrics = champ_bundle["test_metrics"]
    best_params = {}

    # Optional tuning
    if tune and champion_name in ["XGBoost", "Random Forest"]:
        final_model, best_params = tune_champion_model(champion_name, X_train, y_train)
        y_test_pred = final_model.predict(X_test)
        final_metrics = evaluate_predictions(y_test.values, y_test_pred)
        log.info("Post-Tuning Test Metrics for %s: %s", champion_name, final_metrics)

    # Save champion to Model Registry
    if save_champion:
        save_model(
            model=final_model,
            model_name=champion_name,
            metrics=final_metrics,
            feature_names=list(X.columns),
            scaler=final_scaler,
            hyperparameters=best_params,
            output_dir=MODELS_DIR,
        )

    return results_df, champion_name, final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models for Karachi AQI prediction")
    parser.add_argument("--data", type=str, default=str(DATA_PATH), help="Path to features CSV")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning on best model")
    args = parser.parse_args()

    run_training_pipeline(data_path=Path(args.data), tune=args.tune, save_champion=True)
