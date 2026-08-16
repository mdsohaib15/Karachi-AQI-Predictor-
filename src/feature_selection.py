"""
Feature Selection Pipeline
==========================
Analyzes feature correlation with target_pm25, checks for multicollinearity,
computes Mutual Information and tree feature importance, and selects
the optimal feature set for modeling.

Input:
    data/processed/karachi_features.csv

Output:
    data/processed/selected_features.txt
    data/processed/karachi_selected_features.csv

Run:
    python src/feature_selection.py
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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

FEATURES_PATH = Path("data/processed/karachi_features.csv")
OUTPUT_SELECTED_TXT = Path("data/processed/selected_features.txt")
OUTPUT_SELECTED_CSV = Path("data/processed/karachi_selected_features.csv")
TARGET_COL = "target_pm25"
TOP_K_DEFAULT = 18


# ------------------------------------------------------------------ #
#  Load Data
# ------------------------------------------------------------------ #

def load_feature_data(filepath: Path = FEATURES_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load engineered dataset and separate into index, X, and y."""
    log.info("Loading features from %s ...", filepath)
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    
    df_indexed = df.set_index("datetime")
    y = df_indexed[TARGET_COL]
    X = df_indexed.drop(columns=[TARGET_COL])
    
    log.info("  Total rows: %d | Total candidate features: %d", len(df), X.shape[1])
    return df, X, y


# ------------------------------------------------------------------ #
#  Correlation Analysis
# ------------------------------------------------------------------ #

def compute_correlations(df_indexed: pd.DataFrame, target: str = TARGET_COL) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson and Spearman correlation with target."""
    log.info("Computing Pearson & Spearman correlations with %s...", target)
    
    # Pearson
    pearson = df_indexed.corr(method="pearson")[target].drop(target)
    pearson_df = pd.DataFrame({
        "Feature": pearson.index,
        "Pearson_r": pearson.values,
        "Abs_Pearson_r": pearson.abs().values
    })
    
    # Spearman
    spearman = df_indexed.corr(method="spearman")[target].drop(target)
    spearman_df = pd.DataFrame({
        "Feature": spearman.index,
        "Spearman_rho": spearman.values,
        "Abs_Spearman_rho": spearman.abs().values
    })
    
    return pearson_df, spearman_df


# ------------------------------------------------------------------ #
#  Mutual Information
# ------------------------------------------------------------------ #

def compute_mutual_information(X: pd.DataFrame, y: pd.Series, sample_size: int = 5000) -> pd.DataFrame:
    """Compute Shannon mutual information regression scores."""
    log.info("Computing Mutual Information regression scores...")
    
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample, y_sample = X, y
        
    mi_scores = mutual_info_regression(X_sample, y_sample, random_state=42)
    mi_df = pd.DataFrame({
        "Feature": X.columns,
        "MI_Score": mi_scores
    })
    return mi_df


# ------------------------------------------------------------------ #
#  Variance Inflation Factor (VIF)
# ------------------------------------------------------------------ #

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute VIF on standardized predictors using Linear Regression R^2."""
    log.info("Computing Variance Inflation Factors (VIF)...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    vif_list = []
    features = list(X.columns)
    for col in features:
        X_other = X_scaled.drop(columns=[col])
        y_col = X_scaled[col]
        reg = LinearRegression().fit(X_other, y_col)
        r2 = reg.score(X_other, y_col)
        vif = 1.0 / (1.0 - r2) if r2 < 0.99999 else 10000.0
        vif_list.append({
            "Feature": col,
            "R_Squared": round(r2, 4),
            "VIF": round(vif, 2)
        })
    return pd.DataFrame(vif_list)


# ------------------------------------------------------------------ #
#  Tree-Based Feature Importance
# ------------------------------------------------------------------ #

def compute_tree_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compute average feature importance from RF and ExtraTrees."""
    log.info("Computing Tree-based Feature Importance (Random Forest & Extra Trees)...")
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    et = ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    et.fit(X, y)
    
    tree_df = pd.DataFrame({
        "Feature": X.columns,
        "RF_Importance": rf.feature_importances_,
        "ET_Importance": et.feature_importances_,
        "Avg_Tree_Importance": (rf.feature_importances_ + et.feature_importances_) / 2
    })
    return tree_df


# ------------------------------------------------------------------ #
#  Consolidate & Select Features
# ------------------------------------------------------------------ #

def consolidate_and_rank(
    pearson_df: pd.DataFrame,
    spearman_df: pd.DataFrame,
    mi_df: pd.DataFrame,
    tree_df: pd.DataFrame,
    vif_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge all statistical metrics into a composite weighted ranking."""
    master = pearson_df.merge(spearman_df, on="Feature") \
                       .merge(mi_df, on="Feature") \
                       .merge(tree_df, on="Feature") \
                       .merge(vif_df, on="Feature")
    
    # Min-max normalization for composite scoring
    master["Norm_Pearson"] = master["Abs_Pearson_r"] / master["Abs_Pearson_r"].max()
    master["Norm_Spearman"] = master["Abs_Spearman_rho"] / master["Abs_Spearman_rho"].max()
    master["Norm_MI"] = master["MI_Score"] / master["MI_Score"].max()
    master["Norm_Tree"] = master["Avg_Tree_Importance"] / master["Avg_Tree_Importance"].max()
    
    # Weighted composite score
    master["Composite_Score"] = (
        0.30 * master["Norm_Pearson"] +
        0.20 * master["Norm_Spearman"] +
        0.25 * master["Norm_MI"] +
        0.25 * master["Norm_Tree"]
    )
    
    master = master.sort_values(by="Composite_Score", ascending=False).reset_index(drop=True)
    master["Overall_Rank"] = master.index + 1
    return master


# ------------------------------------------------------------------ #
#  Main Execution
# ------------------------------------------------------------------ #

def run_feature_selection(top_k: int = TOP_K_DEFAULT) -> List[str]:
    """Execute feature selection pipeline and save results."""
    t0 = time.time()
    log.info("Starting feature selection pipeline...")
    
    df, X, y = load_feature_data(FEATURES_PATH)
    df_indexed = df.set_index("datetime")
    
    pearson_df, spearman_df = compute_correlations(df_indexed, TARGET_COL)
    mi_df = compute_mutual_information(X, y)
    tree_df = compute_tree_importance(X, y)
    vif_df = compute_vif(X)
    
    master_df = consolidate_and_rank(pearson_df, spearman_df, mi_df, tree_df, vif_df)
    
    selected_features = master_df.head(top_k)["Feature"].tolist()
    
    log.info("Top %d Selected Features:", top_k)
    for rank, feat in enumerate(selected_features, 1):
        comp = master_df.loc[master_df["Feature"] == feat, "Composite_Score"].values[0]
        p_corr = master_df.loc[master_df["Feature"] == feat, "Pearson_r"].values[0]
        vif = master_df.loc[master_df["Feature"] == feat, "VIF"].values[0]
        log.info("  %2d. %-24s | Composite: %.4f | Pearson r: %+.3f | VIF: %6.2f",
                 rank, feat, comp, p_corr, vif)
        
    # Save selected feature list
    OUTPUT_SELECTED_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SELECTED_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(selected_features))
    log.info("Saved selected features list -> %s", OUTPUT_SELECTED_TXT)
    
    # Save selected features dataframe
    selected_cols = ["datetime"] + selected_features + [TARGET_COL]
    df_selected = df[selected_cols]
    df_selected.to_csv(OUTPUT_SELECTED_CSV, index=False)
    log.info("Saved selected features dataset -> %s (%d columns, %d rows)",
             OUTPUT_SELECTED_CSV, df_selected.shape[1], len(df_selected))
    
    elapsed = time.time() - t0
    log.info("Feature selection completed in %.1f seconds.", elapsed)
    return selected_features


if __name__ == "__main__":
    run_feature_selection()
