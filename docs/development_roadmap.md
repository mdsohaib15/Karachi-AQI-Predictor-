# Karachi AQI Predictor



|                    Path                    | Purpose                                                                          |
| :-------------------------------------------: | ---------------------------------------------------------------------------------- |
|  `.github/workflows/feature_pipeline.yml`  | scheduled hourly run of`pipelines/feature_pipeline.py`                           |
|     `.github/workflows/prediction.yml`     | scheduled run of`src/predict.py` to generate fresh forecasts                     |
|  `.github/workflows/training_pipeline.yml`  | scheduled daily run of`pipelines/training_pipeline.py`                           |
|           `app/streamlit_app.py`           | dashboard UI                                                                     |
|           `app/explainability.py`           | SHAP feature-importance visuals for the dashboard                                |
|             `config/config.yml`             | city, coordinates, paths, thresholds                                             |
|                 `data/raw/`                 | untouched API pulls                                                              |
|              `data/processed/`              | cleaned, feature-engineered data                                                 |
|                   `docs/`                   | final report, diagrams                                                           |
|                  `models/`                  | saved trained model files                                                        |
| `notebooks/exploratory_data_analysis.ipynb` | EDA — trends, seasonality, correlations                                         |
|     `notebooks/feature_selection.ipynb`     | testing which features matter                                                    |
|      `notebooks/model_training.ipynb`      | experimentation before promoting code to`src/`                                   |
|       `pipelines/feature_pipeline.py`       | orchestrates: fetch → clean → extract → store                                 |
|      `pipelines/training_pipeline.py`      | orchestrates: read features → train → evaluate → register                     |
|             `src/api_client.py`             | wraps external weather/pollution API calls                                       |
|         `src/fetch_current_aqi.py`         | pulls the live current reading                                                   |
|        `src/backfill_historical.py`        | pulls historical data for training                                               |
|           `src/data_cleaning.py`           | validation/cleaning logic                                                        |
|         `src/feature_extraction.py`         | raw data → model features                                                       |
|           `src/feature_store.py`           | generic read/write wrapper for the feature store                                 |
|             `src/upload_raw.py`             | pushes raw pulled data to storage                                                |
|        `src/upload_to_hopsworks.py`        | pushes processed features to Hopsworks specifically                              |
|           `src/model_training.py`           | training logic (called by`pipelines/training_pipeline.py`)                       |
|           `src/model_registry.py`           | save/load/version models                                                         |
|              `src/predict.py`              | loads latest model + features, produces the forecast (called by`prediction.yml`) |
|                   `venv/`                   | virtual environment (should be gitignored)                                       |
|                   `.env`                   | secrets (API keys, Hopsworks credentials)                                        |
|                `.gitignore`                | exclusions                                                                       |
|             `requirements.txt`             | dependencies                                                                     |
|                 `README.md`                 | setup/usage docs                                                                 |
