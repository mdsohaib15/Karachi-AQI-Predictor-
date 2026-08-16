# 🌫️ Karachi AQI Predictor

Real-time Air Quality Index prediction for Karachi using Machine Learning, powered by OpenWeather API, Hopsworks Feature Store, and FastAPI + Streamlit.

---

## 🏗️ Architecture

```
                    OpenWeather API
                   (Weather + AQI)
                           │
                           ▼
                 Feature Pipeline
                           │
                           ▼
                Feature Engineering
                           │
                           ▼
                  Hopsworks Feature Store
                           │
               ┌───────────┴───────────┐
               │                       │
               ▼                       ▼
      Training Pipeline         Prediction Pipeline
               │                       │
               ▼                       ▼
        Train Best Model        Load Latest Features
               │                       │
               ▼                       ▼
      Hopsworks Model Registry   Predict Next 3 Days
               │                       │
               └───────────┬───────────┘
                           ▼
                        FastAPI
                           │
                           ▼
                      Streamlit UI
                           │
                           ▼
                     GitHub Actions
```

---

## 🛠️ Tech Stack

| Component           | Technology                                   |
|---------------------|----------------------------------------------|
| Language            | Python 3.12                                  |
| Data Collection     | OpenWeather API (Weather + Air Pollution)    |
| Feature Store       | Hopsworks                                    |
| Model Registry      | Hopsworks                                    |
| ML Models           | Scikit-Learn (Random Forest, Ridge), XGBoost |
| Explainability      | SHAP                                         |
| Backend API         | FastAPI                                      |
| Dashboard           | Streamlit                                    |
| Automation          | GitHub Actions                               |
| Version Control     | Git + GitHub                                 |

---

## 📂 Project Structure

```
aqi-predictor/
├── api/                    # FastAPI backend
│   ├── main.py
│   ├── routes.py
│   ├── schemas.py
│   └── prediction.py
├── app/                    # Streamlit frontend
│   ├── streamlit_app.py
│   ├── dashboard.py
│   ├── components.py
│   ├── charts.py
│   └── explainability.py
├── configs/                # Configuration
│   ├── config.py
│   ├── settings.yaml
│   └── constants.py
├── data/                   # Data files
│   ├── raw/
│   ├── processed/
│   └── predictions/
├── feature_store/          # Hopsworks feature group scripts
│   ├── create_feature_group.py
│   ├── insert_features.py
│   ├── read_features.py
│   └── feature_schema.py
├── models/                 # Model training and inference
│   ├── train_model.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── explain.py
│   └── save_model.py
├── pipelines/              # ML pipelines
│   ├── feature_pipeline.py
│   ├── training_pipeline.py
│   ├── prediction_pipeline.py
│   └── backfill_pipeline.py
├── services/               # External service integrations
│   ├── openweather.py
│   ├── hopsworks_service.py
│   └── model_registry.py
├── utils/                  # Shared utilities
│   ├── logger.py
│   ├── helper.py
│   ├── preprocessing.py
│   └── metrics.py
├── notebooks/              # Jupyter notebooks
│   └── eda.ipynb
├── .github/workflows/      # GitHub Actions
│   ├── feature_pipeline.yml
│   ├── training_pipeline.yml
│   └── prediction.yml
├── requirements.txt
├── .env
├── run.py
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file (or export them):

```env
OPENWEATHER_API_KEY=your_key_here
HOPSWORKS_API_KEY=your_key_here
HOPSWORKS_PROJECT=your_project_name
CITY=karachi
MODEL_NAME=aqi_model
```

### 3. Backfill Historical Data

```bash
python run.py backfill
```

### 4. Train Models

```bash
python run.py train
```

### 5. Run Predictions

```bash
python run.py predict
```

### 6. Start the API Server

```bash
python run.py serve
```

### 7. Start the Dashboard

```bash
python run.py dashboard
```

---

## 📡 API Endpoints

| Method | Endpoint     | Description                   |
|--------|-------------|-------------------------------|
| GET    | `/`         | Health check                  |
| GET    | `/current`  | Current AQI reading           |
| GET    | `/predict`  | 3-day AQI forecast            |
| GET    | `/forecast` | Alias for `/predict`          |
| GET    | `/history`  | Historical AQI data           |

---

## 🤖 GitHub Actions

| Workflow             | Schedule  | Description                           |
|---------------------|-----------|---------------------------------------|
| Feature Pipeline    | Hourly    | Fetch latest AQ data → Hopsworks     |
| Training Pipeline   | Daily     | Retrain models → Hopsworks Registry  |
| Prediction Pipeline | Hourly    | Generate forecasts → Save predictions |

---

## 📊 Dashboard Pages

- **Home** — Current AQI gauge, pollutant concentrations, health alerts
- **Current AQI** — Detailed pollutant breakdown
- **Forecast** — 3-day AQI prediction with bar charts
- **Historical Trends** — Interactive AQI trend chart
- **Model Performance** — Latest prediction results
- **Feature Importance** — SHAP explanations

---

## 📄 License

This project was developed as part of the 10Pearls Data Science Internship.
