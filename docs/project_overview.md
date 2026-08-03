# Project Overview

**Pearls AQI Predictor** is an end-to-end Machine Learning project that predicts the **Air Quality Index (AQI) for the next 3 days** using historical and real-time environmental data. The system automatically collects weather and air pollution data from external APIs, transforms the raw data into meaningful features, stores them in a Feature Store, trains forecasting models, and serves predictions through an interactive web dashboard.

The project demonstrates a **production-ready MLOps workflow** by automating data ingestion, feature engineering, model training, evaluation, deployment, and monitoring. It is designed using a **100% serverless architecture**, minimizing infrastructure management while enabling scalable and cost-effective ML pipelines.

The application helps users, researchers, and city authorities monitor upcoming air quality conditions, understand the factors affecting AQI, and receive early warnings for unhealthy pollution levels. The project also includes explainable AI techniques to improve model transparency and support informed decision-making.

---

# Key Features

## 1. Automated Data Collection Pipeline

* Fetch real-time air quality data from AQICN or OpenWeather APIs
* Collect weather variables including temperature, humidity, pressure, rainfall, and wind speed
* Retrieve pollutant concentrations such as PM2.5, PM10, NO₂, SO₂, CO, and O₃
* Validate incoming data before processing
* Handle missing or corrupted API responses automatically
* Store raw API responses for future auditing
* Schedule automatic hourly or daily data collection
* Support multiple cities without code modifications

---

## 2. Feature Engineering Pipeline

* Clean missing and duplicate records
* Generate time-based features

  * Hour
  * Day
  * Week
  * Month
  * Season
  * Weekend indicator
* Generate lag features

  * Previous 1 hour AQI
  * Previous 6 hours AQI
  * Previous 24 hours AQI
* Create rolling statistics

  * Rolling mean
  * Rolling maximum
  * Rolling minimum
  * Rolling standard deviation
* Calculate AQI change rate
* Compute pollutant ratios and interactions
* Normalize or standardize features
* Encode categorical variables
* Detect and remove outliers
* Save processed features into a Feature Store

---

## 3. Feature Store Management

* Store engineered features for reuse
* Maintain consistent features between training and inference
* Version feature datasets
* Retrieve historical features for model training
* Load latest features for real-time prediction
* Prevent feature duplication
* Support feature updates automatically

---

## 4. Historical Data Backfill

* Generate historical training datasets
* Backfill features for previous months or years
* Recreate feature pipelines for any date range
* Produce consistent datasets for retraining
* Support model experimentation on different historical windows
* Enable reproducible experiments

---

## 5. Model Training Pipeline

* Load historical features from Feature Store
* Split datasets into train, validation, and test sets
* Train multiple forecasting models

  * Linear Regression
  * Ridge Regression
  * Random Forest
  * XGBoost
  * LightGBM
  * TensorFlow models
  * PyTorch models
* Perform hyperparameter tuning
* Compare model performance automatically
* Save the best-performing model
* Register trained models in Model Registry

---

## 6. Model Evaluation

* Calculate MAE
* Calculate RMSE
* Calculate R² Score
* Calculate MAPE
* Compare multiple models
* Generate evaluation reports
* Plot predicted vs actual AQI
* Visualize residual errors
* Track model performance over time

---

## 7. Real-Time Prediction Pipeline

* Load latest trained model
* Retrieve newest engineered features
* Predict AQI for the next 3 days
* Support predictions for multiple cities
* Generate confidence scores
* Update predictions automatically
* Cache predictions for faster response

---

## 8. Explainable AI (XAI)

* Explain model decisions using SHAP
* Display feature importance rankings
* Show positive and negative feature contributions
* Explain individual AQI predictions
* Compare feature importance across models
* Increase prediction transparency

---

## 9. Interactive Web Dashboard

* Select city from dropdown
* Display current AQI
* Show 3-day AQI forecast
* Visualize historical AQI trends
* Plot pollutant concentration charts
* Display weather conditions
* Show feature importance graphs
* Present model evaluation metrics
* Refresh predictions automatically
* Responsive design for desktop and mobile

---

## 10. Alerts and Notifications

* Detect hazardous AQI levels
* Generate health warning messages
* Categorize AQI levels

  * Good
  * Moderate
  * Unhealthy for Sensitive Groups
  * Unhealthy
  * Very Unhealthy
  * Hazardous
* Highlight pollution spikes
* Display recommendation messages

---

## 11. CI/CD & Automation

* Schedule hourly feature pipeline execution
* Schedule daily model retraining
* Automatically evaluate newly trained models
* Deploy best-performing models
* Use GitHub Actions or Apache Airflow
* Automate testing before deployment
* Monitor pipeline execution
* Log pipeline failures

---

## 12. Model Monitoring

* Monitor prediction quality
* Detect data drift
* Detect feature drift
* Detect model drift
* Track API failures
* Monitor data freshness
* Log prediction requests
* Track system performance

---

## 13. Logging & Experiment Tracking

* Log every pipeline execution
* Track model versions
* Record training parameters
* Store evaluation metrics
* Maintain experiment history
* Compare previous experiments
* Support reproducible ML workflows

---

## 14. Production Deployment

* Deploy prediction API using Flask or FastAPI
* Serve dashboard with Streamlit
* Host models in Vertex AI or similar platforms
* Expose REST APIs for external applications
* Enable scalable serverless deployment
* Support automatic updates and rollback mechanisms
