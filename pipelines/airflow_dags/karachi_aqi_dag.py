"""
Apache Airflow DAG - Karachi AQI MLOps Pipeline
===============================================
Orchestrates hourly feature ingestion, next-hour inference,
and daily automated model retraining.

Schedules:
    - Hourly Feature & Prediction Pipeline: Cron('0 * * * *')
    - Daily Model Retraining Pipeline: Cron('0 2 * * *')
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "data_science_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}

# ------------------------------------------------------------------ #
#  1. Hourly Feature & Prediction DAG
# ------------------------------------------------------------------ #
with DAG(
    dag_id="karachi_aqi_hourly_pipeline",
    default_args=default_args,
    description="Fetches live weather/air quality and computes hourly PM2.5 forecast",
    schedule_interval="0 * * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["aqi", "feature_store", "inference", "karachi"],
) as hourly_dag:

    task_fetch_features = BashOperator(
        task_id="run_hourly_feature_pipeline",
        bash_command="python pipelines/feature_pipeline.py",
    )

    task_generate_prediction = BashOperator(
        task_id="run_hourly_prediction_pipeline",
        bash_command="python pipelines/prediction_pipeline.py",
    )

    task_fetch_features >> task_generate_prediction


# ------------------------------------------------------------------ #
#  2. Daily Model Retraining DAG
# ------------------------------------------------------------------ #
with DAG(
    dag_id="karachi_aqi_daily_training_pipeline",
    default_args=default_args,
    description="Retrains candidate ML models daily and registers champion model",
    schedule_interval="0 2 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["aqi", "training", "model_registry", "karachi"],
) as daily_dag:

    task_retrain_models = BashOperator(
        task_id="run_daily_training_pipeline",
        bash_command="python pipelines/training_pipeline.py",
    )
