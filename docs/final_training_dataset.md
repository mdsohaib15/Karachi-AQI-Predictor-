# Final Training Dataset

The final training dataset (`training_dataset.csv`) contains the cleaned air quality, weather, temporal, and engineered features used to train the forecasting model. Each row represents one hourly observation. The target variable is `target_pm25` (PM2.5 prediction).

### Sample Rows of Data

| datetime | temperature_2m | relative_humidity_2m | dew_point_2m | apparent_temperature | precipitation | rain | surface_pressure | cloud_cover | wind_speed_10m | wind_direction_10m | pm25 | pm10 | no2 | so2 | co | o3 | year | month | day | hour | weekday | is_weekend | hour_sin | hour_cos | month_sin | month_cos | pm25_lag_24 | pm25_lag_48 | pm25_lag_72 | pm25_lag_168 | pm25_roll_mean_24 | pm25_roll_std_24 | pm25_roll_mean_72 | pm25_roll_std_72 | temp_humidity | wind_pm25 | target_pm25 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2023-01-08 00:00:00 | 17.1 | 38 | 2.5 | 13.1 | 0.0 | 0.0 | 1018.9 | 0 | 16.1 | 42 | 44.1 | 64.3 | 12.1 | 21.3 | 548.0 | 99.0 | 2023 | 1 | 8 | 0 | 6 | 1 | 0.000 | 1.000 | 0.500 | 0.866 | 45.4 | 40.4 | 37.5 | 36.6 | 50.05 | 5.48 | 46.79 | 5.38 | 649.8 | 710.01 | 40.9 |
| 2023-01-08 01:00:00 | 16.6 | 39 | 2.6 | 12.5 | 0.0 | 0.0 | 1018.6 | 0 | 16.2 | 37 | 44.8 | 65.2 | 12.4 | 21.1 | 554.0 | 97.0 | 2023 | 1 | 8 | 1 | 6 | 1 | 0.259 | 0.966 | 0.500 | 0.866 | 45.6 | 41.6 | 36.9 | 33.5 | 50.01 | 5.51 | 46.90 | 5.26 | 647.4 | 725.76 | 34.7 |
| 2023-01-08 02:00:00 | 16.3 | 40 | 2.5 | 12.2 | 0.0 | 0.0 | 1018.3 | 0 | 16.1 | 39 | 46.3 | 67.5 | 12.9 | 20.9 | 563.0 | 94.0 | 2023 | 1 | 8 | 2 | 6 | 1 | 0.500 | 0.866 | 0.500 | 0.866 | 46.8 | 43.2 | 37.4 | 38.8 | 49.99 | 5.52 | 47.03 | 5.13 | 652.0 | 745.43 | 32.0 |

### Dataset Description

* **File Name:** `training_dataset.csv`
* **Observation Frequency:** Hourly
* **Target Variable:** `target_pm25`
* **Total Features:** **36 input features** (excluding datetime and target)
* **Feature Categories:**
  * Weather Features
  * Air Quality Features
  * Time Features
  * Lag Features
  * Rolling Statistics
  * Interaction Features
  * Cyclic Features
