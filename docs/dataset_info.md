Here are all 38 features currently documented in [dataset_info.md](file:///e:/Career/Internships/10%20Pearls%20-%20Data%20Science%20Internship/docs/dataset_info.md):

**☀️ Weather Features (11)**

* `datetime`
* `temperature_2m`
* `relative_humidity_2m`
* `dew_point_2m`
* `apparent_temperature`
* `precipitation`
* `rain`
* `surface_pressure`
* `cloud_cover`
* `wind_speed_10m`
* `wind_direction_10m`

**🌫️ Air Quality Features (6)**

* `PM2.5`
* `PM10`
* `NO₂`
* `SO₂`
* `CO`
* `O₃`

**⏱️ Time Features (6)**

* `year`
* `month`
* `day`
* `hour`
* `weekday`
* `is_weekend`

**⏪ Lag Features (4)**

* `pm25_lag_24`
* `pm25_lag_48`
* `pm25_lag_72`
* `pm25_lag_168`

**📈 Rolling Features (4)**

* `pm25_roll_mean_24`
* `pm25_roll_std_24`
* `pm25_roll_mean_72`
* `pm25_roll_std_72`

**✖️ Weather Interaction Features (2)**

* `temp_humidity`
* `wind_pm25`

**🔄 Cyclic Features (4)**

* `hour_sin`
* `hour_cos`
* `month_sin`
* `month_cos`

**🎯 Target Variable (1)**

* `target_pm25`

# Dataset Features Info

### Weather Features


| Feature              | Definition                                                                | Why                                                                           |
| ---------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| datetime             | Date and time when the observation was recorded.                          | Provides chronological order for time-series analysis.                        |
| temperature_2m       | Air temperature measured at 2 meters above ground (°C).                  | Temperature strongly affects pollutant dispersion and chemical reactions.     |
| relative_humidity_2m | Relative humidity at 2 meters (%).                                        | Humidity influences particle formation and pollutant concentration.           |
| dew_point_2m         | Temperature at which air becomes saturated and condensation begins (°C). | Indicates atmospheric moisture, which affects air quality.                    |
| apparent_temperature | Perceived equivalent temperature (°C).                                   | Affects atmospheric stability and chemical reaction rates.                    |
| precipitation        | Total precipitation (rain, snow, etc.) during the time interval (mm).     | Rain removes pollutants from the atmosphere through wet deposition.           |
| rain                 | Amount of rainfall only (mm).                                             | Rainfall helps reduce airborne particulate matter.                            |
| surface_pressure     | Atmospheric pressure at the ground surface (hPa).                         | Surface pressure affects local weather conditions and pollution accumulation. |
| cloud_cover          | Percentage of the sky covered by clouds (%).                              | Clouds influence solar radiation and atmospheric chemistry.                   |
| wind_speed_10m       | Wind speed measured at 10 meters above ground (m/s or km/h).              | Wind disperses or transports pollutants.                                      |
| wind_direction_10m   | Direction from which the wind is blowing (degrees).                       | Determines the source and movement of pollutants.                             |

---

## Air Quality Features


| Feature | Definition                                                  | Why                                                                  |
| --------- | ------------------------------------------------------------- | ---------------------------------------------------------------------- |
| PM2.5   | Fine particulate matter with diameter ≤ 2.5 μm (µg/m³). | Primary pollutant used to estimate AQI and predict future pollution. |
| PM10    | Particulate matter with diameter ≤ 10 μm (µg/m³).       | Represents larger airborne particles affecting AQI.                  |
| NO₂    | Nitrogen dioxide concentration (µg/m³).                   | Major traffic-related pollutant affecting air quality.               |
| SO₂    | Sulfur dioxide concentration (µg/m³).                     | Industrial pollutant contributing to poor air quality.               |
| CO      | Carbon monoxide concentration (mg/m³ or µg/m³).          | Indicates incomplete combustion and urban pollution.                 |
| O₃     | Ozone concentration (µg/m³).                              | Secondary pollutant formed through photochemical reactions.          |

---

## Time Features


| Feature    | Definition                                   | Why                                                                  |
| ------------ | ---------------------------------------------- | ---------------------------------------------------------------------- |
| year       | Year of the observation.                     | Captures long-term environmental trends.                             |
| month      | Month (1–12).                               | Represents seasonal variation in air quality.                        |
| day        | Day of the month (1–31).                    | Captures daily environmental patterns.                               |
| hour       | Hour of the day (0–23).                     | Air pollution changes throughout the day due to traffic and weather. |
| weekday    | Day of the week (0 = Monday, 6 = Sunday).    | Traffic and industrial activity vary by day of the week.             |
| is_weekend | Binary indicator (1 = weekend, 0 = weekday). | Weekend activity patterns differ from weekdays.                      |

---

## Lag Features


| Feature      | Definition                         | Why                                           |
| -------------- | ------------------------------------ | ----------------------------------------------- |
| pm25_lag_24  | PM2.5 value from 24 hours earlier. | Captures daily recurring pollution patterns.  |
| pm25_lag_48  | PM2.5 value from 48 hours earlier. | Captures 2-day historical pollution patterns. |
| pm25_lag_72  | PM2.5 value from 72 hours earlier. | Captures 3-day historical pollution patterns. |
| pm25_lag_168 | PM2.5 value from 1 week earlier.   | Captures weekly cyclical pollution patterns.  |

---

## Rolling Features


| Feature           | Definition                                              | Why                                                   |
| ------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| pm25_roll_mean_24 | Average PM2.5 over the previous 24 hours.               | Captures daily pollution trend.                       |
| pm25_roll_std_24  | Standard deviation of PM2.5 over the previous 24 hours. | Measures variability and instability in PM2.5 levels. |
| pm25_roll_mean_72 | Average PM2.5 over the previous 72 hours.               | Captures 3-day baseline pollution trend.              |
| pm25_roll_std_72  | Standard deviation of PM2.5 over the previous 72 hours. | Measures variability and instability in PM2.5 levels. |

---

## Weather Interaction Features


| Feature       | Definition                                              | Why                                                   |
| --------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| temp_humidity | Interaction between temperature and relative humidity.  | Combined effects often influence pollutant formation. |
| wind_pm25     | Interaction between wind speed and PM2.5 concentration. | Wind modifies the dispersion of particulate matter.   |

---

## Cyclic Features


| Feature   | Definition                                                      | Why                                                               |
| ----------- | ----------------------------------------------------------------- | ------------------------------------------------------------------- |
| hour_sin  | Sine transformation of hour to preserve daily cyclic pattern.   | Preserves the cyclic nature of hours for machine learning models. |
| hour_cos  | Cosine transformation of hour to preserve daily cyclic pattern. | Complements`hour_sin` to encode daily cycles.                     |
| month_sin | Sine transformation of month to preserve yearly seasonality.    | Represents annual seasonality without discontinuity.              |
| month_cos | Cosine transformation of month to preserve yearly seasonality.  | Complements`month_sin` for yearly cycles.                         |

---

## Target Variable


| Feature         | Definition                                             | Why                                                                                                                                  |
| ----------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **target_pm25** | **PM2.5 predicted 72 hours (3 days) into the future.** | **Provides a practical 3-day forecast for health advisories and environmental planning. Selected as the project's target variable.** |
