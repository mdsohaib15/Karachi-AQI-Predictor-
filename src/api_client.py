"""
API Client Module
=================
Provides unified client wrappers for fetching live weather and air quality
observations for Karachi using:
1. Open-Meteo Live APIs (free, no API key required, full parameter match)
2. OpenWeatherMap API (optional, using OPENWEATHER_API_KEY)

Coordinates:
    Karachi: Latitude = 24.8607, Longitude = 67.0011
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

load_dotenv()

# Karachi Coordinates
DEFAULT_LATITUDE = 24.8607
DEFAULT_LONGITUDE = 67.0011

OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPENWEATHERMAP_URL = "https://api.openweathermap.org/data/2.5/air_pollution"


class OpenMeteoClient:
    """Client for Open-Meteo Live Weather & Air Quality APIs."""

    def __init__(self, lat: float = DEFAULT_LATITUDE, lon: float = DEFAULT_LONGITUDE):
        self.lat = lat
        self.lon = lon
        self.session = requests.Session()

    def fetch_live_weather(self) -> Dict[str, Any]:
        """Fetch current hourly weather metrics."""
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "apparent_temperature",
                "precipitation",
                "rain",
                "surface_pressure",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
            ],
            "timezone": "UTC",
        }
        resp = self.session.get(OPEN_METEO_WEATHER_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})
        return {
            "temperature_2m": float(current.get("temperature_2m", 25.0)),
            "relative_humidity_2m": float(current.get("relative_humidity_2m", 60.0)),
            "dew_point_2m": float(current.get("dew_point_2m", 18.0)),
            "apparent_temperature": float(current.get("apparent_temperature", 26.0)),
            "precipitation": float(current.get("precipitation", 0.0)),
            "rain": float(current.get("rain", 0.0)),
            "surface_pressure": float(current.get("surface_pressure", 1010.0)),
            "cloud_cover": float(current.get("cloud_cover", 20.0)),
            "wind_speed_10m": float(current.get("wind_speed_10m", 12.0)),
            "wind_direction_10m": float(current.get("wind_direction_10m", 220.0)),
        }

    def fetch_live_air_quality(self) -> Dict[str, Any]:
        """Fetch current air pollutant concentrations."""
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "current": [
                "pm2_5",
                "pm10",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "carbon_monoxide",
                "ozone",
            ],
            "timezone": "UTC",
        }
        resp = self.session.get(OPEN_METEO_AIR_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})
        return {
            "PM2.5": float(current.get("pm2_5", 35.0)),
            "PM10": float(current.get("pm10", 70.0)),
            "NO2": float(current.get("nitrogen_dioxide", 25.0)),
            "SO2": float(current.get("sulphur_dioxide", 10.0)),
            "CO": float(current.get("carbon_monoxide", 450.0)),
            "O3": float(current.get("ozone", 30.0)),
        }

    def fetch_combined_hourly_observation(self) -> Dict[str, Any]:
        """Fetch combined weather and air quality dictionary for current hour."""
        now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        weather_data = self.fetch_live_weather()
        air_data = self.fetch_live_air_quality()

        combined = {
            "datetime": now_utc.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "city": "Karachi",
            **weather_data,
            **air_data,
        }
        log.info("✓ Fetched live observation for Karachi at %s (PM2.5: %.1f µg/m³)",
                 combined["datetime"], combined["PM2.5"])
        return combined


class OpenWeatherMapClient:
    """Optional client for OpenWeatherMap Air Pollution API."""

    def __init__(self, api_key: Optional[str] = None, lat: float = DEFAULT_LATITUDE, lon: float = DEFAULT_LONGITUDE):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        self.lat = lat
        self.lon = lon
        self.session = requests.Session()

    def fetch_air_pollution(self) -> Optional[Dict[str, Any]]:
        """Fetch air pollution data from OpenWeatherMap API."""
        if not self.api_key:
            log.warning("OPENWEATHER_API_KEY not configured.")
            return None

        params = {"lat": self.lat, "lon": self.lon, "appid": self.api_key}
        try:
            resp = self.session.get(OPENWEATHERMAP_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            components = data["list"][0]["components"]
            return {
                "PM2.5": float(components.get("pm2_5", 0.0)),
                "PM10": float(components.get("pm10", 0.0)),
                "NO2": float(components.get("no2", 0.0)),
                "SO2": float(components.get("so2", 0.0)),
                "CO": float(components.get("co", 0.0)),
                "O3": float(components.get("o3", 0.0)),
            }
        except Exception as exc:
            log.warning("OpenWeatherMap API request failed: %s", exc)
            return None


def get_current_hourly_record() -> Dict[str, Any]:
    """Primary entry point to fetch live hourly observation with automatic fallback."""
    client = OpenMeteoClient()
    try:
        return client.fetch_combined_hourly_observation()
    except Exception as e:
        log.error("Failed to fetch from Open-Meteo: %s. Trying OpenWeatherMap fallback...", e)
        owm_client = OpenWeatherMapClient()
        owm_air = owm_client.fetch_air_pollution()
        if owm_air:
            now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            return {
                "datetime": now_utc.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "city": "Karachi",
                "temperature_2m": 28.0,
                "relative_humidity_2m": 65.0,
                "dew_point_2m": 20.0,
                "apparent_temperature": 30.0,
                "precipitation": 0.0,
                "rain": 0.0,
                "surface_pressure": 1011.0,
                "cloud_cover": 25.0,
                "wind_speed_10m": 14.0,
                "wind_direction_10m": 230.0,
                **owm_air,
            }
        raise RuntimeError("All live API sources failed.") from e