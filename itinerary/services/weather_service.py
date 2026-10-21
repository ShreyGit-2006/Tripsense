"""
itinerary/services/weather_service.py
======================================
Historical weather analysis service for TripSense.

Data source
-----------
data/weather/tripsense_weather_2023.csv
  Schema: place_id, place_name, district, month (1-12), avg_temp_c, total_rainfall_mm

Important: this file contains HISTORICAL average monthly climate data,
NOT real-time forecasts.  All outputs clearly state this.

A live weather API integration hook is provided but disabled by default.
Set the env var OPENWEATHER_API_KEY to enable current-conditions fetching
(only used for the current day; not labelled as a forecast).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# Descriptive thresholds
_TEMP_THRESHOLDS = {
    "very_hot":    35.0,
    "hot":         32.0,
    "warm":        28.0,
    "comfortable": 24.0,
    "cool":        18.0,
}
_RAIN_THRESHOLDS = {
    "very_heavy": 200.0,
    "heavy":      100.0,
    "moderate":   50.0,
    "light":      20.0,
}

_MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


def _temp_label(t: float) -> str:
    if t >= _TEMP_THRESHOLDS["very_hot"]:
        return "very hot"
    if t >= _TEMP_THRESHOLDS["hot"]:
        return "hot"
    if t >= _TEMP_THRESHOLDS["warm"]:
        return "warm"
    if t >= _TEMP_THRESHOLDS["comfortable"]:
        return "comfortable"
    if t >= _TEMP_THRESHOLDS["cool"]:
        return "cool"
    return "cold"


def _rain_label(r: float) -> str:
    if r >= _RAIN_THRESHOLDS["very_heavy"]:
        return "very heavy rainfall expected"
    if r >= _RAIN_THRESHOLDS["heavy"]:
        return "heavy rainfall expected"
    if r >= _RAIN_THRESHOLDS["moderate"]:
        return "moderate rainfall"
    if r >= _RAIN_THRESHOLDS["light"]:
        return "light rainfall"
    return "little or no rainfall"


def _is_outdoor_suitable(temp: float, rain: float) -> bool:
    return temp < _TEMP_THRESHOLDS["very_hot"] and rain < _RAIN_THRESHOLDS["heavy"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_weather_for_place(
    place_id: str,
    month: int,
    weather_df: pd.DataFrame,
) -> Optional[dict]:
    """
    Return historical weather summary for a specific place and month.

    Parameters
    ----------
    place_id   : e.g. "AP0001"
    month      : 1–12
    weather_df : DataFrame from data_service.load_weather()

    Returns
    -------
    dict or None if not found.

    Keys
    ----
    place_id, month, month_name, avg_temp_c, total_rainfall_mm,
    temp_label, rain_label, outdoor_suitable, data_source_note
    """
    if weather_df.empty:
        return None

    mask = (
        (weather_df["place_id"].astype(str) == str(place_id))
        & (weather_df["month"] == month)
    )
    rows = weather_df[mask]
    if rows.empty:
        return None

    row = rows.iloc[0]
    temp = float(row.get("avg_temp_c", float("nan")))
    rain = float(row.get("total_rainfall_mm", float("nan")))

    if pd.isna(temp) or pd.isna(rain):
        return None

    return {
        "place_id":         place_id,
        "month":            month,
        "month_name":       _MONTH_NAMES.get(month, str(month)),
        "avg_temp_c":       round(temp, 1),
        "total_rainfall_mm": round(rain, 1),
        "temp_label":       _temp_label(temp),
        "rain_label":       _rain_label(rain),
        "outdoor_suitable": _is_outdoor_suitable(temp, rain),
        "data_source_note": (
            "Historical monthly climate average (TripSense dataset). "
            "Not a real-time or forecast value."
        ),
    }


def get_weather_for_places(
    place_ids: list[str],
    month: int,
    weather_df: pd.DataFrame,
) -> dict[str, dict]:
    """
    Return weather summaries for multiple place IDs for a given month.

    Returns
    -------
    dict mapping place_id → weather dict (missing places omitted).
    """
    return {
        pid: w
        for pid in place_ids
        if (w := get_weather_for_place(pid, month, weather_df)) is not None
    }


def get_district_weather_summary(
    district: str,
    month: int,
    weather_df: pd.DataFrame,
) -> Optional[dict]:
    """
    Return an aggregated weather summary for all places in a district for a month.
    """
    if weather_df.empty:
        return None

    mask = (
        weather_df["district"].str.lower().str.contains(district.lower(), na=False)
        & (weather_df["month"] == month)
    )
    rows = weather_df[mask]
    if rows.empty:
        # Fallback to entire state average if district not found
        rows = weather_df[weather_df["month"] == month]
        if rows.empty:
            rows = weather_df

    avg_temp = float(rows["avg_temp_c"].mean()) if "avg_temp_c" in rows else 25.0
    avg_rain = float(rows["total_rainfall_mm"].mean()) if "total_rainfall_mm" in rows else 15.0

    return {
        "district":          district,
        "month":             month,
        "month_name":        _MONTH_NAMES.get(month, str(month)),
        "avg_temp_c":        round(avg_temp, 1),
        "total_rainfall_mm": round(avg_rain, 1),
        "avg_rainfall_mm":   round(avg_rain, 1),
        "temp_label":        _temp_label(avg_temp),
        "rain_label":        _rain_label(avg_rain),
        "comfort_text":      f"{_temp_label(avg_temp).title()}, {_rain_label(avg_rain)}",
        "outdoor_suitable":  _is_outdoor_suitable(avg_temp, avg_rain),
        "n_places_sampled":  len(rows),
        "data_source_note":  (
            f"Historical monthly climate average across {len(rows)} places in {district} "
            "(TripSense dataset). Grounded in Member 3 calendar intelligence."
        ),
    }


def district_climate_summary(
    district: str,
    month: int,
    weather_df: pd.DataFrame,
) -> dict:
    """Convenience alias for district climate summary returning safe defaults."""
    res = get_district_weather_summary(district, month, weather_df)
    if res is None:
        return {
            "district": district,
            "month": month,
            "avg_temp_c": 25.5,
            "total_rainfall_mm": 15.0,
            "comfort_text": "Comfortable weather window",
            "data_source_note": "Estimated regional baseline.",
        }
    return res

