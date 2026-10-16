"""TripSense weather integration using Open-Meteo.

Live forecast:
    from weather_api import get_forecast
    result = get_forecast(18.323442, 82.8805825, "2026-10-20", "2026-10-24")

Historical monthly summaries:
    python member1/src/weather_api.py --build-dataset

The historical dataset is a climate-context aid, not a future forecast.
Open-Meteo API: https://open-meteo.com/
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
PLACES_CSV = ROOT / "data" / "places" / "tripsense_places_v2.csv"
WEATHER_DIR = ROOT / "data" / "weather"
WEATHER_CSV = WEATHER_DIR / "tripsense_weather.csv"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT_SECONDS = 30


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET JSON with a small retry policy for temporary API/network errors."""
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Weather API request failed: {last_error}") from last_error


def get_forecast(
    latitude: float,
    longitude: float,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Get daily forecast for one destination's coordinates.

    Dates must be YYYY-MM-DD. Forecast availability depends on Open-Meteo's
    forecast horizon; this function does not pretend distant dates are forecasts.
    """
    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "auto",
        "daily": (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,precipitation_probability_max"
        ),
    }
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    data = _get_json(FORECAST_URL, params)
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    return {
        "latitude": data.get("latitude", latitude),
        "longitude": data.get("longitude", longitude),
        "timezone": data.get("timezone"),
        "data_type": "forecast",
        "daily": [
            {
                "date": date,
                "temperature_max_c": _at(daily, "temperature_2m_max", i),
                "temperature_min_c": _at(daily, "temperature_2m_min", i),
                "precipitation_mm": _at(daily, "precipitation_sum", i),
                "precipitation_probability_pct": _at(
                    daily, "precipitation_probability_max", i
                ),
                "weather_code": _at(daily, "weather_code", i),
            }
            for i, date in enumerate(dates)
        ],
        "source": "Open-Meteo Forecast API",
    }


def _at(values: dict[str, Any], key: str, index: int) -> Any:
    items = values.get(key, [])
    return items[index] if index < len(items) else None


def build_historical_monthly_dataset(
    places_csv: str | Path = PLACES_CSV,
    output_csv: str | Path = WEATHER_CSV,
    start_year: int = 2019,
    end_year: int = 2024,
    pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """Build 12 historical monthly climate summaries per place.

    This makes one archive request per place, aggregating daily observations
    across the requested years. Run it manually; it may take several minutes.
    """
    places = pd.read_csv(places_csv)
    required = {"place_id", "name", "district", "latitude", "longitude"}
    missing = required - set(places.columns)
    if missing:
        raise ValueError(f"Places CSV is missing required columns: {sorted(missing)}")

    places = places.dropna(subset=["place_id", "latitude", "longitude"]).copy()
    places["latitude"] = pd.to_numeric(places["latitude"], errors="coerce")
    places["longitude"] = pd.to_numeric(places["longitude"], errors="coerce")
    places = places.dropna(subset=["latitude", "longitude"])

    rows: list[dict[str, Any]] = []
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    for index, place in places.reset_index(drop=True).iterrows():
        print(f"[{index + 1}/{len(places)}] {place['name']}")
        try:
            data = _get_json(
                ARCHIVE_URL,
                {
                    "latitude": float(place["latitude"]),
                    "longitude": float(place["longitude"]),
                    "start_date": start_date,
                    "end_date": end_date,
                    "daily": "temperature_2m_mean,precipitation_sum",
                    "timezone": "auto",
                },
            )
            daily = data.get("daily", {})
            frame = pd.DataFrame({
                "date": pd.to_datetime(daily.get("time", [])),
                "temperature_mean_c": daily.get("temperature_2m_mean", []),
                "precipitation_mm": daily.get("precipitation_sum", []),
            })
            if frame.empty:
                raise ValueError("Archive API returned no daily observations")
            frame["month"] = frame["date"].dt.month
            frame["rainy_day"] = frame["precipitation_mm"].fillna(0) >= 1.0

            for month in range(1, 13):
                subset = frame[frame["month"] == month]
                rows.append({
                    "place_id": place["place_id"],
                    "place_name": place["name"],
                    "district": place["district"],
                    "month": month,
                    "avg_temp_c": _mean(subset["temperature_mean_c"]),
                    "avg_rainfall_mm_per_day": _mean(subset["precipitation_mm"]),
                    "avg_rainy_days_per_month": _mean(
                        subset.groupby(subset["date"].dt.year)["rainy_day"].sum()
                    ),
                    "historical_start_date": start_date,
                    "historical_end_date": end_date,
                    "data_type": "historical_climate_summary",
                    "data_source": "Open-Meteo Historical Weather API",
                })
        except Exception as exc:
            print(f"  WARNING: skipped place due to API/data error: {exc}")
            for month in range(1, 13):
                rows.append({
                    "place_id": place["place_id"],
                    "place_name": place["name"],
                    "district": place["district"],
                    "month": month,
                    "avg_temp_c": None,
                    "avg_rainfall_mm_per_day": None,
                    "avg_rainy_days_per_month": None,
                    "historical_start_date": start_date,
                    "historical_end_date": end_date,
                    "data_type": "historical_climate_summary",
                    "data_source": "Open-Meteo Historical Weather API",
                })
        time.sleep(pause_seconds)

    result = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result):,} destination-month rows to {output_path}")
    print(f"Places processed: {len(places):,}; expected rows: {len(places) * 12:,}")
    print("Check rows with missing averages; those locations may need a retry.")
    return result


def _mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(values.mean()), 2) if not values.empty else None


def main() -> None:
    parser = argparse.ArgumentParser(description="TripSense Open-Meteo weather tools")
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="Build monthly historical summaries for every place in the places CSV.",
    )
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2024)
    args = parser.parse_args()

    if args.build_dataset:
        build_historical_monthly_dataset(
            start_year=args.start_year,
            end_year=args.end_year,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
