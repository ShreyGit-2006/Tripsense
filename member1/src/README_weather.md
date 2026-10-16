# TripSense Weather API

## What this provides
- `get_forecast(latitude, longitude, start_date, end_date)` retrieves a daily forecast from Open-Meteo for dates available in its forecast horizon.
- `build_historical_monthly_dataset()` aggregates historical daily weather into monthly summaries for each place in `data/places/tripsense_places_v2.csv`.
- Historical summaries are seasonal context, **not** forecasts for future dates.

## Install
From the repository root:
```bash
pip install pandas requests
```

## Build monthly historical summaries for all places
From the repository root:
```bash
python member1/src/weather_api.py --build-dataset --start-year 2019 --end-year 2024
```

Output:
`data/weather/tripsense_weather.csv`

The builder creates 12 rows per place (expected 5,328 rows for 444 valid places). It prints progress, retries transient requests, and records blank weather values if a place fails. Inspect missing rows and rerun if needed. Historical API availability can vary by coordinate and date range.

## Get a live forecast in Python
Because this file is loaded by path rather than installed as a package, one simple option is:
```python
import importlib.util
spec = importlib.util.spec_from_file_location(
    "weather_api", "member1/src/weather_api.py"
)
weather_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weather_api)

result = weather_api.get_forecast(
    latitude=18.323442,
    longitude=82.8805825,
    start_date="2026-10-20",
    end_date="2026-10-24",
)
print(result)
```

Replace the example dates with the user's trip dates. The forecast API cannot provide reliable daily forecasts arbitrarily far in advance; use historical summaries for distant travel dates.

## Dataset fields
- `place_id`, `place_name`, `district`, `month`: joined to the existing place record
- `avg_temp_c`: average daily mean temperature across available historical observations in that calendar month
- `avg_rainfall_mm_per_day`: average daily precipitation in that calendar month
- `avg_rainy_days_per_month`: average count of days with at least 1 mm precipitation per year for that month, averaged across years
- `historical_start_date`, `historical_end_date`, `data_type`, `data_source`: provenance and interpretation

## Source
Open-Meteo: https://open-meteo.com/
