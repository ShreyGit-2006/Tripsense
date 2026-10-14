"""
member3/src/best_dates.py
=========================
Best Travel Dates recommender for TripSense.

Data sources (all relative to the project root):
  data/places/tripsense_places_v2.csv      – per-place comfort scores by month
  data/calendars/holiday_crowd.csv         – per-day crowd, holiday, bridge, school data
  data/calendars/calendar_india.csv        – national/festival holidays (lighter file)

Scoring formula (configurable via weights dict)
-----------------------------------------------
Monthly score per place
  comfort_score   : comfort_<month> column (0 or 1, binary comfort flag)
  low_crowd_score : 1 − crowd_proxy_normalized  (lower crowd = better for travel)
  base_visit_score: base_visit_score from holiday_crowd calendar (convenience proxy)
  bridge_bonus    : fraction of days in month that are bridge/long-weekend candidates

  composite = w_comfort * comfort
            + w_crowd   * low_crowd
            + w_base    * base_visit
            + w_bridge  * bridge_bonus

  All per-place scores normalised to [0, 1] across the 12 months so places with
  wildly different absolute numbers compare fairly.

Daily score per place
  daily_comfort   : comfort flag for the month that day falls in
  low_crowd       : 1 − crowd_proxy_normalized for that day
  base_visit      : base_visit_score for that day
  bridge_bonus    : is_bridge_day or is_long_weekend_candidate (0/1)
  is_holiday_bonus: is_holiday flag (holiday = more leisure availability)

  composite_day = w_comfort * comfort + w_crowd * low_crowd
                + w_base * base_visit + w_bridge * bridge_bonus
                + w_holiday * is_holiday_bonus

Outputs
  data/calendars/best_months_results.csv  – one row per (place, month)
  data/calendars/best_dates_results.csv   – one row per (place, date)

The source CSVs are NEVER modified.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path helpers (work from any working directory)
# ---------------------------------------------------------------------------
try:
    _HERE = Path(__file__).resolve()
    _REPO_ROOT = _HERE.parents[2]          # member3/src/best_dates.py → repo root
    PLACES_CSV       = _REPO_ROOT / "data" / "places" / "tripsense_places_v2.csv"
    HOLIDAY_CROWD_CSV = _REPO_ROOT / "data" / "calendars" / "holiday_crowd.csv"
    CALENDAR_CSV     = _REPO_ROOT / "data" / "calendars" / "calendar_india.csv"
    OUT_DIR          = _REPO_ROOT / "data" / "calendars"
except NameError:
    # Colab fallback (flat uploads to /content)
    _REPO_ROOT = Path("/content")
    PLACES_CSV       = Path("/content/tripsense_places_v2.csv")
    HOLIDAY_CROWD_CSV = Path("/content/holiday_crowd.csv")
    CALENDAR_CSV     = Path("/content/calendar_india.csv")
    OUT_DIR          = Path("/content")

BEST_MONTHS_OUT  = OUT_DIR / "best_months_results.csv"
BEST_DATES_OUT   = OUT_DIR / "best_dates_results.csv"

# Month name ↔ number helpers
_MONTH_COLS = [
    "comfort_jan", "comfort_feb", "comfort_mar", "comfort_apr",
    "comfort_may", "comfort_jun", "comfort_jul", "comfort_aug",
    "comfort_sep", "comfort_oct", "comfort_nov", "comfort_dec",
]
_MONTH_NUMS  = list(range(1, 13))
_MONTH_NAMES = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December",
]
_NUM_TO_NAME = dict(zip(_MONTH_NUMS, _MONTH_NAMES))
_NAME_TO_NUM = {v: k for k, v in _NUM_TO_NAME.items()}

# Default scoring weights (must sum to 1.0 for monthly; daily has extra holiday term)
DEFAULT_MONTHLY_WEIGHTS = {
    "comfort":    0.40,
    "crowd":      0.30,
    "base_visit": 0.20,
    "bridge":     0.10,
}
DEFAULT_DAILY_WEIGHTS = {
    "comfort":    0.35,
    "crowd":      0.30,
    "base_visit": 0.20,
    "bridge":     0.10,
    "holiday":    0.05,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_places(path: Optional[Path] = None) -> pd.DataFrame:
    """Load tourist places. Returns empty DataFrame on missing file."""
    p = Path(path) if path else PLACES_CSV
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    # Ensure comfort columns are numeric
    for col in _MONTH_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def load_holiday_crowd(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the holiday_crowd calendar. Returns empty DataFrame on missing file."""
    p = Path(path) if path else HOLIDAY_CROWD_CSV
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"])
    numeric_cols = [
        "crowd_proxy_score", "crowd_proxy_normalized", "base_visit_score",
        "is_holiday", "holiday_penalty", "is_bridge_day", "bridge_penalty",
        "school_vacation_proxy", "school_vacation_penalty", "weekend_penalty",
        "is_long_weekend_candidate",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def load_calendar(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the national holiday calendar. Returns empty DataFrame on missing file."""
    p = Path(path) if path else CALENDAR_CSV
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"])
    return df


# ---------------------------------------------------------------------------
# Monthly aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_crowd_by_month(crowd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by month (1-12) with aggregate crowd metrics.
    Falls back to all-zeros if crowd_df is empty.
    """
    if crowd_df.empty:
        return pd.DataFrame({
            "month": _MONTH_NUMS,
            "mean_crowd_normalized": [0.0] * 12,
            "mean_base_visit":       [1.0] * 12,
            "bridge_day_fraction":   [0.0] * 12,
        })

    crowd_df = crowd_df.copy()
    crowd_df["month"] = crowd_df["date"].dt.month

    agg = crowd_df.groupby("month").agg(
        mean_crowd_normalized=("crowd_proxy_normalized", "mean"),
        mean_base_visit=("base_visit_score", "mean"),
        bridge_day_fraction=("is_bridge_day", "mean"),
    ).reset_index()

    # Ensure all 12 months present (fill missing with neutral values)
    full = pd.DataFrame({"month": _MONTH_NUMS})
    agg = full.merge(agg, on="month", how="left")
    agg["mean_crowd_normalized"] = agg["mean_crowd_normalized"].fillna(0.5)
    agg["mean_base_visit"]       = agg["mean_base_visit"].fillna(0.5)
    agg["bridge_day_fraction"]   = agg["bridge_day_fraction"].fillna(0.0)
    return agg


def _safe_minmax(series: pd.Series) -> pd.Series:
    """Min-max normalize; return zeros if constant."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - mn) / (mx - mn)


# ---------------------------------------------------------------------------
# Monthly scoring
# ---------------------------------------------------------------------------

def score_months(
    places_df: pd.DataFrame,
    crowd_df: pd.DataFrame,
    weights: Optional[dict] = None,
    place_id_filter: Optional[list] = None,
    district_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute a composite travel-suitability score for every (place, month) pair.

    Parameters
    ----------
    places_df        : DataFrame from load_places()
    crowd_df         : DataFrame from load_holiday_crowd()
    weights          : dict with keys comfort/crowd/base_visit/bridge (default DEFAULT_MONTHLY_WEIGHTS)
    place_id_filter  : keep only these place_ids
    district_filter  : keep only this district (case-insensitive substring match)

    Returns
    -------
    DataFrame with columns:
        place_id, name, district, category,
        month, month_name,
        comfort_flag, low_crowd_score, base_visit_score, bridge_fraction,
        composite_score, composite_score_norm,
        rank_within_place,
        reasons
    """
    weights = weights or DEFAULT_MONTHLY_WEIGHTS
    if abs(sum(weights.values()) - 1.0) > 0.01:
        raise ValueError("Monthly weights must sum to 1.0")

    if places_df.empty:
        return pd.DataFrame()

    places = places_df.copy()
    if place_id_filter:
        places = places[places["place_id"].isin(place_id_filter)]
    if district_filter:
        mask = places["district"].str.lower().str.contains(
            district_filter.lower(), na=False
        )
        places = places[mask]

    if places.empty:
        return pd.DataFrame()

    crowd_monthly = _aggregate_crowd_by_month(crowd_df)

    rows = []
    for _, place in places.iterrows():
        for m_num, m_col, m_name in zip(_MONTH_NUMS, _MONTH_COLS, _MONTH_NAMES):
            comfort = float(place.get(m_col, 0) or 0)
            crowd_row = crowd_monthly[crowd_monthly["month"] == m_num].iloc[0]
            low_crowd = 1.0 - float(crowd_row["mean_crowd_normalized"])
            base_visit = float(crowd_row["mean_base_visit"])
            bridge_frac = float(crowd_row["bridge_day_fraction"])

            composite = (
                weights["comfort"]    * comfort
                + weights["crowd"]    * low_crowd
                + weights["base_visit"] * base_visit
                + weights["bridge"]   * bridge_frac
            )

            rows.append({
                "place_id":          place["place_id"],
                "name":              place.get("name", ""),
                "district":          place.get("district", ""),
                "category":          place.get("category", ""),
                "month":             m_num,
                "month_name":        m_name,
                "comfort_flag":      int(comfort),
                "low_crowd_score":   round(low_crowd, 4),
                "base_visit_score":  round(base_visit, 4),
                "bridge_fraction":   round(bridge_frac, 4),
                "composite_score":   round(composite, 4),
            })

    result = pd.DataFrame(rows)

    # Normalize composite_score within each place across 12 months
    result["composite_score_norm"] = (
        result.groupby("place_id")["composite_score"]
        .transform(_safe_minmax)
        .round(4)
    )

    # Rank months within each place (1 = best)
    result["rank_within_place"] = (
        result.groupby("place_id")["composite_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # Human-readable reason
    result["reasons"] = result.apply(_month_reason, axis=1)

    result = result.sort_values(
        ["place_id", "rank_within_place"], ascending=[True, True]
    ).reset_index(drop=True)

    return result


def _month_reason(row: pd.Series) -> str:
    parts = []
    if row["comfort_flag"]:
        parts.append("comfortable weather")
    else:
        parts.append("weather may be harsh")
    if row["low_crowd_score"] >= 0.6:
        parts.append("low crowd expected")
    elif row["low_crowd_score"] <= 0.4:
        parts.append("higher crowd expected")
    if row["bridge_fraction"] > 0:
        parts.append("has bridge/long-weekend days")
    if row["base_visit_score"] >= 0.7:
        parts.append("high leisure availability")
    return "; ".join(parts) if parts else "no strong signal"


# ---------------------------------------------------------------------------
# Daily scoring
# ---------------------------------------------------------------------------

def score_dates(
    places_df: pd.DataFrame,
    crowd_df: pd.DataFrame,
    weights: Optional[dict] = None,
    place_id_filter: Optional[list] = None,
    district_filter: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute a composite travel-suitability score for every (place, date) pair.

    Parameters
    ----------
    places_df       : DataFrame from load_places()
    crowd_df        : DataFrame from load_holiday_crowd()
    weights         : dict with keys comfort/crowd/base_visit/bridge/holiday
    place_id_filter : keep only these place_ids
    district_filter : case-insensitive district substring filter
    date_from       : ISO date string, inclusive lower bound
    date_to         : ISO date string, inclusive upper bound

    Returns
    -------
    DataFrame with columns:
        place_id, name, district, date, weekday, season,
        comfort_flag, crowd_proxy_normalized, low_crowd_score,
        base_visit_score, bridge_bonus, is_holiday, holiday_name,
        composite_score, composite_score_norm, rank_within_place,
        reasons
    """
    weights = weights or DEFAULT_DAILY_WEIGHTS
    if abs(sum(weights.values()) - 1.0) > 0.01:
        raise ValueError("Daily weights must sum to 1.0")

    if places_df.empty or crowd_df.empty:
        return pd.DataFrame()

    places = places_df.copy()
    if place_id_filter:
        places = places[places["place_id"].isin(place_id_filter)]
    if district_filter:
        mask = places["district"].str.lower().str.contains(
            district_filter.lower(), na=False
        )
        places = places[mask]

    if places.empty:
        return pd.DataFrame()

    cal = crowd_df.copy()
    if date_from:
        cal = cal[cal["date"] >= pd.Timestamp(date_from)]
    if date_to:
        cal = cal[cal["date"] <= pd.Timestamp(date_to)]
    if cal.empty:
        return pd.DataFrame()

    cal["month"] = cal["date"].dt.month
    # Pre-build comfort lookup: month → {place_id: comfort_flag}
    comfort_by_month: dict[int, dict] = {}
    for m_num, m_col in zip(_MONTH_NUMS, _MONTH_COLS):
        if m_col in places.columns:
            comfort_by_month[m_num] = dict(
                zip(places["place_id"], places[m_col].fillna(0))
            )
        else:
            comfort_by_month[m_num] = {pid: 0 for pid in places["place_id"]}

    rows = []
    place_records = places.to_dict("records")

    for place in place_records:
        pid = place["place_id"]
        for _, day in cal.iterrows():
            m = int(day["month"])
            comfort = float(comfort_by_month.get(m, {}).get(pid, 0))
            crowd_norm = float(day.get("crowd_proxy_normalized", 0))
            low_crowd  = 1.0 - crowd_norm
            base_visit = float(day.get("base_visit_score", 0))
            bridge_b   = float(
                max(
                    day.get("is_bridge_day", 0),
                    day.get("is_long_weekend_candidate", 0),
                )
            )
            is_hol     = float(day.get("is_holiday", 0))

            composite = (
                weights["comfort"]      * comfort
                + weights["crowd"]      * low_crowd
                + weights["base_visit"] * base_visit
                + weights["bridge"]     * bridge_b
                + weights["holiday"]    * is_hol
            )

            rows.append({
                "place_id":               pid,
                "name":                   place.get("name", ""),
                "district":               place.get("district", ""),
                "date":                   day["date"].date(),
                "weekday":                day.get("weekday", ""),
                "season":                 day.get("season", ""),
                "comfort_flag":           int(comfort),
                "crowd_proxy_normalized": round(crowd_norm, 4),
                "low_crowd_score":        round(low_crowd, 4),
                "base_visit_score":       round(base_visit, 4),
                "bridge_bonus":           int(bridge_b),
                "is_holiday":             int(is_hol),
                "holiday_name":           day.get("holiday_name", ""),
                "composite_score":        round(composite, 4),
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Normalize and rank within each place
    result["composite_score_norm"] = (
        result.groupby("place_id")["composite_score"]
        .transform(_safe_minmax)
        .round(4)
    )
    result["rank_within_place"] = (
        result.groupby("place_id")["composite_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    result["reasons"] = result.apply(_date_reason, axis=1)

    result = result.sort_values(
        ["place_id", "rank_within_place"], ascending=[True, True]
    ).reset_index(drop=True)

    return result


def _date_reason(row: pd.Series) -> str:
    parts = []
    if row["comfort_flag"]:
        parts.append("comfortable weather month")
    if row["is_holiday"]:
        hn = row.get("holiday_name", "")
        parts.append(f"public holiday ({hn})" if hn else "public holiday")
    if row["bridge_bonus"]:
        parts.append("bridge/long-weekend day")
    if row["low_crowd_score"] >= 0.6:
        parts.append("low crowd expected")
    if row["base_visit_score"] >= 0.7:
        parts.append("high leisure availability")
    return "; ".join(parts) if parts else "regular weekday"


# ---------------------------------------------------------------------------
# Public API helpers
# ---------------------------------------------------------------------------

def get_best_months(
    destination: Optional[str] = None,
    district: Optional[str] = None,
    top_n: int = 3,
    weights: Optional[dict] = None,
    places_path: Optional[Path] = None,
    crowd_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Return best months for a destination or district.

    Parameters
    ----------
    destination : place name (partial, case-insensitive) or place_id
    district    : district name (partial, case-insensitive)
    top_n       : number of top months to return per place
    weights     : scoring weights
    places_path : override default CSV path
    crowd_path  : override default CSV path

    Returns
    -------
    DataFrame of top-N months per matching place, sorted by rank.
    """
    places = load_places(places_path)
    crowd  = load_holiday_crowd(crowd_path)

    place_filter = None
    if destination:
        # Match by place_id exact OR name substring
        mask = (
            (places["place_id"].str.upper() == destination.upper())
            | places["name"].str.lower().str.contains(destination.lower(), na=False)
        )
        place_filter = places.loc[mask, "place_id"].tolist()
        if not place_filter:
            return pd.DataFrame(columns=["place_id", "name", "month_name", "rank_within_place"])

    result = score_months(places, crowd, weights, place_filter, district)
    if result.empty:
        return result

    # Keep only top_n months per place
    top = result[result["rank_within_place"] <= top_n].copy()
    return top


def get_best_dates(
    destination: Optional[str] = None,
    district: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    top_n: int = 20,
    weights: Optional[dict] = None,
    places_path: Optional[Path] = None,
    crowd_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Return best specific travel dates for a destination within an optional date range.

    Parameters
    ----------
    destination : place name (partial) or place_id
    district    : district substring filter
    date_from   : ISO date, inclusive
    date_to     : ISO date, inclusive
    top_n       : top N dates per place to return
    weights     : scoring weights
    places_path : override CSV path
    crowd_path  : override CSV path

    Returns
    -------
    DataFrame of top-N dates per matching place.
    """
    places = load_places(places_path)
    crowd  = load_holiday_crowd(crowd_path)

    place_filter = None
    if destination:
        mask = (
            (places["place_id"].str.upper() == destination.upper())
            | places["name"].str.lower().str.contains(destination.lower(), na=False)
        )
        place_filter = places.loc[mask, "place_id"].tolist()
        if not place_filter:
            return pd.DataFrame()

    result = score_dates(
        places, crowd, weights, place_filter, district, date_from, date_to
    )
    if result.empty:
        return result

    top = result[result["rank_within_place"] <= top_n].copy()
    return top


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def generate_outputs(
    places_path: Optional[Path] = None,
    crowd_path: Optional[Path] = None,
    months_out: Optional[Path] = None,
    dates_out: Optional[Path] = None,
    weights_monthly: Optional[dict] = None,
    weights_daily: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score all places and write best_months_results.csv and best_dates_results.csv.
    Never overwrites the source CSVs.

    Returns
    -------
    (months_df, dates_df) as DataFrames
    """
    places = load_places(places_path)
    crowd  = load_holiday_crowd(crowd_path)

    months_df = score_months(places, crowd, weights_monthly)
    dates_df  = score_dates(places, crowd, weights_daily)

    m_out = Path(months_out) if months_out else BEST_MONTHS_OUT
    d_out = Path(dates_out)  if dates_out  else BEST_DATES_OUT

    m_out.parent.mkdir(parents=True, exist_ok=True)
    d_out.parent.mkdir(parents=True, exist_ok=True)

    months_df.to_csv(m_out, index=False)
    dates_df.to_csv(d_out, index=False)

    return months_df, dates_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate best travel dates and months for TripSense places."
    )
    parser.add_argument("--destination", help="Place name or place_id to filter")
    parser.add_argument("--district", help="District to filter")
    parser.add_argument("--date-from", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--date-to",   help="End date (YYYY-MM-DD)")
    parser.add_argument("--top-n-months", type=int, default=3)
    parser.add_argument("--top-n-dates",  type=int, default=20)
    parser.add_argument("--generate-all", action="store_true",
                        help="Score all places and write output CSVs")
    args, _ = parser.parse_known_args()

    if args.generate_all:
        print("Scoring all places …")
        m_df, d_df = generate_outputs()
        print(f"Best months -> {BEST_MONTHS_OUT}  ({len(m_df)} rows)")
        print(f"Best dates  -> {BEST_DATES_OUT}  ({len(d_df)} rows)")
    else:
        dest = args.destination
        print(f"\n=== Best months for: {dest or 'all'} ===")
        bm = get_best_months(dest, args.district, args.top_n_months)
        if bm.empty:
            print("No results found.")
        else:
            print(bm[["place_id","name","month_name","rank_within_place","reasons"]].to_string(index=False))

        print(f"\n=== Best dates for: {dest or 'all'} ===")
        bd = get_best_dates(
            dest, args.district, args.date_from, args.date_to, args.top_n_dates
        )
        if bd.empty:
            print("No results found.")
        else:
            print(bd[["place_id","name","date","weekday","rank_within_place","reasons"]].to_string(index=False))
