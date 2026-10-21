"""
itinerary/services/date_service.py
=====================================
Best-date / best-month service for the TripSense itinerary planner.

Bridges to Member 3's best_dates.py for live scoring.
Also uses the pre-computed best_months_results.csv and best_dates_results.csv
as fast-path lookups when available.

Public functions
----------------
recommend_months(place_ids, month, best_months_df) -> dict[str, dict]
    For each place, return the best months and whether the given month is good.

score_trip_dates(start_date, n_days, place_ids, best_dates_df) -> dict
    Given fixed user dates, return date-quality scores for the trip window.

suggest_flexible_dates(month, best_months_df, place_ids, top_n) -> list[dict]
    When dates are flexible, suggest the best months for the set of places.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Member 3 bridge
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
_M3_SRC = _REPO_ROOT / "member3" / "src"
if str(_M3_SRC) not in sys.path:
    sys.path.insert(0, str(_M3_SRC))

try:
    from best_dates import get_best_months as _bd_get_best_months  # type: ignore
    _BD_AVAILABLE = True
except ImportError as _e:
    log.warning("best_dates module unavailable: %s", _e)
    _BD_AVAILABLE = False


_MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_month_suitability(
    place_ids: list[str],
    target_month: int,
    best_months_df: pd.DataFrame,
) -> dict[str, dict]:
    """
    For each place_id, return whether target_month is a good travel month
    and what the top months are.

    Returns
    -------
    dict: place_id → {
        rank: int (1=best, 12=worst),
        is_top3: bool,
        composite_score: float,
        reasons: str,
        top_months: list[str],  (month names)
    }
    """
    result: dict[str, dict] = {}
    if best_months_df.empty or not place_ids:
        return result

    for pid in place_ids:
        place_rows = best_months_df[best_months_df["place_id"].astype(str) == str(pid)]
        if place_rows.empty:
            continue

        # Find the row for target_month
        month_row = place_rows[place_rows["month"] == target_month]
        if month_row.empty:
            # Try month_name match
            tgt_name = _MONTH_NAMES.get(target_month, "")
            month_row = place_rows[place_rows["month_name"] == tgt_name]

        rank = None
        score = None
        reasons = ""
        if not month_row.empty:
            rank = int(month_row.iloc[0].get("rank_within_place", 12))
            score = float(month_row.iloc[0].get("composite_score", 0.0))
            reasons = str(month_row.iloc[0].get("reasons", ""))

        # Top months (rank 1–3)
        top_rows = place_rows[place_rows["rank_within_place"] <= 3].sort_values("rank_within_place")
        top_months = top_rows["month_name"].tolist() if "month_name" in top_rows.columns else []

        result[str(pid)] = {
            "rank":            rank,
            "is_top3":         (rank is not None and rank <= 3),
            "composite_score": round(score, 4) if score is not None else None,
            "reasons":         reasons,
            "top_months":      top_months,
        }
    return result


def score_trip_dates(
    start_date: date,
    n_days: int,
    place_ids: list[str],
    best_dates_df: pd.DataFrame,
) -> dict:
    """
    For a fixed trip window, return date-quality information.

    Uses the pre-computed best_dates_results.csv (2026 dates).
    If the trip dates fall outside 2026, returns a note that scoring
    is unavailable for those dates.

    Returns
    -------
    dict: {
        dates_in_dataset: bool,
        daily_scores: list[dict],    per-day average across places
        trip_avg_score: float,
        crowd_note: str,
    }
    """
    if best_dates_df.empty or not place_ids or n_days < 1:
        return {
            "dates_in_dataset": False,
            "daily_scores": [],
            "trip_avg_score": None,
            "crowd_note": "Date scoring data unavailable.",
        }

    trip_dates = [start_date + timedelta(days=i) for i in range(n_days)]
    date_strs = [str(d) for d in trip_dates]

    if "date" not in best_dates_df.columns:
        return {
            "dates_in_dataset": False,
            "daily_scores": [],
            "trip_avg_score": None,
            "crowd_note": "Date column missing from best_dates dataset.",
        }

    # Filter to our date window and place_ids
    bdf = best_dates_df[
        best_dates_df["place_id"].isin(place_ids)
        & best_dates_df["date"].astype(str).isin(date_strs)
    ].copy()

    if bdf.empty:
        return {
            "dates_in_dataset": False,
            "daily_scores": [],
            "trip_avg_score": None,
            "crowd_note": (
                "Selected trip dates are outside the 2026 date coverage "
                "of the best-dates dataset. Scoring unavailable."
            ),
        }

    # Average across all selected places per day
    score_col = "composite_score"
    crowd_col = "crowd_proxy_normalized"
    daily_agg = (
        bdf.groupby("date").agg(
            avg_score=(score_col, "mean"),
            avg_crowd=(crowd_col, "mean") if crowd_col in bdf.columns else (score_col, "count"),
        )
        .reset_index()
        .sort_values("date")
    )

    daily_scores = []
    for _, row in daily_agg.iterrows():
        daily_scores.append({
            "date":        str(row["date"]),
            "avg_score":   round(float(row["avg_score"]), 4),
            "crowd_level": "lower crowd" if float(row.get("avg_crowd", 0.5)) < 0.3 else
                           "moderate crowd" if float(row.get("avg_crowd", 0.5)) < 0.6 else
                           "higher crowd expected",
        })

    trip_avg = float(daily_agg["avg_score"].mean()) if not daily_agg.empty else None

    crowd_note = "Crowd levels are proxy estimates, not actual visitor counts."

    return {
        "dates_in_dataset": True,
        "daily_scores":     daily_scores,
        "trip_avg_score":   round(trip_avg, 4) if trip_avg is not None else None,
        "crowd_note":       crowd_note,
    }


def suggest_best_months_for_places(
    place_ids: list[str],
    best_months_df: pd.DataFrame,
    top_n: int = 3,
) -> list[str]:
    """
    Return the top-N month names that are consistently good across all given places.
    Aggregates by averaging composite scores across places per month.

    Returns
    -------
    list of month name strings, ordered best-first.
    """
    if best_months_df.empty or not place_ids:
        return []

    subset = best_months_df[best_months_df["place_id"].isin(place_ids)].copy()
    if subset.empty:
        return []

    month_col = "month_name" if "month_name" in subset.columns else None
    score_col = "composite_score" if "composite_score" in subset.columns else None
    if month_col is None or score_col is None:
        return []

    agg = (
        subset.groupby(month_col)[score_col]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    return agg.index.tolist()
