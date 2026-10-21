"""
itinerary/services/hotel_service.py
=====================================
Hotel recommendation and review-insight service for the TripSense itinerary
planner.  Delegates sentiment computation to Member 3's review_intelligence
module where possible; uses pre-computed review_insights.csv as a faster path.

Matching strategy
-----------------
1. Primary:  join hotels to places via hotel_reviews.nearby_place_id
2. Fallback: match hotels to places by district name
3. We never invent coordinates, prices, or ratings.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Member 3 review_intelligence bridge
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
_M3_SRC = _REPO_ROOT / "member3" / "src"
if str(_M3_SRC) not in sys.path:
    sys.path.insert(0, str(_M3_SRC))

try:
    from review_intelligence import (  # type: ignore
        load_reviews as _ri_load_reviews,
        load_hotels as _ri_load_hotels,
        aggregate_by_hotel as _ri_aggregate,
        enrich_reviews as _ri_enrich,
    )
    _RI_AVAILABLE = True
except ImportError as _e:
    log.warning("review_intelligence unavailable: %s", _e)
    _RI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def hotels_for_place(
    place_id: str,
    hotels_df: pd.DataFrame,
    hotel_reviews_df: pd.DataFrame,
    review_insights_df: pd.DataFrame,
    max_hotels: int = 3,
    district: str = "",
) -> pd.DataFrame:
    """
    Return hotels linked to a specific place, enriched with review insights.

    Linking strategy
    ----------------
    1. hotel_reviews_df has nearby_place_id → hotel_id mapping.
       We look up which hotel_ids are linked to place_id there.
    2. We then fetch metadata from hotels_df.
    3. If no direct links exist and district is provided, falls back to matching district.
    4. We attach review_insights if available.

    Returns
    -------
    DataFrame with columns: hotel_id, hotel_name, district, tier,
    star_rating, price_range, amenities, [review metrics if available]
    Empty DataFrame when no hotels are found.
    """
    if hotels_df.empty:
        return pd.DataFrame()

    # Step 1: find hotel_ids linked to this place
    linked_hotel_ids: list[str] = []
    if not hotel_reviews_df.empty and "nearby_place_id" in hotel_reviews_df.columns:
        mask = hotel_reviews_df["nearby_place_id"].astype(str) == str(place_id)
        linked = hotel_reviews_df[mask]["hotel_id"].dropna().unique().tolist()
        linked_hotel_ids = [str(h) for h in linked]

    # Step 2: fetch metadata
    if linked_hotel_ids:
        result = hotels_df[hotels_df["hotel_id"].isin(linked_hotel_ids)].copy()
    else:
        result = pd.DataFrame()

    # Step 3: if no direct links, fall back to district match via places data
    if result.empty and district:
        mask = hotels_df["district"].str.lower().str.contains(str(district).lower(), na=False)
        result = hotels_df[mask].copy()

    if result.empty:
        log.debug("No direct or district hotel link for %s", place_id)
        return pd.DataFrame()

    result = result.drop_duplicates("hotel_id").head(max_hotels).reset_index(drop=True)

    # Step 4: attach review insights
    if not review_insights_df.empty and "hotel_id" in review_insights_df.columns:
        insight_cols = [
            c for c in [
                "hotel_id", "review_count", "mean_star_rating",
                "mean_vader_compound", "pct_positive", "pct_negative",
                "dominant_topic", "rate_cleanliness", "rate_staff",
                "rate_food", "rate_location", "rate_value",
            ]
            if c in review_insights_df.columns
        ]
        insights = review_insights_df[insight_cols]
        result = result.merge(insights, on="hotel_id", how="left")

    return result


def hotels_for_district(
    district: str,
    hotels_df: pd.DataFrame,
    review_insights_df: pd.DataFrame,
    tier_filter: Optional[str] = None,
    max_hotels: int = 5,
) -> pd.DataFrame:
    """
    Return hotels in a district, optionally filtered by tier.

    Parameters
    ----------
    district       : district name (case-insensitive substring match)
    tier_filter    : "budget" | "mid_range" | "luxury" | None (= any)
    """
    if hotels_df.empty or not district:
        return pd.DataFrame()

    mask = hotels_df["district"].str.lower().str.contains(district.lower(), na=False)
    result = hotels_df[mask].copy()

    if tier_filter and "tier" in result.columns:
        result = result[result["tier"].str.lower() == tier_filter.lower()]

    result = result.drop_duplicates("hotel_id").head(max_hotels).reset_index(drop=True)

    if not result.empty and not review_insights_df.empty and "hotel_id" in review_insights_df.columns:
        insight_cols = [
            c for c in [
                "hotel_id", "review_count", "mean_star_rating",
                "mean_vader_compound", "pct_positive", "dominant_topic",
                "rate_cleanliness", "rate_staff", "rate_location",
                "rate_food", "rate_service", "rate_value",
            ]
            if c in review_insights_df.columns
        ]
        insights = review_insights_df[insight_cols]
        result = result.merge(insights, on="hotel_id", how="left")

    return result


def format_hotel_card(row: pd.Series) -> dict:
    """
    Convert a hotel row (from hotels_for_place/district) into a display dict.
    Only includes fields that are actually present and non-null.
    """
    card: dict = {}
    for field in ["hotel_id", "hotel_name", "district", "tier", "star_rating",
                  "price_range", "amenities"]:
        if field in row.index and pd.notna(row[field]) and str(row[field]).strip():
            card[field] = str(row[field]).strip()

    # Review metrics & predicted aspect scores
    for field in ["review_count", "mean_star_rating", "mean_vader_compound",
                  "pct_positive", "dominant_topic", "rate_cleanliness",
                  "rate_staff", "rate_location", "rate_food", "rate_service", "rate_value"]:
        if field in row.index and pd.notna(row[field]):
            card[field] = row[field]

    card["data_source_note"] = (
        "Review sentiment estimated by VADER (approximate NLP model). "
        "Star ratings from synthetic dataset — treat as relative indicators only."
    )
    return card
