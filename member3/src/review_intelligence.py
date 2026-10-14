"""
member3/src/review_intelligence.py
===================================
Hotel Review Intelligence for TripSense.

Data sources:
  data/hotels/tripsense_hotel_reviews_dataset.csv   – hotel reviews
  data/hotels/tripsense_hotels.csv                  – hotel metadata

Pipeline
--------
1. Load & validate reviews (dedup, type-check, drop malformed rows)
2. Sentiment analysis with VADER (local, no API key required)
   - VADER compound score mapped to: positive / neutral / negative
   - Kept separate from the user-provided star rating
3. Topic detection via keyword matching over 8 predefined categories:
   cleanliness, staff, location, food, service, amenities, value, ambiance
4. Aggregation per hotel:
   - Mean VADER sentiment score, mean star rating, review count
   - Topic mention frequencies and dominant topic
5. Write two output CSVs (never overwrites source inputs)

VADER compound score interpretation (standard thresholds):
  >= +0.05  → positive
  <= -0.05  → negative
  otherwise → neutral

Limitations
-----------
- VADER is a lexicon-based rule model; it may misclassify sarcasm or domain-specific language.
- Sentiment predictions are NOT ground truth; they are approximate NLP estimates.
- No transformer models are used to keep the system offline and dependency-light.
- Topic detection is keyword-based; it may miss paraphrased mentions.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
try:
    _HERE      = Path(__file__).resolve()
    _REPO_ROOT = _HERE.parents[2]
    REVIEWS_CSV = _REPO_ROOT / "data" / "hotels" / "tripsense_hotel_reviews_dataset.csv"
    HOTELS_CSV  = _REPO_ROOT / "data" / "hotels" / "tripsense_hotels.csv"
    OUT_DIR     = _REPO_ROOT / "data" / "hotels"
except NameError:
    # Colab fallback
    _REPO_ROOT = Path("/content")
    REVIEWS_CSV = Path("/content/tripsense_hotel_reviews_dataset.csv")
    HOTELS_CSV  = Path("/content/tripsense_hotels.csv")
    OUT_DIR     = Path("/content")

REVIEW_INSIGHTS_OUT = OUT_DIR / "review_insights.csv"
TOPIC_SUMMARY_OUT   = OUT_DIR / "review_topic_summary.csv"

# ---------------------------------------------------------------------------
# Topic keyword taxonomy
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "cleanliness": [
        "clean", "dirty", "hygiene", "hygienic", "spotless", "filthy",
        "sanitation", "tidy", "mess", "odor", "smell", "stain",
    ],
    "staff": [
        "staff", "reception", "receptionist", "helpful", "rude", "friendly",
        "courteous", "service team", "front desk", "check-in", "check-out",
        "manager", "hospitality",
    ],
    "location": [
        "location", "nearby", "central", "accessible", "distance", "view",
        "scenic", "situated", "neighbourhood", "neighborhood", "transport",
        "proximity",
    ],
    "food": [
        "food", "breakfast", "dinner", "lunch", "buffet", "cuisine", "meal",
        "restaurant", "menu", "taste", "delicious", "bland", "vegetarian",
        "snack", "chef",
    ],
    "service": [
        "service", "prompt", "slow", "attentive", "room service", "responsive",
        "quick", "efficient", "poor service", "great service", "timely",
    ],
    "amenities": [
        "pool", "gym", "spa", "wifi", "wi-fi", "parking", "amenities",
        "facilities", "ac", "air conditioning", "television", "tv", "elevator",
        "lift", "bathroom", "bath", "shower",
    ],
    "value": [
        "value", "price", "expensive", "cheap", "affordable", "overpriced",
        "worth", "budget", "cost", "money", "deal", "rate",
    ],
    "ambiance": [
        "ambiance", "ambience", "atmosphere", "decor", "decoration", "cozy",
        "noisy", "quiet", "peaceful", "serene", "comfortable", "charming",
    ],
}

_VADER = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Loading & validation
# ---------------------------------------------------------------------------

def load_reviews(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load and validate hotel reviews.
    Returns an empty DataFrame if the file is missing.
    """
    p = Path(path) if path else REVIEWS_CSV
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p, dtype=str)

    # Standardise numeric columns
    for col in ["rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing critical identifiers
    required = ["review_id", "hotel_id", "review_text"]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA
    df = df.dropna(subset=["hotel_id"])
    df["review_text"] = df["review_text"].fillna("").astype(str)

    # Deduplicate on review_id (keep first)
    if "review_id" in df.columns:
        df = df.drop_duplicates(subset=["review_id"])

    # Drop rows where review_text is blank after stripping
    df = df[df["review_text"].str.strip().str.len() > 0]
    df = df.reset_index(drop=True)
    return df


def load_hotels(path: Optional[Path] = None) -> pd.DataFrame:
    """Load hotel metadata. Returns empty DataFrame on missing file."""
    p = Path(path) if path else HOTELS_CSV
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------

def analyse_sentiment(text: str) -> dict:
    """
    Run VADER on a single text string.

    Returns
    -------
    dict with keys: vader_compound, vader_pos, vader_neu, vader_neg,
                    vader_label (positive/neutral/negative)

    Note: VADER compound is in [-1, 1]; labels use standard ±0.05 thresholds.
    This is an APPROXIMATE NLP estimate, not ground truth.
    """
    if not text or not text.strip():
        return {
            "vader_compound": 0.0,
            "vader_pos": 0.0,
            "vader_neu": 1.0,
            "vader_neg": 0.0,
            "vader_label": "neutral",
        }
    scores = _VADER.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {
        "vader_compound": round(compound, 4),
        "vader_pos": round(scores["pos"], 4),
        "vader_neu": round(scores["neu"], 4),
        "vader_neg": round(scores["neg"], 4),
        "vader_label": label,
    }


def analyse_reviews_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add VADER sentiment columns to each review row.
    Input df must have a 'review_text' column.
    """
    if df.empty:
        return df
    sentiment_rows = df["review_text"].apply(analyse_sentiment)
    sentiment_df = pd.DataFrame(sentiment_rows.tolist(), index=df.index)
    return pd.concat([df, sentiment_df], axis=1)


# ---------------------------------------------------------------------------
# Topic detection
# ---------------------------------------------------------------------------

def detect_topics(text: str, taxonomy: dict[str, list[str]] = TOPIC_KEYWORDS) -> dict:
    """
    Keyword-based topic presence detection.

    Returns
    -------
    dict mapping each topic name → 1 (mentioned) or 0 (not mentioned)
    """
    text_lower = text.lower()
    result = {}
    for topic, keywords in taxonomy.items():
        result[f"topic_{topic}"] = int(
            any(kw in text_lower for kw in keywords)
        )
    return result


def analyse_reviews_topics(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary topic columns to each review row."""
    if df.empty:
        return df
    topic_rows = df["review_text"].apply(detect_topics)
    topic_df = pd.DataFrame(topic_rows.tolist(), index=df.index)
    return pd.concat([df, topic_df], axis=1)


# ---------------------------------------------------------------------------
# Per-review enrichment pipeline
# ---------------------------------------------------------------------------

def enrich_reviews(
    reviews_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full per-review enrichment: sentiment + topic detection.

    Parameters
    ----------
    reviews_df : DataFrame from load_reviews()

    Returns
    -------
    Enriched DataFrame with all original columns plus:
      vader_compound, vader_pos, vader_neu, vader_neg, vader_label,
      topic_<name> for each topic in TOPIC_KEYWORDS
    """
    if reviews_df.empty:
        return reviews_df

    df = reviews_df.copy()
    df = analyse_reviews_sentiment(df)
    df = analyse_reviews_topics(df)
    return df


# ---------------------------------------------------------------------------
# Aggregation by hotel
# ---------------------------------------------------------------------------

TOPIC_COLS = [f"topic_{t}" for t in TOPIC_KEYWORDS]


def aggregate_by_hotel(
    enriched_df: pd.DataFrame,
    hotels_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Aggregate enriched review rows to one row per hotel.

    Metrics computed:
      - review_count
      - mean_star_rating   (from user-provided 'rating' column, if present)
      - mean_vader_compound
      - pct_positive / pct_negative / pct_neutral  (VADER labels)
      - topic mention rates (fraction of reviews mentioning each topic)
      - dominant_topic (most frequently mentioned topic)

    Parameters
    ----------
    enriched_df : output of enrich_reviews()
    hotels_df   : optional hotel metadata (joined on hotel_id for name/district)

    Returns
    -------
    One-row-per-hotel DataFrame
    """
    if enriched_df.empty:
        return pd.DataFrame()

    df = enriched_df.copy()

    agg: dict = {
        "review_count": ("review_id", "count") if "review_id" in df.columns
                        else ("hotel_id", "count"),
    }

    if "rating" in df.columns:
        agg["mean_star_rating"] = ("rating", "mean")

    if "vader_compound" in df.columns:
        agg["mean_vader_compound"] = ("vader_compound", "mean")

    grouped = df.groupby("hotel_id").agg(**agg).reset_index()
    grouped = grouped.round(4)

    # Sentiment label percentages
    if "vader_label" in df.columns:
        label_pcts = (
            df.groupby("hotel_id")["vader_label"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
            .rename(columns=lambda c: f"pct_{c}")
            .reset_index()
        )
        for col in ["pct_positive", "pct_neutral", "pct_negative"]:
            if col not in label_pcts.columns:
                label_pcts[col] = 0.0
        grouped = grouped.merge(label_pcts, on="hotel_id", how="left")

    # Topic rates
    present_topic_cols = [c for c in TOPIC_COLS if c in df.columns]
    if present_topic_cols:
        topic_rates = (
            df.groupby("hotel_id")[present_topic_cols]
            .mean()
            .round(4)
            .reset_index()
        )
        # rename to rate_<topic>
        topic_rates = topic_rates.rename(
            columns={c: c.replace("topic_", "rate_") for c in present_topic_cols}
        )
        grouped = grouped.merge(topic_rates, on="hotel_id", how="left")

        rate_cols = [c.replace("topic_", "rate_") for c in present_topic_cols]
        available_rate_cols = [c for c in rate_cols if c in grouped.columns]
        if available_rate_cols:
            grouped["dominant_topic"] = (
                grouped[available_rate_cols]
                .idxmax(axis=1)
                .str.replace("rate_", "", regex=False)
            )

    # Optional: merge hotel metadata
    if hotels_df is not None and not hotels_df.empty and "hotel_id" in hotels_df.columns:
        meta_cols = ["hotel_id"]
        for c in ["hotel_name", "district", "tier", "star_rating"]:
            if c in hotels_df.columns:
                meta_cols.append(c)
        grouped = grouped.merge(hotels_df[meta_cols], on="hotel_id", how="left")

    return grouped


# ---------------------------------------------------------------------------
# Topic summary
# ---------------------------------------------------------------------------

def build_topic_summary(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute topic-level summary across all hotels:
      - total reviews mentioning topic
      - fraction of all reviews mentioning topic
      - mean VADER score for reviews mentioning topic

    Returns
    -------
    DataFrame: one row per topic
    """
    if enriched_df.empty:
        return pd.DataFrame()

    present_topic_cols = [c for c in TOPIC_COLS if c in enriched_df.columns]
    if not present_topic_cols:
        return pd.DataFrame()

    rows = []
    total = len(enriched_df)
    for tc in present_topic_cols:
        topic_name = tc.replace("topic_", "")
        mentions = enriched_df[enriched_df[tc] == 1]
        mention_count = len(mentions)
        mention_rate  = round(mention_count / total, 4) if total > 0 else 0.0
        mean_vader = (
            round(mentions["vader_compound"].mean(), 4)
            if "vader_compound" in mentions.columns and mention_count > 0
            else None
        )
        rows.append({
            "topic":         topic_name,
            "mention_count": mention_count,
            "mention_rate":  mention_rate,
            "mean_vader_compound_when_mentioned": mean_vader,
        })

    return pd.DataFrame(rows).sort_values("mention_count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    reviews_path: Optional[Path] = None,
    hotels_path: Optional[Path] = None,
    insights_out: Optional[Path] = None,
    topic_out: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full hotel review intelligence pipeline.

    1. Load reviews + hotels
    2. Enrich (sentiment + topics)
    3. Aggregate by hotel → review_insights.csv
    4. Build topic summary → review_topic_summary.csv

    Returns
    -------
    (insights_df, topic_summary_df)
    """
    reviews = load_reviews(reviews_path)
    hotels  = load_hotels(hotels_path)

    if reviews.empty:
        empty_i = pd.DataFrame()
        empty_t = pd.DataFrame()
        return empty_i, empty_t

    enriched = enrich_reviews(reviews)

    insights      = aggregate_by_hotel(enriched, hotels)
    topic_summary = build_topic_summary(enriched)

    i_out = Path(insights_out) if insights_out else REVIEW_INSIGHTS_OUT
    t_out = Path(topic_out)    if topic_out    else TOPIC_SUMMARY_OUT

    i_out.parent.mkdir(parents=True, exist_ok=True)
    t_out.parent.mkdir(parents=True, exist_ok=True)

    insights.to_csv(i_out, index=False)
    topic_summary.to_csv(t_out, index=False)

    return insights, topic_summary


# ---------------------------------------------------------------------------
# Public convenience functions
# ---------------------------------------------------------------------------

def get_hotel_insights(
    hotel_id: Optional[str] = None,
    district: Optional[str] = None,
    reviews_path: Optional[Path] = None,
    hotels_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Return per-review enriched records for a specific hotel or district.
    Useful for exploratory queries.
    """
    reviews = load_reviews(reviews_path)
    if reviews.empty:
        return pd.DataFrame()

    if hotel_id:
        reviews = reviews[reviews["hotel_id"] == hotel_id]
    if district and "district" in reviews.columns:
        reviews = reviews[
            reviews["district"].str.lower().str.contains(district.lower(), na=False)
        ]

    if reviews.empty:
        return pd.DataFrame()

    return enrich_reviews(reviews)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run TripSense hotel review intelligence pipeline."
    )
    parser.add_argument("--hotel-id", help="Filter by hotel_id")
    parser.add_argument("--district", help="Filter by district")
    parser.add_argument("--run-full", action="store_true",
                        help="Run full pipeline and write output CSVs")
    args, _ = parser.parse_known_args()

    if args.run_full:
        print("Running full pipeline …")
        insights, topic_df = run_pipeline()
        print(f"Hotel insights -> {REVIEW_INSIGHTS_OUT}  ({len(insights)} rows)")
        print(f"Topic summary  -> {TOPIC_SUMMARY_OUT}  ({len(topic_df)} rows)")
        if not insights.empty:
            print("\nSample insights (top 5 by review_count):")
            cols = [c for c in ["hotel_id","review_count","mean_star_rating",
                                 "mean_vader_compound","dominant_topic","district"]
                    if c in insights.columns]
            print(insights.nlargest(5, "review_count")[cols].to_string(index=False))
        if not topic_df.empty:
            print("\nTopic summary:")
            print(topic_df.to_string(index=False))
    else:
        df = get_hotel_insights(args.hotel_id, args.district)
        if df.empty:
            print("No results found.")
        else:
            show_cols = ["review_id","hotel_id","rating","vader_label",
                         "vader_compound","review_text"]
            show_cols = [c for c in show_cols if c in df.columns]
            print(df[show_cols].head(10).to_string(index=False))
