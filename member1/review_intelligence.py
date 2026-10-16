"""TripSense Member 1: Andhra Pradesh hotel review intelligence.

Reads:
  data/hotels/tripsense_hotel_reviews_dataset.csv
  data/hotels/tripsense_hotels.csv

Writes:
  data/hotels/review_insights.csv
  data/hotels/review_topic_summary.csv

No hotel booking forecasting or hotel pricing is performed.
Run from the repository root:
  python member1/review_intelligence.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEWS = ROOT / "data/hotels/tripsense_hotel_reviews_dataset.csv"
DEFAULT_HOTELS = ROOT / "data/hotels/tripsense_hotels.csv"
DEFAULT_OUT = ROOT / "data/hotels"

TOPICS = {
    "cleanliness": ["clean", "dirty", "hygiene", "spotless", "filthy", "stain", "smell"],
    "staff": ["staff", "reception", "receptionist", "helpful", "rude", "friendly", "manager"],
    "location": ["location", "nearby", "distance", "view", "accessible", "transport", "beach"],
    "food": ["food", "breakfast", "dinner", "lunch", "restaurant", "meal", "delicious", "taste"],
    "service": ["service", "slow", "attentive", "room service", "responsive", "check-in"],
    "amenities": ["pool", "gym", "spa", "wifi", "wi-fi", "parking", "ac", "bathroom", "shower"],
    "value": ["value", "price", "expensive", "cheap", "affordable", "overpriced", "worth", "budget"],
    "ambiance": ["ambiance", "ambience", "atmosphere", "decor", "cozy", "noisy", "quiet", "peaceful"],
}
ANALYZER = SentimentIntensityAnalyzer()


def load_inputs(reviews_path: Path, hotels_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not reviews_path.exists():
        raise FileNotFoundError(f"Reviews CSV not found: {reviews_path}")
    reviews = pd.read_csv(reviews_path, dtype={"review_id": str, "hotel_id": str})
    hotels = pd.read_csv(hotels_path) if hotels_path.exists() else pd.DataFrame()

    required = {"hotel_id", "review_text"}
    missing = required - set(reviews.columns)
    if missing:
        raise ValueError(f"Reviews dataset is missing required columns: {sorted(missing)}")

    if "review_id" in reviews.columns:
        reviews = reviews.drop_duplicates(subset=["review_id"], keep="first")
    reviews["hotel_id"] = reviews["hotel_id"].astype("string").str.strip()
    reviews["review_text"] = reviews["review_text"].fillna("").astype(str).str.strip()
    reviews = reviews[
        reviews["hotel_id"].notna()
        & reviews["hotel_id"].ne("")
        & reviews["review_text"].ne("")
    ].copy()
    if "rating" in reviews.columns:
        reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    return reviews.reset_index(drop=True), hotels


def sentiment(text: str) -> tuple[float, str]:
    compound = float(ANALYZER.polarity_scores(text)["compound"])
    label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
    return compound, label


def analyse_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    out = reviews.copy()
    scores = out["review_text"].map(sentiment)
    out["sentiment_score"] = scores.map(lambda x: round(x[0], 4))
    out["sentiment_label"] = scores.map(lambda x: x[1])
    lowered = out["review_text"].str.lower()
    for topic, words in TOPICS.items():
        pattern = r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"
        out[f"topic_{topic}"] = lowered.str.contains(pattern, regex=True, na=False)
    topic_cols = [f"topic_{topic}" for topic in TOPICS]
    out["topics_mentioned"] = out[topic_cols].apply(
        lambda row: ", ".join(col.removeprefix("topic_") for col in topic_cols if row[col]), axis=1
    )
    return out


def aggregate_hotels(analysed: pd.DataFrame, hotels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hotel_id, group in analysed.groupby("hotel_id", dropna=False):
        row = {
            "hotel_id": hotel_id,
            "review_count": int(len(group)),
            "mean_sentiment_score": round(float(group["sentiment_score"].mean()), 4),
            "positive_review_pct": round(float((group["sentiment_label"] == "positive").mean() * 100), 2),
            "neutral_review_pct": round(float((group["sentiment_label"] == "neutral").mean() * 100), 2),
            "negative_review_pct": round(float((group["sentiment_label"] == "negative").mean() * 100), 2),
        }
        if "rating" in group.columns:
            row["mean_star_rating"] = round(float(group["rating"].mean()), 2) if group["rating"].notna().any() else None
        for topic in TOPICS:
            row[f"{topic}_mention_pct"] = round(float(group[f"topic_{topic}"].mean() * 100), 2)
        topic_counts = {topic: int(group[f"topic_{topic}"].sum()) for topic in TOPICS}
        row["most_mentioned_topic"] = max(topic_counts, key=topic_counts.get) if max(topic_counts.values(), default=0) else "none"
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not hotels.empty and "hotel_id" in hotels.columns:
        summary = hotels.drop_duplicates("hotel_id").merge(summary, on="hotel_id", how="right")
    return summary.sort_values(["review_count", "mean_sentiment_score"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TripSense Andhra Pradesh hotel reviews.")
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--hotels", type=Path, default=DEFAULT_HOTELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    reviews, hotels = load_inputs(args.reviews, args.hotels)
    analysed = analyse_reviews(reviews)
    summary = aggregate_hotels(analysed, hotels)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_review_path = args.out_dir / "review_insights.csv"
    summary_path = args.out_dir / "review_topic_summary.csv"
    analysed.to_csv(per_review_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Analysed {len(analysed):,} reviews across {analysed['hotel_id'].nunique():,} hotels.")
    print(f"Per-review output: {per_review_path}")
    print(f"Hotel summary: {summary_path}")
    print("Sentiment is an automated estimate; topic tags are keyword-based, not verified facts.")


if __name__ == "__main__":
    main()
