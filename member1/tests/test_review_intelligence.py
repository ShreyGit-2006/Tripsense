from pathlib import Path
import sys

import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from review_intelligence import analyse_reviews, aggregate_hotels, load_inputs


def test_sentiment_and_topics():
    df = pd.DataFrame({
        "review_id": ["r1", "r2"],
        "hotel_id": ["h1", "h1"],
        "review_text": [
            "Excellent clean room and friendly staff; breakfast was delicious.",
            "Terrible dirty room and rude staff. Very overpriced.",
        ],
        "rating": [5, 1],
    })
    out = analyse_reviews(df)
    assert out.loc[0, "sentiment_label"] == "positive"
    assert out.loc[1, "sentiment_label"] == "negative"
    assert bool(out.loc[0, "topic_cleanliness"])
    assert bool(out.loc[1, "topic_value"])


def test_aggregate_hotel_reviews():
    df = analyse_reviews(pd.DataFrame({
        "hotel_id": ["h1", "h1"],
        "review_text": ["Excellent service", "Very bad dirty room"],
        "rating": [5, 1],
    }))
    summary = aggregate_hotels(df, pd.DataFrame({
        "hotel_id": ["h1"], "hotel_name": ["Example Hotel"], "district": ["Example district"]
    }))
    assert int(summary.loc[0, "review_count"]) == 2
    assert "hotel_name" in summary.columns


def test_load_inputs_deduplicates_and_drops_blank_reviews(tmp_path):
    reviews = pd.DataFrame({
        "review_id": ["r1", "r1", "r2"],
        "hotel_id": ["h1", "h1", "h2"],
        "review_text": ["Great stay", "Duplicate", "  "],
        "rating": [5, 5, 2],
    })
    reviews_path = tmp_path / "reviews.csv"
    hotels_path = tmp_path / "hotels.csv"
    reviews.to_csv(reviews_path, index=False)
    pd.DataFrame({"hotel_id": ["h1"]}).to_csv(hotels_path, index=False)
    clean, _ = load_inputs(reviews_path, hotels_path)
    assert len(clean) == 1
    assert clean.iloc[0]["review_text"] == "Great stay"
