"""
member3/tests/test_review_intelligence.py
==========================================
Tests for review_intelligence.py using small synthetic datasets.
No real project CSVs are touched; all test data is temporary.

Covers:
  - Data loading and validation
  - Sentiment analysis (VADER)
  - Topic detection (keyword matching)
  - Per-review enrichment pipeline
  - Hotel aggregation (metrics, topic rates, dominant_topic)
  - Topic summary
  - Full pipeline + output CSV generation
  - Edge cases: empty data, missing text, duplicate records, unknown hotel IDs
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from review_intelligence import (
    TOPIC_KEYWORDS,
    aggregate_by_hotel,
    analyse_sentiment,
    analyse_reviews_sentiment,
    analyse_reviews_topics,
    build_topic_summary,
    detect_topics,
    enrich_reviews,
    get_hotel_insights,
    load_hotels,
    load_reviews,
    run_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_reviews() -> pd.DataFrame:
    """5 hotel reviews across 2 hotels, with known sentiment and topics."""
    return pd.DataFrame({
        "review_id":   ["HR001", "HR002", "HR003", "HR004", "HR005"],
        "hotel_id":    ["HTL001", "HTL001", "HTL001", "HTL002", "HTL002"],
        "hotel_name":  ["Alpha Hotel"] * 3 + ["Beta Inn"] * 2,
        "district":    ["Mumbai"] * 3 + ["Delhi"] * 2,
        "rating":      [4.5, 3.0, 5.0, 2.0, 4.0],
        "sentiment":   ["positive", "neutral", "positive", "negative", "positive"],
        "review_text": [
            "The staff was extremely helpful and the rooms were spotlessly clean.",
            "Location is good. Food was okay but nothing special.",
            "Excellent pool and spa amenities! Great value for money.",
            "Terrible experience. AC wasn't working and the staff was rude.",
            "Comfortable stay near the beach. The breakfast buffet was delicious.",
        ],
        "stay_date": ["2026-01-10", "2026-02-15", "2026-03-05",
                      "2026-03-20", "2026-04-01"],
    })


@pytest.fixture()
def tiny_hotels() -> pd.DataFrame:
    return pd.DataFrame({
        "hotel_id":   ["HTL001", "HTL002"],
        "hotel_name": ["Alpha Hotel", "Beta Inn"],
        "district":   ["Mumbai", "Delhi"],
        "tier":       ["luxury", "mid_range"],
        "star_rating": ["5 Star", "3 Star"],
    })


@pytest.fixture()
def tiny_reviews_csv(tmp_path, tiny_reviews) -> Path:
    p = tmp_path / "reviews.csv"
    tiny_reviews.to_csv(p, index=False)
    return p


@pytest.fixture()
def tiny_hotels_csv(tmp_path, tiny_hotels) -> Path:
    p = tmp_path / "hotels.csv"
    tiny_hotels.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Tests: loading
# ---------------------------------------------------------------------------

class TestLoadReviews:
    def test_returns_empty_on_missing_file(self, tmp_path):
        df = load_reviews(tmp_path / "nonexistent.csv")
        assert df.empty

    def test_loads_valid_csv(self, tiny_reviews_csv, tiny_reviews):
        df = load_reviews(tiny_reviews_csv)
        assert len(df) == len(tiny_reviews)
        assert "hotel_id" in df.columns

    def test_deduplicates_on_review_id(self, tmp_path, tiny_reviews):
        dup = pd.concat([tiny_reviews, tiny_reviews.iloc[[0]]], ignore_index=True)
        p = tmp_path / "dup.csv"
        dup.to_csv(p, index=False)
        df = load_reviews(p)
        assert len(df) == len(tiny_reviews), "Duplicate row not removed"

    def test_drops_rows_with_empty_review_text(self, tmp_path, tiny_reviews):
        blank = tiny_reviews.copy()
        blank.loc[0, "review_text"] = ""
        p = tmp_path / "blank.csv"
        blank.to_csv(p, index=False)
        df = load_reviews(p)
        assert len(df) == len(tiny_reviews) - 1

    def test_drops_rows_missing_hotel_id(self, tmp_path, tiny_reviews):
        missing = tiny_reviews.copy()
        missing.loc[0, "hotel_id"] = None
        p = tmp_path / "missing.csv"
        missing.to_csv(p, index=False)
        df = load_reviews(p)
        assert len(df) == len(tiny_reviews) - 1

    def test_rating_is_numeric(self, tiny_reviews_csv):
        df = load_reviews(tiny_reviews_csv)
        assert pd.api.types.is_numeric_dtype(df["rating"])


class TestLoadHotels:
    def test_returns_empty_on_missing(self, tmp_path):
        df = load_hotels(tmp_path / "no.csv")
        assert df.empty

    def test_loads_correctly(self, tiny_hotels_csv, tiny_hotels):
        df = load_hotels(tiny_hotels_csv)
        assert len(df) == len(tiny_hotels)


# ---------------------------------------------------------------------------
# Tests: sentiment analysis
# ---------------------------------------------------------------------------

class TestAnalyseSentiment:
    def test_positive_text(self):
        result = analyse_sentiment("Excellent service! Very clean and friendly staff.")
        assert result["vader_label"] == "positive"
        assert result["vader_compound"] > 0.05

    def test_negative_text(self):
        result = analyse_sentiment("Terrible stay. Room was dirty and staff was rude.")
        assert result["vader_label"] == "negative"
        assert result["vader_compound"] < -0.05

    def test_empty_text_returns_neutral(self):
        result = analyse_sentiment("")
        assert result["vader_label"] == "neutral"
        assert result["vader_compound"] == 0.0

    def test_all_required_keys_present(self):
        result = analyse_sentiment("Nice place.")
        for key in ["vader_compound", "vader_pos", "vader_neu", "vader_neg", "vader_label"]:
            assert key in result

    def test_compound_in_range(self):
        for text in ["Great!", "Terrible!", "Average."]:
            r = analyse_sentiment(text)
            assert -1.0 <= r["vader_compound"] <= 1.0

    def test_label_matches_compound(self):
        for text in [
            "Amazing wonderful perfect",
            "Awful horrible disgusting",
        ]:
            r = analyse_sentiment(text)
            if r["vader_compound"] >= 0.05:
                assert r["vader_label"] == "positive"
            elif r["vader_compound"] <= -0.05:
                assert r["vader_label"] == "negative"
            else:
                assert r["vader_label"] == "neutral"


class TestAnalyseReviewsSentiment:
    def test_adds_vader_columns(self, tiny_reviews):
        result = analyse_reviews_sentiment(tiny_reviews)
        for col in ["vader_compound", "vader_label"]:
            assert col in result.columns

    def test_empty_df_returns_empty(self):
        result = analyse_reviews_sentiment(pd.DataFrame())
        assert result.empty

    def test_row_count_preserved(self, tiny_reviews):
        result = analyse_reviews_sentiment(tiny_reviews)
        assert len(result) == len(tiny_reviews)


# ---------------------------------------------------------------------------
# Tests: topic detection
# ---------------------------------------------------------------------------

class TestDetectTopics:
    def test_cleanliness_detected(self):
        r = detect_topics("The room was spotlessly clean.")
        assert r["topic_cleanliness"] == 1

    def test_food_detected(self):
        r = detect_topics("The breakfast buffet was excellent.")
        assert r["topic_food"] == 1

    def test_unrelated_text_no_topic(self):
        r = detect_topics("xyzzy quux foo bar blorp")
        for v in r.values():
            assert v == 0

    def test_all_topics_present_in_output(self):
        r = detect_topics("anything")
        for topic in TOPIC_KEYWORDS:
            assert f"topic_{topic}" in r

    def test_multiple_topics_detected(self):
        r = detect_topics("The staff was rude and the room was dirty. Overpriced.")
        assert r["topic_staff"] == 1
        assert r["topic_cleanliness"] == 1
        assert r["topic_value"] == 1


class TestAnalyseReviewsTopics:
    def test_adds_topic_columns(self, tiny_reviews):
        result = analyse_reviews_topics(tiny_reviews)
        for topic in TOPIC_KEYWORDS:
            assert f"topic_{topic}" in result.columns

    def test_empty_df_returns_empty(self):
        result = analyse_reviews_topics(pd.DataFrame())
        assert result.empty

    def test_topic_values_binary(self, tiny_reviews):
        result = analyse_reviews_topics(tiny_reviews)
        topic_cols = [f"topic_{t}" for t in TOPIC_KEYWORDS]
        for col in topic_cols:
            assert result[col].isin([0, 1]).all(), f"{col} has non-binary values"


# ---------------------------------------------------------------------------
# Tests: enrich_reviews
# ---------------------------------------------------------------------------

class TestEnrichReviews:
    def test_adds_both_sentiment_and_topics(self, tiny_reviews):
        result = enrich_reviews(tiny_reviews)
        assert "vader_compound" in result.columns
        assert "vader_label" in result.columns
        assert "topic_cleanliness" in result.columns

    def test_empty_returns_empty(self):
        result = enrich_reviews(pd.DataFrame())
        assert result.empty

    def test_row_count_preserved(self, tiny_reviews):
        result = enrich_reviews(tiny_reviews)
        assert len(result) == len(tiny_reviews)

    def test_review_text_not_lost(self, tiny_reviews):
        result = enrich_reviews(tiny_reviews)
        assert "review_text" in result.columns


# ---------------------------------------------------------------------------
# Tests: aggregate_by_hotel
# ---------------------------------------------------------------------------

class TestAggregateByHotel:
    def test_one_row_per_hotel(self, tiny_reviews, tiny_hotels):
        enriched = enrich_reviews(tiny_reviews)
        result = aggregate_by_hotel(enriched, tiny_hotels)
        assert len(result) == 2  # HTL001 and HTL002
        assert set(result["hotel_id"]) == {"HTL001", "HTL002"}

    def test_review_count_correct(self, tiny_reviews, tiny_hotels):
        enriched = enrich_reviews(tiny_reviews)
        result = aggregate_by_hotel(enriched, tiny_hotels)
        htl001 = result[result["hotel_id"] == "HTL001"]
        assert int(htl001["review_count"].iloc[0]) == 3

    def test_mean_star_rating_present(self, tiny_reviews, tiny_hotels):
        enriched = enrich_reviews(tiny_reviews)
        result = aggregate_by_hotel(enriched, tiny_hotels)
        assert "mean_star_rating" in result.columns
        # HTL001 mean = (4.5+3.0+5.0)/3 = 4.1667
        htl001 = result[result["hotel_id"] == "HTL001"]
        assert abs(float(htl001["mean_star_rating"].iloc[0]) - 4.1667) < 0.01

    def test_mean_vader_compound_in_range(self, tiny_reviews):
        enriched = enrich_reviews(tiny_reviews)
        result = aggregate_by_hotel(enriched)
        if "mean_vader_compound" in result.columns:
            assert result["mean_vader_compound"].between(-1.0, 1.0).all()

    def test_dominant_topic_present(self, tiny_reviews):
        enriched = enrich_reviews(tiny_reviews)
        result = aggregate_by_hotel(enriched)
        assert "dominant_topic" in result.columns
        assert result["dominant_topic"].notna().any()

    def test_hotel_metadata_merged(self, tiny_reviews, tiny_hotels):
        enriched = enrich_reviews(tiny_reviews)
        result = aggregate_by_hotel(enriched, tiny_hotels)
        assert "district" in result.columns

    def test_empty_reviews_returns_empty(self, tiny_hotels):
        result = aggregate_by_hotel(pd.DataFrame(), tiny_hotels)
        assert result.empty

    def test_rate_topic_columns_in_range(self, tiny_reviews):
        enriched = enrich_reviews(tiny_reviews)
        result = aggregate_by_hotel(enriched)
        rate_cols = [c for c in result.columns if c.startswith("rate_")]
        for col in rate_cols:
            assert result[col].between(0.0, 1.0001).all(), f"{col} out of [0,1]"


# ---------------------------------------------------------------------------
# Tests: build_topic_summary
# ---------------------------------------------------------------------------

class TestBuildTopicSummary:
    def test_returns_one_row_per_topic(self, tiny_reviews):
        enriched = enrich_reviews(tiny_reviews)
        result = build_topic_summary(enriched)
        assert len(result) == len(TOPIC_KEYWORDS)
        assert set(result["topic"]) == set(TOPIC_KEYWORDS.keys())

    def test_mention_rate_in_range(self, tiny_reviews):
        enriched = enrich_reviews(tiny_reviews)
        result = build_topic_summary(enriched)
        assert result["mention_rate"].between(0.0, 1.0001).all()

    def test_mention_count_nonneg(self, tiny_reviews):
        enriched = enrich_reviews(tiny_reviews)
        result = build_topic_summary(enriched)
        assert (result["mention_count"] >= 0).all()

    def test_mean_vader_for_mentioned_topic(self, tiny_reviews):
        enriched = enrich_reviews(tiny_reviews)
        result = build_topic_summary(enriched)
        mentioned = result[result["mention_count"] > 0]
        if not mentioned.empty:
            assert mentioned["mean_vader_compound_when_mentioned"].notna().any()

    def test_empty_reviews_returns_empty(self):
        result = build_topic_summary(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# Tests: full pipeline + output files
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def test_writes_two_output_files(self, tmp_path, tiny_reviews_csv, tiny_hotels_csv):
        insights_out = tmp_path / "insights.csv"
        topic_out    = tmp_path / "topics.csv"
        insights, topics = run_pipeline(
            reviews_path=tiny_reviews_csv,
            hotels_path=tiny_hotels_csv,
            insights_out=insights_out,
            topic_out=topic_out,
        )
        assert insights_out.exists(), "review_insights.csv not written"
        assert topic_out.exists(), "review_topic_summary.csv not written"

    def test_insight_schema(self, tmp_path, tiny_reviews_csv, tiny_hotels_csv):
        insights_out = tmp_path / "insights.csv"
        topic_out    = tmp_path / "topics.csv"
        insights, _ = run_pipeline(
            reviews_path=tiny_reviews_csv,
            hotels_path=tiny_hotels_csv,
            insights_out=insights_out,
            topic_out=topic_out,
        )
        for col in ["hotel_id", "review_count"]:
            assert col in insights.columns

    def test_topic_schema(self, tmp_path, tiny_reviews_csv, tiny_hotels_csv):
        insights_out = tmp_path / "insights.csv"
        topic_out    = tmp_path / "topics.csv"
        _, topics = run_pipeline(
            reviews_path=tiny_reviews_csv,
            hotels_path=tiny_hotels_csv,
            insights_out=insights_out,
            topic_out=topic_out,
        )
        for col in ["topic", "mention_count", "mention_rate"]:
            assert col in topics.columns

    def test_source_not_modified(self, tmp_path, tiny_reviews_csv, tiny_hotels_csv):
        import os
        r_before = os.path.getmtime(tiny_reviews_csv)
        h_before = os.path.getmtime(tiny_hotels_csv)

        run_pipeline(
            reviews_path=tiny_reviews_csv,
            hotels_path=tiny_hotels_csv,
            insights_out=tmp_path / "i.csv",
            topic_out=tmp_path / "t.csv",
        )
        assert os.path.getmtime(tiny_reviews_csv) == r_before
        assert os.path.getmtime(tiny_hotels_csv)  == h_before

    def test_empty_reviews_pipeline_returns_empty(self, tmp_path, tiny_hotels_csv):
        empty_p = tmp_path / "empty_reviews.csv"
        # Write valid-schema but 0-data CSV
        pd.DataFrame(columns=[
            "review_id","hotel_id","review_text","rating","stay_date"
        ]).to_csv(empty_p, index=False)

        insights, topics = run_pipeline(
            reviews_path=empty_p,
            hotels_path=tiny_hotels_csv,
            insights_out=tmp_path / "i.csv",
            topic_out=tmp_path / "t.csv",
        )
        assert insights.empty or len(insights) == 0

    def test_missing_reviews_file_returns_empty(self, tmp_path, tiny_hotels_csv):
        insights, topics = run_pipeline(
            reviews_path=tmp_path / "no_file.csv",
            hotels_path=tiny_hotels_csv,
            insights_out=tmp_path / "i.csv",
            topic_out=tmp_path / "t.csv",
        )
        assert insights.empty


# ---------------------------------------------------------------------------
# Tests: get_hotel_insights
# ---------------------------------------------------------------------------

class TestGetHotelInsights:
    def test_filter_by_hotel_id(self, tmp_path, tiny_reviews_csv):
        result = get_hotel_insights("HTL001", reviews_path=tiny_reviews_csv)
        assert not result.empty
        assert all(result["hotel_id"] == "HTL001")

    def test_unknown_hotel_id_returns_empty(self, tmp_path, tiny_reviews_csv):
        result = get_hotel_insights("HTLXXX", reviews_path=tiny_reviews_csv)
        assert result.empty

    def test_result_contains_sentiment(self, tmp_path, tiny_reviews_csv):
        result = get_hotel_insights("HTL001", reviews_path=tiny_reviews_csv)
        assert "vader_label" in result.columns
