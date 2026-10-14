"""
member3/tests/test_rag.py
==========================
Tests for rag.py (RAG Travel Assistant).

All tests run offline:
  - No real project CSVs used (all synthetic temp data)
  - No LLM API keys required
  - TRIPSENSE_LLM_PROVIDER is cleared / absent throughout

Covers:
  - Data loading (load_all_data, missing files)
  - TripSenseRetriever: basic retrieval, empty corpus, zero-match query
  - build_context: places, hotels, review_insights, best_months included correctly
  - _format_local_answer: answer structure, evidence IDs, can_answer flag
  - generate_answer: falls back to local when no LLM configured
  - answer_query: end-to-end, empty query, unrecognised query
  - Evidence traceability: IDs in answer correspond to retrieved records
  - Unsupported / missing data cases
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Remove LLM env vars for all tests in this module
os.environ.pop("TRIPSENSE_LLM_PROVIDER", None)
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag import (
    TripSenseRetriever,
    _format_local_answer,
    answer_query,
    build_context,
    generate_answer,
    load_all_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_places() -> pd.DataFrame:
    return pd.DataFrame({
        "place_id":   ["TST001", "TST002", "TST003"],
        "name":       ["Sunset Beach", "Mountain Fort", "Ancient Temple"],
        "category":   ["beach", "heritage", "temple"],
        "district":   ["Goa", "Rajasthan", "Tamil Nadu"],
        "description": [
            "A beautiful sandy beach perfect for swimming and sunsets.",
            "A historic fort on a hilltop with panoramic views.",
            "An ancient Dravidian temple with intricate carvings.",
        ],
        "tags_text":  [
            "beach sunset swimming water sports goa",
            "fort heritage rajasthan history trekking",
            "temple history culture heritage tamil",
        ],
        "suitable_for": ["family,friends", "adventure,history", "family,culture"],
        "best_months":  ["Oct, Nov, Dec", "Nov, Feb, Mar", "Jan, Feb, Mar"],
        "rating":       [4.3, 4.1, 4.5],
        "review_count": [120, 80, 200],
    })


@pytest.fixture()
def tiny_hotels() -> pd.DataFrame:
    return pd.DataFrame({
        "hotel_id":   ["HTL_A", "HTL_B"],
        "hotel_name": ["Goa Resort", "Fort View Inn"],
        "district":   ["Goa", "Rajasthan"],
        "tier":       ["luxury", "mid_range"],
        "price_range": ["INR 8000+/night", "INR 2000-4000/night"],
        "amenities":  ["Pool, Spa, WiFi", "WiFi, Parking, Restaurant"],
    })


@pytest.fixture()
def tiny_insights() -> pd.DataFrame:
    return pd.DataFrame({
        "hotel_id":           ["HTL_A", "HTL_B"],
        "review_count":       [200, 80],
        "mean_star_rating":   [4.6, 3.8],
        "mean_vader_compound":[0.52, 0.21],
        "dominant_topic":     ["amenities", "location"],
        "district":           ["Goa", "Rajasthan"],
    })


@pytest.fixture()
def tiny_best_months() -> pd.DataFrame:
    return pd.DataFrame({
        "place_id":         ["TST001", "TST001", "TST002"],
        "name":             ["Sunset Beach", "Sunset Beach", "Mountain Fort"],
        "month_name":       ["October", "November", "February"],
        "rank_within_place":[1, 2, 1],
        "composite_score":  [0.91, 0.88, 0.85],
        "reasons":          [
            "comfortable weather; low crowd expected",
            "comfortable weather; bridge day",
            "comfortable weather; low crowd expected",
        ],
    })


@pytest.fixture()
def data(tiny_places, tiny_hotels, tiny_insights, tiny_best_months) -> dict:
    return {
        "places":          tiny_places,
        "hotels":          tiny_hotels,
        "review_insights": tiny_insights,
        "best_months":     tiny_best_months,
    }


@pytest.fixture()
def retriever(tiny_places) -> TripSenseRetriever:
    return TripSenseRetriever(tiny_places)


# ---------------------------------------------------------------------------
# Tests: load_all_data
# ---------------------------------------------------------------------------

class TestLoadAllData:
    def test_returns_four_keys(self, tmp_path):
        result = load_all_data(
            places_path=tmp_path / "no.csv",
            hotels_path=tmp_path / "no.csv",
            insights_path=tmp_path / "no.csv",
            best_months_path=tmp_path / "no.csv",
        )
        assert set(result.keys()) == {"places", "hotels", "review_insights", "best_months"}

    def test_missing_files_return_empty_dfs(self, tmp_path):
        result = load_all_data(
            places_path=tmp_path / "no.csv",
            hotels_path=tmp_path / "no.csv",
            insights_path=tmp_path / "no.csv",
            best_months_path=tmp_path / "no.csv",
        )
        for k, df in result.items():
            assert isinstance(df, pd.DataFrame), f"{k} is not a DataFrame"
            assert df.empty, f"{k} should be empty"

    def test_loads_csv_files(self, tmp_path, tiny_places, tiny_hotels):
        p_path = tmp_path / "p.csv"
        h_path = tmp_path / "h.csv"
        tiny_places.to_csv(p_path, index=False)
        tiny_hotels.to_csv(h_path, index=False)
        result = load_all_data(
            places_path=p_path,
            hotels_path=h_path,
            insights_path=tmp_path / "no.csv",
            best_months_path=tmp_path / "no.csv",
        )
        assert len(result["places"]) == 3
        assert len(result["hotels"]) == 2


# ---------------------------------------------------------------------------
# Tests: TripSenseRetriever
# ---------------------------------------------------------------------------

class TestTripSenseRetriever:
    def test_retrieve_returns_dataframe(self, retriever):
        result = retriever.retrieve("beach sunset swimming")
        assert isinstance(result, pd.DataFrame)

    def test_beach_query_finds_beach_place(self, retriever):
        result = retriever.retrieve("beach sunset swimming")
        assert not result.empty
        assert "TST001" in result["place_id"].values

    def test_heritage_fort_query(self, retriever):
        result = retriever.retrieve("historic fort heritage trekking")
        assert not result.empty
        assert "TST002" in result["place_id"].values

    def test_retrieval_score_column_present(self, retriever):
        result = retriever.retrieve("temple culture")
        if not result.empty:
            assert "retrieval_score" in result.columns

    def test_retrieval_score_in_range(self, retriever):
        result = retriever.retrieve("beach temple fort")
        if not result.empty:
            assert result["retrieval_score"].between(0.0, 1.0001).all()

    def test_empty_query_returns_empty(self, retriever):
        result = retriever.retrieve("")
        assert result.empty

    def test_empty_corpus_returns_empty(self):
        r = TripSenseRetriever(pd.DataFrame())
        result = r.retrieve("anything")
        assert result.empty

    def test_top_k_limits_results(self, retriever):
        result = retriever.retrieve("india travel", top_k=2)
        assert len(result) <= 2

    def test_gibberish_query_returns_empty_or_low_scores(self, retriever):
        result = retriever.retrieve("xyzzy quux blorp zap123")
        # Either no results or all scores very low
        if not result.empty:
            assert result["retrieval_score"].max() < 0.5


# ---------------------------------------------------------------------------
# Tests: build_context
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_returns_required_keys(self, data, retriever):
        ctx = build_context("beaches in Goa", data, retriever)
        for key in ["query", "places", "hotels", "review_insights",
                    "best_months", "evidence_ids", "dataset_note"]:
            assert key in ctx

    def test_query_preserved_in_context(self, data, retriever):
        q = "What are the best waterfalls to visit?"
        ctx = build_context(q, data, retriever)
        assert ctx["query"] == q

    def test_evidence_ids_have_place_ids(self, data, retriever):
        ctx = build_context("beach sunset Goa", data, retriever)
        assert "place_ids" in ctx["evidence_ids"]
        if ctx["places"]:
            assert len(ctx["evidence_ids"]["place_ids"]) > 0

    def test_hotels_matched_to_district(self, data, retriever):
        ctx = build_context("beach Goa swimming", data, retriever)
        # TST001 (Goa) should be retrieved, HTL_A (Goa) should appear
        hotel_ids = ctx["evidence_ids"].get("hotel_ids", [])
        # Not guaranteed but if Goa place retrieved, Goa hotel should be there
        if "TST001" in ctx["evidence_ids"].get("place_ids", []):
            assert "HTL_A" in hotel_ids

    def test_best_months_for_retrieved_places(self, data, retriever):
        ctx = build_context("Sunset Beach travel", data, retriever)
        if "TST001" in ctx["evidence_ids"].get("place_ids", []):
            bm_pids = [r["place_id"] for r in ctx["best_months"]]
            assert "TST001" in bm_pids

    def test_dataset_note_present(self, data, retriever):
        ctx = build_context("temples", data, retriever)
        assert len(ctx["dataset_note"]) > 10

    def test_empty_places_data(self, retriever):
        empty_data = {
            "places": pd.DataFrame(),
            "hotels": pd.DataFrame(),
            "review_insights": pd.DataFrame(),
            "best_months": pd.DataFrame(),
        }
        ctx = build_context("anything", empty_data, retriever)
        assert ctx["places"] == []


# ---------------------------------------------------------------------------
# Tests: _format_local_answer
# ---------------------------------------------------------------------------

class TestFormatLocalAnswer:
    def _make_context(self, data, retriever, query="beach in Goa"):
        return build_context(query, data, retriever)

    def test_returns_required_keys(self, data, retriever):
        ctx = self._make_context(data, retriever)
        ans = _format_local_answer(ctx)
        for key in ["answer", "evidence_ids", "source", "can_answer"]:
            assert key in ans

    def test_source_is_local_fallback(self, data, retriever):
        ctx = self._make_context(data, retriever)
        ans = _format_local_answer(ctx)
        assert ans["source"] == "local_fallback"

    def test_answer_is_string(self, data, retriever):
        ctx = self._make_context(data, retriever)
        ans = _format_local_answer(ctx)
        assert isinstance(ans["answer"], str)

    def test_can_answer_true_when_places_found(self, data, retriever):
        ctx = self._make_context(data, retriever, "beach Goa")
        if ctx["places"]:
            ans = _format_local_answer(ctx)
            assert ans["can_answer"] is True

    def test_can_answer_false_when_no_results(self, retriever):
        empty_ctx = {
            "query": "xyzzy quux nothing",
            "places": [],
            "hotels": [],
            "review_insights": [],
            "best_months": [],
            "evidence_ids": {"place_ids": [], "hotel_ids": []},
            "dataset_note": "test",
        }
        ans = _format_local_answer(empty_ctx)
        assert ans["can_answer"] is False
        assert "xyzzy quux nothing" in ans["answer"]

    def test_evidence_ids_propagated(self, data, retriever):
        ctx = self._make_context(data, retriever, "beach sunset")
        ans = _format_local_answer(ctx)
        assert ans["evidence_ids"] is ctx["evidence_ids"]

    def test_answer_contains_place_id(self, data, retriever):
        ctx = self._make_context(data, retriever, "beach sunset Goa")
        ans = _format_local_answer(ctx)
        if ctx["places"]:
            # At least one place_id should appear in answer
            any_id = ctx["evidence_ids"]["place_ids"][0]
            assert any_id in ans["answer"]


# ---------------------------------------------------------------------------
# Tests: generate_answer (no LLM)
# ---------------------------------------------------------------------------

class TestGenerateAnswer:
    def test_falls_back_to_local_when_no_provider(self, data, retriever):
        ctx = build_context("beach in Goa", data, retriever)
        ans = generate_answer(ctx)
        assert ans["source"] == "local_fallback"

    def test_answer_is_string(self, data, retriever):
        ctx = build_context("temple culture", data, retriever)
        ans = generate_answer(ctx)
        assert isinstance(ans["answer"], str)
        assert len(ans["answer"]) > 0

    def test_evidence_ids_structure(self, data, retriever):
        ctx = build_context("fort rajasthan", data, retriever)
        ans = generate_answer(ctx)
        assert "place_ids" in ans["evidence_ids"]
        assert "hotel_ids" in ans["evidence_ids"]


# ---------------------------------------------------------------------------
# Tests: answer_query (end-to-end)
# ---------------------------------------------------------------------------

class TestAnswerQuery:
    def test_basic_query_returns_answer(self, data, retriever):
        result = answer_query("best beaches in Goa", data=data, retriever=retriever)
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_empty_query_returns_prompt(self, data, retriever):
        result = answer_query("", data=data, retriever=retriever)
        assert result["can_answer"] is False
        assert "Please provide" in result["answer"]

    def test_whitespace_query_returns_prompt(self, data, retriever):
        result = answer_query("   ", data=data, retriever=retriever)
        assert result["can_answer"] is False

    def test_result_has_required_keys(self, data, retriever):
        result = answer_query("temples in Tamil Nadu", data=data, retriever=retriever)
        for key in ["answer", "evidence_ids", "source", "can_answer", "context_used"]:
            assert key in result

    def test_evidence_traceability(self, data, retriever):
        """Every place_id in evidence_ids must have been in the original data."""
        result = answer_query("heritage fort history", data=data, retriever=retriever)
        original_ids = set(data["places"]["place_id"])
        for pid in result["evidence_ids"]["place_ids"]:
            assert pid in original_ids, f"Fabricated place_id: {pid}"

    def test_hotel_evidence_traceability(self, data, retriever):
        """Every hotel_id in evidence_ids must have been in the original data."""
        result = answer_query("beach Goa", data=data, retriever=retriever)
        original_hids = set(data["hotels"]["hotel_id"])
        for hid in result["evidence_ids"]["hotel_ids"]:
            assert hid in original_hids, f"Fabricated hotel_id: {hid}"

    def test_unsupported_query_graceful(self, data, retriever):
        result = answer_query(
            "xyzzy quux blorp zap 1234567890",
            data=data,
            retriever=retriever,
        )
        assert isinstance(result["answer"], str)
        # can_answer may be False (no matches) but must not raise

    def test_empty_datasets_graceful(self):
        empty_data = {
            "places": pd.DataFrame(),
            "hotels": pd.DataFrame(),
            "review_insights": pd.DataFrame(),
            "best_months": pd.DataFrame(),
        }
        result = answer_query("where should I visit", data=empty_data)
        assert result["can_answer"] is False
        assert isinstance(result["answer"], str)

    def test_source_is_local_fallback_without_api_key(self, data, retriever):
        result = answer_query("beach sunset Goa", data=data, retriever=retriever)
        assert result["source"] == "local_fallback"

    def test_context_used_in_result(self, data, retriever):
        result = answer_query("temple culture", data=data, retriever=retriever)
        assert "context_used" in result
        assert isinstance(result["context_used"], dict)

    def test_uses_custom_data_paths(self, tmp_path, tiny_places, tiny_hotels):
        """When custom paths are provided and files exist, data is loaded correctly."""
        p = tmp_path / "p.csv"
        h = tmp_path / "h.csv"
        tiny_places.to_csv(p, index=False)
        tiny_hotels.to_csv(h, index=False)
        result = answer_query(
            "beach sunset",
            places_path=p,
            hotels_path=h,
            insights_path=tmp_path / "no.csv",
            best_months_path=tmp_path / "no.csv",
        )
        assert isinstance(result["answer"], str)
