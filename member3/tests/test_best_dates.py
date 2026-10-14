"""
member3/tests/test_best_dates.py
=================================
Unit tests for best_dates.py using small, self-contained synthetic datasets.
All tests run without touching the real project CSVs.
"""
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure member3/src is importable regardless of CWD
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from best_dates import (
    DEFAULT_DAILY_WEIGHTS,
    DEFAULT_MONTHLY_WEIGHTS,
    _aggregate_crowd_by_month,
    _safe_minmax,
    generate_outputs,
    get_best_dates,
    get_best_months,
    load_holiday_crowd,
    load_places,
    score_dates,
    score_months,
)


# ---------------------------------------------------------------------------
# Fixtures: tiny synthetic DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_places() -> pd.DataFrame:
    """Two places: one comfortable in Jan-Mar only, one year-round."""
    data = {
        "place_id": ["TST001", "TST002"],
        "name":     ["Winter Place", "Summer Place"],
        "district": ["Dist A", "Dist B"],
        "category": ["waterfall", "beach"],
        "comfort_jan": [1, 0],
        "comfort_feb": [1, 0],
        "comfort_mar": [1, 0],
        "comfort_apr": [0, 1],
        "comfort_may": [0, 1],
        "comfort_jun": [0, 1],
        "comfort_jul": [0, 1],
        "comfort_aug": [0, 1],
        "comfort_sep": [0, 1],
        "comfort_oct": [0, 0],
        "comfort_nov": [0, 0],
        "comfort_dec": [1, 0],
        "rating":      [4.2, 3.8],
        "review_count":[50, 30],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def tiny_crowd() -> pd.DataFrame:
    """Minimal crowd calendar: 4 days across 2 months."""
    rows = [
        # Jan
        {
            "date": pd.Timestamp("2026-01-01"), "month": 1, "is_holiday": 0,
            "holiday_name": "", "holiday_type": "", "holiday_penalty": 0,
            "is_bridge_day": 0, "bridge_penalty": 0,
            "school_vacation_proxy": 0, "school_vacation_penalty": 0,
            "crowd_proxy_score": 0.2, "crowd_proxy_normalized": 0.2,
            "base_visit_score": 0.8, "is_long_weekend_candidate": 0,
            "weekend_penalty": 0, "weekday": "Thursday", "season": "winter",
        },
        {
            "date": pd.Timestamp("2026-01-26"), "month": 1, "is_holiday": 1,
            "holiday_name": "Republic Day", "holiday_type": "national",
            "holiday_penalty": 1.0, "is_bridge_day": 0, "bridge_penalty": 0,
            "school_vacation_proxy": 0, "school_vacation_penalty": 0,
            "crowd_proxy_score": 0.5, "crowd_proxy_normalized": 0.5,
            "base_visit_score": 1.0, "is_long_weekend_candidate": 1,
            "weekend_penalty": 0, "weekday": "Monday", "season": "winter",
        },
        # Jun
        {
            "date": pd.Timestamp("2026-06-01"), "month": 6, "is_holiday": 0,
            "holiday_name": "", "holiday_type": "", "holiday_penalty": 0,
            "is_bridge_day": 0, "bridge_penalty": 0,
            "school_vacation_proxy": 1, "school_vacation_penalty": 0.5,
            "crowd_proxy_score": 0.8, "crowd_proxy_normalized": 0.8,
            "base_visit_score": 0.3, "is_long_weekend_candidate": 0,
            "weekend_penalty": 0, "weekday": "Monday", "season": "summer",
        },
        {
            "date": pd.Timestamp("2026-06-15"), "month": 6, "is_holiday": 0,
            "holiday_name": "", "holiday_type": "", "holiday_penalty": 0,
            "is_bridge_day": 1, "bridge_penalty": 0.3,
            "school_vacation_proxy": 1, "school_vacation_penalty": 0.5,
            "crowd_proxy_score": 0.9, "crowd_proxy_normalized": 0.9,
            "base_visit_score": 0.4, "is_long_weekend_candidate": 1,
            "weekend_penalty": 0, "weekday": "Monday", "season": "summer",
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: data loading
# ---------------------------------------------------------------------------

class TestLoadFunctions:
    def test_load_places_returns_empty_on_missing(self, tmp_path):
        df = load_places(tmp_path / "nonexistent.csv")
        assert df.empty

    def test_load_crowd_returns_empty_on_missing(self, tmp_path):
        df = load_holiday_crowd(tmp_path / "nonexistent.csv")
        assert df.empty

    def test_load_places_from_csv(self, tmp_path, tiny_places):
        p = tmp_path / "places.csv"
        tiny_places.to_csv(p, index=False)
        df = load_places(p)
        assert len(df) == 2
        assert "place_id" in df.columns

    def test_load_crowd_from_csv(self, tmp_path, tiny_crowd):
        p = tmp_path / "crowd.csv"
        tiny_crowd.to_csv(p, index=False)
        df = load_holiday_crowd(p)
        assert len(df) == 4
        assert pd.api.types.is_datetime64_any_dtype(df["date"])


# ---------------------------------------------------------------------------
# Tests: _safe_minmax
# ---------------------------------------------------------------------------

class TestSafeMinmax:
    def test_normal_normalization(self):
        s = pd.Series([0.0, 0.5, 1.0])
        n = _safe_minmax(s)
        assert abs(n.min() - 0.0) < 1e-6
        assert abs(n.max() - 1.0) < 1e-6

    def test_constant_series_returns_half(self):
        s = pd.Series([3.0, 3.0, 3.0])
        n = _safe_minmax(s)
        assert all(n == 0.5)


# ---------------------------------------------------------------------------
# Tests: monthly aggregation
# ---------------------------------------------------------------------------

class TestAggregateCrowdByMonth:
    def test_months_covered(self, tiny_crowd):
        agg = _aggregate_crowd_by_month(tiny_crowd)
        # All 12 months present
        assert len(agg) == 12
        assert set(agg["month"]) == set(range(1, 13))

    def test_jan_mean_crowd(self, tiny_crowd):
        agg = _aggregate_crowd_by_month(tiny_crowd)
        jan = agg[agg["month"] == 1]
        # mean of 0.2 and 0.5 = 0.35
        assert abs(float(jan["mean_crowd_normalized"].iloc[0]) - 0.35) < 1e-6

    def test_empty_crowd_returns_defaults(self):
        agg = _aggregate_crowd_by_month(pd.DataFrame())
        assert len(agg) == 12
        assert all(agg["mean_crowd_normalized"] == 0.0)


# ---------------------------------------------------------------------------
# Tests: score_months
# ---------------------------------------------------------------------------

class TestScoreMonths:
    def test_returns_12_months_per_place(self, tiny_places, tiny_crowd):
        result = score_months(tiny_places, tiny_crowd)
        for pid in ["TST001", "TST002"]:
            rows = result[result["place_id"] == pid]
            assert len(rows) == 12, f"{pid} should have 12 month rows"

    def test_required_columns_present(self, tiny_places, tiny_crowd):
        result = score_months(tiny_places, tiny_crowd)
        for col in [
            "place_id", "name", "district", "month", "month_name",
            "comfort_flag", "low_crowd_score", "composite_score",
            "composite_score_norm", "rank_within_place", "reasons",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_composite_score_in_range(self, tiny_places, tiny_crowd):
        result = score_months(tiny_places, tiny_crowd)
        assert result["composite_score"].between(0.0, 1.0001).all()

    def test_normalized_score_in_range(self, tiny_places, tiny_crowd):
        result = score_months(tiny_places, tiny_crowd)
        assert result["composite_score_norm"].between(0.0, 1.0001).all()

    def test_rank_min_is_1(self, tiny_places, tiny_crowd):
        result = score_months(tiny_places, tiny_crowd)
        assert result["rank_within_place"].min() == 1

    def test_rank_max_is_12(self, tiny_places, tiny_crowd):
        result = score_months(tiny_places, tiny_crowd)
        assert result["rank_within_place"].max() == 12

    def test_winter_place_ranks_jan_high(self, tiny_places, tiny_crowd):
        result = score_months(tiny_places, tiny_crowd)
        tst001 = result[result["place_id"] == "TST001"]
        jan_rank = int(tst001[tst001["month"] == 1]["rank_within_place"].iloc[0])
        # Jan should be ranked in top 3 for a winter place
        assert jan_rank <= 3

    def test_place_id_filter(self, tiny_places, tiny_crowd):
        result = score_months(tiny_places, tiny_crowd, place_id_filter=["TST001"])
        assert set(result["place_id"]) == {"TST001"}

    def test_district_filter(self, tiny_places, tiny_crowd):
        result = score_months(tiny_places, tiny_crowd, district_filter="Dist A")
        assert set(result["place_id"]) == {"TST001"}

    def test_empty_places_returns_empty(self, tiny_crowd):
        result = score_months(pd.DataFrame(), tiny_crowd)
        assert result.empty

    def test_invalid_weight_raises(self, tiny_places, tiny_crowd):
        with pytest.raises(ValueError):
            score_months(tiny_places, tiny_crowd, weights={"comfort": 0.9, "crowd": 0.9, "base_visit": 0.0, "bridge": 0.0})

    def test_reasons_column_not_empty(self, tiny_places, tiny_crowd):
        result = score_months(tiny_places, tiny_crowd)
        assert result["reasons"].notna().all()
        assert (result["reasons"].str.len() > 0).all()


# ---------------------------------------------------------------------------
# Tests: score_dates
# ---------------------------------------------------------------------------

class TestScoreDates:
    def test_returns_rows_for_each_place_day(self, tiny_places, tiny_crowd):
        result = score_dates(tiny_places, tiny_crowd)
        # 2 places × 4 days = 8 rows
        assert len(result) == 8

    def test_required_columns_present(self, tiny_places, tiny_crowd):
        result = score_dates(tiny_places, tiny_crowd)
        for col in [
            "place_id", "name", "date", "weekday", "comfort_flag",
            "low_crowd_score", "composite_score", "composite_score_norm",
            "rank_within_place", "reasons",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_composite_score_in_range(self, tiny_places, tiny_crowd):
        result = score_dates(tiny_places, tiny_crowd)
        assert result["composite_score"].between(0.0, 1.0001).all()

    def test_date_filter(self, tiny_places, tiny_crowd):
        result = score_dates(
            tiny_places, tiny_crowd,
            date_from="2026-01-01", date_to="2026-01-31"
        )
        assert all(
            pd.Timestamp(d) <= pd.Timestamp("2026-01-31") for d in result["date"]
        )

    def test_place_id_filter_dates(self, tiny_places, tiny_crowd):
        result = score_dates(tiny_places, tiny_crowd, place_id_filter=["TST002"])
        assert set(result["place_id"]) == {"TST002"}

    def test_empty_places_returns_empty(self, tiny_crowd):
        result = score_dates(pd.DataFrame(), tiny_crowd)
        assert result.empty

    def test_empty_crowd_returns_empty(self, tiny_places):
        result = score_dates(tiny_places, pd.DataFrame())
        assert result.empty

    def test_holiday_day_gets_bonus(self, tiny_places, tiny_crowd):
        result = score_dates(tiny_places, tiny_crowd)
        republic_day = result[
            (result["place_id"] == "TST001")
            & (result["date"].astype(str) == "2026-01-26")
        ]
        normal_jan = result[
            (result["place_id"] == "TST001")
            & (result["date"].astype(str) == "2026-01-01")
        ]
        assert len(republic_day) == 1
        assert len(normal_jan) == 1
        assert float(republic_day["composite_score"].iloc[0]) > float(normal_jan["composite_score"].iloc[0])


# ---------------------------------------------------------------------------
# Tests: get_best_months (public API)
# ---------------------------------------------------------------------------

class TestGetBestMonths:
    def test_returns_top_n_months(self, tmp_path, tiny_places, tiny_crowd):
        p_path = tmp_path / "places.csv"
        c_path = tmp_path / "crowd.csv"
        tiny_places.to_csv(p_path, index=False)
        tiny_crowd.to_csv(c_path, index=False)
        result = get_best_months(
            destination="Winter Place",
            top_n=2,
            places_path=p_path,
            crowd_path=c_path,
        )
        # rank_within_place uses method='min', so ties share the same rank.
        # All returned rows must have rank <= top_n.
        assert len(result) > 0
        assert all(result["rank_within_place"] <= 2)

    def test_unknown_destination_returns_empty(self, tmp_path, tiny_places, tiny_crowd):
        p_path = tmp_path / "places.csv"
        c_path = tmp_path / "crowd.csv"
        tiny_places.to_csv(p_path, index=False)
        tiny_crowd.to_csv(c_path, index=False)
        result = get_best_months(
            destination="NONEXISTENT_PLACE_XYZ",
            places_path=p_path,
            crowd_path=c_path,
        )
        assert result.empty or len(result) == 0


# ---------------------------------------------------------------------------
# Tests: get_best_dates (public API)
# ---------------------------------------------------------------------------

class TestGetBestDates:
    def test_returns_top_n_dates(self, tmp_path, tiny_places, tiny_crowd):
        p_path = tmp_path / "places.csv"
        c_path = tmp_path / "crowd.csv"
        tiny_places.to_csv(p_path, index=False)
        tiny_crowd.to_csv(c_path, index=False)
        result = get_best_dates(
            destination="Winter Place",
            top_n=2,
            places_path=p_path,
            crowd_path=c_path,
        )
        assert len(result) == 2
        assert all(result["rank_within_place"] <= 2)

    def test_date_range_filter_respected(self, tmp_path, tiny_places, tiny_crowd):
        p_path = tmp_path / "places.csv"
        c_path = tmp_path / "crowd.csv"
        tiny_places.to_csv(p_path, index=False)
        tiny_crowd.to_csv(c_path, index=False)
        result = get_best_dates(
            date_from="2026-06-01",
            date_to="2026-06-30",
            top_n=10,
            places_path=p_path,
            crowd_path=c_path,
        )
        if not result.empty:
            dates = pd.to_datetime(result["date"])
            assert (dates >= pd.Timestamp("2026-06-01")).all()
            assert (dates <= pd.Timestamp("2026-06-30")).all()


# ---------------------------------------------------------------------------
# Tests: generate_outputs
# ---------------------------------------------------------------------------

class TestGenerateOutputs:
    def test_writes_two_csv_files(self, tmp_path, tiny_places, tiny_crowd):
        p_path = tmp_path / "places.csv"
        c_path = tmp_path / "crowd.csv"
        m_out  = tmp_path / "months_out.csv"
        d_out  = tmp_path / "dates_out.csv"
        tiny_places.to_csv(p_path, index=False)
        tiny_crowd.to_csv(c_path, index=False)

        m_df, d_df = generate_outputs(
            places_path=p_path,
            crowd_path=c_path,
            months_out=m_out,
            dates_out=d_out,
        )
        assert m_out.exists(), "best_months_results.csv not written"
        assert d_out.exists(), "best_dates_results.csv not written"
        assert len(m_df) > 0
        assert len(d_df) > 0

    def test_source_csvs_not_modified(self, tmp_path, tiny_places, tiny_crowd):
        p_path = tmp_path / "places.csv"
        c_path = tmp_path / "crowd.csv"
        m_out  = tmp_path / "months_out.csv"
        d_out  = tmp_path / "dates_out.csv"
        tiny_places.to_csv(p_path, index=False)
        tiny_crowd.to_csv(c_path, index=False)

        import os
        p_mtime_before = os.path.getmtime(p_path)
        c_mtime_before = os.path.getmtime(c_path)

        generate_outputs(
            places_path=p_path,
            crowd_path=c_path,
            months_out=m_out,
            dates_out=d_out,
        )

        p_mtime_after = os.path.getmtime(p_path)
        c_mtime_after = os.path.getmtime(c_path)
        assert p_mtime_before == p_mtime_after, "places.csv was modified!"
        assert c_mtime_before == c_mtime_after, "crowd.csv was modified!"

    def test_output_not_overwriting_input(self, tmp_path, tiny_places, tiny_crowd):
        """Output paths must be different from input paths."""
        p_path = tmp_path / "places.csv"
        c_path = tmp_path / "crowd.csv"
        tiny_places.to_csv(p_path, index=False)
        tiny_crowd.to_csv(c_path, index=False)

        m_out = tmp_path / "sub" / "months_out.csv"
        d_out = tmp_path / "sub" / "dates_out.csv"

        generate_outputs(
            places_path=p_path,
            crowd_path=c_path,
            months_out=m_out,
            dates_out=d_out,
        )
        # Input files still valid
        reloaded = pd.read_csv(p_path)
        assert len(reloaded) == 2
