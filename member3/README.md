# Member 3 — TripSense

Member 3 contributes three interoperable components: best travel date scoring, hotel review intelligence, and a RAG travel assistant.

## Structure

```
member3/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── best_dates.py          # Stage 1: Best Travel Dates
│   ├── review_intelligence.py # Stage 2: Hotel Review Intelligence
│   └── rag.py                 # Stage 3: RAG Travel Assistant
└── tests/
    ├── test_best_dates.py
    ├── test_review_intelligence.py
    └── test_rag.py
```

## Setup

All commands should be run from the **repository root** (`Tripsense/`).

```bash
pip install -r member3/requirements.txt
```

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `scipy`, `vaderSentiment`, `textblob`, `pytest`

No API keys are required for the core pipeline or tests.

---

## Stage 1 — Best Travel Dates (`best_dates.py`)

### Input Datasets

| File | Key Columns Used |
|---|---|
| `data/places/tripsense_places_v2.csv` | `place_id, name, district, category, comfort_jan…comfort_dec` |
| `data/calendars/holiday_crowd.csv` | `date, crowd_proxy_normalized, base_visit_score, is_bridge_day, is_holiday, is_long_weekend_candidate` |

### Scoring Formula

**Monthly score** (configurable weights, default sum = 1.0):

```
composite = 0.40 × comfort_flag
          + 0.30 × (1 − mean_crowd_proxy_normalized)
          + 0.20 × mean_base_visit_score
          + 0.10 × bridge_day_fraction
```

**Daily score** (default weights):

```
composite = 0.35 × comfort_flag
          + 0.30 × (1 − crowd_proxy_normalized)
          + 0.20 × base_visit_score
          + 0.10 × bridge_bonus
          + 0.05 × is_holiday
```

Both scores are then min-max normalized per place across months/days.

- `comfort_flag` = 1 if the month is weather-comfortable for that place (from `comfort_<month>` columns)
- `crowd_proxy_normalized` ∈ [0, 1] — higher means more crowded (lower score = less crowded = better for travel)
- `base_visit_score` — baseline leisure availability proxy from the calendar
- `bridge_day_fraction` — fraction of days in month that are bridge/long-weekend candidates

### Outputs

| File | Description |
|---|---|
| `data/calendars/best_months_results.csv` | One row per (place × month), 5,328 rows total |
| `data/calendars/best_dates_results.csv` | One row per (place × date), 162,060 rows total |

### CLI Usage

```bash
# Score all places and write output files
python member3/src/best_dates.py --generate-all

# Best months for a specific destination
python member3/src/best_dates.py --destination "Vageti Kona Falls" --top-n-months 3

# Best dates in a date range
python member3/src/best_dates.py --destination "Arch Falls" --date-from 2026-01-01 --date-to 2026-06-30
```

### Python API

```python
from member3.src.best_dates import get_best_months, get_best_dates, generate_outputs

months = get_best_months(destination="Vageti Kona Falls", top_n=3)
dates  = get_best_dates(destination="Vageti Kona Falls", date_from="2026-01-01", date_to="2026-03-31")
m_df, d_df = generate_outputs()   # writes both CSVs
```

### Tests

```bash
python -m pytest member3/tests/test_best_dates.py -v   # 36 tests
```

### Limitations

- Calendar data covers 2026 only. Scores for other years are not available.
- `comfort_flag` is binary (0/1); no continuous weather metric is available.
- Crowd proxy scores are computed estimates, **not** actual visitor counts.

---

## Stage 2 — Hotel Review Intelligence (`review_intelligence.py`)

### Input Datasets

| File | Key Columns |
|---|---|
| `data/hotels/tripsense_hotel_reviews_dataset.csv` | `review_id, hotel_id, hotel_name, district, sentiment, rating, review_text, stay_date` |
| `data/hotels/tripsense_hotels.csv` | `hotel_id, hotel_name, district, tier, star_rating, price_range` |

### Pipeline

1. **Load & validate** — dedup on `review_id`, drop blank `review_text`, drop missing `hotel_id`, coerce `rating` to numeric
2. **VADER sentiment** — compound score in [-1, 1], labelled positive (≥ 0.05) / neutral / negative (≤ -0.05)
3. **Keyword topic detection** — 8 topics (cleanliness, staff, location, food, service, amenities, value, ambiance) via keyword matching
4. **Aggregate by hotel** — mean star rating, mean VADER compound, sentiment label percentages, topic mention rates, dominant topic
5. **Topic summary** — cross-hotel topic frequencies and mean sentiment per topic

### Outputs

| File | Description |
|---|---|
| `data/hotels/review_insights.csv` | One row per hotel (705 rows). Columns: `hotel_id, review_count, mean_star_rating, mean_vader_compound, pct_positive, pct_negative, pct_neutral, rate_<topic>, dominant_topic` |
| `data/hotels/review_topic_summary.csv` | One row per topic (8 rows). Columns: `topic, mention_count, mention_rate, mean_vader_compound_when_mentioned` |

### CLI Usage

```bash
# Run full pipeline
python member3/src/review_intelligence.py --run-full

# Query a specific hotel
python member3/src/review_intelligence.py --hotel-id HTL0001
```

### Python API

```python
from member3.src.review_intelligence import run_pipeline, get_hotel_insights

insights, topic_summary = run_pipeline()
hotel_df = get_hotel_insights("HTL0001")
```

### Tests

```bash
python -m pytest member3/tests/test_review_intelligence.py -v   # 51 tests
```

### Limitations

- **VADER is approximate.** It is a lexicon-based rule model calibrated for social media text; it may misclassify sarcasm, mixed opinions, or domain-specific language. Predictions are **not** ground truth.
- Topic detection is keyword-based; paraphrased mentions may not be captured.
- No transformer model is used (intentional — keeps the system offline and dependency-light).
- The `sentiment` column already present in the source CSV is the dataset provider's label; VADER adds an independent second signal (`vader_label`). The two are kept separate.

---

## Stage 3 — RAG Travel Assistant (`rag.py`)

### Architecture

```
answer_query(query)
    ├── load_all_data()              ← places, hotels, review_insights, best_months
    ├── TripSenseRetriever.retrieve() ← TF-IDF cosine similarity on places
    ├── build_context()              ← evidence dict with place + hotel + review + date records
    └── generate_answer()
            ├── _call_llm()          ← optional (OpenAI / Google Generative AI)
            └── _format_local_answer() ← deterministic fallback (no API key needed)
```

### Data Sources

- `data/places/tripsense_places_v2.csv` — primary retrieval corpus
- `data/hotels/tripsense_hotels.csv` — hotel metadata
- `data/hotels/review_insights.csv` — Stage 2 aggregated insights
- `data/calendars/best_months_results.csv` — Stage 1 scored months

### LLM Configuration (Optional)

Set environment variables to enable LLM-based conversational answers. If unset, the deterministic local fallback is used automatically.

| Variable | Values | Description |
|---|---|---|
| `TRIPSENSE_LLM_PROVIDER` | `openai` or `google` | Enable LLM provider |
| `OPENAI_API_KEY` | Your key | Required if provider = `openai` |
| `OPENAI_MODEL` | e.g. `gpt-4o-mini` | Default: `gpt-4o-mini` |
| `GOOGLE_API_KEY` | Your key | Required if provider = `google` |
| `GOOGLE_MODEL` | e.g. `gemini-1.5-flash` | Default: `gemini-1.5-flash` |

**API keys are never hardcoded.** They are read-only from environment variables at runtime.

### CLI Usage

```bash
# Local fallback (no API key needed)
python member3/src/rag.py "What are the best waterfalls to visit in winter?"

# With OpenAI
TRIPSENSE_LLM_PROVIDER=openai OPENAI_API_KEY=sk-... python member3/src/rag.py "Beaches in Goa"
```

### Python API

```python
from member3.src.rag import answer_query, TripSenseRetriever, load_all_data, build_context

result = answer_query("Best temples to visit in Tamil Nadu")
print(result["answer"])
print("Evidence:", result["evidence_ids"])

# Batch mode (pre-load data + retriever once)
data      = load_all_data()
retriever = TripSenseRetriever(data["places"])
for q in queries:
    r = answer_query(q, data=data, retriever=retriever)
```

### Answer Response Structure

```python
{
    "answer":       str,          # human-readable answer
    "evidence_ids": {
        "place_ids":  [...],      # place_id values cited
        "hotel_ids":  [...],      # hotel_id values cited
    },
    "source":      "local_fallback" | "openai" | "google",
    "can_answer":  bool,          # False = datasets insufficient
    "context_used": {...},        # raw evidence dict for inspection
}
```

### Tests

```bash
python -m pytest member3/tests/test_rag.py -v   # 40 tests (no API key required)
```

### Limitations

- TF-IDF retrieval is keyword-frequency based; it does not capture semantic meaning or intent beyond shared vocabulary.
- The local fallback is structured and transparent but not conversational.
- LLM answers inherit LLM imprecision; always check cited `evidence_ids`.
- The assistant explicitly refuses to fabricate prices, ratings, opening hours, or other facts not in the datasets.
- Member 2's `HybridRecommender` requires a `user_id` (personalised). The RAG module uses its own TF-IDF retriever for non-personalised Q&A and documents Member 2's API for integration.

---

## Running All Tests

```bash
# From repository root
python -m pytest member3/tests/ -v
# Expected: 127 passed
```

## Compatibility with Other Members

| Member | Interface | Notes |
|---|---|---|
| Member 1 | `demand_forecasting.py` — standalone script, no shared API | No conflicts; Member 1 uses `hotel_bookings.csv` (deleted from repo, pre-existing state) |
| Member 2 | `ContentBasedRecommender(places_df).recommend(interests=str, top_n=int)` | Member 3's RAG uses its own TF-IDF for Q&A; calls Member 2's recommender via `member2.src.content_recommender` when a personalised recommendation is needed |

## Data Integrity Guarantees

- Source CSVs in `data/` are **never modified** by any Member 3 script.
- Output files are written to separate paths (`data/calendars/best_*.csv`, `data/hotels/review_*.csv`).
- All test suites verify source-file modification timestamps are unchanged after pipeline runs.
