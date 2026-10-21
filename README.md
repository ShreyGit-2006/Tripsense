# TripSense — AI-Powered Andhra Pradesh Itinerary Planner

TripSense is an AI-powered tourism and hospitality platform that helps travelers decide **when to go, what to visit, and why**, generating personalized, day-wise travel itineraries for tourist destinations across Andhra Pradesh.

---

## 🚀 Quick Start & Local Execution

### 1. Prerequisites & Environment Setup
Python 3.10+ is recommended. Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/ShreyGit-2006/Tripsense.git
cd Tripsense

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Optional Credentials (Optional)
TripSense is designed to run **100% offline and locally without paid API keys**. All recommendations, itineraries, date quality scoring, weather analysis, and local RAG retrieval fallbacks work out of the box.

If you wish to enable conversational LLM answers for the RAG assistant:
```bash
cp .env.example .env
# Edit .env and supply either OPENAI_API_KEY or GOOGLE_API_KEY
```

### 3. Run the Web Application
Launch the interactive Streamlit app:

```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🌟 Key Features

1. **Personalized Trip Planner Form**:
   - Supports traveler types: Solo, Couple, Family, Friends, Group.
   - 1–7 day customizable durations.
   - Fixed dates (with 2026 calendar crowd/holiday scoring) or flexible month selection.
   - District-level filter covering all 26 districts of Andhra Pradesh, plus an all-Andhra-Pradesh option.
   - Comprehensive interest tags: Temples & Spirituality, Beaches, Nature & Forests, Waterfalls, Hill Stations, History & Heritage, Wildlife, Adventure & Trekking, Food, Culture, Photography.
   - Travel pace selection: Relaxed (2–3 places/day), Balanced (4–5 places/day), Packed (6–7 places/day).
   - User ID input for returning reviewers to unlock hybrid collaborative recommendations.

2. **Member 2 Recommendation Engine Integration**:
   - Seamlessly calls Member 2's `RecommendationPipeline` and `ContentBasedRecommender`.
   - Content-based recommendations for new travelers based on TF-IDF cosine similarity.
   - Item-based collaborative filtering and hybrid weighting (70% content, 30% collaborative) for existing reviewers.
   - Preserves genuine place metadata, recommendation scores, methods, and generated natural-language explanations without inventing data.

3. **Member 3 Best Travel Dates & Calendar Intelligence**:
   - Integrated with `member3/src/best_dates.py` and pre-computed seasonal/crowd tables (`best_months_results.csv`, `best_dates_results.csv`).
   - Daily travel quality scoring considering monthly comfort flags, crowd proxies, holiday bonuses, and long weekends.
   - Explicitly notes that crowd metrics are proxy estimates, not live visitor counts.

4. **Weather-Aware Planning**:
   - Integrates `data/weather/tripsense_weather_2023.csv` for historical monthly climate averages (temperature and rainfall).
   - Generates weather suitability warnings and flags outdoor comfort.
   - Clearly labels data as historical monthly averages rather than live forecasts.

5. **Hotel Review Intelligence**:
   - Integrated with `member3/src/review_intelligence.py` and pre-computed `review_insights.csv`.
   - Displays real hotels matched to destinations via `hotel_reviews.nearby_place_id` and district.
   - Shows VADER sentiment scores, star ratings, positive review percentages, and dominant feedback topics (cleanliness, staff, food, value, etc.) without fabricating distances or prices.

6. **Day-Wise Itinerary Scheduling Engine (`member3/src/itinerary.py`)**:
   - Allocates destinations across requested days (1–7 days).
   - Clusters nearby places using Haversine great-circle distance and nearest-neighbor traversal.
   - Respects place `visit_duration_hrs`, buffer intervals, and 1.5-hour meal/rest allocations.
   - Transparently estimates drive times (~40 km/h average) clearly labeled as approximations.
   - Handles missing coordinates and metadata gracefully.

7. **Interactive Map & AI Travel Assistant**:
   - Folium map with color-coded markers for each day's route.
   - Integrated RAG travel assistant (`member3/src/rag.py`) supporting questions on recommendations, weather alternatives, family suitability, and hotel cleanliness with local deterministic fallback.

---

## 🧪 Automated Testing

Run the full automated test suite covering all services, recommendation adapters, date scoring, weather, hotels, and itinerary generation:

```bash
# Run the itinerary planner test suite
python -m pytest tests/test_itinerary_planner.py -v

# Run Member 3's unit tests
python -m pytest member3/tests/ -v
```

---

## 📁 Project Architecture

```
Tripsense/
├── app.py                             # Main Streamlit web application
├── requirements.txt                   # Unified dependency manifest
├── .env.example                       # Template for optional LLM credentials
├── itinerary/                         # Itinerary service architecture
│   ├── __init__.py                    # Itinerary package marker & re-exports
│   └── services/
│       ├── __init__.py
│       ├── data_service.py            # Central dataset loading & validation
│       ├── recommendation_service.py  # Member 2 recommendation engine adapter
│       ├── date_service.py            # Member 3 best dates & month scoring
│       ├── weather_service.py         # Historical climate analysis service
│       ├── hotel_service.py           # Hotel matching & review insights service
│       └── rag_service.py             # Member 3 RAG Q&A assistant bridge
├── member2/
│   ├── src/
│   │   ├── content_recommender.py     # TF-IDF content-based recommender
│   │   ├── collaborative_recommender.py # Item-based collaborative recommender
│   │   ├── hybrid_recommender.py      # Content + collaborative scoring
│   │   ├── recommendation_pipeline.py # Unified pipeline with explanations
│   │   └── explanation.py             # Rule-based natural language explainer
├── member3/
│   ├── src/
│   │   ├── itinerary.py               # Core day-wise itinerary scheduling engine
│   │   ├── best_dates.py              # Calendar crowd & comfort date recommender
│   │   ├── review_intelligence.py     # Hotel review sentiment & topic model
│   │   └── rag.py                     # Retrieval-augmented Q&A assistant
│   └── tests/                         # Member 3 unit tests (127 tests)
├── data/
│   ├── places/tripsense_places_v2.csv # 444+ Andhra Pradesh tourist spots
│   ├── weather/tripsense_weather_2023.csv # Monthly climate statistics
│   ├── calendars/                     # Indian holidays, crowd proxies, scored dates
│   └── hotels/                        # Hotels metadata, reviews, and insights
└── tests/
    └── test_itinerary_planner.py      # Comprehensive integration test suite (41 tests)
```

*(Note: Member 1's demand forecasting code is explicitly excluded from the itinerary planner workflow).*
