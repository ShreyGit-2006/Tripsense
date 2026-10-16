# TripSense

TripSense is an AI-powered Andhra Pradesh tourism assistant that helps travelers decide **when to go, what to visit, and how to choose accommodation**.

## Project Components

### 1. Hotel Review Intelligence
Analyze the project's hotel-review dataset with:
- Sentiment labels and scores (VADER)
- Keyword-based review topics: cleanliness, staff, location, food, service, amenities, value, and ambiance
- Per-hotel summaries and review counts
- Review-based strengths and pain points

Member 1's implementation is in `member1/review_intelligence.py`. It uses the datasets in `data/hotels/` and writes derived CSV outputs separately from the source data.

### 2. Hybrid Tourist-Place Recommendations
Personalized destination recommendations using:
- TF-IDF content-based filtering
- Collaborative filtering
- Hybrid scoring

### 3. Best Travel Dates
Calculate suitable travel dates using:
- Place comfort scores
- Indian holiday calendar
- Crowd/holiday penalty

### 4. Itinerary Assistant
Generate grounded travel itineraries using verified places from the TripSense dataset.

### 5. Streamlit Application
Planned app experiences include trip planning, itinerary and chat, maps, hotel review insights, and model explanations.

## Main Datasets
- `data/places/tripsense_places_v2.csv`: Andhra Pradesh tourist places
- `data/hotels/tripsense_hotels.csv`: hotel metadata
- `data/hotels/tripsense_hotel_reviews_dataset.csv`: hotel reviews
- `data/calendars/`: Indian calendar and travel-date/crowd features
- `data/interactions/`: tourism review/interaction data

## Run Member 1's hotel review pipeline

From the repository root:

```bash
pip install -r member1/requirements.txt
python member1/review_intelligence.py
```

Outputs:
- `data/hotels/review_insights.csv`
- `data/hotels/review_topic_summary.csv`

Sentiment labels are automated estimates and topic labels are keyword-based; neither is ground truth. This project scope does not include hotel booking demand forecasting or dynamic hotel pricing.
