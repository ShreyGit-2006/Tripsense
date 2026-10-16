# Member 1 — Hotel Review Intelligence

Member 1 now focuses on review intelligence for TripSense's Andhra Pradesh hotel data. Hotel demand forecasting and dynamic pricing are out of scope.

## Inputs
- `data/hotels/tripsense_hotel_reviews_dataset.csv`: review text, hotel identifier, and optional rating.
- `data/hotels/tripsense_hotels.csv`: hotel metadata joined using `hotel_id`.

## Run from repository root
```bash
pip install -r member1/requirements.txt
python member1/review_intelligence.py
```

Optional custom paths:
```bash
python member1/review_intelligence.py --reviews path/to/reviews.csv --hotels path/to/hotels.csv --out-dir path/to/output
```

## Outputs
- `data/hotels/review_insights.csv`: cleaned review rows with VADER sentiment score/label and keyword topic flags for cleanliness, staff, location, food, service, amenities, value, and ambiance.
- `data/hotels/review_topic_summary.csv`: per-hotel review count, mean sentiment, sentiment percentages, mean rating when available, and topic mention percentages.

## Notes and limitations
- Sentiment is a model estimate, not ground truth or a substitute for reading reviews.
- Topic detection uses keywords and can miss paraphrases, sarcasm, and context.
- The script does not invent hotel ratings, booking demand, occupancy, or prices.
- Hotel-to-place distance matching requires reliable hotel and place coordinates; this script does not claim proximity based only on district/name.
- Keep source datasets unchanged; outputs are written separately.
