# TripSense — Member 2 Integration Guide

## 1. Purpose

Member 2 provides a recommendation engine that ranks tourist
destinations based on user interests and, when available, rating history.

## 2. Main Components

- content_recommender.py: Content-based recommendations.
- collaborative_recommender.py: Recommendations based on user-rating patterns.
- hybrid_recommender.py: Combines content-based and collaborative scores.
- explanation.py: Generates explanations for recommendations.
- evaluation.py: Evaluates recommendations using Precision@K.
- recommendation_pipeline.py: Provides a unified recommendation interface.

## 3. Inputs

The recommendation pipeline accepts:

- interests: A text description of the user's travel interests.
- user_id: An existing reviewer_name, or None for a new user.
- top_n: Maximum number of recommendations to return.
- content_weight: Weight for content-based scores.
- collaborative_weight: Weight for collaborative scores.

The default hybrid weights are 70% content-based and 30% collaborative.

## 4. Recommendation Behavior

### New users

Users without available rating history receive content-based recommendations.

### Existing users

Users with rating history can receive hybrid recommendations.

## 5. Output

The pipeline returns a pandas DataFrame containing ranked destinations.

Depending on the selected recommendation method, available columns
may include:

- place_id
- name
- category
- district
- suitable_for
- visit_duration_hrs
- best_months
- content_score
- collaborative_score
- hybrid_score
- recommendation_method
- recommendation_reason

The returned columns depend on the underlying recommender.

## 6. Example Usage

```python
recommendations = pipeline.recommend_places(
    interests="temples, history, nature",
    user_id=None,
    top_n=20
)

print(recommendations)
