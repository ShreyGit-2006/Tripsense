import numpy as np
import pandas as pd


def build_user_interests(user_id, train_data, places_df):
    """Build a preference profile from positively rated training places."""

    user_history = train_data[
        (train_data["reviewer_name"] == user_id)
        & (train_data["rating"] >= 4)
    ]

    if user_history.empty:
        return ""

    liked_place_ids = user_history["place_id"].tolist()

    liked_places = places_df[
        places_df["place_id"].isin(liked_place_ids)
    ]

    text_parts = []

    for _, row in liked_places.iterrows():
        for column in [
            "tags_text",
            "description",
            "category",
            "suitable_for",
        ]:
            value = row.get(column, "")
            if pd.notna(value):
                text_parts.append(str(value))

    return " ".join(text_parts)


def precision_at_k(recommended_ids, relevant_ids, k):
    """Calculate Precision@K for one user's ranked recommendations."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    top_k = list(recommended_ids[:k])
    relevant = set(map(str, relevant_ids))

    hits = len(set(map(str, top_k)) & relevant)

    return hits / k


def evaluate_recommender(
    recommendation_function,
    train_data,
    test_data,
    places_df,
    k_values=(5, 10),
):
    """
    Evaluate a recommender using held-out positive interactions.

    recommendation_function must accept:
        (user_id, interests, top_n)

    It must return a DataFrame containing place_id, ranked best first.
    """

    results = {k: [] for k in k_values}
    evaluated_users = 0

    for user_id, user_test in test_data.groupby("reviewer_name"):
        interests = build_user_interests(
            user_id, train_data, places_df
        )

        if not interests.strip():
            continue

        relevant_ids = set(user_test["place_id"].astype(str))

        recommendations = recommendation_function(
            user_id, interests, len(places_df)
        )

        if recommendations is None or recommendations.empty:
            recommended_ids = []
        else:
            if "place_id" not in recommendations.columns:
                raise ValueError(
                    "Recommendations must contain a place_id column."
                )

            seen_ids = set(
                train_data.loc[
                    train_data["reviewer_name"] == user_id,
                    "place_id",
                ].astype(str)
            )

            eligible = recommendations[
                ~recommendations["place_id"].astype(str).isin(seen_ids)
            ]

            recommended_ids = eligible["place_id"].astype(str).tolist()

        evaluated_users += 1

        for k in k_values:
            results[k].append(
                precision_at_k(recommended_ids, relevant_ids, k)
            )

    output = {
        "Users evaluated": evaluated_users,
    }

    for k in k_values:
        output[f"Precision@{k}"] = (
            float(np.mean(results[k])) if results[k] else 0.0
        )

    return output
