import numpy as np
import pandas as pd


def build_user_interests(user_id, train_data, places_df):
    """
    Build a user preference profile from positively rated
    destinations in the training dataset.
    """

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

            if pd.notna(value) and str(value).strip():
                text_parts.append(str(value))

    return " ".join(text_parts)


def precision_at_k(recommended_ids, relevant_ids, k):
    """
    Calculate Precision@K for one user's ranked recommendations.

    Precision@K = relevant recommendations in the top K / K.

    The denominator remains K, even if fewer than K recommendations
    are returned.
    """

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer.")

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

    Parameters
    ----------
    recommendation_function : callable
        Function accepting (user_id, interests, top_n).
        It must return a DataFrame ranked best first and containing
        a place_id column.

    train_data : pandas.DataFrame
        Training interactions used to construct user profiles.

    test_data : pandas.DataFrame
        Held-out positive interactions.

    places_df : pandas.DataFrame
        Tourist destination metadata.

    k_values : tuple
        Values of K for which to calculate Precision@K.

    Returns
    -------
    dict
        Number of evaluated users and Precision@K values.
    """

    if not k_values:
        raise ValueError("k_values must contain at least one value.")

    for k in k_values:
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("Every k value must be a positive integer.")

    required_train_columns = {
        "reviewer_name",
        "place_id",
        "rating",
    }

    required_test_columns = {
        "reviewer_name",
        "place_id",
        "rating",
    }

    missing_train = required_train_columns - set(train_data.columns)
    missing_test = required_test_columns - set(test_data.columns)

    if missing_train:
        raise ValueError(
            f"Training data is missing columns: {sorted(missing_train)}"
        )

    if missing_test:
        raise ValueError(
            f"Test data is missing columns: {sorted(missing_test)}"
        )

    results = {k: [] for k in k_values}
    evaluated_users = 0

    # Only positive held-out interactions count as relevant.
    positive_test = test_data[
        test_data["rating"] >= 4
    ]

    for user_id, user_test in positive_test.groupby("reviewer_name"):

        interests = build_user_interests(
            user_id,
            train_data,
            places_df,
        )

        if not interests.strip():
            continue

        relevant_ids = set(
            user_test["place_id"].astype(str)
        )

        recommendations = recommendation_function(
            user_id,
            interests,
            len(places_df),
        )

        if recommendations is None:
            recommendations = pd.DataFrame(columns=["place_id"])

        if not isinstance(recommendations, pd.DataFrame):
            raise ValueError(
                "The recommendation function must return a DataFrame."
            )

        if "place_id" not in recommendations.columns:
            raise ValueError(
                "Recommendations must contain a place_id column."
            )

        # Exclude destinations already seen in the training data.
        seen_ids = set(
            train_data.loc[
                train_data["reviewer_name"] == user_id,
                "place_id",
            ].astype(str)
        )

        eligible = recommendations[
            ~recommendations["place_id"].astype(str).isin(seen_ids)
        ]

        # Remove duplicate destinations while preserving ranking order.
        eligible = eligible.drop_duplicates(
            subset=["place_id"],
            keep="first",
        )

        recommended_ids = eligible["place_id"].astype(str).tolist()

        evaluated_users += 1

        for k in k_values:
            score = precision_at_k(
                recommended_ids,
                relevant_ids,
                k,
            )
            results[k].append(score)

    output = {
        "Users evaluated": evaluated_users,
    }

    for k in k_values:
        output[f"Precision@{k}"] = (
            float(np.mean(results[k]))
            if results[k]
            else 0.0
        )

    return output
