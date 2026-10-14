import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeRecommender:
    """
    Item-based collaborative filtering recommender.

    Data requirements:
        reviewer_name -> user
        place_id      -> item
        rating        -> user preference
    """

    def __init__(self, interactions_df):
        self.interactions = interactions_df.copy()

        required_columns = [
            "reviewer_name",
            "place_id",
            "rating",
        ]

        missing = [
            col
            for col in required_columns
            if col not in self.interactions.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        # User-place rating matrix
        self.user_item_matrix = self.interactions.pivot_table(
            index="reviewer_name",
            columns="place_id",
            values="rating",
            aggfunc="mean",
            fill_value=0,
        )

        # Item-based similarity
        item_matrix = self.user_item_matrix.T

        similarity_matrix = cosine_similarity(item_matrix)

        self.item_similarity = pd.DataFrame(
            similarity_matrix,
            index=item_matrix.index,
            columns=item_matrix.index,
        )

    def recommend_for_user(self, user_id, top_n=10):
        """
        Recommend places for an existing user.
        """

        if user_id not in self.user_item_matrix.index:
            raise ValueError(
                f"User '{user_id}' was not found."
            )

        user_ratings = self.user_item_matrix.loc[user_id]

        # Places already rated by the user
        rated_places = user_ratings[
            user_ratings > 0
        ].index.tolist()

        scores = {}

        for rated_place in rated_places:

            rating = user_ratings[rated_place]

            similar_places = (
                self.item_similarity[rated_place]
                .sort_values(ascending=False)
            )

            for place_id, similarity in similar_places.items():

                # Do not recommend places already rated
                if place_id in rated_places:
                    continue

                # Ignore zero similarity
                if similarity <= 0:
                    continue

                scores[place_id] = (
                    scores.get(place_id, 0)
                    + similarity * rating
                )

        if not scores:
            return pd.DataFrame(
                columns=[
                    "place_id",
                    "collaborative_score",
                ]
            )

        recommendations = (
            pd.Series(
                scores,
                name="collaborative_score"
            )
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        recommendations.columns = [
            "place_id",
            "collaborative_score",
        ]

        # Normalize score to 0–1
        max_score = recommendations[
            "collaborative_score"
        ].max()

        if max_score > 0:
            recommendations[
                "collaborative_score"
            ] = (
                recommendations[
                    "collaborative_score"
                ] / max_score
            )

        return recommendations
