import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeRecommender:
    """
    Item-based collaborative filtering recommender.

    Uses:
        reviewer_name -> user
        place_id      -> item
        rating        -> preference
    """

    def __init__(self, interactions_df):
        self.interactions = interactions_df.copy()

        # Create user-item rating matrix
        self.user_item_matrix = self.interactions.pivot_table(
            index="reviewer_name",
            columns="place_id",
            values="rating",
            aggfunc="mean",
            fill_value=0,
        )

        # Item-based similarity:
        # transpose so rows represent places
        item_matrix = self.user_item_matrix.T

        self.item_similarity = cosine_similarity(item_matrix)

        self.item_similarity_df = pd.DataFrame(
            self.item_similarity,
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

        # Compare places the user liked with other places
        for rated_place in rated_places:

            rating = user_ratings[rated_place]

            similar_places = (
                self.item_similarity_df[rated_place]
                .sort_values(ascending=False)
            )

            for place, similarity in similar_places.items():

                # Don't recommend something already rated
                if place in rated_places:
                    continue

                scores[place] = (
                    scores.get(place, 0)
                    + similarity * rating
                )

        recommendations = (
            pd.Series(scores, name="collaborative_score")
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        recommendations.columns = [
            "place_id",
            "collaborative_score",
        ]

        return recommendations


if __name__ == "__main__":

    reviews = pd.read_csv(
        "../data/interactions/tripsense_reviews_dataset.csv"
    )

    recommender = CollaborativeRecommender(reviews)

    # Test with the first available user
    test_user = reviews["reviewer_name"].iloc[0]

    print(f"\nRecommendations for: {test_user}")

    print(
        recommender.recommend_for_user(
            test_user,
            top_n=10,
        )
    )
