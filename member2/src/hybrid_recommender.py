import pandas as pd


class HybridRecommender:
    """
    Combines content-based and collaborative
    recommendation scores.
    """

    def __init__(
        self,
        places_df,
        content_recommender,
        collaborative_recommender,
    ):
        self.places = places_df.copy()
        self.content_recommender = content_recommender
        self.collaborative_recommender = (
            collaborative_recommender
        )

    def recommend(
        self,
        user_id,
        interests,
        top_n=10,
        content_weight=0.7,
        collaborative_weight=0.3,
    ):
        """
        Generate hybrid recommendations.

        Default:
            70% content
            30% collaborative
        """

        if abs(
            content_weight
            + collaborative_weight
            - 1.0
        ) > 0.001:
            raise ValueError(
                "content_weight + "
                "collaborative_weight must equal 1.0"
            )

        # ----------------------------------------
        # Content-based recommendations
        # ----------------------------------------
        content_results = (
            self.content_recommender.recommend(
                interests=interests,
                top_n=len(self.places),
            )
        )

        content_scores = content_results[
            ["place_id", "content_score"]
        ].copy()

        # ----------------------------------------
        # Collaborative recommendations
        # ----------------------------------------
        collaborative_results = (
            self.collaborative_recommender
            .recommend_for_user(
                user_id=user_id,
                top_n=len(self.places),
            )
        )

        collaborative_scores = collaborative_results[
            [
                "place_id",
                "collaborative_score",
            ]
        ].copy()

        # ----------------------------------------
        # Merge scores
        # ----------------------------------------
        result = self.places.copy()

        result = result.merge(
            content_scores,
            on="place_id",
            how="left",
        )

        result = result.merge(
            collaborative_scores,
            on="place_id",
            how="left",
        )

        result["content_score"] = (
            result["content_score"]
            .fillna(0)
        )

        result["collaborative_score"] = (
            result["collaborative_score"]
            .fillna(0)
        )

        # ----------------------------------------
        # Hybrid score
        # ----------------------------------------
        result["hybrid_score"] = (
            content_weight
            * result["content_score"]
            +
            collaborative_weight
            * result["collaborative_score"]
        )

        # ----------------------------------------
        # Remove places already rated
        # ----------------------------------------
        user_item_matrix = (
            self.collaborative_recommender
            .user_item_matrix
        )

        if user_id in user_item_matrix.index:

            rated_places = user_item_matrix.loc[
                user_id
            ]

            rated_places = rated_places[
                rated_places > 0
            ].index

            result = result[
                ~result["place_id"].isin(
                    rated_places
                )
            ]

        # ----------------------------------------
        # Sort
        # ----------------------------------------
        result = result.sort_values(
            "hybrid_score",
            ascending=False,
        )

        # ----------------------------------------
        # Useful output columns
        # ----------------------------------------
        output_columns = [
            "place_id",
            "name",
            "category",
            "district",
            "suitable_for",
            "best_months",
            "content_score",
            "collaborative_score",
            "hybrid_score",
        ]

        output_columns = [
            col
            for col in output_columns
            if col in result.columns
        ]

        return result[
            output_columns
        ].head(top_n).reset_index(drop=True)
