import pandas as pd

from content_recommender import ContentBasedRecommender
from collaborative_recommender import CollaborativeRecommender
from hybrid_recommender import HybridRecommender
from explanation import RecommendationExplainer


class RecommendationPipeline:
    """
    Unified interface for TripSense destination recommendations.

    New users receive content-based recommendations.
    Existing users with rating history receive hybrid recommendations.
    Each recommendation includes a place-specific explanation.
    """

    def __init__(self, places_df, interactions_df):
        self.places = places_df.copy()
        self.interactions = interactions_df.copy()

        required_place_columns = {"place_id", "name"}
        required_interaction_columns = {
            "reviewer_name",
            "place_id",
            "rating",
        }

        missing_places = required_place_columns - set(self.places.columns)
        missing_interactions = (
            required_interaction_columns
            - set(self.interactions.columns)
        )

        if missing_places:
            raise ValueError(
                f"Places data is missing columns: {sorted(missing_places)}"
            )

        if missing_interactions:
            raise ValueError(
                "Interactions data is missing columns: "
                f"{sorted(missing_interactions)}"
            )

        self.interactions["rating"] = pd.to_numeric(
            self.interactions["rating"],
            errors="coerce",
        )

        self.interactions = self.interactions.dropna(
            subset=["reviewer_name", "place_id", "rating"]
        ).copy()

        self.content_model = ContentBasedRecommender(
            self.places
        )

        self.collaborative_model = CollaborativeRecommender(
            self.interactions
        )

        self.hybrid_model = HybridRecommender(
            places_df=self.places,
            content_recommender=self.content_model,
            collaborative_recommender=self.collaborative_model,
        )

        self.explainer = RecommendationExplainer(
            self.places
        )

    def recommend_places(
        self,
        interests,
        user_id=None,
        top_n=10,
        content_weight=0.7,
        collaborative_weight=0.3,
    ):
        """
        Return ranked destinations with explanations.

        Parameters
        ----------
        interests : str
            User interests, e.g. 'temples, history, nature'.

        user_id : str or None
            Existing reviewer_name, or None for a new user.

        top_n : int
            Maximum number of destinations to return.

        content_weight : float
            Weight assigned to content-based recommendations.

        collaborative_weight : float
            Weight assigned to collaborative recommendations.

        Returns
        -------
        pandas.DataFrame
            Ranked destinations with available metadata,
            recommendation scores, method, and explanation.
        """

        if (
            not isinstance(top_n, int)
            or isinstance(top_n, bool)
            or top_n < 1
        ):
            raise ValueError(
                "top_n must be a positive integer."
            )

        if not isinstance(interests, str) or not interests.strip():
            raise ValueError(
                "Please provide some travel interests."
            )

        if (
            not isinstance(content_weight, (int, float))
            or not isinstance(collaborative_weight, (int, float))
        ):
            raise ValueError("Recommendation weights must be numeric.")

        if (
            content_weight < 0
            or collaborative_weight < 0
            or abs(
                content_weight + collaborative_weight - 1.0
            ) > 0.001
        ):
            raise ValueError(
                "Weights must be non-negative and sum to 1."
            )

        # Check whether the user exists and has rating history.
        user_has_history = False

        if user_id is not None:
            user_matrix = (
                self.collaborative_model.user_item_matrix
            )

            if user_id in user_matrix.index:
                user_has_history = bool(
                    (user_matrix.loc[user_id] > 0).any()
                )

        # Generate recommendations.
        if user_has_history:
            recommendations = self.hybrid_model.recommend(
                user_id=user_id,
                interests=interests,
                top_n=top_n,
                content_weight=content_weight,
                collaborative_weight=collaborative_weight,
            )

            method = "hybrid"

        else:
            recommendations = self.content_model.recommend(
                interests=interests,
                top_n=top_n,
            )

            method = "content-based"

        recommendations = recommendations.copy()

        if recommendations.empty:
            recommendations["recommendation_method"] = pd.Series(
                dtype="object"
            )
            recommendations["recommendation_reason"] = pd.Series(
                dtype="object"
            )

            return recommendations.reset_index(drop=True)

        # Add method metadata.
        recommendations["recommendation_method"] = method

        # Generate one explanation per destination.
        reasons = []

        for _, row in recommendations.iterrows():

            content_score = pd.to_numeric(
                row.get("content_score", 0),
                errors="coerce",
            )

            collaborative_score = pd.to_numeric(
                row.get("collaborative_score", 0),
                errors="coerce",
            )

            content_score = (
                0.0
                if pd.isna(content_score)
                else float(content_score)
            )

            collaborative_score = (
                0.0
                if pd.isna(collaborative_score)
                else float(collaborative_score)
            )

            reason = self.explainer.explain(
                place_row=row,
                interests=interests,
                content_score=content_score,
                collaborative_score=collaborative_score,
            )

            reasons.append(reason)

        recommendations["recommendation_reason"] = reasons

        return recommendations.reset_index(drop=True)
