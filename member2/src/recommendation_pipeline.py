import pandas as pd

from content_recommender import ContentBasedRecommender
from collaborative_recommender import CollaborativeRecommender
from hybrid_recommender import HybridRecommender
from explanation import RecommendationExplainer


class RecommendationPipeline:
    """
    Unified interface for TripSense destination recommendations.

    New users receive content-based recommendations.
    Existing users with rating history can receive hybrid
    recommendations.
    """

    def __init__(self, places_df, interactions_df):
        self.places = places_df.copy()
        self.interactions = interactions_df.copy()

        self.content_model = ContentBasedRecommender(self.places)

        self.collaborative_model = CollaborativeRecommender(
            self.interactions
        )

        self.hybrid_model = HybridRecommender(
            places_df=self.places,
            content_recommender=self.content_model,
            collaborative_recommender=self.collaborative_model,
        )

        self.explainer = RecommendationExplainer()

    def recommend_places(
        self,
        interests,
        user_id=None,
        top_n=10,
        content_weight=0.7,
        collaborative_weight=0.3,
    ):
        """
        Return ranked tourist destinations.

        Parameters
        ----------
        interests : str
            User's interests, such as 'temples, history, nature'.
        user_id : str or None
            Existing reviewer_name, if available.
        top_n : int
            Maximum number of destinations to return.
        content_weight : float
            Weight assigned to content-based scores.
        collaborative_weight : float
            Weight assigned to collaborative scores.
        """

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        if not isinstance(interests, str) or not interests.strip():
            raise ValueError("Please provide some travel interests.")

        if abs(content_weight + collaborative_weight - 1.0) > 0.001:
            raise ValueError("Recommendation weights must sum to 1.")

        # Use hybrid recommendations only when the user has history.
        user_has_history = (
            user_id is not None
            and user_id in self.collaborative_model.user_item_matrix.index
            and (
                self.collaborative_model.user_item_matrix.loc[user_id] > 0
            ).any()
        )

        if user_has_history:
            recommendations = self.hybrid_model.recommend(
                user_id=user_id,
                interests=interests,
                top_n=top_n,
                content_weight=content_weight,
                collaborative_weight=collaborative_weight,
            )

            recommendations["recommendation_method"] = "hybrid"

        else:
            recommendations = self.content_model.recommend(
                interests=interests,
                top_n=top_n,
            )

            recommendations["recommendation_method"] = "content-based"

        # Add explanations when the explainer supports this interface.
        if not recommendations.empty:
            if hasattr(self.explainer, "add_explanations"):
                recommendations = self.explainer.add_explanations(
                    recommendations,
                    interests,
                )

        return recommendations.reset_index(drop=True)
