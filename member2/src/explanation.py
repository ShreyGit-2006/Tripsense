from pathlib import Path
import re

import pandas as pd


class RecommendationExplainer:
    """
    Generates place-specific explanations for
    TripSense recommendations.
    """

    def __init__(self, places_df):
        self.places = places_df.copy()

        text_columns = [
            "tags_text",
            "description",
            "category",
            "suitable_for",
            "district",
            "best_months",
        ]

        for column in text_columns:
            if column in self.places.columns:
                self.places[column] = (
                    self.places[column]
                    .fillna("")
                    .astype(str)
                )

    def _get_interest_matches(self, interests, place_row):
        """
        Find meaningful words from the user's interests
        that occur in the place information.
        """

        if not interests:
            return []

        words = re.findall(
            r"[a-zA-Z]+",
            str(interests).lower()
        )

        stop_words = {
            "i",
            "want",
            "to",
            "visit",
            "go",
            "see",
            "a",
            "an",
            "the",
            "and",
            "or",
            "with",
            "good",
            "best",
            "places",
            "place",
            "looking",
            "for",
            "like",
            "would",
        }

        words = [
            word
            for word in words
            if word not in stop_words
            and len(word) > 2
        ]

        place_text = " ".join(
            str(place_row.get(column, ""))
            for column in [
                "tags_text",
                "description",
                "category",
                "suitable_for",
            ]
        ).lower()

        matches = []

        for word in words:
            if word in place_text:
                matches.append(word)

        return list(dict.fromkeys(matches))

    def explain(
        self,
        place_row,
        interests=None,
        content_score=0,
        collaborative_score=0,
    ):
        """
        Generate a concise, place-specific explanation.
        """

        name = str(
            place_row.get("name", "This place")
        ).strip()

        category = str(
            place_row.get("category", "")
        ).strip()

        district = str(
            place_row.get("district", "")
        ).strip()

        suitable_for = str(
            place_row.get("suitable_for", "")
        ).strip()

        visit_duration = place_row.get(
            "visit_duration_hrs"
        )

        best_months = str(
            place_row.get("best_months", "")
        ).strip()

        matches = self._get_interest_matches(
            interests,
            place_row
        )

        if suitable_for.lower() in [
            "nan",
            "none",
            ""
        ]:
            suitable_for = ""

        if best_months.lower() in [
            "nan",
            "none",
            ""
        ]:
            best_months = ""

        reasons = []

        # Content-based reason
        if matches:
            readable_matches = ", ".join(
                matches[:3]
            )

            reasons.append(
                f"matches your interest in "
                f"{readable_matches}"
            )

        elif content_score >= 0.05:
            reasons.append(
                "is a strong match for your search"
            )

        # Place category
        if category:
            reasons.append(
                f"offers a {category} experience"
            )

        # Collaborative reason
        if collaborative_score >= 0.70:
            reasons.append(
                "is highly supported by preferences "
                "from similar travelers"
            )

        elif collaborative_score >= 0.40:
            reasons.append(
                "also aligns with preferences "
                "from similar travelers"
            )

        # Build explanation
        if not reasons:
            explanation = (
                f"{name} is recommended based on "
                "your overall preferences."
            )

        elif len(reasons) == 1:
            explanation = (
                f"{name} is recommended because "
                f"it {reasons[0]}."
            )

        else:
            explanation = (
                f"{name} is recommended because "
                f"it {reasons[0]} and {reasons[1]}."
            )

        # Location and suitability
        details = []

        if district:
            details.append(
                f"located in {district}"
            )

        if suitable_for:
            details.append(
                f"suitable for {suitable_for}"
            )

        # Visit duration
        if (
            pd.notna(visit_duration)
            and str(visit_duration).strip() != ""
        ):
            try:
                duration = float(visit_duration)

                if duration > 0:
                    details.append(
                        f"about {duration:g} hours to visit"
                    )

            except (ValueError, TypeError):
                pass

        if details:
            explanation += (
                " It is "
                + ", ".join(details)
                + "."
            )

        # Best months
        if best_months:
            explanation += (
                f" Best months: {best_months}."
            )

        return explanation


def add_explanations(
    recommendations,
    places_df,
    interests=None,
):
    """
    Add recommendation_reason to a recommendation DataFrame.
    """

    explainer = RecommendationExplainer(
        places_df
    )

    results = recommendations.copy()

    explanations = []

    for _, row in results.iterrows():

        place_id = row["place_id"]

        matching_places = places_df[
            places_df["place_id"] == place_id
        ]

        if matching_places.empty:
            explanations.append(
                "Recommended based on your preferences."
            )
            continue

        place_row = matching_places.iloc[0]

        explanation = explainer.explain(
            place_row=place_row,
            interests=interests,
            content_score=row.get(
                "content_score",
                0
            ),
            collaborative_score=row.get(
                "collaborative_score",
                0
            ),
        )

        explanations.append(explanation)

    results["recommendation_reason"] = explanations

    return results


# ============================================================
# TEST CODE
# ============================================================

if __name__ == "__main__":

    # Project paths
    BASE_DIR = Path(__file__).resolve().parents[1]

    PLACES_FILE = (
        BASE_DIR
        / "data"
        / "places"
        / "tripsense_places_v2.csv"
    )

    INTERACTIONS_FILE = (
        BASE_DIR
        / "data"
        / "interactions"
        / "tripsense_reviews_dataset.csv"
    )

    # Load datasets
    places = pd.read_csv(PLACES_FILE)
    interactions = pd.read_csv(
        INTERACTIONS_FILE
    )

    print(
        f"Places loaded: {len(places)}"
    )

    print(
        f"Interactions loaded: {len(interactions)}"
    )

    # Import recommendation models
    from content_recommender import (
        ContentBasedRecommender
    )

    from collaborative_recommender import (
        CollaborativeRecommender
    )

    from hybrid_recommender import (
        HybridRecommender
    )

    # Create recommendation models
    content_model = ContentBasedRecommender(
        places
    )

    collaborative_model = (
        CollaborativeRecommender(
            interactions
        )
    )

    hybrid_model = HybridRecommender(
        places_df=places,
        content_recommender=content_model,
        collaborative_recommender=collaborative_model,
    )

    # Select a test user
    test_user = interactions[
        "reviewer_name"
    ].iloc[0]

    # Traveler's interests
    interests = (
        "museums history art culture"
    )

    # Generate hybrid recommendations
    hybrid_results = hybrid_model.recommend(
        user_id=test_user,
        interests=interests,
        top_n=10,
        content_weight=0.7,
        collaborative_weight=0.3,
    )

    # Add explanations
    explained_results = add_explanations(
        recommendations=hybrid_results,
        places_df=places,
        interests=interests,
    )

    # Display results
    print(
        "\nRecommendations with explanations:\n"
    )

    print(
        explained_results[
            [
                "place_id",
                "name",
                "category",
                "district",
                "hybrid_score",
                "recommendation_reason",
            ]
        ].to_string(index=False)
    )
