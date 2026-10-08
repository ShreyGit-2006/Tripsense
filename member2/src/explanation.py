import re
import pandas as pd


class RecommendationExplainer:
    """
    Generates simple, user-friendly explanations
    for TripSense recommendations.
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

    def _find_interest_matches(self, interests, place_row):
        """
        Find important words from the user's interests
        that also appear in the place information.
        """

        if not interests:
            return []

        # Convert user interests into simple words
        interest_words = re.findall(
            r"[a-zA-Z]+",
            str(interests).lower()
        )

        # Ignore very common words
        stop_words = {
            "i", "want", "to", "visit", "go", "see",
            "a", "an", "the", "and", "or", "with",
            "good", "best", "places", "place"
        }

        interest_words = [
            word
            for word in interest_words
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
                "district",
            ]
        ).lower()

        matches = [
            word
            for word in interest_words
            if word in place_text
        ]

        # Remove duplicates while preserving order
        return list(dict.fromkeys(matches))

    def explain(
        self,
        place_row,
        interests=None,
        content_score=None,
        collaborative_score=None,
    ):
        """
        Generate a simple explanation for one recommendation.
        """

        place_name = place_row.get(
            "name",
            "this place"
        )

        reasons = []

        # ----------------------------------------
        # Content-based reason
        # ----------------------------------------
        matches = self._find_interest_matches(
            interests,
            place_row
        )

        if matches:
            readable_matches = ", ".join(matches[:3])

            reasons.append(
                f"matches your interest in {readable_matches}"
            )

        elif (
            content_score is not None
            and content_score >= 0.05
        ):
            reasons.append(
                "matches the type of experience "
                "you are looking for"
            )

        # ----------------------------------------
        # Collaborative reason
        # ----------------------------------------
        if (
            collaborative_score is not None
            and collaborative_score >= 0.5
        ):
            reasons.append(
                "is supported by preferences "
                "from similar travelers"
            )

        # ----------------------------------------
        # Place characteristics
        # ----------------------------------------
        category = str(
            place_row.get("category", "")
        ).strip()

        district = str(
            place_row.get("district", "")
        ).strip()

        if category:
            reasons.append(
                f"offers a {category} experience"
            )

        # ----------------------------------------
        # Build final explanation
        # ----------------------------------------
        if not reasons:
            return (
                f"{place_name} is recommended "
                "based on your preferences and "
                "TripSense's recommendation model."
            )

        if len(reasons) == 1:
            reason_text = reasons[0]

        elif len(reasons) == 2:
            reason_text = (
                f"{reasons[0]} and {reasons[1]}"
            )

        else:
            reason_text = (
                f"{reasons[0]}, "
                f"{reasons[1]}, and {reasons[2]}"
            )

        if district:
            return (
                f"{place_name} is recommended because it "
                f"{reason_text}. It is located in {district}."
            )

        return (
            f"{place_name} is recommended because it "
            f"{reason_text}."
        )


def add_explanations(
    recommendations,
    places_df,
    interests=None,
):
    """
    Add a recommendation_reason column to
    hybrid recommendation results.
    """

    explainer = RecommendationExplainer(
        places_df
    )

    results = recommendations.copy()

    reasons = []

    for _, row in results.iterrows():

        place_id = row["place_id"]

        matching_places = places_df[
            places_df["place_id"] == place_id
        ]

        if matching_places.empty:
            reasons.append(
                "Recommended based on your preferences."
            )
            continue

        place_row = matching_places.iloc[0]

        reason = explainer.explain(
            place_row=place_row,
            interests=interests,
            content_score=row.get(
                "content_score"
            ),
            collaborative_score=row.get(
                "collaborative_score"
            ),
        )

        reasons.append(reason)

    results["recommendation_reason"] = reasons

    return results
