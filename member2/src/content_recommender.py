from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Content-based tourist-place recommender.

    Uses the following place information:
    - tags_text
    - description
    - category
    - suitable_for
    - district
    """

    def __init__(self, places_df):
        self.places = places_df.copy()

        # Replace missing values so TF-IDF never receives NaN values.
        text_columns = [
            "tags_text",
            "description",
            "category",
            "suitable_for",
            "district",
        ]

        for column in text_columns:
            if column in self.places.columns:
                self.places[column] = (
                    self.places[column]
                    .fillna("")
                    .astype(str)
                )

        # Combine useful textual information.
        self.places["combined_text"] = (
            self.places["tags_text"] + " "
            + self.places["description"] + " "
            + self.places["category"] + " "
            + self.places["suitable_for"] + " "
            + self.places["district"]
        )

        # Convert place descriptions/tags into TF-IDF vectors.
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.places["combined_text"]
        )

        # Calculate similarity between every pair of places.
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        # Map place_id -> dataframe row index.
        self.place_index = {
            str(place_id): index
            for index, place_id in enumerate(self.places["place_id"])
        }

    def recommend(
        self,
        place_id=None,
        interests=None,
        district=None,
        suitable_for=None,
        top_n=10,
    ):
        """
        Return recommended tourist places.

        Parameters
        ----------
        place_id : str, optional
            Existing place used as the recommendation starting point.

        interests : str, optional
            Traveler interests, for example:
            "nature waterfalls trekking"

        district : str, optional
            Optional district filter.

        suitable_for : str, optional
            Optional suitability filter, for example:
            "family"

        top_n : int
            Number of recommendations to return.
        """

        candidates = self.places.copy()

        # ---------------------------------------------------------
        # 1. Filter by district
        # ---------------------------------------------------------
        if district:
            candidates = candidates[
                candidates["district"]
                .str.contains(
                    str(district),
                    case=False,
                    na=False,
                )
            ]

        # ---------------------------------------------------------
        # 2. Filter by suitable_for
        # ---------------------------------------------------------
        if suitable_for:
            candidates = candidates[
                candidates["suitable_for"]
                .str.contains(
                    str(suitable_for),
                    case=False,
                    na=False,
                )
            ]

        # ---------------------------------------------------------
        # 3. Calculate similarity score
        # ---------------------------------------------------------
        if interests:
            # Convert traveler interests into the same TF-IDF space.
            interest_vector = self.vectorizer.transform(
                [str(interests)]
            )

            scores = cosine_similarity(
                interest_vector,
                self.tfidf_matrix,
            ).flatten()

        elif place_id is not None:
            place_key = str(place_id)

            if place_key not in self.place_index:
                raise ValueError(
                    f"place_id '{place_id}' was not found."
                )

            source_index = self.place_index[place_key]

            scores = self.similarity_matrix[source_index]

        else:
            raise ValueError(
                "Provide either 'interests' or 'place_id'."
            )

        # Add scores to a copy of the original dataframe.
        result = candidates.copy()

        # Match scores using the original dataframe indices.
        result["content_score"] = result.index.map(
            lambda index: float(scores[index])
        )

        # Remove the source place when recommending from a place_id.
        if place_id is not None:
            result = result[
                result["place_id"].astype(str)
                != str(place_id)
            ]

        # Highest similarity first.
        result = result.sort_values(
            "content_score",
            ascending=False,
        )

        # Return useful columns.
        output_columns = [
            "place_id",
            "name",
            "category",
            "district",
            "suitable_for",
            "visit_duration_hrs",
            "best_months",
            "content_score",
        ]

        output_columns = [
            column
            for column in output_columns
            if column in result.columns
        ]

        return result[output_columns].head(top_n).reset_index(
            drop=True
        )


def load_places(csv_path=None):
    """Load the TripSense tourist-place dataset."""

    if csv_path is None:
        base_dir = Path(__file__).resolve().parents[1]
        csv_path = (
            base_dir
            / "data"
            / "places"
            / "tripsense_places_v2.csv"
        )

    return pd.read_csv(csv_path)


if __name__ == "__main__":
    places = load_places()

    recommender = ContentBasedRecommender(places)

    print("\nRecommendations for interests:")
    print(
        recommender.recommend(
            interests="nature waterfalls trekking",
            top_n=10,
        )
    )
