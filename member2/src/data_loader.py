from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

PLACES_FILE = BASE_DIR / "data" / "places" / "tripsense_places_v2.csv"
INTERACTIONS_FILE = (
    BASE_DIR / "data" / "interactions" / "tripsense_reviews_dataset.csv"
)


def load_places():
    """Load the tourist places dataset."""
    return pd.read_csv(PLACES_FILE)


def load_interactions():
    """Load the user-place interaction/review dataset."""
    return pd.read_csv(INTERACTIONS_FILE)


def load_data():
    """Load both datasets."""
    places = load_places()
    interactions = load_interactions()

    return places, interactions


if __name__ == "__main__":
    places, interactions = load_data()

    print(f"Places loaded: {len(places)}")
    print(f"Interactions loaded: {len(interactions)}")
