"""
itinerary/services/rag_service.py
====================================
RAG travel assistant bridge for the TripSense itinerary planner.

Delegates to Member 3's rag.py module.  Pre-builds the TripSenseRetriever
index once from the loaded data and reuses it across queries (Streamlit
caching takes care of this at the app layer).

If Member 3's rag module is unavailable the service provides its own
minimal local fallback that answers from context dicts directly.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
_M3_SRC = _REPO_ROOT / "member3" / "src"
if str(_M3_SRC) not in sys.path:
    sys.path.insert(0, str(_M3_SRC))

try:
    from rag import (  # type: ignore
        TripSenseRetriever,
        load_all_data as _rag_load_all,
        build_context,
        generate_answer,
        answer_query as _rag_answer_query,
    )
    _RAG_AVAILABLE = True
except ImportError as _e:
    log.warning("rag module unavailable: %s", _e)
    _RAG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_retriever(places_df: pd.DataFrame) -> Optional[object]:
    """
    Build a TripSenseRetriever from a pre-loaded places DataFrame.
    Returns None if the rag module is unavailable.
    """
    if not _RAG_AVAILABLE or places_df.empty:
        return None
    try:
        return TripSenseRetriever(places_df)
    except Exception as exc:
        log.warning("Failed to build TripSenseRetriever: %s", exc)
        return None


def answer_question(
    query: str,
    data: dict[str, pd.DataFrame],
    retriever: Optional[object] = None,
    top_k_places: int = 5,
) -> dict:
    """
    Answer a travel question using Member 3's RAG pipeline.

    Parameters
    ----------
    query       : Natural-language travel question
    data        : Dict with keys "places", "hotels", "review_insights", "best_months"
    retriever   : Pre-built TripSenseRetriever (built from data if None)
    top_k_places: Max places to retrieve for context

    Returns
    -------
    dict: {
        answer:       str,
        source:       str ("openai" | "google" | "local_fallback"),
        can_answer:   bool,
        evidence_ids: dict,
    }
    """
    if not query or not query.strip():
        return {
            "answer":       "Please enter a travel question.",
            "source":       "none",
            "can_answer":   False,
            "evidence_ids": {},
        }

    if _RAG_AVAILABLE:
        try:
            # Build rag-compatible data dict
            rag_data = {
                "places":          data.get("places", pd.DataFrame()),
                "hotels":          data.get("hotels", pd.DataFrame()),
                "review_insights": data.get("review_insights", pd.DataFrame()),
                "best_months":     data.get("best_months", pd.DataFrame()),
            }
            if retriever is None:
                retriever = TripSenseRetriever(rag_data["places"])
            ctx = build_context(query, rag_data, retriever, top_k_places=top_k_places)
            return generate_answer(ctx)
        except Exception as exc:
            log.warning("RAG answer_query failed: %s", exc)

    # Minimal fallback
    return {
        "answer": (
            f"I couldn't retrieve a detailed answer for: '{query}'.\n"
            "The RAG module is unavailable. "
            "Please check that all dependencies (scikit-learn, etc.) are installed."
        ),
        "source":       "local_fallback",
        "can_answer":   False,
        "evidence_ids": {},
    }
