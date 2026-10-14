"""
member3/src/rag.py
==================
RAG (Retrieval-Augmented Generation) Travel Assistant for TripSense.

Architecture
------------
retrieval  ->  context_builder  ->  answer_generator
   |                                      |
TF-IDF over places + review_insights      optional LLM (env-configured)
+ best_dates for date context             deterministic local fallback when no LLM

Data sources
------------
  data/places/tripsense_places_v2.csv           – tourist places
  data/hotels/tripsense_hotels.csv              – hotel metadata
  data/hotels/review_insights.csv               – per-hotel sentiment/topic aggregates
  data/calendars/best_months_results.csv        – pre-computed monthly scores

LLM configuration (all optional)
---------------------------------
  TRIPSENSE_LLM_PROVIDER  : "openai" | "google" (default: none = local fallback)
  OPENAI_API_KEY          : for OpenAI provider
  OPENAI_MODEL            : model name (default "gpt-4o-mini")
  GOOGLE_API_KEY          : for Google Generative AI provider
  GOOGLE_MODEL            : model name (default "gemini-1.5-flash")

Responsibilities
----------------
- Retrieve relevant tourist places, hotels, review insights, and best-date data.
- Reuse Member 2's ContentBasedRecommender when available; do not duplicate its logic.
- Keep retrieval, context construction, and answer generation in separate functions.
- Include supporting record IDs in all responses.
- Never fabricate prices, ratings, opening hours, or other missing facts.
- State clearly when the datasets cannot answer a question.
- Distinguish dataset facts, calculated scores, predictions, and generated suggestions.
- Provide a deterministic local fallback without any API key.

Limitations
-----------
- TF-IDF retrieval is keyword-frequency based; semantic nuance may be missed.
- The local fallback assembles evidence-based answers without an LLM; phrasing is
  structured rather than conversational.
- LLM-generated answers inherit the imprecision of the LLM and should not be taken
  as authoritative without checking the cited evidence records.
- Hotel review sentiment (VADER) is approximate; see review_intelligence.py.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
try:
    _HERE      = Path(__file__).resolve()
    _REPO_ROOT = _HERE.parents[2]
    _M2_SRC    = _REPO_ROOT / "member2" / "src"
    PLACES_CSV         = _REPO_ROOT / "data" / "places" / "tripsense_places_v2.csv"
    HOTELS_CSV         = _REPO_ROOT / "data" / "hotels" / "tripsense_hotels.csv"
    REVIEW_INSIGHTS_CSV = _REPO_ROOT / "data" / "hotels" / "review_insights.csv"
    BEST_MONTHS_CSV    = _REPO_ROOT / "data" / "calendars" / "best_months_results.csv"
except NameError:
    # Colab fallback
    _REPO_ROOT = Path("/content")
    _M2_SRC    = Path("/content")
    PLACES_CSV         = Path("/content/tripsense_places_v2.csv")
    HOTELS_CSV         = Path("/content/tripsense_hotels.csv")
    REVIEW_INSIGHTS_CSV = Path("/content/review_insights.csv")
    BEST_MONTHS_CSV    = Path("/content/best_months_results.csv")

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_all_data(
    places_path: Optional[Path] = None,
    hotels_path: Optional[Path] = None,
    insights_path: Optional[Path] = None,
    best_months_path: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load all TripSense datasets into a single dict.

    Returns
    -------
    {
      "places"       : DataFrame,
      "hotels"       : DataFrame,
      "review_insights" : DataFrame,
      "best_months"  : DataFrame,
    }
    """
    return {
        "places":          _read_csv_safe(places_path or PLACES_CSV),
        "hotels":          _read_csv_safe(hotels_path or HOTELS_CSV),
        "review_insights": _read_csv_safe(insights_path or REVIEW_INSIGHTS_CSV),
        "best_months":     _read_csv_safe(best_months_path or BEST_MONTHS_CSV),
    }


# ---------------------------------------------------------------------------
# Retrieval index
# ---------------------------------------------------------------------------

class TripSenseRetriever:
    """
    TF-IDF based retriever over tourist places.

    Member 2's ContentBasedRecommender uses TF-IDF on the same places dataset.
    We build an independent index here to avoid modifying Member 2's code and to
    support natural-language queries from non-user-ID contexts (Q&A vs. personalised
    recommendation).  For personalised recommendations, call Member 2 directly.

    The retriever indexes the combined text of each place and finds top-k matches
    for a free-text query.
    """

    def __init__(self, places_df: pd.DataFrame):
        self._empty = places_df.empty
        if self._empty:
            return

        self._df = places_df.copy()

        text_cols = [
            "name", "description", "category", "tags_text",
            "suitable_for", "district", "best_months",
        ]
        combined = self._df.apply(
            lambda row: " ".join(
                str(row[c]) for c in text_cols if c in self._df.columns and pd.notna(row[c])
            ),
            axis=1,
        )

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=8000,
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform(combined)

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """
        Retrieve top-k places matching the query.

        Parameters
        ----------
        query : free-text question or search phrase
        top_k : maximum number of results

        Returns
        -------
        DataFrame with columns from places + retrieval_score
        """
        if self._empty or not query.strip():
            return pd.DataFrame()

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()

        top_idx = sims.argsort()[::-1][:top_k]
        result = self._df.iloc[top_idx].copy()
        result["retrieval_score"] = sims[top_idx].round(4)
        result = result[result["retrieval_score"] > 0].reset_index(drop=True)
        return result


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(
    query: str,
    data: dict[str, pd.DataFrame],
    retriever: TripSenseRetriever,
    top_k_places: int = 5,
    top_k_hotels: int = 3,
) -> dict:
    """
    Build a structured evidence context for a query.

    Retrieval strategy:
      1. TF-IDF place retrieval for the query.
      2. Hotel insights for districts matching retrieved places.
      3. Best-month data for retrieved place_ids.

    Returns
    -------
    dict with:
      query             : original query string
      places            : list of place record dicts (with evidence IDs)
      hotels            : list of hotel record dicts
      review_insights   : list of hotel insight dicts
      best_months       : list of (place_id, best month rows)
      evidence_ids      : all place_ids and hotel_ids included
      dataset_note      : reminder of data source nature
    """
    context: dict = {
        "query":           query,
        "places":          [],
        "hotels":          [],
        "review_insights": [],
        "best_months":     [],
        "evidence_ids":    {"place_ids": [], "hotel_ids": []},
        "dataset_note": (
            "Data sourced from TripSense project datasets (2026). "
            "Comfort and crowd scores are calculated estimates, not visitor counts. "
            "Sentiment scores are VADER approximations. "
            "Do not present computed scores as ground truth."
        ),
    }

    # 1. Place retrieval
    place_results = retriever.retrieve(query, top_k=top_k_places)
    if not place_results.empty:
        keep_cols = [
            "place_id", "name", "category", "district",
            "description", "best_months", "suitable_for",
            "rating", "review_count", "retrieval_score",
        ]
        keep_cols = [c for c in keep_cols if c in place_results.columns]
        for _, row in place_results[keep_cols].iterrows():
            context["places"].append(row.to_dict())
            pid = row.get("place_id")
            if pid:
                context["evidence_ids"]["place_ids"].append(pid)

    retrieved_districts = list(
        {r.get("district", "") for r in context["places"] if r.get("district")}
    )

    # 2. Hotels in matching districts
    hotels_df = data.get("hotels", pd.DataFrame())
    if not hotels_df.empty and retrieved_districts:
        hotel_mask = hotels_df.get("district", pd.Series(dtype=str)).apply(
            lambda d: any(rd.lower() in str(d).lower() for rd in retrieved_districts)
        )
        matched_hotels = hotels_df[hotel_mask].head(top_k_hotels)
        for _, row in matched_hotels.iterrows():
            hid = row.get("hotel_id")
            context["hotels"].append(row.to_dict())
            if hid:
                context["evidence_ids"]["hotel_ids"].append(hid)

    # 3. Review insights for matched hotels
    ri_df = data.get("review_insights", pd.DataFrame())
    if not ri_df.empty and context["evidence_ids"]["hotel_ids"]:
        ri_matched = ri_df[
            ri_df["hotel_id"].isin(context["evidence_ids"]["hotel_ids"])
        ]
        for _, row in ri_matched.iterrows():
            context["review_insights"].append(row.to_dict())

    # 4. Best months for retrieved places
    bm_df = data.get("best_months", pd.DataFrame())
    if not bm_df.empty and context["evidence_ids"]["place_ids"]:
        bm_matched = bm_df[
            bm_df["place_id"].isin(context["evidence_ids"]["place_ids"])
        ]
        # Keep only rank 1-3 months to keep context compact
        if "rank_within_place" in bm_matched.columns:
            bm_matched = bm_matched[bm_matched["rank_within_place"] <= 3]
        bm_cols = [c for c in [
            "place_id", "name", "month_name", "rank_within_place",
            "composite_score", "reasons",
        ] if c in bm_matched.columns]
        for _, row in bm_matched[bm_cols].iterrows():
            context["best_months"].append(row.to_dict())

    return context


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

def _format_local_answer(context: dict) -> dict:
    """
    Deterministic local fallback: compose a structured answer from retrieved evidence.
    No LLM or API key required.

    Returns
    -------
    dict with keys: answer (str), evidence_ids (dict), source (str), context_used (dict)
    """
    query   = context["query"]
    places  = context["places"]
    hotels  = context["hotels"]
    bm      = context["best_months"]
    ri      = context["review_insights"]

    if not places and not hotels:
        return {
            "answer": (
                f"The TripSense datasets do not contain enough information to answer: "
                f"'{query}'. Try a different destination or more specific query."
            ),
            "evidence_ids": context["evidence_ids"],
            "source": "local_fallback",
            "context_used": context,
            "can_answer": False,
        }

    lines = [f"Query: {query}", ""]

    if places:
        lines.append("=== Relevant Tourist Places ===")
        for p in places:
            pid   = p.get("place_id", "?")
            name  = p.get("name", "?")
            cat   = p.get("category", "")
            dist  = p.get("district", "")
            bm_p  = p.get("best_months", "")
            score = p.get("retrieval_score", 0)
            rating = p.get("rating", "")
            rating_str = f"  Rating: {rating}" if pd.notna(rating) and rating != "" else ""
            lines.append(
                f"- [{pid}] {name} ({cat}) | {dist}"
                + (f" | Best months: {bm_p}" if bm_p else "")
                + (f" | Retrieval match: {score:.2f}" if score else "")
                + rating_str
            )
            desc = p.get("description", "")
            if desc and len(str(desc)) > 10:
                lines.append(f"  {str(desc)[:200]}")
        lines.append("")

    if bm:
        lines.append("=== Best Travel Months (calculated comfort + crowd scores) ===")
        for row in bm:
            pid  = row.get("place_id", "")
            name = row.get("name", "")
            mn   = row.get("month_name", "")
            sc   = row.get("composite_score", "")
            rs   = row.get("reasons", "")
            lines.append(f"- [{pid}] {name}: {mn}  (score: {sc}) — {rs}")
        lines.append("")

    if hotels:
        lines.append("=== Nearby Hotels ===")
        for h in hotels:
            hid   = h.get("hotel_id", "?")
            hname = h.get("hotel_name", "?")
            tier  = h.get("tier", "")
            price = h.get("price_range", "")
            lines.append(f"- [{hid}] {hname} ({tier}) — {price}")
        lines.append("")

    if ri:
        lines.append("=== Hotel Review Insights (VADER sentiment — approximate) ===")
        for r in ri:
            hid      = r.get("hotel_id", "?")
            cnt      = r.get("review_count", "?")
            mean_r   = r.get("mean_star_rating", "")
            mean_v   = r.get("mean_vader_compound", "")
            dominant = r.get("dominant_topic", "")
            rating_str = f"  Avg star rating: {mean_r:.2f}" if pd.notna(mean_r) and mean_r != "" else ""
            vader_str  = f"  Avg VADER compound: {mean_v:.3f}" if pd.notna(mean_v) and mean_v != "" else ""
            lines.append(
                f"- [{hid}] {cnt} reviews"
                + rating_str + vader_str
                + (f"  Top topic: {dominant}" if dominant else "")
            )
        lines.append("")

    lines.append(f"[{context['dataset_note']}]")

    return {
        "answer":        "\n".join(lines),
        "evidence_ids":  context["evidence_ids"],
        "source":        "local_fallback",
        "context_used":  context,
        "can_answer":    True,
    }


def _call_llm(context: dict, provider: str) -> Optional[str]:
    """
    Call an LLM to generate a conversational answer from the structured context.
    Returns None on any error (caller will fall back to local).

    Provider behaviour:
      "openai"  : uses openai library + OPENAI_API_KEY
      "google"  : uses google.generativeai + GOOGLE_API_KEY
    """
    system_prompt = (
        "You are a helpful travel assistant for TripSense, a tourism platform for India. "
        "Answer the user's travel question using ONLY the provided dataset evidence. "
        "Do NOT invent prices, ratings, opening hours, or other facts not in the evidence. "
        "Clearly state when the data is insufficient. "
        "Distinguish between dataset facts, calculated comfort/crowd scores, and VADER "
        "sentiment estimates (which are approximate). "
        "Include place_id and hotel_id identifiers when referring to specific records."
    )

    # Compact evidence text
    evidence_lines = [
        f"Query: {context['query']}",
        f"Places: {[p.get('place_id','?') + ' ' + p.get('name','') for p in context['places']]}",
    ]
    if context["best_months"]:
        evidence_lines.append(
            "Best months: "
            + "; ".join(
                f"{r.get('place_id','')} {r.get('month_name','')} (score {r.get('composite_score','')})"
                for r in context["best_months"]
            )
        )
    if context["hotels"]:
        evidence_lines.append(
            "Hotels: "
            + "; ".join(
                f"{h.get('hotel_id','?')} {h.get('hotel_name','')} {h.get('tier','')}"
                for h in context["hotels"]
            )
        )
    if context["review_insights"]:
        evidence_lines.append(
            "Review insights: "
            + "; ".join(
                f"{r.get('hotel_id','?')} reviews={r.get('review_count','')} "
                f"vader={r.get('mean_vader_compound','')}"
                for r in context["review_insights"]
            )
        )
    evidence_lines.append(context["dataset_note"])

    user_message = (
        "\n".join(evidence_lines)
        + "\n\nPlease answer the travel query in a helpful, concise paragraph."
    )

    try:
        if provider == "openai":
            import openai  # type: ignore
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return None
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            return resp.choices[0].message.content

        elif provider == "google":
            import google.generativeai as genai  # type: ignore
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not api_key:
                return None
            model_name = os.environ.get("GOOGLE_MODEL", "gemini-1.5-flash")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name=model_name)
            resp = model.generate_content(
                [system_prompt, user_message],
                generation_config={"temperature": 0.3, "max_output_tokens": 500},
            )
            return resp.text

    except Exception as exc:  # noqa: BLE001
        # Do not raise; fall back to local answer
        warnings.warn(f"LLM call failed ({provider}): {exc}", stacklevel=2)
        return None

    return None


def generate_answer(context: dict) -> dict:
    """
    Generate an answer using the configured LLM provider, or the local fallback.

    LLM provider is read from env var TRIPSENSE_LLM_PROVIDER.
    If not set (or LLM call fails), deterministic local fallback is used.

    Returns
    -------
    dict with:
      answer       : string answer
      evidence_ids : {"place_ids": [...], "hotel_ids": [...]}
      source       : "openai" | "google" | "local_fallback"
      can_answer   : bool (False if datasets can't address the query)
      context_used : the raw context dict
    """
    provider = os.environ.get("TRIPSENSE_LLM_PROVIDER", "").lower().strip()

    llm_answer = None
    if provider in ("openai", "google"):
        llm_answer = _call_llm(context, provider)

    if llm_answer is not None:
        return {
            "answer":       llm_answer,
            "evidence_ids": context["evidence_ids"],
            "source":       provider,
            "can_answer":   True,
            "context_used": context,
        }

    # Deterministic local fallback
    return _format_local_answer(context)


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def answer_query(
    query: str,
    data: Optional[dict[str, pd.DataFrame]] = None,
    retriever: Optional[TripSenseRetriever] = None,
    top_k_places: int = 5,
    top_k_hotels: int = 3,
    places_path: Optional[Path] = None,
    hotels_path: Optional[Path] = None,
    insights_path: Optional[Path] = None,
    best_months_path: Optional[Path] = None,
) -> dict:
    """
    End-to-end RAG pipeline: retrieve -> build context -> generate answer.

    Parameters
    ----------
    query            : Natural-language travel question
    data             : Pre-loaded data dict (useful in tests / batch mode)
    retriever        : Pre-built TripSenseRetriever (optional, built from data if None)
    top_k_places     : Max places to retrieve
    top_k_hotels     : Max hotels to include per query
    *_path           : Override default CSV paths

    Returns
    -------
    dict with answer, evidence_ids, source, can_answer, context_used
    """
    if not query or not query.strip():
        return {
            "answer":       "Please provide a travel question.",
            "evidence_ids": {"place_ids": [], "hotel_ids": []},
            "source":       "local_fallback",
            "can_answer":   False,
            "context_used": {},
        }

    if data is None:
        data = load_all_data(places_path, hotels_path, insights_path, best_months_path)

    if retriever is None:
        retriever = TripSenseRetriever(data.get("places", pd.DataFrame()))

    context = build_context(query, data, retriever, top_k_places, top_k_hotels)
    return generate_answer(context)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TripSense RAG travel assistant.")
    parser.add_argument("query", nargs="?", default="",
                        help="Travel question to answer")
    parser.add_argument("--top-k", type=int, default=5)
    args, _ = parser.parse_known_args()

    q = args.query or input("Enter your travel question: ").strip()
    if not q:
        print("No query provided.")
        sys.exit(1)

    result = answer_query(q, top_k_places=args.top_k)
    print(f"\n[Source: {result['source']}  |  can_answer={result['can_answer']}]")
    print(f"Evidence place_ids : {result['evidence_ids'].get('place_ids', [])}")
    print(f"Evidence hotel_ids : {result['evidence_ids'].get('hotel_ids', [])}")
    print("\n--- Answer ---")
    print(result["answer"])
