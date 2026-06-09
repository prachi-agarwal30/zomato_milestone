"""Phase 3: Prompt Engineering & Groq API Orchestration.

This module connects already-filtered restaurant candidates with the Groq LLM to
produce ranked recommendations and personalized, human-like justifications.

It is intentionally scoped to Phase 3 only: it expects the candidate list to be
provided by the caller (e.g. the Phase 2 filtering layer) and focuses on prompt
construction, the Groq API call, and parsing the structured JSON response.

Example:
    from recommender_engine import get_recommendations

    candidates = [
        {"name": "Toscano", "location": "Bangalore", "cuisine": "Italian",
         "cost_for_two": 2200, "rating": 4.5},
    ]
    prefs = {"location": "Bangalore", "cuisine": "Italian", "budget": "Medium",
             "min_rating": 4.0, "additional": "rooftop seating"}
    result = get_recommendations(candidates, prefs)
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import find_dotenv, load_dotenv
from groq import Groq

# Load .env robustly: first search up from the current working directory, then
# fall back to a .env sitting next to this module, so the key is found
# regardless of which directory the app is launched from.
load_dotenv(find_dotenv(usecwd=True))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DEFAULT_MODEL = "llama-3.3-70b-versatile"

# The LLM persona and output contract. The model is asked to rank the supplied
# candidates and explain why each fits the user's preferences, returning a
# strict JSON object so the response is trivial to parse downstream.
SYSTEM_PROMPT = (
    "You are an Expert Gastronomy Guide, a knowledgeable and friendly food "
    "critic. You are given a list of candidate restaurants (already filtered to "
    "match the user's hard constraints) and the user's preferences. Your job is "
    "to rank the best matches and explain, in a warm and human-like tone, why "
    "each restaurant fits the user's stated and optional preferences "
    "(e.g. 'great for a family dinner', 'good rooftop ambience', 'quick bite').\n"
    "\n"
    "Rules:\n"
    "- Only recommend restaurants from the provided candidate list. Never invent "
    "restaurants or facts.\n"
    "- Order recommendations from best to worst match.\n"
    "- Keep each reason concise (1-2 sentences) and specific to the user's "
    "preferences.\n"
    "- Respond with ONLY a valid JSON object, no markdown, using this schema:\n"
    '{"recommendations": [{"name": str, "cuisine": str, "rating": number, '
    '"cost_for_two": number, "reason": str}]}'
)


class RecommendationError(RuntimeError):
    """Raised when the LLM call or response parsing fails."""


def _get_client(api_key: str | None = None) -> Groq:
    """Build a Groq client, sourcing the API key from the environment if needed."""
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RecommendationError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment."
        )
    return Groq(api_key=key)


def _format_preferences(preferences: dict[str, Any]) -> str:
    """Render the user preferences into a compact, readable block."""
    labels = {
        "location": "Location",
        "cuisine": "Cuisine",
        "budget": "Budget",
        "min_rating": "Minimum rating",
        "additional": "Additional preferences",
    }
    lines = []
    for key, label in labels.items():
        value = preferences.get(key)
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        lines.append(f"- {label}: {value}")
    return "\n".join(lines) if lines else "- (no specific preferences provided)"


def _build_user_prompt(
    candidates: list[dict[str, Any]], preferences: dict[str, Any]
) -> str:
    """Assemble the user-turn prompt with preferences and candidate context."""
    candidates_json = json.dumps(candidates, indent=2, ensure_ascii=False)
    return (
        "User preferences:\n"
        f"{_format_preferences(preferences)}\n\n"
        "Candidate restaurants (JSON):\n"
        f"{candidates_json}\n\n"
        "Rank the candidates that best match the user's preferences and explain "
        "why each one fits. Return only the JSON object described in the system "
        "instructions."
    )


def get_recommendations(
    candidates: list[dict[str, Any]],
    preferences: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.4,
    api_key: str | None = None,
    client: Groq | None = None,
) -> list[dict[str, Any]]:
    """Rank candidate restaurants with Groq and return structured recommendations.

    Args:
        candidates: Filtered restaurant records (each a dict with at least
            ``name``; ideally also ``cuisine``, ``rating``, ``cost_for_two``).
        preferences: User preferences (``location``, ``cuisine``, ``budget``,
            ``min_rating``, ``additional``). Missing keys are ignored.
        model: Groq model id.
        temperature: Sampling temperature.
        api_key: Optional explicit API key (defaults to ``GROQ_API_KEY``).
        client: Optional pre-built Groq client (mainly for testing).

    Returns:
        A list of recommendation dicts, each with ``name``, ``cuisine``,
        ``rating``, ``cost_for_two`` and ``reason``.

    Raises:
        RecommendationError: If there are no candidates, the API call fails, or
            the response cannot be parsed as the expected JSON shape.
    """
    if not candidates:
        raise RecommendationError("No candidate restaurants were provided.")

    groq_client = client or _get_client(api_key)

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(candidates, preferences)},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001 - surface any SDK/network error uniformly
        raise RecommendationError(f"Groq API call failed: {exc}") from exc

    content = response.choices[0].message.content or ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RecommendationError(
            f"Could not parse Groq response as JSON: {exc}\nRaw response: {content!r}"
        ) from exc

    recommendations = parsed.get("recommendations")
    if not isinstance(recommendations, list):
        raise RecommendationError(
            f"Response JSON missing a 'recommendations' list. Got: {parsed!r}"
        )
    return recommendations


if __name__ == "__main__":
    # Manual smoke test against the live Groq API.
    sample_candidates = [
        {"name": "Toscano", "location": "Bangalore", "cuisine": "Italian",
         "cost_for_two": 2200, "rating": 4.5},
        {"name": "Little Italy", "location": "Bangalore", "cuisine": "Italian",
         "cost_for_two": 1600, "rating": 4.1},
        {"name": "Spice Junction", "location": "Bangalore", "cuisine": "Indian",
         "cost_for_two": 900, "rating": 4.3},
    ]
    sample_prefs = {
        "location": "Bangalore",
        "cuisine": "Italian",
        "budget": "Medium",
        "min_rating": 4.0,
        "additional": "good for a date night",
    }

    recs = get_recommendations(sample_candidates, sample_prefs)
    print(json.dumps({"recommendations": recs}, indent=2, ensure_ascii=False))
    print(f"\nGot {len(recs)} recommendation(s) from Groq.")
