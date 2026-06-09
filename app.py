"""Phase 4: Streamlit Dashboard UI.

A visual interface for entering restaurant preferences and reviewing
AI-generated recommendations produced by the Phase 3 Groq engine
(`recommender_engine.get_recommendations`).

The minimal data load + filtering needed to feed the UI live in this file so the
dashboard is runnable on its own; they can later be replaced by the dedicated
Phase 1 (data ingestion) and Phase 2 (filtering) modules.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from recommender_engine import RecommendationError, get_recommendations

DATASET_ID = "ManikaSaini/zomato-restaurant-recommendation"
MAX_CANDIDATES = 10

# Budget buckets keyed to approx cost for two (INR).
BUDGET_RANGES = {
    "Low": (0, 500),
    "Medium": (500, 1500),
    "High": (1500, float("inf")),
}

st.set_page_config(page_title="AI Restaurant Recommender", page_icon="🍽️", layout="wide")

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .hero-title {
        color: #cb202d;
        font-weight: 700;
        font-size: 2.6rem;
        margin-bottom: 0;
    }
    .hero-sub { color: #6b6b6b; font-size: 1.05rem; margin-top: 0; }

    .rec-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        height: 100%;
    }
    .rec-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 26px rgba(203,32,45,0.18);
    }
    .rec-name { font-size: 1.25rem; font-weight: 600; color: #1c1c1c; margin-bottom: 0.4rem; }
    .rating-badge {
        display: inline-block;
        background: #1e7e34;
        color: #fff;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.9rem;
    }
    .price-tag { color: #3d3d3d; font-weight: 600; }
    .cuisine-chip {
        display: inline-block;
        background: #fdecee;
        color: #cb202d;
        border-radius: 12px;
        padding: 2px 10px;
        margin: 2px 4px 2px 0;
        font-size: 0.8rem;
    }
    .ai-reason {
        background: #faf7f0;
        border-left: 3px solid #cb202d;
        padding: 0.6rem 0.8rem;
        border-radius: 6px;
        margin-top: 0.7rem;
        font-size: 0.92rem;
        color: #444;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _parse_rate(value: object) -> float:
    """Normalize a rating like '4.1/5' to 4.1; '-'/'NEW'/blank -> 0.0."""
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text or text.upper() in {"-", "NEW", "NAN"}:
        return 0.0
    match = re.search(r"\d+(\.\d+)?", text)
    return float(match.group()) if match else 0.0


def _parse_cost(value: object) -> float:
    """Normalize an approx-cost string like '1,200' to 1200.0; invalid -> 0.0."""
    if value is None:
        return 0.0
    text = str(value).replace(",", "").strip()
    match = re.search(r"\d+(\.\d+)?", text)
    return float(match.group()) if match else 0.0


@st.cache_data(show_spinner="Loading restaurant data...")
def load_data() -> pd.DataFrame:
    """Load and clean the Zomato dataset into a tidy DataFrame (cached)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="train")
    df = ds.to_pandas()

    cost_col = "approx_cost(for two people)"
    out = pd.DataFrame(
        {
            "name": df["name"].fillna("").astype(str).str.strip(),
            "location": df["location"].fillna("").astype(str).str.strip(),
            "cuisines": df["cuisines"].fillna("").astype(str).str.strip(),
            "rating": df["rate"].map(_parse_rate),
            "cost_for_two": df[cost_col].map(_parse_cost),
            "votes": pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int),
        }
    )
    out = out[out["name"] != ""].drop_duplicates(subset=["name", "location"])
    return out.reset_index(drop=True)


def filter_candidates(
    df: pd.DataFrame,
    location: str,
    cuisines: list[str],
    budget: str,
    min_rating: float,
) -> pd.DataFrame:
    """Apply the UI filters and return the top candidates for the LLM."""
    result = df
    if location and location != "Any":
        result = result[result["location"].str.casefold() == location.casefold()]
    if cuisines:
        pattern = "|".join(re.escape(c) for c in cuisines)
        result = result[result["cuisines"].str.contains(pattern, case=False, na=False)]
    if budget in BUDGET_RANGES:
        low, high = BUDGET_RANGES[budget]
        result = result[(result["cost_for_two"] > low) & (result["cost_for_two"] <= high)]
    result = result[result["rating"] >= min_rating]
    return result.sort_values(["rating", "votes"], ascending=False).head(MAX_CANDIDATES)


def render_card(rec: dict, lookup: dict[str, dict]) -> str:
    """Build the HTML for a single recommendation card."""
    name = str(rec.get("name", "Unknown"))
    src = lookup.get(name.casefold(), {})
    rating = rec.get("rating") or src.get("rating", 0.0)
    cost = rec.get("cost_for_two") or src.get("cost_for_two", 0.0)
    cuisines = rec.get("cuisine") or src.get("cuisines", "")
    reason = str(rec.get("reason", ""))

    chips = "".join(
        f'<span class="cuisine-chip">{c.strip()}</span>'
        for c in str(cuisines).split(",")
        if c.strip()
    )
    return f"""
    <div class="rec-card">
        <div class="rec-name">{name}</div>
        <span class="rating-badge">★ {float(rating):.1f}</span>
        &nbsp;<span class="price-tag">₹{int(float(cost))} for two</span>
        <div style="margin-top:0.5rem;">{chips}</div>
        <div class="ai-reason">🤖 {reason}</div>
    </div>
    """


st.markdown('<p class="hero-title">🍽️ AI Restaurant Recommender</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Tell us what you\'re craving and let our Expert Gastronomy '
    "Guide (powered by Groq) pick the best spots for you.</p>",
    unsafe_allow_html=True,
)

df = load_data()
locations = ["Any"] + sorted(loc for loc in df["location"].unique() if loc)
cuisine_options = sorted(
    {
        c.strip()
        for row in df["cuisines"]
        for c in str(row).split(",")
        if c.strip()
    }
)

with st.sidebar:
    st.header("Your preferences")
    sel_location = st.selectbox("📍 Location", locations, index=0)
    sel_budget = st.radio("💰 Budget", list(BUDGET_RANGES.keys()), index=1, horizontal=True)
    sel_cuisines = st.multiselect("🍜 Cuisine", cuisine_options)
    sel_rating = st.slider("⭐ Minimum rating", 0.0, 5.0, 4.0, 0.1)
    sel_prefs = st.text_input(
        "✨ Additional preferences",
        placeholder="e.g. rooftop seating, good for family, quick bite",
    )
    submitted = st.button("Find restaurants", type="primary", use_container_width=True)

if submitted:
    candidates = filter_candidates(df, sel_location, sel_cuisines, sel_budget, sel_rating)

    if candidates.empty:
        st.warning(
            "No restaurants matched your filters. Try widening the budget, lowering "
            "the minimum rating, or changing the location/cuisine."
        )
    else:
        candidate_records = candidates.to_dict(orient="records")
        lookup = {str(r["name"]).casefold(): r for r in candidate_records}
        preferences = {
            "location": None if sel_location == "Any" else sel_location,
            "cuisine": sel_cuisines,
            "budget": sel_budget,
            "min_rating": sel_rating,
            "additional": sel_prefs,
        }

        try:
            with st.spinner("Asking the Expert Gastronomy Guide..."):
                recommendations = get_recommendations(candidate_records, preferences)
        except RecommendationError as exc:
            st.error(f"Could not generate recommendations: {exc}")
            recommendations = []

        if recommendations:
            st.subheader(f"Top {len(recommendations)} picks for you")
            cols = st.columns(2)
            for i, rec in enumerate(recommendations):
                with cols[i % 2]:
                    st.markdown(render_card(rec, lookup), unsafe_allow_html=True)
        elif recommendations == []:
            st.info("No recommendations were returned. Please try different preferences.")
else:
    st.info("Set your preferences in the sidebar and click **Find restaurants** to begin.")
