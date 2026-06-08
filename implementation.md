# Implementation Plan: AI-Powered Restaurant Recommendation System

This document describes the phase-wise implementation plan for building the AI-Powered Restaurant Recommendation System. It is constructed based on the requirements in [context.md](file:///c:/Zomato_Milestone/context.md) and the system design in [architecture.md](file:///c:/Zomato_Milestone/architecture.md).

---

## 📅 Phase-Wise Execution Timeline

### 🛠️ Phase 1: Environment Setup & Data Ingestion Pipeline
* **Objective:** Establish the project environment and cache cleaned restaurant data from Hugging Face.
* **Tasks:**
  1. Create Python virtual environment (`.venv`).
  2. Install core packages (`pandas`, `streamlit`, `datasets`, `groq`, `python-dotenv`).
  3. Set up configuration templates (`.gitignore`, `.env.example`).
  4. Implement `data_ingestion.py` to:
     * Fetch `ManikaSaini/zomato-restaurant-recommendation` from Hugging Face.
     * Clean rating values (e.g. normalize `"3.8/5"` to `3.8`, convert `"-"` and `"NEW"` to `0.0`).
     * Clean cost values (e.g. convert `"1,200"` string to `1200.0` float).
     * Save preprocessed DataFrame locally to `data/zomato_cleaned.csv`.

### 🔍 Phase 2: Filtering & Integration Layer
* **Objective:** Narrow down candidate restaurants based on user inputs to manage token windows and improve LLM performance.
* **Tasks:**
  1. Implement dataset querying in `recommender_engine.py`.
  2. Filter by:
     * **Location:** Case-insensitive exact match.
     * **Cuisine:** Matches list keywords (e.g., "Italian", "Chinese").
     * **Budget:** High/Medium/Low categories mapped from average cost values.
     * **Rating:** Minimum threshold (e.g., `>= 3.5`).
  3. Sort and slice filtered results (top 5-10 candidates sorted by rating/votes) for the LLM.

### 🤖 Phase 3: Prompt Engineering & Groq API Orchestration
* **Objective:** Connect the filtered data with Groq to generate custom reasons and rankings.
* **Tasks:**
  1. Load API Keys safely using `python-dotenv`.
  2. Implement system instructions detailing the LLM Persona (*"Expert Gastronomy Guide"*).
  3. Request structured JSON output (`response_format={"type": "json_object"}`).
  4. Generate custom justifications emphasizing why each suggestion matches the user's custom preferences (e.g., "rooftop seating", "family-friendly").

### 🎨 Phase 4: Streamlit Dashboard UI
* **Objective:** Present a visual interface for inputting selections and reviewing custom recommendations.
* **Tasks:**
  1. Build `app.py` UI dashboard.
  2. Apply custom styling injections (custom fonts, Zomato-red style headers, card grid shadows, hover animations).
  3. Form inputs (Location selector, Budget boxes, Cuisine multiselect, Rating slider, Custom preferences field).
  4. Streamlit components to render the restaurant list as visual cards containing name, rating badge, price, cuisines, and AI recommendations.

### 🧪 Phase 5: Verification & Testing
* **Objective:** Guarantee data cleanliness, prompt validity, and UI stability.
* **Tasks:**
  1. Verify data parser with edge cases.
  2. Test local Streamlit launch: `streamlit run app.py`.
  3. Confirm dynamic recommendation generation with active API keys.

---

## 🧬 Component Map

```
Zomato_Milestone/
├── .venv/                   # Python virtual environment
├── data/
│   └── zomato_cleaned.csv   # Local preprocessed data cache
├── .env                     # Sensitive environment keys (GROQ_API_KEY)
├── .env.example             # Shared environment template
├── .gitignore               # Ignored build & key files
├── requirements.txt         # Project package requirements
├── data_ingestion.py        # Ingestion pipeline
├── recommender_engine.py    # Filtering & Groq API bridge
└── app.py                   # Streamlit Frontend application
```
