# AI-Powered Restaurant Recommendation System (Zomato Use Case)

This document provides the context, objective, system workflow, and requirements for the AI-Powered Restaurant Recommendation System.

---

## 🎯 Objective
Design and implement an application that intelligently suggests restaurants based on user preferences by combining structured data with a Large Language Model (LLM).

The system will:
1. Take user preferences (location, budget, cuisine, and ratings).
2. Load and process a real-world restaurant dataset.
3. Leverage an LLM to generate personalized, human-like recommendations.
4. Display clear, formatted, and useful results to the user.

---

## ⚙️ System Workflow

### 1. Data Ingestion
* **Source:** Hugging Face Zomato Restaurant Recommendation Dataset 
  * URL: [Hugging Face Dataset](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation)
* **Action:** Load and preprocess the dataset, extracting relevant fields such as:
  * Restaurant Name
  * Location (City/Area)
  * Cuisine type
  * Cost (Average cost for two)
  * Ratings & reviews

### 2. User Input
Collect structured and unstructured preferences from the user:
* **Location** (e.g., Delhi, Bangalore)
* **Budget** (Low, Medium, High)
* **Cuisine** (e.g., Italian, Chinese, Indian)
* **Minimum Rating** (e.g., 3.5+, 4.0+)
* **Additional Preferences** (e.g., family-friendly, rooftop seating, quick service)

### 3. Integration Layer
* Filter the ingested restaurant dataset based on user criteria (location, budget, cuisine, rating).
* Select the top candidates and inject them into an optimized LLM prompt.
* Design the prompt to guide the LLM to reason over and rank the best options.

### 4. Recommendation Engine
* Pass the prompt to the LLM.
* Instruct the LLM to:
  * Rank the best matched restaurants.
  * Provide personalized, human-like explanations detailing why each restaurant fits the user's criteria.
  * Summarize the choices if necessary.

### 5. Output Display
Present the recommendations to the user in a clean, user-friendly format showing:
* **Restaurant Name**
* **Cuisine**
* **Rating**
* **Estimated Cost**
* **AI-Generated Explanation / Reasoning**
