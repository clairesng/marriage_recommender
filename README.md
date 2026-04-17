# 💍 Marriage Recommender

A full-stack data science project that turns relationship survey data into an interactive, two-player compatibility experience.

This project combines:
- **Data analysis** (EDA + feature engineering)
- **Machine learning** (CatBoost classification)
- **Product thinking** (human-friendly outputs, not just raw predictions)
- **App deployment** (Streamlit interface with response logging)

It was built using anonymised data from married and divorced respondents in Singapore.

---

## Background

Relationship compatibility is often discussed in subjective terms, but many of the patterns that shape long-term outcomes can be explored through data. This project started with a simple question: can survey responses about values, family background, finances, and lifestyle preferences be turned into a useful compatibility signal?

The purpose of the project is not to predict marriage success with certainty. Instead, it aims to help users reflect on areas of alignment and possible friction in a more structured, data-informed way. By pairing machine learning with a conversational interface, the app makes the insights easier to understand for both technical and non-technical users.

---

## Why this project matters

For non-technical readers: this app is designed as a **reflection tool** for couples. It does not "decide" whether a relationship will succeed. Instead, it highlights patterns and areas for discussion.

For technical readers: this is an end-to-end project showing the ability to take a problem from **data → model → productised app**.

---

## What I built

### 1) Data and exploration
- Cleaned and explored survey data from married/divorced respondents
- Built feature-ready datasets for modelling
- Created exploratory analysis notebooks to inspect distributions and relationship patterns

### 2) Feature engineering and modelling
- Selected 14 predictive relationship features (categorical + numeric)
- Built preprocessing pipeline using:
	- `SimpleImputer` (missing values)
	- `OneHotEncoder` (categorical encoding)
	- `ColumnTransformer` + `Pipeline` (reproducible transforms)
- Trained and exported a **CatBoostClassifier** model (`.cbm`)
- Matched app-side preprocessing to training-time schema for consistent inference

### 3) Streamlit application
- Designed a **two-player quiz flow** (Girlfriend/Boyfriend answers handled separately)
- Combined answers into model-ready features
- Generated:
	- compatibility score
	- banded interpretation
	- archetype profile
	- growth recommendation
- Added EDA dashboard tab directly in app for transparency
- Logged anonymised submissions to Google Sheets for lightweight analytics

---

## Technical skills demonstrated

- **Python**: data wrangling, feature logic, model integration
- **Pandas**: dataset prep, transformation, cross-tab analysis
- **scikit-learn**: preprocessing pipelines and transformer orchestration
- **CatBoost**: model training/inference for mixed-feature classification
- **Streamlit**: interactive UI, state management, deployment-ready app design
- **Experiment-to-product workflow**: notebooks → trained artefact → production app logic
- **Applied ML communication**: translating prediction output into understandable recommendations

---

## Repository structure

- [divorced_eda.ipynb](divorced_eda.ipynb) — exploratory data analysis
- [divorced_modelling.ipynb](divorced_modelling.ipynb) — modelling workflow and experiments
- [divorced.csv](divorced.csv) — source dataset
- [divorced_feature_store.csv](divorced_feature_store.csv) — model-ready feature store
- [marriage_app/marriageapp.py](marriage_app/marriageapp.py) — Streamlit application
- [marriage_app/catboost_run4_best_model.cbm](marriage_app/catboost_run4_best_model.cbm) — trained CatBoost model
- [requirements.txt](requirements.txt) — project dependencies

---

## How to run locally

1. Clone this repo
2. Install dependencies
	 - `pip install -r requirements.txt`
3. Start the app
	 - `streamlit run marriage_app/marriageapp.py`

---

## Product and modelling notes

- The model output is used as a **probabilistic signal**, not a deterministic verdict.
- App messaging is intentionally framed as **self-reflection and growth guidance**.
- The app includes guardrails (clear interpretation notes, non-clinical framing).

---

## Potential improvements

- Add model evaluation summary section (AUC, F1, confusion matrix snapshots)
- Introduce model monitoring / drift checks for new incoming responses
- Add explainability layer (e.g., SHAP feature contribution view)
- Expand question set and test fairness/generalisation across subgroups

---

## Recruiter quick summary

This project demonstrates the ability to:
- frame a real-world problem,
- build and evaluate an ML pipeline,
- deploy an interactive product,
- communicate outputs responsibly to both technical and non-technical users.

If you are hiring for **data analyst / data scientist / applied ML / analytics engineering** paths, this project reflects strong end-to-end ownership.
