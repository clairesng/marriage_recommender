# 💍 Marriage Compatibility Recommender 

A full-stack data science project that turns relationship survey data into an interactive, two-player compatibility experience. Based on Singaporean survey data of married and divorced couples, the app uses a CatBoost classification model to generate a compatibility score, interpretive band, archetype profile, and growth recommendation for couples. 

This project combines:
- **Data analysis** (End-to-end EDA + feature engineering)
- **Machine learning** (CatBoost classification)
- **Product thinking** (human-friendly outputs, not just raw predictions)
- **App deployment** (Streamlit interface with response logging)

---

## Background

Compatibility is often discussed in subjective terms, and while the rates of marriage and divorce in Singapore are publicly available, the factors that drive a marriage's success or failure haven't been fully explored from a Data Analyst/Scientist perspective. 

---

## Why this project matters

For non-technical readers: this app is designed as a **reflection tool** for couples, helping them gain insights into their relationship dynamics in a structured, data-informed way. 

For technical readers: this is an end-to-end project showing the ability to take a problem from **data → model → productised app**. By taking a "social science meets data science" approach, it demonstrates how machine learning can be used in a domain that is often seen as purely qualitative.

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
- Logged anonymised submissions to Sheets for lightweight analytics

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
- Marriage Recommender app includes guardrails (clear interpretation notes, non-clinical framing).

---

## Potential improvements

- Add model evaluation summary section (AUC, F1, confusion matrix snapshots)
- Increase dataset size and diversity to improve statistical power, reduce sampling bias and strengthen model generalisation
- Introduce model monitoring / drift checks for new incoming responses
- Add explainability layer (e.g., SHAP feature contribution view)
- Expand question set and test generalisation across subgroups

## Conclusion

This project demonstrates one example of how data analytics and machine learning can be applied to a qualitative topic like compatibility. In Singapore, where recent marriage/divorce/fertility rates are of public interest, the Marriage Recommender app offers an accessible and engaging way to provide insights for couples to engage with the data and reflect on relationship dynamics.