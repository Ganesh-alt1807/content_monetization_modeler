# Content Monetization Modeler

Simple Streamlit app to predict estimated YouTube revenue for a video using a pre-trained scikit-learn pipeline.

## What's in this repo
- `app/` – Streamlit app and assets (`assets/` contains background and icon SVGs).
- `data/` – source datasets (CSV).
- `model/` – development notebooks and artifacts.
- `scripts/train_dummy_model.py` – optional helper to create a sample model for testing.

## UI highlights
- Stunning glassmorphism-styled interface with a background image and YouTube icon in the header.
- Inputs are grouped into a clean two-column form; results display in a compact result card.
- The app silently attempts to load `app/yt_revenue_model.pkl` at startup (if present) and will auto-load the model when you first run a prediction if the file is added while the app is running.

## Run locally
1. Create and activate a Python environment (recommended).
2. Install dependencies:

```bash
pip install -r app/requirements.txt
```

3. From the `app/` folder run:

```bash
streamlit run app.py
```

4. Open the URL printed by Streamlit in your browser (e.g. `http://localhost:8501` or the next available port).

## Model
- The app looks for `app/yt_revenue_model.pkl`. If you don't have one, the `scripts/train_dummy_model.py` can create a compatible sample model for local testing.
- If the app cannot load your model, it will show the exception in the UI when attempting to predict.

## Notes
- Small UI tweaks were made (subtitle and model badge removed for a cleaner header per user preference).
- If you want a Git commit/push of the changes or a screenshot of the current UI, ask and I can do that next.

