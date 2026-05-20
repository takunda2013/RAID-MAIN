# Deployment

This directory is the live backend boundary for the UI.

The main entry point is:

```python
from deployment_pipeline import analyze_text

result = analyze_text(essay_text)
```

The UI should call this function and render the returned dictionary. It should
not import model training scripts directly.

## Model Roles

The deployment stack uses the models selected during the probability audit:

| Role | Model | Purpose |
|---|---|---|
| Production classifier | `xgb_semantic_isotonic_calibrated_model.pkl` | Final Human vs AI decision. |
| Auditor model | `logistic_statistical_isotonic_calibrated_model.pkl` | LIME explanations and softer probability movement. |

This separation is intentional. The semantic XGBoost model is best for
classification accuracy, while the statistical logistic model is better for
LIME because its probabilities are less saturated.

## Returned Shape

`analyze_text(text)` returns a dictionary with:

- `label`, `label_id`
- `confidence`, `confidence_percent`
- `prob_ai`, `prob_human`, `ai_probability_percent`
- `production` model details
- `auditor` model details
- `statistical_features`
- `top_tokens`
- `annotated_spans`
- `summary`, `explanation`
- `lime_html`
- `warnings`

## CLI Usage

From the repository root:

```bash
python Robust-Ai-Detector/Deployment/deployment_pipeline.py \
  --essay-file "Robust-Ai-Detector/XGBoost Classifier/test_essay.txt" \
  --json
```

For a faster classification-only check:

```bash
python Robust-Ai-Detector/Deployment/deployment_pipeline.py \
  --essay-file "Robust-Ai-Detector/XGBoost Classifier/test_essay.txt" \
  --no-lime
```

## Live Streamlit App

A live UI replica was added at:

```text
Robust-Ai-Detector/UI/deploy_app.py
```

Run it with:

```bash
streamlit run Robust-Ai-Detector/UI/deploy_app.py
```

Optional tuning:

```bash
AI_DETECTOR_LIME_SAMPLES=20 AI_DETECTOR_LIME_FEATURES=10 \
streamlit run Robust-Ai-Detector/UI/deploy_app.py
```

The live default is intentionally small: `8` LIME samples and `12` features.
GPT-2-large statistical extraction is expensive because LIME calls the predictor
many times. Lower LIME samples make the app faster; higher LIME samples make
explanations more stable.
