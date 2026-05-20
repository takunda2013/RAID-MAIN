# AI Essay Detector UI

This folder contains the Streamlit front end for the AI detector. The current
version is a static demo: it accepts pasted text or an uploaded `.pdf`, `.docx`,
or `.txt` file, then shows example detector outputs that match the planned
pipeline.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app with input, result, explanation, and report sections |
| `requirements.txt` | UI dependencies |
| `Input design.png`, `output design.png`, `menu design.png` | Reference sketches |

## Run the UI

From this folder:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Or from the repository root:

```bash
streamlit run Robust-Ai-Detector/UI/app.py
```

## Current Demo Behavior

The UI currently has three static demo scenarios:

- `Human-written essay`
- `AI-generated essay`
- `Adversarially paraphrased AI`

The file uploader accepts `.pdf`, `.docx`, and `.txt`. If `pypdf` and
`python-docx` are installed, the app tries to extract text from uploaded files.
The prediction itself is still static until the detector backend is connected.

## Login + User Management (SQLite)

The UI now includes basic user management backed by a local SQLite database.

- Users must be logged in to access the educator detector page.
- A `super_admin` can open the **User Management** view to create/disable users, change roles, and reset passwords.

Database location:

- Default: `Robust-Ai-Detector/UI/data/app.db`
- Override with: `AI_DETECTOR_DB_PATH=/absolute/path/to/app.db`

First-run bootstrap super admin (created automatically when the DB is empty):

- Email: `admin@local.com`
- Password: `admin1234`

## Planned Integration Shape

The Streamlit app should call one stable backend function instead of importing
training scripts directly from the UI:

```python
result = analyze_text(text)
```

Recommended return shape:

```python
{
    "label": "Human-written",
    "label_id": 0,
    "confidence": 0.92,
    "probabilities": {
        "human": 0.92,
        "ai": 0.05,
        "paraphrased_ai": 0.03
    },
    "semantic_embedding": [...],
    "statistical_features": {
        "mean_surprisal": 7.82,
        "stdev_surprisal": 1.33
    },
    "top_tokens": [
        {"rank": 1, "token": "therefore", "weight": 0.31, "polarity": "AI"}
    ],
    "annotated_spans": [
        {"text": "therefore", "polarity": "AI"}
    ]
}
```

Keep this function in a new backend module such as:

```text
Robust-Ai-Detector/backend/inference.py
```

That avoids coupling Streamlit to batch training and extraction scripts.

## How The Existing Modules Fit Together

### 1. Text Input Layer

The UI collects text from:

- pasted essay text
- uploaded PDF
- uploaded DOCX
- uploaded TXT

Before inference, normalize the text in one place:

- strip empty whitespace
- enforce a minimum word count warning
- limit model input length consistently with the RoBERTa and GPT-2 extractors
- keep the original text for report generation

### 2. Semantic Features

Use the fine-tuned RoBERTa model from:

```text
Contrastive-Learning/stage2_roberta_finetune.py
```

The model class is `RoBERTaContrastive`. Its `forward()` returns:

| Output | Use |
|---|---|
| `pooled` | semantic feature vector for the detector |
| `projected` | contrastive embedding used during training |

The batch extractor in:

```text
Contrastive-Learning/stage3_feature_extraction.py
```

currently reads CSV splits and writes `.npy` files. For the UI, create a
single-text version of that logic:

```python
semantic_vector = extract_semantic_embedding(text)
```

It should load:

```text
Contrastive-Learning/models/roberta_contrastive/best_model.pt
Contrastive-Learning/models/roberta_contrastive/tokenizer/
```

and return one vector shaped like `(hidden_size,)`.

### 3. Statistical Features

Use:

```text
Statistical-Calculator/code/raid_features.py
```

The class `RaidFeatureExtractor` already supports single-text feature
calculation:

```python
extractor = RaidFeatureExtractor("gpt2-large", device="auto")
stats = extractor.compute(text)
```

The output feature names are:

- `mean_surprisal`
- `stdev_surprisal`
- `var_surprisal`
- `skew_surprisal`
- `kurtosis_surprisal`
- `mean_diff_surprisal`
- `stdev_diff_surprisal`
- `var_second_diff_loglik`
- `entropy_second_diff_loglik`
- `autocorr_second_diff_loglik`

### 4. Concatenation Engine

The current concatenation code is:

```text
Concatenation-engine/feature_concatenate_engine.py
```

It is batch-oriented. It loads saved RoBERTa embeddings and saved statistical
feature CSVs, then writes concatenated `.npy` matrices.

For the UI, add a single-example helper that follows the same order:

```python
combined = concatenate_single(semantic_vector, statistical_vector)
```

The important rule is consistency: the statistical feature order must match the
order used during classifier training.

### 5. Final Classifier

After concatenating features, the backend should load the trained classifier
that maps the combined vector to:

| Label ID | UI Label |
|---|---|
| `0` | Human-written |
| `1` | AI-generated |
| `2` | Adversarially paraphrased AI |

The classifier can be XGBoost, scikit-learn, or another model. Save both the
model artifact and any scaler/feature metadata needed at inference time.

Example backend flow:

```python
def analyze_text(text: str) -> dict:
    clean_text = normalize_text(text)
    semantic = extract_semantic_embedding(clean_text)
    stats = extract_statistical_features(clean_text)
    combined = concatenate_single(semantic, stats)
    probabilities = classifier.predict_proba(combined.reshape(1, -1))[0]
    explanation = explain_prediction(clean_text, combined, probabilities)
    return format_for_ui(clean_text, probabilities, stats, explanation)
```

### 6. Explanation Layer

The sketches include:

- annotated essay highlights
- token weight table
- polarity labels

For the real system, this can come from LIME, SHAP, or a custom token masking
method. Keep it optional at first: the UI can show the prediction immediately
and fill explanation fields only when they are available.

## Notes For Import Paths

Some existing folders use hyphens, such as `Contrastive-Learning` and
`Concatenation-engine`. Python cannot import those as normal packages. The
cleanest long-term options are:

- create import-friendly wrapper modules in `backend/`
- rename folders to use underscores when you are ready for a broader cleanup
- load legacy files with `importlib.util.spec_from_file_location()` inside the
backend only

Prefer the first option for the Streamlit app.

## Next Implementation Step

Replace the static result assignment in `app.py`:

```python
st.session_state.result = DEMO_RESULTS[demo_name]
```

with:

```python
from backend.inference import analyze_text

st.session_state.result = analyze_text(essay_text)
```

Then update the result rendering functions to read the backend dictionary rather
than the `DemoResult` dataclass.
