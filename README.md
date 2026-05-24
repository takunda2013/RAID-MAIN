# RAID

RAID: Interpretable and Robust Detector for Student Essays (Robust AI Detector).

Large Language Models have increased their cleverness so that students can now automatically generate high quality academic essays, which challenges the well-established conventional means of measuring originality and authorship. Though detection tools have been developed and deployed, the problem is no longer one of simple generation but the deliberate and sophisticated obfuscation of that generated text. Evidence of this problem is demonstrated by recent studies which have shown that state of the art AI text detectors are consistently and significantly defeated when the LLM generated content is passed through an adversarial paraphraser a technique which uses another LLM to rewrite the text specifically to evade detection. The black box nature of current detectors further worsens the problem in that a high detection score cannot provide any explanatory evidence on which to base an appeal, making it impossible to confidently defend an academic integrity ruling. The ultimate result of this unresolved issue is the total loss of confidence in student submissions, which degrades the credential itself. Educators and honesty students are both experiencing the negative effects of these unreliable detection systems.

# Robust AI Detector

A paraphrase-resistant AI-text detection system that combines:

- **Semantic features** from contrastive fine-tuned RoBERTa (`1024` dims)
- **Statistical features** from GPT-2 surprisal analysis (`10` dims)
- **Production classifier**: calibrated XGBoost (`xgb_semantic_isotonic`)
- **Auditor classifier**: calibrated logistic statistical model (`logistic_statistical_isotonic`) for LIME-friendly explanations

---

## Repository Layout

```text
Contrastive-Learning/                 # Stage 1-3 semantic pipeline
Statistical-Calculator/code/          # 10-feature surprisal extractor
Concatenation-engine/                 # semantic + statistical feature fusion
XGBoost Classifier/                   # classifier training + audit experiments
Deployment/                           # backend inference boundary (analyze_text)
UI/                                   # Streamlit frontend
```

---

## System Requirements

## 1) Software

- Python **3.10+** (recommended: 3.10 or 3.11)
- `pip` 23+
- OS: Windows, Linux, or macOS
- Internet access at first run to download Hugging Face model weights/tokenizers

## 2) Hardware

### Deployment / inference only (recommended minimum)

- CPU: 4+ cores
- RAM: 16 GB
- Disk: 10+ GB free (models + caches)
- GPU: optional (CPU works but is slower)

### Training / feature extraction (recommended)

- NVIDIA GPU with CUDA (for practical runtime)
- VRAM:
  - `roberta-large` contrastive training: **~24 GB recommended**
  - `roberta-base` can run with less VRAM
- RAM: 32 GB preferred
- Disk: 30+ GB free for artifacts/checkpoints

---

## Installation Instructions

## 1) Create a virtual environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2) Install core runtime dependencies

From repository root:

```bash
python -m pip install -r "Deployment/requirements.txt"
```

This installs the backend + UI runtime stack (`torch`, `transformers`, `xgboost`, `lime`, `streamlit`, etc.).

## 3) (Optional) Install module-specific training dependencies

If you will run full training/ablation pipelines, also install:

```bash
python -m pip install -r "Contrastive-Learning/requirements.txt"
python -m pip install -r "Statistical-Calculator/code/requirements.txt"
python -m pip install -r "UI/requirements.txt"
```

---

## Artifacts Required for Deployment

`Deployment/deployment_pipeline.py` expects these artifacts to exist:

- `Contrastive-Learning/models/roberta_contrastive/best_model.pt`
- `Contrastive-Learning/models/roberta_contrastive/tokenizer/`
- `XGBoost Classifier/probability_audit_fixes/outputs/xgb_semantic_isotonic_calibrated_model.pkl`
- `XGBoost Classifier/probability_audit_fixes/outputs/logistic_statistical_isotonic_calibrated_model.pkl`
- `Concatenation-engine/data/concatenated/unseen_feature_names.txt`

If these are missing, build them using the pipeline below.

---

## Build Pipeline (if you need to regenerate models)

## Stage A — Contrastive semantic pipeline

```bash
python "Contrastive-Learning/stage1_data_prep.py"
python "Contrastive-Learning/stage2_roberta_finetune.py"
python "Contrastive-Learning/stage3_feature_extraction.py"
```

## Stage B — Statistical feature extraction

Example command:

```bash
python "Statistical-Calculator/code/feature_extractor.py" \
  --train-csv "Statistical-Calculator/datasets/train.csv" \
  --val-csv "Statistical-Calculator/datasets/val.csv" \
  --test-csv "Statistical-Calculator/datasets/test.csv" \
  --text-col Text \
  --label-col generated \
  --model-name gpt2-large
```

## Stage C — Concatenate semantic + statistical features

```bash
python "Concatenation-engine/feature_concatenate_engine.py"
```

## Stage D — Train and calibrate XGBoost/auditor models

```bash
python "XGBoost Classifier/probability_audit_fixes/run_probability_audit.py"
```

Outputs are written to:

- `XGBoost Classifier/probability_audit_fixes/outputs/`

---

## Deployment Guidelines

## 1) Backend inference entry point

Use this function as the stable integration boundary:

```python
from deployment_pipeline import analyze_text
result = analyze_text(essay_text)
```

Do **not** import training scripts from the UI/service layer.

## 2) Run backend from CLI

### Full output (with LIME)

```bash
python "Deployment/deployment_pipeline.py" \
  --essay-file "XGBoost Classifier/test_essay.txt" \
  --json
```

### Faster classification path (skip LIME)

```bash
python "Deployment/deployment_pipeline.py" \
  --essay-file "XGBoost Classifier/test_essay.txt" \
  --no-lime \
  --json
```

## 3) Generate performance report

```bash
python "Deployment/generate_performance_report.py" \
  --essay-file "XGBoost Classifier/test_essay.txt"
```

This writes:

- `Deployment/test_essay_performance.json`
- `Deployment/test_essay_performance.md`

## 4) Launch Streamlit UI

```bash
streamlit run "UI/deploy_app.py"
```

Optional tuning:

```bash
AI_DETECTOR_LIME_SAMPLES=8 AI_DETECTOR_LIME_FEATURES=12 streamlit run "UI/deploy_app.py"
```

Lower `LIME_SAMPLES` is faster; higher values improve explanation stability.

## 5) Deployment model roles (recommended)

- **Production decision model**: `xgb_semantic_isotonic_calibrated_model.pkl`
- **Auditor/explanation model**: `logistic_statistical_isotonic_calibrated_model.pkl`

This dual-model setup gives strong classification performance while preserving explanation-friendly probability movement.

---

## Operational Notes

- Essays under 100 words trigger a reliability warning in deployment output.
- First run may be slower due to model download and cache warm-up.
- Major latency contributors are semantic embedding extraction and LIME/statistical processing, not classifier inference.
- For API/server deployment, preload models once at startup and avoid per-request reinitialization.

---

## Quick Troubleshooting

- **`FileNotFoundError` for model artifacts**: run the build pipeline stages and verify paths above.
- **CUDA out-of-memory during training**: reduce batch size, use `roberta-base`, or disable expensive options.
- **Slow UI responses**: use `--no-lime` for classification-only checks or reduce LIME sample count.
- **Tokenizer/model download failures**: verify network access and Hugging Face availability.
