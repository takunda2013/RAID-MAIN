# Robust AI Detector — Paraphrase-Resistant Pipeline

## Project Overview
A hybrid classifier that combines **RoBERTa semantic embeddings** (trained with Supervised Contrastive Loss) with **GPT-2 statistical features** to detect AI-generated text *even after paraphrasing attacks*.

---

## Pipeline Stages

```
Stage 1: stage1_data_prep.py       → Cleans, labels, groups your CSVs
Stage 2: stage2_roberta_finetune.py → Fine-tunes RoBERTa with SupConLoss
Stage 3: stage3_feature_extraction.py → Extracts 768-dim embeddings
Stage 4: (next phase) feature fusion + XGBoost + LIME
```

---

## Directory Structure

```
ai_detector/
├── data/
│   ├── raw/
│   │   ├── human_essays.csv          ← your human essays
│   │   ├── ai_original.csv           ← your AI-generated essays
│   │   ├── paraphrased_gpt_oss.csv   ← GPT-OSS paraphrases  (full coverage)
│   │   ├── paraphrased_llama.csv     ← Llama paraphrases     (full coverage)
│   │   └── paraphrased_deepseek.csv  ← DeepSeek paraphrases  (~50% coverage)
│   ├── processed/                    ← output of stage 1
│   └── embeddings/                   ← output of stage 3
├── models/
│   └── roberta_contrastive/          ← output of stage 2
├── stage1_data_prep.py
├── stage2_roberta_finetune.py
├── stage3_feature_extraction.py
└── requirements.txt
```

---

## Required CSV Format

### `human_essays.csv` and `ai_original.csv`
| text |
|------|
| Essay text here... |

### `paraphrased_*.csv`
| text | source_index |
|------|-------------|
| Paraphrased text... | 42 |

**`source_index`** = the row number (0-based) in `ai_original.csv` that this paraphrase was derived from. This is what links a paraphrase cluster together for contrastive learning.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
```bash
python stage1_data_prep.py
```
Outputs: `data/processed/train.csv`, `val.csv`, `test.csv`

### 3. Fine-tune RoBERTa (needs GPU)
```bash
python stage2_roberta_finetune.py
```
Outputs: `models/roberta_contrastive/best_model.pt`

Expected training time on 24 GB VRAM:
- ~3–6 hours for a typical dataset (10k–50k essays), 5 epochs

### 4. Extract semantic embeddings
```bash
python stage3_feature_extraction.py
```
Outputs: `data/embeddings/train_semantic_embeddings.npy` (and val, test)

---

## Key Design Decisions

### Why `group_id` splitting?
Train/val/test splits are done at the **group level** (not row level). This ensures the model never sees a paraphrase of a training essay at test time — preventing data leakage.

### Why `GroupSampler`?
Supervised Contrastive Loss needs positive pairs in the same batch. A random sampler would rarely put an AI original and its paraphrase in the same batch at small batch sizes. GroupSampler guarantees this.

### DeepSeek partial coverage
DeepSeek only covers ~50% of AI originals. Those AI originals without a DeepSeek paraphrase still appear in the dataset — they just form smaller groups (2 positives instead of 3). SupConLoss handles variable group sizes natively.

### `contrastive_mode` in training
- `"group"` (default): AI-original + all its paraphrases share the same positive class → model learns paraphrase invariance
- `"label"`: Original (label=1) and paraphrased (label=2) are different classes → model learns to distinguish them too

For your use case (detecting paraphrased AI), `"group"` mode is recommended.

---

## VRAM Usage (24 GB)
| Setting | Approx VRAM |
|---------|------------|
| roberta-base, batch=16, grad_checkpointing=True | ~10–12 GB |
| roberta-base, batch=32, grad_checkpointing=True | ~16–18 GB |
| roberta-large, batch=16, grad_checkpointing=True | ~18–20 GB |
