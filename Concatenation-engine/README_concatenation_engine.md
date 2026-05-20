# Concatenation Engine

## Overview

The concatenation engine bridges two independent feature pipelines —
the **Contrastive-Learning** module and the **Statistical-Calculator** module —
into a single unified feature matrix that is fed to the downstream XGBoost classifier.

It produces one `.npy` file per data split containing a row-aligned concatenation
of semantic embeddings and hand-crafted statistical features for every essay.

---

## Where It Fits in the Full Pipeline

```
Stage 1 (Data Prep)
    └─► train.csv / val.csv / unseen.csv

Stage 2 (Contrastive Fine-Tuning)
    └─► RoBERTa-large fine-tuned with SupCon Loss
    └─► val_semantic_embeddings.npy   (N, 1024)
    └─► unseen_semantic_embeddings.npy

Statistical Calculator
    └─► val_features.csv             (N, 10)
    └─► unseen_features.csv

Concatenation Engine  ◄── YOU ARE HERE
    └─► val_concatenated.npy         (N, 1034)
    └─► unseen_concatenated.npy      (N, 1034)
    └─► val_labels.npy
    └─► val_feature_names.txt

Stage 3 (XGBoost Classifier)
    └─► trained on (N, 1034) feature matrix
    └─► predicts: Human vs AI (Original / Paraphrased)
```

---

## Input Files

### Semantic Embeddings
Produced by the Stage 2 RoBERTa contrastive fine-tuning pipeline.

| File | Shape | Description |
|------|-------|-------------|
| `val_semantic_embeddings.npy` | `(N, 1024)` | Pooled CLS embeddings from RoBERTa-large for each val essay |
| `unseen_semantic_embeddings.npy` | `(N, 1024)` | Pooled CLS embeddings for unseen/test essays |

> **Note:** The embedding dimension is **1024** because `roberta-large` is used.
> This is dynamic — read from `model.config.hidden_size` at runtime, never hardcoded.

### Statistical Features
Produced by the Statistical Calculator module.

| File | Columns | Description |
|------|---------|-------------|
| `val_features.csv` | `Text` + 10 features | Surprisal-based statistical features for val essays |
| `unseen_features.csv` | `Text` + 10 features | Surprisal-based statistical features for unseen essays |

**The 10 statistical features:**

| # | Feature | Description |
|---|---------|-------------|
| 1 | `mean_surprisal` | Mean token surprisal across essay |
| 2 | `stdev_surprisal` | Standard deviation of token surprisal |
| 3 | `var_surprisal` | Variance of token surprisal |
| 4 | `skew_surprisal` | Skewness of surprisal distribution |
| 5 | `kurtosis_surprisal` | Kurtosis of surprisal distribution |
| 6 | `mean_diff_surprisal` | Mean of first-order surprisal differences |
| 7 | `stdev_diff_surprisal` | Std dev of first-order surprisal differences |
| 8 | `var_second_diff_loglik` | Variance of second-order log-likelihood differences |
| 9 | `entropy_second_diff_loglik` | Entropy of second-order log-likelihood differences |
| 10 | `autocorr_second_diff_loglik` | Autocorrelation of second-order log-likelihood differences |

---

## Output Files

All outputs are written to `Engine/data/concatenated/`.

| File | Shape | Description |
|------|-------|-------------|
| `{split}_concatenated.npy` | `(N, 1034)` | Final feature matrix (semantic + statistical) |
| `{split}_labels.npy` | `(N,)` | Integer labels (0=Human, 1=AI, 2=Paraphrased) |
| `{split}_feature_names.txt` | — | One feature name per line: `semantic_0` ... `semantic_1023`, then the 10 statistical names |

### Feature Layout Inside the Matrix

```
Columns 0    → 1023  : RoBERTa-large pooled CLS embeddings  (1024 dims)
Columns 1024 → 1033  : Statistical surprisal features        (10 dims)
                       ─────────────────────────────────────────────────
Total                : 1034 features per essay
```

---

## How Essay Matching Works

The engine cannot assume that both input files are in the same row order,
so it matches essays across files using the **essay text itself as the join key**.

**Matching steps:**

**1. Normalize text keys**
```
lowercase + strip whitespace + NFKC unicode normalization
→ produces a consistent key regardless of minor formatting differences
```

**2. Handle duplicate essays with occurrence counting**
```
If the same essay text appears multiple times in both files,
each occurrence is matched in order of appearance:
  ("the essay text...", 0) ↔ ("the essay text...", 0)
  ("the essay text...", 1) ↔ ("the essay text...", 1)
```
This prevents random mismatches when duplicates exist.

**3. Find common keys and align**
```
Essays in semantic but missing from statistical → dropped (warning logged)
Essays in statistical but missing from semantic → dropped (warning logged)
Remaining common essays → aligned and concatenated
```

**4. Concatenate horizontally**
```python
concatenated = np.hstack([aligned_semantic, aligned_statistical])
# (N, 1024) + (N, 10) → (N, 1034)
```

---

## Configuration

All paths and settings are defined in the `CONFIG` dict at the top of the script:

```python
CONFIG = {
    "semantic_dir":    "Contrastive-Learning/data/embeddings/",
    "statistical_dir": "Statistical-Calculator/datasets/",
    "output_dir":      "Engine/data/concatenated/",
    "splits":          ("val", "unseen"),
    "semantic_csvs": {
        "val":    "Contrastive-Learning/data/processed/val.csv",
        "unseen": "Contrastive-Learning/data/unseen.csv",
    },
}
```

To add a new split (e.g. a test set):
```python
"splits": ("val", "unseen", "test"),
"semantic_csvs": {
    ...
    "test": "Contrastive-Learning/data/test.csv",
},
```
And ensure `test_semantic_embeddings.npy` and `test_features.csv` exist.

---

## Usage

```bash
python concatenate_features.py
```

No arguments required — all paths are resolved from `CONFIG`.

---

## Error Handling

| Condition | Behaviour |
|-----------|-----------|
| `.npy` file not found | Raises `FileNotFoundError` with full path |
| `Text` column missing from CSV | Raises `KeyError` |
| No numeric feature columns found | Raises `ValueError` |
| Row count mismatch between files | Logs warning, falls back to text-based alignment |
| Essays missing from one side | Logs warning with example text, drops unmatched rows |
| Duplicate essays | Logs warning, matched by occurrence order |

---

## Key Design Decisions

**Why text-based matching instead of row index?**
The semantic embeddings and statistical features are generated by two
completely independent pipelines that may process essays in different orders.
Using the essay text as the join key makes the engine robust to ordering
differences without requiring a shared ID column.

**Why pooled (1024) and not projected (128)?**
The projected embeddings (128-dim) are L2-normalized vectors optimized for
the SupCon loss function during contrastive training. The pooled CLS embeddings
(1024-dim) carry the full representational capacity of RoBERTa-large and are
richer features for a downstream classifier. The stage2 script comments
explicitly note: `pooled — used for feature extraction in stage 3`.

**Why concatenate instead of using embeddings alone?**
The statistical features capture interpretable surprisal-based signals
(burstiness, entropy, autocorrelation) that complement the deep semantic
features from RoBERTa. Together they give XGBoost both pattern-level and
statistical-level evidence for its classification decision.
