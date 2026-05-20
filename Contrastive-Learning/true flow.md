# True Flow of the AI Detector Pipeline

This project is currently organized into three main Python stages:

1. `stage1_data_prep.py` prepares and splits the raw text data.
2. `stage2_roberta_finetune.py` fine-tunes RoBERTa with supervised contrastive learning.
3. `stage3_feature_extraction.py` uses the fine-tuned RoBERTa model to turn each essay into semantic embedding features.

The main goal is to build a detector that can separate:

- Human-written essays: `label = 0`
- Original AI-written essays: `label = 1`
- Paraphrased AI-written essays: `label = 2`

---

## Stage 1: Data Preparation and Grouping

File:

`stage1_data_prep.py`

### What This Stage Achieves

Stage 1 takes the raw CSV files, cleans the essay text, assigns labels, creates contrastive-learning groups, and produces the processed train, validation, and test files used by the later stages.

This stage is important because Stage 2 needs essays to be grouped correctly. In particular, an AI original and its paraphrases should be linked through the same `group_id`, so the model can learn how AI text changes when it is paraphrased.

### Data Used as Input

Stage 1 reads these raw files:

| File | Purpose | Label Assigned |
|---|---|---|
| `data/raw/human_essays.csv` | Human-written essays | `0` |
| `data/raw/ai_original.csv` | Original AI-generated essays | `1` |
| `data/raw/paraphrased.csv` | Paraphrased AI essays from different paraphraser models | `2` |
| `data/raw/validation.csv` | Pre-built validation set with a `generated` label column | Converted into `label` |

The important columns are:

| Column | Meaning |
|---|---|
| `Text` | The essay text |
| `model` | The paraphraser model used in `paraphrased.csv` |
| `source_index` | Links a paraphrased essay back to its original AI essay |
| `generated` | Validation label column, where `0=human`, `1=AI original`, `2=paraphrased` |

### Cleaning Done

Each dataset is cleaned by:

- Dropping rows where `Text` is missing.
- Converting text to string.
- Stripping whitespace.
- Removing essays with text length less than or equal to 20 characters.

### Labels Created

The script creates a shared `label` column:

| Label | Meaning |
|---|---|
| `0` | Human essay |
| `1` | AI original essay |
| `2` | Paraphrased AI essay |

### Group IDs Created

The `group_id` column is created for contrastive learning:

| Data Type | Group ID Logic |
|---|---|
| AI original | `group_id` equals its row index in `ai_original.csv` |
| Paraphrased AI | `group_id` equals `source_index`, linking it to the AI original |
| Human essay | Gets a unique `group_id` because it does not belong to an AI/paraphrase cluster |

So if AI original essay number `15` has paraphrases, those paraphrases also get `group_id = 15`.

### Output Files Created

Stage 1 writes these files to `data/processed/`:

| Output File | What It Contains |
|---|---|
| `data/processed/full_train_pool.csv` | Combined human, AI original, and paraphrased data before train/test split |
| `data/processed/train.csv` | Main training data |
| `data/processed/test.csv` | Held-out test data |
| `data/processed/val.csv` | Cleaned version of the pre-built validation set |

The train/test split is done at the `group_id` level. This avoids leakage, because an AI original should not appear in train while one of its paraphrases appears in test.

### Important Clarification About the Test Set

The test set used later in Stage 3 is not a separate raw test file from the original data directory.

It is this file:

```text
data/processed/test.csv
```

This file is created by Stage 1 from the main raw data pool:

```text
data/raw/human_essays.csv
data/raw/ai_original.csv
data/raw/paraphrased.csv
```

Stage 1 combines those raw files, assigns labels and `group_id`s, then performs a 90/10 split:

```text
90% -> data/processed/train.csv
10% -> data/processed/test.csv
```

The raw validation file follows a different path:

```text
data/raw/validation.csv -> data/processed/val.csv
```

So the processed splits mean:

| Processed File | Where It Comes From |
|---|---|
| `data/processed/train.csv` | Created from raw human + AI original + paraphrased data |
| `data/processed/test.csv` | Created from raw human + AI original + paraphrased data |
| `data/processed/val.csv` | Cleaned from `data/raw/validation.csv` |

Stage 2 does not use `data/processed/test.csv`. The test set is kept aside until Stage 3 extracts embeddings from it for later evaluation.

---

## Stage 2: RoBERTa Contrastive Fine-Tuning

File:

`stage2_roberta_finetune.py`

### What This Stage Achieves

Stage 2 trains a RoBERTa model to create better essay embeddings for detecting AI and paraphrased AI text.

The model is not just learning ordinary classification. It uses supervised contrastive loss, which trains the embedding space so that texts with the same contrastive label are pulled closer together, while different labels are pushed apart.

In the current script, `contrastive_mode` is set to:

```python
"contrastive_mode": "label"
```

That means the contrastive positives are based on the `label` column:

- Human essays are pulled near other human essays.
- AI original essays are pulled near other AI original essays.
- Paraphrased AI essays are pulled near other paraphrased AI essays.

If this is changed to `"group"`, then AI originals and their paraphrases with the same `group_id` would be pulled together instead.

### Data Used as Input

Stage 2 uses the processed files from Stage 1:

| File | Used For |
|---|---|
| `data/processed/train.csv` | Training RoBERTa |
| `data/processed/val.csv` | Validation after each epoch |

The important columns are:

| Column | Used For |
|---|---|
| `Text` | Tokenized and fed into RoBERTa |
| `label` | Used for supervised contrastive labels when `contrastive_mode = "label"` |
| `group_id` | Used by the batch sampler, and used for contrastive labels if `contrastive_mode = "group"` |

### Model Being Trained

The script builds this model:

```text
RoBERTa backbone -> projection head -> normalized contrastive embedding
```

The model returns two outputs:

| Output | Shape | Purpose |
|---|---|---|
| `pooled` | 768 dimensions for `roberta-base` | Used later in Stage 3 as semantic features |
| `projected` | 128 dimensions | Used during Stage 2 for contrastive loss |

### Batching Logic

The custom `GroupSampler` tries to make sure each training batch contains positive pairs. This matters because supervised contrastive loss needs at least two examples that belong together in the same batch.

The sampler uses `group_id` to place related examples together, even when the current contrastive loss mode is `label`.

### Training Settings

Current important settings:

| Setting | Current Value |
|---|---|
| Model | `roberta-base` |
| Epochs | `3` |
| Batch size | `16` |
| Gradient accumulation | `2` |
| Effective batch size | `32` |
| Learning rate | `2e-5` |
| Max length | `512` tokens |
| Mixed precision | Enabled |
| Gradient checkpointing | Enabled |
| Early stopping patience | `2` epochs |

### Output Files Created

Stage 2 writes model artifacts to:

`models/roberta_contrastive/`

Important outputs:

| Output File | What It Contains |
|---|---|
| `checkpoint_epochX.pt` | Saved checkpoint after each epoch |
| `best_model.pt` | Best RoBERTa model weights based on validation loss |
| `tokenizer/` | Saved tokenizer files |
| `training_history.json` | Training and validation loss history |

The most important file for the next stage is:

`models/roberta_contrastive/best_model.pt`

---

## Stage 3: Semantic Feature Extraction

File:

`stage3_feature_extraction.py`

### What This Stage Achieves

Stage 3 loads the fine-tuned RoBERTa model from Stage 2 and uses it to convert every essay into a semantic embedding vector.

This stage does not train the model. It only runs inference.

The output is a set of `.npy` files containing 768-dimensional embeddings. These embeddings are intended to be used later by another stage, such as feature fusion or final classification.

### Data Used as Input

Stage 3 uses:

| Input | Purpose |
|---|---|
| `models/roberta_contrastive/best_model.pt` | Fine-tuned RoBERTa weights from Stage 2 |
| `models/roberta_contrastive/tokenizer/` | Tokenizer saved during Stage 2 |
| `data/processed/train.csv` | Texts to extract train embeddings |
| `data/processed/val.csv` | Texts to extract validation embeddings |
| `data/processed/test.csv` | Texts to extract test embeddings |

The only required text column is:

| Column | Purpose |
|---|---|
| `Text` | Tokenized and passed through RoBERTa |

The script also saves labels and group IDs if they exist.

### What Is Extracted

For each essay, the script saves the `pooled` RoBERTa output:

```text
1 essay -> 768-dimensional semantic embedding
```

These embeddings represent the meaning/style information learned by RoBERTa after contrastive fine-tuning.

### Output Files Created

Stage 3 writes files to:

`data/embeddings/`

| Output File | What It Contains |
|---|---|
| `train_semantic_embeddings.npy` | RoBERTa embeddings for `train.csv` |
| `val_semantic_embeddings.npy` | RoBERTa embeddings for `val.csv` |
| `test_semantic_embeddings.npy` | RoBERTa embeddings for `test.csv` |
| `train_labels.npy` | Labels for train rows |
| `val_labels.npy` | Labels for validation rows |
| `test_labels.npy` | Labels for test rows |
| `train_group_ids.npy` | Group IDs for train rows |
| `val_group_ids.npy` | Group IDs for validation rows |
| `test_group_ids.npy` | Group IDs for test rows |

---

## End-to-End Data Flow

```text
Raw CSV files
    |
    | stage1_data_prep.py
    v
Processed CSV files
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
    |
    | stage2_roberta_finetune.py
    v
Fine-tuned RoBERTa model
models/roberta_contrastive/best_model.pt
models/roberta_contrastive/tokenizer/
    |
    | stage3_feature_extraction.py
    v
Semantic embedding features
data/embeddings/*_semantic_embeddings.npy
data/embeddings/*_labels.npy
data/embeddings/*_group_ids.npy
```

---

## Short Summary

Stage 1 turns raw essay CSVs into clean, labeled, grouped train/validation/test data.

Stage 2 trains RoBERTa using the processed train and validation data so that its embeddings become useful for separating human, AI original, and paraphrased AI essays.

Stage 3 freezes that trained RoBERTa model and extracts 768-dimensional semantic embeddings for every row in train, validation, and test.
