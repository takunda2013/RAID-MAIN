# Feature Extractor (`feature_extractor.py`)

This script extracts DivEye-style features from CSV text data and writes them to feature CSV files.

It uses the same feature logic as `train_xgb.py` but does **not** train XGBoost.

## What it does

- Loads CSV files (`train`, optional `val`, optional `test`)
- Reads text from `--text-col` and labels from `--label-col`
- Computes the 10 DivEye/surprisal-based features per row
- Saves features to CSV with checkpoint/resume support

## Requirements

- Python packages from `requirements.txt`
- Local file: `diveye_features.py`
- Access to download Hugging Face model weights on first run (unless already cached)

Install dependencies:

```powershell
python -m pip install -r .\requirements.txt
```

## Input CSV format

Each CSV must contain:

- Text column (for example `Text` or `text`)
- Label column (for example `generated`)

Rows with missing text/label are dropped. Very short texts may be skipped if there are too few tokens to compute features.

## Quick start (your current setup)

```powershell
python .\feature_extractor.py `
  --train-csv ..\datasets\train.csv `
  --val-csv ..\datasets\val.csv `
  --test-csv ..\datasets\test.csv `
  --text-col Text `
  --label-col generated `
  --model-name gpt2-medium
```

Default output files:

- `..\datasets\train_features.csv`
- `..\datasets\val_features.csv`
- `..\datasets\test_features.csv`

## Minimal run (train only)

```powershell
python .\feature_extractor.py `
  --train-csv ..\datasets\train.csv `
  --text-col Text `
  --label-col generated `
  --model-name gpt2-medium
```

## Optional custom output paths

```powershell
python .\feature_extractor.py `
  --train-csv ..\datasets\train.csv `
  --val-csv ..\datasets\val.csv `
  --text-col Text `
  --label-col generated `
  --model-name gpt2-medium `
  --train-out-csv .\train_features_custom.csv `
  --val-out-csv .\val_features_custom.csv
```

## Checkpoint and resume behavior

- If an output feature CSV already exists, extraction resumes from the next row.
- Progress is appended row by row, so interruption is safe.
- A `row_index` column is used to keep feature rows aligned to source rows.

## Command-line arguments

- `--train-csv` (required): training CSV path
- `--val-csv` (optional): validation CSV path
- `--test-csv` (optional): test CSV path
- `--text-col` (default `text`): text column name
- `--label-col` (default `generated`): label column name
- `--model-name` (default `gpt2`): Hugging Face causal LM model name
- `--device` (default `auto`): `auto`, `cpu`, or `cuda`
- `--max-length` (default `1024`): max token length for model input
- `--limit` (optional): process only first N rows of each input CSV
- `--train-out-csv` (optional): custom output path for train features
- `--val-out-csv` (optional): custom output path for val features
- `--test-out-csv` (optional): custom output path for test features
- `--no-progress`: disable tqdm progress bars

## Output columns

Feature CSV output includes:

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
- Original text column
- Original label column
- `row_index`

