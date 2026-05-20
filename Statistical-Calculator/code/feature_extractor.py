r"""
Feature extraction only (no model training).

Usage:
python feature_extractor.py --train-csv ../datasets/train.csv --val-csv ..\datasets\val.csv  --test-csv ..\datasets\test.csv  --text-col Text   --label-col generated --model-name gpt2-large

What this writes by default:
- ..\datasets\train_features.csv
- ..\datasets\val_features.csv
- ..\datasets\test_features.csv
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from raid_features import FEATURE_NAMES, RaidFeatureExtractor


def _load_csv(path: Path, text_col: str, label_col: str, limit: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(path, engine='python', on_bad_lines='warn')
    if text_col not in df.columns:
        raise ValueError(f"Missing text column '{text_col}' in {path}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in {path}")
    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    if limit is not None:
        df = df.head(limit)
    return df


def _normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
    except ValueError:
        pass
    return text


def _ensure_row_index(
    features_path: Path,
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
) -> pd.DataFrame:
    full = pd.read_csv(features_path)
    if "row_index" in full.columns:
        return full

    lookup: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx, row in df.iterrows():
        text_key = str(row[text_col]).strip()
        label_key = _normalize_label(row[label_col])
        lookup[(text_key, label_key)].append(int(idx))

    row_indices: List[int] = []
    for _, row in full.iterrows():
        text_key = str(row[text_col]).strip()
        label_key = _normalize_label(row[label_col])
        key = (text_key, label_key)

        if key not in lookup or not lookup[key]:
            row_indices.append(-1)
            continue

        row_indices.append(lookup[key].pop(0))

    full["row_index"] = row_indices
    full.to_csv(features_path, index=False)
    return full


def _extract_features_with_checkpoint(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    extractor: RaidFeatureExtractor,
    features_path: Path,
    *,
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    features_path.parent.mkdir(parents=True, exist_ok=True)

    start_idx = 0
    if features_path.exists():
        existing = _ensure_row_index(features_path, df, text_col, label_col)
        existing_valid = existing[existing["row_index"].astype(int) >= 0]

        n_existing = len(existing_valid)
        if n_existing >= len(df):
            X = existing_valid[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
            kept = existing_valid["row_index"].astype(int).tolist()
            return X, kept
        start_idx = int(existing_valid["row_index"].max()) + 1 if n_existing > 0 else 0
        print(f"Resuming feature extraction from row {start_idx} for {features_path.name}")
    else:
        print(f"Starting feature extraction for {features_path.name}")

    indices = range(start_idx, len(df))
    iterator = tqdm(indices, desc=f"Extracting features -> {features_path.name}") if show_progress else indices

    for idx in iterator:
        row = df.iloc[idx]
        text = str(row[text_col]).strip()
        if not text:
            continue

        try:
            result = extractor.compute(text)
        except ValueError as e:
            msg = str(e)
            if "Need at least 2 tokens" in msg or "Need at least 10 token log-likelihoods" in msg:
                continue
            raise

        feature_values = [float(result.features[name]) for name in FEATURE_NAMES]

        record = {name: value for name, value in zip(FEATURE_NAMES, feature_values)}
        record[label_col] = row[label_col]
        record[text_col] = row[text_col]
        record["row_index"] = idx

        mode = "a" if features_path.exists() else "w"
        header = not features_path.exists()
        pd.DataFrame([record]).to_csv(features_path, mode=mode, header=header, index=False)

    full = pd.read_csv(features_path)
    if "row_index" in full.columns:
        full = full[full["row_index"].astype(int) >= 0]
        X = full[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
        kept = full["row_index"].astype(int).tolist()
    else:
        X = full[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
        kept = list(range(len(full)))
    return X, kept


def _extract_single(
    csv_path: Path,
    out_csv: Optional[Path],
    text_col: str,
    label_col: str,
    limit: Optional[int],
    extractor: RaidFeatureExtractor,
    show_progress: bool,
) -> None:
    df = _load_csv(csv_path, text_col, label_col, limit)
    features_path = out_csv if out_csv is not None else csv_path.with_name(csv_path.stem + "_features.csv")
    X, kept = _extract_features_with_checkpoint(
        df,
        text_col,
        label_col,
        extractor,
        features_path,
        show_progress=show_progress,
    )
    skipped = len(df) - len(kept)
    print(
        f"Wrote: {features_path} | rows_with_features={len(kept)} | "
        f"rows_skipped={skipped} | feature_dim={X.shape[1] if X.size else len(FEATURE_NAMES)}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Extract Raid-style features to CSV (no training)")
    p.add_argument("--train-csv", required=True, help="Path to training CSV")
    p.add_argument("--val-csv", help="Path to validation CSV")
    p.add_argument("--test-csv", help="Path to test CSV")
    p.add_argument("--text-col", default="text", help="Name of the text column")
    p.add_argument("--label-col", default="generated", help="Name of the label column")
    p.add_argument("--model-name", default="gpt2", help="Hugging Face model name")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    p.add_argument("--max-length", type=int, default=1024, help="Max token length")
    p.add_argument("--limit", type=int, help="Limit rows for quick runs")
    p.add_argument("--train-out-csv", help="Optional output path for train features CSV")
    p.add_argument("--val-out-csv", help="Optional output path for val features CSV")
    p.add_argument("--test-out-csv", help="Optional output path for test features CSV")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    args = p.parse_args()

    extractor = RaidFeatureExtractor(
        args.model_name,
        device=args.device,
        max_length=args.max_length,
    )

    _extract_single(
        csv_path=Path(args.train_csv),
        out_csv=Path(args.train_out_csv) if args.train_out_csv else None,
        text_col=args.text_col,
        label_col=args.label_col,
        limit=args.limit,
        extractor=extractor,
        show_progress=not args.no_progress,
    )

    if args.val_csv:
        _extract_single(
            csv_path=Path(args.val_csv),
            out_csv=Path(args.val_out_csv) if args.val_out_csv else None,
            text_col=args.text_col,
            label_col=args.label_col,
            limit=args.limit,
            extractor=extractor,
            show_progress=not args.no_progress,
        )

    if args.test_csv:
        _extract_single(
            csv_path=Path(args.test_csv),
            out_csv=Path(args.test_out_csv) if args.test_out_csv else None,
            text_col=args.text_col,
            label_col=args.label_col,
            limit=args.limit,
            extractor=extractor,
            show_progress=not args.no_progress,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
