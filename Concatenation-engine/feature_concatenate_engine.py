"""
Concatenate contrastive RoBERTa embeddings with statistical essay features.

Creates two feature matrices:
    - val_concatenated.npy
    - test_concatenated.npy

The rows are assumed to already match between the contrastive-learning outputs
and the statistical-calculator outputs.
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


ENGINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENGINE_DIR.parent


CONFIG = {
    "semantic_dir": PROJECT_ROOT / "Contrastive-Learning" / "data" / "embeddings",
    "statistical_dir": PROJECT_ROOT / "Statistical-Calculator" / "datasets",
    "output_dir": ENGINE_DIR / "data" / "concatenated",
    "splits": ("val", "unseen"),
    "semantic_csvs": {
        "val": PROJECT_ROOT / "Contrastive-Learning" / "data" / "processed" / "val.csv",
        # "test": PROJECT_ROOT / "Contrastive-Learning" / "data" / "test.csv",
        "unseen": PROJECT_ROOT / "Contrastive-Learning" / "data" / "unseen.csv",
    },
    "non_feature_cols": {
        "Text",
        "label",
        "generated",
        "row_index",
        "group_id",
        "source",
        "source_split",
        "source_index",
        "paraphrase_index",
        "prompt_id",
        "model",
    },
    "label_cols": ("label", "generated"),
}


def normalize_text_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.normalize("NFKC").str.strip().str.lower()


def load_semantic_embeddings(split: str, semantic_dir: Path) -> np.ndarray:
    path = semantic_dir / f"{split}_semantic_embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(f"Semantic embeddings not found: {path}")

    embeddings = np.load(path)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D semantic embeddings for {split}, got {embeddings.shape}")

    return embeddings.astype(np.float32, copy=False)


def load_semantic_texts(split: str, semantic_csvs: dict[str, Path]) -> pd.DataFrame:
    path = Path(semantic_csvs[split])
    if not path.exists():
        raise FileNotFoundError(f"Semantic source CSV not found for {split}: {path}")

    df = pd.read_csv(path, engine="python", on_bad_lines="warn")
    if "Text" not in df.columns:
        raise KeyError(f"Expected 'Text' column in {path}, got columns: {list(df.columns)}")

    df = df.copy()
    df["__key__"] = normalize_text_series(df["Text"])
    return df


def load_statistical_features(
    split: str,
    statistical_dir: Path,
    non_feature_cols: set[str],
) -> tuple[np.ndarray, list[str], np.ndarray | None, pd.DataFrame]:
    path = statistical_dir / f"{split}_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Statistical features not found: {path}")

    df = pd.read_csv(path, engine="python", on_bad_lines="warn")
    if "Text" not in df.columns:
        raise KeyError(f"Expected 'Text' column in {path}, got columns: {list(df.columns)}")

    feature_cols = [
        col
        for col in df.columns
        if col not in non_feature_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not feature_cols:
        raise ValueError(f"No numeric statistical feature columns found in {path}")

    labels = None
    for label_col in CONFIG["label_cols"]:
        if label_col in df.columns:
            labels = df[label_col].to_numpy()
            break

    df = df.copy()
    df["__key__"] = normalize_text_series(df["Text"])
    features = df[feature_cols].to_numpy(dtype=np.float32)
    return features, feature_cols, labels, df


def align_by_text(
    split: str,
    semantic: np.ndarray,
    semantic_df: pd.DataFrame,
    statistical: np.ndarray,
    labels: np.ndarray | None,
    stat_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if len(semantic) != len(semantic_df):
        raise ValueError(
            f"Semantic metadata mismatch for {split}: "
            f"{len(semantic)} embeddings vs {len(semantic_df)} CSV rows"
        )
    if len(statistical) != len(stat_df):
        raise ValueError(
            f"Statistical metadata mismatch for {split}: "
            f"{len(statistical)} features vs {len(stat_df)} CSV rows"
        )

    semantic_df = semantic_df.copy()
    stat_df = stat_df.copy()
    semantic_df["__occ__"] = semantic_df.groupby("__key__").cumcount()
    stat_df["__occ__"] = stat_df.groupby("__key__").cumcount()
    semantic_df["__join_key__"] = list(zip(semantic_df["__key__"], semantic_df["__occ__"]))
    stat_df["__join_key__"] = list(zip(stat_df["__key__"], stat_df["__occ__"]))

    sem_dup_rows = int(semantic_df["__key__"].duplicated().sum())
    stat_dup_rows = int(stat_df["__key__"].duplicated().sum())
    if sem_dup_rows or stat_dup_rows:
        log.warning(
            "Split %s contains duplicate normalized texts: semantic=%d duplicate rows, statistical=%d duplicate rows. Aligning by text plus occurrence order.",
            split,
            sem_dup_rows,
            stat_dup_rows,
        )

    sem_map = {key: idx for idx, key in enumerate(semantic_df["__join_key__"])}
    stat_map = {key: idx for idx, key in enumerate(stat_df["__join_key__"])}
    common_keys = [key for key in semantic_df["__join_key__"] if key in stat_map]

    missing_stat_keys = [key for key in semantic_df["__join_key__"] if key not in stat_map]
    missing_sem_keys = [key for key in stat_df["__join_key__"] if key not in sem_map]

    if missing_stat_keys:
        sample_text = semantic_df.loc[
            semantic_df["__join_key__"] == missing_stat_keys[0], "Text"
        ].iloc[0]
        log.warning(
            "Split %s: dropping %d semantic rows missing from statistical features. "
            "Example text: %r",
            split,
            len(missing_stat_keys),
            sample_text[:200],
        )
    if missing_sem_keys:
        sample_text = stat_df.loc[
            stat_df["__join_key__"] == missing_sem_keys[0], "Text"
        ].iloc[0]
        log.warning(
            "Split %s: dropping %d statistical rows missing from semantic features. "
            "Example text: %r",
            split,
            len(missing_sem_keys),
            sample_text[:200],
        )

    sem_indices = np.array([sem_map[key] for key in common_keys], dtype=np.int64)
    stat_indices = np.array([stat_map[key] for key in common_keys], dtype=np.int64)

    aligned_semantic = semantic[sem_indices]
    aligned_statistical = statistical[stat_indices]
    aligned_labels = labels[stat_indices] if labels is not None else None
    return aligned_semantic, aligned_statistical, aligned_labels


def concatenate_split(split: str, cfg: dict) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    semantic = load_semantic_embeddings(split, Path(cfg["semantic_dir"]))
    semantic_df = load_semantic_texts(split, cfg["semantic_csvs"])
    statistical, stat_cols, labels, stat_df = load_statistical_features(
        split,
        Path(cfg["statistical_dir"]),
        set(cfg["non_feature_cols"]),
    )

    if len(semantic) != len(statistical):
        log.warning(
            "Row count mismatch for %s before alignment: %d semantic rows vs %d statistical rows",
            split,
            len(semantic),
            len(statistical),
        )
        semantic, statistical, labels = align_by_text(
            split,
            semantic,
            semantic_df,
            statistical,
            labels,
            stat_df,
        )
        log.info(
            "Aligned split %s by text: %d rows remain on both sides",
            split,
            len(semantic),
        )

    concatenated = np.hstack([semantic, statistical]).astype(np.float32, copy=False)
    feature_names = [f"semantic_{idx}" for idx in range(semantic.shape[1])] + stat_cols

    return concatenated, labels, feature_names


def save_split(split: str, features: np.ndarray, labels: np.ndarray | None, feature_names: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / f"{split}_concatenated.npy"
    feature_names_path = output_dir / f"{split}_feature_names.txt"

    np.save(features_path, features)
    feature_names_path.write_text("\n".join(feature_names) + "\n", encoding="utf-8")

    log.info("Saved %s features %s -> %s", split, features.shape, features_path)
    log.info("Saved %s feature names -> %s", split, feature_names_path)

    if labels is not None:
        labels_path = output_dir / f"{split}_labels.npy"
        np.save(labels_path, labels)
        log.info("Saved %s labels %s -> %s", split, labels.shape, labels_path)
    else:
        log.info("No labels found for %s; skipped label output.", split)


def run_concatenation(cfg: dict = CONFIG):
    output_dir = Path(cfg["output_dir"])

    for split in cfg["splits"]:
        log.info("Concatenating split: %s", split)
        features, labels, feature_names = concatenate_split(split, cfg)
        save_split(split, features, labels, feature_names, output_dir)

    log.info("Done. Concatenated datasets are in: %s", output_dir)


if __name__ == "__main__":
    run_concatenation()
