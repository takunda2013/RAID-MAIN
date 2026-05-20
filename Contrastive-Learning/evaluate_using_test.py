"""
Evaluate the contrastive RoBERTa encoder on a genuinely held-out CSV.

Important distinction:
    - The probe is trained on a separate probe-training split.
    - The final reported metrics are computed only on the external test CSV.

The older version trained a logistic regression probe on 70% of the same file
it called "test" and evaluated on the remaining 30%. That measures how
linearly separable that one file is, but it is not a final held-out evaluation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from stage2_roberta_finetune import (
    ContrastiveEssayDataset,
    RoBERTaContrastive,
    TRAIN_CONFIG,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("evaluate_using_test")

BASE_DIR = Path(__file__).resolve().parent
cfg = TRAIN_CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EVAL_CONFIG = {
    "checkpoint": "models/roberta_contrastive/checkpoint_epoch1.pt",
    "fallback_checkpoint": "models/roberta_contrastive/best_model.pt",
    "tokenizer_dir": "models/roberta_contrastive/tokenizer",

    # Train only the lightweight linear probe here. Do not use the eval CSV.
    "probe_train_csv": "data/processed/full_train_pool.csv",
    "probe_text_col": "Text",
    "probe_label_col": "label",
    "probe_group_col": "group_id",

    # Final evaluation file. This top-level file is not the processed split
    # that is included in full_train_pool.csv by the current stage2 config.
    "eval_csv": "data/test.csv",
    "eval_text_col": "Text",
    "eval_label_col": "generated",

    "batch_size": cfg["batch_size"],
    "random_state": 42,
    "drop_exact_text_overlaps": True,
}


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else BASE_DIR / path


def load_csv(path: str | Path, text_col: str, label_col: str) -> pd.DataFrame:
    csv_path = resolve_path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, engine="python", on_bad_lines="warn")
    missing = [col for col in (text_col, label_col) if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} in {csv_path}. Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    df[label_col] = df[label_col].astype(int)

    bad_labels = sorted(set(df[label_col].unique()) - {0, 1, 2})
    if bad_labels:
        raise ValueError(
            f"Unexpected labels in {csv_path}: {bad_labels}. Expected 0, 1, or 2."
        )

    return df


def normalized_texts(df: pd.DataFrame, text_col: str) -> pd.Series:
    return df[text_col].astype(str).str.normalize("NFKC").str.strip().str.lower()


def remove_eval_overlaps(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_text_col: str,
    eval_text_col: str,
) -> pd.DataFrame:
    train_texts = set(normalized_texts(train_df, train_text_col))
    eval_norm = normalized_texts(eval_df, eval_text_col)
    overlap_mask = eval_norm.isin(train_texts)
    overlap_count = int(overlap_mask.sum())

    if overlap_count:
        log.warning(
            "Found %d exact normalized text overlaps between probe train and eval.",
            overlap_count,
        )
        eval_df = eval_df.loc[~overlap_mask].reset_index(drop=True)
        log.warning("Dropped overlap rows from eval. Remaining eval rows: %d", len(eval_df))

    return eval_df


def add_dummy_group_id(df: pd.DataFrame) -> pd.DataFrame:
    if "_group_id" not in df.columns:
        df = df.copy()
        df["_group_id"] = np.arange(len(df), dtype=np.int64)
    return df


def build_loader(
    df: pd.DataFrame,
    tokenizer: RobertaTokenizer,
    text_col: str,
    label_col: str,
    group_col: str,
) -> DataLoader:
    dataset = ContrastiveEssayDataset(
        df,
        tokenizer,
        cfg["max_length"],
        text_col,
        label_col,
        group_id_col=group_col,
    )
    return DataLoader(
        dataset,
        batch_size=EVAL_CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,
    )


@torch.no_grad()
def extract_embeddings(model: RoBERTaContrastive, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    all_pooled, all_labels = [], []

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        pooled, _ = model(input_ids, attn_mask)
        all_pooled.append(pooled.cpu().numpy())
        all_labels.append(batch["label"].numpy())

        if i % 100 == 0:
            log.info("  Processed %d/%d samples", i * EVAL_CONFIG["batch_size"], len(loader.dataset))

    return np.concatenate(all_pooled), np.concatenate(all_labels)


def load_tokenizer() -> RobertaTokenizer:
    tokenizer_path = resolve_path(EVAL_CONFIG["tokenizer_dir"])
    if tokenizer_path.exists():
        log.info("Tokenizer loaded from: %s", tokenizer_path)
        return RobertaTokenizer.from_pretrained(str(tokenizer_path))

    log.info("Tokenizer loaded from HuggingFace: %s", cfg["model_name"])
    return RobertaTokenizer.from_pretrained(cfg["model_name"])


def load_model() -> RoBERTaContrastive:
    ckpt_path = resolve_path(EVAL_CONFIG["checkpoint"])
    if not ckpt_path.exists():
        fallback = resolve_path(EVAL_CONFIG["fallback_checkpoint"])
        if fallback.exists():
            log.warning("Checkpoint not found at %s; using %s", ckpt_path, fallback)
            ckpt_path = fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt_size_mb = os.path.getsize(ckpt_path) / 1e6
    log.info("Checkpoint path: %s", ckpt_path)
    log.info("Checkpoint size: %.1f MB", ckpt_size_mb)

    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state_dict = state["model_state"]
        log.info(
            "Loaded checkpoint wrapper: epoch=%s val_loss=%s",
            state.get("epoch", "unknown"),
            state.get("val_loss", "unknown"),
        )
    else:
        state_dict = state

    log.info("Total model keys in checkpoint: %d", len(state_dict))

    model = RoBERTaContrastive(
        model_name=cfg["model_name"],
        proj_hidden=cfg["proj_hidden"],
        proj_out=cfg["proj_out"],
        grad_checkpointing=False,
    ).to(device)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "The checkpoint did not match RoBERTaContrastive. If this is a full "
            "training checkpoint, it must contain a 'model_state' entry."
        ) from exc

    log.info("Model loaded cleanly; all keys matched.")

    model.eval()
    return model


def report_distribution(name: str, labels: np.ndarray) -> None:
    log.info("%s label counts: human=%d original_ai=%d paraphrased_ai=%d total_ai=%d",
             name, int((labels == 0).sum()), int((labels == 1).sum()),
             int((labels == 2).sum()), int((labels > 0).sum()))


def main() -> None:
    log.info("Device: %s", device)

    train_df = load_csv(
        EVAL_CONFIG["probe_train_csv"],
        EVAL_CONFIG["probe_text_col"],
        EVAL_CONFIG["probe_label_col"],
    )
    eval_df = load_csv(
        EVAL_CONFIG["eval_csv"],
        EVAL_CONFIG["eval_text_col"],
        EVAL_CONFIG["eval_label_col"],
    )

    if EVAL_CONFIG["drop_exact_text_overlaps"]:
        eval_df = remove_eval_overlaps(
            train_df,
            eval_df,
            EVAL_CONFIG["probe_text_col"],
            EVAL_CONFIG["eval_text_col"],
        )

    eval_df = add_dummy_group_id(eval_df)

    log.info("Probe train rows: %d from %s", len(train_df), EVAL_CONFIG["probe_train_csv"])
    log.info("Eval rows: %d from %s", len(eval_df), EVAL_CONFIG["eval_csv"])
    log.info("Probe train label dist:\n%s",
             train_df[EVAL_CONFIG["probe_label_col"]].value_counts().sort_index().to_string())
    log.info("Eval label dist:\n%s",
             eval_df[EVAL_CONFIG["eval_label_col"]].value_counts().sort_index().to_string())

    tokenizer = load_tokenizer()
    model = load_model()

    train_loader = build_loader(
        train_df,
        tokenizer,
        EVAL_CONFIG["probe_text_col"],
        EVAL_CONFIG["probe_label_col"],
        EVAL_CONFIG["probe_group_col"],
    )
    eval_loader = build_loader(
        eval_df,
        tokenizer,
        EVAL_CONFIG["eval_text_col"],
        EVAL_CONFIG["eval_label_col"],
        "_group_id",
    )

    log.info("Extracting probe-train embeddings...")
    X_train, y_train_3class = extract_embeddings(model, train_loader)
    log.info("Extracting held-out eval embeddings...")
    X_eval, y_eval_3class = extract_embeddings(model, eval_loader)

    y_train = (y_train_3class > 0).astype(int)
    y_eval = (y_eval_3class > 0).astype(int)

    report_distribution("Probe train", y_train_3class)
    report_distribution("Held-out eval", y_eval_3class)
    log.info("Train embedding shape: %s", X_train.shape)
    log.info("Eval embedding shape: %s", X_eval.shape)

    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=EVAL_CONFIG["random_state"],
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_eval)
    probs = clf.predict_proba(X_eval)[:, 1]
    auc = roc_auc_score(y_eval, probs)
    report = classification_report(y_eval, preds, target_names=["Human", "AI"], digits=4)
    cm = confusion_matrix(y_eval, preds, labels=[0, 1])

    print("\n" + "=" * 64)
    print("HELD-OUT EVALUATION")
    print("=" * 64)
    print(f"Probe trained on : {EVAL_CONFIG['probe_train_csv']} ({len(train_df)} rows)")
    print(f"Evaluated on     : {EVAL_CONFIG['eval_csv']} ({len(eval_df)} rows)")
    print("\nBinary report (0=Human, 1=AI):")
    print(report)
    print(f"AUC: {auc:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    for label_value, label_name in [(1, "Original AI"), (2, "Paraphrased AI")]:
        mask = y_eval_3class == label_value
        if mask.sum() == 0:
            log.warning("No %s samples found in eval set.", label_name)
            continue
        recall = (preds[mask] == 1).mean()
        print(f"{label_name:14s} detection recall: {recall:.4f} ({int(mask.sum())} rows)")

    print("=" * 64)
    print("Note: this is now a held-out probe evaluation, not a 70/30 split of the test file.")


if __name__ == "__main__":
    main()
