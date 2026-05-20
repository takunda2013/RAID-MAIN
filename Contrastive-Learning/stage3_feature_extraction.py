"""
STAGE 3: Semantic Feature Extraction
======================================
Loads the fine-tuned RoBERTa and extracts 768-dim pooled embeddings
for every essay in the validation set and top-level test set.

These embeddings are saved as .npy files and later fused with the
10-dim GPT-2 statistical features in Stage 4.

Optimised for 24 GB VRAM: processes in batches with no gradient bookkeeping.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

import numpy as np
import pandas as pd
from pathlib import Path
import logging

from transformers import RobertaTokenizer

# Import the model class from stage 2
from stage2_roberta_finetune import RoBERTaContrastive, TRAIN_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else BASE_DIR / path


# ─────────────────────────────────────────────────────────────
# CONFIG  (inherits paths from stage 2 config)
# ─────────────────────────────────────────────────────────────
EXTRACT_CONFIG = {
    # Paths
    "model_weights":  "models/roberta_contrastive/best_model.pt",
    "tokenizer_dir":  "models/roberta_contrastive/tokenizer",
    "model_name":     TRAIN_CONFIG["model_name"],   # fallback if tokenizer_dir missing

    "splits": {
        "unseen": "data/unseen.csv",
        "val":  "data/processed/val.csv",
    },
    "text_col":   "Text",
    "label_cols": ("label", "generated"),

    "output_dir": "data/embeddings",

    "batch_size":  64,    # can push higher since no grad; 64 is comfortable at 24 GB
    "max_length":  512,
    "num_workers": 4,
}


# ─────────────────────────────────────────────────────────────
# MINIMAL DATASET (text only, no labels needed)
# ─────────────────────────────────────────────────────────────
class TextOnlyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts     = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ─────────────────────────────────────────────────────────────
# EXTRACTION FUNCTION
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model, loader, device) -> np.ndarray:
    """
    Runs all batches through the RoBERTa encoder and returns
    a single (N, 768) float32 numpy array of pooled outputs.
    """
    model.eval()
    all_embeddings = []
    total = len(loader)

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        with autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                      enabled=True):
            pooled, _ = model(input_ids, attn_mask)  # (B, 768)

        all_embeddings.append(pooled.float().cpu().numpy())

        if step % 20 == 0:
            log.info(f"  Batch {step+1}/{total}")

    return np.vstack(all_embeddings)   # (N, 768)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def run_extraction(cfg: dict = EXTRACT_CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    out_dir = resolve_path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer ────────────────────────────────────────
    tokenizer_path = resolve_path(cfg["tokenizer_dir"])
    if tokenizer_path.exists():
        tokenizer = RobertaTokenizer.from_pretrained(str(tokenizer_path))
        log.info(f"Tokenizer loaded from: {tokenizer_path}")
    else:
        log.warning(f"Tokenizer dir not found, falling back to: {cfg['model_name']}")
        tokenizer = RobertaTokenizer.from_pretrained(cfg["model_name"])

    # ── Load model ────────────────────────────────────────────
    model = RoBERTaContrastive(
        model_name=cfg["model_name"],
        grad_checkpointing=False,   # not needed for inference
    ).to(device)

    weights_path = resolve_path(cfg["model_weights"])
    assert weights_path.exists(), (
        f"Model weights not found at {weights_path}. "
        f"Run stage2_roberta_finetune.py first."
    )
    state_dict = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state_dict)
    log.info(f"Model weights loaded from: {weights_path}")

    # ── Process configured splits only ────────────────────────
    for split_name, csv_path in cfg["splits"].items():
        log.info(f"\nExtracting embeddings for: {split_name}")
        csv_path = resolve_path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"{split_name} CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, engine='python', on_bad_lines='warn')
        if cfg["text_col"] not in df.columns:
            raise KeyError(
                f"Text column '{cfg['text_col']}' not found in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )
        texts = df[cfg["text_col"]].tolist()

        dataset = TextOnlyDataset(texts, tokenizer, cfg["max_length"])
        loader  = DataLoader(dataset, batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=cfg["num_workers"],
                             pin_memory=True)

        embeddings = extract_embeddings(model, loader, device)
        assert len(embeddings) == len(df), (
            f"Mismatch: {len(embeddings)} embeddings for {len(df)} rows"
        )

        # Save embeddings
        emb_path = out_dir / f"{split_name}_semantic_embeddings.npy"
        np.save(str(emb_path), embeddings)
        log.info(f"  Saved {embeddings.shape} embeddings → {emb_path}")

        # Save labels alongside when a label-like column exists.
        label_col = next((col for col in cfg["label_cols"] if col in df.columns), None)
        if label_col is not None:
            labels_path = out_dir / f"{split_name}_labels.npy"
            np.save(str(labels_path), df[label_col].values)
            log.info(f"  Labels saved from column '{label_col}' → {labels_path}")
        else:
            log.info("  No label column found; skipped labels.")

        # Save group_ids when present. data/test.csv does not have them.
        if "group_id" in df.columns:
            gids_path = out_dir / f"{split_name}_group_ids.npy"
            np.save(str(gids_path), df["group_id"].values)
            log.info(f"  Group ids saved → {gids_path}")
        else:
            log.info("  No group_id column found; skipped group ids.")

    log.info(f"\nExtraction complete. All .npy files in: {out_dir}/")
    log.info("Next: run stage4_feature_fusion.py to combine with GPT-2 statistical features.")


if __name__ == "__main__":
    run_extraction()
