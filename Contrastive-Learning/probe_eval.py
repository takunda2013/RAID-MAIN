"""
Probe Evaluation Script
========================
Evaluates the fine-tuned RoBERTa contrastive model by:
1. Verifying the checkpoint is valid and loaded correctly
2. Extracting CLS embeddings from the val set
3. Training a logistic regression probe on 70% of embeddings
4. Evaluating on the held-out 30% — no data leakage
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer

# ── Import from stage2 ────────────────────────────────────────
from stage2_roberta_finetune import (
    RoBERTaContrastive,
    ContrastiveEssayDataset,
    TRAIN_CONFIG,
)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

cfg    = TRAIN_CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device}")

# ── Step 1: Verify checkpoint ─────────────────────────────────
ckpt_path = "models/roberta_contrastive/best_model.pt"

log.info("\n--- Checkpoint Verification ---")
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

ckpt_size_mb = os.path.getsize(ckpt_path) / 1e6
log.info(f"Checkpoint path  : {ckpt_path}")
log.info(f"Checkpoint size  : {ckpt_size_mb:.1f} MB")

if ckpt_size_mb < 100:
    log.warning("Checkpoint is unexpectedly small — may not be a full fine-tuned model")

state = torch.load(ckpt_path, map_location=device)
log.info(f"Total keys in checkpoint : {len(state)}")
log.info(f"Sample keys              : {list(state.keys())[:5]}")

has_projection = any("projection" in k for k in state.keys())
has_pooler     = any("pooler" in k for k in state.keys())
log.info(f"Has projection head keys : {has_projection}")
log.info(f"Has pooler keys          : {has_pooler}")

if not has_projection:
    log.warning("No projection head found in checkpoint — model may not have loaded correctly")

# ── Step 2: Load tokenizer and val data ───────────────────────
log.info("\n--- Loading Val Data ---")
tokenizer_path = "models/roberta_contrastive/tokenizer"
if os.path.exists(tokenizer_path):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    log.info(f"Tokenizer loaded from: {tokenizer_path}")
else:
    tokenizer = RobertaTokenizer.from_pretrained(cfg["model_name"])
    log.info(f"Tokenizer loaded from HuggingFace: {cfg['model_name']}")

val_df = pd.read_csv(cfg["val_csv"], engine="python", on_bad_lines="warn")
log.info(f"Val rows: {len(val_df)}")
log.info(f"Val label dist:\n{val_df['label'].value_counts().to_string()}")

val_ds = ContrastiveEssayDataset(
    val_df, tokenizer, cfg["max_length"],
    cfg["text_col"], cfg["label_col"], cfg["group_id_col"]
)
val_loader = DataLoader(
    val_ds, batch_size=cfg["batch_size"],
    shuffle=False, num_workers=0
)

# ── Step 3: Load model with fine-tuned weights ─────────────────
log.info("\n--- Loading Model ---")
model = RoBERTaContrastive(
    model_name=cfg["model_name"],
    proj_hidden=cfg["proj_hidden"],
    proj_out=cfg["proj_out"],
    grad_checkpointing=False,   # not needed for inference
).to(device)

# Load fine-tuned weights — strict=False to surface missing/unexpected keys clearly
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    log.warning(f"Missing keys after load  : {missing}")
if unexpected:
    log.warning(f"Unexpected keys after load: {unexpected}")

if not missing and not unexpected:
    log.info("Model loaded cleanly — all keys matched")

model.eval()

# ── Step 4: Extract embeddings ────────────────────────────────
log.info("\n--- Extracting Embeddings ---")
all_pooled, all_labels = [], []

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        pooled, _ = model(input_ids, attn_mask)   # (B, 768) CLS embedding
        all_pooled.append(pooled.cpu().numpy())
        all_labels.append(batch["label"].numpy())

        if i % 50 == 0:
            log.info(f"  Processed {i * cfg['batch_size']}/{len(val_ds)} samples")

X = np.concatenate(all_pooled)   # (N, 768)
y = np.concatenate(all_labels)   # (N,)

# Collapse to binary: 0=Human, 1=AI (label 1 + label 2)
y_binary = (y > 0).astype(int)
log.info(f"Embedding shape : {X.shape}")
log.info(f"Human samples   : {(y_binary == 0).sum()}")
log.info(f"AI samples      : {(y_binary == 1).sum()}")

# ── Step 5: Train/test split — no leakage ─────────────────────
log.info("\n--- Probe Evaluation (70/30 split, no leakage) ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary,
    test_size=0.3,
    random_state=42,
    stratify=y_binary   # preserve class balance in both splits
)
log.info(f"Probe train size : {len(X_train)}")
log.info(f"Probe test size  : {len(X_test)}")

clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:, 1]

# ── Step 6: Results ───────────────────────────────────────────
log.info("\n--- Results ---")
report = classification_report(y_test, preds, target_names=["Human", "AI"])
auc    = roc_auc_score(y_test, probs)

print("\n" + "="*50)
print("LINEAR PROBE RESULTS (held-out 30% of val set)")
print("="*50)
print(report)
print(f"AUC: {auc:.4f}")
print("="*50)

# ── Step 7: Per-class AI breakdown (orig vs paraphrased) ──────
log.info("\n--- AI Subclass Breakdown (Original vs Paraphrased) ---")
# Use the full val set for this analysis
_, X_test_full, _, y_test_full_3class = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Get binary predictions on the 3-class test split
preds_full = clf.predict(X_test_full)
y_binary_full = (y_test_full_3class > 0).astype(int)

# Original AI (label=1) accuracy
orig_mask = y_test_full_3class == 1
if orig_mask.sum() > 0:
    orig_acc = (preds_full[orig_mask] == y_binary_full[orig_mask]).mean()
    print(f"Original AI (label=1) detection accuracy : {orig_acc:.4f}")

# Paraphrased AI (label=2) accuracy
para_mask = y_test_full_3class == 2
if para_mask.sum() > 0:
    para_acc = (preds_full[para_mask] == y_binary_full[para_mask]).mean()
    print(f"Paraphrased AI (label=2) detection accuracy: {para_acc:.4f}")

print("\nNote: A gap between original and paraphrased accuracy")
print("indicates the model is still fooled by paraphrasing.")
