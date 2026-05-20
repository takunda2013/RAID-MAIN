"""
STAGE 2: RoBERTa Contrastive Fine-Tuning
=========================================
Trains RoBERTa with Supervised Contrastive Loss.

Goal: AI originals and their paraphrases cluster together in embedding
space; human essays are pushed into a separate region.

Validation: uses your pre-built val.csv (output of stage 1) which
contains a mix of human, AI original, and paraphrased essays.
Val loss is computed with the same SupConLoss used for training.

Optimised for 24 GB VRAM:
  - Gradient checkpointing  (~30% VRAM saving)
  - Mixed precision bf16/fp16
  - Default batch=16, effective batch=32 via gradient accumulation
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from collections import defaultdict

from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

# ── FIX 3: Only attach handler in main process to prevent duplicate logs ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Suppress duplicate logs from DataLoader worker processes
logging.getLogger("__main__").propagate = False
if os.environ.get("WORKER_PROCESS"):
    logging.disable(logging.INFO)


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    # Data (output of stage 1)
    "train_csv":    "data/processed/full_train_pool.csv",
    "val_csv":      "data/processed/val.csv",     # your pre-built validation set
    "text_col":     "Text",
    "label_col":    "label",
    "group_id_col": "group_id",

    # Model
    "model_name":  "roberta-large",   # swap to roberta-large if you want (needs more VRAM)
    "proj_hidden":  256,
    "proj_out":     128,
    "max_length":   512,

    # Training
    "epochs":            5,
    "batch_size":        16,    # 16 fits in ~12 GB with checkpointing; push to 24 if needed
    "grad_accumulation": 2,     # effective batch = 32
    "lr":                2e-5,
    "weight_decay":      0.01,
    "warmup_ratio":      0.06,
    "temperature":       0.07,  # SupCon temperature — lower = harder negatives
    "max_grad_norm":     1.0,

    # contrastive_mode:
    #   "group"  → AI original + all its paraphrases = same positive class
    #              (best for paraphrase-invariant detection)
    #   "label"  → label=1 (orig) and label=2 (para) are different classes
    #              (also forces model to distinguish orig from para)
    "contrastive_mode": "label",

    # Hardware
    "use_amp":            True,
    "grad_checkpointing": True,
    "num_workers":        4,

    # Output
    "output_dir":          "models/roberta_contrastive",
    "save_every_n_epochs": 1,

    "patience": 2,   # add to TRAIN_CONFIG

}


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class ContrastiveEssayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int,
                 text_col="text", label_col="label", group_id_col="group_id"):
        self.texts     = df[text_col].tolist()
        self.labels    = df[label_col].tolist()
        self.group_ids = df[group_id_col].tolist()
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
            "label":    torch.tensor(self.labels[idx],    dtype=torch.long),
            "group_id": torch.tensor(self.group_ids[idx], dtype=torch.long),
        }


class GroupSampler(torch.utils.data.Sampler):
    """
    Builds batches that guarantee positive pairs (same group_id) land
    together. SupConLoss needs at least one positive pair per batch
    or it produces zero useful gradient.

    Strategy: fill the first half of each batch with members from
    multi-member groups, pad the rest with random samples.

    FIX: Index list is pre-built in __init__ with a fixed seed so that
    __len__ is always accurate and matches what __iter__ actually yields.
    This prevents the step counter overflow seen when len() was computed
    from raw dataset size instead of actual sampler output size.
    """

    def __init__(self, df: pd.DataFrame, batch_size: int,
                 group_id_col="group_id", min_groups_per_batch=6, seed=42):
        self.batch_size  = batch_size
        self.min_groups  = min_groups_per_batch
        self.seed        = seed
        self.all_indices = list(range(len(df)))

        group_map = defaultdict(list)
        for idx, gid in enumerate(df[group_id_col]):
            group_map[gid].append(idx)

        self.multi_groups = {k: v for k, v in group_map.items() if len(v) > 1}
        log.info(f"GroupSampler: {len(self.multi_groups)} multi-member groups "
                 f"out of {len(group_map)} total")

        # ── FIX 1: Pre-build indices so __len__ is always accurate ──
        self._indices = self._build_indices()
        log.info(f"GroupSampler: effective batches per epoch = {len(self)}")

    def _build_indices(self):
        rng = np.random.default_rng(self.seed)

        # Separate indices into two pools:
        # - multi_indices: belong to groups with >1 member (can form positive pairs)
        # - solo_indices:  singletons / no positive partner
        multi_indices = []
        solo_indices  = []
        multi_set     = set()
        for members in self.multi_groups.values():
            for idx in members:
                multi_set.add(idx)

        for idx in self.all_indices:
            if idx in multi_set:
                multi_indices.append(idx)
            else:
                solo_indices.append(idx)

        rng.shuffle(multi_indices)
        rng.shuffle(solo_indices)

        # Build batches so that every batch starts with a guaranteed
        # positive pair (2 members from the same group), then is padded
        # with solo/remaining samples to fill batch_size.
        # No index is overwritten — each index appears exactly once.
        multi_keys = list(self.multi_groups.keys())
        rng.shuffle(multi_keys)

        # Pool of filler samples (solo first, then remaining multi)
        filler = solo_indices.copy()

        batches = []
        used_multi = set()

        for gk in multi_keys:
            members = [m for m in self.multi_groups[gk] if m not in used_multi]
            if len(members) < 2:
                # Not enough unused members — put them in filler
                filler.extend(members)
                continue

            # Pick exactly 2 members as the positive anchor pair
            pair = members[:2]
            remaining = members[2:]  # extras go to filler
            filler.extend(remaining)
            used_multi.update(pair)

            # Pad the batch with filler up to batch_size
            pad_needed = self.batch_size - len(pair)
            pad = filler[:pad_needed]
            filler = filler[pad_needed:]

            if len(pad) < pad_needed:
                # Not enough filler — batch is smaller, still valid for SupCon
                pass

            batches.append(pair + pad)

        # Any leftover filler indices that weren't used: form plain batches
        for i in range(0, len(filler), self.batch_size):
            batches.append(filler[i:i + self.batch_size])

        # Shuffle batch order
        rng.shuffle(batches)

        # Flatten — every original index appears exactly once
        result = []
        for batch in batches:
            result.extend(batch)
        return result

    def __iter__(self):
        # Re-build with a new random seed each epoch so pair assignments vary
        epoch_seed = int(torch.randint(0, 2**31, (1,)).item())
        self.seed  = epoch_seed
        indices    = self._build_indices()
        # Restore stable seed so __len__ stays consistent
        self.seed  = 42

        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        for batch in batches:
            yield from batch

    def __len__(self):
        # Accurate: matches what __iter__ actually yields
        return len(self._indices)


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
class RoBERTaContrastive(nn.Module):
    """
    RoBERTa backbone + projection head.

    forward() returns:
      pooled   — (B, 768) CLS embedding  [used for feature extraction in stage 3]
      projected — (B, 128) L2-normalised  [used for SupConLoss]
    """

    def __init__(self, model_name="roberta-base", proj_hidden=256, proj_out=128,
                 grad_checkpointing=True):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)

        if grad_checkpointing:
            self.roberta.gradient_checkpointing_enable()
            log.info("Gradient checkpointing enabled")

        hidden = self.roberta.config.hidden_size   # 768 for base, 1024 for large
        self.projection = nn.Sequential(
            nn.Linear(hidden, proj_hidden),
            nn.GELU(),
            nn.LayerNorm(proj_hidden),
            nn.Dropout(0.1),
            nn.Linear(proj_hidden, proj_out),
        )

    def forward(self, input_ids, attention_mask):
        out    = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output                           # (B, hidden)
        proj   = F.normalize(self.projection(pooled), dim=-1)  # (B, proj_out)
        return pooled, proj


# ─────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────
class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features : (N, D) — L2-normalised embeddings
        labels   : (N,)   — integer class labels (used to find positive pairs)
        """
        device = features.device
        N = features.shape[0]

        labels = labels.view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)

        # Remove self-comparisons
        self_mask = 1 - torch.eye(N, device=device)
        pos_mask  = pos_mask * self_mask

        # Scaled cosine similarity
        sim = torch.matmul(features, features.T) / self.temperature
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()   # numerical stability

        exp_sim  = torch.exp(sim) * self_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)

        pos_count = pos_mask.sum(1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_count + 1e-9)

        # Only include anchors that have at least one positive in the batch
        has_positive = pos_count > 0
        if has_positive.sum() == 0:
            # Entire batch has no positive pairs — return zero loss with grad
            return (features * 0).sum()

        return -mean_log_prob_pos[has_positive].mean()


# ─────────────────────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────
def get_contrast_labels(batch, mode: str, device):
    """
    Returns the tensor SupConLoss will use to identify positive pairs.
      mode='group'  → same group_id = positive (orig + its paraphrases cluster together)
      mode='label'  → same label    = positive (distinguishes orig from para too)
    """
    if mode == "group":
        return batch["group_id"].to(device)
    return batch["label"].to(device)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion,
                    device, cfg, epoch):
    model.train()
    total_loss, n = 0.0, 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        c_labels  = get_contrast_labels(batch, cfg["contrastive_mode"], device)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with autocast('cuda', dtype=amp_dtype, enabled=cfg["use_amp"]):
            _, projected = model(input_ids, attn_mask)
            loss = criterion(projected, c_labels) / cfg["grad_accumulation"]

        scaler.scale(loss).backward()

        if (step + 1) % cfg["grad_accumulation"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * cfg["grad_accumulation"]
        n += 1

        if step % 50 == 0:
            opt_step = step // cfg["grad_accumulation"]
            log.info(f"  Epoch {epoch} | batch {step}/{len(loader)} | opt_step ~{opt_step} | "
                     f"loss={total_loss/n:.4f} | "
                     f"lr={scheduler.get_last_lr()[0]:.2e}")

    # ── Flush orphaned gradients from last batch if epoch size is odd ──
    if (len(loader)) % cfg["grad_accumulation"] != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg):
    """
    Validation loss on the pre-built val set.
    Uses label-based contrastive loss (not group_id, since val rows have
    dummy group_ids). Same-label pairs are still meaningful positives.
    """
    model.eval()
    total_loss, n = 0.0, 0

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["label"].to(device)   # always use label for val

        # ── FIX 2: Added missing 'cuda' device arg to autocast ──
        with autocast('cuda', dtype=amp_dtype, enabled=cfg["use_amp"]):
            _, projected = model(input_ids, attn_mask)
            loss = criterion(projected, labels)

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def train_roberta(cfg: dict = TRAIN_CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU : {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ────────────────────────────────────────────
    tokenizer = RobertaTokenizer.from_pretrained(cfg["model_name"])

    # ── Data ─────────────────────────────────────────────────
    train_df = pd.read_csv(cfg["train_csv"], engine='python', on_bad_lines='warn')
    val_df   = pd.read_csv(cfg["val_csv"],   engine='python', on_bad_lines='warn')
    log.info(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows")
    log.info(f"Train label dist:\n{train_df['label'].value_counts().to_string()}")
    log.info(f"Val   label dist:\n{val_df  ['label'].value_counts().to_string()}")

    train_ds = ContrastiveEssayDataset(train_df, tokenizer, cfg["max_length"],
                                       cfg["text_col"], cfg["label_col"], cfg["group_id_col"])
    val_ds   = ContrastiveEssayDataset(val_df,   tokenizer, cfg["max_length"],
                                       cfg["text_col"], cfg["label_col"], cfg["group_id_col"])

    train_sampler = GroupSampler(train_df, cfg["batch_size"], cfg["group_id_col"])

    import multiprocessing
    def worker_init_fn(worker_id):
        os.environ["WORKER_PROCESS"] = "1"

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              sampler=train_sampler,
                              num_workers=cfg["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False,
                              num_workers=cfg["num_workers"],
                              pin_memory=True)

    # ── Model ─────────────────────────────────────────────────
    model = RoBERTaContrastive(
        model_name=cfg["model_name"],
        proj_hidden=cfg["proj_hidden"],
        proj_out=cfg["proj_out"],
        grad_checkpointing=cfg["grad_checkpointing"],
    ).to(device)

    # ── Optimiser ─────────────────────────────────────────────
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters()
                    if     any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(params, lr=cfg["lr"])

    # total_steps now uses accurate len(train_loader) after GroupSampler fix
    total_steps  = (len(train_loader) // cfg["grad_accumulation"]) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = SupConLoss(temperature=cfg["temperature"])
    scaler    = GradScaler('cuda', enabled=cfg["use_amp"])

    log.info(f"\nContrastive mode : {cfg['contrastive_mode']}")
    log.info(f"Total steps      : {total_steps} | Warmup: {warmup_steps}")

    # ── Training loop ─────────────────────────────────────────
    history  = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience = cfg["patience"]
    no_improve  = 0   # ← add this

    for epoch in range(1, cfg["epochs"] + 1):
        log.info(f"\n{'='*55}\nEpoch {epoch}/{cfg['epochs']}\n{'='*55}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                     scaler, criterion, device, cfg, epoch)
        val_loss   = evaluate(model, val_loader, criterion, device, cfg)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        log.info(f"\nEpoch {epoch} summary → "
                 f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # Checkpoint
        if epoch % cfg["save_every_n_epochs"] == 0:
            ckpt = out_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    val_loss,
                "config":      cfg,
            }, ckpt)
            log.info(f"Checkpoint saved → {ckpt}")

        # Best model + early stopping
        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            log.info(f"New best model saved (val_loss={best_val:.4f})")
        else:
            no_improve += 1
            log.info(f"No improvement for {no_improve}/{patience} epochs")
            if no_improve >= patience:
                log.info(f"Early stopping triggered at epoch {epoch}")
                break  # ← exits the epoch loop cleanly

    # ── Save tokenizer + history ──────────────────────────────
    tokenizer.save_pretrained(out_dir / "tokenizer")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info(f"\nDone. Best val_loss: {best_val:.4f}")
    log.info(f"Artifacts saved to: {out_dir}/")
    return model, tokenizer, history


if __name__ == "__main__":
    train_roberta()
