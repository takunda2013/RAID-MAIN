"""
STAGE 1: Data Preparation & Grouping
=====================================
Handles your actual file structure:
  - human_essays.csv          → label=0
  - ai_original.csv           → label=1
  - paraphrased.csv           → label=2  (all paraphrasers in one file,
                                           'model' column = gpt_oss / llama / deepseek)
  - validation.csv            → pre-built val set with 'generated' column
                                 (0=human, 1=ai, 2=paraphrased)

Group IDs for Supervised Contrastive Learning:
  - AI original row i  →  group_id = i
  - Paraphrased row    →  group_id = source_index  (links back to ai_original row)
  - Human row          →  unique group_id  (no cluster, just an anchor)

Outputs (all to data/processed/):
  train.csv, test.csv          — 90/10 split of main data, split at group level
  val.csv                      — your pre-built validation set (cleaned)
  full_train_pool.csv          — combined main data before split (for reference)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# CONFIG  — edit paths and column names to match your files
# ─────────────────────────────────────────────────────────────
DATA_CONFIG = {
    # Input files
    "human_csv":       "data/raw/human_essays.csv",
    "ai_orig_csv":     "data/raw/ai_original.csv",
    "paraphrased_csv": "data/raw/paraphrased.csv",   # single file, all models
    "validation_csv":  "data/raw/validation.csv",    # your pre-built val set

    # Column names
    "text_col":         "Text",          # essay text column (same in all files)
    "model_col":        "model",         # in paraphrased.csv: gpt_oss / llama / deepseek
    "source_index_col": "source_index",  # in paraphrased.csv: row index into ai_original
    "val_label_col":    "generated",     # in validation.csv: 0=human, 1=ai, 2=para

    # Output
    "output_dir":  "data/processed",
    "test_size":   0.10,   # 10% of main data held out as test set
    "random_seed": 42,
}

# Known paraphraser model names in your 'model' column — adjust if yours differ
PARAPHRASER_NAMES = {"gpt-oss-120b", "llama3.1-8b", "deepseek r1"}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def load_and_clean(path: str, label_name: str, text_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert text_col in df.columns, (
        f"[{label_name}] Column '{text_col}' not found. "
        f"Columns present: {list(df.columns)}"
    )
    before = len(df)
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 20].reset_index(drop=True)
    log.info(f"[{label_name}] {before} rows → {len(df)} after cleaning")
    return df


def group_split(df: pd.DataFrame, test_size: float, seed: int):
    """Split at the group level so no group straddles train and test."""
    unique_groups = df["group_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    n_test  = max(1, int(len(unique_groups) * test_size))
    test_g  = set(unique_groups[:n_test])
    train_g = set(unique_groups[n_test:])

    train_df = df[df["group_id"].isin(train_g)].reset_index(drop=True)
    test_df  = df[df["group_id"].isin(test_g) ].reset_index(drop=True)
    return train_df, test_df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def prepare_data(cfg: dict = DATA_CONFIG) -> dict:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tc = cfg["text_col"]
    mc = cfg["model_col"]
    sc = cfg["source_index_col"]

    # ── 1. Load ───────────────────────────────────────────────
    df_human   = load_and_clean(cfg["human_csv"],       "Human",       tc)
    df_ai_orig = load_and_clean(cfg["ai_orig_csv"],     "AI-Original", tc)
    df_para    = load_and_clean(cfg["paraphrased_csv"], "Paraphrased", tc)
    df_val_raw = load_and_clean(cfg["validation_csv"],  "Validation",  tc)

    # ── 2. Validate paraphrased file columns ──────────────────
    for col in [mc, sc]:
        assert col in df_para.columns, (
            f"Column '{col}' not found in paraphrased CSV. "
            f"Columns present: {list(df_para.columns)}"
        )

    log.info(f"\nParaphraser model distribution:\n{df_para[mc].value_counts().to_string()}")

    unexpected = set(df_para[mc].unique()) - PARAPHRASER_NAMES
    if unexpected:
        log.warning(f"Unexpected model names found: {unexpected}. "
                    f"They will still be included as label=2.")

    # ── 3. Validate source_index integrity ────────────────────
    df_ai_orig = df_ai_orig.reset_index(drop=True)
    valid_ids  = set(df_ai_orig.index)

    bad_ids = set(df_para[sc].unique()) - valid_ids
    if bad_ids:
        log.warning(
            f"{len(bad_ids)} source_index values have no match in ai_original "
            f"(e.g. {sorted(bad_ids)[:5]}). Dropping those rows."
        )
        df_para = df_para[df_para[sc].isin(valid_ids)].reset_index(drop=True)

    deepseek_covered = set(df_para[df_para[mc] == "deepseek"][sc].unique())
    log.info(f"\nDeepSeek coverage: {len(deepseek_covered)}/{len(df_ai_orig)} "
             f"AI originals ({100*len(deepseek_covered)/len(df_ai_orig):.1f}%)")

    # ── 4. Assign labels ──────────────────────────────────────
    df_human  ["label"] = 0
    df_ai_orig["label"] = 1
    df_para   ["label"] = 2

    # ── 5. Assign group_ids ───────────────────────────────────
    df_ai_orig["group_id"] = df_ai_orig.index                      # 0 … N_ai-1
    df_para   ["group_id"] = df_para[sc].values                    # links to ai_orig row

    human_id_start = len(df_ai_orig)
    df_human = df_human.reset_index(drop=True)
    df_human["group_id"] = np.arange(human_id_start, human_id_start + len(df_human))

    # ── 6. Source tag ─────────────────────────────────────────
    df_human  ["source"] = "human"
    df_ai_orig["source"] = "ai_original"
    df_para   ["source"] = "para_" + df_para[mc].astype(str)

    # ── 7. Combine main pool ──────────────────────────────────
    keep = [tc, "label", "group_id", "source"]
    main_df = pd.concat(
        [df_human[keep], df_ai_orig[keep], df_para[keep]],
        ignore_index=True
    ).sample(frac=1, random_state=cfg["random_seed"]).reset_index(drop=True)

    # ── 8. Train / Test split (group-level, 90/10) ────────────
    train_df, test_df = group_split(main_df, cfg["test_size"], cfg["random_seed"])

    # ── 9. Prepare the pre-built validation set ───────────────
    lc = cfg["val_label_col"]
    assert lc in df_val_raw.columns, (
        f"Label column '{lc}' not found in validation CSV. "
        f"Columns present: {list(df_val_raw.columns)}"
    )

    df_val = df_val_raw.copy()
    df_val = df_val.rename(columns={lc: "label"})
    df_val["label"] = pd.to_numeric(df_val["label"], errors="coerce")

    bad_labels = set(df_val["label"].dropna().unique()) - {0, 1, 2}
    if bad_labels:
        log.warning(f"Unexpected label values in validation set: {bad_labels}")

    df_val = df_val.dropna(subset=["label"]).reset_index(drop=True)
    df_val["label"] = df_val["label"].astype(int)

    # Give val rows dummy group_ids (they're not used for contrastive training)
    val_gid_start = human_id_start + len(df_human) + 1
    df_val["group_id"] = np.arange(val_gid_start, val_gid_start + len(df_val))
    if "source" not in df_val.columns:
        df_val["source"] = "validation"

    val_df = df_val[[c for c in keep if c in df_val.columns]].copy()

    # ── 10. Save ──────────────────────────────────────────────
    main_df .to_csv(out_dir / "full_train_pool.csv", index=False)
    train_df.to_csv(out_dir / "train.csv",           index=False)
    test_df .to_csv(out_dir / "test.csv",            index=False)
    val_df  .to_csv(out_dir / "val.csv",             index=False)

    # ── 11. Summary ───────────────────────────────────────────
    log.info("\n" + "="*58)
    log.info("DATASET SUMMARY")
    log.info("="*58)
    log.info(f"{'Split':<10} {'Total':>8}  {'Human':>8}  {'AI-Orig':>8}  {'Para':>8}")
    log.info("-"*58)
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        vc = df["label"].value_counts()
        log.info(f"{name:<10} {len(df):>8}  "
                 f"{vc.get(0,0):>8}  {vc.get(1,0):>8}  {vc.get(2,0):>8}")
    log.info("="*58)

    gsizes = train_df.groupby("group_id").size()
    log.info(f"\nContrastive group stats (train only):")
    log.info(f"  Mean size : {gsizes.mean():.2f}")
    log.info(f"  Solo (humans)   : {(gsizes == 1).sum()}")
    log.info(f"  Full (4 members): {(gsizes == 4).sum()}")
    log.info(f"  Partial (2-3)   : {((gsizes > 1) & (gsizes < 4)).sum()}")
    log.info(f"\nSaved to: {out_dir}/")

    return {"train_df": train_df, "val_df": val_df,
            "test_df": test_df,  "main_df": main_df}


if __name__ == "__main__":
    prepare_data()
