"""
Fast inference for the binary XGBoost classifier.

Loads:
  - a trained XGBoost model (json)
  - a concatenated feature matrix (.npy)
Optionally loads labels (.npy) to compute metrics.

Label mapping for metrics:
  0 -> Human
  1 or 2 -> AI
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_model(path: Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(path)
    return model


def main() -> int:
    p = argparse.ArgumentParser(description="Predict with trained binary XGBoost model")
    p.add_argument("--model", required=True, help="Path to xgb_model.json")
    p.add_argument("--features", required=True, help="Path to *_concatenated.npy")
    p.add_argument("--labels", help="Optional path to *_labels.npy (0/1/2)")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for AI class")
    p.add_argument("--out", required=True, help="Output CSV path for predictions")
    args = p.parse_args()

    model_path = Path(args.model)
    features_path = Path(args.features)
    labels_path = Path(args.labels) if args.labels else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")

    X = np.load(features_path).astype(np.float32, copy=False)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D features, got {X.shape}")

    model = load_model(model_path)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= args.threshold).astype(np.int32)

    out_df = pd.DataFrame(
        {
            "row_index": np.arange(len(preds), dtype=np.int64),
            "predicted_label": preds,
            "prob_ai": probs,
        }
    )

    if labels_path is not None:
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {labels_path}")
        y_raw = np.load(labels_path)
        if len(y_raw) != len(preds):
            raise ValueError(f"Label length mismatch: {len(y_raw)} vs {len(preds)}")
        y = (y_raw != 0).astype(np.int32)
        out_df.insert(1, "true_label", y)

        accuracy = accuracy_score(y, preds)
        precision = precision_score(y, preds, zero_division=0)
        recall = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        auc = roc_auc_score(y, probs)
        cm = confusion_matrix(y, preds, labels=[0, 1])

        log.info("Accuracy: %.4f", accuracy)
        log.info("Precision (AI): %.4f", precision)
        log.info("Recall (AI): %.4f", recall)
        log.info("F1 (AI): %.4f", f1)
        log.info("AUC: %.4f", auc)
        log.info("Confusion matrix:\n%s", cm)
        print(classification_report(y, preds, target_names=["Human", "AI"], digits=4))

    out_df.to_csv(out_path, index=False)
    log.info("Wrote predictions -> %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

