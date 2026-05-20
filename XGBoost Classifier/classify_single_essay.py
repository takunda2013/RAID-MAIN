"""
Classify a single essay with the trained binary XGBoost detector.

This script extracts:
  - a pooled semantic embedding from the fine-tuned RoBERTa contrastive model
  - 10 RAID statistical features from GPT-2 log-likelihoods

It then concatenates both feature blocks and predicts:
  0 -> Human
  1 -> AI

Example:
    python classify_single_essay.py "Paste the essay text here."
    python classify_single_essay.py --essay-file sample.txt --json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import pickle
import sys
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import RobertaTokenizer
from xgboost import XGBClassifier


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CONTRASTIVE_DIR = PROJECT_ROOT / "Contrastive-Learning"
STATS_DIR = PROJECT_ROOT / "Statistical-Calculator" / "code"
DEFAULT_MODEL_PATH = BASE_DIR / "outputs" / "xgb_model.json"
DEFAULT_CALIBRATED_MODEL_PATH = BASE_DIR / "outputs" / "xgb_model_calibrated.pkl"
DEFAULT_FEATURE_NAMES_PATH = BASE_DIR / "outputs" / "feature_names.txt"
DEFAULT_TRAINING_CONFIG_PATH = BASE_DIR / "outputs" / "training_config.json"

LABEL_NAMES = {
    0: "Human",
    1: "AI",
}

DEFAULT_MIN_WORDS = 120


def load_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def normalize_text(text: str) -> str:
    # Keep the text as close as possible to the training-time extractors.
    # Those pipelines did not Unicode-normalize before tokenization.
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Essay text is empty after normalization.")
    return cleaned


def read_essay(args: argparse.Namespace) -> str:
    if bool(args.essay) == bool(args.essay_file):
        raise ValueError("Provide exactly one of: positional essay text or --essay-file.")

    if args.essay_file:
        text = Path(args.essay_file).read_text(encoding="utf-8")
    else:
        text = args.essay

    return normalize_text(text)


def count_words(text: str) -> int:
    return len(text.split())


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def cleanup_torch(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


@lru_cache(maxsize=1)
def load_stage2_module() -> Any:
    return load_module(
        "stage2_roberta_finetune_inference",
        CONTRASTIVE_DIR / "stage2_roberta_finetune.py",
    )


@lru_cache(maxsize=1)
def load_raid_module() -> Any:
    return load_module(
        "raid_features_inference",
        STATS_DIR / "raid_features.py",
    )


def extract_semantic_embedding(text: str, device: torch.device) -> np.ndarray:
    stage2 = load_stage2_module()
    model_name = stage2.TRAIN_CONFIG["model_name"]
    max_length = int(stage2.TRAIN_CONFIG["max_length"])
    tokenizer_dir = CONTRASTIVE_DIR / "models" / "roberta_contrastive" / "tokenizer"
    weights_path = CONTRASTIVE_DIR / "models" / "roberta_contrastive" / "best_model.pt"

    if tokenizer_dir.exists():
        tokenizer = RobertaTokenizer.from_pretrained(str(tokenizer_dir))
        log.info("Loaded tokenizer from %s", tokenizer_dir)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        log.info("Tokenizer directory missing; fell back to %s", model_name)

    if not weights_path.exists():
        raise FileNotFoundError(f"RoBERTa weights not found: {weights_path}")

    model = stage2.RoBERTaContrastive(
        model_name=model_name,
        proj_hidden=stage2.TRAIN_CONFIG["proj_hidden"],
        proj_out=stage2.TRAIN_CONFIG["proj_out"],
        grad_checkpointing=False,
    ).to(device)

    state_dict = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    log.info("Loaded semantic encoder weights from %s", weights_path)

    encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    amp_context = nullcontext()
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp_context = torch.amp.autocast("cuda", dtype=amp_dtype)

    with torch.no_grad():
        with amp_context:
            pooled, _ = model(input_ids, attention_mask)

    semantic = pooled.squeeze(0).float().cpu().numpy().astype(np.float32, copy=False)

    del model
    cleanup_torch(device)
    return semantic


def extract_statistical_features(
    text: str,
    device: torch.device,
    model_name: str,
    max_length: int,
) -> tuple[np.ndarray, dict[str, float]]:
    raid_module = load_raid_module()
    extractor = raid_module.RaidFeatureExtractor(
        model_name,
        device=str(device),
        max_length=max_length,
    )
    result = extractor.compute(text)
    feature_names = raid_module.FEATURE_NAMES
    stats_vector = np.array(
        [result.features[name] for name in feature_names],
        dtype=np.float32,
    )

    del extractor
    cleanup_torch(device)
    return stats_vector, result.features


def concatenate_features(semantic: np.ndarray, statistical: np.ndarray) -> np.ndarray:
    if semantic.ndim != 1:
        raise ValueError(f"Expected 1D semantic vector, got {semantic.shape}")
    if statistical.ndim != 1:
        raise ValueError(f"Expected 1D statistical vector, got {statistical.shape}")
    return np.concatenate([semantic, statistical]).astype(np.float32, copy=False)


def load_feature_names(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_feature_layout(
    features: np.ndarray,
    feature_names: list[str] | None,
    statistical_feature_names: tuple[str, ...],
) -> None:
    if feature_names is not None and len(feature_names) != len(features):
        raise ValueError(
            f"Feature count mismatch: built {len(features)} features but metadata expects {len(feature_names)}"
        )

    if feature_names is not None:
        stat_tail = feature_names[-len(statistical_feature_names):]
        if tuple(stat_tail) != statistical_feature_names:
            raise ValueError(
                "Statistical feature ordering does not match training metadata: "
                f"{stat_tail} != {list(statistical_feature_names)}"
            )


def load_model(path: Path) -> XGBClassifier:
    if not path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {path}")
    model = XGBClassifier()
    model.load_model(path)
    return model


def load_calibrated_model(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def expected_model_features(model: XGBClassifier) -> int:
    booster = model.get_booster()
    return int(booster.num_features())


def load_default_threshold(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    threshold = data.get("threshold")
    return float(threshold) if threshold is not None else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify a single essay with the XGBoost detector")
    parser.add_argument("essay", nargs="?", help="Essay text to classify. Wrap it in quotes.")
    parser.add_argument("--essay-file", help="Path to a UTF-8 text file containing the essay.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to xgb_model.json")
    parser.add_argument(
        "--calibrated-model",
        default=str(DEFAULT_CALIBRATED_MODEL_PATH),
        help="Path to xgb_model_calibrated.pkl",
    )
    parser.add_argument(
        "--feature-names",
        default=str(DEFAULT_FEATURE_NAMES_PATH),
        help="Path to feature_names.txt used during training",
    )
    parser.add_argument(
        "--training-config",
        default=str(DEFAULT_TRAINING_CONFIG_PATH),
        help="Path to training_config.json for default threshold metadata",
    )
    parser.add_argument("--threshold", type=float, help="Decision threshold for AI class")
    parser.add_argument(
        "--probability-source",
        choices=("raw", "calibrated"),
        default="raw",
        help="Which probability to use for prob_ai and the final label",
    )
    parser.add_argument(
        "--raw-model-only",
        action="store_true",
        help="Deprecated alias for --probability-source raw",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device for semantic and statistical extraction",
    )
    parser.add_argument(
        "--stats-model",
        default="gpt2-large",
        help="Causal LM used for RAID statistical features",
    )
    parser.add_argument(
        "--stats-max-length",
        type=int,
        default=1024,
        help="Maximum token length for RAID statistical extraction",
    )
    parser.add_argument("--json", action="store_true", help="Print the result as JSON")
    parser.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_WORDS,
        help="Warn or abstain when the text is shorter than the essay lengths seen in training",
    )
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Classify short or out-of-domain texts anyway",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    essay = read_essay(args)
    word_count = count_words(essay)
    device = resolve_device(args.device)
    model_path = Path(args.model)
    calibrated_model_path = Path(args.calibrated_model)
    feature_names_path = Path(args.feature_names)
    training_config_path = Path(args.training_config)
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else (load_default_threshold(training_config_path) or 0.5)
    )

    if args.min_words > 0 and word_count < args.min_words and not args.allow_short:
        message = (
            f"Text is too short for reliable essay classification: {word_count} words "
            f"(minimum recommended: {args.min_words}). "
            "This detector was trained on much longer student essays. "
            "Use --allow-short to force classification anyway."
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "out_of_domain",
                        "reason": message,
                        "word_count": word_count,
                        "recommended_min_words": int(args.min_words),
                    },
                    indent=2,
                )
            )
            return 2
        raise ValueError(message)

    log.info("Using device: %s", device)
    log.info("Extracting semantic embedding...")
    semantic = extract_semantic_embedding(essay, device)

    log.info("Extracting statistical features with %s...", args.stats_model)
    statistical, stats_dict = extract_statistical_features(
        essay,
        device,
        args.stats_model,
        args.stats_max_length,
    )

    features = concatenate_features(semantic, statistical)
    feature_names = load_feature_names(feature_names_path)

    raid_module = load_raid_module()
    statistical_feature_names = tuple(raid_module.FEATURE_NAMES)
    validate_feature_layout(features, feature_names, statistical_feature_names)

    model = load_model(model_path)
    expected_features = expected_model_features(model)
    if len(features) != expected_features:
        raise ValueError(
            f"Model expects {expected_features} features, but inference built {len(features)}"
        )

    probability_ai_raw = float(model.predict_proba(features.reshape(1, -1))[0, 1])
    probability_ai_calibrated = None
    calibrated_model = load_calibrated_model(calibrated_model_path)
    if calibrated_model is not None:
        probability_ai_calibrated = float(
            calibrated_model.predict_proba(features.reshape(1, -1))[0, 1]
        )

    probability_source = "raw" if args.raw_model_only else args.probability_source
    if probability_source == "calibrated" and probability_ai_calibrated is not None:
        log.info("Using calibrated probability from %s", calibrated_model_path)
        probability_ai = probability_ai_calibrated
    else:
        if probability_source == "calibrated":
            log.warning("Calibrated model not found; falling back to raw XGBoost probability.")
        log.info("Using raw XGBoost probability from %s", model_path)
        probability_source = "raw"
        probability_ai = probability_ai_raw

    predicted_label = int(probability_ai >= threshold)
    probability_human = 1.0 - probability_ai

    result = {
        "status": "ok",
        "label_id": predicted_label,
        "label": LABEL_NAMES[predicted_label],
        "threshold": float(threshold),
        "probability_source": probability_source,
        "prob_ai": probability_ai,
        "prob_human": probability_human,
        "prob_ai_raw": probability_ai_raw,
        "prob_human_raw": 1.0 - probability_ai_raw,
        "word_count": int(word_count),
        "semantic_dim": int(len(semantic)),
        "statistical_dim": int(len(statistical)),
        "statistical_features": {name: float(stats_dict[name]) for name in statistical_feature_names},
    }
    if probability_ai_calibrated is not None:
        result["prob_ai_calibrated"] = probability_ai_calibrated
        result["prob_human_calibrated"] = 1.0 - probability_ai_calibrated

    if args.min_words > 0 and word_count < args.min_words:
        result["warning"] = (
            f"Short text: {word_count} words. This detector was trained on essays mostly "
            f"{args.min_words}+ words long, so confidence may be unreliable."
        )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Prediction: {result['label']}")
        print(f"Label ID: {result['label_id']}")
        print(f"P(Human): {result['prob_human']:.4f}")
        print(f"P(AI): {result['prob_ai']:.4f}")
        print(f"Probability source: {result['probability_source']}")
        print(f"Raw P(AI): {result['prob_ai_raw']:.4f}")
        if "prob_ai_calibrated" in result:
            print(f"Calibrated P(AI): {result['prob_ai_calibrated']:.4f}")
        print(f"Threshold: {result['threshold']:.2f}")
        print(f"Semantic dims: {result['semantic_dim']}")
        print(f"Statistical dims: {result['statistical_dim']}")
        print("Statistical features:")
        for name in statistical_feature_names:
            print(f"  {name}: {result['statistical_features'][name]:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
