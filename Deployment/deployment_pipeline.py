"""
Live essay classification and LIME audit pipeline.

This module is the stable deployment boundary for the Streamlit UI:

    result = analyze_text(essay_text)

The production decision uses the calibrated semantic model selected from the
probability audit:

    XGBoost semantic-only + isotonic calibration

The explanation/auditor path uses the calibrated statistical model selected for
better probability movement under LIME perturbations:

    Logistic statistical-only + isotonic calibration

The UI should render the returned dictionary. It should not import training
scripts or model internals directly.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import logging
import pickle
import re
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from transformers import RobertaTokenizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


DEPLOYMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOYMENT_DIR.parent
CONTRASTIVE_DIR = PROJECT_ROOT / "Contrastive-Learning"
STATS_DIR = PROJECT_ROOT / "Statistical-Calculator" / "code"
AUDIT_OUTPUTS_DIR = (
    PROJECT_ROOT
    / "XGBoost Classifier"
    / "probability_audit_fixes"
    / "outputs"
)

DEFAULT_PRODUCTION_MODEL_PATH = AUDIT_OUTPUTS_DIR / "xgb_semantic_sigmoid_calibrated_model.pkl"
DEFAULT_AUDITOR_MODEL_PATH = AUDIT_OUTPUTS_DIR / "logistic_statistical_isotonic_calibrated_model.pkl"
DEFAULT_FEATURE_NAMES_PATH = PROJECT_ROOT / "Concatenation-engine" / "data" / "concatenated" / "unseen_feature_names.txt"
DEFAULT_MIN_WORDS = 100

LABEL_NAMES = {
    0: "Human-written",
    1: "AI-generated",
}

STATISTICAL_FEATURE_NAMES = (
    "mean_surprisal",
    "stdev_surprisal",
    "var_surprisal",
    "skew_surprisal",
    "kurtosis_surprisal",
    "mean_diff_surprisal",
    "stdev_diff_surprisal",
    "var_second_diff_loglik",
    "entropy_second_diff_loglik",
    "autocorr_second_diff_loglik",
)


@dataclass(frozen=True)
class DeploymentConfig:
    production_model_path: Path = DEFAULT_PRODUCTION_MODEL_PATH
    auditor_model_path: Path = DEFAULT_AUDITOR_MODEL_PATH
    feature_names_path: Path = DEFAULT_FEATURE_NAMES_PATH
    stats_model_name: str = "gpt2-large"
    stats_max_length: int = 1024
    lime_num_features: int = 12
    # GPT-2-large statistical feature extraction is expensive for every LIME
    # perturbation. Keep the live default small; raise it for slower audits.
    lime_num_samples: int = 20
    # Suppress LIME when auditor confidence reaches this threshold.
    lime_suppression_confidence_percent: int = 99
    # LIME display controls.
    # We currently expose all raw LIME tokens in UI/report when LIME runs.
    lime_max_display_tokens: int = 8
    lime_min_abs_weight: float = 0.02
    lime_include_counterevidence: bool = False
    lime_mixed_counterevidence_low_percent: int = 35
    lime_mixed_counterevidence_high_percent: int = 65
    # Prevent over-highlighting repeated words in long essays.
    lime_max_occurrences_per_token: int = 1
    min_words: int = DEFAULT_MIN_WORDS
    device: str = "auto"


@dataclass(frozen=True)
class AnalysisResult:
    status: str
    label_id: int
    label: str
    confidence: float
    confidence_percent: int
    prob_ai: float
    prob_human: float
    ai_probability_percent: int
    word_count: int
    summary: str
    explanation: str
    production: dict[str, Any]
    auditor: dict[str, Any]
    statistical_features: dict[str, float]
    top_tokens: list[dict[str, Any]]
    annotated_spans: list[dict[str, Any]]
    lime_diagnostics: dict[str, Any]
    warnings: list[str]
    lime_html: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def normalize_text(text: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        raise ValueError("Essay text is empty.")
    return cleaned


def count_words(text: str) -> int:
    return len(text.split())


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@lru_cache(maxsize=1)
def load_stage2_module() -> Any:
    return load_module(
        "stage2_roberta_finetune_deployment",
        CONTRASTIVE_DIR / "stage2_roberta_finetune.py",
    )


@lru_cache(maxsize=1)
def load_raid_module() -> Any:
    return load_module(
        "raid_features_deployment",
        STATS_DIR / "raid_features.py",
    )


@lru_cache(maxsize=4)
def load_pickle_model(path_text: str) -> Any:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=2)
def load_feature_names(path_text: str) -> list[str]:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Feature metadata not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@lru_cache(maxsize=2)
def load_semantic_resources(device_text: str) -> tuple[Any, Any, torch.device]:
    device = resolve_device(device_text)
    stage2 = load_stage2_module()
    cfg = stage2.TRAIN_CONFIG
    tokenizer_dir = CONTRASTIVE_DIR / "models" / "roberta_contrastive" / "tokenizer"
    weights_path = CONTRASTIVE_DIR / "models" / "roberta_contrastive" / "best_model.pt"

    if tokenizer_dir.exists():
        tokenizer = RobertaTokenizer.from_pretrained(str(tokenizer_dir))
        log.info("Loaded RoBERTa tokenizer from %s", tokenizer_dir)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(cfg["model_name"])
        log.info("Tokenizer directory missing; loaded %s", cfg["model_name"])

    if not weights_path.exists():
        raise FileNotFoundError(f"RoBERTa weights not found: {weights_path}")

    model = stage2.RoBERTaContrastive(
        model_name=cfg["model_name"],
        proj_hidden=cfg["proj_hidden"],
        proj_out=cfg["proj_out"],
        grad_checkpointing=False,
    ).to(device)
    state_dict = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    log.info("Loaded RoBERTa deployment encoder from %s", weights_path)
    return model, tokenizer, device


@lru_cache(maxsize=4)
def load_raid_extractor(model_name: str, device_text: str, max_length: int) -> Any:
    raid_module = load_raid_module()
    device = "cuda" if device_text == "auto" and torch.cuda.is_available() else device_text
    if device == "auto":
        device = "cpu"
    extractor = raid_module.RaidFeatureExtractor(
        model_name,
        device=str(device),
        max_length=int(max_length),
    )
    log.info("Loaded RAID feature extractor with %s on %s", model_name, extractor.device)
    return extractor


def extract_semantic_embedding(text: str, config: DeploymentConfig) -> np.ndarray:
    stage2 = load_stage2_module()
    model, tokenizer, device = load_semantic_resources(config.device)
    max_length = int(stage2.TRAIN_CONFIG["max_length"])

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

    return pooled.squeeze(0).float().cpu().numpy().astype(np.float32, copy=False)


def extract_statistical_features(text: str, config: DeploymentConfig) -> tuple[np.ndarray, dict[str, float]]:
    extractor = load_raid_extractor(
        config.stats_model_name,
        config.device,
        config.stats_max_length,
    )
    result = extractor.compute(text)
    feature_vector = np.array(
        [result.features[name] for name in STATISTICAL_FEATURE_NAMES],
        dtype=np.float32,
    )
    return feature_vector, {name: float(result.features[name]) for name in STATISTICAL_FEATURE_NAMES}


def predict_binary(model: Any, features: np.ndarray) -> tuple[int, float, float]:
    X = features.reshape(1, -1).astype(np.float32, copy=False)
    probs = model.predict_proba(X)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError(f"Expected binary predict_proba output, got {probs.shape}")
    prob_ai = float(probs[0, 1])
    prob_human = 1.0 - prob_ai
    label_id = int(prob_ai >= 0.5)
    return label_id, prob_human, prob_ai


def make_auditor_predictor(config: DeploymentConfig):
    auditor_model = load_pickle_model(str(config.auditor_model_path))
    cache: dict[str, np.ndarray] = {}

    def predict_proba(texts: Iterable[str]) -> np.ndarray:
        rows = []
        for item in texts:
            text = normalize_text(item)
            key = text
            if key not in cache:
                try:
                    statistical, _ = extract_statistical_features(text, config)
                except Exception as exc:
                    log.debug("LIME perturbation could not be featurized: %s", exc)
                    statistical = np.full(len(STATISTICAL_FEATURE_NAMES), np.nan, dtype=np.float32)
                cache[key] = statistical
            rows.append(cache[key])

        X = np.vstack(rows).astype(np.float32, copy=False)
        invalid = np.isnan(X).any(axis=1)
        if invalid.any():
            X[invalid] = 0.0

        probs = auditor_model.predict_proba(X)
        if probs.ndim != 2 or probs.shape[1] < 2:
            raise ValueError(f"Expected binary auditor probabilities, got {probs.shape}")

        probs = probs.astype(np.float64, copy=False)
        if invalid.any():
            probs[invalid, 0] = 0.5
            probs[invalid, 1] = 0.5
        return probs

    return predict_proba


def run_lime_audit(text: str, config: DeploymentConfig) -> tuple[dict[str, float], str]:
    explainer = LimeTextExplainer(
        class_names=["Human", "AI"],
        split_expression=r"\s+",
        bow=False,
        random_state=42,
    )
    explanation = explainer.explain_instance(
        text,
        make_auditor_predictor(config),
        num_features=int(config.lime_num_features),
        num_samples=int(config.lime_num_samples),
        labels=(1,),
    )
    weights = {str(word): float(weight) for word, weight in explanation.as_list(label=1)}
    try:
        lime_html = explanation.as_html(labels=(1,))
    except Exception:
        lime_html = ""
    return weights, lime_html


def clean_lookup_token(token: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", token).lower()


def _filter_word_weights_for_display(
    word_weights: dict[str, float],
    *,
    auditor_prob_ai: float,
    max_tokens: int,
    min_abs_weight: float,
    include_counterevidence: bool,
) -> dict[str, float]:
    if not word_weights:
        return {}

    max_tokens = max(1, int(max_tokens))
    min_abs_weight = max(0.0, float(min_abs_weight))
    predicted_ai = auditor_prob_ai >= 0.5

    items = [(str(token), float(weight)) for token, weight in word_weights.items()]
    items = [(token, weight) for token, weight in items if abs(weight) >= min_abs_weight]

    if not include_counterevidence:
        if predicted_ai:
            items = [(token, weight) for token, weight in items if weight > 0]
        else:
            items = [(token, weight) for token, weight in items if weight < 0]
    else:
        # In mixed mode, try to show both sides when available.
        pos = sorted([(token, weight) for token, weight in items if weight > 0], key=lambda x: abs(x[1]), reverse=True)
        neg = sorted([(token, weight) for token, weight in items if weight < 0], key=lambda x: abs(x[1]), reverse=True)

        # If thresholding removed one side entirely, recover one token from the
        # raw LIME output so mixed cases can still show both perspectives.
        raw_all = sorted(
            [(str(token), float(weight)) for token, weight in word_weights.items()],
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        if not pos:
            raw_pos = [(token, weight) for token, weight in raw_all if weight > 0]
            if raw_pos:
                pos = raw_pos[:1]
        if not neg:
            raw_neg = [(token, weight) for token, weight in raw_all if weight < 0]
            if raw_neg:
                neg = raw_neg[:1]

        each = max(1, max_tokens // 2)
        selected = pos[:each] + neg[:each]
        remainder = pos[each:] + neg[each:]
        remainder = sorted(remainder, key=lambda x: abs(x[1]), reverse=True)
        selected = selected + remainder[: max(0, max_tokens - len(selected))]
        items = selected

    if not items:
        fallback = sorted(
            [(str(token), float(weight)) for token, weight in word_weights.items()],
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        if not include_counterevidence:
            if predicted_ai:
                fallback = [(token, weight) for token, weight in fallback if weight > 0]
            else:
                fallback = [(token, weight) for token, weight in fallback if weight < 0]
        items = fallback[:max_tokens]

    filtered: list[tuple[str, float]] = []
    seen: set[str] = set()
    for token, weight in sorted(items, key=lambda item: abs(item[1]), reverse=True):
        norm = clean_lookup_token(token) or token.strip().lower()
        if not norm or len(norm) <= 2:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        filtered.append((token, weight))
        if len(filtered) >= max_tokens:
            break

    # Final fallback: if filtering became too strict, return strongest tokens
    # from raw LIME output so educator still sees evidence when available.
    if not filtered and word_weights:
        for token, weight in sorted(
            [(str(t), float(w)) for t, w in word_weights.items()],
            key=lambda item: abs(item[1]),
            reverse=True,
        ):
            norm = clean_lookup_token(token) or token.strip().lower()
            if not norm:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            filtered.append((token, weight))
            if len(filtered) >= max_tokens:
                break

    # Ensure both sides are shown in counterevidence mode when raw LIME has
    # both positive and negative evidence available.
    if include_counterevidence and filtered:
        has_pos = any(weight > 0 for _, weight in filtered)
        has_neg = any(weight < 0 for _, weight in filtered)

        if not (has_pos and has_neg):
            need_positive = not has_pos

            candidates = []
            for token, weight in sorted(
                [(str(t), float(w)) for t, w in word_weights.items()],
                key=lambda item: abs(item[1]),
                reverse=True,
            ):
                if need_positive and weight <= 0:
                    continue
                if (not need_positive) and weight >= 0:
                    continue
                norm = clean_lookup_token(token) or token.strip().lower()
                if not norm or len(norm) <= 2:
                    continue
                if norm in seen:
                    continue
                candidates.append((token, weight, norm))

            if candidates:
                token, weight, norm = candidates[0]
                # Keep table length stable: if full, replace weakest token from
                # the already-present side.
                if len(filtered) >= max_tokens:
                    same_side_idxs = [
                        idx for idx, (_, w) in enumerate(filtered)
                        if (w > 0) == (not need_positive)
                    ]
                    if same_side_idxs:
                        weakest_idx = min(same_side_idxs, key=lambda idx: abs(filtered[idx][1]))
                    else:
                        weakest_idx = min(range(len(filtered)), key=lambda idx: abs(filtered[idx][1]))
                    removed_token, _ = filtered.pop(weakest_idx)
                    removed_norm = clean_lookup_token(removed_token) or removed_token.strip().lower()
                    if removed_norm in seen:
                        seen.remove(removed_norm)

                seen.add(norm)
                filtered.append((token, weight))

    return {token: weight for token, weight in filtered}


def build_top_tokens(word_weights: dict[str, float]) -> list[dict[str, Any]]:
    sorted_items = sorted(word_weights.items(), key=lambda item: abs(item[1]), reverse=True)
    rows = []
    for rank, (token, weight) in enumerate(sorted_items, start=1):
        rows.append(
            {
                "rank": rank,
                "token": token,
                "weight": float(weight),
                "weight_label": f"{weight:+.4f}",
                "signal": "AI" if weight > 0 else "Human",
            }
        )
    return rows


def build_annotated_spans(
    text: str,
    word_weights: dict[str, float],
    *,
    max_occurrences_per_token: int = 2,
) -> list[dict[str, Any]]:
    direct = {str(key): float(value) for key, value in word_weights.items()}
    cleaned = {clean_lookup_token(str(key)): float(value) for key, value in word_weights.items()}
    spans = []
    max_occurrences_per_token = max(1, int(max_occurrences_per_token))
    counts: dict[str, int] = {}

    for token in re.findall(r"\S+", text):
        key = token.strip()
        clean_key = clean_lookup_token(key)
        weight = direct.get(key)
        if weight is None:
            weight = direct.get(key.lower())
        if weight is None and clean_key:
            weight = cleaned.get(clean_key)

        if weight is not None:
            count_key = clean_key or key.lower()
            seen = counts.get(count_key, 0)
            if seen >= max_occurrences_per_token:
                weight = None
            else:
                counts[count_key] = seen + 1

        if weight is None:
            polarity = "neutral"
            weight_value = 0.0
        else:
            weight_value = float(weight)
            polarity = "ai" if weight_value > 0 else "human"

        spans.append(
            {
                "text": token,
                "polarity": polarity,
                "weight": weight_value,
            }
        )
    return spans


def _educator_risk_band(auditor_prob_ai: float) -> str:
    if auditor_prob_ai >= 0.90:
        return "very_high_ai"
    if auditor_prob_ai >= 0.75:
        return "high_ai"
    if auditor_prob_ai >= 0.55:
        return "leaning_ai"
    if auditor_prob_ai >= 0.45:
        return "mixed"
    if auditor_prob_ai >= 0.25:
        return "leaning_human"
    return "high_human"


def build_summary(auditor_prob_ai: float, warnings: list[str]) -> str:
    auditor_pct = round(auditor_prob_ai * 100)
    band = _educator_risk_band(auditor_prob_ai)

    summary_templates = {
        "very_high_ai": (
            f"The Originality Engine estimates {auditor_pct}% AI probability. "
            "This is a strong AI-likelihood signal and should be reviewed as high priority."
        ),
        "high_ai": (
            f"The Originality Engine estimates {auditor_pct}% AI probability. "
            "The writing pattern closely matches AI-generated text and warrants closer review."
        ),
        "leaning_ai": (
            f"The Originality Engine estimates {auditor_pct}% AI probability. "
            "The writing shows some AI-like patterns; consider the assignment context before drawing conclusions."
        ),
        "mixed": (
            f"The Originality Engine estimates {auditor_pct}% AI probability. "
            "The result is inconclusive — consider reviewing the submission directly or comparing with other work samples."
        ),
        "leaning_human": (
            f"The Originality Engine estimates {auditor_pct}% AI probability. "
            "The writing is generally consistent with human authorship, though the score is not definitive."
        ),
        "high_human": (
            f"The Originality Engine estimates {auditor_pct}% AI probability. "
            "The writing appears strongly consistent with human authorship."
        ),
    }

    summary = summary_templates[band]
    if warnings:
        summary += " Review warning: " + " ".join(warnings)
    return summary


def build_explanation(auditor_prob_ai: float, word_weights: dict[str, float]) -> str:
    auditor_pct = round(auditor_prob_ai * 100)
    band = _educator_risk_band(auditor_prob_ai)

    if word_weights:
        strongest = sorted(word_weights.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
        strongest_text = ", ".join(
            f"{word} ({'AI' if weight > 0 else 'Human'} {weight:+.3f})"
            for word, weight in strongest
        )
    else:
        strongest_text = "No stable LIME tokens were returned."

    educator_next_step = {
        "very_high_ai": "Recommended educator action: verify with draft history, in-class writing samples, and citation/originality checks.",
        "high_ai": "Recommended educator action: perform targeted review and compare with prior student writing style.",
        "leaning_ai": "Recommended educator action: review rubric alignment and request clarification or drafting evidence if needed.",
        "mixed": "Recommended educator action: treat as uncertain and rely on broader evidence rather than score alone.",
        "leaning_human": "Recommended educator action: low-risk flag; keep normal review procedures.",
        "high_human": "Recommended educator action: no elevated AI concern indicated by this analysis.",
    }[band]

    return (
        "The word highlights come from the calibrated statistical auditor model. "
        f"Auditor estimate: {auditor_pct}% AI probability. "
        f"Strongest audit tokens: {strongest_text}. "
        f"{educator_next_step}"
    )


def validate_feature_layout(config: DeploymentConfig, semantic: np.ndarray, statistical: np.ndarray) -> None:
    names = load_feature_names(str(config.feature_names_path))
    semantic_names = [name for name in names if name.startswith("semantic_")]
    stat_names = names[-len(STATISTICAL_FEATURE_NAMES):]
    if len(semantic_names) != len(semantic):
        raise ValueError(
            f"Semantic feature mismatch: metadata has {len(semantic_names)}, inference built {len(semantic)}"
        )
    if tuple(stat_names) != STATISTICAL_FEATURE_NAMES:
        raise ValueError(
            "Statistical feature ordering mismatch: "
            f"{stat_names} != {list(STATISTICAL_FEATURE_NAMES)}"
        )
    if len(statistical) != len(STATISTICAL_FEATURE_NAMES):
        raise ValueError(
            f"Statistical feature mismatch: built {len(statistical)}, expected {len(STATISTICAL_FEATURE_NAMES)}"
        )


def analyze_text(text: str, config: DeploymentConfig | None = None, *, include_lime: bool = True) -> dict[str, Any]:
    config = config or DeploymentConfig()
    essay_text = normalize_text(text)
    word_count = count_words(essay_text)
    warnings: list[str] = []
    if word_count < config.min_words:
        warnings.append(
            f"Text has {word_count} words; {config.min_words}+ words is recommended for reliable essay detection."
        )

    production_model = load_pickle_model(str(config.production_model_path))
    auditor_model = load_pickle_model(str(config.auditor_model_path))

    semantic = extract_semantic_embedding(essay_text, config)
    statistical, statistical_dict = extract_statistical_features(essay_text, config)
    validate_feature_layout(config, semantic, statistical)

    production_label_id, production_prob_human, production_prob_ai = predict_binary(production_model, semantic)
    _, auditor_prob_human, auditor_prob_ai = predict_binary(auditor_model, statistical)

    label = LABEL_NAMES[production_label_id]
    confidence = max(production_prob_human, production_prob_ai)
    confidence_percent = int(round(confidence * 100))

    log.info(
        "Production inference | model=xgb_semantic_sigmoid | label=%s (%d) | prob_ai=%.6f | prob_human=%.6f | confidence=%.6f (%d%%)",
        label,
        production_label_id,
        production_prob_ai,
        production_prob_human,
        confidence,
        confidence_percent,
    )
    log.info(
        "Auditor inference    | model=logistic_statistical_isotonic | prob_ai=%.6f | prob_human=%.6f",
        auditor_prob_ai,
        auditor_prob_human,
    )

    # LIME suppression is based on auditor confidence, since LIME explains
    # the auditor model path.
    lime_suppression_pct = max(0, min(int(config.lime_suppression_confidence_percent), 100))
    auditor_confidence_pct_exact = max(auditor_prob_human, auditor_prob_ai) * 100.0

    raw_word_weights: dict[str, float] = {}
    lime_html = ""
    lime_allowed = include_lime and auditor_confidence_pct_exact < float(lime_suppression_pct)
    if lime_allowed:
        raw_word_weights, lime_html = run_lime_audit(essay_text, config)

    if raw_word_weights:
        raw_items = sorted(raw_word_weights.items(), key=lambda item: abs(float(item[1])), reverse=True)
        raw_tokens_text = "; ".join(f"{token}={float(weight):+.4f}" for token, weight in raw_items)
        log.info("LIME raw tokens      | %s", raw_tokens_text)
    elif lime_allowed:
        log.info("LIME raw tokens      | none returned by explainer")
    elif include_lime:
        log.info(
            "LIME raw tokens      | skipped (auditor_confidence=%.4f%% >= suppression_threshold=%d%%)",
            auditor_confidence_pct_exact,
            lime_suppression_pct,
        )

    # Per request: display all raw LIME tokens when LIME runs.
    include_counterevidence = lime_allowed

    word_weights = {
        str(token): float(weight)
        for token, weight in raw_word_weights.items()
        if np.isfinite(float(weight))
    }

    raw_ai_count = sum(1 for w in raw_word_weights.values() if float(w) > 0)
    raw_human_count = sum(1 for w in raw_word_weights.values() if float(w) < 0)
    displayed_ai_count = sum(1 for w in word_weights.values() if float(w) > 0)
    displayed_human_count = sum(1 for w in word_weights.values() if float(w) < 0)

    lime_diagnostics = {
        "lime_enabled": bool(include_lime),
        "lime_ran": bool(include_lime and auditor_confidence_pct_exact < float(lime_suppression_pct)),
        "suppression_threshold_percent": int(lime_suppression_pct),
        "auditor_confidence_percent": round(auditor_confidence_pct_exact, 4),
        "counterevidence_mode": bool(include_counterevidence),
        "raw_token_counts": {
            "ai": int(raw_ai_count),
            "human": int(raw_human_count),
            "total": int(len(raw_word_weights)),
        },
        "displayed_token_counts": {
            "ai": int(displayed_ai_count),
            "human": int(displayed_human_count),
            "total": int(len(word_weights)),
        },
    }

    log.info(
        "LIME diagnostics     | ran=%s | counterevidence=%s | raw(ai=%d,human=%d,total=%d) | displayed(ai=%d,human=%d,total=%d)",
        lime_diagnostics["lime_ran"],
        lime_diagnostics["counterevidence_mode"],
        lime_diagnostics["raw_token_counts"]["ai"],
        lime_diagnostics["raw_token_counts"]["human"],
        lime_diagnostics["raw_token_counts"]["total"],
        lime_diagnostics["displayed_token_counts"]["ai"],
        lime_diagnostics["displayed_token_counts"]["human"],
        lime_diagnostics["displayed_token_counts"]["total"],
    )

    summary_text = build_summary(auditor_prob_ai, warnings)
    # Keep educator wording aligned with displayed token evidence.
    has_ai_display_token = any(float(w) > 0 for w in word_weights.values())
    if not has_ai_display_token:
        summary_text = summary_text.replace(
            "with minor AI-like cues.",
            "without notable AI-like token cues in this explanation.",
        )

    result = AnalysisResult(
        status="ok",
        label_id=production_label_id,
        label=label,
        confidence=confidence,
        confidence_percent=confidence_percent,
        prob_ai=production_prob_ai,
        prob_human=production_prob_human,
        ai_probability_percent=int(round(production_prob_ai * 100)),
        word_count=word_count,
        summary=summary_text,
        explanation=build_explanation(auditor_prob_ai, word_weights),
        production={
            "model": "xgb_semantic_sigmoid",
            "prob_ai": production_prob_ai,
            "prob_human": production_prob_human,
            "label_id": production_label_id,
            "label": label,
        },
        auditor={
            "model": "logistic_statistical_isotonic",
            "prob_ai": auditor_prob_ai,
            "prob_human": auditor_prob_human,
            "purpose": "LIME audit and probability movement, not final production decision",
        },
        statistical_features=statistical_dict,
        top_tokens=build_top_tokens(word_weights),
        annotated_spans=build_annotated_spans(
            essay_text,
            word_weights,
            max_occurrences_per_token=config.lime_max_occurrences_per_token,
        ),
        lime_diagnostics=lime_diagnostics,
        warnings=warnings,
        lime_html=lime_html,
    )
    return result.to_dict()


def result_to_ui_tuples(result: dict[str, Any]) -> dict[str, Any]:
    """Convert the deployment result into the tuple shapes used by the UI."""
    tokens = tuple(
        (
            int(item["rank"]),
            str(item["token"]),
            str(item["weight_label"]),
            str(item["signal"]),
        )
        for item in result.get("top_tokens", [])[:12]
    )
    annotated_words = tuple(
        (str(item["text"]), str(item["polarity"]))
        for item in result.get("annotated_spans", [])
    )
    stats = result.get("statistical_features", {})
    auditor = result.get("auditor", {})
    auditor_prob_ai = float(auditor.get("prob_ai", 0.0))
    auditor_pct = int(round(auditor_prob_ai * 100))
    metrics = (
        ("Auditor AI probability", f"{auditor_pct}%", "Displayed detector score"),
        ("Mean surprisal", f"{float(stats.get('mean_surprisal', 0.0)):.3f}", "Statistical feature"),
        ("Entropy 2nd diff", f"{float(stats.get('entropy_second_diff_loglik', 0.0)):.3f}", "Statistical feature"),
    )
    return {
        "tokens": tokens,
        "annotated_words": annotated_words,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Robust AI Detector deployment pipeline")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Essay text to classify")
    source.add_argument("--essay-file", type=Path, help="UTF-8 text file containing the essay")
    parser.add_argument("--json", action="store_true", help="Print the full JSON result")
    parser.add_argument("--no-lime", action="store_true", help="Skip LIME and only classify")
    parser.add_argument("--lime-samples", type=int, default=20)
    parser.add_argument("--lime-features", type=int, default=12)
    parser.add_argument(
        "--lime-suppress-confidence-pct",
        type=int,
        default=99,
        help="Skip LIME when auditor confidence percent is >= this value (default: 99).",
    )
    parser.add_argument(
        "--lime-max-display-tokens",
        type=int,
        default=8,
        help="Maximum LIME tokens to display in output (default: 8).",
    )
    parser.add_argument(
        "--lime-min-abs-weight",
        type=float,
        default=0.02,
        help="Minimum absolute token weight to display (default: 0.02).",
    )
    parser.add_argument(
        "--lime-include-counterevidence",
        action="store_true",
        help="Always include opposite-side evidence tokens (disabled by default).",
    )
    parser.add_argument(
        "--lime-mixed-low-pct",
        type=int,
        default=35,
        help="Lower bound of mixed range where both sides are shown (default: 35).",
    )
    parser.add_argument(
        "--lime-mixed-high-pct",
        type=int,
        default=65,
        help="Upper bound of mixed range where both sides are shown (default: 65).",
    )
    parser.add_argument(
        "--lime-max-occurrences-per-token",
        type=int,
        default=1,
        help="Maximum highlighted occurrences per token in annotated text (default: 1).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = args.essay_file.read_text(encoding="utf-8") if args.essay_file else str(args.text)
    config = DeploymentConfig(
        lime_num_samples=args.lime_samples,
        lime_num_features=args.lime_features,
        lime_suppression_confidence_percent=args.lime_suppress_confidence_pct,
        lime_max_display_tokens=args.lime_max_display_tokens,
        lime_min_abs_weight=args.lime_min_abs_weight,
        lime_include_counterevidence=args.lime_include_counterevidence,
        lime_mixed_counterevidence_low_percent=args.lime_mixed_low_pct,
        lime_mixed_counterevidence_high_percent=args.lime_mixed_high_pct,
        lime_max_occurrences_per_token=args.lime_max_occurrences_per_token,
        device=args.device,
    )
    result = analyze_text(text, config=config, include_lime=not args.no_lime)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence_percent']}%")
        print(f"Production P(AI): {result['production']['prob_ai']:.4f}")
        print(f"Auditor P(AI): {result['auditor']['prob_ai']:.4f}")
        if result["warnings"]:
            print("Warnings:")
            for warning in result["warnings"]:
                print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
