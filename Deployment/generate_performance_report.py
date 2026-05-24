from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import deployment_pipeline as dp


def _ms(seconds: float) -> float:
    return round(seconds * 1000.0, 3)


def _run_report(essay_text: str, lime_samples: int, lime_features: int, device: str) -> dict[str, Any]:
    config = dp.DeploymentConfig(
        lime_num_samples=lime_samples,
        lime_num_features=lime_features,
        device=device,
    )

    t0 = time.perf_counter()
    normalized = dp.normalize_text(essay_text)
    word_count = dp.count_words(normalized)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    semantic = dp.extract_semantic_embedding(normalized, config)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    statistical_vec, statistical_features = dp.extract_statistical_features(normalized, config)
    t5 = time.perf_counter()

    t6 = time.perf_counter()
    production_model = dp.load_pickle_model(str(config.production_model_path))
    prod_label, prod_prob_human, prod_prob_ai = dp.predict_binary(production_model, semantic)
    t7 = time.perf_counter()

    t8 = time.perf_counter()
    auditor_model = dp.load_pickle_model(str(config.auditor_model_path))
    _, aud_prob_human, aud_prob_ai = dp.predict_binary(auditor_model, statistical_vec)
    t9 = time.perf_counter()

    t10 = time.perf_counter()
    word_weights, _ = dp.run_lime_audit(normalized, config)
    t11 = time.perf_counter()

    t12 = time.perf_counter()
    full_result = dp.analyze_text(normalized, config=config, include_lime=True)
    t13 = time.perf_counter()

    top_tokens = dp.build_top_tokens(word_weights)[:12]
    label = dp.LABEL_NAMES.get(prod_label, str(prod_label))

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "word_count": word_count,
            "min_words_recommended": config.min_words,
        },
        "prediction": {
            "label_id": prod_label,
            "label": label,
            "production_prob_ai": prod_prob_ai,
            "production_prob_human": prod_prob_human,
            "auditor_prob_ai": aud_prob_ai,
            "auditor_prob_human": aud_prob_human,
        },
        "timing_ms": {
            "normalize_and_wordcount": _ms(t1 - t0),
            "semantic_embedding_extraction": _ms(t3 - t2),
            "statistical_feature_extraction": _ms(t5 - t4),
            "production_model_inference": _ms(t7 - t6),
            "auditor_model_inference": _ms(t9 - t8),
            "lime_explanation": _ms(t11 - t10),
            "full_pipeline_analyze_text_with_lime": _ms(t13 - t12),
            "estimated_total_manual_path": _ms((t1 - t0) + (t3 - t2) + (t5 - t4) + (t7 - t6) + (t9 - t8) + (t11 - t10)),
        },
        "lime": {
            "num_samples": lime_samples,
            "num_features": lime_features,
            "top_tokens": top_tokens,
        },
        "statistical_features": statistical_features,
        "full_pipeline_result": full_result,
    }


def _to_markdown(report: dict[str, Any], essay_file: Path) -> str:
    pred = report["prediction"]
    timing = report["timing_ms"]
    lime = report["lime"]
    lines = [
        "# Deployment Performance Report",
        "",
        f"- Generated (UTC): {report['timestamp_utc']}",
        f"- Source essay: `{essay_file}`",
        f"- Word count: {report['input']['word_count']}",
        "",
        "## Model Outputs",
        f"- Production label: **{pred['label']}** (id={pred['label_id']})",
        f"- Production P(AI): {pred['production_prob_ai']:.6f}",
        f"- Production P(Human): {pred['production_prob_human']:.6f}",
        f"- Auditor P(AI): {pred['auditor_prob_ai']:.6f}",
        f"- Auditor P(Human): {pred['auditor_prob_human']:.6f}",
        "",
        "## Response Times (ms)",
        f"- Normalize + word count: {timing['normalize_and_wordcount']}",
        f"- Semantic embedding extraction: {timing['semantic_embedding_extraction']}",
        f"- Statistical feature extraction: {timing['statistical_feature_extraction']}",
        f"- Production model inference: {timing['production_model_inference']}",
        f"- Auditor model inference: {timing['auditor_model_inference']}",
        f"- LIME explanation: {timing['lime_explanation']}",
        f"- Full pipeline (`analyze_text`, with LIME): {timing['full_pipeline_analyze_text_with_lime']}",
        f"- Estimated total (manual path): {timing['estimated_total_manual_path']}",
        "",
        "## LIME Top Tokens",
        f"- LIME samples: {lime['num_samples']}",
        f"- LIME features: {lime['num_features']}",
        "",
        "| Rank | Token | Weight | Signal |",
        "|---:|---|---:|---|",
    ]
    for row in lime["top_tokens"]:
        lines.append(f"| {row['rank']} | {row['token']} | {row['weight']:.6f} | {row['signal']} |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deployment performance + LIME report from one essay.")
    parser.add_argument(
        "--essay-file",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "XGBoost Classifier" / "test_essay.txt",
        help="Path to input essay text file.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="test_essay_performance",
        help="Prefix for output files in Deployment directory.",
    )
    parser.add_argument("--lime-samples", type=int, default=8)
    parser.add_argument("--lime-features", type=int, default=12)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deployment_dir = Path(__file__).resolve().parent
    essay_text = args.essay_file.read_text(encoding="utf-8")

    report = _run_report(
        essay_text=essay_text,
        lime_samples=args.lime_samples,
        lime_features=args.lime_features,
        device=args.device,
    )

    json_path = deployment_dir / f"{args.output_prefix}.json"
    md_path = deployment_dir / f"{args.output_prefix}.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report, args.essay_file), encoding="utf-8")

    print(f"Saved JSON report: {json_path}")
    print(f"Saved Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
