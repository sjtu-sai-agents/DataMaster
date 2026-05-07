from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from playground.math_posttrain_datatree.core.utils.eval import run_eval
from playground.math_posttrain_datatree.core.utils.io import write_jsonl


def build_smoke_inputs(workspace: Path) -> tuple[Path, Path]:
    benchmark_path = workspace / "aime_2025_smoke.jsonl"
    prediction_path = workspace / "aime_2025_predictions.jsonl"
    write_jsonl(
        benchmark_path,
        [
            {
                "id": "aime_2025_1",
                "problem": "Compute 12^2 + 10^2.",
                "answer": r"\boxed{244}",
            },
            {
                "id": "aime_2025_2",
                "problem": "Find 7 + 8.",
                "answer": "15",
            },
            {
                "id": "aime_2025_3",
                "problem": "What is 3 \\cdot 9?",
                "final_answer": "27",
            },
        ],
    )
    write_jsonl(
        prediction_path,
        [
            {
                "id": "aime_2025_1",
                "response": "12^2 = 144 and 10^2 = 100, so the total is 244. Final answer: 244",
            },
            {
                "id": "aime_2025_2",
                "prediction": r"I get \boxed{14}.",
            },
            {
                "id": "aime_2025_3",
                "output": "Multiply to get 27.",
            },
        ],
    )
    return benchmark_path, prediction_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real AIME evaluation smoke test.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Optional workspace directory for benchmark, predictions, and eval outputs.",
    )
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=None,
        help="Optional AIME benchmark file (.json or .jsonl). If set, use it instead of built-in smoke samples.",
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=None,
        help="Optional prediction file (.json or .jsonl). Must be provided together with --benchmark-file.",
    )
    args = parser.parse_args()

    if (args.benchmark_file is None) != (args.prediction_file is None):
        parser.error("--benchmark-file and --prediction-file must be provided together.")

    if args.workspace is None:
        with tempfile.TemporaryDirectory(prefix="aime_eval_smoke_") as tmpdir:
            result = run_smoke(
                Path(tmpdir),
                benchmark_path=args.benchmark_file,
                prediction_path=args.prediction_file,
            )
    else:
        args.workspace.mkdir(parents=True, exist_ok=True)
        result = run_smoke(
            args.workspace,
            benchmark_path=args.benchmark_file,
            prediction_path=args.prediction_file,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_smoke(
    workspace: Path,
    *,
    benchmark_path: Path | None = None,
    prediction_path: Path | None = None,
) -> dict[str, object]:
    if benchmark_path is None or prediction_path is None:
        benchmark_path, prediction_path = build_smoke_inputs(workspace)
    else:
        benchmark_path = benchmark_path.resolve()
        prediction_path = prediction_path.resolve()
    eval_dir = workspace / "eval"
    report = run_eval(
        eval_dir=eval_dir,
        benchmark_suite=["aime_2025"],
        pack_manifest={"sample_count": 3, "coverage_tags": ["aime", "competition_math"]},
        pack_stats={
            "style_distribution": {"short_answer": 2, "long_reasoning": 1},
            "duplicate_rate": 0.0,
        },
        benchmark_files={"aime_2025": str(benchmark_path)},
        prediction_files={"aime_2025": str(prediction_path)},
    )
    return {
        "status": report.status,
        "overall_accuracy": report.overall_accuracy,
        "benchmark_scores": report.benchmark_scores,
        "sample_results_path": report.sample_results_path,
        "normalized_predictions_path": report.normalized_predictions_path,
        "eval_report_path": str(eval_dir / "eval_report.json"),
        "benchmark_path": str(benchmark_path),
        "prediction_path": str(prediction_path),
    }


if __name__ == "__main__":
    main()
