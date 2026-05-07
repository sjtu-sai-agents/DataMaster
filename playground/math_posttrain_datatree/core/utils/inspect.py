from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import write_json
from .types import InspectReport

TASK_ALIGNED_COVERAGE = {
    "aime_2025": {"aime", "competition_math"},
    "gsm8k": {"gsm8k", "math_word_problem", "competition_math"},
    "human_eval": {"coding", "python", "code"},
}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        rows.append(value)
    return rows


def _extract_benchmark_feedback(eval_report: dict[str, Any]) -> dict[str, Any]:
    metadata = _safe_dict(eval_report.get("metadata"))
    feedback = metadata.get("benchmark_feedback")
    return feedback if isinstance(feedback, dict) else {}


def _has_task_aligned_coverage(benchmark_id: str, coverage_tags: set[str]) -> bool:
    expected_tags = TASK_ALIGNED_COVERAGE.get(benchmark_id, set())
    if not expected_tags:
        return False
    return bool(expected_tags & coverage_tags)


def _feedback_float(entry: dict[str, Any], key: str) -> float | None:
    value = entry.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _feedback_int(entry: dict[str, Any], key: str) -> int | None:
    value = entry.get(key)
    if isinstance(value, int):
        return value
    return None


def _analyze_benchmark_feedback(
    *,
    benchmark_feedback: dict[str, Any],
    coverage_tags: set[str],
    source_count: int,
) -> dict[str, Any] | None:
    failure_clusters: list[str] = []
    source_effect_hypotheses: list[str] = []
    rationale_parts: list[str] = []
    recommended_next_action: str | None = None

    for benchmark_id, raw_entry in benchmark_feedback.items():
        entry = _safe_dict(raw_entry)
        if not entry:
            continue
        profile = str(entry.get("benchmark_profile") or "generic")
        score = _feedback_float(entry, "score")
        num_correct = _feedback_int(entry, "num_correct")
        num_samples = _feedback_int(entry, "num_samples")
        format_rate = _feedback_float(entry, "format_adherence_rate")
        parseable_rate = _feedback_float(entry, "parseable_answer_rate")
        numeric_match_rate = _feedback_float(entry, "numeric_match_rate")
        aligned_coverage = _has_task_aligned_coverage(benchmark_id, coverage_tags)

        if parseable_rate is not None and parseable_rate < 0.8:
            recommended_next_action = "expand_black"
            failure_clusters.append("unparseable_answers")
            source_effect_hypotheses.append(
                f"{benchmark_id}: many outputs are not parseable as final answers; improve prompting or answer extraction before changing source pools."
            )
            rationale_parts.append(
                f"{benchmark_id} parseable_answer_rate={parseable_rate:.2f} indicates evaluation signal is being lost before exact-match scoring."
            )
            continue

        if profile == "sparse_exact_match" and num_samples and num_correct is not None:
            if num_correct > 0 and (parseable_rate is None or parseable_rate >= 0.9):
                recommended_next_action = "expand_black"
                failure_clusters.append("reasoning_gap")
                source_effect_hypotheses.append(
                    f"{benchmark_id}: outputs are already parseable and occasionally correct, so reward sparsity likely reflects reasoning and packing quality more than missing sources."
                )
                rationale_parts.append(
                    f"{benchmark_id} already gets {num_correct}/{num_samples} exact hits with parseable answers; iterate on mixture and strategy before forcing source expansion."
                )
                if format_rate is not None and format_rate < 0.5:
                    failure_clusters.append("format_noise")
                    source_effect_hypotheses.append(
                        f"{benchmark_id}: final-answer formatting is unstable, but because answers are parseable and sometimes correct, treat prompt cleanup as a secondary optimization rather than the main diagnosis."
                    )
                    rationale_parts.append(
                        f"{benchmark_id} format_adherence_rate={format_rate:.2f} is worth cleaning up, but it does not explain most of the lost score."
                    )
                continue

            if (
                num_correct == 0
                and aligned_coverage
                and source_count >= 2
                and (parseable_rate is None or parseable_rate >= 0.9)
            ):
                recommended_next_action = "expand_black"
                failure_clusters.append("reasoning_gap")
                source_effect_hypotheses.append(
                    f"{benchmark_id}: the current source pool is already task-aligned, but exact-match remains zero; try mixture tuning before broadening sources again."
                )
                rationale_parts.append(
                    f"{benchmark_id} has aligned coverage tags and parseable answers, so the bottleneck looks more like reasoning quality than source discovery."
                )
                if format_rate is not None and format_rate < 0.5:
                    failure_clusters.append("format_noise")
                    source_effect_hypotheses.append(
                        f"{benchmark_id}: prompt formatting is still noisy, but the main bottleneck is that task-aligned data is not yet producing exact hits."
                    )
                continue

        if format_rate is not None and format_rate < 0.5:
            recommended_next_action = "expand_black"
            failure_clusters.append("format_noise")
            source_effect_hypotheses.append(
                f"{benchmark_id}: final-answer formatting is unstable; benchmark-specific prompting or postprocessing is likely higher leverage than adding new sources."
            )
            rationale_parts.append(
                f"{benchmark_id} format_adherence_rate={format_rate:.2f} suggests an output-format mismatch."
            )
            continue

        if (
            profile == "exact_match_reasoning"
            and score is not None
            and score < 0.2
            and parseable_rate is not None
            and parseable_rate >= 0.9
            and source_count >= 2
        ):
            recommended_next_action = "expand_black"
            failure_clusters.append("mixture_tuning")
            source_effect_hypotheses.append(
                f"{benchmark_id}: answers are parseable but accuracy is still low, suggesting the current source pool needs better filtering or weighting."
            )
            rationale_parts.append(
                f"{benchmark_id} parseable answers remain low-scoring, which points to training mixture issues rather than pure format failure."
            )

        if numeric_match_rate is not None and numeric_match_rate > 0 and score is not None and score == 0:
            recommended_next_action = "expand_black"
            failure_clusters.append("normalization_gap")
            source_effect_hypotheses.append(
                f"{benchmark_id}: some extracted answers match gold values even though the official score is zero; normalization or formatting likely needs attention."
            )
            rationale_parts.append(
                f"{benchmark_id} numeric_match_rate={numeric_match_rate:.2f} exceeds the official score, which points to scoring-surface mismatch."
            )

    if recommended_next_action is None:
        return None

    return {
        "recommended_next_action": recommended_next_action,
        "failure_clusters": _dedupe_keep_order(failure_clusters),
        "source_effect_hypotheses": _dedupe_keep_order(source_effect_hypotheses),
        "rationale": " ".join(rationale_parts).strip(),
    }


def run_inspect(
    *,
    eval_report: dict,
    pack_manifest: dict,
    pack_stats: dict,
    output_path: str | Path,
) -> InspectReport:
    scores = eval_report.get("benchmark_scores", {}) or {}
    overall = float(eval_report.get("overall_accuracy", 0.0) or 0.0)
    benchmark_feedback = _extract_benchmark_feedback(eval_report)
    style_dist = pack_stats.get("style_distribution", {}) or {}
    duplicate_rate = float(pack_stats.get("duplicate_rate", 0.0) or 0.0)
    source_count = int(pack_stats.get("source_count", 0) or 0)
    coverage_tags = set(pack_manifest.get("coverage_tags", []))

    weak_domains = [name for name, score in scores.items() if score + 0.05 < overall]
    weak_answer_styles: list[str] = []
    short_count = int(style_dist.get("short_answer", 0))
    long_count = int(style_dist.get("long_reasoning", 0))
    total = max(short_count + long_count, 1)
    if short_count / total < 0.25:
        weak_answer_styles.append("short_answer")
    if long_count / total < 0.25:
        weak_answer_styles.append("long_reasoning")

    source_effect_hypotheses: list[str] = []
    failure_clusters: list[str] = []
    recommended_next_action = "expand_black"
    rationale = "Current issues look more like cleaning, mixture, or formatting problems."

    feedback_analysis = _analyze_benchmark_feedback(
        benchmark_feedback=benchmark_feedback,
        coverage_tags=coverage_tags,
        source_count=source_count,
    )
    if feedback_analysis is not None:
        recommended_next_action = str(feedback_analysis.get("recommended_next_action") or recommended_next_action)
        rationale = str(feedback_analysis.get("rationale") or rationale)
        failure_clusters.extend(feedback_analysis.get("failure_clusters") or [])
        source_effect_hypotheses.extend(feedback_analysis.get("source_effect_hypotheses") or [])

        if duplicate_rate > 0.30 or weak_answer_styles:
            source_effect_hypotheses.append("Tune dedup, weighting, and short/long reasoning mixture.")
            failure_clusters.extend(weak_answer_styles or ["mixture_tuning"])
            rationale = (
                rationale + " Packing quality or answer-style balance still looks improvable within the current source pool."
            ).strip()
    else:
        if source_count < 3 or (weak_domains and len(coverage_tags) < 2):
            recommended_next_action = "expand_red"
            rationale = "Weak benchmark coverage suggests missing task/domain coverage in the source pool."
            source_effect_hypotheses.append("Need broader public data coverage for weak domains.")
            failure_clusters.extend(weak_domains or ["coverage_gap"])
        elif overall > 0.20 and duplicate_rate > 0.50:
            recommended_next_action = "expand_red"
            rationale = "High accuracy with high duplicate rate suggests diminishing returns; need diverse sources."
            source_effect_hypotheses.append("High dedup rate indicates data source overlap; seek new sources.")
            failure_clusters.append("source_diversity")
        elif duplicate_rate > 0.30 or weak_answer_styles:
            recommended_next_action = "expand_black"
            rationale = "Current source pool exists, but packing quality and answer-style balance need adjustment."
            source_effect_hypotheses.append("Tune dedup, weighting, and short/long reasoning mixture.")
            failure_clusters.extend(weak_answer_styles or ["format_noise"])
        else:
            source_effect_hypotheses.append("Refine mixture weights before searching for more sources.")
            failure_clusters.extend(weak_domains or ["mixture_tuning"])

    report = InspectReport(
        failure_clusters=_dedupe_keep_order(failure_clusters),
        weak_domains=_dedupe_keep_order(weak_domains),
        weak_answer_styles=_dedupe_keep_order(weak_answer_styles),
        source_effect_hypotheses=_dedupe_keep_order(source_effect_hypotheses),
        recommended_next_action=recommended_next_action,
        rationale=rationale,
    )
    write_json(output_path, report.to_dict())
    return report
