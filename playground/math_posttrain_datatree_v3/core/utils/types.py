from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DatasetEntry:
    source_id: str
    name: str
    license: str
    url: str
    local_path: str
    full_local_path: str = ""
    probe_sample_rows_path: str = ""
    split: str = ""
    config: str = ""
    task_type: str = "math_reasoning"
    answer_style: str = "mixed"
    difficulty: str = "unknown"
    language: str = "en"
    quality_signals: dict[str, Any] = field(default_factory=dict)
    coverage_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload.get("full_local_path") and not payload.get("local_path"):
            payload["local_path"] = payload["full_local_path"]
        return payload


@dataclass
class DatasetManifest:
    manifest_id: str
    created_from_node: str
    search_goal: str
    datasets: list[DatasetEntry]
    coverage_tags: list[str] = field(default_factory=list)
    source_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["datasets"] = [item.to_dict() for item in self.datasets]
        return payload


@dataclass
class MathTrainingExample:
    example_id: str
    source: str
    problem: str
    solution: str
    final_answer: str
    answer_style: str
    difficulty: str = "unknown"
    task_type: str = "math_reasoning"
    instruction: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainPackManifest:
    pack_id: str
    source_datasets: list[str]
    sample_count: int
    short_answer_count: int
    long_reasoning_count: int
    dedup_rule: str
    answer_normalization_rule: str
    format: str
    output_path: str
    source_weights: dict[str, float] = field(default_factory=dict)
    coverage_tags: list[str] = field(default_factory=list)
    strategy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    status: str
    checkpoint_path: str
    recipe_path: str
    train_log_path: str
    command: str
    dry_run: bool
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalReport:
    status: str
    overall_accuracy: float
    benchmark_scores: dict[str, float]
    sample_results_path: str
    normalized_predictions_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InspectReport:
    failure_clusters: list[str]
    weak_domains: list[str]
    weak_answer_styles: list[str]
    source_effect_hypotheses: list[str]
    recommended_next_action: str
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
