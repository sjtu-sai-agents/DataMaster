from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import requests

import playground.math_posttrain_datatree.core.exp.black_exp as black_exp_module
from playground.math_posttrain_datatree.core.exp.black_exp import BlackExp
from playground.math_posttrain_datatree.core.exp.red_exp import RedExp
from playground.math_posttrain_datatree.core.utils import data as data_utils
from playground.math_posttrain_datatree.core.utils.data import (
    apply_transform_script,
    build_train_pack,
    materialize_dataset_entry,
    normalize_final_answer,
)
from playground.math_posttrain_datatree.core.utils.types import MathTrainingExample
from playground.math_posttrain_datatree.core.utils.eval import (
    resolve_posttrainbench_task_dir,
    run_posttrainbench_eval,
    run_eval,
    score_answer_predictions,
)
from playground.math_posttrain_datatree.core.utils.inspect import run_inspect
from playground.math_posttrain_datatree.core.utils.io import write_jsonl
from playground.math_posttrain_datatree.core.utils.llama_factory import (
    _resolve_env_bin_dir,
    create_dataset_registry,
    ensure_llamafactory_available,
    run_llama_factory_sft,
    resolve_llamafactory_command,
    validate_alpaca_dataset,
)
from playground.math_posttrain_datatree.core.utils.tree_helpers import write_uct_trajectory
from playground.math_posttrain_datatree.core.utils.types import EvalReport, InspectReport, TrainPackManifest, TrainResult
from playground.math_posttrain_datatree.core.utils.uct import MetricReview, UCTSearchConfig, UCTSearchManager


def test_normalize_final_answer_handles_boxed_and_prefixes() -> None:
    assert normalize_final_answer(r"\boxed{42}") == "42"
    assert normalize_final_answer("Final answer: 17.") == "17"
    assert normalize_final_answer("The answer is 3/5") == "3/5"


def test_build_train_pack_exports_llama_factory_alpaca(tmp_path: Path) -> None:
    dataset_path = tmp_path / "aime_public.jsonl"
    write_jsonl(
        dataset_path,
        [
            {
                "problem": "What is 1+1?",
                "solution": "We compute 1+1=2.\n\nFinal answer: 2",
                "final_answer": "2",
                "difficulty": "easy",
            },
            {
                "problem": "What is 2+2?",
                "solution": "2+2=4",
                "final_answer": "4",
            },
        ],
    )
    manifest, stats, alpaca_path = build_train_pack(
        [
            {
                "source_id": "aime_public",
                "local_path": str(dataset_path),
                "coverage_tags": ["aime", "competition_math"],
            }
        ],
        tmp_path / "pack",
        pack_id="pack_1",
        max_samples=8,
        short_answer_ratio=0.5,
    )
    ok, _ = validate_alpaca_dataset(alpaca_path)
    assert ok
    assert manifest.format == "alpaca"
    assert stats["sample_count"] == 2
    assert manifest.short_answer_count + manifest.long_reasoning_count == 2


def test_build_train_pack_materializes_remote_dataset(monkeypatch, tmp_path: Path) -> None:
    remote_cache = tmp_path / "remote_cache.jsonl"
    write_jsonl(
        remote_cache,
        [
            {
                "problem": "Compute 3+4.",
                "solution": "3+4=7. Final answer: 7",
                "final_answer": "7",
            }
        ],
    )

    def fake_materialize(entry: dict, cache_dir: Path, *, max_rows: int = 2048) -> str:
        return str(remote_cache)

    monkeypatch.setattr(data_utils, "materialize_dataset_entry", fake_materialize)
    manifest, stats, alpaca_path = build_train_pack(
        [
            {
                "source_id": "EleutherAI/hendrycks_math",
                "url": "https://huggingface.co/datasets/EleutherAI/hendrycks_math",
                "local_path": "",
                "coverage_tags": ["competition_math"],
            }
        ],
        tmp_path / "pack_remote",
        pack_id="pack_remote",
        max_samples=8,
        short_answer_ratio=0.5,
    )
    assert alpaca_path.exists()
    assert manifest.sample_count == 1
    assert stats["sample_count"] == 1


def test_build_train_pack_caps_remote_materialization_budget(monkeypatch, tmp_path: Path) -> None:
    remote_cache = tmp_path / "weighted_remote_cache.jsonl"
    write_jsonl(
        remote_cache,
        [
            {
                "problem": "Compute 6+7.",
                "solution": "6+7=13. Final answer: 13",
                "final_answer": "13",
                "difficulty": "hard",
            }
        ],
    )
    captured: dict[str, int] = {}

    def fake_materialize(entry: dict, cache_dir: Path, *, max_rows: int = 2048) -> str:
        captured["max_rows"] = max_rows
        return str(remote_cache)

    monkeypatch.setattr(data_utils, "materialize_dataset_entry", fake_materialize)
    build_train_pack(
        [
            {
                "source_id": "qwedsacf/competition_math",
                "url": "https://huggingface.co/datasets/qwedsacf/competition_math",
                "local_path": "",
                "coverage_tags": ["competition_math"],
            }
        ],
        tmp_path / "pack_weighted_remote",
        pack_id="pack_weighted_remote",
        max_samples=40_000,
        short_answer_ratio=0.3,
        source_weights={"qwedsacf/competition_math": 3.0},
        processing_config={"max_examples_per_source": 15_000},
    )
    assert captured["max_rows"] == 5336


def test_build_train_pack_applies_processing_config_filters(tmp_path: Path) -> None:
    dataset_path = tmp_path / "math_mix.jsonl"
    write_jsonl(
        dataset_path,
        [
            {
                "problem": "In geometry, find the area of a triangle with base 4 and height 3.",
                "solution": "Use area = 1/2 * base * height, so the area is 6 square units.\n\nFinal answer: 6",
                "final_answer": "6",
                "difficulty": "hard",
                "topic": "geometry",
            },
            {
                "problem": "In geometry, find the area of a triangle with base 4 and height 3.",
                "solution": "Area is 6.",
                "final_answer": "6",
                "difficulty": "hard",
                "topic": "geometry",
            },
            {
                "problem": "Solve x+2=7.",
                "solution": "x=5. Final answer: 5",
                "final_answer": "5",
                "difficulty": "easy",
                "topic": "algebra",
            },
        ],
    )
    manifest, stats, alpaca_path = build_train_pack(
        [
            {
                "source_id": "mixed_math",
                "local_path": str(dataset_path),
                "coverage_tags": ["competition_math"],
            }
        ],
        tmp_path / "pack_filtered",
        pack_id="pack_filtered",
        max_samples=8,
        short_answer_ratio=0.1,
        processing_config={
            "topic_allowlist": ["geometry"],
            "difficulty_allowlist": ["hard"],
            "answer_style_allowlist": ["long_reasoning"],
            "dedup_keep_mode": "long_only",
            "min_problem_chars": 20,
        },
    )
    rows = [json.loads(line) for line in alpaca_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert manifest.sample_count == 1
    assert stats["sample_count"] == 1
    assert stats["filtered_source_distribution"]["mixed_math"] == 1
    assert manifest.strategy["processing_config"]["dedup_keep_mode"] == "long_only"
    assert "triangle" in rows[0]["input"].lower()
    assert "final answer: 6" in rows[0]["output"].lower()


def test_apply_transform_script_modifies_and_filters_examples(tmp_path: Path) -> None:
    script_path = tmp_path / "node_a" / "transform.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        """
def transform(example):
    if "drop me" in example["problem"].lower():
        return None
    example["problem"] = example["problem"] + " [cleaned]"
    example["solution"] = "Transformed solution\\n\\nFinal answer: " + example["final_answer"]
    example["metadata"]["topic"] = "cleaned_topic"
    return example
""".strip(),
        encoding="utf-8",
    )
    examples = [
        MathTrainingExample(
            example_id="a",
            source="unit",
            problem="Keep me",
            solution="Original solution",
            final_answer="7",
            answer_style="short_answer",
            metadata={"topic": "raw"},
        ),
        MathTrainingExample(
            example_id="b",
            source="unit",
            problem="Drop me",
            solution="Original solution",
            final_answer="9",
            answer_style="short_answer",
            metadata={"topic": "raw"},
        ),
    ]
    transformed = apply_transform_script(examples, script_path)
    assert len(transformed) == 1
    assert transformed[0].problem.endswith("[cleaned]")
    assert transformed[0].metadata["topic"] == "cleaned_topic"
    assert transformed[0].answer_style == "long_reasoning"


def test_apply_transform_script_uses_node_specific_paths_without_collision(tmp_path: Path) -> None:
    script_a = tmp_path / "black_a" / "transform.py"
    script_b = tmp_path / "black_b" / "transform.py"
    script_a.parent.mkdir(parents=True, exist_ok=True)
    script_b.parent.mkdir(parents=True, exist_ok=True)
    script_a.write_text(
        """
def transform(example):
    example["problem"] = "A:" + example["problem"]
    return example
""".strip(),
        encoding="utf-8",
    )
    script_b.write_text(
        """
def transform(example):
    example["problem"] = "B:" + example["problem"]
    return example
""".strip(),
        encoding="utf-8",
    )
    sample = [
        MathTrainingExample(
            example_id="x",
            source="unit",
            problem="same-name script",
            solution="Final answer: 1",
            final_answer="1",
            answer_style="short_answer",
            metadata={},
        )
    ]
    out_a = apply_transform_script(sample, script_a)
    out_b = apply_transform_script(sample, script_b)
    assert out_a[0].problem.startswith("A:")
    assert out_b[0].problem.startswith("B:")


def test_materialize_dataset_entry_prefers_manifest_split(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def fake_server_get(*args, **kwargs):
        raise requests.RequestException("server unavailable")

    def fake_get_dataset_config_names(dataset_id: str) -> list[str]:
        assert dataset_id == "nvidia/OpenMathInstruct-2"
        return []

    def fake_get_dataset_split_names(dataset_id: str) -> list[str]:
        assert dataset_id == "nvidia/OpenMathInstruct-2"
        return ["train_1M", "train", "train_5M"]

    class FakeDataset(list):
        pass

    def fake_load_dataset(dataset_id: str, *, split: str):
        calls.append((dataset_id, split))
        return FakeDataset(
            [
                {
                    "problem": "What is 5+6?",
                    "generated_solution": "5+6=11",
                    "expected_answer": "11",
                }
            ]
        )

    monkeypatch.setattr(data_utils.requests, "get", fake_server_get)
    monkeypatch.setattr("datasets.get_dataset_config_names", fake_get_dataset_config_names)
    monkeypatch.setattr("datasets.get_dataset_split_names", fake_get_dataset_split_names)
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    local_path = materialize_dataset_entry(
        {
            "source_id": "nvidia/OpenMathInstruct-2",
            "url": "https://huggingface.co/datasets/nvidia/OpenMathInstruct-2",
            "split": "train_1M",
        },
        tmp_path / "cache",
        max_rows=32,
    )
    assert Path(local_path).exists()
    assert calls[0] == ("nvidia/OpenMathInstruct-2", "train_1M[:32]")


def test_materialize_dataset_entry_prefers_datasets_server(monkeypatch, tmp_path: Path) -> None:
    class FakeResponse:
        def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"http {self.status_code}")

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(url: str, params: dict[str, object], timeout: int):
        if url.endswith("/splits"):
            assert params["dataset"] == "gneubig/aime-1983-2024"
            return FakeResponse(
                {
                    "splits": [
                        {
                            "dataset": "gneubig/aime-1983-2024",
                            "config": "default",
                            "split": "train",
                        }
                    ]
                }
            )
        if url.endswith("/rows"):
            assert params["dataset"] == "gneubig/aime-1983-2024"
            assert params["config"] == "default"
            assert params["split"] == "train"
            return FakeResponse(
                {
                    "rows": [
                        {
                            "row": {
                                "Question": "What is 6+7?",
                                "Answer": "13",
                            }
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(data_utils.requests, "get", fake_get)
    local_path = materialize_dataset_entry(
        {
            "source_id": "gneubig/aime-1983-2024",
            "url": "https://huggingface.co/datasets/gneubig/aime-1983-2024",
        },
        tmp_path / "cache",
        max_rows=8,
    )
    rows = [json.loads(line) for line in Path(local_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["problem"] == "What is 6+7?"
    assert rows[0]["final_answer"] == "13"


def test_materialize_dataset_entry_recovers_when_datasets_server_splits_fails(monkeypatch, tmp_path: Path) -> None:
    class FakeResponse:
        def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"http {self.status_code}")

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(url: str, params: dict[str, object], timeout: int):
        if url.endswith("/splits"):
            raise requests.RequestException("connection reset")
        if url.endswith("/rows"):
            assert params["dataset"] == "gneubig/aime-1983-2024"
            assert params["config"] == "default"
            assert params["split"] == "train"
            return FakeResponse(
                {
                    "rows": [
                        {
                            "row": {
                                "Question": "What is 8+9?",
                                "Answer": "17",
                            }
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected url {url}")

    def fail_load_dataset(*args, **kwargs):
        raise AssertionError("load_dataset should not be called when direct rows fallback succeeds")

    monkeypatch.setattr(data_utils.requests, "get", fake_get)
    monkeypatch.setattr("datasets.load_dataset", fail_load_dataset)
    local_path = materialize_dataset_entry(
        {
            "source_id": "gneubig/aime-1983-2024",
            "url": "https://huggingface.co/datasets/gneubig/aime-1983-2024",
        },
        tmp_path / "cache",
        max_rows=8,
    )
    rows = [json.loads(line) for line in Path(local_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["problem"] == "What is 8+9?"
    assert rows[0]["final_answer"] == "17"


def test_materialize_dataset_entry_retries_datasets_server_splits(monkeypatch, tmp_path: Path) -> None:
    class FakeResponse:
        def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"http {self.status_code}")

        def json(self) -> dict[str, object]:
            return self._payload

    calls = {"splits": 0, "rows": 0}

    def fake_get(url: str, params: dict[str, object], timeout: int):
        if url.endswith("/splits"):
            calls["splits"] += 1
            if calls["splits"] < 3:
                raise requests.RequestException("temporary TLS failure")
            return FakeResponse(
                {
                    "splits": [
                        {
                            "dataset": "gneubig/aime-1983-2024",
                            "config": "default",
                            "split": "train",
                        }
                    ]
                }
            )
        if url.endswith("/rows"):
            calls["rows"] += 1
            if int(params.get("offset", 0)) > 0:
                return FakeResponse({"rows": []})
            return FakeResponse(
                {
                    "rows": [
                        {
                            "row": {
                                "Question": "What is 4+5?",
                                "Answer": "9",
                            }
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected url {url}")

    def fail_load_dataset(*args, **kwargs):
        raise AssertionError("load_dataset should not be called when datasets-server retries recover")

    monkeypatch.setattr(data_utils.requests, "get", fake_get)
    monkeypatch.setattr(data_utils.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("datasets.load_dataset", fail_load_dataset)
    local_path = materialize_dataset_entry(
        {
            "source_id": "gneubig/aime-1983-2024",
            "url": "https://huggingface.co/datasets/gneubig/aime-1983-2024",
        },
        tmp_path / "cache",
        max_rows=8,
    )
    rows = [json.loads(line) for line in Path(local_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert calls["splits"] == 3
    assert calls["rows"] == 2
    assert rows[0]["final_answer"] == "9"


def test_materialize_dataset_entry_retries_datasets_server_rows(monkeypatch, tmp_path: Path) -> None:
    class FakeResponse:
        def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"http {self.status_code}")

        def json(self) -> dict[str, object]:
            return self._payload

    calls = {"rows": 0}

    def fake_get(url: str, params: dict[str, object], timeout: int):
        if url.endswith("/splits"):
            return FakeResponse(
                {
                    "splits": [
                        {
                            "dataset": "gneubig/aime-1983-2024",
                            "config": "default",
                            "split": "train",
                        }
                    ]
                }
            )
        if url.endswith("/rows"):
            calls["rows"] += 1
            if int(params.get("offset", 0)) > 0:
                return FakeResponse({"rows": []})
            if calls["rows"] < 3:
                raise requests.RequestException("temporary row fetch failure")
            return FakeResponse(
                {
                    "rows": [
                        {
                            "row": {
                                "Question": "What is 12+1?",
                                "Answer": "13",
                            }
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected url {url}")

    def fail_load_dataset(*args, **kwargs):
        raise AssertionError("load_dataset should not be called when row retries recover")

    monkeypatch.setattr(data_utils.requests, "get", fake_get)
    monkeypatch.setattr(data_utils.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("datasets.load_dataset", fail_load_dataset)
    local_path = materialize_dataset_entry(
        {
            "source_id": "gneubig/aime-1983-2024",
            "url": "https://huggingface.co/datasets/gneubig/aime-1983-2024",
        },
        tmp_path / "cache",
        max_rows=8,
    )
    rows = [json.loads(line) for line in Path(local_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert calls["rows"] == 4
    assert rows[0]["final_answer"] == "13"


def test_materialize_dataset_entry_can_skip_datasets_server(monkeypatch, tmp_path: Path) -> None:
    def fail_materialize_via_server(*args, **kwargs):
        raise AssertionError("datasets-server should be skipped when explicitly disabled")

    monkeypatch.setenv("MATH_PT_DISABLE_DATASETS_SERVER", "1")
    monkeypatch.setattr(data_utils, "_materialize_via_datasets_server", fail_materialize_via_server)
    monkeypatch.setattr("datasets.get_dataset_config_names", lambda _dataset_id: [])
    monkeypatch.setattr("datasets.get_dataset_split_names", lambda **kwargs: ["train"])
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *args, **kwargs: [{"Question": "What is 10+4?", "Answer": "14"}],
    )

    local_path = materialize_dataset_entry(
        {
            "source_id": "gneubig/aime-1983-2024",
            "url": "https://huggingface.co/datasets/gneubig/aime-1983-2024",
        },
        tmp_path / "cache",
        max_rows=8,
    )
    rows = [json.loads(line) for line in Path(local_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["final_answer"] == "14"


def test_materialize_dataset_entry_skips_test_only_dataset_without_explicit_split(monkeypatch, tmp_path: Path) -> None:
    class FakeResponse:
        def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"http {self.status_code}")

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(url: str, params: dict[str, object], timeout: int):
        if url.endswith("/splits"):
            return FakeResponse(
                {
                    "splits": [
                        {
                            "dataset": "opencompass/AIME2025",
                            "config": "AIME2025-I",
                            "split": "test",
                        },
                        {
                            "dataset": "opencompass/AIME2025",
                            "config": "AIME2025-II",
                            "split": "test",
                        },
                    ]
                }
            )
        raise AssertionError("rows endpoint should not be called for implicit test-only data")

    monkeypatch.setattr(data_utils.requests, "get", fake_get)
    local_path = materialize_dataset_entry(
        {
            "source_id": "opencompass/AIME2025",
            "url": "https://huggingface.co/datasets/opencompass/AIME2025",
            "split": "",
        },
        tmp_path / "cache",
        max_rows=8,
    )
    assert Path(local_path).exists()
    assert Path(local_path).read_text(encoding="utf-8") == ""


def test_materialize_dataset_entry_uses_custom_training_split_from_datasets_server(monkeypatch, tmp_path: Path) -> None:
    class FakeResponse:
        def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
            self._payload = payload
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"http {self.status_code}")

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_get(url: str, params: dict[str, object], timeout: int):
        if url.endswith("/splits"):
            return FakeResponse(
                {
                    "splits": [
                        {"dataset": "nvidia/OpenMathReasoning", "config": "default", "split": "cot"},
                        {"dataset": "nvidia/OpenMathReasoning", "config": "default", "split": "tir"},
                        {"dataset": "nvidia/OpenMathReasoning", "config": "default", "split": "genselect"},
                    ]
                }
            )
        if url.endswith("/rows"):
            assert params["dataset"] == "nvidia/OpenMathReasoning"
            assert params["config"] == "default"
            assert params["split"] == "cot"
            return FakeResponse(
                {
                    "rows": [
                        {
                            "row": {
                                "problem": "What is 10+5?",
                                "generated_solution": "10+5=15",
                                "expected_answer": "15",
                            }
                        }
                    ]
                }
            )
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(data_utils.requests, "get", fake_get)
    local_path = materialize_dataset_entry(
        {
            "source_id": "nvidia/OpenMathReasoning",
            "url": "https://huggingface.co/datasets/nvidia/OpenMathReasoning",
        },
        tmp_path / "cache",
        max_rows=8,
    )
    rows = [json.loads(line) for line in Path(local_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["problem"] == "What is 10+5?"
    assert rows[0]["final_answer"] == "15"


def test_materialize_dataset_entry_recovers_from_missing_default_config(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[tuple[str, ...], str]] = []

    def fake_server_get(*args, **kwargs):
        raise requests.RequestException("server unavailable")

    def fake_get_dataset_config_names(dataset_id: str) -> list[str]:
        assert dataset_id == "EleutherAI/hendrycks_math"
        raise RuntimeError("transient metadata failure")

    def fake_get_dataset_split_names(path: str, config_name: str | None = None) -> list[str]:
        assert path == "EleutherAI/hendrycks_math"
        if config_name:
            return ["train"]
        return []

    class FakeDataset(list):
        pass

    def fake_load_dataset(*args: str, split: str):
        calls.append((args, split))
        if args == ("EleutherAI/hendrycks_math",):
            raise ValueError(
                "Couldn't find cache for EleutherAI/hendrycks_math for config 'default'. "
                "Available configs in the cache: ['algebra', 'geometry']"
            )
        config_name = args[1]
        return FakeDataset(
            [
                {
                    "problem": f"What topic is {config_name}?",
                    "solution": f"Topic is {config_name}. Final answer: {config_name}",
                    "final_answer": config_name,
                }
            ]
        )

    monkeypatch.setattr(data_utils.requests, "get", fake_server_get)
    monkeypatch.setattr("datasets.get_dataset_config_names", fake_get_dataset_config_names)
    monkeypatch.setattr("datasets.get_dataset_split_names", fake_get_dataset_split_names)
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    local_path = materialize_dataset_entry(
        {
            "source_id": "EleutherAI/hendrycks_math",
            "url": "https://huggingface.co/datasets/EleutherAI/hendrycks_math",
        },
        tmp_path / "cache",
        max_rows=8,
    )
    rows = [line for line in Path(local_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert Path(local_path).exists()
    parsed_rows = [json.loads(line) for line in rows]
    assert len(parsed_rows) == 2
    assert {row["dataset_config"] for row in parsed_rows} == {"algebra", "geometry"}
    assert calls[0] == (("EleutherAI/hendrycks_math",), "train[:8]")


def test_build_train_pack_skips_zero_weight_sources(monkeypatch, tmp_path: Path) -> None:
    gsm_path = tmp_path / "gsm.jsonl"
    write_jsonl(
        gsm_path,
        [
            {
                "problem": "What is 8+9?",
                "solution": "8+9=17. Final answer: 17",
                "final_answer": "17",
            }
        ],
    )
    calls: list[str] = []

    def fake_materialize(entry: dict, cache_dir: Path, *, max_rows: int = 2048) -> str:
        source_id = str(entry.get("source_id"))
        calls.append(source_id)
        if source_id == "openai/gsm8k":
            return str(gsm_path)
        raise AssertionError(f"zero-weight source should not materialize: {source_id}")

    monkeypatch.setattr(data_utils, "materialize_dataset_entry", fake_materialize)
    manifest, stats, _ = build_train_pack(
        [
            {"source_id": "opencompass/AIME2025", "local_path": "", "coverage_tags": ["aime"]},
            {"source_id": "openai/gsm8k", "local_path": "", "coverage_tags": ["gsm8k"]},
        ],
        tmp_path / "pack_skip_zero",
        pack_id="pack_skip_zero",
        max_samples=8,
        short_answer_ratio=0.5,
        source_weights={"opencompass/AIME2025": 0.0, "openai/gsm8k": 1.0},
    )
    assert calls == ["openai/gsm8k"]
    assert manifest.sample_count == 1
    assert stats["sample_count"] == 1


def test_black_agent_strategy_uses_json_block_instead_of_finish_message(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"datasets":[]}', encoding="utf-8")

    class FakeAgent:
        def __init__(self) -> None:
            self._prompt_format_kwargs = {}
            self.config = SimpleNamespace(max_turns=8)

        def run(self, task, **kwargs):
            return {
                "dialogs": [
                    {
                        "messages": [
                            {
                                "role": "assistant",
                                "content": (
                                    "```json\n"
                                    "{\n"
                                    '  "max_samples": 512,\n'
                                    '  "processing_config": {\n'
                                    '    "dedup_keep_mode": "long_only",\n'
                                    '    "topic_allowlist": ["geometry"]\n'
                                    "  }\n"
                                    "}\n"
                                    "```"
                                ),
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "finish",
                                            "arguments": json.dumps(
                                                {
                                                    "message": "Used a different strategy but summarized in prose.",
                                                    "task_completed": True,
                                                }
                                            ),
                                        }
                                    }
                                ],
                            }
                        ]
                    }
                ]
            }

    exp = BlackExp(
        agent=FakeAgent(),
        session=None,
        workspace=tmp_path,
        task_workspace=tmp_path,
        config=SimpleNamespace(),
        node=SimpleNamespace(id="black_node", parent=None),
        manifest_path=manifest_path,
        inspect_report_path=None,
    )
    strategy = exp._agent_strategy(
        "Collect and clean math data.",
        {
            "max_samples": 384,
            "short_answer_ratio": 0.5,
            "source_weights": {},
            "processing_config": {"dedup_keep_mode": "short_and_long"},
            "hparam_overrides": {"num_train_epochs": 1, "learning_rate": 1e-4, "max_samples": 384},
        },
    )
    assert strategy["max_samples"] == 512
    assert strategy["processing_config"]["dedup_keep_mode"] == "long_only"
    assert strategy["processing_config"]["topic_allowlist"] == ["geometry"]


def test_black_agent_strategy_uses_strategy_json_file_when_agent_times_out(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"datasets":[]}', encoding="utf-8")
    strategy_path = tmp_path / "artifacts" / "train_packs" / "black_node" / "strategy.json"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        json.dumps(
            {
                "max_samples": 512,
                "processing_config": {
                    "topic_allowlist": ["geometry"],
                    "dedup_keep_mode": "long_only",
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeAgent:
        def __init__(self) -> None:
            self._prompt_format_kwargs = {}
            self.config = SimpleNamespace(max_turns=8)
            self.calls = 0

        def run(self, task, **kwargs):
            self.calls += 1
            return SimpleNamespace(dialogs=[], status="failed", result={"reason": "max_turns_exceeded"})

    agent = FakeAgent()
    exp = BlackExp(
        agent=agent,
        session=None,
        workspace=tmp_path,
        task_workspace=tmp_path,
        config=SimpleNamespace(),
        node=SimpleNamespace(id="black_node", parent=None),
        manifest_path=manifest_path,
        inspect_report_path=None,
    )
    strategy = exp._agent_strategy(
        "Collect and clean math data.",
        {
            "max_samples": 384,
            "short_answer_ratio": 0.5,
            "source_weights": {},
            "processing_config": {"dedup_keep_mode": "short_and_long"},
            "hparam_overrides": {"num_train_epochs": 1, "learning_rate": 1e-4, "max_samples": 384},
        },
    )
    assert agent.calls == 1
    assert strategy["max_samples"] == 512
    assert strategy["processing_config"]["dedup_keep_mode"] == "long_only"
    assert strategy["processing_config"]["topic_allowlist"] == ["geometry"]


def test_black_agent_strategy_retries_short_rescue_after_max_turns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"datasets":[]}', encoding="utf-8")

    class FakeAgent:
        def __init__(self) -> None:
            self._prompt_format_kwargs = {}
            self.config = SimpleNamespace(max_turns=8)
            self.calls = 0

        def run(self, task, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(dialogs=[], status="failed", result={"reason": "max_turns_exceeded"})
            return {
                "dialogs": [
                    {
                        "messages": [
                            {
                                "role": "assistant",
                                "content": (
                                    "```json\n"
                                    "{\n"
                                    '  "max_samples": 448,\n'
                                    '  "processing_config": {\n'
                                    '    "min_solution_chars": 120\n'
                                    "  }\n"
                                    "}\n"
                                    "```"
                                ),
                            }
                        ]
                    }
                ]
            }

    agent = FakeAgent()
    exp = BlackExp(
        agent=agent,
        session=None,
        workspace=tmp_path,
        task_workspace=tmp_path,
        config=SimpleNamespace(),
        node=SimpleNamespace(id="black_retry_node", parent=None),
        manifest_path=manifest_path,
        inspect_report_path=None,
    )
    strategy = exp._agent_strategy(
        "Collect and clean math data.",
        {
            "max_samples": 384,
            "short_answer_ratio": 0.5,
            "source_weights": {},
            "processing_config": {"dedup_keep_mode": "short_and_long", "min_solution_chars": 0},
            "hparam_overrides": {"num_train_epochs": 1, "learning_rate": 1e-4, "max_samples": 384},
        },
    )
    assert agent.calls == 2
    assert agent.config.max_turns == 8
    assert strategy["max_samples"] == 448
    assert strategy["processing_config"]["min_solution_chars"] == 120


def test_black_run_retries_empty_pack_once_with_relaxed_strategy(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "source_id": "qwedsacf/competition_math",
                        "local_path": "",
                        "coverage_tags": ["competition_math"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    alpaca_path = tmp_path / "alpaca.jsonl"
    alpaca_path.write_text('{"instruction":"Solve","input":"1+1","output":"2"}\n', encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_build_train_pack(
        dataset_entries,
        pack_dir,
        *,
        pack_id,
        max_samples,
        short_answer_ratio,
        source_weights,
        processing_config,
        transform_script_path,
    ):
        calls.append(
            {
                "processing_config": dict(processing_config),
                "transform_script_path": str(transform_script_path) if transform_script_path else None,
            }
        )
        sample_count = 0 if len(calls) == 1 else 2
        manifest = TrainPackManifest(
            pack_id=pack_id,
            source_datasets=["qwedsacf/competition_math"],
            sample_count=sample_count,
            short_answer_count=sample_count,
            long_reasoning_count=0,
            dedup_rule="normalized_problem",
            answer_normalization_rule="math_final_answer",
            format="alpaca",
            output_path=str(alpaca_path),
            strategy={"processing_config": dict(processing_config)},
        )
        return manifest, {"sample_count": sample_count}, alpaca_path

    monkeypatch.setattr(black_exp_module, "build_train_pack", fake_build_train_pack)
    monkeypatch.setattr(black_exp_module, "validate_alpaca_dataset", lambda path: (True, ""))
    monkeypatch.setattr(
        black_exp_module,
        "run_llama_factory_sft",
        lambda **kwargs: TrainResult(
            status="completed",
            checkpoint_path=str(tmp_path / "ckpt"),
            recipe_path=str(tmp_path / "recipe.json"),
            train_log_path=str(tmp_path / "train.log"),
            command="train",
            dry_run=True,
        ),
    )
    monkeypatch.setattr(
        black_exp_module,
        "run_eval",
        lambda **kwargs: EvalReport(
            status="completed",
            overall_accuracy=0.25,
            benchmark_scores={"aime_2025": 0.25},
            sample_results_path=str(tmp_path / "samples.jsonl"),
            normalized_predictions_path=str(tmp_path / "preds.jsonl"),
            metadata={},
        ),
    )
    monkeypatch.setattr(
        black_exp_module,
        "run_inspect",
        lambda **kwargs: InspectReport(
            failure_clusters=[],
            weak_domains=[],
            weak_answer_styles=[],
            source_effect_hypotheses=[],
            recommended_next_action="expand_black",
            rationale="retry succeeded",
        ),
    )

    exp = BlackExp(
        agent=None,
        session=None,
        workspace=tmp_path,
        task_workspace=tmp_path,
        config=SimpleNamespace(
            base_model="Qwen3-1.7B",
            benchmark_suite=["aime_2025"],
            evaluation=None,
            llama_factory_env=None,
            dry_run_training=True,
        ),
        node=SimpleNamespace(id="black_retry_pack", parent=None),
        manifest_path=manifest_path,
        inspect_report_path=None,
    )
    exp.transform_script_path.parent.mkdir(parents=True, exist_ok=True)
    exp.transform_script_path.write_text("def transform(example):\n    return example\n", encoding="utf-8")
    result = exp.run("Collect and clean math data.")
    assert len(calls) == 2
    assert calls[0]["transform_script_path"] is not None
    assert calls[1]["transform_script_path"] is None
    assert calls[1]["processing_config"]["min_solution_chars"] == 0
    assert result["metric"] == 0.25


def test_train_runner_smoke_dry_run(tmp_path: Path) -> None:
    dataset_path = tmp_path / "alpaca.jsonl"
    write_jsonl(
        dataset_path,
        [{"instruction": "Solve", "input": "1+1", "output": "Final answer: 2"}],
    )
    result = run_llama_factory_sft(
        dataset_path=dataset_path,
        recipe_path=tmp_path / "recipe.json",
        output_dir=tmp_path / "ckpt",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        overrides={"num_train_epochs": 1},
        dry_run=True,
    )
    assert result.status == "dry_run"
    assert Path(result.recipe_path).exists()
    assert Path(result.checkpoint_path).exists()


def test_train_runner_fails_when_explicit_env_is_missing(tmp_path: Path) -> None:
    dataset_path = tmp_path / "alpaca.jsonl"
    write_jsonl(
        dataset_path,
        [{"instruction": "Solve", "input": "1+1", "output": "Final answer: 2"}],
    )
    result = run_llama_factory_sft(
        dataset_path=dataset_path,
        recipe_path=tmp_path / "recipe.json",
        output_dir=tmp_path / "ckpt_missing_env",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        overrides={"num_train_epochs": 1},
        dry_run=False,
        env_dir=tmp_path / "missing_lf_env",
    )
    assert result.status == "failed"
    assert "configured env_dir not found" in Path(result.train_log_path).read_text(encoding="utf-8")


def test_train_runner_exports_merged_model_for_posttrainbench_eval(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "alpaca.jsonl"
    write_jsonl(
        dataset_path,
        [{"instruction": "Solve", "input": "1+1", "output": "Final answer: 2"}],
    )
    calls: list[list[str]] = []

    class FakeCompletedProcess:
        def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_subprocess_run(args, **kwargs):
        calls.append([str(arg) for arg in args])
        if args[0] == "fake-cli":
            output_dir = Path(kwargs["cwd"])
            (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
            return FakeCompletedProcess(returncode=0, stdout="train ok")
        merged_dir = Path(args[4])
        merged_dir.mkdir(parents=True, exist_ok=True)
        (merged_dir / "config.json").write_text('{"architectures":["Qwen3ForCausalLM"]}', encoding="utf-8")
        return FakeCompletedProcess(returncode=0, stdout=str(merged_dir))

    monkeypatch.setattr(
        "playground.math_posttrain_datatree.core.utils.llama_factory.resolve_llamafactory_command",
        lambda **kwargs: ["fake-cli", "train"],
    )
    monkeypatch.setattr(
        "playground.math_posttrain_datatree.core.utils.llama_factory.subprocess.run",
        fake_subprocess_run,
    )

    result = run_llama_factory_sft(
        dataset_path=dataset_path,
        recipe_path=tmp_path / "recipe.json",
        output_dir=tmp_path / "ckpt_merged",
        base_model="/data/public_model/Qwen3-1.7B",
        overrides={"num_train_epochs": 1},
        dry_run=False,
        merge_for_evaluation=True,
    )
    assert result.status == "completed"
    assert Path(result.checkpoint_path).name == "merged_model"
    assert Path(result.checkpoint_path, "config.json").exists()
    assert any(call[0] == "fake-cli" for call in calls)


def test_create_dataset_registry_for_llamafactory(tmp_path: Path) -> None:
    dataset_path = tmp_path / "alpaca.jsonl"
    write_jsonl(
        dataset_path,
        [{"instruction": "Solve", "input": "1+1", "output": "Final answer: 2"}],
    )
    registry_path, dataset_name = create_dataset_registry(dataset_path, dataset_name="unit_test_set")
    assert registry_path.exists()
    payload = registry_path.read_text(encoding="utf-8")
    assert "unit_test_set" in payload
    assert dataset_name == "unit_test_set"


def test_llamafactory_resolution_is_explicit() -> None:
    cmd = resolve_llamafactory_command()
    ok, detail = ensure_llamafactory_available()
    if cmd is None:
        assert ok is False
        assert "LLaMA Factory is not available" in detail
    else:
        assert ok is True
        assert detail


def test_resolve_env_bin_dir_prefers_explicit_env(tmp_path: Path) -> None:
    env_dir = tmp_path / "lf_env"
    bin_dir = env_dir / "bin"
    bin_dir.mkdir(parents=True)
    python_bin = bin_dir / "python"
    python_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_bin.chmod(0o755)
    assert _resolve_env_bin_dir(env_dir=env_dir) == bin_dir
    assert _resolve_env_bin_dir(python_bin=python_bin) == bin_dir


def test_eval_scores_predictions_and_normalizes_answers(tmp_path: Path) -> None:
    samples = [
        {"id": "a", "answer": r"\boxed{42}"},
        {"id": "b", "answer": "17"},
    ]
    predictions = [
        {"id": "a", "prediction": "Final answer: 42"},
        {"id": "b", "prediction": "18"},
    ]
    acc, rows = score_answer_predictions(samples, predictions)
    assert acc == 0.5
    assert rows[0]["correct"] is True

    bench_path = tmp_path / "aime.jsonl"
    pred_path = tmp_path / "aime_pred.jsonl"
    write_jsonl(bench_path, samples)
    write_jsonl(pred_path, predictions)
    report = run_eval(
        eval_dir=tmp_path / "eval",
        benchmark_suite=["aime_2025"],
        pack_manifest={"sample_count": 10, "coverage_tags": ["aime"]},
        pack_stats={"style_distribution": {"short_answer": 5, "long_reasoning": 5}, "duplicate_rate": 0.0},
        benchmark_files={"aime_2025": str(bench_path)},
        prediction_files={"aime_2025": str(pred_path)},
    )
    assert report.status == "completed"
    assert report.overall_accuracy == 0.5


def test_eval_accepts_generation_like_prediction_fields(tmp_path: Path) -> None:
    bench_path = tmp_path / "aime_like.jsonl"
    pred_path = tmp_path / "aime_like_predictions.jsonl"
    write_jsonl(
        bench_path,
        [
            {"id": "1", "answer": r"\boxed{256}"},
            {"id": "2", "final_answer": "13"},
        ],
    )
    write_jsonl(
        pred_path,
        [
            {"id": "1", "response": "We compute carefully.\nFinal answer: 256"},
            {"id": "2", "output": r"The answer is \boxed{14}"},
        ],
    )
    report = run_eval(
        eval_dir=tmp_path / "eval_like",
        benchmark_suite=["aime_2025"],
        pack_manifest={"sample_count": 10, "coverage_tags": ["aime", "competition_math"]},
        pack_stats={"style_distribution": {"short_answer": 5, "long_reasoning": 5}, "duplicate_rate": 0.0},
        benchmark_files={"aime_2025": str(bench_path)},
        prediction_files={"aime_2025": str(pred_path)},
    )
    assert report.status == "completed"
    assert report.benchmark_scores["aime_2025"] == 0.5


def test_red_exp_preserves_agent_written_manifest(tmp_path: Path) -> None:
    node = type("Node", (), {"id": "red_node_1"})()
    manifest_path = tmp_path / "artifacts" / "manifests" / "dataset_manifest_red_node_1.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        """{
  "manifest_id": "red_node_1",
  "datasets": [
    {
      "dataset_id": "EleutherAI/hendrycks_math",
      "name": "MATH Dataset",
      "huggingface_url": "https://huggingface.co/datasets/EleutherAI/hendrycks_math",
      "categories": ["aime", "competition_math"],
      "quality": "high"
    }
  ]
}""",
        encoding="utf-8",
    )
    exp = RedExp(
        agent=None,
        session=None,
        workspace=tmp_path,
        task_workspace=tmp_path,
        config=None,
        node=node,
        manifest_path=manifest_path,
        search_goal="Find public math data.",
    )
    result = exp.run("Collect math datasets.")
    assert result["metric_detail"]["manifest_ok"] is True
    payload = manifest_path.read_text(encoding="utf-8")
    assert "EleutherAI/hendrycks_math" in payload


def test_red_exp_accepts_agent_manifest_with_name_only(tmp_path: Path) -> None:
    node = type("Node", (), {"id": "red_node_2"})()
    manifest_path = tmp_path / "artifacts" / "manifests" / "dataset_manifest_red_node_2.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        """{
  "datasets": [
    {
      "name": "qwedsacf/competition_math",
      "license": "MIT",
      "subjects": ["Algebra", "Geometry"]
    }
  ]
}""",
        encoding="utf-8",
    )
    exp = RedExp(
        agent=None,
        session=None,
        workspace=tmp_path,
        task_workspace=tmp_path,
        config=None,
        node=node,
        manifest_path=manifest_path,
        search_goal="Find public math data.",
    )
    result = exp.run("Collect math datasets.")
    assert result["metric_detail"]["manifest_ok"] is True
    payload = manifest_path.read_text(encoding="utf-8")
    assert "qwedsacf/competition_math" in payload


def test_red_exp_normalizes_known_dataset_aliases(tmp_path: Path) -> None:
    node = type("Node", (), {"id": "red_node_3"})()
    manifest_path = tmp_path / "artifacts" / "manifests" / "dataset_manifest_red_node_3.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        """{
  "datasets": [
    {
      "name": "MATH Dataset (Hendrycks)",
      "url": "agent_search"
    },
    {
      "source_id": "AIME 1983-2024",
      "license": "CC0 Public Domain"
    },
    {
      "source_id": "GSM8K"
    }
  ]
}""",
        encoding="utf-8",
    )
    exp = RedExp(
        agent=None,
        session=None,
        workspace=tmp_path,
        task_workspace=tmp_path,
        config=None,
        node=node,
        manifest_path=manifest_path,
        search_goal="Find public math data.",
    )
    result = exp.run("Collect math datasets.")
    assert result["metric_detail"]["manifest_ok"] is True
    payload = manifest_path.read_text(encoding="utf-8")
    assert "nlile/hendrycks-MATH-benchmark" in payload
    assert "gneubig/aime-1983-2024" in payload
    assert "openai/gsm8k" in payload


def test_red_exp_filters_web_scale_corpus_datasets(tmp_path: Path) -> None:
    node = type("Node", (), {"id": "red_node_4"})()
    manifest_path = tmp_path / "artifacts" / "manifests" / "dataset_manifest_red_node_4.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        """{
  "datasets": [
    {
      "source_id": "qwedsacf/competition_math",
      "url": "https://huggingface.co/datasets/qwedsacf/competition_math"
    },
    {
      "source_id": "open-web-math/open-web-math",
      "url": "https://huggingface.co/datasets/open-web-math/open-web-math",
      "description": "Large-scale dataset of high-quality mathematical web text from Common Crawl. Contains 6.3M documents and 14.7B tokens."
    }
  ]
}""",
        encoding="utf-8",
    )
    exp = RedExp(
        agent=None,
        session=None,
        workspace=tmp_path,
        task_workspace=tmp_path,
        config=None,
        node=node,
        manifest_path=manifest_path,
        search_goal="Find public math data.",
    )
    result = exp.run("Collect math datasets.")
    assert result["metric_detail"]["manifest_ok"] is True
    payload = manifest_path.read_text(encoding="utf-8")
    assert "qwedsacf/competition_math" in payload
    assert "open-web-math/open-web-math" not in payload


def test_posttrainbench_backend_runs_external_aime_eval(tmp_path: Path) -> None:
    repo_dir = tmp_path / "PostTrainBench"
    task_dir = repo_dir / "src" / "eval" / "tasks" / "aime2025"
    task_dir.mkdir(parents=True)
    evaluate_py = task_dir / "evaluate.py"
    evaluate_py.write_text(
        "\n".join(
            [
                "import argparse",
                "import json",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--model-path')",
                "parser.add_argument('--json-output-file')",
                "parser.add_argument('--limit', default=None)",
                "parser.add_argument('--max-tokens', default=None)",
                "parser.add_argument('--max-connections', default=None)",
                "parser.add_argument('--gpu-memory-utilization', default=None)",
                "parser.add_argument('--templates-dir', default=None)",
                "args = parser.parse_args()",
                "Path(args.model_path).mkdir(parents=True, exist_ok=True)",
                "with open(args.json_output_file, 'w', encoding='utf-8') as f:",
                "    json.dump({'accuracy': 0.75}, f)",
            ]
        ),
        encoding="utf-8",
    )
    model_dir = tmp_path / "checkpoint-last"
    model_dir.mkdir()

    assert resolve_posttrainbench_task_dir(repo_dir, "aime_2025") == task_dir
    report = run_eval(
        eval_dir=tmp_path / "eval_ptb",
        benchmark_suite=["aime_2025"],
        pack_manifest={"sample_count": 10, "coverage_tags": ["aime", "competition_math"]},
        pack_stats={"style_distribution": {"short_answer": 5, "long_reasoning": 5}, "duplicate_rate": 0.0},
        eval_backend="posttrainbench",
        evaluation_options={"repo_dir": str(repo_dir), "python_bin": sys.executable},
        model_path=str(model_dir),
    )
    assert report.status == "completed"
    assert report.overall_accuracy == 0.75
    assert report.metadata["backend_details"]["aime_2025"]["status"] == "completed"
    assert report.metadata["backend_details"]["aime_2025"]["command"][0] == sys.executable


def test_posttrainbench_backend_preserves_explicit_venv_python_path(tmp_path: Path) -> None:
    repo_dir = tmp_path / "PostTrainBench"
    task_dir = repo_dir / "src" / "eval" / "tasks" / "aime2025"
    task_dir.mkdir(parents=True)
    evaluate_py = task_dir / "evaluate.py"
    evaluate_py.write_text(
        "\n".join(
            [
                "import argparse",
                "import json",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--model-path')",
                "parser.add_argument('--json-output-file')",
                "args, _ = parser.parse_known_args()",
                "with open(args.json_output_file, 'w', encoding='utf-8') as f:",
                "    json.dump({'accuracy': 0.5}, f)",
            ]
        ),
        encoding="utf-8",
    )
    model_dir = tmp_path / "checkpoint-last"
    model_dir.mkdir()
    venv_bin = tmp_path / "fake_eval_env" / "bin"
    venv_bin.mkdir(parents=True)
    python_link = venv_bin / "python"
    python_link.symlink_to(Path(sys.executable))

    report = run_eval(
        eval_dir=tmp_path / "eval_ptb_symlink",
        benchmark_suite=["aime_2025"],
        pack_manifest={"sample_count": 10, "coverage_tags": ["aime", "competition_math"]},
        pack_stats={"style_distribution": {"short_answer": 5, "long_reasoning": 5}, "duplicate_rate": 0.0},
        eval_backend="posttrainbench",
        evaluation_options={"repo_dir": str(repo_dir), "python_bin": str(python_link)},
        model_path=str(model_dir),
    )
    assert report.status == "completed"
    assert report.metadata["backend_details"]["aime_2025"]["command"][0] == str(python_link)


def test_posttrainbench_backend_injects_venv_bin_into_path(monkeypatch, tmp_path: Path) -> None:
    repo_dir = tmp_path / "PostTrainBench"
    task_dir = repo_dir / "src" / "eval" / "tasks" / "aime2025"
    task_dir.mkdir(parents=True)
    (task_dir / "evaluate.py").write_text("print('ok')\n", encoding="utf-8")
    venv_bin = tmp_path / "fake_eval_env" / "bin"
    venv_bin.mkdir(parents=True)
    python_link = venv_bin / "python"
    python_link.symlink_to(Path(sys.executable))
    (venv_bin / "vllm").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (venv_bin / "vllm").chmod(0o755)
    seen_env: dict[str, str] = {}

    class FakePopen:
        def __init__(self, command, cwd, stdout, stderr, text, env):
            self.command = command
            self.cwd = cwd
            self.stdout = stdout
            self.stderr = stderr
            self.text = text
            self.env = env
            self.pid = 12345
            self.returncode = 0
            metrics_path = Path(command[command.index("--json-output-file") + 1])
            metrics_path.write_text('{"accuracy": 0.42}', encoding="utf-8")

        def poll(self):
            return 0

        def wait(self):
            return 0

    def fake_popen(command, cwd, stdout, stderr, text, env):
        seen_env.update(env)
        return FakePopen(command, cwd, stdout, stderr, text, env)

    monkeypatch.setattr(
        "playground.math_posttrain_datatree.core.utils.eval.subprocess.Popen",
        fake_popen,
    )
    score, detail = run_posttrainbench_eval(
        eval_dir=tmp_path / "eval_path",
        benchmark_id="aime_2025",
        model_path=tmp_path / "model",
        repo_dir=repo_dir,
        python_bin=python_link,
        auto_select_device=False,
    )
    assert score == 0.42
    assert detail["status"] == "completed"
    assert seen_env["PATH"].split(":")[0] == str(venv_bin)
    assert seen_env["VIRTUAL_ENV"] == str(venv_bin.parent)


def test_posttrainbench_backend_sets_official_hf_env(monkeypatch, tmp_path: Path) -> None:
    repo_dir = tmp_path / "PostTrainBench"
    task_dir = repo_dir / "src" / "eval" / "tasks" / "aime2025"
    task_dir.mkdir(parents=True)
    (task_dir / "evaluate.py").write_text("print('ok')\n", encoding="utf-8")
    venv_bin = tmp_path / "fake_eval_env" / "bin"
    venv_bin.mkdir(parents=True)
    python_link = venv_bin / "python"
    python_link.symlink_to(Path(sys.executable))
    seen_env: dict[str, str] = {}

    class FakePopen:
        def __init__(self, command, cwd, stdout, stderr, text, env):
            self.command = command
            self.cwd = cwd
            self.stdout = stdout
            self.stderr = stderr
            self.text = text
            self.env = env
            self.pid = 12346
            self.returncode = 0
            metrics_path = Path(command[command.index("--json-output-file") + 1])
            metrics_path.write_text('{"accuracy": 0.42}', encoding="utf-8")

        def poll(self):
            return 0

        def wait(self):
            return 0

    def fake_popen(command, cwd, stdout, stderr, text, env):
        seen_env.update(env)
        return FakePopen(command, cwd, stdout, stderr, text, env)

    monkeypatch.setattr(
        "playground.math_posttrain_datatree.core.utils.eval.subprocess.Popen",
        fake_popen,
    )
    score, detail = run_posttrainbench_eval(
        eval_dir=tmp_path / "eval_path",
        benchmark_id="aime_2025",
        model_path=tmp_path / "model",
        repo_dir=repo_dir,
        python_bin=python_link,
        auto_select_device=False,
    )
    assert score == 0.42
    assert detail["status"] == "completed"
    assert seen_env["HF_ENDPOINT"] == "https://huggingface.co"
    assert seen_env["HF_HOME"] == "/data/HF_Cache_dataevo"
    assert seen_env["HUGGINGFACE_HUB_CACHE"] == "/data/HF_Cache_dataevo/hub"
    assert seen_env["HF_DATASETS_CACHE"] == "/data/HF_Cache_dataevo/datasets"
    assert json.loads(seen_env["VLLM_DEFAULT_SERVER_ARGS"]) == {"max_num_seqs": 32}


def test_black_exp_marks_training_failure_as_bug(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "math.jsonl"
    write_jsonl(
        dataset_path,
        [{"problem": "1+1?", "solution": "1+1=2. Final answer: 2", "final_answer": "2"}],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"datasets": [{"source_id": "local_ds", "local_path": str(dataset_path)}]}),
        encoding="utf-8",
    )

    def fake_train(**kwargs):
        return TrainResult(
            status="failed",
            checkpoint_path=str(tmp_path / "checkpoint-last"),
            recipe_path=str(tmp_path / "recipe.json"),
            train_log_path=str(tmp_path / "train.log"),
            command="lf train recipe.json",
            dry_run=False,
            metrics={"reason": "env missing"},
        )

    def fail_eval(**kwargs):
        raise AssertionError("run_eval should not be called after training failure")

    monkeypatch.setattr(
        "playground.math_posttrain_datatree.core.exp.black_exp.run_llama_factory_sft",
        fake_train,
    )
    monkeypatch.setattr("playground.math_posttrain_datatree.core.exp.black_exp.run_eval", fail_eval)

    parent = SimpleNamespace(stage="red", children=[])
    node = SimpleNamespace(id="black_bug_1", parent=parent)
    cfg = SimpleNamespace(
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        benchmark_suite=["aime_2025"],
        evaluation={"backend": "builtin"},
        dry_run_training=False,
        llama_factory_env={},
    )
    exp = BlackExp(
        agent=None,
        session=None,
        workspace=tmp_path,
        task_workspace=tmp_path,
        config=cfg,
        node=node,
        manifest_path=manifest_path,
        inspect_report_path=None,
    )
    result = exp.run("train")
    assert result["metric"] is None
    assert result["metric_detail"]["is_bug"] is True
    assert result["metric_detail"]["has_submission"] is False


def test_black_exp_marks_posttrainbench_backend_failure_as_bug(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "math_eval.jsonl"
    write_jsonl(
        dataset_path,
        [{"problem": "2+2?", "solution": "2+2=4. Final answer: 4", "final_answer": "4"}],
    )
    manifest_path = tmp_path / "manifest_eval.json"
    manifest_path.write_text(
        json.dumps({"datasets": [{"source_id": "local_ds", "local_path": str(dataset_path)}]}),
        encoding="utf-8",
    )

    def fake_train(**kwargs):
        return TrainResult(
            status="completed",
            checkpoint_path=str(tmp_path / "checkpoint-last"),
            recipe_path=str(tmp_path / "recipe.json"),
            train_log_path=str(tmp_path / "train.log"),
            command="lf train recipe.json",
            dry_run=False,
            metrics={"returncode": 0},
        )

    def fake_eval(**kwargs):
        return EvalReport(
            status="proxy",
            overall_accuracy=0.1,
            benchmark_scores={"aime_2025": 0.1},
            sample_results_path=str(tmp_path / "samples.jsonl"),
            normalized_predictions_path=str(tmp_path / "preds.jsonl"),
            metadata={"backend_details": {"aime_2025": {"status": "failed", "reason": "inspect_ai missing"}}},
        )

    def fail_inspect(**kwargs):
        raise AssertionError("run_inspect should not be called after eval backend failure")

    monkeypatch.setattr(
        "playground.math_posttrain_datatree.core.exp.black_exp.run_llama_factory_sft",
        fake_train,
    )
    monkeypatch.setattr("playground.math_posttrain_datatree.core.exp.black_exp.run_eval", fake_eval)
    monkeypatch.setattr("playground.math_posttrain_datatree.core.exp.black_exp.run_inspect", fail_inspect)

    parent = SimpleNamespace(stage="red", children=[])
    node = SimpleNamespace(id="black_bug_2", parent=parent)
    cfg = SimpleNamespace(
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        benchmark_suite=["aime_2025"],
        evaluation={"backend": "posttrainbench", "posttrainbench": {"python_bin": "./.venv_posttrainbench_eval/bin/python"}},
        dry_run_training=False,
        llama_factory_env={},
    )
    exp = BlackExp(
        agent=None,
        session=None,
        workspace=tmp_path,
        task_workspace=tmp_path,
        config=cfg,
        node=node,
        manifest_path=manifest_path,
        inspect_report_path=None,
    )
    result = exp.run("eval")
    assert result["metric"] is None
    assert result["metric_detail"]["is_bug"] is True
    assert result["metric_detail"]["has_submission"] is False


def test_black_exp_accepts_inspect_report_dataclass(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "math_eval_ok.jsonl"
    write_jsonl(
        dataset_path,
        [{"problem": "3+4?", "solution": "3+4=7. Final answer: 7", "final_answer": "7"}],
    )
    manifest_path = tmp_path / "manifest_eval_ok.json"
    manifest_path.write_text(
        json.dumps({"datasets": [{"source_id": "local_ds", "local_path": str(dataset_path)}]}),
        encoding="utf-8",
    )

    def fake_train(**kwargs):
        return TrainResult(
            status="completed",
            checkpoint_path=str(tmp_path / "checkpoint-last"),
            recipe_path=str(tmp_path / "recipe.json"),
            train_log_path=str(tmp_path / "train.log"),
            command="lf train recipe.json",
            dry_run=False,
            metrics={"returncode": 0},
        )

    def fake_eval(**kwargs):
        return EvalReport(
            status="completed",
            overall_accuracy=0.25,
            benchmark_scores={"aime_2025": 0.25},
            sample_results_path=str(tmp_path / "samples.jsonl"),
            normalized_predictions_path=str(tmp_path / "preds.jsonl"),
            metadata={"backend_details": {"aime_2025": {"status": "completed"}}},
        )

    def fake_inspect(**kwargs):
        return InspectReport(
            failure_clusters=["aime_2025"],
            weak_domains=["aime_2025"],
            weak_answer_styles=[],
            source_effect_hypotheses=["Need broader public data coverage for weak domains."],
            recommended_next_action="expand_red",
            rationale="coverage gap",
        )

    monkeypatch.setattr(
        "playground.math_posttrain_datatree.core.exp.black_exp.run_llama_factory_sft",
        fake_train,
    )
    monkeypatch.setattr("playground.math_posttrain_datatree.core.exp.black_exp.run_eval", fake_eval)
    monkeypatch.setattr("playground.math_posttrain_datatree.core.exp.black_exp.run_inspect", fake_inspect)

    parent = SimpleNamespace(stage="red", children=[])
    node = SimpleNamespace(id="black_ok_1", parent=parent)
    cfg = SimpleNamespace(
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        benchmark_suite=["aime_2025"],
        evaluation={"backend": "posttrainbench", "posttrainbench": {"python_bin": "./.venv_posttrainbench_eval/bin/python"}},
        dry_run_training=False,
        llama_factory_env={},
    )
    exp = BlackExp(
        agent=None,
        session=None,
        workspace=tmp_path,
        task_workspace=tmp_path,
        config=cfg,
        node=node,
        manifest_path=manifest_path,
        inspect_report_path=None,
    )
    result = exp.run("eval")
    assert result["metric"] == 0.25
    assert result["metric_detail"]["is_bug"] is False
    assert result["metric_detail"]["recommended_next_action"] == "expand_red"


def test_write_uct_trajectory_persists_records(tmp_path: Path) -> None:
    path = write_uct_trajectory(
        tmp_path,
        [{"stage": "red", "node_id": "n1", "metric": None, "recommended_next_action": None}],
    )
    assert path.exists()
    assert "n1" in path.read_text(encoding="utf-8")


def test_inspect_recommends_expand_red_for_coverage_gaps(tmp_path: Path) -> None:
    report = run_inspect(
        eval_report={
            "overall_accuracy": 0.2,
            "benchmark_scores": {"aime_2025": 0.1, "gsm8k": 0.3},
        },
        pack_manifest={"coverage_tags": ["aime"], "sample_count": 20},
        pack_stats={
            "source_count": 1,
            "style_distribution": {"short_answer": 10, "long_reasoning": 10},
            "duplicate_rate": 0.0,
        },
        output_path=tmp_path / "inspect.json",
    )
    assert report.recommended_next_action == "expand_red"


def test_search_manager_counts_black_nodes_per_red() -> None:
    mgr = UCTSearchManager(UCTSearchConfig())
    seed = mgr.create_child(mgr.root, "seed")
    red = mgr.create_child(seed, "red")
    black_1 = mgr.create_child(red, "black")
    black_1.bound_red_node_id = red.id
    black_2 = mgr.create_child(black_1, "black")
    black_2.bound_red_node_id = red.id
    mgr.ingest_result(red, MetricReview(metric=None, is_bug=False))
    mgr.ingest_result(black_1, MetricReview(metric=0.2, is_bug=False, has_submission=True))
    mgr.ingest_result(black_2, MetricReview(metric=0.3, is_bug=False, has_submission=True))
    assert mgr.count_black_nodes_for_red(red.id) == 2
