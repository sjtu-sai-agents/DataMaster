from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from playground.math_posttrain_datatree.core.exp.black_exp import BlackExp
from playground.math_posttrain_datatree.core.utils.data import (
    synthesize_pack_from_prepared_train_file,
    validate_prepared_train_file,
    validate_train_config,
)
from playground.math_posttrain_datatree.core.utils.types import EvalReport, InspectReport, TrainResult


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_validate_prepared_train_file_passes_with_required_contract(tmp_path):
    train_path = tmp_path / "train.jsonl"
    prep_path = tmp_path / "prep_report.json"
    _write_jsonl(
        train_path,
        [
            {
                "instruction": "Solve carefully.",
                "input": "What is 1+1?",
                "output": "The answer is 2.\n\nFinal answer: 2",
                "metadata": {
                    "source_id": "qwedsacf/competition_math",
                    "topic": "Algebra",
                    "difficulty": "easy",
                },
            }
        ],
    )
    prep_path.write_text(
        json.dumps(
            {
                "selected_sources": ["qwedsacf/competition_math"],
                "raw_rows_seen": 10,
                "rows_written": 1,
                "duplicate_rows_removed": 0,
                "notes": "ok",
            }
        ),
        encoding="utf-8",
    )

    payload = validate_prepared_train_file(train_path, prep_path)

    assert payload["status"] == "passed"
    assert payload["row_count"] == 1
    assert payload["selected_sources"] == ["qwedsacf/competition_math"]


def test_validate_prepared_train_file_accepts_raw_rows_seen_breakdown_dict(tmp_path):
    train_path = tmp_path / "train.jsonl"
    prep_path = tmp_path / "prep_report.json"
    _write_jsonl(
        train_path,
        [
            {
                "instruction": "Solve carefully.",
                "input": "What is 1+1?",
                "output": "The answer is 2.",
                "metadata": {"source_id": "src_a"},
            },
            {
                "instruction": "Solve carefully.",
                "input": "What is 2+2?",
                "output": "The answer is 4.",
                "metadata": {"source_id": "src_b"},
            },
        ],
    )
    prep_path.write_text(
        json.dumps(
            {
                "selected_sources": ["src_a", "src_b"],
                "raw_rows_seen": {"src_a": 3, "src_b": 5},
                "rows_written": 2,
                "duplicate_rows_removed": 0,
                "notes": "ok",
            }
        ),
        encoding="utf-8",
    )

    payload = validate_prepared_train_file(train_path, prep_path)

    assert payload["status"] == "passed"
    assert payload["row_count"] == 2


def test_validate_prepared_train_file_rejects_missing_metadata_source_id(tmp_path):
    train_path = tmp_path / "train.jsonl"
    prep_path = tmp_path / "prep_report.json"
    _write_jsonl(
        train_path,
        [
            {
                "instruction": "Solve carefully.",
                "input": "What is 1+1?",
                "output": "2",
                "metadata": {},
            }
        ],
    )
    prep_path.write_text(
        json.dumps(
            {
                "selected_sources": [],
                "raw_rows_seen": 1,
                "rows_written": 1,
                "duplicate_rows_removed": 0,
                "notes": "bad",
            }
        ),
        encoding="utf-8",
    )

    payload = validate_prepared_train_file(train_path, prep_path)

    assert payload["status"] == "failed"
    assert "metadata.source_id" in payload["reason"]




def test_validate_prepared_train_file_accepts_string_raw_id(tmp_path):
    train_path = tmp_path / "train.jsonl"
    prep_path = tmp_path / "prep_report.json"
    _write_jsonl(
        train_path,
        [
            {
                "instruction": "Solve carefully.",
                "input": "Question A",
                "output": "Answer A",
                "metadata": {"source_id": "src_a", "raw_id": "abc-123"},
            },
            {
                "instruction": "Solve carefully.",
                "input": "Question B",
                "output": "Answer B",
                "metadata": {"source_id": "src_b", "raw_id": "456"},
            },
        ],
    )
    prep_path.write_text(
        json.dumps(
            {
                "selected_sources": ["src_a", "src_b"],
                "raw_rows_seen": 2,
                "rows_written": 2,
                "duplicate_rows_removed": 0,
                "notes": "ok",
            }
        ),
        encoding="utf-8",
    )

    payload = validate_prepared_train_file(train_path, prep_path)

    assert payload["status"] == "passed"



def test_validate_prepared_train_file_rejects_non_string_raw_id(tmp_path):
    train_path = tmp_path / "train.jsonl"
    prep_path = tmp_path / "prep_report.json"
    _write_jsonl(
        train_path,
        [
            {
                "instruction": "Solve carefully.",
                "input": "Question A",
                "output": "Answer A",
                "metadata": {"source_id": "src_a", "raw_id": "abc-123"},
            },
            {
                "instruction": "Solve carefully.",
                "input": "Question B",
                "output": "Answer B",
                "metadata": {"source_id": "src_b", "raw_id": 456},
            },
        ],
    )
    prep_path.write_text(
        json.dumps(
            {
                "selected_sources": ["src_a", "src_b"],
                "raw_rows_seen": 2,
                "rows_written": 2,
                "duplicate_rows_removed": 0,
                "notes": "bad",
            }
        ),
        encoding="utf-8",
    )

    payload = validate_prepared_train_file(train_path, prep_path)

    assert payload["status"] == "failed"
    assert "metadata.raw_id" in payload["reason"]



def test_validate_prepared_train_file_rejects_non_string_tags(tmp_path):
    train_path = tmp_path / "train.jsonl"
    prep_path = tmp_path / "prep_report.json"
    _write_jsonl(
        train_path,
        [
            {
                "instruction": "Solve carefully.",
                "input": "Question A",
                "output": "Answer A",
                "metadata": {"source_id": "src_a", "tags": ["writing", 7]},
            }
        ],
    )
    prep_path.write_text(
        json.dumps(
            {
                "selected_sources": ["src_a"],
                "raw_rows_seen": 1,
                "rows_written": 1,
                "duplicate_rows_removed": 0,
                "notes": "bad",
            }
        ),
        encoding="utf-8",
    )

    payload = validate_prepared_train_file(train_path, prep_path)

    assert payload["status"] == "failed"
    assert "metadata.tags" in payload["reason"]



def test_validate_prepared_train_file_rejects_non_string_topic_or_difficulty(tmp_path):
    train_path = tmp_path / "train.jsonl"
    prep_path = tmp_path / "prep_report.json"
    _write_jsonl(
        train_path,
        [
            {
                "instruction": "Solve carefully.",
                "input": "Question A",
                "output": "Answer A",
                "metadata": {"source_id": "src_a", "topic": 123, "difficulty": "hard"},
            }
        ],
    )
    prep_path.write_text(
        json.dumps(
            {
                "selected_sources": ["src_a"],
                "raw_rows_seen": 1,
                "rows_written": 1,
                "duplicate_rows_removed": 0,
                "notes": "bad",
            }
        ),
        encoding="utf-8",
    )

    payload = validate_prepared_train_file(train_path, prep_path)

    assert payload["status"] == "failed"
    assert "metadata.topic" in payload["reason"]



def test_validate_train_config_passes_with_defaults_and_writes_effective(tmp_path):
    config_path = tmp_path / "train_config.json"
    report_path = tmp_path / "train_config_validation_report.json"
    effective_path = tmp_path / "effective_train_config.json"
    config_path.write_text(
        json.dumps({
            "num_train_epochs": 2,
            "learning_rate": 8e-5,
            "max_samples": 800,
        }),
        encoding="utf-8",
    )

    payload = validate_train_config(
        config_path,
        output_path=report_path,
        effective_output_path=effective_path,
    )

    assert payload["status"] == "passed"
    effective = json.loads(effective_path.read_text(encoding="utf-8"))
    assert effective["num_train_epochs"] == 2.0
    assert effective["learning_rate"] == 8e-5
    assert effective["max_samples"] == 800
    assert effective["per_device_train_batch_size"] == 2


def test_validate_train_config_rejects_unknown_key(tmp_path):
    config_path = tmp_path / "train_config.json"
    config_path.write_text(json.dumps({"unknown": 1}), encoding="utf-8")

    payload = validate_train_config(config_path)

    assert payload["status"] == "failed"
    assert "unsupported keys" in payload["reason"]


def test_validate_train_config_rejects_invalid_types(tmp_path):
    config_path = tmp_path / "train_config.json"
    config_path.write_text(json.dumps({"max_samples": "many"}), encoding="utf-8")

    payload = validate_train_config(config_path)

    assert payload["status"] == "failed"
    assert "must be numeric" in payload["reason"]


def test_validate_train_config_fills_missing_keys_from_defaults(tmp_path):
    config_path = tmp_path / "train_config.json"
    effective_path = tmp_path / "effective_train_config.json"
    config_path.write_text(json.dumps({"learning_rate": 5e-5}), encoding="utf-8")

    payload = validate_train_config(config_path, effective_output_path=effective_path)

    assert payload["status"] == "passed"
    effective = json.loads(effective_path.read_text(encoding="utf-8"))
    assert effective["learning_rate"] == 5e-5
    assert effective["num_train_epochs"] == 1
    assert effective["max_samples"] == 1000


def test_synthesize_pack_from_prepared_train_file_uses_metadata_and_manifest(tmp_path):
    train_path = tmp_path / "train.jsonl"
    prep_path = tmp_path / "prep_report.json"
    _write_jsonl(
        train_path,
        [
            {
                "instruction": "Solve carefully.",
                "input": "Question A",
                "output": "Reasoning...\n\nFinal answer: 7",
                "metadata": {
                    "source_id": "src_a",
                    "topic": "Algebra",
                    "difficulty": "hard",
                    "tags": ["aime", "competition_math"],
                },
            },
            {
                "instruction": "Solve carefully.",
                "input": "Question B",
                "output": "Final answer: 3",
                "metadata": {
                    "source_id": "src_b",
                    "topic": "Geometry",
                    "difficulty": "medium",
                },
            },
            {
                "instruction": "Solve carefully.",
                "input": "Question B",
                "output": "Final answer: 3",
                "metadata": {
                    "source_id": "src_b",
                    "topic": "Geometry",
                    "difficulty": "medium",
                },
            },
        ],
    )
    prep_path.write_text(
        json.dumps(
            {
                "selected_sources": ["src_a", "src_b"],
                "raw_rows_seen": 5,
                "rows_written": 3,
                "duplicate_rows_removed": 1,
                "notes": "ok",
            }
        ),
        encoding="utf-8",
    )

    manifest, stats, output_path = synthesize_pack_from_prepared_train_file(
        [
            {"source_id": "src_a", "coverage_tags": ["algebra"]},
            {"source_id": "src_b", "coverage_tags": ["geometry"]},
        ],
        train_path,
        prep_path,
        pack_id="pack_test",
    )

    assert output_path == train_path
    assert manifest.sample_count == 3
    assert manifest.source_datasets == ["src_a", "src_b"]
    assert set(manifest.coverage_tags) >= {"algebra", "geometry", "aime", "competition_math"}
    assert stats["style_distribution"]["long_reasoning"] >= 1
    assert stats["duplicate_rate"] > 0
    assert stats["raw_rows_seen"] == 5


def test_synthesize_pack_from_prepared_train_file_preserves_raw_rows_seen_breakdown(tmp_path):
    train_path = tmp_path / "train.jsonl"
    prep_path = tmp_path / "prep_report.json"
    _write_jsonl(
        train_path,
        [
            {
                "instruction": "Solve carefully.",
                "input": "Question A",
                "output": "Final answer: 7",
                "metadata": {"source_id": "src_a"},
            }
        ],
    )
    prep_path.write_text(
        json.dumps(
            {
                "selected_sources": ["src_a", "src_b"],
                "raw_rows_seen": {"src_a": 2, "src_b": 9},
                "rows_written": 1,
                "duplicate_rows_removed": 0,
                "notes": {"detail": "kept best rows"},
            }
        ),
        encoding="utf-8",
    )

    _manifest, stats, _output_path = synthesize_pack_from_prepared_train_file(
        [{"source_id": "src_a", "coverage_tags": []}],
        train_path,
        prep_path,
        pack_id="pack_breakdown",
    )

    assert stats["raw_rows_seen"] == 11
    assert stats["raw_rows_seen_breakdown"] == {"src_a": 2, "src_b": 9}


def test_black_exp_run_uses_direct_train_jsonl_pipeline(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    task_workspace = workspace
    (task_workspace / "artifacts" / "train_packs").mkdir(parents=True, exist_ok=True)

    raw_data_path = task_workspace / "materialized.jsonl"
    _write_jsonl(
        raw_data_path,
        [
            {
                "problem": "What is 2+2?",
                "solution": "4",
            }
        ],
    )
    manifest_path = task_workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "source_id": "src_a",
                        "name": "src_a",
                        "license": "mit",
                        "url": "https://example.com/src_a",
                        "local_path": str(raw_data_path),
                        "full_local_path": str(raw_data_path),
                        "coverage_tags": ["algebra"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    node = SimpleNamespace(id="blacknode1", parent=None)
    config = SimpleNamespace(
        base_model="/models/mock",
        benchmark_suite=["aime_2025"],
        dry_run_training=True,
        evaluation={"backend": "builtin"},
        llama_factory_env={},
    )
    exp = BlackExp(
        agent=None,
        session=object(),
        workspace=workspace,
        task_workspace=task_workspace,
        config=config,
        node=node,
        manifest_path=manifest_path,
        inspect_report_path=None,
        global_pool_manifest_path=manifest_path,
        input_black_handoff_path=None,
    )
    exp.pack_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        exp.train_jsonl_path,
        [
            {
                "instruction": "Solve carefully.",
                "input": "What is 2+2?",
                "output": "Final answer: 4",
                "metadata": {
                    "source_id": "src_a",
                    "topic": "Algebra",
                    "difficulty": "easy",
                },
            }
        ],
    )
    exp.prep_report_path.write_text(
        json.dumps(
            {
                "selected_sources": ["src_a"],
                "raw_rows_seen": 1,
                "rows_written": 1,
                "duplicate_rows_removed": 0,
                "notes": "ok",
            }
        ),
        encoding="utf-8",
    )
    exp.train_config_path.write_text(
        json.dumps(
            {
                "num_train_epochs": 2,
                "learning_rate": 8e-5,
                "per_device_train_batch_size": 3,
                "gradient_accumulation_steps": 4,
                "cutoff_len": 2048,
                "max_samples": 5,
            }
        ),
        encoding="utf-8",
    )

    captured = {}
    monkeypatch.setattr(
        "playground.math_posttrain_datatree.core.exp.black_exp.run_llama_factory_sft",
        lambda **kwargs: captured.update({"overrides": kwargs["overrides"]}) or TrainResult(
            status="completed",
            checkpoint_path=str(task_workspace / "ckpt"),
            recipe_path=str(task_workspace / "recipe.json"),
            train_log_path=str(task_workspace / "train.log"),
            command="train",
            dry_run=True,
            metrics={},
        ),
    )
    monkeypatch.setattr(
        "playground.math_posttrain_datatree.core.exp.black_exp.run_eval",
        lambda **kwargs: EvalReport(
            status="completed",
            overall_accuracy=0.12,
            benchmark_scores={"aime_2025": 0.12},
            sample_results_path=str(task_workspace / "samples.jsonl"),
            normalized_predictions_path=str(task_workspace / "preds.jsonl"),
            metadata={"backend_details": {}},
        ),
    )
    monkeypatch.setattr(
        "playground.math_posttrain_datatree.core.exp.black_exp.run_inspect",
        lambda **kwargs: InspectReport(
            failure_clusters=[],
            weak_domains=[],
            weak_answer_styles=[],
            source_effect_hypotheses=[],
            recommended_next_action="keep",
            rationale="ok",
        ),
    )

    result = exp.run("prepare aime data")

    assert result["metric"] == 0.12
    assert (exp.pack_dir / "pack_manifest.json").exists()
    assert (exp.pack_dir / "pack_stats.json").exists()
    manifest = json.loads((exp.pack_dir / "pack_manifest.json").read_text(encoding="utf-8"))
    assert manifest["output_path"] == str(exp.train_jsonl_path)
    assert manifest["sample_count"] == 1
    effective = json.loads(exp.effective_train_config_path.read_text(encoding="utf-8"))
    assert effective["num_train_epochs"] == 2.0
    assert effective["learning_rate"] == 8e-5
    assert effective["max_samples"] == 1
    assert captured["overrides"]["max_samples"] == 1
    assert captured["overrides"]["per_device_train_batch_size"] == 3



def test_black_exp_returns_bug_on_mixed_metadata_schema_before_training(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    task_workspace = workspace
    (task_workspace / "artifacts" / "train_packs").mkdir(parents=True, exist_ok=True)

    raw_data_path = task_workspace / "materialized.jsonl"
    _write_jsonl(raw_data_path, [{"problem": "Q", "solution": "A"}])
    manifest_path = task_workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "datasets": [{
                "source_id": "src_a",
                "name": "src_a",
                "license": "mit",
                "url": "https://example.com/src_a",
                "local_path": str(raw_data_path),
                "full_local_path": str(raw_data_path),
                "coverage_tags": ["writing"],
            }]
        }),
        encoding="utf-8",
    )

    dummy_agent = SimpleNamespace(_prompt_format_kwargs={}, config=SimpleNamespace(max_turns=4))
    node = SimpleNamespace(id="blacknode_mixed_metadata", parent=None)
    config = SimpleNamespace(
        base_model="/models/mock",
        benchmark_suite=["arena_hard_writing"],
        dry_run_training=True,
        evaluation={"backend": "builtin"},
        llama_factory_env={},
    )
    exp = BlackExp(
        agent=dummy_agent,
        session=object(),
        workspace=workspace,
        task_workspace=task_workspace,
        config=config,
        node=node,
        manifest_path=manifest_path,
        inspect_report_path=None,
        global_pool_manifest_path=manifest_path,
        input_black_handoff_path=None,
    )
    exp.pack_dir.mkdir(parents=True, exist_ok=True)

    def fake_run_black_data_agent(*args, **kwargs):
        _write_jsonl(
            exp.train_jsonl_path,
            [
                {
                    "instruction": "Write carefully.",
                    "input": "Question A",
                    "output": "Answer A",
                    "metadata": {"source_id": "src_a", "raw_id": "abc-123"},
                },
                {
                    "instruction": "Write carefully.",
                    "input": "Question B",
                    "output": "Answer B",
                    "metadata": {"source_id": "src_a", "raw_id": 456},
                },
            ],
        )
        exp.prep_report_path.write_text(
            json.dumps({
                "selected_sources": ["src_a"],
                "raw_rows_seen": 2,
                "rows_written": 2,
                "duplicate_rows_removed": 0,
                "notes": "mixed raw_id types",
            }),
            encoding="utf-8",
        )
        exp.train_config_path.write_text(
            json.dumps({"learning_rate": 8e-5, "max_samples": 2}),
            encoding="utf-8",
        )
        return {}

    exp._run_black_data_agent = fake_run_black_data_agent  # type: ignore[method-assign]

    def fail_training(*args, **kwargs):
        raise AssertionError("run_llama_factory_sft should not be called when train data validation fails")

    monkeypatch.setattr(
        "playground.math_posttrain_datatree.core.exp.black_exp.run_llama_factory_sft",
        fail_training,
    )

    result = exp.run("prepare writing data")

    assert result["metric"] is None
    assert result["metric_detail"]["is_bug"] is True
    assert "metadata.raw_id" in result["raw_response"]



def test_black_exp_returns_bug_on_invalid_train_config(tmp_path, monkeypatch):
    """When the agent produces an invalid train config, the framework returns is_bug=True
    without a repair loop -- the agent should have self-validated using its tools."""
    workspace = tmp_path / "workspace"
    task_workspace = workspace
    (task_workspace / "artifacts" / "train_packs").mkdir(parents=True, exist_ok=True)

    raw_data_path = task_workspace / "materialized.jsonl"
    _write_jsonl(raw_data_path, [{"problem": "Q", "solution": "A"}])
    manifest_path = task_workspace / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "datasets": [{
                "source_id": "src_a",
                "name": "src_a",
                "license": "mit",
                "url": "https://example.com/src_a",
                "local_path": str(raw_data_path),
                "full_local_path": str(raw_data_path),
                "coverage_tags": ["algebra"],
            }]
        }),
        encoding="utf-8",
    )

    dummy_agent = SimpleNamespace(_prompt_format_kwargs={}, config=SimpleNamespace(max_turns=4))
    node = SimpleNamespace(id="blacknode2", parent=None)
    config = SimpleNamespace(
        base_model="/models/mock",
        benchmark_suite=["aime_2025"],
        dry_run_training=True,
        evaluation={"backend": "builtin"},
        llama_factory_env={},
    )
    exp = BlackExp(
        agent=dummy_agent,
        session=object(),
        workspace=workspace,
        task_workspace=task_workspace,
        config=config,
        node=node,
        manifest_path=manifest_path,
        inspect_report_path=None,
        global_pool_manifest_path=manifest_path,
        input_black_handoff_path=None,
    )
    exp.pack_dir.mkdir(parents=True, exist_ok=True)

    call_counter = {"count": 0}

    def fake_run_black_data_agent(*args, **kwargs):
        call_counter["count"] += 1
        _write_jsonl(
            exp.train_jsonl_path,
            [{
                "instruction": "Solve carefully.",
                "input": "What is 2+2?",
                "output": "Final answer: 4",
                "metadata": {"source_id": "src_a", "topic": "Algebra", "difficulty": "easy"},
            }],
        )
        exp.prep_report_path.write_text(
            json.dumps({
                "selected_sources": ["src_a"],
                "raw_rows_seen": 1,
                "rows_written": 1,
                "duplicate_rows_removed": 0,
                "notes": "ok",
            }),
            encoding="utf-8",
        )
        # Agent writes an invalid config (unknown key)
        exp.train_config_path.write_text(json.dumps({"unknown": 1}), encoding="utf-8")
        return {"rows_written": 1}

    monkeypatch.setattr(exp, "_run_black_data_agent", fake_run_black_data_agent)

    result = exp.run("prepare aime data")

    # Agent is called only once — no repair loop
    assert call_counter["count"] == 1
    # Result should be a bug since validation fails
    assert result["metric"] is None
    assert result["metric_detail"]["is_bug"] is True
    assert result["metric_detail"]["has_submission"] is False
