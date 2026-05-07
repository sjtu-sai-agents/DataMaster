from __future__ import annotations

from copy import deepcopy
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from evomaster.agent import BaseAgent
from evomaster.utils.types import TaskInstance

from ..utils.data import (
    prepare_dataset_probe,
    synthesize_pack_from_prepared_train_file,
    validate_prepared_train_file,
)
from ..utils.memory import build_prompt_memory
from ..utils.prompt_compaction import summarize_dataset_manifest
from ..utils.handoff import (
    get_black_handoff_path,
    load_json_payload,
    summarize_black_handoff,
    summarize_global_pool_manifest,
)
from ..utils.io import read_json, write_json
from ..utils.eval import DEFAULT_BENCHMARK_SUITE
from ..utils.submit import SubmitError, load_best_submit, submit_training_eval
from ..utils.benchmark_metadata import get_benchmark_info
from . import NodeExp

logger = logging.getLogger(__name__)

RESPONSE_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _config_section_to_dict(section: Any) -> dict[str, Any]:
    def _convert(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {key: _convert(val) for key, val in value.items()}
        if hasattr(value, "model_dump"):
            return {key: _convert(val) for key, val in value.model_dump().items()}
        if hasattr(value, "dict"):
            return {key: _convert(val) for key, val in value.dict().items()}
        if hasattr(value, "__dict__"):
            return {
                key: _convert(val)
                for key, val in vars(value).items()
                if not key.startswith("_")
            }
        if isinstance(value, (list, tuple)):
            return [_convert(item) for item in value]
        return value

    if section is None:
        return {}
    if isinstance(section, dict):
        return _convert(section)
    converted = _convert(section)
    return converted if isinstance(converted, dict) else {}



def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clip_text(value: str | None, limit: int = 240) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _extract_benchmark_feedback(eval_report: Any) -> dict[str, Any]:
    if hasattr(eval_report, "to_dict"):
        payload = eval_report.to_dict()
    elif isinstance(eval_report, dict):
        payload = eval_report
    else:
        payload = {}
    metadata = _safe_dict(payload.get("metadata"))
    feedback = metadata.get("benchmark_feedback")
    return feedback if isinstance(feedback, dict) else {}


def _format_benchmark_feedback_summary(feedback: dict[str, Any]) -> str:
    rows: list[str] = []
    for benchmark_id, entry in feedback.items():
        if not isinstance(entry, dict):
            continue
        parts = [str(benchmark_id)]
        score = entry.get("score")
        if isinstance(score, (int, float)):
            parts.append(f"score={float(score):.4f}")
        num_correct = entry.get("num_correct")
        num_samples = entry.get("num_samples")
        if isinstance(num_correct, int) and isinstance(num_samples, int) and num_samples > 0:
            parts.append(f"correct={num_correct}/{num_samples}")
        for key, label in (
            ("format_adherence_rate", "format"),
            ("parseable_answer_rate", "parseable"),
            ("numeric_match_rate", "match"),
        ):
            value = entry.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{label}={float(value):.2f}")
        rows.append(", ".join(parts))
    return "; ".join(rows)


def _summarize_inspect_report(inspect_report: dict[str, Any]) -> str:
    if not isinstance(inspect_report, dict) or not inspect_report:
        return ""
    parts: list[str] = []
    next_action = inspect_report.get("recommended_next_action")
    if next_action:
        parts.append(f"next={next_action}")
    failure_clusters = inspect_report.get("failure_clusters")
    if isinstance(failure_clusters, list) and failure_clusters:
        parts.append("clusters=" + ",".join(str(item) for item in failure_clusters[:4]))
    weak_styles = inspect_report.get("weak_answer_styles")
    if isinstance(weak_styles, list) and weak_styles:
        parts.append("styles=" + ",".join(str(item) for item in weak_styles[:3]))
    rationale = _clip_text(str(inspect_report.get("rationale") or ""), 220)
    if rationale:
        parts.append(f"rationale={rationale}")
    return " | ".join(parts)


def _summarize_prep_feedback(report: dict[str, Any]) -> str:
    if not isinstance(report, dict) or not report:
        return ""
    parts: list[str] = []
    status = str(report.get("status") or "").strip()
    if status:
        parts.append(f"status={status}")
    reason = _clip_text(str(report.get("reason") or ""), 180)
    if reason:
        parts.append(f"reason={reason}")
    row_count = report.get("row_count")
    if isinstance(row_count, int):
        parts.append(f"rows={row_count}")
    selected_sources = report.get("selected_sources")
    if isinstance(selected_sources, list) and selected_sources:
        parts.append("sources=" + ",".join(str(item) for item in selected_sources[:4]))
    issues = report.get("issues")
    if isinstance(issues, list) and issues:
        parts.append("issues=" + ",".join(_clip_text(str(item), 80) for item in issues[:3]))
    report_path = _clip_text(str(report.get("report_path") or ""), 160)
    if report_path:
        parts.append(f"report={report_path}")
    return " | ".join(parts)


def _extract_agent_json_text(trajectory: Any) -> str:
    dialogs = None
    if isinstance(trajectory, dict):
        dialogs = trajectory.get("dialogs")
    elif hasattr(trajectory, "dialogs"):
        dialogs = trajectory.dialogs
    if not dialogs:
        return ""

    last_dialog = dialogs[-1]
    if isinstance(last_dialog, dict):
        messages = last_dialog.get("messages", [])
    else:
        messages = getattr(last_dialog, "messages", [])

    for message in reversed(messages or []):
        if isinstance(message, dict):
            role = message.get("role", "")
            content = message.get("content", "")
        else:
            role = getattr(message, "role", None)
            role = role.value if hasattr(role, "value") else str(role) if role else ""
            content = getattr(message, "content", "")
        if role != "assistant" or not content:
            continue
        match = RESPONSE_JSON_BLOCK_RE.search(content)
        if match:
            return match.group(1)
        stripped = str(content).strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
    return ""


class BlackExp(NodeExp):
    def __init__(
        self,
        agent,
        session,
        workspace: Path,
        task_workspace: Path,
        config,
        node,
        manifest_path: Path,
        inspect_report_path: Path | None,
        global_pool_manifest_path: Path | None,
        input_black_handoff_path: Path | None,
        exp_index: int = 0,
    ):
        super().__init__(agent, session, workspace, task_workspace, config, node, exp_index)
        self.manifest_path = manifest_path
        self.inspect_report_path = inspect_report_path
        self.global_pool_manifest_path = global_pool_manifest_path or manifest_path
        self.input_black_handoff_path = input_black_handoff_path
        self.pack_dir = self.task_workspace / "artifacts" / "train_packs" / self.node.id
        self.prepare_data_script_path = self.pack_dir / "prepare_data.py"
        self.train_jsonl_path = self.pack_dir / "train.jsonl"
        self.prep_report_path = self.pack_dir / "prep_report.json"
        self.train_config_path = self.pack_dir / "train_config.json"
        self.effective_train_config_path = self.pack_dir / "effective_train_config.json"
        self.probe_dir = self.pack_dir / "_probe"
        self.probe_summary_path = self.probe_dir / "probe_summary.json"
        self.prep_feedback_path = self.pack_dir / "prep_validation_report.json"
        self.train_config_feedback_path = self.pack_dir / "train_config_validation_report.json"
        self.black_handoff_path = get_black_handoff_path(self.task_workspace, self.node.id)
        self.data_links_dir = self.task_workspace / "data_links"
        self.prepared_manifest: dict[str, Any] | None = None

    def _recover_misplaced_agent_outputs(self) -> None:
        task_workspace_str = str(self.task_workspace)
        if "/runs/" not in task_workspace_str:
            return
        prefix, current_suffix = task_workspace_str.split("/runs/", 1)
        if "/" not in current_suffix:
            return
        _, workspace_suffix = current_suffix.split("/", 1)
        runs_root = Path(prefix) / "runs"
        if not runs_root.exists():
            return

        target_to_pattern = {
            self.prepare_data_script_path: f"*/{workspace_suffix}/artifacts/train_packs/{self.node.id}/prepare_data.py",
            self.train_jsonl_path: f"*/{workspace_suffix}/artifacts/train_packs/{self.node.id}/train.jsonl",
            self.prep_report_path: f"*/{workspace_suffix}/artifacts/train_packs/{self.node.id}/prep_report.json",
            self.train_config_path: f"*/{workspace_suffix}/artifacts/train_packs/{self.node.id}/train_config.json",
        }
        for target_path, pattern in target_to_pattern.items():
            if target_path.exists():
                continue
            for candidate in runs_root.glob(pattern):
                if candidate == target_path or not candidate.exists():
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                candidate.replace(target_path)
                logger.warning(
                    "Black node %s: recovered misplaced agent output %s -> %s",
                    self.node.id,
                    candidate,
                    target_path,
                )
                break

    def _load_parent_black_handoff(self) -> dict[str, Any]:
        return load_json_payload(self.input_black_handoff_path)

    def _run_agent_once(
        self,
        task_description: str,
        *,
        enable_final_turn: bool,
        enable_final_turn_prompt: str,
        final_turns_remaining: int = 1,
        max_turns_override: int | None = None,
    ):
        task = TaskInstance(
            task_id=f"{self.node.id}_black",
            task_type="black",
            description=task_description,
            input_data={},
        )
        original_max_turns = getattr(self.agent.config, "max_turns", None)
        if max_turns_override is not None and original_max_turns is not None:
            self.agent.config.max_turns = max_turns_override
        try:
            return self.agent.run(
                task,
                enable_final_turn=enable_final_turn,
                enable_final_turn_prompt=enable_final_turn_prompt,
                final_turns_remaining=final_turns_remaining,
            )
        finally:
            if max_turns_override is not None and original_max_turns is not None:
                self.agent.config.max_turns = original_max_turns

    def _extract_agent_json_summary(self, trajectory: Any) -> dict[str, Any]:
        json_text = _extract_agent_json_text(trajectory)
        if not json_text:
            return {}
        try:
            parsed = json.loads(json_text)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _prepare_black_inputs(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        dataset_manifest = read_json(self.manifest_path, default={}) or {}
        dataset_entries = dataset_manifest.get("datasets", [])
        data_access_cfg = _config_section_to_dict(getattr(self.config, "data_access", None))
        probe_payload = prepare_dataset_probe(
            dataset_entries,
            self.probe_dir,
            materialize_max_rows=None,
            data_access_config=data_access_cfg,
        )
        dataset_manifest["datasets"] = dataset_entries
        self.prepared_manifest = dataset_manifest
        return dataset_entries, probe_payload

    def _run_black_data_agent(
        self,
        task_description: str,
        *,
        probe_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.agent is None:
            return {}
        manifest = deepcopy(self.prepared_manifest) if isinstance(self.prepared_manifest, dict) else (read_json(self.manifest_path, default={}) or {})
        global_pool_manifest = load_json_payload(self.global_pool_manifest_path)
        global_pool_summary = summarize_global_pool_manifest(global_pool_manifest)
        parent_black_handoff = self._load_parent_black_handoff()
        parent_black_handoff_summary = summarize_black_handoff(parent_black_handoff)
        inspect_report = read_json(self.inspect_report_path, default={}) if self.inspect_report_path else {}
        inspect_summary = _summarize_inspect_report(inspect_report)
        final_turn_prompt = (
            "FINAL PHASE. Stop all exploration and produce final outputs NOW.\n"
            f"Required files:\n"
            f"1. `{self.train_jsonl_path}` (training data in Alpaca JSONL format)\n"
            f"Optional but recommended:\n"
            f"2. `{self.train_config_path}` (training hyperparameters, only if overriding defaults)\n"
            f"3. `{self.prepare_data_script_path}` (your data preparation script)\n"
            f"4. `{self.prep_report_path}` (preparation statistics)\n"
            "Do not run more exploratory shell checks.\n"
            "Call validate_train_data. If validation passes, you MUST call submit with "
            f"`train_data_path=\"{self.train_jsonl_path}\"`, "
            f"`benchmark=\"{self.config.benchmark_suite[0] if self.config.benchmark_suite else 'aime_2025'}\"`, "
            f"and `node_id=\"{self.node.id}\"` before calling finish.\n"
            "If submit fails, call finish with a JSON summary that includes the failure reason and artifact paths.\n"
            "Output exactly one JSON block summarizing the data preparation."
        )
        orig_fmt = self.agent._prompt_format_kwargs.copy()
        self.agent._prompt_format_kwargs.update(
            {
                "task_description": task_description,
                "workspace": str(self.workspace),
                "task_workspace": str(self.task_workspace),
                "manifest_path": str(self.manifest_path),
                "global_pool_manifest_path": str(self.global_pool_manifest_path),
                "global_pool_manifest_summary": json.dumps(global_pool_summary, ensure_ascii=False),
                "parent_black_handoff_path": str(self.input_black_handoff_path or ""),
                "parent_black_handoff_summary": json.dumps(parent_black_handoff_summary, ensure_ascii=False),
                "prepare_data_script_path": str(self.prepare_data_script_path),
                "train_jsonl_path": str(self.train_jsonl_path),
                "prep_report_path": str(self.prep_report_path),
                "train_config_path": str(self.train_config_path),
                "effective_train_config_path": str(self.effective_train_config_path),
                "probe_dir": str(self.probe_dir),
                "probe_summary_path": str(self.probe_summary_path),
                "dataset_manifest_summary": summarize_dataset_manifest(manifest),
                "memory_summary": build_prompt_memory(self.task_workspace, self.node),
                "inspect_summary_text": inspect_summary,
                "inspect_rationale_text": str(inspect_report.get("rationale") or ""),
                "inspect_recommended_next_action": str(inspect_report.get("recommended_next_action") or ""),
                "prep_feedback_json": "{}",
                "prep_feedback_summary": "",
                "prep_feedback_path": str(self.prep_feedback_path),
                "train_config_feedback_path": str(self.train_config_feedback_path),
                "data_links_dir": str(self.data_links_dir),
                "node_id": self.node.id,
                "benchmark_info": get_benchmark_info(self.config.benchmark_suite[0] if self.config.benchmark_suite else "unknown"),
            }
        )
        try:
            traj = self._run_agent_once(
                task_description,
                enable_final_turn=True,
                enable_final_turn_prompt=final_turn_prompt,
                final_turns_remaining=5,
            )
            self._recover_misplaced_agent_outputs()
            return self._extract_agent_json_summary(traj)
        finally:
            self.agent._prompt_format_kwargs = orig_fmt

    def run(self, task_description: str) -> dict:
        node_id = self.node.id
        BaseAgent.set_exp_info(exp_name=f"math_black_{node_id[:8]}", exp_index=self.exp_index)
        pack_dir = self.pack_dir
        pack_dir.mkdir(parents=True, exist_ok=True)
        self.data_links_dir.mkdir(parents=True, exist_ok=True)
        dataset_entries, probe_payload = self._prepare_black_inputs()
        agent_summary = self._run_black_data_agent(
            task_description,
            probe_payload=probe_payload,
        )
        prep_report_for_validation = self.prep_report_path if self.prep_report_path.exists() else None
        prep_feedback = validate_prepared_train_file(
            self.train_jsonl_path,
            prep_report_for_validation,
            output_path=self.prep_feedback_path,
        )
        if prep_feedback.get("status") != "passed":
            message = (
                f"prepared train file validation failed: {prep_feedback.get('reason') or 'unknown reason'}; "
                f"see {self.prep_feedback_path}"
            )
            return {
                "plan": message,
                "code": "",
                "raw_response": json.dumps(prep_feedback, ensure_ascii=False),
                "exec": {"stdout": message, "exit_code": 1},
                "metric": None,
                "metric_detail": {
                    "is_bug": True,
                    "has_submission": False,
                    "prep_feedback_path": str(self.prep_feedback_path),
                    "probe_summary_path": str(self.probe_summary_path),
                    "train_jsonl_path": str(self.train_jsonl_path),
                    "prep_report_path": str(self.prep_report_path),
                    "train_config_path": str(self.train_config_path),
                    "train_config_feedback_path": str(self.train_config_feedback_path),
                },
            }

        if self.train_config_path.exists():
            train_config_payload = read_json(self.train_config_path, default={}) or {}
            if not isinstance(train_config_payload, dict):
                train_config_payload = {}
        else:
            train_config_payload = {}
        write_json(self.effective_train_config_path, train_config_payload)
        if not self.prep_report_path.exists():
            auto_row_count = sum(1 for _ in open(self.train_jsonl_path))
            auto_report = {
                "selected_sources": [],
                "raw_rows_seen": auto_row_count,
                "rows_written": auto_row_count,
                "duplicate_rows_removed": 0,
                "notes": "auto-generated: agent did not produce prep_report.json",
            }
            write_json(self.prep_report_path, auto_report)
        pack_manifest, pack_stats, alpaca_path = synthesize_pack_from_prepared_train_file(
            dataset_entries,
            self.train_jsonl_path,
            self.prep_report_path,
            pack_id=f"pack_{node_id}",
        )
        pack_stats["prep_feedback_path"] = str(self.prep_feedback_path)
        pack_stats["probe_summary_path"] = str(self.probe_summary_path)
        pack_stats["prep_feedback_status"] = str(prep_feedback.get("status") or "")
        write_json(pack_dir / "pack_manifest.json", pack_manifest.to_dict())
        write_json(pack_dir / "pack_stats.json", pack_stats)
        logger.info(
            "Black node %s: synthesized train pack ready samples=%s alpaca=%s manifest=%s",
            node_id,
            pack_manifest.sample_count,
            alpaca_path,
            pack_dir / "pack_manifest.json",
        )

        if pack_manifest.sample_count <= 0:
            message = (
                f"empty train pack generated from train_jsonl={self.train_jsonl_path}; "
                "prepared training file resolved to zero usable rows"
            )
            return {
                "plan": message,
                "code": "",
                "raw_response": message,
                "exec": {"stdout": message, "exit_code": 1},
                "metric": None,
                "metric_detail": {
                    "is_bug": True,
                    "has_submission": False,
                    "pack_manifest_path": str(pack_dir / "pack_manifest.json"),
                    "pack_stats_path": str(pack_dir / "pack_stats.json"),
                    "prep_feedback_path": str(self.prep_feedback_path),
                },
            }

        benchmark_suite = list(getattr(self.config, "benchmark_suite", DEFAULT_BENCHMARK_SUITE))
        benchmark = benchmark_suite[0] if benchmark_suite else "aime_2025"
        effective_train_config = read_json(self.effective_train_config_path, default={}) or {}
        if not isinstance(effective_train_config, dict):
            effective_train_config = {}
        if "max_samples" in effective_train_config:
            effective_train_config["max_samples"] = min(
                int(effective_train_config.get("max_samples") or pack_manifest.sample_count),
                pack_manifest.sample_count,
            )
        write_json(self.effective_train_config_path, effective_train_config)
        best_submit = load_best_submit(self.task_workspace, node_id)
        if best_submit is not None:
            submit_payload = best_submit
            logger.info(
                "Black node %s: using existing best submit score=%s trial=%s",
                node_id,
                submit_payload.get("score"),
                submit_payload.get("trial_path"),
            )
        else:
            logger.info(
                "Black node %s: submitting training/eval benchmark=%s train_data=%s train_config=%s",
                node_id,
                benchmark,
                alpaca_path,
                self.effective_train_config_path,
            )
            try:
                submit_result = submit_training_eval(
                    task_workspace=self.task_workspace,
                    config=self.config,
                    node_id=node_id,
                    benchmark=benchmark,
                    train_config=self.effective_train_config_path,
                    train_data_path=alpaca_path,
                    pack_manifest=pack_manifest.to_dict(),
                    pack_stats=pack_stats,
                )
                submit_payload = submit_result.to_dict()
            except SubmitError as exc:
                detail = exc.result or {}
                message = str(exc)
                return {
                    "plan": f"prepared {pack_manifest.sample_count} rows from {len(pack_manifest.source_datasets)} sources",
                    "code": "",
                    "raw_response": message,
                    "exec": {"stdout": message, "exit_code": 1},
                    "metric": None,
                    "metric_detail": {
                        "is_bug": True,
                        "has_submission": False,
                        "pack_manifest_path": str(pack_dir / "pack_manifest.json"),
                        "pack_stats_path": str(pack_dir / "pack_stats.json"),
                        "train_result_path": detail.get("train_result_path"),
                        "eval_report_path": detail.get("eval_report_path"),
                        "trial_path": detail.get("trial_path"),
                        "prep_feedback_path": str(self.prep_feedback_path),
                    },
                }
            best_submit = load_best_submit(self.task_workspace, node_id)
            if best_submit is not None and isinstance(best_submit.get("score"), (int, float)):
                submit_payload = best_submit

        submit_score = submit_payload.get("score")
        recommended_next_action = str(submit_payload.get("recommended_next_action") or "")
        inspect_summary = str(submit_payload.get("inspect_summary") or "")
        benchmark_feedback_summary = str(submit_payload.get("benchmark_feedback_summary") or "")
        black_handoff_payload = {
            "schema_version": 1,
            "node_id": node_id,
            "metric": submit_score,
            "selected_sources": list(pack_manifest.source_datasets),
            "global_pool_manifest_path": str(self.global_pool_manifest_path),
            "recommended_next_action": recommended_next_action,
            "inspect_summary": inspect_summary,
            "benchmark_feedback_summary": benchmark_feedback_summary,
            "best_submit": {
                "trial_path": submit_payload.get("trial_path"),
                "score": submit_score,
                "recipe_path": submit_payload.get("recipe_path"),
                "train_config_path": submit_payload.get("train_config_path"),
                "train_data_path": submit_payload.get("train_data_path"),
                "checkpoint_path": submit_payload.get("checkpoint_path"),
                "eval_report_path": submit_payload.get("eval_report_path"),
            },
        }
        write_json(self.black_handoff_path, black_handoff_payload)

        stdout_lines = [
            f"pack={pack_manifest.output_path}",
            f"rows={pack_manifest.sample_count}",
            f"sources={','.join(pack_manifest.source_datasets)}",
            f"submit_status={submit_payload.get('status')}",
            f"overall_accuracy={submit_score}",
            f"recommended_next_action={recommended_next_action}",
            f"submit_trial={submit_payload.get('trial_path')}",
            f"recipe_path={submit_payload.get('recipe_path')}",
            f"eval_report_path={submit_payload.get('eval_report_path')}",
        ]
        prep_feedback_summary = _summarize_prep_feedback(prep_feedback)
        if prep_feedback_summary:
            stdout_lines.append(f"prep={prep_feedback_summary}")
        if benchmark_feedback_summary:
            stdout_lines.append(f"feedback={benchmark_feedback_summary}")
        if inspect_summary:
            stdout_lines.append(f"inspect={inspect_summary}")

        return {
            "plan": f"prepared {pack_manifest.sample_count} rows from {len(pack_manifest.source_datasets)} sources",
            "code": "",
            "raw_response": json.dumps(agent_summary or read_json(self.prep_report_path, default={}) or {}, ensure_ascii=False),
            "exec": {
                "stdout": "\n".join(stdout_lines),
                "exit_code": 0,
            },
            "metric": submit_score,
            "metric_detail": {
                "is_bug": False,
                "has_submission": True,
                "pack_manifest_path": str(pack_dir / "pack_manifest.json"),
                "pack_stats_path": str(pack_dir / "pack_stats.json"),
                "train_result_path": submit_payload.get("train_result_path"),
                "eval_report_path": submit_payload.get("eval_report_path"),
                "inspect_report_path": submit_payload.get("inspect_report_path"),
                "submit_trial_path": submit_payload.get("trial_path"),
                "submit_train_data_path": submit_payload.get("train_data_path"),
                "submit_train_config_path": submit_payload.get("train_config_path"),
                "submit_recipe_path": submit_payload.get("recipe_path"),
                "checkpoint_path": submit_payload.get("checkpoint_path"),
                "prep_feedback_path": str(self.prep_feedback_path),
                "train_config_feedback_path": str(self.train_config_feedback_path),
                "probe_summary_path": str(self.probe_summary_path),
                "recommended_next_action": recommended_next_action,
                "inspect_summary": inspect_summary,
                "benchmark_feedback_summary": benchmark_feedback_summary,
                "global_pool_manifest_path": str(self.global_pool_manifest_path),
                "black_handoff_path": str(self.black_handoff_path),
                "train_jsonl_path": str(self.train_jsonl_path),
                "prep_report_path": str(self.prep_report_path),
            },
        }
