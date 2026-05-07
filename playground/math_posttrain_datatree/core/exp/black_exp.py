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
    validate_train_config,
)
from ..utils.memory import build_prompt_memory
from ..utils.handoff import (
    get_black_handoff_path,
    load_json_payload,
    summarize_black_handoff,
    summarize_global_pool_manifest,
)
from ..utils.eval import DEFAULT_BENCHMARK_SUITE, run_eval
from ..utils.inspect import run_inspect
from ..utils.io import read_json, write_json
from ..utils.llama_factory import run_llama_factory_sft
from . import NodeExp

logger = logging.getLogger(__name__)

RESPONSE_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)
DEFAULT_NPROC_PER_NODE = 1


def resolve_llama_factory_runtime(config_section: Any) -> tuple[str | None, str | None, dict[str, str]]:
    raw = _config_section_to_dict(config_section)
    python_bin = raw.get("python_bin")
    env_dir = raw.get("env_dir")

    env_overrides: dict[str, str] = {}
    nproc_per_node = raw.get("nproc_per_node", DEFAULT_NPROC_PER_NODE)
    if nproc_per_node is not None:
        env_overrides["NPROC_PER_NODE"] = str(int(nproc_per_node))

    cuda_visible_devices = raw.get("cuda_visible_devices")
    if cuda_visible_devices not in (None, ""):
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    return python_bin, env_dir, env_overrides


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


def _summarize_train_config_feedback(report: dict[str, Any]) -> str:
    if not isinstance(report, dict) or not report:
        return ""
    parts: list[str] = []
    status = str(report.get("status") or "").strip()
    if status:
        parts.append(f"status={status}")
    reason = _clip_text(str(report.get("reason") or ""), 180)
    if reason:
        parts.append(f"reason={reason}")
    provided_keys = report.get("provided_keys")
    if isinstance(provided_keys, list) and provided_keys:
        parts.append("keys=" + ",".join(str(item) for item in provided_keys[:6]))
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
        write_json(self.manifest_path, dataset_manifest)
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
            "Final turn. Stop exploring. Ensure the final data and config files are written now.\n"
            f"Required files: `{self.prepare_data_script_path}`, `{self.train_jsonl_path}`, `{self.prep_report_path}`, `{self.train_config_path}`.\n"
            "Output exactly one JSON block summarizing the generated training data and training config, then call the finish tool."
        )
        orig_fmt = self.agent._prompt_format_kwargs.copy()
        self.agent._prompt_format_kwargs.update(
            {
                "task_description": task_description,
                "workspace": str(self.workspace),
                "task_workspace": str(self.task_workspace),
                "manifest_path": str(self.manifest_path),
                "global_pool_manifest_path": str(self.global_pool_manifest_path),
                "global_pool_manifest_summary_json": json.dumps(global_pool_summary, ensure_ascii=False, indent=2),
                "parent_black_handoff_path": str(self.input_black_handoff_path or ""),
                "parent_black_handoff_json": json.dumps(parent_black_handoff, ensure_ascii=False, indent=2),
                "parent_black_handoff_summary_json": json.dumps(parent_black_handoff_summary, ensure_ascii=False, indent=2),
                "prepare_data_script_path": str(self.prepare_data_script_path),
                "train_jsonl_path": str(self.train_jsonl_path),
                "prep_report_path": str(self.prep_report_path),
                "train_config_path": str(self.train_config_path),
                "effective_train_config_path": str(self.effective_train_config_path),
                "probe_dir": str(self.probe_dir),
                "probe_summary_path": str(self.probe_summary_path),
                "dataset_manifest_json": json.dumps(manifest, ensure_ascii=False, indent=2),
                "probe_summary_json": json.dumps(probe_payload or {}, ensure_ascii=False, indent=2),
                "memory_summary": build_prompt_memory(self.task_workspace, self.node),
                "inspect_report_json": json.dumps(inspect_report or {}, ensure_ascii=False, indent=2),
                "inspect_summary_text": inspect_summary,
                "inspect_rationale_text": str(inspect_report.get("rationale") or ""),
                "inspect_recommended_next_action": str(inspect_report.get("recommended_next_action") or ""),
                "prep_feedback_json": "{}",
                "prep_feedback_summary": "",
                "prep_feedback_path": str(self.prep_feedback_path),
                "train_config_feedback_json": "{}",
                "train_config_feedback_summary": "",
                "train_config_feedback_path": str(self.train_config_feedback_path),
            }
        )
        try:
            traj = self._run_agent_once(
                task_description,
                enable_final_turn=True,
                enable_final_turn_prompt=final_turn_prompt,
            )
            self._recover_misplaced_agent_outputs()
            return self._extract_agent_json_summary(traj)
        finally:
            self.agent._prompt_format_kwargs = orig_fmt

    def run(self, task_description: str) -> dict:
        node_id = self.node.id
        BaseAgent.set_exp_info(exp_name=f"math_black_{node_id[:8]}", exp_index=self.exp_index)
        pack_dir = self.pack_dir
        train_dir = self.task_workspace / "artifacts" / "checkpoints" / node_id
        eval_dir = self.task_workspace / "artifacts" / "evals" / node_id
        inspect_dir = self.task_workspace / "artifacts" / "inspects" / node_id
        pack_dir.mkdir(parents=True, exist_ok=True)
        dataset_entries, probe_payload = self._prepare_black_inputs()
        agent_summary = self._run_black_data_agent(
            task_description,
            probe_payload=probe_payload,
        )
        prep_feedback = validate_prepared_train_file(
            self.train_jsonl_path,
            self.prep_report_path,
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

        train_config_feedback = validate_train_config(
            self.train_config_path,
            output_path=self.train_config_feedback_path,
            effective_output_path=self.effective_train_config_path,
        )
        if train_config_feedback.get("status") != "passed":
            message = (
                f"train config validation failed: {train_config_feedback.get('reason') or 'unknown reason'}; "
                f"see {self.train_config_feedback_path}"
            )
            return {
                "plan": message,
                "code": "",
                "raw_response": json.dumps(train_config_feedback, ensure_ascii=False),
                "exec": {"stdout": message, "exit_code": 1},
                "metric": None,
                "metric_detail": {
                    "is_bug": True,
                    "has_submission": False,
                    "prep_feedback_path": str(self.prep_feedback_path),
                    "train_config_feedback_path": str(self.train_config_feedback_path),
                    "train_jsonl_path": str(self.train_jsonl_path),
                    "prep_report_path": str(self.prep_report_path),
                    "train_config_path": str(self.train_config_path),
                    "effective_train_config_path": str(self.effective_train_config_path),
                },
            }
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

        base_model = getattr(self.config, "base_model", "Qwen/Qwen2.5-1.5B-Instruct")
        benchmark_suite = list(getattr(self.config, "benchmark_suite", DEFAULT_BENCHMARK_SUITE))
        evaluation_cfg = _config_section_to_dict(getattr(self.config, "evaluation", None))
        eval_backend = str(evaluation_cfg.get("backend", "builtin"))
        benchmark_files = _config_section_to_dict(evaluation_cfg.get("benchmark_files"))
        prediction_files = _config_section_to_dict(evaluation_cfg.get("prediction_files"))
        posttrainbench_cfg = _config_section_to_dict(evaluation_cfg.get("posttrainbench"))
        lf_python_bin, lf_env_dir, lf_env_overrides = resolve_llama_factory_runtime(
            getattr(self.config, "llama_factory_env", None)
        )
        logger.info(
            "Black node %s: starting training base_model=%s output_dir=%s dry_run=%s nproc_per_node=%s cuda_visible_devices=%s",
            node_id,
            base_model,
            train_dir,
            bool(getattr(self.config, "dry_run_training", True)),
            lf_env_overrides.get("NPROC_PER_NODE"),
            lf_env_overrides.get("CUDA_VISIBLE_DEVICES", "<inherit>"),
        )
        effective_train_config = read_json(self.effective_train_config_path, default={}) or {}
        if not isinstance(effective_train_config, dict):
            effective_train_config = {}
        training_defaults = _config_section_to_dict(getattr(self.config, "training_defaults", None))
        template_override = training_defaults.get("template") if training_defaults else None
        if training_defaults and "cutoff_len" in training_defaults:
            effective_train_config.setdefault("cutoff_len", training_defaults["cutoff_len"])
        if "max_samples" in effective_train_config:
            effective_train_config["max_samples"] = min(
                int(effective_train_config.get("max_samples") or pack_manifest.sample_count),
                pack_manifest.sample_count,
            )
        write_json(self.effective_train_config_path, effective_train_config)
        train_result = run_llama_factory_sft(
            dataset_path=alpaca_path,
            recipe_path=train_dir / "base_recipe.json",
            output_dir=train_dir,
            base_model=base_model,
            overrides=effective_train_config,
            template_override=template_override,
            dry_run=bool(getattr(self.config, "dry_run_training", True)),
            env_overrides=lf_env_overrides,
            python_bin=lf_python_bin,
            env_dir=lf_env_dir,
            merge_for_evaluation=eval_backend == "posttrainbench",
        )
        logger.info(
            "Black node %s: training finished status=%s checkpoint=%s train_log=%s",
            node_id,
            train_result.status,
            train_result.checkpoint_path,
            train_result.train_log_path,
        )
        if train_result.status != "completed":
            message = f"training failed with status={train_result.status}"
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
                    "train_result_path": str(train_dir / "train_result.json"),
                    "prep_feedback_path": str(self.prep_feedback_path),
                },
            }
        logger.info(
            "Black node %s: starting evaluation backend=%s benchmarks=%s eval_dir=%s",
            node_id,
            eval_backend,
            ",".join(benchmark_suite),
            eval_dir,
        )
        eval_report = run_eval(
            eval_dir=eval_dir,
            benchmark_suite=benchmark_suite,
            pack_manifest=pack_manifest.to_dict(),
            pack_stats=pack_stats,
            benchmark_files=benchmark_files,
            prediction_files=prediction_files,
            eval_backend=eval_backend,
            evaluation_options=posttrainbench_cfg,
            model_path=train_result.checkpoint_path,
        )
        logger.info(
            "Black node %s: evaluation finished status=%s overall_accuracy=%s eval_report=%s",
            node_id,
            eval_report.status,
            eval_report.overall_accuracy,
            eval_dir / "eval_report.json",
        )
        backend_details = (eval_report.metadata or {}).get("backend_details", {}) or {}
        failed_eval_backends = [
            benchmark_id
            for benchmark_id, detail in backend_details.items()
            if isinstance(detail, dict) and detail.get("status") == "failed"
        ]
        if failed_eval_backends:
            message = f"evaluation backend failed for: {', '.join(sorted(failed_eval_backends))}"
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
                    "train_result_path": str(train_dir / "train_result.json"),
                    "eval_report_path": str(eval_dir / "eval_report.json"),
                    "prep_feedback_path": str(self.prep_feedback_path),
                },
            }
        logger.info(
            "Black node %s: starting inspect phase output=%s",
            node_id,
            inspect_dir / "inspect_report.json",
        )
        inspect_report = run_inspect(
            eval_report=eval_report.to_dict(),
            pack_manifest=pack_manifest.to_dict(),
            pack_stats=pack_stats,
            output_path=inspect_dir / "inspect_report.json",
        )
        benchmark_feedback = _extract_benchmark_feedback(eval_report)
        benchmark_feedback_summary = _format_benchmark_feedback_summary(benchmark_feedback)
        inspect_summary = _summarize_inspect_report(inspect_report.to_dict())
        logger.info(
            "Black node %s: inspect finished recommended_next_action=%s feedback=%s",
            node_id,
            inspect_report.recommended_next_action,
            benchmark_feedback_summary or "<none>",
        )
        black_handoff_payload = {
            "schema_version": 1,
            "node_id": node_id,
            "metric": eval_report.overall_accuracy,
            "selected_sources": list(pack_manifest.source_datasets),
            "pack_manifest_path": str(pack_dir / "pack_manifest.json"),
            "pack_stats_path": str(pack_dir / "pack_stats.json"),
            "train_result_path": str(train_dir / "train_result.json"),
            "eval_report_path": str(eval_dir / "eval_report.json"),
            "inspect_report_path": str(inspect_dir / "inspect_report.json"),
            "prepare_data_script_path": str(self.prepare_data_script_path) if self.prepare_data_script_path.exists() else "",
            "train_jsonl_path": str(self.train_jsonl_path),
            "prep_report_path": str(self.prep_report_path),
            "train_config_path": str(self.train_config_path),
            "effective_train_config_path": str(self.effective_train_config_path),
            "prep_feedback_path": str(self.prep_feedback_path),
            "train_config_feedback_path": str(self.train_config_feedback_path),
            "probe_summary_path": str(self.probe_summary_path),
            "global_pool_manifest_path": str(self.global_pool_manifest_path),
            "recommended_next_action": inspect_report.recommended_next_action,
            "inspect_rationale": inspect_report.rationale,
            "inspect_summary": inspect_summary,
            "benchmark_feedback": benchmark_feedback,
            "benchmark_feedback_summary": benchmark_feedback_summary,
        }
        write_json(self.black_handoff_path, black_handoff_payload)

        stdout_lines = [
            f"pack={pack_manifest.output_path}",
            f"rows={pack_manifest.sample_count}",
            f"sources={','.join(pack_manifest.source_datasets)}",
            f"train_status={train_result.status}",
            f"overall_accuracy={eval_report.overall_accuracy}",
            f"recommended_next_action={inspect_report.recommended_next_action}",
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
            "metric": eval_report.overall_accuracy,
            "metric_detail": {
                "is_bug": False,
                "has_submission": True,
                "pack_manifest_path": str(pack_dir / "pack_manifest.json"),
                "pack_stats_path": str(pack_dir / "pack_stats.json"),
                "train_result_path": str(train_dir / "train_result.json"),
                "eval_report_path": str(eval_dir / "eval_report.json"),
                "inspect_report_path": str(inspect_dir / "inspect_report.json"),
                "prep_feedback_path": str(self.prep_feedback_path),
                "train_config_feedback_path": str(self.train_config_feedback_path),
                "probe_summary_path": str(self.probe_summary_path),
                "recommended_next_action": inspect_report.recommended_next_action,
                "inspect_failure_clusters": list(inspect_report.failure_clusters),
                "inspect_rationale": inspect_report.rationale,
                "inspect_summary": inspect_summary,
                "benchmark_feedback_summary": benchmark_feedback_summary,
                "global_pool_manifest_path": str(self.global_pool_manifest_path),
                "black_handoff_path": str(self.black_handoff_path),
                "train_jsonl_path": str(self.train_jsonl_path),
                "prep_report_path": str(self.prep_report_path),
            },
        }
