from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from evomaster.agent import Agent
from evomaster.core import BasePlayground, register_playground

from .exp.black_exp import BlackExp
from .exp.red_exp import RedExp, get_dataset_manifest_path
from .utils.handoff import get_black_handoff_path, get_global_pool_manifest_path
from .utils.eval import DEFAULT_BENCHMARK_SUITE, run_eval
from .utils.inspect import run_inspect
from .utils.io import write_json
from .utils.memory import build_node_memory_index, write_node_memory
from .utils.tree_helpers import append_trajectory, save_node_snapshot, write_uct_trajectory
from .utils.uct import MetricReview, UCTSearchConfig, UCTDecayConfig, UCTSearchManager

logger = logging.getLogger(__name__)


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
            return {key: _convert(val) for key, val in vars(value).items() if not key.startswith("_")}
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


@register_playground("math_posttrain_datatree_v2")
class MathPostTrainDataTreeV2Playground(BasePlayground):
    def __init__(self, config_dir: Path | None = None, config_path: Path | None = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "math_posttrain_datatree_v2"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("red_agent", "black_agent")
        self.trajectories: list[dict[str, Any]] = []

    def setup(self) -> None:
        self._setup_session()
        self._setup_agents()
        self._register_red_probe_tools()
        self._register_black_validation_tools()
        self._setup_workspace_directories(Path(self.session.config.workspace_path))

    def _register_red_probe_tools(self) -> None:
        from .tools import ProbeDatasetRowsTool, WriteDatasetManifestTool

        red_agent = self.agents.red_agent
        for tool in (ProbeDatasetRowsTool(), WriteDatasetManifestTool()):
            red_agent.tools.register(tool)
        if (
            red_agent.enabled_tool_names is not None
            and "*" not in red_agent.enabled_tool_names
        ):
            for name in ("probe_dataset_rows", "write_dataset_manifest"):
                if name not in red_agent.enabled_tool_names:
                    red_agent.enabled_tool_names.append(name)

    def _register_black_validation_tools(self) -> None:
        from .tools import SubmitTool, ValidateTrainDataTool

        black_agent = self.agents.black_agent
        for tool in (ValidateTrainDataTool(), SubmitTool(self.config)):
            black_agent.tools.register(tool)
        if (
            black_agent.enabled_tool_names is not None
            and "*" not in black_agent.enabled_tool_names
        ):
            for name in ("validate_train_data", "submit"):
                if name not in black_agent.enabled_tool_names:
                    black_agent.enabled_tool_names.append(name)

    def _create_worker_agents(self, worker_index: int) -> dict[str, Agent]:
        return {
            "red": self.copy_agent(self.agents.red_agent, new_agent_name=f"math_red_{worker_index}"),
            "black": self.copy_agent(self.agents.black_agent, new_agent_name=f"math_black_{worker_index}"),
        }

    def _setup_workspace_directories(self, workspace: Path) -> None:
        for subdir in (
            "artifacts/manifests",
            "artifacts/train_packs",
            "artifacts/checkpoints",
            "artifacts/evals",
            "artifacts/submits",
            "artifacts/inspects",
            "artifacts/reports",
            "artifacts/memory",
            "data_sources",
            "data_links",
            "memory_tree",
        ):
            (workspace / subdir).mkdir(parents=True, exist_ok=True)

    def _search_config(self) -> UCTSearchConfig:
        return UCTSearchConfig(
            num_red=int(getattr(self.config, "num_red", 1)),
            num_black=int(getattr(self.config, "num_black", 2)),
            max_black_per_red=int(getattr(self.config, "max_black_per_red", 3)),
            max_rounds=int(getattr(self.config, "max_rounds", 8)),
        )

    def _decay_config(self) -> UCTDecayConfig:
        """Build UCTDecayConfig from config.uct_decay section."""
        decay_cfg = getattr(self.config, "uct_decay", {})
        return UCTDecayConfig(
            decay_type=decay_cfg.get("decay_type", "piecewise"),
            exploration_constant=float(decay_cfg.get("exploration_constant", 1.414)),
            lower_bound=float(decay_cfg.get("lower_bound", 0.5)),
            linear_alpha=float(decay_cfg.get("linear_alpha", 0.01)),
            exponential_gamma=float(decay_cfg.get("exponential_gamma", 0.99)),
            piecewise_alpha=float(decay_cfg.get("piecewise_alpha", 0.01)),
            piecewise_phase_ratios=tuple(decay_cfg.get("piecewise_phase_ratios", [0.3, 0.7])),
        )

    def _seed_context(self, workspace: Path, task_description: str, node_id: str) -> None:
        benchmark_suite = list(getattr(self.config, "benchmark_suite", DEFAULT_BENCHMARK_SUITE))
        base_model = getattr(
            self.config,
            "base_model",
            "${BASE_MODEL_PATH}",
        )
        run_context = {
            "task_description": task_description,
            "base_model": base_model,
            "benchmark_suite": benchmark_suite,
            "evaluation": getattr(self.config, "evaluation", {}),
            "training_mode": getattr(self.config, "training_mode", "lora_sft"),
            "lf_dataset_format": getattr(self.config, "lf_dataset_format", "alpaca"),
            "created_from_node": node_id,
        }
        benchmark_registry = {
            "benchmarks": [
                {"benchmark_id": item, "scoring": "final_answer_accuracy"}
                for item in benchmark_suite
            ]
        }
        base_recipe = {
            "stage": "sft",
            "finetuning_type": "full"
            if getattr(self.config, "training_mode", "lora_sft") == "full_sft"
            else "lora",
            "dataset_format": "alpaca",
            "base_model": base_model,
        }
        tree_state = {
            "root_stage": "seed",
            "num_red": getattr(self.config, "num_red", 1),
            "num_black": getattr(self.config, "num_black", 2),
            "max_black_per_red": getattr(self.config, "max_black_per_red", 3),
        }
        write_json(workspace / "artifacts" / "reports" / "run_context.json", run_context)
        write_json(workspace / "artifacts" / "reports" / "benchmark_registry.json", benchmark_registry)
        write_json(workspace / "artifacts" / "reports" / "base_recipe.yaml", base_recipe)
        write_json(workspace / "artifacts" / "reports" / "tree_state.json", tree_state)

    def _run_seed_node(self, *, task_workspace: Path, task_description: str, node: Any) -> dict[str, Any]:
        node_id = node.id
        self._seed_context(task_workspace, task_description, node_id)

        base_model = getattr(
            self.config,
            "base_model",
            "${BASE_MODEL_PATH}",
        )
        benchmark_suite = list(getattr(self.config, "benchmark_suite", DEFAULT_BENCHMARK_SUITE))
        evaluation_cfg = _config_section_to_dict(getattr(self.config, "evaluation", None))
        eval_backend = str(evaluation_cfg.get("backend", "builtin"))
        benchmark_files = _config_section_to_dict(evaluation_cfg.get("benchmark_files"))
        prediction_files = _config_section_to_dict(evaluation_cfg.get("prediction_files"))
        posttrainbench_cfg = _config_section_to_dict(evaluation_cfg.get("posttrainbench"))

        eval_dir = task_workspace / "artifacts" / "evals" / f"seed_{node_id}"
        inspect_dir = task_workspace / "artifacts" / "inspects" / f"seed_{node_id}"
        handoff_path = get_black_handoff_path(task_workspace, node_id)
        pack_manifest = {
            "pack_id": f"seed_baseline_{node_id}",
            "source_datasets": [],
            "sample_count": 0,
            "short_answer_count": 0,
            "long_reasoning_count": 0,
            "dedup_rule": "none",
            "answer_normalization_rule": "none",
            "format": "base_model_eval",
            "output_path": "",
            "source_weights": {},
            "coverage_tags": [],
            "strategy": {"baseline": "base_model"},
        }
        pack_stats = {
            "sample_count": 0,
            "source_count": 0,
            "style_distribution": {},
            "duplicate_rate": 0.0,
        }

        self.logger.info(
            "Seed node %s: starting base model evaluation backend=%s model=%s benchmarks=%s eval_dir=%s",
            node_id,
            eval_backend,
            base_model,
            ",".join(benchmark_suite),
            eval_dir,
        )
        eval_report = run_eval(
            eval_dir=eval_dir,
            benchmark_suite=benchmark_suite,
            pack_manifest=pack_manifest,
            pack_stats=pack_stats,
            benchmark_files=benchmark_files,
            prediction_files=prediction_files,
            eval_backend=eval_backend,
            evaluation_options=posttrainbench_cfg,
            model_path=base_model,
        )
        backend_details = (eval_report.metadata or {}).get("backend_details", {}) or {}
        failed_eval_backends = [
            benchmark_id
            for benchmark_id, detail in backend_details.items()
            if isinstance(detail, dict) and detail.get("status") == "failed"
        ]

        inspect_report = run_inspect(
            eval_report=eval_report.to_dict(),
            pack_manifest=pack_manifest,
            pack_stats=pack_stats,
            output_path=inspect_dir / "inspect_report.json",
        )
        benchmark_feedback = _extract_benchmark_feedback(eval_report)
        benchmark_feedback_summary = _format_benchmark_feedback_summary(benchmark_feedback)
        inspect_summary = _summarize_inspect_report(inspect_report.to_dict())

        handoff_payload = {
            "schema_version": 1,
            "node_id": node_id,
            "role": "seed_base_model_baseline",
            "metric": eval_report.overall_accuracy,
            "selected_sources": [],
            "eval_report_path": str(eval_dir / "eval_report.json"),
            "inspect_report_path": str(inspect_dir / "inspect_report.json"),
            "global_pool_manifest_path": str(get_global_pool_manifest_path(task_workspace)),
            "recommended_next_action": inspect_report.recommended_next_action,
            "inspect_rationale": inspect_report.rationale,
            "inspect_summary": inspect_summary,
            "benchmark_feedback": benchmark_feedback,
            "benchmark_feedback_summary": benchmark_feedback_summary,
            "base_model": str(base_model),
        }
        write_json(handoff_path, handoff_payload)

        stdout_lines = [
            f"seed_base_model={base_model}",
            f"eval_status={eval_report.status}",
            f"overall_accuracy={eval_report.overall_accuracy}",
            f"recommended_next_action={inspect_report.recommended_next_action}",
            f"eval_report={eval_dir / 'eval_report.json'}",
        ]
        if failed_eval_backends:
            stdout_lines.append(f"failed_eval_backends={','.join(sorted(failed_eval_backends))}")
        if benchmark_feedback_summary:
            stdout_lines.append(f"feedback={benchmark_feedback_summary}")
        if inspect_summary:
            stdout_lines.append(f"inspect={inspect_summary}")

        return {
            "plan": "seed base model evaluation",
            "code": "",
            "raw_response": json.dumps(handoff_payload, ensure_ascii=False),
            "exec": {"stdout": "\n".join(stdout_lines), "exit_code": 1 if failed_eval_backends else 0},
            "metric": eval_report.overall_accuracy,
            "metric_detail": {
                "is_bug": bool(failed_eval_backends),
                "has_submission": True,
                "eval_report_path": str(eval_dir / "eval_report.json"),
                "inspect_report_path": str(inspect_dir / "inspect_report.json"),
                "black_handoff_path": str(handoff_path),
                "global_pool_manifest_path": str(get_global_pool_manifest_path(task_workspace)),
                "recommended_next_action": inspect_report.recommended_next_action,
                "inspect_summary": inspect_summary,
                "benchmark_feedback_summary": benchmark_feedback_summary,
                "base_model": str(base_model),
            },
        }

    def _run_one_node(
        self,
        *,
        worker_agents: dict[str, Agent],
        task_workspace: Path,
        stage: str,
        node: Any,
        task_description: str,
    ) -> dict[str, Any]:
        if stage == "seed":
            return self._run_seed_node(
                task_workspace=task_workspace,
                task_description=task_description,
                node=node,
            )
        if stage == "red":
            search_goal = getattr(node, "search_goal", "Find public supervised training data for current weak benchmark domains.")
            exp = RedExp(
                worker_agents["red"],
                self.session,
                task_workspace,
                task_workspace,
                self.config,
                node,
                Path(getattr(node, "output_manifest_path")),
                search_goal,
                Path(getattr(node, "global_pool_manifest_path", get_global_pool_manifest_path(task_workspace))),
                Path(getattr(node, "input_black_handoff_path")) if getattr(node, "input_black_handoff_path", None) else None,
            )
            return exp.run(task_description)
        if stage == "black":
            exp = BlackExp(
                worker_agents["black"],
                self.session,
                task_workspace,
                task_workspace,
                self.config,
                node,
                Path(getattr(node, "bound_manifest_path")),
                Path(getattr(node, "input_inspect_report_path")) if getattr(node, "input_inspect_report_path", None) else None,
                Path(getattr(node, "global_pool_manifest_path", get_global_pool_manifest_path(task_workspace))),
                Path(getattr(node, "input_black_handoff_path")) if getattr(node, "input_black_handoff_path", None) else None,
            )
            return exp.run(task_description)
        raise ValueError(f"Unsupported stage: {stage}")

    def _review_for_stage(self, stage: str, res: dict[str, Any]) -> MetricReview:
        if stage == "red":
            return MetricReview(
                metric=None,
                is_bug=not res.get("metric_detail", {}).get("manifest_ok", False),
                has_submission=False,
                summary=res.get("exec", {}).get("stdout", ""),
                raw_output=res.get("raw_response"),
            )
        return MetricReview(
            metric=res.get("metric"),
            is_bug=res.get("metric_detail", {}).get("is_bug", False),
            has_submission=res.get("metric_detail", {}).get("has_submission", False),
            summary=res.get("exec", {}).get("stdout", ""),
            raw_output=res.get("raw_response"),
        )

    def _make_child(self, search_mgr: UCTSearchManager, parent: Any, stage: str, task_workspace: Path) -> Any:
        child = search_mgr.create_child(parent, stage=stage, plan="", code="")
        if stage == "red":
            child.search_goal = getattr(
                parent,
                "red_search_goal",
                "Search public task-aligned data that improves weak benchmark domains without changing training code.",
            )
            child.output_manifest_path = str(get_dataset_manifest_path(task_workspace, child.id))
            child.global_pool_manifest_path = str(
                getattr(parent, "global_pool_manifest_path", get_global_pool_manifest_path(task_workspace))
            )
            child.input_inspect_report_path = getattr(parent, "inspect_report_path", None)
            child.input_black_handoff_path = getattr(
                parent,
                "black_handoff_path",
                getattr(parent, "input_black_handoff_path", None),
            )
        elif stage == "black":
            if parent.stage == "red":
                child.bound_manifest_path = getattr(
                    parent,
                    "global_pool_manifest_path",
                    getattr(parent, "output_manifest_path"),
                )
                child.bound_red_node_id = parent.id
                child.global_pool_manifest_path = getattr(
                    parent,
                    "global_pool_manifest_path",
                    str(get_global_pool_manifest_path(task_workspace)),
                )
                child.input_inspect_report_path = getattr(parent, "input_inspect_report_path", None)
                child.input_black_handoff_path = getattr(parent, "input_black_handoff_path", None)
            else:
                child.bound_manifest_path = getattr(parent, "bound_manifest_path")
                child.bound_red_node_id = getattr(parent, "bound_red_node_id", None)
                child.global_pool_manifest_path = getattr(
                    parent,
                    "global_pool_manifest_path",
                    getattr(parent, "bound_manifest_path", str(get_global_pool_manifest_path(task_workspace))),
                )
                child.input_inspect_report_path = getattr(parent, "inspect_report_path", None)
                child.input_black_handoff_path = getattr(
                    parent,
                    "black_handoff_path",
                    getattr(parent, "input_black_handoff_path", None),
                )
        search_mgr.push_execution_node(child)
        save_node_snapshot(
            self.run_dir,
            task_workspace,
            child,
            MetricReview(metric=None, is_bug=False, has_submission=False, summary="created"),
            0.0,
            search_mgr,
            snapshot_event="created",
            task_description=getattr(self, "task_description", None),
        )
        return child

    def _select_stages_batch(self, node: Any, search_mgr: UCTSearchManager) -> list[str]:
        cfg = search_mgr.search_cfg
        if node.stage == "root":
            return ["seed"]
        if node.is_buggy:
            return []
        if node.stage == "seed":
            red_children = [child for child in node.children if child.stage == "red"]
            return ["red"] if len(red_children) < cfg.num_red else []
        if node.stage == "red":
            black_children = [
                child for child in node.children
                if child.stage == "black" and getattr(child, "bound_red_node_id", None) == node.id
            ]
            remaining = max(cfg.num_black - len(black_children), 0)
            return ["black"] * remaining
        if node.stage == "black":
            recommended = getattr(node, "recommended_next_action", "expand_black")
            bound_red_node_id = getattr(node, "bound_red_node_id", None)
            black_count = search_mgr.count_black_nodes_for_red(bound_red_node_id)
            red_children = [child for child in node.children if child.stage == "red"]
            if recommended == "expand_red" and len(red_children) < cfg.num_red:
                return ["red"]
            if recommended == "expand_black" and black_count < cfg.max_black_per_red:
                return ["black"]
            if len(red_children) < cfg.num_red:
                return ["red"]
        return []

    def _expand_after_completion(self, node: Any, search_mgr: UCTSearchManager, workspace: Path) -> None:
        for stage in self._select_stages_batch(node, search_mgr):
            self._make_child(search_mgr, node, stage, workspace)

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        self.task_description = task_description
        self.setup()
        self._setup_trajectory_file(output_file)
        workspace = Path(self.session.config.workspace_path)
        self._setup_workspace_directories(workspace)
        search_mgr = UCTSearchManager(self._search_config(), self._decay_config())
        worker_agents = self._create_worker_agents(0)
        results: dict[str, list[dict[str, Any]]] = {"seed": [], "red": [], "black": []}
        state_lock = threading.Lock()
        had_failures = False

        self._make_child(search_mgr, search_mgr.root, "seed", workspace)

        while search_mgr.current_step < search_mgr.search_cfg.max_rounds:
            node = search_mgr.pop_execution_node()
            if node is None:
                break
            try:
                res = self._run_one_node(
                    worker_agents=worker_agents,
                    task_workspace=workspace,
                    stage=node.stage,
                    node=node,
                    task_description=task_description,
                )
            except Exception as exc:
                self.logger.error("Node %s failed: %s", node.id, exc, exc_info=True)
                res = {
                    "plan": "",
                    "code": "",
                    "raw_response": str(exc),
                    "exec": {"stdout": str(exc), "exit_code": -1},
                    "metric": None,
                    "metric_detail": {"is_bug": True, "has_submission": False},
                }

            with state_lock:
                node.stdout = res.get("exec", {}).get("stdout", "")
                node.plan = str(res.get("plan") or "")
                node.analysis = str(res.get("raw_response") or res.get("exec", {}).get("stdout", ""))
                node.exit_code = res.get("exec", {}).get("exit_code")
                review = self._review_for_stage(node.stage, res)
                reward = search_mgr.ingest_result(node, review)
                detail = res.get("metric_detail", {}) or {}
                had_failures = had_failures or review.is_bug or (node.exit_code not in (None, 0))
                if node.stage == "seed":
                    node.eval_report_path = detail.get("eval_report_path")
                    node.inspect_report_path = detail.get("inspect_report_path")
                    node.black_handoff_path = detail.get(
                        "black_handoff_path",
                        getattr(node, "black_handoff_path", str(get_black_handoff_path(workspace, node.id))),
                    )
                    node.global_pool_manifest_path = detail.get(
                        "global_pool_manifest_path",
                        getattr(node, "global_pool_manifest_path", str(get_global_pool_manifest_path(workspace))),
                    )
                    node.recommended_next_action = detail.get("recommended_next_action")
                if node.stage == "red":
                    node.output_manifest_path = detail.get("manifest_path", getattr(node, "output_manifest_path", None))
                    node.global_pool_manifest_path = detail.get(
                        "global_pool_manifest_path",
                        getattr(node, "global_pool_manifest_path", None),
                    )
                if node.stage == "black":
                    node.pack_manifest_path = detail.get("pack_manifest_path")
                    node.pack_stats_path = detail.get("pack_stats_path")
                    node.eval_report_path = detail.get("eval_report_path")
                    node.inspect_report_path = detail.get("inspect_report_path")
                    node.black_handoff_path = detail.get(
                        "black_handoff_path",
                        getattr(node, "black_handoff_path", str(get_black_handoff_path(workspace, node.id))),
                    )
                    node.global_pool_manifest_path = detail.get(
                        "global_pool_manifest_path",
                        getattr(node, "global_pool_manifest_path", None),
                    )
                    node.recommended_next_action = detail.get("recommended_next_action")
                write_node_memory(workspace, node, res, review)
                build_node_memory_index(workspace)
                save_node_snapshot(
                    self.run_dir,
                    workspace,
                    node,
                    review,
                    reward,
                    search_mgr,
                    snapshot_event="completed",
                    task_description=task_description,
                )
                append_trajectory(
                    self,
                    {
                        "stage": node.stage,
                        "node_id": node.id,
                        "metric": node.metric.value,
                        "recommended_next_action": getattr(node, "recommended_next_action", None),
                    },
                )
                results[node.stage].append(res)
                self._expand_after_completion(node, search_mgr, workspace)

        final_status = "failed" if had_failures else "completed"
        final = {
            "status": final_status,
            "best_metric": search_mgr.best_metric,
            "had_failures": had_failures,
            "best_node_id": search_mgr.best_node.id if search_mgr.best_node else None,
            "seed": results["seed"],
            "red": results["red"],
            "black": results["black"],
        }
        write_json(workspace / "artifacts" / "reports" / "final_report.json", final)
        write_uct_trajectory(workspace, self.trajectories)
        return final
