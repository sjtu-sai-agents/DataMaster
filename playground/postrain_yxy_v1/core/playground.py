from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from evomaster.agent import Agent
from evomaster.core import BasePlayground, register_playground

from .exp.black_exp import BlackExp
from .exp.red_exp import RedExp, get_dataset_manifest_path
from .utils.handoff import get_black_handoff_path, get_global_pool_manifest_path
from .utils.eval import DEFAULT_BENCHMARK_SUITE
from .utils.io import write_json
from .utils.memory import build_node_memory_index, write_node_memory
from .utils.tree_helpers import append_trajectory, save_node_snapshot, write_uct_trajectory
from .utils.uct import MetricReview, UCTSearchConfig, UCTDecayConfig, UCTSearchManager

logger = logging.getLogger(__name__)


@register_playground("posttrain_datatree")
class PostTrainDataTreePlayground(BasePlayground):
    def __init__(self, config_dir: Path | None = None, config_path: Path | None = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "posttrain_datatree"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("red_agent", "black_agent")
        self.trajectories: list[dict[str, Any]] = []

    def setup(self) -> None:
        self._setup_session()
        self._setup_agents()
        self._register_black_validation_tools()
        self._setup_workspace_directories(Path(self.session.config.workspace_path))

    def _register_black_validation_tools(self) -> None:
        from .tools import ValidateTrainConfigTool, ValidateTrainDataTool

        black_agent = self.agents.black_agent
        for tool in (ValidateTrainDataTool(), ValidateTrainConfigTool()):
            black_agent.tools.register(tool)
        if (
            black_agent.enabled_tool_names is not None
            and "*" not in black_agent.enabled_tool_names
        ):
            for name in ("validate_train_data", "validate_train_config"):
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
        base_model = getattr(self.config, "base_model", "Qwen/Qwen2.5-1.5B-Instruct")
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
            "finetuning_type": "lora",
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
            self._seed_context(task_workspace, task_description, node.id)
            return {
                "plan": "seed run context",
                "code": "",
                "raw_response": "seed completed",
                "exec": {"stdout": "seed completed", "exit_code": 0},
                "metric": 0.0,
                "metric_detail": {"is_bug": False, "has_submission": False},
            }
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
