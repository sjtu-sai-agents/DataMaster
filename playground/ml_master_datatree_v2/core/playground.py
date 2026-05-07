"""DataTree Playground v2: Red-Scout + Skilled-Black architecture.

与 v1 的核心区别：
1. Red 节点只做数据侦察 → 写 data_manifest.json，不跑训练
2. Red 的成功判定：manifest 存在且有效（不看 F1）
3. Black 节点从 manifest 读取外部数据，并通过 SkillRegistry/use_skill 获取数据处理规程
4. task_workspace（任务级共享目录）传入 Red/Black，用于 manifest 共享
"""

from __future__ import annotations

import json
import logging
import threading
import time
from functools import partial
from pathlib import Path
from typing import Optional, Any, Callable
from datetime import datetime

from evomaster.core import BasePlayground, register_playground
from evomaster.agent import Agent

from playground.ml_master_datatree.core.utils.grading import (
    shutdown_embedded_grading_server,
    validate_submission,
)
from playground.ml_master_datatree.core.utils.uct import (
    UCTSearchConfig,
    UCTDecayConfig,
    UCTSearchManager,
    MetricReview,
)
from playground.ml_master_datatree.core.utils.data_preview import (
    generate as generate_data_preview,
)
from playground.ml_master_datatree.core.utils.playground_helpers import (
    append_trajectory,
    build_review,
    copy_submission,
    save_best,
    save_node_snapshot,
)
from .exp.initial_exp import InitialExp
from .exp.red_exp import RedExp, get_manifest_path
from .exp.black_exp import BlackExp

logger = logging.getLogger(__name__)


@register_playground("ml_master_datatree_v2")
class DataTreePlaygroundV2(BasePlayground):
    """Red-Scout + Skilled-Black playground."""

    def __init__(self, config_dir: Path | None = None, config_path: Path | None = None):
        if config_path is None and config_dir is None:
            config_dir = (
                Path(__file__).parent.parent.parent.parent
                / "configs"
                / "ml_master_datatree"
            )
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("initial_agent", "black_agent", "red_agent", "metric_agent")
        self.trajectories: list[dict[str, Any]] = []
        session_config = self.config.session.get("local", {})
        parallel_config = session_config.get("parallel", {})
        if parallel_config.get("enabled", False):
            self.max_workers = int(parallel_config.get("max_parallel", 1))
        else:
            self.max_workers = 1

    # ------------------------------------------------------------------ #
    # Setup / Teardown
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        self.logger.info("Setting up DataTreePlaygroundV2...")
        self._setup_session()
        self._setup_agents()
        required = ["initial_agent", "black_agent", "red_agent", "metric_agent"]
        missing = [s for s in required if self.agents.get(s) is None]
        if missing:
            raise ValueError(f"config.agents 缺少必需配置: {missing}")
        self._ensure_prepared_links(Path(self.session.config.workspace_path))

    def cleanup(self) -> None:
        try:
            shutdown_embedded_grading_server(timeout=5)
        except Exception as exc:
            self.logger.warning("Failed to shutdown grading server: %s", exc)
        super().cleanup()

    # ------------------------------------------------------------------ #
    # Worker agent factory
    # ------------------------------------------------------------------ #

    def _create_worker_agents(self, worker_index: int) -> dict[str, Agent]:
        return {
            "initial": self.copy_agent(
                self.agents.initial_agent,
                new_agent_name=f"initial_worker_{worker_index}",
            ),
            "black": self.copy_agent(
                self.agents.black_agent,
                new_agent_name=f"black_worker_{worker_index}",
            ),
            "red": self.copy_agent(
                self.agents.red_agent,
                new_agent_name=f"red_worker_{worker_index}",
            ),
            "metric": self.copy_agent(
                self.agents.metric_agent,
                new_agent_name=f"metric_worker_{worker_index}",
            ),
        }

    # ------------------------------------------------------------------ #
    # Workspace helpers (re-used from v1 via delegation)
    # ------------------------------------------------------------------ #

    def _ensure_prepared_links(self, workspace: Path) -> None:
        """Soft-link shared metadata files; public data is exposed as workspace/input by session symlinks."""
        exp_id = getattr(self.config, "exp_id", None)
        data_root = getattr(self.config, "data_root", None)
        if not (exp_id and data_root):
            return
        prepared = Path(data_root) / exp_id / "prepared"
        links = [
            (prepared / "baseline.json", workspace / "baseline.json"),
            (prepared / "grade.py", workspace / "grade.py"),
        ]
        for src, dst in links:
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                    dst.symlink_to(src)
                except FileExistsError:
                    pass

    def _resolve_worker_workspace(self, worker_index: int, main_workspace: Path) -> Path:
        session_config = self.config.session.get("local", {})
        parallel_config = session_config.get("parallel", {})
        split_workspace = parallel_config.get("split_workspace_for_exp", False)
        if split_workspace:
            return main_workspace / f"exp_{worker_index}"
        return main_workspace

    def _setup_workspace_directories(self, workspace: Path) -> Path:
        (workspace / "working").mkdir(parents=True, exist_ok=True)
        (workspace / "best_solution").mkdir(parents=True, exist_ok=True)
        (workspace / "best_submission").mkdir(parents=True, exist_ok=True)
        (workspace / "data_links").mkdir(parents=True, exist_ok=True)
        (workspace / "manifests").mkdir(parents=True, exist_ok=True)
        submission_dir = workspace / "submission"
        submission_dir.mkdir(parents=True, exist_ok=True)
        return submission_dir

    def _ensure_shared_data_links(self, worker_workspace: Path, task_workspace: Path) -> None:
        """Make every worker reuse the task-level data_links directory."""
        if worker_workspace.resolve() == task_workspace.resolve():
            return

        shared_data_links = task_workspace / "data_links"
        shared_data_links.mkdir(parents=True, exist_ok=True)
        worker_data_links = worker_workspace / "data_links"

        if worker_data_links.is_symlink():
            try:
                if worker_data_links.resolve() == shared_data_links.resolve():
                    return
            except FileNotFoundError:
                pass
            worker_data_links.unlink()
        elif worker_data_links.exists():
            if worker_data_links.is_dir():
                for child in list(worker_data_links.iterdir()):
                    target = shared_data_links / child.name
                    if not target.exists():
                        child.rename(target)
                if worker_data_links.exists() and not any(worker_data_links.iterdir()):
                    worker_data_links.rmdir()
            elif worker_data_links.exists():
                worker_data_links.unlink()

        if not worker_data_links.exists():
            worker_data_links.symlink_to(shared_data_links, target_is_directory=True)

    def _build_demand_spec(
        self,
        *,
        task_description: str,
        node: Any,
        memory: str,
    ) -> dict[str, Any]:
        """Build a compact search spec for the next red scout."""
        text = "\n".join(
            [
                task_description,
                getattr(node, "code", "") or "",
                getattr(node, "stdout", "") or "",
                memory or "",
            ]
        ).lower()

        if any(key in text for key in ("comment", "text", "nlp", "insult", "toxic", "language")):
            task_type = "text_classification"
            wanted_data = [
                "comment-level or sentence-level labeled text data",
                "toxicity / insult / abuse / hate-speech style binary or easily mappable labels",
                "csv, json, parquet, or HuggingFace datasets that can be loaded in Python",
            ]
            search_queries = [
                "insult detection dataset comments",
                "toxic comment classification dataset",
                "abusive language short text dataset",
            ]
        elif any(key in text for key in ("image", "jpeg", "png", "pixel", "vision")):
            task_type = "image_classification"
            wanted_data = [
                "labeled image datasets close to the target categories",
                "standard folder/parquet/image-byte formats that can be inspected quickly",
                "datasets with clear class mapping to the competition labels",
            ]
            search_queries = [
                "related image classification dataset",
                "public image dataset with similar labels",
                "huggingface image dataset classification",
            ]
        else:
            task_type = "tabular_or_generic"
            wanted_data = [
                "structured datasets with labels close to the competition target",
                "easy-to-load csv/parquet/json sources",
                "datasets whose fields can be merged or reused inside the DataLoader",
            ]
            search_queries = [
                "related public dataset with labels",
                "competition domain auxiliary dataset",
                "huggingface dataset similar task",
            ]

        bottlenecks = []
        metric_value = getattr(getattr(node, "metric", None), "value", None)
        if metric_value is not None:
            bottlenecks.append(f"Current branch metric is {metric_value:.4f}.")
        if not getattr(node, "bound_manifest_path", None):
            bottlenecks.append("This branch does not yet have its own external-data manifest.")
        if getattr(node, "stage", "") in {"initial", "black"}:
            bottlenecks.append("Need external data that improves the current branch rather than generic exploration.")

        return {
            "parent_node_id": getattr(node, "id", None),
            "parent_stage": getattr(node, "stage", None),
            "task_type": task_type,
            "goal": "find a small number of external datasets that are directly useful for the current branch",
            "current_bottlenecks": bottlenecks,
            "wanted_external_data": wanted_data,
            "hard_constraints": [
                "prefer 2-3 high-quality datasets over broad search",
                "download into task_workspace/data_links",
                "only keep datasets that can be inspected and loaded concretely",
            ],
            "search_queries": search_queries,
        }

    # ------------------------------------------------------------------ #
    # UCT search manager
    # ------------------------------------------------------------------ #

    def _create_search_manager(
        self, submission_dir: Path, task_description: str
    ) -> UCTSearchManager:
        servers = getattr(self.config, "grading_servers", []) or []
        search_mgr = UCTSearchManager(
            search_cfg=UCTSearchConfig(),
            decay_cfg=UCTDecayConfig(),
            grader=lambda exp_id, p: validate_submission(
                exp_id,
                p,
                server_urls=servers,
                dataset_root=getattr(self.config, "data_root", None),
            ),
            exp_id=getattr(self.config, "exp_id", "unknown"),
            submission_dir=submission_dir,
        )
        search_mgr.set_snapshot_fn(
            lambda node, sub, review, reward: save_node_snapshot(
                self.run_dir,
                Path(self.session.config.workspace_path),
                node,
                sub,
                review,
                reward,
                search_mgr,
                task_description=task_description,
            )
        )
        return search_mgr

    # ------------------------------------------------------------------ #
    # Node execution: KEY CHANGES for v2
    # ------------------------------------------------------------------ #

    def _run_one_node(
        self,
        *,
        worker_agents: dict[str, Agent],
        worker_workspace: Path,
        task_workspace: Path,          # NEW: task-level shared workspace
        data_preview: str,
        task_description: str,
        stage: str,
        node: Any,
        prev_code: str,
        term_out: str,
        issue: str,
        best_code: str | None,
        best_metric: float | None,
        memory: str,
        exp_index: int,
    ) -> dict[str, Any]:
        if stage == "initial":
            exp = InitialExp(
                worker_agents["initial"],
                worker_agents["metric"],
                self.session,
                worker_workspace,
                getattr(self.config, "exp_id", None),
                data_preview,
                node,
                exp_index=exp_index,
            )
            return exp.run(task_description)

        if stage == "black":
            exp = BlackExp(
                worker_agents["black"],
                worker_agents["metric"],
                self.session,
                worker_workspace,
                task_workspace,          # pass task_workspace so Black can read manifest
                Path(getattr(node, "bound_manifest_path")) if getattr(node, "bound_manifest_path", None) else None,
                getattr(self.config, "exp_id", None),
                data_preview,
                node,
                exp_index=exp_index,
            )
            return exp.run(
                task_description,
                prev_code=prev_code,
                memory=memory,
                term_out=term_out,
                best_code=best_code,
                best_metric=best_metric,
            )

        if stage == "red":
            exp = RedExp(
                worker_agents["red"],
                worker_agents["metric"],
                self.session,
                worker_workspace,
                task_workspace,          # pass task_workspace so Red can write manifest
                Path(getattr(node, "output_manifest_path")) if getattr(node, "output_manifest_path", None) else None,
                Path(getattr(node, "input_manifest_path")) if getattr(node, "input_manifest_path", None) else None,
                getattr(node, "demand_spec", None),
                getattr(self.config, "exp_id", None),
                data_preview,
                node,
                exp_index=exp_index,
            )
            return exp.run(
                task_description,
                prev_code=prev_code,
                memory=memory,
                term_out=term_out,
                best_code=best_code,
                best_metric=best_metric,
            )

        # terminal
        return {
            "plan": "TERMINATE",
            "code": "TERMINATE",
            "raw_response": "TERMINATE",
            "exec": {"stdout": "TERMINATE", "exit_code": 1},
            "metric": None,
            "metric_detail": {"is_bug": True, "has_submission": False},
        }

    # ------------------------------------------------------------------ #
    # build_review override: Red success ≠ F1, it's manifest presence
    # ------------------------------------------------------------------ #

    def _build_review_for_node(
        self,
        res: dict[str, Any],
        has_submission: bool,
        stage: str,
        task_workspace: Path,
    ) -> MetricReview:
        """v2 override: Red nodes are judged by manifest, not F1."""
        if stage == "red":
            manifest_ok = res.get("metric_detail", {}).get("manifest_ok", False)
            return MetricReview(
                metric=None,
                lower_is_better=None,
                is_bug=not manifest_ok,
                has_submission=False,
                summary=res.get("exec", {}).get("stdout", "")[-500:],
                raw_output=res.get("raw_response"),
            )
        return build_review(res, has_submission)

    # ------------------------------------------------------------------ #
    # UCT node expansion
    # ------------------------------------------------------------------ #

    def _select_stages_batch(
        self, target: Any, search_cfg: UCTSearchConfig
    ) -> list[tuple[str, str, str, str]]:
        if target.stage == "root":
            return [("initial", "", "", "")]
        if target.is_buggy or target.metric.value is None:
            # Red nodes can still expand Black even with metric=None,
            # as long as they're not buggy
            if target.stage == "red" and not target.is_buggy:
                pass  # fall through to normal expansion
            else:
                return [("terminal", getattr(target, "code", ""), getattr(target, "stdout", ""), "")]

        num_red = sum(1 for c in target.children if c.stage == "red")
        num_black = sum(1 for c in target.children if c.stage == "black")
        stages: list[tuple[str, str, str, str]] = []
        if target.stage in {"initial", "black"}:
            remaining_red = search_cfg.num_red - num_red
            for _ in range(max(remaining_red, 0)):
                stages.append(("red", getattr(target, "code", ""), getattr(target, "stdout", ""), ""))
        elif target.stage == "red":
            remaining_black = search_cfg.num_black - num_black
            for _ in range(max(remaining_black, 0)):
                stages.append(("black", getattr(target, "code", ""), getattr(target, "stdout", ""), ""))

        if not stages:
            return [("terminal", getattr(target, "code", ""), getattr(target, "stdout", ""), "")]
        return stages

    # ------------------------------------------------------------------ #
    # Main run loop
    # ------------------------------------------------------------------ #

    def _initialize_search_state(self) -> dict[str, Optional[Any]]:
        return {
            "code": None,
            "metric": None,
            "node_id": None,
            "dispatch_id": 0,
            "active_jobs": 0,
            "initial_completed": False,
        }

    def _create_expand_node_fn(
        self, search_mgr: UCTSearchManager, task_description: str
    ) -> Callable:
        task_workspace = Path(self.session.config.workspace_path)

        def expand_node(node: Any, stage: str, prev_code: str, term_out: str, issue: str) -> Any:
            child = search_mgr.create_child(node, stage=stage, plan="", code="")
            if stage == "red":
                child.input_manifest_path = getattr(node, "bound_manifest_path", None)
                child.output_manifest_path = str(get_manifest_path(task_workspace, child.id))
                child.demand_spec = self._build_demand_spec(
                    task_description=task_description,
                    node=node,
                    memory=node.fetch_child_memory() if hasattr(node, "fetch_child_memory") else "",
                )
            elif stage == "black":
                if node.stage == "red":
                    child.bound_manifest_path = getattr(node, "output_manifest_path", None)
                    child.demand_spec = getattr(node, "demand_spec", None)
                else:
                    child.bound_manifest_path = getattr(node, "bound_manifest_path", None)
            search_mgr.push_execution_node(child)
            save_node_snapshot(
                self.run_dir,
                Path(self.session.config.workspace_path),
                child,
                None,
                MetricReview(metric=None, is_bug=False, has_submission=False, summary="created"),
                0.0,
                search_mgr,
                task_description=task_description,
                snapshot_event="created",
            )
            return child
        return expand_node

    def _select_node_to_execute(
        self,
        worker_index: int,
        search_mgr: UCTSearchManager,
        best_state: dict,
        state_lock: threading.Lock,
        max_steps: int,
    ) -> tuple[bool, Optional[Any]]:
        with state_lock:
            if search_mgr.current_step >= max_steps:
                return (False, None)
            if not best_state["initial_completed"]:
                return self._handle_phase1(worker_index, search_mgr, best_state)
            else:
                return self._handle_phase2(search_mgr, best_state)

    def _handle_phase1(
        self, worker_index: int, search_mgr: UCTSearchManager, best_state: dict
    ) -> tuple[bool, Optional[Any]]:
        if worker_index != 0:
            return (True, None)
        target = search_mgr.select_next()
        if target is None:
            return (False, None)
        if target.stage == "root":
            node = search_mgr.create_child(target, stage="initial", plan="", code="")
            save_node_snapshot(
                self.run_dir,
                Path(self.session.config.workspace_path),
                node, None,
                MetricReview(metric=None, is_bug=False, has_submission=False, summary="created"),
                0.0, search_mgr, snapshot_event="created",
            )
            best_state["active_jobs"] = int(best_state["active_jobs"] or 0) + 1
            return (False, node)
        elif target.is_buggy is None:
            return (True, None)
        elif target.is_buggy:
            self.logger.error("Initial node %s failed, aborting", target.id)
            return (False, None)
        else:
            return self._batch_expand_after_initial(target, search_mgr, best_state)

    def _batch_expand_after_initial(
        self, target: Any, search_mgr: UCTSearchManager, best_state: dict
    ) -> tuple[bool, Optional[Any]]:
        best_state["initial_completed"] = True
        stages_batch = self._select_stages_batch(target, search_mgr.search_cfg)
        for stage, prev_code, term_out, issue in stages_batch:
            if stage == "terminal":
                continue
            child = search_mgr.create_child(target, stage=stage, plan="", code="")
            if stage == "red":
                child.input_manifest_path = getattr(target, "bound_manifest_path", None)
                child.output_manifest_path = str(get_manifest_path(Path(self.session.config.workspace_path), child.id))
                child.demand_spec = self._build_demand_spec(
                    task_description=getattr(self, "task_description", "") or "",
                    node=target,
                    memory=target.fetch_child_memory() if hasattr(target, "fetch_child_memory") else "",
                )
            search_mgr.push_execution_node(child)
            save_node_snapshot(
                self.run_dir,
                Path(self.session.config.workspace_path),
                child, None,
                MetricReview(metric=None, is_bug=False, has_submission=False, summary="created"),
                0.0, search_mgr, snapshot_event="created",
            )
        created = sum(1 for stage, *_ in stages_batch if stage != "terminal")
        best_state["active_jobs"] = int(best_state["active_jobs"] or 0) + created
        self.logger.info("Batch created %d nodes after initial", created)
        return (True, None)

    def _handle_phase2(
        self, search_mgr: UCTSearchManager, best_state: dict
    ) -> tuple[bool, Optional[Any]]:
        node = search_mgr.pop_execution_node()
        if node is None:
            if int(best_state["active_jobs"] or 0) == 0:
                return (False, None)
            return (True, None)
        best_state["active_jobs"] = int(best_state["active_jobs"] or 0) + 1
        return (False, node)

    def _execute_and_process_node(
        self,
        node: Any,
        worker_agents: dict,
        worker_workspace: Path,
        task_workspace: Path,
        worker_submission_dir: Path,
        data_preview: str,
        task_description: str,
        best_state: dict,
        state_lock: threading.Lock,
        submission_dir: Path,
        search_mgr: UCTSearchManager,
        search_cfg: UCTSearchConfig,
        results: dict,
        workspace: Path,
        expand_node: Callable,
    ) -> int:
        stage = node.stage
        prev_code = getattr(node.parent, "code", "") if node.parent else ""
        term_out = getattr(node.parent, "stdout", "") if node.parent else ""
        dispatch_id = int(best_state["dispatch_id"] or 0)
        with state_lock:
            best_state["dispatch_id"] = dispatch_id + 1

        parent_node = node.parent
        memory = parent_node.fetch_child_memory() if parent_node else ""
        best_code = best_state["code"]
        best_metric = best_state["metric"]

        try:
            res = self._run_one_node(
                worker_agents=worker_agents,
                worker_workspace=worker_workspace,
                task_workspace=task_workspace,
                data_preview=data_preview,
                task_description=task_description,
                stage=stage,
                node=node,
                prev_code=prev_code,
                term_out=term_out,
                issue="",
                best_code=best_code,
                best_metric=best_metric,
                memory=memory,
                exp_index=dispatch_id,
            )
        except Exception as exc:
            self.logger.error("Worker failed on node %s: %s", node.id, exc, exc_info=True)
            res = {
                "plan": "",
                "code": "",
                "raw_response": str(exc),
                "exec": {"stdout": str(exc), "exit_code": -1},
                "metric": None,
                "metric_detail": {"is_bug": True, "has_submission": False},
            }

        with state_lock:
            best_state["active_jobs"] = max(int(best_state["active_jobs"] or 0) - 1, 0)
            node.code = res.get("code", "")
            node.plan = res.get("plan", "")
            node.stdout = res.get("exec", {}).get("stdout", "")
            node.exit_code = res.get("exec", {}).get("exit_code", None)
            if stage == "red":
                node.output_manifest_path = res.get("metric_detail", {}).get(
                    "manifest_path",
                    getattr(node, "output_manifest_path", None),
                )
            elif stage == "black":
                node.bound_manifest_path = getattr(node, "bound_manifest_path", None)

            copied = copy_submission(
                submission_dir, node.id, source_submission_dir=worker_submission_dir
            )
            # v2: Red uses manifest-based review, not F1
            review = self._build_review_for_node(res, copied is not None, stage, task_workspace)
            reward = search_mgr.ingest_result(node, review)

            save_node_snapshot(
                self.run_dir,
                Path(self.session.config.workspace_path),
                node, copied, review, reward, search_mgr,
                task_description=task_description,
                snapshot_event="completed",
            )
            # Trajectory record
            trail = {
                "ts": datetime.utcnow().isoformat(),
                "step": search_mgr.current_step,
                "stage": stage,
                "node_id": node.id,
                "parent": getattr(node.parent, "id", None),
                "is_buggy": node.is_buggy,
                "metric": getattr(node.metric, "value", None),
                "has_submission": copied is not None,
            }
            append_trajectory(self, trail, logger=self.logger)
            results[stage].append(res)

            # Update best solution (Black/Initial only, not Red)
            if (
                stage != "red"
                and search_mgr.best_node
                and search_mgr.best_node.id != best_state["node_id"]
                and search_mgr.best_node.metric.value is not None
            ):
                best_state["node_id"] = node.id
                best_state["metric"] = node.metric.value
                best_state["code"] = node.code
                best_sub = submission_dir / f"submission_{node.id}.csv"
                save_best(
                    self.logger, workspace, str(node.code or ""),
                    best_sub if best_sub.exists() else copied,
                )

            # Event-driven expansion
            if best_state["initial_completed"]:
                self._expand_after_node_completion(
                    node, search_mgr, search_cfg, best_state, expand_node
                )
        return 1

    def _expand_after_node_completion(
        self, node, search_mgr, search_cfg, best_state, expand_node
    ) -> None:
        if node.is_buggy:
            return
        if node.stage != "red" and node.metric.value is None:
            return

        stages_batch = self._select_stages_batch(node, search_cfg)
        created = 0
        for stage, prev_code, term_out, issue in stages_batch:
            if stage == "terminal":
                continue
            expand_node(node, stage, prev_code, term_out, issue)
            created += 1
        best_state["active_jobs"] = int(best_state["active_jobs"] or 0) + created

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        try:
            self.task_description = task_description
            self.setup()
            self._setup_trajectory_file(output_file)
            workspace = Path(self.session.config.workspace_path)
            submission_dir = self._setup_workspace_directories(workspace)
            search_mgr = self._create_search_manager(submission_dir, task_description)

            results: dict = {
                "status": "completed",
                "initial": [],
                "black": [],
                "red": [],
            }
            best_state = self._initialize_search_state()
            max_steps = 40
            state_lock = threading.Lock()
            worker_agents_map = {
                i: self._create_worker_agents(i) for i in range(self.max_workers)
            }
            expand_node = self._create_expand_node_fn(search_mgr, task_description)

            def worker_loop(worker_index: int) -> dict[str, Any]:
                worker_agents = worker_agents_map[worker_index]
                worker_workspace = self._resolve_worker_workspace(worker_index, workspace)
                worker_workspace.mkdir(parents=True, exist_ok=True)
                self._ensure_prepared_links(worker_workspace)
                self._ensure_shared_data_links(worker_workspace, workspace)
                (worker_workspace / "working").mkdir(parents=True, exist_ok=True)
                (worker_workspace / "submission").mkdir(parents=True, exist_ok=True)
                worker_submission_dir = worker_workspace / "submission"
                data_preview = generate_data_preview(worker_workspace)
                completed = 0

                while True:
                    should_wait, node = self._select_node_to_execute(
                        worker_index, search_mgr, best_state, state_lock, max_steps
                    )
                    if should_wait:
                        time.sleep(0.1)
                        continue
                    if node is None:
                        break
                    completed += self._execute_and_process_node(
                        node=node,
                        worker_agents=worker_agents,
                        worker_workspace=worker_workspace,
                        task_workspace=workspace,       # task-level shared workspace
                        worker_submission_dir=worker_submission_dir,
                        data_preview=data_preview,
                        task_description=task_description,
                        best_state=best_state,
                        state_lock=state_lock,
                        submission_dir=submission_dir,
                        search_mgr=search_mgr,
                        search_cfg=search_mgr.search_cfg,
                        results=results,
                        workspace=workspace,
                        expand_node=expand_node,
                    )
                return {"worker_index": worker_index, "completed": completed}

            worker_tasks = [partial(worker_loop, i) for i in range(self.max_workers)]
            worker_results = self.execute_parallel_tasks(
                worker_tasks, max_workers=self.max_workers
            )
            for idx, wr in enumerate(worker_results):
                if isinstance(wr, Exception):
                    self.logger.error("Worker %s exception: %s", idx, wr)
                else:
                    self.logger.info("Worker summary: %s", wr)
            return results
        finally:
            self.cleanup()
