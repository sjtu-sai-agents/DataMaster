"""数据探索 Agent Playground：调用 Initial/Black/Red 三个 EXP"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from functools import partial
from pathlib import Path
from typing import Optional, Any, Callable
from datetime import datetime
from urllib.parse import urlparse

import requests
from evomaster.core import BasePlayground, register_playground
from evomaster.agent import Agent

from .utils.grading import shutdown_embedded_grading_server, validate_submission
from .utils.uct import UCTSearchConfig, UCTDecayConfig, UCTSearchManager, MetricReview
from .utils.data_preview import generate as generate_data_preview
from .utils.playground_helpers import (
    append_trajectory,
    build_review,
    copy_submission,
    save_best,
    save_node_snapshot,
)
from .exp.initial_exp import InitialExp
from .exp.black_exp import BlackExp
from .exp.rule_black_exp import RuleBlackExp
from .exp.red_exp import RedExp
from playground.search_dataset_tools.memory_tree import (
    create_node_memory,
    save_node_storage,
)

logger = logging.getLogger(__name__)


@register_playground("data_master")
class DataTreePlayground(BasePlayground):
    def __init__(self, config_dir: Path | None = None, config_path: Path | None = None):
        if config_path is None and config_dir is None:
            config_dir = (
                Path(__file__).parent.parent.parent.parent
                / "configs"
                / "data_master"
            )
            logger.info(f"Config dir: {config_dir}")
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents.declare("nouse_agent", "initial_agent", "black_agent", "red_agent", "metric_agent")
        self.trajectories: list[dict[str, Any]] = []
        session_config = self.config.session.get("local", {})
        parallel_config = session_config.get("parallel", {})
        if parallel_config.get("enabled", False):
            self.max_workers = int(parallel_config.get("max_parallel", 1))
        else:
            self.max_workers = 1

        # Initial 配置
        self.initial_code: str | None = None
        self.initial_instruction: str | None = None

        # 独立 grading server 进程
        self.grading_server_proc: subprocess.Popen | None = None
        self.grading_server_url: str | None = None
        self.grading_server_lock = threading.Lock()

        # Test feedback 配置
        self.test_feedback: bool = False
        self.force_direction: str | None = None
        self.test_metric_agent = None

        # Component ablation 配置
        ablation_cfg = getattr(self.config, "ablation", {}) or {}
        self.use_red_node = bool(ablation_cfg.get("use_red_node", True))
        self.use_memory = bool(ablation_cfg.get("use_memory", True))
        self.black_mode = ablation_cfg.get("black_mode", "agent")

        self.logger.info(
            f"Component ablation config: "
            f"use_red_node={self.use_red_node}, "
            f"use_memory={self.use_memory}, "
            f"black_mode={self.black_mode}"
        )

    def set_initial_config(self, initial_code: str | None = None, initial_instruction: str | None = None) -> None:
        """设置 Initial 节点的配置

        Args:
            initial_code: 初始代码内容（可选）
            initial_instruction: 初始自然语言指令（可选）
        """
        self.initial_code = initial_code
        self.initial_instruction = initial_instruction
        self.logger.info(f"Set initial config: code={bool(initial_code)}, instruction={bool(initial_instruction)}")

    def set_test_feedback_config(self, test_feedback: bool, force_direction: str | None = None) -> None:
        """设置 test_feedback 配置

        注意：test_metric_agent 会在 setup() 中创建，因为此时 session 才初始化

        Args:
            test_feedback: 是否启用测试集反馈模式
            force_direction: 方向强制（"minimize" 或 "maximize"）
        """
        self.test_feedback = test_feedback
        self.force_direction = force_direction
        self.logger.info(f"Test feedback config set: test_feedback={test_feedback}, force_direction={force_direction}")

    def _create_test_metric_agent(self) -> None:
        """创建 test metric agent（在 session 初始化后调用）"""
        if not self.test_feedback:
            return

        from evomaster.agent import Agent, AgentConfig
        from evomaster.utils import create_llm, LLMConfig

        metric_llm_cfg = self._setup_agent_llm("metric")
        output_config = self._get_output_config()
        test_llm = create_llm(LLMConfig(**metric_llm_cfg), output_config=output_config)

        self.test_metric_agent = Agent(
            llm=test_llm,
            session=self.session,
            tools=self.tools,
            system_prompt_file="${PROJECT_ROOT}/playground/data_master/prompts/metric_test/system_prompt.md",
            user_prompt_file="${PROJECT_ROOT}/playground/data_master/prompts/metric_test/user_prompt.md",
            config=AgentConfig(max_turns=1),
            enable_tools=False,
        )
        self.test_metric_agent.set_agent_name("test_metric")
        self.logger.info(
            f"Test metric agent created in setup(): "
            f"test_feedback={self.test_feedback}, force_direction={self.force_direction}"
        )

    def _get_configured_grading_server_url(self) -> str:
        """从当前 config 中读取唯一固定的 grading server URL。"""
        servers = getattr(self.config, "grading_servers", []) or []
        if not servers:
            raise ValueError("config.grading_servers is empty")
        if len(servers) != 1:
            raise ValueError(
                f"Expected exactly one grading server in config, got {len(servers)}"
            )
        return servers[0]

    def _parse_grading_server_host_port(self, server_url: str) -> tuple[str, int]:
        parsed = urlparse(server_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port
        if port is None:
            raise ValueError(f"Invalid grading server url (missing port): {server_url}")
        return host, port

    def _check_grading_server_health(self, server_url: str, timeout: int = 3) -> bool:
        """检查 grading server 是否健康。

        对 localhost/127.0.0.1 请求禁用环境代理，避免被 http_proxy/https_proxy 污染。
        """
        try:
            parsed = urlparse(server_url)
            host = parsed.hostname or ""
            health_url = f"{server_url.rstrip('/')}/health"
            if host in {"127.0.0.1", "localhost"}:
                with requests.Session() as session:
                    session.trust_env = False
                    resp = session.get(health_url, timeout=timeout)
            else:
                resp = requests.get(health_url, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def _start_grading_server_process(self) -> None:
        """按 config 里的固定 host:port 启动 grading server。"""
        server_url = self._get_configured_grading_server_url()
        host, port = self._parse_grading_server_host_port(server_url)

        data_root = getattr(self.config, "data_root", None)
        if not data_root:
            raise ValueError("config.data_root is required to start grading server")

        project_root = Path(__file__).resolve().parents[3]
        runner_path = (
            project_root
            / "search_dataset_tools"
            / "operate_submission"
            / "grading_server_runner.py"
        )

        cmd = [
            sys.executable,
            str(runner_path),
            "--data-root",
            str(data_root),
            "--host",
            host,
            "--port",
            str(port),
        ]

        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )

        deadline = time.time() + 20
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"grading server exited early with code {proc.returncode} "
                    f"on {server_url}"
                )
            if self._check_grading_server_health(server_url, timeout=1):
                self.grading_server_proc = proc
                self.grading_server_url = server_url
                os.environ["ML_MASTER_GRADING_SERVERS"] = server_url
                self.logger.info(f"Started standalone grading server: {server_url}")
                return
            time.sleep(0.5)

        proc.terminate()
        raise RuntimeError(f"grading server failed to become healthy: {server_url}")

    def _stop_grading_server_process(self) -> None:
        """停止当前 task 管理的 grading server 进程。"""
        if self.grading_server_proc is None:
            return

        try:
            self.grading_server_proc.terminate()
            self.grading_server_proc.wait(timeout=5)
        except Exception:
            try:
                self.grading_server_proc.kill()
            except Exception:
                pass
        finally:
            self.logger.info(
                f"Stopped standalone grading server: {self.grading_server_url}"
            )
            self.grading_server_proc = None
            self.grading_server_url = None

    def _restart_grading_server_same_port(self) -> None:
        """在 config 指定的同一个端口上重启 grading server。"""
        server_url = self._get_configured_grading_server_url()
        self.logger.warning(
            f"Grading server unhealthy, restarting on same port: {server_url}"
        )
        self._stop_grading_server_process()
        self._start_grading_server_process()

    def _ensure_grading_server_ready(self) -> None:
        """确保当前 task 使用的 grading server 处于健康状态。"""
        with self.grading_server_lock:
            server_url = self._get_configured_grading_server_url()
            os.environ["ML_MASTER_GRADING_SERVERS"] = server_url

            if self._check_grading_server_health(server_url):
                self.grading_server_url = server_url
                self.logger.info(f"Using healthy grading server from config: {server_url}")
                return

            self._restart_grading_server_same_port()

    # --------------------------- 初始化 --------------------------- #
    def setup(self) -> None:
        self.logger.info("Setting up DataTreePlayground...")
        self._setup_session()
        self._setup_agents()

        required_slots = ["initial_agent", "black_agent", "red_agent", "metric_agent"]
        missing = [slot for slot in required_slots if self.agents.get(slot) is None]
        if missing:
            raise ValueError(f"config.agents 缺少必需配置: {missing}")

        exp_id = getattr(self.config, "exp_id", None)
        data_root = getattr(self.config, "data_root", None)
        if exp_id and data_root:
            os.environ["ML_MASTER_DATA_EXPID"] = exp_id
            os.environ["ML_MASTER_DATA_ROOT"] = str(data_root)
            self.logger.info(f"Set env vars: ML_MASTER_DATA_EXPID={exp_id}, ML_MASTER_DATA_ROOT={data_root}")

        if self.test_feedback and self.test_metric_agent is None:
            self._create_test_metric_agent()

        self._ensure_prepared_links(Path(self.session.config.workspace_path))

    def cleanup(self) -> None:
        try:
            self._stop_grading_server_process()
        except Exception as exc:
            self.logger.warning("Failed to stop standalone grading server: %s", exc)

        try:
            shutdown_embedded_grading_server(timeout=5)
        except Exception as exc:
            self.logger.warning("Failed to shutdown embedded grading server: %s", exc)

        super().cleanup()

    def _ensure_prepared_links(self, workspace: Path) -> None:
        """创建软链接，使用重试机制处理并发竞态条件，带详细日志"""
        import time

        exp_id = getattr(self.config, "exp_id", None)
        data_root = getattr(self.config, "data_root", None)
        if not (exp_id and data_root):
            return

        prepared = Path(data_root) / exp_id / Path("prepared")
        self.logger.info(f"_ensure_prepared_links: workspace={workspace}, prepared={prepared}")

        def _safe_symlink(src: Path, dst: Path, is_dir: bool = False, max_retries: int = 3) -> bool:
            """安全创建符号链接，带重试机制和详细日志"""
            self.logger.debug(f"_safe_symlink: src={src}, dst={dst}, is_dir={is_dir}")

            for attempt in range(max_retries):
                try:
                    if not src.exists():
                        self.logger.warning(f"[{dst.name}] Source not found (attempt {attempt + 1}/{max_retries}): {src}")
                        if attempt < max_retries - 1:
                            time.sleep(0.005 * (attempt + 1))
                            continue
                        return False

                    self.logger.debug(f"[{dst.name}] Source exists, attempt={attempt + 1}/{max_retries}")

                    if dst.is_symlink():
                        try:
                            resolved = dst.resolve()
                            if resolved == src.resolve():
                                self.logger.debug(f"[{dst.name}] Symlink already exists and is valid")
                                return True
                            else:
                                self.logger.debug(f"[{dst.name}] Symlink points to wrong target: {resolved} != {src}")
                        except (FileNotFoundError, OSError) as e:
                            self.logger.warning(f"[{dst.name}] Broken symlink detected: {e}")

                    dst.parent.mkdir(parents=True, exist_ok=True)

                    if dst.exists() or dst.is_symlink():
                        self.logger.info(f"[{dst.name}] Removing existing: exists={dst.exists()}, is_symlink={dst.is_symlink()}")
                        if is_dir and dst.is_dir() and not dst.is_symlink():
                            shutil.rmtree(dst)
                        else:
                            dst.unlink()

                    dst.symlink_to(src)
                    self.logger.info(f"[{dst.name}] Symlink created successfully: {dst} -> {src}")
                    return True

                except FileExistsError as e:
                    self.logger.info(f"[{dst.name}] FileExistsError (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(0.005 * (attempt + 1))
                        continue
                    if dst.is_symlink() or dst.exists():
                        self.logger.info(f"[{dst.name}] File exists after retries (likely created by another worker)")
                        return True

                except FileNotFoundError as e:
                    self.logger.warning(f"[{dst.name}] FileNotFoundError (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(0.005 * (attempt + 1))
                        continue
                    return False

                except OSError as e:
                    self.logger.warning(f"[{dst.name}] OSError (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(0.005 * (attempt + 1))
                        continue
                    return False

            self.logger.error(f"[{dst.name}] Failed to create symlink after {max_retries} attempts")
            return False

        src_base = prepared / "baseline.json"
        dst_base = workspace / "baseline.json"
        if not _safe_symlink(src_base, dst_base):
            self.logger.warning("Failed to create baseline.json symlink, continuing anyway")

        src_grade = prepared / "grade.py"
        dst_grade = workspace / "grade.py"
        if not _safe_symlink(src_grade, dst_grade):
            self.logger.warning("Failed to create grade.py symlink, continuing anyway")

        src_public = prepared / "public"
        dst_public = workspace / "input" / "public"
        if src_public.exists() and src_public.is_dir():
            if not _safe_symlink(src_public, dst_public, is_dir=True):
                self.logger.warning("Failed to create public symlink, continuing anyway")
        else:
            self.logger.debug(f"public directory not found: {src_public}")

    def _resolve_worker_workspace(
        self, worker_index: int, main_workspace: Path
    ) -> Path:
        session_config = self.config.session.get("local", {})
        parallel_config = session_config.get("parallel", {})
        split_workspace = parallel_config.get("split_workspace_for_exp", False)
        if split_workspace:
            return main_workspace / f"exp_{worker_index}"
        return main_workspace

    def _create_worker_agents(self, worker_index: int) -> dict[str, Agent]:
        return {
            "initial": self.copy_agent(
                self.agents.initial_agent,
                new_agent_name=f"initial_worker_{worker_index}",
            ),
            "black": self.copy_agent(
                self.agents.black_agent, new_agent_name=f"black_worker_{worker_index}"
            ),
            "red": self.copy_agent(
                self.agents.red_agent, new_agent_name=f"red_worker_{worker_index}"
            ),
            "metric": self.copy_agent(
                self.agents.metric_agent, new_agent_name=f"metric_worker_{worker_index}"
            ),
        }

    def _reset_working_dir(self, workspace: Path) -> None:
        """Clear and recreate workspace/working after each node execution."""
        working_dir = workspace / "working"
        try:
            if working_dir.exists():
                shutil.rmtree(working_dir)
            working_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.logger.warning("Failed to reset working dir %s: %s", working_dir, exc)

    def _select_stages_batch(
        self, target: Any, search_cfg: UCTSearchConfig
    ) -> list[tuple[str, str, str, str]]:
        """批量选择下一个节点的 stages 和输入参数"""
        if target.stage == "root":
            return [("initial", "", "", "")]

        if target.is_buggy:
            max_buggy_children = 3
            if target.num_children >= max_buggy_children:
                return [
                    (
                        "terminal",
                        getattr(target, "code", ""),
                        getattr(target, "stdout", ""),
                        "",
                    )
                ]

        num_red_children = sum(1 for c in target.children if c.stage == "red")
        num_black_children = sum(1 for c in target.children if c.stage == "black")

        stages_to_create = []

        if self.use_red_node and num_red_children < search_cfg.num_red:
            stages_to_create.append(
                ("red", getattr(target, "code", ""), getattr(target, "stdout", ""), "")
            )
        elif not self.use_red_node and num_red_children < search_cfg.num_red:
            self.logger.info(
                f"No-Red ablation enabled: skip red child creation for node {target.id[:8]}"
            )

        remaining_black = search_cfg.num_black - num_black_children
        for _ in range(remaining_black):
            stages_to_create.append(
                (
                    "black",
                    getattr(target, "code", ""),
                    getattr(target, "stdout", ""),
                    "",
                )
            )

        if not stages_to_create:
            return [
                (
                    "terminal",
                    getattr(target, "code", ""),
                    getattr(target, "stdout", ""),
                    "",
                )
            ]

        return stages_to_create

    def _run_one_node(
        self,
        *,
        worker_agents: dict[str, Agent],
        worker_workspace: Path,
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
        """运行单个节点实验

        Args:
            stage: 节点类型 (initial/black/red)
            node: UCT 节点
            memory: 子节点记忆
        """
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
                initial_code=self.initial_code,
                initial_instruction=self.initial_instruction,
                test_feedback=self.test_feedback,
                force_direction=self.force_direction,
            )
            if self.test_metric_agent:
                exp.test_metric_agent = self.test_metric_agent
            return exp.run(task_description)

        if stage == "black":
            black_mode = getattr(self, "black_mode", "agent")
            exp_cls = RuleBlackExp if black_mode == "rule_based" else BlackExp

            self.logger.info(
                f"Black stage selected: black_mode={black_mode}, exp_cls={exp_cls.__name__}, "
                f"node_id={getattr(node, 'id', '')[:8]}, "
                f"parent_id={getattr(getattr(node, 'parent', None), 'id', '')[:8] if getattr(node, 'parent', None) else None}"
            )

            exp = exp_cls(
                worker_agents["black"],
                worker_agents["metric"],
                self.session,
                worker_workspace,
                getattr(self.config, "exp_id", None),
                data_preview,
                node,
                exp_index=exp_index,
                test_feedback=self.test_feedback,
                force_direction=self.force_direction,
            )
            if self.test_metric_agent:
                exp.test_metric_agent = self.test_metric_agent
            return exp.run(
                task_description,
                prev_code=prev_code,
                memory=memory,
                term_out=term_out,
                best_code=best_code,
                best_metric=best_metric,
            )

        if stage == "red":
            if not self.use_red_node:
                raise RuntimeError(
                    "Red node execution requested while use_red_node=False"
                )

            exp = RedExp(
                worker_agents["red"],
                worker_agents["metric"],
                self.session,
                worker_workspace,
                getattr(self.config, "exp_id", None),
                data_preview,
                node,
                exp_index=exp_index,
                test_feedback=self.test_feedback,
                force_direction=self.force_direction,
            )
            if self.test_metric_agent:
                exp.test_metric_agent = self.test_metric_agent
            return exp.run(
                task_description,
                prev_code=prev_code,
                memory=memory,
                term_out=term_out,
                best_code=best_code,
                best_metric=best_metric,
            )

        return {
            "plan": "TERMINATE, no plan",
            "code": "TERMINATE, no code",
            "raw_response": "TERMINATE",
            "exec": {"stdout": "TERMINATE", "exit_code": 1},
            "metric": None,
            "metric_detail": {"is_bug": True, "has_submission": False},
        }

    def _setup_workspace_directories(self, workspace: Path) -> Path:
        """创建工作空间所需的目录结构"""
        (workspace / "working").mkdir(parents=True, exist_ok=True)
        (workspace / "best_solution").mkdir(parents=True, exist_ok=True)
        (workspace / "best_submission").mkdir(parents=True, exist_ok=True)
        (workspace / "data_links").mkdir(parents=True, exist_ok=True)
        submission_dir = workspace / "submission"
        submission_dir.mkdir(parents=True, exist_ok=True)
        return submission_dir

    def _create_search_manager(
        self, submission_dir: Path, task_description: str
    ) -> UCTSearchManager:
        """创建 UCT 搜索管理器

        只使用当前 config 中唯一固定的 grading server url。
        主循环和外部监控都看这个唯一端口。
        """
        if self.grading_server_url:
            servers = [self.grading_server_url]
        else:
            servers = getattr(self.config, "grading_servers", []) or []

        if not servers:
            raise ValueError("config.grading_servers is empty")

        if len(servers) != 1:
            raise ValueError(
                f"Expected exactly one grading server in config, got {len(servers)}"
            )

        fixed_server_url = servers[0]
        search_cfg = UCTSearchConfig()

        search_mgr = UCTSearchManager(
            search_cfg=search_cfg,
            decay_cfg=UCTDecayConfig(),
            grader=lambda exp_id, p: validate_submission(
                exp_id,
                p,
                server_urls=[fixed_server_url],
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

        self.logger.info(
            "UCT search manager using fixed grading server: %s", fixed_server_url
        )
        print("saving node snapshot to", search_mgr.submission_dir)
        return search_mgr

    def _initialize_search_state(self) -> dict[str, Optional[Any]]:
        """初始化搜索状态"""
        return {
            "code": None,
            "metric": None,
            "node_id": None,
            "dispatch_id": 0,
            "active_jobs": 0,
            "initial_completed": False,
        }

    def _create_expand_node_fn(
        self,
        search_mgr: UCTSearchManager,
        task_description: str,
    ) -> Callable:
        """创建节点扩展函数"""

        def expand_node(node: Any, stage: str, prev_code: str, term_out: str, issue: str) -> Any:
            """扩展单个节点并加入执行堆"""
            child_node = search_mgr.create_child(node, stage=stage, plan="", code="")

            workspace = Path(self.session.config.workspace_path)
            if self.use_memory:
                create_node_memory(workspace, child_node.id, parent_id=node.id)
                self.logger.info(f"Created memory tree for node {child_node.id[:8]}")
            else:
                self.logger.info(
                    f"No-Memory ablation enabled: skip memory tree creation for node {child_node.id[:8]}"
                )

            search_mgr.push_execution_node(child_node)
            save_node_snapshot(
                self.run_dir,
                Path(self.session.config.workspace_path),
                child_node,
                None,
                MetricReview(
                    metric=None,
                    is_bug=False,
                    has_submission=False,
                    summary="node created",
                ),
                0.0,
                search_mgr,
                task_description=task_description,
                snapshot_event="created",
            )
            return child_node

        return expand_node

    def _select_node_to_execute(
        self,
        worker_index: int,
        search_mgr: UCTSearchManager,
        best_state: dict,
        state_lock: threading.Lock,
        max_steps: int,
    ) -> tuple[bool, Optional[Any]]:
        """选择要执行的节点

        Returns:
            (should_wait, node_to_execute)
        """
        should_wait = False
        node_to_execute = None

        with state_lock:
            if search_mgr.current_step >= max_steps:
                return (False, None)

            if not best_state["initial_completed"]:
                should_wait, node_to_execute = self._handle_phase1(
                    worker_index, search_mgr, best_state
                )
            else:
                should_wait, node_to_execute = self._handle_phase2(
                    search_mgr, best_state
                )

        return (should_wait, node_to_execute)

    def _handle_phase1(
        self,
        worker_index: int,
        search_mgr: UCTSearchManager,
        best_state: dict,
    ) -> tuple[bool, Optional[Any]]:
        """处理 Phase 1：串行 initial 阶段"""
        if worker_index != 0:
            return (True, None)

        target = search_mgr.select_next()
        if target is None:
            return (False, None)
        elif target.stage == "root":
            node = search_mgr.create_child(target, stage="initial", plan="", code="")

            workspace = Path(self.session.config.workspace_path)
            if self.use_memory:
                create_node_memory(workspace, node.id, parent_id=None)
                self.logger.info(f"Created memory tree for initial node {node.id[:8]}")
            else:
                self.logger.info(
                    f"No-Memory ablation enabled: skip memory tree creation for initial node {node.id[:8]}"
                )

            self._save_node_created_snapshot(node, search_mgr, best_state)
            best_state["active_jobs"] = int(best_state["active_jobs"] or 0) + 1
            return (False, node)
        elif target.is_buggy is None:
            return (True, None)
        elif target.is_buggy:
            self.logger.warning(
                f"Initial node {target.id} is buggy, but continuing search by expanding repair children"
            )
            return self._batch_expand_after_initial(target, search_mgr, best_state)
        else:
            return self._batch_expand_after_initial(target, search_mgr, best_state)

    def _handle_phase2(
        self,
        search_mgr: UCTSearchManager,
        best_state: dict,
    ) -> tuple[bool, Optional[Any]]:
        """处理 Phase 2：并行执行阶段"""
        node = search_mgr.pop_execution_node()
        if node is None:
            if int(best_state["active_jobs"] or 0) == 0:
                return (False, None)
            return (True, None)

        best_state["active_jobs"] = int(best_state["active_jobs"] or 0) + 1
        return (False, node)

    def _batch_expand_after_initial(
        self,
        target: Any,
        search_mgr: UCTSearchManager,
        best_state: dict,
    ) -> tuple[bool, Optional[Any]]:
        """Initial 完成后批量扩展子节点"""
        best_state["initial_completed"] = True
        stages_batch = self._select_stages_batch(target, search_mgr.search_cfg)

        workspace = Path(self.session.config.workspace_path)

        for stage, prev_code, term_out, issue in stages_batch:
            child_node = search_mgr.create_child(target, stage=stage, plan="", code="")

            if self.use_memory:
                create_node_memory(workspace, child_node.id, parent_id=target.id)
                self.logger.info(f"Created memory tree for node {child_node.id[:8]}")
            else:
                self.logger.info(
                    f"No-Memory ablation enabled: skip memory tree creation for node {child_node.id[:8]}"
                )

            search_mgr.push_execution_node(child_node)

        best_state["active_jobs"] = int(best_state["active_jobs"] or 0) + len(stages_batch)
        self.logger.info(
            f"Initial completed, batch created {len(stages_batch)} nodes"
        )
        return (True, None)

    def _execute_and_process_node(
        self,
        node: Any,
        worker_agents: dict,
        worker_workspace: Path,
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
        """执行节点并处理结果

        Returns:
            完成的任务数（1 或 0）
        """
        self._ensure_grading_server_ready()

        stage = node.stage
        prev_code = getattr(node.parent, "code", "") if node.parent else ""
        term_out = getattr(node.parent, "stdout", "") if node.parent else ""
        issue = ""

        dispatch_id = int(best_state["dispatch_id"] or 0)
        with state_lock:
            best_state["dispatch_id"] = dispatch_id + 1

        parent_node = node.parent
        if self.use_memory and parent_node:
            memory = parent_node.fetch_child_memory()
        else:
            memory = ""
        best_code = best_state["code"]
        best_metric = best_state["metric"]

        try:
            res = self._run_one_node(
                worker_agents=worker_agents,
                worker_workspace=worker_workspace,
                data_preview=data_preview,
                task_description=task_description,
                stage=stage,
                node=node,
                prev_code=prev_code,
                term_out=term_out,
                issue=issue,
                best_code=best_code,
                best_metric=best_metric,
                memory=memory,
                exp_index=dispatch_id,
            )
        except Exception as exc:
            self.logger.error(f"Worker failed on node {node.id}: {exc}", exc_info=True)
            res = self._create_error_result(exc)

        with state_lock:
            best_state["active_jobs"] = max(int(best_state["active_jobs"] or 0) - 1, 0)
            node.code = res.get("code", "")
            node.plan = res.get("plan", "")
            node.stdout = res.get("exec", {}).get("stdout", "")
            node.exit_code = res.get("exec", {}).get("exit_code", None)

            copied = copy_submission(submission_dir, node.id, source_submission_dir=worker_submission_dir)
            review = build_review(res, has_submission=copied is not None)
            reward = search_mgr.ingest_result(node, review)

            self._save_node_completion_snapshot(
                node, copied, review, reward, search_mgr, task_description
            )
            self._append_trajectory(node, stage, copied, search_mgr)
            results[stage].append(res)

            if self.use_memory:
                try:
                    agent_key = stage if stage in ["initial", "black", "red"] else "black"
                    agent = worker_agents.get(agent_key)
                    if agent and agent.trajectory:
                        trajectory_dict = agent.trajectory.model_dump(mode="json")
                    else:
                        trajectory_dict = {}

                    save_node_storage(
                        workspace=workspace,
                        node_id=node.id,
                        trajectory=trajectory_dict,
                        code=res.get("code", ""),
                        stdout=res.get("exec", {}).get("stdout", ""),
                        submission_path=copied,
                    )
                    self.logger.info(f"Saved node storage to memory tree: {node.id[:8]}")
                except Exception as storage_exc:
                    self.logger.warning(f"Failed to save node storage: {storage_exc}")
            else:
                self.logger.info(
                    f"No-Memory ablation enabled: skip memory storage for node {node.id[:8]}"
                )

            if self._should_update_best(node, search_mgr, best_state):
                self._update_best_solution(node, best_state, submission_dir, workspace, copied)

            if best_state["initial_completed"]:
                self._expand_after_node_completion(
                    node, search_mgr, search_cfg, best_state, expand_node
                )

        return 1

    def _create_error_result(self, exc: Exception) -> dict:
        """创建错误结果"""
        return {
            "plan": "",
            "code": "",
            "raw_response": str(exc),
            "exec": {"stdout": str(exc), "exit_code": -1},
            "metric": None,
            "metric_detail": {"is_bug": True, "has_submission": False},
        }

    def _save_node_completion_snapshot(
        self, node: Any, copied, review, reward, search_mgr: UCTSearchManager, task_description: str
    ) -> None:
        """保存节点完成快照"""
        save_node_snapshot(
            self.run_dir,
            Path(self.session.config.workspace_path),
            node,
            copied,
            review,
            reward,
            search_mgr,
            task_description=task_description,
            snapshot_event="completed",
        )

    def _append_trajectory(self, node: Any, stage: str, copied, search_mgr: UCTSearchManager) -> None:
        """追加轨迹记录"""
        trail = {
            "ts": datetime.utcnow().isoformat(),
            "step": search_mgr.current_step,
            "stage": stage,
            "node_id": node.id,
            "parent": getattr(node.parent, "id", None),
            "is_buggy": node.is_buggy,
            "metric": getattr(node.metric, "value", None),
            "has_submission": copied is not None,
            "submission_file": str(copied) if copied else None,
        }
        append_trajectory(self, trail, logger=self.logger)

    def _should_update_best(self, node: Any, search_mgr: UCTSearchManager, best_state: dict) -> bool:
        """判断是否应该更新最佳解"""
        return (
            search_mgr.best_node
            and search_mgr.best_node.id != best_state["node_id"]
            and search_mgr.best_node.metric.value is not None
        )

    def _update_best_solution(
        self, node: Any, best_state: dict, submission_dir: Path, workspace: Path, copied
    ) -> None:
        """更新最佳解"""
        best_state["node_id"] = node.id
        best_state["metric"] = node.metric.value
        best_state["code"] = node.code
        best_sub = submission_dir / f"submission_{node.id}.csv"
        save_best(
            self.logger,
            workspace,
            str(node.code or ""),
            best_sub if best_sub.exists() else copied,
        )

    def _expand_after_node_completion(
        self,
        node: Any,
        search_mgr: UCTSearchManager,
        search_cfg: UCTSearchConfig,
        best_state: dict,
        expand_node: Callable,
    ) -> None:
        """节点完成后的事件驱动扩展"""
        if search_mgr.should_expand_parent(node):
            parent = node.parent
            if parent and not parent.is_buggy:
                self._expand_parent_node(node, search_mgr, search_cfg, best_state, expand_node)

        if search_mgr.should_expand_node(node):
            self._expand_current_node(node, search_mgr, search_cfg, best_state, expand_node)

    def _expand_parent_node(
        self,
        node: Any,
        search_mgr: UCTSearchManager,
        search_cfg: UCTSearchConfig,
        best_state: dict,
        expand_node: Callable,
    ) -> None:
        """扩展父节点（创建兄弟节点）"""
        parent = node.parent
        if not parent or parent.is_buggy:
            return

        import random

        num_red = sum(1 for c in parent.children if c.stage == "red")
        num_black = sum(1 for c in parent.children if c.stage == "black")
        available_stages = []
        if self.use_red_node and num_red < search_cfg.num_red:
            available_stages.append("red")
        elif not self.use_red_node and num_red < search_cfg.num_red:
            self.logger.info(
                f"No-Red ablation enabled: skip red sibling creation for parent {parent.id[:8]}"
            )
        if num_black < search_cfg.num_black:
            available_stages.append("black")

        if available_stages:
            child_stage = random.choice(available_stages)
            new_node = expand_node(parent, child_stage, getattr(parent, "code", ""), getattr(parent, "stdout", ""), "")
            best_state["active_jobs"] = int(best_state["active_jobs"] or 0) + 1
            self.logger.info(f"Expanded parent {parent.id[:8]} with {child_stage} node {new_node.id[:8]} (1/2)")

        num_red = sum(1 for c in parent.children if c.stage == "red")
        num_black = sum(1 for c in parent.children if c.stage == "black")
        available_stages = []
        if self.use_red_node and num_red < search_cfg.num_red:
            available_stages.append("red")
        elif not self.use_red_node and num_red < search_cfg.num_red:
            self.logger.info(
                f"No-Red ablation enabled: skip red sibling creation for parent {parent.id[:8]}"
            )
        if num_black < search_cfg.num_black:
            available_stages.append("black")

        if available_stages:
            child_stage = random.choice(available_stages)
            new_node = expand_node(parent, child_stage, getattr(parent, "code", ""), getattr(parent, "stdout", ""), "")
            best_state["active_jobs"] = int(best_state["active_jobs"] or 0) + 1
            self.logger.info(f"Expanded parent {parent.id[:8]} with {child_stage} node {new_node.id[:8]} (2/2)")

    def _expand_current_node(
        self,
        node: Any,
        search_mgr: UCTSearchManager,
        search_cfg: UCTSearchConfig,
        best_state: dict,
        expand_node: Callable,
    ) -> None:
        """扩展当前节点（创建子节点）"""
        num_red = sum(1 for c in node.children if c.stage == "red")
        num_black = sum(1 for c in node.children if c.stage == "black")

        if self.use_red_node and num_red < search_cfg.num_red:
            child_stage = "red"
        elif num_black < search_cfg.num_black:
            child_stage = "black"
        else:
            if not self.use_red_node and num_red < search_cfg.num_red:
                self.logger.info(
                    f"No-Red ablation enabled: skip red child creation for node {node.id[:8]}"
                )
            return

        new_node = expand_node(node, child_stage, getattr(node, "code", ""), getattr(node, "stdout", ""), "")
        best_state["active_jobs"] = int(best_state["active_jobs"] or 0) + 1
        self.logger.info(f"Expanded node {node.id[:8]} with {child_stage} node {new_node.id[:8]}")

    def _save_node_created_snapshot(
        self, node: Any, search_mgr: UCTSearchManager, best_state: dict
    ) -> None:
        """保存节点创建快照"""
        save_node_snapshot(
            self.run_dir,
            Path(self.session.config.workspace_path),
            node,
            None,
            MetricReview(metric=None, is_bug=False, has_submission=False, summary="node created"),
            0.0,
            search_mgr,
            task_description=best_state.get("task_description", ""),
            snapshot_event="created",
        )

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        try:
            self.setup()
            self._ensure_grading_server_ready()

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
            max_steps = getattr(self.config, "max_steps", None)
            logger.info(f"Current experiments max steps: {max_steps}")
            if max_steps is None:
                logger.error("Current experiments max steps is none!")

            state_lock = threading.Lock()
            worker_agents_map = {
                i: self._create_worker_agents(i) for i in range(self.max_workers)
            }
            expand_node = self._create_expand_node_fn(search_mgr, task_description)

            def worker_loop(worker_index: int) -> dict[str, Any]:
                """Worker 主循环"""
                worker_agents = worker_agents_map[worker_index]
                worker_workspace = self._resolve_worker_workspace(worker_index, workspace)
                worker_workspace.mkdir(parents=True, exist_ok=True)
                self._ensure_prepared_links(worker_workspace)
                (worker_workspace / "working").mkdir(parents=True, exist_ok=True)
                (worker_workspace / "submission").mkdir(parents=True, exist_ok=True)
                worker_submission_dir = worker_workspace / "submission"
                data_preview = generate_data_preview(worker_workspace)
                completed = 0

                while True:
                    should_wait, node_to_execute = self._select_node_to_execute(
                        worker_index, search_mgr, best_state, state_lock, max_steps
                    )

                    if should_wait:
                        time.sleep(0.1)
                        continue

                    if node_to_execute is None:
                        break

                    completed += self._execute_and_process_node(
                        node=node_to_execute,
                        worker_agents=worker_agents,
                        worker_workspace=worker_workspace,
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
                    self.logger.error("Worker %s returned exception: %s", idx, wr)
                else:
                    self.logger.info("Worker summary: %s", wr)

            return results
        finally:
            self.cleanup()