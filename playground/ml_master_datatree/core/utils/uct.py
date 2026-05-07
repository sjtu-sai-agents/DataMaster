"""数据探索 Agent 的 UCT 工具模块。

用途：自动化数据探索和发现的 UCT 搜索逻辑。
节点类型：
- root: 根节点，生成初始代码
- black: 黑色节点，负责数据增强、特征工程
- red: 红色节点，负责外部数据搜索和引入

核心功能：
- 解析执行结果得到验证指标（通常由 LLM 解析器完成）
- 更新节点奖励与 UCT 分数
- 选择下一轮要扩展的节点
- buggy 节点终止，非 buggy 节点生成 black/red 子节点

快速示例（伪代码）：
    from core.utils.uct import (
        MetricReview, UCTDecayConfig, UCTSearchConfig, UCTSearchManager
    )
    mgr = UCTSearchManager(UCTSearchConfig(), UCTDecayConfig())
    root = mgr.root
    node = mgr.create_child(root, stage="black", plan="数据增强", code="...")
    review = MetricReview(metric=0.82, lower_is_better=False, summary="val acc=0.82")
    mgr.ingest_result(node, review)
    next_node = mgr.select_next()
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

StageLiteral = Literal["root", "black", "red"]


# --------------------------------------------------------------------------- #
# 指标处理
# --------------------------------------------------------------------------- #


@dataclass
class MetricReview:
    """外部解析器/LLM 返回的标准化度量结果。

    Args:
        metric: 验证指标值（None 表示缺失或失败）。
        lower_is_better: 若为 True 则代表越小越好，会反转 maximize。
        maximize: 是否越大越好（默认 True）。
        is_bug: 解析认定是否有 bug。
        has_submission: 是否生成了提交文件。
        summary: 对执行结果的简述。
        raw_output: 原始解析文本（可选）。
    """

    metric: Optional[float]
    lower_is_better: Optional[bool] = None
    maximize: bool = True
    is_bug: bool = False
    has_submission: bool = True
    summary: str = ""
    raw_output: Optional[str] = None

    def __post_init__(self) -> None:
        if self.lower_is_better is not None:
            self.maximize = not self.lower_is_better
        if self.metric is not None:
            self.metric = float(self.metric)


@dataclass
class MetricValue:
    """可比较的指标包装，供 UCT 打分使用。"""

    value: Optional[float]
    maximize: bool = True

    def __post_init__(self) -> None:
        if self.value is not None:
            self.value = float(self.value)

    def __gt__(self, other: "MetricValue") -> bool:  # type: ignore[override]
        if self.value is None:
            return False
        if other.value is None:
            return True
        if self.value == other.value:
            return False
        comp = self.value > other.value
        return comp if self.maximize else not comp


class WorstMetricValue(MetricValue):
    """表示最差值（bug/无效）。"""

    def __init__(self) -> None:
        super().__init__(value=None, maximize=True)


# --------------------------------------------------------------------------- #
# 搜索与衰减配置
# --------------------------------------------------------------------------- #


@dataclass
class UCTSearchConfig:
    """数据探索 Agent 搜索超参。"""

    # 节点扩展数量限制
    num_black: int = 5     # black 节点（数据增强）上限
    num_red: int = 1       # red 节点（外部数据源）上限

    # Buggy节点惩罚
    buggy_penalty: float = 1000.0  # buggy节点的UCT惩罚值（从UCT值中减去）

    # 评估参数
    invalid_metric_upper_bound: int = 100
    metric_improvement_threshold: float = 0.0001


@dataclass
class UCTDecayConfig:
    """探索系数衰减配置。"""

    decay_type: Literal[
        "constant",
        "linear",
        "exponential",
        "piecewise",
        "dynamic_piecewise",
    ] = "piecewise"
    exploration_constant: float = 1.414
    lower_bound: float = 0.5

    # 线性衰减
    linear_alpha: float = 0.01
    # 指数衰减
    exponential_gamma: float = 0.99
    # 分段衰减
    piecewise_alpha: float = 0.01
    piecewise_phase_ratios: tuple[float, float] = (0.3, 0.7)
    # 动态分段衰减
    dynamic_alpha: float = 0.01
    dynamic_phase_ratios: tuple[float, float] = (0.85, 1.0)


def _linear_decay(t: int, initial_c: float, alpha: float, lower_bound: float) -> float:
    """线性衰减。

    Args:
        t: 当前步数
        initial_c: 初始探索系数
        alpha: 衰减斜率
        lower_bound: 下限
    Returns:
        衰减后的探索系数
    """
    return max(initial_c - alpha * t, lower_bound)


def _exponential_decay(
    t: int, initial_c: float, gamma: float, lower_bound: float
) -> float:
    """指数衰减。

    Args:
        t: 当前步数
        initial_c: 初始探索系数
        gamma: 衰减系数
        lower_bound: 下限
    Returns:
        衰减后的探索系数
    """
    return max(initial_c * (gamma**t), lower_bound)


def _piecewise_decay(
    t: int,
    initial_c: float,
    t1: int,
    t2: int,
    alpha: float,
    lower_bound: float,
) -> float:
    """分段线性衰减。

    Args:
        t: 当前步数
        initial_c: 初始探索系数
        t1: 第一阶段结束步数
        t2: 第二阶段结束步数
        alpha: 第二阶段斜率
        lower_bound: 下限
    Returns:
        衰减后的探索系数
    """
    if t < t1:
        return initial_c
    if t <= t2:
        return max(initial_c - alpha * (t - t1), lower_bound)
    return lower_bound


def _dynamic_piecewise_decay(
    steps_limit: int,
    n_nodes: int,
    initial_c: float,
    start_time: float,
    time_limit: float,
    alpha: float,
    lower_bound: float,
    phase_ratios: tuple[float, float],
) -> float:
    """动态分段衰减，根据时间/进度估算。

    Args:
        steps_limit: 计划最大节点数
        n_nodes: 已生成节点数
        initial_c: 初始探索系数
        start_time: 搜索起始时间戳
        time_limit: 总时间限制
        alpha: 衰减斜率
        lower_bound: 下限
        phase_ratios: 两阶段分界比例
    Returns:
        衰减后的探索系数
    """
    now = time.time()
    elapsed = now - start_time
    remaining = max(time_limit - elapsed, 1e-5)

    speed = n_nodes / elapsed if elapsed > 0 else 1.0
    n_remaining = round(speed * remaining)
    estimated_total = min(n_nodes + n_remaining, steps_limit)
    progress = n_nodes / estimated_total if estimated_total > 0 else 0.0

    p1, p2 = phase_ratios
    if progress < p1:
        return initial_c
    if progress < p2:
        decay_length = p2 - p1
        decay_progress = (progress - p1) / decay_length if decay_length > 0 else 0
        c_val = initial_c - alpha * decay_progress * estimated_total
        return max(c_val, lower_bound)
    return lower_bound


# --------------------------------------------------------------------------- #
# UCT 最大堆（用于按 UCT 值优先执行节点）
# --------------------------------------------------------------------------- #

import heapq
from typing import Any, Callable

class UCTMaxHeap:
    """基于 UCT 值的最大堆，用于优先执行高价值节点"""

    def __init__(self, uct_func: Callable[[UCTNode], float]):
        """
        Args:
            uct_func: 计算 UCT 值的函数，签名为 (node) -> float
        """
        self.uct_func = uct_func
        self.heap = []  # Python heapq 是最小堆，所以存储负值
        self.nodes = {}  # node_id -> node 映射，用于更新
        self.lock = threading.Lock()

    def push(self, node: UCTNode) -> None:
        """将节点加入堆"""
        with self.lock:
            if node.id in self.nodes:
                return  # 节点已存在
            uct_value = self.uct_func(node)  # 使用传入的 uct_func
            heapq.heappush(self.heap, (-uct_value, node.id))
            self.nodes[node.id] = node

    def pop_max(self) -> Optional[UCTNode]:
        """弹出 UCT 值最大的节点"""
        with self.lock:
            if not self.heap:
                return None
            neg_uct, node_id = heapq.heappop(self.heap)
            node = self.nodes.pop(node_id)
            return node

    def peek_max(self) -> Optional[UCTNode]:
        """查看但不弹出 UCT 值最大的节点"""
        with self.lock:
            if not self.heap:
                return None
            neg_uct, node_id = self.heap[0]
            return self.nodes.get(node_id)

    def is_empty(self) -> bool:
        """堆是否为空"""
        with self.lock:
            return len(self.heap) == 0

    def size(self) -> int:
        """堆中节点数量"""
        with self.lock:
            return len(self.heap)

    def clear(self) -> None:
        """清空堆"""
        with self.lock:
            self.heap.clear()
            self.nodes.clear()





@dataclass(eq=False)
class UCTNode:
    """UCT 树节点，记录计划/代码/奖励/子节点等状态。"""

    stage: StageLiteral
    plan: str = ""
    code: str = ""
    stdout: Optional[str] = None  # 追加：保存最近一次执行的输出，便于调试
    exit_code: Optional[int] = None  # 追加：保存最近一次执行的退出码
    parent: Optional["UCTNode"] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)

    # 执行信息与元数据
    analysis: Optional[str] = None
    metric: MetricValue = field(default_factory=WorstMetricValue)
    is_buggy: Optional[bool] = None
    is_valid: Optional[bool] = None
    grading_status: Optional[str] = None
    grading_detail: Optional[str] = None
    original_is_buggy: Optional[bool] = None  # Save original buggy state for snapshot tracing
    is_terminal: bool = False
    finish_time: Optional[float] = None
    llm_time_cost: float = 0.0
    tool_time_cost: float = 0.0
    total_time_cost: float = 0.0
    is_debug_success: bool = False
    continue_improve: bool = False
    improve_failure_depth: int = 0
    local_best_node: Optional["UCTNode"] = None

    # 树统计
    visits: int = 0
    total_reward: float = 0.0
    children: set["UCTNode"] = field(default_factory=set)
    expected_child_count: int = 0
    locked: bool = False
    # Track first-time stats for logging/debugging.
    initial_reward: float | None = None
    initial_total_reward: float | None = None
    initial_visits: int | None = None
    initial_uct: float | None = None

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def num_children(self) -> int:
        return len(self.children)

    @property
    def debug_depth(self) -> int:
        if self.stage != "debug" or self.parent is None:
            return 0
        return 1 + self.parent.debug_depth

    def expect_child(self) -> None:
        self.expected_child_count += 1

    def complete_child(self) -> None:
        self.expected_child_count = max(self.expected_child_count - 1, 0)

    def is_fully_expanded(self, cfg: UCTSearchConfig) -> bool:
        """判断节点是否已完全扩展。

        root 节点：生成 initial 节点后不再扩展
        black/red 节点：如果 buggy 则终止，否则按 num_black/num_red 限制
        """
        if self.stage == "root":
            # root 只初始化一次
            return self.expected_child_count >= 1

        # buggy 节点不再被硬性阻止，而是通过 UCT 惩罚来降低优先级
        # 但仍然限制扩展数量以避免无限扩展
        if self.is_buggy:
            # buggy 节点限制扩展数量，防止无限生成失败的变体
            max_buggy_children = 3  # 最多允许2个子节点尝试修复
            return self.expected_child_count >= max_buggy_children

        # 非 buggy 节点检查是否达到扩展上限
        if self.stage == "black":
            return self.expected_child_count >= (cfg.num_black + cfg.num_red)
        if self.stage == "red":
            return self.expected_child_count >= (cfg.num_black + cfg.num_red)
        if self.stage == "initial":
            return self.expected_child_count >= (cfg.num_black + cfg.num_red)
        return True

    def _has_non_bug_child(self) -> bool:
        return any(child.is_buggy is False for child in self.children)

    def uct_value(self, exploration_constant: float, parent_visits: int, buggy_penalty: float | None = None, search_cfg: UCTSearchConfig | None = None) -> float:
        """计算节点的 UCT 值，支持对 buggy 节点施加惩罚。

        Args:
            exploration_constant: 探索系数
            parent_visits: 父节点访问次数
            buggy_penalty: buggy节点的惩罚值（从UCT值中减去）。如果为None且提供了search_cfg，则从search_cfg获取
            search_cfg: 搜索配置，用于获取默认的buggy_penalty

        Returns:
            节点的 UCT 值，buggy 节点会减去惩罚值
        """
        # 如果没有明确指定惩罚，从search_cfg获取或使用默认值0
        if buggy_penalty is None:
            if self.is_buggy and search_cfg is not None:
                buggy_penalty = search_cfg.buggy_penalty
            else:
                buggy_penalty = 0.0

        if self.visits == 0:
            base_value = float("inf")
            # buggy节点使用大的负值惩罚，确保优先级低于任何非buggy节点
            return base_value - buggy_penalty if self.is_buggy else base_value

        parent_total = max(parent_visits, 1)
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(math.log(parent_total) / self.visits)
        uct = exploitation + exploration

        # 对buggy节点应用惩罚值
        return uct - buggy_penalty if self.is_buggy else uct

    def update_reward(self, reward: float) -> None:
        self.visits += 1
        self.total_reward += reward

    def fetch_child_memory(self, include_code: bool = False) -> str:
        """Summarize child nodes for prompt Memory (ported from ML-Master MCTSNode)."""
        logger.info("fetch_child_memory")
        summary: list[str] = []
        for child in self.children:
            if child.is_buggy is None:
                continue
            part = f"Design: {child.plan}\n"
            if include_code:
                part += f"Code: {child.code}\n"
            if child.is_buggy:
                part += "Results: The implementation of this design has bugs.\n"
                part += "Insight: Using a different approach may not result in the same bugs as the above approach.\n"
            else:
                if child.analysis:
                    part += f"Results: {child.analysis}\n"
                if child.metric:
                    part += f"Validation Metric: {child.metric.value}\n"
            summary.append(part)
        if not summary:
            summary.append("There is no previous memory")
        return "\n-------------------------------\n".join(summary)

    def fetch_parent_memory(self, include_code: bool = False) -> str:
        """Summarize parent when it is a successful node."""
        logger.info("fetch_parent_memory")
        if self.parent and self.parent.is_buggy is False:
            part = f"Design: {self.parent.plan}\n"
            if include_code:
                part += f"Code: {self.parent.code}\n"
            if self.parent.analysis:
                part += f"Results: {self.parent.analysis}\n"
            if self.parent.metric:
                part += f"Validation Metric: {self.parent.metric.value}\n"
            return part
        return ""


# --------------------------------------------------------------------------- #
# 搜索管理器
# --------------------------------------------------------------------------- #

MetricParser = Callable[[str, str, Optional[str]], MetricReview]


class UCTSearchManager:
    """UCT 状态管理器，复刻 ML-Master 的 select/backprop 流程。"""

    def __init__(
        self,
        search_cfg: UCTSearchConfig,
        decay_cfg: UCTDecayConfig,
        *,
        time_limit: float = 0,
        grader: Optional[Callable[[str, Path], Tuple[bool, dict | str]]] = None,
        exp_id: Optional[str] = None,
        submission_dir: Optional[Path | str] = None,
    ) -> None:
        self.search_cfg = search_cfg
        self.decay_cfg = decay_cfg
        self.time_limit = time_limit
        self.grader = grader
        self.exp_id = exp_id
        self.submission_dir = Path(submission_dir) if submission_dir else None
        # Optional snapshot callback: fn(node, submission_path, review, reward) -> None
        self.snapshot_fn: Optional[Callable[[UCTNode, Optional[Path], MetricReview, float], None]] = None

        self.root = UCTNode(stage="root", plan="virtual root", code="")
        self.best_node: Optional[UCTNode] = None
        self.best_metric: Optional[float] = None

        self.current_step: int = 0
        self.search_start_time = time.time()

        # 基于 UCT 的最大堆，用于按优先级执行节点
        self.execution_heap = UCTMaxHeap(self._get_node_uct_value)

    def _get_node_uct_value(self, node: UCTNode) -> float:
        """获取节点的 UCT 值，用于堆排序（考虑buggy惩罚）"""
        return node.uct_value(
            self._exploration_constant(),
            node.parent.visits if node.parent else 1,
            search_cfg=self.search_cfg
        )

    def push_execution_node(self, node: UCTNode) -> None:
        """将节点加入执行堆"""
        self.execution_heap.push(node)

    def pop_execution_node(self) -> Optional[UCTNode]:
        """从执行堆弹出优先级最高的节点"""
        return self.execution_heap.pop_max()

    def should_expand_parent(self, node: UCTNode) -> bool:
        """判断是否应该扩展父节点（而不是扩展当前节点）"""
        if node.parent is None:
            return False  # root 节点没有父节点

        parent = node.parent
        if parent.stage == "root":
            return False  # root 不再扩展


        # 检查父节点的子节点数量
        if parent.stage == "initial":
            # initial 节点可以有多个子节点
            return parent.num_children < (self.search_cfg.num_black + self.search_cfg.num_red)

        # black/red 节点也可以有子节点
        return parent.num_children < (self.search_cfg.num_black + self.search_cfg.num_red)

    def should_expand_node(self, node: UCTNode) -> bool:
        """判断节点是否应该被扩展（创建子节点）

        buggy 节点也可以扩展，但通过 UCT 惩罚降低优先级。
        """
        if node.is_terminal:
            return False  # terminal 节点不扩展
        if node.stage == "root":
            return False  # root 不再扩展
        return not node.is_fully_expanded(self.search_cfg)

    # ---- 对外 API ----------------------------------------------------- #

    def create_child(
        self,
        parent: UCTNode,
        stage: StageLiteral,
        plan: str = "",
        code: str = "",
    ) -> UCTNode:
        """创建子节点并增加父节点预期子数量。

        Args:
            parent: 父节点
            stage: 节点阶段（root/initial/black/red）
            plan: 方案描述
            code: 生成的代码
        Returns:
            新建的子节点
        """
        parent.expect_child()
        node = UCTNode(stage=stage, plan=plan, code=code, parent=parent)
        logger.info(f"Created child node {node.id} stage={stage} parent={parent.id if parent else None} plan={plan[:80]!r}")
        return node

    def select_next(self, node: Optional[UCTNode] = None) -> Optional[UCTNode]:
        """基于 UCT 的节点选择。

        Args:
            node: 可选，起始节点；默认从 root 开始
        Returns:
            选出的下一个扩展节点，如果没有可扩展的节点则返回 None
        """
        selected = node or self.root
        logger.info(f"===== select_next() starting from node {selected.id[:8]} stage={selected.stage} =====")
        iteration = 0
        while selected and not selected.is_terminal:
            iteration += 1
            logger.info(f"[Iteration {iteration}] At node {selected.id[:8]} stage={selected.stage} visits={selected.visits} expected_child_count={selected.expected_child_count} children={selected.num_children} is_buggy={selected.is_buggy} continue_improve={selected.continue_improve}")

            if not selected.is_fully_expanded(self.search_cfg):
                logger.info(f"[Iteration {iteration}] Node not fully expanded")
                if selected.is_buggy and selected.is_debug_success:
                    selected = self._uct_select(selected)
                    if selected is None:
                        logger.info(f"[Iteration {iteration}] _uct_select returned None (buggy debug path)")
                        return None
                elif selected.continue_improve and selected.children:
                    logger.info(f"[Iteration {iteration}] Node has continue_improve=True and children, calling _uct_select")
                    selected = self._uct_select(selected)
                    if selected is None:
                        logger.info(f"[Iteration {iteration}] _uct_select returned None (continue_improve path)")
                        return None
                else:
                    logger.info(f"[Iteration {iteration}] Selected for expansion: {selected.id[:8]} (not fully expanded, no special conditions)")
                    return selected
            else:
                logger.info(f"[Iteration {iteration}] Node fully expanded")
                # 完全扩展后，选择子节点继续向下（如果有子节点）
                if selected.children:
                    logger.info(f"[Iteration {iteration}] Node has {selected.num_children} children, calling _uct_select")
                    child = self._uct_select(selected)
                    if child is None:
                        # 没有可选择的子节点（所有子节点都在执行或已终止）
                        logger.info(f"[Iteration {iteration}] _uct_select returned None (fully expanded path), returning None")
                        return None
                    logger.info(f"[Iteration {iteration}] _uct_select returned child {child.id[:8]} stage={child.stage}, continuing loop")
                    selected = child
                else:
                    logger.info(f"[Iteration {iteration}] Node fully expanded but no children, returning {selected.id[:8]}")
                    return selected
        logger.info(f"===== select_next() finished: node={selected.id[:8] if selected else None} =====")
        return selected

    def ingest_result(
        self,
        node: UCTNode,
        review: MetricReview,
    ) -> float:
        """写回节点执行结果，计算奖励并回传。

        Args:
            node: 当前节点
            review: 解析后的度量结果
        Returns:
            本次回传的奖励值
        """
        node.finish_time = time.time()
        node.analysis = review.summary

        # Preserve parser output for observability even if later validation marks node buggy.
        node.last_review_metric = review.metric
        node.last_review_lower_is_better = review.lower_is_better
        node.last_review_has_submission = review.has_submission
        node.last_review_is_bug = review.is_bug

        # =========================
        # 核心修复 1：
        # 只把“代码本身被 metric/parser 判定有 bug”视为 is_buggy 的核心来源
        # 不再因为 metric 缺失 / submission 缺失就直接把节点硬判死
        # =========================
        node.is_buggy = review.is_bug

        # 先默认认为有效；后面如果 grader 明确说 invalid，再改成 False
        node.is_valid = not node.is_buggy

        # metric 缺失时给 WorstMetricValue，但不直接等于 buggy
        node.metric = (
            WorstMetricValue()
            if review.metric is None
            else MetricValue(review.metric, maximize=review.maximize)
        )

        # 记录一些弱错误信息，但不直接判死
        if review.metric is None:
            node.analysis = f"{node.analysis or ''}\n[metric] metric missing".strip()
        if not review.has_submission:
            node.analysis = f"{node.analysis or ''}\n[submission] submission missing".strip()

        # Save original buggy state on first ingest for accurate snapshot tracing
        if node.original_is_buggy is None:
            node.original_is_buggy = node.is_buggy

        # Reject nodes whose metric direction conflicts with existing best node.
        if (
            node.is_buggy is False
            and review.metric is not None
            and self.best_node
            and self.best_node.metric
            and node.metric.maximize != self.best_node.metric.maximize
        ):
            logger.warning(
                "Metric direction conflict: node %s maximize=%s vs best maximize=%s. Marking node as buggy.",
                node.id,
                node.metric.maximize,
                self.best_node.metric.maximize,
            )
            node.metric = WorstMetricValue()
            node.is_buggy = True
            node.is_valid = False
            node.analysis = f"{node.analysis or ''}\n[metric] direction mismatch with best node".strip()

        node.continue_improve = (node.is_buggy is False)

        # 如果父节点是 buggy 而当前节点成功，标记父节点为非 buggy
        if node.parent and node.parent.is_buggy and node.is_buggy is False:
            node.parent.is_buggy = False

        if node.parent and node.parent.stage != "root":
            node.parent.continue_improve = node.continue_improve

        # =========================
        # 核心修复 2：
        # grading server 调用失败，不再直接设成 is_buggy=True
        # 只有 grader 明确返回 is_valid=False，才认为 submission 非法
        # =========================
        if (
            self.grader
            and self.exp_id
            and self.submission_dir
            and not node.is_buggy
        ):
            submission_path = self.submission_dir / f"submission_{node.id}.csv"
            if submission_path.exists():
                ok, res = self.grader(self.exp_id, submission_path)

                if ok:
                    if isinstance(res, dict):
                        is_valid = res.get("is_valid", True)
                        detail = res.get("result") or res.get("details") or ""

                        if is_valid:
                            node.is_valid = True
                            node.grading_status = "passed"
                            node.grading_detail = detail or None
                        else:
                            # 只有 grader 明确返回 invalid，才影响节点语义
                            node.is_valid = False
                            node.is_buggy = True
                            node.metric = WorstMetricValue()
                            node.grading_status = "invalid_submission"
                            node.grading_detail = detail or "submission 格式非法"
                            logger.info(
                                "Grader marked node %s as invalid: %s",
                                node.id,
                                node.grading_detail,
                            )
                            node.analysis = (
                                f"{node.analysis or ''}\n[grading] {node.grading_detail}"
                            ).strip()
                    else:
                        node.is_valid = None
                        node.grading_status = "bad_response"
                        node.grading_detail = str(res)
                        logger.warning(
                            "Unexpected grader response for node %s: %r",
                            node.id,
                            res,
                        )
                        node.analysis = (
                            f"{node.analysis or ''}\n[grading] unexpected grader response"
                        ).strip()
                else:
                    # 关键修复：grader 不可用 ≠ node buggy
                    # 注意：server unavailable 表示验证未完成，不等于 submission invalid
                    node.is_valid = None
                    node.grading_status = "server_unavailable"
                    node.grading_detail = str(res)
                    logger.warning(
                        "Grading server unavailable for node %s: %s",
                        node.id,
                        res,
                    )
                    node.analysis = (
                        f"{node.analysis or ''}\n[grading] grading server unavailable, skipped"
                    ).strip()

        # grading 结果可能修改了 is_buggy，这里再同步一次 continue_improve
        node.continue_improve = (node.is_buggy is False)

        # 额外的 metric 合法性防护，防止异常放大
        if (
            not node.is_buggy
            and node.metric.value is not None
            and not self._check_metric_valid(node)
        ):
            node.metric = WorstMetricValue()
            node.is_buggy = True
            node.analysis = f"{node.analysis or ''}\n[metric] invalid metric detected".strip()

        # 依据改进成效更新状态，控制继续改进/终止
        self._check_improvement(node)

        # 计算 reward 并回传，同时记录详细信息
        reward = self._get_node_reward(node)
        logger.info(
            f"Ingested result for node {node.id}: "
            f"stage={node.stage} is_buggy={node.is_buggy} "
            f"metric={getattr(node.metric, 'value', None)} "
            f"grading_status={node.grading_status} reward={reward:.3f}"
        )
        self._backpropagate(node, reward)
        logger.debug(
            f"After backpropagate: node {node.id} visits={node.visits} total_reward={node.total_reward}"
        )

        # Record initial stats the first time this node itself is ingested.
        if node.initial_reward is None:
            node.initial_reward = reward
            node.initial_total_reward = node.total_reward
            node.initial_visits = node.visits
            # Cache initial uct value at first ingestion for logging.
            try:
                parent_visits = node.parent.visits if node.parent else 1
                node.initial_uct = node.uct_value(
                    self._exploration_constant(),
                    parent_visits,
                    search_cfg=self.search_cfg,
                )
            except Exception:
                node.initial_uct = None

        # Persist snapshots for current node and its ancestors with latest rewards.
        if self.snapshot_fn:
            submission_path = (
                self.submission_dir / f"submission_{node.id}.csv"
                if self.submission_dir and (self.submission_dir / f"submission_{node.id}.csv").exists()
                else None
            )
            current = node
            while current:
                sub = submission_path if current is node else None
                try:
                    self.snapshot_fn(current, sub, review, reward)
                except Exception as exc:
                    logger.warning("Snapshot callback failed for node %s: %s", current.id, exc)
                current = current.parent  # type: ignore[assignment]

        self.current_step += 1
        return reward

    def set_snapshot_fn(
        self,
        fn: Callable[[UCTNode, Optional[Path], MetricReview, float], None],
    ) -> None:
        """Register a callback to persist node snapshots after each backprop."""
        self.snapshot_fn = fn

    # ---- 内部实现 ------------------------------------------------------ #

    def _backpropagate(self, node: UCTNode, reward: float) -> None:
        current = node
        while current is not None:
            if current.stage == "initial" and current.locked:
                current.locked = False
            current.update_reward(reward)
            logger.debug(f"Backpropagate node {current.id}: visits={current.visits} total_reward={current.total_reward}")
            current = current.parent  # type: ignore[assignment]

    def _get_node_reward(self, node: UCTNode) -> float:
        if node.is_buggy or node.metric.value is None:
            return -1.0

        reward = 1.0
        parent = node.parent
        if parent and parent.is_buggy:
            reward += 1.0

        if (
            self.best_node
            and self.best_node.metric
            and self.best_node.metric.maximize == node.metric.maximize
            and self.best_metric is not None
            and node.metric.value is not None
        ):
            improvement = (
                node.metric.value - self.best_metric
                if node.metric.maximize
                else self.best_metric - node.metric.value
            )
            if improvement > 0:
                reward += 1.0

        if node.metric.value is not None:
            # Only update best when metric direction matches current best (or when best is None).
            if self.best_node is None or self.best_node.metric.maximize == node.metric.maximize:
                # Guard again before writing best_metric to avoid invalid spikes.
                if self._check_metric_valid(node):
                    if self.best_metric is None or (self.best_node and node.metric > self.best_node.metric):
                        self.best_metric = node.metric.value
                        self.best_node = node

        return reward

    def _check_metric_valid(self, node: UCTNode, upper_bound: int | None = None) -> bool:
        """Guard against abnormally large/small metrics compared to current best."""
        bound = upper_bound or getattr(self.search_cfg, "invalid_metric_upper_bound", 100)
        v1 = self.best_metric
        v2 = node.metric.value
        if v1 is None or v2 is None:
            return True
        if v1 == 0 or v2 == 0:
            return abs(v1 - v2) <= bound
        ratio = max(abs(v1), abs(v2)) / min(abs(v1), abs(v2))
        return ratio <= bound

    def _check_improvement(self, node: UCTNode) -> None:
        """Update improvement bookkeeping for data exploration Agent.

        In the new design:
        - Buggy nodes terminate (no debug retries)
        - Non-buggy nodes continue to improve
        - Focus on metric improvement threshold
        """
        parent = node.parent
        local_best = node.local_best_node or (parent.local_best_node if parent else None) or parent
        scfg = self.search_cfg

        if node.is_buggy is False:
            new_metric = node.metric.value
            if parent and parent.is_buggy:
                node.continue_improve = False
                node.is_terminal = False
                return

            if new_metric is not None and local_best and local_best.metric.value is not None:
                improvement = (
                    new_metric - local_best.metric.value
                    if node.metric.maximize
                    else local_best.metric.value - new_metric
                )
                if improvement < scfg.metric_improvement_threshold:
                    # Continue trying to improve
                    node.continue_improve = True
                else:
                    node.local_best_node = node
                    node.continue_improve = True
            elif new_metric is not None:
                node.local_best_node = node
                node.continue_improve = True
            else:
                node.continue_improve = False
        elif node.is_buggy is True:
            # Buggy nodes should NOT be marked as terminal
            # They should still be expandable for repair attempts
            # Just mark continue_improve=False to indicate no improvement happened
            node.continue_improve = False
            # 🔧 修复：不设置 is_terminal=True，允许 buggy 节点继续扩展
            # node.is_terminal = True  # 这行会导致搜索提前终止！
        else:
            node.continue_improve = False


    def _uct_select(self, node: UCTNode) -> Optional[UCTNode]:
        """从子节点中选择 UCT 值最大的节点。

        Returns:
            选中的子节点，如果没有可选择的子节点则返回 None
        """
        c_val = self._exploration_constant()
        if node.stage == "root":
            unlocked = [child for child in node.children if not child.locked]
            if not unlocked:
                # 所有子节点都被锁定，返回 None
                logger.info("All root children are locked, no selectable node")
                return None

            # 🔧 修复：优先选择非 buggy 且非 terminal 的节点
            available = [child for child in unlocked if child.is_buggy is not None and not child.is_terminal and child.is_buggy is False]

            if not available:
                # 如果没有理想的节点，尝试选择可扩展的 buggy 节点
                logger.warning(f"No non-buggy children available, trying buggy nodes")
                buggy_available = [
                    child for child in unlocked
                    if child.is_buggy is True
                    and child.is_buggy is not None  # 已完成执行
                    and not child.is_terminal  # 未终止
                    and child.num_children < (self.search_cfg.num_black + self.search_cfg.num_red)  # 未完全扩展
                ]
                if buggy_available:
                    # 选择 UCT 最高的 buggy 节点（即使它是 buggy）
                    picked = max(buggy_available, key=lambda child: child.uct_value(c_val, node.visits, search_cfg=self.search_cfg))
                    logger.info(f"_uct_select(root) selecting buggy node {picked.id[:8]} as fallback (UCT={picked.uct_value(c_val, node.visits, search_cfg=self.search_cfg):.4f})")
                    return picked
                else:
                    logger.info(f"_uct_select(root) no selectable nodes (all executing/terminal or fully expanded)")
                    return None

            # 记录可选子节点的 UCT 值
            scores = [(child, child.uct_value(c_val, node.visits, search_cfg=self.search_cfg)) for child in available]
            for ch, sc in scores:
                logger.info(f"Root child {ch.id[:8]} stage={ch.stage} uct={sc:.4f} is_buggy={ch.is_buggy} visits={ch.visits}")
            picked = max(available, key=lambda child: child.uct_value(c_val, node.visits, search_cfg=self.search_cfg))
            if picked.stage == "initial":
                picked.locked = True
            logger.info(f"_uct_select(root) returning child {picked.id[:8]} stage={picked.stage}")
            return picked

        # 对非 root 节点
        if not node.children:
            logger.info(f"_uct_select({node.id[:8]}) has no children, returning None")
            return None

        # 🔧 修复：优先选择非 buggy 且非 terminal 的子节点
        available = [
            child for child in node.children
            if child.is_buggy is not None  # 已完成执行
            and not child.is_terminal  # 未终止
            and child.is_buggy is False  # 非 buggy
        ]

        if not available:
            # 如果没有理想的节点，尝试选择可扩展的 buggy 节点
            logger.warning(f"_uct_select({node.id[:8]}) no non-buggy children available, trying buggy nodes")
            buggy_available = [
                child for child in node.children
                if child.is_buggy is True
                and child.is_buggy is not None  # 已完成执行
                and not child.is_terminal  # 未终止
                and child.num_children < (self.search_cfg.num_black + self.search_cfg.num_red)  # 未完全扩展
            ]
            if buggy_available:
                picked = max(buggy_available, key=lambda child: child.uct_value(c_val, node.visits, search_cfg=self.search_cfg))
                logger.info(f"_uct_select({node.id[:8]}) selecting buggy child {picked.id[:8]} as fallback")
                return picked
            else:
                logger.info(f"_uct_select({node.id[:8]}) no selectable nodes (all executing/terminal or fully expanded)")
                return None

        # 对非 root 节点也记录 UCT 值
        scores = [(child, child.uct_value(c_val, node.visits, search_cfg=self.search_cfg)) for child in available]
        for ch, sc in scores:
            logger.info(f"Child {ch.id[:8]} stage={ch.stage} uct={sc:.4f} is_buggy={ch.is_buggy}")
        picked = max(available, key=lambda child: child.uct_value(c_val, node.visits, search_cfg=self.search_cfg))
        logger.info(f"_uct_select({node.id[:8]}) returning {picked.id[:8]} stage={picked.stage}")
        return picked

    def _exploration_constant(self) -> float:
        cfg = self.decay_cfg
        t = self.current_step
        if cfg.decay_type == "linear":
            c_val = _linear_decay(t, cfg.exploration_constant, cfg.linear_alpha, cfg.lower_bound)
        elif cfg.decay_type == "exponential":
            c_val = _exponential_decay(
                t,
                cfg.exploration_constant,
                cfg.exponential_gamma,
                cfg.lower_bound,
            )
        elif cfg.decay_type == "piecewise":
            t1 = round(cfg.piecewise_phase_ratios[0] * max(self.current_step, 1))
            t2 = round(cfg.piecewise_phase_ratios[1] * max(self.current_step, 1))
            c_val = _piecewise_decay(
                t,
                cfg.exploration_constant,
                t1,
                t2,
                cfg.piecewise_alpha,
                cfg.lower_bound,
            )
        elif cfg.decay_type == "dynamic_piecewise":
            c_val = _dynamic_piecewise_decay(
                steps_limit=max(self.current_step, 1),
                n_nodes=self.current_step,
                initial_c=cfg.exploration_constant,
                start_time=self.search_start_time,
                time_limit=self.time_limit or 1e6,
                alpha=cfg.dynamic_alpha,
                lower_bound=cfg.lower_bound,
                phase_ratios=cfg.dynamic_phase_ratios,
            )
        else:
            c_val = cfg.exploration_constant

        logger.debug(f"Exploration constant chosen: {c_val:.4f} (decay_type={cfg.decay_type} step={self.current_step})")
        return c_val
