from __future__ import annotations

import heapq
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricReview:
    metric: Optional[float]
    lower_is_better: Optional[bool] = None
    maximize: bool = True
    is_bug: bool = False
    has_submission: bool = False
    summary: str = ""
    raw_output: Optional[str] = None

    def __post_init__(self) -> None:
        if self.lower_is_better is not None:
            self.maximize = not self.lower_is_better
        if self.metric is not None:
            self.metric = float(self.metric)


@dataclass
class MetricValue:
    value: Optional[float]
    maximize: bool = True

    def __gt__(self, other: "MetricValue") -> bool:
        if self.value is None:
            return False
        if other.value is None:
            return True
        if self.value == other.value:
            return False
        return self.value > other.value if self.maximize else self.value < other.value


@dataclass
class UCTSearchConfig:
    num_red: int = 1
    num_black: int = 2
    max_black_per_red: int = 3
    metric_improvement_threshold: float = 0.0001
    exploration_constant: float = 1.414
    max_rounds: int = 8


@dataclass(eq=False)
class UCTNode:
    stage: str
    plan: str = ""
    code: str = ""
    parent: Optional["UCTNode"] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    stdout: str | None = None
    exit_code: int | None = None
    metric: MetricValue = field(default_factory=lambda: MetricValue(None, True))
    is_buggy: Optional[bool] = None
    visits: int = 0
    total_reward: float = 0.0
    children: set["UCTNode"] = field(default_factory=set)
    expected_child_count: int = 0
    is_terminal: bool = False

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    def __hash__(self) -> int:
        return hash(self.id)

    def expect_child(self) -> None:
        self.expected_child_count += 1

    def complete_child(self) -> None:
        self.expected_child_count = max(self.expected_child_count - 1, 0)

    def update_reward(self, reward: float) -> None:
        self.visits += 1
        self.total_reward += reward

    def uct_value(self, exploration_constant: float, parent_visits: int) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(math.log(max(parent_visits, 1)) / self.visits)
        return exploitation + exploration

    def fetch_child_memory(self) -> str:
        parts: list[str] = []
        for child in sorted(self.children, key=lambda item: item.created_at):
            if child.is_buggy is None:
                continue
            section = [
                f"stage={child.stage} id={child.id[:8]} metric={child.metric.value} buggy={child.is_buggy}"
            ]
            plan = getattr(child, "plan", "")
            if plan:
                section.append(f"plan={plan}")
            analysis = getattr(child, "analysis", "")
            if analysis:
                compact = str(analysis).strip().replace("\n", " ")
                if len(compact) > 220:
                    compact = compact[:217].rstrip() + "..."
                section.append(f"summary={compact}")
            next_action = getattr(child, "recommended_next_action", None)
            if next_action:
                section.append(f"next={next_action}")
            parts.append(" | ".join(section))
        return "\n".join(parts) if parts else "There is no previous memory"


class UCTSearchManager:
    def __init__(self, search_cfg: UCTSearchConfig):
        self.search_cfg = search_cfg
        self.root = UCTNode(stage="root", plan="virtual root", code="")
        self.best_node: UCTNode | None = None
        self.best_metric: float | None = None
        self.current_step = 0
        self._counter = 0
        self._heap: list[tuple[float, int, UCTNode]] = []
        self.nodes_by_id: dict[str, UCTNode] = {self.root.id: self.root}

    def _exploration_constant(self) -> float:
        return self.search_cfg.exploration_constant

    def create_child(self, parent: UCTNode, stage: str, plan: str = "", code: str = "") -> UCTNode:
        parent.expect_child()
        node = UCTNode(stage=stage, plan=plan, code=code, parent=parent)
        self.nodes_by_id[node.id] = node
        return node

    def _ancestor_best_metric(self, node: UCTNode) -> float:
        best = 0.0
        cursor = node.parent
        while cursor is not None:
            val = cursor.metric.value if cursor.metric else None
            if val is not None and val > best:
                best = val
            cursor = cursor.parent
        return best

    def push_execution_node(self, node: UCTNode, priority: float | None = None) -> None:
        if priority is None:
            priority = self._ancestor_best_metric(node)
            if node.stage == "seed":
                priority = 1.0
            elif node.stage == "red":
                priority += 0.10
        self._counter += 1
        heapq.heappush(self._heap, (-priority, self._counter, node))

    def pop_execution_node(self) -> UCTNode | None:
        if not self._heap:
            return None
        _, _, node = heapq.heappop(self._heap)
        return node

    def count_black_nodes_for_red(self, bound_red_node_id: str | None) -> int:
        if not bound_red_node_id:
            return 0
        return sum(
            1
            for node in self.nodes_by_id.values()
            if node.stage == "black" and getattr(node, "bound_red_node_id", None) == bound_red_node_id
        )

    def ingest_result(self, node: UCTNode, review: MetricReview) -> float:
        node.is_buggy = review.is_bug
        node.metric = MetricValue(review.metric, review.maximize)
        reward = 0.01 if node.stage == "red" and not review.is_bug else float(review.metric or 0.0)
        cursor: UCTNode | None = node
        while cursor is not None:
            cursor.update_reward(reward)
            cursor = cursor.parent
        if node.parent is not None:
            node.parent.complete_child()
        self.current_step += 1
        if not review.is_bug and review.metric is not None:
            if self.best_metric is None or review.metric > self.best_metric:
                self.best_metric = review.metric
                self.best_node = node
        return reward
