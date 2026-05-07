"""MCTS Node implementation for ML-Master tree search"""

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from .metric import MetricValue, get_worst_metric
from .node import Node

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class MCTSNode(Node):
    """MCTS Node with UCT calculation and expansion strategies.

    This extends the base Node class with MCTS-specific attributes:
    - visits: Number of times this node has been visited
    - total_reward: Cumulative reward from this node
    - stage: One of "root", "draft", "improve", "debug"
    - local_best_node: Best node in the subtree rooted at this node
    """

    visits: int = field(default=0, kw_only=True)
    total_reward: float = field(default=0.0, kw_only=True)
    is_terminal: bool = field(default=False, kw_only=True)

    # MCTS-specific attributes
    local_best_node: Optional["MCTSNode"] = field(default=None, kw_only=True)
    is_debug_success: bool = field(default=False, kw_only=True)
    continue_improve: bool = field(default=False, kw_only=True)
    stage: Literal["root", "improve", "debug", "draft"] = "draft"
    improve_failure_depth: int = field(default=0, kw_only=True)
    lock: bool = field(default=False, kw_only=True)
    expected_child_count: int = field(default=0, kw_only=True)
    finish_time: str = field(default=None, kw_only=True)
    _uct: float = field(default=0.0, kw_only=True)

    # Thread-safe lock for child count operations
    _child_count_lock: threading.Lock = field(default_factory=threading.Lock, kw_only=True, compare=False)

    def __post_init__(self):
        super().__post_init__()
        if self.stage not in ["root", "improve", "debug", "draft"]:
            raise ValueError(f"Invalid stage: {self.stage}")

    def uct_value(self, exploration_constant: float = 1.414) -> float:
        """Calculate the UCT (Upper Confidence Bound for Trees) value.

        UCT = Q + c * sqrt(ln(N) / n), where:
        - Q = total_reward / visits (average reward)
        - c = exploration_constant
        - N = parent_visits (number of visits to the parent node)
        - n = visits (number of visits to the current node)

        Args:
            exploration_constant: The exploration constant (c in UCT formula)

        Returns:
            The UCT value. Unvisited nodes return infinity.
        """
        parent_visits = 0
        if self.parent:
            parent_visits = self.parent.visits

        if self.visits == 0:
            return float('inf')

        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * (math.log(max(1, parent_visits)) / self.visits) ** 0.5
        self._uct = exploitation + exploration
        return self._uct

    def is_fully_expanded(self, scfg: "SearchConfig") -> bool:
        """Check if this node is fully expanded based on stage-specific limits.

        Different expansion strategies for different node types:
        - Draft node: num_drafts children (default 5)
        - Bug node: Stop after getting a non-buggy child, max num_bugs (default 3)
        - Others: num_improves children (default 3)

        Args:
            scfg: Search configuration with expansion limits

        Returns:
            True if the node is fully expanded
        """
        if self.step == 0:
            return self.num_children >= scfg.num_drafts
        else:
            if self.is_buggy:
                if self.has_no_bug_child():
                    return True
                else:
                    return self.num_children >= scfg.num_bugs
            else:
                return self.num_children >= scfg.num_improves

    def is_fully_expanded_with_expected(self, scfg: "SearchConfig") -> bool:
        """Check if this node is fully expanded using expected child count.

        This version uses expected_child_count instead of actual num_children,
        which is important for parallel search where children are being created concurrently.

        Args:
            scfg: Search configuration with expansion limits

        Returns:
            True if the node is fully expanded
        """
        with self._child_count_lock:
            if self.step == 0:
                return self.expected_child_count >= scfg.num_drafts
            else:
                if self.is_buggy:
                    if self.has_no_bug_child():
                        return True
                    else:
                        return self.expected_child_count >= scfg.num_bugs
                else:
                    return self.expected_child_count >= scfg.num_improves

    def has_no_bug_child(self) -> bool:
        """Check if this node has any non-buggy children.

        Returns:
            True if at least one child is not buggy
        """
        for child in self.children:
            if not child.is_buggy:
                return True
        return False

    @property
    def num_children(self) -> int:
        """Get the number of children."""
        return len(self.children)

    def update(self, result: float, add: bool = True) -> None:
        """Update node statistics after a simulation.

        Args:
            result: The reward to add
            add: If True, increment visits and add reward. If False, just update.
        """
        if add:
            self.visits += 1
            self.total_reward += result

    def fetch_child_memory(self, include_code: bool = False) -> str:
        """Fetch memory from children nodes for agent context.

        Args:
            include_code: If True, include code in the memory

        Returns:
            A formatted string containing memory from all children
        """
        logger.debug("fetch_child_memory")
        summary = []
        for n in self.children:
            if n.is_buggy is not None:
                summary_part = f"Design: {n.plan}\n"
                if include_code:
                    summary_part += f"Code: {n.code}\n"
                if n.is_buggy:
                    summary_part += "Results: The implementation of this design has bugs.\n"
                    summary_part += "Insight: Using a different approach may not result in the same bugs.\n"
                else:
                    if n.analysis:
                        summary_part += f"Results: {n.analysis}\n"
                    if n.metric and n.metric.value is not None:
                        summary_part += f"Validation Metric: {n.metric.value}\n"
                summary.append(summary_part)

        if not summary:
            summary.append("There is no previous memory")
        return "\n-------------------------------\n".join(summary)

    def fetch_parent_memory(self, include_code: bool = False) -> str:
        """Fetch memory from parent node for agent context.

        Args:
            include_code: If True, include code in the memory

        Returns:
            A formatted string containing memory from parent
        """
        logger.debug("fetch_parent_memory")
        if self.parent is not None and self.parent.is_buggy is not None and not self.parent.is_buggy:
            summary = []
            summary_part = f"Design: {self.parent.plan}\n"
            if include_code:
                summary_part += f"Code: {self.parent.code}\n"
            if self.parent.analysis:
                summary_part += f"Results: {self.parent.analysis}\n"
            if self.parent.metric and self.parent.metric.value is not None:
                summary_part += f"Validation Metric: {self.parent.metric.value}\n"
            summary.append(summary_part)
            return "\n-------------------------------\n".join(summary)
        return "There is no parent memory"

    def add_expected_child_count(self) -> None:
        """Atomically increment expected child count."""
        with self._child_count_lock:
            self.expected_child_count += 1
            logger.debug(f"Node {self.id} expected_child_count is now {self.expected_child_count}")

    def sub_expected_child_count(self) -> None:
        """Atomically decrement expected child count."""
        with self._child_count_lock:
            self.expected_child_count -= 1
            logger.debug(f"Node {self.id} expected_child_count is now {self.expected_child_count}")

    def __getstate__(self) -> dict:
        """Prepare for pickling by removing unpicklable lock."""
        state = self.__dict__.copy()
        state.pop('_child_count_lock', None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore from pickle by recreating lock."""
        self.__dict__.update(state)
        self._child_count_lock = threading.Lock()


@dataclass
class SearchConfig:
    """Configuration for MCTS search behavior."""

    max_debug_depth: int = 3
    debug_prob: float = 0.0
    num_drafts: int = 5
    invalid_metric_upper_bound: int = 50
    metric_improvement_threshold: float = 0.001
    back_debug_depth: int = 1
    num_bugs: int = 3
    num_improves: int = 3
    max_improve_failure: int = 2
    parallel_search_num: int = 3
    exploration_constant: float = 1.414
