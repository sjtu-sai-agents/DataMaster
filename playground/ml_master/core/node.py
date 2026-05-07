"""Node class for ML-Master solution tree"""

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from .metric import MetricValue, get_worst_metric


@dataclass(eq=False)
class Node:
    """A single node in the solution tree. Contains code, execution results, and evaluation information."""

    # ---- code & plan ----
    code: str
    plan: str = field(default=None, kw_only=True)

    # ---- general attrs ----
    step: int = field(default=None, kw_only=True)
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["Node"] = field(default=None, kw_only=True)
    children: set["Node"] = field(default_factory=set, kw_only=True)

    # ---- execution info ----
    _term_out: list[str] = field(default_factory=list, kw_only=True)
    exec_time: float = field(default=None, kw_only=True)
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    analysis: str = field(default=None, kw_only=True)
    metric: MetricValue = field(default_factory=lambda: get_worst_metric(True), kw_only=True)
    # whether the agent decided that the code is buggy
    # -> always True if exc_type is not None or no valid metric
    is_buggy: bool = field(default=None, kw_only=True)
    is_valid: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """
        Return the stage of the node:
        - "draft" if the node is an initial solution draft
        - "debug" if the node is the result of a debugging step
        - "improve" if the node is the result of an improvement step
        """
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution."""
        if not self._term_out:
            return ""
        return "".join(self._term_out)

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        return not self.children

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """
        Length of the current debug path
        - 0 if the node is not a debug node (parent is not buggy)
        - 1 if the parent is buggy but the skip parent isn't
        - n if there were n consecutive debugging steps
        """
        if self.stage_name != "debug":
            return 0
        if self.parent is None:
            return 0
        return self.parent.debug_depth + 1

    def absorb_exec_result(self, exec_result: "ExecutionResult") -> None:
        """Absorb the result of executing the code from this node.

        Args:
            exec_result: ExecutionResult containing terminal output, execution time, etc.
        """
        self._term_out = exec_result.term_out if hasattr(exec_result, 'term_out') else []
        self.exec_time = exec_result.exec_time if hasattr(exec_result, 'exec_time') else None
        self.exc_type = exec_result.exc_type if hasattr(exec_result, 'exc_type') else None
        self.exc_info = exec_result.exc_info if hasattr(exec_result, 'exc_info') else None
        self.exc_stack = exec_result.exc_stack if hasattr(exec_result, 'exc_stack') else None


@dataclass
class ExecutionResult:
    """Result of executing code in an interpreter."""

    term_out: list[str] = field(default_factory=list)
    exec_time: float = 0.0
    exc_type: str | None = None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None
    exit_code: int = 0
    node_id: str = ""

    def has_error(self) -> bool:
        """Check if execution resulted in an error."""
        return self.exc_type is not None or self.exit_code != 0
