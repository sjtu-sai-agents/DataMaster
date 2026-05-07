"""Metric value and comparison utilities for ML-Master"""

import math
from dataclasses import dataclass, field
from typing import Any, override


@dataclass(frozen=True)
class MetricValue:
    """Represents a metric value with comparison semantics.

    Args:
        value: The numeric metric value
        maximize: If True, higher values are better. If False, lower values are better.
    """

    value: float | None
    maximize: bool = True

    def __float__(self) -> float:
        if self.value is None:
            return -math.inf if self.maximize else math.inf
        return float(self.value)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, MetricValue):
            return NotImplemented
        return float(self) < float(other)

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, MetricValue):
            return NotImplemented
        return float(self) <= float(other)

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, MetricValue):
            return NotImplemented
        return float(self) > float(other)

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, MetricValue):
            return NotImplemented
        return float(self) >= float(other)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MetricValue):
            return NotImplemented
        return float(self) == float(other)

    def __hash__(self) -> int:
        return hash((self.value, self.maximize))

    def __repr__(self) -> str:
        if self.value is None:
            return f"MetricValue(None, maximize={self.maximize})"
        return f"MetricValue({self.value:.4f}, maximize={self.maximize})"


@dataclass(frozen=True)
class WorstMetricValue(MetricValue):
    """Represents the worst possible metric value."""

    def __init__(self, maximize: bool = True):
        # Use a very small value for maximization, very large for minimization
        if maximize:
            value = -1e9
        else:
            value = 1e9
        super().__init__(value=value, maximize=maximize)


def get_worst_metric(maximize: bool = True) -> MetricValue:
    """Get the worst possible metric value.

    Args:
        maximize: If True, higher values are better. If False, lower values are better.

    Returns:
        The worst possible MetricValue
    """
    return WorstMetricValue(maximize=maximize)
