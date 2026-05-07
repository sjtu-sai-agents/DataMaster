"""Memory system data type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


MemoryCategory = Literal["preference", "fact", "decision", "entity", "other"]


@dataclass
class MemoryEntry:
    """A single memory record."""

    id: str
    user_id: str
    content: str
    category: MemoryCategory = "other"
    importance: float = 0.5
    source: str = "auto"  # auto / manual / compaction
    created_at: float = 0.0
    updated_at: float = 0.0
    access_count: int = 0
    score: float = 0.0  # Match score during search (not persisted)

    # Category display names (Chinese)
    _CATEGORY_LABELS: dict[str, str] = field(
        default_factory=lambda: {
            "preference": "偏好",
            "fact": "事实",
            "decision": "决策",
            "entity": "实体",
            "other": "其他",
        },
        init=False,
        repr=False,
    )

    @property
    def category_label(self) -> str:
        """Get the display label for the memory category."""
        return self._CATEGORY_LABELS.get(self.category, self.category)
