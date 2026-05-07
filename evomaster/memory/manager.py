"""Memory manager -- business logic layer.

Responsible for auto-capture, auto-recall, and the tool API.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from .store import MemoryStore
from .types import MemoryCategory, MemoryEntry

if TYPE_CHECKING:
    from evomaster.utils.llm import BaseLLM

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Auto-capture rules
# ------------------------------------------------------------------

CAPTURE_PATTERNS: list[tuple[re.Pattern, MemoryCategory]] = [
    # Chinese -- explicit memory instructions
    (re.compile(r"记住[：:\s].+", re.S), "fact"),
    (re.compile(r"请?记住.+"), "fact"),
    # Chinese -- preference
    (re.compile(r"我(喜欢|偏好|习惯|倾向于|一般用|经常用|更喜欢)"), "preference"),
    (re.compile(r"以后(都|总是|一直|请|帮我)"), "preference"),
    (re.compile(r"(每次|下次|将来)(都|请|帮我)"), "preference"),
    (re.compile(r"不要(再|总是|每次)"), "preference"),
    # Chinese -- fact
    (re.compile(r"我的.{1,15}(是|叫|为|用的是)"), "fact"),
    (re.compile(r"我(是|叫|在|住在|来自|从事|负责|管理)"), "fact"),
    # Chinese -- decision
    (re.compile(r"我(决定|选择|确定|打算)(了|要)?"), "decision"),
    # English
    (re.compile(r"(?i)\bremember\b.+"), "fact"),
    (re.compile(r"(?i)\bi (prefer|like|love|hate|always|never|usually)\b"), "preference"),
    (re.compile(r"(?i)\bmy .{1,30} is\b"), "fact"),
    (re.compile(r"(?i)\bi('m| am| work| live)\b"), "fact"),
    (re.compile(r"(?i)\bi (decided|chose|will use|want to use)\b"), "decision"),
]

# Messages that are too short or too long are not extracted
_MIN_CAPTURE_LEN = 6
_MAX_CAPTURE_LEN = 500

# LLM extraction prompt
_EXTRACT_PROMPT = """\
从以下对话内容中提取值得长期记忆的关键信息。

只提取以下类型的信息：
- preference: 用户的偏好、喜好、习惯
- fact: 关于用户的事实信息（姓名、职业、项目、技术栈等）
- decision: 用户做出的重要决策
- entity: 重要的实体名称（项目名、公司名等）

如果没有值得记忆的内容，返回空数组。

对话内容：
{content}

以 JSON 数组格式返回，每条记忆包含 content、category、importance(0-1)：
```json
[{{"content": "...", "category": "preference", "importance": 0.7}}]
```"""


class MemoryManager:
    """Memory manager.

    Args:
        store: Underlying storage instance.
        llm: Optional LLM instance, used for capture_with_llm mode.
        config: Memory configuration dictionary.
    """

    def __init__(
        self,
        store: MemoryStore,
        llm: BaseLLM | None = None,
        config: dict[str, Any] | None = None,
    ):
        self._store = store
        self._llm = llm
        self._config = config or {}

    @property
    def store(self) -> MemoryStore:
        """Get the underlying memory store."""
        return self._store

    # ------------------------------------------------------------------
    # Auto-recall
    # ------------------------------------------------------------------

    def recall_for_context(
        self, user_id: str, query: str, limit: int = 5
    ) -> str:
        """Search for relevant memories based on the user message and return a formatted context text block.

        If the user has no memories, returns an empty string.
        """
        entries = self._store.search(user_id, query, limit=limit)
        if not entries:
            return ""

        lines = ["## 用户记忆", "", "以下是关于当前用户的历史记忆，仅供参考："]
        for e in entries:
            lines.append(f"- [{e.category_label}] {e.content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Auto-capture (rule matching)
    # ------------------------------------------------------------------

    def extract_from_message(self, user_id: str, message: str) -> list[str]:
        """Extract memorable content from a user message (rule matching).

        Returns a list of newly added memory contents.
        """
        message = message.strip()
        if len(message) < _MIN_CAPTURE_LEN or len(message) > _MAX_CAPTURE_LEN:
            return []

        saved: list[str] = []
        for pattern, category in CAPTURE_PATTERNS:
            if pattern.search(message):
                memory_id = self._store.add(
                    user_id=user_id,
                    content=message,
                    category=category,
                    importance=0.6,
                    source="auto",
                )
                if memory_id:
                    saved.append(message)
                    self._enforce_user_limit(user_id)
                break  # A single message matches at most once

        return saved

    # ------------------------------------------------------------------
    # Auto-capture (LLM extraction)
    # ------------------------------------------------------------------

    def extract_from_summary(self, user_id: str, summary: str) -> list[str]:
        """Extract key facts/preferences/decisions from a compaction summary using LLM.

        Falls back to rule matching if LLM is not available.
        """
        if not self._llm:
            return self.extract_from_message(user_id, summary)

        try:
            from evomaster.utils.types import Dialog, UserMessage

            prompt = _EXTRACT_PROMPT.format(content=summary[:4000])
            dialog = Dialog(messages=[UserMessage(content=prompt)], tools=[])
            response = self._llm.query(dialog)
            raw = response.content or ""

            items = self._parse_json_array(raw)
            saved: list[str] = []
            for item in items[:5]:  # Extract at most 5 items
                content = item.get("content", "").strip()
                category = item.get("category", "other")
                importance = float(item.get("importance", 0.5))
                if not content or len(content) < _MIN_CAPTURE_LEN:
                    continue
                if category not in ("preference", "fact", "decision", "entity", "other"):
                    category = "other"
                memory_id = self._store.add(
                    user_id=user_id,
                    content=content,
                    category=category,
                    importance=min(max(importance, 0.0), 1.0),
                    source="compaction",
                )
                if memory_id:
                    saved.append(content)

            self._enforce_user_limit(user_id)
            return saved

        except Exception:
            logger.exception("LLM memory extraction failed")
            return []

    # ------------------------------------------------------------------
    # Tool API
    # ------------------------------------------------------------------

    def search(self, user_id: str, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Search memories (called by the memory_search tool)."""
        return self._store.search(user_id, query, limit=limit)

    def save(
        self, user_id: str, content: str, category: str = "other"
    ) -> str | None:
        """Save a memory (called by the memory_save tool)."""
        memory_id = self._store.add(
            user_id=user_id,
            content=content,
            category=category,
            importance=0.8,  # Memories explicitly saved by the user/agent have higher importance
            source="manual",
        )
        self._enforce_user_limit(user_id)
        return memory_id

    def forget(
        self, user_id: str, query: str | None = None, memory_id: str | None = None
    ) -> str:
        """Delete memories (called by the memory_forget tool)."""
        if memory_id:
            ok = self._store.delete(memory_id)
            return f"已删除记忆 {memory_id}" if ok else f"未找到记忆 {memory_id}"
        if query:
            count = self._store.delete_by_query(user_id, query)
            return f"已删除 {count} 条匹配的记忆" if count > 0 else "未找到匹配的记忆"
        return "请提供 query 或 memory_id"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enforce_user_limit(self, user_id: str) -> None:
        """Enforce the per-user memory count limit, deleting the oldest excess entries."""
        max_count = self._config.get("max_memories_per_user", 500)
        deleted = self._store.enforce_limit(user_id, max_count)
        if deleted > 0:
            logger.info(
                "Enforced memory limit for user %s: deleted %d oldest",
                user_id, deleted,
            )

    @staticmethod
    def _parse_json_array(text: str) -> list[dict]:
        """Parse a JSON array from LLM output (tolerating markdown code blocks)."""
        # Remove markdown code blocks
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove leading and trailing ``` lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Try to extract the first JSON array
        match = re.search(r"\[.*\]", text, re.S)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        return []
