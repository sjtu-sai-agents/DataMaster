"""Data search experiment for MCTS V1."""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI
from evomaster.agent import BaseAgent
from evomaster.utils.types import TaskInstance

from . import NodeExp
from ..utils.runtime import extract_agent_response

logger = logging.getLogger(__name__)


class DataSearchExp(NodeExp):
    """Run a data search attempt and return data context for downstream improve."""

    DATA_DIRECTION_NAMES = [
        "Data Enhancement",
        "External Data",
        "Data Augmentation",
        "Additional Data",
    ]

    _JUDGE_SYSTEM_PROMPT = """You are a binary classification expert. Your task is to determine whether a given improvement direction for a machine learning task is a "Data Enhancement" direction.

Respond with ONLY "YES" or "NO"."""

    def __init__(
        self,
        agent,
        metric_agent,
        session,
        workspace,
        exp_id: str | None,
        data_preview: str,
        node,
        exp_index: int = 0,
    ):
        super().__init__(agent, metric_agent, session, workspace, exp_id, data_preview, node, exp_index)
        self._direction_cache: dict[str, bool] = {}

    def is_data_direction(self, direction_name: str) -> bool:
        if direction_name in self._direction_cache:
            return self._direction_cache[direction_name]

        lowered = direction_name.lower()
        for name in self.DATA_DIRECTION_NAMES:
            if name.lower() in lowered:
                self._direction_cache[direction_name] = True
                return True

        try:
            judged = self._llm_judge_direction(direction_name)
        except Exception as exc:
            logger.warning("Data direction judge failed for %s: %s", direction_name, exc)
            judged = any(
                token in lowered
                for token in ("data", "dataset", "external", "augment", "additional", "merge")
            )
        self._direction_cache[direction_name] = judged
        return judged

    def _llm_judge_direction(self, direction_name: str) -> bool:
        judge_cfg = getattr(self.session.config, "data_judge", {}) or {}
        base_url = judge_cfg.get("base_url", "http://localhost:8899/v1")
        model = judge_cfg.get("model", "gpt-oss")
        api_key = judge_cfg.get("api_key", "EMPTY")

        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self._JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Analyze whether the following direction is data enhancement. "
                        f'Direction: "{direction_name}". Reply YES or NO.'
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=20,
        )
        content = (response.choices[0].message.content or "").strip().upper()
        return "YES" in content or "TRUE" in content

    def run(
        self,
        task_description: str,
        best_code: str,
        memory: str,
        task_id: str = "exp_001",
    ) -> dict[str, Any]:
        node_id = self.node.id
        BaseAgent.set_exp_info(exp_name=f"data_search_{node_id[:8]}", exp_index=self.exp_index)
        fmt = {
            "task_description": task_description,
            "data_preview": self.data_preview,
            "best_code": best_code,
            "memory": memory,
            "workspace": str(self.workspace),
        }
        original = self.agent._prompt_format_kwargs.copy()
        self.agent._prompt_format_kwargs.update(fmt)
        try:
            task = TaskInstance(
                task_id=f"{node_id}_data_search",
                task_type="data_search",
                description="search external data for current node",
                input_data={},
            )
            trajectory = self.agent.run(task)
            response = extract_agent_response(trajectory)
        finally:
            self.agent._prompt_format_kwargs = original

        lowered = (response or "").lower()
        data_ok = bool(response) and "no_dataset_found" not in lowered and "no dataset found" not in lowered
        summary = (response or "").strip()
        if len(summary) > 1000:
            summary = summary[:1000]

        return {
            "plan": "data_search",
            "code": best_code,
            "raw_response": response,
            "exec": {"stdout": summary, "exit_code": 0 if data_ok else 1},
            "metric": None,
            "metric_detail": {"is_bug": not data_ok, "has_submission": False},
            "data_context": response if data_ok else "",
            "data_ok": data_ok,
        }

