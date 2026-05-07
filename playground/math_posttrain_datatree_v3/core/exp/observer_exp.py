from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from evomaster.agent import BaseAgent
from evomaster.core.exp import extract_agent_response
from evomaster.utils.types import TaskInstance

from ..utils.io import json_dumps_safe, write_json
from . import NodeExp

logger = logging.getLogger(__name__)

RESPONSE_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def get_global_advice_path(task_workspace: Path) -> Path:
    return task_workspace / "artifacts" / "reports" / "global_advice.json"


def get_observer_history_path(task_workspace: Path) -> Path:
    return task_workspace / "artifacts" / "reports" / "observer_history.jsonl"


def _extract_json_payload(text: str) -> dict[str, Any]:
    if not text:
        return {}
    match = RESPONSE_JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1)
    text = text.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


class ObserverExp(NodeExp):
    def __init__(
        self,
        agent,
        session,
        workspace: Path,
        task_workspace: Path,
        config,
        node,
        global_advice_path: Path,
        observer_history_path: Path,
        exp_index: int = 0,
    ):
        super().__init__(agent, session, workspace, task_workspace, config, node, exp_index)
        self.global_advice_path = global_advice_path
        self.observer_history_path = observer_history_path

    def _fallback_payload(self, reason: str, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "selected_next_node_id": "",
            "scheduler_reason": reason,
            "global_strategy": "Use the existing UCT scheduler until observer advice is available.",
            "node_advice": {},
            "red_advice": "",
            "black_advice": "",
            "latest_completed_node_id": context.get("latest_completed_node_id") or "",
            "generated_at": time.time(),
        }

    def _normalize_payload(self, payload: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        pending_ids = {
            str(item.get("id"))
            for item in context.get("pending_nodes", [])
            if isinstance(item, dict) and item.get("id")
        }
        selected = str(payload.get("selected_next_node_id") or "").strip()
        if selected and selected not in pending_ids:
            selected = ""

        normalized = {
            "schema_version": 1,
            "selected_next_node_id": selected,
            "scheduler_reason": str(payload.get("scheduler_reason") or "").strip(),
            "global_strategy": str(payload.get("global_strategy") or "").strip(),
            "node_advice": _safe_dict(payload.get("node_advice")),
            "red_advice": str(payload.get("red_advice") or "").strip(),
            "black_advice": str(payload.get("black_advice") or "").strip(),
            "latest_completed_node_id": context.get("latest_completed_node_id") or "",
            "generated_at": time.time(),
        }
        if not normalized["scheduler_reason"]:
            normalized["scheduler_reason"] = "Observer did not provide a scheduler rationale."
        if not normalized["global_strategy"]:
            normalized["global_strategy"] = "Continue with UCT-guided exploration."
        return normalized

    def _write_outputs(self, payload: dict[str, Any], context: dict[str, Any], raw_response: str) -> None:
        output = dict(payload)
        output["observer_context_summary"] = {
            "latest_completed_node_id": context.get("latest_completed_node_id") or "",
            "latest_completed_stage": context.get("latest_completed_stage") or "",
            "pending_node_count": len(context.get("pending_nodes", []) or []),
            "current_step": context.get("current_step"),
            "best_metric": context.get("best_metric"),
        }
        output["raw_response"] = raw_response
        write_json(self.global_advice_path, output)

        self.observer_history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.observer_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json_dumps_safe(output, ensure_ascii=False) + "\n")

    def run(self, task_description: str, observer_context: dict[str, Any]) -> dict[str, Any]:
        node_id = getattr(self.node, "id", "observer")
        BaseAgent.set_exp_info(exp_name=f"math_observer_{node_id[:8]}", exp_index=self.exp_index)
        raw_response = ""
        if self.agent is None:
            payload = self._fallback_payload("observer_agent is not configured", observer_context)
            self._write_outputs(payload, observer_context, raw_response)
            return payload

        orig_fmt = self.agent._prompt_format_kwargs.copy()
        self.agent._prompt_format_kwargs.update(
            {
                "task_description": task_description,
                "workspace": str(self.workspace),
                "task_workspace": str(self.task_workspace),
                "global_advice_path": str(self.global_advice_path),
                "observer_history_path": str(self.observer_history_path),
                "observer_context_json": json.dumps(observer_context, ensure_ascii=False, indent=2),
            }
        )
        try:
            task = TaskInstance(
                task_id=f"{node_id}_observer",
                task_type="observer",
                description=task_description,
                input_data={},
            )
            traj = self.agent.run(
                task,
                enable_final_turn=True,
                enable_final_turn_prompt=(
                    "FINAL TURN. Output exactly one JSON object matching the required schema. "
                    "Do not include prose outside the JSON object."
                ),
            )
            raw_response = extract_agent_response(traj)
            payload = _extract_json_payload(raw_response)
            if not payload:
                payload = self._fallback_payload("observer response did not contain valid JSON", observer_context)
            else:
                payload = self._normalize_payload(payload, observer_context)
        except Exception as exc:
            logger.warning("Observer agent execution failed: %s", exc, exc_info=True)
            payload = self._fallback_payload(f"observer execution failed: {exc}", observer_context)
        finally:
            self.agent._prompt_format_kwargs = orig_fmt

        self._write_outputs(payload, observer_context, raw_response)
        return payload
