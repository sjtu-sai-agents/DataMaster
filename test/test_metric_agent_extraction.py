#!/usr/bin/env python3
"""Test metric extraction through NodeExp._run_metric_agent."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

warmup_root = project_root / "playground" / "ml_master_warmup"
if str(warmup_root) not in sys.path:
    sys.path.insert(0, str(warmup_root))

from core.exp import NodeExp


class FakeMetricAgent:
    def __init__(self, response_text: str):
        self._prompt_format_kwargs: dict = {}
        self._response_text = response_text

    def run(self, _task):
        assistant_msg = SimpleNamespace(
            role=SimpleNamespace(value="assistant"),
            content=self._response_text,
        )
        dialog = SimpleNamespace(messages=[assistant_msg])
        return SimpleNamespace(dialogs=[dialog])


class TestMetricAgentExtraction(unittest.TestCase):
    def test_run_metric_agent_extracts_auc_from_trajectory(self):
        metric_agent = FakeMetricAgent("Validation AUC: 0.89566")
        exp = NodeExp(
            agent=None,
            metric_agent=metric_agent,
            session=None,
            workspace=Path("."),
            exp_id=None,
            data_preview="",
            node=None,
            exp_index=0,
        )

        result = exp._run_metric_agent(code="print('x')", stdout="unused")

        self.assertAlmostEqual(result["metric"], 0.89566, places=5)
        self.assertFalse(result["is_bug"])


if __name__ == "__main__":
    unittest.main()

