#!/usr/bin/env python3
"""Unit tests for ml_master_warmup metric flow."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

warmup_root = project_root / "playground" / "ml_master_warmup"
if str(warmup_root) not in sys.path:
    sys.path.insert(0, str(warmup_root))

from core.exp.debug_exp import DebugExp
from core.exp.draft_exp import DraftExp
from core.exp.improve_exp import ImproveExp
from core.utils.playground_helpers import _serialize_node, build_review


class _FakeAgent:
    def __init__(self):
        self._prompt_format_kwargs = {}

    def run(self, _task):
        return SimpleNamespace(dialogs=[])


class TestWarmupMetricPipeline(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(".")
        self.node = SimpleNamespace(id="n1")
        self.agent = _FakeAgent()

    @patch("core.exp.draft_exp.run_code_via_bash", return_value={"stdout": "Validation AUC: 0.91", "exit_code": 0})
    @patch("core.exp.draft_exp.extract_python_code", return_value="print('ok')")
    @patch("core.exp.draft_exp.extract_text_up_to_code", return_value="plan")
    @patch("core.exp.draft_exp.extract_agent_response", return_value="raw")
    @patch.object(DraftExp, "_run_metric_agent", return_value={"metric": 0.91, "is_bug": False, "has_submission": True})
    def test_draft_exp_propagates_metric(self, *_mocks):
        exp = DraftExp(self.agent, metric_agent=None, session=None, workspace=self.workspace, exp_id=None, data_preview="", node=self.node, exp_index=0)
        result = exp.run(task_description="t", memory="")
        self.assertEqual(result["metric"], 0.91)
        self.assertEqual(result["metric_detail"]["metric"], 0.91)

    @patch("core.exp.debug_exp.run_code_via_bash", return_value={"stdout": "Validation AUC: 0.87", "exit_code": 0})
    @patch("core.exp.debug_exp.extract_python_code", return_value="print('ok')")
    @patch("core.exp.debug_exp.extract_agent_response", return_value="raw")
    @patch.object(DebugExp, "_run_metric_agent", return_value={"metric": 0.87, "is_bug": False, "has_submission": True})
    def test_debug_exp_propagates_metric(self, *_mocks):
        exp = DebugExp(self.agent, metric_agent=None, session=None, workspace=self.workspace, exp_id=None, data_preview="", node=self.node, exp_index=0)
        result = exp.run(task_description="t", prev_code="x", term_out="y", issue="z")
        self.assertEqual(result["metric"], 0.87)
        self.assertEqual(result["metric_detail"]["metric"], 0.87)

    @patch("core.exp.improve_exp.run_code_via_bash", return_value={"stdout": "Validation AUC: 0.93", "exit_code": 0})
    @patch("core.exp.improve_exp.extract_python_code", return_value="print('ok')")
    @patch("core.exp.improve_exp.extract_text_up_to_code", return_value="plan")
    @patch("core.exp.improve_exp.extract_agent_response", return_value="raw")
    @patch.object(ImproveExp, "_run_metric_agent", return_value={"metric": 0.93, "is_bug": False, "has_submission": True})
    def test_improve_exp_propagates_metric(self, *_mocks):
        exp = ImproveExp(self.agent, metric_agent=None, session=None, workspace=self.workspace, exp_id=None, data_preview="", node=self.node, exp_index=0)
        result = exp.run(task_description="t", best_code="x", best_metric=0.8, memory="", term_out="")
        self.assertEqual(result["metric"], 0.93)
        self.assertEqual(result["metric_detail"]["metric"], 0.93)

    def test_snapshot_serialization_keeps_metric_and_submission_score(self):
        metric_obj = SimpleNamespace(value=0.95, maximize=True)
        node = SimpleNamespace(
            id="n2",
            stage="improve",
            action_type="improve",
            parent=None,
            plan="",
            code="",
            stdout="",
            exit_code=0,
            is_buggy=False,
            is_valid=True,
            is_terminal=False,
            visits=1,
            total_reward=0.0,
            metric=metric_obj,
            data_ok=None,
            children=[],
            uct_value=lambda *_args, **_kwargs: 0.1,
        )
        search_mgr = SimpleNamespace(_exploration_constant=lambda: 1.0)

        payload = _serialize_node(node, search_mgr)
        self.assertEqual(payload["metric"], 0.95)
        self.assertEqual(payload["submission_score"], 0.95)

    def test_build_review_keeps_numeric_metric_even_if_metric_detail_is_bug_true(self):
        res = {
            "metric": 0.88,
            "metric_detail": {"is_bug": True, "has_submission": True},
            "exec": {"stdout": "Validation AUC: 0.88"},
        }
        review = build_review(res, has_submission=True)
        self.assertEqual(review.metric, 0.88)
        self.assertFalse(review.is_bug)

    def test_build_review_marks_bug_for_non_numeric_metric(self):
        res = {
            "metric": "metric",
            "metric_detail": {"is_bug": False, "has_submission": True},
            "exec": {"stdout": "Validation AUC: 0.88"},
        }
        review = build_review(res, has_submission=True)
        self.assertIsNone(review.metric)
        self.assertTrue(review.is_bug)


if __name__ == "__main__":
    unittest.main()

