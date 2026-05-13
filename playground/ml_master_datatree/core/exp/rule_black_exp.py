"""Rule-based Black Experiment - No-op / Pass-through Black Node.

This file implements the rule_based black-node ablation.

Purpose:
    Keep Black nodes in the search tree, but remove their functional
    data-processing contribution.

Behavior:
    - Does NOT call the Black LLM agent.
    - Does NOT read or use Black prompts.
    - Does NOT generate new data.
    - Does NOT modify data_links.
    - Copies the parent template unchanged.
    - Copies the parent DataLoader unchanged.
    - Assembles and runs the inherited solution.
    - Uses the normal metric agent for evaluation.

This makes Black nodes structurally present but functionally inactive.
"""

import logging
import shutil
from pathlib import Path

from evomaster.agent import BaseAgent

from . import NodeExp
from ..utils.runtime import run_code_via_bash
from playground.search_dataset_tools.operate_submission._submission_utils import _assemble_code


logger = logging.getLogger(__name__)


class RuleBlackExp(NodeExp):
    """Rule-based Black experiment used as a no-op pass-through ablation.

    This class intentionally does not ask the agent to improve the DataLoader.
    It only preserves the parent implementation so that the search tree still
    contains Black nodes while Black-node data-processing ability is removed.
    """

    def __init__(
        self,
        agent,
        metric_agent,
        session,
        workspace: Path,
        exp_id: str | None,
        data_preview: str,
        node,
        exp_index: int = 0,
        test_feedback: bool = False,
        force_direction: str | None = None,
    ):
        super().__init__(
            agent,
            metric_agent,
            session,
            workspace,
            exp_id,
            data_preview,
            node,
            exp_index,
            test_feedback=test_feedback,
            force_direction=force_direction,
        )

    def run(
        self,
        task_description: str,
        prev_code: str,
        memory: str,
        term_out: str = "",
        best_code: str | None = None,
        best_metric: float | None = None,
    ) -> dict:
        """Run a no-op Black node.

        Args:
            task_description: Task description. Kept for API compatibility.
            prev_code: Parent node assembled code.
            memory: Child memory. Ignored in no-op mode.
            term_out: Parent execution output. Ignored in no-op mode.
            best_code: Global best code. Ignored in no-op mode.
            best_metric: Global best metric. Ignored in no-op mode.

        Returns:
            A dict with the same shape as BlackExp.run().
        """
        node_id = self.node.id
        parent_node = self.node.parent if self.node.parent else None
        parent_id = parent_node.id if parent_node else None

        BaseAgent.set_exp_info(
            exp_name=f"rule_black_{node_id[:8]}",
            exp_index=self.exp_index,
        )

        logger.info(
            "RuleBlack node %s, parent: %s",
            node_id[:8],
            parent_id[:8] if parent_id else "none",
        )

        if not parent_id:
            msg = "RuleBlackExp requires a parent node, but parent_id is None."
            logger.error(msg)
            return self._error_result(msg)

        parent_template_path = self.workspace / f"code_{parent_id}_template.py"
        parent_dataloader_path = self.workspace / f"code_{parent_id}_dataloader.py"

        child_template_path = self.workspace / f"code_{node_id}_template.py"
        child_dataloader_path = self.workspace / f"code_{node_id}_dataloader.py"

        if not parent_template_path.exists():
            msg = f"Parent template not found: {parent_template_path}"
            logger.error(msg)
            return self._error_result(msg)

        if not parent_dataloader_path.exists():
            msg = f"Parent dataloader not found: {parent_dataloader_path}"
            logger.error(msg)
            return self._error_result(msg)

        # Step 1: Copy parent template unchanged.
        shutil.copyfile(parent_template_path, child_template_path)
        logger.info(
            "RuleBlack copied parent template unchanged: %s -> %s",
            parent_template_path,
            child_template_path,
        )

        # Step 2: Copy parent DataLoader unchanged.
        shutil.copyfile(parent_dataloader_path, child_dataloader_path)
        logger.info(
            "RuleBlack copied parent dataloader unchanged: %s -> %s",
            parent_dataloader_path,
            child_dataloader_path,
        )

        # Step 3: Assemble code.
        logger.info("Assembling pass-through code for RuleBlack node %s", node_id)

        assembled_code = _assemble_code(str(node_id), str(self.workspace))
        if assembled_code:
            assembled_path = self.workspace / f"code_{node_id}.py"
            assembled_path.write_text(assembled_code, encoding="utf-8")
            logger.info("Assembled RuleBlack code saved to %s", assembled_path)
        else:
            logger.warning(
                "Failed to assemble RuleBlack code through _assemble_code. "
                "Checking for existing code file."
            )

        # Step 4: Run inherited solution.
        logger.info("Running inherited parent solution for RuleBlack node %s", node_id)
        exec_res = run_code_via_bash(self.agent, self.workspace, node_id)

        # Step 5: Evaluate normally with metric agent.
        metric_info = self._run_metric_agent(
            exec_res.get("code", ""),
            exec_res.get("stdout", ""),
        )

        raw_response = (
            "RuleBlackExp no-op pass-through ablation.\n"
            "The parent template and parent DataLoader were copied unchanged.\n"
            "No LLM modification was requested.\n"
            "No data cleaning, preprocessing, feature engineering, augmentation, "
            "local data synthesis, or data_links modification was performed."
        )

        return {
            "plan": "No-op pass-through Black node. Parent implementation reused unchanged.",
            "code": exec_res.get("code", ""),
            "raw_response": raw_response,
            "exec": exec_res,
            "metric": metric_info.get("metric"),
            "metric_detail": metric_info,
        }

    def _error_result(self, message: str) -> dict:
        """Return an error result with the same general shape as BlackExp.run()."""
        return {
            "plan": "RuleBlackExp failed before execution.",
            "code": "",
            "raw_response": message,
            "exec": {
                "stdout": message,
                "stderr": message,
                "exit_code": -1,
            },
            "metric": None,
            "metric_detail": {
                "metric": None,
                "is_bug": True,
                "has_submission": False,
                "summary": message,
            },
        }