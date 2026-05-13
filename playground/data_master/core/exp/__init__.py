# coding: utf-8
"""Exp 基础类与导出。辅助工具已拆分utils包中
"""

import logging
from pathlib import Path

from evomaster.agent import BaseAgent
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance
from openai import OpenAI

from ..utils.runtime import extract_agent_response
from ..utils.metric import parse_metric_content

logger = logging.getLogger(__name__)


class NodeExp(BaseExp):
    """针对单个UCT节点的基础Exp，统一持有节点与数据预览信息，便于后续复用"""

    def __init__(self, agent, metric_agent, session, workspace: Path, exp_id: str | None, data_preview: str, node, exp_index: int = 0,
                 test_feedback: bool = False, force_direction: str | None = None):
        super().__init__(agent=agent, config=None)
        self.metric_agent: BaseAgent = metric_agent
        self.session = session
        self.workspace = workspace
        self.exp_id = exp_id
        self.data_preview = data_preview
        self.node = node
        self.exp_index = exp_index
        # Test feedback 相关
        self.test_feedback = test_feedback
        self.force_direction = force_direction
        self.test_metric_agent = None  # 将由 playground 注入

    def _run_metric_agent(self, code: str, stdout: str) -> dict:
        if not self.metric_agent:
            return {"metric": None, "is_bug": True, "has_submission": False}

        # ========== Test-feedback 模式：使用测试集分数 ==========
        if self.test_feedback and self.test_metric_agent:
            # 获取测试集分数
            from playground.search_dataset_tools.operate_submission._submission_utils import grade_code_sync

            grade_result = grade_code_sync(
                node_id=self.node.id,
                workspace=str(self.workspace),
                timeout=300,
                logger=self.logger
            )

            if grade_result.get("success"):
                grade_output = grade_result.get("output", "")

                # 使用 test_metric_agent 解析 grade 输出
                lower_is_better = (self.force_direction == "minimize")
                direction_str = "minimize" if lower_is_better else "maximize"

                orig_fmt = self.test_metric_agent._prompt_format_kwargs.copy()
                self.test_metric_agent._prompt_format_kwargs.update({
                    "grade_output": grade_output,
                    "direction": direction_str,
                    "lower_is_better": str(lower_is_better).lower(),
                })

                try:
                    task = TaskInstance(task_id="parse_test_metric", task_type="metric_test",
                                      description="parse test metric", input_data={})
                    traj = self.test_metric_agent.run(task)
                    resp = extract_agent_response(traj)
                    result = parse_metric_content(resp)

                    if result.get("metric") is not None:
                        result["test_metric"] = result["metric"]
                    self.logger.info(f"Test feedback mode: test_metric={result.get('metric')}, direction={direction_str}")
                    return result
                finally:
                    self.test_metric_agent._prompt_format_kwargs = orig_fmt
            else:
                self.logger.warning(f"Grade failed: {grade_result.get('error')}, returning error result")
                return {"metric": None, "is_bug": True, "has_submission": False}
        # ========== Test-feedback 模式结束 ==========

        # ========== 默认模式：使用验证集分数 ==========
        orig_fmt = self.metric_agent._prompt_format_kwargs.copy()
        self.metric_agent._prompt_format_kwargs.update({"code": code, "stdout": stdout})
        try:
            task = TaskInstance(task_id="parse_metric", task_type="metric", description="parse metric", input_data={})
            traj = self.metric_agent.run(task)
            resp = extract_agent_response(traj)
            return parse_metric_content(resp)
        finally:
            self.metric_agent._prompt_format_kwargs = orig_fmt
        # ========== 默认模式结束 ==========
        
