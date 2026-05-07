"""Initial Experiment for v2 playground.

bypass 模式：当 initial_code_structured.py 存在时，直接写入并执行预写的
结构化代码（BaseDataLoader 格式），跳过 LLM agent 调用，加速树状探索启动。

回退模式：若结构化文件不存在，走与 v1 相同的 LLM-based 重构流程。
"""

import logging
from pathlib import Path

from evomaster.agent import BaseAgent
from evomaster.utils.types import TaskInstance

from . import NodeExp
from playground.ml_master_datatree.core.utils.runtime import (
    extract_agent_response,
    run_code_via_bash,
)

logger = logging.getLogger(__name__)

_STRUCTURED_CODE_FILENAME = "initial_code_structured.py"
_INITIAL_CODE_BASE_DIR = Path("playground/ml_master_datatree/initial_code")


class InitialExp(NodeExp):
    """v2 Initial 实验：优先使用预写的结构化代码，跳过 LLM 重构步骤。"""

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
        )

    # ------------------------------------------------------------------
    # Bypass: directly execute pre-written structured code
    # ------------------------------------------------------------------

    def _run_bypass(self, node_id: str, structured_path: Path) -> dict:
        """跳过 LLM，直接将结构化代码写入 workspace 并执行。"""
        logger.info(
            "InitialExp [v2 bypass]: loading pre-written structured code from %s",
            structured_path,
        )
        code_content = structured_path.read_text(encoding="utf-8")

        script = self.workspace / f"code_{node_id}.py"
        self.workspace.mkdir(parents=True, exist_ok=True)
        script.write_text(code_content, encoding="utf-8")
        logger.info("Structured code written to %s", script)

        exec_res = run_code_via_bash(self.agent, self.workspace, node_id)
        metric_info = self._run_metric_agent(
            exec_res.get("code", ""), exec_res.get("stdout", "")
        )
        return {
            "plan": "v2 bypass: pre-written structured initial code",
            "code": exec_res.get("code", ""),
            "raw_response": "bypass mode — no LLM call",
            "exec": exec_res,
            "metric": metric_info.get("metric"),
            "metric_detail": metric_info,
        }

    # ------------------------------------------------------------------
    # Fallback: LLM-based refactoring (same as v1)
    # ------------------------------------------------------------------

    def _run_llm(self, node_id: str, task_description: str) -> dict:
        """调用 LLM agent 将 initial_code.py 重构为 BaseDataLoader 格式。"""
        config = {
            "SUBMISSION_FILE": str(self.workspace / "submission" / "submission.csv"),
            "SERVER_URL": "http://localhost:5003/validate",
        }

        with open(
            "playground/ml_master_datatree/prompts/general/general_instructions.md",
            "r",
            encoding="utf-8",
        ) as f:
            general_instruction_context = f.read()
        general_instruction_context.format(**config)

        initial_code_path = _INITIAL_CODE_BASE_DIR / self.exp_id / "initial_code.py"
        if not initial_code_path.exists():
            logger.error("initial code path not found: %s", initial_code_path)
        with open(initial_code_path, "r", encoding="utf-8") as f:
            initial_code_content = f.read()

        with open(
            "playground/ml_master_datatree/prompts/general/data_loader_usage.md",
            "r",
            encoding="utf-8",
        ) as f:
            data_loader_usage = f.read()

        with open(
            "playground/ml_master_datatree/prompts/general/tools_manual.md",
            "r",
            encoding="utf-8",
        ) as f:
            tools_manual_readme = f.read()

        fmt = {
            "general_instruction_content": general_instruction_context,
            "initial_code_content": initial_code_content,
            "task_description": task_description,
            "data_preview": self.data_preview,
            "data_loader_readme": data_loader_usage,
            "node_id": node_id,
            "workspace": str(self.workspace),
            "operation_tools_readme": tools_manual_readme,
        }

        final_turn_prompt = (
            "ATTENTION! This is the final turn of your current dialog, read your "
            "historical messages and generate the overall response. You are not allowed "
            "to call more tools! Your response should be a brief outline/sketch of your "
            "proposed solution in natural language (3-5 sentences), followed by a single "
            "markdown code block (wrapped in ```) which implements this solution and prints "
            "out the evaluation metric. There should be no additional headings or text in "
            "your response. Just natural language text followed by a newline and then the "
            "markdown code block."
        )

        orig_fmt = self.agent._prompt_format_kwargs.copy()
        self.agent._prompt_format_kwargs.update(fmt)
        try:
            task = TaskInstance(
                task_id=f"{node_id}_initial",
                task_type="initial",
                description=task_description,
                input_data={},
            )
            traj = self.agent.run(
                task, enable_final_turn=True, enable_final_turn_prompt=final_turn_prompt
            )
            text = extract_agent_response(traj)
        finally:
            self.agent._prompt_format_kwargs = orig_fmt

        exec_res = run_code_via_bash(self.agent, self.workspace, node_id)
        metric_info = self._run_metric_agent(
            exec_res.get("code", ""), exec_res.get("stdout", "")
        )
        return {
            "plan": "",
            "code": exec_res.get("code", ""),
            "raw_response": text,
            "exec": exec_res,
            "metric": metric_info.get("metric"),
            "metric_detail": metric_info,
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, task_description: str) -> dict:
        """运行初始节点实验。

        优先走 bypass 模式（exp_id/initial_code_structured.py 存在时）；
        否则 fallback 到 LLM-based 重构。
        """
        node_id = self.node.id
        BaseAgent.set_exp_info(
            exp_name=f"initial_{node_id[:8]}", exp_index=self.exp_index
        )

        if self.exp_id:
            structured_path = (
                _INITIAL_CODE_BASE_DIR / self.exp_id / _STRUCTURED_CODE_FILENAME
            )
            if structured_path.exists():
                return self._run_bypass(node_id, structured_path)
            logger.info(
                "InitialExp [v2]: structured code not found (%s), falling back to LLM mode",
                structured_path,
            )

        return self._run_llm(node_id, task_description)
