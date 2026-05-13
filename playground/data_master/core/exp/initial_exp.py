"""Initial Experiment Implementation for Root Node."""

import logging
from pathlib import Path

from evomaster.agent import BaseAgent
from evomaster.utils.types import TaskInstance

from . import NodeExp
from ..utils.runtime import (
    extract_agent_response,
    extract_python_code,
    run_code_via_bash,
)

from playground.search_dataset_tools.operate_submission._submission_utils import _assemble_code

logger = logging.getLogger(__name__)


def _assemble_mode_specific_prompt(
    initial_code: str | None = None,
    initial_instruction: str | None = None,
) -> str:
    """根据 initial_code 和 initial_instruction 参数组装模式特定的提示词

    Args:
        initial_code: 初始代码内容（可选）
        initial_instruction: 初始指令（可选）

    Returns:
        组装好的模式特定提示词内容
    """
    # 判断四种模式
    has_code = bool(initial_code)
    has_instruction = bool(initial_instruction)

    # 根据模式选择对应的提示词文件
    if has_code and has_instruction:
        # 模式1: 两者都有
        template_file = Path("playground/data_master/prompts/initial/mode_code_and_instruction.md")
        logger.info("Code and Instruction Both!")
    elif has_code:
        # 模式2: 只有代码
        template_file = Path("playground/data_master/prompts/initial/mode_code_only.md")
        logger.info("Using initial mode: using initial_code only")
    elif has_instruction:
        # 模式3: 只有指令
        template_file = Path("playground/data_master/prompts/initial/mode_instruction_only.md")
        logger.info("Using initial mode: using initial instruction only")
    else:
        # 模式4: 都没有（自由模式）
        template_file = Path("playground/data_master/prompts/initial/mode_free.md")
        logger.info("Using Free mode, code and instruction disabled.")

    # 读取模板文件
    try:
        with open(template_file, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        logger.error(f"模板文件不存在: {template_file}")
        exit(0)
        
    # 替换变量占位符
    content = content.replace("{initial_code_content}", initial_code or "")
    content = content.replace("{initial_instruction}", initial_instruction or "")
    return content


class InitialExp(NodeExp):
    """Initial 实验：根节点生成初始版本的代码"""

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
        initial_code: str | None = None,
        initial_instruction: str | None = None,
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
        self.initial_code = initial_code
        self.initial_instruction = initial_instruction

    def run(self, task_description: str) -> dict:
        """运行初始节点实验，生成第一版代码

        Args:
            task_description: 任务描述

        Returns:
            包含 plan, code, raw_response, exec, metric 等字段的字典
        """
        node_id = self.node.id
        BaseAgent.set_exp_info(
            exp_name=f"initial_{node_id[:8]}", exp_index=self.exp_index
        )
        
        
        # get data loader usage
        data_loader_usage_path = Path(
            "playground/data_master/prompts/tools/dataloader_en.md"
        )
        with open(data_loader_usage_path, "r", encoding="utf-8") as file:
            data_loader_usage = file.read()

        # get operation tools readme
        # * special tools and prompts for initial-exp
        tools_manual_readme_path = Path(
            "playground/data_master/prompts/tools/operate/for_initial_exp_en.md"
        )
        with open(tools_manual_readme_path, "r", encoding="utf-8") as file:
            tools_manual_readme = file.read()

        # memory_tree tools readme
        memory_tree_path = "playground/data_master/prompts/tools/memory.md"
        with open(memory_tree_path, "r", encoding="utf-8") as file:
            memory_tree_manual = file.read()

        # 组装模式特定的提示词
        mode_specific_content = _assemble_mode_specific_prompt(
            initial_code=self.initial_code,
            initial_instruction=self.initial_instruction,
        )

        # 构建 prompt 格式变量
        fmt = {
            "mode_specific_content": mode_specific_content,
            "task_description": task_description,
            "data_preview": self.data_preview,
            "data_loader_readme": data_loader_usage,
            "node_id": node_id,
            "workspace": str(self.workspace),
            "operation_tools_readme": tools_manual_readme,
            "memory_tree_manual": memory_tree_manual,
        }
 
        # todo call summary tools
        final_turn_prompt = "ATTENTION! This is the final turn of your current dialog, read your historical messages and generate the overall response. You are not allowed to call more tools! Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block."

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

        # Step 3: 检查并验证生成的文件
        # agent 应该已经通过 write_code() 工具生成了 template 和 dataloader 文件
        template_path = self.workspace / f"code_{node_id}_template.py"
        dataloader_path = self.workspace / f"code_{node_id}_dataloader.py"

        # 验证文件是否存在
        if not template_path.exists():
            logger.warning(f"Template file not found: {template_path}")
            logger.info("Attempting to extract template from agent response...")

        if not dataloader_path.exists():
            logger.warning(f"Dataloader file not found: {dataloader_path}")
            logger.info("Attempting to extract dataloader from agent response...")
            # 尝试从回复中提取代码
            dataloader_code = extract_python_code(text, self.workspace, node_id)
            if dataloader_code:
                dataloader_path.write_text(dataloader_code, encoding="utf-8")
                logger.info(f"Extracted and saved dataloader code to {dataloader_path}")
            else:
                logger.error("Failed to extract dataloader code from agent response")

        # Step 4: 组装代码并运行
        # 代码拼装顺序：base_dataloader + "\n\n" + dataloader + template
        logger.info(f"Assembling code for node {node_id}")

        # 尝试自动组装代码
        assembled_code = _assemble_code(str(node_id), str(self.workspace))
        if assembled_code:
            # 将组装后的代码写入 code_{node_id}.py
            assembled_path = self.workspace / f"code_{node_id}.py"
            assembled_path.write_text(assembled_code, encoding="utf-8")
            logger.info(f"Assembled code saved to {assembled_path}")
        else:
            logger.warning(
                "Failed to assemble code, checking for existing code file..."
            )

        # 运行代码
        logger.info(f"Running code for node {node_id}")
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
