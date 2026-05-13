"""Black Experiment Implementation - Data Augmentation and Feature Engineering."""

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


class BlackExp(NodeExp):
    """Black 实验：对现有数据进行数据增强、特征工程和整合

    Black 节点负责：
    - 数据清洗和预处理
    - 特征工程（创建新特征、特征转换）
    - 数据增强（SMOTE、噪声注入等）
    - 特征选择和降维
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
        """运行 black 节点实验

        Args:
            task_description: 任务描述
            prev_code: 父节点的代码
            memory: 子节点记忆（之前尝试的数据操作）
            term_out: 前一次执行的输出
            best_code: 全局最佳代码
            best_metric: 全局最佳指标

        Returns:
            包含 plan, code, raw_response, exec, metric 等字段的字典
        """
        node_id = self.node.id
        parent_node = self.node.parent if self.node.parent else None
        parent_id = parent_node.id if parent_node else None

        BaseAgent.set_exp_info(
            exp_name=f"black_{node_id[:8]}", exp_index=self.exp_index
        )
        logger.info(
            f"Black node {node_id[:8]}, parent: {parent_id[:8] if parent_id else 'none'}"
        )

        # Step 1: 从父节点复制模板
        if parent_id:
            parent_template_path = self.workspace / f"code_{parent_id}_template.py"
            if not parent_template_path.exists():
                logger.error(f"Parent template not found: {parent_template_path}")
                return {"error": "parent template not found"}

            # 读取父节点模板并修改 import 语句
            template_code = parent_template_path.read_text(encoding="utf-8")

            # 保存模板
            template_path = self.workspace / f"code_{node_id}_template.py"
            template_path.write_text(template_code, encoding="utf-8")
            logger.info(f"Template copied from parent and saved to {template_path}")

        # get data loader usage
        data_loader_usage_path = Path(
            "playground/ml_master_datatree/prompts/tools/dataloader_en.md"
        )
        with open(data_loader_usage_path, "r", encoding="utf-8") as file:
            data_loader_usage = file.read()

        # get operation tools readme - black node uses for_datanode tools
        tools_manual_readme_path = Path(
            "playground/ml_master_datatree/prompts/tools/operate/for_datanode_en.md"
        )
        with open(tools_manual_readme_path, "r", encoding="utf-8") as file:
            tools_manual_readme = file.read()

        # 提取父节点的 DataLoader 代码作为参考
        parent_dataloader_code = ""
        if parent_id:
            parent_dataloader_path = self.workspace / f"code_{parent_id}_dataloader.py"
            if parent_dataloader_path.exists():
                parent_dataloader_code = parent_dataloader_path.read_text(
                    encoding="utf-8"
                )

        # memory_tree tools readme
        memory_tree_path = (
            "playground/ml_master_datatree/prompts/tools/memory.md"
        )
        with open(memory_tree_path, "r", encoding="utf-8") as file:
            memory_tree_manual = file.read()

        fmt = {
            "task_description": task_description,
            "previous_code": prev_code,
            "memory": memory,
            "execution_output": term_out,
            "best_code": best_code or "",
            "best_metric": best_metric,
            "data_preview": self.data_preview,
            "data_loader_readme": data_loader_usage,
            "node_id": node_id,
            "workspace": str(self.workspace),
            "operation_tools_readme": tools_manual_readme,
            "parent_dataloader": parent_dataloader_code,
            "memory_tree_manual": memory_tree_manual,
        }
        
        # todo rewrite final turn prompt
        final_turn_prompt = "ATTENTION! This is the final turn of your current dialog, read your historical messages and generate the overall response. You are not allowed to call more tools! Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block."

        orig_fmt = self.agent._prompt_format_kwargs.copy()
        self.agent._prompt_format_kwargs.update(fmt)
        try:
            task = TaskInstance(
                task_id=f"{node_id}_black",
                task_type="black",
                description=task_description,
                input_data={},
            )
            traj = self.agent.run(
                task, enable_final_turn=True, enable_final_turn_prompt=final_turn_prompt
            )
            text = extract_agent_response(traj)
        finally:
            self.agent._prompt_format_kwargs = orig_fmt

        # Step 3: 检查并保存 DataLoader 代码
        # agent 应该已经通过 write_code() 工具生成了 dataloader 文件
        dataloader_path = self.workspace / f"code_{node_id}_dataloader.py"

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
