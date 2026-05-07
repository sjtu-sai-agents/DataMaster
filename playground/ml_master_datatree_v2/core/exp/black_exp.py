"""Black Experiment v2 - Skilled Data Cleaner.

Black 节点负责：
1. 读取 data_manifest.json（如果 Red 已写入）
2. 根据任务类型从 black-dataops skill 中选择合适的数据清洗技能
3. 将 manifest 中的外部数据 + skills 融入 MyDataLoader.setup() 中
4. 跑完整训练并产出 submission

与 v1 的核心区别：
- 严禁自行搜索/下载外部数据（那是 Red 的工作）
- Prompt 中注入 manifest（外部数据路径/格式/loading snippet 已知）
- 使用 EvoMaster SkillRegistry 暴露的 `black-dataops` skill，而不是旧的 prompt 内联 catalog
"""

import json
import logging
from pathlib import Path

from evomaster.agent import BaseAgent
from evomaster.utils.types import TaskInstance

from . import NodeExp
from .red_exp import load_manifest
from playground.ml_master_datatree.core.utils.runtime import (
    extract_agent_response,
    extract_last_run_code_result,
    run_code_via_bash,
)

logger = logging.getLogger(__name__)


class BlackExp(NodeExp):
    """Black 实验 v2：有技能包的数据清洗 agent。"""

    def __init__(
        self,
        agent,
        metric_agent,
        session,
        workspace: Path,
        task_workspace: Path,
        manifest_path: Path | None,
        exp_id: str | None,
        data_preview: str,
        node,
        exp_index: int = 0,
    ):
        super().__init__(agent, metric_agent, session, workspace, exp_id, data_preview, node, exp_index)
        self.task_workspace = task_workspace
        self.manifest_path = manifest_path

    def run(
        self,
        task_description: str,
        prev_code: str,
        memory: str,
        term_out: str = "",
        best_code: str | None = None,
        best_metric: float | None = None,
    ) -> dict:
        node_id = self.node.id
        BaseAgent.set_exp_info(exp_name=f"black_{node_id[:8]}", exp_index=self.exp_index)

        # Pre-seed the node's code file with parent code
        if prev_code:
            code_file = self.workspace / f"code_{node_id}.py"
            if not code_file.exists():
                code_file.parent.mkdir(parents=True, exist_ok=True)
                code_file.write_text(prev_code, encoding="utf-8")

        # 读取 manifest（Red 写入，Black 消费）
        manifest = load_manifest(self.task_workspace, self.manifest_path)
        manifest_section = self._build_manifest_section(manifest, self.manifest_path)

        tools_manual_path = Path("playground/ml_master_datatree/prompts/general/tools_manual.md")
        with open(tools_manual_path, encoding="utf-8") as f:
            tools_manual = f.read()

        general_instructions_path = Path("playground/ml_master_datatree/prompts/general/general_instructions.md")
        with open(general_instructions_path, encoding="utf-8") as f:
            general_instructions = f.read()

        data_loader_usage_path = Path("playground/ml_master_datatree/prompts/general/data_loader_usage.md")
        with open(data_loader_usage_path, encoding="utf-8") as f:
            data_loader_usage = f.read()

        fmt = {
            "task_description": task_description,
            "general_instruction_content": general_instructions,
            "previous_code": prev_code,
            "execution_output": term_out,
            "memory": memory,
            "best_code": best_code or "",
            "best_metric": best_metric or "N/A",
            "data_preview": self.data_preview,
            "data_loader_readme": data_loader_usage,
            "workspace": str(self.workspace),
            "task_workspace": str(self.task_workspace),
            "node_id": node_id,
            "operation_tools_readme": tools_manual,
            "manifest_section": manifest_section,
        }

        final_turn_prompt = (
            "ATTENTION! This is the final turn of your current dialog, "
            "read your historical messages and generate the overall response. "
            "You are not allowed to call more tools! Your response should be a brief "
            "outline/sketch of your proposed solution in natural language (3-5 sentences), "
            "followed by a single markdown code block (wrapped in ```) which implements "
            "this solution and prints out the evaluation metric. "
            "There should be no additional headings or text in your response. "
            "Just natural language text followed by a newline and then the markdown code block."
        )

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

        exec_res = (
            extract_last_run_code_result(traj, self.workspace, node_id)
            or run_code_via_bash(self.agent, self.workspace, node_id)
        )
        metric_info = self._run_metric_agent(exec_res.get("code", ""), exec_res.get("stdout", ""))

        return {
            "plan": "",
            "code": exec_res.get("code", ""),
            "raw_response": text,
            "exec": exec_res,
            "metric": metric_info.get("metric"),
            "metric_detail": metric_info,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_manifest_section(
        self,
        manifest: dict | None,
        manifest_path: Path | None,
    ) -> str:
        """将 manifest 格式化为 prompt 中的 Markdown 段落。"""
        manifest_path_str = str(manifest_path) if manifest_path else "（未绑定 manifest）"
        if not manifest:
            return (
                "## 外部数据 Manifest\n\n"
                f"当前绑定的 manifest 路径：`{manifest_path_str}`\n\n"
                "⚠️ 暂无可用 manifest（Red agent 尚未运行或未找到外部数据）。\n"
                "你只能使用 `input/` 目录下的原始数据。\n"
                "**严禁自行搜索或下载外部数据**。"
            )

        datasets = manifest.get("external_datasets", [])
        if not datasets:
            return (
                "## 外部数据 Manifest\n\n"
                f"当前绑定的 manifest 路径：`{manifest_path_str}`\n\n"
                "⚠️ Manifest 文件存在，但没有找到有效的外部数据集。只使用原始数据。"
            )

        lines = [
            "## 外部数据 Manifest",
            "",
            f"当前绑定的 manifest 路径：`{manifest_path_str}`",
            "",
            f"Red agent 已发现并下载了 **{len(datasets)}** 个外部数据集：",
            "",
        ]
        for i, ds in enumerate(datasets, 1):
            lines += [
                f"### 数据集 {i}：{ds.get('name', 'Unknown')}",
                f"- **本地路径**：`{ds.get('local_path', '')}`",
                f"- **格式**：`{ds.get('format', 'unknown')}`",
                f"- **样本数**：{ds.get('statistics', {}).get('num_samples', '未知')}",
                f"- **说明**：{ds.get('description', '')}",
                "",
            ]
            schema = ds.get("schema", {})
            if schema:
                lines.append("**Schema**：")
                for k, v in schema.items():
                    lines.append(f"  - `{k}`: {v}")
                lines.append("")

            stats = ds.get("statistics", {})
            img_size = stats.get("image_size")
            if img_size:
                lines.append(f"**实测图片尺寸**：{img_size}")
                lines.append("")

            label_mapping = ds.get("label_mapping", {})
            if label_mapping:
                lines.append("**Label 映射**（外部 → 比赛）：")
                for ext_label, comp_label in label_mapping.items():
                    lines.append(f"  - `{ext_label}` → `{comp_label}`")
                lines.append("")

            snippet = ds.get("loading_snippet", "")
            if snippet:
                lines += [
                    "**已验证的 Loading Snippet**（直接复制使用）：",
                    "```python",
                    snippet,
                    "```",
                    "",
                ]

        lines += [
            "---",
            "⚠️ **重要**：请直接使用上面的 loading snippet 加载外部数据，"
            "不要自行猜测数据格式。",
        ]
        return "\n".join(lines)
