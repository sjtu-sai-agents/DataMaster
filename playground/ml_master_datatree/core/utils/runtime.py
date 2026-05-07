"""运行相关的通用工具：代码提取、agent 回复提取、模拟工具调用执行代码"""

import json
import re
from pathlib import Path
from typing import Any

import ast

from search_dataset_tools.operate_submission._submission_utils import run_code_sync, get_cached_execution_result

def is_valid_python_content(code_string: str) -> bool:
    try:
        ast.parse(code_string)
        return True
    except SyntaxError as e:
        print(f"Error: syntax error for current code string: {e}\n{code_string[:100]}")
        return False


def extract_python_code(text: str, workspace, node_id) -> str:
    """从带 Markdown 的回复中提取首个 Python 代码块；若无则返回原文"""
    if not text:
        return ""
    m = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    extracted_content = m.group(1).strip() if m else text.strip()
    return extracted_content 


def extract_json_code(text: str) -> str:
    """从带 Markdown 的回复中提取首个 JSON 代码块；若无则返回原文"""
    if not text:
        return ""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def extract_agent_response(trajectory: Any) -> str:
    """抽取 Agent 轨迹中最后一 assistant 文本"""
    if not trajectory or not getattr(trajectory, "dialogs", None):
        return ""
    last_dialog = trajectory.dialogs[-1]
    for msg in reversed(last_dialog.messages):
        if hasattr(msg, "role") and msg.role.value == "assistant":
            if getattr(msg, "content", None):
                return msg.content
    return ""



def run_code_via_bash(agent, workspace: Path, node_id: str) -> dict[str, Any]:
    """运行代码并返回结果，优先从缓存获取，未命中时调用 run_code_sync。

    缓存文件位置: workspace/cache/{node_id}.json

    Args:
        agent: Agent 实例（保留兼容性，当前未使用）
        workspace: 工作目录路径
        node_id: 节点 ID

    Returns:
        包含 stdout, exit_code, stderr, success, elapsed_time, script, code 的字典
    """
    workspace.mkdir(parents=True, exist_ok=True)
    script = workspace / f"code_{node_id}.py"
    if script.exists():
        final_code = script.read_text(encoding="utf-8")
    else:
        final_code = "CODE NOT FOUND, Submission is invalid"

    # Try to get cached result first
    cached = get_cached_execution_result(str(node_id), str(workspace))
    if cached is not None:
        return {
            "stdout": cached.get("stdout", ""),
            "exit_code": cached.get("exit_code", -1),
            "stderr": cached.get("stderr", ""),
            "success": cached.get("success", False),
            "elapsed_time": cached.get("elapsed_time", 0),
            "script": str(script),
            "code": final_code,
            "from_cache": True,
        }

    # No cache, run the code
    result = run_code_sync(str(node_id), str(workspace))
    result["from_cache"] = False
    return result

def extract_last_run_code_result(traj, workspace: Path, node_id: str) -> dict[str, Any] | None:
    """从 trajectory 中提取最后一次 operate_submission_run_code 的工具结果。

    找不到时返回 None，由调用方 fallback 到 run_code_via_bash。
    """
    for step in reversed(traj.steps):
        for msg in reversed(step.tool_responses):
            if msg.name == "operate_submission_run_code":
                try:
                    data = json.loads(msg.content)
                    script = workspace / f"code_{node_id}.py"
                    return {
                        "stdout": data.get("stdout", ""),
                        "exit_code": data.get("exit_code", -1),
                        "script": str(script),
                        "code": script.read_text(encoding="utf-8") if script.exists() else "",
                    }
                except Exception:
                    pass
    return None


# Helper to extract natural language plan before first code block
def extract_text_up_to_code(text: str, workspace, node_id) -> str:
    """提取首个代码块前的自然语言文本 """
    if not text:
        return ""
    if "```" not in text:
        return text.strip()
    return text.split("```", 1)[0].strip()
