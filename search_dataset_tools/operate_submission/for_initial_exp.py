"""MCP tools for Initial node - operates on both template and dataloader files.

Initial 节点可以修改 template 和 dataloader 两个文件。
"""

import asyncio
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from _submission_utils import (
    _get_dataloader_file_path,
    _get_template_file_path,
    _get_base_dataloader_path,
    _run_code_async,
    _validate_submission_async,
    _grade_code_async,
    ensure_grading_server,
    _validate_workspace
)

logger = logging.getLogger(__name__)
mcp = FastMCP("operate-submission-initial")
logging.basicConfig(level=logging.INFO)


@mcp.tool()
def read_code(node_id: str, workspace: str) -> str:
    """读取所有代码组件：base_dataloader + dataloader + template。"""
    try:
        _validate_workspace(workspace)
        parts = []
        
        # Base dataloader
        base_path = _get_base_dataloader_path()
        if base_path.exists():
            parts.append("===== BASE DATALOADER =====")
            parts.append(base_path.read_text(encoding="utf-8"))
        
        parts.append("")
        
        # Dataloader
        dataloader_path = _get_dataloader_file_path(node_id, workspace)
        if dataloader_path.exists():
            parts.append("===== MY DATALOADER =====")
            parts.append(dataloader_path.read_text(encoding="utf-8"))
        
        parts.append("")
        
        # Template
        template_path = _get_template_file_path(node_id, workspace)
        if template_path.exists():
            parts.append("===== TEMPLATE =====")
            parts.append(template_path.read_text(encoding="utf-8"))
        
        return "\n".join(parts)
    except Exception as e:
        logger.error(str(e))
        return str(e)


@mcp.tool()
def write_code(code: str, node_id: str, workspace: str, file_type: str = "dataloader", override: bool = False) -> str:
    """写入代码（template 或 dataloader）。

    Args:
        code: Python 代码字符串
        node_id: 节点 ID
        workspace: 工作目录
        file_type: "template" 或 "dataloader"（默认）
        override: 是否覆盖已有内容
    """
    try:
        _validate_workspace(workspace)
        if file_type == "template":
            file_path = _get_template_file_path(node_id, workspace)
        elif file_type == "dataloader":
            file_path = _get_dataloader_file_path(node_id, workspace)
        else:
            return f"Error: file_type 必须是 'template' 或 'dataloader'，得到：{file_type}"
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not override and file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            if content:
                return f"Warning: {file_type} 已有内容\n{content}\n\n如需覆盖请设置 override=True"
        
        file_path.write_text(code, encoding="utf-8")
        return f"{file_type.capitalize()} 写入成功：{file_path}"
    except Exception as e:
        logger.error(str(e))
        return str(e)


@mcp.tool()
def fix_code(old_string: str, new_string: str, node_id: str, workspace: str, file_type: str = "dataloader", replace_all: bool = False) -> str:
    """修改代码（template 或 dataloader）。

    Args:
        old_string: 要替换的原始字符串
        new_string: 新字符串
        node_id: 节点 ID
        workspace: 工作目录
        file_type: "template" 或 "dataloader"（默认）
        replace_all: 是否替换所有匹配项
    """
    try:
        _validate_workspace(workspace)
        if file_type == "template":
            file_path = _get_template_file_path(node_id, workspace)
        elif file_type == "dataloader":
            file_path = _get_dataloader_file_path(node_id, workspace)
        else:
            return f"Error: file_type 必须是 'template' 或 'dataloader'，得到：{file_type}"
        
        if not file_path.exists():
            return f"{file_type.capitalize()} 文件不存在：{file_path}"
        
        content = file_path.read_text(encoding="utf-8")
        
        if old_string not in content:
            return f"未找到指定的代码片段：{old_string[:50]}..."
        
        new_content = content.replace(old_string, new_string, -1 if replace_all else 1)
        file_path.write_text(new_content, encoding="utf-8")
        
        return f"{file_type.capitalize()} 代码修改成功"
    except Exception as e:
        logger.error(str(e))
        return str(e)


@mcp.tool()
async def run_code(node_id: str, workspace: str, timeout: int = 3600) -> str:
    """执行代码（自动拼装 base_dataloader + dataloader + template）。"""
    _validate_workspace(workspace)
    return await _run_code_async(node_id, workspace, timeout, None, logger)


@mcp.tool()
async def validate_submission(node_id: str, workspace: str) -> str:
    """验证提交文件。"""
    _validate_workspace(workspace)
    return await _validate_submission_async(node_id, workspace, ensure_grading_server, logger)


@mcp.tool()
async def grade_code(node_id: str, workspace: str, timeout: int = 300) -> str:
    """评分。"""
    _validate_workspace(workspace)
    return await _grade_code_async(node_id, workspace, timeout, logger)


if __name__ == "__main__":
    mcp.run()
