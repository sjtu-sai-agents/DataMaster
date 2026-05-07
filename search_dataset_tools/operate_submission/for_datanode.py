"""MCP tools for Black/Red nodes - operates on DataLoader files ONLY.

Black/Red 节点只能修改 dataloader 文件，template 是从父节点继承的，只读不可修改。
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
mcp = FastMCP("operate-submission-datanode")
logging.basicConfig(level=logging.INFO)


@mcp.tool()
def read_code(node_id: str, workspace: str) -> str:
    """读取所有代码组件：base_dataloader + dataloader + template。
    
    Black/Red 节点注意：你只能修改 dataloader 文件！template 是只读的！
    """
    try:
        _validate_workspace(workspace)
        parts = []
        
        # Base dataloader
        base_path = _get_base_dataloader_path()
        if base_path.exists():
            parts.append("===== BASE DATALOADER =====")
            parts.append(base_path.read_text(encoding="utf-8"))
        
        parts.append("")
        
        # Dataloader (MODIFIABLE)
        dataloader_path = _get_dataloader_file_path(node_id, workspace)
        if dataloader_path.exists():
            parts.append("===== MY DATALOADER (可修改) =====")
            parts.append(dataloader_path.read_text(encoding="utf-8"))
        
        parts.append("")
        
        # Template (READ-ONLY)
        template_path = _get_template_file_path(node_id, workspace)
        if template_path.exists():
            parts.append("===== TEMPLATE (只读，从父节点继承) =====")
            parts.append(template_path.read_text(encoding="utf-8"))
        
        return "\n".join(parts)
    except Exception as e:
        logger.error(str(e))
        return str(e)


@mcp.tool()
def write_code(code: str, node_id: str, workspace: str, override: bool = False) -> str:
    """写入 DataLoader 代码。

    ⚠️ 只能写入 dataloader 文件，不能修改 template！
    """
    try:
        _validate_workspace(workspace)
        dataloader_path = _get_dataloader_file_path(node_id, workspace)
        dataloader_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not override and dataloader_path.exists():
            content = dataloader_path.read_text(encoding="utf-8")
            if content:
                return f"Warning: dataloader 已有内容\n{content}\n\n如需覆盖请设置 override=True"
        
        dataloader_path.write_text(code, encoding="utf-8")
        return f"Dataloader 写入成功：{dataloader_path}"
    except Exception as e:
        logger.error(str(e))
        return str(e)


@mcp.tool()
def fix_code(old_string: str, new_string: str, node_id: str, workspace: str, replace_all: bool = False) -> str:
    """修改 DataLoader 代码。

    ⚠️ 只能修改 dataloader 文件，不能修改 template！
    """
    try:
        _validate_workspace(workspace)
        dataloader_path = _get_dataloader_file_path(node_id, workspace)
        
        if not dataloader_path.exists():
            return f"Dataloader 文件不存在：{dataloader_path}"
        
        content = dataloader_path.read_text(encoding="utf-8")
        
        if old_string not in content:
            return f"未找到指定的代码片段：{old_string[:50]}..."
        
        new_content = content.replace(old_string, new_string, -1 if replace_all else 1)
        dataloader_path.write_text(new_content, encoding="utf-8")
        
        return "Dataloader 代码修改成功"
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
