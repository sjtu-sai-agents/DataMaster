"""Tools for Black/Red nodes - operates on DataLoader files ONLY.

Black/Red nodes can only modify dataloader files; template is inherited from parent node and read-only.
"""

import asyncio
import logging
from pathlib import Path

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
logging.basicConfig(level=logging.INFO)


def read_code(node_id: str, workspace: str) -> str:
    """Read all code components: base_dataloader + dataloader + template.

    Black/Red nodes: you can only modify the dataloader file! Template is read-only!
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
            parts.append("===== MY DATALOADER (modifiable) =====")
            parts.append(dataloader_path.read_text(encoding="utf-8"))

        parts.append("")

        # Template (READ-ONLY)
        template_path = _get_template_file_path(node_id, workspace)
        if template_path.exists():
            parts.append("===== TEMPLATE (read-only, inherited from parent) =====")
            parts.append(template_path.read_text(encoding="utf-8"))

        return "\n".join(parts)
    except Exception as e:
        logger.error(str(e))
        return str(e)


def write_code(code: str, node_id: str, workspace: str, override: bool = False) -> str:
    """Write DataLoader code.

    Only writes to the dataloader file, cannot modify template!
    """
    try:
        _validate_workspace(workspace)
        dataloader_path = _get_dataloader_file_path(node_id, workspace)
        dataloader_path.parent.mkdir(parents=True, exist_ok=True)

        if not override and dataloader_path.exists():
            content = dataloader_path.read_text(encoding="utf-8")
            if content:
                return f"Warning: dataloader already has content\n{content}\n\nSet override=True to overwrite"

        dataloader_path.write_text(code, encoding="utf-8")
        return f"Dataloader written successfully: {dataloader_path}"
    except Exception as e:
        logger.error(str(e))
        return str(e)


def fix_code(old_string: str, new_string: str, node_id: str, workspace: str, replace_all: bool = False) -> str:
    """Modify DataLoader code.

    Only modifies the dataloader file, cannot modify template!
    """
    try:
        _validate_workspace(workspace)
        dataloader_path = _get_dataloader_file_path(node_id, workspace)

        if not dataloader_path.exists():
            return f"Dataloader file does not exist: {dataloader_path}"

        content = dataloader_path.read_text(encoding="utf-8")

        if old_string not in content:
            return f"Code snippet not found: {old_string[:50]}..."

        new_content = content.replace(old_string, new_string, -1 if replace_all else 1)
        dataloader_path.write_text(new_content, encoding="utf-8")

        return "Dataloader code modified successfully"
    except Exception as e:
        logger.error(str(e))
        return str(e)


async def run_code(node_id: str, workspace: str, timeout: int = 3600) -> str:
    """Execute code (automatically assembles base_dataloader + dataloader + template)."""
    _validate_workspace(workspace)
    return await _run_code_async(node_id, workspace, timeout, None, logger)


async def validate_submission(node_id: str, workspace: str) -> str:
    """Validate submission file."""
    _validate_workspace(workspace)
    return await _validate_submission_async(node_id, workspace, ensure_grading_server, logger)


async def grade_code(node_id: str, workspace: str, timeout: int = 300) -> str:
    """Grade submission."""
    _validate_workspace(workspace)
    return await _grade_code_async(node_id, workspace, timeout, logger)
