"""Memory Tree Interface - Functions for agents to interact with memory tree.

Provides functions for:
1. Update their own manifest (TL;DR and recordings)
2. Read other nodes' manifests
3. Navigate the memory tree structure
4. Access node storage (code, stdout, trajectory)
"""

from pathlib import Path
import re
import logging

from memory_tree import (
    read_node_manifest,
    write_node_manifest,
    list_child_nodes,
    get_parent_id,
    read_storage_file,
    get_memory_tree_root,
    read_global_memory as read_global_memory_impl,
    add_global_memory as add_global_memory_impl,
    show_all_data as show_all_data_impl,
    show_detailed_data as show_detailed_data_impl,
    add_new_data as add_new_data_impl,
    add_data_record as add_data_record_impl,
)

logger = logging.getLogger(__name__)


def _validate_workspace(workspace: str) -> Path:
    """验证 workspace 是否存在，如果不存在则抛出 ValueError。

    Args:
        workspace: 工作目录路径（字符串）

    Returns:
        Path: 转换后的 Path 对象

    Raises:
        ValueError: 当 workspace 不存在时
    """
    workspace_path = Path(workspace).resolve()
    if not workspace_path.exists():
        raise ValueError(
            f"Workspace 不存在：{workspace_path}\n"
            f"请提供正确的 **WorkSpace 绝对路径**。\n"
        )
    return workspace_path


# =============================================================================
# Manifest Parsing Helpers
# =============================================================================


def parse_manifest(content: str) -> dict:
    """解析 manifest.md 内容为结构化数据。

    Args:
        content: manifest.md 的原始 markdown 内容

    Returns:
        包含 'tldr' 和 'recordings' 键的字典
        - tldr: 总体摘要（## TL;DR 之后的内容）
        - recordings: 包含 'id'、'summary'、'content' 的字典列表
    """
    result = {"tldr": "", "recordings": []}

    # Split into sections
    lines = content.split("\n")
    current_section = None
    current_recording_id = None
    current_recording_summary = None
    current_recording_content = []

    for line in lines:
        # Check for TL;DR section
        if line.strip() == "## TL;DR":
            current_section = "tldr"
            continue

        # Check for Recordings section
        if line.strip() == "## Recordings":
            current_section = "recordings"
            continue

        # Process TL;DR content
        if current_section == "tldr" and not line.startswith("#"):
            result["tldr"] += line + "\n"

        # Process Recordings
        if current_section == "recordings":
            # Check for new recording (e.g., "1. Recording 1: summary here")
            recording_match = re.match(
                r"^(\d+)\.\s+Recording\s+(\d+):\s*(.+)$", line.strip()
            )
            if recording_match:
                # Save previous recording if exists
                if current_recording_id is not None:
                    result["recordings"].append(
                        {
                            "id": current_recording_id,
                            "summary": current_recording_summary or "",
                            "content": "\n".join(current_recording_content).strip(),
                        }
                    )

                current_recording_id = int(recording_match.group(2))
                current_recording_summary = recording_match.group(3).strip()
                current_recording_content = []
            elif current_recording_id is not None:
                # Accumulate recording content
                current_recording_content.append(line)

    # Save last recording
    if current_recording_id is not None:
        result["recordings"].append(
            {
                "id": current_recording_id,
                "summary": current_recording_summary or "",
                "content": "\n".join(current_recording_content).strip(),
            }
        )

    return result


def format_manifest(tldr: str, recordings: list[dict]) -> str:
    """将结构化数据格式化为 manifest.md 内容。

    Args:
        tldr: 总体摘要
        recordings: 包含 'summary' 和 'content' 键的字典列表

    Returns:
        格式化的 markdown 内容
    """
    lines = [
        f"# Manifest",
        "",
        "## TL;DR",
        "",
        tldr.strip(),
        "",
        "## Recordings",
        "",
    ]

    for i, recording in enumerate(recordings, start=1):
        lines.append(f"{i}. Recording {i}: {recording['summary']}")
        lines.append(recording["content"])
        lines.append("")

    return "\n".join(lines)


def get_next_recording_id(workspace: Path, node_id: str) -> int:
    """获取下一个可用的 recording ID。

    Args:
        workspace: 工作空间路径
        node_id: 节点 ID

    Returns:
        下一个 recording ID（如果没有 recordings 则返回 1）
    """
    content = read_node_manifest(workspace, node_id)
    parsed = parse_manifest(content)

    if not parsed["recordings"]:
        return 1

    return max(r["id"] for r in parsed["recordings"]) + 1


# =============================================================================
# Update Self Memories
# =============================================================================



def update_current_summary(workspace: str, node_id: str, summary: str) -> str:
    """更新当前节点的总体摘要（TL;DR 部分）。

    此操作会用新的摘要覆盖现有的 TL;DR 部分。
    所有 recordings 保持不变。

    Args:
        workspace: 工作空间目录路径
        node_id: 当前节点的 ID
        summary: 替换现有 TL;DR 的新总体摘要

    Returns:
        成功消息，包含更新后的 manifest 文件路径

    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    content = read_node_manifest(workspace_path, node_id)
    parsed = parse_manifest(content)

    # Update TL;DR
    new_content = format_manifest(summary, parsed["recordings"])
    return write_node_manifest(workspace_path, node_id, new_content, mode="overwrite")



def append_current_recordings(
    workspace: str, node_id: str, recording_summary: str, recording_content: str
) -> str:
    """向当前节点的 manifest 添加新的 recording。

    recording 会自动分配下一个可用的 ID。

    Args:
        workspace: 工作空间目录路径
        node_id: 当前节点的 ID
        recording_summary: 此 recording 的简要摘要（显示在 "Recording N:" 之后）
        recording_content: recording 的详细内容

    Returns:
        成功消息，包含新的 recording ID
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    content = read_node_manifest(workspace_path, node_id)
    parsed = parse_manifest(content)

    # Add new recording
    new_recording = {"summary": recording_summary, "content": recording_content}
    parsed["recordings"].append(new_recording)

    new_content = format_manifest(parsed["tldr"], parsed["recordings"])
    return write_node_manifest(workspace_path, node_id, new_content, mode="overwrite")



def delete_current_recordings(workspace: str, node_id: str, recording_id: int) -> str:
    """从当前节点的 manifest 中删除指定的 recording。

    Args:
        workspace: 工作空间目录路径
        node_id: 当前节点的 ID
        recording_id: 要删除的 recording ID

    Returns:
        成功/错误消息
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    content = read_node_manifest(workspace_path, node_id)
    parsed = parse_manifest(content)

    # Find and remove the recording
    original_count = len(parsed["recordings"])
    parsed["recordings"] = [r for r in parsed["recordings"] if r["id"] != recording_id]

    if len(parsed["recordings"]) == original_count:
        return f"Error: Recording {recording_id} not found in node {node_id}"

    # Reformat content with remaining recordings
    new_content = format_manifest(parsed["tldr"], parsed["recordings"])
    return write_node_manifest(workspace_path, node_id, new_content, mode="overwrite")



def modify_current_recordings(
    workspace: str,
    node_id: str,
    recording_id: int,
    recording_summary: str | None = None,
    recording_content: str | None = None,
) -> str:
    """修改指定 recording 的 summary 和/或 content。

    参数为 None 时保持不变。

    Args:
        workspace: 工作空间目录路径
        node_id: 当前节点的 ID
        recording_id: 要修改的 recording ID
        recording_summary: 新的 summary（为 None 则保持现有值）
        recording_content: 新的 content（为 None 则保持现有值）

    Returns:
        成功/错误消息
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    content = read_node_manifest(workspace_path, node_id)
    parsed = parse_manifest(content)

    # Find the recording
    recording = next((r for r in parsed["recordings"] if r["id"] == recording_id), None)
    if recording is None:
        return f"Error: Recording {recording_id} not found in node {node_id}"

    # Update fields if provided
    if recording_summary is not None:
        recording["summary"] = recording_summary
    if recording_content is not None:
        recording["content"] = recording_content

    # Reformat content
    new_content = format_manifest(parsed["tldr"], parsed["recordings"])
    return write_node_manifest(workspace_path, node_id, new_content, mode="overwrite")


# =============================================================================
# Reading Other's Memories
# =============================================================================



def get_current_tree(workspace: str) -> str:
    """获取从根节点开始的完整内存树可视化表示。

    显示所有节点及其 ID 的层次结构。

    Args:
        workspace: 工作空间目录路径

    Returns:
        内存树结构的 ASCII 树形可视化
    """
    _validate_workspace(workspace)
    import subprocess

    workspace_path = Path(workspace)
    memory_root = get_memory_tree_root(workspace_path)

    if not memory_root.exists():
        return f"Error: Memory tree not found at {memory_root}"

    try:
        # 使用 tree 命令，-I 参数排除 storage 目录
        result = subprocess.run(
            ["tree", "-I", "storage", "-I", "manifest.md", str(memory_root)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout
        else:
            # tree 命令执行失败，返回错误信息
            return f"Memory tree at: {memory_root}\ntree command failed with error:\n{result.stderr}"
    except FileNotFoundError:
        # tree 命令未安装
        return f"Memory tree at: {memory_root}\n(tree command not found. Install with: apt install tree)"



def get_all_manifest(workspace: str) -> str:
    """获取内存树中所有 manifest 的摘要（TL;DR）。

    这提供了所有节点总体摘要的快速概览，
    无需加载完整内容。

    Args:
        workspace: 工作空间目录路径

    Returns:
        格式化的字符串，包含所有节点 ID 及其 TL;DR 摘要
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    memory_root = get_memory_tree_root(workspace_path)

    if not memory_root.exists():
        return f"Error: Memory tree not found at {memory_root}"

    results = []

    def collect_manifests(path: Path):
        """Recursively collect all manifests."""
        # Check for manifest in current directory
        manifest_path = path / "manifest.md"
        if manifest_path.exists() and path != memory_root:
            node_id = path.name
            content = manifest_path.read_text(encoding="utf-8")
            parsed = parse_manifest(content)
            results.append({"node_id": node_id, "tldr": parsed["tldr"].strip()})

        # Recurse into subdirectories (excluding storage)
        for item in path.iterdir():
            if item.is_dir() and item.name != "storage":
                collect_manifests(item)

    collect_manifests(memory_root)

    if not results:
        return "No manifests found in memory tree"

    output_lines = []
    for result in results:
        output_lines.append(f"Node: {result['node_id']}")
        output_lines.append(f"TL;DR: {result['tldr']}")
        output_lines.append("")

    return "\n".join(output_lines)



def get_parent_manifest(workspace: str, node_id: str) -> str:
    """获取父节点的完整 manifest 内容。

    Args:
        workspace: 工作空间目录路径
        node_id: 当前节点的 ID

    Returns:
        父节点的完整 manifest 内容，如果不存在父节点则返回错误
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    parent_id = get_parent_id(workspace_path, node_id)

    if parent_id is None:
        return f"Node '{node_id}' is a root node and has no parent"

    return read_node_manifest(workspace_path, parent_id)



def get_manifest_summary(workspace: str, node_id: str) -> str:
    """获取节点 manifest 的摘要（仅 TL;DR 和 recording 摘要）。

    这提供了 manifest 的结构，不包含完整的 recording 内容。

    Args:
        workspace: 工作空间目录路径
        node_id: 要查询的节点 ID

    Returns:
        格式化的字符串，包含 TL;DR 和 recording 摘要
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    content = read_node_manifest(workspace_path, node_id)

    if not content.strip():
        return f"Manifest for node '{node_id}' is empty or does not exist"

    parsed = parse_manifest(content)

    lines = [
        f"Node: {node_id}",
        "",
        "TL;DR:",
        parsed["tldr"].strip(),
        "",
        "Recordings:",
    ]

    for recording in parsed["recordings"]:
        lines.append(
            f"{recording['id']}. Recording {recording['id']}: {recording['summary']}"
        )

    return "\n".join(lines)



def get_manifest_all(workspace: str, node_id: str) -> str:
    """获取节点的完整 manifest 内容。

    Args:
        workspace: 工作空间目录路径
        node_id: 要查询的节点 ID

    Returns:
        完整的 manifest.md 内容，包括所有 recording 详情
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    content = read_node_manifest(workspace_path, node_id)

    if not content.strip():
        return f"Manifest for node '{node_id}' is empty or does not exist"

    return content


# =============================================================================
# Storage Access
# =============================================================================



def get_node_code(workspace: str, node_id: str) -> str:
    """获取节点的存储的 Python 代码。

    Args:
        workspace: 工作空间目录路径
        node_id: 节点 ID

    Returns:
        指定节点的 storage/code.py 内容
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    return read_storage_file(workspace_path, node_id, "code.py")



def get_node_output(workspace: str, node_id: str) -> str:
    """获取节点执行的标准输出。

    Args:
        workspace: 工作空间目录路径
        node_id: 节点 ID

    Returns:
        指定节点的 storage/stdout.txt 内容
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    return read_storage_file(workspace_path, node_id, "stdout.txt")



def get_node_trajectory(workspace: str, node_id: str) -> str:
    """获取节点的轨迹（对话历史）。

    Args:
        workspace: 工作空间目录路径
        node_id: 节点 ID

    Returns:
        指定节点的 storage/trajectory.json 内容
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    return read_storage_file(workspace_path, node_id, "trajectory.json")



def list_children(workspace: str, node_id: str) -> str:
    """列出给定节点的所有子节点 ID。

    Args:
        workspace: 工作空间目录路径
        node_id: 父节点 ID

    Returns:
        逗号分隔的子节点 ID 列表
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    children = list_child_nodes(workspace_path, node_id)

    if not children:
        return f"Node '{node_id}' has no children"

    return ", ".join(children)


# =============================================================================
# Global Memory Tools
# =============================================================================



def read_global_memory(workspace: str) -> str:
    """读取全局记忆文件内容。

    Args:
        workspace: 工作空间目录路径

    Returns:
        global_memory.md 的完整内容
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    return read_global_memory_impl(workspace_path)



def add_global_memory(
    workspace: str, memory_summary: str, memory_content: str
) -> str:
    """向全局记忆文件添加新的记忆条目。

    Args:
        workspace: 工作空间目录路径
        memory_summary: 记忆条目的简要标题
        memory_content: 记忆条目的详细内容

    Returns:
        操作结果消息
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    return add_global_memory_impl(workspace_path, memory_summary, memory_content)


# =============================================================================
# Data Link Tools
# =============================================================================



def show_all_data(workspace: str) -> str:
    """展示所有数据集的摘要信息。

    包含每个数据集的初始描述和最多 3 条最近评论。

    Args:
        workspace: 工作空间目录路径

    Returns:
        格式化的数据集摘要字符串
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    return show_all_data_impl(workspace_path)



def show_detailed_data(workspace: str, dataset_id: int) -> str:
    """展示指定数据集的详细信息。

    包含数据集的所有评论记录。

    Args:
        workspace: 工作空间目录路径
        dataset_id: 数据集 ID

    Returns:
        格式化的数据集详细信息字符串
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    return show_detailed_data_impl(workspace_path, dataset_id)



def add_new_data(workspace: str, dataset_path: str, init_descriptions: str) -> str:
    """添加新的数据集记录。

    Args:
        workspace: 工作空间目录路径
        dataset_path: 数据集路径（绝对路径）
        init_descriptions: 初始描述

    Returns:
        操作结果消息
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    return add_new_data_impl(workspace_path, dataset_path, init_descriptions)



def add_data_record(
    workspace: str, dataset_id: int, node_id: str, comment: str
) -> str:
    """为数据集添加评论记录。

    Args:
        workspace: 工作空间目录路径
        dataset_id: 数据集 ID
        node_id: 节点 ID
        comment: 评论内容

    Returns:
        操作结果消息
    """
    _validate_workspace(workspace)
    workspace_path = Path(workspace)
    return add_data_record_impl(workspace_path, dataset_id, node_id, comment)


