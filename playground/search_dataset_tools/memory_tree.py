"""Memory Tree - 树形文件系统形式的节点 memory 管理。

每个节点都有对应的文件夹结构：
memory_tree/
├── {node_id}/
│   ├── manifest.md           # 由 agent 通过工具写入（初始为空）
│   └── storage/
│       ├── trajectory.json   # 多轮对话记录（系统自动保存）
│       ├── code.py           # 节点代码备份
│       ├── stdout.txt        # 运行结果和控制台输出
│       └── submission.csv    # submission 文件备份
├── global_memory.md          # 全局记忆文件
└── data_link.json            # 数据集链接记录

与 UCT 搜索树结构一一对应，支持持久化存储和检索。
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


def find_node_path(workspace: Path, node_id: str) -> Path | None:
    """在 memory_tree 中查找节点的完整路径。

    递归搜索 memory_tree/ 目录，找到匹配 node_id 的文件夹。
    优先返回深度更大的路径（更具体的节点）。

    Args:
        workspace: 工作空间路径
        node_id: 节点 ID

    Returns:
        节点的完整路径，如果找不到返回 None
    """
    memory_root = get_memory_tree_root(workspace)
    if not memory_root.exists():
        return None

    # 收集所有匹配的路径，返回深度最大的
    queue = deque([memory_root])
    matches = []

    while queue:
        current_dir = queue.popleft()
        if current_dir.name == node_id and current_dir != memory_root:
            matches.append(current_dir)

        # 添加子目录到队列（排除 storage）
        for item in current_dir.iterdir():
            if item.is_dir() and item.name != "storage":
                queue.append(item)

    if not matches:
        return None

    # 返回深度最大的匹配路径（相对于 memory_root 的路径长度）
    def get_depth(path: Path) -> int:
        return len(path.relative_to(memory_root).parts)

    return max(matches, key=get_depth)


def get_memory_tree_root(workspace: Path) -> Path:
    """获取 memory tree 的根目录。

    Args:
        workspace: 工作空间路径

    Returns:
        memory_tree/ 文件夹路径
    """
    return workspace / "memory_tree"


def get_node_memory_path(workspace: Path, node_id: str) -> Path:
    """获取节点的 memory 文件夹路径。

    Args:
        workspace: 工作空间路径
        node_id: 节点 ID

    Returns:
        节点的 memory 文件夹路径
        如果找到则返回完整路径（如 workspace/memory_tree/initial_id/child_id/）
        如果找不到则返回默认路径（workspace/memory_tree/node_id/）
    """
    # 先尝试查找节点的实际位置（处理嵌套节点）
    actual_path = find_node_path(workspace, node_id)
    if actual_path is not None:
        return actual_path

    # 如果找不到，返回默认路径（兼容旧代码）
    logger.error(f"Error, {node_id} memory path not found!")
    return get_memory_tree_root(workspace) / node_id


def get_node_storage_path(workspace: Path, node_id: str) -> Path:
    """获取节点的 storage 文件夹路径。

    Args:
        workspace: 工作空间路径
        node_id: 节点 ID

    Returns:
        节点的 storage 文件夹路径 (workspace/memory_tree/{node_id}/storage/)
    """
    return get_node_memory_path(workspace, node_id) / "storage"


def get_node_manifest_path(workspace: Path, node_id: str) -> Path:
    """获取节点的 manifest.md 文件路径。

    Args:
        workspace: 工作空间路径
        node_id: 节点 ID

    Returns:
        节点的 manifest.md 文件路径 (workspace/memory_tree/{node_id}/manifest.md)
    """
    return get_node_memory_path(workspace, node_id) / "manifest.md"


def create_node_memory(
    workspace: Path,
    node_id: str,
    parent_id: str | None = None,
) -> Path:
    """创建节点的 memory 文件夹结构。

    创建以下结构：
    - memory_tree/{parent_id}/{node_id}/
    - memory_tree/{parent_id}/{node_id}/manifest.md (空文件)
    - memory_tree/{parent_id}/{node_id}/storage/

    如果 parent_id 为 None，直接在 memory_tree/ 下创建（用于 initial 节点）。

    Args:
        workspace: 工作空间路径
        node_id: 节点 ID
        parent_id: 父节点 ID（如果为 None，直接在 memory_tree/ 下创建）

    Returns:
        创建的节点 memory 路径

    Raises:
        FileNotFoundError: 如果父节点存在但找不到
    """
    memory_root = get_memory_tree_root(workspace)

    # 首先检查节点是否已经存在（避免重复创建）
    existing_path = find_node_path(workspace, node_id)
    if existing_path is not None:
        logger.info(
            f"Node {node_id} already exists at {existing_path}, skipping creation"
        )
        return existing_path

    # 确定父路径
    if parent_id is None:
        # initial 节点，直接在 memory_root/ 下创建
        # 需要 parents=True 来创建 memory_root
        parent_path = memory_root
        node_path = memory_root / node_id
        node_path.mkdir(parents=True, exist_ok=True)
    else:
        # 子节点，需要找到父节点的完整路径
        parent_path = find_node_path(workspace, parent_id)
        if parent_path is None:
            raise FileNotFoundError(
                f"Parent node '{parent_id}' not found in memory_tree. "
                f"Cannot create child node '{node_id}'. "
                f"Make sure parent node is created first."
            )

        node_path = parent_path / node_id
        # 不使用 parents=True，只在已存在的父节点下创建
        node_path.mkdir(exist_ok=True)

    # 创建空的 manifest.md
    manifest_path = node_path / "manifest.md"
    if not manifest_path.exists():
        manifest_path.write_text("", encoding="utf-8")
        logger.info(f"Created empty manifest: {manifest_path}")

    # 创建 storage 文件夹
    storage_path = node_path / "storage"
    if not storage_path.exists():
        storage_path.mkdir(exist_ok=True)
        logger.info(f"Created storage directory: {storage_path}")

    logger.info(f"Created node memory structure: {node_path}")
    return node_path


def save_node_storage(
    workspace: Path,
    node_id: str,
    trajectory: dict[str, Any] | list,
    code: str,
    stdout: str,
    submission_path: Path | None = None,
) -> None:
    """保存节点执行数据到 storage/ 文件夹。

    保存以下文件：
    - storage/trajectory.json: 多轮对话记录
    - storage/code.py: 节点代码
    - storage/stdout.txt: 运行结果和控制台输出
    - storage/submission.csv: submission 文件备份（如果存在）

    Args:
        workspace: 工作空间路径
        node_id: 节点 ID
        trajectory: 多轮对话记录（dict 或 list）
        code: 节点代码
        stdout: 运行结果和控制台输出
        submission_path: submission 文件路径（如果存在）
    """
    storage_path = get_node_storage_path(workspace, node_id)

    # 确保 storage 文件夹存在
    storage_path.mkdir(parents=True, exist_ok=True)

    # 保存 trajectory.json
    trajectory_path = storage_path / "trajectory.json"
    try:
        with trajectory_path.open("w", encoding="utf-8") as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved trajectory: {trajectory_path}")
    except Exception as e:
        logger.error(f"Failed to save trajectory: {e}")

    # 保存 code.py
    code_path = storage_path / "code.py"
    try:
        code_path.write_text(code, encoding="utf-8")
        logger.info(f"Saved code: {code_path}")
    except Exception as e:
        logger.error(f"Failed to save code: {e}")

    # 保存 stdout.txt
    stdout_path = storage_path / "stdout.txt"
    try:
        stdout_path.write_text(stdout, encoding="utf-8")
        logger.info(f"Saved stdout: {stdout_path}")
    except Exception as e:
        logger.error(f"Failed to save stdout: {e}")

    # 复制 submission.csv（如果存在）
    if submission_path and submission_path.exists():
        submission_backup_path = storage_path / "submission.csv"
        try:
            shutil.copy(submission_path, submission_backup_path)
            logger.info(f"Saved submission: {submission_backup_path}")
        except Exception as e:
            logger.error(f"Failed to save submission: {e}")


def read_node_manifest(workspace: Path, node_id: str) -> str:
    """读取节点的 manifest.md 内容。

    Args:
        workspace: 工作空间路径
        node_id: 节点 ID

    Returns:
        manifest.md 的内容，如果文件不存在返回空字符串
    """
    manifest_path = get_node_manifest_path(workspace, node_id)
    if not manifest_path.exists():
        logger.warning(f"Manifest not found: {manifest_path}")
        return ""
    return manifest_path.read_text(encoding="utf-8")


def write_node_manifest(
    workspace: Path,
    node_id: str,
    content: str,
    mode: str = "overwrite",
) -> str:
    """写入节点的 manifest.md。

    Args:
        workspace: 工作空间路径
        node_id: 节点 ID
        content: 要写入的内容
        mode: 写入模式
            - "overwrite": 覆盖整个文件
            - "append": 追加到文件末尾
            - "prepend": 插入到文件开头

    Returns:
        操作结果消息
    """
    manifest_path = get_node_manifest_path(workspace, node_id)

    # 确保父目录存在
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if mode == "overwrite":
            manifest_path.write_text(content, encoding="utf-8")
        elif mode == "append":
            existing = (
                manifest_path.read_text(encoding="utf-8")
                if manifest_path.exists()
                else ""
            )
            manifest_path.write_text(existing + content, encoding="utf-8")
        elif mode == "prepend":
            existing = (
                manifest_path.read_text(encoding="utf-8")
                if manifest_path.exists()
                else ""
            )
            manifest_path.write_text(content + existing, encoding="utf-8")
        else:
            return f"Error: Invalid mode '{mode}', must be 'overwrite', 'append', or 'prepend'"

        logger.info(f"Updated manifest: {manifest_path} (mode={mode})")
        return f"Successfully updated manifest: {manifest_path}"
    except Exception as e:
        logger.error(f"Failed to write manifest: {e}")
        return f"Error: Failed to write manifest: {e}"


def list_child_nodes(workspace: Path, node_id: str) -> list[str]:
    """列出节点的所有子节点。

    Args:
        workspace: 工作空间路径
        node_id: 父节点 ID

    Returns:
        子节点 ID 列表
    """
    node_path = get_node_memory_path(workspace, node_id)
    if not node_path.exists():
        logger.warning(f"Node path not found: {node_path}")
        return []

    children = []
    for item in node_path.iterdir():
        if item.is_dir() and item.name != "storage":
            children.append(item.name)

    return sorted(children)


def get_parent_id(workspace: Path, node_id: str) -> str | None:
    """获取节点的父节点 ID。

    通过检查节点的父文件夹名称来确定。

    Args:
        workspace: 工作空间路径
        node_id: 节点 ID

    Returns:
        父节点 ID，如果是顶层节点（在 memory_tree/ 下）返回 None
    """
    # 使用 find_node_path 而不是 get_node_memory_path，因为节点可能是嵌套的
    node_path = find_node_path(workspace, node_id)
    if node_path is None:
        # 如果找不到，尝试使用简单路径
        node_path = get_node_memory_path(workspace, node_id)
        if not node_path.exists():
            return None

    parent_path = node_path.parent
    memory_root = get_memory_tree_root(workspace)

    # 如果父节点是 memory_root，说明是顶层节点
    if parent_path == memory_root:
        return None

    # 否则返回父文件夹名称
    return parent_path.name


def read_storage_file(workspace: Path, node_id: str, file_name: str) -> str:
    """读取节点 storage 文件夹中的文件。

    Args:
        workspace: 工作空间路径
        node_id: 节点 ID
        file_name: 文件名 (trajectory.json, code.py, stdout.txt, submission.csv)

    Returns:
        文件内容，如果文件不存在返回错误消息
    """
    storage_path = get_node_storage_path(workspace, node_id)
    file_path = storage_path / file_name

    if not file_path.exists():
        return f"Error: File not found: {file_path}"

    try:
        if file_name.endswith(".json"):
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return json.dumps(data, ensure_ascii=False, indent=2)
        else:
            return file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read storage file {file_path}: {e}")
        return f"Error: Failed to read file: {e}"


def summarize_node_memory(
    workspace: Path,
    node_id: str,
    include_parent: bool = True,
    include_children: bool = True,
) -> dict[str, Any]:
    """获取节点的完整 memory 总结。

    Args:
        workspace: 工作空间路径
        node_id: 节点 ID
        include_parent: 是否包含父节点 manifest
        include_children: 是否包含子节点 manifest

    Returns:
        包含节点完整 memory 的字典
    """
    summary = {
        "node_id": node_id,
        "manifest": read_node_manifest(workspace, node_id),
        "parent": None,
        "children": [],
    }

    # 获取父节点 manifest
    if include_parent:
        parent_id = get_parent_id(workspace, node_id)
        if parent_id:
            summary["parent"] = {
                "node_id": parent_id,
                "manifest": read_node_manifest(workspace, parent_id),
            }

    # 获取子节点 manifest
    if include_children:
        child_ids = list_child_nodes(workspace, node_id)
        for child_id in child_ids:
            summary["children"].append(
                {
                    "node_id": child_id,
                    "manifest": read_node_manifest(workspace, child_id),
                }
            )

    return summary


# =============================================================================
# Global Memory
# =============================================================================


def get_global_memory_path(workspace: Path) -> Path:
    """获取 global_memory.md 文件路径。

    Args:
        workspace: 工作空间路径

    Returns:
        global_memory.md 文件路径
    """
    return get_memory_tree_root(workspace) / "global_memory.md"


def init_global_memory(workspace: Path) -> Path:
    """初始化 global_memory.md 文件。

    如果文件不存在，则创建空文件。

    Args:
        workspace: 工作空间路径

    Returns:
        global_memory.md 文件路径
    """
    global_memory_path = get_global_memory_path(workspace)
    memory_root = get_memory_tree_root(workspace)

    # 确保 memory_tree 目录存在
    memory_root.mkdir(parents=True, exist_ok=True)

    # 如果文件不存在，创建空文件
    if not global_memory_path.exists():
        global_memory_path.write_text(
            "# Global Memory\n\n全局记忆文件，用于存储跨节点的通用知识和经验。\n",
            encoding="utf-8",
        )
        logger.info(f"Created global_memory: {global_memory_path}")

    return global_memory_path


def read_global_memory(workspace: Path) -> str:
    """读取 global_memory.md 内容。

    Args:
        workspace: 工作空间路径

    Returns:
        global_memory.md 的内容，如果文件不存在则初始化并返回空内容
    """
    global_memory_path = init_global_memory(workspace)
    return global_memory_path.read_text(encoding="utf-8")


def add_global_memory(
    workspace: Path, memory_summary: str, memory_content: str
) -> str:
    """向 global_memory.md 添加新的记忆条目。

    Args:
        workspace: 工作空间路径
        memory_summary: 记忆条目的简要标题
        memory_content: 记忆条目的详细内容

    Returns:
        操作结果消息
    """
    global_memory_path = init_global_memory(workspace)
    content = global_memory_path.read_text(encoding="utf-8")

    # 解析现有的记忆条目数量
    lines = content.split("\n")
    next_id = 1
    for line in lines:
        if line.strip().startswith(f"{next_id}. "):
            next_id += 1

    # 添加新的记忆条目
    new_entry = f"\n{next_id}. {memory_summary}\n{memory_content}\n"
    updated_content = content + new_entry

    try:
        global_memory_path.write_text(updated_content, encoding="utf-8")
        logger.info(f"Added global memory entry {next_id}")
        return f"Successfully added global memory entry {next_id}"
    except Exception as e:
        logger.error(f"Failed to add global memory: {e}")
        return f"Error: Failed to add global memory: {e}"


# =============================================================================
# Data Link Manifest
# =============================================================================


def get_data_link_path(workspace: Path) -> Path:
    """获取 data_link.json 文件路径。

    Args:
        workspace: 工作空间路径

    Returns:
        data_link.json 文件路径
    """
    return get_memory_tree_root(workspace) / "data_link.json"


def init_data_link(workspace: Path) -> Path:
    """初始化 data_link.json 文件。

    如果文件不存在，则创建空的数据结构。

    Args:
        workspace: 工作空间路径

    Returns:
        data_link.json 文件路径
    """
    data_link_path = get_data_link_path(workspace)
    memory_root = get_memory_tree_root(workspace)

    # 确保 memory_tree 目录存在
    memory_root.mkdir(parents=True, exist_ok=True)

    # 如果文件不存在，创建空数据结构
    if not data_link_path.exists():
        empty_data = {}
        try:
            with data_link_path.open("w", encoding="utf-8") as f:
                json.dump(empty_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Created data_link.json: {data_link_path}")
        except Exception as e:
            logger.error(f"Failed to create data_link.json: {e}")
            raise

    return data_link_path


def read_data_link(workspace: Path) -> dict[str, Any]:
    """读取 data_link.json 内容。

    Args:
        workspace: 工作空间路径

    Returns:
        data_link.json 的内容（字典格式）
    """
    data_link_path = init_data_link(workspace)

    try:
        with data_link_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read data_link.json: {e}")
        return {}


def write_data_link(workspace: Path, data: dict[str, Any]) -> str:
    """写入 data_link.json。

    Args:
        workspace: 工作空间路径
        data: 要写入的数据字典

    Returns:
        操作结果消息
    """
    data_link_path = get_data_link_path(workspace)

    try:
        with data_link_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Updated data_link.json: {data_link_path}")
        return f"Successfully updated data_link.json"
    except Exception as e:
        logger.error(f"Failed to write data_link.json: {e}")
        return f"Error: Failed to write data_link.json: {e}"


def show_all_data(workspace: Path) -> str:
    """展示所有数据集的摘要信息。

    Args:
        workspace: 工作空间路径

    Returns:
        格式化的数据集摘要字符串
    """
    data = read_data_link(workspace)

    if not data:
        return "No datasets found in data_link.json"

    output_lines = []
    for data_name, data_info in data.items():
        output_lines.append(f"Dataset: {data_name}")
        output_lines.append(f"  ID: {data_info.get('dataset_id', 'N/A')}")
        output_lines.append(f"  Path: {data_info.get('path', 'N/A')}")
        output_lines.append(f"  Init Description: {data_info.get('init_description', 'N/A')}")

        # 显示最多 3 条评论
        comments = data_info.get("comment", [])
        if comments:
            output_lines.append(f"  Recent Comments (showing max 3):")
            for comment in comments[:3]:
                output_lines.append(
                    f"    - Node {comment.get('node_id', 'Unknown')}: {comment.get('comment', '')}"
                )
            if len(comments) > 3:
                output_lines.append(f"    ... and {len(comments) - 3} more comments")
        else:
            output_lines.append("  Recent Comments: None")

        output_lines.append("")

    return "\n".join(output_lines)


def show_detailed_data(workspace: Path, dataset_id: int) -> str:
    """展示指定数据集的详细信息。

    Args:
        workspace: 工作空间路径
        dataset_id: 数据集 ID

    Returns:
        格式化的数据集详细信息字符串
    """
    data = read_data_link(workspace)

    # 查找匹配的数据集
    target_data = None
    target_name = None
    for data_name, data_info in data.items():
        if data_info.get("dataset_id") == dataset_id:
            target_data = data_info
            target_name = data_name
            break

    if target_data is None:
        return f"Dataset with ID {dataset_id} not found"

    output_lines = [
        f"Dataset: {target_name}",
        f"ID: {target_data.get('dataset_id', 'N/A')}",
        f"Path: {target_data.get('path', 'N/A')}",
        f"Init Description: {target_data.get('init_description', 'N/A')}",
        "",
        "All Comments:",
    ]

    comments = target_data.get("comment", [])
    if comments:
        for i, comment in enumerate(comments, 1):
            output_lines.append(
                f"  {i}. Node {comment.get('node_id', 'Unknown')}: {comment.get('comment', '')}"
            )
    else:
        output_lines.append("  No comments")

    return "\n".join(output_lines)


def add_new_data(
    workspace: Path, dataset_path: str, init_descriptions: str
) -> str:
    """添加新的数据集。

    Args:
        workspace: 工作空间路径
        dataset_path: 数据集路径（绝对路径）
        init_descriptions: 初始描述

    Returns:
        操作结果消息
    """
    data = read_data_link(workspace)

    # 生成新的 dataset_id（找到最大 ID + 1）
    max_id = 0
    for data_info in data.values():
        current_id = data_info.get("dataset_id", 0)
        if current_id > max_id:
            max_id = current_id

    new_id = max_id + 1
    # 使用数据集名称（路径的最后一部分）作为键
    data_name = Path(dataset_path).name

    # 检查是否已存在同名数据集
    if data_name in data:
        return f"Error: Dataset '{data_name}' already exists"

    # 添加新数据集
    data[data_name] = {
        "dataset_id": new_id,
        "path": dataset_path,
        "init_description": init_descriptions,
        "comment": [],
    }

    return write_data_link(workspace, data)


def add_data_record(
    workspace: Path, dataset_id: int, node_id: str, comment: str
) -> str:
    """为数据集添加评论记录。

    Args:
        workspace: 工作空间路径
        dataset_id: 数据集 ID
        node_id: 节点 ID
        comment: 评论内容

    Returns:
        操作结果消息
    """
    data = read_data_link(workspace)

    # 查找匹配的数据集
    target_name = None
    for data_name, data_info in data.items():
        if data_info.get("dataset_id") == dataset_id:
            target_name = data_name
            break

    if target_name is None:
        return f"Error: Dataset with ID {dataset_id} not found"

    # 添加新评论
    data[target_name]["comment"].append({"node_id": node_id, "comment": comment})

    return write_data_link(workspace, data)
