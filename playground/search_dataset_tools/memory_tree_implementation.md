# Memory Tree 实现总结

## 实现概述

已成功实现 **manifest.py** 第一点：**把 memory 变成一个文件树**。

## 文件结构

```
workspace/memory_tree/
├── {initial_node_id}/
│   ├── manifest.md           # 由 agent 通过工具写入（初始为空）
│   └── storage/
│       ├── trajectory.json   # 多轮对话记录（系统自动保存）
│       ├── code.py           # 节点代码备份
│       ├── stdout.txt        # 运行结果和控制台输出
│       └── submission.csv    # submission 文件备份
│   ├── {black_child1}/       # 子节点
│   │   ├── manifest.md
│   │   └── storage/
│   │   └── {grandchild}/     # 孙节点（递归结构）
│   │       ├── manifest.md
│   │       └── storage/
│   └── {red_child1}/
│       ├── manifest.md
│       └── storage/
```

## 核心文件

### 1. `playground/search_dataset_tools/memory_tree.py` - 核心模块

提供以下功能：

#### 路径获取函数
- `get_memory_tree_root(workspace)` - 获取 memory_tree 根目录
- `get_node_memory_path(workspace, node_id)` - 获取节点文件夹路径
- `get_node_storage_path(workspace, node_id)` - 获取 storage 文件夹路径
- `get_node_manifest_path(workspace, node_id)` - 获取 manifest.md 文件路径

#### 节点操作函数
- `create_node_memory(workspace, node_id, parent_id=None)` - 创建节点文件夹结构
- `save_node_storage(workspace, node_id, trajectory, code, stdout, submission_path)` - 保存节点执行数据

#### Manifest 操作函数
- `read_node_manifest(workspace, node_id)` - 读取 manifest 内容
- `write_node_manifest(workspace, node_id, content, mode="overwrite")` - 写入 manifest

#### 辅助函数
- `find_node_path(workspace, node_id)` - 在树中查找节点的完整路径
- `list_child_nodes(workspace, node_id)` - 列出节点的所有子节点
- `get_parent_id(workspace, node_id)` - 获取父节点 ID
- `read_storage_file(workspace, node_id, file_name)` - 读取 storage 中的文件
- `summarize_node_memory(workspace, node_id)` - 获取节点的完整 memory 总结

### 2. `playground/ml_master_datatree/core/playground.py` - 集成点

已集成到以下位置：

#### 节点创建时（创建 memory 文件夹结构）
- `_create_expand_node_fn()` (line 386-389) - 事件驱动扩展时创建
- `_batch_expand_after_initial()` (line 496-498) - Initial 完成后批量创建子节点

#### 节点完成时（保存 storage 数据）
- `_execute_and_process_node()` (line 573-595) - 节点执行完成后保存数据

## 设计特点

### 1. 树形结构与 UCT 搜索树一一对应
- 每个 UCTNode 都有对应的文件系统文件夹
- 父子关系通过文件夹嵌套体现

### 2. 职责分离
- **manifest.md**: Agent 主动写入的全局 memory 总结（初始为空）
- **storage/**: 系统自动保存的执行数据

### 3. 自动化数据保存
- 系统在节点创建时自动创建文件夹结构
- 系统在节点完成时自动保存 storage 数据
- Agent 无需关心文件系统操作

### 4. 智能节点查找
- `find_node_path()` 使用 BFS + 深度优先策略
- 当有多个同名节点时，选择深度最大的（最具体的）

## 使用示例

### 创建节点 memory

```python
from search_dataset_tools.memory_tree import create_node_memory
from pathlib import Path

workspace = Path("/path/to/workspace")

# 创建 initial 节点（无父节点）
initial_path = create_node_memory(workspace, "initial_abc123", parent_id=None)

# 创建子节点
child_path = create_node_memory(workspace, "black_xyz", parent_id="initial_abc123")
```

### 保存节点数据

```python
from search_dataset_tools.memory_tree import save_node_storage

# 获取 agent trajectory
trajectory_dict = agent.trajectory.model_dump()

# 保存到 storage
save_node_storage(
    workspace=workspace,
    node_id="black_xyz",
    trajectory=trajectory_dict,
    code="print('hello')",
    stdout="hello\n",
    submission_path=Path("/path/to/submission.csv")
)
```

### 读取 manifest

```python
from search_dataset_tools.memory_tree import read_node_manifest, write_node_manifest

# 读取
content = read_node_manifest(workspace, "black_xyz")

# 写入
write_node_manifest(
    workspace, "black_xyz",
    content="# My Plan\n\nThis is my plan.",
    mode="overwrite"
)

# 追加
write_node_manifest(
    workspace, "black_xyz",
    content="\n## Update\n\nPlan updated.",
    mode="append"
)
```

## 验证

运行测试脚本验证功能：

```bash
python playground/search_dataset_tools/test_memory_tree.py
```

所有测试应通过，输出：
```
✅ 所有测试通过！
```

## 下一步

为实现 manifest.py 的第二点（设计工具接口）和第三点（实现工具接口），已做好准备：

### 工具接口设计思路
基于 FastMCP 框架，提供以下工具：
- `read_manifest(node_id, workspace)` - 读取节点的 manifest.md
- `write_manifest(node_id, content, workspace, mode)` - 写入 manifest.md
- `read_storage(node_id, file_name, workspace)` - 读取 storage 文件
- `list_children(node_id, workspace)` - 列出子节点
- `read_parent_manifest(node_id, workspace)` - 读取父节点 manifest
- `summarize_memory(node_id, workspace)` - 获取完整 memory 总结

这些工具将允许 Agent 动态读取和更新 memory，而不是在 prompt 构建时静态注入。

## 注意事项

1. **并发访问**: 当前实现未处理并发写入冲突。如果使用并行 workers，每个 worker 应该有独立的 workspace（exp_0/, exp_1/）

2. **节点 ID 唯一性**: 系统依赖 UCTNode 的 UUID 作为 node_id，确保唯一性

3. **父节点查找**: `find_node_path()` 使用 BFS 搜索，在节点数量很多时可能有性能影响。实际使用中节点数量通常可控（<100）

4. **存储空间**: storage 文件可能很大（尤其是 trajectory.json 和 submission.csv），需要定期清理策略
