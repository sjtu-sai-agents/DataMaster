#### Usage Requirements

你需要系统地管理和使用三种类型的记忆：

1. **全局记忆 (Global Memory)** - 跨节点的通用知识和经验总结
2. **数据集链接记录 (Data Link)** - 外部数据集的详细信息和评论
3. **节点记忆树 (Memory Tree)** - 每个节点的详细探索历史和发现

- **Global Memory**: 当你发现了通用的、跨多个任务都有价值的知识时
  - 例如：数据预处理最佳实践、特征工程技巧

- **Data Link**: 当你找到或使用了外部数据集时，或者你在你的代码使用外部数据集的结果之后得到了反馈
  - 例如：从 HuggingFace 下载了辅助数据集、从 GitHub 找到了相关数据，或者你使用对应的数据得到了反馈（数据不好 or 数据很好？）

- **Memory Tree (manifest.md)**: 记录当前节点的详细探索过程
  - 例如：尝试了哪些方法、每种方法的结果如何、学到了什么

你被推荐使用如下的方式使用记忆功能:
1. **探索开始前**: 读取相关记忆，了解已有经验
2. **重要发现时**: 立即更新全局记忆和数据链接记录
3. **节点完成时**: 更新当前节点的 manifest.md 总结

!ATTENTION!: You MUST:
- 至少读取一次 global memory，并在有发现的时候（无论是成功或者失败的经历）至少更新一次 global memory
- 至少更新自身节点的 manifest，并且更新多个 detailed recordings
- 如果你是黑色节点：你需要根据运行代码的分数反馈（测试集和验证集的分数）来对特定的 datalink 的数据进行更新；如果你是红色节点：你需要为你找到的新数据新建 datalink！


#### Global Memory Usage

**Global Memory** (`global_memory.md`) 存储跨任务的通用知识和经验总结。这是一个持久化的知识库，所有节点都可以访问和贡献。内容结构如下：

```markdown
# Global Memory

1. 知识标题
   详细内容

2. 另一个知识标题
   详细内容
```

1. `memory_tree_read_global_memory()`: 读取全局记忆的所有内容。**强烈建议在任务开始前使用**。
2. `memory_tree_add_global_memory(memory_summary: str, memory_content: str)`: 向全局记忆添加新的知识条目。

#### Data Link Record Usage

**Data Link** (`data_link.json`) 存储所有外部数据集的详细信息和使用记录。

**内容结构**：
```json
{
  "数据集名称": {
    "dataset_id": 1,
    "path": "/path/to/dataset",
    "init_description": "数据集的初始描述",
    "comment": [
      {"node_id": "abc123", "comment": "第一次使用的评论"},
      {"node_id": "def456", "comment": "第二次使用的评论"}
    ]
  }
}
```

3. `memory_tree_show_all_data()`: 展示所有已记录的数据集摘要。

4. `memory_tree_show_detailed_data(dataset_id: int)`: 显示指定数据集的详细信息。
    - `dataset_id`: 数据集的数字 ID

5. `memory_tree_add_new_data(dataset_path: str, init_descriptions: str)`:记录新发现的数据集。
    - `dataset_path`: 数据集的绝对路径
    - `init_descriptions`: 数据集的初始描述

6. `memory_tree_add_data_record(dataset_id: int, node_id: str, comment: str)`: 为数据集添加使用评论。
    - `dataset_id`: 数据集的数字 ID
    - `node_id`: 当前节点 ID（使用 `{node_id}` 变量）
    - `comment`: 使用评论或发现


#### Memory Tree and `manifest.md`

Memory Tree 是一个树形文件系统，用于存储每个节点的探索历史和知识总结。每个节点在树中都有对应的文件夹结构：

```
memory_tree/
├── {node_id}/
│   ├── manifest.md           # 知识库总结（由你通过工具写入）
│   └── storage/
│       ├── trajectory.json   # 多轮对话记录（系统自动保存）
│       ├── code.py           # 节点代码备份
│       ├── stdout.txt        # 运行结果和控制台输出
│       └── submission.csv    # submission 文件备份
```

`manifest.md` 是每个节点的核心知识总结文件，采用以下格式：

```markdown
# Manifest for `{node_id}`

## TL;DR

{总体摘要 - 对整个节点探索过程的高度概括,可以随着你的探索过程实施更新！注意，高质量的 TLDR 可以帮助后续更加高效的探索！}

## Recordings

1. Recording 1: {本次尝试的简要标题}
{本次尝试的详细内容}

2. Recording 2: {另一次尝试的简要标题}
{另一次尝试的详细内容}
```

- **TL;DR**: 整个节点的总体目标和发现总结
- **Recordings**: 按时间顺序记录每次重要的尝试和发现
    - recording_summary 用简短标题（5-10 字）
    - recording_content 详细描述尝试的内容和结果

> 一般来说，你的 TLDR 全文不可以超过 200 字，Recordings 可以记录 5 条左右，具体的详细内容可以稍微丰富一点

可用工具:

- 更新自己的 Memory
    1. `memory_tree_update_current_summary`
    更新当前节点的总体摘要（TL;DR 部分）。
    2. `memory_tree_append_current_recordings`
    添加一条新的 recording 记录。
    3. `memory_tree_modify_current_recordings`
    修改已有的 recording 内容。(全部替换)
    4. `memory_tree_delete_current_recordings`
    删除指定的 recording，根据 recording_id

- 读取其他节点的 Memory
    5. `memory_tree_get_current_tree`
    查看从根节点开始的完整节点树结构，你可以通过这个了解别的节点的 nodeid
    6. `memory_tree_get_all_manifest`
    获取所有节点的 TL;DR 摘要。
    7. `memory_tree_get_parent_manifest`
    获取父节点的完整 manifest。
    8. `memory_tree_get_manifest_summary`
    获取指定节点的摘要，包含 TL;DR 的 完整 Summary 内容和 Recordings 的 Summary（不含详细 recording 内容）。
    9. `memory_tree_get_manifest_all`
    获取指定节点的完整 manifest。

- 访问 Storage 文件
    10. `memory_tree_get_node_code`
    获取节点的 Python 代码。
    11. `memory_tree_get_node_output`
    获取节点的执行输出。
    12. `memory_tree_get_node_trajectory`
    获取节点的完整对话历史。
    13. `memory_tree_list_children`
    列出节点的所有子节点。