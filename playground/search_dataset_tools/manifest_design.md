# Manifest Design

## Introduction

每一个节点在 memory tree 中存储的结构如下：

```text
${PROJECT_ROOT}/runs/data_master_20260326_165235/workspaces/task_0/memory_tree
`-- 405189dacb8d424dacc94ea9b2e030cf
    |-- 243b68bdeb6143959bca328fce03825c
    |-- 55374a4c32074bacb50c6dff0b9a0186
    |-- 7e9df25e2a8847d8b337ae30b66adf9d
    |-- 82becc83359a48a4b581254f3976cb7a
    |-- 9de16a22167349a887e076e80fd8692b
    |-- c426b976841a450eb09dcf9460ac76f7
    |-- manifest.md
    `-- storage
```

- 对应文件夹是对应 node 的 `node_id`
- 该文件夹下存在若干也以 `node_id` 作为编号的子文件夹作为其子节点
- storage 文件下存储
    ```
    -- storage
            |-- code.py
            |-- stdout.txt
            |-- submission.csv
            `-- trajectory.json
    ```
- `manifest.md` 文件存储这个 node 的知识库总结，按照二阶段 Skills 的格式进行存储

## `manifest.md`

manifest.md 是每一个节点浓缩核心总结的一个 markdown 文件，由 agent 自己在探索过程中探索，尝试，更新，不同节点的 agent 也可以查看别人的 manifest 文件。

```text
# Manifest for `{node_id}`

## TL;DR

{overall_summary}

## Recordings

1. Recordings 1: {content_summary}
{recording_1_content}

2. Recordings 2: {content_summary}
{recording_1_content}

2. Recordings 3: {content_summary}
{recording_3_content}

```

你需要设计一套给 agent 的接口，让 agent 可以实时地存储自己的 memory，并且更加自由的导入，读取别人的 memory

### Updating Self Memories

- `update_current_summary(workspace: str, node_id: str, summary: str)`: 更新当前的 overall summary（注意，是直接覆盖）
- `append_current_recordings(workspace: str, node_id: str, recording_summary: str, recording_content: str)`: 添加一条 recordings，每一个 recording 有一个 `recording_id`，从 1 开始递增
- `delete_current_recordings(workspace: str, node_id: str, recording_id: int)`: 删除某一条 recordings
- `modify_current_recordings(workspace: str, node_id: str, recording_summary: str, recording_content: str, recording_id: int)`: 修改某一条 recordings 的内容

### Reading Other's Memories

- `get_current_tree(workspace: str, node_id: str)`: 可视化当前的 id 树
- `get_all_manifest(workspace: str)`: 获取所有 manifest 的摘要
    - 只导入每一个 menifest 的 overall_summary
- `get_parent_manifest(workspace: str, node_id: str)`: 获取祖先 manifest 的全部内容
- `get_manifest_summary(workspace: str, node_id: str)`: 查看对应的 node_id 的 manifest 的内容，包含所有 recording_summary & overall_summary
- `get_manifest_all(workspace: str, node_id: str)`: 查看对应的 node_id 的 manifest 的全部内容

## `storage`

- storage 文件下存储
    ```
    -- storage
            |-- code.py
            |-- stdout.txt
            |-- submission.csv
            `-- trajectory.json
    ```

因此，对应的工具包含：

- `get_node_code(workspace: str, node_id: str)`: 获取当前 code 的代码
- `get_node_output(workspace: str, node_id: str)`: 读取 stdout.txt