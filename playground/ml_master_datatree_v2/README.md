# ML Master DataTree v2

`ml_master_datatree_v2` 是对 v1 的一版简化重构，核心目标是把 `red` 和 `black` 的职责拆清楚，并避免并行时的数据竞争。

## 当前设计

- `initial` / `black` 节点不再直接长出一批 `black`
- 每个可继续优化的解法节点，先长出 `red`
- 每个成功的 `red`，再长出一批绑定到该 `red` 结果的 `black`

当前默认搜索形态是：

```text
initial
  -> red
    -> black_1
    -> black_2
    -> ...

black
  -> red
    -> black_1
    -> black_2
    -> ...
```

对应默认配置仍沿用 UCT 全局设置：

- `num_red = 1`
- `num_black = 5`

所以当前不是 `1 red -> 1 black`，而是 `1 red -> 最多 5 black`。

## 这次修改的重点

### 1. 树生长逻辑改成交替式

v1 / 旧版 v2 的问题是：`initial` 完成后会同时放出 `1 个 red + 多个 black`，导致 black 可能在 red 还没准备好数据时先跑。

现在的规则是：

- `initial` 或 `black` 只扩 `red`
- `red` 只扩 `black`
- black 不能再因为“全局已经有 manifest”就直接抢跑

这让每个分支都变成“先侦察，再实现”的局部闭环。

### 2. 每个 red 使用自己的 manifest

不再只依赖一个全局 `task_workspace/data_manifest.json`。

现在每个 red 节点都会写自己的 manifest：

```text
{task_workspace}/manifests/manifest_<red_node_id>.json
```

同时节点之间会显式记录：

- `input_manifest_path`
- `output_manifest_path`
- `bound_manifest_path`

这样 black 读取的是“自己绑定的 red 结果”，不是某个全局最新文件。

### 3. red 会收到结构化 demand_spec

为了避免 red 泛搜，现在 playground 会根据父节点生成一个很轻量的 `demand_spec`，内容包括：

- 当前分支的任务类型
- 当前瓶颈
- 希望搜索的数据类型
- 搜索关键词建议
- 硬约束

这个 `demand_spec` 不是单独的新节点，而是一个简单函数生成的结构化上下文，直接注入 red prompt。

这样可以保持实现简单，同时让 red 的搜索方向更稳定。

### 4. data_links 改成 task 级共享

在 `split_workspace_for_exp: true` 下，每个 worker 有独立 `exp_i` 工作目录。

这次修改后，所有 worker 的：

```text
exp_i/data_links
```

都会绑定到：

```text
task_workspace/data_links
```

因此 red 下载的数据不会再只存在于某个 `exp_0/data_links` 里而导致其他 black 看不到。

## 运行时约束

### Red

`red` 只负责：

- 搜索外部数据
- 下载到 `task_workspace/data_links`
- 探查数据格式
- 写 manifest

`red` 不跑训练，不产出 submission。

### Black

`black` 只负责：

- 读取绑定的 manifest
- 使用 manifest 中给出的本地路径和 loading snippet
- 只修改 `DataLoader` 层
- 跑训练并产出 submission

`black` 不允许自行搜索或下载外部数据。

## 与当前日志/可视化的关系

为了方便排查和盯进度，节点快照里额外记录了：

- `bound_manifest_path`
- `input_manifest_path`
- `output_manifest_path`
- `demand_spec`

因此看 `logs/uct_nodes/*.json` 或 `node.json` 时，可以直接看到：

- 这个 black 绑定的是哪个 red 的 manifest
- 这个 red 是基于什么需求去搜索的
- manifest 路径是否正确落在 task 级目录

## 关键文件

- `playground/ml_master_datatree_v2/core/playground.py`
  - 树生长逻辑
  - demand_spec 生成
  - data_links 共享绑定

- `playground/ml_master_datatree_v2/core/exp/red_exp.py`
  - red 执行器
  - 节点专属 manifest 路径

- `playground/ml_master_datatree_v2/core/exp/black_exp.py`
  - black 执行器
  - black 对绑定 manifest 的消费逻辑

- `playground/ml_master_datatree_v2/prompts/red/user_prompt.md`
  - red 的需求注入与 manifest 写入说明

- `playground/ml_master_datatree/core/utils/playground_helpers.py`
  - 节点快照字段补充

## 后续可选优化

- 如果希望严格变成 `1 red -> 1 black`，可以把 `num_black` 调成 `1`
- 如果发现 `demand_spec` 仍然不够稳定，再考虑把需求分析升级成单独节点
- 如果需要更强的一致性，可以让 red 先写临时 manifest，再原子替换正式 manifest
