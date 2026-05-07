# ML-Master Data Tree

## 数据增强版代码

这是 ML-Master 的源代码，你需要在这个基础上维持大框架不变、但是进行功能性的转变：转变为一个能够进行自动化数据探索和发现的 Data Agent

### 节点状态

- root：根节点，负责从 initial code 开始，生成最初版本的代码
- black: 黑色节点，主要负责对现有训练数据进行数据增强、数据整合
- red: 红色节点，主要负责通过调用外部搜索接口进行数据搜索、下载新的数据集到本地、引入新的数据源

同时，每一个节点的其他基本定义仍然保持一致:

```python
plan: str                    # Agent 生成的计划/方案描述
      code: str                    # Agent 生成的代码
      parent: Optional["UCTNode"]  # 父节点
      id: str                      # 唯一标识 (UUID)
      created_at: float            # 创建时间戳

      # === 执行结果 ===
      stdout: Optional[str]        # 代码执行输出
      exit_code: Optional[int]     # 退出码
      analysis: Optional[str]      # metric_agent 的分析结果
      metric: MetricValue          # 指标值 (带 maximize 方向)
      finish_time: Optional[float] # 完成时间

      # === 状态标志 ===
      is_buggy: Optional[bool]     # 是否有 bug (None=未评估, True=有bug, False=无bug)
      is_valid: Optional[bool]     # 是否有效
      original_is_buggy: Optional[bool]  # 保存原始 buggy 状态
      is_terminal: bool            # 是否终止节点
      is_debug_success: bool       # debug 是否成功
      continue_improve: bool       # 是否继续改进
      locked: bool                 # 是否被锁定

      # === UCT 统计 ===
      visits: int                  # 访问次数
      total_reward: float          # 累计奖励
      children: set                # 子节点集合
      expected_child_count: int    # 预期子节点数
```

### 节点扩展

- 根节点唯一初始化，你需要实现一个 InitialExp
- 如果一个节点最终提交的代码是 buggy 的:
    - 终止该节点，该节点停止扩展
- 如果一个节点最终提交的代码是 not buggy 的：
    - 生成一个红色节点和一个黑色节点
    - 红色节点的上限为 1，黑色节点的上限为 5
    - 每一个节点内部是类似于 ImproveExp，但是导入的提示词不同
  
每一个节点还是会提交一份完整的代码进行评判，以 validation score 作为评判分数
UCT 算法仍然保持不变 评测方式仍然保持不变

## 数据索引

维护一个全局的数据索引

- 每一个红色节点和黑色节点除了要生成可提交的代码和自然语言的 Plan，还会让 Agent 通过工具调用形成一个全局的数据索引
- 每一条数据索引都有：
    - 具体数据文件存储的地址
    - 一个 README 文件 包含数据集的介绍
- 这个是一个可扩展的接口 因此不需要考虑数据重复和冲突的问题！但是这个要在节点中体现
- 你暂时不用实现这个
