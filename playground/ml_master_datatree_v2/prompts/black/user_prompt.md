{general_instruction_content}

## COMPETITION INSTRUCTIONS

{task_description}

---

## YOUR TASK: BLACK NODE v2 — 数据清洗 & 特征工程（使用 black-dataops）

**你是一个数据清洗专家，可以通过 `use_skill` 调用 `black-dataops` 技能文档。**

你的任务是：使用 `black-dataops` 技能文档和必要的 reference，再结合外部数据 Manifest（如果可用），改进 `MyDataLoader` 的 `setup()` 方法，从而提升模型性能。

---

{manifest_section}

---

## 当前最佳方案

当前全局最佳方案的指标为 **{best_metric}**：

```python
{best_code}
```

## 父节点方案

```python
{previous_code}
```

父节点执行输出：
```
{execution_output}
```

## 历史尝试记录

{memory}

---

## black-dataops Skill 使用建议

优先按下面顺序使用技能系统：

1. 先查看当前上下文中暴露的 skill metadata
2. 使用 `use_skill(skill_name="black-dataops", action="get_info")` 获取总指南
3. 根据当前问题按需读取 reference：
   - `cleaning_methods.md`
   - `data_merge_methods.md`
   - `validation_diagnosis.md`
4. 根据 skill 文档给出的规则和模式实现最小必要改动

如果当前任务涉及：

- 图像损坏、零字节图、解码失败：优先看 `cleaning_methods.md`
- 比赛数据和外部数据如何合并：优先看 `data_merge_methods.md`
- 根据 val 预测或阈值行为判断数据问题：优先看 `validation_diagnosis.md`

---

## 你的核心任务

1. **分析父节点的不足**：根据执行输出和指标，找出当前数据处理的瓶颈
2. **选择合适的技能**：优先从 `black-dataops` 中选择 1-3 个最有可能提升性能的技能
3. **整合外部数据**（如 manifest 可用）：使用 manifest 中的 loading_snippet 加载外部数据，并用选定的清洗技能处理它
4. **实现 MyDataLoader**：将选定的技能集成到 `setup()` 中

## 重要约束

1. **禁止搜索或下载任何外部数据** — 只能使用 `input/` 目录和 manifest 中列出的路径
2. **禁止修改算法逻辑** — 只改 DataLoader 层
3. **必须与历史方案有实质性差异** — 不要重复已尝试过的技能组合
4. **使用 manifest 中的 loading_snippet** — 不要自己猜测外部数据格式
5. **外部数据只能加入训练集，绝不能进入验证集**
6. **严禁先拼接原始数据和外部数据，再对 combined dataset 做 `train_test_split`**
7. **不要尝试获取 hidden-test / private score** — 只能根据代码打印出的验证集指标和 submission 合法性做决策
8. **如果 black-dataops 对某个外部标签映射或 merge 方式持保守态度，优先遵守 skill 约束**

{data_loader_readme}

## Data Preview

{data_preview}

---

## Tools Manual

Your workspace: {workspace}
Your node_id: {node_id}

{operation_tools_readme}

流程提示：
1. 禁止用 execute_bash 执行完整 Python 训练代码
2. 必须先用 operate_submission_write_code 写入代码，再用 operate_submission_run_code 执行
3. 可使用 operate_submission_validate_submission 检查 submission 格式是否合法
4. 不要调用任何真实评分/hidden-test 工具；代码运行成功且 submission 生成后即可停止，不要继续修改

---

## ⚠️ CRITICAL: 固定验证集要求

**你必须使用预先分割的验证集 `input/val.csv`，不得使用随机划分！**

在你的 `MyDataLoader.setup()` 中：

1. **检查 `input/val.csv` 是否存在**：
   ```python
   if os.path.exists('input/val.csv'):
       val_df = pd.read_csv('input/val.csv')
       # Remove val samples from train
       val_images = set(val_df['image'].values)
       train_df = train_full_df[~train_full_df['image'].isin(val_images)]
   ```

2. **禁止使用 `train_test_split` 进行随机划分**
   - 如果 `val.csv` 存在，直接使用它
   - 所有节点必须在相同的 val set 上评估，这样 metric 才能真正反映改进

3. **如果父节点的代码已经正确使用了 `val.csv`**：
   - **保留这部分逻辑**
   - 只修改数据增强、外部数据加载、特征工程等部分
   - 不要重写整个 `setup()` 函数

4. **如果 `input/val.csv` 不存在**：
   - 只能从**原始比赛训练集**中切分 `train/val`
   - **先**从原始比赛训练集切出 `X_train_orig, X_val, y_train_orig, y_val`
   - **再**把外部数据追加到 `X_train_orig / y_train_orig`
   - 最终得到的 `X_val / y_val` 必须只包含原始比赛数据，不能包含任何外部样本

5. **绝对禁止的写法**：
   ```python
   X_combined = np.concatenate([X_orig, X_external])
   y_combined = np.concatenate([y_orig, y_external])
   X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, ...)
   ```
   上面这种写法会把外部数据泄漏到验证集，导致 val metric 不可信。

**为什么这很重要？**
- 如果每个节点用不同的 val set，metric 无法比较
- UCT 搜索会失效（比较的是苹果和橙子）
- 无法判断改进是否真的有效
- 如果外部数据进入 val，分数会被污染，无法反映比赛数据上的真实提升
