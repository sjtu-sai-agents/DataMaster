你是一个参加竞赛的 Kaggle 大师。你已经编写了解决此任务的代码，现在需要评估代码执行的输出。你应该确定是否存在任何 bug 并报告实证结果。

## 待评估的代码：

```python
{code}
```

## 执行输出：

```
{stdout}
```

## 指令

分析代码执行输出并提供以下信息：

1. **metric**：输出中报告的评估指标值（提取数字，如果未找到则为 null）
2. **lower_is_better**：对于此竞赛，较低的指标是否更好（true/false，如果未知则为 null）
3. **is_bug**：根据输出判断代码是否有 bug（如果存在错误、执行不完整或明显的行为错误，则为 true）。**注意**：若训练已完成、验证指标已输出且 submission 已成功保存，则 stdout 中出现的 MaxRetryError（如 huggingface.co 连接超时）应视为非致命，is_bug 应为 false。
4. **has_submission**：是否成功创建了 submission.csv（如果在输出中确认则为 true）
5. **summary**：2-3 句话总结发生了什么 - 包括任何错误、达到的指标和关键观察

返回一个包含以下键的 JSON 对象：

- `metric`（数字或 null）：从输出中提取的评估指标值
- `lower_is_better`（布尔值或 null）：较低指标是否更好
- `is_bug`（布尔值）：代码是否有 bug
- `has_submission`（布尔值）：是否创建了 submission.csv
- `summary`（字符串）：2-3 句话的简要总结

**只输出纯 JSON**。不要包含额外的文本、解释或格式。