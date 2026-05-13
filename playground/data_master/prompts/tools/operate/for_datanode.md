你被允许使用 **`operate_submission`** 工具组件，该工具集提供了创建脚本、运行和调试脚本、验证提交文件、获取测试集评分反馈等功能。

### 工作机制概述

你有自己的工作空间 `workspace` 和节点标识 `node_id`：

- **`workspace`** 是你当前的工作目录，内部包含 `input`、`submission` 等承担不同功能的子文件夹。**如果你使用文件操作和 bash 工具，你的所有读写操作都必须在当前的 workspace 中进行！**

- **`node_id`** 是你拥有的唯一 ID，在当前任务下，你会在 workspace 下生成：
  - `code_{node_id}_template.py`：模板文件（**从父节点继承，只读，不可修改**）
  - `code_{node_id}_dataloader.py`：**你唯一可以修改的 DataLoader 代码文件**
  - `submission/submission_{node_id}.csv`：你的唯一有效结果提交文件,不过注意**你在代码中写的脚本可以是 submission/submission.csv**，我们会自动提取并转化！

**⚠️ ATTENTION!** 你必须调用如下工具来形成**有效的工具提交**。工具会自动将代码脚本提交到正确的位置并形成合法的提交。使用其他任何手段形成的提交都视为无效提交，将不会有分数。
**注意！operate_submission_run_code() 函数是非常耗时的，如果调用成功就不要再重复调用了！**

### 代码架构（重要）

**代码拼装顺序**（调用 `operate_submission_run_code` 时自动执行）：
1. `base_dataloader.py` - BaseDataLoader 抽象类（系统提供）
2. `"\n\n"`
3. `code_{node_id}_dataloader.py` - MyDataLoader 派生类（**你编写的代码**）
4. `code_{node_id}_template.py` - 训练脚本（从父节点继承，**只读**）

**⚠️ 关键约束**：
- **你只能修改 `code_{node_id}_dataloader.py`！**
- 模板文件（`code_{node_id}_template.py`）是从父节点继承的，**不能修改**！
- DataLoader 代码**不需要 import BaseDataLoader 语句**，系统会自动拼装成完整的可执行脚本！
- 模板中的训练代码会使用你的 `MyDataLoader` 类，请务必保证命名正确！

---

### 代码文件管理规则（重要）

**⚠️ 禁止创建额外的代码文件！**
**⚠️ 不要过度的使用 execute_bash 文件！**

1. **唯一有效的代码文件是 `code_{node_id}_dataloader.py`**: 这是通过 `write_code` 工具创建的
2. **模板文件不可修改**: `code_{node_id}_template.py` 是从父节点继承的，你只能读取不能修改
3. **所有代码操作必须通过 MCP 工具完成**：
   - 使用 `operate_submission_write_code()` 创建/覆盖 DataLoader 代码文件
   - 使用 `operate_submission_read_code()` 查看所有代码（base_dataloader + dataloader + template）
   - 使用 `operate_submission_fix_code()` 修改 DataLoader 代码
   - 使用 `operate_submission_run_code()` 执行代码（会自动拼装所有组件）

---

### 工具使用说明

#### `operate_submission_read_code(node_id, workspace)`

**功能**：读取所有代码组件的内容。

**返回内容**：
- `base_dataloader.py` - BaseDataLoader 抽象类（只读）
- `code_{node_id}_dataloader.py` - 你的 MyDataLoader 类（可修改）
- `code_{node_id}_template.py` - 训练脚本（从父节点继承，只读）

**使用注意事项**：
- 在使用 `fix_code` 修改代码之前，建议先使用此工具查看代码的当前内容
- 你可以看到完整的代码上下文，包括模板文件（但模板不可修改）

**关键参数**：
- `node_id`: 你的节点唯一标识符
- `workspace`: 你的工作空间路径

---

#### `operate_submission_write_code(code, node_id, workspace, override=False)`

**功能**：将你的 Python DataLoader 代码保存到 dataloader 文件中。

**⚠️ 重要约束**：此工具**只能**写入 `code_{node_id}_dataloader.py`，不能修改模板！

**使用注意事项**：
- 默认处于安全模式，如果代码文件已有内容，会显示警告并拒绝覆盖
- 如需覆盖已有代码，必须显式设置 `override=True`
- 代码会被保存到 `{workspace}/code_{node_id}_dataloader.py`
- 支持多行代码字符串
- DataLoader 代码**不需要 import 语句**

**关键参数**：
- `code`: 你的 DataLoader Python 代码字符串
- `node_id`: 你的节点唯一标识符
- `workspace`: 你的工作空间路径
- `override`: 是否覆盖已有代码（默认 False）

---

#### `operate_submission_fix_code(old_string, new_string, node_id, workspace, replace_all=False)`

**功能**：对 DataLoader 代码文件执行精确的字符串替换。

**⚠️ 重要约束**：此工具**只能**修改 `code_{node_id}_dataloader.py`，不能修改模板！

**使用注意事项**：
- **必须先用 `read_code` 查看代码内容**，确保 `old_string` 与文件中的内容完全匹配
- `old_string` 必须与代码中的内容**完全一致**（包括缩进、空格、换行等）
- 如果 `old_string` 在文件中不唯一，替换可能会影响错误的代码片段
- 建议提供足够的上下文以确保 `old_string` 在文件中唯一
- 使用 `replace_all=True` 时会替换所有匹配项，适合变量重命名场景

**关键参数**：
- `old_string`: 要替换的原始字符串（必须完全匹配）
- `new_string`: 替换后的新字符串
- `node_id`: 你的节点唯一标识符
- `workspace`: 你的工作空间路径
- `replace_all`: 是否替换所有匹配项（默认只替换第一个）

---

#### `operate_submission_run_code(node_id, workspace, timeout=300)`

**功能**：执行你的 Python 代码文件并捕获输出。

**系统会自动**按以下顺序拼装代码：
1. `base_dataloader.py` - BaseDataLoader 抽象类
2. `"\n\n"`
3. `code_{node_id}_dataloader.py` - 你的 MyDataLoader 类
4. `code_{node_id}_template.py` - 从父节点继承的训练脚本

**使用注意事项**：
- 拼装后的完整代码会被执行
- 你的 DataLoader 脚本**必须**配合模板生成 `submission/submission_{node_id}.csv` 文件
- 脚本会在 workspace 目录下执行
- 执行超时默认为 300 秒
- 执行后建议使用 `validate_submission` 验证提交文件是否合法

**关键参数**：
- `node_id`: 你的节点唯一标识符
- `workspace`: 你的工作空间路径
- `timeout`: 执行超时时间（秒）

---

#### `operate_submission_validate_submission(node_id, workspace)`

**功能**：通过评分服务器验证你的提交文件判断提交文件是否合法。

**使用注意事项**：
- 检查 `submission/submission_{node_id}.csv` 是否存在
- 将提交文件发送到评分服务器进行格式验证和评分
- 在使用 `grade_code` 获取最终评分前，建议先使用此工具验证
- 如果提交文件不存在，会返回错误信息

**关键参数**：
- `node_id`: 你的节点唯一标识符
- `workspace`: 你的工作空间路径

---

#### `operate_submission_grade_code(node_id, workspace, timeout=300)`

**功能**：使用本地 grading 脚本对你的提交文件进行评分。

**使用注意事项**：
- **确保提交文件已存在**（通过 `operate_submission_run_code` 生成）
- **建议先使用 `validate_submission` 验证提交文件合法性**
- 使用 ground truth 文件进行真实评分
- 评分结果包含分数和详细输出信息

**关键参数**：
- `node_id`: 你的节点唯一标识符
- `workspace`: 你的工作空间路径
- `timeout`: 执行超时时间（秒）

---

### 建议的工作流程

```
1. 查看现有代码
   └─ 使用 read_code() 查看 base_dataloader + dataloader + template
   └─ 理解父节点提供的模板和契约

2. 编写/修改 DataLoader 代码
   └─ 使用 write_code() 编写 MyDataLoader 类
   └─ 记住：不能修改模板，只能修改 dataloader！

3. 运行并调试
   └─ 使用 operate_submission_run_code() 执行代码（自动拼装所有组件）
   └─ 检查输出中的错误信息

4. 修复问题
   └─ 使用 read_code() 查看代码
   └─ 使用 fix_code() 进行针对性修复
   └─ 或使用 write_code(override=True) 完全重写 dataloader

5. 验证提交
   └─ 使用 validate_submission() 验证提交文件合法性

6. 获取评分
   └─ 使用 grade_code() 获取最终评分
```

---

### `execute_bash`

You have access to the following tools to help you complete the task:

1. **execute_bash** - Execute bash commands in the terminal
   - Use this to check files: `head -20 input/train.csv`
   - **Do not use this to set environment variables or create or modify files!**
