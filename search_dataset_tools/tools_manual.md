# Tools Manual

## `operate_submission` Toolset

你被允许使用 **`operate_submission`** 工具组件，该工具集提供了创建脚本、运行和调试脚本、验证提交文件、获取测试集评分反馈等功能。

---

### 工作机制概述

你有自己的工作空间 `workspace` 和节点标识 `node_id`：

- **`workspace`** 是你当前的工作目录，内部包含 `input`、`submission` 等承担不同功能的子文件夹。**如果你使用文件操作和 bash 工具，你的所有读写操作都必须在当前的 workspace 中进行！**

- **`node_id`** 是你拥有的唯一 ID，在当前任务下，你会在 workspace 下生成 `code_{node_id}.py` 作为你的**唯一 Python 脚本**，该脚本将会在 submission 文件夹下生成 `submission/submission_{node_id}.csv`，作为你**唯一有效的结果提交文件**。

**⚠️ ATTENTION!** 你必须调用如下工具来形成**有效的工具提交**。工具会自动将代码脚本提交到正确的位置并形成合法的提交。使用其他任何手段形成的提交都视为无效提交，将不会有分数。

---

### 代码文件管理规则（重要）

**⚠️ 禁止创建额外的代码文件！**

1. **唯一有效的代码文件是 `code_{node_id}.py`** - 这是通过 `operate_submission_write_code` 工具创建的
2. **禁止使用 `str_replace_editor` 或其他工具创建 `solution.py`、`my_code.py` 等额外文件**
3. **所有代码操作必须通过 MCP 工具完成**：
   - 使用 `write_code()` 创建/覆盖代码文件
   - 使用 `read_code()` 查看代码内容
   - 使用 `fix_code()` 修改代码
   - 使用 `run_code()` 执行代码

4. **系统会自动清理违规创建的中间文件** - 如果你创建了 `solution.py` 等文件，它们可能会被系统自动删除

**正确的代码管理流程**：
```
operate_submission_write_code() → code_{node_id}.py (唯一代码文件)
     ↓
operate_submission_read_code() → 查看内容
     ↓
operate_submission_fix_code() → 修改代码
     ↓
operate_submission_run_code() → 执行代码
```

**错误的做法**：
```
❌ 使用 str_replace_editor 创建 solution.py
❌ 使用 echo/cat 等命令创建 my_script.py
❌ 创建任何除 code_{node_id}.py 之外的代码文件
```

---

### 工具使用说明

#### `operate_submission_write_code(code, node_id, workspace, override=False)`

**功能**：将你的 Python 代码保存到指定的代码文件中。

**使用注意事项**：
- 默认处于安全模式，如果代码文件已有内容，会显示警告并拒绝覆盖
- 如需覆盖已有代码，必须显式设置 `override=True`
- 代码会被保存到 `{workspace}/code_{node_id}.py`
- 支持多行代码字符串

**关键参数**：
- `code`: 完整的 Python 代码字符串
- `node_id`: 你的节点唯一标识符
- `workspace`: 你的工作空间路径
- `override`: 是否覆盖已有代码（默认 False）

---

#### `operate_submission_read_code(node_id, workspace)`

**功能**：读取你之前保存的代码文件内容。

**使用注意事项**：
- 在使用 `fix_code` 修改代码之前，建议先使用此工具查看代码的当前内容
- 返回的是完整的代码文件内容

**关键参数**：
- `node_id`: 你的节点唯一标识符
- `workspace`: 你的工作空间路径

---

#### `operate_submission_fix_code(old_string, new_string, node_id, workspace, replace_all=False)`

**功能**：对代码文件执行精确的字符串替换。

**使用注意事项**：
- **必须先用 `read_code` 查看代码内容**，确保 `old_string` 与文件中的内容完全匹配
- `old_string` 必须与代码中的内容**完全一致**（包括缩进、空格、换行等，注意有必要是加上转义字符！）
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

**功能**：执行你保存的 Python 代码文件并捕获输出。

**使用注意事项**：
- 执行 `{workspace}/code_{node_id}.py` 脚本
- 你的脚本**必须**生成 `submission/submission_{node_id}.csv` 文件或者 `submission/submission.csv`
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
- **确保提交文件已存在**（通过 `run_code` 生成）
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
1. 理解任务和数据
   └─ 使用 bash 工具查看 input/ 目录下的数据文件

2. 编写初始代码
   └─ 使用 operate_submission_write_code() 保存你的代码

3. 运行并调试
   └─ 使用 operate_submission_run_code() 执行代码
   └─ 检查输出中的错误信息

4. 修复问题
   └─ 使用 operate_submission_read_code() 查看代码
   └─ 使用 operate_submission_fix_code() 进行针对性修复
   └─ 或使用 operate_submission_write_code(override=True) 完全重写

5. 验证提交
   └─ 使用 operate_submission_validate_submission() 验证提交文件合法性

6. 获取评分
   └─ 使用 operate_submission_grade_code() 获取最终评分
```

## Base Operation Tools

You have access to the following tools to help you complete the task:

1. **execute_bash** - Execute bash commands in the terminal
   - Use this to run Python code and do many more things in the bash command line.
   - Use this to check files: `head -20 input/train.csv`
   - Use this to test your code before finalizing

2. **think** - Think about the problem (does not affect the environment)
   - Use this to plan your approach before writing code


## Web Search Tools

你可以利用多种信息源搜索互联网上的新的数据集：

- `search_github`
- `search_huggingface`
- `search_scholar`
- `search_web`

- 互联网上的公开数据集：你可以参考：
    - https://raw.githubusercontent.com/awesomedata/awesome-public-datasets/refs/heads/master/README.rst
    - https://github.com/awesomedata/apd-core
    - https://github.com/awesomedata/awesome-public-datasets
    （你可以利用 bash 工具下载到 workspace 的 new_data 文件夹下并进行查看）
- 还有其他的互联网上的公开数据集，可以使用 Google Search 进行搜索得到
- 一些学术论文和学术信息网站也会包含一些 release 最新数据集的工作，你可以使用 Arxiv，Google Scholar 等相关信息源进行搜索，并在 HuggingFace，Kaggle 等网站上查看数据集的详细信息、下载数据集！
- **注意！Huggingface 上有很多大规模的数据集可供下载**，但是 HuggingFace 不支持模糊的语义搜索，关键在于每一个数据集都有一个 dataset_id 和若干个 config_name 和 split，你可以首先使用 scholar_search, web_search, web_parse 等泛化搜索工具进行搜索，得到 HuggingFace 的 dataset_id 之后再使用 search_huggingface 进行搜索。

