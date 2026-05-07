# 工具使用指南

## Available Tools

You have access to the following tools to help you complete the task:

1. **execute_bash** - Execute bash commands in the terminal
   - Use this to run Python code and do many more things in the bash command line.
   - Use this to check files: `head -20 input/train.csv`
   - Use this to test your code before finalizing

2. **str_replace_editor** - View, create, and edit files
   - Use `create` command to create new Python files
   - Use `str_replace` command to edit existing files
   - Use `view` command to view file contents

3. **think** - Think about the problem (does not affect the environment)
   - Use this to plan your approach before writing code

4. **finish** - Signal that you have completed the task
   - Use this when you have finalized your solution
   - The system will execute your code and save results to `submission/`

## 代码库管理

在读取完数据之后，你被允许在 `{tmp_code_dir}/` 文件夹下编写一些 Python 文件进行 Debug 测试，你可以使用 `execute_bash` 工具进行 Python 代码执行操作。

**DEBUGGING RULES - Read Carefully:**

- `{tmp_code_dir}/` 是一个项目全局的探索文件，因此，**我们建议你在执行一次探索的时候**在该文件夹下创建一个新的文件夹，名称叫做 `trial_[index]`，例如 `trial_1`, `trial_2` 等等（你可以按照当前已有的进行编号），接下来在这个新的子文件夹中编写你的代码
    - * 或许你可以查看之前的 trial 的文件！不过**他们极有可能是尝试失败的文件**，因此无需过度依赖他们！
- **需要被执行的Debug代码不允许直接将答案写入** `submission/` 文件夹！这样提交得到的内容不会作为最终的评判依据。
    - 你可以输出在**当前 trial 文件夹下的某个位置**，重要的是要得到一些 validation 的输出
    - **在 finish 工具中存储的 submission 文件需要存储在 `submission/` 文件夹中！**

> 系统会自动执行最终提交的代码，将对应的 submission 处理到 `submission/` 文件夹中并添加哈希值作为有效提交。


## Recommended Workflow

1. **Explore**: Check data structure in `input/`
2. **Think**: Plan your approach
3. **Create**: Write code in `{tmp_code_dir}/trial_[index]/`
4. **Test**: Run and verify your code works
5. **Iterate**: Try multiple versions in `experiments/` folder
6. **Finalize**: Use `finish` tool with your best solution

**Important Notes:**
- Keep experiments organized in subfolders
- Your final solution should be ready to execute
- The system will handle submission file creation automatically

## CRITICAL: Final Output Format

When you use the **finish** tool, you MUST provide:
1. A brief plan/sketch of your solution (2-3 sentences in natural language)
2. The complete Python code in a markdown code block

Example final message format:

Based on the data exploration, I will use XGBoost with feature engineering...

```python
import pandas as pd
from xgboost import XGBClassifier
...
```


Do NOT forget to include the code block in your final finish message!

**在 finish 工具中生成的最终代码需要将最终生成的 submission 文件存储在 `submission/` 文件夹中！**

## Important Notes

- Test your code in `tmp_code/` before using `finish`
- Your final code will be executed by the system to generate submission files
- Do NOT hardcode submission paths in your code (let the system handle it)
