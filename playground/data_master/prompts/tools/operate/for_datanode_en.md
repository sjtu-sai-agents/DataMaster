You are allowed to use the **`operate_submission`** tool component, which provides functionality for creating scripts, running and debugging scripts, validating submission files, and getting test set scoring feedback.

### Overview of How It Works

You have your own workspace `workspace` and node identifier `node_id`:

- **`workspace`** is your current working directory, which contains subfolders like `input`, `submission`, etc. that serve different functions. **If you use file operations and bash tools, all your read/write operations must be done within the current workspace!**

- **`node_id`** is your unique ID. Under the current task, you will generate in the workspace:
  - `code_{node_id}_template.py`: Template file (**inherited from parent node, read-only, cannot be modified**)
  - `code_{node_id}_dataloader.py`: **The only DataLoader code file you can modify**
  - `submission/submission_{node_id}.csv`: Your only valid result submission file (note that **the script you write in code can be submission/submission.csv**, we will automatically extract and convert it!)

**⚠️ ATTENTION!** You must call the following tools to form a **valid tool submission**. The tools will automatically submit the code script to the correct location and form a valid submission. Submissions formed by any other means are considered invalid and will not receive a score.
**Note! The operate_submission_run_code() function is very time-consuming. If it succeeds, do not call it repeatedly!**

### Code Architecture (Important)

**Code Assembly Order** (automatically executed when calling `operate_submission_run_code`):
1. `base_dataloader.py` - BaseDataLoader abstract class (provided by system)
2. `"\n\n"`
3. `code_{node_id}_dataloader.py` - MyDataLoader derived class (**your code**)
4. `code_{node_id}_template.py` - Training script (inherited from parent node, **read-only**)

**⚠️ Key Constraints**:
- **You can only modify `code_{node_id}_dataloader.py`!**
- The template file (`code_{node_id}_template.py`) is inherited from the parent node and **cannot be modified!**
- DataLoader code **does not need import BaseDataLoader statement**; the system will automatically assemble it into a complete executable script!
- The training code in the template will use your `MyDataLoader` class, so please ensure the naming is correct!

---

### Code File Management Rules (Important)

**⚠️ Do not create additional code files!**
**⚠️ Do not overuse the execute_bash command!**

1. **The only valid code file is `code_{node_id}_dataloader.py`**: This is created through the `write_code` tool
2. **Template files cannot be modified**: `code_{node_id}_template.py` is inherited from the parent node, you can only read but not modify it
3. **All code operations must be completed through MCP tools**:
   - Use `operate_submission_write_code()` to create/overwrite DataLoader code files
   - Use `operate_submission_read_code()` to view all code (base_dataloader + dataloader + template)
   - Use `operate_submission_fix_code()` to modify DataLoader code
   - Use `operate_submission_run_code()` to execute code (automatically assembles all components)

---

### Tool Usage Instructions

#### `operate_submission_read_code(node_id, workspace)`

**Function**: Read the content of all code components.

**Returned Content**:
- `base_dataloader.py` - BaseDataLoader abstract class (read-only)
- `code_{node_id}_dataloader.py` - Your MyDataLoader class (modifiable)
- `code_{node_id}_template.py` - Training script (inherited from parent node, read-only)

**Usage Notes**:
- Before using `fix_code` to modify code, it is recommended to use this tool first to view the current content of the code
- You can see the complete code context, including the template file (but the template cannot be modified)

**Key Parameters**:
- `node_id`: Your unique node identifier
- `workspace`: Your workspace path

---

#### `operate_submission_write_code(code, node_id, workspace, override=False)`

**Function**: Save your Python DataLoader code to the dataloader file.

**⚠️ Important Constraint**: This tool **can only** write to `code_{node_id}_dataloader.py`, cannot modify the template!

**Usage Notes**:
- By default in safe mode, if the code file already has content, a warning will be displayed and overwriting will be refused
- To overwrite existing code, you must explicitly set `override=True`
- Code will be saved to `{workspace}/code_{node_id}_dataloader.py`
- Supports multi-line code strings
- DataLoader code **does not need import statements**

**Key Parameters**:
- `code`: Your DataLoader Python code string
- `node_id`: Your unique node identifier
- `workspace`: Your workspace path
- `override`: Whether to overwrite existing code (default False)

---

#### `operate_submission_fix_code(old_string, new_string, node_id, workspace, replace_all=False)`

**Function**: Perform precise string replacement on the DataLoader code file.

**⚠️ Important Constraint**: This tool **can only** modify `code_{node_id}_dataloader.py`, cannot modify the template!

**Usage Notes**:
- **Must first use `read_code` to view the code content** to ensure `old_string` matches exactly with the content in the file
- `old_string` must be **completely consistent** with the content in the code (including indentation, spaces, newlines, etc.)
- If `old_string` is not unique in the file, replacement may affect the wrong code fragment
- It is recommended to provide sufficient context to ensure `old_string` is unique in the file
- When using `replace_all=True`, all matches will be replaced, suitable for variable renaming scenarios

**Key Parameters**:
- `old_string`: The original string to be replaced (must match exactly)
- `new_string`: The new string after replacement
- `node_id`: Your unique node identifier
- `workspace`: Your workspace path
- `replace_all`: Whether to replace all matching items (default only replaces the first one)

---

#### `operate_submission_run_code(node_id, workspace, timeout=300)`

**Function**: Execute your Python code file and capture the output.

**The system will automatically** assemble the code in the following order:
1. `base_dataloader.py` - BaseDataLoader abstract class
2. `"\n\n"`
3. `code_{node_id}_dataloader.py` - Your MyDataLoader class
4. `code_{node_id}_template.py` - Training script inherited from parent node

**Usage Notes**:
- The assembled complete code will be executed
- Your DataLoader script **must** work with the template to generate `submission/submission_{node_id}.csv` file
- The script will be executed in the workspace directory
- Default execution timeout is 300 seconds
- After execution, it is recommended to use `validate_submission` to verify if the submission file is valid

**Key Parameters**:
- `node_id`: Your unique node identifier
- `workspace`: Your workspace path
- `timeout`: Execution timeout (seconds)

---

#### `operate_submission_validate_submission(node_id, workspace)`

**Function**: Validate your submission file through the grading server to determine if the submission file is valid.

**Usage Notes**:
- Check if `submission/submission_{node_id}.csv` exists
- Send the submission file to the grading server for format validation and scoring
- Before using `grade_code` to get the final score, it is recommended to use this tool first to validate
- If the submission file does not exist, an error message will be returned

**Key Parameters**:
- `node_id`: Your unique node identifier
- `workspace`: Your workspace path

---

#### `operate_submission_grade_code(node_id, workspace, timeout=300)`

**Function**: Use the local grading script to score your submission file.

**Usage Notes**:
- **Ensure the submission file exists** (generated through `operate_submission_run_code`)
- **It is recommended to use `validate_submission` first to validate the submission file**
- Use ground truth file for real scoring
- The scoring result includes the score and detailed output information

**Key Parameters**:
- `node_id`: Your unique node identifier
- `workspace`: Your workspace path
- `timeout`: Execution timeout (seconds)

---

### Recommended Workflow

```
1. View existing code
   └─ Use read_code() to view base_dataloader + dataloader + template
   └─ Understand the template and contract provided by the parent node

2. Write/modify DataLoader code
   └─ Use write_code() to write MyDataLoader class
   └─ Remember: cannot modify template, can only modify dataloader!

3. Run and debug
   └─ Use operate_submission_run_code() to execute code (automatically assembles all components)
   └─ Check error messages in the output

4. Fix issues
   └─ Use read_code() to view the code
   └─ Use fix_code() for targeted fixes
   └─ Or use write_code(override=True) to completely rewrite dataloader

5. Validate submission
   └─ Use validate_submission() to validate the submission file

6. Get score
   └─ Use grade_code() to get the final score
```

---

### `execute_bash`

You have access to the following tools to help you complete the task:

1. **execute_bash** - Execute bash commands in the terminal
   - Use this to check files: `head -20 input/train.csv`
   - **Do not use this to set environment variables or create or modify files!**
