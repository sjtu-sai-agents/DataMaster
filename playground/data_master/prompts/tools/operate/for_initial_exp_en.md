You are allowed to use the **`operate_submission`** tool component, which provides functionality for creating scripts, running and debugging scripts, validating submission files, and getting test set scoring feedback.

### Overview of How It Works

You have your own workspace `workspace` and node identifier `node_id`:

- **`workspace`** is your current working directory, which contains subfolders like `input`, `submission`, etc. that serve different functions. **If you use file operations and bash tools, all your read/write operations must be done within the current workspace!**

- **`node_id`** is your unique ID. Under the current task, you will generate in the workspace:
  - `code_{node_id}_template.py`: **The training script and template code you write**
  - `code_{node_id}_dataloader.py`: **The DataLoader code you write**
  - `submission/submission_{node_id}.csv`: Your only valid result submission file (note that **the script you write in code can be submission/submission.csv**, we will automatically extract and convert it!)

**⚠️ ATTENTION!** You must call the following tools to form a **valid tool submission**. The tools will automatically submit the code script to the correct location and form a valid submission. Submissions formed by any other means are considered invalid and will not receive a score.
**Note! The run_code() function is very time-consuming. If it succeeds, do not call it repeatedly!**

### Code Architecture (Important)

**Code Assembly Order** (automatically executed when calling `run_code`):
1. `base_dataloader.py` - BaseDataLoader abstract class (provided by system)
2. `"\n\n"`
3. `code_{node_id}_dataloader.py` - MyDataLoader derived class (**you write**)
4. `code_{node_id}_template.py` - Training script (**you write**)

**⚠️ Special Responsibilities of Initial Node**:
- You need to **generate structured code from scratch** (template + dataloader)
- Both files can be modified (unlike Black/Red nodes)
- DataLoader code **does not need import BaseDataLoader**! We will automatically assemble the code
- The training code in the template **must use the MyDataLoader class**


### Code File Management Rules (Important)

**⚠️ Do not create additional code files!**
**⚠️ Do not overuse the execute_bash command!**

1. **Both code files need to be written by you**:
   - `code_{node_id}_template.py` - Training script (functions like train_model, save_submission, main, etc.)
   - `code_{node_id}_dataloader.py` - MyDataLoader class

2. **All code operations must be completed through tool calls**:
   - Use `operate_submission_for_initial_write_code(code, ..., file_type="template")` to write the template
   - Use `operate_submission_for_initial_write_code(code, ..., file_type="dataloader")` to write the dataloader
   - Use `operate_submission_for_initial_read_code()` to view all code (base_dataloader + dataloader + template)
   - Use `operate_submission_for_initial_fix_code(..., file_type="template")` to modify the template
   - Use `operate_submission_for_initial_fix_code(..., file_type="dataloader")` to modify the dataloader
   - Use `operate_submission_for_initial_run_code()` to execute code (automatically assembles all components)

**Correct Code Management Workflow**:
```
1. Write DataLoader
   operate_submission_for_initial_write_code(code, ..., file_type="dataloader")

2. Write Template
   operate_submission_for_initial_write_code(code, ..., file_type="template")

3. View complete code
   operate_submission_for_initial_read_code()

4. Modify/Debug
   operate_submission_for_initial_fix_code(..., file_type="dataloader")
   operate_submission_for_initial_fix_code(..., file_type="template")

5. Run
   operate_submission_for_initial_run_code()
```

---

### Tool Usage Instructions

#### `operate_submission_for_initial_read_code(node_id, workspace)`

**Function**: Read the content of all code components.

**Returned Content**:
- `base_dataloader.py` - BaseDataLoader abstract class (read-only)
- `code_{node_id}_dataloader.py` - Your MyDataLoader class (modifiable)
- `code_{node_id}_template.py` - Training script (modifiable)

**Usage Notes**:
- Before using `operate_submission_for_initial_fix_code` to modify code, it is recommended to use this tool first to view the current content of the code
- You can see the complete code context


#### `operate_submission_for_initial_write_code(code, node_id, workspace, file_type="dataloader", override=False)`

**Function**: Save your Python code to the specified file.

**⚠️ Important**: You can choose to write to `template` or `dataloader` files.

**Usage Notes**:
- `file_type="template"` - Write training script
- `file_type="dataloader"` - Write MyDataLoader class (default)
- By default in safe mode, if the code file already has content, a warning will be displayed and overwriting will be refused
- To overwrite existing code, you must explicitly set `override=True`
- Supports multi-line code strings
- DataLoader code **does not need import statements**

**Key Parameters**:
- `code`: Your Python code string
- `node_id`: Your unique node identifier
- `workspace`: Your workspace path
- `file_type`: "template" or "dataloader" (default "dataloader")
- `override`: Whether to overwrite existing code (default False)

---

#### `operate_submission_for_initial_fix_code(old_string, new_string, node_id, workspace, file_type="dataloader", replace_all=False)`

**Function**: Perform precise string replacement on the specified file.

**⚠️ Important**: You can choose to modify `template` or `dataloader` files.

**Usage Notes**:
- **Must first use `read_code` to view the code content**
- `old_string` must be **completely consistent** with the content in the code (including indentation, spaces, newlines, etc.)
- If `old_string` is not unique in the file, replacement may affect the wrong code fragment
- It is recommended to provide sufficient context to ensure `old_string` is unique in the file

**Key Parameters**:
- `old_string`: The original string to be replaced (must match exactly)
- `new_string`: The new string after replacement
- `node_id`: Your unique node identifier
- `workspace`: Your workspace path
- `file_type`: "template" or "dataloader" (default "dataloader")
- `replace_all`: Whether to replace all matching items (default only replaces the first one)

---

#### `operate_submission_for_initial_run_code(node_id, workspace, timeout=300)`

**Function**: Execute your Python code file and capture the output.

**The system will automatically** assemble the code in the following order:
1. `base_dataloader.py` - BaseDataLoader abstract class
2. `"\n\n"`
3. `code_{node_id}_dataloader.py` - Your MyDataLoader class
4. `code_{node_id}_template.py` - The training script you wrote

**Usage Notes**:
- The assembled complete code will be executed
- The training script **must** work with the template to generate `submission/submission_{node_id}.csv` file
- The script will be executed in the workspace directory
- Default execution timeout is 300 seconds

**Key Parameters**:
- `node_id`: Your unique node identifier
- `workspace`: Your workspace path
- `timeout`: Execution timeout (seconds)

---

#### `operate_submission_for_initial_validate_submission(node_id, workspace)`

**Function**: Validate your submission file through the grading server.

**Usage Notes**:
- Check if `submission/submission_{node_id}.csv` exists
- Send the submission file to the grading server for format validation
- Before using `grade_code` to get the final score, it is recommended to use this tool first to validate

---

#### `operate_submission_for_initial_grade_code(node_id, workspace, timeout=300)`

**Function**: Use the local grading script to score your submission file.

**Usage Notes**:
- **Ensure the submission file exists** (generated through `operate_submission_for_initial_run_code`)
- **It is recommended to use `validate_submission` first to validate**
- Use ground truth file for real scoring

### `execute_bash`

You have access to the following tools to help you complete the task:

1. **execute_bash** - Execute bash commands in the terminal
   - Use this to check files: `head -20 input/train.csv`
   - **Do not use this to set environment variables or create or modify files!**

1. It is prohibited to use execute_bash to execute complete Python training code.
2. You must first use operate_submission_write_code to write partial code, or use execute_bash command to implement overall code modification (this applies when many parts of the code need to be modified), then use operate_submission_run_code to execute.
3. When you think you have fixed it, you can try running it. If a valid submission has been generated, you can terminate the run.
