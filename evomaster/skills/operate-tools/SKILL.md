---
name: operate-tools
description: Code execution and submission tools for DataNode (Black/Red) agents. Manages DataLoader code, runs assembled scripts, validates and grades submissions.
license: Complete terms in LICENSE.txt
---

# Operate Tools (DataNode)

Code management, execution, validation, and grading for Black/Red nodes. Nodes can only modify `code_{node_id}_dataloader.py`; template is inherited and read-only.

## Code Assembly (auto on run)

```
base_dataloader.py + "\n\n" + code_{node_id}_dataloader.py + code_{node_id}_template.py
```

## CLI Usage

All commands require `-w WORKSPACE -n NODE_ID`.

### Code Management

| Tool | CLI |
|------|-----|
| `read_code` | `python scripts/operate_cli.py -w WORKSPACE -n NODE_ID read-code` |
| `write_code` | `python scripts/operate_cli.py -w WORKSPACE -n NODE_ID write-code --code "..."` |
| `write_code` (file) | `python scripts/operate_cli.py -w WORKSPACE -n NODE_ID write-code --code-file /path/to/file.py` |
| `write_code` (override) | `python scripts/operate_cli.py -w WORKSPACE -n NODE_ID write-code --code "..." --override` |
| `fix_code` | `python scripts/operate_cli.py -w WORKSPACE -n NODE_ID fix-code --old "..." --new "..."` |
| `fix_code` (all) | `python scripts/operate_cli.py -w WORKSPACE -n NODE_ID fix-code --old "..." --new "..." --replace-all` |

### Execution

| Tool | CLI |
|------|-----|
| `run_code` | `python scripts/operate_cli.py -w WORKSPACE -n NODE_ID run-code [--timeout 3600]` |

### Validation & Grading

| Tool | CLI |
|------|-----|
| `validate_submission` | `python scripts/operate_cli.py -w WORKSPACE -n NODE_ID validate` |
| `grade_code` | `python scripts/operate_cli.py -w WORKSPACE -n NODE_ID grade [--timeout 300]` |

## Workspace File Layout

```
{workspace}/
├── input/                          # Input data (train.csv, test.csv, val.csv, ...)
├── submission/
│   └── submission_{node_id}.csv    # Generated submission file
├── code_{node_id}_dataloader.py    # YOUR code (modifiable)
├── code_{node_id}_template.py      # Parent's template (read-only)
└── code_{node_id}.py               # Assembled code (auto-generated on run)
```

## DataLoader Pattern

`code_{node_id}_dataloader.py` must define a `MyDataLoader(BaseDataLoader)` class:

```python
# No import statements needed - auto-assembled with base_dataloader.py

class MyDataLoader(BaseDataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        # Load data, feature engineering, augmentation
        # Must set self.train_data and self.test_data
        self.train_data = ...
        self.test_data = ...

    def describe(self) -> str:
        return "Description of data processing approach"
```

Constraints:
- Class name must be `MyDataLoader`, inheriting `BaseDataLoader`
- No import for `BaseDataLoader` (auto-assembled)
- Do not implement training logic (that's in the template)
- Use `input/val.csv` if it exists; do not use `train_test_split` randomly

## Recommended Workflow

```
1. read-code      → View base + dataloader + template
2. write-code     → Write MyDataLoader class
3. run-code       → Execute (auto-assembles all components)
4. fix-code       → Fix issues in dataloader
5. validate       → Validate submission file
6. grade          → Get final score
```

## Scripts Structure

```
scripts/
├── operate_cli.py           # CLI wrapper
├── for_datanode.py          # Tool functions (from operate_submission/)
├── _submission_utils.py     # Shared utilities (from operate_submission/)
├── base_dataloader.py       # BaseDataLoader ABC
└── grading_server_runner.py # Standalone grading server
```
