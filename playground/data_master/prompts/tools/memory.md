#### Usage Requirements

You need to systematically manage and use three types of memory:

1. **Global Memory** - Cross-node general knowledge and experience summaries
2. **Data Link Record** - Detailed information and comments on external datasets
3. **Memory Tree** - Detailed exploration history and findings for each node

- **Global Memory**: When you discover general knowledge valuable across multiple tasks
  - Examples: Best practices for data preprocessing, feature engineering techniques

- **Data Link**: When you find or use external datasets, or when you receive feedback after using external datasets in your code
  - Examples: Downloaded auxiliary datasets from HuggingFace, found related data from GitHub, or received feedback on dataset quality (good or bad?)

- **Memory Tree (manifest.md)**: Record the detailed exploration process of the current node
  - Examples: What methods were tried, results of each method, what was learned

You are recommended to use memory functions as follows:
1. **Before exploration**: Read relevant memory to understand existing experience
2. **When making important findings**: Immediately update global memory and data link records
3. **When node completes**: Update the current node's manifest.md summary

!ATTENTION!: You MUST:
- Read global memory at least once, and update global memory at least once when you make findings (whether successful or failed experiences)
- Update your own node's manifest at least once, and update multiple detailed recordings
- If you are a BLACK node: You need to update specific datalink records based on the score feedback from running the code (test set and validation set scores); if you are a RED node: You need to create new datalink entries for the new datasets you found!


#### Global Memory Usage

**Global Memory** (`global_memory.md`) stores general knowledge and experience summaries across tasks. This is a persistent knowledge base that all nodes can access and contribute to. The content structure is as follows:

```markdown
# Global Memory

1. {Knowledge Title}
   {Detailed Content}

2. {Another Knowledge Title}
   {Detailed Content}
```

1. `memory_tree_read_global_memory()`: Read all contents of global memory. **Strongly recommended to use before starting a task**.
2. `memory_tree_add_global_memory(memory_summary: str, memory_content: str)`: Add a new knowledge entry to global memory.

#### Data Link Record Usage

**Data Link** (`data_link.json`) stores detailed information and usage records of all external datasets.

**Content Structure**:
```json
{
  "Dataset Name": {
    "dataset_id": 1,
    "path": "/path/to/dataset",
    "init_description": "Initial description of the dataset",
    "comment": [
      {"node_id": "abc123", "comment": "First use comment"},
      {"node_id": "def456", "comment": "Second use comment"}
    ]
  }
}
```

3. `memory_tree_show_all_data()`: Display summaries of all recorded datasets.

4. `memory_tree_show_detailed_data(dataset_id: int)`: Display detailed information of a specified dataset.
    - `dataset_id`: The numeric ID of the dataset

5. `memory_tree_add_new_data(dataset_path: str, init_descriptions: str)`: Record a newly discovered dataset.
    - `dataset_path`: Absolute path to the dataset
    - `init_descriptions`: Initial description of the dataset

6. `memory_tree_add_data_record(dataset_id: int, node_id: str, comment: str)`: Add a usage comment for a dataset.
    - `dataset_id`: The numeric ID of the dataset
    - `node_id`: Current node ID (use the `{node_id}` variable)
    - `comment`: Usage comment or findings


#### Memory Tree and `manifest.md`

Memory Tree is a tree-structured file system for storing exploration history and knowledge summaries for each node. Each node has a corresponding folder structure in the tree:

```
memory_tree/
├── {node_id}/
│   ├── manifest.md           # Knowledge summary (written by you through tools)
│   └── storage/
│       ├── trajectory.json   # Multi-turn conversation records (auto-saved by system)
│       ├── code.py           # Node code backup
│       ├── stdout.txt        # Execution results and console output
│       └── submission.csv    # Submission file backup
```

`manifest.md` is the core knowledge summary file for each node, using the following format:

```markdown
# Manifest for `{node_id}`

## TL;DR

{Overall summary - High-level overview of the entire node exploration process, can be updated in real-time as you explore! Note: High-quality TLDR can help subsequent exploration be more efficient!}

## Recordings

1. Recording 1: {Brief title of this attempt}
{Detailed content of this attempt}

2. Recording 2: {Brief title of another attempt}
{Detailed content of another attempt}
```

- **TL;DR**: Overall goals and findings summary of the entire node
- **Recordings**: Record each important attempt and finding in chronological order
    - recording_summary uses a brief title (5-10 words)
    - recording_content describes the attempt content and results in detail

> Generally, your TLDR should not exceed 200 words, and Recordings can record about 5 entries with slightly richer detailed content

Available Tools:

- Update Your Own Memory
    1. `memory_tree_update_current_summary`
    Update the overall summary of the current node (TL;DR section).
    2. `memory_tree_append_current_recordings`
    Add a new recording entry.
    3. `memory_tree_modify_current_recordings`
    Modify existing recording content (full replacement).
    4. `memory_tree_delete_current_recordings`
    Delete a specified recording by recording_id

- Read Other Nodes' Memory
    5. `memory_tree_get_current_tree`
    View the complete node tree structure starting from the root node, through which you can understand other nodes' node IDs
    6. `memory_tree_get_all_manifest`
    Get TL;DR summaries of all nodes.
    7. `memory_tree_get_parent_manifest`
    Get the complete manifest of the parent node.
    8. `memory_tree_get_manifest_summary`
    Get the summary of a specified node, including the complete TL;DR content and Recordings summary (without detailed recording content).
    9. `memory_tree_get_manifest_all`
    Get the complete manifest of a specified node.

- Access Storage Files
    10. `memory_tree_get_node_code`
    Get the node's Python code.
    11. `memory_tree_get_node_output`
    Get the node's execution output.
    12. `memory_tree_get_node_trajectory`
    Get the complete conversation history of the node.
    13. `memory_tree_list_children`
    List all child nodes of a node.
