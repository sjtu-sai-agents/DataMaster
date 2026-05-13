---
name: memory-tools
description: Tree-structured memory management for node-based exploration. Manages global memory, data link records, and per-node manifest (TL;DR + recordings). Use when agents need to persist, share, or retrieve exploration knowledge.
license: Complete terms in LICENSE.txt
---

# Memory Tools

Tree-structured memory system for managing exploration history across nodes. Three memory types:

1. **Global Memory** (`global_memory.md`) - Cross-node knowledge and experience
2. **Data Link** (`data_link.json`) - External dataset records and comments
3. **Memory Tree** (`manifest.md` per node) - Per-node exploration history (TL;DR + recordings)

## File Structure

```
memory_tree/
├── global_memory.md
├── data_link.json
└── {node_id}/
    ├── manifest.md           # TL;DR + recordings (written via tools)
    └── storage/
        ├── trajectory.json   # Conversation history (auto-saved)
        ├── code.py           # Code backup
        ├── stdout.txt        # Execution output
        └── submission.csv    # Submission backup
```

## Manifest Format

```markdown
# Manifest

## TL;DR
{Overall summary of this node's exploration}

## Recordings
1. Recording 1: {Brief title}
{Detailed content}

2. Recording 2: {Brief title}
{Detailed content}
```

## CLI Usage

All commands require `-w /path/to/workspace`. Node-specific commands also require `-n NODE_ID`.

### Update Own Manifest

| Tool | CLI |
|------|-----|
| `update_current_summary` | `python scripts/memory_cli.py -w WORKSPACE update-summary -n NODE_ID --summary "..."` |
| `append_current_recordings` | `python scripts/memory_cli.py -w WORKSPACE append-recording -n NODE_ID --summary "title" --content "detail"` |
| `modify_current_recordings` | `python scripts/memory_cli.py -w WORKSPACE modify-recording -n NODE_ID --rid 1 --summary "new" --content "new"` |
| `delete_current_recordings` | `python scripts/memory_cli.py -w WORKSPACE delete-recording -n NODE_ID --rid 1` |

### Read Other Nodes' Memory

| Tool | CLI |
|------|-----|
| `get_current_tree` | `python scripts/memory_cli.py -w WORKSPACE tree` |
| `get_all_manifest` | `python scripts/memory_cli.py -w WORKSPACE all-manifest` |
| `get_parent_manifest` | `python scripts/memory_cli.py -w WORKSPACE parent-manifest -n NODE_ID` |
| `get_manifest_summary` | `python scripts/memory_cli.py -w WORKSPACE manifest-summary -n NODE_ID` |
| `get_manifest_all` | `python scripts/memory_cli.py -w WORKSPACE manifest-all -n NODE_ID` |
| `list_children` | `python scripts/memory_cli.py -w WORKSPACE list-children -n NODE_ID` |

### Access Storage Files

| Tool | CLI |
|------|-----|
| `get_node_code` | `python scripts/memory_cli.py -w WORKSPACE node-code -n NODE_ID` |
| `get_node_output` | `python scripts/memory_cli.py -w WORKSPACE node-output -n NODE_ID` |
| `get_node_trajectory` | `python scripts/memory_cli.py -w WORKSPACE node-trajectory -n NODE_ID` |

### Global Memory

| Tool | CLI |
|------|-----|
| `read_global_memory` | `python scripts/memory_cli.py -w WORKSPACE read-global` |
| `add_global_memory` | `python scripts/memory_cli.py -w WORKSPACE add-global --summary "title" --content "detail"` |

### Data Link

| Tool | CLI |
|------|-----|
| `show_all_data` | `python scripts/memory_cli.py -w WORKSPACE show-all-data` |
| `show_detailed_data` | `python scripts/memory_cli.py -w WORKSPACE show-data --dataset-id 1` |
| `add_new_data` | `python scripts/memory_cli.py -w WORKSPACE add-data --path /abs/path --desc "description"` |
| `add_data_record` | `python scripts/memory_cli.py -w WORKSPACE add-data-record --dataset-id 1 -n NODE_ID --comment "..."` |

## Usage Guidelines

1. **Before exploration**: Read global memory (`read-global`) and parent manifest (`parent-manifest`)
2. **During exploration**: Append recordings as findings emerge (`append-recording`)
3. **On important findings**: Update global memory (`add-global`)
4. **On node completion**: Update TL;DR summary (`update-summary`)
5. **For external data**: RED nodes create data link entries (`add-data`), BLACK nodes add score feedback (`add-data-record`)

## Scripts Structure

```
scripts/
├── memory_tree.py            # Core implementation (from playground/search_dataset_tools/)
├── memory_tree_interface.py  # MCP interface (from playground/search_dataset_tools/)
└── memory_cli.py             # CLI wrapper
```
