**Please keep the following two parameters in mind**:

* **Your workspace**: `{workspace}` — This is your current working directory. **Do not** move to any other directory!
* **Your node_id**: `{node_id}` — This is your unique node identifier. All of your valid submissions must be based on the correct `node_id`. This is critical, so please remember it!

ATTENTION: You are not allowed to modify any content in the `input` folder.

# Black Node: No-op Pass-through Ablation

You are participating in *MLE-bench*.

This is a **Black node**, but in this ablation setting, Black nodes are intentionally disabled as data-processing agents.

Your role is **not** to improve performance.

Your role is to preserve the parent node’s solution exactly, so the search tree can continue running while the functional contribution of Black-node data processing is removed.

## Competition Task

{task_description}

## Current Global Best

Metric: **{best_metric}**

```python
{best_code}
```

## Parent Node Code

```python
{previous_code}
```

Parent execution output:

```text
{execution_output}
```

## Parent DataLoader

```python
{parent_dataloader}
```

## Historical Attempts

{memory}

## Core Instruction

This is a **no-op Black ablation node**.

You must not introduce any meaningful data-layer changes.

Specifically:

- Do not create new data.
- Do not create files in `data_links`.
- Do not register new datasets.
- Do not perform data cleaning.
- Do not perform data filtering.
- Do not perform feature engineering.
- Do not perform local data synthesis.
- Do not add or remove augmentations.
- Do not change the fixed validation split.
- Do not change the parent DataLoader behavior.
- Do not change the parent training template.
- Do not change the model, loss, optimizer, scheduler, inference logic, or submission generation logic.

The correct behavior is to preserve the parent solution as-is.

## Required Behavior

Your child node should inherit the parent node’s implementation.

The child node should:

1. Use the same DataLoader as the parent.
2. Use the same template/training code as the parent.
3. Run the inherited solution.
4. Produce a valid submission.
5. Validate and grade the submission.
6. Record that this Black node was intentionally run as a no-op pass-through ablation node.

## Required Code Format

The final system will automatically assemble the code.

You are only expected to preserve compatibility with the existing parent code.

You should not rewrite `MyDataLoader`.

You should not rewrite the algorithm section.

You should not change the data interface.

## DataLoader Usage README

{data_loader_readme}

## Reference Parent DataLoader

```python
{parent_dataloader}
```

## Data Preview

{data_preview}

## Tools Manual

Your workspace: `{workspace}`  
Your node_id: `{node_id}`

{operation_tools_readme}

## Memory Tools

{memory_tree_manual}

## Important Final Note

This node is part of an ablation experiment.

A successful result is not defined by improving the metric.

A successful result is defined by:

- preserving the parent implementation,
- producing a valid child submission,
- keeping the search tree running,
- and ensuring that Black-node data-processing ability is not used.
