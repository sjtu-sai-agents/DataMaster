## Task

{task_description}

{benchmark_info}

## Node Info

- Node ID: `{node_id}`
- Workspace: `{task_workspace}`

## Available Context

- Data links: `{data_links_dir}`
- Global pool: `{global_pool_manifest_path}`

Use the global pool manifest as the data inventory. Each source entry should include the shared-cache `local_path`, a compact `sample`, and compact `notes`. If you need exact rows, inspect `local_path` directly.

## Global Pool Source Summary

{dataset_manifest_summary}

## Prior Signals

- Global pool: {global_pool_manifest_summary}
- Parent handoff: {parent_black_handoff_summary}
- Parent inspect: {inspect_summary_text}
- Memory: {memory_summary}
- Previous data validation: {prep_feedback_summary}

## Required Outputs

- Write Alpaca JSONL training data to `{train_jsonl_path}`.
- Optional reproducibility script: `{prepare_data_script_path}`.
- Optional preparation report: `{prep_report_path}`.

Prefer a compact, high-signal dataset. Keep `max_samples` at or below 10000 unless there is strong evidence to scale. After a submit, continue only if eval feedback suggests a concrete data fix.

## Required `train.jsonl` Row Contract

Each JSONL row must contain:

- `instruction`: non-empty string
- `input`: string, use `""` when there is no extra input
- `output`: non-empty string
- `metadata`: optional object

Use `metadata.source_id` when convenient for provenance. Metadata is diagnostic only.

## Optional submit Training Hyperparameters

If overriding defaults, pass these keys directly to `submit` or under its `train_config` object. You do not need to write a train_config.json file.

- `num_train_epochs`
- `learning_rate`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `cutoff_len`
- `max_samples`

`submit` validates these hyperparameters before training starts.

## Optional `prep_report.json`

If you write a prep report, include:

- `selected_sources`
- `raw_rows_seen`
- `rows_written`
- `duplicate_rows_removed`
- `notes`

If omitted, the framework auto-generates a minimal report.

## Completion Criteria

This black node is complete only when it has produced a benchmark-tested training data variant:

- `train.jsonl` exists at `{train_jsonl_path}`.
- `validate_train_data` reports `"status": "passed"` for that file.
- `submit` has been called with this node's training data:
  - `train_data_path`: `{train_jsonl_path}`
  - `benchmark`: the target benchmark name from the benchmark section above
  - `node_id`: `{node_id}`
- The final summary reports the data recipe and the submit outcome, including score and eval path when available.

If validation fails, the node is not complete until the data is fixed and validation passes.
If submit fails, report the submit failure reason and artifact paths in the final summary.

## Not To Do

- Do not inspect benchmark implementation source code, evaluator scripts, scorer files, or prompt templates.
- Do not infer the benchmark contract from eval artifacts when the contract is already stated above.
- Do not spend the main budget on repeated exploratory checks after you have enough evidence to build a training data variant.
- Do not call `finish` before `submit` has been attempted.

## Final Response

Return a single JSON block with any useful summary of:

- `selected_sources`
- `raw_rows_seen`
- `rows_written`
- `duplicate_rows_removed`
- `notes`

Do not include prose outside the JSON block.
