## Task

{task_description}

## Node Info

- Node ID: `{node_id}`
- Workspace: `{task_workspace}`

## Data Links Directory

Path: `{data_links_dir}`

This directory is shared across all nodes. You can browse existing cleaned datasets from other nodes and save your own intermediate results here.

Check what's available:
```bash
ls -la {data_links_dir}/
```

Use memory_tree tools to see registered datasets:
- `show_all_data` — Browse all datasets with descriptions
- `show_detailed_data` — Get full details on a specific dataset

When you save cleaned data here, use descriptive filenames like `cleaned_gsm8k_v1.jsonl`, `deduped_math_combined.jsonl`, `augmented_bfcl_normalized.jsonl`.

Register your datasets with `add_new_data` so other nodes can discover them.

## Dataset Manifest

```json
{dataset_manifest_json}
```

## Global Pool Manifest

Path: `{global_pool_manifest_path}`

```json
{global_pool_manifest_summary_json}
```

## Upstream Black Handoff

Path: `{parent_black_handoff_path}`

```json
{parent_black_handoff_summary_json}
```

Treat this handoff as a useful starting point for source selection and filtering, but prefer current evidence from local data files and the current benchmark goal.

## Parent Inspect Report

```json
{inspect_report_json}
```

## Dataset Probe Summary

The framework has already prepared probe rows for the current data pool.

- Probe directory: `{probe_dir}`
- Probe summary path: `{probe_summary_path}`

```json
{probe_summary_json}
```

## Data Exploration Suggestions

You have full freedom to explore and clean data however you see fit. Here are some patterns that often work well (not mandatory):

**Quick data inspection:**
```bash
wc -l /path/to/dataset.jsonl
head -5 /path/to/dataset.jsonl | python3 -m json.tool
```

**Quality assessment:**
```bash
python3 -c "
import json
with open('/path/to/dataset.jsonl') as f:
    rows = [json.loads(l) for l in f]
print(f'Total: {{len(rows)}}')
print(f'Fields: {{list(rows[0].keys())}}')
missing = sum(1 for r in rows if not r.get('output'))
print(f'Missing output: {{missing}}')
"
```

**Batch cleaning:** Write a Python script, run it via bash, save results to `{data_links_dir}/`.

You can also:
- Check `{data_links_dir}/` for existing cleaned datasets from other nodes
- Use memory_tree `show_all_data` to see what's been registered
- Use `read_global_memory` to learn from prior attempts

## Final Output Paths

Write your outputs to these exact paths:

- Final training file (required): `{train_jsonl_path}`
- Training config (required): `{train_config_path}`
- Prepare script (optional): `{prepare_data_script_path}`
- Preparation report (optional): `{prep_report_path}`

## Input Data Usage Rules

- Start small: prefer a compact, high-signal training set for the first attempt instead of dumping every possible row.
- Keep `max_samples` at or below 3000 unless you have strong evidence that more data will help. Training time scales linearly with sample count.
- **Progressive validation strategy**: In the early rounds of tree search (especially your first 1-2 attempts on a new dataset), use only 300-500 samples with `num_train_epochs: 0.5` to quickly validate whether the data direction is promising. This keeps each validation run under 5 minutes. Only scale up to 1000-3000 samples once the inspect report confirms the data is improving benchmark scores.
- If the memory/inspect summary from prior nodes shows that a particular data strategy already failed, do NOT repeat the same approach at full scale — either pivot to a different data mix or keep the sample count minimal for a quick sanity check.

## Hardware & Training Speed Hints

- The training hardware is NVIDIA H20 (96 GB HBM3). You can safely use a large `per_device_train_batch_size` (e.g. 4-16) for LoRA SFT on models up to ~8B parameters to speed up training.
- When you increase `per_device_train_batch_size`, reduce `gradient_accumulation_steps` proportionally to keep the effective batch size similar.
- Faster iteration is better: a quick, focused training run gives you signal sooner.

## Required `train.jsonl` Row Contract

Each JSONL row must contain:

- `instruction`: non-empty string
- `input`: string, use `""` when there is no extra input
- `output`: non-empty string
- `metadata`: object

`metadata` must contain:

- `source_id`: non-empty string

Optional `metadata` keys:

- `topic`
- `difficulty`
- `tags`
- `raw_id`

Metadata schema rules:
- Keep each metadata field type consistent across all rows.
- If `raw_id` is present, write it as a string for every row.
- `topic` and `difficulty` must be strings when present.
- `tags` must be a list of strings when present.
- Mixed types like `{{"raw_id": "abc"}}` on one row and `{{"raw_id": 123}}` on another will fail validation.

## `train_config.json` Contract

Write a JSON object to `{train_config_path}`.

Allowed keys only:

- `num_train_epochs`
- `learning_rate`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `cutoff_len`
- `max_samples`

Guidance:
- Do not assume more samples are always better.
- Prefer smaller, cleaner first-attempt training configs when you are unsure.
- The framework will validate this file before training starts.

## Optional `prep_report.json` Contract

If you choose to write a prep_report.json, include these keys:

- `selected_sources`
- `raw_rows_seen`
- `rows_written`
- `duplicate_rows_removed`
- `notes`

`rows_written` should match the final line count of `{train_jsonl_path}`.

If you don't write this file, the framework will auto-generate a minimal version.

## Historical Attempts

{memory_summary}

## Preparation Validation Feedback

If the previous preparation output failed validation, the framework will populate this section.

- Validation feedback path: `{prep_feedback_path}`

```json
{prep_feedback_json}
```

Summary: {prep_feedback_summary}

## Training Config Validation Feedback

If the previous training config failed validation, the framework will populate this section.

- Training config feedback path: `{train_config_feedback_path}`

```json
{train_config_feedback_json}
```

Summary: {train_config_feedback_summary}

## Validation Before Finish

Before you call `finish`, validate your outputs using the validation tools:

1. Call `validate_train_data` with `train_jsonl_path="{train_jsonl_path}"`
2. Call `validate_train_config` with `train_config_path="{train_config_path}"`

If either tool reports `"status": "failed"`, fix the reported issues and re-validate.
Do not call `finish` until both return `"status": "passed"`.

## Final Response

Return a single JSON block with any useful summary of:

- `selected_sources`
- `raw_rows_seen`
- `rows_written`
- `duplicate_rows_removed`
- `notes`

Do not include prose outside the JSON block.
