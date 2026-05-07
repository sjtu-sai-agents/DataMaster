## Task

{task_description}

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

## Final Output Paths

Write your outputs to these exact paths:

- Prepare script: `{prepare_data_script_path}`
- Final training file: `{train_jsonl_path}`
- Preparation report: `{prep_report_path}`
- Training config: `{train_config_path}`

## Input Data Usage Rules

- Start small: prefer a compact, high-signal training set for the first attempt instead of dumping every possible row.
- Keep `max_samples` at or below 3000 unless you have strong evidence that more data will help. Training time scales linearly with sample count.
- **Progressive validation strategy**: In the early rounds of tree search (especially your first 1-2 attempts on a new dataset), use only 300-500 samples with `num_train_epochs: 0.5` to quickly validate whether the data direction is promising. This keeps each validation run under 5 minutes. Only scale up to 1000-3000 samples once the inspect report confirms the data is improving benchmark scores. Do not waste 30 minutes training on 3000 samples when you are still uncertain about data quality.
- If the memory/inspect summary from prior nodes shows that a particular data strategy already failed, do NOT repeat the same approach at full scale — either pivot to a different data mix or keep the sample count minimal for a quick sanity check.

## Hardware & Training Speed Hints

- The training hardware is NVIDIA H20 (96 GB HBM3). You can safely use a large `per_device_train_batch_size` (e.g. 4-16h) for LoRA SFT on models up to ~8B parameters to speed up training.
- When you increase `per_device_train_batch_size`, reduce `gradient_accumulation_steps` proportionally to keep the effective batch size similar, unless you intentionally want a larger effective batch.
- Faster iteration is better: a quick, focused training run gives you signal sooner.


- Prefer reading the fully materialized local dataset files from the manifest.
- You may use probe sample files for quick inspection before writing the full script.
- Do not assume the framework will apply any hidden filtering after your script runs.
- The `train.jsonl` you write is the final training data that will be used for post-training.

## Required `train.jsonl` Row Contract

Each JSONL row must contain:

- `instruction`: non-empty string
- `input`: string, use `""` when there is no extra input
- `output`: non-empty string
- `metadata`: object

**CRITICAL: Answer Format Requirement**

For math/reasoning tasks, the `output` field MUST end with the final answer in this exact format:

```
ANSWER: <value>
```

Where `<value>` is the numerical answer or letter choice. This format is required for the evaluation scorer to extract answers correctly.

Examples:
- For numeric answers: `"output": "Step 1... Step 2... Therefore, ANSWER: 42"`
- For multiple choice: `"output": "Analysis... ANSWER: B"`

Do NOT use alternative formats like:
- ❌ `Final answer: 42`
- ❌ `\\boxed{{42}}`
- ❌ `The answer is 42.`

The evaluation scorer strictly requires `ANSWER:` prefix on the last line.

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
- Mixed types like `{{"raw_id": "abc"}}` on one row and `{{"raw_id": 123}}` on another will fail validation and later break training.

## Required `prep_report.json` Contract

The report must contain these keys:

- `selected_sources`
- `raw_rows_seen`
- `rows_written`
- `duplicate_rows_removed`
- `notes`

Important:
- `rows_written` must match the final line count of `{train_jsonl_path}`.

## Required `train_config.json` Contract

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

## Historical Attempts

{memory_summary}

## Preparation Validation Feedback

If the previous preparation output failed validation, the framework will populate this section.

- Validation feedback path: `{prep_feedback_path}`

```json
{prep_feedback_json}
```

Summary: {prep_feedback_summary}

Use the summary first, and read the report file only if you need more detail.

## Training Config Validation Feedback

If the previous training config failed validation, the framework will populate this section.

- Training config feedback path: `{train_config_feedback_path}`

```json
{train_config_feedback_json}
```

Summary: {train_config_feedback_summary}

Use the summary first, and read the report file only if you need more detail.

## Validation Before Finish

Before you call `finish`, validate your outputs using the validation tools:

1. Call `validate_train_data` with `train_jsonl_path=”{train_jsonl_path}”` and `prep_report_path=”{prep_report_path}”`
2. Call `validate_train_config` with `train_config_path=”{train_config_path}”`

If either tool reports `”status”: “failed”`, fix the reported issues and re-validate.
Do not call `finish` until both return `”status”: “passed”`.

These tools run the exact same checks the framework enforces, including metadata schema consistency. The framework will NOT repair your outputs after you finish — if validation fails post-finish, the node is marked as buggy.

When validation reports a field type or schema mismatch, fix the data itself before finishing. Do not assume matching line counts or file existence is enough.

## Final Response

Return a single JSON block with any useful summary of:

- `selected_sources`
- `raw_rows_seen`
- `rows_written`
- `duplicate_rows_removed`
- `notes`

Do not include prose outside the JSON block.
