## Task

{task_description}

{benchmark_info}

## Role

Search for public datasets that can improve the current benchmark weak spots.

Current search goal:

{search_goal}

## Historical Attempts

{memory_summary}

## Global Pool Manifest

Path: `{global_pool_manifest_path}`

```json
{global_pool_manifest_summary_json}
```

## Upstream Black Handoff

Path: `{input_black_handoff_path}`

```json
{input_black_handoff_json}
```

## Global Observer Advice

Path: `{global_advice_path}`

```json
{global_advice_json}
```

Use observer advice as current tree-level guidance for what this node should prioritize. If it conflicts with manifest validity, dataset safety, or the task contract below, follow the task contract.

Use the upstream handoff as preference context for what kinds of sources or mixtures already looked promising. Use the global pool first to avoid rediscovering sources that previous red nodes already registered. Grow the global pool by adding high-value new sources; do not assume old pool entries must be deleted.

Search pragmatically:

- Call `probe_dataset_rows` for each candidate you plan to include. Pass only `source_id`, `output_dir="{task_workspace}/artifacts/data_pool/red_probe_{node_id}"`, optional `split`, and optional `sample_count` (default 2). The tool uses datasets-server rows first and returns raw samples.
- Register the final shortlist with `write_dataset_manifest`; do not manually write the manifest file with filesystem tools.
- Prefer well-known public datasets over obscure variants.
- Only include supervised datasets with explicit problem/solution or question/answer examples.
- Do not include web-scale corpora, Common Crawl extracts, or general pretraining text collections such as `open-web-math/open-web-math`.
- Avoid million-document or token-count corpora; prefer benchmark-like, task-aligned, or instruction-style datasets.
- Do not select a dataset only because its README sounds relevant. The returned samples must have input fields and answer/solution fields compatible with the target benchmark testset format.
- Avoid repeated equivalent searches and stop once additional searches are unlikely to change the shortlist.
- Finish once you have enough sample-inspected, task-aligned sources for a concise shortlist.
- When you write the manifest, each dataset must use the real Hugging Face dataset id in `source_id` and `name`, for example `nlile/hendrycks-MATH-benchmark`.
- Do not use descriptive aliases like `MATH Dataset (Hendrycks)` or placeholder URLs like `agent_search`.
- Prefer a real `url` such as `https://huggingface.co/datasets/<org>/<dataset>`.

## Required Manifest JSON Schema

The runner reads the JSON file at `{manifest_path}`. Use `write_dataset_manifest` to create it and to update the shared global pool at `{global_pool_manifest_path}`. The manifest must have a top-level key named exactly `datasets`.

Do not write a top-level key named `sources`.
Do not leave only a prose summary.
Do not call `finish` unless the manifest file contains a non-empty `datasets` list.

Write a JSON object shaped like this:

```json
{{
  "manifest_id": "dataset_manifest_<node_id>",
  "datasets": [
    {{
      "source_id": "org/dataset",
      "name": "org/dataset",
      "url": "https://huggingface.co/datasets/org/dataset",
      "license": "unknown",
      "local_path": "",
      "split": "",
      "config": "",
      "task_type": "math_reasoning",
      "answer_style": "mixed",
      "difficulty": "unknown",
      "language": "en",
      "quality_signals": {{
        "relevance": "short note about why this source helps",
        "sample_evidence": "short note from probe_dataset_rows describing the inspected raw samples"
      }},
      "coverage_tags": ["competition_math"]
    }}
  ]
}}
```

Only `datasets` is required for the runner, but every selected dataset entry should include the fields above whenever you know them.

Before calling `finish`:
- call `write_dataset_manifest` with `manifest_path="{manifest_path}"`, `global_pool_manifest_path="{global_pool_manifest_path}"`, `node_id="{node_id}"`, `search_goal="{search_goal}"`, and your `datasets` list
- verify the tool reports `Status: passed` and a non-empty dataset count
- make sure each chosen dataset uses canonical Hugging Face ids and URLs
- make sure each chosen dataset has row-level sample evidence that its input/output shape matches the target benchmark

Write concise notes only. The fixed runner will build the final manifest at:

`{manifest_path}`

Workspaces:

- worker workspace: `{workspace}`
- task workspace: `{task_workspace}`
