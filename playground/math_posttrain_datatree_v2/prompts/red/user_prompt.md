## Task

{task_description}

{benchmark_info}

## Role

Search for public training datasets whose row-level distribution is close to the target benchmark and can improve the current benchmark weak spots.

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

Use the upstream handoff as preference context for what kinds of sources or mixtures already looked promising. Use the global pool first to avoid rediscovering sources that previous red nodes already registered. Grow the global pool by adding high-value new sources; do not assume old pool entries must be deleted.
If the handoff includes `best_submit`, those paths describe the upstream black node's best successful submit trial.

Search pragmatically:

- Call `probe_dataset_rows` for each candidate you plan to include. Pass only `source_id`, `output_dir="{task_workspace}/artifacts/data_pool/red_probe_{node_id}"`, optional `split`, and optional `sample_count` (default 2). The tool uses datasets-server rows first and returns raw samples.
- Register the final shortlist with `write_dataset_manifest`; do not manually write the manifest file with filesystem tools.
- Prefer well-known public datasets over obscure variants when their row samples match the benchmark distribution.
- Only include supervised datasets with explicit problem/solution or question/answer examples.
- Do not include web-scale corpora, Common Crawl extracts, or general pretraining text collections such as `open-web-math/open-web-math`.
- Avoid million-document or token-count corpora; prefer benchmark-like, task-aligned, or instruction-style datasets whose sampled rows resemble the benchmark.
- Do not select a dataset only because its README sounds relevant. The returned samples must be distribution-aligned with the target benchmark, not merely topically related.
- Use the probed raw samples to judge whether the candidate matches the target benchmark along these axes:
  - task family and domain;
  - difficulty level;
  - problem style and input shape;
  - expected answer form;
  - reasoning depth and solution format.
- Prefer sources whose sampled rows resemble the current benchmark's examples and evaluation contract. Avoid sources dominated by substantially different task formats, difficulty levels, answer styles, or solution structures unless the sampled rows show they can be filtered into the target distribution by black nodes.
- Avoid repeated equivalent searches and stop once additional searches are unlikely to change the shortlist.
- Finish once you have enough sample-inspected, task-aligned sources for a concise shortlist.
- When you write the manifest, each dataset must use the real Hugging Face dataset id in `source_id` and `name`, for example `nlile/hendrycks-MATH-benchmark`.
- Do not use descriptive aliases like `MATH Dataset (Hendrycks)` or placeholder URLs like `agent_search`.
- Prefer a real `url` such as `https://huggingface.co/datasets/<org>/<dataset>`.

## Required Global Manifest Update

Use `write_dataset_manifest` to update the shared global pool at `{global_pool_manifest_path}`. The global pool is the canonical manifest consumed by black nodes. Do not manually write manifest files with filesystem tools.

Do not write a top-level key named `sources`.
Do not leave only a prose summary.
Do not call `finish` unless the tool reports a non-empty dataset count.

Register dataset entries shaped like this:

```json
{{
  "source_id": "org/dataset",
  "name": "org/dataset",
  "url": "https://huggingface.co/datasets/org/dataset",
  "split": "",
  "config": "",
  "task_type": "math_reasoning",
  "quality_signals": {{
    "relevance": "short note about why this source helps",
    "sample_evidence": "short note from probe_dataset_rows describing the inspected raw samples",
    "distribution_match": "short note comparing sampled rows to the target benchmark on difficulty, answer form, and reasoning style"
  }},
  "coverage_tags": ["competition_math"]
}}
```

The runner will enrich the global pool with shared-cache `local_path`, one compact `sample`, and compact `notes` after your tool call.

Before calling `finish`:
- call `write_dataset_manifest` with `global_pool_manifest_path="{global_pool_manifest_path}"`, `node_id="{node_id}"`, `search_goal="{search_goal}"`, and your `datasets` list
- verify the tool reports `Status: passed` and a non-empty dataset count
- make sure each chosen dataset uses canonical Hugging Face ids and URLs
- make sure each chosen dataset has row-level sample evidence that its distribution, not only its input/output shape, matches the target benchmark

Write concise notes only. The final manifest is the global pool:

`{global_pool_manifest_path}`

Workspaces:

- worker workspace: `{workspace}`
- task workspace: `{task_workspace}`
