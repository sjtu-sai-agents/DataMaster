## Task

{task_description}

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

Use the upstream handoff as preference context for what kinds of sources or mixtures already looked promising. Grow the global pool by adding high-value new sources; do not assume old pool entries must be deleted.

Search efficiently:

- Do at most 3 dataset searches.
- Inspect at most 3 candidate READMEs.
- Prefer well-known public datasets over obscure variants.
- Only include supervised datasets with explicit problem/solution or question/answer examples.
- Do not include web-scale corpora, Common Crawl extracts, or general pretraining text collections such as `open-web-math/open-web-math`.
- Avoid million-document or token-count corpora; prefer benchmark-like, task-aligned, or instruction-style datasets.
- Finish once you have a concise shortlist.
- When you write the manifest, each dataset must use the real Hugging Face dataset id in `source_id` and `name`, for example `nlile/hendrycks-MATH-benchmark`.
- Do not use descriptive aliases like `MATH Dataset (Hendrycks)` or placeholder URLs like `agent_search`.
- Prefer a real `url` such as `https://huggingface.co/datasets/<org>/<dataset>`.

## Required Manifest JSON Schema

The runner only reads the JSON file at `{manifest_path}`. It expects a top-level key named exactly `datasets`.

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
        "relevance": "short note about why this source helps"
      }},
      "coverage_tags": ["competition_math"]
    }}
  ]
}}
```

Only `datasets` is required for the runner, but every selected dataset entry should include the fields above whenever you know them.

Before calling `finish`:
- write the JSON to `{manifest_path}`
- read the file back and verify that `datasets` exists and has at least 1 item
- make sure each chosen dataset uses canonical Hugging Face ids and URLs

Write concise notes only. The fixed runner will build the final manifest at:

`{manifest_path}`

Workspaces:

- worker workspace: `{workspace}`
- task workspace: `{task_workspace}`
