## Task

{task_description}

## Observer Context

```json
{observer_context_json}
```

## Output Paths

- Latest advice: `{global_advice_path}`
- Advice history: `{observer_history_path}`

## Required JSON Schema

Return exactly one JSON object shaped like:

```json
{{
  "selected_next_node_id": "",
  "scheduler_reason": "short reason for selecting this pending node, or why UCT should decide",
  "global_strategy": "short tree-level strategy for the next few nodes",
  "node_advice": {{
    "<pending_node_id>": "specific advice for that pending node"
  }},
  "red_advice": "fallback advice for red dataset-search nodes",
  "black_advice": "fallback advice for black data-prep/training nodes"
}}
```

Rules:
- `selected_next_node_id` must be empty or one of the ids listed in `pending_nodes`.
- Prefer a valid pending node only when the context shows a concrete reason.
- Do not ask for a node that is not already pending.
- Keep advice concise and operational.
- Do not include markdown outside the JSON object.
