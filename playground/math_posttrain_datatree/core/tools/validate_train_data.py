from __future__ import annotations

import json
from typing import Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams


class ValidateTrainDataParams(BaseToolParams):
    """Validate your prepared training data files.
Call this after writing train.jsonl (and optionally prep_report.json) to check
whether they pass the framework's validation rules.

Checks performed:
- train.jsonl exists and every line is valid JSON
- Each row has required fields: instruction (non-empty str), input (str),
  output (non-empty str), metadata (object with non-empty source_id)
- If prep_report_path is provided: checks required keys (selected_sources,
  raw_rows_seen, rows_written, duplicate_rows_removed, notes) and verifies
  rows_written matches the actual line count of train.jsonl

Returns a structured result with status ("passed" or "failed"), reason,
row count, selected sources, and a list of specific issues."""

    name: ClassVar[str] = "validate_train_data"
    train_jsonl_path: str = Field(description="Absolute path to train.jsonl to validate")
    prep_report_path: str = Field(
        default="",
        description="Absolute path to prep_report.json (optional). "
        "If provided, validates consistency between the report and train file.",
    )


class ValidateTrainDataTool(BaseTool):
    name: ClassVar[str] = "validate_train_data"
    params_class: ClassVar[type[BaseToolParams]] = ValidateTrainDataParams

    def execute(self, session: Any, args_json: str) -> tuple[str, dict[str, Any]]:
        params = self.parse_params(args_json)
        try:
            from playground.math_posttrain_datatree.core.utils.data import (
                validate_prepared_train_file,
            )

            result = validate_prepared_train_file(
                params.train_jsonl_path,
                params.prep_report_path or None,
            )
        except Exception as exc:
            result = {"status": "failed", "reason": f"validation error: {exc}"}

        observation = _format_train_data_observation(result)
        return observation, {"validation_result": result}


def _format_train_data_observation(result: dict[str, Any]) -> str:
    lines = [
        f"Status: {result.get('status', 'unknown')}",
        f"Reason: {result.get('reason', '')}",
    ]
    row_count = result.get("row_count")
    if row_count is not None:
        lines.append(f"Row count: {row_count}")
    sources = result.get("selected_sources")
    if sources:
        lines.append(f"Selected sources: {', '.join(str(s) for s in sources)}")
    issues = result.get("issues", [])
    if issues:
        lines.append("Issues:")
        for issue in issues[:10]:
            lines.append(f"  - {issue}")
    preview = result.get("preview_rows")
    if preview:
        lines.append(f"Preview (first row): {json.dumps(preview[0], ensure_ascii=False)[:300]}")
    return "\n".join(lines)
