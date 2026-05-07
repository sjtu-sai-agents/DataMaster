from __future__ import annotations

import json
from typing import Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams


class ValidateTrainConfigParams(BaseToolParams):
    """Validate your training config file.
Call this after writing train_config.json to check whether it passes
the framework's validation rules.

Checks performed:
- File exists and is valid JSON
- Only allowed keys: num_train_epochs, learning_rate, per_device_train_batch_size,
  gradient_accumulation_steps, cutoff_len, max_samples
- All values are numeric (not boolean or string)
- Values are within allowed ranges

Returns a structured result with status ("passed" or "failed"), reason,
the effective config (with defaults filled in), and a list of specific issues."""

    name: ClassVar[str] = "validate_train_config"
    train_config_path: str = Field(description="Absolute path to train_config.json to validate")


class ValidateTrainConfigTool(BaseTool):
    name: ClassVar[str] = "validate_train_config"
    params_class: ClassVar[type[BaseToolParams]] = ValidateTrainConfigParams

    def execute(self, session: Any, args_json: str) -> tuple[str, dict[str, Any]]:
        params = self.parse_params(args_json)
        try:
            from playground.math_posttrain_datatree.core.utils.data import (
                validate_train_config as _validate_train_config,
            )

            result = _validate_train_config(params.train_config_path)
        except Exception as exc:
            result = {"status": "failed", "reason": f"validation error: {exc}"}

        observation = _format_train_config_observation(result)
        return observation, {"validation_result": result}


def _format_train_config_observation(result: dict[str, Any]) -> str:
    lines = [
        f"Status: {result.get('status', 'unknown')}",
        f"Reason: {result.get('reason', '')}",
    ]
    provided_keys = result.get("provided_keys")
    if provided_keys:
        lines.append(f"Provided keys: {', '.join(str(k) for k in provided_keys)}")
    effective = result.get("effective_config")
    if effective:
        lines.append(f"Effective config: {json.dumps(effective, ensure_ascii=False)}")
    issues = result.get("issues", [])
    if issues:
        lines.append("Issues:")
        for issue in issues[:10]:
            lines.append(f"  - {issue}")
    return "\n".join(lines)
