from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams, ToolError

from ..utils.submit import SubmitError, compact_submit_result, format_submit_observation, submit_training_eval


class SubmitParams(BaseToolParams):
    """Train a model from a prepared Alpaca JSONL file, run benchmark evaluation, and return score plus artifact paths.

Use this after preparing and validating a training data file.
The tool creates a new independent submit trial under artifacts/submits/{node_id}/.
It returns both human-readable diagnostics and structured fields including score,
eval_path, checkpoint_path, and trial_path.

Training hyperparameters are optional. Either pass `train_config` as a JSON object
or path, or pass individual hyperparameter fields directly. If omitted, the
runner uses the experiment training_defaults."""

    name: ClassVar[str] = "submit"
    train_config: Any = Field(default=None, description="Optional training config JSON object or absolute path to train_config.json")
    train_data_path: str = Field(description="Absolute path to the Alpaca JSONL training data")
    benchmark: str = Field(description="Benchmark name, e.g. aime_2025")
    node_id: str = Field(description="Current black node id")
    num_train_epochs: float | None = Field(default=None, description="Optional number of SFT epochs")
    learning_rate: float | None = Field(default=None, description="Optional SFT learning rate")
    per_device_train_batch_size: int | None = Field(default=None, description="Optional per-device train batch size")
    gradient_accumulation_steps: int | None = Field(default=None, description="Optional gradient accumulation steps")
    cutoff_len: int | None = Field(default=None, description="Optional training cutoff length")
    max_samples: int | None = Field(default=None, description="Optional max training samples")


class SubmitTool(BaseTool):
    name: ClassVar[str] = "submit"
    params_class: ClassVar[type[BaseToolParams]] = SubmitParams

    def __init__(self, config: Any):
        super().__init__()
        self.config = config

    def execute(self, session: Any, args_json: str) -> tuple[str, dict[str, Any]]:
        params = self.parse_params(args_json)
        workspace_path = getattr(getattr(session, "config", None), "workspace_path", None)
        if not workspace_path:
            raise ToolError("submit requires session.config.workspace_path")
        direct_train_config = {
            key: value
            for key, value in {
                "num_train_epochs": params.num_train_epochs,
                "learning_rate": params.learning_rate,
                "per_device_train_batch_size": params.per_device_train_batch_size,
                "gradient_accumulation_steps": params.gradient_accumulation_steps,
                "cutoff_len": params.cutoff_len,
                "max_samples": params.max_samples,
            }.items()
            if value is not None
        }
        train_config = params.train_config
        if train_config is None:
            train_config = direct_train_config
        elif isinstance(train_config, dict) and direct_train_config:
            train_config = {**train_config, **direct_train_config}
        try:
            result = submit_training_eval(
                task_workspace=workspace_path,
                config=self.config,
                node_id=params.node_id,
                benchmark=params.benchmark,
                train_config=train_config,
                train_data_path=params.train_data_path,
            ).to_dict()
        except SubmitError as exc:
            if exc.result:
                compact = compact_submit_result(exc.result)
                raise ToolError(format_submit_observation(compact) + f"\nerror={exc}") from exc
            raise ToolError(str(exc)) from exc

        compact = compact_submit_result(result)
        return format_submit_observation(compact), {"submit_result": compact}
