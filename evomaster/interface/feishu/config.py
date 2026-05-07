"""Feishu Bot configuration model and loading

Load Feishu Bot configuration from a YAML file, reusing EvoMaster's _substitute_env pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

# Reuse the environment variable substitution from evomaster.config
from evomaster.config import _substitute_env


class FeishuBotConfig(BaseModel):
    """Feishu Bot configuration."""

    app_id: str = Field(description="Feishu application App ID")
    app_secret: str = Field(description="Feishu application App Secret")
    domain: str = Field(
        default="https://open.feishu.cn",
        description="Feishu API domain",
    )
    connection_mode: str = Field(
        default="websocket",
        description="Connection mode: websocket or webhook",
    )
    default_agent: str = Field(
        default="chat_agent",
        description="Default playground agent name to use",
    )
    default_config_path: Optional[str] = Field(
        default=None,
        description="Default config file path (relative to project_root); if not set, uses configs/{agent}/config.yaml",
    )
    max_concurrent_tasks: int = Field(
        default=4,
        description="Maximum number of concurrent tasks",
    )
    task_timeout: int = Field(
        default=600,
        description="Timeout for a single task in seconds",
    )
    allow_from: List[str] = Field(
        default_factory=list,
        description="List of allowed user open_ids; empty list means all users are allowed",
    )
    doc_folder_token: Optional[str] = Field(
        default=None,
        description="Feishu folder token for storing trajectory documents; empty means root directory of the application",
    )
    max_sessions: int = Field(
        default=100,
        description="Maximum number of concurrent sessions",
    )

    class Config:
        extra = "allow"


def load_feishu_config(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> FeishuBotConfig:
    """Load Feishu Bot configuration.

    Args:
        config_path: Path to the configuration file.
        project_root: Project root directory, used for searching the .env file.

    Returns:
        A FeishuBotConfig instance.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Feishu config not found: {config_path}")

    # Load .env
    if load_dotenv is not None:
        if project_root:
            env_file = Path(project_root) / ".env"
            if env_file.exists():
                load_dotenv(env_file)
            else:
                load_dotenv()
        else:
            load_dotenv()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw = _substitute_env(raw)

    # Extract the feishu section
    feishu_section = raw.get("feishu", raw)

    return FeishuBotConfig(**feishu_section)
