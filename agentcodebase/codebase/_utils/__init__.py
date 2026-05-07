"""
Utilities module for AgentCodeBase.

This module provides shared utilities including:
- Global configuration management via Config
- Logging infrastructure via get_logger
- Feishu bot integration via FeishuBot

Example usage:
    >>> from codebase._utils import get_config, get_logger, get_feishu_bot
    >>>
    >>> # Load configuration
    >>> config = get_config()
    >>> feishu_url = config.feishu.url
    >>>
    >>> # Create logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Application started")
    >>>
    >>> # Send Feishu notification
    >>> bot = get_feishu_bot()
    >>> bot.send_message("Test completed successfully")
"""

from codebase._utils._config import Config, get_config
from codebase._utils._logging import get_logger
from codebase._utils._send_feishu_message import FeishuBot, get_feishu_bot, send_feishu_message

__all__ = [
    'Config',
    'get_config',
    'get_logger',
    'FeishuBot',
    'get_feishu_bot',
    'send_feishu_message',
]
