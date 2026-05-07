"""Feishu API communication layer

Encapsulates Feishu API interactions: client creation, message sending, and document operations.
"""

from .client import create_feishu_client
from .sender import (
    send_text_message,
    send_card_message,
    patch_card_message,
    build_card_with_actions,
    _build_card_json,
)
from .document import FeishuDocumentWriter

__all__ = [
    "create_feishu_client",
    "send_text_message",
    "send_card_message",
    "patch_card_message",
    "build_card_with_actions",
    "_build_card_json",
    "FeishuDocumentWriter",
]
