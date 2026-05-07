"""Feishu event parsing

Parse im.message.receive_v1 events and extract message context.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FeishuMessageContext:
    """Parsed Feishu message context."""

    chat_id: str
    message_id: str
    sender_open_id: str
    chat_type: str  # "p2p" or "group"
    content: str  # Plain text content
    message_type: str  # "text", "post", etc.
    mentions: list[str] | None = None  # List of @mentioned open_ids


def parse_event(event_data) -> Optional[FeishuMessageContext]:
    """Extract message context from a Feishu SDK event object.

    Args:
        event_data: lark_oapi event data (P2ImMessageReceiveV1Data).

    Returns:
        A FeishuMessageContext instance, or None on parse failure.
    """
    try:
        message = event_data.message
        sender = event_data.sender

        chat_id = message.chat_id
        message_id = message.message_id
        message_type = message.message_type
        chat_type = message.chat_type

        sender_open_id = sender.sender_id.open_id if sender and sender.sender_id else ""

        content = parse_message_content(message.content, message_type)

        # Extract @mention list
        mentions = []
        if hasattr(message, "mentions") and message.mentions:
            for mention in message.mentions:
                if hasattr(mention, "id") and mention.id:
                    open_id = getattr(mention.id, "open_id", None)
                    if open_id:
                        mentions.append(open_id)

        return FeishuMessageContext(
            chat_id=chat_id,
            message_id=message_id,
            sender_open_id=sender_open_id,
            chat_type=chat_type,
            content=content,
            message_type=message_type,
            mentions=mentions or None,
        )
    except Exception:
        logger.exception("Failed to parse Feishu event")
        return None


def parse_message_content(raw_content: str, message_type: str) -> str:
    """Parse message content into plain text.

    Args:
        raw_content: Feishu message content JSON string.
        message_type: Message type.

    Returns:
        Plain text content.
    """
    try:
        data = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        return raw_content or ""

    if message_type == "text":
        return data.get("text", "").strip()

    if message_type == "post":
        # post type is rich text; extract all text segments
        parts: list[str] = []
        # Try to extract the title (Chinese or English)
        title = data.get("title", "")
        if title:
            parts.append(title)

        content_blocks = data.get("content", [])
        for line in content_blocks:
            for element in line:
                tag = element.get("tag", "")
                if tag == "text":
                    parts.append(element.get("text", ""))
                elif tag == "a":
                    parts.append(element.get("text", element.get("href", "")))
                elif tag == "at":
                    # @mention tag, skip
                    pass
        return "\n".join(parts).strip()

    # Other types (image, file, etc.) return a type hint
    return f"[{message_type}]"
