"""Feishu message sending

Send text messages, card messages, and reply messages to Feishu. Supports in-place card message updates (PATCH).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    PatchMessageRequest,
    PatchMessageRequestBody,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

logger = logging.getLogger(__name__)

# Card message content limit (Feishu interactive messages ~30KB, using a conservative 15KB)
_MAX_CARD_CONTENT_LENGTH = 15000


def _build_card_json(
    title: str,
    content: str,
    header_template: str = "blue",
) -> str:
    """Build a card JSON string."""
    if len(content) > _MAX_CARD_CONTENT_LENGTH:
        content = content[:_MAX_CARD_CONTENT_LENGTH] + "\n\n...(内容过长已截断)"

    return json.dumps({
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": header_template,
        },
        "elements": [
            {"tag": "markdown", "content": content},
        ],
    })


def build_card_with_actions(
    title: str,
    content: str,
    actions: list[dict],
    header_template: str = "blue",
) -> str:
    """Build a card JSON string with action buttons.

    Args:
        title: Card title.
        content: Card Markdown content.
        actions: Button list, each item format:
            {"text": "Confirm", "type": "primary", "value": {"action": "confirm"}}
            type options: "default", "primary", "danger"
        header_template: Card header color template.

    Returns:
        Card JSON string.
    """
    if len(content) > _MAX_CARD_CONTENT_LENGTH:
        content = content[:_MAX_CARD_CONTENT_LENGTH] + "\n\n...(内容过长已截断)"

    elements: list[dict] = [{"tag": "markdown", "content": content}]

    if actions:
        action_items = []
        for a in actions:
            action_items.append({
                "tag": "button",
                "text": {"tag": "plain_text", "content": a["text"]},
                "type": a.get("type", "default"),
                "value": a.get("value", {}),
            })
        elements.append({"tag": "action", "actions": action_items})

    return json.dumps({
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": header_template,
        },
        "elements": elements,
    })


def send_text_message(
    client: lark.Client,
    chat_id: str,
    text: str,
    reply_to_message_id: str | None = None,
) -> bool:
    """Send or reply with a text message.

    Args:
        client: Feishu Client instance.
        chat_id: Chat ID.
        text: Message text.
        reply_to_message_id: Message ID to reply to (optional).

    Returns:
        True on successful send.
    """
    content = json.dumps({"text": text})

    try:
        if reply_to_message_id:
            request = (
                ReplyMessageRequest.builder()
                .message_id(reply_to_message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type("text")
                    .content(content)
                    .build()
                )
                .build()
            )
            response = client.im.v1.message.reply(request)
        else:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("text")
                    .content(content)
                    .build()
                )
                .build()
            )
            response = client.im.v1.message.create(request)

        if not response.success():
            logger.error(
                "Failed to send message: code=%s, msg=%s",
                response.code,
                response.msg,
            )
            return False

        logger.debug("Message sent to chat %s", chat_id)
        return True

    except Exception:
        logger.exception("Error sending message to chat %s", chat_id)
        return False


def send_card_message(
    client: lark.Client,
    chat_id: str,
    title: str,
    content: str,
    reply_to_message_id: str | None = None,
    header_template: str = "blue",
    card_json: str | None = None,
) -> str | None:
    """Send a card message (supports Markdown formatting, suitable for longer content).

    Args:
        client: Feishu Client instance.
        chat_id: Chat ID.
        title: Card title.
        content: Card content (supports Feishu Markdown subset).
        reply_to_message_id: Message ID to reply to (optional).
        header_template: Card header color template.
        card_json: Pre-built card JSON (optional; when provided, title/content/header_template are ignored).

    Returns:
        The new message's message_id on success, None on failure.
    """
    card = card_json or _build_card_json(title, content, header_template)

    try:
        if reply_to_message_id:
            request = (
                ReplyMessageRequest.builder()
                .message_id(reply_to_message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type("interactive")
                    .content(card)
                    .build()
                )
                .build()
            )
            response = client.im.v1.message.reply(request)
        else:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("interactive")
                    .content(card)
                    .build()
                )
                .build()
            )
            response = client.im.v1.message.create(request)

        if not response.success():
            logger.error(
                "Failed to send card message: code=%s, msg=%s",
                response.code,
                response.msg,
            )
            return None

        message_id = response.data.message_id
        logger.debug("Card message sent to chat %s, message_id=%s", chat_id, message_id)
        return message_id

    except Exception:
        logger.exception("Error sending card message to chat %s", chat_id)
        return None


def upload_image(
    client: lark.Client,
    image_path: str,
) -> str | None:
    """Upload an image to Feishu IM and return the image_key.

    Args:
        client: Feishu Client instance.
        image_path: Local image file path.

    Returns:
        image_key on success, None on failure.
    """
    from lark_oapi.api.im.v1 import CreateImageRequest, CreateImageRequestBody

    try:
        with open(image_path, "rb") as f:
            request = (
                CreateImageRequest.builder()
                .request_body(
                    CreateImageRequestBody.builder()
                    .image_type("message")
                    .image(f)
                    .build()
                )
                .build()
            )
            response = client.im.v1.image.create(request)

        if not response.success():
            logger.error(
                "Failed to upload image: code=%s, msg=%s",
                response.code, response.msg,
            )
            return None

        image_key = response.data.image_key
        logger.debug("Image uploaded: %s -> %s", image_path, image_key)
        return image_key

    except Exception:
        logger.exception("Error uploading image %s", image_path)
        return None


def send_image_message(
    client: lark.Client,
    chat_id: str,
    image_key: str,
    reply_to_message_id: str | None = None,
) -> str | None:
    """Send an image message.

    Args:
        client: Feishu Client instance.
        chat_id: Chat ID.
        image_key: Uploaded image key (obtained from upload_image).
        reply_to_message_id: Message ID to reply to (optional).

    Returns:
        message_id on success, None on failure.
    """
    content = json.dumps({"image_key": image_key})

    try:
        if reply_to_message_id:
            request = (
                ReplyMessageRequest.builder()
                .message_id(reply_to_message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type("image")
                    .content(content)
                    .build()
                )
                .build()
            )
            response = client.im.v1.message.reply(request)
        else:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("image")
                    .content(content)
                    .build()
                )
                .build()
            )
            response = client.im.v1.message.create(request)

        if not response.success():
            logger.error(
                "Failed to send image message: code=%s, msg=%s",
                response.code, response.msg,
            )
            return None

        message_id = response.data.message_id
        logger.debug("Image message sent to chat %s, message_id=%s", chat_id, message_id)
        return message_id

    except Exception:
        logger.exception("Error sending image message to chat %s", chat_id)
        return None


# File extension -> Feishu file_type mapping
_FILE_TYPE_MAP = {
    ".pdf": "pdf",
    ".doc": "doc", ".docx": "doc",
    ".xls": "xls", ".xlsx": "xls",
    ".ppt": "ppt", ".pptx": "ppt",
    ".mp4": "mp4",
    ".opus": "opus", ".ogg": "opus",
}


def upload_file(
    client: lark.Client,
    file_path: str,
) -> str | None:
    """Upload a file to Feishu IM and return the file_key.

    Args:
        client: Feishu Client instance.
        file_path: Local file path.

    Returns:
        file_key on success, None on failure.
    """
    from lark_oapi.api.im.v1 import CreateFileRequest, CreateFileRequestBody

    p = Path(file_path)
    suffix = p.suffix.lower()
    file_type = _FILE_TYPE_MAP.get(suffix, "stream")
    file_name = p.name

    try:
        with open(file_path, "rb") as f:
            request = (
                CreateFileRequest.builder()
                .request_body(
                    CreateFileRequestBody.builder()
                    .file_type(file_type)
                    .file_name(file_name)
                    .file(f)
                    .build()
                )
                .build()
            )
            response = client.im.v1.file.create(request)

        if not response.success():
            logger.error(
                "Failed to upload file: code=%s, msg=%s",
                response.code, response.msg,
            )
            return None

        file_key = response.data.file_key
        logger.debug("File uploaded: %s -> %s (type=%s)", file_path, file_key, file_type)
        return file_key

    except Exception:
        logger.exception("Error uploading file %s", file_path)
        return None


def send_file_message(
    client: lark.Client,
    chat_id: str,
    file_key: str,
    reply_to_message_id: str | None = None,
) -> str | None:
    """Send a file message.

    Args:
        client: Feishu Client instance.
        chat_id: Chat ID.
        file_key: Uploaded file key (obtained from upload_file).
        reply_to_message_id: Message ID to reply to (optional).

    Returns:
        message_id on success, None on failure.
    """
    content = json.dumps({"file_key": file_key})

    try:
        if reply_to_message_id:
            request = (
                ReplyMessageRequest.builder()
                .message_id(reply_to_message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .msg_type("file")
                    .content(content)
                    .build()
                )
                .build()
            )
            response = client.im.v1.message.reply(request)
        else:
            request = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("file")
                    .content(content)
                    .build()
                )
                .build()
            )
            response = client.im.v1.message.create(request)

        if not response.success():
            logger.error(
                "Failed to send file message: code=%s, msg=%s",
                response.code, response.msg,
            )
            return None

        message_id = response.data.message_id
        logger.debug("File message sent to chat %s, message_id=%s", chat_id, message_id)
        return message_id

    except Exception:
        logger.exception("Error sending file message to chat %s", chat_id)
        return None


def patch_card_message(
    client: lark.Client,
    message_id: str,
    title: str = "",
    content: str = "",
    header_template: str = "blue",
    card_json: str | None = None,
) -> bool:
    """Update an already-sent card message in-place (PATCH API).

    Only supports updating interactive-type messages sent by the bot itself.

    Args:
        client: Feishu Client instance.
        message_id: Message ID to update.
        title: Updated card title.
        content: Updated card content.
        header_template: Card header color template.
        card_json: Pre-built card JSON (optional; when provided, title/content/header_template are ignored).

    Returns:
        True on successful update.
    """
    card = card_json or _build_card_json(title, content, header_template)

    try:
        request = (
            PatchMessageRequest.builder()
            .message_id(message_id)
            .request_body(
                PatchMessageRequestBody.builder()
                .content(card)
                .build()
            )
            .build()
        )
        response = client.im.v1.message.patch(request)

        if not response.success():
            logger.error(
                "Failed to patch card message: code=%s, msg=%s",
                response.code,
                response.msg,
            )
            return False

        logger.debug("Card message patched: %s", message_id)
        return True

    except Exception:
        logger.exception("Error patching card message %s", message_id)
        return False
