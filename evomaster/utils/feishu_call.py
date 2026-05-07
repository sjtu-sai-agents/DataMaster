"""Utility functions for Feishu message sending."""

import json
import logging

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageResponse,
    CreateMessageRequestBody,
)

logger = logging.getLogger(__name__)


def create_feishu_client(app_id: str, app_secret: str) -> lark.Client:
    """
    Create and configure Lark client.

    Args:
        app_id: Feishu APP_ID.
        app_secret: Feishu APP_SECRET.

    Returns:
        Configured Lark client instance.
    """
    return (
        lark.Client.builder()
        .app_id(app_id)
        .app_secret(app_secret)
        .log_level(lark.LogLevel.INFO)
        .build()
    )


def send_feishu_message(
    message_content: str,
    client: lark.Client,
    chat_id: str,
    msg_type: str = "text",
    receive_id_type: str = "chat_id",
) -> bool:
    """
    Send message to Feishu chat.

    Args:
        message_content: Message content to send.
        client: Lark client instance.
        chat_id: Feishu chat ID.
        msg_type: Message type (default: "text").
        receive_id_type: Receive ID type (default: "chat_id").

    Returns:
        True if message sent successfully, False otherwise.
    """
    try:
        content = json.dumps({"text": message_content})

        request: CreateMessageRequest = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type(msg_type)
                .content(content)
                .build()
            )
            .build()
        )

        response: CreateMessageResponse = client.im.v1.message.create(request)

        if not response.success():
            logger.error(
                f"Failed to send message: code={response.code}, msg={response.msg}"
            )
            return False

        logger.info("Message sent to Feishu successfully")
        return True

    except Exception as e:
        logger.error(f"Error sending message: {e}")
        return False


def send_feishu_message_with_creds(
    message_content: str,
    app_id: str,
    app_secret: str,
    chat_id: str,
    msg_type: str = "text",
    receive_id_type: str = "chat_id",
) -> bool:
    """
    Send message to Feishu chat with credentials (creates client internally).

    Args:
        message_content: Message content to send.
        app_id: Feishu APP_ID.
        app_secret: Feishu APP_SECRET.
        chat_id: Feishu chat ID.
        msg_type: Message type (default: "text").
        receive_id_type: Receive ID type (default: "chat_id").

    Returns:
        True if message sent successfully, False otherwise.
    """
    client = create_feishu_client(app_id, app_secret)
    return send_feishu_message(message_content, client, chat_id, msg_type, receive_id_type)