"""FeishuBot main class

Lifecycle management: initialization -> receive events -> parse -> deduplicate -> dispatch -> return results.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import lark_oapi as lark
from lark_oapi.api.im.v1 import P2ImMessageReceiveV1, P2ImMessageMessageReadV1, P2ImMessageRecalledV1
from lark_oapi.event.callback.model.p2_card_action_trigger import (
    P2CardActionTrigger,
    P2CardActionTriggerResponse,
    CallBackCard,
    CallBackToast,
)

from .messaging.client import create_feishu_client
from .config import FeishuBotConfig
from .dedup import MessageDedup
from .dispatcher import TaskDispatcher
from .messaging.document import FeishuDocumentWriter
from .event_handler import parse_event
from .messaging.sender import send_card_message, send_text_message
from .step_reporter import FeishuStepReporter

logger = logging.getLogger(__name__)

# Use card messages for content exceeding this length (supports longer content and Markdown)
_CARD_THRESHOLD = 2000

# Regex for the /agent <name> <task> command
_COMMAND_RE = re.compile(r"^/agent\s+(\S+)\s+(.+)$", re.DOTALL)


# ---------------------------------------------------------------------------
# Monkey-patch: lark-oapi 1.5.3 ws.Client silently drops CARD messages.
# Fix: route CARD through do_without_validation(), same as EVENT.
# ---------------------------------------------------------------------------
def _patch_ws_client_for_card_actions() -> None:
    """Patch lark.ws.Client._handle_data_frame to process CARD messages."""
    from lark_oapi.ws.enum import MessageType

    _original = lark.ws.Client._handle_data_frame

    async def _patched_handle_data_frame(self, frame):
        from lark_oapi.ws.const import (
            HEADER_MESSAGE_ID, HEADER_TRACE_ID,
            HEADER_SUM, HEADER_SEQ, HEADER_TYPE, HEADER_BIZ_RT,
        )
        from lark_oapi.core.const import UTF_8
        from lark_oapi.ws.model import Response
        import base64
        import http
        import time as _time

        hs = frame.headers  # protobuf RepeatedCompositeFieldContainer
        type_ = None
        for h in hs:
            if h.key == HEADER_TYPE:
                type_ = h.value
                break

        if type_ is None:
            return await _original(self, frame)

        try:
            message_type = MessageType(type_)
        except ValueError:
            return await _original(self, frame)

        # Only intercept CARD; let everything else go to original
        if message_type != MessageType.CARD:
            return await _original(self, frame)

        # --- CARD handling (copied structure from EVENT handling) ---
        msg_id = None
        sum_ = "1"
        seq = "0"
        for h in hs:
            if h.key == HEADER_MESSAGE_ID:
                msg_id = h.value
            elif h.key == HEADER_SUM:
                sum_ = h.value
            elif h.key == HEADER_SEQ:
                seq = h.value

        pl = frame.payload
        if int(sum_) > 1:
            pl = self._combine(msg_id, int(sum_), int(seq), pl)
            if pl is None:
                return

        resp = Response(code=http.HTTPStatus.OK)
        try:
            start = int(round(_time.time() * 1000))
            result = self._event_handler.do_without_validation(pl)
            end = int(round(_time.time() * 1000))
            header = hs.add()
            header.key = HEADER_BIZ_RT
            header.value = str(end - start)
            if result is not None:
                from lark_oapi.core.json import JSON
                resp.data = base64.b64encode(JSON.marshal(result).encode(UTF_8))
        except Exception as e:
            logger.error("Handle CARD message failed: msg_id=%s, err=%s", msg_id, e)
            resp = Response(code=http.HTTPStatus.INTERNAL_SERVER_ERROR)

        from lark_oapi.core.json import JSON
        frame.payload = JSON.marshal(resp).encode(UTF_8)

    lark.ws.Client._handle_data_frame = _patched_handle_data_frame
    logger.info("Patched lark.ws.Client._handle_data_frame to support CARD messages")


_patch_ws_client_for_card_actions()


class FeishuBot:
    """Feishu Bot main class."""

    def __init__(
        self,
        config: FeishuBotConfig,
        project_root: str | Path,
    ):
        """
        Args:
            config: Feishu Bot configuration.
            project_root: EvoMaster project root directory.
        """
        self._config = config
        self._project_root = Path(project_root)

        # Create Feishu Client
        self._client = create_feishu_client(
            app_id=config.app_id,
            app_secret=config.app_secret,
            domain=config.domain,
        )

        # Message deduplication
        self._dedup = MessageDedup()

        # Get the bot's own open_id (used for group chat @mention filtering)
        self._bot_open_id = self._fetch_bot_open_id()

        # Real-time progress report factory
        client = self._client
        doc_writer = FeishuDocumentWriter(
            client,
            folder_token=config.doc_folder_token,
            domain=config.domain,
        )

        def _create_step_reporter(chat_id: str, reply_to_message_id: str | None = None, sender_open_id: str | None = None):
            """Create a FeishuStepReporter for the given chat."""
            return FeishuStepReporter(client, chat_id, reply_to_message_id, document_writer=doc_writer, sender_open_id=sender_open_id)

        # Task dispatcher
        self._dispatcher = TaskDispatcher(
            project_root=self._project_root,
            default_agent=config.default_agent,
            default_config_path=config.default_config_path,
            max_workers=config.max_concurrent_tasks,
            task_timeout=config.task_timeout,
            max_sessions=getattr(config, "max_sessions", 100),
            on_result=self._send_result,
            step_reporter_factory=_create_step_reporter,
            feishu_app_id=config.app_id,
            feishu_app_secret=config.app_secret,
            feishu_domain=config.domain,
            feishu_doc_folder_token=config.doc_folder_token,
        )

        self._ws_client: Optional[lark.ws.Client] = None

    def _handle_message_event(self, data: P2ImMessageReceiveV1) -> None:
        """Handle a received message event."""
        ctx = parse_event(data.event)
        if ctx is None:
            logger.warning("Failed to parse event, skipping")
            return

        # Deduplicate
        if not self._dedup.try_record_message(ctx.message_id, scope=ctx.chat_id):
            return

        # Permission check
        if self._config.allow_from and ctx.sender_open_id not in self._config.allow_from:
            logger.info("Message from unauthorized user: %s", ctx.sender_open_id)
            send_text_message(
                self._client,
                ctx.chat_id,
                "抱歉，您没有权限使用此 Bot。",
                reply_to_message_id=ctx.message_id,
            )
            return

        # Group chat: only process messages that @mention the bot
        if ctx.chat_type == "group":
            if not ctx.mentions or self._bot_open_id not in ctx.mentions:
                logger.debug("Ignoring group message not mentioning bot: %s", ctx.message_id)
                return

        # Ignore non-text messages
        if ctx.message_type not in ("text", "post"):
            logger.debug("Ignoring non-text message: %s", ctx.message_type)
            return

        # Parse command
        agent_name, task_text = self._parse_command(ctx.content)

        if not task_text.strip():
            send_text_message(
                self._client,
                ctx.chat_id,
                "请提供任务描述。\n用法：直接发送消息对话，或使用 /agent <agent名称> <任务描述>\n命令：/new（新会话）、/shutdown（关闭 Bot）",
                reply_to_message_id=ctx.message_id,
            )
            return

        # Special commands are dispatched directly without sending a confirmation message
        stripped = task_text.strip()
        if stripped in ("/new", "/shutdown"):
            self._dispatcher.dispatch(
                chat_id=ctx.chat_id,
                message_id=ctx.message_id,
                task_text=task_text,
            )
            return

        # Dispatch task
        self._dispatcher.dispatch(
            chat_id=ctx.chat_id,
            message_id=ctx.message_id,
            task_text=task_text,
            agent_name=agent_name,
            sender_open_id=ctx.sender_open_id,
        )

    def _handle_message_read_event(self, data: P2ImMessageMessageReadV1) -> None:
        """Handle message-read events (ignored; registered only to prevent SDK errors)."""
        pass

    def _handle_message_recalled_event(self, data: P2ImMessageRecalledV1) -> None:
        """Handle message-recalled events (ignored; registered only to prevent SDK errors)."""
        pass

    def _handle_card_action(self, data: P2CardActionTrigger) -> P2CardActionTriggerResponse:
        """Handle card button click events."""
        resp = P2CardActionTriggerResponse()

        try:
            event = data.event
            action_value = event.action.value or {}
            chat_id = event.context.open_chat_id
            card_message_id = event.context.open_message_id
            operator_id = event.operator.open_id

            action = action_value.get("action", "")
            logger.info(
                "Card action received: action=%s, chat_id=%s, operator=%s",
                action, chat_id, operator_id,
            )

            if action == "confirm_agent_build":
                session_key = action_value.get("session_key", "")
                agent_name = action_value.get("agent_name", "")

                if not session_key or not agent_name:
                    logger.warning("Missing session_key or agent_name in card action")
                    toast = CallBackToast()
                    toast.type = "error"
                    toast.content = "参数缺失，请重新发起 /agent agent_builder 命令"
                    resp.toast = toast
                    return resp

                original_answer = action_value.get("original_answer", "")

                self._dispatcher.dispatch_card_action(
                    chat_id=chat_id,
                    session_key=session_key,
                    agent_name=agent_name,
                    task_text="确认",
                    sender_open_id=operator_id,
                    card_message_id=card_message_id,
                    original_answer=original_answer,
                )

                # Update the card in-place via callback response: keep original content, remove buttons, append status line
                import json
                from .messaging.sender import _build_card_json
                content_parts = []
                if original_answer:
                    content_parts.append(original_answer)
                content_parts.append("---")
                content_parts.append("> ⏳ 方案已确认，正在生成 Agent 文件...")
                content = "\n\n".join(content_parts)

                card_dict = json.loads(_build_card_json(
                    title="⏳ Agent 生成中...",
                    content=content,
                    header_template="wathet",
                ))
                card = CallBackCard()
                card.type = "raw"
                card.data = card_dict
                resp.card = card

                toast = CallBackToast()
                toast.type = "info"
                toast.content = "正在生成 Agent..."
                resp.toast = toast
                return resp

            elif action == "cancel_agent_build":
                session_key = action_value.get("session_key", "")
                agent_name = action_value.get("agent_name", "")

                if session_key:
                    self._dispatcher._session_manager.remove(session_key)
                    logger.info("Cancelled and removed session: %s", session_key)

                # Update the card in-place via callback response: keep original content, remove buttons, append cancellation status
                import json
                from .messaging.sender import _build_card_json
                original_answer = action_value.get("original_answer", "")
                content_parts = []
                if original_answer:
                    content_parts.append(original_answer)
                content_parts.append("---")
                content_parts.append("> ❌ Agent 生成已取消。")
                content = "\n\n".join(content_parts)

                card_dict = json.loads(_build_card_json(
                    title="❌ 已取消",
                    content=content,
                    header_template="red",
                ))
                card = CallBackCard()
                card.type = "raw"
                card.data = card_dict
                resp.card = card

                toast = CallBackToast()
                toast.type = "info"
                toast.content = "已取消"
                resp.toast = toast
                return resp

            elif action == "answer_question":
                session_key = action_value.get("session_key", "")
                agent_name = action_value.get("agent_name", "")
                answer_text = action_value.get("answer_text", "")

                if session_key and answer_text:
                    self._dispatcher.dispatch_card_action(
                        chat_id=chat_id,
                        session_key=session_key,
                        agent_name=agent_name,
                        task_text=answer_text,
                        sender_open_id=operator_id,
                        card_message_id=card_message_id,
                        action_type="answer_question",
                    )

                # Immediately update card: remove buttons, show selected option (keep original question)
                import json
                from .messaging.sender import _build_card_json

                original_question = action_value.get("original_question", "")
                parts = []
                if original_question:
                    parts.append(original_question)
                    parts.append("---")
                parts.append(f"你选择了: **{answer_text}**")
                parts.append("\n> 正在继续处理...")
                content = "\n\n".join(parts)

                card_dict = json.loads(_build_card_json(
                    title="💬 已回复",
                    content=content,
                    header_template="wathet",
                ))
                card = CallBackCard()
                card.type = "raw"
                card.data = card_dict
                resp.card = card

                toast = CallBackToast()
                toast.type = "info"
                toast.content = f"已选择: {answer_text}"
                resp.toast = toast
                return resp

            else:
                logger.warning("Unknown card action: %s", action)
                toast = CallBackToast()
                toast.type = "warning"
                toast.content = f"未知操作: {action}"
                resp.toast = toast
                return resp

        except Exception:
            logger.exception("Error handling card action")
            toast = CallBackToast()
            toast.type = "error"
            toast.content = "处理按钮操作时出错"
            resp.toast = toast
            return resp

    def _parse_command(self, text: str) -> tuple[Optional[str], str]:
        """Parse the command prefix.

        Supported formats:
            /agent <name> <task>    -> (name, task)
            <task>                  -> (None, task)

        Returns:
            (agent_name, task_text)
        """
        match = _COMMAND_RE.match(text.strip())
        if match:
            return match.group(1), match.group(2).strip()
        return None, text

    def _fetch_bot_open_id(self) -> str:
        """Fetch the bot's own open_id at startup (used for group chat @mention filtering)."""
        import json as _json
        try:
            from lark_oapi.core.model import BaseRequest
            from lark_oapi.core import HttpMethod, AccessTokenType

            req = BaseRequest.builder() \
                .http_method(HttpMethod.GET) \
                .uri("/open-apis/bot/v3/info/") \
                .token_types({AccessTokenType.TENANT}) \
                .build()
            resp = self._client.request(req)
            body = _json.loads(resp.raw.content)
            bot_info = body.get("bot", {})
            open_id = bot_info.get("open_id", "")
            if open_id:
                logger.info("Bot open_id: %s", open_id)
            else:
                logger.warning("Failed to get bot open_id from API response: %s", body)
            return open_id
        except Exception:
            logger.exception("Failed to fetch bot open_id")
            return ""

    def _send_result(self, chat_id: str, message_id: str, result_text: str) -> None:
        """Result callback: send result to Feishu.

        Short text uses plain text messages; long text uses card messages (supports Markdown).
        """
        if len(result_text) > _CARD_THRESHOLD:
            # Use card message for long content
            send_card_message(
                self._client,
                chat_id,
                title="任务完成",
                content=result_text,
                reply_to_message_id=message_id,
            )
        else:
            send_text_message(
                self._client,
                chat_id,
                result_text,
                reply_to_message_id=message_id,
            )

    def start(self) -> None:
        """Start the bot (blocking).

        Currently supports WebSocket mode.
        """
        logger.info(
            "Starting FeishuBot: agent=%s, mode=%s",
            self._config.default_agent,
            self._config.connection_mode,
        )

        # Build event handler
        event_handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._handle_message_event)
            .register_p2_im_message_message_read_v1(self._handle_message_read_event)
            .register_p2_im_message_recalled_v1(self._handle_message_recalled_event)
            .register_p2_card_action_trigger(self._handle_card_action)
            .build()
        )

        if self._config.connection_mode == "websocket":
            self._ws_client = (
                lark.ws.Client(
                    self._config.app_id,
                    self._config.app_secret,
                    event_handler=event_handler,
                    log_level=lark.LogLevel.INFO,
                )
            )
            logger.info("FeishuBot is running via WebSocket. Press Ctrl+C to stop.")
            self._ws_client.start()
        else:
            raise ValueError(
                f"Unsupported connection mode: {self._config.connection_mode}. "
                "Currently only 'websocket' is supported."
            )

    def stop(self) -> None:
        """Stop the bot.

        Waits for active tasks to complete before shutting down (up to task_timeout seconds).
        """
        logger.info("Stopping FeishuBot, waiting for active tasks to finish...")
        self._dispatcher.shutdown(wait=True)

        if self._ws_client is not None:
            try:
                # lark-oapi ws.Client uses daemon threads, which terminate automatically on process exit.
                # If the SDK provides a stop/close method in the future, call it here.
                logger.debug("WebSocket client reference cleared")
            except Exception as e:
                logger.warning("Error cleaning up WebSocket client: %s", e)
            finally:
                self._ws_client = None

        logger.info("FeishuBot stopped")
