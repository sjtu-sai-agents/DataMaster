"""Feishu card real-time progress reporter

The card only displays execution progress and a document link; the full trajectory is written to a Feishu document.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import lark_oapi as lark
    from .messaging.document import FeishuDocumentWriter

logger = logging.getLogger(__name__)


class FeishuStepReporter:
    """Maintain a Feishu card showing execution progress; write the full trajectory to a Feishu document.

    Usage::

        reporter = FeishuStepReporter(client, chat_id, reply_to, document_writer=writer)
        reporter.send_initial_card("Calculate 3+5")
        playground.run(task_description=text, on_step=reporter.on_step)
        reporter.finalize("completed", "3+5=8")
    """

    def __init__(
        self,
        client: lark.Client,
        chat_id: str,
        reply_to_message_id: str | None = None,
        document_writer: FeishuDocumentWriter | None = None,
        sender_open_id: str | None = None,
    ):
        self._client = client
        self._chat_id = chat_id
        self._reply_to = reply_to_message_id
        self._card_message_id: str | None = None
        self._task_text: str = ""
        self._start_time: float = 0.0
        self._step_count: int = 0

        # Feishu document (full trajectory)
        self._doc_writer = document_writer
        self._sender_open_id = sender_open_id
        self._document_id: str | None = None
        self._document_url: str | None = None

        # TODO progress checklist
        self._todo_items: list[dict] = []  # [{"label": "...", "done": False}, ...]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def card_message_id(self) -> str | None:
        """The current card's message_id (available after finalize for external storage)."""
        return self._card_message_id

    def set_todo_items(self, items: list[str]) -> None:
        """Set TODO checklist items (for progress tracking in scenarios like agent_builder)."""
        self._todo_items = [{"label": item, "done": False} for item in items]

    def has_incomplete_todos(self) -> bool:
        """Check if there are incomplete TODO items."""
        if not self._todo_items:
            return False
        return any(not item["done"] for item in self._todo_items)

    def get_incomplete_todo_labels(self) -> list[str]:
        """Get the list of incomplete TODO labels."""
        return [item["label"] for item in self._todo_items if not item["done"]]

    def send_initial_card(self, task_text: str) -> bool:
        """Send the initial 'processing' card and capture the message_id for subsequent PATCHes."""
        from .messaging.sender import send_card_message

        self._task_text = task_text[:200]
        self._start_time = time.time()

        # Create Feishu document (full trajectory)
        self._create_trajectory_document(task_text)

        content = self._build_progress_content(0, 0, running=True)

        message_id = send_card_message(
            self._client,
            self._chat_id,
            title="🤖 Agent 执行中...",
            content=content,
            reply_to_message_id=self._reply_to,
            header_template="wathet",
        )
        if message_id:
            self._card_message_id = message_id
            return True
        return False

    def on_step(self, step_record: Any, step_number: int, max_steps: int) -> None:
        """Per-step callback: update card progress and write full content to the document."""
        if self._card_message_id is None:
            return

        self._step_count = step_number

        # Check TODO completion markers (via the think tool's PROGRESS marker)
        self._check_todo_progress(step_record)

        # Card: update progress only
        content = self._build_progress_content(step_number, max_steps, running=True)
        self._patch(
            title=f"🤖 Agent 执行中... (Step {step_number}/{max_steps})",
            content=content,
            template="wathet",
        )

        # Document: full content (no truncation)
        if self._doc_writer and self._document_id:
            try:
                self._append_step_to_document(step_record, step_number, max_steps)
            except Exception:
                logger.exception("Failed to append step %d to document", step_number)

    def finalize(
        self, status: str, final_answer: str = "", actions: list[dict] | None = None
    ) -> None:
        """Final card update (task completed/failed).

        Args:
            status: Completion status ("completed", "failed", etc.).
            final_answer: The final answer text.
            actions: Optional button list, same format as build_card_with_actions.
        """
        if self._card_message_id is None:
            return

        elapsed = time.time() - self._start_time

        content = self._build_progress_content(
            self._step_count, self._step_count, running=False
        )
        content += (
            f"\n\n---\n"
            f"**状态:** {status} | "
            f"**耗时:** {elapsed:.1f}s | "
            f"**步数:** {self._step_count}"
        )

        _CARD_ANSWER_PREVIEW = 800
        send_full_answer = False

        if final_answer:
            if len(final_answer) > _CARD_ANSWER_PREVIEW:
                # Only show a preview in the card; send full content separately
                preview = self._sanitize_for_card(
                    final_answer[:_CARD_ANSWER_PREVIEW]
                )
                content += f"\n\n**最终回答:**\n{preview}\n\n..."
                if self._document_url:
                    content += (
                        f"\n> 回答较长，"
                        f"[点击查看完整回答]({self._document_url})"
                    )
                send_full_answer = True
            else:
                display_answer = self._sanitize_for_card(final_answer)
                content += f"\n\n**最终回答:**\n{display_answer}"

        if status == "completed":
            template, title = "green", "✅ 任务完成"
        else:
            template, title = "red", f"❌ 任务{status}"

        if actions:
            self._patch_with_actions(
                title=title, content=content, template=template, actions=actions
            )
        else:
            self._patch(title=title, content=content, template=template)

        # Long answers: send an additional standalone full card message
        if send_full_answer and final_answer:
            self._send_full_answer(final_answer, template)

        # Document: append summary
        self._finalize_document(status, elapsed)

    def finalize_as_question(
        self, question_text: str, actions: list[dict] | None = None
    ) -> None:
        """Update the card to a 'waiting for user answer' state."""
        if self._card_message_id is None:
            return

        content = f"**任务:** {self._task_text}\n\n"
        if self._document_url:
            content += f"[📄 查看完整轨迹]({self._document_url})\n\n"
        content += f"---\n\n{question_text}"

        if actions:
            self._patch_with_actions(
                title="🤔 需要补充信息",
                content=content,
                template="orange",
                actions=actions,
            )
        else:
            self._patch(
                title="🤔 需要补充信息",
                content=content,
                template="orange",
            )

    # ------------------------------------------------------------------
    # Internal — Card
    # ------------------------------------------------------------------

    def _send_full_answer(self, answer: str, template: str) -> None:
        """Send an independent full-answer card (when the answer exceeds preview length)."""
        from .messaging.sender import send_card_message

        full_content = self._sanitize_for_card(answer[:3000])
        if len(answer) > 3000 and self._document_url:
            full_content += (
                f"\n\n...\n> 内容仍有省略，"
                f"[点击查看完整回答]({self._document_url})"
            )
        try:
            send_card_message(
                self._client,
                self._chat_id,
                title="📝 完整回答",
                content=full_content,
                reply_to_message_id=self._reply_to,
                header_template=template,
            )
        except Exception:
            logger.exception("Failed to send full answer card")

    @staticmethod
    def _sanitize_for_card(text: str) -> str:
        """Sanitize Markdown formatting to avoid conflicts with Feishu card structure.

        - Remove Markdown heading markers (## -> plain text)
        - Remove horizontal rules (---)
        - Remove tables (| col | col |)
        """
        lines = text.splitlines()
        cleaned: list[str] = []
        for line in lines:
            stripped = line.strip()
            # Remove horizontal rules
            if re.fullmatch(r'-{3,}|_{3,}|\*{3,}', stripped):
                continue
            # Remove heading markers, keep the text
            if stripped.startswith('#'):
                line = re.sub(r'^#+\s*', '', stripped)
            # Remove table separator rows (|---|---|)
            if re.fullmatch(r'\|[\s\-:|]+\|', stripped):
                continue
            cleaned.append(line)
        return '\n'.join(cleaned)

    def _build_progress_content(
        self, current_step: int, max_steps: int, running: bool
    ) -> str:
        """Build card content: task info + TODO checklist + progress + document link."""
        parts = [f"**任务:** {self._task_text}"]

        if self._document_url:
            parts.append(f"[📄 查看完整轨迹]({self._document_url})")

        # TODO checklist
        todo_content = self._build_todo_content()
        if todo_content:
            parts.append("---")
            parts.append(todo_content)

        parts.append("---")

        if running:
            elapsed = time.time() - self._start_time
            if current_step > 0:
                parts.append(f"> 正在执行 Step {current_step}/{max_steps}... ({elapsed:.0f}s)")
            else:
                parts.append("> 正在处理...")

        return "\n\n".join(parts)

    def _patch(self, title: str, content: str, template: str) -> None:
        """Execute a PATCH call to update the card."""
        from .messaging.sender import patch_card_message

        try:
            patch_card_message(
                self._client,
                self._card_message_id,
                title=title,
                content=content,
                header_template=template,
            )
        except Exception:
            logger.exception("Failed to patch card %s", self._card_message_id)

    def _patch_with_actions(
        self, title: str, content: str, template: str, actions: list[dict]
    ) -> None:
        """Execute a PATCH call with action buttons."""
        from .messaging.sender import build_card_with_actions, patch_card_message

        try:
            card_json = build_card_with_actions(
                title=title,
                content=content,
                actions=actions,
                header_template=template,
            )
            patch_card_message(
                self._client,
                self._card_message_id,
                card_json=card_json,
            )
        except Exception:
            logger.exception("Failed to patch card with actions %s", self._card_message_id)

    # ------------------------------------------------------------------
    # Internal — TODO Progress
    # ------------------------------------------------------------------

    def _build_todo_content(self) -> str:
        """Build markdown content for the TODO checklist."""
        if not self._todo_items:
            return ""
        lines = ["**构建进度:**"]
        for item in self._todo_items:
            check = "✅" if item["done"] else "⬜"
            lines.append(f"{check} {item['label']}")
        done_count = sum(1 for i in self._todo_items if i["done"])
        lines.append(f"\n> {done_count}/{len(self._todo_items)} 完成")
        return "\n".join(lines)

    def _check_todo_progress(self, step_record: Any) -> None:
        """Detect whether the builder reported PROGRESS markers via the think tool."""
        if not self._todo_items:
            return

        assistant_msg = getattr(step_record, "assistant_message", None)
        if not assistant_msg:
            return

        tool_calls = getattr(assistant_msg, "tool_calls", None) or []
        for tc in tool_calls:
            func = getattr(tc, "function", None)
            if func and getattr(func, "name", "") == "think":
                raw_args = getattr(func, "arguments", "")
                try:
                    args_obj = json.loads(raw_args)
                    thought = (
                        args_obj.get("thought", "")
                        or args_obj.get("content", "")
                        or ""
                    )
                except (json.JSONDecodeError, TypeError):
                    thought = str(raw_args)

                if "PROGRESS:" in thought and "[x]" in thought:
                    progress_text = thought.split("PROGRESS:", 1)[1].strip()
                    label = progress_text.replace("[x]", "").strip()
                    self._fuzzy_mark_done(label)

    def _fuzzy_mark_done(self, completed_label: str) -> None:
        """Fuzzy match and mark a TODO item as done."""
        completed_lower = completed_label.lower()
        for item in self._todo_items:
            if not item["done"]:
                item_lower = item["label"].lower()
                if (
                    completed_lower in item_lower
                    or item_lower in completed_lower
                ):
                    item["done"] = True
                    break

    # ------------------------------------------------------------------
    # Internal — Document
    # ------------------------------------------------------------------

    def _create_trajectory_document(self, task_text: str) -> None:
        """Create a Feishu document for storing the full trajectory. Silently degrades on failure."""
        if not self._doc_writer:
            return

        try:
            doc_id = self._doc_writer.create_document(
                title=f"Agent Trajectory: {task_text[:100]}"
            )
            if not doc_id:
                return

            self._document_id = doc_id
            self._doc_writer.set_public_readable(doc_id)
            self._document_url = self._doc_writer.get_document_url(doc_id)

            # Transfer document ownership to the message sender
            if self._sender_open_id:
                self._doc_writer.transfer_ownership(doc_id, self._sender_open_id)

            # Write document title and task description
            self._doc_writer.append_heading(doc_id, f"Task: {task_text[:500]}", level=1)
            self._doc_writer.append_text(
                doc_id,
                f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            )
            self._doc_writer.append_divider(doc_id)
        except Exception:
            logger.exception("Failed to create trajectory document")

    def _append_step_to_document(
        self, step_record: Any, step_num: int, max_steps: int
    ) -> None:
        """Append full step content to the Feishu document (no truncation)."""
        from .messaging.document import (
            _build_code_block,
            _build_divider_block,
            _build_heading_block,
            _build_text_block,
        )

        blocks = []

        # Step heading
        blocks.append(_build_heading_block(f"Step {step_num}/{max_steps}", level=3))

        assistant_msg = getattr(step_record, "assistant_message", None)
        if assistant_msg is None:
            blocks.append(_build_text_block("(no assistant message)"))
            blocks.append(_build_divider_block())
            self._doc_writer.append_blocks(self._document_id, blocks)
            return

        # Thinking / text content (full)
        content = getattr(assistant_msg, "content", "") or ""
        tool_calls = getattr(assistant_msg, "tool_calls", None) or []
        if content.strip():
            # When tool_calls exist, content is thinking; otherwise it's the final text answer
            if tool_calls:
                blocks.append(_build_text_block("Thinking:", bold=True))
            else:
                blocks.append(_build_text_block("Response:", bold=True))
            blocks.append(_build_text_block(content))

        # Tool calls (full arguments)
        for tc in tool_calls:
            func = getattr(tc, "function", None)
            if func is None:
                continue
            name = getattr(func, "name", "?")
            raw_args = getattr(func, "arguments", "")
            try:
                args_obj = json.loads(raw_args)
                args_str = json.dumps(args_obj, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                args_str = str(raw_args)
            blocks.append(_build_text_block(f"Tool Call: {name}", bold=True))
            blocks.append(_build_code_block(args_str, "json"))

        # Tool responses (full content)
        tool_responses = getattr(step_record, "tool_responses", None) or []
        for tr in tool_responses:
            tr_name = getattr(tr, "name", "?")
            tr_content = getattr(tr, "content", "") or ""
            blocks.append(_build_text_block(f"Result ({tr_name}):", bold=True))
            blocks.append(_build_code_block(tr_content))

        blocks.append(_build_divider_block())

        # Batch append (single API call)
        self._doc_writer.append_blocks(self._document_id, blocks)

    def _finalize_document(self, status: str, elapsed: float) -> None:
        """Append a summary to the document."""
        if not self._doc_writer or not self._document_id:
            return

        try:
            self._doc_writer.append_divider(self._document_id)
            self._doc_writer.append_heading(self._document_id, "Summary", level=2)
            summary = (
                f"Status: {status}\n"
                f"Duration: {elapsed:.1f}s\n"
                f"Steps: {self._step_count}"
            )
            self._doc_writer.append_text(self._document_id, summary)
        except Exception:
            logger.exception("Failed to finalize trajectory document")
