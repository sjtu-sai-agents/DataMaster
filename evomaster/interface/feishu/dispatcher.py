"""Task dispatcher

Dispatch Feishu messages to a thread pool, using ChatSessionManager for multi-turn conversation context persistence.
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_playgrounds_imported = False


def _ensure_playgrounds_imported(project_root: Path) -> None:
    """Ensure all playground modules are imported (triggering @register_playground decorators).

    Reuses the logic from run.py:auto_import_playgrounds().
    """
    global _playgrounds_imported
    if _playgrounds_imported:
        return

    # Ensure project_root is in sys.path
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    playground_dir = project_root / "playground"
    if not playground_dir.exists():
        logger.warning("Playground directory not found: %s", playground_dir)
        _playgrounds_imported = True
        return

    imported_count = 0

    # Collect agent directories to scan: top-level + _generated/ subdirectories
    agent_dirs: list[tuple[Path, str]] = []  # (dir_path, module_prefix)
    for child in playground_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name == "_generated":
            for gen_dir in child.iterdir():
                if gen_dir.is_dir() and not gen_dir.name.startswith("_"):
                    agent_dirs.append((gen_dir, f"playground._generated.{gen_dir.name}"))
        elif not child.name.startswith("_"):
            agent_dirs.append((child, f"playground.{child.name}"))

    for agent_dir, module_prefix in agent_dirs:
        module_name = f"{module_prefix}.core.playground"
        try:
            importlib.import_module(module_name)
            logger.info("Imported playground: %s", module_name)
            imported_count += 1
        except ImportError as e:
            logger.warning("Failed to import %s: %s", module_name, e)
        except Exception as e:
            logger.warning("Error importing %s: %s", module_name, e)

    logger.info("Auto-imported %d playground modules", imported_count)
    _playgrounds_imported = True


def _extract_final_answer(result: dict[str, Any]) -> str:
    """Extract the final answer from an execution result."""
    from evomaster.core import extract_agent_response

    trajectory = result.get("trajectory")
    if not trajectory:
        error = result.get("error", "")
        if error:
            return f"任务执行失败: {error}"
        return f"任务完成，状态: {result.get('status', 'unknown')}"

    # Check if step limit was reached
    traj_result = getattr(trajectory, "result", None)
    if isinstance(traj_result, dict) and traj_result.get("reason") == "max_turns_exceeded":
        return "超过步数限制"

    answer = extract_agent_response(trajectory)
    if answer:
        return answer

    status = result.get("status", "unknown")
    steps = result.get("steps", 0)
    return f"任务完成（状态: {status}，步骤: {steps}），但未提取到文本回答。"


# Subtask agents requiring multi-turn sessions (use independent session keys)
_SESSION_SUBTASK_AGENTS = {"agent_builder"}

# Subtask agents requiring confirmation buttons after completion (show "Confirm Generation" button after Phase 1)
_CONFIRM_SUBTASK_AGENTS = {"agent_builder"}


class TaskDispatcher:
    """Task dispatcher: implements multi-turn conversation context persistence via session management."""

    def __init__(
        self,
        project_root: Path,
        default_agent: str = "chat_agent",
        default_config_path: Optional[str] = None,
        max_workers: int = 4,
        task_timeout: int = 600,
        max_sessions: int = 100,
        on_result: Optional[Callable[[str, str, str], None]] = None,
        step_reporter_factory: Optional[Callable[[str, str | None], Any]] = None,
        feishu_app_id: Optional[str] = None,
        feishu_app_secret: Optional[str] = None,
        feishu_domain: str = "https://open.feishu.cn",
        feishu_doc_folder_token: Optional[str] = None,
    ):
        """
        Args:
            project_root: Project root directory.
            default_agent: Default agent name.
            default_config_path: Default config file path (relative to project_root).
            max_workers: Maximum number of concurrent threads.
            task_timeout: Single-turn task timeout in seconds.
            max_sessions: Maximum number of concurrent sessions.
            on_result: Result callback (chat_id, message_id, result_text) -> None.
            step_reporter_factory: Factory function for creating FeishuStepReporter instances.
            feishu_app_id: Feishu App ID (used for injecting Feishu-specific tools).
            feishu_app_secret: Feishu App Secret.
            feishu_domain: Feishu API domain.
            feishu_doc_folder_token: Feishu folder token (used by document writing tools).
        """
        from .session_manager import ChatSessionManager

        self._project_root = project_root
        self._default_agent = default_agent
        self._default_config_path = default_config_path
        self._task_timeout = task_timeout
        self._on_result = on_result
        self._step_reporter_factory = step_reporter_factory
        self._server_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="feishu-task",
        )
        self._active_tasks: dict[str, Any] = {}
        self._session_manager = ChatSessionManager(max_sessions=max_sessions)

        # Store Feishu credentials (used for dynamically creating tools)
        self._feishu_app_id = feishu_app_id
        self._feishu_app_secret = feishu_app_secret
        self._feishu_domain = feishu_domain
        self._feishu_doc_folder_token = feishu_doc_folder_token

        # Feishu Client (used for card patching and other operations)
        self._feishu_client = None
        if feishu_app_id and feishu_app_secret:
            from .messaging.client import create_feishu_client
            self._feishu_client = create_feishu_client(
                app_id=feishu_app_id,
                app_secret=feishu_app_secret,
                domain=feishu_domain,
            )

        # Feishu-specific tools (shared across all agents)
        self._feishu_tools: list = []
        if feishu_app_id and feishu_app_secret:
            from .tools.doc_reader import FeishuDocReadTool

            self._feishu_tools.append(
                FeishuDocReadTool(
                    app_id=feishu_app_id,
                    app_secret=feishu_app_secret,
                    domain=feishu_domain,
                )
            )

        # Ensure _generated directories exist (agents generated by agent_builder are placed here)
        (project_root / "configs" / "_generated").mkdir(parents=True, exist_ok=True)
        (project_root / "playground" / "_generated").mkdir(parents=True, exist_ok=True)

        # Preload playgrounds
        _ensure_playgrounds_imported(project_root)

    def dispatch(
        self,
        chat_id: str,
        message_id: str,
        task_text: str,
        agent_name: Optional[str] = None,
        sender_open_id: Optional[str] = None,
    ) -> None:
        """Submit a task to the thread pool.

        Special commands:
        - /new: Clear the current session context.
        - /help: Show usage help.
        """
        stripped = task_text.strip()

        # /new command: clear session
        if stripped == "/new":
            self._session_manager.remove(chat_id)
            # Also clear all session-level subtask sessions for this chat
            for agent_name in _SESSION_SUBTASK_AGENTS:
                self._session_manager.remove(f"{chat_id}:{agent_name}")
            self._send_welcome_card(chat_id, message_id)
            return

        # /help command: show usage help
        if stripped == "/help":
            self._send_help_card(chat_id, message_id)
            return

        agent = agent_name or self._default_agent
        future = self._executor.submit(
            self._run_task_with_session,
            chat_id,
            message_id,
            task_text,
            agent,
            sender_open_id,
        )
        self._active_tasks[message_id] = future
        future.add_done_callback(lambda f: self._on_task_done(f, chat_id, message_id))

        # Timeout watchdog thread
        def _timeout_guard():
            """Wait for the task future and handle timeout or errors."""
            try:
                future.result(timeout=self._task_timeout)
            except TimeoutError:
                logger.warning(
                    "Task timed out: message_id=%s, timeout=%ds",
                    message_id,
                    self._task_timeout,
                )
                future.cancel()
            except Exception:
                pass

        threading.Thread(
            target=_timeout_guard,
            daemon=True,
            name=f"timeout-{message_id[:8]}",
        ).start()

    def _create_playground(self, agent_name: str, sender_open_id: str | None = None):
        """Create a playground instance (without calling setup)."""
        from evomaster.core import get_playground_class

        if agent_name == self._default_agent and self._default_config_path:
            config_path = self._project_root / self._default_config_path
        else:
            config_path = self._project_root / "configs" / agent_name / "config.yaml"
            # Fallback: check _generated directory (agents generated by agent_builder are placed here)
            if not config_path.exists():
                config_path = self._project_root / "configs" / "_generated" / agent_name / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # Dynamically import _generated playgrounds (may be generated after bot startup)
        self._try_import_generated_playground(agent_name)

        playground = get_playground_class(agent_name, config_path=config_path)

        # Create hierarchical run directory: runs/feishu_{server_start}/{user_id}/{agent}_{timestamp}/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        feishu_base = self._project_root / "runs" / f"feishu_{self._server_start_time}"
        user_dir = sender_open_id or "unknown"
        run_dir = feishu_base / user_dir / f"{agent_name}_{timestamp}"
        task_id = f"feishu_{agent_name}"
        playground.set_run_dir(run_dir, task_id=task_id)

        return playground

    def _try_import_generated_playground(self, agent_name: str) -> None:
        """Try to dynamically import a _generated playground module.

        Agents generated by agent_builder may be created after the bot starts,
        so _ensure_playgrounds_imported at startup will not scan for them.
        """
        from evomaster.core.registry import _PLAYGROUND_REGISTRY

        if agent_name in _PLAYGROUND_REGISTRY:
            return  # Already registered, no need to import again

        module_name = f"playground._generated.{agent_name}.core.playground"
        try:
            importlib.import_module(module_name)
            logger.info("Dynamically imported generated playground: %s", module_name)
        except ImportError:
            pass  # No custom playground, will fallback to BasePlayground
        except Exception:
            logger.warning("Error importing generated playground: %s", module_name, exc_info=True)

    def _run_task_with_session(
        self,
        chat_id: str,
        message_id: str,
        task_text: str,
        agent_name: str,
        sender_open_id: Optional[str] = None,
    ) -> str:
        """Execute a task in a background thread, reusing session context.

        If agent_name differs from the default agent, use subtask mode:
        run the specified agent independently and inject results into the chat_agent context.
        """
        from evomaster.utils.types import TaskInstance

        # Always use the default agent to create/get the session
        session = self._session_manager.get_or_create(
            chat_id,
            playground_factory=lambda: self._create_playground(self._default_agent, sender_open_id),
        )

        # Serialize processing within the same chat
        with session.lock:
            session.last_activity = time.monotonic()
            session.message_count += 1
            # Register current thread to playground (for log filtering)            session.playground.register_thread()

            # Create real-time progress reporter
            reporter = None
            on_step = None
            if self._step_reporter_factory:
                try:
                    reporter = self._step_reporter_factory(
                        chat_id, message_id, sender_open_id
                    )
                    reporter.send_initial_card(task_text)
                    on_step = reporter.on_step
                except Exception:
                    logger.exception("Failed to create step reporter")

            try:
                # Subtask mode: /agent specified a non-default agent
                if agent_name != self._default_agent:
                    # Session-level subtask: supports multi-turn conversation (e.g. agent_builder)
                    if agent_name in _SESSION_SUBTASK_AGENTS:
                        answer, sub_trajectory = self._run_session_subtask(
                            chat_id, agent_name, task_text, on_step, sender_open_id
                        )
                    else:
                        answer = self._run_subtask(agent_name, task_text, on_step, chat_id=chat_id, sender_open_id=sender_open_id)
                        sub_trajectory = None

                    # Check waiting_for_input (agent is asking the user a question)
                    # 检查 waiting_for_input（agent 在向用户提问）
                    if sub_trajectory and sub_trajectory.status == "waiting_for_input":
                        if reporter:
                            try:
                                sub_session_key = f"{chat_id}:{agent_name}"
                                sub_session = self._session_manager.get(sub_session_key)
                                self._finalize_subtask_with_question(
                                    reporter, sub_trajectory, sub_session_key,
                                    agent_name, sub_session,
                                )
                                return None
                            except Exception:
                                logger.exception("Failed to finalize question card")
                        return answer

                    # Inject result into chat_agent's dialog as context
                    if session.initialized and session.agent:
                        summary = (
                            f"[子任务结果 - {agent_name}]\n"
                            f"用户请求: {task_text}\n"
                            f"结果: {answer}"
                        )
                        session.agent.add_user_message(summary)

                    if reporter:
                        try:
                            # Confirmation-type agents: add confirm/cancel buttons on finalize
                            if agent_name in _CONFIRM_SUBTASK_AGENTS:
                                session_key = f"{chat_id}:{agent_name}"
                                # Truncate answer for button value; used to preserve original card content on callback
                                _answer_for_button = answer[:2000] if answer else ""
                                actions = [
                                    {
                                        "text": "✅ 确认生成",
                                        "type": "primary",
                                        "value": {
                                            "action": "confirm_agent_build",
                                            "session_key": session_key,
                                            "agent_name": agent_name,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                    {
                                        "text": "❌ 取消",
                                        "type": "danger",
                                        "value": {
                                            "action": "cancel_agent_build",
                                            "session_key": session_key,
                                            "agent_name": agent_name,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                ]
                                reporter.finalize("completed", answer, actions=actions)
                            else:
                                reporter.finalize("completed", answer)
                            return None  # Card already contains the answer
                        except Exception:
                            logger.exception("Failed to finalize step reporter")

                    return answer

                # === Active subtask routing ===
                # If there is an active subtask session (e.g. agent_builder planner),
                # route subsequent messages directly to it for multi-turn plan modification
                active_subtask = self._find_active_subtask(chat_id)
                if active_subtask:
                    # Multi-turn modification: patch old card to remove buttons
                    sub_session_key = f"{chat_id}:{active_subtask}"
                    sub_session = self._session_manager.get(sub_session_key)
                    if sub_session and sub_session.last_card_message_id and self._feishu_client:
                        self._patch_phase1_card(
                            sub_session.last_card_message_id,
                            "📝 方案修改中",
                            "用户正在修改方案，请以最新卡片为准。",
                            "grey",
                        )
                        sub_session.last_card_message_id = None

                    answer, sub_trajectory = self._run_session_subtask(
                        chat_id, active_subtask, task_text, on_step, sender_open_id
                    )

                    # Check waiting_for_input (agent is asking the user a question)
                    if sub_trajectory and sub_trajectory.status == "waiting_for_input":
                        if reporter:
                            try:
                                self._finalize_subtask_with_question(
                                    reporter, sub_trajectory, sub_session_key,
                                    active_subtask, sub_session,
                                )
                                return None
                            except Exception:
                                logger.exception("Failed to finalize question card")
                        return answer

                    if session.initialized and session.agent:
                        summary = (
                            f"[子任务结果 - {active_subtask}]\n"
                            f"用户请求: {task_text}\n"
                            f"结果: {answer}"
                        )
                        session.agent.add_user_message(summary)
                    if reporter:
                        try:
                            if active_subtask in _CONFIRM_SUBTASK_AGENTS:
                                session_key = f"{chat_id}:{active_subtask}"
                                _answer_for_button = answer[:2000] if answer else ""
                                actions = [
                                    {
                                        "text": "✅ 确认生成",
                                        "type": "primary",
                                        "value": {
                                            "action": "confirm_agent_build",
                                            "session_key": session_key,
                                            "agent_name": active_subtask,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                    {
                                        "text": "❌ 取消",
                                        "type": "danger",
                                        "value": {
                                            "action": "cancel_agent_build",
                                            "session_key": session_key,
                                            "agent_name": active_subtask,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                ]
                                reporter.finalize("completed", answer, actions=actions)
                                # Store current card ID, can be patched to remove buttons on next multi-turn
                                if sub_session:
                                    sub_session.last_card_message_id = reporter.card_message_id
                            else:
                                reporter.finalize("completed", answer)
                            return None
                        except Exception:
                            logger.exception("Failed to finalize step reporter")
                    return answer

                # Normal chat_agent flow
                # Get memory system (if playground is initialized)
                memory_manager = getattr(session.playground, "_memory_manager", None)
                memory_config = getattr(session.playground, "_memory_config", {})
                user_id = sender_open_id or "unknown"

                if not session.initialized:
                    # First message: full setup + agent.run()
                    logger.info(
                        "First message in session chat_id=%s, running setup",
                        chat_id,
                    )
                    session.playground.setup()
                    session.playground._setup_trajectory_file()
                    session.agent = session.playground.agent

                    # Re-fetch (only available after setup for _memory_manager)
                    memory_manager = getattr(session.playground, "_memory_manager", None)
                    memory_config = getattr(session.playground, "_memory_config", {})

                    # Inject Feishu-specific tools
                    self._inject_feishu_tools(session.playground)
                    self._inject_send_file_tool(session.playground, chat_id)
                    self._inject_ask_user_tool(session.agent)
                    self._inject_memory_tools(session.agent, memory_manager, user_id)

                    # Set up memory extraction hook before compaction
                    if memory_manager and memory_config.get("auto_capture", True):
                        _mm = memory_manager
                        _uid = user_id
                        def _on_compaction(old_messages, mm=_mm, uid=_uid):
                            """Callback invoked on context compaction to capture memories from discarded messages."""
                            from evomaster.utils.types import UserMessage
                            for msg in old_messages:
                                if isinstance(msg, UserMessage):
                                    text = msg.content if isinstance(msg.content, str) else ""
                                    if text:
                                        mm.extract_from_message(uid, text)
                        session.agent.context_manager.on_before_compaction = _on_compaction

                    task = TaskInstance(
                        task_id=f"feishu_{message_id}",
                        task_type="chat",
                        description=task_text,
                    )

                    # Automatically recall related memories (inject into system prompt)
                    self._memory_auto_recall(
                        session.agent, memory_manager, memory_config, user_id, task_text,
                    )

                    trajectory = session.agent.run(task, on_step=on_step)
                    session.initialized = True
                    self._memory_auto_capture(memory_manager, memory_config, user_id, task_text)
                else:
                    # Subsequent messages: continue_run()
                    logger.info(
                        "Continuing session chat_id=%s (message #%d)",
                        chat_id,
                        session.message_count,
                    )

                    # Automatically recall related memories (inject into system prompt)
                    self._memory_auto_recall(
                        session.agent, memory_manager, memory_config, user_id, task_text,
                    )

                    trajectory = session.agent.continue_run(
                        task_text, on_step=on_step
                    )

                    # Automatically extract memories from user messages
                    self._memory_auto_capture(memory_manager, memory_config, user_id, task_text)

                # === ask_user detection ===
                # chat_agent called ask_user, present questions one by one via cards
                if trajectory and trajectory.status == "waiting_for_input":
                    if reporter:
                        try:
                            questions = (trajectory.result or {}).get("questions", [])
                            if questions:
                                # Only present the first question
                                first = [questions[0]]
                                question_text = self._format_questions_for_card(first)
                                option_actions = self._build_question_actions(
                                    first, chat_id, "chat_agent",
                                    question_text=question_text,
                                )
                                reporter.finalize_as_question(question_text, actions=option_actions)
                                # Store remaining questions to session
                                session.pending_questions = questions[1:]
                                session.collected_answers = []
                            return None
                        except Exception:
                            logger.exception("Failed to finalize chat_agent question card")
                    return _extract_final_answer(
                        {"trajectory": trajectory, "status": trajectory.status}
                    )

                # === Delegation detection ===
                # chat_agent may have triggered delegation via the delegate_to_agent tool
                delegation = self._check_delegation(session)
                if delegation:
                    delegated_agent = delegation["agent_name"]
                    delegated_task = delegation["task"]
                    logger.info(
                        "Delegation detected: agent=%s, task=%s",
                        delegated_agent, delegated_task[:100],
                    )

                    # First finalize the chat_agent card (show delegation message)
                    chat_answer = _extract_final_answer(
                        {"trajectory": trajectory, "status": trajectory.status}
                    )
                    if reporter:
                        try:
                            reporter.finalize("completed", chat_answer)
                        except Exception:
                            logger.exception("Failed to finalize chat reporter")

                    # Create a reporter for the subtask
                    subtask_reporter = None
                    subtask_on_step = None
                    if self._step_reporter_factory:
                        try:
                            subtask_reporter = self._step_reporter_factory(
                                chat_id, message_id, sender_open_id
                            )
                            subtask_reporter.send_initial_card(
                                f"[{delegated_agent}] {delegated_task[:200]}"
                            )
                            subtask_on_step = subtask_reporter.on_step
                        except Exception:
                            logger.exception("Failed to create subtask reporter")

                    answer, sub_trajectory = self._run_session_subtask(
                        chat_id, delegated_agent, delegated_task,
                        subtask_on_step, sender_open_id,
                    )

                    # Check waiting_for_input (agent is asking the user a question)
                    if sub_trajectory and sub_trajectory.status == "waiting_for_input":
                        if subtask_reporter:
                            try:
                                sub_session_key = f"{chat_id}:{delegated_agent}"
                                sub_session = self._session_manager.get(sub_session_key)
                                self._finalize_subtask_with_question(
                                    subtask_reporter, sub_trajectory, sub_session_key,
                                    delegated_agent, sub_session,
                                )
                                return None
                            except Exception:
                                logger.exception("Failed to finalize question card")
                        return None

                    if session.initialized and session.agent:
                        summary = (
                            f"[子任务结果 - {delegated_agent}]\n"
                            f"用户请求: {delegated_task}\n"
                            f"结果: {answer}"
                        )
                        session.agent.add_user_message(summary)
                    if subtask_reporter:
                        try:
                            if delegated_agent in _CONFIRM_SUBTASK_AGENTS:
                                session_key = f"{chat_id}:{delegated_agent}"
                                _answer_for_button = answer[:2000] if answer else ""
                                actions = [
                                    {
                                        "text": "✅ 确认生成",
                                        "type": "primary",
                                        "value": {
                                            "action": "confirm_agent_build",
                                            "session_key": session_key,
                                            "agent_name": delegated_agent,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                    {
                                        "text": "❌ 取消",
                                        "type": "danger",
                                        "value": {
                                            "action": "cancel_agent_build",
                                            "session_key": session_key,
                                            "agent_name": delegated_agent,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                ]
                                subtask_reporter.finalize(
                                    "completed", answer, actions=actions
                                )
                                # Store card ID, can be patched to remove old buttons on next multi-turn
                                sub_session = self._session_manager.get(session_key)
                                if sub_session:
                                    sub_session.last_card_message_id = subtask_reporter.card_message_id
                            else:
                                subtask_reporter.finalize("completed", answer)
                            return None
                        except Exception:
                            logger.exception("Failed to finalize subtask reporter")
                    return None

                # No delegation: return normally
                answer = _extract_final_answer(
                    {"trajectory": trajectory, "status": trajectory.status}
                )
                logger.info(
                    "Task completed in session chat_id=%s, status=%s",
                    chat_id,
                    trajectory.status,
                )

                if reporter:
                    try:
                        reporter.finalize("completed", answer)
                        return None  # Card already contains the answer, no extra message needed
                    except Exception:
                        logger.exception("Failed to finalize step reporter")

                return answer

            except Exception as e:
                logger.exception("Task failed in session chat_id=%s", chat_id)
                if reporter:
                    try:
                        reporter.finalize("failed")
                    except Exception:
                        logger.exception(
                            "Failed to finalize step reporter on error"
                        )
                return f"任务执行出错: {e}"

    def _run_subtask(
        self, agent_name: str, task_text: str, on_step: Optional[Callable] = None,
        chat_id: Optional[str] = None, sender_open_id: Optional[str] = None,
    ) -> str:
        """Run a subtask with the specified agent independently, without reusing session context."""
        from evomaster.utils.types import TaskInstance

        logger.info("Running subtask with agent=%s", agent_name)
        playground = self._create_playground(agent_name, sender_open_id)
        # Register the current thread with the playground (for log filtering)
        playground.register_thread()
        try:
            playground.setup()
            playground._setup_trajectory_file()
            self._inject_feishu_tools(playground)
            if chat_id:
                self._inject_send_file_tool(playground, chat_id)
            agent = playground.agent
            task = TaskInstance(
                task_id=f"subtask_{agent_name}",
                task_type="subtask",
                description=task_text,
            )
            trajectory = agent.run(task, on_step=on_step)
            return _extract_final_answer(
                {"trajectory": trajectory, "status": trajectory.status}
            )
        except Exception as e:
            logger.exception("Subtask failed: agent=%s", agent_name)
            return f"子任务执行出错: {e}"
        finally:
            try:
                playground.cleanup()
            except Exception:
                logger.exception("Subtask cleanup failed")

    def _inject_feishu_tools(self, playground) -> None:
        """Inject Feishu-specific tools into all agents in the playground."""
        if not self._feishu_tools:
            return
        for agent in playground.agents.values():
            for tool in self._feishu_tools:
                agent.tools.register(tool)

    def _inject_send_file_tool(self, playground, chat_id: str) -> None:
        """Inject the file/image sending tool into all agents in the playground."""
        if not self._feishu_client:
            return
        from .tools.send_file import SendFileTool

        tool = SendFileTool(client=self._feishu_client, chat_id=chat_id)
        for agent in playground.agents.values():
            agent.tools.register(tool)

    def _run_session_subtask(
        self,
        chat_id: str,
        agent_name: str,
        task_text: str,
        on_step: Optional[Callable] = None,
        sender_open_id: Optional[str] = None,
    ) -> tuple[str, Any]:
        """Run a session-level subtask: an independent agent session supporting multi-turn conversation.

        Uses {chat_id}:{agent_name} as the session key, supporting continue_run().

        Returns:
            (answer_text, trajectory) tuple. trajectory may be None on exception.
        """
        from evomaster.utils.types import TaskInstance

        session_key = f"{chat_id}:{agent_name}"
        session = self._session_manager.get_or_create(
            session_key,
            playground_factory=lambda: self._create_playground(agent_name, sender_open_id),
        )

        # Session-level subtask also processes serially
        with session.lock:
            session.last_activity = time.monotonic()
            session.message_count += 1
            # Register current thread to playground (for log filtering)            
            session.playground.register_thread()

            try:
                if not session.initialized:
                    logger.info(
                        "First message in session subtask key=%s, agent=%s",
                        session_key, agent_name,
                    )
                    session.playground.setup()
                    session.playground._setup_trajectory_file()
                    session.agent = session.playground.agent

                    self._inject_feishu_tools(session.playground)
                    self._inject_send_file_tool(session.playground, chat_id)
                    self._inject_doc_write_tool(session.playground, sender_open_id)
                    self._inject_ask_user_tool(session.agent)

                    task = TaskInstance(
                        task_id=f"session_subtask_{agent_name}",
                        task_type="session_subtask",
                        description=task_text,
                    )
                    trajectory = session.agent.run(task, on_step=on_step)
                    session.initialized = True
                else:
                    logger.info(
                        "Continuing session subtask key=%s (message #%d)",
                        session_key, session.message_count,
                    )
                    trajectory = session.agent.continue_run(
                        task_text, on_step=on_step
                    )

                answer = _extract_final_answer(
                    {"trajectory": trajectory, "status": trajectory.status}
                )
                return answer, trajectory

            except Exception as e:
                logger.exception(
                    "Session subtask failed: key=%s, agent=%s", session_key, agent_name
                )
                return f"会话子任务执行出错: {e}", None

    def _inject_doc_write_tool(self, playground, sender_open_id: str | None) -> None:
        """Inject the Feishu document writing tool into all agents in the playground."""
        if not self._feishu_app_id or not self._feishu_app_secret:
            return

        from .messaging.client import create_feishu_client
        from .messaging.document import FeishuDocumentWriter
        from playground.agent_builder.tools.feishu_doc_write import FeishuDocWriteTool

        client = create_feishu_client(
            app_id=self._feishu_app_id,
            app_secret=self._feishu_app_secret,
            domain=self._feishu_domain,
        )
        writer = FeishuDocumentWriter(
            client,
            folder_token=self._feishu_doc_folder_token,
            domain=self._feishu_domain,
        )
        tool = FeishuDocWriteTool(
            document_writer=writer,
            sender_open_id=sender_open_id,
        )

        for agent in playground.agents.values():
            agent.tools.register(tool)

    @staticmethod
    def _inject_ask_user_tool(agent) -> None:
        """Inject the ask_user tool (only used in interactive contexts)."""
        from evomaster.interface.tools.ask_user import AskUserTool
        agent.tools.register(AskUserTool())

    @staticmethod
    def _inject_memory_tools(agent, memory_manager, user_id: str) -> None:
        """Inject memory tools into the agent (memory_search / memory_save / memory_forget)."""
        if memory_manager is None:
            return
        from playground.chat_agent.tools.memory_tools import (
            MemorySearchTool, MemorySaveTool, MemoryForgetTool,
        )
        for tool_cls in (MemorySearchTool, MemorySaveTool, MemoryForgetTool):
            tool = tool_cls(memory_manager=memory_manager, user_id=user_id)
            agent.tools.register(tool)

    @staticmethod
    def _memory_auto_recall(agent, memory_manager, memory_config, user_id: str, query: str) -> None:
        """Automatically recall related memories and inject them at the end of the system prompt."""
        if memory_manager is None:
            return
        if not memory_config.get("auto_recall", True):
            return
        limit = memory_config.get("recall_limit", 5)
        memory_context = memory_manager.recall_for_context(
            user_id=user_id, query=query, limit=limit,
        )
        if not memory_context:
            return
        # Append memories to the end of the system prompt
        dialog = agent.current_dialog
        if dialog and dialog.messages and dialog.messages[0].role.value == "system":
            dialog.messages[0].content = (
                dialog.messages[0].content + "\n\n" + memory_context
            )

    @staticmethod
    def _memory_auto_capture(memory_manager, memory_config, user_id: str, message: str) -> None:
        """Automatically extract memories from user messages."""
        if memory_manager is None:
            return
        if not memory_config.get("auto_capture", True):
            return
        try:
            memory_manager.extract_from_message(user_id, message)
        except Exception:
            logger.debug("Memory auto-capture failed", exc_info=True)

    @staticmethod
    def _format_questions_for_card(questions: list[dict]) -> str:
        """Format questions as card markdown (with header grouping support)."""
        parts = []
        for q in questions:
            header = q.get("header", "")
            title = f"**{header}: {q.get('question', '')}**" if header else f"**{q.get('question', '')}**"
            parts.append(title)
            for opt in q.get("options", []):
                desc = f" — {opt['description']}" if opt.get("description") else ""
                parts.append(f"  - {opt['label']}{desc}")
            parts.append("")  # Blank line to separate questions
        parts.append("> You can also reply with text to provide more details")
        return "\n".join(parts)

    @staticmethod
    def _build_question_actions(
        questions: list[dict], session_key: str, agent_name: str,
        question_text: str = "",
    ) -> list[dict]:
        """Build buttons for the first question's options only (maximum 4)."""
        if not questions or not questions[0].get("options"):
            return []
        actions = []
        for opt in questions[0]["options"][:4]:
            actions.append({
                "text": opt.get("label", ""),
                "type": "default",
                "value": {
                    "action": "answer_question",
                    "session_key": session_key,
                    "agent_name": agent_name,
                    "answer_text": opt.get("label", ""),
                    "original_question": question_text[:1500],
                },
            })
        return actions

    def _finalize_subtask_with_question(
        self, reporter, trajectory, sub_session_key: str, agent_name: str, sub_session
    ) -> None:
        """When a subtask returns waiting_for_input, present questions one by one via cards.

        Only render the first question (with full buttons); remaining questions are stored
        in session.pending_questions and presented one by one in _continue_session_subtask
        after the user answers.
        """
        questions = (getattr(trajectory, "result", None) or {}).get("questions", [])
        if not questions:
            return

        # Only present the first question
        first = [questions[0]]
        question_text = self._format_questions_for_card(first)
        option_actions = self._build_question_actions(
            first, sub_session_key, agent_name, question_text=question_text
        )
        reporter.finalize_as_question(question_text, actions=option_actions)
        if sub_session:
            sub_session.last_card_message_id = reporter.card_message_id
            # Store remaining questions, clear collected answers
            sub_session.pending_questions = questions[1:]
            sub_session.collected_answers = []

    def dispatch_card_action(
        self,
        chat_id: str,
        session_key: str,
        agent_name: str,
        task_text: str,
        sender_open_id: str | None = None,
        card_message_id: str | None = None,
        original_answer: str = "",
        action_type: str = "confirm",
    ) -> None:
        """Handle card button callbacks, triggering continue_run for session-level subtasks.

        Args:
            chat_id: Chat ID (used for sending results).
            session_key: Session key (format: {chat_id}:{agent_name}).
            agent_name: Agent name.
            task_text: Text to send to the agent (e.g. "Confirm").
            sender_open_id: Operator's open_id.
            card_message_id: Card message ID that triggered the button.
            original_answer: Phase 1's original answer content (preserved when updating card).
            action_type: Button type ("confirm" = Phase 2 generation, "answer_question" = answer question to continue Phase 1).
        """
        message_id = card_message_id or f"card_action_{session_key}"
        future = self._executor.submit(
            self._continue_session_subtask,
            chat_id,
            session_key,
            agent_name,
            task_text,
            sender_open_id,
            card_message_id,
            original_answer,
            action_type,
        )
        self._active_tasks[message_id] = future
        future.add_done_callback(
            lambda f: self._on_task_done(f, chat_id, message_id)
        )

    def _continue_session_subtask(
        self,
        chat_id: str,
        session_key: str,
        agent_name: str,
        task_text: str,
        sender_open_id: str | None = None,
        card_message_id: str | None = None,
        original_answer: str = "",
        action_type: str = "confirm",
    ) -> str | None:
        """Continue an existing session-level subtask (triggered by card button).

        Args:
            action_type: "confirm" = Phase 2 builder run, "answer_question" = continue planner.
        """
        session = self._session_manager.get(session_key)
        if session is None or not session.initialized:
            logger.warning(
                "No active session for card action: key=%s", session_key
            )
            return f"会话已过期或不存在，请重新发起 /agent {agent_name} 命令。"

        with session.lock:
            session.last_activity = time.monotonic()
            session.message_count += 1
            # Register current thread to playground (for log filtering)            session.playground.register_thread()

            # Create progress reporter
            reporter = None
            on_step = None
            if self._step_reporter_factory:
                try:
                    reporter = self._step_reporter_factory(
                        chat_id, card_message_id, sender_open_id
                    )
                    # agent_builder confirm: delay sending card, wait for TODO parsing then send at once
                    # answer_question: send immediately, so on_step can update in real-time
                    if action_type == "answer_question" or agent_name not in _CONFIRM_SUBTASK_AGENTS:
                        reporter.send_initial_card(f"[{agent_name}] {task_text}")
                    on_step = reporter.on_step
                except Exception:
                    logger.exception("Failed to create step reporter for card action")

            try:
                # === Sequential questioning: check if there are follow-up questions to present ===
                if action_type == "answer_question" and session.pending_questions:
                    session.collected_answers.append(task_text)
                    next_q = session.pending_questions.pop(0)
                    first = [next_q]
                    question_text = self._format_questions_for_card(first)
                    option_actions = self._build_question_actions(
                        first, session_key, agent_name,
                        question_text=question_text,
                    )
                    if reporter:
                        reporter.finalize_as_question(
                            question_text, actions=option_actions
                        )
                        session.last_card_message_id = reporter.card_message_id
                    return None  # Don't resume agent, wait for the next answer

                # === Sequential questioning: all questions answered, merge answers ===
                if action_type == "answer_question" and session.collected_answers:
                    session.collected_answers.append(task_text)
                    task_text = "\n".join(session.collected_answers)
                    session.collected_answers = []

                logger.info(
                    "Continuing session subtask via card action: key=%s (message #%d)",
                    session_key, session.message_count,
                )

                # agent_builder dual-agent mode: Phase 2 uses builder_agent (fresh run)
                # Only triggered on confirm, answer_question goes through planner continue_run
                if (
                    action_type == "confirm"
                    and agent_name == "agent_builder"
                    and hasattr(session.playground, "agents")
                    and hasattr(session.playground.agents, "builder_agent")
                ):
                    from evomaster.utils.types import TaskInstance

                    # Parse TODO checklist from planner output and set it on the reporter
                    todo_items = self._parse_plan_todos(original_answer)
                    if reporter:
                        if todo_items:
                            reporter.set_todo_items(todo_items)
                        reporter.send_initial_card(
                            f"[{agent_name}] 正在生成 Agent 文件..."
                        )
                        on_step = reporter.on_step

                    builder_agent = session.playground.agents.builder_agent
                    # Inject Feishu tools into builder agent (already injected during setup, but ensure availability)
                    if self._feishu_tools:
                        for tool in self._feishu_tools:
                            builder_agent.tools.register(tool)
                    # Build handoff task: pass the planner's plan summary to the builder
                    plan_task = TaskInstance(
                        task_id=f"builder_{agent_name}",
                        task_type="builder",
                        description=(
                            "请根据以下设计方案生成 Agent 文件。\n\n"
                            f"## 方案摘要\n{original_answer}\n\n"
                            "请使用 feishu_doc_read 工具读取飞书文档获取完整方案，然后生成所有文件。"
                        ),
                    )
                    trajectory = builder_agent.run(plan_task, on_step=on_step)

                    # After builder completes, check if all TODOs are done; if not, trigger another round
                    if reporter and reporter.has_incomplete_todos():
                        incomplete = reporter.get_incomplete_todo_labels()
                        reminder = (
                            "你还有以下 TODO 项未完成，请逐一完成并上报 PROGRESS 后再调用 finish：\n"
                            + "\n".join(f"- [ ] {label}" for label in incomplete)
                        )
                        logger.info(
                            "Builder has %d incomplete TODOs, triggering continue_run",
                            len(incomplete),
                        )
                        trajectory = builder_agent.continue_run(reminder, on_step=on_step)
                else:
                    trajectory = session.agent.continue_run(
                        task_text, on_step=on_step
                    )
                answer = _extract_final_answer(
                    {"trajectory": trajectory, "status": trajectory.status}
                )

                # === answer_question path: processing after planner continue_run ===
                if action_type == "answer_question":
                    # Check if the planner is asking again
                    if trajectory and trajectory.status == "waiting_for_input":
                        if reporter:
                            try:
                                self._finalize_subtask_with_question(
                                    reporter, trajectory, session_key,
                                    agent_name, session,
                                )
                                return None
                            except Exception:
                                logger.exception("Failed to finalize question card (answer_question)")
                        return None

                    # planner finished: show confirm/cancel buttons
                    if reporter:
                        try:
                            if agent_name in _CONFIRM_SUBTASK_AGENTS:
                                _answer_for_button = answer[:2000] if answer else ""
                                actions = [
                                    {
                                        "text": "✅ 确认生成",
                                        "type": "primary",
                                        "value": {
                                            "action": "confirm_agent_build",
                                            "session_key": session_key,
                                            "agent_name": agent_name,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                    {
                                        "text": "❌ 取消",
                                        "type": "danger",
                                        "value": {
                                            "action": "cancel_agent_build",
                                            "session_key": session_key,
                                            "agent_name": agent_name,
                                            "original_answer": _answer_for_button,
                                        },
                                    },
                                ]
                                reporter.finalize("completed", answer, actions=actions)
                                session.last_card_message_id = reporter.card_message_id
                            else:
                                reporter.finalize("completed", answer)
                        except Exception:
                            logger.exception("Failed to finalize step reporter (answer_question)")

                    # Inject result into chat_agent context
                    chat_session = self._session_manager.get(chat_id)
                    if chat_session and chat_session.initialized and chat_session.agent:
                        summary = (
                            f"[子任务结果 - {agent_name}]\n"
                            f"结果: {answer}"
                        )
                        chat_session.agent.add_user_message(summary)

                    return None

                # === confirm path: processing after Phase 2 builder completion ===
                # Inject result into chat_agent context
                chat_session = self._session_manager.get(chat_id)
                if chat_session and chat_session.initialized and chat_session.agent:
                    summary = (
                        f"[子任务结果 - {agent_name} Phase 2]\n"
                        f"结果: {answer}"
                    )
                    chat_session.agent.add_user_message(summary)

                if reporter:
                    try:
                        reporter.finalize("completed", answer)
                    except Exception:
                        logger.exception("Failed to finalize step reporter")

                # Update Phase 1 card: change from "generating" to "completed", keeping original plan content
                phase1_content = original_answer + "\n\n---\n> ✅ Agent 已成功创建。详情请查看下方回复。" if original_answer else "Agent 已成功创建。\n\n详情请查看下方回复。"
                self._patch_phase1_card(
                    card_message_id, "✅ Agent 创建完成",
                    phase1_content, "green",
                )

                # Phase 2 complete, clean up subtask session, subsequent messages go back to chat_agent
                self._session_manager.remove(session_key)

                return None

            except Exception as e:
                logger.exception(
                    "Card action subtask failed: key=%s", session_key
                )
                if reporter:
                    try:
                        reporter.finalize("failed")
                    except Exception:
                        logger.exception("Failed to finalize reporter on error")

                if action_type == "confirm":
                    # Update Phase 1 card: show failure status, keeping original plan content
                    phase1_content = original_answer + f"\n\n---\n> ❌ Agent 创建过程中出错：{str(e)[:500]}" if original_answer else f"Agent 创建过程中出错。\n\n{str(e)[:500]}"
                    self._patch_phase1_card(
                        card_message_id, "❌ Agent 创建失败",
                        phase1_content, "red",
                    )

                return f"会话子任务执行出错: {e}"

    def _patch_phase1_card(
        self,
        card_message_id: str | None,
        title: str,
        content: str,
        header_template: str,
    ) -> None:
        """Update the Phase 1 card status (called after Phase 2 completion/failure)."""
        if not card_message_id or not self._feishu_client:
            return
        try:
            from .messaging.sender import patch_card_message
            patch_card_message(
                self._feishu_client,
                card_message_id,
                title=title,
                content=content,
                header_template=header_template,
            )
        except Exception:
            logger.exception("Failed to update Phase 1 card: %s", card_message_id)

    @staticmethod
    def _check_delegation(session) -> dict[str, str] | None:
        """Check if chat_agent triggered delegation via delegate_to_agent.

        Scan the last few steps of the trajectory's ToolMessages for a delegated=True marker.
        """
        if not session.initialized or not session.agent:
            return None
        traj = session.agent.trajectory
        if not traj or not traj.steps:
            return None
        for step in reversed(traj.steps[-3:]):
            for resp in step.tool_responses:
                if getattr(resp, "name", "") == "delegate_to_agent":
                    info = (getattr(resp, "meta", None) or {}).get("info", {})
                    if info.get("delegated"):
                        return {
                            "agent_name": info["agent_name"],
                            "task": info["task"],
                        }
        return None

    def _find_active_subtask(self, chat_id: str) -> str | None:
        """Find if there is an active subtask session for this chat.

        If one exists, subsequent messages are routed directly to the subtask session
        (supporting multi-turn plan modification, etc.).
        """
        for agent_name in _SESSION_SUBTASK_AGENTS:
            session_key = f"{chat_id}:{agent_name}"
            sub = self._session_manager.get(session_key)
            if sub and sub.initialized:
                logger.info(
                    "Active subtask session found: key=%s, routing there",
                    session_key,
                )
                return agent_name
        return None

    @staticmethod
    def _parse_plan_todos(plan_text: str) -> list[str]:
        """Parse the TODO list from planner output.

        Expected format::

            ---PLAN_TODO---
            - [ ] Create directory structure
            - [ ] Create system_prompt.txt
            ---END_TODO---
        """
        todos: list[str] = []
        in_todo = False
        for line in plan_text.split("\n"):
            stripped = line.strip()
            if "---PLAN_TODO---" in stripped:
                in_todo = True
                continue
            if "---END_TODO---" in stripped:
                break
            if in_todo and stripped.startswith("- [ ]"):
                label = stripped[5:].strip()
                if label:
                    todos.append(label)
        return todos

    def _on_task_done(self, future, chat_id: str, message_id: str) -> None:
        """Task completion callback."""
        self._active_tasks.pop(message_id, None)

        try:
            result_text = future.result(timeout=0)
        except TimeoutError:
            result_text = f"任务超时（超过 {self._task_timeout} 秒）"
        except Exception as e:
            result_text = f"任务执行异常: {e}"

        # None means the reporter card already contains the answer, no extra message needed
        if result_text is None:
            return

        if self._on_result:
            try:
                self._on_result(chat_id, message_id, result_text)
            except Exception:
                logger.exception("Error in on_result callback")

    def _send_welcome_card(self, chat_id: str, message_id: str) -> None:
        """Send a welcome card introducing bot features and usage."""
        if not self._feishu_client:
            # fallback: plain text
            if self._on_result:
                self._on_result(
                    chat_id, message_id,
                    "新会话已开始。直接发送消息即可对话，或使用 /agent <名称> <任务> 调用专属智能体。",
                )
            return

        from .messaging.sender import send_card_message

        content = (
            "**直接对话**\n"
            "发送任何消息即可开始对话，我会记住上下文进行多轮交流。\n\n"
            "**创建智能体**\n"
            "直接告诉我你想创建什么智能体，例如：「帮我创建一个能总结文档的 agent」，"
            "我会自动委派给 Agent Builder 完成设计与构建。\n\n"
            "**指定智能体执行任务**\n"
            "`/agent <名称> <任务描述>`\n"
            "例如：`/agent doc_summarizer 总结这个文件 README.md`\n\n"
            "---\n"
            "**常用命令**\n"
            "`/help` — 显示本帮助信息\n"
            "`/new` — 清除上下文，开始新会话"
        )

        send_card_message(
            self._feishu_client,
            chat_id,
            title="👋 新会话已开始",
            content=content,
            reply_to_message_id=message_id,
            header_template="green",
        )

    def _send_help_card(self, chat_id: str, message_id: str) -> None:
        """Send a usage help card."""
        if not self._feishu_client:
            if self._on_result:
                self._on_result(
                    chat_id, message_id,
                    "使用帮助：直接发消息对话；/agent <名称> <任务> 调用智能体；/new 新会话。",
                )
            return

        from .messaging.sender import send_card_message

        content = (
            "**直接对话**\n"
            "发送任何消息即可开始多轮对话，我会记住上下文。\n\n"
            "**创建智能体**\n"
            "直接描述你的需求，例如：「帮我创建一个能总结文档的 agent」，"
            "我会自动委派给 Agent Builder 完成设计与构建。\n\n"
            "**指定智能体执行任务**\n"
            "`/agent <名称> <任务描述>`\n"
            "例如：`/agent doc_summarizer 总结这个文件 README.md`\n\n"
            "---\n"
            "**命令列表**\n"
            "`/help` — 显示本帮助信息\n"
            "`/new` — 清除上下文，开始新会话"
        )

        send_card_message(
            self._feishu_client,
            chat_id,
            title="📖 使用帮助",
            content=content,
            reply_to_message_id=message_id,
            header_template="blue",
        )

    def shutdown(self, wait: bool = False) -> None:
        """Shut down the dispatcher and all sessions."""
        logger.info("Shutting down task dispatcher...")
        self._session_manager.shutdown()
        self._executor.shutdown(wait=wait)
        logger.info("Task dispatcher shut down")
