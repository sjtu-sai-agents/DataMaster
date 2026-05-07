"""LLM query utilities for ML-Master

This module provides functions for querying LLMs with prompts.
It adapts ML-Master's query interface to work with EvoMaster's LLM infrastructure.
"""

import logging
from typing import Any, TYPE_CHECKING
from evomaster.agent import Agent, AgentConfig
from evomaster.utils.types import Dialog, UserMessage, SystemMessage, TaskInstance
from .response import extract_code, extract_text_up_to_code, extract_review

if TYPE_CHECKING:
    from evomaster.utils.types import Dialog


logger = logging.getLogger(__name__)


def extract_after_think(text: str) -> str:
    """Extract content after `` tag (for steerable reasoning models).

    Args:
        text: The full response text

    Returns:
        Content after `` tag, or original text if tag not found
    """
    if "" in text:
        # Find the content after the closing `` tag
        parts = text.split("", 1)
        if len(parts) > 1:
            return parts[1].strip()
    return text


def _messages_to_dialog(messages: list[dict]) -> "Dialog":
    """Convert messages list to EvoMaster Dialog object.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Dialog object
    """

    dialog = Dialog()
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            dialog.add_message(SystemMessage(content=content))
        else:  # user or assistant
            # For simplicity, treat all non-system messages as user messages
            dialog.add_message(UserMessage(content=content))

    return dialog


def plan_and_code_query(
    llm,
    prompt: str | list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
    steerable_reasoning: bool = False,
    **kwargs,
) -> tuple[str, str]:
    """Generate a natural language plan + code in the same LLM call.

    This function combines ML-Master's plan_and_code_query approach with
    EvoMaster's LLM interface.

    Args:
        llm: The LLM instance to query
        prompt: The prompt (can be string or list of messages)
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        steerable_reasoning: Whether to extract content after `` tag
        **kwargs: Additional arguments

    Returns:
        A tuple of (plan, code)
    """

    # Convert prompt to messages if needed
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    # Convert messages to Dialog
    dialog = _messages_to_dialog(messages)

    # Query the LLM using EvoMaster's interface
    assistant_message = llm.query(
        dialog,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if not assistant_message or not assistant_message.content:
        logger.warning("LLM returned empty response")
        return "", ""

    completion_text = assistant_message.content

    # Extract content after `` for steerable reasoning models
    if steerable_reasoning:
        completion_text = extract_after_think(completion_text)
        logger.info("Extracted content after `` tag", extra={"verbose": True})

    # Extract plan and code
    code = extract_code(completion_text)
    nl_text = extract_text_up_to_code(completion_text)

    if code and nl_text:
        return nl_text, code

    logger.warning("Failed to extract plan and code, returning full response")
    return nl_text or "", completion_text


def query_with_feedback(
    llm,
    system_prompt: dict,
    user_prompt: str | None,
    func_spec: Any | None = None,
    temperature: float = 0.7,
    **kwargs,
) -> dict:
    """Query LLM for feedback/review.

    This function is used for getting LLM feedback on code execution results.

    Args:
        llm: The LLM instance
        system_prompt: System prompt as a dictionary
        user_prompt: User prompt string
        func_spec: Optional function spec for tool calling
        temperature: Temperature for generation
        **kwargs: Additional arguments

    Returns:
        Response dictionary
    """
    # Convert system prompt dict to string
    system_message = _compile_prompt_to_md(system_prompt)

    messages = [
        {"role": "system", "content": system_message},
    ]

    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    else:
        # * use default system prompt
        messages.append(
            {
                "role": "user",
                "content": "Please evaluate the code execution output above.",
            }
        )

    # Convert messages to Dialog
    dialog = _messages_to_dialog(messages)
    logger.info(
        f"Querying LLM for feedback: system_prompt_len={len(system_message)}, has_user_prompt={bool(user_prompt)}"
    )

    # Query the LLM using EvoMaster's interface
    assistant_message = llm.query(dialog, temperature=temperature)

    if not assistant_message or not assistant_message.content:
        logger.warning("LLM returned empty response for feedback")
        return {"is_bug": True, "metric": None, "summary": "Failed to get feedback"}

    # Try to extract structured response

    result = extract_review(assistant_message.content)

    if result:
        return result

    # Fallback to simple parsing
    return {
        "is_bug": False,
        "metric": None,
        "summary": assistant_message.content[:500],
        "lower_is_better": False,
    }


def _compile_prompt_to_md(prompt: dict | str, level: int = 1) -> str:
    """Compile a prompt dict to markdown format.

    Args:
        prompt: Prompt dict or string
        level: Heading level for keys

    Returns:
        Compiled markdown string
    """
    if isinstance(prompt, str):
        return prompt

    md_parts = []
    for key, value in prompt.items():
        if isinstance(value, dict):
            md_parts.append(f"{'#' * level} {key}")
            md_parts.append(_compile_prompt_to_md(value, level + 1))
        elif isinstance(value, list):
            md_parts.append(f"{'#' * level} {key}")
            for item in value:
                if isinstance(item, dict):
                    md_parts.append(_compile_prompt_to_md(item, level + 1))
                else:
                    md_parts.append(f"- {item}")
        else:
            md_parts.append(f"{'#' * level} {key}")
            md_parts.append(str(value))

    return "\n".join(md_parts)


def code_query(
    llm,
    prompt: str | list[dict],
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> str:
    """Query LLM for code generation.

    Args:
        llm: The LLM instance
        prompt: The prompt
        temperature: Temperature
        max_tokens: Maximum tokens

    Returns:
        Generated code
    """

    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    # Convert messages to Dialog
    dialog = _messages_to_dialog(messages)

    # Query the LLM using EvoMaster's interface
    assistant_message = llm.query(
        dialog, temperature=temperature, max_tokens=max_tokens
    )

    if not assistant_message or not assistant_message.content:
        return ""

    code = extract_code(assistant_message.content)
    return code if code else assistant_message.content


def plan_and_code_query_with_agent(
    llm,
    session,
    tools,
    system_prompt: str,
    user_prompt: str,
    max_turns: int = 10,
    temperature: float = 0.7,
    **kwargs,
) -> tuple[str, str]:
    """使用 Agent 框架进行 plan_and_code 查询，支持多轮工具调用。

    这个函数将原本的单次 LLM 调用替换为一个 Agent 实例，
    Agent 可以使用工具（execute_bash, str_replace_editor, think 等）来辅助完成任务。

    Args:
        llm: The LLM instance to query
        session: Session instance for tool execution
        tools: ToolRegistry instance containing available tools
        system_prompt: System prompt for the agent
        user_prompt: User prompt (task description)
        max_turns: Maximum turns for the agent to run
        temperature: Temperature for generation
        **kwargs: Additional arguments

    Returns:
        A tuple of (plan, code)
        - plan: Natural language description of the approach
        - code: Generated Python code
    """

    # Create tmp_code directory for Agent's temporary code files
    import os
    workspace_path = session.config.workspace_path
    tmp_code_dir = os.path.join(workspace_path, "tmp_code")
    os.makedirs(tmp_code_dir, exist_ok=True)
    logger.info(f"Created tmp_code directory: {tmp_code_dir}")

    agent_cfg = AgentConfig(max_turns=max_turns)
    # * creating sub-agents
    agent = Agent(
        llm=llm,
        session=session,
        tools=tools,
        config=agent_cfg,
        enable_tools=True,
    )

    workspace_path = session.config.workspace_path
    tmp_code_dir = os.path.join(workspace_path, "tmp_code")
    os.makedirs(tmp_code_dir, exist_ok=True)
    logger.info(f"Created tmp_code directory: {tmp_code_dir}")

    # Load tool usage instructions from external file
    tool_usage_instruction_path = "playground/ml_master/prompts/new_prompts/tool_usage_instructions.md"
    try:
        with open(tool_usage_instruction_path, "r", encoding='utf-8') as file:
            tool_usage_instructions = file.read().format(tmp_code_dir=tmp_code_dir)
    except FileNotFoundError:
        logger.warning(f"Tool usage instructions file not found: {tool_usage_instruction_path}")

    agent._system_prompt = system_prompt + tool_usage_instructions

    task = TaskInstance(
        task_id="ml_master_task",
        task_type="code_generation",
        description=user_prompt,
    )

    logger.info(f"Starting Agent with max_turns={max_turns}, tools enabled")
    trajectory = agent.run(task)

    if trajectory.dialogs:
        # 获取最后一个对话的最后一条助手消息
        final_dialog = trajectory.dialogs[-1]
        for msg in reversed(final_dialog.messages):
            if hasattr(msg, "role") and msg.role == "assistant":
                final_content = msg.content
                break
        else:
            final_content = ""
    else:
        final_content = ""

    if not final_content:
        logger.warning("Agent returned empty response")
        return "", ""

    # 提取 plan 和 code
    code = extract_code(final_content)
    plan = extract_text_up_to_code(final_content)

    if code and plan:
        logger.info(
            f"Agent completed: extracted plan ({len(plan)} chars) and code ({len(code)} chars)"
        )
        return plan, code

    logger.warning("Failed to extract plan and code from agent response")
    return plan or "", final_content
