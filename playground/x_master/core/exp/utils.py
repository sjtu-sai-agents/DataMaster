"""X-Master 实验工具函数"""

import json
from typing import Any


def strip_think_and_exec(text: str) -> str:
    """清理文本中的 </think> 和 </execution_results> 标签及其之前的内容

    保留可见的答案部分，移除思考过程和执行结果的尾部标记。
    这个函数用于在传递给下游 Agent（如 Critic/Rewriter/Selector）之前
    清理上游 Agent 的输出，确保下游只看到最终答案而非中间过程。

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    if text is None:
        return ""
    out = text
    if "</think>" in out:
        out = out.split("</think>")[-1]
    if "</execution_results>" in out:
        out = out.split("</execution_results>")[-1]
    return out.strip()


def extract_agent_response(trajectory: Any) -> str:
    """从轨迹中提取Agent的最终回答

    Args:
        trajectory: 执行轨迹

    Returns:
        Agent的回答文本
    """
    if not trajectory or not trajectory.dialogs:
        return ""

    # 获取最后一个对话
    last_dialog = trajectory.dialogs[-1]
    
    # 查找最后一个助手消息
    for message in reversed(last_dialog.messages):
        if hasattr(message, 'role') and message.role.value == 'assistant':
            # 正常 assistant content
            if hasattr(message, 'content') and message.content:
                return message.content

            # tool_calls 形式的最终回答
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if not hasattr(tool_call, 'function'):
                        continue
                    func = tool_call.function

                    if not hasattr(func, 'arguments'):
                        continue

                    args = func.arguments

                    if not args:
                        continue

                    # arguments 可能是 JSON 字符串
                    try:
                        args_dict = json.loads(args)
                    except Exception:
                        continue

                    # 优先取 message 字段
                    if "message" in args_dict and args_dict["message"]:
                        return args_dict["message"]
            
    return ""
