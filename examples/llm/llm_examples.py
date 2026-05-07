"""LLM 使用示例

包含多种 LLM 使用场景的示例代码。
"""

from pathlib import Path

import yaml

from evomaster.agent import (
    Dialog,
    FunctionSpec,
    SystemMessage,
    ToolSpec,
    UserMessage,
)
from evomaster.utils.llm import LLMConfig, create_llm


def load_config():
    """加载配置"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def example_simple_chat():
    """示例 1: 简单对话"""
    print("\n" + "=" * 60)
    print("示例 1: 简单对话")
    print("=" * 60)

    # 加载配置
    config = load_config()
    llm_name = config["llm"]["default"]
    llm_config = LLMConfig(**config["llm"][llm_name])

    # 创建 LLM
    llm = create_llm(llm_config)
    print(f"使用 LLM: {llm_config.model}")
    print(f"Base URL: {llm_config.base_url}")

    # 创建对话
    dialog = Dialog(
        messages=[
            SystemMessage(content="你是一个友好的助手。"),
            UserMessage(content="你好，请用一句话介绍一下你自己。"),
        ]
    )

    # 调用 LLM
    print("\n发送请求中...")
    response = llm.query(dialog)

    print(f"\n助手回复:\n{response.content}")

    # 查看元数据
    if response.meta:
        print(f"\n元数据:")
        print(f"  模型: {response.meta.get('model', 'N/A')}")
        if "usage" in response.meta:
            usage = response.meta["usage"]
            print(f"  输入 tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"  输出 tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"  总计 tokens: {usage.get('total_tokens', 'N/A')}")


def example_multi_turn_chat():
    """示例 2: 多轮对话"""
    print("\n" + "=" * 60)
    print("示例 2: 多轮对话")
    print("=" * 60)

    # 加载配置
    config = load_config()
    llm_name = config["llm"]["default"]
    llm_config = LLMConfig(**config["llm"][llm_name])

    # 创建 LLM
    llm = create_llm(llm_config)
    print(f"使用 LLM: {llm_config.model}")

    # 初始化对话
    dialog = Dialog(
        messages=[
            SystemMessage(content="你是一个数学助手，擅长解答数学问题。"),
        ]
    )

    # 第一轮对话
    print("\n第一轮对话:")
    user_msg_1 = "请帮我计算 15 的平方根，保留两位小数。"
    print(f"  用户: {user_msg_1}")

    dialog.add_message(UserMessage(content=user_msg_1))
    response_1 = llm.query(dialog)
    dialog.add_message(response_1)

    print(f"  助手: {response_1.content}")

    # 第二轮对话（依赖前面的上下文）
    print("\n第二轮对话:")
    user_msg_2 = "那这个数字的 3 次方是多少？"
    print(f"  用户: {user_msg_2}")

    dialog.add_message(UserMessage(content=user_msg_2))
    response_2 = llm.query(dialog)
    dialog.add_message(response_2)

    print(f"  助手: {response_2.content}")

    # 第三轮对话
    print("\n第三轮对话:")
    user_msg_3 = "总结一下我们刚才的计算过程。"
    print(f"  用户: {user_msg_3}")

    dialog.add_message(UserMessage(content=user_msg_3))
    response_3 = llm.query(dialog)
    dialog.add_message(response_3)

    print(f"  助手: {response_3.content}")


def example_tool_calling():
    """示例 3: 工具调用"""
    print("\n" + "=" * 60)
    print("示例 3: 工具调用")
    print("=" * 60)

    # 加载配置
    config = load_config()
    llm_name = config["llm"]["default"]
    llm_config = LLMConfig(**config["llm"][llm_name])

    # 创建 LLM
    llm = create_llm(llm_config)
    print(f"使用 LLM: {llm_config.model}")

    # 定义工具
    tools = [
        ToolSpec(
            type="function",
            function=FunctionSpec(
                name="get_weather",
                description="获取指定城市的天气信息",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市名称，例如：北京、上海",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度单位",
                        },
                    },
                    "required": ["location"],
                },
            ),
        ),
        ToolSpec(
            type="function",
            function=FunctionSpec(
                name="calculate",
                description="执行数学计算",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学表达式，例如：2+2, sqrt(16)",
                        }
                    },
                    "required": ["expression"],
                },
            ),
        ),
    ]

    # 创建对话
    dialog = Dialog(
        messages=[
            SystemMessage(content="你是一个助手，可以使用工具来帮助用户。"),
            UserMessage(content="北京今天天气怎么样？温度用摄氏度表示。"),
        ],
        tools=tools,
    )

    # 调用 LLM
    print("\n发送请求中...")
    response = llm.query(dialog)

    print(f"\n助手回复: {response.content if response.content else '(无文本回复)'}")

    # 检查工具调用
    if response.tool_calls:
        print(f"\n工具调用:")
        for tool_call in response.tool_calls:
            print(f"  工具: {tool_call.function.name}")
            print(f"  参数: {tool_call.function.arguments}")

        # 模拟工具执行结果
        print("\n模拟工具执行结果:")
        tool_result = {
            "location": "北京",
            "temperature": 15,
            "unit": "celsius",
            "condition": "晴天",
        }
        print(f"  {tool_result}")


def example_different_models():
    """示例 4: 使用不同的模型"""
    print("\n" + "=" * 60)
    print("示例 4: 使用不同的模型")
    print("=" * 60)

    # 加载配置
    config = load_config()

    # 测试问题
    question = "用一句话解释什么是机器学习。"

    # 遍历所有配置的模型
    for model_name, model_config in config["llm"].items():
        if model_name == "default":
            continue

        print(f"\n使用模型: {model_name}")
        print("-" * 40)

        try:
            # 创建 LLM 配置
            llm_config = LLMConfig(**model_config)
            llm = create_llm(llm_config)

            # 创建对话
            dialog = Dialog(
                messages=[
                    SystemMessage(content="你是一个简洁的助手。"),
                    UserMessage(content=question),
                ]
            )

            # 查询
            response = llm.query(dialog)
            print(f"回复: {response.content}")

        except Exception as e:
            print(f"错误: {e}")


def example_streaming():
    """示例 5: 流式输出 (如果支持)"""
    print("\n" + "=" * 60)
    print("示例 5: 流式输出")
    print("=" * 60)

    # 加载配置
    config = load_config()
    llm_name = config["llm"]["default"]
    llm_config = LLMConfig(**config["llm"][llm_name])

    # 创建 LLM
    llm = create_llm(llm_config)
    print(f"使用 LLM: {llm_config.model}")

    # 创建对话
    dialog = Dialog(
        messages=[
            SystemMessage(content="你是一个助手。"),
            UserMessage(content="请写一首关于春天的短诗，四行即可。"),
        ]
    )

    print("\n生成中...")
    print("-" * 40)

    # 检查是否支持流式输出
    if hasattr(llm, "stream"):
        # 流式输出
        for chunk in llm.stream(dialog):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print()
    else:
        # 普通输出
        response = llm.query(dialog)
        print(response.content)


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("LLM 使用示例")
    print("=" * 60)

    # 运行各个示例
    example_simple_chat()
    example_multi_turn_chat()
    example_tool_calling()
    example_different_models()
    example_streaming()

    print("\n" + "=" * 60)
    print("所有示例运行完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
