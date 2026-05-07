#!/usr/bin/env python3
"""测试 MCP 服务器环境变量是否正确传递"""

import asyncio
import os
import sys
import json
from pathlib import Path

# 添加 evomaster 到路径
sys.path.insert(0, str(Path(__file__).parent))

from evomaster.agent.tools.mcp.mcp_connection import MCPConnectionStdio
from evomaster.agent.tools.mcp.mcp_manager import MCPToolManager


async def test_mcp_env_injection():
    """测试 MCP 连接时环境变量是否被注入"""
    print("=" * 60)
    print("测试 MCP 环境变量注入")
    print("=" * 60)

    # 模拟配置文件中的环境变量
    config_env = {
        "HF_TOKEN": "HF_TOKEN_PLACEHOLDER",
        "HF_ENDPOINT": "https://hf-mirror.com",
        "HF_HOME": "/data/HF_Cache_dataevo"
    }

    print("\n1. 配置的环境变量:")
    for key, value in config_env.items():
        print(f"  {key}: {value}")

    print("\n2. 创建 MCPConnectionStdio...")
    # 创建 MCP 连接（不实际启动，只检查 env 属性）
    conn = MCPConnectionStdio(
        command="python",
        args=["search_dataset_tools/search_huggingface.py"],
        env=config_env
    )

    print("\n3. 合并后的环境变量（os.environ | config_env）:")
    # 检查关键环境变量是否存在
    test_vars = ["HF_TOKEN", "HF_ENDPOINT", "HF_HOME"]
    all_present = True
    for var in test_vars:
        if var in conn.env:
            value = conn.env[var]
            if var == "HF_TOKEN":
                # 隐藏 token 的大部分
                masked = value[:15] + "..." if len(value) > 15 else "***"
                print(f"  ✅ {var}: {masked}")
            else:
                print(f"  ✅ {var}: {value}")
        else:
            print(f"  ❌ {var}: 未设置")
            all_present = False

    print("\n" + "=" * 60)
    if all_present:
        print("✅ 所有环境变量都已正确设置！")
        print("环境变量会被传递给 MCP 子进程。")
    else:
        print("❌ 部分环境变量未设置！")
    print("=" * 60)


async def test_with_real_mcp_start():
    """测试实际启动 MCP 服务器时的环境变量"""
    print("\n" + "=" * 60)
    print("测试真实 MCP 服务器启动")
    print("=" * 60)

    config_env = {
        "HF_TOKEN": "HF_TOKEN_PLACEHOLDER",
        "HF_ENDPOINT": "https://hf-mirror.com",
        "HF_HOME": "/data/HF_Cache_dataevo"
    }

    print("\n1. 尝试启动 search_huggingface MCP 服务器...")

    try:
        async with MCPConnectionStdio(
            command="python",
            args=["search_dataset_tools/search_huggingface.py"],
            env=config_env
        ) as conn:
            print("  ✅ MCP 服务器启动成功！")

            # 列出可用工具
            tools = await conn.list_tools()
            print(f"\n2. 发现 {len(tools)} 个工具:")
            for tool in tools[:3]:  # 只显示前3个
                print(f"  - {tool['name']}")

            print("\n3. 测试 search_datasets 工具（调用时会使用环境变量）...")
            result = await conn.call_tool("search_datasets", {"query": "test", "limit": 5})
            print(f"  结果长度: {len(str(result))} 字符")

    except Exception as e:
        print(f"  ❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 测试1：环境变量注入
    asyncio.run(test_mcp_env_injection())

    # 测试2：真实启动（可选）
    print("\n是否测试真实 MCP 服务器启动？(y/n): ", end="")
    # 如果是自动化测试，跳过输入
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        print("自动模式，跳过真实启动测试")
    else:
        try:
            choice = input().strip().lower()
            if choice == 'y':
                asyncio.run(test_with_real_mcp_start())
        except (EOFError, KeyboardInterrupt):
            print("\n跳过真实启动测试")
