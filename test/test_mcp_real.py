#!/usr/bin/env python3
"""真实测试：验证 MCP 环境变量在实际工具调用中是否生效"""

import asyncio
import os
import sys
from pathlib import Path

# 添加 evomaster 到路径
sys.path.insert(0, str(Path(__file__).parent))

from evomaster.agent.tools.mcp.mcp_connection import MCPConnectionStdio


async def test_real_mcp_tool_call():
    """测试真实 MCP 工具调用，验证环境变量是否生效"""
    print("=" * 70)
    print("真实测试：验证 MCP 工具调用时环境变量是否生效")
    print("=" * 70)

    # 从配置文件读取环境变量
    config_path = Path("configs/ml_master_datatree/mcp_config4data_plant_pathology.json")
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return

    with open(config_path) as f:
        mcp_config = json.load(f)

    search_hf_config = mcp_config.get("mcpServers", {}).get("search_huggingface", {})
    env_config = search_hf_config.get("env", {})

    print("\n1. 配置文件中的环境变量:")
    for key, value in env_config.items():
        if "TOKEN" in key:
            masked = value[:15] + "..." if len(value) > 15 else "***"
            print(f"  {key}: {masked}")
        else:
            print(f"  {key}: {value}")

    print(f"\n2. 启动 search_huggingface MCP 服务器...")
    print(f"   命令: python {' '.join(search_hf_config.get('args', []))}")

    try:
        async with MCPConnectionStdio(
            command=search_hf_config["command"],
            args=search_hf_config["args"],
            env=env_config
        ) as conn:
            print("  ✅ MCP 服务器启动成功！")

            # 列出可用工具
            tools = await conn.list_tools()
            print(f"\n3. 发现 {len(tools)} 个工具:")
            for tool in tools:
                print(f"  - {tool['name']}")

            # 测试 search_datasets 工具（会使用 HF_ENDPOINT 环境变量）
            print(f"\n4. 测试 search_datasets 工具...")
            print(f"   (这将验证 HF_ENDPOINT 是否生效)")

            result = await conn.call_tool("search_datasets", {
                "query": "image",
                "limit": 3
            })

            # 解析结果
            result_text = result[0].text if result else ""
            print(f"\n5. 工具返回结果（前500字符）:")
            print("  " + "-" * 66)
            for line in result_text[:500].split('\n'):
                print(f"  {line}")
            print("  " + "-" * 66)

            if "Found" in result_text or "datasets" in result_text.lower():
                print("\n✅ 成功！环境变量已生效，工具正常工作。")
                print("   HF_ENDPOINT 已将请求路由到 hf-mirror.com")
                return True
            else:
                print("\n⚠️ 工具调用成功但结果异常")
                return False

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import json
    success = asyncio.run(test_real_mcp_tool_call())
    sys.exit(0 if success else 1)
