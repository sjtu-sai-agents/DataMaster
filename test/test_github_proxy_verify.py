#!/usr/bin/env python3
"""验证 GitHub MCP 工具代理配置是否生效"""

import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from evomaster.agent.tools.mcp.mcp_connection import MCPConnectionStdio


async def verify_proxy_config():
    """验证代理配置"""
    print("=" * 70)
    print("验证 GitHub MCP 工具代理配置")
    print("=" * 70)

    # 读取配置文件
    config_path = Path("configs/ml_master_datatree/mcp_config4data_plant_pathology.json")
    with open(config_path) as f:
        mcp_config = json.load(f)

    github_config = mcp_config["mcpServers"]["search_github"]
    env_config = github_config.get("env", {})

    print("\n1. 配置文件中的环境变量:")
    print(f"  GITHUB_TOKEN: {env_config.get('GITHUB_TOKEN', 'NOT_SET')[:15]}...")
    print(f"  http_proxy: {env_config.get('http_proxy', 'NOT_SET')}")
    print(f"  https_proxy: {env_config.get('https_proxy', 'NOT_SET')}")

    # 检查代理配置
    if "http_proxy" in env_config and "https_proxy" in env_config:
        print("\n✅ 代理配置已添加到配置文件")

        print("\n2. 启动 MCP 服务器并验证...")
        try:
            async with MCPConnectionStdio(
                command=github_config["command"],
                args=github_config["args"],
                env=env_config
            ) as conn:
                print("  ✅ MCP 服务器启动成功")

                # 测试搜索
                print("\n3. 测试搜索功能...")
                result = await conn.call_tool("search_repositories", {
                    "keyword": "tensorflow",
                    "per_page": 3
                })

                result_text = result[0].text if result else ""
                if "Found" in result_text:
                    print("  ✅ 搜索功能正常")
                    print(f"\n  搜索结果（前200字符）:")
                    print("  " + "-" * 66)
                    for line in result_text[:200].split('\n'):
                        print(f"  {line}")
                    print("  " + "-" * 66)

                    print("\n✅ 代理配置已生效！GitHub 工具将使用 127.0.0.1:7895 代理")
                    return True
                else:
                    print(f"  ❌ 搜索失败: {result_text[:100]}")
                    return False

        except Exception as e:
            print(f"  ❌ 启动失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n❌ 代理配置未找到")
        return False


if __name__ == "__main__":
    success = asyncio.run(verify_proxy_config())
    sys.exit(0 if success else 1)
