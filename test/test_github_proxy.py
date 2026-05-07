#!/usr/bin/env python3
"""
测试 GitHub MCP 工具使用代理前后的性能对比

requests 库会自动从环境变量读取代理设置：
- http_proxy
- https_proxy
- HTTP_PROXY
- HTTPS_PROXY
"""

import asyncio
import os
import sys
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evomaster.agent.tools.mcp.mcp_connection import MCPConnectionStdio


def test_requests_with_proxy():
    """先测试 requests 库是否能正确使用代理"""
    import requests

    print("=" * 70)
    print("步骤 1: 验证 requests 库代理支持")
    print("=" * 70)

    # 测试代理是否可用
    proxy = {
        "http": "http://127.0.0.1:7895",
        "https": "http://127.0.0.1:7895"
    }

    print("\n1.1 测试代理连接...")
    try:
        # 先测试代理是否可用（连接到 Google）
        resp = requests.get(
            "https://www.google.com",
            proxies=proxy,
            timeout=5
        )
        print(f"  ✅ 代理可用: 127.0.0.1:7895")
    except Exception as e:
        print(f"  ⚠️  代理不可用或连接失败: {e}")
        print(f"  提示: 请确保代理服务正在运行 (127.0.0.1:7895)")
        return False

    # 测试 GitHub API 请求
    print("\n1.2 测试 GitHub API 请求（直接请求）...")
    try:
        start = time.time()
        resp = requests.get(
            "https://api.github.com/zen",
            timeout=10
        )
        elapsed = time.time() - start
        print(f"  ✅ 直接请求成功: {elapsed:.3f}s")
    except Exception as e:
        print(f"  ❌ 直接请求失败: {e}")

    print("\n1.3 测试 GitHub API 请求（通过代理）...")
    try:
        start = time.time()
        resp = requests.get(
            "https://api.github.com/zen",
            proxies=proxy,
            timeout=10
        )
        elapsed = time.time() - start
        print(f"  ✅ 代理请求成功: {elapsed:.3f}s")
    except Exception as e:
        print(f"  ❌ 代理请求失败: {e}")
        return False

    return True


async def benchmark_github_tool(use_proxy=False, iterations=3):
    """基准测试 GitHub MCP 工具性能"""
    from evomaster.agent.tools.mcp.mcp_connection import MCPConnectionStdio

    # 从配置文件读取
    import json
    config_path = Path("configs/ml_master_datatree/mcp_config4data_plant_pathology.json")
    with open(config_path) as f:
        mcp_config = json.load(f)

    github_config = mcp_config["mcpServers"]["search_github"]
    env_config = github_config.get("env", {}).copy()

    # 添加代理设置
    if use_proxy:
        env_config["http_proxy"] = "127.0.0.1:7895"
        env_config["https_proxy"] = "127.0.0.1:7895"

    mode = "使用代理" if use_proxy else "不使用代理"
    print(f"\n{'=' * 70}")
    print(f"步骤 2: 基准测试 GitHub MCP 工具 - {mode}")
    print(f"{'=' * 70}")

    if use_proxy:
        print(f"\n代理设置:")
        print(f"  http_proxy: {env_config['http_proxy']}")
        print(f"  https_proxy: {env_config['https_proxy']}")

    times = []

    for i in range(iterations):
        print(f"\n第 {i + 1}/{iterations} 次测试...")
        try:
            start = time.time()

            async with MCPConnectionStdio(
                command=github_config["command"],
                args=github_config["args"],
                env=env_config
            ) as conn:
                # 启动时间
                startup_time = time.time() - start
                print(f"  启动耗时: {startup_time:.3f}s")

                # 测试搜索仓库
                start_search = time.time()
                result = await conn.call_tool("search_repositories", {
                    "keyword": "machine learning",
                    "language": "python",
                    "per_page": 10
                })
                search_time = time.time() - start_search

                total_time = time.time() - start
                times.append(total_time)

                # 解析结果
                result_text = result[0].text if result else ""
                if "Found" in result_text or "repositories" in result_text.lower():
                    print(f"  ✅ 搜索成功: 总耗时 {total_time:.3f}s (搜索: {search_time:.3f}s)")
                else:
                    print(f"  ⚠️  搜索异常: {result_text[:100]}")

        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    # 统计
    if times:
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)

        print(f"\n{'=' * 70}")
        print(f"统计结果 ({mode}):")
        print(f"{'=' * 70}")
        print(f"  平均耗时: {avg_time:.3f}s")
        print(f"  标准差:   {std_dev:.3f}s")
        print(f"  最快:     {min_time:.3f}s")
        print(f"  最慢:     {max_time:.3f}s")

        return times

    return None


async def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("GitHub MCP 工具代理性能测试")
    print("=" * 70)

    # 步骤 1: 验证代理
    proxy_available = test_requests_with_proxy()

    if not proxy_available:
        print("\n⚠️  代理不可用，跳过代理测试")
        print("提示: 请先启动代理服务 (127.0.0.1:7895)")
        return

    # 步骤 2: 测试不使用代理的情况
    print("\n\n" + "=" * 70)
    print("开始基准测试...")
    print("=" * 70)

    times_no_proxy = await benchmark_github_tool(use_proxy=False, iterations=3)

    # 步骤 3: 测试使用代理的情况
    times_with_proxy = await benchmark_github_tool(use_proxy=True, iterations=3)

    # 步骤 4: 对比结果
    if times_no_proxy and times_with_proxy:
        avg_no_proxy = statistics.mean(times_no_proxy)
        avg_with_proxy = statistics.mean(times_with_proxy)

        print("\n\n" + "=" * 70)
        print("最终对比结果")
        print("=" * 70)

        print(f"\n不使用代理:")
        print(f"  平均耗时: {avg_no_proxy:.3f}s")

        print(f"\n使用代理 (127.0.0.1:7895):")
        print(f"  平均耗时: {avg_with_proxy:.3f}s")

        if avg_with_proxy < avg_no_proxy:
            speedup = ((avg_no_proxy - avg_with_proxy) / avg_no_proxy) * 100
            print(f"\n✅ 代理提速: {speedup:.1f}%")
            print(f"   节省时间: {avg_no_proxy - avg_with_proxy:.3f}s")
        else:
            slowdown = ((avg_with_proxy - avg_no_proxy) / avg_no_proxy) * 100
            print(f"\n❌ 代理减速: {slowdown:.1f}%")
            print(f"   增加时间: {avg_with_proxy - avg_no_proxy:.3f}s")
            print(f"\n   可能原因:")
            print(f"   - 代理服务器延迟较高")
            print(f"   - 本地网络已经很快，代理反而增加了额外跳转")

        print("\n" + "=" * 70)

        # 推荐
        if avg_with_proxy < avg_no_proxy:
            print("\n💡 建议: 在配置文件中添加代理设置")
            print("\n在 mcp_config4data_plant_pathology.json 的 search_github 部分添加:")
            print("""
    "search_github": {
      "command": "python",
      "args": ["search_dataset_tools/search_github.py"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxx",
        "http_proxy": "127.0.0.1:7895",
        "https_proxy": "127.0.0.1:7895"
      }
    }
            """)


if __name__ == "__main__":
    asyncio.run(main())
