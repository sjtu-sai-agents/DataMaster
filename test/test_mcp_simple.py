#!/usr/bin/env python3
"""
简化版MCP测试 - 直接测试第一次调用慢的问题
"""

import json
import subprocess
import sys
import time
from pathlib import Path


def test_single_mcp_call(query, limit=3, timeout=15):
    """测试单次MCP调用"""
    script_path = Path(__file__).parent / "search_dataset_tools" / "search_huggingface_math_posttrain.py"
    python_bin = Path(__file__).parent / ".venv_gpu2" / "bin" / "python"

    # 构造MCP请求
    requests_text = (
        '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05",'
        '"capabilities":{},"clientInfo":{"name":"test","version":"1.0"}},"id":0}\n'
        f'{{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"search_datasets",'
        f'"arguments":{{"query":"{query}","limit":{limit}}}}},"id":1}}\n'
    )

    print(f"Query: {query}, Limit: {limit}, Timeout: {timeout}s")
    start = time.time()

    try:
        result = subprocess.run(
            [str(python_bin), str(script_path)],
            input=requests_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start

        # 检查是否成功
        success = False
        if result.returncode == 0 and result.stdout:
            for line in result.stdout.strip().split('\n'):
                if '"result"' in line and '"content"' in line:
                    success = True
                    break

        print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}, Time: {elapsed:.2f}s")
        if not success:
            print(f"Returncode: {result.returncode}")
            if result.stderr:
                print(f"Stderr: {result.stderr[:200]}")
        return success, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"Result: ⏱️  TIMEOUT, Time: {elapsed:.2f}s")
        return False, elapsed


def main():
    print("=" * 70)
    print("简化版 MCP 调用测试")
    print("=" * 70)
    print()

    queries = [
        "math reasoning",
        "competition math",
        "problem solving",
    ]

    results = []
    for i, query in enumerate(queries, 1):
        print(f"\n[测试 {i}/3]")
        success, elapsed = test_single_mcp_call(query, limit=3, timeout=15)
        results.append((query, success, elapsed))

        if not success:
            print(f"⚠️  第{i}次调用失败")

        # 短暂等待
        if i < len(queries):
            time.sleep(0.5)

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    for i, (query, success, elapsed) in enumerate(results, 1):
        status = "✅" if success else "❌"
        print(f"{status} 调用#{i} ({query}): {elapsed:.2f}s")

    failures = [i+1 for i, (_, success, _) in enumerate(results) if not success]
    if failures:
        print(f"\n❌ 失败: 第 {', '.join(map(str, failures))} 次调用")
        if 1 in failures:
            print("   → 第1次调用就失败/超时，说明MCP Server启动或首次调用有问题")
        elif 2 in failures:
            print("   → 第2次调用失败，符合Event Loop状态机bug特征")
    else:
        print("\n✅ 全部成功")


if __name__ == "__main__":
    main()
