#!/usr/bin/env python3
"""测试 Grade HTTP 服务器（端口 7777）"""
import asyncio
import sys
from pathlib import Path

import aiohttp
import requests


def test_health():
    """测试健康检查"""
    try:
        resp = requests.get("http://127.0.0.1:7777/health", timeout=5)
        resp.raise_for_status()
        print(f"✅ Health check passed: {resp.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_grade_api(submission_path: str, exp_id: str = "leaf-classification"):
    """测试评分 API"""
    if not Path(submission_path).exists():
        print(f"❌ Submission file not found: {submission_path}")
        return False

    payload = {
        "exp_id": exp_id,
        "submission_path": submission_path,
        "timeout": 60,
    }

    try:
        resp = requests.post(
            "http://127.0.0.1:7777/grade",
            json=payload,
            timeout=70,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("success"):
            print(f"✅ Grade API test passed")
            print(f"   Return code: {data.get('returncode')}")
            print(f"   Stdout length: {len(data.get('stdout', ''))}")
            print(f"   Stderr length: {len(data.get('stderr', ''))}")
            return True
        else:
            print(f"❌ Grade API returned error: {data.get('error')}")
            return False

    except Exception as e:
        print(f"❌ Grade API test failed: {e}")
        return False


async def test_async_grade_api(submission_path: str, exp_id: str = "leaf-classification"):
    """测试异步评分 API（模拟 Agent 调用）"""
    if not Path(submission_path).exists():
        print(f"❌ Submission file not found: {submission_path}")
        return False

    url = "http://127.0.0.1:7777/grade"
    payload = {
        "exp_id": exp_id,
        "submission_path": submission_path,
        "timeout": 60,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=70)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"❌ Async grade API returned status {resp.status}: {error_text}")
                    return False

                data = await resp.json()

                if data.get("success"):
                    print(f"✅ Async grade API test passed")
                    return True
                else:
                    print(f"❌ Async grade API returned error: {data.get('error')}")
                    return False

    except Exception as e:
        print(f"❌ Async grade API test failed: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test Grade HTTP Server (Port 7777)")
    parser.add_argument("--submission", type=str, default="${PROJECT_ROOT}/runs/ml_master_datatree_20260411_201646/workspaces/task_0/submission/submission_0ed14555cfec4f57bb39e6a35ecd618b.csv", help="Path to submission CSV file for testing")
    parser.add_argument("--exp-id", type=str, default="dog-breed-identification", help="Experiment ID")
    args = parser.parse_args()

    print("=" * 60)
    print("Grade HTTP Server Test Suite (Port 7777)")
    print("=" * 60)

    # 测试 1: 健康检查
    print("\n[Test 1] Health check...")
    if not test_health():
        print("\n❌ Server is not running. Please start it first:")
        print("   python initialize_grade_port.py --port 7777")
        return 1

    # 如果没有提供 submission 文件，只测试健康检查
    if not args.submission:
        print("\n✅ Basic test passed. Use --submission to test grade API.")
        return 0

    # 测试 2: 同步评分 API
    print("\n[Test 2] Synchronous grade API...")
    if not test_grade_api(args.submission, args.exp_id):
        return 1

    # 测试 3: 异步评分 API
    print("\n[Test 3] Asynchronous grade API...")
    if not asyncio.run(test_async_grade_api(args.submission, args.exp_id)):
        return 1

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
