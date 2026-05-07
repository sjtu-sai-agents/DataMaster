#!/usr/bin/env python3
"""
MCP调用死锁诊断脚本

区分两种可能的原因：
1. Event loop状态机问题（代码逻辑bug）
2. 线程竞争问题（资源耗尽）
"""

import asyncio
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_result(test_name, success, elapsed, details=""):
    status = f"{Colors.GREEN}✅ PASS" if success else f"{Colors.RED}❌ FAIL"
    print(f"{status}{Colors.RESET} {test_name}: {elapsed:.2f}s")
    if details:
        print(f"    {Colors.YELLOW}{details}{Colors.RESET}")


def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")


def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")


# ============================================================================
# 场景1: 测试 event loop 状态机（无负载）
# ============================================================================

class EventLoopTester:
    """模拟 evomaster/agent/tools/mcp/mcp.py 的 _call_mcp_tool_sync 逻辑"""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.call_count = 0

    async def _async_task(self, task_id: int):
        """模拟异步MCP调用"""
        await asyncio.sleep(0.1)  # 模拟网络延迟
        return f"Task {task_id} completed"

    def call_sync_v1_run_until_complete(self, task_id: int):
        """方法1: 总是用 run_until_complete（应该稳定）"""
        coro = self._async_task(task_id)
        if self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
        return self.loop.run_until_complete(coro)

    def call_sync_v2_original_logic(self, task_id: int):
        """方法2: 原始逻辑（可能有状态机bug）"""
        coro = self._async_task(task_id)

        try:
            if not self.loop.is_running():
                # 第一次调用走这里
                return self.loop.run_until_complete(coro)

            # 第二次调用走这里（可能死锁）
            fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
            return fut.result(timeout=5)  # 5秒超时用于测试

        except concurrent.futures.TimeoutError:
            raise TimeoutError("Event loop deadlock detected")

    def call_sync_v3_new_loop_each_time(self, task_id: int):
        """方法3: 每次创建新loop（避免状态问题）"""
        return asyncio.run(self._async_task(task_id))


def test_event_loop_state_machine():
    """测试event loop状态机是否有问题"""
    print_section("场景1: Event Loop 状态机测试（无负载）")

    results = []

    # 测试方法1: 总是用 run_until_complete
    print("测试方法1: 总是使用 run_until_complete")
    tester1 = EventLoopTester()
    for i in range(1, 4):
        start = time.time()
        try:
            result = tester1.call_sync_v1_run_until_complete(i)
            elapsed = time.time() - start
            print_result(f"  调用#{i}", True, elapsed, result)
            results.append(("v1", i, True, elapsed))
        except Exception as e:
            elapsed = time.time() - start
            print_result(f"  调用#{i}", False, elapsed, str(e))
            results.append(("v1", i, False, elapsed))

    # 测试方法2: 原始逻辑（状态切换）
    print("\n测试方法2: 原始逻辑（状态切换 - run_until_complete → run_coroutine_threadsafe）")
    tester2 = EventLoopTester()
    for i in range(1, 4):
        start = time.time()
        try:
            result = tester2.call_sync_v2_original_logic(i)
            elapsed = time.time() - start
            print_result(f"  调用#{i}", True, elapsed, result)
            results.append(("v2", i, True, elapsed))
        except Exception as e:
            elapsed = time.time() - start
            print_result(f"  调用#{i}", False, elapsed, str(e))
            results.append(("v2", i, False, elapsed))

    # 测试方法3: 每次新loop
    print("\n测试方法3: 每次创建新event loop")
    tester3 = EventLoopTester()
    for i in range(1, 4):
        start = time.time()
        try:
            result = tester3.call_sync_v3_new_loop_each_time(i)
            elapsed = time.time() - start
            print_result(f"  调用#{i}", True, elapsed, result)
            results.append(("v3", i, True, elapsed))
        except Exception as e:
            elapsed = time.time() - start
            print_result(f"  调用#{i}", False, elapsed, str(e))
            results.append(("v3", i, False, elapsed))

    # 分析结果
    print("\n结论:")
    v2_failures = [r for r in results if r[0] == "v2" and not r[2]]
    if v2_failures:
        print_error("⚠️  方法2（原始逻辑）有失败 → 存在 Event Loop 状态机 bug")
        return False
    else:
        print(f"{Colors.GREEN}✅ 所有方法都成功 → Event Loop 状态机在无负载下正常{Colors.RESET}")
        return True


# ============================================================================
# 场景2: 测试线程竞争（模拟高负载）
# ============================================================================

def create_thread_load(num_threads=100, duration=10):
    """创建线程负载"""
    print(f"  创建 {num_threads} 个竞争线程，持续 {duration} 秒...")

    def busy_worker():
        end_time = time.time() + duration
        while time.time() < end_time:
            # 模拟CPU和I/O混合负载
            _ = sum(range(1000))
            time.sleep(0.001)

    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=busy_worker, daemon=True)
        t.start()
        threads.append(t)

    return threads


def test_with_thread_competition():
    """测试在线程竞争下的行为"""
    print_section("场景2: 线程竞争测试（高负载）")

    # 创建负载
    print("步骤1: 创建线程竞争环境")
    threads = create_thread_load(num_threads=100, duration=15)
    time.sleep(1)  # 等待负载稳定

    print("\n步骤2: 在高负载下测试 MCP 调用")
    tester = EventLoopTester()
    results = []

    for i in range(1, 4):
        start = time.time()
        try:
            result = tester.call_sync_v2_original_logic(i)
            elapsed = time.time() - start
            print_result(f"  高负载调用#{i}", True, elapsed, result)
            results.append((i, True, elapsed))
        except Exception as e:
            elapsed = time.time() - start
            print_result(f"  高负载调用#{i}", False, elapsed, str(e))
            results.append((i, False, elapsed))

    # 等待负载结束
    print("\n步骤3: 等待负载线程结束...")
    for t in threads[:10]:  # 只等待前10个作为示例
        t.join(timeout=0.1)

    # 分析
    print("\n结论:")
    failures = [r for r in results if not r[1]]
    if failures:
        print_error(f"⚠️  高负载下有 {len(failures)} 次失败 → 线程竞争导致问题")
        return False
    else:
        print(f"{Colors.GREEN}✅ 高负载下全部成功 → 线程竞争不是主要问题{Colors.RESET}")
        return True


# ============================================================================
# 场景3: 真实 MCP 工具测试
# ============================================================================

def test_real_mcp_tool():
    """测试真实的 MCP 工具调用"""
    print_section("场景3: 真实 MCP 工具测试")

    # 检查环境
    script_path = Path(__file__).parent / "search_dataset_tools" / "search_huggingface_math_posttrain.py"
    if not script_path.exists():
        print_warning(f"MCP 脚本不存在: {script_path}")
        return None

    python_bin = Path(__file__).parent / ".venv_gpu2" / "bin" / "python"
    if not python_bin.exists():
        print_warning(f"Python 不存在: {python_bin}")
        return None

    # 构造3次连续调用
    queries = [
        {"query": "math reasoning", "limit": 3},
        {"query": "competition math", "limit": 3},
        {"query": "problem solving", "limit": 3},
    ]

    print("测试连续3次 MCP 调用...")
    results = []

    for i, query_params in enumerate(queries, 1):
        print(f"\n调用#{i}: {query_params}")

        # 构造MCP请求
        requests_text = (
            '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05",'
            '"capabilities":{},"clientInfo":{"name":"test","version":"1.0"}},"id":0}\n'
            f'{{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"search_datasets",'
            f'"arguments":{json.dumps(query_params)}}},"id":{i}}}\n'
        )

        start = time.time()
        try:
            result = subprocess.run(
                [str(python_bin), str(script_path)],
                input=requests_text,
                capture_output=True,
                text=True,
                timeout=10,
            )
            elapsed = time.time() - start

            # 解析响应
            success = False
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if '"result"' in line and '"content"' in line:
                        success = True
                        break

            print_result(f"  MCP调用#{i}", success, elapsed,
                        f"返回码: {result.returncode}" if not success else "")
            results.append((i, success, elapsed))

        except subprocess.TimeoutExpired:
            elapsed = 10.0
            print_result(f"  MCP调用#{i}", False, elapsed, "超时 (10秒)")
            results.append((i, False, elapsed))
        except Exception as e:
            elapsed = time.time() - start
            print_result(f"  MCP调用#{i}", False, elapsed, str(e))
            results.append((i, False, elapsed))

    # 分析
    print("\n结论:")
    failures = [r for r in results if not r[1]]
    if failures:
        print_error(f"⚠️  真实MCP工具有 {len(failures)} 次失败")
        if len(failures) > 1 and failures[0][0] == 2:
            print_error("   → 第2次调用失败，符合 Event Loop 状态机 bug 特征")
        return False
    else:
        print(f"{Colors.GREEN}✅ 真实MCP工具调用全部成功{Colors.RESET}")
        return True


# ============================================================================
# 场景4: 真实 MCP + 线程竞争
# ============================================================================

def test_real_mcp_with_thread_competition():
    """测试真实MCP在线程竞争下的行为"""
    print_section("场景4: 真实 MCP + 线程竞争测试")

    # 创建负载
    print("步骤1: 创建线程竞争环境")
    threads = create_thread_load(num_threads=100, duration=30)
    time.sleep(1)

    print("\n步骤2: 在高负载下测试真实 MCP")
    result = test_real_mcp_tool()

    print("\n步骤3: 等待负载结束...")
    time.sleep(1)

    return result


# ============================================================================
# 主程序
# ============================================================================

def main():
    print(f"\n{Colors.BOLD}MCP 调用死锁诊断工具{Colors.RESET}")
    print(f"目标: 区分 Event Loop 状态机 bug vs 线程竞争问题\n")

    results = {}

    # 场景1: Event loop状态机
    try:
        results['event_loop'] = test_event_loop_state_machine()
    except Exception as e:
        print_error(f"场景1异常: {e}")
        results['event_loop'] = None

    # 场景2: 线程竞争
    try:
        results['thread_competition'] = test_with_thread_competition()
    except Exception as e:
        print_error(f"场景2异常: {e}")
        results['thread_competition'] = None

    # 场景3: 真实MCP
    try:
        results['real_mcp'] = test_real_mcp_tool()
    except Exception as e:
        print_error(f"场景3异常: {e}")
        results['real_mcp'] = None

    # 场景4: 真实MCP + 线程竞争
    try:
        results['real_mcp_competition'] = test_real_mcp_with_thread_competition()
    except Exception as e:
        print_error(f"场景4异常: {e}")
        results['real_mcp_competition'] = None

    # 最终诊断
    print_section("最终诊断报告")

    print("测试结果总结:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else ("❌ FAIL" if result is False else "⚠️  SKIP")
        print(f"  {status} {test_name}")

    print("\n诊断结论:")

    # 诊断逻辑
    if results.get('event_loop') is False:
        print_error("🔴 主要问题: Event Loop 状态机 bug")
        print("   → 即使在无负载下，第2次调用也会失败")
        print("   → 建议修复: evomaster/agent/tools/mcp/mcp.py 第169-174行")
        print("   → 避免 run_until_complete 和 run_coroutine_threadsafe 状态切换")

    elif results.get('thread_competition') is False and results.get('event_loop') is True:
        print_error("🟡 主要问题: 线程竞争导致")
        print("   → 无负载时正常，高负载时失败")
        print("   → 建议: 减少并发任务数，或增加资源限制")

    elif results.get('real_mcp') is False:
        print_error("🟠 问题在真实MCP工具中复现")
        if results.get('real_mcp_competition') is False:
            print("   → 高负载加剧问题")
            print("   → 可能是 Event Loop + 线程竞争 组合原因")
        else:
            print("   → 仅在特定条件下触发")

    else:
        print(f"{Colors.GREEN}🟢 所有测试通过！{Colors.RESET}")
        print("   → 当前环境下未能复现问题")
        print("   → 问题可能与特定机器环境有关")

    print("\n建议的下一步:")
    print("  1. 在出问题的机器上运行此脚本")
    print("  2. 对比两台机器的诊断结果")
    print("  3. 根据诊断结果选择修复方案")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}测试被用户中断{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.RED}测试异常: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
