#!/usr/bin/env python3
"""
MCP资源竞争诊断测试

测试不同资源的竞争情况：
1. HF缓存目录文件锁
2. 网络连接池
3. hf-mirror.com API限制
4. 进程间竞争
"""

import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple


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


def print_result(success, elapsed, details=""):
    status = f"{Colors.GREEN}✅ PASS" if success else f"{Colors.RED}❌ FAIL"
    print(f"{status}{Colors.RESET}: {elapsed:.2f}s")
    if details:
        print(f"    {Colors.YELLOW}{details}{Colors.RESET}")


# ============================================================================
# 辅助函数：调用MCP工具
# ============================================================================

def call_mcp_tool(query: str, limit: int = 3, timeout: int = 15,
                  cache_dir: str = None, python_bin: str = None) -> Tuple[bool, float, str]:
    """
    调用MCP工具

    Returns:
        (success, elapsed_time, message)
    """
    if python_bin is None:
        python_bin = str(Path(__file__).parent / ".venv_gpu2" / "bin" / "python")

    script_path = Path(__file__).parent / "search_dataset_tools" / "search_huggingface_math_posttrain.py"

    # 构造MCP请求
    requests_text = (
        '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05",'
        '"capabilities":{},"clientInfo":{"name":"test","version":"1.0"}},"id":0}\n'
        f'{{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"search_datasets",'
        f'"arguments":{{"query":"{query}","limit":{limit}}}}},"id":1}}\n'
    )

    # 设置环境变量
    env = os.environ.copy()
    if cache_dir:
        env['HF_HOME'] = cache_dir
        env['HF_DATASETS_CACHE'] = f"{cache_dir}/datasets"
        env['HUGGINGFACE_HUB_CACHE'] = f"{cache_dir}/hub"

    start = time.time()
    try:
        result = subprocess.run(
            [python_bin, str(script_path)],
            input=requests_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        elapsed = time.time() - start

        # 检查成功
        success = False
        if result.returncode == 0 and result.stdout:
            for line in result.stdout.strip().split('\n'):
                if '"result"' in line and '"content"' in line:
                    success = True
                    break

        msg = "OK" if success else f"returncode={result.returncode}"
        return success, elapsed, msg

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return False, elapsed, "TIMEOUT"
    except Exception as e:
        elapsed = time.time() - start
        return False, elapsed, str(e)


# ============================================================================
# 场景1: 单进程串行调用（基线测试）
# ============================================================================

def test_baseline_serial():
    """基线测试：单进程串行调用，无竞争"""
    print_section("场景1: 基线测试 - 单进程串行调用（无竞争）")

    queries = ["math reasoning", "competition math", "problem solving"]

    print("测试3次连续调用...")
    results = []
    for i, query in enumerate(queries, 1):
        print(f"\n  调用#{i}: {query}")
        success, elapsed, msg = call_mcp_tool(query, limit=3, timeout=15)
        print_result(success, elapsed, msg)
        results.append((i, success, elapsed))

        if i < len(queries):
            time.sleep(0.5)  # 短暂间隔

    # 分析
    print("\n结论:")
    failures = [r for r in results if not r[1]]
    if failures:
        print(f"{Colors.RED}❌ 基线测试失败 {len(failures)} 次{Colors.RESET}")
        if failures[0][0] == 2:
            print(f"{Colors.RED}   → 第2次调用失败，即使无竞争也有问题{Colors.RESET}")
        return False, results
    else:
        avg_time = sum(r[2] for r in results) / len(results)
        print(f"{Colors.GREEN}✅ 基线正常，平均耗时 {avg_time:.2f}s{Colors.RESET}")
        return True, results


# ============================================================================
# 场景2: 共享缓存 vs 独立缓存
# ============================================================================

def test_cache_contention():
    """测试HF缓存目录竞争"""
    print_section("场景2: HF缓存目录竞争测试")

    # 获取默认缓存目录
    default_cache = os.getenv('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))

    # 创建临时独立缓存
    temp_cache = tempfile.mkdtemp(prefix='hf_cache_test_')

    print(f"默认缓存: {default_cache}")
    print(f"临时缓存: {temp_cache}")

    # 测试1: 使用默认共享缓存
    print("\n测试1: 使用共享缓存（可能有竞争）")
    shared_results = []
    for i in range(1, 4):
        query = f"test shared {i}"
        print(f"  调用#{i}: {query}")
        success, elapsed, msg = call_mcp_tool(query, limit=2, timeout=15, cache_dir=None)
        print_result(success, elapsed, msg)
        shared_results.append((success, elapsed))
        time.sleep(0.3)

    # 测试2: 使用独立缓存
    print("\n测试2: 使用独立缓存（无竞争）")
    isolated_results = []
    for i in range(1, 4):
        query = f"test isolated {i}"
        print(f"  调用#{i}: {query}")
        success, elapsed, msg = call_mcp_tool(query, limit=2, timeout=15, cache_dir=temp_cache)
        print_result(success, elapsed, msg)
        isolated_results.append((success, elapsed))
        time.sleep(0.3)

    # 清理
    try:
        shutil.rmtree(temp_cache)
    except:
        pass

    # 分析
    print("\n结论:")
    shared_fail = sum(1 for s, _ in shared_results if not s)
    isolated_fail = sum(1 for s, _ in isolated_results if not s)
    shared_avg = sum(e for s, e in shared_results if s) / max(1, len(shared_results) - shared_fail)
    isolated_avg = sum(e for s, e in isolated_results if s) / max(1, len(isolated_results) - isolated_fail)

    print(f"  共享缓存: 失败{shared_fail}次, 平均{shared_avg:.2f}s")
    print(f"  独立缓存: 失败{isolated_fail}次, 平均{isolated_avg:.2f}s")

    if shared_fail > isolated_fail or shared_avg > isolated_avg * 1.5:
        print(f"{Colors.RED}❌ 共享缓存有明显性能下降 → 缓存文件锁竞争{Colors.RESET}")
        return False
    else:
        print(f"{Colors.GREEN}✅ 缓存竞争不是主要问题{Colors.RESET}")
        return True


# ============================================================================
# 场景3: 并发进程测试
# ============================================================================

def worker_concurrent_calls(worker_id: int, num_calls: int, cache_dir: str, result_queue):
    """Worker进程：执行多次MCP调用"""
    results = []
    for i in range(num_calls):
        query = f"worker{worker_id}_call{i}"
        success, elapsed, msg = call_mcp_tool(query, limit=2, timeout=15, cache_dir=cache_dir)
        results.append((worker_id, i+1, success, elapsed, msg))
        time.sleep(0.2)

    result_queue.put((worker_id, results))


def test_concurrent_processes():
    """测试多进程并发调用"""
    print_section("场景3: 多进程并发调用测试（模拟真实负载）")

    num_workers = 5  # 模拟5个并发任务
    calls_per_worker = 3

    # 创建独立缓存避免文件锁干扰
    temp_caches = [tempfile.mkdtemp(prefix=f'hf_cache_w{i}_') for i in range(num_workers)]

    print(f"启动 {num_workers} 个并发进程，每个调用 {calls_per_worker} 次...")

    result_queue = multiprocessing.Queue()
    processes = []

    start = time.time()
    for i in range(num_workers):
        p = multiprocessing.Process(
            target=worker_concurrent_calls,
            args=(i, calls_per_worker, temp_caches[i], result_queue)
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)  # 错开启动时间

    # 等待完成
    for p in processes:
        p.join(timeout=60)
        if p.is_alive():
            print(f"{Colors.RED}⚠️  进程超时，强制终止{Colors.RESET}")
            p.terminate()
            p.join()

    total_elapsed = time.time() - start

    # 收集结果
    all_results = []
    while not result_queue.empty():
        worker_id, worker_results = result_queue.get()
        all_results.extend(worker_results)

    # 清理缓存
    for cache_dir in temp_caches:
        try:
            shutil.rmtree(cache_dir)
        except:
            pass

    # 分析
    print(f"\n总耗时: {total_elapsed:.2f}s")
    print(f"收到结果: {len(all_results)} / {num_workers * calls_per_worker}")

    # 按worker分组
    by_worker = {}
    for worker_id, call_num, success, elapsed, msg in all_results:
        if worker_id not in by_worker:
            by_worker[worker_id] = []
        by_worker[worker_id].append((call_num, success, elapsed, msg))

    print("\n各Worker结果:")
    for worker_id in sorted(by_worker.keys()):
        results = by_worker[worker_id]
        failures = [r for r in results if not r[1]]
        avg_time = sum(r[2] for r in results if r[1]) / max(1, len(results) - len(failures))
        status = f"{Colors.GREEN}✅" if not failures else f"{Colors.RED}❌"
        print(f"  {status} Worker#{worker_id}: {len(failures)}失败, 平均{avg_time:.2f}s{Colors.RESET}")

        # 检查第2次调用
        second_calls = [r for r in results if r[0] == 2]
        if second_calls and not second_calls[0][1]:
            print(f"    {Colors.RED}⚠️  第2次调用失败{Colors.RESET}")

    # 结论
    print("\n结论:")
    total_failures = sum(1 for _, _, s, _, _ in all_results if not s)
    if total_failures > num_workers * 0.3:  # 超过30%失败率
        print(f"{Colors.RED}❌ 并发调用失败率高 ({total_failures}/{len(all_results)}) → 系统负载问题{Colors.RESET}")
        return False
    else:
        print(f"{Colors.GREEN}✅ 并发调用基本正常 (失败{total_failures}/{len(all_results)}){Colors.RESET}")
        return True


# ============================================================================
# 场景4: 模拟10个MCP Server同时存在
# ============================================================================

def keep_mcp_server_alive(worker_id: int, duration: int, cache_dir: str):
    """保持一个MCP Server进程运行"""
    script_path = Path(__file__).parent / "search_dataset_tools" / "search_huggingface_math_posttrain.py"
    python_bin = str(Path(__file__).parent / ".venv_gpu2" / "bin" / "python")

    env = os.environ.copy()
    env['HF_HOME'] = cache_dir

    # 启动MCP Server但不发送请求，让它等待
    proc = subprocess.Popen(
        [python_bin, str(script_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    # 发送初始化请求
    init_req = '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}},"id":0}\n'
    try:
        proc.stdin.write(init_req.encode())
        proc.stdin.flush()
    except:
        pass

    # 保持运行
    time.sleep(duration)

    # 清理
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except:
        proc.kill()


def test_with_background_mcp_servers():
    """测试在后台有多个MCP Server运行时的情况"""
    print_section("场景4: 模拟10个MCP Server运行（真实负载模拟）")

    num_bg_servers = 10
    test_duration = 20  # 秒

    # 为后台服务器创建缓存
    bg_caches = [tempfile.mkdtemp(prefix=f'hf_bg{i}_') for i in range(num_bg_servers)]

    print(f"启动 {num_bg_servers} 个后台MCP Server进程...")
    bg_processes = []
    for i in range(num_bg_servers):
        p = multiprocessing.Process(
            target=keep_mcp_server_alive,
            args=(i, test_duration, bg_caches[i])
        )
        p.start()
        bg_processes.append(p)
        time.sleep(0.05)

    time.sleep(2)  # 等待后台进程启动完成

    print(f"\n后台环境已建立，开始测试前台调用...")

    # 使用独立缓存进行前台测试
    fg_cache = tempfile.mkdtemp(prefix='hf_fg_')

    queries = ["foreground test 1", "foreground test 2", "foreground test 3"]
    results = []

    for i, query in enumerate(queries, 1):
        print(f"\n  前台调用#{i}: {query}")
        success, elapsed, msg = call_mcp_tool(query, limit=2, timeout=15, cache_dir=fg_cache)
        print_result(success, elapsed, msg)
        results.append((i, success, elapsed))
        time.sleep(0.5)

    # 清理后台进程
    print("\n清理后台进程...")
    for p in bg_processes:
        if p.is_alive():
            p.terminate()
        p.join(timeout=1)
        if p.is_alive():
            p.kill()

    # 清理缓存
    for cache_dir in bg_caches + [fg_cache]:
        try:
            shutil.rmtree(cache_dir)
        except:
            pass

    # 分析
    print("\n结论:")
    failures = [r for r in results if not r[1]]
    if failures:
        print(f"{Colors.RED}❌ 后台负载下有 {len(failures)} 次失败{Colors.RESET}")
        if any(r[0] == 2 for r in failures):
            print(f"{Colors.RED}   → 第2次调用失败，后台MCP Server竞争是主要原因{Colors.RESET}")
        return False
    else:
        avg_time = sum(r[2] for r in results) / len(results)
        print(f"{Colors.GREEN}✅ 后台负载下正常，平均耗时 {avg_time:.2f}s{Colors.RESET}")
        return True


# ============================================================================
# 场景5: 检查当前系统MCP Server状态
# ============================================================================

def test_current_system_state():
    """检查当前系统上的MCP Server状态"""
    print_section("场景5: 当前系统状态检查")

    # 检查运行中的MCP Server
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [l for l in result.stdout.split('\n') if 'search_huggingface' in l and 'grep' not in l]

        print(f"当前运行的MCP Server进程数: {len(lines)}")

        if lines:
            print("\n进程详情:")
            for i, line in enumerate(lines[:5], 1):  # 只显示前5个
                parts = line.split()
                if len(parts) >= 11:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    time = parts[9]
                    print(f"  {i}. PID={pid}, CPU={cpu}%, MEM={mem}%, TIME={time}")

            if len(lines) > 5:
                print(f"  ... (还有 {len(lines) - 5} 个进程)")

        # 检查线程数
        if lines:
            first_pid = lines[0].split()[1]
            thread_result = subprocess.run(
                ["ps", "-p", first_pid, "-T"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            thread_count = len(thread_result.stdout.strip().split('\n')) - 1
            print(f"\n示例进程(PID {first_pid})的线程数: {thread_count}")

            if thread_count > 50:
                print(f"{Colors.YELLOW}⚠️  线程数异常高 (正常应该<20){Colors.RESET}")

        # 给出建议
        print("\n建议:")
        if len(lines) >= 10:
            print(f"{Colors.YELLOW}⚠️  MCP Server进程数较多 ({len(lines)}个){Colors.RESET}")
            print("   → 可能需要清理残留进程: pkill -f search_huggingface")
        elif len(lines) >= 5:
            print(f"{Colors.YELLOW}⚠️  MCP Server进程数偏多 ({len(lines)}个){Colors.RESET}")
            print("   → 可以考虑减少并发任务")
        else:
            print(f"{Colors.GREEN}✅ MCP Server进程数正常 ({len(lines)}个){Colors.RESET}")

        return len(lines)

    except Exception as e:
        print(f"{Colors.RED}检查失败: {e}{Colors.RESET}")
        return None


# ============================================================================
# 主程序
# ============================================================================

def main():
    print(f"\n{Colors.BOLD}MCP 资源竞争诊断测试{Colors.RESET}")
    print(f"目标: 识别导致第2次调用卡住的具体资源\n")

    results = {}

    # 场景5: 先检查当前系统状态
    try:
        current_mcp_count = test_current_system_state()
        results['current_state'] = current_mcp_count
    except Exception as e:
        print(f"{Colors.RED}场景5异常: {e}{Colors.RESET}")
        results['current_state'] = None

    # 场景1: 基线测试
    try:
        baseline_ok, baseline_data = test_baseline_serial()
        results['baseline'] = baseline_ok
    except Exception as e:
        print(f"{Colors.RED}场景1异常: {e}{Colors.RESET}")
        results['baseline'] = None

    # 场景2: 缓存竞争
    try:
        results['cache'] = test_cache_contention()
    except Exception as e:
        print(f"{Colors.RED}场景2异常: {e}{Colors.RESET}")
        results['cache'] = None

    # 场景3: 并发进程
    try:
        results['concurrent'] = test_concurrent_processes()
    except Exception as e:
        print(f"{Colors.RED}场景3异常: {e}{Colors.RESET}")
        results['concurrent'] = None

    # 场景4: 后台MCP Server
    try:
        results['background'] = test_with_background_mcp_servers()
    except Exception as e:
        print(f"{Colors.RED}场景4异常: {e}{Colors.RESET}")
        results['background'] = None

    # 最终诊断
    print_section("最终诊断报告")

    print("测试结果:")
    print(f"  系统MCP Server数量: {results.get('current_state', 'N/A')}")
    print(f"  基线测试: {'✅ PASS' if results.get('baseline') else '❌ FAIL'}")
    print(f"  缓存竞争: {'✅ PASS' if results.get('cache') else '❌ FAIL'}")
    print(f"  并发进程: {'✅ PASS' if results.get('concurrent') else '❌ FAIL'}")
    print(f"  后台负载: {'✅ PASS' if results.get('background') else '❌ FAIL'}")

    print("\n根本原因诊断:")

    if results.get('baseline') is False:
        print(f"{Colors.RED}🔴 代码/网络层面问题{Colors.RESET}")
        print("   → 即使无竞争也失败")
        print("   → 检查网络连接、HF API状态、代码bug")

    elif results.get('cache') is False:
        print(f"{Colors.RED}🟠 HF缓存文件锁竞争{Colors.RESET}")
        print("   → 多个进程共享缓存目录导致")
        print("   → 解决方案: 为每个任务配置独立缓存目录")

    elif results.get('background') is False and results.get('concurrent') is True:
        print(f"{Colors.RED}🟡 后台MCP Server进程数过多{Colors.RESET}")
        print("   → 当前系统运行了过多MCP Server")
        print("   → 解决方案: 清理残留进程，或限制并发任务数")
        print(f"   → 命令: pkill -f search_huggingface")

    elif results.get('concurrent') is False:
        print(f"{Colors.RED}🟡 系统负载/资源耗尽{Colors.RESET}")
        print("   → 线程/进程竞争激烈")
        print("   → 解决方案: 减少并发任务，或增加系统资源")

    else:
        print(f"{Colors.GREEN}🟢 测试环境下未复现问题{Colors.RESET}")
        print("   → 可能是间歇性问题或特定条件触发")
        print("   → 建议在问题机器上运行此测试")

    print("\n推荐操作:")
    if results.get('current_state', 0) and results.get('current_state', 0) >= 5:
        print(f"  1. {Colors.YELLOW}清理当前MCP Server进程{Colors.RESET}")
        print("     pkill -f search_huggingface")
    print(f"  2. {Colors.YELLOW}为每个任务配置独立HF缓存{Colors.RESET}")
    print("     export HF_HOME=/tmp/hf_cache_$$")
    print(f"  3. {Colors.YELLOW}限制并发任务数{Colors.RESET}")
    print("     同时运行的任务不超过3-5个")


if __name__ == "__main__":
    # 设置multiprocessing启动方式
    multiprocessing.set_start_method('spawn', force=True)

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
