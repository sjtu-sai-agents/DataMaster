#!/usr/bin/env python3
"""
精确诊断：HF-mirror限流 vs HF缓存竞争

测试设计：
1. 禁用缓存，纯测网络（HF-mirror限流）
2. 启用缓存，测试缓存读写（文件锁竞争）
3. 对比第1次vs第2次调用的差异
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Tuple, Optional


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_result(success, elapsed, details=""):
    status = f"{Colors.GREEN}✅" if success else f"{Colors.RED}❌"
    print(f"{status} {elapsed:.2f}s{Colors.RESET} {details}")


# ============================================================================
# 辅助：调用MCP工具
# ============================================================================

def call_mcp_with_env(
    query: str,
    limit: int = 3,
    timeout: int = 30,
    cache_dir: Optional[str] = None,
    disable_cache: bool = False,
    verbose: bool = False
) -> Tuple[bool, float, str, str]:
    """
    调用MCP工具

    Returns:
        (success, elapsed_time, message, stderr)
    """
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

    if disable_cache:
        # 禁用HF缓存的各种方法
        env['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
        env['HF_HUB_OFFLINE'] = '0'
        # 使用每次不同的缓存目录强制走网络
        temp_nocache = tempfile.mkdtemp(prefix='nocache_')
        env['HF_HOME'] = temp_nocache
        env['HF_DATASETS_CACHE'] = f"{temp_nocache}/datasets"

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

        if verbose and not success:
            stderr_preview = result.stderr[:200] if result.stderr else ""
        else:
            stderr_preview = ""

        # 清理临时缓存
        if disable_cache:
            try:
                shutil.rmtree(temp_nocache)
            except:
                pass

        return success, elapsed, msg, stderr_preview

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return False, elapsed, "TIMEOUT", ""


# ============================================================================
# 测试1: 禁用缓存，纯测网络（HF-mirror限流）
# ============================================================================

def test_network_only_no_cache():
    """禁用缓存，强制每次走网络，测试HF-mirror限流"""
    print_section("测试1: 禁用缓存 - 纯网络请求（检测HF-mirror限流）")

    print("每次调用使用不同临时缓存目录，强制走网络...")
    print("如果第2次明显变慢 → HF-mirror限流")
    print()

    queries = [
        ("network test alpha", "首次网络请求"),
        ("network test beta", "第2次网络请求（可能触发限流）"),
        ("network test gamma", "第3次网络请求"),
    ]

    results = []
    for i, (query, desc) in enumerate(queries, 1):
        print(f"调用#{i} ({desc}): {query}")
        success, elapsed, msg, stderr = call_mcp_with_env(
            query, limit=2, timeout=20, disable_cache=True
        )
        print_result(success, elapsed, msg)
        results.append((i, success, elapsed))

        if i < len(queries):
            time.sleep(1)  # 短暂间隔

    # 分析
    print("\n分析:")
    times = [e for _, s, e in results if s]

    if len(times) >= 2:
        first_time = times[0]
        second_time = times[1]
        slowdown = second_time / first_time if first_time > 0 else 0

        print(f"  第1次耗时: {first_time:.2f}s")
        print(f"  第2次耗时: {second_time:.2f}s")
        print(f"  减速比: {slowdown:.2f}x")

        if slowdown > 2.0:
            print(f"{Colors.RED}❌ 第2次明显变慢 → HF-mirror有限流{Colors.RESET}")
            return "rate_limited", results
        elif slowdown > 1.3:
            print(f"{Colors.YELLOW}⚠️  第2次略慢 → HF-mirror可能有轻度限流{Colors.RESET}")
            return "maybe_rate_limited", results
        else:
            print(f"{Colors.GREEN}✅ 时间稳定 → HF-mirror无明显限流{Colors.RESET}")
            return "no_rate_limit", results
    else:
        print(f"{Colors.RED}❌ 成功调用不足2次，无法判断{Colors.RESET}")
        return "insufficient_data", results


# ============================================================================
# 测试2: 启用缓存，测试缓存读写性能
# ============================================================================

def test_cache_read_write():
    """测试缓存读写性能，检测文件锁竞争"""
    print_section("测试2: 启用缓存 - 测试缓存读写（检测文件锁竞争）")

    cache_dir = tempfile.mkdtemp(prefix='hf_cache_test_')
    print(f"使用缓存目录: {cache_dir}")
    print("第1次调用写缓存，第2次调用读缓存")
    print("如果第2次反而更慢 → 缓存文件锁竞争")
    print()

    # 使用相同的查询，第2次应该读缓存
    query = "cache test query"

    # 第1次：写缓存（网络请求）
    print(f"调用#1 (写缓存): {query}")
    s1, t1, msg1, _ = call_mcp_with_env(query, limit=3, timeout=20, cache_dir=cache_dir)
    print_result(s1, t1, msg1)

    time.sleep(1)

    # 第2次：读缓存（应该更快）
    print(f"\n调用#2 (读缓存): {query}")
    s2, t2, msg2, _ = call_mcp_with_env(query, limit=3, timeout=20, cache_dir=cache_dir)
    print_result(s2, t2, msg2)

    time.sleep(1)

    # 第3次：再次读缓存
    print(f"\n调用#3 (读缓存): {query}")
    s3, t3, msg3, _ = call_mcp_with_env(query, limit=3, timeout=20, cache_dir=cache_dir)
    print_result(s3, t3, msg3)

    # 清理
    try:
        shutil.rmtree(cache_dir)
    except:
        pass

    # 分析
    print("\n分析:")
    if s1 and s2:
        speedup = t1 / t2 if t2 > 0 else 0
        print(f"  第1次(写缓存): {t1:.2f}s")
        print(f"  第2次(读缓存): {t2:.2f}s")
        print(f"  加速比: {speedup:.2f}x")

        if t2 > t1 * 1.2:
            print(f"{Colors.RED}❌ 读缓存反而更慢 → 缓存访问有问题{Colors.RESET}")
            return "cache_slower", [(1, s1, t1), (2, s2, t2), (3, s3, t3)]
        elif speedup > 1.5:
            print(f"{Colors.GREEN}✅ 读缓存明显加速 → 缓存工作正常{Colors.RESET}")
            return "cache_working", [(1, s1, t1), (2, s2, t2), (3, s3, t3)]
        else:
            print(f"{Colors.CYAN}ℹ️  缓存效果不明显 → 可能没有用到缓存{Colors.RESET}")
            return "cache_unclear", [(1, s1, t1), (2, s2, t2), (3, s3, t3)]
    else:
        print(f"{Colors.RED}❌ 调用失败，无法判断{Colors.RESET}")
        return "failed", [(1, s1, t1), (2, s2, t2), (3, s3, t3)]


# ============================================================================
# 测试3: 共享缓存下的并发竞争
# ============================================================================

def test_shared_cache_concurrent():
    """测试多个进程共享同一缓存时的文件锁竞争"""
    print_section("测试3: 共享缓存并发 - 检测文件锁竞争")

    shared_cache = tempfile.mkdtemp(prefix='hf_shared_cache_')
    print(f"共享缓存目录: {shared_cache}")
    print("启动5个并发调用，共享同一缓存")
    print("如果大量超时/失败 → 缓存文件锁竞争严重")
    print()

    import multiprocessing

    def worker(worker_id, cache_dir, result_queue):
        query = f"shared cache worker {worker_id}"
        success, elapsed, msg, _ = call_mcp_with_env(
            query, limit=2, timeout=20, cache_dir=cache_dir
        )
        result_queue.put((worker_id, success, elapsed, msg))

    result_queue = multiprocessing.Queue()
    processes = []

    start = time.time()
    for i in range(5):
        p = multiprocessing.Process(
            target=worker,
            args=(i, shared_cache, result_queue)
        )
        p.start()
        processes.append(p)

    # 等待完成
    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join()

    total_time = time.time() - start

    # 收集结果
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # 清理
    try:
        shutil.rmtree(shared_cache)
    except:
        pass

    # 分析
    print("结果:")
    for worker_id, success, elapsed, msg in sorted(results):
        print_result(success, elapsed, f"Worker#{worker_id}: {msg}")

    print(f"\n总耗时: {total_time:.2f}s")
    success_count = sum(1 for _, s, _, _ in results if s)
    print(f"成功/总数: {success_count}/{len(results)}")

    if success_count < 3:
        print(f"{Colors.RED}❌ 大量失败 → 共享缓存文件锁竞争严重{Colors.RESET}")
        return "severe_contention", results
    elif success_count == 5 and total_time < 15:
        print(f"{Colors.GREEN}✅ 全部成功且快速 → 共享缓存工作良好{Colors.RESET}")
        return "no_contention", results
    else:
        print(f"{Colors.YELLOW}⚠️  部分成功或较慢 → 有一定竞争{Colors.RESET}")
        return "moderate_contention", results


# ============================================================================
# 测试4: 检查HF API响应延迟
# ============================================================================

def test_hf_api_latency():
    """测试HF API本身的响应速度"""
    print_section("测试4: HF API延迟测试")

    print("测试hf-mirror.com的/api/datasets端点响应速度")
    print()

    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    hf_token = os.getenv("HF_TOKEN", "")

    session = requests.Session()
    retry = Retry(total=0)  # 不重试，直接测延迟
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    headers = {"Accept": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    # 测试3次
    results = []
    for i in range(1, 4):
        query = f"api test {i}"
        url = f"{hf_endpoint}/api/datasets"
        params = {"search": query, "limit": 3}

        print(f"API请求#{i}: {query}")
        start = time.time()
        try:
            resp = session.get(url, headers=headers, params=params, timeout=15)
            elapsed = time.time() - start
            success = resp.status_code == 200

            msg = f"HTTP {resp.status_code}"
            if resp.status_code == 429:
                msg += " (Rate Limited!)"

            print_result(success, elapsed, msg)
            results.append((i, success, elapsed, resp.status_code))

        except requests.exceptions.Timeout:
            elapsed = time.time() - start
            print_result(False, elapsed, "TIMEOUT")
            results.append((i, False, elapsed, 0))
        except Exception as e:
            elapsed = time.time() - start
            print_result(False, elapsed, str(e)[:50])
            results.append((i, False, elapsed, 0))

        time.sleep(1)

    # 分析
    print("\n分析:")
    status_codes = [code for _, s, _, code in results if s]
    times = [t for _, s, t, _ in results if s]

    if 429 in status_codes:
        print(f"{Colors.RED}❌ 检测到429错误 → HF API明确限流{Colors.RESET}")
        return "rate_limited_429", results
    elif len(times) >= 2 and times[1] > times[0] * 2:
        print(f"{Colors.YELLOW}⚠️  后续请求变慢 → 可能有隐式限流{Colors.RESET}")
        return "possible_throttling", results
    elif times and sum(times)/len(times) > 5:
        print(f"{Colors.YELLOW}⚠️  平均延迟>5秒 → API响应慢{Colors.RESET}")
        return "slow_api", results
    elif times:
        avg = sum(times)/len(times)
        print(f"{Colors.GREEN}✅ API响应正常，平均 {avg:.2f}s{Colors.RESET}")
        return "normal", results
    else:
        print(f"{Colors.RED}❌ 无成功请求{Colors.RESET}")
        return "failed", results


# ============================================================================
# 主程序
# ============================================================================

def main():
    print(f"\n{Colors.BOLD}MCP 精确诊断：HF-mirror限流 vs HF缓存竞争{Colors.RESET}\n")

    # 设置multiprocessing
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    results = {}

    # 测试4: 先测API本身
    try:
        api_result, api_data = test_hf_api_latency()
        results['api_latency'] = api_result
    except Exception as e:
        print(f"{Colors.RED}测试4异常: {e}{Colors.RESET}")
        results['api_latency'] = 'error'

    # 测试1: 网络限流
    try:
        network_result, network_data = test_network_only_no_cache()
        results['network'] = network_result
    except Exception as e:
        print(f"{Colors.RED}测试1异常: {e}{Colors.RESET}")
        results['network'] = 'error'

    # 测试2: 缓存性能
    try:
        cache_result, cache_data = test_cache_read_write()
        results['cache'] = cache_result
    except Exception as e:
        print(f"{Colors.RED}测试2异常: {e}{Colors.RESET}")
        results['cache'] = 'error'

    # 测试3: 并发竞争
    try:
        concurrent_result, concurrent_data = test_shared_cache_concurrent()
        results['concurrent'] = concurrent_result
    except Exception as e:
        print(f"{Colors.RED}测试3异常: {e}{Colors.RESET}")
        results['concurrent'] = 'error'

    # 最终诊断
    print_section("最终诊断结果")

    print("测试结果总结:")
    print(f"  API延迟测试: {results.get('api_latency', 'N/A')}")
    print(f"  网络限流测试: {results.get('network', 'N/A')}")
    print(f"  缓存性能测试: {results.get('cache', 'N/A')}")
    print(f"  并发竞争测试: {results.get('concurrent', 'N/A')}")

    print("\n" + "="*70)
    print(f"{Colors.BOLD}根本原因判断:{Colors.RESET}\n")

    # 判断逻辑
    api_status = results.get('api_latency', '')
    network_status = results.get('network', '')
    cache_status = results.get('cache', '')
    concurrent_status = results.get('concurrent', '')

    if api_status == 'rate_limited_429':
        print(f"{Colors.RED}🔴 HF-mirror API明确限流 (429错误){Colors.RESET}")
        print("   → 检测到HTTP 429响应")
        print("   → 解决方案:")
        print("     1. 减少请求频率")
        print("     2. 使用官方HF Hub (需要代理)")
        print("     3. 在请求间增加延迟 (sleep)")

    elif network_status in ('rate_limited', 'maybe_rate_limited'):
        print(f"{Colors.RED}🟠 HF-mirror隐式限流{Colors.RESET}")
        print("   → 第2次网络请求明显变慢")
        print("   → 虽然没有429错误，但响应被延迟")
        print("   → 解决方案:")
        print("     1. 在连续请求间增加1-2秒延迟")
        print("     2. 启用缓存，避免重复请求")

    elif concurrent_status == 'severe_contention':
        print(f"{Colors.RED}🟡 HF缓存文件锁竞争严重{Colors.RESET}")
        print("   → 多进程共享缓存时大量失败")
        print("   → 解决方案:")
        print("     1. 为每个任务配置独立缓存:")
        print("        export HF_HOME=/tmp/hf_cache_task_$$")
        print("     2. 减少并发任务数")

    elif cache_status == 'cache_slower':
        print(f"{Colors.RED}🟡 缓存读取比网络请求更慢{Colors.RESET}")
        print("   → 可能是缓存目录I/O瓶颈")
        print("   → 解决方案:")
        print("     1. 检查缓存目录所在磁盘")
        print("     2. 使用SSD或内存文件系统 (/dev/shm)")

    elif api_status == 'slow_api':
        print(f"{Colors.YELLOW}🟤 HF-mirror API本身响应慢{Colors.RESET}")
        print("   → 平均延迟>5秒")
        print("   → 可能原因: 网络、服务器负载")
        print("   → 解决方案:")
        print("     1. 增加超时时间")
        print("     2. 考虑更换mirror")

    elif network_status == 'no_rate_limit' and cache_status == 'cache_working':
        print(f"{Colors.GREEN}🟢 MCP工具本身工作正常{Colors.RESET}")
        print("   → 网络无限流，缓存正常")
        print("   → 问题可能在:")
        print("     1. 当前系统的10个后台MCP Server竞争")
        print("     2. 系统整体负载过高 (810线程)")
        print("   → 解决方案:")
        print("     pkill -f search_huggingface  # 清理后台进程")

    else:
        print(f"{Colors.CYAN}ℹ️  需要更多信息{Colors.RESET}")
        print("   → 测试结果不够明确")
        print("   → 建议:")
        print("     1. 清理后台MCP Server后重新测试")
        print("     2. 在低负载时段重新测试")
        print("     3. 检查网络连接")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}测试被用户中断{Colors.RESET}")
    except Exception as e:
        print(f"\n\n{Colors.RED}测试异常: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
