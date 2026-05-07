#!/usr/bin/env python3
"""
MCP Server长连接测试 - 模拟真实Agent场景

目标：
1. 启动一个长期运行的MCP Server
2. 通过stdin/stdout管道连续发送请求
3. 精确定位哪次请求、哪个环节超时
"""

import json
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, List


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_step(step_name, status="INFO"):
    color = {
        "INFO": Colors.CYAN,
        "SUCCESS": Colors.GREEN,
        "ERROR": Colors.RED,
        "WARNING": Colors.YELLOW,
    }.get(status, Colors.CYAN)
    print(f"{color}[{status}]{Colors.RESET} {step_name}")


def print_timeline(events: List[Tuple[float, str]]):
    """打印时间线"""
    if not events:
        return

    start_time = events[0][0]
    print(f"\n{Colors.BOLD}时间线:{Colors.RESET}")
    for timestamp, event in events:
        elapsed = timestamp - start_time
        print(f"  {elapsed:6.2f}s | {event}")


class MCPServerTester:
    """MCP Server长连接测试器"""

    def __init__(self, cache_dir: Optional[str] = None, timeout: int = 30):
        self.python_bin = str(Path(__file__).parent / ".venv_gpu2" / "bin" / "python")
        self.script_path = Path(__file__).parent / "search_dataset_tools" / "search_huggingface_math_posttrain.py"
        self.cache_dir = cache_dir
        self.timeout = timeout

        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self.timeline: List[Tuple[float, str]] = []

    def start_server(self) -> bool:
        """启动MCP Server进程"""
        print_step("启动MCP Server进程...", "INFO")
        self.timeline.append((time.time(), "启动MCP Server"))

        env = os.environ.copy()
        if self.cache_dir:
            env['HF_HOME'] = self.cache_dir
            env['HF_DATASETS_CACHE'] = f"{self.cache_dir}/datasets"

        try:
            self.process = subprocess.Popen(
                [self.python_bin, str(self.script_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                bufsize=0,  # 无缓冲
            )

            time.sleep(0.5)  # 等待进程启动

            if self.process.poll() is not None:
                print_step("MCP Server进程启动后立即退出", "ERROR")
                stderr = self.process.stderr.read().decode('utf-8', errors='ignore')
                print(f"  Stderr: {stderr[:500]}")
                return False

            print_step(f"MCP Server进程已启动 (PID: {self.process.pid})", "SUCCESS")
            self.timeline.append((time.time(), f"进程启动成功 PID={self.process.pid}"))
            return True

        except Exception as e:
            print_step(f"启动失败: {e}", "ERROR")
            return False

    def send_request(self, request: dict) -> Tuple[bool, Optional[dict], float, str]:
        """
        发送请求并等待响应

        Returns:
            (success, response, elapsed_time, error_message)
        """
        if not self.process or self.process.poll() is not None:
            return False, None, 0.0, "MCP Server进程未运行"

        request_json = json.dumps(request) + "\n"
        start_time = time.time()

        try:
            # 1. 发送请求
            self.timeline.append((time.time(), f"发送请求 method={request.get('method')} id={request.get('id')}"))
            self.process.stdin.write(request_json.encode('utf-8'))
            self.process.stdin.flush()
            send_time = time.time() - start_time
            self.timeline.append((time.time(), f"请求已发送 ({send_time:.3f}s)"))

            # 2. 等待响应（使用select实现超时）
            self.timeline.append((time.time(), f"等待响应 (timeout={self.timeout}s)"))
            response_line = self._read_response_with_timeout(self.timeout)

            if response_line is None:
                elapsed = time.time() - start_time
                self.timeline.append((time.time(), f"❌ 响应超时 ({elapsed:.2f}s)"))

                # 检查进程状态
                if self.process.poll() is not None:
                    return False, None, elapsed, "MCP Server进程已退出"
                else:
                    return False, None, elapsed, f"等待响应超时 ({self.timeout}s)"

            elapsed = time.time() - start_time
            self.timeline.append((time.time(), f"收到响应 ({elapsed:.3f}s)"))

            # 3. 解析响应
            try:
                response = json.loads(response_line)

                if "error" in response:
                    self.timeline.append((time.time(), f"❌ 响应包含错误"))
                    return False, response, elapsed, f"MCP错误: {response['error']}"

                self.timeline.append((time.time(), f"✅ 响应成功"))
                return True, response, elapsed, ""

            except json.JSONDecodeError as e:
                self.timeline.append((time.time(), f"❌ JSON解析失败"))
                return False, None, elapsed, f"JSON解析失败: {e}"

        except BrokenPipeError:
            elapsed = time.time() - start_time
            self.timeline.append((time.time(), "❌ 管道断开 (BrokenPipe)"))
            return False, None, elapsed, "管道断开 (MCP Server可能崩溃)"

        except Exception as e:
            elapsed = time.time() - start_time
            self.timeline.append((time.time(), f"❌ 异常: {type(e).__name__}"))
            return False, None, elapsed, f"异常: {e}"

    def _read_response_with_timeout(self, timeout: float) -> Optional[str]:
        """使用select读取响应，支持超时"""
        deadline = time.time() + timeout
        buffer = b""

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break

            # 使用select检查stdout是否可读
            ready, _, _ = select.select([self.process.stdout], [], [], min(remaining, 1.0))

            if ready:
                # 逐字节读取，直到遇到换行
                try:
                    char = self.process.stdout.read(1)
                    if not char:
                        # EOF
                        return None

                    buffer += char

                    if char == b'\n':
                        # 完整的一行
                        line = buffer.decode('utf-8', errors='ignore').strip()
                        if line:  # 跳过空行
                            return line
                        buffer = b""

                except Exception as e:
                    return None
            else:
                # 没有数据可读，继续等待
                time.sleep(0.01)

        # 超时
        return None

    def initialize(self) -> bool:
        """发送初始化请求"""
        print_step("发送initialize请求...", "INFO")

        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"}
            },
            "id": self.request_id
        }
        self.request_id += 1

        success, response, elapsed, error = self.send_request(init_request)

        if success:
            print_step(f"初始化成功 ({elapsed:.2f}s)", "SUCCESS")
            return True
        else:
            print_step(f"初始化失败 ({elapsed:.2f}s): {error}", "ERROR")
            return False

    def call_tool(self, query: str, limit: int = 3) -> Tuple[bool, float, str]:
        """调用search_datasets工具"""
        tool_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "search_datasets",
                "arguments": {
                    "query": query,
                    "limit": limit
                }
            },
            "id": self.request_id
        }
        self.request_id += 1

        success, response, elapsed, error = self.send_request(tool_request)

        if success and response:
            # 检查是否有实际结果
            result = response.get("result", {})
            if isinstance(result, dict) and "content" in result:
                return True, elapsed, "OK"
            else:
                return False, elapsed, "响应格式错误"
        else:
            return False, elapsed, error

    def get_process_info(self) -> dict:
        """获取进程信息"""
        if not self.process or self.process.poll() is not None:
            return {"status": "not_running"}

        try:
            # 获取线程数
            proc_path = f"/proc/{self.process.pid}/status"
            threads = None
            if Path(proc_path).exists():
                with open(proc_path, 'r') as f:
                    for line in f:
                        if line.startswith('Threads:'):
                            threads = int(line.split()[1])
                            break

            # 检查stdout是否有pending数据
            ready, _, _ = select.select([self.process.stdout], [], [], 0)
            has_pending_data = bool(ready)

            return {
                "status": "running",
                "pid": self.process.pid,
                "threads": threads,
                "has_pending_stdout": has_pending_data,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def cleanup(self):
        """清理资源"""
        print_step("清理MCP Server进程...", "INFO")

        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

            print_step("进程已清理", "SUCCESS")


def create_thread_load(num_threads: int = 100, duration: float = 60) -> List[threading.Thread]:
    """创建线程负载"""
    print_step(f"创建{num_threads}个负载线程 (持续{duration}秒)...", "INFO")

    def busy_worker():
        end_time = time.time() + duration
        while time.time() < end_time:
            _ = sum(range(1000))
            time.sleep(0.001)

    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=busy_worker, daemon=True)
        t.start()
        threads.append(t)

    return threads


def test_serial_calls(cache_dir: Optional[str] = None, timeout: int = 30):
    """测试串行调用（基线）"""
    print_section("场景1: 串行调用测试（无负载）")

    tester = MCPServerTester(cache_dir=cache_dir, timeout=timeout)

    try:
        # 启动Server
        if not tester.start_server():
            return False, []

        # 初始化
        if not tester.initialize():
            print_step("初始化失败，放弃测试", "ERROR")
            return False, []

        # 连续调用5次
        queries = [
            "math reasoning",
            "competition math",  # 关键的第2次
            "problem solving",
            "algorithm",
            "data structures",
        ]

        results = []
        for i, query in enumerate(queries, 1):
            print(f"\n{Colors.BOLD}调用#{i}: {query}{Colors.RESET}")

            success, elapsed, error = tester.call_tool(query, limit=3)

            if success:
                print_step(f"成功 ({elapsed:.2f}s)", "SUCCESS")
            else:
                print_step(f"失败 ({elapsed:.2f}s): {error}", "ERROR")

                # 失败时检查进程状态
                info = tester.get_process_info()
                print(f"  进程状态: {info}")

            results.append((i, query, success, elapsed, error))

            time.sleep(0.5)  # 短暂间隔

        # 打印时间线
        print_timeline(tester.timeline)

        # 分析结果
        print(f"\n{Colors.BOLD}结果分析:{Colors.RESET}")
        failures = [r for r in results if not r[2]]

        if not failures:
            print_step("全部成功 ✅", "SUCCESS")
            return True, results
        else:
            print_step(f"{len(failures)}次失败 ❌", "ERROR")
            for call_num, query, _, elapsed, error in failures:
                print(f"  - 调用#{call_num} ({query}): {error}")

            if any(r[0] == 2 for r in failures):
                print(f"\n{Colors.RED}⚠️  第2次调用失败！符合你报告的问题特征{Colors.RESET}")

            return False, results

    finally:
        tester.cleanup()


def test_with_background_load(cache_dir: Optional[str] = None, timeout: int = 30):
    """测试在后台负载下调用"""
    print_section("场景2: 后台负载测试（模拟高负载环境）")

    # 创建负载
    load_threads = create_thread_load(num_threads=100, duration=120)
    time.sleep(2)  # 等待负载稳定

    print_step("后台负载已建立，开始测试...", "INFO")

    success, results = test_serial_calls(cache_dir=cache_dir, timeout=timeout)

    return success, results


def test_multiple_servers():
    """测试多个MCP Server同时运行时的情况"""
    print_section("场景3: 多Server测试（模拟真实机器环境）")

    print_step("启动5个后台MCP Server...", "INFO")

    # 启动5个后台Server（不发请求，只是让它们运行）
    bg_servers = []
    for i in range(5):
        cache = f"/tmp/hf_bg_test_{i}"
        os.makedirs(cache, exist_ok=True)

        tester = MCPServerTester(cache_dir=cache, timeout=30)
        if tester.start_server():
            tester.initialize()
            bg_servers.append(tester)

    print_step(f"已启动{len(bg_servers)}个后台Server", "SUCCESS")
    time.sleep(2)

    # 在前台测试
    print_step("在前台测试新的Server...", "INFO")
    fg_cache = "/tmp/hf_fg_test"
    os.makedirs(fg_cache, exist_ok=True)

    success, results = test_serial_calls(cache_dir=fg_cache, timeout=30)

    # 清理后台Server
    print_step("清理后台Server...", "INFO")
    for tester in bg_servers:
        tester.cleanup()

    return success, results


def main():
    print(f"\n{Colors.BOLD}MCP Server 长连接测试 - 精确定位超时环节{Colors.RESET}\n")

    all_results = {}

    # 场景1: 基线（独立缓存，无负载）
    print_step("准备独立缓存目录...", "INFO")
    test_cache = "/tmp/hf_test_cache_" + str(int(time.time()))
    os.makedirs(test_cache, exist_ok=True)

    try:
        success1, results1 = test_serial_calls(cache_dir=test_cache, timeout=30)
        all_results['baseline'] = success1
    except Exception as e:
        print_step(f"场景1异常: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        all_results['baseline'] = False

    # 场景2: 后台负载
    try:
        success2, results2 = test_with_background_load(cache_dir=test_cache, timeout=30)
        all_results['with_load'] = success2
    except Exception as e:
        print_step(f"场景2异常: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        all_results['with_load'] = False

    # 场景3: 多Server
    try:
        success3, results3 = test_multiple_servers()
        all_results['multiple_servers'] = success3
    except Exception as e:
        print_step(f"场景3异常: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        all_results['multiple_servers'] = False

    # 最终报告
    print_section("最终诊断报告")

    print("测试结果:")
    print(f"  场景1 (基线): {'✅ PASS' if all_results.get('baseline') else '❌ FAIL'}")
    print(f"  场景2 (后台负载): {'✅ PASS' if all_results.get('with_load') else '❌ FAIL'}")
    print(f"  场景3 (多Server): {'✅ PASS' if all_results.get('multiple_servers') else '❌ FAIL'}")

    print("\n诊断结论:")
    if all_results.get('baseline') is False:
        print(f"{Colors.RED}🔴 MCP Server本身有问题{Colors.RESET}")
        print("   → 即使在理想条件下也失败")
        print("   → 检查MCP工具代码、网络连接")
    elif all_results.get('with_load') is False and all_results.get('baseline') is True:
        print(f"{Colors.YELLOW}🟡 系统负载导致问题{Colors.RESET}")
        print("   → 基线测试通过，但高负载下失败")
        print("   → 解决方案: 减少并发任务，增加系统资源")
    elif all_results.get('multiple_servers') is False:
        print(f"{Colors.YELLOW}🟠 多Server竞争导致问题{Colors.RESET}")
        print("   → 单个Server正常，多个Server时失败")
        print("   → 解决方案: 清理残留MCP Server进程")
    else:
        print(f"{Colors.GREEN}🟢 测试环境下正常{Colors.RESET}")
        print("   → 无法在测试中复现问题")
        print("   → 可能是间歇性问题或特定触发条件")


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
