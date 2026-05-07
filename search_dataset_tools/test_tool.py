#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化测试框架

使用方式:
    python test_tool.py search_huggingface
    python test_tool.py search_github
    python test_tool.py search_web
    python test_tool.py search_scholar
    python test_tool.py all
"""

import os
import sys
import json
import importlib
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


# ============================================================================
# 配置路径
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
CONFIG_FILE = BASE_DIR / "configs" / "mcp_config4data.json"
TEST_PARAMS_DIR = Path(__file__).parent / "test_params"
TOOLS_DIR = Path(__file__).parent


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class TestCase:
    """测试用例"""
    tool: str
    input_params: Dict[str, Any]
    expected_contains: Optional[List[str]] = None
    expected_not_contains: Optional[List[str]] = None


@dataclass
class TestResult:
    """测试结果"""
    tool_name: str
    test_case: TestCase
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0


# ============================================================================
# 环境变量加载
# ============================================================================

def load_env_from_config(config_path: Path = CONFIG_FILE) -> Dict[str, str]:
    """从 mcp_config4data.json 加载环境变量"""
    env_vars = {}

    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}")
        return env_vars

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    for server_name, server_config in config.get("mcpServers", {}).items():
        server_env = server_config.get("env", {})
        for key, value in server_env.items():
            if value:
                env_vars[key] = value

    return env_vars


def setup_environment(env_vars: Dict[str, str]):
    """设置环境变量"""
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Set env: {key}=***{value[-4:] if len(value) > 4 else value}")


# ============================================================================
# 测试参数加载
# ============================================================================

def load_test_params(tool_name: str) -> List[TestCase]:
    """加载测试参数"""
    param_file = TEST_PARAMS_DIR / f"{tool_name}.json"

    if not param_file.exists():
        print(f"Warning: Test params file not found: {param_file}")
        return []

    with open(param_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_cases = []
    for item in data:
        test_cases.append(TestCase(
            tool=item.get("tool", ""),
            input_params=item.get("input_params", {}),
            expected_contains=item.get("expected_contains"),
            expected_not_contains=item.get("expected_not_contains")
        ))

    return test_cases


# ============================================================================
# 工具函数执行
# ============================================================================

class ToolExecutor:
    """工具执行器 - 通过 MCP 协议调用工具"""

    def __init__(self, tool_module_path: Path):
        self.tool_module_path = tool_module_path
        self.module_name = tool_module_path.stem

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> tuple[str, float]:
        """
        通过 MCP stdio 协议执行工具

        由于直接导入 FastMCP 模块可能会有初始化问题，
        这里使用 subprocess 调用并构造 MCP 请求
        """
        import time

        # 构造 MCP 请求
        mcp_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            }
        }

        start_time = time.time()

        try:
            # 使用 subprocess 调用 MCP 工具
            result = subprocess.run(
                [sys.executable, str(self.tool_module_path)],
                input=json.dumps(mcp_request) + "\n",
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.tool_module_path.parent)
            )

            execution_time = time.time() - start_time

            # 解析响应
            if result.stdout:
                try:
                    # MCP 协议可能返回多行 JSON
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            response = json.loads(line)
                            if "result" in response:
                                content = response["result"]
                                if isinstance(content, dict) and "content" in content:
                                    for item in content["content"]:
                                        if item.get("type") == "text":
                                            return item.get("text", ""), execution_time
                                elif isinstance(content, str):
                                    return content, execution_time
                except json.JSONDecodeError:
                    pass

            # 如果无法解析 MCP 响应，返回原始输出
            return result.stdout or result.stderr or "No output", execution_time

        except subprocess.TimeoutExpired:
            return "Error: Tool execution timeout", 60.0
        except Exception as e:
            return f"Error: {str(e)}", time.time() - start_time


class DirectToolExecutor:
    """直接导入执行器 - 直接导入模块并调用函数"""

    def __init__(self, tool_module_path: Path):
        self.tool_module_path = tool_module_path
        self.module = None
        self._import_module()

    def _import_module(self):
        """导入工具模块"""
        try:
            # 添加模块路径到 sys.path
            if str(self.tool_module_path.parent) not in sys.path:
                sys.path.insert(0, str(self.tool_module_path.parent))

            # 动态导入模块
            module_name = self.tool_module_path.stem
            self.module = importlib.import_module(module_name)

        except Exception as e:
            print(f"Warning: Failed to import module {self.tool_module_path}: {e}")

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> tuple[str, float]:
        """直接调用工具函数"""
        import time

        if self.module is None:
            return "Error: Module not loaded", 0.0

        start_time = time.time()

        try:
            # 查找工具函数
            if hasattr(self.module, tool_name):
                func = getattr(self.module, tool_name)
                result = func(**params)
                execution_time = time.time() - start_time
                return str(result), execution_time
            else:
                # 查找 @mcp.tool() 装饰的函数
                # FastMCP 将工具函数存储在 _tool_manager 中
                if hasattr(self.module, 'mcp'):
                    mcp_instance = self.module.mcp
                    # 遍历已注册的工具
                    tool_manager = mcp_instance._tool_manager
                    for registered_tool in tool_manager._tools.values():
                        if registered_tool.name == tool_name:
                            result = registered_tool.fn(**params)
                            execution_time = time.time() - start_time
                            return str(result), execution_time

                return f"Error: Tool '{tool_name}' not found in module", time.time() - start_time

        except Exception as e:
            execution_time = time.time() - start_time
            return f"Error executing tool: {str(e)}", execution_time


# ============================================================================
# 结果验证
# ============================================================================

def validate_output(output: str, test_case: TestCase) -> tuple[bool, List[str]]:
    """验证输出是否符合预期"""
    errors = []

    if test_case.expected_contains:
        for expected in test_case.expected_contains:
            if expected not in output:
                errors.append(f"Expected to contain: '{expected}'")

    if test_case.expected_not_contains:
        for not_expected in test_case.expected_not_contains:
            if not_expected in output:
                errors.append(f"Should not contain: '{not_expected}'")

    # 基本验证：输出不能为空且不包含错误信息
    if not output or output.strip() == "":
        errors.append("Output is empty")

    error_keywords = ["Error:", "error:", "Exception", "Traceback"]
    if any(keyword in output for keyword in error_keywords):
        errors.append(f"Output contains error keywords: {[k for k in error_keywords if k in output]}")

    return len(errors) == 0, errors


# ============================================================================
# 测试运行器
# ============================================================================

def run_test(tool_name: str, test_case: TestCase, executor) -> TestResult:
    """运行单个测试"""
    print(f"\n{'='*60}")
    print(f"Testing: {tool_name} -> {test_case.tool}")
    print(f"Params: {json.dumps(test_case.input_params, ensure_ascii=False)}")
    print(f"{'='*60}")

    output, exec_time = executor.execute_tool(test_case.tool, test_case.input_params)

    success, errors = validate_output(output, test_case)

    result = TestResult(
        tool_name=tool_name,
        test_case=test_case,
        success=success,
        output=output,
        error="; ".join(errors) if errors else None,
        execution_time=exec_time
    )

    return result


def print_result(result: TestResult, verbose: bool = True):
    """打印测试结果"""
    status_icon = "✅" if result.success else "❌"
    print(f"\n{status_icon} Test Result: {result.test_case.tool}")
    print(f"   Execution Time: {result.execution_time:.2f}s")

    if result.success:
        print("   Status: PASSED")
    else:
        print(f"   Status: FAILED")
        print(f"   Error: {result.error}")

    if verbose and result.output:
        # 显示输出预览
        preview = result.output[:500]
        if len(result.output) > 500:
            preview += "..."
        print(f"\n   Output Preview:")
        for line in preview.split('\n')[:10]:
            print(f"   {line}")
            
    else:
        print(result.output)


def run_tool_tests(tool_name: str) -> List[TestResult]:
    """运行指定工具的所有测试"""
    print(f"\n{'#'*60}")
    print(f"# Running tests for: {tool_name}")
    print(f"{'#'*60}")

    # 加载环境变量
    env_vars = load_env_from_config()
    setup_environment(env_vars)

    # 加载测试参数
    test_cases = load_test_params(tool_name)
    if not test_cases:
        print(f"No test cases found for {tool_name}")
        return []

    # 创建执行器
    tool_module_path = TOOLS_DIR / f"{tool_name}.py"
    if not tool_module_path.exists():
        print(f"Tool module not found: {tool_module_path}")
        return []

    executor = DirectToolExecutor(tool_module_path)

    # 运行测试
    results = []
    for test_case in test_cases:
        result = run_test(tool_name, test_case, executor)
        results.append(result)
        print_result(result, verbose=False)

    # 汇总
    passed = sum(1 for r in results if r.success)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} tests passed")
    print(f"{'='*60}")

    return results


def run_all_tests() -> Dict[str, List[TestResult]]:
    """运行所有工具的测试"""
    all_results = {}

    # 查找所有测试参数文件
    test_files = list(TEST_PARAMS_DIR.glob("*.json"))

    tool_names = [f.stem for f in test_files]

    for tool_name in tool_names:
        results = run_tool_tests(tool_name)
        all_results[tool_name] = results

    # 总体汇总
    print(f"\n{'#'*60}")
    print(f"# Overall Test Summary")
    print(f"{'#'*60}")

    total_passed = sum(sum(1 for r in results if r.success) for results in all_results.values())
    total_tests = sum(len(results) for results in all_results.values())

    for tool_name, results in all_results.items():
        passed = sum(1 for r in results if r.success)
        total = len(results)
        status = "✅" if passed == total else "❌"
        print(f"{status} {tool_name}: {passed}/{total}")

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    return all_results


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python test_tool.py <tool_name>|all")
        print("\nAvailable tools:")
        for f in TEST_PARAMS_DIR.glob("*.json"):
            print(f"  - {f.stem}")
        sys.exit(1)

    tool_name = sys.argv[1]

    if tool_name == "all":
        run_all_tests()
    else:
        run_tool_tests(tool_name)


if __name__ == "__main__":
    main()
