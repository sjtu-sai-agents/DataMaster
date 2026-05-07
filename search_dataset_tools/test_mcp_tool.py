#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP 工具交互式测试终端

使用方式:
    python test_mcp_tool.py --config_file <config_file>

命令:
    help                    - 显示帮助信息
    list                    - 展示所有可用的工具
    call <tool_name>        - 交互式测试特定工具
    call_batch <config>     - 批量测试工具
    exit/quit               - 退出终端
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from mcp_modules.server_manager import MultiServerManager


# ============================================================================
# 配置路径
# ============================================================================

DEFAULT_CONFIG_FILE = ROOT_DIR / "configs/ml_master_datatree/base_all.json"
TEST_BATCH_DIR = Path(__file__).parent / "mcp_test"


# ============================================================================
# 颜色输出辅助类
# ============================================================================

class Colors:
    """终端颜色代码"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_success(msg: str):
    """打印成功消息"""
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")


def print_error(msg: str):
    """打印错误消息"""
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")


def print_warning(msg: str):
    """打印警告消息"""
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")


def print_info(msg: str):
    """打印信息消息"""
    print(f"{Colors.OKCYAN}ℹ {msg}{Colors.ENDC}")


def print_header(msg: str):
    """打印标题"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{msg}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}\n")


# ============================================================================
# MCP 管理器类
# ============================================================================

class MCPTestManager:
    """MCP 测试管理器"""

    def __init__(self, config_file: Path):
        self.config_file = config_file
        self.server_manager: Optional[MultiServerManager] = None
        self.all_tools: Dict[str, Any] = {}

    async def initialize(self):
        """初始化 MCP 服务器连接"""
        print_info(f"Loading config from: {self.config_file}")

        if not self.config_file.exists():
            print_error(f"Config file not found: {self.config_file}")
            return False

        # 加载配置
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        server_configs = []
        for server_name, server_config in config_data.get("mcpServers", {}).items():
            server_configs.append({
                "name": server_name,
                "command": [server_config["command"]] + server_config.get("args", []),
                "env": server_config.get("env", {}),
                "cwd": str(ROOT_DIR),
                "transport": "stdio"
            })

        print_info(f"Found {len(server_configs)} server configurations")

        # 创建服务器管理器
        self.server_manager = MultiServerManager(
            server_configs=server_configs,
            filter_problematic_tools=False
        )

        # 连接所有服务器
        print_header("Connecting to MCP Servers")
        try:
            self.all_tools = await self.server_manager.connect_all_servers()
            print_success(f"Connected! Discovered {len(self.all_tools)} tools")
            return True
        except Exception as e:
            print_error(f"Failed to connect: {e}")
            return False

    def list_tools(self):
        """列出所有可用工具"""
        print_header("Available Tools")

        if not self.all_tools:
            print_warning("No tools available")
            return

        # 按服务器分组
        server_tools: Dict[str, List[str]] = {}
        for tool_key, tool_info in self.all_tools.items():
            server = tool_info.get("server", "unknown")
            if server not in server_tools:
                server_tools[server] = []
            server_tools[server].append(tool_key)

        # 显示每个服务器的工具
        for server, tools in sorted(server_tools.items()):
            print(f"\n{Colors.BOLD}{Colors.OKCYAN}Server: {server}{Colors.ENDC}")
            for tool in sorted(tools):
                tool_info = self.all_tools[tool]
                desc = tool_info.get("description", "No description")
                print(f"  {Colors.OKGREEN}•{Colors.ENDC} {tool}")
                print(f"    {Colors.WARNING}{desc}{Colors.ENDC}")

    def get_tool_info(self, tool_name: str) -> Optional[tuple[str, Dict[str, Any]]]:
        """获取工具信息，返回 (完整工具名, 工具信息) 或 None"""
        # 支持完整名称或短名称
        if tool_name in self.all_tools:
            return tool_name, self.all_tools[tool_name]

        # 尝试查找匹配的工具
        for key, info in self.all_tools.items():
            if key.endswith(f"_{tool_name}") or info.get("original_name") == tool_name:
                return key, info

        return None

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """调用工具"""
        tool_result = self.get_tool_info(tool_name)
        if not tool_result:
            print_error(f"Tool not found: {tool_name}")
            return None

        full_tool_name, tool_info = tool_result
        print_info(f"Calling tool: {full_tool_name}")
        print_info(f"Parameters: {json.dumps(params, ensure_ascii=False, indent=2)}")

        try:
            call_result = await self.server_manager.call_tool(full_tool_name, params)
            return call_result
        except Exception as e:
            print_error(f"Error calling tool: {e}")
            return None

    def print_result(self, result: Any):
        """打印结果"""
        if result is None:
            return

        print_header("Result")
        if isinstance(result, dict):
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif isinstance(result, list):
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(str(result))

    async def interactive_call(self, tool_name: str):
        """交互式调用工具"""
        tool_result = self.get_tool_info(tool_name)
        if not tool_result:
            print_error(f"Tool not found: {tool_name}")
            print_info("Use 'list' to see available tools")
            return

        full_tool_name, tool_info = tool_result
        input_schema = tool_info.get("input_schema", {})

        print_header(f"Tool: {full_tool_name}")
        print(f"Description: {tool_info.get('description', 'No description')}")
        print(f"Server: {tool_info.get('server', 'unknown')}")

        # 解析参数
        params = {}
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        if properties:
            print(f"\n{Colors.BOLD}Parameters:{Colors.ENDC}")
            for param_name, param_info in properties.items():
                is_required = param_name in required
                req_str = f"{Colors.FAIL}(required){Colors.ENDC}" if is_required else f"{Colors.OKCYAN}(optional){Colors.ENDC}"
                param_desc = param_info.get("description", "")
                param_type = param_info.get("type", "any")

                print(f"\n  {Colors.BOLD}{param_name}{Colors.ENDC} {req_str}")
                print(f"    Type: {param_type}")
                if param_desc:
                    print(f"    Description: {param_desc}")

                # 获取用户输入
                default = param_info.get("default")
                default_suffix = f" [default: {default}]" if default is not None else ""

                user_input = input(f"  {Colors.OKBLUE}>>>{Colors.ENDC} {default_suffix}: ").strip()

                if user_input:
                    # 尝试解析 JSON
                    try:
                        params[param_name] = json.loads(user_input)
                    except json.JSONDecodeError:
                        params[param_name] = user_input
                elif default is not None:
                    params[param_name] = default
                elif is_required:
                    print_error(f"Parameter '{param_name}' is required but no value provided")
                    return

        # 调用工具
        result = await self.call_tool(full_tool_name, params)
        self.print_result(result)

    async def batch_call(self, batch_config: str):
        """批量调用工具"""
        # 处理路径
        batch_path = Path(batch_config)
        if not batch_path.exists():
            # 尝试在默认目录中查找
            batch_path = TEST_BATCH_DIR / batch_config
            if not batch_path.exists():
                batch_path = TEST_BATCH_DIR / f"{batch_config}.json"

        if not batch_path.exists():
            print_error(f"Batch config file not found: {batch_config}")
            return

        print_info(f"Loading batch config from: {batch_path}")

        try:
            with open(batch_path, 'r', encoding='utf-8') as f:
                test_cases = json.load(f)
        except Exception as e:
            print_error(f"Failed to load batch config: {e}")
            return

        print_header(f"Running {len(test_cases)} test cases")

        passed = 0
        failed = 0

        for i, test_case in enumerate(test_cases, 1):
            tool_name = test_case.get("tool")
            input_params = test_case.get("input_params", {})

            print(f"\n[{i}/{len(test_cases)}] {Colors.BOLD}Testing: {tool_name}{Colors.ENDC}")
            print(f"  Parameters: {json.dumps(input_params, ensure_ascii=False)}")

            result = await self.call_tool(tool_name, input_params)

            if result is not None:
                passed += 1
                print_success(f"Test passed")
                # 只显示结果摘要
                if isinstance(result, dict):
                    print(f"  Result keys: {list(result.keys())[:5]}")
                elif isinstance(result, list):
                    print(f"  Result length: {len(result)} items")
                else:
                    preview = str(result)
                    print(f"  Result: {preview}")
            else:
                failed += 1
                print_error(f"Test failed")

        print_header("Batch Test Summary")
        print(f"Total: {len(test_cases)}")
        print(f"{Colors.OKGREEN}Passed: {passed}{Colors.ENDC}")
        print(f"{Colors.FAIL}Failed: {failed}{Colors.ENDC}")

    async def close(self):
        """关闭连接"""
        if self.server_manager:
            await self.server_manager.close_all_connections()
            print_info("Closed all connections")


# ============================================================================
# 交互式终端
# ============================================================================

class InteractiveTerminal:
    """交互式测试终端"""

    def __init__(self, manager: MCPTestManager):
        self.manager = manager
        self.running = True

    def show_help(self):
        """显示帮助信息"""
        print_header("MCP Tool Test Terminal - Help")
        print(f"""
{Colors.BOLD}Available Commands:{Colors.ENDC}
  {Colors.OKGREEN}help{Colors.ENDC}                    - 显示此帮助信息
  {Colors.OKGREEN}list{Colors.ENDC}                    - 列出所有可用的工具
  {Colors.OKGREEN}call <tool_name>{Colors.ENDC}        - 交互式测试指定工具
  {Colors.OKGREEN}call_batch <config>{Colors.ENDC}     - 批量测试工具
  {Colors.OKGREEN}exit{Colors.ENDC} / {Colors.OKGREEN}quit{Colors.ENDC}           - 退出终端

{Colors.BOLD}Examples:{Colors.ENDC}
  {Colors.OKCYAN}call search_web_google_search{Colors.ENDC}
  {Colors.OKCYAN}call_batch test_web_search.json{Colors.ENDC}
  {Colors.OKCYAN}call search_huggingface_search_datasets{Colors.ENDC}

{Colors.BOLD}Note:{Colors.ENDC}
  - 工具名称支持完整名称或短名称
  - 使用 'list' 命令查看所有可用工具
  - 批量测试配置文件放在 {TEST_BATCH_DIR} 目录下
        """)

    async def run(self):
        """运行交互式终端"""
        print_header("MCP Tool Test Terminal")
        print(f"Type {Colors.OKGREEN}'help'{Colors.ENDC} for available commands\n")

        while self.running:
            try:
                # 获取用户输入
                user_input = input(f"{Colors.BOLD}{Colors.OKBLUE}mcp-test>{Colors.ENDC} ").strip()

                if not user_input:
                    continue

                # 解析命令
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                # 处理命令
                if command in ["exit", "quit"]:
                    await self.shutdown()
                    break
                elif command == "help":
                    self.show_help()
                elif command == "list":
                    self.manager.list_tools()
                elif command == "call":
                    if not args:
                        print_error("Usage: call <tool_name>")
                        continue
                    await self.manager.interactive_call(args.strip())
                elif command == "call_batch":
                    if not args:
                        print_error("Usage: call_batch <config_file>")
                        continue
                    await self.manager.batch_call(args.strip())
                else:
                    print_error(f"Unknown command: {command}")
                    print_info("Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\n")
                await self.shutdown()
                break
            except EOFError:
                await self.shutdown()
                break
            except Exception as e:
                print_error(f"Error: {e}")

    async def shutdown(self):
        """关闭终端"""
        print_info("Shutting down...")
        self.running = False
        await self.manager.close()


# ============================================================================
# 主程序
# ============================================================================

async def main():
    """主函数"""
    # 解析命令行参数
    config_file = DEFAULT_CONFIG_FILE

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--config_file" and i + 1 < len(sys.argv):
            config_file = Path(sys.argv[i + 1])
        elif arg in ["--help", "-h"]:
            print("Usage: python test_mcp_tool.py [--config_file <config_file>]")
            print(f"Default config file: {DEFAULT_CONFIG_FILE}")
            return

    # 创建管理器
    manager = MCPTestManager(config_file)

    # 初始化
    if not await manager.initialize():
        print_error("Failed to initialize MCP manager")
        return

    # 运行交互式终端
    terminal = InteractiveTerminal(manager)
    await terminal.run()


if __name__ == "__main__":
    asyncio.run(main())
