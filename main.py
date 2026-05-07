import asyncio
import json
import logging
import os
import argparse
import mcp

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from mcp_modules.server_manager import MultiServerManager

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("mcp_tool_caller")
time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")


def load_server_configs(config_path: Path):
    """从 MCP config 文件加载并转换 server 配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    servers = []

    for name, conf in cfg.get("mcpServers", {}).items():
        if conf.get("transport") == "sse":
            servers.append(
                {
                    "name": name,
                    "url": conf.get("url"),
                    "transport": conf.get("transport", "sse"),
                }
            )
        else:
            servers.append(
                {
                    "name": name,
                    "command": [conf.get("command")] + conf.get("args", []),
                    "env": conf.get("env"),
                    "cwd": conf.get("cwd"),
                    "transport": conf.get("transport", "stdio"),
                    "port": conf.get("port", None),
                    "endpoint": conf.get("endpoint", "/mcp"),
                }
            )
    return servers


async def get_tool_response(config_path: str | Path) -> Any:
    """连接 MCP server 并调用工具，返回结果"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    server_configs = load_server_configs(config_path)
    if not server_configs:
        raise ValueError("No servers found in config file.")

    manager = MultiServerManager(server_configs)
    try:
        logger.info("🔌 Connecting and discovering tools...")
        all_tools = await manager.connect_all_servers()
        print(
            f"All tools:\n{json.dumps(all_tools, indent=2, ensure_ascii=False)}\n\n"
            + "=" * 20
            + "\n\n"
        )
        # Replace colons with underscores in keys
        def replace_colons_in_keys(obj):
            if isinstance(obj, dict):
                new_dict = {}
                for key, value in obj.items():
                    new_key = key.replace(":", "_")
                    new_dict[new_key] = replace_colons_in_keys(value)
                return new_dict
            elif isinstance(obj, list):
                return [replace_colons_in_keys(item) for item in obj]
            return obj

        all_tools = replace_colons_in_keys(all_tools)

        with open(
            f"./results/{time_stamp}_list_tools.json", "w", encoding="utf-8"
        ) as file:
            json.dump(all_tools, file, indent=4, ensure_ascii=False)

    finally:
        await manager.close_all_connections()


async def run_example(server_path: str):
    with open(server_path, "r", encoding="utf-8") as file:
        server_data = json.load(file)
    tool_prefix = list(server_data["mcpServers"].keys())[0]
    result: mcp.types.CallToolRequest = await get_tool_response(config_path=server_path)


async def main():
    parser = argparse.ArgumentParser(description="MCP Validator")
    parser.add_argument("--server_path", type=str)
    args = parser.parse_args()
    server_path = args.server_path
    os.makedirs("results", exist_ok=True)
    result = await run_example(server_path=server_path)


if __name__ == "__main__":
    asyncio.run(main())
