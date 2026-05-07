"""
MCP Manager Module

Manages multiple MCP server connections and coordinates tool discovery and execution.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any

import aiohttp
from mcp import ClientSession
from mcp.client.stdio import stdio_client

import config.config_loader as config_loader

from mcp_modules.connector import MCPConnector
from mcp_modules.tool_cache import get_cache

logger = logging.getLogger(__name__)

TOOL_CALL_ERROR = 35
logging.addLevelName(TOOL_CALL_ERROR, 'TOOL CALL ERROR')


def update_server_params(old_params, new_key, new_profile):
    """
    仅当 server_params.args 中包含 '@smithery/cli' 时，
    替换其中的 --key 和 --profile 参数。
    """
    args = list(old_params.args)

    # 仅处理使用 Smithery CLI 的情况
    if not any("smithery" in str(a) for a in args):
        # 不包含 smithery，不修改参数
        return old_params

    def replace_arg(flag, new_value):
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                args[idx + 1] = new_value
            else:
                args.append(new_value)
        else:
            args.extend([flag, new_value])

    replace_arg("--key", new_key)
    replace_arg("--profile", new_profile)

    return old_params.__class__(
        command=old_params.command,
        args=args,
        env=old_params.env,
        cwd=old_params.cwd,
        encoding=old_params.encoding,
        encoding_error_handler=old_params.encoding_error_handler,
    )


class MultiServerManager:
    """Manages multiple MCP server connections and coordinates tool discovery."""
    
    def __init__(self, server_configs: List[Dict[str, Any]],
                 filter_problematic_tools: bool = False):
        self.server_configs = server_configs
        self.connectors: Dict[str, MCPConnector] = {}
        self.sessions: Dict[str, ClientSession] = {}
        self.clients: Dict[str, Any] = {}
        self.all_tools: Dict[str, Any] = {}
        logger.info(f"MultiServerManager initialized with {len(server_configs)} server configurations")


        for config in server_configs:
            server_name = config["name"]
            transport_type = config.get("transport", "stdio")
            
            self.connectors[server_name] = MCPConnector(
                server_name, 
                # config["command"], 
                config.get("command",None),
                config.get("env",None),
                config.get("cwd",None),
                transport_type=transport_type,
                port=config.get("port",None),
                endpoint=config.get("endpoint", "/mcp"),
                server_url=config.get("url",None)
            )

    async def connect_all_servers(self) -> Dict[str, Any]:
        """Connects to all configured servers and discovers their tools."""
        logger.info(f"Connecting to {len(self.server_configs)} MCP servers...")
        
        connection_tasks = []
        for config in self.server_configs:
            server_name = config["name"]
            connection_tasks.append(self._connect_single_server(server_name))
        
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        successful_connections = 0
        for i, result in enumerate(results):
            server_name = self.server_configs[i]["name"]
            if isinstance(result, Exception):
                logger.error(f"Failed to connect to {server_name}: {result}")
            elif result is not None:
                # 只有当 result 不为 None 且是字典时才更新
                if isinstance(result, dict):
                    successful_connections += 1
                    self.all_tools.update(result)
                else:
                    logger.warning(f"Unexpected result type from {server_name}: {type(result)}")
            else:
                logger.warning(f"Connection to {server_name} returned None (all key/profile combinations may have failed)")
        
        logger.info(f"Successfully connected to {successful_connections}/{len(self.server_configs)} servers")
        logger.info(f"Total tools discovered: {len(self.all_tools)}")
        
        return self.all_tools

    async def _connect_single_server(self, server_name: str) -> Dict[str, Any]:
        """Connects to a single server and discovers its tools."""
        connector = self.connectors[server_name]
        
        if connector.transport_type == "http":
            print("HTTP server")
            if connector.server_url:
                return await self._connect_url_server(server_name)
            else:
                return await self._connect_http_server(server_name)
        elif connector.transport_type == "sse":
            print("SSE server")
            if connector.server_url:
                return await self._connect_url_server(server_name)
            else:
                return await self._connect_sse_server(server_name)
        else:
            print("STDIO server")
            return await self._connect_stdio_server(server_name)
    
    async def _connect_stdio_server(self, server_name: str) -> Dict[str, Any]:
        """Connects to a STDIO MCP server."""
        connector = self.connectors[server_name]
        
        print(f"Connecting to {server_name} with STDIO params: {connector.server_params}")

        async def full_call():
            async with stdio_client(connector.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    # return await session.call_tool(tool_name, parameters)
                    tools = await connector.discover_tools(session)
                    logger.debug("tools: %s", tools)

                    self.sessions[server_name] = None
                    # print("[debug] tools: %s", tools)
                    return tools

        timeout = 120

        # 检查是否为 Smithery 服务器
        is_smithery_server = any("smithery" in str(arg) for arg in connector.server_params.args)
        print("is_smithery_server: ", is_smithery_server)
        if not is_smithery_server:
            # 非 Smithery 服务器，直接连接，不使用 key/profile
            try:
                result = await asyncio.wait_for(full_call(), timeout=timeout)
                print(f"✅ Connected successfully to {server_name}")
                return result
            except asyncio.TimeoutError:
                print(f"⚠️ Timeout connecting to {server_name}")
                raise
            except Exception as e:
                print(f"❌ Error connecting to {server_name}: {e}")
                raise
        

    async def _connect_http_server(self, server_name: str) -> Dict[str, Any]:
        """Connects to an HTTP MCP server."""
        connector = self.connectors[server_name]
        
        logger.info(f"Connecting to {server_name} with HTTP transport on port {connector.port}")
        try:
            if not await connector.start_http_server():
                raise Exception(f"Failed to start HTTP server for {server_name}")
            
            tools = await connector.discover_tools_http()
            logger.debug("tools: %s", tools)
            
            return tools
            
        except Exception as e:
            logger.error(f"ERROR in connecting to HTTP server {server_name}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            await connector.stop_http_server()
            raise

    async def _connect_sse_server(self, server_name: str) -> Dict[str, Any]:
        """Connects to an SSE MCP server."""
        connector = self.connectors[server_name]
        
        logger.info(f"Connecting to {server_name} with SSE transport on port {connector.port}")
        try:
            if not await connector.start_http_server():
                raise Exception(f"Failed to start SSE server for {server_name}")
            
            tools = await connector.discover_tools_sse()
            logger.debug("tools: %s", tools)
            print("tools: %s", tools)
            return tools
            
        except Exception as e:
            logger.error(f"ERROR in connecting to SSE server {server_name}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            await connector.stop_sse_server()
            raise

    async def _connect_url_server(self, server_name: str) -> Dict[str, Any]:
        """Connects to a URL-based MCP server (HTTP or SSE)."""
        connector = self.connectors[server_name]
        
        logger.info(f"Connecting to {server_name} with URL-based {connector.transport_type} transport: {connector.server_url}")
        try:
            tools = await connector.discover_tools_url()
            logger.debug("tools: %s", tools)
            
            return tools
            
        except Exception as e:
            logger.error(f"ERROR in connecting to URL server {server_name}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            await connector.stop_url_server()
            raise

    async def call_tool(self, tool_name: str, parameters: Dict[str, Any], use_cache: bool = True) -> Any:
        """Calls a tool on the appropriate server by creating a new connection."""
        if tool_name not in self.all_tools:
            print("[debug] tool not found: ", tool_name)
            raise ValueError(f"Tool '{tool_name}' not found")
        
        tool_info = self.all_tools[tool_name]
        server_name = tool_info["server"]
        original_tool_name = tool_info["original_name"]
        
        # Check cache first if enabled
        cache = get_cache()
        if use_cache and cache.enabled:
            cached_result = cache.get(server_name, original_tool_name, parameters)
            if cached_result is not None:
                return cached_result
        
        connector = self.connectors[server_name]
        
        logger.info(f"Calling tool '{original_tool_name}' on server '{server_name}' with params: {json.dumps(parameters)}")
        
        if connector.transport_type == "http":
            if connector.server_url:
                result = await self._call_tool_url(connector, original_tool_name, parameters)
            else:
                result = await self._call_tool_http(connector, original_tool_name, parameters)
        elif connector.transport_type == "sse":
            if connector.server_url:
                result = await self._call_tool_url(connector, original_tool_name, parameters)
            else:
                result = await self._call_tool_sse(connector, original_tool_name, parameters)
        else:
            result = await self._call_tool_stdio(connector, original_tool_name, parameters)
        
        # Store in cache if successful and enabled
        # Additional validation before caching
        if use_cache and cache.enabled:
            # Only cache if result is valid and not empty
            if result and result != {} and result != []:
                cache.set(server_name, original_tool_name, parameters, result)
            else:
                logger.debug(f"Skipping cache for empty/invalid result from {server_name}:{original_tool_name}")
        
        return result

    async def _call_tool_stdio(self, connector: MCPConnector, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call tool using STDIO transport, with key/profile rotation on timeout."""

        async def full_call():
            print("[debug] spawn stdio...")
            async with stdio_client(connector.server_params) as (read, write):
                print("[debug] stdio ok, create session...")
                async with ClientSession(read, write) as session:
                    print("[debug] calling initialize()")
                    await session.initialize()
                    print("[debug] initialize OK, calling tool...")
                    return await session.call_tool(tool_name, parameters)

        timeout = 1800

        # 非 Smithery CLI：直接调用一次
        is_smithery_server = any("smithery" in str(arg) for arg in connector.server_params.args)
        if not is_smithery_server:
            try:
                return await asyncio.wait_for(full_call(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.error(f"Tool call '{tool_name}' timed out after {timeout} seconds")
                raise
            except Exception as e:
                # 详细记录错误信息
                logger.error(f"Error calling tool '{tool_name}': {type(e).__name__}: {e}")
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise

    
    async def _call_tool_http(self, connector: MCPConnector, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call tool using HTTP transport."""
        print("[debug] calling http tool...")
        base_url = f"http://localhost:{connector.port}{connector.endpoint}"
        
        tool_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": parameters
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    base_url,
                    json=tool_request,
                    headers={
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    timeout=config_loader.get_mcp_timeout()
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if "error" in result:
                        raise Exception(f"MCP Error: {result['error']}")
                    
                    # Check if we got a valid result
                    tool_result = result.get("result")
                    if tool_result is None:
                        # If no result field, check if the entire response is the result
                        if result and result != {}:
                            return result
                        else:
                            raise Exception(f"No valid result returned from tool '{tool_name}'")
                    
                    return tool_result
                    
        except Exception as e:
            logger.log(TOOL_CALL_ERROR, f"ERROR in calling HTTP tool '{tool_name}': {e}")
            import traceback
            logger.log(TOOL_CALL_ERROR, f"Full traceback: {traceback.format_exc()}")
            raise

    async def _call_tool_sse(self, connector: MCPConnector, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call tool using SSE transport."""
        try:
            print("[debug] calling sse tool...")
            return await connector.call_tool_sse(tool_name, parameters)
        except Exception as e:
            logger.log(TOOL_CALL_ERROR, f"ERROR in calling SSE tool '{tool_name}': {e}")
            import traceback
            logger.log(TOOL_CALL_ERROR, f"Full traceback: {traceback.format_exc()}")
            raise

    async def _call_tool_url(self, connector: MCPConnector, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Call tool using URL-based connection."""
        try:
            print("[debug] calling url tool...")
            return await connector.call_tool_url(tool_name, parameters)
        except Exception as e:
            logger.log(TOOL_CALL_ERROR, f"ERROR in calling URL tool '{tool_name}': {e}")
            import traceback
            logger.log(TOOL_CALL_ERROR, f"Full traceback: {traceback.format_exc()}")
            raise

    async def close_all_connections(self):
        """Closes all server connections."""
        logger.info(f"Closing connections to {len(self.connectors)} MCP servers...")
        
        # Stop all HTTP/SSE/URL servers concurrently for faster cleanup
        server_cleanup_tasks = []
        for server_name, connector in self.connectors.items():
            if connector.transport_type == "http":
                if connector.server_url:
                    logger.info(f"Scheduling cleanup for URL server {server_name}")
                    server_cleanup_tasks.append(self._cleanup_url_server(server_name, connector))
                else:
                    logger.info(f"Scheduling cleanup for HTTP server {server_name}")
                    server_cleanup_tasks.append(self._cleanup_http_server(server_name, connector))
            elif connector.transport_type == "sse":
                if connector.server_url:
                    logger.info(f"Scheduling cleanup for URL server {server_name}")
                    server_cleanup_tasks.append(self._cleanup_url_server(server_name, connector))
                else:
                    logger.info(f"Scheduling cleanup for SSE server {server_name}")
                    server_cleanup_tasks.append(self._cleanup_sse_server(server_name, connector))
        
        # Wait for all HTTP/SSE servers to be cleaned up
        if server_cleanup_tasks:
            logger.info(f"Waiting for {len(server_cleanup_tasks)} servers to shutdown...")
            results = await asyncio.gather(*server_cleanup_tasks, return_exceptions=True)
            
            # Log any cleanup failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    server_name = list(self.connectors.keys())[i]
                    logger.error(f"Failed to cleanup server {server_name}: {result}")
        
        # Clear references
        self.sessions.clear()
        self.clients.clear()
        logger.info("All MCP server connections closed")
    
    async def _cleanup_http_server(self, server_name: str, connector):
        """Helper method to cleanup individual HTTP server with error handling."""
        try:
            await connector.stop_http_server()
        except Exception as e:
            logger.error(f"ERROR in cleaning up HTTP server {server_name}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    async def _cleanup_sse_server(self, server_name: str, connector):
        """Helper method to cleanup individual SSE server with error handling."""
        try:
            await connector.stop_sse_server()
        except Exception as e:
            logger.error(f"ERROR in cleaning up SSE server {server_name}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    async def _cleanup_url_server(self, server_name: str, connector):
        """Helper method to cleanup individual URL server with error handling."""
        try:
            await connector.stop_url_server()
        except Exception as e:
            logger.error(f"ERROR in cleaning up URL server {server_name}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise