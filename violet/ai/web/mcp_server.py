"""Built-in web MCP server: web_search and fetch_url (phase 5B).

A standard MCP server (FastMCP). Violet mounts it in-process through the
existing MCP bridge (transport "inprocess": no resident service, no
subprocess); it goes through the same allowlist (mcp_tool_policy.json,
profile "violet-web"), per-agent assignment, timeouts, circuit breaker and
untrusted-output wrapping as any other MCP server.

It can also be run standalone for other MCP clients:
    python -m ai.web.mcp_server            (stdio; no database; daily budgets persist in a small state file)
"""

from __future__ import annotations

import json

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ai.web.service import WebService

SERVER_NAME = "violet-web"
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True)


def build_server(service: WebService) -> FastMCP:
    server = FastMCP(SERVER_NAME)

    @server.tool(annotations=READ_ONLY)
    async def web_search(query: str, recency_days: int | None = None, limit: int = 5) -> str:
        """联网搜索最新消息、天气、实时事件或外部资料（闲聊别搜；query 用简短关键词，不含隐私）。"""
        result = await anyio.to_thread.run_sync(            # 2026-10-30: the first pages read too (search_and_read)
            lambda: service.search_and_read(query, origin="chat", recency_days=recency_days, limit=limit))
        return json.dumps(result, ensure_ascii=False)

    @server.tool(annotations=READ_ONLY)
    async def fetch_url(url: str) -> str:
        """读取公开网页正文（用户发来的或本轮搜索返回的链接；拒绝内网地址）。"""
        result = await anyio.to_thread.run_sync(lambda: service.fetch(url, origin="chat"))
        return json.dumps(result, ensure_ascii=False)

    return server


def build_default_server() -> FastMCP:
    """Factory used by the MCP bridge: shares Violet's database for findings, logs and budgets."""
    from ai.chat import violetdb
    return build_server(WebService(violetdb))


def main() -> None:  # pragma: no cover - manual use
    build_server(WebService(None)).run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
