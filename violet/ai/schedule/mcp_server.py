"""Phase 9.4.2: the built-in, read-only timetable MCP server "violet-schedule" (in-process, like violet-web).

Mounted by ai/mcp.py (BUILTIN_SERVER_FACTORIES) when VIOLET_SCHEDULE_ENABLED=1; goes through the same allowlist
(mcp_tool_policy.json profile "violet-schedule": four "read" tools), per-agent assignment (chat only), on-demand offer,
timeouts, circuit breaker and untrusted-output wrapping as every other MCP server. It reads the user's own
timetable from Violet's database through ScheduleStore -- it never writes and never copies the data anywhere.

The first sentence of each description is what IntentAgent sees in its tool catalog."""
from __future__ import annotations

import json
import logging

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ai.schedule.tools import ScheduleTools

SERVER_NAME = "violet-schedule"
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
logger = logging.getLogger("violet.schedule")
DETAILS_DOC = " details 仅对方本轮明确问地点/老师/座位时传 [\"location\"]/[\"teacher\"]/[\"seat\"]，否则留空。"
DESCRIPTIONS = {
    "get_day_schedule": (
        "查用户某天的课和考试。date 支持 today/tomorrow/YYYY-MM-DD/周一…周日。"
        "返回当天安排与 data_state（区分没课/未录课表/学期未开始或已结束）。" + DETAILS_DOC),
    "get_schedule_range": (
        "查用户一段日子（1–14 天）的课表和考试。start_date 支持 today/tomorrow/YYYY-MM-DD/this_week/next_week。"
        + DETAILS_DOC),
    "get_current_status": (
        "按课表推算现在是否上课、下课时间及剩余课程（仅为计划，以用户亲口说的为准）。" + DETAILS_DOC),
    "find_course": (
        "按课名简称或老师名查当前学期接下来 28 天内的课和考试（最多 5 条；查不到勿编造）。" + DETAILS_DOC),
}


def build_server(tools: ScheduleTools) -> FastMCP:
    server = FastMCP(SERVER_NAME)

    async def run(fn):
        try:
            result = await anyio.to_thread.run_sync(fn)
        except Exception:
            logger.exception("[schedule] lookup failed")
            result = {"ok": False, "error_code": "schedule_error", "message": "课表查询失败"}
        return json.dumps(result, ensure_ascii=False)

    @server.tool(annotations=READ_ONLY, description=DESCRIPTIONS["get_day_schedule"])
    async def get_day_schedule(date: str = "today", details: list[str] | None = None) -> str:
        return await run(lambda: tools.day(date, details))

    @server.tool(annotations=READ_ONLY, description=DESCRIPTIONS["get_schedule_range"])
    async def get_schedule_range(start_date: str = "today", days: int = 7, details: list[str] | None = None) -> str:
        return await run(lambda: tools.range(start_date, days, details))

    @server.tool(annotations=READ_ONLY, description=DESCRIPTIONS["get_current_status"])
    async def get_current_status(details: list[str] | None = None) -> str:
        return await run(lambda: tools.status(details))

    @server.tool(annotations=READ_ONLY, description=DESCRIPTIONS["find_course"])
    async def find_course(query: str, details: list[str] | None = None) -> str:
        return await run(lambda: tools.find(query, details))

    return server


def build_default_server() -> FastMCP:
    """Factory used by the MCP bridge: Violet's own database, Violet's own clock."""
    from ai.chat import violetdb
    from ai.schedule.store import ScheduleStore
    return build_server(ScheduleTools(ScheduleStore(violetdb)))
