"""Phase 9.2.1: what the real test p92-tools-1 left open (offline; fake SDK, the fake campus MCP server as a stdio child).

#11  Chat was offered two tools (tool_choice=required) and the relay answered HTTP 502 three times; no MCP call ran.
     Proven here: the request Violet builds for two tools, and that Chat runs two tool calls -- in one response or one
     after the other -- and answers from both results; what a 502 does (retries, backoff, fallback reply, memory).
     Whether the relay / upstream rejects that request shape is NOT provable offline (docs/handoff/log, p921).
#9   A switched-off tool was reported to Chat as "not configured": Intent's catalog never listed it, so Intent could only
     write tools=[] + missing. Fixed: the catalog lists switched-off tools and the full names of unreachable ones."""

from __future__ import annotations

import json
import re

import httpx
import openai
import pytest

from tests.fake_mcp import campus_server, not_started
from tests.fakes import completion, intent_json
from tests.test_phase2_scenarios import make_world  # noqa: F401  (fixture)
from tests.test_tools_p92 import fresh_registry, offered, system, world  # noqa: F401  (fresh_registry: fixture)


@pytest.fixture(autouse=True)
def _planned_tools_only(monkeypatch):
    """These tests describe the tools offered only when Intent plans them (VIOLET_CHAT_TOOLS_DIRECT=0); the default
    since 2026-10-30 (every read-only tool on every turn) is tested in tests/test_chat_polish_violet.py."""
    monkeypatch.setenv("VIOLET_CHAT_TOOLS_DIRECT", "0")

QUESTION = "周三下午要是没课，我想去西湖边走走。帮我查查那天下午有没有课，天气适不适合出门？"
BOTH = ("campus__get_course_schedule", "campus__get_weather")


def asks_for_both(kw):
    return intent_json("chat", tool_query={"needed": True, "tools": list(BOTH), "reason": "周三下午的课和杭州的天气"})


def tool_messages(kw):
    return [m for m in kw["messages"] if m["role"] == "tool"]


def answer_from(kw):
    results = "\n".join(m["content"] for m in tool_messages(kw))
    has_class = "体育" in results
    rain = "小雨" in results
    return f"周三下午{'有体育课，15:50 开始' if has_class else '没课'}；{'那天可能下小雨，记得带伞' if rain else '天气还行'}"


def bad_gateway():
    response = httpx.Response(502, request=httpx.Request("POST", "http://relay.invalid/v1/chat/completions"))
    return openai.InternalServerError(
        "Error code: 502 - {'error': {'message': 'Upstream service temporarily unavailable', 'type': 'upstream_error'}}",
        response=response, body={"error": {"message": "Upstream service temporarily unavailable"}})


def intent_reading_the_catalog(tool_suffix, missing):
    """An Intent that can only name tools its catalog shows (the fake used before named tools it could not see)."""
    def respond(kw):
        names = re.findall(r"campus__\w+", system(kw))
        pick = [n for n in dict.fromkeys(names) if n.endswith(tool_suffix)]
        query = {"needed": True, "tools": pick, "reason": "想知道"} if pick else {"needed": True, "tools": [], "missing": missing}
        return intent_json("chat", tool_query=query)
    return respond

