"""Phase 9.2.2: tool_choice for one vs several planned tools (offline; fake SDK, the fake campus MCP server via stdio).

The real A-E comparison through the current relay (local/p921/probe-results.jsonl): two tools + "required" -> HTTP 502
(A, D, E, reproducible); two tools + "auto" -> 200 with both tool calls (B); one tool + "required" -> 200 (C). Where
in the relay chain the 502 comes from is not known. Violet now forces the first call only when exactly one planned tool
is offered; with several it uses "auto", and when the model answers having looked up only some (or none) of them it
is told once which are missing: call them, or say that part was not looked up -- never fill it in."""

from __future__ import annotations

import pytest

from tests.fake_mcp import campus_server, not_started
from tests.fakes import completion, intent_json
from tests.test_phase2_scenarios import make_world, say  # noqa: F401  (make_world: fixture)
from tests.test_tools_p92 import fresh_registry, offered, system, world  # noqa: F401  (fresh_registry: fixture)


@pytest.fixture(autouse=True)
def _planned_tools_only(monkeypatch):
    """These tests describe the tools offered only when Intent plans them (VIOLET_CHAT_TOOLS_DIRECT=0); the default
    since 2026-10-30 (every read-only tool on every turn) is tested in tests/test_chat_polish_violet.py."""
    monkeypatch.setenv("VIOLET_CHAT_TOOLS_DIRECT", "0")

SCHEDULE, WEATHER, SEARCH = "campus__get_course_schedule", "campus__get_weather", "campus__search_web"
QUESTION = "周三下午要是没课，我想去西湖边走走。帮我查查那天下午有没有课，天气适不适合出门？"
ARGS = {SCHEDULE: {"day": "周三"}, WEATHER: {"city": "杭州"}, SEARCH: {"query": "西湖 周三"}}


def asks_for(*tools):
    return lambda kw: intent_json("chat", tool_query={"needed": True, "tools": list(tools), "reason": "周三下午"})


def tool_results(kw) -> list[str]:
    return [m["content"] for m in kw["messages"] if m["role"] == "tool"]


def reminded(kw) -> bool:
    return any(m["role"] in ("system", "user") and (m.get("content") or "").startswith("【还没查】") for m in kw["messages"])


def call(*names):
    return completion(tool_calls=[{"name": n, "arguments": ARGS[n]} for n in names])


def honest_answer(kw) -> str:
    text = "\n".join(tool_results(kw))
    parts = ["周三下午有体育课" if "体育" in text else "课表这次没查到",
             "那天可能下小雨" if "小雨" in text else "天气这次没查到"]
    return "；".join(parts)


def mcp_calls() -> dict[str, int]:
    from ai.mcp import get_health_snapshot
    return {h["server"]: h["total_calls"] for h in get_health_snapshot()}


def test_two_tools_one_after_the_other_need_no_reminder(make_world, monkeypatch):
    def chat(kw):
        done = tool_results(kw)
        return call(SCHEDULE) if not done else call(WEATHER) if len(done) == 1 else honest_answer(kw)
    w, chats = world(make_world, monkeypatch, campus_server(), intent=asks_for(SCHEDULE, WEATHER), chat=chat)
    say(w, QUESTION)
    assert [kw["tool_choice"] for kw in chats] == ["auto", "auto", "auto"] and not any(reminded(kw) for kw in chats)
    assert w.client.sent_texts[-1] == "周三下午有体育课；那天可能下小雨"


def test_a_failed_tool_counts_as_looked_up_and_is_reported_as_failed(make_world, monkeypatch):
    chat = lambda kw: call(SCHEDULE, WEATHER) if not tool_results(kw) else honest_answer(kw)
    w, chats = world(make_world, monkeypatch, campus_server(mode="error"), intent=asks_for(SCHEDULE, WEATHER),
                     chat=chat)
    say(w, QUESTION)
    assert len(chats) == 2 and not reminded(chats[1])
    assert all('"status": "tool_error"' in r for r in tool_results(chats[1]))
    assert w.client.sent_texts[-1] == "课表这次没查到；天气这次没查到"

