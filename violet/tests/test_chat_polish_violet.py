"""2026-10-30 rework step 2 ("打磨聊天"): the read-only tools go to Chat on every turn (VIOLET_CHAT_TOOLS_DIRECT), the
token ledger (ai/usage.py) and the overview, the last prompt on the admin page, the persona first for the prefix
cache. Offline; the fake campus MCP server via stdio, fake SDK."""

from __future__ import annotations

from types import SimpleNamespace

from tests.fake_mcp import campus_server
from tests.fakes import completion, intent_json
from tests.test_phase2_scenarios import make_world, say  # noqa: F401  (make_world: fixture)
from tests.test_qq_social_violet import SAME, admin, db  # noqa: F401
from tests.test_tools_p92 import fresh_registry, offered, system, world  # noqa: F401  (fresh_registry: fixture)

SCHEDULE, WEATHER, SEARCH = "campus__get_course_schedule", "campus__get_weather", "campus__search_web"


def tool_results(kw) -> list[str]:
    return [m["content"] for m in kw["messages"] if m["role"] == "tool"]


def test_usage_ledger_numbers_and_summary(violetdb):
    from ai import usage
    from ai.settings import CURRENT_TURN
    violetdb.execute("DELETE FROM llm_usage")
    assert usage.numbers({"prompt_tokens": 1000, "completion_tokens": 50, "prompt_cache_hit_tokens": 800}) == (1000, 50, 800)
    details = SimpleNamespace(prompt_tokens=10, completion_tokens=2, prompt_tokens_details=SimpleNamespace(cached_tokens=4))
    assert usage.numbers(details) == (10, 2, 4) and usage.numbers(None) == (0, 0, 0)
    token = CURRENT_TURN.set("t-1")
    try:
        usage.record("MemoryAgent", "m", {"prompt_tokens": 1000, "completion_tokens": 50, "prompt_cache_hit_tokens": 800},
                     db=violetdb)
        usage.record("ChatAgent", "m", {"prompt_tokens": 3000, "completion_tokens": 150}, db=violetdb)
    finally:
        CURRENT_TURN.reset(token)
    usage.record("social", "m", {"prompt_tokens": 2000, "completion_tokens": 100}, db=violetdb)
    s = usage.summary(violetdb)
    assert s["calls"] == 3 and s["total"] == 6300 and s["cached"] == 800 and s["cache_rate"] == round(800 / 6000, 3)
    assert s["turns"] == 1 and s["per_turn"] == 4200 and s["calls_per_turn"] == 2.0
    assert [p["label"] for p in s["parts"]] == ["回复", "群聊 / 别人私聊", "记忆整理"]
    assert usage.turn(violetdb, "t-1")["sources"] == ["记忆整理", "回复"]
    usage.record("x", "m", object(), db=violetdb)          # odd usage objects never raise
    usage.record("x", "m", {"prompt_tokens": 1}, db=None)


def test_overview_usage_and_last_prompt(admin, violetdb):
    from ai.agents.chat_agent import LAST_PROMPT, remember_prompt
    LAST_PROMPT.clear()
    assert admin.get("/api/control/last-prompt").get_json()["data"] is None
    remember_prompt([{"role": "system", "content": "【规则】\n第一条"}, {"role": "user", "content": "你好"}])
    d = admin.get("/api/control/last-prompt").get_json()["data"]
    assert d["chars"] == len("【规则】\n第一条") + 2 and d["parts"][0]["title"] == "【规则】"
    o = admin.get("/api/control/overview").get_json()["data"]
    assert "usage" in o


def test_persona_comes_first_for_the_prefix_cache(violetdb):
    from ai.context_engine import ContextEngine
    packet = ContextEngine(violetdb).build("chat", "你好")
    names = list(packet.report["sections"])
    assert names and names[0] == "persona_core"
