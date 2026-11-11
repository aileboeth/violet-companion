"""Phase 9.2: unified tool management and Violet's own use of MCP tools (offline).

The fake campus MCP server (tools/fake_mcp_server.py) runs as a real stdio child process with FICTIONAL data; the
models are the fake SDK. What is proven: which tools are offered to which agent and when, what is called, what the
reply is built from, that nothing is invented or turned into a task when a tool is missing / off / down, and that the
Control Center's switches act on the running system. Whether a real model picks tools well is not provable offline
(the real-model plan is in tools/TESTING-HISTORY.md §23)."""

from __future__ import annotations

import json
import time

import pytest

from tests.fake_mcp import campus_server, not_started, use
from tests.fakes import completion, intent_json


@pytest.fixture(autouse=True)
def fresh_registry(tmp_path, monkeypatch):
    import ai.mcp as mcp
    import ai.tool_manager as tm
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    tm._STORE = None
    mcp.reset_registry()
    yield
    mcp.reset_registry()
    tm._STORE = None


def run(coro):
    import asyncio
    return asyncio.run(coro)


# ----------------------------------------------------------------------------------------------- the layer

def test_remote_servers_must_use_https_and_local_ones_need_no_network():
    from ai.mcp import MCPToolBridge, remote_problem
    assert remote_problem({"url": "http://192.168.1.20:8787/mcp"}) == "a remote MCP server must use https"
    assert remote_problem({"url": "https://mcp.example.com/mcp"}) is None
    assert remote_problem({"url": "http://127.0.0.1:8787/mcp"}) is None
    assert remote_problem({"transport": "stdio", "command": ["x"]}) is None
    bridge = MCPToolBridge([{"name": "lan", "url": "http://192.168.1.20:8787/mcp", "tools": {"a": "read"}}])
    run(bridge.connect())
    assert bridge.tools == [] and bridge.unavailable[0]["error_code"] == "MCP_CONFIG_ERROR"


def test_errors_reach_the_model_as_a_category_without_addresses_paths_or_secrets():
    from ai.mcp import MCPErrorCode, safe_error_text, tool_error
    text = safe_error_text("ConnectError at http://10.0.0.8:9000/x?key=abc, file C:\\violet\\data\\a.db, host 172.16.0.3")
    assert "10.0.0.8" not in text and "172.16.0.3" not in text and "C:\\violet" not in text and "key=abc" not in text
    for code, status in ((MCPErrorCode.CONNECT, "unavailable"), (MCPErrorCode.TOOL, "tool_error"),
                         (MCPErrorCode.DENIED, "permission_denied"), (MCPErrorCode.UNKNOWN_TOOL, "not_configured"),
                         (MCPErrorCode.TIMEOUT, "timeout"), (MCPErrorCode.AUTH, "permission_denied")):
        payload = json.loads(tool_error(code, "boom at /opt/violet/x", "s", "t"))
        assert payload["ok"] is False and payload["status"] == status and "不要编造" in payload["note"]
        assert "/opt/violet" not in payload["message"]


def test_settings_persist_apply_at_once_and_never_show_a_credential(tmp_path):
    from ai.tool_manager import ToolSettingsStore, apply_tool_settings, tool_store
    store = tool_store()
    with pytest.raises(ValueError):
        store.save_server("phone", {"url": "http://evil.example/mcp"})          # address / command: .env only
    with pytest.raises(ValueError):
        store.save_server("phone", {"agents": ["memory"]})
    store.save_server("phone", {"enabled": False, "call_timeout": 12, "offer": "always"})
    store.set_tool("phone", "battery", False)
    result = store.replace_secret("phone", "Authorization", "Bearer new-secret-token-123")
    assert "new-secret" not in json.dumps(result) and result["is_set"] and len(result["fingerprint"]) == 8
    servers = apply_tool_settings([{"name": "phone", "url": "http://127.0.0.1:8787/mcp",
                                    "headers": {"Authorization": "Bearer old"}}, {"name": "other", "url": "x"}])
    assert servers[0]["enabled"] is False and servers[0]["call_timeout"] == 12 and servers[0]["offer"] == "always"
    assert servers[0]["disabled_tools"] == ["battery"] and servers[0]["headers"]["Authorization"] == "Bearer new-secret-token-123"
    assert servers[1] == {"name": "other", "url": "x"}
    again = ToolSettingsStore(tmp_path / "data")                                 # "after a restart"
    assert again.servers()["phone"]["enabled"] is False
    assert "new-secret" not in again.history_path.read_text(encoding="utf-8")
    assert "new-secret" not in again.path.read_text(encoding="utf-8")


def test_failures_are_classified_and_a_dead_server_is_not_retried_in_the_same_turn():
    from ai.mcp import MCPToolBridge, get_health_snapshot, tool_turn

    async def call(server, tool, args):
        async with MCPToolBridge([server], agent_name="chat") as bridge:
            return json.loads(await bridge.call_tool(tool, args)) if bridge.tools else None
    with tool_turn():
        error = run(call(campus_server(mode="error", name="err"), "err__get_weather", {"city": "示例市"}))
    assert error["status"] == "tool_error" and "10.0.0.8" not in error["message"]
    with tool_turn():
        crashed = run(call(campus_server(mode="crash", name="crash"), "crash__get_weather", {"city": "示例市"}))
        again = run(call(campus_server(mode="crash", name="crash"), "crash__get_weather", {"city": "示例市"}))
    assert crashed["status"] == "unavailable"
    assert again is None or again["error_code"] == "MCP_FAILED_THIS_TURN"       # not started again in this turn
    calls = {h["server"]: h["total_calls"] for h in get_health_snapshot()}
    assert calls["crash"] == 2                                                  # one call + one reconnect, then stop
    with tool_turn():
        start = time.monotonic()
        slow = run(call(campus_server(mode="slow", name="slow", call_timeout=2), "slow__get_weather", {"city": "示例市"}))
    assert slow["status"] in ("timeout", "unavailable") and time.monotonic() - start < 15


def test_after_a_failed_call_the_next_call_of_the_turn_is_not_sent():
    from ai.mcp import MCPToolBridge, get_health_snapshot, tool_turn

    async def two_calls():
        async with MCPToolBridge([campus_server(mode="crash", name="crash")], agent_name="chat") as bridge:
            first = json.loads(await bridge.call_tool("crash__get_weather", {"city": "示例市"}))
            second = json.loads(await bridge.call_tool("crash__get_weather", {"city": "示例市"}))
            return first, second
    with tool_turn():
        first, second = run(two_calls())
    assert first["status"] == "unavailable" and second["error_code"] == "MCP_FAILED_THIS_TURN"
    assert {h["server"]: h["total_calls"] for h in get_health_snapshot()}["crash"] == 2      # nothing more was sent


def test_bad_arguments_are_refused_before_anything_is_sent_and_a_stdio_server_needs_a_command():
    from ai.mcp import MCPToolBridge, get_health_snapshot

    async def bad_call():
        async with MCPToolBridge([campus_server()], agent_name="chat") as bridge:
            return json.loads(await bridge.call_tool("campus__get_course_schedule", {"day": 3}))
    result = run(bad_call())
    assert result["status"] == "invalid_arguments" and "day" in result["message"]
    assert {h["server"]: h["total_calls"] for h in get_health_snapshot()}["campus"] == 0
    bridge = MCPToolBridge([{"name": "nocmd", "transport": "stdio", "tools": {"a": "read"}}])
    run(bridge.connect())
    assert bridge.unavailable[0]["error_code"] == "MCP_CONFIG_ERROR" and "command" in bridge.unavailable[0]["message"]


# ----------------------------------------------------------------------------------------------- whole turns

def world(make_world, monkeypatch, *servers, intent=None, chat=None):
    use(monkeypatch, *servers)
    w = make_world()
    if intent is not None:
        w.llm.responders["fake-intent"] = intent
    chats: list[dict] = []

    def chat_responder(kw):
        chats.append(kw)
        return chat(kw) if chat else "嗯嗯"
    w.llm.responders["fake-chat"] = chat_responder
    return w, chats


def asks_for(*tools, missing=None):
    query = {"needed": True, "tools": list(tools), "reason": "用户想知道的"}
    if missing:
        query["missing"] = missing
    return lambda kw: intent_json("chat", tool_query=query)


def look_up_then_answer(tool, args):
    def respond(kw):
        results = [m["content"] for m in kw["messages"] if m["role"] == "tool"]
        if not results:
            return completion(tool_calls=[{"name": tool, "arguments": args}])
        return "查到啦：" + tool_data(results[-1])[:200]
    return respond


def tool_data(message: str) -> str:
    """What a model takes from a tool message: the data, not Violet's 【外部工具数据…】 wrapper lines (phase 10: the
    outbound guard refuses a reply that repeats the wrapper)."""
    return "\n".join(line for line in message.splitlines() if not line.startswith("【")).strip()


def offered(kw) -> list[str]:
    return [t["function"]["name"] for t in kw.get("tools") or []]


def system(kw) -> str:
    return "\n".join(m.get("content") or "" for m in kw["messages"] if m["role"] == "system")


def test_the_connection_dropping_mid_call_ends_in_an_honest_reply(make_world, monkeypatch):
    from tests.test_phase2_scenarios import say
    w, chats = world(make_world, monkeypatch, campus_server(mode="crash"), intent=asks_for("campus__get_weather"),
                     chat=look_up_then_answer("campus__get_weather", {"city": "示例市"}))
    say(w, "明天要不要带伞")
    result = json.loads(next(m["content"] for m in chats[-1]["messages"] if m["role"] == "tool"))
    assert result["status"] == "unavailable" and "不要编造" in result["note"]
    assert w.client.sent_texts and w.db.list_task_sessions() == []


def test_a_tool_error_is_a_failed_query_not_a_result(make_world, monkeypatch):
    from tests.test_phase2_scenarios import say
    w, chats = world(make_world, monkeypatch, campus_server(mode="error"), intent=asks_for("campus__get_weather"),
                     chat=look_up_then_answer("campus__get_weather", {"city": "示例市"}))
    say(w, "明天天气怎么样")
    tool_text = next(m["content"] for m in chats[-1]["messages"] if m["role"] == "tool")
    assert json.loads(tool_text)["status"] == "tool_error" and "10.0.0.8" not in tool_text and "/internal" not in tool_text


def test_a_replaced_credential_is_never_shown(make_world, monkeypatch, admin):
    from tests.test_control_center_p90 import SAME
    use(monkeypatch, {"name": "phone", "url": "http://127.0.0.1:1/mcp", "headers": {"Authorization": "Bearer old-token-1"},
                      "tools": {"battery": "read"}})
    assert admin.post("/api/control/tools/servers/phone/secret", headers=SAME,
                      json={"header": "Authorization", "value": "Bearer new-token-2"}).status_code == 400   # no confirm
    resp = admin.post("/api/control/tools/servers/phone/secret", headers=SAME,
                      json={"header": "Authorization", "value": "Bearer new-token-2", "confirm": True})
    assert resp.status_code == 200
    page = admin.get("/api/control/tools", headers=SAME).get_data(as_text=True)
    assert "new-token-2" not in page and "old-token-1" not in page and "fingerprint" in page
    from ai.chat import get_mcp_servers
    assert get_mcp_servers()[0]["headers"]["Authorization"] == "Bearer new-token-2"





from tests.test_control_center_p90 import admin, app, env_guard, made  # noqa: E402,F401  (fixtures)
from tests.test_phase2_scenarios import make_world  # noqa: E402,F401  (fixture)


@pytest.fixture(autouse=True)
def _planned_tools_only(monkeypatch):
    """These tests describe the tools offered only when Intent plans them (VIOLET_CHAT_TOOLS_DIRECT=0); the default
    since 2026-10-30 (every read-only tool on every turn) is tested in tests/test_chat_polish_violet.py."""
    monkeypatch.setenv("VIOLET_CHAT_TOOLS_DIRECT", "0")
