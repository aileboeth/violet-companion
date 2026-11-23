"""2026-11-01 capability rework (scenario eval on the server: Intent took 7-22 s and ~7.7k tokens per message and
marked voice / pictures "not configured"): the owner's turn without the Intent call, Chat's own memory lookup, the
abilities list naming her voice."""

from __future__ import annotations

import json

import pytest

from tests.fakes import FakeLLM, completion, default_responders, install_fake_llm


@pytest.fixture()
def agents(monkeypatch, violetdb):
    monkeypatch.setenv("VIOLET_INTENT_STEP", "0")
    fake = install_fake_llm(monkeypatch, FakeLLM(default_responders()))
    from ai.agents import ChatAgent, MemoryAgent
    return fake, {"chat": ChatAgent(), "memory": MemoryAgent()}


def _run(a, text):
    from ai.chat import run_pipeline
    return run_pipeline(user_message=text, chat_agent=a["chat"],
                        memory_agent=a["memory"])


def _tool_names(call):
    return [t["function"]["name"] for t in call.get("tools") or []]


def test_the_turn_has_no_intent_call(agents):
    fake, a = agents
    result = _run(a, "今天好累")
    assert result["chat"]["content"] == "好呀\n我在听" and not result["chat_failed"]
    assert not fake.calls_for("fake-intent") and len(fake.calls_for("fake-chat")) == 1
    assert "intent" not in result["errors"]


def test_chat_looks_memories_up_itself(agents, violetdb):
    fake, a = agents
    violetdb.add_event_memory("用户下周三有高数考试", category="event")
    answers = iter([completion(tool_calls=[{"name": "query_memory", "arguments": {"query": "高数 考试"}}]),
                    completion("你周三考高数吧 加油")])
    fake.responders["fake-chat"] = lambda kw: next(answers)
    result = _run(a, "你还记得我说的考试吗")
    first = fake.calls_for("fake-chat")[0]
    assert "query_memory" in _tool_names(first)
    assert result["chat"]["content"] == "你周三考高数吧 加油"
    assert any("高数" in (m.get("content") or "") for m in result["recalled_memories"])


def test_the_abilities_list_names_her_voice(monkeypatch, violetdb):
    from ai import voice
    from ai.tool_manager import Catalog, abilities_note
    monkeypatch.setenv("VIOLET_INTENT_STEP", "0")
    monkeypatch.setattr(voice, "ready", lambda: True)
    note = abilities_note(Catalog(), [])
    assert "- 语音：可以" in note and "query_memory" in note
    monkeypatch.setattr(voice, "ready", lambda: False)
    assert "发不了语音" in abilities_note(Catalog(), [])


# ------------------------------------------------------------------------------------------------ prompt caching

def test_cache_hints_per_provider(monkeypatch):
    import uuid
    from ai import model_params as mp
    msgs = [{"role": "system", "content": "规则", mp.CACHE_MARK: True}, {"role": "user", "content": "你好"}]
    gpt = mp.apply({"model": "gpt-6-luna", "messages": msgs, "cache_key": "violet-ChatAgent"})
    assert gpt["extra_body"] == {"prompt_cache_key": "violet-ChatAgent"} and "cache_key" not in gpt
    assert gpt["messages"][0] == {"role": "system", "content": "规则"}                  # the mark never leaves
    claude = mp.apply({"model": "claude-opus-5-5", "messages": msgs, "cache_key": "k"})
    assert claude["messages"][0]["content"] == [{"type": "text", "text": "规则", "cache_control": {"type": "ephemeral"}}]
    assert "extra_body" not in claude and claude["messages"][1] == {"role": "user", "content": "你好"}
    grok = mp.apply({"model": "grok-4.6", "messages": msgs, "cache_key": "violet-ChatAgent"})
    assert "cache_key" not in grok and grok["messages"][0] == {"role": "system", "content": "规则"}
    cid = grok["extra_headers"]["x-grok-conv-id"]
    assert uuid.UUID(cid).version == 5
    assert grok["extra_headers"] == {
        "x-grok-conv-id": cid, "session_id": cid, "conversation_id": cid, "x-session-id": cid
    }
    assert grok["extra_body"] == {"prompt_cache_key": cid}
    # deterministic across calls, isolated across different installations (API keys)
    grok_again = mp.apply({"model": "grok-4.6", "messages": msgs, "cache_key": "violet-ChatAgent"})
    assert grok_again["extra_headers"]["x-grok-conv-id"] == cid
    grok_other_agent = mp.apply({"model": "grok-4.6", "messages": msgs, "cache_key": "violet-MemoryAgent"})
    assert grok_other_agent["extra_headers"]["x-grok-conv-id"] != cid
    monkeypatch.setenv("LLM_API_KEY", "different-install-key")
    grok_other_install = mp.apply({"model": "grok-4.6", "messages": msgs, "cache_key": "violet-ChatAgent"})
    assert grok_other_install["extra_headers"]["x-grok-conv-id"] != cid
    for model in ("deepseek-chat", "qwen-max"):
        plain = mp.apply({"model": model, "messages": msgs, "cache_key": "k"})
        assert "extra_body" not in plain and "extra_headers" not in plain
        assert plain["messages"][0] == {"role": "system", "content": "规则"}
    monkeypatch.setenv("VIOLET_LLM_PROVIDER", "claude")                                # chosen on the page
    assert isinstance(mp.apply({"model": "my-relay-model", "messages": msgs})["messages"][0]["content"], list)
    monkeypatch.setenv("VIOLET_LLM_PROVIDER", "grok")
    custom_grok = mp.apply({"model": "my-relay-model", "messages": msgs, "cache_key": "k"})
    assert "x-grok-conv-id" in custom_grok["extra_headers"] and "prompt_cache_key" in custom_grok["extra_body"]


def test_grok_refused_prompt_cache_key_keeps_headers(monkeypatch):
    """Direct xAI / strict relays that reject prompt_cache_key in the body still keep x-grok-conv-id & session_id."""
    import httpx
    import openai
    from ai import model_params as mp
    sent = []

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    sent.append(kw)
                    if "extra_body" in kw:
                        response = httpx.Response(400, request=httpx.Request("POST", "http://127.0.0.1:9/v1"))
                        raise openai.BadRequestError("Extra inputs are not permitted", response=response, body=None)
                    return completion("好")

    request = {"model": "grok-strict-endpoint", "messages": [{"role": "user", "content": "hi"}], "cache_key": "violet-ChatAgent"}
    assert mp.create(Client, request).choices[0].message.content == "好"
    assert len(sent) == 2
    assert "extra_body" in sent[0] and "x-grok-conv-id" in sent[0]["extra_headers"]
    assert "extra_body" not in sent[1] and "x-grok-conv-id" in sent[1]["extra_headers"]
    mp.create(Client, request)
    assert len(sent) == 3 and "extra_body" not in sent[2] and "x-grok-conv-id" in sent[2]["extra_headers"]


def test_standing_memories_order_stays_stable_across_turns(violetdb, monkeypatch):
    """Matching a different word in user_message / phone_note must not shuffle the same selected standing facts
    inside the cached system prompt."""
    from ai.context_engine import ContextEngine
    monkeypatch.setenv("VIOLET_MEMORY_STANDING_FACTS", "5")
    violetdb.add_event_memory("用户喜欢喝冰美式咖啡", kind="fact", importance=3.0)
    violetdb.add_event_memory("用户养了一只叫团子的橘猫", kind="fact", importance=2.0)
    violetdb.add_event_memory("用户周末常去打羽毛球", kind="fact", importance=1.0)
    engine = ContextEngine(violetdb)
    order_a = [f["content"] for f in engine.standing_memories("今天去打羽毛球")[0]]
    order_b = [f["content"] for f in engine.standing_memories("刚给橘猫团子喂饭")[0]]
    order_c = [f["content"] for f in engine.standing_memories("")[0]]
    assert order_a == order_b == order_c


def test_standing_section_prefix_stays_identical_with_more_than_five_facts(violetdb, monkeypatch):
    """With >5 standing facts, per-turn _user_message / phone_note must never change which 5 facts enter the
    stable (cached) system prompt or their order."""
    from ai.context_engine import ContextEngine
    monkeypatch.setenv("VIOLET_MEMORY_STANDING_FACTS", "5")
    for i, (text, imp) in enumerate([
        ("用户喜欢喝冰美式咖啡", 5.0),
        ("用户养了一只叫团子的橘猫", 4.5),
        ("用户周末常去打羽毛球", 4.0),
        ("用户是计算机专业研究生", 3.5),
        ("用户蓝牙连的是手表不是耳机", 3.0),
        ("用户平时爱玩塞尔达传说", 2.0),
        ("用户对海鲜轻微过敏", 1.5),
        ("用户喜欢在下雨天听爵士乐", 1.0),
    ]):
        violetdb.add_event_memory(text, kind="fact", importance=imp)
    engine = ContextEngine(violetdb)
    sec_a, ids_a = engine._standing_section({"_user_message": "昨晚打塞尔达打到两点", "phone_note": "在玩塞尔达"})
    sec_b, ids_b = engine._standing_section({"_user_message": "海鲜过敏又犯了，听听爵士乐", "phone_note": "连着蓝牙手表"})
    sec_c, ids_c = engine._standing_section({})
    assert sec_a.render() == sec_b.render() == sec_c.render()
    assert ids_a == ids_b == ids_c and len(ids_a) == 5
    # Verify stable (non-volatile) sections from build() are byte-for-byte identical across turns
    pkt_1 = engine.build("chat", "昨晚打塞尔达打到两点", {"_user_message": "昨晚打塞尔达打到两点"})
    pkt_2 = engine.build("chat", "海鲜过敏又犯了", {"_user_message": "海鲜过敏又犯了"})
    stable_1 = [s for s in pkt_1.sections if s not in pkt_1.volatile]
    stable_2 = [s for s in pkt_2.sections if s not in pkt_2.volatile]
    assert stable_1 == stable_2


def test_agent_cache_key_scopes_mode_and_persona(violetdb):
    from ai import memory_scope
    from ai.agents.base_agent import _CALL_CONTEXT
    from ai.agents.chat_agent import ChatAgent
    from ai.agents.memory_agent import MemoryAgent
    chat = ChatAgent()
    mem = MemoryAgent()
    assert chat._cache_key() == "violet-ChatAgent"
    assert mem._cache_key() == "violet-MemoryAgent"
    tok = _CALL_CONTEXT.set({"is_proactive": True})
    try:
        assert chat._cache_key() == "violet-ChatAgent-proactive"
    finally:
        _CALL_CONTEXT.reset(tok)
    with memory_scope.use("stepsister"):
        assert chat._cache_key() == "violet-ChatAgent-stepsister"
        assert mem._cache_key() == "violet-MemoryAgent-stepsister"
        tok2 = _CALL_CONTEXT.set({"is_proactive": True})
        try:
            assert chat._cache_key() == "violet-ChatAgent-stepsister-proactive"
        finally:
            _CALL_CONTEXT.reset(tok2)
    # MemoryAgent merges its leading system instructions into one system block with a Claude cache breakpoint
    tok3 = _CALL_CONTEXT.set({})
    try:
        mem_msgs = mem.build_context_messages("今天吃了拉面")
    finally:
        _CALL_CONTEXT.reset(tok3)
    from ai import model_params as mp
    applied = mp.apply({"model": "claude-opus-5-5", "messages": mem_msgs, "cache_key": "violet-MemoryAgent"})
    sys_msgs = [m for m in applied["messages"] if m["role"] == "system"]
    assert len(sys_msgs) == 1
    assert isinstance(sys_msgs[-1]["content"], list) and sys_msgs[-1]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_system_messages_after_the_conversation_began_go_as_user(monkeypatch):
    """--probe-cache on the user's relay: a system message after the history had 0 cache hits (the relay moves every
    system message to the front); the same text inside a user message was cached."""
    from ai import model_params as mp
    msgs = [{"role": "system", "content": "规则"}, {"role": "system", "content": "人设"},
            {"role": "user", "content": "昨天的话"}, {"role": "system", "content": "当前时间 20:01"},
            {"role": "user", "content": "今天好累"}]
    sent = mp.apply({"model": "gpt-6-luna", "messages": msgs})["messages"]
    assert [m["role"] for m in sent] == ["system", "system", "user", "user", "user"]
    assert sent[3]["content"] == mp.LATE_SYSTEM_LABEL + "当前时间 20:01" and sent[:3] == msgs[:3]
    # no history yet: the fixed part ends at the cache mark, this turn's notes right after it are not system either
    fresh = [{"role": "system", "content": "规则", mp.CACHE_MARK: True}, {"role": "system", "content": "当前时间 20:14"},
             {"role": "user", "content": "早安"}]
    assert [m["role"] for m in mp.apply({"model": "gpt-6-luna", "messages": fresh})["messages"]] == ["system", "user", "user"]


def test_chat_sends_only_the_fixed_part_as_system(agents, monkeypatch, violetdb):
    """With and without history, every system message the model gets is part of the fixed beginning."""
    fake, a = agents
    _run(a, "早安")                                                            # no history
    violetdb.add_message("user", "incoming", "text", "昨天的话")
    _run(a, "今天好累")                                                        # with history
    first, second = fake.calls_for("fake-chat")[-2:]
    for call in (first, second):
        roles = [m["role"] for m in call["messages"]]
        last_system = max(i for i, r in enumerate(roles) if r == "system")
        assert all(r == "system" for r in roles[:last_system + 1])            # one block at the front
        assert not any("当前时间" in str(m["content"]) for m in call["messages"] if m["role"] == "system")
    systems = lambda call: [m for m in call["messages"] if m["role"] == "system"]
    assert systems(first) == systems(second)


def test_a_relay_that_refuses_a_hint_never_gets_it_again(monkeypatch):
    import openai
    from ai import model_params as mp
    sent = []

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    sent.append(kw)
                    if "extra_body" in kw:
                        import httpx
                        response = httpx.Response(400, request=httpx.Request("POST", "http://127.0.0.1:9/v1"))
                        raise openai.BadRequestError("Unrecognized request argument: prompt_cache_key",
                                                     response=response, body=None)
                    return completion("好")
    request = {"model": "gpt-relay-x", "messages": [{"role": "user", "content": "hi"}], "cache_key": "k"}
    assert mp.create(Client, request).choices[0].message.content == "好"
    assert len(sent) == 2 and "extra_body" in sent[0] and "extra_body" not in sent[1]
    mp.create(Client, request)
    assert len(sent) == 3 and "extra_body" not in sent[2]                               # remembered


def test_chat_puts_this_turn_after_the_history(agents, violetdb):
    fake, a = agents
    violetdb.add_message("user", "incoming", "text", "昨天的话")
    violetdb.add_message("assistant", "outgoing", "text", "嗯嗯")
    _run(a, "今天好累")
    messages = fake.calls_for("fake-chat")[-1]["messages"]
    roles = [m["role"] for m in messages]
    first_history = roles.index("user")
    assert messages[-1]["role"] == "user" and messages[-1]["content"].endswith("今天好累")
    # the time (base section) is this turn's: after the history, the persona before it
    time_at = next(i for i, m in enumerate(messages) if m["role"] in ("system", "user") and "当前时间" in (m["content"] or ""))
    persona_at = next(i for i, m in enumerate(messages) if "人设" in str(m.get("content")))
    assert persona_at < first_history < time_at
    assert all("_cache_break" not in m for m in messages)


def test_two_different_turns_begin_the_same(agents, violetdb):
    """Everything before the history is the same from turn to turn (the provider's prompt cache serves it). The second
    server eval: "【你们的相处】第一次聊天在 …（1 分钟前）" sat in front and changed every call -> 0% cache."""
    fake, a = agents
    for text in ("今天好累", "发语音给我听听"):
        violetdb.execute("DELETE FROM messages")
        violetdb.add_message("user", "incoming", "text", "随便聊聊" + text)
        _run(a, text)
    first, second = fake.calls_for("fake-chat")[-2:]

    def front(call):
        out = []
        for m in call["messages"]:
            if m["role"] != "system":
                break
            out.append(m)
        return out
    assert front(first) == front(second) and len(front(first)) == 1
    assert [t["function"]["name"] for t in first.get("tools") or []] == [t["function"]["name"] for t in second.get("tools") or []]


def test_the_history_window_keeps_its_start(monkeypatch, violetdb):
    from ai.context_engine import ContextEngine, _HISTORY_ANCHORS
    monkeypatch.setenv("VIOLET_CONTEXT_STABLE_HISTORY", "1")
    monkeypatch.setenv("VIOLET_CONTEXT_HISTORY_TURNS", "4")
    monkeypatch.setenv("VIOLET_CONTEXT_STYLE", "plain")
    _HISTORY_ANCHORS.clear()

    def say(i):
        violetdb.add_message(*(("user", "incoming") if i % 2 == 0 else ("assistant", "outgoing")), "text", f"话{i}")
    for i in range(6):
        say(i)
    engine = ContextEngine(violetdb)
    first = engine.history([])[0]["content"]
    for i in range(6, 9):                                                   # the window would slide by three
        say(i)
        assert engine.history([])[0]["content"] == first                    # ... but keeps its start
    for i in range(9, 14):
        say(i)
    assert engine.history([])[0]["content"] != first                        # twice the size: jumps ahead


def test_the_group_history_keeps_its_start(violetdb):
    from ai.social.engine import SocialEngine
    e = SocialEngine(type("R", (), {"db": violetdb, "channels": None})())
    rows = [{"id": i} for i in range(1, 31)]                               # the engine fetches 2 x size
    assert e._stepped(7, rows[-20:], 10)[0]["id"] == 21
    rows += [{"id": i} for i in range(31, 41)]
    assert e._stepped(7, rows[-20:], 10)[0]["id"] == 21                     # 20 lines now: same start
    rows += [{"id": 41}]
    assert e._stepped(7, rows[-20:], 10)[0]["id"] == 32                     # past twice the size: the last 10 again


def test_a_group_nobody_calls_her_in_is_looked_at_in_batches(violetdb, monkeypatch):
    from ai.social.engine import SocialEngine
    made = []

    class Timer:
        def __init__(self, delay, fn, args=()):
            self.delay, self.daemon = delay, True
            made.append(self)

        def start(self):
            pass

        def cancel(self):
            pass
    e = SocialEngine(type("R", (), {"db": violetdb, "channels": None})(), timer_factory=Timer)
    monkeypatch.setattr(e, "_hot", lambda cid: False)
    e.schedule(5)
    assert made[-1].delay == pytest.approx(180, abs=1)          # others talking: one look in three minutes
    e.schedule(5)
    assert made[-1].delay <= 180                                 # a new message does not push it later
    e.schedule(5, urgent=True)
    assert made[-1].delay <= 3                                   # called by name / @: at once
    monkeypatch.setattr(e, "_hot", lambda cid: True)
    e.schedule(6)
    assert made[-1].delay <= 6                                   # she is in the conversation: keeps up


def test_a_repeated_search_in_one_turn_is_not_run():
    """The three queries the eval saw for "最近有什么科技新闻吗" (2026-11-01)."""
    from ai.web.tool_guard import WebToolGuard
    guard = WebToolGuard("最近有什么科技新闻吗")
    first = "近期 最新 科技新闻 2026年11月 科技 AI 芯片 苹果 谷歌 OpenAI 航天 科技报道"
    assert guard.check("violet-web__web_search", {"query": first}) is None
    again = guard.check("violet-web__web_search", {"query": "2026年11月 最新 科技新闻 AI 芯片 科技媒体"})
    assert json.loads(again)["error_code"] == "SAME_SEARCH"
    assert guard.check("violet-web__web_search", {"query": "杭州 明天 天气"}) is None     # something else: fine

