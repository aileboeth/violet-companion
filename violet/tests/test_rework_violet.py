"""2026-11-11 rework (audit of 0.7.66): one turn = Chat with its tools -> checks -> Memory.

  * a reminder is a tool Chat calls (any wording, any language), not a regex on "提醒我"
  * what she looked up with tools stays with the chat for the next turns
  * a rewrite (consistency / "别老问我") never runs tools again
  * a chat request is ONE system message, the chat, ONE last user message -- also with MCP tools, also for the
    proactive decision
  * what she is shown about the owner does not depend on keywords in the message
  * a group reply that is plain text instead of JSON is sent as text
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from tests.fake_mcp import campus_server, use
from tests.fakes import FakeLLM, completion, default_responders, install_fake_llm
from tests.test_phase2_scenarios import make_world, say  # noqa: F401  (make_world: fixture)
from tests.test_tools_p92 import fresh_registry  # noqa: F401  (fixture)


def _names(call) -> list[str]:
    return [t["function"]["name"] for t in call.get("tools") or []]


def _script(*steps):
    queue = list(steps)
    return lambda kw: queue.pop(0) if len(queue) > 1 else queue[0]


def _tomorrow(db, clock="07:00") -> str:
    day = (datetime.fromisoformat(db.now()) + timedelta(days=1)).strftime("%Y-%m-%d")
    return f"{day} {clock}"


# ------------------------------------------------------------------------------------------------ reminders

@pytest.mark.parametrize("words", ["明早七点叫我起床", "wake me up at 7 tomorrow", "明日の7時に起こして"])
def test_a_reminder_in_any_words_is_set_by_the_tool(make_world, words):
    w = make_world()
    due = _tomorrow(w.db)
    w.llm.responders["fake-chat"] = _script(
        completion(tool_calls=[{"name": "set_reminder", "arguments": {"content": "起床", "due_at": due}}]), "好，到时候叫你")
    say(w, words)
    rows = w.db.list_commitments(["pending"])
    assert [(r["content"], r["due_at"][:16].replace("T", " ")) for r in rows] == [("起床", due)]
    assert rows[0]["evidence"] == words and w.client.sent_texts[-1] == "好，到时候叫你"
    first = w.llm.calls_for("fake-chat")[0]
    assert {"set_reminder", "cancel_reminder"} <= set(_names(first))
    tool = next(m for m in w.llm.calls_for("fake-chat")[-1]["messages"] if m["role"] == "tool")
    assert json.loads(tool["content"])["ok"] is True


def test_a_time_in_the_past_or_unreadable_sets_nothing_and_says_so(violetdb):
    from ai import owner_tools
    past = (datetime.fromisoformat(violetdb.now()) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
    for due in (past, "明天早上", ""):
        out = json.loads(owner_tools.set_reminder(violetdb, {}, {"content": "喝水", "due_at": due}))
        assert out["ok"] is False and out["message"].startswith("没有设")
    assert json.loads(owner_tools.set_reminder(violetdb, {}, {"content": " ", "due_at": _tomorrow(violetdb)}))["ok"] is False
    assert violetdb.list_commitments(["pending"]) == []


def test_the_same_reminder_twice_is_one_and_it_can_be_cancelled_by_its_number(violetdb):
    from ai import owner_tools
    from ai.context_engine import ContextEngine
    due = _tomorrow(violetdb, "09:30")
    context: dict = {"turn_id": "t1"}
    one = json.loads(owner_tools.set_reminder(violetdb, context, {"content": "交报告", "due_at": due}))
    two = json.loads(owner_tools.set_reminder(violetdb, {"turn_id": "t2"}, {"content": "交报告", "due_at": due}))
    assert one["ok"] and two["ok"] and one["reminder_id"] == two["reminder_id"] and "没有重复设" in two["message"]
    assert context["_actions"] == [{"tool": "set_reminder", "commitment_id": one["reminder_id"]}]
    shown = "\n".join(ContextEngine(violetdb).build("chat", "早").sections)
    assert f"#{one['reminder_id']} " in shown and "交报告" in shown                    # the number cancel_reminder takes
    assert json.loads(owner_tools.cancel_reminder(violetdb, {}, {"reminder_id": one["reminder_id"]}))["ok"] is True
    assert violetdb.get_commitment(one["reminder_id"])["status"] == "cancelled"
    assert json.loads(owner_tools.cancel_reminder(violetdb, {}, {"reminder_id": one["reminder_id"]}))["ok"] is False


def test_the_memory_step_does_not_set_the_reminder_a_second_time(make_world):
    w = make_world()
    due = _tomorrow(w.db)
    w.llm.responders["fake-chat"] = _script(
        completion(tool_calls=[{"name": "set_reminder", "arguments": {"content": "起床", "due_at": due}}]), "好")
    w.llm.responders["fake-memory"] = _script(
        completion(tool_calls=[{"name": "create_commitment", "arguments": {
            "content": "叫他起床", "due_time": due.replace(" ", "T") + ":00", "evidence": "明早七点叫我起床"}}]),
        json.dumps({"should_save": False}))
    say(w, "明早七点叫我起床")
    assert len(w.db.fetch_all("SELECT id FROM commitments")) == 1


def test_her_own_messages_and_test_mode_get_no_reminder_tools(monkeypatch, violetdb):
    from ai import test_mode
    from ai.agents import ChatAgent
    llm = install_fake_llm(monkeypatch, FakeLLM(default_responders("嗯")))
    ChatAgent().run("[proactive event]", call_context={"is_proactive": True, "proactive_result": {"action_type": "chat"}})
    assert not {"set_reminder", "cancel_reminder", "query_memory"} & set(_names(llm.calls[-1]))
    test_mode.enable(violetdb, 1)
    try:
        ChatAgent().run("十分钟后叫我")
        assert not {"set_reminder", "cancel_reminder"} & set(_names(llm.calls[-1]))
    finally:
        test_mode.disable()


# ------------------------------------------------------------------------------------------------ tool results stay

def test_what_she_looked_up_is_still_there_on_the_next_turn(make_world, monkeypatch, fresh_registry):  # noqa: F811
    use(monkeypatch, campus_server())
    w = make_world()
    w.llm.responders["fake-chat"] = _script(
        completion(tool_calls=[{"name": "campus__get_weather", "arguments": {"city": "杭州"}}]), "查了，今天多云")
    say(w, "杭州今天天气怎么样")
    stored = w.db.fetch_one("SELECT tool_notes FROM messages WHERE role = 'user' ORDER BY id DESC LIMIT 1")["tool_notes"]
    assert stored and stored.startswith("- get_weather(") and "【外部工具数据" not in stored
    w.llm.responders["fake-chat"] = lambda kw: "明天的没查到"
    say(w, "那明天呢")
    request = w.llm.calls_for("fake-chat")[-1]["messages"]
    mark = next(m["content"] for m in request if m["role"] == "user" and "你那时用工具查到的" in m["content"])
    assert "get_weather" in mark and "〔记录｜不是对方说的话〕" in mark
    assert not any("你那时用工具查到的" in str(m.get("content")) for m in request if m["role"] == "assistant")


def test_small_talk_stores_no_tool_note(make_world):
    w = make_world()
    say(w, "早")
    assert w.db.fetch_one("SELECT tool_notes FROM messages WHERE role = 'user'")["tool_notes"] is None


# ------------------------------------------------------------------------------------------------ rewrite, no tools

def test_a_rewrite_sees_the_draft_and_gets_no_tools(make_world, monkeypatch, fresh_registry):  # noqa: F811
    use(monkeypatch, campus_server())
    w = make_world()
    calls = {"weather": 0}

    def chat(kw):
        last = kw["messages"][-1]
        if last["role"] == "user" and "里面有和已记录的事实对不上的说法" in last["content"]:
            return "今天多云\n想去海边走走"
        if any(m["role"] == "tool" for m in kw["messages"]):
            return "今天多云\n我上周末去海边玩了一整天"
        calls["weather"] += 1
        return completion(tool_calls=[{"name": "campus__get_weather", "arguments": {"city": "杭州"}}])
    w.llm.responders["fake-chat"] = chat
    say(w, "今天天气怎么样")
    assert w.client.sent_texts[-2:] == ["今天多云", "想去海边走走"]
    rewrite = w.llm.calls_for("fake-chat")[-1]
    assert rewrite.get("tools") is None and calls["weather"] == 1                # the lookup ran once
    assert rewrite["messages"][-2] == {"role": "assistant", "content": "今天多云\n我上周末去海边玩了一整天"}
    assert "这一轮工具查到的" in rewrite["messages"][-1]["content"]
    assert [m["role"] for m in rewrite["messages"]].count("system") == 1


# ------------------------------------------------------------------------------------------------ one shape

def test_a_request_with_external_tools_is_still_one_system_message(make_world, monkeypatch, fresh_registry):  # noqa: F811
    from ai.mcp import UNTRUSTED_TOOL_NOTICE
    use(monkeypatch, campus_server())
    w = make_world()
    say(w, "早")
    request = w.llm.calls_for("fake-chat")[-1]["messages"]
    assert [m["role"] for m in request] == ["system", "user"]
    assert UNTRUSTED_TOOL_NOTICE in request[0]["content"] and "【工具怎么用】" in request[0]["content"]
    assert any(n.startswith("campus__") for n in _names(w.llm.calls_for("fake-chat")[-1]))


def test_out_of_tool_rounds_the_note_is_not_a_system_message(monkeypatch, violetdb, fresh_registry):  # noqa: F811
    from ai.agents import ChatAgent
    use(monkeypatch, campus_server())
    monkeypatch.setenv("VIOLET_CHAT_MAX_TOOL_ROUNDS", "1")
    again = completion(tool_calls=[{"name": "campus__get_weather", "arguments": {"city": "杭州"}}])
    llm = install_fake_llm(monkeypatch, FakeLLM({"fake-chat": lambda kw: again if kw.get("tools") else "没查全"}))
    result = ChatAgent().run("天气")
    assert result["content"] == "没查全" and result["max_rounds_reached"]
    last = llm.calls[-1]["messages"]
    assert [m["role"] for m in last].count("system") == 1 and last[-1]["role"] == "user"
    assert last[-1]["content"].startswith("〔系统提示｜不是对方说的话〕")


def test_the_proactive_decision_has_the_same_shape(monkeypatch, violetdb):
    from ai.agents import ProactiveAgent
    violetdb.add_message("user", "incoming", "text", "今天好累")
    violetdb.add_message("assistant", "outgoing", "text", "早点休息")
    llm = install_fake_llm(monkeypatch, FakeLLM(default_responders()))
    event = json.dumps({"event_type": "scheduled_check", "conversation_state": {}}, ensure_ascii=False)
    ProactiveAgent().run(event)
    request = llm.calls[-1]["messages"]
    assert [m["role"] for m in request] == ["system", "user", "assistant", "user"]
    assert "action_type 只能是 none 或 chat" in request[0]["content"] and "PerformerAgent" not in request[0]["content"]
    assert json.loads(request[-1]["content"].splitlines()[-1])["event_type"] == "scheduled_check"


# ------------------------------------------------------------------------------------------------ no keyword gates

def test_what_she_knows_does_not_depend_on_the_words_of_the_message(violetdb):
    from ai.context_engine import ContextEngine
    from ai.profile_store import Profile
    Profile(violetdb).facts.remember("identity.status", "大学生", evidence="我是大学生", user_message="我是大学生")

    def shown(text):
        packet = ContextEngine(violetdb).build("chat", text)
        # (the optional "get to know you" question has its own pacing: asked at most now and then)
        keep = [s for s in packet.sections if "基础信息：" not in s and not s.startswith("【可选的了解】")]
        return [s.split("\n")[0] for s in keep], packet
    a, packet = shown("今天的晚霞好好看")
    b, _ = shown("我心情特别差，今天被老师骂了一顿，真的好难受好难受好难受好难受好难受")
    c, _ = shown("你在干嘛呀")
    assert a == b == c
    profile = next(s for s in packet.sections if s.startswith("【你了解的对方"))
    assert "身份：大学生" in profile and profile not in packet.volatile          # in the fixed part of the request
    assert any(s.startswith("【你的生活状态") for s in packet.sections)
    assert not hasattr(packet, "turn_note")


# ------------------------------------------------------------------------------------------------ groups

def test_a_plain_text_group_answer_is_sent_as_text():
    from ai.social import llm
    assert llm._plain_text_decision("晚安啦大家\n早点睡") == {
        "action": "say", "messages": ["晚安啦大家", "早点睡"], "reason": "模型没有按 JSON 回答，按她写的文字处理"}
    assert llm._plain_text_decision("来了|||在呢")["messages"] == ["来了", "在呢"]
    assert llm._plain_text_decision('{"action": "say", "messages": ["缺了括号"') is None       # broken JSON stays an error
    assert llm._plain_text_decision("") is None and llm._plain_text_decision("长" * 601) is None


def test_the_group_format_says_voice_is_not_sent_as_text_too():
    from ai.social import prompts
    assert "会先作为文字发出" not in prompts.OUTPUT_FORMAT
    assert "只发语音，不会再发一遍文字" in prompts.OUTPUT_FORMAT


def test_lyrics_in_a_voice_message_are_not_checked_as_her_own_life(violetdb, monkeypatch):
    from ai.chat import guard_reply
    monkeypatch.setenv("VIOLET_CONSISTENCY_GUARD", "enforce")
    sung = "<voice>我上周末去海边玩了一整天\n我妈妈给我做了一桌菜</voice>"
    asked = []
    text, found = guard_reply("行 给你唱一段\n" + sung, regenerate=lambda note: asked.append(note) or "x")
    assert (text, found, asked) == ("行 给你唱一段\n" + sung, [], [])
    assert guard_reply(sung, regenerate=lambda note: "x") == (sung, [])
    # her written lines are checked as before
    text, found = guard_reply("我上周末去海边玩了一整天\n" + sung, regenerate=None)
    assert found and "上周末去海边" not in text.split("<voice>")[0]


# ------------------------------------------------------------------------------------------------ her mood is hers

def test_the_owners_words_move_her_mood_only_when_she_says_so(make_world):
    from ai import mood
    w = make_world()
    say(w, "笨蛋 你怎么这么可爱哈哈")                        # keywords read this as scolding (and as praise)
    say(w, "你真的很烦，闭嘴")
    assert mood.current(w.db)["value"] == 0 and mood.note(w.db) == ""
    assert "note_feeling" in _names(w.llm.calls_for("fake-chat")[-1])
    w.llm.responders["fake-chat"] = _script(
        completion(tool_calls=[{"name": "note_feeling", "arguments": {"change": "worse", "reason": "被他认真凶了"}},
                               {"name": "note_feeling", "arguments": {"change": "much_worse", "reason": "又来"}}]),
        "……知道了")
    say(w, "别跟我说话了")
    assert mood.current(w.db)["value"] == pytest.approx(-0.35, abs=0.02)          # once per turn
    assert "被他认真凶了" in mood.note(w.db)


def test_a_wrong_feeling_is_refused_and_groups_still_use_the_words(violetdb):
    from ai import mood, owner_tools
    assert json.loads(owner_tools.note_feeling(violetdb, {}, {"change": "angry", "reason": "x"}))["ok"] is False
    assert mood.current(violetdb)["value"] == 0
    assert mood.feel(violetdb, "你闭嘴吧 垃圾", "路人", weight=0.5) < 0             # someone in a group


# ------------------------------------------------------------------------------------------------ a guess deletes nothing

def test_a_sentence_she_rewrote_is_not_cut_for_a_keyword_guess(make_world):
    """The first draft claims something with no record; the rewrite words it differently and the rules still guess
    "no record": it is sent whole. A line left exactly as it was is cut, as before."""
    w = make_world()

    def chat(kw):
        last = kw["messages"][-1]
        if last["role"] == "user" and "里面有和已记录的事实对不上的说法" in last["content"]:
            return "我上个月去山里住了两天\n你呢"
        return "我昨天去了海边\n你呢"
    w.llm.responders["fake-chat"] = chat
    say(w, "最近有出去玩吗")
    assert w.client.sent_texts[-2:] == ["我上个月去山里住了两天", "你呢"]
    w.llm.responders["fake-chat"] = lambda kw: "我昨天去了海边\n你呢"              # ignores the correction
    say(w, "真的吗")
    assert w.client.sent_texts[-1] == "你呢" and "我昨天去了海边" not in w.client.sent_texts[-2:]
