"""2026-11-09: fixes from a real night of chat (user: 完全记不住事，聊过没多久就直接忘记了).

What happened (server data): "我蓝牙连的那个不是耳机是手表你要记住哦" was saved as a memory at 00:34; at 01:13 and in the
next morning's message she said "耳机" again -- the memory reached a reply only when the owner's message shared a word
with it, and the phone line said "戴着耳机" every turn. "明天早八" said at 01:03 was stored for the wrong day. A teasing
"笨蛋" made her "生气" for half an hour. Five or six lines kept coming after the owner had moved on.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

TZ = timezone(timedelta(hours=8))
WATCH = "用户蓝牙连接的是手表，不是耳机（用户要求记住）"


def _sent_text(llm) -> str:
    return "\n".join(str(m.get("content")) for c in llm.calls for m in c["messages"] if isinstance(m, dict))


# ------------------------------------------------------------------------------------------- standing memories

def test_a_fact_the_owner_told_her_reaches_a_turn_that_does_not_mention_it(violetdb, monkeypatch):
    from tests.fakes import FakeLLM, default_responders, install_fake_llm
    violetdb.add_event_memory(WATCH, category="other", kind="fact", importance=2.0)
    violetdb.add_event_memory("用户上周去看了一场电影", kind="event", progress="done")
    llm = install_fake_llm(monkeypatch, FakeLLM(default_responders("mua")))
    from ai.agents.chat_agent import ChatAgent
    agent = ChatAgent()
    agent.run("mua~", call_context={"phone_note": "【主人手机此刻（01:13 读取）】连着蓝牙设备「Watch S1」"})
    text = _sent_text(llm)
    assert WATCH in text and "以这里为准" in text
    assert "看了一场电影" not in text                            # an ended event is not a standing memory
    assert WATCH in str(agent.last_context_report) or agent.last_context_report["standing_memory_ids"]


def test_standing_memories_in_proactive_runs_and_the_switch(violetdb, monkeypatch):
    from ai.context_engine import ContextEngine
    violetdb.add_event_memory(WATCH, kind="fact", importance=2.0)
    packet = ContextEngine(violetdb).build("proactive", "[proactive event]", {})
    assert any(WATCH in s for s in packet.sections)
    assert not any(WATCH in s for s in packet.volatile)          # stable: before the history, served from the cache
    monkeypatch.setenv("VIOLET_MEMORY_STANDING_FACTS", "0")
    packet = ContextEngine(violetdb).build("proactive", "[proactive event]", {})
    assert not any(WATCH in s for s in packet.sections)


def test_the_phone_lines_words_pick_which_facts_come_first(violetdb, monkeypatch):
    from ai.context_engine import ContextEngine
    monkeypatch.setenv("VIOLET_MEMORY_STANDING_FACTS", "1")
    violetdb.add_event_memory("用户是计算机专业的大三学生", kind="fact", importance=4.0)
    violetdb.add_event_memory(WATCH, kind="fact", importance=2.0)
    engine = ContextEngine(violetdb)
    facts, _ = engine.standing_memories("")
    assert [f["content"] for f in facts] == ["用户是计算机专业的大三学生"]
    facts, _ = engine.standing_memories("【主人手机此刻】连着蓝牙设备（不一定是耳机，也可能是手表、音箱）")
    assert [f["content"] for f in facts] == [WATCH]


def test_plans_of_the_next_hours_are_shown_with_their_weekday(violetdb):
    from ai.context_engine import ContextEngine
    from ai.memory_model import WEEKDAYS, parse_time
    now = parse_time(violetdb.now())
    soon = (now + timedelta(hours=7)).replace(second=0, microsecond=0)
    violetdb.add_event_memory("用户早上 8:30 有早八课", kind="event", progress="planned", event_time=soon.isoformat())
    violetdb.add_event_memory("用户下周要考试", kind="event", progress="planned",
                              event_time=(now + timedelta(days=5)).isoformat())
    section, ids = ContextEngine(violetdb)._standing_section({})
    text = section.render()
    assert f"接下来的安排（{soon.strftime('%m-%d')} 周{WEEKDAYS[soon.weekday()]} {soon.strftime('%H:%M')}）" in text
    assert "早八课" in text and "下周要考试" not in text and len(ids) == 1


def test_a_standing_memory_is_not_repeated_in_the_recall_section(violetdb):
    from ai.context_engine import ContextEngine
    mid = violetdb.add_event_memory(WATCH, kind="fact", importance=2.0)
    recalled = [{"id": mid, "content": WATCH, "kind": "fact", "status": "active",
                 "retrieval": {"reason": "user_words", "matched_terms": ["手表"]}}]
    packet = ContextEngine(violetdb).build("chat", "手表", {"recalled_memories": recalled})
    assert sum(s.count(WATCH) for s in packet.sections) == 1
    assert not any("没有找到" in s for s in packet.sections)


# ------------------------------------------------------------------------------------------------- time words

def test_after_midnight_tomorrow_means_the_coming_day():
    from ai.memory_model import clock_text, day_words
    night = datetime(2026, 11, 9, 1, 3, tzinfo=TZ)                  # a Monday
    assert clock_text(night) == "2026-11-09 01:03 周一 凌晨"
    words = day_words(night)
    assert words.startswith("今天 11-09 周一；明天 11-10 周二；后天 11-11 周三")
    assert "“明天”多半是天亮以后的这一天（11-09 周一）" in words
    assert "凌晨" not in day_words(night.replace(hour=14))


def test_the_memory_agent_and_the_reply_get_the_weekday(violetdb, monkeypatch):
    from tests.fakes import FakeLLM, default_responders, install_fake_llm
    llm = install_fake_llm(monkeypatch, FakeLLM(default_responders("嗯")))
    from ai.agents.memory_agent import MemoryAgent
    MemoryAgent().run("用户说：明天早八")
    last = llm.calls[-1]["messages"][-1]["content"]
    assert last.startswith("当前时间：") and " 周" in last.split("\n")[0] and "写进记忆的日期和星期必须和这里一致" in last
    from ai.chat import format_base_info
    assert "日期：今天 " in format_base_info()


# ------------------------------------------------------------------------------------------------------ phone

def test_a_bluetooth_device_is_not_called_headphones():
    from ai.phone_state import audio_text
    watch = {"headphonesOrBluetooth": True, "devices": [{"type": "bluetooth", "name": "Xiaomi Watch S1"}]}
    assert audio_text(watch) == "连着蓝牙设备「Xiaomi Watch S1」（不一定是耳机，也可能是手表、音箱）"
    assert audio_text({"headphonesOrBluetooth": True, "devices": [{"type": "wired_headphones", "name": None}]}) == "插着有线耳机"
    assert audio_text({"headphonesOrBluetooth": True}) == "连着耳机或蓝牙设备"         # an older phone app: no device list
    assert audio_text({"headphonesOrBluetooth": False}) == ""


def test_replies_leave_out_a_far_alarm_and_one_read_serves_both(monkeypatch):
    from ai import phone_state
    now = datetime(2026, 11, 9, 0, 56, tzinfo=TZ)
    alarm = {"hasAlarm": True, "time": {"epochMs": 0, "iso": now.replace(hour=7, minute=30).isoformat()}}
    reads = []
    monkeypatch.setenv("VIOLET_PHONE_CONTEXT_CHAT", "1")
    monkeypatch.setenv("VIOLET_PHONE_CONTEXT", "1")
    monkeypatch.setattr(phone_state, "_read", lambda calls=None: reads.append(1) or
                        {"next_alarm": alarm, "battery": {"percent": 40, "charging": False}})
    import ai.proactive_scheduler as ps
    monkeypatch.setattr(ps, "now_china", lambda: now)
    phone_state.clear_cache()
    try:
        assert "闹钟" not in phone_state.line("chat")                         # 6.5 hours away
        assert "下一个闹钟 今天 07:30" in phone_state.line("proactive")         # a morning message may use it
        assert len(reads) == 1
        assert "对不上时按对方说的来" in phone_state.line("chat")
    finally:
        phone_state.clear_cache()
    near = phone_state.compose({"next_alarm": alarm}, "06:00", now.replace(hour=6), "chat")
    assert "下一个闹钟 今天 07:30" in near


def test_the_headphones_event_names_the_device():
    from ai import phone_events
    at = datetime(2026, 11, 9, 0, 30, tzinfo=TZ)
    assert phone_events.describe({"type": "headphones_connected", "detail": {"name": "Watch S1"}}, at) == "00:30 连上「Watch S1」"
    assert phone_events.describe({"type": "headphones_disconnected", "detail": {}}, at) == "00:30 断开耳机或蓝牙设备"


# ------------------------------------------------------------------------------------------------------- mood

def test_teasing_is_not_scolding(violetdb):
    from ai import mood
    # the two lines of the real chat that made her angry for half an hour
    mood.feel(violetdb, "[引用你之前说的：那今天走了0步，手表估计也在笑你]\n现在凌晨十二点，我有步数才奇怪，笨蛋")
    mood.feel(violetdb, "笨蛋还不去睡觉")
    assert mood.note(violetdb) == "" and mood.current(violetdb)["value"] == pytest.approx(-0.2, abs=0.01)
    mood.feel(violetdb, "小笨蛋，爱你")                                  # pet name + love: a sweet word, not scolding
    assert mood.current(violetdb)["value"] > 0
    assert mood.scold_strength("哈哈哈笨蛋") == 0 and mood.scold_strength("你好烦啊~") == 0
    assert mood.scold_strength("闭嘴") == 1.0 and mood.scold_strength("滚啦哈哈") == pytest.approx(0.4)


def test_hard_words_still_upset_her_and_the_note_lets_the_model_judge(violetdb):
    from ai import mood
    mood.feel(violetdb, "闭嘴吧你")
    note = mood.note(violetdb)
    assert "不高兴" in note and "主人对你说了「闭嘴」" in note
    assert "话也会比平时多" in note                                  # 2026-11-03 decision kept
    assert "听得出是玩笑" in note and "别每次回复都再说" in note


# --------------------------------------------------------------------------------------------- interruption

def test_new_words_stop_the_rest_of_a_reply(violetdb):
    from ai.channels.outbound import chat_payload
    from tests.fakes import FakeILinkClient
    from violet_runtime import MessageSender
    client = FakeILinkClient()
    sender = MessageSender(client, violetdb, simulate_typing=False, sleep=lambda s: None)
    lines = chat_payload("哦，这就睡了\n我气还没消呢\n晚安啦，笨蛋")["messages"]
    checks = []

    def owner_wrote():
        checks.append(1)
        return len(checks) >= 2                       # the owner's new message arrives after her second line
    report = sender.send_messages(lines, origin="reply", turn_id="t1", interrupted=owner_wrote)
    assert [m["content"] for m in report.sent] == ["哦，这就睡了", "我气还没消呢"]
    assert report.interrupted and [m.get("error") for m in report.skipped] == ["interrupted"]
    assert report.ok and len(client.sent) == 2
    rows = violetdb.fetch_all("SELECT content, status, error FROM outbound_messages ORDER BY id")
    assert rows[-1]["content"] == "晚安啦，笨蛋" and rows[-1]["status"] == "skipped"
    # nothing sent yet -> never asked; no check -> everything goes out
    report = sender.send_messages(chat_payload("一\n二")["messages"], origin="reply", interrupted=lambda: True)
    assert [m["content"] for m in report.sent] == ["一"]
    report = sender.send_messages(chat_payload("一\n二")["messages"], origin="reply")
    assert len(report.sent) == 2 and not report.interrupted
