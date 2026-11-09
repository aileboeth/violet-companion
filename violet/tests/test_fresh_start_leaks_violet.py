"""2026-11-09 (user: "我明明已经将他的记忆给清除了……清除之后他还问我叫什么，但是后面他说我熬夜，我问他为什么他说看到了
聊天记录，这个完全清不干净"). After "清除记忆 + 翻篇" nothing from before the cut may reach the model: not the chat lines
(that part already worked), and not what other parts derive from the message log -- how long you have known each other,
today's message count, the "you already told them to sleep early today" note, the chat MemoryAgent reads, the proactive
reasons. Fakes only; the clock is fixed at 16:35 on the day of the user's export."""

from __future__ import annotations

from datetime import datetime

import pytest

from tests.test_qq_social_violet import admin, db  # noqa: F401  (the admin client + database fixtures)

DAY ="2026-11-09"
CUT_AT = f"{DAY}T13:20:00+08:00"
NOW = datetime.fromisoformat(f"{DAY}T16:35:00+08:00")
BEFORE = [      # from the user's export (D:\unity\violet-chat-2026-11-09-2026-11-09.json), 00:28 - 12:07
    ("user", "00:28:50", "在干啥呢"),
    ("assistant", "00:29:08", "半夜不睡觉，跑来问我在干嘛"),
    ("user", "00:56:22", "我今晚通宵"),
    ("assistant", "00:57:05", "七点半的闹钟都设着，你还想通宵，明天是不是准备睡到下午"),
    ("user", "01:25:50", "晚安"),
    ("assistant", "01:26:01", "老公，早点睡"),
    ("assistant", "08:12:22", "我刚睁眼就想起你，昨晚聊到快两点，也不知道你睡够没有"),
    ("assistant", "12:06:58", "中午了，你该不会又把饭跳过去了吧"),
]
AFTER = [
    ("user", "13:27:47", "你好~"),
    ("assistant", "13:28:04", "对了，我该怎么叫你"),
    ("user", "13:28:22", "叫老公~"),
    ("assistant", "13:28:30", "哟，第一天就想当老公了"),
]
LEAKED = ("通宵", "早点睡", "快两点", "00:28", "半夜不睡觉")


@pytest.fixture()
def fresh_start(violetdb, tmp_path, monkeypatch):
    """The user's day: chats before 13:20, 清除记忆 + 翻篇 at 13:20, a few messages after; now 16:35."""
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    from ai import context_cut, settings
    for role, clock, text in BEFORE:
        add = violetdb.add_user_message if role == "user" else violetdb.add_assistant_message
        add("text", text, created_at=f"{DAY}T{clock}+08:00")
    monkeypatch.setattr(settings, "now_local", lambda: datetime.fromisoformat(CUT_AT))
    context_cut.cut(violetdb, "all")
    for role, clock, text in AFTER:
        add = violetdb.add_user_message if role == "user" else violetdb.add_assistant_message
        add("text", text, created_at=f"{DAY}T{clock}+08:00")
    monkeypatch.setattr(settings, "now_local", lambda: NOW)
    return violetdb


def _chat_prompt(db, text: str) -> str:
    from ai.context_engine import ContextEngine
    packet = ContextEngine(db).build("chat", user_message=text)
    return "\n".join([*packet.sections, *(h["content"] for h in packet.history), packet.turn_note or ""])


def test_the_chat_prompt_after_a_fresh_start_has_nothing_from_before(fresh_start):
    prompt = _chat_prompt(fresh_start, "好累好累啊")
    for word in LEAKED:
        assert word not in prompt, word
    assert "你好~" in prompt                                     # the chat after the cut is still there


def test_how_long_you_have_known_each_other_starts_at_the_cut(fresh_start):
    from ai.context_engine import ContextEngine
    line = "\n".join(ContextEngine(fresh_start)._relationship_section().lines)
    assert "13:27" in line and "00:28" not in line


def test_todays_message_count_starts_at_the_cut(fresh_start):
    from ai.context_engine import ContextEngine
    lines = "\n".join(ContextEngine(fresh_start)._ai_experience_section().lines)
    assert f"一共聊了 {len(AFTER)} 条消息" in lines


def test_memory_agent_reads_only_the_chat_after_the_cut(fresh_start):
    from ai.chat import format_recent_messages
    text = format_recent_messages(10)
    for word in LEAKED:
        assert word not in text, word
    assert "叫老公~" in text


def test_the_relationship_view_starts_at_the_cut(fresh_start):
    from ai.relationship import assess
    view = assess(fresh_start)
    assert view.chat_days == 1 and view.initiations == 1          # only "你好~" started a conversation after the cut


def test_proactive_reasons_do_not_come_from_before_the_cut(fresh_start):
    from ai.proactive_scheduler import ProactiveScheduler
    scheduler = ProactiveScheduler.__new__(ProactiveScheduler)
    glow = scheduler._afterglow(NOW)
    assert glow is None or glow["quote"] not in ("晚安", "我今晚通宵")


def test_her_proactive_messages_from_before_the_cut_are_not_shown_again(fresh_start):
    """"昨晚聊到快两点" (sent at 08:12) goes to the proactive decision as "what you sent lately"; it stays in the table
    (the daily limit counts it) but is not shown after the cut."""
    import json
    from ai.proactive_scheduler import ProactiveScheduler
    for clock, text in (("08:12:22", "我刚睁眼就想起你，昨晚聊到快两点"), ("15:00:00", "下午好呀")):
        at = f"{DAY}T{clock}+08:00"
        fresh_start.execute("INSERT INTO proactive_events (event_type, status, sent_messages, created_at, updated_at) "
                            "VALUES ('idle', 'sent', ?, ?, ?)", (json.dumps([{"content": text}], ensure_ascii=False), at, at))
    scheduler = ProactiveScheduler.__new__(ProactiveScheduler)
    recent, _ = scheduler._recent_proactive(NOW)
    assert [r["text"] for r in recent] == ["下午好呀"]


def test_deleting_the_chat_history_from_the_settings_page(admin, db, tmp_path, monkeypatch):
    """User: "删除记忆之后……在聊天记录那里的记录也给删掉，这样不就一劳永逸了". The chat goes (archived first), sends still
    queued stay, inbound rows keep their dedup key but lose the text, and the page is turned too."""
    import json
    from ai import context_cut
    from ai.social.store import SocialStore
    from tests.test_qq_social_violet import SAME
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    db.add_user_message("text", "我今晚通宵")
    db.add_assistant_message("text", "老公，早点睡")
    now = db.now()
    for status, text in (("sent", "老公，早点睡"), ("queued", "还在排队的一条")):
        db.execute("INSERT INTO outbound_messages (client_id, origin, message_type, content, status, created_at, updated_at) "
                   "VALUES (?, 'reply', 'text', ?, ?, ?, ?)", (f"c-{status}", text, status, now, now))
    for key, status in (("k-done", "done"), ("k-new", "pending")):
        db.execute("INSERT INTO inbound_messages (dedup_key, content, status, received_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                   (key, "我今晚通宵", status, now, now))
    store = SocialStore(db)
    conv = store.conversation("napcat", "group", "701", "群", default_mode="auto")
    db.execute("INSERT INTO social_messages (conversation_id, sender_id, sender_name, content, is_self, created_at) "
               "VALUES (?, '9', '群友', '昨晚谁熬夜了', 0, ?)", (conv["id"], now))

    r = admin.post("/api/control/reset", json={"memory": True, "history": True, "confirm": True}, headers=SAME).get_json()
    assert r["ok"] and r["data"]["history"]["removed"]["messages"] == 2
    assert db.fetch_one("SELECT COUNT(*) AS n FROM messages")["n"] == 0
    assert db.fetch_one("SELECT COUNT(*) AS n FROM social_messages")["n"] == 0
    assert [row["status"] for row in db.fetch_all("SELECT status FROM outbound_messages")] == ["queued"]
    inbound = {row["dedup_key"]: row["content"] for row in db.fetch_all("SELECT dedup_key, content FROM inbound_messages")}
    assert inbound == {"k-done": None, "k-new": "我今晚通宵"}
    archive = json.loads(open(r["data"]["history"]["archive"], encoding="utf-8").read())
    assert [m["content"] for m in archive["tables"]["messages"]] == ["我今晚通宵", "老公，早点睡"]
    assert archive["tables"]["social_messages"][0]["content"] == "昨晚谁熬夜了"
    assert r["data"]["cut"]["owner"] and context_cut.status()["social"]          # deleting the chat turns the page


def test_clearing_memory_also_drops_queued_memory_jobs(violetdb, tmp_path, monkeypatch):
    """A memory job queued before the clear would write the old chat back afterwards."""
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    from ai.persona_templates import clear_memory
    violetdb.execute("INSERT INTO memory_jobs (turn_id, user_message, status, next_attempt_at, created_at, updated_at) "
                     "VALUES ('t-old', '我今晚通宵', 'pending', ?, ?, ?)", (violetdb.now(),) * 3)
    _, counts = clear_memory(violetdb)
    assert counts["memory_jobs"] == 1
    assert violetdb.fetch_one("SELECT COUNT(*) AS n FROM memory_jobs")["n"] == 0
