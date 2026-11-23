"""Phase 2 proactive policy and background-memory behaviours (one test per behaviour)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import requests

from tests.fakes import far_quiet_hours, FakeILinkClient, FakeLLM, default_responders, install_fake_llm

PROJECT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def world(monkeypatch, violetdb, tmp_path):
    monkeypatch.setenv("DEBOUNCE_NORMAL", "0.05")
    monkeypatch.setenv("DEBOUNCE_EXTRA", "0.1")
    monkeypatch.setenv("VIOLET_SEND_RETRY_DELAY_SECONDS", "0")
    llm = install_fake_llm(monkeypatch, FakeLLM(default_responders("嗯嗯")))
    from ai.agents import ChatAgent, MemoryAgent
    from ai.proactive_scheduler import ProactiveScheduler
    from violet_runtime import VioletRuntime, MessageSender
    client = FakeILinkClient()
    agents = {"chat": ChatAgent(), "memory": MemoryAgent()}
    config = tmp_path / "proactive.json"

    def write_config(**overrides):
        base = {"enabled": True, "quiet_hours": far_quiet_hours(), "daily_message_limit": 3, "min_minutes_after_user_message": 90,
                "min_minutes_between_proactive_messages": 180,
                "random_delay_minutes": {k: [60, 60] for k in ("morning", "noon", "afternoon", "evening", "night", "other")}}
        base.update(overrides)
        config.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")

    write_config()
    holder = {}
    scheduler = ProactiveScheduler(agents["chat"], send_messages=lambda m, e=None: holder["rt"].send_proactive(m, e), config_path=config)
    rt = VioletRuntime(client, violetdb, agents, proactive_scheduler=scheduler,
                    sender=MessageSender(client, violetdb, simulate_typing=False, sleep=lambda s: None))
    holder["rt"] = rt
    rt.start_worker()
    yield type("W", (), dict(rt=rt, client=client, llm=llm, db=violetdb, scheduler=scheduler, write_config=write_config))
    rt.stop(timeout=2)


def hhmm(dt):
    return dt.strftime("%H:%M")


def now(db):
    return datetime.fromisoformat(db.now())


def due_commitment(db, content="喝水"):
    result = db.add_commitment(content, (now(db) + timedelta(hours=1)).isoformat(), evidence=f"一小时后提醒我{content}")
    past = (now(db) - timedelta(minutes=1)).isoformat()
    return db.update_commitment(result["commitment"]["id"], due_at=past, next_attempt_at=past)


# --- background memory ---------------------------------------------------------------------

def test_reply_is_sent_before_memory_consolidation_runs(world):
    seen = []

    def memory(kw):
        jobs = world.db.list_memory_jobs()
        seen.append((list(world.client.sent_texts), jobs[0]["status"] if jobs else None))
        return json.dumps({"should_save": False, "reason": "t"})
    world.llm.responders["fake-memory"] = memory
    world.rt.ingest_text("今天好累")
    assert world.rt.wait_idle() and world.rt.wait_memory_idle()
    # exactly one consolidation, run by the background job after the reply went out
    assert seen == [(["嗯嗯"], "running")]
    job = world.db.list_memory_jobs()[0]
    assert (job["status"], job["assistant_reply"], job["user_message"]) == ("done", "嗯嗯", "今天好累")


def test_memory_can_be_switched_back_inline(monkeypatch, violetdb):
    monkeypatch.setenv("VIOLET_ASYNC_MEMORY", "0")
    install_fake_llm(monkeypatch, FakeLLM(default_responders()))
    from ai.agents import ChatAgent, MemoryAgent
    from violet_runtime import VioletRuntime
    rt = VioletRuntime(FakeILinkClient(), violetdb, {"chat": ChatAgent(),
                                               "memory": MemoryAgent()})
    assert rt.memory_worker is None and rt.async_memory is False


# --- defaults --------------------------------------------------------------------------------

def test_proactive_frequency_defaults_are_low():
    from ai.proactive_scheduler import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["daily_message_limit"] <= 3
    assert DEFAULT_CONFIG["min_minutes_after_user_message"] >= 60
    assert DEFAULT_CONFIG["min_minutes_between_proactive_messages"] >= 120
    shipped = json.loads((PROJECT / "proactive_config.json").read_text(encoding="utf-8"))
    assert (shipped["daily_message_limit"], shipped["min_minutes_after_user_message"],
            shipped["min_minutes_between_proactive_messages"]) == (3, 90, 180)


# --- backoff (pure unanswered count) -----------------------------------------------------------

def test_unanswered_proactive_backs_off_and_any_user_reply_resets(world):
    db = world.db
    db.add_user_message("text", "早", created_at=(now(db) - timedelta(hours=3)).isoformat())
    db.add_proactive_event("scheduled_proactive_check", status="sent", action_type="chat")
    assert world.scheduler.unanswered_count() == 1
    nxt = world.scheduler._backoff_next_time(now(db))
    gap = (nxt - now(db)).total_seconds() / 60
    assert 118 <= gap <= 121                      # 1 unanswered -> x2
    # Any reply (even brief "嗯") resets unanswered count to 0 and restores normal interval
    db.add_user_message("text", "嗯", created_at=(now(db) + timedelta(seconds=5)).isoformat())
    assert world.scheduler.unanswered_count() == 0
    world.scheduler.on_user_message()
    gap = (datetime.fromisoformat(db.get_proactive_setting("next_proactive_at")) - now(db)).total_seconds() / 60
    assert 58 <= gap <= 61


def test_backoff_is_capped(world):
    for _ in range(10):
        world.db.add_proactive_event("scheduled_proactive_check", status="sent", action_type="chat")
    nxt = world.scheduler._backoff_next_time(now(world.db))
    assert (nxt - now(world.db)).total_seconds() / 60 <= 60 * 8 + 1


def test_single_call_proactive_no_reply_skips_without_sending(world):
    db = world.db
    db.add_user_message("text", "早", created_at=(now(db) - timedelta(hours=3)).isoformat())
    db.set_proactive_setting("next_proactive_at", (now(db) - timedelta(minutes=1)).isoformat())
    world.llm.responders["fake-chat"] = lambda kw: "[NO_REPLY]"
    world.scheduler.tick()
    assert world.client.sent_texts == []
    ev = db.list_proactive_events()[0]
    assert ev["status"] == "skipped"


# --- commitments -------------------------------------------------------------------------------

def test_a_reminder_goes_out_on_time_even_in_quiet_hours(world):
    """2026-11-12 (virtual test): "明早七点半叫我起床" went out after 8:00 -- quiet hours are for her own messages."""
    current = now(world.db)
    world.write_config(quiet_hours={"start": hhmm(current - timedelta(hours=1)), "end": hhmm(current + timedelta(hours=1))})
    commitment = due_commitment(world.db)
    world.scheduler.tick()
    assert world.client.sent_texts
    assert world.db.get_commitment(commitment["id"])["status"] == "sent"


def test_reminder_falls_back_to_template_when_generation_fails(world):
    world.llm.responders["fake-chat"] = lambda kw: (_ for _ in ()).throw(RuntimeError("chat down"))
    commitment = due_commitment(world.db, "吃药")
    world.scheduler.tick()
    assert world.client.sent_texts == ["提醒你一下：吃药"]
    delivered = world.db.get_commitment(commitment["id"])
    assert delivered["status"] == "sent" and "chat down" in delivered["last_error"]
    assert world.db.get_proactive_event(delivered["proactive_event_id"])["reason"] == "fallback template used"


def test_reminder_text_must_mention_the_reminded_thing(world):
    world.llm.responders["fake-chat"] = lambda kw: "在干嘛呢"          # model drifted off-topic
    due_commitment(world.db, "吃药")
    world.scheduler.tick()
    assert world.client.sent_texts == ["提醒你一下：吃药"]


def test_reminder_is_claimed_only_once(world):
    commitment = due_commitment(world.db)
    assert world.db.claim_commitment(commitment["id"]) is True
    assert world.db.claim_commitment(commitment["id"]) is False
    world.scheduler.tick()
    assert world.client.sent_texts == []          # it is 'sending' elsewhere; not sent twice


def test_reminder_send_failure_is_retried_then_marked_failed(world):
    world.write_config(commitments={"enabled": True, "max_attempts": 2, "retry_minutes": 0})
    world.llm.responders["fake-chat"] = lambda kw: "该喝水啦"
    world.client.send_failures = [requests.ReadTimeout("x"), requests.ReadTimeout("y")]
    commitment = due_commitment(world.db)
    world.scheduler.tick()
    first = world.db.get_commitment(commitment["id"])
    assert (first["status"], first["attempts"]) == ("pending", 1) and "x" in first["last_error"]
    world.scheduler.tick()
    second = world.db.get_commitment(commitment["id"])
    assert (second["status"], second["attempts"]) == ("failed", 2)
    assert world.client.sent_texts == []
    statuses = [e["status"] for e in world.db.list_proactive_events(event_type="commitment_reminder")]
    assert statuses == ["failed", "failed"]


# --- policy units ----------------------------------------------------------------------------------

def test_no_reply_token_means_do_not_send():
    from ai.chat_output import normalize_chat_output
    assert normalize_chat_output("[NO_REPLY]") == {"should_send": False, "messages": [], "declined": True}
    assert [m["content"] for m in normalize_chat_output("嗯\n[NO_REPLY]")["messages"]] == ["嗯"]


def test_topic_guard_only_blocks_clear_matches():
    from ai.memory_model import touches_topic
    assert touches_topic("对了你那个服务器修好了没", "用户的服务器出现故障")
    assert not touches_topic("晚饭吃了吗", "用户中午吃了火锅")
    assert not touches_topic("工作忙吗", "用户工作压力大")


def test_proactive_chat_less_disables_delay_scale_and_updates_note(world):
    from ai import persona_templates as pt
    from ai.profile_store import Profile

    pt.switch(world.db, "stepsister", "keep")
    pt.set_stage(None, 3)
    world.write_config(
        daily_message_limit=3,
        min_minutes_after_user_message=90,
        min_minutes_between_proactive_messages=180,
        random_delay_minutes={k: [120, 240] for k in ("morning", "noon", "afternoon", "evening", "night", "other")},
    )
    world.scheduler.load_config()
    Profile(world.db).preferences.set_explicit("proactive_chat", "less", evidence="少主动发", user_message="以后少主动发消息")

    after, between, lvl, pace = world.scheduler._effective_intervals()
    assert lvl >= 3
    assert (after, between) == (90.0, 180.0)
    assert pace["delay_scale"] == 1.0

    afternoon = datetime(2026, 11, 23, 15, 0, tzinfo=now(world.db).tzinfo)
    for _ in range(25):
        delay = (world.scheduler._random_next_time(afternoon) - afternoon).total_seconds() / 60
        assert 120 <= delay <= 240
    backoff = (world.scheduler._backoff_next_time(afternoon) - afternoon).total_seconds() / 60
    assert 240 <= backoff <= 480

    ctx = world.scheduler._build_event_context(afternoon)
    assert ctx["daily_message_limit"] == 1
    assert "0/1 次" in ctx["proactive_note"]
    assert "0/3 次" not in ctx["proactive_note"]
    assert "自然热情地来蹭个存在感" not in ctx["proactive_note"]
    pt.set_stage(None, None)
    pt.switch(world.db, "violet", "keep")


def test_miss_unanswered_summary_topic_preserved_with_life_events(world):
    from ai.relationship import RelationshipView
    note, topic = world.scheduler._compose_proactive_note(
        view=RelationshipView(stage="close", chat_days=12),
        level=2,
        since_user=300.0,
        sent_today=1,
        daily_limit=3,
        unanswered=[{"at": "11:00", "hours": 4.5, "elapsed_hours": 4.5}],
        life_events=[{"id": 1, "content": "去楼下买了个热乎的包子", "type": "violet_daily_event"}],
        upcoming=[],
        open_plans=[],
        web_findings=[],
        schedule_moments=[],
        recent=[],
        habit_note=None,
        class_ok=True,
    )
    assert topic == "miss_unanswered"
    assert "热乎的包子" in note


def test_note_web_mentions_chinese_site_aliases_and_offered_overlap(violetdb):
    from ai.chat import note_web_mentions
    from ai.web.findings import FindingsStore

    store = FindingsStore(violetdb)
    pub = now(violetdb).date().isoformat()
    f_douban = store.upsert(
        url="https://book.douban.com/review/123",
        title="2026秋季独立音乐新专辑盘点：五张值得单曲循环的冷门佳作",
        summary="整理了几张冷门独立音乐专辑",
        source_kind="fetched_page", origin="explorer", published_at=pub, published_verified=True,
    )
    f_paper = store.upsert(
        url="https://www.thepaper.cn/news/456",
        title="城市独立书店观察：为什么年轻人重新走进街角旧书店",
        summary="实体小书店观察报道",
        source_kind="fetched_page", origin="explorer", published_at=pub, published_verified=True,
    )
    f_bili = store.upsert(
        url="https://www.bilibili.com/video/BV1xx",
        title="手工木作修复百年老书桌全过程记录",
        summary="木工修复记录",
        source_kind="fetched_page", origin="explorer", published_at=pub, published_verified=True,
    )
    f_ithome = store.upsert(
        url="https://www.ithome.com/0/999.htm",
        title="新款轻量电子墨水阅读器发布",
        summary="电子墨水阅读器评测",
        source_kind="fetched_page", origin="explorer", published_at=pub, published_verified=True,
    )

    # Unrelated reply must NOT mark an offered finding
    assert note_web_mentions("下午去楼下买了个热乎的包子，顺手散了会儿步", used_finding_ids=[f_douban, f_paper]) == []
    assert store.get(f_douban)["share_count"] == 0 and store.get(f_paper)["share_count"] == 0

    # Chinese site aliases match subdomains and root domains
    assert note_web_mentions("刚才在豆瓣看到一篇盘点，推了五张新专辑", used_finding_ids=[f_douban]) == [f_douban]
    assert note_web_mentions("在B站刷到一个木工修复视频", used_finding_ids=[f_bili]) == [f_bili]
    assert note_web_mentions("刚在IT之家看到新款阅读器", used_finding_ids=[f_ithome]) == [f_ithome]

    # Concise paraphrase without domain/alias still matches when offered in used_finding_ids
    assert note_web_mentions("看到一篇写旧书店的小文章，挺有意思的", used_finding_ids=[f_paper]) == [f_paper]
    assert store.get(f_paper)["share_count"] == 1


def test_empty_or_failed_chat_triggers_fallback_and_proactive_failed(world):
    from ai.chat import run_pipeline, run_proactive_pipeline
    from violet_runtime import DEFAULT_FALLBACK_REPLY

    class _StubChat:
        def __init__(self, payload):
            self.payload = payload

        def run(self, *args, **kwargs):
            return dict(self.payload)

    # 1. run_pipeline treats failed=True or empty content (without [NO_REPLY]) as chat_failed=True
    r_filter = run_pipeline(
        user_message="你好",
        chat_agent=_StubChat({"content": "", "messages": [], "raw": {}, "failed": True, "finish_reason": "content_filter"}),
        memory_agent=None,
        defer_memory=True,
    )
    assert r_filter["chat_failed"] is True and "content_filter" in r_filter["errors"]["chat"]

    r_empty = run_pipeline(
        user_message="你好",
        chat_agent=_StubChat({"content": "", "messages": [], "raw": {}}),
        memory_agent=None,
        defer_memory=True,
    )
    assert r_empty["chat_failed"] is True

    r_declined = run_pipeline(
        user_message="嗯",
        chat_agent=_StubChat({"content": "[NO_REPLY]", "messages": [], "raw": {}}),
        memory_agent=None,
        defer_memory=True,
    )
    assert r_declined["chat_failed"] is False

    # 2. Proactive pipeline marks safety-filter / empty response as failed (NOT skipped)
    p_filter = run_proactive_pipeline(
        event_context={"event_type": "scheduled_proactive_check"},
        chat_agent=_StubChat({"content": "", "messages": [], "raw": {}, "failed": True, "finish_reason": "content_filter"}),
    )
    assert p_filter["should_send"] is False and "content_filter" in (p_filter.get("error") or "")
    ev_filter = world.db.get_proactive_event(p_filter["event_id"])
    assert ev_filter["status"] == "failed" and "content_filter" in (ev_filter["reason"] or "")

    p_empty = run_proactive_pipeline(
        event_context={"event_type": "scheduled_proactive_check"},
        chat_agent=_StubChat({"content": None, "messages": [], "raw": {}}),
    )
    assert world.db.get_proactive_event(p_empty["event_id"])["status"] == "failed"

    p_skip = run_proactive_pipeline(
        event_context={"event_type": "scheduled_proactive_check"},
        chat_agent=_StubChat({"content": "[NO_REPLY]", "messages": [], "raw": {}}),
    )
    ev_skip = world.db.get_proactive_event(p_skip["event_id"])
    assert ev_skip["status"] == "skipped" and ev_skip["reason"] == "ChatAgent decided not to send"

    # 3. Runtime sends fallback once on empty/failed turn (never silently "done"), and stays quiet on [NO_REPLY]
    world.rt.agents["chat"] = _StubChat({"content": "", "messages": [], "raw": {}, "failed": True, "finish_reason": "content_filter"})
    world.rt.ingest_text("测试空回复")
    assert world.rt.wait_idle()
    assert world.client.sent_texts == [DEFAULT_FALLBACK_REPLY]
    row1 = world.db.fetch_one("SELECT status, error FROM inbound_messages ORDER BY id DESC LIMIT 1")
    assert row1["status"] == "failed" and "content_filter" in (row1["error"] or "")

    # Second immediate failure respects _fallback_just_said() and marks failed without repeating fallback
    world.rt.ingest_text("测试第二次空回复")
    assert world.rt.wait_idle()
    assert world.client.sent_texts == [DEFAULT_FALLBACK_REPLY]
    row2 = world.db.fetch_one("SELECT status, error FROM inbound_messages ORDER BY id DESC LIMIT 1")
    assert row2["status"] == "failed" and "fallback not repeated" in (row2["error"] or "")

    # Intentional [NO_REPLY] in normal chat does NOT send fallback and completes as done
    world.rt.agents["chat"] = _StubChat({"content": "[NO_REPLY]", "messages": [], "raw": {}})
    world.rt.ingest_text("嗯")
    assert world.rt.wait_idle()
    assert world.client.sent_texts == [DEFAULT_FALLBACK_REPLY]
    row3 = world.db.fetch_one("SELECT status FROM inbound_messages ORDER BY id DESC LIMIT 1")
    assert row3["status"] == "done"


def test_find_touched_topic_blocks_two_char_avoid_topics_and_passes_unrelated(world):
    from ai.chat import find_touched_topic, run_proactive_pipeline
    from ai.profile_store import Profile

    for keyword, draft in (
        ("相亲", "对了你周末有去相亲吗？"),
        ("前任", "你最近还有和前任联系吗"),
        ("考研", "考研复习得怎么样啦"),
        ("体重", "最近体重有变化吗"),
        ("工资", "这个月工资发了没"),
        ("催婚", "家里又催婚了吗"),
        ("分手", "上次分手那件事缓过来了没"),
        ("成绩", "期末成绩出来了吗"),
        ("GPA", "你这学期GPA多少呀"),
        ("ex", "did your ex text you again?"),
    ):
        restricted = [{"memory_id": None, "content": keyword, "reason": "user asked not to dwell on this"}]
        assert find_touched_topic(draft, restricted) == restricted[0], keyword
        assert find_touched_topic("下午去公园吹了吹风，晚饭想吃面", restricted) is None, keyword

    # Empty, 1-char, stop-word, partial ASCII substring, and longer non-matching topics must NOT over-match
    assert find_touched_topic("今天天气真好", [{"content": ""}]) is None
    assert find_touched_topic("今天花了不少钱", [{"content": "钱"}]) is None
    assert find_touched_topic("今天天气真好", [{"content": "今天"}]) is None
    assert find_touched_topic("see you next week", [{"content": "ex"}]) is None
    assert find_touched_topic("工作忙吗", [{"content": "用户工作压力大"}]) is None

    # End-to-end proactive pipeline hard-blocks 2-char avoid_topic from Profile
    Profile(world.db).preferences.set_explicit("avoid_topic", "相亲", evidence="别跟我提相亲", user_message="别跟我提相亲")
    state = world.scheduler._conversation_state()

    class _StubChat:
        def __init__(self, text):
            self.text = text

        def run(self, *args, **kwargs):
            return {"content": self.text, "messages": [], "raw": {}, "failed": False, "finish_reason": "stop"}

    blocked = run_proactive_pipeline(
        event_context={"event_type": "scheduled_proactive_check", "conversation_state": state},
        chat_agent=_StubChat("对了你周末有去相亲吗？"),
    )
    assert blocked["should_send"] is False
    ev = world.db.get_proactive_event(blocked["event_id"])
    assert ev["status"] == "skipped" and "restricted topic" in (ev["reason"] or "")

    allowed = run_proactive_pipeline(
        event_context={"event_type": "scheduled_proactive_check", "conversation_state": state},
        chat_agent=_StubChat("刚才在楼下买了杯热拿铁，今天风挺舒服的"),
    )
    assert allowed["should_send"] is True
