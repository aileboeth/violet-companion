"""Regression suite for the 2026-11-18 aliveness, prompt-budget, consistency, social-meme, and per-scope runtime rework.

Offline only: fake LLM, temp data dirs, zero network calls.
"""

from __future__ import annotations

import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

from tests.fakes import FakeILinkClient, FakeLLM, default_responders, install_fake_llm
from tests.image_fixtures import PNG


@pytest.fixture()
def isolated_data(tmp_path, monkeypatch):
    import ai.control_config as cc
    from ai import memes

    saved = dict(os.environ)
    data_dir = tmp_path / "data"
    monkeypatch.setenv("VIOLET_DATA_DIR", str(data_dir))
    monkeypatch.setenv("VIOLET_ENV_FILE", str(tmp_path / "missing.env"))
    cc._ORIGINALS.clear()
    cc._DEFAULT_STORE = None
    cc._CHECKED_INSTALLER_V1.clear()
    memes._INDEX.stamp = None
    yield data_dir
    os.environ.clear()
    os.environ.update(saved)
    cc._ORIGINALS.clear()
    cc._DEFAULT_STORE = None
    cc._CHECKED_INSTALLER_V1.clear()
    memes._INDEX.stamp = None


# ---------------------------------------------------------------------------
# 1. Temperature, chat.v4 default, and installer v1 migration
# ---------------------------------------------------------------------------

def test_chat_agent_uses_085_temperature_and_v4_prompt_by_default(isolated_data, violetdb, monkeypatch):
    from ai import persona_templates as pt
    from ai.agents import ChatAgent

    monkeypatch.delenv("CHAT_LLM_MODEL_TEMPERATURE", raising=False)
    monkeypatch.delenv("CHAT_LLM_PROMPT", raising=False)
    agent = ChatAgent()
    assert agent._config.reply_temperature == 0.85
    assert "抓重点，别面面俱到" in agent._config.reply_prompt
    assert "禁塑料套话" in agent._config.reply_prompt

    from ai.agents.base_agent import _CALL_CONTEXT
    tok = _CALL_CONTEXT.set({})
    try:
        # AI persona uses chat.ai.v1.txt when CHAT_LLM_PROMPT is not locked
        pt.switch(violetdb, "ai", "keep")
        msgs = agent.build_context_messages("你好")
        assert "用你自己本来的方式说话" in msgs[0]["content"]

        # Whole-prompt persona (大肥鱼) uses chat.own.v1.txt
        pt.switch(violetdb, "dafeiyu", "keep")
        msgs = agent.build_context_messages("你好")
        assert "完全按后面“你自己”那一段来" in msgs[0]["content"]
        pt.switch(violetdb, "violet", "keep")
    finally:
        _CALL_CONTEXT.reset(tok)


def test_installer_v1_prompt_is_migrated_while_manual_dashboard_v1_is_kept(isolated_data, monkeypatch, tmp_path):
    from ai.control_config import ControlStore
    from tools import violet_setup

    monkeypatch.setenv("VIOLET_DB_PATH", str(isolated_data / "violet.db"))
    monkeypatch.delenv("CHAT_LLM_PROMPT", raising=False)
    monkeypatch.delenv("PROACTIVE_LLM_PROMPT", raising=False)

    # Fresh setup no longer seeds v1
    res = violet_setup.init()
    assert res["prompts_selected"] == {}

    # Simulate a legacy install where installer wrote v1, then admin saved CHAT_LLM_MODEL on dashboard
    store = ControlStore(isolated_data)
    store.save({"prompt.chat": "v1"}, "installer", "安装时选择稳定的提示词版本")
    store.save({"CHAT_LLM_MODEL": "gemini-2.5-pro"}, "dashboard", "set model")
    assert store.values()["prompt.chat"] == "v1"

    env: dict[str, str] = {}
    store.apply_to_environ(env)
    assert "prompt.chat" not in store.values()
    assert "CHAT_LLM_PROMPT" not in env
    assert env["CHAT_LLM_MODEL"] == "gemini-2.5-pro"
    assert store.history(limit=1)[0]["actor"] == "migration"

    # Manual dashboard choice of v1 MUST be preserved
    store.save({"prompt.chat": "v1"}, "dashboard", "prefer classic v1")
    env2: dict[str, str] = {}
    store.apply_to_environ(env2)
    assert store.values()["prompt.chat"] == "v1"
    assert "你是 Violet" in env2["CHAT_LLM_PROMPT"]


# ---------------------------------------------------------------------------
# 2. Prompt character-budget gate & sticker prefix-cache stability
# ---------------------------------------------------------------------------

def test_prompt_budget_and_sticker_cache_stability(isolated_data, violetdb, monkeypatch):
    from ai import stickers
    from ai.agents import ChatAgent
    from ai.tool_manager import Catalog, abilities_note, direct_note

    monkeypatch.delenv("CHAT_LLM_PROMPT", raising=False)
    # Insert 25 verbose stickers in the 4-field recognition format
    ids = []
    for i in range(25):
        raw = PNG + f"sticker-{i}".encode("utf-8")
        desc = (f"画面：一只白色小猫抱着抱枕翻白眼，上面写着“就这{i}？”；"
                f"什么时候发：对方吹牛、说离谱的话或者开玩笑时用来嫌弃和吐槽；"
                f"关键词：嫌弃、白眼、小猫、吐槽{i}")
        row = stickers.add_bytes(violetdb, raw, source="upload", description=desc)
        ids.append(row["id"])
        if i < 12:
            stickers.mark_used(violetdb, row["id"])

    lines = stickers.prompt_lines(violetdb)
    assert len(lines) == 12
    # Sorted by id ASC for prefix-cache stability, and each line is compact
    line_ids = [int(l.split("：", 1)[0].lstrip("- ")) for l in lines]
    assert line_ids == sorted(line_ids)
    assert all(len(l) <= 52 for l in lines), lines
    assert "关键词" not in "\n".join(lines)

    from ai.agents.base_agent import _CALL_CONTEXT

    agent = ChatAgent()
    cat = Catalog()
    ctx = {"direct_tools": [], "tool_note": direct_note([]), "abilities_note": abilities_note(cat, [])}
    tok = _CALL_CONTEXT.set(dict(ctx))
    try:
        before = agent.build_context_messages("今天上了好多课，好累")[0]["content"]
    finally:
        _CALL_CONTEXT.reset(tok)
    assert len(before) < 3600, f"system prompt bloated to {len(before)} chars"

    # Using an already-top-12 sticker bumps use_count, yet system prompt stays 100% byte-identical!
    stickers.mark_used(violetdb, ids[5])
    stickers.mark_used(violetdb, ids[5])
    tok = _CALL_CONTEXT.set(dict(ctx))
    try:
        after = agent.build_context_messages("那我先歇会儿")[0]["content"]
    finally:
        _CALL_CONTEXT.reset(tok)
    assert after == before


# ---------------------------------------------------------------------------
# 3. Persona aliveness, Violet 4-stage progression, and daily pools
# ---------------------------------------------------------------------------

def test_default_violet_stages_and_immersion_clean_render(isolated_data, violetdb):
    from ai import persona_templates as pt
    from ai.context_engine import ContextEngine
    from ai.persona import PersonaCore, life_config
    from ai.settings import PROJECT_ROOT

    core = PersonaCore.load()
    rendered = core.render()
    assert "Violet 的稳定人设" in rendered
    assert "是一个虚构的女孩" not in rendered
    assert "口吻示例" in rendered and "不要这样" in rendered

    # Stage 0 -> 1 -> 2 -> 3 by chat days on 'close'
    assert pt.stage(None, violetdb)["label"] == "初识慢热"
    for d in range(1, 5):
        violetdb.add_message("user", "incoming", "text", "早", created_at=f"2026-10-{d:02d} 09:00:00")
    assert pt.stage(None, violetdb)["label"] == "熟络分享"
    for d in range(5, 10):
        violetdb.add_message("user", "incoming", "text", "早", created_at=f"2026-10-{d:02d} 09:00:00")
    assert pt.stage(None, violetdb)["label"] == "亲近依赖"
    for d in range(10, 20):
        violetdb.add_message("user", "incoming", "text", "早", created_at=f"2026-10-{d:02d} 09:00:00")
    assert pt.stage(None, violetdb)["label"] == "黏人偏爱"
    assert "阶段（4/4，黏人偏爱）" in "\n".join(ContextEngine(violetdb).build("chat", "在干嘛").sections)

    # Manual stage pin on day 0 must not emit FIRST_MEETING contradiction
    pt.clear_history(violetdb)
    pt.set_stage(None, 3)
    bond = "\n".join(ContextEngine(violetdb).build("chat", "嗨").sections)
    assert "阶段（4/4，黏人偏爱）" in bond and "这是你们第一次聊天" not in bond
    pt.set_stage(None, None)

    # Expanded daily pools (>= 8 per slot across life_config and all 4 anime personas)
    for slot in ("morning", "afternoon", "evening"):
        assert len(life_config()["daily"]["slots"][slot]["items"]) >= 8
    for key in ("stepsister", "childhood", "junior", "quiet"):
        tpl = json.loads((PROJECT_ROOT / "persona" / "templates" / f"{key}.json").read_text(encoding="utf-8"))
        for slot in ("morning", "afternoon", "evening"):
            assert len(tpl["daily"][slot]["items"]) >= 8, (key, slot)


# ---------------------------------------------------------------------------
# 4. ConsistencyGuard: backstory backing + conversational micro-reactions
# ---------------------------------------------------------------------------

def test_consistency_guard_backs_persona_backstory_and_allows_micro_reactions(isolated_data, violetdb):
    from ai import persona_templates as pt
    from ai.consistency import ConsistencyGuard
    from ai.persona import PersonaCore

    pt.switch(violetdb, "stepsister", "keep")
    guard = ConsistencyGuard(violetdb, PersonaCore.load())
    assert guard.check_reply("上次下雨我们还共过一把伞，我半边肩膀都湿了") == []
    assert guard.check_reply("上次你生病我还照顾了你一整夜") == []
    assert [v.check for v in guard.check_reply("我昨天去了海边冲浪")] == ["unrecorded_experience"]

    pt.switch(violetdb, "quiet", "keep")
    guard_q = ConsistencyGuard(violetdb, PersonaCore.load())
    assert guard_q.check_reply("你之前推荐的书和电影我都看了") == []

    pt.switch(violetdb, "violet", "keep")
    guard_v = ConsistencyGuard(violetdb, PersonaCore.load())
    for natural in (
        "笑死，我刚喝了一口水差点喷出来",
        "我看了一眼时间，居然都这么晚了",
        "我拍了下脑门才反应过来",
        "我做了个鬼脸，才不信你",
    ):
        assert guard_v.check_reply(natural) == [], natural
    # Real unrecorded event is still caught
    assert [v.check for v in guard_v.check_reply("我刚喝了一杯珍珠奶茶")] == ["unrecorded_experience"]


# ---------------------------------------------------------------------------
# 5. QQ group wiki-only meme scan (never leaks owner_memes)
# ---------------------------------------------------------------------------

def test_social_engine_scans_wiki_memes_and_never_leaks_owner_memes(isolated_data, violetdb, monkeypatch):
    from ai import memes
    from ai.social.engine import SocialEngine
    from ai.social.store import SocialStore
    from tests.test_memes_violet import ENTRIES, Site
    from tests.test_qq_social_violet import FakeAdapter, NoTimer, say

    memes.sync(fetch=Site(ENTRIES), list_pause=0, detail_pause=0)
    memes.own_save(violetdb, "暗号开摆", "我和主人私下约定的通宵打游戏暗号")

    store = SocialStore(violetdb)
    conv = store.conversation("napcat", "group", "888", "测试群", default_mode="auto")
    adapter = FakeAdapter(owners={"9"})
    runtime = SimpleNamespace(db=violetdb, channels=SimpleNamespace(get=lambda cid: adapter))
    eng = SocialEngine(runtime, timer_factory=NoTimer)
    eng.receive(adapter, kind="group", peer_id="888", sender_id="2", sender_name="群友甲",
                text="暗号开摆！算了还是去码头整点薯条吧", platform_msg_id="501")

    seen: dict = {}
    say(monkeypatch, {"action": "silent", "reason": "test"}, seen)
    eng.decide(conv["id"])

    cached_prefix, turn_suffix = seen["user"]
    assert "去码头整点薯条" in turn_suffix and memes.SOCIAL_HEADER in turn_suffix
    assert memes.SOCIAL_HEADER not in cached_prefix                   # cached prefix stays untouched
    assert "通宵打游戏暗号" not in str(seen["user"])                  # owner_memes NEVER leaked to group!


# ---------------------------------------------------------------------------
# 6. Per-scope VioletRuntime concurrent turn execution
# ---------------------------------------------------------------------------

def test_runtime_processes_different_persona_scopes_concurrently(isolated_data, violetdb, monkeypatch):
    from ai import persona_templates as pt
    from ai.agents import ChatAgent, MemoryAgent
    from ai.channels.base import InboundMessage, Part
    from violet_runtime import MessageSender, VioletRuntime

    monkeypatch.setenv("DEBOUNCE_NORMAL", "0.01")
    monkeypatch.setenv("DEBOUNCE_EXTRA", "0.02")
    monkeypatch.setenv("VIOLET_ASYNC_MEMORY", "0")
    install_fake_llm(monkeypatch, FakeLLM(default_responders("收到")))

    pt.set_channel("telegram", "stepsister")                          # telegram -> scope 'stepsister', wechat -> 'main'

    wechat_entered = threading.Event()
    release_wechat = threading.Event()
    telegram_finished_while_wechat_blocked = []

    def slow_main_fast_other(user_message, **kwargs):
        from ai import memory_scope
        scope = memory_scope.current()
        if scope == "main":
            wechat_entered.set()
            release_wechat.wait(timeout=3.0)
        else:
            if not release_wechat.is_set():
                telegram_finished_while_wechat_blocked.append(scope)
        return {"chat": {"content": f"来自{scope}:{user_message}", "messages": []}, "memory": {}, "errors": {}}

    class TgAdapter:
        channel_id = "telegram"
        sent: list[str] = []

        def is_running(self):
            return True

        def send_text(self, peer_id, text, **_):
            self.sent.append(text)
            return {"ok": True}

        def send_typing(self, peer_id, status=1):
            return True

    tg = TgAdapter()
    client = FakeILinkClient()
    agents = {"chat": ChatAgent(), "memory": MemoryAgent()}
    rt = VioletRuntime(client, violetdb, agents, pipeline_runner=slow_main_fast_other,
                       sender=MessageSender(client, violetdb, simulate_typing=False, sleep=lambda _: None))
    rt.channels = SimpleNamespace(
        get=lambda cid: tg if cid == "telegram" else None,
        accept_policy=lambda cid: "owner",
        context_for=lambda cid: SimpleNamespace(channel=cid, channel_type="telegram", label="Telegram"),
    )
    assert rt.identities.admit("telegram", "u1", accept="first_peer", display_name="owner") == "owner"
    rt.start_worker()
    try:
        rt.ingest_text("微信慢请求")
        assert wechat_entered.wait(timeout=2.0)
        assert rt.is_busy("main") and not rt.is_busy("stepsister")

        rt.ingest_message(InboundMessage(channel="telegram", channel_type="telegram", peer_id="u1", peer_name="owner",
                                         message_id="tg-1", dedup_key="tg:1", parts=[Part("text", "电报快请求")]))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not tg.sent:
            time.sleep(0.02)
        assert tg.sent == ["来自stepsister:电报快请求"]
        assert telegram_finished_while_wechat_blocked == ["stepsister"]

        release_wechat.set()
        assert rt.wait_idle(timeout=2.0)
        assert client.sent_texts == ["来自main:微信慢请求"]
    finally:
        release_wechat.set()
        rt.stop(timeout=2.0)


# ---------------------------------------------------------------------------
# 7. Mood third-party vent & meme filter (grok-4.6 QQ group eval finding)
# ---------------------------------------------------------------------------

def test_mood_ignores_third_party_vents_and_memes(isolated_data, violetdb):
    from ai import mood

    # Complaining about a third party / meme in a group must NOT count as scolding her
    for vent in (
        "今天上班又是吗喽的一天，甲方这需求真是合法但有病",
        "这破游戏太垃圾了，策划脑残吧",
        "那老板有病吧，加班到十点",
        "食堂这菜真恶心",
    ):
        assert mood.scold_strength(vent) == 0.0, vent
        assert mood.feel(violetdb, vent, who="路人甲", weight=0.5) is None

    # Direct scolding directed at her still triggers mood shift
    for scold in ("闭嘴", "你真烦", "你有病吧", "滚", "真是没用"):
        assert mood.scold_strength(scold) > 0.0, scold


# ---------------------------------------------------------------------------
# 8. Stage A: Proactive daily-life loop & persona-stage motive alignment
# ---------------------------------------------------------------------------

def test_proactive_daily_life_materialization_and_stage_motive_alignment(isolated_data, violetdb, monkeypatch):
    from datetime import datetime
    from ai import persona_templates as pt
    from ai.daily_life import DailyLife
    from ai.proactive_scheduler import ProactiveScheduler
    from ai.relationship import RelationshipView, warmth_level

    monkeypatch.setenv("VIOLET_DAILY_LIFE", "1")
    pt.switch(violetdb, "stepsister", "keep")
    now = datetime(2026, 11, 19, 20, 30, 0)
    # Materialize today's daily events even without any chat turn today
    view = DailyLife(violetdb).ensure_today(now)
    assert view is not None
    # Seed one guaranteed today unshared life_daily event + one yesterday life_daily event
    violetdb.execute(
        "INSERT OR REPLACE INTO life_events (uid, event_type, source, content, occurred_at, status, share_count, created_at, updated_at) "
        "VALUES ('daily:2026-11-19:test_today', 'daily', 'life_daily', '下午把阳台上的绿萝换了水', '2026-11-19T15:00:00', 'happened', 0, '2026-11-19T15:00:00', '2026-11-19T15:00:00')"
    )
    violetdb.execute(
        "INSERT OR REPLACE INTO life_events (uid, event_type, source, content, occurred_at, status, share_count, created_at, updated_at) "
        "VALUES ('daily:2026-11-18:test_old', 'daily', 'life_daily', '昨天的旧日常不该再主动翻出来', '2026-11-18T15:00:00', 'happened', 0, '2026-11-18T15:00:00', '2026-11-18T15:00:00')"
    )

    monkeypatch.setattr(violetdb, "now", lambda: "2026-11-19T20:30:00")
    sched = object.__new__(ProactiveScheduler)
    sched.config = {}
    sched.tracker = SimpleNamespace(topic_state=lambda _: {"unengaged": 0, "cooling": False})
    shareable = sched._shareable_life_events()
    daily_items = [e for e in shareable if e.get("type") == "violet_daily_event"]
    assert len(daily_items) == 1
    assert "昨天的旧日常" not in daily_items[0]["content"]

    # Persona stage directly determines warmth_level (1 -> warming, 2 -> active, 3 -> deep)
    assert warmth_level(RelationshipView(stage="new"), {"index": 1, "total": 5, "label": "慢慢靠近"}) == 1
    assert warmth_level(RelationshipView(stage="new"), {"index": 2, "total": 5, "label": "不只是妹妹"}) == 2
    assert warmth_level(RelationshipView(stage="new"), {"index": 3, "total": 5, "label": "藏不住了"}) == 3

    # AI persona skips DailyLife materialization
    pt.switch(violetdb, "ai", "keep")
    assert DailyLife(violetdb).ensure_today(now) is None
    pt.switch(violetdb, "violet", "keep")


# ---------------------------------------------------------------------------
# 9. Stage B: Per-stage signature samples & enriched stage() payload
# ---------------------------------------------------------------------------

def test_per_stage_samples_and_enriched_stage_payload(isolated_data, violetdb):
    from ai import persona_templates as pt
    from ai.context_engine import ContextEngine
    from ai.settings import PROJECT_ROOT

    for key in ("violet", "stepsister", "childhood", "junior", "quiet"):
        data = json.loads((PROJECT_ROOT / "persona" / "templates" / f"{key}.json").read_text(encoding="utf-8"))
        stages = data.get("stages") or []
        assert len(stages) >= 4, key
        for idx, st in enumerate(stages):
            assert str(st.get("sample") or "").strip(), (key, idx)

    pt.switch(violetdb, "stepsister", "keep")
    st = pt.stage(None, violetdb)
    assert st is not None
    assert st["total"] == 5 and st["index"] == 0
    assert st["next_label"] == "慢慢靠近" and st["days_to_next"] == 3
    assert len(st["items"]) == 5 and all(it["sample"] and "{call}" not in it["sample"] for it in st["items"])
    _rel, rel_text = pt.relationship_text(None, violetdb)
    assert "语气感受" in rel_text and st["items"][0]["sample"] in rel_text

    # Default Violet also injects active stage sample in _bond_section
    pt.switch(violetdb, "violet", "keep")
    st_v = pt.stage(None, violetdb)
    bond = "\n".join(ContextEngine(violetdb).build("chat", "你好").sections)
    assert "语气感受" in bond and st_v["items"][0]["sample"] in bond


# ---------------------------------------------------------------------------
# 10. Stage C: Overview persona/mood, proactive recent, & stateless sandbox
# ---------------------------------------------------------------------------

def test_overview_proactive_recent_and_stateless_sandbox_api(isolated_data, violetdb, monkeypatch):
    from ai import mood, persona_templates as pt
    from ai.agents import ChatAgent
    from wx_ilink_py import ILinkClient, create_app

    token = os.environ["VIOLET_ADMIN_TOKEN"]
    headers = {"Authorization": f"Bearer {token}", "Origin": "http://localhost"}
    install_fake_llm(monkeypatch, FakeLLM(default_responders("刚收完碗\n怎么啦")))

    pt.switch(violetdb, "stepsister", "keep")
    mood.shift(violetdb, 0.45, "主人夸了你")
    today = violetdb.now()[:10]
    violetdb.execute(
        "INSERT INTO life_events (uid, event_type, source, content, occurred_at, status, share_count, created_at, updated_at) "
        "VALUES ('daily:api:1', 'daily', 'life_daily', '把桌上的书理了一遍', ?, 'happened', 0, ?, ?)",
        (f"{today}T10:00:00", f"{today}T10:00:00", f"{today}T10:00:00"),
    )
    violetdb.execute(
        "INSERT INTO proactive_events (event_type, status, action_type, reason, decision, sent_messages, created_at, updated_at) "
        "VALUES ('scheduled_check', 'sent', 'chat', 'daily_share', ?, ?, ?, ?)",
        (
            json.dumps({"should_act": True, "topic_key": "理书桌", "reason": "顺嘴聊一句"}, ensure_ascii=False),
            json.dumps(["刚把书桌理好了"], ensure_ascii=False),
            f"{today}T11:00:00",
            f"{today}T11:00:00",
        ),
    )

    chat_agent = ChatAgent()
    app = create_app(
        client=ILinkClient(data_dir=isolated_data, auto_login=False),
        violetdb=violetdb,
        proactive_scheduler=None,
        all_agents=[chat_agent],
        admin_token=token,
    )
    client = app.test_client()

    # 1. Overview exposes persona (with stage items), mood, and synthesized proactive summary
    ov = client.get("/api/control/overview", headers=headers).get_json()["data"]
    assert ov["persona"]["stage"]["total"] == 5
    assert len(ov["persona"]["stage"]["items"]) == 5
    assert ov["mood"]["label"] == "心情不错" and ov["mood"]["state"] == "ok"
    assert ov["proactive"]["daily_sent_count"] == 1

    # 2. Proactive recent returns status, today's daily count, and decision events cleanly
    pro = client.get("/api/control/proactive/recent", headers=headers).get_json()["data"]
    assert pro["status"]["daily_sent_count"] == 1
    assert pro["today_daily"]["total"] == 1 and pro["today_daily"]["unshared"] == 1
    assert len(pro["events"]) == 1 and pro["events"][0]["topic"] == "理书桌"

    # 3. Sandbox preview works for any template/stage with ZERO DB side effects
    before_msgs = violetdb.fetch_one("SELECT COUNT(*) AS n FROM messages")["n"]
    before_jobs = violetdb.fetch_one("SELECT COUNT(*) AS n FROM memory_jobs")["n"]
    before_mood = mood.current(violetdb)["value"]

    assert client.post("/api/persona/sandbox", json={"message": ""}, headers=headers).status_code == 400
    assert client.post("/api/persona/sandbox", json={"message": "在吗", "template": "stepsister", "stage": 99}, headers=headers).status_code == 400

    sb = client.post(
        "/api/persona/sandbox",
        json={"message": "在吗", "template": "quiet", "stage": 2, "history": [{"role": "user", "content": "嗨"}]},
        headers=headers,
    )
    assert sb.status_code == 200
    body = sb.get_json()
    assert body["messages"] == ["刚收完碗", "怎么啦"]
    assert body["stage_index"] == 2 and body["stage_label"] == "不安"
    assert body["prompt_chars"] > 500 and len(body["prompt_parts"]) >= 2

    assert violetdb.fetch_one("SELECT COUNT(*) AS n FROM messages")["n"] == before_msgs
    assert violetdb.fetch_one("SELECT COUNT(*) AS n FROM memory_jobs")["n"] == before_jobs
    assert mood.current(violetdb)["value"] == before_mood


# ---------------------------------------------------------------------------
# 11. Stage-scaled progressive warmth, distance recovery & class/quick-reply rules (v0.7.77)
# ---------------------------------------------------------------------------

def test_stage_scaled_proactive_warmth_pace_and_daily_life(isolated_data, violetdb, monkeypatch):
    from datetime import date, datetime, timezone, timedelta
    from ai import persona_templates as pt
    from ai.daily_life import DailyLife
    from ai.profile_store import Profile
    from ai.proactive_scheduler import ProactiveScheduler
    from ai.relationship import RelationshipView, warmth_level

    monkeypatch.setenv("VIOLET_DAILY_LIFE", "1")
    pt.switch(violetdb, "stepsister", "keep")
    sched = object.__new__(ProactiveScheduler)
    sched.config = {"min_minutes_after_user_message": 90, "min_minutes_between_proactive_messages": 180,
                    "random_delay_minutes": {"evening": [90, 180]}}
    sched.profile = Profile(violetdb)
    sched.tracker = SimpleNamespace(topic_state=lambda _: {"unengaged": 0, "cooling": False})

    # Stage 0: polite shipped defaults unchanged (90 / 180)
    pt.set_stage(None, 0)
    after0, between0, lvl0, _ = sched._effective_intervals()
    assert (after0, between0, lvl0) == (90.0, 180.0, 0)

    # Stage 1 -> warming (60 / 120)
    pt.set_stage(None, 1)
    after1, between1, lvl1, _ = sched._effective_intervals()
    assert (lvl1, round(after1), round(between1)) == (1, 60, 121)

    # Stage 2 ("不只是妹妹") -> active warmth (45 / 90)
    pt.set_stage(None, 2)
    after2, between2, lvl2, _ = sched._effective_intervals()
    assert (lvl2, after2, between2) == (2, 45.0, 90.0)

    # Stage 3+ ("藏不住了") -> deep warmth (36 / 72)
    pt.set_stage(None, 3)
    after3, between3, lvl3, _ = sched._effective_intervals()
    assert (lvl3, after3, between3) == (3, 36.0, 72.0)

    # Distance view always drops warmth back to 0
    assert warmth_level(RelationshipView(stage="distance"), {"index": 3, "total": 5}) == 0

    # DailyLife at Stage 2 never rolls a 0-item quiet day and plans all 3 daily slots
    pt.set_stage(None, 2)
    dl = DailyLife(violetdb)
    dl.daily = {**dl.daily, "quiet_day_probability": 0.99}
    tz = timezone(timedelta(hours=8))
    plan = dl.plan_for(date(2026, 11, 23), tz)
    assert len(plan) == 3

    # Single-call proactive note includes warmth level and non-DND note when VIOLET_SCHEDULE_PROACTIVE=0
    monkeypatch.setenv("VIOLET_SCHEDULE_PROACTIVE", "0")
    note, _ = sched._compose_proactive_note(
        view=RelationshipView(stage="warming", chat_days=10), level=2, since_user=180.0,
        sent_today=1, daily_limit=5, unanswered=[], life_events=[], upcoming=[], open_plans=[],
        web_findings=[], schedule_moments=[], recent=[], habit_note=None, class_ok=True,
    )
    assert "熟络活跃" in note and "打游戏" in note and "上课都不算勿扰" in note

    # Explicit "less" preference disables interval speedup
    sched.profile.preferences.set_explicit("proactive_chat", "less", evidence="少主动发", user_message="以后少主动发消息")
    after_less, between_less, _, _ = sched._effective_intervals()
    assert (after_less, between_less) == (90.0, 180.0)


def test_distance_preference_and_admin_clear_api(isolated_data, violetdb, monkeypatch):
    from ai import persona_templates as pt
    from ai.agents import ChatAgent
    from ai.profile_store import Profile
    from ai.relationship import assess
    from tests.fakes import FakeLLM, default_responders, install_fake_llm
    from wx_ilink_py import ILinkClient, create_app

    pt.switch(violetdb, "stepsister", "keep")
    # 1. Casual / joke messages NEVER trigger regex distance lock anymore
    violetdb.add_user_message("text", "哈哈别太黏人了，这台词好肉麻")
    assert assess(violetdb).stage != "distance"

    # 2. Explicit closeness=distance recorded by MemoryAgent activates distance stage cleanly
    pref_res = Profile(violetdb).preferences.set_explicit(
        "closeness", "distance", evidence="以后保持点距离，别撒娇了",
        user_message="我们以后保持点距离，别撒娇了",
    )
    assert pref_res["action"] == "set"
    assert assess(violetdb).stage == "distance"

    # 3. Distance status shows up in /api/persona/templates and clears cleanly via POST /api/persona/distance/clear
    token = "test-distance-token"
    headers = {"Authorization": f"Bearer {token}", "Origin": "http://localhost"}
    install_fake_llm(monkeypatch, FakeLLM(default_responders("嗯")))
    app = create_app(
        client=ILinkClient(data_dir=isolated_data, auto_login=False),
        violetdb=violetdb,
        proactive_scheduler=None,
        all_agents=[ChatAgent()],
        admin_token=token,
    )
    client = app.test_client()
    tpls = client.get("/api/persona/templates", headers=headers).get_json()
    assert tpls["in_effect"]["relationship_state"]["stage"] == "distance"

    cleared = client.post("/api/persona/distance/clear", json={}, headers=headers)
    assert cleared.status_code == 200
    assert cleared.get_json()["in_effect"]["relationship_state"]["stage"] != "distance"
    assert assess(violetdb).stage != "distance"


def test_class_situation_and_quick_reply_cold_streak(isolated_data, violetdb, monkeypatch):
    from ai.conversation_state import TopicTracker
    from ai.profile_store import Profile
    from ai.proactive_scheduler import ProactiveScheduler

    sched = object.__new__(ProactiveScheduler)
    sched.profile = Profile(violetdb)

    # Ordinary class activity does NOT block when VIOLET_SCHEDULE_PROACTIVE is OFF
    monkeypatch.setenv("VIOLET_SCHEDULE_PROACTIVE", "0")
    sched.profile.situations.observe("activity", "在上课", "user_stated", evidence="在上课", user_message="我在上课好无聊")
    assert sched._busy_situation() is None

    # When VIOLET_SCHEDULE_PROACTIVE is ON, class activity DOES block
    monkeypatch.setenv("VIOLET_SCHEDULE_PROACTIVE", "1")
    assert sched._busy_situation() is not None

    # Explicit DND / busy / exam ALWAYS blocks even with VIOLET_SCHEDULE_PROACTIVE=0
    monkeypatch.setenv("VIOLET_SCHEDULE_PROACTIVE", "0")
    sched.profile.situations.observe("availability", "上课别打扰", "user_stated", evidence="上课别打扰", user_message="上课别打扰我")
    assert sched._busy_situation() is not None

    # Quick 2-turn short reply in active exchange resets cold_streak
    tracker = TopicTracker(violetdb)
    violetdb.add_proactive_event("scheduled_proactive_check", status="sent", action_type="chat")
    tracker.record_eve_topic("daily_life")
    tracker.evaluate("嗯")
    assert tracker.cold_streak() == 1
    violetdb.add_assistant_message("text", "那你先忙呀")
    res = tracker.evaluate("好")
    assert res["continued_chat"] is True
    assert tracker.cold_streak() == 0


def test_chat_agent_compose_never_starts_with_bare_assistant_turn(isolated_data, violetdb):
    from ai.agents.chat_agent import ChatAgent, RECORD_NOTE, compose

    # Leading assistant turn without marks gets a RECORD_NOTE user separator first
    msgs = compose(["规则"], [{"role": "assistant", "content": "刚煮了一锅银耳羹"}], [], "哈哈小心烫到")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"].startswith(RECORD_NOTE)
    assert msgs[2]["content"] == "刚煮了一锅银耳羹"
    assert msgs[3]["content"] == "哈哈小心烫到"

    # Multiple leading assistant turns merge cleanly after one separator
    msgs2 = compose(
        ["规则"],
        [
            {"role": "assistant", "content": "上午好呀"},
            {"role": "assistant", "content": "刚逛完公园回来"},
            {"role": "user", "content": "早呀"},
        ],
        [],
        "午饭吃什么",
    )
    assert [m["role"] for m in msgs2] == ["system", "user", "assistant", "user"]
    assert msgs2[2]["content"] == "上午好呀\n刚逛完公园回来"

    # Leading system mark already emits a user RECORD_NOTE turn, so no duplicate separator is added
    msgs3 = compose(
        ["规则"],
        [
            {"role": "system", "content": "【聊天记录：这里隔了 3 小时】"},
            {"role": "assistant", "content": "醒了没"},
        ],
        [],
        "刚醒",
    )
    assert [m["role"] for m in msgs3] == ["system", "user", "assistant", "user"]
    assert "隔了 3 小时" in msgs3[1]["content"] and "接着之前的聊天" not in msgs3[1]["content"]

    # End-to-end ChatAgent.build_context_messages with only an assistant message in history
    violetdb.add_assistant_message("text", "刚才路过花店买了束小雏菊")
    from ai.agents.base_agent import _CALL_CONTEXT
    tok = _CALL_CONTEXT.set({})
    try:
        built = ChatAgent().build_context_messages("好看吗")
    finally:
        _CALL_CONTEXT.reset(tok)
    roles = [m["role"] for m in built]
    assert roles[0] == "system" and roles[1] == "user" and roles[-1] == "user"
    assert all(a != b for a, b in zip(roles[1:], roles[2:])), roles
    assert any(m["role"] == "assistant" and "小雏菊" in m["content"] for m in built)



