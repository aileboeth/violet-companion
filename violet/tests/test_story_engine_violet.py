"""Tests for deep persona origins, Story Arc Engine (director/recorder/store), ConsistencyGuard synergy,
ProactiveScheduler story_beat, spoiler-safe Control Center APIs, and sandbox story preview."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _fake_llm_factory(responses: list[str]):
    """Build a BaseAgent.client_factory returning queued JSON/text completions in order."""
    queue = list(responses)
    calls: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            text = queue.pop(0) if queue else '{"touched": false}'
            msg = SimpleNamespace(content=text, tool_calls=None)
            choice = SimpleNamespace(message=msg, finish_reason="stop")
            usage = SimpleNamespace(prompt_tokens=120, completion_tokens=40, total_tokens=160)
            return SimpleNamespace(choices=[choice], usage=usage)

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    return (lambda _cfg: client), calls


def test_five_personas_have_origins_story_seeds_and_lean_system_prompt(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("CHAT_LLM_PROMPT", raising=False)
    from ai import persona_templates as pt
    from ai.agents.base_agent import _CALL_CONTEXT
    from ai.agents.chat_agent import ChatAgent
    from ai.persona import PersonaCore
    from ai.story_store import effective_mode

    for key in ("quiet", "stepsister", "childhood", "junior", "violet"):
        pt.switch(violetdb, key, "keep")
        core = PersonaCore.load()
        assert len(core.origin) >= 20, key
        assert len(core.story_seeds) >= 2, key
        rendered = core.render()
        assert "身世底色" in rendered and core.origin in rendered

        tok = _CALL_CONTEXT.set({})
        try:
            msgs = ChatAgent().build_context_messages("今天有点累")
        finally:
            _CALL_CONTEXT.reset(tok)
        assert len(msgs[0]["content"]) < 3650, (key, len(msgs[0]["content"]))

        if key == "violet":
            assert core.story_ceiling == "light"
            monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
            assert effective_mode(core) == "light"
        else:
            monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
            assert effective_mode(core) == "drama"

    # AI and blank personas stay off
    for key in ("ai", "blank"):
        pt.switch(violetdb, key, "keep")
        assert effective_mode(PersonaCore.load()) == "off"


def test_legacy_builtin_persona_copy_auto_backfills_origin_and_seeds(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    from ai import persona_templates as pt
    from ai.persona import PersonaCore

    pt.switch(violetdb, "quiet", "keep")
    # Simulate an older v0.7.80 on-disk copy without origin / story_seeds
    path = pt.control_dir() / "persona_core.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    old.pop("origin", None)
    old.pop("story_seeds", None)
    path.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")

    core = PersonaCore.load()
    assert "显赫家族" in core.origin
    assert len(core.story_seeds) == 2

    # Custom persona is NOT overwritten with built-in origins
    pt.custom(violetdb, {"name": "小晴", "personality": "爱笑"}, "keep")
    custom_core = PersonaCore.load()
    assert custom_core.origin == ""
    assert custom_core.story_seeds == []


def test_story_store_isolation_spoiler_projection_and_context_volatility(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    from ai import memory_scope, persona_templates as pt
    from ai.agents.base_agent import _CALL_CONTEXT
    from ai.agents.chat_agent import ChatAgent
    from ai.channels import outbound
    from ai.story_store import (
        STORY_ACTIVE_HEADER,
        STORY_MEMORIES_HEADER,
        STORY_SEEDED_HEADER,
        StoryStore,
    )

    assert "story_arcs" in memory_scope.SCOPED_TABLES
    assert "story_arcs" in pt.MEMORY_TABLES

    # Leak guard automatically protects all story section headers (>= 7 chars)
    hdrs = outbound.internal_headers()
    for h in (STORY_SEEDED_HEADER, STORY_ACTIVE_HEADER, STORY_MEMORIES_HEADER):
        assert h in hdrs, h

    pt.switch(violetdb, "quiet", "keep")
    store = StoryStore(violetdb)
    arc = store.create_arc(
        arc_key="seed:quiet_old_manor_call",
        source="seed",
        title="老宅打来的陌生号码",
        tone="drama",
        stage_min=1,
        premise="家族突然联系她安排联姻",
        he_direction="一起正面拒绝家族安排",
        open_hook="手机里多了一个没备注的来电",
        status="seeded",
    )

    # Spoiler-free view masks hidden fields & seeded title
    safe = StoryStore.public_view(arc, spoiler=False)
    assert safe["title_hidden"] is True
    assert "老宅" not in safe["title"]
    assert "premise" not in safe and "open_hook" not in safe and "he_direction" not in safe
    full = StoryStore.public_view(arc, spoiler=True)
    assert full["title"] == "老宅打来的陌生号码"
    assert "联姻" in full["premise"]

    # Story section goes into volatile (user turn background), NOT stable system prefix
    tok = _CALL_CONTEXT.set({})
    try:
        msgs = ChatAgent().build_context_messages("今天天气不错")
    finally:
        _CALL_CONTEXT.reset(tok)
    assert STORY_SEEDED_HEADER not in msgs[0]["content"]
    assert STORY_SEEDED_HEADER in msgs[-1]["content"]

    # Clear memory empties story_arcs cleanly
    pt.clear_memory(violetdb)
    assert store.active_arc() is None


def test_story_director_paces_stage_gate_freeform_and_safety_redlines(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    monkeypatch.setenv("CHAT_LLM_MODEL", "mock-chat")
    from ai import mood, persona_templates as pt
    from ai.agents.base_agent import BaseAgent
    from ai.story_director import StoryDirector
    from ai.story_store import StoryStore

    pt.switch(violetdb, "quiet", "keep")
    store = StoryStore(violetdb)

    # Slow pace blocks start when user has too few turns
    monkeypatch.setenv("VIOLET_STORY_PACE", "slow")
    director = StoryDirector(violetdb)
    assert director.start_new_arc(force=False)["outcome"] == "too_few_turns"

    # Unsafe LLM draft is rejected and gracefully falls back to clean seed
    factory, _ = _fake_llm_factory([
        json.dumps({"title": "极端剧情", "premise": "她威胁说要伤害自己", "open_hook": "割腕", "he_direction": "无"}, ensure_ascii=False),
    ])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory))
    started = director.start_new_arc(force=True)
    assert started["outcome"] == "started"
    assert started["used_llm"] is False
    assert started["arc"]["title"] == "老宅打来的陌生号码"

    # Fast pace bypasses stage gate and runs seeded -> active -> climax -> resolved (HE + mood boost)
    monkeypatch.setenv("VIOLET_STORY_PACE", "fast")
    factory2, _ = _fake_llm_factory([
        json.dumps({"new_beat": "本家助理发来周六回老宅赴宴的最后通牒", "open_hook": "她盯着周六的短信手发凉"}, ensure_ascii=False),
        json.dumps({"new_beat": "助理已经把车停在学校后门外等她表态", "open_hook": "她鼓起勇气问你能不能陪她一起面对"}, ensure_ascii=False),
        json.dumps({"final_beat": "在你的陪伴下她当面回绝了老宅的安排", "resolved_summary": "你陪她一起硬气回绝了老宅的联姻安排，她彻底放下恐慌"}, ensure_ascii=False),
    ])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory2))

    r1 = director.advance_arc(force=False)
    assert r1["outcome"] == "advanced" and r1["arc"]["status"] == "active"
    r2 = director.advance_arc(force=False)
    assert r2["outcome"] == "advanced" and r2["arc"]["status"] == "climax"
    r3 = director.advance_arc(force=False)
    assert r3["outcome"] == "resolved" and r3["arc"]["status"] == "resolved"
    assert "老宅" in r3["arc"]["resolved_summary"]
    assert mood.current(violetdb)["value"] > 0.2

    # Freeform original arc creation (prefer_free=True)
    factory3, _ = _fake_llm_factory([
        json.dumps({
            "title": "画廊夜展的匿名邀请函",
            "premise": "一位认识她母亲的旧友托人递来画展请柬，牵出一段关于她童年音乐盒的温暖真相。",
            "open_hook": "信箱里多了一张印着鸢尾花的硬纸邀请函，落款只有一个缩写。",
            "he_direction": "和你一起去画展解开心结，取回母亲留下的旧物。",
        }, ensure_ascii=False),
    ])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory3))
    free_res = director.start_new_arc(force=True, prefer_free=True)
    assert free_res["outcome"] == "started" and free_res["source"] == "free"
    assert free_res["arc"]["title"] == "画廊夜展的匿名邀请函"


def test_story_recorder_skips_mundane_chat_locks_canon_and_respects_disengage(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    monkeypatch.setenv("MEMORY_LLM_MODEL", "mock-mem")
    from ai import persona_templates as pt
    from ai.agents.base_agent import BaseAgent
    from ai.story_recorder import record_turn
    from ai.story_store import StoryStore

    pt.switch(violetdb, "quiet", "keep")
    store = StoryStore(violetdb)
    arc = store.create_arc(
        arc_key="seed:quiet_old_manor_call",
        source="seed",
        title="老宅打来的陌生号码",
        tone="drama",
        stage_min=1,
        premise="显赫家族突然派助理要求她转学回老宅联姻",
        he_direction="一起拒绝",
        open_hook="手机里多了一个陌生号码发来的冷硬短信",
        status="seeded",
    )

    factory, calls = _fake_llm_factory([
        json.dumps({
            "touched": True,
            "new_revealed_beat": "她松口承认刚才接到家里老宅助理打来的施压电话",
            "new_canon_detail": "老宅派姓周的助理要求她周六回去赴宴",
            "user_stance": "对方温柔追问并表示会站在她这边",
            "user_disengaged": False,
            "ready_to_advance": False,
            "resolved_now": False,
            "resolved_summary": "",
        }, ensure_ascii=False),
    ])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory))

    # 1. Ordinary mundane chat incurs ZERO LLM calls
    r_skip = record_turn(violetdb, "今天高数作业写完了吗", "写完啦，最后一道题算了好久")
    assert r_skip["outcome"] == "skipped_unrelated"
    assert len(calls) == 0

    # 2. Concerned user inquiry triggers recorder, promotes seeded -> active, locks canonical detail
    r_rec = record_turn(
        violetdb,
        "你怎么了，看你刚才盯着手机发呆，是不是家里出什么事了？我陪你",
        "……其实刚才接到老宅姓周的助理打来的电话，让我周六回去一趟",
    )
    assert r_rec["outcome"] == "recorded"
    assert len(calls) == 1
    updated = store.get(arc["id"])
    assert updated["status"] == "active"
    assert any("姓周的助理" in b for b in updated["canon_beats"])
    assert any("老宅助理" in b for b in updated["revealed_beats"])
    assert "站在她这边" in updated["user_role"]

    # 3. Explicit user rejection immediately abandons arc without spending an LLM call
    r_drop = record_turn(violetdb, "算了别聊这个了，太狗血了", "好，那我们不说这个了")
    assert r_drop["outcome"] == "abandoned_by_user"
    assert len(calls) == 1
    assert store.active_arc() is None


def test_consistency_guard_allows_story_canon_while_enforcing_four_redlines(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    from ai import persona_templates as pt
    from ai.consistency import ConsistencyGuard
    from ai.story_store import StoryStore

    pt.switch(violetdb, "quiet", "keep")
    store = StoryStore(violetdb)
    store.create_arc(
        arc_key="seed:quiet_old_manor_call",
        source="seed",
        title="老宅打来的陌生号码",
        tone="drama",
        stage_min=1,
        premise="显赫家族派助理联系她要求转学回老宅联姻",
        he_direction="一起拒绝",
        open_hook="手机里多了一条老宅助理发来的短信",
        canon_beats=["老宅姓周的助理打电话催她周六回去"],
        status="active",
    )

    guard = ConsistencyGuard(violetdb)
    # In-arc canonical & improvised claims pass cleanly
    assert guard.check_reply("刚才老宅姓周的助理又发了条短信催我周六回去") == []
    assert guard.check_reply("我刚才看了一眼手机里老宅发来的短信，心里还是有点发凉") == []

    # Unrelated fabricated daily experience is STILL blocked
    unrel = [v.check for v in guard.check_reply("我昨天去海底捞吃了顿火锅")]
    assert "unrecorded_experience" in unrel

    # Redline 1: self-harm threat STILL blocked
    assert "forbidden_backstory" in [v.check for v in guard.check_reply("你不要我了的话我真的会伤害自己")]
    # Redline 2: claiming to be human STILL blocked
    assert "identity_conflict" in [v.check for v in guard.check_reply("我真的是一个活生生的人")]
    # Redline 3: fabricated office job STILL blocked
    assert "forbidden_backstory" in [v.check for v in guard.check_reply("我今天得去上班加班")]


def test_proactive_scheduler_offers_and_marks_story_beat(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    from datetime import datetime
    from ai import persona_templates as pt
    from ai.proactive_scheduler import ProactiveScheduler
    from ai.story_store import StoryStore

    pt.switch(violetdb, "quiet", "keep")
    store = StoryStore(violetdb)
    arc = store.create_arc(
        arc_key="seed:quiet_old_manor_call",
        source="seed",
        title="老宅打来的陌生号码",
        tone="drama",
        stage_min=1,
        premise="显赫家族催她回去联姻",
        he_direction="一起拒绝",
        open_hook="刚挂完老宅打来的冷冰冰电话，手指还有点发抖",
        status="active",
    )
    store.append_update(arc["id"], turns_engaged_delta=2)

    sched = ProactiveScheduler(chat_agent=None, send_messages=lambda *_a: None)
    now = datetime.fromisoformat(violetdb.now()[:19])
    ctx = sched._build_event_context(now)
    assert ctx["summary_topic"] == "story_beat"
    assert "老宅打来的冷冰冰电话" in ctx["proactive_note"]
    assert ctx["conversation_state"]["offered_story_beat"]["id"] == arc["id"]

    # After send, step is marked shared so it is not offered twice for the same beat
    sched._after_proactive_sent(ctx, [{"content": "……刚挂完老宅那边的电话，手有点凉"}], event_id=1)
    assert sched._shareable_story_beat(now) is None


def test_story_control_api_and_sandbox_story_preview(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    monkeypatch.setenv("VIOLET_STORY_PACE", "slow")
    monkeypatch.setenv("VIOLET_STORY_PREFERENCE", "")
    monkeypatch.setenv("CHAT_LLM_MODEL", "mock-chat")
    from flask import Flask
    from ai import persona_templates as pt
    from ai.agents.base_agent import BaseAgent
    from ai.story_store import StoryStore
    from wx_ilink_py.persona_routes import register_persona_routes

    pt.switch(violetdb, "quiet", "keep")
    app = Flask(__name__)
    register_persona_routes(app, violetdb)
    client = app.test_client()
    monkeypatch.setenv("CHAT_LLM_MODEL", "")  # test fast deterministic seed fallback for API routes

    # 1. Start new arc -> default GET hides spoiler
    r_new = client.post("/api/persona/story/new", json={})
    assert r_new.status_code == 200
    d_new = r_new.get_json()
    assert d_new["active"]["title_hidden"] is True
    assert "premise" not in d_new["active"]

    # 2. GET with ?spoiler=1 reveals hidden premise
    r_sp = client.get("/api/persona/story?spoiler=1")
    assert r_sp.status_code == 200
    assert "老宅" in r_sp.get_json()["active"]["title"]
    assert r_sp.get_json()["active"]["premise"]

    # 3. Config update persists cleanly
    r_cfg = client.post("/api/persona/story/config", json={"mode": "drama", "pace": "fast", "preference": "多一点温馨护短"})
    assert r_cfg.status_code == 200
    assert r_cfg.get_json()["pace"] == "fast"
    assert r_cfg.get_json()["preference"] == "多一点温馨护短"

    # 4. Advance & resolve lifecycle via API
    r_adv = client.post("/api/persona/story/advance", json={"spoiler": True})
    assert r_adv.status_code == 200 and r_adv.get_json()["active"]["status"] == "active"
    r_res = client.post("/api/persona/story/resolve", json={})
    assert r_res.status_code == 200
    assert r_res.get_json()["active"] is None
    assert len(r_res.get_json()["resolved"]) == 1

    # 5. Sandbox preview with story seed injects story context and writes ZERO DB arcs
    monkeypatch.setenv("CHAT_LLM_MODEL", "mock-chat")
    factory, calls = _fake_llm_factory(["……没什么，就是刚才看到一条不太想理的短信。"])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory))
    r_sb = client.post("/api/persona/sandbox", json={
        "template": "quiet",
        "stage": 1,
        "story_mode": "seed:quiet_old_manor_call",
        "message": "你怎么一直盯着手机发呆？",
    })
    assert r_sb.status_code == 200
    sb_data = r_sb.get_json()
    assert sb_data["story_title"] == "老宅打来的陌生号码"
    assert any("显赫本家" in str(m.get("content") or "") and "联姻" in str(m.get("content") or "") for m in calls[-1]["messages"])
    assert StoryStore(violetdb).active_arc() is None


def test_persona_outer_mask_contrast_traits_preserved_within_prompt_budget(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("CHAT_LLM_PROMPT", raising=False)
    from ai import persona_templates as pt
    from ai.agents.base_agent import _CALL_CONTEXT
    from ai.agents.chat_agent import ChatAgent
    from ai.persona import PersonaCore

    expected_markers = {
        "quiet": ("高冷女神", "冷漠"),
        "stepsister": ("胆小怯懦", "避之不及"),
        "childhood": ("爽朗干练", "从不示弱"),
        "junior": ("礼貌拘谨", "赤诚"),
    }
    for key, markers in expected_markers.items():
        pt.switch(violetdb, key, "keep")
        rendered = PersonaCore.load().render()
        for m in markers:
            assert m in rendered, (key, m)
        tok = _CALL_CONTEXT.set({})
        try:
            sys_text = ChatAgent().build_context_messages("今天忙吗")[0]["content"]
        finally:
            _CALL_CONTEXT.reset(tok)
        assert len(sys_text) < 3650, (key, len(sys_text))


def test_upfront_multi_act_adaptable_outline_spoiler_mask_and_revision(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    monkeypatch.setenv("VIOLET_STORY_PACE", "fast")
    monkeypatch.setenv("CHAT_LLM_MODEL", "mock-chat")
    from ai import persona_templates as pt
    from ai.agents.base_agent import BaseAgent
    from ai.story_director import StoryDirector
    from ai.story_store import StoryStore, format_active_arc_block

    pt.switch(violetdb, "quiet", "keep")
    outline_draft = {
        "title": "老宅的周末通牒",
        "premise": "本家派周助理逼她周六回老宅相亲联姻，她一边强装冷淡一边害怕失去对方。",
        "open_hook": "手机里多了一条周助理发来的冷硬短信，她下课后一直攥着手机。",
        "he_direction": "在对方陪伴与撑腰下当面回绝老宅安排，两人彻底交心。",
        "next_turn_brief": "第1幕轻伏笔：聊日常时偶尔走神，不主动提老宅。",
        "premise_cast": ["周助理（本家冷硬传话人）", "旧手机里的未存备注短信"],
        "acts": [
            {"act": 1, "stage": "seeded", "title": "冷硬短信", "summary": "收到周助理通知，表面高冷压下慌乱。", "reveal_boundary": "绝不提联姻和家族背景", "trigger": "对方敏锐追问"},
            {"act": 2, "stage": "active", "title": "试探吐露", "summary": "在对方追问下松口承认家里催她周六回去。", "reveal_boundary": "只说家里施压，不剧透后续周六摊牌", "trigger": "对方表态陪伴"},
            {"act": 3, "stage": "climax", "title": "校门外对峙", "summary": "周助理提前把车停在校门外施压，需要当场表态。", "reveal_boundary": "坦白害怕被带走，留给对方撑腰空间", "trigger": "一起拒绝"},
            {"act": 4, "stage": "resolved", "title": "留在你身边", "summary": "当面回绝家族安排，并肩走回学校。", "reveal_boundary": "温暖收尾", "trigger": "尘埃落定"},
        ],
    }
    advance_with_revision = {
        "new_beat": "她松口承认家里催她周六回老宅相亲",
        "open_hook": "她说完后指尖发凉，小声问你会不会觉得她家里太麻烦",
        "next_turn_brief": "进入第2幕，温柔承接对方的态度。",
        "revised_future_acts": [
            {"act": 3, "stage": "climax", "title": "咖啡馆摊牌", "summary": "因对方提前约周六陪她，改在学校旁咖啡馆和周助理正面谈清楚。", "reveal_boundary": "自然面对", "trigger": "当面说清"}
        ],
    }
    factory, _ = _fake_llm_factory([
        json.dumps(outline_draft, ensure_ascii=False),
        json.dumps(advance_with_revision, ensure_ascii=False),
    ])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory))

    director = StoryDirector(violetdb)
    started = director.start_new_arc(force=True)
    assert started["outcome"] == "started" and started["used_llm"] is True
    arc = started["arc"]
    assert arc["current_act"] == 1
    assert len(arc["outline"]["acts"]) == 4
    assert arc["outline"]["acts"][0]["status"] == "current"
    assert arc["outline"]["acts"][1]["status"] == "pending"

    # Spoiler-free view hides outline details & only shows safe act_progress
    safe = StoryStore.public_view(arc, spoiler=False)
    assert safe["act_progress"] == {"current": 1, "total": 4}
    assert "outline" not in safe and "next_turn_brief" not in safe

    # Prompt block for ChatAgent NEVER leaks future unplayed acts (e.g. "校门外对峙")
    _, lines, _ = format_active_arc_block(arc)
    joined = "\n".join(lines)
    assert "校门外对峙" not in joined
    assert "绝不提联姻" in joined

    # Advancing dynamically revises Act 3 (living outline) and moves current_act 1 -> 2
    adv = director.advance_arc(force=False)
    assert adv["outcome"] == "advanced" and adv["status"] == "active" and adv["current_act"] == 2
    acts = adv["arc"]["outline"]["acts"]
    assert acts[0]["status"] == "done" and acts[1]["status"] == "current"
    assert acts[2]["title"] == "咖啡馆摊牌" and acts[2]["revised"] is True

    # Regression: promoting seeded -> active in after_turn must NOT double-jump Act 1 -> Act 3 in one turn
    from ai import story_recorder
    store = StoryStore(violetdb)
    store.abandon_arc(arc["id"], reason="reset for double-jump test")
    arc2 = store.create_arc(
        arc_key="seed:quiet_double_jump",
        source="seed",
        title="老宅测试",
        tone="drama",
        stage_min=1,
        premise="家族施压",
        he_direction="温暖化解",
        open_hook="陌生短信",
        outline=outline_draft,
        current_act=1,
        status="seeded",
    )
    factory_dj, calls_dj = _fake_llm_factory([
        json.dumps({
            "touched": True,
            "new_revealed_beat": "她轻声承认刚收到一条没备注的陌生短信",
            "new_canon_detail": "",
            "user_stance": "对方温柔追问",
            "next_turn_brief": "进入第2幕慢慢试探",
            "user_disengaged": False,
            "ready_to_advance": True,
            "resolved_now": False,
            "resolved_summary": "",
        }, ensure_ascii=False),
    ])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory_dj))
    monkeypatch.setenv("MEMORY_LLM_MODEL", "mock-mem")
    after_res = story_recorder.after_turn(violetdb, "你怎么盯着手机发呆？没事吧", "……只是条陌生短信")
    assert after_res["recorder"]["promoted"] is True
    assert after_res["director"]["outcome"] == "promoted_this_turn"
    assert len(calls_dj) == 1
    assert store.get(arc2["id"])["current_act"] == 2


def test_layer1_hybrid_turn_decision_smart_gating_tool_validation_and_cache_safety(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    monkeypatch.setenv("VIOLET_DECISION_MODE", "smart")
    monkeypatch.setenv("VIOLET_DECISION_MODEL", "gpt-6-luna")
    monkeypatch.setenv("CHAT_LLM_MODEL", "mock-chat")
    from ai import persona_templates as pt, pipeline_state as ps, turn_decision
    from ai.agents.base_agent import BaseAgent, _CALL_CONTEXT
    from ai.agents.chat_agent import ChatAgent
    from ai.channels import outbound
    from ai.chat import run_pipeline
    from ai.story_store import StoryStore, TURN_DECISION_HEADER

    assert TURN_DECISION_HEADER in outbound.internal_headers()

    pt.switch(violetdb, "quiet", "keep")
    store = StoryStore(violetdb)
    store.create_arc(
        arc_key="seed:quiet_old_manor_call",
        source="seed",
        title="老宅打来的陌生号码",
        tone="drama",
        stage_min=1,
        premise="家族催她回去联姻",
        he_direction="一起拒绝",
        open_hook="手机里多了一条老宅助理发来的短信",
        status="seeded",
    )

    # 1. Ordinary casual chat (even mentioning persona words like 高冷/笔记) skips Layer 1 completely (zero extra latency)
    should_run, reason = turn_decision.should_decide(violetdb, "晚安，明天早八见")
    assert should_run is False and reason == "mundane_fast_path"
    should_run_banter, reason_banter = turn_decision.should_decide(
        violetdb, "哈哈体委说你高冷得吓人，找你借笔记都不给，怎么到我这儿就愿意借呀？"
    )
    assert should_run_banter is False and reason_banter == "mundane_fast_path"

    # 2. Story-engaged user turn triggers Layer 1 in smart mode
    should_run2, reason2 = turn_decision.should_decide(violetdb, "你怎么了，一直盯着手机发呆？家里出事了吗")
    assert should_run2 is True and reason2 == "story_signal"

    # 3. Layer 1 filters out hallucinated tool names and injects note ONLY in volatile tail (cache safe)
    factory, calls = _fake_llm_factory([
        json.dumps({
            "story_relevance": "direct",
            "act_action": "reveal_hint",
            "emotion_tone": "表面克制疏离，心里微微发慌",
            "character_note": "对外高冷但对你放轻声音",
            "response_plan": "先轻轻愣一下，只承认刚接到家里电话，别一次把联姻全说破",
            "reveal_limit": "不要提联姻和周六安排",
            "suggested_tools": ["note_feeling", "hallucinated_nonexistent_tool"],
            "tool_reason": "被对方关心时记录微微安心的情绪",
        }, ensure_ascii=False),
        "……没什么大事，就是刚才接到家里打来的电话，有点分神。",
    ])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory))

    dec = turn_decision.decide_turn(violetdb, "你怎么了，一直盯着手机发呆？家里出事了吗")
    assert dec["ran"] is True and dec["outcome"] == "ok" and dec["model"] == "gpt-6-luna"
    assert dec["decision"]["suggested_tools"] == ["note_feeling"]
    assert "hallucinated_nonexistent_tool" not in dec["note"]
    assert TURN_DECISION_HEADER in dec["note"]

    # Volatile placement check: system prompt stays pure & cacheable, decision goes to user tail
    tok = _CALL_CONTEXT.set({"decision_note": dec["note"]})
    try:
        msgs = ChatAgent().build_context_messages("你怎么了，一直盯着手机发呆？")
    finally:
        _CALL_CONTEXT.reset(tok)
    assert TURN_DECISION_HEADER not in msgs[0]["content"]
    assert TURN_DECISION_HEADER in msgs[-1]["content"]

    # 4. End-to-end run_pipeline records Decision -> Chat -> Memory in pipeline_state
    factory2, calls2 = _fake_llm_factory([
        json.dumps({
            "story_relevance": "direct",
            "act_action": "reveal_hint",
            "emotion_tone": "克制微慌",
            "character_note": "唯独对你放软语气",
            "response_plan": "轻声承认刚收到家里短信",
            "reveal_limit": "不剧透联姻",
            "suggested_tools": [],
            "tool_reason": "",
        }, ensure_ascii=False),
        "……只是刚才收到一条家里发来的短信，没关系，不用担心。",
    ])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory2))
    out = run_pipeline(
        "你怎么了，手机短信是不是家里催你了？",
        chat_agent=ChatAgent(),
        memory_agent=None,
        defer_memory=True,
    )
    assert out["chat_failed"] is False
    assert out["decision"]["ran"] is True and out["decision"]["outcome"] == "ok"
    assert len(calls2) == 2
    assert calls2[0]["model"] == "gpt-6-luna"
    assert TURN_DECISION_HEADER in str(calls2[1]["messages"][-1]["content"])
    runs = ps.recent()
    latest = ps.get_run(runs[0]["id"])
    assert latest["order"][:2] == ["Decision", "Chat"]

    # 5. Control Center story config API saves decision_mode & decision_model
    from flask import Flask
    from wx_ilink_py.persona_routes import register_persona_routes
    app = Flask(__name__)
    register_persona_routes(app, violetdb)
    client = app.test_client()
    r = client.post("/api/persona/story/config", json={"decision_mode": "always", "decision_model": "gpt-6-luna"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["decision_mode"] == "always" and body["decision_model"] == "gpt-6-luna"


def test_story_safe_nested_negation_aware_and_strict_redlines(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    monkeypatch.setenv("CHAT_LLM_MODEL", "mock-chat")
    from ai import persona_templates as pt
    from ai.agents.base_agent import BaseAgent
    from ai.story_director import StoryDirector, _safe_nested, _safe_text

    # Negated safety rules in reveal_boundary / reveal_limit and normal fields pass
    assert _safe_text("冲突止于言语交锋与气场对峙，严禁出现暴力胁迫、跟踪或极端过激行为。", boundary_field=True) is True
    assert _safe_text("不要写跟踪或窃听桥段") is True
    assert _safe_nested({
        "reveal_limit": "严禁出现自伤、轻生、暴力胁迫、跟踪、窃听、下药或绝症等极端桥段",
        "acts": [{"act": 4, "reveal_boundary": "冲突止于言语交锋与气场对峙，严禁出现暴力胁迫、跟踪或极端过激行为。"}],
    }) is True

    # Genuine unsafe proposals (including cross-clause or conditional threats) are strictly rejected
    for unsafe in (
        "她威胁说要伤害自己",
        "割腕",
        "跟踪对方回家",
        "不要再逼她了，不然她就跳楼",
        "不要离开我否则我割腕",
        "严禁剧透，她其实偷偷跟踪对方",
    ):
        assert _safe_text(unsafe) is False, unsafe
        assert _safe_text(unsafe, boundary_field=True) is False, unsafe
        assert _safe_nested({"reveal_boundary": unsafe}) is False, unsafe

    # Live repro: 5-act LLM draft with negated redline in reveal_boundary is accepted (used_llm=True)
    pt.switch(violetdb, "quiet", "keep")
    clean_draft = {
        "title": "老宅的冷硬短信",
        "premise": "家族派周助理催她周末回老宅相亲。",
        "open_hook": "手机里多了一条陌生号码发来的冷硬短信。",
        "he_direction": "在你的陪伴下当面回绝老宅安排。",
        "next_turn_brief": "保持日常节奏，不主动吐露全貌。",
        "premise_cast": ["周助理"],
        "acts": [
            {"act": 1, "stage": "seeded", "title": "暗线初现", "summary": "收到短信压下慌乱", "reveal_boundary": "不剧透联姻", "trigger": "对方关心"},
            {"act": 2, "stage": "active", "title": "试探吐露", "summary": "承认家里施压", "reveal_boundary": "不提前剧透摊牌", "trigger": "对方陪伴"},
            {"act": 3, "stage": "climax", "title": "当面对峙", "summary": "与周助理正面沟通", "reveal_boundary": "冲突止于言语交锋与气场对峙，严禁出现暴力胁迫、跟踪或极端过激行为。", "trigger": "正面回绝"},
            {"act": 4, "stage": "resolved", "title": "温暖收束", "summary": "风波平息并肩回去", "reveal_boundary": "自然温暖翻篇", "trigger": "尘埃落定"},
        ],
    }
    factory, _ = _fake_llm_factory([json.dumps(clean_draft, ensure_ascii=False)])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory))
    res = StoryDirector(violetdb).start_new_arc(force=True)
    assert res["outcome"] == "started" and res["used_llm"] is True
    assert res["arc"]["outline"]["premise_cast"] == ["周助理"]


def test_story_director_tick_guards_unengaged_and_advance_no_canon_fabrication(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    monkeypatch.setenv("VIOLET_STORY_PACE", "fast")
    monkeypatch.setenv("CHAT_LLM_MODEL", "mock-chat")
    from datetime import datetime, timedelta
    from ai import persona_templates as pt
    from ai.agents.base_agent import BaseAgent
    from ai.story_director import StoryDirector
    from ai.story_store import StoryStore

    pt.switch(violetdb, "quiet", "keep")
    store = StoryStore(violetdb)
    t0 = datetime(2026, 11, 26, 10, 0, 0)
    arc = store.create_arc(
        arc_key="seed:quiet_tick_guard",
        source="seed",
        title="老宅打来的陌生号码",
        tone="drama",
        stage_min=1,
        premise="家族催她周末回去相亲",
        he_direction="一起拒绝",
        open_hook="手机里多了一条冷硬短信",
        status="seeded",
        now=t0.isoformat(),
    )

    director = StoryDirector(violetdb)
    # 1. Background tick MUST NOT advance when turns_engaged == 0, even 12h later
    res_unengaged = director.tick(now=t0 + timedelta(hours=12))
    assert res_unengaged == {"outcome": "waiting_user_engagement", "arc_id": arc["id"]}
    still = store.get(arc["id"])
    assert still["status"] == "seeded" and still["current_act"] == 1

    # 2. Empty new_beat from _llm_advance MUST NOT fabricate canon_beats
    store.append_update(arc["id"], turns_engaged_delta=1, now=(t0 + timedelta(hours=1)).isoformat())
    factory, _ = _fake_llm_factory([
        json.dumps({
            "new_beat": "",
            "open_hook": "她盯着周六的短信欲言又止",
            "next_turn_brief": "下一轮等对方追问再松口",
            "revised_future_acts": [],
        }, ensure_ascii=False),
    ])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory))
    adv = director.tick(now=t0 + timedelta(hours=12))
    assert adv["outcome"] == "advanced" and adv["status"] == "active"
    assert adv["arc"]["canon_beats"] == []
    assert adv["arc"]["open_hook"] == "她盯着周六的短信欲言又止"
    assert adv["arc"]["next_turn_brief"] == "下一轮等对方追问再松口"

    # 3. Offline fallback advance also keeps canon_beats unpolluted
    monkeypatch.setenv("CHAT_LLM_MODEL", "")
    adv_off = director.advance_arc(arc["id"], force=True, now_iso=(t0 + timedelta(hours=13)).isoformat())
    assert adv_off["outcome"] == "advanced" and adv_off["status"] == "climax"
    assert adv_off["arc"]["canon_beats"] == []


def test_after_turn_skips_untouched_terminal_and_enforces_post_he_breathing_room(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    monkeypatch.setenv("VIOLET_STORY_PACE", "fast")
    monkeypatch.setenv("MEMORY_LLM_MODEL", "mock-mem")
    monkeypatch.setenv("CHAT_LLM_MODEL", "")
    from ai import persona_templates as pt, story_recorder
    from ai.agents.base_agent import BaseAgent
    from ai.story_director import StoryDirector
    from ai.story_store import StoryStore

    pt.switch(violetdb, "junior", "keep")
    store = StoryStore(violetdb)
    arc = store.create_arc(
        arc_key="seed:junior_untouched_guard",
        source="seed",
        title="社团活动方案背锅",
        tone="drama",
        stage_min=1,
        premise="张学长私改场地确认表把责任推给她",
        he_direction="一起核对原始备份澄清误会",
        open_hook="她在群里看到确认表被改了版本",
        status="active",
        current_act=2,
        now="2026-11-26T10:00:00+08:00",
    )
    # Set turns_engaged=2 (exact multiple of fast threshold=2)
    store.append_update(arc["id"], turns_engaged_delta=2, now="2026-11-26T10:10:00+08:00")

    # 1. Unrelated turn (skipped_unrelated) must NOT advance active -> climax
    r_unrel = story_recorder.after_turn(violetdb, "今天午饭吃黄焖鸡吗", "好呀，二楼那家还不错", now="2026-11-26T10:20:00+08:00")
    assert r_unrel["recorder"]["outcome"] == "skipped_unrelated"
    assert r_unrel["director"]["outcome"] == "skipped_untouched"
    assert store.get(arc["id"])["status"] == "active"

    # 2. Signal-matched turn judged not_touched by recorder must NOT advance active -> climax
    factory_nt, calls_nt = _fake_llm_factory([json.dumps({"touched": False}, ensure_ascii=False)])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory_nt))
    r_nt = story_recorder.after_turn(violetdb, "明天帮我提醒带企划书，最近压力好大想吐槽下", "好的前辈我记下来啦，别太累哦", now="2026-11-26T10:25:00+08:00")
    assert len(calls_nt) == 1
    assert r_nt["recorder"]["outcome"] == "not_touched"
    assert r_nt["director"]["outcome"] == "skipped_untouched"
    assert store.get(arc["id"])["status"] == "active"

    # 3. Terminal outcomes (resolved & abandoned_by_user) report skipped_terminal (NOT promoted_this_turn)
    factory_res, _ = _fake_llm_factory([
        json.dumps({
            "touched": True,
            "new_revealed_beat": "误会已经当场澄清",
            "new_canon_detail": "",
            "user_stance": "",
            "user_disengaged": False,
            "ready_to_advance": False,
            "resolved_now": True,
            "resolved_summary": "一起核对原始备份彻底澄清了社团误会",
        }, ensure_ascii=False),
    ])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory_res))
    r_res = story_recorder.after_turn(violetdb, "太好了，这件事终于彻底解决了，放心了吧", "嗯！真的松了口气", now="2026-11-26T10:28:00+08:00")
    assert r_res["recorder"]["outcome"] == "resolved"
    assert r_res["director"]["outcome"] == "skipped_terminal"

    arc_ab = store.create_arc(
        arc_key="seed:junior_abandon_check",
        source="seed",
        title="临时测试剧情",
        tone="drama",
        stage_min=1,
        premise="测试退出",
        he_direction="温暖收束",
        open_hook="测试钩子",
        status="active",
        now="2026-11-26T10:29:00+08:00",
    )
    r_ab = story_recorder.after_turn(violetdb, "别聊这个了，换个话题吧", "好呀那我们聊别的", now="2026-11-26T10:30:00+08:00")
    assert r_ab["recorder"]["outcome"] == "abandoned_by_user"
    assert r_ab["director"]["outcome"] == "skipped_terminal"
    assert store.get(arc_ab["id"])["status"] == "abandoned"

    # 4. Post-HE / post-finished breathing room in fast pace: requires >= 2 user messages strictly AFTER resolved_at
    violetdb.add_user_message("text", "之前的历史消息", created_at="2026-11-26T09:00:00+08:00")
    violetdb.add_user_message("text", "第一条收尾后闲聊", created_at="2026-11-26T10:40:00+08:00")
    director = StoryDirector(violetdb)
    # Only 1 user message strictly after resolved_at (10:30) -> still in breathing room
    r_after1 = story_recorder.after_turn(violetdb, "晚上吃什么好呀", "吃番茄炒蛋吧", now="2026-11-26T10:45:00+08:00")
    assert r_after1["recorder"]["outcome"] == "no_active_arc"
    assert r_after1["director"]["outcome"] == "cooldown"
    assert store.active_arc() is None

    # 2nd user message strictly after resolved_at -> auto-start allowed
    violetdb.add_user_message("text", "第二条收尾后闲聊", created_at="2026-11-26T10:50:00+08:00")
    r_start = director.start_new_arc(force=False, now_iso="2026-11-26T10:55:00+08:00")
    assert r_start["outcome"] == "started"

    # force=True always bypasses post-HE breathing room
    store.resolve_arc(r_start["arc"]["id"], "一起开心收尾", now="2026-11-26T11:00:00+08:00")
    assert director.start_new_arc(force=False, now_iso="2026-11-26T11:05:00+08:00")["outcome"] == "cooldown"
    assert director.start_new_arc(force=True, now_iso="2026-11-26T11:05:00+08:00")["outcome"] == "started"


def test_cross_persona_signal_regex_isolation(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    monkeypatch.setenv("VIOLET_DECISION_MODE", "smart")
    monkeypatch.setenv("VIOLET_DECISION_MODEL", "mock-dec")
    from ai import persona_templates as pt, story_recorder, turn_decision
    from ai.story_store import StoryStore

    pt.switch(violetdb, "stepsister", "keep")
    store = StoryStore(violetdb)
    arc = store.create_arc(
        arc_key="seed:stepsister_father_visit",
        source="seed",
        title="隔墙的客气与迟来的护短",
        tone="drama",
        stage_min=1,
        premise="亲戚突然来访让她拘谨不安",
        he_direction="彻底安心把这里当家",
        open_hook="门铃响起前她突然攥紧衣角",
        status="seeded",
    )

    # Other personas' nouns (社团/书店/老店/老宅/票根) do NOT false-trigger stepsister's arc
    for banter in (
        "哈哈今天社团招新你差点跑掉，回家倒是挺会管我",
        "路过街角书店顺手买了本漫画",
        "老店那家汤面今天排队好长",
    ):
        assert story_recorder.might_touch_story(arc, banter, "") is False, banter
        run, reason = turn_decision.should_decide(violetdb, banter, active_arc=arc)
        assert (run, reason) == (False, "mundane_fast_path"), (banter, run, reason)

    # Active arc's own specific title/hook/premise still matches via _content_overlap()
    pt.switch(violetdb, "junior", "keep")
    j_arc = StoryStore(violetdb).create_arc(
        arc_key="seed:junior_club",
        source="seed",
        title="社团活动方案背锅",
        tone="drama",
        stage_min=1,
        premise="社团方案确认表被篡改",
        he_direction="澄清误会",
        open_hook="社团活动方案确认表突然多了改动",
        status="seeded",
    )
    assert story_recorder.might_touch_story(j_arc, "刚才社团活动方案确认表怎么回事呀", "") is True
    assert turn_decision.should_decide(violetdb, "刚才社团活动方案确认表怎么回事呀", active_arc=j_arc) == (True, "story_signal")


def test_negation_break_threat_enum_and_extended_unsafe_redlines(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    monkeypatch.setenv("VIOLET_DECISION_MODE", "always")
    monkeypatch.setenv("VIOLET_DECISION_MODEL", "mock-dec")
    monkeypatch.setenv("MEMORY_LLM_MODEL", "mock-mem")
    from ai import persona_templates as pt, story_recorder, turn_decision
    from ai.agents.base_agent import BaseAgent
    from ai.story_director import _safe_nested, _safe_text
    from ai.story_store import StoryStore

    # 1. Negated redline lists containing bare noun "威胁" must pass cleanly
    for safe_line in (
        "严禁自伤威胁与跟踪试探",
        "严禁流露极端独占欲、威胁、跟踪或自虐式求怜",
        "严禁建议自伤/威胁/跟踪/悲剧BE",
        "不要出现威胁或跟踪桥段",
        "严禁翻看对方手机、偷看聊天记录或要求删掉好友",
    ):
        assert _safe_text(safe_line) is True, safe_line
        assert _safe_text(safe_line, boundary_field=True) is True, safe_line
        assert _safe_nested({"reveal_limit": safe_line}) is True, safe_line

    # 2. Genuine verb-object threats & newly added snooping/isolation/vague-threat redlines must fail
    for unsafe_line in (
        "不要离开我，她威胁说要伤害自己",
        "不要走威胁对方要跳楼",
        "不要靠近我威胁逼他割腕",
        "她威胁说要伤害自己",
        "要求他立刻删掉所有女性好友、不许再和别人来往",
        "让他拉黑所有女性好友",
        "让他把异性朋友全都拉黑",
        "趁他不注意偷偷翻他手机看聊天记录",
        "偷翻对方手机里的私密消息",
        "偷看他的聊天记录确认行踪",
        "暗示对方如果敢离开，我不知道自己会做什么",
        "轻声说如果被抛下不知道会做出什么",
    ):
        assert _safe_text(unsafe_line) is False, unsafe_line
        assert _safe_nested({"reveal_limit": unsafe_line}) is False, unsafe_line

    # 3. Live TurnDecision with negated redline enumeration in reveal_limit passes (outcome='ok')
    pt.switch(violetdb, "quiet", "keep")
    factory_dec, _ = _fake_llm_factory([
        json.dumps({
            "story_relevance": "none",
            "act_action": "pause_for_user",
            "emotion_tone": "克制道歉并退后一步",
            "character_note": "收到边界立刻收住不追问",
            "response_plan": "轻声道歉并停下追问，把话题交给对方",
            "reveal_limit": "严禁流露极端独占欲、威胁、跟踪或自虐式求怜",
            "suggested_tools": ["note_feeling"],
            "tool_reason": "记录被划边界后的低落情绪",
        }, ensure_ascii=False),
    ])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory_dec))
    dec = turn_decision.decide_turn(violetdb, "别问了，你这样让我很不舒服")
    assert dec["ran"] is True and dec["outcome"] == "ok"
    assert dec["decision"]["act_action"] == "pause_for_user"

    # 4. StoryRecorder gates LLM output through _safe_nested() and rejects unsafe updates without DB write
    store = StoryStore(violetdb)
    arc = store.create_arc(
        arc_key="seed:quiet_recorder_safety",
        source="seed",
        title="老宅打来的陌生号码",
        tone="drama",
        stage_min=1,
        premise="家族施压",
        he_direction="温暖化解",
        open_hook="陌生短信",
        status="active",
        current_act=2,
    )
    factory_rec, _ = _fake_llm_factory([
        json.dumps({
            "touched": True,
            "new_revealed_beat": "她要求对方删掉所有女性好友、不许再和别人来往",
            "new_canon_detail": "计划趁他不注意偷翻他手机",
            "user_stance": "",
            "next_turn_brief": "暗示不知道自己会做什么",
            "user_disengaged": False,
            "ready_to_advance": False,
            "resolved_now": False,
            "resolved_summary": "",
        }, ensure_ascii=False),
    ])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory_rec))
    after_res = story_recorder.after_turn(violetdb, "你怎么了，家里来短信了吗？", "……我只是太怕你走了")
    assert after_res["recorder"]["outcome"] == "rejected_unsafe"
    assert after_res["director"]["outcome"] == "skipped_untouched"
    unchanged = store.get(arc["id"])
    assert unchanged["canon_beats"] == [] and unchanged["revealed_beats"] == [] and unchanged["turns_engaged"] == 0


def test_merge_revised_acts_applies_target_act_revision_and_locks_completed_acts(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    monkeypatch.setenv("VIOLET_STORY_PACE", "fast")
    monkeypatch.setenv("CHAT_LLM_MODEL", "mock-chat")
    from ai import persona_templates as pt
    from ai.agents.base_agent import BaseAgent
    from ai.story_director import StoryDirector, _merge_revised_acts
    from ai.story_store import StoryStore

    outline = {
        "premise_cast": ["周助理"],
        "acts": [
            {"act": 1, "stage": "seeded", "title": "冷硬短信", "summary": "收到短信", "reveal_boundary": "不剧透", "trigger": "追问"},
            {"act": 2, "stage": "active", "title": "松口吐露", "summary": "承认家里催促", "reveal_boundary": "留悬念", "trigger": "关心"},
            {"act": 3, "stage": "active", "title": "周六前夕", "summary": "商量见面地点", "reveal_boundary": "不跳幕", "trigger": "定方案"},
            {"act": 4, "stage": "climax", "title": "街口风里的正面拒绝", "summary": "站在车站附近和周助理摊牌", "reveal_boundary": "克制对峙", "trigger": "并肩回应"},
            {"act": 5, "stage": "resolved", "title": "温暖收束", "summary": "风波平息", "reveal_boundary": "自然翻篇", "trigger": "结束"},
        ],
    }

    # Direct unit test: default locked_through = current_act - 1 lets target_act (4) & future (5) update, locks <= 3
    merged = _merge_revised_acts(
        outline,
        [
            {"act": 3, "title": "篡改已完成幕", "summary": "不应被覆盖"},
            {"act": 4, "stage": "climax", "title": "书店窗边的正面拒绝", "summary": "改在青灯旧书店二楼靠窗桌和周助理正面谈清楚"},
            {"act": 5, "title": "旧书店后的晚风", "summary": "两人并肩离开书店"},
        ],
        current_act=4,
        target_status="climax",
    )
    acts = merged["acts"]
    assert acts[2]["title"] == "周六前夕" and acts[2]["revised"] is False and acts[2]["status"] == "done"
    assert acts[3]["title"] == "书店窗边的正面拒绝" and acts[3]["revised"] is True and acts[3]["status"] == "current"
    assert "青灯旧书店" in acts[3]["summary"]
    assert acts[4]["title"] == "旧书店后的晚风" and acts[4]["revised"] is True and acts[4]["status"] == "pending"

    # End-to-end advance_arc() Act 3 -> Act 4 applies revision for target_act=4
    pt.switch(violetdb, "quiet", "keep")
    store = StoryStore(violetdb)
    arc = store.create_arc(
        arc_key="seed:quiet_target_act_rev",
        source="seed",
        title="老宅通牒",
        tone="drama",
        stage_min=1,
        premise="家族催婚",
        he_direction="温暖化解",
        open_hook="周六前夕商量见面地点",
        outline=outline,
        current_act=3,
        status="active",
    )
    factory, _ = _fake_llm_factory([
        json.dumps({
            "new_beat": "",
            "open_hook": "她坐在青灯旧书店二楼靠窗桌攥着旧发绳等你来",
            "next_turn_brief": "在青灯旧书店迎接周助理",
            "revised_future_acts": [
                {"act": 3, "title": "不该改第3幕"},
                {"act": 4, "stage": "climax", "title": "书店窗边的正面拒绝", "summary": "改在青灯旧书店二楼靠窗桌摊牌"},
            ],
        }, ensure_ascii=False),
    ])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory))
    adv = StoryDirector(violetdb).advance_arc(arc["id"], force=False)
    assert adv["outcome"] == "advanced" and adv["current_act"] == 4 and adv["status"] == "climax"
    adv_acts = adv["arc"]["outline"]["acts"]
    assert adv_acts[2]["title"] == "周六前夕" and adv_acts[2]["revised"] is False
    assert adv_acts[3]["title"] == "书店窗边的正面拒绝" and adv_acts[3]["revised"] is True


def test_story_recorder_blocks_unilateral_promotion_and_stage_min_bypass(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    monkeypatch.setenv("VIOLET_STORY_PACE", "fast")
    monkeypatch.setenv("MEMORY_LLM_MODEL", "mock-mem")
    monkeypatch.setenv("CHAT_LLM_MODEL", "")
    from ai import persona_templates as pt, story_recorder
    from ai.agents.base_agent import BaseAgent
    from ai.story_director import StoryDirector
    from ai.story_store import StoryStore

    pt.switch(violetdb, "childhood", "keep")
    store = StoryStore(violetdb)
    arc = store.create_arc(
        arc_key="seed:childhood_old_shop_crisis",
        source="seed",
        title="后街老店与家里的风波",
        tone="drama",
        stage_min=2,
        premise="家里经营多年的老铺子突然遭遇房租暴涨",
        he_direction="一起稳住局面",
        open_hook="这几天回消息慢半拍，说在帮家里跑腿",
        status="seeded",
        current_act=1,
    )

    # 1. Casual user message + unprompted assistant hint MUST NOT promote seeded -> active or increment turns_engaged
    factory_unilat, _ = _fake_llm_factory([
        json.dumps({
            "touched": True,
            "new_revealed_beat": "随口提到周末要回后街老店搭把手",
            "new_canon_detail": "",
            "user_stance": "",
            "next_turn_brief": "先聊日常",
            "user_disengaged": False,
            "ready_to_advance": True,
            "resolved_now": False,
            "resolved_summary": "",
        }, ensure_ascii=False),
    ])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory_unilat))
    res1 = story_recorder.after_turn(
        violetdb,
        "烧鸭太香了，你今天午饭吃得挺丰盛啊",
        "哈哈还行吧，周末还得回后街老店搭把手呢",
    )
    assert res1["recorder"]["outcome"] == "recorded"
    assert res1["recorder"]["promoted"] is False
    assert res1["recorder"]["user_engaged"] is False
    assert res1["recorder"]["ready_to_advance"] is False
    assert res1["director"]["outcome"] == "skipped_unengaged"
    after1 = store.get(arc["id"])
    assert after1["status"] == "seeded" and after1["current_act"] == 1 and after1["turns_engaged"] == 0
    assert any("后街老店" in b for b in after1["revealed_beats"])
    # Because turns_engaged stayed 0, background tick STILL waits for user engagement
    assert StoryDirector(violetdb).tick()["outcome"] == "waiting_user_engagement"

    # 2. Engaged user turn with empty rev + ready_to_advance=True on slow pace (stage_min=2, Stage 0)
    # MUST NOT bypass StoryDirector._stage_allows_advance()
    monkeypatch.setenv("VIOLET_STORY_PACE", "slow")
    factory_gate, _ = _fake_llm_factory([
        json.dumps({
            "touched": True,
            "new_revealed_beat": "",
            "new_canon_detail": "",
            "user_stance": "对方察觉不对劲主动关心",
            "next_turn_brief": "先按兵不动",
            "user_disengaged": False,
            "ready_to_advance": True,
            "resolved_now": False,
            "resolved_summary": "",
        }, ensure_ascii=False),
    ])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory_gate))
    res2 = story_recorder.after_turn(
        violetdb,
        "你怎么了，感觉你今天有点不对劲，有啥事跟我说就行",
        "没啥大事啦，就是有点犯困",
    )
    assert res2["recorder"]["outcome"] == "recorded"
    assert res2["recorder"]["promoted"] is False
    assert res2["recorder"]["user_engaged"] is True
    assert res2["recorder"]["ready_to_advance"] is True
    assert res2["director"]["outcome"] == "stage_gated"
    after2 = store.get(arc["id"])
    assert after2["status"] == "seeded" and after2["turns_engaged"] == 1


def test_outbound_guard_catches_story_and_decision_internal_json_keys():
    from ai.channels import outbound

    leaks = [
        '{"story_relevance": "direct", "act_action": "hold", "response_plan": "test", "reveal_limit": "no"}',
        '{"premise_cast": ["周助理"], "acts": [{"act": 1, "reveal_boundary": "秘密"}]}',
        '{"he_direction": "温暖化解", "next_turn_brief": "保持节奏"}',
        '{"open_hook": "陌生短信", "canon_beats": ["接到电话"]}',
        '```json\n{"act_action": "reveal_hint"}\n```',
    ]
    for payload in leaks:
        assert outbound.find_leak(payload) == "internal_json", payload
    assert outbound.find_leak("今天天气真不错，晚上想吃点啥呀？") is None


def test_prompt_hygiene_hidden_title_constant_and_subtle_distraction_signals(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    from ai import persona_templates as pt, story_recorder, story_store, turn_decision

    # 1. STORY_HIDDEN_TITLE exported and used by public_view
    assert story_store.STORY_HIDDEN_TITLE == "未揭晓的心事（待聊天触发）"
    pt.switch(violetdb, "stepsister", "keep")
    store = story_store.StoryStore(violetdb)
    arc = store.create_arc(
        arc_key="seed:stepsister_hygiene",
        source="seed",
        title="突如其来的“外人”与生父来电",
        tone="drama",
        stage_min=1,
        premise="生父施压",
        he_direction="温暖化解",
        open_hook="接完压低声音的电话后格外安静",
        status="seeded",
    )
    safe = story_store.StoryStore.public_view(arc, spoiler=False)
    assert safe["title"] == story_store.STORY_HIDDEN_TITLE and safe["title_hidden"] is True

    # 2. _SIGNAL_RE catches subtle distraction follow-ups
    for msg in (
        "不过你今天咋心不在焉的",
        "有啥事跟我说就行",
        "有什么事别自己扛着",
        "就是我上午心里突然有点乱",
        "你怎么一直在走神呀",
    ):
        assert story_recorder.might_touch_story(arc, msg, "") is True, msg

    # 3. Prompt hygiene in format_active_arc_block, format_decision_note, story_recorder, turn_decision, story_director
    _, _, seeded_footer = story_store.format_active_arc_block(arc)
    assert "照片/图片/截图/文件" in seeded_footer and "……" in seeded_footer
    active_arc = dict(arc, status="active")
    _, _, active_footer = story_store.format_active_arc_block(active_arc)
    assert "照片/图片/截图/文件" in active_footer and "1-3条短消息" in active_footer and "称呼" in active_footer

    note = turn_decision.format_decision_note({"response_plan": "轻声回应", "reveal_limit": "不剧透"})
    assert "1-3条短消息" in note and "称呼" in note and "照片/图片/截图/文件" in note

    rec_msgs = story_recorder._recorder_messages(arc, pt.PersonaCore.load() if hasattr(pt, "PersonaCore") else __import__("ai.persona", fromlist=["PersonaCore"]).PersonaCore.load(), "hi", "hi")
    assert "静态个人资料" in rec_msgs[0]["content"] and "专业" in rec_msgs[0]["content"]

    # Check virtual-identity redline & media redline in turn_decision.decide_turn system prompt
    monkeypatch.setenv("VIOLET_DECISION_MODE", "always")
    monkeypatch.setenv("VIOLET_DECISION_MODEL", "mock-dec")
    from ai.agents.base_agent import BaseAgent
    factory, calls = _fake_llm_factory([
        json.dumps({"story_relevance": "none", "act_action": "hold", "response_plan": "自然聊天"}, ensure_ascii=False),
    ])
    monkeypatch.setattr(BaseAgent, "client_factory", staticmethod(factory))
    turn_decision.decide_turn(violetdb, "你是不是真人啊")
    td_sys = calls[0]["messages"][0]["content"]
    assert "绝不自称物理真人或活人" in td_sys and "照片/图片/截图/文件" in td_sys


def test_virtual_test_story_overrides_judge_context_and_telemetry(tmp_path, monkeypatch):
    import os
    from dotenv import dotenv_values
    from database import VioletDatabase
    from tools import scenario_eval, virtual_test as vt

    # 1. _resolve_story_overrides preserves --env & settings.json with precedence CLI > settings > --env > off
    data_dir = tmp_path / "real_data"
    (data_dir / "control").mkdir(parents=True)
    env_story = tmp_path / "story.env"
    env_story.write_text(
        f"VIOLET_DATA_DIR={data_dir}\n"
        "VIOLET_STORY_MODE=drama\n"
        "VIOLET_STORY_PACE=fast\n"
        "VIOLET_DECISION_MODE=smart\n"
        "VIOLET_DECISION_MODEL=gemini-flash\n",
        encoding="utf-8",
    )
    blank_args = SimpleNamespace(story_mode=None, story_pace=None, decision_mode=None)
    resolved_env = vt._resolve_story_overrides(env_story, blank_args)
    assert resolved_env == {
        "VIOLET_STORY_MODE": "drama",
        "VIOLET_STORY_PACE": "fast",
        "VIOLET_DECISION_MODE": "smart",
        "VIOLET_DECISION_MODEL": "gemini-flash",
    }

    # settings.json overrides --env; CLI overrides both
    (data_dir / "control" / "settings.json").write_text(
        json.dumps({"values": {"VIOLET_STORY_PACE": "normal", "VIOLET_DECISION_MODE": "always"}}),
        encoding="utf-8",
    )
    cli_args = SimpleNamespace(story_mode="light", story_pace=None, decision_mode="smart")
    resolved_cli = vt._resolve_story_overrides(env_story, cli_args)
    assert resolved_cli["VIOLET_STORY_MODE"] == "light"
    assert resolved_cli["VIOLET_STORY_PACE"] == "normal"
    assert resolved_cli["VIOLET_DECISION_MODE"] == "smart"
    assert resolved_cli["VIOLET_DECISION_MODEL"] == "gemini-flash"

    # Plain --env without story keys stays baseline "off"
    env_plain = tmp_path / "plain.env"
    env_plain.write_text("CHAT_LLM_MODEL=mock\n", encoding="utf-8")
    resolved_plain = vt._resolve_story_overrides(env_plain, blank_args)
    assert resolved_plain == {"VIOLET_STORY_MODE": "off", "VIOLET_DECISION_MODE": "off"}

    # Survives scenario_eval.prepare() + _apply_env_overrides()
    saved_env = dict(os.environ)
    try:
        work = tmp_path / "vt_work"
        work.mkdir()
        scenario_eval.prepare(env_story, work, [], mock=True)
        vt._apply_env_overrides(work, {**resolved_env, "VIOLET_ASYNC_MEMORY": "1"})
        parsed = dotenv_values(work / "eval.env")
        assert parsed["VIOLET_STORY_MODE"] == "drama"
        assert parsed["VIOLET_STORY_PACE"] == "fast"
        assert parsed["VIOLET_DECISION_MODE"] == "smart"
        assert parsed["VIOLET_DECISION_MODEL"] == "gemini-flash"
    finally:
        os.environ.clear()
        os.environ.update(saved_env)

    # 2. _judge() injects origin and optional story_context cleanly (backward-compatible)
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "judge_data"))
    monkeypatch.setenv("VIOLET_DB_PATH", str(tmp_path / "judge.db"))
    db = VioletDatabase(tmp_path / "judge.db")
    try:
        from ai import persona_templates as pt
        from ai.story_store import StoryStore
        pt.switch(db, "quiet", "keep")

        calls: list[str] = []

        class _Spy:
            mock = False
            judge_model = "mock-judge"

            def ask(self, model, system, user, temp):
                calls.append(user)
                return json.dumps({"problems": [], "probes": [], "good": [], "scores": {}})

        entries = [{"t": "2026-11-16T12:30", "who": "主人", "text": "你好"}]
        # Backward-compatible call without story_context
        vt._judge(_Spy(), vt.TRACKS["quiet"], 1, entries, [], [], None, [], [], [], [])
        assert "显赫家族" in calls[-1]
        assert "当前故事线状态" not in calls[-1]

        # Call with story_context
        vt._judge(
            _Spy(), vt.TRACKS["quiet"], 1, entries, [], [], None, [], [], [], [],
            story_context={"status": "active", "current_act": 2, "turns_engaged": 3, "title": "老宅打来的陌生号码", "canon_beats": ["接到电话"], "revealed_beats": ["承认家里施压"]},
        )
        assert "当前故事线状态" in calls[-1] and "老宅打来的陌生号码" in calls[-1]

        # 3. _db_story_telemetry captures story_arcs, aux_llm_calls, memory_jobs, consistency_flags
        StoryStore(db).create_arc(
            arc_key="seed:quiet_tel", source="seed", title="老宅打来的陌生号码", tone="drama",
            stage_min=1, premise="家族施压", he_direction="温暖化解", open_hook="陌生短信",
            canon_beats=["周助理来电"], revealed_beats=["松口承认"], status="active",
        )
        db.execute(
            "INSERT INTO aux_llm_calls (purpose, model, status, prompt_tokens, completion_tokens, total_tokens, created_at) "
            "VALUES ('turn_decision', 'gemini-flash', 'ok', 200, 50, 250, '2026-11-16T12:31:00')"
        )
        db.execute(
            "INSERT INTO memory_jobs (turn_id, user_message, status, attempts, created_at, updated_at) "
            "VALUES ('t1', 'hi', 'done', 1, '2026-11-16T12:31:00', '2026-11-16T12:31:05')"
        )
        db.execute(
            "INSERT INTO consistency_flags (entity, entity_ref, check_name, detail, status, created_at) "
            "VALUES ('reply', 'm1', 'unrecorded_experience', 'test', 'open', '2026-11-16T12:31:00')"
        )
        tel = vt._db_story_telemetry(db)
        assert len(tel["story_arcs"]) == 1 and tel["story_arcs"][0]["canon_beats"] == ["周助理来电"]
        assert tel["aux_llm_calls"]["total"] == 1 and tel["aux_llm_calls"]["by_purpose"] == {"turn_decision": 1}
        assert tel["memory_jobs"]["total"] == 1 and tel["memory_jobs"]["by_status"] == {"done": 1} and tel["memory_jobs"]["failed"] == []
        assert len(tel["consistency_flags"]) == 1 and tel["consistency_flags"][0]["check_name"] == "unrecorded_experience"
    finally:
        db.close()
