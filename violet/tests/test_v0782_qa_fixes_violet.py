"""Regression tests for Violet v0.7.82 Wave 2 runtime & memory QA fixes:
1. MemoryAgent prompt/rules + active Story Arc exclusion note prevent story fiction polluting user memories.
2. ProactiveScheduler gates unengaged active/climax & Stage-0 seeded story beats, and strips _seal before DB save.
3. LifeScheduler.tick() runs StoryDirector.tick() even when VIOLET_LIFE_ENGINE=0 or paused.
4. ConsistencyGuard _HEDGE includes '到时候' so future promises are not falsely flagged as past claims.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from ai.channels import outbound
from ai.consistency import ConsistencyGuard
from ai.life_scheduler import LifeScheduler
from ai.persona import PersonaCore
from ai.proactive_scheduler import ProactiveScheduler
from ai.story_store import StoryStore


# ---------------------------------------------------------------------------
# 1. MemoryAgent Story Arc exclusion
# ---------------------------------------------------------------------------

def test_memory_agent_prompt_and_active_story_exclusion(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    from ai import persona_templates as pt
    from ai.agents.memory_agent import MemoryAgent, PROFILE_RULES

    prompt_path = Path(__file__).resolve().parent.parent / "persona" / "prompts" / "memory.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    assert "虚拟故事线" in prompt_text and "故事场记" in prompt_text
    assert "虚拟故事线" in PROFILE_RULES and "user_situations" in PROFILE_RULES and "event_memories" in PROFILE_RULES

    pt.switch(violetdb, "quiet", "keep")
    agent = MemoryAgent()

    # No active arc -> no exclusion line
    msgs_none = agent.build_context_messages("用户说：晚上想吃番茄炒蛋\nViolet 实际回复：好呀")
    assert "【排除虚构故事线】" not in msgs_none[0]["content"]

    # Active arc with premise_cast -> 1-line exclusion naming title + cast
    store = StoryStore(violetdb)
    store.create_arc(
        arc_key="seed:quiet_old_manor_call",
        source="seed",
        title="老宅打来的陌生号码",
        tone="drama",
        stage_min=1,
        premise="家族助理催她转学联姻",
        he_direction="一起拒绝",
        open_hook="手机响起陌生来电",
        status="active",
        outline={
            "premise_cast": ["周助理", "青灯旧书店", "转学通知函"],
            "acts": [{"act": 1, "stage": "seeded", "title": "陌生来电", "summary": "接到电话"}],
        },
    )
    msgs_active = agent.build_context_messages("用户说：我陪你去青灯旧书店把话说清楚\nViolet 实际回复：嗯")
    sys_content = msgs_active[0]["content"]
    assert "【排除虚构故事线】" in sys_content
    assert "老宅打来的陌生号码" in sys_content
    assert "周助理" in sys_content and "青灯旧书店" in sys_content
    assert "event_memories" in sys_content and "user_situations" in sys_content

    # Story mode off -> omitted even when a row exists
    monkeypatch.setenv("VIOLET_STORY_MODE", "off")
    msgs_off = agent.build_context_messages("用户说：晚安\nViolet 实际回复：晚安")
    assert "【排除虚构故事线】" not in msgs_off[0]["content"]

    # Normal replies are not affected by outbound internal header scan
    outbound.reset_for_tests()
    assert outbound.find_leak("晚上想吃西红柿炒蛋呀") is None


# ---------------------------------------------------------------------------
# 2. ProactiveScheduler story_beat gating + _seal stripping
# ---------------------------------------------------------------------------

def test_proactive_story_beat_gates_unengaged_active_and_stage0_seeded(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    from ai import persona_templates as pt

    pt.switch(violetdb, "quiet", "keep")
    store = StoryStore(violetdb)
    now = datetime.fromisoformat(violetdb.now()[:19])
    old_ts = (now - timedelta(hours=18)).isoformat()
    sched = ProactiveScheduler(chat_agent=None, send_messages=lambda *_a: None)

    # (a) Stage-0 (warmth_level == 0) seeded arc (>12h old) MUST be skipped
    arc = store.create_arc(
        arc_key="seed:quiet_old_manor_call",
        source="seed",
        title="老宅打来的陌生号码",
        tone="drama",
        stage_min=0,
        premise="家里来电施压",
        he_direction="一起拒绝",
        open_hook="刚挂完老宅打来的电话，手指发凉",
        status="seeded",
        now=old_ts,
    )
    assert sched._shareable_story_beat(now) is None
    ctx0 = sched._build_event_context(now)
    assert ctx0["warmth_level"] == 0
    assert ctx0["summary_topic"] != "story_beat"
    assert "offered_story_beat" not in ctx0["conversation_state"]

    # (b) Stage-1+ (warmth_level >= 1) seeded arc (>12h old) IS allowed with light seeded tone
    pt.set_stage(None, 1)
    beat_seeded = sched._shareable_story_beat(now)
    assert beat_seeded is not None and beat_seeded["status"] == "seeded"

    # (c) Unengaged active/climax arc (turns_engaged == 0) MUST be skipped even at Stage 1+
    store.append_update(arc["id"], status="active")
    assert sched._shareable_story_beat(now) is None
    store.append_update(arc["id"], status="climax")
    assert sched._shareable_story_beat(now) is None

    # (d) Once user engages (turns_engaged >= 1), active/climax beat IS offered
    store.append_update(arc["id"], status="active", turns_engaged_delta=1)
    beat_engaged = sched._shareable_story_beat(now)
    assert beat_engaged is not None and beat_engaged["id"] == arc["id"] and beat_engaged["turns_engaged"] == 1
    ctx_engaged = sched._build_event_context(now)
    assert ctx_engaged["summary_topic"] == "story_beat"
    assert ctx_engaged["conversation_state"]["offered_story_beat"]["id"] == arc["id"]


def test_proactive_handle_pipeline_result_strips_seal_sentinel(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    from ai import persona_templates as pt

    pt.switch(violetdb, "quiet", "keep")
    store = StoryStore(violetdb)
    arc = store.create_arc(
        arc_key="seed:quiet_old_manor_call",
        source="seed",
        title="老宅打来的陌生号码",
        tone="drama",
        stage_min=0,
        premise="家里来电施压",
        he_direction="一起拒绝",
        open_hook="刚挂完家里的电话，心里有点烦",
        status="active",
    )
    store.append_update(arc["id"], turns_engaged_delta=1)

    # Case 1: send_messages returns None -> sealed payload messages must be cleaned before DB JSON dump
    seen_sealed: list[list[dict]] = []

    def fake_send_none(messages, _event_id):
        assert all(outbound.is_sealed(m) for m in messages)
        seen_sealed.append(messages)
        return None

    sched_none = ProactiveScheduler(chat_agent=None, send_messages=fake_send_none)
    now = datetime.fromisoformat(violetdb.now()[:19])
    ctx = sched_none._build_event_context(now)
    ev1 = violetdb.add_proactive_event("scheduled_proactive_check", status="planned", action_type="chat")
    sched_none._handle_pipeline_result(
        {"event_id": ev1, "should_send": True, "chat": {"content": "刚挂了家里的电话，心里有点烦"}},
        event_context=ctx,
    )
    assert len(seen_sealed) == 1
    row1 = violetdb.fetch_one("SELECT status, sent_messages FROM proactive_events WHERE id = ?", (ev1,))
    assert row1["status"] == "sent"
    parsed1 = json.loads(row1["sent_messages"])
    assert len(parsed1) == 1 and parsed1[0]["content"] == "刚挂了家里的电话，心里有点烦"
    assert not any(k.startswith("_") for k in parsed1[0])
    # Story beat step was marked shared cleanly
    assert sched_none._shareable_story_beat(now) is None

    # Case 2: send_messages returns raw sealed dicts inside report["sent_messages"]
    def fake_send_raw_sealed(messages, _event_id):
        return {"sent_messages": [{**m, "_extra_sentinel": object()} for m in messages]}

    sched_raw = ProactiveScheduler(chat_agent=None, send_messages=fake_send_raw_sealed)
    ev2 = violetdb.add_proactive_event("scheduled_proactive_check", status="planned", action_type="chat")
    sched_raw._handle_pipeline_result(
        {"event_id": ev2, "should_send": True, "chat": {"content": "突然想听你随便说点什么"}},
        event_context={},
    )
    row2 = violetdb.fetch_one("SELECT status, sent_messages FROM proactive_events WHERE id = ?", (ev2,))
    assert row2["status"] == "sent"
    parsed2 = json.loads(row2["sent_messages"])
    assert parsed2[0]["content"] == "突然想听你随便说点什么"
    assert not any(k.startswith("_") for k in parsed2[0])


# ---------------------------------------------------------------------------
# 3. LifeScheduler.tick() runs StoryDirector.tick() when life engine is off/paused
# ---------------------------------------------------------------------------

def test_life_scheduler_tick_runs_story_director_when_life_off_or_paused(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_LIFE_ENGINE", "0")
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")
    monkeypatch.setenv("VIOLET_STORY_PACE", "fast")
    monkeypatch.setenv("CHAT_LLM_MODEL", "")
    from ai import persona_templates as pt

    pt.switch(violetdb, "quiet", "keep")
    violetdb.add_user_message("text", "你好呀")
    violetdb.add_user_message("text", "今天天气不错")

    now = datetime.fromisoformat(violetdb.now()[:19])
    sched = LifeScheduler(violetdb, clock=lambda: now)
    out_off = sched.tick()
    assert out_off["outcome"] == "off"
    assert out_off.get("story") == "started"
    arc = StoryStore(violetdb).active_arc()
    assert arc is not None

    # Clear arc, turn life engine on but paused -> tick() still runs StoryDirector and preserves outcome == "paused"
    violetdb.execute("DELETE FROM story_arcs")
    monkeypatch.setenv("VIOLET_LIFE_ENGINE", "1")
    sched.control.set_paused(True, actor="test", reason="qa")
    out_paused = sched.tick()
    assert out_paused["outcome"] == "paused"
    assert out_paused.get("story") == "started"
    assert StoryStore(violetdb).active_arc() is not None

    # When both life engine and story engine are off -> clean {"outcome": "off"} with no "story" key
    violetdb.execute("DELETE FROM story_arcs")
    monkeypatch.setenv("VIOLET_LIFE_ENGINE", "0")
    monkeypatch.setenv("VIOLET_STORY_MODE", "off")
    out_both_off = sched.tick()
    assert out_both_off == {"outcome": "off"}
    assert StoryStore(violetdb).active_arc() is None


# ---------------------------------------------------------------------------
# 4. ConsistencyGuard _HEDGE includes '到时候'
# ---------------------------------------------------------------------------

def test_consistency_guard_hedges_future_daoshihou_without_weakening_past_claims(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    from ai import persona_templates as pt

    pt.switch(violetdb, "stepsister", "keep")
    violetdb.save_event_memory(
        content="用户晚上想吃西红柿炒蛋，沈念初答应晚上来做",
        category="event",
        kind="event",
        progress="planned",
        importance=3.0,
        emotional_weight=2.0,
    )
    guard = ConsistencyGuard(violetdb, PersonaCore.load())

    # Future promise with '到时候' must NOT trigger user_experience_as_own or unrecorded_experience
    assert guard.check_reply("对了，晚上除了西红柿炒蛋，你还想再加个什么菜，我到时候顺手一起做了") == []
    assert guard.check_reply("我到时候给你带一份回去") == []

    # Genuine past-tense claim without hedge is STILL caught
    past_checks = [v.check for v in guard.check_reply("我刚才已经把西红柿炒蛋做了")]
    assert "user_experience_as_own" in past_checks or "unrecorded_experience" in past_checks
