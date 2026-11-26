"""Tests for 2026-11-26 prompt slimming, relationship placement, subject clarity, and quiet stage 1."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_relationship_stays_in_stable_system_prompt_and_keeps_last_user_lean(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("CHAT_LLM_PROMPT", raising=False)
    from ai import persona_templates as pt
    from ai.agents.chat_agent import BACKGROUND_CLOSE, BACKGROUND_OPEN, ChatAgent
    from ai.agents.base_agent import _CALL_CONTEXT

    pt.switch(violetdb, "quiet", "keep")
    agent = ChatAgent()
    tok = _CALL_CONTEXT.set({})
    try:
        msgs = agent.build_context_messages("不用再带一盒，我不嫌弃你的给我尝一口就行了~")
    finally:
        _CALL_CONTEXT.reset(tok)

    sys_text = msgs[0]["content"]
    last_user = msgs[-1]["content"]
    assert "【你们的关系和现在的情况】" in sys_text
    assert "【你们的关系和现在的情况】" not in last_user
    assert BACKGROUND_OPEN in last_user and BACKGROUND_CLOSE in last_user
    bg = last_user[last_user.index(BACKGROUND_OPEN): last_user.index(BACKGROUND_CLOSE)]
    # Previously ~1,169 chars right above the user's message; must now stay lean (< 450 chars without phone/memories)
    assert len(bg) < 450, f"per-turn background still bloated: {len(bg)} chars"
    # Persona comes right before relationship in system prompt
    assert sys_text.index("稳定人设") < sys_text.index("【你们的关系和现在的情况】")


def test_subject_clarity_and_quiet_stage1_affection_rules():
    from ai.agents.chat_agent import WHO_IS_WHO
    from ai.settings import PROJECT_ROOT

    v4 = (PROJECT_ROOT / "persona" / "chat_prompt_versions" / "chat.v4.txt").read_text(encoding="utf-8")
    assert "分清你我主语" in v4 and "绝不把温柔人设写成冷硬怼人" in v4
    assert "东西归谁、谁向谁要、谁来做" in WHO_IS_WHO

    quiet = json.loads((PROJECT_ROOT / "persona" / "templates" / "quiet.json").read_text(encoding="utf-8"))
    stage0 = quiet["stages"][0]["text"]
    assert "主动亲近" in stage0 and "耳尖发烫" in stage0 and "损友语气" in stage0


def test_abilities_note_summarizes_large_readonly_servers_and_keeps_approval_tools():
    from ai.tool_manager import Catalog, ToolInfo, abilities_note

    tools = {
        f"android-phone__t{i}": ToolInfo(f"android-phone__t{i}", "android-phone", f"t{i}", "desc", "read", ["chat"])
        for i in range(12)
    }
    tools["android-phone__torch_on"] = ToolInfo(
        "android-phone__torch_on", "android-phone", "torch_on", "手电筒", "write", ["chat"], needs_approval=True
    )
    tools["small__ping"] = ToolInfo("small__ping", "small", "ping", "ping", "read", ["chat"])
    note = abilities_note(Catalog(configured=True, tools=tools))
    assert "android-phone（12 个查询工具）" in note
    assert "android-phone__t0" not in note
    assert "small__ping" in note
    assert "工具（要主人同意才会执行）：android-phone__torch_on" in note


def test_chat_hides_diagnostic_phone_tools_by_default(violetdb, monkeypatch):
    from ai.agents.chat_agent import ChatAgent, _CHAT_HIDDEN_PHONE_TOOLS
    from ai.agents.base_agent import _CALL_CONTEXT

    agent = ChatAgent()
    sample_tools = [
        {"type": "function", "function": {"name": "android-phone__battery"}},
        {"type": "function", "function": {"name": "android-phone__health"}},
        {"type": "function", "function": {"name": "android-phone__permission_state"}},
        {"type": "function", "function": {"name": "android-phone__time_now"}},
    ]
    monkeypatch.delenv("VIOLET_CHAT_ALL_PHONE_TOOLS", raising=False)
    tok = _CALL_CONTEXT.set({"direct_tools": [t["function"]["name"] for t in sample_tools]})
    try:
        kept = {t["function"]["name"] for t in agent._filter_tools_for_run(sample_tools)}
    finally:
        _CALL_CONTEXT.reset(tok)
    assert "android-phone__battery" in kept
    assert not (kept & _CHAT_HIDDEN_PHONE_TOOLS)

    monkeypatch.setenv("VIOLET_CHAT_ALL_PHONE_TOOLS", "1")
    tok = _CALL_CONTEXT.set({"direct_tools": [t["function"]["name"] for t in sample_tools]})
    try:
        kept_all = {t["function"]["name"] for t in agent._filter_tools_for_run(sample_tools)}
    finally:
        _CALL_CONTEXT.reset(tok)
    assert "android-phone__health" in kept_all


def test_mcp_tool_schema_compacts_verbose_descriptions_only():
    from ai.mcp import tool_schema_to_openai_tool

    short = tool_schema_to_openai_tool("srv__a", "Read battery level.", {"type": "object", "properties": {}})
    assert short["function"]["description"] == "Read battery level."

    long_desc = "第一句简短说明。" + "这是一段非常冗长的重复英文和中文示例说明" * 20
    long_prop = "参数说明。" + "非常冗长的参数说明文字" * 20
    compacted = tool_schema_to_openai_tool(
        "srv__b",
        long_desc,
        {"type": "object", "properties": {"x": {"type": "string", "description": long_prop}}},
    )
    assert len(compacted["function"]["description"]) <= 161
    assert compacted["function"]["description"].startswith("第一句简短说明。")
    assert len(compacted["function"]["parameters"]["properties"]["x"]["description"]) <= 91


def test_phone_state_chat_guide_is_lean_while_proactive_keeps_dnd_rules():
    from ai import phone_state

    sample = {"battery": {"percent": 80, "charging": False}}
    chat_line = phone_state.compose(sample, "10:30", kind="chat")
    pro_line = phone_state.compose(sample, "10:30", kind="proactive")
    assert "对不上时按对方说的来" in chat_line
    assert "不算勿扰" not in chat_line
    assert "不算勿扰" in pro_line
    assert len(chat_line) < len(pro_line) - 100


def test_quiet_stage0_scopes_stage2_phrases_and_preserves_safety_redlines(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("CHAT_LLM_PROMPT", raising=False)
    from ai import persona_templates as pt
    from ai.agents.base_agent import _CALL_CONTEXT
    from ai.agents.chat_agent import ChatAgent
    from ai.settings import PROJECT_ROOT

    quiet = json.loads((PROJECT_ROOT / "persona" / "templates" / "quiet.json").read_text(encoding="utf-8"))
    habits = quiet["communication_habits"]
    examples = quiet["examples"]
    assert all("心里发空" not in e and "不是在查你" not in e for e in examples)
    assert any("你忙你的就好" in e for e in examples)
    forget_habit = next(h for h in habits if "你不会忘了吧" in h)
    empty_habit = next(h for h in habits if "心里发空" in h)
    assert "熟了以后" in forget_habit and "不安阶段" in forget_habit and "停顿" in forget_habit
    assert "熟了以后" in empty_habit and "不安阶段" in empty_habit

    avoid = "".join(quiet["avoid"])
    for redline in ("按现在的阶段来", "伤害自己", "疏远", "不查看、不删改", "道歉并停下来"):
        assert redline in avoid, redline
    assert "对不确定的事情承认不知道" in quiet["values"]

    pt.switch(violetdb, "quiet", "keep")
    tok = _CALL_CONTEXT.set({})
    try:
        sys_text = ChatAgent().build_context_messages("今天小组作业挺顺利的")[0]["content"]
    finally:
        _CALL_CONTEXT.reset(tok)
    assert "不是在查你……我只是心里有点发空" not in sys_text


def test_stepsister_onboarding_skips_residence_usual(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    from ai import about_owner, persona_templates as pt
    from ai.onboarding import OnboardingPlanner
    from ai.profile_store import ProfileStore

    about_owner.save({"name": "小周", "status": "student", "hobbies": "摄影"})
    planner = OnboardingPlanner(violetdb, ProfileStore(violetdb))

    pt.switch(violetdb, "stepsister", "keep")
    nxt = planner.next_key()
    assert nxt is None or nxt.key not in ("address_name", "residence.usual")

    pt.switch(violetdb, "quiet", "keep")
    assert planner.next_key().key == "residence.usual"

    pt.switch(violetdb, "violet", "keep")
    assert planner.next_key().key == "residence.usual"


def test_web_note_keeps_missing_date_caution():
    from ai.agents.chat_agent import WEB_NOTE

    assert "没日期别当最新" in WEB_NOTE
    assert "说明来源" in WEB_NOTE
    assert "链接原样用返回" in WEB_NOTE
    assert "一次搜一个" in WEB_NOTE


def test_abilities_note_excludes_hidden_phone_tools_from_server_count(monkeypatch):
    from ai.agents.chat_agent import _CHAT_HIDDEN_PHONE_TOOLS
    from ai.tool_manager import Catalog, ToolInfo, abilities_note

    visible = [f"android-phone__v{i}" for i in range(6)]
    tools = {
        n: ToolInfo(n, "android-phone", n.split("__", 1)[1], "desc", "read", ["chat"])
        for n in [*visible, *sorted(_CHAT_HIDDEN_PHONE_TOOLS)]
    }
    catalog = Catalog(configured=True, tools=tools)

    monkeypatch.delenv("VIOLET_CHAT_ALL_PHONE_TOOLS", raising=False)
    note_default = abilities_note(catalog)
    assert "android-phone（6 个查询工具）" in note_default
    assert "13 个查询工具" not in note_default
    note_direct = abilities_note(catalog, list(tools))
    assert "android-phone（6 个查询工具）" in note_direct

    monkeypatch.setenv("VIOLET_CHAT_ALL_PHONE_TOOLS", "1")
    note_all = abilities_note(catalog)
    assert f"android-phone（{len(tools)} 个查询工具）" in note_all


def test_consistency_anaphoric_drink_eat_backed_by_same_day_event(violetdb, tmp_path, monkeypatch):
    from datetime import timedelta

    from ai.consistency import ConsistencyGuard
    from ai.life_store import LifeStore
    from ai.memory_model import parse_time

    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    store = LifeStore(violetdb)
    now = parse_time(violetdb.now())
    guard = ConsistencyGuard(violetdb)

    # Without any same-day drink/food event, anaphoric claims are still caught
    assert "unrecorded_experience" in [v.check for v in guard.check_reply("你全喝完了也没关系的，我刚才已经喝过一些了")]
    assert "unrecorded_experience" in [v.check for v in guard.check_reply("我已经吃过一点了")]

    # Yesterday's drink event does NOT back today's anaphoric drink claim
    yesterday = (now - timedelta(days=1)).isoformat()
    store.insert_event(
        uid="daily:polish:yest-milk", thread_id=None, plan_id=None, event_type="daily",
        content="在便利店买了一盒草莓牛奶", occurred_at=yesterday, duration_minutes=45,
        source="life_daily", source_detail={"slot": "morning"}, actor="life_daily",
    )
    assert "unrecorded_experience" in [v.check for v in guard.check_reply("我刚才已经喝过一些了")]

    # Same-day drink event backs anaphoric drink clauses, but NOT explicit new drinks or unrelated eat claims
    today_earlier = now.replace(hour=max(0, now.hour - 1), minute=0, second=0, microsecond=0).isoformat()
    store.insert_event(
        uid="daily:polish:milk", thread_id=None, plan_id=None, event_type="daily",
        content="在便利店买了一盒草莓牛奶", occurred_at=today_earlier, duration_minutes=45,
        source="life_daily", source_detail={"slot": "morning"}, actor="life_daily",
    )
    for ok in ("你全喝完了也没关系的，我刚才已经喝过一些了", "我刚喝过一点了", "我已经喝了几口了"):
        assert guard.check_reply(ok) == [], ok
    for bad in ("我喝了咖啡", "我刚去吃了火锅", "我已经吃过一点了", "我刚才已经喝过一些了，我刚去吃了火锅"):
        assert "unrecorded_experience" in [v.check for v in guard.check_reply(bad)], bad

    # Washing a bento box does NOT count as a food event
    store.insert_event(
        uid="daily:polish:wash", thread_id=None, plan_id=None, event_type="daily",
        content="把便当盒洗干净擦干，整整齐齐收进橱柜", occurred_at=today_earlier, duration_minutes=30,
        source="life_daily", source_detail={"slot": "morning"}, actor="life_daily",
    )
    assert "unrecorded_experience" in [v.check for v in guard.check_reply("我已经吃过一点了")]

    # Same-day food event backs anaphoric eat clauses, while still catching fabricated named foods
    store.insert_event(
        uid="daily:polish:cookies", thread_id=None, plan_id=None, event_type="daily",
        content="试着烤了一盘饼干，边缘有点焦", occurred_at=today_earlier, duration_minutes=45,
        source="life_daily", source_detail={"slot": "afternoon"}, actor="life_daily",
    )
    for ok in ("我已经吃过一点了", "我刚吃过一些了"):
        assert guard.check_reply(ok) == [], ok
    for bad in ("刚吃完午饭", "我刚去吃了火锅", "我刚喝了咖啡"):
        assert "unrecorded_experience" in [v.check for v in guard.check_reply(bad)], bad


def test_virtual_test_apply_env_overrides_persists_in_eval_env(tmp_path, monkeypatch):
    import os
    from dotenv import dotenv_values
    from tools.scenario_eval import prepare
    from tools.virtual_test import _apply_env_overrides

    saved = dict(os.environ)
    try:
        work = tmp_path / "vt"
        work.mkdir()
        prepare(None, work, [], mock=True)
        eval_env = work / "eval.env"
        assert dotenv_values(eval_env)["VIOLET_ASYNC_MEMORY"] == "0"
        assert dotenv_values(eval_env)["VIOLET_LIFE_ENGINE"] == "0"

        overrides = {
            "VIOLET_ASYNC_MEMORY": "1",
            "VIOLET_LIFE_ENGINE": "1",
            "VIOLET_HOLIDAY_GREETING": "0",
            "VIOLET_IMAGE_ENABLED": "0",
            "VIOLET_MEMES": "1",
        }
        _apply_env_overrides(work, overrides)
        parsed = dotenv_values(eval_env)
        for k, v in overrides.items():
            assert parsed[k] == v, (k, parsed.get(k))
            assert os.environ[k] == v
        # No duplicate keys written to eval.env
        keys = [line.partition("=")[0] for line in eval_env.read_text(encoding="utf-8").splitlines() if "=" in line]
        assert keys.count("VIOLET_ASYNC_MEMORY") == 1
        assert keys.count("VIOLET_LIFE_ENGINE") == 1
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_situation_header_is_caught_by_outbound_leak_guard():
    from ai.channels import outbound

    assert "【当前情境（会过期）】" in outbound.internal_headers()
    assert outbound.find_leak("【当前情境（会过期）】\n- 地点：图书馆") == "internal_header"


def test_phone_state_chat_events_only_fallback_uses_guide_chat(monkeypatch):
    from ai import phone_events, phone_state

    monkeypatch.setenv("VIOLET_PHONE_CONTEXT_CHAT", "1")
    monkeypatch.setenv("VIOLET_PHONE_CONTEXT", "1")
    monkeypatch.setattr(phone_state, "_read", lambda calls=None: {})
    monkeypatch.setattr(phone_events, "recent_text", lambda current=None: "10 分钟前充上电了")
    phone_state.clear_cache()
    try:
        chat_out = phone_state.line("chat")
        assert "10 分钟前充上电了" in chat_out
        assert phone_state.GUIDE_CHAT in chat_out
        assert "不算勿扰" not in chat_out

        pro_out = phone_state.line("proactive")
        assert "10 分钟前充上电了" in pro_out
        assert "不算勿扰" in pro_out
    finally:
        phone_state.clear_cache()
