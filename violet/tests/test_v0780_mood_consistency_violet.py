"""Regression tests for v0.7.80 chat, mood, consistency & prompt fixes.

Covers:
1. Owner-chat serious scolding mood fallback (zero false positives on teasing/affection/vents, no double-shift).
2. ConsistencyGuard fabricated daily completions, subjectless experiential forms, and same-day self-contradictions.
3. Prompt/context refinements (anti-looping follow-up, symmetric WHO_IS_WHO, prop-free stage samples) within budget.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta

import pytest

from tests.fakes import FakeLLM, completion, default_responders, install_fake_llm
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


def _insert_daily(db, content: str, *, hours_ago: int = 1, doing: str | None = None, uid: str = "daily:v780:1"):
    from ai.life_store import LifeStore
    from ai.memory_model import parse_time

    store = LifeStore(db)
    occurred = (parse_time(db.now()) - timedelta(hours=hours_ago)).isoformat()
    detail = {"slot": "evening"}
    if doing is not None:
        detail["doing"] = doing
    return store.insert_event(
        uid=uid, thread_id=None, plan_id=None, event_type="daily", content=content,
        occurred_at=occurred, duration_minutes=45, source="life_daily", source_detail=detail, actor="life_daily",
    )


# ---------------------------------------------------------------------------
# 1. Mood: serious scolding fallback + schema/abilities_note + zero FP
# ---------------------------------------------------------------------------

def test_note_feeling_schema_and_abilities_note_cover_brief_retreat_and_stickers():
    from ai.owner_tools import NOTE_FEELING_SCHEMA
    from ai.tool_manager import Catalog, abilities_note

    desc = NOTE_FEELING_SCHEMA["function"]["description"]
    assert "道谢" in desc and "举手之劳" in desc and "不打扰" in desc and "表情包" in desc and "必须" in desc
    assert "平淡闲聊" in desc and "玩笑互损" in desc
    note = abilities_note(Catalog(), [])
    assert "note_feeling" in note and "真心夸/谢/哄" in note and "→better" in note and "→worse" in note
    assert "客气带过" in note and "退一步" in note and "表情包" in note and "玩笑互损不用" in note


def test_mood_shift_drops_straight_below_zero_on_serious_scold_from_positive(isolated_data, violetdb):
    from ai import mood

    # Positive (+0.30) followed by serious scold (-0.35) drops from 0.0 -> -0.35 (crosses THRESHOLD)
    mood.shift(violetdb, 0.30, "被夸可爱")
    assert mood.current(violetdb)["value"] == pytest.approx(0.30)
    assert mood.shift(violetdb, -0.35, "被认真嫌弃") == pytest.approx(-0.35)
    assert "不高兴" in mood.note(violetdb)

    # Second serious scold from already negative stays additive (-0.35 + -0.35 = -0.70)
    assert mood.shift(violetdb, -0.35, "又被凶") == pytest.approx(-0.70)

    # Positive recovery from negative stays additive (-0.70 + 0.50 = -0.20)
    assert mood.shift(violetdb, 0.50, "被认真哄好了") == pytest.approx(-0.20)

    # Mild negative / half-weight group scold from positive stays additive (never drops straight below 0)
    violetdb.execute("DELETE FROM mood_state")
    mood.shift(violetdb, 0.30, "开心")
    assert mood.shift(violetdb, -0.10, "轻损") == pytest.approx(0.20)
    assert mood.shift(violetdb, -0.175, "群里路人半权重") == pytest.approx(0.025, abs=0.002)

    # Strong much_worse (-0.60) from positive drops straight to -0.60
    violetdb.execute("DELETE FROM mood_state")
    mood.shift(violetdb, 0.45, "很开心")
    assert mood.shift(violetdb, -0.60, "被狠狠骂了") == pytest.approx(-0.60)


def test_mood_scold_regex_and_owner_scold_fallback_precision(isolated_data, violetdb, monkeypatch):
    from ai import mood, stickers, test_mode

    scold_msg = "你真的很烦，说了多少遍了还是记不住，别跟我说话了"
    assert mood.scold_strength(scold_msg) > 0.0

    # Zero-false-positive cases: teasing, affection, mild scold alone, third-party vents, plain chat, work vent
    safe_cases = [
        ("笨蛋 你怎么这么可爱哈哈", "哈哈你才笨蛋"),
        ("笨蛋还不去睡觉", "马上睡啦"),
        ("小笨蛋，爱你", "我也爱你"),
        ("哈哈别跟我说话了笨蛋", "哼"),
        ("你好烦啊~别理我了", "好嘛好嘛"),
        ("甲方这需求真是合法但有病", "太离谱了"),
        ("甲方有病，别烦我了", "辛苦啦"),
        ("这破游戏太垃圾了，策划脑残吧", "哈哈确实"),
        ("那老板有病吧，加班到十点", "太过分了"),
        ("中午吃了碗面", "什么浇头呀"),
        ("今天工作烦死了，别跟我说话了让我静静", "好，你先静静"),
        ("闭嘴吧你，别剧透了", "哈哈行行行我不剧透了"),
        ("你真的很烦，闭嘴", "嗯嗯"),
    ]
    for msg, reply in safe_cases:
        violetdb.execute("DELETE FROM mood_state")
        assert mood.owner_scold_fallback(violetdb, msg, reply) is None, (msg, reply)
        assert mood.current(violetdb)["value"] == 0.0, (msg, reply)

    # Serious scolding + explicit dismissal reliably moves mood down <= -0.35 regardless of brief reply
    for msg, reply in [
        (scold_msg, "好，那我先不打扰你"),
        (scold_msg, "好"),
        ("真没用，别理我了", "……"),
        ("你太烦了，离我远点", "知道了"),
    ]:
        violetdb.execute("DELETE FROM mood_state")
        val = mood.owner_scold_fallback(violetdb, msg, reply)
        assert val is not None and val <= -0.35, (msg, val)

    # Hard insult + explicit dismissal moves even stronger
    violetdb.execute("DELETE FROM mood_state")
    assert mood.owner_scold_fallback(violetdb, "滚远点，白痴", "好") <= -0.45

    # Severe earnest abuse directed at her moves mood down
    violetdb.execute("DELETE FROM mood_state")
    assert mood.owner_scold_fallback(violetdb, "你真是个废物", "哦") <= -0.35

    # Hard command ("闭嘴") moves mood when her reply shows hurt/retreat OR a hurt/angry sticker
    hurt_sticker = stickers.add_bytes(violetdb, PNG + b"hurt", source="upload", description="委屈：眼泪汪汪；委屈、被冷落时发")
    violetdb.execute("DELETE FROM mood_state")
    assert mood.owner_scold_fallback(violetdb, "你真的很烦，闭嘴", f"[[表情:{hurt_sticker['id']}]]") <= -0.35

    violetdb.execute("DELETE FROM mood_state")
    assert mood.owner_scold_fallback(violetdb, "闭嘴吧你", "好，那我先不打扰你") <= -0.35

    # Respects VIOLET_MOOD=0 and test_mode
    violetdb.execute("DELETE FROM mood_state")
    monkeypatch.setenv("VIOLET_MOOD", "0")
    assert mood.owner_scold_fallback(violetdb, scold_msg, "好，那我先不打扰你") is None
    monkeypatch.setenv("VIOLET_MOOD", "1")

    test_mode.enable(violetdb, 1)
    try:
        assert mood.owner_scold_fallback(violetdb, scold_msg, "好，那我先不打扰你") is None
    finally:
        test_mode.disable()


def test_run_pipeline_owner_scold_fallback_and_no_double_shift(isolated_data, violetdb, monkeypatch):
    from ai import mood, stickers
    from ai.agents import ChatAgent, MemoryAgent
    from ai.chat import run_pipeline

    scold_msg = "你真的很烦，说了多少遍了还是记不住，别跟我说话了"

    # 1. Model misses note_feeling and only replies briefly to step back -> fallback shifts mood <= -0.35
    install_fake_llm(monkeypatch, FakeLLM(default_responders("好，那我先不打扰你")))
    violetdb.execute("DELETE FROM mood_state")
    run_pipeline(scold_msg, ChatAgent(), MemoryAgent(), defer_memory=True)
    assert mood.current(violetdb)["value"] <= -0.35

    # 2. Model misses note_feeling and only sends a hurt sticker -> fallback shifts mood <= -0.35
    hurt_sticker = stickers.add_bytes(violetdb, PNG + b"cry", source="upload", description="委屈：眼泪汪汪；委屈时发")
    install_fake_llm(monkeypatch, FakeLLM(default_responders(f"[[表情:{hurt_sticker['id']}]]")))
    violetdb.execute("DELETE FROM mood_state")
    run_pipeline(scold_msg, ChatAgent(), MemoryAgent(), defer_memory=True)
    assert mood.current(violetdb)["value"] <= -0.35

    # 3. Model DOES call note_feeling -> fallback does NOT double-shift
    steps = [
        completion(tool_calls=[{"name": "note_feeling", "arguments": {"change": "worse", "reason": "被认真嫌弃了"}}]),
        "好，那我先不打扰你",
    ]
    install_fake_llm(monkeypatch, FakeLLM({"fake-chat": lambda kw: steps.pop(0) if len(steps) > 1 else steps[0]}))
    violetdb.execute("DELETE FROM mood_state")
    run_pipeline(scold_msg, ChatAgent(), MemoryAgent(), defer_memory=True)
    assert mood.current(violetdb)["value"] == pytest.approx(-0.35, abs=0.02)

    # 4. Teasing / plain / third-party vent stay strictly 0.0 through full pipeline
    install_fake_llm(monkeypatch, FakeLLM(default_responders("哈哈好啦")))
    for benign in (
        "笨蛋 你怎么这么可爱哈哈",
        "中午吃了碗面",
        "笨蛋还不去睡觉",
        "甲方这需求真是合法但有病",
    ):
        violetdb.execute("DELETE FROM mood_state")
        run_pipeline(benign, ChatAgent(), MemoryAgent(), defer_memory=True)
        assert mood.current(violetdb)["value"] == 0.0, benign


# ---------------------------------------------------------------------------
# 2. ConsistencyGuard: fabricated completions & same-day self-contradictions
# ---------------------------------------------------------------------------

def test_consistency_catches_fabricated_daily_completions_and_experiential_claims(isolated_data, violetdb):
    from ai.consistency import ConsistencyGuard

    guard = ConsistencyGuard(violetdb)
    fabricated = [
        "刚吃完午饭",
        "刚吃完火锅",
        "刚吃完火锅，真香",
        "刚看完电影，挺好看的",
        "刚煮好面",
        "刚考完试",
        "早饭吃到快中午",
        "面煮好了，加了蛋",
        "我上学期刚考过",
        "上学期刚考过四级",
        "刚吃过午饭",
        "昨天刚烤好饼干",
        "刚泡了杯热茶",
        "衣服洗完了",
        "我今天睡到快中午",
        "刚刚忙完了",
        "我刚洗完澡",
        "刚洗完就清爽多了",
        "我刚看完这个就去吃饭，心理素质挺强啊",
        "你吃了吗，我刚吃完火锅",
        "你自己说的，吃完番茄炒蛋去午睡了，我刚吃完火锅",
        "你自己说的，吃完番茄炒蛋。刚吃完火锅",
        "你呢，刚吃完午饭",
    ]
    for line in fabricated:
        checks = [v.check for v in guard.check_reply(line)]
        assert "unrecorded_experience" in checks, (line, checks)

    # Idioms, user advice, comforting phrases, user-attributed recalls, questions, hedges MUST stay 0 false positives
    harmless = [
        "没事呀，忙完了就好，别太累",
        "老师这是直接把倍速拉满了吧，听完不得大脑一片空白",
        "看完岂不是更难受",
        "跑完还不累趴",
        "行，那快去洗吧，洗完就清爽多了",
        "洗完澡就舒服了",
        "喝完就精神了",
        "诶，刚看完这个就去吃饭，心理素质挺强啊",
        "刚考完试就去打球，心态真好啊",
        "考完了就好",
        "考完了就轻松啦",
        "考完就轻松啦",
        "忙完了好好休息",
        "忙完了就歇会",
        "忙完好好歇歇",
        "吃完了早点睡",
        "吃完饭了早点睡",
        "洗完澡了别着凉",
        "洗完澡早点睡",
        "吃完饭再聊",
        "睡醒了再回我",
        "睡醒再回我",
        "吃了就好",
        "喝了就暖和了",
        "我忙完就来找你",
        "我忙完了就来找你",
        "你自己说的，吃完食堂的番茄炒蛋然后去午睡了",
        "你中午跟我说，吃了黄焖鸡然后下午还有课",
        "你刚才说，考完试要去吃顿好的",
        "你之前提过，买了把新吉他",
        "你自己说的，已经回到家了",
        "我做好心理准备了",
        "我已经做好准备了",
        "我思考了一下你说的话",
        "我参考了你的建议",
        "我思考过这个问题",
        "睡到自然醒最舒服了",
        "早饭吃了吗？",
        "晚饭吃了没",
        "早饭吃了没有",
        "衣服洗完没",
        "作业写完了没",
        "吃完饭没呀",
        "洗完澡没呢",
        "你刚吃完午饭吗",
        "吃过饭记得喝水",
        "看过这部剧的人都说好",
        "上学期没考过",
        "我做了个决定，明天早起",
    ]
    for line in harmless:
        assert guard.check_reply(line) == [], line


def test_consistency_backs_recorded_cooking_and_oversleep_daily_events(isolated_data, violetdb):
    from ai.consistency import ConsistencyGuard

    _insert_daily(violetdb, "晚饭随便煮了碗面，卧了个荷包蛋", hours_ago=1, doing="在煮面当晚饭", uid="daily:v780:noodle")
    _insert_daily(violetdb, "午睡不小心睡过头了，醒来有点懵", hours_ago=2, doing="在午睡", uid="daily:v780:nap")
    _insert_daily(violetdb, "洗完澡吹干头发，整个人懒洋洋的", hours_ago=1, doing="刚洗完澡在吹头发", uid="daily:v780:bath")
    _insert_daily(violetdb, "泡了杯蜂蜜水，坐在阳台发了会儿呆", hours_ago=1, doing="在阳台发呆", uid="daily:v780:tea")

    guard = ConsistencyGuard(violetdb)
    assert guard.check_reply("晚饭煮了碗面，加了个蛋") == []
    assert guard.check_reply("我午睡睡过头了，现在还有点懵") == []
    assert guard.check_reply("刚洗完澡，整个人懒洋洋的") == []
    assert guard.check_reply("刚泡了杯蜂蜜水") == []

    # Unrecorded food still caught
    assert [v.check for v in guard.check_reply("刚烤了一盘蛋挞")] == ["unrecorded_experience"]


def test_consistency_same_day_self_contradiction_and_life_section_header(isolated_data, violetdb):
    from ai import context_cut
    from ai.chat import CONSISTENCY_HINTS, guard_reply
    from ai.consistency import ConsistencyGuard
    from ai.context_engine import ContextEngine
    from ai.memory_model import parse_time

    assert "same_day_contradiction" in CONSISTENCY_HINTS
    now = parse_time(violetdb.now())
    day = now.strftime("%Y-%m-%d")

    # Earlier today Violet stated she already ate lunch
    violetdb.add_assistant_message("text", "刚吃完午饭，正打算歇会儿", created_at=f"{day}T12:15:00+08:00")
    guard = ConsistencyGuard(violetdb)

    # Later today claiming she hasn't eaten lunch OR ate breakfast until noon -> flagged
    v_not_eaten = guard.check_reply("我午饭还没吃呢", now=f"{day}T14:00:00+08:00")
    assert "same_day_contradiction" in [v.check for v in v_not_eaten]

    v_late_bf = guard.check_reply("今天早饭吃到快中午", now=f"{day}T14:00:00+08:00")
    assert "same_day_contradiction" in [v.check for v in v_late_bf]

    # Non-contradicting remarks must NOT trigger same_day_contradiction
    for ok in ("其实午饭没吃饱", "你午饭还没吃吗？", "晚饭还没吃呢"):
        assert "same_day_contradiction" not in [v.check for v in guard.check_reply(ok, now=f"{day}T14:00:00+08:00")], ok

    # Different day does NOT contradict today
    next_day = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    assert "same_day_contradiction" not in [v.check for v in guard.check_reply("我午饭还没吃呢", now=f"{next_day}T11:30:00+08:00")]

    # After context_cut, pre-cut messages do NOT trigger same_day_contradiction
    violetdb.execute("DELETE FROM messages")
    violetdb.add_assistant_message("text", "刚吃完午饭", created_at=f"{day}T12:00:00+08:00")
    context_cut.cut(violetdb, "all")
    later_now = (parse_time(violetdb.now()) + timedelta(minutes=5)).isoformat()
    assert "same_day_contradiction" not in [v.check for v in guard.check_reply("我午饭还没吃", now=later_now)]

    # Time moving forward ("还没吃午饭" -> "刚吃完午饭") is NOT a contradiction
    violetdb.execute("DELETE FROM messages")
    (isolated_data / "control" / "context-cut.json").unlink(missing_ok=True)
    violetdb.add_assistant_message("text", "午饭还没吃呢", created_at=f"{day}T11:30:00+08:00")
    assert "same_day_contradiction" not in [v.check for v in guard.check_reply("刚吃完午饭", now=f"{day}T13:00:00+08:00")]

    # Went out earlier today -> claiming stayed home all day is a contradiction
    violetdb.execute("DELETE FROM messages")
    violetdb.add_assistant_message("text", "下午去了公园散步", created_at=f"{day}T15:00:00+08:00")
    v_home = guard.check_reply("我今天一整天没出门", now=f"{day}T19:00:00+08:00")
    assert "same_day_contradiction" in [v.check for v in v_home]

    # Non-today day markers ("昨天", "上周", "以前", "平时") and user-attributed recalls MUST NOT poison today's facts
    for prev_msg, curr_msg in [
        ("昨天去了公园散步", "我今天一整天没出门"),
        ("昨天上午去了书店，顺便吃了午饭", "我午饭还没吃呢"),
        ("昨天吃了午饭就犯困", "我午饭还没吃呢"),
        ("昨天早饭没吃", "刚吃完早饭"),
        ("平时一般都早饭没吃", "刚吃完早饭"),
        ("你自己说的，刚吃完午饭然后去午睡了", "我午饭还没吃呢"),
    ]:
        violetdb.execute("DELETE FROM messages")
        violetdb.add_assistant_message("text", prev_msg, created_at=f"{day}T10:00:00+08:00")
        checks = [v.check for v in guard.check_reply(curr_msg, now=f"{day}T13:00:00+08:00")]
        assert "same_day_contradiction" not in checks, (prev_msg, curr_msg, checks)

    # guard_reply rewrites (or drops) same_day_contradiction as a certain finding
    violetdb.execute("DELETE FROM messages")
    violetdb.add_assistant_message("text", "刚吃完午饭", created_at=f"{day}T12:00:00+08:00")
    rewritten, found = guard_reply("我午饭还没吃呢\n你下午有课吗", regenerate=lambda note: "下午还有点犯困\n你下午有课吗")
    assert "same_day_contradiction" in [v.check for v in found]
    assert rewritten == "下午还有点犯困\n你下午有课吗"

    # _life_section header reinforces consistency with today's earlier replies
    packet = ContextEngine(violetdb).build("chat", "你在干嘛")
    life_sec = next(s for s in packet.sections if s.startswith("【你的生活状态"))
    assert "别现编" in life_sec and "矛盾" in life_sec


# ---------------------------------------------------------------------------
# 3. Prompt & context refinements + system prompt char budget & cache stability
# ---------------------------------------------------------------------------

def test_prompt_refinements_stage_sample_guard_and_budget(isolated_data, violetdb, monkeypatch):
    from ai import persona_templates as pt, stickers
    from ai.agents import ChatAgent
    from ai.agents.base_agent import _CALL_CONTEXT
    from ai.agents.chat_agent import WHO_IS_WHO
    from ai.context_engine import ContextEngine
    from ai.settings import PROJECT_ROOT
    from ai.tool_manager import Catalog, abilities_note, direct_note

    monkeypatch.delenv("CHAT_LLM_PROMPT", raising=False)

    # 3a. Anti-looping follow-up in chat.v4.txt
    v4 = (PROJECT_ROOT / "persona" / "chat_prompt_versions" / "chat.v4.txt").read_text(encoding="utf-8")
    assert "最多顺口提一次就放手" in v4 and "绝不反复追问" in v4

    # 3b. Symmetric WHO_IS_WHO (keeps existing rule + adds transcript-backed pushback)
    assert "不要说“你没跟我说过 / 你没提”" in WHO_IS_WHO
    assert "我记混了" in WHO_IS_WHO and "干脆放人" in WHO_IS_WHO

    # 3c. stepsister.json stage 0 & 1 samples no longer plant soup/cookies props
    ss = json.loads((PROJECT_ROOT / "persona" / "templates" / "stepsister.json").read_text(encoding="utf-8"))
    stage_samples = "\n".join(s.get("sample", "") for s in ss["stages"])
    assert "汤在锅里温着" not in stage_samples and "烤饼干" not in stage_samples

    # Both default Violet (kind='close') and stepsister (kind='persona') get the strengthened sample_hint
    for tpl in ("violet", "stepsister"):
        pt.switch(violetdb, tpl, "keep")
        bond = "\n".join(ContextEngine(violetdb)._bond_section().lines)
        assert "严禁照抄原句或复用里面的具体物件/事件" in bond, tpl
        assert "【你的生活状态】" in bond and "真实居住情况" in bond, tpl

    # System prompt budget (< 3600 chars) and prefix-cache byte stability with 25 stickers
    pt.switch(violetdb, "violet", "keep")
    for i in range(25):
        raw = PNG + f"v780-sticker-{i}".encode("utf-8")
        desc = f"画面：白色小猫翻白眼写着就这{i}；什么时候发：开玩笑吐槽时发；关键词：吐槽{i}"
        row = stickers.add_bytes(violetdb, raw, source="upload", description=desc)
        if i < 12:
            stickers.mark_used(violetdb, row["id"])

    agent = ChatAgent()
    ctx = {"direct_tools": [], "tool_note": direct_note([]), "abilities_note": abilities_note(Catalog(), [])}
    tok = _CALL_CONTEXT.set(dict(ctx))
    try:
        sys1 = agent.build_context_messages("今天好累")[0]["content"]
    finally:
        _CALL_CONTEXT.reset(tok)
    assert len(sys1) < 3600, f"system prompt exceeded budget: {len(sys1)} chars"

    tok = _CALL_CONTEXT.set(dict(ctx))
    try:
        sys2 = agent.build_context_messages("明天再聊吧")[0]["content"]
    finally:
        _CALL_CONTEXT.reset(tok)
    assert sys1 == sys2


def test_persona_sandbox_aligns_stage_sample_anti_copy_hint(isolated_data, violetdb, monkeypatch):
    from ai.agents import ChatAgent
    from wx_ilink_py import ILinkClient, create_app

    token = os.environ["VIOLET_ADMIN_TOKEN"]
    headers = {"Authorization": f"Bearer {token}", "Origin": "http://localhost"}
    llm = install_fake_llm(monkeypatch, FakeLLM(default_responders("嗯，你先忙")))

    app = create_app(
        client=ILinkClient(data_dir=isolated_data, auto_login=False),
        violetdb=violetdb,
        proactive_scheduler=None,
        all_agents=[ChatAgent()],
        admin_token=token,
    )
    resp = app.test_client().post(
        "/api/persona/sandbox",
        json={"message": "在吗", "template": "stepsister", "stage": 0},
        headers=headers,
    )
    assert resp.status_code == 200
    joined = "\n".join(str(m.get("content") or "") for m in llm.calls[-1]["messages"])
    assert "严禁照抄原句或复用里面的具体物件/事件" in joined
    assert "【你的生活状态】" in joined and "真实居住情况" in joined
    assert "汤在锅里温着" not in joined


def test_proactive_life_section_includes_time_of_day_period_labels(isolated_data, violetdb, monkeypatch):
    from ai.context_engine import ContextEngine
    from ai.life_store import LifeStore

    monkeypatch.setattr(violetdb, "now", lambda: "2026-11-25T20:30:00+08:00")
    store = LifeStore(violetdb)
    store.insert_event(
        uid="daily:2026-11-25:afternoon", thread_id=None, plan_id=None, event_type="daily",
        content="在公用自习室改完了一份小组作业的排版",
        occurred_at="2026-11-25T15:15:00+08:00", duration_minutes=45,
        source="life_daily", source_detail={"slot": "afternoon", "label": "下午", "doing": "在自习室改作业排版"},
        actor="life_daily",
    )
    store.insert_event(
        uid="daily:2026-11-25:evening", thread_id=None, plan_id=None, event_type="daily",
        content="泡了杯蜂蜜水，坐在阳台发了会儿呆",
        occurred_at="2026-11-25T18:05:00+08:00", duration_minutes=45,
        source="life_daily", source_detail={"slot": "evening"},
        actor="life_daily",
    )
    store.insert_event(
        uid="engine:2026-11-24:morning", thread_id=None, plan_id=None, event_type="progress",
        content="读了《边城》的一部分",
        occurred_at="2026-11-24T09:30:00+08:00", duration_minutes=60,
        source="life_engine:template", source_detail={},
        actor="life_engine",
    )

    packet = ContextEngine(violetdb).build("proactive", "[proactive event]", {})
    life_sec = next(s for s in packet.sections if s.startswith("【你的生活状态"))
    assert "今天下午（11-25）：在公用自习室改完了一份小组作业的排版（已发生）" in life_sec
    assert "今天晚上（11-25）：泡了杯蜂蜜水，坐在阳台发了会儿呆（已发生）" in life_sec
    assert "昨天上午（11-24）：读了《边城》的一部分（已发生）" in life_sec
    assert "早些时段的事别说成“刚”在做" in life_sec
    assert life_sec in packet.volatile

