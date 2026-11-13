"""2026-11-12 rework after the virtual long-run test (user: 其他问题继续修，尽量不要打补丁，将有问题的这方面重构;
承不承认 AI 无所谓，倒希望不是很承认; 只有同住或同校的人设能演见面; 第一次聊天像刚认识，慢慢熟)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from tests.test_qq_social_violet import db  # noqa: F401


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def _bond(db) -> str:
    from ai.context_engine import ContextEngine
    return "\n".join(ContextEngine(db)._bond_section().lines)


# ------------------------------------------------------------------------------------------------ who you are now
def test_the_first_chat_is_a_first_meeting_and_she_is_not_next_to_you(db, data_dir):
    from ai.context_engine import ContextEngine
    text = _bond(db)
    assert ContextEngine.FIRST_MEETING in text and ContextEngine.APART in text and ContextEngine.IDENTITY in text
    db.add_message("user", "incoming", "text", "你好呀")
    db.add_message("user", "incoming", "text", "我叫小周")
    assert ContextEngine.NEW_TODAY in _bond(db)                    # later that day: still getting to know each other
    full = "\n".join(ContextEngine(db).build("chat", "在吗").sections)
    assert full.count("【你们的关系和现在的情况】") == 1 and "【你们的相处" not in full


def test_only_a_persona_whose_story_puts_her_with_you_may_meet_you(db, data_dir):
    from ai import persona_templates as pt
    from ai.context_engine import ContextEngine
    pt.switch(db, "quiet", "keep")
    text = _bond(db)
    assert "在同一所学校、住得也近" in text and ContextEngine.APART not in text
    assert "长什么样、穿什么，只按对方自己说的" in text and "你手头还没有和对方的聊天记录" in text
    pt.switch(db, "stepsister", "keep")
    assert "你们住在一起" in _bond(db)
    pt.switch(db, "violet", "keep")
    assert ContextEngine.APART in _bond(db)
    pt.switch(db, "ai", "keep")
    assert ContextEngine.IDENTITY not in _bond(db)                 # the AI persona says what it is itself


def test_a_chosen_relationship_is_given_not_guessed(db, data_dir):
    from ai import persona_templates as pt
    pt.set_relationship("partner", "宝宝")
    text = _bond(db)
    assert "你们是恋人" in text and "宝宝" in text and "刚认识" not in text


# ------------------------------------------------------------------------------------------------ what she is
def test_to_the_owner_she_may_dodge_but_never_claims_to_be_human(db, data_dir):
    from ai.consistency import ConsistencyGuard
    guard = ConsistencyGuard(db)
    for fine in ("你才AI", "我才不是AI呢，哼", "你猜", "我是不是AI重要吗"):
        assert not [v for v in guard._persona_violations(fine) if v.check == "identity_conflict"], fine
    for wrong in ("我是真人啊", "我也是个活人", "我就是人类"):
        assert [v for v in guard._persona_violations(wrong) if v.check == "identity_conflict"], wrong
    from ai.social.prompts import HARD_RULES
    assert "有人认真问你是不是 AI" in HARD_RULES and "不要否认" in HARD_RULES     # other people: not denied


# ------------------------------------------------------------------------------------------------ reminders
def test_reminders_are_shown_as_the_records_say(db, data_dir):
    from ai.context_engine import ContextEngine
    from ai.memory_model import parse_time
    now = parse_time(db.now())
    add = lambda **kw: db.add_commitment(**kw)["commitment"]["id"]           # noqa: E731
    soon = add(content="交作业", due_at=(now + timedelta(hours=2)).isoformat(),
                                evidence="两小时后提醒我交作业")
    late = add(content="起床", due_at=(now - timedelta(minutes=20)).isoformat(),
                                evidence="叫我起床")
    sent = add(content="喝水", due_at=(now - timedelta(hours=1)).isoformat(),
                                evidence="提醒我喝水")
    event = db.add_proactive_event(event_type="commitment_reminder", status="sent", action_type="chat", reason="x")
    db.update_proactive_event(event, status="sent", sent_messages=[{"type": "text", "content": "该喝水啦"}])
    db.update_commitment(sent, status="sent", delivered_at=(now - timedelta(minutes=59)).isoformat(),
                         proactive_event_id=event)
    section = ContextEngine(db)._commitment_section()
    text = "\n".join(section.lines)
    assert "发没发以这里为准" in section.header
    assert f"#{soon}" in text and "还没到时间" in text
    assert f"#{late}" in text and "时间到了，系统正在发，还没发出去" in text
    assert "喝水" in text and "已经提醒过了（发的是：该喝水啦）" in text


# ------------------------------------------------------------------------------------------------ said once
def test_a_line_the_voice_already_says_is_not_sent_again():
    from ai.chat_output import normalize_chat_output
    raw = ("鸡腿饭好，批准了。去排队吧，别急着回我，好好吃完再拍。我在这儿等着。\n"
           "<voice>鸡腿饭挺好，能吃上肉就行。</voice>\n<voice>你先去排队吧，不用急着回我，好好吃完饭再拍给我看</voice>\n"
           "排到了记得先吃两口再拍，菜别凉了")
    out = normalize_chat_output(raw, sticker_for_label=None, sticker_for_description=None)["messages"]
    assert [m["type"] for m in out] == ["voice", "voice", "text"]
    assert out[-1]["content"].startswith("排到了")


# ------------------------------------------------------------------------------------------------ memory: who is who
def test_the_memory_agent_writes_her_by_her_name_and_says_who_promised(db, data_dir):
    from ai import persona_templates as pt
    from ai.agents.memory_agent import who_is_who
    assert "Violet" in who_is_who() and "她现在叫" not in who_is_who()
    pt.switch(db, "stepsister", "keep")
    note = who_is_who()
    assert "她现在叫「沈念初」" in note and "沈念初答应给用户留一份" in note


# ------------------------------------------------------------------------------------------------ her own days
def test_each_persona_lives_her_own_days(db, data_dir):
    from datetime import date
    from ai import persona_templates as pt
    from ai.daily_life import DailyLife, _install_seed
    from ai.settings import local_timezone
    seed = _install_seed()
    assert seed and seed == _install_seed()                        # made once, then kept
    days = [date(2026, 11, d) for d in range(1, 29)]

    def lives(key):
        pt.switch(db, key, "keep")
        life = DailyLife(db)
        return [i["done"] for d in days for i in life.plan_for(d, local_timezone())]
    violet, junior = lives("violet"), lives("junior")
    assert junior and not set(junior) & set(violet)                 # her own list, not Violet's flat
    assert any("社团" in x or "图书馆" in x or "实验" in x for x in junior)
    assert "在家试着做了一次蛋炒饭，有点咸" not in junior
