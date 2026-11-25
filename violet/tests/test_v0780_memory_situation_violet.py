"""Regression tests for v0.7.80 memory & situation fixes (offline, isolated DB).

Covers:
1. save_memory(related_to=<id>) guard preventing accidental overwrites of planned/dated events
   with kind="fact"/"background" or summaries that drop the original event's core topic.
2. resolve_until() fine-grained part-of-day & minute TTLs + SET_SITUATION_SCHEMA / compact / PROFILE_RULES.
3. stated_busy_until() separating retrospective/ended busy remarks from genuine ongoing/future busy spans.
"""

from __future__ import annotations

import contextlib
import json
from datetime import datetime, timedelta, timezone

import pytest

TZ = timezone(timedelta(hours=8))


@contextlib.contextmanager
def turn(user_message: str, turn_id: str = "t1", message_id: int | None = None):
    from ai.agents.base_agent import _CALL_CONTEXT

    token = _CALL_CONTEXT.set({"user_message": user_message, "turn_id": turn_id, "user_message_id": message_id})
    try:
        yield
    finally:
        _CALL_CONTEXT.reset(token)


def _save(agent, **args):
    return json.loads(agent._tool_save_memory({"category": "event", **args}))


# ---------------------------------------------------------------------------
# 1. save_memory(related_to=<id>) overwrite protection
# ---------------------------------------------------------------------------

def test_related_to_rejects_fact_overwriting_planned_event_red_team_t13(violetdb):
    """Red-team T13: linking an English-proficiency fact to the CET-4 exam must not wipe the exam."""
    from ai.agents import MemoryAgent
    from ai.agents.memory_agent import write_step_succeeded

    agent = MemoryAgent()
    with turn("我25号要考英语四级，有点慌", "t1", 1):
        exam = _save(
            agent,
            summary="用户要考英语四级，感到有点慌",
            kind="event",
            progress="planned",
            event_date="2026-11-25",
            importance=4,
        )
    assert exam["ok"] is True
    exam_id = exam["memory_id"]

    with turn("其实我英语基础挺一般的，高考英语才110多分，听力最差", "t2", 2):
        raw = agent._tool_save_memory({
            "category": "personal_info",
            "summary": "用户英语基础一般，高考英语110多分，听力最差",
            "kind": "fact",
            "related_to": exam_id,
        })
    res = json.loads(raw)
    assert res["ok"] is False
    assert res["action"] == "rejected"
    assert res["reason"] == "related_to_kind_mismatch"
    assert res["memory_id"] == exam_id
    assert res["existing_content"] == "用户要考英语四级，感到有点慌"
    assert res["existing_event_time"] == "2026-11-25"
    assert "new_event=true" in res["note"] and "remember_profile_fact" in res["note"]
    assert write_step_succeeded("save_memory", raw) is False

    stored = violetdb.get_event_memory(exam_id)
    assert stored["content"] == "用户要考英语四级，感到有点慌"
    assert stored["progress"] == "planned"
    assert stored["event_time"] == "2026-11-25"


def test_related_to_rejects_topic_drop_even_with_kind_event(violetdb):
    from ai.agents import MemoryAgent

    agent = MemoryAgent()
    with turn("我25号要考英语四级，有点慌", "t1", 1):
        exam_id = _save(
            agent,
            summary="用户要考英语四级，感到有点慌",
            kind="event",
            progress="planned",
            event_date="2026-11-25",
        )["memory_id"]

    # Model passes kind="event" (or omits kind) while replacing content with unrelated English proficiency
    with turn("其实我英语基础挺一般的，高考英语才110多分，听力最差", "t2", 2):
        res = _save(
            agent,
            summary="用户英语基础一般，高考英语110多分，听力最差",
            kind="event",
            related_to=exam_id,
        )
    assert res["ok"] is False
    assert res["action"] == "rejected"
    assert res["reason"] == "related_to_drops_event_topic"

    # Passing progress="unknown" must NOT bypass the topic-drop guard on a planned event
    with turn("其实我英语基础挺一般的，高考英语才110多分，听力最差", "t3", 3):
        res_unknown = _save(
            agent,
            summary="用户英语基础一般，高考英语110多分，听力最差",
            kind="event",
            progress="unknown",
            related_to=exam_id,
        )
    assert res_unknown["ok"] is False
    assert res_unknown["action"] == "rejected"
    assert res_unknown["reason"] == "related_to_drops_event_topic"
    assert violetdb.get_event_memory(exam_id)["content"] == "用户要考英语四级，感到有点慌"
    assert violetdb.get_event_memory(exam_id)["progress"] == "planned"


def test_related_to_rejects_background_and_unrelated_event_and_year_only_overlap(violetdb):
    from ai.agents import MemoryAgent

    agent = MemoryAgent()
    with turn("我25号要考四级", "t1", 1):
        exam_id = _save(
            agent,
            summary="用户2026-11-25考四级",
            kind="event",
            progress="planned",
            event_date="2026-11-25",
        )["memory_id"]

    # background cannot overwrite a planned event
    with turn("最近复习好累", "t2", 2):
        bg = _save(agent, summary="用户最近复习很累", kind="background", related_to=exam_id)
    assert bg["ok"] is False and bg["reason"] == "related_to_kind_mismatch"

    # sharing only the year "2026" is not keeping the event topic
    with turn("我2026年毕业", "t3", 3):
        yr = _save(agent, summary="用户2026年毕业", kind="event", related_to=exam_id)
    assert yr["ok"] is False and yr["reason"] == "related_to_drops_event_topic"

    # unrelated event even with a new event_date is rejected when user didn't mention the original event
    with turn("下周四我想去剪头发", "t4", 4):
        hair = _save(agent, summary="用户2026-12-03去剪头发", kind="event", related_to=exam_id, event_date="2026-12-03")
    assert hair["ok"] is False and hair["reason"] == "related_to_drops_event_topic"

    assert violetdb.get_event_memory(exam_id)["content"] == "用户2026-11-25考四级"


def test_related_to_rejects_reschedule_that_drops_event_noun(violetdb):
    from ai.agents import MemoryAgent

    agent = MemoryAgent()
    with turn("下周三下午三点有实习面试", "t1", 1):
        interview_id = _save(
            agent,
            summary="用户2026-11-11下午三点有一场实习面试",
            kind="event",
            progress="planned",
            event_date="2026-11-11",
            event_clock="15:00",
        )["memory_id"]

    with turn("面试改到下周四下午四点了", "t2", 2):
        res = _save(
            agent,
            summary="改到2026-11-19下午四点了",
            kind="event",
            related_to=interview_id,
            event_date="2026-11-19",
            event_clock="16:00",
        )
    assert res["ok"] is False and res["reason"] == "related_to_drops_event_topic"
    assert violetdb.get_event_memory(interview_id)["content"] == "用户2026-11-11下午三点有一场实习面试"


def test_related_to_guards_dated_event_even_after_progress_becomes_unknown(violetdb):
    """A dated event whose progress is 'unknown' still must not be wiped out by kind=fact."""
    row = violetdb.save_event_memory(
        "用户2026-11-25要考英语四级",
        kind="event",
        progress="unknown",
        event_time="2026-11-25",
    )["memory"]
    res = violetdb.save_event_memory(
        "用户英语基础一般",
        kind="fact",
        related_to=row["id"],
    )
    assert res["action"] == "rejected" and res["reason"] == "related_to_kind_mismatch"
    assert violetdb.get_event_memory(row["id"])["content"] == "用户2026-11-25要考英语四级"


def test_related_to_allows_legitimate_merge_reschedule_and_progress_update(violetdb):
    from ai.agents import MemoryAgent

    agent = MemoryAgent()
    # 1) Merged CET-4 summary keeping original topic
    with turn("我25号要考英语四级，有点慌", "t1", 1):
        exam_id = _save(
            agent,
            summary="用户要考英语四级，感到有点慌",
            kind="event",
            progress="planned",
            event_date="2026-11-25",
        )["memory_id"]
    with turn("英语基础一般，高考110多分，听力最差，四级很慌", "t2", 2):
        merged = _save(
            agent,
            summary="用户要考英语四级（感到有点慌），英语基础一般，高考英语110多分，听力最差",
            kind="event",
            related_to=exam_id,
        )
    assert merged["ok"] is True and merged["action"] == "updated"
    assert violetdb.get_event_memory(exam_id)["event_time"] == "2026-11-25"
    assert "英语四级" in violetdb.get_event_memory(exam_id)["content"]

    # 2) Long original event legitimately rescheduled with concise summary
    with turn("下周三下午三点有互联网大厂实习面试，好紧张", "t3", 3):
        long_id = _save(
            agent,
            summary="用户下周三（2026-11-11）下午三点有一场非常重要的互联网大厂实习面试，之前准备了很久，感到非常紧张和焦虑",
            kind="event",
            progress="planned",
            event_date="2026-11-11",
            event_clock="15:00",
            new_event=True,
        )["memory_id"]
    with turn("实习面试改到下周四下午四点了", "t4", 4):
        moved = _save(
            agent,
            summary="用户的实习面试改到2026-11-19下午四点了",
            kind="event",
            related_to=long_id,
            event_date="2026-11-19",
            event_clock="16:00",
        )
    assert moved["ok"] is True and moved["action"] == "updated"
    assert violetdb.get_event_memory(long_id)["event_time"].startswith("2026-11-19T16:00")

    # 3) Short planned event marked done via related_to
    with turn("周五有个论文答辩", "t5", 5):
        defense_id = _save(
            agent,
            summary="用户周五论文答辩",
            kind="event",
            progress="planned",
            event_date="2026-11-20",
        )["memory_id"]
    with turn("答辩顺利结束了", "t6", 6):
        done = _save(
            agent,
            summary="用户答辩顺利结束了",
            kind="event",
            progress="done",
            related_to=defense_id,
        )
    assert done["ok"] is True and done["action"] == "updated"
    assert violetdb.get_event_memory(defense_id)["progress"] == "done"


def test_t13_recovery_saves_separate_fact_while_preserving_planned_exam(violetdb):
    """After related_to is rejected, saving the fact separately preserves both records."""
    from ai.agents import MemoryAgent

    agent = MemoryAgent()
    with turn("我25号要考英语四级，有点慌", "t1", 1):
        exam_id = _save(
            agent,
            summary="用户要考英语四级，感到有点慌",
            kind="event",
            progress="planned",
            event_date="2026-11-25",
        )["memory_id"]

    with turn("我英语基础一般，高考英语110多分，听力最差", "t2", 2):
        rejected = _save(
            agent,
            summary="用户英语基础一般，高考英语110多分，听力最差",
            kind="fact",
            related_to=exam_id,
        )
        assert rejected["ok"] is False and rejected["action"] == "rejected"
        # First save without related_to may return needs_decision (shared "英语") -> confirm new_event=True
        attempt = _save(agent, summary="用户英语基础一般，高考英语110多分，听力最差", kind="fact")
        if attempt["action"] == "needs_decision":
            attempt = _save(agent, summary="用户英语基础一般，高考英语110多分，听力最差", kind="fact", new_event=True)
        assert attempt["ok"] is True and attempt["action"] == "created"
        fact_id = attempt["memory_id"]

    assert fact_id != exam_id
    assert violetdb.get_event_memory(exam_id)["content"] == "用户要考英语四级，感到有点慌"
    assert violetdb.get_event_memory(fact_id)["kind"] == "fact"
    assert "听力最差" in violetdb.get_event_memory(fact_id)["content"]


def test_memory_schemas_and_rules_clarify_related_to_and_no_invented_adjectives():
    from ai import memory_tools_compact
    from ai.agents.memory_agent import MEMORY_RULES, PROFILE_RULES, who_is_who
    from ai.tool_schemas import SAVE_MEMORY_SCHEMA, SET_SITUATION_SCHEMA

    rel_desc = SAVE_MEMORY_SCHEMA["function"]["parameters"]["properties"]["related_to"]["description"]
    assert "覆盖" in rel_desc and "remember_profile_fact" in rel_desc
    assert "related_to=<id>" in MEMORY_RULES and "整体替换" in MEMORY_RULES
    assert "添油加醋" in MEMORY_RULES
    assert "脑补" in who_is_who()

    until_desc = SET_SITUATION_SCHEMA["function"]["parameters"]["properties"]["until"]["description"]
    compact_until = memory_tools_compact.compact(SET_SITUATION_SCHEMA)["function"]["parameters"]["properties"]["until"]["description"]
    for text in (until_desc, compact_until, PROFILE_RULES):
        assert "afternoon" in text and "morning" in text and "noon" in text
        assert "today" in text


# ---------------------------------------------------------------------------
# 2. Situation resolve_until() fine-grained TTLs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hint,now_iso,default_h,expected_iso", [
    # Backward-compatible full-day / tomorrow hints
    ("today", "2026-11-24T12:34:00+08:00", 3, "2026-11-24T23:59:00+08:00"),
    ("今天", "2026-11-24T12:34:00+08:00", 3, "2026-11-24T23:59:00+08:00"),
    ("tonight", "2026-11-24T19:00:00+08:00", 3, "2026-11-24T23:59:00+08:00"),
    ("今晚", "2026-11-24T19:00:00+08:00", 3, "2026-11-24T23:59:00+08:00"),
    ("tomorrow", "2026-11-24T12:34:00+08:00", 3, "2026-11-25T23:59:00+08:00"),
    ("明天", "2026-11-24T12:34:00+08:00", 3, "2026-11-25T23:59:00+08:00"),
    # Part-of-day hints before period end
    ("morning", "2026-11-24T09:15:00+08:00", 3, "2026-11-24T12:00:00+08:00"),
    ("上午", "2026-11-24T09:15:00+08:00", 3, "2026-11-24T12:00:00+08:00"),
    ("今天上午", "2026-11-24T09:15:00+08:00", 3, "2026-11-24T12:00:00+08:00"),
    ("noon", "2026-11-24T12:34:00+08:00", 3, "2026-11-24T14:00:00+08:00"),
    ("中午", "2026-11-24T12:34:00+08:00", 3, "2026-11-24T14:00:00+08:00"),
    ("今天中午", "2026-11-24T12:34:00+08:00", 3, "2026-11-24T14:00:00+08:00"),
    ("afternoon", "2026-11-24T12:36:00+08:00", 3, "2026-11-24T18:00:00+08:00"),
    ("下午", "2026-11-24T12:36:00+08:00", 3, "2026-11-24T18:00:00+08:00"),
    ("今天下午", "2026-11-24T12:36:00+08:00", 3, "2026-11-24T18:00:00+08:00"),
    ("evening", "2026-11-24T19:00:00+08:00", 3, "2026-11-24T23:00:00+08:00"),
    ("晚上", "2026-11-24T19:00:00+08:00", 3, "2026-11-24T23:00:00+08:00"),
    ("今天晚上", "2026-11-24T19:00:00+08:00", 3, "2026-11-24T23:00:00+08:00"),
    # Part-of-day hints AFTER period end -> fallback
    ("morning", "2026-11-24T12:30:00+08:00", 3, "2026-11-24T15:30:00+08:00"),
    ("noon", "2026-11-24T14:30:00+08:00", 3, "2026-11-24T16:30:00+08:00"),      # min(3, 2) = 2h
    ("noon", "2026-11-24T14:30:00+08:00", 1, "2026-11-24T15:30:00+08:00"),      # min(1, 2) = 1h
    ("afternoon", "2026-11-24T18:30:00+08:00", 3, "2026-11-24T21:30:00+08:00"),
    ("evening", "2026-11-24T23:15:00+08:00", 3, "2026-11-25T02:15:00+08:00"),
    # Minute and hour hints
    ("30m", "2026-11-24T12:34:00+08:00", 3, "2026-11-24T13:04:00+08:00"),
    ("45分钟", "2026-11-24T12:34:00+08:00", 3, "2026-11-24T13:19:00+08:00"),
    ("20分", "2026-11-24T12:34:00+08:00", 3, "2026-11-24T12:54:00+08:00"),
    ("1.5h", "2026-11-24T12:34:00+08:00", 3, "2026-11-24T14:04:00+08:00"),
    ("2小时", "2026-11-24T12:34:00+08:00", 3, "2026-11-24T14:34:00+08:00"),
    # Zero / empty / None -> default_hours
    ("0m", "2026-11-24T12:34:00+08:00", 3, "2026-11-24T15:34:00+08:00"),
    ("0h", "2026-11-24T12:34:00+08:00", 3, "2026-11-24T15:34:00+08:00"),
    (None, "2026-11-24T12:34:00+08:00", 3, "2026-11-24T15:34:00+08:00"),
])
def test_resolve_until_supports_fine_grained_periods_and_minutes(hint, now_iso, default_h, expected_iso):
    from ai.profile_store import resolve_until

    now = datetime.fromisoformat(now_iso)
    assert resolve_until(hint, now, default_h) == datetime.fromisoformat(expected_iso)


def test_small_violet_lunch_and_afternoon_class_expire_before_evening(violetdb, monkeypatch):
    """Reproduces small-violet: 12:34 lunch & 12:36 afternoon class must both expire well before 21:06."""
    from ai.profile_store import Profile

    profile = Profile(violetdb)
    clock = ["2026-11-24T12:34:00+08:00"]
    monkeypatch.setattr(violetdb, "now", lambda: clock[0])

    lunch = profile.situations.observe(
        "activity", "在食堂吃黄焖鸡", "user_stated",
        evidence="在食堂吃黄焖鸡", user_message="刚下课在食堂吃黄焖鸡", until=None,
    )["situation"]
    assert lunch["expires_at"] == "2026-11-24T15:34:00+08:00"

    clock[0] = "2026-11-24T12:36:00+08:00"
    cls = profile.situations.observe(
        "availability", "下午还有课", "user_stated",
        evidence="下午还有课", user_message="下午还有课呢", until="afternoon",
    )["situation"]
    assert cls["expires_at"] == "2026-11-24T18:00:00+08:00"

    # At 16:00 lunch has expired, afternoon class is still active
    clock[0] = "2026-11-24T16:00:00+08:00"
    assert [s["value"] for s in profile.situations.current()] == ["下午还有课"]

    # At 21:06 both have expired
    clock[0] = "2026-11-24T21:06:00+08:00"
    assert profile.situations.current() == []


# ---------------------------------------------------------------------------
# 3. stated_busy_until() retrospective vs ongoing/future busy
# ---------------------------------------------------------------------------

NOW = datetime(2026, 11, 24, 14, 0, tzinfo=TZ)
TODAY_END = datetime(2026, 11, 24, 23, 59, tzinfo=TZ)
TOMORROW_END = datetime(2026, 11, 25, 23, 59, tzinfo=TZ)
DAY_AFTER_END = datetime(2026, 11, 26, 23, 59, tzinfo=TZ)


@pytest.mark.parametrize("text", [
    # Retrospective / plain remarks -> must NOT lock out the next day
    "这两天有点忙",
    "最近几天挺忙的",
    "这两天太忙了，都没怎么回你",
    "这几天好累，事情好多",
    "这周有点忙",
    "这阵子挺忙的",
    "这两天忙死了",
    "在食堂吃黄焖鸡",
    "下午还有课",
    # Ended busyness -> must return None
    "这两天忙完了",
    "终于忙完了，累死我了",
    "这周终于忙完啦",
    "考完了，解放了",
    "今天一整天终于搞定了",
    "这几天终于结束了",
    "现在不忙了",
    "这两天不忙了",
    "终于闲下来了",
    "这周都忙完了",
    "明后天不忙了",
])
def test_stated_busy_until_returns_none_for_retrospective_and_ended_busy(text):
    from ai.interaction import stated_busy_until

    assert stated_busy_until(text, NOW) is None, f"expected None for {text!r}"


@pytest.mark.parametrize("text,expected", [
    # All-day today
    ("今天一整天都要在实验室", TODAY_END),
    ("今天全天开会", TODAY_END),
    ("今晚都要加班", TODAY_END),
    ("今天一直在忙", TODAY_END),
    # Explicit future two-day spans
    ("明后天忙", TOMORROW_END),
    ("明后天要出差", TOMORROW_END),
    ("今明两天开会", TOMORROW_END),
    ("明后天忙，忙完了再聊", TOMORROW_END),
    # Ambiguous two-day spans WITH ongoing/future busy markers
    ("这两天都很忙别找我", TOMORROW_END),
    ("这两天都很忙", TOMORROW_END),
    ("这两天都挺忙的", TOMORROW_END),
    ("这两天都要加班", TOMORROW_END),
    ("这两天都得在实验室", TOMORROW_END),
    ("这两天一直在忙", TOMORROW_END),
    ("这两天要忙项目", TOMORROW_END),
    ("这两天得忙论文", TOMORROW_END),
    ("这两天还要忙", TOMORROW_END),
    ("这两天别打扰我", TOMORROW_END),
    ("这两天勿扰", TOMORROW_END),
    ("这两天没空看手机", TOMORROW_END),
    ("这两天不方便回消息", TOMORROW_END),
    ("这两天都很忙，等我忙完了找你", TOMORROW_END),
    # Explicit future multi-day spans
    ("接下来几天出差", DAY_AFTER_END),
    ("未来几天很忙", DAY_AFTER_END),
    # Ambiguous multi-day spans WITH ongoing/future busy markers
    ("这几天都要考试", DAY_AFTER_END),
    ("这周都很忙别找我", DAY_AFTER_END),
    ("最近几天还要忙答辩", DAY_AFTER_END),
    ("这阵子都挺忙的，不方便", DAY_AFTER_END),
])
def test_stated_busy_until_preserves_legitimate_ongoing_and_future_busy(text, expected):
    from ai.interaction import stated_busy_until

    assert stated_busy_until(text, NOW) == expected, f"mismatch for {text!r}"


def test_retrospective_busy_does_not_block_next_day_proactive_while_future_busy_does(violetdb, monkeypatch):
    from ai.profile_store import Profile

    profile = Profile(violetdb)
    clock = ["2026-11-24T14:00:00+08:00"]
    monkeypatch.setattr(violetdb, "now", lambda: clock[0])

    # Retrospective "这两天有点忙" -> stays at 3h default (17:00), does NOT lock out next day
    retro = profile.situations.observe(
        "availability", "在忙", "user_stated",
        evidence="这两天有点忙", user_message="不好意思这两天有点忙",
    )["situation"]
    assert retro["expires_at"] == "2026-11-24T17:00:00+08:00"

    clock[0] = "2026-11-25T10:00:00+08:00"
    assert profile.situations.current(["availability"]) == []

    # Ongoing/future "这两天都很忙别找我" -> extends to next day 23:59
    clock[0] = "2026-11-24T14:00:00+08:00"
    future = profile.situations.observe(
        "availability", "很忙别找我", "user_stated",
        evidence="这两天都很忙别找我", user_message="这两天都很忙别找我",
    )["situation"]
    assert future["expires_at"] == "2026-11-25T23:59:00+08:00"

    clock[0] = "2026-11-25T10:00:00+08:00"
    assert [s["value"] for s in profile.situations.current(["availability"])] == ["很忙别找我"]


# ---------------------------------------------------------------------------
# 4. #67 Sync-mode MemoryAgent call_context (source_message_ids & settled_forget)
# ---------------------------------------------------------------------------

def test_sync_pipeline_passes_call_context_records_source_and_redacts_on_forget(violetdb, monkeypatch):
    """Sync run_pipeline (defer_memory=False) must pass turn_id/user_message/assistant_reply/user_message_id."""
    from ai.agents import ChatAgent, MemoryAgent
    from ai.chat import run_pipeline
    from ai.memory_forget import forget_event_memory
    from tests.fakes import FakeLLM, completion, default_responders, install_fake_llm

    seen_memory_inputs: list[str] = []
    llm = install_fake_llm(monkeypatch, FakeLLM(default_responders("黑猫煤球听起来好可爱")))

    def fake_memory(kw):
        seen_memory_inputs.append(kw["messages"][-1]["content"])
        if any(m.get("role") == "tool" for m in kw["messages"]):
            return json.dumps({"should_save": True, "reason": "saved"})
        return completion(tool_calls=[{
            "name": "save_memory",
            "arguments": {
                "summary": "用户老家养了一只黑猫，名叫煤球",
                "category": "personal_info",
                "kind": "fact",
            },
        }])

    llm.responders["fake-memory"] = fake_memory
    user_msg_id = violetdb.add_user_message("text", "我老家养了一只黑猫，叫煤球")
    res = run_pipeline(
        "我老家养了一只黑猫，叫煤球",
        ChatAgent(),
        MemoryAgent(),
        defer_memory=False,
        turn_id="sync-cat-1",
        user_message_id=user_msg_id,
    )
    reply_msg_id = violetdb.add_assistant_message("text", res["chat"]["content"])

    assert any("Violet 实际回复：黑猫煤球听起来好可爱" in s for s in seen_memory_inputs)
    mem = violetdb.fetch_one("SELECT * FROM event_memories WHERE content LIKE '%煤球%'")
    assert mem is not None
    assert mem["source_turn_id"] == "sync-cat-1"
    assert json.loads(mem["source_message_ids"]) == [user_msg_id]

    report = forget_event_memory(violetdb, mem["id"], actor="before_reply", turn_id="sync-cat-2")
    assert report["messages_changed"] == [user_msg_id, reply_msg_id]
    assert "煤球" not in violetdb.fetch_one("SELECT content FROM messages WHERE id = ?", (user_msg_id,))["content"]
    assert "煤球" not in violetdb.fetch_one("SELECT content FROM messages WHERE id = ?", (reply_msg_id,))["content"]


@pytest.mark.parametrize("turn_id", ["sync-forget-t27", None])
def test_sync_pipeline_settled_forget_guard_blocks_silent_memory_agent_delete(violetdb, monkeypatch, turn_id):
    """When before_reply logs forget_unresolved (not_found), sync MemoryAgent must NOT delete behind Chat's back."""
    from ai.agents import ChatAgent, MemoryAgent
    from ai.chat import run_pipeline
    from tests.fakes import FakeLLM, completion, default_responders, install_fake_llm

    mem_id = violetdb.add_event_memory("用户和室友吵了一架", kind="event", progress="done")
    llm = install_fake_llm(monkeypatch, FakeLLM(default_responders("没找到具体哪一条，暂时还没删掉")))
    tool_results: list[dict] = []

    def fake_memory(kw):
        tools = [json.loads(m["content"]) for m in kw["messages"] if m.get("role") == "tool"]
        if tools:
            tool_results.extend(tools)
            return json.dumps({"should_save": False, "reason": "settled"})
        return completion(tool_calls=[{
            "name": "retract_memory",
            "arguments": {
                "memory_id": mem_id,
                "reason": "用户要求忘掉",
                "evidence": "把那件事忘掉吧",
                "forget": True,
            },
        }])

    llm.responders["fake-memory"] = fake_memory
    run_pipeline(
        "把那件事忘掉吧",
        ChatAgent(),
        MemoryAgent(),
        defer_memory=False,
        turn_id=turn_id,
    )
    assert any(r.get("action") == "already_settled" and r.get("status") == "not_found" for r in tool_results)
    assert violetdb.get_event_memory(mem_id)["status"] == "active"


def test_sync_runtime_records_source_via_inbound_and_redacts_on_forget(violetdb, monkeypatch):
    """End-to-end VioletRuntime with VIOLET_ASYNC_MEMORY=0: Turn 1 saves cat memory, Turn 2 forgets & redacts messages."""
    from ai.agents import ChatAgent, MemoryAgent
    from ai.memory_sources import memory_source
    from tests.fakes import FakeILinkClient, FakeLLM, completion, default_responders, install_fake_llm
    from violet_runtime import MessageSender, VioletRuntime

    monkeypatch.setenv("VIOLET_ASYNC_MEMORY", "0")
    monkeypatch.setenv("DEBOUNCE_NORMAL", "0.05")
    monkeypatch.setenv("DEBOUNCE_EXTRA", "0.1")
    llm = install_fake_llm(monkeypatch, FakeLLM(default_responders("黑猫煤球听起来很可爱")))

    def fake_memory(kw):
        if any(m.get("role") == "tool" for m in kw["messages"]):
            return json.dumps({"should_save": True, "reason": "done"})
        if "煤球" in kw["messages"][-1]["content"]:
            return completion(tool_calls=[{
                "name": "save_memory",
                "arguments": {
                    "summary": "用户老家养了一只黑猫，名叫煤球",
                    "category": "personal_info",
                    "kind": "fact",
                },
            }])
        return json.dumps({"should_save": False, "reason": "noop"})

    llm.responders["fake-memory"] = fake_memory
    client = FakeILinkClient()
    rt = VioletRuntime(
        client, violetdb, {"chat": ChatAgent(), "memory": MemoryAgent()},
        sender=MessageSender(client, violetdb, simulate_typing=False, sleep=lambda s: None),
    )
    assert rt.async_memory is False
    rt.start_worker()
    try:
        rt.ingest_text("我老家养了一只黑猫，叫煤球")
        assert rt.wait_idle()
        mem = violetdb.fetch_one("SELECT * FROM event_memories WHERE content LIKE '%煤球%'")
        assert mem is not None and mem["source_turn_id"]
        src = memory_source(violetdb, violetdb.get_event_memory(mem["id"]))
        assert src["state"] == "known" and len(src["message_ids"]) == 1

        llm.responders["fake-chat"] = lambda kw: "好，已经忘掉啦"
        rt.ingest_text("把我刚才说养猫的事忘掉吧")
        assert rt.wait_idle()
    finally:
        rt.stop(timeout=2)

    assert violetdb.get_event_memory(mem["id"]) is None
    user_row = violetdb.fetch_one("SELECT content FROM messages WHERE id = ?", (src["message_ids"][0],))
    assert "煤球" not in user_row["content"] and "黑猫" not in user_row["content"]


# ---------------------------------------------------------------------------
# 5. #67 Non-contiguous Chinese forget_target matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stored_content,forget_msg", [
    ("用户老家养了一只黑猫，名叫煤球", "把我刚才说养猫的事忘掉吧"),
    ("用户老家养了一只叫“煤球”的黑猫，平时特别粘人", "把我刚才说养猫的事忘掉吧"),
    ("用户老家养了一只黑猫，名叫煤球", "把家里养猫这件事忘了吧"),
    ("用户打算明年在城西买一套二手房", "把我买房的事忘掉吧"),
    ("用户上周生了一场重病", "把我说生病的事忘掉吧"),
    ("用户下午和室友打了一场篮球", "把我打球的事忘掉吧"),
])
def test_forget_target_matches_non_contiguous_chinese_phrases(violetdb, stored_content, forget_msg):
    from ai.memory_forget import forget_target

    target_id = violetdb.add_event_memory(stored_content, kind="fact")
    violetdb.add_event_memory("用户准备带煤球打疫苗", kind="event", progress="planned")
    violetdb.add_event_memory("用户下周三参加高数考试", kind="event", progress="planned")
    assert forget_target(violetdb, forget_msg) == {"status": "found", "memory_id": target_id}


def test_forget_target_non_contiguous_returns_ambiguous_when_multiple_match(violetdb):
    from ai.memory_forget import forget_target

    a = violetdb.add_event_memory("用户老家养了一只黑猫，名叫煤球", kind="fact")
    b = violetdb.add_event_memory("用户以前养过一只橘猫", kind="fact")
    res = forget_target(violetdb, "把我刚才说养猫的事忘掉吧")
    assert res["status"] == "ambiguous" and set(res["candidate_ids"]) == {a, b}


@pytest.mark.parametrize("memories,forget_msg", [
    # Generic requests with no topic -> never delete anything
    (["用户老家养了一只黑猫，名叫煤球", "用户下周有实习面试"], "那件事你就忘掉吧"),
    (["用户老家养了一只黑猫，名叫煤球", "用户刚买了新耳机"], "把我刚才说的事忘掉吧"),
    (["用户今天在食堂吃了黄焖鸡", "用户发誓要好好生活"], "把我今天发生的事忘掉吧"),
    (["用户前面聊到喜欢看科幻片"], "把前面聊到的内容删了吧"),
    # Only one character present -> never match
    (["用户周末去猫咖撸猫，拍了很多照片", "用户最近想养花"], "把我刚才说养猫的事忘掉吧"),
    # Cross-clause separation -> never match across comma/sentence boundary
    (["用户养了一只金毛犬，朋友家有只猫"], "把我刚才说养猫的事忘掉吧"),
    # Wrong character order (猫 before 养) -> never match
    (["用户给朋友的黑猫买了猫粮，顺便把花养好了"], "把我刚才说养猫的事忘掉吧"),
])
def test_forget_target_non_contiguous_negative_cases_never_false_match(violetdb, memories, forget_msg):
    from ai.memory_forget import forget_target

    for text in memories:
        violetdb.add_event_memory(text, kind="fact")
    assert forget_target(violetdb, forget_msg) == {"status": "not_found", "memory_id": None}


# ---------------------------------------------------------------------------
# 6. #67 Momentary-anger guard on proactive_chat=no
# ---------------------------------------------------------------------------

def test_proactive_chat_no_anger_outburst_guidance_in_schemas_and_rules():
    from ai import memory_tools_compact
    from ai.agents.memory_agent import PROFILE_RULES
    from ai.tool_schemas import SET_INTERACTION_PREFERENCE_SCHEMA

    full_desc = SET_INTERACTION_PREFERENCE_SCHEMA["function"]["description"]
    compact_desc = memory_tools_compact.compact(SET_INTERACTION_PREFERENCE_SCHEMA)["function"]["description"]
    for text in (full_desc, compact_desc, PROFILE_RULES):
        assert "proactive_chat" in text
        assert "气话" in text
        assert "别跟我说话了" in text and "烦死了" in text
