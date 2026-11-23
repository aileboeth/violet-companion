"""Regression tests for ConsistencyGuard false positives and backstory guards (v0.7.78+ fixes)."""

from __future__ import annotations

from datetime import timedelta

from ai.consistency import ConsistencyGuard
from ai.life_store import LifeStore
from ai.memory_model import parse_time
from ai.persona import PersonaCore


def _insert_daily(store: LifeStore, content: str, *, hours_ago: int = 1, doing: str | None = None, uid: str = "daily:t:1"):
    occurred = (parse_time(store.db.now()) - timedelta(hours=hours_ago)).isoformat()
    detail = {"slot": "evening"}
    if doing is not None:
        detail["doing"] = doing
    return store.insert_event(
        uid=uid, thread_id=None, plan_id=None, event_type="daily", content=content,
        occurred_at=occurred, duration_minutes=45, source="life_daily", source_detail=detail, actor="life_daily",
    )


def test_completion_re_only_enforced_for_ongoing_threads_not_life_daily(violetdb):
    store = LifeStore(violetdb)
    _insert_daily(store, "关掉大灯只留了盏小夜灯，听了几首慢歌", hours_ago=1, doing="开着小夜灯听慢歌", uid="daily:t:song")

    # Completed single-shot life_daily event (thread_id IS NULL) backs "刚听完那几首慢歌..."
    guard = ConsistencyGuard(violetdb)
    assert guard.check_reply("今晚其实挺闲的，刚听完那几首慢歌就发着呆，也没做什么特别的事") == []

    # Ongoing narrative thread event (thread_id IS NOT NULL) STILL blocks claiming completion early
    started = (parse_time(violetdb.now()) - timedelta(days=5)).isoformat()
    tid = store.create_thread("reading:边城", "reading", "读《边城》", "边城", started)
    occurred = (parse_time(violetdb.now()) - timedelta(hours=12)).isoformat()
    store.insert_event(
        uid="engine:t:bc", thread_id=tid, plan_id=None, event_type="progress",
        content="读了《边城》的一部分", occurred_at=occurred, duration_minutes=60, source="life_engine:template",
    )
    planned_for = (parse_time(violetdb.now()) + timedelta(days=2)).isoformat()
    store.create_plan(tid, "这周想把《边城》读完", planned_for)
    assert [v.check for v in guard.check_reply("我昨天把《边城》读完了，结尾好安静")] == ["plan_described_as_done"]


def test_consistency_guard_backs_daily_doing_phrases_and_clauses_without_false_passes(tmp_path, monkeypatch, violetdb):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    from ai import persona_templates as pt

    pt.switch(violetdb, "stepsister", "keep")
    store = LifeStore(violetdb)
    # Test without storing `doing` in source_detail too (verifies automatic pool lookup)
    _insert_daily(store, "煲了一锅玉米排骨汤，厨房里全是香味", hours_ago=1, doing=None, uid="daily:t:soup")

    guard = ConsistencyGuard(violetdb, PersonaCore.load())
    assert guard.check_reply("我刚在厨房看了会儿汤") == []
    assert guard.check_reply("我刚才在厨房看了会儿排骨汤") == []
    assert guard.check_reply("我刚做了点排骨汤") == []

    # Unrelated claims in the same room or elsewhere must still be flagged
    assert [v.check for v in guard.check_reply("我刚在厨房做了个牛排")] == ["unrecorded_experience"]
    assert [v.check for v in guard.check_reply("我刚在厨房看了本书")] == ["unrecorded_experience"]
    assert [v.check for v in guard.check_reply("我昨天去了游乐园")] == ["unrecorded_experience"]


def test_forbidden_backstory_catches_going_to_work_across_templates(tmp_path, monkeypatch, violetdb):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    from ai import persona_templates as pt

    # Fresh default persona_core.json
    guard_default = ConsistencyGuard(violetdb, PersonaCore.load())
    for bad in ("我今天去上班了", "我去上班了", "我明天要去加班"):
        assert "forbidden_backstory" in {v.check for v in guard_default.check_reply(bad)}, bad
    for ok in ("你今天去上班了吗", "我不上班"):
        assert "forbidden_backstory" not in {v.check for v in guard_default.check_reply(ok)}, ok

    # Switched violet & stepsister templates
    for key in ("violet", "stepsister"):
        pt.switch(violetdb, key, "keep")
        guard = ConsistencyGuard(violetdb, PersonaCore.load())
        for bad in ("我今天去上班了", "我去上班了", "我今天要去上班"):
            assert "forbidden_backstory" in {v.check for v in guard.check_reply(bad)}, (key, bad)
        for ok in ("你今天去上班了吗", "我不上班"):
            assert "forbidden_backstory" not in {v.check for v in guard.check_reply(ok)}, (key, ok)
