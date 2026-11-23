"""Phase 4: stable persona, Violet's virtual life, consistency guard (fictional data only).

No real model is involved: the life engine is template based by default and
every chat/proactive reply comes from the FakeLLM.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta

import pytest

from tests.test_phase2_scenarios import (  # noqa: F401  (make_world is a fixture)
    age_everything, arm, chat, decide, latest_event, make_world, say, script, tool,
)

TOKEN = os.environ["VIOLET_ADMIN_TOKEN"]
AUTH = {"Authorization": f"Bearer {TOKEN}"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def tz():
    from ai.settings import local_timezone
    return local_timezone()


def day(offset_days: int, hour: int = 22) -> datetime:
    """A local evening `offset_days` from today (negative = in the past)."""
    base = datetime.now(tz()).replace(hour=hour, minute=0, second=0, microsecond=0)
    return base + timedelta(days=offset_days)


def config(**overrides):
    from ai.persona import life_config
    cfg = copy.deepcopy(life_config())
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


@pytest.fixture()
def life(violetdb, monkeypatch):
    monkeypatch.setenv("VIOLET_LIFE_ENGINE", "1")
    from ai.life_engine import LifeEngine

    def build(**overrides):
        return LifeEngine(violetdb, config=config(**overrides))
    return build


@pytest.fixture()
def store(violetdb):
    from ai.life_store import LifeStore
    return LifeStore(violetdb)


@pytest.fixture()
def guard(violetdb):
    from ai.consistency import ConsistencyGuard
    return ConsistencyGuard(violetdb)


def happened(store, content, hours_ago=20, thread=None, plan_id=None, uid=None):
    occurred = datetime.fromisoformat(store.db.now()) - timedelta(hours=hours_ago)
    return store.insert_event(uid=uid or f"t:{content}:{hours_ago}", thread_id=thread, plan_id=plan_id,
                              event_type="progress", content=content, occurred_at=occurred.isoformat(),
                              duration_minutes=60, source="life_engine:template")


def new_thread(store, title="读《边城》", item="边城", days_ago=5, key=None):
    started = (datetime.fromisoformat(store.db.now()) - timedelta(days=days_ago)).isoformat()
    return store.create_thread(key or f"reading:{item}", "reading", title, item, started)


def system_text(call):
    return "\n".join(m.get("content") or "" for m in call["messages"] if m["role"] in ("system", "user"))


def last_normal_chat(w):
    from tests.test_phase2_scenarios import is_proactive_chat
    return next(c for c in reversed(w.llm.calls_for("fake-chat")) if not is_proactive_chat(c))


@pytest.fixture()
def api(tmp_path, violetdb):
    from wx_ilink_py import ILinkClient, create_app
    app = create_app(client=ILinkClient(data_dir=tmp_path / "data", auto_login=False), violetdb=violetdb,
                     proactive_scheduler=None, all_agents=[], admin_token=TOKEN)
    app.testing = True
    return app.test_client()


# ---------------------------------------------------------------------------
# 1. 30-day simulation
# ---------------------------------------------------------------------------

def test_30_day_simulation_is_sparse_ordinary_and_consistent(life, store, violetdb):
    engine = life()
    outcomes = []
    for offset in range(-30, 0):
        outcomes.append(engine.run_day(day(offset))["outcome"])
        assert len(store.threads(["active"])) <= 2                     # never more than 2 active threads
    events = store.events(("happened",))
    assert 3 <= len(events) <= 15
    # a weekly budget, never a daily diary
    by_week = {}
    for e in events:
        week = datetime.fromisoformat(e["occurred_at"]).isocalendar()[:2]
        by_week[week] = by_week.get(week, 0) + 1
    assert max(by_week.values()) <= 3
    days_with_events = {e["occurred_at"][:10] for e in events}
    assert len(days_with_events) == len(events)                        # at most one event per day
    quiet = [o for o in outcomes if o.startswith("quiet")]
    assert len(quiet) >= 15                                             # most days nothing happens
    # nothing in the future, nothing dramatic, nothing outside the persona
    now = datetime.now(tz())
    from ai.consistency import DRAMATIC_RE
    for e in events:
        assert datetime.fromisoformat(e["occurred_at"]) <= now
        assert not DRAMATIC_RE.search(e["content"]) and e["is_virtual"] == 1
        assert e["source"] == "life_engine:template"
    # every completed plan is backed by a recorded event of its own thread
    for plan in store.plans(["completed"]):
        event = store.event(plan["resolution_event_id"])
        assert event["status"] == "happened" and event["plan_id"] == plan["id"]
    # each day ran exactly once
    assert violetdb.fetch_one("SELECT COUNT(*) AS c FROM life_engine_runs")["c"] == 30
    assert all(r["outcome"] != "error" for r in violetdb.fetch_all("SELECT outcome FROM life_engine_runs"))


def test_weekly_budget_and_thread_cap_hold_even_when_every_day_wants_an_event(life, store):
    engine = life(daily_event_probability=1.0, new_thread_probability=1.0, weekly_max_events=3)
    for offset in range(-21, 0):
        engine.run_day(day(offset))
        assert len(store.threads(["active"])) <= 2
    by_week = {}
    for e in store.events(("happened",)):
        week = datetime.fromisoformat(e["occurred_at"]).isocalendar()[:2]
        by_week[week] = by_week.get(week, 0) + 1
    assert by_week and max(by_week.values()) == 3


def test_simulation_is_reproducible(life, store, violetdb):
    engine = life()
    for offset in range(-20, 0):
        engine.run_day(day(offset))
    first = [(e["occurred_at"], e["content"]) for e in store.events(("happened",))]
    for table in ("life_threads", "life_plans", "life_events", "life_revisions", "life_engine_runs",
                  "consistency_flags", "persona_dynamic"):
        violetdb.execute(f"DELETE FROM {table}")
    engine = life()
    for offset in range(-20, 0):
        engine.run_day(day(offset))
    assert [(e["occurred_at"], e["content"]) for e in store.events(("happened",))] == first


# ---------------------------------------------------------------------------
# 2. Several days with no events; no backfill after downtime
# ---------------------------------------------------------------------------

def test_days_without_events_are_normal_and_downtime_is_not_backfilled(life, store, violetdb):
    engine = life(daily_event_probability=0.0)
    for offset in range(-10, -5):
        assert engine.run_day(day(offset))["outcome"] == "quiet"
    assert store.events(("happened",)) == []

    engine = life(daily_event_probability=1.0, new_thread_probability=1.0)
    engine.run_day(day(-20))             # some activity long ago
    events_before = len(store.events(("happened",)))
    # the process was down for 10 days; on restart only "today" is considered
    engine.run_day(day(-1))
    runs = [r["run_key"] for r in violetdb.fetch_all("SELECT run_key FROM life_engine_runs ORDER BY id")]
    assert day(-2).date().isoformat() not in runs and day(-6).date().isoformat() in runs
    assert len(store.events(("happened",))) <= events_before + 1
    # not before the evening (the day is not over yet) and not claimed either
    assert engine.run_day(day(0, hour=10))["outcome"] == "too_early"
    assert day(0).date().isoformat() not in [r["run_key"] for r in violetdb.fetch_all("SELECT run_key FROM life_engine_runs")]


# ---------------------------------------------------------------------------
# 3. Thread progress, pause, end
# ---------------------------------------------------------------------------

def test_thread_progress_pause_resume_and_end(life, store):
    from ai.life_store import TransitionError
    engine = life(daily_event_probability=1.0, new_thread_probability=0.0, weekly_max_events=7)
    thread_id = new_thread(store, days_ago=30)
    engine.run_day(day(-6))
    assert store.thread(thread_id)["progress_steps"] == 1

    store.set_thread_status(thread_id, "paused", "最近没心情", "dashboard")
    out = engine.run_day(day(-5))
    assert out["outcome"] in ("event", "rejected", "quiet_no_thread") and out.get("thread_id") != thread_id
    assert all(e["thread_id"] != thread_id or e["occurred_at"] < day(-5).date().isoformat()
               for e in store.events(("happened",)))

    store.set_thread_status(thread_id, "active", "又想看了", "dashboard")
    store.set_thread_status(thread_id, "abandoned", "不想看了", "dashboard")
    with pytest.raises(TransitionError):
        store.set_thread_status(thread_id, "active", "x", "dashboard")       # terminal
    actions = [(r["action"], json.loads(r["new_value"])) for r in store.revisions("thread", thread_id)]
    assert actions[0][0] == "create" and [a[1] for a in actions[1:]] == ["paused", "active", "abandoned"]
    # the guard refuses events for a thread that is not active
    from ai.consistency import ConsistencyGuard
    v = ConsistencyGuard(store.db).check_event({"thread_id": thread_id, "content": "又读了几页《边城》",
                                                "occurred_at": day(-1).isoformat()}, store.db.now())
    assert "thread_not_active" in [x.check for x in v]


def test_thread_completes_only_through_a_recorded_event(life, store):
    engine = life(daily_event_probability=1.0, new_thread_probability=0.0, weekly_max_events=7,
                  plan_resolution={"complete_probability": 1.0})
    thread_id = new_thread(store, days_ago=40)
    for offset in range(-20, 0):
        engine.run_day(day(offset))
        if store.thread(thread_id)["status"] == "completed":
            break
    thread = store.thread(thread_id)
    assert thread["status"] == "completed"
    plan = store.plans(thread_id=thread_id)[0]
    finish = store.event(plan["resolution_event_id"])
    assert plan["status"] == "completed" and finish["event_type"] == "thread_completed" and "读完了《边城》" == finish["content"]
    assert finish["occurred_at"] >= plan["planned_for"][:10]


# ---------------------------------------------------------------------------
# 4. Restart does not duplicate
# ---------------------------------------------------------------------------

def test_restart_never_duplicates_a_day_or_an_event(life, store, violetdb):
    engine = life(daily_event_probability=1.0, new_thread_probability=1.0)
    first = engine.run_day(day(-3))
    assert first["outcome"] == "event"
    assert engine.run_day(day(-3))["outcome"] == "already_ran"
    assert life(daily_event_probability=1.0).run_day(day(-3, hour=23))["outcome"] == "already_ran"   # "restarted"
    assert len(store.events(("happened",))) == 1
    event = store.events(("happened",))[0]
    again = store.insert_event(uid=event["uid"], thread_id=event["thread_id"], plan_id=None, event_type="progress",
                               content=event["content"], occurred_at=event["occurred_at"], duration_minutes=60,
                               source="life_engine:template")
    assert again is None and len(store.events(("happened",))) == 1
    # a crash between claiming the day and finishing it does not re-run the day either
    assert violetdb.fetch_one("SELECT outcome FROM life_engine_runs WHERE run_key = ?", (day(-3).date().isoformat(),))["outcome"] == "event"


# ---------------------------------------------------------------------------
# 5. Time conflicts
# ---------------------------------------------------------------------------

def test_time_plausibility_checks(store, guard):
    thread_id = new_thread(store, days_ago=3)
    happened(store, "又往后读了几章《边城》", hours_ago=20, thread=thread_id)
    now = store.db.now()
    base = datetime.fromisoformat(now)

    def checks(**proposal):
        proposal = {"thread_id": thread_id, "content": "在《边城》里停在一段写得很慢的地方", "source": "life_engine:template",
                    **proposal}
        return [v.check for v in guard.check_event(proposal, now)]

    assert "time_conflict" in checks(occurred_at=(base - timedelta(hours=20, minutes=30)).isoformat())
    assert "future_event" in checks(occurred_at=(base + timedelta(hours=2)).isoformat())
    assert "before_thread_start" in checks(occurred_at=(base - timedelta(days=4)).isoformat())
    assert "out_of_order" in checks(occurred_at=(base - timedelta(hours=30)).isoformat())
    assert checks(occurred_at=(base - timedelta(hours=2)).isoformat()) == []


# ---------------------------------------------------------------------------
# 6. A plan that has not happened is never described as the past
# ---------------------------------------------------------------------------

def test_plans_are_not_past_events(life, store, guard):
    from ai.life_store import TransitionError
    thread_id = new_thread(store, days_ago=10)
    happened(store, "读了《边城》的一部分", hours_ago=48, thread=thread_id)
    planned_for = (datetime.fromisoformat(store.db.now()) - timedelta(days=2)).isoformat()
    plan_id = store.create_plan(thread_id, "这周想把《边城》读完", planned_for)

    # Violet claims she finished it: the progress event does not back that claim
    v = guard.check_reply("我昨天把《边城》读完了，结尾好安静")
    assert [x.check for x in v] == ["plan_described_as_done"]
    # even when the claim reuses the wording of a recorded progress event
    happened(store, "又往后读了几章《边城》", hours_ago=24, thread=thread_id)
    v = guard.check_reply("我又往后读了几章，把《边城》读完了")
    assert [x.check for x in v] == ["plan_described_as_done"]
    assert guard.check_reply("这周想把《边城》读完，不过还差一点") == []          # talking about the plan is fine
    assert guard.check_reply("前天读了《边城》的一部分，写得好慢") == []           # the recorded progress is fine
    assert guard.check_reply("我前天读了《边城》的一部分") == []

    # completing without an event is impossible; time passing does not complete it
    with pytest.raises(TransitionError):
        store.set_plan_status(plan_id, "completed", "time passed", "life_engine")
    life(daily_event_probability=0.0).run_day(day(-1))
    assert store.plan(plan_id)["status"] == "planned"
    # a long overdue plan is abandoned, never silently completed
    store.db.execute("UPDATE life_plans SET planned_for = ? WHERE id = ?",
                     ((datetime.fromisoformat(store.db.now()) - timedelta(days=20)).isoformat(), plan_id))
    life(daily_event_probability=0.0).run_day(day(0, hour=23))
    assert store.plan(plan_id)["status"] == "abandoned"
    assert guard.check_reply("我把《边城》读完了")[0].check == "unrecorded_experience"

    # an engine proposal that "finishes" something without a plan is rejected and flagged
    v = guard.check_event({"thread_id": thread_id, "content": "读完了《边城》", "event_type": "progress",
                           "occurred_at": (datetime.fromisoformat(store.db.now()) - timedelta(hours=1)).isoformat()},
                          store.db.now())
    assert "completion_without_plan" in [x.check for x in v]


def test_unrecorded_past_claims_are_caught(store, guard):
    happened(store, "反复听了一张老的爵士专辑", hours_ago=26)
    assert guard.check_reply("我昨天反复听了一张老的爵士专辑") == []
    found = guard.check_reply("昨天我去了海边，风好大\n你呢")
    assert [(x.check, x.line) for x in found] == [("unrecorded_experience", "昨天我去了海边，风好大")]
    # a hedge or "你" in ANOTHER clause does not excuse the claim
    for line in ("我昨天去了海边，你呢", "我昨天去了海边，下次带你去"):
        assert [x.check for x in guard.check_reply(line)] == ["unrecorded_experience"], line
    # opinions, plans, hedges and the user's own doings are not claims
    for line in ("我想去海边看看", "我觉得读了也挺好", "我没看完", "我打算明天去书店", "你昨天去了哪里", "我看了你发的照片"):
        assert guard.check_reply(line) == [], line


# ---------------------------------------------------------------------------
# 7. The persona core cannot be changed by the system
# ---------------------------------------------------------------------------

def test_persona_core_is_read_only(api, violetdb):
    from ai.persona import PersonaCore, PersonaDynamics, persona_dir
    path = persona_dir() / "persona_core.json"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    core = PersonaCore.load(violetdb)
    assert core.sha256 == before and violetdb.fetch_one("SELECT version FROM persona_core_versions")["version"] == core.version
    for method in ("put", "patch", "post", "delete"):
        r = getattr(api, method)("/api/persona/core", headers=AUTH, json={"personality": ["外向"]})
        assert r.status_code == 405 and "人工编辑" in r.get_json()["error"]
    # proposals touching the core never activate on their own
    result = PersonaDynamics(violetdb).propose("personality.extroversion", "opinion", "变得很外向", "用户喜欢外向的人",
                                            evidence_kind="conversation")
    assert result["action"] == "proposed" and result["entry"]["requires_review"] == 1
    result = PersonaDynamics(violetdb).propose("core_interest.basketball", "preference", "喜欢篮球", "经常听用户聊篮球",
                                            evidence_kind="conversation")
    assert result["action"] == "proposed"
    assert PersonaDynamics(violetdb).active() == []
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    # a human can approve via the dashboard -- that is the only way
    entry_id = result["entry"]["id"]
    r = api.post(f"/api/persona/dynamics/{entry_id}/decide", headers=AUTH, json={"approve": False})
    assert r.status_code == 200 and r.get_json()["status"] == "rejected"


# ---------------------------------------------------------------------------
# 8. Preference changes need a reason and evidence over time
# ---------------------------------------------------------------------------

def test_preference_change_requires_reason_and_supporting_events(store, violetdb):
    from ai.persona import PersonaDynamics
    dyn = PersonaDynamics(violetdb)
    assert dyn.propose("taste.music", "preference", "更喜欢钢琴曲", "")["action"] == "rejected"         # no reason
    one = happened(store, "反复听了几首钢琴曲", hours_ago=24)
    r = dyn.propose("taste.music", "preference", "更喜欢钢琴曲", "最近常听", related_event_ids=[one])
    assert r["action"] == "proposed" and "not enough" in r["reason"]                                    # one event, no span
    two = happened(store, "把几首钢琴曲从头听了一遍", hours_ago=24 * 9)
    r = dyn.propose("taste.music", "preference", "更喜欢钢琴曲", "这一周多常听钢琴曲", related_event_ids=[one, two])
    assert r["action"] == "activated"
    entry = r["entry"]
    assert (entry["reason"], json.loads(entry["related_event_ids"]), entry["previous_value"]) == ("这一周多常听钢琴曲", [one, two], None)
    # the next change keeps the old value, and is rate limited within a week
    r = dyn.propose("taste.music", "preference", "又喜欢爵士了", "听了爵士", related_event_ids=[one, two])
    assert r["action"] == "proposed" and r["reason"] == "preference changes are rate limited"
    assert r["entry"]["previous_value"] == "更喜欢钢琴曲"
    revisions = store.revisions("persona_dynamic", entry["id"])
    assert revisions[0]["action"] == "activate" and revisions[0]["reason"] == "这一周多常听钢琴曲"
    # temporary interests expire on their own
    t = dyn.propose("interest.reading.recent", "temporary_interest", "这阵子对看书兴致比较高", "刚读完一本", related_event_ids=[one])
    assert t["action"] == "activated" and t["entry"]["valid_until"]
    violetdb.execute("UPDATE persona_dynamic SET valid_until = ? WHERE id = ?",
                  ((datetime.fromisoformat(violetdb.now()) - timedelta(minutes=1)).isoformat(), t["entry"]["id"]))
    assert all(x["id"] != t["entry"]["id"] for x in dyn.active())
    assert dyn.get(t["entry"]["id"])["status"] == "expired"


# ---------------------------------------------------------------------------
# 9. Violet does not change just because the user likes something
# ---------------------------------------------------------------------------

def test_no_pandering_to_the_users_interests(life, store, violetdb):
    from ai.persona import PersonaDynamics
    evidence = happened(store, "反复听了几首钢琴曲", hours_ago=30)
    r = PersonaDynamics(violetdb).propose("taste.sport", "opinion", "也开始喜欢篮球", "用户很喜欢篮球",
                                       related_event_ids=[evidence], evidence_kind="user_preference")
    assert r["action"] == "rejected" and "not a reason" in r["reason"]
    # the user talks about their own hobby a lot; Violet's life is generated from her own interests only
    from ai.profile_store import Profile
    Profile(violetdb).facts.remember("interest", "很喜欢打篮球", evidence="我超爱打篮球", user_message="我超爱打篮球")
    engine = life(daily_event_probability=1.0, new_thread_probability=1.0)
    for offset in range(-15, 0):
        engine.run_day(day(offset))
    contents = " ".join(e["content"] for e in store.events(statuses=None))
    assert "篮球" not in contents
    assert {t["interest_key"] for t in store.threads()} <= {"reading", "music", "notes"}


def test_user_experience_is_not_recorded_as_eves(store, guard, violetdb):
    from ai.profile_store import Profile
    r = Profile(violetdb).facts.remember("interest", "最近在读《边城》", evidence="我最近在读边城", user_message="我最近在读边城")
    assert r["fact"]["status"] == "active"
    thread_id = new_thread(store, days_ago=3)
    v = guard.check_event({"thread_id": thread_id, "content": "最近在读《边城》",
                           "occurred_at": (datetime.fromisoformat(violetdb.now()) - timedelta(hours=1)).isoformat()}, violetdb.now())
    assert "user_experience" in [x.check for x in v]


# ---------------------------------------------------------------------------
# 10. The user asks what Violet has been doing
# ---------------------------------------------------------------------------

def test_user_asks_about_eves_recent_life(make_world):
    from ai.life_store import LifeStore
    w = make_world()
    w.llm.responders["fake-memory"] = lambda kw: json.dumps({"should_save": False, "reason": "t"})
    # nothing recorded yet: Violet may talk about thoughts, not invent a past
    say(w, "你最近在干嘛")
    text = system_text(last_normal_chat(w))
    assert "Violet 的稳定人设" in text and "最近没有记录的生活事件" in text

    store = LifeStore(w.db)
    thread_id = new_thread(store, days_ago=4)
    happened(store, "又往后读了几章《边城》", hours_ago=20, thread=thread_id)
    store.create_plan(thread_id, "这周想把《边城》读完", (datetime.fromisoformat(w.db.now()) + timedelta(days=2)).isoformat())
    chat(w, normal="昨天又往后读了几章《边城》\n写得好慢，但挺舒服的")
    say(w, "你最近在干嘛")
    text = system_text(last_normal_chat(w))
    assert "又往后读了几章《边城》（已发生）" in text and "正在进行：读《边城》" in text
    assert "这周想把《边城》读完（计划，还没做，不能说成已经做了）" in text
    assert w.client.sent_texts[-2:] == ["昨天又往后读了几章《边城》", "写得好慢，但挺舒服的"]
    event = store.events(("happened",))[0]
    assert event["share_count"] == 1 and event["shared_with_user_at"]          # told_to_user


def test_invented_experience_in_a_reply_is_regenerated_or_dropped(make_world):
    from ai.life_store import LifeStore
    w = make_world()
    w.llm.responders["fake-memory"] = lambda kw: json.dumps({"should_save": False, "reason": "t"})

    def reply(kw):
        if "里面有和已记录的事实对不上的说法" in system_text(kw):
            return "也没干嘛\n这两天在想要不要重新开始记笔记"
        return "我昨天去了海边\n风好大"
    w.llm.responders["fake-chat"] = reply
    say(w, "你最近在干嘛")
    assert w.client.sent_texts[-2:] == ["也没干嘛", "这两天在想要不要重新开始记笔记"]
    flags = LifeStore(w.db).flags()
    assert [f["check_name"] for f in flags] == ["unrecorded_experience"] and "海边" in flags[0]["detail"]

    # the model insists: only the contradicting line is dropped
    w.llm.responders["fake-chat"] = lambda kw: "我昨天去了海边\n你今天累不累"
    say(w, "你呢")
    assert w.client.sent_texts[-1] == "你今天累不累" and "我昨天去了海边" not in w.client.sent_texts


# ---------------------------------------------------------------------------
# 11-13. Proactive sharing: uninterested user, no repeats, busy user
# ---------------------------------------------------------------------------

def share_world(make_world):
    from ai.life_store import LifeStore
    w = make_world()
    w.llm.responders["fake-memory"] = lambda kw: json.dumps({"should_save": False, "reason": "t"})
    say(w, "早")
    age_everything(w.db, 3)
    store = LifeStore(w.db)
    thread_id = new_thread(store, days_ago=4)
    return w, store, thread_id


def test_life_events_are_only_candidates_and_never_shared_twice(make_world):
    w, store, thread_id = share_world(make_world)
    event_id = happened(store, "又往后读了几章《边城》", hours_ago=20, thread=thread_id)
    state = w.scheduler._conversation_state()
    assert [e["id"] for e in state["allowed_life_events"]] == [event_id]

    # the model may decide not to share at all: that is normal ([NO_REPLY])
    chat(w, proactive="[NO_REPLY]")
    arm(w)
    w.scheduler.tick()
    assert latest_event(w)["status"] == "skipped" and store.event(event_id)["share_count"] == 0

    chat(w, proactive="昨天又往后读了几章《边城》\n写得好慢")
    arm(w)
    w.scheduler.tick()
    assert w.client.sent_texts[-2:] == ["昨天又往后读了几章《边城》", "写得好慢"]
    assert store.event(event_id)["share_count"] == 1
    assert w.scheduler._conversation_state()["allowed_life_events"] == []

    # once shared, the event is no longer offered as an allowed candidate and share_count does not increase again
    say(w, "哈哈好呀")
    age_everything(w.db, 4)
    chat(w, proactive="今天天气挺舒服的")
    arm(w)
    w.scheduler.tick()
    last_prompt = "\n".join(str(m.get("content", "")) for m in w.llm.calls_for("fake-chat")[-1]["messages"])
    assert "你最近的生活经历" not in last_prompt
    assert store.event(event_id)["share_count"] == 1
    assert w.client.sent_texts.count("昨天又往后读了几章《边城》") == 1


def test_user_uninterested_in_eves_life_stops_life_shares(make_world):
    w, store, thread_id = share_world(make_world)
    for i, content in enumerate(["又往后读了几章《边城》", "在《边城》里停在一段写得很慢的地方"]):
        event_id = happened(store, content, hours_ago=10 - i * 2, thread=thread_id, uid=f"u{i}")
        chat(w, proactive=content)
        arm(w)
        w.scheduler.tick()
        assert w.client.sent_texts[-1] == content
        say(w, "嗯")                                      # brush-off
        age_everything(w.db, 4)
        w.db.set_proactive_setting("last_engaged_user_turn_at", w.db.now())   # isolate from the cold streak rule
    third = happened(store, "反复听了一张老的爵士专辑", hours_ago=1, uid="u3")
    assert w.scheduler.tracker.topic_state("violet_life")["unengaged"] == 2
    assert w.scheduler._conversation_state()["allowed_life_events"] == []
    # the user later asks about Violet's life themselves -> life topics are fine again
    chat(w, normal="昨天反复听了一张老的爵士专辑")
    say(w, "你最近在听什么")
    assert w.scheduler.tracker.topic_state("violet_life")["unengaged"] == 0
    assert store.event(third)["share_count"] == 1


def test_busy_user_is_not_disturbed_even_with_new_life_events(make_world):
    from ai.profile_store import Profile
    w, store, thread_id = share_world(make_world)
    happened(store, "又往后读了几章《边城》", hours_ago=2, thread=thread_id)
    Profile(w.db).situations.observe("availability", "在开会", "user_stated", evidence="在开会", user_message="我在开会", until="2h")
    calls = len(w.llm.calls_for("fake-chat"))
    arm(w)
    w.scheduler.tick()
    assert len(w.llm.calls_for("fake-chat")) == calls
    assert w.scheduler.status_snapshot()["due_reason"].startswith("the user said they are busy")


# ---------------------------------------------------------------------------
# 14. Violet's life never becomes the user's memory or profile
# ---------------------------------------------------------------------------

def test_eves_life_never_enters_user_memory_or_profile(make_world):
    from ai.life_store import LifeStore
    w = make_world()
    store = LifeStore(w.db)
    happened(store, "又往后读了几章《边城》", hours_ago=20, thread=new_thread(store))
    w.llm.responders["fake-memory"] = script(
        tool("save_memory", summary="又往后读了几章《边城》", category="event", kind="event"),
        tool("remember_profile_fact", key="interest", value="又往后读了几章《边城》", evidence="嗯嗯"))
    chat(w, normal="昨天又往后读了几章《边城》")
    say(w, "嗯嗯")
    assert w.db.fetch_all("SELECT * FROM event_memories") == []
    assert w.db.fetch_all("SELECT * FROM user_profile_facts") == []
    # but when the USER says they read it, it is the user's memory
    w.llm.responders["fake-memory"] = script(tool("save_memory", summary="用户也在读《边城》", category="event", kind="event"))
    say(w, "我也在读边城")
    assert [m["content"] for m in w.db.fetch_all("SELECT content FROM event_memories")] == ["用户也在读《边城》"]


# ---------------------------------------------------------------------------
# 15. External information is never first-hand experience
# ---------------------------------------------------------------------------

def test_external_information_is_not_eves_experience(store, guard, violetdb):
    when = (datetime.fromisoformat(violetdb.now()) - timedelta(hours=3)).isoformat()
    for proposal in ({"content": "看到一篇讲爵士乐历史的文章", "occurred_at": when, "source": "web:search"},
                     {"content": "一个乐队发布了新专辑", "occurred_at": when, "source": "life_engine:x",
                      "source_detail": {"origin": "news"}}):
        assert "external_as_experience" in [v.check for v in guard.check_event(proposal, violetdb.now())]


class WebProposer:
    """A future Web-MCP-backed proposer that (wrongly) turns news into Violet's life."""

    def new_thread(self, interest_key, rng, used_items):
        return "听一位独立乐队的新专辑", "一位独立乐队的新专辑"

    def progress(self, thread, rng, exclude):
        return "我和男朋友去了那个乐队的演唱会"

    def finish(self, thread):
        return "去现场看了巡演"

    def plan(self, thread):
        return "去看巡演"

    def steps_to_finish(self, thread):
        return 2


def test_model_or_plugin_output_that_contradicts_the_persona_is_rejected(life, store, violetdb):
    from ai.life_engine import LifeEngine
    from ai.persona import life_config
    cfg = config(daily_event_probability=1.0, new_thread_probability=0.0)
    engine = LifeEngine(violetdb, config=cfg, proposer=WebProposer())
    thread_id = store.create_thread("music:x", "music", "听一位独立乐队的新专辑", "一位独立乐队的新专辑",
                                    (datetime.fromisoformat(violetdb.now()) - timedelta(days=5)).isoformat())
    out = engine.run_day(day(-1))
    assert out["outcome"] == "rejected" and "forbidden_backstory" in out["violations"]
    assert store.events(("happened",), thread_id=thread_id) == []
    assert "forbidden_backstory" in [f["check_name"] for f in store.flags()]
    assert life_config() == config()              # nothing rewrote the configuration


def test_reply_contradicting_the_persona_is_not_sent(make_world):
    w = make_world()
    w.llm.responders["fake-memory"] = lambda kw: json.dumps({"should_save": False, "reason": "t"})
    w.llm.responders["fake-chat"] = lambda kw: "我妈妈今天又念叨我了\n我是真人啦"
    say(w, "你家里人呢")
    assert not any("妈妈" in t or "真人" in t for t in w.client.sent_texts)
    checks = {f["check_name"] for f in w.db.fetch_all("SELECT check_name FROM consistency_flags")}
    assert {"forbidden_backstory", "identity_conflict"} <= checks
    assert w.client.sent_texts, "a fallback is sent instead of silence"


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def test_dashboard_views_and_controls(api, store, violetdb):
    thread_id = new_thread(store)
    event_id = happened(store, "又往后读了几章《边城》", thread=thread_id)
    assert api.get("/api/life").status_code == 401                                        # same auth as everything else
    data = api.get("/api/life", headers=AUTH).get_json()
    assert [t["id"] for t in data["threads"]] == [thread_id] and data["events"][0]["is_virtual"] == 1
    assert api.get("/api/persona", headers=AUTH).get_json()["core"]["data"]["identity"]["name"] == "Violet"
    page = api.get("/persona", headers=AUTH)                 # stage 3: a page of the Control Center now
    assert page.status_code == 302 and page.headers["Location"].endswith("/control#persona")

    assert api.post(f"/api/life/threads/{thread_id}/pause", headers=AUTH, json={}).get_json()["status"] == "paused"
    assert api.post(f"/api/life/threads/{thread_id}/pause", headers=AUTH, json={}).status_code == 400
    assert api.post(f"/api/life/threads/{thread_id}/resume", headers=AUTH, json={}).get_json()["status"] == "active"

    r = api.patch(f"/api/life/events/{event_id}", headers=AUTH, json={"content": "又读了几页《边城》", "reason": "写夸张了"})
    assert r.status_code == 200 and r.get_json()["content"] == "又读了几页《边城》"
    r = api.patch(f"/api/life/events/{event_id}", headers=AUTH, json={"content": "和我男朋友一起读", "reason": "x"})
    assert r.status_code == 400 and "forbidden_backstory" in r.get_json()["error"]
    history = api.get(f"/api/life/event/{event_id}/history", headers=AUTH).get_json()["revisions"]
    assert [h["action"] for h in history] == ["create", "correct"] and history[1]["reason"] == "写夸张了"
    assert json.loads(history[1]["old_value"]) == "又往后读了几章《边城》"
    assert api.post(f"/api/life/events/{event_id}/retract", headers=AUTH, json={}).status_code == 400   # reason required
    assert api.post(f"/api/life/events/{event_id}/retract", headers=AUTH, json={"reason": "没发生过"}).get_json()["status"] == "retracted"
    assert store.recent_happened(days=3) == []                    # retracted events are never cited


# ---------------------------------------------------------------------------
# Migration and rollback
# ---------------------------------------------------------------------------

def test_phase4_migration_and_rollback_scripts(tmp_path, violetdb):
    from pathlib import Path
    migrations = Path(__file__).resolve().parent.parent / "migrations"
    db_path = tmp_path / "copy.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT)")          # a phase-3 table
    conn.execute("INSERT INTO messages (content) VALUES ('你好')")
    for _ in range(2):                                                                      # idempotent
        conn.executescript((migrations / "phase4_up.sql").read_text(encoding="utf-8"))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    expected = {"persona_core_versions", "persona_dynamic", "life_threads", "life_plans", "life_events",
                "life_revisions", "life_engine_runs", "consistency_flags"}
    assert expected <= tables
    conn.executescript((migrations / "phase4_down.sql").read_text(encoding="utf-8"))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert not (expected & tables) and conn.execute("SELECT content FROM messages").fetchone()[0] == "你好"
    conn.close()
    # the live schema contains exactly the same tables as the migration script
    live = {r["name"] for r in violetdb.fetch_all("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert expected <= live


# ---------------------------------------------------------------------------
# Multi-turn samples for future real-model evaluation (format check only)
# ---------------------------------------------------------------------------

def test_chat_samples_are_well_formed():
    from pathlib import Path
    path = Path(__file__).resolve().parent / "samples" / "phase4_chat_samples.json"
    from tools.legacy import upgrade
    samples = upgrade(json.loads(path.read_text(encoding="utf-8")))      # recorded before the rename to Violet
    assert len(samples["samples"]) >= 8
    for sample in samples["samples"]:
        assert sample["id"] and sample["setup"] is not None and sample["turns"] and sample["expect"]
        assert all(t["role"] in ("user", "violet") for t in sample["turns"])


# ---------------------------------------------------------------------------
# DailyLife persistence, doing/paraphrase matching, weekend classes, onboarding call
# ---------------------------------------------------------------------------

def test_daily_life_persists_happened_events_when_warmth_drops(tmp_path, violetdb, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    from datetime import date, datetime, timedelta, timezone
    from ai import persona_templates as pt
    from ai.daily_life import DailyLife, SOURCE
    from ai.profile_store import Profile

    monkeypatch.setenv("VIOLET_DAILY_LIFE", "1")
    pt.switch(violetdb, "violet", "keep")
    tz_cn = timezone(timedelta(hours=8))
    today = date(2026, 11, 19)
    now = datetime(2026, 11, 19, 21, 30, tzinfo=tz_cn)
    monkeypatch.setattr(violetdb, "now", lambda: now.isoformat())

    # High warmth (Stage 2 -> 3 items, no quiet day): materialize today's 3 items
    pt.set_stage(None, 2)
    dl_warm = DailyLife(violetdb)
    warm_view = dl_warm.day(today, now)
    assert len(warm_view["done"]) == 3 and not warm_view["quiet"]
    warm_ids = {int(e["id"]) for e in warm_view["done"]}

    # Warmth drops to 0 (explicit distance) and quiet_day_probability=1.0 makes plan_for() return []
    Profile(violetdb).preferences.set_explicit(
        "closeness", "distance", evidence="保持点距离", user_message="我们保持点距离"
    )
    dl_cool = DailyLife(violetdb)
    dl_cool.daily = {**dl_cool.daily, "quiet_day_probability": 1.0}
    assert dl_cool.plan_for(today, tz_cn) == []
    cool_view = dl_cool.day(today, now)
    assert not cool_view["quiet"]
    assert {int(e["id"]) for e in cool_view["done"]} == warm_ids
    assert all(e["source"] == SOURCE and e.get("slot_label") and e.get("doing") for e in cool_view["done"])


def test_daily_life_mid_doing_window_warmth_shift_preserves_stored_doing_and_done(tmp_path, violetdb, monkeypatch):
    """When warmth_level shifts mid-doing-window so plan_for() rolls a different item/time for the same slot,
    DailyLife.day() must derive view['doing'] (and later view['done']) strictly from the already-stored DB row."""
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    from datetime import date, datetime, timedelta, timezone
    from ai import persona_templates as pt
    from ai.daily_life import DailyLife

    monkeypatch.setenv("VIOLET_DAILY_LIFE", "1")
    pt.switch(violetdb, "stepsister", "keep")
    tz_cn = timezone(timedelta(hours=8))
    today = date(2026, 11, 19)
    start_at = datetime(2026, 11, 19, 19, 0, tzinfo=tz_cn)
    mid_now = datetime(2026, 11, 19, 19, 20, tzinfo=tz_cn)
    after_now = datetime(2026, 11, 19, 20, 0, tzinfo=tz_cn)
    monkeypatch.setattr(violetdb, "now", lambda: mid_now.isoformat())

    original_item = {
        "slot": "evening", "label": "晚上",
        "done": "煲了一锅玉米排骨汤，厨房里全是香味", "doing": "在厨房看汤",
        "start": start_at, "minutes": 45, "uid": "daily:2026-11-19:evening",
    }
    rerolled_item = {
        "slot": "evening", "label": "晚上",
        "done": "晚上看了一集恋爱剧，剧情很假", "doing": "在看一部恋爱剧",
        "start": datetime(2026, 11, 19, 20, 30, tzinfo=tz_cn), "minutes": 45, "uid": "daily:2026-11-19:evening",
    }

    dl = DailyLife(violetdb)
    # 1. Materialize under initial warmth (mid-doing-window at 19:20)
    monkeypatch.setattr(dl, "plan_for", lambda d, tz: [original_item])
    v1 = dl.day(today, mid_now)
    assert v1["doing"] is not None
    assert v1["doing"]["doing"] == "在厨房看汤"
    assert v1["doing"]["done"] == "煲了一锅玉米排骨汤，厨房里全是香味"
    assert v1["doing"]["label"] == "晚上"
    assert v1["doing"]["start"] == start_at
    stored_id = v1["doing"]["event"]["id"]

    # 2. Warmth shifts mid-doing-window -> plan_for() returns a completely different item & start time for same uid
    monkeypatch.setattr(dl, "plan_for", lambda d, tz: [rerolled_item])
    v2 = dl.day(today, mid_now)
    assert v2["doing"] is not None
    assert v2["doing"]["event"]["id"] == stored_id
    assert v2["doing"]["doing"] == "在厨房看汤"
    assert v2["doing"]["done"] == "煲了一锅玉米排骨汤，厨房里全是香味"
    assert v2["doing"]["label"] == "晚上"
    assert v2["doing"]["start"] == start_at

    # 2b. Even for a legacy DB row lacking doing/label in source_detail, never leak rerolled item's doing
    violetdb.execute("UPDATE life_events SET source_detail = ? WHERE id = ?", ('{"slot": "evening"}', stored_id))
    v2_legacy = dl.day(today, mid_now)
    assert v2_legacy["doing"] is not None
    assert v2_legacy["doing"]["doing"] == "在厨房看汤"
    assert v2_legacy["doing"]["done"] == "煲了一锅玉米排骨汤，厨房里全是香味"
    assert v2_legacy["doing"]["label"] == "晚上"

    # 3. Once the original 45m window ends (20:00), view['done'] STILL carries original doing/done (not rerolled)
    monkeypatch.setattr(violetdb, "now", lambda: after_now.isoformat())
    v3 = dl.day(today, after_now)
    assert v3["doing"] is None
    assert len(v3["done"]) == 1
    assert v3["done"][0]["id"] == stored_id
    assert v3["done"][0]["content"] == "煲了一锅玉米排骨汤，厨房里全是香味"
    assert v3["done"][0]["doing"] == "在厨房看汤"
    assert v3["done"][0]["slot_label"] == "晚上"


def test_mentions_life_matches_doing_and_natural_paraphrases():
    from ai.daily_life import mentions_life

    # Natural Gemini paraphrases & doing-vs-done wording that previously missed under strict 0.25 bigram ratio
    assert mentions_life(
        "下午削苹果差点把皮削断，结果屏住气居然一整条连下来了，本来还想顺手显摆一下的",
        "切苹果的时候差点削断皮，最后居然连着削完了",
    )
    assert mentions_life("刚还在厨房看排骨汤呢", "煲了一锅玉米排骨汤，厨房里全是香味")
    assert mentions_life("开着小夜灯在听慢歌", "关掉大灯只留了盏小夜灯，听了几首慢歌")
    assert mentions_life("今晚其实挺闲的，刚听完那几首慢歌就发着呆", "关掉大灯只留了盏小夜灯，听了几首慢歌")
    assert mentions_life("早上自己冲咖啡手一抖水倒太多，喝起来淡得跟白开水似的", "手冲了一杯咖啡，不小心倒多了水")
    assert mentions_life("今天睡懒觉起晚了，慢悠悠啃完早饭才清醒", "起得有点晚，慢吞吞吃了早饭")

    # Unrelated messages must stay False (no over-matching)
    assert not mentions_life("你吃早饭了吗", "起得有点晚，慢吞吞吃了早饭")
    assert not mentions_life("今天天气真不错", "切苹果的时候差点削断皮，最后居然连着削完了")
    assert not mentions_life("我刚在厨房煎了个牛排", "煲了一锅玉米排骨汤，厨房里全是香味")
    assert not mentions_life("我刚在厨房煎了个牛排", "煎鸡蛋的时候翻面翻碎了一个，自己吃掉了")
    assert not mentions_life("我在宿舍楼下散步", "在公园散了会儿步，吹了吹晚风")


def test_daily_life_skips_school_classes_on_weekends(tmp_path, violetdb, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    from datetime import date, timedelta, timezone
    from ai import persona_templates as pt
    from ai.daily_life import DailyLife

    monkeypatch.setenv("VIOLET_DAILY_LIFE", "1")
    pt.switch(violetdb, "stepsister", "keep")
    pt.set_stage(None, 2)  # force 3 items/day, no quiet days
    dl = DailyLife(violetdb)
    tz_cn = timezone(timedelta(hours=8))

    # Sample 10 consecutive weekends (20 Sat/Sun days): none may pick 早课 / 高数 / 上完课
    start = date(2026, 10, 10)  # Saturday
    assert start.weekday() == 5
    for w in range(10):
        for offset in (0, 1):
            d = start + timedelta(days=w * 7 + offset)
            plan = dl.plan_for(d, tz_cn)
            assert len(plan) == 3
            for item in plan:
                text = f"{item['done']} {item['doing']}"
                assert not any(tok in text for tok in ("早课", "高数", "上完课")), (d, item)


def test_onboarding_skips_address_name_when_persona_defines_relation_call(tmp_path, monkeypatch, violetdb):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    from ai import persona_templates as pt
    from ai.onboarding import OnboardingPlanner
    from ai.profile_store import Profile

    profile = Profile(violetdb)
    planner = OnboardingPlanner(violetdb, profile.facts)

    # Default violet has no relation.call -> asks address_name first
    pt.switch(violetdb, "violet", "keep")
    assert planner.next_key().key == "address_name"

    # stepsister (relation.call = 哥/姐) & junior (前辈) -> skip address_name
    pt.switch(violetdb, "stepsister", "keep")
    assert planner.next_key().key != "address_name"
    pt.switch(violetdb, "junior", "keep")
    assert planner.next_key().key != "address_name"

    # childhood (relation.call = {}) -> still asks address_name
    pt.switch(violetdb, "childhood", "keep")
    assert planner.next_key().key == "address_name"
