"""Phase 9.1: eight weeks of chat through the real runtime (offline).

Fictional user, isolated DB, fake WeChat, fake SDK, scenario clock, nightly decay, and the real Control Center routes for
the dashboard edits. The fake memory model makes the tool calls a careful model would make; the fake Intent model
NEVER calls query_memory (the worst case of R-M2). What is checked is what really reaches the Chat request -- the
memory lines, their state labels and what is absent -- and what is stored. Whether a real model then words it
naturally is not provable offline (see the real-model plan in tools/TESTING-HISTORY.md §22).
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from tests.fakes import completion
from tests.test_control_center_p90 import SAME, admin, app, env_guard, made  # noqa: F401  (fixtures)
from tests.test_phase2_scenarios import make_world, say  # noqa: F401  (fixture)

START = datetime.fromisoformat("2026-11-02T20:00:00+08:00")       # a Monday
INTERVIEW = "用户11月11日下午三点有一场实习面试，有点紧张"
MOVIE = "用户和Eve约好面试结束后一起云看电影《海上钢琴师》"
PET = "用户养了一只柯基，叫豆豆"
BREAKUP = "用户上个月刚和前任分手，现在好多了"


@pytest.fixture()
def clock(monkeypatch):
    from tests.fakes import ScenarioClock, install_clock, uninstall_clock
    c = ScenarioClock(START)
    saved = install_clock(c)
    yield c
    uninstall_clock(saved)


def at(text: str) -> datetime:
    return datetime.fromisoformat(text)


class MemoryModel:
    """Deterministic memory "model": per user message, the tool calls to make, round by round.

    A step is a list of tool calls or a callable (tool results so far -> list of tool calls or None)."""

    def __init__(self, db):
        self.db = db
        self.plan: list[tuple[str, list]] = []
        self.results: list[dict] = []

    def find(self, needle: str) -> int:
        row = self.db.fetch_one("SELECT id FROM event_memories WHERE content LIKE ? AND status != 'deleted' "
                                "ORDER BY id DESC LIMIT 1", (f"%{needle}%",))
        assert row, f"no memory containing {needle!r}"
        return row["id"]

    def __call__(self, kw):
        messages = kw["messages"]
        last = max(i for i, m in enumerate(messages) if m["role"] == "user")
        said = messages[last]["content"].split("\nViolet 实际回复：")[0]
        after = messages[last + 1:]
        results = [json.loads(m["content"]) for m in after if m["role"] == "tool"]
        self.results.extend(results[-1:] if results else [])
        rounds = sum(1 for m in after if m["role"] == "assistant")
        for trigger, steps in self.plan:
            if trigger in said and rounds < len(steps):
                step = steps[rounds]
                calls = step(results) if callable(step) else step
                if calls:
                    return completion(tool_calls=calls)
        return "整理完了"


def save(**args):
    return {"name": "save_memory", "arguments": {"category": "general", **args}}


def new_event_if_asked(**args):
    """Second round: a needs_decision answer is decided as "another event" (the model saw the candidates)."""
    def step(results):
        if results and results[-1].get("action") == "needs_decision":
            return [save(**args, new_event=True)]
        return None
    return step


def recall_lines(system: str, memory_id: int) -> list[str]:
    return [line for line in system.splitlines() if line.startswith(f"- id={memory_id};")]


def standing_lines(system: str) -> list[str]:
    """The lines of the every-turn section of the owner's facts (2026-11-09)."""
    from ai.context_engine import ContextEngine
    parts = system.split(ContextEngine.STANDING_HEADER, 1)
    out = []
    for line in (parts[1].strip("\n").splitlines() if len(parts) == 2 else []):
        if not line.startswith("- "):
            break
        out.append(line)
    return out


def test_pet_names_and_greetings_do_not_pull_up_intimate_memories(violetdb):
    from ai.recall_search import user_word_recall
    violetdb.add_event_memory("用户喜欢Eve叫他宝贝，觉得很甜", kind="fact")
    violetdb.add_event_memory("用户说和Eve的第一次深夜长聊让他很心动", kind="event", progress="done", emotional_weight=4)
    for message in ("宝贝晚安", "早安呀宝贝", "抱抱", "哈哈哈哈", "今天好累啊"):
        assert user_word_recall(violetdb, message) == [], message


def test_word_recall_respects_limit_switch_deleted_and_disabled(violetdb, monkeypatch):
    from ai.recall_search import user_word_recall
    ids = [violetdb.add_event_memory(f"用户的吉他课第{i}次上课", kind="event", progress="done") for i in range(5)]
    violetdb.retract_event_memory(ids[0], reason="test", source="dashboard")
    violetdb.set_event_memory_enabled(ids[1], False)
    found = [m["id"] for m in user_word_recall(violetdb, "今天吉他课好难")]
    assert len(found) == 3 and ids[0] not in found and ids[1] not in found
    monkeypatch.setenv("VIOLET_MEMORY_WORD_RECALL", "0")
    assert user_word_recall(violetdb, "今天吉他课好难") == []


def test_important_memory_stops_at_archived_while_small_talk_may_fade(violetdb):
    important = violetdb.add_event_memory("用户和Eve第一次一起跨年", category="event", kind="event", progress="done",
                                       importance=3)
    small = violetdb.add_event_memory("用户今天吃了麻辣烫", category="event", kind="event", progress="done")
    for _ in range(90):
        violetdb.decay_event_memories()
    assert violetdb.get_event_memory(important)["status"] == "archived"
    assert violetdb.get_event_memory(small)["status"] == "forgotten"


def test_restore_tool_lists_only_rows_without_a_user_deletion(violetdb, tmp_path):
    import shutil
    from tools.memory_restore_decayed import main
    decayed = violetdb.add_event_memory("用户小时候养过一只兔子", kind="event", progress="done")
    violetdb.execute("UPDATE event_memories SET status = 'deleted' WHERE id = ?", (decayed,))     # what old decay did
    by_user = violetdb.add_event_memory("用户不想再提的事", kind="event", progress="done")
    violetdb.retract_event_memory(by_user, reason="用户删除", source="dashboard")
    copy = tmp_path / "copy.db"
    violetdb.conn.execute("PRAGMA wal_checkpoint(FULL)")
    shutil.copy(violetdb.db_path, copy)
    import sqlite3
    assert main(["--db", str(copy)]) == 0                                   # list only
    conn = sqlite3.connect(copy)
    assert conn.execute("SELECT status FROM event_memories WHERE id = ?", (decayed,)).fetchone()[0] == "deleted"
    conn.close()
    assert main(["--db", str(copy), "--apply", "--ids", str(by_user)]) == 2  # a user deletion is never a candidate
    assert main(["--db", str(copy), "--apply"]) == 0
    conn = sqlite3.connect(copy)
    rows = dict(conn.execute("SELECT id, status FROM event_memories").fetchall())
    actions = [r[0] for r in conn.execute("SELECT action FROM memory_revisions WHERE memory_id = ?", (decayed,))]
    conn.close()
    assert rows == {decayed: "forgotten", by_user: "deleted"} and actions == ["restore"]
