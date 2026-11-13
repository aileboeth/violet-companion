"""2026-11-12: the virtual clock behind tools/virtual_test.py, and what the test counts by code."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def test_the_virtual_clock_moves_every_now_and_goes_back():
    from ai.settings import now_local, set_clock
    from database import VioletDatabase
    fixed = datetime(2026, 11, 16, 7, 0, tzinfo=timezone(timedelta(hours=8)))
    try:
        set_clock(lambda: fixed)
        assert now_local().isoformat().startswith("2026-11-16T07:00")
        assert VioletDatabase.now().startswith("2026-11-16T07:00")
    finally:
        set_clock(None)
    assert abs((now_local() - datetime.now(now_local().tzinfo)).total_seconds()) < 5


def test_code_checks_count_answers_repeats_and_questions():
    from tools.virtual_test import code_checks
    entries = [{"who": "主人", "text": "在吗"}, {"who": "她", "text": "在呀"}, {"who": "她", "text": "怎么啦？"},
               {"who": "主人", "text": "没事"}, {"who": "她", "text": "那我陪你待一会儿。怎么啦？"},
               {"who": "她（主动）", "text": "哥，吃饭了吗"}]
    c = code_checks(entries, "哥")
    assert c["answers"] == 2 and c["messages"] == 3 and c["max_messages_per_answer"] == 2
    assert c["answers_ending_with_question"] == 2 and c["call_in_messages"] == 0
    assert c["repeated_sentences"] == [] and code_checks(entries * 2, "哥")["repeated_sentences"]


def test_every_track_is_well_formed():
    from tools.virtual_test import TRACKS
    for name, track in TRACKS.items():
        assert track["sessions"] and track["persona"], name
        days = [s["day"] for s in track["sessions"]]
        assert days == sorted(days), name
        for s in track["sessions"]:
            assert 1 <= s["max_turns"] <= 10 and s["agenda"]
            for _, due in s["reminders"]:
                assert int(due.split()[0]) >= s["day"]
