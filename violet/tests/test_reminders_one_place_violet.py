"""2026-11-13 (0.7.73 rerun after the relay outage; user: 可以按你建议来做):
- "明早记得叫我一声" (no time): she only asked "几点叫，还是七点半？" and set nothing; the memory step then added its own
  reminder "明早叫用户一声（起床提醒）" at a guessed 08:00, and at 08:00 the fallback template sent it word for word.
  Reminders are set in ONE place now (Chat, set_reminder); with no time given she uses the time set before and says it.
- the relay's model channel was down for 45 minutes: every message got the same fallback line back, a dozen times."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from tests.test_conditional_memory_p921 import relay_down
from tests.test_phase2_scenarios import make_world, say  # noqa: F401  (make_world: fixture)


@pytest.mark.parametrize("record, said", [
    ("明早叫用户一声（起床提醒）", "叫你一声"),
    ("起床，九点考试别迟到", "起床，九点考试别迟到"),
    ("交报告", "交报告"),
    ("明天提醒对方交作业", "提醒你交作业"),
])
def test_a_reminder_is_said_to_the_owner_not_read_out_as_a_record(record, said):
    from ai.chat import spoken_reminder
    assert spoken_reminder(record) == said


def test_the_fallback_reminder_uses_the_spoken_form(monkeypatch):
    from ai import chat

    class Silent:                                  # the model wrote nothing usable
        def run(self, *a, **k):
            return {"content": ""}
    out = chat.run_commitment_pipeline({"id": 1, "content": "明早叫用户一声（起床提醒）"}, Silent())
    assert out["used_fallback"] and out["messages"][0]["content"] == "提醒你一下：叫你一声"


def test_setting_a_reminder_with_no_time_given_uses_the_time_set_before():
    from ai.owner_tools import SET_REMINDER_SCHEMA
    from ai.tool_manager import Catalog, abilities_note
    text = SET_REMINDER_SCHEMA["function"]["description"]
    assert "以前定过的时间" in text and "只问不设的话，对方睡着了就没人叫他了" in text
    assert "提醒只有你这里能设" in abilities_note(Catalog(), [])


def test_the_fallback_line_is_said_once_while_the_model_stays_down(make_world, monkeypatch):
    monkeypatch.setenv("LLM_RETRY_BASE_DELAY", "0")
    w = make_world()
    w.llm.responders["fake-chat"] = relay_down
    for text in ("在吗", "怎么不说话", "你是不是卡住了"):
        say(w, text)
    fallback = "我这边刚刚卡了一下，没接上你的话，你再说一次好不好"
    assert w.client.sent_texts == [fallback]
    w.llm.responders["fake-chat"] = lambda kw: "刚才信号不好，现在好了"
    say(w, "好了没")
    assert w.client.sent_texts[-1] == "刚才信号不好，现在好了"
    w.llm.responders["fake-chat"] = relay_down                 # down again after a real reply: said once again
    say(w, "又卡了？")
    assert w.client.sent_texts[-1] == fallback and w.client.sent_texts.count(fallback) == 2


def test_an_old_fallback_does_not_keep_her_quiet(make_world, monkeypatch):
    monkeypatch.setenv("LLM_RETRY_BASE_DELAY", "0")
    w = make_world()
    w.llm.responders["fake-chat"] = relay_down
    say(w, "在吗")
    old = (datetime.fromisoformat(w.db.now()) - timedelta(hours=3)).isoformat()
    w.db.execute("UPDATE outbound_messages SET created_at = ? WHERE origin = 'fallback'", (old,))
    say(w, "在吗在吗")
    assert len(w.client.sent_texts) == 2
