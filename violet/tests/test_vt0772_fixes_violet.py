"""2026-11-13 fixes after the 0.7.72 real-model test on the server (user: 按你建议来，所有都修一下):
1. she promised "晚点再帮你查" and her later message of her own -- which has no tools -- said "又翻了一遍，还是没查到";
2. the owner called his moved exam "补考" and she said "这个你倒是没提" with that exam in her records;
3. the proactive decision's topic said "下午的课（包括上机课）" and she wrote it, after saying at noon it was English;
4. "我叫小周，刚来找你聊聊" counted as warmth, so on day two she was "warming" and wrote "突然有点想你了"."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from tests.test_qq_social_violet import db  # noqa: F401


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def test_she_knows_her_own_messages_have_no_tools_so_she_checks_now():
    from ai.tool_manager import Catalog, abilities_note
    note = abilities_note(Catalog(), ["violet-web__web_search"])
    assert "你自己主动发消息时什么工具都用不了" in note and "现在就查好告诉对方，不要答应以后再查" in note


@pytest.mark.parametrize("line, promised", [
    ("查到的只有今天的，明天有没有雨没查到，我晚点再试一次", True),        # real runs after the abilities note
    ("你专心做卷子吧，我回头再帮你看看", True),
    ("行，晚点我再瞅一眼，有雨提前告诉你", True),                      # the 0.7.72 virtual test
    ("等会儿帮你查查", True),
    ("你先安心做卷子，明早出门前再翻一眼就行", False),                # the owner checks
    ("我晚点看看书再睡", False),                                       # her own evening
    ("晚点再给你补一句", False),
    ("我查了一下，明天没搜到准的", False),                             # done now
])
def test_a_lookup_promised_for_later_is_caught(line, promised):
    from ai.consistency import PROMISED_LOOKUP_RE
    assert bool(PROMISED_LOOKUP_RE.search(line)) is promised


@pytest.mark.parametrize("line, promised", [
    ("你专心做卷子，晚点我再瞄一眼", True),                           # 5 real runs of the narrow pattern
    ("你先把卷子做了，我这边再帮你找找", True),
    ("搜到的只有广州今天的，明天早上会不会下雨查不到，等我晚上再看一眼", True),
    ("我刚看了一眼，只翻到今天广州多云转晴", False),
    ("明早的雨我这边查不到准的，你出门前自己刷一下天气app吧", False),
    ("我翻了两个网页，都只写着广州今天多云转晴", False),
])
def test_on_a_turn_she_searched_any_later_lookup_of_hers_is_caught(line, promised):
    from ai.consistency import promised_lookup
    assert promised_lookup(line, searched=True) is promised


def test_without_a_search_this_turn_only_the_narrow_form_counts():
    from ai.consistency import promised_lookup
    assert promised_lookup("晚点我再瞄一眼", searched=False)
    assert not promised_lookup("我这边再帮你找找那本书的书签", searched=False)


def test_the_reply_check_asks_for_a_rewrite_of_such_a_promise(db, data_dir):
    from ai.chat import CONSISTENCY_HINTS
    from ai.consistency import ConsistencyGuard
    found = ConsistencyGuard(db).check_reply("没搜到明天的\n我回头再帮你看看")
    assert [v.check for v in found] == ["promised_later_lookup"] and found[0].line == "我回头再帮你看看"
    assert "promised_later_lookup" in CONSISTENCY_HINTS


def test_her_own_message_says_nothing_was_checked_and_the_chat_wins_over_the_brief():
    from ai.agents.chat_agent import PROACTIVE_REQUEST
    from ai.settings import PROJECT_ROOT
    assert "用不了任何工具，也没有查过什么" in PROACTIVE_REQUEST
    assert "以上面的聊天原文和背景为准" in PROACTIVE_REQUEST
    decision = (PROJECT_ROOT / "persona/proactive_prompt_versions/proactive.v4.txt").read_text(encoding="utf-8")
    assert "主动发的消息里查不了东西" in decision and "不要替她补具体细节" in decision


def test_a_different_name_for_a_known_thing_is_not_news():
    from ai.agents.chat_agent import WHO_IS_WHO
    assert "多半是同一件事换了个说法" in WHO_IS_WHO and "不要说“你没跟我说过 / 你没提”" in WHO_IS_WHO


@pytest.mark.parametrize("text, warm", [
    ("我叫小周，刚来找你聊聊哈哈", False),       # how anyone starts a chat
    ("又来找你啦", False),
    ("今天好累 想找你聊聊", True),
    ("以后多找你聊", True),
    ("就想来找你", True),
    ("想你了 来找你聊会儿", True),
    ("今天好忙 都没顾上找你", False),
])
def test_coming_to_chat_is_not_warmth_wanting_to_is(text, warm):
    from ai.relationship import is_warm
    assert is_warm(text) is warm


def test_a_plain_first_hello_does_not_make_day_two_warming(db, data_dir):
    from ai.relationship import assess
    first = datetime.fromisoformat(db.now()) - timedelta(days=1)
    for i, text in enumerate(["我叫小周，刚来找你聊聊哈哈", "今天上了一天课"]):
        db.add_message("user", "incoming", "text", text)
        db.execute("UPDATE messages SET created_at = ? WHERE id = (SELECT MAX(id) FROM messages)",
                   ((first + timedelta(minutes=i)).isoformat(),))
    db.add_message("user", "incoming", "text", "中午好")
    assert assess(db).stage == "new"
