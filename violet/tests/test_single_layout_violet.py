"""0.7.63 (user, 2026-11-10: "不到半个小时又出现左脑和右脑互搏……不要再打补丁了，重构吧"; export 2026-11-09 22:51-23:13).

What the replay of that export showed, and what is checked here:
- one message of the owner reached the model behind 5-7 separate notes that a relay turns into *user* messages -> now
  ONE system message, the chat strictly in turns, ONE last user message (a fenced background block + the owner's words);
- "你们是恋人" stood next to "第一次聊天在 22:51（15 分钟前），认识还不到一天，还不太熟" -> a chosen relationship is how
  close the two are, the log only says how far her records go; no running minute count for anyone;
- clearing the memory left her mood (with a quote of the cleared chat) behind;
- 流程: a finished 生成回复 step could not be opened (a variable hid the function of the same name).
Fakes only; the chat lines here are invented (same shape as the export: a quoted reply after a role swap)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1]
DAY = "2026-11-09"
CHAT = [("user", "22:51:47", "你好呀~"), ("assistant", "22:52:29", "哟 终于舍得来找我了"),
        ("assistant", "22:52:31", "说说吧 今天都干嘛了"), ("user", "23:06:59", "叫我一声哥哥~"),
        ("assistant", "23:07:11", "想得美"), ("assistant", "23:07:14", "要不你先叫声老师来听听"),
        ("user", "23:11:12", "老师~\n好听吗"), ("assistant", "23:11:32", "这一声老师叫得我心情挺好")]
SAID = "[引用你之前说的：这一声老师叫得我心情挺好]\n就这样？该你叫了"
PHONE = "【主人手机此刻（23:12 读取）】屏幕亮着，正在用「QQ」"


@pytest.fixture(autouse=True)
def _production_layout(monkeypatch):
    monkeypatch.setenv("VIOLET_PROMPT_LAYOUT", "single")          # conftest pins the old tests to "classic"


def test_the_default_is_the_single_layout(monkeypatch):
    from ai.agents.chat_agent import single_layout
    monkeypatch.delenv("VIOLET_PROMPT_LAYOUT", raising=False)
    assert single_layout()


@pytest.fixture()
def night(violetdb, monkeypatch):
    from ai import settings
    for role, clock, text in CHAT:
        add = violetdb.add_user_message if role == "user" else violetdb.add_assistant_message
        add("text", text, created_at=f"{DAY}T{clock}+08:00")
    monkeypatch.setattr(settings, "now_local", lambda: datetime.fromisoformat(f"{DAY}T23:12:11+08:00"))
    return violetdb


def _partner(monkeypatch):
    from ai import persona_templates
    monkeypatch.setattr(persona_templates, "relationship", lambda channel=None: {
        "kind": "partner", "label": persona_templates.RELATIONSHIPS["partner"][0], "call": ""})


def _sent(monkeypatch, text=SAID, context=None):
    from tests.fakes import FakeLLM, default_responders, install_fake_llm
    from ai.agents.chat_agent import ChatAgent
    llm = install_fake_llm(monkeypatch, FakeLLM(default_responders("好")))
    ChatAgent().run(text, call_context={"phone_note": PHONE, **(context or {})})
    return llm.calls[0]["messages"]


def test_one_system_message_real_turns_and_one_last_user_message(night, monkeypatch):
    from ai.agents.chat_agent import BACKGROUND_CLOSE, BACKGROUND_OPEN, SAID_NOW
    messages = _sent(monkeypatch)
    roles = [m["role"] for m in messages]
    assert roles[0] == "system" and "system" not in roles[1:]
    assert all(a != b for a, b in zip(roles[1:], roles[2:])), roles          # strictly user / assistant in turn
    assert roles[-1] == "user"
    last = messages[-1]["content"]
    assert last.startswith(BACKGROUND_OPEN) and last.endswith(SAID_NOW + "\n" + SAID)
    background = last[:last.index(BACKGROUND_CLOSE)]
    assert PHONE in background and "当前时间" in background                 # this moment's notes are all in the block
    assert SAID not in background
    assert messages[-2] == {"role": "assistant", "content": "这一声老师叫得我心情挺好"}
    assert messages[1]["content"] == "你好呀~"                               # the chat itself is untouched


def test_nothing_is_left_for_the_relay_workaround_to_turn_into_user_messages(night, monkeypatch):
    from ai import model_params
    out = model_params.apply({"model": "deepseek-chat", "messages": _sent(monkeypatch)})["messages"]
    assert not any(model_params.LATE_SYSTEM_LABEL.strip() in str(m.get("content")) for m in out)
    assert [m["role"] for m in out].count("system") == 1
    assert not any("_cache_break" in m for m in out)


def test_a_chosen_relationship_is_not_contradicted_by_a_short_chat_log(night, monkeypatch):
    _partner(monkeypatch)
    text = "\n".join(str(m["content"]) for m in _sent(monkeypatch))
    assert "你们的关系：恋人" in text
    for wrong in ("还不太熟", "认识还不到一天", "分钟前", "还在互相熟悉"):
        assert wrong not in text, wrong
    assert "聊天记录从 11-09 22:51 开始" in text and "按“你们的关系”来" in text


def test_no_running_minute_count_without_a_chosen_relationship_either(night, monkeypatch):
    text = "\n".join(str(m["content"]) for m in _sent(monkeypatch))
    assert "分钟前" not in text and "你们今天才开始聊天" in text


def test_the_two_people_are_named_once_and_her_life_is_hers(night, monkeypatch):
    from ai.agents.chat_agent import WHO_IS_WHO
    messages = _sent(monkeypatch)
    assert WHO_IS_WHO in messages[0]["content"]
    text = "\n".join(str(m["content"]) for m in messages)
    assert "用户此刻在说自己的事" not in text and "才是她做过的事" not in text


def test_she_answers_in_the_language_she_is_written_to(night, monkeypatch):
    """user, 2026-11-10: "输入的什么语言回答什么语言吧，中间夹杂几句外语也比较正常" -- one rule in the fixed part."""
    from ai.agents.chat_agent import REPLY_LANGUAGE
    messages = _sent(monkeypatch, "what are you doing?")
    assert REPLY_LANGUAGE in messages[0]["content"] and "夹几个外语词" in REPLY_LANGUAGE


def test_a_long_gap_is_a_line_in_a_user_turn_never_her_words(violetdb, monkeypatch):
    from ai import settings
    from ai.agents.chat_agent import RECORD_NOTE
    violetdb.add_user_message("text", "晚安", created_at=f"{DAY}T01:00:00+08:00")
    violetdb.add_assistant_message("text", "晚安呀", created_at=f"{DAY}T01:00:20+08:00")
    violetdb.add_assistant_message("text", "中午了，吃了没", created_at=f"{DAY}T12:00:00+08:00")
    monkeypatch.setattr(settings, "now_local", lambda: datetime.fromisoformat(f"{DAY}T12:01:00+08:00"))
    messages = _sent(monkeypatch, "吃了")
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[3]["content"].startswith(RECORD_NOTE) and "隔了" in messages[3]["content"]
    assert messages[4]["content"] == "中午了，吃了没"


def test_an_unanswered_message_and_the_new_one_are_one_user_turn(violetdb, monkeypatch):
    from ai import settings
    violetdb.add_user_message("text", "在吗", created_at=f"{DAY}T12:00:00+08:00")
    monkeypatch.setattr(settings, "now_local", lambda: datetime.fromisoformat(f"{DAY}T12:00:30+08:00"))
    messages = _sent(monkeypatch, "人呢")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"].startswith("在吗\n") and messages[1]["content"].endswith("人呢")


def test_a_proactive_message_uses_the_same_layout(night, monkeypatch):
    from tests.fakes import FakeLLM, default_responders, install_fake_llm
    from ai.agents.chat_agent import BACKGROUND_OPEN, ChatAgent, PROACTIVE_REQUEST
    llm = install_fake_llm(monkeypatch, FakeLLM(default_responders("在干嘛")))
    ChatAgent().run("[proactive event]", call_context={"is_proactive": True, "proactive_result": "想她了"})
    messages = llm.calls[0]["messages"]
    assert [m["role"] for m in messages].count("system") == 1 and messages[-1]["role"] == "user"
    assert BACKGROUND_OPEN in messages[-1]["content"] and "想她了" in messages[-1]["content"]
    assert PROACTIVE_REQUEST[:20] in messages[-1]["content"] and "[proactive event]" not in messages[-1]["content"]


def test_a_whole_turn_through_the_pipeline_in_the_production_layout(night, monkeypatch):
    """Intent off (as in production), abilities note, memories, the reply guards: one turn end to end."""
    from tests.fakes import FakeLLM, default_responders, install_fake_llm
    monkeypatch.setenv("VIOLET_INTENT_STEP", "0")
    night.add_event_memory("对方蓝牙连接的是手表，不是耳机", category="other", kind="fact", importance=2.0)
    fake = install_fake_llm(monkeypatch, FakeLLM(default_responders("好啦 哥哥")))
    from ai.agents import ChatAgent, IntentAgent, MemoryAgent, PerformerAgent
    from ai.agents.chat_agent import notes_and_words
    from ai.chat import run_pipeline
    result = run_pipeline(user_message=SAID, intent_agent=IntentAgent(), chat_agent=ChatAgent(),
                          performer_agent=PerformerAgent(), memory_agent=MemoryAgent(), defer_memory=True)
    assert not result["chat_failed"] and result["chat"]["content"] == "好啦 哥哥"
    sent = next(c["messages"] for c in fake.calls if c["messages"][-1]["role"] == "user"
                and SAID in str(c["messages"][-1]["content"]))
    assert [m["role"] for m in sent].count("system") == 1
    notes, words = notes_and_words(sent)
    assert words == SAID
    assert "你现在能做的事" in sent[0]["content"]                      # the abilities: fixed part
    assert "手表" in sent[0]["content"] + notes                       # what the owner told her reaches the turn


def test_the_old_layout_is_still_there_to_compare(night, monkeypatch):
    monkeypatch.setenv("VIOLET_PROMPT_LAYOUT", "classic")
    messages = _sent(monkeypatch)
    assert [m["role"] for m in messages].count("system") > 3 and messages[-1] == {"role": "user", "content": SAID}


def test_clearing_the_memory_also_clears_her_mood(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    from ai import mood, persona_templates
    mood._row(violetdb)                                             # the table exists
    violetdb.execute("INSERT INTO mood_state (id, value, cause, updated_at) VALUES (1, -0.6, ?, ?) "
                     "ON CONFLICT(id) DO UPDATE SET value = excluded.value, cause = excluded.cause",
                     ("主人对你说了「你好烦」", violetdb.now()))
    assert mood.note(violetdb)
    persona_templates.clear_memory(violetdb)
    assert not mood.note(violetdb)


def test_flow_page_step_with_a_model_conversation_can_be_drawn():
    """messagesView() named its total `chars` -- the same name as the chars() function it calls two lines below, so a
    finished 生成回复 threw "chars is not a function" and showed nothing (a skipped step has no conversation)."""
    js = (APP / "wx_ilink_py" / "static" / "control_flow.js").read_text(encoding="utf-8")
    body = js[js.index("function messagesView"):js.index("function watch")]
    assert "chars(String(text).length)" in body and "const chars" not in body and "let chars" not in body


def test_the_language_can_be_changed_in_the_general_settings():
    html = (APP / "wx_ilink_py" / "templates" / "control.html").read_text(encoding="utf-8")
    assert 'id="set-g-language"' in html and 'id="set-lang-box"' in html
    js = (APP / "wx_ilink_py" / "static" / "control_i18n.js").read_text(encoding="utf-8")
    assert 'getElementById("set-lang-box")' in js
    assert '"set-g-language"' in (APP / "wx_ilink_py" / "static" / "control.js").read_text(encoding="utf-8")
