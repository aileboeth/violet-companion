"""2026-11-10 (user's QQ screenshot, "这个问题已经遇到好几次了，发出的是表情包的介绍"; the server's outbound_messages had
'2026-11-10T13:03:54 text sent [表情包：角色眯眼微笑着，嘴巴微张呈猫咪状…' two minutes after two real stickers).

A sent sticker is kept in the chat as "[表情包：<description>]"; since 0.7.63 her past turns are real assistant turns,
the model copied that form and the description went out as words. Checked here:
- a copied sticker label in a reply becomes the sticker it means, or is dropped -- never sent as text;
- picture / voice labels likewise; the real markers are untouched;
- in the prompt her past sticker reads as its marker [[表情:N]] (nothing to copy as words);
- the same in groups (the social decision's messages).
Fakes only; the sticker descriptions are invented in the shape of the real ones."""

from __future__ import annotations

from datetime import datetime

import pytest

from tests.image_fixtures import PNG
from tests.test_qq_social_violet import engine, say  # noqa: F401
from tests.test_stickers_qzone_violet import GIF, StickerAdapter

SMILE = "角色眯眼微笑着，嘴巴微张呈猫咪状，脸颊泛红；开心、卖萌或得意的时候发；关键词：开心 卖萌 得意"
LEAKED = "[表情包：角色眯眼微笑着，嘴巴微张呈猫咪状，脸颊泛红，表达开心、卖萌或得意的情绪。]"


@pytest.fixture()
def db(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    yield violetdb


def _kinds(payload):
    return [(m["type"], m.get("content") or m.get("sticker_id")) for m in payload["messages"]]


def test_the_screenshot_reply_sends_the_sticker_not_its_description(db):
    from ai import stickers
    from ai.chat_output import normalize_chat_output
    s = stickers.add_bytes(db, PNG, source="upload", description=SMILE)
    payload = normalize_chat_output("小可爱这种词，我不太认\n你要真觉得可爱，就多看两眼，别光嘴上讲\n" + LEAKED)
    assert _kinds(payload) == [("text", "小可爱这种词，我不太认"), ("text", "你要真觉得可爱，就多看两眼，别光嘴上讲"),
                               ("sticker", s["id"])]


def test_a_label_that_fits_no_sticker_is_dropped(db):
    from ai import stickers
    from ai.chat_output import normalize_chat_output
    stickers.add_bytes(db, PNG, source="upload", description="小狗点头，表示同意")
    payload = normalize_chat_output("好呀\n[表情包：一只企鹅在雪地里打滚]\n[表情包]")
    assert _kinds(payload) == [("text", "好呀")]
    assert normalize_chat_output("[表情包：一只企鹅在雪地里打滚]")["should_send"] is False


def test_a_label_inside_a_line_is_taken_out_of_the_words(db):
    from ai import stickers
    from ai.chat_output import normalize_chat_output
    s = stickers.add_bytes(db, PNG, source="upload", description=SMILE)
    payload = normalize_chat_output("哼 " + LEAKED + " 才不是")
    assert _kinds(payload) == [("text", "哼"), ("sticker", s["id"]), ("text", "才不是")]


def test_picture_and_voice_labels_and_the_real_markers(db):
    from ai import stickers
    from ai.chat_output import normalize_chat_output
    s = stickers.add_bytes(db, PNG, source="upload", description="比心")
    payload = normalize_chat_output("[图片：她画的 AI 插画]\n[语音] 晚安呀\n看这个\n[[图片:7]]")
    assert _kinds(payload) == [("voice", "晚安呀"), ("text", "看这个"), ("image", None)]
    assert payload["messages"][2]["asset_id"] == 7
    assert _kinds(normalize_chat_output(f"[[表情:{s['id']}]]")) == [("sticker", s["id"])]       # untouched
    assert _kinds(normalize_chat_output(f"[表情:{s['id']}]")) == [("sticker", s["id"])]         # one bracket
    assert _kinds(normalize_chat_output("[[语音]]早安")) == [("voice", "早安")]
    assert _kinds(normalize_chat_output("我在[图书馆]呢")) == [("text", "我在[图书馆]呢")]           # other brackets


def test_switched_off_stickers_are_not_sent_from_a_label(db):
    from ai import stickers
    from ai.chat_output import normalize_chat_output
    s = stickers.add_bytes(db, PNG, source="upload", description=SMILE)
    stickers.update(db, s["id"], enabled_flag=False)
    assert normalize_chat_output(LEAKED)["should_send"] is False


@pytest.fixture()
def single(monkeypatch):
    monkeypatch.setenv("VIOLET_PROMPT_LAYOUT", "single")


def test_in_the_prompt_her_sticker_reads_as_its_marker(db, single, monkeypatch):
    from ai import settings, stickers
    from tests.fakes import FakeLLM, default_responders, install_fake_llm
    from ai.agents.chat_agent import ChatAgent
    s = stickers.add_bytes(db, PNG, source="upload", description=SMILE)
    gone = stickers.add_bytes(db, GIF, source="upload", description="小狗点头，表示同意")
    day = "2026-11-10"
    db.add_user_message("text", "宝宝", created_at=f"{day}T13:01:00+08:00")
    db.add_assistant_message("text", "干嘛", created_at=f"{day}T13:01:20+08:00")
    db.add_assistant_message("text", f"[表情包：{SMILE}]", created_at=f"{day}T13:01:21+08:00")
    db.add_assistant_message("text", "[表情包：小狗点头，表示同意]", created_at=f"{day}T13:01:22+08:00")
    stickers.delete(db, gone["id"])
    monkeypatch.setattr(settings, "now_local", lambda: datetime.fromisoformat(f"{day}T13:03:00+08:00"))
    llm = install_fake_llm(monkeypatch, FakeLLM(default_responders("好")))
    ChatAgent().run("那你是小可爱吗")
    messages = llm.calls[0]["messages"]
    mine = [m["content"] for m in messages if m["role"] == "assistant"]
    assert mine == [f"干嘛\n[[表情:{s['id']}]]"]
    assert not any("[表情包：" in str(m["content"]) for m in messages if m["role"] == "assistant")


def test_a_turn_of_only_a_gone_sticker_is_left_out(db, single):
    from ai.agents.chat_agent import compose
    out = compose(["规则"],
                         [{"role": "user", "content": "在吗"}, {"role": "assistant", "content": "[表情包：早就删掉的]"}],
                         [], "人呢")
    assert [m["role"] for m in out] == ["system", "user"] and "表情包" not in out[1]["content"]


def test_in_a_group_a_copied_label_becomes_her_sticker(db, tmp_path, monkeypatch):
    from ai import stickers
    from ai.social.store import SocialStore
    store = SocialStore(db)
    conv = store.conversation("napcat", "group", "900", "表情包群", default_mode="auto")
    a = StickerAdapter(tmp_path)
    e = engine(db, a)
    mine = stickers.add_bytes(db, GIF, source="upload", description="小狗点头，表示同意")
    e.receive(a, kind="group", peer_id="900", sender_id="1", sender_name="小周", text="明天一起去吗",
              platform_msg_id="601")
    say(monkeypatch, {"action": "say", "messages": ["去呀", "[表情包：小狗点头，表示同意]"]}, {})
    store.update(conv["id"], mode="auto")         # an earlier test file's settings may leave new groups off
    r = e.decide(conv["id"])
    assert a.sent == [("去呀", False), (None, True)] and r["sent"] == 2
    assert stickers.get(db, mine["id"])["use_count"] == 1
    say(monkeypatch, {"action": "say", "messages": ["[表情包：一只企鹅在打滚]"]}, {})
    e.receive(a, kind="group", peer_id="900", sender_id="1", sender_name="小周", text="好耶", platform_msg_id="602")
    e.decide(conv["id"])
    assert len(a.sent) == 2                                   # nothing fit: nothing sent, never the words
