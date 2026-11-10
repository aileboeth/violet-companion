"""2026-11-10 (user: "将 SnowLuma 里面的一些功能集成上去 …… SnowLuma 不是还有些功能吗一起做了"): more of what QQ can do
through SnowLuma's OneBot actions, and an install that no longer depends on SnowLuma's image being downloadable
(its repository and Docker Hub image went offline that day).

Checked here: a voice message read through QQ's speech-to-text, a forwarded chat log opened, a group's owner / admins
marked, the missed group messages caught up, "正在输入", taking back her own messages (owner chat and groups), emoji
reactions, group admin actions only when allowed and asked by the owner, her small QQ touches (likes, a comment, the
signature) only when turned on, the install from the image on the server / a saved copy, and the image kept on
uninstall. Fakes only: no network, no model, no QQ, no Docker."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from tests.test_qq_social_violet import FakeHost, FakeWS, NoTimer, engine, say  # noqa: F401

SELF = "2860001234"


@pytest.fixture()
def db(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    for table in ("social_messages", "social_decisions", "social_people", "social_conversations", "social_done",
                  "approvals"):
        violetdb.execute(f"DELETE FROM {table}")
    yield violetdb


class Inline:
    """threading.Thread stand-in: runs the target at start()."""

    def __init__(self, target=None, args=(), daemon=None, name=None, **_):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


def channel(monkeypatch, answers=None, owners=("9",)):
    """A connected QQ channel whose actions answer from `answers` (action -> data, or an Exception to raise)."""
    from ai.channels import snowluma
    ch = snowluma.SnowLumaChannel("napcat", ws_factory=lambda *a, **k: FakeWS([]))
    ch._ws, ch.self_id = FakeWS([]), SELF
    ch.runtime = SimpleNamespace(identities=SimpleNamespace(bound_peers=lambda c: list(owners)))
    calls = []

    def call(action, params=None, timeout=20.0):
        calls.append((action, params))
        value = (answers or {}).get(action, {})
        if isinstance(value, Exception):
            raise value
        return value(params) if callable(value) else value
    monkeypatch.setattr(ch, "call", call)
    monkeypatch.setattr(ch, "is_running", lambda: True)
    monkeypatch.setattr(snowluma, "threading", Proxy(threading, Thread=Inline))      # only this module's threads
    return ch, calls


class Proxy:
    """A module with a few names replaced (never the real module's attributes)."""

    def __init__(self, module, **replaced):
        self._module, self._replaced = module, replaced

    def __getattr__(self, name):
        return self._replaced[name] if name in self._replaced else getattr(self._module, name)


# ------------------------------------------------------------------------------------------------ reading more

def test_a_voice_message_reaches_her_as_text(monkeypatch):
    ch, calls = channel(monkeypatch, {"fetch_ptt_text": {"text": "今天好累啊"}})
    got = []
    ch._sink = lambda m: got.append(m) or True
    ch.handle_event({"post_type": "message", "message_type": "private", "user_id": 9, "self_id": SELF, "message_id": 41,
                     "message": [{"type": "record", "data": {"file": "a.silk"}}]})
    assert calls == [("fetch_ptt_text", {"message_id": 41})]
    assert got and "[语音，转成文字是：今天好累啊]" in got[0].parts[0].text


def test_a_voice_that_cannot_be_read_still_arrives(monkeypatch):
    from ai.channels.snowluma import QQError
    ch, calls = channel(monkeypatch, {"fetch_ptt_text": QQError("fetch_ptt_text: retcode=1404 不支持")})
    got = []
    ch._sink = lambda m: got.append(m) or True
    ch.handle_event({"post_type": "message", "message_type": "private", "user_id": 9, "self_id": SELF, "message_id": 42,
                     "message": [{"type": "record", "data": {"file": "a.silk"}}]})
    assert got and "语音" in got[0].parts[0].text and "转成文字是" not in got[0].parts[0].text


def test_a_forwarded_chat_log_is_opened(monkeypatch):
    nodes = [{"sender": {"nickname": "阿凯"}, "message": [{"type": "text", "data": {"text": "周末去爬山吗"}}]},
             {"sender": {"card": "小周"}, "message": "好啊[CQ:face,id=14]"}]
    ch, calls = channel(monkeypatch, {"get_forward_msg": {"messages": nodes}})
    social = []
    monkeypatch.setattr(ch, "_social", lambda: SimpleNamespace(receive=lambda adapter, **kw: social.append(kw)))
    ch.handle_event({"post_type": "message", "message_type": "group", "group_id": 614, "user_id": 7, "self_id": SELF,
                     "message_id": 43, "sender": {"card": "阿凯", "role": "admin"},
                     "message": [{"type": "forward", "data": {"id": "res-1"}}]})
    assert calls == [("get_forward_msg", {"id": "res-1"})]
    text = social[0]["text"]
    assert "[转发来的聊天记录]" in text and "阿凯：周末去爬山吗" in text and "小周：好啊" in text
    assert social[0]["sender_role"] == "admin"


def test_group_roles_are_marked_in_her_chat_log(db):
    from ai.social import prompts
    from ai.social.store import SocialStore
    store = SocialStore(db)
    conv = store.conversation("napcat", "group", "614", "球友群", default_mode="auto")
    store.add_message(conv["id"], content="今晚八点开打", sender_id="1", sender_name="老王", sender_role="owner",
                      platform_msg_id="1")
    store.add_message(conv["id"], content="收到", sender_id="2", sender_name="阿凯", sender_role="member",
                      platform_msg_id="2")
    history = store.messages(conv["id"])
    text = str(prompts.user_prompt_parts(conv, history=history, unread_from=0, people=[], owner_ids=set(),
                                         today={}, last_said_at=None, now=db.now(), trigger="messages"))
    assert "老王(1)［群主］：今晚八点开打" in text and "阿凯(2)：收到" in text


def test_missed_group_messages_are_caught_up_with_their_own_time(db, monkeypatch):
    from ai.social.store import SocialStore
    store = SocialStore(db)
    conv = store.conversation("napcat", "group", "614", "球友群", default_mode="auto")
    store.add_message(conv["id"], content="早", sender_id="1", sender_name="老王", platform_msg_id="100")
    now = time.time()
    history = {"messages": [
        {"message_id": 100, "user_id": 1, "time": now - 600, "message": "早"},                     # stored already
        {"message_id": 101, "user_id": 7, "time": now - 300, "sender": {"card": "阿凯", "role": "admin"},
         "message": [{"type": "text", "data": {"text": "有人在吗"}}]},
        {"message_id": 99, "user_id": 7, "time": now - 8 * 3600, "message": "很久以前"},           # older than 6 h
        {"message_id": 102, "user_id": int(SELF), "time": now - 200, "message": "她自己"}]}
    ch, calls = channel(monkeypatch, {"get_group_msg_history": history})
    a = SimpleNamespace(channel_id="napcat", is_running=lambda: True, owner_ids=lambda: set(), send_to=None)
    e = engine(db, a)
    monkeypatch.setattr(ch, "_social", lambda: e)
    assert ch.catch_up() == 1
    assert calls == [("get_group_msg_history", {"group_id": 614, "count": 20})]
    rows = store.messages(conv["id"])
    added = rows[-1]
    assert added["platform_msg_id"] == "101" and added["content"] == "有人在吗" and added["sender_role"] == "admin"
    at = datetime.fromisoformat(added["created_at"]).timestamp()
    assert abs(at - (now - 300)) < 2
    assert NoTimer.made == []                                 # no decision of its own


def test_typing_in_the_owner_chat(monkeypatch):
    from ai.channels.snowluma import QQError
    ch, calls = channel(monkeypatch)
    ch.send_typing("9", True)
    ch.send_typing("9", False)
    assert calls == [("set_input_status", {"user_id": 9, "event_type": 1}), ("set_input_status", {"user_id": 9, "event_type": 0})]
    monkeypatch.setattr(ch, "call", lambda *a, **k: (_ for _ in ()).throw(QQError("set_input_status: retcode=1404 ")))
    ch.send_typing("9", True)
    assert ch._no_typing                                      # not offered: not asked again
    monkeypatch.setenv("VIOLET_QQ_TYPING", "0")
    assert "typing" in ch.capabilities


def test_a_friend_request_says_who_it_is(db, monkeypatch):
    from ai import permissions
    ch, calls = channel(monkeypatch, {"get_stranger_info": {"nickname": "小雨", "sex": "female", "age": 20,
                                                            "long_nick": "今天也要加油"}})
    said = []
    ch.runtime = SimpleNamespace(db=db, identities=ch.runtime.identities)
    monkeypatch.setitem(permissions._RUNTIME, "rt", SimpleNamespace(send_system_text=said.append))
    ch._friend_request({"post_type": "request", "request_type": "friend", "user_id": 555, "comment": "你好", "flag": "f"})
    assert "QQ 555（昵称 小雨，女，20 岁，签名：今天也要加油）想加我好友" in said[0]


# ------------------------------------------------------------------------------------------------ owner chat tools

def _on_qq(monkeypatch, ch):
    from ai import permissions
    from ai.channels import context
    monkeypatch.setitem(permissions._RUNTIME, "rt", SimpleNamespace(channels=SimpleNamespace(get=lambda cid: ch)))
    token = context._CURRENT.set(context.ChannelContext(channel="napcat", channel_type="napcat", label="QQ"))
    return token


def test_she_takes_back_what_she_just_sent(monkeypatch):
    from ai import qq_tools
    from ai.channels import context
    ch, calls = channel(monkeypatch, {"send_private_msg": {"message_id": -77}})
    ch.last_owner_peer = "9"
    ch.send_text("9", "我发错了")
    assert not qq_tools.available()                           # not on QQ: not offered
    token = _on_qq(monkeypatch, ch)
    try:
        assert qq_tools.available()
        answer = json.loads(qq_tools.recall({}))
        assert answer["ok"] and answer["recalled"] == ["我发错了"] and calls[-1] == ("delete_msg", {"message_id": -77})
        assert json.loads(qq_tools.recall({}))["error_code"] == "NOTHING_TO_RECALL"   # gone from the list
        ch.send_text("9", "很早以前")
        ch._sent[-1]["at"] -= 300                             # older than two minutes
        assert json.loads(qq_tools.recall({}))["error_code"] == "NOTHING_TO_RECALL"
        assert json.loads(qq_tools.like({}))["ok"] and calls[-1] == ("send_like", {"user_id": 9, "times": 1})
    finally:
        context._CURRENT.reset(token)


def test_the_qq_tools_are_offered_only_on_qq(monkeypatch):
    from ai.agents import base_agent
    from ai.agents.chat_agent import ChatAgent
    from ai.channels import context
    agent = ChatAgent()

    def names(call_context):
        token = base_agent._CALL_CONTEXT.set(dict(call_context))
        try:
            return {t["function"]["name"] for t in agent._filter_tools_for_run(list(agent._tools))}
        finally:
            base_agent._CALL_CONTEXT.reset(token)
    assert not names({}) & {"qq_recall", "qq_like"}
    ch, _ = channel(monkeypatch)
    token = _on_qq(monkeypatch, ch)
    try:
        assert {"qq_recall", "qq_like"} <= names({})
        assert not names({"is_proactive": True}) & {"qq_recall", "qq_like"}
    finally:
        context._CURRENT.reset(token)


# ------------------------------------------------------------------------------------------------ in groups

class GroupAdapter:
    channel_id = "napcat"

    def __init__(self, owners=("9",)):
        self.sent, self.done, self.owners = [], [], set(owners)

    def is_running(self):
        return True

    def owner_ids(self):
        return self.owners

    def send_to(self, kind, peer_id, text, reply_to=None, **_):
        self.sent.append(text)
        return f"out{len(self.sent)}"

    def react(self, message_id, emoji):
        from ai.channels.snowluma import reaction_id
        if reaction_id(emoji) is None:
            raise RuntimeError("不能用这个表情")
        self.done.append(("react", message_id, emoji))

    def recall(self, message_id):
        self.done.append(("recall", message_id))

    def group_admin(self, group_id, do, **kw):
        self.done.append(("admin", group_id, do, {k: v for k, v in kw.items() if v not in (None, "")}))


def _group(db, monkeypatch, decision, sender="7", text="哈哈哈笑死"):
    from ai.social.store import SocialStore
    store = SocialStore(db)
    conv = store.conversation("napcat", "group", "614", "球友群", default_mode="auto")
    store.update(conv["id"], mode="auto")
    a = GroupAdapter()
    e = engine(db, a)
    e.receive(a, kind="group", peer_id="614", sender_id=sender, sender_name="阿凯", text=text,
              platform_msg_id=f"m{len(store.messages(conv['id']))}")
    seen = {}
    say(monkeypatch, decision, seen)
    return store, conv, a, e, seen


def test_she_can_just_react_instead_of_talking(db, monkeypatch):
    store, conv, a, e, seen = _group(db, monkeypatch, {"action": "silent", "react": {"message": "#m0", "emoji": "😂"}})
    e.decide(conv["id"])
    assert a.sent == [] and a.done == [("react", "m0", "😂")]
    assert "react" in seen["system"] and "😂" in seen["system"]
    say(monkeypatch, {"action": "silent", "react": {"message": "m1", "emoji": "🦄"}})
    e.receive(a, kind="group", peer_id="614", sender_id="7", sender_name="阿凯", text="再来", platform_msg_id="m1")
    e.decide(conv["id"])
    assert len(a.done) == 1                                   # not an allowed emoji: nothing


def test_she_takes_back_her_own_fresh_message_in_a_group(db, monkeypatch):
    store, conv, a, e, _ = _group(db, monkeypatch, {"action": "say", "messages": ["发错群了"]})
    e.decide(conv["id"])
    mine = [m for m in store.messages(conv["id"]) if m["is_self"]][-1]
    say(monkeypatch, {"action": "silent", "recall": mine["platform_msg_id"]})
    e.receive(a, kind="group", peer_id="614", sender_id="7", sender_name="阿凯", text="？", platform_msg_id="m9")
    e.decide(conv["id"])
    assert ("recall", mine["platform_msg_id"]) in a.done
    assert store.find_message(conv["id"], mine["platform_msg_id"])["content"].startswith("[你撤回了这条]")
    old = (datetime.fromisoformat(db.now()[:19]) - timedelta(minutes=5)).isoformat()
    store.add_message(conv["id"], content="老消息", is_self=True, platform_msg_id="old1", created_at=old)
    say(monkeypatch, {"action": "silent", "recall": "old1"})
    e.receive(a, kind="group", peer_id="614", sender_id="7", sender_name="阿凯", text="！", platform_msg_id="m10")
    e.decide(conv["id"])
    assert ("recall", "old1") not in a.done


def test_group_admin_is_off_by_default(db, monkeypatch):
    store, conv, a, e, seen = _group(db, monkeypatch, {"action": "say", "messages": ["好"],
                                                       "admin": {"do": "ban", "user": "7", "minutes": 10}}, sender="9",
                                     text="把阿凯禁言十分钟")
    e.decide(conv["id"])
    assert not [d for d in a.done if d[0] == "admin"] and "【群管理】" not in seen["system"]


def test_group_admin_when_allowed_only_for_the_owner(db, monkeypatch):
    from ai import permissions
    permissions.set_level("qq_group_admin", "self")
    store, conv, a, e, seen = _group(db, monkeypatch, {"action": "say", "messages": ["好"],
                                                       "admin": {"do": "ban", "user": "7", "minutes": 10}}, sender="9",
                                     text="把阿凯禁言十分钟")
    e.decide(conv["id"])
    assert ("admin", "614", "ban", {"user_id": "7", "minutes": 10}) in a.done and "【群管理】" in seen["system"]
    a.done.clear()
    say(monkeypatch, {"action": "say", "messages": ["不行哦"], "admin": {"do": "notice", "text": "全体放假"}})
    e.receive(a, kind="group", peer_id="614", sender_id="7", sender_name="阿凯", text="发个公告：全体放假",
              platform_msg_id="m20")
    e.decide(conv["id"])
    assert a.done == []                                       # not the owner asking: nothing

    permissions.set_level("qq_group_admin", "ask")
    told = []
    monkeypatch.setitem(permissions._RUNTIME, "rt", SimpleNamespace(send_system_text=told.append,
                                                                    channels=SimpleNamespace(get=lambda cid: a)))
    say(monkeypatch, {"action": "say", "messages": ["好"], "admin": {"do": "card", "user": "7", "card": "球王"}})
    e.receive(a, kind="group", peer_id="614", sender_id="9", sender_name="主人", text="把阿凯的群名片改成球王",
              platform_msg_id="m21")
    e.decide(conv["id"])
    pending = permissions.pending(db)
    assert a.done == [] and pending and pending[0]["ability"] == "qq_group_admin" and "球王" in told[0]
    result = permissions.decide(db, pending[0]["id"], True)
    assert result["status"] == "approved" and a.done == [("admin", "614", "card", {"user_id": "7", "card": "球王"})]


# ------------------------------------------------------------------------------------------------ her small QQ touches

class LifeAdapter(GroupAdapter):
    def __init__(self, feeds=()):
        super().__init__()
        self.feeds = list(feeds)

    def like(self, user_id, times=1):
        self.done.append(("like", str(user_id)))

    def qzone_feeds(self, count=10):
        return self.feeds

    def qzone_like(self, tid, uin):
        self.done.append(("qzone_like", tid))

    def qzone_comment(self, tid, uin, text):
        self.done.append(("comment", tid, text))

    def set_signature(self, text):
        self.done.append(("signature", text))


def _life(monkeypatch, adapter):
    from ai.social import qzone
    monkeypatch.setattr(qzone, "_qq", lambda runtime: adapter)


def test_small_touches_are_off_until_turned_on(db, monkeypatch):
    from ai.social import qq_life
    a = LifeAdapter()
    _life(monkeypatch, a)
    assert qq_life.tick(SimpleNamespace(db=db)) is None and a.done == []


def test_small_touches_like_and_comment_once(db, monkeypatch):
    import random
    from ai import permissions
    from ai.social import qq_life
    permissions.set_level("qq_touches", "self")
    now = time.time()
    feeds = [{"uin": 9, "tid": "t1", "appid": 311, "time": now - 600, "content": "今天跑了五公里"},
             {"uin": 5, "tid": "t2", "appid": 311, "time": now - 600, "content": "别人的"},
             {"uin": 9, "tid": "t3", "appid": 311, "time": now - 5 * 86400, "content": "很久以前的"}]
    a = LifeAdapter(feeds)
    _life(monkeypatch, a)
    say(monkeypatch, {"comment": "好厉害！"})
    out = qq_life.tick(SimpleNamespace(db=db), rng=SimpleNamespace(random=lambda: 0.0))      # the comment's chance: yes
    assert ("like", "9") in a.done and ("qzone_like", "t1") in a.done and ("comment", "t1", "好厉害！") in a.done
    assert not any(d[0] == "qzone_like" and d[1] in ("t2", "t3") for d in a.done)
    assert out["liked"] == 1 and out["commented"] == 1 and out["profile_likes"] == 1
    a.done.clear()
    qq_life.tick(SimpleNamespace(db=db), rng=random.Random(0))
    assert a.done == []                                       # done today / looked lately


def test_the_signature_changes_at_most_once_a_day(db, monkeypatch):
    from ai import permissions
    from ai.social import qq_life
    permissions.set_level("qq_signature", "self")
    assert permissions.choices("qq_signature") == ("self", "forbid")
    a = LifeAdapter()
    _life(monkeypatch, a)
    monkeypatch.setattr(qq_life, "_now", lambda db: datetime(2026, 11, 10, 10, 0))
    say(monkeypatch, {"change": True, "text": "今天也要开心呀 ☀"})
    qq_life.tick(SimpleNamespace(db=db))
    qq_life.tick(SimpleNamespace(db=db))
    assert a.done == [("signature", "今天也要开心呀 ☀")]
    monkeypatch.setattr(qq_life, "_now", lambda db: datetime(2026, 11, 11, 23, 30))
    qq_life.tick(SimpleNamespace(db=db))
    assert len(a.done) == 1                                   # not at night


# ------------------------------------------------------------------------------------------------ install

class OfflineHost(FakeHost):
    """SnowLuma's image is nowhere to download; maybe it is on the server already, maybe a saved copy is loaded."""

    def __init__(self, image=False):
        super().__init__()
        self.image = image
        self.streamed.append(["apt-get", "install", "-y"])     # docker is there already

    def run(self, cmd, **kw):
        if cmd[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=0 if self.image else 1, stdout="sha256:abc" if self.image else "", stderr="")
        return super().run(cmd, **kw)

    def stream(self, cmd, on_line, env):
        self.streamed.append(cmd)
        if cmd[:2] == ["docker", "load"]:
            self.image = True
            return 0, "Loaded image"
        return 1, "manifest unknown"


def _layout(tmp_path):
    return SimpleNamespace(root=tmp_path / "opt", data=tmp_path / "opt" / "data")


def _install(sl, layout, host, **kw):
    return sl.install(layout, {"user": None}, port_free=lambda p, h: True, run=host.run, stream=host.stream,
                      which=host.which, wait=0, check_desktop=lambda port: True, napcat_unit=layout.root / "none", **kw)


def test_the_image_on_the_server_is_used_as_it_is(tmp_path):
    from tools import violet_snowluma as sl
    host = OfflineHost(image=True)
    result = _install(sl, _layout(tmp_path), host)
    assert result["installed"] and not any(c[:2] == ["docker", "pull"] for c in host.streamed)


def test_a_saved_copy_is_loaded_instead_of_downloading(tmp_path):
    from tools import violet_snowluma as sl
    layout = _layout(tmp_path)
    saved = layout.root / sl.IMAGE_FILE_DIR / "snowluma-v2.2.1.tar.gz"
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b"\x1f\x8b fake")
    host = OfflineHost()
    _install(sl, layout, host)
    assert ["docker", "load", "-i", str(saved)] in host.streamed and not any(c[:2] == ["docker", "pull"] for c in host.streamed)
    other = tmp_path / "elsewhere.tar"
    other.write_bytes(b"tar")
    host = OfflineHost()
    _install(sl, layout, host, image_file=other)
    assert ["docker", "load", "-i", str(other)] in host.streamed
    with pytest.raises(sl.SnowlumaError, match="找不到镜像文件"):
        _install(sl, layout, OfflineHost(), image_file=tmp_path / "missing.tar.gz")


def test_without_any_image_the_error_says_how_to_use_a_saved_copy(tmp_path):
    from tools import violet_snowluma as sl
    host = OfflineHost()
    with pytest.raises(sl.SnowlumaError) as err:
        _install(sl, _layout(tmp_path), host)
    assert sl.IMAGE_FILE_DIR in str(err.value) and "--image-file /root/snowluma-v2.2.1.tar.gz" in str(err.value)
    pulled = [c[2] for c in host.streamed if c[:2] == ["docker", "pull"]]
    assert pulled[0] == "docker.m.daocloud.io/" + sl.IMAGE  # the mirror that still had it, first


def test_only_a_purge_removes_the_image(tmp_path):
    from tools import violet_snowluma as sl
    layout = _layout(tmp_path)
    host = OfflineHost(image=True)
    state = {"user": None}
    sl.install(layout, state, port_free=lambda p, h: True, run=host.run, stream=host.stream, which=host.which, wait=0,
               check_desktop=lambda port: True, napcat_unit=layout.root / "none")
    sl.uninstall(layout, dict(state), run=host.run, which=host.which, napcat_unit=layout.root / "none")
    assert not any(c[:2] == ["docker", "rmi"] for c in host.ran)
    removed = sl.uninstall(layout, dict(state), run=host.run, which=host.which, napcat_unit=layout.root / "none",
                           remove_image=True)
    assert ["docker", "rmi", sl.IMAGE] in host.ran and f"镜像 {sl.IMAGE}" in removed


def test_the_qq_command_takes_an_image_file(tmp_path, monkeypatch):
    from tools import violet_installer as vi
    seen = {}
    fake = SimpleNamespace(install=lambda layout, state, **kw: seen.update(kw) or {"installed": True}, SnowlumaError=RuntimeError,
                           uninstall=lambda *a, **k: [], IMAGE="img")
    monkeypatch.setattr(vi, "_snowluma_module", lambda: fake)
    monkeypatch.setattr(vi, "load_state", lambda layout: {"user": None})
    monkeypatch.setattr(vi, "save_state", lambda layout, state: None)
    monkeypatch.setattr(vi, "_is_root", lambda: True)
    monkeypatch.setattr(vi, "os", Proxy(os, name="posix"))                          # the root check runs on Linux
    vi.qq_command(vi.Options(root=tmp_path), "install", image_file=tmp_path / "x.tar.gz")
    assert seen["image_file"] == tmp_path / "x.tar.gz"
    out = vi.qq_command(vi.Options(root=tmp_path), "uninstall")
    assert "docker rmi img" in out["notes"][0]
