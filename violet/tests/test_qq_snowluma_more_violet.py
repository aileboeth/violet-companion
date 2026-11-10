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
    assert str(err.value).splitlines()[0].startswith("这台服务器组装不了 QQ 容器")       # the way out first
    assert any(line.startswith("组装：") for line in str(err.value).splitlines())         # then what was tried
    pulled = [c[2] for c in host.streamed if c[:2] == ["docker", "pull"] and c[2].endswith(sl.IMAGE)]
    assert pulled[0] == "docker.m.daocloud.io/" + sl.IMAGE  # the mirror that still had it, first


def test_a_mirror_answering_with_a_web_page_stays_one_short_line(tmp_path):
    from tools import violet_snowluma as sl
    page = "<html><script>var a = '" + "x" * 3000 + "';</script></html>"
    with pytest.raises(sl.SnowlumaError) as err:
        sl.pull(lambda cmd, **kw: None, lambda cmd, on_line, env: (1, page), lambda line: None, {})
    assert all(len(line) <= 200 for line in str(err.value).splitlines())


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


# ------------------------------------------------------------------------------------------------ built here (2026-11-10)
# user: "用原本 SnowLuma 下载 QQ 的那个方法吧，这样哪里都可以下载了" -- Violet builds the container itself

class BuildHost(OfflineHost):
    """No image anywhere; Docker Hub has the base image; `docker build` works (or fails when told to)."""

    def __init__(self, build_ok=True, buildx=True):
        super().__init__()
        self.have, self.build_ok, self.buildx, self.env = set(), build_ok, buildx, None

    def run(self, cmd, **kw):
        if cmd[:3] == ["docker", "image", "inspect"]:
            ok = cmd[-1] in self.have
            return SimpleNamespace(returncode=0 if ok else 1, stdout="sha256:abc" if ok else "", stderr="")
        if cmd[:3] == ["docker", "buildx", "version"]:
            return SimpleNamespace(returncode=0 if self.buildx else 1, stdout="", stderr="")
        return super().run(cmd, **kw)

    def stream(self, cmd, on_line, env):
        from tools import violet_snowluma as sl
        self.streamed.append(cmd)
        if cmd[:2] == ["docker", "pull"] and cmd[2] == sl.BASE_IMAGE:
            self.have.add(sl.BASE_IMAGE)
            return 0, "Status: Downloaded"
        if cmd[:2] == ["docker", "build"]:
            self.env = env
            for line in ("#5 [1/4] FROM node", "QQ: trying https://qqdl.gtimg.cn/x", "#9 [4/4] RUN setcap"):
                on_line(line)
            if self.build_ok:
                self.have.add(sl.BUILT_IMAGE)
                return 0, "naming to violet/snowluma"
            return 1, "QQ 3.2.32_260812 could not be downloaded from any source"
        return 1, "manifest unknown"


def test_without_an_image_violet_builds_the_container_itself(tmp_path):
    from tools import violet_snowluma as sl
    host, steps, state = BuildHost(), [], {"user": None}
    result = sl.install(_layout(tmp_path), state, port_free=lambda p, h: True, run=host.run, stream=host.stream,
                        which=host.which, wait=0, check_desktop=lambda port: True, napcat_unit=tmp_path / "none",
                        apt=lambda: "http://mirrors.ustc.edu.cn", machine=lambda: "x86_64",
                        progress=lambda pct, step: steps.append((pct, step)))
    build = next(c for c in host.streamed if c[:2] == ["docker", "build"])
    assert "--progress=plain" in build and f"BASE_IMAGE={sl.BASE_IMAGE}" in build
    assert "APT_MIRROR=http://mirrors.ustc.edu.cn" in build and build[-1] == str(sl.BUILD_DIR)
    assert host.env["DOCKER_BUILDKIT"] == "1"
    assert ["docker", "pull", sl.BASE_IMAGE] in host.streamed                        # only the base image is pulled
    assert not any(c[:2] == ["docker", "pull"] and sl.IMAGE in c[2] for c in host.streamed)
    run = next(c for c in host.ran if c[:2] == ["docker", "run"])
    assert run[-1] == sl.BUILT_IMAGE and state["snowluma"]["image"] == sl.BUILT_IMAGE and state["snowluma"]["base_pulled"]
    assert result["installed"] and "组装" in state["snowluma"]["source"]
    assert any("第 4/4 步" in s for _, s in steps) and any(s.startswith("下载 QQ：") for _, s in steps)
    # a reinstall uses the image it built (no second build)
    before = len(host.streamed)
    sl.install(_layout(tmp_path), state, port_free=lambda p, h: True, run=host.run, stream=host.stream, which=host.which,
               wait=0, check_desktop=lambda port: True, napcat_unit=tmp_path / "none", machine=lambda: "x86_64")
    assert not any(c[:2] == ["docker", "build"] for c in host.streamed[before:])
    # a purge removes what Violet built and the base image it pulled
    removed = sl.uninstall(_layout(tmp_path), dict(state), run=host.run, which=host.which, napcat_unit=tmp_path / "none",
                           remove_image=True)
    assert f"镜像 {sl.BUILT_IMAGE}" in removed and f"镜像 {sl.BASE_IMAGE}" in removed


def test_a_failed_build_falls_back_to_the_old_image_and_says_why(tmp_path):
    from tools import violet_snowluma as sl
    host = BuildHost(build_ok=False, buildx=False)
    with pytest.raises(sl.SnowlumaError) as err:
        _install(sl, _layout(tmp_path), host, machine=lambda: "x86_64")
    build = next(c for c in host.streamed if c[:2] == ["docker", "build"])
    assert "--progress=plain" not in build and host.env["DOCKER_BUILDKIT"] == "0"   # an old Docker without buildx
    text = str(err.value)
    assert "组装：组装 QQ 容器失败" in text and "could not be downloaded" in text and "下载：" in text
    assert any(c[:2] == ["docker", "pull"] and sl.IMAGE in c[2] for c in host.streamed)   # the old image was tried


def test_no_build_on_arm_or_when_switched_off(tmp_path, monkeypatch):
    from tools import violet_snowluma as sl
    assert sl.build_blocked(lambda: "aarch64").startswith("只有 x64")
    assert sl.build_blocked(lambda: "AMD64") is None
    monkeypatch.setenv("VIOLET_QQ_BUILD", "0")
    assert "VIOLET_QQ_BUILD" in sl.build_blocked(lambda: "x86_64")
    host = BuildHost()
    with pytest.raises(sl.SnowlumaError, match="设置里关了组装"):
        _install(sl, _layout(tmp_path), host, machine=lambda: "x86_64")
    assert not any(c[:2] == ["docker", "build"] for c in host.streamed)


def test_the_fastest_package_source_is_used():
    from tools import violet_snowluma as sl
    # the user's server: deb.debian.org 64 KB/s, USTC 4.7 MB/s, Aliyun 1.1 MB/s
    measured = {"": 63847, "http://mirrors.ustc.edu.cn": 4758532, "http://mirrors.aliyun.com": 1086757}
    assert sl.apt_mirror(speed=lambda base: measured[base]) == "http://mirrors.ustc.edu.cn"
    assert sl.apt_mirror(speed=lambda base: {"": 9e6}.get(base, 1e6)) == ""              # abroad: Debian itself
    assert sl.apt_mirror(speed=lambda base: 0.0) == ""                                   # nothing answered: as it is


def test_a_purge_removes_the_parts_before_the_program_and_names_what_failed(tmp_path, monkeypatch):
    """2026-11-10, the user's server: `/opt/violet/violet uninstall --purge` deleted <root>/app first; the relay / QQ
    modules are imported from <root>/app/tools when needed, so every part failed to import and was skipped in silence
    (the QQ container, the phone relay and their folders stayed)."""
    from tools import violet_installer as vi
    root = tmp_path / "opt"
    layout = vi.Layout(root)
    layout.app.mkdir(parents=True)
    (layout.app / "tools").mkdir()
    vi.save_state(layout, {"service": "none", "user": None})
    seen = {}

    def fake_purge_system(layout_, state, run=None, linux_root=None, problems=None):
        seen["app_still_there"] = layout_.app.exists()
        problems.append("QQ（SnowLuma）（ModuleNotFoundError: No module named 'violet_snowluma'）")
        return ["容器 violet-snowluma"]
    monkeypatch.setattr(vi, "purge_system", fake_purge_system)
    result = vi.uninstall(vi.Options(root=root), purge=True)
    assert seen["app_still_there"]                                  # the parts first, while their code is there
    assert "容器 violet-snowluma" in result["removed"] and not layout.app.exists()
    assert result["problems"] and "没能删掉" in result["note"] and "QQ（SnowLuma）" in result["note"]


def test_a_part_that_fails_is_named_not_skipped(tmp_path, monkeypatch):
    from tools import violet_installer as vi
    layout = vi.Layout(tmp_path / "opt")

    def broken():
        raise ModuleNotFoundError("No module named 'violet_snowluma'")
    for name in ("_relay_module", "_searxng_module", "_voice_module"):
        monkeypatch.setattr(vi, name, lambda: SimpleNamespace(uninstall=lambda *a, **k: []))
    monkeypatch.setattr(vi, "_snowluma_module", broken)
    monkeypatch.setattr(vi, "remove_units", lambda layout, run: [])
    problems: list[str] = []
    vi.purge_system(layout, {}, run=lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
                    linux_root=True, problems=problems)
    assert len(problems) == 1 and problems[0].startswith("QQ（SnowLuma）") and "violet_snowluma" in problems[0]


def test_the_fastest_qq_address_goes_first():
    """0.7.66 (user: "就算用官方下载出问题了，为什么不用加速下载呢")."""
    from tools import violet_snowluma as sl
    sources = sl.qq_sources()
    tencent, github, ip8217, jackyu = sources
    # the user's server: Tencent 4.5 MB/s, GitHub nothing, 8.217 0.9 MB/s, jackyu slower
    measured = {tencent: 4493441, github: 0.0, ip8217: 877048, jackyu: 120000}
    assert sl.qq_order(speed=lambda s: measured[s]) == [tencent, ip8217, jackyu, github]
    # Tencent did not answer this time: the mirror first, never GitHub itself while it hangs
    assert sl.qq_order(speed=lambda s: {ip8217: 877048, jackyu: 120000}.get(s, 0.0))[:2] == [ip8217, jackyu]
    assert sl.qq_order(speed=lambda s: 0.0) == sources                      # nothing answered: the Dockerfile's order


def test_the_qq_download_shows_mb_speed_and_source(tmp_path):
    from tools import violet_snowluma as sl
    steps, clock = [], iter([100.0, 105.0, 110.0])

    class Host(BuildHost):
        def stream(self, cmd, on_line, env):
            self.streamed.append(cmd)
            if cmd[:2] == ["docker", "build"]:
                self.env = env
                for line in ("#6 [2/6] RUN set -eux; apt-get update", "#6 12.1 Get:41 http://mirrors.ustc.edu.cn/debian x",
                             "#7 [3/6] RUN set -eux; arch=...", "#7 0.3 QQ: trying https://github.com/Rodert/qq-versions/x",
                             "#7 31.0 QQ: https://github.com/Rodert/qq-versions/x failed or stalled",
                             "#7 31.1 QQ: trying http://8.217.135.233:9999/https://github.com/Rodert/qq-versions/x",
                             "#7 36.1 QQ: got 10485760 bytes", "#7 41.1 QQ: got 31457280 bytes",
                             "#7 46.1 QQ: got 185599488 bytes", "#10 [6/6] WORKDIR /app/data"):
                    on_line(line)
                self.have.add(sl.BUILT_IMAGE)
                return 0, "done"
            return super().stream(cmd, on_line, env)
    host = Host()
    host.have.add(sl.BASE_IMAGE)
    order = ["http://8.217.135.233:9999/https://github.com/Rodert/qq-versions/releases/download/x", "https://qqdl/y"]
    sl.build_image(host.run, host.stream, {}, lambda pct, step: steps.append((pct, step)),
                   pull_base=lambda: "", mirror=lambda: "", order=lambda: order, clock=lambda: next(clock))
    build = next(c for c in host.streamed if c[:2] == ["docker", "build"])
    assert "QQ_SOURCES=" + " ".join(order) in build                         # the measured order goes to the build
    texts = [s for _, s in steps]
    assert "组装 QQ 容器：安装系统软件，已下载 41 个软件包" in texts
    assert "下载 QQ：连接 github.com" in texts and "下载 QQ：github.com 没连上或太慢，换下一个" in texts
    assert "下载 QQ：已下 10 / 约 177 MB · 来自 8.217.135.233:9999 → github.com" in texts
    assert "下载 QQ：已下 30 / 约 177 MB · 每秒 4096 KB · 来自 8.217.135.233:9999 → github.com" in texts
    percents = [p for p, _ in steps]
    assert percents == sorted(percents)                                      # the bar only moves forward
    qq = [p for p, s in steps if s.startswith("下载 QQ：已下")]
    assert qq[0] < qq[-1] and qq[-1] <= 30 + 54 * 3 // 6                     # QQ fills step 3 of 6, no further


def test_install_sh_falls_back_to_the_same_mirrors_as_the_updates():
    """0.7.66 (the user's server in China: install.sh came through a mirror, the archive then timed out on GitHub)."""
    import re
    from pathlib import Path
    from tools import violet_sources
    script = (Path(__file__).resolve().parents[1] / "install.sh").read_text(encoding="utf-8")
    line = re.search(r'https://github\.com/\*\) bases\+=\((.+)\) ;;', script).group(1)
    prefixes = [p.split("$release_base")[0] for p in re.findall(r'"([^"]+)"', line)]
    assert prefixes == [prefix for _key, prefix, _label in violet_sources.MIRRORS]
    assert 'for b in "$base" "${bases[@]}"' in script                      # the checksum: that address first, then others
    assert 'fetch_quick "$b/violet.tar.gz.sha256"' in script              # ... with the same time limits
    assert "--connect-timeout" in script and "--speed-limit" in script and "--max-time" in script


def test_official_images_are_under_library_on_a_mirror():
    from tools import violet_snowluma as sl
    assert sl.mirror_ref("docker.m.daocloud.io/", "node:24.22.1-bookworm-slim") == \
        "docker.m.daocloud.io/library/node:24.22.1-bookworm-slim"
    assert sl.mirror_ref("", "node:24.22.1-bookworm-slim") == "node:24.22.1-bookworm-slim"
    assert sl.mirror_ref("docker.1ms.run/", sl.IMAGE) == "docker.1ms.run/" + sl.IMAGE


def test_the_build_folder_holds_what_the_dockerfile_needs():
    import re
    import tarfile
    from tools import violet_snowluma as sl
    folder = sl.BUILD_DIR
    docker = (folder / "Dockerfile").read_text(encoding="utf-8")
    copied = re.search(r"^COPY (.+) /tmp/violet/$", docker, re.M).group(1).split()
    assert all((folder / name).is_file() for name in copied)
    sources = re.search(r'ARG QQ_SOURCES="([^"]+)"', docker).group(1).split()
    assert sources[0].startswith("https://qqdl.gtimg.cn/") and "Rodert/qq-versions" in sources[1]   # Tencent first
    # 0.7.66: qq-versions also through Violet's GitHub mirrors, and a stalled source is left (it hung for 15 minutes)
    from tools import violet_sources
    for _key, prefix, _label in violet_sources.MIRRORS:
        assert any(s == prefix + sources[1] for s in sources[2:])
    assert "--speed-limit" in docker and "--max-time" in docker
    assert "QQ_AMD64_SHA256=d085dd89397225061eb9f194308f688129818ed445777e97a4a0a16e13d7b0e8" in docker
    assert "sha256sum -c" in docker and f"ARG BASE_IMAGE={sl.BASE_IMAGE}" in docker
    with tarfile.open(folder / "snowluma-runtime-2.2.1-linux-x64.tar.gz") as tar:
        names = {n.lstrip("./") for n in tar.getnames()}
    assert {"index.mjs", "native/snowluma-linux-x64.node", "native/snowluma-linux-x64.so",
            "native/websocket-linux-x64.node"} <= names
    assert "MIT License" in (folder / "LICENSE.SnowLuma.txt").read_text(encoding="utf-8")
    for script in ("start.sh", "supervisord.conf", "Dockerfile"):              # Linux files: no BOM, no CRLF
        raw = (folder / script).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw
    start = (folder / "start.sh").read_text(encoding="utf-8")
    assert "qqpatch.gtimg.cn" in start and "x11vnc" in start and start.startswith("#!/usr/bin/env bash")

