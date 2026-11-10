"""The channel "QQ" (2026-10-29): a personal QQ account through SnowLuma (OneBot v11; NapCat until the user found its
risk control too strict), installed by Violet itself (tools/violet_snowluma.py, the channels page's one-click install).
The channel keeps its stored type and id "napcat" from 0.7.8, so bindings, groups and history stay as they are.

One local WebSocket (SnowLuma's "wsServers" entry, published on 127.0.0.1 only, token) carries both directions: events
come in as JSON, actions go out as {"action", "params", "echo"} and their answers come back with the same echo. The
receive loop is the only reader; a sender waits for its echo. SnowLuma opens it only once QQ is logged in.

Who gets what:
  * a private message from the owner (a bound account, or "/bind <code>") -> the sink: Violet's normal one-to-one chat,
    exactly like WeChat / Telegram;
  * everything else -- group messages, private messages of other people -> Violet's social life (ai/social/engine.py),
    which stores, decides and answers on its own. Nothing of it reaches the owner's chat or memories.

The QR login happens in QQ's own window (ai/channels/vnc.py), not through here; probe_login() tells whether it is done.
"""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import requests

from ai.channels.base import ChannelAdapter, InboundMessage, NeedsLogin, Part
from ai.channels.normalize import clean_text, coalesce, safe_error
from ai.channels.onebot11 import parse_cq_string, segment_parts

MAX_DOWNLOAD = 10 * 1024 * 1024
logger = logging.getLogger("violet.snowluma")
# 2026-11-10: what QQ lets a normal member take back (2 minutes), a little less to be safe
RECALL_SECONDS = 110
# 2026-11-10: the emoji reactions she may put under a group message (set_msg_emoji_like takes the code point)
REACTIONS = ("👍", "👏", "😂", "🤣", "😭", "😘", "🥰", "😮", "🤔", "👀", "🎉", "🔥", "💪", "🙏", "😅", "❤", "🌹", "🐶", "🐱",
             "☕", "👌", "😁", "😳", "💯")


def reaction_id(emoji: Any) -> str | None:
    """The emoji_id of an allowed reaction ("👍" -> "128077"), or None."""
    text = str(emoji or "").replace("️", "").strip()
    return str(ord(text)) if len(text) == 1 and text in REACTIONS else None


class QQError(RuntimeError):
    pass


def accept_friend(db, payload: dict[str, Any]) -> str:
    """Accepts a friend request (ai/permissions "friend_add"; also the handler of an approval)."""
    from ai import permissions
    rt = permissions._RUNTIME.get("rt")
    manager = getattr(rt, "channels", None)
    adapter = manager.get(str(payload.get("channel") or "")) if manager is not None else None
    if adapter is None or not hasattr(adapter, "call") or not adapter.is_running():
        raise RuntimeError("QQ 没有连接")
    adapter.call("set_friend_add_request", {"flag": str(payload["flag"]), "approve": True})
    return f"好，我加了 QQ {payload.get('user_id')} 为好友。"


def account_token(config_dir: str | None) -> str | None:
    """The token SnowLuma really uses for the logged-in account: its config/onebot_<uin>.json (the newest one), the
    WebSocket server on the container's port 3001. None when there is none or the folder cannot be read."""
    if not config_dir or not Path(config_dir).is_absolute():
        return None
    try:
        files = sorted(Path(config_dir).glob("onebot_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return None
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            servers = (data.get("networks") or {}).get("wsServers") or []
        except (OSError, ValueError, AttributeError):
            continue
        for server in servers if isinstance(servers, list) else []:
            if isinstance(server, dict) and server.get("enabled", True) and str(server.get("port")) == "3001" \
                    and server.get("accessToken"):
                return str(server["accessToken"])
    return None


def connection() -> dict[str, Any] | None:
    """Where SnowLuma listens and its token, the VNC of QQ's window: what the installer exported
    (<data>/control/snowluma.json), with the account file's token when SnowLuma wrote its own."""
    from ai.root_tasks import snowluma_export
    info = snowluma_export()
    if info is None:
        return None
    token = account_token(info.get("config_dir"))
    return {**info, "ws_token": token} if token else info


def probe_login(info: dict[str, Any], ws_factory=None, timeout: float = 4.0) -> dict[str, Any] | None:
    """{"uin", "nick"} when QQ is logged in (SnowLuma answers get_login_info on its WebSocket), else None."""
    if ws_factory is None:
        from ai.channels.wsclient import WebSocket
        ws_factory = WebSocket.connect
    token = str(info.get("ws_token") or "")
    try:
        ws = ws_factory(info["ws_url"], timeout=timeout, headers={"Authorization": f"Bearer {token}"} if token else {})
    except Exception:                                    # noqa: BLE001 -- not logged in yet: nothing listens
        return None
    try:
        ws.send_text(json.dumps({"action": "get_login_info", "params": {}, "echo": "probe"}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = ws.recv(timeout=max(0.2, deadline - time.monotonic()))
            if text is None:
                continue
            try:
                frame = json.loads(text)
            except ValueError:
                continue
            if isinstance(frame, dict) and frame.get("echo") == "probe":
                data = frame.get("data") or {}
                uin = str(data.get("user_id") or "")
                return {"uin": uin, "nick": clean_text(str(data.get("nickname") or ""), 40)} if uin else None
    except Exception:                                    # noqa: BLE001
        return None
    finally:
        try:
            ws.close()
        except Exception:
            pass
    return None


def message_text(segments: list[dict[str, Any]], self_id: str) -> tuple[str, bool, str | None]:
    """(plain text for the record, mentions her?, the platform id of a quoted message)."""
    from ai.channels.normalize import render_parts
    mentioned, reply_to, parts = False, None, []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        kind, data = seg.get("type"), seg.get("data") or {}
        if kind == "at":
            qq = str(data.get("qq") or "")
            if qq and qq == self_id:
                mentioned = True
                parts.append(Part("mention", "你"))
                continue
            if qq == "all":
                parts.append(Part("mention", "全体成员"))
                continue
        if kind == "reply" and data.get("id") is not None:
            reply_to = str(data.get("id"))
            continue
        parts.extend(segment_parts([seg]))
    return render_parts(coalesce(parts)) if parts else "", mentioned, reply_to


def forward_text(nodes: list[Any], self_id: str = "", limit: int = 20) -> str:
    """A forwarded chat log (get_forward_msg's messages) as "名字：内容" lines."""
    lines = []
    for node in nodes[:limit] if isinstance(nodes, list) else []:
        if not isinstance(node, dict):
            continue
        sender = node.get("sender") or {}
        name = clean_text(str(sender.get("card") or sender.get("nickname") or node.get("user_id") or "某人"), 30)
        message = node.get("message") if node.get("message") is not None else node.get("content")
        segments = parse_cq_string(message) if isinstance(message, str) else (message or [])
        text = message_text([s for s in segments if isinstance(s, dict)], self_id)[0].replace("\n", " ").strip()
        if text:
            lines.append(f"{name}：{clean_text(text, 200)}")
    if isinstance(nodes, list) and len(nodes) > limit:
        lines.append(f"……（后面还有 {len(nodes) - limit} 条）")
    return "\n".join(lines)


class SnowLumaChannel(ChannelAdapter):
    channel_type = "napcat"                    # the stored type since 0.7.8 (see the module text)
    label = "QQ"
    capabilities = frozenset({"text", "image_in", "image_out", "typing"})

    def __init__(self, channel_id: str, config: dict[str, Any] | None = None, secrets: dict[str, str] | None = None,
                 data_dir: str | os.PathLike[str] | None = None, ws_factory=None, session: requests.Session | None = None,
                 **kwargs):
        super().__init__(channel_id, config, secrets, **kwargs)
        if ws_factory is None:
            from ai.channels.wsclient import WebSocket
            ws_factory = WebSocket.connect
        self.ws_factory = ws_factory
        self.session = session or requests.Session()
        base = Path(data_dir or os.environ.get("VIOLET_DATA_DIR") or "./data")
        self.media_dir = base / "media" / "channels" / channel_id
        self.runtime = None                    # set by the channel manager: the owner check and the social engine
        self._ws = None
        self._echo = itertools.count(1)
        self._waiting: dict[str, dict[str, Any]] = {}
        self._waiting_lock = threading.Lock()
        self.self_id = ""
        self.nickname = ""
        # 2026-11-02 (user: 看我的话只看了一半): the owner's chat quoted a message and she saw only
        # "[引用了一条消息]" -- what was said lately, by message id, so a quote shows its words
        self._said: "OrderedDict[str, tuple[bool, str]]" = OrderedDict()
        # 2026-11-10: what she sent lately (kind, peer, message id, when, text) -- for taking a message back
        from collections import deque
        self._sent: "deque[dict[str, Any]]" = deque(maxlen=200)
        self._no_typing = False                  # the implementation has no set_input_status: stop asking
        self.last_owner_peer = ""                # the owner's account that wrote last (the owner chat's tools)

    # -- connection -------------------------------------------------------------------------------------------------
    def _target(self) -> tuple[str, str]:
        url = str(self.config.get("ws_url") or "").strip()
        token = str(self.secrets.get("ws_token") or "")
        if not url:
            info = connection()
            if not info:
                raise NeedsLogin("还没有安装 QQ（SnowLuma）：在渠道页点“一键安装”")
            url, token = info["ws_url"], str(info.get("ws_token") or "")
        return url, token

    def prepare(self) -> None:
        self.on_stop()
        url, token = self._target()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            self._ws = self.ws_factory(url, timeout=15, headers=headers)
        except Exception as exc:                        # noqa: BLE001 -- QQ still starting / not logged in yet
            raise QQError("连不上 QQ（可能还在启动，或者还没有扫码登录）：" + safe_error(exc)) from None
        info = self._call_inline("get_login_info") or {}
        self.self_id = str(info.get("user_id") or "")
        self.nickname = clean_text(str(info.get("nickname") or ""), 40)
        if not self.self_id:
            raise NeedsLogin("QQ 还没有登录：在渠道页扫码")
        self.status_info.extra.update(self_id=self.self_id, nickname=self.nickname)
        self._set_state("connected", f"已登录 {self.nickname or 'QQ'}（{self.self_id}）")
        self._social()                                   # starts the social engine: the wake-ups she had planned
        self._sync_stickers_soon()
        self._catch_up_soon()

    def _catch_up_soon(self, delay: float = 8.0) -> None:
        """2026-11-10: after (re)connecting, the group messages she missed meanwhile (catch_up; the actions need the
        receive loop that starts when prepare() returns)."""
        from ai.settings import env_bool
        if self.runtime is None or getattr(self.runtime, "db", None) is None or not env_bool("VIOLET_QQ_CATCH_UP", True):
            return
        timer = threading.Timer(delay, lambda: self.is_running() and self.catch_up())
        timer.daemon = True
        timer.start()

    def catch_up(self, hours: float = 6.0, count: int = 20) -> int:
        """The messages of the groups she takes part in that came while she was offline (get_group_msg_history):
        stored as they were, with their own time, without a decision of their own -- she sees them with the next
        message. Returns how many were added. Never raises."""
        from datetime import datetime
        from ai import trace
        social = self._social()
        if social is None:
            return 0
        store, added = social.store, 0
        try:
            zone = datetime.fromisoformat(social.db.now()).tzinfo
            cutoff = time.time() - hours * 3600
            for conv in store.conversations(self.channel_id, kind="group"):
                if conv["mode"] == "off":
                    continue
                try:
                    data = self.call("get_group_msg_history", {"group_id": int(conv["peer_id"]), "count": int(count)},
                                     timeout=20) or {}
                except Exception as exc:                 # noqa: BLE001 -- an older implementation / no right
                    logger.info("no history of group %s: %s", conv["peer_id"], safe_error(exc))
                    continue
                for msg in (data.get("messages") or []) if isinstance(data, dict) else []:
                    if not isinstance(msg, dict) or str(msg.get("user_id") or "") == self.self_id:
                        continue
                    msg_id = str(msg.get("message_id") or "")
                    at = float(msg.get("time") or 0)
                    if not msg_id or at < cutoff or store.find_message(conv["id"], msg_id) is not None:
                        continue
                    message = msg.get("message")
                    segments = parse_cq_string(message) if isinstance(message, str) else (message or [])
                    text, mentioned, reply_to = message_text(segments, self.self_id)
                    sender = msg.get("sender") or {}
                    images = [s.get("data") or {} for s in segments if isinstance(s, dict) and s.get("type") == "image"]
                    if social.receive(self, kind="group", peer_id=str(conv["peer_id"]), sender_id=str(msg.get("user_id") or ""),
                                      sender_name=clean_text(str(sender.get("card") or sender.get("nickname") or ""), 40)
                                      or None, text=text, mentioned=mentioned, reply_to=reply_to, platform_msg_id=msg_id,
                                      images=images, sender_role=str(sender.get("role") or "") or None,
                                      created_at=datetime.fromtimestamp(at, tz=zone).isoformat(), catch_up=True):
                        added += 1
        except Exception:                                # noqa: BLE001
            logger.exception("catching up on QQ groups failed")
        if added:
            trace.step("补看漏掉的群消息", 渠道=self.channel_id, 条数=added)
        return added

    def _sync_stickers_soon(self, delay: float = 20.0) -> None:
        """The account's favourite stickers into her collection once connected (stickers.sync_qq; after a while,
        because the actions need the receive loop that starts when prepare() returns)."""
        db = getattr(self.runtime, "db", None)
        if db is None:
            return
        from ai import stickers
        timer = threading.Timer(delay, lambda: self.is_running() and stickers.sync_qq(db, self))
        timer.daemon = True
        timer.start()

    def on_stop(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        with self._waiting_lock:
            for slot in self._waiting.values():
                slot["error"] = "connection closed"
                slot["event"].set()

    def idle_delay(self) -> float:
        return 0.0                                      # recv() waits by itself

    def _read(self, timeout: float) -> dict[str, Any] | None:
        ws = self._ws
        if ws is None:
            raise QQError("not connected")
        text = ws.recv(timeout=timeout)
        if text is None:
            return None
        try:
            frame = json.loads(text)
        except ValueError:
            return None
        return frame if isinstance(frame, dict) else None

    def receive_once(self) -> int:
        frame = self._read(timeout=30)
        if frame is None:
            return 0
        if "echo" in frame and "post_type" not in frame:
            self._answer(frame)
            return 0
        return self.handle_event(frame)

    # -- actions --------------------------------------------------------------------------------------------------------
    def _answer(self, frame: dict[str, Any]) -> None:
        with self._waiting_lock:
            slot = self._waiting.get(str(frame.get("echo")))
        if slot is not None:
            slot["frame"] = frame
            slot["event"].set()

    def _call_inline(self, action: str, params: dict[str, Any] | None = None, timeout: float = 15.0) -> Any:
        """During prepare(): this thread is the reader, so it reads until its own answer arrives."""
        echo = f"v{next(self._echo)}"
        self._ws.send_text(json.dumps({"action": action, "params": params or {}, "echo": echo}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame = self._read(timeout=max(0.5, deadline - time.monotonic()))
            if frame is None:
                continue
            if str(frame.get("echo")) == echo:
                return self._result(action, frame)
            if "post_type" in frame:
                self.handle_event(frame)
        raise QQError(f"{action}: no answer")

    def call(self, action: str, params: dict[str, Any] | None = None, timeout: float = 20.0) -> Any:
        """An OneBot action from any thread; the receive loop hands back the answer."""
        ws = self._ws
        if ws is None:
            raise QQError("QQ 没有连接")
        echo = f"v{next(self._echo)}"
        slot = {"event": threading.Event()}
        with self._waiting_lock:
            self._waiting[echo] = slot
        try:
            ws.send_text(json.dumps({"action": action, "params": params or {}, "echo": echo}))
            if not slot["event"].wait(timeout):
                raise QQError(f"{action}: QQ 没有在 {timeout:.0f} 秒内回答")
            if slot.get("error"):
                raise QQError(f"{action}: {slot['error']}")
            return self._result(action, slot["frame"])
        finally:
            with self._waiting_lock:
                self._waiting.pop(echo, None)

    @staticmethod
    def _result(action: str, frame: dict[str, Any]) -> Any:
        if frame.get("status") == "failed" or frame.get("retcode") not in (0, None):
            raise QQError(f"{action}: retcode={frame.get('retcode')} "
                          + clean_text(str(frame.get("wording") or frame.get("message") or ""), 120))
        return frame.get("data")

    # -- events -----------------------------------------------------------------------------------------------------
    def handle_event(self, event: dict[str, Any]) -> int:
        post = event.get("post_type")
        if post == "meta_event":
            return 0
        if post == "request" and event.get("request_type") == "friend":
            # not in the receive loop: answering QQ / asking the owner waits for this loop
            threading.Thread(target=self._friend_request, args=(event,), daemon=True, name="qq-friend-request").start()
            return 0
        if post == "notice" and event.get("sub_type") == "poke":
            return self._poked(event)
        if post != "message":
            return 0
        if str(event.get("user_id") or "") == self.self_id:
            return 0                                     # her own messages (sent from another device)
        message = event.get("message")
        segments = parse_cq_string(message) if isinstance(message, str) else (message or [])
        if event.get("message_type") == "private" and self._for_owner(event, segments):
            unknown = [str((s.get("data") or {}).get("id")) for s in segments if isinstance(s, dict)
                       and s.get("type") == "reply" and (s.get("data") or {}).get("id") is not None
                       and str((s.get("data") or {}).get("id")) not in self._said]
            if (unknown or self._lookups(segments)) and self._ws is not None:
                # asking QQ for the quoted message / the voice's text / a forwarded log needs this receive loop:
                # done in a thread of its own
                threading.Thread(target=self._deliver_quoting, args=(event, segments, unknown[0] if unknown else None),
                                 daemon=True, name="qq-lookup").start()
                return 1
            return 1 if self.deliver(self._inbound(event, segments)) else 0
        social = self._social()
        if social is None:
            return 0
        if self._lookups(segments) and self._ws is not None:
            threading.Thread(target=lambda: self._to_social(social, event, self._enrich(event, segments)), daemon=True,
                             name="qq-lookup").start()
            return 1
        self._to_social(social, event, segments)
        return 1

    def _to_social(self, social, event: dict[str, Any], segments: list[dict[str, Any]]) -> None:
        text, mentioned, reply_to = message_text(segments, self.self_id)
        images = [s.get("data") or {} for s in segments if isinstance(s, dict) and s.get("type") == "image"]
        sender = event.get("sender") or {}
        group = event.get("message_type") == "group"
        try:
            social.receive(self, kind="group" if group else "private",
                           peer_id=str(event.get("group_id") if group else event.get("user_id")),
                           sender_id=str(event.get("user_id") or ""),
                           sender_name=clean_text(str(sender.get("card") or sender.get("nickname") or ""), 40) or None,
                           text=text, mentioned=mentioned, reply_to=reply_to,
                           platform_msg_id=str(event.get("message_id") or "") or None, images=images,
                           group_name=clean_text(str(event.get("group_name") or ""), 60) or None,
                           sender_role=(str(sender.get("role") or "") or None) if group else None)
        except Exception:                                # noqa: BLE001
            logger.exception("QQ message not handed to the social engine")
        self.status_info.received += 1

    # -- looking things up before a message is passed on (2026-11-10) ------------------------------------------------
    @staticmethod
    def _lookups(segments: list[Any]) -> bool:
        """A voice message to turn into text, or a forwarded chat log to open."""
        from ai.settings import env_bool
        for seg in segments or []:
            if not isinstance(seg, dict) or (seg.get("data") or {}).get("_text"):
                continue
            if seg.get("type") == "record" and env_bool("VIOLET_QQ_VOICE_TO_TEXT", True):
                return True
            if seg.get("type") == "forward" and (seg.get("data") or {}).get("id"):
                return True
        return False

    def _enrich(self, event: dict[str, Any], segments: list[Any]) -> list[Any]:
        """The segments with "_text" filled: QQ's own speech-to-text of a voice message (fetch_ptt_text), the messages
        of a forwarded chat log (get_forward_msg). What cannot be looked up stays as it was. Never raises."""
        from ai import trace
        out = []
        for seg in segments or []:
            if not isinstance(seg, dict):
                out.append(seg)
                continue
            kind, data = seg.get("type"), dict(seg.get("data") or {})
            try:
                if kind == "record" and not data.get("_text") and event.get("message_id") is not None:
                    answer = self.call("fetch_ptt_text", {"message_id": int(event["message_id"])}, timeout=20) or {}
                    data["_text"] = clean_text(str((answer or {}).get("text") or ""), 1000).strip() \
                        if isinstance(answer, dict) else ""
                    trace.step("QQ 语音转文字", 结果="转好了" if data["_text"] else "没有文字", 内容=trace.text(data["_text"]))
                elif kind == "forward" and not data.get("_text") and data.get("id"):
                    answer = self.call("get_forward_msg", {"id": str(data["id"])}, timeout=20) or {}
                    nodes = answer.get("messages") if isinstance(answer, dict) else answer
                    data["_text"] = forward_text(nodes or [], self.self_id)
                    trace.step("QQ 转发的聊天记录", 条数=len(nodes or []) if isinstance(nodes, list) else 0)
            except Exception as exc:                     # noqa: BLE001 -- shown as before: "a voice message" etc.
                logger.info("QQ %s not looked up: %s", kind, safe_error(exc))
                trace.step("QQ 消息查不到", 类型=kind, 原因=safe_error(exc))
            out.append({**seg, "data": data})
        return out

    def _poked(self, event: dict[str, Any]) -> int:
        """2026-10-30: someone 拍了拍 her -- like calling her: the owner's private poke reaches the owner's chat, any
        other goes to the social engine (mentioned)."""
        user = str(event.get("user_id") or event.get("sender_id") or "")
        if str(event.get("target_id") or "") != self.self_id or not user or user == self.self_id:
            return 0
        group = str(event.get("group_id") or "")
        key = f"poke:{user}:{event.get('time') or time.time():.0f}"
        if not group and self._for_owner(event, []):
            return 1 if self.deliver(InboundMessage(channel=self.channel_id, channel_type=self.channel_type, peer_id=user,
                                                    dedup_key=f"nc:{self.channel_id}:{key}",
                                                    parts=[Part("text", "[拍了拍你]")], message_id=key)) else 0
        social = self._social()
        if social is None:
            return 0
        social.receive(self, kind="group" if group else "private", peer_id=group or user, sender_id=user,
                       sender_name=None, text="", poke=True, platform_msg_id=None)
        return 1

    def poke(self, kind: str, peer_id: str, user_id: str) -> None:
        """拍一拍 someone back (send_poke; older OneBot implementations: group_poke / friend_poke)."""
        params = {"user_id": int(user_id)}
        if kind == "group":
            params["group_id"] = int(peer_id)
        try:
            self.call("send_poke", params, timeout=10)
        except QQError:
            self.call("group_poke" if kind == "group" else "friend_poke", params, timeout=10)

    def _for_owner(self, event: dict[str, Any], segments: list[dict[str, Any]]) -> bool:
        peer = str(event.get("user_id") or "")
        text = "".join(str((s.get("data") or {}).get("text") or "") for s in segments
                       if isinstance(s, dict) and s.get("type") == "text").strip()
        if text.startswith("/bind"):
            return True
        runtime = self.runtime
        identities = getattr(runtime, "identities", None)
        if identities is None:
            return True                                   # no social engine (tests): the old one-to-one behaviour
        try:
            return peer in identities.bound_peers(self.channel_id)
        except Exception:
            return False

    def _remember(self, message_id: Any, text: str | None, mine: bool) -> None:
        if message_id in (None, "") or not text:
            return
        self._said[str(message_id)] = (mine, clean_text(text, 200))
        self._said.move_to_end(str(message_id))
        while len(self._said) > 500:
            self._said.popitem(last=False)

    def _deliver_quoting(self, event: dict[str, Any], segments: list[dict[str, Any]], message_id: str | None) -> None:
        segments = self._enrich(event, segments)
        if message_id is None:
            try:
                self.deliver(self._inbound(event, segments))
            except Exception:                             # noqa: BLE001
                logger.exception("owner message not delivered")
            return
        try:
            data = self.call("get_msg", {"message_id": int(message_id)}, timeout=6) or {}
            quoted = data.get("message") or []
            quoted = parse_cq_string(quoted) if isinstance(quoted, str) else quoted
            text, _mentioned, _reply = message_text([s for s in quoted if isinstance(s, dict)
                                                     and s.get("type") != "reply"], self.self_id)
            sender = str((data.get("sender") or {}).get("user_id") or data.get("user_id") or "")
            self._remember(message_id, text or "[图片]", sender == self.self_id)
        except Exception as exc:                          # noqa: BLE001 -- too old / not found: shown as unknown
            logger.info("quoted QQ message %s not found: %s", message_id, safe_error(exc))
        try:
            self.deliver(self._inbound(event, segments))
        except Exception:                                 # noqa: BLE001
            logger.exception("owner message with a quote not delivered")

    def _parts(self, segments: list[dict[str, Any]]) -> list[Part]:
        parts: list[Part] = []
        for seg in segments or []:
            if isinstance(seg, dict) and seg.get("type") == "reply":
                mine, text = self._said.get(str((seg.get("data") or {}).get("id")), (False, ""))
                parts.append(Part("quote", text, {"mine": mine}))
            else:
                parts.extend(segment_parts([seg]))
        return parts

    def _inbound(self, event: dict[str, Any], segments: list[dict[str, Any]]) -> InboundMessage:
        sender = event.get("sender") or {}
        own = [s for s in segments or [] if not (isinstance(s, dict) and s.get("type") == "reply")]
        self._remember(event.get("message_id"), message_text(own, self.self_id)[0], False)
        self.last_owner_peer = str(event.get("user_id") or "") or self.last_owner_peer
        return InboundMessage(channel=self.channel_id, channel_type=self.channel_type, peer_id=str(event.get("user_id")),
                              dedup_key=f"nc:{self.channel_id}:{event.get('self_id')}:{event.get('message_id')}",
                              parts=coalesce(self._parts(segments)) or [Part("unsupported", "")],
                              message_id=str(event.get("message_id") or ""),
                              peer_name=clean_text(str(sender.get("nickname") or ""), 60) or None)

    def _social(self):
        if self.runtime is None:
            return None
        from ai.social.engine import engine_for
        return engine_for(self.runtime)

    def owner_ids(self) -> set[str]:
        identities = getattr(self.runtime, "identities", None)
        if identities is None:
            return set()
        try:
            return set(identities.bound_peers(self.channel_id))
        except Exception:
            return set()

    def admin_ids(self) -> set[str]:
        """2026-10-30: the accounts the owner made admins (not the owner; a few requests in groups)."""
        identities = getattr(self.runtime, "identities", None)
        if identities is None or not hasattr(identities, "admin_peers"):
            return set()
        try:
            return set(identities.admin_peers(self.channel_id))
        except Exception:
            return set()

    def _friend_request(self, event: dict[str, Any]) -> None:
        """2026-10-30 (user: 让她有同意加好友的能力，默认要经我同意): someone wants to add her. permissions
        "friend_add": self -> accepted, ask (default) -> the owner is asked ("同意 N"), forbid -> left unanswered."""
        from ai import permissions, trace
        flag = str(event.get("flag") or "")
        user = str(event.get("user_id") or "")
        comment = clean_text(str(event.get("comment") or ""), 80)
        db = getattr(self.runtime, "db", None)
        if not flag or not user or db is None:
            return
        level = permissions.level("friend_add")
        trace.step("好友申请", 渠道=self.channel_id, QQ=user, 验证消息=comment or None, 处理=permissions.LEVELS[level])
        payload = {"channel": self.channel_id, "flag": flag, "user_id": user, "comment": comment}
        about = self.stranger(user) if level != "forbid" else ""     # 2026-11-10: who it is (get_stranger_info)
        who = f"QQ {user}（{about}）" if about else f"QQ {user} "
        try:
            if level == "self":
                accept_friend(db, payload)
                permissions._tell_owner(f"{who}加了我好友，我同意了" + (f"（验证消息：{comment}）" if comment else "") + "。")
            elif level == "ask":
                permissions.request(db, "friend_add", f"{who}想加我好友" + (f"，验证消息：{comment}" if comment else "")
                                    + "。", payload)
        except Exception:                                  # noqa: BLE001
            logger.exception("friend request from %s not handled", user)

    def stranger(self, user_id: str) -> str:
        """2026-11-10: a few words about a QQ account (get_stranger_info): "昵称 小周，女，20 岁，签名：…"; "" when
        nothing is known. Never raises."""
        try:
            data = self.call("get_stranger_info", {"user_id": int(user_id)}, timeout=10) or {}
        except Exception as exc:                          # noqa: BLE001
            logger.info("no profile of QQ %s: %s", user_id, safe_error(exc))
            return ""
        if not isinstance(data, dict):
            return ""
        bits = []
        if data.get("nickname"):
            bits.append("昵称 " + clean_text(str(data["nickname"]), 30))
        sex = {"male": "男", "female": "女"}.get(str(data.get("sex") or ""))
        if sex:
            bits.append(sex)
        try:
            if 0 < int(data.get("age") or 0) < 120:
                bits.append(f"{int(data['age'])} 岁")
        except (TypeError, ValueError):
            pass
        sign = clean_text(str(data.get("long_nick") or data.get("longNick") or ""), 40).replace("\n", " ").strip()
        if sign:
            bits.append("签名：" + sign)
        return "，".join(bits)

    # -- sending ----------------------------------------------------------------------------------------------------
    def _note_sent(self, kind: str, peer_id: Any, data: Any, text: str) -> None:
        """2026-11-10: what she sent, for taking it back (recent_sent / recall)."""
        message_id = (data or {}).get("message_id") if isinstance(data, dict) else None
        if message_id not in (None, ""):
            self._sent.append({"kind": kind, "peer": str(peer_id), "message_id": str(message_id), "at": time.time(),
                               "text": clean_text(text or "", 80)})

    def send_typing(self, peer_id: str, on: bool) -> None:
        """2026-11-10: "对方正在输入…" in the owner's private chat (set_input_status 1 = typing, 0 = stop)."""
        from ai.settings import env_bool
        if self._no_typing or self._ws is None or not env_bool("VIOLET_QQ_TYPING", True):
            return
        try:
            self.call("set_input_status", {"user_id": int(peer_id), "event_type": 1 if on else 0}, timeout=3)
        except QQError as exc:
            if "retcode" in str(exc):                     # not offered by this implementation
                self._no_typing = True
        except Exception:                                 # noqa: BLE001 -- cosmetic
            pass

    def send_text(self, peer_id: str, text: str, client_id: str | None = None) -> dict[str, Any]:
        data = self.call("send_private_msg", {"user_id": int(peer_id), "message": [{"type": "text", "data": {"text": text}}]})
        self.status_info.sent += 1
        self._remember((data or {}).get("message_id"), text, True)
        self._note_sent("private", peer_id, data, text)
        return {"ok": True, "message_id": (data or {}).get("message_id")}

    def send_image(self, peer_id: str, path: str) -> dict[str, Any]:
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        data = self.call("send_private_msg", {"user_id": int(peer_id),
                                              "message": [{"type": "image", "data": {"file": f"base64://{encoded}"}}]},
                         timeout=60)
        self._note_sent("private", peer_id, data, "[图片]")
        return {"ok": True, "message_id": (data or {}).get("message_id")}

    @staticmethod
    def _sticker_segment(path: str) -> dict[str, Any]:
        """A picture QQ shows as a sticker (small, no preview frame), like one sent from the favourites."""
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return {"type": "image", "data": {"file": f"base64://{encoded}", "sub_type": 1, "summary": "[动画表情]"}}

    def send_sticker(self, peer_id: str, path: str) -> dict[str, Any]:
        data = self.call("send_private_msg", {"user_id": int(peer_id), "message": [self._sticker_segment(path)]},
                         timeout=60)
        self._remember((data or {}).get("message_id"), "[一个表情包]", True)
        self._note_sent("private", peer_id, data, "[表情包]")
        return {"ok": True, "message_id": (data or {}).get("message_id")}

    @staticmethod
    def _voice_segment(path: str) -> dict[str, Any]:
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return {"type": "record", "data": {"file": f"base64://{encoded}"}}

    def send_voice(self, peer_id: str, path: str) -> dict[str, Any]:
        """2026-10-31: a voice message (QQ's silk from ai/voice.py)."""
        data = self.call("send_private_msg", {"user_id": int(peer_id), "message": [self._voice_segment(path)]},
                         timeout=60)
        self._remember((data or {}).get("message_id"), "[一条语音]", True)
        self._note_sent("private", peer_id, data, "[语音]")
        return {"ok": True, "message_id": (data or {}).get("message_id")}

    def send_to(self, kind: str, peer_id: str, text: str | None, reply_to: str | None = None,
                sticker_path: str | None = None, image_path: str | None = None,
                voice_path: str | None = None) -> str | None:
        """The social engine's send: a group or someone's private chat, optionally quoting a message; text, one
        sticker of her collection, or (2026-10-30) one picture she drew."""
        message: list[dict[str, Any]] = []
        if reply_to:
            message.append({"type": "reply", "data": {"id": str(reply_to)}})
        if voice_path:                                   # a voice message stands alone (no quote)
            message = [self._voice_segment(voice_path)]
        elif sticker_path:
            message.append(self._sticker_segment(sticker_path))
        elif image_path:
            encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
            message.append({"type": "image", "data": {"file": f"base64://{encoded}"}})
        else:
            message.append({"type": "text", "data": {"text": text or ""}})
        timeout = 60 if (sticker_path or image_path or voice_path) else None
        extra = {"timeout": timeout} if timeout else {}
        if kind == "group":
            data = self.call("send_group_msg", {"group_id": int(peer_id), "message": message}, **extra)
        else:
            data = self.call("send_private_msg", {"user_id": int(peer_id), "message": message}, **extra)
        self.status_info.sent += 1
        self._note_sent(kind, peer_id, data, text or ("[语音]" if voice_path else "[表情包]" if sticker_path else "[图片]"))
        return str((data or {}).get("message_id") or "") or None

    # -- 2026-11-10: more of what QQ can do through SnowLuma (each one an OneBot action; errors are raised) ----------
    def recent_sent(self, kind: str, peer_id: str, within: float = RECALL_SECONDS) -> list[dict[str, Any]]:
        """Her messages to this chat that may still be taken back, newest first."""
        now = time.time()
        return [dict(s) for s in reversed(self._sent)
                if s["kind"] == kind and s["peer"] == str(peer_id) and now - s["at"] <= within]

    def recall(self, message_id: Any) -> None:
        """Take back one of her messages (delete_msg)."""
        self.call("delete_msg", {"message_id": int(message_id)}, timeout=10)
        self._sent = type(self._sent)((s for s in self._sent if s["message_id"] != str(message_id)),
                                      maxlen=self._sent.maxlen)

    def react(self, message_id: Any, emoji: str) -> None:
        """An emoji reaction under a group message (set_msg_emoji_like)."""
        emoji_id = reaction_id(emoji)
        if emoji_id is None:
            raise QQError(f"不能用这个表情回应：{emoji}")
        self.call("set_msg_emoji_like", {"message_id": int(message_id), "emoji_id": emoji_id, "set": True}, timeout=10)

    def like(self, user_id: Any, times: int = 1) -> None:
        """A like on someone's QQ profile card (send_like)."""
        self.call("send_like", {"user_id": int(user_id), "times": max(1, min(int(times), 10))}, timeout=10)

    def set_signature(self, text: str) -> None:
        """Her QQ 个性签名 (set_self_longnick)."""
        self.call("set_self_longnick", {"long_nick": clean_text(text, 60).replace("\n", " ").strip()}, timeout=15)

    def qzone_feeds(self, count: int = 10) -> list[dict[str, Any]]:
        """Her QQ 空间 friends' feed (get_qzone_feeds), newest first."""
        data = self.call("get_qzone_feeds", {"page_num": 1, "count": int(count)}, timeout=30) or {}
        feeds = data.get("feeds") if isinstance(data, dict) else data
        return [f for f in feeds or [] if isinstance(f, dict)]

    def qzone_like(self, tid: str, owner_uin: Any) -> None:
        self.call("like_qzone", {"tid": str(tid), "target_uin": int(owner_uin)}, timeout=20)

    def qzone_comment(self, tid: str, owner_uin: Any, text: str) -> None:
        self.call("comment_qzone", {"tid": str(tid), "target_uin": int(owner_uin),
                                    "content": clean_text(text, 200).replace("\n", " ").strip()}, timeout=20)

    def group_admin(self, group_id: Any, do: str, *, user_id: Any = None, minutes: Any = None, card: str = "",
                    message_id: Any = None, text: str = "") -> None:
        """A group admin's action (she must be an admin of that group; QQ refuses otherwise): ban (minutes, 0 =
        lift) / unban / card / essence / notice."""
        group = int(group_id)
        if do in ("ban", "unban"):
            seconds = 0 if do == "unban" else max(1, min(int(minutes or 10), 30 * 24 * 60)) * 60
            self.call("set_group_ban", {"group_id": group, "user_id": int(user_id), "duration": seconds}, timeout=15)
        elif do == "card":
            self.call("set_group_card", {"group_id": group, "user_id": int(user_id),
                                         "card": clean_text(card, 30).replace("\n", " ").strip()}, timeout=15)
        elif do == "essence":
            self.call("set_essence_msg", {"message_id": int(message_id)}, timeout=15)
        elif do == "notice":
            self.call("_send_group_notice", {"group_id": group, "content": clean_text(text, 500).strip()}, timeout=30)
        else:
            raise QQError(f"不认识的群管理操作：{do}")

    def groups(self) -> list[dict[str, Any]]:
        rows = self.call("get_group_list", {}) or []
        return [{"group_id": str(g.get("group_id")), "group_name": clean_text(str(g.get("group_name") or ""), 60),
                 "member_count": g.get("member_count")} for g in rows if isinstance(g, dict) and g.get("group_id")]

    def favourite_stickers(self, count: int = 48) -> list[str]:
        """The addresses of the account's own favourite stickers (QQ's 收藏的表情)."""
        rows = self.call("fetch_custom_face", {"count": int(count), "return_type": "url"}, timeout=30) or []
        if isinstance(rows, dict):
            rows = rows.get("urls") or rows.get("data") or []
        return [str(u) for u in rows if isinstance(u, str) and re.match(r"^https?://", u)]

    def publish_qzone(self, text: str) -> str:
        """A QQ 空间 post ("说说") visible to everyone, through SnowLuma's own send_qzone_msg. Returns its id."""
        data = self.call("send_qzone_msg", {"content": text, "ugc_right": 1}, timeout=60) or {}
        return str(data.get("tid") or "") if isinstance(data, dict) else ""

    def download(self, part: Part) -> Path:
        url = str(part.data.get("url") or "")
        if not re.match(r"^https?://", url):
            raise QQError("image has no download address")
        try:
            response = self.session.get(url, timeout=(10, 60), stream=True)
            response.raise_for_status()
            self.media_dir.mkdir(parents=True, exist_ok=True)
            target, size = self.media_dir / (hashlib.sha256(url.encode("utf-8")).hexdigest()[:24] + ".jpg"), 0
            with open(target, "wb") as fh:
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > MAX_DOWNLOAD:
                        raise QQError("image too large")
                    fh.write(chunk)
            return target
        except requests.RequestException as exc:
            raise QQError(safe_error(exc)) from None

    def probe(self) -> dict[str, Any]:
        info = self.call("get_login_info") or {}
        return {"ok": bool(info.get("user_id")), "self_id": info.get("user_id"), "nickname": info.get("nickname")}


def _register() -> None:
    from ai import permissions
    permissions.register("friend_add", accept_friend)


_register()
