"""What a social decision may do on QQ besides talking (2026-11-10, user: "将 SnowLuma 里面的一些功能集成上去").

  react    {"message": "<id>", "emoji": "👍"} -- an emoji reaction under someone's group message (a light "I saw it"
           that costs less than a message); one per decision, never under her own message
  recall   "<id>" -- take back one of her own messages, only within two minutes (what QQ allows a normal member)
  admin    {"do": ban | unban | card | essence | notice, ...} -- a group admin's action. Only when the permission
           "qq_group_admin" is not "不允许" (default), only in a group, only when the owner asked in this decision;
           "先问你" waits for the owner's "同意 N" like the other approvals. She must be an admin there (QQ refuses
           otherwise -- the error is in the trace).

Every step is in the troubleshooting log; nothing here raises (a decision's words were already sent).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from ai import trace

logger = logging.getLogger("violet.social.qq_actions")
ADMIN_DOS = {"ban": "禁言", "unban": "解除禁言", "card": "改群名片", "essence": "设精华", "notice": "发群公告"}


def _id(value: Any) -> str:
    return str(value or "").strip().lstrip("#＃").strip()


def react(store, conv: dict[str, Any], value: Any, adapter) -> bool:
    if not isinstance(value, dict) or conv["kind"] != "group" or adapter is None or not hasattr(adapter, "react"):
        return False
    message_id, emoji = _id(value.get("message")), str(value.get("emoji") or "").strip()
    row = store.find_message(conv["id"], message_id) if message_id else None
    if row is None or row["is_self"]:
        trace.step("表情回应", 会话=conv["id"], 消息=message_id or None, 结果="没点", 原因="不是这里别人的消息")
        return False
    try:
        adapter.react(message_id, emoji)
    except Exception as exc:                                 # noqa: BLE001
        trace.step("表情回应", 会话=conv["id"], 消息=message_id, 表情=emoji, 结果="失败", 原因=str(exc)[:120])
        return False
    trace.step("表情回应", 会话=conv["id"], 消息=message_id, 表情=emoji, 结果="点了")
    return True


def recall(db, store, conv: dict[str, Any], value: Any, adapter, seconds: float = 115.0) -> bool:
    message_id = _id(value)
    if not message_id or adapter is None or not hasattr(adapter, "recall"):
        return False
    row = store.find_message(conv["id"], message_id)
    try:
        fresh = row is not None and datetime.fromisoformat(db.now()[:19]) - datetime.fromisoformat(
            str(row["created_at"])[:19]) <= timedelta(seconds=seconds)
    except ValueError:
        fresh = False
    if row is None or not row["is_self"] or not fresh:
        trace.step("撤回消息", 会话=conv["id"], 消息=message_id, 结果="没撤回", 原因="不是她两分钟内发的消息")
        return False
    try:
        adapter.recall(message_id)
    except Exception as exc:                                 # noqa: BLE001
        trace.step("撤回消息", 会话=conv["id"], 消息=message_id, 结果="失败", 原因=str(exc)[:120])
        return False
    db.execute("UPDATE social_messages SET content = ? WHERE id = ?", ("[你撤回了这条] " + str(row["content"] or ""), row["id"]))
    trace.step("撤回消息", 会话=conv["id"], 消息=message_id, 结果="撤回了")
    return True


def admin_allowed() -> bool:
    from ai import permissions
    return permissions.level("qq_group_admin") != "forbid"


def admin_note(conv: dict[str, Any], obey_owner: bool) -> str | None:
    """What she is told about group admin actions (None when they are not allowed here)."""
    if conv["kind"] != "group" or not obey_owner or not admin_allowed():
        return None
    from ai.social.prompts import ADMIN_ACTIONS
    return ADMIN_ACTIONS


def _clean(conv: dict[str, Any], value: dict[str, Any]) -> dict[str, Any] | None:
    do = str(value.get("do") or "").strip().lower()
    if do not in ADMIN_DOS:
        return None
    out: dict[str, Any] = {"group_id": str(conv["peer_id"]), "do": do}
    if do in ("ban", "unban", "card"):
        user = _id(value.get("user"))
        if not user.isdigit():
            return None
        out["user_id"] = user
    if do == "ban":
        try:
            out["minutes"] = max(1, min(int(float(value.get("minutes") or 10)), 30 * 24 * 60))
        except (TypeError, ValueError):
            out["minutes"] = 10
    if do == "card":
        out["card"] = " ".join(str(value.get("card") or "").split())[:30]
    if do == "essence":
        message = _id(value.get("message"))
        if not message.lstrip("-").isdigit():
            return None
        out["message_id"] = message
    if do == "notice":
        out["text"] = str(value.get("text") or "").strip()[:500]
        if not out["text"]:
            return None
    return out


def describe(action: dict[str, Any]) -> str:
    what = ADMIN_DOS[action["do"]]
    if action["do"] == "ban":
        return f"在群 {action['group_id']} 把 QQ {action['user_id']} {what} {action['minutes']} 分钟"
    if action["do"] in ("unban", "card"):
        tail = f"改成“{action.get('card') or '（清空）'}”" if action["do"] == "card" else ""
        return f"在群 {action['group_id']} 给 QQ {action['user_id']} {what}{tail}"
    if action["do"] == "essence":
        return f"在群 {action['group_id']} 把消息 {action['message_id']} {what}"
    return f"在群 {action['group_id']} {what}：{action['text'][:80]}"


def _run(adapter, action: dict[str, Any]) -> None:
    adapter.group_admin(action["group_id"], action["do"], user_id=action.get("user_id"), minutes=action.get("minutes"),
                        card=action.get("card") or "", message_id=action.get("message_id"), text=action.get("text") or "")


def admin(db, conv: dict[str, Any], value: Any, adapter, owner_asked: bool) -> str | None:
    """Carry out (or ask about) one admin action. Returns what happened, or None when nothing was tried."""
    if not isinstance(value, dict) or not value:
        return None
    from ai import permissions
    level = permissions.level("qq_group_admin")
    action = _clean(conv, value) if conv["kind"] == "group" else None
    if action is None or level == "forbid" or not owner_asked or adapter is None or not hasattr(adapter, "group_admin"):
        why = ("设置里不允许" if level == "forbid" else "不是主人这次要求的" if not owner_asked
               else "QQ 没有连接" if adapter is None else "看不懂要做什么")
        trace.step("群管理", 会话=conv["id"], 结果="没做", 原因=why, 内容=trace.text(value))
        return None
    summary = describe(action)
    if level == "ask":
        number = permissions.request(db, "qq_group_admin", f"我想{summary}。", {"channel": adapter.channel_id, **action})
        trace.step("群管理", 会话=conv["id"], 结果="等主人批准", 编号=number, 内容=summary)
        return f"等主人批准（{number}）"
    try:
        _run(adapter, action)
    except Exception as exc:                                 # noqa: BLE001
        trace.step("群管理", 会话=conv["id"], 结果="失败", 原因=str(exc)[:160], 内容=summary)
        return f"没做成：{str(exc)[:120]}"
    trace.step("群管理", 会话=conv["id"], 结果="做了", 内容=summary)
    return "做了"


def _approved(db, payload: dict[str, Any]) -> str:
    """ai/permissions.py: the owner said yes to an admin action."""
    from ai.permissions import _RUNTIME
    runtime = _RUNTIME.get("rt")
    manager = getattr(runtime, "channels", None)
    adapter = manager.get(str(payload.get("channel") or "")) if manager is not None else None
    if adapter is None or not adapter.is_running():
        raise RuntimeError("QQ 没有连接")
    action = {k: payload[k] for k in ("group_id", "do", "user_id", "minutes", "card", "message_id", "text") if k in payload}
    _run(adapter, action)
    return f"好，{describe(action)}，做好了。"


def _register() -> None:
    from ai import permissions
    permissions.register("qq_group_admin", _approved)


_register()
