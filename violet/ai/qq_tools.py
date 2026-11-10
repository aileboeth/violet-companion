"""Tools of the Chat agent in the owner's QQ chat (2026-11-10, SnowLuma's actions; user: "将 SnowLuma 里面的一些功能
集成上去").

  qq_recall   take back what she sent to the owner in the last two minutes (the newest one, or the last few)
  qq_like     a like on the owner's QQ profile card, when he asks for it

Offered only when this turn is on the QQ channel (SnowLuma) and it is connected; never in proactive messages.
"""

from __future__ import annotations

import json
from typing import Any

TOOL_NAMES = {"qq_recall", "qq_like"}

SCHEMAS = [
    {"type": "function", "function": {
        "name": "qq_recall",
        "description": "撤回你刚才在 QQ 上发给对方的消息（只能撤回两分钟之内的）。对方让你撤回、或者你发错了 / 发重了时用。",
        "parameters": {"type": "object", "properties": {
            "count": {"type": "integer", "description": "撤回最近几条，默认 1，最多 5"}}}}},
    {"type": "function", "function": {
        "name": "qq_like",
        "description": "给对方的 QQ 名片点个赞。对方让你给他点赞时用。",
        "parameters": {"type": "object", "properties": {}}}},
]


def _answer(**payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def adapter():
    """The QQ channel of this turn when it is connected, else None."""
    from ai.channels.context import current
    from ai.permissions import _RUNTIME
    context = current()
    if context.channel_type != "napcat":
        return None
    manager = getattr(_RUNTIME.get("rt"), "channels", None)
    found = manager.get(context.channel) if manager is not None else None
    return found if found is not None and found.is_running() and hasattr(found, "recall") else None


def available() -> bool:
    try:
        return adapter() is not None
    except Exception:                                       # noqa: BLE001
        return False


def _owner_peer(qq) -> str:
    peer = str(getattr(qq, "last_owner_peer", "") or "")
    if peer:
        return peer
    owners = sorted(qq.owner_ids()) if hasattr(qq, "owner_ids") else []
    return str(owners[0]) if owners else ""


def recall(args: dict[str, Any]) -> str:
    from ai import trace
    qq = adapter()
    if qq is None:
        return _answer(ok=False, error_code="QQ_UNAVAILABLE", message="现在不是在 QQ 上聊天，或者 QQ 没有连接")
    try:
        count = max(1, min(int(args.get("count") or 1), 5))
    except (TypeError, ValueError):
        count = 1
    recent = qq.recent_sent("private", _owner_peer(qq))[:count]
    if not recent:
        return _answer(ok=False, error_code="NOTHING_TO_RECALL", message="两分钟之内没有发给对方的消息，撤回不了了")
    done, failed = [], []
    for item in recent:
        try:
            qq.recall(item["message_id"])
            done.append(item["text"])
        except Exception as exc:                            # noqa: BLE001
            failed.append(str(exc)[:80])
    trace.step("撤回消息", 渠道="QQ（主人）", 撤回=len(done), 失败=len(failed) or None)
    if not done:
        return _answer(ok=False, error_code="RECALL_FAILED", message="QQ 没让撤回：" + "；".join(failed))
    return _answer(ok=True, recalled=done, message=f"撤回了 {len(done)} 条。回复里不用重复那几句的内容。")


def like(args: dict[str, Any]) -> str:
    from ai import trace
    qq = adapter()
    peer = _owner_peer(qq) if qq is not None else ""
    if qq is None or not peer:
        return _answer(ok=False, error_code="QQ_UNAVAILABLE", message="现在不是在 QQ 上聊天，或者 QQ 没有连接")
    try:
        qq.like(peer, 1)
    except Exception as exc:                                # noqa: BLE001 -- e.g. the day's likes are used up
        trace.step("QQ 小互动", 做什么="给你的名片点赞", 结果="失败", 原因=str(exc)[:120])
        return _answer(ok=False, error_code="LIKE_FAILED", message=f"QQ 没让点：{str(exc)[:120]}")
    trace.step("QQ 小互动", 做什么="给你的名片点赞", 结果="点了")
    return _answer(ok=True, message="点好了。")
