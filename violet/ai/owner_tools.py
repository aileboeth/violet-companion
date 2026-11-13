"""Chat's own tools for the owner's chat that work on Violet's records: looking a memory up, reminders.

2026-11-11 (audit A3, rework): a reminder used to be set only when the owner's message matched the Chinese words
"提醒我" plus a date the code could read (今天 / 明天 / 后天, "十分钟后") -- "叫我起床", "周五下午三点", every other
language set nothing while the reply said "好，到时候提醒你". Now she sets it herself, like every other action: the
model reads the request in whatever words and language, the code checks the time and stores it. What the tool returns
is what really happened, so the reply cannot claim more.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ai.memory_model import parse_time, recall_search_text

QUERY_MEMORY_SCHEMA = {"type": "function", "function": {
    "name": "query_memory",
    "description": ("查你们以前聊过的事（记忆库）。对方提到以前说过的事、计划、经历、喜好，而背景里的记忆没有、你又想不起来时用；"
                    "普通闲聊不用查。返回的 status：active 清楚，expired / archived 已经模糊或很久以前，forgotten 几乎忘了。"),
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "要找的事，例如：他上周说的考试、他喜欢的奶茶"},
        "limit": {"type": "integer", "description": "最多几条，默认 5"}},
        "required": ["query"]}}}

SET_REMINDER_SCHEMA = {"type": "function", "function": {
    "name": "set_reminder",
    "description": ("对方明确要你到某个时间提醒 / 叫 / 喊他做某件事时用（“明早七点叫我起床”“周五下午三点提醒我交报告”"
                    "“半小时后喊我”）。到时间系统会让你发消息提醒。只有调用成功才算设好了。"
                    # 2026-11-13 (0.7.73 virtual test): "明早记得叫我一声" -> she only asked "几点叫，还是七点半？", he
                    # went to sleep, and nothing she set would have woken him
                    "对方没说几点：你们以前定过的时间（背景“提醒”那一段或记忆里有）就按那个设好，回复里说出时间，对方可以改；"
                    "从来没定过、也猜不到，才先问几点——只问不设的话，对方睡着了就没人叫他了。"
                    "对方只是提到一件事、没有要你提醒，不用设——但你自己想说“到时候叫你 / 提醒你”，就必须先调用它设好；"
                    "没调用就不要说会提醒。"),
    "parameters": {"type": "object", "properties": {
        "content": {"type": "string", "description": "要提醒的事，用对方的说法，简短，例如：起床、交报告（到点会直接念给对方听）"},
        "due_at": {"type": "string", "description": "提醒的时间，按背景里的“现在”算出来，格式 YYYY-MM-DD HH:MM（24 小时制，对方所在的时区）"}},
        "required": ["content", "due_at"]}}}

CANCEL_REMINDER_SCHEMA = {"type": "function", "function": {
    "name": "cancel_reminder",
    "description": "对方说不用提醒了、取消某个提醒时用。编号是背景里“提醒”那一段每条前面的 #编号。",
    "parameters": {"type": "object", "properties": {
        "reminder_id": {"type": "integer", "description": "要取消的提醒的编号"}},
        "required": ["reminder_id"]}}}

NOTE_FEELING_SCHEMA = {"type": "function", "function": {
    "name": "note_feeling",
    "description": ("你自己的心情因为对方刚说的话明显变了的时候用，这样之后一段时间你还记得这份心情："
                    "被真心夸了、谢了、哄开心了 → better；被认真地骂、嫌弃、冷落、伤到了 → worse。"
                    "开玩笑、打情骂俏、互损不算；对方自己心情不好也不算。大多数时候不用调用。"),
    "parameters": {"type": "object", "properties": {
        "change": {"type": "string", "enum": ["much_better", "better", "worse", "much_worse"]},
        "reason": {"type": "string", "description": "一句话，你自己的说法，为什么（20 字以内）"}},
        "required": ["change", "reason"]}}}
FEELING_DELTAS = {"much_better": 0.5, "better": 0.3, "worse": -0.35, "much_worse": -0.6}

SCHEMAS = [QUERY_MEMORY_SCHEMA, SET_REMINDER_SCHEMA, CANCEL_REMINDER_SCHEMA, NOTE_FEELING_SCHEMA]
REMINDER_TOOLS = {"set_reminder", "cancel_reminder"}
# tools that change something: a reply that used one is never thrown away or written a second time
ACTION_TOOLS = REMINDER_TOOLS


def _dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False)


def query_memory(db, args: dict[str, Any]) -> dict[str, Any]:
    query = recall_search_text(args.get("query", ""))
    if not query:
        return {"memories": []}
    db.refresh_event_states()
    try:
        limit = max(1, min(10, int(args.get("limit", 5))))
    except (TypeError, ValueError):
        limit = 5
    memories = db.search_event_memories(query=query, category=args.get("category"), limit=limit,
                                        min_similarity=0.35, recall=True)
    for memory in memories:
        db.mark_event_memory_recalled(memory["id"], weight_bonus=0.0)     # bookkeeping only
    return {"memories": [
        {"id": m["id"], "content": m["content"], "category": m["category"], "kind": m.get("kind"),
         "progress": m.get("progress"), "event_time": m.get("event_time"), "status": m["status"],
         "memory_strength": round(m["forget_weight"], 2), "similarity": round(m.get("similarity", 0), 2),
         "retrieval": m.get("retrieval")}
        for m in memories if m["status"] != "deleted"]}


def has_memories(db) -> bool:
    from ai import test_mode
    return not test_mode.active() and db.has_searchable_event_memories()


def _when(text: Any):
    text = str(text or "").strip().replace("/", "-").replace("T", " ")
    if len(text) == 16:                                    # YYYY-MM-DD HH:MM
        text += ":00"
    return parse_time(text.replace(" ", "T", 1)) if text else None


def set_reminder(db, context: dict[str, Any], args: dict[str, Any]) -> str:
    content = str(args.get("content") or "").strip()[:200]
    if not content:
        return _dumps({"ok": False, "message": "没有设：要提醒的事是空的。"})
    due = _when(args.get("due_at"))
    now = parse_time(db.now())
    if due is None:
        return _dumps({"ok": False, "message": "没有设：时间看不懂，要写成 YYYY-MM-DD HH:MM。时间不清楚就先问对方。"})
    if due <= now:
        return _dumps({"ok": False, "message": f"没有设：{due:%Y-%m-%d %H:%M} 已经过去了（现在是 {now:%Y-%m-%d %H:%M}）。"
                                               "按现在的时间重新算，或者问对方是哪一天。"})
    due_at = due.isoformat(timespec="seconds")              # as every stored time: local, with its offset
    evidence = str(context.get("_user_message") or "")[:300] or None
    try:
        result = db.add_commitment(content, due_at, kind="reminder", evidence=evidence,
                                   source_turn_id=context.get("turn_id"), created_by="user")
        row = db.get_commitment(int(result["commitment"]["id"]))          # what is really stored
    except Exception:                                                     # noqa: BLE001
        return _dumps({"ok": False, "message": "没有设：保存时出错了。照实告诉对方这次没设上。"})
    if not row or row["status"] not in ("pending", "sending"):
        return _dumps({"ok": False, "message": "没有设：保存时出错了。照实告诉对方这次没设上。"})
    context.setdefault("_actions", []).append({"tool": "set_reminder", "commitment_id": row["id"]})
    at = parse_time(row["due_at"])
    label = f"{at.month}月{at.day}日 {at:%H:%M}" if at else row["due_at"]
    if result["action"] == "duplicate":
        return _dumps({"ok": True, "reminder_id": row["id"], "message": f"同样的提醒之前已经设好了（{label} 提醒“{row['content']}”），这次没有重复设。"})
    return _dumps({"ok": True, "reminder_id": row["id"], "due_at": row["due_at"],
                   "message": f"设好了：{label} 提醒“{row['content']}”。回复里简短确认时间和事情就行。"})


def cancel_reminder(db, context: dict[str, Any], args: dict[str, Any]) -> str:
    try:
        number = int(args.get("reminder_id"))
    except (TypeError, ValueError):
        return _dumps({"ok": False, "message": "没有取消：编号不对。"})
    row = db.get_commitment(number)
    if not row or row["status"] != "pending":
        return _dumps({"ok": False, "message": "没有取消：没有这个编号的待提醒事项（可能已经提醒过或取消过）。"})
    db.update_commitment(number, status="cancelled", last_error="cancelled by user")
    context.setdefault("_actions", []).append({"tool": "cancel_reminder", "commitment_id": number})
    return _dumps({"ok": True, "message": f"取消了：“{row['content']}”这个提醒不会再发。"})


def note_feeling(db, context: dict[str, Any], args: dict[str, Any]) -> str:
    from ai import mood
    delta = FEELING_DELTAS.get(str(args.get("change") or ""))
    if delta is None:
        return _dumps({"ok": False, "message": "没有记：change 只能是 much_better / better / worse / much_worse。"})
    if context.get("_felt"):
        return _dumps({"ok": True, "message": "这一轮已经记过了。"})
    context["_felt"] = True
    reason = re.sub(r"\s+", " ", str(args.get("reason") or "")).strip()[:40] or None
    mood.shift(db, delta, reason)
    return _dumps({"ok": True, "message": "记下了。照这份心情自然地回复就行，不用说“我记下了”。"})
