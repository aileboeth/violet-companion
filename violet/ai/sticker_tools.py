"""Sticker tools of the Chat agent: looking through her collection, keeping one the owner sent.

2026-11-11 (audit B3): there is ONE way to send a sticker in the owner's chat -- a line "[[表情:N]]" in the reply.
It is what her own earlier stickers look like in the chat she is shown, it needs no extra model call, and it is how
voice (<voice>) and pictures ([[图片:N]]) go out too. The send_sticker tool (2026-10-30) did the same thing a second
way at the price of one more model round per sticker; with both on offer she sometimes wrote the label of a sticker
as text (0.7.64). Two tools stay, for what a line in the reply cannot do:

  find_stickers   look through her whole collection (also the ones without a description yet)
  save_sticker    keep a picture / sticker the owner sent recently (the same picture twice stays one sticker)

Only numbers of this collection can be sent; never a path or an URL (checked when the reply is sent).
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from typing import Any

from ai import stickers

TOOL_NAMES = {"find_stickers", "save_sticker"}

SCHEMAS = [
    {"type": "function", "function": {
        "name": "find_stickers",
        "description": "翻一下你收藏的表情包（全部，包括还没写说明的）。想发表情包、常用的那几个里又没有合适的时用它；query 可以写情绪或画面，比如“开心”“猫”。找到后在回复里单独一行写 [[表情:编号]] 就发出去了。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "可选：想找的情绪、画面或文字"}}}}},
    {"type": "function", "function": {
        "name": "save_sticker",
        "description": "把对方最近发来的图片或表情包收藏起来，以后可以发。对方让你收藏、或你自己很喜欢时用。",
        "parameters": {"type": "object", "properties": {
            "image_number": {"type": "integer", "description": "可选：消息里写的图片编号；不填就是最近一张"},
            "meaning": {"type": "string", "description": "可选：这个表情包适合什么时候用，20 字以内"}}}}},
]


def _answer(**payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _words(query: str) -> list[str]:
    parts = [p for p in re.split(r"[\s,，、。；;/|]+", query or "") if p]
    return [p[:12] for p in parts][:6]


def find(db, args: dict[str, Any]) -> str:
    if not stickers.enabled():
        return _answer(ok=False, error_code="STICKERS_OFF", message="表情包功能关着")
    rows = stickers.listing(db, enabled_only=True, limit=500)
    if not rows:
        return _answer(ok=True, count=0, stickers=[], message="收藏里还没有表情包。对方发来表情包时可以用 save_sticker 收藏。")
    words = _words(str(args.get("query") or ""))

    def score(row: dict[str, Any]) -> int:
        text = row.get("description") or ""
        return sum(1 for w in words if w in text)
    if words:
        rows = sorted(rows, key=lambda r: (-score(r), -(r.get("use_count") or 0), -r["id"]))
    else:
        rows = sorted(rows, key=lambda r: (-(r.get("use_count") or 0), -r["id"]))
    shown = [{"number": r["id"], "description": r.get("description") or "（还没有说明，看不到画面）",
              "used": r.get("use_count") or 0} for r in rows[:20]]
    return _answer(ok=True, count=len(rows), stickers=shown,
                   message="想发哪个，就在回复里单独一行写 [[表情:编号]]。" + ("没有对得上的：这次就不发表情包，不要硬配一个意思不对的。"
                                                          if words and not score(rows[0]) else ""))


def save(db, ctx: dict[str, Any], args: dict[str, Any]) -> str:
    from ai import trace
    if not stickers.enabled():
        return _answer(ok=False, error_code="STICKERS_OFF", message="表情包功能关着，收藏不了")
    from ai.images.media import MediaAssetStore
    from ai.memory_model import parse_time
    since = (parse_time(db.now()) - timedelta(hours=24)).isoformat()
    number = args.get("image_number")
    if number not in (None, ""):
        try:
            number = int(number)
        except (TypeError, ValueError):
            number = None
        row = db.fetch_one("SELECT * FROM media_assets WHERE id = ? AND direction = 'inbound' AND status = 'ready' "
                           "AND created_at >= ?", (number, since)) if number is not None else None
    else:
        row = db.fetch_one("SELECT * FROM media_assets WHERE direction = 'inbound' AND status = 'ready' AND created_at >= ? "
                           "ORDER BY id DESC LIMIT 1", (since,))
    if row is None:
        trace.step("收藏表情包", 来源="你发的", 结果="没收藏", 原因="最近一天没有收到能收藏的图片")
        return _answer(ok=False, error_code="NO_RECENT_IMAGE", message="最近一天没有收到图片或表情包，没有东西可以收藏")
    try:
        path = MediaAssetStore(db).path_of(row)
        meaning = str(args.get("meaning") or "").strip()[:60] or None
        saved = stickers.add_file(db, path, source="owner", description=meaning or row.get("caption"),
                                  file_key=f"asset:{row['id']}")
    except Exception as exc:                                  # noqa: BLE001 -- told to the model, never raised
        trace.step("收藏表情包", 来源="你发的", 结果="没收藏", 原因=str(exc)[:120])
        return _answer(ok=False, error_code="SAVE_FAILED", message=f"没收藏成功：{str(exc)[:120]}")
    if not saved.get("description"):
        described = stickers.describe(path)
        if described:
            saved = stickers.update(db, saved["id"], description=described) or saved
    trace.step("收藏表情包", 来源="你发的", 编号=saved["id"], 结果="已收藏", 说明=saved.get("description"))
    return _answer(ok=True, number=saved["id"], description=saved.get("description"),
                   message=f"收藏好了，编号 {saved['id']}，以后在回复里单独一行写 [[表情:{saved['id']}]] 就能发。")
