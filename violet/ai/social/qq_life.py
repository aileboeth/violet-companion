"""Small things she does on QQ by herself (2026-11-10, user: "SnowLuma不是还有些功能吗一起做了").

Both are off until the owner turns them on (permissions "qq_touches" / "qq_signature", 自己做 / 不允许) -- the account
was kicked offline once, so nothing here runs unless he wants it, and every action is rare:

  qq_touches     once a day a like on the owner's QQ profile card (send_like); his new QQ 空间 posts (the last two
                 days) get a like, and now and then one short comment written by the model (at most one a round)
  qq_signature   at most once a day, between 09:00 and 22:59: she may change her QQ 个性签名 to fit her mood and
                 her life lately (one model call; she may also keep it)

Called by the social engine's background round (every ~25 minutes, SocialEngine.start_posting). What was done is
kept in social_done, so nothing is done twice. Never raises.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from typing import Any

from ai import trace

logger = logging.getLogger("violet.social.qq_life")
FEED_EVERY_MINUTES = 110
COMMENT_CHANCE = 0.35

COMMENT_RULES = """【给主人的说说写评论】
你的主人在 QQ 空间发了一条说说，你想在下面评论一句。像熟人那样随口一句：短（20 字以内），自然，跟着他说的内容；
不提你们私聊过的事（别人都看得到），不 @ 人，不发链接。不知道说什么、或者不适合评论时就不评论。
【你的回答】只输出一个 JSON 对象：{"comment": "要评论的话（不评论就留空）"}"""

SIGN_RULES = """【QQ 个性签名】
你在想要不要改一下自己的 QQ 个性签名。签名是你此刻的一句心情或状态，像真人那样：短（20 字以内），口语，可以带一个 emoji；
素材只能来自你自己的心情和最近的生活，不写主人的事，不发链接、联系方式。现在的签名还挺好、或者没什么想写的，就不改。
【你的回答】只输出一个 JSON 对象：{"change": true 或 false, "text": "新的签名（不改时留空）"}"""


def _done(db, kind: str, key: str) -> bool:
    return db.fetch_one("SELECT 1 FROM social_done WHERE kind = ? AND key = ?", (kind, key)) is not None


def _mark(db, kind: str, key: str, detail: str = "") -> None:
    db.execute("INSERT OR IGNORE INTO social_done (kind, key, detail, created_at) VALUES (?, ?, ?, ?)",
               (kind, key, detail[:200], db.now()))


def _now(db) -> datetime:
    return datetime.fromisoformat(db.now()[:19])


def _owners(adapter) -> list[str]:
    try:
        return sorted(str(o) for o in adapter.owner_ids() if str(o).isdigit())
    except Exception:                                        # noqa: BLE001
        return []


def like_owner(db, adapter) -> int:
    """The day's like on each of the owner's accounts. Returns how many were given."""
    day, given = _now(db).date().isoformat(), 0
    for uin in _owners(adapter):
        if _done(db, "profile_like", f"{day}:{uin}"):
            continue
        try:
            adapter.like(uin, 1)
        except Exception as exc:                             # noqa: BLE001 -- e.g. not friends yet: tried tomorrow
            trace.step("QQ 小互动", 做什么="给你的名片点赞", 结果="失败", 原因=str(exc)[:120])
            _mark(db, "profile_like", f"{day}:{uin}", f"failed: {str(exc)[:120]}")
            continue
        _mark(db, "profile_like", f"{day}:{uin}")
        trace.step("QQ 小互动", 做什么="给你的名片点赞", 结果="点了")
        given += 1
    return given


def _comment(db, text: str) -> str:
    from ai.social import llm
    from ai.social.engine import _persona_text
    try:
        decision, _model, _ = llm.decide(_persona_text(db, "napcat") + "\n\n" + COMMENT_RULES,
                                         f"主人的说说：{text[:400]}\n\n决定评论什么，只输出 JSON。")
    except llm.SocialLLMError as exc:
        logger.info("no qzone comment: %s", exc)
        return ""
    return " ".join(str(decision.get("comment") or "").split())[:60]


def owner_posts(db, adapter, rng: random.Random | None = None) -> dict[str, int]:
    """Likes (and now and then one comment) on the owner's new QQ 空间 posts."""
    rng = rng or random
    out = {"liked": 0, "commented": 0}
    now = _now(db)
    last = db.fetch_one("SELECT MAX(created_at) AS at FROM social_done WHERE kind = 'feed_look'")
    try:
        if last and last["at"] and now - datetime.fromisoformat(str(last["at"])[:19]) < timedelta(minutes=FEED_EVERY_MINUTES):
            return out
    except ValueError:
        pass
    owners = set(_owners(adapter))
    if not owners:
        return out
    _mark(db, "feed_look", now.isoformat(timespec="minutes"))
    try:
        feeds = adapter.qzone_feeds(10)
    except Exception as exc:                                 # noqa: BLE001
        trace.step("QQ 小互动", 做什么="看你的说说", 结果="失败", 原因=str(exc)[:120])
        return out
    since = (now - timedelta(days=2)).timestamp()
    for feed in feeds:
        uin, tid = str(feed.get("uin") or ""), str(feed.get("tid") or "")
        if uin not in owners or not tid or int(feed.get("appid") or 311) != 311 or float(feed.get("time") or 0) < since:
            continue
        if _done(db, "qzone_like", tid):
            continue
        try:
            adapter.qzone_like(tid, uin)
        except Exception as exc:                             # noqa: BLE001
            trace.step("QQ 小互动", 做什么="给你的说说点赞", 结果="失败", 原因=str(exc)[:120])
            _mark(db, "qzone_like", tid, f"failed: {str(exc)[:120]}")
            continue
        _mark(db, "qzone_like", tid)
        out["liked"] += 1
        trace.step("QQ 小互动", 做什么="给你的说说点赞", 结果="点了", 说说=trace.text(feed.get("content") or ""))
        text = str(feed.get("content") or "").strip()
        if text and not out["commented"] and rng.random() < COMMENT_CHANCE:
            comment = _comment(db, text)
            if comment:
                try:
                    adapter.qzone_comment(tid, uin, comment)
                    _mark(db, "qzone_comment", tid, comment)
                    out["commented"] += 1
                    trace.step("QQ 小互动", 做什么="评论你的说说", 结果="评论了", 内容=comment)
                except Exception as exc:                     # noqa: BLE001
                    trace.step("QQ 小互动", 做什么="评论你的说说", 结果="失败", 原因=str(exc)[:120])
    return out


def signature(db, adapter) -> str | None:
    """Maybe a new 个性签名 (at most once a day). Returns it when changed."""
    now = _now(db)
    day = now.date().isoformat()
    if not 9 <= now.hour <= 22 or _done(db, "signature", day):
        return None
    _mark(db, "signature", day)
    from ai import mood
    from ai.social import llm
    from ai.social.engine import _persona_text
    from ai.social.qzone import _material
    lines = [f"现在是 {db.now()[:16].replace('T', ' ')}。", mood.note(db) or "",
             "", "【你最近的生活】", *(_material(db) or ["（这几天没有记下特别的事）"]), "", "决定改不改，只输出 JSON。"]
    try:
        decision, _model, _ = llm.decide(_persona_text(db, "napcat") + "\n\n" + SIGN_RULES, "\n".join(lines))
    except llm.SocialLLMError as exc:
        logger.info("no signature: %s", exc)
        return None
    text = " ".join(str(decision.get("text") or "").split())[:40]
    if not decision.get("change") or not text:
        trace.step("QQ 签名", 结果="这次不改")
        return None
    try:
        adapter.set_signature(text)
    except Exception as exc:                                 # noqa: BLE001
        trace.step("QQ 签名", 结果="失败", 原因=str(exc)[:120])
        return None
    db.execute("UPDATE social_done SET detail = ? WHERE kind = 'signature' AND key = ?", (text, day))
    trace.step("QQ 签名", 结果="改了", 内容=text)
    return text


def tick(runtime, rng: random.Random | None = None) -> dict[str, Any] | None:
    """One background round. Never raises."""
    try:
        from ai import permissions, test_mode
        if test_mode.active():
            return None
        touches, sign = permissions.level("qq_touches") == "self", permissions.level("qq_signature") == "self"
        if not touches and not sign:
            return None
        from ai.social.qzone import _qq
        adapter = _qq(runtime)
        if adapter is None or not adapter.is_running():
            return None
        db, out = runtime.db, {}
        if touches:
            out["profile_likes"] = like_owner(db, adapter)
            out.update(owner_posts(db, adapter, rng))
        if sign:
            out["signature"] = signature(db, adapter)
        return out
    except Exception:                                        # noqa: BLE001
        logger.exception("qq life round failed")
        return None
