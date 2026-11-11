"""Her own mood (2026-10-30, user: 人设更有灵动性; idea from dsh-aurabot-qq).

One number, -1 (upset) .. +1 (happy), and what caused it. Words said to her move it -- praise and thanks up, scolding
and insults down -- and it drifts back to calm with a half-life (VIOLET_MOOD_HALF_LIFE_HOURS, 3).

Who decides that it moved (2026-11-11, audit A12):
  * the owner's chat: she does, with the note_feeling tool (ai/owner_tools.py). The reply model reads the tone -- a
    teasing "笨蛋" is not an insult, "谢谢" out of politeness is not delight -- which keywords never could (0.7.60
    had to special-case 打情骂俏, and the cause line "主人对你说了「笨蛋」" then argued with what she could read
    herself);
  * groups and other people's private chats: keywords (feel, below) -- many messages, mostly not addressed to her.

What it changes, only while it is clearly there (|mood| >= 0.3):
  * a short line in the owner's chat and in the social prompts: how she feels and why, and that it may colour her
    tone (never an attack on anyone, never a reason to refuse help);
  * stickers: in a good mood she is told she may use one more readily;
  * groups: the chance that she starts talking by herself in a quiet group goes up / down.
Off: VIOLET_MOOD=0. Test mode never reads or writes it.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

from ai.settings import env_bool, env_float

SCHEMA = """
CREATE TABLE IF NOT EXISTS mood_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    value REAL NOT NULL DEFAULT 0,
    cause TEXT,
    updated_at TEXT NOT NULL
);
"""

PRAISE = re.compile(r"(谢谢|谢啦|多谢|感谢|辛苦了|真棒|好棒|太棒|厉害|好厉害|真聪明|聪明|可爱|好可爱|喜欢你|爱你|想你|真好|好乖|"
                    r"么么|抱抱|贴贴|夸夸|你最好|做得好|干得好|nice|awesome|love you|thank)", re.I)
# 2026-11-03: "工作好烦" / "烦死了" is the owner's own mood, not scolding her -- only "你好烦" / "烦死你了" counts
SCOLD = re.compile(r"(笨蛋|蠢|傻[逼瓜子]|白痴|弱智|废物|垃圾|滚|闭嘴|你(好|真|很|太)?烦|烦死你|讨厌你|恶心|没用|有病|脑残|"
                   r"别吵|闭麦|shut up|stupid|idiot)", re.I)
# 2026-11-09 (real chat, user: 打情骂俏可能触发的骂人): "现在凌晨十二点，我有步数才奇怪，笨蛋" and "笨蛋还不去睡觉" were
# counted as scolding -- her mood went to 有点不高兴 for half an hour and every reply said "我气还没消呢". Between people
# who are close, 笨蛋 / 傻瓜 / 讨厌 / 你好烦 are mostly teasing (嗔怪 / 打情骂俏) -- whether a word hurts depends on the tone
# and the relationship, which a word list cannot see and the reply model can (it reads the message itself). So these
# words alone move her mood only a little (several in a row still add up); the hard insults count as before, and less
# when the same message is plainly affectionate.
MILD_SCOLD = re.compile(r"(笨蛋|蠢|傻瓜|傻子|你(好|真|很|太)?烦|烦死你|讨厌你|没用)", re.I)
# sweet words, pet names, laughing, a teasing tone or a sticker read as playful (the vision description is in the text)
AFFECTION = re.compile(r"(宝贝|宝宝|臭宝|宝儿|亲爱的|老婆|老公|媳妇|爱你|喜欢你|想你|么么|mua|亲亲|抱抱|贴贴|"
                       r"(小|大|臭)笨蛋|笨蛋蛋|哈哈|嘿嘿|嘻嘻|hhh|233|开玩笑|逗你|[~～]|"
                       r"俏皮|调侃|卖萌|撒娇|搞怪|戏谑|得意|捉弄)", re.I)
MILD_DELTA = -0.1
SAD = re.compile(r"(难过|伤心|好累|崩溃|想哭|哭了|失眠|好烦啊|压力好大|不开心|郁闷)")
THRESHOLD = 0.3


def scold_strength(text: str) -> float:
    """How much a message scolds her: 0 (not at all) .. 1 (a hard insult, said in earnest)."""
    found = [m.group(0) for m in SCOLD.finditer(text or "")]
    if not found:
        return 0.0
    affectionate = bool(AFFECTION.search(text))
    if all(MILD_SCOLD.fullmatch(w) for w in found):
        return 0.0 if affectionate else abs(MILD_DELTA) / 0.35
    return 0.4 if affectionate else 1.0


def _quoted(text: str) -> str:
    """The scolding words themselves, for the cause line ("主人说你「笨蛋」")."""
    words = []
    for m in SCOLD.finditer(text or ""):
        if m.group(0) not in words:
            words.append(m.group(0))
    return "、".join(f"「{w}」" for w in words[:2])


def enabled() -> bool:
    return env_bool("VIOLET_MOOD", True)


def _row(db) -> dict[str, Any]:
    return db.fetch_one("SELECT * FROM mood_state WHERE id = 1") or {"value": 0.0, "cause": None, "updated_at": db.now()}


def _decayed(value: float, since: str, now: str) -> float:
    try:
        hours = (datetime.fromisoformat(now[:19]) - datetime.fromisoformat(str(since)[:19])).total_seconds() / 3600
    except ValueError:
        return value
    half = max(0.1, env_float("VIOLET_MOOD_HALF_LIFE_HOURS", 3.0))
    return value * math.pow(0.5, max(0.0, hours) / half)


def current(db) -> dict[str, Any]:
    """{"value": -1..1, "cause": str | None} as of now."""
    row = _row(db)
    return {"value": round(_decayed(float(row["value"]), row["updated_at"], db.now()), 3), "cause": row.get("cause")}


def feel(db, text: str, who: str = "主人", weight: float = 1.0) -> float | None:
    """Words said to her move her mood. Returns the new value, or None when nothing moved it."""
    from ai import test_mode
    if not enabled() or test_mode.active() or not text:
        return None
    scold = scold_strength(text)
    praise, sad = bool(PRAISE.search(text)), bool(SAD.search(text))
    if not (praise or scold or sad):
        return None
    delta, cause = 0.0, None
    if scold:
        delta, cause = -0.35 * scold, f"{who}对你说了{_quoted(text)}"
    elif praise:
        delta, cause = 0.3, f"{who}夸了你 / 谢了你"
    elif sad and who == "主人":
        delta, cause = -0.15, "主人心情不好，你有点担心"
    if not delta:
        return None
    return shift(db, delta * weight, cause)


def shift(db, delta: float, cause: str | None) -> float:
    """Move her mood by ``delta`` (kept within -1 .. 1) and remember why. Returns the new value."""
    value = max(-1.0, min(1.0, current(db)["value"] + delta))
    db.execute("INSERT INTO mood_state (id, value, cause, updated_at) VALUES (1, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
               "value = excluded.value, cause = excluded.cause, updated_at = excluded.updated_at", (value, cause, db.now()))
    from ai import trace
    trace.step("心情", 变化=round(delta, 2), 现在=round(value, 2), 原因=cause)
    return value


def _words(value: float) -> str:
    if value >= 0.65:
        return "很开心"
    if value >= THRESHOLD:
        return "心情不错"
    if value <= -0.65:
        return "很不高兴，有点委屈"
    return "有点不高兴"


def note(db) -> str:
    """The line for the prompts, or "" when she is calm."""
    from ai import test_mode
    if not enabled() or test_mode.active():
        return ""
    mood = current(db)
    value = mood["value"]
    if abs(value) < THRESHOLD:
        return ""
    why = f"（因为{mood['cause']}）" if mood.get("cause") else ""
    if value > 0:
        how = "说话可以更轻快、更爱开玩笑一点，也更愿意配个表情包。"
    else:
        # 2026-11-03 (user: 被凶了情绪起伏不应该更大，语言不应该更多吗): an upset girl says MORE, not less
        how = ("情绪要明显地表现出来，话也会比平时多：委屈地抱怨、气鼓鼓地怼回去、嘴硬、碎碎念为什么不开心，"
               "可以连发几条；按你自己的性格来。但不攻击任何人、不说伤人的话，不冷着对方、不装没事，该帮的忙照样帮；"
               "对方哄你或道歉，就嘴硬一会儿再慢慢好起来。"
               # 2026-11-09 (real chat: "我气还没消呢" in reply after reply for twenty minutes, about a teasing 笨蛋)
               "那句话要是听得出是玩笑、撒娇或打情骂俏，就别当真生气，用玩笑回过去；"
               "同一句抱怨（比如“气还没消”）说过了就别每次回复都再说。")
    return f"【你现在的心情】{_words(value)}{why}。{how}心情会自己慢慢平复，不用一直提。"


def initiative_factor(db) -> float:
    """For starting a topic in a quiet group: 0.5 (upset) .. 1.5 (happy)."""
    if not enabled():
        return 1.0
    return 1.0 + 0.5 * current(db)["value"]
