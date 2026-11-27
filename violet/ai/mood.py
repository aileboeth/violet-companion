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
# 2026-11-03: "工作好烦" / "烦死了" is the owner's own mood, not scolding her -- only "你好烦" / "你真的很烦" / "烦死你了" counts
_YOU_ANNOYING = r"你(?:真的|真是|怎么这么|怎么那么)?(?:好|真|很|太)?烦"
SCOLD = re.compile(rf"(笨蛋|蠢|傻[逼瓜子]|白痴|弱智|废物|垃圾|滚|闭嘴|{_YOU_ANNOYING}|烦死你|讨厌你|恶心|没用|有病|脑残|"
                   r"别吵|闭麦|shut up|stupid|idiot)", re.I)
# 2026-11-09 (real chat, user: 打情骂俏可能触发的骂人): "现在凌晨十二点，我有步数才奇怪，笨蛋" and "笨蛋还不去睡觉" were
# counted as scolding -- her mood went to 有点不高兴 for half an hour and every reply said "我气还没消呢". Between people
# who are close, 笨蛋 / 傻瓜 / 讨厌 / 你好烦 are mostly teasing (嗔怪 / 打情骂俏) -- whether a word hurts depends on the tone
# and the relationship, which a word list cannot see and the reply model can (it reads the message itself). So these
# words alone move her mood only a little (several in a row still add up); the hard insults count as before, and less
# when the same message is plainly affectionate.
MILD_SCOLD = re.compile(rf"(笨蛋|蠢|傻瓜|傻子|{_YOU_ANNOYING}|烦死你|讨厌你|没用)", re.I)
# sweet words, pet names, laughing, a teasing tone or a sticker read as playful (the vision description is in the text)
AFFECTION = re.compile(r"(宝贝|宝宝|臭宝|宝儿|亲爱的|老婆|老公|媳妇|爱你|喜欢你|想你|么么|mua|亲亲|抱抱|贴贴|"
                       r"(小|大|臭)笨蛋|笨蛋蛋|哈哈|嘿嘿|嘻嘻|hhh|233|开玩笑|逗你|[~～]|"
                       r"俏皮|调侃|卖萌|撒娇|搞怪|戏谑|得意|捉弄)", re.I)
MILD_DELTA = -0.1
SAD = re.compile(r"(难过|伤心|好累|崩溃|想哭|哭了|失眠|好烦啊|压力好大|不开心|郁闷)")
THRESHOLD = 0.3


# 2026-11-18 (grok-4.6 QQ group eval: "甲方这需求真是合法但有病" triggered "路人甲对你说了「有病」"):
# venting about a third party / object / meme ("合法但有病", "这破游戏太垃圾", "那老板有病吧") is not scolding her.
_DIRECT_SCOLD = re.compile(rf"^(?:滚|闭嘴|别吵|闭麦|shut up|{_YOU_ANNOYING}|烦死你|讨厌你)$", re.I)
# 2026-11-25 (v0.7.80): explicit dismissal / rejection directed at her + severe earnest abuse for owner_scold_fallback
_OWNER_DISMISSAL = re.compile(
    r"(别跟我说话|别和我说话|不要跟我说话|不要和我说话|别再跟我说话|"
    r"别烦我|别来烦我|别再烦我|别理我|别再理我|离我远点|滚远点|给我滚|"
    r"不想跟你聊|不想和你聊|不想跟你说话|不想和你说话|不想理你|不想看你|"
    r"别再找我|别找我了|少跟我说话|少来烦我|懒得理你)"
)
_SEVERE_INSULT = re.compile(r"^(?:傻逼|白痴|弱智|废物|垃圾|恶心|有病|脑残|stupid|idiot)$", re.I)
_HURT_REPLY_RE = re.compile(
    r"(不打扰|不吵你|不烦你|不说话了|先安静|先退|先走开|我先撤|那我先不|"
    r"对不起|抱歉|委屈|难过|伤心|难受|这么凶|好凶|太凶|干嘛凶|干嘛骂|别凶|"
    r"生气了|不生气|消消气|别气|讨厌我|嫌我|嫌弃我|是我的错|我错了|知道了|"
    r"那算了|行吧，我不说|好，我不说|哼|呜)"
)
_PLAYFUL_REPLY_RE = re.compile(r"(哈哈|嘿嘿|嘻嘻|hhh|233|笑死|才不|开玩笑|逗你)", re.I)
_STICKER_MARKER_RE = re.compile(r"\[\[表情:(\d+)\]\]")
_LABEL_STICKER_RE = re.compile(r"<sticker:([^>]+)>")
_HURT_STICKER_RE = re.compile(r"(生气|委屈|难过|伤心|哭|流泪|眼泪|冷落|气鼓鼓|失落|受伤|不开心)")
_THIRD_PARTY_SUBJECT = re.compile(
    r"(?:合法但|甲方|乙方|老板|领导|同事|老师|导师|教授|室友|舍友|队友|同学|策划|官方|运营|客服|商家|店家|物业|快递|外卖|司机|产品|"
    r"对方|别人|那人|这人|他们|她们|他|她|它|"
    r"这(?:个|破|坨|玩意|需求|操作|东西|游戏|课|题|作业|天气|软件|系统|代码|电影|剧|饭|菜)|"
    r"那(?:个|破|家|种|需求|操作|游戏|老板|人))"
)
_CLAUSE_BREAK = re.compile(r"[，。！？,.!?\n;；:：]")


def _is_third_party_vent(text: str, match: re.Match[str]) -> bool:
    word = match.group(0)
    if _DIRECT_SCOLD.fullmatch(word):
        return False
    prefix = text[:match.start()]
    clause = _CLAUSE_BREAK.split(prefix)[-1][-18:]
    if "你" in clause:
        return False
    return bool(_THIRD_PARTY_SUBJECT.search(clause))


def scold_strength(text: str) -> float:
    """How much a message scolds her: 0 (not at all) .. 1 (a hard insult, said in earnest)."""
    text = text or ""
    found = [m.group(0) for m in SCOLD.finditer(text) if not _is_third_party_vent(text, m)]
    if not found:
        return 0.0
    affectionate = bool(AFFECTION.search(text))
    if all(MILD_SCOLD.fullmatch(w) for w in found):
        return 0.0 if affectionate else abs(MILD_DELTA) / 0.35
    return 0.4 if affectionate else 1.0


def _quoted(text: str) -> str:
    """The scolding words themselves, for the cause line ("主人说你「笨蛋」")."""
    text = text or ""
    words = []
    for m in SCOLD.finditer(text):
        if _is_third_party_vent(text, m):
            continue
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


def _reply_shows_hurt(db, reply_text: str) -> bool:
    """True when Violet's own reply shows retreat, hurt, apology, or a hurt/angry sticker."""
    text = (reply_text or "").strip()
    if not text:
        return False
    for m in _STICKER_MARKER_RE.finditer(text):
        try:
            row = db.fetch_one("SELECT description FROM stickers WHERE id = ?", (int(m.group(1)),))
        except Exception:                                   # noqa: BLE001
            row = None
        if row and _HURT_STICKER_RE.search(row.get("description") or ""):
            return True
    for m in _LABEL_STICKER_RE.finditer(text):
        if _HURT_STICKER_RE.search(m.group(1) or ""):
            return True
    if _PLAYFUL_REPLY_RE.search(text):
        return False
    return bool(_HURT_REPLY_RE.search(text))


def owner_scold_fallback(db, user_message: str, reply_text: str = "") -> float | None:
    """Strict, zero-false-positive fallback for owner chat when the model missed note_feeling
    on unambiguous serious scolding (e.g. only stepped back briefly or sent a hurt sticker).

    Never fires on teasing / affection, third-party vents, plain messages, or mild scolding alone.
    """
    from ai import test_mode
    if not enabled() or test_mode.active() or not user_message:
        return None
    if AFFECTION.search(user_message):
        return None
    found = [m.group(0) for m in SCOLD.finditer(user_message) if not _is_third_party_vent(user_message, m)]
    if not found:
        return None
    strength = scold_strength(user_message)
    if strength <= 0.0:
        return None
    dismissed = bool(_OWNER_DISMISSAL.search(user_message))
    severe_abuse = any(_SEVERE_INSULT.fullmatch(w) for w in found)
    hard_with_hurt = strength >= 1.0 and _reply_shows_hurt(db, reply_text)
    if not (dismissed or severe_abuse or hard_with_hurt):
        return None
    delta = -0.45 if (strength >= 1.0 and dismissed) else -0.35
    quoted = _quoted(user_message)
    cause = f"主人认真凶了你（{quoted}）" if quoted else "主人认真凶了你"
    return shift(db, delta, cause)


def shift(db, delta: float, cause: str | None) -> float:
    """Move her mood by ``delta`` (kept within -1 .. 1) and remember why. Returns the new value."""
    cur = current(db)["value"]
    # 2026-11-25 (v0.7.80): a serious scold (delta <= -0.3) from a positive mood drops straight from calm (0.0)
    # so it immediately reflects upset mood instead of cancelling out near 0 below THRESHOLD.
    base = 0.0 if (cur > 0 and delta <= -0.3) else cur
    value = max(-1.0, min(1.0, base + delta))
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


def note(db, include_cause: bool = True) -> str:
    """The line for the prompts, or "" when she is calm.

    ``include_cause=False`` omits the private ``cause`` text (e.g. resolved story titles or owner-private
    ``note_feeling`` reasons) for untrusted social and public signature prompts.
    """
    from ai import test_mode
    if not enabled() or test_mode.active():
        return ""
    mood = current(db)
    value = mood["value"]
    if abs(value) < THRESHOLD:
        return ""
    why = f"（因为{mood['cause']}）" if (include_cause and mood.get("cause")) else ""
    if value > 0:
        how = "说话可以更轻快、更爱开玩笑一点，也更愿意配个表情包。"
    else:
        # 2026-11-03 (user: 被凶了情绪起伏不应该更大，语言不应该更多吗): an upset girl says MORE, not less --
        # 2026-11-18 (grok-4.6 vt-quiet): respect quiet personas & stop-boundaries, and step down once comforted
        # 2026-11-25 (phase 5): let quiet/introverted personas show genuine stung quietness & soft aching honesty
        how = ("情绪要按你自己的性格真切表现出来（外向或爱撒娇时话也会比平时多，委屈抱怨或嘴硬几句；安静内敛时不必硬装没事，可用微停顿、轻声刺痛的坦白或克制的疏离把委屈和发空透出来）。"
               "但不攻击任何人、不说伤人的话，不冷着对方，该帮的忙照样帮；若人设要求对方喊停就停下，绝不反复解释委屈让对方内疚。"
               "对方已经在哄你、打圆场或叫你别多想时，顺台阶翻篇好起来（可调 note_feeling 恢复），同一句抱怨别每次回复都再说；"
               "听得出是玩笑或打情骂俏就别当真生气。")
    return f"【你现在的心情】{_words(value)}{why}。{how}心情会自己慢慢平复，不用一直提。"


def initiative_factor(db) -> float:
    """For starting a topic in a quiet group: 0.5 (upset) .. 1.5 (happy)."""
    if not enabled():
        return 1.0
    return 1.0 + 0.5 * current(db)["value"]
