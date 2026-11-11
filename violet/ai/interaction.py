"""Reading a line of chat with rules: is it a question, does a reply end by asking.

2026-11-11 (audit R3): this file used to decide, from keywords in the owner's message, which notes and which parts of
the context the model got on a turn ("这一句…", her life shown or hidden). That is gone -- the model reads the message
itself. What is left serves two things only: the "别老问我问题" preference check (one rewrite when a reply to a plain
share still ends with a question) and how long a busy period the owner stated lasts.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from ai.conversation_state import ASK_ABOUT_VIOLET_RE
from ai.onboarding import DISTRESS_RE, QUESTION_RE

# The user hands the asking over to Violet.
INVITE_RE = re.compile(r"(有什么想问|有啥想问|想问我什么|想问我啥|你问吧|问我吧|你可以问|随便问|想知道我什么|想了解我什么|你想知道什么)")
# Greetings and open small talk: nothing of the user's to attend to; a natural place for Violet's own news.
# phase 8.9.8: plain greetings too ("早呀" the next morning: Violet's day and what was agreed the night before belong there)
OPEN_RE = re.compile(r"^(在吗|在不在|在嘛|在么|你在吗|在干嘛呢?|干嘛呢|在干什么呢?|在做什么呢?|忙什么呢?|忙啥呢?|"
                     r"早|早安|早上好|中午好|下午好|晚上好|晚好|你好|嗨|哈喽|hi|hello|"
                     r"好?无聊(死了)?|随便聊(点|聊)?.*|聊点什么.*|陪我聊.*|想听你.*|说点什么.*|你说.{0,6}吧)"
                     r"[呀啊呢嘛吧～~!！。.…]*$")
# "Do you remember / know ... about me": the user asks what Violet knows of them.
SELF_RECALL_RE = re.compile(r"(还记得|记不记得|记得吗|你知道|忘了|忘记).{0,12}我|我.{0,12}(来着|还记得吗|记得吗|你知道吗)")
# A trailing question in Violet's reply (rhetorical ones included: a rewrite is cheap, a missed one is not).
REPLY_QUESTION_END = re.compile(r"([?？]|[吗呢嘛][～~。!！…]*)\s*$")
FOCUS_MIN_CHARS = 30


# Questions without a final particle: A-not-A ("是不是都很自律") and asking for Violet's opinion ("你觉得…").
_ASKING_RE = re.compile(r"(是不是|有没有|要不要|好不好|能不能|会不会|对不对|可不可以|行不行|该不该|你觉得|你认为)")


def is_question(text: str | None) -> bool:
    text = (text or "").strip()
    if not (QUESTION_RE.search(text) or _ASKING_RE.search(text)):
        return False
    # phase 8.9.8: a statement that only ends in 呢 ("没有哦，我也在找呢", "还在上课呢") is not a question
    return not statement_ne(text)


# Phase 8.9.8 (8.9.7 diagnosis of the real free chat local/free_chat/cys-close-1): Violet's questions came without a
# question mark ("今天摸的哪门课", "你是喜欢看哪种类型的", "你这两天怎么样呀", "有没有哪门能偷偷打个盹哈哈"), so the
# notes about her questions fired once in eight turns. A line asks when its last clause carries a question word or an
# A-not-A form -- unless the word is used indefinitely ("你看哪个顺眼就一起看", "什么都行"), inside a statement
# ("我也不知道为什么") or rhetorically ("不摸鱼怎么活", "哪有"). Trailing laughter / emoji do not hide a question.
QUESTION_WORD = (r"(怎么样|怎么|咋样|咋|干嘛|干啥|什么|啥|哪(个|种|门|些|里|儿|样|天|本|首|部|家|位)?|谁|"
                 r"几(点|个|天|次|号|岁|本)|多少|多久|多大|为什么|为啥|有没有|是不是|要不要|会不会|能不能|可不可以|"
                 r"想不想|喜不喜欢|好不好|对不对|行不行|去不去)")
_QUESTION_WORD_RE = re.compile(QUESTION_WORD)
_A_NOT_A_RE = re.compile(r"([一-鿿])不\1")
_TRAILING_RE = re.compile(r"([\s～~。.!！…]|哈|嘿|嘻|hh|[\U0001F300-\U0001FAFF☀-➿])+$")
_INDEFINITE_RE = re.compile(QUESTION_WORD + r"[^，,。！!？?]{0,10}(就|都|也)")
# phase 8.9.11: thinking aloud is not asking -- "正想着要不要去泡杯什么喝" (real ab-v2c) counted as a question
_EMBEDDED_RE = re.compile(r"(知道|记得|清楚|明白|确定|想不起来?|说不上来?|说不清|想想|看看|问问|猜猜|研究|"
                          r"想着|在想|纠结|犹豫|琢磨|考虑)[^，,。！!？?]{0,6}" + QUESTION_WORD)
_RHETORICAL_WH_RE = re.compile(r"(不|没)[^，,。！!？?]{0,4}(怎么|咋)[^，,。！!？?]{0,2}$|哪有|谁不|怎么可能|有什么好|何必")
# "你呢", "那你呢", "你那边呢", "然后呢", "还有呢" -- asking back; "我也在找呢" / "我好累呢" are not
_NE_TOPIC_RE = re.compile(r"^(那|那么)?(你|您|你们|他|她|他们|然后|还有|其他|别的|这个|那个|后来)[^在着也还正]{0,3}呢$")


def _last_clause(line: str) -> str:
    line = _TRAILING_RE.sub("", (line or "").strip())
    return re.split(r"[，,。；;！!]", line)[-1].strip() if line else ""


def statement_ne(text: str | None) -> bool:
    """A line that only ends in 呢 as a statement ("我也在找呢", "还在上课呢") -- not "你呢", "然后呢", "在干嘛呢"."""
    clause = _last_clause(text)
    if not clause.endswith("呢") or re.search(r"[?？]\s*$", (text or "").strip()):
        return False
    if _QUESTION_WORD_RE.search(clause) or _A_NOT_A_RE.search(clause):
        return bool(_INDEFINITE_RE.search(clause) or _EMBEDDED_RE.search(clause))
    return not _NE_TOPIC_RE.match(clause)


def asks(line: str | None) -> bool:
    """Phase 8.9.8: Violet's reply line ends by asking the user something (with or without a question mark)."""
    raw = (line or "").strip()
    clause = _last_clause(raw)
    if not clause:
        return False
    if re.search(r"[?？]", raw[-3:]) or re.search(r"(吗|(?<![什怎这那多])么)$", clause):
        return not _RHETORICAL_WH_RE.search(clause)
    if clause.endswith("呢"):
        return not statement_ne(clause)
    if re.search(r"了没有?[呀啊呢嘛]?$", clause):            # "下课了没呀", "吃了没" (8.9.7 real proactive)
        return True
    if re.search(r"(吧|啦|了)$", clause):                  # a suggestion or a statement ("周末一起找吧")
        return False
    has_word = bool(_QUESTION_WORD_RE.search(clause) or _A_NOT_A_RE.search(clause))
    if not has_word or _INDEFINITE_RE.search(clause) or _EMBEDDED_RE.search(clause) or _RHETORICAL_WH_RE.search(clause):
        return False
    # about herself, without ？ / 吗: self-talk ("我是不是没救了呀", 8.9.2), not a question for the user to answer
    return not (clause.startswith("我") and not clause.startswith("我们") and "你" not in clause)


# Phase 8.9: the user is leaving or going to sleep ("我去洗澡了", "那我去忙啦", "不说了 练琴去", "晚安").
CLOSING_RE = re.compile(r"(晚安|拜拜|再见|回头聊|明天聊|改天聊|先这样|不说了|不聊了|先撤|先下了|"
                        r"(先|要|得)?(去)?睡(了|啦|觉了|觉去)|困(了|死了)|好困|"
                        r"(我|那我|先)?(先|要|得)?去(忙|洗澡|洗漱|上课|吃饭|做饭|上班|开会|写作业|练琴|收拾|跑步|健身|学习)"
                        r"(了|啦|咯|喽|吧)?|练琴去|忙去了)")
# Phase 8.9.12 (real ab-v2d): stepping back so as not to disturb her ("好啦，那我不打扰你看书了", "你先忙吧", "不吵你了")
# is consideration, not the user leaving -- Intent read it as "准备结束这轮对话" and Violet said goodbye. Leaving
# ("我去睡了", "晚安") stays "closing" and wins when both are said.
YIELD_RE = re.compile(r"(不打扰你|不打扰了|不吵你|不烦你|不耽误你|你先忙|你忙吧|你去忙吧|你忙你的|你先去忙|"
                      r"你去看书吧|你先看书|你继续看|你继续忙|你看你的|你接着看|你接着忙|你先休息|你歇着吧)")
# A bare acknowledgement: nothing new to answer.
ACK_RE = re.compile(r"^(嗯+|嗯呢|嗯哼|哦+|噢+|喔+|好+|好的|好吧|行吧?|ok|okay|OK|en|emm+|额|呃)[。.!！~～…]*$")


def turn_kind(text: str | None) -> str:
    """invite (asks Violet to ask) | open (greeting, idle small talk) | question | closing (leaving, going to sleep)
    | yield (stepping back so as not to disturb her, phase 8.9.12) | ack (a bare "嗯") | share (anything else)."""
    text = (text or "").strip()
    if INVITE_RE.search(text):
        return "invite"
    if OPEN_RE.match(text):
        return "open"
    if is_question(text):
        return "question"
    if CLOSING_RE.search(text):
        return "closing"
    if YIELD_RE.search(text):
        return "yield"
    if ACK_RE.match(text):
        return "ack"
    return "share"


def follow_up_preference(preferences: dict[str, Any]) -> str | None:
    """'no' or 'less' when the user asked not to be questioned (an inferred 'less' is too weak to act on)."""
    pref = preferences.get("follow_up_questions") or {}
    if pref.get("value") == "no" or (pref.get("value") == "less" and pref.get("origin") == "explicit"):
        return pref["value"]
    return None


FOLLOW_UP_KINDS = ("share", "open", "ack", "closing")


def answers_invited_question(previous_user: str | None) -> bool:
    """The user's previous message asked Violet to ask: this message is their answer (one follow-up is welcome)."""
    return bool(previous_user) and turn_kind(previous_user) == "invite"


def follow_up_violation(reply: str | None, user_text: str | None, follow_up: str | None,
                        previous_user: str | None = None) -> bool:
    """Violet ended with a question on a plain share although the user asked not to be questioned."""
    if not follow_up or turn_kind(user_text) not in FOLLOW_UP_KINDS or answers_invited_question(previous_user):
        return False
    lines = [l.strip() for l in (reply or "").splitlines() if l.strip()]
    return bool(lines) and asks(lines[-1])


FOLLOW_UP_NOTE = ("对方说过不喜欢被追问，而这条回复以提问结尾：顺着对方的话接、说说你的感受或看法，不要以提问结尾")

# How long a busy state the user announced lasts, from their own words (scenario-time based).
_ALL_DAY_RE = re.compile(r"(今天一整天|一整天|今天一天|今天全天|全天|今天都|今天要一直|今天得一直|今天一直|今晚都|晚上都)")
_TWO_DAYS_RE = re.compile(r"(这两天|这俩天|这一两天|明后天)")
_FEW_DAYS_RE = re.compile(r"(这几天|最近几天|接下来几天|这阵子|这周|这个星期|这礼拜)")


def _end_of_day(day: datetime) -> datetime:
    return day.replace(hour=23, minute=59, second=0, microsecond=0)


def stated_busy_until(text: str | None, now: datetime) -> datetime | None:
    """End of a busy period the user stated, or None when they gave none (or a short one: the default applies).

    今天一整天 / 今天都 -> end of today; 这两天 -> end of tomorrow; 这几天 / 这周 -> end of the day after tomorrow
    (a deliberately short reading of "a few days": the user can always say again). Never longer than stated."""
    text = text or ""
    if _FEW_DAYS_RE.search(text):
        return _end_of_day(now + timedelta(days=2))
    if _TWO_DAYS_RE.search(text):
        return _end_of_day(now + timedelta(days=1))
    if _ALL_DAY_RE.search(text):
        return _end_of_day(now)
    return None
