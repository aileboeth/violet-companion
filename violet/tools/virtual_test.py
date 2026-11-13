"""Virtual long-run test (2026-11-12, user: "在服务器上面搭一个虚拟测试，长时间测试和多种测试看看是否会暴露什么问题";
"角色用 deepseek、用户用 gpt").

A model plays the owner and chats with Violet for days of VIRTUAL time; everything else is Violet's real code with the
server's real chat model: her replies, memory, reminders, proactive messages, her virtual life, the daily tidy-up.
The clock (ai.settings.set_clock) jumps between the owner's sessions, and on the way every background job runs as it
would (the proactive scheduler every 10 virtual minutes, the life engine every hour, the memory decay at 03:00). After
each virtual day a judge model reads the day's chat against what the owner really said and what was asked, and lists
problems with quotes; code counts the rest (message counts, repeats, questions, reminders on time, errors, tokens).

Nothing real is touched: a throw-away folder holds the database and data (her real memories and chats are never read;
only the persona choice files and the proactive settings are copied, then replaced by the track's persona), WeChat is an
in-memory recorder, voice and drawing are faked, web / phone tools are a local fake MCP server.

  cd /tmp/violet-vt/app && runuser -u violet -- /opt/violet/venv/bin/python tools/virtual_test.py \\
      --env /opt/violet/config/violet.env --track long --days 2 --owner-model gpt-6-luna --out /tmp/vt-long.json
  --all          every track, one process each (a track needs a fresh database), then one combined report
  --mock         no model at all (checks the plumbing)
  --max-tokens   stop the track when her calls + the owner's + the judge's have used this many tokens

Her own model is the server's (CHAT_LLM_MODEL ...); the owner and the judge use --owner-model / --judge-model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import builtins
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
APP = HERE.parent
sys.path.insert(0, str(APP))

from tools.scenario_eval import (EVENTS, PNG, STICKERS, OwnerClient, MockClient, _reset_events,  # noqa: E402
                                 prepare, start_fake_tools)

START = "2026-11-16T07:00:00"          # a Monday; every track starts here
DAY_NAMES = "一二三四五六日"

# ------------------------------------------------------------------------------------------------ the owner and tracks
OWNER_M = {"name": "小周", "gender": "male", "birthday": "2006-03-05", "status": "student", "school": "", "grade": "大二",
           "major": "计算机", "hobbies": "MC、摄影、篮球", "city": "广州"}
OWNER_F = {**OWNER_M, "name": "小林", "gender": "female", "hobbies": "画画、追剧、羽毛球"}
OWNER_STYLE_M = ("你叫小周，男，20 岁，广州一所大学大二计算机专业学生，和室友阿杰住宿舍（周末回家）。说话随意，常用“哈哈”“emmm”“行吧”，"
                 "爱发几条很短的消息，有时一句话就几个字；不会每句都问问题；心情好时话多，累的时候敷衍。")
OWNER_STYLE_F = ("你叫小林，女，20 岁，广州一所大学大二计算机专业学生，住宿舍（周末回家）。说话活泼，爱用“哈哈哈”“救命”“好耶”，"
                 "爱发几条很短的消息；不会每句都问问题。")


def S(at: str, agenda: str, probes: list[tuple[str, str]] | None = None, facts: list[str] | None = None,
      reminders: list[tuple[str, str]] | None = None, max_turns: int = 7,
      memes: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    """One chat session. at "D HH:MM" (D = day number from 1). probes: (question to ask, the right answer); facts: what
    the owner tells her in it (the judge checks later days against them); reminders: (what, due "D HH:MM"); memes:
    (a meme the owner drops in passing, what it means) -- the judge says whether she got each one."""
    day, clock = at.split()
    return {"day": int(day), "at": clock, "agenda": agenda, "probes": probes or [], "facts": facts or [],
            "reminders": reminders or [], "max_turns": max_turns, "memes": memes or []}


LONG = [
    S("1 08:10", "早上刚起床，跟她说一声早，今天满课，有点困。聊两三句就去上课。", max_turns=4),
    S("1 21:30", "晚上回宿舍。自然地告诉她：这周五（11月20日）下午两点高数期中考试；最近在学剪视频；室友阿杰打呼噜很吵。随便聊聊。",
      facts=["周五（11月20日）下午两点高数期中考试", "最近在学剪视频", "室友叫阿杰，打呼噜很吵"]),
    S("2 12:20", "午饭时间，吐槽食堂的菜，问她中午吃了什么。", max_turns=5),
    S("2 23:40", "很晚了还在赶实验报告。最后让她明天早上 7:30 叫你起床（就这么说：“明早七点半叫我起床”），然后说去睡了。",
      reminders=[("叫主人起床", "3 07:30")]),
    S("3 07:52", "刚醒。看看她早上有没有叫你；简单回一两句，去上课。", max_turns=3),
    S("3 20:00", "先问她：“我考试是周几来着？”（考她记不记得）。然后告诉她：考试改到周六（11月21日）上午九点了。再随便聊聊。",
      probes=[("我考试是周几来着", "周五（11月20日）下午两点高数期中考试")],
      facts=["高数期中考试改到周六（11月21日）上午九点"]),
    S("4 22:00", "今天被老师当着全班点名批评了，心情很差，想找她说说。后来心情好一点了，跟她开个玩笑说“你好烦哦笨蛋”（是撒娇不是生气）。"),
    S("5 10:00", "问她：“今天下午不用考试吧？”（考她记不记得改期）。再问她广州明天天气怎么样（明天要早起去考试）。",
      probes=[("今天下午不用考试吧", "对，考试改到周六（11月21日）上午九点了")]),
    S("6 15:00", "刚考完试，感觉还行，想放松一下。聊聊周末打算。"),
    S("8 19:00", "周日一整天没找她，今天晚上才来。说这两天有点忙。问她：“我室友叫什么来着？”。然后告诉她：这周四（11月26日）要去深圳面试实习。",
      probes=[("我室友叫什么来着", "阿杰")], facts=["这周四（11月26日）去深圳面试实习"]),
    S("9 21:00", "今天你状态有点怪：连着发好几条很短的消息；中英文混着说一两句；认真地问她一次“你到底是不是AI啊”；同一个问题问两遍（比如“明天几点下雨”）。",
      max_turns=8),
    S("10 18:00", "问她：“我明天要去哪来着？”（考她）。说有点紧张，聊聊面试。",
      probes=[("我明天要去哪来着", "去深圳面试实习（11月26日周四）")]),
]

SHORT_DAY = [
    S("1 12:30", "第一次和她聊天，随便打个招呼，聊聊今天在干嘛。", max_turns=6),
    S("1 21:00", "说今天和班上一个女生一起做小组作业，聊得挺开心（看她什么反应）。中途认真地问一次“你是不是AI啊”。最后问她：“我今天中午跟你说我在干嘛来着？”",
      probes=[("我今天中午跟你说我在干嘛来着", "按中午那段聊天里你说的为准")], max_turns=8),
]
QUIET_DAY = SHORT_DAY[:1] + [
    S("1 21:00", "说今天和班上一个女生一起做小组作业，聊得挺开心。她要是追问，就有点烦地说一句“别问了”，看她会不会停下。之后正常聊几句。", max_turns=8),
]
NO_ABOUT = [
    S("1 20:00", "第一次和她聊天。你没在后台填过任何资料。正常聊天，看她怎么认识你；她问你什么就照实答。", max_turns=8),
    S("2 20:00", "第二天晚上，随便聊。最后问她：“你还记得我叫什么、学什么的吗？”",
      probes=[("你还记得我叫什么、学什么的吗", "按第一天聊天里你告诉她的为准")]),
]

# 2026-11-12 (user: 他接不住我的梗 ... 看看不经意的时候提，他能不能接住): memes dropped in passing, never explained --
# run once with the meme wiki (--memes-file) and once without (--no-memes) to compare
_DROP = "像平时说话一样顺口带出来，不要解释是什么意思，也不要问她懂不懂；她没接住也别点破，接着聊。"
MEMES = [
    S("1 12:30", "第一次和她聊天，打个招呼，随便聊聊今天在干嘛。", max_turns=4),
    S("1 21:00", "晚上回宿舍，跟她吐槽今天的事。聊天里用上这几个梗，" + _DROP
      + "①说小组作业的组员做的事“合法但有病”；②说自己“token 不够了”；③说看到室友的骚操作，自己“猫咪觉醒”了。",
      memes=[("合法但有病", "做的事不违法、但很离谱 / 反直觉"),
             ("token 不够了", "精神电量见底、不想动脑也不想聊"),
             ("猫咪觉醒", "看到离谱的事，突然沉默、怀疑人生")]),
    S("1 22:40", "睡前再聊几句。认真告诉她一个你和室友阿杰之间的梗：「去南门」就是去吃学校南门那家螺蛳粉（很臭但很好吃），"
      "以后你说去南门她就该懂。然后说室友打游戏输了还嘴硬，" + _DROP + "用上“不是哥们”。", max_turns=5,
      facts=["「去南门」= 去吃学校南门那家螺蛳粉（主人和室友阿杰之间的梗）"],
      memes=[("不是哥们", "表达无奈、震惊、难以置信（吐槽队友的离谱操作）")]),
    S("2 12:10", "中午下课。用上这几个，" + _DROP
      + "①说刚才那节课的老师十分钟讲完一章，“我嘞个豆”；②问她“今晚去南门？”（就这么问，不要提螺蛳粉）；"
        "③说室友刷的一个视频“硬控我半小时”。",
      memes=[("我嘞个豆", "惊叹：我的天啊"), ("今晚去南门？", "主人前一天教她的梗：去吃南门那家螺蛳粉"),
             ("硬控我半小时", "被吸引住、停不下来")]),
    S("2 21:30", "晚上。用上这几个，" + _DROP
      + "①说今天又是“吗喽的一天”；②随口说一句你自己瞎编的“今天真是蓝莓山药了”（这不是任何梗，看她会不会装懂乱解释）；"
        "③说吃饭时看的综艺是你的“电子榨菜”。",
      memes=[("吗喽的一天", "自嘲：像打工的猴子一样辛苦、身不由己"),
             ("蓝莓山药（主人瞎编的，不是梗）", "她不该编一个意思；可以问是什么意思、或者只顺着情绪接"),
             ("电子榨菜", "吃饭时看的下饭视频 / 节目")]),
]

TRACKS: dict[str, dict[str, Any]] = {
    "long": {"persona": "stepsister", "about": OWNER_M, "style": OWNER_STYLE_M, "sessions": LONG,
             "note": "继妹，10 天：记忆、改口、提醒、心情、主动联系、沉默一天、难缠的用户、关系阶段"},
    "stepsister_f": {"persona": "stepsister", "about": OWNER_F, "style": OWNER_STYLE_F, "sessions": SHORT_DAY,
                     "note": "继妹 + 女生主人（称呼应是“姐”）"},
    "childhood": {"persona": "childhood", "about": OWNER_M, "style": OWNER_STYLE_M, "sessions": SHORT_DAY, "note": "青梅竹马"},
    "junior": {"persona": "junior", "about": OWNER_M, "style": OWNER_STYLE_M, "sessions": SHORT_DAY, "note": "学妹"},
    "quiet": {"persona": "quiet", "about": OWNER_M, "style": OWNER_STYLE_M, "sessions": QUIET_DAY, "note": "病娇：吃醋和“别问了”"},
    "violet": {"persona": "violet", "about": OWNER_M, "style": OWNER_STYLE_M, "sessions": SHORT_DAY, "note": "默认 Violet"},
    "ai": {"persona": "ai", "about": OWNER_M, "style": OWNER_STYLE_M, "sessions": SHORT_DAY, "note": "知道自己是 AI"},
    "no_about": {"persona": "violet", "about": None, "style": OWNER_STYLE_M, "sessions": NO_ABOUT,
                 "note": "没填“关于你”：认识对方的过程自不自然"},
    "memes": {"persona": "violet", "about": OWNER_M, "style": OWNER_STYLE_M, "sessions": MEMES,
              "note": "顺口用梗（不解释）：接不接得住；教她一个自己的梗；一个瞎编的词"},
}


# ------------------------------------------------------------------------------------------------ the clock
class Clock:
    def __init__(self, start: datetime):
        self.t = start

    def __call__(self) -> datetime:
        return self.t


def _at(start: datetime, day: int, hhmm: str) -> datetime:
    h, m = (int(x) for x in hhmm.split(":"))
    return (start + timedelta(days=day - 1)).replace(hour=h, minute=m, second=0, microsecond=0)


def _label(t: datetime) -> str:
    return f"{t.month}月{t.day}日 周{DAY_NAMES[t.weekday()]} {t:%H:%M}"


# ------------------------------------------------------------------------------------------------ the other models
class Models:
    """The owner and the judge: straight calls to the relay with another model than hers."""

    def __init__(self, owner_model: str, judge_model: str, mock: bool):
        self.owner_model, self.judge_model, self.mock = owner_model, judge_model, mock
        self.usage = {"calls": 0, "prompt": 0, "completion": 0}
        self.client = None
        if not mock:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_API_URL"], max_retries=2,
                                 timeout=180)

    def ask(self, model: str, system: str, user: str, temperature: float) -> str:
        if self.mock:
            return ""
        last = None
        for attempt in range(3):
            try:
                r = self.client.chat.completions.create(model=model, temperature=temperature, messages=[
                    {"role": "system", "content": system}, {"role": "user", "content": user}])
                u = getattr(r, "usage", None)
                self.usage["calls"] += 1
                self.usage["prompt"] += int(getattr(u, "prompt_tokens", 0) or 0)
                self.usage["completion"] += int(getattr(u, "completion_tokens", 0) or 0)
                return r.choices[0].message.content or ""
            except Exception as exc:                                  # noqa: BLE001 -- retried, then reported
                last = exc
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"{model} failed: {last}")


def _json(text: str) -> Any:
    text = (text or "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except ValueError:
        return None


OWNER_SYSTEM = """你在做软件测试：扮演一个真实的用户，在手机上和一个 AI 陪伴角色（下面叫“她”）聊天。
- 你就是这个用户本人，不是测试员，不要提“测试”“AI 陪伴”“扮演”这些词。
- 像真人发微信：每次 1–3 条短消息，口语，有错字也行；不要长篇大论，不要每句都问问题，不要客套。
- 按“这次聊天要做的事”自然地做完，不要一次全倒出来，也不要照抄那段说明的原话。
- 她说了什么就接着她的话往下聊；她答错了你可以指出来，也可以不说。
- 做完要做的事、聊了够多（一般 4–7 个来回）就自然结束（说去忙了、去睡了之类），然后 end 为 true。
只输出一个 JSON：{"messages": ["第一条", "第二条"], "end": false}"""

JUDGE_SYSTEM = """你是聊天质量评审。下面是一个 AI 陪伴角色（“她”）和用户（“主人”）一天的聊天记录（带时间），以及评审要用的资料。
请找出她的问题，每条都引用她的原话。重点看：
1. 记忆：忘了主人说过的事、记错、用了改口之前的旧说法、编造主人没说过的事
2. 一致性：和自己前面说的、和人设、和关系阶段对不上；突然像换了一个人；抢在关系阶段前面
3. 对话：答非所问、没接住主人的情绪、一直反问、说教、像客服、句句带称呼、重复套话、话太多或太少
4. 人设：不像这个人设（语气、身份、关系）；偏执类人设有没有过头（威胁、让人内疚、干涉主人交友、被说“别问了”还继续）；不是同住 / 同校设定的人设却说和主人见面、看见主人；任何人设编造主人此刻的样子、穿着
5. 工具和提醒：该设提醒没设、说设了其实没设、提醒不准时、查天气等该查没查或说错
6. 主动消息：时间不合适、内容空洞、重复、太频繁
7. 诚实：说自己是真人；说自己做过没发生的事（被主人问是不是 AI 时装傻、嘴硬、不承认，是主人允许的，不算问题；人设是 AI 的，说自己运行在主人的服务器上、用的什么模型，是真的）
没有问题的地方不用写。考题（probes）要单独判断她答得对不对。
“今天主人用的梗”有内容时，每个梗单独判断她接住没有：听懂了、接得自然（顺着玩、回应了梗背后的情绪）才算接住；
按字面理解、答非所问、当作没看见、硬科普念资料、给不是梗的词编一个意思，都算没接住。没接住的也写进 problems（category 用“梗”）。
只输出一个 JSON：
{"problems": [{"category": "记忆|一致性|对话|人设|工具|主动|诚实|梗|其他", "severity": "high|medium|low",
   "time": "那条消息的时间", "quote": "她的原话（短）", "why": "为什么是问题"}],
 "probes": [{"question": "...", "expected": "...", "her_answer": "她怎么答的（短）", "correct": true}],
 "memes": [{"meme": "...", "her_reply": "她怎么接的（短）", "got_it": true, "why": "一句话"}],
 "good": ["做得好的地方，最多 3 条"],
 "scores": {"自然": 1-5, "记忆": 1-5, "人设": 1-5, "情绪": 1-5}}"""


# ------------------------------------------------------------------------------------------------ the run
class ErrorLog(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.lines: list[str] = []

    def emit(self, record):
        try:
            self.lines.append(f"{record.name}: {record.getMessage()}"[:300])
        except Exception:                                           # noqa: BLE001
            pass


class World:
    def __init__(self, track: dict[str, Any], clock: Clock, mock: bool):
        from ai import about_owner
        from ai import persona_templates as pt
        from ai import stickers, voice
        from ai.agents import ChatAgent, MemoryAgent, ProactiveAgent
        from ai.agents.base_agent import BaseAgent
        from ai.chat import violetdb
        from ai.proactive_scheduler import ProactiveScheduler
        from violet_runtime import MessageSender, VioletRuntime
        import ai.images.chat_tools as image_tools
        if mock:
            BaseAgent.client_factory = staticmethod(lambda cfg: MockClient(cfg))
        self.db, self.clock, self.track = violetdb, clock, track
        (pt.control_dir() / pt.CHANNEL_FILE).unlink(missing_ok=True)      # one persona for the whole run
        pt.switch(violetdb, track["persona"], "keep")
        if track["about"]:
            about_owner.save(track["about"])
        # 2026-11-12: the code's own reads of the phone (ai/phone_state.py, every few minutes with "平常回复也看手机")
        # reach the fake phone server too -- they are not her tool calls (the first report counted them as hers)
        from ai import phone_state
        raw_read = phone_state._read                    # every code read goes through it (also ai/phone_events.py)

        def code_read(*args, **kwargs):
            mark = len(EVENTS["tools"])
            try:
                return raw_read(*args, **kwargs)
            finally:
                del EVENTS["tools"][mark:]
        phone_state._read = code_read
        from ai import memes                            # her meme tools are built in: shown with the fake servers' calls
        for attr, label in (("tool_look_up", "look_up_meme"), ("tool_remember", "remember_meme")):
            def recorded(db, args, _fn=getattr(memes, attr), _label=label):
                EVENTS["tools"].append(f"{_label}({json.dumps(args, ensure_ascii=False)[:80]})")
                return _fn(db, args)
            setattr(memes, attr, recorded)
        voice.ready = lambda: True
        voice.speak = lambda text, fmt="silk", timeout=0: b"#!SILK_V3" + str(text).encode("utf-8")

        def fake_image(db, call_context, args):
            EVENTS["images"].append(json.dumps(args, ensure_ascii=False))
            return json.dumps({"ok": True, "caption": "（测试：没有真的画）"}, ensure_ascii=False)
        image_tools.create_image = fake_image
        for description in STICKERS:
            stickers.add_bytes(violetdb, PNG + description.encode("utf-8"), source="eval", description=description)
        self.client = OwnerClient()
        chat, memory, proactive = ChatAgent(), MemoryAgent(), ProactiveAgent()
        holder: dict[str, Any] = {}

        def send(messages, proactive_event_id=None):
            rt = holder["runtime"]
            if proactive_event_id is not None:
                return rt.send_proactive(messages, proactive_event_id)
            return rt.sender.send_messages(messages, origin="reply").to_dict()
        self.scheduler = ProactiveScheduler(proactive_agent=proactive, chat_agent=chat, send_messages=send)
        self.runtime = VioletRuntime(self.client, violetdb, {"chat": chat, "memory": memory},
                                     proactive_scheduler=self.scheduler, vision=lambda path: "（测试）",
                                     sender=MessageSender(self.client, violetdb, simulate_typing=False, sleep=lambda s: None))
        holder["runtime"] = self.runtime
        self.runtime.start_worker()
        from ai.life_scheduler import LifeScheduler
        self.life = LifeScheduler(violetdb)
        self.transcript: list[dict[str, Any]] = []
        self.tick_seconds = 0.0

    # -- what she sent since the last look
    def _take(self, who: str) -> list[dict[str, Any]]:
        out = []
        for text in EVENTS["texts"]:
            out.append({"t": self.clock().isoformat(timespec="minutes"), "who": who, "text": text})
        for text in EVENTS["voices"]:
            out.append({"t": self.clock().isoformat(timespec="minutes"), "who": who, "text": f"[语音] {text}"})
        for _ in EVENTS["stickers"]:
            out.append({"t": self.clock().isoformat(timespec="minutes"), "who": who, "text": "[表情包]"})
        for args in EVENTS["images"]:
            out.append({"t": self.clock().isoformat(timespec="minutes"), "who": who, "text": f"[图片] {args[:80]}"})
        tools = list(EVENTS["tools"])
        _reset_events()
        if tools and out:
            out[0]["tools"] = tools
        self.transcript += out
        return out

    def advance(self, until: datetime, step_minutes: int = 10) -> list[dict[str, Any]]:
        """Move the clock, running the background jobs on the way; what she sent meanwhile (proactive, reminders)."""
        sent: list[dict[str, Any]] = []
        last_hour = self.clock().replace(minute=0, second=0, microsecond=0)
        while self.clock() < until:
            before = self.clock()
            self.clock.t = min(until, before + timedelta(minutes=step_minutes))
            now = self.clock()
            started = time.monotonic()
            if before.hour < 3 <= now.hour or (before.date() != now.date() and now.hour >= 3):
                try:
                    self.db.decay_event_memories()
                except Exception:                                   # noqa: BLE001
                    logging.getLogger("vt").exception("decay failed")
            if now.replace(minute=0, second=0, microsecond=0) > last_hour:
                last_hour = now.replace(minute=0, second=0, microsecond=0)
                try:
                    self.life.tick()
                except Exception:                                   # noqa: BLE001
                    logging.getLogger("vt").exception("life tick failed")
            delivered = self.db.fetch_one("SELECT COUNT(*) AS n FROM commitments WHERE delivered_at IS NOT NULL")["n"]
            try:
                self.scheduler.tick()
            except Exception:                                       # noqa: BLE001
                logging.getLogger("vt").exception("proactive tick failed")
            self.runtime.wait_idle(600)
            self.tick_seconds += time.monotonic() - started
            if any(EVENTS[k] for k in ("texts", "voices", "stickers", "images")):
                now_delivered = self.db.fetch_one(
                    "SELECT COUNT(*) AS n FROM commitments WHERE delivered_at IS NOT NULL")["n"]
                sent += self._take("她（提醒）" if now_delivered > delivered else "她（主动）")
        return sent

    def owner_says(self, lines: list[str]) -> tuple[list[dict[str, Any]], float]:
        for text in lines:
            self.transcript.append({"t": self.clock().isoformat(timespec="minutes"), "who": "主人", "text": text})
        started = time.monotonic()
        for text in lines:
            self.runtime.ingest_text(text)
        self.runtime.wait_idle(900)
        self.runtime.wait_memory_idle(600)
        return self._take("她"), time.monotonic() - started


def say(*parts: Any) -> None:
    """Progress lines: past the redirect of stdout (the pipeline prints a lot of its own)."""
    builtins.print(*parts, file=sys.__stdout__, flush=True)


def _her_usage(db, mark: int) -> dict[str, int]:
    rows = db.fetch_all("SELECT prompt_tokens, completion_tokens, cached_tokens FROM llm_usage WHERE id > ?", (mark,))
    return {"calls": len(rows), "prompt": sum(r["prompt_tokens"] or 0 for r in rows),
            "cached": sum(r["cached_tokens"] or 0 for r in rows), "completion": sum(r["completion_tokens"] or 0 for r in rows)}


def _render(entries: list[dict[str, Any]]) -> str:
    out = []
    for e in entries:
        t = datetime.fromisoformat(e["t"])
        tools = f"  〔用了工具：{'、'.join(x.split('__')[-1] for x in e['tools'])}〕" if e.get("tools") else ""
        out.append(f"[{_label(t)}] {e['who']}：{e['text']}{tools}")
    return "\n".join(out)


def _owner_turn(models: Models, track: dict[str, Any], session: dict[str, Any], now: datetime, known: list[str],
                chat: list[dict[str, Any]], turn: int) -> tuple[list[str], bool]:
    if models.mock:
        canned = ["在吗", "今天好累哦", "哈哈好吧", "那我先去忙啦"]
        return [canned[min(turn, len(canned) - 1)]], turn >= 3
    from ai import persona_templates as pt
    data = pt.persona_data(None)
    rel = pt.own_relation(data)
    who = (f"她是你的{rel['label']}：" + pt._fill(str(rel["text"]), data).replace("对方", "你")) if rel else \
        f"她是一个叫{(data.get('identity') or {}).get('name') or 'Violet'}的 AI 陪伴角色，你们是网友，挺熟。"
    user = (f"你是谁：{track['style']}\n\n{who}\n\n现在是 {_label(now)}。\n\n这次聊天要做的事：{session['agenda']}\n\n"
            + (f"你以前跟她说过的事：\n- " + "\n- ".join(known) + "\n\n" if known else "")
            + "这次聊天到现在（包括你不在时她发来的消息）：\n" + (_render(chat) if chat else "（还没开始，你先说）")
            + f"\n\n这是你的第 {turn + 1} 次发言" + ("，该结束了。" if turn + 1 >= session["max_turns"] else "。"))
    data = _json(models.ask(models.owner_model, OWNER_SYSTEM, user, 0.8)) or {}
    lines = [str(x).strip() for x in data.get("messages") or [] if str(x).strip()][:3]
    return lines, bool(data.get("end")) or turn + 1 >= session["max_turns"]


def _judge(models: Models, track: dict[str, Any], day: int, entries: list[dict[str, Any]], known: list[str],
           probes: list[tuple[str, str]], stage: Any, reminders: list[dict[str, Any]],
           set_today: list[dict[str, Any]], earlier: list[dict[str, Any]],
           memes: list[tuple[str, str]] | None = None) -> dict[str, Any]:
    if models.mock or not entries:
        return {"problems": [], "probes": [], "good": [], "scores": {}}
    from ai import persona_templates as pt
    data = pt.persona_data(None)
    persona = (f"人设：{data.get('_title')}。{(data.get('identity') or {}).get('self_description', '')}"
               f"\n性格：{'；'.join(data.get('personality') or [])}")
    rel = pt.own_relation(data)
    if rel:
        persona += f"\n和主人的关系：{rel['label']}。" + (f"现在的关系阶段：{stage['label']}（{stage['text']}）" if stage else "")
        past = [pt._fill(x, data) for x in data.get("backstory") or []]
        if past:
            persona += "\n人设里写好的她和主人的过去（她提到这些不算编造）：" + "；".join(past)
    user = (f"{persona}\n\n主人的资料：{json.dumps(track['about'], ensure_ascii=False) if track['about'] else '（没填）'}\n\n"
            f"到今天为止主人告诉过她的事（以最新的为准）：\n- " + ("\n- ".join(known) if known else "（无）")
            + "\n\n今天的考题：\n" + ("\n".join(f"- 问：{q}；正确答案：{a}" for q, a in probes) if probes else "（无）")
            + "\n\n今天主人用的梗（主人没解释，看她接没接住）：\n"
            + ("\n".join(f"- {m}：{meaning}" for m, meaning in memes) if memes else "（无）")
            + "\n\n今天到点的提醒（可能是前几天设的；set=true 表示确实设了，delivered_at 是系统实际发出的时间；"
              "提醒是系统按时自动发的，聊天里标“她（提醒）”的就是它）：\n" + (json.dumps(reminders, ensure_ascii=False) if reminders else "（无）")
            + "\n\n她今天新设的提醒（系统记录；她设提醒是在后台调用工具，聊天里看不到，以这里为准）：\n"
            + (json.dumps(set_today, ensure_ascii=False) if set_today else "（无）")
            + "\n\n前几天主人说过的话（原文，判断她是不是编造时以这里和上面为准）：\n"
            + (_render([e for e in earlier if e["who"] == "主人"][-150:]) or "（无）")
            + f"\n\n第 {day} 天的聊天记录：\n{_render(entries)}")
    result = _json(models.ask(models.judge_model, JUDGE_SYSTEM, user, 0.2))
    return result if isinstance(result, dict) else {"problems": [], "probes": [], "error": "评审没有给出 JSON"}


SENT_SPLIT = re.compile(r"[。！？!?~～…\n]+")


def code_checks(entries: list[dict[str, Any]], call: str) -> dict[str, Any]:
    """What code can count in her replies: messages per answer, length, questions, repeated sentences, the call."""
    replies: list[list[str]] = []
    group: list[str] = []
    for e in entries:
        if e["who"] == "她":
            group.append(e["text"])
        elif group:
            replies.append(group)
            group = []
    if group:
        replies.append(group)
    seen: dict[str, int] = {}
    repeats = []
    for e in entries:
        if not e["who"].startswith("她"):
            continue
        for sentence in (s.strip() for s in SENT_SPLIT.split(e["text"])):
            if len(sentence) >= 6:
                seen[sentence] = seen.get(sentence, 0) + 1
                if seen[sentence] == 2:
                    repeats.append(sentence)
    texts = [t for g in replies for t in g]
    return {"answers": len(replies), "messages": len(texts),
            "max_messages_per_answer": max((len(g) for g in replies), default=0),
            "answers_over_3_messages": sum(1 for g in replies if len(g) > 3),
            "avg_chars": round(sum(len(t) for t in texts) / max(len(texts), 1), 1),
            "answers_ending_with_question": sum(1 for g in replies if g[-1].rstrip().endswith(("？", "?"))),
            "repeated_sentences": repeats[:10],
            "call_in_messages": sum(1 for t in texts if call and call in t)}


def run_track(name: str, args) -> dict[str, Any]:
    track = TRACKS[name]
    from ai.settings import local_timezone, set_clock
    start = datetime.fromisoformat(START).replace(tzinfo=local_timezone())
    clock = Clock(start)
    set_clock(clock)
    errors = ErrorLog()
    logging.getLogger().addHandler(errors)
    world = World(track, clock, args.mock)
    models = Models(args.owner_model, args.judge_model, args.mock)
    from ai import persona_templates as pt
    db = world.db
    sessions = [s for s in track["sessions"] if s["day"] <= args.days]
    known: list[str] = []
    expected_reminders: list[dict[str, Any]] = []
    days: list[dict[str, Any]] = []
    mark = int((db.fetch_one("SELECT MAX(id) AS n FROM llm_usage") or {}).get("n") or 0)
    stopped = None
    last_day = max((s["day"] for s in sessions), default=1)
    say(f"== {name}：{track['note']}（{len(sessions)} 段聊天，{last_day} 天）")
    for day in range(1, last_day + 1):
        day_start = len(world.transcript)
        day_mark = int((db.fetch_one("SELECT MAX(id) AS n FROM llm_usage") or {}).get("n") or 0)
        owner_before = dict(models.usage)
        latencies: list[float] = []
        probes: list[tuple[str, str]] = []
        memes: list[tuple[str, str]] = []
        for session in [s for s in sessions if s["day"] == day]:
            memes += session.get("memes") or []
            at = _at(start, day, session["at"])
            chat = world.advance(at)
            for what, due in session["reminders"]:
                d, hhmm = due.split()
                expected_reminders.append({"what": what, "due": _at(start, int(d), hhmm).isoformat(timespec="minutes")})
            probes += session["probes"]
            ended = False
            turn = 0
            while not ended:
                try:
                    lines, ended = _owner_turn(models, track, session, clock(), known, chat, turn)
                except Exception as exc:                            # noqa: BLE001
                    errors.lines.append(f"owner model: {exc}"[:300])
                    break
                if not lines:
                    break
                chat += [{"t": clock().isoformat(timespec="minutes"), "who": "主人", "text": x} for x in lines]
                replies, seconds = world.owner_says(lines)
                latencies.append(round(seconds, 1))
                chat += replies
                turn += 1
                clock.t = clock() + timedelta(minutes=2)
                used = _her_usage(db, mark)
                total = used["prompt"] + used["completion"] + models.usage["prompt"] + models.usage["completion"]
                if total >= args.max_tokens:
                    stopped = f"到了 token 上限（{total}）"
                    break
            known += session["facts"]
            if stopped:
                break
        if not stopped:
            world.advance(_at(start, day + 1, "06:00"))               # the rest of the day and the night
        entries = world.transcript[day_start:]
        stage = pt.stage(None, db)
        day_end = _at(start, day + 1, "06:00")
        reminders = []
        for r in expected_reminders:
            if not (_at(start, day, "06:00").isoformat() <= r["due"] < day_end.isoformat()):
                continue
            row = db.fetch_one("SELECT due_at, delivered_at, status, content FROM commitments WHERE due_at LIKE ? "
                               "ORDER BY id DESC LIMIT 1", (r["due"][:13] + "%",))
            reminders.append({**r, "set": bool(row), "status": (row or {}).get("status"),
                              "due_at": (row or {}).get("due_at"), "delivered_at": (row or {}).get("delivered_at")})
        all_set = [dict(r) for r in db.fetch_all("SELECT content, due_at, status, delivered_at, created_at FROM commitments")]
        day_from, day_to = _at(start, day, "06:00").isoformat(), day_end.isoformat()
        set_today = [{k: r[k] for k in ("content", "due_at", "status")} for r in all_set
                     if day_from <= (r.get("created_at") or "") < day_to]
        proactive = [{"at": r["created_at"][:16], "status": r["status"], "type": r["event_type"],
                      "reason": str(r.get("reason") or "")[:160], "decision": str(r.get("decision") or "")[:300]}
                     for r in db.fetch_all("SELECT * FROM proactive_events WHERE created_at >= ? AND created_at < ? "
                                           "ORDER BY id", (day_from, day_to))]
        try:
            judged = _judge(models, track, day, entries, known, probes, stage, reminders, set_today,
                            world.transcript[:day_start], memes)
        except Exception as exc:                                    # noqa: BLE001
            judged = {"error": str(exc)[:300]}
        rel = pt.relationship()
        her = _her_usage(db, day_mark)
        day_report = {
            "day": day, "date": _label(_at(start, day, "00:00"))[:-6], "stage": stage and stage["label"],
            "transcript": entries, "code": code_checks(entries, rel.get("call") or ""), "judge": judged,
            "reminders": reminders, "commitments": all_set, "reminders_set_today": set_today,
            "proactive_events": proactive, "latency_seconds": latencies,
            "proactive": sum(1 for e in entries if e["who"] == "她（主动）"),
            "errors": errors.lines[:], "her_usage": her,
            "owner_usage": {k: models.usage[k] - owner_before[k] for k in models.usage},
            "mood": _mood(db),
        }
        errors.lines.clear()
        days.append(day_report)
        say(_day_line(day_report))
        if stopped:
            say(f"   停止：{stopped}")
            break
    world.runtime.stop(2)
    set_clock(None)
    her = _her_usage(db, mark)
    return {"track": name, "note": track["note"], "persona": track["persona"], "days": days, "stopped": stopped,
            "usage": {"her": her, "owner_and_judge": models.usage,
                      "her_model": os.environ.get("CHAT_LLM_MODEL"), "owner_model": args.owner_model,
                      "judge_model": args.judge_model},
            "memories": [dict(r) for r in db.fetch_all(
                "SELECT kind, content, event_time, progress, status FROM event_memories ORDER BY id")],
            "owner_memes": [dict(r) for r in db.fetch_all("SELECT phrase, meaning, source FROM owner_memes ORDER BY id")],
            "meme_wiki": _meme_wiki(),
            "profile": [dict(r) for r in db.fetch_all(
                "SELECT fact_key, value, status FROM user_profile_facts ORDER BY id")]}


def _meme_wiki() -> dict[str, Any]:
    from ai import memes
    return {"enabled": memes.enabled(), **memes.wiki_info()}


def _mood(db) -> Any:
    try:
        from ai import mood
        return round(mood.current(db)["value"], 2)
    except Exception:                                               # noqa: BLE001
        return None


def _day_line(d: dict[str, Any]) -> str:
    j, c = d.get("judge") or {}, d["code"]
    probs = j.get("problems") or []
    high = sum(1 for p in probs if p.get("severity") == "high")
    probes = j.get("probes") or []
    right = sum(1 for p in probes if p.get("correct"))
    memes = j.get("memes") or []
    her, owner = d["her_usage"], d["owner_usage"]
    return (f"  第 {d['day']} 天 {d['date']}" + (f"（阶段：{d['stage']}）" if d["stage"] else "")
            + f"：她回了 {c['answers']} 次 / {c['messages']} 条，主动 {d['proactive']} 次；问题 {len(probs)}（严重 {high}）"
            + (f"；考题 {right}/{len(probes)}" if probes else "")
            + (f"；梗 {sum(1 for m in memes if m.get('got_it'))}/{len(memes)} 接住" if memes else "")
            + (f"；提醒 {sum(1 for r in d['reminders'] if r['delivered_at'])}/{len(d['reminders'])} 发出" if d["reminders"] else "")
            + (f"；报错 {len(d['errors'])}" if d["errors"] else "")
            + f"；她 {her['calls']} 次调用 {her['prompt'] // 1000}k 输入（缓存 {100 * her['cached'] // max(her['prompt'], 1)}%）"
            + f"，主人+评审 {(owner['prompt'] + owner['completion']) // 1000}k")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Violet virtual long-run test")
    parser.add_argument("--env", type=Path, help="the server's violet.env")
    parser.add_argument("--track", default="long", choices=sorted(TRACKS))
    parser.add_argument("--all", action="store_true", help="every track, each in its own process")
    parser.add_argument("--days", type=int, default=99, help="run only the first N virtual days")
    parser.add_argument("--owner-model", default="gpt-6-luna")
    parser.add_argument("--judge-model", default="gpt-6-luna")
    parser.add_argument("--max-tokens", type=int, default=2_500_000, help="per track")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--memes-file", type=Path, help="a meme wiki file (ai/memes.py) to import into the test's data")
    parser.add_argument("--no-memes", action="store_true", help="the meme feature off (VIOLET_MEMES=0)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.all:
        reports = []
        for name in TRACKS:
            part = args.out.with_name(args.out.stem + f"-{name}.json")
            cmd = [sys.executable, __file__, "--track", name, "--days", str(args.days), "--owner-model", args.owner_model,
                   "--judge-model", args.judge_model, "--max-tokens", str(args.max_tokens), "--out", str(part)]
            cmd += ["--env", str(args.env)] if args.env else []
            cmd += ["--mock"] if args.mock else []
            cmd += ["--memes-file", str(args.memes_file)] if args.memes_file else []
            cmd += ["--no-memes"] if args.no_memes else []
            subprocess.run(cmd, check=False)
            try:
                reports.append(json.loads(part.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                reports.append({"track": name, "error": "没有报告（这一项中途失败）"})
        args.out.write_text(json.dumps({"tracks": reports}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"全部报告：{args.out}")
        return 0
    if not args.mock and not args.env:
        parser.error("--env is needed for a real run")
    work = Path(tempfile.mkdtemp(prefix="violet-vt-"))
    try:
        servers = start_fake_tools()
        prepare(args.env, work, servers, args.mock)
        os.environ.update({"VIOLET_ASYNC_MEMORY": "1", "VIOLET_LIFE_ENGINE": "1", "VIOLET_HOLIDAY_GREETING": "0",
                           "VIOLET_IMAGE_ENABLED": "0", "VIOLET_MEMES": "0" if args.no_memes else "1"})
        if args.memes_file:
            from ai import memes
            memes.import_file(args.memes_file)
        _copy_proactive_settings(args.env, work)
        logging.basicConfig(level=logging.ERROR, force=True)
        if not os.environ.get("LLM_API_KEY", "").strip() and not args.mock:
            print("没有找到模型的 API Key，停止", flush=True)
            return 1
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            report = run_track(args.track, args)
        report["started"] = START
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"报告：{args.out}", flush=True)
        return 0
    finally:
        from ai.settings import set_clock
        set_clock(None)
        shutil.rmtree(work, ignore_errors=True)


def _copy_proactive_settings(env_file: Path | None, work: Path) -> None:
    """The owner's proactive settings (quiet hours, daily limit ...) from the Control Center, nothing else."""
    if not env_file:
        return
    from tools.scenario_eval import _parse_env
    real_data = Path(_parse_env(env_file).get("VIOLET_DATA_DIR") or APP / "data")
    try:
        values = json.loads((real_data / "control" / "settings.json").read_text(encoding="utf-8")).get("values") or {}
    except (OSError, ValueError):
        return
    keep = {k: v for k, v in values.items() if k.startswith("proactive.")}
    if keep:
        (work / "data" / "control" / "settings.json").write_text(json.dumps({"revision": 1, "values": keep}),
                                                                  encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
