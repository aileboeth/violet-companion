"""Consistency guard (phase 4): checks run in code, not by prompt.

1. check_event(proposal)  -- before a life event is stored:
     external_as_experience   internet/news material (or a public fact) proposed as something Violet lived
     reading_record_*         a reading record must only say "read an article about X" and carry its url
     future_event             occurs after "now"
     thread_not_active        thread missing or not active
     before_thread_start      earlier than the thread began
     out_of_order             earlier than the thread's latest event
     time_conflict            overlaps another recorded event
     plan_not_open            completes a plan that is not open / not in this thread
     completion_without_plan  says "finished X" without resolving the plan
     forbidden_backstory      family / job / school / romance ... (persona core)
     identity_conflict        claims to be human etc. (persona core)
     user_experience          looks like (or is marked as) the USER's experience
     too_dramatic             accident, illness, moving abroad ...
     duplicate                same content in this thread recently
   Any violation -> nothing is written; a consistency_flag is recorded.

2. check_reply(text)  -- Violet's generated message, line by line:
     unrecorded_experience    "我昨天去了…" with no recorded happened event behind it
     plan_described_as_done   talks about a plan (not completed) as already done
     user_experience_as_own   retells the user's experience as Violet's
     forbidden_backstory / identity_conflict
     external_as_experience   retells a web finding as something Violet did (phase 5B)
     camera_claim             presents a picture as a photo / selfie / camera shot (phase 6)
   certain:   forbidden_backstory, identity_conflict, plan_described_as_done, external_as_experience, camera_claim
   uncertain: unrecorded_experience, user_experience_as_own -- may be reviewed by
              the optional semantic check (ai/semantic_guard.py)
   The caller (ai.chat.guard_reply) asks for one natural rephrase, then drops
   only the lines that still fail.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from ai.memory_model import mentions_topic, parse_time, touches_topic
from ai.persona import PersonaCore, life_config
from ai.settings import env_str

EXTERNAL_ORIGINS = {"web", "news", "search", "external", "rss", "web_mcp"}
COMPLETION_RE = re.compile(r"(读完|听完|写完|看完|整理好|整理完|整理成|完成了|听熟了)")
DRAMATIC_RE = re.compile(r"(车祸|事故|住院|手术|生病|分手|失业|辞职|中奖|出国|移民|搬家|去世|死了|吵架|打架|被偷|受伤)")
# first person + a completed action within the same clause: "我昨天把《边城》读完了", "昨天我去了…"
_PAST_VERBS = (r"(?:去了|读了|看了|听了|写了|买了|学了|练了|吃了|玩了|做了|逛了|见了|弹了|画了|记了|整理了|"
               r"爬了|跑了|游了|骑了|拍了|唱了|喝了|参加了|报了|逛完|爬完|跑完|"
               r"读完|看完|听完|写完|整理好|整理完|听熟了|"
               r"吃完|喝完|煮了|煮好|做好|炒了|蒸了|烤了|烤好|泡了|泡好|洗完|洗了|睡醒|"
               r"(?<![思参])考了|(?<![思参])考完|吃到|睡到|忙完|加了个?蛋)")
PAST_CLAIM_RE = re.compile(r"我[^，,。！!？?；;、\n]{0,12}?" + _PAST_VERBS)
# Phase 8.9.10 (real free chat ab-v2b): Chinese drops the "我" -- "看了开头几章，还没到特别揪心的地方" (asked "好看吗",
# about a book she had only planned to pick up). A clause that starts with the completed action is her claim too.
# 2026-11-25 (v0.7.80): also allow food/daily object subjects ("早饭吃到快中午", "面煮好了", "衣服洗完了") and two-part
# time/aspect prefixes ("今天刚吃完午饭").
_SUBJECTLESS_PREFIX = (
    r"(?:《[^》]{1,20}》|这本书?|那本书?|"
    r"(?:早|午|中|晚)(?:饭|餐)|(?:一[碗盘锅杯个份])?(?:饭|面条?|汤|粥|菜|饼干|面包|蛋糕|咖啡|奶茶|茶|牛奶|衣服|澡|头发|作业|笔记))?"
)
SUBJECTLESS_CLAIM_RE = re.compile(
    r"^" + _SUBJECTLESS_PREFIX + r"(?:我)?"
    r"(?:今天|昨天|前天|早上|上午|中午|下午|晚上|这两天|最近)?"
    r"(?:刚才|刚刚|刚|已经|又|也|才){0,2}"
    + _PAST_VERBS
)
# phase 8.9.13 (real ab-v2e): fear, worry, guesses and hearsay before the action are no claim either --
# "我怕我看完难受好几天", "都说看完心里沉甸甸的"
_HEDGE = (r"(没|不|别|想|打算|准备|要不要|要是|如果|会不会|可能会|下次|以后|觉得|感觉|好像|"
          r"怕|担心|万一|估计|可能|也许|会|听说|都说|据说|有人说|别人说|听人说)")
# "这名字我听过好多回了": heard OF a name / a title / what people say -- knowledge, not having listened to something
# words that only make sense after something said before them
_LEANING_START_RE = re.compile(r"^(不过|但是|可是|只是|而且|所以|然后|另外|再说)[，,、\s]*")
_HEARSAY_RE = re.compile(r"(名字|这名|书名|片名|听说|有人说|都说|据说|人提|提起|说起|听人)")
_VERB_FORMS = (
    r"(看了一下|看了看|看了一眼|看了眼|想了想|想了一下|看了你|听了你|吃了一惊|吃了亏|吃了一嘴|"
    r"做了个?(?:梦|鬼脸|深呼吸|决定|选择)|做好(?:了)?(?:心理)?(?:准备|打算|功课|计划)|"
    r"拍了(?:拍|一下|下)(?:桌子|大腿|脑门|胸口|脸|头)|"
    r"喝了(?:一?口水)?[^，,。！!？?]{0,6}(?:呛|喷)|"
    # a reaction, not an experience: "我听了好难过", "看了都想哭", "听了差点笑出声"
    r"(?:听|看)了[^，,。！!？?]{0,5}(?:难过|开心|心疼|感动|生气|高兴|难受|激动|担心|羡慕|好笑|想哭|想笑|笑出|愣|呆|困|饿|馋))"
)
# comforting/conditional/advice phrases with resultatives ("忙完了就好", "考完了就好", "吃完了早点睡", "洗完澡了别着凉", "我忙完就来")
_BARE_RESULTATIVE_RE = re.compile(r"(?:吃完|喝完|煮好|做好|烤好|泡好|洗完|考完|忙完|睡醒|写完|看完|读完|听完|逛完|爬完|跑完)$")
_COMPLETED_PREFIX_RE = re.compile(r"(?:刚才|刚刚|刚|已经|昨天|前天|上学期)")
_SLEEP_TIME_PREFIX_RE = re.compile(r"(?:刚才|刚刚|刚|已经|今天|昨天|前天|早上|上午|中午|下午|晚上)")
_FUTURE_OR_ADVICE_RE = re.compile(
    r"^[^，,。！!？?]{0,4}(?:"
    r"就(?:好|行|对|成|轻松|解放|舒坦|舒服|清爽|精神|清醒|踏实|安心|省心|暖和|凉快|痛快|香|爽|美|开心|自在|没事|完事|歇|睡|休息|算|能|可以|别|来|找|去|不|没)|"
    r"(?:还)?不得|岂不|还不|怕是|怕不是|肯定|绝对|估计|大概|都得|应该|能不|哪能|"
    r"再(?:说|聊|回|看|吃|喝|睡|去|联系|找|忙|想)|"
    r"记得|早点|好好|赶紧|快(?:去|吃|喝|睡|歇)|别(?:太|累|着凉|熬夜|多想|忘|客气)|"
    r"歇[会儿一歇]|休息|吹干|趁热|发(?:给)?我看)"
)
# bare subjectless "V完就…" without 了/刚/我 ("洗完就清爽多了", "喝完就精神了") is conditional encouragement, not a past claim
_BARE_SUBJECTLESS_JIU_RE = re.compile(r"^[^，,。！!？?了啦]{0,3}就")
# evaluative commentary on a sequence of actions the user just described ("刚看完这个就去吃饭，心理素质挺强啊")
_USER_TRAIT_COMMENT_RE = re.compile(
    r"(?:心理素质|心态(?:真|挺|好|够|太)|胆子(?:真|挺|好|够|太)|心(?:真|挺|好)大|"
    r"真有你的|不愧是你|挺佩服|真佩服|够淡定|挺淡定|好淡定|抗压能力)"
)
_SUBJECTLESS_COMFORT_TAIL_RE = re.compile(
    r"^[^，,。！!？?]{0,3}(?:就(?:好|行|轻松|舒坦|暖和|歇|睡|休息)|早点睡|好好休|好好歇|记得休息|记得喝水|别太累|别着凉|再聊|再说)"
)
_SENTENCE_END_RE = re.compile(r"[。！!？?；;]")
_USER_ATTRIBUTION_RE = re.compile(
    r"你[^，,。！!？?；;\n]{0,8}?(?:说|提|讲|告诉|聊|发|写|问|报备|答应|承诺|打算|计划|决定)"
    r"|按你(?:刚才|之前|中午|早上|上午|下午|晚上|昨天)?说的"
)
NOT_A_CLAIM_RE = re.compile(_HEDGE[:-1] + "|" + _VERB_FORMS[1:])
_HEDGE_RE, _VERB_FORMS_RE = re.compile(_HEDGE), re.compile(_VERB_FORMS)
_QUESTION_END_RE = re.compile(r"(?:吗|么|[?？]|(?<![淹沉埋吞隐出])没(?:有)?[呀啊呢嘛]?)\s*$")
# Phase 7 B.9.1: the experiential aspect is a claim of having done it too ("我也爬过山", "我去过泰山",
# "以前我爬过一次黄山", "我也去过"). Same verbs as the completed forms above -- the same policy, no wider net --
# plus going / visiting / staying somewhere. "过来 / 过去" (a direction) is not an experience.
_EXPERIENCE_VERBS = r"去|读|看|听|写|买|学|练|吃|玩|做|逛|见|弹|画|记|爬|跑|游|骑|拍|唱|喝|参加|报|到|登|住|(?<![思参])考|煮|烤|泡|洗|睡"
EXPERIENCE_CLAIM_RE = re.compile(rf"我[^，,。！!？?；;、\n]{{0,12}}?(?:{_EXPERIENCE_VERBS})过(?![来去])")
SUBJECTLESS_EXPERIENCE_RE = re.compile(
    r"^(?=(?:刚才|刚刚|刚|已经|以前|之前|去年|上学期|上个月|上周|前阵子|昨天|今天|前天))"
    r"(?:以前|之前|去年|上学期|上个月|上周|前阵子|昨天|今天|前天)?"
    r"(?:刚才|刚刚|刚|已经|也|才|就){0,2}"
    r"(?:吃|喝|(?<![思参])考|去|看|读|煮|烤|泡|洗)过(?![来去头])"
)
# only for the experiential form: never ("从未去过") and questions ("我们一起去过吗", "我吃过没") are not claims
EXPERIENCE_NOT_A_CLAIM_RE = re.compile(r"(从未|未曾|(?:吗|么|(?<![淹沉埋吞隐出])没(?:有)?[呀啊呢嘛]?)\s*$)")
# a recorded "读了几章《边城》" backs "我读过《边城》": compare the experiential form as the completed one
EXPERIENCE_AS_DONE_RE = re.compile(rf"({_EXPERIENCE_VERBS})过(?![来去])")
USER_REFERENCE_RE = re.compile(r"你")
# Phase 8.9.9 (real free chat ab-v2): being somewhere or having got something, said without the completed verbs above --
# "我出门啦，人已经站在书店门口了" (the brief said "等下就要出门"), "听你的，我带回家了" (the user had only suggested it).
# The same records decide as for "我去了…". Suppositions and plans ("等我到了", "待会儿出门", "周末要去"), questions and
# what the user does ("你到家了吗") are not claims.
_PLACES = r"(书店|店里|店门口|门口|公园|咖啡馆|超市|图书馆|商场|书架前|柜台)"
STATE_CLAIM_RE = re.compile(rf"((出门|出发)(了|啦)|已经(站在|在|到)[^，,。！!？?]{{0,8}}{_PLACES}|"
                            rf"(站在|坐在|到了|到啦)[^，,。！!？?]{{0,6}}{_PLACES}|到{_PLACES}(了|啦)|"
                            r"(带|拿|买|借)回(家|来)(了|啦)|(买|拿|借)到(了|啦)|(回到家|到家)(了|啦))")
# "站在书架前肯定又要纠结半天" (the real 22:01 message) is what happens whenever she is there, not where she is now
STATE_HEDGE_RE = re.compile(r"(等|待会|一会儿|等会|等下|马上|要|会|肯定|总|每次|时候|的话|才|要是|如果|打算|准备|想|"
                            r"下次|以后|明天|周末|吗|么|？|\?)")
# Phase 9.3: the act of looking something up online this turn ("刚查了一下", "看了原文", "搜到一篇") -- not a lived
# experience ("去了", "吃了", "见到" stay checked: a page's content is never Violet's life).
WEB_ACTION_RE = re.compile(
    r"(查|搜|搜索|看|读|翻|点开)(了|过|到)?(一下|下|一遍)?(网上|网页|原文|文章|这篇|那篇|报道|新闻|资料|搜索结果|链接|结果|官网|公告)"
    r"|(刚|刚刚|刚才)(帮你)?(查|搜|搜索|看)(了|过)?(一下|下)?"
    r"|(网上|搜索结果|原文|那篇|这篇)(里|上)?(说|写|提到)")
LIVED_RE = re.compile(r"(去了|去过|吃了|吃完|吃到|喝了|喝完|煮了|煮好|烤了|烤好|泡了|泡好|见了|见到|买了|玩了|参加了|逛了|拍了|(?<![思参])考了|(?<![思参])考完|(?<![思参])考过)")
# Phase 9.4.2: the act of looking at the user's own timetable this turn ("看了下你的课表", "查了一下课表").
SCHEDULE_ACTION_RE = re.compile(
    r"(查|看|翻|对)(了|过)?(一下|下|一眼|了一下)?(你的)?(课表|课程表)"
    r"|(刚|刚刚|刚才)(帮你)?(查|看|翻)(了|过)?(一下|下)?")

# Phase 6: Violet's pictures are AI-generated illustrations, never photos (first person only:
# "你的自拍好可爱" is fine).
CAMERA_CLAIM_RE = re.compile(
    r"(我的?自拍|我(?:刚|刚刚|刚才|今天|昨天)?(?:用手机)?(?:拍|照)(?:的|了|下|给你)|(?:用|我的)摄像头|实拍|"
    r"这是(?:一张)?真实的?照片|真人照片|现拍|原图直出|手机拍的)")
CAMERA_NEGATED_RE = re.compile(r"(不是|并非|不算|没有|没|不能算)[^，,。！!？?]{0,6}(自拍|拍|照片|摄像头|实拍)")
_PAT_GESTURE_RE = re.compile(r"拍(?:了(?:拍|一下|下)|拍)(?:桌子|大腿|脑门|胸口|脸|头|肩|肩膀|手|你)")
CLAUSE_SPLIT_RE = re.compile(r"[，,。！!？?；;~～…]+")
# Phase 9.4 (real run cys-p94-sched-1 #4, "我们英语老师要是也这样就好了"): the persona's school / work patterns only see
# "我们老师", "我们的同学"; a word between the possessive and the role ("我们英语老师", "我的数学同学", "我们这边公司")
# slipped through. Up to four characters are allowed there -- but not an action or a link ("我们一起去问老师",
# "如果我们是同学" stay allowed: they do not say Violet has that teacher / class / company). Hypotheticals without the
# possessive ("要是我也有个英语老师") are not touched.
_OWNED_ROLE_RE = re.compile(r"(我们|我的)的?([一-鿿]{0,4}?)(老师|导师|同学|室友|班主任|辅导员|学校|班上|公司|单位|老板|同事|领导)")
_NOT_A_MODIFIER = set("去问找跟和给让叫说告帮请见想要会能起也都还就是像当被把对向天")    # 天: "我的天" (an exclamation)
_WORK_BACKSTORY_RE = re.compile(r"我(?:今天|明天|昨天)?(?:要|得|在)?去?(?:上班|下班|加班|出差)")


CLAIMS_HUMAN_RE = re.compile(r"我(是|也是|就是|可是)(个|一个)?(真人|真实的人|人类|活人|活生生的人)")

# 2026-11-13 (0.7.72 server test, then 2 of 3 real runs after the abilities note said it): "我晚点再试一次",
# "我回头再帮你看看", "晚点我再瞅一眼" -- a lookup promised for later. Her later messages of her own have no tools, so
# such a promise is never kept (twice it became "天气又翻了一遍" without a search). Narrow on purpose: she, later, a
# lookup verb -- "你明早出门前再看一眼" (the owner) and "我晚点看看书" (her own life) are not it.
_LATER = r"(?:晚点|晚些|回头|等会儿?|等下|待会儿?|过会儿?|一会儿)"
_LOOKUP = r"(?:查|搜|瞅|瞄|再试|试一次|试试|看一眼|帮你(?:看|查|找|搜|瞅|瞄|问|盯))"
PROMISED_LOOKUP_RE = re.compile(rf"我[^，,。！!？?；;\n]{{0,4}}?{_LATER}[^，,。！!？?；;\n]{{0,4}}?{_LOOKUP}"
                                rf"|{_LATER}[^，,。！!？?；;\n]{{0,2}}?(?:我|帮你)[^，,。！!？?；;\n]{{0,3}}?{_LOOKUP}")
# On a turn where she did search (5 real runs of the narrow pattern: "晚点我再瞄一眼", "我这边再帮你找找", "等我晚上再
# 看一眼" got through), the lookup is what the turn was about: any clause of hers with a "later / again" word and a
# looking verb is that promise. Not what is done or failed now ("刚看了一眼", "明早的雨我这边查不到") and not the
# owner's own checking (no 我 / 帮你). "明早 / 明天" name what was searched for, not when.
LATER_WIDE_RE = re.compile(r"(再|晚点|晚些|回头|等会|等下|待会|过会|一会儿|晚上)")
LOOK_RE = re.compile(r"(查|搜|瞅|瞄|看|找|试|刷|盯|问问)")
DONE_NOW_RE = re.compile(r"(刚|已经|又|了|过|不到|没|不着|不了)")


def promised_lookup(line: str, searched: bool) -> bool:
    """A lookup promised for later in this line (``searched``: this turn used the web tools)."""
    if PROMISED_LOOKUP_RE.search(line):
        return True
    if not searched:
        return False
    return any(re.search(r"我|帮你", c) and LATER_WIDE_RE.search(c) and LOOK_RE.search(c) and not DONE_NOW_RE.search(c)
               for c in CLAUSE_SPLIT_RE.split(line))


def owned_role_claim(text: str) -> str | None:
    """"我们英语老师" -> the matched words; None when nothing claims a teacher / class / company of Violet's own."""
    for match in _OWNED_ROLE_RE.finditer(text or ""):
        if not set(match.group(2)) & _NOT_A_MODIFIER:
            return match.group(0)
    return None


# Phase 8.9.4: bigrams that only carry grammar. "我刚刚也随手听了会儿歌" was "backed" by the recorded "在公园散了会儿步"
# through 了会 / 会儿 alone (8.9.2 real E1). Backing a claim needs shared CONTENT.
FILLER_BIGRAMS = frozenset({"了会", "会儿", "一会", "了一", "一下", "一点", "有点", "一遍", "了个", "一个", "一些", "了些",
                            "了下", "下了", "着呢", "了吧", "了呀", "的时", "时候", "一直", "然后", "就是", "还是", "了好"})


def _content_bigrams(text: str) -> set[str]:
    from ai.memory_model import _bigrams
    return _bigrams(text) - FILLER_BIGRAMS


def backed_by(text: str, content: str) -> bool:
    """A recorded event backs a line when they share content -- same thresholds as mentions_topic, without fillers."""
    text_grams = _content_bigrams(text)
    if not text_grams:
        return False
    topic = _content_bigrams(content)
    if not topic:
        return False
    hits = topic & text_grams
    if len(topic) <= 4:
        return len(hits) >= 1
    if len(hits) >= 3 or (len(hits) >= 2 and len(hits) / len(topic) >= 0.25):
        return True
    clauses = [c.strip() for c in re.split(r"[，,。；;]+", content or "") if c.strip()]
    if len(clauses) > 1:
        for clause in clauses:
            ctopic = _content_bigrams(clause)
            if len(ctopic) >= 4:
                chits = ctopic & text_grams
                if len(chits) >= 2 and len(chits) / len(ctopic) >= 0.25:
                    return True
    return False


_POSSESSION_CLAIM_RE = re.compile(r"(带|拿|买|借)(回|到)")
_POSSESSION_EVENT_RE = re.compile(r"(买|挑|借|拿|带回|选了|入手)")


def _state_backed(line: str, recent: list[dict], now) -> bool:
    """Phase 8.9.9: "人已经站在书店门口了" is backed by a recorded trip to the bookstore within the last day, "带回家了" by
    something recorded as bought / picked / borrowed then -- being somewhere or having something follows from them."""
    places = set(m.group(0) for m in re.finditer(_PLACES, line))
    possession = bool(_POSSESSION_CLAIM_RE.search(line))
    for event in recent:
        if now - parse_time(event["occurred_at"]) > timedelta(hours=24):
            continue
        content = event["content"] or ""
        if any(p in content for p in places) or (possession and _POSSESSION_EVENT_RE.search(content)):
            return True
    return False


# what an earlier clause of the same sentence sets in the future or the conditional: "我大概会挑个白天看，看完了去公园
# 走走" (real ab-v2) -- the subjectless "看完了" is part of that plan
_FUTURE_RE = re.compile(r"(会|要|打算|想|准备|等|如果|要是|的话|周末|明天|下次|以后|待会|一会儿|到时候)")


# a subjectless clause that asks ("看了什么电影呀？") is a question to the user, not her claim
# ("几" / "多少" are left out: "看了开头几章" is a claim)
_ASKING_WORD_RE = re.compile(r"(什么|啥|哪|谁|怎么|有没有|是不是)")


def _claims(line: str) -> list[str]:
    out, future, user_attributed = [], False, False
    pieces = re.split(f"({CLAUSE_SPLIT_RE.pattern})", line)
    # Pre-scan sentence spans for second-person trait commentary ("刚看完这个就去吃饭，心理素质挺强啊")
    clauses = [(pieces[i], pieces[i + 1] if i + 1 < len(pieces) else "") for i in range(0, len(pieces), 2)]
    sent_start = 0
    trait_comment_idx: set[int] = set()
    for idx, (clause, after) in enumerate(clauses):
        if _SENTENCE_END_RE.search(after) or idx == len(clauses) - 1:
            span = clauses[sent_start:idx + 1]
            span_text = "".join(c for c, _ in span)
            if "我" not in span_text and any(_USER_TRAIT_COMMENT_RE.search(c) for c, _ in span):
                for j in range(sent_start, idx + 1):
                    if "就" in clauses[j][0] and not _USER_TRAIT_COMMENT_RE.search(clauses[j][0]):
                        trait_comment_idx.add(j)
            sent_start = idx + 1

    for idx, (clause, after) in enumerate(clauses):
        asking = bool(re.search(r"[?？]", after) or _ASKING_WORD_RE.search(clause))
        has_self = bool(re.search(r"我", clause))
        if has_self and not _USER_ATTRIBUTION_RE.search(clause):
            user_attributed = False
        is_trait_comment = idx in trait_comment_idx
        part = _claim(
            clause,
            subjectless=not (future or asking or user_attributed or is_trait_comment),
            user_attributed=user_attributed or is_trait_comment,
        )
        if part is not None:
            out.append(part)
        if _SENTENCE_END_RE.search(after):
            future = False
            user_attributed = False
        else:
            future = future or bool(_FUTURE_RE.search(clause))
            if _USER_ATTRIBUTION_RE.search(clause) and not asking:
                user_attributed = True
    return out


def _claim(clause: str, subjectless: bool = True, user_attributed: bool = False) -> str | None:
    """The part of a clause that claims Violet did something (from its start to the claimed action), or None.

    Phase 8.9.10: for completed actions a hedge or "你" counts only BEFORE the action -- "我记了两处想跟你说的" is a claim
    (想 / 你 are about the notes), "我想去了" / "我陪你去了" are not her claim. Verb forms that are no experience
    ("看了一下", "看了你发的", "听了好难过") still leave the whole clause out."""
    clause = clause.strip()
    if not clause or _VERB_FORMS_RE.search(clause):
        return None
    m = PAST_CLAIM_RE.search(clause) or (SUBJECTLESS_CLAIM_RE.search(clause)
                                         if subjectless and not _QUESTION_END_RE.search(clause) else None)
    if m is not None:
        part = clause[:m.end()]
        if _HEDGE_RE.search(part):
            return None
        tail = clause[m.end():]
        if _BARE_RESULTATIVE_RE.search(m.group(0)) and not _COMPLETED_PREFIX_RE.search(part):
            if _FUTURE_OR_ADVICE_RE.search(tail) or ("我" not in part and _BARE_SUBJECTLESS_JIU_RE.search(tail)):
                return None
        if "我" not in part and not _COMPLETED_PREFIX_RE.search(part) and _SUBJECTLESS_COMFORT_TAIL_RE.search(tail):
            return None
        if (m.group(0).endswith("睡到") and "自然醒" in tail
                and not _SLEEP_TIME_PREFIX_RE.search(part) and "我" not in part and not re.match(r"^自然醒[了啦]", tail)):
            return None
        return part
    if NOT_A_CLAIM_RE.search(clause):
        return None
    if EXPERIENCE_CLAIM_RE.search(clause) and not EXPERIENCE_NOT_A_CLAIM_RE.search(clause):
        if "听过" in clause and _HEARSAY_RE.search(clause):
            return None
        return clause
    if (subjectless and not _QUESTION_END_RE.search(clause)
            and SUBJECTLESS_EXPERIENCE_RE.search(clause) and not EXPERIENCE_NOT_A_CLAIM_RE.search(clause)):
        return clause
    if (not (user_attributed and "我" not in clause)
            and STATE_CLAIM_RE.search(clause) and not STATE_HEDGE_RE.search(clause) and not USER_REFERENCE_RE.search(clause)):
        return clause
    return None


# 2026-11-25 (v0.7.80): lightweight deterministic same-day self-contradiction guard.
# Compares factual statements Violet made about herself today (meals eaten vs not eaten, late breakfast after
# already finishing lunch, stayed-in-all-day after already going out) -- time-directional, zero false positives.
_MEAL_TERMS = {
    "breakfast": r"(?:早饭|早餐|早点)",
    "lunch": r"(?:午饭|午餐|中饭)",
    "dinner": r"(?:晚饭|晚餐)",
}
_MEAL_LABELS = {"breakfast": "早饭", "lunch": "午饭", "dinner": "晚饭"}
_MEAL_NEG_OR_PLAN_RE = re.compile(r"(?:没|不|别|还没|未|想|打算|准备|要去|待会|等会|一会儿|等下|再吃|要不要)")
_MEAL_HYPO_RE = re.compile(r"(?:如果|要是|本来|差点|以为|听说|都说|据说)")
_NON_TODAY_RE = re.compile(
    r"(?:昨天|昨晚|昨儿|前天|大前天|明天|明早|明晚|后天|大后天|"
    r"上周|上礼拜|上星期|上个月|上学期|去年|前年|以前|之前|上次|那天|那次|周末|平时|一般|每次|通常)"
)
_LATE_BREAKFAST_RE = re.compile(
    r"(?:早饭|早餐)[^，,。！!？?]{0,4}吃到(?:快)?中午|快中午才吃(?:上|完)?(?:早饭|早餐)"
)
_WENT_OUT_RE = re.compile(
    r"(?:刚才|刚刚|刚|今天|早上|上午|中午|下午|晚上|已经)?(?:出门[了啦]|"
    r"去了(?:公园|书店|超市|咖啡馆|图书馆|食堂|操场|菜市场|后街|小卖部|便利店|文具店|面包店|花店|奶茶店|教室|教学楼|活动室|自习室)|"
    r"去(?:公园|书店|超市|咖啡馆|图书馆|操场|后街|夜市)[^，,。！!？?]{0,6}[了啦完])"
)
_STAYED_IN_ALL_DAY_RE = re.compile(
    r"(?:今天|一整天|一天都|全天)都?(?:没出门|没有出门|一步没出|一直在家|一直待在家|一直宅在家|哪儿?[也没都]没去)"
)
_MEAL_ATE_RES = {
    meal: (
        re.compile(rf"(?:吃完|吃过|吃了|刚吃|已经吃|吃好|吃到)[^，,。！!？?]{{0,4}}{term}"),
        re.compile(rf"{term}[^，,。！!？?]{{0,6}}(?:吃完|吃过|吃了|吃好|吃到)"),
    )
    for meal, term in _MEAL_TERMS.items()
}
_MEAL_NOT_EATEN_RES = {
    meal: (
        re.compile(rf"(?:还没|没有|一直没|根本没|没)(?:来得及)?吃(?![饱完好多少啥什么几口太])(?:过|完|上)?[^，,。！!？?]{{0,3}}{term}"),
        re.compile(rf"{term}[^，,。！!？?]{{0,4}}(?:还没|没有|一直没|根本没|没)(?:来得及)?吃(?![饱完好多少啥什么几口太])"),
    )
    for meal, term in _MEAL_TERMS.items()
}
_BREAKFAST_SKIPPED_RE = re.compile(
    r"(?:(?:今天|早上)(?:没吃|跳过|省了|没来得及吃)(?:早饭|早餐)|"
    r"(?:早饭|早餐)(?:没吃|省了|跳过了)(?![饱完好多少啥什么几口太呢呀哦]))"
)


def _extract_today_facts(text: str) -> set[str]:
    """Extract monotonic self-facts stated by Violet in ``text`` (ignoring questions, hedges, non-today days, and user references)."""
    facts: set[str] = set()
    for raw_line in (text or "").splitlines():
        in_other_day = False
        user_attributed = False
        pieces = re.split(f"({CLAUSE_SPLIT_RE.pattern})", raw_line)
        for i in range(0, len(pieces), 2):
            clause = pieces[i].strip()
            after = pieces[i + 1] if i + 1 < len(pieces) else ""
            if not clause:
                if _SENTENCE_END_RE.search(after):
                    user_attributed = False
                continue
            has_self = bool(re.search(r"我", clause))
            if has_self and not _USER_ATTRIBUTION_RE.search(clause):
                user_attributed = False
            asking = bool(re.search(r"[?？]", after) or _QUESTION_END_RE.search(clause) or _ASKING_WORD_RE.search(clause))
            if _USER_ATTRIBUTION_RE.search(clause) and not asking:
                user_attributed = True
            has_today = "今天" in clause
            if has_today:
                in_other_day = False
            if _NON_TODAY_RE.search(clause):
                if not has_today:
                    in_other_day = True
                if _SENTENCE_END_RE.search(after):
                    user_attributed = False
                continue
            if in_other_day or user_attributed or USER_REFERENCE_RE.search(clause) or asking or _MEAL_HYPO_RE.search(clause):
                if _SENTENCE_END_RE.search(after):
                    user_attributed = False
                continue
            if _LATE_BREAKFAST_RE.search(clause) and not _MEAL_NEG_OR_PLAN_RE.search(clause[:_LATE_BREAKFAST_RE.search(clause).start()]):
                facts.add("late_breakfast")
                facts.add("ate:breakfast")
            for meal, (p1, p2) in _MEAL_NOT_EATEN_RES.items():
                if p1.search(clause) or p2.search(clause):
                    facts.add(f"not_eaten:{meal}")
            if "还没" not in clause and _BREAKFAST_SKIPPED_RE.search(clause) and not re.search(r"[呢呀哦]$", clause):
                facts.add("skipped:breakfast")
            for meal, (p1, p2) in _MEAL_ATE_RES.items():
                if f"not_eaten:{meal}" in facts:
                    continue
                m = p1.search(clause) or p2.search(clause)
                if m is not None and not _MEAL_NEG_OR_PLAN_RE.search(clause[:m.end()]):
                    facts.add(f"ate:{meal}")
            m_out = _WENT_OUT_RE.search(clause)
            if m_out is not None and not _MEAL_NEG_OR_PLAN_RE.search(clause[:m_out.start()]):
                facts.add("went_out")
            if _STAYED_IN_ALL_DAY_RE.search(clause):
                facts.add("stayed_in_all_day")
            if _SENTENCE_END_RE.search(after):
                user_attributed = False
    return facts


def _same_day_contradiction(line: str, prior_facts: set[str]) -> str | None:
    """Return a short reason if ``line`` directly contradicts what Violet already stated earlier today."""
    if not prior_facts:
        return None
    curr = _extract_today_facts(line)
    if not curr:
        return None
    for meal, label in _MEAL_LABELS.items():
        if f"ate:{meal}" in prior_facts and f"not_eaten:{meal}" in curr:
            return f"earlier today said she already ate {label}, now says she has not eaten {label}"
    if "ate:lunch" in prior_facts and "late_breakfast" in curr:
        return "earlier today said she already ate lunch, now says breakfast lasted until noon"
    if "skipped:breakfast" in prior_facts and ("ate:breakfast" in curr or "late_breakfast" in curr):
        return "earlier today said she skipped breakfast, now says she ate breakfast"
    if "went_out" in prior_facts and "stayed_in_all_day" in curr:
        return "earlier today said she went out, now says she stayed home all day"
    return None


def mentions_life_strict(text: str, content: str, config: dict | None = None) -> bool:
    """Titles / configured items / their details (ai.daily_life.life_terms), or content overlap without fillers."""
    from ai.daily_life import life_terms
    return any(t in (text or "") for t in life_terms(content, config)) or backed_by(text or "", content or "")


# Phase 8.9.4: Violet has no weather of her own. "今天也冷" (8.9.2 E1, the cold was the day before), "雨天确实容易无聊" /
# "很适合这种下雨天" (8.9.2 E5, the rain was two days before) retell an EARLIER conversation's weather as now. When the
# user has not brought the weather up in this conversation, such a line is stale. "适合下雨天看" is not about now.
STALE_WEATHER_RE = re.compile(
    r"((今天|今晚|现在|外面|这会儿|此刻)(也|又|还|真|好|挺|有点|特别|这么|那么)?(冷|热|凉|闷|下雨|在下雨|下着雨|下雪|刮风|降温|晴)"
    r"|今天的雨|(这种|这样的|今天这种)(下雨天|雨天|天气)|(雨天|下雨天)(确实|真的|就是)|雨下得)")
WEATHER_WORDS_RE = re.compile(r"(冷|热|凉|雨|雪|风|降温|升温|天气|晴|阴天|闷|晒|太阳|雾|霾)")


# Phase 8.9.5: the model's own drafting notes leaking into a reply (8.9.4 real A4, model output, unchanged by Violet's
# processing): "…说起呀陪伴? no. Question okay but perhaps don't mechanical. "我给你留着位置呢" maybe service-y but
# cute. Use 2 lines." A run of 5+ English words with 3+ distinct drafting words; "I miss you so much", "好的ok",
# "赶deadline" are untouched.
_ENGLISH_RUN_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*(?:[\s,.;:!?\"“”'()/]+[A-Za-z][A-Za-z'’\-]*){4,}")
# conversational words ("okay", "I'll", "keep") are left out: "I'll be there, don't worry about it okay" is a reply
_DRAFT_WORDS = frozenset({"maybe", "perhaps", "don't", "dont", "should", "shouldn't", "use", "line", "lines",
                          "question", "questions", "reply", "replies", "tone", "mechanical", "service", "service-y",
                          "draft", "output", "user", "avoid", "natural", "sounds", "emoji", "instead",
                          "options", "option", "version"})


def leaked_draft(line: str) -> bool:
    for run in _ENGLISH_RUN_RE.finditer(line or ""):
        words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'’\-]*", run.group(0))}
        if len(words & _DRAFT_WORDS) >= 3:
            return True
    return False


@dataclass
class Violation:
    check: str
    detail: str
    line: str | None = None
    semantic: str | None = None      # phase 5A: verdict / reason from the optional semantic review


def guard_mode() -> str:
    return env_str("VIOLET_PERSONA_REPLY_GUARD", "enforce").lower()     # enforce | flag | off


def _owner_call() -> str:
    """What she calls the owner in the relationship of this channel ("" when nothing set)."""
    try:
        from ai.channels.context import current
        from ai.persona_templates import relationship
        return relationship(current().channel).get("call") or ""
    except Exception:                                     # noqa: BLE001
        return ""


class ConsistencyGuard:
    def __init__(self, db, core: PersonaCore | None = None, config: dict[str, Any] | None = None):
        self.db = db
        self.core = core or PersonaCore.load()
        self.config = config or life_config()
        # phase 8.9.4: replies to the user are checked for weather stated as now; a proactive weather task has
        # real data behind it (Performer), so guard_reply turns this on for entity="reply" only
        self.check_weather = False

    def _stale_weather(self, now: str) -> bool:
        """Weather came up in an EARLIER conversation and not in this one (the last few hours of the user's own
        messages): a weather line now can only be the old weather retold as current. With no weather in the
        history at all, a passing "今天降温了" is left alone (phase 5A: harmless chat)."""
        from datetime import timedelta
        from ai.settings import env_float
        current = parse_time(now)
        since = current - timedelta(hours=env_float("VIOLET_CONTEXT_SESSION_GAP_HOURS", 3.0))
        from ai import context_cut                        # 翻篇: the weather talk before it is not "this chat's" either
        rows = [r for r in context_cut.owner_rows(self.db.fetch_all("SELECT role, content, created_at FROM messages ORDER BY id DESC LIMIT 200"))
                if r.get("created_at") and parse_time(r["created_at"]) <= current and WEATHER_WORDS_RE.search(r["content"] or "")]
        if any(r["role"] == "user" and parse_time(r["created_at"]) >= since for r in rows):
            return False                                  # the user brought the weather up in this conversation
        return any(parse_time(r["created_at"]) < since for r in rows)

    # ------------------------------------------------------------------
    # Persona patterns
    # ------------------------------------------------------------------
    def _persona_violations(self, text: str) -> list[Violation]:
        found = []
        call = _owner_call()
        for pattern in self.core.forbidden_backstory():
            match = pattern.search(text)
            # 2026-10-30: with the relationship "女儿" she calls the owner 爸爸 -- "我爸爸" is the owner, not a backstory
            kin = re.sub(r"^我(们家|们|的|和|跟|家)?", "", match.group(0)) if match else ""
            if match and not (call and kin and kin in call and call in text):
                found.append(Violation("forbidden_backstory", f"matches {pattern.pattern}"))
        if not found and any("上班" in p.pattern for p in self.core.forbidden_backstory()):
            work_match = _WORK_BACKSTORY_RE.search(text)
            if work_match:
                found.append(Violation("forbidden_backstory", f"matches {_WORK_BACKSTORY_RE.pattern}"))
        if not found and not self.core.has_school:         # 2026-11-12: a persona that goes to school has classmates
            claim = owned_role_claim(text)                 # phase 9.4 real run: "我们英语老师要是也这样就好了"
            if claim:
                found.append(Violation("forbidden_backstory", f"owns a school / work role: {claim}"))
        # 2026-11-12 (user: 承不承认 AI 都没什么关系，倒希望不是很承认): to the owner she may dodge the question in
        # character; the one thing she never says is that she is a real person. One rule for every persona (it was a
        # per-template list that also forbade "我不是AI")
        match = CLAIMS_HUMAN_RE.search(text)
        if match:
            found.append(Violation("identity_conflict", f"claims to be human: {match.group(0)}"))
        return found

    def _user_experiences(self) -> list[str]:
        rows = self.db.fetch_all("SELECT content FROM event_memories WHERE status != 'deleted' AND kind IN ('event', 'legacy')")
        facts = self.db.fetch_all("SELECT value FROM user_profile_facts WHERE status = 'active'")
        return [r["content"] for r in rows] + [f["value"] for f in facts]

    def _backstory_lines(self) -> list[str]:
        """2026-11-18: shared past lines from a persona template's 'backstory' back her claims about those memories."""
        raw = [str(x) for x in (self.core.data.get("backstory") or []) if x]
        if not raw:
            return []
        call = _owner_call() or "对方"
        out: list[str] = []
        for item in raw:
            filled = item.replace("{call}", call)
            out.append(filled)
            if "对方" in filled:
                out.append(filled.replace("对方", "你"))
        return out

    # ------------------------------------------------------------------
    # 1. Life event proposals
    # ------------------------------------------------------------------
    def check_event(self, proposal: dict[str, Any], now: str) -> list[Violation]:
        v: list[Violation] = []
        content = (proposal.get("content") or "").strip()
        now_dt = parse_time(now)
        occurred = parse_time(proposal.get("occurred_at"))
        duration = timedelta(minutes=int(proposal.get("duration_minutes") or 60))
        source = str(proposal.get("source") or "")
        origin = str((proposal.get("source_detail") or {}).get("origin") or "")

        info_kind = str((proposal.get("source_detail") or {}).get("info_kind") or "")
        if origin in EXTERNAL_ORIGINS or source.split(":")[0] in EXTERNAL_ORIGINS or info_kind == "public_fact":
            v.append(Violation("external_as_experience", f"source {source or origin} is external material, not Violet's life"))
        if info_kind == "user_real":
            v.append(Violation("user_experience", "a user's experience is never a life event"))
        if proposal.get("event_type") == "reading_record" or origin == "reading":
            # only "Violet read an article" is hers; what the article says is not (ai/life_interfaces.py)
            from ai.life_interfaces import check_reading_record
            v += [Violation(name, "a reading record may only describe the act of reading, with its source url")
                  for name in check_reading_record(proposal)]
        if not content or occurred is None:
            v.append(Violation("invalid", "content and occurred_at are required"))
            return v
        if occurred > now_dt:
            v.append(Violation("future_event", f"{proposal['occurred_at']} is after now"))

        thread = self.db.fetch_one("SELECT * FROM life_threads WHERE id = ?", (proposal.get("thread_id"),)) \
            if proposal.get("thread_id") else None
        if proposal.get("thread_id") and (not thread or thread["status"] != "active"):
            v.append(Violation("thread_not_active", f"thread {proposal.get('thread_id')} is not active"))
        if thread:
            if occurred < parse_time(thread["started_at"]):
                v.append(Violation("before_thread_start", "event is earlier than its thread"))
            last = self.db.fetch_one("SELECT occurred_at FROM life_events WHERE thread_id = ? AND status = 'happened' "
                                     "ORDER BY occurred_at DESC LIMIT 1", (thread["id"],))
            if last and occurred < parse_time(last["occurred_at"]):
                v.append(Violation("out_of_order", "event is earlier than the thread's latest event"))
            window = timedelta(days=float(self.config.get("duplicate_window_days", 14)))
            for other in self.db.fetch_all("SELECT content, occurred_at FROM life_events WHERE thread_id = ? AND status = 'happened'",
                                           (thread["id"],)):
                if other["content"] == content and abs(parse_time(other["occurred_at"]) - occurred) < window:
                    v.append(Violation("duplicate", "the same thing was just recorded"))
                    break

        for other in self.db.fetch_all("SELECT id, occurred_at, duration_minutes FROM life_events WHERE status = 'happened'"):
            start = parse_time(other["occurred_at"])
            end = start + timedelta(minutes=int(other["duration_minutes"] or 60))
            if occurred < end and start < occurred + duration:
                v.append(Violation("time_conflict", f"overlaps event {other['id']}"))
                break

        plan_id = proposal.get("plan_id")
        if proposal.get("completes_plan"):
            plan = self.db.fetch_one("SELECT * FROM life_plans WHERE id = ?", (plan_id,)) if plan_id else None
            if not plan or plan["status"] not in ("planned", "active") or (thread and plan["thread_id"] != thread["id"]):
                v.append(Violation("plan_not_open", f"plan {plan_id} cannot be completed by this event"))
        elif COMPLETION_RE.search(content) and proposal.get("event_type") != "thread_completed":
            v.append(Violation("completion_without_plan", "says something was finished without resolving a plan"))

        # event text has no subject ("和男朋友一起读…"): it is Violet's own act, so check it as "我…" too
        seen = set()
        for item in self._persona_violations(content) + self._persona_violations("我" + content):
            if item.check not in seen:
                seen.add(item.check)
                v.append(item)
        if DRAMATIC_RE.search(content):
            v.append(Violation("too_dramatic", "ordinary life only"))
        for experience in self._user_experiences():
            if touches_topic(content, experience):
                v.append(Violation("user_experience", "looks like the user's experience, not Violet's"))
                break
        return v

    # ------------------------------------------------------------------
    # 2. Generated replies
    # ------------------------------------------------------------------
    turn_used_web = False          # phase 9.3: set by guard_reply when this turn's Chat run used the web tools
    turn_used_schedule = False     # phase 9.4.2: ... or the timetable tools

    def _today_assistant_messages(self, now: str) -> list[str]:
        """Violet's own assistant messages from today (since midnight & context cut, up to now)."""
        from ai import context_cut
        day = str(now)[:10]
        now_dt = parse_time(now)
        rows = context_cut.owner_rows(self.db.fetch_all(
            "SELECT role, content, created_at FROM messages "
            "WHERE role != 'user' AND substr(created_at, 1, 10) = ? ORDER BY id ASC",
            (day,),
        ))
        out: list[str] = []
        for r in rows:
            at = parse_time(r.get("created_at")) if r.get("created_at") else None
            if at is not None and now_dt is not None and at > now_dt:
                continue
            if r.get("content"):
                out.append(str(r["content"]))
        return out

    def check_reply(self, text: str, now: str | None = None) -> list[Violation]:
        now = now or self.db.now()
        recent = self.db.fetch_all(
            "SELECT content, occurred_at, thread_id, source, source_detail "
            "FROM life_events WHERE status = 'happened' AND is_virtual = 1"
        )
        recent = [e for e in recent if parse_time(e["occurred_at"]) <= parse_time(now)]
        # phase 8.9: "把几首钢琴曲从头听了一遍" is backed by its configured details (德彪西的《月光》…), so naming
        # one of those pieces is not an unrecorded experience
        backs = lambda text, content: mentions_life_strict(text, content, self.config)
        open_plans = self.db.fetch_all("SELECT content FROM life_plans WHERE status IN ('planned', 'active', 'paused')")
        from ai.daily_life import SOURCE as DAILY_SOURCE, daily_doing_for, doing_matches, life_terms
        reading_terms = [t for r in self.db.fetch_all("SELECT title FROM life_threads WHERE status = 'active'")
                         for t in life_terms(r["title"], self.config)]
        backstory = self._backstory_lines()
        experiences = self._user_experiences()
        prior_today_facts: set[str] = set()
        if not getattr(self.core, "is_ai", False):
            for prev in self._today_assistant_messages(now):
                prior_today_facts |= _extract_today_facts(prev)

        def _daily_doing(event: dict[str, Any]) -> str:
            raw = event.get("source_detail")
            if isinstance(raw, dict) and raw.get("doing"):
                return str(raw["doing"])
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict) and parsed.get("doing"):
                        return str(parsed["doing"])
                except ValueError:
                    pass
            return daily_doing_for(event.get("content") or "", self.config)
        # phase 5B: what web pages say is never Violet's own experience
        finding_titles = [r["title"] for r in self.db.fetch_all(
            "SELECT title FROM web_findings WHERE status = 'active' ORDER BY id DESC LIMIT 60")]
        violations: list[Violation] = []
        stale = None
        for line in [l.strip() for l in (text or "").splitlines() if l.strip()]:
            if leaked_draft(line):                       # phase 8.9.5: every entity -- never meant for the user
                violations.append(Violation("leaked_draft", "the model's drafting notes in the reply", line))
            if self.check_weather and STALE_WEATHER_RE.search(line):
                if stale is None:
                    stale = self._stale_weather(now)
                if stale:
                    violations.append(Violation("stale_weather", "earlier weather stated as now", line))
            for persona in self._persona_violations(line):
                persona.line = line
                violations.append(persona)
            if CAMERA_CLAIM_RE.search(line) and not CAMERA_NEGATED_RE.search(line) and not _PAT_GESTURE_RE.search(line):
                violations.append(Violation("camera_claim", "an AI-generated picture told as a photo", line))
            if promised_lookup(line, self.turn_used_web):
                violations.append(Violation("promised_later_lookup", "a lookup promised for later", line))
            if getattr(self.core, "is_ai", False):
                # 2026-10-30: the AI persona's experiences are its real records, not virtual life events; what it must
                # not claim (a body, a family, human days) is in its own forbidden patterns, checked above
                continue
            contradiction = _same_day_contradiction(line, prior_today_facts)
            if contradiction:
                violations.append(Violation("same_day_contradiction", contradiction, line))
            else:
                prior_today_facts |= _extract_today_facts(line)
            # hedges ("想", "下次") and references to the user ("你") only count inside the claiming clause:
            # "我昨天去了海边，下次带你去" is still a claim
            claims = _claims(line)
            if not claims:
                continue
            if self.turn_used_web and WEB_ACTION_RE.search(line) and not LIVED_RE.search(line):
                # phase 9.3: "刚查了一下 / 看了原文" on a turn where Violet really searched / read a page is that action,
                # not an unrecorded experience (what the page says is still not hers -- the notice tells Chat so)
                continue
            if self.turn_used_schedule and SCHEDULE_ACTION_RE.search(line) and not LIVED_RE.search(line):
                # phase 9.4.2: the same for looking at the user's timetable this turn
                continue
            as_done = EXPERIENCE_AS_DONE_RE.sub(r"\1了", line)      # records are written in the completed form
            claims_completion = bool(COMPLETION_RE.search(line))

            # a progress event on an ongoing narrative thread ("读了《边城》的一部分") does not back a claim of
            # finishing it; already-completed single-shot events (thread_id IS NULL, e.g. life_daily) do.
            def _event_backs(e: dict[str, Any]) -> bool:
                if e.get("thread_id") is not None and claims_completion and not COMPLETION_RE.search(e["content"]):
                    return False
                if backs(as_done, e["content"]) or (as_done != line and backs(line, e["content"])):
                    return True
                if e.get("source") == DAILY_SOURCE:
                    doing = _daily_doing(e)
                    if doing and (doing_matches(as_done, doing) or (as_done != line and doing_matches(line, doing))):
                        return True
                return False

            if any(_event_backs(e) for e in recent):
                continue
            if backstory and any(backs(as_done, b) for b in backstory):
                continue
            if STATE_CLAIM_RE.search(line) and _state_backed(line, recent, parse_time(now)):
                continue
            # phase 8.9.10: a book she is reading backs reading a part of it -- not finishing it, and not another book
            if not claims_completion and any(t in line for t in reading_terms):
                continue
            if any(mentions_topic(as_done, p["content"]) for p in open_plans):
                violations.append(Violation("plan_described_as_done", "a plan that has not happened", line))
            elif all(USER_REFERENCE_RE.search(c) for c in claims):     # the claimed part itself is about the user
                continue
            elif any(touches_topic(as_done, e) for e in experiences):
                violations.append(Violation("user_experience_as_own", "the user's experience told as Violet's", line))
            elif any(touches_topic(as_done, t) for t in finding_titles):
                violations.append(Violation("external_as_experience", "something read on the web told as lived", line))
            else:
                violations.append(Violation("unrecorded_experience", "no recorded event behind this claim", line))
        return violations

    @staticmethod
    def drop_lines(text: str, violations: list[Violation]) -> str:
        """Phase 8.9.13: a line kept right after a dropped one no longer opens with a word that leans on it -- the real
        ab-v2e reply became just "不过你说的余味，我好像又有点想要了" once its first line was dropped."""
        bad = {v.line for v in violations if v.line}
        kept, dropped_before = [], False
        for line in (text or "").splitlines():
            if not line.strip():
                continue
            if line.strip() in bad:
                dropped_before = True
                continue
            if dropped_before and not kept:
                line = _LEANING_START_RE.sub("", line.strip()) or line
            kept.append(line)
            dropped_before = False
        return "\n".join(kept)
