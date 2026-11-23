"""Phase 8.9: how close Violet and the user are, read from the message log -- never from a counter of days alone.

The view is recomputed for every reply from what actually happened (nothing is stored, so it can always be
re-derived and audited):

* chat days and the span between the first and the latest one;
* how often the USER opened a conversation (a message after a long quiet gap) -- reaching out is a better sign
  of closeness than message volume;
* warmth the user showed themselves ("想你", "抱抱", "找你", calling Violet cute ...) and playfulness ("哈哈哈", "逗");
* an explicit wish for distance ("别撒娇", "别这么黏", "保持距离", "肉麻") -- it wins until the user invites
  closeness again ("可以撒娇", "想你了"); a long silence also cools a close relationship back down.

The result is a stage (new / warming / close / distance) and the evidence behind it, rendered as a description
of the relationship plus what it allows. It is not a score, it does not choose lines and nothing is random:
whether Violet is clingy in a given message is still up to the moment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from ai.memory_model import parse_time

SESSION_GAP = timedelta(hours=3)
# Phase 8.9.1: only wording that is warm in itself. A bare "找你" / "跟你说" is mostly narration ("今天好忙 都没顾上
# 找你", "跟你说个事"), so seeking Violet out counts only in forms like "想找你" / "想跟你说说话".
# 2026-11-13 (virtual test 0.7.72: the first message "我叫小周，刚来找你聊聊" counted as warmth -> the next day was
# "warming" and she wrote "突然有点想你了"): "来找你聊聊" is how anyone starts a chat. Only WANTING to come is warm.
WARM_RE = re.compile(r"(想你|想我没|想我了|有没有想我|喜欢你|抱抱|亲亲|贴贴|么么|想(来|去)?找你|(多|常|只)(来)?找你|"
                     r"想(跟|和)你(说|聊)|你真好(?!笑)|你好可爱|你挺可爱|你说话挺可爱|可爱死|乖乖|晚安.{0,2}[宝贝]|宝贝|陪我)")
# a warm phrase right after a negation is not warmth: "没顾上找你", "不想你", "没人陪我", "不是喜欢你"
NEGATED_RE = re.compile(r"(没|不|别|顾不上|懒得)[^，。！？,.!?\s]{0,3}$")
PLAYFUL_RE = re.compile(r"(哈哈哈|嘿嘿|笑死|逗你|被我说中|就这[？?]|你是不是又)")
DISTANCE_RE = re.compile(r"(别(再|老|总)?撒娇|不要(再)?撒娇|别(这么|那么|太)?黏|不要(这么|那么|太)?黏|太黏了|"
                         r"别(再)?叫我(宝|亲爱)|不要(再)?叫我(宝|亲爱)|肉麻|保持(点|一点)?距离|我们(又)?不熟|"
                         r"别(太|这么)亲密|别装可爱|别卖萌|正常(点|一点)?说话)")
# Negations / questions / third-party mentions before a distance phrase ("我又没让你保持距离", "这台词好肉麻", "这样肉麻吗")
DISTANCE_NEGATED_RE = re.compile(
    r"(没|不是|才不|并不|不会|哪有|怎么会|干嘛|为什么|谁说|不用|不必|不要让我|没让你|没叫你|不觉得|不算|哪里|一点也不|也不)"
    r"[^，。！？,.!?\s]{0,4}$"
)
DISTANCE_THIRD_PARTY_RE = re.compile(r"(这句|这词|台词|歌词|剧情|电影|小说|情话|广告|文案|剧里|戏里|别人)[^，。！？,.!?\s]{0,6}$")
DISTANCE_QUESTION_SUFFIX_RE = re.compile(r"^\s*[吗嘛么？?]")
CLOSE_INVITE_RE = re.compile(
    r"(可以撒娇|多撒娇|撒个娇|你可以黏|黏着我|黏人点|粘着我|想你了|好想你|我也想你|可以叫我宝|"
    r"别这么冷|热乎点|热情点|多找我|多来找我|多主动找我|主动找我|抱抱|亲亲|贴贴|喜欢你)"
)

CLOSE_MIN_DAYS = 5
CLOSE_MIN_SPAN = timedelta(days=14)
CLOSE_MIN_INITIATIONS = 4
CLOSE_MIN_SHARED = 3
COOL_DOWN_AFTER = timedelta(days=21)

# 2026-11-23: a single accidental DISTANCE_RE match must not permanently freeze proactive outreach for weeks.
# After DISTANCE_DECAY_DAYS of continued multi-day chatting with no new distance request, it naturally lifts;
# the owner can also clear it immediately from the Control Center (stored in proactive_settings).
DISTANCE_CLEARED_KEY = "relationship_distance_cleared_at"
DISTANCE_DECAY_DAYS = 3
DISTANCE_DECAY_MIN_DAYS = 2
DISTANCE_DECAY_MIN_MSGS = 4

WARMTH_LABELS = {
    0: "初识克制",
    1: "慢慢升温",
    2: "熟络活跃",
    3: "亲密黏人",
}

# Stage-scaled proactive cadence: level 0 keeps the shipped conservative baseline (1.0x) so new personas
# stay polite and default limits remain low; higher stages warm up smoothly within daily_message_limit.
PROACTIVE_PACE: dict[int, dict[str, float | int | bool | None]] = {
    0: {"interval_scale": 1.00, "after_user_scale": 1.00, "delay_scale": 1.00,
        "miss_hours": None, "checkin_hours": None, "daily_min_items": 1, "allow_quiet_day": True, "max_daily_offers": 1},
    1: {"interval_scale": 0.67, "after_user_scale": 0.67, "delay_scale": 0.75,
        "miss_hours": 8.0, "checkin_hours": 4.5, "daily_min_items": 2, "allow_quiet_day": False, "max_daily_offers": 1},
    2: {"interval_scale": 0.50, "after_user_scale": 0.50, "delay_scale": 0.55,
        "miss_hours": 4.0, "checkin_hours": 2.5, "daily_min_items": 3, "allow_quiet_day": False, "max_daily_offers": 2},
    3: {"interval_scale": 0.40, "after_user_scale": 0.40, "delay_scale": 0.45,
        "miss_hours": 3.0, "checkin_hours": 2.0, "daily_min_items": 3, "allow_quiet_day": False, "max_daily_offers": 2},
}


@dataclass
class RelationshipView:
    stage: str                                    # new | warming | close | distance
    chat_days: int = 0
    span_days: int = 0
    initiations: int = 0
    warm: int = 0
    playful: int = 0
    distance_quote: str | None = None
    distance_at: str | None = None
    cooled: bool = False
    shared: int = 0                               # phase 8.9.3: things of their own the user told (memories)
    evidence: list[str] = field(default_factory=list)


def assess(db, now: datetime | None = None) -> RelationshipView:
    from ai import context_cut
    now = now or parse_time(db.now())
    # 2026-11-09: after 翻篇 how close you are is judged from the chat since then
    rows = [r for r in context_cut.owner_rows(db.fetch_all("SELECT role, content, created_at FROM messages ORDER BY created_at, id"))
            if r.get("created_at") and parse_time(r["created_at"]) <= now]
    user = [r for r in rows if r["role"] == "user"]
    if not user:
        return RelationshipView("new")
    days = sorted({r["created_at"][:10] for r in user})
    first = parse_time(user[0]["created_at"])
    times = [parse_time(r["created_at"]) for r in rows]
    initiations = sum(1 for i, r in enumerate(rows)
                      if r["role"] == "user" and (i == 0 or times[i] - times[i - 1] >= SESSION_GAP))
    # the quiet time right before the current conversation (now counts: the user may be coming back right now)
    silence = timedelta(0)
    marks = times + [now]
    for later, earlier in zip(reversed(marks[1:]), reversed(marks[:-1])):
        if later - earlier >= SESSION_GAP:
            silence = later - earlier
            break
    warm = sum(1 for r in user if is_warm(r["content"] or ""))
    playful = sum(1 for r in user if PLAYFUL_RE.search(r["content"] or ""))
    cleared_at = _distance_cleared_at(db)
    distance = invite = None
    for r in user:
        msg_time = parse_time(r["created_at"])
        text = r["content"] or ""
        if is_distance_request(text) and (cleared_at is None or msg_time > cleared_at):
            distance = r
        if is_close_invite(text):
            invite = r
    # the latest of the two wins: an invitation after "别撒娇" lifts it, "别撒娇" after an invitation stands
    if distance and invite and parse_time(invite["created_at"]) > parse_time(distance["created_at"]):
        distance = None
    # 2026-11-23: natural warm-up recovery after multi-day active chatting with no new distance request
    if distance:
        dist_at = parse_time(distance["created_at"])
        after = [r for r in user if parse_time(r["created_at"]) > dist_at]
        after_days = {r["created_at"][:10] for r in after}
        if (now - dist_at >= timedelta(days=DISTANCE_DECAY_DAYS)
                and len(after) >= DISTANCE_DECAY_MIN_MSGS
                and len(after_days) >= DISTANCE_DECAY_MIN_DAYS):
            distance = None
    view = RelationshipView("new", chat_days=len(days), span_days=(parse_time(user[-1]["created_at"]) - first).days,
                            initiations=initiations, warm=warm, playful=playful, shared=_shared_memories(db))
    if distance:
        view.stage, view.distance_quote, view.distance_at = "distance", (distance["content"] or "")[:30], distance["created_at"]
        return view
    # Phase 8.9.3: telling Violet about one's own life (what MemoryAgent kept) is closeness too, not only warm words
    close = (view.chat_days >= CLOSE_MIN_DAYS and now - first >= CLOSE_MIN_SPAN
             and initiations >= CLOSE_MIN_INITIATIONS and (warm >= 1 or playful >= 2 or view.shared >= CLOSE_MIN_SHARED))
    if close and silence >= COOL_DOWN_AFTER:
        close, view.cooled = False, True
    if close:
        view.stage = "close"
    elif view.chat_days >= 3 or (view.chat_days >= 2 and warm >= 1) or view.cooled:
        view.stage = "warming"
    view.evidence = _evidence(view)
    return view


def _distance_cleared_at(db) -> datetime | None:
    try:
        raw = db.get_proactive_setting(DISTANCE_CLEARED_KEY)
        return parse_time(raw) if raw else None
    except Exception:
        return None


def clear_distance(db) -> str:
    """Explicitly lift an active 'distance' lock (from Control Center): ignores hits at or before now."""
    stamp = db.now()
    db.set_proactive_setting(DISTANCE_CLEARED_KEY, stamp)
    return stamp


def is_distance_request(text: str) -> bool:
    """True when the user is actually asking Violet to keep distance (not negating, questioning, or quoting a drama)."""
    raw = text or ""
    for m in DISTANCE_RE.finditer(raw):
        prefix = raw[max(0, m.start() - 12):m.start()]
        suffix = raw[m.end():m.end() + 4]
        if DISTANCE_NEGATED_RE.search(prefix):
            continue
        if m.group(0) == "肉麻" and (DISTANCE_THIRD_PARTY_RE.search(prefix) or DISTANCE_QUESTION_SUFFIX_RE.search(suffix)):
            continue
        return True
    return False


def is_close_invite(text: str) -> bool:
    """User invites closeness / warmth (and not negated right before it)."""
    raw = text or ""
    return any(not NEGATED_RE.search(raw[:m.start()]) for m in CLOSE_INVITE_RE.finditer(raw))


def warmth_level(view: RelationshipView, persona_stage: dict | None = None, rel_kind: str | None = None) -> int:
    """0 (new/distant) -> 1 (warming) -> 2 (close) -> 3 (deep/clingy)."""
    if view.stage == "distance":
        return 0
    rel_level = {"new": 0, "warming": 1, "close": 2}.get(view.stage, 0)
    if view.stage == "close" and not view.cooled and (view.chat_days >= 14 or view.warm >= 3):
        rel_level = 3
    if rel_kind == "lover" and not view.cooled:
        rel_level = max(rel_level, 2)
    stage_level = 0
    if persona_stage and not view.cooled:
        idx = int(persona_stage.get("index", 0) or 0)
        total = int(persona_stage.get("total") or len(persona_stage.get("labels") or []) or 0)
        if idx >= 3 or (total >= 3 and idx >= total - 1):
            stage_level = 3
        elif idx >= 2:
            stage_level = 2
        elif idx >= 1:
            stage_level = 1
    return max(rel_level, stage_level)


def proactive_pace(level: int) -> dict[str, float | int | bool | None]:
    return dict(PROACTIVE_PACE[max(0, min(3, int(level or 0)))])


def scaled_minutes(base: float, scale: float, floor: float) -> float:
    """Scale a configured minute threshold down with warmth, never raising it above the user's own config."""
    if base <= 0 or scale >= 1.0:
        return float(base)
    return min(float(base), max(float(floor), round(float(base) * scale, 1)))


# Phase 8.9.3: why Violet might want to reach out on her own -- reasons, not a schedule. The scheduler still decides
# WHEN a proactive check may happen (quiet hours, caps, busy, cold streak); these only say what she would come for.
# 2026-11-23: stage-scaled progressive warmth (warming 8h, close 4h; level 3 deep stage 3h via PROACTIVE_PACE).
MISS_AFTER_HOURS = {"close": 4, "warming": 8}


def proactive_motives(view: RelationshipView, hours_since_user: float | None, cold_streak: int,
                      life_events: list[dict], open_plans: list[dict],
                      talked_plan_ids: set | None = None, unanswered: list[dict] | None = None,
                      afterglow: dict | None = None, persona_stage: dict | None = None,
                      rel_kind: str | None = None) -> list[dict]:
    """Violet's own reasons to get in touch now: something of hers to share, missing the user, a small decision she
    would like the user's say on, or (phase 8.9.5) a plan of hers the two of you talked about lately, to pick up
    again. Empty is normal (and with a wish for distance there is nothing to add)."""
    if view.stage == "distance":
        return []
    level = warmth_level(view, persona_stage, rel_kind=rel_kind)
    pace = proactive_pace(level)
    eff_stage = "close" if level >= 2 else ("warming" if level == 1 else view.stage)
    talked_plan_ids = talked_plan_ids or set()
    unanswered = unanswered or []
    # 2026-11-23: if the user DID reply (unanswered is empty), a single brief reply (cold_streak == 1) should not
    # permanently suppress missing / warm check-in motives.
    can_warm_reach = (cold_streak == 0) or (cold_streak == 1 and not unanswered)
    motives = []
    # phase 8.9.11 (real ab-v2c): "忙着想你呢" at 20:30, and at 22:03 her first message was the bookstore -- the only
    # reason offered. What the two of you were just feeling is a reason of its own (once: before she writes again).
    if afterglow and eff_stage in ("warming", "close") and not view.cooled:
        motives.append({"type": "afterglow", "at": afterglow.get("at"), "note": (
            f"你们刚才（{afterglow.get('at')}）聊得正亲近，对方说了“{afterglow.get('quote')}”。可以接着那份情绪来找对方："
            "说说你后来还惦记着那句话、害羞或开心的心情，或者顺着再逗对方一下；不用硬换成别的话题，也不用提问。"
            if eff_stage == "close" else
            f"你们刚才（{afterglow.get('at')}）聊得挺开心，对方说了“{afterglow.get('quote')}”。可以接着那会儿的心情说一句，"
            "不用硬换话题，也不用提问。")})
    daily_events = [e for e in (life_events or []) if e.get("type") == "violet_daily_event"]
    thread_events = [e for e in (life_events or []) if e.get("type") != "violet_daily_event"]
    if thread_events:
        motives.append({"type": "share", "note": "有件自己的小事想告诉对方（见 allowed_life_events）"})
    if daily_events:
        ev = daily_events[-1]
        motives.append({"type": "daily_share", "event_id": ev.get("id"), "about": ev.get("content"),
                        "note": "这是你今天刚发生的一件日常小事（还没告诉过对方），很适合按你的性格随口分享一句、吐槽一下，或借着聊两句（用了把它的 id 写进 mentioned_life_event_ids）"})
    miss_after = pace.get("miss_hours") or MISS_AFTER_HOURS.get(eff_stage)
    if miss_after and not view.cooled and can_warm_reach and hours_since_user is not None and hours_since_user >= miss_after:
        hours = max(1, int(hours_since_user))
        # phase 8.9.5: every real "miss" message became "想你了 + 今天过得怎么样？" -- a line of her own is enough
        motives.append({"type": "miss", "hours_since_user": hours, "note": (
            f"已经 {hours} 小时没聊了，很想找对方说说话：可以直白说一句想对方了、撒个娇讨点关注、轻轻蹭一下存在感，或者说一件突然想到对方的小事；"
            "不一定要问对方近况；不埋怨对方没来找她，也不催对方回" if level >= 3 else
            f"已经 {hours} 小时没聊了，有点想对方：可以只说一句想对方了、撒个娇讨点关注，或者说一件突然想到对方的小事；"
            "不一定要问对方近况；不埋怨对方没来找她，也不催对方回" if eff_stage == "close" else
            "有一阵子没聊了：可以轻轻说一句有点想对方，或者拿一件小事当理由来找对方说话；不埋怨、不催")})
    # 2026-11-23: when warmed up (level >= 1) and no concrete share/afterglow/miss is queued yet, give her a natural
    # stage-grounded reason to drop a short warm line after checkin_hours so she doesn't starve with motives=[].
    checkin_after = pace.get("checkin_hours")
    if (level >= 1 and checkin_after and not view.cooled and not unanswered and can_warm_reach
            and hours_since_user is not None and hours_since_user >= checkin_after
            and not any(m["type"] in ("daily_share", "share", "afterglow", "miss") for m in motives)):
        note = (
            "你们已经很亲近了，也有一阵子没说话了：可以很自然地主动来蹭个存在感——随口分享一个刚冒出来的小念头、轻轻撒个娇、逗对方一句，或找个顺手的小借口搭话；贴合你的人设和当前阶段，简短自然，别用“吃了吗/在干嘛”这种模板问候。"
            if level >= 3 else
            "你们已经挺熟了，有两三个小时没聊了：可以按你的性格主动来搭一句——随口说个刚想到的小念头、轻轻逗对方一下，或找个贴合你们关系的小理由说一句；简短自然，不用客套寒暄，也别问“吃了吗/在干嘛”。"
            if level == 2 else
            "你们正在慢慢熟起来，有几个小时没说话了：可以轻轻找个自然的小话题搭一句（贴合你的人设，不要客套寒暄，也不要硬问近况）。"
        )
        motives.append({"type": "warm_checkin", "warmth_level": level, "note": note})
    # phase 8.9.10 (real ab-v2b): close, she wrote once on her own and nobody answered for hours -- missing the user,
    # a little hurt and wanting attention is a reason of its own. Once: after two unanswered messages the scheduler
    # waits (max_cold_streak), and busy / "don't write" / quiet hours are decided before any motive is looked at.
    # phase 9.1.1: the exact time since her message decides, never the rounded "hours" (5.95 -> 6.0 passed the gate)
    waited = float(unanswered[0].get("elapsed_hours", unanswered[0].get("hours", 0))) if unanswered else 0.0
    unanswered_gate = float(pace.get("miss_hours") or MISS_AFTER_HOURS["close"])
    if (eff_stage == "close" and not view.cooled and len(unanswered) == 1
            and waited >= unanswered_gate):
        last = {**unanswered[0], "hours": waited}
        motives.append({"type": "miss_unanswered", "hours_since_her_message": int(last["hours"]), "note": (
            # phase 8.9.11 (real ab-v2c): "像轻轻戳一下对方" came back word for word in the brief, and with every kind of
            # complaint ruled out the message had no hurt in it at all ("等你忙完，记得来理我一下"). Playful sulking is
            # allowed; real reproach is not. The wording describes what she may feel -- nothing to copy.
            f"你 {last.get('at', '')} 主动发的那条，对方过了 {int(last['hours'])} 小时还没回。已经很亲近了，可以带着小情绪再找"
            "对方一次：想对方、有点委屈、可怜巴巴，或者玩笑式地闹个小别扭（假装生气、嘟囔自己被晾着了、要对方哄一哄），"
            "这些都是撒娇，可以说出来；只找这一次。不行的是认真地指责、翻旧账、让对方愧疚、连着追问，或者上升成感情问题。"
            "别把上一条的内容换个说法再说一遍")})
    if eff_stage in ("warming", "close"):
        talked = [p for p in open_plans if p.get("id") in talked_plan_ids]
        fresh = [p for p in open_plans if p.get("id") not in talked_plan_ids]
        if talked:
            motives.append({"type": "continue", "plan_id": talked[0].get("id"), "about": talked[0].get("content"),
                            "note": "最近跟对方聊到过这件事，可以接着说说你的新想法或小期待（还没做的不要说成做了）"})
        if fresh:
            motives.append({"type": "ask_opinion", "plan_id": fresh[0].get("id"), "about": fresh[0].get("content"),
                            "note": "可以跟对方聊聊这件打算，问问对方的看法，或者请对方帮她出个主意（比如先做哪件、怎么安排）"})
    return motives


def _shared_memories(db) -> int:
    """How many things of their own the user has told Violet that are still remembered (active events and facts)."""
    try:
        row = db.fetch_one("SELECT COUNT(*) AS n FROM event_memories WHERE status = 'active' AND kind IN ('event', 'fact')")
        return int(row["n"]) if row else 0
    except Exception:
        return 0


def is_warm(text: str) -> bool:
    """The user says something warm to Violet: a warm phrase that is not negated right before it."""
    return any(not NEGATED_RE.search(text[:m.start()]) for m in WARM_RE.finditer(text or ""))


def getting_to_know(view: RelationshipView) -> bool:
    """Phase 8.9.1: basic get-to-know questions ("上学还是工作") belong to the early days. Not once you are close,
    not when the user asked for distance, and not after weeks of talking on many days."""
    if view.stage in ("close", "distance"):
        return False
    return not (view.chat_days >= CLOSE_MIN_DAYS and view.span_days >= CLOSE_MIN_SPAN.days)


def _evidence(view: RelationshipView) -> list[str]:
    out = [f"聊过 {view.chat_days} 天"]
    if view.initiations >= CLOSE_MIN_INITIATIONS:
        out.append("对方常主动来找你")
    if view.warm:
        out.append("对方对你说过亲近的话")
    if view.playful >= 2:
        out.append("对方常跟你开玩笑")
    if view.shared >= CLOSE_MIN_SHARED:
        out.append("对方跟你说过不少自己的事")
    return out


# Phase 8.9.3: Violet grows from shy to clingy as the relationship does. Each stage says what is natural now -- never
# a quota, never a line to say; the last sentence of "close" is the boundary that does not move.
def describe(view: RelationshipView, persona_stage: dict | None = None) -> str:
    """One line for Violet's context: what the relationship is and what it allows -- not what to say."""
    if view.stage == "distance":
        when = (view.distance_at or "")[5:16].replace("T", " ")
        return (f"- 对方 {when} 说过“{view.distance_quote}”：保持轻松友好的距离，不撒娇、不用亲昵的称呼；"
                "对方自己放开了再说。")
    eff_stage = view.stage
    if persona_stage and not view.cooled:
        idx = int(persona_stage.get("index", 0) or 0)
        if idx >= 2:
            eff_stage = "close"
        elif idx >= 1 and eff_stage == "new":
            eff_stage = "warming"
    if eff_stage == "close":
        # phase 8.9.8: "日常小事请对方帮你拿主意" is said once, in the persona; repeated here it made the real free chat's
        # Violet undecided four times in one evening (local/free_chat/cys-close-1)
        ev = f"（{'，'.join(view.evidence)}）" if view.evidence else ""
        return (f"- 已经很亲近了{ev}：可以明显黏人一点——主动讨点关注、分享小心事、"
                "想念就说出来、自己有主意就直接说、聊得开心时舍不得结束，偶尔害羞、俏皮或闹点小脾气也自然；"
                "看当下的气氛来，不是每句都要有。"
                "想念不等于要求：不让对方愧疚，不要求对方只陪你。")
    if eff_stage == "warming":
        if view.cooled:
            return "- 之前挺熟，但有好一阵子没聊了：先自然地接上，亲近慢慢找回来，不埋怨对方。"
        return ("- 慢慢熟起来了：可以主动找话题、开开玩笑，偶尔撒个娇、说一句有点想对方；"
                "还不用太黏，跟着对方的回应慢慢来。")
    # 2026-11-12 (virtual test: "晚上忽然有点想你了" on the second day; user: 第一次聊天像刚认识，慢慢熟)
    return ("- 还在互相熟悉：温柔、有点害羞就好，可以多聊聊自己喜欢的东西；还不说想念、不吃醋、不撒娇，"
            "主动找对方时也只说具体的事，这些等熟一点再说。")
