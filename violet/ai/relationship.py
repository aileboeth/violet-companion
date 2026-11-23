"""Unified relationship state and stage-scaled proactive rhythm (reworked v0.7.78).

Single source of truth -- no sentiment/distance regexes guessing human tone:
* Closeness comes from the active persona's stage when it has stages, otherwise from chat days since the last
  context cut (0-2d new, 3-7d warming, 8d+ close; long 21-day silence cools back to warming).
* An explicit wish for distance ("closeness=distance") is recorded by MemoryAgent with full conversational
  context (never keyword regexes) and can always be cleared in one click from the Control Center.
* Proactive cadence (intervals, random delays, daily-life supply) warms smoothly with warmth_level (0..3)
  while keeping level 0 at the conservative shipped defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from ai.memory_model import parse_time

SESSION_GAP = timedelta(hours=3)
WARMING_MIN_DAYS = 3
CLOSE_MIN_DAYS = 8
DEEP_MIN_DAYS = 14
COOL_DOWN_AFTER = timedelta(days=21)

WARMTH_LABELS = {
    0: "初识克制",
    1: "慢慢升温",
    2: "熟络活跃",
    3: "亲密黏人",
}

PROACTIVE_PACE: dict[int, dict[str, float | int | bool]] = {
    0: {"interval_scale": 1.00, "after_user_scale": 1.00, "delay_scale": 1.00,
        "daily_min_items": 1, "allow_quiet_day": True, "max_daily_offers": 1},
    1: {"interval_scale": 0.67, "after_user_scale": 0.67, "delay_scale": 0.75,
        "daily_min_items": 2, "allow_quiet_day": False, "max_daily_offers": 1},
    2: {"interval_scale": 0.50, "after_user_scale": 0.50, "delay_scale": 0.55,
        "daily_min_items": 3, "allow_quiet_day": False, "max_daily_offers": 2},
    3: {"interval_scale": 0.40, "after_user_scale": 0.40, "delay_scale": 0.45,
        "daily_min_items": 3, "allow_quiet_day": False, "max_daily_offers": 2},
}


@dataclass
class RelationshipView:
    stage: str                                    # new | warming | close | distance
    chat_days: int = 0
    span_days: int = 0
    initiations: int = 0
    distance_quote: str | None = None
    distance_at: str | None = None
    cooled: bool = False
    evidence: list[str] = field(default_factory=list)


def _distance_preference(db) -> dict | None:
    """Explicit closeness=distance recorded by MemoryAgent (or dashboard), or None."""
    try:
        from ai.profile_store import Profile
        pref = (Profile(db).preferences.effective().get("closeness") or {})
        if pref.get("origin") == "explicit" and pref.get("value") == "distance":
            row = db.fetch_one(
                "SELECT evidence, updated_at, created_at FROM interaction_preferences WHERE id = ?",
                (pref.get("id"),),
            ) if pref.get("id") else None
            return {**pref, **(row or {})}
    except Exception:                                        # noqa: BLE001
        pass
    return None


def clear_distance(db) -> str:
    """Lift an active explicit closeness=distance preference (from Control Center)."""
    from ai.profile_store import Profile
    Profile(db).preferences.clear_explicit("closeness", value="distance", actor="dashboard")
    return db.now()


def assess(db, now: datetime | None = None) -> RelationshipView:
    from ai import context_cut
    now = now or parse_time(db.now())
    rows = [r for r in context_cut.owner_rows(db.fetch_all("SELECT role, created_at FROM messages ORDER BY created_at, id"))
            if r.get("created_at") and parse_time(r["created_at"]) <= now]
    user = [r for r in rows if r["role"] == "user"]
    chat_days = len({r["created_at"][:10] for r in user})
    first = parse_time(user[0]["created_at"]) if user else now
    span_days = (parse_time(user[-1]["created_at"]) - first).days if user else 0
    times = [parse_time(r["created_at"]) for r in rows]
    initiations = sum(1 for i, r in enumerate(rows)
                      if r["role"] == "user" and (i == 0 or times[i] - times[i - 1] >= SESSION_GAP))

    dist = _distance_preference(db)
    if dist is not None:
        return RelationshipView(
            stage="distance", chat_days=chat_days, span_days=span_days, initiations=initiations,
            distance_quote=(dist.get("evidence") or "")[:30] or None,
            distance_at=dist.get("updated_at") or dist.get("created_at"),
        )
    if not user:
        return RelationshipView("new")

    silence = timedelta(0)
    marks = times + [now]
    for later, earlier in zip(reversed(marks[1:]), reversed(marks[:-1])):
        if later - earlier >= SESSION_GAP:
            silence = later - earlier
            break

    view = RelationshipView("new", chat_days=chat_days, span_days=span_days, initiations=initiations)
    if chat_days >= CLOSE_MIN_DAYS:
        if silence >= COOL_DOWN_AFTER:
            view.stage, view.cooled = "warming", True
        else:
            view.stage = "close"
    elif chat_days >= WARMING_MIN_DAYS:
        view.stage = "warming"
    view.evidence = [f"聊过 {view.chat_days} 天"]
    return view


def warmth_level(view: RelationshipView, persona_stage: dict | None = None, rel_kind: str | None = None) -> int:
    """Single warmth ladder 0..3: persona stage wins when present; explicit distance always drops to 0."""
    if view.stage == "distance":
        return 0
    if persona_stage:
        idx = int(persona_stage.get("index", 0) or 0)
        total = int(persona_stage.get("total") or len(persona_stage.get("labels") or []) or 0)
        if idx >= 3 or (total >= 3 and idx >= total - 1):
            lvl = 3
        elif idx >= 2:
            lvl = 2
        elif idx >= 1:
            lvl = 1
        else:
            lvl = 0
        return min(lvl, 1) if view.cooled else lvl
    lvl = {"new": 0, "warming": 1, "close": 2}.get(view.stage, 0)
    if view.stage == "close" and not view.cooled and view.chat_days >= DEEP_MIN_DAYS:
        lvl = 3
    if rel_kind in ("partner", "lover") and not view.cooled:
        lvl = max(lvl, 2)
    return lvl


def proactive_pace(level: int) -> dict[str, float | int | bool]:
    return dict(PROACTIVE_PACE[max(0, min(3, int(level or 0)))])


def scaled_minutes(base: float, scale: float, floor: float) -> float:
    """Scale a configured minute threshold down with warmth, never raising it above the user's own config."""
    if base <= 0 or scale >= 1.0:
        return float(base)
    return min(float(base), max(float(floor), round(float(base) * scale, 1)))


def getting_to_know(view: RelationshipView) -> bool:
    """Basic onboarding questions belong only to the early days."""
    return view.stage not in ("close", "distance") and view.chat_days < CLOSE_MIN_DAYS


def describe(view: RelationshipView, persona_stage: dict | None = None) -> str:
    """One line for Violet's context: how close the relationship is and what feels natural now."""
    if view.stage == "distance":
        when = (view.distance_at or "")[5:16].replace("T", " ")
        quote = f"（“{view.distance_quote}”）" if view.distance_quote else ""
        return (f"- 对方{'在 ' + when + ' ' if when else ''}明确要求保持轻松友好的距离{quote}："
                "不撒娇、不用亲昵的称呼，对方自己放开了再说。")
    lvl = warmth_level(view, persona_stage)
    if lvl >= 2:
        ev = f"（{'，'.join(view.evidence)}）" if view.evidence else ""
        return (f"- 已经很亲近了{ev}：可以明显黏人一点——主动讨点关注、分享小心事、"
                "想念就说出来、自己有主意就直接说、聊得开心时舍不得结束，偶尔害羞、俏皮或闹点小脾气也自然；"
                "看当下的气氛来，不是每句都要有。想念不等于要求：不让对方愧疚，不要求对方只陪你。")
    if lvl == 1:
        if view.cooled:
            return "- 之前挺熟，但有好一阵子没聊了：先自然地接上，亲近慢慢找回来，不埋怨对方。"
        return ("- 慢慢熟起来了：可以主动找话题、开开玩笑，偶尔撒个娇、说一句有点想对方；"
                "还不用太黏，跟着对方的回应慢慢来。")
    return ("- 还在互相熟悉：温柔、有点害羞就好，可以多聊聊自己喜欢的东西；还不说想念、不吃醋、不撒娇，"
            "主动找对方时也只说具体的事，这些等熟一点再说。")
