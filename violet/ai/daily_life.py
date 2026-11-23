"""Phase 8.9: Violet's ordinary day -- small, mundane things beside the long life threads (ai/life_engine.py).

Without it Violet has nothing recorded on most days, so "你今天干嘛了" is answered from the persona summary
("听了会儿音乐，也翻了几页书") -- an unrecorded experience the reply guard cannot see (no subject, no "我").

* No model: items come from persona/life_config.json "daily" and a seeded RNG (salt + date), so a day is the
  same every time it is looked at. 0-3 items per day, in fixed time slots; a quiet day is allowed.
* An item becomes a recorded event (source life_daily) only once its start time has passed, through the same
  ConsistencyGuard.check_event as every life event (time conflicts, the user's own experiences, persona rules);
  a rejected item is simply not part of the day. Items still ahead are plans of today, never stored as done.
* Lazily materialised when Violet's context is built (today and the last ``backfill_days``), oldest day first,
  so what she says about yesterday stays the same tomorrow. Idempotent by uid ``daily:<date>:<slot>``.
* Not counted in the life engine's weekly budget, never offered as a proactive share (too small to send).
"""
from __future__ import annotations

import json
import logging
import random
import re
from datetime import date, datetime, timedelta
from typing import Any

from ai.consistency import ConsistencyGuard
from ai.life_store import LifeStore
from ai.memory_model import _bigrams, mentions_topic, parse_time
from ai.persona import life_config
from ai.settings import env_bool

logger = logging.getLogger("violet.life.daily")

SOURCE = "life_daily"
_TITLE_RE = re.compile(r"《([^》]{1,30})》")
_QUANTIFIER_RE = re.compile(r"^(几首|几条|几本|一张|一位|一些|一首|一本)")
_WEEKDAY_CLASS_RE = re.compile(r"(早课|高数|早八|专业课|体育课|上课|上完课|上一节|上了一|教学楼|教室|早读)")
_SLOT_FALLBACK_LABELS = {"morning": "上午", "afternoon": "下午", "evening": "晚上"}
_DOING_PREFIX_RE = re.compile(r"^(?:刚|刚刚)?(?:正)?在")

_DAILY_FILLER_BIGRAMS = frozenset({
    "了会", "会儿", "一会", "了一", "一下", "一点", "有点", "一遍", "了个", "一个", "一些", "了些",
    "了下", "下了", "着呢", "了吧", "了呀", "的时", "时候", "一直", "然后", "就是", "还是", "了好",
    "上的", "里的", "下的", "边的", "前的", "后的", "天的", "东西", "一部", "试着", "顺手", "刚才",
    "刚刚", "已经", "今天", "明天", "昨天", "早上", "上午", "中午", "下午", "晚上", "现在",
    "几张", "几本", "几天", "几行", "几支", "一双", "一页", "一份", "一卷", "一杯", "一碗", "一锅",
    "一盘", "一盒", "一圈", "一起", "不是", "没有", "怎么", "什么", "这个", "那个", "自己", "结果",
    "居然", "最后", "差点", "不小", "小心", "理了", "拾了", "买了", "吃了", "看了", "听了",
})
_AMBIGUOUS_SINGLE_BIGRAMS = frozenset({
    "收拾", "整理", "厨房", "阳台", "楼下", "房间", "宿舍", "学校", "教室", "家里", "回家",
    "附近", "出门", "手机", "东西", "发呆", "早饭", "晚饭", "衣服", "作业", "笔记", "上课",
    "看书", "听歌", "电影", "超市", "书桌", "桌上", "窗边", "公园", "食堂", "买东", "买早",
    "吃早", "吃晚", "洗衣", "叠衣", "散步",
})
_DAILY_STOP_CHARS = frozenset(
    "的了是在我你他她它们这那有个不也都很太就又还把被给让去来到上下里中大好小"
    "一二三四五六七八九十几么吗呢吧啊呀哦嗯哼哈着过得地和跟与或但而因为所以如果会要能可想说看听吃做买走坐站拿放打开关"
)
_DAILY_GLUE_CHARS = frozenset("的了在着个把给去回到来上下里一几")
_TEMPLATE_DOING_CACHE: dict[str, str] | None = None


def _match_tz(dt: datetime, tz) -> datetime:
    if tz is None:
        return dt.replace(tzinfo=None)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def _parse_source_detail(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
    return {}


def _iter_slot_items(slots: Any):
    if not isinstance(slots, dict):
        return
    for spec in slots.values():
        if isinstance(spec, dict):
            for entry in spec.get("items") or []:
                if isinstance(entry, dict) and entry.get("done"):
                    yield entry


def _template_doing_map() -> dict[str, str]:
    global _TEMPLATE_DOING_CACHE
    if _TEMPLATE_DOING_CACHE is not None:
        return _TEMPLATE_DOING_CACHE
    mapping: dict[str, str] = {}
    try:
        from ai.persona_templates import TEMPLATE_DIR
        for path in sorted(TEMPLATE_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            for entry in _iter_slot_items(data.get("daily")):
                if entry.get("doing"):
                    mapping.setdefault(str(entry["done"]), str(entry["doing"]))
    except Exception:                                        # noqa: BLE001
        pass
    _TEMPLATE_DOING_CACHE = mapping
    return mapping


def daily_doing_for(content: str, config: dict[str, Any] | None = None) -> str:
    """Look up the `doing` phrasing for a DailyLife `done` event across config, active persona, and shipped templates."""
    if not content:
        return ""
    cfg = config if config is not None else life_config()
    for entry in _iter_slot_items((cfg.get("daily") or {}).get("slots")):
        if entry.get("done") == content and entry.get("doing"):
            return str(entry["doing"])
    try:
        from ai.persona import PersonaCore
        for entry in _iter_slot_items(PersonaCore.load().data.get("daily")):
            if entry.get("done") == content and entry.get("doing"):
                return str(entry["doing"])
    except Exception:                                        # noqa: BLE001
        pass
    return _template_doing_map().get(content, "")


def _daily_content_bigrams(text: str) -> set[str]:
    return {
        g for g in _bigrams(text) - _DAILY_FILLER_BIGRAMS
        if not (len(g) == 2 and (g[0] in _DAILY_GLUE_CHARS or g[1] in _DAILY_GLUE_CHARS))
    }


def _daily_content_chars(text: str) -> set[str]:
    return {ch for ch in re.findall(r"[\u4e00-\u9fff]", text or "") if ch not in _DAILY_STOP_CHARS}


def doing_matches(text: str, doing: str) -> bool:
    """Match a reply against a DailyLife item's `doing` phrase (e.g. '在厨房看汤', '在削苹果', '开着小夜灯听慢歌')."""
    if not text or not doing:
        return False
    core = _DOING_PREFIX_RE.sub("", doing.strip())
    dg = _daily_content_bigrams(core)
    if not dg:
        return False
    hits = dg & _daily_content_bigrams(text)
    if not hits:
        return False
    specific_hits = hits - _AMBIGUOUS_SINGLE_BIGRAMS
    if not specific_hits:
        return False
    if len(dg) <= 3:
        if len(hits) >= 2 and (hits & _AMBIGUOUS_SINGLE_BIGRAMS):
            core_chars = set(re.findall(r"[\u4e00-\u9fff]", core))
            text_chars = set(re.findall(r"[\u4e00-\u9fff]", text))
            if core_chars and len(core_chars & text_chars) / len(core_chars) < 0.80:
                return False
        return True
    if len(hits) >= 2 and (len(dg) <= 5 or len(hits) / len(dg) >= 0.25):
        if len(hits) == 2 and (hits & _AMBIGUOUS_SINGLE_BIGRAMS):
            core_chars = set(re.findall(r"[\u4e00-\u9fff]", core))
            text_chars = set(re.findall(r"[\u4e00-\u9fff]", text))
            if core_chars and len(core_chars & text_chars) / len(core_chars) < 0.80:
                return False
        return True
    return False


def _paraphrase_matches(text: str, content: str) -> bool:
    """Catch natural Chinese paraphrases of long DailyLife items without false-matching unrelated chat."""
    cg = _daily_content_bigrams(content)
    if not cg:
        return False
    hits = cg & _daily_content_bigrams(text)
    if not hits:
        return False
    specific_hits = hits - _AMBIGUOUS_SINGLE_BIGRAMS
    if len(hits) >= 3 and specific_hits:
        return True
    if len(hits) >= 2 and specific_hits and len(hits) / len(cg) >= 0.20:
        return True
    cc = _daily_content_chars(content)
    if cc:
        ch_hits = cc & _daily_content_chars(text)
        if len(ch_hits) >= 4 and len(ch_hits) / len(cc) >= 0.55:
            return True
    return False


def _install_seed() -> str:
    """A random string made once per install (so two installs do not live the same days)."""
    import secrets
    from ai.persona_templates import control_dir
    path = control_dir() / "life-seed"
    try:
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    except OSError:
        pass
    value = secrets.token_hex(8)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    except OSError:                                          # read-only: the same within this run
        logger.warning("life seed could not be saved")
    return value


class DailyLife:
    def __init__(self, db, config: dict[str, Any] | None = None):
        self.db = db
        self.config = config or life_config()
        self.daily = self.config.get("daily") or {}
        self.store = LifeStore(db)

    def enabled(self) -> bool:
        return bool(self.daily.get("enabled", False) and self.daily.get("slots")) and env_bool("VIOLET_DAILY_LIFE", True)

    # ------------------------------------------------------------------
    def _persona(self) -> tuple[str, dict[str, Any]]:
        """(her name, her own day's slots or the shipped ones)."""
        try:
            from ai.persona import PersonaCore
            data = PersonaCore.load().data
        except Exception:                                    # noqa: BLE001
            data = {}
        own = data.get("daily")
        name = str((data.get("identity") or {}).get("name") or "")
        return name, own if isinstance(own, dict) and own else self.daily["slots"]

    def _warmth_level(self) -> int:
        try:
            from ai import persona_templates as pt
            from ai.relationship import assess, warmth_level
            rel = pt.relationship(None)
            return warmth_level(assess(self.db), pt.stage(None, self.db), rel_kind=rel.get("kind"))
        except Exception:                                    # noqa: BLE001
            return 0

    def plan_for(self, day: date, tz) -> list[dict[str, Any]]:
        """The whole day, decided from the date (and what the days before it already had).

        2026-11-12 (virtual test: every persona of every install said "昨天在家做了蛋炒饭，有点咸" on its first day):
        the seed was the date alone and the slots one shared list -- the same day for everybody, in Violet's flat. Now a
        persona brings her own slots ("daily" in her template: the 继妹 cooks at home, the 学妹 sits in the club room),
        and the seed is this install's own (control/life-seed), her name and the date: still the same day every time
        it is looked at."""
        name, slots_spec = self._persona()
        rng = random.Random(f"{self.daily.get('seed_salt', 'eve-day')}:{_install_seed()}:{name}:{day.isoformat()}")
        from ai.relationship import proactive_pace
        pace = proactive_pace(self._warmth_level())
        quiet_roll = rng.random()
        if pace["allow_quiet_day"] and quiet_roll < float(self.daily.get("quiet_day_probability", 0.15)):
            return []
        slots = list(slots_spec.items())
        lo, hi = (list(self.daily.get("items_per_day") or [1, 3]) + [3])[:2]
        lo = max(int(lo), int(pace["daily_min_items"]))
        hi = max(int(hi), lo)
        count = min(rng.randint(lo, hi), len(slots))
        day_start = datetime(day.year, day.month, day.day, tzinfo=tz)
        window = timedelta(days=float(self.daily.get("repeat_window_days", 5)))
        recent = {e["content"] for e in self.store.events(("happened",), since=(day_start - window).isoformat(),
                                                          until=day_start.isoformat())
                  if e["source"] == SOURCE}
        minutes = int(self.daily.get("duration_minutes", 45))
        items = []
        for index in sorted(rng.sample(range(len(slots)), count)):
            name, spec = slots[index]
            pool = list(spec.get("items") or [])
            if day.weekday() >= 5:
                weekend_pool = [
                    i for i in pool
                    if not _WEEKDAY_CLASS_RE.search(f"{i.get('done', '')} {i.get('doing', '')}")
                ]
                if weekend_pool:
                    pool = weekend_pool
            options = [i for i in pool if i["done"] not in recent] or pool
            if not options:
                continue
            choice = rng.choice(options)
            first, last = spec.get("hours") or [9, 21]
            start = day_start.replace(hour=rng.randint(int(first), int(last) - 1), minute=rng.choice((0, 15, 30, 45)))
            items.append({"slot": name, "label": spec.get("label", name), "done": choice["done"],
                          "doing": choice.get("doing") or choice["done"], "start": start, "minutes": minutes,
                          "uid": f"daily:{day.isoformat()}:{name}"})
        return items

    def _messages(self, day: date) -> list[tuple[datetime, str]]:
        """(time, role) of the chat messages of ``day`` -- Violet's real activity that day."""
        rows = self.db.fetch_all("SELECT role, created_at FROM messages WHERE substr(created_at, 1, 10) = ?",
                                 (day.isoformat(),))
        return sorted((parse_time(r["created_at"]), r["role"]) for r in rows if r.get("created_at"))

    def chatting_at(self, start: datetime, messages: list[tuple[datetime, str]]) -> bool:
        """Phase 8.9.1: Violet was in the middle of a conversation when the item would start (messages shortly before
        AND shortly after it), so she was not off napping or at the café then."""
        gap = timedelta(minutes=float(self.daily.get("chat_gap_minutes", 20)))
        return (any(start - gap <= t < start for t, _ in messages)
                and any(start <= t <= start + gap for t, _ in messages))

    def day(self, day: date, now: datetime) -> dict[str, Any]:
        """{'day', 'done': [event rows], 'doing': item | None, 'later': [items]} -- stores what has started."""
        view: dict[str, Any] = {"day": day, "done": [], "doing": None, "later": [], "quiet": False}
        plan = self.plan_for(day, now.tzinfo)
        _, slots_spec = self._persona()
        messages = [(_match_tz(t, now.tzinfo), role) for t, role in self._messages(day)]
        # the day is only looked at while a user message is being answered (ContextEngine), and that message may
        # not be in the log yet (the IntentAgent's context is built first): "now" is chat activity too
        activity = messages + ([(now, "user")] if now.date() == day else [])
        default_minutes = int(self.daily.get("duration_minutes", 45))

        def _row_meta(row: dict[str, Any], fallback_item: dict[str, Any] | None = None) -> dict[str, Any]:
            detail = _parse_source_detail(row.get("source_detail"))
            row_uid = str(row.get("uid") or (fallback_item["uid"] if fallback_item else ""))
            occ = parse_time(row.get("occurred_at"))
            start = _match_tz(occ, now.tzinfo) if occ is not None else (
                fallback_item["start"] if fallback_item else datetime(day.year, day.month, day.day, tzinfo=now.tzinfo)
            )
            slot_name = str(detail.get("slot") or "")
            if not slot_name and row_uid.startswith(f"daily:{day.isoformat()}:"):
                slot_name = row_uid.split(":", 2)[2]
            if not slot_name and fallback_item:
                slot_name = str(fallback_item.get("slot") or "")
            if not slot_name:
                slot_name = "morning" if start.hour < 12 else "afternoon" if start.hour < 18 else "evening"
            slot_label = (
                str(detail.get("label") or "")
                or str((slots_spec.get(slot_name) or {}).get("label") or "")
                or (str(fallback_item.get("label") or "") if fallback_item else "")
                or _SLOT_FALLBACK_LABELS.get(slot_name, "")
            )
            content = str(row.get("content") or "")
            doing_text = (
                str(detail.get("doing") or "")
                or daily_doing_for(content, self.config)
                or (
                    str(fallback_item.get("doing") or "")
                    if fallback_item and fallback_item.get("done") == content
                    else ""
                )
                or content
            )
            duration = int(row.get("duration_minutes") or (fallback_item["minutes"] if fallback_item else default_minutes))
            return {
                "slot": slot_name,
                "label": slot_label,
                "done": content,
                "doing": doing_text,
                "start": start,
                "minutes": duration,
                "uid": row_uid,
                "event": row,
            }

        def _place_stored(row: dict[str, Any], fallback_item: dict[str, Any] | None = None) -> None:
            if row.get("status") != "happened":
                return
            meta = _row_meta(row, fallback_item=fallback_item)
            if meta["start"] > now:
                return
            end = meta["start"] + timedelta(minutes=meta["minutes"])
            # once Violet has replied since it started she has stopped to chat: it is done, not "正在做"
            replied = any(meta["start"] <= t <= now and role != "user" for t, role in messages)
            if now < end and not replied and view["doing"] is None:
                view["doing"] = meta
            else:
                view["done"].append({**row, "slot_label": meta["label"], "doing": meta["doing"]})

        seen_ids: set[int] = set()
        seen_uids: set[str] = set()
        for item in plan:
            stored = self.store.event_by_uid(item["uid"])
            if stored is None and item["start"] <= now:
                # an item that would start while she was chatting did not happen (never stored; already stored
                # events are history and are kept as they are)
                if self.chatting_at(item["start"], activity):
                    logger.info("daily item %s skipped: Violet was chatting at %s", item["uid"], item["start"])
                    continue
                stored = self._store(item, now)
                if stored is None:
                    continue                                   # rejected: simply not part of the day
            if stored is not None:
                seen_ids.add(int(stored["id"]))
                seen_uids.add(item["uid"])
                _place_stored(stored, fallback_item=item)
            else:
                view["later"].append(item)

        # Always preserve already-happened life_daily events for this day even if warmth_level later dropped
        # and plan_for() picked fewer items or rolled a quiet day.
        stored_today = self.db.fetch_all(
            "SELECT * FROM life_events WHERE source = ? AND status = 'happened' "
            "AND substr(occurred_at, 1, 10) = ? ORDER BY occurred_at, id",
            (SOURCE, day.isoformat()),
        )
        for row in stored_today:
            row_id = int(row["id"])
            row_uid = str(row.get("uid") or "")
            if row_id in seen_ids or (row_uid and row_uid in seen_uids):
                continue
            _place_stored(row)

        view["done"].sort(key=lambda e: (str(e.get("occurred_at") or ""), int(e.get("id") or 0)))
        view["quiet"] = (not plan) and (not view["done"]) and (view["doing"] is None)
        return view

    def ensure_today(self, now: datetime | None = None) -> dict[str, Any] | None:
        """Materialise today's daily items on demand (e.g. before a proactive check so she still has a day even when the
        owner has not messaged yet). Skips AI personas and disabled daily life."""
        if not self.enabled():
            return None
        try:
            from ai.persona import PersonaCore
            if PersonaCore.load().is_ai:
                return None
        except Exception:                                    # noqa: BLE001
            pass
        now = now or parse_time(self.db.now())
        try:
            return self.day(now.date(), now)
        except Exception:                                    # noqa: BLE001
            logger.exception("daily life ensure_today for %s unavailable", now.date())
            return None

    def recent(self, now: datetime | None = None) -> list[dict[str, Any]]:
        """Today and the days before it (oldest first); empty when disabled."""
        if not self.enabled():
            return []
        now = now or parse_time(self.db.now())
        days = int(self.daily.get("backfill_days", 2))
        out = []
        for back in range(days, -1, -1):
            try:
                out.append(self.day(now.date() - timedelta(days=back), now))
            except Exception:
                logger.exception("daily life for %s unavailable", now.date() - timedelta(days=back))
        return out

    def _store(self, item: dict[str, Any], now: datetime) -> dict[str, Any] | None:
        proposal = {"thread_id": None, "plan_id": None, "event_type": "daily", "content": item["done"],
                    "occurred_at": item["start"].isoformat(), "duration_minutes": item["minutes"],
                    "source": SOURCE,
                    "source_detail": {"slot": item["slot"], "label": item["label"], "doing": item["doing"]},
                    "completes_plan": False}
        violations = ConsistencyGuard(self.db, config=self.config).check_event(proposal, now.isoformat())
        if violations:
            logger.info("daily item %s skipped: %s", item["uid"], [v.check for v in violations])
            return None
        self.store.insert_event(uid=item["uid"], thread_id=None, plan_id=None, event_type="daily", content=item["done"],
                                occurred_at=proposal["occurred_at"], duration_minutes=item["minutes"], source=SOURCE,
                                source_detail=proposal["source_detail"], actor=SOURCE)
        return self.store.event_by_uid(item["uid"])


# ----------------------------------------------------------------------
# What a life line is "about": book titles, the configured item ("钢琴曲") and its details ("《月光》")
# ----------------------------------------------------------------------
def item_details(content: str, config: dict[str, Any] | None = None) -> str:
    """Concrete details configured for an item mentioned in ``content`` ("几首钢琴曲" -> 德彪西的《月光》…)."""
    config = config if config is not None else life_config()
    for spec in (config.get("interests") or {}).values():
        for item, detail in (spec.get("details") or {}).items():
            if item in (content or ""):
                return detail
    return ""


def life_terms(content: str, config: dict[str, Any] | None = None) -> list[str]:
    config = config if config is not None else life_config()
    content = content or ""
    terms = _TITLE_RE.findall(content)
    for spec in (config.get("interests") or {}).values():
        for item in spec.get("items") or []:
            if item in content:
                bare = _QUANTIFIER_RE.sub("", item)
                terms.append(bare if len(bare) >= 2 else item)
    terms += _TITLE_RE.findall(item_details(content, config))
    return [t for t in dict.fromkeys(terms) if len(t) >= 2]


def mentions_life(text: str, content: str, config: dict[str, Any] | None = None, *,
                  doing: str | None = None) -> bool:
    """``text`` talks about this life line: a shared title / item, topic overlap, doing phrase, or natural paraphrase."""
    text = text or ""
    if not text or not content:
        return False
    if any(t in text for t in life_terms(content, config)) or mentions_topic(text, content):
        return True
    resolved_doing = doing if doing is not None else daily_doing_for(content, config)
    if resolved_doing and doing_matches(text, resolved_doing):
        return True
    return _paraphrase_matches(text, content)


# ----------------------------------------------------------------------
# Phase 8.9.1: what earlier conversations already covered (derived from the message log, nothing stored)
# ----------------------------------------------------------------------
# The user asking about "your book" while Violet had exactly one book going is about that book.
YOUR_THING_RE = {"reading": re.compile(r"你.{0,6}(书|小说)"), "music": re.compile(r"你.{0,6}(曲子|钢琴曲|歌|专辑)"),
                 "notes": re.compile(r"你.{0,6}(笔记|本子)")}
REPLY_WITHIN = timedelta(hours=3)


def earlier_messages(db, before: datetime, limit: int = 300) -> list[dict[str, Any]]:
    """Messages before ``before`` (oldest first) as {'at', 'role', 'text'} -- after 翻篇 only those since then."""
    from ai import context_cut
    rows = context_cut.owner_rows(db.fetch_all(
        "SELECT role, content, created_at FROM messages ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)))
    out = [{"at": parse_time(r["created_at"]), "role": r["role"], "text": r["content"] or ""}
           for r in reversed(rows) if r.get("created_at")]       # phase 8.9.8: equal times keep the order they were sent in
    return sorted((m for m in out if m["at"] < before), key=lambda m: m["at"])


TALK_WITHIN = timedelta(days=2)
TALK_GAP = timedelta(hours=3)


def last_talk(db, content: str, now: datetime, config: dict[str, Any] | None = None, tail: int = 4) -> dict[str, Any] | None:
    """Phase 8.9.8: how the latest conversation about one of Violet's plans ended, from the message log (nothing stored,
    nothing inferred about it having happened). The conversation is the one with the latest message that mentions the
    plan; it runs on while messages follow each other within TALK_GAP; its last ``tail`` messages are returned.

    In the real free chat (local/free_chat/cys-close-1) the plan "周末想去书店挑一本新书" was talked over until "那正好，
    周末一起找吧 / 我去书店的时候拍封面给你"; the next morning Violet wrote "一直没定下来到底去不去"."""
    rows = [m for m in earlier_messages(db, now + timedelta(seconds=1)) if m["at"] >= now - TALK_WITHIN]
    hits = [i for i, m in enumerate(rows) if mentions_life(m["text"], content, config)]
    if not hits:
        return None
    end = hits[-1]
    while end + 1 < len(rows) and rows[end + 1]["at"] - rows[end]["at"] < TALK_GAP:
        end += 1
    part = rows[hits[-1]:end + 1][-tail:]
    return {"at": part[-1]["at"], "lines": [(m["role"], m["text"]) for m in part]}


def talk_note(talk: dict[str, Any] | None) -> str:
    """'上次聊到（11-06 23:50）最后是这样说的：…' -- the words, so the model continues from them; or ''."""
    if not talk:
        return ""
    said = " / ".join(f"{'对方' if role == 'user' else '你'}：“{text.strip()[:40]}”" for role, text in talk["lines"])
    return (f"上次聊到（{talk['at']:%m-%d %H:%M}）最后是这样说的：{said}。按聊到的往下说，别当成还没商量过；"
            "还没做的也别说成做了")


def violet_told(content: str, since: datetime, messages: list[dict[str, Any]],
                config: dict[str, Any] | None = None, *, doing: str | None = None) -> bool:
    """Violet herself talked about this life line in an earlier message written after it happened."""
    return any(m["role"] != "user" and m["at"] >= since and mentions_life(m["text"], content, config, doing=doing)
               for m in messages)


def thread_discussed(thread: dict[str, Any], threads: list[dict[str, Any]], messages: list[dict[str, Any]],
                     config: dict[str, Any] | None = None) -> bool:
    """An earlier conversation was about this thread while it was going on: a message names it (title / item), or
    the user asked about "your book" and Violet answered while this was her only thread of that kind."""
    start = parse_time(thread["started_at"])
    end = parse_time(thread["ended_at"]) if thread.get("ended_at") else None
    active = lambda th, t: parse_time(th["started_at"]) <= t and (not th.get("ended_at") or t <= parse_time(th["ended_at"]))
    terms = life_terms(thread["title"], config)
    pattern = YOUR_THING_RE.get(thread["interest_key"])
    for i, m in enumerate(messages):
        if m["at"] < start or (end and m["at"] > end):
            continue
        if any(t in m["text"] for t in terms):
            return True
        if m["role"] == "user" and pattern and pattern.search(m["text"]):
            answered = any(r["role"] != "user" and m["at"] <= r["at"] <= m["at"] + REPLY_WITHIN for r in messages[i + 1:i + 3])
            unique = not any(o["id"] != thread["id"] and o["interest_key"] == thread["interest_key"] and active(o, m["at"])
                             for o in threads)
            if answered and unique:
                return True
    return False
