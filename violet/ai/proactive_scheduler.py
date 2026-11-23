"""Scheduler for Violet's proactive outreach, explicit reminders, and holiday greetings (reworked v0.7.78).

Single-model architecture:
* Code enforces ONLY the 7 hard safety/rhythm gates (enabled/test mode, quiet hours, user off/busy,
  optional timetable gate, daily limit, consecutive unanswered count, and stage-scaled spacing).
* When due, the scheduler prepares a compact proactive note (warmth level, today's fresh daily/life
  events, optional upcoming user event / open plan / schedule moment / web finding, recent proactive
  openings, and phone snapshot) and calls ChatAgent ONCE.
* ChatAgent either writes a short natural message or outputs [NO_REPLY] (skipped, nothing sent).
* Post-send tracking marks shared life events, upcoming events, web findings, and schedule moments
  directly from the actual sent text.
"""
from __future__ import annotations

import json
import logging
import random
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from ai.chat import note_web_mentions, run_commitment_pipeline, run_proactive_pipeline, violetdb
from ai.conversation_state import (
    LIFE_TOPIC,
    VIOLET_SOURCES,
    WEB_TOPIC,
    TopicTracker,
    life_topic_key,
    memory_topic_key,
)
from ai.memory_model import PROGRESS_LABELS, event_deadline, parse_time, touches_topic
from ai.profile_schema import BUSY_VALUES_NO_CLASS_RE, BUSY_VALUES_RE
from ai.profile_store import Profile
from ai.settings import env_bool

logger = logging.getLogger("violet.proactive")

CHINA_TZ = timezone(timedelta(hours=8))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "proactive_config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "check_interval_seconds": 60,
    "quiet_hours": {"start": "23:30", "end": "08:00"},
    "daily_message_limit": 3,
    "min_minutes_after_user_message": 90,
    "min_minutes_between_proactive_messages": 180,
    "random_delay_minutes": {
        "morning": [90, 180],
        "noon": [90, 180],
        "afternoon": [120, 240],
        "evening": [90, 180],
        "night": [180, 300],
        "other": [120, 240],
    },
    # Backoff on completely unanswered proactive messages (any user reply resets to 0).
    "unanswered_backoff_factor": 2.0,
    "max_backoff_multiplier": 8,
    "max_cold_streak": 2,
    "event_topics": {
        "enabled": True,
        "lookahead_hours": 24,
        "max_days_since_user_mention": 2,
    },
    "commitments": {
        "enabled": True,
        "max_attempts": 3,
        "retry_minutes": 5,
    },
}

CHECKIN_RE = re.compile(
    r"(今天|今晚|今早|最近|这两天|这几天)[^，。！？!?\n]{0,8}(怎么样|怎么过|还好吗|还顺|顺不顺|累不累|忙不忙|过得)|"
    r"吃(饭)?了(吗|没)|吃得怎么样|在干嘛|在做什么|起床了(吗|没)"
)


def now_china() -> datetime:
    """Current time in the configured zone (VIOLET_TIMEZONE; name kept for compatibility)."""
    from ai.settings import now_local
    return now_local()


def _zone():
    from ai.settings import local_timezone
    return local_timezone()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=_zone())
    return parsed.astimezone(_zone())


def time_bucket(current: datetime | None = None) -> str:
    hour = (current or now_china()).hour
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 14:
        return "noon"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 23:
        return "evening"
    return "night"


def minutes_since(value: str | None, current: datetime | None = None) -> float | None:
    parsed = parse_iso(value)
    if not parsed:
        return None
    return ((current or now_china()) - parsed).total_seconds() / 60


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")
        return dict(default)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(default)
    if not isinstance(loaded, dict):
        return dict(default)
    return deep_merge(default, loaded)


def _format_gap(minutes: float | None) -> str:
    if minutes is None or minutes < 0:
        return "很久"
    total = int(minutes)
    if total < 60:
        return f"{max(1, total)} 分钟"
    hours, rem = divmod(total, 60)
    if hours >= 24:
        return f"{hours // 24} 天多"
    return f"{hours} 小时" + (f" {rem} 分钟" if rem >= 10 else "")


def _format_schedule_moment(sm: dict[str, Any]) -> str:
    """Render one schedule moment with its structured details (course, kind, start / last_end / day counts)."""
    mtype = sm.get("type")
    note = str(sm.get("note") or "").strip()
    if mtype == "exam_tomorrow":
        exam = str(sm.get("exam") or "").strip()
        kind = str(sm.get("kind") or "考试").strip()
        kind_label = f"{kind}考试" if kind in ("期末", "期中") else kind
        when = f"（{sm['start']}）" if sm.get("start") else ""
        head = f"明天有「{exam}」{kind_label}{when}" if exam else f"明天有{kind_label}{when}"
        return f"{head}：{note}" if note else head
    if mtype == "classes_over_today":
        last_end = sm.get("last_end")
        count = sm.get("classes_today")
        parts = []
        if last_end:
            parts.append(f"最后一节课到 {last_end}")
        if count:
            parts.append(f"今天共 {count} 节课")
        head = f"今天课表已结束（{'，'.join(parts)}）" if parts else ""
        return f"{head}：{note}" if head and note else (head or note or str(mtype or ""))
    if mtype == "dense_week":
        full_days = sm.get("full_days")
        total = sm.get("classes_next_7_days")
        parts = []
        if full_days:
            parts.append(f"未来 7 天有 {full_days} 天满课")
        if total:
            parts.append(f"共 {total} 节课")
        head = "，".join(parts)
        return f"{head}：{note}" if head and note else (head or note or str(mtype or ""))
    return note or str(mtype or "")


class ProactiveScheduler:
    """Background scheduler for Violet's proactive messages, explicit reminders, and holiday greetings."""

    HOLIDAY_ATTEMPTS = 2

    def __init__(
        self,
        chat_agent,
        send_messages: Callable[[list[dict[str, Any]], int], dict[str, Any] | None],
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        is_busy: Callable[[], bool] | None = None,
        pipeline_lock=None,
    ):
        self.chat_agent = chat_agent
        self.send_messages = send_messages
        self.config_path = Path(config_path)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._is_busy = is_busy or (lambda *_a: False)
        self._pipeline_lock = pipeline_lock or threading.RLock()
        self._scope_lock: Callable[[str | None], Any] | None = None
        self.tracker = TopicTracker(violetdb)
        self.profile = Profile(violetdb)
        self.config = self.load_config()

    def attach_runtime(self, is_busy: Callable[..., bool], pipeline_lock, channel_context: Callable | None = None,
                       scope_lock: Callable[[str | None], Any] | None = None) -> None:
        self._is_busy = is_busy
        self._pipeline_lock = pipeline_lock
        self._scope_lock = scope_lock
        if channel_context is not None:
            self._channel_context = channel_context

    def _busy_in_scope(self, scope: str) -> bool:
        try:
            return bool(self._is_busy(scope))
        except TypeError:
            return bool(self._is_busy())

    def _channel_scope(self):
        from ai.channels.context import channel_scope
        provider = getattr(self, "_channel_context", None)
        try:
            context = provider() if provider is not None else None
        except Exception:
            logger.exception("[ProactiveScheduler] channel unknown; using WeChat")
            context = None
        if context is None and provider is not None:
            from ai import memory_scope
            if memory_scope.current() != memory_scope.MAIN:
                return None
        if context is None:
            from ai import memory_scope
            if memory_scope.for_channel("wechat") != memory_scope.current():
                return None
        return channel_scope(context)

    def _each_scope(self, work) -> None:
        from ai import memory_scope
        for scope in memory_scope.all_scopes():
            if self._scope_lock is not None and self._busy_in_scope(scope):
                continue
            lock = self._scope_lock(scope) if self._scope_lock is not None else None
            if lock is not None and not lock.acquire(blocking=False):
                continue
            try:
                with memory_scope.run_as(scope):
                    scoped = self._channel_scope()
                    if scoped is None:
                        continue
                    try:
                        with scoped:
                            work()
                    except Exception:
                        logger.exception("[ProactiveScheduler] tick failed for persona %s", scope)
            finally:
                if lock is not None:
                    lock.release()

    def load_config(self) -> dict[str, Any]:
        self.config = load_json(self.config_path, DEFAULT_CONFIG)
        try:
            from ai.control_config import store
            overrides = store().proactive_overrides()
            if overrides:
                self.config = deep_merge(self.config, overrides)
        except Exception:
            logger.exception("[ProactiveScheduler] control center overrides unavailable; using config file")
        return self.config

    def reload(self) -> None:
        with self._lock:
            self.load_config()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def on_user_message(self) -> None:
        """Any message from the user resets unanswered backoff and schedules the next check cleanly."""
        with self._lock:
            self.load_config()
            self._set_next_proactive_at(self._random_next_time(now_china()))

    def _get_next_proactive_at(self) -> datetime | None:
        return parse_iso(violetdb.get_proactive_setting("next_proactive_at"))

    def _set_next_proactive_at(self, value: datetime) -> None:
        violetdb.set_proactive_setting("next_proactive_at", value.isoformat())

    # ------------------------------------------------------------------
    # Stage-scaled warmth & spacing
    # ------------------------------------------------------------------
    def _warmth_profile(self, current: datetime | None = None) -> tuple[Any, dict[str, Any] | None, str | None, int, dict[str, Any]]:
        from ai.relationship import RelationshipView, assess, proactive_pace, warmth_level
        try:
            view = assess(violetdb, current)
        except Exception:                                    # noqa: BLE001
            view = RelationshipView("new")
        try:
            from ai import persona_templates as pt
            persona_stage = pt.stage(None, violetdb)
            rel_kind = (pt.relationship(None) or {}).get("kind")
        except Exception:                                    # noqa: BLE001
            persona_stage, rel_kind = None, None
        level = warmth_level(view, persona_stage, rel_kind=rel_kind)
        return view, persona_stage, rel_kind, level, proactive_pace(level)

    def _effective_intervals(self, current: datetime | None = None) -> tuple[float, float, int, dict[str, Any]]:
        from ai.relationship import scaled_minutes
        _, _, _, level, pace = self._warmth_profile(current)
        try:
            pref = self.profile.preferences.effective().get("proactive_chat") or {}
        except Exception:                                    # noqa: BLE001
            pref = {}
        less = pref.get("value") == "less"
        if less:
            pace = {**pace, "delay_scale": 1.0, "after_user_scale": 1.0, "interval_scale": 1.0}
        base_after = float(self.config.get("min_minutes_after_user_message", 30))
        base_between = float(self.config.get("min_minutes_between_proactive_messages", 120))
        after_scale = 1.0 if less else float(pace["after_user_scale"])
        between_scale = 1.0 if less else float(pace["interval_scale"])
        return (
            scaled_minutes(base_after, after_scale, floor=25),
            scaled_minutes(base_between, between_scale, floor=50),
            level,
            pace,
        )

    def _random_next_time(self, current: datetime) -> datetime:
        from ai.relationship import scaled_minutes
        bucket = time_bucket(current)
        ranges = self.config.get("random_delay_minutes", {})
        min_delay, max_delay = ranges.get(bucket) or ranges.get("other") or [60, 150]
        _, _, _, pace = self._effective_intervals(current)
        scale = float(pace["delay_scale"])
        lo = int(round(scaled_minutes(float(min_delay), scale, floor=20)))
        hi = max(lo, int(round(scaled_minutes(float(max_delay), scale, floor=25))))
        return current + timedelta(minutes=random.randint(lo, hi))

    def _preference_multiplier(self) -> float:
        pref = self.profile.preferences.effective().get("proactive_chat") or {}
        if pref.get("value") != "less":
            return 1.0
        return 2.0 if pref.get("origin") == "explicit" else 1.5

    def _backoff_next_time(self, current: datetime) -> datetime:
        base = self._random_next_time(current)
        unanswered = self.unanswered_count(current)
        factor = float(self.config.get("unanswered_backoff_factor", 2.0))
        cap = float(self.config.get("max_backoff_multiplier", 8))
        multiplier = min(cap, factor ** unanswered) * self._preference_multiplier()
        return current + (base - current) * multiplier

    def unanswered_count(self, current: datetime | None = None) -> int:
        """Consecutive proactive chat messages sent since the user's latest message (any reply resets to 0)."""
        return len(self._unanswered_proactive(current or now_china()))

    # ------------------------------------------------------------------
    # Loop & ticks
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("[ProactiveScheduler] tick failed")
            interval = int(self.config.get("check_interval_seconds", 60))
            self._stop_event.wait(max(5, interval))

    def tick(self) -> None:
        if self._scope_lock is not None:
            if not self._busy_in_scope("main"):
                main_lock = self._scope_lock("main")
                if main_lock.acquire(blocking=False):
                    try:
                        self._poll_phone_events()
                    finally:
                        main_lock.release()
            self._each_scope(self._tick_locked)
            return
        if self._is_busy():
            return
        if not self._pipeline_lock.acquire(blocking=False):
            return
        try:
            self._poll_phone_events()
            self._each_scope(self._tick_locked)
        finally:
            self._pipeline_lock.release()

    def _tick_locked(self) -> None:
        with self._lock:
            self.load_config()
            if not self.config.get("enabled", True):
                return
            current = now_china()
            violetdb.refresh_event_states()
            self.tracker.expire_unanswered()

        # Explicit user-requested reminders and holiday greetings run outside the regular chat cap.
        if self._deliver_due_commitments(current):
            return
        if self._holiday_greeting(current):
            return

        with self._lock:
            due, reason = self._is_due(current)
            moment = self._phone_moment(current) if not due and reason == "not due yet" else None
            if moment:
                due, reason = self._is_due(current, ignore_next_at=True)
                reason = "phone moment" if due else reason
            if not due:
                return
            event_context = self._build_event_context(current)
            if moment:
                event_context["phone_moment"] = moment
                from ai import phone_events
                phone_events.consume_moment()
            self._set_next_proactive_at(self._random_next_time(current))

        self._add_phone(event_context)
        logger.info("[ProactiveScheduler] triggering proactive pipeline: %s", reason)
        pipeline = run_proactive_pipeline(event_context=event_context, chat_agent=self.chat_agent)
        self._handle_pipeline_result(pipeline, event_context, kind="scheduled")

    @staticmethod
    def _latest_user_message_at() -> str | None:
        from ai import context_cut
        return violetdb.get_latest_message_at(role="user", since=context_cut.owner_floor() or None)

    @staticmethod
    def _latest_proactive_sent_at() -> str | None:
        from ai import context_cut
        return violetdb.get_latest_proactive_event_at(status="sent", since=context_cut.owner_floor() or None)

    def status_snapshot(self) -> dict[str, Any]:
        with self._lock:
            self.load_config()
            current = now_china()
            due, reason = self._is_due(current)
            from ai.relationship import WARMTH_LABELS
            view, _, _, level, _ = self._warmth_profile(current)
            eff_after, eff_between, _, _ = self._effective_intervals(current)
            unanswered = self.unanswered_count(current)
            last_user = self._latest_user_message_at()
            last_sent = self._latest_proactive_sent_at()
            return {
                "enabled": self.config.get("enabled", True),
                "current_time": current.isoformat(),
                "next_proactive_at": violetdb.get_proactive_setting("next_proactive_at"),
                "last_user_message_at": last_user,
                "last_proactive_sent_at": last_sent,
                "minutes_since_user_message": minutes_since(last_user, current),
                "minutes_since_last_proactive": minutes_since(last_sent, current),
                "daily_sent_count": self._today_sent_count(current),
                "daily_message_limit": self.config.get("daily_message_limit"),
                "relationship_stage": view.stage,
                "warmth_level": level,
                "warmth_label": WARMTH_LABELS.get(level, ""),
                "effective_min_after_user_minutes": eff_after,
                "effective_min_between_minutes": eff_between,
                "is_due": due,
                "due_reason": reason,
                "unanswered_count": unanswered,
                "cold_streak": unanswered,
                "cooling_topics": self.tracker.cooling_topics(),
                "pending_commitments": violetdb.list_commitments(["pending", "sending"], limit=20),
            }

    def force_run(self, send: bool = False, event_context_patch: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            self.load_config()
            current = now_china()
            event_context = self._build_event_context(current)
            if event_context_patch:
                event_context = deep_merge(event_context, event_context_patch)
            event_context["event_type"] = "manual_proactive_test"
            event_context["manual_test"] = True

        if not self._pipeline_lock.acquire(timeout=180):
            raise RuntimeError("pipeline busy: a user turn is still being processed")
        try:
            scoped = self._channel_scope()
            if scoped is None:
                raise RuntimeError("这个人设没有可以发消息的渠道")
            with scoped:
                pipeline = run_proactive_pipeline(event_context=event_context, chat_agent=self.chat_agent)
                if send:
                    self._handle_pipeline_result(pipeline, event_context)
        finally:
            self._pipeline_lock.release()
        return pipeline

    # ------------------------------------------------------------------
    # Hard gates (7 simple rules)
    # ------------------------------------------------------------------
    def _is_due(self, current: datetime, ignore_next_at: bool = False) -> tuple[bool, str]:
        from ai import test_mode
        if test_mode.active():
            return False, "test mode"
        next_at = self._get_next_proactive_at()
        if not next_at and not ignore_next_at:
            self._set_next_proactive_at(self._random_next_time(current))
            return False, "next proactive time initialized"
        if not ignore_next_at and current < next_at:
            return False, "not due yet"
        if self._in_quiet_hours(current):
            self._set_next_proactive_at(self._random_next_time(current))
            return False, "quiet hours"
        since_user = minutes_since(self._latest_user_message_at(), current)
        if since_user is None:
            return False, "no user message yet"
        min_after_user, min_between, _, _ = self._effective_intervals(current)
        if since_user < min_after_user:
            return False, "too soon after user message"
        since_sent = minutes_since(self._latest_proactive_sent_at(), current)
        if since_sent is not None and since_sent < min_between:
            return False, "too soon after proactive message"
        prefs = self.profile.preferences.effective()
        proactive_pref = prefs.get("proactive_chat") or {}
        if proactive_pref.get("origin") == "explicit" and proactive_pref.get("value") == "no":
            return False, "the user asked Violet not to start conversations"
        daily_limit = int(self.config.get("daily_message_limit", 3))
        if proactive_pref.get("origin") == "explicit" and proactive_pref.get("value") == "less":
            daily_limit = min(daily_limit, 1)
        busy = self._busy_situation()
        if busy is not None:
            return False, f"the user said they are busy ({busy['value']}) until {busy['expires_at'][5:16]}"
        gate = self._schedule_gate(current)
        if gate is not None and gate.blocked:
            self._postpone_past(gate)
            return False, gate.reason()
        if self._today_sent_count(current) >= daily_limit:
            return False, "daily proactive limit reached"
        unanswered = self.unanswered_count(current)
        if unanswered >= int(self.config.get("max_cold_streak", 2)):
            return False, f"waiting for the user to reply ({unanswered} unanswered)"
        return True, "due"

    def _in_quiet_hours(self, current: datetime) -> bool:
        quiet = self.config.get("quiet_hours") or {}
        start, end = quiet.get("start"), quiet.get("end")
        if not start or not end:
            return False
        now_m = current.hour * 60 + current.minute
        start_m, end_m = self._hhmm_to_minutes(start), self._hhmm_to_minutes(end)
        if start_m <= end_m:
            return start_m <= now_m < end_m
        return now_m >= start_m or now_m < end_m

    @staticmethod
    def _hhmm_to_minutes(value: str) -> int:
        hour, minute = value.split(":", 1)
        return int(hour) * 60 + int(minute)

    def _day_start(self, current: datetime) -> str:
        return current.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    def _today_sent_count(self, current: datetime) -> int:
        from ai import context_cut
        return violetdb.count_proactive_events_since(
            context_cut.owner_floor(self._day_start(current)),
            status="sent",
            exclude_event_types=("commitment_reminder", "holiday_greeting"),
        )

    def _busy_situation(self) -> dict[str, Any] | None:
        pattern = BUSY_VALUES_RE if env_bool("VIOLET_SCHEDULE_PROACTIVE", False) else BUSY_VALUES_NO_CLASS_RE
        for situation in self.profile.situations.current(["availability", "activity"]):
            if pattern.search(situation["value"]):
                return situation
        return None

    # ------------------------------------------------------------------
    # Single-call proactive context & direct note for ChatAgent
    # ------------------------------------------------------------------
    def _build_event_context(self, current: datetime) -> dict[str, Any]:
        try:
            from ai.daily_life import DailyLife
            DailyLife(violetdb).ensure_today(current)
        except Exception:                                    # noqa: BLE001
            logger.debug("[ProactiveScheduler] daily life ensure_today skipped", exc_info=True)

        from ai.relationship import WARMTH_LABELS
        view, _, _, level, _ = self._warmth_profile(current)
        state = self._conversation_state(current)
        schedule_moments = self._schedule_moments(current)
        if schedule_moments:
            state["allowed_schedule_moments"] = schedule_moments
        unanswered = self._unanswered_proactive(current)
        recent, habit_note = self._recent_proactive(current)
        plans = self._shareable_open_plans(current)
        since_user = minutes_since(self._latest_user_message_at(), current)
        sent_today = self._today_sent_count(current)
        try:
            proactive_pref = self.profile.preferences.effective().get("proactive_chat") or {}
        except Exception:                                    # noqa: BLE001
            proactive_pref = {}
        less_proactive = proactive_pref.get("value") == "less"
        daily_limit = int(self.config.get("daily_message_limit", 3))
        if proactive_pref.get("origin") == "explicit" and less_proactive:
            daily_limit = min(daily_limit, 1)
        class_ok = not env_bool("VIOLET_SCHEDULE_PROACTIVE", False)

        note, summary_topic = self._compose_proactive_note(
            view=view,
            level=level,
            since_user=since_user,
            sent_today=sent_today,
            daily_limit=daily_limit,
            unanswered=unanswered,
            life_events=state.get("allowed_life_events") or [],
            upcoming=[m for m in state.get("allowed_memories") or [] if m.get("type") == "upcoming_event"],
            open_plans=plans,
            web_findings=state.get("allowed_web_findings") or [],
            schedule_moments=schedule_moments,
            recent=recent,
            habit_note=habit_note,
            class_ok=class_ok,
            less_proactive=less_proactive,
        )
        return {
            "event_type": "scheduled_proactive_check",
            "scheduled_at": current.isoformat(),
            "current_time": current.isoformat(),
            "time_bucket": time_bucket(current),
            "minutes_since_user_message": since_user,
            "minutes_since_last_proactive": minutes_since(self._latest_proactive_sent_at(), current),
            "daily_sent_count": sent_today,
            "daily_message_limit": daily_limit,
            "relationship_stage": view.stage,
            "warmth_level": level,
            "warmth_label": WARMTH_LABELS.get(level, ""),
            "class_proactive_allowed": class_ok,
            "summary_topic": summary_topic,
            "summary_reason": f"热度：{WARMTH_LABELS.get(level, '')}（{summary_topic}）",
            "proactive_note": note,
            "conversation_state": state,
        }

    def _compose_proactive_note(
        self,
        *,
        view,
        level: int,
        since_user: float | None,
        sent_today: int,
        daily_limit: int,
        unanswered: list[dict[str, Any]],
        life_events: list[dict[str, Any]],
        upcoming: list[dict[str, Any]],
        open_plans: list[dict[str, Any]],
        web_findings: list[dict[str, Any]],
        schedule_moments: list[dict[str, Any]],
        recent: list[dict[str, Any]],
        habit_note: str | None,
        class_ok: bool,
        less_proactive: bool = False,
    ) -> tuple[str, str]:
        from ai.relationship import WARMTH_LABELS
        gap = _format_gap(since_user)
        lines: list[str] = ["【主动联系参考】"]
        if view.stage == "distance":
            lines.append("- 对方明确要求保持轻松友好的距离：除非下面有很具体值得分享的小事，否则直接输出 [NO_REPLY]；即便开口也不撒娇、不用亲昵称呼。")
            topic = "distance"
        elif less_proactive:
            lines.append(
                f"- 对方希望你平时少主动发消息（当前热度：{WARMTH_LABELS.get(level, WARMTH_LABELS[0])}，"
                f"距对方上次说话约 {gap}，今日已主动 {sent_today}/{daily_limit} 次）："
                "尊重对方的节奏，克制一点；下面有具体值得分享的小事或临近事项时再简短自然地说一句，"
                "没有合适内容就直接输出 [NO_REPLY]，不用刻意来蹭存在感。"
            )
            topic = "less_proactive"
        elif level >= 3:
            lines.append(f"- 你们已经很亲近（{WARMTH_LABELS[level]}），距对方上次说话约 {gap}（今日已主动 {sent_today}/{daily_limit} 次）："
                         "可以自然热情地来蹭个存在感、分享刚发生的小事、撒个娇或说一句想对方，贴合你的人设和阶段，简短自然即可。")
            topic = "warm_deep"
        elif level == 2:
            lines.append(f"- 你们已经挺熟了（{WARMTH_LABELS[level]}），距对方上次说话约 {gap}（今日已主动 {sent_today}/{daily_limit} 次）："
                         "只要有今天的小事、想念或随口冒出的小念头，就大方发一句（不用过度犹豫，也别问空洞的“吃了吗/在干嘛”）。")
            topic = "warm_active"
        elif level == 1:
            lines.append(f"- 你们正在慢慢熟起来（{WARMTH_LABELS[level]}），距对方上次说话约 {gap}（今日已主动 {sent_today}/{daily_limit} 次）："
                         "有下面的生活小事、计划，或隔了一阵子想轻轻搭一句时，自然说一句就好。")
            topic = "warming"
        else:
            lines.append(f"- 你们还在互相熟悉（{WARMTH_LABELS[0]}），距对方上次说话约 {gap}："
                         "下面有具体生活小事或临近事项时再顺口发一句；没有合适话题就输出 [NO_REPLY]，还不说想念、不撒娇。")
            topic = "initial"

        if unanswered:
            last = unanswered[0]
            waited = float(last.get("elapsed_hours", last.get("hours", 0)))
            gate = 3.0 if level >= 3 else 4.0
            if level >= 2 and not view.cooled and not less_proactive and len(unanswered) == 1 and waited >= gate:
                lines.append(f"- 你 {last.get('at', '')} 主动发过一条，对方过了约 {int(waited)} 小时还没回："
                             "已经很亲近了，可以带点轻巧小情绪再找一次（想对方、小委屈或玩笑式撒娇），只找这一次；不认真指责、不重复上一条内容。")
                topic = "miss_unanswered"
            else:
                lines.append(f"- 你 {last.get('at', '')} 发的那条对方还没回：若没新鲜事就输出 [NO_REPLY]，别重复催问。")

        if class_ok:
            lines.append("- 时机判断：对方打游戏、刷视频、刷手机或平时上课都不算勿扰，照常可以发简短轻松的消息；"
                         "只有深夜屏幕熄灭很久疑似睡觉，或对方亲口说过正忙/开会/别打扰时才输出 [NO_REPLY]。")
        else:
            lines.append("- 时机判断：对方打游戏、刷视频、刷手机不算勿扰；只有深夜疑似睡觉或对方正忙/别打扰时才输出 [NO_REPLY]。")

        if life_events:
            if topic != "miss_unanswered":
                topic = "daily_share" if any(e.get("type") == "violet_daily_event" for e in life_events) else "life_share"
            lines.append("- 你最近刚发生、还没告诉对方的小事（选一件自然随口提一句，用自己的语气）：")
            for ev in life_events:
                lines.append(f"  · {ev['content']}")
        if upcoming:
            topic = "upcoming_event"
            lines.append("- 对方即将到来的事（可在发生前自然提一次比如祝顺利，别追问结果）：")
            for m in upcoming[:2]:
                lines.append(f"  · {m['content']}（{m.get('event_time') or ''}）")
        if open_plans:
            lines.append("- 你自己近期的打算（可选聊一句看法或新想法，还没做的别说成做了）：")
            for p in open_plans[:2]:
                extra = f"；{p['last_talk']}" if p.get("last_talk") else ""
                lines.append(f"  · {p['content']}{extra}")
        if schedule_moments:
            lines.append("- 课表小背景（仅供参考，别说成确定事实，不提也正常）：")
            for sm in schedule_moments[:2]:
                lines.append(f"  · {_format_schedule_moment(sm)}")
        if web_findings:
            lines.append("- 你在网上看到、可能合对方胃口的资料（不是你亲身经历，若提到请说明来源网站，不提也正常）：")
            for wf in web_findings[:1]:
                lines.append(f"  · {wf['title']}（来源：{wf['domain']}）")
        if recent:
            preview = " / ".join(f"{r['at']}「{r['text'].replace(chr(10), ' ')[:26]}」" for r in recent[-4:])
            lines.append(f"- 你最近主动发过：{preview}（换个新角度，别重复同样的话）。")
        if habit_note:
            lines.append(f"- {habit_note}")
        return "\n".join(lines), topic

    def _conversation_state(self, current: datetime | None = None) -> dict[str, Any]:
        cooling = self.tracker.cooling_topics()
        cooling_ids = {c.get("memory_id") for c in cooling if c.get("memory_id")}
        upcoming = self._mentionable_upcoming_events(cooling_ids)
        upcoming_ids = {m["id"] for m in upcoming}
        restricted = [
            {
                "memory_id": m["id"],
                "content": m["content"],
                "reason": f"{m.get('kind')}: {PROGRESS_LABELS.get(m.get('progress'), m.get('progress')) or m.get('progress')}",
            }
            for m in violetdb.list_recent_event_memories(kinds=["event", "legacy", "background"], limit=20)
            if m["id"] not in upcoming_ids
        ]
        avoid = self.profile.preferences.effective().get("avoid_topic", [])
        restricted += [{"memory_id": None, "content": item["value"], "reason": "user asked not to dwell on this"}
                       for item in avoid]
        allowed = [
            {"id": m["id"], "content": m["content"], "type": "fact"}
            for m in violetdb.list_recent_event_memories(kinds=["fact"], limit=10)
            if m.get("status") == "active" and m["id"] not in cooling_ids
        ]
        allowed += upcoming
        unanswered = self.unanswered_count(current)
        return {
            "unanswered_count": unanswered,
            "cold_streak": unanswered,
            "cooling_topics": cooling,
            "restricted_memories": restricted,
            "allowed_memories": allowed,
            "allowed_life_events": self._shareable_life_events(),
            "allowed_web_findings": self._shareable_web_findings(),
            "avoid_topics": [item["value"] for item in avoid],
        }

    # ------------------------------------------------------------------
    # Shareable content collectors
    # ------------------------------------------------------------------
    def _shareable_life_events(self) -> list[dict[str, Any]]:
        try:
            from ai.life_store import LifeStore
            from ai.persona import life_config
            cfg = life_config().get("sharing") or {}
        except Exception:
            logger.exception("[life] sharing config unavailable")
            return []
        if not cfg.get("enabled", True):
            return []
        if self.tracker.topic_state(LIFE_TOPIC)["unengaged"] >= int(cfg.get("cooling_after_disengaged_shares", 2)):
            return []
        result, daily_today = [], []
        today_iso = violetdb.now()[:10]
        for event in LifeStore(violetdb).recent_happened(days=float(cfg.get("recent_days", 3))):
            if int(event["share_count"]) >= int(cfg.get("max_shares_per_event", 1)):
                continue
            if self.tracker.topic_state(life_topic_key(event["id"]))["cooling"]:
                continue
            if event["source"] == "life_daily":
                if str(event.get("occurred_at") or "")[:10] == today_iso:
                    daily_today.append({
                        "id": event["id"], "content": event["content"], "occurred_at": event["occurred_at"],
                        "type": "violet_daily_event",
                    })
                continue
            result.append({
                "id": event["id"], "content": event["content"], "occurred_at": event["occurred_at"],
                "type": "violet_life_event",
            })
        _, _, _, _, pace = self._warmth_profile()
        picked = result[-2:]
        if len(picked) < 2 and daily_today:
            room = min(int(pace.get("max_daily_offers") or 1), 2 - len(picked))
            picked.extend(daily_today[-room:])
        return picked

    def _shareable_open_plans(self, current: datetime) -> list[dict[str, Any]]:
        try:
            from ai.daily_life import last_talk, talk_note
            from ai.life_store import LifeStore
            plans = LifeStore(violetdb).plans(["planned", "active"])
            raised = self._raised_unanswered(plans)
            out = []
            for plan in plans:
                if plan["id"] in raised:
                    continue
                note = talk_note(last_talk(violetdb, plan["content"], current))
                out.append({"id": plan["id"], "content": plan["content"], "last_talk": note or None})
            return out
        except Exception:                                    # noqa: BLE001
            return []

    def _shareable_web_findings(self) -> list[dict[str, Any]]:
        try:
            from ai.memory_model import mentions_topic
            from ai.web.explorer import explore_config
            from ai.web.findings import FindingsStore
            from ai.web.providers import _bigrams
            cfg = explore_config().get("sharing") or {}
        except Exception:
            logger.exception("[web] sharing config unavailable")
            return []
        if not cfg.get("enabled", True):
            return []
        store = FindingsStore(violetdb)
        now = parse_time(violetdb.now())
        if self.tracker.topic_state(WEB_TOPIC)["unengaged"] >= int(cfg.get("cooling_after_disengaged_shares", 2)):
            return []
        last = violetdb.fetch_one("SELECT MAX(shared_with_user_at) AS t FROM web_findings")
        if last and last["t"] and now - parse_time(last["t"]) < timedelta(days=float(cfg.get("min_days_between_shares", 3))):
            return []
        interests = [f["value"] for f in self.profile.facts.list_facts(["active"], fact_key="interest")
                     if f.get("use_in_context", 1)]
        if not interests:
            return []
        avoid = [item["value"] for item in self.profile.preferences.effective().get("avoid_topic", [])]
        shared = [(f["id"], f["title"]) for f in violetdb.fetch_all("SELECT id, title FROM web_findings WHERE share_count > 0")]
        result = []
        for f in store.list(limit=50, status="active", origin="explorer"):
            text = f"{f['title']} {f['summary'] or ''}"
            if f["share_count"] or not store.is_latest(f, float(cfg.get("freshness_days", 14))):
                continue
            if not any(mentions_topic(text, interest) for interest in interests):
                continue
            if any(touches_topic(text, a) or mentions_topic(text, a) for a in avoid):
                continue
            a = _bigrams(f["title"])
            if any(a and len(a & _bigrams(t)) / len(a | _bigrams(t)) >= 0.5
                   for other_id, t in shared if other_id != f["id"] and _bigrams(t)):
                continue
            result.append({"id": f["id"], "title": f["title"], "domain": f["domain"],
                           "summary": (f["summary"] or "")[:200], "published_at": f["published_at"],
                           "type": "web_finding"})
        return result[: int(cfg.get("max_candidates", 1))]

    def _mentionable_upcoming_events(self, cooling_memory_ids: set) -> list[dict[str, Any]]:
        cfg = self.config.get("event_topics") or {}
        if not cfg.get("enabled", True):
            return []
        now = parse_time(violetdb.now())
        horizon = now + timedelta(hours=float(cfg.get("lookahead_hours", 24)))
        fresh_after = now - timedelta(days=float(cfg.get("max_days_since_user_mention", 2)))
        result = []
        for m in violetdb.list_recent_event_memories(kinds=["event"], include_ended=False, limit=30):
            event_time = event_deadline(m.get("event_time"))
            raised_by_user = parse_time(m.get("last_user_mention_at")) or parse_time(m.get("created_at"))
            if m.get("progress") != "planned" or not event_time or not (now < event_time <= horizon):
                continue
            if raised_by_user < fresh_after or m["id"] in cooling_memory_ids or m.get("status") != "active":
                continue
            if violetdb.list_topic_mentions(topic_key=memory_topic_key(m["id"]), sources=VIOLET_SOURCES,
                                            since=raised_by_user.isoformat(), limit=1):
                continue
            result.append({"id": m["id"], "content": m["content"], "type": "upcoming_event",
                           "event_time": m["event_time"]})
        return result

    def _unanswered_proactive(self, current: datetime) -> list[dict[str, Any]]:
        from ai import context_cut
        last_user = context_cut.owner_floor(violetdb.get_latest_message_at(role="user"))
        last_user_at = parse_time(last_user) if last_user else None
        out = []
        for row in violetdb.fetch_all(
            "SELECT created_at FROM proactive_events WHERE status = 'sent' "
            "AND event_type NOT IN ('commitment_reminder', 'holiday_greeting') ORDER BY id DESC LIMIT 20"
        ):
            at = parse_time(row["created_at"]) if row["created_at"] else None
            if at is None or (last_user_at is not None and at <= last_user_at):
                continue
            elapsed = (current - at).total_seconds() / 3600
            out.append({"at": at.strftime("%H:%M"), "hours": round(elapsed, 1), "elapsed_hours": elapsed})
        return out[::-1]

    def _raised_unanswered(self, plans: list[dict[str, Any]]) -> dict[int, datetime]:
        from ai import context_cut
        from ai.daily_life import mentions_life
        from ai.life_store import LifeStore
        last_user = context_cut.owner_floor(violetdb.get_latest_message_at(role="user"))
        last_user_at = parse_time(last_user) if last_user else None
        raised: dict[int, datetime] = {}
        for row in violetdb.fetch_all(
            "SELECT created_at, sent_messages FROM proactive_events WHERE status = 'sent' "
            "AND event_type NOT IN ('commitment_reminder', 'holiday_greeting') ORDER BY id DESC LIMIT 20"
        ):
            at = parse_time(row["created_at"]) if row["created_at"] else None
            if at is None or (last_user_at is not None and at <= last_user_at):
                continue
            try:
                sent = json.loads(row["sent_messages"] or "[]")
            except (TypeError, ValueError):
                sent = []
            text = "\n".join(str(m.get("content") if isinstance(m, dict) else m) for m in sent if m)
            for plan in plans:
                if plan["id"] not in raised and mentions_life(text, plan["content"]):
                    raised[plan["id"]] = at
        if raised:
            contents = {p["id"]: p["content"] for p in plans}
            for event in LifeStore(violetdb).recent_happened(days=3):
                when = parse_time(event["occurred_at"])
                for pid, at in list(raised.items()):
                    if when > at and mentions_life(event["content"], contents[pid]):
                        del raised[pid]
        return raised

    def _recent_proactive(self, current: datetime) -> tuple[list[dict[str, Any]], str | None]:
        from ai import context_cut
        since = current - timedelta(days=3)
        cut = context_cut.owner_floor()
        if cut:
            since = max(since, parse_time(cut))
        recent = []
        for row in violetdb.fetch_all(
            "SELECT created_at, sent_messages FROM proactive_events WHERE status = 'sent' "
            "AND event_type NOT IN ('commitment_reminder', 'holiday_greeting') ORDER BY id DESC LIMIT 20"
        )[::-1]:
            if not row["created_at"] or parse_time(row["created_at"]) < since:
                continue
            try:
                sent = json.loads(row["sent_messages"] or "[]")
            except (TypeError, ValueError):
                sent = []
            text = "\n".join(str(m.get("content") if isinstance(m, dict) else m) for m in sent if m)
            if text:
                recent.append({"at": row["created_at"][5:16].replace("T", " "), "text": text,
                               "checkin": bool(CHECKIN_RE.search(text))})
        checkins = [r for r in recent if r["checkin"]]
        if not checkins:
            return recent, None
        example = checkins[-1]["text"].replace("\n", " ")[:40]
        return recent, (f"这几天你主动找对方 {len(recent)} 次，有 {len(checkins)} 次是用问近况收尾的（比如“{example}”）。"
                        "这次换个样子：只说你想说的那件事、一句想念或撒娇就好，不用再问对方怎么样。")

    # ------------------------------------------------------------------
    # Send handling & post-send tracking (derived directly from sent text)
    # ------------------------------------------------------------------
    def _handle_pipeline_result(self, pipeline: dict[str, Any], event_context: dict[str, Any] | None = None,
                                kind: str | None = None) -> None:
        event_id = pipeline.get("event_id")
        chat_result = pipeline.get("chat")
        from ai import trace
        trace.step("主动联系", 方式=kind or "手动测试", 事件=event_id,
                   要发送=pipeline.get("should_send", True), 内容=trace.text((chat_result or {}).get("content")),
                   错误=pipeline.get("error"))
        if event_id is None or not chat_result or not pipeline.get("should_send", True):
            return
        event_id = int(event_id)
        content = chat_result.get("content", "")
        from ai.channels.outbound import chat_payload
        payload = chat_payload(content if isinstance(content, str) else "", images=False)
        if not payload.get("should_send", False):
            violetdb.update_proactive_event(event_id, status="skipped", reason="ChatAgent decided not to send")
            return
        if kind is not None and self._schedule() is not None:
            with self._lock:
                blocked = self._final_check(now_china())
            if blocked:
                logger.info("[ProactiveScheduler] not sent, final check: %s", blocked)
                violetdb.update_proactive_event(event_id, status="skipped", reason=f"final check: {blocked}"[:500])
                return
        messages = payload.get("messages", [])
        try:
            report = self.send_messages(messages, event_id)
        except Exception as exc:
            logger.exception("proactive send crashed for event %s", event_id)
            violetdb.update_proactive_event(event_id, status="failed", reason=f"send error: {exc}"[:500])
            return
        if report is None:
            violetdb.update_proactive_event(event_id, status="sent", sent_messages=messages)
            self._after_proactive_sent(event_context, messages, event_id)
            return
        sent = report.get("sent_messages") or []
        if sent:
            reason = None
            if report.get("failed_count") or report.get("cancelled_count"):
                reason = f"partially sent {len(sent)}/{len(messages)}: {report.get('error')}"[:500]
            violetdb.update_proactive_event(event_id, status="sent", sent_messages=sent, reason=reason)
            self._after_proactive_sent(event_context, sent, event_id)
        else:
            violetdb.update_proactive_event(event_id, status="failed", sent_messages=[],
                                            reason=f"send failed: {report.get('error')}"[:500])

    def _after_proactive_sent(self, event_context: dict[str, Any] | None, sent: list[Any], event_id: int) -> None:
        try:
            sent_text = "\n".join(str(m.get("content") if isinstance(m, dict) else m) for m in sent if m)
            state = (event_context or {}).get("conversation_state") or {}

            # 1. Mark any offered life / daily events mentioned in the sent text as shared
            from ai.daily_life import mentions_life
            from ai.life_store import LifeStore
            store = LifeStore(violetdb)
            shared_life_ids: list[int] = []
            for ev in state.get("allowed_life_events") or []:
                eid = ev.get("id")
                if eid is not None and mentions_life(sent_text, ev.get("content") or ""):
                    shared_life_ids.append(int(eid))
            for eid in shared_life_ids:
                if store.event(eid):
                    store.mark_shared(eid)
                    violetdb.add_topic_mention(life_topic_key(eid), "violet_life", proactive_event_id=event_id)
            if shared_life_ids:
                violetdb.add_topic_mention(LIFE_TOPIC, "violet_life", proactive_event_id=event_id)

            # 2. Mark any offered upcoming user event mentioned in the sent text (one-shot wish)
            mentioned_mem_ids = [
                int(m["id"]) for m in state.get("allowed_memories") or []
                if m.get("type") == "upcoming_event" and m.get("id") is not None
                and touches_topic(sent_text, m.get("content") or "")
            ]
            self.tracker.record_eve_topic(
                (event_context or {}).get("summary_topic") or "proactive",
                mentioned_mem_ids,
                proactive_event_id=event_id,
            )

            # 3. Mark mentioned web findings & schedule moments
            offered_web_ids = [int(f["id"]) for f in state.get("allowed_web_findings") or [] if f.get("id") is not None]
            note_web_mentions(sent_text, used_finding_ids=offered_web_ids)
            offered_schedule = state.get("allowed_schedule_moments") or []
            if offered_schedule:
                from ai.schedule.awareness import record_mentions
                record_mentions(violetdb, offered_schedule, None, sent_text, event_id)

            self._set_next_proactive_at(self._backoff_next_time(now_china()))
        except Exception:
            logger.exception("could not record post-send tracking for event %s", event_id)

    # ------------------------------------------------------------------
    # Timetable awareness (VIOLET_SCHEDULE_ENABLED + VIOLET_SCHEDULE_PROACTIVE)
    # ------------------------------------------------------------------
    def _schedule(self):
        from ai.schedule import awareness
        if not awareness.enabled():
            return None
        from ai.schedule.store import ScheduleStore
        return awareness.Awareness(ScheduleStore(violetdb))

    def _schedule_gate(self, current: datetime):
        aware = self._schedule()
        if aware is None:
            return None
        gate = aware.gate(current)
        if gate.blocked:
            logger.info("[schedule-gate] %s until %s", gate.code, gate.until.strftime("%H:%M"))
        return gate

    def _postpone_past(self, gate) -> None:
        later = self._random_next_time(gate.until)
        current_next = self._get_next_proactive_at()
        if current_next is None or current_next < later:
            self._set_next_proactive_at(later)

    def _schedule_moments(self, current: datetime) -> list[dict[str, Any]]:
        aware = self._schedule()
        if aware is None:
            return []
        avoid = [item["value"] for item in self.profile.preferences.effective().get("avoid_topic", [])]
        from ai.context_engine import PLAN_ANSWERED_KINDS, USER_STATED_SOURCES
        stated = any(s.get("source") in USER_STATED_SOURCES for s in self.profile.situations.current(list(PLAN_ANSWERED_KINDS)))
        return aware.moments(
            current, violetdb, avoid_topics=avoid,
            unanswered=bool(self._unanswered_proactive(current)), user_stated_now=stated,
        )

    def _final_check(self, current: datetime) -> str | None:
        if self._in_quiet_hours(current):
            return "quiet hours"
        pref = self.profile.preferences.effective().get("proactive_chat") or {}
        if pref.get("origin") == "explicit" and pref.get("value") == "no":
            return "the user asked Violet not to start conversations"
        busy = self._busy_situation()
        if busy is not None:
            return f"the user said they are busy ({busy['value']})"
        gate = self._schedule_gate(current)
        if gate is not None and gate.blocked:
            self._postpone_past(gate)
            return gate.reason()
        limit = int(self.config.get("daily_message_limit", 3))
        if pref.get("origin") == "explicit" and pref.get("value") == "less":
            limit = min(limit, 1)
        if self._today_sent_count(current) >= limit:
            return "daily proactive limit reached"
        return None

    # ------------------------------------------------------------------
    # Phone snapshot & wake moments
    # ------------------------------------------------------------------
    @staticmethod
    def _poll_phone_events() -> None:
        try:
            from ai import phone_events
            phone_events.poll(now_china())
        except Exception:                                   # noqa: BLE001
            logger.exception("[ProactiveScheduler] phone events unavailable")

    @staticmethod
    def _phone_moment(current: datetime) -> str | None:
        try:
            from ai import phone_events
            return phone_events.pending_moment(current) if phone_events.enabled() else None
        except Exception:                                   # noqa: BLE001
            return None

    @staticmethod
    def _add_phone(event_context: dict[str, Any]) -> None:
        try:
            from ai import phone_state
            text = phone_state.line("proactive")
        except Exception:                                   # noqa: BLE001
            logger.exception("[ProactiveScheduler] phone state unavailable")
            text = ""
        if text:
            event_context["phone_now"] = text

    # ------------------------------------------------------------------
    # Holiday greeting & explicit commitments
    # ------------------------------------------------------------------
    def _holiday_greeting(self, current: datetime) -> bool:
        from ai import holidays, test_mode
        if not holidays.enabled() or test_mode.active() or self._in_quiet_hours(current):
            return False
        try:
            from ai.channels.context import current as current_channel
            channel = current_channel().channel
        except Exception:                                   # noqa: BLE001
            channel = "wechat"
        found, rel = holidays.today_for_channel(violetdb, current.date(), channel)
        if not found:
            return False
        key = "holiday_greeted:" + current.date().isoformat()
        state = violetdb.get_proactive_setting(key) or ""
        if state in ("sent", "chat") or state.startswith("gave_up"):
            return False
        last_user = parse_iso(self._latest_user_message_at())
        if last_user is not None and last_user.date() == current.date():
            violetdb.set_proactive_setting(key, "chat")
            return False
        pref = self.profile.preferences.effective().get("proactive_chat") or {}
        if pref.get("origin") == "explicit" and pref.get("value") == "no":
            return False
        if self._busy_situation() is not None:
            return False
        gate = self._schedule_gate(current)
        if gate is not None and gate.blocked:
            return False
        names = "、".join(h.name for h in found)
        hints = "；".join(f"{h.name}：{h.hint}" for h in found if h.hint)
        note = (f"【节日】今天是{names}。你们的关系：{rel.get('label') or '亲近'}。" + (hints + "。" if hints else "")
                + "现在主动给主人发节日祝福：用你平常的语气，真心、具体、短一点（一两条消息），像你自己想到的，"
                  "不要群发祝福语那种套话，也不要提“系统”“提醒”。")
        decision = {"action_type": "holiday_greeting", "reason": f"今天是{names}",
                    "chat_brief": {"topic": f"{names}的祝福", "must_say": []}}
        from ai.channels.outbound import chat_payload
        try:
            result = self.chat_agent.run("[holiday]", call_context={"is_proactive": True, "proactive_result": decision,
                                                                    "commitment": None, "proactive_note": note})
            payload = chat_payload(str(result.get("content") or ""), images=False)
        except Exception:                                   # noqa: BLE001
            logger.exception("[holiday] greeting could not be written")
            payload = {}
        attempts = int(state.split(":", 1)[1]) if state.startswith("tried:") else 0
        if not payload.get("should_send") or not payload.get("messages"):
            attempts += 1
            violetdb.set_proactive_setting(key, "gave_up" if attempts >= self.HOLIDAY_ATTEMPTS else f"tried:{attempts}")
            return False
        event_id = violetdb.add_proactive_event(event_type="holiday_greeting", status="planned", action_type="chat",
                                                reason=f"今天是{names}", decision=decision,
                                                metadata={"holidays": [h.key for h in found], "channel": channel})
        try:
            report = self.send_messages(payload["messages"], event_id)
        except Exception as exc:                            # noqa: BLE001
            logger.exception("[holiday] send crashed")
            report = {"sent_messages": [], "error": f"{exc.__class__.__name__}: {exc}"}
        if report is None:
            report = {"sent_messages": payload["messages"]}
        sent = [{k: v for k, v in m.items() if not str(k).startswith("_")} if isinstance(m, dict) else m
                for m in report.get("sent_messages") or []]
        if sent:
            violetdb.update_proactive_event(event_id, status="sent", sent_messages=sent)
            violetdb.set_proactive_setting(key, "sent")
            logger.info("[holiday] greeting sent for %s", names)
            return True
        attempts += 1
        violetdb.update_proactive_event(event_id, status="failed", reason=str(report.get("error") or "not sent")[:500])
        violetdb.set_proactive_setting(key, "gave_up" if attempts >= self.HOLIDAY_ATTEMPTS else f"tried:{attempts}")
        return False

    def _deliver_due_commitments(self, current: datetime) -> int:
        cfg = self.config.get("commitments") or {}
        if not cfg.get("enabled", True):
            return 0
        due = violetdb.list_due_commitments(current.isoformat())
        if not due:
            return 0
        delivered = 0
        for commitment in due:
            if not violetdb.claim_commitment(commitment["id"]):
                continue
            commitment = violetdb.get_commitment(commitment["id"])
            event_id = violetdb.add_proactive_event(
                event_type="commitment_reminder",
                status="planned",
                action_type="chat",
                reason=f"commitment #{commitment['id']} due",
                decision={"commitment_id": commitment["id"], "content": commitment["content"]},
                metadata={"commitment": commitment},
                scheduled_at=commitment["due_at"],
            )
            result = run_commitment_pipeline(commitment, self.chat_agent)
            try:
                report = self.send_messages(result["messages"], event_id)
            except Exception as exc:
                logger.exception("[commitments] send crashed for #%s", commitment["id"])
                report = {"sent_messages": [], "error": f"{exc.__class__.__name__}: {exc}"}
            if report is None:
                report = {"sent_messages": result["messages"]}
            sent = report.get("sent_messages") or []
            if sent:
                violetdb.update_commitment(commitment["id"], status="sent", delivered_at=violetdb.now(),
                                           proactive_event_id=event_id, last_error=result.get("error"))
                violetdb.update_proactive_event(event_id, status="sent", sent_messages=sent,
                                                reason="fallback template used" if result.get("used_fallback") else None)
                self.tracker.record_eve_topic(f"commitment:{commitment['id']}", (), proactive_event_id=event_id,
                                              source="violet_commitment")
                delivered += 1
                continue
            error = str(report.get("error") or "not sent")[:500]
            max_attempts = int(cfg.get("max_attempts", 3))
            if commitment["attempts"] >= max_attempts:
                violetdb.update_commitment(commitment["id"], status="failed", proactive_event_id=event_id, last_error=error)
            else:
                retry_at = (current + timedelta(minutes=float(cfg.get("retry_minutes", 5)))).isoformat()
                violetdb.update_commitment(commitment["id"], status="pending", next_attempt_at=retry_at,
                                           proactive_event_id=event_id, last_error=error)
            violetdb.update_proactive_event(event_id, status="failed", sent_messages=[], reason=error)
        return delivered
