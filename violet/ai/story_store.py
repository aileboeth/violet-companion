"""Persistent storage, configuration, spoiler-safe view, and prompt rendering for Violet's Story Arc Engine.

Each persona scope (ai/memory_scope.py) has its own isolated ``story_arcs`` table and at most one active arc at a
time (status in seeded / active / climax). Resolved arcs condense into short shared memories ("你们一起渡过的事").
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any

from ai.memory_model import parse_time
from ai.settings import env_int, env_str

logger = logging.getLogger("violet.story")

ACTIVE_STATUSES = ("seeded", "active", "climax")
ALL_STATUSES = ("seeded", "active", "climax", "resolved", "abandoned")
MODES = ("off", "light", "drama")
PACES = ("slow", "normal", "fast")

# Headers >= 7 chars so outbound.internal_headers() automatically prevents leaking them to replies.
STORY_SEEDED_HEADER = "【你心里藏着的事（别直接说破）】"
STORY_ACTIVE_HEADER = "【最近正在经历的事（故事线）】"
STORY_MEMORIES_HEADER = "【你们一起渡过的事（聊到再提）】"
TURN_DECISION_HEADER = "【本轮导演决断（只供执行，绝不照念）】"

ACT_STAGES = ("seeded", "active", "climax", "resolved")

STATUS_LABELS = {
    "seeded": "暗线埋伏笔",
    "active": "展开中",
    "climax": "关键转折",
    "resolved": "圆满收束",
    "abandoned": "已淡出",
}

STORY_HIDDEN_TITLE = "未揭晓的心事（待聊天触发）"

PACE_CONFIG: dict[str, dict[str, Any]] = {
    "slow": {
        "min_turns_to_start": 4,
        "cooldown_hours_between_arcs": 48.0,
        "advance_hours": 24.0,
        "chat_turns_to_advance": 3,
        "bypass_stage_gate": False,
        "max_calls_per_day": 4,
    },
    "normal": {
        "min_turns_to_start": 2,
        "cooldown_hours_between_arcs": 12.0,
        "advance_hours": 12.0,
        "chat_turns_to_advance": 2,
        "bypass_stage_gate": False,
        "max_calls_per_day": 8,
    },
    "fast": {
        "min_turns_to_start": 1,
        "cooldown_hours_between_arcs": 0.0,
        "advance_hours": 0.0,
        "chat_turns_to_advance": 2,
        "bypass_stage_gate": True,
        "max_calls_per_day": 20,
    },
}

_MODE_RANK = {"off": 0, "light": 1, "drama": 2}


def configured_mode() -> str:
    raw = (env_str("VIOLET_STORY_MODE", "drama") or "drama").strip().lower()
    return raw if raw in MODES else "drama"


def story_pace() -> str:
    raw = (env_str("VIOLET_STORY_PACE", "slow") or "slow").strip().lower()
    return raw if raw in PACES else "slow"


def story_preference() -> str:
    return (env_str("VIOLET_STORY_PREFERENCE", "") or "").strip()[:200]


def pace_params(pace: str | None = None) -> dict[str, Any]:
    p = pace if pace in PACE_CONFIG else story_pace()
    cfg = dict(PACE_CONFIG[p])
    override = env_int("VIOLET_STORY_MAX_CALLS_PER_DAY", 0)
    if override > 0:
        cfg["max_calls_per_day"] = override
    return cfg


def effective_mode(core=None) -> str:
    """Effective story mode after applying persona capability & ceiling (e.g. AI/blank off, Violet capped at light)."""
    mode = configured_mode()
    if mode == "off":
        return "off"
    if core is None:
        try:
            from ai.persona import PersonaCore
            core = PersonaCore.load()
        except Exception:                                   # noqa: BLE001
            return mode
    if getattr(core, "is_ai", False) or getattr(core, "is_blank", False):
        return "off"
    ceiling = getattr(core, "story_ceiling", "") or ""
    if ceiling in _MODE_RANK and _MODE_RANK[mode] > _MODE_RANK[ceiling]:
        return ceiling
    return mode


def _parse_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if not raw:
        return []
    try:
        parsed = json.loads(str(raw))
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (TypeError, ValueError):
        pass
    return []


def _parse_outline(raw: Any) -> dict[str, Any]:
    data = raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return {}
    if not isinstance(data, dict):
        return {}
    return normalize_outline(data)


def normalize_outline(raw: Any, *, current_act: int = 1, arc_status: str = "seeded") -> dict[str, Any]:
    """Validate and normalize a multi-act story outline dict."""
    if not isinstance(raw, dict):
        return {}
    cast_in = raw.get("premise_cast")
    cast: list[str] = []
    if isinstance(cast_in, list):
        for item in cast_in[:8]:
            s = str(item or "").strip()[:50]
            if s and s not in cast:
                cast.append(s)
    acts_in = raw.get("acts")
    acts: list[dict[str, Any]] = []
    if isinstance(acts_in, list):
        for idx, item in enumerate(acts_in[:6], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()[:36]
            summary = str(item.get("summary") or "").strip()[:140]
            if not (title or summary):
                continue
            stage = str(item.get("stage") or "").strip().lower()
            if stage not in ACT_STAGES:
                stage = "seeded" if idx == 1 else ("resolved" if idx == len(acts_in) else "active")
            acts.append({
                "act": idx,
                "stage": stage,
                "title": title or f"第{idx}幕",
                "summary": summary,
                "reveal_boundary": str(item.get("reveal_boundary") or "").strip()[:90],
                "trigger": str(item.get("trigger") or "").strip()[:80],
                "status": str(item.get("status") or "pending").strip().lower(),
                "revised": bool(item.get("revised")),
            })
    if not acts and not cast:
        return {}
    out = {"premise_cast": cast, "acts": acts}
    if acts:
        sync_outline_statuses(out, current_act=current_act, arc_status=arc_status)
    return out


def sync_outline_statuses(outline: dict[str, Any], *, current_act: int, arc_status: str) -> dict[str, Any]:
    acts = outline.get("acts")
    if not isinstance(acts, list) or not acts:
        return outline
    total = len(acts)
    cur = max(1, min(int(current_act or 1), total))
    for idx, act in enumerate(acts, start=1):
        act["act"] = idx
        if arc_status == "resolved":
            act["status"] = "done"
        elif idx < cur:
            act["status"] = "done"
        elif idx == cur:
            act["status"] = "current"
        else:
            act["status"] = "pending"
    return outline


def current_act_entry(arc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not arc:
        return None
    acts = (arc.get("outline") or {}).get("acts") or []
    if not acts:
        return None
    idx = max(1, min(int(arc.get("current_act") or 1), len(acts))) - 1
    return acts[idx]


def _hydrate(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    out = dict(row)
    out["canon_beats"] = _parse_list(out.get("canon_beats"))
    out["revealed_beats"] = _parse_list(out.get("revealed_beats"))
    out["current_act"] = max(1, int(out.get("current_act") or 1))
    out["next_turn_brief"] = str(out.get("next_turn_brief") or "").strip()
    outline = _parse_outline(out.get("outline"))
    if outline.get("acts"):
        sync_outline_statuses(outline, current_act=out["current_act"], arc_status=str(out.get("status") or "seeded"))
    out["outline"] = outline
    return out


class StoryStore:
    def __init__(self, db):
        self.db = db

    def active_arc(self) -> dict[str, Any] | None:
        row = self.db.fetch_one(
            "SELECT * FROM story_arcs WHERE status IN ('seeded', 'active', 'climax') ORDER BY id DESC LIMIT 1"
        )
        return _hydrate(row)

    def get(self, arc_id: int) -> dict[str, Any] | None:
        row = self.db.fetch_one("SELECT * FROM story_arcs WHERE id = ?", (int(arc_id),))
        return _hydrate(row)

    def recent_resolved(self, limit: int = 3) -> list[dict[str, Any]]:
        rows = self.db.fetch_all(
            "SELECT * FROM story_arcs WHERE status = 'resolved' AND resolved_summary IS NOT NULL "
            "AND trim(resolved_summary) != '' ORDER BY id DESC LIMIT ?",
            (max(1, int(limit)),),
        )
        return [_hydrate(r) for r in reversed(rows) if r]

    def list_arcs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.fetch_all("SELECT * FROM story_arcs ORDER BY id DESC LIMIT ?", (max(1, int(limit)),))
        return [_hydrate(r) for r in rows if r]

    def latest_finished(self) -> dict[str, Any] | None:
        row = self.db.fetch_one(
            "SELECT * FROM story_arcs WHERE status IN ('resolved', 'abandoned') ORDER BY id DESC LIMIT 1"
        )
        return _hydrate(row)

    def used_seed_keys(self) -> set[str]:
        rows = self.db.fetch_all("SELECT arc_key FROM story_arcs WHERE source = 'seed'")
        out: set[str] = set()
        for r in rows:
            key = str(r.get("arc_key") or "")
            if key.startswith("seed:"):
                out.add(key[5:])
            elif key:
                out.add(key)
        return out

    def create_arc(
        self,
        *,
        arc_key: str,
        source: str,
        title: str,
        tone: str,
        stage_min: int,
        premise: str,
        he_direction: str,
        open_hook: str,
        status: str = "seeded",
        outline: dict[str, Any] | None = None,
        current_act: int = 1,
        next_turn_brief: str = "",
        canon_beats: list[str] | None = None,
        revealed_beats: list[str] | None = None,
        now: str | None = None,
    ) -> dict[str, Any]:
        ts = now or self.db.now()
        if status not in ACTIVE_STATUSES:
            status = "seeded"
        cur_act = max(1, int(current_act or 1))
        norm_outline = normalize_outline(outline or {}, current_act=cur_act, arc_status=status)
        cur = self.db.execute(
            """
            INSERT INTO story_arcs (
                arc_key, source, title, tone, stage_min, status, premise, he_direction,
                outline, current_act, next_turn_brief,
                canon_beats, revealed_beats, open_hook, user_role,
                created_at, last_advanced_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?)
            """,
            (
                str(arc_key)[:80],
                "seed" if source == "seed" else "free",
                str(title).strip()[:60] or "一段正在发生的小事",
                "light" if tone == "light" else "drama",
                max(0, int(stage_min)),
                status,
                str(premise).strip()[:320],
                str(he_direction).strip()[:200],
                json.dumps(norm_outline, ensure_ascii=False),
                cur_act,
                str(next_turn_brief or "").strip()[:120],
                json.dumps((canon_beats or [])[:8], ensure_ascii=False),
                json.dumps((revealed_beats or [])[:8], ensure_ascii=False),
                str(open_hook).strip()[:160],
                ts,
                ts,
                ts,
            ),
        )
        return self.get(int(cur.lastrowid)) or {}

    def append_update(
        self,
        arc_id: int,
        *,
        canon_beat: str | None = None,
        revealed_beat: str | None = None,
        user_role: str | None = None,
        open_hook: str | None = None,
        status: str | None = None,
        outline: dict[str, Any] | None = None,
        current_act: int | None = None,
        next_turn_brief: str | None = None,
        turns_engaged_delta: int = 0,
        turns_ignored_delta: int = 0,
        reset_ignored: bool = False,
        advanced: bool = False,
        now: str | None = None,
    ) -> dict[str, Any] | None:
        arc = self.get(arc_id)
        if not arc:
            return None
        ts = now or self.db.now()
        canon = list(arc["canon_beats"])
        revealed = list(arc["revealed_beats"])
        c_txt = (canon_beat or "").strip()[:80]
        r_txt = (revealed_beat or "").strip()[:80]
        if c_txt and c_txt not in canon:
            canon = (canon + [c_txt])[-8:]
        if r_txt and r_txt not in revealed:
            revealed = (revealed + [r_txt])[-8:]
            if r_txt not in canon:
                canon = (canon + [r_txt])[-8:]

        new_status = status if status in ALL_STATUSES else arc["status"]
        new_hook = open_hook.strip()[:160] if isinstance(open_hook, str) and open_hook.strip() else arc["open_hook"]
        new_user_role = user_role.strip()[:160] if isinstance(user_role, str) and user_role.strip() else arc["user_role"]
        new_act = max(1, int(current_act)) if current_act is not None else int(arc.get("current_act") or 1)
        raw_outline = outline if outline is not None else (arc.get("outline") or {})
        norm_outline = normalize_outline(raw_outline, current_act=new_act, arc_status=new_status)
        if next_turn_brief is not None:
            new_brief = str(next_turn_brief).strip()[:120]
        else:
            new_brief = str(arc.get("next_turn_brief") or "").strip()[:120]
        engaged = max(0, int(arc.get("turns_engaged") or 0) + int(turns_engaged_delta))
        ignored = 0 if reset_ignored else max(0, int(arc.get("turns_ignored") or 0) + int(turns_ignored_delta))
        last_adv = ts if advanced else arc["last_advanced_at"]

        self.db.execute(
            """
            UPDATE story_arcs
               SET status = ?, outline = ?, current_act = ?, next_turn_brief = ?,
                   canon_beats = ?, revealed_beats = ?, open_hook = ?, user_role = ?,
                   turns_engaged = ?, turns_ignored = ?, last_advanced_at = ?, updated_at = ?
             WHERE id = ?
            """,
            (
                new_status,
                json.dumps(norm_outline, ensure_ascii=False),
                new_act,
                new_brief,
                json.dumps(canon, ensure_ascii=False),
                json.dumps(revealed, ensure_ascii=False),
                new_hook,
                new_user_role,
                engaged,
                ignored,
                last_adv,
                ts,
                int(arc_id),
            ),
        )
        return self.get(arc_id)

    def mark_proactive_shared(self, arc_id: int, step_key: str, now: str | None = None) -> None:
        ts = now or self.db.now()
        self.db.execute(
            "UPDATE story_arcs SET proactive_shared_step = ?, updated_at = ? WHERE id = ?",
            (str(step_key)[:80], ts, int(arc_id)),
        )

    def resolve_arc(
        self,
        arc_id: int,
        summary: str,
        *,
        final_beat: str | None = None,
        boost_mood: bool = True,
        now: str | None = None,
    ) -> dict[str, Any] | None:
        arc = self.get(arc_id)
        if not arc:
            return None
        ts = now or self.db.now()
        canon = list(arc["canon_beats"])
        revealed = list(arc["revealed_beats"])
        fb = (final_beat or "").strip()[:80]
        if fb:
            if fb not in canon:
                canon = (canon + [fb])[-8:]
            if fb not in revealed:
                revealed = (revealed + [fb])[-8:]
        clean_summary = (summary or "").strip()[:100] or f"和对方一起平稳渡过了「{arc['title']}」这件事"
        cool_hours = float(pace_params()["cooldown_hours_between_arcs"])
        cooldown = (parse_time(ts) + timedelta(hours=cool_hours)).isoformat() if cool_hours > 0 else ts
        outline = arc.get("outline") or {}
        acts = outline.get("acts") or []
        final_act = max(int(arc.get("current_act") or 1), len(acts)) if acts else int(arc.get("current_act") or 1)
        norm_outline = normalize_outline(outline, current_act=final_act, arc_status="resolved")
        self.db.execute(
            """
            UPDATE story_arcs
               SET status = 'resolved', outline = ?, current_act = ?, next_turn_brief = '',
                   canon_beats = ?, revealed_beats = ?,
                   resolved_summary = ?, open_hook = '', resolved_at = ?,
                   cooldown_until = ?, last_advanced_at = ?, updated_at = ?
             WHERE id = ?
            """,
            (
                json.dumps(norm_outline, ensure_ascii=False),
                final_act,
                json.dumps(canon, ensure_ascii=False),
                json.dumps(revealed, ensure_ascii=False),
                clean_summary,
                ts,
                cooldown,
                ts,
                ts,
                int(arc_id),
            ),
        )
        if boost_mood:
            try:
                from ai import mood
                if mood.enabled():
                    mood.shift(self.db, 0.35, f"和对方一起渡过了「{arc['title']}」")
            except Exception:                               # noqa: BLE001
                logger.debug("story mood boost skipped", exc_info=True)
        return self.get(arc_id)

    def abandon_arc(self, arc_id: int, reason: str = "", now: str | None = None) -> dict[str, Any] | None:
        arc = self.get(arc_id)
        if not arc:
            return None
        ts = now or self.db.now()
        cool_hours = float(pace_params()["cooldown_hours_between_arcs"])
        cooldown = (parse_time(ts) + timedelta(hours=cool_hours)).isoformat() if cool_hours > 0 else ts
        self.db.execute(
            """
            UPDATE story_arcs
               SET status = 'abandoned', open_hook = ?, cooldown_until = ?, resolved_at = ?, updated_at = ?
             WHERE id = ?
            """,
            ((reason or "").strip()[:120], cooldown, ts, ts, int(arc_id)),
        )
        return self.get(arc_id)

    # ------------------------------------------------------------------
    # Projections for UI (spoiler-safe), ContextEngine, ConsistencyGuard
    # ------------------------------------------------------------------
    @staticmethod
    def public_view(arc: dict[str, Any] | None, *, spoiler: bool = False) -> dict[str, Any] | None:
        if not arc:
            return None
        revealed = list(arc.get("revealed_beats") or [])
        status = str(arc.get("status") or "seeded")
        hide_title = (not spoiler) and status == "seeded" and not revealed
        outline = arc.get("outline") or {}
        acts = outline.get("acts") or []
        cur_act = max(1, int(arc.get("current_act") or 1))
        act_progress = {"current": min(cur_act, len(acts)), "total": len(acts)} if acts else None
        view: dict[str, Any] = {
            "id": arc["id"],
            "source": arc.get("source") or "seed",
            "tone": arc.get("tone") or "drama",
            "status": status,
            "status_label": STATUS_LABELS.get(status, status),
            "title": STORY_HIDDEN_TITLE if hide_title else arc.get("title", ""),
            "title_hidden": hide_title,
            "act_progress": act_progress,
            "revealed_beats": revealed,
            "user_role": arc.get("user_role") or "",
            "resolved_summary": arc.get("resolved_summary") or "",
            "turns_engaged": int(arc.get("turns_engaged") or 0),
            "created_at": arc.get("created_at"),
            "last_advanced_at": arc.get("last_advanced_at"),
            "resolved_at": arc.get("resolved_at"),
            "has_hidden_spoiler": bool(
                arc.get("premise") or arc.get("open_hook") or arc.get("he_direction") or acts
            ),
        }
        if spoiler:
            view["title"] = arc.get("title", "")
            view["title_hidden"] = False
            view["premise"] = arc.get("premise") or ""
            view["open_hook"] = arc.get("open_hook") or ""
            view["he_direction"] = arc.get("he_direction") or ""
            view["outline"] = outline
            view["current_act"] = cur_act
            view["next_turn_brief"] = arc.get("next_turn_brief") or ""
            view["canon_beats"] = list(arc.get("canon_beats") or [])
            view["stage_min"] = int(arc.get("stage_min") or 0)
        return view

    def context_blocks(self, core=None) -> list[tuple[str, list[str], str]]:
        """Return 0–2 compact volatile blocks: active story arc + recent resolved shared memories."""
        mode = effective_mode(core)
        blocks: list[tuple[str, list[str], str]] = []
        if mode != "off":
            arc = self.active_arc()
            if arc:
                header, lines, footer = format_active_arc_block(arc)
                if lines:
                    blocks.append((header, lines, footer))
        resolved = self.recent_resolved(limit=3)
        if resolved:
            mem_lines = [f"- {r['resolved_summary']}" for r in resolved if r.get("resolved_summary")]
            if mem_lines:
                blocks.append((STORY_MEMORIES_HEADER, mem_lines, ""))
        return blocks

    def backing_texts(self) -> list[str]:
        """All canonical story texts (active + recent resolved) recognized by ConsistencyGuard."""
        out: list[str] = []
        arc = self.active_arc()
        if arc:
            for field in ("title", "premise", "open_hook", "he_direction", "user_role", "next_turn_brief"):
                val = str(arc.get(field) or "").strip()
                if val:
                    out.append(val)
            out.extend(arc.get("canon_beats") or [])
            out.extend(arc.get("revealed_beats") or [])
            outline = arc.get("outline") or {}
            out.extend(outline.get("premise_cast") or [])
            for act in outline.get("acts") or []:
                if act.get("status") in ("done", "current"):
                    for k in ("title", "summary"):
                        v = str(act.get(k) or "").strip()
                        if v:
                            out.append(v)
        for r in self.recent_resolved(limit=8):
            for field in ("title", "resolved_summary"):
                val = str(r.get(field) or "").strip()
                if val:
                    out.append(val)
            out.extend(r.get("revealed_beats") or [])
        return out


def format_active_arc_block(arc: dict[str, Any]) -> tuple[str, list[str], str]:
    """Render one active story arc into a lean (~120-210 char) prompt block (current act ONLY, never spoiling future acts)."""
    status = str(arc.get("status") or "seeded")
    cur_act = current_act_entry(arc)
    acts = (arc.get("outline") or {}).get("acts") or []
    brief = str(arc.get("next_turn_brief") or "").strip()
    lines: list[str] = []
    if status == "seeded":
        if arc.get("open_hook"):
            lines.append(f"- 伏笔：{arc['open_hook']}")
        elif cur_act and cur_act.get("summary"):
            lines.append(f"- 伏笔：{cur_act['summary'][:90]}")
        elif arc.get("premise"):
            lines.append(f"- 伏笔：{arc['premise'][:80]}")
        if cur_act and cur_act.get("reveal_boundary"):
            lines.append(f"- 暴露边界：{cur_act['reveal_boundary'][:80]}")
        if arc.get("revealed_beats"):
            lines.append(f"- 已露出的细节：{'；'.join(arc['revealed_beats'][-2:])}")
        if brief:
            lines.append(f"- 下步参考：{brief[:80]}")
        footer = (
            "规则：这只是埋在心底的轻伏笔，日常照常聊天，不要主动一次性全倒出来，别动不动用“……”开头；"
            "只在顺口时偶露一丝心不在焉，对方敏锐追问时才慢慢松一点口；"
            "严禁声称已发出或发送照片/图片/截图/文件，用文字口头描述细节即可。"
        )
        return STORY_SEEDED_HEADER, lines, footer

    if arc.get("premise"):
        lines.append(f"- 暗线背景：{arc['premise'][:100]}")
    if cur_act and cur_act.get("summary"):
        pos = f"第{cur_act.get('act', 1)}/{len(acts)}幕" if acts else "当前阶段"
        lines.append(f"- {pos}（{cur_act.get('title', '')}）：{cur_act['summary'][:95]}")
    shown = arc.get("canon_beats") or arc.get("revealed_beats") or []
    if shown:
        lines.append(f"- 已坐实进展：{'；'.join(shown[-3:])}")
    if arc.get("user_role"):
        lines.append(f"- 对方的态度：{arc['user_role'][:80]}")
    if arc.get("open_hook"):
        tag = "关键转折" if status == "climax" else "此刻悬念"
        lines.append(f"- {tag}：{arc['open_hook'][:95]}")
    if cur_act and cur_act.get("reveal_boundary"):
        lines.append(f"- 本幕边界：{cur_act['reveal_boundary'][:80]}")
    if brief:
        lines.append(f"- 下步参考：{brief[:80]}")
    footer = (
        "规则：别强行抢戏，对方聊别的就先陪对方；聊到或对方关心时自然展开并可补充合理细节，"
        "别推翻已坐实进展也别提前跳幕剧透，严禁声称已发出或发送照片/图片/截图/文件，用文字口头描述细节即可；"
        "保持微信聊天节奏（每次1-3条短消息，别每句重复称呼对方，别动不动用“……”开头），"
        "对方选择会改变走向，最终朝温暖化解的 Happy Ending 走。"
    )
    return STORY_ACTIVE_HEADER, lines, footer
