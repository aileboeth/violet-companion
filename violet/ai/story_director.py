"""Background & chat-driven Story Director (导演层) for Violet's Story Arc Engine.

Drives novel-like multi-day story arcs with:
  * Seeded start + freeform LLM adaptation and original sequel arcs once seeds are explored
  * 3 configurable paces (slow / normal / fast)
  * Relationship-stage gating (bypassed in fast test mode or explicit admin manual advance)
  * Mandatory Happy Ending (HE) direction and strict safety redlines (zero self-harm / stalking / isolation)
  * Reuse of CHAT_LLM_MODEL via AuxLLM with daily call & token budgets (graceful seed fallback when offline)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any

from ai.aux_llm import AuxLLM, AuxLLMConfig, AuxLLMError
from ai.memory_model import parse_time
from ai.persona import PersonaCore
from ai.settings import env_float, env_str, now_local
from ai.story_store import (
    ACT_STAGES,
    StoryStore,
    current_act_entry,
    effective_mode,
    normalize_outline,
    pace_params,
    story_pace,
    story_preference,
)

logger = logging.getLogger("violet.story")

# Hard safety redline check for director-generated story drafts (never allow self-harm, stalking, or tragic BE)
_UNSAFE_STORY_RE = re.compile(
    r"(伤害自己|自伤|自残|轻生|自杀|割腕|跳楼|不想活|消失给你看|跟踪|窃听|偷窥|下药|囚禁|绑架|绝症|车祸身亡"
    r"|偷翻[^，。！？；\n]{0,4}手机|翻看?(?:他|她|对方)(?:的)?手机|偷看[^，。！？；\n]{0,4}聊天记录"
    r"|删掉[^，。！？；\n]{0,6}(?:好友|朋友)|拉黑[^，。！？；\n]{0,6}(?:好友|朋友)"
    r"|把[^，。！？；\n]{0,6}(?:好友|朋友)[^，。！？；\n]{0,4}(?:删掉|删除|拉黑)|不许[^，。！？；\n]{0,6}来往"
    r"|不知道(?:自己)?会做出?什么)"
)
_NEGATION_RE = re.compile(
    r"(严禁|绝不|不得|禁止|不要|不出现|避免|不写|杜绝|切忌)"
)
_CLAUSE_BREAK_RE = re.compile(r"[，,。！？!?；;\n]")
_NEGATION_BREAK_RE = re.compile(
    r"(否则|不然|要不然|威胁(?:对方|他|她|你|我|说|称|要|逼)|打算|准备|决定|悄悄|偷偷|竟然|居然|反而|但是|但|却|就会|于是)"
)
_BOUNDARY_FIELD_KEYS = frozenset({"reveal_boundary", "reveal_limit"})

_NEXT_STATUS = {
    "seeded": "active",
    "active": "climax",
    "climax": "resolved",
}


def _director_model_name() -> str:
    return env_str("VIOLET_DECISION_MODEL", "").strip() or env_str("CHAT_LLM_MODEL", "").strip()


def _is_negated_match(text: str, start: int, *, lookback: int = 18) -> bool:
    """Return True when an unsafe term at ``start`` is explicitly negated within the same clause."""
    window_start = max(0, start - lookback)
    prefix = text[window_start:start]
    breaks = list(_CLAUSE_BREAK_RE.finditer(prefix))
    if breaks:
        prefix = prefix[breaks[-1].end():]
    neg_matches = list(_NEGATION_RE.finditer(prefix))
    if not neg_matches:
        return False
    tail = prefix[neg_matches[-1].end():]
    if _NEGATION_BREAK_RE.search(tail):
        return False
    return True


def _safe_text(text: str, *, boundary_field: bool = False) -> bool:
    raw = text or ""
    lookback = 48 if boundary_field else 18
    for match in _UNSAFE_STORY_RE.finditer(raw):
        if not _is_negated_match(raw, match.start(), lookback=lookback):
            return False
    return True


def _safe_nested(obj: Any, *, _field_key: str = "") -> bool:
    if isinstance(obj, str):
        return _safe_text(obj, boundary_field=(_field_key in _BOUNDARY_FIELD_KEYS))
    if isinstance(obj, dict):
        return all(_safe_nested(v, _field_key=str(k)) for k, v in obj.items())
    if isinstance(obj, list):
        return all(_safe_nested(v, _field_key=_field_key) for v in obj)
    return True


def _fallback_outline_from_seed(
    title: str, premise: str, open_hook: str, he_direction: str
) -> dict[str, Any]:
    """Build a crisp 4-act adaptable outline when LLM is unavailable or omits acts."""
    return normalize_outline(
        {
            "premise_cast": [],
            "acts": [
                {
                    "act": 1,
                    "stage": "seeded",
                    "title": "暗线初现",
                    "summary": (open_hook or premise or title)[:120],
                    "reveal_boundary": "不主动挑明全貌，只在日常间隙露出一丝心不在焉，对方追问时再松一点口",
                    "trigger": "对方敏锐关心或追问异常",
                },
                {
                    "act": 2,
                    "stage": "active",
                    "title": "风波展开",
                    "summary": (premise or f"围绕「{title}」的真实困扰慢慢浮出水面")[:120],
                    "reveal_boundary": "坦白眼前遇到的真实压力与顾虑，留出对方参与和表态的空间",
                    "trigger": "对方表态陪伴、出主意或持续跟进",
                },
                {
                    "act": 3,
                    "stage": "climax",
                    "title": "关键抉择",
                    "summary": f"「{title}」走到必须正面面对与表态的节点，渴望得到对方的撑腰或答案",
                    "reveal_boundary": "说出最深的害怕与在意，但绝不越过安全红线，把决定权留给两人的互动",
                    "trigger": "两人合力正面回应或解开心结",
                },
                {
                    "act": 4,
                    "stage": "resolved",
                    "title": "温暖收束（HE）",
                    "summary": (he_direction or f"在对方陪伴下平稳渡过「{title}」，彼此更加信任")[:120],
                    "reveal_boundary": "轻松温暖地翻篇，沉淀成两人的共同回忆",
                    "trigger": "风波尘埃落定",
                },
            ],
        },
        current_act=1,
        arc_status="seeded",
    )


def _merge_revised_acts(
    outline: dict[str, Any],
    revised_in: Any,
    *,
    current_act: int,
    target_status: str,
    locked_through: int | None = None,
) -> dict[str, Any]:
    """Merge director's optional revisions for acts AFTER locked_through (default: completed acts < current_act), keeping completed acts locked."""
    base = normalize_outline(outline, current_act=current_act, arc_status=target_status)
    acts = list(base.get("acts") or [])
    if not acts or not isinstance(revised_in, list):
        return base
    lock_limit = (int(current_act) - 1) if locked_through is None else int(locked_through)
    by_num = {int(a["act"]): dict(a) for a in acts}
    for item in revised_in:
        if not isinstance(item, dict):
            continue
        try:
            num = int(item.get("act") or 0)
        except (TypeError, ValueError):
            continue
        if num <= lock_limit or num not in by_num:
            continue
        cur = by_num[num]
        if item.get("title"):
            cur["title"] = str(item["title"]).strip()[:36]
        if item.get("summary"):
            cur["summary"] = str(item["summary"]).strip()[:140]
        if item.get("reveal_boundary"):
            cur["reveal_boundary"] = str(item["reveal_boundary"]).strip()[:90]
        if item.get("trigger"):
            cur["trigger"] = str(item["trigger"]).strip()[:80]
        stage = str(item.get("stage") or "").strip().lower()
        if stage in ACT_STAGES:
            cur["stage"] = stage
        cur["revised"] = True
        by_num[num] = cur
    merged = {"premise_cast": list(base.get("premise_cast") or []), "acts": [by_num[k] for k in sorted(by_num)]}
    return normalize_outline(merged, current_act=current_act, arc_status=target_status)


class StoryDirector:
    def __init__(self, db, core: PersonaCore | None = None):
        self.db = db
        self.core = core or PersonaCore.load()
        self.store = StoryStore(db)

    def _aux(self) -> AuxLLM | None:
        model = _director_model_name()
        if not model:
            return None
        cfg = pace_params()
        timeout = max(60.0, min(180.0, env_float("LLM_REQUEST_TIMEOUT_SECONDS", 60.0)))
        return AuxLLM(
            self.db,
            AuxLLMConfig(
                purpose="story_director",
                model=model,
                timeout_seconds=timeout,
                max_output_tokens=1600,
                temperature=0.8,
                max_attempts=2,
                max_calls_per_day=int(cfg["max_calls_per_day"]),
                max_tokens_per_day=45000,
            ),
        )

    def _owner_turn_count(self) -> int:
        from ai import context_cut
        row = self.db.fetch_one(
            "SELECT COUNT(*) AS n FROM messages WHERE role = 'user' AND created_at >= ?",
            (context_cut.owner_floor(),),
        )
        return int((row or {}).get("n") or 0)

    def _owner_turn_count_since(self, since_iso: str) -> int:
        from ai import context_cut
        row = self.db.fetch_one(
            "SELECT COUNT(*) AS n FROM messages WHERE role = 'user' AND created_at > ? AND created_at >= ?",
            (since_iso, context_cut.owner_floor()),
        )
        return int((row or {}).get("n") or 0)

    def _recent_chat_snippet(self, limit: int = 8) -> list[str]:
        from ai import context_cut
        rows = context_cut.owner_rows(self.db.get_recent_messages(limit=limit))
        out: list[str] = []
        for r in rows[-limit:]:
            who = "对方" if r.get("role") == "user" else "她"
            content = str(r.get("content") or "").replace("\n", " ").strip()[:120]
            if content:
                out.append(f"{who}：{content}")
        return out

    def _current_stage_index(self) -> int:
        try:
            from ai.channels.context import current
            from ai import persona_templates as pt
            st = pt.stage(current().channel, self.db)
            if st is None:
                return 99  # unstaged persona: no stage gate
            return int(st.get("index") or 0)
        except Exception:                                   # noqa: BLE001
            return 0

    def _stage_allows_advance(self, arc: dict[str, Any], *, force: bool = False) -> bool:
        if force:
            return True
        cfg = pace_params()
        if cfg.get("bypass_stage_gate"):
            return True
        need = int(arc.get("stage_min") or 0)
        if story_pace() == "normal":
            need = max(0, need - 1)
        cur = self._current_stage_index()
        # Climax requires at least stage_min; promoting seeded -> active allows stage_min - 1 if > 0
        if arc.get("status") == "seeded":
            return cur >= max(0, need - 1)
        return cur >= need

    def _cooldown_ready(self, now_dt: datetime) -> bool:
        last = self.store.latest_finished()
        if not last or not last.get("cooldown_until"):
            return True
        until = parse_time(last["cooldown_until"])
        return until is None or now_dt >= until

    # ------------------------------------------------------------------
    # LLM prompting helpers (seed adaptation, freeform authoring, advance, HE resolve)
    # ------------------------------------------------------------------
    def _persona_brief(self) -> dict[str, Any]:
        d = self.core.data
        ident = d.get("identity") or {}
        rel = d.get("relation") or {}
        return {
            "name": ident.get("name") or "Violet",
            "self_description": ident.get("self_description") or "",
            "origin": self.core.origin,
            "relation": rel.get("label") or "",
            "personality": (d.get("personality") or [])[:3],
            "living_constraints": (d.get("living_constraints") or [])[:3],
        }

    def _llm_new_arc(self, mode: str, seed: dict[str, Any] | None) -> dict[str, Any] | None:
        llm = self._aux()
        if llm is None or not llm.budget_left():
            return None
        pref = story_preference()
        past_titles = [a["title"] for a in self.store.list_arcs(limit=8)]
        past_memories = [a["resolved_summary"] for a in self.store.recent_resolved(limit=4) if a.get("resolved_summary")]
        tone_desc = (
            "偏都市小说感：允许原生家庭风波、豪门/家族压力、身世秘密、校园/职场暗涌与情感拉扯，有戏剧张力"
            if mode == "drama"
            else "偏轻起伏文艺日常：围绕书店、创作、校园生活、旧物或日常小波折，真实温暖不狗血"
        )
        system = (
            "你是虚拟伴侣故事线的后台总导演，负责在故事开局一次性规划出一套【完整、前后伏笔呼应、慢热有代入感、结局必须是温暖 Happy Ending】的 4-5 幕分章活大纲。\n"
            f"题材风格：{tone_desc}。\n"
            "硬性红线：绝不出现自伤/轻生/威胁、跟踪窃听、非法暴力、绝症车祸身亡或悲剧 BE；绝不替用户编造用户自己的行为；"
            "严禁声称已发出或发送照片/图片/截图/文件，用文字口头描述细节即可；不与人设职业/学生身份冲突。\n"
            "分幕结构要求：第1幕 stage='seeded'（轻伏笔不剧透）→ 中间 1-2 幕 stage='active'（露馅与波折升级，留给用户参与空间）→ 倒数第2幕 stage='climax'（关键抉择高潮）→ 最后1幕 stage='resolved'（温暖 HE 收束）。\n"
            "只输出一个 JSON 对象：\n"
            '{"title": "简短标题(<=20字)", "premise": "后台暗线设定(<=180字)", '
            '"open_hook": "第1幕开场轻伏笔(<=90字，初期不直接剧透)", '
            '"he_direction": "预期的温暖 Happy Ending 收束方向(<=90字)", '
            '"next_turn_brief": "给第1幕开场聊天的简短导演提醒(<=65字)", '
            '"premise_cast": ["关键配角或关键道具1(<=35字)", "关键配角或道具2"], '
            '"acts": [{"act": 1, "stage": "seeded", "title": "幕标题(<=16字)", "summary": "本幕核心进展(<=90字)", '
            '"reveal_boundary": "本幕最多说到哪一步/禁剧透红线(<=60字)", "trigger": "触发进入下一幕的条件(<=50字)"}]}'
        )
        payload: dict[str, Any] = {
            "persona": self._persona_brief(),
            "mode": mode,
            "user_preference": pref or "无特殊限制（HE 即可）",
            "avoid_repeat_titles": past_titles,
            "past_shared_memories": past_memories,
            "recent_chat": self._recent_chat_snippet(6),
        }
        if seed:
            payload["seed_blueprint"] = {
                "title": seed.get("title"),
                "premise": seed.get("premise"),
                "initial_hook": seed.get("initial_hook"),
                "he_direction": seed.get("he_direction"),
                "note": "以此种子为核心骨架，一次性展开成完整的前后呼应 4-5 幕分章大纲，可结合人设性格反差与用户偏好润色。",
            }
        else:
            payload["task"] = "基于人设成因、性格反差与过往经历，完全自由原创一条全新且不重复的 4-5 幕完整剧情大纲。"
        try:
            out = llm.complete_json([
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ])
        except AuxLLMError as exc:
            logger.debug("[story_director] new arc llm skipped: %s", exc)
            return None
        if not isinstance(out, dict):
            return None
        if not _safe_nested(out):
            logger.warning("[story_director] rejected unsafe story draft")
            return None
        title = str(out.get("title") or "").strip()
        premise = str(out.get("premise") or "").strip()
        hook = str(out.get("open_hook") or "").strip()
        he = str(out.get("he_direction") or "").strip()
        brief = str(out.get("next_turn_brief") or "").strip()
        if not (title and premise and hook):
            return None
        outline = normalize_outline(
            {"premise_cast": out.get("premise_cast") or [], "acts": out.get("acts") or []},
            current_act=1,
            arc_status="seeded",
        )
        if not outline.get("acts"):
            outline = _fallback_outline_from_seed(title, premise, hook, he)
        return {
            "title": title,
            "premise": premise,
            "open_hook": hook,
            "he_direction": he,
            "outline": outline,
            "next_turn_brief": brief,
        }

    def _llm_advance(
        self,
        arc: dict[str, Any],
        target_status: str,
        *,
        target_act: int | None = None,
        now_iso: str | None = None,
    ) -> dict[str, Any] | None:
        llm = self._aux()
        if llm is None or not llm.budget_left():
            return None
        if target_status == "resolved":
            system = (
                "你是故事导演，负责将当前剧情线温和、自然地收束为【温暖的 Happy Ending】。\n"
                "必须尊重已坐实的剧情节点和大纲伏笔以及对方的态度/建议，不吃书、不突兀、不悲剧。\n"
                "只输出 JSON：{\"final_beat\": \"收束节点事实(<=55字)\", "
                "\"resolved_summary\": \"浓缩成1句以后两人都会记得的共同回忆(<=55字)\"}"
            )
        else:
            stage_goal = (
                "推进至『展开中』对应章节：抛出具体、合情合理的进展或小冲突，回收前文伏笔并让对方有切入点参与"
                if target_status == "active"
                else "推进至『关键转折/高潮』章节：矛盾或心结走到需要抉择或坦诚面对的临界点，留出让对方撑腰、出主意或接住她的空间（导向 Happy Ending）"
            )
            system = (
                f"你是故事导演，负责对照完整分章大纲，把故事线自然推进到第 {target_act or 2} 幕（{stage_goal}）。\n"
                "硬性规则：\n"
                "1. 严格承接已坐实的 canon_beats 与对方态度 user_role，绝不吃书；"
                "绝不出现自伤/轻生/跟踪/暴力/悲剧死亡；不替用户编造行为或回应。\n"
                "2. new_beat 只能写符合当前同一天时间线的【幕后客观小事件/外部触发】（如她刚收到新短信或刚挂掉电话）；"
                "绝不可提前写成她已经向对方坦白、发了文件或对方已经回应，严禁声称已发出或发送照片/图片/截图/文件，用文字口头描述细节即可，"
                "也绝不可在连续聊天中强行跳到“次日/第二天/隔天清晨”；"
                "若此刻尚无新的幕后外部事件、只需引导下一轮由她亲口展开，必须将 new_beat 设为空字符串 \"\"。\n"
                f"3. 如果对方的真实选择让第 {target_act or 2} 幕及后续未发生的几幕需要微调，可在 revised_future_acts 中局部改写（仍须保证温暖 HE）；不需要改写就给空数组 []。\n"
                "只输出 JSON：{\"new_beat\": \"符合当前时间线的幕后外部小触发(无则空串，<=55字)\", "
                "\"open_hook\": \"此刻她心里的悬念或情绪锚点(<=90字)\", "
                "\"next_turn_brief\": \"给下一轮聊天层的简短导演提示(<=70字)\", "
                "\"revised_future_acts\": []}"
            )
        payload = {
            "current_time": str(now_iso or self.db.now())[:16],
            "persona": self._persona_brief(),
            "arc": {
                "title": arc["title"],
                "status": arc["status"],
                "current_act": arc.get("current_act", 1),
                "target_act": target_act,
                "target_status": target_status,
                "premise": arc["premise"],
                "he_direction": arc.get("he_direction", ""),
                "outline": arc.get("outline") or {},
                "canon_beats": arc["canon_beats"],
                "revealed_beats": arc["revealed_beats"],
                "open_hook": arc["open_hook"],
                "user_role": arc["user_role"],
            },
            "recent_chat": self._recent_chat_snippet(8),
        }
        try:
            out = llm.complete_json([
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ])
        except AuxLLMError as exc:
            logger.debug("[story_director] advance llm skipped: %s", exc)
            return None
        if not isinstance(out, dict):
            return None
        if not _safe_nested(out):
            logger.warning("[story_director] rejected unsafe advance output")
            return None
        return out

    # ------------------------------------------------------------------
    # Public actions
    # ------------------------------------------------------------------
    def start_new_arc(
        self,
        *,
        force: bool = False,
        prefer_free: bool = False,
        now_iso: str | None = None,
    ) -> dict[str, Any]:
        mode = effective_mode(self.core)
        if mode == "off":
            return {"outcome": "off"}
        ts = now_iso or self.db.now()
        now_dt = parse_time(ts) or now_local()

        existing = self.store.active_arc()
        if existing:
            if not force:
                return {"outcome": "already_active", "arc": existing}
            self.store.abandon_arc(existing["id"], reason="开启了新的故事线", now=ts)

        cfg = pace_params()
        if not force:
            if self._owner_turn_count() < int(cfg["min_turns_to_start"]):
                return {"outcome": "too_few_turns"}
            if not self._cooldown_ready(now_dt):
                return {"outcome": "cooldown"}
            last_done = self.store.latest_finished()
            if last_done:
                since_iso = str(last_done.get("resolved_at") or last_done.get("updated_at") or "")
                if since_iso and self._owner_turn_count_since(since_iso) < 2:
                    return {"outcome": "cooldown"}

        used = self.store.used_seed_keys()
        seeds = [
            s for s in self.core.story_seeds
            if str(s.get("key") or "") not in used
            and (mode == "drama" or str(s.get("tone") or "drama") == "light")
        ]
        chosen_seed = None if (prefer_free or not seeds) else seeds[0]
        draft = self._llm_new_arc(mode, chosen_seed)

        if draft:
            arc_key = f"seed:{chosen_seed['key']}" if chosen_seed else f"free:{ts[:19]}"
            source = "seed" if chosen_seed else "free"
            tone = str((chosen_seed or {}).get("tone") or mode)
            stage_min = int((chosen_seed or {}).get("stage_min") or (1 if mode == "light" else 1))
            arc = self.store.create_arc(
                arc_key=arc_key,
                source=source,
                title=draft["title"],
                tone=tone,
                stage_min=stage_min,
                premise=draft["premise"],
                he_direction=draft.get("he_direction") or (chosen_seed or {}).get("he_direction") or "在对方陪伴与支持下温暖化解，彼此更加信任。",
                open_hook=draft["open_hook"],
                outline=draft.get("outline"),
                current_act=1,
                next_turn_brief=draft.get("next_turn_brief") or "",
                status="seeded",
                now=ts,
            )
            return {"outcome": "started", "source": source, "used_llm": True, "arc": arc}

        if chosen_seed:
            seed_title = str(chosen_seed["title"])
            seed_premise = str(chosen_seed.get("premise") or "")
            seed_hook = str(chosen_seed.get("initial_hook") or "")
            seed_he = str(chosen_seed.get("he_direction") or "在对方陪伴下温暖化解。")
            arc = self.store.create_arc(
                arc_key=f"seed:{chosen_seed['key']}",
                source="seed",
                title=seed_title,
                tone=str(chosen_seed.get("tone") or mode),
                stage_min=int(chosen_seed.get("stage_min") or 1),
                premise=seed_premise,
                he_direction=seed_he,
                open_hook=seed_hook,
                outline=_fallback_outline_from_seed(seed_title, seed_premise, seed_hook, seed_he),
                current_act=1,
                status="seeded",
                now=ts,
            )
            return {"outcome": "started", "source": "seed", "used_llm": False, "arc": arc}

        return {"outcome": "no_model_for_freeform"}

    def advance_arc(
        self,
        arc_id: int | None = None,
        *,
        force: bool = False,
        now_iso: str | None = None,
    ) -> dict[str, Any]:
        mode = effective_mode(self.core)
        if mode == "off":
            return {"outcome": "off"}
        ts = now_iso or self.db.now()
        arc = self.store.get(arc_id) if arc_id is not None else self.store.active_arc()
        if not arc or arc["status"] not in _NEXT_STATUS:
            return self.start_new_arc(force=force, now_iso=ts)

        if not self._stage_allows_advance(arc, force=force):
            return {"outcome": "stage_gated", "arc": arc, "stage_min": arc.get("stage_min")}

        outline = arc.get("outline") or {}
        acts = outline.get("acts") or []
        cur_act = max(1, int(arc.get("current_act") or 1))
        if acts and cur_act < len(acts):
            target_act = cur_act + 1
            next_entry = acts[target_act - 1]
            target = str(next_entry.get("stage") or "").strip().lower()
            if target not in ("active", "climax", "resolved"):
                target = _NEXT_STATUS[arc["status"]]
        else:
            target_act = cur_act + 1
            target = _NEXT_STATUS[arc["status"]]

        out = self._llm_advance(arc, target, target_act=target_act, now_iso=ts)

        if target == "resolved":
            final_beat = str((out or {}).get("final_beat") or "").strip() or (
                arc.get("he_direction") or f"在对方的陪伴下，顺利渡过了「{arc['title']}」"
            )[:75]
            summary = str((out or {}).get("resolved_summary") or "").strip() or (
                f"一起面对并温暖化解了「{arc['title']}」"
            )
            resolved = self.store.resolve_arc(arc["id"], summary, final_beat=final_beat, boost_mood=True, now=ts)
            return {"outcome": "resolved", "used_llm": bool(out), "arc": resolved}

        next_outline = _merge_revised_acts(
            outline,
            (out or {}).get("revised_future_acts"),
            current_act=target_act,
            target_status=target,
            locked_through=cur_act,
        )
        next_act_obj = current_act_entry({"outline": next_outline, "current_act": target_act})
        new_beat = str((out or {}).get("new_beat") or "").strip()
        new_hook = str((out or {}).get("open_hook") or "").strip()
        new_brief = str((out or {}).get("next_turn_brief") or "").strip()
        if not new_hook:
            new_hook = (
                str(next_act_obj.get("summary") or "")[:90]
                if next_act_obj and next_act_obj.get("summary")
                else (
                    arc["premise"][:90]
                    if target == "active"
                    else (arc.get("he_direction") or "这件事到了必须做出选择的时刻，心里很希望听听你的想法")[:90]
                )
            )
        updated = self.store.append_update(
            arc["id"],
            canon_beat=new_beat or None,
            open_hook=new_hook,
            status=target,
            outline=next_outline,
            current_act=target_act,
            next_turn_brief=new_brief or None,
            advanced=True,
            now=ts,
        )
        return {"outcome": "advanced", "status": target, "current_act": target_act, "used_llm": bool(out), "arc": updated}

    def resolve_now(self, arc_id: int | None = None, *, now_iso: str | None = None) -> dict[str, Any]:
        ts = now_iso or self.db.now()
        arc = self.store.get(arc_id) if arc_id is not None else self.store.active_arc()
        if not arc or arc["status"] not in _NEXT_STATUS:
            return {"outcome": "no_active_arc"}
        out = self._llm_advance(arc, "resolved")
        final_beat = str((out or {}).get("final_beat") or "").strip() or (
            arc.get("he_direction") or f"和对方一起温暖化解了「{arc['title']}」"
        )[:75]
        summary = str((out or {}).get("resolved_summary") or "").strip() or (
            f"和对方一起平稳渡过了「{arc['title']}」"
        )
        resolved = self.store.resolve_arc(arc["id"], summary, final_beat=final_beat, boost_mood=True, now=ts)
        return {"outcome": "resolved", "used_llm": bool(out), "arc": resolved}

    def tick(self, now: datetime | None = None) -> dict[str, Any]:
        """Background periodic check (called from LifeScheduler.tick() per scope)."""
        from ai import test_mode
        if test_mode.active():
            return {"outcome": "test_mode"}
        if effective_mode(self.core) == "off":
            return {"outcome": "off"}
        now_dt = (
            (parse_time(now.isoformat()) or now)
            if (now is not None and now.tzinfo is None)
            else (now or now_local())
        )
        ts = now_dt.isoformat()
        arc = self.store.active_arc()
        if not arc:
            return self.start_new_arc(force=False, now_iso=ts)

        # Never auto-advance an unengaged arc behind the user's back on wall-clock ticks.
        if int(arc.get("turns_engaged") or 0) == 0:
            return {"outcome": "waiting_user_engagement", "arc_id": arc["id"]}

        cfg = pace_params()
        # Even in fast pace, background ticks wait at least 6h so idle time doesn't auto-finish a story unread.
        min_hours = max(6.0, float(cfg["advance_hours"]))
        last_adv = parse_time(arc.get("last_advanced_at"))
        if last_adv is not None and (now_dt - last_adv) < timedelta(hours=min_hours):
            return {"outcome": "waiting", "arc_id": arc["id"]}
        # Do not auto-resolve climax in background without the user participating in chat!
        if arc["status"] == "climax":
            return {"outcome": "waiting_user_at_climax", "arc_id": arc["id"]}
        return self.advance_arc(arc["id"], force=False, now_iso=ts)

    def maybe_advance_after_chat(self, *, ready_hint: bool = False, now_iso: str | None = None) -> dict[str, Any]:
        """Chat-driven check called right after StoryRecorder runs."""
        if effective_mode(self.core) == "off":
            return {"outcome": "off"}
        ts = now_iso or self.db.now()
        arc = self.store.active_arc()
        if not arc:
            return self.start_new_arc(force=False, now_iso=ts)
        cfg = pace_params()
        threshold = max(1, int(cfg["chat_turns_to_advance"]))
        engaged = int(arc.get("turns_engaged") or 0)
        should_step = bool(ready_hint) or (engaged >= threshold and engaged % threshold == 0)
        if not should_step:
            return {"outcome": "steady", "arc_id": arc["id"]}
        # Climax resolves only when StoryRecorder explicitly saw resolution (resolved_now) or after sustained climax engagement.
        if arc["status"] == "climax":
            return {"outcome": "climax_in_progress", "arc_id": arc["id"]}
        return self.advance_arc(arc["id"], force=False, now_iso=ts)
