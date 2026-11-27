"""Layer 1 hybrid pre-turn decision engine (回合决断层) for Violet (Option A).

Separates cool-headed plot/state/tool decision making from in-character dialogue writing:
  * ``off``    : disabled (rely only on full outline + async post-turn StoryRecorder/StoryDirector).
  * ``smart``  : (default, Option A) zero extra latency on ordinary casual chat; synchronously runs a
                 lightweight Layer 1 JSON decision before ChatAgent when an active story arc is engaged,
                 at climax, or on complex multi-intent turns.
  * ``always`` : synchronously runs Layer 1 before every chat turn.

Safety & performance guarantees:
  * Hard per-call timeout (``VIOLET_DECISION_TIMEOUT_SECONDS``, default 15s, max_attempts=1): any timeout
    or relay hiccup gracefully falls back to single-call ChatAgent without ever stalling or failing a reply.
  * Soft tool hints ONLY: Layer 1 suggests tools from the real offered tool list, while ChatAgent keeps full
    native function-calling autonomy (avoiding V26 IntentAgent tool-starvation regressions).
  * Volatile tail injection ONLY: the short decision note is injected at the volatile tail before SAID_NOW,
    keeping the stable system prompt prefix cache intact.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ai.aux_llm import AuxLLM, AuxLLMConfig, AuxLLMError
from ai.persona import PersonaCore
from ai.settings import env_float, env_int, env_str
from ai.story_director import _safe_nested
from ai.story_recorder import might_touch_story
from ai.story_store import (
    StoryStore,
    TURN_DECISION_HEADER,
    current_act_entry,
    effective_mode,
)

logger = logging.getLogger("violet.decision")

DECISION_MODES = ("off", "smart", "always")

# Multi-intent / tool + emotion patterns that benefit from pre-turn planning even without an active story hit
_COMPLEX_INTENT_RE = re.compile(
    r"(提醒我|定个闹钟|查一下|搜一下|帮我找|课表|画一张|画个|改一下图|之前说过|还记得吗|答应过|以后怎么办|怎么选)"
)


def decision_mode() -> str:
    raw = (env_str("VIOLET_DECISION_MODE", "smart") or "smart").strip().lower()
    return raw if raw in DECISION_MODES else "smart"


def decision_model_name() -> str:
    return (
        env_str("VIOLET_DECISION_MODEL", "").strip()
        or env_str("MEMORY_LLM_MODEL", "").strip()
        or env_str("CHAT_LLM_MODEL", "").strip()
    )


def _recent_messages(db, limit: int = 4) -> list[dict[str, str]]:
    from ai import context_cut
    rows = context_cut.owner_rows(db.get_recent_messages(limit=limit))
    out: list[dict[str, str]] = []
    for r in rows[-limit:]:
        content = str(r.get("content") or "").replace("\n", " ").strip()[:160]
        if content:
            out.append({"role": "user" if r.get("role") == "user" else "assistant", "content": content})
    return out


def _last_assistant_text(db) -> str:
    from ai import context_cut
    rows = context_cut.owner_rows(db.get_recent_messages(limit=6))
    for r in reversed(rows):
        if r.get("role") != "user":
            return str(r.get("content") or "")
    return ""


def _current_stage_label(db) -> str:
    try:
        from ai.channels.context import current
        from ai import persona_templates as pt
        st = pt.stage(current().channel, db)
        return str((st or {}).get("label") or "")
    except Exception:                                       # noqa: BLE001
        return ""


def should_decide(
    db,
    user_message: str,
    *,
    core: PersonaCore | None = None,
    active_arc: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Cheap zero-LLM gate deciding whether Layer 1 runs synchronously on this turn."""
    from ai import test_mode
    if test_mode.active():
        return False, "test_mode"
    mode = decision_mode()
    if mode == "off":
        return False, "mode_off"
    core = core or PersonaCore.load()
    if core.is_ai or core.is_blank:
        return False, "disabled_persona"
    if not decision_model_name():
        return False, "no_model"
    if mode == "always":
        return True, "mode_always"

    # mode == "smart" (Option A)
    msg = (user_message or "").strip()
    if not msg:
        return False, "empty_message"

    if effective_mode(core) != "off":
        arc = active_arc if active_arc is not None else StoryStore(db).active_arc()
        if arc:
            if arc.get("status") == "climax":
                return True, "story_climax"
            if might_touch_story(arc, msg, ""):
                return True, "story_signal"
            if int(arc.get("turns_engaged") or 0) > 0:
                last_reply = _last_assistant_text(db)
                if last_reply and might_touch_story(arc, "", last_reply):
                    return True, "story_continuation"

    if len(msg) >= 16 and _COMPLEX_INTENT_RE.search(msg):
        return True, "complex_intent"
    return False, "mundane_fast_path"


def available_tool_hints(db, direct_tools: list[str] | None = None) -> list[str]:
    """Return real tool names currently available to ChatAgent so Layer 1 only suggests valid tools."""
    from ai import memes, mood, owner_tools, stickers
    from ai.images.service import images_enabled

    names: list[str] = []
    if owner_tools.has_memories(db):
        names.append("query_memory")
    names.extend(["set_reminder", "cancel_reminder"])
    if mood.enabled():
        names.append("note_feeling")
    if memes.enabled():
        names.append("remember_meme")
        if memes.wiki_available() or memes.own_list(db):
            names.append("look_up_meme")
    if stickers.enabled() and db.fetch_one("SELECT 1 FROM stickers WHERE enabled = 1 LIMIT 1"):
        names.append("find_stickers")
    if images_enabled():
        names.append("create_image")
    for t in (direct_tools or []):
        s = str(t or "").strip()
        if s and s not in names:
            names.append(s)
    return names[:30]


def format_decision_note(decision: dict[str, Any]) -> str:
    """Render Layer 1 structured JSON decision into a lean volatile director brief for Layer 2 ChatAgent."""
    if not isinstance(decision, dict):
        return ""
    lines: list[str] = []
    tone = str(decision.get("emotion_tone") or "").strip()[:40]
    char_note = str(decision.get("character_note") or "").strip()[:55]
    if tone or char_note:
        combined = f"{tone}（{char_note}）" if (tone and char_note) else (tone or char_note)
        lines.append(f"- 情绪与反差：{combined}")
    plan = str(decision.get("response_plan") or "").strip()[:100]
    if plan:
        lines.append(f"- 本轮推进策略：{plan}")
    limit = str(decision.get("reveal_limit") or "").strip()[:75]
    if limit:
        lines.append(f"- 边界与防剧透：{limit}")
    tools = [str(t).strip() for t in (decision.get("suggested_tools") or []) if str(t).strip()]
    tool_reason = str(decision.get("tool_reason") or "").strip()[:60]
    if tools:
        why = f"（{tool_reason}）" if tool_reason else ""
        lines.append(f"- 工具建议：可优先考虑调用 {'、'.join(tools[:3])}{why}；若回复中需要其他工具仍可自行调用。")
    if not lines:
        return ""
    body = "\n".join(lines)
    return (
        f"{TURN_DECISION_HEADER}\n"
        f"{body}\n"
        "规则：消化后完全用角色自己的自然口吻聊天（每次1-3条短消息，别每句重复称呼对方，别动不动用“……”开头），"
        "严禁声称已发出或发送照片/图片/截图/文件，绝不照念以上文字，也不提“大纲/决断/策略/工具”。"
    )


def decide_turn(
    db,
    user_message: str,
    *,
    direct_tools: list[str] | None = None,
    core: PersonaCore | None = None,
    prechecked: tuple[bool, str] | None = None,
) -> dict[str, Any]:
    """Run Layer 1 pre-turn decision if gated on; returns structured decision + volatile note."""
    core = core or PersonaCore.load()
    arc = StoryStore(db).active_arc() if effective_mode(core) != "off" else None
    run_flag, reason = prechecked if prechecked is not None else should_decide(
        db, user_message, core=core, active_arc=arc
    )
    if not run_flag:
        return {"ran": False, "reason": reason, "note": ""}

    model = decision_model_name()
    if not model:
        return {"ran": False, "reason": "no_model", "note": ""}

    timeout = max(3.0, min(120.0, env_float("VIOLET_DECISION_TIMEOUT_SECONDS", 15.0)))
    max_calls = max(0, env_int("VIOLET_DECISION_MAX_CALLS_PER_DAY", 120))
    llm = AuxLLM(
        db,
        AuxLLMConfig(
            purpose="turn_decision",
            model=model,
            timeout_seconds=timeout,
            max_output_tokens=640,
            temperature=0.25,
            max_attempts=1,
            max_calls_per_day=max_calls,
            max_tokens_per_day=80000,
        ),
    )
    if not llm.budget_left():
        return {"ran": False, "reason": "budget", "note": ""}

    offered_tools = available_tool_hints(db, direct_tools)
    ident = core.data.get("identity") or {}
    rel = core.data.get("relation") or {}
    persona_brief = {
        "name": ident.get("name") or "Violet",
        "relation": rel.get("label") or "",
        "stage": _current_stage_label(db),
        "origin": core.origin[:120],
        "personality": (core.data.get("personality") or [])[:2],
    }

    arc_brief = None
    if arc:
        cur_act = current_act_entry(arc)
        acts = (arc.get("outline") or {}).get("acts") or []
        arc_brief = {
            "title": arc.get("title"),
            "status": arc.get("status"),
            "current_act": arc.get("current_act", 1),
            "total_acts": len(acts),
            "premise": arc.get("premise"),
            "current_act_detail": cur_act,
            "future_acts_preview": [
                {"act": a["act"], "stage": a["stage"], "title": a["title"]}
                for a in acts
                if int(a.get("act") or 0) > int(arc.get("current_act") or 1)
            ],
            "canon_beats": (arc.get("canon_beats") or [])[-4:],
            "revealed_beats": (arc.get("revealed_beats") or [])[-4:],
            "open_hook": arc.get("open_hook"),
            "user_role": arc.get("user_role"),
            "he_direction": arc.get("he_direction"),
        }

    system = (
        "你是虚拟伴侣对话的第一层【回合决断导演】。你【不直接和用户说话】，只做冷静快速的局势判断，"
        "供第二层【聊天模型】照着演戏，防止聊天模型忘大纲、提前剧透、人设跑偏或漏用关键工具。\n"
        "硬性红线：严禁建议自伤/威胁/跟踪/悲剧BE；绝不自称物理真人或活人（被问是否AI时按人设坦率承认、装傻或岔开均可）；"
        "严禁声称已发出或发送照片/图片/截图/文件，用文字口头描述细节即可；"
        "严禁提前剧透 future_acts_preview 里尚未发生的后续章节；"
        "若对方此刻在聊普通日常或换了话题，必须优先顺着对方聊、绝不强行把话题拽回剧情；"
        "suggested_tools 只能从 available_tools 里挑选确实需要的工具名，不需要工具就输出空数组 []。\n"
        "只输出一个紧凑 JSON 对象：\n"
        '{"story_relevance": "none|subtle|direct", '
        '"act_action": "hold|reveal_hint|advance_act|climax_push|resolve_he|pause_for_user", '
        '"emotion_tone": "本轮情绪基调(<=28字)", '
        '"character_note": "人设反差/语气微调提醒(<=40字)", '
        '"response_plan": "本轮聊天怎么接、推进到什么程度(<=75字)", '
        '"reveal_limit": "本轮不能剧透或越界的事(<=55字)", '
        '"suggested_tools": [], "tool_reason": "建议工具原因(不用则空，<=45字)"}'
    )
    payload = {
        "trigger_reason": reason,
        "persona": persona_brief,
        "active_arc": arc_brief,
        "recent_chat": _recent_messages(db, limit=4),
        "user_message": (user_message or "")[:600],
        "available_tools": offered_tools,
    }
    try:
        out = llm.complete_json([
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ])
    except AuxLLMError as exc:
        logger.debug("[turn_decision] skipped on fallback: %s", exc)
        return {"ran": True, "outcome": "fallback", "reason": reason, "model": model, "error": str(exc)[:140], "note": ""}

    if not isinstance(out, dict):
        return {"ran": True, "outcome": "invalid", "reason": reason, "model": model, "note": ""}
    if not _safe_nested(out):
        logger.warning("[turn_decision] rejected unsafe decision output")
        return {"ran": True, "outcome": "rejected_unsafe", "reason": reason, "model": model, "note": ""}

    allowed_set = set(offered_tools)
    raw_tools = out.get("suggested_tools")
    clean_tools = [
        str(t).strip()
        for t in (raw_tools if isinstance(raw_tools, list) else [])
        if str(t).strip() in allowed_set
    ][:3]

    clean_decision = {
        "story_relevance": str(out.get("story_relevance") or "none").strip().lower(),
        "act_action": str(out.get("act_action") or "hold").strip().lower(),
        "emotion_tone": str(out.get("emotion_tone") or "").strip()[:40],
        "character_note": str(out.get("character_note") or "").strip()[:55],
        "response_plan": str(out.get("response_plan") or "").strip()[:100],
        "reveal_limit": str(out.get("reveal_limit") or "").strip()[:75],
        "suggested_tools": clean_tools,
        "tool_reason": str(out.get("tool_reason") or "").strip()[:60] if clean_tools else "",
    }
    note = format_decision_note(clean_decision)
    return {
        "ran": True,
        "outcome": "ok",
        "reason": reason,
        "model": model,
        "decision": clean_decision,
        "note": note,
    }
