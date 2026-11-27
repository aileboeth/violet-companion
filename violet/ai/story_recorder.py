"""Async post-turn story recorder (场记层) to lock canonical story facts and prevent self-contradiction.

Runs off the critical reply path (inside MemoryWorker, or after sync run_pipeline), reusing MEMORY_LLM_MODEL.
Ordinary non-story chat turns are filtered out cheaply in pure Python with zero LLM calls and zero contamination of
the user's own event_memories.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ai.aux_llm import AuxLLM, AuxLLMConfig, AuxLLMError
from ai.memory_model import _bigrams
from ai.persona import PersonaCore
from ai.settings import env_float, env_str
from ai.story_director import _safe_nested
from ai.story_store import StoryStore, current_act_entry, effective_mode, pace_params, story_pace

logger = logging.getLogger("violet.story")

# Universal concern / emotion / tension keywords that may signal a turn touched the active storyline
# (persona-specific plot nouns are intentionally excluded; _content_overlap() matches the active arc itself)
_SIGNAL_RE = re.compile(
    r"(怎么了|不对劲|有心事|发生什么|还好吗|别怕|没事吧|我陪你|站你这边|我帮你|撑腰|别理他们|"
    r"心不在焉|心里[^，。！？\n]{0,5}乱|有啥事|有什么事|走神|"
    r"家里|电话|短信|秘密|真相|瞒着|不想聊|别提|换个话题|"
    r"委屈|发抖|红了眼|难过|崩溃|硬撑|压力|决定|答应|拒绝|解决|放心|松了口气)"
)

_DISENGAGE_RE = re.compile(
    r"(别聊这个|不想聊这个|别演了|太狗血|换个话题|别提家里|别提这事|不想听这个)"
)


def _memory_model_name() -> str:
    return env_str("MEMORY_LLM_MODEL", "").strip() or env_str("CHAT_LLM_MODEL", "").strip()


# Common character/everyday bigrams that appear in persona premises and must not false-trigger story detection
_GENERIC_PERSONA_BIGRAMS = frozenset({
    "高冷", "女神", "冰山", "冷淡", "冷漠", "疏离", "温柔", "安静", "笔记", "记本",
    "学校", "大学", "同学", "普通", "平时", "日常", "聊天", "消息", "手机", "对方",
})


def _content_overlap(arc: dict[str, Any], text: str) -> bool:
    grams = _bigrams(text or "") - _GENERIC_PERSONA_BIGRAMS
    if len(grams) < 2:
        return False
    specific_corpus = " ".join(
        [
            str(arc.get("title") or ""),
            str(arc.get("open_hook") or ""),
            " ".join(arc.get("canon_beats") or []),
            " ".join(arc.get("revealed_beats") or []),
        ]
    )
    specific_grams = _bigrams(specific_corpus) - _GENERIC_PERSONA_BIGRAMS
    if len(grams & specific_grams) >= 2:
        return True
    premise_grams = _bigrams(str(arc.get("premise") or "")) - _GENERIC_PERSONA_BIGRAMS
    return len(grams & premise_grams) >= 4


def might_touch_story(arc: dict[str, Any], user_message: str, assistant_reply: str) -> bool:
    """Cheap zero-LLM filter: returns True only if this turn plausibly engaged the active storyline."""
    combined = f"{user_message or ''}\n{assistant_reply or ''}"
    if _DISENGAGE_RE.search(user_message or ""):
        return True
    if _content_overlap(arc, combined):
        return True
    if _SIGNAL_RE.search(combined):
        return True
    return False


def _recorder_messages(arc: dict[str, Any], core: PersonaCore, user_message: str, assistant_reply: str) -> list[dict[str, str]]:
    name = str((core.data.get("identity") or {}).get("name") or "Violet")
    cur_act = current_act_entry(arc)
    acts = (arc.get("outline") or {}).get("acts") or []
    system = (
        f"你是虚拟角色「{name}」的故事场记，负责把这一轮聊天里真正推进或坐实的剧情细节记进档案，防止以后吃书，并为下一轮准备简短导演便签。\n"
        "只输出一个 JSON 对象，不要输出任何额外文字：\n"
        '{"touched": bool, "new_revealed_beat": "她本轮亲口向对方吐露的新线索(无则空，<=48字)", '
        '"new_canon_detail": "仅限本条故事线内她即兴补充的具体情节事实/配角名/时间(无则空，<=48字，绝不记用户称呼/专业/年级等资料)", '
        '"user_stance": "对方对这件事的态度/建议/站队(有变化才写，<=80字)", '
        '"next_turn_brief": "下一轮聊天时她自然衔接当前大纲的简短提醒(无变化则空，<=65字)", '
        '"user_disengaged": bool, "ready_to_advance": bool, '
        '"resolved_now": bool, "resolved_summary": "若本轮已温暖圆满收束，浓缩为1句共同回忆(<=55字，否则空)"}\n'
        "规则：\n"
        "1. 普通闲聊没沾边就输出 {\"touched\": false}；绝不把用户的真实生活琐事误记成她的剧情。\n"
        "2. new_canon_detail 严格只记本条故事线的情节事实，绝不把用户的姓名/称呼/专业/年级/学校等静态个人资料写进 new_canon_detail，也绝不记录实际未发出的照片/截图/文件。\n"
        "3. 只记录本轮聊天原文里确实说出口的内容，不脑补没说过的新转折。\n"
        "4. user_disengaged=true 仅在用户明确表示不想聊这条剧情、反感或叫停时才设。\n"
        "5. resolved_now=true 仅在冲突/心结已经在本轮实质化解、走向温暖 Happy Ending 时才设；"
        "若两人只是约好明天/周末再去处理、事情还没真正落地，不要把 resolved_now 或 ready_to_advance 设为 true。"
    )
    payload = {
        "arc": {
            "title": arc.get("title"),
            "status": arc.get("status"),
            "current_act": f"{arc.get('current_act', 1)}/{len(acts)}" if acts else str(arc.get("current_act", 1)),
            "current_act_goal": cur_act if cur_act else None,
            "premise": arc.get("premise"),
            "open_hook": arc.get("open_hook"),
            "canon_beats": arc.get("canon_beats", [])[-4:],
            "revealed_beats": arc.get("revealed_beats", [])[-4:],
            "user_role": arc.get("user_role", ""),
        },
        "turn": {
            "user": (user_message or "")[:600],
            "assistant": (assistant_reply or "")[:800],
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def record_turn(db, user_message: str, assistant_reply: str, *, core: PersonaCore | None = None,
                now: str | None = None) -> dict[str, Any]:
    """Inspect one completed turn and update the active story arc (if any). Returns outcome dict."""
    from ai import test_mode
    if test_mode.active():
        return {"outcome": "test_mode"}
    core = core or PersonaCore.load()
    if effective_mode(core) == "off":
        return {"outcome": "off"}
    if not (assistant_reply or "").strip():
        return {"outcome": "empty_reply"}

    store = StoryStore(db)
    arc = store.active_arc()
    if not arc:
        return {"outcome": "no_active_arc"}

    # Explicit user rejection short-circuits without spending an LLM call
    if _DISENGAGE_RE.search(user_message or ""):
        store.abandon_arc(arc["id"], reason="用户表示不想继续聊这条剧情", now=now)
        return {"outcome": "abandoned_by_user", "arc_id": arc["id"]}

    if not might_touch_story(arc, user_message, assistant_reply):
        return {"outcome": "skipped_unrelated", "arc_id": arc["id"]}

    model = _memory_model_name()
    if not model:
        return {"outcome": "no_model", "arc_id": arc["id"]}

    pace_cfg = pace_params()
    timeout = max(60.0, min(180.0, env_float("LLM_REQUEST_TIMEOUT_SECONDS", 60.0)))
    llm = AuxLLM(
        db,
        AuxLLMConfig(
            purpose="story_recorder",
            model=model,
            timeout_seconds=timeout,
            max_output_tokens=1024,
            temperature=0.2,
            max_attempts=2,
            max_calls_per_day=max(12, int(pace_cfg["max_calls_per_day"]) * 3),
            max_tokens_per_day=30000,
        ),
    )
    if not llm.budget_left():
        return {"outcome": "budget", "arc_id": arc["id"]}

    try:
        out = llm.complete_json(_recorder_messages(arc, core, user_message, assistant_reply))
    except AuxLLMError as exc:
        logger.debug("[story_recorder] skipped: %s", exc)
        return {"outcome": "llm_error", "error": str(exc)[:120], "arc_id": arc["id"]}

    if not isinstance(out, dict):
        return {"outcome": "invalid", "arc_id": arc["id"]}
    if not _safe_nested(out):
        logger.warning("[story_recorder] rejected unsafe recorder output")
        return {"outcome": "rejected_unsafe", "arc_id": arc["id"]}

    if out.get("user_disengaged") is True:
        updated = store.append_update(arc["id"], turns_ignored_delta=1, now=now)
        if updated and int(updated.get("turns_ignored") or 0) >= 3:
            store.abandon_arc(arc["id"], reason="连续多次未承接，温和淡出", now=now)
            return {"outcome": "abandoned_ignored", "arc_id": arc["id"]}
        return {"outcome": "disengaged", "arc_id": arc["id"]}

    if out.get("touched") is not True:
        return {"outcome": "not_touched", "arc_id": arc["id"]}

    rev = str(out.get("new_revealed_beat") or "").strip()
    canon = str(out.get("new_canon_detail") or "").strip()
    stance = str(out.get("user_stance") or "").strip()
    next_brief = str(out.get("next_turn_brief") or "").strip()
    resolved_now = bool(out.get("resolved_now")) and arc["status"] in ("active", "climax")
    resolved_summary = str(out.get("resolved_summary") or "").strip()

    if resolved_now:
        final_beat = rev or canon or None
        resolved = store.resolve_arc(
            arc["id"],
            resolved_summary or f"和对方一起温暖渡过了「{arc['title']}」",
            final_beat=final_beat,
            boost_mood=True,
            now=now,
        )
        return {"outcome": "resolved", "arc_id": arc["id"], "arc": resolved}

    # Only count engagement and auto-promote seeded -> active when the user actually engaged AND a beat was revealed
    # (never promote unilaterally on an unprompted assistant hint, and never bypass StoryDirector._stage_allows_advance()
    # via ready_to_advance when rev is empty).
    user_engaged = bool(stance) or might_touch_story(arc, user_message, "")
    next_status = None
    next_act = None
    if arc["status"] == "seeded" and rev and user_engaged:
        next_status = "active"
        acts = (arc.get("outline") or {}).get("acts") or []
        if int(arc.get("current_act") or 1) == 1 and len(acts) >= 2:
            next_act = 2

    updated = store.append_update(
        arc["id"],
        canon_beat=canon or None,
        revealed_beat=rev or None,
        user_role=stance or None,
        status=next_status,
        current_act=next_act,
        next_turn_brief=next_brief or None,
        turns_engaged_delta=1 if user_engaged else 0,
        reset_ignored=user_engaged,
        advanced=bool(next_status),
        now=now,
    )
    return {
        "outcome": "recorded",
        "arc_id": arc["id"],
        "promoted": bool(next_status),
        "user_engaged": user_engaged,
        "ready_to_advance": bool(out.get("ready_to_advance")) and user_engaged,
        "turns_engaged": int((updated or {}).get("turns_engaged") or 0),
        "status": (updated or {}).get("status"),
    }


def after_turn(db, user_message: str, assistant_reply: str, *, core: PersonaCore | None = None,
               now: str | None = None) -> dict[str, Any]:
    """Post-reply hook: run story recorder, then let StoryDirector check chat-driven progression."""
    core = core or PersonaCore.load()
    if effective_mode(core) == "off":
        return {"outcome": "off"}
    rec = record_turn(db, user_message, assistant_reply, core=core, now=now)
    rec_outcome = rec.get("outcome")
    if rec_outcome in ("resolved", "abandoned_by_user", "abandoned_ignored"):
        return {"recorder": rec, "director": {"outcome": "skipped_terminal"}}
    # Never double-advance in a single turn when record_turn already promoted seeded -> active (Act 1 -> Act 2).
    if rec.get("promoted"):
        return {"recorder": rec, "director": {"outcome": "promoted_this_turn"}}
    # Only allow chat-driven progression when checking auto-start (no_active_arc) or this turn engaged the arc.
    if rec_outcome not in ("no_active_arc", "recorded"):
        return {"recorder": rec, "director": {"outcome": "skipped_untouched"}}
    if rec_outcome == "recorded" and not rec.get("user_engaged"):
        return {"recorder": rec, "director": {"outcome": "skipped_unengaged"}}
    try:
        from ai.story_director import StoryDirector
        director = StoryDirector(db, core=core)
        dir_out = director.maybe_advance_after_chat(
            ready_hint=bool(rec.get("ready_to_advance")),
            now_iso=now,
        )
        return {"recorder": rec, "director": dir_out}
    except Exception as exc:                                # noqa: BLE001
        logger.debug("[story] post-turn director skipped: %s", exc)
        return {"recorder": rec, "director": {"outcome": "error", "error": str(exc)[:120]}}
