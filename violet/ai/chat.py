import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

from database import VioletDatabase
from dotenv import load_dotenv
from ai.debug_visualizer import visualize_pipeline_start, visualize_pipeline_end, visualize_step
import ai.pipeline_state as _ps
from ai.settings import ENV_PATH, env_bool, env_str
from ai.chat_output import NO_REPLY_TOKEN, normalize_chat_output, strip_images
from ai.conversation_state import TopicTracker
from ai.memory_model import EVENT_KINDS, KIND_LABELS, PROGRESS_LABELS, event_time_label, is_date_only, touches_topic

violetdb = VioletDatabase()
logger = logging.getLogger("violet.pipeline")

# Load .env from project root (same directory as main.py); VIOLET_ENV_FILE overrides.
_ENV_PATH = ENV_PATH
load_dotenv(_ENV_PATH)


# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------

def get_llm_api_key() -> str:
    key = os.getenv("LLM_API_KEY", "")
    if not key:
        raise ValueError("LLM_API_KEY is not set in .env")
    return key


def get_llm_api_url() -> str:
    return os.getenv("LLM_API_URL", "https://api.openai.com/v1")


def get_mcp_servers(apply_settings: bool = True) -> list[dict[str, str]]:
    """Every MCP server Violet knows: MCP_SERVERS, the servers added in the Control Center, the built-in ones -- with the
    Control Center's settings applied (phase 9.2: enabled, agents, timeouts, switched-off tools, replaced headers).
    apply_settings=False: as configured (agents keep that and apply the settings at the start of every run, so a
    switch in the Control Center acts on the next call without reloading anything)."""
    from ai.mcp import builtin_servers
    raw = os.getenv("MCP_SERVERS", "").strip()
    servers = []
    if raw:
        try:
            servers = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("MCP_SERVERS must be valid JSON") from exc
        if not isinstance(servers, list):
            raise ValueError("MCP_SERVERS must be a JSON list")
    # phase 10.5: servers added in the Control Center (a name already used in .env keeps the .env one)
    try:
        from ai.tool_manager import tool_store
        taken = {s.get("name") for s in servers if isinstance(s, dict)}
        servers = servers + [dict(s) for s in tool_store().custom_servers()
                             if isinstance(s, dict) and s.get("name") not in taken]
    except Exception:
        logger.exception("servers added in the Control Center could not be read")
    # built-in servers (phase 5B: violet-web) cannot be shadowed by a configured server of the same name
    builtin = builtin_servers()
    reserved = {b["name"] for b in builtin}
    merged = [s for s in servers if not (isinstance(s, dict) and s.get("name") in reserved)] + builtin
    if not apply_settings:
        return merged
    from ai.tool_manager import apply_tool_settings
    return apply_tool_settings(merged)


def llm_configured() -> bool:
    """Phase 10: whether the model service is set up (API address and key). Violet starts without it -- the Control
    Center's first-run wizard asks for both -- and only the model calls fail until then."""
    return bool(os.getenv("LLM_API_KEY", "").strip() and os.getenv("LLM_API_URL", "").strip())


class LLMConfig:
    """Holds all LLM configuration loaded from environment variables."""

    def __init__(self):
        # phase 10: a missing key no longer stops Violet from starting (agents were built at import time, so a fresh
        # install without .env crashed before the dashboard came up); a model call without a key fails clearly
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.api_url = get_llm_api_url()
        self.reply_model: str = ""         # overridden per-agent
        self.reply_prompt: str = ""        # overridden per-agent
        self.reply_temperature: float = 0.2  # overridden per-agent
        self.fallback_model: str = ""      # optional <AGENT>_LLM_FALLBACK_MODEL
        self.mcp_servers = get_mcp_servers(apply_settings=False)   # settings applied per run (BaseAgent)

    def __repr__(self) -> str:
        return f"LLMConfig(model={self.reply_model}, api_url={self.api_url})"


# ---------------------------------------------------------------------------
# Context helpers (shared across agents)
# ---------------------------------------------------------------------------

def format_recent_messages(limit: int = 30) -> str:
    """Build a compact recent-message context block from SQLite (MemoryAgent's reference: after 翻篇 only the chat
    since then -- 2026-11-09, the old "我今晚通宵" was there to be remembered again after 清除记忆 + 翻篇)."""
    from ai import context_cut
    rows = context_cut.owner_rows(violetdb.get_recent_messages(limit=limit))
    if not rows:
        return "最近聊天记录：无"

    lines = ["最近聊天记录："]
    for row in rows:
        speaker = "用户" if row["role"] == "user" else "assistant"
        message_type = row["message_type"]
        content = row.get("content") or ""
        body = content if message_type == "text" else f"[{message_type}] {content}".strip()
        lines.append(f"- {row['created_at']} {speaker}: {body}")

    return "\n".join(lines)


def format_base_info() -> str:
    # 2026-11-09: the weekday and what "明天" means after midnight, written by the code (an ISO stamp alone left both to
    # the model)
    from ai.memory_model import clock_text, day_words, parse_time
    now = parse_time(violetdb.now())
    lines = ["基础信息：", f"当前时间: {clock_text(now)}", f"日期：{day_words(now)}"]
    return "\n".join(lines) + "\n"


def extract_tool_memories(agent_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract query_memory tool results from an agent run result."""
    recalled: list[dict[str, Any]] = []
    for message in agent_result.get("messages", []):
        if message.get("role") != "tool":
            continue
        content = message.get("content") or ""
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        for memory in payload.get("memories", []):
            if isinstance(memory, dict) and memory not in recalled:
                recalled.append(memory)
    return recalled


def format_recalled_memories(memories: list[dict[str, Any]]) -> str:
    memories = [m for m in memories if m.get("status") != "deleted"]
    if not memories:
        return "相关记忆：无"
    status_labels = {"expired": "已过有效期", "archived": "很久以前", "forgotten": "印象很模糊"}
    lines = ["相关记忆（来自记忆库，可能与本轮有关；只在自然相关时使用，不要逐条复述）："]
    for memory in memories:
        kind = memory.get("kind") or "legacy"
        tags = [KIND_LABELS.get(kind, kind)]
        progress = memory.get("progress")
        if kind in EVENT_KINDS and progress:
            tags.append(PROGRESS_LABELS.get(progress, progress))
        intent = memory.get("intent_level") or memory.get("intent")        # a database row / a query_memory view
        if intent == "conditional":
            # phase 9.2.1: "要是周三下午没课就去西湖" is not a plan until the user says so
            condition = memory.get("intent_condition") or memory.get("condition")
            tags.append(f"有条件的打算，还没定（条件：{condition or '见内容'}）")
        elif intent == "wish":
            tags.append("只是随口的想法，不是计划")
        if memory.get("event_time"):
            # phase 9.1.1: a day without a clock says so (it used to be shown as 23:59)
            when = event_time_label(memory["event_time"]) if is_date_only(memory["event_time"]) else memory["event_time"]
            tags.append(f"时间 {when}")
        if memory.get("status") in status_labels:
            tags.append(status_labels[memory["status"]])
        if (memory.get("retrieval") or {}).get("reason") == "user_words":
            tags.append("按这句话里的词找到，不一定相关")       # phase 9.1: found by code, not by a search
        if memory.get("id") is not None:
            # phase 9.1.2: an event record that still uses a value the user later corrected (not rewritten, flagged)
            from ai.memory_consistency import conflict_tag, pending_conflicts
            conflicts = pending_conflicts(violetdb, "event_memories", int(memory["id"]), memory.get("content"))
            if conflicts:
                tags.append(conflict_tag(conflicts))
        lines.append(f"- id={memory.get('id')}; [{' / '.join(tags)}] {memory.get('content')}")
    rule = "进展未知或很久以前的事别假定现状或追问；已结束的事别当正在发生。"
    if any((m.get("kind") or "legacy") == "legacy" for m in memories):
        # 0.7.49 (user: Eve 的记忆没迁移到位): Eve's own records ("夜间互动适度收敛，可称呼'宝宝'或'老公'") are notes for HER,
        # the one replying -- read without that, she asked the owner to call her 老公
        rule += (f"“{KIND_LABELS['legacy']}”是旧系统记下的：“用户”就是对方，没写主语的称呼、语气、做法指你怎么对对方"
                 "（如“可称呼宝宝”= 你叫对方宝宝）。")
    lines.append(rule)
    return "\n".join(lines)


CONSISTENCY_HINTS = {
    "unrecorded_experience": "没有记录能说明这件事真的发生过",
    "plan_described_as_done": "这还只是计划，没有发生",
    "user_experience_as_own": "这是用户的经历，不是你的",
    "forbidden_backstory": "人设里没有这样的背景",
    "identity_conflict": "与虚拟身份设定不符",
    "external_as_experience": "这是网上看到的外部资料，不是你的亲身经历，可以说“看到一篇文章说…”",
    "camera_claim": "图片是 AI 画的插画，不是照片，不能说是拍的、自拍或摄像头拍的",
    # phase 8.9.4
    "stale_weather": "对方这次没提天气，你也不知道现在的天气；以前聊到的天气是那时候的，别说成现在",
    # phase 8.9.5
    "leaked_draft": "这一行混进了英文的草稿和分析，那不是给对方的话；只写要发给对方的消息",
    # 2026-11-13
    "promised_later_lookup": "以后你自己发消息时查不了东西，这个答应做不到：查到的现在说清楚，没查到就照实说，"
                             "需要的话让对方到时候自己看一眼；不要答应晚点 / 回头再查",
    # 2026-11-25
    "same_day_contradiction": "这句和你今天前面自己说过的话自相矛盾，请按前面已经说过的事实保持一致",
    # 2026-11-27
    "unbacked_media_send": "这一轮并没有真正发出照片、图片、截图或文件，不要说已经发给对方了；把想让对方知道的细节用文字直接说出来就好",
}


def consistency_note(violations: list[Any]) -> str:
    """Correction note for one rewrite: name the lines, ask for a natural rephrase, keep the rest."""
    from ai.semantic_guard import is_uncertain
    parts = [f"“{v.line}”（{CONSISTENCY_HINTS.get(v.check, v.detail)}）" for v in violations if v.line]
    note = "；".join(dict.fromkeys(parts))
    if any(is_uncertain(v) for v in violations):
        note += "。这些说法无法确认，请自然地改成想法、打算或感受，不要描述没有记录的具体经历"
    return note + "。其余内容和语气尽量保持不变"


def _certain(violations: list[Any]) -> list[Any]:
    """2026-11-11 (audit A8): only findings the rules are sure of (a made-up family, a plan told as done, a "photo",
    drafting notes) are cut out of a reply. "No recorded event behind this" and "sounds like the owner's experience"
    are guesses from keywords: they get the one rewrite, and what the model writes then is sent -- a guess no longer
    deletes a line (the reply used to arrive with a sentence missing, sometimes down to one leaning half-line)."""
    from ai.semantic_guard import is_uncertain
    return [v for v in violations if not is_uncertain(v)]


def _review(guard, semantic, text: str) -> list[Any]:
    violations = guard.check_reply(text)
    if violations and semantic is not None:
        try:
            violations = semantic.review(violations)
        except Exception:
            logger.exception("semantic review failed; rule findings kept")
    return violations


def guard_reply(content: str, regenerate=None, entity: str = "reply", turn_used_web: bool = False,
                turn_used_schedule: bool = False, turn_sent_media: bool = False) -> tuple[str, list[Any]]:
    """Keep Violet's text consistent with her recorded life and persona (phase 4, 5A).

    1. rules (ai/consistency.py);
    2. optional semantic review of UNCERTAIN findings only (ai/semantic_guard.py):
       may clear a finding with evidence, never changes recorded history;
    3. enforce: one natural rephrase with a correction note; after it only lines the rules are
       sure about are dropped (_certain) -- a keyword guess never deletes a line of a rewritten reply.
    flag: record only; off: nothing. Returns (text, violations_found).
    """
    from ai.consistency import ConsistencyGuard, guard_mode
    from ai.semantic_guard import SemanticGuard
    mode = guard_mode()
    if mode == "off" or not content:
        return content, []
    # 2026-11-11 (eval on the user's server: "给我唱首歌嘛" -> the lyrics inside <voice> were read as claims about her
    # own life, and the rewrite lost the voice message): what she reads, sings or recites in a voice message is a
    # performance, not a statement -- only her written lines are checked
    def written(text: str) -> str:
        return re.sub(r"<voice>.*?</voice>", "", text or "", flags=re.S)
    if not written(content).strip():
        return content, []
    try:
        guard = ConsistencyGuard(violetdb)
        guard.check_weather = entity == "reply"          # phase 8.9.4: a proactive weather task has real data
        guard.turn_used_web = turn_used_web               # phase 9.3: "刚查了一下" is backed by this turn's search
        guard.turn_used_schedule = turn_used_schedule     # phase 9.4.2: "看了下你的课表" is this turn's lookup
        guard.turn_sent_media = turn_sent_media           # 2026-11-27: media/image actually sent in this turn
        semantic = SemanticGuard(violetdb)
        violations = _review(guard, semantic, written(content))
    except Exception:
        logger.exception("consistency guard failed; reply not checked")
        return content, []
    if not violations:
        return content, []
    from ai.life_store import LifeStore
    store = LifeStore(violetdb)
    for v in violations:
        semantic_note = f" | semantic: {v.semantic}" if v.semantic else ""
        store.flag(entity, None, v.check, f"{v.detail} | {v.line or ''}{semantic_note}")
    if mode != "enforce":
        return content, violations
    if regenerate is not None:
        try:
            retry = regenerate(consistency_note(violations))
            # a line the model left exactly as it was ignored the correction: out, as before. A sentence it did
            # rewrite is only cut when the rules are sure about it (_certain)
            ignored = {v.line for v in violations if v.line}
            remaining = [v for v in _review(guard, semantic, written(retry))
                         if v.line in ignored or v in _certain([v])]
            if not remaining:
                return retry, violations
            return guard.drop_lines(retry, remaining), violations
        except Exception:
            logger.exception("regeneration after consistency violation failed")
    # no rewrite was possible (her own proactive message): a guessed finding is still left out, as before
    return guard.drop_lines(content, violations if regenerate is None else _certain(violations)), violations


def enforce_follow_up_preference(user_message: str, content: str, regenerate=None) -> tuple[str, dict[str, Any] | None]:
    """Phase 8.8: the user said "别老问我问题" and Violet still ended a reply to a plain share with a question.

    One rewrite with a note; the rewrite is used only if it no longer ends with a question and passes the
    consistency guard. Otherwise the original stays: no question mark is ever cut out, and a question the user
    invited, a question of the user's own, or a turn without the preference is never touched.
    Returns (text, check) -- check is None when nothing applied."""
    from ai.interaction import FOLLOW_UP_NOTE, follow_up_preference, follow_up_violation
    from ai.profile_store import Profile
    from ai.settings import env_bool
    if not content or regenerate is None or not env_bool("VIOLET_FOLLOW_UP_CHECK", True):
        return content, None
    try:
        pref = follow_up_preference(Profile(violetdb).preferences.effective())
        # phase 8.9: the answer to a question the user invited may get one interested follow-up
        from ai.context_engine import ContextEngine
        session = ContextEngine(violetdb)._session_messages() if pref else []
        previous_user = next((m["content"] for m in reversed(session) if m["role"] == "user"), None)
    except Exception:
        logger.exception("follow-up preference unavailable; reply kept")
        return content, None
    if not follow_up_violation(content, user_message, pref, previous_user):
        return content, None
    try:
        retry = (regenerate(FOLLOW_UP_NOTE) or "").strip()
    except Exception:
        logger.exception("rewrite for the follow-up preference failed; reply kept")
        return content, {"rewritten": False, "reason": "rewrite_failed"}
    if not retry:
        return content, {"rewritten": False, "reason": "empty_rewrite"}
    if follow_up_violation(retry, user_message, pref, previous_user):
        return content, {"rewritten": False, "reason": "rewrite_still_asks"}
    checked, violations = guard_reply(retry)
    if violations or not checked.strip():
        return content, {"rewritten": False, "reason": "rewrite_failed_consistency"}
    return checked, {"rewritten": True, "reason": None}


def note_life_mentions(reply_text: str, user_message: str = "", skip_event_ids: list[int] | None = None) -> list[int]:
    """Life events Violet talked about in a normal reply: mark them shared so they are not offered again."""
    from ai.conversation_state import LIFE_TOPIC, life_topic_key
    from ai.life_store import LifeStore
    from ai.memory_model import mentions_topic
    from ai.daily_life import mentions_life
    if not reply_text:
        return []
    store = LifeStore(violetdb)
    mentioned = []
    skip = {int(i) for i in skip_event_ids or []}          # already counted by a picture in this reply
    for event in store.recent_happened(days=30):
        if event["id"] in skip:
            continue
        if mentions_topic(reply_text, event["content"]) or mentions_life(reply_text, event["content"]):
            store.mark_shared(event["id"])
            violetdb.add_topic_mention(life_topic_key(event["id"]), "violet_reply")
            mentioned.append(event["id"])
    if mentioned:
        from ai.conversation_state import ASK_ABOUT_VIOLET_RE
        # the user asked about Violet's life: that is engagement, not Violet pushing her own topic
        violetdb.add_topic_mention(LIFE_TOPIC, "user" if ASK_ABOUT_VIOLET_RE.search(user_message or "") else "violet_reply",
                                user_engaged=1 if ASK_ABOUT_VIOLET_RE.search(user_message or "") else None)
    return mentioned


def cooling_topics_for_turn(recalled_memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Topics Violet should not raise herself right now, minus what this turn is about."""
    try:
        cooling = TopicTracker(violetdb).cooling_topics()
    except Exception:
        logger.exception("could not compute cooling topics")
        return []
    recalled_ids = {m.get("id") for m in recalled_memories or []}
    return [c for c in cooling if c.get("content") and c.get("memory_id") not in recalled_ids]


def format_core_memories(limit: int = 20) -> str:
    memories = violetdb.list_core_memories(limit=limit)
    if not memories:
        return "核心长期记忆：无"
    lines = ["核心长期记忆（长期用户画像和回复偏好）："]
    for memory in memories:
        lines.append(
            f"- id={memory['id']}; key={memory['memory_key']}; "
            f"category={memory['category']}; priority={memory['priority']}; "
            f"content={memory['content']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Robust JSON handling for agent outputs
# ---------------------------------------------------------------------------

def parse_json_object(raw: str | None) -> dict[str, Any] | None:
    """Parse the first JSON object in an LLM reply.

    Tolerates ```json fences and leading/trailing prose. Returns None when no
    JSON object can be recovered (callers decide the fallback).
    """
    text = (raw or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


# ---------------------------------------------------------------------------
# One turn of the owner's chat
# ---------------------------------------------------------------------------
# 2026-11-11 (audit, rework): the turn is  tools available -> Chat (with its tools) -> checks -> Memory.
# The Intent call, the Performer and the per-turn "tool plan" are gone: they had been off since 0.7.28 and only the
# tests still ran them. Chat decides itself what to look up or do; every action is one of its tools.

TOOL_NOTE_CHARS = 400


def tool_notes_from(messages: list[dict[str, Any]]) -> str | None:
    """What she looked up in this turn, kept with the chat (audit A1): the external tools and memory searches she
    called and the beginning of what came back. Without it the next turn saw only her words -- "那明天呢" after a
    weather search had nothing to go on, and she searched again or made it up."""
    calls: dict[str, tuple[str, str]] = {}
    lines: list[str] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant":
            for call in m.get("tool_calls") or []:
                fn = (call or {}).get("function") or {}
                calls[str(call.get("id"))] = (str(fn.get("name") or ""), str(fn.get("arguments") or ""))
        elif m.get("role") == "tool":
            name, args = calls.get(str(m.get("tool_call_id")), ("", ""))
            if "__" not in name and name != "query_memory":
                continue                                  # stickers, pictures, reminders show up in the chat itself
            text = str(m.get("content") or "")
            text = re.sub(r"【外部工具数据[^】]*】|【外部工具数据结束】", "", text).strip()
            text = re.sub(r"\s+", " ", text)
            lines.append(f"- {name.split('__', 1)[-1]}({args[:120]}) → {text[:TOOL_NOTE_CHARS]}")
    return "\n".join(lines[-4:]) or None


def rewrite_reply(chat_agent, user_message: str, chat_context: dict[str, Any], chat_result: dict[str, Any],
                  draft: str, note: str) -> str:
    """One corrected version of a reply, written WITHOUT tools (audit A2): the model sees its draft, what the tools
    of this turn returned and what is wrong with it. The old way ran the whole turn again -- tools included, so a
    like, a recall or a reminder could happen twice, and the picked sticker changed."""
    found = tool_notes_from(chat_result.get("messages") or [])
    result = chat_agent.run(user_message, call_context={
        **{k: v for k, v in chat_context.items() if not k.startswith("_")},
        "rewrite": {"draft": draft, "note": note, "tool_notes": found}})
    return result.get("content", "") or ""


def run_pipeline(
    user_message: str,
    chat_agent,
    memory_agent,
    defer_memory: bool = False,
    turn_id: str | None = None,
    turn_key: str | None = None,
    user_message_id: int | None = None,
) -> dict[str, Any]:
    """One user turn (see _run_pipeline). The turn is one tool scope -- an MCP server that fails in it is not tried
    again later in the same turn (ai.mcp.tool_turn)."""
    from ai.diagnostics import turn_scope
    from ai.mcp import tool_turn
    from ai.schedule.access import turn_message
    with turn_scope(turn_id), tool_turn(), turn_message(user_message):
        try:
            return _run_pipeline(user_message, chat_agent, memory_agent, defer_memory=defer_memory, turn_id=turn_id,
                                 turn_key=turn_key, user_message_id=user_message_id)
        except Exception as exc:
            _ps.pipeline_end(error=f"{exc.__class__.__name__}: {exc}")   # the 流程 page: not "running" forever
            raise


def _used_note_feeling(chat_result: dict[str, Any]) -> bool:
    """True when this chat turn successfully called note_feeling (so post-turn fallback must not double-shift)."""
    saw_tool_msg = False
    for m in chat_result.get("messages") or []:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "tool" and m.get("name") == "note_feeling":
            saw_tool_msg = True
            try:
                if json.loads(str(m.get("content") or "{}")).get("ok") is True:
                    return True
            except (TypeError, ValueError):
                return True
    if saw_tool_msg:
        return False
    if "note_feeling" in (chat_result.get("tools_called") or []):
        return True
    for m in chat_result.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if ((tc or {}).get("function") or {}).get("name") == "note_feeling":
                    return True
    return False


def _phone_line_for_chat() -> str:
    try:
        from ai import phone_state
        return phone_state.line("chat")
    except Exception:                                       # noqa: BLE001 -- the phone is optional
        logger.exception("phone state unavailable for this reply")
        return ""


def _run_pipeline(
    user_message: str,
    chat_agent,
    memory_agent,
    defer_memory: bool = False,
    turn_id: str | None = None,
    turn_key: str | None = None,
    user_message_id: int | None = None,
) -> dict[str, Any]:
    """Chat -> (checks) -> Memory. defer_memory=True: MemoryAgent runs later (ai/memory_jobs.py, after the reply went
    out). No stage makes the turn raise: Chat failing gives chat["failed"], Memory failing is skipped.

    Returns {"chat", "memory", "recalled_memories", "user_raised_memory_ids", "actions", "tool_notes", "errors",
    "chat_failed"}."""
    visualize_pipeline_start(user_message)
    effective_turn_id = turn_id or (None if defer_memory else f"sync-{uuid.uuid4().hex[:12]}")
    _ps.pipeline_start("chat", user_message[:60], turn_id=turn_id)
    errors: dict[str, str] = {}

    from ai.tool_manager import Catalog, abilities_note, direct_note, direct_tools, discover
    try:
        catalog = discover()
    except Exception:
        logger.exception("tool discovery failed; this turn runs without external tools")
        catalog = Catalog()
    from ai import trace
    trace.step("可用工具", 工具=sorted(catalog.tools) or None, 连不上=[u.get("server") for u in catalog.unavailable] or None,
               已停用=catalog.disabled or None, 未授权=catalog.refused or None)

    from ai import test_mode
    testing = test_mode.active()          # no memories, no topic bookkeeping, no persona checks
    recalled_memories: list[dict[str, Any]] = []
    try:
        # memories the owner's own words are about (a search by the code; she can search more with query_memory)
        from ai.recall_search import user_word_recall
        if not testing:
            recalled_memories = user_word_recall(violetdb, user_message, exclude_ids=[])
    except Exception:
        logger.exception("word recall failed; continuing without")
    try:
        user_raised = [] if testing else TopicTracker(violetdb).note_user_mentions(user_message, recalled_memories)
    except Exception:
        logger.exception("could not record user-raised topics")
        user_raised = []

    forget = None
    try:
        # "那件事你忘掉吧" is carried out here, before the reply, by the memory store itself
        from ai.memory_forget import forget_note, settle_forget_request
        forget = settle_forget_request(violetdb, user_message, effective_turn_id)
    except Exception:
        logger.exception("forget request check failed for turn %s", effective_turn_id)
    if forget is not None and forget["status"] == "forgotten":
        recalled_memories = [m for m in recalled_memories if m.get("id") != forget["memory_id"]]

    direct = direct_tools(catalog)
    decision_res: dict[str, Any] = {"ran": False, "reason": "testing" if testing else "skipped", "note": ""}
    if not testing:
        try:
            from ai import turn_decision
            run_dec, dec_reason = turn_decision.should_decide(violetdb, user_message)
            if run_dec:
                visualize_step("Turn Decision", "第一层回合决断（剧情进展与工具建议）")
                _ps.node_start("Decision", input_data={"user_message": user_message, "trigger_reason": dec_reason})
                decision_res = turn_decision.decide_turn(
                    violetdb,
                    user_message,
                    direct_tools=direct,
                    prechecked=(True, dec_reason),
                )
                _ps.node_done(
                    "Decision",
                    "Chat",
                    output_data={k: v for k, v in decision_res.items() if k != "note"},
                )
            else:
                decision_res = {"ran": False, "reason": dec_reason, "note": ""}
        except Exception as exc:                            # noqa: BLE001
            logger.debug("turn decision skipped: %s", exc)
            decision_res = {"ran": False, "reason": "error", "error": str(exc)[:120], "note": ""}

    visualize_step("Chat Generation", "生成回复消息")
    _ps.node_start("Chat", input_data={"user_message": user_message, "recalled_memories_count": len(recalled_memories)})
    chat_context = {
        "recalled_memories": recalled_memories,
        "restricted_topics": [] if testing else cooling_topics_for_turn(recalled_memories),
        "turn_id": turn_id,
        "turn_key": turn_key or turn_id,       # stable across crash recovery (inbound ids): idempotent images
        "forget_note": forget_note(forget) if forget is not None else None,
        "direct_tools": direct,                # the connected tools assigned to Chat: offered on every turn
        "tool_note": direct_note(direct),
        "abilities_note": abilities_note(catalog, direct),
        "phone_note": _phone_line_for_chat(),  # only with VIOLET_PHONE_CONTEXT_CHAT (default off)
        "decision_note": decision_res.get("note") or None,
    }
    actions: list[dict[str, Any]] = []
    tool_notes = None
    try:
        chat_result = chat_agent.run(user_message, call_context=chat_context)
        actions = list(chat_result.get("actions") or [])
        tool_notes = tool_notes_from(chat_result.get("messages") or [])
        known = {m.get("id") for m in recalled_memories}
        recalled_memories += [m for m in extract_tool_memories(chat_result) if m.get("id") not in known]
        raw_content = chat_result.get("content")
        is_empty_reply = (not isinstance(raw_content, str) or not raw_content.strip()) and not chat_result.get("image_asset_ids")
        if chat_result.get("failed") or is_empty_reply:
            finish_reason = chat_result.get("finish_reason")
            reason = str(chat_result.get("error") or (
                f"empty model response (finish_reason={finish_reason})" if finish_reason else "empty model response"
            ))
            errors["chat"] = reason
            chat_result = {**chat_result, "failed": True, "error": chat_result.get("error") or reason}
        else:
            # a link in the reply must come from this turn's tools, the user's words or the recent conversation
            from ai.web.citations import checked_reply
            recent = [r.get("content") for r in violetdb.get_recent_messages(limit=30)]
            chat_result = checked_reply(chat_result, user_message, recent)

            def used(server: str) -> bool:
                return any(isinstance(m, dict) and m.get("role") == "tool"
                           and str(m.get("content") or "").startswith(f"【外部工具数据｜来源 {server}/")
                           for m in chat_result.get("messages") or [])
            guard_flags = {
                "turn_used_web": used("violet-web"),
                "turn_sent_media": bool(chat_result.get("image_asset_ids")),
            }
            if used("violet-schedule"):
                guard_flags["turn_used_schedule"] = True
            first = chat_result
            guarded, violations = (chat_result.get("content", ""), []) if testing else guard_reply(
                chat_result.get("content", ""),
                regenerate=lambda note: rewrite_reply(
                    chat_agent, user_message, chat_context, first, first.get("content", ""),
                    "里面有和已记录的事实对不上的说法：" + note + "。没有记录的事不要说自己做过，计划不要说成已经完成，"
                    "不要编造家人、工作、学校等背景"),
                **guard_flags,
            )
            if violations:
                chat_result = {**chat_result, "content": guarded, "consistency_violations": [v.check for v in violations]}
                if not guarded.strip():
                    errors["chat"] = "consistency: every line contradicted recorded facts"
                    chat_result["failed"] = True
            if not chat_result.get("failed") and not testing:
                before = chat_result.get("content", "")
                content, follow_up = enforce_follow_up_preference(
                    user_message, before,
                    regenerate=lambda note: rewrite_reply(chat_agent, user_message, chat_context, first, before, note))
                if follow_up is not None:
                    chat_result = {**chat_result, "content": content, "follow_up_check": follow_up}
                if not _used_note_feeling(chat_result):
                    try:
                        from ai import mood
                        mood.owner_scold_fallback(violetdb, user_message, chat_result.get("content", ""))
                    except Exception:
                        logger.exception("owner scold mood fallback failed")
    except Exception as exc:
        logger.exception("ChatAgent failed")
        errors["chat"] = f"{exc.__class__.__name__}: {exc}"
        chat_result = {"content": "", "messages": [], "raw": {}, "error": errors["chat"], "failed": True}
    _ps.node_done("Chat", "Memory", output_data={"messages": chat_result.get("messages", []), "content": chat_result.get("content"),
                                                "usage": (chat_result.get("raw") or {}).get("usage"),
                                                "error": chat_result.get("error")})

    def result(memory_result: dict[str, Any]) -> dict[str, Any]:
        visualize_pipeline_end()
        _ps.pipeline_end(error="; ".join(f"{k}: {v}" for k, v in errors.items())[:500] if errors else None)
        return {"chat": chat_result, "memory": memory_result, "recalled_memories": recalled_memories,
                "user_raised_memory_ids": user_raised, "actions": actions, "tool_notes": tool_notes,
                "decision": {k: v for k, v in decision_res.items() if k != "note"},
                "errors": errors, "chat_failed": bool(chat_result.get("failed"))}

    if defer_memory or testing:
        if testing:
            _ps.node_skip("Memory", "测试模式不记")
        else:
            _ps.node_queue("Memory", "回复发出后在后台整理，结果会显示在这里")
        return result({"content": "", "messages": [], "raw": {}, "skipped": True, "deferred": True})

    visualize_step("Memory Processing", "处理记忆存储")
    from ai.memory_jobs import build_memory_input
    assistant_reply = chat_result.get("content") or ""
    memory_input = build_memory_input(user_message, assistant_reply)
    resolved_msg_id = user_message_id
    if resolved_msg_id is None and effective_turn_id:
        row = violetdb.fetch_one(
            "SELECT stored_message_id FROM inbound_messages WHERE turn_id = ? AND stored_message_id IS NOT NULL "
            "ORDER BY id DESC LIMIT 1",
            (effective_turn_id,),
        )
        if row and row.get("stored_message_id") is not None:
            resolved_msg_id = int(row["stored_message_id"])
    memory_context = {
        "turn_id": effective_turn_id,
        "user_message": user_message,
        "assistant_reply": assistant_reply,
        "user_message_id": resolved_msg_id,
    }
    _ps.node_start("Memory", input_data={"memory_input": memory_input})
    try:
        memory_result = memory_agent.run(memory_input, call_context=memory_context)
    except Exception as exc:
        logger.warning("MemoryAgent failed, continue without memory update: %s", exc)
        errors["memory"] = f"{exc.__class__.__name__}: {exc}"
        memory_result = {"content": json.dumps({"should_save": False, "reason": f"MemoryAgent failed: {exc}"},
                                               ensure_ascii=False), "messages": [], "raw": {}, "error": str(exc)}
    _ps.node_done("Memory", output_data={"messages": memory_result.get("messages", []), "content": memory_result.get("content")})
    if not chat_result.get("failed") and assistant_reply.strip():
        try:
            from ai import story_recorder
            story_recorder.after_turn(violetdb, user_message, assistant_reply)
        except Exception:                                   # noqa: BLE001
            logger.debug("sync story post-turn hook failed", exc_info=True)
    return result(memory_result)


def run_proactive_pipeline(
    event_context: dict[str, Any],
    chat_agent,
) -> dict[str, Any]:
    """Single-call proactive pipeline: ChatAgent decides and writes directly (or outputs [NO_REPLY]).

    Any unexpected error is recorded as proactive_events.status = 'failed'.
    """
    state: dict[str, Any] = {"event_id": None}
    try:
        return _run_proactive_pipeline(event_context, chat_agent, state)
    except Exception as exc:
        logger.exception("proactive pipeline failed")
        reason = f"pipeline error: {exc.__class__.__name__}: {exc}"[:500]
        try:
            if state["event_id"] is None:
                state["event_id"] = violetdb.add_proactive_event(
                    event_type=event_context.get("event_type", "scheduled_check"),
                    status="failed",
                    action_type="none",
                    reason=reason,
                    metadata=event_context,
                    scheduled_at=event_context.get("scheduled_at"),
                )
            else:
                violetdb.update_proactive_event(state["event_id"], status="failed", reason=reason)
        except Exception:
            logger.exception("could not record proactive failure")
        visualize_pipeline_end()
        _ps.pipeline_end(error=reason)
        return {
            "event_id": state["event_id"],
            "chat": None,
            "should_send": False,
            "error": reason,
        }


def _run_proactive_pipeline(
    event_context: dict[str, Any],
    chat_agent,
    state: dict[str, Any],
) -> dict[str, Any]:
    from ai.channels.outbound import chat_payload

    event_type = event_context.get("event_type", "scheduled_check")
    visualize_pipeline_start(f"[proactive] {event_type}")
    _ps.pipeline_start("proactive", event_type)

    conversation_state = event_context.get("conversation_state") or {}
    restricted = restricted_topics_from_state(conversation_state)
    phone_note = "\n".join(x for x in (event_context.get("phone_moment"), event_context.get("phone_now")) if x)

    visualize_step("Chat Generation", "判断并生成主动联系消息")
    _ps.node_start("Chat", input_data={"trigger": "[proactive event]", "event_type": event_type,
                                       "warmth_level": event_context.get("warmth_level")})
    chat_result = chat_agent.run(
        "[proactive event]",
        call_context={
            "is_proactive": True,
            "restricted_topics": restricted,
            "proactive_note": event_context.get("proactive_note"),
            "phone_note": phone_note,
        },
    )
    raw_content = chat_result.get("content")
    _ps.node_done("Chat", output_data={"messages": chat_result.get("messages", []), "content": raw_content or "",
                                       "error": chat_result.get("error")})

    is_empty_reply = not isinstance(raw_content, str) or not raw_content.strip()
    if chat_result.get("failed") or is_empty_reply:
        finish_reason = chat_result.get("finish_reason")
        reason = str(chat_result.get("error") or (
            f"empty model response (finish_reason={finish_reason})" if finish_reason else "empty model response"
        ))[:500]
        event_id = violetdb.add_proactive_event(
            event_type=event_type,
            status="failed",
            action_type="none",
            reason=reason,
            decision={"should_act": False, "action_type": "none", "reason": reason},
            metadata=event_context,
            scheduled_at=event_context.get("scheduled_at"),
        )
        state["event_id"] = event_id
        visualize_pipeline_end()
        _ps.pipeline_end(error=reason)
        return {"event_id": event_id, "chat": chat_result, "should_send": False, "error": reason}

    payload = chat_payload(raw_content, images=False)
    if not payload.get("should_send", False):
        reason = "ChatAgent decided not to send"
        event_id = violetdb.add_proactive_event(
            event_type=event_type,
            status="skipped",
            action_type="none",
            reason=reason,
            decision={"should_act": False, "action_type": "none", "reason": reason},
            metadata=event_context,
            scheduled_at=event_context.get("scheduled_at"),
        )
        state["event_id"] = event_id
        visualize_pipeline_end()
        _ps.pipeline_end()
        return {"event_id": event_id, "chat": chat_result, "should_send": False}

    guarded, life_violations = guard_reply(raw_content, entity="proactive_reply")
    if life_violations:
        chat_result = {**chat_result, "content": guarded}
        raw_content = guarded
        if not guarded.strip():
            reason = "policy: proactive message contradicted recorded facts"
            event_id = violetdb.add_proactive_event(
                event_type=event_type,
                status="skipped",
                action_type="none",
                reason=reason,
                decision={"should_act": False, "action_type": "none", "reason": reason},
                metadata=event_context,
                scheduled_at=event_context.get("scheduled_at"),
            )
            state["event_id"] = event_id
            visualize_pipeline_end()
            _ps.pipeline_end()
            return {"event_id": event_id, "chat": chat_result, "should_send": False, "error": reason}

    touched = find_touched_topic(raw_content, restricted)
    if touched is not None:
        ref = touched.get("memory_id") or touched.get("topic_key") or touched.get("content")
        reason = f"policy: generated message touches restricted topic (memory {ref})"
        logger.info(reason)
        event_id = violetdb.add_proactive_event(
            event_type=event_type,
            status="skipped",
            action_type="none",
            reason=reason,
            decision={"should_act": False, "action_type": "none", "reason": reason},
            metadata=event_context,
            scheduled_at=event_context.get("scheduled_at"),
        )
        state["event_id"] = event_id
        visualize_pipeline_end()
        _ps.pipeline_end()
        return {"event_id": event_id, "chat": chat_result, "should_send": False, "error": reason}

    summary_reason = event_context.get("summary_reason") or "主动联系"
    decision_record = {"should_act": True, "action_type": "chat", "reason": summary_reason,
                       "topic_key": event_context.get("summary_topic") or "proactive"}
    event_id = violetdb.add_proactive_event(
        event_type=event_type,
        status="planned",
        action_type="chat",
        reason=summary_reason,
        decision=decision_record,
        metadata=event_context,
        scheduled_at=event_context.get("scheduled_at"),
    )
    state["event_id"] = event_id
    visualize_pipeline_end()
    _ps.pipeline_end()
    return {
        "event_id": event_id,
        "chat": chat_result,
        "should_send": True,
    }


# ---------------------------------------------------------------------------
# Proactive policy helpers
# ---------------------------------------------------------------------------

WEB_DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "douban.com": ("豆瓣",),
    "thepaper.cn": ("澎湃",),
    "zhihu.com": ("知乎",),
    "bilibili.com": ("b站", "哔哩哔哩"),
    "weibo.com": ("微博",),
    "weibo.cn": ("微博",),
    "36kr.com": ("36氪",),
    "ithome.com": ("it之家",),
    "xinhuanet.com": ("新华社", "新华网"),
}


def _domain_mentioned(domain: str | None, reply_text: str) -> bool:
    dom = (domain or "").strip().lower()
    if not dom or not reply_text:
        return False
    low = reply_text.lower()
    if dom in low:
        return True
    for suffix, aliases in WEB_DOMAIN_ALIASES.items():
        if dom == suffix or dom.endswith("." + suffix):
            if any(alias in low for alias in aliases):
                return True
    return False


def _offered_text_overlap(reply_text: str, text: str, *, min_ratio: float = 0.10, min_hits: int = 2) -> bool:
    from ai.memory_model import RECALL_STOP_BIGRAMS, _bigrams
    target = _bigrams(text) - RECALL_STOP_BIGRAMS
    if not target:
        return False
    hits = target & (_bigrams(reply_text) - RECALL_STOP_BIGRAMS)
    if len(hits) >= max(3, min_hits):
        return True
    return len(hits) >= min_hits and (len(hits) / len(target)) >= min_ratio


def note_web_mentions(reply_text: str, used_finding_ids: list[int] | None = None) -> list[int]:
    """Findings Violet talked about in a sent reply count as shared (never offered proactively again).

    A finding used by this turn's web tools is shared when the reply names its site or talks about it;
    any other recent finding only when the reply clearly talks about its title.
    """
    from datetime import datetime, timedelta

    from ai.conversation_state import web_topic_key
    from ai.memory_model import mentions_topic, touches_topic
    from ai.web.findings import FindingsStore
    if not reply_text:
        return []
    store = FindingsStore(violetdb)
    since = (datetime.fromisoformat(violetdb.now()) - timedelta(days=3)).isoformat()
    used = {int(i) for i in used_finding_ids or []}
    shared = []
    for f in violetdb.fetch_all("SELECT * FROM web_findings WHERE status = 'active' AND updated_at >= ?", (since,)):
        if f["id"] in used:
            mentioned = (
                _domain_mentioned(f["domain"], reply_text)
                or mentions_topic(reply_text, f["title"])
                or _offered_text_overlap(reply_text, f["title"])
                or bool(f["summary"] and touches_topic(reply_text, f["summary"]))
            )
        else:
            mentioned = touches_topic(reply_text, f["title"])
        if mentioned:
            store.mark_shared(f["id"])
            violetdb.add_topic_mention(web_topic_key(f["id"]), "violet_reply")
            shared.append(f["id"])
    return shared


def restricted_topics_from_state(conversation_state: dict[str, Any]) -> list[dict[str, Any]]:
    restricted = [dict(m) for m in conversation_state.get("restricted_memories") or [] if m.get("content")]
    restricted += [dict(c) for c in conversation_state.get("cooling_topics") or [] if c.get("content")]
    return restricted


def find_touched_topic(text: str, restricted: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Last line of defence: block a proactive message that clearly talks about a restricted topic."""
    if not text or not env_bool("VIOLET_PROACTIVE_TOPIC_GUARD", True):
        return None
    from ai.memory_model import _bigrams
    text_grams = _bigrams(text)
    for topic in restricted:
        content = (topic.get("content") or "").strip()
        if not content:
            continue
        if touches_topic(text, content):
            return topic
        topic_grams = _bigrams(content)
        if len(topic_grams) == 1:
            if topic_grams & text_grams:
                return topic
        elif not topic_grams and re.fullmatch(r"[A-Za-z0-9]{2}", content):
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(content)}(?![A-Za-z0-9])", text, re.I):
                return topic
    return None


def spoken_reminder(content: str) -> str:
    """The reminder as said to the owner at its time. 2026-11-13 (0.7.73 virtual test): a record written for the
    system -- "明早叫用户一声（起床提醒）" -- went out word for word at 08:00 as "提醒你一下：明早叫用户一声（起床提醒）".
    The owner is "你", notes in brackets are not said, and "明早 / 明天" at the start is now."""
    text = re.sub(r"（[^）]*）|\([^)]*\)", "", content or "")
    text = text.replace("用户", "你").replace("对方", "你")
    text = re.sub(r"^(明早|明天早上|明天上午|明天下午|明天晚上|明晚|明天|今晚|今天)", "", text.strip()).strip(" ，,。")
    return text or (content or "").strip()


def run_commitment_pipeline(commitment: dict[str, Any], chat_agent) -> dict[str, Any]:
    """Generate the text of a due reminder the user explicitly asked for.

    The reminder must be delivered: if generation fails, declines ([NO_REPLY])
    or does not mention the reminded thing, a plain template is used instead
    (VIOLET_COMMITMENT_FALLBACK_TEMPLATE).
    """
    content = commitment["content"]
    decision = {
        "action_type": "commitment_reminder",
        "reason": "用户明确要求的提醒已到时间",
        "chat_brief": {"topic": f"按用户的要求提醒：{content}", "must_say": [content]},
        "commitment": {"content": content, "due_at": commitment.get("due_at"), "evidence": commitment.get("evidence")},
    }
    error = None
    text = ""
    try:
        result = chat_agent.run(
            "[commitment reminder]",
            call_context={"is_proactive": True, "proactive_result": decision, "commitment": commitment},
        )
        text = result.get("content", "")
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"
        logger.warning("commitment %s: chat generation failed: %s", commitment.get("id"), error)
    from ai.channels.outbound import chat_payload, fixed_text   # phase 10.6: sealed, sendable messages only
    payload = chat_payload(text if isinstance(text, str) else "", images=False)   # reminders never send images
    joined = "".join(m["content"] for m in payload.get("messages", []))
    said = spoken_reminder(content)
    mentions = any(VioletDatabase.normalize_text(c) in VioletDatabase.normalize_text(joined) or touches_topic(joined, c)
                   for c in dict.fromkeys((content, said)) if c)
    if payload.get("should_send") and mentions:
        return {"messages": payload["messages"], "used_fallback": False, "error": error}
    template = env_str("VIOLET_COMMITMENT_FALLBACK_TEMPLATE", "提醒你一下：{content}") or "提醒你一下：{content}"
    return {
        "messages": fixed_text(template.format(content=said)),
        "used_fallback": True,
        "error": error or ("generated text did not mention the reminder" if payload.get("should_send") else "no text generated"),
    }


def run_notice_pipeline(note: str, chat_agent, must_say: list[str] | None = None) -> dict[str, Any]:
    """2026-10-30 (user: the fixed texts did not sound like her): something the code has to tell the owner -- an
    approval's outcome, a request to approve -- written by the Chat agent in her own voice from this note. Falls back
    to the note itself when generation fails, declines, or leaves out a must-say piece (e.g. "同意 3").
    VIOLET_NOTICE_VOICE=0: always the plain note."""
    from ai.channels.outbound import chat_payload, fixed_text
    if not env_bool("VIOLET_NOTICE_VOICE", True) or chat_agent is None:
        return {"messages": fixed_text(note), "used_fallback": True, "error": None}
    decision = {"action_type": "notice", "reason": "系统要告诉主人一件事（不是用户消息）",
                "chat_brief": {"topic": note, "must_say": list(must_say or [])}}
    error, text = None, ""
    try:
        result = chat_agent.run("[notice]", call_context={
            "is_proactive": True, "proactive_result": decision, "commitment": None,
            "proactive_note": ("【要告诉主人的事】" + note + "\n用你平常和主人说话的方式说出来，一两条消息就够；"
                               "事实（做没做成、编号、结果）一个字都不能改，也不要添没发生的事。"
                               + ("其中必须原样包含：" + "、".join(must_say) if must_say else ""))})
        text = result.get("content", "")
    except Exception as exc:                                  # noqa: BLE001 -- the plain note still goes out
        error = f"{exc.__class__.__name__}: {exc}"
        logger.warning("notice: chat generation failed: %s", error)
    payload = chat_payload(text if isinstance(text, str) else "", images=False)
    joined = "".join(m.get("content") or "" for m in payload.get("messages", []))
    if payload.get("should_send") and all(piece in joined for piece in must_say or []):
        return {"messages": payload["messages"], "used_fallback": False, "error": error}
    return {"messages": fixed_text(note), "used_fallback": True, "error": error or "no usable text generated"}
