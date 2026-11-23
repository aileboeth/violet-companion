"""Per-model request parameter policy (phase 7, R-M1).

Different models / relays accept different parameters (e.g. some reasoning models
reject a non-default temperature or want max_completion_tokens instead of
max_tokens). Nothing is assumed here: config/model_params.json ships with
defaults that reproduce the current behaviour and NO model entries. An entry is
added only after a real call has shown what a model accepts.

Policy fields (defaults, overridable per model name or glob pattern like "gemini-3*"):
  send_temperature         true  -> pass temperature; false -> leave it out
  default_temperature      optional float -> overrides temperature when send_temperature is true
  max_tokens_field         "max_tokens" | "max_completion_tokens" | "none"
  allowed_extra_params     extra fields that may be sent (e.g. "reasoning_effort")
  default_reasoning_effort optional string -> injected when caller omits reasoning_effort
  extra_body               optional dict -> merged into request extra_body (e.g. safety_settings)
"""

from __future__ import annotations

from fnmatch import fnmatchcase
import hashlib
import hmac
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

from ai.settings import PROJECT_ROOT, env_str

logger = logging.getLogger("violet.model_params")
DEFAULTS = {"send_temperature": True, "max_tokens_field": "max_tokens", "allowed_extra_params": ["reasoning_effort"]}
EXTRA_PARAMS = ("reasoning_effort",)
_CACHE: dict[str, Any] = {"path": None, "mtime": None, "data": None}
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.S | re.I)


# ------------------------------------------------------------------------------------------------ wire hygiene helpers

def strip_think(text: str | None) -> str:
    """Remove closed <think>...</think> blocks emitted by thinking relays; unclosed tags stay for leak guards."""
    if not text:
        return ""
    return _THINK_BLOCK_RE.sub("", text)


def extract_json_object(text: str | None) -> dict[str, Any]:
    """Parse a JSON object from model output after stripping closed <think> blocks.

    Uses json.JSONDecoder().raw_decode scanning from each '{' instead of greedy regex so stray braces
    in surrounding prose or thinking blocks cannot corrupt parsing. Raises ValueError on failure.
    """
    cleaned = strip_think(text).strip()
    if not cleaned:
        raise ValueError("empty output")
    if cleaned.startswith("```"):
        # Remove opening fence line (``` or ```json) and trailing ```
        lines = cleaned.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
        raise ValueError("output is not a JSON object")
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    pos = cleaned.find("{")
    saw_non_dict = False
    while pos != -1:
        try:
            value, _end = decoder.raw_decode(cleaned, pos)
            if isinstance(value, dict):
                return value
            saw_non_dict = True
        except ValueError:
            pass
        pos = cleaned.find("{", pos + 1)
    if saw_non_dict:
        raise ValueError("output is not a JSON object")
    raise ValueError("no JSON object in output")


def sanitize_assistant_message(message: Any) -> dict[str, Any]:
    """Whitelist fields appended back into chat_history from an assistant ChatCompletionMessage.

    Preserves role, content, extra_content, and tool_calls (including per-call extra_content /
    thought_signature required by Gemini 2.5/3) while stripping reasoning_content, annotations,
    refusal, audio, etc. that relays inject and strict endpoints reject on round 2.
    """
    if isinstance(message, dict):
        raw = dict(message)
    elif hasattr(message, "model_dump"):
        raw = message.model_dump(exclude_none=True)
    else:
        raw = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None),
            "tool_calls": getattr(message, "tool_calls", None),
            "extra_content": getattr(message, "extra_content", None),
        }
    out: dict[str, Any] = {"role": raw.get("role") or "assistant"}
    if raw.get("content") is not None:
        out["content"] = raw["content"]
    if raw.get("extra_content") is not None:
        out["extra_content"] = raw["extra_content"]
    if raw.get("thought_signature") is not None:
        out["thought_signature"] = raw["thought_signature"]
    raw_calls = raw.get("tool_calls")
    if raw_calls:
        cleaned_calls = []
        for tc in raw_calls:
            if isinstance(tc, dict):
                tc_raw = tc
            elif hasattr(tc, "model_dump"):
                tc_raw = tc.model_dump(exclude_none=True)
            else:
                fn_obj = getattr(tc, "function", None)
                tc_raw = {
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(fn_obj, "name", None),
                        "arguments": getattr(fn_obj, "arguments", ""),
                    } if fn_obj is not None else None,
                    "extra_content": getattr(tc, "extra_content", None),
                    "thought_signature": getattr(tc, "thought_signature", None),
                }
            entry: dict[str, Any] = {}
            if tc_raw.get("id") is not None:
                entry["id"] = tc_raw["id"]
            if tc_raw.get("type") is not None:
                entry["type"] = tc_raw["type"]
            fn_raw = tc_raw.get("function")
            if isinstance(fn_raw, dict):
                fn_clean: dict[str, Any] = {}
                if fn_raw.get("name") is not None:
                    fn_clean["name"] = fn_raw["name"]
                if fn_raw.get("arguments") is not None:
                    fn_clean["arguments"] = fn_raw["arguments"]
                entry["function"] = fn_clean
            if tc_raw.get("extra_content") is not None:
                entry["extra_content"] = tc_raw["extra_content"]
            if tc_raw.get("thought_signature") is not None:
                entry["thought_signature"] = tc_raw["thought_signature"]
            cleaned_calls.append(entry)
        out["tool_calls"] = cleaned_calls
    return out


def _strip_empty_tool_parameters(tools: list[Any]) -> list[Any]:
    """Omit empty OBJECT parameters ({'type': 'object', 'properties': {}}) that Gemini/relays reject with 400."""
    cleaned = []
    for tool in tools:
        if not isinstance(tool, dict):
            cleaned.append(tool)
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and "parameters" in fn:
            params = fn.get("parameters")
            if isinstance(params, dict) and params.get("type") == "object" and not params.get("properties"):
                new_fn = {k: v for k, v in fn.items() if k != "parameters"}
                cleaned.append({**tool, "function": new_fn})
                continue
        cleaned.append(tool)
    return cleaned


def policy_path() -> Path:
    override = env_str("VIOLET_MODEL_PARAMS_FILE", "")
    return Path(override) if override else PROJECT_ROOT / "config" / "model_params.json"


def _load() -> dict[str, Any]:
    path = policy_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {"defaults": DEFAULTS, "models": {}}
    if _CACHE["path"] == str(path) and _CACHE["mtime"] == mtime:
        return _CACHE["data"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.error("model parameter policy %s unreadable (%s): using defaults", path, exc)
        data = {}
    data = {"defaults": {**DEFAULTS, **(data.get("defaults") or {})}, "models": data.get("models") or {}}
    _CACHE.update(path=str(path), mtime=mtime, data=data)
    return data


def policy_for(model: str | None) -> dict[str, Any]:
    data = _load()
    models = data["models"]
    target = model or ""
    matched = models.get(target)
    if matched is None and target:
        target_lower = target.lower()
        candidates: list[tuple[int, int, dict[str, Any]]] = []
        for pattern, entry in models.items():
            if not isinstance(entry, dict) or ("*" not in pattern and "?" not in pattern):
                continue
            if fnmatchcase(target_lower, pattern.lower()):
                literal_len = len(pattern.replace("*", "").replace("?", ""))
                candidates.append((literal_len, len(pattern), entry))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            matched = candidates[0][2]
    return {**data["defaults"], **{k: v for k, v in (matched or {}).items() if not k.startswith("_")}}


def apply(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of chat.completions.create kwargs adjusted to the model's policy."""
    out = dict(kwargs)
    model = str(out.get("model") or "")
    policy = policy_for(model)
    if not policy.get("send_temperature", True):
        out.pop("temperature", None)
    elif policy.get("default_temperature") is not None:
        out["temperature"] = policy["default_temperature"]
    if "max_tokens" in out:
        value = out.pop("max_tokens")
        field = policy.get("max_tokens_field", "max_tokens")
        if field in ("max_tokens", "max_completion_tokens"):
            out[field] = value
    allowed = set(policy.get("allowed_extra_params") or [])
    default_effort = policy.get("default_reasoning_effort") or policy.get("reasoning_effort")
    if (default_effort and "reasoning_effort" not in out and "reasoning_effort" in allowed
            and (model, "reasoning_effort") not in _REFUSED):
        out["reasoning_effort"] = default_effort
    for name in EXTRA_PARAMS:
        if name in out and (name not in allowed or (model, name) in _REFUSED):
            out.pop(name)
    policy_extra_body = policy.get("extra_body")
    if isinstance(policy_extra_body, dict) and policy_extra_body:
        out["extra_body"] = {**policy_extra_body, **(out.get("extra_body") or {})}
    if isinstance(out.get("tools"), list):
        out["tools"] = _strip_empty_tool_parameters(out["tools"])
    return _cache_hints(out)


# ------------------------------------------------------------------------------------------------ prompt caching
# 2026-11-01 (token rework; user: GPT now, DeepSeek too, maybe Claude later -- "都做吧，可以选"). Every provider caches
# the same beginning of a prompt; callers order their prompts for that and mark where the unchanging part ends with
PLATFORM_KEYS = {"wechat_ilink": "VIOLET_MODEL_WECHAT", "napcat": "VIOLET_MODEL_QQ", "qqbot": "VIOLET_MODEL_QQ",
                 "onebot11": "VIOLET_MODEL_QQ", "telegram": "VIOLET_MODEL_TELEGRAM"}


def channel_chat_model(default: str, channel_type: str | None = None) -> str:
    """2026-11-02 (user: 不同渠道用不同的模型): the Chat model of the platform being talked on (VIOLET_MODEL_WECHAT /
    _QQ / _TELEGRAM, set on that channel's settings tab), else `default` (the Chat model)."""
    if channel_type is None:
        from ai.channels.context import current
        channel_type = current().channel_type
    key = PLATFORM_KEYS.get(channel_type or "")
    return (env_str(key, "").strip() if key else "") or default


# "_cache_break" on a message, plus "cache_key" on the request. What is sent depends on VIOLET_LLM_PROVIDER:
#   openai    prompt_cache_key (routes calls with the same beginning to the same cache)
#   grok      x-grok-conv-id + session_id headers & prompt_cache_key (xAI node affinity + relay sticky account)
#   claude    cache_control on the marked messages (at most 4) -- Claude caches only what is marked
#   deepseek  nothing: its cache is automatic
#   gemini    automatic implicit cache; sticky session headers on relays (omitted on direct googleapis.com)
#   other     nothing
# A relay that refuses one of these fields gets it no more (create() retries once without it).
PROVIDERS = ("auto", "openai", "grok", "deepseek", "claude", "gemini", "other")
CACHE_MARK = "_cache_break"
_GROK_NS = uuid.UUID("7f3b5c42-9a18-4d6e-b3c1-2e8a4f09d51c")
_REFUSED: set[tuple[str, str]] = set()             # (model, field) a relay answered 400 for


def _grok_conv_id(key: str) -> str:
    """Stable UUIDv5 per (installation, cache_key): keeps xAI backend affinity & relay sticky sessions across turns
    and restarts while preventing collisions between different Violet instances sharing the same relay."""
    salt = env_str("VIOLET_CACHE_INSTANCE_ID", "").strip()
    if not salt:
        seed = f"{os.getenv('LLM_API_KEY', '')}:{os.getenv('VIOLET_DATA_DIR', '')}".encode("utf-8")
        salt = hmac.new(b"violet-grok-cache-id", seed, hashlib.sha256).hexdigest()[:16]
    return str(uuid.uuid5(_GROK_NS, f"{salt}:{key}"))


def provider(model: str | None) -> str:
    chosen = env_str("VIOLET_LLM_PROVIDER", "auto").strip().lower()
    if chosen in PROVIDERS and chosen != "auto":
        return chosen
    name = (model or "").lower()
    if "claude" in name:
        return "claude"
    if "deepseek" in name:
        return "deepseek"
    if "grok" in name:
        return "grok"
    if "gemini" in name:
        return "gemini"
    if name.startswith(("gpt", "o1", "o3", "o4", "chatgpt")) or "/gpt" in name:
        return "openai"
    return "other"


LATE_SYSTEM_LABEL = "【系统给你的提示，不是对方说的话】\n"


def _late_system_as_user(messages: list[Any]) -> list[Any]:
    """2026-11-01 (--probe-cache on the user's relay: a system message after the conversation began had 0 cache hits,
    the same text inside a user message 7040 of 7250 -- the relay moves every system message to the front, so this
    turn's time / memories / notes changed the beginning of every call). After the first non-system message, system
    messages go as labelled user messages. (2026-11-11: a chat request has no such message any more -- one system
    message, the chat, one user message; this stays as the net under the other callers.)"""
    # the unchanging beginning ends at the first CACHE_MARK (ChatAgent marks it; with an empty history this turn's notes
    # follow the fixed system messages directly -- the second probe run: still 0%), else at the first non-system one
    out, started = [], False
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system" and started:
            content = m.get("content")
            if isinstance(content, str):
                m = {**m, "role": "user", "content": LATE_SYSTEM_LABEL + content}
            elif isinstance(content, list):
                m = {**m, "role": "user", "content": [{"type": "text", "text": LATE_SYSTEM_LABEL.strip()}, *content]}
        elif isinstance(m, dict) and (m.get("role") != "system" or m.get(CACHE_MARK)):
            started = True
        out.append(m)
    return out


def _merge_consecutive_roles(messages: list[Any]) -> list[Any]:
    """Merge adjacent string-content system or user messages for strict Gemini role alternation."""
    merged: list[Any] = []
    for m in messages:
        if (merged and isinstance(m, dict) and isinstance(merged[-1], dict)
                and m.get("role") in ("system", "user")
                and merged[-1].get("role") == m.get("role")
                and isinstance(merged[-1].get("content"), str)
                and isinstance(m.get("content"), str)):
            prev_text = merged[-1]["content"]
            next_text = m["content"]
            combined = f"{prev_text}\n\n{next_text}" if prev_text and next_text else (prev_text or next_text)
            merged[-1] = {**merged[-1], "content": combined}
        else:
            merged.append(m)
    return merged


def _gemini_sticky_enabled() -> bool:
    override = env_str("VIOLET_GEMINI_STICKY_HEADERS", "").strip().lower()
    if override in ("0", "false", "no", "off"):
        return False
    if override in ("1", "true", "yes", "on"):
        return True
    url = env_str("LLM_API_URL", "").strip().lower()
    return not url.startswith("https://generativelanguage.googleapis.com")


def _cache_hints(out: dict[str, Any]) -> dict[str, Any]:
    model = str(out.get("model") or "")
    kind = provider(model)
    key = out.pop("cache_key", None)
    if isinstance(out.get("messages"), list):
        out["messages"] = _late_system_as_user(out["messages"])
    messages = out.get("messages")
    if isinstance(messages, list) and any(isinstance(m, dict) and CACHE_MARK in m for m in messages):
        marks = 0
        claude = kind == "claude" and (model, "cache_control") not in _REFUSED
        cleaned = []
        for m in messages:
            if isinstance(m, dict) and CACHE_MARK in m:
                m = {k: v for k, v in m.items() if k != CACHE_MARK}
                if claude and marks < 4 and isinstance(m.get("content"), str) and m["content"]:
                    m["content"] = [{"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}]
                    marks += 1
            cleaned.append(m)
        out["messages"] = cleaned
    if kind == "gemini" and isinstance(out.get("messages"), list):
        out["messages"] = _merge_consecutive_roles(out["messages"])
    if key and kind == "openai" and (model, "prompt_cache_key") not in _REFUSED:
        out["extra_body"] = {**(out.get("extra_body") or {}), "prompt_cache_key": str(key)[:64]}
    elif key and kind == "grok":
        # 2026-11-20 (live probe on sub2api + grok-4.6): x-grok-conv-id alone missed (relay spread across upstream
        # accounts), prompt_cache_key alone missed (xAI Chat Completions routes backend KV cache by x-grok-conv-id);
        # sending both raised cache hit rate from 5% (512 global base) to 97-99%. If a direct xAI endpoint rejects
        # prompt_cache_key in the body, _refused_field drops it and keeps the headers.
        conv_id = _grok_conv_id(str(key))
        out["extra_headers"] = {
            "x-grok-conv-id": conv_id,
            "session_id": conv_id,
            "conversation_id": conv_id,
            "x-session-id": conv_id,
            **(out.get("extra_headers") or {}),
        }
        if (model, "prompt_cache_key") not in _REFUSED:
            out["extra_body"] = {**(out.get("extra_body") or {}), "prompt_cache_key": conv_id}
    elif key and kind == "gemini" and _gemini_sticky_enabled():
        conv_id = _grok_conv_id(str(key))
        out["extra_headers"] = {
            "session_id": conv_id,
            "x-session-id": conv_id,
            **(out.get("extra_headers") or {}),
        }
    return out


def _refused_field(exc: Exception, request: dict[str, Any]) -> str | None:
    """The cache/extra field a relay refused (HTTP 400 naming it), else None."""
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status not in (400, 422):
        return None
    text = str(exc).lower()
    if "prompt_cache_key" in (request.get("extra_body") or {}) and (
        "prompt_cache_key" in text or "extra inputs" in text or "unrecognized" in text or "unknown field" in text
    ):
        return "prompt_cache_key"
    if "reasoning_effort" in request and (
        "reasoning_effort" in text or "unsupported parameter" in text or "unknown field" in text
    ):
        return "reasoning_effort"
    sent_marks = any(isinstance(m, dict) and isinstance(m.get("content"), list) and
                     any(isinstance(p, dict) and "cache_control" in p for p in m["content"])
                     for m in request.get("messages") or [])
    if sent_marks and ("cache_control" in text or "content" in text):
        return "cache_control"
    return None


def create(client, kwargs: dict[str, Any]):
    """client.chat.completions.create with the model's policy and the cache hints; a refused hint is dropped for
    good and the call made once more without it."""
    request = apply(kwargs)
    try:
        return client.chat.completions.create(**request)
    except Exception as exc:
        refused = _refused_field(exc, request)
        if refused is None:
            raise
        _REFUSED.add((str(request.get("model") or ""), refused))
        logger.warning("model %s: the relay refused %s; it is not sent any more", request.get("model"), refused)
        return client.chat.completions.create(**apply(kwargs))
