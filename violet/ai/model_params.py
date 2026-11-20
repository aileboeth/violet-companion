"""Per-model request parameter policy (phase 7, R-M1).

Different models / relays accept different parameters (e.g. some reasoning models
reject a non-default temperature or want max_completion_tokens instead of
max_tokens). Nothing is assumed here: config/model_params.json ships with
defaults that reproduce the current behaviour and NO model entries. An entry is
added only after a real call has shown what a model accepts.

Policy fields (defaults, overridable per model name):
  send_temperature      true  -> pass temperature; false -> leave it out
  max_tokens_field      "max_tokens" | "max_completion_tokens" | "none"
  allowed_extra_params  extra fields that may be sent (e.g. "reasoning_effort")
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from ai.settings import PROJECT_ROOT, env_str

logger = logging.getLogger("violet.model_params")
DEFAULTS = {"send_temperature": True, "max_tokens_field": "max_tokens", "allowed_extra_params": ["reasoning_effort"]}
EXTRA_PARAMS = ("reasoning_effort",)
_CACHE: dict[str, Any] = {"path": None, "mtime": None, "data": None}


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
    return {**data["defaults"], **{k: v for k, v in (data["models"].get(model or "") or {}).items() if not k.startswith("_")}}


def apply(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of chat.completions.create kwargs adjusted to the model's policy."""
    out = dict(kwargs)
    policy = policy_for(out.get("model"))
    if not policy.get("send_temperature", True):
        out.pop("temperature", None)
    if "max_tokens" in out:
        value = out.pop("max_tokens")
        field = policy.get("max_tokens_field", "max_tokens")
        if field in ("max_tokens", "max_completion_tokens"):
            out[field] = value
    allowed = set(policy.get("allowed_extra_params") or [])
    for name in EXTRA_PARAMS:
        if name in out and name not in allowed:
            out.pop(name)
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
#   other     nothing
# A relay that refuses one of these fields gets it no more (create() retries once without it).
PROVIDERS = ("auto", "openai", "grok", "deepseek", "claude", "other")
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
    return out


def _refused_field(exc: Exception, request: dict[str, Any]) -> str | None:
    """The cache field a relay refused (HTTP 400 naming it), else None."""
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status not in (400, 422):
        return None
    text = str(exc).lower()
    if "prompt_cache_key" in (request.get("extra_body") or {}) and (
        "prompt_cache_key" in text or "extra inputs" in text or "unrecognized" in text or "unknown field" in text
    ):
        return "prompt_cache_key"
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
