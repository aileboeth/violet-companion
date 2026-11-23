"""One model call of the social engine: the same address, key and client rules as the chat agents (offline tests use
BaseAgent.client_factory). The model is VIOLET_SOCIAL_MODEL, else the Chat model."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from ai.model_params import extract_json_object, sanitize_assistant_message, strip_think
from ai.settings import env_float, env_str


class SocialLLMError(RuntimeError):
    pass


def model_name() -> str:
    # 2026-11-02: the QQ channel's own Chat model (settings tab QQ) when there is no separate social model
    return env_str("VIOLET_SOCIAL_MODEL", "") or env_str("VIOLET_MODEL_QQ", "") or env_str("CHAT_LLM_MODEL", "")


def _client():
    import os
    from openai import OpenAI, Timeout
    from ai.agents.base_agent import BaseAgent, _local_address
    from ai.chat import LLMConfig, get_llm_api_url
    if BaseAgent.client_factory is not None:
        cfg = LLMConfig()
        cfg.reply_model = model_name()
        return BaseAgent.client_factory(cfg)
    url, key = get_llm_api_url(), os.getenv("LLM_API_KEY", "").strip()
    if not key:
        raise SocialLLMError("还没有设置模型服务的 API Key")
    extra = {}
    if _local_address(url):
        from openai import DefaultHttpxClient
        extra["http_client"] = DefaultHttpxClient(trust_env=False)
    return OpenAI(api_key=key, base_url=url, max_retries=1,
                  timeout=Timeout(env_float("LLM_REQUEST_TIMEOUT_SECONDS", 90.0), connect=10.0), **extra)


def parse_object(text: str | None) -> dict[str, Any]:
    if not text or not strip_think(text).strip():
        raise SocialLLMError("模型没有回答")
    try:
        return extract_json_object(text)
    except ValueError as exc:
        msg = str(exc)
        if "empty" in msg:
            raise SocialLLMError("模型没有回答") from None
        if "not a JSON object" in msg:
            raise SocialLLMError("模型的回答不是 JSON 对象") from None
        raise SocialLLMError("模型的回答不是 JSON") from None


def _stopped_for_length(choice: Any) -> bool:
    return str(getattr(choice, "finish_reason", "") or "").lower() in ("length", "max_tokens")


def decide(system: str, user: str | tuple[str, str], source: str = "social", tools=None,
           max_rounds: int = 4, allow_plain_text: bool = True) -> tuple[dict[str, Any], str, int]:
    """(the decision object, model, latency ms). Raises SocialLLMError. source: for the token ledger (ai/usage.py).
    tools (2026-10-30): an ai.social.tools.SocialTools -- the model may call them first, then answers with the JSON.
    user (2026-11-01): one text, or (what changes slowly, this decision) -- the first part ends the cached beginning.
    allow_plain_text (2026-11-23): True for chat decisions; False for structured JSON callers (qzone / qq_life)."""
    from openai import BadRequestError
    from ai.model_params import CACHE_MARK, create
    model = model_name()
    if not model:
        raise SocialLLMError("还没有设置 Chat 模型")
    started = time.monotonic()
    messages = [{"role": "system", "content": system, CACHE_MARK: True}]
    if isinstance(user, (tuple, list)):
        messages += [{"role": "user", "content": user[0], CACHE_MARK: True}, {"role": "user", "content": user[1]}]
    else:
        messages.append({"role": "user", "content": user})
    schemas = list(getattr(tools, "schemas", None) or [])
    from ai import usage
    try:
        client = _client()
        content = None
        for round_index in range(max_rounds + 1):
            request: dict[str, Any] = {
                "model": model, "messages": messages, "temperature": 0.8, "max_tokens": 2048,
                "cache_key": f"violet-{source}",
            }
            is_wrap_up = round_index == max_rounds
            if schemas and not is_wrap_up:
                request.update(tools=schemas, tool_choice="auto")
            elif schemas and is_wrap_up:
                request.update(tools=schemas, tool_choice="none")
            try:
                response = create(client, request)
            except BadRequestError:
                if not (schemas and is_wrap_up and "tools" in request):
                    raise
                bare = {k: v for k, v in request.items() if k not in ("tools", "tool_choice")}
                response = create(client, bare)
                request = bare
            usage.record(source, getattr(response, "model", None) or model, getattr(response, "usage", None))
            if not getattr(response, "choices", None):
                raise SocialLLMError("模型没有回答")
            choice = response.choices[0]
            message = choice.message
            calls = getattr(message, "tool_calls", None) or []
            # 2026-11-23 (P1-24): retry on length truncation BEFORE checking tool calls, and preserve
            # retry tool_calls so the tool loop still executes them on non-wrap-up rounds.
            if _stopped_for_length(choice):
                try:
                    retry_resp = create(client, {**request, "max_tokens": 4096})
                    usage.record(source, getattr(retry_resp, "model", None) or model, getattr(retry_resp, "usage", None))
                    if getattr(retry_resp, "choices", None):
                        choice = retry_resp.choices[0]
                        message = choice.message
                        calls = getattr(message, "tool_calls", None) or []
                except Exception:                     # noqa: BLE001 -- keep first response if retry fails
                    pass
            if is_wrap_up and "tools" in request and calls and not strip_think(message.content or "").strip():
                bare = {k: v for k, v in request.items() if k not in ("tools", "tool_choice")}
                response = create(client, bare)
                usage.record(source, getattr(response, "model", None) or model, getattr(response, "usage", None))
                if not getattr(response, "choices", None):
                    raise SocialLLMError("模型没有回答")
                choice = response.choices[0]
                message = choice.message
                calls = getattr(message, "tool_calls", None) or []
                request = bare
            if not calls or is_wrap_up or "tools" not in request:
                content = strip_think(message.content or "")
                break
            messages.append(sanitize_assistant_message(message))             # appended: the beginning stays cached
            for call in calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except ValueError:
                    args = {}
                result = tools.run(call.function.name, args if isinstance(args, dict) else {})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.function.name,
                    "content": result,
                })
            if round_index == max_rounds - 1:
                # a user-side note: a relay that hoists system messages would move a late one to the front
                messages.append({"role": "user", "content": "〔系统提示｜不是群里的消息〕工具已经用够了，现在按拿到的结果只输出 JSON。"})
    except SocialLLMError:
        raise
    except Exception as exc:                          # noqa: BLE001 -- the engine records it and goes on
        raise SocialLLMError(f"模型调用失败：{exc.__class__.__name__}") from None
    try:
        decision = parse_object(content)
    except SocialLLMError:
        decision = _plain_text_decision(content) if allow_plain_text else None
        if decision is None:
            raise
    return decision, model, int((time.monotonic() - started) * 1000)


def call_decide(system: str, user: str | tuple[str, str], **kwargs: Any) -> tuple[dict[str, Any], str, int]:
    """Call module-level decide(), dropping kwargs unsupported by 2-arg test fakes."""
    import inspect
    fn = decide
    if kwargs:
        try:
            sig = inspect.signature(fn)
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        except (TypeError, ValueError):
            kwargs = {}
    return fn(system, user, **kwargs)


def _plain_text_decision(content: str | None) -> dict[str, Any] | None:
    """2026-11-11 (eval on the user's server, persona Eve: "模型的回答不是 JSON" -> she said nothing when the owner
    called her in a group): a persona that is one whole prompt of its own often answers with plain lines instead of
    the JSON. Short plain text is what she wanted to say -- sent as that, never dropped; anything that looks like a
    broken JSON object is still an error."""
    text = strip_think(content).strip()
    if not text or len(text) > 600 or "{" in text or "}" in text:
        return None
    lines = [part.strip() for part in re.split(r"\n+|\|\|\|", text) if part.strip()]
    if not lines:
        return None
    return {"action": "say", "messages": lines[:3], "reason": "模型没有按 JSON 回答，按她写的文字处理"}
