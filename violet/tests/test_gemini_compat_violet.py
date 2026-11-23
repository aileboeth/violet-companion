"""Gemini OpenAI-compatibility wire-level tests (W-P0-1..8, W-P1-1..6, W-P2-1/3/4/5).

Covers every wire fix offline (real openai==3.17.0 SDK + httpx.MockTransport + FakeLLM + real FastMCP servers):
  - No `"tools": null` / `"tool_choice": null` on tool-less calls
  - Recursive MCP schema sanitization (FastMCP violet-web / violet-schedule + $defs/$ref/anyOf+null/const)
  - Zero-arg tool schemas omitting empty `parameters` + optional-only `"required": []` + defensive strip in model_params
  - `"name"` on every `role: "tool"` message (BaseAgent & SocialLLM)
  - `max_tool_rounds` wrap-up sending `tools` + `tool_choice="none"` with 400 fallback
  - Thinking-model `max_tokens` ceilings + `finish_reason="length"` retries
  - Empty `choices=[]` retry + fallback_model, empty/safety content returning `failed=True` & `finish_reason`
  - GIF first-frame -> PNG conversion in vision.py + per-sticker continue vs service-wide break in stickers.py
  - Assistant history field whitelist preserving `extra_content.google.thought_signature` while stripping `reasoning_content`/`annotations`
  - Closed `<think>...</think>` stripping & `raw_decode` JSON parsing (unclosed `<think>` still caught by leak guard)
  - `"gemini"` provider, wildcard policy (`gemini-3*`, `default_temperature`, `default_reasoning_effort`, `extra_body`),
    MemoryAgent single system/user block, `usage.reasoning_tokens`, `list_models` stripping `models/`, and `VIOLET_LLM_STREAM=1`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import zlib
from types import SimpleNamespace

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion

from tests.fakes import FakeLLM, completion, default_responders, install_fake_llm


def _bad_request(message: str = "Bad Request", status: int = 400) -> openai.BadRequestError:
    resp = httpx.Response(status, request=httpx.Request("POST", "https://relay.example/v1/chat/completions"))
    return openai.BadRequestError(message, response=resp, body={"error": {"message": message}})


def _make_valid_gif(width: int = 2, height: int = 2, *, transparent: bool = False) -> bytes:
    """Build a valid minimal GIF89a image with known RGB pixels (2x2: red, green, blue, white)."""
    # Global color table: 4 entries (size field = 1 -> 2^(1+1) = 4 colors)
    # 0=red(255,0,0), 1=green(0,255,0), 2=blue(0,0,255), 3=white(255,255,255)
    gct = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
    header = b"GIF89a" + struct.pack("<HHBBB", width, height, 0x80 | 0x01, 0, 0) + gct
    gce = b""
    if transparent:
        # Graphic Control Extension: transparent index = 0
        gce = b"\x21\xF9\x04\x01\x00\x00\x00\x00"
    img_desc = b"\x2C" + struct.pack("<HHHHB", 0, 0, width, height, 0x00)
    # LZW min_code_size = 2 (clear=4, eoi=5). Code width starts at 3, bumps to 4 after 3rd pixel (table size 8).
    codes_and_widths = [(4, 3), (0, 3), (1, 3), (2, 3), (3, 4), (5, 4)]
    bit_buf = 0
    bit_cnt = 0
    out = bytearray()
    for c, w_bits in codes_and_widths:
        bit_buf |= c << bit_cnt
        bit_cnt += w_bits
        while bit_cnt >= 8:
            out.append(bit_buf & 0xFF)
            bit_buf >>= 8
            bit_cnt -= 8
    if bit_cnt:
        out.append(bit_buf & 0xFF)
    lzw = bytes([2, len(out)]) + bytes(out) + b"\x00"
    return header + gce + img_desc + lzw + b"\x3B"


def _decode_png_rgba(png_bytes: bytes) -> tuple[int, int, bytes]:
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    pos = 8
    w = h = 0
    idat = bytearray()
    while pos < len(png_bytes):
        length = struct.unpack_from(">I", png_bytes, pos)[0]
        tag = png_bytes[pos + 4:pos + 8]
        payload = png_bytes[pos + 8:pos + 8 + length]
        crc = struct.unpack_from(">I", png_bytes, pos + 8 + length)[0]
        assert (zlib.crc32(tag + payload) & 0xFFFFFFFF) == crc
        pos += 12 + length
        if tag == b"IHDR":
            w, h, bit_depth, color_type = struct.unpack_from(">IIBB", payload, 0)
            assert bit_depth == 8 and color_type == 6
        elif tag == b"IDAT":
            idat.extend(payload)
        elif tag == b"IEND":
            break
    raw = zlib.decompress(bytes(idat))
    rgba = bytearray()
    stride = w * 4
    for y in range(h):
        row_start = y * (stride + 1)
        assert raw[row_start] == 0
        rgba.extend(raw[row_start + 1:row_start + 1 + stride])
    return w, h, bytes(rgba)


# ---------------------------------------------------------------------------
# W-P0-1: Never serialize "tools": null or "tool_choice": null
# ---------------------------------------------------------------------------

def test_w_p0_1_toolless_call_omits_tools_and_tool_choice_on_wire(monkeypatch):
    from ai.agents.base_agent import BaseAgent

    captured_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={
            "id": "cmpl-wire-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gemini-2.5-flash",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "你好呀"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        })

    def factory(cfg):
        return openai.OpenAI(
            api_key="test-key",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            max_retries=0,
        )

    monkeypatch.setattr(BaseAgent, "client_factory", factory)
    monkeypatch.setenv("CHAT_LLM_MODEL", "gemini-2.5-flash")

    class BareAgent(BaseAgent):
        model_env_key = "CHAT_LLM_MODEL"

    res = BareAgent().run("早安")
    assert res["content"] == "你好呀" and res["failed"] is False and res["finish_reason"] == "stop"
    assert len(captured_bodies) == 1
    body = captured_bodies[0]
    assert "tools" not in body and "tool_choice" not in body


# ---------------------------------------------------------------------------
# W-P0-2 & W-P0-3 & W-P2-5: Schema sanitization & zero-arg tools
# ---------------------------------------------------------------------------

def _assert_gemini_clean_schema(node, path="root"):
    banned = {"anyOf", "oneOf", "default", "title", "$defs", "definitions", "$ref", "additionalProperties", "$schema"}
    if isinstance(node, dict):
        for k in banned:
            assert k not in node, f"{path} still contains forbidden key {k!r}: {node}"
        assert node.get("type") != "null", f"{path} has null type"
        assert not isinstance(node.get("type"), list), f"{path} has list type"
        for k, v in node.items():
            _assert_gemini_clean_schema(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _assert_gemini_clean_schema(item, f"{path}[{i}]")


def test_w_p0_2_real_fastmcp_schemas_sanitized_for_gemini():
    from ai.mcp import tool_schema_to_openai_tool
    from ai.web.mcp_server import build_default_server as build_web
    from ai.schedule.mcp_server import build_default_server as build_sched

    async def collect():
        out = []
        for label, srv in (("violet-web", build_web()), ("violet-schedule", build_sched())):
            for t in await srv.list_tools():
                out.append(tool_schema_to_openai_tool(f"{label}__{t.name}", t.description, t.inputSchema))
        return out

    tools = asyncio.run(collect())
    assert len(tools) >= 6
    by_name = {t["function"]["name"]: t["function"] for t in tools}
    for name, fn in by_name.items():
        _assert_gemini_clean_schema(fn.get("parameters"), name)
    ws_props = by_name["violet-web__web_search"]["parameters"]["properties"]
    assert ws_props["recency_days"]["type"] == "integer"
    assert ws_props["limit"]["type"] == "integer"
    sched_props = by_name["violet-schedule__get_day_schedule"]["parameters"]["properties"]
    assert sched_props["details"] == {"type": "array", "items": {"type": "string"}}


def test_w_p0_2_synthetic_third_party_mcp_schema_sanitization():
    from ai.mcp import tool_schema_to_openai_tool

    raw = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ComplexInput",
        "type": "object",
        "additionalProperties": False,
        "$defs": {
            "Filter": {
                "type": "object",
                "title": "Filter",
                "properties": {
                    "tag": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "title": "Tag"},
                    "mode": {"const": "fast"},
                },
                "additionalProperties": False,
            },
            "Loop": {"$ref": "#/$defs/Loop"},
        },
        "properties": {
            "filter": {"$ref": "#/$defs/Filter", "description": "Filter criteria"},
            "nullable_str": {"type": ["string", "null"], "default": None},
            "tags": {"type": "array"},
            "loop": {"$ref": "#/$defs/Loop"},
        },
        "required": ["filter", "nonexistent_field"],
    }
    tool = tool_schema_to_openai_tool("ext__search", "desc", raw)
    params = tool["function"]["parameters"]
    _assert_gemini_clean_schema(params)
    assert params["properties"]["filter"]["properties"]["tag"] == {"type": "string"}
    assert params["properties"]["filter"]["properties"]["mode"] == {"enum": ["fast"], "type": "string"}
    assert params["properties"]["nullable_str"] == {"type": "string"}
    assert params["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
    assert params["required"] == ["filter"]

    # Zero-arg MCP tool omits "parameters"
    zero = tool_schema_to_openai_tool("phone__battery", "Battery level", {"type": "object", "properties": {}})
    assert "parameters" not in zero["function"]


def test_w_p0_3_and_p2_5_zero_arg_and_optional_local_tools_and_defensive_strip():
    from ai import model_params as mp
    from ai import qq_tools, sticker_tools, tool_schemas

    assert "parameters" not in tool_schemas.LIST_ACTIVE_TASKS_SCHEMA["function"]

    qq_by_name = {s["function"]["name"]: s["function"] for s in qq_tools.SCHEMAS}
    assert "parameters" not in qq_by_name["qq_like"]
    assert qq_by_name["qq_recall"]["parameters"]["required"] == []

    for s in sticker_tools.SCHEMAS:
        assert s["function"]["parameters"]["required"] == []

    # Wire-level enforcement via model_params.apply(): strips empty OBJECT parameters on memory_done & any raw tool
    # without mutating the caller's original schema dicts.
    applied_done = mp.apply({
        "model": "gemini-2.5-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [tool_schemas.MEMORY_DONE_SCHEMA],
    })
    assert "parameters" not in applied_done["tools"][0]["function"]
    assert "parameters" in tool_schemas.MEMORY_DONE_SCHEMA["function"]  # original not mutated

    raw_tool = {"type": "function", "function": {"name": "ping", "description": "p",
                                                 "parameters": {"type": "object", "properties": {}}}}
    applied = mp.apply({"model": "gemini-2.5-pro", "messages": [{"role": "user", "content": "hi"}], "tools": [raw_tool]})
    assert "parameters" not in applied["tools"][0]["function"]
    assert "parameters" in raw_tool["function"]  # original not mutated


# ---------------------------------------------------------------------------
# W-P0-4, W-P0-5, W-P1-1: tool name, max_tool_rounds wrap-up, thought_signature
# ---------------------------------------------------------------------------

def test_w_p0_4_p1_1_base_agent_tool_name_whitelist_and_thought_signature(monkeypatch):
    from ai.agents.base_agent import BaseAgent

    sig = "CiQBq9J2_thought_signature_token_xyz"
    first_completion = ChatCompletion.model_validate({
        "id": "cmpl-t1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gemini-3-pro",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning_content": "secret chain of thought that must NOT bleed back",
                "annotations": [],
                "refusal": None,
                "extra_content": {"google": {"msg_sig": "ms1"}},
                "tool_calls": [{
                    "id": "call_sig_1",
                    "type": "function",
                    "function": {"name": "lookup_weather", "arguments": '{"city": "上海"}'},
                    "extra_content": {"google": {"thought_signature": sig}},
                }],
            },
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    })
    second_completion = completion("上海晴天 22 度", model="gemini-3-pro")
    replies = iter([first_completion, second_completion])
    fake = install_fake_llm(monkeypatch, FakeLLM(default=lambda kw: next(replies)))

    class WeatherAgent(BaseAgent):
        model_env_key = "CHAT_LLM_MODEL"

        def register_tools(self):
            self.register_tool(
                {"type": "function", "function": {
                    "name": "lookup_weather", "description": "weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                }},
                lambda args: f"{args['city']}晴",
            )

    monkeypatch.setenv("CHAT_LLM_MODEL", "gemini-3-pro")
    res = WeatherAgent().run("上海天气")
    assert res["content"] == "上海晴天 22 度" and res["failed"] is False
    assert len(fake.calls) == 2

    round2_msgs = fake.calls[1]["messages"]
    asst_msg = next(m for m in round2_msgs if m.get("role") == "assistant")
    assert "reasoning_content" not in asst_msg
    assert "annotations" not in asst_msg
    assert "refusal" not in asst_msg
    assert asst_msg["extra_content"] == {"google": {"msg_sig": "ms1"}}
    assert asst_msg["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": sig}}

    tool_msg = next(m for m in round2_msgs if m.get("role") == "tool")
    assert tool_msg["name"] == "lookup_weather"
    assert tool_msg["tool_call_id"] == "call_sig_1"


def test_w_p0_5_base_agent_wrapup_passes_tools_none_and_recovers_on_400(monkeypatch):
    from ai.agents.base_agent import BaseAgent

    class WrapAgent(BaseAgent):
        model_env_key = "CHAT_LLM_MODEL"
        answer_after_max_rounds = True

        def register_tools(self):
            self.register_tool(
                {"type": "function", "function": {"name": "ping", "description": "ping"}},
                lambda args: "pong",
            )

    monkeypatch.setenv("CHAT_LLM_MODEL", "gemini-2.5-pro")

    # Case 1: Gemini honors tool_choice="none" on wrap-up while keeping tools list
    def responder_gemini(kw):
        if kw.get("tool_choice") == "none":
            assert kw.get("tools"), "tools must stay present on Gemini wrap-up call"
            return "总结完了"
        return completion(tool_calls=[{"name": "ping", "arguments": {}}], model="gemini-2.5-pro")

    fake1 = install_fake_llm(monkeypatch, FakeLLM(default=responder_gemini))
    res1 = WrapAgent().run("ping", max_tool_rounds=1)
    assert res1["content"] == "总结完了" and res1["max_rounds_reached"] is True and res1["failed"] is False
    assert fake1.calls[-1]["tool_choice"] == "none" and fake1.calls[-1]["tools"]

    # Case 2: Relay rejects tool_choice="none" with 400 -> retries without tools/tool_choice
    def responder_strict_relay(kw):
        if kw.get("tool_choice") == "none":
            raise _bad_request("tool_choice='none' not supported")
        if not kw.get("tools"):
            return "降级总结"
        return completion(tool_calls=[{"name": "ping", "arguments": {}}], model="gemini-2.5-pro")

    fake2 = install_fake_llm(monkeypatch, FakeLLM(default=responder_strict_relay))
    res2 = WrapAgent().run("ping", max_tool_rounds=1)
    assert res2["content"] == "降级总结" and res2["max_rounds_reached"] is True
    assert "tools" not in fake2.calls[-1] and "tool_choice" not in fake2.calls[-1]


def test_w_p0_4_p0_5_p0_6_p1_1_social_llm_wire_compat(monkeypatch):
    from ai.social import llm as social_llm

    monkeypatch.setenv("CHAT_LLM_MODEL", "gemini-2.5-flash")
    monkeypatch.delenv("VIOLET_SOCIAL_MODEL", raising=False)
    monkeypatch.delenv("VIOLET_MODEL_QQ", raising=False)

    sig = "social_thought_sig_abc"

    class DummyTools:
        schemas = [{"type": "function", "function": {
            "name": "web_search", "description": "search",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        }}]

        @staticmethod
        def run(name, args):
            return f"result:{name}:{args.get('q')}"

    # Test 1: tool round + thought_signature + role=tool name + wrap-up tool_choice="none" + max_tokens=2048
    def responder(kw):
        if kw.get("tool_choice") == "none":
            assert kw["max_tokens"] == 2048
            return '<think>hmm {not json}</think>\n{"action": "say", "messages": ["查到了"], "reason": "ok"}'
        return ChatCompletion.model_validate({
            "id": "cmpl-s1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gemini-2.5-flash",
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "hidden thoughts",
                    "annotations": [],
                    "tool_calls": [{
                        "id": "sc1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"q": "cat"}'},
                        "extra_content": {"google": {"thought_signature": sig}},
                    }],
                },
            }],
        })

    fake = install_fake_llm(monkeypatch, FakeLLM(default=responder))
    decision, _model, _ms = social_llm.decide("sys", ("slow_part", "fast_part"), tools=DummyTools(), max_rounds=1)
    assert decision["action"] == "say" and decision["messages"] == ["查到了"]

    # Gemini merged the 2 consecutive user messages
    round1_roles = [m["role"] for m in fake.calls[0]["messages"]]
    assert round1_roles == ["system", "user"]

    # Round 2 has whitelisted assistant message + tool name + tools with tool_choice="none"
    wrap_call = fake.calls[1]
    assert wrap_call["tool_choice"] == "none" and wrap_call["tools"]
    asst = next(m for m in wrap_call["messages"] if m["role"] == "assistant")
    assert "reasoning_content" not in asst and "annotations" not in asst
    assert asst["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": sig}}
    tool_msg = next(m for m in wrap_call["messages"] if m["role"] == "tool")
    assert tool_msg["name"] == "web_search"

    # Test 2: finish_reason == "length" retries once with 4096
    length_calls = []

    def length_responder(kw):
        length_calls.append(kw["max_tokens"])
        if len(length_calls) == 1:
            return ChatCompletion.model_validate({
                "id": "cmpl-len1", "object": "chat.completion", "created": 1700000000, "model": "gemini-2.5-flash",
                "choices": [{"index": 0, "finish_reason": "length",
                             "message": {"role": "assistant", "content": '{"action": "sa'}}],
            })
        return '{"action": "silent", "reason": "ok"}'

    install_fake_llm(monkeypatch, FakeLLM(default=length_responder))
    decision2, _, _ = social_llm.decide("sys", "user", max_rounds=1)
    assert decision2 == {"action": "silent", "reason": "ok"}
    assert length_calls == [2048, 4096]


# ---------------------------------------------------------------------------
# W-P0-6 & W-P1-2 & W-P2-3: max_tokens, <think> stripping, raw_decode, list_models
# ---------------------------------------------------------------------------

def test_w_p0_6_p1_2_aux_llm_semantic_guard_life_proposer_and_explorer(violetdb, monkeypatch):
    from ai.aux_llm import AuxLLM, AuxLLMConfig, parse_json_object
    from ai.life_proposer_llm import DEFAULT_LLM
    from ai.semantic_guard import SemanticGuard
    from ai.social.llm import parse_object

    assert AuxLLMConfig(purpose="p", model="m").max_output_tokens == 1024
    assert DEFAULT_LLM["max_output_tokens"] == 1024
    sg = SemanticGuard(violetdb)
    assert sg.llm.config.max_output_tokens == 1024 and sg.llm.config.timeout_seconds == 15.0

    # raw_decode + <think> stripping ignores braces in <think> and trailing text
    tricky = '<think>let me check {bad json} first</think>\nSure, here it is: {"ok": true, "n": 2}\ntrailing {ignored}'
    assert parse_json_object(tricky) == {"ok": True, "n": 2}
    assert parse_object(tricky) == {"ok": True, "n": 2}

    # AuxLLM lifts low max_output_tokens for gemini and retries on finish_reason="length"
    seen_max = []

    def resp(kw):
        seen_max.append(kw["max_tokens"])
        if len(seen_max) == 1:
            return ChatCompletion.model_validate({
                "id": "cmpl-aux-1", "object": "chat.completion", "created": 1700000000, "model": kw["model"],
                "choices": [{"index": 0, "finish_reason": "length",
                             "message": {"role": "assistant", "content": "<think>long thinking</think>"}}],
            })
        return '<think>done</think>\n{"index": 0}'

    install_fake_llm(monkeypatch, FakeLLM(default=resp))
    aux = AuxLLM(violetdb, AuxLLMConfig(purpose="web_explorer", model="gemini-2.5-pro",
                                        max_output_tokens=150, max_attempts=1))
    assert aux.complete_json([{"role": "user", "content": "pick"}]) == {"index": 0}
    assert seen_max == [1024, 2048]


def test_w_p2_3_setup_status_list_models_strips_models_prefix():
    from ai.setup_status import list_models

    class FakeModelsClient:
        class models:
            @staticmethod
            def list():
                return SimpleNamespace(data=[
                    SimpleNamespace(id="models/gemini-2.5-pro"),
                    SimpleNamespace(id="models/gemini-3-flash"),
                    {"id": "google/gemini-2.5-flash"},
                    SimpleNamespace(id="gpt-4o"),
                ])

    agent = SimpleNamespace(_config=SimpleNamespace(), _get_client=lambda cfg: FakeModelsClient())
    res = list_models(agent)
    assert res["ok"] is True
    assert res["models"] == ["gemini-2.5-pro", "gemini-3-flash", "google/gemini-2.5-flash", "gpt-4o"]


# ---------------------------------------------------------------------------
# W-P0-7 & W-P1-2: Empty choices -> retry/fallback; empty/safety content -> failed=True; <think> handling
# ---------------------------------------------------------------------------

def test_w_p0_7_empty_choices_triggers_fallback_model(monkeypatch):
    from ai.agents.base_agent import BaseAgent

    monkeypatch.setenv("CHAT_LLM_MODEL", "gemini-primary")
    monkeypatch.setenv("CHAT_LLM_FALLBACK_MODEL", "gemini-backup")
    monkeypatch.setenv("LLM_MAX_RETRY_ATTEMPTS", "1")

    empty = ChatCompletion.model_validate({
        "id": "cmpl-empty", "object": "chat.completion", "created": 1700000000, "model": "gemini-primary",
        "choices": [],
    })
    fake = install_fake_llm(monkeypatch, FakeLLM({
        "gemini-primary": lambda kw: empty,
        "gemini-backup": lambda kw: "备用回复来了",
    }))

    class ProbeAgent(BaseAgent):
        model_env_key = "CHAT_LLM_MODEL"

    res = ProbeAgent().run("hi")
    assert res["content"] == "备用回复来了" and res["failed"] is False and res["finish_reason"] == "stop"
    assert [c["model"] for c in fake.calls] == ["gemini-primary", "gemini-backup"]


def test_w_p0_7_and_p1_2_empty_safety_and_think_blocks(monkeypatch):
    from ai.agents.base_agent import BaseAgent
    from ai.channels.outbound import find_leak

    monkeypatch.setenv("CHAT_LLM_MODEL", "gemini-2.5-pro")

    class ProbeAgent(BaseAgent):
        model_env_key = "CHAT_LLM_MODEL"

    # 1. content_filter with empty content -> failed=True, finish_reason="content_filter"
    blocked = ChatCompletion.model_validate({
        "id": "cmpl-cf", "object": "chat.completion", "created": 1700000000, "model": "gemini-2.5-pro",
        "choices": [{"index": 0, "finish_reason": "content_filter",
                     "message": {"role": "assistant", "content": None}}],
    })
    install_fake_llm(monkeypatch, FakeLLM(default=lambda kw: blocked))
    r_cf = ProbeAgent().run("hi")
    assert r_cf["failed"] is True and r_cf["finish_reason"] == "content_filter" and r_cf["content"] == ""

    # 2. Only <think>...</think> (thinking starved) -> cleaned empty -> failed=True
    starved = ChatCompletion.model_validate({
        "id": "cmpl-st", "object": "chat.completion", "created": 1700000000, "model": "gemini-2.5-pro",
        "choices": [{"index": 0, "finish_reason": "length",
                     "message": {"role": "assistant", "content": "<think>I ran out of tokens</think>"}}],
    })
    install_fake_llm(monkeypatch, FakeLLM(default=lambda kw: starved))
    r_st = ProbeAgent().run("hi")
    assert r_st["failed"] is True and r_st["finish_reason"] == "length" and r_st["content"] == ""

    # 3. Closed <think>...</think> + normal reply -> stripped cleanly & passes find_leak()
    install_fake_llm(monkeypatch, FakeLLM(default=lambda kw: "<think>内部思考</think>\n晚安呀"))
    r_ok = ProbeAgent().run("晚安")
    assert r_ok["failed"] is False and r_ok["content"] == "晚安呀" and find_leak(r_ok["content"]) is None

    # 4. Unclosed <think> stays intact so outbound.find_leak() still catches it!
    install_fake_llm(monkeypatch, FakeLLM(default=lambda kw: "<think>未闭合思考泄露"))
    r_leak = ProbeAgent().run("hi")
    assert find_leak(r_leak["content"]) == "reasoning"

    # 5. Intentional [NO_REPLY] -> failed=False
    install_fake_llm(monkeypatch, FakeLLM(default=lambda kw: "[NO_REPLY]"))
    r_no = ProbeAgent().run("hi")
    assert r_no["failed"] is False and r_no["content"] == "[NO_REPLY]"


# ---------------------------------------------------------------------------
# W-P0-8: GIF first-frame -> PNG in vision.py & per-sticker continue in stickers.py
# ---------------------------------------------------------------------------

def test_w_p0_8_gif_transcoded_to_png_for_vision(tmp_path, monkeypatch):
    import ai.vision as vision

    gif_bytes = _make_valid_gif(2, 2, transparent=True)
    png_bytes = vision.gif_first_frame_to_png(gif_bytes)
    assert png_bytes is not None
    w, h, rgba = _decode_png_rgba(png_bytes)
    assert (w, h) == (2, 2)
    # Also verify canonical real-world 1x1 GIF89a
    real_1x1 = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
    png_1x1 = vision.gif_first_frame_to_png(real_1x1)
    assert png_1x1 is not None and _decode_png_rgba(png_1x1)[:2] == (1, 1)
    # Pixel 0 is index 0 (transparent red -> alpha 0), pixel 1 green, pixel 2 blue, pixel 3 white
    assert rgba[0:4] == bytes([255, 0, 0, 0])
    assert rgba[4:8] == bytes([0, 255, 0, 255])
    assert rgba[8:12] == bytes([0, 0, 255, 255])
    assert rgba[12:16] == bytes([255, 255, 255, 255])

    gif_file = tmp_path / "anim.gif"
    gif_file.write_bytes(gif_bytes)
    monkeypatch.setenv("LLM_API_KEY", "sk-vision-test-key")
    monkeypatch.setenv("CHAT_LLM_MODEL", "gemini-2.5-flash")
    monkeypatch.delenv("VISION_LLM_MODEL", raising=False)

    seen_calls = []

    class FakeVisionClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    seen_calls.append(kw)
                    return SimpleNamespace(
                        model="gemini-2.5-flash",
                        usage=None,
                        choices=[SimpleNamespace(finish_reason="stop",
                                                 message=SimpleNamespace(content="<think>check</think>一只彩色方块"))],
                    )

    monkeypatch.setattr(vision, "_client", lambda url, key: FakeVisionClient())
    desc = vision.describe_image(gif_file)
    assert desc == "一只彩色方块"
    img_url = seen_calls[0]["messages"][0]["content"][1]["image_url"]["url"]
    assert img_url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(img_url.split(",", 1)[1])
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")


def test_w_p0_8_stickers_describe_pending_skips_single_400_but_breaks_on_auth(violetdb, tmp_path, monkeypatch):
    from ai import stickers
    import ai.vision as vision
    from tests.image_fixtures import PNG

    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    violetdb.execute("DELETE FROM stickers")
    violetdb.execute("DELETE FROM sticker_blocklist")

    s1 = stickers.add_bytes(violetdb, PNG + b"bad-gif", source="qq")
    s2 = stickers.add_bytes(violetdb, PNG + b"good-1", source="qq")
    s3 = stickers.add_bytes(violetdb, PNG + b"good-2", source="qq")

    # Case 1: first sticker fails with HTTP 400 (unsupported single sticker) -> continues to s2 & s3!
    calls = []

    def fake_describe_image_400(path, prompt=None):
        calls.append(str(path))
        if len(calls) == 1:
            err = _bad_request("Unsupported MIME type: image/gif")
            raise vision.VisionError("识图模型 gemini 调用失败：HTTP 400：Unsupported MIME type: image/gif") from err
        return "画面：小猫比心\n什么时候发：表达喜欢\n关键词：比心、猫\n判断：正常"

    monkeypatch.setattr(vision, "describe_image", fake_describe_image_400)
    assert stickers.describe_pending(violetdb) == 2
    assert len(calls) == 3
    assert stickers.get(violetdb, s1["id"])["description"] is None
    assert stickers.get(violetdb, s2["id"])["description"] is not None
    assert stickers.get(violetdb, s3["id"])["description"] is not None

    # Case 2: service-wide HTTP 401 auth error -> breaks immediately after first attempt!
    s4 = stickers.add_bytes(violetdb, PNG + b"good-3", source="qq")
    auth_calls = []

    def fake_describe_image_401(path, prompt=None):
        auth_calls.append(str(path))
        resp = httpx.Response(401, request=httpx.Request("POST", "https://relay.example/v1"))
        err = openai.AuthenticationError("invalid api key", response=resp, body=None)
        raise vision.VisionError("识图模型 gemini 调用失败：HTTP 401：invalid api key") from err

    monkeypatch.setattr(vision, "describe_image", fake_describe_image_401)
    assert stickers.describe_pending(violetdb) == 0
    assert len(auth_calls) == 1  # aborted batch on s1, did NOT hit API again for s4


# ---------------------------------------------------------------------------
# W-P1-3..6 & W-P2-1 & W-P2-4: gemini provider, wildcards, MemoryAgent, usage, streaming
# ---------------------------------------------------------------------------

def test_w_p1_4_p1_5_p2_4_gemini_provider_wildcards_and_control_config(tmp_path, monkeypatch):
    from ai import control_config as cc
    from ai import model_params as mp

    assert "gemini" in mp.PROVIDERS
    assert mp.provider("gemini-2.5-pro") == "gemini"
    assert mp.provider("google/gemini-3-flash") == "gemini"
    assert cc.validate("VIOLET_LLM_PROVIDER", "gemini") == "gemini"

    # Relay URL gets sticky headers, NO prompt_cache_key, NO cache_control
    monkeypatch.setenv("LLM_API_URL", "https://my-relay.example/v1")
    msgs = [
        {"role": "system", "content": "sys1"},
        {"role": "system", "content": "sys2", mp.CACHE_MARK: True},
        {"role": "user", "content": "u1"},
        {"role": "user", "content": "u2"},
    ]
    relay_req = mp.apply({"model": "gemini-2.5-pro", "messages": msgs, "cache_key": "violet-ChatAgent"})
    assert "extra_body" not in relay_req
    assert "session_id" in relay_req["extra_headers"] and "x-session-id" in relay_req["extra_headers"]
    assert [m["role"] for m in relay_req["messages"]] == ["system", "user"]
    assert "sys1" in relay_req["messages"][0]["content"] and "sys2" in relay_req["messages"][0]["content"]
    assert "u1" in relay_req["messages"][1]["content"] and "u2" in relay_req["messages"][1]["content"]

    # Direct Google URL omits sticky headers
    monkeypatch.setenv("LLM_API_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    direct_req = mp.apply({"model": "gemini-2.5-pro", "messages": msgs, "cache_key": "violet-ChatAgent"})
    assert "extra_headers" not in direct_req and "extra_body" not in direct_req

    # Wildcard policy matching (gemini-3* + specificity + default_temperature + default_reasoning_effort + extra_body)
    policy_file = tmp_path / "model_params.json"
    policy_file.write_text(json.dumps({
        "models": {
            "gemini-3*": {
                "default_temperature": 1.0,
                "default_reasoning_effort": "low",
                "extra_body": {"google": {"safety_settings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]}},
            },
            "gemini-3-flash*": {
                "default_temperature": 0.9,
                "default_reasoning_effort": "minimal",
            },
        }
    }), encoding="utf-8")
    monkeypatch.setenv("VIOLET_MODEL_PARAMS_FILE", str(policy_file))
    pro = mp.apply({"model": "gemini-3-pro-preview", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.2})
    assert pro["temperature"] == 1.0
    assert pro["reasoning_effort"] == "low"
    assert "google" in pro["extra_body"]

    flash = mp.apply({"model": "gemini-3-flash-preview", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.2})
    assert flash["temperature"] == 0.9 and flash["reasoning_effort"] == "minimal"


def test_w_p1_6_memory_agent_merges_system_and_user_blocks(violetdb):
    from ai.agents import MemoryAgent
    from ai.agents.base_agent import _CALL_CONTEXT
    from ai import model_params as mp

    mem = MemoryAgent()
    token = _CALL_CONTEXT.set({"previous_writes": [{"tool": "save_memory", "ok": True, "summary": "测试"}]})
    try:
        msgs = mem.build_context_messages("用户说：我喜欢喝乌龙茶")
    finally:
        _CALL_CONTEXT.reset(token)

    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0].get(mp.CACHE_MARK) is True
    assert "参考资料" in msgs[1]["content"] and "我喜欢喝乌龙茶" in msgs[1]["content"]

    # Claude still gets cache_control on the single merged system block
    claude = mp.apply({"model": "claude-opus-5-5", "messages": msgs})
    assert len([m for m in claude["messages"] if m["role"] == "system"]) == 1
    assert claude["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_w_p2_1_usage_reasoning_tokens_without_breaking_numbers():
    from ai import usage

    obj = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=45,
        prompt_tokens_details=SimpleNamespace(cached_tokens=60),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=30),
    )
    assert usage.numbers(obj) == (100, 45, 60)
    assert usage.reasoning_tokens(obj) == 30
    assert usage.reasoning_tokens({"completion_tokens_details": {"reasoning_tokens": 12}}) == 12
    assert usage.reasoning_tokens({"thoughts_token_count": 7}) == 7
    assert usage.reasoning_tokens(None) == 0


def test_w_p1_3_optional_streaming_accumulation_and_thought_signature(monkeypatch):
    from ai.agents.base_agent import BaseAgent, EmptyCompletionError

    monkeypatch.setenv("VIOLET_LLM_STREAM", "1")
    monkeypatch.setenv("CHAT_LLM_MODEL", "gemini-2.5-pro")
    monkeypatch.setenv("LLM_MAX_RETRY_ATTEMPTS", "1")

    sig = "streamed_thought_sig_999"
    sent_kwargs = []

    def make_chunk(**kw):
        return SimpleNamespace(
            id=kw.get("id", "cmpl-s"),
            created=1700000000,
            model="gemini-2.5-pro",
            choices=kw.get("choices", []),
            usage=kw.get("usage"),
        )

    round_num = [0]

    class StreamClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    sent_kwargs.append(kw)
                    # Reject stream_options on first call to test fallback
                    if "stream_options" in kw and len(sent_kwargs) == 1:
                        raise _bad_request("Unknown field: stream_options")
                    round_num[0] += 1
                    if round_num[0] == 1:
                        # Streamed tool call split across 2 chunks with thought_signature
                        return iter([
                            make_chunk(choices=[SimpleNamespace(
                                finish_reason=None,
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[SimpleNamespace(
                                        index=0, id="call_st_1", type="function",
                                        function=SimpleNamespace(name="echo_tool", arguments='{"msg":'),
                                        extra_content={"google": {"thought_signature": sig}},
                                    )],
                                ),
                            )]),
                            make_chunk(choices=[SimpleNamespace(
                                finish_reason="tool_calls",
                                delta=SimpleNamespace(
                                    content=None,
                                    tool_calls=[SimpleNamespace(
                                        index=0, id=None, type=None,
                                        function=SimpleNamespace(name="", arguments=' "hi"}'),
                                        extra_content=None,
                                    )],
                                ),
                            )], usage={"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35}),
                        ])
                    # Round 2: streamed text chunks + usage
                    return iter([
                        make_chunk(choices=[SimpleNamespace(
                            finish_reason=None,
                            delta=SimpleNamespace(content="<think>ok</think>流式"),
                        )]),
                        make_chunk(choices=[SimpleNamespace(
                            finish_reason="stop",
                            delta=SimpleNamespace(content="回复成功"),
                        )], usage={"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40}),
                    ])

    monkeypatch.setattr(BaseAgent, "client_factory", lambda cfg: StreamClient())

    class StreamAgent(BaseAgent):
        model_env_key = "CHAT_LLM_MODEL"

        def register_tools(self):
            self.register_tool(
                {"type": "function", "function": {
                    "name": "echo_tool", "description": "echo",
                    "parameters": {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
                }},
                lambda args: f"echo:{args['msg']}",
            )

    res = StreamAgent().run("test stream")
    assert res["content"] == "流式回复成功" and res["failed"] is False and res["finish_reason"] == "stop"
    assert res["tools_called"] == ["echo_tool"]
    # Verify thought_signature survived streamed tool call into round 2!
    r2_msgs = sent_kwargs[-1]["messages"]
    asst = next(m for m in r2_msgs if m["role"] == "assistant")
    assert asst["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": sig}}
    tool_m = next(m for m in r2_msgs if m["role"] == "tool")
    assert tool_m["name"] == "echo_tool"

    # Empty stream raises EmptyCompletionError
    class EmptyStreamClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return iter([])

    monkeypatch.setattr(BaseAgent, "client_factory", lambda cfg: EmptyStreamClient())
    with pytest.raises(EmptyCompletionError):
        StreamAgent().run("empty")


# ---------------------------------------------------------------------------
# Phase 3 residual edge polish tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw_fr, expected_fr, expect_failed", [
    ("safety", "content_filter", True),
    ("PROHIBITED_CONTENT", "content_filter", True),
    ("recitation", "content_filter", True),
    ("blocklist", "content_filter", True),
    ("spii", "content_filter", True),
    ("max_tokens", "length", False),
    ("malformed_function_call", "stop", False),
    ("SOME_FUTURE_REASON", "stop", False),
])
def test_p3_1_stream_non_openai_finish_reasons_never_crash_pydantic(monkeypatch, raw_fr, expected_fr, expect_failed):
    from ai.agents.base_agent import BaseAgent, _accumulate_stream

    text_content = "partial text" if not expect_failed else "blocked partial"
    chunks = [
        SimpleNamespace(
            id="cmpl-ns",
            created=1700000000,
            model="gemini-2.5-pro",
            choices=[SimpleNamespace(finish_reason=None, delta=SimpleNamespace(content=text_content))],
            usage=None,
        ),
        SimpleNamespace(
            id="cmpl-ns",
            created=1700000000,
            model="gemini-2.5-pro",
            choices=[SimpleNamespace(finish_reason=raw_fr, delta=SimpleNamespace(content=None))],
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        ),
    ]
    comp = _accumulate_stream(iter(chunks), fallback_model="gemini-2.5-pro")
    assert comp.choices[0].finish_reason == expected_fr
    assert getattr(comp.choices[0], "raw_finish_reason", None) == raw_fr

    # End-to-end BaseAgent with VIOLET_LLM_STREAM=1 on safety stream -> failed=True, no ValidationError
    if expect_failed:
        monkeypatch.setenv("VIOLET_LLM_STREAM", "1")
        monkeypatch.setenv("CHAT_LLM_MODEL", "gemini-2.5-pro")

        class SafetyStreamClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kw):
                        return iter([
                            SimpleNamespace(
                                id="cmpl-sf", created=1700000000, model="gemini-2.5-pro",
                                choices=[SimpleNamespace(finish_reason=raw_fr, delta=SimpleNamespace(content=None))],
                                usage=None,
                            )
                        ])

        monkeypatch.setattr(BaseAgent, "client_factory", lambda cfg: SafetyStreamClient())

        class Probe(BaseAgent):
            model_env_key = "CHAT_LLM_MODEL"

        res = Probe().run("hi")
        assert res["failed"] is True and res["finish_reason"] == "content_filter"


def test_p3_2_chat_completions_adapter_gif_extra_body_and_strip_think(tmp_path, monkeypatch):
    from ai.images import adapters
    from ai.images.adapters import AdapterError, AdapterRequest, ChatCompletionsAdapter, InputImage
    from ai.images.registry import ModelEntry

    policy_file = tmp_path / "mp_img.json"
    policy_file.write_text(json.dumps({
        "models": {
            "gemini-2.5-flash*": {
                "default_reasoning_effort": "low",
                "extra_body": {"google": {"safety_settings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]}},
            }
        }
    }), encoding="utf-8")
    monkeypatch.setenv("VIOLET_MODEL_PARAMS_FILE", str(policy_file))
    monkeypatch.setenv("TEST_IMG_KEY", "sk-img-test-123456")

    seen_bodies = []
    seen_headers = []
    responses = iter([
        {"choices": [{"finish_reason": "stop", "message": {"content": "<think>examining colors</think>\n一个四色方块"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "<think>only thoughts</think>"}}]},
    ])

    def handler(req: httpx.Request) -> httpx.Response:
        seen_bodies.append(json.loads(req.content.decode("utf-8")))
        seen_headers.append(dict(req.headers))
        return httpx.Response(200, json=next(responses))

    prev_factory = adapters._TRANSPORT_FACTORY
    adapters.set_transport_factory(lambda entry: httpx.MockTransport(handler))
    try:
        entry = ModelEntry(
            provider_id="relay",
            model_id="gemini-2.5-flash-vision",
            adapter_type="chat_completions",
            base_url="https://relay.example/v1",
            api_key_env="TEST_IMG_KEY",
            enabled=True,
            capabilities={"vision": "verified"},
            options={"extra_body": {"custom_flag": True}},
        )
        adapter = ChatCompletionsAdapter(entry)
        gif_img = InputImage(data=_make_valid_gif(2, 2, transparent=False), mime="image/gif")
        out = adapter.run(AdapterRequest(kind="vision_analyze", entry=entry, prompt="describe", inputs=[gif_img]))
        assert out.text == "一个四色方块"

        body = seen_bodies[0]
        # GIF transcoded to PNG
        img_url = body["messages"][0]["content"][1]["image_url"]["url"]
        assert img_url.startswith("data:image/png;base64,")
        assert base64.b64decode(img_url.split(",", 1)[1]).startswith(b"\x89PNG\r\n\x1a\n")
        # extra_body flattened into top-level JSON, NEVER sent as literal "extra_body" / "extra_headers"
        assert "extra_body" not in body and "extra_headers" not in body
        assert body["google"]["safety_settings"][0]["threshold"] == "BLOCK_NONE"
        assert body["custom_flag"] is True
        assert body["reasoning_effort"] == "low"

        # Also verify _prepare_chat_payload pops extra_headers into HTTP headers
        json_p, hdr_p = adapters._prepare_chat_payload({
            "model": "gemini-2.5-flash-vision",
            "messages": [{"role": "user", "content": "hi"}],
            "cache_key": "violet-vision",
        })
        assert "extra_headers" not in json_p and "extra_body" not in json_p
        assert "x-session-id" in hdr_p and "session_id" in hdr_p

        # Pure <think>...</think> becomes empty -> raises AdapterError("empty_response")
        with pytest.raises(AdapterError) as exc:
            adapter.run(AdapterRequest(kind="vision_analyze", entry=entry, prompt="describe", inputs=[gif_img]))
        assert exc.value.code == "empty_response"
    finally:
        adapters.set_transport_factory(prev_factory)


def test_p3_3_mcp_nested_propertyless_objects_normalized_for_gemini():
    from ai.mcp import tool_schema_to_openai_tool

    raw = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "metadata": {"type": "object", "additionalProperties": True, "description": "Arbitrary metadata"},
            "empty_nested": {"type": "object", "properties": {}},
            "records": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
            "structured": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            },
        },
        "required": ["query"],
    }
    tool = tool_schema_to_openai_tool("ext__open_dict", "desc", raw)
    params = tool["function"]["parameters"]
    _assert_gemini_clean_schema(params)

    # Assert NO object node anywhere in the tree has empty/missing properties
    def assert_no_empty_object(node, path="root"):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("properties"), f"{path} is an object without properties: {node}"
            assert node, f"{path} is an empty schema dict"
            for k, v in node.items():
                assert_no_empty_object(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                assert_no_empty_object(item, f"{path}[{i}]")

    assert_no_empty_object(params)
    assert params["properties"]["metadata"]["type"] == "string"
    assert "JSON" in params["properties"]["metadata"]["description"]
    assert params["properties"]["empty_nested"]["type"] == "string"
    assert params["properties"]["records"] == {"type": "array", "items": {"type": "string", "description": "JSON object"}}
    assert params["properties"]["structured"] == {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}


def test_p3_4_aux_llm_and_setup_status_use_model_params_create_and_classify_empty(violetdb, tmp_path, monkeypatch):
    from ai import model_params as mp
    from ai.agents.base_agent import EmptyCompletionError
    from ai.aux_llm import AuxLLM, AuxLLMConfig
    from ai.setup_status import classify_error, test_model

    # 1. classify_error handles EmptyCompletionError explicitly
    kind, msg = classify_error(EmptyCompletionError("empty choices from model"))
    assert kind == "empty_response" and "空结果" in msg

    # Configure wildcard policy injecting reasoning_effort
    policy_file = tmp_path / "mp_refuse.json"
    policy_file.write_text(json.dumps({
        "models": {"gemini-refuse-*": {"default_reasoning_effort": "low"}}
    }), encoding="utf-8")
    monkeypatch.setenv("VIOLET_MODEL_PARAMS_FILE", str(policy_file))
    mp._REFUSED.discard(("gemini-refuse-aux", "reasoning_effort"))
    mp._REFUSED.discard(("gemini-refuse-test", "reasoning_effort"))

    # 2. AuxLLM recovers in a single attempt via model_params.create() when relay refuses reasoning_effort
    aux_seen = []

    def aux_resp(kw):
        aux_seen.append(dict(kw))
        if "reasoning_effort" in kw:
            raise _bad_request("Unsupported parameter: reasoning_effort")
        return '{"ok": true}'

    install_fake_llm(monkeypatch, FakeLLM(default=aux_resp))
    aux = AuxLLM(violetdb, AuxLLMConfig(purpose="p3_test", model="gemini-refuse-aux", max_attempts=1))
    assert aux.complete_json([{"role": "user", "content": "hi"}]) == {"ok": True}
    assert len(aux_seen) == 2
    assert "reasoning_effort" in aux_seen[0] and "reasoning_effort" not in aux_seen[1]

    # 3. setup_status.test_model() recovers via model_params.create() when relay refuses reasoning_effort
    test_seen = []

    class TestClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    test_seen.append(dict(kw))
                    if "reasoning_effort" in kw:
                        raise _bad_request("Unsupported parameter: reasoning_effort")
                    return completion("在的", model="gemini-refuse-test")

    fake_agent = SimpleNamespace(
        _config=SimpleNamespace(reply_model="gemini-refuse-test"),
        _get_client=lambda cfg: TestClient(),
    )
    res = test_model(fake_agent, data_dir=tmp_path / "data", model="gemini-refuse-test")
    assert res["ok"] is True and res["kind"] == "ok"
    assert len(test_seen) == 2
    assert "reasoning_effort" in test_seen[0] and "reasoning_effort" not in test_seen[1]


# ---------------------------------------------------------------------------
# Phase 3 stress-test bugfixes (#51)
# ---------------------------------------------------------------------------

def test_p3_stress_stream_usage_omitting_total_tokens_and_empty_delta_clobber():
    from ai.agents.base_agent import _accumulate_stream

    sig = "CiQ_valid_gemini_thought_signature_123"
    chunks = [
        SimpleNamespace(
            id="cmpl-sse-1",
            created=1700000000,
            model="gemini-3-pro",
            choices=[SimpleNamespace(
                finish_reason=None,
                delta=SimpleNamespace(
                    content=None,
                    extra_content={"google": {"msg_sig": "m1"}},
                    thought_signature="top_sig_1",
                    tool_calls=[SimpleNamespace(
                        index=0, id="call_1", type="function",
                        function=SimpleNamespace(name="search", arguments='{"q":'),
                        extra_content={"google": {"thought_signature": sig}},
                        thought_signature="tc_sig_1",
                    )],
                ),
            )],
            usage=None,
        ),
        # Continuation chunk echoes empty dicts/strings -> MUST NOT clobber chunk 1's valid signatures!
        SimpleNamespace(
            id="cmpl-sse-1",
            created=1700000000,
            model="gemini-3-pro",
            choices=[SimpleNamespace(
                finish_reason="tool_calls",
                delta=SimpleNamespace(
                    content=None,
                    extra_content={"google": {"msg_sig": "", "extra": "kept"}},
                    thought_signature="",
                    tool_calls=[SimpleNamespace(
                        index=0, id="", type="",
                        function=SimpleNamespace(name="", arguments=' "cat"}'),
                        extra_content={"google": {"thought_signature": ""}},
                        thought_signature="",
                    )],
                ),
            )],
            # Partial usage dict omitting total_tokens (common in SSE relays) -> must backfill instead of crashing!
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        ),
        # Trailing chunk echoing completely empty {} extra_content
        SimpleNamespace(
            id="cmpl-sse-1",
            created=1700000000,
            model="gemini-3-pro",
            choices=[SimpleNamespace(
                finish_reason="tool_calls",
                delta=SimpleNamespace(
                    content=None,
                    extra_content={},
                    thought_signature="",
                    tool_calls=[SimpleNamespace(
                        index=0, id=None, type=None,
                        function=SimpleNamespace(name="", arguments=""),
                        extra_content={},
                        thought_signature="",
                    )],
                ),
            )],
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        ),
    ]

    comp = _accumulate_stream(iter(chunks), fallback_model="gemini-3-pro")
    assert comp.usage is not None
    assert comp.usage.prompt_tokens == 100
    assert comp.usage.completion_tokens == 20
    assert comp.usage.total_tokens == 120

    msg = comp.choices[0].message
    assert msg.extra_content == {"google": {"msg_sig": "m1", "extra": "kept"}}
    assert msg.thought_signature == "top_sig_1"
    tc0 = msg.tool_calls[0]
    assert tc0.extra_content == {"google": {"thought_signature": sig}}
    assert tc0.thought_signature == "tc_sig_1"
    assert tc0.function.name == "search" and tc0.function.arguments == '{"q": "cat"}'


def test_p3_stress_mcp_implicit_nested_objects_and_arrays_infer_proper_type():
    from ai.mcp import tool_schema_to_openai_tool

    raw = {
        "type": "object",
        "properties": {
            "implicit_obj": {
                "properties": {
                    "nested_key": {"type": "string"},
                },
                "required": ["nested_key"],
            },
            "implicit_arr": {
                "items": {"type": "integer"},
            },
            "implicit_empty_obj": {
                "properties": {},
            },
        },
        "required": ["implicit_obj"],
    }
    tool = tool_schema_to_openai_tool("ext__implicit", "desc", raw)
    props = tool["function"]["parameters"]["properties"]
    _assert_gemini_clean_schema(tool["function"]["parameters"])
    assert props["implicit_obj"] == {
        "type": "object",
        "properties": {"nested_key": {"type": "string"}},
        "required": ["nested_key"],
    }
    assert props["implicit_arr"] == {"type": "array", "items": {"type": "integer"}}
    assert props["implicit_empty_obj"] == {"type": "string"}
