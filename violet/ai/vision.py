"""Image understanding for incoming pictures (the "看图" model of the Control Center's 图片 page).

0.7.6: the call goes through the same OpenAI SDK client as the chat agents (same address, key, proxy handling and
request shape), instead of a hand-made requests.post that a relay could treat differently. The model is
VISION_LLM_MODEL, or the Chat model when that is empty (the old default "vision" exists on almost no service).
A request the service rejects (HTTP 400, e.g. a parameter it does not take) is tried once more with only model and
messages. Failures raise VisionError with the HTTP status and the service's own message, so the page and the log
say why.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import struct
import zlib

from dotenv import load_dotenv

from ai.model_params import strip_think
from ai.settings import ENV_PATH, env_float, env_str

logger = logging.getLogger("violet.vision")

_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
_MAGIC = ((b"\x89PNG", "image/png"), (b"\xff\xd8", "image/jpeg"), (b"GIF8", "image/gif"))

VISION_PROMPT = (
    "请仔细观察这张图片，只描述你实际看到的内容。"
    "不要猜测图片中没有明确看到的信息。"
    "请用简洁自然的中文描述，方便另一个聊天模型继续理解用户发来的图片。"
)


class VisionError(RuntimeError):
    pass


def vision_model() -> str:
    """VISION_LLM_MODEL, else the Chat model. "vision" was the old env_example placeholder (copied into installs'
    .env), not a real model name: it counts as not chosen."""
    chosen = env_str("VISION_LLM_MODEL", "")
    return chosen if chosen and chosen != "vision" else env_str("CHAT_LLM_MODEL", "")


def _mime(file_path: str, data: bytes) -> str:
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return _MIME_MAP.get(os.path.splitext(file_path)[1].lower(), "image/jpeg")


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)


def _encode_rgba_png(width: int, height: int, rgba: bytes | bytearray) -> bytes:
    stride = width * 4
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw.extend(rgba[y * stride:(y + 1) * stride])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _png_chunk(b"IEND", b"")
    )


def _gif_first_frame_to_png_stdlib(data: bytes) -> bytes | None:
    """Decode frame 0 of a GIF87a/GIF89a byte stream to RGBA PNG using only the Python stdlib."""
    if len(data) < 13 or data[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    canvas_w, canvas_h, packed, bg_idx, _aspect = struct.unpack_from("<HHBBB", data, 6)
    pos = 13
    gct = b""
    if packed & 0x80:
        gct_len = 3 * (1 << ((packed & 0x07) + 1))
        if pos + gct_len > len(data):
            return None
        gct = data[pos:pos + gct_len]
        pos += gct_len

    has_trans = False
    trans_idx = -1
    max_pixels = 16_000_000

    while pos < len(data):
        intro = data[pos]
        pos += 1
        if intro == 0x3B:
            return None
        if intro == 0x21:
            if pos >= len(data):
                return None
            label = data[pos]
            pos += 1
            chunks = []
            while pos < len(data):
                sz = data[pos]
                pos += 1
                if sz == 0:
                    break
                if pos + sz > len(data):
                    return None
                chunks.append(data[pos:pos + sz])
                pos += sz
            if label == 0xF9 and chunks:
                gce = b"".join(chunks)
                if len(gce) >= 4:
                    has_trans = bool(gce[0] & 0x01)
                    trans_idx = gce[3]
            continue
        if intro != 0x2C:
            return None
        if pos + 9 > len(data):
            return None
        left, top, w, h, ipacked = struct.unpack_from("<HHHHB", data, pos)
        pos += 9
        if w <= 0 or h <= 0 or w * h > max_pixels:
            return None
        palette = gct
        if ipacked & 0x80:
            lct_len = 3 * (1 << ((ipacked & 0x07) + 1))
            if pos + lct_len > len(data):
                return None
            palette = data[pos:pos + lct_len]
            pos += lct_len
        if not palette or pos >= len(data):
            return None
        min_code_size = data[pos]
        pos += 1
        if min_code_size < 2 or min_code_size > 8:
            return None
        lzw_parts = []
        while pos < len(data):
            sz = data[pos]
            pos += 1
            if sz == 0:
                break
            if pos + sz > len(data):
                return None
            lzw_parts.append(data[pos:pos + sz])
            pos += sz
        lzw_bytes = b"".join(lzw_parts)
        if not lzw_bytes:
            return None

        # Standard GIF variable-width LSB-first LZW decompression
        clear_code = 1 << min_code_size
        eoi_code = clear_code + 1
        code_size = min_code_size + 1
        table: list[bytes] = [bytes([i]) for i in range(clear_code)] + [b"", b""]
        bit_buf = 0
        bit_count = 0
        byte_idx = 0
        n_bytes = len(lzw_bytes)
        total_pixels = w * h
        indices = bytearray()
        prev_entry: bytes | None = None

        while len(indices) < total_pixels:
            while bit_count < code_size and byte_idx < n_bytes:
                bit_buf |= lzw_bytes[byte_idx] << bit_count
                bit_count += 8
                byte_idx += 1
            if bit_count < code_size:
                break
            code = bit_buf & ((1 << code_size) - 1)
            bit_buf >>= code_size
            bit_count -= code_size

            if code == clear_code:
                table = [bytes([i]) for i in range(clear_code)] + [b"", b""]
                code_size = min_code_size + 1
                prev_entry = None
                continue
            if code == eoi_code:
                break
            if code < len(table):
                entry = table[code]
            elif code == len(table) and prev_entry is not None:
                entry = prev_entry + prev_entry[:1]
            else:
                return None
            indices.extend(entry)
            if prev_entry is not None and len(table) < 4096:
                table.append(prev_entry + entry[:1])
                if len(table) == (1 << code_size) and code_size < 12:
                    code_size += 1
            prev_entry = entry

        if len(indices) < total_pixels:
            return None
        indices = indices[:total_pixels]

        if ipacked & 0x40:
            # De-interlace GIF 4-pass rows
            rows: list[bytes] = [b""] * h
            src_row = 0
            for start_row, step in ((0, 8), (4, 8), (2, 4), (1, 2)):
                for dst_row in range(start_row, h, step):
                    rows[dst_row] = bytes(indices[src_row * w:(src_row + 1) * w])
                    src_row += 1
            indices = bytearray(b"".join(rows))

        out_w = max(canvas_w, left + w)
        out_h = max(canvas_h, top + h)
        if out_w <= 0 or out_h <= 0 or out_w * out_h > max_pixels:
            return None
        n_colors = len(palette) // 3
        rgba_lut = [
            bytes((palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2],
                   0 if (has_trans and i == trans_idx) else 255))
            for i in range(n_colors)
        ]
        bg_rgba = b"\x00\x00\x00\x00" if has_trans else (rgba_lut[bg_idx] if bg_idx < n_colors else b"\xff\xff\xff\xff")
        canvas = bytearray(bg_rgba * (out_w * out_h))
        fallback_px = b"\x00\x00\x00\xff"
        for y in range(h):
            row_off = y * w
            dst_row_off = ((top + y) * out_w + left) * 4
            for x in range(w):
                idx = indices[row_off + x]
                px = rgba_lut[idx] if idx < n_colors else fallback_px
                dst = dst_row_off + x * 4
                canvas[dst:dst + 4] = px
        return _encode_rgba_png(out_w, out_h, canvas)
    return None


def gif_first_frame_to_png(data: bytes) -> bytes | None:
    """Convert a GIF's first frame to PNG bytes for Gemini Vision compatibility (which rejects image/gif)."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as img:
            img.seek(0)
            frame = img.convert("RGBA")
            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            return buf.getvalue()
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 -- fall back to stdlib decoder
        logger.debug("PIL GIF conversion failed, trying stdlib decoder: %s", exc)
    try:
        return _gif_first_frame_to_png_stdlib(data)
    except Exception as exc:  # noqa: BLE001
        logger.debug("stdlib GIF conversion failed: %s", exc)
        return None


def _client(api_url: str, api_key: str):
    from openai import OpenAI, Timeout
    from ai.agents.base_agent import BaseAgent, _local_address
    if BaseAgent.client_factory is not None:              # the offline tests' fake model service
        from ai.chat import LLMConfig
        cfg = LLMConfig()
        cfg.reply_model = vision_model()
        return BaseAgent.client_factory(cfg)
    extra = {}
    if _local_address(api_url):                           # same rule as the chat agents: no system proxy for LAN
        from openai import DefaultHttpxClient
        extra["http_client"] = DefaultHttpxClient(trust_env=False)
    return OpenAI(api_key=api_key, base_url=api_url, max_retries=0,
                  timeout=Timeout(env_float("VISION_TIMEOUT_SECONDS", 60.0), connect=10.0), **extra)


def _reason(exc: Exception) -> str:
    """"HTTP 404：model vision not found" -- status and the service's message, short."""
    status = getattr(exc, "status_code", None)
    message = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        message = str(err.get("message") or "") if isinstance(err, dict) else str(err)
    if not message:
        message = str(getattr(exc, "message", "") or exc)
    message = " ".join(message.split())[:240]
    if status:
        return f"HTTP {status}：{message}" if message else f"HTTP {status}"
    return f"{exc.__class__.__name__}：{message}" if message else exc.__class__.__name__


def describe_image(file_path: str | os.PathLike[str], prompt: str | None = None) -> str:
    """Return a short Chinese description of the image (prompt: another question, e.g. for a sticker).
    Raises VisionError."""
    file_path = os.fspath(file_path)
    if not os.path.exists(file_path):
        raise VisionError(f"图片文件不见了：{os.path.basename(file_path)}")

    load_dotenv(ENV_PATH, override=False)
    from ai.chat import get_llm_api_url
    api_url = get_llm_api_url().rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise VisionError("还没有设置模型服务的 API Key")
    model = vision_model()
    if not model:
        raise VisionError("还没有选识图模型（也没有 Chat 模型可用）")

    with open(file_path, "rb") as f:
        data = f.read()
    mime = _mime(file_path, data)
    if mime == "image/gif":
        png_bytes = gif_first_frame_to_png(data)
        if png_bytes:
            mime, data = "image/png", png_bytes
    url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt or VISION_PROMPT},
                                             {"type": "image_url", "image_url": {"url": url}}]}]

    from openai import BadRequestError
    from ai import trace
    from ai.model_params import apply as apply_model_params
    client = _client(api_url, api_key)
    # 0.7.47 (user: telegram 表情包的描述只剩半句): 500 tokens cut answers ("左侧角色：露出了"), most of all on
    # thinking models whose reasoning shares the budget. 1500, and once more with 4000 when it still stops for length.
    full = apply_model_params({"model": model, "messages": messages, "reasoning_effort": "low", "max_tokens": 1500})
    try:
        try:
            response = client.chat.completions.create(**full)
            if _stopped_for_length(response):
                trace.step("识图", 模型=model, 结果="重试", 原因="回答被截断，给更多字数再看一次")
                try:
                    response = client.chat.completions.create(
                        **apply_model_params({**full, "max_tokens": 4000}))
                except Exception:                            # noqa: BLE001 -- keep the cut answer
                    pass
        except BadRequestError as exc:
            # a service that does not take max_tokens / reasoning_effort: once more with the bare request
            trace.step("识图", 模型=model, 结果="重试", 原因=_reason(exc))
            response = client.chat.completions.create(model=model, messages=messages)
    except Exception as exc:
        reason = _reason(exc)
        trace.step("识图", 模型=model, 结果="失败", 原因=reason)
        raise VisionError(f"识图模型 {model} 调用失败：{reason}") from exc

    from ai import usage
    usage.record("vision", getattr(response, "model", None) or model, getattr(response, "usage", None))
    try:
        description = strip_think(response.choices[0].message.content or "").strip()
    except (AttributeError, IndexError, TypeError):
        description = ""
    if not description:
        trace.step("识图", 模型=model, 结果="空回复")
        raise VisionError(f"识图模型 {model} 没有返回描述（这个模型可能不支持看图）")
    trace.step("识图", 模型=model, 结果="ok", 描述=trace.text(description))
    return description


def _stopped_for_length(response) -> bool:
    try:
        return str(response.choices[0].finish_reason or "").lower() in ("length", "max_tokens")
    except (AttributeError, IndexError, TypeError):
        return False
