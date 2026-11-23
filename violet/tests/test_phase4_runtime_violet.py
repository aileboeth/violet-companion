"""Phase 4 Domain 4 regression & security tests:
outbound guard, TurnBuffer generation token, outbox ReadTimeout/504 unknown status,
OneBot/SnowLuma SSRF & pairing commands, Telegram concurrent downloads,
truncated WebP validation, Eve import zip-bomb/corrupt-zip/schema/secrets, and voice hardening.
"""

from __future__ import annotations

import concurrent.futures
import http.server
import io
import json
import socketserver
import sqlite3
import struct
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests


# ---------------------------------------------------------------------------
# D4-01 & D4-08: Outbound guard leak detection & false-positive prevention
# ---------------------------------------------------------------------------

def test_outbound_catches_multiline_fenced_inline_json_and_thought_tags():
    from ai.channels import outbound

    outbound.reset_for_tests()

    ml_fenced = (
        "```json\n"
        "{\n"
        '  "should_reply": true,\n'
        '  "route": "chat",\n'
        '  "memory_ops": [{"action": "add"}]\n'
        "}\n"
        "```"
    )
    ml_raw = '{\n  "should_reply": true,\n  "route": "chat"\n}'
    truncated = '{\n  "should_reply": true,\n  "memory_ops": ['
    inline = '好的，{"should_reply": false, "route": "chat"}'

    assert outbound.find_leak(ml_fenced) == "internal_json"
    assert outbound.find_leak(ml_raw) == "internal_json"
    assert outbound.find_leak(truncated) == "internal_json"
    assert outbound.find_leak(inline) == "internal_json"

    # Multi-message split via normalize_chat_output must still be caught by find_reply_leak
    split_msgs = outbound.chat_payload(ml_fenced)["messages"]
    assert len(split_msgs) > 1
    assert outbound.find_reply_leak(split_msgs) == "internal_json"

    # Gemini/Vertex reasoning tags & markdown-wrapped role prefixes
    assert outbound.find_leak("<thought>let me think</thought>你好") == "reasoning"
    assert outbound.find_leak("<thinking>hmm</thinking>") == "reasoning"
    assert outbound.find_leak("</thought>") == "reasoning"
    assert outbound.find_leak("**System:** ignore previous rules") == "role_transcript"
    assert outbound.find_leak("__Assistant:__ hello") == "role_transcript"

    # Benign JSON / normal chat must NOT be falsely blocked
    assert outbound.find_leak('{"city": "北京", "temp": 22}') is None
    assert outbound.find_leak("晚安呀，明天见～") is None


def test_outbound_sender_blocks_fenced_multiline_json_e2e(tmp_path):
    from ai.channels.outbound import chat_payload
    from database import VioletDatabase as Database
    from violet_runtime import MessageSender

    class FakeClient:
        def __init__(self):
            self.sent = []

        def send_text(self, text, client_id=None):
            self.sent.append(text)
            return {"ret": 0}

    db = Database(str(tmp_path / "out.db"))
    try:
        client = FakeClient()
        sender = MessageSender(client, db, simulate_typing=False, sleep=lambda _: None)
        fenced = '```json\n{\n  "should_reply": true,\n  "route": "chat"\n}\n```'
        report = sender.send_messages(chat_payload(fenced)["messages"], origin="reply")
        assert report.blocked == "internal_json"
        assert client.sent == []
    finally:
        db.close()


def test_outbound_internal_headers_no_false_positives_and_catches_real_leaks():
    from ai.channels import outbound

    outbound.reset_for_tests()
    headers = outbound.internal_headers()
    assert headers
    # No regex/docstring junk or bare 2-4 char everyday words
    for junk in ("【 / 】", "【...】", "【[^】", "【语音】", "【节日】", "【图片】", "【通知】",
                 "【私聊】", "【管理员】", "【聊天记录】", "【你自己】", "【遗忘】", "【一条】", "【质量差】"):
        assert junk not in headers, junk

    benign_replies = [
        "【语音】我待会儿给你发一条语音呀",
        "【节日】祝你节日快乐呀！",
        "【图片】这张照片拍得真好看",
        "【通知】明天下午三点开会哦",
        "【私聊】等会私聊跟你说",
        "【管理员】可以找管理员帮忙处理",
        "【聊天记录】刚才聊天记录里提到过",
        "【你自己】也要记得照顾好自己呀",
        "【遗忘】别遗忘带伞哦",
        "【一条】就发这一条消息",
        "【质量差】这个东西质量差了点",
    ]
    for text in benign_replies:
        assert outbound.find_leak(text) is None, f"falsely blocked: {text}"

    real_leaks = [
        "【工具怎么用】这一轮你可以自己决定调用工具",
        "【外部工具数据｜来源 violet-web/web_search】结果",
        "【图片】需要配图或对方想看图时可用 create_image",
        "【语音】你会发语音（用你自己的声音）。平时聊天打字就好",
        "【聊天记录】（编号 时间 名字(QQ号)：内容）",
        "【管理员】（主人设的管理员：主人信任的人）",
        "【这次聊天的地方】你们现在是在 Telegram 上聊天",
    ]
    for text in real_leaks:
        assert outbound.find_leak(text) == "internal_header", f"missed leak: {text}"


# ---------------------------------------------------------------------------
# D4-05: TurnBuffer stale-timer generation guard
# ---------------------------------------------------------------------------

def test_turn_buffer_stale_timer_cannot_flush_or_leak_newer_timer():
    from violet_runtime import TurnBuffer

    timers = []

    class FakeTimer:
        def __init__(self, delay, fn):
            self.delay = delay
            self.fn = fn
            self.cancelled = False
            self.daemon = False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

        def fire(self):
            self.fn()

    emitted = []
    buf = TurnBuffer(emitted.append, normal_delay=5.0, extra_delay=10.0, timer_factory=FakeTimer)
    buf.add_text(1, "第一句")
    t1 = timers[0]

    # Simulate T1 having already woken right before add_text(2) acquires the lock
    buf.add_text(2, "第二句")
    t2 = timers[1]
    assert t1.cancelled is True and t2.cancelled is False

    # Stale T1 runs anyway -> must NOT flush and must NOT wipe self._timer (t2)
    t1.fire()
    assert emitted == []
    assert buf._timer is t2

    # Message 3 arrives -> must cancel T2 cleanly and schedule T3
    buf.add_text(3, "第三句")
    t3 = timers[2]
    assert t2.cancelled is True

    # Stale T2 firing must also be ignored
    t2.fire()
    assert emitted == []

    # Current T3 flushes all 3 messages together
    t3.fire()
    assert len(emitted) == 1
    assert emitted[0].inbound_ids == [1, 2, 3]
    assert emitted[0].texts == ["第一句", "第二句", "第三句"]


def test_turn_buffer_cancel_and_flush_now_invalidate_stale_timers():
    from violet_runtime import TurnBuffer

    timers = []

    class FakeTimer:
        def __init__(self, delay, fn):
            self.fn = fn
            self.cancelled = False
            self.daemon = False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    emitted = []
    buf = TurnBuffer(emitted.append, normal_delay=5.0, extra_delay=10.0, timer_factory=FakeTimer)
    buf.add_text(1, "hello")
    t1 = timers[-1]
    buf.cancel()
    buf.add_text(2, "after-cancel")
    t1.fn()  # stale pre-cancel timer must not flush message 2
    assert emitted == []
    buf.flush_now()
    assert len(emitted) == 1 and emitted[0].inbound_ids == [2]


# ---------------------------------------------------------------------------
# D4-06 & D4-12: Outbox ReadTimeout/504 -> 'unknown' & is_busy(scope)
# ---------------------------------------------------------------------------

def test_send_error_classification_read_timeout_and_504():
    from violet_runtime import is_retryable_send_error, is_uncertain_send_error

    def http_err(code: int) -> requests.HTTPError:
        resp = requests.Response()
        resp.status_code = code
        return requests.HTTPError(f"HTTP {code}", response=resp)

    assert is_retryable_send_error(requests.ConnectTimeout("connect")) is True
    assert is_uncertain_send_error(requests.ConnectTimeout("connect")) is False
    assert is_retryable_send_error(requests.ConnectionError("reset")) is True
    assert is_retryable_send_error(http_err(502)) is True
    assert is_retryable_send_error(http_err(503)) is True

    # Delivery-uncertain errors MUST NOT retry and MUST be marked unknown
    for exc in (requests.ReadTimeout("slow"), TimeoutError("timed out"), http_err(504)):
        assert is_retryable_send_error(exc) is False
        assert is_uncertain_send_error(exc) is True


def test_outbox_read_timeout_and_504_mark_text_and_image_unknown(tmp_path, monkeypatch):
    from ai.channels.outbound import _seal
    from database import VioletDatabase as Database
    from ai.images.media import MediaAssetStore as MediaStore
    from ai.reconcile import pending_items
    from violet_runtime import MessageSender

    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VIOLET_MEDIA_DIR", str(tmp_path / "media"))

    class FakeClient:
        def __init__(self):
            self.text_attempts = 0
            self.img_attempts = 0

        def send_text(self, text, client_id=None):
            self.text_attempts += 1
            resp = requests.Response()
            resp.status_code = 504
            raise requests.HTTPError("504 Gateway Timeout", response=resp)

        def send_image(self, path):
            self.img_attempts += 1
            raise requests.ReadTimeout("read timed out after upload")

    db = Database(str(tmp_path / "rt.db"))
    try:
        client = FakeClient()
        sender = MessageSender(client, db, simulate_typing=False, sleep=lambda _: None, max_attempts=3)

        # Text hitting HTTP 504 -> 1 attempt, status='unknown', visible in reconcile
        rep_txt = sender.send_messages(_seal([{"type": "text", "content": "早安"}]), origin="reply")
        assert rep_txt.ok is False
        assert client.text_attempts == 1
        out_txt = db.fetch_one("SELECT status, attempts FROM outbound_messages ORDER BY id DESC LIMIT 1")
        assert out_txt["status"] == "unknown" and out_txt["attempts"] == 1

        # Create an authorized AI-generated outbound PNG asset
        png = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        store = MediaStore(db, tmp_path / "media")
        asset = store.store_bytes(png, direction="outbound", media_kind="generated", ai_generated=True,
                                  task_id="t-1", caption="测试图")
        db.execute(
            "INSERT INTO image_tasks (task_id, origin, kind, status, turn_key, prompt, output_asset_ids, created_at, updated_at) "
            "VALUES ('t-1', 'chat', 'image_generate', 'succeeded', 'tk-1', 'cat', ?, ?, ?)",
            (json.dumps([asset["id"]]), db.now(), db.now()),
        )

        # Image hitting ReadTimeout -> 1 attempt, BOTH outbox & media_assets marked 'unknown'
        img_msg = _seal([{"type": "image", "asset_id": asset["id"], "turn_key": "tk-1"}])
        rep_img = sender.send_messages(img_msg, origin="reply")
        assert rep_img.ok is False
        assert client.img_attempts == 1
        row_img = db.fetch_one("SELECT send_status FROM media_assets WHERE id = ?", (asset["id"],))
        assert row_img["send_status"] == "unknown"

        pend = pending_items(db)
        assert len(pend["outbound"]) == 2
        assert len(pend["media"]) == 1 and pend["media"][0]["id"] == asset["id"]

        # Re-attempting the unknown asset must NEVER resend (skipped!)
        rep_img2 = sender.send_messages(_seal([{"type": "image", "asset_id": asset["id"], "turn_key": "tk-1"}]),
                                        origin="reply")
        assert client.img_attempts == 1
        assert len(rep_img2.skipped) == 1
    finally:
        db.close()


def test_runtime_is_busy_scope_sees_pre_worker_turn_queue(tmp_path):
    from violet_runtime import Turn, VioletRuntime

    rt = VioletRuntime.__new__(VioletRuntime)
    rt.buffers = {}
    rt.turn_queue = __import__("queue").Queue()
    rt._scope_queues = {}
    rt._workers_started = False
    rt._scope_of_channel = lambda ch: "stepsister" if ch == "qq" else "main"

    rt._enqueue_turn(Turn(inbound_ids=[1], texts=["hi"], received_at="2026-11-23T00:00:00", channel="qq"))
    assert rt.is_busy("stepsister") is True
    assert rt.is_busy("other") is False


# ---------------------------------------------------------------------------
# D4-02 & D4-09: OneBot / SnowLuma SSRF block & SnowLuma pairing commands
# ---------------------------------------------------------------------------

def test_onebot_and_snowluma_download_block_loopback_and_private_ssrf(tmp_path):
    from ai.channels.base import Part
    from ai.channels.onebot11 import OneBotChannel, OneBotError
    from ai.channels.snowluma import QQError, SnowLumaChannel

    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(b"INTERNAL_SECRET")

        def log_message(self, *args):
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        ob = OneBotChannel("qq", {}, {}, data_dir=tmp_path / "ob")
        sl = SnowLumaChannel("napcat", {}, {}, data_dir=tmp_path / "sl")

        bad_urls = [
            f"http://127.0.0.1:{port}/secret.jpg",
            "http://127.0.0.1/secret.jpg",
            "http://localhost/secret.jpg",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.1/internal.png",
            "http://[::1]/loopback.png",
        ]
        for url in bad_urls:
            with pytest.raises(OneBotError):
                ob.download(Part("image", "", {"url": url}))
            with pytest.raises(QQError):
                sl.download(Part("image", "", {"url": url}))
        assert hits == []
    finally:
        server.shutdown()
        server.server_close()


def test_snowluma_for_owner_accepts_chinese_and_case_insensitive_bind(tmp_path):
    from ai.channels.snowluma import SnowLumaChannel

    fake_identities = SimpleNamespace(bound_peers=lambda ch: {"99999"})
    sl = SnowLumaChannel("napcat", {}, {}, data_dir=tmp_path)
    sl.runtime = SimpleNamespace(identities=fake_identities)

    def seg(text: str):
        return [{"type": "text", "data": {"text": text}}]

    stranger = {"user_id": 12345}
    owner = {"user_id": 99999}

    assert sl._for_owner(stranger, seg("/bind 123456")) is True
    assert sl._for_owner(stranger, seg("/BIND 123456")) is True
    assert sl._for_owner(stranger, seg("绑定 123456")) is True
    assert sl._for_owner(stranger, seg("绑定123456")) is True
    assert sl._for_owner(stranger, seg("/start 123456")) is True
    assert sl._for_owner(stranger, seg("/bind wrong")) is True
    assert sl._for_owner(stranger, seg("绑定 wrong")) is True
    assert sl._for_owner(stranger, seg("你好呀")) is False
    assert sl._for_owner(stranger, seg("/start")) is False
    assert sl._for_owner(owner, seg("你好呀")) is True


# ---------------------------------------------------------------------------
# D4-03: Telegram concurrent download unique paths & parse_update dedup_key
# ---------------------------------------------------------------------------

def test_telegram_parse_update_sets_dedup_key_and_concurrent_downloads_do_not_collide(tmp_path):
    from ai.channels.telegram import TelegramChannel, parse_update

    u1 = {
        "update_id": 100,
        "message": {
            "message_id": 1,
            "chat": {"id": 42, "type": "private"},
            "from": {"id": 42, "is_bot": False},
            "photo": [{"file_id": "FILE_A", "file_unique_id": "uniq_A", "width": 100}],
        },
    }
    u2 = {
        "update_id": 101,
        "message": {
            "message_id": 2,
            "chat": {"id": 42, "type": "private"},
            "from": {"id": 42, "is_bot": False},
            "sticker": {"file_id": "FILE_B", "file_unique_id": "uniq_B", "emoji": "😀"},
        },
    }
    p1 = parse_update(u1, "tg1").parts[0]
    p2 = parse_update(u2, "tg1").parts[0]
    assert p1.data.get("dedup_key") and p2.data.get("dedup_key")
    assert p1.data["dedup_key"] != p2.data["dedup_key"]

    bytes_a = b"\x89PNG\r\n\x1a\n" + b"A" * 12000
    bytes_b = b"\x89PNG\r\n\x1a\n" + b"B" * 12000

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            fid = body.get("file_id")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "result": {"file_path": f"photos/{fid}.png"}}).encode())

        def do_GET(self):
            data = bytes_a if "FILE_A" in self.path else bytes_b
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    srv = ThreadedHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        tg = TelegramChannel("tg1", {"api_base": f"http://127.0.0.1:{port}"},
                             {"bot_token": "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"}, data_dir=tmp_path)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(tg.download, p1)
            f2 = pool.submit(tg.download, p2)
            path1, path2 = f1.result(), f2.result()
        assert path1 != path2
        assert path1.read_bytes() == bytes_a
        assert path2.read_bytes() == bytes_b

        # Even if part.data has NO dedup_key, two downloads must not collide
        from ai.channels.base import Part
        q1 = tg.download(Part("image", "", {"file_id": "FILE_A"}))
        q2 = tg.download(Part("image", "", {"file_id": "FILE_A"}))
        assert q1 != q2 and q1.read_bytes() == bytes_a and q2.read_bytes() == bytes_a
        assert not list(tg.media_dir.glob("*.tmp"))
    finally:
        srv.shutdown()
        srv.server_close()


# ---------------------------------------------------------------------------
# D4-04: Truncated WebP header bounds (MediaError, never IndexError / 1x1)
# ---------------------------------------------------------------------------

def test_truncated_webp_headers_raise_media_error_never_index_error(tmp_path):
    from database import VioletDatabase as Database
    from ai.images.chat_tools import intake_inbound_image
    from ai.images.media import MediaError, sniff_image

    for bad in (
        b"RIFF\x00\x00\x00\x00WEBPVP8L",
        b"RIFF\x00\x00\x00\x00WEBPVP8X",
        b"RIFF\x00\x00\x00\x00WEBPVP8 ",
        b"RIFF\x00\x00\x00\x00WEBPVP8L\x00\x00\x00\x00",
        b"RIFF\x00\x00\x00\x00WEBPVP8X" + b"\x00" * 10,
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00",
    ):
        with pytest.raises(MediaError) as ei:
            sniff_image(bad)
        assert ei.value.code == "invalid_image"

    # Valid VP8L (25 bytes) still parses cleanly
    # 1x1 lossless WebP: bit0..13 = width-1 = 0, bit14..27 = height-1 = 0
    valid_vp8l = b"RIFF\x11\x00\x00\x00WEBPVP8L\x05\x00\x00\x00\x2f\x00\x00\x00\x00"
    info = sniff_image(valid_vp8l)
    assert info.mime == "image/webp" and info.width == 1 and info.height == 1

    # intake_inbound_image must NOT raise on truncated VP8L
    bad_file = tmp_path / "bad.webp"
    bad_file.write_bytes(b"RIFF\x00\x00\x00\x00WEBPVP8L")
    db = Database(str(tmp_path / "img.db"))
    try:
        text = intake_inbound_image(db, inbound_id=1, path=bad_file)
        assert isinstance(text, str) and text
    finally:
        db.close()


# ---------------------------------------------------------------------------
# D4-07 & D4-11: Eve import zip-bomb, corrupt/encrypted zip, schema, SECRET_RE
# ---------------------------------------------------------------------------

def test_eve_import_blocks_zip_bomb_and_cumulative_cap(tmp_path, monkeypatch):
    from ai import eve_import

    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path))

    # 1. 20 KB zip -> 20 MB proactive_config.json bomb MUST be rejected
    z_bomb = io.BytesIO()
    with zipfile.ZipFile(z_bomb, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("proactive_config.json", b"{" + b" " * (20 * 1024 * 1024) + b"}")
    with pytest.raises(eve_import.ImportError_):
        eve_import.stage(io.BytesIO(z_bomb.getvalue()), "bomb.zip")

    # 2. Multi-member cumulative extraction exceeding MAX_UPLOAD MUST be rejected
    z_multi = io.BytesIO()
    with zipfile.ZipFile(z_multi, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("proactive_config.json", b"{" + b" " * (800 * 1024) + b"}")
        zf.writestr("Memories.db", b"SQLite format 3\x00" + b"\x00" * (800 * 1024))
    monkeypatch.setattr(eve_import, "MAX_UPLOAD", 1024 * 1024)
    with pytest.raises(eve_import.ImportError_):
        eve_import.stage(io.BytesIO(z_multi.getvalue()), "multi.zip")


def test_eve_import_handles_corrupt_zip_bad_sqlite_and_redacts_secrets(tmp_path, monkeypatch):
    from ai import eve_import
    from database import VioletDatabase as Database

    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path))

    # 1. Corrupt-CRC ZIP raises ImportError_ (not raw BadZipFile)
    z_bad = io.BytesIO()
    with zipfile.ZipFile(z_bad, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("proactive_config.json", b'{"enabled": true}')
    raw_bad = bytearray(z_bad.getvalue())
    # Corrupt compressed payload bytes near local header data
    raw_bad[40] ^= 0xFF
    with pytest.raises(eve_import.ImportError_):
        eve_import.stage(io.BytesIO(bytes(raw_bad)), "corrupt.zip")

    # 1b. Encrypted ZIP raises ImportError_ (not raw RuntimeError)
    raw_enc = bytearray(z_bad.getvalue())
    raw_enc[6] |= 0x01  # local file header encrypted bit
    cd_pos = raw_enc.find(b"PK\x01\x02")
    if cd_pos != -1:
        raw_enc[cd_pos + 8] |= 0x01  # central directory encrypted bit
    with pytest.raises(eve_import.ImportError_):
        eve_import.stage(io.BytesIO(bytes(raw_enc)), "encrypted.zip")

    # 2. Schema-mismatched SQLite raises ImportError_ in preview() and commit()
    bad_db = tmp_path / "bad_eve.db"
    conn = sqlite3.connect(bad_db)
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE core_memories (foo TEXT)")
    conn.commit()
    conn.close()

    tok = eve_import.stage(io.BytesIO(bad_db.read_bytes()), "Memories.db")
    vdb = Database(str(tmp_path / "violet.db"))
    try:
        with pytest.raises(eve_import.ImportError_):
            eve_import.preview(tok, vdb)
        with pytest.raises(eve_import.ImportError_):
            eve_import.commit(tok, vdb, channel="wechat", messages=False, memories=True, settings=[], actor="test")
    finally:
        vdb.close()

    # 3. SECRET_RE strips DB_PASS, API_AUTH, SESSION_COOKIE, USER_CRED
    parsed = eve_import.parse_env(
        "DB_PASS=secret1\nAPI_AUTH=secret2\nSESSION_COOKIE=secret3\nUSER_CRED=secret4\nSAFE_NUM=42\n"
    )
    assert parsed == {"SAFE_NUM": "42"}


# ---------------------------------------------------------------------------
# D4-10: Voice RIFF/WebP rejection, empty 200, queue cap, atomic save, chunks
# ---------------------------------------------------------------------------

def test_voice_hardening_webp_rejection_empty_200_chunks_atomic_save_and_queue_cap(tmp_path, monkeypatch):
    from ai import voice

    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path))

    # 1. WebP (RIFF....WEBP) rejected as WAV; real RIFF....WAVE accepted
    with pytest.raises(voice.VoiceError):
        voice.save_reference(b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 20, "bad")
    res = voice.save_reference(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 20, "ok")
    assert res["uploaded"] is True and res["file"] == "reference.wav"

    # 2. Failed write during replacement must NOT delete existing working reference
    orig_replace = voice.os.replace

    def boom_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(voice.os, "replace", boom_replace)
    with pytest.raises(OSError):
        voice.save_reference(b"ID3" + b"\x00" * 20, None)
    monkeypatch.setattr(voice.os, "replace", orig_replace)
    assert (tmp_path / "voice" / "reference.wav").is_file()

    # 3. speak() raises VoiceError on empty HTTP 200 body and oversized response
    class Handler(http.server.BaseHTTPRequestHandler):
        mode = "empty"

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            if Handler.mode == "oversize":
                self.wfile.write(b"x" * 200)

        def log_message(self, *args):
            pass

    srv = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        monkeypatch.setattr(voice, "service", lambda: {"url": f"http://127.0.0.1:{port}"})
        with pytest.raises(voice.VoiceError):
            voice.speak("你好", "silk")

        Handler.mode = "oversize"
        monkeypatch.setattr(voice, "MAX_SPEAK_BYTES", 100)
        with pytest.raises(voice.VoiceError):
            voice.speak("你好", "silk")
    finally:
        srv.shutdown()
        srv.server_close()

    # 4. chunks() does not emit bare punctuation on leading comma
    assert voice.chunks("，" + "啊" * 35, 30) == ["啊" * 30, "啊" * 5]
    assert voice.chunks("，，，", 30) == []

    # 5. say_later() returns False when pending queue would exceed MAX_PENDING_SENTENCES
    monkeypatch.setattr(voice, "ready", lambda: True)
    monkeypatch.setattr(voice, "MAX_PENDING_SENTENCES", 2)
    with voice._lock:
        old_pending = voice._pending[0]
        voice._pending[0] = 2
    try:
        assert voice.say_later("你好呀", lambda p, s: None) is False
    finally:
        with voice._lock:
            voice._pending[0] = old_pending
