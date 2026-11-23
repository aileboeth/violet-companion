"""Phase 4 Domain 2 security & ops regression suite.

Covers:
  - P0-1: Root helper file/directory symlink & NTFS junction LPE across all 5 tools + backup()
  - P1-1: log_viewer.py redaction (39-char Gemini AIza..., sk-ant-, xai-, sk-, mcp_secrets, proxies,
          snowluma/mcp-relay exports, colon headers, JSON fields, cookies, URL credentials)
  - P1-2: tls_server._redirect_to_https() bare LF / CRLF HTTP response splitting
  - P1-5: control_config.ControlStore._restore() 0o600 permission preservation on secrets.json
  - P1-6 & P2: auth.py atomic /auth/account validation, CRLF next rejection, URL-quoted next redirect,
          _same_origin() scheme check, and Origin: null rejection on /auth/login
  - P2 quick wins: persona sandbox <think> stripping, control.html stale copy removal, root_tasks
          urlparse loopback validation, updates.request_update() busy guard, diagnostics._plan_findings()
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

TOKEN = os.environ["VIOLET_ADMIN_TOKEN"]
SAME = {"Origin": "http://localhost"}
JSON = {"Accept": "application/json"}
PASSWORD = "StrongPass!2026#xy"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_dir_link(link: Path, target: Path) -> None:
    """Create a directory symlink, falling back to an NTFS junction on Windows without symlink privilege."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except OSError:
        if os.name == "nt":
            res = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                 capture_output=True, text=True)
            if res.returncode == 0:
                return
    pytest.skip("neither symlinks nor NTFS junctions are available in this environment")


def _try_file_symlink(link: Path, target: Path) -> bool:
    try:
        os.symlink(target, link)
        return True
    except OSError:
        return False


@pytest.fixture()
def env_guard(tmp_path, monkeypatch):
    import ai.control_config as cc
    import ai.tool_manager as tm
    saved = dict(os.environ)
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    cc._ORIGINALS.clear()
    cc._DEFAULT_STORE = None
    tm._STORE = None
    yield tmp_path / "data"
    os.environ.clear()
    os.environ.update(saved)
    cc._ORIGINALS.clear()
    cc._DEFAULT_STORE = None
    tm._STORE = None


@pytest.fixture()
def app(env_guard, violetdb):
    from wx_ilink_py import ILinkClient, create_app
    made = create_app(client=ILinkClient(data_dir=env_guard, auto_login=False), violetdb=violetdb,
                      proactive_scheduler=None, all_agents=[], admin_token=TOKEN)
    made.testing = True
    return made


# ==============================================================================
# P0-1: Root helper symlink / directory-symlink / junction LPE + backup()
# ==============================================================================

def test_p0_progress_write_and_env_progress_block_dir_symlink_escape(tmp_path, monkeypatch):
    from tools import violet_installer as inst

    outside = tmp_path / "root_etc_sim"
    outside.mkdir()
    root = tmp_path / "install_root"
    (root / "data").mkdir(parents=True)
    _make_dir_link(root / "data" / "control", outside)

    layout = inst.Layout(root)
    # Direct Progress.write must NOT write outside <data>
    prog = inst.Progress(inst.progress_path(layout, "task"), user=None, root=layout.data, task="relay-install")
    prog.done("attempt escape")
    assert not (outside / "task-progress.json").exists()
    assert list(outside.iterdir()) == []

    # Even without explicit root= argument, Progress(path) blocks symlinked control dir
    prog_no_root = inst.Progress(inst.progress_path(layout, "update"), user=None, task="update")
    prog_no_root(50, "attempt escape 2")
    assert not (outside / "update-progress.json").exists()

    # env_progress must reject when <data>/control is a symlink/junction
    monkeypatch.setenv("VIOLET_PROGRESS_FILE", str(inst.progress_path(layout, "update")))
    assert inst.env_progress(layout, {}).path is None
    monkeypatch.setenv("VIOLET_PROGRESS_FILE", str(outside / "update-progress.json"))
    assert inst.env_progress(layout, {}).path is None

    # take_update_request / take_task_request must not read or delete files in symlinked control dir
    (outside / "update-request").write_text("manual", encoding="utf-8")
    (outside / "task-request").write_text("relay-install", encoding="utf-8")
    assert inst.take_update_request(layout) is None
    assert inst.take_task_request(layout) is None
    assert (outside / "update-request").exists()
    assert (outside / "task-request").exists()


def test_p0_all_four_helper_exports_block_dir_symlink_and_use_oexcl(tmp_path):
    from tools import violet_relay, violet_searxng, violet_snowluma, violet_voice

    outside = tmp_path / "outside_root"
    outside.mkdir()
    root = tmp_path / "vroot"
    (root / "data").mkdir(parents=True)
    _make_dir_link(root / "data" / "control", outside)
    layout = SimpleNamespace(root=root, data=root / "data")

    # 1. violet_voice._export
    with pytest.raises(OSError):
        violet_voice._export(layout, {"voice": {"port": 18099, "model": "cosy"}, "user": None})
    assert not (outside / "voice.json").exists()

    # 2. violet_relay.export
    r_unit = tmp_path / "relay.service"
    r_data = tmp_path / "relay_data"
    r_data.mkdir()
    (r_data / "device.secret").write_text("relay-secret-123\n", encoding="utf-8")
    r_unit.write_text(f"[Service]\nExecStart=/bin/true -data-dir {r_data.as_posix()} -device-listen :18443 -client-listen 127.0.0.1:8787\n",
                      encoding="utf-8")
    with pytest.raises(OSError):
        violet_relay.export(layout, {"user": "violet", "public_ip": "1.2.3.4"}, unit_path=r_unit)
    assert not (outside / "mcp-relay.json").exists()

    # 3. violet_searxng._export & read_request
    with pytest.raises(OSError):
        violet_searxng._export(layout, {"searxng": {"port": 18089, "installed_at": 1}, "user": None},
                               {"results": 1, "note": "ok"})
    assert not (outside / "searxng.json").exists()
    (outside / "searxng-request.json").write_text('{"engines": "default"}', encoding="utf-8")
    with pytest.raises(violet_searxng.SearxngError):
        violet_searxng.read_request(layout.data)
    assert (outside / "searxng-request.json").exists()

    # 4. violet_snowluma.export
    with pytest.raises(OSError):
        violet_snowluma.export(layout, {"snowluma": {"ws_port": 3001, "ws_token": "tok", "vnc_port": 5900,
                                                     "vnc_password": "pass"}, "user": None})
    assert not (outside / "snowluma.json").exists()

    # Normal write in a genuine directory MUST use O_EXCL (and violet_relay must never call shutil.chown)
    clean_root = tmp_path / "clean"
    clean_layout = SimpleNamespace(root=clean_root, data=clean_root / "data")
    opened_flags: list[tuple[str, int]] = []
    real_open = os.open

    def spy_open(p, flags, *a, **kw):
        opened_flags.append((str(p), flags))
        return real_open(p, flags, *a, **kw)

    with patch("os.open", side_effect=spy_open):
        violet_relay.export(clean_layout, {"user": "violet", "public_ip": "1.2.3.4"}, unit_path=r_unit)
        violet_voice._export(clean_layout, {"voice": {"port": 18099, "model": "cosy"}, "user": None})
        violet_searxng._export(clean_layout, {"searxng": {"port": 18089, "installed_at": 1}, "user": None},
                               {"results": 1, "note": "ok"})
        violet_snowluma.export(clean_layout, {"snowluma": {"ws_port": 3001, "ws_token": "tok", "vnc_port": 5900,
                                                           "vnc_password": "pass"}, "user": None})
    write_opens = [f for _, f in opened_flags if (f & os.O_WRONLY) or (f & os.O_CREAT)]
    assert len(write_opens) >= 4
    assert all((f & os.O_EXCL) != 0 for f in write_opens)
    relay_src = (REPO_ROOT / "tools" / "violet_relay.py").read_text(encoding="utf-8")
    assert "shutil.chown" not in relay_src


def test_p0_file_symlinks_never_overwrite_victim_if_supported(tmp_path):
    from tools import violet_relay, violet_voice

    victim = tmp_path / "victim.txt"
    victim.write_text("DO_NOT_TOUCH", encoding="utf-8")
    root = tmp_path / "vroot"
    ctrl = root / "data" / "control"
    ctrl.mkdir(parents=True)
    if not _try_file_symlink(ctrl / "voice.json", victim):
        pytest.skip("file symlinks require privilege on this OS")

    layout = SimpleNamespace(root=root, data=root / "data")
    violet_voice._export(layout, {"voice": {"port": 18099, "model": "cosy"}, "user": None})
    assert victim.read_text(encoding="utf-8") == "DO_NOT_TOUCH"
    assert not (ctrl / "voice.json").is_symlink()

    victim2 = tmp_path / "victim2.txt"
    victim2.write_text("DO_NOT_TOUCH_2", encoding="utf-8")
    os.symlink(victim2, ctrl / "mcp-relay.json.tmp")
    r_unit = tmp_path / "relay.service"
    r_data = tmp_path / "relay_data"
    r_data.mkdir()
    (r_data / "device.secret").write_text("sec\n", encoding="utf-8")
    r_unit.write_text(f"[Service]\nExecStart=/bin/true -data-dir {r_data.as_posix()} -device-listen :18443 -client-listen 127.0.0.1:8787\n",
                      encoding="utf-8")
    violet_relay.export(layout, {"user": "violet", "public_ip": "1.2.3.4"}, unit_path=r_unit)
    assert victim2.read_text(encoding="utf-8") == "DO_NOT_TOUCH_2"


def test_p0_backup_skips_symlinked_files_and_dirs(tmp_path):
    from tools import violet_installer as inst

    root = tmp_path / "backup_root"
    (root / "data").mkdir(parents=True)
    (root / "config").mkdir(parents=True)
    (root / "install.json").write_text('{"version":"0.7.79"}', encoding="utf-8")
    (root / "data" / "legit.txt").write_text("safe data", encoding="utf-8")

    ext_dir = tmp_path / "etc_sim"
    ext_dir.mkdir()
    ext_file = ext_dir / "shadow.txt"
    ext_file.write_text("root:$6$SUPERSECRET_SHADOW_HASH:19999\n", encoding="utf-8")

    _make_dir_link(root / "data" / "loot_dir", ext_dir)
    _try_file_symlink(root / "data" / "loot_file.txt", ext_file)

    archive = inst.backup(inst.Layout(root), label="sec")
    with tarfile.open(archive, "r:gz") as tf:
        names = tf.getnames()
        assert "data/legit.txt" in names
        assert not any("shadow.txt" in n or "loot_dir" in n or "loot_file.txt" in n for n in names)


# ==============================================================================
# P1-1: ai/log_viewer.py secret redaction
# ==============================================================================

def test_p1_log_viewer_redacts_gemini_provider_keys_stores_and_headers(app, env_guard, monkeypatch):
    from ai import log_viewer
    from ai.control_config import ControlStore
    from ai.tool_manager import tool_store

    cs = ControlStore(env_guard)
    cs.save({"LLM_API_KEY": "sk-old-rotated-secret-0001"}, "test", "seed1")
    cs.save({"LLM_API_KEY": "sk-live-known-secret-1234567890"}, "test", "seed2")

    ts = tool_store()
    ts.replace_secret("android-phone", "X-Api-Key", "mcp_hdr_secret_987654321", "test")
    ts.replace_web_key("BSA1234567890abcdef1234567890ab", "test")

    (env_guard / "control" / "snowluma.json").write_text(json.dumps({
        "installed": True, "ws_url": "ws://127.0.0.1:18080", "ws_token": "ws_tok_short_998877",
        "vnc_port": 5900, "vnc_password": "VncPass8",
    }), encoding="utf-8")
    (env_guard / "control" / "mcp-relay.json").write_text(json.dumps({
        "installed": True, "device_secret": "dev_sec_short_445566", "client_url": "http://127.0.0.1:8787/mcp",
    }), encoding="utf-8")
    monkeypatch.setenv("HTTPS_PROXY", "http://alice:MyProxyPass99@proxy.corp.internal:8080")
    monkeypatch.setenv("ALL_PROXY", "socks5://bob:AllProxyPass77@127.0.0.1:1080")

    known = log_viewer.known_secrets(env_guard)
    for expected in (
        "sk-live-known-secret-1234567890",
        "sk-old-rotated-secret-0001",
        "mcp_hdr_secret_987654321",
        "BSA1234567890abcdef1234567890ab",
        "ws_tok_short_998877",
        "VncPass8",
        "dev_sec_short_445566",
        "MyProxyPass99",
        "AllProxyPass77",
    ):
        assert expected in known, expected

    probes = [
        ("400 Bad Request for key AIzaSyD4xK9pQ2mR7vT1wY5zB8cN3fH6jL0kM9n", "AIzaSyD4xK9pQ2mR7vT1wY5zB8cN3fH6jL0kM9n"),
        ('{"x-goog-api-key": "AIzaSyD4xK9pQ2mR7vT1wY5zB8cN3fH6jL0kM9n"}', "AIzaSyD4xK9pQ2mR7vT1wY5zB8cN3fH6jL0kM9n"),
        ("x-goog-api-key: AIzaSyD4xK9pQ2mR7vT1wY5zB8cN3fH6jL0kM9n", "AIzaSyD4xK9pQ2mR7vT1wY5zB8cN3fH6jL0kM9n"),
        ('{"x-api-key": "sk-ant-api-AbCdEfGhIjKlMnOpQrStUvWxYz"}', "sk-ant-api-AbCdEfGhIjKlMnOpQrStUvWxYz"),
        ("key xai-AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGhI", "xai-AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGhI"),
        ('{"api_key": "sk-AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGh"}', "sk-AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGh"),
        ("ProxyError via http://alice:MyProxyPass99@proxy.corp.internal:8080", "MyProxyPass99"),
        ("ProxyError via http://user:OtherProxySecret@proxy.other:8080", "OtherProxySecret"),
        ('{"api_key": "my_secret_key_99887766", "password": "MyPassword123!"}', "MyPassword123!"),
        ('{"api_key": "my_secret_key_99887766", "password": "MyPassword123!"}', "my_secret_key_99887766"),
        ("Authorization: Basic YWRtaW46c3VwZXJzZWNyZXQ=", "YWRtaW46c3VwZXJzZWNyZXQ="),
        ("Cookie: violet_admin_session=eyJ2aW9sZXQiOjF9.Z1abc2.shortSig99", "eyJ2aW9sZXQiOjF9"),
        ("ws_token=ws_tok_short_998877 vnc_password: VncPass8", "VncPass8"),
        ("ws_token=unlisted_ws_tok_123456 vnc_password: UnlistedPass9", "unlisted_ws_tok_123456"),
        ("device_secret: dev_sec_short_445566", "dev_sec_short_445566"),
        ("X-Api-Key: mcp_hdr_secret_987654321 refused", "mcp_hdr_secret_987654321"),
        ("Brave error with key BSA1234567890abcdef1234567890ab", "BSA1234567890abcdef1234567890ab"),
    ]
    for sample, secret in probes:
        redacted = log_viewer.redact(sample, known)
        assert secret not in redacted, (sample, redacted)

    # Preserve Bearer <redacted> formatting
    assert "Bearer <redacted>" in log_viewer.redact("Authorization: Bearer my-bearer-secret-12345", known)

    # End-to-end download & search oracle check
    log_file = env_guard / "main.log"
    monkeypatch.setenv("VIOLET_LOG_FILE", str(log_file))
    log_file.write_text(
        "2026-11-23 10:00:00,000 ERROR [violet.llm] Gemini AIzaSyD4xK9pQ2mR7vT1wY5zB8cN3fH6jL0kM9n\n"
        "2026-11-23 10:00:01,000 ERROR [violet.net] Proxy http://alice:MyProxyPass99@proxy.corp:8080\n"
        "2026-11-23 10:00:02,000 ERROR [violet.http] Cookie: violet_admin_session=eyJ2aW9sZXQiOjF9.abc.def\n",
        encoding="utf-8",
    )
    c = app.test_client()
    hdrs = {"Authorization": f"Bearer {TOKEN}"}
    dl = c.get("/api/control/logs/download", headers=hdrs).get_data(as_text=True)
    for leaked in ("AIzaSyD4xK9pQ2mR7vT1wY5zB8cN3fH6jL0kM9n", "MyProxyPass99", "eyJ2aW9sZXQiOjF9"):
        assert leaked not in dl
    assert c.get("/api/control/logs?q=AIzaSyD4xK9pQ2mR7vT1wY5zB8cN3fH6jL0kM9n", headers=hdrs).get_json()["data"]["entries"] == []


# ==============================================================================
# P1-2: wx_ilink_py/tls_server.py HTTP response splitting
# ==============================================================================

@pytest.mark.parametrize("payload", [
    b"GET /ok\n HTTP/1.1\r\nHost: example.com\r\n\r\n",
    b"GET /ok HTTP/1.1\r\nHost: example.com\n\r\n\r\n",
    b"GET /ok\nX-Injected:1 HTTP/1.1\r\nHost: example.com\r\n\r\n",
    b"GET /ok HTTP/1.1\r\nHost: example.com\nX-Injected:1\r\n\r\n",
])
def test_p1_tls_redirect_rejects_bare_lf_response_splitting(payload):
    from wx_ilink_py import tls_server

    s1, s2 = socket.socketpair()
    try:
        t = threading.Thread(target=tls_server._redirect_to_https, args=(s1, 8443))
        t.start()
        s2.sendall(payload)
        s2.shutdown(socket.SHUT_WR)
        raw = b""
        while True:
            chunk = s2.recv(4096)
            if not chunk:
                break
            raw += chunk
        t.join(timeout=3)
    finally:
        s1.close()
        s2.close()
    assert raw.startswith(b"HTTP/1.1 400")
    assert b"Location:" not in raw


# ==============================================================================
# P1-5: ai/control_config.py _restore() 0o600 permissions
# ==============================================================================

def test_p1_control_config_restore_preserves_0600_on_secrets_json(env_guard):
    from ai.control_config import ControlStore

    cs = ControlStore(env_guard)
    cs.save({"DEBOUNCE_NORMAL": 7.0, "LLM_API_KEY": "sk-initial-key-11111111"}, "test", "init")
    raw_sec_before = cs.secrets_path.read_bytes()

    chmod_calls: list[tuple[str, int]] = []
    real_chmod = os.chmod

    def spy_chmod(p, mode, *a, **kw):
        chmod_calls.append((str(p), mode))
        return real_chmod(p, mode, *a, **kw)

    def boom():
        if cs.values().get("DEBOUNCE_NORMAL") == 12.0:
            raise RuntimeError("simulated apply crash")

    with patch("os.chmod", side_effect=spy_chmod):
        with pytest.raises(RuntimeError):
            cs.save({"DEBOUNCE_NORMAL": 12.0, "LLM_API_KEY": "sk-new-key-22222222222"}, "test", "fail", apply=boom)

    assert cs.secrets_path.read_bytes() == raw_sec_before
    secret_chmods = [(p, m) for p, m in chmod_calls if "secrets.json" in p and m == 0o600]
    # Once during _write, once during _restore!
    assert len(secret_chmods) >= 2
    if os.name != "nt":
        assert (cs.secrets_path.stat().st_mode & 0o777) == 0o600


# ==============================================================================
# P1-6 & P2: wx_ilink_py/auth.py hardening
# ==============================================================================

def test_p1_auth_account_validates_all_fields_before_mutation_no_self_lockout(app):
    store = app.extensions["violet_admin_store"]
    store.set_username("violet_admin1")
    store.set_password(PASSWORD)
    entrance = store.set_entrance("k3x9q2m7wz8a")

    c = app.test_client()
    assert c.get(f"/{entrance}").status_code == 302
    assert c.post("/auth/login", data={"username": "violet_admin1", "password": PASSWORD}, headers=SAME).status_code == 302

    ver_before = store.session_version()
    # Valid username + invalid reserved entrance -> 400, zero state change, session stays valid!
    r = c.post("/auth/account",
               json={"current": PASSWORD, "username": "violet_renamed2", "entrance": "api"},
               headers=SAME)
    assert r.status_code == 400
    assert store.username() == "violet_admin1"
    assert store.entrance() == entrance
    assert store.session_version() == ver_before
    assert c.get("/auth/status", headers=JSON).status_code == 200

    # Valid username + weak password -> 400, zero state change, session stays valid!
    r2 = c.post("/auth/account",
                json={"current": PASSWORD, "username": "violet_renamed2", "password": "short"},
                headers=SAME)
    assert r2.status_code == 400
    assert store.username() == "violet_admin1"
    assert store.session_version() == ver_before
    assert c.get("/auth/status", headers=JSON).status_code == 200

    # Valid username + valid entrance -> succeeds atomically and refreshes session & entrance mark
    r3 = c.post("/auth/account",
                json={"current": PASSWORD, "username": "violet_renamed2", "entrance": "new_entrance_99"},
                headers=SAME)
    assert r3.status_code == 200
    assert sorted(r3.get_json()["changed"]) == ["entrance", "username"]
    assert store.username() == "violet_renamed2"
    assert store.entrance() == "new_entrance_99"
    assert c.get("/auth/status", headers=JSON).status_code == 200


def test_p2_auth_crlf_next_quoted_redirect_scheme_csrf_and_origin_null(app):
    store = app.extensions["violet_admin_store"]
    store.set_username("violet_admin1")
    store.set_password(PASSWORD)
    entrance = store.set_entrance("k3x9q2m7wz8a")

    c = app.test_client()
    c.get(f"/{entrance}")

    # 1. CRLF in login next must fall back to /control (302, no 500, no header injection)
    r_crlf = c.post("/auth/login",
                    data={"username": "violet_admin1", "password": PASSWORD, "next": "/control\r\nX-Injected: pwn"},
                    headers=SAME)
    assert r_crlf.status_code == 302
    assert r_crlf.headers["Location"] == "/control"

    # 2. Unauthenticated HTML redirect quotes next query params so &lang=ru cannot mutate violet_lang cookie
    c_anon = app.test_client()
    c_anon.get(f"/{entrance}")
    r_unauth = c_anon.get("/control?foo=1&lang=ru", headers={"Accept": "text/html"})
    assert r_unauth.status_code == 302
    loc = r_unauth.headers["Location"]
    assert "&lang=ru" not in loc
    r_follow = c_anon.get(loc)
    assert not any("violet_lang=ru" in h for h in r_follow.headers.getlist("Set-Cookie"))

    # 3. _same_origin() enforces URL scheme: HTTPS POST with Origin: http://localhost -> 403
    r_scheme_bad = c.post("/api/control/permissions",
                          base_url="https://localhost",
                          json={"ability": "qq_touches", "level": "forbid"},
                          headers={"Origin": "http://localhost"})
    assert r_scheme_bad.status_code == 403
    r_scheme_ok = c.post("/api/control/permissions",
                         base_url="https://localhost",
                         json={"ability": "qq_touches", "level": "forbid"},
                         headers={"Origin": "https://localhost"})
    assert r_scheme_ok.status_code == 200

    # 4. POST /auth/login rejects Origin: null and cross-scheme Origin
    c_null = app.test_client()
    c_null.get(f"/{entrance}")
    assert c_null.post("/auth/login",
                       data={"username": "violet_admin1", "password": PASSWORD},
                       headers={"Origin": "null"}).status_code == 403
    assert c_null.post("/auth/login",
                       base_url="https://localhost",
                       data={"username": "violet_admin1", "password": PASSWORD},
                       headers={"Origin": "http://localhost"}).status_code == 403


# ==============================================================================
# P2 quick wins: sandbox <think>, control.html, root_tasks, updates, diagnostics
# ==============================================================================

def test_p2_persona_sandbox_strips_think_tags(env_guard, violetdb):
    from ai.agents.chat_agent import ChatAgent
    from wx_ilink_py import ILinkClient, create_app

    chat_agent = ChatAgent()
    app = create_app(client=ILinkClient(data_dir=env_guard, auto_login=False), violetdb=violetdb,
                     proactive_scheduler=None, all_agents=[chat_agent], admin_token=TOKEN)
    app.testing = True

    def fake_create(client, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="<think>internal secret reasoning</think>\n早呀，今天睡得好吗？"
            ))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )

    c = app.test_client()
    with patch.object(chat_agent, "_get_client", return_value=object()), \
         patch.object(chat_agent, "_create_completion_with_retry", side_effect=fake_create):
        resp = c.post("/api/persona/sandbox",
                      json={"message": "你好", "template": "stepsister", "stage": 0},
                      headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert "<think>" not in (body.get("raw") or "")
    assert all("<think>" not in m for m in body.get("messages") or [])
    assert any("早呀" in m for m in body.get("messages") or [])


def test_p2_control_html_removes_stale_proactive_copy():
    html = (REPO_ROOT / "wx_ilink_py" / "templates" / "control.html").read_text(encoding="utf-8")
    assert "Memory / Proactive" not in html
    assert "Memory 默认 0.2" in html


def test_p2_root_tasks_exports_validate_loopback_with_urlparse(env_guard):
    from ai import root_tasks

    ctrl = env_guard / "control"
    ctrl.mkdir(parents=True, exist_ok=True)

    bad_http_urls = [
        "http://127.0.0.1:@evil.com/",
        "http://127.0.0.1:80@evil.com/",
        "http://user:pass@127.0.0.1:18089/",
        "https://127.0.0.1:18089/",
        "http://127.0.0.1:0/",
        "http://127.0.0.1:70000/",
        "http://127.0.0.1/",
        "http://evil.com:18089/",
    ]
    for bad in bad_http_urls:
        (ctrl / "searxng.json").write_text(json.dumps({"url": bad}), encoding="utf-8")
        (ctrl / "voice.json").write_text(json.dumps({"url": bad}), encoding="utf-8")
        assert root_tasks.searxng_export() is None, bad
        assert root_tasks.voice_export() is None, bad

    for bad_ws in ("ws://127.0.0.1:@evil.com/", "ws://user:pass@127.0.0.1:3001/", "http://127.0.0.1:3001/",
                   "ws://127.0.0.1:0/"):
        (ctrl / "snowluma.json").write_text(json.dumps({"ws_url": bad_ws, "vnc_port": 5900}), encoding="utf-8")
        assert root_tasks.snowluma_export() is None, bad_ws

    # Legitimate loopback exports pass
    (ctrl / "searxng.json").write_text(json.dumps({"url": "http://127.0.0.1:18089"}), encoding="utf-8")
    (ctrl / "voice.json").write_text(json.dumps({"url": "http://127.0.0.1:18095"}), encoding="utf-8")
    (ctrl / "snowluma.json").write_text(json.dumps({"ws_url": "ws://127.0.0.1:3001", "vnc_port": 5900}), encoding="utf-8")
    assert root_tasks.searxng_export()["url"] == "http://127.0.0.1:18089"
    assert root_tasks.voice_export()["url"] == "http://127.0.0.1:18095"
    assert root_tasks.snowluma_export()["ws_url"] == "ws://127.0.0.1:3001"


def test_p2_updates_request_update_does_not_clobber_running_progress(env_guard, tmp_path):
    from ai import root_tasks, updates

    fake_root = tmp_path / "fake_install"
    fake_app = fake_root / "app"
    fake_app.mkdir(parents=True)
    (fake_root / "install.json").write_text(json.dumps({"auto_update_agent": "systemd"}), encoding="utf-8")
    (fake_app / "VERSION").write_text("0.7.79", encoding="utf-8")

    ctrl = env_guard / "control"
    ctrl.mkdir(parents=True, exist_ok=True)
    now = int(time.time())
    (ctrl / "update-progress.json").write_text(json.dumps({
        "task": "update", "state": "running", "percent": 65, "step": "安装依赖", "at": now, "started_at": now,
    }), encoding="utf-8")

    res = updates.request_update("manual", root=fake_app)
    assert res["requested"] is True and res.get("already_running") is True
    prog = root_tasks.read_progress("update")
    assert prog["state"] == "running" and prog["percent"] == 65


def test_p2_diagnostics_plan_findings_flags_proactive_only_tools_correctly():
    from ai import diagnostics

    fake_cat = SimpleNamespace(tools={
        "srv__proactive_only": SimpleNamespace(name="srv__proactive_only", agents=["proactive"]),
        "srv__no_agents": SimpleNamespace(name="srv__no_agents", agents=[]),
        "srv__chat_only": SimpleNamespace(name="srv__chat_only", agents=["chat"]),
        "srv__both": SimpleNamespace(name="srv__both", agents=["chat", "proactive"]),
    })
    with patch("ai.tool_manager.discover", return_value=fake_cat):
        findings = diagnostics._plan_findings()
    assert len(findings) == 1
    msg = findings[0]["message"]
    assert "srv__proactive_only" in msg
    assert "srv__no_agents" not in msg
    assert "srv__chat_only" not in msg
    assert "srv__both" not in msg


def test_p0_posix_dir_fd_and_fchown_path_simulated(tmp_path, monkeypatch):
    """Exercise the POSIX O_DIRECTORY|O_NOFOLLOW + dir_fd + fchown + atomic replace branch even on Windows."""
    import io
    from tools import violet_installer, violet_relay, violet_searxng, violet_snowluma, violet_voice

    events: list[tuple] = []
    next_fd = [100]
    files: dict[int, io.StringIO] = {}

    def fake_open(path, flags, mode=0o777, *, dir_fd=None):
        events.append(("open", str(path), flags, mode, dir_fd))
        fd = next_fd[0]
        next_fd[0] += 1
        return fd

    class FakeFH:
        def __init__(self, fd):
            self._fd = fd
            self.buf = io.StringIO()

        def fileno(self):
            return self._fd

        def write(self, s):
            return self.buf.write(s)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            files[self._fd] = self.buf
            return False

    def fake_fdopen(fd, *a, **kw):
        return FakeFH(fd)

    def fake_close(fd):
        events.append(("close", fd))

    def fake_fchmod(fd, mode):
        events.append(("fchmod", fd, oct(mode)))

    def fake_fchown(fd, uid, gid):
        events.append(("fchown", fd, uid, gid))

    def fake_replace(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        events.append(("replace", str(src), str(dst), src_dir_fd, dst_dir_fd))

    def fake_unlink(path, *, dir_fd=None):
        events.append(("unlink", str(path), dir_fd))

    O_DIRECTORY = 0x10000
    O_NOFOLLOW = 0x20000
    monkeypatch.setattr(os, "O_DIRECTORY", O_DIRECTORY, raising=False)
    monkeypatch.setattr(os, "O_NOFOLLOW", O_NOFOLLOW, raising=False)
    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "fdopen", fake_fdopen)
    monkeypatch.setattr(os, "close", fake_close)
    monkeypatch.setattr(os, "fchmod", fake_fchmod, raising=False)
    monkeypatch.setattr(os, "fchown", fake_fchown, raising=False)
    monkeypatch.setattr(os, "replace", fake_replace)
    monkeypatch.setattr(os, "unlink", fake_unlink)
    monkeypatch.setattr(os, "supports_dir_fd", {fake_open, fake_replace, fake_unlink})

    data_dir = tmp_path / "data"
    (data_dir / "control").mkdir(parents=True)

    for mod, call in (
        (violet_installer, lambda: violet_installer._safe_write_owned(data_dir / "control" / "p.json", '{"a":1}', root=data_dir, user="violet", mode=0o640)),
        (violet_relay, lambda: violet_relay._safe_write_owned(data_dir / "control" / "r.json", '{"a":1}', root=data_dir, user="violet", mode=0o600)),
        (violet_voice, lambda: violet_voice._safe_write_owned(data_dir / "control" / "v.json", '{"a":1}', root=data_dir, user="violet", mode=0o640)),
        (violet_searxng, lambda: violet_searxng._write_owned(data_dir / "control" / "s.json", '{"a":1}', "violet", mode=0o640, root=data_dir)),
        (violet_snowluma, lambda: violet_snowluma._write_json(data_dir / "control" / "q.json", {"a": 1}, 1001, 1002, mode=0o640, root=data_dir)),
    ):
        events.clear()
        if hasattr(mod, "_resolve_ids"):
            monkeypatch.setattr(mod, "_resolve_ids", lambda u: (1001, 1002))
        call()
        dir_opens = [e for e in events if e[0] == "open" and (e[2] & O_DIRECTORY)]
        file_opens = [e for e in events if e[0] == "open" and (e[2] & os.O_CREAT)]
        replaces = [e for e in events if e[0] == "replace"]
        fchowns = [e for e in events if e[0] == "fchown"]
        assert len(dir_opens) == 2, (mod.__name__, events)
        assert all((e[2] & O_NOFOLLOW) != 0 for e in dir_opens)
        assert len(file_opens) == 1
        assert (file_opens[0][2] & (os.O_EXCL | O_NOFOLLOW)) == (os.O_EXCL | O_NOFOLLOW)
        assert file_opens[0][4] is not None  # opened with dir_fd!
        assert len(fchowns) == 1 and fchowns[0][2:] == (1001, 1002)
        assert len(replaces) == 1 and replaces[0][3] == replaces[0][4] == file_opens[0][4]

