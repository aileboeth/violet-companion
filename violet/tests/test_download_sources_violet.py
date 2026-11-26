"""Download sources and signed releases (Violet 0.7.59; user 2026-11-08: updates, the phone relay, web search and voice
must work on servers in mainland China through GitHub mirrors, and a mirror must not be able to change the program).
Offline: nothing here contacts GitHub or a mirror."""
from __future__ import annotations

import base64
import json

import pytest

from tools import violet_installer as inst
from tools import violet_sign as sign
from tools import violet_sources as vs


# ---------------------------------------------------------------------------------------------- Ed25519

# RFC 8032 section 7.1, tests 1-3
VECTORS = [
    ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
     "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
     "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
    ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
     "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "72",
     "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"),
    ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
     "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025", "af82",
     "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"),
]


@pytest.mark.parametrize("seed, public, message, signature", VECTORS)
def test_ed25519_matches_rfc_8032(seed, public, message, signature):
    seed, public, message, signature = map(bytes.fromhex, (seed, public, message, signature))
    assert sign.public_key(seed) == public and sign.sign(seed, message) == signature
    assert sign.verify(public, message, signature)
    assert not sign.verify(public, message + b"!", signature)
    assert not sign.verify(public, message, signature[:63] + bytes([signature[63] ^ 1]))


def test_check_reads_sig_files_and_refuses_junk():
    seed = bytes(range(32))
    public = base64.b64encode(sign.public_key(seed)).decode()
    text = sign.signature_text(seed, b"release")
    assert text.endswith("\n") and sign.check(public, b"release", text) and sign.check(public, b"release", text.encode())
    for bad in (None, "", "not base64 !!", base64.b64encode(b"short").decode()):
        assert not sign.check(public, b"release", bad)
    assert not sign.check("__VIOLET_RELEASE_KEY__", b"release", text)       # an unset key never accepts
    other = base64.b64encode(sign.public_key(bytes(32))).decode()
    assert not sign.check(other, b"release", text)


def test_the_real_public_keys_are_set():
    for key in (sign.VIOLET_RELEASE_KEY, sign.MCP_RELAY_KEY):
        assert len(base64.b64decode(key, validate=True)) == 32
    assert sign.VIOLET_RELEASE_KEY != sign.MCP_RELAY_KEY


def test_keygen_never_prints_or_overwrites_the_private_key(tmp_path, capsys):
    key = tmp_path / "k.key"
    assert sign.main(["keygen", str(key)]) == 0
    printed = capsys.readouterr().out.strip()
    seed = bytes.fromhex(key.read_text(encoding="ascii").strip())
    assert printed == base64.b64encode(sign.public_key(seed)).decode() and seed.hex() not in printed
    with pytest.raises(SystemExit, match="not overwritten"):
        sign.main(["keygen", str(key)])
    target = tmp_path / "release.json"
    target.write_bytes(b"{}")
    assert sign.main(["sign", "--key-file", str(key), str(target)]) == 0
    assert sign.main(["verify", "--public", printed, str(target)]) == 0
    target.write_bytes(b"{ }")
    assert sign.main(["verify", "--public", printed, str(target)]) == 1


# ---------------------------------------------------------------------------------------------- sources

def test_the_automatic_order_is_github_then_the_built_in_mirrors():
    got = vs.sources("auto")
    assert [s.key for s in got] == ["github", "jackyu", "ip8217"] and got[0].direct
    assert got[1].url("https://github.com/a/b") == "https://p.jackyu.cn/https://github.com/a/b"
    assert got[2].url("https://github.com/a/b") == "http://8.217.135.233:9999/https://github.com/a/b"
    assert [s.key for s in vs.sources("github")] == ["github"]
    assert [s.key for s in vs.sources("ip8217")] == ["ip8217"]
    assert [s.key for s in vs.sources("nonsense")] == ["github", "jackyu", "ip8217"]


@pytest.mark.parametrize("text, prefix", [
    ("https://gh.example.com", "https://gh.example.com/"), ("http://1.2.3.4:8080/", "http://1.2.3.4:8080/"),
    ("https://gh.example.com/proxy/", "https://gh.example.com/proxy/"),
    ("https://user:pw@gh.example.com/", None), ("https://gh.example.com/?x=1", None), ("ftp://gh.example.com/", None),
    ("https://github.com/", None), ("", None), ("https://gh.example.com/a b", None),
])
def test_a_custom_mirror_address_is_checked(text, prefix):
    assert vs.custom_checked(text) == prefix


def test_custom_is_used_only_where_the_download_can_be_checked():
    custom = vs.sources("custom", "https://gh.example.com")
    assert [s.prefix for s in custom] == ["https://gh.example.com/"] and "gh.example.com" in custom[0].label
    assert [s.key for s in vs.sources("custom", "https://gh.example.com", allow_custom=False)] == \
        ["github", "jackyu", "ip8217"]
    assert [s.key for s in vs.sources("custom", "not an address")] == ["github", "jackyu", "ip8217"]


def test_first_tries_each_source_and_lists_every_reason():
    import socket
    import urllib.error
    calls = []

    def attempt(source):
        calls.append(source.key)
        if source.key == "github":
            raise socket.timeout("timed out")
        if source.key == "jackyu":
            raise urllib.error.HTTPError("u", 502, "bad", {}, None)
        return "ok"
    assert vs.first(vs.sources("auto"), attempt) == ("ok", vs.BUILT_IN[1]) and calls == ["github", "jackyu", "ip8217"]
    with pytest.raises(vs.SourcesFailed) as info:
        vs.first(vs.sources("auto")[:2], attempt)
    assert str(info.value) == "GitHub：连接超时；加速站 p.jackyu.cn：返回错误 502"


# ---------------------------------------------------------------------------------------------- the setting

def test_the_root_service_reads_the_pages_choice_from_the_service_files(tmp_path):
    layout = inst.Layout(tmp_path / "violet")
    (layout.data / "control").mkdir(parents=True)
    layout.config.mkdir(parents=True)
    assert [s.key for s in inst.download_sources(layout)] == ["github", "jackyu", "ip8217"]      # nothing set
    layout.env_file.write_text("VIOLET_DOWNLOAD_SOURCE=github\n", encoding="utf-8")
    assert [s.key for s in inst.download_sources(layout)] == ["github"]
    (layout.data / "control" / "settings.json").write_text(json.dumps({"values": {
        "VIOLET_DOWNLOAD_SOURCE": "custom", "VIOLET_DOWNLOAD_MIRROR": "https://gh.example.com/"}}), encoding="utf-8")
    assert [s.prefix for s in inst.download_sources(layout)] == ["https://gh.example.com/"]
    # web search: nothing can check it -> never the address the (unprivileged) page wrote
    assert "custom" not in [s.key for s in inst.download_sources(layout, allow_custom=False)]
    (layout.data / "control" / "settings.json").write_text(json.dumps({"values": {
        "VIOLET_DOWNLOAD_SOURCE": "custom", "VIOLET_DOWNLOAD_MIRROR": "https://x@evil/;rm"}}), encoding="utf-8")
    assert [s.key for s in inst.download_sources(layout)] == ["github", "jackyu", "ip8217"]


def test_the_settings_page_offers_every_source_with_auto_as_default():
    from ai import control_config as cc
    setting = cc.BY_KEY["VIOLET_DOWNLOAD_SOURCE"]
    assert setting.default == "auto" and [v for v, _ in setting.choices] == list(vs.KEYS)
    assert setting.group == cc.BY_KEY["VIOLET_UPDATE_MODE"].group == "updates"
    assert cc.validate("VIOLET_DOWNLOAD_MIRROR", "https://gh.example.com") == "https://gh.example.com/"
    for bad in ("https://github.com/x", "https://a@b.example/"):
        with pytest.raises(ValueError):
            cc.validate("VIOLET_DOWNLOAD_MIRROR", bad)
    with pytest.raises(ValueError):
        cc.validate("VIOLET_DOWNLOAD_SOURCE", "somewhere")


def test_the_control_centers_check_uses_the_pages_choice(monkeypatch, tmp_path):
    from ai import updates
    seen = {}
    monkeypatch.setattr(inst, "latest_release", lambda repo=None, **kw: seen.update(kw) or {"version": "9.9.9"})
    monkeypatch.setenv("VIOLET_DOWNLOAD_SOURCE", "jackyu")
    updates._fetch_default(tmp_path / "app")()
    assert [s.key for s in seen["sources"]] == ["jackyu"]


# ---------------------------------------------------------------------------------------------- web search, voice

def test_web_search_source_never_comes_from_the_custom_address(monkeypatch):
    from tools import violet_searxng as sx
    asked = []

    def fetch(url, limit, report=None):
        asked.append(url)
        if url.startswith("https://github.com/"):
            raise OSError("unreachable")
        return b"archive"
    data, label = sx._download_source(fetch, None, [vs.GITHUB, vs.Source("custom", "https://evil.example/", "x"),
                                                    vs.BUILT_IN[1]])
    assert data == b"archive" and label == "加速站 8.217.135.233"
    assert asked == [sx.SOURCE_URL, "http://8.217.135.233:9999/" + sx.SOURCE_URL]
    with pytest.raises(sx.SearxngError, match="GitHub"):
        sx._download_source(lambda url, limit, report=None: (_ for _ in ()).throw(OSError("down")), None, [vs.GITHUB])


def test_voice_sources_from_a_mirror_must_match_the_recorded_hashes(tmp_path):
    from tools import violet_voice as vv
    assert set(vv.SOURCE_SHA256) == set(vv.SOURCES)
    good = {"src": b"cosyvoice", "matcha": b"matcha"}
    import hashlib

    def fetch(url, target):
        name = "src" if "CosyVoice" in url else "matcha"
        target.write_bytes(good[name])
    original = dict(vv.SOURCE_SHA256)
    try:
        vv.SOURCE_SHA256.update({k: hashlib.sha256(v).hexdigest() for k, v in good.items()})
        assert vv._download_sources(tmp_path, fetch, None, [vs.BUILT_IN[0]]) == "加速站 p.jackyu.cn"
        vv.SOURCE_SHA256["matcha"] = "0" * 64
        with pytest.raises(vv.VoiceError, match="SHA-256"):
            vv._download_sources(tmp_path, fetch, None, [vs.BUILT_IN[0]])
        # github.com itself is trusted as before (an archive GitHub regenerated still installs)
        assert vv._download_sources(tmp_path, fetch, None, [vs.GITHUB]) == "GitHub"
    finally:
        vv.SOURCE_SHA256.clear()
        vv.SOURCE_SHA256.update(original)


# ---------------------------------------------------------------------------------------------- install.sh

def test_install_sh_takes_the_signed_archive_hash_and_a_mirror_address():
    from pathlib import Path
    sh = (Path(inst.__file__).resolve().parent.parent / "install.sh").read_text(encoding="utf-8")
    assert 'if [ -n "${VIOLET_EXPECT_SHA256:-}" ]; then' in sh and 'want="$VIOLET_EXPECT_SHA256"' in sh
    assert "--release-base=*)" in sh
    # Parallel mirror race + local management shortcut (no download on uninstall when already installed)
    assert 'mkdir "$tmp/winner.lock"' in sh and 'cp -R "$target_root/app/tools"' in sh
    assert 'script_bases+=("https://p.jackyu.cn/$release_base" "http://8.217.135.233:9999/$release_base")' in sh


def test_order_fastest_prefers_fast_direct_github_abroad_and_fastest_mirror_in_china():
    auto = vs.sources("auto")
    # Overseas: GitHub answers quickly (<= 0.8s) -> stays first
    abroad = vs.order_fastest(auto, "https://github.com/a/b", probe=lambda u, t: 0.25 if u.startswith("https://github.com/") else 0.18)
    assert abroad[0].key == "github" and set(s.key for s in abroad) == {"github", "jackyu", "ip8217"}

    # Mainland China: GitHub unreachable, 8.217 faster than jackyu -> 8.217 first, GitHub last as fallback
    def china_probe(url, timeout):
        if url.startswith("http://8.217"):
            return 0.2
        if url.startswith("https://p.jackyu.cn"):
            return 0.6
        return None
    china = vs.order_fastest(auto, "https://github.com/a/b", probe=china_probe)
    assert [s.key for s in china] == ["ip8217", "jackyu", "github"]

    # Nothing answers -> preserve original order
    assert [s.key for s in vs.order_fastest(auto, "https://github.com/a/b", probe=lambda u, t: None)] == \
        ["github", "jackyu", "ip8217"]
    # Single source -> no probe needed
    called = []
    assert [s.key for s in vs.order_fastest([vs.GITHUB], "https://github.com/a/b", probe=lambda u, t: called.append(u))] == ["github"]
    assert called == []


def test_fetch_stream_aborts_crawling_source_when_fallbacks_remain_and_keeps_last_source(tmp_path):
    class FakeResp:
        def __init__(self, chunks):
            self._chunks = list(chunks)
            self.headers = {"Content-Length": str(sum(len(c) for c in chunks))}

        def read(self, n=-1):
            return self._chunks.pop(0) if self._chunks else b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    # 10 KB/s after 12s -> aborts with SlowDownloadError when allow_slow=False
    ticks = iter([0.0, 12.0, 24.0])
    with pytest.raises(vs.SlowDownloadError):
        vs.fetch_stream(
            "https://example.invalid/bin", 1024 * 1024, allow_slow=False,
            min_bps=32 * 1024, grace=10.0, clock=lambda: next(ticks),
            opener=lambda req, timeout=None: FakeResp([b"x" * 1024, b"y" * 1024]),
        )
    assert vs.reason(vs.SlowDownloadError("too slow")) == "下载太慢"

    # Same slow stream succeeds when allow_slow=True (last candidate)
    ticks2 = iter([0.0, 12.0, 24.0])
    out = vs.fetch_stream(
        "https://example.invalid/bin", 1024 * 1024, allow_slow=True,
        min_bps=32 * 1024, grace=10.0, clock=lambda: next(ticks2),
        opener=lambda req, timeout=None: FakeResp([b"hello", b"world"]),
    )
    assert out == b"helloworld"

    # OverflowError deletes partial target file
    target = tmp_path / "partial.tar.gz"
    with pytest.raises(OverflowError):
        vs.fetch_stream(
            "https://example.invalid/big", 5, target=target,
            opener=lambda req, timeout=None: FakeResp([b"1234", b"5678"]),
        )
    assert not target.exists()


def test_pip_sources_respects_env_and_picks_fastest_mirror(monkeypatch):
    monkeypatch.setenv("PIP_INDEX_URL", "https://custom.pypi/simple")
    assert vs.pip_sources(probe=lambda u, t: (_ for _ in ()).throw(AssertionError("should not probe"))) == [[]]
    monkeypatch.delenv("PIP_INDEX_URL", raising=False)
    monkeypatch.setattr(vs, "_has_pip_config", lambda: False)

    # China: official PyPI fails, Tsinghua fastest -> Tsinghua, Aliyun, PyPI
    def china_pip(url, timeout):
        if "tsinghua" in url:
            return 0.1
        if "aliyun" in url:
            return 0.3
        return None
    assert vs.pip_sources(probe=china_pip) == [["-i", vs.PYPI_TSINGHUA], ["-i", vs.PYPI_ALIYUN], []]

    # Abroad: official PyPI answers in 0.2s -> official PyPI first
    assert vs.pip_sources(probe=lambda u, t: 0.2 if "pypi.org" in u else 0.9)[0] == []


def test_mcp_relay_export_is_silent_before_install_json_exists(tmp_path, monkeypatch):
    from tools import violet_relay as vr
    monkeypatch.setattr(vr, "UNIT_PATH", tmp_path / "nonexistent.service")
    res = inst.mcp_relay(inst.Options(root=tmp_path / "fresh"), "export")
    assert res == {"ok": True, "installed": False}


def test_snowluma_measure_parallel_handles_failures():
    from tools import violet_snowluma as sl

    def fake(item):
        if item == "bad":
            raise RuntimeError("boom")
        return 1234.5 if item == "fast" else 0.0

    got = sl._measure_parallel(["bad", "fast", "dead"], fake)
    assert got == {"bad": 0.0, "fast": 1234.5, "dead": 0.0}
