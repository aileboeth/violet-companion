"""Releases and updates (offline): version rules, the GitHub "latest release" reader, `violet update`, the Control
Center check and the release workflow. Nothing here contacts GitHub."""
from __future__ import annotations

import io
import json
import re
import urllib.error
from pathlib import Path

import pytest

from tools import violet_installer as inst

PROJECT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT.parent
RELEASE = {"tag": "v0.2.0", "version": "0.2.0", "url": "https://github.com/o/r/releases/tag/v0.2.0",
           "published_at": "2026-11-01T00:00:00Z", "notes": "新版本",
           "install_url": "https://github.com/o/r/releases/download/v0.2.0/install.sh"}


# ---------------------------------------------------------------------------------------------- versions

@pytest.mark.parametrize("text,key", [("0.1.0", (0, 1, 0)), ("v1.2", (1, 2, 0)), ("2", (2, 0, 0)),
                                      ("0.1.0-rc1", (0, 1, 0)), ("unknown", None), ("2.0.0-p10x.1", (2, 0, 0)),
                                      ("", None), ("1.2.3.4", None)])
def test_version_key(text, key):
    assert inst.version_key(text) == key


def test_is_newer_needs_two_real_versions():
    assert inst.is_newer("0.1.1", "0.1.0") and inst.is_newer("v1.0.0", "0.9.9")
    assert not inst.is_newer("0.1.0", "0.1.0") and not inst.is_newer("0.0.9", "0.1.0")
    assert not inst.is_newer("0.2.0", "unknown") and not inst.is_newer(None, "0.1.0")


def test_the_program_version_is_a_release_version():
    assert inst.version_key((PROJECT / "VERSION").read_text(encoding="utf-8")) is not None


# ---------------------------------------------------------------------------------------------- latest release

class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


SEED = bytes(range(32))                                    # a test key; the real one is only on GitHub
OFFICIAL = inst.UPDATE_REPO
MIRROR = "https://mirror.example/"


@pytest.fixture()
def keys(monkeypatch):
    """The release key replaced by the test key (the module object the installer itself uses)."""
    import base64
    sign = inst._tool("violet_sign")
    monkeypatch.setattr(sign, "VIOLET_RELEASE_KEY", base64.b64encode(sign.public_key(SEED)).decode())
    return sign


def _meta(version="0.3.1", repo=OFFICIAL, **extra):
    meta = {"version": version, "base": f"https://github.com/{repo}/releases/download/v{version}",
            "built_at": "2026-11-08T10:00:00", "notes": "Violet 0.3.1: 说明",
            "sha256": {"install.sh": "a" * 64, "violet.tar.gz": "b" * 64}, **extra}
    return json.dumps(meta).encode()


def _opener(files, seen=None):
    """urlopen stand-in: files maps a URL to bytes or an exception; anything else is a 404."""
    def opener(request, timeout):
        if seen is not None:
            seen.append(request.full_url)
        assert request.get_header("User-agent")
        value = files.get(request.full_url)
        if isinstance(value, BaseException):
            raise value
        if value is None:
            raise urllib.error.HTTPError(request.full_url, 404, "nf", {}, None)
        return _Response(value)
    return opener


def _latest(repo=OFFICIAL, prefix=""):
    return f"{prefix}https://github.com/{repo}/releases/latest/download/release.json"


def _src(*keys):
    from tools import violet_sources as vs
    out = []
    for key in keys:
        out.append(vs.GITHUB if key == "github" else vs.Source("custom", MIRROR, "自定义加速地址 mirror.example"))
    return out


def test_an_unsigned_release_from_github_itself_is_accepted(keys):
    seen = []
    rel = inst.latest_release(OFFICIAL, opener=_opener({_latest(): _meta()}, seen))
    assert seen == [_latest(), _latest() + ".sig"]                         # release.json, not the GitHub API
    assert rel["version"] == "0.3.1" and rel["signed"] is False and rel["sha256"] == {}
    assert rel["install_url"] == f"https://github.com/{OFFICIAL}/releases/download/v0.3.1/install.sh"
    assert rel["url"] == f"https://github.com/{OFFICIAL}/releases/tag/v0.3.1" and rel["notes"].startswith("Violet")


def test_a_signed_release_carries_the_checked_hashes(keys):
    body = _meta()
    rel = inst.latest_release(OFFICIAL, opener=_opener({_latest(): body, _latest() + ".sig":
                                                         keys.signature_text(SEED, body).encode()}))
    assert rel["signed"] is True and rel["sha256"] == {"install.sh": "a" * 64, "violet.tar.gz": "b" * 64}


def test_a_wrong_signature_is_refused_even_from_github(keys):
    body = _meta()
    with pytest.raises(inst.InstallError, match="签名不对"):
        inst.latest_release(OFFICIAL, opener=_opener({_latest(): body, _latest() + ".sig":
                                                      keys.signature_text(SEED, body + b" ").encode()}))


def test_a_mirror_needs_the_signature_and_then_serves_everything_itself(keys):
    body = _meta()
    files = {_latest(prefix=MIRROR): body}
    with pytest.raises(inst.InstallError, match="必须有签名"):
        inst.latest_release(OFFICIAL, opener=_opener(files), sources=_src("mirror"))
    files[_latest(prefix=MIRROR) + ".sig"] = keys.signature_text(SEED, body).encode()
    rel = inst.latest_release(OFFICIAL, opener=_opener(files), sources=_src("mirror"))
    assert rel["signed"] and rel["source"] == "自定义加速地址 mirror.example"
    base = f"https://github.com/{OFFICIAL}/releases/download/v0.3.1"
    assert rel["install_url"] == MIRROR + base + "/install.sh" and rel["release_base"] == MIRROR + base


def test_a_mirror_cannot_change_the_version_or_the_address(keys):
    for body in (_meta(base=f"https://github.com/{OFFICIAL}/releases/download/v9.9.9"),
                 _meta(base="https://evil.example/v0.3.1"), _meta(version="nightly")):
        files = {_latest(prefix=MIRROR): body, _latest(prefix=MIRROR) + ".sig": keys.signature_text(SEED, body).encode()}
        with pytest.raises(inst.InstallError, match="不符|不规范"):
            inst.latest_release(OFFICIAL, opener=_opener(files), sources=_src("mirror"))
    unsigned_hashes = json.dumps({**json.loads(_meta()), "sha256": {}}).encode()
    files = {_latest(prefix=MIRROR): unsigned_hashes,
             _latest(prefix=MIRROR) + ".sig": keys.signature_text(SEED, unsigned_hashes).encode()}
    with pytest.raises(inst.InstallError, match="SHA-256"):
        inst.latest_release(OFFICIAL, opener=_opener(files), sources=_src("mirror"))


def test_sources_are_tried_in_order_and_every_failure_is_explained(keys):
    import socket
    body = _meta()
    files = {_latest(): socket.timeout("timed out"), _latest(prefix=MIRROR): body,
             _latest(prefix=MIRROR) + ".sig": keys.signature_text(SEED, body).encode()}
    rel = inst.latest_release(OFFICIAL, opener=_opener(files), sources=_src("github", "mirror"))
    assert rel["source"].startswith("自定义加速地址")
    files[_latest(prefix=MIRROR)] = urllib.error.URLError("refused")
    with pytest.raises(inst.InstallError) as info:
        inst.latest_release(OFFICIAL, opener=_opener(files), sources=_src("github", "mirror"))
    assert "GitHub：连接超时" in str(info.value) and "自定义加速地址 mirror.example：连不上" in str(info.value)


def test_nothing_published_yet_is_said_plainly(keys):
    with pytest.raises(inst.InstallError, match="还没有发布"):
        inst.latest_release(OFFICIAL, opener=_opener({}))


def test_a_fork_is_only_read_from_github_itself(keys):
    seen = []
    body = _meta(repo="someone/fork")
    rel = inst.latest_release("someone/fork", opener=_opener({_latest("someone/fork"): body}, seen),
                              sources=_src("mirror"))
    assert rel["signed"] is False and all(url.startswith("https://github.com/") for url in seen)


def test_the_repository_can_be_overridden(monkeypatch):
    monkeypatch.setenv("VIOLET_UPDATE_REPO", "someone/fork")
    assert inst.update_repo() == "someone/fork"
    monkeypatch.delenv("VIOLET_UPDATE_REPO")
    assert inst.update_repo() == inst.UPDATE_REPO == "aileboeth/violet-companion"


# ---------------------------------------------------------------------------------------------- violet update

def _installed(tmp_path, version="0.1.0", service="supervisor"):
    root = tmp_path / "violet"
    (root / "app").mkdir(parents=True)
    (root / "app" / "VERSION").write_text(version + "\n", encoding="utf-8")
    (root / "install.json").write_text(json.dumps({"version": version, "service": service, "port": 7100}),
                                       encoding="utf-8")
    return inst.Options(root=root, log=lambda s: None)


class _Run:
    def __init__(self, stdout, code=0):
        self.stdout, self.code, self.calls = stdout, code, []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        import subprocess
        return subprocess.CompletedProcess(cmd, self.code, self.stdout, "")


def test_update_downloads_the_releases_installer_and_runs_its_upgrade(tmp_path):
    opts = _installed(tmp_path)
    got = []
    run = _Run(json.dumps({"ok": True, "from": "0.1.0", "to": "0.2.0", "backup": "/b.tar.gz"}) + "\n")

    def download(url, target):
        got.append(url)
        target.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    result = inst.update(opts, release=RELEASE, download=download, run=run, windows=False)
    assert got == [RELEASE["install_url"]]
    assert run.calls[0][0] == "bash" and run.calls[0][2:] == ["upgrade", "--root", str(opts.root), "--json"]
    assert result["updated"] is True and result["to"] == "0.2.0" and result["backup"] == "/b.tar.gz"


def test_a_signed_update_runs_only_the_signed_script_and_passes_the_archive_hash(tmp_path):
    import hashlib
    script = b"#!/usr/bin/env bash\necho signed\n"
    rel = {**RELEASE, "signed": True, "source": "加速站 p.jackyu.cn",
           "install_url": "https://p.jackyu.cn/https://github.com/o/r/releases/download/v0.2.0/install.sh",
           "release_base": "https://p.jackyu.cn/https://github.com/o/r/releases/download/v0.2.0",
           "sha256": {"install.sh": hashlib.sha256(script).hexdigest(), "violet.tar.gz": "c" * 64}}
    seen = {}

    def run(cmd, **kw):
        import subprocess
        seen["env"] = kw.get("env") or {}
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"ok": True, "to": "0.2.0"}) + "\n", "")
    result = inst.update(_installed(tmp_path), release=rel, download=lambda u, t: t.write_bytes(script), run=run,
                         windows=False)
    assert result["updated"] and result["source"] == "加速站 p.jackyu.cn"
    assert seen["env"]["VIOLET_EXPECT_SHA256"] == "c" * 64 and seen["env"]["VIOLET_RELEASE_BASE"] == rel["release_base"]
    never = lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run"))  # noqa: E731
    with pytest.raises(inst.InstallError, match="不一致"):
        inst.update(_installed(tmp_path / "b"), release=rel, download=lambda u, t: t.write_bytes(script + b"x"),
                    run=never, windows=False)


def test_update_does_nothing_when_already_current_or_only_checking(tmp_path):
    opts = _installed(tmp_path, version="0.2.0")
    never = lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not download"))  # noqa: E731
    assert inst.update(opts, release=RELEASE, download=never, run=never, windows=False)["updated"] is False
    opts = _installed(tmp_path / "x", version="0.1.0")
    checked = inst.update(opts, check_only=True, release=RELEASE, download=never, run=never, windows=False)
    assert checked["newer"] is True and checked["updated"] is False


def test_a_failed_upgrade_is_reported_with_its_reason(tmp_path):
    opts = _installed(tmp_path)
    run = _Run(json.dumps({"ok": False, "error": "升级失败，已恢复到 0.1.0，旧版本已重新运行。"}) + "\n", code=2)
    with pytest.raises(inst.InstallError, match="已恢复到 0.1.0"):
        inst.update(opts, release=RELEASE, download=lambda u, t: t.write_text("#!", encoding="utf-8"), run=run,
                    windows=False)


def test_a_systemd_install_needs_root_and_windows_gets_instructions(tmp_path):
    never = lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run"))  # noqa: E731
    opts = _installed(tmp_path, service="systemd")
    if not inst._is_root():
        with pytest.raises(inst.InstallError, match="需要 root"):
            inst.update(opts, release=RELEASE, download=never, run=never, windows=False)
    with pytest.raises(inst.InstallError, match="Windows"):
        inst.update(_installed(tmp_path / "w"), release=RELEASE, download=never, run=never, windows=True)


def test_no_installation_no_update(tmp_path):
    with pytest.raises(inst.InstallError, match="没有安装"):
        inst.update(inst.Options(root=tmp_path / "none", log=lambda s: None), release=RELEASE)


# ---------------------------------------------------------------------------------------------- Control Center

@pytest.fixture()
def updates(monkeypatch):
    from ai import updates
    updates.reset_for_tests()
    monkeypatch.setenv("VIOLET_UPDATE_CHECK", "1")
    yield updates
    updates.reset_for_tests()


def test_check_reports_a_newer_release_and_caches_it(updates, tmp_path):
    root = tmp_path / "violet" / "app"
    root.mkdir(parents=True)
    (root / "VERSION").write_text("0.1.0\n", encoding="utf-8")
    (root.parent / "install.json").write_text("{}", encoding="utf-8")
    calls = []
    fetch = lambda: calls.append(1) or RELEASE  # noqa: E731
    t = [1000.0]
    clock = lambda: t[0]  # noqa: E731
    first = updates.check(fetch=fetch, now=clock, root=root)
    assert first["newer"] and first["latest"] == "0.2.0" and first["current"] == "0.1.0"
    assert first["command"] == f"{root.parent / 'violet'} update" and first["url"].startswith("https://github.com/")
    t[0] += 3600
    assert updates.check(fetch=fetch, now=clock, root=root)["newer"]   # an hour later: from the cache
    assert len(calls) == 1
    t[0] += 6 * 3600
    updates.check(fetch=fetch, now=clock, root=root)                   # seven hours after the first: asked again
    assert len(calls) == 2


def test_check_can_be_switched_off_and_explains_failures(updates, monkeypatch, tmp_path):
    def broken():
        raise inst.InstallError("查询最新版本失败（网络不通或被拦截）：URLError")
    failed = updates.check(fetch=broken, refresh=True)
    assert failed["error"].startswith("查询最新版本失败") and failed["newer"] is False
    # 0.2.0's switch maps to "manual" (stage 3): nothing is fetched until the admin presses "check for updates"
    monkeypatch.setenv("VIOLET_UPDATE_CHECK", "0")
    updates.reset_for_tests()
    off = updates.check(fetch=lambda: (_ for _ in ()).throw(AssertionError("no request in manual mode")))
    assert off["mode"] == "manual" and off["enabled"] is False and off["latest"] is None


def test_a_link_that_is_not_github_is_dropped(updates):
    result = updates.check(fetch=lambda: {**RELEASE, "url": "https://evil.example/x"}, refresh=True)
    assert result["url"] is None


def test_without_an_installer_layout_the_command_says_how_to_upgrade(updates, tmp_path):
    assert "install.sh upgrade" in updates.update_command(tmp_path / "src")


# ---------------------------------------------------------------------------------------------- release workflow

def test_the_release_workflow_publishes_what_make_release_builds():
    flow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert 'tags: ["v*"]' in flow and "contents: write" in flow
    assert 'want="v$(tr -d' in flow and "violet/VERSION" in flow                      # tag must match the version
    assert "releases/download/${GITHUB_REF_NAME}" in flow and "make_release.py" in flow
    source = (PROJECT / "tools" / "make_release.py").read_text(encoding="utf-8")
    built = re.findall(r'"([^"]+)"', re.search(r"names = \[([^\]]*)\]", source).group(1))
    assert "install.sh" in built and "violet.tar.gz" in built
    for name in built + ["SHA256SUMS", "release.json.sig"]:
        assert f"dist/{name}" in flow, name
    # 0.7.59: always signed; the secret is required (make_release refuses an empty one)
    assert "--sign-key-env VIOLET_RELEASE_SIGNING_KEY" in flow and "secrets.VIOLET_RELEASE_SIGNING_KEY" in flow


def test_make_release_signs_release_json_with_the_hashes(tmp_path, keys, monkeypatch):
    import hashlib
    from tools import make_release
    meta =make_release.build("https://github.com/o/r/releases/download/v1", tmp_path, worktree=True,
                              sign_key=SEED.hex())
    body = (tmp_path / "release.json").read_bytes()
    assert keys.check(keys.VIOLET_RELEASE_KEY, body, (tmp_path / "release.json.sig").read_bytes())
    assert meta["sha256"]["violet.tar.gz"] == hashlib.sha256((tmp_path / "violet.tar.gz").read_bytes()).hexdigest()
    assert meta["sha256"]["install.sh"] == hashlib.sha256((tmp_path / "install.sh").read_bytes()).hexdigest()
    assert "release.json.sig" in (tmp_path / "SHA256SUMS").read_text(encoding="utf-8")
    monkeypatch.setenv("EMPTY_KEY", "")
    with pytest.raises(SystemExit, match="secret"):
        make_release.main(["--base", "https://x.example/v1", "--out", str(tmp_path / "o"), "--worktree",
                           "--sign-key-env", "EMPTY_KEY"])


def test_the_readme_and_install_guide_give_the_github_one_liner():
    line = "curl -fsSL https://github.com/aileboeth/violet-companion/releases/latest/download/install.sh | bash"
    assert line in (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert line in (PROJECT / "INSTALL.md").read_text(encoding="utf-8")
