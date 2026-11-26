"""Where Violet downloads things from GitHub (Violet 0.7.59; user 2026-11-08: updates, the phone relay, web search and
voice must also work on servers in mainland China, where github.com is often unreachable).

A GitHub mirror is a prefix put in front of the github.com address ("https://p.jackyu.cn/" +
"https://github.com/..."). The setting "下载来源" (VIOLET_DOWNLOAD_SOURCE, page 设置 -> 版本更新) picks:

  auto     GitHub first, then each built-in mirror in turn (default)
  github   only GitHub
  jackyu / ip8217   only that built-in mirror
  custom   only the address in VIOLET_DOWNLOAD_MIRROR

What protects a download that came through a mirror (the mirror is a stranger's server):
  * Violet updates: release.json signed by the release key (tools/violet_sign.py) and the SHA-256 in it
  * the phone relay: Violet MCP's SHA256SUMS.txt signed by its relay key
  * voice: the source archives' SHA-256 written in tools/violet_voice.py
  * web search: nothing can be checked (SearXNG's current main line) -- so only GitHub and the BUILT-IN mirrors are
    used for it, never the custom address: the setting is written by Violet's unprivileged service account, and the
    install runs as root.
Built-in mirrors tested 2026-11-08 from China Mobile (release files, archives and release pages; not the GitHub API).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, TypeVar
import urllib.request

SETTING = "VIOLET_DOWNLOAD_SOURCE"
CUSTOM_SETTING = "VIOLET_DOWNLOAD_MIRROR"
MIRRORS = (("jackyu", "https://p.jackyu.cn/", "加速站 p.jackyu.cn"),
           ("ip8217", "http://8.217.135.233:9999/", "加速站 8.217.135.233"))
CHOICES = (("auto", "自动：先连 GitHub，连不上依次试加速站"),
           ("github", "只用 GitHub"),
           ("jackyu", "只用加速站 p.jackyu.cn（HTTPS）"),
           ("ip8217", "只用加速站 8.217.135.233:9999（HTTP，通常更快）"),
           ("custom", "只用自定义加速地址（填在下面）"))
KEYS = tuple(key for key, _label in CHOICES)
_CUSTOM = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?(/[A-Za-z0-9._~%!$&'()*+,;=:@-]*)*$")
T = TypeVar("T")

PYPI_TSINGHUA = "https://pypi.tuna.tsinghua.edu.cn/simple"
PYPI_ALIYUN = "https://mirrors.aliyun.com/pypi/simple/"
PYPI_CANDIDATES = (
    ("", "https://pypi.org/simple/pip/"),
    (PYPI_TSINGHUA, PYPI_TSINGHUA + "/pip/"),
    (PYPI_ALIYUN, PYPI_ALIYUN.rstrip("/") + "/pip/"),
)
SLOW_MIN_BPS = 32 * 1024
SLOW_GRACE_SECONDS = 10.0
SLOW_WINDOW_SECONDS = 15.0


@dataclass(frozen=True)
class Source:
    key: str
    prefix: str          # "" = github.com itself
    label: str

    @property
    def direct(self) -> bool:
        return not self.prefix

    def url(self, github_url: str) -> str:
        return self.prefix + github_url


GITHUB = Source("github", "", "GitHub")
BUILT_IN = tuple(Source(key, prefix, label) for key, prefix, label in MIRRORS)


class SourcesFailed(RuntimeError):
    """Every source failed; the message lists each one's reason."""


class SlowDownloadError(TimeoutError):
    """A download stayed below the speed floor while another source is still available."""


def custom_checked(text: str | None) -> str | None:
    """A mirror prefix the user typed, normalised to end with "/" -- or None when it is not an http(s) address
    without user name, query or fragment."""
    value = (text or "").strip()
    if not value or not _CUSTOM.match(value) or len(value) > 300:
        return None
    if value.lower().startswith(("https://github.com", "http://github.com")):
        return None
    return value if value.endswith("/") else value + "/"


def choice_checked(value: str | None) -> str:
    value = (value or "").strip().lower()
    return value if value in KEYS else "auto"


def sources(choice: str | None, custom: str | None = None, *, allow_custom: bool = True) -> list[Source]:
    """The sources to try, in order. "custom" without a usable address, or for a download that may not use it,
    falls back to the automatic order."""
    choice = choice_checked(choice)
    if choice == "github":
        return [GITHUB]
    for mirror in BUILT_IN:
        if choice == mirror.key:
            return [mirror]
    if choice == "custom" and allow_custom:
        prefix = custom_checked(custom)
        if prefix:
            host = re.sub(r"^https?://", "", prefix).split("/", 1)[0]
            return [Source("custom", prefix, f"自定义加速地址 {host}")]
    return [GITHUB, *BUILT_IN]


def from_values(values: dict, *, allow_custom: bool = True) -> list[Source]:
    """sources() from a settings mapping (the Control Center's settings.json values, or the environment)."""
    return sources(str(values.get(SETTING) or ""), str(values.get(CUSTOM_SETTING) or ""), allow_custom=allow_custom)


def first(candidates: list[Source], attempt: Callable[[Source], T],
          on_try: Callable[[Source], None] | None = None) -> tuple[T, Source]:
    """attempt(source) for each source until one returns; any exception means "try the next one". SourcesFailed
    lists every reason (the last line of each)."""
    reasons = []
    for source in candidates:
        if on_try is not None:
            on_try(source)
        try:
            return attempt(source), source
        except Exception as exc:                                   # noqa: BLE001 -- reported, next source
            reasons.append(f"{source.label}：{reason(exc)}")
    raise SourcesFailed("；".join(reasons) or "没有可用的下载来源")


def reason(exc: BaseException) -> str:
    if isinstance(exc, SlowDownloadError):
        return "下载太慢"
    text = str(exc).strip().splitlines()
    detail = text[-1] if text else ""
    name = exc.__class__.__name__
    if name in ("timeout", "TimeoutError") or "timed out" in detail:
        return "连接超时"
    if name in ("URLError", "ConnectionRefusedError", "ConnectionResetError", "gaierror", "RemoteDisconnected"):
        return f"连不上（{detail[:120] or name}）"
    if name == "HTTPError":
        return f"返回错误 {getattr(exc, 'code', '')}".strip()
    return detail[:200] or name


def _probe_url(url: str, timeout: float = 5.0, sample_bytes: int = 65536) -> float | None:
    """Measure how quickly ``url`` delivers its first bytes (seconds, or None when unreachable / HTTP error)."""
    req = urllib.request.Request(url, headers={"User-Agent": "violet-installer",
                                               "Range": f"bytes=0-{sample_bytes - 1}"})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- fixed https/mirror URLs
            status = getattr(resp, "status", 200) or 200
            if status >= 400:
                return None
            chunk = resp.read(sample_bytes)
            if not chunk:
                return None
        return max(0.001, time.monotonic() - start)
    except Exception:                                               # noqa: BLE001 -- probe failure = unavailable
        return None


def _race_probes(items: list[tuple[Any, str]], probe: Callable[[str, float], float | None],
                 *, timeout: float, settle: float = 0.8,
                 fast_ok: Callable[[Any, float], bool] | None = None) -> dict[int, float | None]:
    """Probe ``items`` in parallel on daemon threads; return shortly after the first hit instead of waiting for a
    hung host to time out."""
    results: dict[int, float | None] = {}
    done = [0]
    cond = threading.Condition()

    def worker(idx: int, url: str) -> None:
        val: float | None = None
        try:
            val = probe(url, timeout)
        except Exception:                                           # noqa: BLE001
            val = None
        with cond:
            results[idx] = val
            done[0] += 1
            cond.notify_all()

    for idx, (_item, url) in enumerate(items):
        threading.Thread(target=worker, args=(idx, url), daemon=True).start()

    deadline = time.monotonic() + timeout
    first_hit: float | None = None
    with cond:
        while done[0] < len(items):
            if fast_ok is not None and any(
                val is not None and fast_ok(items[i][0], val) for i, val in results.items()
            ):
                break
            if first_hit is None and any(val is not None for val in results.values()):
                first_hit = time.monotonic()
            target = min(deadline, first_hit + settle) if first_hit is not None else deadline
            remaining = target - time.monotonic()
            if remaining <= 0:
                break
            cond.wait(remaining)
        return dict(results)


def order_fastest(candidates: list[Source], github_url: str, *,
                  probe: Callable[[str, float], float | None] | None = None,
                  timeout: float = 5.0, direct_bonus: float = 0.8) -> list[Source]:
    """Reorder ``candidates`` (already authorised by the caller) so the fastest reachable source goes first.

    A direct GitHub source that answers within ``direct_bonus`` seconds keeps first place (overseas servers stay
    direct); on mainland China servers where GitHub stalls or crawls, the fastest built-in mirror wins in < 1.5 s
    while keeping all remaining candidates behind it as fallbacks.
    """
    items = list(candidates)
    if len(items) <= 1:
        return items
    timings = _race_probes(
        [(s, s.url(github_url)) for s in items],
        probe or _probe_url,
        timeout=timeout,
        fast_ok=lambda s, dt: bool(s.direct and dt <= direct_bonus),
    )
    if not any(v is not None for v in timings.values()):
        return items

    def sort_key(pair: tuple[int, Source]) -> tuple[int, float, int]:
        idx, src = pair
        dt = timings.get(idx)
        if dt is None:
            return (2, float("inf"), idx)
        if src.direct and dt <= direct_bonus:
            return (0, dt, idx)
        return (1, dt, idx)

    return [src for _idx, src in sorted(enumerate(items), key=sort_key)]


def fetch_stream(url: str, limit: int, *, timeout: float = 20.0,
                 report: Callable[[int, int], None] | None = None,
                 allow_slow: bool = True, min_bps: float = SLOW_MIN_BPS,
                 grace: float = SLOW_GRACE_SECONDS, window: float = SLOW_WINDOW_SECONDS,
                 target: Path | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 opener: Callable[..., Any] | None = None) -> bytes:
    """Download ``url`` in chunks (to memory or ``target``). When ``allow_slow`` is False (more fallback sources
    remain), abort with ``SlowDownloadError`` if throughput stays under ``min_bps``."""
    req = urllib.request.Request(url, headers={"User-Agent": "violet-installer"})
    chunks: list[bytes] = []
    got = 0
    fh = open(target, "wb") if target is not None else None
    try:
        with (opener or urllib.request.urlopen)(req, timeout=timeout) as resp:  # noqa: S310 -- fixed URLs
            try:
                total = int(resp.headers.get("Content-Length") or 0)
            except ValueError:
                total = 0
            start = clock()
            win_start, win_bytes = start, 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                got += len(chunk)
                if got > limit:
                    raise OverflowError(f"download exceeds {limit} bytes: {url}")
                if fh is not None:
                    fh.write(chunk)
                else:
                    chunks.append(chunk)
                if report is not None:
                    report(got, total)
                if not allow_slow:
                    now = clock()
                    elapsed = now - start
                    win_bytes += len(chunk)
                    if elapsed >= grace and got / max(0.001, elapsed) < min_bps:
                        raise SlowDownloadError(f"average {got / elapsed:.0f} B/s")
                    if now - win_start >= window:
                        if win_bytes / max(0.001, now - win_start) < min_bps:
                            raise SlowDownloadError(f"window {win_bytes / (now - win_start):.0f} B/s")
                        win_start, win_bytes = now, 0
        return b"" if fh is not None else b"".join(chunks)
    except Exception:
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass
            fh = None
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        if fh is not None:
            fh.close()


def _has_pip_config() -> bool:
    if os.environ.get("PIP_INDEX_URL"):
        return True
    paths = [Path("/etc/pip.conf"), Path("/root/.pip/pip.conf"), Path("/root/.config/pip/pip.conf")]
    try:
        home = Path.home()
        paths += [home / ".pip" / "pip.conf", home / ".config" / "pip" / "pip.conf"]
    except RuntimeError:
        pass
    return any(p.is_file() for p in paths)


def pip_sources(*, probe: Callable[[str, float], float | None] | None = None,
                timeout: float = 3.0, direct_bonus: float = 0.5) -> list[list[str]]:
    """pip index arguments to try in order (fastest first; respects PIP_INDEX_URL / pip.conf)."""
    if _has_pip_config():
        return [[]]
    timings = _race_probes(
        [(idx_url, probe_url) for idx_url, probe_url in PYPI_CANDIDATES],
        probe or _probe_url,
        timeout=timeout,
        settle=0.6,
        fast_ok=lambda idx_url, dt: bool(idx_url == "" and dt <= direct_bonus),
    )

    def sort_key(pair: tuple[int, tuple[str, str]]) -> tuple[int, float, int]:
        i, (idx_url, _probe_url) = pair
        dt = timings.get(i)
        if dt is None:
            return (2, float("inf"), i)
        if idx_url == "" and dt <= direct_bonus:
            return (0, dt, i)
        return (1, dt, i)

    if any(v is not None for v in timings.values()):
        ordered = [idx_url for _i, (idx_url, _p) in sorted(enumerate(PYPI_CANDIDATES), key=sort_key)]
    else:
        ordered = [PYPI_TSINGHUA, PYPI_ALIYUN, ""]
    return [(["-i", url] if url else []) for url in ordered]
