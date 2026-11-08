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

import re
from dataclasses import dataclass
from typing import Callable, TypeVar

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
