"""The Control Center's language (2026-11-09; the README's ten languages).

The pages are written in Simplified Chinese. The language is the `violet_lang` cookie (set by the picker in the
menu), else the browser's Accept-Language, else Chinese. For another language the page loads /control/i18n.js
(window.VIOLET_I18N = {lang, dict}) and static/control_i18n.js translates what is shown; the login page (no
scripts) is translated here. Dictionaries: wx_ilink_py/i18n/<lang>.json, keyed by the Chinese text
(tools/i18n_extract.py lists the keys; zh-CN has no file).
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

SUPPORTED = ("zh-CN", "en", "zh-TW", "ja", "ko", "es", "fr", "de", "ru", "pt-BR")
# each language in its own words (the login page's picker; the menu's picker in control_i18n.js has the same list)
NAMES = (("zh-CN", "简体中文"), ("en", "English"), ("zh-TW", "繁體中文"), ("ja", "日本語"), ("ko", "한국어"),
         ("es", "Español"), ("fr", "Français"), ("de", "Deutsch"), ("ru", "Русский"), ("pt-BR", "Português (Brasil)"))
YEAR = 365 * 24 * 3600
SOURCE = "zh-CN"
COOKIE = "violet_lang"
DIR = Path(__file__).resolve().parent / "i18n"
_cache: dict[str, tuple[float, dict[str, str]]] = {}
_lock = threading.Lock()


def normalize(code: str | None) -> str | None:
    """A browser / cookie language tag -> one of SUPPORTED, or None."""
    if not code:
        return None
    c = code.strip().replace("_", "-").lower()
    if c in ("zh-tw", "zh-hk", "zh-mo", "zh-hant") or c.startswith("zh-hant"):
        return "zh-TW"
    if c.startswith("zh"):
        return "zh-CN"
    if c.startswith("pt"):
        return "pt-BR"
    base = c.split("-")[0]
    return base if base in SUPPORTED else None


def pick(request: Any) -> str:
    chosen = normalize(request.cookies.get(COOKIE))
    if chosen:
        return chosen
    for code, _quality in request.accept_languages:
        found = normalize(code)
        if found:
            return found
    return SOURCE


def dictionary(lang: str) -> dict[str, str]:
    if lang == SOURCE or lang not in SUPPORTED:
        return {}
    path = DIR / f"{lang}.json"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    with _lock:
        hit = _cache.get(lang)
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data = {k: v for k, v in data.items() if isinstance(v, str) and v}
        _cache[lang] = (mtime, data)
        return data


def translator(lang: str):
    """t(text) for server-rendered pages: the exact text, else a key with {0}-style values ("请 {0} 分钟后再试")."""
    d = dictionary(lang)

    def t(text: str) -> str:
        if not text or text in d:
            return d.get(text, text)
        for key, value in d.items():
            if "{" not in key:
                continue
            parts = re.split(r"(\{\d+\})", key)
            rx = "".join("(.*?)" if i % 2 else re.escape(p) for i, p in enumerate(parts))
            m = re.fullmatch(rx, text)
            if m:
                names = [int(p[1:-1]) for i, p in enumerate(parts) if i % 2]
                values = dict(zip(names, m.groups()))
                return re.sub(r"\{(\d+)\}", lambda x: values.get(int(x.group(1)), ""), value)
        return text
    return t


def script(lang: str) -> str:
    return "window.VIOLET_I18N = " + json.dumps({"lang": lang, "dict": dictionary(lang)}, ensure_ascii=False) + ";\n"
