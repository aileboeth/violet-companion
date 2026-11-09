"""Collect every Chinese text the Control Center can show (2026-11-09: the web pages in the README's languages).

The pages keep their Chinese source text; wx_ilink_py/static/control_i18n.js translates what is on the page from a
dictionary per language (wx_ilink_py/i18n/<lang>.json), keyed by the Chinese text itself. This tool lists those keys:

  * control.html: text and the placeholder / title / aria-label / alt attributes;
  * static/*.js: string literals; template literals become patterns ("已连接，手机提供了 {0} 个工具。");
    "a" + "b" literal concatenations are joined;
  * the Python modules whose words reach the page (setting labels and help, errors, diagnoses, task notes): string
    literals and f-strings (patterns), not docstrings and not log messages.

  python -m tools.i18n_extract            -> writes wx_ilink_py/i18n/source.json, prints counts
  python -m tools.i18n_extract --check    -> exit 1 when a language misses keys (tests run the same check)
"""

from __future__ import annotations

import ast
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
WEB = APP / "wx_ilink_py"
OUT = WEB / "i18n"
CJK = re.compile(r"[㐀-鿿！-～、-】]")
HAN = re.compile(r"[㐀-鿿]")
LANGS = ("en", "zh-TW", "ja", "ko", "es", "fr", "de", "ru", "pt-BR")

# Python modules whose strings are shown on the page (labels, help, errors, diagnoses, progress notes)
PY_FILES = [
    "wx_ilink_py/auth.py", "wx_ilink_py/channel_routes.py", "wx_ilink_py/control_routes.py", "wx_ilink_py/image_routes.py",
    "wx_ilink_py/mcp_routes.py", "wx_ilink_py/ops_page_routes.py", "wx_ilink_py/ops_routes.py",
    "wx_ilink_py/persona_routes.py", "wx_ilink_py/phone_mcp_routes.py", "wx_ilink_py/profile_routes.py",
    "wx_ilink_py/qq_routes.py", "wx_ilink_py/restart.py", "wx_ilink_py/routes.py", "wx_ilink_py/schedule_routes.py",
    "wx_ilink_py/setup_routes.py", "wx_ilink_py/voice_routes.py", "wx_ilink_py/web_routes.py",
    "ai/control_config.py", "ai/permissions.py", "ai/mcp_admin.py", "ai/diagnostics.py", "ai/root_tasks.py",
    "ai/persona_templates.py", "ai/context_cut.py", "ai/schedule/importer.py", "ai/schedule/school_xls.py",
    # 2026-11-09 (found by rendering every tab in English): setup checklist, log page, download sources, voice install
    "ai/setup_status.py", "ai/log_viewer.py", "ai/updates.py", "tools/violet_sources.py", "tools/violet_voice.py",
]
# modules that also hold model prompts: only the named dicts / the text arguments of the named calls reach the page
PY_PARTS = {
    "ai/chat.py": {"calls": {"node_skip": 1, "node_queue": 1}},          # the flow page's step notes
    "ai/social/store.py": {"names": {"MODE_TEXT"}},                       # how she takes part in a group
    "ai/stickers.py": {"names": {"SOURCES"}},                             # where a sticker came from
    "ai/tool_manager.py": {"names": {"problem"}},                         # the web tools' status on the tools page
}
# shown on the persona page: each template's title and summary
JSON_FIELDS = {"persona/templates/*.json": ("_title", "_summary")}
PY_SKIP_CALLS = {"debug", "info", "warning", "error", "exception", "critical", "log"}
# not words on the page: template / HTML source, regular expressions, shell commands
NOT_TEXT = re.compile(r"\{\{|\{%|</?[a-z][a-z0-9]*[\s>/]|^\^|\\[sdwbS]|2>/dev/null|\$\(|; then |<!doctype", re.I)


def has_text(s: str) -> bool:
    return bool(HAN.search(s))


def has_pattern_text(s: str) -> bool:
    """A pattern counts with only CJK punctuation fixed ("MCP {0}：{1}"): the values bring the Chinese."""
    if has_text(s):
        return True
    fixed = re.sub(r"\{\d+\}", "", s)
    # control_i18n.js uses a pattern only with 2+ fixed letters ("{0}：{1}" / "【{0}】" alone match anything)
    return bool(CJK.search(fixed)) and len(re.sub(r"[\s\W_]", "", fixed)) >= 2


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ----------------------------------------------------------------------------------------------------- JavaScript
class JS:
    """A small JavaScript scanner: comments, strings, template literals (nested ${...}), regex literals."""

    REGEX_BEFORE = set("(,=:[!&|?{};+-*%<>~^") | {""}

    def __init__(self, text: str):
        self.t, self.i, self.out = text, 0, []

    def scan(self) -> list[str]:
        self._code(end=None)
        return self.out

    def _prev_significant(self) -> str:
        j = self.i - 1
        while j >= 0 and self.t[j] in " \t\r\n":
            j -= 1
        if j < 0:
            return ""
        if self.t[j].isalnum() or self.t[j] in "_$":
            k = j
            while k >= 0 and (self.t[k].isalnum() or self.t[k] in "_$"):
                k -= 1
            word = self.t[k + 1:j + 1]
            return "" if word in ("return", "typeof", "case", "of", "in", "void", "delete") else "a"
        return self.t[j]

    def _code(self, end: str | None) -> None:
        t = self.t
        depth = 0
        pending: list[tuple[str, bool, list[str]]] = []      # literal pieces joined by +
        while self.i < len(t):
            c = t[self.i]
            if end == "}" and c == "}" and depth == 0:
                self._flush(pending)
                return
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            if t.startswith("//", self.i):
                self.i = t.find("\n", self.i) if t.find("\n", self.i) != -1 else len(t)
                continue
            if t.startswith("/*", self.i):
                self.i = t.find("*/", self.i + 2) + 2
                continue
            if c in "\"'":
                value = self._string(c)
                self._push(pending, value, False, [])
                continue
            if c == "`":
                value, is_pattern = self._template()
                self._push(pending, value, is_pattern, [])
                continue
            if c == "/" and self._prev_significant() in self.REGEX_BEFORE:
                self._regex()
                continue
            if c == "+" and pending and self._next_is_literal():
                self.i += 1
                pending.append(("+", False, []))
                continue
            if c not in " \t\r\n":
                self._flush(pending)
            self.i += 1
        self._flush(pending)

    def _next_is_literal(self) -> bool:
        j = self.i + 1
        while j < len(self.t) and self.t[j] in " \t\r\n":
            j += 1
        return j < len(self.t) and self.t[j] in "\"'`"

    def _push(self, pending, value, is_pattern, _):
        if pending and pending[-1][0] == "+":
            pending.pop()
            prev_value, prev_pat, _ = pending.pop()
            if is_pattern or prev_pat:                     # renumber the second pattern's placeholders
                count = len(re.findall(r"\{\d+\}", prev_value))
                value = re.sub(r"\{(\d+)\}", lambda m: "{%d}" % (int(m.group(1)) + count), value)
            pending.append((prev_value + value, prev_pat or is_pattern, []))
        else:
            self._flush(pending)
            pending.append((value, is_pattern, []))

    def _flush(self, pending) -> None:
        for value, is_pattern, _ in pending:
            if value != "+" and (has_pattern_text(value) if is_pattern else has_text(value)):
                self.out.append(value)
        pending.clear()

    def _string(self, q: str) -> str:
        t, i = self.t, self.i + 1
        buf = []
        while i < len(t) and t[i] != q:
            if t[i] == "\\":
                nxt = t[i + 1]
                buf.append({"n": "\n", "t": "\t", "\\": "\\", "\"": "\"", "'": "'"}.get(nxt, nxt))
                i += 2
                continue
            buf.append(t[i])
            i += 1
        self.i = i + 1
        return "".join(buf)

    def _template(self) -> tuple[str, bool]:
        t = self.t
        self.i += 1
        buf, n, pattern = [], 0, False
        while self.i < len(t) and t[self.i] != "`":
            if t[self.i] == "\\":
                nxt = t[self.i + 1]
                buf.append({"n": "\n", "t": "\t"}.get(nxt, nxt))
                self.i += 2
                continue
            if t.startswith("${", self.i):
                self.i += 2
                self._code(end="}")                         # strings inside the expression are keys of their own
                self.i += 1
                buf.append("{%d}" % n)
                n += 1
                pattern = True
                continue
            buf.append(t[self.i])
            self.i += 1
        self.i += 1
        return "".join(buf), pattern

    def _regex(self) -> None:
        t, i, in_class = self.t, self.i + 1, False
        while i < len(t):
            if t[i] == "\\":
                i += 2
                continue
            if t[i] == "[":
                in_class = True
            elif t[i] == "]":
                in_class = False
            elif t[i] == "/" and not in_class:
                break
            elif t[i] == "\n":
                break
            i += 1
        self.i = i + 1


# ----------------------------------------------------------------------------------------------------------- HTML
class Page(HTMLParser):
    ATTRS = ("placeholder", "title", "aria-label", "alt")

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        for k, v in attrs:
            if k in self.ATTRS and v and has_text(v):
                self.out.append(norm(v))

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip and has_text(data):
            self.out.append(norm(data))


# --------------------------------------------------------------------------------------------------------- Python
def python_strings(path: Path, part: dict | None = None) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if part:                                       # only the named dicts / call arguments (see PY_PARTS)
        picked: list[ast.AST] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in part.get("names", ())
                                                    for t in node.targets):
                picked.append(node.value)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name)):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                index = part.get("calls", {}).get(name)
                if index is not None and len(node.args) > index:
                    picked.append(node.args[index])
        tree = ast.Module(body=[ast.Expr(value=n) for n in picked], type_ignores=[])
    skip: set[int] = set()
    for node in ast.walk(tree):
        # internal keyword lists (e.g. the reply words in ai/permissions.py): matched, never shown
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.endswith("_WORDS") for t in node.targets):
            for sub in ast.walk(node):
                skip.add(id(sub))
        # docstrings
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                skip.add(id(first.value))
        # log messages
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in PY_SKIP_CALLS:
            for sub in ast.walk(node):
                skip.add(id(sub))
    out = []
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.JoinedStr):
            parts, n = [], 0
            for v in node.values:
                if isinstance(v, ast.Constant):
                    parts.append(str(v.value))
                else:
                    parts.append("{%d}" % n)
                    n += 1
                for sub in ast.walk(v):
                    skip.add(id(sub))
            text = "".join(parts)
            if has_pattern_text(text) if n else has_text(text):
                out.append(text)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and has_text(node.value):
            out.append(node.value)
    return out


# ------------------------------------------------------------------------------------------------------------ all
def split_lines(texts: list[str]) -> list[str]:
    """Texts with line breaks are shown as separate lines (white-space: pre-line) or whole; keep both forms."""
    out = []
    for s in texts:
        out.append(s)
        if "\n" in s:
            out.extend(x for x in s.split("\n") if has_text(x))
    return out


def collect() -> dict[str, list[str]]:
    found: dict[str, set[str]] = {}

    def add(key: str, where: str) -> None:
        key = key.strip()
        ok = has_pattern_text(key) if re.search(r"\{\d+\}", key) else has_text(key)
        # "<redacted>" is the page's own word for a hidden secret, not an HTML tag
        if key and ok and len(key) <= 1200 and not NOT_TEXT.search(key.replace("<redacted>", "")):
            found.setdefault(key, set()).add(where)
            # " · 用过 {0} 次" appended to another text: the page shows it after a " · ", which the page splits on
            if key.startswith("· ") and len(key) > 2:
                add(key[2:], where)

    page = Page()
    page.feed((WEB / "templates" / "control.html").read_text(encoding="utf-8"))
    for s in page.out:
        add(s, "control.html")
    for js in sorted((WEB / "static").glob("*.js")):
        if js.name == "control_i18n.js":
            continue
        for s in split_lines(JS(js.read_text(encoding="utf-8")).scan()):
            add(s, js.name)
    for rel in PY_FILES:
        path = APP / rel
        if path.is_file():
            for s in split_lines(python_strings(path)):
                add(s, rel)
    for rel, part in PY_PARTS.items():
        path = APP / rel
        if path.is_file():
            for s in split_lines(python_strings(path, part)):
                add(s, rel)
    for pattern, fields in JSON_FIELDS.items():
        for path in sorted(APP.glob(pattern)):
            data = json.loads(path.read_text(encoding="utf-8"))
            for f in fields:
                if isinstance(data.get(f), str):
                    add(data[f], path.relative_to(APP).as_posix())
    # the login page (server-rendered): its t("...") calls
    login = (WEB / "auth.py").read_text(encoding="utf-8")
    for m in re.finditer(r"""\bt\((["'])(.+?)\1\)""", login):
        add(m.group(2), "login page")
    return {k: sorted(v) for k, v in sorted(found.items())}


def missing(lang: str, keys) -> list[str]:
    try:
        data = json.loads((OUT / f"{lang}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return [k for k in keys if not data.get(k)]


def main(argv: list[str]) -> int:
    found = collect()
    if "--check" in argv:
        bad = {lang: len(missing(lang, found)) for lang in LANGS}
        print(json.dumps(bad))
        return 1 if any(bad.values()) else 0
    OUT.mkdir(exist_ok=True)
    (OUT / "source.json").write_text(json.dumps(found, ensure_ascii=False, indent=0), encoding="utf-8")
    by_file: dict[str, int] = {}
    for where in found.values():
        for w in where:
            by_file[w] = by_file.get(w, 0) + 1
    for k, v in sorted(by_file.items(), key=lambda x: -x[1]):
        print(f"{v:5d}  {k}")
    print("keys", len(found), "chars", sum(len(k) for k in found),
          "patterns", sum(1 for k in found if re.search(r"\{\d+\}", k)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
