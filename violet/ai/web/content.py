"""Turn fetched web content into short, inert text (phase 5B).

* HTML -> plain text with the standard library parser; scripts, styles,
  templates, iframes, forms and hidden elements are dropped.
* Publication time only from structured metadata (article:published_time,
  datePublished, <time datetime>, ...) -- never guessed from the text. A date in
  the future is discarded.
* Prompt-injection neutralisation: lines that try to instruct a model
  ("ignore previous instructions", "你现在是…", "调用工具", chat-template
  tokens, ...) are replaced by a marker, and the result says so. The text is
  then only ever handed to a model inside the untrusted-data wrapper of the MCP
  bridge (never as a system message).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser

SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe", "object", "form", "button",
             "select", "textarea", "nav", "footer", "head"}
BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "tr", "section", "article",
              "pre", "dd", "dt"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track",
             "wbr"}
DATE_META = {"article:published_time", "og:published_time", "datepublished", "pubdate", "publishdate",
             "date", "dc.date", "dc.date.issued", "sailthru.date", "parsely-pub-date"}
INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|messages)"
    r"|disregard\s+(all\s+)?(previous|prior|above)"
    r"|you\s+are\s+now\b|new\s+instructions\s*:|system\s+prompt|developer\s+mode|jailbreak|admin\s+mode"
    r"|<\|im_start\|>|<\|im_end\|>|<\|system\|>|\[/?INST\]|<<SYS>>|^\s*(system|assistant)\s*:"
    r"|call\s+the\s+\w+\s+tool|use\s+the\s+\w+\s+tool|fetch\s+https?://"
    r"|忽略(之前|以上|前面|上面|先前)(的)?(所有)?(指令|提示|要求|规则|设定)"
    r"|无视(之前|以上|前面)(的)?(指令|规则)|你现在是|你的新(任务|指令)|系统提示词?|开发者模式|管理员模式"
    r"|调用.{0,8}工具|把.{0,20}(发送|发给|上传|泄露)|访问\s*https?://)",
    re.IGNORECASE | re.MULTILINE,
)
INJECTION_MARK = "[已移除疑似指令的文本]"
_ZERO_WIDTH_RE = re.compile(r"[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
_CJK_SPACE_RE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")
_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
_HIDDEN_STYLE_RE = re.compile(
    r"(?:^|;)\s*(?:"
    r"display\s*:\s*none"
    r"|visibility\s*:\s*(?:hidden|collapse)"
    r"|opacity\s*:\s*0(?:\.0+)?%?"
    r"|font-size\s*:\s*0(?:\.0+)?(?:px|em|rem|pt|%|ch|vw|vh)?"
    r")(?:\s*!important)?\s*(?:;|$)",
    re.IGNORECASE,
)


def _is_hidden_style(style: str) -> bool:
    if not style:
        return False
    cleaned = _CSS_COMMENT_RE.sub("", style)
    return bool(_HIDDEN_STYLE_RE.search(cleaned))


@dataclass
class PageContent:
    title: str
    description: str
    text: str
    published_at: str | None
    injection_removed: int
    truncated: bool = False                 # phase 9.3: the text was cut at max_chars


class _Extractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.in_title = False
        self.title_parts: list[str] = []
        self.parts: list[str] = []
        self.meta: dict[str, str] = {}
        self.times: list[str] = []
        self.ld_json: list[str] = []
        self._in_ld = False
        self.stack: list[tuple[str, bool]] = []

    def handle_starttag(self, tag, attrs):
        attrs = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            key = (attrs.get("property") or attrs.get("name") or attrs.get("itemprop") or "").lower()
            if key and attrs.get("content"):
                self.meta.setdefault(key, attrs["content"])
            return
        if tag == "time" and attrs.get("datetime"):
            self.times.append(attrs["datetime"])
        if tag == "script" and attrs.get("type", "").lower() == "application/ld+json":
            self._in_ld = True
        if tag == "title":
            self.in_title = True
        if tag in VOID_TAGS:
            if tag == "br":
                self.parts.append("\n")
            return
        hidden = ("hidden" in attrs or _is_hidden_style(attrs.get("style", ""))
                  or attrs.get("aria-hidden", "").strip().lower() == "true")
        skipping = tag in SKIP_TAGS or hidden
        self.stack.append((tag, skipping))
        if skipping:
            self.skip += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag == "script":
            self._in_ld = False
        if not any(t == tag for t, _ in self.stack):
            return                          # stray end tag
        while self.stack:
            open_tag, skipping = self.stack.pop()
            if skipping:
                self.skip -= 1
            if open_tag == tag:
                break
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_ld:
            self.ld_json.append(data)
            return
        if self.in_title:
            self.title_parts.append(data)
            return
        if not self.skip:
            self.parts.append(data)


def parse_published(value: str | None, now: datetime | None = None) -> str | None:
    if not value:
        return None
    value = value.strip()
    candidates = [value, value.replace("Z", "+00:00"), value[:10]]
    for c in candidates:
        try:
            dt = datetime.fromisoformat(c)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt > (now or datetime.now(timezone.utc)) + timedelta(days=1):
            return None
        return dt.isoformat()
    return None


def _join_window(window: list[str]) -> str:
    joined = " ".join(part.strip() for part in window if part.strip())
    return _CJK_SPACE_RE.sub("", joined)


def neutralize(text: str) -> tuple[str, int]:
    raw = _ZERO_WIDTH_RE.sub("", text or "")
    lines = raw.splitlines()
    if not lines:
        return "", 0
    # Mark spans: None = kept as-is, int = span end index (collapsed to one INJECTION_MARK), -1 = covered by earlier span
    state: list[int | None] = [None] * len(lines)
    removed = 0
    for i, line in enumerate(lines):
        if INJECTION_RE.search(line):
            state[i] = i
            removed += 1
    for width in (2, 3, 4):
        i = 0
        while i + width <= len(lines):
            if any(state[j] is not None for j in range(i, i + width)):
                i += 1
                continue
            if not lines[i].strip() or not lines[i + width - 1].strip():
                i += 1
                continue
            if INJECTION_RE.search(_join_window(lines[i:i + width])):
                state[i] = i + width - 1
                for j in range(i + 1, i + width):
                    state[j] = -1
                removed += 1
                i += width
            else:
                i += 1
    out: list[str] = []
    for i, line in enumerate(lines):
        if state[i] is None:
            out.append(line)
        elif state[i] != -1:
            out.append(INJECTION_MARK)
    return "\n".join(out), removed


def _clean(text: str) -> str:
    text = _ZERO_WIDTH_RE.sub("", text or "")
    text = re.sub(r"[ \t　\xa0]+", " ", text)
    lines = [l.strip() for l in text.splitlines()]
    return "\n".join(l for l in lines if l)


def extract(body: str, content_type: str, max_chars: int = 4000) -> PageContent:
    if content_type in ("text/html", "application/xhtml+xml"):
        parser = _Extractor()
        try:
            parser.feed(body)
            parser.close()
        except Exception:
            pass
        title = _clean("".join(parser.title_parts))[:200]
        description = (parser.meta.get("description") or parser.meta.get("og:description") or "")[:400]
        published = None
        for key in DATE_META:
            published = published or parse_published(parser.meta.get(key))
        for raw in parser.ld_json:
            if published:
                break
            try:
                data = json.loads(raw)
            except ValueError:
                continue
            for item in data if isinstance(data, list) else [data]:
                if isinstance(item, dict):
                    published = published or parse_published(str(item.get("datePublished") or ""))
        for t in parser.times:
            published = published or parse_published(t)
        text = _clean("".join(parser.parts))
    else:
        title, description, published, text = "", "", None, _clean(body)
    text, removed = neutralize(text)
    title, r1 = neutralize(title)
    description, r2 = neutralize(description)
    return PageContent(title=title, description=description, text=text[:max_chars], published_at=published,
                       injection_removed=removed + r1 + r2, truncated=len(text) > max_chars)
