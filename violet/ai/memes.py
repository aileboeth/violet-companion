"""Memes she gets (2026-11-12, user: "他还是感觉不太理解我有的梗"; idea from the DSH plugin thawline/dsh-memewiki,
itself inspired by the AstrBot plugin astrbot_plugin_memewiki).

Two places a meme can come from:

  the meme wiki   <data>/memes/cnmeme.db -- entries of cnmeme.wiki (title, aliases, tags, how it is used, where it
                  comes from). NOT shipped with Violet (the site states no licence for its entries): the owner downloads
                  it from the site's public API on the tools page (sync(): the list first -- usable after ~2 minutes --
                  then every entry's full text, slowly, in the background; resumable), or imports a file in the plugin's
                  format (import_file). Reading it never goes on the network.
  your memes      table owner_memes (one per persona, like her other memories; cleared with them): what the owner told
                  her a meme / catchphrase / running joke of theirs means -- she keeps it with the remember_meme tool, the
                  owner can edit it on the tools page.

Every owner turn the message is scanned for known memes (scan(): whole titles / aliases of 3+ characters, a 2-character
one only when it is the whole message; ASCII ones only as a whole word) and at most MAX_SHOWN entries go into her context
(context_engine._meme_section) -- nothing when nothing matches. For the rest she has the look_up_meme tool.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ai.settings import env_bool

logger = logging.getLogger("violet.memes")

WIKI_API = "https://cnmeme.wiki/api/v1/public/memes"
WIKI_SITE = "https://cnmeme.wiki"
USER_AGENT = "CNMemeWikiPlugin/1.0 (Violet)"      # the site's API docs: set this one, a default client may be blocked
PAGE = 30                                          # the list API allows at most 30 per page
LIST_PAUSE = 0.15
DETAIL_PAUSE = 0.35
DETAIL_WORKERS = 3
MAX_SHOWN = 2
TOOL_NAMES = {"look_up_meme", "remember_meme"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS owner_memes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phrase TEXT NOT NULL UNIQUE,
    meaning TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'chat',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

WIKI_SCHEMA = """
CREATE TABLE IF NOT EXISTS memes (
    title TEXT PRIMARY KEY,
    slug TEXT,
    tags TEXT,
    usage TEXT,
    origin TEXT,
    other TEXT,
    summary TEXT,
    aliases TEXT,
    keywords TEXT,
    updated_at TEXT,
    detail_at TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""
_WIKI_COLUMNS = ("title", "slug", "tags", "usage", "origin", "other", "summary", "aliases", "keywords", "updated_at",
                 "detail_at")

# ordinary words that are also wiki titles: never matched in a message (look_up_meme still finds them)
PLAIN_WORDS = frozenset({"表情包", "吃什么", "为什么", "怎么了", "没关系", "不知道", "好不好", "是不是", "干什么", "怎么办",
                         "睡不着", "好的呢", "哈哈哈", "笑死我", "我知道", "真的吗", "不是吧"})


def enabled() -> bool:
    return env_bool("VIOLET_MEMES", True)


def _data_dir() -> Path:
    return Path(os.environ.get("VIOLET_DATA_DIR") or "./data")


def wiki_path() -> Path:
    return _data_dir() / "memes" / "cnmeme.db"


def _clip(text: Any, n: int) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[:n] + "…"


def _norm(text: str) -> str:
    return (text or "").strip().lower()


# ---------------------------------------------------------------------------------------------- the wiki file
def _connect(path: Path, write: bool = False) -> sqlite3.Connection:
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=10, check_same_thread=False)
        conn.executescript(WIKI_SCHEMA)
        have = {r[1] for r in conn.execute("PRAGMA table_info(memes)")}
        for col in _WIKI_COLUMNS:
            if col not in have:                       # a file in the plugin's format (no keywords / updated_at / detail_at)
                conn.execute(f"ALTER TABLE memes ADD COLUMN {col} TEXT")
        conn.commit()
    else:
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def wiki_available() -> bool:
    return enabled() and wiki_path().is_file()


def wiki_info() -> dict[str, Any]:
    path = wiki_path()
    out = {"installed": path.is_file(), "count": 0, "full": 0, "built_at": "", "size": 0}
    if not path.is_file():
        return out
    try:
        conn = _connect(path)
        try:
            out["count"] = conn.execute("SELECT COUNT(*) FROM memes").fetchone()[0]
            cols = {r[1] for r in conn.execute("PRAGMA table_info(memes)")}
            full = "usage IS NOT NULL AND usage != ''" if "detail_at" not in cols else \
                "detail_at IS NOT NULL OR (usage IS NOT NULL AND usage != '')"
            out["full"] = conn.execute(f"SELECT COUNT(*) FROM memes WHERE {full}").fetchone()[0]
            row = conn.execute("SELECT value FROM meta WHERE key = 'built_at'").fetchone()
            out["built_at"] = row[0] if row else ""
        finally:
            conn.close()
        out["size"] = path.stat().st_size
    except sqlite3.Error as exc:
        out["error"] = str(exc)[:200]
    return out


class _Index:
    """Titles and aliases of the wiki in memory, rebuilt when the file changes."""

    def __init__(self) -> None:
        self.stamp: tuple[float, int] | None = None
        self.first: dict[str, list[str]] = {}       # first character -> keys starting with it, longest first
        self.keys: dict[str, str] = {}               # key (lower case) -> title
        self.lock = threading.Lock()

    def load(self) -> "_Index":
        path = wiki_path()
        try:
            st = path.stat()
        except OSError:
            self.stamp, self.first, self.keys = None, {}, {}
            return self
        stamp = (st.st_mtime, st.st_size)
        with self.lock:
            if stamp == self.stamp:
                return self
            keys: dict[str, str] = {}
            try:
                conn = _connect(path)
                try:
                    for row in conn.execute("SELECT title, aliases FROM memes"):
                        title = str(row["title"] or "").strip()
                        if not title:
                            continue
                        for key in [title, *str(row["aliases"] or "").split(",")]:
                            key = _norm(key)
                            if len(key) >= 2 and key not in keys:
                                keys[key] = title
                finally:
                    conn.close()
            except sqlite3.Error:
                logger.exception("meme wiki unreadable")
            first: dict[str, list[str]] = {}
            for key in keys:
                first.setdefault(key[0], []).append(key)
            for bucket in first.values():
                bucket.sort(key=len, reverse=True)
            self.stamp, self.first, self.keys = stamp, first, keys
        return self


_INDEX = _Index()
_ASCII = re.compile(r"^[0-9a-z][0-9a-z .'\-]*$")
_WORD_CHAR = re.compile(r"[0-9a-z]")
_MIXED = re.compile(r"[一-鿿][0-9a-z]|[0-9a-z][一-鿿]")
_PUNCT = re.compile(r"[\s,，。.!！?？~～…、；;：:\"'“”‘’()（）\[\]【】<>《》]+")


def _wiki_rows(titles: list[str]) -> list[dict[str, Any]]:
    if not titles or not wiki_path().is_file():
        return []
    try:
        conn = _connect(wiki_path())
        try:
            rows = {r["title"]: dict(r) for r in conn.execute(
                f"SELECT * FROM memes WHERE title IN ({','.join('?' * len(titles))})", titles)}
        finally:
            conn.close()
    except sqlite3.Error:
        logger.exception("meme wiki unreadable")
        return []
    return [rows[t] for t in titles if t in rows]


def _scan_keys(text: str, keys: dict[str, Any], first: dict[str, list[str]], whole_only_short: bool) -> list[tuple[int, int, str]]:
    """(start, end, key) of every key found in text, longest first, not overlapping."""
    low = _norm(text)
    bare = _PUNCT.sub("", low)
    found: list[tuple[int, int, str]] = []
    for i, ch in enumerate(low):
        for key in first.get(ch, ()):
            if not low.startswith(key, i):
                continue
            n = len(key)
            if key in PLAIN_WORDS:
                continue
            if _ASCII.match(key):
                if n < 3:
                    continue
                before = low[i - 1] if i else ""
                after = low[i + n] if i + n < len(low) else ""
                if _WORD_CHAR.match(before or " ") or _WORD_CHAR.match(after or " "):
                    continue                          # "emo" in "demo", "sc" in "school"
            elif whole_only_short and n < 3 and bare != key and not _MIXED.search(key):
                # "得吃" in "还得吃饭": a 2-character Chinese title only as the whole message ("栓q" -- Chinese and
                # Latin mixed -- is no ordinary word: anywhere)
                continue
            found.append((i, i + n, key))
    found.sort(key=lambda f: (-(f[1] - f[0]), f[0]))
    taken: list[tuple[int, int, str]] = []
    for f in found:
        if all(f[1] <= t[0] or f[0] >= t[1] for t in taken):
            taken.append(f)
    return taken


# ---------------------------------------------------------------------------------------------- your memes
def own_list(db) -> list[dict[str, Any]]:
    try:
        return db.fetch_all("SELECT * FROM owner_memes ORDER BY updated_at DESC, id DESC")
    except Exception:                                  # noqa: BLE001 -- an old database without the table
        return []


def own_save(db, phrase: str, meaning: str, source: str = "chat") -> dict[str, Any]:
    phrase = " ".join(str(phrase or "").split())[:40]
    meaning = " ".join(str(meaning or "").split())[:300]
    if len(phrase) < 1 or not meaning:
        raise ValueError("梗和意思都要写")
    now = db.now()
    old = db.fetch_one("SELECT id FROM owner_memes WHERE phrase = ?", (phrase,))
    if old:
        db.execute("UPDATE owner_memes SET meaning = ?, source = ?, updated_at = ? WHERE id = ?",
                   (meaning, source, now, old["id"]))
    else:
        db.execute("INSERT INTO owner_memes (phrase, meaning, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                   (phrase, meaning, source, now, now))
    return db.fetch_one("SELECT * FROM owner_memes WHERE phrase = ?", (phrase,))


def own_update(db, meme_id: int, phrase: str | None = None, meaning: str | None = None) -> dict[str, Any] | None:
    row = db.fetch_one("SELECT * FROM owner_memes WHERE id = ?", (meme_id,))
    if row is None:
        return None
    phrase = " ".join(str(row["phrase"] if phrase is None else phrase).split())[:40]
    meaning = " ".join(str(row["meaning"] if meaning is None else meaning).split())[:300]
    if not phrase or not meaning:
        raise ValueError("梗和意思都要写")
    clash = db.fetch_one("SELECT id FROM owner_memes WHERE phrase = ? AND id != ?", (phrase, meme_id))
    if clash:
        raise ValueError(f"已经有「{phrase}」了")
    db.execute("UPDATE owner_memes SET phrase = ?, meaning = ?, source = 'page', updated_at = ? WHERE id = ?",
               (phrase, meaning, db.now(), meme_id))
    return db.fetch_one("SELECT * FROM owner_memes WHERE id = ?", (meme_id,))


def own_delete(db, meme_id: int) -> bool:
    if db.fetch_one("SELECT id FROM owner_memes WHERE id = ?", (meme_id,)) is None:
        return False
    db.execute("DELETE FROM owner_memes WHERE id = ?", (meme_id,))
    return True


# ---------------------------------------------------------------------------------------------- finding them
def scan(db, text: str, limit: int = MAX_SHOWN) -> list[dict[str, Any]]:
    """The memes a message uses: yours first, then the wiki's. [{kind, title, ...}]"""
    if not enabled() or not (text or "").strip():
        return []
    out: list[dict[str, Any]] = []
    own = {_norm(r["phrase"]): r for r in own_list(db) if _norm(r["phrase"])}
    if own:
        first: dict[str, list[str]] = {}
        for key in sorted(own, key=len, reverse=True):
            first.setdefault(key[0], []).append(key)
        for _, _, key in _scan_keys(text, own, first, whole_only_short=False):
            out.append({"kind": "own", "title": own[key]["phrase"], "meaning": own[key]["meaning"]})
    if len(out) < limit and wiki_path().is_file():
        index = _INDEX.load()
        mine = {_norm(o["title"]) for o in out}
        titles = []
        for _, _, key in _scan_keys(text, index.keys, index.first, whole_only_short=True):
            title = index.keys[key]
            if _norm(title) not in mine and title not in titles:
                titles.append(title)
        out += [{"kind": "wiki", **r} for r in _wiki_rows(titles[:limit - len(out)])]
    return out[:limit]


def lookup(db, query: str, limit: int = 3) -> list[dict[str, Any]]:
    """look_up_meme: the meme a word / phrase is (the DSH plugin's order: exact, alias, prefix, the query containing a
    title, a title containing the query)."""
    q = " ".join(str(query or "").split())[:60]
    if not q:
        return []
    ql = q.lower()
    out: list[dict[str, Any]] = []
    for r in own_list(db):
        p = _norm(r["phrase"])
        if p and (p == ql or p in ql or (len(ql) >= 2 and ql in p)):
            out.append({"kind": "own", "title": r["phrase"], "meaning": r["meaning"]})
    if not wiki_available():
        return out[:limit]
    try:
        conn = _connect(wiki_path())
        try:
            seen = {o["title"] for o in out}
            queries = [
                ("SELECT * FROM memes WHERE lower(title) = ?", (ql,)),
                ("SELECT * FROM memes WHERE aliases IS NOT NULL AND instr(',' || lower(aliases) || ',', ',' || ? || ',') > 0 "
                 "LIMIT 2", (ql,)),
                ("SELECT * FROM memes WHERE lower(title) LIKE ? || '%' ORDER BY length(title) LIMIT 2", (ql,)),
                ("SELECT * FROM memes WHERE length(title) >= 2 AND instr(?, lower(title)) > 0 ORDER BY length(title) DESC "
                 "LIMIT 2", (ql,)),
                ("SELECT * FROM memes WHERE keywords IS NOT NULL AND instr(' ' || lower(keywords) || ' ', ' ' || ? || ' ') > 0 "
                 "ORDER BY length(title) LIMIT 2", (ql,)),
            ]
            if len(ql) >= 3 or (len(ql) >= 2 and _ASCII.match(ql)):
                queries.append(("SELECT * FROM memes WHERE instr(lower(title), ?) > 0 ORDER BY length(title) LIMIT 2", (ql,)))
                queries.append(("SELECT * FROM memes WHERE keywords IS NOT NULL AND instr(lower(keywords), ?) > 0 "
                                "ORDER BY length(title) LIMIT 2", (ql,)))
            for sql, params in queries:
                for row in conn.execute(sql, params):
                    if row["title"] not in seen:
                        seen.add(row["title"])
                        out.append({"kind": "wiki", **dict(row)})
                if len(out) >= limit:
                    break
        finally:
            conn.close()
    except sqlite3.Error:
        logger.exception("meme lookup failed")
    return out[:limit]


def describe(entry: dict[str, Any], short: bool = True) -> str:
    """One entry as she reads it: how it is used first (so she can play along), then what it means / where from."""
    if entry.get("kind") == "own":
        # not "你们之间的梗": the owner's joke with a roommate made her say she was coming along (virtual test 11-13)
        return f"「{entry['title']}」（对方以前教过你的说法）：{_clip(entry.get('meaning'), 200)}"
    usage = _clip(entry.get("usage"), 160 if short else 500)
    meaning = _clip(entry.get("summary") or entry.get("origin"), 140 if short else 400)
    parts = [f"「{entry.get('title')}」"]
    tags = "、".join([t for t in re.split(r"[、,，\s]+", str(entry.get("tags") or "")) if t][:3])
    if tags:
        parts.append(f"（{tags}）")
    if usage:
        parts.append(f"：怎么用：{usage}")
        if meaning:
            parts.append(f"；意思和出处：{meaning}")
    elif meaning:
        parts.append(f"：{meaning}")
    return "".join(parts)


CONTEXT_HEADER = "【对方这句话里可能用了梗（查到的资料，不是对方说的；也可能只是字面意思，按上下文判断）】"
CONTEXT_FOOTER = ("用了的话就像你本来就知道一样自然接住、顺着玩，别科普、别念资料；对方问“是什么梗”才简单说一两句。"
                  "资料里没有的梗别硬编：没把握就用 look_up_meme 查，查不到就直说不知道或者问对方。")


def context_lines(db, text: str) -> list[str]:
    try:
        return ["- " + describe(e) for e in scan(db, text)]
    except Exception:                                  # noqa: BLE001 -- never break a reply over this
        logger.exception("meme scan failed")
        return []


# ---------------------------------------------------------------------------------------------- her tools
SCHEMAS = [
    {"type": "function", "function": {
        "name": "look_up_meme",
        "description": "查一个网络梗、黑话、缩写，或你们之间的梗是什么意思、怎么接（本地词库，不联网，很快）。对方说了你没把握的新词、"
                       "奇怪的说法或缩写时先查，别硬猜、别编。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "要查的词或那句话里的梗"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "remember_meme",
        "description": "对方告诉你一个梗、口头禅或他自己圈子里的说法是什么意思（或者纠正你理解错了）时，把它记下来，以后对方再说你就懂了。"
                       "只记对方自己讲明白的；你猜的、对方只含糊附和一句“差不多”的不记。meaning 里写清楚这是谁和谁之间的说法。",
        "parameters": {"type": "object", "properties": {
            "phrase": {"type": "string", "description": "梗本身：对方平时会怎么说（40 字以内）"},
            "meaning": {"type": "string", "description": "意思、来历和怎么接（300 字以内）"}}, "required": ["phrase", "meaning"]}}},
]


def tool_look_up(db, args: dict[str, Any]) -> str:
    from ai import trace
    query = str(args.get("query") or "").strip()
    if not query:
        return json.dumps({"ok": False, "message": "要查什么？给一个词。"}, ensure_ascii=False)
    found = lookup(db, query)
    trace.step("查梗", 查=query[:40], 结果=("、".join(f["title"] for f in found) or "没查到"))
    if not found:
        return json.dumps({"ok": True, "found": [], "message": f"词库里没有「{query[:40]}」。别硬解释：说不知道，或者问对方是什么梗。"},
                          ensure_ascii=False)
    return json.dumps({"ok": True, "found": [describe(f, short=False) for f in found],
                       "message": "看懂了就自然接住，别把资料念给对方；不确定是哪一个就按上下文挑，或者问一句。"}, ensure_ascii=False)


def tool_remember(db, args: dict[str, Any]) -> str:
    from ai import trace
    try:
        row = own_save(db, str(args.get("phrase") or ""), str(args.get("meaning") or ""))
    except ValueError as exc:
        return json.dumps({"ok": False, "message": str(exc)}, ensure_ascii=False)
    trace.step("记住一个梗", 梗=row["phrase"], 意思=_clip(row["meaning"], 60))
    return json.dumps({"ok": True, "message": f"记住了「{row['phrase']}」，以后对方再说就会想起来。"}, ensure_ascii=False)


# ---------------------------------------------------------------------------------------------- downloading the wiki
_lock = threading.Lock()
_state: dict[str, Any] = {"running": False, "phase": "", "listed": 0, "added": 0, "changed": 0, "details": 0,
                          "details_total": 0, "failed": 0, "message": "", "started_at": "", "finished_at": ""}
_stop = threading.Event()


def status() -> dict[str, Any]:
    with _lock:
        return dict(_state)


def _text(html_part: str) -> str:
    t = re.sub(r"<br\s*/?>", "\n", html_part or "", flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    return html.unescape(t).replace(" ", " ").strip()


def _blocks(html_part: str) -> str:
    out = []
    for m in re.finditer(r"<(p|ul|ol|blockquote|pre)[^>]*>([\s\S]*?)</\1>", html_part or "", flags=re.I):
        tag, inner = m.group(1).lower(), m.group(2)
        if tag in ("ul", "ol"):
            items = [" ".join(_text(x).split()) for x in re.findall(r"<li[^>]*>([\s\S]*?)</li>", inner, flags=re.I)]
            items = [x for x in items if x]
            if items:
                out.append("\n".join("- " + x for x in items))
        elif tag == "blockquote":
            inner_text = _blocks(inner) or _text(inner)
            if inner_text:
                out.append(inner_text)
        else:
            t = _text(inner)
            if t:
                out.append(t)
    return "\n".join(out)


def sections_of(content_html: str) -> dict[str, str]:
    """An entry's HTML (h2 sections) -> {usage, origin, other}."""
    found = []
    for m in re.finditer(r"<h2[^>]*>([\s\S]*?)</h2>([\s\S]*?)(?=<h2|$)", content_html or "", flags=re.I):
        name = _text(m.group(1)).strip()
        if name:
            found.append((name, _blocks(m.group(2)).strip()))
    pick = lambda n: next((body for name, body in found if name == n), "")   # noqa: E731
    other = "\n\n".join(f"{name}\n{body}" if body else name for name, body in found if name not in ("来源", "用法"))
    return {"usage": pick("用法"), "origin": pick("来源"), "other": other}


def _http_get(url: str, params: dict[str, Any] | None = None) -> Any:
    import requests
    last: Exception | None = None
    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, timeout=20,
                                headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            if resp.status_code == 429 or resp.status_code >= 500:
                last = RuntimeError(f"HTTP {resp.status_code}")
                time.sleep(min(15.0, 1.5 * 2 ** attempt))
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:                       # noqa: BLE001 -- retried, then raised
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(str(last)[:200] if last else "请求失败")


def _join(value: Any, sep: str) -> str:
    if isinstance(value, list):
        return sep.join(str(v) for v in value if v is not None)
    return "" if value is None else str(value)


def _now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sync(fetch: Callable[[str, dict[str, Any] | None], Any] | None = None, details: bool = True,
         list_pause: float = LIST_PAUSE, detail_pause: float = DETAIL_PAUSE) -> dict[str, Any]:
    """Brings <data>/memes/cnmeme.db up to the site: the whole public list (new entries, changed ones), then the full
    text of every entry that has none yet or changed. Runs in the caller's thread (start() runs it in the background).
    Stops early when stop() is called; what is done stays done."""
    fetch = fetch or _http_get
    path = wiki_path()
    conn = _connect(path, write=True)
    try:
        known = {r["title"]: r["updated_at"] or "" for r in conn.execute("SELECT title, updated_at FROM memes")}
        with _lock:
            _state.update(phase="list")
        offset = 0
        while not _stop.is_set():
            data = fetch(WIKI_API, {"limit": PAGE, "offset": offset}) or {}
            items = data.get("items") or []
            if not items:
                break
            for it in items:
                title = str(it.get("title") or "").strip()
                if not title:
                    continue
                updated = str(it.get("updated_at") or "")
                row = (str(it.get("slug") or title), _join(it.get("tags"), "、"), str(it.get("summary") or ""),
                       _join(it.get("aliases"), ","), _join(it.get("keywords"), " "), updated)
                if title not in known:
                    conn.execute("INSERT INTO memes (slug, tags, summary, aliases, keywords, updated_at, title) "
                                 "VALUES (?, ?, ?, ?, ?, ?, ?)", (*row, title))
                    with _lock:
                        _state["added"] += 1
                elif updated and updated != known[title]:
                    # changed on the site: new list fields, the full text fetched again below -- except for an entry of
                    # an imported file seen in the list for the first time (no time of its own yet: its text stays).
                    # Aliases the file already had (filled in by hand) stay next to the site's.
                    again = ", detail_at = NULL" if known[title] else ""
                    old = conn.execute("SELECT aliases FROM memes WHERE title = ?", (title,)).fetchone()
                    aliases = []
                    for a in [*str(row[3]).split(","), *str((old["aliases"] if old else "") or "").split(",")]:
                        if a.strip() and a.strip() not in aliases:
                            aliases.append(a.strip())
                    row = (*row[:3], ",".join(aliases), *row[4:])
                    conn.execute("UPDATE memes SET slug = ?, tags = ?, summary = COALESCE(NULLIF(?, ''), summary), "
                                 "aliases = ?, keywords = ?, updated_at = ?" + again +
                                 " WHERE title = ?", (*row, title))
                    if known[title]:
                        with _lock:
                            _state["changed"] += 1
                known[title] = updated
            conn.commit()
            offset += len(items)
            with _lock:
                _state["listed"] = offset
            if not data.get("has_more"):
                break
            time.sleep(list_pause)
        _set_meta(conn)
        if details and not _stop.is_set():
            todo = [dict(r) for r in conn.execute(
                "SELECT title, slug FROM memes WHERE detail_at IS NULL AND slug IS NOT NULL AND slug != '' ORDER BY rowid")]
            with _lock:
                _state.update(phase="details", details_total=len(todo), details=0)

            def one(item: dict[str, Any]) -> tuple[dict[str, Any], Any, Any]:
                # 2026-11-13 (user's server: one by one would take 1.5-3 hours): DETAIL_WORKERS requests at a time,
                # each worker pausing between its requests (the site asks clients to limit concurrency)
                if _stop.is_set():
                    return item, None, "stopped"
                try:
                    return item, fetch(f"{WIKI_API}/{item['slug']}", None), None
                except Exception as exc:               # noqa: BLE001 -- one entry; the next run tries again
                    return item, None, exc
                finally:
                    time.sleep(detail_pause)
            from concurrent.futures import ThreadPoolExecutor
            done = 0
            with ThreadPoolExecutor(DETAIL_WORKERS, thread_name_prefix="meme-detail") as pool:
                for item, body, err in pool.map(one, todo):      # results written here, in this thread only
                    if err == "stopped":
                        continue
                    if err is not None:
                        logger.info("meme detail %s failed: %s", item["slug"], err)
                        with _lock:
                            _state["failed"] += 1
                        continue
                    entry = body.get("item") if isinstance(body, dict) and isinstance(body.get("item"), dict) else body
                    parts = sections_of(str((entry or {}).get("content_html") or "")) if isinstance(entry, dict) else {}
                    conn.execute("UPDATE memes SET usage = COALESCE(NULLIF(?, ''), usage), origin = COALESCE(NULLIF(?, ''), origin), "
                                 "other = COALESCE(NULLIF(?, ''), other), detail_at = ? WHERE title = ?",
                                 (parts.get("usage", ""), parts.get("origin", ""), parts.get("other", ""), _now_text(),
                                  item["title"]))
                    done += 1
                    if done % 20 == 0:
                        conn.commit()
                    with _lock:
                        _state["details"] = done
            conn.commit()
        _set_meta(conn)
    finally:
        conn.close()
    _INDEX.stamp = None
    return status()


def _set_meta(conn: sqlite3.Connection) -> None:
    count = conn.execute("SELECT COUNT(*) FROM memes").fetchone()[0]
    for key, value in (("built_at", _now_text()), ("count", str(count)), ("source", WIKI_API)):
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


def start(fetch=None) -> dict[str, Any]:
    """sync() in the background (the tools page's button). ValueError when one is running already."""
    with _lock:
        if _state["running"]:
            raise ValueError("正在下载，等这一次完成")
        _state.update(running=True, phase="list", listed=0, added=0, changed=0, details=0, details_total=0, failed=0,
                      message="", started_at=_now_text(), finished_at="")
    _stop.clear()

    def run() -> None:
        try:
            sync(fetch=fetch)
            s = status()
            message = f"列表 {s['listed']} 条（新增 {s['added']}，有更新 {s['changed']}）"
            if s["details_total"]:
                message += f"；补全了 {s['details']}/{s['details_total']} 条的详细用法"
            if s["failed"]:
                message += f"，{s['failed']} 条没取到（下次再取）"
            if _stop.is_set():
                message = "已停止：" + message + "（下次点“更新”会接着下载）"
        except Exception as exc:                       # noqa: BLE001 -- shown on the page
            logger.exception("meme wiki download failed")
            message = f"下载失败：{str(exc)[:200]}"
        with _lock:
            _state.update(running=False, phase="", message=message, finished_at=_now_text())
    threading.Thread(target=run, daemon=True, name="meme-wiki").start()
    return status()


def stop() -> None:
    _stop.set()


def import_file(src: Path) -> dict[str, Any]:
    """A meme file in the DSH plugin's format (or one of ours) becomes the wiki file: its rows are merged in (an entry
    already here with more text keeps it). Raises ValueError when it is not one."""
    try:
        conn = sqlite3.connect(f"file:{Path(src).as_posix()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memes)")}
    except sqlite3.Error as exc:
        raise ValueError(f"不是词库文件：{exc}") from None
    if "title" not in cols:
        conn.close()
        raise ValueError("不是词库文件：没有 memes 表")
    wanted = [c for c in _WIKI_COLUMNS if c in cols]
    rows = [dict(r) for r in conn.execute(f"SELECT {', '.join(wanted)} FROM memes")]
    conn.close()
    dst = _connect(wiki_path(), write=True)
    added = merged = 0
    try:
        for r in rows:
            title = str(r.get("title") or "").strip()
            if not title:
                continue
            old = dst.execute("SELECT * FROM memes WHERE title = ?", (title,)).fetchone()
            if old is None:
                dst.execute(f"INSERT INTO memes ({', '.join(wanted)}) VALUES ({', '.join('?' * len(wanted))})",
                            [r.get(c) for c in wanted])
                if "detail_at" not in r and r.get("usage"):
                    dst.execute("UPDATE memes SET detail_at = ? WHERE title = ?", (_now_text(), title))
                added += 1
            else:
                updates = {c: r[c] for c in wanted if c != "title" and r.get(c)
                           and len(str(r[c])) > len(str(old[c] or ""))}
                if updates:
                    dst.execute(f"UPDATE memes SET {', '.join(c + ' = ?' for c in updates)} WHERE title = ?",
                                [*updates.values(), title])
                    merged += 1
        dst.commit()
        _set_meta(dst)
    finally:
        dst.close()
    _INDEX.stamp = None
    return {"added": added, "merged": merged, **wiki_info()}


def delete_wiki() -> bool:
    path = wiki_path()
    if not path.is_file():
        return False
    for p in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        p.unlink(missing_ok=True)
    _INDEX.stamp = None
    return True
