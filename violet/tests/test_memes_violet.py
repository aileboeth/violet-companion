"""2026-11-12 (user: 他还是感觉不太理解我有的梗): ai/memes.py -- the meme wiki (downloaded / imported, never shipped),
the memes the owner taught her, the scan of every owner message, her two tools and the Control Center routes.
Fake site; no network."""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

TOKEN = os.environ["VIOLET_ADMIN_TOKEN"]
SAME = {"Origin": "http://localhost"}

ENTRIES = [
    {"title": "去码头整点薯条", "slug": "去码头整点薯条", "tags": ["图像梗", "解构"], "aliases": ["码头薯条"],
     "summary": "来源 两只海鸥的漫画。用法 解构宏大叙事", "keywords": ["海鸥"], "updated_at": "2026-11-01T00:00:00+00:00"},
    {"title": "典", "slug": "典", "tags": ["贴吧梗"], "aliases": [], "summary": "太经典了，果然会这么说",
     "updated_at": "2026-11-01T00:00:00+00:00"},
    {"title": "得吃", "slug": "得吃", "tags": [], "aliases": [], "summary": "值得吃", "updated_at": "2026-11-01T00:00:00+00:00"},
    {"title": "emo", "slug": "emo", "tags": [], "aliases": [], "summary": "情绪低落", "updated_at": "2026-11-01T00:00:00+00:00"},
    {"title": "表情包", "slug": "表情包", "tags": [], "aliases": [], "summary": "图片", "updated_at": "2026-11-01T00:00:00+00:00"},
    {"title": "哈基米", "slug": "哈基米", "tags": ["声音梗"], "aliases": ["hajimi"], "summary": "指可爱的猫",
     "updated_at": "2026-11-01T00:00:00+00:00"},
]
DETAIL_HTML = "<h2>来源</h2><p>漫画里一只海鸥问要飞向何方</p><h2>用法</h2><ul><li>别人谈理想时回一句</li></ul><h2>演化</h2><p>变体很多</p>"


class Site:
    """cnmeme.wiki's public API: the list (30 a page) and the detail of one entry."""

    def __init__(self, entries):
        self.entries = [dict(e) for e in entries]
        self.calls: list[str] = []

    def __call__(self, url, params):
        from ai import memes
        self.calls.append(url)
        if url == memes.WIKI_API:
            off, lim = int(params["offset"]), int(params["limit"])
            page = self.entries[off:off + lim]
            return {"schema_version": 1, "items": page, "has_more": off + lim < len(self.entries)}
        slug = url.rsplit("/", 1)[1]
        e = next((x for x in self.entries if x["slug"] == slug), None)
        return {"schema_version": 1, "item": {**e, "content_html": DETAIL_HTML}} if e else None


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    from ai import memes
    memes._INDEX.stamp = None
    memes._stop.clear()
    yield tmp_path / "data"
    memes._INDEX.stamp = None


@pytest.fixture()
def wiki(data_dir):
    from ai import memes
    memes.sync(fetch=Site(ENTRIES), list_pause=0, detail_pause=0)
    return memes.wiki_path()


def test_download_lists_then_fills_in_each_entry_and_only_fetches_what_changed(data_dir):
    from ai import memes
    site = Site(ENTRIES)
    memes.sync(fetch=site, list_pause=0, detail_pause=0)
    info = memes.wiki_info()
    assert info["count"] == 6 and info["full"] == 6
    row = memes.lookup(None, "去码头整点薯条")[0]
    assert row["usage"] == "- 别人谈理想时回一句" and row["origin"].startswith("漫画里") and "演化" in row["other"]
    assert row["aliases"] == "码头薯条" and row["tags"] == "图像梗、解构"
    # again: nothing new -- only the list is read
    site.calls.clear()
    memes.sync(fetch=site, list_pause=0, detail_pause=0)
    assert all(u == memes.WIKI_API for u in site.calls)
    # one changed, one new: two details
    site.entries[1]["updated_at"] = "2026-11-10T00:00:00+00:00"
    site.entries.append({"title": "班味", "slug": "班味", "tags": [], "aliases": [], "summary": "上班的疲惫感",
                         "updated_at": "2026-11-10T00:00:00+00:00"})
    site.calls.clear()
    memes.sync(fetch=site, list_pause=0, detail_pause=0)
    assert sorted(u.rsplit("/", 1)[1] for u in site.calls if u != memes.WIKI_API) == ["典", "班味"]
    assert memes.wiki_info()["count"] == 7


def test_a_stopped_download_goes_on_where_it_stopped(data_dir):
    from ai import memes
    site = Site(ENTRIES)

    def stopping(url, params):
        if url != memes.WIKI_API and len([u for u in site.calls if u != memes.WIKI_API]) >= 2:
            memes.stop()
        return site(url, params)
    memes.sync(fetch=stopping, list_pause=0, detail_pause=0)
    assert memes.wiki_info()["count"] == 6 and memes.wiki_info()["full"] < 6
    memes._stop.clear()
    memes.sync(fetch=site, list_pause=0, detail_pause=0)
    assert memes.wiki_info()["full"] == 6


def test_a_file_of_the_plugin_is_imported_and_later_updates_keep_its_text(data_dir, tmp_path):
    from ai import memes
    src = tmp_path / "plugin.db"
    conn = sqlite3.connect(src)
    conn.executescript("CREATE TABLE memes (title TEXT PRIMARY KEY, slug TEXT, tags TEXT, usage TEXT, origin TEXT, "
                       "other TEXT, summary TEXT, aliases TEXT); CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);")
    conn.execute("INSERT INTO memes VALUES ('去码头整点薯条', '去码头整点薯条', '解构', '谈理想时回一句', '海鸥漫画', '', "
                 "'摘要', 'kskbl')")
    conn.commit()
    conn.close()
    out = memes.import_file(src)
    assert out["added"] == 1 and out["count"] == 1 and out["full"] == 1
    site = Site(ENTRIES)
    memes.sync(fetch=site, list_pause=0, detail_pause=0)
    details = [u.rsplit("/", 1)[1] for u in site.calls if u != memes.WIKI_API]
    assert "去码头整点薯条" not in details and len(details) == 5        # the imported one keeps its text
    row = memes.lookup(None, "kskbl")[0]                                 # its alias too (the site has another)
    assert row["title"] == "去码头整点薯条" and row["usage"] == "谈理想时回一句"
    with pytest.raises(ValueError):
        bad = tmp_path / "bad.db"
        sqlite3.connect(bad).execute("CREATE TABLE x (a)").connection.commit()
        memes.import_file(bad)


def test_scan_finds_whole_titles_but_not_short_or_plain_words(violetdb, wiki):
    from ai import memes
    titles = lambda text: [m["title"] for m in memes.scan(violetdb, text)]     # noqa: E731
    assert titles("今天开会又在画饼，我只想去码头整点薯条") == ["去码头整点薯条"]
    assert titles("码头薯条走起") == ["去码头整点薯条"]                         # an alias
    assert titles("还得吃饭呢") == [] and titles("得吃！") == ["得吃"]        # 2 characters: only as the whole message
    assert titles("这个demo写完了") == [] and titles("今天有点emo了") == ["emo"]   # ASCII: a whole word
    assert titles("给你发个表情包") == []                                   # an ordinary word
    assert titles("典") == []                                               # 1 character never
    assert titles("我家哈基米又去码头整点薯条了") == ["去码头整点薯条", "哈基米"]   # longest first, at most 2


def test_the_memes_she_was_taught_come_first_and_belong_to_one_persona(violetdb, wiki):
    from ai import memes, persona_templates
    out = json.loads(memes.tool_remember(violetdb, {"phrase": "开摆", "meaning": "我们说开摆就是今天不学了，一起打游戏"}))
    assert out["ok"] is True
    found = memes.scan(violetdb, "开摆！去码头整点薯条")
    assert [f["kind"] for f in found] == ["own", "wiki"] and "一起打游戏" in memes.describe(found[0])
    json.loads(memes.tool_remember(violetdb, {"phrase": "开摆", "meaning": "改了：今天什么都不干"}))
    assert len(memes.own_list(violetdb)) == 1 and memes.own_list(violetdb)[0]["meaning"] == "改了：今天什么都不干"
    assert json.loads(memes.tool_remember(violetdb, {"phrase": "开摆", "meaning": " "}))["ok"] is False
    persona_templates.clear_memory(violetdb)                                 # cleared with her memories
    assert memes.own_list(violetdb) == []


def test_look_up_finds_by_alias_prefix_and_contained_title_and_says_when_it_does_not_know(violetdb, wiki):
    from ai import memes
    assert memes.lookup(violetdb, "hajimi")[0]["title"] == "哈基米"
    assert memes.lookup(violetdb, "去码头")[0]["title"] == "去码头整点薯条"
    assert memes.lookup(violetdb, "去码头整点薯条是什么梗")[0]["title"] == "去码头整点薯条"
    assert memes.lookup(violetdb, "海鸥")[0]["title"] == "去码头整点薯条"      # keywords
    out = json.loads(memes.tool_look_up(violetdb, {"query": "蚌埠住了"}))
    assert out["found"] == [] and "别硬解释" in out["message"]
    hit = json.loads(memes.tool_look_up(violetdb, {"query": "去码头整点薯条"}))
    assert "怎么用" in hit["found"][0] and "别人谈理想时回一句" in hit["found"][0]


def test_her_context_has_the_meme_only_when_the_message_uses_one(violetdb, wiki):
    from ai import memes
    from ai.context_engine import ContextEngine
    with_meme = ContextEngine(violetdb).build("chat", "今天开会好累，只想去码头整点薯条")
    text = "\n".join(with_meme.sections)
    assert memes.CONTEXT_HEADER in text and "去码头整点薯条" in text and "别科普" in text
    assert any(memes.CONTEXT_HEADER in v for v in with_meme.volatile)        # after the history: the cache stays
    plain = "\n".join(ContextEngine(violetdb).build("chat", "今天开会好累").sections)
    assert memes.CONTEXT_HEADER not in plain


def test_the_switch_turns_it_all_off(violetdb, wiki, monkeypatch):
    from ai import memes
    from ai.context_engine import ContextEngine
    monkeypatch.setenv("VIOLET_MEMES", "0")
    assert memes.scan(violetdb, "去码头整点薯条") == [] and not memes.wiki_available()
    assert memes.CONTEXT_HEADER not in "\n".join(ContextEngine(violetdb).build("chat", "去码头整点薯条").sections)


def test_her_tools_are_offered_when_they_can_do_something_and_never_in_her_own_messages(violetdb, data_dir, monkeypatch):
    from types import SimpleNamespace
    from ai.agents.base_agent import _CALL_CONTEXT
    from ai.agents.chat_agent import ChatAgent
    agent = ChatAgent.__new__(ChatAgent)
    agent._config = SimpleNamespace(mcp_servers=[])
    tools = [{"type": "function", "function": {"name": n}} for n in ("look_up_meme", "remember_meme", "query_memory")]

    def names(ctx):
        token = _CALL_CONTEXT.set(ctx)
        try:
            return {t["function"]["name"] for t in agent._filter_tools_for_run(tools)}
        finally:
            _CALL_CONTEXT.reset(token)
    assert names({}) >= {"remember_meme"} and "look_up_meme" not in names({})     # no wiki, nothing taught yet
    from ai import memes
    memes.own_save(violetdb, "开摆", "今天不学了")
    assert {"look_up_meme", "remember_meme"} <= names({})
    assert not names({"is_proactive": True}) & memes.TOOL_NAMES
    monkeypatch.setenv("VIOLET_MEMES", "0")
    assert not names({}) & memes.TOOL_NAMES


@pytest.fixture()
def admin(data_dir, violetdb):
    import ai.control_config as cc
    cc._ORIGINALS.clear()
    cc._DEFAULT_STORE = None
    from wx_ilink_py import ILinkClient, create_app
    app = create_app(client=ILinkClient(data_dir=data_dir, auto_login=False), violetdb=violetdb, proactive_scheduler=None,
                     all_agents=[], admin_token=TOKEN)
    app.testing = True
    c = app.test_client()
    assert c.post("/auth/login", data={"token": TOKEN}, headers=SAME).status_code == 302
    yield c
    cc._ORIGINALS.clear()
    cc._DEFAULT_STORE = None


def test_the_page_imports_a_file_lists_and_edits_your_memes(admin, data_dir, tmp_path):
    from ai import memes
    d = admin.get("/api/control/memes").get_json()["data"]
    assert d["wiki"]["installed"] is False and d["own"] == [] and d["enabled"] is True
    src = tmp_path / "src"
    os.environ["VIOLET_DATA_DIR"] = str(src)
    memes.sync(fetch=Site(ENTRIES), list_pause=0, detail_pause=0)
    raw = memes.wiki_path().read_bytes()
    os.environ["VIOLET_DATA_DIR"] = str(data_dir)
    r = admin.post("/api/control/memes/import", data=raw, headers={**SAME, "Content-Type": "application/octet-stream"})
    assert r.status_code == 200 and r.get_json()["data"]["count"] == 6
    assert admin.post("/api/control/memes/import", data=b"not a db", headers=SAME).status_code == 400
    found = admin.get("/api/control/memes/lookup?q=hajimi").get_json()["data"]["found"]
    assert found[0]["title"] == "哈基米"
    assert admin.post("/api/control/memes/download", json={}, headers=SAME).status_code == 400     # must confirm
    new = admin.post("/api/control/memes/own", json={"phrase": "开摆", "meaning": "今天不学了"}, headers=SAME).get_json()["data"]
    assert admin.post(f"/api/control/memes/own/{new['id']}", json={"meaning": "今天什么都不干"}, headers=SAME).status_code == 200
    assert admin.get("/api/control/memes").get_json()["data"]["own"][0]["meaning"] == "今天什么都不干"
    assert admin.delete(f"/api/control/memes/own/{new['id']}", headers=SAME).status_code == 200
    assert admin.delete("/api/control/memes/wiki", headers=SAME).get_json()["data"]["deleted"] is True
    assert not memes.wiki_path().is_file()


def test_the_page_has_the_card_and_the_settings_group():
    from pathlib import Path
    from ai.control_config import BY_KEY, GROUPS
    html = (Path(__file__).resolve().parent.parent / "wx_ilink_py" / "templates" / "control.html").read_text(encoding="utf-8")
    assert 'id="tools-memes"' in html and 'data-groups="memes"' in html and "control_memes.js" in html
    assert BY_KEY["VIOLET_MEMES"].group == "memes" and "memes" in GROUPS
