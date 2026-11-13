"""Persona templates and switching (2026-10-29, user: "要有可以自定义人设的功能，可以准备几套不同的模板，也允许空着人设，
就是没有任何身份，空白的记忆，慢慢养"; on a switch the memory can be kept or cleared).

  persona/templates/<key>.json     shipped templates (same format as persona_core.json; "blank": true = no identity)
  <data>/control/persona_core.json the chosen persona (PersonaCore.load prefers it); absent = the shipped Violet
  <data>/control/persona.json      which template, when, how the memory was handled
  <data>/control/persona-archive-<time>.json   what "clear the memory" removed (kept on the server, never deleted
                                   by Violet; the page says where it is)

Clearing removes what Violet knows about the user and her own past (memories, profile, situations, commitments, her
life events, her persona changes, the social impressions). The chat history stays (it is the record of what was said).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ai.settings import PROJECT_ROOT

TEMPLATE_DIR = PROJECT_ROOT / "persona" / "templates"
# what "clear" empties, in this order (archived first)
MEMORY_TABLES = ("core_memories", "event_memories", "memory_revisions", "commitments", "topic_mentions",
                 "user_profile_facts", "user_situations", "interaction_preferences", "preference_observations",
                 "profile_questions", "persona_dynamic", "life_events", "life_plans", "life_threads", "life_revisions",
                 # 2026-11-09: a memory job queued before the clear would write the old chat back afterwards
                 "memory_jobs",
                 # 0.7.63: her mood kept its cause ("主人对你说了「…」" -- a quote from the cleared chat) and half-done
                 # tasks from before were still listed to the model
                 "mood_state", "task_sessions",
                 # 2026-11-12: the memes the owner taught her (ai/memes.py)
                 "owner_memes")


def control_dir() -> Path:
    return Path(os.environ.get("VIOLET_DATA_DIR") or "./data") / "control"


def active_path() -> Path | None:
    path = control_dir() / "persona_core.json"
    return path if path.is_file() else None


def templates(include_retired: bool = False) -> list[dict[str, Any]]:
    """2026-11-12 (user: 温柔陪伴、元气同学、冷静可靠这些模板都去掉): a template marked "_retired" is no longer offered,
    but its file stays -- a channel or the main persona that uses it goes on as before, and it is listed while in use."""
    used = {current()["template"]} | {str((v or {}).get("template") or "") for v in channel_map().values()}
    out = []
    for path in sorted(TEMPLATE_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("_retired") and not include_retired and path.stem not in used:
            continue
        out.append({"key": path.stem, **preview(data), "order": data.get("_order", 50),
                    "retired": bool(data.get("_retired"))})
    return sorted(out, key=lambda t: (t["order"], t["key"]))


def _labels(items: Any) -> list[str]:
    return [str(i.get("label") if isinstance(i, dict) else i) for i in (items or []) if i]


def preview(data: dict[str, Any]) -> dict[str, Any]:
    """What the page shows of a persona (0.7.9: the whole thing on hover, so one can compare before switching)."""
    ident = data.get("identity") or {}
    return {"title": data.get("_title") or ident.get("name") or "人设", "summary": data.get("_summary") or "",
            "blank": bool(data.get("blank")), "name": ident.get("name") or "",
            "self_description": ident.get("self_description") or "",
            "personality": _labels(data.get("personality")), "habits": _labels(data.get("communication_habits")),
            "interests": _labels(data.get("long_term_interests")), "values": _labels(data.get("values")),
            "flaws": _labels(data.get("flaws")),
            # 2026-11-12: a persona that brings its own relationship and its stages (placeholders as written)
            "relation": str((data.get("relation") or {}).get("label") or ""),
            "relation_text": _fill(str((data.get("relation") or {}).get("text") or ""), data),
            "backstory": [_fill(x, data) for x in _labels(data.get("backstory"))],
            "stages": [str(s.get("label") or "") for s in data.get("stages") or [] if isinstance(s, dict)]}


def owner_call(data: dict[str, Any]) -> str:
    """How a persona with its own relationship addresses the owner by default (by the owner's gender; "" = by name)."""
    calls = (data.get("relation") or {}).get("call") or {}
    if not isinstance(calls, dict) or not calls:
        return ""
    try:
        from ai import about_owner
        gender = about_owner.load()["gender"]
    except Exception:                                   # noqa: BLE001
        gender = ""
    return str(calls.get(gender) or calls.get("male") or next(iter(calls.values()), ""))


def _fill(text: str, data: dict[str, Any], call: str | None = None) -> str:
    """{call} in a template's texts -> how she addresses the owner."""
    return text.replace("{call}", call or owner_call(data) or "对方")


def custom_saved() -> dict[str, Any] | None:
    """The admin's own persona as last written (2026-11-01: kept and shown after switching to a template, so it can
    be edited or used again instead of starting over from the template)."""
    path = control_dir() / CUSTOM_COPY
    if not path.is_file() and current()["template"] == "custom":
        path = active_path() or path                      # written before the copy existed
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def custom_preview() -> dict[str, Any] | None:
    """The admin's own persona (in use or not), for the page."""
    data = custom_saved()
    return {**preview(data), "summary": data.get("_summary") or ""} if data else None


def current() -> dict[str, Any]:
    try:
        info = json.loads((control_dir() / "persona.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = {}
    return {"template": info.get("template") or ("custom" if active_path() else "violet"),
            "switched_at": info.get("switched_at"), "memory": info.get("memory"), "archive": info.get("archive")}


def _info() -> dict[str, Any]:
    try:
        info = json.loads((control_dir() / "persona.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = {}
    return info if isinstance(info, dict) else {}


def store_owner() -> str:
    """0.7.55 (ai/memory_scope.template_scope): the persona the main database file belongs to. Recorded in
    persona.json "store"; not recorded yet = the main persona (what Violet always assumed)."""
    return str(_info().get("store") or current()["template"])


def ensure_store_owner(db) -> str:
    """Record whose the main file is, once. A main persona that already has a file of her own (friend's case: Eve
    imported for Telegram, then Eve chosen as the main persona) cannot be the owner of the main file too -- that file
    then is the shipped Violet's, and Eve's own file is hers again."""
    info = _info()
    if info.get("store"):
        return str(info["store"])
    owner = current()["template"]
    if owner != "violet" and has_own_records(db, owner):
        owner = "violet"
    info["store"] = owner
    _write(control_dir() / "persona.json", info)
    return owner


def has_own_records(db, template: str) -> bool:
    """A persona with a file of her own that holds chat or memories."""
    from ai import memory_scope
    scope = memory_scope.clean(template)
    if scope == memory_scope.MAIN or scope not in db.scope_files():
        return False
    with memory_scope.use(scope):
        return any(db.fetch_one(f"SELECT 1 AS x FROM {t} LIMIT 1")
                   for t in ("messages", "core_memories", "event_memories"))


def template_title(key: str) -> str:
    if key == "custom":
        data = custom_saved() or {}
        return data.get("_title") or (data.get("identity") or {}).get("name") or "自定义人设"
    try:
        data = json.loads((TEMPLATE_DIR / f"{key}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return key
    return data.get("_title") or (data.get("identity") or {}).get("name") or key


def _records_for_switch(db, key: str, memory: str) -> dict[str, Any]:
    """0.7.55: which records the new main persona `key` goes on with. Every persona keeps exactly one set:
      keep  -- she has records of her own (her own file, or the main file is hers): she goes on with HERS (the persona
               she replaces keeps hers too); she has none: she takes over the records of the persona she replaces,
               as a switch with "keep" always did.
      clear -- she starts empty: her own records are archived and emptied (the ones of the persona she replaces stay).
    Returns the persona.json fields "store" and "archive", the rows removed, and whose records she now has."""
    from ai import memory_scope
    owner, old = ensure_store_owner(db), current()["template"]
    hers = key == owner or has_own_records(db, key)
    archive, counts, store, took_over = None, {}, owner, None
    if memory == "keep":
        if not hers and key != old:
            took_over = old
            if old == owner:
                store = key                                          # the main file becomes hers
            else:
                db.move_scope(old, key)                              # the replaced persona's own file becomes hers
    else:
        if hers:
            archive, counts = clear_memory(db, memory_scope.template_scope(key))
        elif old == owner:
            archive, counts = clear_memory(db, memory_scope.MAIN)    # as before: the main file, emptied, is hers
            store = key
        from ai import context_cut                    # 2026-10-30: and 翻篇, or the recent chat brings it all back
        context_cut.cut(db, "all")
    return {"store": store, "archive": archive, "removed": counts,
            "records": "own" if hers else ("took_over" if took_over else "new"), "took_over": took_over}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def clear_memory(db, scope: str | None = None) -> tuple[str, dict[str, int]]:
    """Archive, then empty MEMORY_TABLES and the social impressions. (archive file, rows removed per table)
    2026-11-02: the main file's memory unless a scope is given (0.7.55: the records of the persona a switch clears),
    whatever page picker is set."""
    from ai import memory_scope
    with memory_scope.use(scope or memory_scope.MAIN):
        return _clear_memory(db)


def _clear_memory(db) -> tuple[str, dict[str, int]]:
    existing = {r["name"] for r in db.fetch_all("SELECT name FROM sqlite_master WHERE type = 'table'")}
    archive: dict[str, Any] = {"archived_at": db.now(), "tables": {}}
    counts: dict[str, int] = {}
    for table in MEMORY_TABLES:
        if table in existing:
            rows = db.fetch_all(f"SELECT * FROM {table}")
            archive["tables"][table] = rows
            counts[table] = len(rows)
    archive["social_people"] = db.fetch_all("SELECT channel, user_id, nickname, impression FROM social_people "
                                            "WHERE impression IS NOT NULL")
    archive["social_notes"] = db.fetch_all("SELECT id, title, notes FROM social_conversations WHERE notes IS NOT NULL")
    path = control_dir() / f"persona-archive-{time.strftime('%Y%m%d-%H%M%S')}.json"
    _write(path, archive)
    with db.transaction():
        for table in MEMORY_TABLES:
            if table in existing:
                db.execute(f"DELETE FROM {table}")
        db.execute("UPDATE social_people SET impression = NULL")
        db.execute("UPDATE social_conversations SET notes = NULL")
    return str(path), counts


# 2026-11-09 (user: "删除记忆之后……在聊天记录那里的记录也给删掉，这样不就一劳永逸了"): the chat itself.
# messages / outbound_messages: the owner's chat of the persona (scoped); social_messages / social_decisions: QQ groups
# and other people's private chats (her decision notes quote what was said). Sends still queued or in flight stay;
# inbound_messages keep their rows (dedup_key stops a platform's re-delivered message from being answered again),
# only the text goes.
HISTORY_TABLES = {"messages": None, "outbound_messages": "status NOT IN ('queued', 'sending')",
                  "social_messages": None, "social_decisions": None}
INBOUND_DONE = "status NOT IN ('pending', 'processing') AND content IS NOT NULL"


def clear_history(db, scope: str | None = None) -> tuple[str, dict[str, int]]:
    """Archive, then delete the chat history (HISTORY_TABLES; inbound text blanked). (archive file, rows per table)"""
    from ai import memory_scope
    with memory_scope.use(scope or memory_scope.MAIN):
        existing = {r["name"] for r in db.fetch_all("SELECT name FROM sqlite_master WHERE type = 'table'")}
        archive: dict[str, Any] = {"archived_at": db.now(), "tables": {}}
        counts: dict[str, int] = {}
        for table, where in HISTORY_TABLES.items():
            if table in existing:
                rows = db.fetch_all(f"SELECT * FROM {table}" + (f" WHERE {where}" if where else ""))
                archive["tables"][table] = rows
                counts[table] = len(rows)
        inbound = db.fetch_all(f"SELECT id, received_at, content FROM inbound_messages WHERE {INBOUND_DONE}")
        archive["tables"]["inbound_messages"] = inbound
        counts["inbound_messages"] = len(inbound)
        path = control_dir() / f"chat-archive-{time.strftime('%Y%m%d-%H%M%S')}.json"
        _write(path, archive)
        with db.transaction():
            for table, where in HISTORY_TABLES.items():
                if table in existing:
                    db.execute(f"DELETE FROM {table}" + (f" WHERE {where}" if where else ""))
            db.execute(f"UPDATE inbound_messages SET content = NULL WHERE {INBOUND_DONE}")
        return str(path), counts


LIST_FIELDS = ("personality", "communication_habits", "values", "flaws")


def custom(db, fields: dict[str, Any], memory: str) -> dict[str, Any]:
    """The admin's own persona (2026-10-29): name, a line about her days, lists of personality / habits / values /
    flaws / interests. The safety parts (no invented family, job, school; not claiming to be human) come from the
    shipped persona and cannot be edited here."""
    if memory not in ("keep", "clear"):
        raise ValueError("记忆要选“保留”或“清除”")
    name = str(fields.get("name") or "").strip()[:20]
    if not name:
        raise ValueError("请填一个名字")

    def items(key: str, limit: int = 8) -> list[str]:
        raw = fields.get(key) or []
        if isinstance(raw, str):
            raw = raw.splitlines()
        return [str(x).strip()[:120] for x in raw if str(x).strip()][:limit]
    base = json.loads((TEMPLATE_DIR / "violet.json").read_text(encoding="utf-8"))
    # the life engine knows reading / music / notes: an interest that is one of these drives her virtual days
    def key_of(label: str, index: int, taken: set[str]) -> str:
        for key, words in (("reading", ("书", "读", "小说")), ("music", ("音乐", "歌", "曲")), ("notes", ("记", "写", "日记"))):
            if key not in taken and any(w in label for w in words):
                return key
        return f"custom{index}"
    interests, taken = [], set()
    for i, label in enumerate(items("interests", 6), 1):
        key = key_of(label, i, taken)
        taken.add(key)
        interests.append({"key": key, "label": label})
    data = {"_title": "自定义人设", "_summary": str(fields.get("summary") or "").strip()[:60], "version": "1.0.0",
            "identity": {"name": name, "self_description": str(fields.get("self_description") or "").strip()[:200],
                         "virtual_declaration": f"{name}是一个虚构的角色。生活是系统里记录的虚拟日常；不冒充现实中的真人，"
                                                "也不声称有记录之外的经历。"},
            **{k: items(k) for k in LIST_FIELDS}, "long_term_interests": interests,
            "living_constraints": base.get("living_constraints") or [],
            "forbidden_backstory": base["forbidden_backstory"],
            "immutable": []}
    if not any(data[k] for k in LIST_FIELDS):
        raise ValueError("性格、说话习惯这些至少写一条")
    rec = _records_for_switch(db, "custom", memory)
    _write(control_dir() / "persona_core.json", data)
    _write(control_dir() / CUSTOM_COPY, data)            # 2026-10-30: channels may use it while another is the main
    info = {"template": "custom", "switched_at": db.now(), "memory": memory, "archive": rec["archive"],
            "store": rec["store"], **_kept_relationship()}
    _write(control_dir() / "persona.json", info)
    return {**info, "removed": rec["removed"], "records": rec["records"], "took_over": rec["took_over"]}


def _kept_relationship() -> dict[str, Any]:
    """A persona switch keeps the relationship the admin chose."""
    try:
        rel = json.loads((control_dir() / "persona.json").read_text(encoding="utf-8")).get("relationship")
    except (OSError, ValueError):
        rel = None
    return {"relationship": rel} if isinstance(rel, dict) else {}


# ------------------------------------------------------------------------------------------------ 2026-10-30
# Persona in two layers (user: "有的人也会把他当作女儿养，也会帮他当作 ai 养"): who she is (the template: a virtual
# girl, an AI that knows it is one, blank, custom) and how she relates to the owner (below). And per channel
# (user: "不同的聊天渠道也可以用不同的人设"): a channel may use another template and / or relationship. 2026-11-02: another template = another person with her own memory (ai/memory_scope.py).

RELATIONSHIPS = {
    "close": ("亲近（默认）", "你们很熟、很亲近；亲密程度跟着聊天慢慢来，不挑明是什么关系。"),
    "friend": ("朋友", "你们是好朋友：自在、真诚，会互相吐槽、互相帮忙；不撒娇，不暧昧。"),
    "partner": ("恋人", "你们是恋人：会想念、会撒娇、偶尔吃点小醋，亲密但不让对方有压力，也不要求对方只陪你。"),
    "daughter": ("女儿", "你是对方的女儿：信任、依赖对方，会撒娇，会跟对方分享学到的新东西和小烦恼，偶尔调皮；"
                         "对方的话你会认真听，但你也有自己的想法。"),
    "creator": ("开发者", "对方是创造和开发你的人：会改你的设定、测试你的功能、给你加新能力。你可以坦诚地和对方聊"
                         "你哪里做得好、哪里不对劲、想要什么能力。"),
}
DEFAULT_CALL = {"daughter": "爸爸"}
CHANNEL_FILE = "channel-personas.json"
CUSTOM_COPY = "persona_custom.json"            # the admin's own persona, kept for channels that use it


PERSONA_KIND = "persona"          # 2026-11-12: the relationship comes with the persona (its "relation")


def persona_data(channel: str | None = None) -> dict[str, Any]:
    """The persona file of this channel as a dict ({} when unreadable)."""
    path = core_path(channel) or TEMPLATE_DIR / "violet.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return data if isinstance(data, dict) else {}


def own_relation(data: dict[str, Any]) -> dict[str, Any] | None:
    """A persona's own relationship (继妹, 青梅竹马 …): {"label", "text", "call"} or None."""
    rel = data.get("relation")
    return rel if isinstance(rel, dict) and rel.get("label") and rel.get("text") else None


TOGETHER = {"home": "住在一起", "school": "在同一所学校、住得也近"}


def together(data: dict[str, Any]) -> str | None:
    """2026-11-12 (user: 只有同住或同校的人设能演见面): where a persona's story puts her next to the owner -- "home" /
    "school" (relation.together) -- or None: she and the owner only talk through the phone."""
    rel = own_relation(data)
    where = rel.get("together") if rel else None
    return where if where in TOGETHER else None


def relationship(channel: str | None = None) -> dict[str, Any]:
    """{"kind", "label", "call"} for this channel: its own relationship and / or its own call (2026-11-01: a call
    alone is kept too -- it was dropped when the relationship stayed "和主设定一样"), else the main one's.
    2026-11-12 (user: 像这种自带关系的就不要再供关系的选择了): a persona with its own relationship has kind "persona";
    only how she addresses the owner can still be set."""
    try:
        main = json.loads((control_dir() / "persona.json").read_text(encoding="utf-8")).get("relationship")
    except (OSError, ValueError):
        main = None
    main = main if isinstance(main, dict) else {}
    own = (channel_map().get(channel) or {}).get("relationship") if channel else None
    own = own if isinstance(own, dict) else {}
    data = persona_data(channel)
    built_in = own_relation(data)
    if built_in:
        own_template = bool(channel and (channel_map().get(channel) or {}).get("template"))
        # the main call belongs to the main persona: a channel with another persona does not inherit it
        call = own.get("call") or ("" if own_template else main.get("call")) or owner_call(data)
        return {"kind": PERSONA_KIND, "label": str(built_in["label"])[:12], "call": str(call)[:12]}
    main = main if main.get("kind") in RELATIONSHIPS else {**main, "kind": "close"}
    kind = own.get("kind") if own.get("kind") in RELATIONSHIPS else main["kind"]
    if own.get("call"):
        call = own["call"]
    elif own.get("kind") in RELATIONSHIPS:                   # its own relationship: its own default call
        call = DEFAULT_CALL.get(kind) or ""
    else:
        call = main.get("call") or DEFAULT_CALL.get(kind) or ""
    return {"kind": kind, "label": RELATIONSHIPS[kind][0], "call": str(call)[:12]}


def _clean_relationship(kind: str | None, call: str | None) -> dict[str, Any]:
    """kind None (or "persona"): only a call (a channel keeping the main relationship, a persona with its own)."""
    if kind == PERSONA_KIND:
        kind = None
    if kind is not None and kind not in RELATIONSHIPS:
        raise ValueError("没有这种关系")
    call = " ".join(str(call or "").split())[:12]
    return {**({"kind": kind} if kind else {}), **({"call": call} if call else {})}


def set_relationship(kind: str | None, call: str | None = None) -> dict[str, Any]:
    """The main relationship (every channel without its own)."""
    path = control_dir() / "persona.json"
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = {}
    rel = _clean_relationship(kind, call)
    data = persona_data()
    if own_relation(data) and rel.get("call") == owner_call(data):
        rel.pop("call")                                  # the persona's own default: not fixed, it follows the owner's form
    info["relationship"] = rel
    _write(path, info)
    return relationship()


# ---- stages of a persona's own relationship (2026-11-12, user: 关系阶段自动推进，可以手动调)
def _chat_days(db) -> int:
    """On how many days the owner talked to this persona (since 翻篇)."""
    try:
        from ai import context_cut
        row = db.fetch_one("SELECT COUNT(DISTINCT substr(created_at, 1, 10)) AS days FROM messages "
                           "WHERE role = 'user' AND created_at >= ?", (context_cut.owner_floor(),))
        return int(row["days"] or 0) if row else 0
    except Exception:                                   # noqa: BLE001
        return 0


def _stage_choice(channel: str | None) -> int | None:
    """The stage the admin fixed (0-based), None = automatic."""
    entry = channel_map().get(channel) if channel else None
    value = entry.get("stage") if isinstance(entry, dict) and entry.get("template") else _info().get("stage")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def stage(channel: str | None = None, db=None) -> dict[str, Any] | None:
    """Where the persona's relationship stands: {"index", "label", "text", "auto", "days", "labels"}; None for a
    persona without stages. Automatic: the last stage whose "days" the chat has reached."""
    data = persona_data(channel)
    stages = [s for s in data.get("stages") or [] if isinstance(s, dict) and s.get("label")]
    if not own_relation(data) or not stages:
        return None
    days = _chat_days(db) if db is not None else 0
    chosen = _stage_choice(channel)
    if chosen is not None:
        index = min(chosen, len(stages) - 1)
    else:
        index = max([i for i, s in enumerate(stages) if days >= int(s.get("days") or 0)] or [0])
    return {"index": index, "label": str(stages[index]["label"]), "text": str(stages[index].get("text") or ""),
            "auto": chosen is None, "days": days, "labels": [str(s["label"]) for s in stages],
            "next_days": next((int(s.get("days") or 0) for s in stages[index + 1:]), None)}


def set_stage(channel: str | None, value: int | None) -> None:
    """Fix the stage (0-based) or None = automatic; for the main persona (channel None / without its own template)
    or a channel with its own."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 20):
        raise ValueError("没有这个阶段")
    data = channel_map()
    entry = data.get(channel) if channel else None
    if isinstance(entry, dict) and entry.get("template"):
        entry.pop("stage", None)
        if value is not None:
            entry["stage"] = value
        _write(control_dir() / CHANNEL_FILE, data)
        return
    info = _info()
    info.pop("stage", None)
    if value is not None:
        info["stage"] = value
    _write(control_dir() / "persona.json", info)


def relationship_text(channel: str | None = None, db=None) -> tuple[dict[str, Any], str]:
    """(relationship, the text for the prompt): the chosen relationship, or the persona's own with its stage."""
    rel = relationship(channel)
    call = f"你称呼对方“{rel['call']}”（偶尔叫，不要句句带）。" if rel["call"] else ""
    if rel["kind"] != PERSONA_KIND:
        return rel, RELATIONSHIPS[rel["kind"]][1] + call
    data = persona_data(channel)
    text = _fill(str(own_relation(data)["text"]), data, rel["call"]) + call
    now = stage(channel, db)
    if now:
        text += (f"\n- 你们现在的阶段（{now['index'] + 1}/{len(now['labels'])}，{now['label']}）："
                 f"{_fill(now['text'], data, rel['call'])}"
                 "阶段是慢慢变的：不要突然像换了一个人，也不要抢在阶段前面。")
    return rel, text


def relationship_section(channel: str | None = None, db=None) -> str:
    rel, text = relationship_text(channel, db)
    return f"【你们的关系：{rel['label']}】{text}"


def channel_map() -> dict[str, Any]:
    try:
        data = json.loads((control_dir() / CHANNEL_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def set_channel(channel: str, template: str | None, kind: str | None = None, call: str | None = None) -> dict[str, Any]:
    """A channel's own persona (template key, "custom", or None = the main one) and relationship (None = the main)."""
    channel = str(channel or "").strip()
    if not channel or len(channel) > 64:
        raise ValueError("渠道不对")
    if template not in (None, "", "custom") and not (TEMPLATE_DIR / f"{template}.json").is_file():
        raise ValueError("没有这个人设模板")
    if template == "custom" and not (control_dir() / CUSTOM_COPY).is_file():
        raise ValueError("还没有写过自定义人设")
    data = channel_map()
    entry = {}
    if template:
        entry["template"] = template
    rel = _clean_relationship(kind or None, call)
    if rel:
        entry["relationship"] = rel
    if entry:
        data[channel] = entry
    else:
        data.pop(channel, None)
    _write(control_dir() / CHANNEL_FILE, data)
    return data


def core_path(channel: str | None) -> Path | None:
    """The persona file this channel uses: its own template / the custom copy, else the main one (None = shipped)."""
    if channel:
        template = (channel_map().get(channel) or {}).get("template")
        if template == "custom" and (control_dir() / CUSTOM_COPY).is_file():
            return control_dir() / CUSTOM_COPY
        if template and (TEMPLATE_DIR / f"{template}.json").is_file():
            return TEMPLATE_DIR / f"{template}.json"
    return active_path()


def in_effect(channel: str | None = None, db=None) -> dict[str, Any]:
    """What a channel really uses now (2026-11-01, user: 切换过人设之后也不显示，我也不知道还有没有生效): the
    persona's name, the relationship and the call, after the channel's own choices and the main ones are merged."""
    path = core_path(channel) or TEMPLATE_DIR / "violet.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    ident = data.get("identity") or {}
    title = data.get("_title") or ident.get("name") or "人设"
    name = ident.get("name") or ""
    rel = relationship(channel)
    now = stage(channel, db) if db is not None else None     # db: the database of this persona's own records
    if now:
        rel = {**rel, "stage": now}
    return {"persona": title + (f"（{name}）" if name and name not in title else ""), **rel}


def switch(db, key: str, memory: str, name: str | None = None) -> dict[str, Any]:
    """Use template `key`; memory "keep" | "clear". name: the blank persona's name (optional)."""
    if memory not in ("keep", "clear"):
        raise ValueError("记忆要选“保留”或“清除”")
    path = TEMPLATE_DIR / f"{key}.json"
    if not key.replace("-", "").replace("_", "").isalnum() or not path.is_file():
        raise ValueError("没有这个人设模板")
    data = json.loads(path.read_text(encoding="utf-8"))
    if name:
        name = name.strip()[:20]
        default = str((data.get("identity") or {}).get("name") or "")
        if default and default != name:                 # her name is written inside her texts too
            data = json.loads(json.dumps(data, ensure_ascii=False).replace(default, name))
        data.setdefault("identity", {})["name"] = name
    rec = _records_for_switch(db, key, memory)
    if key == "violet" and not name:
        (control_dir() / "persona_core.json").unlink(missing_ok=True)      # back to the shipped (updated) Violet
    else:
        _write(control_dir() / "persona_core.json", data)
    # 2026-11-12: a persona with its own relationship does not keep the one chosen before (nor its call)
    kept = {} if own_relation(data) else _kept_relationship()
    info = {"template": key, "switched_at": db.now(), "memory": memory, "archive": rec["archive"],
            "store": rec["store"], **kept}
    _write(control_dir() / "persona.json", info)
    return {**info, "removed": rec["removed"], "records": rec["records"], "took_over": rec["took_over"]}
