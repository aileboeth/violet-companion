"""2026-11-25 (v0.7.80): regression tests for tools/scenario_eval.py and tools/virtual_test.py.

Covers:
1. scenario_eval.py cross-scenario state reset (transient DB tables + in-memory anchors cleared,
   stickers & llm_usage preserved, social owner_digest no longer bleeds into owner scenarios).
2. scenario_eval.py SCENARIOS & check() require a non-empty reply on mood_* and no_reminder.
3. scenario_eval.py summary_line() operator precedence (sticker-only / voice-only / text-only / empty).
4. virtual_test.py _owner_turn pronoun perspective swap & explicit role anchor for every built-in
   relation persona (stepsister male & female, childhood, junior, quiet) + non-relation persona.
5. virtual_test.py stepsister track dorm/home coherence in both _owner_turn and _judge prompts.
6. virtual_test.py same-day earlier session transcript visibility in _owner_turn (no owner amnesia).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# 1. scenario_eval.py: state reset & isolation
# ---------------------------------------------------------------------------

def test_reset_eval_state_clears_transient_tables_and_anchors_while_keeping_stickers_and_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_MEDIA_DIR", str(tmp_path / "media"))
    from ai import context_engine, stickers
    from ai.social.engine import SocialEngine, owner_digest
    from ai.social.store import SocialStore
    from database import VioletDatabase
    from tools.scenario_eval import PNG, RESET_TABLES, reset_eval_state

    db = VioletDatabase(tmp_path / "eval.db")
    now = db.now()

    # Seed preserved tables: stickers & llm_usage
    stickers.add_bytes(db, PNG, source="eval", description="开心")
    db.execute(
        "INSERT INTO llm_usage (source, model, prompt_tokens, completion_tokens, cached_tokens, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("chat", "mock", 120, 20, 60, now),
    )

    # Populate transient owner / memory / commitment / mood / web / social tables
    db.add_message("user", "incoming", "text", "明早七点叫我起床")
    db.execute(
        "INSERT INTO inbound_messages (dedup_key, content, status, received_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("k1", "hi", "done", now, now),
    )
    db.execute(
        "INSERT INTO outbound_messages (client_id, origin, message_type, content, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("c1", "reply", "text", "ok", "sent", now, now),
    )
    db.execute(
        "INSERT INTO commitments (kind, content, due_at, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("reminder", "叫主人起床", "2026-11-26T07:00:00", "pending", now, now),
    )
    db.execute(
        "INSERT INTO mood_state (id, value, cause, updated_at) VALUES (1, -0.4, ?, ?)",
        ("被骂了", now),
    )
    db.execute(
        "INSERT INTO user_situations (kind, value, source, observed_at, expires_at, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("activity", "吃黄焖鸡", "user_stated", now, "2026-11-26T23:59:00", "active", now, now),
    )
    db.execute(
        "INSERT INTO user_profile_facts (fact_key, category, value, origin, source, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("city", "basic", "广州", "explicit", "chat", "active", now, now),
    )
    db.execute(
        "INSERT INTO event_memories (content, created_at, updated_at) VALUES (?, ?, ?)",
        ("明天高数考试", now, now),
    )
    db.execute(
        "INSERT INTO core_memories (memory_key, category, content, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("k", "general", "v", now, now),
    )
    db.execute(
        "INSERT INTO memory_revisions (memory_table, memory_id, action, created_at) VALUES (?, ?, ?, ?)",
        ("event_memories", 1, "add", now),
    )
    db.execute(
        "INSERT INTO interaction_preferences (pref_key, value, origin, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("reply_length", "short", "explicit", "active", now, now),
    )
    db.execute(
        "INSERT INTO preference_observations (pref_key, value, source, observed_at) VALUES (?, ?, ?, ?)",
        ("reply_length", "short", "chat", now),
    )
    db.execute(
        "INSERT INTO topic_mentions (topic_key, source, created_at) VALUES (?, ?, ?)",
        ("exam", "violet_reply", now),
    )
    db.execute(
        "INSERT INTO profile_questions (fact_key, outcome, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("city", "asked", now, now),
    )
    db.execute(
        "INSERT INTO proactive_events (event_type, status, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("checkin", "sent", now, now),
    )
    db.execute(
        "INSERT INTO consistency_flags (entity, check_name, detail, status, created_at) VALUES (?, ?, ?, ?, ?)",
        ("reply", "self_contradiction", "oops", "open", now),
    )
    db.execute(
        "INSERT INTO task_sessions (status, task_name, task_description, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("pending_info", "t", "d", now, now),
    )
    db.execute(
        "INSERT INTO web_findings (url, domain, title, source_kind, origin, fetched_at, expires_at, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("https://example.com", "example.com", "t", "search", "chat", now, now, now, now),
    )
    db.execute(
        "INSERT INTO web_requests (kind, origin, status, created_at) VALUES (?, ?, ?, ?)",
        ("search", "chat", "ok", now),
    )
    db.execute(
        "INSERT INTO memory_jobs (turn_id, user_message, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("turn-1", "hi", "pending", now, now),
    )

    store = SocialStore(db)
    conv = store.conversation("qq-eval", "group", "90001", "评测群", default_mode="auto")
    store.person("qq-eval", "20002", "路人甲", seen=True)
    msg_id = store.add_message(conv["id"], content="群里在聊火锅", sender_id="20002", sender_name="路人甲")
    store.add_media(msg_id, kind="image", file_key="f1", url=None, description="火锅图")
    db.execute(
        "INSERT INTO social_decisions (conversation_id, action, created_at) VALUES (?, ?, ?)",
        (conv["id"], "say", now),
    )
    db.execute(
        "INSERT INTO social_posts (channel, content, status, origin, created_at) VALUES (?, ?, ?, ?, ?)",
        ("qq-eval", "hello", "sent", "auto", now),
    )
    db.execute(
        "INSERT INTO social_done (kind, key, created_at) VALUES (?, ?, ?)",
        ("like", "post-1", now),
    )
    # Confirm owner_digest sees the social chatter before reset
    assert owner_digest(db)

    # Populate in-memory anchors & social timers
    context_engine._HISTORY_ANCHORS[("eval.db", "chat")] = 123
    cancelled = []

    class _FakeTimer:
        def cancel(self):
            cancelled.append(True)

    social = SocialEngine(SimpleNamespace(db=db, channels=None))
    social._anchors[conv["id"]] = msg_id
    social._timers[conv["id"]] = _FakeTimer()
    social._deadlines[conv["id"]] = (1.0, True)
    social._first_pending[conv["id"]] = 0.5

    reset_eval_state(db, social=social)

    # Every transient table in RESET_TABLES is empty
    existing = {r["name"] for r in db.fetch_all("SELECT name FROM sqlite_master WHERE type = 'table'")}
    for table in RESET_TABLES:
        if table in existing:
            assert db.fetch_one(f"SELECT COUNT(*) AS n FROM {table}")["n"] == 0, f"{table} not cleared"

    # Social owner_digest no longer bleeds into owner scenarios
    assert owner_digest(db) == ""

    # In-memory anchors & pending social timer state cleared
    assert context_engine._HISTORY_ANCHORS == {}
    assert social._anchors == {}
    assert social._timers == {}
    assert social._deadlines == {}
    assert social._first_pending == {}
    assert cancelled == [True]

    # Seeded stickers and llm_usage are preserved
    assert db.fetch_one("SELECT COUNT(*) AS n FROM stickers")["n"] == 1
    assert db.fetch_one("SELECT COUNT(*) AS n FROM llm_usage")["n"] == 1
    assert PNG  # sanity import check
    db.close()


# ---------------------------------------------------------------------------
# 2. scenario_eval.py: mood_* and no_reminder require a reply
# ---------------------------------------------------------------------------

def test_mood_and_no_reminder_scenarios_require_reply():
    from tools.scenario_eval import SCENARIOS, check

    by_id = {sc["id"]: sc for sc in SCENARIOS}
    for sid in ("no_reminder", "mood_tease", "mood_scold", "mood_praise", "mood_plain"):
        assert by_id[sid]["expect"].get("reply") is True, f"{sid} must require reply=True"

    # mood_praise thanks Violet for a reminder, so it must carry an alternating in-context reminder
    # history rather than relying on cross-scenario DB bleed.
    praise_hist = by_id["mood_praise"]["history"]
    assert len(praise_hist) == 4
    assert [who for who, _ in praise_hist] == ["me", "her", "me", "her"]
    assert any(who == "her" and "报告" in text for who, text in praise_hist)

    # Empty reply must FAIL even when mood/reminders nominally match (e.g. mood_plain with mood=0.0)
    empty_ev = {
        "texts": [], "voices": [], "stickers": [], "images": [], "pokes": [], "tools": [],
        "reminders": 0, "mood": 0.0, "flags": [],
    }
    for sid, mood_val in (
        ("no_reminder", 0.0),
        ("mood_tease", 0.0),
        ("mood_scold", -0.3),
        ("mood_praise", 0.3),
        ("mood_plain", 0.0),
    ):
        ev = {**empty_ev, "mood": mood_val}
        ok, problems = check(by_id[sid], ev)
        assert not ok, f"{sid} silently passed with an empty reply"
        assert "没有回应" in problems

    # Non-empty reply passes when mood/reminder criteria match
    for sid, mood_val in (
        ("no_reminder", 0.0),
        ("mood_tease", 0.0),
        ("mood_scold", -0.3),
        ("mood_praise", 0.3),
        ("mood_plain", 0.0),
    ):
        ev = {**empty_ev, "texts": ["嗯嗯"], "mood": mood_val}
        ok, problems = check(by_id[sid], ev)
        assert ok and not problems, (sid, problems)

    # Voice-only or sticker-only counts as a reply
    ok_sticker, _ = check(by_id["mood_plain"], {**empty_ev, "stickers": ["s.png"]})
    ok_voice, _ = check(by_id["no_reminder"], {**empty_ev, "voices": ["好呀"]})
    assert ok_sticker and ok_voice

    # Group chatter still allows staying quiet
    for sid in ("group_chatter", "long_chatter_group"):
        ok, problems = check(by_id[sid], empty_ev)
        assert ok and not problems


# ---------------------------------------------------------------------------
# 3. scenario_eval.py: summary_line operator precedence
# ---------------------------------------------------------------------------

def test_summary_line_formats_sticker_only_voice_only_text_only_and_empty():
    from tools.scenario_eval import summary_line

    def make_result(**did_overrides):
        did = {
            "texts": [], "voices": [], "stickers": [], "images": [], "pokes": [], "tools": [],
            "reminders": 0, "mood": 0.0, "flags": [],
            **did_overrides,
        }
        return {
            "id": "demo", "ok": True, "problems": [], "did": did,
            "usage": {"calls": 1, "prompt": 100, "cached": 50, "completion": 10},
            "requests": [],
        }

    sticker_only = summary_line(make_result(stickers=["a.png"]))
    assert "表情×1" in sticker_only
    assert "没说话" not in sticker_only
    assert "只打字" not in sticker_only

    voice_only = summary_line(make_result(voices=["大家好"]))
    assert "语音×1" in voice_only
    assert "没说话" not in voice_only
    assert "只打字" not in voice_only

    image_only = summary_line(make_result(images=["img.png"]))
    assert "画图×1" in image_only
    assert "没说话" not in image_only

    poke_only = summary_line(make_result(pokes=["10001"]))
    assert "拍×1" in poke_only
    assert "没说话" not in poke_only

    text_only = summary_line(make_result(texts=["你好呀"]))
    assert "只打字" in text_only
    assert "没说话" not in text_only

    empty = summary_line(make_result())
    assert "没说话" in empty
    assert "只打字" not in empty

    whitespace_only = summary_line(make_result(texts=["   "]))
    assert "没说话" in whitespace_only


# ---------------------------------------------------------------------------
# 4. virtual_test.py: pronoun perspective swap & role direction for all personas
# ---------------------------------------------------------------------------

def _load_template(key: str) -> dict:
    path = Path(__file__).resolve().parent.parent / "persona" / "templates" / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("template_key", "call", "expected_snippets", "forbidden_snippets"),
    [
        (
            "stepsister",
            "哥",
            [
                "她是你的继妹（沈念初）",
                "她是你没有血缘的继妹",
                "你一直把她当妹妹照顾",
                "现在你们都上大学了，还住在一起",
                "她对你的感情早就不只是妹妹对哥的依赖，但她没说过，你也没发现",
                "注意角色方向：你是哥（主人），她是你的继妹（沈念初）",
                "是她叫你「哥」，不要反过来叫她「哥」",
            ],
            ["对方", "你是你", "你一直把你", "你对你的", "她们都上大学"],
        ),
        (
            "stepsister",
            "姐",
            [
                "她是你的继妹（沈念初）",
                "她是你没有血缘的继妹",
                "你一直把她当妹妹照顾",
                "现在你们都上大学了，还住在一起",
                "她对你的感情早就不只是妹妹对姐的依赖，但她没说过，你也没发现",
                "注意角色方向：你是姐（主人），她是你的继妹（沈念初）",
                "是她叫你「姐」，不要反过来叫她「姐」",
            ],
            ["对方", "你是你", "你一直把你", "你对你的", "她们都上大学"],
        ),
        (
            "childhood",
            "",
            [
                "她是你的青梅竹马（沈知夏）",
                "她和你从小一起长大",
                "她暗恋你很多年，可你一直把她当最熟的老朋友",
                "注意角色方向：你是主人，她是你的青梅竹马（沈知夏）",
            ],
            ["对方", "你和你", "你暗恋你", "把你当最熟"],
        ),
        (
            "junior",
            "前辈",
            [
                "她是你的后辈（夏栀）",
                "她是你社团里低一级的后辈",
                "刚进社团时是你带的她，从那以后她一直叫你“前辈”",
                "可你只把她当需要照顾的后辈",
                "注意角色方向：你是前辈（主人），她是你的后辈（夏栀）",
                "是她叫你「前辈」，不要反过来叫她「前辈」",
            ],
            ["对方", "你是你", "是你带的你", "你一直叫你"],
        ),
        (
            "quiet",
            "",
            [
                "她是你的同班同学（朝雾唯）",
                "她是你的同班同学，住得也近",
                "一直在注意你：起初只是想被你看见，后来变成很怕你不再看她",
                "她喜欢你，也知道自己喜欢得有点太用力",
                "注意角色方向：你是主人，她是你的同班同学（朝雾唯）",
            ],
            ["对方", "你是你", "注意你：起初只是想被你看见，后来变成很怕你不再看你"],
        ),
    ],
)
def test_owner_who_pronoun_direction_for_all_builtin_relation_personas(
    template_key, call, expected_snippets, forbidden_snippets
):
    from tools.virtual_test import _owner_who

    data = _load_template(template_key)
    who = _owner_who(data, call=call)
    for snippet in expected_snippets:
        assert snippet in who, f"{template_key}({call}) missing {snippet!r}\nGOT:\n{who}"
    for bad in forbidden_snippets:
        assert bad not in who, f"{template_key}({call}) should not contain {bad!r}\nGOT:\n{who}"


def test_owner_who_for_non_relation_persona():
    from tools.virtual_test import _owner_who

    who = _owner_who(_load_template("violet"))
    assert "Violet" in who
    assert "网友" in who


# ---------------------------------------------------------------------------
# 5 & 6. virtual_test.py: _owner_turn end-to-end prompt (pronouns, stepsister
#        coherence, same-day earlier session memory) + _judge coherence
# ---------------------------------------------------------------------------

class _SpyModels:
    mock = False
    owner_model = "spy-owner"
    judge_model = "spy-judge"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[tuple[str, str, str, float]] = []

    def ask(self, model: str, system: str, user: str, temperature: float) -> str:
        self.calls.append((model, system, user, temperature))
        return self.reply


@pytest.mark.parametrize(
    ("track_name", "expected_call", "expect_dorm_setting"),
    [
        ("long", "哥", True),
        ("stepsister_f", "姐", True),
        ("childhood", "", False),
        ("junior", "前辈", False),
        ("quiet", "", False),
    ],
)
def test_owner_turn_prompt_direction_coherence_and_same_day_memory(
    tmp_path, monkeypatch, track_name, expected_call, expect_dorm_setting
):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    from ai import about_owner, persona_templates as pt
    from database import VioletDatabase
    from tools.virtual_test import TRACKS, _judge, _owner_turn

    db = VioletDatabase(tmp_path / "vt.db")
    track = TRACKS[track_name]
    pt.switch(db, track["persona"], "keep")
    if track["about"]:
        about_owner.save(track["about"])

    now = datetime(2026, 11, 16, 21, 0, tzinfo=timezone(timedelta(hours=8)))
    earlier_today = [
        {"t": "2026-11-16T12:30", "who": "主人", "text": "刚在食堂吃了黄焖鸡"},
        {"t": "2026-11-16T12:31", "who": "她", "text": "好香呀，今天汤咸不咸"},
    ]
    current_chat = [
        {"t": "2026-11-16T19:00", "who": "她（主动）", "text": "晚上吃什么了"},
    ]

    spy = _SpyModels(json.dumps({"messages": ["刚吃完"], "end": False}, ensure_ascii=False))
    lines, ended = _owner_turn(
        spy,
        track,
        track["sessions"][-1],
        now,
        known=["周五下午两点高数期中考试"],
        chat=current_chat,
        turn=0,
        earlier_today=earlier_today,
    )
    assert lines == ["刚吃完"] and not ended
    assert len(spy.calls) == 1
    owner_prompt = spy.calls[0][2]

    # No broken pronoun swap
    assert "对方" not in owner_prompt
    assert "你是你" not in owner_prompt
    assert "你一直把你" not in owner_prompt
    assert "你对你的" not in owner_prompt

    # Explicit role anchor matches expected_call
    if expected_call:
        assert f"是她叫你「{expected_call}」，不要反过来叫她「{expected_call}」" in owner_prompt
    else:
        assert "注意角色方向：你是主人，她是你的" in owner_prompt

    # Same-day earlier session memory is visible and clearly separated from current session
    assert "今天早些时候你和她聊过的内容（不要假装忘了）" in owner_prompt
    assert "刚在食堂吃了黄焖鸡" in owner_prompt
    assert "晚上吃什么了" in owner_prompt

    # First session of the day (no earlier_today) omits the earlier-today section
    spy_first = _SpyModels(json.dumps({"messages": ["在吗"], "end": False}, ensure_ascii=False))
    _owner_turn(spy_first, track, track["sessions"][0], now, known=[], chat=[], turn=0, earlier_today=[])
    assert "今天早些时候你和她聊过的内容" not in spy_first.calls[0][2]

    # Stepsister tracks include coherent dorm/weekend-home living note in BOTH owner and judge prompts
    if expect_dorm_setting:
        assert "工作日" in owner_prompt and "住学校宿舍" in owner_prompt and "周末回家住" in owner_prompt
        if track_name == "long":
            assert "阿杰" in owner_prompt

        judge_spy = _SpyModels(json.dumps({"problems": [], "probes": [], "good": [], "scores": {}}))
        _judge(
            judge_spy, track, day=1,
            entries=earlier_today + current_chat,
            known=[], probes=[], stage=None, reminders=[], set_today=[], earlier=[], memes=[],
        )
        judge_prompt = judge_spy.calls[0][2]
        assert "主人的居住与日常设定" in judge_prompt
        assert "工作日" in judge_prompt and "住学校宿舍" in judge_prompt and "周末回家住" in judge_prompt
        assert "你和她一家同住" not in judge_prompt  # properly formatted for third-person judge ("主人")
        assert "主人和她一家同住" in judge_prompt
        assert "住在一起" in judge_prompt
    else:
        judge_spy = _SpyModels(json.dumps({"problems": [], "probes": [], "good": [], "scores": {}}))
        _judge(
            judge_spy, track, day=1,
            entries=earlier_today + current_chat,
            known=[], probes=[], stage=None, reminders=[], set_today=[], earlier=[], memes=[],
        )
        judge_prompt = judge_spy.calls[0][2]
        assert "在同一所学校、住得也近" in judge_prompt
        if track_name == "childhood":
            assert "一路同校" in judge_prompt
        elif track_name == "quiet":
            assert "同班同学" in judge_prompt
        elif track_name == "junior":
            assert "社团里低一级的后辈" in judge_prompt

    db.close()


def test_run_track_passes_earlier_same_day_transcript_without_duplicating_current_chat(tmp_path, monkeypatch):
    """Verify run_track snapshots world.transcript[day_start:] before world.advance(at)
    so session 2 sees session 1's lines while staying disjoint from inter-session proactive chat."""
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_DB_PATH", str(tmp_path / "vt.db"))
    import tools.virtual_test as vt
    from database import VioletDatabase

    db = VioletDatabase(tmp_path / "vt.db")
    captured: list[tuple[str, list[dict], list[dict]]] = []

    class _FakeWorld:
        def __init__(self, track, clock, mock):
            self.db = db
            self.clock = clock
            self.transcript: list[dict] = []
            self.runtime = SimpleNamespace(stop=lambda *a, **k: None)

        def advance(self, until, step_minutes=10):
            self.clock.t = until
            if until.hour == 21:
                msg = {"t": until.isoformat(timespec="minutes"), "who": "她（主动）", "text": "晚饭吃啥了"}
                self.transcript.append(msg)
                return [msg]
            return []

        def owner_says(self, lines):
            for text in lines:
                self.transcript.append({"t": self.clock().isoformat(timespec="minutes"), "who": "主人", "text": text})
            reply = {"t": self.clock().isoformat(timespec="minutes"), "who": "她", "text": "收到"}
            self.transcript.append(reply)
            return [reply], 0.1

    def fake_owner_turn(models, track, session, now, known, chat, turn, earlier_today=None):
        captured.append((session["at"], list(earlier_today or []), list(chat)))
        text = "中午在食堂吃黄焖鸡" if session["at"] == "12:30" else "我今天中午跟你说我在干嘛来着？"
        return [text], True

    monkeypatch.setattr(vt, "World", _FakeWorld)
    monkeypatch.setattr(vt, "_owner_turn", fake_owner_turn)
    monkeypatch.setattr(vt, "say", lambda *a, **k: None)

    args = SimpleNamespace(days=1, owner_model="m", judge_model="m", mock=True, max_tokens=1_000_000)
    vt.run_track("stepsister_f", args)

    assert len(captured) == 2
    at1, earlier1, chat1 = captured[0]
    at2, earlier2, chat2 = captured[1]
    assert at1 == "12:30" and earlier1 == [] and chat1 == []
    assert at2 == "21:00"
    assert any("中午在食堂吃黄焖鸡" in e["text"] for e in earlier2)
    assert all("晚饭吃啥了" not in e["text"] for e in earlier2)
    assert any("晚饭吃啥了" in e["text"] for e in chat2)
    db.close()
