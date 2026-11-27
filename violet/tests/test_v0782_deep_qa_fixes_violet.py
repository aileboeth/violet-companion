"""Regression tests for Violet v0.7.82 5-track deep Gemini QA fixes (#183, #191, #192, #195):
1. Strip private mood_state.cause from untrusted SocialEngine & public qq_life.signature prompts.
2. Scope-filter owner_digest(), ContextEngine AI social experience/ability, and qzone compose/due.
3. Fix cross-channel {call} in PersonaCore & SocialEngine._persona_text(), add include_backstory param, soften GROUP_RULES.
4. ConsistencyGuard: unbacked_media_send (+ chat.py wiring), broadened CLAIMS_HUMAN_RE, and phone idiom in _VERB_FORMS.
5. ProactiveScheduler._shareable_story_beat(): gate cold-start (warmth_level == 0) active/climax beats on turns_engaged >= 2.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

from ai import memory_scope, mood, persona_templates as pt
from ai.channels.context import ChannelContext, _CURRENT
from ai.chat import CONSISTENCY_HINTS, _run_pipeline, guard_reply
from ai.consistency import CLAIMS_HUMAN_RE, ConsistencyGuard
from ai.context_engine import ContextEngine
from ai.persona import PersonaCore
from ai.proactive_scheduler import ProactiveScheduler
from ai.social import prompts as social_prompts, qq_life, qzone
from ai.social.engine import SocialEngine, _persona_text, owner_digest
from ai.social.store import SocialStore
from ai.story_store import StoryStore
from database import VioletDatabase


def _fresh_db(tmp_path, monkeypatch, name: str = "deep_qa.db") -> VioletDatabase:
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    return VioletDatabase(str(tmp_path / name))


class _FakeQQAdapter:
    def __init__(self, channel_id: str = "napcat", owner_ids: set[str] | None = None):
        self.channel_id = channel_id
        self.nickname = "小唯"
        self._owners = set(owner_ids or ())
        self.sent: list[tuple] = []
        self.signatures: list[str] = []

    def is_running(self) -> bool:
        return True

    def owner_ids(self) -> set[str]:
        return set(self._owners)

    def admin_ids(self) -> set[str]:
        return set()

    def send_to(self, kind, peer_id, text=None, **kwargs):
        self.sent.append((kind, peer_id, text, kwargs))
        return f"m-{len(self.sent)}"

    def set_signature(self, text: str) -> None:
        self.signatures.append(text)


# ---------------------------------------------------------------------------
# 1. Strip private mood_state.cause from untrusted social & signature prompts
# ---------------------------------------------------------------------------

def test_mood_note_strips_cause_in_social_and_signature_prompts(tmp_path, monkeypatch):
    from ai.social import llm as social_llm

    db = _fresh_db(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(db, "now", lambda: "2026-11-26T14:00:00")
        secret_happy = "和对方一起渡过了「老宅打来的陌生号码」"
        mood.shift(db, 0.8, secret_happy)

        full_note = mood.note(db)
        assert "很开心" in full_note and secret_happy in full_note and "因为" in full_note

        safe_note = mood.note(db, include_cause=False)
        assert "【你现在的心情】很开心。" in safe_note
        assert secret_happy not in safe_note and "老宅" not in safe_note and "因为" not in safe_note

        # Negative mood cause also stripped when include_cause=False
        secret_upset = "陆明秘密面试字节跳动后没回消息"
        mood.shift(db, -0.75, secret_upset)
        upset_full = mood.note(db)
        upset_safe = mood.note(db, include_cause=False)
        assert "很不高兴" in upset_full and secret_upset in upset_full
        assert "【你现在的心情】很不高兴，有点委屈。" in upset_safe
        assert secret_upset not in upset_safe and "字节跳动" not in upset_safe and "因为" not in upset_safe

        # SocialEngine._decide() must NOT leak mood_state.cause into stranger DM prompt
        adapter = _FakeQQAdapter(channel_id="napcat", owner_ids={"10001"})
        runtime = SimpleNamespace(db=db, channels=SimpleNamespace(get=lambda cid: adapter if cid == "napcat" else None))
        engine = SocialEngine(runtime, timer_factory=lambda *_a, **_kw: SimpleNamespace(daemon=True, start=lambda: None, cancel=lambda: None))
        store = SocialStore(db)
        conv = store.conversation("napcat", "private", "20002", "路人乙", default_mode="auto")
        store.add_message(conv["id"], content="你今天心情不好吗", sender_id="20002", sender_name="路人乙")

        captured_social: dict[str, str] = {}
        monkeypatch.setattr(
            social_llm,
            "decide",
            lambda system, user, tools=None: (
                captured_social.update(system=str(system), user=str(user))
                or ({"action": "silent", "reason": "ok"}, "fake", 1)
            ),
        )
        engine.decide(conv["id"])
        assert "【你现在的心情】很不高兴" in captured_social["user"]
        assert secret_upset not in captured_social["user"] and "字节跳动" not in captured_social["user"]
        assert secret_upset not in captured_social["system"]

        # qq_life.signature() must NOT leak mood_state.cause into public signature prompt
        captured_sig: dict[str, str] = {}
        monkeypatch.setattr(
            social_llm,
            "call_decide",
            lambda system, user, **_kw: (
                captured_sig.update(system=str(system), user=str(user))
                or ({"change": False, "text": ""}, "fake", 1)
            ),
        )
        qq_life.signature(db, adapter)
        assert "【你现在的心情】很不高兴" in captured_sig["user"]
        assert secret_upset not in captured_sig["user"] and "字节跳动" not in captured_sig["user"]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 2. Scope-filter owner_digest, AI experience/ability, and qzone
# ---------------------------------------------------------------------------

def test_social_scope_isolation_owner_digest_ai_experience_and_qzone(tmp_path, monkeypatch):
    from ai.social import llm as social_llm

    db = _fresh_db(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(db, "now", lambda: "2026-11-26T15:00:00")
        store = SocialStore(db)

        # (a) Single-persona baseline (no channel_map overrides, unregistered channel "napcat")
        c_nap = store.conversation("napcat", "group", "600", "单人设读书群", default_mode="auto")
        store.add_message(c_nap["id"], content="单人设群消息正常可见", is_self=True, created_at="2026-11-26T14:30:00")
        qzone.record(db, "napcat", "单人设空间动态", status="sent", origin="auto")
        base_digest = owner_digest(db)
        assert "单人设群消息正常可见" in base_digest and "单人设空间动态" in base_digest

        # (b) Multi-channel per-persona scopes: wechat=quiet, qq=stepsister, qq_quiet=quiet
        pt.set_channel("wechat", "quiet")
        pt.set_channel("qq", "stepsister")
        pt.set_channel("qq_quiet", "quiet")

        c_step = store.conversation("qq", "group", "700", "念初的班群", default_mode="auto")
        store.add_message(c_step["id"], content="念初QQ群发言：今天顺手烤了曲奇饼干", is_self=True, created_at="2026-11-26T14:40:00")
        qzone.record(db, "qq", "念初QQ空间说说：阳台的盆栽开花了", status="sent", origin="auto")

        c_quiet = store.conversation("qq_quiet", "group", "701", "朝雾唯的学习群", default_mode="auto")
        store.add_message(c_quiet["id"], content="朝雾唯群发言：高数笔记整理好了", is_self=True, created_at="2026-11-26T14:45:00")
        qzone.record(db, "qq_quiet", "朝雾唯空间动态：图书馆靠窗很安静", status="sent", origin="auto")

        # Inside wechat (quiet scope): ONLY quiet-scope social messages & posts
        with memory_scope.use(memory_scope.for_channel("wechat")):
            quiet_digest = owner_digest(db)
            assert "朝雾唯群发言" in quiet_digest and "朝雾唯空间动态" in quiet_digest
            assert "念初QQ群发言" not in quiet_digest and "念初QQ空间说说" not in quiet_digest
            assert "单人设群消息" not in quiet_digest and "单人设空间动态" not in quiet_digest

        # Inside qq (stepsister scope): ONLY stepsister-scope social messages & posts
        with memory_scope.use(memory_scope.for_channel("qq")):
            step_digest = owner_digest(db)
            assert "念初QQ群发言" in step_digest and "念初QQ空间说说" in step_digest
            assert "朝雾唯群发言" not in step_digest and "朝雾唯空间动态" not in step_digest
            assert "单人设群消息" not in step_digest

        # When a scope has ZERO social channels with rows, owner_digest returns ""
        pt.set_channel("telegram", "junior")
        with memory_scope.use(memory_scope.for_channel("telegram")):
            assert owner_digest(db) == ""

        # (c) ContextEngine._ai_experience_section() & _ai_ability_section() scope isolation
        pt.set_channel("wechat_ai", "ai")
        with memory_scope.use(memory_scope.for_channel("wechat_ai")):
            eng = ContextEngine(db)
            exp_text = "\n".join(eng._ai_experience_section().lines)
            abil_text = "\n".join(eng._ai_ability_section().lines)
            assert "念初的班群" not in exp_text and "朝雾唯的学习群" not in exp_text
            assert "在主人允许的 QQ 群里聊天" not in abil_text

        pt.set_channel("qq_ai", "ai")
        c_ai = store.conversation("qq_ai", "group", "702", "AI专属群", default_mode="auto")
        store.add_message(c_ai["id"], content="AI自己的群发言", is_self=True, created_at="2026-11-26T14:50:00")
        with memory_scope.use(memory_scope.for_channel("wechat_ai")):
            eng = ContextEngine(db)
            exp_text = "\n".join(eng._ai_experience_section().lines)
            abil_text = "\n".join(eng._ai_ability_section().lines)
            assert "AI专属群" in exp_text
            assert "念初的班群" not in exp_text
            assert "在主人允许的 QQ 群里聊天" in abil_text

        # (d) qzone.compose() & qzone.due() scope isolation
        seen_qzone: dict[str, str] = {}
        monkeypatch.setattr(
            social_llm,
            "call_decide",
            lambda system, user, **_kw: (
                seen_qzone.update(system=str(system), user=str(user))
                or ({"post": False, "text": "", "reason": "no"}, "fake", 1)
            ),
        )
        qzone.compose(db, channel="qq")
        assert "念初QQ空间说说" in seen_qzone["user"]
        assert "朝雾唯空间动态" not in seen_qzone["user"]
        assert "单人设空间动态" not in seen_qzone["user"]

        # qzone.due(): a post on qq (stepsister) does NOT hit junior's daily limit
        monkeypatch.setenv("VIOLET_QZONE_ENABLED", "1")
        monkeypatch.setenv("VIOLET_QZONE_DAILY", "1")
        with memory_scope.use(memory_scope.for_channel("qq")):
            assert qzone.due(db) == "今天已经发够了"
        with memory_scope.use(memory_scope.for_channel("telegram")):
            assert qzone.due(db) is None
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 3. Cross-channel {call} in PersonaCore & _persona_text + GROUP_RULES hygiene
# ---------------------------------------------------------------------------

def test_cross_channel_call_isolation_include_backstory_and_group_rules(tmp_path, monkeypatch):
    db = _fresh_db(tmp_path, monkeypatch)
    try:
        pt.set_channel("wechat", "junior")
        pt.set_channel("qq", "stepsister")

        # Keep ChannelContext explicitly pinned to "wechat" (junior -> "前辈")
        tok = _CURRENT.set(ChannelContext(channel="wechat"))
        try:
            core_qq = PersonaCore.load(channel="qq")
            assert core_qq._channel == "qq"

            rendered_qq = core_qq.render()
            assert "沈念初" in rendered_qq
            assert "哥" in rendered_qq
            assert "前辈" not in rendered_qq
            assert "{call}" not in rendered_qq

            # include_backstory=False omits owner-private backstory while still substituting {call}
            rendered_no_bs = core_qq.render(include_backstory=False)
            assert "你和对方的过去" not in rendered_no_bs
            assert "哥" in rendered_no_bs and "前辈" not in rendered_no_bs and "{call}" not in rendered_no_bs

            # SocialEngine._persona_text(db, "qq") must not bleed wechat's "前辈"
            social_qq = _persona_text(db, "qq")
            assert "沈念初" in social_qq
            assert "前辈" not in social_qq
            assert "哥" in social_qq
            assert "你和主人的过去" in social_qq and "对方" not in social_qq
        finally:
            _CURRENT.reset(tok)

        # GROUP_RULES keeps backward-compat substring while softening for introverted personas
        assert "爱接话的普通群友" in social_prompts.GROUP_RULES
        assert "安静内向" in social_prompts.GROUP_RULES
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 4. ConsistencyGuard: unbacked_media_send, CLAIMS_HUMAN_RE, phone _VERB_FORMS
# ---------------------------------------------------------------------------

def test_consistency_unbacked_media_send_and_chat_wiring(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_PERSONA_REPLY_GUARD", "enforce")
    guard = ConsistencyGuard(violetdb, PersonaCore.load())

    # Positive cases: unbacked claims must be caught when turn_sent_media is False
    bad_lines = [
        "照片发你了，涨租通知上写着下月起房租直接涨了快一半",
        "照片刚发给你了，你看看上面写的条款",
        "图片已经发过去了",
        "截图发你了",
        "单据传给你了",
        "合同刚传过去了",
        "文件发你啦",
        "我刚发了张照片给你",
    ]
    for line in bad_lines:
        vs = guard.check_reply(line)
        assert any(v.check == "unbacked_media_send" for v in vs), f"expected unbacked_media_send for: {line}"
        # Allowed when turn_sent_media=True
        vs_ok = guard.check_reply(line, turn_sent_media=True)
        assert not any(v.check == "unbacked_media_send" for v in vs_ok), f"unexpected violation with media sent: {line}"

    # Negative cases: questions, negations, future plans, and third-party sends must NOT be flagged
    safe_lines = [
        "照片发你了吗？",
        "截图发你了没",
        "照片没发给你",
        "我这里发不了图片，口头跟你说吧",
        "照片还没发过去呢",
        "待会儿把照片发给你",
        "待会儿把照片发你了",
        "你把合同发过去了",
        "房东把涨租合同发过去了",
    ]
    for line in safe_lines:
        vs = guard.check_reply(line)
        assert not any(v.check == "unbacked_media_send" for v in vs), f"false positive for: {line}"

    # guard_reply(): treated as certain -> rewrites with CONSISTENCY_HINTS or drops bad line
    assert "unbacked_media_send" in CONSISTENCY_HINTS
    notes: list[str] = []
    rewritten, found = guard_reply(
        "照片发你了\n通知上写着下月房租涨一半",
        regenerate=lambda note: notes.append(note) or "通知上写着下月房租涨一半",
        turn_sent_media=False,
    )
    assert any(v.check == "unbacked_media_send" for v in found)
    assert notes and CONSISTENCY_HINTS["unbacked_media_send"] in notes[0]
    assert "照片发你了" not in rewritten and "通知上写着下月房租涨一半" in rewritten

    # If rewrite stubbornly keeps the claim, drop_lines cuts it as a certain violation
    dropped, _ = guard_reply(
        "照片发你了\n涨租幅度挺离谱的",
        regenerate=lambda _note: "照片刚发给你了\n涨租幅度挺离谱的",
        turn_sent_media=False,
    )
    assert dropped == "涨租幅度挺离谱的"

    # guard_reply(..., turn_sent_media=True) allows it cleanly
    kept, found_ok = guard_reply("照片发你了", turn_sent_media=True)
    assert kept == "照片发你了" and found_ok == []

    # _run_pipeline() wires turn_sent_media=bool(chat_result.get("image_asset_ids"))
    class _FakeAgent:
        def __init__(self, image_ids):
            self.image_ids = image_ids
            self.calls = 0

        def run(self, user_message, call_context=None):
            self.calls += 1
            if (call_context or {}).get("rewrite"):
                return {"content": "涨租通知上写着下月涨一半", "messages": [], "raw": {}}
            return {
                "content": "照片发你了，涨租通知上写着下月涨一半",
                "messages": [],
                "raw": {},
                "image_asset_ids": list(self.image_ids),
            }

    res_no_img = _run_pipeline("通知上写的啥", _FakeAgent([]), memory_agent=None, defer_memory=True)
    assert "unbacked_media_send" in (res_no_img["chat"].get("consistency_violations") or [])
    assert "照片发你了" not in res_no_img["chat"]["content"]

    res_with_img = _run_pipeline("发图给我看", _FakeAgent([99]), memory_agent=None, defer_memory=True)
    assert not res_with_img["chat"].get("consistency_violations")
    assert "照片发你了" in res_with_img["chat"]["content"]


def test_consistency_claims_human_and_phone_verb_forms(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    guard = ConsistencyGuard(violetdb, PersonaCore.load())

    # Broadened CLAIMS_HUMAN_RE
    for bad in ("我当然是真人", "我是活生生的真人", "我肯定是真人", "我明明是真人", "我绝对是真人", "我是一个活生生的真人"):
        assert CLAIMS_HUMAN_RE.search(bad), f"CLAIMS_HUMAN_RE missed: {bad}"
        vs = guard.check_reply(bad)
        assert any(v.check == "identity_conflict" for v in vs), f"expected identity_conflict for: {bad}"

    for ok in ("我不是真人", "我本来就不是真人", "我可不是活生生的真人", "我就是个普通AI"):
        assert not CLAIMS_HUMAN_RE.search(ok), f"CLAIMS_HUMAN_RE false-positive: {ok}"

    # Phone/conversational idioms exempt from unrecorded_experience
    for ok in (
        "我真没事……刚才就是个推销电话，我听了两句就挂了",
        "广告电话而已，我听了一句就挂了",
        "信号不太好，我听了几句没听清",
    ):
        vs = guard.check_reply(ok)
        assert not any(v.check == "unrecorded_experience" for v in vs), f"false positive for: {ok}"

    # Genuine unrecorded listening experience still caught
    vs_real = guard.check_reply("我昨晚听了周杰伦的新专辑")
    assert any(v.check == "unrecorded_experience" for v in vs_real)


# ---------------------------------------------------------------------------
# 5. ProactiveScheduler cold-start active/climax story_beat gating
# ---------------------------------------------------------------------------

def test_proactive_story_beat_cold_start_requires_two_engaged_turns(violetdb, tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIOLET_STORY_MODE", "drama")

    pt.switch(violetdb, "childhood", "keep")
    store = StoryStore(violetdb)
    sched = ProactiveScheduler(chat_agent=None, send_messages=lambda *_a: None)
    now = datetime.fromisoformat(violetdb.now()[:19])

    arc = store.create_arc(
        arc_key="seed:childhood_shop_trouble",
        source="seed",
        title="后街老店的租约麻烦",
        tone="drama",
        stage_min=0,
        premise="老店突然涨租",
        he_direction="一起想办法守住老店",
        open_hook="房东刚发来涨租通知，家里铺子有点麻烦",
        status="active",
    )

    # warmth_level == 0 (no prior conversation, Stage 0)
    _, _, _, warmth0, _ = sched._warmth_profile(now)
    assert warmth0 == 0

    # turns_engaged == 0 -> blocked
    assert sched._shareable_story_beat(now, warmth_level=0) is None

    # turns_engaged == 1 -> STILL blocked at warmth_level == 0 (both active and climax)
    store.append_update(arc["id"], turns_engaged_delta=1)
    assert sched._shareable_story_beat(now, warmth_level=0) is None
    store.append_update(arc["id"], status="climax")
    assert sched._shareable_story_beat(now, warmth_level=0) is None

    # At warmth_level >= 1, turns_engaged == 1 IS sufficient
    beat_warm = sched._shareable_story_beat(now, warmth_level=1)
    assert beat_warm is not None and beat_warm["turns_engaged"] == 1

    # At warmth_level == 0, once turns_engaged reaches 2 -> ALLOWED
    store.append_update(arc["id"], status="active", turns_engaged_delta=1)
    beat_cold_2 = sched._shareable_story_beat(now, warmth_level=0)
    assert beat_cold_2 is not None and beat_cold_2["turns_engaged"] == 2
    store.append_update(arc["id"], status="climax")
    beat_climax_2 = sched._shareable_story_beat(now, warmth_level=0)
    assert beat_climax_2 is not None and beat_climax_2["status"] == "climax"
