"""2026-11-03 (user: 冷漠、惜字如金、完全感受不到情绪 -- WeChat with the Eve persona, relationship 恋人)."""

from __future__ import annotations

from tests.test_phase2_scenarios import make_world  # noqa: F401  (pytest fixture)


def test_a_chosen_relationship_is_not_overridden_by_the_guessed_stage(monkeypatch, violetdb):
    from ai import persona_templates
    from ai.context_engine import ContextEngine
    from ai.profile_store import Profile
    violetdb.execute("INSERT INTO messages (role, direction, message_type, content, created_at) VALUES "
                     "('user', 'incoming', 'text', '以后保持点距离，别撒娇了', '2026-09-01T10:00:00+08:00')")
    Profile(violetdb).preferences.set_explicit("closeness", "distance", evidence="保持点距离，别撒娇了",
                                               user_message="以后保持点距离，别撒娇了")
    violetdb.add_user_message("text", "在干嘛")
    monkeypatch.setattr(persona_templates, "relationship", lambda channel: {"kind": "close", "label": "亲近（默认）", "call": None})
    default = "\n".join(ContextEngine(violetdb)._bond_section().lines)
    assert "保持轻松友好的距离" in default                      # the default relationship respects explicit distance
    monkeypatch.setattr(persona_templates, "relationship", lambda channel: {"kind": "partner", "label": "恋人", "call": "宝宝"})
    chosen = "\n".join(ContextEngine(violetdb)._bond_section().lines)
    assert "保持轻松友好的距离" not in chosen and "你们是恋人" in chosen and "宝宝" in chosen


def test_the_owners_own_bad_day_is_not_scolding_and_an_upset_mood_is_not_cold(violetdb):
    from ai import mood
    assert not mood.SCOLD.search("今天工作好烦") and not mood.SCOLD.search("烦死了")
    assert mood.SCOLD.search("你好烦") and mood.SCOLD.search("烦死你了") and mood.SCOLD.search("笨蛋")
    mood.feel(violetdb, "你好烦啊，闭嘴")        # 2026-11-09: a teasing 笨蛋 alone hardly moves her mood any more
    mood.feel(violetdb, "你好烦啊，闭嘴")
    line = mood.note(violetdb)
    assert "不高兴" in line and "短一点" not in line and "淡一点" not in line and "不冷着对方" in line
    assert "话也会比平时多" in line                             # upset = more words, not fewer


def test_the_preferences_list_says_them_in_words():
    import os
    js = open(os.path.join(os.path.dirname(__file__), "..", "wx_ilink_py", "static", "control.js"), encoding="utf-8").read()
    assert 'reply_length: { short: "喜欢简短的回复"' in js and "prefText(r).includes(q)" in js
    assert 'closeness: { normal: "正常相处", distance: "希望保持一点距离" }' in js
    from ai.profile_schema import PREF_LABELS
    assert PREF_LABELS["reply_length"]["short"] == "喜欢简短的回复"          # same words as the page
    assert "距离" in PREF_LABELS["closeness"]["distance"]


def test_proactive_usage_recorded_separately_from_chat(make_world):
    from tests.test_phase2_scenarios import age_everything, arm, chat, say
    from ai import usage
    w = make_world()
    recorded = []
    orig_record = usage.record
    def spy(source, model, u, **kw):
        recorded.append(source)
        return orig_record(source, model, u, **kw)
    usage.record = spy
    try:
        chat(w, normal="在呀", proactive="午休了吗")
        say(w, "在吗")
        age_everything(w.db, 3)
        arm(w)
        w.scheduler.tick()
    finally:
        usage.record = orig_record
    assert "ChatAgent" in recorded and "proactive" in recorded
    assert usage.LABELS["proactive"] == "主动联系"


def test_closeness_distance_via_memory_and_clear(violetdb):
    from ai import relationship
    from ai.profile_store import Profile
    stage2 = {"index": 2, "total": 3, "label": "熟了之后"}
    assert relationship.assess(violetdb).stage != "distance"
    Profile(violetdb).preferences.set_explicit("closeness", "distance", evidence="保持点距离", user_message="我们保持点距离")
    view = relationship.assess(violetdb)
    assert view.stage == "distance" and view.distance_quote == "保持点距离" and view.distance_at
    assert relationship.warmth_level(view, persona_stage=stage2) == 0
    assert relationship.clear_distance(violetdb)
    assert relationship.assess(violetdb).stage != "distance"
    assert relationship.warmth_level(relationship.assess(violetdb), persona_stage=stage2) == 3
    assert relationship.warmth_level(relationship.assess(violetdb), rel_kind="partner") == 2
