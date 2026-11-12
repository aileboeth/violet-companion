"""2026-11-12 (user: 人设不太行；开局先收集用户的基本信息；四个动漫式模板，自带关系的不要再选关系，阶段自动推进；
温柔陪伴 / 元气同学 / 冷静可靠去掉): the owner's own form, personas that bring their relationship and its stages, the
retired templates."""

from __future__ import annotations

import json
from datetime import date

import pytest

from tests.test_qq_social_violet import SAME, admin, db  # noqa: F401

NEW = ("stepsister", "childhood", "junior", "quiet")


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


@pytest.fixture()
def tidy():
    """The admin fixture's data folder is shared by the test run: leave no persona choice or form behind."""
    yield
    from ai import persona_templates as pt
    (pt.control_dir() / "about-owner.json").unlink(missing_ok=True)
    data = pt.channel_map()
    if data.pop("wechat", None) is not None:
        pt._write(pt.control_dir() / pt.CHANNEL_FILE, data)


def _context(db, message="在吗"):
    from ai.context_engine import ContextEngine
    return "\n".join(ContextEngine(db).build("chat", message).sections)


# ------------------------------------------------------------------------------------------------ about the owner
def test_the_owners_form_is_validated_and_reaches_every_reply(db, data_dir):
    from ai import about_owner
    assert about_owner.lines() == [] and about_owner.filled() == 0
    assert "对方的基本情况" not in _context(db)                   # nothing filled in: no empty section
    with pytest.raises(ValueError):
        about_owner.save({"gender": "robot"})
    with pytest.raises(ValueError):
        about_owner.save({"birthday": "13-40"})
    with pytest.raises(ValueError):
        about_owner.save({"name": "名" * 21})
    saved = about_owner.save({"name": " 小周 ", "gender": "male", "birthday": "2006-3-5", "status": "student",
                              "grade": "大二", "major": "计算机", "hobbies": "MC、摄影", "city": "广州"})
    assert saved["name"] == "小周" and saved["birthday"] == "2006-03-05"
    assert about_owner.birthday() == (3, 5) and about_owner.age(date(2026, 11, 12)) == 20
    assert about_owner.age(date(2026, 3, 4)) == 19
    text = _context(db)
    for needle in ("对方的基本情况", "名字：小周", "性别：男", "生日：3 月 5 日", "学生（大二，计算机）", "MC、摄影", "广州"):
        assert needle in text, needle
    about_owner.save({"birthday": "03-05", "job": "程序员"})        # no year: no age; a job alone means working
    assert about_owner.lines(date(2026, 11, 12)) == ["- 生日：3 月 5 日", "- 身份：已经工作（程序员）"]


def test_the_form_answers_the_onboarding_questions_and_the_birthday(db, data_dir):
    from ai import about_owner, holidays
    from ai.onboarding import OnboardingPlanner
    from ai.profile_store import ProfileStore
    profile = ProfileStore(db)
    onboarding = OnboardingPlanner(db, profile)
    assert onboarding.next_key().key == "address_name"          # "你希望我怎么称呼你" comes first
    about_owner.save({"name": "小周", "status": "student", "hobbies": "摄影", "birthday": "03-05"})
    assert about_owner.known_keys() == {"address_name", "identity.status", "interest"}
    assert onboarding.next_key().key == "residence.usual"       # only what the form does not say
    days = holidays.for_owner(date(2027, 3, 5), "close", "", holidays.owner_texts(db))
    assert days and days[0].key == "birthday"


def test_the_form_over_the_api_and_on_the_checklist(admin, db, tidy):
    r = admin.get("/api/persona/templates", headers=SAME).get_json()
    assert r["about"]["fields"]["name"] == "" and set(r["about"]["genders"]) == {"male", "female", "other"}
    assert admin.put("/api/persona/about", json={"gender": "x"}, headers=SAME).status_code == 400
    r = admin.put("/api/persona/about", json={"name": "小周", "gender": "female"}, headers=SAME)
    assert r.status_code == 200 and r.get_json()["fields"]["gender"] == "female"
    items = {i["id"]: i for i in admin.get("/api/control/setup", headers=SAME).get_json()["data"]["items"]}
    assert items["about"]["state"] == "ok" and items["about"]["anchor"] == "pl-about" and not items["about"]["required"]


# ------------------------------------------------------------------------------------------------ the templates
def test_the_library_offers_the_new_personas_and_not_the_retired_ones(db, data_dir):
    from ai import persona_templates as pt
    keys = [t["key"] for t in pt.templates()]
    assert keys[:5] == ["violet", *NEW] and {"ai", "dafeiyu", "eve", "blank"} <= set(keys)
    assert not {"gentle", "genki", "calm"} & set(keys)
    assert {"gentle", "genki", "calm"} <= {t["key"] for t in pt.templates(include_retired=True)}
    # one in use stays listed (and goes on working): the file is still there
    pt.set_channel("telegram", "genki")
    listed = {t["key"]: t for t in pt.templates()}
    assert listed["genki"]["retired"] and "gentle" not in listed
    from ai.persona import PersonaCore
    assert PersonaCore.load(channel="telegram").data["identity"]["name"] == "小晴"


@pytest.mark.parametrize("key", NEW)
def test_every_new_template_is_complete_and_keeps_the_fixed_lines(key):
    from ai.settings import PROJECT_ROOT
    d = json.loads((PROJECT_ROOT / "persona" / "templates" / f"{key}.json").read_text(encoding="utf-8"))
    assert d["relation"]["label"] and d["relation"]["text"] and d["has_school"] is True
    assert len(d["stages"]) == 5 and d["stages"][0]["days"] == 0
    assert [s["days"] for s in d["stages"]] == sorted(s["days"] for s in d["stages"])
    for field in ("personality", "communication_habits", "examples", "backstory", "avoid", "values", "flaws"):
        assert len(d[field]) >= 4, field
    whole = json.dumps(d, ensure_ascii=False)
    assert "虚构" in d["identity"]["virtual_declaration"]
    assert "对不确定的事情承认不知道" in d["values"]
    assert "大" in d["identity"]["self_description"] + "".join(d["living_constraints"])   # a university student
    assert "高一" not in whole and "高二" not in whole and "高中" not in whole and "男主" not in whole
    assert any("真人" in p for p in d["identity_violations"]["patterns"])      # never claims to be a real person
    assert any("按现在的阶段来" in line for line in d["avoid"])


def test_the_quiet_one_is_possessive_but_never_threatens_or_isolates():
    from ai.settings import PROJECT_ROOT
    d = json.loads((PROJECT_ROOT / "persona" / "templates" / "quiet.json").read_text(encoding="utf-8"))
    avoid = "".join(d["avoid"])
    for needle in ("伤害自己", "疏远", "不查看、不删改", "道歉并停下来"):
        assert needle in avoid, needle
    lines = "".join(d["examples"] + d["communication_habits"] + [s["text"] for s in d["stages"]])
    for bad in ("不知道会做什么", "不需要别人", "删掉", "跟踪"):
        assert bad not in lines, bad


# ------------------------------------------------------------------------------------------------ relationship + stage
def test_a_persona_with_its_own_relationship_replaces_the_choice(db, data_dir):
    from ai import about_owner
    from ai import persona_templates as pt
    pt.set_relationship("daughter", "老爸")
    pt.switch(db, "stepsister", "keep")
    rel = pt.relationship()
    assert rel == {"kind": "persona", "label": "继妹", "call": "哥"}        # not 女儿 / 老爸 from before
    about_owner.save({"gender": "female"})
    assert pt.relationship()["call"] == "姐"                               # by the owner's gender
    assert pt.set_relationship("persona", "姐")["call"] == "姐"            # the default is not stored ...
    about_owner.save({"gender": "male"})
    assert pt.relationship()["call"] == "哥"                               # ... so it still follows the form
    assert pt.set_relationship("persona", "哥哥")["call"] == "哥哥"        # the owner's own choice is kept
    text = _context(db)
    assert "【你们的关系：继妹】" in text and "没有血缘的继妹" in text and "你称呼对方“哥哥”" in text
    assert "{call}" not in text and "你和对方的过去" in text and "口吻示例" in text and "不要这样" in text
    assert "不要编造家人、工作、学校、恋爱等背景" not in text            # her family and school are hers
    assert "还不太熟" not in text and "互相熟悉" not in text
    # back to a persona without one: an ordinary relationship again
    pt.switch(db, "violet", "keep")
    assert pt.relationship()["kind"] == "close"


def test_the_stage_moves_with_the_days_talked_and_can_be_fixed(db, data_dir):
    from ai import persona_templates as pt
    pt.switch(db, "childhood", "keep")
    assert pt.stage(None, db)["index"] == 0 and pt.stage(None, db)["auto"]
    assert "阶段（1/5，死党日常）" in _context(db)
    for day in range(1, 10):                                              # nine days with a message each
        db.add_message("user", "incoming", "text", "在吗", created_at=f"2026-10-{day:02d} 20:00:00")
    now = pt.stage(None, db)
    assert (now["index"], now["label"], now["days"], now["next_days"]) == (2, "吃醋", 9, 16)
    assert "阶段（3/5，吃醋）" in _context(db)
    pt.set_stage(None, 4)
    now = pt.stage(None, db)
    assert now["index"] == 4 and not now["auto"] and "阶段（5/5，说开）" in _context(db)
    pt.set_stage(None, None)
    assert pt.stage(None, db)["index"] == 2
    with pytest.raises(ValueError):
        pt.set_stage(None, 99)
    pt.switch(db, "violet", "keep")
    assert pt.stage(None, db) is None                                     # no stages without such a persona


def test_a_channel_with_its_own_such_persona_and_the_api(admin, db, tidy):
    from ai import persona_templates as pt
    r = admin.post("/api/persona/channel", json={"channel": "wechat", "template": "junior", "kind": "partner"}, headers=SAME)
    row = next(c for c in r.get_json()["channels"] if c["id"] == "wechat")
    assert row["in_effect"]["kind"] == "persona" and row["in_effect"]["label"] == "后辈" and row["in_effect"]["call"] == "前辈"
    assert row["in_effect"]["stage"]["labels"][0] == "被带的新人" and row["in_effect"]["stage"]["auto"]
    r = admin.post("/api/persona/stage", json={"channel": "wechat", "stage": 3}, headers=SAME)
    assert r.status_code == 200 and r.get_json()["in_effect"]["stage"]["index"] == 3
    assert pt.channel_map()["wechat"]["stage"] == 3 and "stage" not in pt._info()     # the channel's, not the main one's
    assert admin.post("/api/persona/stage", json={"channel": "wechat", "stage": "x"}, headers=SAME).status_code == 400
    d = admin.get("/api/persona/templates", headers=SAME).get_json()
    card = next(t for t in d["templates"] if t["key"] == "junior")
    assert card["relation"] == "后辈" and len(card["stages"]) == 5 and "{call}" not in json.dumps(card, ensure_ascii=False)
    assert d["in_effect"]["kind"] == "close"                              # the main persona is still Violet


def test_renaming_her_changes_the_name_inside_her_texts(db, data_dir):
    from ai import persona_templates as pt
    from ai.persona import PersonaCore
    pt.switch(db, "quiet", "keep", name="小唯")
    core = PersonaCore.load(channel="wechat")
    assert core.data["identity"]["name"] == "小唯" and "朝雾唯" not in json.dumps(core.data, ensure_ascii=False)
    assert "小唯是一个虚构的女孩" in core.render()


# ------------------------------------------------------------------------------------------------ the reply check
def test_her_own_family_and_school_are_not_an_invented_background(db, data_dir):
    from ai import persona_templates as pt
    from ai.consistency import ConsistencyGuard
    from ai.persona import PersonaCore
    pt.switch(db, "stepsister", "keep")
    guard = ConsistencyGuard(db, PersonaCore.load(channel="wechat"))
    for fine in ("我妈让我叫你回家吃饭", "我们学校今天运动会", "我同学都说你对我太好了", "我哥才不会这样"):
        assert not guard._persona_violations(fine), fine
    for wrong in ("我男朋友也这么说", "我要加班", "我是真人"):
        assert guard._persona_violations(wrong), wrong
    pt.switch(db, "violet", "keep")                                       # the default persona: as before
    guard = ConsistencyGuard(db, PersonaCore.load(channel="wechat"))
    assert guard._persona_violations("我妈让我早点睡") and guard._persona_violations("我们英语老师要是也这样就好了")


def test_in_a_group_her_past_is_with_the_owner_not_the_speaker(db, data_dir):
    from ai import persona_templates as pt
    from ai.social.engine import _persona_text
    pt.switch(db, "childhood", "keep")
    text = _persona_text(db, "napcat")
    assert "【你和你的主人的关系：青梅竹马】" in text and "你和主人从小一起长大" in text
    assert "你和主人的过去" in text and "对方" not in text


def test_the_page_has_the_form_and_the_stage_picker():
    from ai.settings import PROJECT_ROOT
    html = (PROJECT_ROOT / "wx_ilink_py" / "templates" / "control.html").read_text(encoding="utf-8")
    js = (PROJECT_ROOT / "wx_ilink_py" / "static" / "control_persona.js").read_text(encoding="utf-8")
    assert html.index('id="pl-about"') < html.index('id="pl-who"')
    for needle in ("function renderAbout(", '"/api/persona/about"', "function stageSelect(", '"/api/persona/stage"',
                   'kind: builtIn ? "persona"'):
        assert needle in js, needle
