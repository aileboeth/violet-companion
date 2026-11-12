"""What the owner filled in about themselves on the persona page (2026-11-12, user: "开局要先收集一些用户的基本信息，
比如说名字、性别、年龄、爱好和职务…这样就不会一来就问你叫什么").

  <data>/control/about-owner.json   {name, gender, birthday, status, school, grade, major, job, hobbies, city, note}

One file for every persona (the profile facts she learns in chat are per persona and are cleared with her memory;
this is the owner's own form and stays). Every field is optional. It is shown to her on every owner turn, the
"慢慢认识你" questions skip what is answered here, and the birthday greeting reads the birthday.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

FIELDS = {"name": 20, "gender": 8, "birthday": 10, "status": 8, "school": 40, "grade": 20, "major": 40, "job": 40,
          "hobbies": 120, "city": 30, "note": 200}
GENDERS = {"male": "男", "female": "女", "other": "其他"}
STATUSES = {"student": "学生", "working": "已经工作", "other": "其他"}
_BIRTHDAY_RE = re.compile(r"^(?:(\d{4})-)?(\d{1,2})-(\d{1,2})$")
HEADER = "【对方的基本情况（对方自己填的；对方后来在聊天里说的不一样时，以后来说的为准）】"


def _path():
    from ai.persona_templates import control_dir
    return control_dir() / "about-owner.json"


def load() -> dict[str, str]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data = data if isinstance(data, dict) else {}
    return {k: str(data.get(k) or "") for k in FIELDS}


def clean(fields: dict[str, Any]) -> dict[str, str]:
    """Validated values; ValueError with a message for the page."""
    out = {}
    for key, limit in FIELDS.items():
        value = " ".join(str(fields.get(key) or "").split())
        if len(value) > limit:
            raise ValueError(f"{key} 太长（最多 {limit} 个字）")
        out[key] = value
    if out["gender"] and out["gender"] not in GENDERS:
        raise ValueError("性别不对")
    if out["status"] and out["status"] not in STATUSES:
        raise ValueError("身份不对")
    if out["birthday"]:
        match = _BIRTHDAY_RE.match(out["birthday"])
        try:
            date(int(match.group(1) or 2000), int(match.group(2)), int(match.group(3)))
        except (AttributeError, ValueError):
            raise ValueError("生日写成 2005-03-05 或 03-05") from None
        year = f"{int(match.group(1)):04d}-" if match.group(1) else ""
        out["birthday"] = f"{year}{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    return out


def save(fields: dict[str, Any]) -> dict[str, str]:
    from ai.persona_templates import _write
    data = clean(fields)
    _write(_path(), data)
    return data


def birthday(data: dict[str, str] | None = None) -> tuple[int, int] | None:
    """(month, day) for the birthday greeting."""
    match = _BIRTHDAY_RE.match((data or load())["birthday"])
    return (int(match.group(2)), int(match.group(3))) if match else None


def age(today: date, data: dict[str, str] | None = None) -> int | None:
    match = _BIRTHDAY_RE.match((data or load())["birthday"])
    if not match or not match.group(1):
        return None
    years = today.year - int(match.group(1)) - ((today.month, today.day) < (int(match.group(2)), int(match.group(3))))
    return years if 0 < years < 120 else None


def lines(today: date | None = None) -> list[str]:
    """The lines of the prompt section; [] when nothing was filled in."""
    d = load()
    out = []
    if d["name"]:
        out.append(f"- 名字：{d['name']}")
    if d["gender"]:
        out.append(f"- 性别：{GENDERS[d['gender']]}")
    if d["birthday"]:
        month, day = birthday(d)
        years = age(today, d) if today else None
        out.append(f"- 生日：{month} 月 {day} 日" + (f"（今年 {years} 岁）" if years else ""))
    study = "，".join(x for x in (d["school"], d["grade"], d["major"]) if x)
    if d["status"] == "student" or (not d["status"] and study):
        out.append("- 身份：学生" + (f"（{study}）" if study else ""))
    elif d["status"] == "working" or (not d["status"] and d["job"]):
        out.append("- 身份：已经工作" + (f"（{d['job']}）" if d["job"] else ""))
    elif d["status"] == "other" and (d["job"] or study):
        out.append(f"- 身份：{d['job'] or study}")
    if d["hobbies"]:
        out.append(f"- 爱好：{d['hobbies']}")
    if d["city"]:
        out.append(f"- 所在城市：{d['city']}")
    if d["note"]:
        out.append(f"- 其他：{d['note']}")
    return out


def known_keys() -> set[str]:
    """Profile keys (ai/profile_schema) the form already answers: the onboarding questions skip them."""
    d = load()
    keys = set()
    if d["name"]:
        keys.add("address_name")
    if d["status"] or d["school"] or d["grade"] or d["job"]:
        keys.add("identity.status")
    if d["hobbies"]:
        keys.add("interest")
    return keys


def filled() -> int:
    return sum(1 for v in load().values() if v)
