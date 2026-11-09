"""2026-11-09 (user: 写入 / 高风险 有什么区别，既然不能操作手机为什么要写入).

* write and high_risk are handled alike everywhere (ask first, or "允许直接用"); the page offers 直接用 / 先问你 /
  不给她用 and the phone's read-only tools get no choice at all;
* "会改动东西的工具" offered 自己做 / 先问你 / 不允许, but 自己做 behaved like 不允许 (only "ask" turns an unallowed
  tool on). Now the choice is 先问你 / 不允许, and a saved 自己做 reads as what it did: 不允许.
"""

import json
from pathlib import Path

import pytest

from ai import mcp, permissions

WEB = Path(__file__).resolve().parents[1] / "wx_ilink_py"


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("VIOLET_DATA_DIR", str(tmp_path))
    (tmp_path / "control").mkdir()
    return tmp_path


def test_write_and_high_risk_are_the_same_gate(monkeypatch):
    monkeypatch.delenv("MCP_ALLOWED_RISKY_TOOLS", raising=False)
    server = {"name": "s", "allow_tools": ["b"]}
    for level in ("write", "high_risk"):
        assert mcp.is_tool_allowed(server, "s", "a", level) is False
        assert mcp.is_tool_allowed(server, "s", "b", level) is True
    assert mcp.is_tool_allowed(server, "s", "a", "read") is True


def test_risky_tools_offers_ask_and_forbid_only(data):
    rows = {r["key"]: r for r in permissions.describe()}
    assert rows["risky_tools"]["choices"] == ["ask", "forbid"]
    assert rows["qzone_post"]["choices"] == ["self", "ask", "forbid"]
    assert "手机 App 的工具都只能查看" in rows["risky_tools"]["help"]
    with pytest.raises(ValueError):
        permissions.set_level("risky_tools", "self")
    assert permissions.set_level("risky_tools", "forbid")["risky_tools"] == "forbid"


def test_a_saved_self_reads_as_what_it_did(data):
    (data / "control" / "permissions.json").write_text(json.dumps({"risky_tools": "self"}), encoding="utf-8")
    assert permissions.level("risky_tools") == "forbid"
    server = {"name": "s"}
    assert permissions.tool_needs_approval(server, "s", "t", "write") is False      # same as before: not offered


def test_the_page_offers_three_plain_choices():
    js = (WEB / "static" / "control_mcp.js").read_text(encoding="utf-8")
    assert "直接用（只是查看）" in js and "先问你（会改动东西）" in js and "不给她用" in js
    assert "写入：" not in js and "高风险：" not in js
    assert 's.name === "android-phone" && t.permission === "read" && !t.admin_level' in js   # no choice for the phone
    control = (WEB / "static" / "control.js").read_text(encoding="utf-8")
    assert "a.choices.includes(v)" in control
