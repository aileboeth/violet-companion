"""Regression tests for v0.7.82 Control Center DOM & 9-language i18n fixes (#178).

Covers:
1. Unique DOM IDs in control.html (`#set-hist-list` in `#tab-history` vs `#hist-list` in `#tab-chat`)
   and corresponding selectors in control.js.
2. `ai/story_store.py` UI strings (`STATUS_LABELS` + hidden title `STORY_HIDDEN_TITLE`) registered in
   `tools/i18n_extract.py` `PY_PARTS`, extracted into `source.json`, and translated in all 9 non-zh-CN locales
   with 0 missing keys and 0 `{n}` placeholder mismatches.
3. Story spoiler/label lines in `wx_ilink_py/static/control_persona.js` keeping static labels translatable
   while wrapping only the dynamic Chinese values in `translate: "no"` spans.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
WEB = APP / "wx_ilink_py"
I18N_DIR = WEB / "i18n"

STORY_LABELS = (
    "暗线背景：",
    "当前悬念：",
    "HE 方向：",
    "下轮导演便签：",
    "关键人物与物件：",
    "第 {0} 幕 ·",
    "暴露边界：",
    "你的态度与选择：",
)

STORY_STORE_UI_STRINGS = (
    "暗线埋伏笔",
    "展开中",
    "关键转折",
    "圆满收束",
    "已淡出",
    "未揭晓的心事（待聊天触发）",
)


def test_control_html_has_no_duplicate_ids_and_separates_hist_lists():
    html = (WEB / "templates" / "control.html").read_text(encoding="utf-8")
    ids = re.findall(r'\bid="([^"]+)"', html)
    dups = {k: v for k, v in Counter(ids).items() if v > 1}
    assert dups == {}, f"Duplicate DOM IDs found in control.html: {dups}"

    chat_sec = html[html.index('id="tab-chat"'):html.index('id="tab-memories"')]
    hist_sec = html[html.index('id="tab-history"'):html.index('id="tab-account"')]
    assert 'id="hist-list"' in chat_sec
    assert 'id="set-hist-list"' in hist_sec
    assert 'id="hist-list"' not in hist_sec

    js = (WEB / "static" / "control.js").read_text(encoding="utf-8")
    hist_loader = js[js.index("loaders.history ="):js.index("async function rollback(")]
    assert '$("#set-hist-list")' in hist_loader
    assert '$("#hist-list")' not in hist_loader
    # #tab-chat sidebar must still render into #hist-list
    assert '$("#hist-list").replaceChildren(' in js


def test_story_store_ui_strings_extracted_and_translated_in_all_locales():
    from ai import story_store
    from tools import i18n_extract

    part = i18n_extract.PY_PARTS.get("ai/story_store.py")
    assert part is not None
    assert {"STATUS_LABELS", "STORY_HIDDEN_TITLE"} <= set(part.get("names", ()))
    assert hasattr(story_store, "STORY_HIDDEN_TITLE")
    assert story_store.STORY_HIDDEN_TITLE == "未揭晓的心事（待聊天触发）"

    found = i18n_extract.collect()
    for text in STORY_STORE_UI_STRINGS:
        assert text in found, f"Missing from i18n_extract.collect(): {text}"
        assert "ai/story_store.py" in found[text]
    for label in STORY_LABELS:
        assert label in found, f"Missing split label from i18n_extract.collect(): {label}"
        assert "control_persona.js" in found[label]

    source_json = json.loads((I18N_DIR / "source.json").read_text(encoding="utf-8"))
    assert source_json == found, "wx_ilink_py/i18n/source.json is out of sync with tools.i18n_extract.collect()"

    placeholder_re = re.compile(r"\{\d+\}")
    for lang in i18n_extract.LANGS:
        data = json.loads((I18N_DIR / f"{lang}.json").read_text(encoding="utf-8"))
        missing = i18n_extract.missing(lang, found)
        assert missing == [], f"{lang}.json is missing keys: {missing}"
        for key in (*STORY_STORE_UI_STRINGS, *STORY_LABELS):
            assert data.get(key), f"{lang}.json missing translation for {key!r}"
            if lang != "zh-TW" and key not in ("已淡出",):
                assert data[key] != key or lang == "ja", f"{lang}.json left {key!r} untranslated"
        for k, v in data.items():
            src_ph = set(placeholder_re.findall(k))
            dst_ph = set(placeholder_re.findall(str(v)))
            assert src_ph == dst_ph, f"Placeholder mismatch in {lang}.json for {k!r}: {src_ph} != {dst_ph}"


def test_control_persona_story_labels_not_blocked_by_translate_no():
    js = (WEB / "static" / "control_persona.js").read_text(encoding="utf-8")
    body = js[js.index("async function renderStory("):js.index("function openSandbox(")]

    # None of the 8 mixed label+value patterns should sit inside a single translate:"no" node
    blocked_patterns = (
        r'text:\s*`暗线背景：\$\{[^`]+\}`[^)]*translate:\s*"no"',
        r'text:\s*`当前悬念：\$\{[^`]+\}`[^)]*translate:\s*"no"',
        r'text:\s*`HE 方向：\$\{[^`]+\}`[^)]*translate:\s*"no"',
        r'text:\s*`下轮导演便签：\$\{[^`]+\}`[^)]*translate:\s*"no"',
        r'text:\s*`关键人物与物件：\$\{[^`]+\}`[^)]*translate:\s*"no"',
        r'text:\s*`第 \$\{[^}]+\} 幕 · \$\{[^`]+\}`[^)]*translate:\s*"no"',
        r'text:\s*`暴露边界：\$\{[^`]+\}`[^)]*translate:\s*"no"',
        r'text:\s*`你的态度与选择：\$\{[^`]+\}`[^)]*translate:\s*"no"',
    )
    for pat in blocked_patterns:
        assert not re.search(pat, body), f"Parent translate='no' still blocks label matching {pat}"

    # Each static label must be rendered separately while the dynamic value is wrapped in a translate="no" span
    for label in ("暗线背景：", "当前悬念：", "HE 方向：", "下轮导演便签：", "关键人物与物件：", "暴露边界：", "你的态度与选择："):
        assert re.search(
            rf'h\("p",\s*\{{\s*class:\s*"note"\s*\}},\s*h\("span",\s*\{{\s*text:\s*"{re.escape(label)}"\s*\}}\),\s*" ",\s*h\("span",\s*\{{\s*translate:\s*"no",\s*text:',
            body,
        ), f"Expected translatable label + translate='no' value span for {label!r}"

    assert re.search(
        r'h\("strong",\s*\{\},\s*`第 \$\{a\.act\} 幕 · `,\s*h\("span",\s*\{\s*translate:\s*"no",\s*text:\s*a\.title\s*\}\)\)',
        body,
    ), "Expected `第 ${a.act} 幕 · ` to remain translatable with only a.title in translate='no' span"
