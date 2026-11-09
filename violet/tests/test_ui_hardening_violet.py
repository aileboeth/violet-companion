"""2026-11-09 security pass on the Control Center (the user asked about cookie flags, CSRF, security headers and
template injection; those were already in place -- these are the three small additions).

* scripts only from the page's own files: CSP script-src without 'unsafe-inline', and no page has an inline script
  or an on...= handler that would need it;
* the DOM helper h() refuses link / picture addresses that are not http(s), a path here, a data: picture or blob:;
* (the plain-HTTP redirect on the HTTPS port is in test_tls_server_violet.py).
"""

import re
from pathlib import Path

from wx_ilink_py.auth import CSP

APP = Path(__file__).resolve().parents[1]
WEB = APP / "wx_ilink_py"


def _directive(name):
    for part in CSP.split(";"):
        bits = part.split()
        if bits and bits[0] == name:
            return bits[1:]
    return None


def test_csp_allows_only_our_own_script_files():
    assert _directive("script-src") == ["'self'"]
    assert "'unsafe-eval'" not in CSP
    assert _directive("object-src") == ["'none'"] and _directive("frame-ancestors") == ["'none'"]


def test_no_page_needs_inline_script():
    pages = [WEB / "templates" / "control.html"]
    from wx_ilink_py import auth
    sources = [p.read_text(encoding="utf-8") for p in pages] + [auth._LOGIN_PAGE]
    for text in sources:
        for tag in re.findall(r"<script\b[^>]*>", text, re.I):
            assert "src=" in tag, tag                                  # every script is a file
        assert not re.search(r"\son[a-z]+\s*=", text, re.I)          # no onclick= / onload= ...


def test_scripts_do_not_build_html_from_strings():
    for js in (WEB / "static").glob("*.js"):
        text = js.read_text(encoding="utf-8")
        assert "innerHTML" not in text and "insertAdjacentHTML" not in text and "document.write" not in text, js.name
        assert not re.search(r"\beval\(|new Function\(", text), js.name


def test_dom_helper_refuses_script_urls():
    text = (WEB / "static" / "control.js").read_text(encoding="utf-8")
    guard = re.search(r'k === "href" \|\| k === "src"\) && !/(.+?)/i\.test', text)
    assert guard, "h() must check href / src"
    pattern = re.compile(guard.group(1).replace("(?!\\/)", "(?!/)"), re.I)
    ok = ["https://github.com/aileboeth/violet-companion/releases", "http://example.com", "/api/control/qq/screen",
          "#x", "data:image/png;base64,AAAA", "blob:https://x/1"]
    bad = ["javascript:alert(1)", " JavaScript:alert(1)".strip(), "data:text/html,<script>x</script>", "vbscript:x",
           "//evil.example/x", "file:///etc/passwd"]
    assert all(pattern.match(u) for u in ok)
    assert not any(pattern.match(u) for u in bad)
