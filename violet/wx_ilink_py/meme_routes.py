"""Control Center routes of the memes she gets (2026-11-12, ai/memes.py). Behind the admin authentication like every
route; "your memes" follow the page's persona picker (each persona has her own).

  GET    /api/control/memes                 the switch, the wiki file, the download, your memes
  POST   /api/control/memes/download        {confirm: true}: download / update the wiki from cnmeme.wiki (background)
  POST   /api/control/memes/stop            stop the download (what is done stays)
  POST   /api/control/memes/import          the body is a meme file (SQLite, the DSH plugin's format): merged in
  DELETE /api/control/memes/wiki            delete the wiki file
  GET    /api/control/memes/lookup?q=       what she would find for a word
  POST   /api/control/memes/own             {phrase, meaning}: one of your memes
  POST   /api/control/memes/own/<id>        {phrase, meaning}; DELETE .../own/<id>
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

MAX_IMPORT = 64 * 1024 * 1024


def register_meme_routes(app: Flask, violetdb) -> None:
    from ai import memes

    def ok(data: Any = None) -> Response:
        return jsonify({"ok": True, "data": data})

    def bad(message: str, status: int = 400) -> Response:
        return jsonify({"ok": False, "error": message}), status

    @app.get("/api/control/memes")
    def control_memes():
        return ok({"enabled": memes.enabled(), "wiki": memes.wiki_info(), "sync": memes.status(),
                   "own": memes.own_list(violetdb), "site": memes.WIKI_SITE})

    @app.post("/api/control/memes/download")
    def control_memes_download():
        if (request.get_json(silent=True) or {}).get("confirm") is not True:
            return bad("要确认")
        try:
            return ok(memes.start())
        except ValueError as exc:
            return bad(str(exc), 409)

    @app.post("/api/control/memes/stop")
    def control_memes_stop():
        memes.stop()
        return ok(memes.status())

    @app.post("/api/control/memes/import")
    def control_memes_import():
        if memes.status()["running"]:
            return bad("正在下载，等下载完再导入", 409)
        if (request.content_length or 0) > MAX_IMPORT:
            return bad("文件太大了（最多 64 MB）", 413)
        data = request.stream.read(MAX_IMPORT + 1)
        if len(data) > MAX_IMPORT:
            return bad("文件太大了（最多 64 MB）", 413)
        if not data.startswith(b"SQLite format 3\x00"):
            return bad("不是词库文件（要 SQLite 的 .db 文件，例如 dsh-memewiki 的 data/memes.db）")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "import.db"
            path.write_bytes(data)
            try:
                return ok(memes.import_file(path))
            except ValueError as exc:
                return bad(str(exc))

    @app.delete("/api/control/memes/wiki")
    def control_memes_delete_wiki():
        if memes.status()["running"]:
            return bad("正在下载，先停止", 409)
        return ok({"deleted": memes.delete_wiki()})

    @app.get("/api/control/memes/lookup")
    def control_memes_lookup():
        found = memes.lookup(violetdb, request.args.get("q") or "")
        return ok({"found": [{"title": f["title"], "kind": f["kind"], "text": memes.describe(f, short=False)}
                             for f in found]})

    @app.post("/api/control/memes/own")
    def control_memes_own_add():
        body = request.get_json(silent=True) or {}
        try:
            return ok(memes.own_save(violetdb, body.get("phrase"), body.get("meaning"), source="page"))
        except ValueError as exc:
            return bad(str(exc))

    @app.post("/api/control/memes/own/<int:meme_id>")
    def control_memes_own_update(meme_id: int):
        body = request.get_json(silent=True) or {}
        try:
            row = memes.own_update(violetdb, meme_id, body.get("phrase"), body.get("meaning"))
        except ValueError as exc:
            return bad(str(exc))
        return ok(row) if row else bad("没有这个梗", 404)

    @app.delete("/api/control/memes/own/<int:meme_id>")
    def control_memes_own_delete(meme_id: int):
        return ok({"deleted": meme_id}) if memes.own_delete(violetdb, meme_id) else bad("没有这个梗", 404)
