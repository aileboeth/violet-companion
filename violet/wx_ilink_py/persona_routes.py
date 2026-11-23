"""Dashboard API for Violet's persona and virtual life (phase 4).

Behind the same admin authentication as every other route (wx_ilink_py/auth.py).
The persona core is read-only here on purpose: it is changed only by editing
persona/persona_core.json by hand (and bumping its version).
"""

from __future__ import annotations

from flask import Flask, jsonify, redirect, request

from ai.consistency import ConsistencyGuard
from ai.life_store import LifeStore, TransitionError
from ai.life_scheduler import LifeControl
from ai.persona import PersonaCore, PersonaDynamics, persona_dir

CORE_EDIT_NOTE = "核心人设只能人工编辑 persona/persona_core.json 并递增 version；系统和 Dashboard 都不能修改。"


def register_persona_routes(app: Flask, violetdb, all_agents=None) -> None:
    store = LifeStore(violetdb)

    def ok(payload, status=200):
        return jsonify(payload), status

    def bad(message, status=400):
        return jsonify({"error": message}), status

    def body() -> dict:
        return request.get_json(silent=True) or {}

    @app.get("/persona")
    def persona_page():
        # Violet stage 3: the page is part of the Control Center now (same look, same navigation)
        return redirect("/control#persona")

    # -- persona -----------------------------------------------------------------------
    @app.get("/api/persona")
    def persona_overview():
        core = PersonaCore.load(violetdb)
        dynamics = PersonaDynamics(violetdb)
        dynamics.active()       # expires temporary interests
        return ok({
            "core": {"version": core.version, "sha256": core.sha256, "data": core.data,
                     "file": str(persona_dir() / "persona_core.json"), "edit_policy": CORE_EDIT_NOTE},
            "core_versions": violetdb.fetch_all(
                "SELECT id, version, sha256, loaded_at FROM persona_core_versions ORDER BY id DESC LIMIT 20"),
            "dynamics": dynamics.list(),
        })

    @app.route("/api/persona/core", methods=["PUT", "PATCH", "POST", "DELETE"])
    def persona_core_write():
        return bad(CORE_EDIT_NOTE, 405)

    # -- 2026-10-29: persona templates (incl. the blank one); the memory is kept or cleared (archived first) -------------
    @app.get("/api/persona/templates")
    def persona_templates():
        from ai import persona_templates as pt
        from ai import test_mode
        return ok({"templates": pt.templates(), "current": pt.current(), "custom": pt.custom_preview(),
                   "test_mode": test_mode.status(),
                   # 2026-10-30: the second layer, and per channel
                   "relationship": pt.relationship(), "in_effect": _in_effect(pt, None),
                   "about": _about(),                # 2026-11-12: the owner's own form
                   "relationships": [{"key": k, "label": v[0], "text": v[1]} for k, v in pt.RELATIONSHIPS.items()],
                   "channels": (rows := _channel_rows(pt)), "used_by": _used_by(pt, rows),
                   "records": _records(pt)})

    def _records(pt) -> dict[str, int]:
        """0.7.55: template key -> messages in that persona's own records (the switch dialog says she goes on with
        hers instead of taking over the current persona's)."""
        from ai import memory_scope
        files, out = set(violetdb.scope_files()), {}
        for key in [t["key"] for t in pt.templates()] + ["custom"]:
            scope = memory_scope.template_scope(key)
            if scope != memory_scope.MAIN and scope not in files:
                continue
            with memory_scope.use(scope):
                n = violetdb.fetch_one("SELECT COUNT(*) AS n FROM messages")["n"]
            if n:
                out[key] = n
        return out

    # 0.7.55 (user: 导入时选了人设，人设页却没变，两边同步一下): every channel the 渠道 page can add, also one an import
    # set a persona for before it was added
    KINDS = {"napcat": "QQ", "qq": "QQ（机器人）", "telegram": "Telegram"}

    def _channel_rows(pt) -> list[dict]:
        from ai.channels.manager import ChannelStore
        stored = ChannelStore().channels()
        names = {"wechat": (stored.get("wechat") or {}).get("label") or "微信"}
        for cid, entry in stored.items():
            if cid != "wechat" and isinstance(entry, dict):
                names[cid] = entry.get("label") or cid
        own = pt.channel_map()
        for cid in own:
            if cid not in names:
                names[cid] = f"{KINDS.get(cid, cid)}（还没添加）"
        return [{"id": cid, "label": label, "template": (own.get(cid) or {}).get("template"),
                 "relationship": (own.get(cid) or {}).get("relationship"), "in_effect": _in_effect(pt, cid)}
                for cid, label in names.items()]

    def _in_effect(pt, channel) -> dict:
        """What a channel uses now; a persona with stages: where it stands, counted in that persona's own records."""
        from ai import memory_scope
        with memory_scope.use(memory_scope.for_channel(channel) if channel else memory_scope.MAIN):
            return pt.in_effect(channel, violetdb)

    def _about() -> dict:
        from ai import about_owner
        return {"fields": about_owner.load(), "genders": about_owner.GENDERS, "statuses": about_owner.STATUSES}

    @app.get("/api/persona/about")
    def persona_about():
        return ok(_about())

    @app.put("/api/persona/about")
    def persona_about_save():
        from ai import about_owner
        try:
            about_owner.save(body())
        except ValueError as exc:
            return bad(str(exc))
        return ok(_about())

    @app.post("/api/persona/stage")
    def persona_stage():
        """{channel: id | null (the main persona), stage: 0-based number | null (automatic)}"""
        from ai import persona_templates as pt
        data = body()
        try:
            pt.set_stage(str(data.get("channel") or "") or None, data.get("stage"))
        except ValueError as exc:
            return bad(str(exc))
        return ok({"in_effect": _in_effect(pt, str(data.get("channel") or "") or None)})

    @app.post("/api/persona/distance/clear")
    def persona_distance_clear():
        """{channel: id | null (the main persona)} -- lift an active 'distance' relationship lock."""
        from ai import memory_scope, persona_templates as pt
        from ai.relationship import clear_distance
        data = body()
        channel = str(data.get("channel") or "") or None
        scope = memory_scope.for_channel(channel) if channel else memory_scope.MAIN
        with memory_scope.use(scope):
            cleared_at = clear_distance(violetdb)
        return ok({"cleared_at": cleared_at, "in_effect": _in_effect(pt, channel)})

    def _used_by(pt, rows: list[dict]) -> dict[str, list[str]]:
        """template key -> the channels talking as that persona (their own choice, else the main persona)."""
        main = pt.current()["template"]
        out: dict[str, list[str]] = {}
        for r in rows:
            out.setdefault(r["template"] or main, []).append(r["label"])
        return out

    @app.post("/api/persona/relationship")
    def persona_relationship():
        from ai import persona_templates as pt
        data = body()
        try:
            # 2026-11-12: "persona" = the persona's own relationship, only the call is kept
            return ok(pt.set_relationship(str(data.get("kind") or ""), data.get("call")))
        except ValueError as exc:
            return bad(str(exc))

    @app.post("/api/persona/channel")
    def persona_channel():
        """{channel, template: key | "custom" | null (the main one), kind: relationship | null, call}"""
        from ai import persona_templates as pt
        data = body()
        try:
            pt.set_channel(str(data.get("channel") or ""), data.get("template") or None, data.get("kind") or None,
                           data.get("call"))
        except ValueError as exc:
            return bad(str(exc))
        from ai import memory_scope                      # 2026-11-02: its chat so far goes with its first own persona
        moved = memory_scope.split_channels(violetdb)
        return ok({"channels": _channel_rows(pt), "moved": moved})

    # -- 2026-10-30: test mode (ai/test_mode.py) -- on top of the persona, which stays as it is ------------------------
    @app.post("/api/persona/test-mode")
    def persona_test_mode():
        from ai import test_mode
        data = body()
        if data.get("on") is True:
            try:
                return ok(test_mode.enable(violetdb, data.get("hours", test_mode.DEFAULT_HOURS)))
            except ValueError as exc:
                return bad(str(exc))
        if data.get("on") is False:
            return ok(test_mode.disable())
        return bad("on 要是 true 或 false")

    @app.post("/api/persona/custom")
    def persona_custom():
        from ai import persona_templates as pt
        data = body()
        if data.get("confirm") is not True:
            return bad("换人设需要确认（confirm: true）")
        fields = data.get("persona") if isinstance(data.get("persona"), dict) else {}
        try:
            return ok(pt.custom(violetdb, fields, str(data.get("memory") or "")))
        except ValueError as exc:
            return bad(str(exc))

    @app.post("/api/persona/switch")
    def persona_switch():
        from ai import persona_templates as pt
        data = body()
        if data.get("confirm") is not True:
            return bad("换人设需要确认（confirm: true）")
        try:
            result = pt.switch(violetdb, str(data.get("template") or ""), str(data.get("memory") or ""),
                               str(data.get("name") or "") or None)
        except ValueError as exc:
            return bad(str(exc))
        return ok(result)

    @app.post("/api/persona/sandbox")
    def persona_sandbox():
        """Stateless in-memory persona chat preview: zero DB writes, zero memory/mood/channel side effects."""
        import json as _json
        import time as _time
        from ai import about_owner, persona_templates as pt
        from ai.agents.base_agent import LLMNotConfigured
        from ai.agents.chat_agent import AI_PROMPT_FILE, ChatAgent, OWN_PROMPT_FILE, compose
        from ai.chat_output import normalize_chat_output
        from ai.context_engine import ai_natural
        from ai.settings import PROJECT_ROOT

        data = body()
        message = str(data.get("message") or "").strip()
        if not message or len(message) > 1000:
            return bad("请输入试聊消息（1 到 1000 字）")

        tpl_key = str(data.get("template") or "").strip()
        pdata = None
        if not tpl_key:
            pdata = PersonaCore.load().data
        elif tpl_key == "custom":
            pdata = pt.custom_saved()
            if not pdata:
                return bad("还没有保存过自定义人设")
        else:
            if not tpl_key.replace("-", "").replace("_", "").isalnum():
                return bad("没有这个人设模板")
            tpath = pt.TEMPLATE_DIR / f"{tpl_key}.json"
            if not tpath.is_file():
                return bad("没有这个人设模板")
            try:
                pdata = _json.loads(tpath.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return bad("人设模板文件读取失败", 500)

        stages = [s for s in (pdata.get("stages") or []) if isinstance(s, dict) and s.get("label")]
        stage_req = data.get("stage")
        stage_idx: int | None = None
        if stage_req is not None and stage_req != "":
            if isinstance(stage_req, bool) or not isinstance(stage_req, int) or not 0 <= stage_req < len(stages):
                return bad("阶段编号不对")
            stage_idx = stage_req
        elif stages:
            if not tpl_key or tpl_key == pt.current().get("template"):
                cur_stage = pt.stage(None, violetdb)
                stage_idx = cur_stage["index"] if cur_stage else 0
            else:
                stage_idx = 0

        core = PersonaCore(pdata, "sandbox")
        ident = pdata.get("identity") or {}
        title = str(pdata.get("_title") or ident.get("name") or "人设")
        own_rel = pt.own_relation(pdata)
        call = pt.owner_call(pdata) if own_rel else (pt.relationship().get("call") or "")
        call_line = f"你称呼对方“{call}”（偶尔叫，不要句句带）。" if call else ""
        stage_label = ""
        stage_line = ""
        if stages and stage_idx is not None:
            st = stages[stage_idx]
            stage_label = str(st["label"])
            sample_raw = str(st.get("sample") or "")
            sample_hint = (f"这一阶段语气感受（只拿捏亲疏感，严禁照抄原句）：「{pt._fill(sample_raw, pdata, call)}」"
                           if sample_raw else "")
            stage_line = (f"\n- 你们现在的阶段（{stage_idx + 1}/{len(stages)}，{stage_label}）："
                          f"{pt._fill(str(st.get('text') or ''), pdata, call)}{sample_hint}"
                          "阶段是慢慢变的：不要突然像换了一个人，也不要抢在阶段前面。")
        if own_rel:
            rel_label = str(own_rel.get("label") or "")
            rel_text = pt._fill(str(own_rel.get("text") or ""), pdata, call) + call_line + stage_line
        else:
            rel = pt.relationship()
            rel_label = rel["label"]
            rel_text = pt.RELATIONSHIPS.get(rel["kind"], pt.RELATIONSHIPS["close"])[1] + call_line + stage_line
        rel_section = f"【你们的关系：{rel_label}】{rel_text}"

        chat_agent = next((a for a in (all_agents or []) if isinstance(a, ChatAgent)), None)
        if chat_agent is None:
            chat_agent = ChatAgent()
        cfg = chat_agent._config
        if core.is_ai and ai_natural():
            try:
                base_prompt = (PROJECT_ROOT / AI_PROMPT_FILE).read_text(encoding="utf-8").strip()
            except OSError:
                base_prompt = cfg.reply_prompt
        elif pdata.get("prompt_text"):
            try:
                base_prompt = (PROJECT_ROOT / OWN_PROMPT_FILE).read_text(encoding="utf-8").strip()
            except OSError:
                base_prompt = cfg.reply_prompt
        else:
            base_prompt = cfg.reply_prompt

        stable = [base_prompt, core.render(), rel_section]
        history_in = data.get("history") if isinstance(data.get("history"), list) else []
        history: list[dict[str, str]] = []
        for item in history_in[-12:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = str(item.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                history.append({"role": role, "content": content[:1000]})
        notes: list[str] = []
        owner_lines = about_owner.lines()
        if owner_lines:
            notes.append(owner_lines)
        notes.append(f"现在是 {violetdb.now()[:16].replace('T', ' ')}（试聊预览，不保存记录）。")

        messages = compose(stable, history, notes, message)
        prompt_parts = []
        for m in messages:
            text = str(m.get("content") or "")
            first = text.strip().splitlines()[0][:60] if text.strip() else ""
            prompt_parts.append({"role": m.get("role"), "title": first, "chars": len(text)})

        started = _time.monotonic()
        try:
            client = chat_agent._get_client(cfg)
            resp = chat_agent._create_completion_with_retry(
                client,
                model=cfg.reply_model,
                messages=messages,
                temperature=cfg.reply_temperature,
                cache_key=f"violet-sandbox-{tpl_key or 'active'}",
            )
        except LLMNotConfigured:
            return bad("还没有设置模型服务的 API Key（请先在模型服务页配置）")
        except Exception as exc:
            return bad(f"试聊调用失败（{exc.__class__.__name__}）", 502)
        latency_ms = int((_time.monotonic() - started) * 1000)

        raw = ""
        try:
            raw = str((resp.choices[0].message.content if resp.choices else "") or "").strip()
        except Exception:
            raw = ""
        normalized = normalize_chat_output(raw, sticker_for_label=None, sticker_for_description=None)
        out_lines = [str(m.get("content") or "") for m in normalized.get("messages") or [] if m.get("content")]
        if not out_lines and raw:
            out_lines = [raw]
        usage_obj = getattr(resp, "usage", None)
        usage = None
        if usage_obj is not None:
            from ai import usage as usage_ledger
            pt, ct, cached = usage_ledger.numbers(usage_obj)
            usage = {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "cached_tokens": cached,
            }
        return ok({
            "messages": out_lines,
            "raw": raw,
            "model": cfg.reply_model,
            "temperature": cfg.reply_temperature,
            "latency_ms": latency_ms,
            "persona_title": title,
            "stage_label": stage_label,
            "stage_index": stage_idx,
            "prompt_chars": sum(p["chars"] for p in prompt_parts),
            "prompt_parts": prompt_parts,
            "usage": usage,
        })

    @app.post("/api/persona/dynamics/<int:entry_id>/decide")
    def persona_decide(entry_id: int):
        approve = body().get("approve")
        if not isinstance(approve, bool):
            return bad("approve must be true or false")
        try:
            return ok(PersonaDynamics(violetdb).decide(entry_id, approve))
        except ValueError as exc:
            return bad(str(exc))

    @app.get("/api/persona/dynamics/<int:entry_id>/history")
    def persona_history(entry_id: int):
        entry = PersonaDynamics(violetdb).get(entry_id)
        if not entry:
            return bad("not found", 404)
        return ok({"entry": entry, "revisions": store.revisions("persona_dynamic", entry_id)})

    # -- life ----------------------------------------------------------------------------
    @app.get("/api/life")
    def life_overview():
        return ok({
            "threads": store.threads(),
            "plans": store.plans(),
            "events": store.events(statuses=None, limit=200),
            "flags": store.flags("open"),
            "runs": violetdb.fetch_all("SELECT * FROM life_engine_runs ORDER BY id DESC LIMIT 30"),
            "engine": LifeControl(violetdb).status(),
            "llm_usage": violetdb.fetch_all(
                "SELECT purpose, status, COUNT(*) AS calls, SUM(total_tokens) AS tokens FROM aux_llm_calls "
                "WHERE created_at >= ? GROUP BY purpose, status", (violetdb.now()[:10],)),
        })

    @app.get("/api/life/engine")
    def life_engine_status():
        return ok(LifeControl(violetdb).status())

    @app.post("/api/life/engine/<action>")
    def life_engine_action(action: str):
        if action not in ("pause", "resume"):
            return bad("unknown action", 404)
        control = LifeControl(violetdb)
        if control.mode() == "off":
            return bad("life engine is off (VIOLET_LIFE_ENGINE=0 or life_config.enabled=false)", 409)
        return ok({"mode": control.set_paused(action == "pause", "dashboard", body().get("reason"))})

    @app.post("/api/life/threads/<int:thread_id>/<action>")
    def life_thread_action(thread_id: int, action: str):
        targets = {"pause": "paused", "resume": "active", "complete": "completed", "abandon": "abandoned",
                   "cancel": "cancelled"}
        if action not in targets:
            return bad("unknown action", 404)
        try:
            return ok(store.set_thread_status(thread_id, targets[action], body().get("reason") or "dashboard", "dashboard"))
        except TransitionError as exc:
            return bad(str(exc))

    @app.post("/api/life/plans/<int:plan_id>/<action>")
    def life_plan_action(plan_id: int, action: str):
        targets = {"pause": "paused", "resume": "planned", "cancel": "cancelled", "abandon": "abandoned"}
        if action not in targets:      # completing needs a recorded event: not offered here
            return bad("unknown action", 404)
        try:
            return ok(store.set_plan_status(plan_id, targets[action], body().get("reason") or "dashboard", "dashboard"))
        except TransitionError as exc:
            return bad(str(exc))

    @app.patch("/api/life/events/<int:event_id>")
    def life_event_correct(event_id: int):
        data = body()
        content, reason = (data.get("content") or "").strip(), (data.get("reason") or "").strip()
        if not content or not reason:
            return bad("content and reason are required")
        violations = ConsistencyGuard(violetdb)._persona_violations(content)
        if violations:
            return bad("correction conflicts with the persona core: " + ", ".join(v.check for v in violations))
        try:
            return ok(store.correct_event(event_id, content, reason))
        except TransitionError as exc:
            return bad(str(exc))

    @app.post("/api/life/events/<int:event_id>/retract")
    def life_event_retract(event_id: int):
        reason = (body().get("reason") or "").strip()
        if not reason:
            return bad("reason required")
        try:
            return ok(store.retract_event(event_id, reason))
        except TransitionError as exc:
            return bad(str(exc))

    @app.get("/api/life/<entity>/<int:entity_id>/history")
    def life_history(entity: str, entity_id: int):
        getters = {"thread": store.thread, "plan": store.plan, "event": store.event}
        if entity not in getters:
            return bad("unknown entity", 404)
        row = getters[entity](entity_id)
        if not row:
            return bad("not found", 404)
        return ok({"entity": row, "revisions": store.revisions(entity, entity_id)})

    @app.post("/api/life/flags/<int:flag_id>/resolve")
    def life_flag_resolve(flag_id: int):
        store.resolve_flag(flag_id)
        return ok({"ok": True})
