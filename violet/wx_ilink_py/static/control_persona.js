// Violet Control Center: persona and virtual life (Violet stage 3; was the separate /persona page). Uses control.js's
// helpers ($, h, api, toast, pill, fmt, when, confirmBox, loaders). Built with DOM nodes only, never parsed as HTML.
"use strict";
(() => {
  const STATUS = {
    active: ["进行中", "ok"], paused: ["已暂停", "warn"], completed: ["已完成", "ok"], abandoned: ["已放弃", ""],
    cancelled: ["已取消", ""], planned: ["计划中", "warn"], happened: ["已发生", "ok"], retracted: ["已撤回", "bad"],
    proposed: ["待批准", "warn"], approved: ["已批准", "ok"], rejected: ["已拒绝", "bad"], open: ["待处理", "warn"],
  };
  const status = (s) => { const [t, k] = STATUS[s] || [s, ""]; return pill(t, k); };
  const ENGINE = { on: ["运行中", "ok"], paused: ["已暂停", "warn"], off: ["已关闭", ""] };
  const day = (s) => (s ? String(s).replace("T", " ").slice(0, 16) : "—");
  const empty = (text) => h("p", { class: "note", text });

  async function act(url, body, done) {
    try { await api("POST", url, body || {}); toast(done || "已完成"); loaders.persona(); }
    catch (e) { toast("操作失败：" + e.message, false); }
  }

  // ---------------------------------------------------------------- history dialog
  let dialog = null;
  function historyDialog() {
    if (dialog) return dialog;
    dialog = h("dialog", { class: "pl-history" }, h("form", { method: "dialog", class: "stack" },
      h("h3", { id: "pl-history-title" }), h("div", { id: "pl-history-list", class: "history" }),
      h("div", { class: "row" }, h("button", { value: "close", class: "ghost", text: "关闭" }))));
    document.body.append(dialog);
    return dialog;
  }
  async function showHistory(title, url) {
    try {
      const r = await api("GET", url);
      const d = historyDialog();
      $("#pl-history-title").textContent = title + " · 修改历史";
      const rows = r.revisions || [];
      $("#pl-history-list").replaceChildren(...(rows.length ? rows.map((v) => {
        const rest = Object.fromEntries(Object.entries(v).filter(([k]) => !["id", "entity", "entity_id", "created_at", "actor", "reason"].includes(k)));
        return h("div", {}, h("div", { text: `${day(v.created_at)} · ${fmt(v.actor)} · ${fmt(v.reason)}` }),
          h("pre", { class: "note", text: JSON.stringify(rest, null, 1) }));
      }) : [empty("没有修改记录")]));
      d.showModal();
    } catch (e) { toast("读取失败：" + e.message, false); }
  }

  // ---------------------------------------------------------------- render
  function listBlock(title, items) {
    if (!items || !items.length) return null;
    return h("div", { class: "pl-list" }, h("div", { class: "k", text: title }),
      h("ul", {}, ...items.map((i) => h("li", { text: typeof i === "string" ? i : (i.label || JSON.stringify(i)) }))));
  }

  function renderCore(p) {
    const d = (p.core && p.core.data) || {};
    $("#pl-core").replaceChildren(
      h("div", { class: "row" }, h("strong", { text: (d.identity && d.identity.name) || (d.blank ? "空白人设（还没有名字）" : "Violet") }),
        pill(`v${fmt(p.core && p.core.version)}`), h("span", { class: "note", text: ((p.core && p.core.sha256) || "").slice(0, 12) })),
      h("p", { text: (d.identity && d.identity.virtual_declaration) || "" }),
      h("div", { class: "pl-grid" }, listBlock("性格", d.personality), listBlock("说话习惯", d.communication_habits),
        listBlock("长期兴趣", d.long_term_interests), listBlock("价值观", d.values), listBlock("缺点", d.flaws),
        listBlock("生活边界", d.living_constraints)),
      h("p", { class: "note", text: (p.core && p.core.edit_policy) || "" }));
  }

  function renderDynamics(rows) {
    $("#pl-dynamics").replaceChildren(...(rows.length ? rows.map((x) => h("div", { class: "card" },
      h("div", { class: "row" }, status(x.status), h("strong", { text: x.aspect_key }), h("span", { class: "note", text: x.category }),
        h("button", { class: "ghost", text: "历史", onclick: () => showHistory(x.aspect_key, `/api/persona/dynamics/${x.id}/history`) })),
      h("p", { text: `现在：${fmt(x.value)}` }),
      x.previous_value ? h("p", { class: "note", text: `之前：${x.previous_value}` }) : null,
      h("p", { class: "note", text: `原因：${fmt(x.reason)}` }),
      x.status === "proposed" ? h("div", { class: "row" },
        h("button", { text: "批准", onclick: () => act(`/api/persona/dynamics/${x.id}/decide`, { approve: true }, "已批准") }),
        h("button", { class: "ghost", text: "拒绝", onclick: () => act(`/api/persona/dynamics/${x.id}/decide`, { approve: false }, "已拒绝") })) : null))
      : [empty("还没有变化")]));
  }

  function renderThreads(rows) {
    $("#pl-threads").replaceChildren(...(rows.length ? rows.map((t) => h("div", { class: "card" },
      h("div", { class: "row" }, status(t.status), h("strong", { text: t.title }), h("span", { class: "note", text: fmt(t.interest_key) }),
        h("button", { class: "ghost", text: "历史", onclick: () => showHistory(t.title, `/api/life/thread/${t.id}/history`) })),
      h("p", { class: "note", text: `开始于 ${day(t.started_at)} · 进度 ${fmt(t.progress_steps)}` }),
      h("div", { class: "row" },
        t.status === "active" ? h("button", { class: "ghost", text: "暂停", onclick: () => act(`/api/life/threads/${t.id}/pause`, {}, "已暂停") }) : null,
        t.status === "paused" ? h("button", { class: "ghost", text: "恢复", onclick: () => act(`/api/life/threads/${t.id}/resume`, {}, "已恢复") }) : null,
        ["active", "paused"].includes(t.status) ? h("button", { class: "ghost", text: "放弃", onclick: async () => {
          if (await confirmBox(`放弃生活线“${t.title}”？之后不会再推进。`)) act(`/api/life/threads/${t.id}/abandon`, {}, "已放弃");
        } }) : null)))
      : [empty("还没有生活线")]));
  }

  function renderPlans(rows) {
    $("#pl-plans").replaceChildren(...(rows.length ? rows.map((x) => h("div", { class: "card" },
      h("div", { class: "row" }, status(x.status), h("span", { text: x.content }),
        h("button", { class: "ghost", text: "历史", onclick: () => showHistory("计划", `/api/life/plan/${x.id}/history`) })),
      h("p", { class: "note", text: `计划时间 ${day(x.planned_for)}` }),
      ["planned", "active", "paused"].includes(x.status) ? h("div", { class: "row" },
        h("button", { class: "ghost", text: "取消计划", onclick: () => act(`/api/life/plans/${x.id}/cancel`, {}, "已取消") })) : null))
      : [empty("没有计划")]));
  }

  function renderEvents(rows) {
    $("#pl-events").replaceChildren(...(rows.length ? rows.slice().reverse().slice(0, 60).map((e) => h("div", { class: "card" },
      h("div", { class: "row" }, status(e.status), h("span", { class: "note", text: `${day(e.occurred_at)} · ${fmt(e.event_type)} · ${fmt(e.source)}` }),
        e.share_count ? pill("已经告诉过对方") : null,
        h("button", { class: "ghost", text: "历史", onclick: () => showHistory("生活事件", `/api/life/event/${e.id}/history`) })),
      h("p", { text: e.content }),
      e.status === "happened" ? h("div", { class: "row" },
        h("button", { class: "ghost", text: "更正", onclick: async () => {
          const content = window.prompt("更正后的内容（只改 Violet 的虚拟经历）", e.content); if (!content) return;
          const reason = window.prompt("更正原因"); if (!reason) return;
          try { await api("PATCH", `/api/life/events/${e.id}`, { content, reason }); toast("已更正"); loaders.persona(); }
          catch (err) { toast("更正失败：" + err.message, false); }
        } }),
        h("button", { class: "ghost", text: "撤回", onclick: async () => {
          const reason = window.prompt("撤回原因（这件事就当没发生过）"); if (reason) act(`/api/life/events/${e.id}/retract`, { reason }, "已撤回");
        } })) : null))
      : [empty("还没有生活事件")]));
  }

  function renderFlags(rows) {
    $("#pl-flags").replaceChildren(...(rows.length ? rows.map((f) => h("div", { class: "card" },
      h("div", { class: "row" }, pill(f.check_name, "warn"), h("span", { class: "note", text: `${day(f.created_at)} · ${fmt(f.entity)}` }),
        h("button", { class: "ghost", text: "已处理", onclick: () => act(`/api/life/flags/${f.id}/resolve`, {}, "已标记为处理") })),
      h("p", { text: fmt(f.detail) })))
      : [empty("没有需要处理的地方")]));
  }

  // 2026-10-29: persona templates -- switching asks whether to keep the memory or start empty (archived first)
  // 0.7.55: every persona has ONE set of records -- one that already has her own (an Eve import) goes on with hers
  function switchDialog(t, own) {
    return new Promise((resolve) => {
      const keep = h("input", { type: "radio", name: "pt-mem", value: "keep", checked: true });
      const clear = h("input", { type: "radio", name: "pt-mem", value: "clear" });
      const name = h("input", { maxlength: 20, placeholder: "可以不填，以后聊天时再起" });
      const d = h("dialog", {}, h("form", { method: "dialog", class: "stack" },
        h("h3", { text: `把主人设换成“${t.title}”` }),
        h("p", { class: "note", text: t.summary }),
        h("p", { class: "note", text: "主人设是没有单独设人设的渠道用的（下面“不同聊天渠道用不同人设”里单独设过的渠道不变）。" }),
        t.blank ? h("label", {}, "给她起个名字（可选） ", name) : null,
        h("label", { class: "check" }, keep, own
          ? ` 用她自己的记录：她已经有自己的聊天记录和记忆（${own} 条聊天），换上后接着用她的；现在这个人设的记录也原样保留，换回来还在`
          : " 保留记忆：她接着用现在这个人设的记忆和聊天记录，只是性格和说话方式换了"),
        h("label", { class: "check" }, clear, " 清除记忆：她从零开始（她的记忆、你的资料、她的生活经历都清空，并且翻篇：她看不到以前的聊天；聊天记录保留在后台；清除前会在服务器上存一份备份）"),
        h("div", { class: "row" },
          h("button", { type: "button", class: "ghost", text: "取消", onclick: () => { d.close(); d.remove(); resolve(null); } }),
          h("button", { type: "button", text: "换人设", onclick: () => {
            const memory = clear.checked ? "clear" : "keep"; d.close(); d.remove(); resolve({ memory, name: name.value.trim() }); } }))));
      document.body.append(d); d.showModal();
    });
  }
  // the admin's own persona: starts from what was written last time (2026-11-01: it was lost from view after switching
  // to a template), else from the persona in use; one item per line
  async function customDialog(saved) {
    let core = {};
    if (saved) core = { identity: { name: saved.name, self_description: saved.self_description }, personality: saved.personality,
      communication_habits: saved.habits, long_term_interests: saved.interests, values: saved.values, flaws: saved.flaws };
    else try { core = ((await api("GET", "/api/persona")).core || {}).data || {}; } catch (_) { /* empty form */ }
    const ident = core.identity || {};
    const area = (value, rows, ph) => { const t = h("textarea", { rows, placeholder: ph }); t.value = value; return t; };
    const lines = (list) => (list || []).map((x) => (typeof x === "string" ? x : x.label)).join("\n");
    const f = {
      name: h("input", { maxlength: 20, placeholder: "她的名字" }),
      summary: h("input", { maxlength: 60, placeholder: "一句话介绍（只在后台显示）" }),
      self_description: area(ident.self_description || "", 2, "她平时过着什么样的日子"),
      personality: area(lines(core.personality), 4, "性格，每行一条"),
      communication_habits: area(lines(core.communication_habits), 3, "说话习惯，每行一条"),
      interests: area(lines(core.long_term_interests), 3, "兴趣爱好，每行一条（看书、音乐、写日记会带动她的虚拟生活）"),
      values: area(lines(core.values), 3, "看重什么，每行一条"),
      flaws: area(lines(core.flaws), 2, "小缺点，每行一条"),
    };
    f.name.value = ident.name || "";
    f.summary.value = (saved && saved.summary) || "";
    const keep = h("input", { type: "radio", name: "pc-mem", checked: true }), clear = h("input", { type: "radio", name: "pc-mem" });
    const err = h("p", { class: "err" });
    const d = h("dialog", { class: "wide" }, h("form", { method: "dialog", class: "stack" },
      h("h3", { text: "自定义人设" }),
      h("p", { class: "note", text: (saved ? "这是你上次写的人设。" : "从现在用的人设开始改。") + "不能编造家人、工作、学校这些背景，也不会自称真人，这两条是固定的。" }),
      h("label", {}, "名字 ", f.name), h("label", {}, "介绍 ", f.summary), h("label", {}, "日常 ", f.self_description),
      h("label", {}, "性格 ", f.personality), h("label", {}, "说话习惯 ", f.communication_habits),
      h("label", {}, "兴趣 ", f.interests), h("label", {}, "看重 ", f.values), h("label", {}, "小缺点 ", f.flaws),
      h("label", { class: "check" }, keep, " 保留记忆"), h("label", { class: "check" }, clear, " 清除记忆并翻篇（先在服务器上存备份；聊天记录保留在后台）"),
      err,
      h("div", { class: "row" },
        h("button", { type: "button", class: "ghost", text: "取消", onclick: () => { d.close(); d.remove(); } }),
        h("button", { type: "button", text: "保存并使用", onclick: async () => {
          err.textContent = "";
          const memory = clear.checked ? "clear" : "keep";
          if (memory === "clear" && !(await confirmBox("确定清除全部记忆？她会忘记你和以前的一切（服务器上会保留一份备份文件）。"))) return;
          const persona = Object.fromEntries(Object.entries(f).map(([k, el]) => [k, el.value]));
          try {
            await api("POST", "/api/persona/custom", { persona, memory, confirm: true });
            d.close(); d.remove(); toast("已换成自定义人设"); loaders.persona();
          } catch (e) { err.textContent = e.message; }
        } }))));
    document.body.append(d); d.showModal();
  }

  // 2026-11-09 (user: 预览人设的很丑而且不好用): the hover pop-up that covered the cards below is gone. A click on a
  // card (or "查看") opens the whole persona in a dialog: what she is like in two columns, ‹ › (or the arrow keys)
  // to go through the personas one by one and compare, and "设为主人设" right there.
  const BLOCKS = [["personality", "性格"], ["habits", "说话习惯"], ["interests", "兴趣"], ["values", "看重"], ["flaws", "小缺点"]];
  function personaDialog(entries, start) {
    let i = start;
    const body = h("div", { class: "pv-body" });
    const d = h("dialog", { class: "pv-dialog", "aria-label": "人设详情" }, body);
    function show() {
      const e = entries[i], t = e.t;
      const blocks = BLOCKS.filter(([k]) => (t[k] || []).length).map(([k, title]) => h("section", { class: "pv-block" },
        h("h4", { text: title }), h("ul", {}, ...t[k].map((x) => h("li", { text: typeof x === "string" ? x : x.label })))));
      body.replaceChildren(...[                      // replaceChildren would print "null"
        h("div", { class: "pv-head" },
          h("div", {}, h("div", { class: "pv-count", text: `${i + 1} / ${entries.length}` }),
            h("h3", { text: t.blank ? "空白人设" + (t.name ? `（${t.name}）` : "") : (e.title || t.name || t.title) })),
          h("span", { class: "pv-pills" }, ...e.pills),
          h("button", { type: "button", class: "ghost pv-close", "aria-label": "关闭", text: "✕", onclick: () => d.close() })),
        h("p", { class: "pv-lead", text: e.summary || t.summary || "" }),
        t.self_description ? h("blockquote", { class: "pv-quote", text: t.self_description }) : null,
        t.blank ? h("p", { class: "note", text: "没有名字、性格和经历，什么都从和你聊天开始慢慢长出来。" }) : null,
        blocks.length ? h("div", { class: "pv-grid" }, ...blocks) : (t.blank ? null : h("p", { class: "note", text: "这个人设没有写细节。" })),
        e.users.length ? h("p", { class: "note used-by", text: "在用：" + e.users.join("、") }) : null,
        h("div", { class: "pv-foot" },
          h("button", { type: "button", class: "ghost", text: "‹ 上一个", disabled: entries.length < 2 ? true : null, onclick: () => go(-1) }),
          h("button", { type: "button", class: "ghost", text: "下一个 ›", disabled: entries.length < 2 ? true : null, onclick: () => go(1) }),
          h("span", { class: "grow" }),
          e.action ? h("button", { type: "button", text: e.action.label, onclick: () => { d.close(); e.action.run(); } }) : h("span", { class: "note", text: e.mainNote || "" }))
      ].filter(Boolean));
      body.scrollTop = 0;
    }
    const go = (step) => { i = (i + step + entries.length) % entries.length; show(); };
    d.addEventListener("keydown", (ev) => { if (ev.key === "ArrowLeft") go(-1); else if (ev.key === "ArrowRight") go(1); });
    d.addEventListener("click", (ev) => { if (ev.target === d) d.close(); });          // a click outside closes it
    d.addEventListener("close", () => d.remove());
    document.body.append(d);
    show();
    d.showModal();
  }
  // a card opens the dialog when clicked anywhere except its own buttons
  function openable(card, entries, index) {
    card.classList.add("pt-card");
    card.tabIndex = 0;
    card.title = "点开看完整人设";
    card.addEventListener("click", (ev) => { if (!ev.target.closest("button")) personaDialog(entries, index); });
    card.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && ev.target === card) personaDialog(entries, index); });
    return card;
  }

  // 2026-10-30: the relationship (second layer) and a persona / relationship per channel
  function relationshipSelect(d, value, withMain) {
    const sel = h("select", { "aria-label": "关系" }, ...(withMain ? [h("option", { value: "", text: "和主设定一样" })] : []),
      ...d.relationships.map((r) => h("option", { value: r.key, text: r.label, title: r.text })));
    sel.value = value || "";
    return sel;
  }
  // 2026-11-01 (user: 设置了称呼，切换人设之后也不显示，不知道还有没有生效): what is really used now, after merging
  const effectText = (x) => `现在：${x.persona} · ${x.label} · ${x.call ? `叫你“${x.call}”` : "不特别称呼你"}`;
  // 2026-11-09 (user: 人设与生活的UI也要改一下): the relationship and the per-channel personas were two cards of
  // stacked full-width fields. Now one table: the main setting first, then every channel -- persona / relationship /
  // what she calls you / save, and under each row what is really in effect.
  function renderWho(d) {
    const mainTpl = d.current.template === "custom" ? "自定义人设"
      : ((d.templates.find((t) => t.key === d.current.template) || {}).title || d.current.template || "—");
    const rel = d.relationship || { kind: "close" };
    const mainRel = relationshipSelect(d, rel.kind, false);
    const mainCall = h("input", { maxlength: 12, placeholder: "可空，例如 爸爸", "aria-label": "称呼" }); mainCall.value = rel.call || "";
    const relText = h("span", { class: "who-sub" });
    const showRel = () => { const r = d.relationships.find((x) => x.key === mainRel.value); relText.textContent = (r ? r.text + "　" : "") + (d.in_effect ? effectText(d.in_effect) : ""); };
    mainRel.addEventListener("change", showRel); showRel();
    const head = h("div", { class: "who-row who-head" }, h("span", { text: "" }), h("span", { text: "人设（她是谁）" }),
      h("span", { text: "关系（你们是什么关系）" }), h("span", { text: "她怎么叫你" }), h("span", { text: "" }));
    const main = h("div", { class: "who-row who-main" },
      h("div", {}, h("strong", { text: "主设定" }), h("div", { class: "note", text: "没单独设的渠道都用它" })),
      h("div", { class: "who-tpl" }, h("strong", { text: mainTpl }),
        h("button", { type: "button", class: "ghost small", text: "换", onclick: () => scrollToAnchor("pl-sec-templates") })),
      mainRel, mainCall,
      h("button", { type: "button", text: "保存", onclick: async () => {
        try { await api("POST", "/api/persona/relationship", { kind: mainRel.value, call: mainCall.value }); toast("已保存，下一条消息起生效"); loaders.persona(); }
        catch (e) { toast(e.message, false); }
      } }), relText);
    const options = [h("option", { value: "", text: "和主设定一样" }),
      ...d.templates.map((t) => h("option", { value: t.key, text: t.title })), ...(d.custom ? [h("option", { value: "custom", text: "自定义人设" })] : [])];
    const rows = (d.channels || []).map((c) => {
      const tpl = h("select", { "aria-label": `${c.label} 的人设` }, ...options.map((o) => o.cloneNode(true))); tpl.value = c.template || "";
      const crel = relationshipSelect(d, (c.relationship || {}).kind, true);
      const call = h("input", { maxlength: 12, "aria-label": `${c.label} 的称呼`, placeholder: rel.call ? `空 = 和主设定一样（${rel.call}）` : "空 = 和主设定一样" });
      call.value = (c.relationship || {}).call || "";
      return h("div", { class: "who-row" + (c.template ? " own" : "") },
        h("div", {}, h("strong", { text: c.label }), c.template ? h("div", { class: "note", text: "单独的人设 · 自己的记忆" }) : null),
        tpl, crel, call,
        h("button", { class: "ghost", type: "button", text: "保存", onclick: async () => {
          try { await api("POST", "/api/persona/channel", { channel: c.id, template: tpl.value || null, kind: crel.value || null, call: call.value }); toast(`${c.label} 已保存`); loaders.persona(); loadPersonaPicker(); }
          catch (e) { toast(e.message, false); }
        } }),
        c.in_effect ? h("span", { class: "who-sub", text: effectText(c.in_effect) }) : null);
    });
    $("#pl-who").replaceChildren(...[                       // replaceChildren would print "null"
      h("h3", { id: "pl-sec-who", text: "谁在和你聊" }),
      h("div", { class: "who-table" }, head, main, ...rows),
      rows.length ? null : h("p", { class: "note", text: "还没有聊天渠道。" }),
      h("details", {}, h("summary", { text: "渠道用不同的人设会怎样？" }),
        h("p", { class: "note", text: "比如：微信上是“Violet + 恋人”，QQ 上是“AI + 开发者”。人设不同就是不同的人：聊天记录、记忆、提醒、生活和心情各是各的（聊天记录、记忆页上方和本页下面可以选看谁的）；课表共用。第一次给一个渠道选自己的人设时，这个渠道以前的聊天会跟着过去。QQ 群里也按 QQ 这一行的设定。导入 Eve 时选的人设也显示在这里。" }))].filter(Boolean));
  }

  // test mode: its own bar (it is not a persona -- it runs on top of the one in use and switches itself off)
  function renderTest(d) {
    const tm = d.test_mode || { on: false };
    const hours = h("select", { "aria-label": "多久后自动关闭" }, ...[1, 2, 4, 8]
      .map((v) => h("option", { value: v, text: `${v} 小时后自动关闭` })));
    hours.value = "2";
    const box = $("#pl-test");
    box.classList.toggle("test-on", !!tm.on);
    box.replaceChildren(
      h("div", { class: "test-text" }, h("div", { class: "row" }, h("strong", { text: "测试模式" }),
        tm.on ? pill(`测试中 · 还有 ${tm.minutes_left} 分钟`, "warn") : pill("排查问题用", "")),
        h("p", { class: "note", text: "不用人设、不看记忆和以前的聊天，什么都不记下来；每条回复后面附一段程序写的诊断（她判断你要做什么、能用什么、实际做了什么、哪里没成）。私聊和群聊都生效，群里的诊断群友也能看到；关掉后回到现在的人设，记忆不受影响。" })),
      h("div", { class: "row test-ctl" }, tm.on
        ? h("button", { type: "button", text: "关闭测试", onclick: async () => {
          try { await api("POST", "/api/persona/test-mode", { on: false }); toast("测试模式已关闭"); loaders.persona(); } catch (e) { toast(e.message, false); }
        } })
        : [hours, h("button", { type: "button", class: "ghost", text: "开始测试", onclick: async () => {
          try { await api("POST", "/api/persona/test-mode", { on: true, hours: Number(hours.value) }); toast("测试模式已打开"); loaders.persona(); } catch (e) { toast(e.message, false); }
        } })]));
  }

  async function renderTemplates() {
    let d;
    try { d = await api("GET", "/api/persona/templates"); } catch (e) { $("#pl-templates").replaceChildren(h("p", { class: "err", text: e.message })); return; }
    renderWho(d);
    renderTest(d);
    const ownUsers = (d.used_by || {}).custom || [];
    const switchTo = (t) => async () => {
      const choice = await switchDialog(t, (d.records || {})[t.key]);
      if (!choice) return;
      if (choice.memory === "clear" && !(await confirmBox("确定清除她的全部记忆？她会忘记你和以前的一切（服务器上会保留一份备份文件）。"))) return;
      try {
        const r = await api("POST", "/api/persona/switch", { template: t.key, memory: choice.memory, name: choice.name, confirm: true });
        toast(r.archive ? `已换主人设，她的记忆已清除（备份：${r.archive}）`
          : r.records === "own" ? "已换主人设，她用自己的记录" : "已换主人设，记忆保留");
        loaders.persona(); loadPersonaPicker();
      } catch (e) { toast(e.message, false); }
    };
    // the persona in use first, then the rest in their order
    const order = d.templates.slice().sort((a, b) => (((d.used_by || {})[b.key] || []).length > 0 || d.current.template === b.key) -
      (((d.used_by || {})[a.key] || []).length > 0 || d.current.template === a.key));
    const entries = order.map((t) => {
      const isMain = d.current.template === t.key;
      const users = (d.used_by || {})[t.key] || [];
      return { t, isMain, users, title: t.title, summary: t.summary,
        pills: [isMain ? pill("主人设", "ok") : users.length ? pill("正在使用", "ok") : null].filter(Boolean),
        mainNote: isMain ? (d.current.switched_at ? `主人设 · ${when(d.current.switched_at)} 换上` : "主人设（默认）") : "",
        action: isMain ? null : { label: "设为主人设", run: switchTo(t) } };
    });
    const customMain = d.current.template === "custom";
    if (d.custom) entries.push({ t: d.custom, isMain: customMain, users: ownUsers, title: d.custom.name || "自定义人设",
      summary: d.custom.summary || "你自己写的人设", pills: [customMain ? pill("主人设", "ok") : ownUsers.length ? pill("正在使用", "ok") : pill("自己写")],
      action: { label: customMain ? "修改" : "修改并换回它", run: () => customDialog(d.custom) } });
    const cards = entries.filter((e) => e.t !== d.custom).map((e, index) => openable(
      h("div", { class: "card platform pt-small" + (e.users.length || e.isMain ? " current" : "") },
        h("div", { class: "platform-head" }, h("strong", { text: e.title }), ...e.pills),
        h("p", { class: "note pt-summary", text: e.summary }),
        // 0.7.55 (user: 导入时选了人设，人设页却看不出来，两边同步): which channels talk as her, wherever it was chosen
        usedLine(e.users, e.isMain),
        h("div", { class: "row platform-foot" },
          h("button", { type: "button", class: "ghost", text: "查看", onclick: () => personaDialog(entries, index) }),
          e.action ? h("button", { type: "button", class: "ghost", text: e.action.label, onclick: e.action.run })
            : h("span", { class: "note", text: e.mainNote }))), entries, index));
    const own = h("div", { class: "card platform pt-small" + (ownUsers.length || customMain ? " current" : "") },
      h("div", { class: "platform-head" }, h("strong", { text: d.custom ? d.custom.name || "自定义人设" : "自定义人设" }),
        ownUsers.length ? pill("正在使用", "ok") : customMain ? pill("主人设", "ok") : pill("自己写", "")),
      h("p", { class: "note pt-summary", text: d.custom ? (d.custom.summary || "你自己写的人设") : "名字、性格、说话方式、兴趣都自己写；从现在用的人设开始改。" }),
      usedLine(ownUsers, customMain),
      h("div", { class: "row platform-foot" },
        d.custom ? h("button", { type: "button", class: "ghost", text: "查看", onclick: () => personaDialog(entries, entries.length - 1) }) : null,
        h("button", { type: "button", class: "ghost",
          text: customMain ? "修改" : (d.custom ? "修改并换回它" : "自己写一个"), onclick: () => customDialog(d.custom) })));
    if (d.custom) openable(own, entries, entries.length - 1);
    $("#pl-templates").replaceChildren(...cards, own);
  }
  function usedLine(users, isMain) {
    if (users.length) return h("p", { class: "note used-by", text: "在用：" + users.join("、") });
    return isMain ? h("p", { class: "note used-by", text: "所有渠道都单独设了别的人设，现在没有渠道用它" }) : null;
  }

  // 2026-11-02: her look (what a selfie / a picture of her shows) -- per persona (the picker in the menu), kept in the
  // data folder; GET / PUT /api/images/visual-profile
  const LOOK = [["appearance", "overall", "整体"], ["appearance", "hair", "头发"], ["appearance", "face", "脸"],
    ["appearance", "build", "身形和特征"], ["wardrobe", "style", "衣服"], ["wardrobe", "accessories", "配饰"]];
  async function renderLook() {
    const box = $("#pl-look");
    let d;
    try { d = await api("GET", "/api/images/visual-profile"); } catch (e) { box.replaceChildren(h("p", { class: "err", text: e.message })); return; }
    const p = d.profile || {};
    const inputs = LOOK.map(([group, key, label]) => {
      const area = h("textarea", { rows: 2, maxlength: 300, "aria-label": label });
      area.value = ((p[group] || {})[key]) || "";
      area.dataset.group = group; area.dataset.key = key;
      return h("label", { class: "stack" }, h("span", { class: "note", text: label }), area);
    });
    const colors = h("input", { maxlength: 120, "aria-label": "衣服颜色" });
    colors.value = ((p.wardrobe || {}).colors || []).join("、");
    box.replaceChildren(
      h("p", { class: "note", text: `她画自己（自拍、照片）时按这里画。现在看的是：${d.persona || "主人设"}（上面“下面这些是谁的”可以换）。改了会记一个新版本，升级 Violet 不会冲掉。` }),
      ...inputs, h("label", { class: "stack" }, h("span", { class: "note", text: "衣服颜色（用顿号隔开）" }), colors),
      h("div", { class: "row" }, h("button", { type: "button", text: "保存她的样子", onclick: async () => {
        const patch = { appearance: { ...(p.appearance || {}) }, wardrobe: { ...(p.wardrobe || {}) } };
        box.querySelectorAll("textarea").forEach((a) => { patch[a.dataset.group][a.dataset.key] = a.value.trim(); });
        patch.wardrobe.colors = colors.value.split(/[、,，\s]+/).map((c) => c.trim()).filter(Boolean);
        try { await api("PUT", "/api/images/visual-profile", { patch, note: "人设页修改" }); toast("已保存，下一张图起生效"); renderLook(); }
        catch (e) { toast(e.message, false); } } }),
        h("span", { class: "note", text: `版本 ${p.version || "—"}` })));
  }

  loaders.persona = async () => {
    jumpNav($("#pl-jump"), [["谁在和你聊", "pl-who"], ["人设库", "pl-sec-templates"], ["测试模式", "pl-test"],
      ["她的样子", "pl-look-box"], ["人设详情", "pl-core-box"], ["她的生活", "pl-life"]]);
    renderTemplates();
    renderLook();
    try {
      const [p, l] = await Promise.all([api("GET", "/api/persona"), api("GET", "/api/life")]);
      renderCore(p);
      renderDynamics(p.dynamics || []);
      const [et, ek] = ENGINE[(l.engine || {}).mode] || [fmt((l.engine || {}).mode), ""];
      const engine = $("#pl-engine"); engine.textContent = et; engine.className = "pill " + ek;
      renderThreads(l.threads || []);
      renderPlans(l.plans || []);
      renderEvents(l.events || []);
      renderFlags(l.flags || []);
    } catch (e) { toast("人设与生活读取失败：" + e.message, false); }
  };
  $("#pl-refresh").addEventListener("click", () => loaders.persona());
  $("#pl-pause").addEventListener("click", () => act("/api/life/engine/pause", {}, "生活引擎已暂停"));
  $("#pl-resume").addEventListener("click", () => act("/api/life/engine/resume", {}, "生活引擎已恢复"));
})();
