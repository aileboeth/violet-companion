// Violet Control Center: the memes she gets (2026-11-12, ai/memes.py), on the tools page: the meme wiki (downloaded
// from cnmeme.wiki or imported from a file -- not shipped with Violet), a test search, and "your memes" -- what you
// taught her a meme of yours means (each persona has her own, like her memories). DOM nodes only.
"use strict";
(() => {
  const MAX = 64 * 1024 * 1024;
  let timer = null;

  async function act(method, url, body, done) {
    try { const r = await api(method, url, body || {}); if (done) toast(done); load(); return r; }
    catch (e) { toast(e.message, false); return null; }
  }

  function importFile() {
    const input = h("input", { type: "file", accept: ".db,.sqlite,application/octet-stream" });
    input.addEventListener("change", async () => {
      const f = input.files[0];
      if (!f) return;
      if (f.size > MAX) return toast("文件太大了（最多 64 MB）", false);
      try {
        const resp = await fetch("/api/control/memes/import", { method: "POST", credentials: "same-origin",
          headers: { "Content-Type": "application/octet-stream", Accept: "application/json" }, body: f });
        const d = await resp.json();
        if (!resp.ok || !d.ok) throw new Error(d.error || ("导入失败 " + resp.status));
        toast(`导入好了：新增 ${d.data.added} 条，补充 ${d.data.merged} 条，现在共 ${d.data.count} 条`);
      } catch (e) { toast(e.message, false); }
      load();
    });
    input.click();
  }

  function wikiBox(d) {
    const w = d.wiki, s = d.sync;
    const status = s.running ? pill("正在下载", "warn")
      : (w.installed ? pill(`${w.count} 条`, "ok") : pill("还没有下载", "warn"));
    const progress = s.running
      ? (s.phase === "details" ? `第 2 步：补全每条的详细用法 ${s.details}/${s.details_total}（这一步慢，期间已经能用）`
        : `第 1 步：取词条列表，已取 ${s.listed} 条`)
      : (s.message || "");
    const buttons = [];
    if (s.running) {
      buttons.push(h("button", { type: "button", class: "ghost", text: "停止下载", onclick: () => act("POST", "/api/control/memes/stop", {}, "停下了") }));
    } else {
      buttons.push(h("button", { type: "button", text: w.installed ? "更新梗百科" : "下载梗百科", onclick: async () => {
        const text = w.installed ? "从 cnmeme.wiki 取新增和改过的词条？已有的不用重新下载。"
          : "从 cnmeme.wiki 下载梗百科？先取词条列表（约 5–10 分钟，取完就能用），再在后台慢慢补全每条的详细用法（可能要 1–2 小时，可以随时停，下次接着下）。只读取网站的公开词条，不会发送你的聊天内容。";
        if (await confirmBox(text, { danger: false, ok: "下载" })) act("POST", "/api/control/memes/download", { confirm: true }, "开始下载了");
      } }));
      buttons.push(h("button", { type: "button", class: "ghost", text: "从文件导入", title: "例如 dsh-memewiki 插件里的 data/memes.db", onclick: importFile }));
      if (w.installed) buttons.push(h("button", { type: "button", class: "ghost", text: "删除梗百科", onclick: async () => {
        if (await confirmBox("删除下载的梗百科？你教她的梗不受影响。")) act("DELETE", "/api/control/memes/wiki", null, "已删除");
      } }));
    }
    clearTimeout(timer);
    if (s.running) timer = setTimeout(load, 4000);
    return h("div", { class: "stack" },
      h("div", { class: "row" }, h("strong", { text: "梗百科" }), status),
      h("p", { class: "note", text: "你说的话里有梗百科收录的梗时，她会同时看到这个梗的意思和用法，像本来就懂一样接住；没把握的新词她也会自己查。"
        + "查词都在这台服务器上，不联网、不发送聊天内容。" }),
      h("p", { class: "note", text: "词条来自 cnmeme.wiki 的公开词条（版权归原作者和网站），Violet 不自带，要在这里下载或导入。" }),
      w.installed ? h("p", { class: "note", text: `共 ${w.count} 条，其中 ${w.full} 条有详细用法` + (w.built_at ? `；最近更新 ${w.built_at}` : "") }) : null,
      progress ? h("p", { class: "note", text: progress }) : null,
      h("div", { class: "row" }, ...buttons));
  }

  function tryBox() {
    const input = h("input", { maxlength: 60, placeholder: "例如：去码头整点薯条" });
    const out = h("div", { class: "stack" });
    const go = async () => {
      if (!input.value.trim()) return;
      try {
        const r = await api("GET", "/api/control/memes/lookup?q=" + encodeURIComponent(input.value.trim()));
        out.replaceChildren(...(r.found.length ? r.found.map((f) => h("p", { class: "note", text: f.text }))
          : [h("p", { class: "note", text: "没查到。她遇到这个词时会说不知道，或者问你是什么梗。" })]));
      } catch (e) { toast(e.message, false); }
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    return h("div", { class: "stack" }, h("strong", { text: "试试查一个梗" }),
      h("label", { class: "row grow" }, input, h("button", { type: "button", class: "ghost", text: "查", onclick: go })), out);
  }

  function ownItem(m) {
    const phrase = h("input", { maxlength: 40 }); phrase.value = m.phrase;
    const meaning = h("input", { maxlength: 300 }); meaning.value = m.meaning;
    return h("div", { class: "row grow" }, phrase, meaning,
      h("button", { type: "button", class: "ghost", text: "保存", onclick: () => act("POST", `/api/control/memes/own/${m.id}`, { phrase: phrase.value, meaning: meaning.value }, "已保存") }),
      h("button", { type: "button", class: "ghost", text: "删除", onclick: async () => {
        if (await confirmBox(`删除「${m.phrase}」？`)) act("DELETE", `/api/control/memes/own/${m.id}`, null, "已删除");
      } }));
  }

  function ownBox(d) {
    const phrase = h("input", { maxlength: 40, placeholder: "梗：你平时怎么说" });
    const meaning = h("input", { maxlength: 300, placeholder: "意思、来历、怎么接" });
    return h("div", { class: "stack" },
      h("div", { class: "row" }, h("strong", { text: "你教她的梗" }), h("span", { class: "note", text: `${d.own.length} 个` })),
      h("p", { class: "note", text: "你们之间的梗、你的口头禅：你在聊天里给她讲明白一个梗时，她会自己记在这里；也可以在这里加、改。每个人设各记各的，清除记忆时一起清除。" }),
      ...d.own.map(ownItem),
      h("div", { class: "row grow" }, phrase, meaning,
        h("button", { type: "button", text: "加上", onclick: async () => {
          if (await act("POST", "/api/control/memes/own", { phrase: phrase.value, meaning: meaning.value }, "加好了")) { phrase.value = ""; meaning.value = ""; }
        } })));
  }

  async function load() {
    const box = $("#tools-memes");
    if (!box) return;
    let d;
    try { d = await api("GET", "/api/control/memes"); } catch (e) { box.replaceChildren(h("p", { class: "err", text: e.message })); return; }
    box.replaceChildren(h("div", { class: "card", id: "memes-card" },
      h("div", { class: "row" }, h("h3", { text: "听懂梗" }), d.enabled ? pill("开", "ok") : pill("关着", "warn")),
      d.enabled ? null : h("p", { class: "note", text: "下面的“听懂梗”关着：她现在不会看梗的资料，也不能查。" }),
      wikiBox(d), tryBox(), ownBox(d)));
  }

  const previous = loaders.tools;
  loaders.tools = async () => { if (previous) await previous(); load(); };
})();
