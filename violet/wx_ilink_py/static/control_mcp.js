// Violet Control Center: MCP servers and tool permissions (phase 10.5). Replaces control.js's toolServerCard with a card
// that explains every state, offers safe fixes and lets the admin classify, authorise and test tools. The server
// decides and validates everything; unclassified tools stay refused until the admin classifies them one by one.
"use strict";
(() => {
  const LEVEL = { read: "只读", write: "写入", high_risk: "高风险", deny: "拒绝", unknown: "未分类（拒绝）" };
  // 2026-11-09 (user: 不太理解 写入 和 高风险 有什么区别): in the code they are handled exactly alike (both: ask the
  // owner first, or "允许直接用"), so the page offers three plain choices; "high_risk" stays readable in the files
  // and shows as the same choice
  const LEVEL_HINT = { read: "直接用（只是查看）", write: "先问你（会改动东西）", high_risk: "先问你（会改动东西）", deny: "不给她用" };
  const SHORT = { read: "直接用", write: "先问你", high_risk: "先问你", deny: "不给她用", unknown: "不给她用（还没分类）" };
  const STATUS = { connected: ["已连接", "ok"], unavailable: ["未连接", "bad"], disabled: ["已停用", ""], unknown: ["还没检查", "warn"] };
  const FIND = { ok: "ok", info: "", warn: "warn", error: "bad" };
  const AGENTS = { chat: "Chat（聊天）", proactive: "Proactive（主动消息）" };
  const enc = encodeURIComponent;

  async function call(method, url, body, done) {
    try { const r = await api(method, url, body); if (done) toast(done); loaders.tools(); return r; }
    catch (e) { const p = e.data && e.data.problems; toast((p ? Object.values(p).join("；") : e.message), false); return null; }
  }

  function findingRow(s, f) {
    const fix = f.action === "apply_profile" ? h("button", { text: `一键套用模板“${f.profile}”`, onclick: () =>
        call("POST", `/api/control/tools/servers/${enc(s.name)}/profile`, { profile: f.profile }, "已套用模板，正在重新检查").then(() =>
          api("POST", "/api/control/tools/refresh", { server: s.name }).then(() => loaders.tools())) })
      : f.action === "check" ? h("button", { class: "ghost", text: "检查连接", onclick: () =>
        call("POST", "/api/control/tools/refresh", { server: s.name }, "已检查") })
      : f.action === "enable_server" ? h("button", { class: "ghost", text: "启用", onclick: () =>
        call("POST", `/api/control/tools/servers/${enc(s.name)}`, { enabled: true }, "已启用") }) : null;
    return h("div", { class: "row" }, pill({ ok: "正常", info: "提示", warn: "需要处理", error: "问题" }[f.level] || f.level, FIND[f.level]),
      h("span", { text: f.message }), fix);
  }

  // 2026-11-09 (user: 连上手机之后下面一大堆工具，要看到下一个工具要向下滑很远): the tool list is folded into one line
  // ("18 个工具 · 18 个可用"), opened it is one compact row per tool; the model-facing description is one line until
  // clicked. What is open survives the re-render after every change (loaders.tools redraws the whole page).
  const openLists = new Map(), openDescs = new Set();
  // the Violet MCP app's own plain names for its tools (mcp ToolKit.PLAIN)
  const PHONE_TITLES = { get_screen_state: "屏幕亮没亮", get_foreground_app: "正在用的 App", get_app_usage_stats: "各 App 用了多久",
    next_alarm: "下一个闹钟", now_playing: "正在播放", audio_output: "耳机 / 蓝牙", network_state: "Wi-Fi 还是流量",
    step_count: "今天的步数", calendar_events: "今天和明天的日程", phone_events: "手机上的动静", battery: "电量",
    device_info: "手机型号", screen_info: "屏幕参数", storage_info: "存储空间", locale_info: "语言和时区",
    system_properties: "系统版本细节", time_now: "手机时间", app_info: "某个 App 装没装", installed_apps: "装了哪些 App",
    permission_state: "本应用的权限", health: "连接自检" };

  function toolRow(s, t) {
    // 2026-11-09 (user: 选择写入的话难道他真的有权限可以修改吗): the level is Violet's own gate -- whether she may use
    // the tool and whether she asks first. It gives the tool no new power (what a tool can do is decided by the phone /
    // server that provides it). Marking more strictly is always safe; marking a changing tool "只读" is the risky way,
    // so that one asks for confirmation.
    const levels = ["read", "write", ...(t.admin_level === "high_risk" ? ["high_risk"] : []), "deny"];
    const select = h("select", { "aria-label": "她能不能用", class: "tool-level", title: "Violet 这边的闸门：能不能用、用之前要不要问你" },
      h("option", { value: "", text: t.level_source === "template" ? `默认：${SHORT[t.permission] || t.permission}` : "还没分类（不给她用）" }),
      ...levels.map((v) => h("option", { value: v, text: LEVEL_HINT[v] })));
    select.value = t.admin_level || "";
    // the phone app only offers tools that look at things: no choice to make there (停用 still works)
    const fixed = s.name === "android-phone" && t.permission === "read" && !t.admin_level;
    select.addEventListener("change", async () => {
      const level = select.value || null;
      if (level === "read" && t.permission !== "read") {
        if (!(await confirmBox(`让她直接用 ${t.name}？\n\n选“直接用”后她用它不会再问你。只有确定这个工具不会改动任何东西（只是查看）时才这样选。` +
          "\n（工具如果自己声明会改动或删除东西，Violet 仍会按高风险处理。）"))) { select.value = t.admin_level || ""; return; }
      }
      call("POST", `/api/control/tools/servers/${enc(s.name)}/tools/${enc(t.name)}/level`, { level }, "已保存分类");
    });
    const allow = t.needs_second_key ? h("button", { class: t.allowed_risky ? "danger" : "ghost", text: t.allowed_risky ? "撤销直接用" : "允许直接用",
      title: t.allowed_risky ? "改回：每次先问你（或按“她能自己做的事”的设置）" : "不再每次问你，她需要时直接用",
      onclick: async () => {
        if (!t.allowed_risky && !(await confirmBox(`允许 Violet 不问你、直接使用 ${t.name}？这是会改动东西的工具，她会真的执行它的操作（工具本身能做什么由提供它的服务器决定）。`))) return;
        call("POST", `/api/control/tools/servers/${enc(s.name)}/tools/${enc(t.name)}/allow`, { allowed: !t.allowed_risky, confirm: true }, "已保存");
      } }) : null;
    const test = t.available && t.permission === "read" ? h("button", { class: "ghost", text: "测试", onclick: async () => {
      const props = Object.keys((t.input_schema || {}).properties || {});
      const raw = window.prompt(`测试 ${t.name}：参数（JSON）` + (props.length ? `，可用参数：${props.join("、")}` : ""), "{}");
      if (raw === null) return;
      let args; try { args = JSON.parse(raw || "{}"); } catch (_) { toast("参数不是合法的 JSON", false); return; }
      try {
        const r = await api("POST", `/api/control/tools/servers/${enc(s.name)}/tools/${enc(t.name)}/test`, { arguments: args });
        out.textContent = r.ok ? `成功（${r.latency_ms} 毫秒）：${r.result}` : `失败：${r.message}${r.code ? "（" + r.code + "）" : ""}`;
      } catch (e) { out.textContent = "失败：" + e.message; }
    } }) : null;
    const out = h("pre", { class: "note tool-out" });
    const key = s.name + "/" + t.name;
    const desc = h("div", { class: "tool-desc" + (openDescs.has(key) ? " open" : ""), title: "点一下看完整说明",
      text: (t.description || "（没有说明）") + (t.reason ? `　【${t.reason}】` : "") });
    desc.addEventListener("click", () => { desc.classList.toggle("open"); openDescs[desc.classList.contains("open") ? "add" : "delete"](key); });
    const title = s.name === "android-phone" && PHONE_TITLES[t.name];
    return h("div", { class: "tool-item" + (t.enabled ? "" : " off") },
      h("div", { class: "tool-name" }, h("strong", { text: title || t.name }), title ? h("code", { text: t.name }) : null,
        t.enabled ? pill(t.available ? "可用" : "不可用", t.available ? "ok" : "") : pill("已停用")),
      h("div", { class: "tool-ctl" }, fixed ? null : select, allow, test,
        h("button", { class: "ghost", text: t.enabled ? "停用" : "启用", onclick: () =>
          call("POST", `/api/control/tools/servers/${enc(s.name)}/tools/${enc(t.name)}`, { enabled: !t.enabled }, t.enabled ? "已停用" : "已启用") })),
      desc, out);
  }

  function toolList(s) {
    if (!s.tools.length) return h("p", { class: "note", text: "还没有发现工具（连接成功后会列出）" });
    const usable = s.tools.filter((t) => t.enabled && t.available).length;
    const off = s.tools.filter((t) => !t.enabled).length;
    const todo = s.tools.filter((t) => t.permission === "unknown").length;
    const open = openLists.has(s.name) ? openLists.get(s.name) : s.tools.length <= 4 || todo > 0;
    const list = h("details", { class: "tool-list", open: open ? true : null },
      h("summary", {}, h("strong", { text: `${s.tools.length} 个工具` }), ` · ${usable} 个可用` + (off ? ` · ${off} 个已停用` : ""),
        todo ? pill(`${todo} 个待分类`, "warn") : null, h("span", { class: "tool-hint", text: "展开 / 收起" })),
      h("div", { class: "tool-rows" }, h("p", { class: "note tool-explain", text: s.name === "android-phone"
        ? "手机 App 的工具都只能查看（电量、正在用的 App、闹钟……），不能改手机上的任何东西，所以她直接用，不用你选。不想让她看哪一项，点那一项的“停用”。"
        : "每个工具选一种：“直接用”——只是查看、不会改动东西的工具；“先问你”——会改动东西的（发消息、改文件……），她每次要用都先问你，你回“同意 N”才执行，" +
          "点“允许直接用”后就不再问（这一页最上面的“她能自己做的事”可以改成干脆不给她）；“不给她用”——她看不到这个工具。" +
          "这只是 Violet 这边的关卡，不会让工具多出能力：工具能做什么由提供它的服务器决定。真正会改东西的工具不要选“直接用”。" }),
        ...s.tools.map((t) => toolRow(s, t))));
    list.addEventListener("toggle", () => openLists.set(s.name, list.open));
    return list;
  }

  window.toolServerCard = function (s) {
    const [st, kind] = STATUS[s.status] || [s.status, ""];
    const agents = h("div", { class: "row" }, h("span", { text: "授权给：" }), ...Object.entries(AGENTS).map(([a, label]) => {
      const box = h("input", { type: "checkbox" }); box.checked = s.agents.includes(a);
      box.addEventListener("change", () => {
        const next = Object.keys(AGENTS).filter((x) => (x === a ? box.checked : s.agents.includes(x)));
        call("POST", `/api/control/tools/servers/${enc(s.name)}`, { agents: next }, "已保存");
      });
      return h("label", { class: "check" }, box, " " + label);
    }));
    const origin = s.builtin ? "内置" : s.custom ? "后台添加" : ".env";
    const headers = (s.headers || []).map((x) => h("div", { class: "row" }, h("span", { text: `${x.header}：${x.is_set ? "已设置（" + x.fingerprint + "）" : "未设置"}` }),
      h("button", { class: "ghost", text: "替换", onclick: () => replaceToolSecret(s.name, x.header) })));
    const who = s.agents.map((a) => (AGENTS[a] || a).replace(/（.*/, "")).join("、") || "没有授权";
    const more = h("details", { class: "srv-more", open: openLists.get("more:" + s.name) ? true : null },
      h("summary", { text: `授权和凭据（现在给：${who}）` }), agents, ...headers);
    more.addEventListener("toggle", () => openLists.set("more:" + s.name, more.open));
    return h("div", { class: "card tool-srv" },
      h("div", { class: "srv-head" }, h("strong", { text: s.name }), pill(st, kind), pill(origin),
        s.tool_profile ? pill("模板 " + s.tool_profile) : null,
        h("span", { class: "srv-actions" },
          h("button", { class: "ghost", text: "检查连接", onclick: () => call("POST", "/api/control/tools/refresh", { server: s.name }, "已检查") }),
          h("button", { class: "ghost", text: s.enabled ? "停用服务器" : "启用服务器", onclick: async () => {
            if (s.enabled && !(await confirmBox(`停用 ${s.name}？它的工具都不会再给 Violet，随时可以再启用。`))) return;
            call("POST", `/api/control/tools/servers/${enc(s.name)}`, { enabled: !s.enabled }, "已保存");
          } }),
          s.custom ? h("button", { class: "danger", text: "删除", onclick: async () => {
            if (!(await confirmBox(`删除 MCP 服务器“${s.name}”？它的凭据和工具分类会一起删除。`))) return;
            call("POST", `/api/control/tools/servers/${enc(s.name)}/delete`, { confirm: true }, "已删除");
          } }) : null)),
      h("div", { class: "note", text: [s.url ? `地址 ${s.url}` : null, s.command, `传输 ${s.transport}`,
        `调用 ${s.total_calls} 次 / 失败 ${s.total_failures} 次`].filter(Boolean).join(" · ") }),
      ...(s.diagnosis || []).map((f) => findingRow(s, f)),
      more, toolList(s));
  };

  $("#mcp-add").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const err = $("#mcp-add-error"); err.textContent = "";
    const body = { name: $("#mcp-name").value.trim(), url: $("#mcp-url").value.trim(), transport: $("#mcp-transport").value,
      agents: [...document.querySelectorAll("#mcp-add input[data-agent]:checked")].map((i) => i.dataset.agent),
      header: $("#mcp-header").value.trim() || "Authorization", header_value: $("#mcp-header-value").value || null };
    try {
      const s = await api("POST", "/api/control/tools/servers", body);
      ev.target.reset();
      const first = (s.diagnosis || [])[0];
      toast(first ? first.message : "已添加", !first || first.level === "ok");
      loaders.tools();
    } catch (e) { const p = e.data && e.data.problems; err.textContent = p ? Object.values(p).join("；") : e.message; }
  });
})();
