// Violet Control Center: the page in the language the admin picked (2026-11-09; the README's languages).
// The pages are written in Chinese; this translates what is shown -- text, placeholder / title / aria-label / alt,
// and the browser's alert / confirm / prompt -- from window.VIOLET_I18N ({lang, dict}), loaded before this file
// (/control/i18n.js; nothing when the language is 简体中文). The dictionary is keyed by the Chinese text itself
// (tools/i18n_extract.py lists the keys); "{0}" in a key stands for a value filled in at run time. Text the
// program does not write -- chat messages, memories, logs, prompts -- sits under translate="no" and stays as it is.
// A text without a translation stays Chinese; window.violetI18nMisses() lists them (for completing the dictionary).
"use strict";
(() => {
  const pack = window.VIOLET_I18N;
  const LANGS = [["zh-CN", "简体中文"], ["en", "English"], ["zh-TW", "繁體中文"], ["ja", "日本語"], ["ko", "한국어"],
    ["es", "Español"], ["fr", "Français"], ["de", "Deutsch"], ["ru", "Русский"], ["pt-BR", "Português (Brasil)"]];
  const current = (pack && pack.lang) || "zh-CN";

  // the language picker at the bottom of the menu (sets a cookie the server reads, then reloads)
  function picker() {
    const side = document.getElementById("side");
    if (!side || document.getElementById("lang-pick")) return;
    const sel = document.createElement("select");
    sel.id = "lang-pick";
    sel.setAttribute("aria-label", "Language");
    sel.setAttribute("translate", "no");
    for (const [code, name] of LANGS) {
      const o = document.createElement("option");
      o.value = code; o.textContent = name;
      sel.append(o);
    }
    sel.value = current;
    sel.addEventListener("change", () => {
      const secure = location.protocol === "https:" ? "; Secure" : "";
      document.cookie = `violet_lang=${encodeURIComponent(sel.value)}; Path=/; Max-Age=31536000; SameSite=Strict${secure}`;
      location.reload();
    });
    const box = document.createElement("label");
    box.className = "lang-pick";
    box.setAttribute("translate", "no");
    box.append(sel);
    side.append(box);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", picker); else picker();
  if (!pack || !pack.dict) return;

  document.documentElement.lang = current;
  const HAN = /[㐀-鿿]/;
  const exact = new Map();
  const patterns = [];
  const prefixes = [];
  // 2026-11-09: keys that begin with "，" "；" "（" are tails the page appends to another text ("模型调用正常" +
  // "（这次启动后还没有调用过）", "…" + "，命中缓存 {0}"); found inside a text, the two halves are translated apart
  const TAIL = /^[，；（、]/;
  const tails = [];
  const CJK_OUT = /^(zh|ja)/.test(current);              // languages written without spaces between sentences
  for (const [key, value] of Object.entries(pack.dict)) {
    if (!value) continue;
    const k = key.replace(/\s+/g, " ").trim();
    if (/\{\d+\}/.test(k)) {
      // "a{0}b{1}c" -> fixed pieces [a, b, c] and the value numbers [0, 1]; matched by indexOf (linear, no regex
      // backtracking). Two values side by side ("{0}{1}") have no border to find: such a key is not used.
      const parts = k.split(/(\{\d+\})/);
      const pieces = parts.filter((p, i) => i % 2 === 0);
      const order = parts.filter((p, i) => i % 2 === 1).map((p) => Number(p.slice(1, -1)));
      const literal = pieces.join("");
      const fixed = literal.replace(/[\s\p{P}]/gu, "").length;
      // too little fixed text to match safely -- except one Chinese character around values that hold none
      // ("{0} 字", "群 {0}", "{0} 毫秒": numbers / ids only)
      if (fixed < 2 && !(fixed === 1 && HAN.test(literal))) continue;
      if (pieces.slice(1, -1).some((p) => p === "")) continue;
      const p = { pieces, order, value, weight: literal.length, short: fixed < 2 };
      patterns.push(p);
      if (TAIL.test(k)) tails.push(p);
    } else {
      exact.set(k, value);
      if (/[：:]$/.test(k) && k.length >= 3) prefixes.push(k);
      if (TAIL.test(k) && k.length >= 3) tails.push({ pieces: [k], exact: value, weight: k.length });
    }
  }
  patterns.sort((a, b) => b.weight - a.weight);
  prefixes.sort((a, b) => b.length - a.length);
  tails.sort((a, b) => b.weight - a.weight);
  const cache = new Map();
  const misses = new Set();

  // the values between the fixed pieces, or null: text must start with the first piece, end with the last, and
  // contain the others in order (each value is the shortest that lets the rest follow)
  function match(pieces, text) {
    const first = pieces[0], last = pieces[pieces.length - 1];
    if (!text.startsWith(first) || !text.endsWith(last) || text.length < first.length + last.length) return null;
    const end = text.length - last.length;
    const values = [];
    let at = first.length;
    for (let i = 1; i < pieces.length - 1; i++) {
      const j = text.indexOf(pieces[i], at);
      if (j < 0 || j > end) return null;
      values.push(text.slice(at, j));
      at = j + pieces[i].length;
    }
    if (at > end) return null;
    values.push(text.slice(at, end));
    return values;
  }

  // the first pattern (longest fixed text first) the whole text matches, filled in; null when none does
  function byPattern(text, depth, list) {
    for (const p of list) {
      const got = match(p.pieces, text);
      if (!got || (p.short && got.some((v) => HAN.test(v)))) continue;
      const values = p.order.map((n, i) => ({ n, v: got[i] }));
      return p.value.replace(/\{(\d+)\}/g, (_, n) => {
        const found = values.find((x) => x.n === Number(n));
        return found ? translateAt(found.v, depth + 1) : "";
      });
    }
    return null;
  }

  // a text that ends with a known tail: "head" + "，命中缓存 7040" / "（这次启动后还没有调用过）"
  function byTail(text, depth) {
    for (const t of tails) {
      const first = t.pieces[0];
      for (let i = text.indexOf(first, 1); i > 0; i = text.indexOf(first, i + 1)) {
        const rest = text.slice(i);
        const tail = t.exact !== undefined ? (rest === first ? t.exact : null) : byPattern(rest, depth, [t]);
        if (tail === null) continue;
        const headText = text.slice(0, i);
        const head = translateAt(headText, depth + 1);
        if (HAN.test(head) && head === headText) continue;          // the head stays Chinese: not this split
        return head + (/^[(\[]/.test(tail) && /[^\s(]$/.test(head) ? " " : "") + tail;
      }
    }
    return null;
  }

  // pieces joined by a separator, each translated; null unless one changed (all must, when `whole`)
  function bySplit(text, depth, splitter, joiner, whole) {
    const parts = text.split(splitter).filter((p) => p !== "");
    if (parts.length < 2) return null;
    const done = parts.map((p) => translateAt(p, depth + 1));
    const changed = done.map((d, i) => d !== parts[i] || !HAN.test(d));
    if (!changed.some((c, i) => c && HAN.test(parts[i]))) return null;
    if (whole && !changed.every(Boolean)) return null;
    return done.join(joiner);
  }

  function core(text, depth) {
    if (!HAN.test(text)) return text;
    const hit = exact.get(text);
    if (hit !== undefined) return hit;
    const flat = text.replace(/\s+/g, " ");                 // a key written over several lines in the page source
    if (flat !== text && exact.has(flat)) return exact.get(flat);
    const bare = text.replace(/。$/, "");                    // a phrase the page ends with "。" ("…文件" + "。")
    if (bare !== text && exact.has(bare)) {
      const v = exact.get(bare);
      return /[.!?。！？]$/.test(v) ? v : v + (CJK_OUT ? "。" : ".");
    }
    if (cache.has(text)) return cache.get(text);
    let out = null;
    if (depth < 4) {
      out = byPattern(text, depth, patterns);
      if (out === null) {                            // "上次安装失败：" + a message
        for (const k of prefixes) {
          if (text.length > k.length && text.startsWith(k)) {
            const rest = text.slice(k.length);
            out = exact.get(k) + (/^\s/.test(rest) || /\s$/.test(exact.get(k)) ? "" : " ") + translateAt(rest.trim(), depth + 1);
            break;
          }
        }
      }
      if (out === null && text.includes(" · ")) {          // "工具与手机 · Violet", "a · b · c"
        // the longest run of neighbouring parts that is a key of its own goes first ("<状态> · 收到 3 · 发出 2")
        const parts = text.split(" · ");
        const done = [];
        let changed = false;
        for (let i = 0; i < parts.length;) {
          let j = parts.length;
          let hit = null;
          for (; j > i + 1; j--) {
            const run = parts.slice(i, j).join(" · ");
            hit = exact.has(run) ? exact.get(run) : byPattern(run, depth, patterns);
            if (hit !== null) break;
          }
          if (hit === null) { hit = translateAt(parts[i], depth + 1); j = i + 1; }
          if (hit !== parts.slice(i, j).join(" · ")) changed = true;
          done.push(hit);
          i = j;
        }
        if (changed) out = done.join(" · ");
      }
      if (out === null) out = byTail(text, depth);
      // two sentences the page shows together ("…打开）；这里仍然可以录入和整理。 还没有课表：…")
      if (out === null && /[。！？]\s*\S/.test(text)) out = bySplit(text, depth, /(?<=[。！？])\s*/, CJK_OUT ? "" : " ", false);
      // a list of names ("微信、QQ（机器人）、Telegram"): only when every name has a translation
      if (out === null && text.includes("、")) out = bySplit(text, depth, "、", CJK_OUT ? "、" : ", ", true);
      if (out === null && text.includes("\n")) {
        const lines = text.split("\n");
        if (lines.length > 1) out = lines.map((l) => (l.trim() ? translateAt(l, depth + 1) : l)).join("\n");
      }
    }
    if (out === null) { misses.add(text); out = text; }
    cache.set(text, out);
    return out;
  }
  function translateAt(text, depth) {
    if (!text || !HAN.test(text)) return text;
    // a list bullet in front ("· CPU 支持 AVX2") stays as it is
    const bullet = /^\s*[·•]\s+/.exec(text);
    const body0 = text.slice(bullet ? bullet[0].length : 0).trim();
    const start = text.indexOf(body0, bullet ? bullet[0].length : 0);
    const lead = text.slice(0, start), trail = text.slice(start + body0.length);
    const body = body0.replace(/[ \t]+/g, " ");
    const out = core(body, depth);
    return out === body ? text : lead + out + trail;
  }
  const translate = (text) => translateAt(text, 0);
  window.violetT = translate;
  window.violetI18nMisses = () => [...misses].sort();

  // data, not the page's own words: prompts / results (pre), log lines, values (code), what the admin is typing
  const SKIP = 'script,style,textarea,code,pre,.log-msg,[translate="no"],[contenteditable="true"]';
  const ATTRS = ["placeholder", "title", "aria-label", "alt"];
  const done = new WeakMap();                  // text node -> what we wrote into it
  function textNode(node) {
    const p = node.parentElement;
    if (!p || p.closest(SKIP)) return;
    if (done.get(node) === node.data) return;
    const out = translate(node.data);
    if (out !== node.data) node.data = out;
    done.set(node, node.data);
  }
  // an attribute is written only when its translation differs, and what we wrote is remembered: writing it fires
  // the observer again, and a text that stays (partly) Chinese must not be written over and over
  const wrote = new WeakMap();                 // element -> {attribute: value we wrote}
  function attribute(el, a) {
    const v = el.getAttribute(a);
    if (!v || !HAN.test(v)) return;
    const mine = wrote.get(el);
    if (mine && mine[a] === v) return;
    const out = translate(v);
    if (out !== v) el.setAttribute(a, out);
    wrote.set(el, { ...(mine || {}), [a]: out });
  }
  function element(el) {
    if (el.closest(SKIP)) return;
    for (const a of ATTRS) attribute(el, a);
    // the guides linked from the page: the English ones outside Chinese
    if (el.tagName === "A" && !current.startsWith("zh")) {
      const href = el.getAttribute("href");
      if (href && /\.zh-CN\.md(#.*)?$/.test(href)) el.setAttribute("href", href.replace(".zh-CN.md", ".md"));
    }
  }
  function walk(root) {
    if (root.nodeType === Node.TEXT_NODE) { textNode(root); return; }
    if (root.nodeType !== Node.ELEMENT_NODE && root.nodeType !== Node.DOCUMENT_FRAGMENT_NODE) return;
    if (root.nodeType === Node.ELEMENT_NODE) {
      if (root.closest(SKIP)) return;
      element(root);
    }
    const tw = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (n.nodeType === Node.ELEMENT_NODE && n.matches(SKIP) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
    });
    for (let n = tw.nextNode(); n; n = tw.nextNode()) {
      if (n.nodeType === Node.TEXT_NODE) textNode(n); else element(n);
    }
  }
  const observer = new MutationObserver((records) => {
    for (const r of records) {
      if (r.type === "childList") r.addedNodes.forEach(walk);
      else if (r.type === "characterData") textNode(r.target);
      else if (r.type === "attributes" && r.target.nodeType === Node.ELEMENT_NODE && !r.target.closest(SKIP)) {
        attribute(r.target, r.attributeName);
      }
    }
  });
  function start() {
    walk(document.documentElement);             // the title too (the page sets it on every tab)
    observer.observe(document.documentElement, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ATTRS });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();

  // the browser's own boxes
  const wrap = (name) => { const original = window[name].bind(window); window[name] = (msg, ...rest) => original(translate(String(msg ?? "")), ...rest); };
  ["alert", "confirm", "prompt"].forEach(wrap);
})();
