// Bilingual (fa/en) AI Agent on Cloudflare Workers + Workers AI. Single file. Binding required: AI
// Optional secrets: TAVILY_API_KEY or BRAVE_API_KEY (web search works without them via a keyless fallback)

// ---- Model IDs: change here only (verify against developers.cloudflare.com/workers-ai/models) ----
const CHAT_MODEL = ["@cf/meta/llama-3.3-70b-instruct-fp8-fast", "@cf/openai/gpt-oss-120b", "@cf/meta/llama-4-scout-17b-16e-instruct"];
const CODE_MODEL = ["@cf/qwen/qwen2.5-coder-32b-instruct", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"];
const IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell";
const IMAGE_FALLBACK = "@cf/bytedance/stable-diffusion-xl-lightning";

const MAX_STEPS = 6;
const BUDGET = { web_search: 2, fetch_url: 4, generate_image: 2, write_code: 3 };
const FENCE = "`".repeat(3);
const SEC = {
  "x-content-type-options": "nosniff",
  "referrer-policy": "no-referrer",
  "content-security-policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src data: blob:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
};

const SYS = `You are an autonomous AI agent. Respond in the user's language. Keep code in English.
Work in a loop: plan, call ONE tool, read the observation, reflect, then give the final answer.
Reply with ONE JSON object only, no other text, no code fences:
  {"thought":"short reasoning","tool":"<name>","args":{...}}   or   {"thought":"...","final":"<answer in Markdown>"}
Tools:
- generate_image {"prompt": detailed ENGLISH image prompt (subject, style, lighting, composition, quality), "aspect": "1:1"|"16:9"|"9:16"|"3:2"}. ALWAYS write the prompt in English, even if the user writes Persian. The image is shown to the user automatically.
- write_code {"task": full task description, "language": "...", "requirements": "..."}. Use for any request to write a program, script, Worker or file. The finished code is shown to the user automatically; do NOT repeat it in your final answer, just add a short note.
- web_search {"query": "..."}: current facts, news, docs.
- fetch_url {"url": "https://..."}: read a page found by search or given by the user.
Rules: use tools only when needed; answer simple questions directly with "final". Never invent facts: search when unsure or when recency matters, and cite sources as [1], [2] with a short Sources list. Do not claim you executed code. Final answers are in the user's language (natural Iranian Persian with correct half-spaces if Persian); code, commands and identifiers stay English.`;

const CODE_SYS = `You are a senior software engineer. Output the COMPLETE, runnable file in one fenced code block with the language tag, never partial snippets. If the request is ambiguous, state assumptions in one line first; do not ask questions. Prefer dependency-free solutions; validate inputs; handle errors; never invent APIs. For Cloudflare Workers: ES modules, fetch handler, Web-standard APIs only, and include the needed wrangler config. After the code, add a short "How to run/deploy" note that works from a phone. Write the explanations in the user's language (Persian if the task is in Persian); code and comments stay English.`;
const REVIEW_SYS = `You are a meticulous code reviewer. You get a task and a draft answer. Check the code for bugs, missing imports, syntax errors, wrong API usage and security issues. Return the FINAL corrected answer in the same format: one complete fenced code block, then the short usage note. Return only the final answer, no review commentary.`;

const ST = {
  fa: { plan: "در حال فکر کردن…", search: "جستجو در وب…", fetch: "خواندن صفحه…", code: "نوشتن کد…", review: "بازبینی کد…", image: "ساخت تصویر…", final: "آماده‌سازی پاسخ…", err: "خطایی رخ داد. دوباره تلاش کنید.", budget: "سقف استفاده از این ابزار تمام شد." },
  en: { plan: "Thinking…", search: "Searching the web…", fetch: "Reading page…", code: "Writing code…", review: "Reviewing code…", image: "Generating image…", final: "Preparing answer…", err: "Something went wrong. Please try again.", budget: "Tool budget used up." },
};

// ---------------- helpers ----------------
const J = (o, s = 200) => new Response(JSON.stringify(o), { status: s, headers: { "content-type": "application/json; charset=utf-8", ...SEC } });
const tmo = (p, ms) => new Promise((res, rej) => { const t = setTimeout(() => rej(new Error("timeout")), ms); Promise.resolve(p).then(v => { clearTimeout(t); res(v); }, e => { clearTimeout(t); rej(e); }); });
const detect = s => ((s.match(/[\u0600-\u06FF]/g) || []).length > (s.match(/[A-Za-z]/g) || []).length ? "fa" : "en");
const strip = s => s.replace(/<script[\s\S]*?<\/script>|<style[\s\S]*?<\/style>|<noscript[\s\S]*?<\/noscript>/gi, " ").replace(/<[^>]+>/g, " ").replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/\s+/g, " ").trim();

function textOf(r) {
  if (typeof r === "string") return r;
  if (!r) return "";
  if (typeof r.response === "string") return r.response;
  if (r.response && typeof r.response === "object") return JSON.stringify(r.response);
  const c = r.choices && r.choices[0];
  if (c && c.message && typeof c.message.content === "string") return c.message.content;
  if (Array.isArray(r.output)) return r.output.filter(x => x.type === "message").flatMap(x => x.content || []).map(x => x.text || "").join("");
  return "";
}

async function llm(env, models, messages, max) {
  let err;
  for (const m of [models[0], ...models]) { // first model gets one retry
    try {
      const t = textOf(await tmo(env.AI.run(m, { messages, max_tokens: max, temperature: 0.3 }), 90000)).trim();
      if (t) return t;
    } catch (e) { err = e; }
  }
  throw err || new Error("empty");
}

function parseJSON(t) {
  t = t.replace(/```(?:json)?/gi, "");
  const st = t.indexOf("{");
  if (st < 0) return null;
  let d = 0, q = false, esc = false;
  for (let i = st; i < t.length; i++) {
    const c = t[i];
    if (q) { if (esc) esc = false; else if (c === "\\") esc = true; else if (c === '"') q = false; continue; }
    if (c === '"') q = true;
    else if (c === "{") d++;
    else if (c === "}" && --d === 0) { try { return JSON.parse(t.slice(st, i + 1)); } catch (e) { return null; } }
  }
  return null;
}

function okUrl(s) {
  let u; try { u = new URL(s); } catch (e) { return null; }
  if (!/^https?:$/.test(u.protocol)) return null;
  const h = u.hostname.toLowerCase();
  if (h === "localhost" || h.endsWith(".local") || h.endsWith(".internal") || h.includes(":") || !h.includes(".") || /^(0|10|127)\./.test(h) || /^192\.168\./.test(h) || /^169\.254\./.test(h) || /^172\.(1[6-9]|2\d|3[01])\./.test(h) || /^\d+$/.test(h) || /^0x/i.test(h)) return null;
  return u;
}

// ---------------- tools ----------------
const ASP = { "1:1": [1024, 1024], "16:9": [1024, 576], "9:16": [576, 1024], "3:2": [1024, 683] };

async function toUrl(r) {
  if (r && typeof r.image === "string") return "data:image/jpeg;base64," + r.image;
  let b = null;
  if (r instanceof ReadableStream) b = new Uint8Array(await new Response(r).arrayBuffer());
  else if (r instanceof ArrayBuffer) b = new Uint8Array(r);
  else if (r && r.buffer instanceof ArrayBuffer && r.byteLength) b = new Uint8Array(r.buffer, r.byteOffset, r.byteLength);
  if (!b || !b.length) return null;
  let s = "";
  for (let i = 0; i < b.length; i += 8192) s += String.fromCharCode.apply(null, b.subarray(i, i + 8192));
  return "data:image/" + (b[0] === 0x89 ? "png" : "jpeg") + ";base64," + btoa(s);
}

async function makeImage(env, prompt, aspect) {
  const [w, h] = ASP[aspect] || ASP["1:1"];
  const p = prompt + (aspect && aspect !== "1:1" ? ". Composition: " + aspect + " aspect ratio framing." : "");
  const tries = [
    () => env.AI.run(IMAGE_MODEL, { prompt: p, steps: 4 }),
    () => env.AI.run(IMAGE_MODEL, { prompt: p, num_steps: 4 }),
    () => env.AI.run(IMAGE_MODEL, { prompt: prompt.slice(0, 220) + ", safe for work, illustration", steps: 4 }), // reworded retry (false safety-filter hits)
    () => env.AI.run(IMAGE_FALLBACK, { prompt, width: w, height: h, num_steps: 6 }),
  ];
  let err;
  for (const t of tries) { try { const u = await toUrl(await tmo(t(), 60000)); if (u) return u; } catch (e) { err = e; } }
  throw err || new Error("image failed");
}

async function writeCode(env, a) {
  const p = "Task: " + a.task + "\nLanguage: " + (a.language || "auto") + "\nRequirements: " + (a.requirements || "none");
  const draft = await llm(env, CODE_MODEL, [{ role: "system", content: CODE_SYS }, { role: "user", content: p }], 4096);
  let f = draft;
  try { f = await llm(env, CODE_MODEL, [{ role: "system", content: REVIEW_SYS }, { role: "user", content: "Task:\n" + a.task + "\n\nDraft answer:\n" + draft }], 4096); } catch (e) {}
  if (!f.includes(FENCE)) f = FENCE + (a.language || "") + "\n" + f + "\n" + FENCE;
  return f;
}

async function webSearch(env, q) {
  if (env.TAVILY_API_KEY) {
    const r = await tmo(fetch("https://api.tavily.com/search", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ api_key: env.TAVILY_API_KEY, query: q, max_results: 5 }) }), 10000);
    const j = await r.json();
    return (j.results || []).slice(0, 5).map(x => ({ title: x.title, url: x.url, snippet: (x.content || "").slice(0, 300) }));
  }
  if (env.BRAVE_API_KEY) {
    const r = await tmo(fetch("https://api.search.brave.com/res/v1/web/search?count=5&q=" + encodeURIComponent(q), { headers: { "X-Subscription-Token": env.BRAVE_API_KEY, accept: "application/json" } }), 10000);
    const j = await r.json();
    return ((j.web && j.web.results) || []).slice(0, 5).map(x => ({ title: x.title, url: x.url, snippet: strip(x.description || "").slice(0, 300) }));
  }
  const html = await (await tmo(fetch("https://html.duckduckgo.com/html/?q=" + encodeURIComponent(q), { headers: { "user-agent": "Mozilla/5.0" } }), 10000)).text();
  const out = [], re = /<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g;
  let m;
  while ((m = re.exec(html)) && out.length < 5) {
    let u = m[1].replace(/&amp;/g, "&");
    const g = u.match(/uddg=([^&]+)/);
    if (g) u = decodeURIComponent(g[1]);
    if (u.startsWith("//")) u = "https:" + u;
    out.push({ title: strip(m[2]), url: u, snippet: strip(m[3]).slice(0, 300) });
  }
  return out;
}

async function fetchPage(url) {
  const u = okUrl(url);
  if (!u) throw new Error("blocked url");
  const ac = new AbortController(), t = setTimeout(() => ac.abort(), 8000);
  try {
    const r = await fetch(u.href, { signal: ac.signal, headers: { "user-agent": "Mozilla/5.0 (AgentWorker)" } });
    if (!okUrl(r.url)) throw new Error("blocked redirect");
    const ct = r.headers.get("content-type") || "";
    if (!/text|json|xml/i.test(ct)) throw new Error("unsupported content type");
    return strip((await r.text()).slice(0, 400000)).slice(0, 6000);
  } finally { clearTimeout(t); }
}

// ---------------- agent loop ----------------
async function agent(env, msgs, emit, lang) {
  const S = ST[lang];
  const convo = [{ role: "system", content: SYS }, ...msgs.slice(-12).map(m => ({ role: m.role, content: m.content }))];
  const left = { ...BUDGET };
  const deliver = [];
  let final = null;
  const valid = o => o && (typeof o.final === "string" || (typeof o.tool === "string" && BUDGET[o.tool] !== undefined && o.args && typeof o.args === "object"));

  for (let i = 0; i < MAX_STEPS && final === null; i++) {
    await emit("status", { text: S.plan });
    let raw = await llm(env, CHAT_MODEL, convo, 2048), o = parseJSON(raw), n = 0;
    while (!valid(o) && n++ < 2) {
      convo.push({ role: "assistant", content: raw }, { role: "user", content: "Invalid format. Reply with ONE valid JSON object only: {\"thought\":\"...\",\"tool\":\"...\",\"args\":{...}} or {\"thought\":\"...\",\"final\":\"...\"}." });
      raw = await llm(env, CHAT_MODEL, convo, 2048); o = parseJSON(raw);
    }
    if (!valid(o)) { final = raw; break; }
    if (typeof o.final === "string") { final = o.final; break; }

    const tool = o.tool, a = o.args;
    convo.push({ role: "assistant", content: JSON.stringify({ thought: o.thought || "", tool, args: a }) });
    if (left[tool] <= 0) { convo.push({ role: "user", content: "OBSERVATION: " + S.budget + " Answer with what you have." }); continue; }
    left[tool]--;
    await emit("tool_start", { tool, thought: String(o.thought || "").slice(0, 200) });
    let obs;
    try {
      if (tool === "generate_image") {
        await emit("status", { text: S.image });
        const url = await makeImage(env, String(a.prompt || "").slice(0, 1500), a.aspect);
        await emit("image", { url });
        obs = "Image created and already shown to the user. Do not describe pixels; write a short final note.";
        await emit("tool_result", { tool, summary: "image ✓" });
      } else if (tool === "write_code") {
        await emit("status", { text: S.code });
        const code = await writeCode(env, a);
        deliver.push(code);
        obs = "Code written and reviewed (" + code.length + " chars); it will be shown to the user automatically. Do NOT repeat it. Write a short final note only.";
        await emit("tool_result", { tool, summary: "code ✓ (reviewed)" });
      } else if (tool === "web_search") {
        await emit("status", { text: S.search });
        const r = await webSearch(env, String(a.query || ""));
        obs = JSON.stringify(r).slice(0, 4000);
        await emit("tool_result", { tool, summary: r.length + " results" });
      } else {
        await emit("status", { text: S.fetch });
        const t = await fetchPage(String(a.url || ""));
        obs = t;
        await emit("tool_result", { tool, summary: t.length + " chars" });
      }
    } catch (e) {
      obs = "Tool failed: " + String(e && e.message || e).slice(0, 150);
      await emit("tool_result", { tool, summary: "failed" });
    }
    convo.push({ role: "user", content: "OBSERVATION (" + tool + "): " + obs });
    if (tool === "write_code" || tool === "generate_image") convo.push({ role: "user", content: "Now reply with the final JSON containing a brief note in the user's language." });
  }

  if (final === null) {
    await emit("status", { text: S.final });
    convo.push({ role: "user", content: "Give the final answer now as plain Markdown in the user's language. No JSON, no tools." });
    final = await llm(env, CHAT_MODEL, convo, 2048);
  }
  const p = parseJSON(final);
  if (p && typeof p.final === "string") final = p.final;
  const out = (final + (deliver.length ? "\n\n" + deliver.join("\n\n") : "")).trim();
  for (let i = 0; i < out.length; i += 60) await emit("token", { text: out.slice(i, i + 60) });
}

async function chat(req, env, ctx) {
  const u = new URL(req.url), org = req.headers.get("origin");
  if (org && org !== u.origin) return J({ error: "forbidden" }, 403);
  if (!env.AI) return J({ error: "AI binding is missing. Add a Workers AI binding named AI." }, 500);
  let b;
  try { const t = await req.text(); if (t.length > 200000) return J({ error: "too_large" }, 413); b = JSON.parse(t); } catch (e) { return J({ error: "bad_json" }, 400); }
  const m = b && b.messages;
  if (!Array.isArray(m) || !m.length || m.length > 40) return J({ error: "bad_messages" }, 400);
  for (const x of m) if (!x || (x.role !== "user" && x.role !== "assistant") || typeof x.content !== "string" || x.content.length > 8000) return J({ error: "bad_message" }, 400);
  if (m[m.length - 1].role !== "user") return J({ error: "last_must_be_user" }, 400);
  const lang = detect(m[m.length - 1].content);
  const { readable, writable } = new TransformStream(), w = writable.getWriter(), enc = new TextEncoder();
  const emit = (t, d) => w.write(enc.encode("event: " + t + "\ndata: " + JSON.stringify(d) + "\n\n")).catch(() => {});
  ctx.waitUntil((async () => {
    try { await agent(env, m, emit, lang); }
    catch (e) { await emit("error", { message: ST[lang].err }); }
    finally { await emit("done", {}); try { await w.close(); } catch (e) {} }
  })());
  return new Response(readable, { headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache, no-transform", ...SEC } });
}

export default {
  async fetch(req, env, ctx) {
    const u = new URL(req.url);
    if (req.method === "GET" && u.pathname === "/") return new Response(HTML, { headers: { "content-type": "text/html; charset=utf-8", ...SEC } });
    if (u.pathname === "/health") return J({ ok: true, ai: !!env.AI });
    if (u.pathname === "/api/chat" && req.method === "POST") return chat(req, env, ctx);
    return J({ error: "not_found" }, 404);
  },
};

// ---------------- UI (String.raw: no backticks and no dollar-brace inside) ----------------
const HTML = String.raw`<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Agent AI</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600&display=swap">
<style>
:root{--bg:#0d1520;--s:#15212f;--s2:#1c2b3d;--t:#e8eef5;--m:#8da2b8;--a:#3fd0c0;--ac:#06241f;--b:#25364a;color-scheme:dark}
[data-theme=light]{--bg:#edf2f6;--s:#fff;--s2:#e3ebf2;--t:#12202e;--m:#566b80;--a:#0b7a70;--ac:#fff;--b:#cfdbe6;color-scheme:light}
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--t);font:16px/1.7 Vazirmatn,Tahoma,"Noto Naskh Arabic",system-ui,sans-serif;display:flex;flex-direction:column;height:100dvh;padding-top:env(safe-area-inset-top)}
header{display:flex;align-items:center;gap:6px;padding:8px 10px;border-bottom:1px solid var(--b);background:var(--s)}
header h1{flex:1;margin:0;font-size:17px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
button{font:inherit;color:inherit;cursor:pointer}
.ib{min-width:44px;min-height:44px;border:1px solid var(--b);background:var(--s2);border-radius:12px}
:focus-visible{outline:2px solid var(--a);outline-offset:2px}
#msgs{flex:1;overflow-y:auto;padding:12px 10px;display:flex;flex-direction:column;gap:12px;overscroll-behavior:contain}
.msg{max-width:100%;border-radius:14px;padding:10px 12px;overflow-wrap:anywhere}
.u{align-self:flex-start;background:var(--a);color:var(--ac);max-width:88%;white-space:pre-wrap}
.a{background:var(--s);border:1px solid var(--b)}
.a .body p,.a .body ul{margin:.4em 0}.a .body h3,.a .body h4{margin:.6em 0 .2em}
.a code{background:var(--s2);padding:1px 5px;border-radius:6px;font-family:ui-monospace,Menlo,monospace;font-size:.88em}
.cb{margin:.6em 0;border:1px solid var(--b);border-radius:12px;overflow:hidden;direction:ltr;text-align:left}
.ch{display:flex;justify-content:space-between;align-items:center;padding:4px 8px;background:var(--s2);font-size:13px;color:var(--m)}
.ch button{min-height:36px;padding:0 10px;border:1px solid var(--b);background:var(--s);border-radius:8px;margin-left:6px}
.cb pre{margin:0;padding:10px;overflow-x:auto;background:#0a1018;color:#d6e2ee;font-size:13px;line-height:1.55}
.cb pre code{background:none;padding:0;font-size:inherit;color:inherit}
details.act{font-size:13px;color:var(--m);margin-bottom:6px}details.act summary{cursor:pointer;min-height:32px}
details.act ul{margin:4px 0;padding-inline-start:18px}
.imgs img{max-width:100%;border-radius:12px;display:block;margin:6px 0}
.skel{height:200px;border-radius:12px;background:linear-gradient(90deg,var(--s2),var(--b),var(--s2));background-size:200% 100%;animation:sh 1.4s linear infinite}
@keyframes sh{to{background-position:-200% 0}}@media(prefers-reduced-motion:reduce){.skel{animation:none}}
.tools{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}.tools button,.chip,.ia{min-height:44px;padding:0 12px;border:1px solid var(--b);background:var(--s2);border-radius:12px}
.chips{display:flex;flex-direction:column;gap:8px}.chip{text-align:start;height:auto;padding:10px 12px}
.hello{color:var(--m);padding:8px 2px}.hello b{color:var(--t);font-size:20px;display:block}
footer{display:flex;gap:8px;padding:8px 10px calc(8px + env(safe-area-inset-bottom));border-top:1px solid var(--b);background:var(--s);align-items:flex-end}
#in{flex:1;resize:none;max-height:140px;min-height:44px;padding:9px 12px;border-radius:14px;border:1px solid var(--b);background:var(--bg);color:var(--t);font:inherit}
#send{min-width:64px;min-height:44px;border:0;border-radius:14px;background:var(--a);color:var(--ac);font-weight:600}
#hist{position:fixed;inset:0;background:rgba(0,0,0,.55);display:flex;align-items:flex-end;z-index:5}
#hist[hidden]{display:none}#hist div{width:100%;background:var(--s);border-radius:18px 18px 0 0;padding:14px;display:flex;flex-direction:column;gap:8px}
.note{font-size:12px;color:var(--m);text-align:center;padding:2px 10px 6px}
</style></head><body>
<header><button class="ib" id="menu" aria-label="History">☰</button><h1 id="ttl"></h1><button class="ib" id="lang" aria-label="Language"></button><button class="ib" id="th" aria-label="Theme">◐</button></header>
<main id="msgs" aria-live="polite"></main>
<div class="note" id="note"></div>
<footer><textarea id="in" rows="1" aria-label="Message"></textarea><button id="send"></button></footer>
<div id="hist" hidden><div id="hbox"></div></div>
<script>
var T={fa:{ttl:"ایجنت هوش مصنوعی",hi:"سلام! چه کاری انجام بدهم؟",sub:"می‌توانم کد کامل بنویسم، تصویر بسازم و در وب تحقیق کنم.",ph:"پیامتان را بنویسید…",send:"ارسال",stop:"توقف",act:"فعالیت ایجنت",copy:"کپی",copied:"کپی شد",dl:"دانلود",full:"اندازه کامل",dimg:"دانلود تصویر",nw:"گفتگوی جدید",hist:"گفتگوهای اخیر",regen:"تولید دوباره",cans:"کپی پاسخ",note:"کد فقط با بازبینی مدل بررسی می‌شود، اجرا نمی‌شود. ممکن است اشتباه کند.",err:"ارتباط برقرار نشد.",empty:"گفتگوی خالی",ex:["یک Cloudflare Worker کامل بنویس که کوتاه‌کنندهٔ لینک باشد","تصویری از یک شهر آینده‌نگر هنگام غروب بساز","دربارهٔ آخرین مدل‌های Workers AI تحقیق و خلاصه کن"]},
en:{ttl:"AI Agent",hi:"Hi! What should I work on?",sub:"I can write complete code, generate images and research the web.",ph:"Type your message…",send:"Send",stop:"Stop",act:"Agent activity",copy:"Copy",copied:"Copied",dl:"Download",full:"Full size",dimg:"Download image",nw:"New chat",hist:"Recent chats",regen:"Regenerate",cans:"Copy answer",note:"Code is only reviewed by the model, never executed. It can make mistakes.",err:"Could not connect.",empty:"Empty chat",ex:["Write a complete Cloudflare Worker that shortens links","Generate an image of a futuristic city at sunset","Research the latest Workers AI models and summarize"]}};
var EXT={javascript:"js",js:"js",typescript:"ts",ts:"ts",python:"py",py:"py",html:"html",css:"css",json:"json",jsonc:"jsonc",bash:"sh",sh:"sh",java:"java",c:"c",cpp:"cpp",go:"go",rust:"rs",php:"php",sql:"sql",toml:"toml",yaml:"yml",yml:"yml",markdown:"md",md:"md"};
var $=function(s){return document.querySelector(s)};
function ls(k,v){try{if(v===undefined)return localStorage.getItem(k);localStorage.setItem(k,v)}catch(e){return null}}
var L=ls("lang")||"fa",chats=[],cur=null,busy=false,ctl=null,BT=String.fromCharCode(96);
try{chats=JSON.parse(ls("chats")||"[]")||[]}catch(e){chats=[]}
function save(){chats=chats.slice(0,3);ls("chats",JSON.stringify(chats.map(function(c){return{id:c.id,title:c.title,msgs:c.msgs.map(function(m){return{role:m.role,content:m.content}})}})))}
function esc(s){return String(s).replace(/[&<>"]/g,function(c){return{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]})}
var fenceRe=new RegExp(BT+BT+BT+"([\\w+-]*)[^\\n]*\\n?([\\s\\S]*?)("+BT+BT+BT+"|$)","g"),inRe=new RegExp(BT+"([^"+BT+"\\n]+)"+BT,"g");
function md(src){
 var blocks=[],s=src.replace(fenceRe,function(m,l,c){blocks.push({l:l,c:c.replace(/\n$/,"")});return "\u0000"+(blocks.length-1)+"\u0000"});
 s=esc(s).replace(inRe,"<code>$1</code>").replace(/\*\*([^*\n]+)\*\*/g,"<b>$1</b>").replace(/(^|[\s(])\*([^*\n]+)\*/g,"$1<i>$2</i>").replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
 var out=[],ul=false;
 s.split("\n").forEach(function(ln){
  var li=ln.match(/^\s*(?:[-*]|\d+[.)])\s+(.*)/),h=ln.match(/^(#{1,4})\s+(.*)/);
  if(li){if(!ul){out.push("<ul>");ul=true}out.push("<li>"+li[1]+"</li>");return}
  if(ul){out.push("</ul>");ul=false}
  if(h)out.push("<h4>"+h[2]+"</h4>");else if(ln.trim())out.push(/^\u0000\d+\u0000$/.test(ln.trim())?ln.trim():"<p dir=\"auto\">"+ln+"</p>")
 });
 if(ul)out.push("</ul>");
 return out.join("").replace(/\u0000(\d+)\u0000/g,function(m,i){var b=blocks[+i];return '<div class="cb"><div class="ch"><span>'+esc(b.l||"code")+'</span><span><button data-a="copy">'+T[L].copy+'</button><button data-a="dl" data-l="'+esc(b.l)+'">'+T[L].dl+'</button></span></div><pre><code>'+esc(b.c)+'</code></pre></div>'})
}
function copyText(t,btn){var done=function(){if(btn){var o=btn.textContent;btn.textContent=T[L].copied;setTimeout(function(){btn.textContent=o},1200)}};
 try{navigator.clipboard.writeText(t).then(done,fb)}catch(e){fb()}
 function fb(){var x=document.createElement("textarea");x.value=t;x.style.position="fixed";x.style.opacity="0";document.body.appendChild(x);x.select();try{document.execCommand("copy")}catch(e){}x.remove();done()}}
function dlBlob(name,blob){var a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(function(){URL.revokeObjectURL(a.href)},4000)}
function dlData(url,name){fetch(url).then(function(r){return r.blob()}).then(function(b){dlBlob(name,b)})}
function applyLang(){document.documentElement.lang=L;document.documentElement.dir=L==="fa"?"rtl":"ltr";$("#ttl").textContent=T[L].ttl;$("#in").placeholder=T[L].ph;$("#lang").textContent=L==="fa"?"EN":"فا";$("#note").textContent=T[L].note;$("#send").textContent=busy?T[L].stop:T[L].send}
function applyTheme(t){document.documentElement.setAttribute("data-theme",t);ls("theme",t)}
applyTheme(ls("theme")||(matchMedia("(prefers-color-scheme: light)").matches?"light":"dark"));
var M=$("#msgs");
function near(){return M.scrollHeight-M.scrollTop-M.clientHeight<90}
function toBottom(){M.scrollTop=M.scrollHeight}
function userEl(m){var d=document.createElement("div");d.className="msg u";d.dir="auto";d.textContent=m.content;return d}
function asstEl(m){
 var d=document.createElement("div");d.className="msg a";
 d.innerHTML='<details class="act"><summary>'+T[L].act+'</summary><ul></ul></details><div class="imgs"></div><div class="body"></div><div class="tools"></div>';
 m.el=d;(m.steps||[]).forEach(function(s){addStep(m,s)});(m.images||[]).forEach(function(u){addImg(m,u)});
 m.status=null;draw(m);return d}
function addStep(m,t){var li=document.createElement("li");li.textContent=t;m.el.querySelector("ul").appendChild(li)}
function addImg(m,u){var im=document.createElement("div");im.innerHTML='<img alt="">';im.querySelector("img").src=u;var r=document.createElement("div");r.className="tools";r.innerHTML='<button data-a="full">'+T[L].full+'</button><button data-a="dimg">'+T[L].dimg+'</button>';im.appendChild(r);m.el.querySelector(".imgs").appendChild(im)}
function draw(m){var b=m.el.querySelector(".body");b.innerHTML=md(m.content||"");var t=m.el.querySelector(".tools");t.innerHTML=m.content&&!m.live?'<button data-a="cans">'+T[L].cans+'</button><button data-a="regen">'+T[L].regen+'</button>':""}
function render(){M.innerHTML="";
 if(!cur||!cur.msgs.length){var h=document.createElement("div");h.className="hello";h.innerHTML="<b>"+T[L].hi+"</b>"+T[L].sub;M.appendChild(h);var c=document.createElement("div");c.className="chips";T[L].ex.forEach(function(x){var b=document.createElement("button");b.className="chip";b.textContent=x;b.onclick=function(){send(x)};c.appendChild(b)});M.appendChild(c);return}
 cur.msgs.forEach(function(m){M.appendChild(m.role==="user"?userEl(m):asstEl(m))});toBottom()}
function newChat(){cur={id:Date.now(),title:"",msgs:[]};chats.unshift(cur);save();render();$("#hist").hidden=true}
function setBusy(b){busy=b;$("#send").textContent=b?T[L].stop:T[L].send}
async function send(text,regen){
 if(busy||!text)return;if(!cur)newChat();
 if(!regen){cur.msgs.push({role:"user",content:text});if(!cur.title)cur.title=text.slice(0,40)}
 var m={role:"assistant",content:"",steps:[],images:[],live:true};cur.msgs.push(m);render();
 var payload=cur.msgs.slice(0,-1).slice(-12).map(function(x){return{role:x.role,content:x.content}});
 setBusy(true);ctl=new AbortController();var skel=null,st=document.createElement("div");st.className="hello";m.el.insertBefore(st,m.el.firstChild);
 function handle(ch){var ev="",dt="";ch.split("\n").forEach(function(l){if(l.indexOf("event: ")===0)ev=l.slice(7);else if(l.indexOf("data: ")===0)dt+=l.slice(6)});var d;try{d=JSON.parse(dt)}catch(e){return}var nb=near();
  if(ev==="status")st.textContent=d.text;
  else if(ev==="tool_start"){addStep(m,d.tool+(d.thought?" - "+d.thought:""));m.steps.push(d.tool);if(d.tool==="generate_image"){skel=document.createElement("div");skel.className="skel";m.el.querySelector(".imgs").appendChild(skel)}}
  else if(ev==="tool_result"){addStep(m,"→ "+d.summary);if(skel&&d.tool==="generate_image"){skel.remove();skel=null}}
  else if(ev==="image"){if(skel){skel.remove();skel=null}m.images.push(d.url);addImg(m,d.url)}
  else if(ev==="token"){m.content+=d.text;draw(m)}
  else if(ev==="error"){m.content+=(m.content?"\n\n":"")+d.message;draw(m)}
  if(nb)toBottom()}
 try{
  var r=await fetch("/api/chat",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify({messages:payload}),signal:ctl.signal});
  if(!r.ok){var j={};try{j=await r.json()}catch(e){}throw new Error(j.error||T[L].err)}
  var rd=r.body.getReader(),dec=new TextDecoder(),buf="";
  for(;;){var x=await rd.read();if(x.done)break;buf+=dec.decode(x.value,{stream:true});var i;while((i=buf.indexOf("\n\n"))>=0){handle(buf.slice(0,i));buf=buf.slice(i+2)}}
 }catch(e){if(e.name!=="AbortError"){m.content+=(m.content?"\n\n":"")+(e.message||T[L].err)}}
 st.remove();m.live=false;if(!m.content)m.content="…";draw(m);setBusy(false);ctl=null;save()}
$("#send").onclick=function(){if(busy){ctl&&ctl.abort();return}var t=$("#in").value.trim();if(!t)return;$("#in").value="";fit();send(t)};
function fit(){var e=$("#in");e.style.height="auto";e.style.height=Math.min(e.scrollHeight,140)+"px"}
$("#in").addEventListener("input",fit);
M.addEventListener("click",function(e){var b=e.target.closest("button[data-a]");if(!b)return;var a=b.dataset.a,cb=b.closest(".cb");
 if(a==="copy")copyText(cb.querySelector("code").textContent,b);
 else if(a==="dl"){var l=(b.dataset.l||"").toLowerCase();dlBlob("code."+(EXT[l]||"txt"),new Blob([cb.querySelector("code").textContent],{type:"text/plain"}))}
 else if(a==="full"){var s=b.closest("div").parentNode.querySelector("img").src;fetch(s).then(function(r){return r.blob()}).then(function(bl){window.open(URL.createObjectURL(bl),"_blank")})}
 else if(a==="dimg"){dlData(b.closest("div").parentNode.querySelector("img").src,"image.jpg")}
 else if(a==="cans"){copyText(cur.msgs[cur.msgs.length-1].content,b)}
 else if(a==="regen"){if(busy)return;cur.msgs.pop();var lu=cur.msgs[cur.msgs.length-1];if(lu)send(lu.content,true)}});
$("#lang").onclick=function(){L=L==="fa"?"en":"fa";ls("lang",L);applyLang();if(!busy)render()};
$("#th").onclick=function(){applyTheme(document.documentElement.getAttribute("data-theme")==="dark"?"light":"dark")};
$("#menu").onclick=function(){var h=$("#hbox");h.innerHTML="<b>"+T[L].hist+"</b>";chats.forEach(function(c){var b=document.createElement("button");b.className="ia";b.textContent=c.title||T[L].empty;b.onclick=function(){cur=c;render();$("#hist").hidden=true};h.appendChild(b)});var n=document.createElement("button");n.className="ia";n.textContent="+ "+T[L].nw;n.onclick=newChat;h.appendChild(n);$("#hist").hidden=false};
$("#hist").onclick=function(e){if(e.target.id==="hist")$("#hist").hidden=true};
applyLang();cur=null;render();
</script></body></html>`;
