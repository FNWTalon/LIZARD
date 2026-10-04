// ai: The phone's Chrome from the laptop, over adb and the DevTools protocol: every phone session's driver (it was
// ai: scratch until 2026-09-26). One phone on adb (ANDROID_SERIAL names one of several; adb reads it), Chrome open
// ai: on it with at least one tab. Each call forwards tcp:9222 to localabstract:chrome_devtools_remote first.
// ai:   node lizard-web/phone.mjs tabs                the page tabs: id, url, title
// ai:   node lizard-web/phone.mjs eval '<expression>' the value in the tab, promises awaited, as JSON
// ai:   node lizard-web/phone.mjs nav <url>           the tab navigated (Android's Chrome has no /json/new: an open
// ai:                                                 tab is reused), waiting for its load event
// ai:   node lizard-web/phone.mjs listen              the tab's console and browser log from now on
// ai:   node lizard-web/phone.mjs run <url>           nav, then listen, lines from before the navigation dropped
// ai:   node lizard-web/phone.mjs shot <file.png>     a screenshot of the tab
// ai: Options: --tab=<text> the tab whose id, url or title holds it (default the first on the rig, "/lizard-web/" in its
// ai: url, else the first); listen and run: --secs=N (default 60), --until=<regex> (stop at the first line that
// ai: matches: exit 0, or 1 if N seconds pass first), --out=<file> (the lines appended there too). A line is
// ai: "<level> <text>" with the page's arguments joined by spaces (objects as JSON). The DevTools session replays the
// ai: console lines it holds on Runtime.enable, so listen drops every line stamped before the phone's clock when it
// ai: started (run: before the navigation).
// ai: As a module: forward, tabs, pickTab, open (a Session: send, on, close), evaluate, navigate, listen, screenshot;
// ai: tabs and open take a port, so a local Chrome's --remote-debugging-port is driven the same way (lizard-web/check_send.mjs).
import { execFileSync } from "node:child_process";
import { appendFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const PORT = 9222;
// ai: the DevTools session is Node's own WebSocket, a global from Node 22: on an older Node the checks' connect loops
// ai: would retry and say only "no DevTools"
if (typeof WebSocket !== "function") throw new Error(`lizard-web/phone.mjs needs Node 22 or later (a global WebSocket); this is node ${process.version}`);

export function forward() {
  execFileSync("adb", ["forward", `tcp:${PORT}`, "localabstract:chrome_devtools_remote"], { stdio: ["ignore", "ignore", "inherit"] });
}

export async function tabs(port = PORT) {
  const res = await fetch(`http://127.0.0.1:${port}/json/list`);
  if (!res.ok) throw new Error(`/json/list: HTTP ${res.status}`);
  return (await res.json()).filter((t) => t.type === "page");
}

// ai: The tab whose id, url or title holds `want`; with none, the first on the rig, else the first.
export function pickTab(list, want = null) {
  const t = want ? list.find((x) => [x.id, x.url, x.title].some((s) => s?.includes(want))) : list.find((x) => x.url.includes("/lizard-web/")) ?? list[0];
  if (!t) throw new Error(want ? `no tab holds "${want}" (${list.length} tabs)` : "no page tab: open Chrome on the phone");
  return t;
}

// ai: A DevTools session on one tab. send(method, params) resolves with the result or throws the protocol's error;
// ai: on(event, fn) subscribes to an event.
export async function open(tab, port = PORT) {
  // ai: Android reports webSocketDebuggerUrl without a usable host, so the url is built from the forwarded port.
  const ws = new WebSocket(`ws://127.0.0.1:${port}/devtools/page/${tab.id}`);
  await new Promise((ok, no) => { ws.onopen = ok; ws.onerror = () => no(new Error(`cannot open the DevTools socket of tab ${tab.id}`)); });
  let next = 1;
  const waiting = new Map(), handlers = new Map();
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && waiting.has(msg.id)) {
      const { ok, no } = waiting.get(msg.id);
      waiting.delete(msg.id);
      if (msg.error) no(new Error(`${msg.error.message} (${msg.error.code})`)); else ok(msg.result);
    } else if (msg.method) for (const fn of handlers.get(msg.method) ?? []) fn(msg.params);
  };
  ws.onclose = () => { for (const { no } of waiting.values()) no(new Error("the DevTools socket closed")); waiting.clear(); };
  return {
    send: (method, params = {}) => new Promise((ok, no) => { const id = next++; waiting.set(id, { ok, no }); ws.send(JSON.stringify({ id, method, params })); }),
    on: (event, fn) => { if (!handlers.has(event)) handlers.set(event, []); handlers.get(event).push(fn); },
    close: () => ws.close(),
  };
}

export async function evaluate(s, expression) {
  const r = await s.send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(`eval: ${r.exceptionDetails.exception?.description ?? r.exceptionDetails.text}`);
  return r.result.value;
}

export async function navigate(s, url, timeoutMs = 30000) {
  await s.send("Page.enable");
  const loaded = new Promise((ok) => s.on("Page.loadEventFired", ok));
  const r = await s.send("Page.navigate", { url });
  if (r.errorText) throw new Error(`navigate ${url}: ${r.errorText}`);
  await Promise.race([loaded, new Promise((_, no) => setTimeout(() => no(new Error(`navigate ${url}: no load event in ${timeoutMs / 1000} s`)), timeoutMs))]);
}

const argText = (a) => (a.type === "string" ? a.value : a.value !== undefined ? (typeof a.value === "object" ? JSON.stringify(a.value) : String(a.value)) : a.description ?? a.type);

// ai: Console and browser-log lines stamped at or after `since` (the phone's ms since the epoch; default its clock
// ai: now), each to onLine; resolves { matched, lines } at the first line `until` matches, or after secs.
export async function listen(s, { secs = 60, until = null, since = null, onLine = () => {} } = {}) {
  const from = since ?? (await evaluate(s, "Date.now()"));
  const lines = [];
  let done;
  const finished = new Promise((ok) => { done = ok; });
  const take = (ts, level, text) => {
    if (ts < from) return;
    const line = `${level} ${text}`;
    lines.push(line);
    onLine(line);
    if (until && until.test(text)) done(true);
  };
  s.on("Runtime.consoleAPICalled", (p) => take(p.timestamp, p.type, p.args.map(argText).join(" ")));
  s.on("Log.entryAdded", ({ entry: e }) => take(e.timestamp, `${e.level}(${e.source})`, e.text));
  await s.send("Runtime.enable");
  await s.send("Log.enable");
  const timer = setTimeout(() => done(false), secs * 1000);
  const matched = await finished;
  clearTimeout(timer);
  return { matched, lines };
}

export async function screenshot(s, file) {
  const r = await s.send("Page.captureScreenshot", { format: "png" });
  writeFileSync(file, Buffer.from(r.data, "base64"));
}

async function main(argv) {
  const opts = {}, pos = [];
  for (const a of argv) { const m = /^--([a-z]+)=(.*)$/s.exec(a); if (m) opts[m[1]] = m[2]; else pos.push(a); }
  const [cmd, arg] = pos;
  const usage = "usage: node lizard-web/phone.mjs tabs | eval <expression> | nav <url> | listen | run <url> | shot <file.png>  [--tab=] [--secs=] [--until=] [--out=]";
  if (!cmd || (["eval", "nav", "run", "shot"].includes(cmd) && !arg)) throw new Error(usage);
  forward();
  const list = await tabs();
  if (cmd === "tabs") { for (const t of list) console.log(`${t.id}  ${t.url}  ${JSON.stringify(t.title)}`); return 0; }
  const tab = pickTab(list, opts.tab), s = await open(tab);
  try {
    if (cmd === "eval") { console.log(JSON.stringify(await evaluate(s, arg))); return 0; }
    if (cmd === "nav") { await navigate(s, arg); console.log(`tab ${tab.id} at ${arg}`); return 0; }
    if (cmd === "shot") { await screenshot(s, arg); console.log(`tab ${tab.id} (${tab.url}) to ${arg}`); return 0; }
    if (cmd === "listen" || cmd === "run") {
      const until = opts.until ? new RegExp(opts.until) : null, secs = +(opts.secs ?? 60);
      const onLine = (l) => { console.log(l); if (opts.out) appendFileSync(opts.out, `${l}\n`); };
      const since = await evaluate(s, "Date.now()");
      const heard = listen(s, { secs, until, since, onLine });
      if (cmd === "run") await navigate(s, arg);
      const { matched } = await heard;
      if (until && !matched) { console.error(`phone: no line matched /${opts.until}/ in ${secs} s`); return 1; }
      return 0;
    }
    throw new Error(usage);
  } finally { s.close(); }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) main(process.argv.slice(2)).then((code) => process.exit(code), (e) => { console.error(`phone: ${e.message}`); process.exit(1); });
