// ai: Home (2026-10-01): Send and Receive (the solid one Receive on a touch screen, Send otherwise: one solid button at a time), the received
// ai: files (lizard-web/library.mjs: a row opens its file; its more menu holds Open, Save, Share where the browser shares such
// ai: a file, and Delete) and Settings (the files' size and Delete all, About, Source code). Receive goes to
// ai: recv.html?auto: the tap is the wish to start.
import { bytes, about, registerApp, confirmTap, icon } from "./ui.mjs";
import { list, open, remove, removeAll, watch, keeps, unfinished, forgetUnfinished } from "./library.mjs";

const $ = (id) => document.getElementById(id);
about($("about")); about($("aboutPc"));
registerApp(() => { about($("about"), true); about($("aboutPc"), true); });
// ai: About on a PC: the bar's button opens the credits in a dialog (the phone's fold holds the same)
$("aboutTop").onclick = () => $("aboutDlg").showModal();
$("aboutClose").onclick = () => $("aboutDlg").close();
$(matchMedia("(pointer: coarse)").matches ? "recv" : "send").classList.add("primary");
// ai: the PC's layout (index.html's media query): its received files' section stays with no files, an empty box in it
const PC = matchMedia("(min-width: 900px) and (pointer: fine), (min-width: 900px) and (pointer: none)");
PC.addEventListener("change", () => show());

// ai: Each row's file is a File over its OPFS bytes (nothing read until used), linked by an object URL kept while its
// ai: entry stands: a drawing (every change heard, every return to the page) reuses it and lets go only the URLs of
// ai: entries deleted or received again, so a tab opened from Open goes on reading its file (2026-10-01; before,
// ai: every drawing let them all go, a video in such a tab losing its link when the page came back into view).
let links = new Map();   // ai: `${id}@${at}`, an entry as filed: { file, url }
const when = (t) => (t ? new Date(t).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "");
function button(text, onclick) {
  const b = document.createElement("button");
  b.type = "button"; b.textContent = text;
  if (onclick) b.onclick = onclick;
  return b;
}
// ai: A row: the file's icon, its name and "size · date" (a link that opens it, as a tap on a row opens it in an app),
// ai: on a PC the size and the date in columns of their own and Open beside the more menu (index.html, 2026-11-06),
// ai: and the more menu (a details element, so it opens with no script; one open at a time, an action or a tap elsewhere
// ai: closes it; Delete asks on itself first, and stays open for the answer).
async function row(e, next) {
  const key = `${e.id}@${e.at ?? ""}`;
  let link = links.get(key);
  if (!link) { const file = await open(e).catch(() => null); link = file && { file, url: URL.createObjectURL(file) }; }
  if (link) next.set(key, link);
  const li = document.createElement("li"), f = link?.file;
  li.className = "item"; li.dataset.id = e.id;
  const lead = document.createElement("span"), name = document.createElement(f ? "a" : "span"), meta = document.createElement("span");
  lead.className = "lead"; lead.append(icon("file"));
  name.className = "name"; name.textContent = e.name; name.title = e.name;
  if (f) { name.href = link.url; name.target = "_blank"; name.rel = "noopener"; }
  meta.className = "meta"; meta.textContent = [bytes(e.size ?? f?.size ?? 0), when(e.at)].filter(Boolean).join(" · ");
  name.append(meta);
  const menu = document.createElement("details"), more = document.createElement("summary"), acts = document.createElement("div");
  menu.className = "menu";
  more.className = "btn icon"; more.title = `More for ${e.name}`; more.append(icon("more"));
  acts.className = "acts";
  const shut = () => { menu.open = false; };
  if (f) {
    const url = link.url, save = document.createElement("a");
    save.className = "btn"; save.href = url; save.download = e.name; save.textContent = "Save"; save.addEventListener("click", shut);
    acts.append(button("Open", () => { shut(); window.open(url, "_blank"); }), save);
    let shares = false;
    try { shares = !!navigator.canShare?.({ files: [f] }); } catch {}
    if (shares) acts.append(button("Share", () => { shut(); navigator.share({ files: [f], title: e.name }).catch(() => {}); }));
  }
  const del = button("Delete");
  del.classList.add("danger");
  const disarm = confirmTap(del, "Delete it?", () => remove(e.id).then(show));
  acts.append(del);
  menu.append(more, acts);
  menu.addEventListener("toggle", () => {
    if (!menu.open) return disarm();
    for (const m of document.querySelectorAll("#list details.menu[open]")) if (m !== menu) m.open = false;
  });
  const size = cell("c-size", bytes(e.size ?? f?.size ?? 0)), at = cell("c-when", when(e.at)), act = cell("c-act");
  if (f) { const o = button("Open", () => window.open(link.url, "_blank")); o.className = "c-open"; act.append(o); }
  act.append(menu);
  li.append(lead, name, size, at, act);
  return li;
}
// ai: a PC column's cell (index.html .c-size, .c-when, .c-act; hidden on a phone)
function cell(cls, text = "") { const c = document.createElement("span"); c.className = cls; c.textContent = text; return c; }
// ai: An unfinished transfer's row (2026-10-22, library.mjs unfinished): its name and "37% of 3.1 MB · date", the more
// ai: menu Continue (Receive, which resumes it once the file is in the light) and Delete (asked on itself first).
function partialRow(e) {
  const li = document.createElement("li"), lead = document.createElement("span"), name = document.createElement("span"), meta = document.createElement("span");
  li.className = "item"; li.dataset.id = e.id;
  lead.className = "lead"; lead.append(icon("file"));
  name.className = "name"; name.textContent = e.name || "a file"; name.title = e.name || "";
  meta.className = "meta"; meta.textContent = [`${Math.floor(100 * (e.fraction ?? 0))}% of ${bytes(e.length ?? 0)}`, when(e.at)].filter(Boolean).join(" · ");
  name.append(meta);
  const menu = document.createElement("details"), more = document.createElement("summary"), acts = document.createElement("div");
  menu.className = "menu";
  more.className = "btn icon"; more.title = `More for ${e.name}`; more.append(icon("more"));
  acts.className = "acts";
  const go = document.createElement("a");
  go.className = "btn"; go.href = "recv.html?auto"; go.textContent = "Continue";
  const del = button("Delete");
  del.classList.add("danger");
  const disarm = confirmTap(del, "Delete it?", () => forgetUnfinished(e.id).then(show));
  acts.append(go, del);
  menu.append(more, acts);
  menu.addEventListener("toggle", () => {
    if (!menu.open) return disarm();
    for (const m of document.querySelectorAll("details.menu[open]")) if (m !== menu) m.open = false;
  });
  // ai: on a PC: the progress as a small meter and its words, the date, and Continue beside the more menu
  const frac = Math.max(0, Math.min(1, e.fraction ?? 0)), prog = cell("c-size"), bar = document.createElement("span"), fill = document.createElement("span");
  bar.className = "meter mini"; fill.style.width = `${Math.floor(100 * frac)}%`; bar.append(fill);
  prog.append(bar, `${Math.floor(100 * frac)}% of ${bytes(e.length ?? 0)}`);
  const cont = document.createElement("a"), act = cell("c-act");
  cont.className = "btn c-open"; cont.href = "recv.html?auto"; cont.textContent = "Continue";
  act.append(cont, menu);
  li.append(lead, name, prog, cell("c-when", when(e.at)), act);
  return li;
}
document.addEventListener("click", (ev) => { for (const m of document.querySelectorAll("details.menu[open]")) if (!m.contains(ev.target)) m.open = false; });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") for (const m of document.querySelectorAll("details.menu[open]")) { m.open = false; m.querySelector("summary").focus(); } });
// ai: One drawing at a time: a change heard while one runs draws once more after it.
let drawing = null, again = false;
async function show() {
  if (drawing) { again = true; return drawing; }
  drawing = (async () => {
    const entries = await list(), kept = await keeps(), next = new Map(), parts = await unfinished();
    $("partialList").replaceChildren(...parts.map(partialRow));
    $("partial").hidden = !parts.length;
    $("list").replaceChildren(...(await Promise.all(entries.map((e) => row(e, next)))));
    for (const [key, { url }] of links) if (!next.has(key)) URL.revokeObjectURL(url);
    links = next;
    // ai: the section where there are files, or where this browser keeps none: then its warning alone
    $("files").hidden = !entries.length && kept && !PC.matches;
    $("files").classList.toggle("none", !entries.length);
    $("emptyPc").hidden = entries.length > 0 || !kept;
    $("empty").hidden = entries.length > 0 || kept;
    const total = entries.reduce((s, e) => s + (e.size || 0), 0);
    $("usage").textContent = entries.length ? `${entries.length} file${entries.length > 1 ? "s" : ""} kept, ${bytes(total)}` : "No files kept.";
    $("clearAll").hidden = $("clearAllTop").hidden = !entries.length;
  })();
  try { await drawing; } finally { drawing = null; }
  if (again) { again = false; await show(); }
}
// ai: Delete all, the phone's in Settings and the PC's on the files' heading, each asking on itself first
for (const id of ["clearAll", "clearAllTop"]) confirmTap($(id), () => { const n = $("list").children.length; return `Delete ${n} file${n > 1 ? "s" : ""}?`; }, () => removeAll().then(show));
watch(show);
show();
