// ai: The helpers every Lizard page shares, for the parts in ui.css. Page-only: no worker imports this.

// ai: remember(details, key): keeps a <details>' open state across reloads (a phone session is a series of reloads).
// ai: Storage can throw (a private window, blocked site data); the page then opens as its markup says.
// ai: collapser(button, key): the tool pages' sidebar collapser (2026-10-02): body.collapsed toggled and kept under key (each
// ai: page's inline script restores it before the first paint); landscape only, ui.css draws the rail.
export function collapser(button, key) {
  const show = () => {
    const on = document.body.classList.contains("collapsed");
    button.title = on ? "Show sidebar" : "Hide sidebar"; button.setAttribute("aria-expanded", String(!on));
  };
  try { document.body.classList.toggle("collapsed", localStorage.getItem(key) === "1"); } catch {}
  show();
  button.addEventListener("click", () => {
    const on = document.body.classList.toggle("collapsed");
    try { localStorage.setItem(key, on ? "1" : "0"); } catch {}
    show();
  });
}

// ai: sideResizer(key, floor): the landscape column's width set by dragging its edge (2026-10-05, as the Android app's): a
// ai: strip on the column's right edge (.side-edge, ui.css), dragged sideways, sets --side, within 180 px (or floor(),
// ai: px, where a page gives one: the least its content needs with no word broken, read at each drag's start and at the
// ai: restore; recv.mjs actionsFloor) and three fifths of the window; kept under `key` on release (each page's inline
// ai: script restores it before the first paint). The sender re-picks as it does for a resized window.
export const SIDE_MIN = 180, sideClamp = (px, min = SIDE_MIN) => Math.max(min, Math.min(px, innerWidth * 0.6));
export function sideResizer(key, floor) {
  const root = document.documentElement, edge = document.createElement("div"), lo = () => Math.max(SIDE_MIN, floor?.() || 0);
  edge.className = "side-edge";
  edge.setAttribute("role", "separator"); edge.setAttribute("aria-orientation", "vertical"); edge.setAttribute("aria-label", "Sidebar width");
  document.body.append(edge);
  try { const v = +localStorage.getItem(key); if (v > 0) root.style.setProperty("--side", `${sideClamp(v, lo())}px`); } catch {}
  edge.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    edge.setPointerCapture(e.pointerId);
    document.body.classList.add("resizing");
    const min = lo(), move = (m) => root.style.setProperty("--side", `${sideClamp(m.clientX, min)}px`);
    const up = () => {
      for (const [t, f] of [["pointermove", move], ["pointerup", up], ["pointercancel", up]]) edge.removeEventListener(t, f);
      document.body.classList.remove("resizing");
      try { localStorage.setItem(key, String(parseFloat(root.style.getPropertyValue("--side")) || "")); } catch {}
    };
    for (const [t, f] of [["pointermove", move], ["pointerup", up], ["pointercancel", up]]) edge.addEventListener(t, f);
  });
}

export function remember(details, key) {
  try { const v = localStorage.getItem(key); if (v !== null) details.open = v === "1"; } catch {}
  details.addEventListener("toggle", () => { try { localStorage.setItem(key, details.open ? "1" : "0"); } catch {} });
}

// ai: persist(el, key): keeps a setting's value across reloads in localStorage (2026-09-29), a
// ai: <select> or an <input>: restored now where the saved value is one it can take (one of a select's options), saved
// ai: whenever the user changes it (a page's own assignments fire no change and save nothing). A page presets from its
// ai: URL after this, so a URL still wins for that load. restoreSaved(el, key): the restore alone, for a select whose
// ai: options come later (the receiver's camera list). Storage that throws (a private window, blocked site data) keeps nothing.
export function restoreSaved(el, key) {
  try {
    const v = localStorage.getItem(key);
    if (v === null) return;
    if (el.tagName === "SELECT" ? [...el.options].some((o) => o.value === v) : el.type !== "file") el.value = v;
  } catch {}
}
export function persist(el, key) {
  restoreSaved(el, key);
  el.addEventListener("change", () => { try { localStorage.setItem(key, el.value); } catch {} });
}

// ai: say(el, text, tone): sets one status line; tone "bad" or "good" colours it, anything else leaves it black.
export function say(el, text, tone = null) {
  el.textContent = text;
  el.classList.toggle("bad", tone === "bad");
  el.classList.toggle("good", tone === "good");
}

// ai: meter(el, frac): a .meter element at frac (0 to 1) of full; null hides it.
export function meter(el, frac) {
  el.hidden = frac == null;
  if (frac == null) return;
  const f = Math.max(0, Math.min(1, frac));
  (el.firstElementChild ?? el.appendChild(document.createElement("span"))).style.width = `${f * 100}%`;
  el.setAttribute("aria-valuenow", String(Math.round(f * 100)));
}

// ai: bytes(n): a size for people, 1 KB = 1000 B as the rates count it: "812 B", "7.4 MB", "31 MB".
export function bytes(n) {
  if (n < 1000) return `${n} B`;
  // ai: the unit chosen after rounding, so 999,500 B reads "1.0 MB", not "1000 KB"
  const at = (v) => (v < 9.95 ? v.toFixed(1) : String(Math.round(v)));
  const [v, u] = n < 999.5e3 ? [n / 1e3, "KB"] : n < 999.5e6 ? [n / 1e6, "MB"] : [n / 1e9, "GB"];
  return `${at(v)} ${u}`;
}

// ai: The transfer's figures for people (2026-10-01: progress, size, speed, time left), the Android app's
// ai: Readout.kt the same: rate(kbs) a speed from KB/s (1 KB = 1000 B), "640 KB/s" under 1,000 and "1.57 MB/s" from
// ai: there; left(secs) the time still to go, "4 s left", "2 min left" from 100 s. A figure keeps its unit on its line
// ai: (a no-break space).
export function rate(kbs) {
  return kbs < 999.5 ? `${Math.round(kbs)} KB/s` : `${(kbs / 1000).toFixed(2)} MB/s`;
}
export function left(secs) {
  const s = Math.max(1, Math.ceil(secs));
  return s < 100 ? `${s} s left` : `${Math.round(s / 60)} min left`;
}

// ai: First-run tips (2026-10-01): one flag for every page of this origin, kept once a tip's "Got it" is pressed (or the
// ai: sender has painted a file), cleared by Settings' "Show tips again". tips(root) shows root's .tip cards while the
// ai: flag is unset and wires their [data-got] buttons; tipsSeen() is the flag, for a page's own inline tip.
// ai: The icons a script puts on a page (2026-10-01; Feather's, MIT): ui.css svg.i draws them on a 24 grid, 2 px strokes
// ai: in the text's colour. The pages inline the same set in their static markup (back, gear, swap, camera, monitor,
// ai: camera-off, info), and the Android app has them as vector drawables (res/drawable/ic_*.xml).
const ICONS = {
  file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
  more: '<circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/>',
};
export function icon(name) {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("class", "i");
  s.setAttribute("aria-hidden", "true");
  s.innerHTML = ICONS[name];
  return s;
}

const TIPS = "lizard:tips";
export function tipsSeen() { try { return localStorage.getItem(TIPS) === "seen"; } catch { return false; } }
export function tipsDone() {
  try { localStorage.setItem(TIPS, "seen"); } catch {}
  for (const t of document.querySelectorAll(".tip")) t.hidden = true;
}
export function tips(root = document) {
  for (const t of root.querySelectorAll(".tip")) {
    t.hidden = tipsSeen();
    t.querySelector("[data-got]")?.addEventListener("click", tipsDone);
  }
}
export function tipsAgain() {
  try { localStorage.removeItem(TIPS); } catch {}
  for (const t of document.querySelectorAll(".tip")) t.hidden = false;
}


// ai: confirmTap(button, ask, act): a destructive button asks on itself ("Delete 3 files?") for 4 s, and a second tap
// ai: within them acts; ask may be a function, for a count that changes.
export function confirmTap(btn, ask, act) {
  let armed = 0;
  const label = btn.textContent;
  const disarm = () => { if (armed) { clearTimeout(armed); armed = 0; btn.textContent = label; } };
  btn.addEventListener("click", () => {
    if (armed) { disarm(); act(); return; }
    btn.textContent = typeof ask === "function" ? ask() : ask;
    armed = setTimeout(disarm, 4000);
  });
  return disarm;   // ai: for a menu that closes under an armed button (home.mjs), so reopening it asks again
}

// ai: The built app (lizard-web/pwa/build.mjs puts <meta name="lizard-build"> in each page it ships): "<when> <cache name>", or
// ai: null on the rig's source pages, which are not the app and register no worker.
export function buildInfo() { return document.querySelector('meta[name="lizard-build"]')?.content ?? null; }
// ai: about(el): the About line, the build this page is.
export function about(el, failed = false) {
  const b = buildInfo();
  // ai: the build's time alone, and nothing on the rig's pages (2026-10-02)
  el.textContent = b ? `Build ${b.split(" ")[0].replace("T", " ").replace(/:\d\d(\.\d+)?Z$/, " UTC")}${failed ? ". Update failed" : ""}` : "";
}

// ai: registerApp(onFail): the installable app's worker (lizard-web/pwa/service-worker.js, a ported scheme), only on a built
// ai: page. Like that scheme's page script it only induces and observes: an update check whenever the page comes back into view
// ai: (an installed app that never navigates hears of a new build no other way), and onFail when an update's install
// ai: failed (a worker that went from installing straight to redundant; one superseded after it activated did not fail).
// ai: Nothing here reloads a page: a new build is served from the next navigation.
export function registerApp(onFail) {
  if (!buildInfo() || !navigator.serviceWorker || location.protocol === "file:") return;
  navigator.serviceWorker.register("service-worker.js").then((reg) => {
    const watch = (w) => {
      if (!w) return;
      let passed = w.state !== "installing";
      w.addEventListener("statechange", () => {
        if (w.state !== "installing" && w.state !== "redundant") passed = true;
        if (w.state === "redundant" && !passed) { console.warn("lizard: an update of the app failed to install"); onFail?.(); }
      });
    };
    watch(reg.installing);
    reg.addEventListener("updatefound", () => watch(reg.installing));
    document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") reg.update().catch(() => {}); });
  }).catch((e) => console.warn("lizard: the app's worker did not register:", e));
}
