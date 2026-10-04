// ai: Development logs to the rig server (lizard-web/server.mjs), the only traffic a page has with it (2026-09-26).
// ai: logPost(path, body): a POST, never awaited,
// ai: its answer unread; body a Blob, a string, or an object sent as JSON. A path whose post failed (no rig on a private
// ai: host: a static server there answers /api with 404) rests REST_MS before its next, so such a host sees one request
// ai: a path in 10 s, not one a second.
// ai: Only where a rig can answer (2026-10-03, the app on Cloudflare Pages): the rig is the page's own origin, reached on
// ai: the machine or the LAN, so a page served from a loopback or private address or a .local name posts, and one
// ai: served from a public host (a static deploy) posts nothing.
const REST_MS = 10000, restUntil = new Map(), HOST = location.hostname;
const RIG = /^(localhost|127(\.\d+){3}|10(\.\d+){3}|192\.168(\.\d+){2}|172\.(1[6-9]|2\d|3[01])(\.\d+){2}|\[::1\])$/.test(HOST) || HOST.endsWith(".local");

export function logPost(path, body) {
  if (!RIG || (restUntil.get(path) ?? 0) > performance.now()) return;
  const data = body instanceof Blob || typeof body === "string" ? body : JSON.stringify(body);
  fetch(path, { method: "POST", body: data })
    .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); })
    .catch(() => restUntil.set(path, performance.now() + REST_MS));
}
