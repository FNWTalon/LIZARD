// ai: The received files (2026-10-01): one folder a file in the
// ai: origin private file system, lizard-files/<id>/ holding the file as `data` and its meta.json ({ name, type, size,
// ai: root, at }), the id the first 16 hex digits of its BLAKE3 root, so a file received again replaces its entry. The
// ai: fountain worker files a finished transfer there (fountain-worker.mjs fileAway); the pages list, open and delete
// ai: here. A change is announced on a BroadcastChannel, so an open Home sees a file the receiver just filed.
// ai: Without OPFS (a private window, ?store=memory) nothing is kept: the receiver offers its file until the page closes.
export const DIR = "lizard-files";
const CHANNEL = "lizard-files";
const bc = typeof BroadcastChannel === "function" ? new BroadcastChannel(CHANNEL) : null;

async function folder(create = false) {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(DIR, { create });
}

// ai: Whether this browser keeps files at all (OPFS reachable from a page).
export async function keeps() {
  try { await navigator.storage.getDirectory(); return true; } catch { return false; }
}

// ai: Every kept file, newest first: { id, name, type, size, root, at }. A folder whose meta.json does not read (a
// ai: filing cut short) is left out.
export async function list() {
  let d;
  try { d = await folder(); } catch { return []; }
  const out = [];
  for await (const [id, h] of d.entries()) {
    if (h.kind !== "directory") continue;
    try {
      const meta = JSON.parse(await (await (await h.getFileHandle("meta.json")).getFile()).text());
      out.push({ ...meta, id });
    } catch {}
  }
  return out.sort((a, b) => (b.at ?? 0) - (a.at ?? 0));
}

// ai: The file of an entry, under its own name and type (the bytes stay on disk: a File over the OPFS one).
export async function open(entry) {
  const d = await (await folder()).getDirectoryHandle(entry.id);
  const f = await (await d.getFileHandle("data")).getFile();
  return new File([f], entry.name || "received.bin", { type: entry.type || f.type || "" });
}

export async function remove(id) {
  await (await folder()).removeEntry(id, { recursive: true });
  changed();
}
export async function removeAll() {
  await (await navigator.storage.getDirectory()).removeEntry(DIR, { recursive: true }).catch(() => {});
  changed();
}

export function changed() { bc?.postMessage("changed"); }
// ai: cb on a change from another page (or the fountain worker), and whenever this page comes back into view.
export function watch(cb) {
  bc?.addEventListener("message", () => cb());
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") cb(); });
}
