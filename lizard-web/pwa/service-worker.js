"use strict";

// ai: The installed app's worker (2026-10-01): another project's service worker, ported, its design unchanged.
// ai: lizard-web/pwa/build.mjs copies
// ai: this template into the built tree (lizard-web/app/) and fills the four lines marked "build.mjs"; the rig's source
// ai: pages register no worker. Cache-first for every GET of this origin, navigations included, so the app opens and
// ai: receives with no network; an install is all or nothing, each file checked against SHELL_HASHES; the content-
// ai: addressed files live in one long-lived cache across builds, so a build that changed one module fetches that module.

const CACHE_PREFIX = "lizard-";
const CACHE_NAME = `${CACHE_PREFIX}v0`; // build.mjs: the build's time
const SW_SCOPE = self.registration.scope;

const SHELL_PAGE_URLS = []; // build.mjs: the pages
const SHELL_ASSET_URLS = []; // build.mjs: the manifest and every file under assets/

// ai: the arrays stay relative so they can key SHELL_HASHES; resolving happens here
const SHELL_PATHS = [...SHELL_PAGE_URLS, ...SHELL_ASSET_URLS];
function shellUrl(path) {
  return new URL(path, SW_SCOPE).toString();
}

const SHELL_HASHES = {}; // build.mjs: SHA-256 of every shell path's bytes

// ai: Two caches, split on whether a URL can ever mean different bytes. The pages keep their names, so theirs is
// ai: versioned per build and replaced whole; the assets are content-addressed (lizard-web/pwa/hash.mjs), so a URL there names
// ai: one byte sequence for ever and several builds' assets share one cache without colliding. That split is what lets
// ai: an open page keep working while a newer build takes over: its own assets are still there under their own URLs.
const ASSET_CACHE = `${CACHE_PREFIX}assets`;
// ai: whether this URL is immutable, a fact of the naming scheme (hash.mjs --verify pins it at build time)
const ADDRESSED_RE = /\.[0-9a-f]{10}\.[^.]+$/;
const ADDRESSED = SHELL_ASSET_URLS.some((path) => ADDRESSED_RE.test(path));
function cacheFor(path) {
  return ADDRESSED_RE.test(path) ? ASSET_CACHE : CACHE_NAME;
}

// ai: Served from cache, navigations included, so a page load never waits on the network: this build's pages, then the
// ai: asset cache (which may also hold an older build's files, the point: a page claimed by a newer worker still finds
// ai: the assets its own html names). A miss is a URL outside the shell (the rig's /api, a page the app does not carry)
// ai: and falls through to the host. recv.html?auto finds recv.html through ignoreSearch, and is answered under its own
// ai: URL, query and all (2026-10-01, the one change to the ported worker): a stored response carries the URL it was
// ai: stored under, and a worker's location (a module's import.meta.url too) is its response's URL, so the GPU worker
// ai: asked for as recv-gpu-worker.mjs?prec=f16 read no switches from location.search. A response made from the stored
// ai: one has no URL of its own, and the browser gives it the request's.
// ai: A path with no extension is also looked up as its .html page (2026-10-03, the app on Cloudflare Pages, which
// ai: serves send.html at /send and sends a visit there before this worker controls the page): offline, /send is the
// ai: page send.html is.
async function cacheFirst(request) {
  const url = new URL(request.url);
  const page = /\/[^/.]+$/.test(url.pathname) ? new Request(`${url.origin}${url.pathname}.html${url.search}`) : null;
  for (const asked of page ? [request, page] : [request]) {
    for (const name of [CACHE_NAME, ASSET_CACHE]) {
      const cache = await caches.open(name);
      const cached = await cache.match(asked);
      if (cached) return asked === request ? cached : new Response(cached.body, cached);
      const bare = await cache.match(asked, { ignoreSearch: true });
      if (bare) return new Response(bare.body, bare);
    }
  }
  try {
    return await fetch(request);
  } catch {
    return Response.error();
  }
}

async function sha256Hex(response) {
  const digest = await crypto.subtle.digest("SHA-256", await response.arrayBuffer());
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

// ai: a host error is a resolved fetch, so !ok is refused by hand, or an error page would be cached as the shell. A
// ai: response a redirect brought (Cloudflare Pages sends send.html to /send and index.html to /, 2026-10-03) is stored
// ai: as its bytes and headers alone: a redirected response cannot answer a page's own load, whose redirect mode is
// ai: manual, and the browser refuses the page.
async function fetchVerified(path) {
  let response = await fetch(new Request(shellUrl(path), { cache: "reload" }));
  if (!response.ok) throw new Error(`${response.status} for ${path}`);
  if (response.redirected) response = new Response(await response.arrayBuffer(), { status: response.status, statusText: response.statusText, headers: response.headers });
  const expected = SHELL_HASHES[path];
  if (expected && (await sha256Hex(response.clone())) !== expected) throw new Error(`hash mismatch for ${path}`);
  return response;
}

// ai: An addressed path already in the asset cache needs no network and no hashing: its URL names its bytes, and nothing
// ai: reaches that cache without passing fetchVerified. That is the differential install. Pages are never reused.
async function reusable(path) {
  if (cacheFor(path) !== ASSET_CACHE) return null;
  const cache = await caches.open(ASSET_CACHE);
  return (await cache.match(shellUrl(path))) || null;
}

// ai: All or nothing: every file fetched and verified before any is written, so a build is complete or never activated.
// ai: A throw leaves the previous complete build serving, and the browser tries again at a later update check.
async function precacheShell() {
  const responses = await Promise.all(SHELL_PATHS.map(async (path) => (await reusable(path)) || fetchVerified(path)));
  const opened = new Map();
  for (const name of [CACHE_NAME, ASSET_CACHE]) opened.set(name, await caches.open(name));
  await Promise.all(SHELL_PATHS.map((path, i) => opened.get(cacheFor(path)).put(shellUrl(path), responses[i])));
}

// ai: Taking over at once is safe because an open page's assets stay under their own URLs; gated on content addressing,
// ai: since an unhashed tree taking over at once is the mixed-build bug.
self.addEventListener("install", (event) => {
  event.waitUntil(precacheShell().then(() => (ADDRESSED ? self.skipWaiting() : undefined)));
});

// ai: Two speeds: a superseded page cache goes at once; the asset cache is pruned to this build's files only when no
// ai: window is open, since an open page may still ask it for the files its own html names.
async function retireOldBuilds() {
  const keys = await caches.keys();
  await Promise.all(keys.filter((key) => key.startsWith(CACHE_PREFIX) && key !== CACHE_NAME && key !== ASSET_CACHE).map((key) => caches.delete(key)));
  const windows = await self.clients.matchAll({ type: "window" });
  if (windows.length > 0) return;
  const wanted = new Set(SHELL_PATHS.filter((path) => cacheFor(path) === ASSET_CACHE).map(shellUrl));
  const cache = await caches.open(ASSET_CACHE);
  const stale = (await cache.keys()).filter((request) => !wanted.has(request.url));
  await Promise.all(stale.map((request) => cache.delete(request)));
}

self.addEventListener("activate", (event) => {
  event.waitUntil(retireOldBuilds().then(() => self.clients.claim()));
});

// ai: GETs of this origin only: the rig's development logs are POSTs and pass untouched.
self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  if (new URL(event.request.url).origin !== self.location.origin) return;
  event.respondWith(cacheFirst(event.request));
});
