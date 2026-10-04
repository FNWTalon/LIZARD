// ai: The installed app's worker in a stubbed ServiceWorkerGlobalScope: another project's worker test, ported. It loads
// ai: a BUILT worker (app/service-worker.js, the template's
// ai: lists being empty until lizard-web/pwa/build.mjs fills them) and serves the built tree's bytes off disk, so the embedded
// ai: SHA-256 manifest is checked through the worker's own hashing:
// ai:   cache-first  every shell request answered from cache, never the network, a page's query (recv.html?auto) too
// ai:   atomic       any failure during install writes nothing and activates nothing
// ai:   versioned    activate drops superseded page caches and keeps the asset cache
// ai:   retained     a superseded asset outlives its build while any window is still open
// ai: Usage: node lizard-web/pwa/test-sw.mjs [service-worker.js; lizard-web/app's by default]. About a second.
import { readFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash, webcrypto } from "node:crypto";
import vm from "node:vm";
import assert from "node:assert";

const ORIGIN = "https://example.test";
const swPath = resolve(process.argv[2] ?? join(dirname(fileURLToPath(import.meta.url)), "../app/service-worker.js"));
if (!existsSync(swPath)) { console.error(`no built worker at ${swPath}: node lizard-web/pwa/build.mjs first`); process.exit(2); }
const SOURCE = readFileSync(swPath, "utf8"), SW_DIR = dirname(swPath);
const PAGE = `${ORIGIN}/recv.html`;
// ai: the stylesheet read out of the shell list: its name is a hash of its bytes
const CSS_PATH = (SOURCE.match(/"(\.\/assets\/lizard-web\/ui\.[0-9a-f]{10}\.css)"/) || [])[1];
if (!CSS_PATH) throw new Error("no stylesheet in SHELL_ASSET_URLS");
const CSS = ORIGIN + CSS_PATH.slice(1), CSS_REL = CSS_PATH.slice(2);

let passed = 0;
async function test(name, fn) { await fn(); passed++; console.log(`  ok  ${name}`); }

// ai: A stub scope around src; serve(url) decides what the network returns: bytes, { status } for an HTTP error, or null
// ai: for a fetch that throws.
function makeScope(serve, src = SOURCE) {
  const listeners = {}, caches = new Map(), log = [], windowClients = [], calls = { skipWaiting: 0 };
  function openCache(name) {
    if (!caches.has(name)) caches.set(name, new Map());
    const store = caches.get(name);
    return {
      async match(request, options) {
        let url = typeof request === "string" ? request : request.url;
        if (options?.ignoreSearch) url = url.split("?")[0];
        return store.has(url) ? store.get(url).clone() : undefined;
      },
      async put(request, response) { store.set(typeof request === "string" ? request : request.url, response); },
      async keys() { return [...store.keys()].map((url) => new Request(url)); },
      async delete(request) { return store.delete(typeof request === "string" ? request : request.url); },
    };
  }
  const scope = {
    registration: { scope: `${ORIGIN}/` },
    location: { origin: ORIGIN },
    addEventListener(name, fn) { listeners[name] = fn; },
    skipWaiting: async () => { calls.skipWaiting++; },
    clients: { claim: async () => {}, matchAll: async () => windowClients.slice() },
    caches: {
      open: async (name) => openCache(name),
      keys: async () => [...caches.keys()],
      delete: async (name) => caches.delete(name),
    },
    fetch: async (request) => {
      const url = typeof request === "string" ? request : request.url;
      log.push(url);
      const body = serve(url);
      if (body === null || body === undefined) throw new TypeError("offline");
      if (body?.status) return new Response("error page", { status: body.status });
      return new Response(body, { status: 200 });
    },
    crypto: webcrypto,
    console: { warn() {}, log() {} },
    URL, Request, Response, TextEncoder,
  };
  scope.self = scope;
  vm.createContext(scope);
  vm.runInContext(src, scope);
  const api = {
    listeners, caches, log, windowClients, calls,
    cacheName: vm.runInContext("CACHE_NAME", scope),
    assetCache: vm.runInContext("ASSET_CACHE", scope),
    addressed: vm.runInContext("ADDRESSED", scope),
    shellPaths: vm.runInContext("SHELL_PATHS", scope),
    store() { return caches.get(api.cacheName); },
    assets() { return caches.get(api.assetCache); },
    cached(url) { for (const name of [api.cacheName, api.assetCache]) { const s = caches.get(name); if (s?.has(url)) return s.get(url); } return undefined; },
    stored() { return [api.store(), api.assets()].reduce((n, s) => n + (s ? s.size : 0), 0); },
    install() { let w; listeners.install({ waitUntil: (p) => { w = p; } }); return w; },
    activate() { let w; listeners.activate({ waitUntil: (p) => { w = p; } }); return w; },
    request(url, mode = "no-cors", method = "GET") {
      const request = new Request(url, { method });
      Object.defineProperty(request, "mode", { value: mode });
      let responded;
      listeners.fetch({ request, respondWith: (p) => { responded = p; }, waitUntil() {} });
      return responded;
    },
  };
  return api;
}

// ai: a shell path's bytes off the built tree ("./" is index.html); a listed path with no file is the host's 404
function bodyFor(rel) {
  try { return readFileSync(join(SW_DIR, rel === "" ? "index.html" : rel)); } catch { return { status: 404 }; }
}
function shellServer(getScope, overrides = {}) {
  return (url) => {
    if (url in overrides) return overrides[url];
    const rel = url.slice(ORIGIN.length + 1);
    return getScope().shellPaths.some((p) => p.replace(/^\.\//, "") === rel) ? bodyFor(rel) : { status: 404 };
  };
}
function installedScope(overrides = {}, src = SOURCE) {
  let scope;
  scope = makeScope((url) => shellServer(() => scope, overrides)(url), src);
  return scope;
}
const shellUrlOf = (relPath) => `${ORIGIN}/${relPath.replace(/^\.\//, "")}`;
function withHashes(hashes) {
  const out = SOURCE.replace(/^const SHELL_HASHES = \{[\s\S]*?\};$/m, `const SHELL_HASHES = ${JSON.stringify(hashes, null, 2)};`);
  assert.notStrictEqual(out, SOURCE, "SHELL_HASHES declaration not found");
  return out;
}
// ai: the un-hashed branch against a built worker: its asset names without their hashes, and no manifest
function withoutAddressedAssets() {
  return SOURCE.replace(/^const SHELL_HASHES = \{[\s\S]*?\};$/m, "const SHELL_HASHES = {};").replace(/^(  "\.\/assets\/[^"]+)\.[0-9a-f]{10}(\.[a-z0-9]+",)$/gm, "$1$2");
}
const text = async (r) => Buffer.from(await r.arrayBuffer()).toString("latin1");
const disk = (rel) => Buffer.from(bodyFor(rel)).toString("latin1");

console.log(`service worker self-test (${swPath})`);
await test("install precaches every shell path", async () => {
  const scope = installedScope();
  await scope.install();
  assert.strictEqual(scope.stored(), scope.shellPaths.length);
  assert.ok(scope.store().has(`${ORIGIN}/`), "the directory alias cached under its own URL");
});
await test("navigations are served from cache without touching the network", async () => {
  const scope = installedScope();
  await scope.install();
  scope.log.length = 0;
  assert.strictEqual(await text(await scope.request(PAGE, "navigate")), disk("recv.html"));
  assert.deepStrictEqual(scope.log, []);
});
await test("a page's query (recv.html?auto) is served the page, offline", async () => {
  const scope = installedScope();
  await scope.install();
  scope.log.length = 0;
  assert.strictEqual(await text(await scope.request(`${PAGE}?auto`, "navigate")), disk("recv.html"));
  assert.deepStrictEqual(scope.log, []);
});
await test("subresources are served from cache without touching the network", async () => {
  const scope = installedScope();
  await scope.install();
  scope.log.length = 0;
  assert.strictEqual(await text(await scope.request(CSS)), disk(CSS_REL));
  assert.deepStrictEqual(scope.log, []);
});
await test("a miss falls through to the host", async () => {
  const scope = installedScope();
  await scope.install();
  assert.strictEqual((await scope.request(`${ORIGIN}/api/stats`)).status, 404, "the host's own answer reaches the page");
});
await test("a miss while offline returns an error rather than throwing", async () => {
  let scope;
  scope = makeScope((url) => (url === `${ORIGIN}/nope/` ? null : shellServer(() => scope)(url)));
  await scope.install();
  assert.strictEqual((await scope.request(`${ORIGIN}/nope/`, "navigate")).type, "error");
});
await test("a host 500 on one file aborts the whole install", async () => {
  const scope = installedScope({ [CSS]: { status: 500 } });
  await assert.rejects(scope.install(), /500 for/);
  assert.strictEqual(scope.stored(), 0, "nothing written when one file fails");
});
await test("a dropped connection during install writes nothing", async () => {
  let scope;
  scope = makeScope((url) => (url === CSS ? null : shellServer(() => scope)(url)));
  await assert.rejects(scope.install());
  assert.strictEqual(scope.stored(), 0);
});
await test("a failed install leaves the previous build serving", async () => {
  const first = installedScope();
  await first.install();
  const previous = first.store();
  const broken = installedScope({ [CSS]: { status: 500 } });
  broken.caches.set("lizard-vprevious", previous);
  await assert.rejects(broken.install());
  assert.ok(previous.has(PAGE), "the previous cache is untouched");
});
await test("a tampered file aborts the install", async () => {
  const scope = installedScope({}, withHashes({ [CSS_PATH]: "0".repeat(64) }));
  await assert.rejects(scope.install(), /hash mismatch/);
  assert.strictEqual(scope.stored(), 0, "nothing written on a bad hash");
});
await test("the embedded manifest matches the bytes on disk", async () => {
  const digest = createHash("sha256").update(bodyFor(CSS_REL)).digest("hex");
  assert.ok(SOURCE.includes(`"${CSS_PATH}": "${digest}"`), "the built worker names the stylesheet's own hash");
});
await test("activate keeps the current build and the asset cache", async () => {
  const scope = installedScope();
  await scope.install();
  scope.caches.set("lizard-vold", new Map());
  scope.caches.set("unrelated-cache", new Map());
  await scope.activate();
  assert.ok(scope.caches.has(scope.cacheName));
  assert.ok(scope.caches.has(scope.assetCache));
  assert.ok(!scope.caches.has("lizard-vold"), "old page cache dropped");
  assert.ok(scope.caches.has("unrelated-cache"), "caches from elsewhere left alone");
});
const addressedAsset = (scope) => scope.shellPaths.find((p) => /\.[0-9a-f]{10}\.[^.]+$/.test(p));
await test("an addressed build takes over as soon as it is installed", async () => {
  const scope = installedScope();
  assert.ok(scope.addressed, "the built tree is addressed");
  await scope.install();
  assert.strictEqual(scope.calls.skipWaiting, 1);
});
await test("an unaddressed build waits instead", async () => {
  const scope = installedScope({}, withoutAddressedAssets());
  assert.ok(!scope.addressed);
  await scope.install().catch(() => {});
  assert.strictEqual(scope.calls.skipWaiting, 0, "taking over unhashed is the mixed-build bug");
});
await test("a superseded asset outlives the build that named it", async () => {
  const scope = installedScope();
  await scope.install();
  const gone = `${ORIGIN}/assets/lizard-web/ui.deadbeef00.mjs`;
  scope.assets().set(gone, new Response("an older build's file"));
  scope.windowClients.push({ id: "a" });
  await scope.activate();
  assert.ok(scope.assets().has(gone), "an open page can still be served its own build");
  scope.log.length = 0;
  assert.strictEqual(await (await scope.request(gone)).text(), "an older build's file");
  assert.deepStrictEqual(scope.log, []);
});
await test("it is collected once no window is left to strand", async () => {
  const scope = installedScope();
  await scope.install();
  const gone = `${ORIGIN}/assets/lizard-web/ui.deadbeef00.mjs`, kept = shellUrlOf(addressedAsset(scope));
  scope.assets().set(gone, new Response("an older build's file"));
  await scope.activate();
  assert.ok(!scope.assets().has(gone), "superseded entries dropped");
  assert.ok(scope.assets().has(kept), "this build's are not");
});
await test("an asset already held is not refetched, a page always is", async () => {
  const first = installedScope();
  await first.install();
  const second = installedScope();
  second.caches.set(second.assetCache, first.assets());
  await second.install();
  assert.ok(!second.log.includes(shellUrlOf(addressedAsset(second))), "an immutable URL names bytes already verified");
  assert.ok(second.log.includes(PAGE), "a page's URL is mutable and always refetched");
});
await test("non-GET and cross-origin requests are not intercepted", async () => {
  const scope = installedScope();
  await scope.install();
  assert.strictEqual(await scope.request("https://other.test/x.css"), undefined);
  assert.strictEqual(await scope.request(`${ORIGIN}/api/stats`, "no-cors", "POST"), undefined, "the dev log's POSTs pass");
});
console.log(`service worker self-test passed (${passed} checks).`);
