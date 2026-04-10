import { createHash } from "crypto";
import { promises as fsp, mkdirSync } from "fs";
import { join } from "path";

export interface CacheStats {
  hits: number;
  misses: number;
  size: number;
  ttlMinutes: number;
  enabled: boolean;
  maxEntries: number;
  inflight: number;
  persistent: boolean;
}

interface CacheEntry {
  data: unknown;
  expiresAt: number;
  model: string;
  chunks?: string[]; // SSE chunks for streaming responses
}

const DEFAULT_TTL_MS = 60 * 60 * 1000; // 1 hour
const DEFAULT_MAX_ENTRIES = 500;
const GC_INTERVAL_MS = 10 * 60 * 1000; // 10 minutes

let ttlMs = DEFAULT_TTL_MS;
let maxEntries = DEFAULT_MAX_ENTRIES;
let enabled = true;

const cache = new Map<string, CacheEntry>();
let hits = 0;
let misses = 0;

// ---------------------------------------------------------------------------
// Disk persistence
//
// The in-memory cache is lost on every server restart (Replit dev mode
// rebuilds on every code save, and the container can restart at any time).
// To survive restarts, the cache is written through to a JSON file.
//
// Strategy:
//   • On module load: read the file and populate the in-memory Map.
//   • On every SET: schedule a debounced async write (100 ms grace period so
//     burst writes are coalesced into a single I/O operation).
//   • On CLEAR: delete the file synchronously, cancel any pending write.
//   • Write is atomic: write to a .tmp file then rename, so a crash mid-write
//     never corrupts the last good snapshot.
// ---------------------------------------------------------------------------

const CACHE_DIR  = join(process.cwd(), ".cache");
const CACHE_FILE = join(CACHE_DIR, "responses.json");
const CACHE_TMP  = join(CACHE_DIR, "responses.json.tmp");

// Ensure the .cache directory exists at startup (sync so it's ready before
// the first cacheSet; only runs once on process start).
try { mkdirSync(CACHE_DIR, { recursive: true }); } catch { /* already exists */ }

let diskWriteTimer: ReturnType<typeof setTimeout> | null = null;
let diskWritePending = false; // true while writeCacheToDisk is in progress

function scheduleDiskWrite(): void {
  if (diskWriteTimer) clearTimeout(diskWriteTimer);
  diskWriteTimer = setTimeout(() => {
    diskWriteTimer = null;
    if (diskWritePending) {
      // A write is already in progress; re-schedule so this batch isn't lost.
      scheduleDiskWrite();
      return;
    }
    diskWritePending = true;
    writeCacheToDisk()
      .catch(() => { /* I/O errors are non-fatal */ })
      .finally(() => { diskWritePending = false; });
  }, 100);
}

async function writeCacheToDisk(): Promise<void> {
  const now = Date.now();
  const snapshot: Record<string, CacheEntry> = {};
  for (const [k, v] of cache.entries()) {
    if (now <= v.expiresAt) snapshot[k] = v; // only persist live entries
  }
  const json = JSON.stringify(snapshot);
  await fsp.writeFile(CACHE_TMP, json, "utf8");
  await fsp.rename(CACHE_TMP, CACHE_FILE);        // atomic on POSIX
}

async function loadCacheFromDisk(): Promise<void> {
  let raw: string;
  try {
    raw = await fsp.readFile(CACHE_FILE, "utf8");
  } catch {
    return; // file doesn't exist yet — that's fine on first run
  }
  let snapshot: Record<string, CacheEntry>;
  try {
    snapshot = JSON.parse(raw) as Record<string, CacheEntry>;
  } catch {
    return; // corrupt file — start fresh
  }
  const now = Date.now();
  let loaded = 0;
  for (const [k, v] of Object.entries(snapshot)) {
    if (now <= v.expiresAt && cache.size < maxEntries) {
      cache.set(k, v);
      loaded++;
    }
  }
  if (loaded > 0) {
    process.stdout.write(`[cache] Loaded ${loaded} entries from disk\n`);
  }
}

// Export a promise that resolves once the initial disk load is complete.
// index.ts awaits this before calling app.listen() so no request is handled
// before the cache has been restored from disk.
export const cacheReady: Promise<void> = loadCacheFromDisk().catch(() => { /* I/O errors are non-fatal */ });

// ---------------------------------------------------------------------------
// Background GC — .unref() so this timer never prevents clean process exit.
// ---------------------------------------------------------------------------

setInterval(() => {
  const now = Date.now();
  let changed = false;
  for (const [k, v] of cache.entries()) {
    if (now > v.expiresAt) { cache.delete(k); changed = true; }
  }
  if (changed) scheduleDiskWrite(); // persist the GC result
}, GC_INTERVAL_MS).unref();

function evictExpired(): void {
  const now = Date.now();
  for (const [k, v] of cache.entries()) {
    if (now > v.expiresAt) cache.delete(k);
  }
}

// ---------------------------------------------------------------------------
// Request hashing
//
// Fields that affect ONLY routing, billing, or observability — not the
// response content.  These are excluded from the cache key so that adding
// or removing them does not cause spurious cache misses.
// ---------------------------------------------------------------------------
const HASH_EXCLUDE_FIELDS = new Set([
  "stream",             // delivery mode, not content
  "cache_control",      // Anthropic/Gemini prompt-caching breakpoints (billing only)
  "provider",           // OpenRouter routing preference
  "route",              // OpenRouter routing
  "session_id",         // OpenRouter observability
  "trace",              // OpenRouter observability
  "metadata",           // OpenRouter metadata
  "service_tier",       // billing tier
  "speed",              // performance tier
  "user",               // end-user identifier
  "x_use_prompt_tools", // internal proxy flag
  "stream_options",     // SSE delivery option
]);

/**
 * JSON replacer that:
 *  - drops fields that only affect billing/routing (see HASH_EXCLUDE_FIELDS)
 *  - converts `undefined` → `null` so omitted fields hash the same as explicit null
 *  - sorts object keys alphabetically so { a:1, b:2 } and { b:2, a:1 } hash the same
 *
 * Using a blacklist instead of a whitelist means any new content-affecting
 * parameter (thinking, reasoning, verbosity, response_format, etc.) is
 * automatically included in the hash without requiring code changes here.
 */
function stableReplacer(key: string, value: unknown): unknown {
  if (HASH_EXCLUDE_FIELDS.has(key)) return undefined;
  if (value === undefined) return null;
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    );
  }
  return value;
}

function stripMessageBlockCacheControl(value: unknown, path: string[] = []): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => stripMessageBlockCacheControl(item, path));
  }
  if (value === null || typeof value !== "object") {
    return value;
  }

  const record = value as Record<string, unknown>;
  const next: Record<string, unknown> = {};

  for (const [key, child] of Object.entries(record)) {
    const nextPath = [...path, key];
    const isMessageContentBlockCacheControl =
      key === "cache_control" &&
      path.length >= 4 &&
      path[path.length - 1] !== "content" &&
      path[path.length - 2] === "content" &&
      path[path.length - 4] === "messages";

    if (isMessageContentBlockCacheControl) continue;
    next[key] = stripMessageBlockCacheControl(child, nextPath);
  }

  return next;
}

/**
 * Produce a stable SHA-256 cache key for a non-streaming request body.
 *
 * All content-affecting fields are included.  Fields that affect only
 * billing, routing, or observability are excluded (see HASH_EXCLUDE_FIELDS).
 * Additionally, message content blocks are normalized so block-level
 * cache_control noise does not perturb the local response-cache key.
 */
export function hashRequest(body: Record<string, unknown>): string {
  let payload: string;
  try {
    const normalizedBody = stripMessageBlockCacheControl(body);
    payload = JSON.stringify(normalizedBody, stableReplacer);
  } catch {
    // Extremely unlikely (HTTP bodies cannot contain circular refs / BigInt),
    // but fall back to a key that will never collide with a real entry.
    payload = `__unserializable__${String(body["model"])}__${Date.now()}__${Math.random()}`;
  }
  return createHash("sha256").update(payload).digest("hex").slice(0, 40);
}

// ---------------------------------------------------------------------------
// In-flight deduplication (thundering herd / cache stampede prevention)
//
// Problem: Two identical non-streaming requests arriving within milliseconds
// of each other both pass the cacheGet() check before either has completed,
// so both end up calling the upstream AI — doubling cost and latency.
//
// Solution: The first request for a given cache key registers itself as
// "in-flight".  Any subsequent identical request awaits that in-flight
// promise and then retries the cache lookup instead of firing its own
// upstream call.
//
// Node.js is single-threaded: all synchronous code runs without interleaving,
// so the Map mutations below are safe.  Only async boundaries (await) allow
// other requests to run between the cache miss check and the upstream call.
// ---------------------------------------------------------------------------

interface InflightEntry {
  promise: Promise<void>;
  resolve: () => void;
}
const inflightRequests = new Map<string, InflightEntry>();

/**
 * Register the current request as "in-flight" for this cache key.
 * Returns a finish() function the caller MUST invoke (in a finally block)
 * when the upstream call completes (success or failure).
 *
 * Always resolves (never rejects) so waiting requests fall through to a
 * cache lookup and, on miss, make their own upstream call.
 *
 * If a request is already in-flight for this key, returns null —
 * the caller should await waitForInflight() instead.
 */
export function markInflight(key: string): (() => void) | null {
  if (inflightRequests.has(key)) return null; // already in-flight
  let resolveFn!: () => void;
  const promise = new Promise<void>((res) => { resolveFn = res; });
  inflightRequests.set(key, { promise, resolve: resolveFn });
  return () => {
    const entry = inflightRequests.get(key);
    if (entry) {
      inflightRequests.delete(key);
      entry.resolve();
    }
  };
}

/**
 * Wait for the in-flight request for this key to finish, then return true
 * so the caller can retry the cache lookup.  Returns false if no request
 * is in-flight (caller should proceed normally).
 */
export async function waitForInflight(key: string): Promise<boolean> {
  const entry = inflightRequests.get(key);
  if (!entry) return false;
  try {
    await entry.promise;
  } catch {
    // The in-flight request failed — fall through and let the current
    // request try its own upstream call.
  }
  return true;
}

// ---------------------------------------------------------------------------
// Cache get / set / clear
// ---------------------------------------------------------------------------

/**
 * Retrieve a cached response. Returns the full entry so callers can check for 'chunks'.
 */
export function cacheGet(key: string): CacheEntry | null {
  if (!enabled) { misses++; return null; }
  const entry = cache.get(key);
  if (!entry) { misses++; return null; }
  if (Date.now() > entry.expiresAt) { cache.delete(key); misses++; return null; }
  hits++;
  return entry;
}

/**
 * Store a response in the cache, then schedule an async disk write so the
 * entry survives server restarts.
 */
export function cacheSet(key: string, data: unknown, model: string, chunks?: string[]): void {
  if (!enabled) return;
  if (cache.size >= maxEntries) {
    evictExpired();
    if (cache.size >= maxEntries) {
      // Evict the oldest entry (Map preserves insertion order).
      const oldestKey = cache.keys().next().value;
      if (oldestKey) cache.delete(oldestKey);
    }
  }
  cache.set(key, { data, expiresAt: Date.now() + ttlMs, model, chunks });
  scheduleDiskWrite();
}

export function cacheClear(): void {
  cache.clear();
  hits = 0;
  misses = 0;
  // Cancel any pending write and delete the disk file.
  if (diskWriteTimer) { clearTimeout(diskWriteTimer); diskWriteTimer = null; }
  fsp.unlink(CACHE_FILE).catch(() => { /* file may not exist */ });
}

export function getCacheStats(): CacheStats {
  evictExpired();
  return {
    hits,
    misses,
    size: cache.size,
    ttlMinutes: Math.round(ttlMs / 60000),
    enabled,
    maxEntries,
    inflight: inflightRequests.size,
    persistent: true,
  };
}

export function setCacheTtl(minutes: number): void {
  ttlMs = Math.max(1, minutes) * 60 * 1000;
}

export function setCacheEnabled(e: boolean): void {
  enabled = e;
}

export function setCacheMaxEntries(n: number): void {
  maxEntries = Math.max(1, n);
}
