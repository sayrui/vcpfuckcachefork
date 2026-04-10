import fs from "fs";
import path from "path";

export interface ModelEntry {
  id: string;
  enabled: boolean;
}

export interface ModelGroup {
  id: string;
  name: string;
  enabled: boolean;
  models: ModelEntry[];
}

const CONFIG_PATH = path.join(process.cwd(), "model-groups.json");

// ── Default definitions (mirrors models.ts) ────────────────────────────────

const CLAUDE_BASE = [
  "claude-opus-4-6", "claude-opus-4-5", "claude-opus-4-1",
  "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-haiku-4-5",
];
const CLAUDE_MODELS: string[] = [];
for (const b of CLAUDE_BASE) {
  CLAUDE_MODELS.push(b, `${b}-thinking`, `${b}-thinking-visible`);
}

const OPENAI_MODELS = [
  "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5-mini", "gpt-5-nano",
  "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini",
  "o4-mini", "o3", "o3-mini", "o4-mini-thinking", "o3-thinking", "o3-mini-thinking",
];

const GEMINI_BASE = [
  "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash",
];
const GEMINI_MODELS: string[] = [];
for (const b of GEMINI_BASE) {
  GEMINI_MODELS.push(b, `${b}-thinking`, `${b}-thinking-visible`);
}

const OPENROUTER_MODELS = [
  "x-ai/grok-4.20", "x-ai/grok-4.1-fast", "x-ai/grok-4-fast",
  "meta-llama/llama-4-maverick", "meta-llama/llama-4-scout",
  "deepseek/deepseek-v3.2", "deepseek/deepseek-r1", "deepseek/deepseek-r1-0528",
  "mistralai/mistral-small-2603", "qwen/qwen3.5-122b-a10b",
  "google/gemini-2.5-pro", "anthropic/claude-opus-4.6",
];

export const DEFAULT_GROUPS: ModelGroup[] = [
  {
    id: "anthropic",
    name: "Anthropic (Claude)",
    enabled: true,
    models: CLAUDE_MODELS.map(id => ({ id, enabled: true })),
  },
  {
    id: "openai",
    name: "OpenAI (GPT / o-series)",
    enabled: true,
    models: OPENAI_MODELS.map(id => ({ id, enabled: true })),
  },
  {
    id: "gemini",
    name: "Google (Gemini)",
    enabled: true,
    models: GEMINI_MODELS.map(id => ({ id, enabled: true })),
  },
  {
    id: "openrouter",
    name: "OpenRouter",
    enabled: true,
    models: OPENROUTER_MODELS.map(id => ({ id, enabled: true })),
  },
];

// ── Merge saved config with defaults ──────────────────────────────────────
// New models added to defaults are always present; saved toggles are respected.

function mergeWithDefaults(saved: ModelGroup[]): ModelGroup[] {
  return DEFAULT_GROUPS.map(defaultGroup => {
    const savedGroup = saved.find(g => g.id === defaultGroup.id);
    if (!savedGroup) return defaultGroup;
    return {
      ...defaultGroup,
      enabled: savedGroup.enabled,
      models: defaultGroup.models.map(dm => {
        const sm = savedGroup.models.find(m => m.id === dm.id);
        return sm ?? dm;
      }),
    };
  });
}

// ── In-process cache ──────────────────────────────────────────────────────
// model-groups.json almost never changes at runtime. Caching avoids a
// synchronous disk read on every /v1/chat/completions and /v1/models request.
// The cache is invalidated immediately after any write.

let _cache: ModelGroup[] | null = null;

// ── Public API ─────────────────────────────────────────────────────────────

export function readGroups(): ModelGroup[] {
  if (_cache) return _cache;
  try {
    const raw = fs.readFileSync(CONFIG_PATH, "utf-8");
    const saved = JSON.parse(raw) as ModelGroup[];
    _cache = mergeWithDefaults(saved);
    return _cache;
  } catch {
    // Config missing or corrupt — return defaults without caching so a
    // subsequent successful write will be picked up on the next request.
    return DEFAULT_GROUPS.map(g => ({ ...g, models: g.models.map(m => ({ ...m })) }));
  }
}

export function writeGroups(groups: ModelGroup[]): void {
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(groups, null, 2), "utf-8");
  _cache = null; // invalidate so the next read picks up the new config
}

/** Returns true if the model is allowed to handle requests. */
export function isModelEnabled(modelId: string): boolean {
  const groups = readGroups();
  for (const group of groups) {
    if (!group.enabled) continue;
    const model = group.models.find(m => m.id === modelId);
    if (model) return model.enabled;
  }
  // Model not listed in any group -- pass it through
  return true;
}

/** Returns all model IDs that are currently enabled. */
export function getEnabledModelIds(): string[] {
  const ids: string[] = [];
  for (const group of readGroups()) {
    if (!group.enabled) continue;
    for (const m of group.models) {
      if (m.enabled) ids.push(m.id);
    }
  }
  return ids;
}
