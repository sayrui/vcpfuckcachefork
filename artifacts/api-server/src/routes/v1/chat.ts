import { Router, type IRouter, type Request, type Response } from "express";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenAI } from "@google/genai";
import { authMiddleware } from "../../middlewares/auth";
import { isModelEnabled } from "../../lib/modelGroups";
import {
  buildPromptToolsInstruction,
  parsePromptToolsResponse,
  buildCompletionFromPromptTools,
  type PromptTool,
} from "../../lib/promptTools";
import { hashRequest, cacheGet, cacheSet, markInflight, waitForInflight } from "../../lib/responseCache";
import { logger } from "../../lib/logger";

const runtimeTimers = globalThis as unknown as {
  setInterval: (handler: () => void, timeout?: number) => number;
  clearInterval: (id?: number) => void;
};

// Fetch a remote image URL and return a base64 data URI
async function fetchImageAsBase64(url: string): Promise<string> {
  const res = await fetch(url, {
    signal: AbortSignal.timeout(30_000), // don't let a slow image host hang the request indefinitely
    headers: {
      "User-Agent": "Mozilla/5.0 (compatible; AI-Proxy/1.0)",
      "Accept": "image/*,*/*;q=0.8",
    },
    redirect: "follow",
  });
  if (!res.ok) throw new Error(`Failed to fetch image: ${res.status} ${url}`);
  const contentType = (res.headers.get("content-type") ?? "image/jpeg").split(";")[0]!;
  const buf = await res.arrayBuffer();
  const b64 = Buffer.from(buf).toString("base64");
  return `data:${contentType};base64,${b64}`;
}

// Convert image_url parts: replace remote URLs with base64 data URIs
// so the Replit AI Integrations proxy doesn't need to fetch external URLs itself
export async function resolveImageUrls(messages: OAIMessage[]): Promise<OAIMessage[]> {
  return Promise.all(
    messages.map(async (msg) => {
      if (!Array.isArray(msg.content)) return msg;
      const resolvedContent = await Promise.all(
        msg.content.map(async (part) => {
          if (
            part.type === "image_url" &&
            typeof (part as { image_url?: { url: string } }).image_url?.url === "string"
          ) {
            const { url } = (part as { type: "image_url"; image_url: { url: string } }).image_url;
            if (!url.startsWith("data:")) {
              try {
                const dataUri = await fetchImageAsBase64(url);
                return { ...part, image_url: { ...(part as { image_url: object }).image_url, url: dataUri } };
              } catch {
                // keep original if fetch fails
              }
            }
          }
          return part;
        })
      );
      return { ...msg, content: resolvedContent };
    })
  );
}

const router: IRouter = Router();

export const openai = new OpenAI({
  apiKey: process.env["AI_INTEGRATIONS_OPENAI_API_KEY"] ?? "dummy",
  baseURL: process.env["AI_INTEGRATIONS_OPENAI_BASE_URL"],
});

export const anthropic = new Anthropic({
  apiKey: process.env["AI_INTEGRATIONS_ANTHROPIC_API_KEY"] ?? "dummy",
  baseURL: process.env["AI_INTEGRATIONS_ANTHROPIC_BASE_URL"],
});

export const gemini = new GoogleGenAI({
  apiKey: process.env["AI_INTEGRATIONS_GEMINI_API_KEY"] ?? "dummy",
  httpOptions: {
    // Replit AI Integrations proxy does not use a /v1/ or /v1beta/ path prefix.
    // Setting apiVersion to "" removes the version segment from the URL so the
    // SDK calls {baseUrl}/models/{model}:generateContent instead of
    // {baseUrl}/v1beta/models/{model}:generateContent (INVALID_ENDPOINT).
    apiVersion: "",
    baseUrl: process.env["AI_INTEGRATIONS_GEMINI_BASE_URL"],
  },
});

export const openrouter = new OpenAI({
  apiKey: process.env["AI_INTEGRATIONS_OPENROUTER_API_KEY"] ?? "dummy",
  baseURL: process.env["AI_INTEGRATIONS_OPENROUTER_BASE_URL"],
});

// ----------------------------------------------------------------------
// Claude output ceilings used by the direct Anthropic route.
// Keep explicit low-cap models constrained, but do not artificially compress
// newer Sonnet / Opus generations to the old 64k ceiling.
// ----------------------------------------------------------------------
const CLAUDE_MAX_TOKENS: Record<string, number> = {
  "claude-haiku-4-5": 8096,
  "claude-sonnet-4-5": 128000,
  "claude-sonnet-4-6": 128000,
  "claude-opus-4-1": 128000,
  "claude-opus-4-5": 128000,
  "claude-opus-4-6": 128000,
};

export function getClaudeMaxTokens(model: string): number {
  return CLAUDE_MAX_TOKENS[model] ?? 200000;
}

// ----------------------------------------------------------------------
// OpenRouter reasoning model detection & defaults
// ----------------------------------------------------------------------

// All known OpenRouter reasoning models.
// OpenRouter normalises reasoning: { effort } ↔ reasoning: { max_tokens } across
// all providers, so using effort: "xhigh" universally avoids hardcoding any token
// limit while still requesting the maximum available reasoning budget.
const OPENROUTER_REASONING_MODELS: RegExp[] = [
  /^openai\/o\d/,               // o1, o3, o4 series
  /^openai\/gpt-5/,             // GPT-5 series
  /^x-ai\/grok.*mini/,          // Grok mini reasoning variants
  /^anthropic\/claude-3-7-sonnet/,
  /^anthropic\/claude-opus-4/,
  /^anthropic\/claude-sonnet-4/,
  /thinking/i,                  // Gemini :thinking, Qwen-thinking, etc.
  /^deepseek\/deepseek-r\d/,    // DeepSeek R-series
];

export function isOpenRouterReasoningModel(model: string): boolean {
  return OPENROUTER_REASONING_MODELS.some((re) => re.test(model));
}

/**
 * Returns { reasoning: { effort: "xhigh" } } for all known reasoning models.
 * OpenRouter normalises effort → max_tokens for providers that require it
 * (Anthropic, Gemini, etc.), so no token limit is hardcoded.
 * Returns {} if the caller already provided "reasoning" or the model is not
 * a recognised reasoning model.
 */
export function getOpenRouterReasoningDefault(
  model: string,
  passThrough: Record<string, unknown>
): Record<string, unknown> {
  if ("reasoning" in passThrough) return {};
  if (!isOpenRouterReasoningModel(model)) return {};
  return { reasoning: { effort: "xhigh" } };
}

/**
 * OpenRouter rejects requests where reasoning contains BOTH "effort" and
 * "max_tokens" simultaneously.  If the caller sent both, keep only "effort"
 * (the higher-level field that OpenRouter normalises across providers).
 * Returns a new passThrough object — does not mutate the original.
 */
export function sanitizeOpenRouterReasoning(
  passThrough: Record<string, unknown>
): Record<string, unknown> {
  const reasoning = passThrough["reasoning"];
  if (
    reasoning !== null &&
    typeof reasoning === "object" &&
    !Array.isArray(reasoning)
  ) {
    const r = reasoning as Record<string, unknown>;
    if ("effort" in r && "max_tokens" in r) {
      const { max_tokens: _dropped, ...rest } = r;
      return { ...passThrough, reasoning: rest };
    }
  }
  return passThrough;
}

/**
 * Returns a verbosity default for Opus models:
 *   - "max"  for Opus 4.6+ and Opus 5+ (only these support the "max" level per OR docs)
 *   - "high" for all other Opus variants (e.g. claude-opus-4-5)
 *
 * OR model IDs use dashes, e.g. "anthropic/claude-opus-4-6", so we match
 * both dash and dot separators between the minor-version digits.
 */
export function getOpenRouterVerbosityDefault(
  model: string,
  passThrough: Record<string, unknown>
): Record<string, unknown> {
  if ("verbosity" in passThrough) return {};
  // Opus 4.6+ (dash or dot separator) and Opus major version 5+
  if (/^anthropic\/claude-opus-4[-.]([6-9]|\d{2,})|^anthropic\/claude-opus-[5-9]/.test(model)) {
    return { verbosity: "max" };
  }
  // All other Opus models (e.g. 4-5) — "high" is the maximum supported level
  if (/^anthropic\/claude-opus-/.test(model)) {
    return { verbosity: "high" };
  }
  return {};
}

/** Max thinking budget: as large as possible while leaving at least 1024 for output. */
export function getThinkingBudget(maxTokens: number): number {
  return Math.max(1024, maxTokens - 1024);
}

export function stripClaudeSuffix(model: string): {
  baseModel: string;
  thinkingEnabled: boolean;
  thinkingVisible: boolean;
} {
  if (model.endsWith("-thinking-visible")) {
    return {
      baseModel: model.slice(0, -"-thinking-visible".length),
      thinkingEnabled: true,
      thinkingVisible: true,
    };
  }
  if (model.endsWith("-thinking")) {
    return {
      baseModel: model.slice(0, -"-thinking".length),
      thinkingEnabled: true,
      thinkingVisible: false,
    };
  }
  return { baseModel: model, thinkingEnabled: false, thinkingVisible: false };
}

// ----------------------------------------------------------------------
// Type aliases for incoming OpenAI-format request body
// ----------------------------------------------------------------------
type OAIContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string; detail?: string } }
  | { type: "tool_result"; tool_use_id?: string; content?: string } // not real OAI but keep safe
  | Record<string, unknown>;

export interface OAIMessage {
  role: string;
  content: string | OAIContentPart[] | null;
  name?: string;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string; // role === "tool"
}

interface OAITool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

type OAIToolChoice =
  | "auto"
  | "none"
  | "required"
  | { type: "function"; function: { name: string } };

export interface ChatBody {
  model: string;
  messages: OAIMessage[];
  stream?: boolean;
  // generation params
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  n?: number;
  stop?: string | string[];
  seed?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  logprobs?: boolean;
  top_logprobs?: number;
  user?: string;
  // tools
  tools?: OAITool[];
  tool_choice?: OAIToolChoice;
  parallel_tool_calls?: boolean;
  // response_format
  response_format?: { type: string };
  // Anthropic top-level prompt caching (OpenRouter automatic caching or direct Anthropic)
  cache_control?: { type: "ephemeral"; ttl?: string };
  // Prompt-based tool calling fallback (any model, any route)
  x_use_prompt_tools?: boolean;
  // allow any provider-specific extra fields (e.g. OpenRouter: provider, transforms, route, etc.)
  [key: string]: unknown;
}

// ----------------------------------------------------------------------
// Message conversion: OpenAI -> Anthropic
// ----------------------------------------------------------------------

export function oaiContentToAnthropic(
  content: string | OAIContentPart[] | null
): Anthropic.ContentBlockParam[] {
  if (content === null || content === undefined) return [];
  if (typeof content === "string") {
    // Anthropic rejects empty text blocks
    return content.length > 0 ? [{ type: "text", text: content }] : [];
  }

  const blocks: Anthropic.ContentBlockParam[] = [];
  for (const part of content) {
    if (part.type === "text" && typeof (part as { text?: string }).text === "string") {
      const text = (part as { text: string }).text;
      if (text.length === 0) continue; // Anthropic rejects empty text blocks
      // Preserve cache_control if present (e.g. for Anthropic prompt caching breakpoints)
      const cc = (part as Record<string, unknown>)["cache_control"];
      const block: Record<string, unknown> = { type: "text", text };
      if (cc !== undefined) block["cache_control"] = cc;
      blocks.push(block as Anthropic.ContentBlockParam);
    } else if (part.type === "image_url") {
      const { url } = (part as { type: "image_url"; image_url: { url: string } }).image_url;
      if (url.startsWith("data:")) {
        const commaIdx = url.indexOf(",");
        const meta = url.slice(5, commaIdx); // strip "data:"
        const data = url.slice(commaIdx + 1);
        const mediaType = meta.split(";")[0] as Anthropic.Base64ImageSource["media_type"];
        blocks.push({ type: "image", source: { type: "base64", media_type: mediaType, data } });
      } else {
        blocks.push({ type: "image", source: { type: "url", url } });
      }
    }
    // skip unknown part types
  }
  return blocks;
}

export function convertMessagesToAnthropic(messages: OAIMessage[]): {
  system: string | Anthropic.TextBlockParam[] | undefined;
  messages: Anthropic.MessageParam[];
} {
  let system: string | Anthropic.TextBlockParam[] | undefined;
  const converted: Anthropic.MessageParam[] = [];

  // We may need to merge consecutive tool results into a single user message
  // (Anthropic requires user/assistant alternation)
  let pendingToolResults: Anthropic.ToolResultBlockParam[] = [];

  const flushToolResults = () => {
    if (pendingToolResults.length > 0) {
      converted.push({ role: "user", content: [...pendingToolResults] });
      pendingToolResults = [];
    }
  };

  for (const msg of messages) {
    // -- system ------------------------------------------------------
    if (msg.role === "system") {
      if (typeof msg.content === "string") {
        system = msg.content;
      } else if (Array.isArray(msg.content)) {
        // If any block carries cache_control, preserve as TextBlockParam[] so
        // Anthropic's caching breakpoints are respected.
        const hasCache = msg.content.some(
          (p) => (p as Record<string, unknown>)["cache_control"] !== undefined,
        );
        if (hasCache) {
          // Filter to text blocks only — Anthropic's `system` field does not accept image blocks.
          const allBlocks = oaiContentToAnthropic(msg.content);
          const textBlocks = allBlocks.filter(
            (b): b is Anthropic.TextBlockParam => b.type === "text",
          );
          // Only use array form when there's actually content; fall back to string otherwise.
          system = textBlocks.length > 0 ? textBlocks : "";
        } else {
          system = msg.content
            .filter((p) => (p as { type: string }).type === "text")
            .map((p) => (p as { text: string }).text)
            .join("\n");
        }
      } else {
        system = "";
      }
      continue;
    }

    // -- tool result (role === "tool") --------------------------------
    if (msg.role === "tool") {
      const resultContent =
        typeof msg.content === "string"
          ? msg.content
          : Array.isArray(msg.content)
            ? msg.content.map(p => (p as { text?: string }).text ?? "").join("")
            : "";
      pendingToolResults.push({
        type: "tool_result",
        tool_use_id: msg.tool_call_id ?? "",
        content: resultContent,
      });
      continue;
    }

    // If we had accumulated tool results and now see a non-tool role, flush
    flushToolResults();

    // -- assistant ----------------------------------------------------
    if (msg.role === "assistant") {
      const content: Anthropic.ContentBlockParam[] = [];

      // text content
      const textBlocks = oaiContentToAnthropic(msg.content);
      content.push(...textBlocks);

      // tool_calls -> tool_use blocks
      if (Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls) {
          let parsedInput: Record<string, unknown> = {};
          try {
            parsedInput = JSON.parse(tc.function.arguments || "{}");
          } catch {
            parsedInput = {};
          }
          content.push({
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input: parsedInput,
          });
        }
      }

      if (content.length === 0) content.push({ type: "text", text: "" });
      converted.push({ role: "assistant", content });
      continue;
    }

    // -- user ---------------------------------------------------------
    const userContent = oaiContentToAnthropic(msg.content);
    if (userContent.length === 0) continue;
    converted.push({ role: "user", content: userContent });
  }

  flushToolResults();
  return { system, messages: converted };
}

// ----------------------------------------------------------------------
// Tool conversion: OpenAI tools -> Anthropic tools
// ----------------------------------------------------------------------

export function convertToolsToAnthropic(tools: OAITool[]): Anthropic.Tool[] {
  return tools.map((t) => ({
    name: t.function.name,
    description: t.function.description,
    input_schema: (t.function.parameters ?? { type: "object", properties: {} }) as Anthropic.Tool["input_schema"],
  }));
}

export function convertToolChoiceToAnthropic(
  tc: OAIToolChoice | undefined
): Anthropic.ToolChoiceParam | undefined {
  if (!tc || tc === "none") return undefined;
  if (tc === "auto") return { type: "auto" };
  if (tc === "required") return { type: "any" };
  if (typeof tc === "object" && tc.type === "function") {
    return { type: "tool", name: tc.function.name };
  }
  return { type: "auto" };
}

// ----------------------------------------------------------------------
// Anthropic prompt-caching auto-injection
// ----------------------------------------------------------------------

/**
 * Auto-inject Anthropic prompt-caching breakpoints into a Claude request.
 *
 * Strategy (Anthropic docs):
 *   1. System prompt  → last TextBlock gets cache_control
 *   2. Stable history → last content block of the penultimate message gets
 *      cache_control (the final message is the new query — skip it)
 *   3. Tool definitions → last tool gets cache_control
 *
 * No-op when:
 *   - The caller already placed any cache_control on a block (manual wins)
 *   - Nothing worth caching (no system, single-turn, no tools)
 *
 * Returns new objects; inputs are never mutated.
 */
export function autoInjectPromptCaching(opts: {
  system: string | Anthropic.TextBlockParam[] | undefined;
  messages: Anthropic.MessageParam[];
  tools: Anthropic.Tool[] | undefined;
}): {
  system: string | Anthropic.TextBlockParam[] | undefined;
  messages: Anthropic.MessageParam[];
  tools: Anthropic.Tool[] | undefined;
} {
  const cc = { type: "ephemeral" as const, ttl: "1h" };

  // Bail out if caller already set cache_control anywhere — respect manual control.
  const hasCC = (x: unknown): boolean =>
    !!(x !== null && typeof x === "object" && "cache_control" in (x as object) &&
      (x as Record<string, unknown>)["cache_control"]);
  const alreadyCached =
    (Array.isArray(opts.system) && opts.system.some(hasCC)) ||
    opts.messages.some(
      (m) => Array.isArray(m.content) && (m.content as unknown[]).some(hasCC),
    ) ||
    (opts.tools ?? []).some(hasCC);

  if (alreadyCached) return opts;

  let { system, messages, tools } = opts;

  // 1. System prompt caching
  //    string → convert to TextBlockParam[] so we can attach cache_control
  //    TextBlockParam[] → add cache_control to the last block
  if (typeof system === "string" && system.length > 0) {
    system = [{ type: "text", text: system, cache_control: cc }];
  } else if (Array.isArray(system) && system.length > 0) {
    const last = system[system.length - 1];
    system = [...system.slice(0, -1), { ...last, cache_control: cc }];
  }

  // 2. Stable conversation history: last *cacheable* content block of the penultimate message.
  //    Only text / image / tool_result blocks support cache_control.
  //    tool_use and thinking blocks do NOT — Anthropic returns 400 if you try.
  //    The very last message is the current user query — skip it.
  if (messages.length >= 2) {
    const CACHEABLE_BLOCK_TYPES = new Set(["text", "image", "tool_result"]);
    const ti = messages.length - 2;
    const target = messages[ti];
    if (Array.isArray(target.content)) {
      const blocks = target.content as Anthropic.ContentBlockParam[];
      // Find the last block whose type supports cache_control (walk backwards)
      let insertAt = -1;
      for (let i = blocks.length - 1; i >= 0; i--) {
        const bt = (blocks[i] as unknown as Record<string, unknown>)["type"];
        if (typeof bt === "string" && CACHEABLE_BLOCK_TYPES.has(bt)) {
          insertAt = i;
          break;
        }
      }
      if (insertAt >= 0) {
        const updated = blocks.map((b, i) =>
          i === insertAt
            ? ({ ...(b as unknown as Record<string, unknown>), cache_control: cc } as Anthropic.ContentBlockParam)
            : b,
        );
        messages = [
          ...messages.slice(0, ti),
          { ...target, content: updated },
          ...messages.slice(ti + 1),
        ];
      }
    }
  }

  // 3. Tool definitions: last tool gets cache_control
  if (tools && tools.length > 0) {
    const lastTool = tools[tools.length - 1];
    tools = [
      ...tools.slice(0, -1),
      { ...(lastTool as unknown as Record<string, unknown>), cache_control: cc } as Anthropic.Tool,
    ];
  }

  return { system, messages, tools };
}

function isOpenRouterAnthropicModel(model: string): boolean {
  return /^anthropic\/claude-/.test(model);
}

function hasBlockLevelCacheControlInOpenAIMessages(messages: OAIMessage[]): boolean {
  return messages.some((message) => {
    if (!Array.isArray(message.content)) return false;
    return message.content.some((part) => {
      const cacheControl = (part as Record<string, unknown>)["cache_control"];
      return cacheControl !== undefined && cacheControl !== null;
    });
  });
}

function cloneMessageContentWithInjectedCacheControl(
  message: OAIMessage,
  targetPartIndex?: number,
): OAIMessage | null {
  if (typeof message.content === "string") {
    if (!message.content.length) return null;
    return {
      ...message,
      content: [{ type: "text", text: message.content, cache_control: { type: "ephemeral" as const } }],
    };
  }

  if (!Array.isArray(message.content) || message.content.length === 0) return null;

  const textPartIndexes: number[] = [];
  for (let i = 0; i < message.content.length; i++) {
    const part = message.content[i];
    if (part.type === "text" && typeof (part as { text?: string }).text === "string" && (part as { text: string }).text.length > 0) {
      textPartIndexes.push(i);
    }
  }

  if (textPartIndexes.length === 0) return null;

  const injectAt = targetPartIndex ?? textPartIndexes[textPartIndexes.length - 1]!;
  const nextContent = message.content.map((part, index) =>
    index === injectAt
      ? ({ ...(part as Record<string, unknown>), cache_control: { type: "ephemeral" as const } } as OAIContentPart)
      : part,
  );

  return { ...message, content: nextContent };
}

type OpenRouterAnthropicCacheStrategy =
  | "not_applicable"
  | "pass_through_existing_block"
  | "block"
  | "no_eligible_block";

function prepareOpenRouterAnthropicCachingRequest(messages: OAIMessage[]): {
  messages: OAIMessage[];
  strategy: OpenRouterAnthropicCacheStrategy;
} {
  if (hasBlockLevelCacheControlInOpenAIMessages(messages)) {
    return { messages, strategy: "pass_through_existing_block" };
  }

  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.role !== "system") continue;
    const injected = cloneMessageContentWithInjectedCacheControl(message);
    if (!injected) continue;
    return {
      messages: messages.map((current, index) => (index === i ? injected : current)),
      strategy: "block",
    };
  }

  const firstNonSystemIndex = messages.findIndex((message) => message.role !== "system");
  for (let i = Math.max(0, firstNonSystemIndex + 1); i < messages.length; i++) {
    const message = messages[i];
    if (message.role !== "user") continue;
    const injected = cloneMessageContentWithInjectedCacheControl(message);
    if (!injected) continue;
    return {
      messages: messages.map((current, index) => (index === i ? injected : current)),
      strategy: "block",
    };
  }

  return { messages, strategy: "no_eligible_block" };
}

function shouldSkipDynamicSinkingForStickyRouting(model: string, messages: OAIMessage[]): boolean {
  if (!isOpenRouterAnthropicModel(model)) return false;
  return !messages.some((message) => message.role === "assistant" || message.role === "tool");
}

function extractOpenRouterPromptCacheMetrics(
  response: unknown,
  fallbackModel: string,
  cacheStrategy: OpenRouterAnthropicCacheStrategy,
): {
  model: string;
  cacheStrategy: OpenRouterAnthropicCacheStrategy;
  promptTokens?: number;
  completionTokens?: number;
  cachedTokens?: number;
  cacheWriteTokens?: number;
  hasUsage: boolean;
  activity: "read" | "write" | "none";
} {
  const payload =
    response !== null && typeof response === "object"
      ? (response as Record<string, unknown>)
      : {};
  const usage =
    payload["usage"] !== null && typeof payload["usage"] === "object"
      ? (payload["usage"] as Record<string, unknown>)
      : null;
  const promptTokenDetails =
    usage?.["prompt_tokens_details"] !== null &&
    typeof usage?.["prompt_tokens_details"] === "object"
      ? (usage["prompt_tokens_details"] as Record<string, unknown>)
      : null;

  const promptTokens =
    typeof usage?.["prompt_tokens"] === "number" ? usage["prompt_tokens"] : undefined;
  const completionTokens =
    typeof usage?.["completion_tokens"] === "number"
      ? usage["completion_tokens"]
      : undefined;
  const cachedTokens =
    typeof promptTokenDetails?.["cached_tokens"] === "number"
      ? promptTokenDetails["cached_tokens"]
      : undefined;
  const cacheWriteTokens =
    typeof promptTokenDetails?.["cache_write_tokens"] === "number"
      ? promptTokenDetails["cache_write_tokens"]
      : undefined;

  let activity: "read" | "write" | "none" = "none";
  if ((cachedTokens ?? 0) > 0) {
    activity = "read";
  } else if ((cacheWriteTokens ?? 0) > 0) {
    activity = "write";
  }

  return {
    model:
      typeof payload["model"] === "string" && payload["model"].length > 0
        ? (payload["model"] as string)
        : fallbackModel,
    cacheStrategy,
    promptTokens,
    completionTokens,
    cachedTokens,
    cacheWriteTokens,
    hasUsage: usage !== null,
    activity,
  };
}

function logOpenRouterPromptCacheMetrics(
  req: Request,
  source: "stream" | "non_stream",
  metrics: ReturnType<typeof extractOpenRouterPromptCacheMetrics>,
): void {
  const payload = {
    source,
    model: metrics.model,
    cacheStrategy: metrics.cacheStrategy,
    promptTokens: metrics.promptTokens,
    completionTokens: metrics.completionTokens,
    cachedTokens: metrics.cachedTokens ?? 0,
    cacheWriteTokens: metrics.cacheWriteTokens ?? 0,
  };

  if (!metrics.hasUsage) {
    req.log.info(payload, "openrouter upstream prompt cache usage unavailable");
    return;
  }

  if (metrics.activity === "read") {
    req.log.info(payload, "openrouter upstream prompt cache read");
    return;
  }

  if (metrics.activity === "write") {
    req.log.info(payload, "openrouter upstream prompt cache write");
    return;
  }

  req.log.info(payload, "openrouter upstream no prompt cache activity");
}

// ----------------------------------------------------------------------
// Gemini model config helpers
// ----------------------------------------------------------------------

export function stripGeminiSuffix(model: string): { baseModel: string; thinkingEnabled: boolean } {
  if (model.endsWith("-thinking-visible")) {
    return { baseModel: model.slice(0, -"-thinking-visible".length), thinkingEnabled: true };
  }
  if (model.endsWith("-thinking")) {
    return { baseModel: model.slice(0, -"-thinking".length), thinkingEnabled: true };
  }
  return { baseModel: model, thinkingEnabled: false };
}

// ----------------------------------------------------------------------
// Message conversion: OpenAI -> Gemini
// ----------------------------------------------------------------------

type GeminiPart = { text: string };
type GeminiContent = { role: "user" | "model"; parts: GeminiPart[] };

function oaiContentToGeminiParts(content: string | OAIContentPart[] | null): GeminiPart[] {
  if (!content) return [{ text: "" }];
  if (typeof content === "string") return [{ text: content }];
  const parts: GeminiPart[] = [];
  for (const part of content) {
    if (part.type === "text" && typeof (part as { text?: string }).text === "string") {
      parts.push({ text: (part as { text: string }).text });
    }
  }
  return parts.length > 0 ? parts : [{ text: "" }];
}

export function convertMessagesToGemini(messages: OAIMessage[]): {
  systemInstruction: string | undefined;
  contents: GeminiContent[];
} {
  let systemInstruction: string | undefined;
  const contents: GeminiContent[] = [];

  for (const msg of messages) {
    if (msg.role === "system") {
      const text = typeof msg.content === "string"
        ? msg.content
        : (Array.isArray(msg.content)
          ? (msg.content as OAIContentPart[])
              .filter(p => p.type === "text")
              .map(p => (p as { text: string }).text)
              .join("\n")
          : "");
      systemInstruction = systemInstruction ? `${systemInstruction}\n${text}` : text;
    } else {
      const role: "user" | "model" = msg.role === "assistant" ? "model" : "user";
      const parts = oaiContentToGeminiParts(msg.content);
      // Merge consecutive same-role messages into one to satisfy Gemini alternation rule
      const last = contents[contents.length - 1];
      if (last && last.role === role) {
        last.parts.push(...parts);
      } else {
        contents.push({ role, parts });
      }
    }
  }

  // Gemini requires the first turn to be user; inject a stub if needed
  if (contents.length === 0 || contents[0].role !== "user") {
    contents.unshift({ role: "user", parts: [{ text: "." }] });
  }

  return { systemInstruction, contents };
}

// ----------------------------------------------------------------------
// SSE chunk helpers
// ----------------------------------------------------------------------

function makeChunk(
  id: string,
  model: string,
  delta: Record<string, unknown>,
  finishReason?: string | null
) {
  return {
    id,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason ?? null }],
  };
}

function sseWrite(res: Response, data: unknown) {
  if (res.writableEnded) return; // client already disconnected
  let json: string;
  try {
    json = JSON.stringify(data);
  } catch {
    // Circular refs or BigInt in upstream payload — emit a safe error chunk instead
    json = JSON.stringify({ error: { message: "Response serialization error", type: "proxy_error" } });
  }
  try {
    res.write(`data: ${json}\n\n`);
  } catch { /* socket closed between writableEnded check and write — ignore */ }
}

/**
 * Write a keepalive heartbeat as a proper SSE `data:` event, not a comment.
 *
 * Why not just `": keepalive\n\n"` (SSE comment)?
 * Replit's reverse proxy (and many others) measure "activity" at the HTTP
 * response-body level — meaning only bytes in `data:` lines are guaranteed
 * to reset the idle timer.  Comment lines (`: ...`) are valid SSE but some
 * proxies treat them as zero-payload and don't count them.
 *
 * An empty-choices chunk is harmless: every compliant OAI client checks
 * `choices.length` before processing deltas and silently skips empty arrays.
 *
 * Interval: 200 s — safely below Replit's 300 s hard timeout while keeping
 * the stream noise minimal.
 */
const SSE_KEEPALIVE_MS = 200_000; // 200 seconds

function sseKeepalive(res: Response, model: string) {
  sseWrite(res, {
    id: `ka-${Date.now()}`,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [],
  });
}

/**
 * Instant replay of a cached stream.
 */
function replayStream(res: Response, chunks: string[]) {
  setSseHeaders(res);
  for (const chunk of chunks) {
    res.write(`data: ${chunk}\n\n`);
  }
  res.write("data: [DONE]\n\n");
  res.end();
}

function setSseHeaders(res: Response) {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("X-Accel-Buffering", "no");
  if (res.socket) {
    // Disable Nagle's algorithm — each token chunk is sent immediately without buffering.
    res.socket.setNoDelay(true);
    // Belt-and-suspenders: reset the socket-level idle timeout to 0 (infinite) so that
    // active token streams are never cut by a stale OS/proxy socket timeout.
    res.socket.setTimeout(0);
  }
}

/**
 * Prevents Replit's 300-second proxy timeout on non-streaming JSON responses.
 * Writes a JSON-safe newline (leading whitespace is valid per JSON spec) every
 * 20 seconds so the proxy sees data flowing and does not cut the connection.
 * Call clearInterval(returned id) then res.end(json) when the upstream resolves.
 */
function startNonStreamKeepalive(res: Response): number {
  res.setHeader("Content-Type", "application/json");
  res.setHeader("X-Accel-Buffering", "no");
  return runtimeTimers.setInterval(() => {
    if (!res.writableEnded) res.write("\n");
  }, 20_000);
}

function endNonStream(res: Response, data: unknown, cacheParams?: { key: string; model: string }): void {
  // Serialize first — we need the JSON string regardless of whether the client is still connected.
  let json: string;
  let serialized = true;
  try {
    json = JSON.stringify(data);
  } catch {
    // Serialization failure — send a safe error, but do NOT cache broken data.
    json = JSON.stringify({ error: { message: "Response serialization error", type: "proxy_error" } });
    serialized = false;
  }

  // Cache BEFORE the writableEnded guard so future requests benefit even when
  // this client disconnected mid-response (a common cause of missed cache writes).
  if (cacheParams && serialized) {
    cacheSet(cacheParams.key, data, cacheParams.model);
    logger.info({ cacheKey: cacheParams.key.slice(0, 8), model: cacheParams.model }, "cache SET");
  }

  if (res.writableEnded) {
    logger.warn({ cacheKey: cacheParams?.key.slice(0, 8) }, "endNonStream: response already ended, skipping send");
    return;
  }
  res.end(json);
}

function endNonStreamError(res: Response, statusCode: number, message: string, type: string): void {
  if (!res.writableEnded) {
    if (!res.headersSent) res.status(statusCode);
    res.end(JSON.stringify({ error: { message, type } }));
  }
}

/**
 * Extract a human-readable message and HTTP status from an upstream API error.
 * OpenAI/Anthropic SDK errors carry a numeric `.status` property; all other
 * thrown values default to 502 Bad Gateway.
 */
function extractUpstreamError(err: unknown): { status: number; message: string } {
  const message = err instanceof Error ? err.message : String(err);
  const status =
    err !== null && typeof err === "object" && "status" in err &&
    typeof (err as { status: unknown }).status === "number"
      ? Math.max(400, (err as { status: number }).status)
      : 502;
  return { status, message };
}

// ----------------------------------------------------------------------
// Anthropic - streaming
// ----------------------------------------------------------------------

async function handleClaudeStream(
  _req: Request,
  res: Response,
  body: ChatBody,
  cacheKey?: string,
) {
  // Establish SSE connection immediately -- before any async work so the client
  // does not wait for image-URL resolution before the first byte arrives.
  setSseHeaders(res);
  res.write(": init\n\n");

  const { model, temperature, top_p, stop, tools, tool_choice } = body;
  const messages = await resolveImageUrls(body.messages);
  const { baseModel, thinkingEnabled, thinkingVisible } = stripClaudeSuffix(model);
  const modelMax = getClaudeMaxTokens(baseModel);
  // Clamp caller value to the model's hard limit so Anthropic never returns a 422.
  // If the caller omitted max_tokens entirely, default to the model's maximum.
  const rawMaxTokens = body.max_tokens && body.max_tokens > 0 ? body.max_tokens : modelMax;
  const maxTokens = Math.min(rawMaxTokens, modelMax);

  const { system: rawSystem, messages: rawAnthropicMessages } = convertMessagesToAnthropic(messages);

  const rawAnthropicTools = (tools && tools.length > 0 && tool_choice !== "none")
    ? convertToolsToAnthropic(tools)
    : undefined;
  const anthropicToolChoice = (tool_choice && tool_choice !== "none")
    ? convertToolChoiceToAnthropic(tool_choice)
    : undefined;

  // Auto-inject prompt-caching breakpoints (no-op if caller already set cache_control).
  // Wrapped in try-catch: any injection error falls back to original params so the
  // request always proceeds normally.
  let system = rawSystem;
  let anthropicMessages = rawAnthropicMessages;
  let anthropicTools = rawAnthropicTools;
  try {
    ({ system, messages: anthropicMessages, tools: anthropicTools } = autoInjectPromptCaching({
      system: rawSystem,
      messages: rawAnthropicMessages,
      tools: rawAnthropicTools,
    }));
  } catch { /* ignore — proceed without caching */ }

  const params: Record<string, unknown> = {
    model: baseModel,
    max_tokens: maxTokens,
    messages: anthropicMessages,
    stream: true,
  };
  if (system) params["system"] = system;
  if (!thinkingEnabled) {
    if (temperature !== undefined) params["temperature"] = temperature;
    else if (top_p !== undefined) params["top_p"] = top_p;
    if (stop) params["stop_sequences"] = Array.isArray(stop) ? stop : [stop];
  }
  // Thinking budget: max possible while leaving at least 1024 tokens for visible output.
  if (thinkingEnabled) params["thinking"] = { type: "enabled", budget_tokens: getThinkingBudget(maxTokens) };
  if (anthropicTools) params["tools"] = anthropicTools;
  if (anthropicToolChoice) params["tool_choice"] = anthropicToolChoice;
  // Top-level cache_control for automatic Anthropic prompt caching (requires Anthropic provider)
  if (body["cache_control"]) params["cache_control"] = body["cache_control"];

  const id = `chatcmpl-${Date.now()}`;
  const recordedChunks: string[] = [];
  const sseWriteAndRecord = (data: unknown) => {
    const json = JSON.stringify(data);
    recordedChunks.push(json);
    sseWrite(res, data);
  };

  let inThinking = false;
  let inputTokens = 0; // captured from message_start, used in final usage chunk
  let cacheReadTokens = 0;
  let cacheCreationTokens = 0;
  // Map Anthropic content-block index -> OAI tool_calls index (0-based among tool_use blocks only)
  const blockIdxToToolIdx: Record<number, number> = {};
  let toolCallCount = 0;

  // NOTE: ": init\n\n" has already been written above, so headers are always committed
  // before we reach this try block. The catch must always use the SSE error path —
  // re-throwing into the outer catch would find headersSent=true and send nothing,
  // leaving the client connection hanging indefinitely.
  const keepaliveInterval = runtimeTimers.setInterval(() => {
    sseKeepalive(res, model);
  }, SSE_KEEPALIVE_MS);

  try {
    const stream = anthropic.messages.stream(params as Anthropic.MessageCreateParamsStreaming);

    for await (const event of stream) {
      if (event.type === "message_start") {
        // Capture input token count (including cache tokens) for the final usage report
        inputTokens = event.message.usage?.input_tokens ?? 0;
        cacheReadTokens = (event.message.usage as Record<string, unknown>)?.["cache_read_input_tokens"] as number ?? 0;
        cacheCreationTokens = (event.message.usage as Record<string, unknown>)?.["cache_creation_input_tokens"] as number ?? 0;
        sseWriteAndRecord(makeChunk(id, model, { role: "assistant", content: "" }));

      } else if (event.type === "content_block_start") {
        const block = event.content_block;
        const idx = event.index;

        if (block.type === "thinking") {
          inThinking = true;
          if (thinkingVisible) {
            sseWriteAndRecord(makeChunk(id, model, { content: "<thinking>\n" }));
          }
        } else if (block.type === "text") {
          if (inThinking && thinkingVisible) {
            sseWriteAndRecord(makeChunk(id, model, { content: "\n</thinking>\n\n" }));
          }
          inThinking = false;
        } else if (block.type === "tool_use") {
          // Assign a sequential OAI tool call index (0-based) independent of content block index
          const toolIdx = toolCallCount++;
          blockIdxToToolIdx[idx] = toolIdx;
          sseWriteAndRecord(makeChunk(id, model, {
            tool_calls: [{
              index: toolIdx,
              id: block.id,
              type: "function",
              function: { name: block.name, arguments: "" },
            }],
          }));
        }

      } else if (event.type === "content_block_delta") {
        const delta = event.delta;
        const idx = event.index;

        if (delta.type === "thinking_delta") {
          if (thinkingVisible) {
            sseWriteAndRecord(makeChunk(id, model, { content: delta.thinking }));
          }
        } else if (delta.type === "text_delta") {
          sseWriteAndRecord(makeChunk(id, model, { content: delta.text }));
        } else if (delta.type === "input_json_delta") {
          // Use the mapped OAI tool call index, not the Anthropic content block index.
          // B3 fix: guard null/undefined partial_json — can occur on the first delta.
          const toolIdx = blockIdxToToolIdx[idx] ?? 0;
          sseWriteAndRecord(makeChunk(id, model, {
            tool_calls: [{
              index: toolIdx,
              function: { arguments: delta.partial_json ?? "" },
            }],
          }));
        }

      } else if (event.type === "message_delta") {
        // B2 fix: if the model produced only thinking blocks (no text block ever opened),
        // the </thinking> closing tag was never emitted — close it now before the final chunk.
        if (inThinking && thinkingVisible) {
          sseWriteAndRecord(makeChunk(id, model, { content: "\n</thinking>\n\n" }));
          inThinking = false;
        }
        const stopReason = event.delta.stop_reason;
        const finishReason =
          stopReason === "tool_use" ? "tool_calls"
          : stopReason === "end_turn" ? "stop"
          : (stopReason ?? "stop");
        // Build accurate usage: input_tokens from message_start + output_tokens from message_delta
        const outputTokens = event.usage?.output_tokens ?? 0;
        const usage: Record<string, unknown> = {
          prompt_tokens: inputTokens,
          completion_tokens: outputTokens,
          total_tokens: inputTokens + outputTokens,
        };
        // Pass through Anthropic cache token fields so clients can observe cache hits/misses
        if (cacheReadTokens > 0 || cacheCreationTokens > 0) {
          usage["cache_read_input_tokens"] = cacheReadTokens;
          usage["cache_creation_input_tokens"] = cacheCreationTokens;
          usage["prompt_tokens_details"] = { cached_tokens: cacheReadTokens };
        }
        sseWriteAndRecord({ ...makeChunk(id, model, {}, finishReason), usage });
      }
    }

    if (cacheKey && recordedChunks.length > 0) {
      cacheSet(cacheKey, { recorded: true }, model, recordedChunks);
      logger.info({ cacheKey: cacheKey.slice(0, 8), model }, "stream cache SET");
    }

    if (!res.writableEnded) res.write("data: [DONE]\n\n");
  } catch (streamErr) {
    // Always use the SSE error path here — headers were committed by ": init\n\n" above,
    // so re-throwing would leave the client connection hanging with no response body.
    try {
      sseWrite(res, {
        error: {
          message: streamErr instanceof Error ? streamErr.message : "Stream error",
          type: "stream_error",
        },
      });
      if (!res.writableEnded) res.write("data: [DONE]\n\n");
    } catch { /* ignore write errors during cleanup */ }
  } finally {
    runtimeTimers.clearInterval(keepaliveInterval);
    if (!res.writableEnded) res.end();
  }
}

// ----------------------------------------------------------------------
// Anthropic - non-streaming
// ----------------------------------------------------------------------

async function handleClaudeNonStream(
  _req: Request,
  res: Response,
  body: ChatBody,
  cacheKey?: string,
) {
  const { model, temperature, top_p, stop, tools, tool_choice } = body;
  const messages = await resolveImageUrls(body.messages);
  const { baseModel, thinkingEnabled, thinkingVisible } = stripClaudeSuffix(model);
  const modelMax = getClaudeMaxTokens(baseModel);
  // Clamp caller value to the model's hard limit so Anthropic never returns a 422.
  const rawMaxTokens = body.max_tokens && body.max_tokens > 0 ? body.max_tokens : modelMax;
  const maxTokens = Math.min(rawMaxTokens, modelMax);

  const { system: rawSystem, messages: rawAnthropicMessages } = convertMessagesToAnthropic(messages);

  // When tool_choice is "none", suppress tools entirely (Anthropic has no "none" option)
  const rawAnthropicTools = (tools && tools.length > 0 && tool_choice !== "none")
    ? convertToolsToAnthropic(tools)
    : undefined;
  const anthropicToolChoice = (tool_choice && tool_choice !== "none")
    ? convertToolChoiceToAnthropic(tool_choice)
    : undefined;

  // Auto-inject prompt-caching breakpoints (no-op if caller already set cache_control).
  // Wrapped in try-catch: any injection error falls back to original params so the
  // request always proceeds normally.
  let system = rawSystem;
  let anthropicMessages = rawAnthropicMessages;
  let anthropicTools = rawAnthropicTools;
  try {
    ({ system, messages: anthropicMessages, tools: anthropicTools } = autoInjectPromptCaching({
      system: rawSystem,
      messages: rawAnthropicMessages,
      tools: rawAnthropicTools,
    }));
  } catch { /* ignore — proceed without caching */ }

  const params: Record<string, unknown> = {
    model: baseModel,
    max_tokens: maxTokens,
    messages: anthropicMessages,
    stream: false,
  };
  if (system) params["system"] = system;
  if (!thinkingEnabled) {
    if (temperature !== undefined) params["temperature"] = temperature;
    else if (top_p !== undefined) params["top_p"] = top_p;
    if (stop) params["stop_sequences"] = Array.isArray(stop) ? stop : [stop];
  }
  // Thinking budget: max possible while leaving at least 1024 tokens for visible output.
  if (thinkingEnabled) params["thinking"] = { type: "enabled", budget_tokens: getThinkingBudget(maxTokens) };
  if (anthropicTools) params["tools"] = anthropicTools;
  if (anthropicToolChoice) params["tool_choice"] = anthropicToolChoice;
  if (body["cache_control"]) params["cache_control"] = body["cache_control"];

  const ka = startNonStreamKeepalive(res);
  try {
    const response = await anthropic.messages.create(params as Anthropic.MessageCreateParamsNonStreaming);

    // Collect blocks
    let thinkingText = "";
    let bodyText = "";
    const toolCallResults: Array<{ id: string; name: string; input: unknown }> = [];

    for (const block of response.content) {
      if (block.type === "thinking") {
        thinkingText += (block as { thinking?: string }).thinking ?? "";
      } else if (block.type === "text") {
        bodyText += block.text;
      } else if (block.type === "tool_use") {
        toolCallResults.push({ id: block.id, name: block.name, input: block.input });
      }
    }

    const stopReason = response.stop_reason;
    const finishReason =
      stopReason === "tool_use" ? "tool_calls"
      : stopReason === "end_turn" ? "stop"
      : (stopReason ?? "stop");

    // Compose message
    let fullContent: string | null = bodyText || null;
    if (thinkingText && thinkingVisible) {
      fullContent = `<thinking>${thinkingText}</thinking>\n\n${bodyText}`;
    }

    const id = `chatcmpl-${Date.now()}`;

    const assistantMessage: Record<string, unknown> = {
      role: "assistant",
      content: fullContent,
    };

    if (toolCallResults.length > 0) {
      assistantMessage["tool_calls"] = toolCallResults.map((tc) => ({
        id: tc.id,
        type: "function",
        function: {
          name: tc.name,
          arguments: JSON.stringify(tc.input),
        },
      }));
    }

    endNonStream(res, {
      id,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [{
        index: 0,
        message: assistantMessage,
        finish_reason: finishReason,
      }],
      usage: (() => {
        const u = response.usage as Record<string, unknown>;
        const cacheRead = u["cache_read_input_tokens"] as number ?? 0;
        const cacheCreate = u["cache_creation_input_tokens"] as number ?? 0;
        const base: Record<string, unknown> = {
          prompt_tokens: response.usage.input_tokens,
          completion_tokens: response.usage.output_tokens,
          total_tokens: response.usage.input_tokens + response.usage.output_tokens,
        };
        if (cacheRead > 0 || cacheCreate > 0) {
          base["cache_read_input_tokens"] = cacheRead;
          base["cache_creation_input_tokens"] = cacheCreate;
          base["prompt_tokens_details"] = { cached_tokens: cacheRead };
        }
        return base;
      })(),
    }, cacheKey ? { key: cacheKey, model } : undefined);
  } catch (err: unknown) {
    const { status, message } = extractUpstreamError(err);
    endNonStreamError(res, status, message, "upstream_error");
  } finally {
    runtimeTimers.clearInterval(ka);
  }
}

// ----------------------------------------------------------------------
// Gemini - streaming
// ----------------------------------------------------------------------

async function handleGeminiStream(
  _req: Request,
  res: Response,
  body: ChatBody
) {
  const { model, max_tokens, temperature, top_p } = body;
  const { baseModel, thinkingEnabled } = stripGeminiSuffix(model);
  const { systemInstruction, contents } = convertMessagesToGemini(body.messages);

  setSseHeaders(res);
  res.write(": init\n\n");

  const id = `chatcmpl-${Date.now()}`;

  const keepaliveInterval = runtimeTimers.setInterval(() => {
    sseKeepalive(res, model);
  }, SSE_KEEPALIVE_MS);

  try {
    const config: Record<string, unknown> = {
      maxOutputTokens: max_tokens ?? 65536,
    };
    if (temperature !== undefined) config["temperature"] = temperature;
    if (top_p !== undefined) config["topP"] = top_p;
    if (systemInstruction) config["systemInstruction"] = systemInstruction;
    if (thinkingEnabled) config["thinkingConfig"] = { thinkingBudget: -1 };

    sseWrite(res, makeChunk(id, model, { role: "assistant", content: "" }));

    const stream = await gemini.models.generateContentStream({
      model: baseModel,
      contents,
      config: config as Parameters<typeof gemini.models.generateContentStream>[0]["config"],
    });

    let inputTokens = 0;
    let outputTokens = 0;

    for await (const chunk of stream) {
      // chunk.text is a getter that can throw on safety blocks or malformed candidates
      let text: string | undefined;
      try { text = chunk.text ?? undefined; } catch { /* safety/error block — skip text */ }
      if (text) {
        sseWrite(res, makeChunk(id, model, { content: text }));
      }
      if (chunk.usageMetadata) {
        inputTokens = chunk.usageMetadata.promptTokenCount ?? inputTokens;
        outputTokens = chunk.usageMetadata.candidatesTokenCount ?? outputTokens;
      }
    }

    sseWrite(res, makeChunk(id, model, {}, "stop"));
    sseWrite(res, {
      id, object: "chat.completion.chunk",
      created: Math.floor(Date.now() / 1000), model,
      choices: [],
      usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
    });
    if (!res.writableEnded) res.write("data: [DONE]\n\n");
  } catch (streamErr) {
    try {
      sseWrite(res, {
        error: {
          message: streamErr instanceof Error ? streamErr.message : "Stream error",
          type: "stream_error",
        },
      });
      if (!res.writableEnded) res.write("data: [DONE]\n\n");
    } catch { /* ignore */ }
  } finally {
    runtimeTimers.clearInterval(keepaliveInterval);
    if (!res.writableEnded) res.end();
  }
}

// ----------------------------------------------------------------------
// Gemini - non-streaming
// ----------------------------------------------------------------------

async function handleGeminiNonStream(
  _req: Request,
  res: Response,
  body: ChatBody,
  cacheKey?: string,
) {
  const { model, max_tokens, temperature, top_p } = body;
  const { baseModel, thinkingEnabled } = stripGeminiSuffix(model);
  const { systemInstruction, contents } = convertMessagesToGemini(body.messages);

  const config: Record<string, unknown> = {
    maxOutputTokens: max_tokens ?? 65536,
  };
  if (temperature !== undefined) config["temperature"] = temperature;
  if (top_p !== undefined) config["topP"] = top_p;
  if (systemInstruction) config["systemInstruction"] = systemInstruction;
  if (thinkingEnabled) config["thinkingConfig"] = { thinkingBudget: -1 };

  const ka = startNonStreamKeepalive(res);
  try {
    const response = await gemini.models.generateContent({
      model: baseModel,
      contents,
      config: config as Parameters<typeof gemini.models.generateContent>[0]["config"],
    });

    const text = response.text ?? "";
    const inputTokens = response.usageMetadata?.promptTokenCount ?? 0;
    const outputTokens = response.usageMetadata?.candidatesTokenCount ?? 0;
    const id = `chatcmpl-${Date.now()}`;

    endNonStream(res, {
      id,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [{
        index: 0,
        message: { role: "assistant", content: text },
        finish_reason: "stop",
      }],
      usage: {
        prompt_tokens: inputTokens,
        completion_tokens: outputTokens,
        total_tokens: inputTokens + outputTokens,
      },
    }, cacheKey ? { key: cacheKey, model } : undefined);
  } catch (err: unknown) {
    const { status, message } = extractUpstreamError(err);
    endNonStreamError(res, status, message, "upstream_error");
  } finally {
    runtimeTimers.clearInterval(ka);
  }
}

// ----------------------------------------------------------------------
// OpenRouter - streaming (OpenAI-compatible, uses openrouter client)
// ----------------------------------------------------------------------

async function handleOpenRouterStream(
  req: Request,
  res: Response,
  body: ChatBody
) {
  // Establish SSE connection immediately before image-URL resolution.
  setSseHeaders(res);
  res.write(": init\n\n");

  const resolvedMessages = await resolveImageUrls(body.messages);
  const isOpenRouterAnthropic = isOpenRouterAnthropicModel(body.model);
  const cachingPrep = isOpenRouterAnthropic
    ? prepareOpenRouterAnthropicCachingRequest(resolvedMessages)
    : { messages: resolvedMessages, strategy: "not_applicable" as const };

  if (cachingPrep.strategy === "block") {
    req.log.info({ model: body.model }, "openrouter anthropic cache strategy=block");
  } else if (cachingPrep.strategy === "pass_through_existing_block") {
    req.log.info({ model: body.model }, "openrouter anthropic cache strategy=pass_through_existing_block");
  }

  // Strip proxy-internal fields, then spread the rest so ALL caller-supplied
  // OpenRouter-specific parameters (provider, transforms, route, cache_control,
  // extra_headers, etc.) are forwarded transparently to the OpenRouter API.
  const { model: _m, messages: _msgs, stream: _s, ...passThrough } = body;

  // For known reasoning models, inject the appropriate reasoning default.
  // For Claude Opus 4.6+, also inject verbosity: "max".
  // Callers can always override by including these keys in their request body.
  const pt = sanitizeOpenRouterReasoning(passThrough as Record<string, unknown>);
  const reasoningDefault = getOpenRouterReasoningDefault(body.model, pt);
  const verbosityDefault = getOpenRouterVerbosityDefault(body.model, pt);

  const params = {
    ...reasoningDefault,
    ...verbosityDefault,
    ...pt,
    model: body.model,
    messages: cachingPrep.messages as OpenAI.ChatCompletionMessageParam[],
    stream: true as const,
    stream_options: { include_usage: true },
  } as OpenAI.Chat.ChatCompletionCreateParamsStreaming;

  const keepaliveInterval = runtimeTimers.setInterval(() => {
    sseKeepalive(res, body.model);
  }, SSE_KEEPALIVE_MS);

  try {
    const stream = await openrouter.chat.completions.create(params);
    let finalUsageChunk: unknown;
    for await (const chunk of stream) {
      if (
        isOpenRouterAnthropic &&
        chunk !== null &&
        typeof chunk === "object" &&
        "usage" in (chunk as object)
      ) {
        finalUsageChunk = chunk;
      }
      sseWrite(res, chunk);
    }

    if (isOpenRouterAnthropic) {
      if (finalUsageChunk !== undefined) {
        logOpenRouterPromptCacheMetrics(
          req,
          "stream",
          extractOpenRouterPromptCacheMetrics(finalUsageChunk, body.model, cachingPrep.strategy),
        );
      } else {
        logOpenRouterPromptCacheMetrics(
          req,
          "stream",
          extractOpenRouterPromptCacheMetrics(null, body.model, cachingPrep.strategy),
        );
      }
    }

    if (!res.writableEnded) res.write("data: [DONE]\n\n");
  } catch (streamErr) {
    try {
      sseWrite(res, {
        error: {
          message: streamErr instanceof Error ? streamErr.message : "Stream error",
          type: "stream_error",
        },
      });
      if (!res.writableEnded) res.write("data: [DONE]\n\n");
    } catch { /* ignore */ }
  } finally {
    runtimeTimers.clearInterval(keepaliveInterval);
    if (!res.writableEnded) res.end();
  }
}

// ----------------------------------------------------------------------
// OpenRouter - non-streaming
// ----------------------------------------------------------------------

async function handleOpenRouterNonStream(
  req: Request,
  res: Response,
  body: ChatBody,
  cacheKey?: string,
) {
  const resolvedMessages = await resolveImageUrls(body.messages);
  const isOpenRouterAnthropic = isOpenRouterAnthropicModel(body.model);
  const cachingPrep = isOpenRouterAnthropic
    ? prepareOpenRouterAnthropicCachingRequest(resolvedMessages)
    : { messages: resolvedMessages, strategy: "not_applicable" as const };

  if (cachingPrep.strategy === "block") {
    req.log.info({ model: body.model }, "openrouter anthropic cache strategy=block");
  } else if (cachingPrep.strategy === "pass_through_existing_block") {
    req.log.info({ model: body.model }, "openrouter anthropic cache strategy=pass_through_existing_block");
  }

  // Spread all caller-supplied fields so OpenRouter-specific parameters
  // (provider, transforms, route, cache_control, etc.) pass through untouched.
  const { model: _m, messages: _msgs, stream: _s, ...passThrough } = body;

  // For known reasoning models, inject the appropriate reasoning default.
  // For Claude Opus 4.6+, also inject verbosity: "max".
  // Callers can always override by including these keys in their request body.
  const pt = sanitizeOpenRouterReasoning(passThrough as Record<string, unknown>);
  const reasoningDefault = getOpenRouterReasoningDefault(body.model, pt);
  const verbosityDefault = getOpenRouterVerbosityDefault(body.model, pt);

  const params = {
    ...reasoningDefault,
    ...verbosityDefault,
    ...pt,
    model: body.model,
    messages: cachingPrep.messages as OpenAI.ChatCompletionMessageParam[],
    stream: false as const,
  } as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming;

  const ka = startNonStreamKeepalive(res);
  try {
    const response = await openrouter.chat.completions.create(params);

    if (isOpenRouterAnthropic) {
      logOpenRouterPromptCacheMetrics(
        req,
        "non_stream",
        extractOpenRouterPromptCacheMetrics(response, body.model, cachingPrep.strategy),
      );
    }

    endNonStream(res, response, cacheKey ? { key: cacheKey, model: body.model } : undefined);
  } catch (err: unknown) {
    const { status, message } = extractUpstreamError(err);
    endNonStreamError(res, status, message, "upstream_error");
  } finally {
    runtimeTimers.clearInterval(ka);
  }
}

// ----------------------------------------------------------------------
// OpenAI - streaming
// ----------------------------------------------------------------------

async function handleOpenAIStream(
  _req: Request,
  res: Response,
  body: ChatBody
) {
  setSseHeaders(res);
  res.write(": init\n\n"); // flush connection immediately -- prevents proxy timeout before first AI token

  const resolvedMessages = await resolveImageUrls(body.messages);

  // Pass through all OpenAI-compatible params
  const params: OpenAI.Chat.ChatCompletionCreateParamsStreaming = {
    model: body.model,
    messages: resolvedMessages as OpenAI.ChatCompletionMessageParam[],
    stream: true,
    stream_options: { include_usage: true },
  };

  if (body.temperature !== undefined) params.temperature = body.temperature;
  if (body.top_p !== undefined) params.top_p = body.top_p;
  if (body.max_tokens !== undefined) params.max_tokens = body.max_tokens;
  if (body.stop !== undefined) params.stop = body.stop as string | string[];
  if (body.seed !== undefined) params.seed = body.seed;
  if (body.presence_penalty !== undefined) params.presence_penalty = body.presence_penalty;
  if (body.frequency_penalty !== undefined) params.frequency_penalty = body.frequency_penalty;
  if (body.n !== undefined) params.n = body.n;
  if (body.user !== undefined) params.user = body.user;
  if (body.response_format !== undefined) params.response_format = body.response_format as OpenAI.ResponseFormatText;
  if (body.logprobs !== undefined) params.logprobs = body.logprobs;
  if (body.top_logprobs !== undefined) params.top_logprobs = body.top_logprobs;
  if (body.tools && body.tools.length > 0) {
    params.tools = body.tools as OpenAI.ChatCompletionTool[];
    if (body.tool_choice !== undefined && body.tool_choice !== "none") {
      params.tool_choice = body.tool_choice as OpenAI.ChatCompletionToolChoiceOption;
    }
    if (body.parallel_tool_calls !== undefined) {
      params.parallel_tool_calls = body.parallel_tool_calls;
    }
  }

  const keepaliveInterval = runtimeTimers.setInterval(() => {
    sseKeepalive(res, body.model);
  }, SSE_KEEPALIVE_MS);

  try {
    const stream = await openai.chat.completions.create(params);
    for await (const chunk of stream) {
      sseWrite(res, chunk);
    }
    if (!res.writableEnded) res.write("data: [DONE]\n\n");
  } catch (streamErr) {
    try {
      sseWrite(res, {
        error: {
          message: streamErr instanceof Error ? streamErr.message : "Stream error",
          type: "stream_error",
        },
      });
      if (!res.writableEnded) res.write("data: [DONE]\n\n");
    } catch { /* ignore write errors during cleanup */ }
  } finally {
    runtimeTimers.clearInterval(keepaliveInterval);
    if (!res.writableEnded) res.end();
  }
}

// ----------------------------------------------------------------------
// OpenAI - non-streaming
// ----------------------------------------------------------------------

async function handleOpenAINonStream(
  _req: Request,
  res: Response,
  body: ChatBody,
  cacheKey?: string,
) {
  const resolvedMessages = await resolveImageUrls(body.messages);

  const params: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming = {
    model: body.model,
    messages: resolvedMessages as OpenAI.ChatCompletionMessageParam[],
    stream: false,
  };

  if (body.temperature !== undefined) params.temperature = body.temperature;
  if (body.top_p !== undefined) params.top_p = body.top_p;
  if (body.max_tokens !== undefined) params.max_tokens = body.max_tokens;
  if (body.stop !== undefined) params.stop = body.stop as string | string[];
  if (body.seed !== undefined) params.seed = body.seed;
  if (body.presence_penalty !== undefined) params.presence_penalty = body.presence_penalty;
  if (body.frequency_penalty !== undefined) params.frequency_penalty = body.frequency_penalty;
  if (body.n !== undefined) params.n = body.n;
  if (body.user !== undefined) params.user = body.user;
  if (body.response_format !== undefined) params.response_format = body.response_format as OpenAI.ResponseFormatText;
  if (body.logprobs !== undefined) params.logprobs = body.logprobs;
  if (body.top_logprobs !== undefined) params.top_logprobs = body.top_logprobs;
  if (body.tools && body.tools.length > 0) {
    params.tools = body.tools as OpenAI.ChatCompletionTool[];
    if (body.tool_choice !== undefined && body.tool_choice !== "none") {
      params.tool_choice = body.tool_choice as OpenAI.ChatCompletionToolChoiceOption;
    }
    if (body.parallel_tool_calls !== undefined) {
      params.parallel_tool_calls = body.parallel_tool_calls;
    }
  }

  const ka = startNonStreamKeepalive(res);
  try {
    const response = await openai.chat.completions.create(params);
    endNonStream(res, response, cacheKey ? { key: cacheKey, model: body.model } : undefined);
  } catch (err: unknown) {
    const { status, message } = extractUpstreamError(err);
    endNonStreamError(res, status, message, "upstream_error");
  } finally {
    runtimeTimers.clearInterval(ka);
  }
}

// ----------------------------------------------------------------------
// Dynamic Sinking & LCP Engine — Prompt Caching Optimisation
// ----------------------------------------------------------------------

type SystemLayerTier = "stable" | "low" | "volatile";

interface SystemLayerRule {
  label: string;
  tier: Exclude<SystemLayerTier, "stable">;
  pattern: RegExp;
}

interface SystemLayerMatch {
  start: number;
  end: number;
  text: string;
  label: string;
  tier: Exclude<SystemLayerTier, "stable">;
}

interface SystemLayerChunk {
  tier: SystemLayerTier;
  label: string;
  start: number;
  end: number;
  text: string;
}

interface LayeredSystemAnalysis {
  original: string;
  comparableSystem: string;
  systemWithoutVolatile: string;
  stableText: string;
  lowFrequencyText: string;
  volatileText: string;
  chunks: SystemLayerChunk[];
  stableLength: number;
  lowFrequencyLength: number;
  volatileLength: number;
  totalLength: number;
  lowFrequencyLabels: string[];
  volatileLabels: string[];
}

interface LCPSplitResult {
  stable: string;
  dynamic: string;
  lcpLength: number;
  divergeIndex: number;
  divergenceSource: string;
}

interface LayeredSinkingResult {
  system: string;
  messages: OAIMessage[];
  sunk: boolean;
  analysis: LayeredSystemAnalysis;
}

const SYSTEM_LAYER_RULES: SystemLayerRule[] = [
  {
    label: "rag_block",
    tier: "volatile",
    pattern: /<!-- VCP_RAG_BLOCK_START[\s\S]*?VCP_RAG_BLOCK_END -->/g,
  },
  {
    label: "memory_block",
    tier: "volatile",
    pattern: /(?:^|\n)————记忆区————\n[\s\S]*?\n————以上是过往记忆区————/g,
  },
  {
    label: "date_weather_context",
    tier: "volatile",
    pattern: /(?:^|\n)今天是20\d{2}\/[^\n]*/g,
  },
  {
    label: "weather_payload",
    tier: "volatile",
    pattern: /(?:^|\n)当前天气是\{\{[\s\S]*?\}\}[。.]?/g,
  },
  {
    label: "system_info_line",
    tier: "volatile",
    pattern: /(?:^|\n)系统信息是[^\n]+/g,
  },
  {
    label: "current_runtime_meta",
    tier: "volatile",
    pattern: /(?:^|\n)# Current (?:Time|Cost)\n(?:[^\n]*\n)*?(?=(?:# [^\n]+)|$)/g,
  },
  {
    label: "expanded_time_runtime",
    tier: "volatile",
    pattern: /\{\{(?:Date|Time|Today|Festival)\}\}/g,
  },
  {
    label: "async_result",
    tier: "volatile",
    pattern: /\{\{VCP_ASYNC_RESULT::[\s\S]*?\}\}/g,
  },
  {
    label: "expanded_var_tar",
    tier: "low",
    pattern: /\{\{(?:Var|Tar)[^}\r\n]{20,}\}\}/g,
  },
  {
    label: "meta_thinking_block",
    tier: "low",
    pattern: /(?:^|\n)————【VCP元思考】————\n[\s\S]*?\n————【VCP元思考】加载结束—————/g,
  },
  {
    label: "timeline_block",
    tier: "low",
    pattern: /(?:^|\n)————日记时间线————\n[\s\S]*?(?=\n(?:————记忆区————|Nova的个人记忆二合一:))/g,
  },
  {
    label: "toolbox_section",
    tier: "low",
    pattern: /(?:^|\n)# VCP [^\n]*工具箱能力收纳\n[\s\S]*?(?=\n(?:---\n\n)?# VCP [^\n]*工具箱能力收纳|\n—— 日记 \(DailyNote\) ——|\n额外指令:|\n————表情包系统————|\n====|$)/g,
  },
  {
    label: "rendering_guide_block",
    tier: "low",
    pattern: /(?:^|\n)额外指令:当前Vchat客户端支持高级流式输出渲染器[\s\S]*?(?=\n(?:日记编辑工具：|————表情包系统————|====)|$)/g,
  },
  {
    label: "dailynote_guide_block",
    tier: "low",
    pattern: /(?:^|\n)—— 日记 \(DailyNote\) ——\n[\s\S]*?(?=\n(?:额外指令:|————表情包系统————|====)|$)/g,
  },
  {
    label: "emoji_catalog_block",
    tier: "low",
    pattern: /(?:^|\n)————表情包系统————\n[\s\S]*?(?=\n(?:可选音乐列表：|\(VCP Agent\)|====)|$)/g,
  },
  {
    label: "toolbox_hint",
    tier: "low",
    pattern: /(?:^|\n)\*\(提示：当前上下文中还隐藏收纳了另外 \d+ 个工具模块分组，您可以通过明确提问或强调相关语境来获得展开。\)\*/g,
  },
];

function collectSystemLayerMatches(system: string): SystemLayerMatch[] {
  const matches: SystemLayerMatch[] = [];
  for (const rule of SYSTEM_LAYER_RULES) {
    const regex = new RegExp(rule.pattern.source, rule.pattern.flags);
    let match: RegExpExecArray | null;
    while ((match = regex.exec(system)) !== null) {
      const text = match[0] ?? "";
      if (!text) {
        if (regex.lastIndex === match.index) regex.lastIndex++;
        continue;
      }
      matches.push({
        start: match.index,
        end: match.index + text.length,
        text,
        label: rule.label,
        tier: rule.tier,
      });
      if (regex.lastIndex === match.index) regex.lastIndex++;
    }
  }
  matches.sort((a, b) => a.start - b.start || b.end - a.end);
  const accepted: SystemLayerMatch[] = [];
  let cursor = -1;
  for (const match of matches) {
    if (match.start < cursor) continue;
    accepted.push(match);
    cursor = match.end;
  }
  return accepted;
}

function analyzeSystemLayers(system: string): LayeredSystemAnalysis {
  const matches = collectSystemLayerMatches(system);
  const chunks: SystemLayerChunk[] = [];
  const stableParts: string[] = [];
  const lowParts: string[] = [];
  const volatileParts: string[] = [];
  const keptSystemParts: string[] = [];
  const lowFrequencyLabels: string[] = [];
  const volatileLabels: string[] = [];

  let cursor = 0;
  for (const match of matches) {
    if (match.start > cursor) {
      const text = system.slice(cursor, match.start);
      chunks.push({ tier: "stable", label: "stable_text", start: cursor, end: match.start, text });
      stableParts.push(text);
      keptSystemParts.push(text);
    }
    chunks.push({ tier: match.tier, label: match.label, start: match.start, end: match.end, text: match.text });
    if (match.tier === "low") {
      lowParts.push(match.text);
      keptSystemParts.push(match.text);
      if (!lowFrequencyLabels.includes(match.label)) lowFrequencyLabels.push(match.label);
    } else {
      volatileParts.push(match.text);
      if (!volatileLabels.includes(match.label)) volatileLabels.push(match.label);
    }
    cursor = match.end;
  }
  if (cursor < system.length) {
    const text = system.slice(cursor);
    chunks.push({ tier: "stable", label: "stable_text", start: cursor, end: system.length, text });
    stableParts.push(text);
    keptSystemParts.push(text);
  }

  return {
    original: system,
    comparableSystem: keptSystemParts.join("").trim(),
    systemWithoutVolatile: keptSystemParts.join("").trim(),
    stableText: stableParts.join(""),
    lowFrequencyText: lowParts.join(""),
    volatileText: volatileParts.join(""),
    chunks,
    stableLength: stableParts.join("").length,
    lowFrequencyLength: lowParts.join("").length,
    volatileLength: volatileParts.join("").length,
    totalLength: system.length,
    lowFrequencyLabels,
    volatileLabels,
  };
}

/**
 * Extracts dynamic content from system text and appends it to the last user message.
 */
function applyDynamicSinking(system: string, messages: OAIMessage[]): LayeredSinkingResult {
  const analysis = analyzeSystemLayers(system);
  if (!analysis.volatileText.trim()) {
    return { system, messages, sunk: false, analysis };
  }

  const newMsgs = [...messages];
  let lastUserIdx = -1;
  for (let i = newMsgs.length - 1; i >= 0; i--) {
    if (newMsgs[i].role === "user") { lastUserIdx = i; break; }
  }

  if (lastUserIdx === -1) return { system, messages, sunk: false, analysis };

  const sunkContent = analysis.volatileText.trim();
  const lastMsg = { ...newMsgs[lastUserIdx] };
  if (typeof lastMsg.content === "string") {
    lastMsg.content = (lastMsg.content + "\n\n" + sunkContent).trim();
  } else if (Array.isArray(lastMsg.content)) {
    lastMsg.content = [...lastMsg.content, { type: "text", text: sunkContent } as OAIContentPart];
  } else {
    return { system, messages, sunk: false, analysis };
  }

  newMsgs[lastUserIdx] = lastMsg;
  return { system: analysis.systemWithoutVolatile, messages: newMsgs, sunk: true, analysis };
}

const _systemStabilityCache = new Map<string, string>();
const _prevSystemTextCache = new Map<string, string>();

function checkSystemStability(key: string, text: string): boolean {
  const hash = String(text.length) + ":" + String(text.split("").reduce((acc, ch) => ((acc * 31) + ch.charCodeAt(0)) >>> 0, 0));
  const prev = _systemStabilityCache.get(key);
  _systemStabilityCache.set(key, hash);
  return !!prev && prev === hash;
}

function computeLCPSplit(key: string, currentText: string): LCPSplitResult | null {
  const prev = _prevSystemTextCache.get(key);
  _prevSystemTextCache.set(key, currentText);
  if (!prev || prev === currentText) return null;
  const minLen = Math.min(prev.length, currentText.length);
  let divergeIdx = 0;
  while (divergeIdx < minLen && prev.charCodeAt(divergeIdx) === currentText.charCodeAt(divergeIdx)) divergeIdx++;
  let boundary = currentText.lastIndexOf('\n', divergeIdx);
  if (boundary <= 0) return null;
  boundary += 1;
  if (boundary < 4000) return null;
  const stable = currentText.slice(0, boundary);
  const dynamic = currentText.slice(boundary);
  if (!dynamic.trim()) return null;
  return { stable, dynamic, lcpLength: stable.length, divergeIndex: divergeIdx, divergenceSource: "system_text" };
}

interface HistoryCacheProbeResult {
  mode: "string" | "array" | "none";
  blockIndex: number;
  cacheable: boolean;
  alreadyCached: boolean;
}

function probeHistoryCacheAnchor(content: unknown): HistoryCacheProbeResult {
  if (typeof content === "string") {
    return { mode: "string", blockIndex: 0, cacheable: content.length > 0, alreadyCached: false };
  }
  if (!Array.isArray(content) || content.length === 0) {
    return { mode: "none", blockIndex: -1, cacheable: false, alreadyCached: false };
  }
  for (let i = content.length - 1; i >= 0; i--) {
    const block = content[i] as Record<string, unknown> | null | undefined;
    if (!block || typeof block !== "object") continue;
    const type = typeof block.type === "string" ? block.type : "";
    const cacheable = type === "text" || type === "tool_result";
    if (!cacheable) continue;
    return { mode: "array", blockIndex: i, cacheable: true, alreadyCached: !!block.cache_control };
  }
  return { mode: "array", blockIndex: -1, cacheable: false, alreadyCached: false };
}

function injectHistoryBreakpoint(msgs: OAIMessage[]): { messages: OAIMessage[], applied: boolean } {
  if (!msgs || msgs.length < 3) return { messages: msgs, applied: false };
  let lastUserIdx = -1;
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === "user") { lastUserIdx = i; break; }
  }
  if (lastUserIdx <= 0) return { messages: msgs, applied: false };

  let anchorUserIdx = -1;
  let anchorProbe: HistoryCacheProbeResult = { mode: "none", blockIndex: -1, cacheable: false, alreadyCached: false };

  for (let i = lastUserIdx - 1; i >= 0; i--) {
    if (msgs[i].role !== "user") continue;
    const probe = probeHistoryCacheAnchor(msgs[i].content);
    if (!probe.cacheable && !probe.alreadyCached) continue;
    anchorUserIdx = i;
    anchorProbe = probe;
    break;
  }

  if (anchorUserIdx < 0 || anchorProbe.alreadyCached) return { messages: msgs, applied: anchorProbe.alreadyCached };

  const msg = msgs[anchorUserIdx];
  const newMsgs = [...msgs];
  const cc = { type: "ephemeral" as const };

  if (anchorProbe.mode === "string" && typeof msg.content === "string") {
    newMsgs[anchorUserIdx] = { ...msg, content: [{ type: "text", text: msg.content, cache_control: cc } as OAIContentPart] };
    return { messages: newMsgs, applied: true };
  }
  if (anchorProbe.mode === "array" && Array.isArray(msg.content)) {
    const content = [...msg.content];
    content[anchorProbe.blockIndex] = { ...content[anchorProbe.blockIndex], cache_control: cc };
    newMsgs[anchorUserIdx] = { ...msg, content };
    return { messages: newMsgs, applied: true };
  }
  return { messages: msgs, applied: false };
}

// ----------------------------------------------------------------------
// Layered cache helpers for OpenAI-compatible routes
// ----------------------------------------------------------------------

function flattenSystemTextContent(content: string | OAIContentPart[] | null | undefined): string {
  if (!content) return "";
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((part) => (part as { type?: string }).type === "text")
    .map((part) => ((part as { text?: string }).text ?? ""))
    .join("");
}

function hasAnyOpenAIBlockCacheControl(messages: OAIMessage[]): boolean {
  return messages.some((message) =>
    Array.isArray(message.content) &&
    message.content.some((part) => !!(part as Record<string, unknown>)?.cache_control)
  );
}

// ----------------------------------------------------------------------
// Prompt-based tool calling fallback
// Triggered by `"x_use_prompt_tools": true` in the request body.
// Works for any model/route. Injects a structured system prompt with the
// tool schema, calls the model without native tool_calls, then parses the
// JSON response and returns it in the standard OpenAI tool_calls format.
// ----------------------------------------------------------------------

async function handlePromptTools(
  _req: Request,
  res: Response,
  originalBody: ChatBody,
): Promise<void> {
  const { model, messages, tools, stream } = originalBody;

  const toolInstruction = buildPromptToolsInstruction((tools ?? []) as PromptTool[]);

  // Merge tool instruction into the existing system message (or create one)
  const sysMsg = messages.find((m) => m.role === "system");
  const existingSystem =
    sysMsg
      ? typeof sysMsg.content === "string"
        ? sysMsg.content
        : Array.isArray(sysMsg.content)
          ? (sysMsg.content as OAIContentPart[])
              .filter((p) => (p as { type: string }).type === "text")
              .map((p) => (p as { text: string }).text)
              .join("\n")
          : ""
      : "";
  const augmentedSystem = existingSystem
    ? `${existingSystem}\n\n${toolInstruction}`
    : toolInstruction;

  // Build message list: replace system, keep everything else
  const augmentedMessages: OAIMessage[] = [
    { role: "system", content: augmentedSystem },
    ...messages.filter((m) => m.role !== "system"),
  ];

  // Call the upstream model without tools (non-streaming internally)
  let responseText = "";
  let promptTokens = 0;
  let completionTokens = 0;

  const isClaude = model.startsWith("claude-");
  const isGemini = model.startsWith("gemini-");
  const isOpenRouterModel = !isClaude && !isGemini && model.includes("/");

  // B1 fix: when the caller expects SSE, open the connection immediately BEFORE the upstream call.
  // Without this, the Replit proxy can timeout (300s) on long-running model calls because
  // no bytes flow to the client until after the upstream finishes.
  //
  // Keepalive: handlePromptTools calls the upstream in NON-streaming mode and waits for the
  // full response before writing SSE chunks. To prevent the reverse proxy from cutting the
  // idle connection during that wait, send an SSE comment every 5 s — proxies (including
  // Replit's) treat any byte as activity and reset their idle timer.
  let promptToolsKeepalive: number | undefined;
  if (stream) {
    setSseHeaders(res);
    res.write(": init\n\n");
    promptToolsKeepalive = runtimeTimers.setInterval(() => {
      sseKeepalive(res, model);
    }, SSE_KEEPALIVE_MS);
  } else {
    // Non-streaming: send whitespace keepalive so Replit's 300s proxy timeout
    // doesn't cut long Anthropic / Gemini calls before the response is ready.
    promptToolsKeepalive = startNonStreamKeepalive(res);
  }

  try {
    if (isClaude) {
      const { baseModel, thinkingEnabled } = stripClaudeSuffix(model);
      const modelMax = getClaudeMaxTokens(baseModel);
      const maxTokens = Math.min(
        originalBody.max_tokens && originalBody.max_tokens > 0 ? originalBody.max_tokens : modelMax,
        modelMax,
      );
      const { system, messages: anthropicMessages } = convertMessagesToAnthropic(augmentedMessages);
      const p: Record<string, unknown> = {
        model: baseModel,
        max_tokens: maxTokens,
        messages: anthropicMessages,
        stream: false,
      };
      if (system) p["system"] = system;
      if (thinkingEnabled) p["thinking"] = { type: "enabled", budget_tokens: getThinkingBudget(maxTokens) };
      const resp = await anthropic.messages.create(p as Anthropic.MessageCreateParamsNonStreaming);
      responseText = resp.content
        .filter((b: Anthropic.ContentBlock) => b.type === "text")
        .map((b: Anthropic.ContentBlock) => (b as { text: string }).text)
        .join("");
      promptTokens = resp.usage.input_tokens;
      completionTokens = resp.usage.output_tokens;

    } else if (isGemini) {
      const { baseModel, thinkingEnabled } = stripGeminiSuffix(model);
      const { systemInstruction, contents } = convertMessagesToGemini(augmentedMessages);
      const config: Record<string, unknown> = { maxOutputTokens: originalBody.max_tokens ?? 65536 };
      if (systemInstruction) config["systemInstruction"] = systemInstruction;
      if (thinkingEnabled) config["thinkingConfig"] = { thinkingBudget: -1 };
      const resp = await gemini.models.generateContent({
        model: baseModel,
        contents,
        config: config as Parameters<typeof gemini.models.generateContent>[0]["config"],
      });
      responseText = resp.text ?? "";
      promptTokens = resp.usageMetadata?.promptTokenCount ?? 0;
      completionTokens = resp.usageMetadata?.candidatesTokenCount ?? 0;

    } else {
      const client = isOpenRouterModel ? openrouter : openai;
      const resp = await client.chat.completions.create({
        model,
        messages: augmentedMessages as OpenAI.ChatCompletionMessageParam[],
        stream: false,
        ...(originalBody.max_tokens !== undefined && { max_tokens: originalBody.max_tokens }),
        ...(originalBody.temperature !== undefined && { temperature: originalBody.temperature }),
        ...(originalBody.top_p !== undefined && { top_p: originalBody.top_p }),
      });
      responseText = resp.choices[0]?.message?.content ?? "";
      promptTokens = resp.usage?.prompt_tokens ?? 0;
      completionTokens = resp.usage?.completion_tokens ?? 0;
    }
  } catch (err: unknown) {
    // Stop keepalive before replying — no more writes needed after this.
    runtimeTimers.clearInterval(promptToolsKeepalive);
    const { status, message } = extractUpstreamError(err);
    if (stream) {
      // Client expected SSE — emit the error as an SSE event so they don't hang.
      if (!res.headersSent) setSseHeaders(res);
      sseWrite(res, { error: { message, type: "upstream_error" } });
      if (!res.writableEnded) { res.write("data: [DONE]\n\n"); res.end(); }
    } else {
      if (!res.headersSent) res.status(status);
      res.setHeader("Content-Type", "application/json");
      if (!res.writableEnded) res.end(JSON.stringify({ error: { message, type: "upstream_error" } }));
    }
    return;
  }

  // Upstream call finished — keepalive is no longer needed.
  runtimeTimers.clearInterval(promptToolsKeepalive);

  const parsed = parsePromptToolsResponse(responseText);
  const completion = buildCompletionFromPromptTools(parsed, model, {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
  });

  if (stream) {
    const id = completion["id"] as string;
    const created = completion["created"] as number;
    const choice = (completion["choices"] as Array<Record<string, unknown>>)[0];
    const finishReason = choice["finish_reason"] as string;

    // Headers already set + flushed (": init\n\n") before the upstream call — do not call setSseHeaders again.

    if (parsed.isToolCall && parsed.calls && parsed.calls.length > 0) {
      // Role chunk
      sseWrite(res, makeChunk(id, model, { role: "assistant", content: null }));
      // Tool call chunks
      for (const [i, call] of parsed.calls.entries()) {
        sseWrite(res, makeChunk(id, model, {
          tool_calls: [{ index: i, id: call.id, type: "function", function: { name: call.name, arguments: "" } }],
        }));
        sseWrite(res, makeChunk(id, model, {
          tool_calls: [{ index: i, function: { arguments: call.arguments } }],
        }));
      }
    } else {
      sseWrite(res, makeChunk(id, model, { role: "assistant", content: "" }));
      sseWrite(res, makeChunk(id, model, { content: parsed.content }));
    }

    // Final stop chunk + usage
    sseWrite(res, {
      ...makeChunk(id, model, {}, finishReason),
      usage: completion["usage"],
    });
    if (!res.writableEnded) {
      res.write("data: [DONE]\n\n");
      res.end();
    }
  } else {
    res.setHeader("Content-Type", "application/json");
    if (!res.writableEnded) res.end(JSON.stringify(completion));
  }
}

// ----------------------------------------------------------------------
// Embeddings route
// Proxies /v1/embeddings to OpenRouter (model contains "/") or OpenAI.
// The entire request body is forwarded as-is so non-standard input formats
// (e.g. multimodal image embeddings for nvidia/llama-nemotron-embed-vl-*)
// pass through without any transformation.
// ----------------------------------------------------------------------

router.post("/embeddings", authMiddleware, async (req: Request, res: Response) => {
  const body = req.body as Record<string, unknown>;

  if (typeof body["model"] !== "string" || !(body["model"] as string).trim()) {
    res.status(400).json({ error: { message: "'model' must be a non-empty string", type: "invalid_request_error" } });
    return;
  }
  if (body["input"] === undefined || body["input"] === null) {
    res.status(400).json({ error: { message: "'input' is required", type: "invalid_request_error" } });
    return;
  }

  const modelName = body["model"] as string;
  const client = modelName.includes("/") ? openrouter : openai;

  const ka = startNonStreamKeepalive(res);
  try {
    // Cast to any so non-standard multimodal inputs (content arrays) pass through the SDK unchanged.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const response = await client.embeddings.create(body as any);
    endNonStream(res, response);
  } catch (err: unknown) {
    const { status, message } = extractUpstreamError(err);
    endNonStreamError(res, status, message, "upstream_error");
  } finally {
    runtimeTimers.clearInterval(ka);
  }
});

// ----------------------------------------------------------------------
// Route
// ----------------------------------------------------------------------

router.post("/chat/completions", authMiddleware, async (req: Request, res: Response) => {
  try {
    const body = req.body as ChatBody;
    const { model, messages, stream } = body;

    // ── strict input validation ──────────────────────────────────────────────
    if (typeof model !== "string" || !model.trim()) {
      res.status(400).json({
        error: { message: "'model' must be a non-empty string", type: "invalid_request_error" },
      });
      return;
    }

    if (!Array.isArray(messages) || messages.length === 0) {
      res.status(400).json({
        error: { message: "'messages' must be a non-empty array", type: "invalid_request_error" },
      });
      return;
    }

    // Ensure each message is an object with a string role — reject early rather than crash later
    const badMsg = messages.find(
      (m) => !m || typeof m !== "object" || typeof (m as { role?: unknown }).role !== "string"
    );
    if (badMsg !== undefined) {
      res.status(400).json({
        error: { message: "Each message must be an object with a string 'role' field", type: "invalid_request_error" },
      });
      return;
    }
    // ────────────────────────────────────────────────────────────────────────

    if (!isModelEnabled(model)) {
      res.status(403).json({
        error: {
          message: `Model '${model}' is currently disabled by the proxy administrator.`,
          type: "invalid_request_error",
          code: "model_disabled",
        },
      });
      return;
    }

    const isClaude = model.startsWith("claude-");
    const isGemini = model.startsWith("gemini-");
    const isOpenRouter = !isClaude && !isGemini && model.includes("/");

    // ── Layered Claude cache plan ────────────────────────────────────────────
    // Tier 1: stable system → top-level cache_control
    // Tier 2: unstable but long common prefix → explicit system breakpoint
    // P2: conversation history breakpoint
    // If nothing is cacheable, still sink volatile layers out of system.
    let finalMessages = messages;
    let finalBody = body;

    const sysMsgIndex = messages.findIndex((m) => m.role === "system");
    const sysMsg = sysMsgIndex >= 0 ? messages[sysMsgIndex] : undefined;
    const isClaudeFamily = model.includes("claude") || isOpenRouterAnthropicModel(model);
    const skipDynamicSinkingForStickyRouting = shouldSkipDynamicSinkingForStickyRouting(model, messages);

    if (skipDynamicSinkingForStickyRouting) {
      req.log.info({ model }, "dynamic sinking skipped for sticky routing");
    } else if (
      isClaudeFamily &&
      sysMsg &&
      !hasAnyOpenAIBlockCacheControl(messages)
    ) {
      const systemText = flattenSystemTextContent(sysMsg.content);
      if (systemText) {
        const stableKey = `chat|${model}|${systemText.slice(0, 256)}`;
        const sinkRes = applyDynamicSinking(systemText, messages);
        finalMessages = sinkRes.messages;

        const comparableSystem = sinkRes.analysis.comparableSystem;
        const stable = comparableSystem.length > 0 ? checkSystemStability(stableKey, comparableSystem) : false;
        const lcpResult = comparableSystem.length > 0 ? computeLCPSplit(stableKey, comparableSystem) : null;

        let rewrittenSystemContent: string | OAIContentPart[] = sinkRes.system;
        let cachePlan = "none";

        if (stable) {
          finalBody = { ...body, cache_control: body.cache_control ?? { type: "ephemeral", ttl: "1h" } };
          rewrittenSystemContent = sinkRes.system;
          cachePlan = "T1";
        } else if (lcpResult) {
          rewrittenSystemContent = [
            { type: "text", text: lcpResult.stable, cache_control: { type: "ephemeral", ttl: "1h" } } as OAIContentPart,
            { type: "text", text: lcpResult.dynamic } as OAIContentPart,
          ];
          cachePlan = "T2";
        }

        finalMessages = finalMessages.map((m, idx) =>
          idx === sysMsgIndex ? { ...m, content: rewrittenSystemContent } : m
        );

        const historyResult = injectHistoryBreakpoint(finalMessages);
        finalMessages = historyResult.messages;
        if (historyResult.applied) {
          cachePlan = cachePlan === "none" ? "P2" : `${cachePlan}+P2`;
        }

        finalBody = { ...finalBody, messages: finalMessages };

        req.log.info({
          model,
          cachePlan,
          stableLayerLength: sinkRes.analysis.stableLength,
          lowFrequencyLayerLength: sinkRes.analysis.lowFrequencyLength,
          volatileLayerLength: sinkRes.analysis.volatileLength,
          comparableSystemLength: sinkRes.analysis.comparableSystem.length,
          lcpLength: lcpResult?.lcpLength ?? 0,
          sunk: sinkRes.sunk,
          historyApplied: historyResult.applied,
        }, "layered claude cache plan applied");
      }
    }

    // Prompt-based tool calling fallback — intercept before native routing.
    // Activated by `"x_use_prompt_tools": true` in the request body.
    if (finalBody.x_use_prompt_tools === true && finalBody.tools?.length) {
      await handlePromptTools(req, res, finalBody);
      return;
    }

    // ── Response cache (non-streaming only) ─────────────────────────────────
    //
    // Streaming responses are handled by the job queue which provides its own
    // buffer and reconnect logic.  The response cache targets non-streaming
    // calls: identical requests (same model + messages + params) are served
    // from memory after the first upstream round-trip.
    //
    // Thundering-herd / cache-stampede protection:
    //   If two identical requests arrive simultaneously (common in retries /
    //   parallel batch calls), both pass the first cacheGet() check before
    //   either has written to the cache.  Without protection both would call
    //   the upstream AI.  Instead, the SECOND request awaits the in-flight
    //   promise registered by the first, then retries the cache lookup.
    //   If the first request failed (upstream error), the second falls through
    //   and makes its own attempt rather than silently dropping the request.
    // ────────────────────────────────────────────────────────────────────────
    let cacheKey: string | undefined;
    let finishInflight: (() => void) | undefined;

    if (!stream) {
      // For OpenRouter requests, apply the same reasoning/verbosity defaults
      // that the handler would inject so that "implicit defaults" and
      // "explicitly-supplied defaults" hash to the same key.
      // Example: { model: "anthropic/claude-opus-4-5", messages } and
      //          { model: "...", messages, reasoning: { effort: "xhigh" }, verbosity: "high" }
      // both call OR with the same params and must share a cache entry.
      let bodyForHash: Record<string, unknown> = finalBody;
      if (isOpenRouter) {
        const pt = sanitizeOpenRouterReasoning(finalBody as Record<string, unknown>);
        const rDef = getOpenRouterReasoningDefault(finalBody.model, pt);
        const vDef = getOpenRouterVerbosityDefault(finalBody.model, pt);
        if (Object.keys(rDef).length > 0 || Object.keys(vDef).length > 0) {
          bodyForHash = { ...rDef, ...vDef, ...finalBody };
        }
      }
      cacheKey = hashRequest(bodyForHash);

      // 1. Fast path — cache hit.
      const cached = cacheGet(cacheKey);
      if (cached !== null) {
        req.log.info({ cacheKey: cacheKey.slice(0, 8), model, isStream: stream }, "cache HIT");
        if (stream && cached.chunks) {
          res.setHeader("X-Cache", "HIT-STREAM");
          replayStream(res, cached.chunks);
          return;
        } else if (!stream) {
          res.setHeader("Content-Type", "application/json");
          res.setHeader("X-Cache", "HIT");
          res.end(JSON.stringify(cached.data));
          return;
        }
      }

      // 2. Thundering-herd guard — wait for an identical in-flight request.
      const wasInflight = await waitForInflight(cacheKey);
      if (wasInflight) {
        const coalesced = cacheGet(cacheKey);
        if (coalesced !== null) {
          req.log.info({ cacheKey: cacheKey.slice(0, 8), model }, "cache COALESCED");
          res.setHeader("Content-Type", "application/json");
          res.setHeader("X-Cache", "COALESCED");
          res.end(JSON.stringify(coalesced.data));
          return;
        }
        // The in-flight request failed (upstream error) — fall through and
        // make our own upstream call rather than returning nothing.
      }

      // 3. Register this request as in-flight so concurrent duplicates wait.
      const finish = markInflight(cacheKey);
      if (finish) finishInflight = finish;
      req.log.info({ cacheKey: cacheKey.slice(0, 8), model }, "cache MISS");

      res.setHeader("X-Cache", "MISS");
    }

    try {
      if (isClaude) {
        if (stream) {
          await handleClaudeStream(req, res, finalBody, cacheKey);
        } else {
          await handleClaudeNonStream(req, res, finalBody, cacheKey);
        }
      } else if (isGemini) {
        if (stream) {
          await handleGeminiStream(req, res, finalBody);
        } else {
          await handleGeminiNonStream(req, res, finalBody, cacheKey);
        }
      } else if (isOpenRouter) {
        if (stream) {
          await handleOpenRouterStream(req, res, finalBody);
        } else {
          await handleOpenRouterNonStream(req, res, finalBody, cacheKey);
        }
      } else {
        if (stream) {
          await handleOpenAIStream(req, res, finalBody);
        } else {
          await handleOpenAINonStream(req, res, finalBody, cacheKey);
        }
      }
    } finally {
      // Always resolve the in-flight promise so waiting requests don't hang.
      finishInflight?.();
    }
  } catch (err: unknown) {
    req.log.error({ err }, "Error in /v1/chat/completions");
    if (!res.headersSent) {
      // Forward upstream API errors with their original status and message
      if (
        err &&
        typeof err === "object" &&
        "status" in err &&
        typeof (err as { status: unknown }).status === "number"
      ) {
        const apiErr = err as { status: number; message?: string; error?: { message?: string; type?: string } };
        const status = apiErr.status >= 400 ? apiErr.status : 502;
        const message = apiErr.error?.message ?? apiErr.message ?? "Upstream API error";
        const type = apiErr.error?.type ?? "upstream_error";
        res.status(status).json({ error: { message, type } });
      } else {
        res.status(500).json({ error: { message: "Internal server error", type: "server_error" } });
      }
    }
  }
});

export default router;
