/**
 * Multi-format API compatibility layer.
 *
 * Adds two additional endpoint families so clients targeting the native
 * Anthropic or Gemini API can point their base URL at this proxy without
 * any request-body changes:
 *
 *  POST /api/v1/messages                        — Anthropic Messages API
 *  POST /api/v1beta/models/:model:generateContent — Gemini generateContent
 *  POST /api/v1beta/models/:model:streamGenerateContent — Gemini streaming
 *
 * Each adapter:
 *   1. Authenticates via the shared authMiddleware.
 *   2. Validates the minimal required fields.
 *   3. Calls the upstream SDK with the native format (no round-tripping
 *      through the OAI pipeline) so all provider-specific fields pass through.
 *   4. For Anthropic: auto-injects prompt-caching breakpoints (same logic as
 *      the main /v1/chat/completions handler).
 *   5. Returns the upstream response in its native format so callers receive
 *      exactly the schema their SDK expects.
 */

import { Router, type IRouter, type Request, type Response } from "express";
import Anthropic from "@anthropic-ai/sdk";
import { authMiddleware } from "../middlewares/auth";
import {
  anthropic,
  gemini,
  getClaudeMaxTokens,
  autoInjectPromptCaching,
  stripClaudeSuffix,
  getThinkingBudget,
} from "./v1/chat";
import { logger } from "../lib/logger";

// ────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ────────────────────────────────────────────────────────────────────────────

const NON_STREAM_KEEPALIVE_MS = 20_000;
const SSE_KEEPALIVE_MS = 30_000;

const rt = globalThis as unknown as {
  setInterval: (fn: () => void, ms: number) => number;
  clearInterval: (id: number) => void;
};

function extractUpstreamError(err: unknown): { status: number; message: string } {
  const message = err instanceof Error ? err.message : String(err);
  const status =
    err !== null && typeof err === "object" && "status" in err &&
    typeof (err as { status: unknown }).status === "number"
      ? Math.max(400, (err as { status: number }).status)
      : 502;
  return { status, message };
}

function startNonStreamKeepalive(res: Response, contentType = "application/json"): number {
  res.setHeader("Content-Type", contentType);
  res.setHeader("X-Accel-Buffering", "no");
  return rt.setInterval(() => {
    if (!res.writableEnded) res.write("\n");
  }, NON_STREAM_KEEPALIVE_MS);
}

// ────────────────────────────────────────────────────────────────────────────
// Anthropic Messages API  (/v1/messages)
// ────────────────────────────────────────────────────────────────────────────

export const anthropicCompatRouter: IRouter = Router();

/**
 * Build Anthropic MessageCreateParams from a native Anthropic Messages API
 * request body.  Applies auto-inject prompt caching and extended-thinking
 * support (via the -thinking / -thinking-visible model suffixes).
 */
function buildAnthropicParams(body: Record<string, unknown>): Anthropic.MessageCreateParams {
  const rawModel = body["model"] as string;
  const { baseModel, thinkingEnabled } = stripClaudeSuffix(rawModel);

  const messages = body["messages"] as Anthropic.MessageParam[];
  const system = body["system"] as string | Anthropic.TextBlockParam[] | undefined;
  const anthropicTools = body["tools"] as Anthropic.Tool[] | undefined;

  // Auto-inject Anthropic prompt caching breakpoints (same as main handler).
  const cached = autoInjectPromptCaching({ system, messages, tools: anthropicTools });

  const maxTokens = (body["max_tokens"] as number | undefined) ?? getClaudeMaxTokens(baseModel);

  const params: Anthropic.MessageCreateParams = {
    model: baseModel,
    messages: cached.messages,
    max_tokens: maxTokens,
    system: cached.system,
    tools: cached.tools,
    // Pass through standard generation params
    ...(body["temperature"]    !== undefined && { temperature:    body["temperature"]    as number }),
    ...(body["top_p"]          !== undefined && { top_p:          body["top_p"]          as number }),
    ...(body["top_k"]          !== undefined && { top_k:          body["top_k"]          as number }),
    ...(body["stop_sequences"] !== undefined && { stop_sequences: body["stop_sequences"] as string[] }),
    ...(body["tool_choice"]    !== undefined && { tool_choice:    body["tool_choice"]    as Anthropic.ToolChoiceParam }),
    ...(body["metadata"]       !== undefined && { metadata:       body["metadata"]       as Anthropic.MessageCreateParams["metadata"] }),
    stream: false,
  };

  // Extended thinking — activated by the -thinking suffix on the model name.
  // temperature / top_p / top_k are incompatible with thinking mode.
  if (thinkingEnabled) {
    const budget = getThinkingBudget(maxTokens);
    (params as Record<string, unknown>)["thinking"] = { type: "enabled", budget_tokens: budget };
    delete params.temperature;
    delete params.top_p;
    delete params.top_k;
  }

  return params;
}

anthropicCompatRouter.post("/messages", authMiddleware, async (req: Request, res: Response) => {
  const body = req.body as Record<string, unknown>;

  // ── Input validation ──────────────────────────────────────────────────────
  if (typeof body["model"] !== "string" || !(body["model"] as string).trim()) {
    res.status(400).json({
      type: "error",
      error: { type: "invalid_request_error", message: "'model' must be a non-empty string" },
    });
    return;
  }
  if (!Array.isArray(body["messages"]) || (body["messages"] as unknown[]).length === 0) {
    res.status(400).json({
      type: "error",
      error: { type: "invalid_request_error", message: "'messages' must be a non-empty array" },
    });
    return;
  }

  const isStream = Boolean(body["stream"]);
  const model = body["model"] as string;

  logger.info({ model, stream: isStream, adapter: "anthropic-compat" }, "compat /v1/messages");

  let params: Anthropic.MessageCreateParams;
  try {
    params = buildAnthropicParams(body);
  } catch (err) {
    const { message } = extractUpstreamError(err);
    res.status(400).json({ type: "error", error: { type: "invalid_request_error", message } });
    return;
  }

  if (isStream) {
    // ── Streaming: pipe Anthropic SSE events directly ─────────────────────
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders();

    const ka = rt.setInterval(() => {
      if (!res.writableEnded) res.write(": ping\n\n");
    }, SSE_KEEPALIVE_MS);

    try {
      const stream = await anthropic.messages.create({ ...params, stream: true });
      for await (const event of stream) {
        if (res.writableEnded) break;
        res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
      }
    } catch (err) {
      const { message } = extractUpstreamError(err);
      const errPayload = { type: "error", error: { type: "api_error", message } };
      if (!res.writableEnded) {
        res.write(`event: error\ndata: ${JSON.stringify(errPayload)}\n\n`);
      }
    } finally {
      rt.clearInterval(ka);
      if (!res.writableEnded) res.end();
    }
  } else {
    // ── Non-streaming: keepalive newlines while waiting for upstream ──────
    const ka = startNonStreamKeepalive(res);
    try {
      const response = await anthropic.messages.create({ ...params, stream: false });
      if (!res.writableEnded) res.end(JSON.stringify(response));
    } catch (err) {
      const { status, message } = extractUpstreamError(err);
      if (!res.writableEnded) {
        if (!res.headersSent) res.status(status);
        res.end(JSON.stringify({ type: "error", error: { type: "api_error", message } }));
      }
    } finally {
      rt.clearInterval(ka);
    }
  }
});

// ────────────────────────────────────────────────────────────────────────────
// Gemini Native API  (/v1beta/models/:modelAction)
//
// The Gemini SDK path looks like:
//   POST /v1beta/models/gemini-1.5-pro:generateContent
//   POST /v1beta/models/gemini-1.5-pro:streamGenerateContent
//
// Express parses the path segment after /models/ into req.params.modelAction.
// The model name and action are separated by the last colon in that string.
// ────────────────────────────────────────────────────────────────────────────

export const geminiCompatRouter: IRouter = Router();

geminiCompatRouter.post("/models/:modelAction", authMiddleware, async (req: Request, res: Response) => {
  const modelAction = req.params["modelAction"] ?? "";
  const colonIdx = modelAction.lastIndexOf(":");
  const modelName = colonIdx >= 0 ? modelAction.slice(0, colonIdx) : modelAction;
  const action    = colonIdx >= 0 ? modelAction.slice(colonIdx + 1) : "generateContent";

  const body = req.body as Record<string, unknown>;

  if (!modelName) {
    res.status(400).json({ error: { code: 400, message: "model name is required in the path", status: "INVALID_ARGUMENT" } });
    return;
  }
  if (!Array.isArray(body["contents"]) || (body["contents"] as unknown[]).length === 0) {
    res.status(400).json({ error: { code: 400, message: "'contents' must be a non-empty array", status: "INVALID_ARGUMENT" } });
    return;
  }

  const isStream = action === "streamGenerateContent";

  logger.info({ model: modelName, action, stream: isStream, adapter: "gemini-compat" }, "compat /v1beta/models");

  // Build the Gemini SDK config from the standard Gemini request fields.
  // We pass them through via a generic Record so provider-specific fields
  // (responseMimeType, responseSchema, thinkingConfig, etc.) are preserved.
  const generationConfig = (body["generationConfig"] as Record<string, unknown> | undefined) ?? {};
  const config: Record<string, unknown> = { ...generationConfig };

  if (body["systemInstruction"] !== undefined) config["systemInstruction"] = body["systemInstruction"];
  if (body["tools"]             !== undefined) config["tools"]             = body["tools"];
  if (body["toolConfig"]        !== undefined) config["toolConfig"]        = body["toolConfig"];
  if (body["safetySettings"]    !== undefined) config["safetySettings"]    = body["safetySettings"];
  if (body["cachedContent"]     !== undefined) config["cachedContent"]     = body["cachedContent"];

  type GeminiContents = Parameters<typeof gemini.models.generateContent>[0]["contents"];
  type GeminiConfig   = Parameters<typeof gemini.models.generateContent>[0]["config"];

  const contents = body["contents"] as GeminiContents;

  if (isStream) {
    // ── Streaming ─────────────────────────────────────────────────────────
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    res.flushHeaders();

    const ka = rt.setInterval(() => {
      if (!res.writableEnded) res.write(": ping\n\n");
    }, SSE_KEEPALIVE_MS);

    try {
      const stream = await gemini.models.generateContentStream({
        model: modelName,
        contents,
        config: config as GeminiConfig,
      });
      for await (const chunk of stream) {
        if (res.writableEnded) break;
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }
    } catch (err) {
      const { message, status } = extractUpstreamError(err);
      if (!res.writableEnded) {
        res.write(`data: ${JSON.stringify({ error: { code: status, message, status: "INTERNAL" } })}\n\n`);
      }
    } finally {
      rt.clearInterval(ka);
      if (!res.writableEnded) res.end();
    }
  } else {
    // ── Non-streaming ─────────────────────────────────────────────────────
    const ka = startNonStreamKeepalive(res);
    try {
      const response = await gemini.models.generateContent({
        model: modelName,
        contents,
        config: config as GeminiConfig,
      });
      if (!res.writableEnded) res.end(JSON.stringify(response));
    } catch (err) {
      const { status, message } = extractUpstreamError(err);
      if (!res.writableEnded) {
        if (!res.headersSent) res.status(status);
        res.end(JSON.stringify({ error: { code: status, message, status: "INTERNAL" } }));
      }
    } finally {
      rt.clearInterval(ka);
    }
  }
});
