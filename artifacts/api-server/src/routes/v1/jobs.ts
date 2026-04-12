/**
 * Async job endpoints — relay layer for long-running AI streams.
 *
 * Endpoints:
 *
 *   POST  /v1/jobs
 *     Same body as POST /v1/chat/completions.
 *     Starts the upstream call in the BACKGROUND, returns immediately:
 *       { job_id: "job-xxx", status: "running", model: "..." }
 *     The job keeps running even if the client disconnects.
 *
 *   GET   /v1/jobs/:id/stream
 *     SSE stream for the job.  Replays all buffered chunks, then live-
 *     streams new ones as they arrive.  Event IDs are "<jobId>:<chunkIdx>".
 *     Supports Last-Event-ID reconnect:
 *       - Client reconnects after Replit's 300 s proxy cut
 *       - Header: "Last-Event-ID: job-xxx:42"
 *       - Server resumes from chunk 43 — no tokens lost
 *
 *   GET   /v1/jobs/:id
 *     Non-streaming status check:
 *       { job_id, status: "running"|"done"|"error", model,
 *         done, error, chunk_count }
 *
 * Auth: same Bearer token as /v1/chat/completions.
 *
 * Usage example (curl, reconnect-safe):
 *
 *   JOB=$(curl -s -X POST $BASE/v1/jobs \
 *     -H "Authorization: Bearer vcspeeper" \
 *     -H "Content-Type: application/json" \
 *     -d '{"model":"claude-opus-4-5","messages":[...]}' | jq -r .job_id)
 *
 *   curl -s --no-buffer "$BASE/v1/jobs/$JOB/stream" \
 *     -H "Authorization: Bearer vcspeeper"
 *
 *   # On disconnect, reconnect with Last-Event-ID:
 *   curl -s --no-buffer "$BASE/v1/jobs/$JOB/stream" \
 *     -H "Authorization: Bearer vcspeeper" \
 *     -H "Last-Event-ID: $JOB:42"
 */

import { Router, type Request, type Response } from "express";
import { authMiddleware } from "../../middlewares/auth.js";
import {
  type Job,
  createJob,
  getJob,
  appendJobChunk,
  finishJob,
  failJob,
  streamJobToResponse,
  parseLastEventId,
} from "../../lib/jobQueue.js";
import { isModelEnabled } from "../../lib/modelGroups.js";

import {
  anthropic,
  gemini,
  openrouter,
  openai,
  stripClaudeSuffix,
  getClaudeMaxTokens,
  getThinkingBudget,
  ensureMaxReasoningEffort,
  sanitizeOpenRouterReasoning,
  getOpenRouterVerbosityDefault,
  isOpenRouterAnthropicModel,
  isPromptCacheSupportedModel,
  prepareOpenRouterCachingRequest,
  applyVCPDynamicSinking,
  convertMessagesToAnthropic,
  convertMessagesToGemini,
  convertToolsToAnthropic,
  convertToolChoiceToAnthropic,
  autoInjectPromptCaching,
  resolveImageUrls,
  stripGeminiSuffix,
  type ChatBody,
} from "./chat.js";

import type Anthropic from "@anthropic-ai/sdk";
import type OpenAI from "openai";

const router = Router();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function setSseHeaders(res: Response) {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("X-Accel-Buffering", "no");
  if (res.socket) {
    res.socket.setNoDelay(true);
    res.socket.setTimeout(0);
  }
}

// ---------------------------------------------------------------------------
// Background runner — mirrors the streaming logic in chat.ts but writes
// to the job buffer instead of directly to the HTTP response.
// ---------------------------------------------------------------------------

async function runJobBackground(body: ChatBody, job: Job): Promise<void> {
  const { model } = body;

  try {
    const messages = await resolveImageUrls(body.messages);

    // ── Anthropic ─────────────────────────────────────────────────────────
    if (model.startsWith("claude-")) {
      const { baseModel, thinkingEnabled, thinkingVisible } = stripClaudeSuffix(model);
      const modelMax = getClaudeMaxTokens(baseModel);
      const rawMaxTokens = body.max_tokens && body.max_tokens > 0 ? body.max_tokens : modelMax;
      const maxTokens = Math.min(rawMaxTokens, modelMax);
      const { system: rawSystem, messages: rawAnthropicMessages } = convertMessagesToAnthropic(messages);
      const rawAnthropicTools = (body.tools?.length && body.tool_choice !== "none")
        ? convertToolsToAnthropic(body.tools)
        : undefined;
      const anthropicToolChoice = (body.tool_choice && body.tool_choice !== "none")
        ? convertToolChoiceToAnthropic(body.tool_choice)
        : undefined;

      // Auto-inject prompt-caching breakpoints (no-op if caller already set cache_control).
      // Wrapped in try-catch: any injection error falls back to original params.
      let system = rawSystem;
      let anthropicMessages = rawAnthropicMessages;
      let anthropicTools = rawAnthropicTools;
      try {
        ({ system, messages: anthropicMessages, tools: anthropicTools } = autoInjectPromptCaching({
          system: rawSystem,
          messages: rawAnthropicMessages,
          tools: rawAnthropicTools,
          model: baseModel,
        }));
      } catch { /* ignore — proceed without caching */ }

      const params: Record<string, unknown> = {
        model: baseModel, max_tokens: maxTokens, messages: anthropicMessages, stream: true,
      };
      if (system) params["system"] = system;
      if (!thinkingEnabled) {
        if (body.temperature !== undefined) params["temperature"] = body.temperature;
        else if (body.top_p !== undefined) params["top_p"] = body.top_p;
        if (body.stop) params["stop_sequences"] = Array.isArray(body.stop) ? body.stop : [body.stop];
      }
      if (thinkingEnabled) params["thinking"] = { type: "enabled", budget_tokens: getThinkingBudget(maxTokens) };
      if (anthropicTools) params["tools"] = anthropicTools;
      if (anthropicToolChoice) params["tool_choice"] = anthropicToolChoice;
      if (body["cache_control"]) params["cache_control"] = body["cache_control"];

      const id = `chatcmpl-${Date.now()}`;
      let inThinking = false;
      let inputTokens = 0;
      let cacheReadTokens = 0;
      let cacheCreationTokens = 0;
      const blockIdxToToolIdx: Record<number, number> = {};
      let toolCallCount = 0;

      const chunk = (delta: Record<string, unknown>, finishReason?: string | null) => ({
        id, object: "chat.completion.chunk", created: Math.floor(Date.now() / 1000), model,
        choices: [{ index: 0, delta, finish_reason: finishReason ?? null }],
      });

      appendJobChunk(job, chunk({ role: "assistant", content: "" }));

      const stream = anthropic.messages.stream(params as Anthropic.MessageCreateParamsStreaming);
      for await (const event of stream) {
        if (event.type === "message_start") {
          inputTokens = event.message.usage?.input_tokens ?? 0;
          cacheReadTokens = (event.message.usage as Record<string, unknown>)?.["cache_read_input_tokens"] as number ?? 0;
          cacheCreationTokens = (event.message.usage as Record<string, unknown>)?.["cache_creation_input_tokens"] as number ?? 0;

        } else if (event.type === "content_block_start") {
          const block = event.content_block;
          const idx = event.index;
          if (block.type === "thinking") {
            inThinking = true;
            if (thinkingVisible) appendJobChunk(job, chunk({ content: "<thinking>\n" }));
          } else if (block.type === "text") {
            if (inThinking && thinkingVisible) appendJobChunk(job, chunk({ content: "\n</thinking>\n\n" }));
            inThinking = false;
          } else if (block.type === "tool_use") {
            const toolIdx = toolCallCount++;
            blockIdxToToolIdx[idx] = toolIdx;
            appendJobChunk(job, chunk({
              tool_calls: [{ index: toolIdx, id: block.id, type: "function", function: { name: block.name, arguments: "" } }],
            }));
          }

        } else if (event.type === "content_block_delta") {
          const delta = event.delta;
          const idx = event.index;
          if (delta.type === "thinking_delta") {
            if (thinkingVisible) appendJobChunk(job, chunk({ content: delta.thinking }));
          } else if (delta.type === "text_delta") {
            appendJobChunk(job, chunk({ content: delta.text }));
          } else if (delta.type === "input_json_delta") {
            const toolIdx = blockIdxToToolIdx[idx] ?? 0;
            appendJobChunk(job, chunk({ tool_calls: [{ index: toolIdx, function: { arguments: delta.partial_json ?? "" } }] }));
          }

        } else if (event.type === "message_delta") {
          if (inThinking && thinkingVisible) {
            appendJobChunk(job, chunk({ content: "\n</thinking>\n\n" }));
            inThinking = false;
          }
          const stopReason = event.delta.stop_reason;
          const finishReason = stopReason === "tool_use" ? "tool_calls" : stopReason === "end_turn" ? "stop" : (stopReason ?? "stop");
          const outputTokens = event.usage?.output_tokens ?? 0;
          const jobUsage: Record<string, unknown> = {
            prompt_tokens: inputTokens,
            completion_tokens: outputTokens,
            total_tokens: inputTokens + outputTokens,
          };
          if (cacheReadTokens > 0 || cacheCreationTokens > 0) {
            jobUsage["cache_read_input_tokens"] = cacheReadTokens;
            jobUsage["cache_creation_input_tokens"] = cacheCreationTokens;
            jobUsage["prompt_tokens_details"] = { cached_tokens: cacheReadTokens };
          }
          appendJobChunk(job, { ...chunk({}, finishReason), usage: jobUsage });
        }
      }

    // ── Gemini ────────────────────────────────────────────────────────────
    } else if (model.startsWith("gemini-")) {
      const { baseModel, thinkingEnabled } = stripGeminiSuffix(model);
      const { systemInstruction, contents } = convertMessagesToGemini(messages);
      const config: Record<string, unknown> = { maxOutputTokens: body.max_tokens ?? 65536 };
      if (body.temperature !== undefined) config["temperature"] = body.temperature;
      if (body.top_p !== undefined) config["topP"] = body.top_p;
      if (systemInstruction) config["systemInstruction"] = systemInstruction;
      if (thinkingEnabled) config["thinkingConfig"] = { thinkingBudget: -1 };

      const id = `chatcmpl-${Date.now()}`;
      const chunk = (delta: Record<string, unknown>, finishReason?: string | null) => ({
        id, object: "chat.completion.chunk", created: Math.floor(Date.now() / 1000), model,
        choices: [{ index: 0, delta, finish_reason: finishReason ?? null }],
      });
      appendJobChunk(job, chunk({ role: "assistant", content: "" }));

      const stream = await gemini.models.generateContentStream({
        model: baseModel,
        contents,
        config: config as Parameters<typeof gemini.models.generateContentStream>[0]["config"],
      });
      let inputTokens = 0, outputTokens = 0;
      let toolCallCount = 0;
      let hasToolCalls = false;
      for await (const c of stream) {
        // Parse candidates/parts to handle both text and functionCall responses.
        const candidates = (c as unknown as Record<string, unknown>)["candidates"] as Array<Record<string, unknown>> | undefined;
        const parts = (candidates?.[0]?.["content"] as Record<string, unknown> | undefined)?.["parts"] as Array<Record<string, unknown>> | undefined;
        if (parts) {
          for (const part of parts) {
            if (typeof part["text"] === "string" && part["text"]) {
              appendJobChunk(job, chunk({ content: part["text"] }));
            } else if (part["functionCall"] && typeof part["functionCall"] === "object") {
              const fc = part["functionCall"] as { name?: string; args?: Record<string, unknown> };
              const toolIdx = toolCallCount++;
              hasToolCalls = true;
              appendJobChunk(job, chunk({
                tool_calls: [{
                  index: toolIdx,
                  id: `call_${id}_${toolIdx}`,
                  type: "function",
                  function: { name: fc.name ?? "", arguments: JSON.stringify(fc.args ?? {}) },
                }],
              }));
            }
          }
        } else {
          // Fallback for simple text-only chunks (no candidates wrapper)
          let text: string | undefined;
          try { text = c.text ?? undefined; } catch { /* safety/error block */ }
          if (text) appendJobChunk(job, chunk({ content: text }));
        }
        if (c.usageMetadata) {
          inputTokens = c.usageMetadata.promptTokenCount ?? inputTokens;
          outputTokens = c.usageMetadata.candidatesTokenCount ?? outputTokens;
        }
      }
      appendJobChunk(job, chunk({}, hasToolCalls ? "tool_calls" : "stop"));
      appendJobChunk(job, {
        id, object: "chat.completion.chunk", created: Math.floor(Date.now() / 1000), model,
        choices: [],
        usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
      });

    // ── OpenRouter ────────────────────────────────────────────────────────
    } else if (model.includes("/")) {
      const { model: _m, messages: _msgs, stream: _s, ...passThrough } = body;
      // Ensure maximum reasoning effort + sanitize conflict
      let pt = ensureMaxReasoningEffort(model, passThrough as Record<string, unknown>);
      pt = sanitizeOpenRouterReasoning(pt);
      const verbosityDefault = getOpenRouterVerbosityDefault(model, pt);

      // Anthropic extended-thinking param compatibility: clamp temperature to 1,
      // strip top_p/top_k only for Claude models with reasoning enabled.
      if (isOpenRouterAnthropicModel(model) && pt["reasoning"]) {
        if (pt["temperature"] !== undefined) pt["temperature"] = 1;
        delete pt["top_p"];
        delete pt["top_k"];
      }

      // ── VCP-aware Dynamic Sinking ──────────────────────────────────────
      // Extract volatile content from system prompt and sink it to the last
      // user message, keeping the system prefix stable for prefix caching.
      const vcpSink = applyVCPDynamicSinking(messages);
      const messagesAfterSink = vcpSink.messages;

      // For OpenRouter Anthropic models inject per-block cache_control (1h TTL)
      // so the upstream Anthropic provider caches the prompt.
      // Skip if top-level cache_control already present (would downgrade TTL 1h→5min).
      const hasTopLevelCC = !!pt["cache_control"];
      const processedMessages = (isPromptCacheSupportedModel(model) && !hasTopLevelCC)
        ? prepareOpenRouterCachingRequest(messagesAfterSink, model).messages
        : messagesAfterSink;

      const params = {
        ...verbosityDefault, ...pt,
        model,
        messages: processedMessages as OpenAI.ChatCompletionMessageParam[],
        stream: true as const,
        stream_options: { include_usage: true },
      } as OpenAI.Chat.ChatCompletionCreateParamsStreaming;

      const stream = await openrouter.chat.completions.create(params);
      for await (const c of stream) appendJobChunk(job, c);

    // ── OpenAI ────────────────────────────────────────────────────────────
    } else {
      const params: OpenAI.Chat.ChatCompletionCreateParamsStreaming = {
        model, messages: messages as OpenAI.ChatCompletionMessageParam[],
        stream: true, stream_options: { include_usage: true },
      };
      if (body.temperature !== undefined) params.temperature = body.temperature;
      if (body.top_p !== undefined) params.top_p = body.top_p;
      if (body.max_tokens !== undefined) params.max_tokens = body.max_tokens;
      if (body.stop !== undefined) params.stop = body.stop as string | string[];
      if (body.seed !== undefined) params.seed = body.seed;
      if (body.presence_penalty !== undefined) params.presence_penalty = body.presence_penalty;
      if (body.frequency_penalty !== undefined) params.frequency_penalty = body.frequency_penalty;
      if (body.tools?.length) {
        params.tools = body.tools as OpenAI.ChatCompletionTool[];
        if (body.tool_choice !== undefined && body.tool_choice !== "none")
          params.tool_choice = body.tool_choice as OpenAI.ChatCompletionToolChoiceOption;
      }
      if (body.response_format !== undefined) params.response_format = body.response_format as OpenAI.ResponseFormatText;

      const stream = await openai.chat.completions.create(params);
      for await (const c of stream) appendJobChunk(job, c);
    }

    finishJob(job);

  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    failJob(job, msg);
  }
}

// ---------------------------------------------------------------------------
// POST /v1/jobs  — create async job
// ---------------------------------------------------------------------------

router.post("/", authMiddleware, async (req: Request, res: Response) => {
  const body = req.body as ChatBody;
  const { model, messages } = body;

  if (typeof model !== "string" || !model.trim()) {
    res.status(400).json({ error: { message: "'model' must be a non-empty string", type: "invalid_request_error" } });
    return;
  }
  if (!Array.isArray(messages) || messages.length === 0) {
    res.status(400).json({ error: { message: "'messages' must be a non-empty array", type: "invalid_request_error" } });
    return;
  }
  if (!isModelEnabled(model)) {
    res.status(403).json({ error: { message: `Model '${model}' is disabled.`, type: "invalid_request_error", code: "model_disabled" } });
    return;
  }

  const job = createJob(model);

  // Start background — do NOT await.  The job keeps running after we respond.
  runJobBackground(body, job).catch(() => { /* errors are captured in failJob */ });

  res.json({ job_id: job.id, status: "running", model });
});

// ---------------------------------------------------------------------------
// GET /v1/jobs/:id/stream  — SSE stream, reconnectable via Last-Event-ID
// ---------------------------------------------------------------------------

router.get("/:id/stream", authMiddleware, async (req: Request, res: Response) => {
  const job = getJob(req.params.id);
  if (!job) {
    res.status(404).json({ error: { message: "Job not found or expired", type: "not_found" } });
    return;
  }

  setSseHeaders(res);
  // Keepalive: send a real data event every 200 s so Replit's proxy (and any
  // intermediate proxy) sees bytes and resets its 300 s idle timeout.
  // 200 s < 300 s gives a 100 s safety margin.
  const keepalive = setInterval(() => {
    if (!res.writableEnded) {
      res.write(
        `data: ${JSON.stringify({ id: `ka-${Date.now()}`, object: "chat.completion.chunk", created: Math.floor(Date.now() / 1000), model: job.model, choices: [] })}\n\n`,
      );
    }
  }, 200_000);

  // Determine resume point from Last-Event-ID header.
  const lastEventId = req.headers["last-event-id"] as string | undefined;
  const resume = parseLastEventId(lastEventId);
  const fromIdx = Math.max(0, (resume?.jobId === job.id) ? (resume.lastIdx + 1) : 0);

  try {
    await streamJobToResponse(res, job, fromIdx);
  } finally {
    clearInterval(keepalive);
    if (!res.writableEnded) res.end();
  }
});

// ---------------------------------------------------------------------------
// GET /v1/jobs/:id  — status check (non-streaming)
// ---------------------------------------------------------------------------

router.get("/:id", authMiddleware, (req: Request, res: Response) => {
  const job = getJob(req.params.id);
  if (!job) {
    res.status(404).json({ error: { message: "Job not found or expired", type: "not_found" } });
    return;
  }
  res.json({
    job_id: job.id,
    status: job.error ? "error" : job.done ? "done" : "running",
    model: job.model,
    done: job.done,
    error: job.error,
    chunk_count: job.chunks.length,
  });
});

export default router;
