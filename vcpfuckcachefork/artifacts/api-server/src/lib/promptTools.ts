/**
 * Prompt-based tool calling fallback.
 *
 * Activated by setting `"x_use_prompt_tools": true` in the request body.
 * Useful for models that don't support native function calling.
 *
 * The proxy:
 *   1. Strips `tools` / `tool_choice` from the upstream request.
 *   2. Injects a system-prompt section that teaches the model the tool schema
 *      and instructs it to respond with a specific JSON format.
 *   3. After receiving the response, parses the JSON and converts it into
 *      OpenAI-compatible `tool_calls` or plain text.
 */

export interface PromptTool {
  type?: string;
  function?: {
    name?: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

const INSTRUCTION = `
你是一个严格遵守输出格式的助手。根据用户的需求和下面提供的工具定义，判断是否需要调用工具。

### 工具定义
{{TOOLS_SCHEMA}}

### 输出格式（严格遵守，只输出 JSON，不要任何解释或 markdown）
- 如果不需要调用工具：{"type":"text","content":"你的回答"}
- 如果需要调用一个或多个工具：{"type":"tool_calls","calls":[{"name":"工具名","parameters":{"参数名":"值"}}]}

注意：只输出上述 JSON 格式中的一种，不能混用，不能在 JSON 外部添加任何文字。`.trim();

/**
 * Build the system-prompt injection for prompt-based tool calling.
 * Silently skips malformed tool entries (missing name / function field).
 */
export function buildPromptToolsInstruction(tools: PromptTool[]): string {
  const valid = tools
    .filter((t) => t?.function?.name)
    .map((t) => ({
      name: t.function!.name ?? "",
      description: t.function!.description ?? "",
      parameters: t.function!.parameters ?? {},
    }));

  const schema = JSON.stringify(valid, null, 2);
  return INSTRUCTION.replace("{{TOOLS_SCHEMA}}", schema);
}

export interface ParsedPromptToolsResult {
  isToolCall: boolean;
  calls?: Array<{ id: string; name: string; arguments: string }>;
  content: string;
}

/**
 * Walk forward from `start` in `text` and find the index of the closing `}`
 * that matches the opening `{` at `start`.
 * Returns -1 if no balanced match is found.
 *
 * Handles:
 *   - Nested objects and arrays
 *   - String literals (including escaped quotes inside them)
 */
function findMatchingBrace(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  let i = start;

  while (i < text.length) {
    const ch = text[i];

    if (inString) {
      if (ch === "\\") {
        i += 2; // skip escaped character
        continue;
      }
      if (ch === '"') inString = false;
    } else {
      if (ch === '"') {
        inString = true;
      } else if (ch === "{") {
        depth++;
      } else if (ch === "}") {
        depth--;
        if (depth === 0) return i;
      }
    }
    i++;
  }
  return -1;
}

/**
 * Parse the model's JSON response and convert to structured result.
 *
 * Robust against:
 *   - Markdown code fences (```json ... ```)
 *   - Leading/trailing prose around the JSON object
 *   - Escaped characters inside string values
 *   - Extra closing braces after the JSON (e.g. template variables like {x})
 */
export function parsePromptToolsResponse(raw: string): ParsedPromptToolsResult {
  let text = raw.trim();

  // Strip markdown fences (```json\n...\n``` or ```\n...\n```)
  text = text.replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/, "").trim();

  // Find the first `{` in the text
  const start = text.indexOf("{");
  if (start === -1) return { isToolCall: false, content: raw };

  // Find the matching `}` using a proper bracket-depth scanner
  const end = findMatchingBrace(text, start);
  if (end === -1) return { isToolCall: false, content: raw };

  try {
    const parsed = JSON.parse(text.slice(start, end + 1)) as Record<string, unknown>;

    if (parsed["type"] === "tool_calls" && Array.isArray(parsed["calls"])) {
      const calls = parsed["calls"] as Array<{ name?: unknown; parameters?: Record<string, unknown> }>;
      // B4 fix: if all call entries are malformed (no valid name), the filtered list is empty.
      // In that case fall through to text — an empty tool_calls array is an invalid OAI response.
      const validCalls = calls
        .filter((c) => typeof c.name === "string" && c.name)
        .map((c, i) => ({
          id: `call_pt_${Date.now()}_${i}`,
          name: c.name as string,
          arguments: JSON.stringify(c.parameters ?? {}),
        }));
      if (validCalls.length > 0) {
        return { isToolCall: true, calls: validCalls, content: "" };
      }
    }

    if (parsed["type"] === "text" && typeof parsed["content"] === "string") {
      return { isToolCall: false, content: parsed["content"] };
    }
  } catch {
    // Invalid JSON — treat the whole original response as plain text
  }

  return { isToolCall: false, content: raw };
}

/**
 * Build an OpenAI chat completion object from a prompt-tools result.
 * Used for the non-streaming path.
 */
export function buildCompletionFromPromptTools(
  result: ParsedPromptToolsResult,
  model: string,
  usage: { prompt_tokens: number; completion_tokens: number },
): Record<string, unknown> {
  const id = `chatcmpl-pt-${Date.now()}`;

  const message: Record<string, unknown> = {
    role: "assistant",
    content: result.isToolCall ? null : result.content,
  };

  if (result.isToolCall && result.calls && result.calls.length > 0) {
    message["tool_calls"] = result.calls.map((c) => ({
      id: c.id,
      type: "function",
      function: { name: c.name, arguments: c.arguments },
    }));
  }

  return {
    id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message,
        finish_reason: result.isToolCall && result.calls?.length ? "tool_calls" : "stop",
      },
    ],
    usage: {
      ...usage,
      total_tokens: usage.prompt_tokens + usage.completion_tokens,
    },
  };
}
