# AI Monorepo

## Overview

AI Monorepo (v1.1.7) — A lightweight OpenAI-compatible API proxy that routes requests to the appropriate backend based on model name. Includes a dark-themed React admin portal at `/`. Auth via `PROXY_API_KEY`.

### Model Routing

| Model name pattern | Backend |
|---|---|
| `claude-*` | Anthropic (Claude) via Replit AI Integration |
| `gemini-*` | Google Gemini via Replit AI Integration |
| Contains `/` (e.g. `meta-llama/llama-4-maverick`) | OpenRouter via Replit AI Integration |
| Everything else (`gpt-4o`, `o3`, …) | OpenAI via Replit AI Integration |

### Thinking Mode Suffixes

Append to any Claude or Gemini model name:
- `-thinking` — extended reasoning, output only
- `-thinking-visible` — extended reasoning with `<thinking>` block visible

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Build**: esbuild (ESM bundle)
- **Frontend**: React + Vite + Tailwind CSS

## Artifacts

### `artifacts/api-server` — Express + TypeScript API proxy

- Serves at `/api`
- Routes all `/v1/*` endpoints (OpenAI-compatible)
- Auth: `Authorization: Bearer`, `x-goog-api-key`, or `?key=` query param
- Full tool/function calling support (streaming + non-streaming) for Claude and OpenAI
- Image recognition: prefetches remote `image_url` parts server-side (base64)
- SSE keepalive every 5s on all streaming handlers
- All HTTP server timeouts disabled (safe for 10+ min long generations)
- Model groups management: enable/disable models per group, persisted in `model-groups.json`
- Body limit: 50mb, global CORS enabled

### `artifacts/api-portal` — React + Vite admin portal

- Serves at `/` (root)
- **Setup wizard**: detects missing `PROXY_API_KEY` and/or AI Integrations, shows status for all 4 providers
- **Version & Update panel**: auto-checks `/api/version`, shows changelog, update detection
- **Model Groups panel**: per-group and per-model enable/disable toggles
- Service health indicator (polls `/api/healthz` every 30s)
- Dark theme `hsl(222,47%,11%)`

## Key API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/healthz` | No | Health check |
| GET | `/api/setup-status` | No | Returns configured + integrations status for all 4 providers |
| GET | `/api/version` | No | Current version + optional remote update check |
| GET | `/api/v1/models` | Yes | List enabled models |
| POST | `/api/v1/chat/completions` | Yes | Proxy chat completions (streaming/non-streaming) |
| GET | `/api/admin/model-groups` | No | Read model group config |
| POST | `/api/admin/model-groups` | Yes | Write model group config |

## Environment Variables

- `PROXY_API_KEY` — Unified auth key (set manually via Replit Secrets)
- `AI_INTEGRATIONS_ANTHROPIC_API_KEY` / `_BASE_URL` — Auto-injected by Replit (configured)
- `AI_INTEGRATIONS_OPENAI_API_KEY` / `_BASE_URL` — Auto-injected by Replit (configured)
- `AI_INTEGRATIONS_GEMINI_API_KEY` / `_BASE_URL` — Auto-injected by Replit (configured)
- `AI_INTEGRATIONS_OPENROUTER_API_KEY` / `_BASE_URL` — Auto-injected by Replit (configured)
- `SESSION_SECRET` — Reserved

## Key Commands

- `pnpm --filter @workspace/api-server run build` — Build the API server
- `pnpm --filter @workspace/api-server run dev` — Run API server locally
- `pnpm --filter @workspace/api-portal run dev` — Run portal locally
