# AI Monorepo Proxy v1.2.0

## Overview

OpenAI-compatible AI reverse proxy that routes requests to Anthropic (Claude), OpenAI (GPT/o-series), Google Gemini, and OpenRouter. Supports tool calling, streaming, image recognition, and Claude thinking mode.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM (lib/db — not actively used by proxy)
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)
- **Frontend**: React + Vite (artifacts/api-portal)

## Artifacts

- `artifacts/api-server` — Express backend proxy server (port assigned by Replit, paths: `/api`)
- `artifacts/api-portal` — React frontend management portal (previewPath: `/`)

## Source

Cloned from: https://github.com/sayrui/vcpfuckcachefork (version 1.2.0)

## AI Integrations (Replit-managed, no user key required)

- `AI_INTEGRATIONS_ANTHROPIC_API_KEY` / `AI_INTEGRATIONS_ANTHROPIC_BASE_URL` — Claude series
- `AI_INTEGRATIONS_OPENAI_API_KEY` / `AI_INTEGRATIONS_OPENAI_BASE_URL` — GPT/o series
- `AI_INTEGRATIONS_GEMINI_API_KEY` / `AI_INTEGRATIONS_GEMINI_BASE_URL` — Gemini series
- `AI_INTEGRATIONS_OPENROUTER_API_KEY` / `AI_INTEGRATIONS_OPENROUTER_BASE_URL` — Llama/Grok/DeepSeek etc.

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-server run build` — build API server
- `pnpm --filter @workspace/api-server run dev` — run API server locally
- `pnpm --filter @workspace/api-portal run dev` — run frontend locally

## Features

- OpenAI-compatible API proxy (`/api/v1/chat/completions`, `/api/v1/models`)
- Anthropic Messages API compat (`/api/v1/messages`)
- Gemini Native API compat (`/api/v1beta/models/:model:generateContent`)
- Model group management with enable/disable per model
- Response caching (disk-backed)
- Setup wizard for first-time configuration
- Version panel with auto-update detection
- Fleet manager for multiple proxy nodes
- PROXY_API_KEY authentication support
