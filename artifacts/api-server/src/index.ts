import app from "./app";
import { logger } from "./lib/logger";
import { startUpdateChecker } from "./lib/updateChecker";
import { cacheReady } from "./lib/responseCache";

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error(
    "PORT environment variable is required but was not provided.",
  );
}

const port = Number(rawPort);

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

// Startup env checks -- warn but don't crash so the setup wizard can guide users
const missing: string[] = [];
if (!process.env["AI_INTEGRATIONS_ANTHROPIC_API_KEY"]) missing.push("AI_INTEGRATIONS_ANTHROPIC_API_KEY");
if (!process.env["AI_INTEGRATIONS_ANTHROPIC_BASE_URL"]) missing.push("AI_INTEGRATIONS_ANTHROPIC_BASE_URL");
if (!process.env["AI_INTEGRATIONS_OPENAI_API_KEY"]) missing.push("AI_INTEGRATIONS_OPENAI_API_KEY");
if (!process.env["AI_INTEGRATIONS_OPENAI_BASE_URL"]) missing.push("AI_INTEGRATIONS_OPENAI_BASE_URL");
if (!process.env["AI_INTEGRATIONS_GEMINI_API_KEY"]) missing.push("AI_INTEGRATIONS_GEMINI_API_KEY");
if (!process.env["AI_INTEGRATIONS_GEMINI_BASE_URL"]) missing.push("AI_INTEGRATIONS_GEMINI_BASE_URL");
if (!process.env["AI_INTEGRATIONS_OPENROUTER_API_KEY"]) missing.push("AI_INTEGRATIONS_OPENROUTER_API_KEY");
if (!process.env["AI_INTEGRATIONS_OPENROUTER_BASE_URL"]) missing.push("AI_INTEGRATIONS_OPENROUTER_BASE_URL");
if (missing.length > 0) {
  logger.warn(
    { missing },
    "Missing environment variables -- visit the portal setup wizard or ask the Replit AI assistant to configure them"
  );
}

// Wait for the cache to be restored from disk before accepting requests.
// This prevents the first few requests after a restart from experiencing
// false cache misses because the disk load hadn't completed yet.
await cacheReady;

const server = app.listen(port, (err) => {
  if (err) {
    logger.error({ err }, "Error listening on port");
    process.exit(1);
  }

  logger.info({ port }, "Server listening");

  // Start background version check (logs warning if update is available)
  startUpdateChecker();
});

server.headersTimeout   = 0;
server.requestTimeout   = 0;
server.timeout          = 0;
server.keepAliveTimeout = 0;
