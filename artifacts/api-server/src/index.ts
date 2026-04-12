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

// Wait for the cache to be restored from disk before accepting requests.
await cacheReady;

const server = app.listen(port, (err) => {
  if (err) {
    logger.error({ err }, "Error listening on port");
    process.exit(1);
  }

  logger.info({ port }, "Server listening");

  startUpdateChecker();
});

server.headersTimeout   = 0;
server.requestTimeout   = 0;
server.timeout          = 0;
server.keepAliveTimeout = 0;
