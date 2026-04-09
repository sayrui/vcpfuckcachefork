import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { logger } from "./logger";
import { findProjectRoot, resolveVersionUrl, compareVersions } from "./versionUtils";

// -- Shared update state (read by app.ts to inject response headers) -----------

export const updateState = {
  currentVersion: "unknown",
  latestVersion: null as string | null,
  updateAvailable: false,
};

// -- Helpers -------------------------------------------------------------------

function readLocalVersion(): string {
  try {
    const raw = readFileSync(resolve(findProjectRoot(), "version.json"), "utf-8");
    const parsed = JSON.parse(raw) as { version?: string };
    return parsed.version ?? "unknown";
  } catch {
    return "unknown";
  }
}

// -- Check function -------------------------------------------------------------

async function checkForUpdate(): Promise<void> {
  const sourceUrl = process.env["UPDATE_SOURCE_URL"] ?? "https://github.com/ygffgiwkfgywgef8-ai/vcrep";
  const local = readLocalVersion();
  updateState.currentVersion = local;

  try {
    const versionUrl = resolveVersionUrl(sourceUrl);
    const r = await fetch(versionUrl);
    if (!r.ok) return;
    const body = await r.json() as { current?: { version?: string }; version?: string };
    const remoteVersion = body.current?.version ?? (body as { version?: string }).version ?? null;
    if (!remoteVersion) return;

    updateState.latestVersion = remoteVersion;
    updateState.updateAvailable = local !== "unknown" && compareVersions(remoteVersion, local) > 0;

    if (updateState.updateAvailable) {
      logger.warn(
        { currentVersion: local, latestVersion: remoteVersion, portalUrl: sourceUrl },
        `[UP]  AI Monorepo update available: v${local} -> v${remoteVersion}. ` +
        `Open the portal and copy the Agent update prompt to apply the update.`
      );
    } else {
      logger.info({ version: local }, "AI Monorepo is up to date");
    }
  } catch {
    // Remote check failed — not a hard error, silently skip
  }
}

// -- Startup + periodic checker ------------------------------------------------

const CHECK_INTERVAL_MS = 60 * 60 * 1000; // 1 hour

export function startUpdateChecker(): void {
  // Immediate check after a short delay so the server finishes starting first
  setTimeout(() => { void checkForUpdate(); }, 3000);
  setInterval(() => { void checkForUpdate(); }, CHECK_INTERVAL_MS);
}
