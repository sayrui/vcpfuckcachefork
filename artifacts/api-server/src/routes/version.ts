import { Router, type IRouter, type Request, type Response } from "express";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { findProjectRoot, resolveVersionUrl, compareVersions } from "../lib/versionUtils";

const router: IRouter = Router();

interface VersionFile {
  version: string;
  releaseDate?: string;
  changelog?: string[];
  history?: Array<{ version: string; releaseDate?: string; changelog?: string[] }>;
}

function readVersionFile(): VersionFile {
  try {
    const raw = readFileSync(resolve(findProjectRoot(), "version.json"), "utf-8");
    const parsed = JSON.parse(raw) as VersionFile;
    if (parsed.version) return parsed;
  } catch { /* fall through */ }
  return { version: "unknown" };
}

// -- GET /api/version ---------------------------------------------------------

router.get("/version", async (_req: Request, res: Response) => {
  const local = readVersionFile();
  const updateSourceUrl = process.env["UPDATE_SOURCE_URL"] ?? "https://github.com/ygffgiwkfgywgef8-ai/vcrep";

  let remote: { version: string; releaseDate?: string; changelog?: string[] } | null = null;
  let updateAvailable = false;

  try {
    const versionUrl = resolveVersionUrl(updateSourceUrl);
    const r = await fetch(versionUrl);
    if (r.ok) {
      const body = await r.json() as { current?: typeof remote };
      remote = (body.current ?? body) as typeof remote;
      if (remote && local.version !== "unknown") {
        updateAvailable = compareVersions(remote.version ?? "0", local.version) > 0;
      }
    }
  } catch { /* remote check failed — not a hard error */ }

  res.json({ current: local, remote, updateAvailable, updateSourceUrl });
});

// -- GET /api/source.tar.gz ---------------------------------------------------
// Serves the project source as a gzipped tarball (no auth — code is public).
// Excludes node_modules, dist, .git, .local, secrets, and log files.

router.get("/source.tar.gz", (_req: Request, res: Response) => {
  const root = findProjectRoot();
  res.setHeader("Content-Type", "application/gzip");
  res.setHeader("Content-Disposition", "attachment; filename=source.tar.gz");

  const tar = spawn("tar", [
    "-czf", "-",
    "--exclude=node_modules",
    "--exclude=.git",
    "--exclude=dist",
    "--exclude=.local",
    "--exclude=.cache",
    "--exclude=.env",
    "--exclude=*.log",
    "--exclude=*.secret",
    "-C", root, ".",
  ]);

  tar.stdout.pipe(res);
  tar.stderr.on("data", () => { /* suppress tar warnings */ });
  tar.on("error", () => { if (!res.headersSent) res.status(500).end("tar failed"); });
  res.on("close", () => tar.kill());
});

export default router;
