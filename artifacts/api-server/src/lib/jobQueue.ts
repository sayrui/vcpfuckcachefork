/**
 * Background Job Queue — streaming relay that decouples the upstream AI call
 * from the client HTTP connection.
 *
 * Problem: Replit's autoscale proxy has a hard 300-second request timeout.
 * Any SSE connection that stays alive longer than 300 s gets forcibly closed.
 *
 * Solution:
 *   1. Client calls POST /v1/jobs  →  job starts in background, response is
 *      { job_id, status } returned IMMEDIATELY (< 100 ms).
 *   2. Client opens GET /v1/jobs/:id/stream  →  server replays buffered chunks
 *      and live-streams new ones.  Event IDs are "<jobId>:<chunkIndex>".
 *   3. If Replit's proxy cuts the connection at 300 s, the client reconnects
 *      with "Last-Event-ID: <jobId>:<lastSeen>" — standard SSE reconnect.
 *      The server resumes from lastSeen+1.  No tokens are lost.
 *   4. The upstream AI call runs entirely in the background (Node.js I/O event
 *      loop keeps it alive regardless of client connection state).
 *
 * Memory: chunks are plain JSON strings.  Large responses (say 100 k tokens)
 * cost ~1 MB — acceptable for an in-process proxy.
 * GC: jobs expire after JOB_TTL_MS of inactivity (default 20 min).
 */

import { EventEmitter } from "events";
import type { Response } from "express";

export interface Job {
  id: string;
  model: string;
  chunks: string[];   // serialised JSON for each SSE data value
  done: boolean;
  error: string | null;
  emitter: EventEmitter;
  createdAt: number;
  lastAccessAt: number;
}

const jobs = new Map<string, Job>();
const JOB_TTL_MS = 20 * 60 * 1000;   // 20 minutes
const GC_INTERVAL_MS = 5 * 60 * 1000; // check every 5 minutes

// Periodic GC — unref() so this timer doesn't keep the process alive on exit.
setInterval(() => {
  const cutoff = Date.now() - JOB_TTL_MS;
  for (const [id, job] of jobs) {
    if (job.lastAccessAt < cutoff) jobs.delete(id);
  }
}, GC_INTERVAL_MS).unref();

/** Create a new job and register it in the store. */
export function createJob(model: string): Job {
  const id = `job-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const job: Job = {
    id,
    model,
    chunks: [],
    done: false,
    error: null,
    emitter: new EventEmitter(),
    createdAt: Date.now(),
    lastAccessAt: Date.now(),
  };
  job.emitter.setMaxListeners(32); // allow many concurrent stream reconnects
  jobs.set(id, job);
  return job;
}

/** Look up an existing job. */
export function getJob(id: string): Job | undefined {
  const job = jobs.get(id);
  if (job) job.lastAccessAt = Date.now();
  return job;
}

/** Append one SSE data payload (any serialisable value) to the job buffer. */
export function appendJobChunk(job: Job, data: unknown): void {
  const json = JSON.stringify(data);
  job.chunks.push(json);
  job.emitter.emit("chunk");
}

/** Mark a job as successfully finished. */
export function finishJob(job: Job): void {
  job.done = true;
  job.emitter.emit("done");
}

/** Mark a job as failed. */
export function failJob(job: Job, error: string): void {
  job.error = error;
  job.done = true;
  job.emitter.emit("done");
}

/**
 * Stream job chunks to an SSE response.
 *
 * @param res       Express response (SSE headers must be set by caller).
 * @param job       The job to stream.
 * @param fromIdx   Resume from this chunk index (0 = start from beginning).
 *
 * Returns a promise that resolves when:
 *   - the job is done and all chunks have been sent, OR
 *   - the client disconnects (res.writableEnded / 'close' event).
 *
 * The caller is responsible for writing "data: [DONE]\n\n" only when the
 * job is actually finished — not on client disconnect.  This function
 * writes [DONE] automatically only on true job completion.
 */
export function streamJobToResponse(
  res: Response,
  job: Job,
  fromIdx: number,
): Promise<void> {
  return new Promise<void>((resolve) => {
    let nextIdx = fromIdx;

    function sendPending(): boolean {
      while (nextIdx < job.chunks.length) {
        if (res.writableEnded) return false;
        // Event id encodes both jobId and chunk index so clients can resume.
        res.write(`id: ${job.id}:${nextIdx}\ndata: ${job.chunks[nextIdx]}\n\n`);
        nextIdx++;
      }
      return true; // still going
    }

    function finish() {
      if (!res.writableEnded) {
        if (job.error) {
          res.write(
            `data: ${JSON.stringify({ error: { message: job.error, type: "job_error" } })}\n\n`,
          );
        }
        res.write("data: [DONE]\n\n");
      }
    }

    // Send any chunks already in the buffer immediately.
    if (!sendPending() || job.done) {
      if (job.done) finish();
      resolve();
      return;
    }

    // Set up listeners for future chunks / job completion / client disconnect.
    const onChunk = () => {
      if (res.writableEnded) { cleanup(); resolve(); return; }
      sendPending();
    };

    const onDone = () => {
      if (!res.writableEnded) sendPending();
      finish();
      cleanup();
      resolve();
    };

    const onClose = () => { cleanup(); resolve(); };

    function cleanup() {
      job.emitter.removeListener("chunk", onChunk);
      job.emitter.removeListener("done", onDone);
      res.removeListener("close", onClose);
    }

    job.emitter.on("chunk", onChunk);
    job.emitter.on("done", onDone);
    res.on("close", onClose);
  });
}

/**
 * Parse a Last-Event-ID header value of the form "<jobId>:<chunkIdx>".
 * Returns null if the format is unrecognised.
 */
export function parseLastEventId(
  raw: string | undefined,
): { jobId: string; lastIdx: number } | null {
  if (!raw) return null;
  const colon = raw.lastIndexOf(":");
  if (colon === -1) return null;
  const jobId = raw.slice(0, colon);
  const idx = parseInt(raw.slice(colon + 1), 10);
  if (!jobId || Number.isNaN(idx) || idx < 0) return null;
  return { jobId, lastIdx: idx };
}
