import { Router, type IRouter, type Request, type Response } from "express";
import { authMiddleware } from "../middlewares/auth";
import {
  getCacheStats,
  cacheClear,
  setCacheEnabled,
  setCacheTtl,
  setCacheMaxEntries,
} from "../lib/responseCache";

const router: IRouter = Router();

router.get("/admin/cache", authMiddleware, (_req: Request, res: Response) => {
  res.json(getCacheStats());
});

router.post("/admin/cache/clear", authMiddleware, (_req: Request, res: Response) => {
  cacheClear();
  res.json({ ok: true });
});

router.patch("/admin/cache", authMiddleware, (req: Request, res: Response) => {
  const { enabled, ttlMinutes, maxEntries } = req.body as {
    enabled?: boolean;
    ttlMinutes?: number;
    maxEntries?: number;
  };
  if (typeof enabled === "boolean") setCacheEnabled(enabled);
  if (typeof ttlMinutes === "number" && ttlMinutes > 0) setCacheTtl(ttlMinutes);
  if (typeof maxEntries === "number" && maxEntries > 0) setCacheMaxEntries(maxEntries);
  res.json({ ok: true, stats: getCacheStats() });
});

export default router;
