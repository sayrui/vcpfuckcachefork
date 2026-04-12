import { Router, type IRouter, type Request, type Response } from "express";
import { authMiddleware } from "../middlewares/auth";
import { readGroups, writeGroups, type ModelGroup } from "../lib/modelGroups";

const router: IRouter = Router();

// GET /api/admin/model-groups -- read current config (no auth: read-only public info)
router.get("/admin/model-groups", (_req: Request, res: Response) => {
  res.json(readGroups());
});

// POST /api/admin/model-groups -- full replace (auth required)
router.post("/admin/model-groups", authMiddleware, (req: Request, res: Response) => {
  const groups = req.body as ModelGroup[];
  if (!Array.isArray(groups) || groups.length === 0) {
    res.status(400).json({ error: { message: "Expected non-empty array of groups", type: "invalid_request_error" } });
    return;
  }
  writeGroups(groups);
  res.json({ ok: true, groups: readGroups() });
});

export default router;
