import { Router, type IRouter } from "express";
import { authMiddleware } from "../../middlewares/auth";
import { getEnabledModelIds } from "../../lib/modelGroups";

const router: IRouter = Router();

const CREATED_AT = Math.floor(Date.now() / 1000);

// Only return models that are currently enabled in the model-groups config.
router.get("/models", authMiddleware, (_req, res) => {
  const data = getEnabledModelIds().map(id => ({ id, object: "model", created: CREATED_AT, owned_by: "proxy" }));
  res.json({ object: "list", data });
});

export default router;
