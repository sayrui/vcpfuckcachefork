import { Router, type IRouter } from "express";
import healthRouter from "./health";
import setupRouter from "./setup";
import versionRouter from "./version";
import modelGroupsRouter from "./model-groups";
import modelsRouter from "./v1/models";
import chatRouter from "./v1/chat";
import jobsRouter from "./v1/jobs";
import adminRouter from "./admin";
import { anthropicCompatRouter, geminiCompatRouter } from "./compat";

const router: IRouter = Router();

router.use(healthRouter);
router.use(setupRouter);
router.use(versionRouter);
router.use(modelGroupsRouter);
router.use("/v1", modelsRouter);
router.use("/v1", chatRouter);
router.use("/v1/jobs", jobsRouter);
router.use(adminRouter);

// Multi-format compatibility adapters
// Anthropic Messages API: POST /api/v1/messages
router.use("/v1", anthropicCompatRouter);
// Gemini Native API:     POST /api/v1beta/models/:model:generateContent
//                        POST /api/v1beta/models/:model:streamGenerateContent
router.use("/v1beta", geminiCompatRouter);

export default router;
