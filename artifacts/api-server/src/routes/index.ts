import { Router, type IRouter } from "express";
import healthRouter from "./health";
import setupRouter from "./setup";
import versionRouter from "./version";
import modelGroupsRouter from "./model-groups";
import modelsRouter from "./v1/models";
import chatRouter from "./v1/chat";
import jobsRouter from "./v1/jobs";
import adminRouter from "./admin";

const router: IRouter = Router();

router.use(healthRouter);
router.use(setupRouter);
router.use(versionRouter);
router.use(modelGroupsRouter);
router.use("/v1", modelsRouter);
router.use("/v1", chatRouter);
router.use("/v1/jobs", jobsRouter);
router.use(adminRouter);

export default router;
