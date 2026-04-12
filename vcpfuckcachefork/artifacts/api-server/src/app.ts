import express, { type Express, type Request, type Response, type NextFunction } from "express";
import cors from "cors";
import pinoHttp from "pino-http";
import router from "./routes";
import { logger } from "./lib/logger";
import { updateState } from "./lib/updateChecker";

const app: Express = express();

app.use(
  pinoHttp({
    logger,
    serializers: {
      req(req) {
        return {
          id: req.id,
          method: req.method,
          url: req.url?.split("?")[0],
        };
      },
      res(res) {
        return {
          statusCode: res.statusCode,
        };
      },
    },
  }),
);
app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

// Inject version/update headers on all /api/v1/* responses so clients can
// detect available updates without opening the portal.
app.use("/api/v1", (_req: Request, res: Response, next: NextFunction) => {
  res.setHeader("X-Proxy-Version", updateState.currentVersion);
  if (updateState.updateAvailable && updateState.latestVersion) {
    res.setHeader("X-Proxy-Update-Available", "true");
    res.setHeader("X-Proxy-Latest-Version", updateState.latestVersion);
  }
  next();
});

app.use("/api", router);

export default app;
