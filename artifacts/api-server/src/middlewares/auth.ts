import { type Request, type Response, type NextFunction } from "express";

const PROXY_API_KEY = "vcspeeper";

export function authMiddleware(req: Request, res: Response, next: NextFunction): void {
  const authHeader = req.headers["authorization"];
  const xApiKey = req.headers["x-api-key"];
  const googleKey = req.headers["x-goog-api-key"];
  const queryKey = req.query["key"];

  let providedKey: string | undefined;

  if (authHeader && authHeader.startsWith("Bearer ")) {
    providedKey = authHeader.slice(7);
  } else if (typeof xApiKey === "string" && xApiKey) {
    providedKey = xApiKey;
  } else if (typeof googleKey === "string" && googleKey) {
    providedKey = googleKey;
  } else if (typeof queryKey === "string" && queryKey) {
    providedKey = queryKey;
  }

  if (!providedKey || providedKey !== PROXY_API_KEY) {
    res.status(401).json({ error: { message: "Invalid or missing API key", type: "invalid_request_error", code: "invalid_api_key" } });
    return;
  }

  next();
}
