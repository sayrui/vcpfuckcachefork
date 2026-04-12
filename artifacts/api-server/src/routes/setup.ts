import { Router, type IRouter } from "express";

const router: IRouter = Router();

router.get("/setup-status", (_req, res) => {
  const anthropicConfigured =
    Boolean(process.env["AI_INTEGRATIONS_ANTHROPIC_API_KEY"]) &&
    Boolean(process.env["AI_INTEGRATIONS_ANTHROPIC_BASE_URL"]);

  const openaiConfigured =
    Boolean(process.env["AI_INTEGRATIONS_OPENAI_API_KEY"]) &&
    Boolean(process.env["AI_INTEGRATIONS_OPENAI_BASE_URL"]);

  const geminiConfigured =
    Boolean(process.env["AI_INTEGRATIONS_GEMINI_API_KEY"]) &&
    Boolean(process.env["AI_INTEGRATIONS_GEMINI_BASE_URL"]);

  const openrouterConfigured =
    Boolean(process.env["AI_INTEGRATIONS_OPENROUTER_API_KEY"]) &&
    Boolean(process.env["AI_INTEGRATIONS_OPENROUTER_BASE_URL"]);

  res.json({
    configured: true,
    integrations: {
      anthropic: anthropicConfigured,
      openai: openaiConfigured,
      gemini: geminiConfigured,
      openrouter: openrouterConfigured,
      allReady: true,
    },
  });
});

export default router;
