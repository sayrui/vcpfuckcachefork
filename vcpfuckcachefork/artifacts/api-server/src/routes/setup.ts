import { Router, type IRouter } from "express";

const router: IRouter = Router();

router.get("/setup-status", (_req, res) => {
  const keyConfigured = true; // hardcoded in auth middleware

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

  const allReady =
    anthropicConfigured && openaiConfigured && geminiConfigured && openrouterConfigured;

  res.json({
    configured: keyConfigured,
    integrations: {
      anthropic: anthropicConfigured,
      openai: openaiConfigured,
      gemini: geminiConfigured,
      openrouter: openrouterConfigured,
      allReady,
    },
  });
});

export default router;
