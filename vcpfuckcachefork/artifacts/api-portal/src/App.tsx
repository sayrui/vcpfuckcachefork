import { useState, useEffect, useCallback } from "react";

const BG = "hsl(222,47%,11%)";
const CARD = "hsl(222,47%,15%)";
const CARD2 = "hsl(222,47%,18%)";
const BORDER = "hsl(222,47%,22%)";
const TEXT = "hsl(213,31%,91%)";
const MUTED = "hsl(215,20%,55%)";
const GREEN = "hsl(142,71%,45%)";
const RED = "hsl(0,84%,60%)";
const ACCENT = "hsl(217,91%,60%)";
const YELLOW = "hsl(43,96%,56%)";
const PURPLE = "hsl(262,83%,68%)";


// ── Types ──────────────────────────────────────────────────────────────

interface SetupStatus {
  configured: boolean;
  integrations: {
    anthropic: boolean;
    openai: boolean;
    gemini: boolean;
    openrouter: boolean;
    allReady: boolean;
  };
}

interface VersionInfo {
  version: string;
  releaseDate?: string;
  changelog?: string[];
  history?: Array<{ version: string; releaseDate?: string; changelog?: string[] }>;
}

interface VersionResponse {
  current: VersionInfo;
  remote: VersionInfo | null;
  updateAvailable: boolean;
  updateSourceUrl: string | null;
}

type AppState = "checking" | "needs-setup" | "ready";
type UpdateState = "idle" | "checking" | "up-to-date" | "available" | "error";

// ── Config Incomplete Banner (sticky) ────────────────────────────────────────
function ConfigBanner({ status, onGoToSetup }: { status: SetupStatus; onGoToSetup: () => void }) {
  const missing: string[] = [];
  if (!status.integrations.anthropic) missing.push("Anthropic Integration");
  if (!status.integrations.openai) missing.push("OpenAI Integration");
  if (!status.integrations.gemini) missing.push("Gemini Integration");
  if (!status.integrations.openrouter) missing.push("OpenRouter Integration");
  if (missing.length === 0) return null;

  return (
    <div style={{
      position: "fixed", top: 0, left: 0, right: 0, zIndex: 9999,
      background: `linear-gradient(90deg, ${RED}ee, #b91c1c)`,
      borderBottom: `2px solid ${RED}`,
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "0 16px", gap: 12, minHeight: 48,
      boxShadow: `0 2px 16px ${RED}60`,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flex: 1, minWidth: 0 }}>
        <span style={{ fontSize: 18, flexShrink: 0 }}>⚠</span>
        <div style={{ fontSize: 12.5, color: "#fff", fontWeight: 600, lineHeight: 1.4 }}>
          配置未完成，代理暂无法使用
          <span style={{ fontWeight: 400, marginLeft: 8, opacity: 0.85 }}>
            缺少：{missing.join(" · ")}
          </span>
        </div>
      </div>
      <button
        onClick={onGoToSetup}
        style={{
          padding: "6px 16px", borderRadius: 6, border: "2px solid rgba(255,255,255,0.9)",
          background: "rgba(255,255,255,0.15)", color: "#fff",
          fontSize: 12.5, fontWeight: 700, cursor: "pointer", flexShrink: 0,
          backdropFilter: "blur(4px)", transition: "background 0.15s",
          letterSpacing: "0.02em",
        }}
        onMouseEnter={e => (e.currentTarget.style.background = "rgba(255,255,255,0.28)")}
        onMouseLeave={e => (e.currentTarget.style.background = "rgba(255,255,255,0.15)")}
      >
        立即配置 →
      </button>
    </div>
  );
}

// ── Shared helpers ─────────────────────────────────────────────────────

// ── Model Groups Panel ─────────────────────────────────────────────────────

interface ModelEntry { id: string; enabled: boolean; }
interface ModelGroupConfig { id: string; name: string; enabled: boolean; models: ModelEntry[]; }

function Toggle({ on, onChange, disabled }: { on: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <div
      onClick={() => !disabled && onChange(!on)}
      style={{
        width: 36, height: 20, borderRadius: 10,
        background: on && !disabled ? GREEN : BORDER,
        position: "relative", cursor: disabled ? "not-allowed" : "pointer",
        transition: "background 0.2s", flexShrink: 0, opacity: disabled ? 0.5 : 1,
      }}
    >
      <div style={{
        position: "absolute", top: 2, left: on ? 18 : 2,
        width: 16, height: 16, borderRadius: "50%", background: "#fff",
        transition: "left 0.2s",
      }} />
    </div>
  );
}

function ModelGroupsPanel() {
  const [groups, setGroups] = useState<ModelGroupConfig[] | null>(null);
  const [original, setOriginal] = useState<string>("");
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [apiKey, setApiKey] = useState(() => localStorage.getItem("proxy_admin_key") ?? "");
  const [showKey, setShowKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveResult, setSaveResult] = useState<"ok" | "err" | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/model-groups");
      if (!res.ok) throw new Error("fetch failed");
      const data = await res.json() as ModelGroupConfig[];
      setGroups(data);
      setOriginal(JSON.stringify(data));
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const dirty = groups ? JSON.stringify(groups) !== original : false;

  const toggleGroup = (gid: string, enabled: boolean) => {
    setGroups(g => g ? g.map(gr => gr.id === gid ? { ...gr, enabled } : gr) : g);
  };

  const toggleModel = (gid: string, mid: string, enabled: boolean) => {
    setGroups(g => g ? g.map(gr =>
      gr.id === gid
        ? { ...gr, models: gr.models.map(m => m.id === mid ? { ...m, enabled } : m) }
        : gr
    ) : g);
  };

  const toggleAllInGroup = (gid: string, enabled: boolean) => {
    setGroups(g => g ? g.map(gr =>
      gr.id === gid ? { ...gr, models: gr.models.map(m => ({ ...m, enabled })) } : gr
    ) : g);
  };

  const save = async () => {
    if (!apiKey.trim()) { setShowKey(true); return; }
    localStorage.setItem("proxy_admin_key", apiKey.trim());
    setSaving(true); setSaveResult(null);
    try {
      const res = await fetch("/api/admin/model-groups", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": `Bearer ${apiKey.trim()}` },
        body: JSON.stringify(groups),
      });
      if (res.ok) { setOriginal(JSON.stringify(groups)); setSaveResult("ok"); }
      else setSaveResult("err");
    } catch { setSaveResult("err"); }
    finally { setSaving(false); setTimeout(() => setSaveResult(null), 3000); }
  };

  if (!groups) {
    return (
      <div style={{ background: CARD, border: `1px solid ${BORDER}`, borderRadius: 10, padding: "20px 18px", color: MUTED, fontSize: 12 }}>
        正在加载模型组配置…
      </div>
    );
  }

  const saveColor = saveResult === "ok" ? GREEN : saveResult === "err" ? RED : ACCENT;

  return (
    <div style={{ background: CARD, border: `1px solid ${BORDER}`, borderRadius: 10, overflow: "hidden" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8, padding: "14px 18px", borderBottom: `1px solid ${BORDER}` }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: TEXT }}>模型组管理</span>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          {!dirty && <span style={{ fontSize: 11, color: MUTED }}>点击开关修改，再保存</span>}
          {dirty && showKey && (
            <input
              type="password"
              value={apiKey}
              onChange={e => setApiKey(e.target.value)}
              placeholder="PROXY_API_KEY"
              style={{
                padding: "5px 10px", borderRadius: 6, border: `1px solid ${BORDER}`,
                background: BG, color: TEXT, fontSize: 12, fontFamily: "monospace", outline: "none", width: 160,
              }}
              onKeyDown={e => e.key === "Enter" && save()}
            />
          )}
          {dirty && (
            <button
              onClick={save}
              disabled={saving}
              style={{
                padding: "5px 16px", borderRadius: 6, background: saveColor,
                border: "none", color: "#fff", fontSize: 12, fontWeight: 600,
                cursor: saving ? "not-allowed" : "pointer", transition: "background 0.2s",
              }}
            >
              {saving ? "保存中…" : saveResult === "ok" ? "✓ 已保存" : saveResult === "err" ? "✗ 失败" : "保存更改"}
            </button>
          )}
        </div>
      </div>

      {/* Group rows */}
      {groups.map(group => {
        const isOpen = !!expanded[group.id];
        const enabledCount = group.models.filter(m => m.enabled).length;
        return (
          <div key={group.id} style={{ borderBottom: `1px solid ${BORDER}` }}>
            {/* Group header row */}
            <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 18px", background: group.enabled ? "transparent" : `${RED}08` }}>
              <Toggle on={group.enabled} onChange={v => toggleGroup(group.id, v)} />
              <button
                onClick={() => setExpanded(e => ({ ...e, [group.id]: !e[group.id] }))}
                style={{ flex: 1, background: "none", border: "none", cursor: "pointer", display: "flex", alignItems: "center", gap: 8, padding: 0 }}
              >
                <span style={{ fontSize: 13, fontWeight: 600, color: group.enabled ? TEXT : MUTED }}>{group.name}</span>
                <span style={{ fontSize: 11, color: MUTED }}>
                  {group.enabled ? `${enabledCount} / ${group.models.length} 启用` : "整组停用"}
                </span>
                <span style={{ marginLeft: "auto", color: MUTED, fontSize: 11, transform: isOpen ? "rotate(180deg)" : "none", transition: "transform 0.15s" }}>▼</span>
              </button>
              {/* Batch: enable all / disable all */}
              {isOpen && (
                <div style={{ display: "flex", gap: 6 }}>
                  <button onClick={() => toggleAllInGroup(group.id, true)} style={{ padding: "3px 8px", borderRadius: 5, border: `1px solid ${GREEN}50`, background: `${GREEN}12`, color: GREEN, fontSize: 11, cursor: "pointer" }}>全开</button>
                  <button onClick={() => toggleAllInGroup(group.id, false)} style={{ padding: "3px 8px", borderRadius: 5, border: `1px solid ${RED}50`, background: `${RED}10`, color: RED, fontSize: 11, cursor: "pointer" }}>全关</button>
                </div>
              )}
            </div>

            {/* Model list */}
            {isOpen && (
              <div style={{ background: BG, padding: "4px 18px 10px" }}>
                {group.models.map(model => (
                  <div key={model.id} style={{
                    display: "flex", alignItems: "center", gap: 10,
                    padding: "5px 0", borderBottom: `1px solid ${BORDER}28`,
                    opacity: group.enabled ? 1 : 0.4,
                  }}>
                    <Toggle
                      on={model.enabled}
                      onChange={v => { if (group.enabled) toggleModel(group.id, model.id, v); }}
                      disabled={!group.enabled}
                    />
                    <span style={{ fontFamily: "monospace", fontSize: 11, color: model.enabled && group.enabled ? TEXT : MUTED }}>
                      {model.id}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function CopyBox({ value, label, inline }: { value: string; label?: string; inline?: boolean }) {
  const [copied, setCopied] = useState(false);
  const doCopy = () => { navigator.clipboard.writeText(value).then(() => { setCopied(true); setTimeout(() => setCopied(false), 2000); }); };

  if (inline) {
    return (
      <button onClick={doCopy} style={{
        padding: "3px 10px", borderRadius: 5, border: `1px solid ${BORDER}`,
        background: copied ? `${GREEN}20` : "transparent", color: copied ? GREEN : MUTED,
        fontSize: 11, cursor: "pointer", transition: "all 0.15s", whiteSpace: "nowrap",
      }}>{copied ? "已复制" : "复制代码"}</button>
    );
  }

  return (
    <div style={{ marginTop: 10 }}>
      {label && <div style={{ fontSize: 11, color: MUTED, marginBottom: 5, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase" }}>{label}</div>}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 8, background: BG, border: `1px solid ${BORDER}`, borderRadius: 8, padding: "10px 14px" }}>
        <code style={{ flex: 1, fontFamily: "monospace", fontSize: 12, color: ACCENT, lineHeight: 1.7, wordBreak: "break-all" }}>{value}</code>
        <button onClick={doCopy} style={{
          padding: "4px 12px", borderRadius: 6, border: `1px solid ${BORDER}`,
          background: copied ? `${GREEN}20` : "transparent", color: copied ? GREEN : MUTED,
          fontSize: 12, cursor: "pointer", flexShrink: 0, transition: "all 0.15s", whiteSpace: "nowrap",
        }}>{copied ? "已复制" : "复制"}</button>
      </div>
    </div>
  );
}

function StatusRow({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "7px 0", borderBottom: `1px solid ${BORDER}` }}>
      <span style={{
        width: 18, height: 18, borderRadius: "50%", fontSize: 10, fontWeight: 700,
        display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
        background: ok ? `${GREEN}20` : `${RED}18`, color: ok ? GREEN : RED, border: `1px solid ${ok ? GREEN : RED}40`,
      }}>{ok ? "✓" : "✗"}</span>
      <span style={{ fontSize: 13, color: ok ? TEXT : MUTED }}>{label}</span>
    </div>
  );
}

// ── Update Panel ───────────────────────────────────────────────────────

const GITHUB_URL = "https://github.com/ygffgiwkfgywgef8-ai/vcrep";

function buildAgentPrompt(remoteVersion: string): string {
  return `请帮我把 AI Monorepo 更新到最新版本（v${remoteVersion}）。

源代码仓库：${GITHUB_URL}

操作步骤（全程无需我介入）：

1. 先用 read 工具读取并记录 .replit 文件的完整内容（用于后续恢复）。

2. 在项目根目录执行 git 拉取：
   git fetch ${GITHUB_URL} main
   git -c core.hooksPath=/dev/null reset --hard FETCH_HEAD

3. 用 write 工具将步骤 1 中记录的 .replit 内容写回 .replit，确保 Replit 工作区配置不被覆盖。

4. 执行：
   pnpm install

5. 重启名为 "artifacts/api-server: API Server" 的 workflow。
   重启后服务会自动编译新代码并上线，无需任何额外操作。

完成后告知我。`;
}

function UpdatePanel() {
  const [state, setState] = useState<UpdateState>("idle");
  const [versionData, setVersionData] = useState<VersionResponse | null>(null);
  const [copied, setCopied] = useState(false);

  const checkUpdate = useCallback(async () => {
    setState("checking");
    try {
      const res = await fetch("/api/version");
      if (!res.ok) throw new Error("fetch failed");
      const data = await res.json() as VersionResponse;
      setVersionData(data);
      setState(data.updateAvailable ? "available" : "up-to-date");
    } catch {
      setState("error");
    }
  }, []);

  useEffect(() => { void checkUpdate(); }, [checkUpdate]);

  const current = versionData?.current;
  const remote = versionData?.remote;

  const agentPrompt = remote?.version ? buildAgentPrompt(remote.version) : "";

  const copyPrompt = async () => {
    try {
      await navigator.clipboard.writeText(agentPrompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch { /* ignore */ }
  };

  return (
    <div style={{ background: CARD, border: `1px solid ${BORDER}`, borderRadius: 10, overflow: "hidden" }}>
      {/* Header row */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "14px 18px", borderBottom: `1px solid ${BORDER}` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: TEXT }}>版本 & 更新</span>
          {current && (
            <span style={{ fontFamily: "monospace", fontSize: 11, color: PURPLE, background: `${PURPLE}18`, border: `1px solid ${PURPLE}30`, borderRadius: 12, padding: "2px 8px" }}>
              v{current.version}
            </span>
          )}
          {state === "available" && remote && (
            <span style={{ fontFamily: "monospace", fontSize: 11, color: GREEN, background: `${GREEN}15`, border: `1px solid ${GREEN}30`, borderRadius: 12, padding: "2px 8px" }}>
              → v{remote.version} 可用
            </span>
          )}
          {state === "up-to-date" && <span style={{ fontSize: 11, color: GREEN }}>✓ 已是最新</span>}
          {state === "error"      && <span style={{ fontSize: 11, color: RED   }}>✗ 检测失败</span>}
        </div>
        <button
          onClick={checkUpdate}
          disabled={state === "checking"}
          style={{
            padding: "5px 14px", borderRadius: 6, border: `1px solid ${BORDER}`,
            background: "transparent", color: state === "checking" ? MUTED : ACCENT,
            fontSize: 12, cursor: state === "checking" ? "not-allowed" : "pointer",
          }}
        >
          {state === "checking" ? "检测中…" : "检测更新"}
        </button>
      </div>

      {/* Remote changelog */}
      {state === "available" && remote?.changelog && (
        <div style={{ padding: "12px 18px", borderBottom: `1px solid ${BORDER}`, background: `${GREEN}06` }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: GREEN, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 8 }}>
            v{remote.version} 更新内容
          </div>
          {remote.changelog.map((item, i) => (
            <div key={i} style={{ fontSize: 12, color: MUTED, lineHeight: 1.6, paddingLeft: 8, borderLeft: `2px solid ${GREEN}40`, marginBottom: 4 }}>
              {item}
            </div>
          ))}
        </div>
      )}

      {/* Current changelog (idle / up-to-date) */}
      {(state === "idle" || state === "up-to-date") && current?.changelog && (
        <div style={{ padding: "12px 18px", borderBottom: `1px solid ${BORDER}` }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: MUTED, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 8 }}>当前版本内容</div>
          {current.changelog.map((item, i) => (
            <div key={i} style={{ fontSize: 12, color: MUTED, lineHeight: 1.6, paddingLeft: 8, borderLeft: `2px solid ${BORDER}`, marginBottom: 4 }}>
              {item}
            </div>
          ))}
        </div>
      )}

      {/* ── Update available: Agent prompt box ── */}
      {state === "available" && remote && (
        <div style={{ padding: "16px 18px", borderTop: `1px solid ${BORDER}` }}>
          {/* Explanation */}
          <div style={{
            fontSize: 12, color: MUTED, lineHeight: 1.65, marginBottom: 14,
            background: `${YELLOW}0c`, border: `1px solid ${YELLOW}28`,
            borderRadius: 8, padding: "10px 14px",
          }}>
            <span style={{ color: YELLOW, fontWeight: 700 }}>为什么需要这一步？</span>
            {"  "}服务器是编译型 Node.js 程序，新代码必须重新编译并重启才能生效，无法热更新。
            将下方提示词复制给 <strong style={{ color: TEXT }}>Replit Agent</strong> 即可完成整个更新流程。
          </div>

          {/* Prompt preview */}
          <div style={{
            background: BG, border: `1px solid ${BORDER}`, borderRadius: 8,
            padding: "12px 14px", fontFamily: "monospace", fontSize: 11,
            color: MUTED, lineHeight: 1.8, whiteSpace: "pre-wrap", wordBreak: "break-all",
            maxHeight: 220, overflowY: "auto", marginBottom: 10,
          }}>
            {agentPrompt}
          </div>

          {/* GitHub link + copy button */}
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noreferrer"
              style={{
                flex: 1, padding: "9px 0", borderRadius: 7,
                border: `1px solid ${BORDER}`, background: "transparent",
                color: ACCENT, fontSize: 12, fontWeight: 600,
                textAlign: "center", textDecoration: "none",
                display: "block",
              }}
            >
              ↗ 查看 GitHub 源码
            </a>
            <button
              onClick={copyPrompt}
              style={{
                flex: 2, padding: "9px 0", borderRadius: 7,
                background: copied ? GREEN : ACCENT,
                border: "none", color: "#fff",
                fontSize: 13, fontWeight: 700, cursor: "pointer",
                transition: "background 0.2s",
              }}
            >
              {copied ? "✓ 已复制！粘贴给 Agent 即可" : "复制更新提示词"}
            </button>
          </div>
        </div>
      )}

      {/* Version history */}
      {current?.history && current.history.length > 0 && (
        <div style={{ padding: "14px 18px", borderTop: `1px solid ${BORDER}` }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: MUTED, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 12 }}>版本历史 / Release History</div>
          {current.history.map((ver) => (
            <details key={ver.version} style={{ marginBottom: 8 }}>
              <summary style={{
                cursor: "pointer", fontSize: 12, color: MUTED, fontFamily: "monospace",
                listStyle: "none", display: "flex", alignItems: "center", gap: 8,
                padding: "4px 0", userSelect: "none",
              }}>
                <span style={{ color: TEXT, fontWeight: 600 }}>v{ver.version}</span>
                {ver.releaseDate && <span style={{ color: MUTED, fontSize: 11 }}>{ver.releaseDate}</span>}
                <span style={{ marginLeft: "auto", fontSize: 10, color: MUTED }}>▶</span>
              </summary>
              <div style={{ paddingLeft: 12, paddingTop: 6 }}>
                {(ver.changelog ?? []).map((item, i) => (
                  <div key={i} style={{
                    fontSize: 11, color: MUTED, lineHeight: 1.65,
                    paddingLeft: 8, borderLeft: `2px solid ${BORDER}`, marginBottom: 4,
                  }}>{item}</div>
                ))}
              </div>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Setup Wizard ────────────────────────────────────────────────────────

const LS_DISMISSED = "aim_setup_dismissed";

// Integration setup code — executed via Agent's code_execution tool
const INTEGRATION_CODE = `const integrations = [
  { label: "Anthropic",  providerSlug: "anthropic",  providerApiKeyEnvVarName: "AI_INTEGRATIONS_ANTHROPIC_API_KEY",  providerUrlEnvVarName: "AI_INTEGRATIONS_ANTHROPIC_BASE_URL" },
  { label: "OpenAI",     providerSlug: "openai",     providerApiKeyEnvVarName: "AI_INTEGRATIONS_OPENAI_API_KEY",     providerUrlEnvVarName: "AI_INTEGRATIONS_OPENAI_BASE_URL" },
  { label: "Gemini",     providerSlug: "gemini",     providerApiKeyEnvVarName: "AI_INTEGRATIONS_GEMINI_API_KEY",     providerUrlEnvVarName: "AI_INTEGRATIONS_GEMINI_BASE_URL" },
  { label: "OpenRouter", providerSlug: "openrouter", providerApiKeyEnvVarName: "AI_INTEGRATIONS_OPENROUTER_API_KEY", providerUrlEnvVarName: "AI_INTEGRATIONS_OPENROUTER_BASE_URL" },
];
for (const i of integrations) {
  console.log("Setting up " + i.label + "...");
  const r = await setupReplitAIIntegrations({ providerSlug: i.providerSlug, providerApiKeyEnvVarName: i.providerApiKeyEnvVarName, providerUrlEnvVarName: i.providerUrlEnvVarName });
  console.log(i.label + ":", JSON.stringify(r));
}
console.log("All integrations configured.");`;

function buildUnifiedPrompt(needsIntegrations: boolean): string {
  const lines: string[] = [
    "请帮我完成 AI Monorepo 的初始配置，按以下步骤执行（全程无需我介入）：",
    "",
    "【绝对禁止】不得向我索取任何第三方 API Key（OpenAI、Anthropic、Google、OpenRouter 等），Replit 平台会自动注入。",
    "",
  ];
  let step = 1;

  if (needsIntegrations) {
    lines.push(`第${step++}步：通过 code_execution 工具执行以下代码，依次配置全部 4 个 AI Integration：`);
    lines.push("");
    lines.push(INTEGRATION_CODE);
    lines.push("");
  }

  lines.push(`第${step}步：重启 API Server workflow，让以上配置全部生效。`);
  lines.push("");
  lines.push(`完成后告知我，我将点击门户中的\u201c重新检测\u201d按钮验证。`);

  return lines.join("\n");
}

type WizardPhase = "overview" | "copy-prompt" | "verify";

function SetupWizard({ status, onRecheck, onDismiss }: {
  status: SetupStatus;
  onRecheck: () => void;
  onDismiss?: () => void;
}) {
  const needsIntegrations = !status.integrations.allReady;

  const phases: WizardPhase[] = ["overview", "copy-prompt", "verify"];

  const [phaseIdx, setPhaseIdx] = useState(0);
  const [checking, setChecking] = useState(false);

  const phase = phases[phaseIdx];
  const totalSteps = phases.length - 1;
  const currentStep = phaseIdx;

  const unifiedPrompt = buildUnifiedPrompt(needsIntegrations);

  const goNext = () => setPhaseIdx(i => Math.min(i + 1, phases.length - 1));

  const handleRecheck = async () => {
    setChecking(true);
    await new Promise((r) => setTimeout(r, 600));
    onRecheck();
    setChecking(false);
  };

  const PrimaryBtn = ({ onClick, disabled, children }: { onClick: () => void; disabled?: boolean; children: React.ReactNode }) => (
    <button onClick={onClick} disabled={disabled} style={{
      width: "100%", padding: "11px 0", borderRadius: 8,
      background: disabled ? `${ACCENT}30` : ACCENT, border: "none", color: "#fff",
      fontSize: 14, fontWeight: 600, cursor: disabled ? "not-allowed" : "pointer",
      transition: "background 0.15s", marginTop: 4,
    }}>{children}</button>
  );

  return (
    <div style={{ minHeight: "100vh", background: BG, color: TEXT, fontFamily: "'Inter', system-ui, sans-serif", display: "flex", justifyContent: "center", padding: "48px 16px" }}>
      <div style={{ width: "100%", maxWidth: 500 }}>

        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 32 }}>
          <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, letterSpacing: "-0.02em" }}>AI Monorepo</h1>
          {phaseIdx === 0 ? (
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 12px", borderRadius: 20, fontSize: 12, fontWeight: 600, background: `${YELLOW}18`, color: YELLOW, border: `1px solid ${YELLOW}40` }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: YELLOW }} /> 需要配置
            </span>
          ) : (
            <span style={{ fontSize: 12, color: MUTED }}>第 {currentStep} / {totalSteps} 步</span>
          )}
        </div>

        {/* Step dots (visible on action steps) */}
        {phaseIdx > 0 && (
          <div style={{ display: "flex", gap: 6, marginBottom: 24, justifyContent: "center" }}>
            {phases.slice(1).map((_, i) => (
              <div key={i} style={{
                width: i + 1 === currentStep ? 20 : 7, height: 7, borderRadius: 4,
                background: i + 1 <= currentStep ? ACCENT : BORDER,
                transition: "all 0.25s",
              }} />
            ))}
          </div>
        )}

        {/* Phase card */}
        <div style={{ background: CARD, border: `1px solid ${BORDER}`, borderRadius: 12, padding: "24px", marginBottom: 16 }}>

          {/* ── OVERVIEW ── */}
          {phase === "overview" && (<>
            <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 6 }}>初次使用，需要简单配置</div>
            <div style={{ fontSize: 13, color: MUTED, lineHeight: 1.6, marginBottom: 20 }}>
              以下是当前配置状态。向导会逐步引导你完成，<strong style={{ color: TEXT }}>全程只需复制一段提示词发给 AI 助手即可</strong>。
            </div>
            <StatusRow ok={status.integrations.anthropic} label="Anthropic Integration — Claude 系列" />
            <StatusRow ok={status.integrations.openai} label="OpenAI Integration — GPT / o 系列" />
            <StatusRow ok={status.integrations.gemini} label="Gemini Integration — Gemini 系列" />
            <StatusRow ok={status.integrations.openrouter} label="OpenRouter Integration — Llama / Grok / DeepSeek 等" />
          </>)}

          {/* ── COPY PROMPT ── */}
          {phase === "copy-prompt" && (<>
            <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 6 }}>将提示词发给 Replit Assistant</div>
            <div style={{ fontSize: 13, color: MUTED, lineHeight: 1.6, marginBottom: 18 }}>
              点击复制，然后打开 <strong style={{ color: TEXT }}>Replit Assistant（右下角 AI 图标）</strong>，把内容粘贴发送。
              助手会自动完成 Integration 配置和服务器重启，无需你做任何其他操作。
            </div>
            <CopyBox value={unifiedPrompt} label="点击复制提示词" />
          </>)}

          {/* ── VERIFY ── */}
          {phase === "verify" && (<>
            <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 6 }}>等待助手完成后验证</div>
            <div style={{ fontSize: 13, color: MUTED, lineHeight: 1.7, marginBottom: 4 }}>
              助手通常需要 1–2 分钟完成配置和重启。完成后，点击下方按钮检测是否生效。
            </div>
            <div style={{ fontSize: 12, color: MUTED, lineHeight: 1.6 }}>
              检测通过后将自动跳转到主页面，无需手动刷新。
            </div>
          </>)}

        </div>

        {/* Action buttons */}
        {phase === "overview" && (
          <PrimaryBtn onClick={goNext}>开始配置 →</PrimaryBtn>
        )}
        {phase === "copy-prompt" && (
          <PrimaryBtn onClick={goNext}>已发送给助手，下一步 →</PrimaryBtn>
        )}
        {phase === "verify" && (
          <PrimaryBtn onClick={handleRecheck} disabled={checking}>
            {checking ? "检测中…" : "✓  重新检测配置"}
          </PrimaryBtn>
        )}

        {/* Back link (on non-overview steps) */}
        {phaseIdx > 0 && (
          <div style={{ textAlign: "center", marginTop: 14 }}>
            <button onClick={() => setPhaseIdx(i => Math.max(i - 1, 0))} style={{
              background: "none", border: "none", color: MUTED, fontSize: 12,
              cursor: "pointer", padding: "4px 8px", borderRadius: 4,
            }}>← 上一步</button>
          </div>
        )}

        {/* Skip link (on overview only) */}
        {phase === "overview" && onDismiss && (
          <div style={{ marginTop: 20, textAlign: "center" }}>
            <button onClick={onDismiss} style={{
              background: "none", border: "none", color: MUTED,
              fontSize: 12, cursor: "pointer", padding: "4px 8px",
              textDecoration: "underline", textUnderlineOffset: 3,
            }}>
              稍后再说，先进主界面
            </button>
          </div>
        )}

      </div>
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────

function MainPage({ status, onGoToSetup }: { status: SetupStatus; onGoToSetup?: () => void }) {
  const [health, setHealth] = useState<"online" | "offline" | "checking">("checking");
  const [copied, setCopied] = useState(false);
  const baseUrl = window.location.origin + "/api";

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch("/api/healthz");
      setHealth(res.ok ? "online" : "offline");
    } catch { setHealth("offline"); }
  }, []);

  useEffect(() => {
    checkHealth();
    const id = setInterval(checkHealth, 30000);
    return () => clearInterval(id);
  }, [checkHealth]);

  const statusColor = health === "online" ? GREEN : health === "offline" ? RED : MUTED;
  const statusText = health === "online" ? "Online" : health === "offline" ? "Offline" : "Checking…";

  const incomplete = onGoToSetup && (!status.configured || !status.integrations.allReady);

  return (
    <>
      {incomplete && onGoToSetup && <ConfigBanner status={status} onGoToSetup={onGoToSetup} />}
      <div style={{ minHeight: "100vh", background: BG, color: TEXT, fontFamily: "'Inter', system-ui, sans-serif", display: "flex", justifyContent: "center", padding: `${incomplete ? 88 : 40}px 16px 40px` }}>
      <div style={{ width: "100%", maxWidth: 560 }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, letterSpacing: "-0.02em" }}>AI Monorepo</h1>
          <span style={{
            display: "inline-flex", alignItems: "center", gap: 6, padding: "4px 12px", borderRadius: 20, fontSize: 12, fontWeight: 600,
            background: `${statusColor}18`, color: statusColor, border: `1px solid ${statusColor}40`,
          }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: statusColor }} /> {statusText}
          </span>
        </div>

        {/* Base URL */}
        {(() => {
          const host = window.location.hostname;
          const isDev = host === "localhost" || host.endsWith(".replit.dev") || host.endsWith(".spock.replit.dev") || host.endsWith(".kirk.replit.dev");
          return (
            <div style={{ background: CARD, borderRadius: 10, border: `1px solid ${isDev ? YELLOW : BORDER}`, padding: "16px 20px", marginBottom: 12 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", color: MUTED, textTransform: "uppercase" }}>Base URL</div>
                {isDev && (
                  <span style={{ fontSize: 10.5, fontWeight: 600, padding: "2px 8px", borderRadius: 4, background: `${YELLOW}18`, color: YELLOW, border: `1px solid ${YELLOW}40`, letterSpacing: "0.04em" }}>
                    临时链接
                  </span>
                )}
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontFamily: "monospace", fontSize: 13, color: ACCENT, flex: 1, wordBreak: "break-all" }}>{baseUrl}</span>
                <button onClick={() => { navigator.clipboard.writeText(baseUrl).then(() => { setCopied(true); setTimeout(() => setCopied(false), 2000); }); }} style={{
                  padding: "5px 12px", borderRadius: 6, border: `1px solid ${BORDER}`,
                  background: copied ? `${GREEN}20` : "transparent", color: copied ? GREEN : MUTED,
                  fontSize: 12, cursor: "pointer", flexShrink: 0, transition: "all 0.15s",
                }}>{copied ? "Copied!" : "Copy"}</button>
              </div>
              {isDev && (
                <div style={{ marginTop: 10, paddingTop: 10, borderTop: `1px solid ${BORDER}`, fontSize: 12, color: MUTED, lineHeight: 1.65 }}>
                  ⚠ 此链接为 <strong style={{ color: YELLOW }}>Replit 开发环境临时地址</strong>，每次重启后可能变化，且长期关闭后会失效。<br />
                  如需稳定、持久的 API 端点，请通过 Replit <strong style={{ color: TEXT }}>Publish（发布）</strong> 功能部署后使用
                  <code style={{ marginLeft: 4, fontFamily: "monospace", color: ACCENT, fontSize: 11 }}>*.replit.app</code> 域名。
                </div>
              )}
            </div>
          );
        })()}

        {/* Hint */}
        <div style={{ background: CARD2, borderRadius: 8, border: `1px solid ${BORDER}`, padding: "12px 16px", marginBottom: 24 }}>
          <div style={{ fontSize: 12, color: MUTED, lineHeight: 1.7 }}>
            使用此 Base URL，配合你的{" "}
            <code style={{ fontFamily: "monospace", color: TEXT, background: `${BORDER}80`, padding: "1px 5px", borderRadius: 4 }}>PROXY_API_KEY</code>
            {" "}即可接入任意兼容 OpenAI 格式的客户端。
          </div>
        </div>

        {/* Version & Update */}
        <div style={{ marginBottom: 20 }}>
          <UpdatePanel />
        </div>

        {/* Model Groups */}
        <div style={{ marginBottom: 20 }}>
          <ModelGroupsPanel />
        </div>
      </div>
    </div>
    </>
  );
}

// ── Root ───────────────────────────────────────────────────────────────

export default function App() {
  const [appState, setAppState] = useState<AppState>("checking");
  const [setupStatus, setSetupStatus] = useState<SetupStatus>({
    configured: false,
    integrations: { anthropic: false, openai: false, gemini: false, openrouter: false, allReady: false },
  });
  // dismissed = user explicitly chose to skip the wizard; persisted in localStorage
  const [dismissed, setDismissed] = useState(false);

  const handleDismiss = () => {
    localStorage.setItem(LS_DISMISSED, "1");
    setDismissed(true);
  };

  const handleGoToSetup = () => {
    localStorage.removeItem(LS_DISMISSED);
    setDismissed(false);
  };

  const checkSetup = useCallback(async () => {
    try {
      const res = await fetch("/api/setup-status");
      if (res.ok) {
        const data = await res.json() as SetupStatus;
        setSetupStatus(data);
        const allReady = data.configured && data.integrations.allReady;
        if (allReady) {
          // Fully configured — clear any dismissal and go to main page
          localStorage.removeItem(LS_DISMISSED);
          setDismissed(false);
          setAppState("ready");
        } else {
          // Not fully configured — check if user previously dismissed the wizard
          const wasDismissed = localStorage.getItem(LS_DISMISSED) === "1";
          setDismissed(wasDismissed);
          setAppState("needs-setup");
        }
      } else {
        setAppState("needs-setup");
      }
    } catch {
      setAppState("needs-setup");
    }
  }, []);

  useEffect(() => { checkSetup(); }, [checkSetup]);

  if (appState === "checking") {
    return (
      <div style={{ minHeight: "100vh", background: BG, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <span style={{ color: MUTED, fontSize: 14 }}>正在初始化…</span>
      </div>
    );
  }

  // Show wizard only when not configured AND user has not dismissed it
  if (appState === "needs-setup" && !dismissed) {
    return (
      <SetupWizard
        status={setupStatus}
        onRecheck={checkSetup}
        onDismiss={handleDismiss}
      />
    );
  }

  return (
    <MainPage
      status={setupStatus}
      onGoToSetup={dismissed ? handleGoToSetup : undefined}
    />
  );
}
