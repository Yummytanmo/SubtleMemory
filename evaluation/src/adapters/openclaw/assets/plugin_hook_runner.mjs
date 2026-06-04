#!/usr/bin/env node
/**
 * Minimal OpenClaw plugin hook runner for evaluation adapters.
 *
 * Payload: JSON file with pluginPath, pluginId, mode, action, hookName,
 * pluginConfig, openclawConfig, event, ctx, and optional context-engine fields.
 */

import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const SECRET_KEY_RE = /(authorization|api[-_]?key|token|secret|password|bearer|credential)/i;
const SECRET_VALUE_RE =
  /(sk-[A-Za-z0-9_-]{12,}|m0-[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._-]+|Token\s+[A-Za-z0-9._-]+)/g;
const OPENCLAW_ROOT = process.env.OPENCLAW_ROOT;
if (!OPENCLAW_ROOT) {
  throw new Error("OPENCLAW_ROOT is not configured. Set it in .env or the shell.");
}

function buildOpenClawSdkAlias() {
  const alias = {};
  const sdkDir = path.join(OPENCLAW_ROOT, "dist", "plugin-sdk");
  alias["openclaw/plugin-sdk"] = path.join(sdkDir, "index.js");
  alias["@openclaw/plugin-sdk"] = path.join(sdkDir, "index.js");
  for (const name of [
    "plugin-entry",
    "core",
    "index",
    "group-access",
    "compat",
    "config-schema",
    "memory-core",
    "memory-core-engine-runtime",
    "memory-core-host-engine-qmd",
    "memory-core-host-query",
    "memory-core-host-multimodal",
    "memory-host-status",
    "memory-host-markdown",
    "runtime-env",
    "runtime-store",
    "host-runtime",
  ]) {
    const target = path.join(sdkDir, `${name}.js`);
    alias[`openclaw/plugin-sdk/${name}`] = target;
    alias[`@openclaw/plugin-sdk/${name}`] = target;
  }
  return alias;
}

function redactString(value) {
  return String(value ?? "").replace(SECRET_VALUE_RE, (match) => `${match.slice(0, 8)}...[REDACTED]`);
}

function redact(value, depth = 0) {
  if (depth > 8) return "[TRUNCATED]";
  if (value == null) return value;
  if (typeof value === "string") return redactString(value);
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (Array.isArray(value)) return value.slice(0, 50).map((item) => redact(item, depth + 1));
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, child]) => [
        key,
        SECRET_KEY_RE.test(key) ? "[REDACTED]" : redact(child, depth + 1),
      ]),
    );
  }
  return String(value);
}

function summarizePayload(value) {
  const redacted = redact(value);
  const text = JSON.stringify(redacted);
  if (!text || text.length <= 4000) return redacted;
  return { truncated: true, char_count: text.length, preview: text.slice(0, 4000) };
}

function resolveEnvRefs(value, depth = 0) {
  if (depth > 8 || value == null) return value;
  if (typeof value === "string") {
    return value.replace(/\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}/g, (_match, key, fallback) => {
      const resolved = process.env[key];
      return resolved != null && resolved !== "" ? resolved : fallback ?? "";
    });
  }
  if (Array.isArray(value)) return value.map((item) => resolveEnvRefs(item, depth + 1));
  if (typeof value === "object") {
    if (
      value.source === "env" &&
      typeof value.id === "string" &&
      (value.provider === "default" || value.provider == null)
    ) {
      return process.env[value.id] || "";
    }
    return Object.fromEntries(
      Object.entries(value).map(([key, child]) => [key, resolveEnvRefs(child, depth + 1)]),
    );
  }
  return value;
}

function safeJsonParse(text) {
  if (!text) return undefined;
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function normalizeHeaders(headers) {
  const out = {};
  if (!headers) return out;
  if (typeof headers.forEach === "function") {
    headers.forEach((value, key) => {
      out[key] = SECRET_KEY_RE.test(key) ? "[REDACTED]" : redactString(value);
    });
    return out;
  }
  for (const [key, value] of Object.entries(headers)) {
    out[key] = SECRET_KEY_RE.test(key) ? "[REDACTED]" : redactString(value);
  }
  return out;
}

function patchFetch(fetchRecords) {
  const originalFetch = globalThis.fetch;
  if (typeof originalFetch !== "function") return () => {};
  globalThis.fetch = async (input, init = {}) => {
    const startedAt = Date.now();
    const url = typeof input === "string" ? input : input?.url;
    let requestBody = init?.body;
    if (typeof requestBody === "string") requestBody = safeJsonParse(requestBody) ?? requestBody;
    const record = {
      method: String(init?.method || input?.method || "GET").toUpperCase(),
      url: redactString(url),
      request_headers: normalizeHeaders(init?.headers || input?.headers),
      request_payload_summary: summarizePayload(requestBody),
    };
    fetchRecords.push(record);
    if (
      process.env.OPENCLAW_PLUGIN_EVAL_HOOK_RUNNER === "1" &&
      /127\.0\.0\.1:18789\/ready|localhost:18789\/ready/.test(String(url || ""))
    ) {
      record.status = 204;
      record.ok = true;
      record.response_headers = {};
      record.response_payload_summary = "";
      record.timing_ms = Date.now() - startedAt;
      return new Response("", { status: 204 });
    }
    try {
      const response = await originalFetch(input, init);
      const text = await response.clone().text().catch(() => "");
      record.status = response.status;
      record.ok = response.ok;
      record.response_headers = normalizeHeaders(response.headers);
      record.response_payload_summary = summarizePayload(safeJsonParse(text) ?? text.slice(0, 4000));
      record.timing_ms = Date.now() - startedAt;
      return response;
    } catch (error) {
      record.error = redactString(error?.message || String(error));
      record.timing_ms = Date.now() - startedAt;
      throw error;
    }
  };
  return () => {
    globalThis.fetch = originalFetch;
  };
}

async function resolvePluginEntry(pluginPath) {
  const stat = await fs.stat(pluginPath);
  if (!stat.isDirectory()) return pluginPath;
  for (const manifestName of ["openclaw.plugin.json", "package.json"]) {
    const manifestPath = path.join(pluginPath, manifestName);
    try {
      const manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
      const main =
        manifest.openclaw?.extensions?.[0] ||
        manifest.main ||
        manifest.module ||
        manifest.exports?.["."]?.import;
      if (main) return path.resolve(pluginPath, main);
    } catch {
      /* try next */
    }
  }
  throw new Error(`Unable to resolve plugin entry from ${pluginPath}`);
}

async function importPluginModule(entry, pluginPath) {
  const shouldUseJiti =
    /\.(ts|mts|cts)$/i.test(entry) || String(pluginPath || "").includes("/mem0/openclaw");
  if (!shouldUseJiti) return import(pathToFileURL(entry).href);
  const requireFromOpenClaw = createRequire(path.join(OPENCLAW_ROOT, "package.json"));
  const { createJiti } = requireFromOpenClaw("jiti");
  const jiti = createJiti(entry, {
    tryNative: false,
    interopDefault: true,
    alias: buildOpenClawSdkAlias(),
  });
  return jiti(entry);
}

async function loadPlugin(pluginPath) {
  const entry = await resolvePluginEntry(pluginPath);
  const mod = await importPluginModule(entry, pluginPath);
  const candidate = mod.default ?? mod.plugin ?? mod.memoryPlugin ?? mod.register ?? mod.activate ?? mod;
  const register =
    typeof candidate === "function"
      ? candidate
      : candidate && typeof candidate.register === "function"
        ? candidate.register.bind(candidate)
        : typeof mod.register === "function"
          ? mod.register
          : typeof mod.activate === "function"
            ? mod.activate
            : null;
  if (!register) throw new Error(`Plugin ${pluginPath} has no register/activate/default export`);
  return { register, entry };
}

function makeLogger(logs, level) {
  return (...args) =>
    logs.push({
      level,
      message: args
        .map((arg) => redactString(typeof arg === "string" ? arg : JSON.stringify(redact(arg))))
        .join(" "),
    });
}

function buildApi(payload, registrations, logs) {
  const registerHook = (events, handler, opts = {}) => {
    const names = Array.isArray(events) ? events : [events];
    for (const name of names) registrations.hooks.push({ name, handler, opts });
  };
  const unsupportedRegistration = (method) => (...args) => {
    const registration = {
      method,
      args: summarizePayload(args),
    };
    registrations.unsupported.push(registration);
    logs.push({
      level: "warn",
      message: `${method} is unsupported by the evaluation hook runner and was not registered`,
    });
  };
  const runtime = {
    stateDir: payload.stateDir || process.env.OPENCLAW_STATE_DIR,
    workspaceDir: payload.workspaceDir || process.env.OPENCLAW_WORKSPACE_DIR,
    configPath: payload.configPath || process.env.OPENCLAW_CONFIG_PATH,
    sessionId: payload.ctx?.sessionId || payload.sessionId,
    sessionKey: payload.ctx?.sessionKey || payload.sessionKey,
  };
  return {
    id: payload.pluginId,
    name: payload.pluginId,
    rootDir: payload.pluginPath,
    config: payload.openclawConfig || {},
    pluginConfig: resolveEnvRefs(payload.pluginConfig || {}),
    runtime,
    logger: {
      info: makeLogger(logs, "info"),
      warn: makeLogger(logs, "warn"),
      error: makeLogger(logs, "error"),
      debug: makeLogger(logs, "debug"),
    },
    resolvePath: (input) => path.resolve(payload.pluginPath || process.cwd(), input),
    registerHook,
    on: registerHook,
    registerContextEngine: (id, factory) => registrations.contextEngines.push({ id, factory }),
    registerService: (service) => registrations.services.push(service),
    registerTool: unsupportedRegistration("registerTool"),
    registerCommand: unsupportedRegistration("registerCommand"),
    registerCli: unsupportedRegistration("registerCli"),
    registerMemoryRuntime: unsupportedRegistration("registerMemoryRuntime"),
    registerMemoryPromptSection: unsupportedRegistration("registerMemoryPromptSection"),
    registerProvider: unsupportedRegistration("registerProvider"),
  };
}

function assertSupportedRegistrationAvailable(registrations) {
  if (registrations.hooks.length || registrations.contextEngines.length) return;
  if (!registrations.unsupported.length) return;
  const methods = [...new Set(registrations.unsupported.map((item) => item.method))].join(", ");
  throw new Error(
    `Plugin only registered unsupported OpenClaw SDK APIs (${methods}); ` +
      "the evaluation hook runner supports lifecycle hooks and context engines only.",
  );
}

function applyPayloadEnv(payload) {
  if (payload.stateDir) process.env.OPENCLAW_STATE_DIR = String(payload.stateDir);
  if (payload.workspaceDir) process.env.OPENCLAW_WORKSPACE_DIR = String(payload.workspaceDir);
  if (payload.configPath) process.env.OPENCLAW_CONFIG_PATH = String(payload.configPath);
  if (payload.env && typeof payload.env === "object" && !Array.isArray(payload.env)) {
    for (const [key, value] of Object.entries(payload.env)) {
      if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(key) && value != null) {
        process.env[key] = String(value);
      }
    }
  }
}

function contextFromHookResult(result) {
  if (!result || typeof result !== "object") return "";
  const parts = [];
  const seen = new Set();
  for (const key of [
    "prependSystemContext",
    "appendSystemContext",
    "prependContext",
    "appendContext",
    "systemPromptAddition",
  ]) {
    if (typeof result[key] !== "string") continue;
    const text = result[key].trim();
    if (!text || seen.has(text)) continue;
    seen.add(text);
    parts.push(text);
  }
  return parts.join("\n\n").trim();
}

async function invokeContextEngine(payload, registrations) {
  if (!registrations.contextEngines.length) throw new Error("Plugin registered no ContextEngine");
  const selected =
    registrations.contextEngines.find((engine) => engine.id === payload.pluginId) ||
    registrations.contextEngines[0];
  const engine = await selected.factory(payload.pluginConfig || {});
  const params = {
    sessionId: payload.ctx?.sessionId || payload.sessionId || "evaluation-session",
    sessionKey: payload.ctx?.sessionKey || payload.sessionKey,
    sessionFile: payload.sessionFile || "",
    messages: payload.event?.messages || payload.messages || [],
    prePromptMessageCount: Number.isFinite(payload.prePromptMessageCount)
      ? payload.prePromptMessageCount
      : 0,
    tokenBudget: payload.tokenBudget,
    prompt: payload.event?.prompt || payload.prompt,
  };
  const calls = [];
  if (payload.action === "add") {
    if (typeof engine.bootstrap === "function") {
      calls.push({ hook_name: "bootstrap", result: await engine.bootstrap(params) });
    }
    if (typeof engine.afterTurn !== "function") throw new Error("ContextEngine has no afterTurn hook");
    const turns = Array.isArray(payload.turns) && payload.turns.length ? payload.turns : [{ messages: params.messages }];
    for (const [index, turn] of turns.entries()) {
      const result = await engine.afterTurn({
        ...params,
        messages: turn.messages || [],
        prePromptMessageCount: Number.isFinite(turn.prePromptMessageCount)
          ? turn.prePromptMessageCount
          : params.prePromptMessageCount,
      });
      calls.push({ hook_name: "afterTurn", turn_index: turn.turnIndex ?? index, result: result ?? null });
    }
    return { selected_context_engine: selected.id, calls };
  }
  if (payload.action === "recall") {
    if (typeof engine.assemble !== "function") throw new Error("ContextEngine has no assemble hook");
    const result = await engine.assemble(params);
    calls.push({ hook_name: "assemble", result: result ?? null, formatted_context: contextFromHookResult(result) });
    return { selected_context_engine: selected.id, calls };
  }
  throw new Error(`Unsupported context-engine action: ${payload.action}`);
}

async function startRegisteredServices(payload, registrations) {
  const started = [];
  const serviceRuntime = {
    stateDir: payload.stateDir || process.env.OPENCLAW_STATE_DIR,
    workspaceDir: payload.workspaceDir || process.env.OPENCLAW_WORKSPACE_DIR,
    configPath: payload.configPath || process.env.OPENCLAW_CONFIG_PATH,
  };
  for (const service of registrations.services) {
    if (service && typeof service.start === "function") {
      await service.start(serviceRuntime);
      started.push(service);
    }
  }
  return started;
}

async function stopRegisteredServices(startedServices) {
  for (const service of startedServices.toReversed()) {
    if (service && typeof service.stop === "function") {
      await service.stop();
    }
  }
}

async function invokeLifecycleHook(payload, registrations) {
  const requested = payload.hookName || (payload.action === "add" ? "agent_end" : "before_prompt_build");
  const handlers = registrations.hooks.filter((hook) => hook.name === requested);
  if (!handlers.length) throw new Error(`Plugin registered no handler for hook ${requested}`);
  const calls = [];
  for (const hook of handlers) {
    const result = await hook.handler(payload.event || {}, payload.ctx || {});
    if (payload.waitAfterHookMs) await new Promise((resolve) => setTimeout(resolve, payload.waitAfterHookMs));
    calls.push({ hook_name: hook.name, result: result ?? null, formatted_context: contextFromHookResult(result) });
  }
  return { selected_hook_name: requested, calls };
}

async function main() {
  const payloadPath = process.argv[2];
  if (!payloadPath) throw new Error("usage: node plugin_hook_runner.mjs <payload.json>");
  const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
  applyPayloadEnv(payload);
  const fetchRecords = [];
  const logs = [];
  const registrations = { hooks: [], contextEngines: [], services: [], unsupported: [] };
  const restoreFetch = patchFetch(fetchRecords);
  let output;
  try {
    const { register, entry } = await loadPlugin(payload.pluginPath);
    await register(buildApi(payload, registrations, logs));
    assertSupportedRegistrationAvailable(registrations);
    const startedServices = await startRegisteredServices(payload, registrations);
    const mode = payload.mode || (registrations.contextEngines.length ? "context_engine" : "lifecycle");
    let invocation;
    try {
      invocation =
        mode === "context_engine"
          ? await invokeContextEngine(payload, registrations)
          : await invokeLifecycleHook(payload, registrations);
    } finally {
      await stopRegisteredServices(startedServices);
    }
    output = {
      ok: true,
      plugin_id: payload.pluginId,
      plugin_path: payload.pluginPath,
      plugin_entry: entry,
      mode,
      action: payload.action,
      registrations: {
        hooks: registrations.hooks.map((hook) => hook.name),
        context_engines: registrations.contextEngines.map((engine) => engine.id),
        unsupported: registrations.unsupported,
      },
      unsupported_registrations: registrations.unsupported,
      ...invocation,
      fetch_records: fetchRecords,
      logs,
    };
  } catch (error) {
    output = {
      ok: false,
      plugin_id: payload.pluginId,
      plugin_path: payload.pluginPath,
      action: payload.action,
      error: redactString(error?.stack || error?.message || String(error)),
      unsupported_registrations: registrations.unsupported,
      fetch_records: fetchRecords,
      logs,
    };
  } finally {
    restoreFetch();
  }
  await new Promise((resolve) => process.stdout.write(`${JSON.stringify(output, null, 2)}\n`, resolve));
  process.exit(output.ok ? 0 : 1);
}

main().catch((error) => {
  process.stdout.write(`${JSON.stringify({ ok: false, error: redactString(error?.stack || String(error)) }, null, 2)}\n`);
  process.exitCode = 1;
});
