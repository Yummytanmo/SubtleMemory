import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

const PLUGIN_ID = "evermind-ai-everos";
const DEFAULT_MEMORY_TYPES = ["episodic_memory"];
const DEFAULT_RETRIEVE_METHOD = "hybrid";

function normalizeApiBaseUrl(value) {
  const url = String(value || "").trim().replace(/\/+$/, "");
  if (!url) return "";
  const exactSuffixes = [
    "/api/v0/memories/search",
    "/api/v0/memories",
    "/api/v1/memories/search",
    "/api/v1/memories",
  ];
  for (const suffix of exactSuffixes) {
    if (url.endsWith(suffix)) {
      return `${url.slice(0, -suffix.length)}/api/v1`;
    }
  }
  if (url.endsWith("/api/v0")) return `${url.slice(0, -"/api/v0".length)}/api/v1`;
  if (url.endsWith("/api/v1")) return url;
  return `${url}/api/v1`;
}

function firstNonEmpty(...values) {
  for (const value of values) {
    if (typeof value !== "string") continue;
    const trimmed = value.trim();
    if (trimmed) return trimmed;
  }
  return "";
}

function normalizePositiveInteger(value, fallback) {
  const numeric = Number(value);
  return Number.isInteger(numeric) && numeric > 0 ? numeric : fallback;
}

function normalizeNonNegativeInteger(value, fallback) {
  const numeric = Number(value);
  return Number.isInteger(numeric) && numeric >= 0 ? numeric : fallback;
}

function normalizeMemoryTypes(value) {
  if (!Array.isArray(value)) return [...DEFAULT_MEMORY_TYPES];
  const cleaned = value.map((item) => String(item || "").trim()).filter(Boolean);
  return cleaned.length ? cleaned : [...DEFAULT_MEMORY_TYPES];
}

function toText(value) {
  if (Array.isArray(value)) {
    return value
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part === "object") {
          if (typeof part.text === "string") return part.text;
          if (typeof part.content === "string") return part.content;
        }
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  if (value && typeof value === "object") {
    if (typeof value.text === "string") return value.text;
    if (typeof value.content === "string") return value.content;
  }
  return value == null ? "" : String(value);
}

function isSessionResetPrompt(query) {
  const text = String(query || "").trim().toLowerCase();
  if (!text) return false;
  return (
    text === "reset" ||
    text === "/reset" ||
    text.includes("clear memory") ||
    text.includes("forget previous conversation") ||
    text.includes("start a new session")
  );
}

function convertMessage(message) {
  const role = String(message?.role || "").trim();
  const content = toText(message?.content).trim();
  if (!content) return { role, content: "" };
  const timestamp = message?.timestamp ?? message?.timestampMs ?? null;
  return {
    role,
    content,
    timestamp,
    sender_id: message?.sender_id,
    sender_name: message?.sender_name,
  };
}

function parseSearchResponse(raw) {
  const bucket = raw?.data ?? raw?.result ?? raw ?? {};
  const episodicSource = []
    .concat(Array.isArray(bucket?.episodic) ? bucket.episodic : [])
    .concat(Array.isArray(bucket?.episodes) ? bucket.episodes : [])
    .concat(Array.isArray(bucket?.memories) ? bucket.memories : []);
  const pendingSource = Array.isArray(bucket?.pending)
    ? bucket.pending
    : Array.isArray(bucket?.raw_messages)
      ? bucket.raw_messages
      : [];

  return {
    episodic: episodicSource
      .map((item) => {
        const text = toSingleLine(
          item?.text || item?.episode || item?.summary || item?.content || item?.memory || "",
        );
        if (!text) return null;
        return {
          text,
          timestamp: item?.timestamp ?? item?.create_time ?? item?.message_create_time ?? null,
          score: item?.score ?? null,
        };
      })
      .filter(Boolean),
    pending: pendingSource
      .map((item) => {
        const text = toSingleLine(item?.text || item?.content || item?.memory || "");
        if (!text) return null;
        return {
          text,
          timestamp: item?.timestamp ?? item?.create_time ?? item?.message_create_time ?? null,
          score: item?.score ?? null,
        };
      })
      .filter(Boolean),
  };
}

function buildMemoryPrompt(parsed, { wrapInCodeBlock = false } = {}) {
  const lines = [];
  let index = 1;
  for (const item of parsed?.episodic || []) {
    const text = toSingleLine(item?.text || "");
    if (!text) continue;
    const timestamp = toSingleLine(item?.timestamp || "");
    lines.push(`${index}. ${timestamp ? `${timestamp}: ${text}` : text}`);
    index += 1;
  }
  for (const item of parsed?.pending || []) {
    const text = toSingleLine(item?.text || "");
    if (!text) continue;
    const timestamp = toSingleLine(item?.timestamp || "");
    lines.push(`${index}. ${timestamp ? `${timestamp}: ${text}` : text}`);
    index += 1;
  }
  if (!lines.length) return "";
  const body = `## Relevant Long-Term Memory\n${lines.join("\n")}`;
  return wrapRecalledMemories(body);
}

function wrapRecalledMemories(context) {
  const text = String(context || "").trim();
  if (!text) return "";
  if (/^<recalled-memories\b/i.test(text)) return text;
  return `<recalled-memories>\n${text}\n</recalled-memories>`;
}

function extractBenchmarkQuestion(query) {
  const text = String(query || "").trim();
  if (!text) return "";
  const match = text.match(/(?:^|\n)\s*Question:\s*([\s\S]*?)(?:\n\s*Answer:\s*$|\n\s*Answer:\s*|\s*$)/i);
  if (!match) return text;
  return String(match[1] || "").trim() || text;
}

function buildRecallQueries(query) {
  const candidates = [];
  const seen = new Set();

  function add(value) {
    const text = String(value || "").trim();
    if (text.length < 3 || seen.has(text)) return;
    seen.add(text);
    candidates.push(text);
  }

  const text = extractBenchmarkQuestion(query);
  add(text);
  const beforeBrief = text.split(/\n\s*\nAvailable resource brief:\s*/i)[0]?.trim();
  add(beforeBrief);
  const firstSentence = text.split(/[.!?]\s+/)[0]?.trim();
  add(firstSentence);
  if (beforeBrief && beforeBrief.length > 160) {
    const shortened = beforeBrief.slice(0, 160).trim();
    const lastSpace = shortened.lastIndexOf(" ");
    add(lastSpace > 0 ? shortened.slice(0, lastSpace).trim() : shortened);
  }
  return candidates.slice(0, 4);
}

function toSingleLine(value) {
  return value == null ? "" : String(value).replace(/[\r\n]+/g, " ").trim();
}

function normalizeTimestampMs(value) {
  if (value == null || value === "") return null;
  if (typeof value === "number" && Number.isFinite(value)) return Math.trunc(value);
  const text = String(value).trim();
  if (!text) return null;
  const numeric = Number(text);
  if (Number.isFinite(numeric)) return Math.trunc(numeric);
  const parsed = Date.parse(text);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseSearchResponseLenient(raw) {
  const parsed = parseSearchResponse(raw);
  const parsedCount = (parsed?.episodic?.length || 0) + (parsed?.pending?.length || 0);
  if (parsedCount > 0) {
    return parsed;
  }

  const bucket = raw?.data ?? raw?.result ?? raw;
  const memories = Array.isArray(bucket?.memories) ? bucket.memories : [];
  const episodes = Array.isArray(bucket?.episodes) ? bucket.episodes : [];
  const rawMessages = Array.isArray(bucket?.raw_messages) ? bucket.raw_messages : [];
  if (!memories.length && !episodes.length && !rawMessages.length) {
    return parsed;
  }

  return {
    episodic: [...memories, ...episodes]
      .map((episode) => {
        const body = toSingleLine(
          episode?.episode || episode?.summary || episode?.content || episode?.text || "",
        );
        const subject = toSingleLine(episode?.subject || "");
        if (!body && !subject) return null;
        return {
          text: subject ? `${subject}: ${body}` : body,
          timestamp: episode?.timestamp ?? episode?.create_time ?? episode?.message_create_time ?? null,
        };
      })
      .filter(Boolean),
    pending: rawMessages
      .map((message) => {
        const body = toSingleLine(message?.content || "");
        const who = toSingleLine(
          message?.sender_name || message?.sender || message?.sender_id || message?.user_id || "",
        );
        if (!body) return null;
        return {
          text: who ? `${who}: ${body}` : body,
          timestamp: message?.message_create_time ?? message?.created_at ?? message?.timestamp ?? message?.create_time ?? null,
        };
      })
      .filter(Boolean),
  };
}

function parseHttpStatus(error) {
  const message = String(error?.message || error || "");
  const match = message.match(/\bHTTP\s+(\d{3})\b/);
  return match ? Number(match[1]) : 0;
}

function shouldRetryRequest(error) {
  const status = parseHttpStatus(error);
  if (status === 429) return true;
  if (status >= 500 && status < 600) return true;
  return status === 0;
}

function buildHeaders(cfg) {
  const headers = { "Content-Type": "application/json" };
  if (cfg.apiKey) headers.Authorization = `Bearer ${cfg.apiKey}`;
  return headers;
}

function normalizeConfig(config = {}) {
  return {
    serverUrl: normalizeApiBaseUrl(firstNonEmpty(config.baseUrl, process.env.EVERMEMOS_API_URL, "http://localhost:1995")),
    apiKey: firstNonEmpty(config.apiKey, process.env.EVERMEMOS_API_KEY),
    userId: firstNonEmpty(config.userId, "everos-user"),
    topK: normalizePositiveInteger(config.topK, 5),
    memoryTypes: normalizeMemoryTypes(config.memoryTypes),
    retrieveMethod: firstNonEmpty(config.retrieveMethod, DEFAULT_RETRIEVE_METHOD) || DEFAULT_RETRIEVE_METHOD,
    captureStrategy: firstNonEmpty(config.captureStrategy, "incremental_turns"),
    asyncMode: config.asyncMode !== false,
    requestIntervalMs: normalizeNonNegativeInteger(config.requestIntervalMs, 200),
    maxRetries: normalizePositiveInteger(config.maxRetries, 5),
    retryBaseMs: normalizeNonNegativeInteger(config.retryBaseMs, 1000),
    retryMaxMs: normalizeNonNegativeInteger(config.retryMaxMs, 8000),
    personalBatchMaxMessages: normalizePositiveInteger(config.personalBatchMaxMessages, 5),
    personalBatchMaxChars: normalizePositiveInteger(config.personalBatchMaxChars, 5500),
    saveEnabled: config.saveEnabled !== false,
    readbackEnabled: config.readbackEnabled === true,
  };
}

function collectLastUserTurn(messages) {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i]?.role === "user") return messages.slice(i);
  }
  return [];
}

function buildSessionState() {
  return { turnCount: 0, savedUpTo: 0, lastActiveTime: Date.now() };
}

function normalizeSessionId(sessionKey) {
  const text = String(sessionKey || "").trim();
  if (!text) return "";
  return text.includes(":") ? text.split(":").filter(Boolean).at(-1) || text : text;
}

async function readStagedReadbackManifest(workspaceDir, sessionKey, sessionId) {
  const root = String(workspaceDir || "").trim();
  const answerSessionId = normalizeSessionId(sessionId || sessionKey);
  if (!root || !answerSessionId) return null;
  const manifestPath = path.join(root, ".openclaw-readback", `${answerSessionId}.json`);
  try {
    const raw = await fs.readFile(manifestPath, "utf8");
    const parsed = JSON.parse(raw);
    return { manifestPath, manifest: parsed };
  } catch {
    return null;
  }
}

function buildReadbackContextFromManifest(manifest) {
  const formatted = String(manifest?.formattedContext || "").trim();
  if (formatted) return wrapRecalledMemories(formatted);
  const lines = [];
  for (const file of manifest?.files || []) {
    const text = toSingleLine(file?.content || "");
    if (!text) continue;
    lines.push(`${lines.length + 1}. ${text}`);
  }
  for (const item of manifest?.memories || []) {
    const text = toSingleLine(item?.content || "");
    if (!text) continue;
    lines.push(`${lines.length + 1}. ${text}`);
  }
  return lines.length ? wrapRecalledMemories(`## Relevant Long-Term Memory\n${lines.join("\n")}`) : "";
}

function readbackMemoryCount(manifest) {
  const fileCount = Array.isArray(manifest?.files)
    ? manifest.files.filter((file) => file?.content || file?.absolutePath || file?.relativeLabel).length
    : 0;
  const memoryCount = Array.isArray(manifest?.memories)
    ? manifest.memories.filter((item) => item?.content).length
    : 0;
  return Math.max(fileCount, memoryCount);
}

function toProviderPersonalMessage(message, userId) {
  const providerMessage = { ...message };
  const role = String(providerMessage.role || "").trim();
  if (providerMessage.timestamp == null && providerMessage.timestampMs != null) {
    providerMessage.timestamp = providerMessage.timestampMs;
  }
  const timestampMs = normalizeTimestampMs(providerMessage.timestamp);
  if (timestampMs == null) {
    delete providerMessage.timestamp;
  } else {
    providerMessage.timestamp = timestampMs;
  }
  delete providerMessage.timestampMs;
  if (role === "user") {
    providerMessage.sender_id = userId;
    if (!providerMessage.sender_name) providerMessage.sender_name = userId;
  } else if (role === "assistant" && providerMessage.sender_id === userId) {
    delete providerMessage.sender_id;
  }
  return providerMessage;
}

function stableMessageId(sessionKey, index, message) {
  if (message.message_id) return String(message.message_id);
  const role = String(message.role || "user");
  const content = String(message.content || "");
  const timestamp = String(message.timestamp ?? message.timestampMs ?? "");
  const hash = createHash("sha256")
    .update(`${sessionKey || "evaluation-session"}:${index}:${role}:${timestamp}:${content}`)
    .digest("hex")
    .slice(0, 24);
  return `em_${hash}`;
}

async function savePersonalSessionMemories(cfg, { sessionKey, messages, log }) {
  const sessionId = normalizeSessionId(sessionKey);
  const providerMessages = messages.map((message, index) => {
    const providerMessage = toProviderPersonalMessage(message, cfg.userId);
    providerMessage.message_id = stableMessageId(sessionKey, index, providerMessage);
    return providerMessage;
  });
  const batches = [];
  let currentBatch = [];
  let currentChars = 0;
  for (const providerMessage of providerMessages) {
    const messageChars = JSON.stringify(providerMessage).length;
    const wouldExceedCount = currentBatch.length >= cfg.personalBatchMaxMessages;
    const wouldExceedChars =
      currentBatch.length > 0 && currentChars + messageChars > cfg.personalBatchMaxChars;
    if (wouldExceedCount || wouldExceedChars) {
      batches.push(currentBatch);
      currentBatch = [];
      currentChars = 0;
    }
    currentBatch.push(providerMessage);
    currentChars += messageChars;
  }
  if (currentBatch.length) {
    batches.push(currentBatch);
  }
  for (const [index, batch] of batches.entries()) {
    const payload = {
      user_id: cfg.userId,
      messages: batch,
      ...(sessionId ? { session_id: sessionId } : {}),
      async_mode: cfg.asyncMode,
    };
    await requestWithRetry(cfg, "POST", "/memories", payload, log);
    if (cfg.requestIntervalMs > 0 && index < batches.length - 1) {
      await sleep(cfg.requestIntervalMs);
    }
  }
  log.info?.(`[${PLUGIN_ID}] POST /api/v1/memories personal messages=${providerMessages.length} batches=${batches.length}`);
  return { savedCount: providerMessages.length, transport: "POST /api/v1/memories personal chunked-batch" };
}

async function requestWithRetry(cfg, method, path, payload, log) {
  let lastError = null;
  for (let attempt = 0; attempt < cfg.maxRetries; attempt += 1) {
    try {
      return await requestOnce(cfg, method, path, payload);
    } catch (error) {
      lastError = error;
      if (!shouldRetryRequest(error) || attempt >= cfg.maxRetries - 1) {
        throw error;
      }
      const retryBackoffMs = cfg.retryBaseMs * 2 ** attempt;
      const cappedBackoffMs = cfg.retryMaxMs > 0
        ? Math.min(cfg.retryMaxMs, retryBackoffMs)
        : retryBackoffMs;
      const delay = Math.max(cfg.requestIntervalMs, cappedBackoffMs);
      log.warn?.(
        `[${PLUGIN_ID}] ${method} ${path} failed (${error?.message || String(error)}); retrying in ${delay}ms`,
      );
      if (delay > 0) await sleep(delay);
    }
  }
  throw lastError ?? new Error(`${method} ${path} failed`);
}

async function requestOnce(cfg, method, path, payload) {
  const response = await fetch(`${cfg.serverUrl}${path}`, {
    method,
    headers: buildHeaders(cfg),
    body: payload == null ? undefined : JSON.stringify(payload),
  });
  const text = await response.text().catch(() => "");
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}${text ? ` - ${text.slice(0, 200)}` : ""}`);
  }
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

async function searchMemories(cfg, params, log) {
  const payload = {
    query: params.query,
    top_k: params.top_k,
    memory_types: params.memory_types,
    method: params.retrieve_method,
    retrieve_method: params.retrieve_method,
    filters: {},
  };
  if (params.user_id) payload.filters.user_id = params.user_id;
  log.info?.(
    `[${PLUGIN_ID}] POST /api/v1/memories/search top_k=${payload.top_k}`,
  );
  return requestWithRetry(cfg, "POST", "/memories/search", payload, log);
}

async function saveMemories(cfg, { userId, messages, idSeed }, log) {
  const payload = {
    user_id: userId,
    messages,
    async_mode: cfg.asyncMode,
    id_seed: idSeed,
  };
  await requestWithRetry(cfg, "POST", "/memories", payload, log);
  return {
    savedCount: messages.length,
    transport: "POST /api/v1/memories",
  };
}

function createEngine(config, logger, runtime = {}) {
  const cfg = normalizeConfig(config);
  const log = logger || console;
  const stateBySession = new Map();
  const workspaceDir = runtime.workspaceDir || process.env.OPENCLAW_WORKSPACE_DIR || "";

  function stateFor(sessionKey) {
    const key = sessionKey || "evaluation-session";
    if (!stateBySession.has(key)) stateBySession.set(key, buildSessionState());
    return stateBySession.get(key);
  }

  async function assembleContext(query, messages, turnCount) {
    const topK = cfg.topK;
    for (const candidate of buildRecallQueries(query)) {
      const params = {
        query: candidate,
        user_id: cfg.userId,
        memory_types: cfg.memoryTypes,
        retrieve_method: cfg.retrieveMethod,
        top_k: topK,
      };
      const result = await searchMemories(cfg, params, log);
      const parsed = parseSearchResponseLenient(result) || { episodic: [], pending: [] };
      const memoryCount = (parsed.episodic?.length || 0) + (parsed.pending?.length || 0);
      if (!memoryCount) continue;
      return {
        context: buildMemoryPrompt(parsed, { wrapInCodeBlock: true }),
        memoryCount,
      };
    }
    return { context: "", memoryCount: 0 };
  }

  function contextResult(context, extra = {}) {
    return {
      messages: extra.messages || [],
      estimatedTokens: Math.floor(String(context || "").length / 4),
      prependContext: context,
      systemPromptAddition: context,
      ...extra,
    };
  }

  async function assembleForPrompt({ sessionKey, sessionId, messages = [], prompt }) {
    const state = stateFor(sessionKey);
    state.lastActiveTime = Date.now();
    if (cfg.readbackEnabled) {
      const staged = await readStagedReadbackManifest(
        workspaceDir,
        sessionKey,
        normalizeSessionId(sessionId || sessionKey),
      );
      if (!staged) {
        return { messages, estimatedTokens: 0, readback: true, readbackMissing: true };
      }
      const context = buildReadbackContextFromManifest(staged.manifest);
      if (!context) {
        return { messages, estimatedTokens: 0, readback: true, readbackEmpty: true };
      }
      return contextResult(context, {
        messages,
        readback: true,
        readbackMemoryCount: readbackMemoryCount(staged.manifest),
        readbackManifestPath: staged.manifestPath,
      });
    }
    const latestUser = [...messages].reverse().find((message) => message.role === "user");
    const query = toText(prompt) || toText(latestUser?.content);
    if (!query || query.length < 3 || isSessionResetPrompt(query)) {
      return { messages, estimatedTokens: 0 };
    }

    try {
      const { context, memoryCount } = await assembleContext(query, messages, state.turnCount);
      if (!memoryCount) return { messages, estimatedTokens: 0 };
      return contextResult(context, { messages });
    } catch (error) {
      log.warn?.(`[${PLUGIN_ID}] assemble: failed: ${error?.message || String(error)}`);
      return { messages, estimatedTokens: 0 };
    }
  }

  return {
    info: {
      id: PLUGIN_ID,
      name: "SubtleMemory OpenClaw Plugin Evaluation Wrapper",
      version: "1.0.0",
      ownsCompaction: false,
    },

    async ingest() {
      return { ingested: false };
    },

    async afterTurn({ sessionKey, messages = [], prePromptMessageCount }) {
      if (!cfg.saveEnabled) return { ok: true, savedCount: 0, skipped: "save_disabled" };
      const state = stateFor(sessionKey);
      state.turnCount += 1;
      state.lastActiveTime = Date.now();
      if (state.savedUpTo > messages.length) state.savedUpTo = 0;

      const sliceStart =
        cfg.captureStrategy === "full_session"
          ? 0
          : prePromptMessageCount !== undefined
          ? Math.max(prePromptMessageCount, state.savedUpTo)
          : state.savedUpTo || 0;
      const newMessages =
        cfg.captureStrategy === "full_session"
          ? messages
          : sliceStart > 0
            ? messages.slice(sliceStart)
            : collectLastUserTurn(messages);
      const converted = newMessages
        .filter((message) => message.role !== "toolResult" && message.role !== "tool")
        .map(convertMessage)
        .filter((message) => message.content);
      if (!converted.length) return { ok: true, savedCount: 0, skipped: "no_semantic_messages" };

      try {
        const result =
          cfg.captureStrategy === "full_session"
            ? await savePersonalSessionMemories(cfg, { sessionKey, messages: converted, log })
            : await saveMemories(
                cfg,
                {
                  userId: cfg.userId,
                  messages: converted,
                  idSeed: `${sessionKey || "evaluation-session"}:${state.turnCount}`,
                },
                log,
              );
        state.savedUpTo = messages.length;
        return { ok: true, savedCount: converted.length, transport: result?.transport };
      } catch (error) {
        log.warn?.(`[${PLUGIN_ID}] afterTurn: save failed: ${error?.message || String(error)}`);
        return { ok: false, error: error?.message || String(error) };
      }
    },

    async assemble({ sessionKey, messages = [], prompt }) {
      return assembleForPrompt({ sessionKey, messages, prompt });
    },

    async compact() {
      return {
        ok: true,
        compacted: false,
        reason: "provider_managed_context_engine",
        result: {
          tokensBefore: 0,
        },
      };
    },

    async dispose({ sessionKey } = {}) {
      if (sessionKey) stateBySession.delete(sessionKey);
      else stateBySession.clear();
      return { ok: true };
    },
  };
}

export default function register(api) {
  const pluginConfig = api.pluginConfig || {};
  const log = api.logger || console;
  log.info?.(`[${PLUGIN_ID}] registering evaluation wrapper`);
  const makeEngine = () => createEngine(pluginConfig, log, api.runtime || {});
  api.registerContextEngine(PLUGIN_ID, makeEngine);
  api.on?.("before_prompt_build", async (event = {}, ctx = {}) => {
    const engine = makeEngine();
    const result = await engine.assemble({
      sessionKey: ctx.sessionKey,
      sessionId: ctx.sessionId,
      messages: event.messages || [],
      prompt: event.prompt,
    });
    const context = String(result?.prependContext || result?.systemPromptAddition || "").trim();
    if (!context) return result;
    return {
      prependContext: context,
      systemPromptAddition: context,
      ...(result?.readback ? { readback: result.readback } : {}),
      ...(result?.readbackMemoryCount != null ? { readbackMemoryCount: result.readbackMemoryCount } : {}),
      ...(result?.readbackManifestPath ? { readbackManifestPath: result.readbackManifestPath } : {}),
    };
  });
}
