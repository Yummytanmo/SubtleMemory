import fs from "node:fs/promises";
import path from "node:path";

const PLUGIN_ID = "openclaw-mem0-readback-facade";

function normalizeText(value) {
  return String(value ?? "").trim();
}

function resolveSessionId(ctx = {}) {
  const explicit = normalizeText(ctx.sessionId);
  if (explicit) return explicit;
  const sessionKey = normalizeText(ctx.sessionKey);
  if (!sessionKey) return "";
  const parts = sessionKey.split(":");
  return normalizeText(parts[parts.length - 1]);
}

function workspaceDir(api, ctx = {}) {
  return (
    normalizeText(ctx.workspaceDir) ||
    normalizeText(api?.runtime?.workspaceDir) ||
    normalizeText(process.env.OPENCLAW_WORKSPACE_DIR)
  );
}

async function readStagedPayload(api, ctx) {
  const root = workspaceDir(api, ctx);
  if (!root) return null;
  const sessionId = resolveSessionId(ctx);
  if (sessionId) {
    const exactPath = path.join(root, ".openclaw-readback", `${sessionId}.json`);
    try {
      return JSON.parse(await fs.readFile(exactPath, "utf8"));
    } catch {
      // Fall back to latest staged payload for hook-runner diagnostics.
    }
  }
  try {
    const stageRoot = path.join(root, ".openclaw-readback");
    const entries = await fs.readdir(stageRoot, { withFileTypes: true });
    const candidates = await Promise.all(
      entries
        .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
        .map(async (entry) => {
          const filePath = path.join(stageRoot, entry.name);
          return { filePath, mtimeMs: (await fs.stat(filePath)).mtimeMs };
        }),
    );
    candidates.sort((left, right) => right.mtimeMs - left.mtimeMs);
    if (!candidates.length) return null;
    return JSON.parse(await fs.readFile(candidates[0].filePath, "utf8"));
  } catch {
    return null;
  }
}

const CATEGORY_ORDER = [
  "identity",
  "configuration",
  "rule",
  "preference",
  "decision",
  "technical",
  "relationship",
  "project",
  "operational",
];

function metadataOf(memory) {
  return memory?.metadata && typeof memory.metadata === "object"
    ? memory.metadata
    : {};
}

function categoriesOf(memory) {
  const metadata = metadataOf(memory);
  const values = Array.isArray(metadata.categories)
    ? metadata.categories
    : Array.isArray(memory?.categories)
      ? memory.categories
      : [];
  return values.map(normalizeText).filter(Boolean);
}

function categoryRank(category) {
  if (category === "uncategorized") return CATEGORY_ORDER.length + 1;
  const normalized = category.toLowerCase();
  const index = CATEGORY_ORDER.indexOf(normalized);
  return index >= 0 ? index : CATEGORY_ORDER.length;
}

function categoryLabel(category) {
  const words = category
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((part) => part.toLowerCase());
  if (!words.length) return "Uncategorized";
  const label = words.join(" ");
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function memoryTimestamp(memory) {
  const metadata = memory?.metadata && typeof memory.metadata === "object"
    ? memory.metadata
    : {};
  return normalizeText(metadata.created_at || metadata.updated_at || memory?.created_at || memory?.updated_at);
}

function compareByTimestamp(left, right) {
  const leftTime = Date.parse(memoryTimestamp(left));
  const rightTime = Date.parse(memoryTimestamp(right));
  const leftValid = Number.isFinite(leftTime);
  const rightValid = Number.isFinite(rightTime);
  if (leftValid && rightValid && leftTime !== rightTime) {
    return leftTime - rightTime;
  }
  if (leftValid !== rightValid) {
    return leftValid ? -1 : 1;
  }
  return 0;
}

function userIdForPayload(payload) {
  const fromReadback = Array.isArray(payload?.readback?.metadata?.user_ids)
    ? payload.readback.metadata.user_ids.map(normalizeText).find(Boolean)
    : "";
  if (fromReadback) return fromReadback;
  const fromMemory = Array.isArray(payload?.memories)
    ? payload.memories
        .map((memory) => normalizeText(metadataOf(memory).user_id))
        .find(Boolean)
    : "";
  return fromMemory || normalizeText(payload?.conversationId) || "unknown";
}

function groupMemories(memories) {
  const groups = new Map();
  for (const memory of memories) {
    const categories = categoriesOf(memory);
    const category = categories[0] || "uncategorized";
    if (!groups.has(category)) groups.set(category, []);
    groups.get(category).push(memory);
  }
  return Array.from(groups.entries()).sort(([left], [right]) => {
    const leftRank = categoryRank(left);
    const rightRank = categoryRank(right);
    if (leftRank !== rightRank) return leftRank - rightRank;
    return left.localeCompare(right);
  });
}

function buildContext(payload) {
  const stagedContext =
    normalizeText(payload?.prependContext) ||
    normalizeText(payload?.formattedContext);
  if (stagedContext) return stagedContext;

  const memories = Array.isArray(payload?.memories)
    ? payload.memories.filter((memory) => normalizeText(memory?.content))
    : [];
  if (!memories.length) return "";
  const userId = userIdForPayload(payload);
  const lines = [
    "<recalled-memories>",
    `Stored memories for "${userId}" (${memories.length} total):`,
  ];
  const sessionIds = Array.isArray(payload?.sessionIds)
    ? payload.sessionIds.map(normalizeText).filter(Boolean)
    : [];
  if (sessionIds.length) {
    lines.push(`Target sessions: ${sessionIds.join(", ")}`);
  }
  lines.push("");

  for (const [category, group] of groupMemories(memories)) {
    lines.push(`${categoryLabel(category)}:`);
    for (const memory of [...group].sort(compareByTimestamp)) {
      lines.push(`- ${normalizeText(memory.content)}`);
    }
    lines.push("");
  }
  lines.push("</recalled-memories>");
  return lines.join("\n");
}

export default function register(api) {
  const handler = async (event = {}, ctx) => {
    const payload = await readStagedPayload(api, ctx);
    const prependContext = buildContext(payload);
    if (!prependContext) return undefined;
    const prompt = normalizeText(event?.prompt);
    if (prompt && prompt.includes(prependContext)) return undefined;
    return {
      questionId: payload?.questionId,
      prependContext,
    };
  };
  if (typeof api.on === "function") {
    api.on("before_prompt_build", handler);
  }
}

export const plugin = { id: PLUGIN_ID, register };
