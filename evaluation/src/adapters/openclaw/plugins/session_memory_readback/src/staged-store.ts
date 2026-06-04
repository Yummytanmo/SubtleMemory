import fs from "node:fs/promises";
import path from "node:path";

export type ReadbackStagedFile = {
  sourceSessionId?: string;
  absolutePath?: string;
  relativeLabel?: string;
  content?: string;
};

export type ReadbackStagedManifest = {
  sessionId: string;
  questionId?: string;
  conversationId?: string;
  sessionIds?: string[];
  formattedContext?: string;
  prependContext?: string;
  files: ReadbackStagedFile[];
};

async function readLatestManifest(stageRoot: string): Promise<ReadbackStagedManifest | null> {
  try {
    const entries = await fs.readdir(stageRoot, { withFileTypes: true });
    const candidates = await Promise.all(
      entries
        .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
        .map(async (entry) => {
          const manifestPath = path.join(stageRoot, entry.name);
          return { manifestPath, mtimeMs: (await fs.stat(manifestPath)).mtimeMs };
        }),
    );
    candidates.sort((left, right) => right.mtimeMs - left.mtimeMs);
    if (!candidates.length) return null;
    return JSON.parse(await fs.readFile(candidates[0].manifestPath, "utf-8"));
  } catch {
    return null;
  }
}

function normalizeManifest(value: ReadbackStagedManifest | null): ReadbackStagedManifest | null {
  if (!value) return null;
  const formattedContext = String(value.prependContext || value.formattedContext || "").trim();
  if (!formattedContext) return { ...value, prependContext: "" };
  const prependContext = formattedContext.startsWith("<recalled-memories>")
    ? formattedContext
    : `<recalled-memories>\n${formattedContext}\n</recalled-memories>`;
  return {
    ...value,
    formattedContext,
    prependContext,
    files: Array.isArray(value.files) ? value.files : [],
    sessionIds: Array.isArray(value.sessionIds) ? value.sessionIds : [],
  };
}

export async function loadReadbackManifest(params: {
  workspaceDir: string;
  sessionId?: string;
}): Promise<ReadbackStagedManifest | null> {
  const workspaceDir = String(params.workspaceDir || "").trim();
  if (!workspaceDir) return null;
  const stageRoot = path.join(workspaceDir, ".openclaw-readback");
  const sessionId = String(params.sessionId || "").trim();
  if (sessionId) {
    try {
      const raw = await fs.readFile(path.join(stageRoot, `${sessionId}.json`), "utf-8");
      return normalizeManifest(JSON.parse(raw));
    } catch {
      // Fall back to latest staged payload for hook-runner diagnostics.
    }
  }
  return normalizeManifest(await readLatestManifest(stageRoot));
}
