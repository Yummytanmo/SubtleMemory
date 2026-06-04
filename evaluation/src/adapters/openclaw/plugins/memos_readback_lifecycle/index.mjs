import fs from "node:fs/promises";
import path from "node:path";

async function readLatestManifest(stageRoot) {
  try {
    const entries = await fs.readdir(stageRoot, { withFileTypes: true });
    const paths = entries
      .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
      .map((entry) => path.join(stageRoot, entry.name));
    if (paths.length === 0) return null;
    const stats = await Promise.all(
      paths.map(async (manifestPath) => ({
        manifestPath,
        mtimeMs: (await fs.stat(manifestPath)).mtimeMs,
      })),
    );
    stats.sort((left, right) => right.mtimeMs - left.mtimeMs);
    return JSON.parse(await fs.readFile(stats[0].manifestPath, "utf8"));
  } catch {
    return null;
  }
}

async function readManifest(workspaceDir, sessionId) {
  const stageRoot = path.join(workspaceDir, ".openclaw-readback");
  const explicitSessionId = String(sessionId || "").trim();
  if (explicitSessionId) {
    try {
      const raw = await fs.readFile(path.join(stageRoot, `${explicitSessionId}.json`), "utf8");
      return JSON.parse(raw);
    } catch {
      /* fall back to latest staged manifest */
    }
  }
  return await readLatestManifest(stageRoot);
}

export default function register(api) {
  const handler = async (_event, ctx = {}) => {
    const workspaceDir = String(
      ctx?.workspaceDir ||
        api.runtime?.workspaceDir ||
        process.env.OPENCLAW_WORKSPACE_DIR ||
        "",
    ).trim();
    if (!workspaceDir) return undefined;
    const manifest = await readManifest(workspaceDir, ctx?.sessionId || api.runtime?.sessionId);
    if (!manifest) return undefined;
    const appendSystemContext = String(manifest.appendSystemContext || "").trim();
    const prependContext = String(manifest.prependContext || manifest.formattedContext || "").trim();
    if (!appendSystemContext && !prependContext) return undefined;
    return { appendSystemContext, prependContext };
  };
  if (typeof api?.on === "function") {
    api.on("before_prompt_build", handler);
  } else if (typeof api?.registerHook === "function") {
    api.registerHook("before_prompt_build", handler);
  }
}
