import { loadReadbackManifest } from "./src/staged-store.js";

export default function register(api: any) {
  const handler = async (_event: any, ctx: any = {}) => {
    const workspaceDir = String(
      ctx?.workspaceDir ||
        api?.runtime?.workspaceDir ||
        process.env.OPENCLAW_WORKSPACE_DIR ||
        "",
    ).trim();
    const sessionId = String(
      ctx?.sessionId || api?.runtime?.sessionId || "",
    ).trim();
    const manifest = await loadReadbackManifest({ workspaceDir, sessionId });
    if (!manifest?.prependContext) {
      return undefined;
    }
    return {
      questionId: manifest.questionId,
      prependContext: manifest.prependContext,
    };
  };
  if (typeof api?.on === "function") {
    api.on("before_prompt_build", handler);
  } else if (typeof api?.registerHook === "function") {
    api.registerHook("before_prompt_build", handler);
  }
}

export const plugin = { id: "openclaw-session-memory-readback", register };
