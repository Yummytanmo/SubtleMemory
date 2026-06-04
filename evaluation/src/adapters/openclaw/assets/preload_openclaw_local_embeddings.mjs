import path from "node:path";
import { pathToFileURL } from "node:url";

const openclawRoot = process.env.OPENCLAW_ROOT;

if (openclawRoot) {
  const memoryCoreRuntimeUrl = pathToFileURL(
    path.join(openclawRoot, "dist", "extensions", "memory-core", "runtime-api.js"),
  ).href;
  const memoryCoreRuntime = await import(memoryCoreRuntimeUrl);
  const registryKey = Symbol.for("openclaw.memoryEmbeddingProviders");
  const globalStore = globalThis;
  const registry =
    globalStore[registryKey] instanceof Map ? globalStore[registryKey] : new Map();
  globalStore[registryKey] = registry;

  memoryCoreRuntime.registerBuiltInMemoryEmbeddingProviders({
    registerMemoryEmbeddingProvider: (adapter) => {
      registry.set(adapter.id, { adapter, ownerPluginId: "memory-core" });
    },
  });
}
