import fs from "node:fs/promises";
import path from "node:path";

function requireString(payload, key) {
  const value = payload[key];
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`missing required string field: ${key}`);
  }
  return value;
}

async function snapshotDirectory(dirPath) {
  try {
    return new Set(await fs.readdir(dirPath));
  } catch {
    return new Set();
  }
}

function diffNewFiles(before, after) {
  return [...after].filter((entry) => !before.has(entry)).sort();
}

async function main() {
  const payloadPath = process.argv[2];
  if (!payloadPath) {
    throw new Error("usage: node session_memory_bridge.mjs <payload.json>");
  }

  const payload = JSON.parse(await fs.readFile(payloadPath, "utf-8"));
  const handlerPath = requireString(payload, "handlerPath");
  const configPath = requireString(payload, "configPath");
  const finalMemoryDir = requireString(payload, "memoryDir");
  const handlerWorkspaceDir = requireString(payload, "handlerWorkspaceDir");
  const sessionKey = requireString(payload, "sessionKey");
  const sessionId = requireString(payload, "sessionId");
  const sessionFile = requireString(payload, "sessionFile");
  const timestamp = requireString(payload, "timestamp");
  const targetFileName = requireString(payload, "targetFileName");
  const resultPath = requireString(payload, "resultPath");
  const commandSource =
    typeof payload.commandSource === "string" && payload.commandSource.trim()
      ? payload.commandSource.trim()
      : "evaluation-import";

  const cfg = JSON.parse(await fs.readFile(configPath, "utf-8"));
  const handlerMemoryDir = path.join(handlerWorkspaceDir, "memory");

  await fs.rm(handlerWorkspaceDir, { recursive: true, force: true });
  await fs.mkdir(handlerMemoryDir, { recursive: true });
  await fs.mkdir(finalMemoryDir, { recursive: true });
  await fs.mkdir(path.dirname(resultPath), { recursive: true });

  const beforeFiles = await snapshotDirectory(handlerMemoryDir);
  const { default: saveSessionToMemory } = await import(handlerPath);
  await saveSessionToMemory({
    type: "command",
    action: "new",
    sessionKey,
    context: {
      cfg,
      workspaceDir: handlerWorkspaceDir,
      commandSource,
      previousSessionEntry: {
        sessionId,
        sessionFile,
      },
    },
    timestamp: new Date(timestamp),
    messages: [],
  });

  const afterFiles = await snapshotDirectory(handlerMemoryDir);
  const newFiles = diffNewFiles(beforeFiles, afterFiles);
  if (newFiles.length !== 1) {
    throw new Error(
      `expected exactly 1 handler memory file, found ${newFiles.length}${
        newFiles.length > 0 ? ` (${newFiles.join(", ")})` : ""
      }`,
    );
  }

  const generatedFileName = newFiles[0];
  const generatedPath = path.join(handlerMemoryDir, generatedFileName);
  const targetPath = path.join(finalMemoryDir, targetFileName);
  await fs.rm(targetPath, { force: true });
  await fs.rename(generatedPath, targetPath);
  const content = await fs.readFile(targetPath, "utf-8");
  await fs.rm(handlerWorkspaceDir, { recursive: true, force: true });

  await fs.writeFile(
    resultPath,
    `${JSON.stringify(
      {
        generatedFileName,
        memoryPath: targetPath,
        header: content.split(/\r?\n/, 1)[0] ?? "",
        hasConversationSummary: content.includes("## Conversation Summary"),
      },
      null,
      2,
    )}\n`,
    "utf-8",
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack || error.message : String(error));
  process.exit(1);
});
