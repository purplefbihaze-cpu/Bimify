const fs = require("fs");
const path = require("path");

const PROJECT_ROOT = path.join(__dirname, "..");
const DEST_DIR = path.join(PROJECT_ROOT, "public", "wasm", "web-ifc");
const CANDIDATE_DIRS = [
  path.join(PROJECT_ROOT, "node_modules", "web-ifc-three", "node_modules", "web-ifc"),
  path.join(PROJECT_ROOT, "node_modules", "web-ifc"),
];
const FILES = ["web-ifc.wasm", "web-ifc-mt.wasm"];

function findSourceDir() {
  for (const candidate of CANDIDATE_DIRS) {
    if (FILES.every((file) => fs.existsSync(path.join(candidate, file)))) {
      return candidate;
    }
  }
  return null;
}

function copyWasmFiles() {
  const sourceDir = findSourceDir();
  if (!sourceDir) {
    throw new Error("Could not locate web-ifc wasm files in node_modules.");
  }

  fs.mkdirSync(DEST_DIR, { recursive: true });

  for (const file of FILES) {
    const src = path.join(sourceDir, file);
    const dest = path.join(DEST_DIR, file);
    fs.copyFileSync(src, dest);
  }

  console.log(`Copied web-ifc wasm files from ${sourceDir} to ${DEST_DIR}`);
}

try {
  copyWasmFiles();
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
}









