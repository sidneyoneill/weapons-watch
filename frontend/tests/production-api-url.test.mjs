import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const apiUrl = "https://weapons-watch-api.example.test";
const projectRoot = fileURLToPath(new URL("..", import.meta.url));

function readJavaScriptFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);

    if (entry.isDirectory()) return readJavaScriptFiles(path);
    if (!entry.name.endsWith(".js")) return [];

    return readFileSync(path, "utf8");
  });
}

test("the production build uses VITE_API_URL instead of localhost", () => {
  execFileSync("npm", ["run", "build"], {
    cwd: projectRoot,
    env: { ...process.env, VITE_API_URL: apiUrl },
    stdio: "pipe",
  });

  const bundle = readJavaScriptFiles(
    fileURLToPath(new URL("../dist", import.meta.url)),
  ).join("\n");

  assert.match(bundle, new RegExp(apiUrl.replaceAll(".", "\\.")));
  assert.doesNotMatch(bundle, /https?:\/\/(?:localhost|127\.0\.0\.1):8000/);
});

test("the production build rejects unsafe API URLs", () => {
  const cases = [
    ["", /VITE_API_URL (?:must be set|is required)/],
    ["http://weapons-watch-api.example.test", /must use HTTPS/],
    ["https://localhost:8000", /contains a localhost API URL/],
  ];

  for (const [candidate, expectedError] of cases) {
    const result = spawnSync("npm", ["run", "build"], {
      cwd: projectRoot,
      env: { ...process.env, VITE_API_URL: candidate },
      encoding: "utf8",
    });
    const output = `${result.stdout}\n${result.stderr}`;

    assert.notEqual(result.status, 0, `build unexpectedly accepted ${candidate}`);
    assert.match(output, expectedError);
  }
});
