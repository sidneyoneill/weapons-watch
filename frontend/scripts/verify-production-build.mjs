import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadEnv } from "vite";

const projectRoot = fileURLToPath(new URL("..", import.meta.url));
const environment = {
  ...loadEnv("production", projectRoot, "VITE_"),
  ...process.env,
};
const apiUrl = environment.VITE_API_URL?.trim().replace(/\/+$/, "");

if (!apiUrl) {
  throw new Error("VITE_API_URL is required for a production build.");
}

const parsedApiUrl = new URL(apiUrl);
if (parsedApiUrl.protocol !== "https:") {
  throw new Error("VITE_API_URL must use HTTPS in production.");
}

function readJavaScriptFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);

    if (entry.isDirectory()) return readJavaScriptFiles(path);
    if (!entry.name.endsWith(".js")) return [];

    return readFileSync(path, "utf8");
  });
}

const bundle = readJavaScriptFiles(join(projectRoot, "dist")).join("\n");

if (!bundle.includes(apiUrl)) {
  throw new Error("The production bundle does not contain VITE_API_URL.");
}

if (/https?:\/\/(?:localhost|127\.0\.0\.1):8000/.test(bundle)) {
  throw new Error("The production bundle contains a localhost API URL.");
}

console.log(`Verified production API URL: ${apiUrl}`);
