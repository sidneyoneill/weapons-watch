const configuredApiUrl = import.meta.env.VITE_API_URL?.trim();
const apiUrl = import.meta.env.DEV
  ? configuredApiUrl || "http://localhost:8000"
  : configuredApiUrl;

if (!apiUrl) {
  throw new Error(
    "VITE_API_URL must be set when building the production frontend.",
  );
}

const API_URL = apiUrl.replace(/\/+$/, "");

export { API_URL };
