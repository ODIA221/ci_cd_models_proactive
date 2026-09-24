import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "/ui/": le build est servi par FastAPI sous /ui (src/api/main.py).
// En dev (npm run dev), les appels API sont relayés vers l'API locale.
export default defineConfig({
  base: "/ui/",
  plugins: [react()],
  server: {
    proxy: {
      "/causal": "http://localhost:8000",
      "/runs": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
