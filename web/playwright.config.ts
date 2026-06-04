import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright smoke tests for the docs site.
 *
 * Runtime coverage the static contract audit (scripts/audit-docs.mjs) can't
 * see: client hydration, console / page errors, and interactive behavior
 * (the scroll-spy ToC).  Runs against `pnpm dev:fast` — `SKIP_API_BUILD=1`
 * so it needs neither Griffe nor a built Lucid, and dev mode leaves basePath
 * empty (it's prod-only), so URLs are plain.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: "http://localhost:3000",
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "pnpm dev:fast",
    url: "http://localhost:3000",
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});
