import { test, expect, type Page } from "@playwright/test";

/**
 * Runtime smoke: load representative pages and assert they hydrate without
 * uncaught exceptions or console errors, then exercise the scroll-spy ToC.
 * These are failure modes the static-HTML contract audit cannot observe.
 */

// Console noise that isn't a real defect (dev-only fast-refresh / resource
// 404s for assets not served by `next dev`).  Keep this list tight.
const BENIGN = [
  /Download the React DevTools/i,
  /\[Fast Refresh\]/i,
  /favicon/i,
];

function attachErrorCollectors(page: Page): { errors: string[] } {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (msg) => {
    if (msg.type() !== "error") return;
    const text = msg.text();
    if (BENIGN.some((re) => re.test(text))) return;
    errors.push(`console.error: ${text}`);
  });
  return { errors };
}

const PAGES = [
  { path: "/", name: "home" },
  { path: "/api", name: "api landing" },
  { path: "/api/lucid.nn", name: "module overview" },
  { path: "/api/lucid.models.vision.resnet/resnet_50", name: "model member" },
];

for (const { path, name } of PAGES) {
  test(`${name} hydrates without errors`, async ({ page }) => {
    const { errors } = attachErrorCollectors(page);
    await page.goto(path, { waitUntil: "networkidle" });
    await expect(page.locator("#main-content, main").first()).toBeVisible();
    expect(errors, errors.join("\n")).toEqual([]);
  });
}

test("model member page renders the Model Size card", async ({ page }) => {
  await page.goto("/api/lucid.models.vision.resnet/resnet_50", {
    waitUntil: "networkidle",
  });
  await expect(page.getByText("Model Size", { exact: false })).toBeVisible();
});

test("ToC entry click scrolls + updates the hash", async ({ page }) => {
  await page.goto("/api/lucid.nn", { waitUntil: "networkidle" });
  // The "On this page" ToC is an <aside> of in-page anchor links (xl only).
  await page.setViewportSize({ width: 1440, height: 900 });
  const tocLinks = page.locator('aside a[href*="#"]');
  const count = await tocLinks.count();
  test.skip(count === 0, "no ToC on this page");
  await tocLinks.first().click();
  await expect(page).toHaveURL(/#.+/);
});
