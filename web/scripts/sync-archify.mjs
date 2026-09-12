/**
 * Copy the archify diagram viewers into the static site.
 *
 *   node scripts/sync-archify.mjs      (run by predev / prebuild)
 *
 * Source of truth is content/architecture/<name>.html — the viewers archify
 * delivered beside their JSON specs.  Each copy lands in public/archify/
 * (gitignored) with the host bridge linked in before </head>: host.css and
 * host.js live beside the copies and are tracked, because they are source.
 * Copies whose source viewer is gone are removed, so a deleted diagram can't
 * linger on the site.
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const SRC = join(process.cwd(), "content", "architecture");
const OUT = join(process.cwd(), "public", "archify");
const BRIDGE = '<link rel="stylesheet" href="host.css">\n<script src="host.js"></script>\n';

if (!existsSync(SRC)) {
  console.warn(`[sync-archify] ${SRC} not found — no viewers copied`);
  process.exit(0);
}
mkdirSync(OUT, { recursive: true });

// Visual-check sidecars (contact sheets) are .html too, but not viewers.
const viewers = readdirSync(SRC).filter((f) => f.endsWith(".html") && !f.includes(".visual-check."));

for (const name of viewers) {
  const html = readFileSync(join(SRC, name), "utf8");
  // The viewer's inline scripts carry markup strings; anchor on the real
  // head/body seam rather than the first "</head>" in the file.
  const seam = html.search(/<\/head>\s*<body[\s>]/);
  if (seam < 0) throw new Error(`[sync-archify] ${name}: no </head><body> seam`);
  writeFileSync(join(OUT, name), html.slice(0, seam) + BRIDGE + html.slice(seam));
}

for (const name of readdirSync(OUT)) {
  if (name.endsWith(".html") && !viewers.includes(name)) rmSync(join(OUT, name));
}

console.log(`[sync-archify] ${viewers.length} viewer(s) → public/archify/`);
