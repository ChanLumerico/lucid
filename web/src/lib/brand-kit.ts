// Build-time read of the repository brand kit's palette (``../assets``).
// Server code only — it touches the filesystem.  The logo files themselves
// reach the site through ``scripts/sync-brand.mjs``; this module serves the
// build-time renderers (the web manifest) that need the kit's colours, so no
// hex value is copied by hand.

import { readFileSync } from "node:fs";
import { join } from "node:path";

/** ``colors`` of ``assets/source/spec.json`` — the palette table of the kit's
 *  README, as ``#RRGGBB``. */
export interface BrandColors {
  violet: string;
  violet_tint: string;
  violet_shade: string;
  ink: string;
  night: string;
  lavender: string;
  text_on_dark: string;
  text_on_light: string;
}

export const brandColors: BrandColors = (
  JSON.parse(
    readFileSync(join(process.cwd(), "..", "assets", "source", "spec.json"), "utf8"),
  ) as { colors: BrandColors }
).colors;
