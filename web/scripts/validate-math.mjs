#!/usr/bin/env node
// validate-math.mjs — strict KaTeX validation across all api-data JSON.
//
// Walks every string field in every api-data file, extracts $...$ and $$...$$
// expressions, and runs them through KaTeX in strict mode.  Exits non-zero
// when any expression fails to parse, listing the offender's source location
// (file + member path) so the writer can fix the docstring at the source.
//
// Run via `pnpm validate:math` (added to package.json) or directly.

import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { dirname } from "node:path";
import katex from "katex";

const __dirname = dirname(fileURLToPath(import.meta.url));
const API_DATA_DIR = join(__dirname, "..", "public", "api-data");

// Strict-mode KaTeX render — throws on anything KaTeX considers invalid
// (unknown macros, unbalanced braces, etc.).  We don't keep the output;
// we only care whether it parses.
function validateExpr(expr, display) {
  try {
    katex.renderToString(expr, {
      strict: "error",
      throwOnError: true,
      displayMode: display,
      // KaTeX trust setting: we don't render to a browser here, so the
      // default (don't run \href etc.) is fine.
    });
    return null;
  } catch (err) {
    return err.message ?? String(err);
  }
}

// Markdown math extraction.  We match the same syntax the MathText
// component (remark-math + rehype-katex) accepts:
//   - $$ ... $$   (display, can span lines)
//   - $ ... $     (inline, single line)
// Order matters: try $$ first so the inline pattern doesn't eat one half
// of a display block.
const DISPLAY_RE = /\$\$([\s\S]+?)\$\$/g;
const INLINE_RE = /(?<!\$)\$([^\n$]+?)\$(?!\$)/g;

function* extractMath(text) {
  if (typeof text !== "string" || !text.includes("$")) return;
  // Display first, with placeholder for inline pass to skip the same span.
  const displayRanges = [];
  for (const m of text.matchAll(DISPLAY_RE)) {
    displayRanges.push([m.index, m.index + m[0].length]);
    yield { expr: m[1].trim(), display: true, start: m.index };
  }
  // Inline math — skip anything covered by a display range.
  const inDisplay = (idx) =>
    displayRanges.some(([lo, hi]) => idx >= lo && idx < hi);
  for (const m of text.matchAll(INLINE_RE)) {
    if (inDisplay(m.index)) continue;
    yield { expr: m[1].trim(), display: false, start: m.index };
  }
}

// Recursive walk.  Yields {path, value} for every string descendant so the
// error message can name the source field.  ``path`` is a dotted breadcrumb
// like "members[3].extended".
function* walkStrings(value, path = "") {
  if (typeof value === "string") {
    yield { path, value };
  } else if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) {
      yield* walkStrings(value[i], `${path}[${i}]`);
    }
  } else if (value && typeof value === "object") {
    for (const [k, v] of Object.entries(value)) {
      yield* walkStrings(v, path ? `${path}.${k}` : k);
    }
  }
}

function validateFile(file) {
  const full = join(API_DATA_DIR, file);
  let data;
  try {
    data = JSON.parse(readFileSync(full, "utf-8"));
  } catch (err) {
    return [{ file, path: "<parse>", expr: "", err: `JSON parse failed: ${err.message}` }];
  }
  const errors = [];
  for (const { path, value } of walkStrings(data)) {
    for (const { expr, display } of extractMath(value)) {
      const err = validateExpr(expr, display);
      if (err) errors.push({ file, path, expr, display, err });
    }
  }
  return errors;
}

function main() {
  const files = readdirSync(API_DATA_DIR)
    .filter((f) => f.endsWith(".json") && !f.startsWith("_"));
  let totalErrors = 0;
  let totalExprs = 0;
  let totalFiles = 0;

  for (const file of files) {
    const errors = validateFile(file);
    // Cheap counter: re-walk to count expressions (so we report coverage).
    const data = JSON.parse(readFileSync(join(API_DATA_DIR, file), "utf-8"));
    let exprCount = 0;
    for (const { value } of walkStrings(data)) {
      for (const _ of extractMath(value)) exprCount++;
    }
    totalExprs += exprCount;
    totalFiles++;

    if (errors.length === 0) continue;
    totalErrors += errors.length;
    console.error(`\n${file} — ${errors.length} math error(s):`);
    for (const e of errors) {
      const mode = e.display ? "$$" : "$";
      const snippet = e.expr.length > 80 ? e.expr.slice(0, 77) + "..." : e.expr;
      console.error(`  at ${e.path}`);
      console.error(`    ${mode}${snippet}${mode}`);
      console.error(`    → ${e.err}`);
    }
  }

  console.log(`\nScanned ${totalExprs} math expressions in ${totalFiles} files.`);
  if (totalErrors > 0) {
    console.error(`\n✗ ${totalErrors} KaTeX validation error(s).`);
    process.exit(1);
  }
  console.log("✓ All math expressions parse cleanly.");
}

main();
