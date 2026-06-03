/**
 * Docs contract audit — exhaustive conformance check for the built API site.
 *
 *   pnpm build && node scripts/audit-docs.mjs
 *
 * Crawls every generated page in ``out/`` (the complete page set), cross-refs
 * the ``public/api-data`` JSON, and checks each COMPONENT TYPE against a fixed
 * CONTRACT (a set of invariants).  Every deviation is classified as:
 *   - conforming            (passes)
 *   - registered-exception  (matches a declarative waiver in audit/exceptions)
 *   - VIOLATION             (unregistered → fails the audit, exit 1)
 *
 * Plus DEAD-WAIVER detection (registry entries that never matched) so the
 * exception list can't rot.  This turns the design rules in
 * obsidian/architecture/arch-docs-site-rules.md into executable, regression-
 * proof gates — the production-level alternative to ad-hoc spot-fixing.
 */

import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join, relative } from "node:path";
import { parse } from "node-html-parser";
import { WAIVERS, matchWaiver } from "./audit/exceptions.mjs";

const ROOT = process.cwd();                       // run from web/
const OUT = join(ROOT, "out");
const API_DATA = join(ROOT, "public", "api-data");
const LABELS_TS = join(ROOT, "src", "lib", "labels.ts");
const BASE = "/lucid";

// ── reST / markdown leak patterns (unambiguous — used by prose contracts) ────
const REST_LEAKS = [
  { name: "cross-ref role", re: /:(?:class|func|meth|mod|attr|obj|data|exc|ref|file|cite|term|doc|envvar|samp|command|option|paramref|py:[a-z]+):`?/ },
  { name: "directive", re: /\.\.[ \t]+[a-z][a-z-]*::/ },
  { name: "reST hyperlink", re: /<https?:\/\/[^>]+>`_/ },
  { name: "double-backtick", re: /``[^`]/ },
];
// Inline math that escaped KaTeX: a $…\cmd…$ visible as text.
const RAW_MATH = /\$[^$\n]{0,200}?\\[a-zA-Z]+[^$\n]{0,200}?\$/;

// ───────────────────────────── helpers ──────────────────────────────────────

function walkHtml(dir, acc = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) walkHtml(p, acc);
    else if (name.endsWith(".html")) acc.push(p);
  }
  return acc;
}

/** out/api/lucid.fft.html → /api/lucid.fft ; out/index.html → / */
function fileToRoute(file) {
  let r = "/" + relative(OUT, file).replace(/\\/g, "/");
  r = r.replace(/\/index\.html$/, "").replace(/\.html$/, "");
  return r === "" ? "/" : r;
}

/** /lucid/api/lucid.fft#x?y → /api/lucid.fft (or null if not base-prefixed) */
function linkToRoute(href) {
  let h = href.split("#")[0].split("?")[0];
  if (h === BASE) return "/";
  if (!h.startsWith(BASE + "/")) return null;
  h = h.slice(BASE.length);
  if (h.length > 1 && h.endsWith("/")) h = h.slice(0, -1);
  return h;
}

const SKIP_HREF = (h) =>
  !h ||
  /^(https?:|mailto:|#|\/\/)/.test(h) ||
  h.startsWith("/_next") ||
  h.startsWith(BASE + "/_next") ||
  /\.(svg|png|jpe?g|ico|xml|txt|json|css|js|woff2?|webmanifest|bib)$/.test(h);

function stripScripts(html) {
  return html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, "");
}

function loadJson(p) {
  try { return JSON.parse(readFileSync(p, "utf8")); } catch { return null; }
}

// ───────────────────────── load corpus ──────────────────────────────────────

if (!existsSync(OUT)) {
  console.error("✗ out/ not found — run `pnpm build` first.");
  process.exit(2);
}

const files = walkHtml(OUT);
const routeSet = new Set(files.map(fileToRoute));
// Case-insensitive view: macOS/APFS collides e.g. AlexNet.html / alexnet.html
// (model class vs factory) into one file, so the class route looks "missing"
// locally though both exist on case-sensitive CI.  Resolve case-insensitively
// — Next generates pages and links from the same member names, so a real case
// mismatch is impossible by construction.
const routeSetLower = new Set([...routeSet].map((r) => r.toLowerCase()));

const apiBySlug = new Map();      // slug → module JSON
for (const f of readdirSync(API_DATA)) {
  if (!f.endsWith(".json") || f.startsWith("_")) continue;
  const d = loadJson(join(API_DATA, f));
  if (d) apiBySlug.set(f.slice(0, -5), d);
}

// PACKAGE_LABELS keys declared in labels.ts (for the label.coverage contract).
const labelKeys = new Set();
if (existsSync(LABELS_TS)) {
  const src = readFileSync(LABELS_TS, "utf8");
  const blk = src.slice(src.indexOf("const PACKAGE_LABELS"));
  for (const m of blk.matchAll(/"(lucid[\w.]*)":/g)) labelKeys.add(m[1]);
}

const isFamilyLeaf = (slug) => {
  const p = slug.split(".");
  return p.length === 4 && p[0] === "lucid" && p[1] === "models" &&
    ["vision", "text", "generative"].includes(p[2]);
};

const global = { files, routeSet, apiBySlug, labelKeys };

// ─────────────────────────── contracts ──────────────────────────────────────
//
// Each contract: { id, component, kind: "page"|"global", applies?(ctx),
//   check(ctxOrGlobal) → [{ scope, detail }] }.  ``scope`` is the slug/route a
// waiver matches against.

const PAGE_CONTRACTS = [
  // ── Links (every page) ─────────────────────────────────────────────────────
  {
    id: "link.basepath", component: "InternalLink", kind: "page",
    check(ctx) {
      const bad = [];
      for (const a of ctx.dom.querySelectorAll("a")) {
        const h = a.getAttribute("href");
        if (SKIP_HREF(h) || !h.startsWith("/")) continue;
        if (h !== BASE && !h.startsWith(BASE + "/"))
          bad.push({ scope: ctx.scope, detail: `<a href="${h}"> missing ${BASE} basePath` });
      }
      return bad;
    },
  },
  {
    id: "link.resolves", component: "InternalLink", kind: "page",
    check(ctx) {
      const bad = [];
      for (const a of ctx.dom.querySelectorAll("a")) {
        const h = a.getAttribute("href");
        if (SKIP_HREF(h) || !h.startsWith(BASE)) continue;
        const r = linkToRoute(h);
        if (r && !routeSet.has(r) && !routeSetLower.has(r.toLowerCase()))
          bad.push({ scope: ctx.scope, detail: `link → ${h} has no generated page` });
      }
      return bad;
    },
  },
  // ── Prose / rendering (every page) ─────────────────────────────────────────
  {
    id: "prose.no-visible-rest", component: "ProseField", kind: "page",
    check(ctx) {
      const out = [];
      for (const { name, re } of REST_LEAKS) {
        const m = ctx.visible.match(re);
        if (m) out.push({ scope: ctx.scope, detail: `${name} leaked: "${m[0].slice(0, 40)}"` });
      }
      return out;
    },
  },
  {
    id: "math.no-raw-leak", component: "MathBlock", kind: "page",
    check(ctx) {
      const m = ctx.visible.match(RAW_MATH);
      return m ? [{ scope: ctx.scope, detail: `unrendered math: "${m[0].slice(0, 40)}"` }] : [];
    },
  },
  {
    id: "doctest.not-blockquote", component: "ExamplesBlock", kind: "page",
    check(ctx) {
      const bad = ctx.dom.querySelectorAll("blockquote").some((b) => b.text.includes(">>>"));
      return bad ? [{ scope: ctx.scope, detail: "doctest >>> rendered as <blockquote> (should be a code fence)" }] : [];
    },
  },
  // ── Module overview page ───────────────────────────────────────────────────
  {
    id: "module.has-heading", component: "ModulePage", kind: "page",
    applies: (ctx) => ctx.kind === "api-module",
    check(ctx) {
      return ctx.dom.querySelector("h1") ? [] : [{ scope: ctx.scope, detail: "no <h1> module heading" }];
    },
  },
  {
    id: "card.well-formed", component: "SubpackageCard", kind: "page",
    applies: (ctx) => ctx.kind === "api-module",
    check(ctx) {
      const out = [];
      const secs = ctx.dom.querySelectorAll("section").filter((s) =>
        /^(Model Families|Infrastructure|Sub-packages)$/.test(s.querySelector("h2")?.text?.trim() || ""));
      for (const sec of secs) {
        for (const card of sec.querySelectorAll("a")) {
          const title = card.querySelector("h3")?.text?.trim();
          if (!title) out.push({ scope: ctx.scope, detail: `sub-package card with empty title (href=${card.getAttribute("href")})` });
        }
      }
      return out;
    },
  },
  // ── UI design language (rendered-class conformance) ────────────────────────
  // The design rules in arch-docs-site-rules.md, made executable: every
  // component must use the canonical token recipe so the visual language stays
  // uniform.  Drift (a card with rounded-md, an off-palette text-gray-400) is a
  // build failure, not a code-review catch.
  {
    id: "ui.card-shape", component: "Card", kind: "page",
    applies: (ctx) => ctx.kind === "api-module",
    check(ctx) {
      const out = [];
      const secs = ctx.dom.querySelectorAll("section").filter((s) =>
        /^(Model Families|Infrastructure|Sub-packages|Classes|Functions)$/.test(s.querySelector("h2")?.text?.trim() || ""));
      for (const sec of secs) {
        for (const a of sec.querySelectorAll("a")) {
          const cls = a.getAttribute("class") || "";
          // Card links are block/flex containers tagged ``group``; skip inline links.
          if (!/\bgroup\b/.test(cls) || !/\b(block|flex)\b/.test(cls)) continue;
          if (!/\brounded-xl\b/.test(cls) || !/border-lucid-border/.test(cls) || !/bg-lucid-surface/.test(cls))
            out.push({ scope: ctx.scope, detail: `card off-recipe (need rounded-xl + border-lucid-border + bg-lucid-surface): ${a.getAttribute("href")}` });
        }
      }
      return out;
    },
  },
  {
    id: "ui.section-heading", component: "SectionHeading", kind: "page",
    applies: (ctx) => ctx.kind === "api-module",
    check(ctx) {
      const out = [];
      for (const sec of ctx.dom.querySelectorAll("section")) {
        const h2 = sec.querySelector("h2");
        const t = h2?.text?.trim() || "";
        if (!/^(Model Families|Infrastructure|Sub-packages|Classes|Functions)$/.test(t)) continue;
        const cls = h2.getAttribute("class") || "";
        if (!/tracking-widest/.test(cls) || !/uppercase/.test(cls) || !/text-lucid-text-disabled/.test(cls))
          out.push({ scope: ctx.scope, detail: `section heading "${t}" off-recipe (need tracking-widest uppercase text-lucid-text-disabled)` });
      }
      return out;
    },
  },
  {
    id: "ui.palette", component: "DesignToken", kind: "page",
    check(ctx) {
      // No off-design-system colours: only lucid-* / api-* / task-* tokens are
      // allowed.  Catches drift to stock Tailwind palettes or arbitrary hex.
      const m = ctx.mainHtml.match(/\b(?:text|bg|border|ring|fill|stroke|decoration)-(?:gray|slate|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\d{2,3}\b/)
        || ctx.mainHtml.match(/\b(?:text|bg|border)-\[#[0-9a-fA-F]{3,8}\]/);
      return m ? [{ scope: ctx.scope, detail: `off-palette colour class: ${m[0]}` }] : [];
    },
  },
  {
    id: "models.weights-separated", component: "ModulePage", kind: "page",
    applies: (ctx) => ctx.route === "/api/lucid.models",
    check(ctx) {
      const sectionCards = (heading) => {
        const sec = ctx.dom.querySelectorAll("section").find((s) => s.querySelector("h2")?.text?.trim() === heading);
        return sec ? sec.querySelectorAll("a").map((a) => a.getAttribute("href") || "") : null;
      };
      const fams = sectionCards("Model Families");
      const infra = sectionCards("Infrastructure");
      const out = [];
      if (!fams) out.push({ scope: ctx.scope, detail: "no 'Model Families' section" });
      else if (fams.some((h) => h.includes("lucid.models.weights")))
        out.push({ scope: ctx.scope, detail: "weights card is inside 'Model Families' (must be 'Infrastructure')" });
      if (!infra || !infra.some((h) => h.includes("lucid.models.weights")))
        out.push({ scope: ctx.scope, detail: "weights card missing from a separate 'Infrastructure' section" });
      return out;
    },
  },
];

const GLOBAL_CONTRACTS = [
  {
    id: "module.has-summary", component: "ModulePage", kind: "global", severity: "warn",
    check(g) {
      const out = [];
      for (const [slug, d] of g.apiBySlug)
        if (d.kind === "module" && !(d.summary && d.summary.trim()))
          out.push({ scope: slug, detail: `${slug}: module has no summary` });
      return out;
    },
  },
  {
    id: "member.has-signature", component: "MemberPage", kind: "global",
    check(g) {
      const out = [];
      for (const [slug, d] of g.apiBySlug) {
        if (d.kind !== "module") continue;
        for (const m of d.members || [])
          if ((m.kind === "function" || m.kind === "class") && !m.signature && !(m.methods))
            out.push({ scope: `${slug}/${m.name}`, detail: `${slug}.${m.name}: no signature` });
      }
      return out;
    },
  },
  {
    id: "prose.no-rest-leak", component: "ProseField", kind: "global",
    check(g) {
      const out = [];
      const fields = ["summary", "extended", "theory", "citation"];
      const scan = (slug, s) => {
        if (typeof s !== "string") return;
        // Ignore protected regions — fenced code (```…```) and math ($$…$$ / $…$)
        // legitimately contain `` (LaTeX left-quotes) and \commands.
        const prose = s
          .replace(/```[\s\S]*?```/g, "")
          .replace(/\$\$[\s\S]*?\$\$/g, "")
          .replace(/\$[^$\n]+?\$/g, "");
        for (const { name, re } of REST_LEAKS)
          if (re.test(prose)) out.push({ scope: slug, detail: `${slug}: ${name} in prose` });
      };
      const walk = (slug, o) => {
        if (Array.isArray(o)) o.forEach((x) => walk(slug, x));
        else if (o && typeof o === "object")
          for (const [k, v] of Object.entries(o)) {
            if (fields.includes(k) || k === "description") scan(slug, v);
            else if (k === "notes" || k === "warns") (v || []).forEach((x) => scan(slug, x));
            else walk(slug, v);
          }
      };
      for (const [slug, d] of g.apiBySlug) walk(slug, d);
      return out;
    },
  },
  {
    id: "label.coverage", component: "SidebarEntry", kind: "global",
    check(g) {
      const out = [];
      for (const slug of g.apiBySlug.keys()) {
        if (isFamilyLeaf(slug)) continue;        // family leaves use canonical_name
        if (slug === "lucid") continue;
        if (!g.labelKeys.has(slug))
          out.push({ scope: slug, detail: `${slug}: no PACKAGE_LABELS alias (add to labels.ts or waive)` });
      }
      return out;
    },
  },
  {
    id: "sidebar.complete", component: "SidebarEntry", kind: "global",
    check(g) {
      // Use the /api landing page's sidebar as the canonical rendering.
      const landing = join(OUT, "api.html");
      const idx = join(OUT, "api", "index.html");
      const file = existsSync(landing) ? landing : existsSync(idx) ? idx : null;
      if (!file) return [{ scope: "/api", detail: "cannot locate /api page to read sidebar" }];
      const dom = parse(readFileSync(file, "utf8"));
      const hrefs = new Set(dom.querySelectorAll("a").map((a) => a.getAttribute("href")));
      const out = [];
      for (const slug of g.apiBySlug.keys()) {
        const depth = slug.split(".").length;
        if (depth !== 2) continue;               // top-level packages must be in the sidebar
        if (slug === "lucid") continue;
        if (!hrefs.has(`${BASE}/api/${slug}`))
          out.push({ scope: slug, detail: `${slug}: top-level package not linked in sidebar` });
      }
      return out;
    },
  },
];

// ─────────────────────────── run ────────────────────────────────────────────

const hitWaivers = new Set();
const findings = [];   // { contractId, component, scope, detail, status }

function record(contract, raw) {
  const sev = contract.severity || "error";   // "error" = hard violation; "warn" = advisory
  for (const f of raw) {
    const waiver = matchWaiver(contract.id, f.scope, hitWaivers);
    const status = waiver ? "exception" : sev === "warn" ? "warning" : "violation";
    findings.push({ contractId: contract.id, component: contract.component, scope: f.scope, detail: f.detail, status, waiver });
  }
}

// Per-page contracts.
let pageCount = 0;
for (const file of files) {
  const route = fileToRoute(file);
  let kind = "other";
  let scope = route, data = null, memberName = null;
  if (route === "/") kind = "home";
  else if (route === "/api") kind = "api-landing";
  else if (route === "/changelog") kind = "changelog";
  else if (route.startsWith("/docs")) kind = "docs";
  else if (route.startsWith("/api/")) {
    const rest = route.slice(5).split("/");
    scope = rest.length === 1 ? rest[0] : `${rest[0]}/${rest.slice(1).join("/")}`;
    data = apiBySlug.get(rest[0]) || null;
    if (rest.length === 1) kind = "api-module";
    else { kind = "api-member"; memberName = rest.slice(1).join("/"); }
  }
  const dom = parse(readFileSync(file, "utf8"));
  dom.querySelectorAll("script,style,template,noscript").forEach((n) => n.remove());
  // Visible-text checks scope to <main> only: Next's RSC hydration payload
  // (serialized props) lives in <script> blocks at the end of <body>, which
  // node-html-parser doesn't always strip cleanly — scoping to the rendered
  // content area is the robust way to ignore it.
  const main = dom.querySelector("#main-content") || dom.querySelector("main") || dom;
  const ctx = { route, scope, kind, data, memberName, dom, visible: main.text, mainHtml: main.toString() };
  for (const c of PAGE_CONTRACTS) {
    if (c.applies && !c.applies(ctx)) continue;
    record(c, c.check(ctx));
  }
  pageCount++;
}

// Global contracts.
for (const c of GLOBAL_CONTRACTS) record(c, c.check(global));

// ─────────────────────────── report ─────────────────────────────────────────

const violations = findings.filter((f) => f.status === "violation");
const warnings = findings.filter((f) => f.status === "warning");
const exceptions = findings.filter((f) => f.status === "exception");
const deadWaivers = WAIVERS.filter((w) => !hitWaivers.has(w.id));

const byContract = (list) => {
  const m = new Map();
  for (const f of list) m.set(f.contractId, (m.get(f.contractId) || 0) + 1);
  return [...m.entries()].sort((a, b) => b[1] - a[1]);
};

console.log("─".repeat(72));
console.log(`Docs contract audit — ${pageCount} pages, ${apiBySlug.size} api modules`);
console.log("─".repeat(72));
console.log(`contracts: ${PAGE_CONTRACTS.length + GLOBAL_CONTRACTS.length}   waivers: ${WAIVERS.length}`);
console.log(`exceptions: ${exceptions.length}   warnings: ${warnings.length}   VIOLATIONS: ${violations.length}\n`);

if (exceptions.length) {
  console.log("Registered exceptions (waived):");
  for (const [c, n] of byContract(exceptions)) console.log(`  · ${c}: ${n}`);
  console.log();
}

if (warnings.length) {
  console.log("⚠️  Warnings (advisory — do not fail the audit):");
  for (const [c, n] of byContract(warnings)) {
    console.log(`  · ${c}: ${n}`);
    for (const w of warnings.filter((x) => x.contractId === c).slice(0, 10))
      console.log(`        ${w.scope}: ${w.detail}`);
  }
  console.log();
}

if (deadWaivers.length) {
  console.log("⚠️  DEAD waivers (registered but never matched — remove or fix scope):");
  for (const w of deadWaivers) console.log(`  · ${w.id}`);
  console.log();
}

if (violations.length) {
  console.log("✗ VIOLATIONS:");
  for (const [c, n] of byContract(violations)) {
    console.log(`\n  [${c}] — ${n}`);
    const sample = violations.filter((v) => v.contractId === c).slice(0, 12);
    for (const v of sample) console.log(`      ${v.scope}: ${v.detail}`);
    if (n > sample.length) console.log(`      … +${n - sample.length} more`);
  }
  console.log(`\n✗ AUDIT FAILED — ${violations.length} unregistered violation(s).`);
  process.exit(1);
}

console.log("✓ AUDIT PASSED — every component conforms (or is a registered exception).");
process.exit(0);
