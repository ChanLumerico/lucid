import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Sidebar, type SidebarItem } from "@/components/layout/Sidebar";
import { loadApiData, getAllModuleSlugs } from "@/lib/api-loader";
import { isApiModule, isApiClassModule, isApiClass } from "@/lib/types";
import type { ApiMember } from "@/lib/types";

// ---------------------------------------------------------------------------
// Subcategory display: per-module ordering + friendly labels.
// Slugs come from build-api-data.py (file-basename fallback or explicit rules).
// Per-slug labels are best-effort — anything not listed gets auto-titlecased.
// ---------------------------------------------------------------------------

// Explicit display order for the `lucid` top-level (mirrors lucid/ on disk).
const LUCID_ORDER: string[] = [
  "tensor", "types", "dtypes", "device",
  "factories", "ops", "composite",
  "autograd", "predicates", "serialization",
  "globals", "threads", "dispatch", "vmap",
];

// Friendly display labels keyed by subcategory slug.  Anything missing here
// falls back to ``titleize(slug)``.
const LABEL_OVERRIDES: Record<string, string> = {
  // lucid top-level
  tensor: "Tensor",
  types: "Types & Protocols",
  dtypes: "Data Types",
  device: "Device",
  factories: "Tensor Creation",
  ops: "Tensor Operations",
  composite: "Composite Ops",
  autograd: "Autograd",
  predicates: "Predicates",
  serialization: "Serialization",
  globals: "Global Settings",
  threads: "Threading",
  dispatch: "Dispatch",
  vmap: "vmap",
  // nn / nn.functional file basenames
  modules: "Modules",
  hooks: "Hooks",
  activations: "Activations",
  activation: "Activations",
  attention: "Attention",
  container: "Containers",
  conv: "Convolution",
  dropout: "Dropout",
  flatten: "Flatten",
  linear: "Linear",
  loss: "Loss",
  normalization: "Normalization",
  padding: "Padding",
  pooling: "Pooling",
  rnn: "Recurrent",
  sparse: "Sparse",
  transformer: "Transformer",
  upsampling: "Upsampling",
  sampling: "Sampling",
  // optim
  sgd: "SGD",
  adam: "Adam / AdamW",
  lbfgs: "L-BFGS",
  lr_scheduler: "LR Schedulers",
  others: "Other Optimizers",
  // autograd
  function: "Function",
  graph: "Graph",
  gradcheck: "Gradient Check",
  checkpoint: "Checkpointing",
  profiler: "Profiler",
  // utils.data
  dataloader: "DataLoader",
  dataset: "Dataset",
  sampler: "Sampler",
  // amp
  autocast: "Autocast",
  grad_scaler: "Gradient Scaler",
  // nn.utils
  clip_grad: "Gradient Clipping",
  convert_parameters: "Parameter Conversion",
  fusion: "Fusion",
  parametrize: "Parametrization",
  parametrizations: "Parametrizations",
  prune: "Pruning",
  skip_init: "Skip Init",
  spectral_norm: "Spectral Norm",
  weight_norm: "Weight Norm",
  // lucid.ops arity buckets
  unary: "Unary",
  binary: "Binary",
  ternary: "Ternary",
  variadic: "Variadic",
  // distributions
  transforms: "Transforms",
  kl: "KL Divergence",
  normal: "Normal",
  bernoulli: "Bernoulli",
  categorical: "Categorical",
  uniform: "Uniform",
  exponential: "Exponential",
  gamma: "Gamma",
  student: "Student-t",
  multivariate: "Multivariate",
  matrix: "Matrix",
  discrete: "Discrete",
  continuous_extra: "Continuous (extra)",
  independent: "Independent",
  mixture: "Mixture",
  relaxed: "Relaxed",
  extra: "Extra",
};

function titleize(slug: string): string {
  return slug
    .split(/[_\-]/)
    .map((s) => (s.length === 0 ? s : s[0].toUpperCase() + s.slice(1)))
    .join(" ");
}

function labelFor(slug: string): string {
  return LABEL_OVERRIDES[slug] ?? titleize(slug);
}

// ---------------------------------------------------------------------------
// Top-level package aliases — every sidebar entry uses a human-friendly name
// instead of the raw Python slug.  Keeps the namespace consistent: half the
// slugs (lucid.creation, lucid.ops, lucid.ops.composite) are synthesised and
// don't exist in real Python anyway; showing all aliases avoids the
// confusion of "is lucid.creation a real package?" vs "is lucid.nn?".
// The slug (and therefore the URL like /api/lucid.nn/Linear) stays the same;
// only the displayed title changes.
// ---------------------------------------------------------------------------

const PACKAGE_LABELS: Record<string, string> = {
  "lucid.tensor":         "Tensor",
  "lucid.creation":       "Tensor Creation",
  "lucid.ops":            "Tensor Operations",
  "lucid.ops.composite":  "Composite Ops",
  "lucid.nn":             "Neural Networks",
  "lucid.nn.functional":  "Functional",
  "lucid.nn.init":        "Init",
  "lucid.nn.utils":       "NN Utils",
  "lucid.optim":          "Optimizers",
  "lucid.autograd":       "Autograd",
  "lucid.func":           "Functional Transforms",
  "lucid.linalg":         "Linear Algebra",
  "lucid.fft":            "FFT",
  "lucid.signal":         "Signal",
  "lucid.special":        "Special Functions",
  "lucid.distributions":  "Distributions",
  "lucid.einops":         "Einops",
  "lucid.amp":            "Mixed Precision",
  "lucid.profiler":       "Profiler",
  "lucid.serialization":  "Serialization",
  "lucid.utils":          "Utils",         // synthetic parent
  "lucid.utils.data":     "Data",
};

function packageLabel(slug: string): string {
  if (PACKAGE_LABELS[slug]) return PACKAGE_LABELS[slug];
  // Fallback: strip "lucid." prefix, titleize.
  const stripped = slug.startsWith("lucid.") ? slug.slice(6) : slug;
  return titleize(stripped);
}

// Real Python path tag shown next to the alias label.
//
// Default: the slug itself (real public package, e.g. ``lucid.nn``).
// Overrides:
//   - ``lucid.tensor`` → ``lucid.Tensor`` (class, not a module).
//   - Synthesised slugs (``lucid.creation``, ``lucid.ops``,
//     ``lucid.ops.composite``) → no tag.  The underlying source
//     (``_factories/``, ``_ops/``, ``_ops/composite/``) is intentionally
//     private and the alias is the only stable label, so showing a tag
//     would be misleading either way.
//   - Synthetic hierarchy headers (``lucid.utils``) → no tag.
const PACKAGE_TAG_OVERRIDES: Record<string, string | null> = {
  "lucid.tensor":        "lucid.Tensor",
  "lucid.creation":      null,
  "lucid.ops":           null,
  "lucid.ops.composite": null,
};

function packageTag(slug: string, isSynthetic: boolean): string | undefined {
  if (isSynthetic) return undefined;
  if (slug in PACKAGE_TAG_OVERRIDES) {
    return PACKAGE_TAG_OVERRIDES[slug] ?? undefined;
  }
  return slug;
}

// Top-level slugs that are pinned to the head of the sidebar in this exact
// order — the canonical Tensor entry plus its closest sibling concepts
// (creation, operations).  Everything else sorts alphabetically by alias
// after these.
const TOP_PINNED: string[] = [
  "lucid.tensor",
  "lucid.creation",
  "lucid.ops",
];

// ---------------------------------------------------------------------------
// Sidebar — flat package-per-entry, no artificial meta-categories
// ---------------------------------------------------------------------------
//
// Each module slug discovered by ``getAllModuleSlugs()`` becomes a top-level
// sidebar entry.  Sort order: ``lucid`` first, then ``lucid.tensor``, then
// the remaining slugs alphabetically.  No hardcoded category mapping — the
// sidebar shape mirrors exactly the set of packages produced by the build.

// ---------------------------------------------------------------------------
// Build sidebar tree — server-side, reads public/api-data/*.json
// ---------------------------------------------------------------------------

function memberLeaf(slug: string, m: ApiMember): SidebarItem {
  return {
    title: m.name,
    href: `/api/${slug}/${m.name}`,
    badge: isApiClass(m) ? "C" : "f",
  };
}

/** Group module members into subcategory sub-items mirroring the source tree.
 *
 *  ``subcategory`` is a slash-separated path (e.g. ``"modules/conv"``).  The
 *  renderer recursively buckets by the leading segment, so a module like
 *  ``lucid.nn`` (with members from ``lucid/nn/modules/*.py`` plus a few from
 *  ``lucid/nn/{hooks,parameter,module}.py``) produces a two-level tree —
 *  ``Modules → Convolution → Conv2d`` — that mirrors ``ls lucid/nn/``.
 *
 *  Flattening rules:
 *  - All members share the same subcategory → flat list.
 *  - Bucket has exactly 1 member → direct link (no "Foo (1) → Foo" wrapper).
 *  - A subdirectory bucket whose children have ≤ 1 unique subgroup → flat
 *    inside that bucket (no degenerate "Other" sub-tree).
 */
function groupMembers(
  slug: string,
  members: ApiMember[],
  explicitOrder?: string[],
): SidebarItem[] {
  // Bucket by the FIRST segment of subcategory ("modules/conv" → "modules").
  const buckets = new Map<string | null, ApiMember[]>();
  for (const m of members) {
    const first = m.subcategory?.split("/")[0] ?? null;
    if (!buckets.has(first)) buckets.set(first, []);
    buckets.get(first)!.push(m);
  }

  // Only one bucket total — nesting adds no value.
  if (buckets.size === 1) {
    return members.map((m) => memberLeaf(slug, m));
  }

  /** Recurse into a bucket: strip the leading segment from each member's
   *  subcategory, then group again.  Returns the list of child SidebarItems. */
  function buildSubtree(group: ApiMember[]): SidebarItem[] {
    const stripped: ApiMember[] = group.map((m) => {
      const rest = (m.subcategory ?? "").split("/").slice(1).join("/");
      return { ...m, subcategory: rest || null };
    });
    return groupMembers(slug, stripped);
  }

  /** Normalise a label/name for comparison: lowercase + strip spaces and
   *  underscores.  Used by the exact-match singleton flatten rule. */
  function norm(s: string): string {
    return s.toLowerCase().replace(/[\s_]+/g, "");
  }

  /** Push a bucket to the result, applying singleton flattening + recursive
   *  subgrouping when the bucket's members carry deeper subcategory paths.
   *
   *  Singleton flatten rule (anti-redundancy):
   *  - If a bucket has exactly 1 member AND the member's name normalises to
   *    the same string as the bucket's label, render it as a direct link
   *    (no "Foo → Foo" wrapper).  Examples that flatten: ``Module → Module``,
   *    ``Parameter → Parameter``, ``device → device``.
   *  - Otherwise keep the bucket — preserves semantic context like
   *    ``Attention (1) → scaled_dot_product_attention`` where the bucket
   *    label adds info the leaf name doesn't carry. */
  function pushBucket(label: string, group: ApiMember[]) {
    if (group.length === 1 && norm(group[0].name) === norm(label)) {
      result.push(memberLeaf(slug, group[0]));
      return;
    }
    const anyDeeper = group.some((m) => (m.subcategory ?? "").includes("/"));
    const items = anyDeeper
      ? buildSubtree(group)
      : group.map((m) => memberLeaf(slug, m));
    result.push({
      title: label,
      badge: `${group.length}`,
      items,
    });
  }

  const result: SidebarItem[] = [];
  const seen = new Set<string>();

  // 1. Explicit order first (only buckets that actually have members)
  if (explicitOrder) {
    for (const cat of explicitOrder) {
      const group = buckets.get(cat);
      if (!group?.length) continue;
      seen.add(cat);
      pushBucket(labelFor(cat), group);
    }
  }

  // 2. Remaining named buckets — alphabetical by display label
  const remaining: Array<{ cat: string; label: string }> = [];
  for (const cat of buckets.keys()) {
    if (cat === null || seen.has(cat)) continue;
    remaining.push({ cat, label: labelFor(cat) });
  }
  remaining.sort((a, b) => a.label.localeCompare(b.label));
  for (const { cat, label } of remaining) {
    pushBucket(label, buckets.get(cat)!);
  }

  // 3. Null-bucket members (defined directly in the module's __init__.py)
  //    are surfaced as direct links rather than wrapped in an "Other" group.
  //    This matches user expectation: a tensor-creation factory at
  //    `lucid/nn/__init__.py` (if any) should sit at the module's top level.
  const nullGroup = buckets.get(null);
  if (nullGroup?.length) {
    for (const m of nullGroup) result.push(memberLeaf(slug, m));
  }

  return result;
}

function buildModuleItem(slug: string): SidebarItem {
  let memberItems: SidebarItem[] | undefined;
  let badge: string | undefined;

  try {
    const data = loadApiData(slug);

    if (isApiModule(data)) {
      const members = data.members.filter(
        (m) => isApiClass(m) || m.kind === "function",
      );

      if (members.length > 0) {
        // For the `lucid` top-level, skip the Tensor class itself — it has
        // its own dedicated entry at `lucid.tensor` where all 260 methods
        // are surfaced.  Listing it twice in the sidebar would just create
        // a redundant single-item "Tensor" group.
        const visible =
          slug === "lucid"
            ? members.filter((m) => m.subcategory !== "tensor")
            : members;
        badge = `${visible.length}`;
        // Path-based subcategory grouping applies to every module.  The
        // lucid top-level uses an explicit order; others fall back to
        // alphabetical-by-display-label inside groupMembers.
        const order = slug === "lucid" ? LUCID_ORDER : undefined;
        memberItems = groupMembers(slug, visible, order);
      }
    } else if (isApiClassModule(data)) {
      // lucid.tensor — the module IS the class; its methods are not listed
      // here per the user's request ("클래스 내 메소드는 제외").
      badge = `${data.methods.length}`;
      memberItems = undefined;
    }
  } catch {
    // JSON not yet built — gracefully degrade to a plain link
  }

  return {
    title: packageLabel(slug),
    tag: packageTag(slug, false),
    href: `/api/${slug}`,
    badge,
    items: memberItems,
  };
}

/** Find a slug's documented parent (the longest documented prefix), excluding
 *  the bare ``lucid`` root.  Depth-1 (``lucid``) and depth-2 (``lucid.nn``,
 *  ``lucid.optim`` …) slugs are always top-level — they sit as siblings to
 *  mirror how Python's own docs (PyTorch, NumPy) lay out their first-level
 *  subpackages.  Depth-3+ slugs (``lucid.nn.functional`` etc.) nest under
 *  whichever shorter prefix is also in the slug set. */
function findParent(slug: string, slugSet: Set<string>): string | null {
  const parts = slug.split(".");
  if (parts.length <= 2) return null;
  for (let i = parts.length - 1; i >= 2; i--) {
    const candidate = parts.slice(0, i).join(".");
    if (slugSet.has(candidate)) return candidate;
  }
  return null;
}

function buildApiSidebar(): SidebarItem[] {
  const docSlugs = new Set(getAllModuleSlugs());

  // Synthetic parents: any depth-3+ slug whose intermediate ancestors aren't
  // in the doc set gets a non-clickable placeholder so the Python hierarchy
  // is preserved.  Example: ``lucid.utils.data`` exists but ``lucid.utils``
  // is `_EXCLUDED_SELF` — we still inject a ``lucid.utils`` sidebar header
  // (no link, just a toggle) so ``Data`` shows up under it instead of at
  // the top level.
  const synthetic = new Set<string>();
  for (const slug of docSlugs) {
    const parts = slug.split(".");
    if (parts.length < 3) continue;
    let hasAncestor = false;
    for (let i = parts.length - 1; i >= 2; i--) {
      if (docSlugs.has(parts.slice(0, i).join("."))) {
        hasAncestor = true;
        break;
      }
    }
    if (!hasAncestor) synthetic.add(parts.slice(0, 2).join("."));
  }

  const allSlugs = new Set([...docSlugs, ...synthetic]);

  // Bucket each slug under its parent (or null for top-level).
  const childrenOf = new Map<string | null, string[]>();
  for (const slug of allSlugs) {
    const parent = findParent(slug, allSlugs);
    if (!childrenOf.has(parent)) childrenOf.set(parent, []);
    childrenOf.get(parent)!.push(slug);
  }

  // Top-level sort: TOP_PINNED entries in their given order, then everything
  // else alphabetically by alias.  Nested levels sort alphabetically by alias.
  for (const [parent, kids] of childrenOf) {
    if (parent === null) {
      const pinRank = (s: string) => {
        const i = TOP_PINNED.indexOf(s);
        return i === -1 ? TOP_PINNED.length : i;
      };
      kids.sort((a, b) => {
        const rd = pinRank(a) - pinRank(b);
        if (rd !== 0) return rd;
        return packageLabel(a).localeCompare(packageLabel(b));
      });
    } else {
      kids.sort((a, b) => packageLabel(a).localeCompare(packageLabel(b)));
    }
  }

  /** Build a SidebarItem for a slug, recursively attaching any documented
   *  sub-packages as nested children (placed AT THE END of the item's
   *  existing children, after the slug's own member subcategories). */
  function buildItem(slug: string, parent: string | null): SidebarItem {
    const isSynthetic = synthetic.has(slug);
    const base: SidebarItem = isSynthetic
      ? { title: packageLabel(slug), tag: packageTag(slug, true) }
      : buildModuleItem(slug);
    // ``buildModuleItem`` already runs the slug through ``packageLabel`` so
    // titles are aliases at every depth (e.g. ``Functional`` for
    // ``lucid.nn.functional``).  No further prefix-stripping needed.
    const subPkgs = childrenOf.get(slug) ?? [];
    if (subPkgs.length === 0) return base;
    return {
      ...base,
      items: [
        ...(base.items ?? []),
        ...subPkgs.map((s) => buildItem(s, slug)),
      ],
    };
  }

  const topLevel = childrenOf.get(null) ?? [];
  return topLevel.map((s) => buildItem(s, null));
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

export default function ApiLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const sidebar = buildApiSidebar();

  return (
    <div className="flex min-h-dvh flex-col">
      <Header />
      <div className="mx-auto flex w-full max-w-screen-xl flex-1 gap-0 px-4 sm:px-6 pt-14">
        <Sidebar
          items={sidebar}
          className="sticky top-14 h-[calc(100dvh-3.5rem)]"
        />
        <main className="flex-1 min-w-0 pt-10 pb-12 lg:px-8">{children}</main>
      </div>
      <Footer />
    </div>
  );
}
