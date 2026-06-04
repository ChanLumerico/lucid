import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Sidebar, type SidebarItem } from "@/components/layout/Sidebar";
import { MobileSidebarProvider } from "@/components/layout/MobileSidebarContext";
import { loadApiData, getAllModuleSlugs } from "@/lib/api-loader";
import { isApiModule, isApiClassModule, isApiClass } from "@/lib/types";
import { packageLabel, segmentLabel, titleize } from "@/lib/labels";
import type { ApiMember, ApiModule } from "@/lib/types";

// ---------------------------------------------------------------------------
// Display labels (slug aliases + subcategory/segment labels + acronym-aware
// titleization) all live in ``@/lib/labels`` — the single source of truth.
// This file owns only the *navigation-specific* policy: ordering, tags, and
// the synthetic-parent / sub-package nesting that shapes the sidebar tree.
// ---------------------------------------------------------------------------

// Explicit display order for the `lucid` top-level (mirrors lucid/ on disk).
const LUCID_ORDER: string[] = [
  "tensor", "types", "dtypes", "device",
  "factories", "ops", "composite",
  "autograd", "predicates", "serialization",
  "globals", "threads", "dispatch", "vmap",
];

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
  // Model Zoo family roots — the label ("Vision Models" etc.) already says
  // everything; the real path tag is redundant noise in the sidebar.
  "lucid.models.vision":     null,
  "lucid.models.text":       null,
  "lucid.models.generative": null,
  // C++ engine — pinned to the bottom under its own divider; the path tag
  // ("lucid._C.engine") just duplicates what the divider already conveys.
  "lucid._C.engine":         null,
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

// Top-level slugs pinned to the **bottom** of the sidebar, in order.
// Used for low-level / engine-internal surfaces that we want present but
// not alphabetised into the middle of the user-facing Python namespaces.
const BOTTOM_PINNED: string[] = [
  "lucid._C.engine",
];

// ---------------------------------------------------------------------------
// Build sidebar tree — server-side, reads public/api-data/*.json
// ---------------------------------------------------------------------------

function memberLeaf(slug: string, m: ApiMember): SidebarItem {
  // Encode the class-kind into the badge code so the sidebar pill picks
  // up the matching colour token.  Keep ``"C"`` for plain classes so
  // existing call sites don't shift colour; add ``"D" / "A" / "P"`` for
  // dataclass / abstract / protocol.  Functions stay ``"f"``.
  let badge: string;
  if (isApiClass(m)) {
    switch (m.class_kind) {
      case "dataclass": badge = "D"; break;
      case "abstract":  badge = "A"; break;
      case "protocol":  badge = "P"; break;
      default:          badge = "C";
    }
  } else {
    badge = "f";
  }
  return {
    title: m.name,
    href: `/api/${slug}/${m.name}`,
    badge,
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
      pushBucket(segmentLabel(cat), group);
    }
  }

  // 2. Remaining named buckets — alphabetical by display label
  const remaining: Array<{ cat: string; label: string }> = [];
  for (const cat of buckets.keys()) {
    if (cat === null || seen.has(cat)) continue;
    remaining.push({ cat, label: segmentLabel(cat) });
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

// True for slugs of the form ``lucid.models.{vision|text|generative}.<family>`` —
// the leaf level under Model Zoo where each model family lives.  Used to swap
// the default subcategory-bucketed sidebar tree for the strict 4-slot layout
// (Config / Direct Model / Task Wrappers / Pretrained).
function isFamilyLeaf(slug: string): boolean {
  const parts = slug.split(".");
  if (parts.length !== 4) return false;
  if (parts[0] !== "lucid" || parts[1] !== "models") return false;
  return parts[2] === "vision" || parts[2] === "text" || parts[2] === "generative";
}

/** Build the 4-slot family-leaf sidebar item:
 *
 *   <CanonicalName>
 *   ├── <FamilyConfig>           (slot 1, direct leaf)
 *   ├── <DirectModelClass>       (slot 2, direct leaf)
 *   ├── Task Wrappers ──┐        (slot 3, sub-group: every *For* class)
 *   │   └── ...
 *   └── Pretrained ─────┘        (slot 4, sub-group: every factory function)
 *       └── ...
 *
 *  ``*Output`` dataclasses are hidden — they're forward-return types, not
 *  user-facing entry points.
 */
function buildFamilyLeafItem(slug: string, data: ApiModule): SidebarItem {
  const dirName = slug.split(".").pop() ?? slug;
  const title = data.canonical_name && data.canonical_name.length > 0
    ? data.canonical_name
    : titleize(dirName);

  const allMembers = data.members.filter(
    (m) => isApiClass(m) || m.kind === "function",
  );

  // Partition into the strict family slots.
  const configClass = allMembers.find(
    (m) => isApiClass(m) && m.name.endsWith("Config"),
  );
  const taskWrappers = allMembers.filter(
    (m) =>
      isApiClass(m) &&
      m.name.includes("For") &&
      !m.name.endsWith("Output") &&
      !m.name.endsWith("Config"),
  );
  // *Output classes are dataclasses for forward return types — not entry
  // points; suppress from the sidebar to keep the family tree clean.
  const outputClasses = allMembers.filter(
    (m) => isApiClass(m) && m.name.endsWith("Output"),
  );
  // Pretrained-weight enums (``<Family>Weights``).  The model-zoo-wide
  // ``Weights`` slot — every family owns its weight classes directly (the
  // ``lucid.models.weights`` aggregator is not documented as a package), so
  // they surface here next to the model rather than in a separate tree.
  const weightClasses = allMembers.filter(
    (m) => isApiClass(m) && m.name.endsWith("Weights"),
  );
  const directModel = allMembers.find(
    (m) =>
      isApiClass(m) &&
      m !== configClass &&
      !taskWrappers.includes(m) &&
      !outputClasses.includes(m) &&
      !weightClasses.includes(m),
  );
  const pretrained = allMembers.filter((m) => m.kind === "function");

  const items: SidebarItem[] = [];
  if (configClass) items.push(memberLeaf(slug, configClass));
  if (directModel) items.push(memberLeaf(slug, directModel));
  if (taskWrappers.length > 0) {
    items.push({
      title: "Task Wrappers",
      badge: `${taskWrappers.length}`,
      items: taskWrappers.map((m) => memberLeaf(slug, m)),
    });
  }
  if (pretrained.length > 0) {
    items.push({
      title: "Pretrained",
      badge: `${pretrained.length}`,
      items: pretrained.map((m) => memberLeaf(slug, m)),
    });
  }
  if (weightClasses.length > 0) {
    items.push({
      title: "Weights",
      badge: `${weightClasses.length}`,
      items: weightClasses.map((m) => memberLeaf(slug, m)),
    });
  }

  const badgeCount =
    (configClass ? 1 : 0) +
    (directModel ? 1 : 0) +
    taskWrappers.length +
    pretrained.length +
    weightClasses.length;

  return {
    title,
    // No tag on family-leaf entries — the canonical name (ResNet, BERT, …)
    // is the user-facing identifier; the raw ``lucid.models.vision.resnet``
    // path is just noise here.  Same rationale as the family-root tag
    // suppression in PACKAGE_TAG_OVERRIDES.
    href: `/api/${slug}`,
    badge: `${badgeCount}`,
    items,
  };
}

function buildModuleItem(slug: string): SidebarItem {
  let memberItems: SidebarItem[] | undefined;
  let badge: string | undefined;

  try {
    const data = loadApiData(slug);

    // Family-leaf pages get the strict 4-slot tree (Config / Direct /
    // Task Wrappers / Pretrained).  Bypass the default subcategory
    // bucketing entirely.
    if (isApiModule(data) && isFamilyLeaf(slug)) {
      return buildFamilyLeafItem(slug, data);
    }

    if (isApiModule(data)) {
      const members = data.members.filter(
        (m) => isApiClass(m) || m.kind === "function",
      );

      // Family-root pages (lucid.models.vision/text/generative) — and the
      // top-level lucid.models — carry a ``family_groups`` array instead of
      // a flat member list.  Show that count for the badge so users see
      // "Vision (45)" not "Vision (0)".
      if (data.family_groups && data.family_groups.length > 0 && members.length === 0) {
        badge = `${data.family_groups.length}`;
      } else if (members.length > 0) {
        // For the `lucid` top-level, skip the Tensor class itself — it has
        // its own dedicated entry at `lucid.tensor` where all of its
        // methods are surfaced.  Listing it twice in the sidebar would
        // just create a redundant single-item "Tensor" group.
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
  //
  // Private-segment skip: when a slug routes through a ``_``-prefixed
  // namespace (e.g. ``lucid._C.engine``), don't manufacture a synthetic
  // parent for it — ``lucid._C`` has no public-API meaning, and exposing
  // it as a sidebar header would only confuse readers.  Such slugs are
  // treated as top-level entries instead.
  const synthetic = new Set<string>();
  for (const slug of docSlugs) {
    const parts = slug.split(".");
    if (parts.length < 3) continue;
    // Skip synthetic-parent generation for slugs routed through a
    // ``_``-prefixed (private) namespace — they bubble to top-level
    // instead of producing a meaningless header like ``lucid._C``.
    if (parts.slice(1, -1).some((p) => p.startsWith("_"))) continue;
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

  // Top-level sort: TOP_PINNED first in declared order, then everything
  // else alphabetically by alias, then BOTTOM_PINNED last in declared
  // order (used for the C++ engine surface).  Nested levels stay
  // alphabetical.
  for (const [parent, kids] of childrenOf) {
    if (parent === null) {
      const rank = (s: string): number => {
        const top = TOP_PINNED.indexOf(s);
        if (top !== -1) return top;                          // 0..N-1
        const bot = BOTTOM_PINNED.indexOf(s);
        if (bot !== -1) return 1_000_000 + bot;              // sink
        return TOP_PINNED.length;                            // alphabetic middle
      };
      kids.sort((a, b) => {
        const rd = rank(a) - rank(b);
        if (rd !== 0) return rd;
        return packageLabel(a).localeCompare(packageLabel(b));
      });
    } else {
      kids.sort((a, b) => packageLabel(a).localeCompare(packageLabel(b)));
    }
  }

  // Slugs whose sub-packages are the *primary* content rather than
  // auxiliary — render them FIRST in the items list so the family-root
  // entries (e.g. Vision / Text / Generative under Model Zoo) sit above
  // the slug's own bucketed members instead of being buried at the bottom.
  const SUBPKGS_FIRST = new Set<string>(["lucid.models"]);

  /** Build a SidebarItem for a slug, recursively attaching any documented
   *  sub-packages as nested children. */
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
    const subPkgItems = subPkgs.map((s) => buildItem(s, slug));
    const ownItems = base.items ?? [];
    return {
      ...base,
      items: SUBPKGS_FIRST.has(slug)
        ? [...subPkgItems, ...ownItems]
        : [...ownItems, ...subPkgItems],
    };
  }

  const topLevel = childrenOf.get(null) ?? [];
  const items = topLevel.map((s) => buildItem(s, null));

  // Inject a visual divider directly before the first BOTTOM_PINNED slug so
  // the engine-internal surface (C++ Engine) reads as its own section rather
  // than the alphabetic tail of the Python namespaces.
  const firstBottomIdx = topLevel.findIndex((s) => BOTTOM_PINNED.includes(s));
  if (firstBottomIdx > 0) {
    items.splice(firstBottomIdx, 0, { title: "__bottom-pinned__", separator: true });
  }
  return items;
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
    <MobileSidebarProvider items={sidebar}>
      <div className="flex min-h-dvh flex-col">
        <Header />
        <div className="mx-auto flex w-full max-w-screen-2xl flex-1 gap-0 px-4 sm:px-6 pt-14">
          <Sidebar
            items={sidebar}
            className="sticky top-14 h-[calc(100dvh-3.5rem)]"
          />
          <main
            id="main-content"
            tabIndex={-1}
            className="flex-1 min-w-0 pt-10 pb-12 lg:px-8 focus:outline-none"
          >
            {children}
          </main>
        </div>
        <Footer />
      </div>
    </MobileSidebarProvider>
  );
}
