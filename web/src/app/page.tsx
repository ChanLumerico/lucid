import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { FadeIn, FadeInStagger } from "@/components/motion/FadeIn";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import {
  Cpu,
  Boxes,
  Code2,
  Layers,
  BrainCircuit,
  Sigma,
  Gauge,
  Workflow,
  GitBranch,
  ArrowRight,
} from "lucide-react";
import { highlight } from "@/lib/shiki";
import { cn } from "@/lib/utils";
import { buildMeta } from "@/lib/build-meta";
import { loadApiData, getAllModuleSlugs } from "@/lib/api-loader";
import { isApiModule } from "@/lib/types";

// ---------------------------------------------------------------------------
// Static features — replaced emoji with on-brand lucide icons so the row
// reads as a coherent system rather than a sticker collection.
// ---------------------------------------------------------------------------

const FEATURES = [
  {
    icon: Cpu,
    title: "MLX Native GPU",
    description:
      "Metal-accelerated compute on every forward and backward pass. No CUDA, no compromise — purpose-built for Apple Silicon.",
  },
  {
    icon: Gauge,
    title: "Accelerate CPU Kernels",
    description:
      "vDSP, vForce, and BLAS/LAPACK from Apple's Accelerate framework power the CPU stream. Zero third-party dependencies.",
  },
  {
    icon: Workflow,
    title: "PyTorch-compatible API",
    description:
      "Familiar interface — nn.Module, autograd, optim, DataLoader — so you can focus on the model, not the framework.",
  },
  {
    icon: Sigma,
    title: "Comprehensive Op Surface",
    description:
      "linalg, fft, einops, distributions, signal, special math — every standard primitive plus the model zoo.",
  },
] as const;

const INSTALL_CMD = "pip install lucid";

const HERO_CODE = `import lucid
import lucid.nn as nn
from lucid.optim import AdamW

# Mac Studio M4 Max — MLX on GPU, Accelerate on CPU.
device = lucid.metal_if_available()
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.GELU(),
    nn.Linear(256, 10),
).to(device)

opt = AdamW(model.parameters(), lr=3e-4)
for x, y in dataloader:
    loss = nn.functional.cross_entropy(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step()`;

// ---------------------------------------------------------------------------
// Featured Model-Zoo families — hand-curated subset anchoring the
// (currently 49, grows over time) full roster.  Each slug must resolve
// to a JSON file under ``public/api-data/`` (model family detail page).
// ---------------------------------------------------------------------------

const FEATURED_FAMILIES: Array<{
  slug: string;
  category: string;
  blurb: string;
}> = [
  { slug: "lucid.models.vision.resnet",     category: "Vision",     blurb: "Deep residual learning — He et al. 2015." },
  { slug: "lucid.models.vision.vit",        category: "Vision",     blurb: "Vision Transformer — Dosovitskiy 2020." },
  { slug: "lucid.models.vision.convnext",   category: "Vision",     blurb: "ConvNet for the 2020s — Liu 2022." },
  { slug: "lucid.models.text.bert",         category: "Text",       blurb: "Bidirectional encoder — Devlin 2018." },
  { slug: "lucid.models.text.gpt2",         category: "Text",       blurb: "Autoregressive decoder — Radford 2019." },
  { slug: "lucid.models.generative.dcgan",  category: "Generative", blurb: "Deep convolutional GAN — Radford 2015." },
];

// ---------------------------------------------------------------------------
// Module quick-access — six top-trafficked surfaces, mirrors what the
// sidebar's TOP_PINNED + main API landing prioritise.
// ---------------------------------------------------------------------------

function buildQuickModules(stats: ReturnType<typeof computeStats>): Array<{
  slug: string;
  label: string;
  icon: typeof Cpu;
  blurb: string;
}> {
  // Blurbs interpolate the same counts the StatsStrip / Model Zoo section
  // render — single source of truth is ``computeStats()`` reading the build's
  // JSON payloads, so when a new layer / optimizer / family lands no copy
  // edit is needed here.
  return [
    { slug: "lucid.tensor",    label: "Tensor",           icon: Boxes,        blurb: `The value type — ${stats.tensorMethods} methods across views, casts, autograd lifecycle, bridges.` },
    { slug: "lucid.nn",        label: "Neural Networks",  icon: Workflow,     blurb: `Module, Parameter, and ${stats.nnClasses} layer classes.` },
    { slug: "lucid.autograd",  label: "Autograd",         icon: GitBranch,    blurb: "Function, grad, gradcheck, functional transforms." },
    { slug: "lucid.optim",     label: "Optimizers",       icon: Gauge,        blurb: `${stats.optimizers} optimizers + ${stats.lrSchedulers} LR schedulers.` },
    { slug: "lucid.models",    label: "Model Zoo",        icon: BrainCircuit, blurb: `${stats.modelFamilies} paper-cited families with pretrained factories.` },
    { slug: "lucid._C.engine", label: "C++ Engine",       icon: Cpu,          blurb: "Storage, ops, autograd graph, backend dispatch." },
  ];
}

// ---------------------------------------------------------------------------
// Build-time stats — pull real counts from the API JSONs the docs site
// already builds.  Renders the 4-cell "by the numbers" strip without
// any maintenance burden when the surface grows or shrinks.
// ---------------------------------------------------------------------------

function computeStats(): {
  tensorMethods: number;
  nnClasses: number;
  engineMembers: number;
  modelFamilies: number;
  optimizers: number;
  lrSchedulers: number;
} {
  const fallback = {
    tensorMethods: 0, nnClasses: 0, engineMembers: 0,
    modelFamilies: 0, optimizers: 0, lrSchedulers: 0,
  };
  try {
    const tensor = loadApiData("lucid.tensor");
    const nn = loadApiData("lucid.nn");
    const engine = loadApiData("lucid._C.engine");
    const tensorMethods = tensor.kind === "class-module" ? tensor.methods.length : 0;
    const nnClasses = isApiModule(nn) ? nn.members.length : 0;
    const engineMembers = isApiModule(engine) ? engine.members.length : 0;
    let modelFamilies = 0;
    for (const cat of ["vision", "text", "generative"]) {
      try {
        const d = loadApiData(`lucid.models.${cat}`);
        if (isApiModule(d) && d.family_groups) {
          modelFamilies += d.family_groups.length;
        }
      } catch {
        // category file missing — skip
      }
    }
    // ``lucid.optim`` lists Optimizer + LRScheduler subclasses together;
    // partition by the source-file subcategory the build attaches to each
    // entry.  Classes under ``lr_scheduler`` belong to the scheduler bucket;
    // anything else (``optimizer``, ``adam``, ``sgd``, …) is an optimizer.
    let optimizers = 0;
    let lrSchedulers = 0;
    try {
      const optim = loadApiData("lucid.optim");
      if (isApiModule(optim)) {
        for (const m of optim.members) {
          if (m.kind !== "class") continue;
          const sub = (m.subcategory || "").toLowerCase();
          if (sub.includes("lr_scheduler") || sub.includes("scheduler")) {
            lrSchedulers += 1;
          } else {
            optimizers += 1;
          }
        }
      }
    } catch {
      // optim file missing — leave at 0
    }
    return { tensorMethods, nnClasses, engineMembers, modelFamilies, optimizers, lrSchedulers };
  } catch {
    return fallback;
  }
}

// Cheap landing-card data for featured families — pull canonical name +
// task tags from each family's own JSON so the hero strip on the landing
// page can't drift from the actual model zoo entries.
function loadFamilyCard(slug: string): {
  slug: string;
  name: string;
  paperCitation: string | null;
} | null {
  try {
    const d = loadApiData(slug);
    if (!isApiModule(d)) return null;
    return {
      slug,
      name: d.canonical_name ?? d.name,
      paperCitation: d.citation ?? null,
    };
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Stats / install / code subcomponents
// ---------------------------------------------------------------------------

function InstallSnippet() {
  return (
    <div className="relative overflow-hidden rounded-xl border border-lucid-border bg-lucid-surface px-5 py-3.5">
      <div className="flex items-center gap-3">
        <span className="select-none text-lucid-text-disabled text-sm">$</span>
        <code className="font-mono text-sm text-lucid-text-high">{INSTALL_CMD}</code>
      </div>
      <div className="absolute right-3 top-1/2 -translate-y-1/2">
        <Badge variant="secondary" className="text-[10px]">Python 3.14+</Badge>
      </div>
    </div>
  );
}

function StatsStrip({ stats }: { stats: ReturnType<typeof computeStats> }) {
  // The strip lives directly below the hero — no animation here since the
  // hero already absorbed the user's attention on first paint.
  const cells = [
    { label: "Tensor methods",     value: stats.tensorMethods,  hint: "lucid.Tensor" },
    { label: "nn classes & fns",   value: stats.nnClasses,      hint: "lucid.nn"     },
    { label: "C++ engine members", value: stats.engineMembers,  hint: "lucid._C.engine" },
    { label: "model families",     value: stats.modelFamilies,  hint: "lucid.models" },
  ];
  return (
    <section className="border-y border-lucid-border bg-lucid-surface/40 mb-24">
      <div className="mx-auto max-w-screen-2xl px-4 sm:px-6 py-8">
        <dl className="grid grid-cols-2 md:grid-cols-4 gap-y-6 gap-x-4">
          {cells.map(({ label, value, hint }) => (
            <div key={label} className="flex flex-col items-center md:items-start text-center md:text-left">
              <dt className="text-[11px] font-semibold tracking-widest text-lucid-text-disabled uppercase mb-1.5">
                {label}
              </dt>
              <dd className="flex items-baseline gap-2">
                <span className="font-mono text-3xl sm:text-4xl font-bold text-lucid-text-high">
                  {value.toLocaleString()}
                </span>
                <code className="hidden sm:inline text-[10px] text-lucid-text-low font-mono">
                  {hint}
                </code>
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}

async function CodeSnippet() {
  const html = await highlight(HERO_CODE, "python");
  return (
    <section className="mx-auto max-w-screen-2xl px-4 sm:px-6 mb-24">
      <FadeIn inView>
        <div className="mb-8 flex flex-col items-center text-center">
          <Badge variant="secondary" className="mb-3 text-[10px] uppercase tracking-widest">
            Quick taste
          </Badge>
          <h2 className="text-2xl sm:text-3xl font-bold text-lucid-text-high mb-2">
            From import to first step in 13 lines
          </h2>
          <p className="text-sm text-lucid-text-low max-w-xl">
            A drop-in PyTorch-shaped API.  Device-aware optimisers, AMP-ready, no
            CUDA path to keep alive.
          </p>
        </div>
      </FadeIn>
      <FadeIn inView delay={0.05}>
        <div
          className={cn(
            "mx-auto max-w-3xl rounded-2xl border border-lucid-border bg-lucid-surface overflow-hidden",
            "[&_pre]:px-6 [&_pre]:py-5 [&_pre]:overflow-x-auto [&_pre]:!bg-transparent",
            "[&_code]:font-mono [&_code]:text-[13px] [&_code]:leading-relaxed",
          )}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </FadeIn>
    </section>
  );
}

function ModelZooShowcase({ familyCount }: { familyCount: number }) {
  const cards = FEATURED_FAMILIES.map((f) => ({
    ...f,
    data: loadFamilyCard(f.slug),
  })).filter((c) => c.data !== null) as Array<{
    slug: string;
    category: string;
    blurb: string;
    data: { slug: string; name: string; paperCitation: string | null };
  }>;
  if (cards.length === 0) return null;
  return (
    <section className="mx-auto max-w-screen-2xl px-4 sm:px-6 py-24 border-t border-lucid-border">
      <FadeIn inView>
        <div className="mb-12 flex flex-col items-center text-center">
          <Badge variant="secondary" className="mb-3 text-[10px] uppercase tracking-widest">
            Model Zoo
          </Badge>
          <h2 className="text-3xl sm:text-4xl font-bold text-lucid-text-high mb-3">
            {familyCount} paper-cited families
          </h2>
          <p className="text-lucid-text-low max-w-xl">
            Every architecture ships with a frozen config, the direct model, task wrappers, and a pretrained factory — all stamped against the original paper.
          </p>
        </div>
      </FadeIn>
      <FadeInStagger inView staggerDelay={0.05} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {cards.map(({ slug, category, blurb, data }) => (
          <Link
            key={slug}
            href={`/api/${slug}`}
            className="group block rounded-xl border border-lucid-border bg-lucid-surface/40 hover:bg-lucid-surface hover:border-lucid-primary/40 transition-colors"
          >
            <div className="px-5 pt-4 pb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-semibold tracking-widest text-lucid-text-disabled uppercase">
                  {category}
                </span>
                <ArrowRight className="h-3.5 w-3.5 text-lucid-text-disabled transition-colors group-hover:text-lucid-primary" />
              </div>
              <h3 className="font-mono text-lg font-bold text-lucid-primary mb-1.5">
                {data.name}
              </h3>
              <p className="text-xs text-lucid-text-low leading-relaxed">
                {blurb}
              </p>
            </div>
          </Link>
        ))}
      </FadeInStagger>
      <FadeIn inView delay={0.15}>
        <div className="mt-8 text-center">
          <Button variant="secondary" size="sm" asChild>
            <Link href="/api/lucid.models">Browse all {familyCount} families</Link>
          </Button>
        </div>
      </FadeIn>
    </section>
  );
}

function QuickAccessGrid({ stats }: { stats: ReturnType<typeof computeStats> }) {
  const modules = buildQuickModules(stats);
  return (
    <section className="mx-auto max-w-screen-2xl px-4 sm:px-6 py-24 border-t border-lucid-border">
      <FadeIn inView>
        <div className="mb-12 flex flex-col items-center text-center">
          <Badge variant="secondary" className="mb-3 text-[10px] uppercase tracking-widest">
            Quick access
          </Badge>
          <h2 className="text-3xl sm:text-4xl font-bold text-lucid-text-high mb-3">
            Jump straight in
          </h2>
        </div>
      </FadeIn>
      <FadeInStagger inView staggerDelay={0.04} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {modules.map(({ slug, label, icon: Icon, blurb }) => (
          <Link
            key={slug}
            href={`/api/${slug}`}
            className="group flex items-start gap-3 rounded-xl border border-lucid-border bg-lucid-surface/40 px-5 py-4 hover:bg-lucid-surface hover:border-lucid-primary/40 transition-colors"
          >
            <span
              className="shrink-0 inline-flex h-9 w-9 items-center justify-center rounded-lg border border-lucid-primary/35 bg-lucid-primary/10 text-lucid-primary"
              aria-hidden
            >
              <Icon className="h-4 w-4" />
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline gap-2 mb-1">
                <code className="font-mono text-sm font-semibold text-lucid-primary">
                  {slug}
                </code>
                <span className="text-[10px] text-lucid-text-disabled">{label}</span>
              </div>
              <p className="text-xs text-lucid-text-low leading-relaxed">
                {blurb}
              </p>
            </div>
          </Link>
        ))}
      </FadeInStagger>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default async function Home() {
  const stats = computeStats();
  return (
    <div className="flex min-h-dvh flex-col">
      <Header />

      <main id="main-content" tabIndex={-1} className="flex-1 focus:outline-none">
        {/* Hero */}
        <section className="relative flex min-h-[calc(100dvh-3.5rem)] items-center justify-center overflow-hidden pt-14">
          {/* Background glow */}
          <div
            className="pointer-events-none absolute inset-0 -z-10"
            aria-hidden="true"
          >
            <div className="absolute left-1/2 top-1/3 -translate-x-1/2 -translate-y-1/2 h-[600px] w-[900px] rounded-full bg-lucid-primary/[0.06] blur-[120px]" />
            <div className="absolute left-1/3 top-2/3 -translate-x-1/2 -translate-y-1/2 h-[400px] w-[600px] rounded-full bg-lucid-blue/[0.05] blur-[100px]" />
          </div>

          <div className="mx-auto max-w-4xl px-4 sm:px-6 text-center">
            <FadeIn delay={0}>
              <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-lucid-primary/25 bg-lucid-primary/10 px-4 py-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-lucid-primary animate-pulse" />
                <span className="text-xs font-medium text-lucid-primary tracking-wide">
                  {buildMeta.lucid_version ? `Lucid ${buildMeta.lucid_version} — Now available` : "Lucid 3.0 — Now available"}
                </span>
              </div>
            </FadeIn>

            <FadeIn delay={0.05}>
              <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight leading-[1.08] mb-6">
                <span className="gradient-hero-text">Production ML</span>
                <br />
                <span className="text-lucid-text-high">for Apple Silicon</span>
              </h1>
            </FadeIn>

            <FadeIn delay={0.1}>
              <p className="mx-auto max-w-2xl text-lg sm:text-xl text-lucid-text-mid leading-relaxed mb-10">
                MLX + Accelerate native backend. PyTorch-compatible API.
                Zero third-party dependencies in the compute path.
              </p>
            </FadeIn>

            <FadeIn delay={0.15}>
              <div className="flex flex-col items-center gap-4">
                <div className="flex flex-wrap items-center justify-center gap-3">
                  <Button size="lg" asChild>
                    <Link href="/docs/quickstart">Get Started</Link>
                  </Button>
                  <Button variant="secondary" size="lg" asChild>
                    <Link href="/api">API Reference</Link>
                  </Button>
                </div>
                <div className="w-full max-w-sm">
                  <InstallSnippet />
                </div>
              </div>
            </FadeIn>
          </div>
        </section>

        {/* Stats — sits at the seam between hero and content so the visitor
            gets a concrete sense of scale before the prose starts. */}
        <StatsStrip stats={stats} />

        {/* Quick taste */}
        <CodeSnippet />

        {/* Features */}
        <section className="mx-auto max-w-screen-2xl px-4 sm:px-6 py-24 border-t border-lucid-border">
          <FadeIn inView>
            <div className="mb-12 text-center">
              <Badge variant="secondary" className="mb-3 text-[10px] uppercase tracking-widest">
                Why Lucid
              </Badge>
              <h2 className="text-3xl sm:text-4xl font-bold text-lucid-text-high mb-3">
                Built for the hardware
              </h2>
              <p className="text-lucid-text-low max-w-xl mx-auto">
                Every layer of the stack is optimized for Apple Silicon's unified memory architecture.
              </p>
            </div>
          </FadeIn>

          <FadeInStagger
            inView
            staggerDelay={0.08}
            className="grid grid-cols-1 sm:grid-cols-2 gap-4"
          >
            {FEATURES.map(({ icon: Icon, title, description }) => (
              <div
                key={title}
                className="group relative overflow-hidden rounded-2xl border border-lucid-border bg-lucid-surface p-6 transition-all duration-200 hover:border-lucid-primary/40 hover:bg-lucid-elevated"
              >
                <div className="mb-4 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-lucid-primary/35 bg-lucid-primary/10 text-lucid-primary">
                  <Icon className="h-5 w-5" aria-hidden />
                </div>
                <h3 className="mb-2 text-base font-semibold text-lucid-text-high">
                  {title}
                </h3>
                <p className="text-sm text-lucid-text-low leading-relaxed">
                  {description}
                </p>
                <div className="pointer-events-none absolute inset-0 rounded-2xl bg-gradient-to-br from-lucid-primary/[0.03] to-transparent opacity-0 transition-opacity duration-200 group-hover:opacity-100" />
              </div>
            ))}
          </FadeInStagger>
        </section>

        {/* Model zoo featured families */}
        <ModelZooShowcase familyCount={stats.modelFamilies} />

        {/* Module quick access */}
        <QuickAccessGrid stats={stats} />
      </main>

      <Footer />
    </div>
  );
}

export const dynamic = "force-static";
void getAllModuleSlugs;
