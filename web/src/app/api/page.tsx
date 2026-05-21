import type { Metadata } from "next";
import Link from "next/link";
import type { LucideIcon } from "lucide-react";
import {
  Activity,
  Boxes,
  BrainCircuit,
  Cpu,
  Database,
  Dices,
  FunctionSquare,
  Gauge,
  GitBranch,
  Network,
  Save,
  Shuffle,
  Sigma,
  Sparkles,
  Star,
  Timer,
  TrendingDown,
  Waves,
  Wrench,
  Workflow,
} from "lucide-react";
import { FadeIn, FadeInStagger } from "@/components/motion/FadeIn";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export const metadata: Metadata = {
  title: "API Reference",
  description: "Complete API reference for the Lucid ML framework.",
};

interface ModuleCard {
  name: string;
  slug: string;          // for href; "/api/<slug>"
  description: string;
  icon: LucideIcon;
  badge?: string;
}

interface ModuleGroup {
  category: string;
  modules: ModuleCard[];
}

const MODULE_GROUPS: ModuleGroup[] = [
  {
    category: "Core",
    modules: [
      { name: "lucid.Tensor", slug: "lucid.tensor", description: "Primary tensor class — 309 methods and properties.", icon: Boxes, badge: "309 methods" },
    ],
  },
  {
    category: "Neural Networks",
    modules: [
      { name: "lucid.nn",            slug: "lucid.nn",            description: "Module, Parameter, and 151 layer classes.",        icon: Network,        badge: "151 modules" },
      { name: "lucid.nn.functional", slug: "lucid.nn.functional", description: "Stateless functional operations (70 functions).",  icon: FunctionSquare },
      { name: "lucid.nn.init",       slug: "lucid.nn.init",       description: "Weight initialization strategies (13 functions).", icon: Sparkles       },
      { name: "lucid.nn.utils",      slug: "lucid.nn.utils",      description: "Gradient clipping, weight norm, RNN packing.",     icon: Wrench         },
    ],
  },
  {
    category: "Model Zoo",
    modules: [
      { name: "lucid.models", slug: "lucid.models", description: "49 paper-cited vision / text / generative families with pretrained factories.", icon: BrainCircuit, badge: "49 families" },
    ],
  },
  {
    category: "Optimization",
    modules: [
      { name: "lucid.optim", slug: "lucid.optim", description: "13 optimizers + 16 learning rate schedulers.", icon: TrendingDown },
    ],
  },
  {
    category: "Differentiation",
    modules: [
      { name: "lucid.autograd", slug: "lucid.autograd", description: "Function, grad, gradcheck, functional transforms.", icon: GitBranch },
      { name: "lucid.func",     slug: "lucid.func",     description: "vmap, grad, vjp, jvp, jacrev, jacfwd, hessian.",    icon: Workflow  },
    ],
  },
  {
    category: "Math",
    modules: [
      { name: "lucid.linalg",  slug: "lucid.linalg",  description: "31 decomposition, norm, and solve operations.",     icon: Sigma    },
      { name: "lucid.fft",     slug: "lucid.fft",     description: "22 DFT / Hermitian / shift / frequency functions.", icon: Waves    },
      { name: "lucid.special", slug: "lucid.special", description: "12 special-math functions (erf, sinc, gamma, …).",  icon: Star     },
      { name: "lucid.signal",  slug: "lucid.signal",  description: "12 spectral window functions.",                     icon: Activity },
    ],
  },
  {
    category: "Probabilistic",
    modules: [
      { name: "lucid.distributions", slug: "lucid.distributions", description: "17 distributions + base class + KL registry.", icon: Dices },
    ],
  },
  {
    category: "Utilities",
    modules: [
      { name: "lucid.utils.data",    slug: "lucid.utils.data",    description: "Dataset, Sampler, DataLoader.",     icon: Database },
      { name: "lucid.einops",        slug: "lucid.einops",        description: "Einops-style tensor manipulation.", icon: Shuffle  },
      { name: "lucid.amp",           slug: "lucid.amp",           description: "Automatic mixed precision.",        icon: Gauge    },
      { name: "lucid.profiler",      slug: "lucid.profiler",      description: "Performance profiling utilities.",  icon: Timer    },
      { name: "lucid.serialization", slug: "lucid.serialization", description: "State dict save / load.",           icon: Save     },
    ],
  },
  {
    category: "Engine",
    modules: [
      { name: "lucid._C.engine", slug: "lucid._C.engine", description: "C++ compute core — storage, ops, autograd graph, backend dispatch.", icon: Cpu, badge: "813 members" },
    ],
  },
];

const ACCENT = "var(--color-lucid-primary)";

function ModuleCardLink({ mod }: { mod: ModuleCard }) {
  const Icon = mod.icon;
  const bg = `color-mix(in srgb, ${ACCENT} 14%, transparent)`;
  const ring = `color-mix(in srgb, ${ACCENT} 38%, transparent)`;
  return (
    <Link
      href={`/api/${mod.slug}`}
      className={cn(
        "group block rounded-xl border border-lucid-border bg-lucid-surface/40",
        "transition-colors hover:bg-lucid-surface hover:border-lucid-primary/40",
      )}
    >
      <div className="px-5 pt-4 pb-4 flex items-start gap-3">
        <span
          className="shrink-0 inline-flex h-9 w-9 items-center justify-center rounded-lg border"
          style={{ backgroundColor: bg, borderColor: ring, color: ACCENT }}
          aria-hidden
        >
          <Icon className="h-4 w-4" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-2">
            <code
              className="font-mono text-sm font-semibold"
              style={{ color: ACCENT }}
            >
              {mod.name}
            </code>
            {mod.badge && (
              <Badge variant="secondary" className="font-mono text-[10px]">
                {mod.badge}
              </Badge>
            )}
          </div>
          <p className="mt-1 text-xs text-lucid-text-low leading-relaxed">
            {mod.description}
          </p>
        </div>
      </div>
    </Link>
  );
}

export default function ApiPage() {
  return (
    <FadeIn>
      <div className="max-w-4xl">
        <header className="mb-12">
          <div className="flex flex-wrap items-center gap-3 mb-3">
            <span className="text-xs font-semibold tracking-widest uppercase text-lucid-text-disabled">
              reference
            </span>
            <h1 className="font-mono text-3xl font-bold text-lucid-text-high">
              API Reference
            </h1>
          </div>
          <p className="max-w-3xl text-base text-lucid-text-mid leading-relaxed">
            Complete reference documentation, auto-generated from Lucid's
            Python sources via{" "}
            <a
              href="https://github.com/mkdocstrings/griffe"
              className="text-lucid-primary hover:text-lucid-primary-light transition-colors"
            >
              Griffe
            </a>{" "}
            and the C++ engine via libclang.  Browse by module below, or jump
            straight into a surface from the sidebar.
          </p>
        </header>

        <div className="space-y-10">
          {MODULE_GROUPS.map(({ category, modules }) => (
            <section key={category}>
              <FadeIn>
                <h2 className="text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase mb-3">
                  {category}
                </h2>
              </FadeIn>
              <FadeInStagger staggerDelay={0.04} className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {modules.map((mod) => (
                  <ModuleCardLink key={mod.slug} mod={mod} />
                ))}
              </FadeInStagger>
            </section>
          ))}
        </div>
      </div>
    </FadeIn>
  );
}
