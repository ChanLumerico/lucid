import type { Metadata } from "next";
import Link from "next/link";
import { FadeIn, FadeInStagger } from "@/components/motion/FadeIn";
import { Badge } from "@/components/ui/badge";

export const metadata: Metadata = {
  title: "API Reference",
  description: "Complete API reference for the Lucid ML framework.",
};

const MODULE_GROUPS = [
  {
    category: "Core",
    modules: [
      { name: "lucid.Tensor", description: "Primary tensor class — 309 methods and properties", href: "/api/lucid.tensor", badge: "309 methods" },
    ],
  },
  {
    category: "Neural Networks",
    modules: [
      { name: "lucid.nn", description: "Module, Parameter, and 151 layer classes", href: "/api/lucid.nn", badge: "151 modules" },
      { name: "lucid.nn.functional", description: "Stateless functional operations (70 functions)", href: "/api/lucid.nn.functional" },
      { name: "lucid.nn.init", description: "Weight initialization strategies (13 functions)", href: "/api/lucid.nn.init" },
      { name: "lucid.nn.utils", description: "Gradient clipping, weight norm, RNN packing", href: "/api/lucid.nn.utils" },
    ],
  },
  {
    category: "Optimization",
    modules: [
      { name: "lucid.optim", description: "13 optimizers + 16 learning rate schedulers", href: "/api/lucid.optim" },
    ],
  },
  {
    category: "Differentiation",
    modules: [
      { name: "lucid.autograd", description: "Function, grad, gradcheck, functional transforms", href: "/api/lucid.autograd" },
      { name: "lucid.func", description: "vmap, grad, vjp, jvp, jacrev, jacfwd, hessian", href: "/api/lucid.func" },
    ],
  },
  {
    category: "Math",
    modules: [
      { name: "lucid.linalg", description: "31 decomposition, norm, and solve operations", href: "/api/lucid.linalg" },
      { name: "lucid.fft", description: "22 DFT / Hermitian / shift / frequency functions", href: "/api/lucid.fft" },
      { name: "lucid.special", description: "12 special-math functions (erf, sinc, gamma, …)", href: "/api/lucid.special" },
      { name: "lucid.signal", description: "12 spectral window functions", href: "/api/lucid.signal" },
    ],
  },
  {
    category: "Probabilistic",
    modules: [
      { name: "lucid.distributions", description: "17 distributions + base class + KL registry", href: "/api/lucid.distributions" },
    ],
  },
  {
    category: "Utilities",
    modules: [
      { name: "lucid.utils.data", description: "Dataset, Sampler, DataLoader", href: "/api/lucid.utils.data" },
      { name: "lucid.einops", description: "Einops-style tensor manipulation", href: "/api/lucid.einops" },
      { name: "lucid.amp", description: "Automatic mixed precision", href: "/api/lucid.amp" },
      { name: "lucid.profiler", description: "Performance profiling utilities", href: "/api/lucid.profiler" },
      { name: "lucid.serialization", description: "State dict save / load", href: "/api/lucid.serialization" },
    ],
  },
] as const;

export default function ApiPage() {
  return (
    <FadeIn>
      <div className="max-w-4xl">
        <h1 className="text-3xl font-bold text-lucid-text-high mb-3">
          API Reference
        </h1>
        <p className="text-lucid-text-mid mb-12 text-base leading-relaxed">
          Complete reference documentation, auto-generated from Lucid source code.
        </p>

        <div className="space-y-10">
          {MODULE_GROUPS.map(({ category, modules }) => (
            <section key={category}>
              <FadeIn>
                <h2 className="text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase mb-3">
                  {category}
                </h2>
              </FadeIn>
              <FadeInStagger staggerDelay={0.04} className="space-y-2">
                {modules.map(({ name, description, href, ...rest }) => {
                  const badge = "badge" in rest ? rest.badge : undefined;
                  return (
                    <Link
                      key={href}
                      href={href}
                      className="group flex items-start justify-between rounded-xl border border-lucid-border bg-lucid-surface px-4 py-3.5 transition-all hover:border-lucid-primary/40 hover:bg-lucid-elevated"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2 mb-1">
                          <code className="text-sm font-mono font-medium text-lucid-primary">
                            {name}
                          </code>
                          {badge && (
                            <Badge variant="secondary" className="font-mono text-[10px]">
                              {badge}
                            </Badge>
                          )}
                        </div>
                        <p className="text-xs text-lucid-text-low leading-relaxed">
                          {description}
                        </p>
                      </div>
                    </Link>
                  );
                })}
              </FadeInStagger>
            </section>
          ))}
        </div>
      </div>
    </FadeIn>
  );
}
