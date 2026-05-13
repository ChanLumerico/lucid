import type { Transition } from "framer-motion";

export const springs = {
  smooth: { type: "spring" as const, stiffness: 280, damping: 28, mass: 0.9 },
  snappy: { type: "spring" as const, stiffness: 350, damping: 30, mass: 0.8 },
  gentle: { type: "spring" as const, stiffness: 200, damping: 30, mass: 1.0 },
  bouncy: { type: "spring" as const, stiffness: 300, damping: 20, mass: 0.8 },
  micro:  { type: "spring" as const, stiffness: 400, damping: 35, mass: 0.6 },
} satisfies Record<string, Transition>;

export type SpringKey = keyof typeof springs;
