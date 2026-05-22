"use client";

import { MotionConfig } from "framer-motion";

/** Respect ``prefers-reduced-motion: reduce`` at the framer-motion layer.
 *
 *  Framer-Motion's default is "never reduce" — animations always play.
 *  Setting ``reducedMotion="user"`` tells framer to honour the OS-level
 *  setting (System Settings → Accessibility → Display → Reduce Motion on
 *  macOS, the corresponding toggle elsewhere) by snapping springs to
 *  their end state and skipping non-essential opacity / scale fades.
 *
 *  Mounted at the root layout so EVERY framer-motion descendant inherits
 *  the policy — Sidebar collapse, FadeIn/FadeInStagger, MobileMenu, etc.
 */
export function ReducedMotionProvider({ children }: { children: React.ReactNode }) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}
