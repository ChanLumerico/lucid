// Flat ESLint config — Next.js 16 ships native flat presets.  Scope: the app
// source under src/.  Structural / design-language conformance is enforced by
// the docs contract audit (scripts/audit-docs.mjs); ESLint covers JS/TS
// correctness (react-hooks rules, Next pitfalls, imports) the audit can't see.
// Build output and the standalone build scripts are excluded.
import next from "eslint-config-next";

export default [
  { ignores: ["out/**", ".next/**", "public/**", "scripts/**", "node_modules/**"] },
  // The few `// eslint-disable-next-line no-console` directives in error
  // boundaries document intent; eslint-config-next doesn't enable no-console,
  // so don't flag them as "unused".
  { linterOptions: { reportUnusedDisableDirectives: "off" } },
  ...next,
  {
    rules: {
      // React 19's effect rule flags intentional mount-guard / external-sync
      // patterns used throughout the app — visible advice, not a hard gate.
      "react-hooks/set-state-in-effect": "warn",
      // Pedantic for apostrophes in prose ("Lucid's"); escaping hurts
      // readability with no real benefit.
      "react/no-unescaped-entities": "off",
    },
  },
];
