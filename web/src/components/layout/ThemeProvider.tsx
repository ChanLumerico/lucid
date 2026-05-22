"use client";

import * as React from "react";

type Theme = "light" | "dark";

interface ThemeContextValue {
  theme: Theme;
  setTheme: (t: Theme) => void;
  toggle: () => void;
}

const ThemeContext = React.createContext<ThemeContextValue>({
  theme: "dark",
  setTheme: () => {},
  toggle: () => {},
});

const STORAGE_KEY = "lucid-docs-theme";

/** Read-and-apply the persisted theme BEFORE React hydration so the
 *  user never sees a dark→light or light→dark flash.  Inlined into
 *  ``<head>`` via ``ThemeBoot`` below — runs synchronously on first
 *  paint. */
export function ThemeBoot() {
  const script = `(() => {
    try {
      const stored = localStorage.getItem(${JSON.stringify(STORAGE_KEY)});
      const prefers = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
      const theme = stored === 'light' || stored === 'dark' ? stored : prefers;
      document.documentElement.dataset.theme = theme;
    } catch {}
  })();`;
  return <script dangerouslySetInnerHTML={{ __html: script }} />;
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = React.useState<Theme>("dark");

  // Sync state with the DOM value the bootstrap script wrote.  This
  // never causes a visual flicker because the DOM is already correct;
  // we're just teaching React what's there.
  React.useEffect(() => {
    const current = (document.documentElement.dataset.theme as Theme) ?? "dark";
    setThemeState(current);
  }, []);

  const setTheme = React.useCallback((t: Theme) => {
    setThemeState(t);
    document.documentElement.dataset.theme = t;
    try {
      localStorage.setItem(STORAGE_KEY, t);
    } catch {
      // Storage disabled — accept that the preference won't survive a
      // page reload.  Better than crashing.
    }
  }, []);

  const toggle = React.useCallback(() => {
    setTheme(theme === "dark" ? "light" : "dark");
  }, [theme, setTheme]);

  const value = React.useMemo(() => ({ theme, setTheme, toggle }), [theme, setTheme, toggle]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  return React.useContext(ThemeContext);
}
