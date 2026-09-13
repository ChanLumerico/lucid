/*
 * Host bridge for archify viewers embedded in the Lucid docs site.
 *
 * scripts/sync-archify.mjs links this into every viewer it copies to
 * public/archify/.  Opened on its own, a viewer is left untouched.  Inside
 * the site's same-origin iframe — loaded with ?present=1, archify's stage
 * layout, where the diagram fills the frame — the bridge
 *
 *   1. marks <html data-lucid-host> so host.css can drop what the site owns
 *      (the title and fact cards the page renders, and the whole toolbar:
 *      theme, style, stage and export) and lay the diagram straight onto the
 *      page, with no canvas box of its own;
 *   2. repaints the viewer's default "classic" palette from the site's design
 *      tokens, read live from the parent document, so no colour value is
 *      duplicated here and a token change in globals.css reaches the viewer;
 *   3. holds the viewer to the site's light/dark theme, and to the stage;
 *   4. swallows the T / S / F / E shortcuts of the removed controls and drops
 *      them from the guide.
 */
(() => {
  let host;
  try {
    if (window.parent === window) return;
    host = window.parent.document.documentElement;
  } catch {
    return; // cross-origin parent — stay a plain standalone viewer
  }

  const root = document.documentElement;
  root.setAttribute("data-lucid-host", "true");

  const hostTheme = () => {
    const theme = host.getAttribute("data-theme");
    return theme === "light" || theme === "dark" ? theme : null;
  };

  // archify resolves its initial theme from the URL, then its own storage
  // key, then the OS preference.  Seed the storage key with the site's theme
  // so every step agrees before first paint.
  const initial = hostTheme();
  if (initial) {
    root.setAttribute("data-theme", initial);
    try {
      localStorage.setItem("archify-theme", initial);
    } catch {
      // storage blocked — the attribute alone still themes this page load
    }
  }

  const palette = document.createElement("style");
  palette.id = "lucid-host-palette";
  document.head.appendChild(palette);

  const paint = () => {
    const styles = window.parent.getComputedStyle(host);
    const token = (name, fallback) => styles.getPropertyValue(name).trim() || fallback;
    const bg = token("--color-lucid-bg", "");
    if (!bg) return; // tokens unreadable — keep archify's own palette

    const surface = token("--color-lucid-surface", bg);
    const elevated = token("--color-lucid-elevated", surface);
    const border = token("--color-lucid-border", elevated);
    const high = token("--color-lucid-text-high", "");
    const mid = token("--color-lucid-text-mid", high);
    const low = token("--color-lucid-text-low", mid);
    const disabled = token("--color-lucid-text-disabled", low);
    const primary = token("--color-lucid-primary", mid);
    const warning = token("--color-lucid-warning", primary);

    const kinds = {
      frontend: token("--color-lucid-blue", primary),
      backend: token("--color-lucid-success", primary),
      database: primary,
      cloud: warning,
      security: token("--color-lucid-error", primary),
      messagebus: token("--color-api-cpp-operator", warning),
      external: low,
    };
    const tint = `${hostTheme() === "light" ? 10 : 16}%`;

    const decls = [
      ["--bg", bg],
      // The diagram is drawn on the page itself: no grid behind it, and the
      // masks under edge labels match the page.  --panel stays the surface
      // colour for the overlays (finder, lens, guide) that still use it.
      ["--grid", "transparent"],
      ["--mask", bg],
      ["--panel", surface],
      ["--panel-border", border],
      ["--text", high],
      ["--text-muted", mid],
      ["--text-dim", disabled],
      ["--text-faint", low],
      ["--lane-fill", `color-mix(in srgb, ${elevated} 45%, transparent)`],
      ["--lane-stroke", border],
      ["--arrow", low],
      ["--arrow-emphasis", primary],
      ["--toolbar-bg", `color-mix(in srgb, ${surface} 92%, transparent)`],
      ["--toolbar-border", border],
      ["--toolbar-text", mid],
      ["--toolbar-hover", elevated],
      ["--toolbar-menu-bg", surface],
    ];
    for (const [kind, colour] of Object.entries(kinds)) {
      decls.push([`--${kind}-fill`, `color-mix(in srgb, ${colour} ${tint}, ${surface})`]);
      decls.push([`--${kind}-stroke`, colour]);
    }

    // (0,2,1) beats archify's classic `[data-theme]` blocks (0,1,0) and never
    // matches the `[data-preset="…"][data-theme]` blocks of other presets.
    palette.textContent =
      'html[data-lucid-host]:not([data-preset]),html[data-lucid-host][data-preset="classic"]{' +
      decls.map(([name, value]) => `${name}:${value};`).join("") +
      "}";
  };
  paint();

  // The site owns the theme.  archify exposes only `toggle`, so flip the
  // viewer whenever the two differ — after the site toggle, and after a
  // change on the viewer's side: archify follows OS theme changes while its
  // storage key is unset, which it is when storage is blocked.
  const follow = () => {
    const theme = hostTheme();
    if (!theme || root.getAttribute("data-theme") === theme) return;
    const api = window.Archify && window.Archify.theme;
    if (api && typeof api.toggle === "function") {
      api.toggle();
    } else {
      root.setAttribute("data-theme", theme);
    }
  };
  new MutationObserver(() => {
    paint();
    follow();
  }).observe(host, { attributes: true, attributeFilter: ["data-theme"] });
  new MutationObserver(follow).observe(root, { attributes: true, attributeFilter: ["data-theme"] });

  // The stage is the only layout here, but archify's Esc also leaves it —
  // step straight back in.
  new MutationObserver(() => {
    if (root.getAttribute("data-present") === "true") return;
    const api = window.Archify && window.Archify.presentation;
    if (api && typeof api.enter === "function") api.enter();
  }).observe(root, { attributes: true, attributeFilter: ["data-present"] });

  // Shortcuts of the toolbar host.css removes (theme, style, stage, export).
  // archify handles them on document; a capture-phase listener on window
  // runs before it.  Never while the reader is typing.
  const REMOVED_KEYS = new Set(["t", "s", "f", "e"]);
  window.addEventListener(
    "keydown",
    (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || typeof e.key !== "string") return;
      if (!REMOVED_KEYS.has(e.key.toLowerCase())) return;
      const el = e.target;
      if (
        el instanceof HTMLElement &&
        (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))
      ) {
        return;
      }
      e.stopImmediatePropagation();
    },
    true,
  );

  // The guide's shortcut list would still advertise the removed keys.
  document.addEventListener("DOMContentLoaded", () => {
    for (const kbd of document.querySelectorAll(".diagram-guide-shortcuts kbd")) {
      if (REMOVED_KEYS.has(kbd.textContent.trim().toLowerCase())) {
        kbd.parentElement.style.display = "none";
      }
    }
  });
})();
