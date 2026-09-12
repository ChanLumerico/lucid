/*
 * Host bridge for archify viewers embedded in the Lucid docs site.
 *
 * scripts/sync-archify.mjs links this into every viewer it copies to
 * public/archify/.  Opened on its own, a viewer is left untouched.  Inside
 * the site's same-origin iframe the bridge
 *
 *   1. marks <html data-lucid-host> so host.css can drop the viewer's own
 *      title and fact cards — the page renders both natively;
 *   2. repaints the viewer's default "classic" palette from the site's design
 *      tokens, read live from the parent document, so no colour value is
 *      duplicated here and a token change in globals.css reaches the viewer;
 *      the other presets keep their own palettes;
 *   3. keeps light/dark in lockstep both ways: the site's toggle drives the
 *      viewer, and the viewer's own toggle is posted back to the page
 *      (DiagramViewer.tsx), which updates the site theme.
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
    const subtle = token("--color-lucid-border-subtle", border);
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
      ["--grid", subtle],
      ["--panel", surface],
      ["--panel-border", border],
      ["--text", high],
      ["--text-muted", mid],
      ["--text-dim", disabled],
      ["--text-faint", low],
      ["--mask", surface],
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

  // Site → viewer.  archify exposes only `toggle`, so flip when they differ.
  new MutationObserver(() => {
    const theme = hostTheme();
    if (!theme) return;
    paint();
    if (root.getAttribute("data-theme") === theme) return;
    const api = window.Archify && window.Archify.theme;
    if (api && typeof api.toggle === "function") {
      api.toggle();
    } else {
      root.setAttribute("data-theme", theme);
    }
  }).observe(host, { attributes: true, attributeFilter: ["data-theme"] });

  // Viewer → site.  After a site-driven flip the two already agree, so this
  // only fires for the viewer's own toggle.
  new MutationObserver(() => {
    const theme = root.getAttribute("data-theme");
    if ((theme === "light" || theme === "dark") && theme !== hostTheme()) {
      window.parent.postMessage({ type: "lucid:archify-theme", theme }, window.location.origin);
    }
  }).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
})();
