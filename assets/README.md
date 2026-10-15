# Lucid — Logo System

All SVGs are **outlined fills** (no live strokes, no font dependency), so they render
identically in browsers, Figma, Illustrator and favicon pipelines.

## Files

| File | Use |
|---|---|
| `svg/lucid-mark.svg` | Primary symbol — Lucid Violet, transparent |
| `svg/lucid-mark-white.svg` / `-black.svg` | Monochrome (on photos, violet, print) |
| `svg/lucid-mark-duotone.svg` | Two-tone variant (illustrations, hero sections) |
| `svg/lucid-icon-dark.svg` | App icon / avatar (same tile as the original icon) |
| `svg/lucid-icon-violet.svg` / `-light.svg` | Inverse and light-theme tiles |
| `svg/lucid-icon-dark-square.svg` | Full-bleed square (iOS / Android maskable) |
| `svg/favicon.svg` | Optically corrected for 16–32 px (heavier stroke, wider gaps) |
| `svg/lucid-wordmark-on-dark.svg` / `-on-light.svg` | Wordmark only |
| `svg/lucid-lockup-horizontal-on-{dark,light}.svg` | Navbar / header |
| `svg/lucid-lockup-stacked-on-{dark,light}.svg` | Hero, splash, README banner |
| `svg/lucid-og.svg` → `png/og-image.png` | Social preview 1200×630 (platforms need PNG) |
| `png/*`, `favicon.ico` | Raster exports for `<head>` and the web manifest |
| `source/lucid-mark.stroke.svg` | Editable stroke master (two open Bézier ellipses) |
| `source/spec.json`, `source/lucid_logo.py` | Geometry spec and the generator (logo-as-code) |

## `<head>` snippet

```html
<link rel="icon" href="/favicon.ico" sizes="any">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<link rel="manifest" href="/site.webmanifest">
<meta name="theme-color" content="#121214">
<meta property="og:image" content="https://<your-domain>/og-image.png">
<meta name="twitter:card" content="summary_large_image">
```

```json
{
  "name": "Lucid", "short_name": "Lucid",
  "icons": [
    { "src": "/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icon-512.png", "sizes": "512x512", "type": "image/png" },
    { "src": "/icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }
  ],
  "theme_color": "#121214", "background_color": "#121214", "display": "standalone"
}
```

Theme-aware navbar logo:

```html
<picture>
  <source srcset="/lucid-lockup-horizontal-on-dark.svg" media="(prefers-color-scheme: dark)">
  <img src="/lucid-lockup-horizontal-on-light.svg" alt="Lucid" height="32">
</picture>
```

## Palette

| Token | Hex | Role |
|---|---|---|
| Lucid Violet | `#9576E4` | Mark (sampled from the original icon) |
| Violet Tint | `#C4B3F4` | Duotone partner |
| Violet Shade | `#6E4FD0` | Hover / pressed |
| Ink | `#282337` | Icon tile |
| Night | `#121214` | Dark background |
| Lavender | `#F1EDFC` | Light tile |
| Text on dark / light | `#F5F4F8` / `#16141F` | Wordmark |

## Geometry

E±(φ) = c + R(±π/4) · [A cos φ, B sin φ]ᵀ, with B/A = 0.44, stroke s/A = 0.185,
over/under clearance g/A = 0.135. E₊ passes over at the top crossing and under at the
bottom one — a Hopf link (linking number 1).

**Clear space:** keep at least 0.5 × mark height free on every side.
**Minimum size:** 16 px (use `favicon.svg` below 48 px), lockup 20 px tall.
**Wordmark:** Wanted Sans Bold, tracking −0.012 em, outlined (tagline: Wanted Sans Medium). Font: SIL OFL 1.1, see `source/fonts/OFL.txt`.

Don't: recolor outside the palette, rotate, stretch, add strokes/shadows, or fill the loops.
