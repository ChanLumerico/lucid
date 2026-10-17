import type { MetadataRoute } from "next";
import { brandColors } from "@/lib/brand-kit";
import { BASE_PATH, brandAsset } from "@/lib/site";

// ``output: export`` needs every metadata route pinned static.
export const dynamic = "force-static";

/** Web app manifest — the brand kit README's ``site.webmanifest``.  Next
 *  links it with the base path, but leaves its contents as written, so the
 *  URLs inside carry the base path themselves. */
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Lucid",
    short_name: "Lucid",
    description: "Production-grade ML framework for Apple Silicon.",
    start_url: `${BASE_PATH}/`,
    scope: `${BASE_PATH}/`,
    display: "standalone",
    theme_color: brandColors.night,
    background_color: brandColors.night,
    icons: [
      { src: brandAsset("icon-192.png"), sizes: "192x192", type: "image/png" },
      { src: brandAsset("icon-512.png"), sizes: "512x512", type: "image/png" },
      {
        src: brandAsset("icon-maskable-512.png"),
        sizes: "512x512",
        type: "image/png",
        purpose: "maskable",
      },
    ],
  };
}
