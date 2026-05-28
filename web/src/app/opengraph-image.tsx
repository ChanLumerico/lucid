import { ImageResponse } from "next/og";

/** Root-level Open-Graph card.  Picked up automatically by Next for the
 *  ``/`` route and inherited by any page that doesn't define its own
 *  per-route ``opengraph-image``. */
//
// ``output: export`` (in next.config.ts) statically prerenders every
// route at build time — the previous ``runtime = "edge"`` setting is
// incompatible with that mode and ``dynamic`` defaults to ``"auto"``
// which Next.js then flags as an error.  Drop the edge runtime
// (defaulting to nodejs is fine here — the image is generated once at
// build time, never at request time) and pin ``dynamic`` to
// ``"force-static"`` so the static export collector knows this route
// is build-time-only.
export const dynamic = "force-static";
export const alt = "Lucid — Production ML for Apple Silicon";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OG() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-start",
          justifyContent: "center",
          padding: "72px 96px",
          background:
            "linear-gradient(135deg, #0d0c15 0%, #1e1c30 50%, #14121f 100%)",
          color: "#f0eeff",
          fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
        }}
      >
        {/* Background glow accent — subtle violet halo top-right. */}
        <div
          style={{
            position: "absolute",
            right: -120,
            top: -120,
            width: 480,
            height: 480,
            borderRadius: 9999,
            background: "rgba(149, 128, 255, 0.18)",
            filter: "blur(60px)",
          }}
        />
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 16,
            marginBottom: 28,
          }}
        >
          <span
            style={{
              fontSize: 30,
              fontWeight: 800,
              color: "#f0eeff",
              letterSpacing: -0.5,
            }}
          >
            Lucid
          </span>
          <span
            style={{
              fontSize: 14,
              fontWeight: 700,
              padding: "5px 12px",
              borderRadius: 8,
              background: "rgba(149, 128, 255, 0.12)",
              border: "1px solid rgba(149, 128, 255, 0.35)",
              color: "#9580ff",
              letterSpacing: 1.5,
              textTransform: "uppercase",
            }}
          >
            3.0
          </span>
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            fontSize: 78,
            fontWeight: 800,
            lineHeight: 1.05,
            letterSpacing: -2,
          }}
        >
          <span
            style={{
              background: "linear-gradient(135deg, #b8abff 0%, #a3b8ff 60%, #9580ff 100%)",
              backgroundClip: "text",
              color: "transparent",
            }}
          >
            Production ML
          </span>
          <span>for Apple Silicon</span>
        </div>
        <p
          style={{
            marginTop: 28,
            fontSize: 26,
            color: "#b4afce",
            lineHeight: 1.4,
            maxWidth: 880,
          }}
        >
          MLX + Accelerate native backend. PyTorch-compatible API. Zero
          third-party deps in the compute path.
        </p>
        <div
          style={{
            position: "absolute",
            bottom: 60,
            left: 96,
            display: "flex",
            alignItems: "center",
            gap: 12,
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, monospace",
            fontSize: 18,
            color: "#6e6a8a",
          }}
        >
          <span style={{ color: "#3d3a55" }}>$</span>
          <span style={{ color: "#f0eeff" }}>pip install lucid</span>
        </div>
      </div>
    ),
    size,
  );
}
