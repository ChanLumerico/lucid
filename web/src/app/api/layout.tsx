import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Sidebar, type SidebarItem } from "@/components/layout/Sidebar";
import { loadApiData } from "@/lib/api-loader";
import { isApiModule, isApiClassModule, isApiClass } from "@/lib/types";

// ---------------------------------------------------------------------------
// Sidebar group manifest — defines category ordering + which slugs belong
// ---------------------------------------------------------------------------

const SIDEBAR_GROUPS: Array<{ title: string; slugs: string[] }> = [
  {
    title: "Core",
    slugs: ["lucid", "lucid.tensor"],
  },
  {
    title: "Neural Networks",
    slugs: [
      "lucid.nn",
      "lucid.nn.functional",
      "lucid.nn.init",
      "lucid.nn.utils",
    ],
  },
  {
    title: "Optimization",
    slugs: ["lucid.optim"],
  },
  {
    title: "Differentiation",
    slugs: ["lucid.autograd", "lucid.func"],
  },
  {
    title: "Math",
    slugs: ["lucid.linalg", "lucid.fft", "lucid.special", "lucid.signal"],
  },
  {
    title: "Probabilistic",
    slugs: ["lucid.distributions"],
  },
  {
    title: "Utilities",
    slugs: [
      "lucid.utils.data",
      "lucid.einops",
      "lucid.amp",
      "lucid.profiler",
      "lucid.serialization",
    ],
  },
];

// ---------------------------------------------------------------------------
// Build sidebar tree — server-side, reads public/api-data/*.json
// ---------------------------------------------------------------------------

function buildModuleItem(slug: string): SidebarItem {
  let memberItems: SidebarItem[] | undefined;
  let badge: string | undefined;

  try {
    const data = loadApiData(slug);

    if (isApiModule(data)) {
      // Show top-level classes and functions.
      // Exclude methods within classes — the user navigates into them from
      // the class detail page, not from the sidebar.
      const members = data.members.filter(
        (m) => isApiClass(m) || m.kind === "function",
      );

      if (members.length > 0) {
        badge = `${members.length}`;
        memberItems = members.map((m) => ({
          title: m.name,
          href: `/api/${slug}/${m.name}`,
          // Pass kind as badge so Sidebar can render a small indicator
          badge: isApiClass(m) ? "C" : "f",
        }));
      }
    } else if (isApiClassModule(data)) {
      // lucid.tensor — the module IS the class; its methods are not listed
      // here per the user's request ("클래스 내 메소드는 제외").
      badge = `${data.methods.length}`;
      memberItems = undefined;
    }
  } catch {
    // JSON not yet built — gracefully degrade to a plain link
  }

  return {
    title: slug,
    href: `/api/${slug}`,
    badge,
    items: memberItems,
  };
}

function buildApiSidebar(): SidebarItem[] {
  return SIDEBAR_GROUPS.map(({ title, slugs }) => ({
    title,
    items: slugs.map(buildModuleItem),
  }));
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

export default function ApiLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const sidebar = buildApiSidebar();

  return (
    <div className="flex min-h-dvh flex-col">
      <Header />
      <div className="mx-auto flex w-full max-w-screen-xl flex-1 gap-0 px-4 sm:px-6 pt-14">
        <Sidebar
          items={sidebar}
          className="sticky top-14 h-[calc(100dvh-3.5rem)]"
        />
        <main className="flex-1 min-w-0 pt-10 pb-12 lg:px-8">{children}</main>
      </div>
      <Footer />
    </div>
  );
}
