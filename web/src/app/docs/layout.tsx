import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Sidebar, type SidebarItem } from "@/components/layout/Sidebar";

const DOCS_SIDEBAR: SidebarItem[] = [
  {
    title: "Getting Started",
    items: [
      { title: "Installation", href: "/docs/installation" },
      { title: "Quickstart", href: "/docs/quickstart" },
    ],
  },
  {
    title: "Guides",
    items: [
      { title: "Autograd", href: "/docs/autograd" },
      { title: "Metal Device", href: "/docs/metal-device" },
    ],
  },
  {
    title: "Concepts",
    items: [
      { title: "Tensor", href: "/docs/tensor" },
      { title: "Modules", href: "/docs/modules" },
      { title: "Optimizers", href: "/docs/optimizers" },
    ],
  },
];

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-dvh flex-col">
      <Header />
      <div className="mx-auto flex w-full max-w-screen-xl flex-1 gap-0 px-4 sm:px-6 pt-14">
        <Sidebar items={DOCS_SIDEBAR} className="sticky top-14 h-[calc(100dvh-3.5rem)]" />
        <main className="flex-1 min-w-0 pt-10 pb-12 lg:px-8">{children}</main>
      </div>
      <Footer />
    </div>
  );
}
