"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Menu, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { springs } from "@/components/motion/springs";

const NAV_LINKS = [
  { href: "/docs", label: "Docs" },
  { href: "/api", label: "API Reference" },
  { href: "/changelog", label: "Changelog" },
] as const;

function GitHubIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 16 16"
      className={cn("fill-current", className)}
      aria-hidden="true"
    >
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

function LucidLogo() {
  return (
    <Link
      href="/"
      className="group flex items-center gap-2.5 outline-none focus-visible:ring-2 focus-visible:ring-lucid-primary focus-visible:ring-offset-2 focus-visible:ring-offset-lucid-bg rounded-md"
      aria-label="Lucid home"
    >
      <span className="text-[17px] font-bold tracking-tight text-lucid-text-high transition-colors duration-150 group-hover:text-lucid-primary">
        Lucid
      </span>
      <span
        className={cn(
          "rounded-md border border-lucid-primary/35 bg-lucid-primary/10",
          "px-1.5 py-0.5 text-[9px] font-bold tracking-[0.12em] text-lucid-primary uppercase",
        )}
      >
        3.0
      </span>
    </Link>
  );
}

function DesktopNav({ pathname }: { pathname: string }) {
  return (
    <nav
      className="hidden md:flex items-center gap-0.5"
      aria-label="Primary navigation"
    >
      {NAV_LINKS.map(({ href, label }) => {
        const active = pathname === href || pathname.startsWith(`${href}/`);
        return (
          <Link
            key={href}
            href={href}
            className={cn(
              "relative rounded-md px-3 py-1.5 text-sm font-medium transition-colors duration-150 outline-none",
              "focus-visible:ring-2 focus-visible:ring-lucid-primary focus-visible:ring-offset-1 focus-visible:ring-offset-lucid-bg",
              active
                ? "text-lucid-text-high"
                : "text-lucid-text-mid hover:text-lucid-text-high",
            )}
            aria-current={active ? "page" : undefined}
          >
            {active && (
              <motion.span
                layoutId="header-active-pill"
                className="absolute inset-0 rounded-md bg-lucid-surface border border-lucid-border"
                transition={springs.snappy}
              />
            )}
            <span className="relative">{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}

function MobileMenu({
  open,
  onClose,
  pathname,
}: {
  open: boolean;
  onClose: () => void;
  pathname: string;
}) {
  React.useEffect(() => {
    if (open) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            key="backdrop"
            className="fixed inset-0 z-40 bg-lucid-bg/80 backdrop-blur-sm md:hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
          />
          <motion.nav
            key="menu"
            className="fixed inset-x-0 top-14 z-50 border-b border-lucid-border bg-lucid-surface px-6 py-4 md:hidden"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={springs.snappy}
            aria-label="Mobile navigation"
          >
            <ul className="flex flex-col gap-1">
              {NAV_LINKS.map(({ href, label }) => {
                const active = pathname === href || pathname.startsWith(`${href}/`);
                return (
                  <li key={href}>
                    <Link
                      href={href}
                      onClick={onClose}
                      className={cn(
                        "flex items-center rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                        active
                          ? "bg-lucid-elevated text-lucid-text-high"
                          : "text-lucid-text-mid hover:bg-lucid-elevated hover:text-lucid-text-high",
                      )}
                      aria-current={active ? "page" : undefined}
                    >
                      {label}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </motion.nav>
        </>
      )}
    </AnimatePresence>
  );
}

export function Header() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = React.useState(false);
  const [mobileOpen, setMobileOpen] = React.useState(false);

  React.useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 8);
    handleScroll();
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  React.useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  return (
    <>
      <header
        className={cn(
          "fixed inset-x-0 top-0 z-50 h-14 transition-all duration-300",
          scrolled
            ? "border-b border-lucid-border bg-lucid-bg/95 backdrop-blur-xl backdrop-saturate-150 shadow-sm shadow-black/20"
            : "border-b border-lucid-border/40 bg-lucid-bg/70 backdrop-blur-md",
        )}
      >
        <div className="mx-auto flex h-full max-w-screen-xl items-center justify-between px-4 sm:px-6">
          <div className="flex items-center gap-8">
            <LucidLogo />
            <DesktopNav pathname={pathname} />
          </div>

          <div className="flex items-center gap-1.5">
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Search"
              className="text-lucid-text-low hover:text-lucid-text-high"
            >
              <Search className="h-4 w-4" />
            </Button>

            <a
              href="https://github.com/ChanLumerico/lucid"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Lucid on GitHub"
              className={cn(
                "hidden sm:flex h-7 w-7 items-center justify-center rounded-md",
                "text-lucid-text-low transition-colors hover:text-lucid-text-high",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-lucid-primary",
              )}
            >
              <GitHubIcon className="h-4 w-4" />
            </a>

            <Button
              variant="ghost"
              size="icon-sm"
              className="md:hidden text-lucid-text-low hover:text-lucid-text-high"
              aria-label={mobileOpen ? "Close menu" : "Open menu"}
              aria-expanded={mobileOpen}
              onClick={() => setMobileOpen((v) => !v)}
            >
              <AnimatePresence mode="wait" initial={false}>
                {mobileOpen ? (
                  <motion.span
                    key="close"
                    initial={{ rotate: -90, opacity: 0 }}
                    animate={{ rotate: 0, opacity: 1 }}
                    exit={{ rotate: 90, opacity: 0 }}
                    transition={springs.micro}
                  >
                    <X className="h-4 w-4" />
                  </motion.span>
                ) : (
                  <motion.span
                    key="menu"
                    initial={{ rotate: 90, opacity: 0 }}
                    animate={{ rotate: 0, opacity: 1 }}
                    exit={{ rotate: -90, opacity: 0 }}
                    transition={springs.micro}
                  >
                    <Menu className="h-4 w-4" />
                  </motion.span>
                )}
              </AnimatePresence>
            </Button>
          </div>
        </div>
      </header>

      <MobileMenu
        open={mobileOpen}
        onClose={() => setMobileOpen(false)}
        pathname={pathname}
      />
    </>
  );
}
