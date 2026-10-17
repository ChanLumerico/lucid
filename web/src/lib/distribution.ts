// The PyPI distribution name, read from ``pyproject.toml`` at build time.
// Server code only — it touches the filesystem.  The package imports as
// ``lucid`` but installs as ``lucid-dl``; ``pip install lucid`` fetches an
// unrelated project, and a hand-spelled install command on the landing page
// drifted to exactly that.

import { readFileSync } from "node:fs";
import { join } from "node:path";

function projectName(): string {
  const toml = readFileSync(join(process.cwd(), "..", "pyproject.toml"), "utf8");
  // Body of the ``[project]`` table: up to the next table header.
  const table = toml.split(/^\[project\]\s*$/m)[1]?.split(/^\[/m)[0] ?? "";
  const name = table.match(/^name\s*=\s*"([^"]+)"/m)?.[1];
  if (!name) throw new Error("[distribution] ../pyproject.toml has no [project].name");
  return name;
}

/** ``[project].name`` of the repository's ``pyproject.toml``. */
export const DISTRIBUTION = projectName();

export const INSTALL_COMMAND = `pip install ${DISTRIBUTION}`;
