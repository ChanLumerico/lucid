// Cross-link loader — exposes Python ↔ C++ symbol mapping built by
// `scripts/build-cross-links.py` at prebuild time.
//
// UI components call ``cppFor(pythonPath)`` and ``pythonFor(cppName)``
// to find the corresponding implementation on the other side of the
// pybind11 boundary; the data is keyed by the same identifiers the
// detail-page routes use (slug + member name).

import { readFileSync } from "node:fs";
import { join } from "node:path";

export interface CrossLinkCppRef {
  /** C++ symbol name as it appears in the engine module
   *  (e.g. ``"LinearBackward"``, ``"linear_op"``). */
  name: string;
  /** Whether this is the autograd backward node (typically a class) or
   *  the free-function dispatch wrapper. */
  kind: "backward_node" | "free_function";
}

export interface CrossLinkPythonRef {
  /** Fully-qualified Python path (e.g. ``"lucid.nn.Linear"``). */
  path: string;
  /** Owning module slug (``"lucid.nn"``).  Useful for building the URL
   *  without re-parsing the path. */
  module: string;
  kind: "class" | "function";
}

interface CrossLinksFile {
  python_to_cpp: Record<string, CrossLinkCppRef[]>;
  cpp_to_python: Record<string, CrossLinkPythonRef[]>;
}

const FILE = join(process.cwd(), "public", "api-data", "_cross_links.json");

const FALLBACK: CrossLinksFile = {
  python_to_cpp: {},
  cpp_to_python: {},
};

function load(): CrossLinksFile {
  try {
    return JSON.parse(readFileSync(FILE, "utf-8")) as CrossLinksFile;
  } catch {
    return FALLBACK;
  }
}

const data = load();

/** C++ engine symbols implementing the given Python symbol, or empty
 *  when no mapping is known.  ``pythonPath`` should be the fully-qualified
 *  path (slug + member name), e.g. ``"lucid.nn.Linear"``. */
export function cppFor(pythonPath: string): CrossLinkCppRef[] {
  return data.python_to_cpp[pythonPath] ?? [];
}

/** Python symbols wrapping the given C++ symbol — typically a 1:1 for
 *  ops, 1:N for templated backward nodes (``ConvNdBackward`` →
 *  ``Conv1d / Conv2d / Conv3d``). */
export function pythonFor(cppName: string): CrossLinkPythonRef[] {
  return data.cpp_to_python[cppName] ?? [];
}

/** Construct the docs href for a C++ engine symbol given its bare name. */
export function cppHref(cppName: string): string {
  return `/api/lucid._C.engine/${cppName}`;
}

/** Construct the docs href for a Python symbol given its qualified path
 *  and owning module slug.  The detail-page routing key is ``module/name``
 *  (everything after the last dot is the member name within ``module``). */
export function pythonHref(ref: CrossLinkPythonRef): string {
  // ref.path === "<module>.<name>"; strip the module prefix to recover name.
  const name = ref.path.startsWith(`${ref.module}.`)
    ? ref.path.slice(ref.module.length + 1)
    : ref.path;
  return `/api/${ref.module}/${name}`;
}
