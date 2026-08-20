// Generate the Python API reference MDX from the griffe JSON produced by
// `gen_api_dump.py` (see the `gen-api` recipe in the justfile).
//
// `api-mdx.mjs`'s `convert`/`write` do the heavy lifting; this script only
//  1. prunes the griffe dump to the public surface (minus private `_`-prefixed members),
//  2. fixes the package-name segment that `convert` puts in cross-links but
//     `write` strips from file paths, and
//  3. emits a `meta.json` so the API section is ordered like the old reference.
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import { convert, write } from './api-mdx.mjs';
import { API_BASE_URL, buildXrefMap, resolve } from './xref.mjs';

const JSON_PATH = path.resolve('monoprop.json');
const OUT_DIR = path.resolve('content/docs/api');

// Public modules, in the order they should appear in the sidebar
const MODULES = [
  ['majorana_propagator', 'Majorana Propagator'],
  ['pauli_propagator', 'Pauli Propagator'],
  ['monomial_propagator', 'Shared propagator surface'],
  ['circuit', 'Circuits and gates'],
  ['fermi', 'Fermionic & Majorana operators'],
  ['pauli', 'Qubit (Pauli) operators'],
  ['qiskit_conversion', 'Qiskit conversion'],
  ['integral_conversion', 'Integral conversion'],
  ['utils', 'Utilities'],
  ['majorana', 'Internal representations'],
  ['quantum_data', 'Protocols'],
  ['exceptions', 'Exceptions'],
];
const KEEP = new Set(MODULES.map(([name]) => name));
const MODULE_GITHUB_PATHS = new Map(MODULES.map(([name]) => [name, `src/monoprop/${name}.py`]));

/** Drop single-underscore (non-dunder) members from a {name: node} map. */
function prunePrivate(map) {
  for (const name of Object.keys(map)) {
    if (name.startsWith('_') && !name.startsWith('__')) delete map[name];
  }
}

/** Recursively strip private classes/functions/submodules from a module node. */
function pruneModule(mod) {
  prunePrivate(mod.classes ?? {});
  prunePrivate(mod.functions ?? {});
  for (const cls of Object.values(mod.classes ?? {})) prunePrivate(cls.functions ?? {});
  for (const sub of Object.values(mod.modules ?? {})) pruneModule(sub);
}

/**
 * Derive the fully-qualified scope a rendered page documents, from its file
 * path (before `write` strips the leading package segment):
 *   `monoprop/circuit/index.mdx`   -> `monoprop.circuit`            (module)
 *   `monoprop/circuit/ExpGate.mdx` -> `monoprop.circuit.ExpGate`    (class)
 *   `monoprop/index.mdx`           -> `monoprop`                    (package)
 */
function scopeFromPath(filePath) {
  return filePath
    .replace(/\/index\.mdx$/, '')
    .replace(/\.mdx$/, '')
    .split('/')
    .join('.');
}

/**
 * Resolve cross-references in a rendered MDX page into real links. Targets are
 * Python paths, resolved fully-qualified or relative to `scope`, the page's own
 * fully-qualified scope (see `scopeFromPath`):
 *   `[Circuit][monoprop.circuit.Circuit]`  ->  `[Circuit](/api/circuit/Circuit)`
 *   `[ExpGate][]`  (on a page scoped to `monoprop.circuit`)  ->  `[ExpGate](/api/circuit/ExpGate)`
 * Unresolved targets are collected in `unresolved` (surfaced by the caller) and
 * left untouched.
 */
function resolveXrefs(content, xref, unresolved, scope) {
  // Avoid double-wrapping a display text the docstring already backticked.
  const code = (text) => (/^`.*`$/.test(text) ? text : `\`${text}\``);

  return content.replaceAll(/\[([^\]]+)\]\[([^\]]*)\]/g, (match, display, target) => {
    const name = (target || display).trim().replace(/^`|`$/g, '');
    const url = resolve(name, scope, xref);
    if (url) return `[${code(display)}](${url})`;
    unresolved.add(name);
    return match;
  });
}

function foldToOneLine(text) {
  return String(text).replace(/\s*\n\s*/g, ' ').replace(/\s+/g, ' ').trim();
}

/**
 * Unescape braces within LaTeX/KaTeX math expressions.
 *
 * Upstream markdown serialization escapes braces (`\{` and `\}`), but KaTeX
 * requires unescaped braces for proper grouping (subscripts, superscripts, and
 * command arguments). This function selectively unescapes braces only inside
 * `$...$` (inline) and `$$...$$` (display) math spans, e.g. `$r_\{12\}$` -> `$r_{12}$`.
 */
function normalizeMathGroupingEscapes(content) {
  const normalizeMathSegment = (math) =>
    math
      .replace(/\\\{/g, '{')
      .replace(/\\\}/g, '}');

  return content.replace(/\$\$[\s\S]*?\$\$|\$(?:\\.|[^$\\])+\$/g, (math) =>
    normalizeMathSegment(math),
  );
}

/**
 * Extract the raw `Returns:` block from a function source docstring.
 * We prefer source extraction because some upstream parsed `returns.description`
 * values are already truncated to the first wrapped line.
 */
function extractReturnsFromSource(source) {
  if (typeof source !== 'string' || source.length === 0) return null;

  // Find a Returns section inside the function source, ending at the next
  // section header (e.g. Args:, Raises:, Note:) or the closing docstring.
  const match = source.match(
    /\n\s+Returns:\n([\s\S]*?)(?=\n\s+[A-Z][A-Za-z_ ]+:\n|\n\s+["']{3})/,
  );
  if (!match) return null;

  const text = match[1]
    .split('\n')
    .map((line) => line.replace(/^\s+/, ''))
    .join(' ')
    .trim();

  return text || null;
}

/**
 * griffe preserves wrapped descriptions in parsed docstring sections, and the
 * dump's `returns` entry truncates multi-line Returns text in some cases, so we
 * re-read it from the function source and fold it to one line before
 * conversion. Fixing this in `gen_api_dump.py` would retire this whole path.
 */
function normalizeFunctionReturnsSection(funcNode) {
  const fromSource = extractReturnsFromSource(funcNode?.source);
  if (fromSource && funcNode?.returns && typeof funcNode.returns === 'object') {
    funcNode.returns.description = foldToOneLine(fromSource);
  }

  const parsed = funcNode?.docstring?.parsed;
  if (!Array.isArray(parsed)) return;

  for (const section of parsed) {
    if (section?.kind !== 'returns') continue;

    // Depending on parser shape, `value` can be a list of return entries or a
    // single entry-like object.
    const values = Array.isArray(section.value) ? section.value : [section.value];
    for (const item of values) {
      if (item && typeof item.description === 'string') {
        item.description = foldToOneLine(item.description);
      }
    }
  }
}

function normalizeFunctionReturns(mod) {
  if (!mod || typeof mod !== 'object') return;

  for (const fn of Object.values(mod.functions ?? {})) {
    normalizeFunctionReturnsSection(fn);
  }

  for (const cls of Object.values(mod.classes ?? {})) {
    for (const fn of Object.values(cls.functions ?? {})) {
      normalizeFunctionReturnsSection(fn);
    }
  }

  for (const sub of Object.values(mod.modules ?? {})) {
    normalizeFunctionReturns(sub);
  }
}

async function main() {
  const pkg = JSON.parse(await fs.readFile(JSON_PATH, 'utf8'));

  // Keep only the documented public submodules.
  pkg.modules = Object.fromEntries(
    Object.entries(pkg.modules ?? {}).filter(([name]) => KEEP.has(name)),
  );
  pkg.attributes = []; // drop the package-level __all__ dump
  pruneModule(pkg);
  normalizeFunctionReturns(pkg);

  const xref = buildXrefMap(pkg);
  const unresolved = new Set();

  const files = convert(pkg, { baseUrl: API_BASE_URL });

  for (const file of files) {
    file.content = resolveXrefs(file.content, xref, unresolved, scopeFromPath(file.path));

    // `convert` keeps the package name ("monoprop") in hrefs, but `write`
    // strips that leading segment from file paths. Realign the links.
    file.content = file.content.replaceAll(`${API_BASE_URL}/monoprop`, API_BASE_URL);

    // Attach source paths to frontmatter so docs can build GitHub links
    // without duplicating the module list in app code.
    const modulePrefixMatch = file.path.match(/^monoprop\/([^/]+)(?:\/|$)/);
    const moduleName = modulePrefixMatch?.[1];
    const githubPath = moduleName ? MODULE_GITHUB_PATHS.get(moduleName) : undefined;
    if (githubPath) {
      file.frontmatter.githubPath = githubPath;
    }
    file.content = normalizeMathGroupingEscapes(file.content);

    // Add titles to the frontmatter for the modules we documented. The title
    // is used in the sidebar and in the page's <title> tag.
    const m = file.path.match(/^monoprop\/([^/]+)\/index\.mdx$/);
    if (m) {
      const entry = MODULES.find(([name]) => name === m[1]);
      if (entry) file.frontmatter.title = entry[0]; // use entry[1] for the human-readable title if you want it in the sidebar
    }
    if (file.path === 'monoprop/index.mdx') {
      file.frontmatter.title = 'Python API';
      file.frontmatter.description = 'Generated reference for the public monoprop Python package.';
      file.frontmatter.githubPath = 'src/monoprop/__init__.py';
    }
  }

  if (unresolved.size > 0) {
    console.warn(
      `Warning: ${unresolved.size} cross-reference target(s) did not resolve and were left as-is:\n  ` +
        [...unresolved].sort().join('\n  '),
    );
  }

  await fs.rm(OUT_DIR, { recursive: true, force: true });
  await write(files, { outDir: OUT_DIR });

  // Sidebar ordering for the API section. `index.mdx` is intentionally omitted:
  // fumadocs treats it as the folder's own index (so the "Python API" title
  // links to `/api`); listing it would add a redundant child duplicating
  // the section title in the sidebar.
  await fs.writeFile(
    path.join(OUT_DIR, 'meta.json'),
    JSON.stringify({ title: 'Python API', pages: [...MODULES.map(([n]) => n)] }, null, 2),
  );

  console.log(`Wrote ${files.length} API pages to ${OUT_DIR}`);
}

await main();
