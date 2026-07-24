// Generate the Python API reference MDX from the griffe JSON produced by
// `fumapy-generate monoprop` (see the `gen-api` recipe in the justfile).
//
// fumadocs-python's `convert`/`write` do the heavy lifting; this script only
//  1. prunes the griffe dump to the public surface (the modules that the old
//     docs reference documented, minus private `_`-prefixed members),
//  2. fixes the package-name segment that `convert` puts in cross-links but
//     `write` strips from file paths, and
//  3. emits a `meta.json` so the API section is ordered like the old reference.
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import { convert, write, frontmatter } from 'fumadocs-python';

const JSON_PATH = path.resolve('monoprop.json');
const OUT_DIR = path.resolve('content/docs/api');
const BASE_URL = '/api';

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
 * Turn a fully-qualified symbol path into the URL of the page that documents
 * it. Mirrors fumadocs-python's `getHref`, then drops the leading `monoprop`
 * segment that `write` strips from file paths (the same realignment applied to
 * the rendered content below).
 */
function pageUrl(dottedPath) {
  const url =
    '/' +
    [...BASE_URL.split('/'), ...dottedPath.split('.')].filter((v) => v.length > 0).join('/');
  return url.replace(`${BASE_URL}/monoprop`, BASE_URL);
}

/**
 * Build a map from fully-qualified symbol path to the doc URL that documents it.
 *
 * Classes and modules get their own page. Functions, methods and attributes are
 * rendered inline on their parent's page with no per-member anchor (see the
 * `PyFunction`/`PyAttribute` components), so they resolve to that parent page.
 */
function buildXrefMap(pkg) {
  const map = new Map();
  const visit = (mod) => {
    map.set(mod.path, pageUrl(mod.path));
    for (const fn of Object.values(mod.functions ?? {})) map.set(fn.path, pageUrl(mod.path));
    for (const attr of mod.attributes ?? []) map.set(`${mod.path}.${attr.name}`, pageUrl(mod.path));
    for (const cls of Object.values(mod.classes ?? {})) {
      const clsUrl = pageUrl(cls.path);
      map.set(cls.path, clsUrl);
      for (const fn of Object.values(cls.functions ?? {})) map.set(fn.path, clsUrl);
      for (const attr of cls.attributes ?? []) map.set(`${cls.path}.${attr.name}`, clsUrl);
    }
    for (const sub of Object.values(mod.modules ?? {})) visit(sub);
  };
  visit(pkg);
  return map;
}

/**
 * Resolve cross-references in rendered MDX into real links, using `xref`
 * (fully-qualified path -> page URL). Two input syntaxes are supported:
 *
 *  1. Markdown reference links whose target is a Python path -- the idiomatic
 *     griffe/mkdocstrings form, and the one docstrings should migrate to:
 *       `[Circuit][monoprop.circuit.Circuit]`  ->  `[Circuit](/api/circuit/Circuit)`
 *       `[monoprop.circuit.Circuit][]`          ->  `[...](/api/circuit/Circuit)`
 *
 *  2. Legacy Sphinx roles (`:class:`, `:meth:`, `:func:`, `:attr:`), so a
 *     partial migration still builds. A qualified target becomes a link; a bare
 *     or unknown one degrades to a code span (the previous behaviour). With a
 *     leading `~` the display text is shortened to the final path segment.
 *
 * Unresolved targets are collected in `unresolved` (surfaced by the caller) and
 * left untouched rather than silently dropped.
 */
function resolveXrefs(content, xref, unresolved) {
  content = content.replaceAll(/\[([^\]]+)\]\[([^\]]*)\]/g, (match, display, target) => {
    const fqn = (target || display).trim().replace(/^`|`$/g, '');
    const url = xref.get(fqn);
    if (url) return `[${display}](${url})`;
    unresolved.add(fqn);
    return match;
  });

  content = content.replaceAll(/:\w+:`([^`]+)`/g, (match, inner) => {
    const shortDisplay = inner.startsWith('~');
    const fqn = inner.replace(/^~/, '');
    const display = shortDisplay ? fqn.split('.').pop() : fqn;
    const url = xref.get(fqn);
    return url ? `[${display}](${url})` : `\`${display}\``;
  });

  return content;
}

/** Collapse internal whitespace/newlines to a single line. */
function foldToOneLine(text) {
  return String(text).replace(/\s*\n\s*/g, ' ').replace(/\s+/g, ' ').trim();
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
 * griffe preserves wrapped descriptions in parsed docstring sections. The
 * fumadocs-python renderer currently truncates multi-line Returns text in some
 * cases, so we normalize function Returns descriptions to one line before
 * conversion.
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

/** Recursively visit modules/classes and normalize function Returns text. */
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

  const files = convert(pkg, { baseUrl: BASE_URL });

  for (const file of files) {
    // Resolve cross-references (Markdown `[text][path]` and legacy Sphinx
    // roles) into real links before other processing.
    file.content = resolveXrefs(file.content, xref, unresolved);

    // `convert` keeps the package name ("monoprop") in hrefs, but `write`
    // strips that leading segment from file paths. Realign the links.
    file.content = file.content.replaceAll(`${BASE_URL}/monoprop`, BASE_URL);

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
