// Shared cross-reference machinery for the docs pipeline.
//
// The API-reference generator (`generate-api.mjs`) and the prose remark plugin
// (`remark-xref.mjs`) both need to turn a fully-qualified (or scope-relative)
// Python symbol path into the doc URL that documents it. These pure helpers are
// the single source of truth for that mapping, built from the griffe dump
// `monoprop.json`.

export const API_BASE_URL = '/api';

/**
 * Turn a fully-qualified symbol path into the URL of the page that documents
 * it. Mirrors fumadocs-python's `getHref`, then drops the leading `monoprop`
 * segment that `write` strips from file paths (the same realignment applied to
 * the rendered content).
 */
export function pageUrl(dottedPath) {
  const url =
    '/' +
    [...API_BASE_URL.split('/'), ...dottedPath.split('.')].filter((v) => v.length > 0).join('/');
  return url.replace(`${API_BASE_URL}/monoprop`, API_BASE_URL);
}

/**
 * Build a map from fully-qualified symbol path to the doc URL that documents it.
 *
 * Classes and modules get their own page. Functions, methods and attributes are
 * rendered inline on their parent's page; the `getMDXComponents` wrappers give
 * each `PyFunction`/`PyAttribute` card an `id` equal to its member name, so those
 * members resolve to their parent page with a `#<name>` fragment that jumps to
 * the definition.
 */
export function buildXrefMap(pkg) {
  const map = new Map();

  // A module or class page: itself at the page top, its inline members at `#<name>`.
  const addScope = (node, pageUrl) => {
    map.set(node.path, pageUrl);
    for (const fn of Object.values(node.functions ?? {})) map.set(fn.path, `${pageUrl}#${fn.name}`);
    for (const attr of node.attributes ?? [])
      map.set(`${node.path}.${attr.name}`, `${pageUrl}#${attr.name}`);
  };

  const visit = (mod) => {
    addScope(mod, pageUrl(mod.path));
    for (const cls of Object.values(mod.classes ?? {})) addScope(cls, pageUrl(cls.path));
    for (const sub of Object.values(mod.modules ?? {})) visit(sub);
  };
  visit(pkg);
  return map;
}

/**
 * Resolve a cross-reference target to a page URL, or `null`. A fully-qualified
 * target hits `xref` directly; an unqualified (or class-relative) target is
 * resolved against the current page's scope, walking outward from the nearest
 * enclosing symbol to the package root. Nearest scope wins, so resolution is
 * deterministic -- a bare name only ever binds to something reachable from its
 * own scope chain, never to a same-named symbol in an unrelated module.
 */
export function resolve(name, scope, xref) {
  if (xref.has(name)) return xref.get(name);
  const parts = scope.split('.');
  for (let i = parts.length; i > 0; i--) {
    const url = xref.get(`${parts.slice(0, i).join('.')}.${name}`);
    if (url) return url;
  }
  return null;
}
