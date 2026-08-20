// Render the griffe API dump (`gen_api_dump.py`) to Fumadocs MDX.
//
// Derived from the Node half of fumadocs-python (MIT, (c) 2023 Fuma); see
// `gen_api_dump.py` for the full notice. Kept deliberately close to that
// original so the emitted MDX -- and therefore the `/api/...` URL scheme the
// cross-reference index in `xref.mjs` mirrors -- stays unchanged.
//
// Page granularity is the contract: one page per module (`<module>/index.mdx`)
// and one per class (`<module>/<Class>.mdx`), with functions and attributes
// rendered inline on their parent's page and addressed by a `#<name>` fragment.
// `xref.mjs` encodes the same rule, so the two must move together.

import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import { stringify } from 'yaml';

/**
 * Render a module tree to a flat list of `{path, frontmatter, content}` files.
 *
 * Hrefs keep the package's own leading segment (`/api/monoprop/...`), which
 * `write` then strips from file paths; `generate-api.mjs` realigns the two.
 */
export function convert(mod, options = {}) {
  const files = [];
  const content = [];
  const tabs = [];
  const tabContents = [];

  if (mod.description) content.push(encodeText(mod.description));
  for (const attr of mod.attributes) content.push(convertAttribute(attr));

  if (Object.keys(mod.classes).length > 0) {
    tabs.push('Class');
    const lines = [];
    for (const cls of Object.values(mod.classes)) {
      files.push(...convertClass(cls));
      lines.push(element('Card', { title: cls.name, href: getHref(cls, options) }));
    }
    tabContents.push(element('Cards', undefined, lines.join('\n')));
  }

  if (Object.keys(mod.functions).length > 0) {
    tabs.push('Functions');
    const lines = [];
    for (const func of Object.values(mod.functions)) lines.push(convertFunction(func));
    tabContents.push(lines.join('\n'));
  }

  if (Object.keys(mod.modules).length > 0) {
    tabs.push('Modules');
    const lines = [];
    for (const submod of Object.values(mod.modules)) {
      files.push(...convert(submod, options));
      lines.push(element('Card', { href: getHref(submod, options), title: submod.name }));
    }
    tabContents.push(element('Cards', undefined, lines.join('\n')));
  }

  if (tabs.length > 0) {
    content.push(
      element(
        'Tabs',
        { items: tabs },
        tabContents.map((tab, i) => element('Tab', { value: tabs[i] }, tab)).join('\n'),
      ),
    );
  }

  files.push({
    path: [...mod.path.split('.'), 'index.mdx'].join('/'),
    frontmatter: { title: mod.name },
    content: content.join('\n\n'),
  });
  return files;
}

function convertClass(cls) {
  const content = [];

  if (cls.description) content.push(encodeText(cls.description));
  if (cls.attributes.length > 0) content.push(heading(2, 'Attributes'));
  for (const attr of cls.attributes) content.push(convertAttribute(attr));
  if (Object.keys(cls.functions).length > 0) {
    content.push(heading(2, 'Functions'));
    for (const func of Object.values(cls.functions)) content.push(convertFunction(func));
  }

  return [
    {
      path: cls.path.replaceAll('.', '/') + '.mdx',
      frontmatter: { title: cls.name },
      content: content.join('\n\n'),
    },
  ];
}

function convertFunction(func) {
  return element(
    'PyFunction',
    { name: func.name, type: func.signature },
    [
      func.description ? encodeText(func.description) : null,
      convertDoc(func.docstring ?? []),
      func.source.length > 0
        ? element('PySourceCode', undefined, codeblock('python', func.source))
        : null,
      func.parameters.length > 0
        ? element('div', undefined, func.parameters.map(convertParameter).join('\n'))
        : null,
      element(
        'PyFunctionReturn',
        { type: func.returns.annotation },
        [func.returns.description ? encodeText(func.returns.description) : null]
          .filter(Boolean)
          .join('\n'),
      ),
    ]
      .filter(Boolean)
      .join('\n\n'),
  );
}

function convertParameter(param) {
  const lines = [];
  if (param.description) {
    lines.push(
      typeof param.description === 'string' ? param.description : convertDoc(param.description),
    );
  }
  return element(
    'PyParameter',
    { name: param.name, type: param.annotation, value: param.value },
    lines.join('\n'),
  );
}

function convertAttribute(attribute) {
  return element(
    'PyAttribute',
    { name: attribute.name, type: attribute.annotation, value: attribute.value },
    [attribute.description ? convertDoc(attribute.description) : null].filter(Boolean).join('\n'),
  );
}

/** Render the docstring sections `simplify_docstring` left in `remainder`. */
function convertDoc(docstring) {
  const lines = [];
  for (const item of docstring) {
    if (item.kind === 'text') lines.push(encodeText(item.value));
    if (item.kind === 'admonition') {
      lines.push(
        element(
          'Callout',
          { title: item.title, type: item.value.annotation },
          encodeText(item.value.description),
        ),
      );
    }
  }
  return lines.join('\n\n');
}

function heading(depth, content) {
  return ['#'.repeat(depth), content].join(' ');
}

function codeblock(meta, code) {
  const delimit = '```';
  return `${delimit}${meta}\n${code.replaceAll(delimit, '\\```')}\n${delimit}`;
}

/** Emit an MDX element, passing every prop as an expression (`name={"x"}`). */
function element(name, props = {}, children) {
  const propsStr = [];
  for (const key in props) propsStr.push(`${key}={${JSON.stringify(props[key])}}`);
  if (children) {
    return `<${name} ${propsStr.join(' ')}>\n\n${children}\n\n</${name}>`;
  }
  return `<${name} ${propsStr.join(' ')} />`;
}

function getHref(ele, options) {
  const { baseUrl = '/' } = options;
  return (
    '/' + [...baseUrl.split('/'), ...ele.path.split('.')].filter((v) => v.length > 0).join('/')
  );
}

// MDX parses `<` as JSX and `{` as an expression, so docstring prose has to
// escape both.
function encodeText(v) {
  return v.replaceAll('<', '\\<').replaceAll('{', '\\{').replaceAll('}', '\\}');
}

/**
 * Write the rendered files under `outDir`, dropping the leading path segment
 * (the package name) so `monoprop/circuit/...` lands at `<outDir>/circuit/...`.
 */
export async function write(output, options = {}) {
  await Promise.all(
    output.map(async (file) => {
      const filePath = path.resolve(
        options.outDir ?? './',
        file.path.split('/').slice(1).join('/'),
      );
      await fs.mkdir(path.dirname(filePath), { recursive: true });
      await fs.writeFile(
        filePath,
        Object.keys(file.frontmatter).length > 0
          ? `${frontmatter(file.frontmatter)}\n\n${file.content}`
          : file.content,
      );
    }),
  );
}

export function frontmatter(obj) {
  return `---\n${stringify(obj, { compat: 'yaml-1.1', singleQuote: true }).trim()}\n---`;
}
