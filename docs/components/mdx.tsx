import defaultMdxComponents from 'fumadocs-ui/mdx';
import * as Py from 'fumadocs-python/components';
import type { MDXComponents } from 'mdx/types';

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    // Py* components (plus Tab/Tabs) used by the generated Python API pages.
    ...Py,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
