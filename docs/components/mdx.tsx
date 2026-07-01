import defaultMdxComponents from 'fumadocs-ui/mdx';
import * as Py from 'fumadocs-python/components';
import type { MDXComponents } from 'mdx/types';
import type { ImgHTMLAttributes } from 'react';

// The tutorial pages embed matplotlib figures as raw `<img src="data:…">` tags
// with no intrinsic dimensions. fumadocs' default `img` wraps Next.js' `Image`,
// which throws ("missing required width property") without width/height. Render
// those as a plain `<img>`; anything that does carry width/height (real image
// files) still goes through the optimized fumadocs component.
function Img({ src, ...props }: ImgHTMLAttributes<HTMLImageElement>) {
  const DefaultImg = defaultMdxComponents.img!;
  if (props.width == null && props.height == null) {
    // eslint-disable-next-line @next/next/no-img-element
    return <img src={src} {...props} />;
  }
  return <DefaultImg src={src} {...props} />;
}

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    // Py* components (plus Tab/Tabs) used by the generated Python API pages.
    ...Py,
    img: Img,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
