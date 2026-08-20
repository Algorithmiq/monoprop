import defaultMdxComponents from 'fumadocs-ui/mdx';
import * as Py from 'fumadocs-python/components';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from 'fumadocs-ui/components/ui/collapsible';
import { buttonVariants } from 'fumadocs-ui/components/ui/button';
import { ChevronRight } from 'lucide-react';
import { cn } from '@/lib/cn';
import type { MDXComponents } from 'mdx/types';
import type { ComponentProps, ImgHTMLAttributes, ReactElement, ReactNode } from 'react';

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

// Wrap the generated Py{Function,Attribute} cards in an anchor so cross-references
// to a specific member (e.g. `monoprop.circuit.ExpGate.__init__`) can jump to its
// definition: the `id` must stay equal to the member `name`, which is the `#<name>`
// fragment `buildXrefMap` (docs/scripts/xref.mjs) appends. `scroll-mt` keeps the
// target clear of the sticky header.
function anchored<P extends { name?: string }>(Component: (props: P) => ReactElement) {
  return function Anchored(props: P) {
    return (
      <div id={props.name} className="scroll-mt-24">
        <Component {...props} />
      </div>
    );
  };
}

const PyFunction = anchored(Py.PyFunction as (props: ComponentProps<typeof Py.PyFunction>) => ReactElement);
const PyAttribute = anchored(Py.PyAttribute as (props: ComponentProps<typeof Py.PyAttribute>) => ReactElement);

// Replaces `Py.PySourceCode` from fumadocs-python which is bugged.
function PySourceCode({ children }: { children?: ReactNode }) {
  return (
    <Collapsible className="my-6">
      <CollapsibleTrigger
        className={cn(buttonVariants({ color: 'secondary', size: 'sm', className: 'group' }))}
      >
        Source Code
        <ChevronRight className="size-3.5 text-fd-muted-foreground group-data-[state=open]:rotate-90" />
      </CollapsibleTrigger>
      <CollapsibleContent className="prose-no-margin">{children}</CollapsibleContent>
    </Collapsible>
  );
}

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    // Py* components (plus Tab/Tabs) used by the generated Python API pages.
    ...Py,
    PyFunction,
    PyAttribute,
    PySourceCode,
    img: Img,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
