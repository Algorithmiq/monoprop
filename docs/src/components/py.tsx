// The component vocabulary of the generated Python API reference: the MDX under
// `content/docs/api/` is written in terms of these, by `scripts/api-mdx.mjs`.
//
// Derived from the React half of fumadocs-python (MIT, (c) 2023 Fuma); see
// `scripts/gen_api_dump.py` for the full notice.
//
// `PyFunction` and `PyAttribute` carry `id={name}`: that is the `#<member>`
// fragment `buildXrefMap` (scripts/xref.mjs) appends for members, which are
// documented inline on their parent's page rather than on one of their own.

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from 'fumadocs-ui/components/ui/collapsible';
import { buttonVariants } from 'fumadocs-ui/components/ui/button';
import { highlight } from 'fumadocs-core/highlight';
import { ChevronRight } from 'lucide-react';
import { cn } from '@/lib/cn';
import type { ComponentProps, ReactNode } from 'react';

export { Tab, Tabs } from 'fumadocs-ui/components/tabs';

// The `scroll-mt` keeps a card clear of the sticky header when an anchor jumps
// to it, so it belongs with the card surface itself, not the anchored members.
const card = 'bg-fd-card rounded-lg text-sm my-6 p-3 border scroll-mt-24';

/** The kind badge in a card's header; the colours come from `global.css`. */
function Badge({ kind }: { kind: 'func' | 'attribute' | 'param' }) {
  const color = {
    func: 'bg-fdpy-func/10 text-fdpy-func border-fdpy-func/50',
    attribute: 'bg-fdpy-attribute/10 text-fdpy-attribute border-fdpy-attribute/50',
    param: 'bg-fd-primary/10 text-fd-primary border-fd-primary/10',
  }[kind];
  return (
    <code className={cn('text-xs font-medium border p-1 rounded-lg not-prose', color)}>{kind}</code>
  );
}

/**
 * A syntax-highlighted code span. Async (it awaits shiki), so it only ever
 * renders on the server -- which is fine, every caller here is a server
 * component. Everything rendered here is Python, so the language is not a prop.
 */
async function InlineCode({ code, ...rest }: { code: string } & ComponentProps<'span'>) {
  return highlight(code, {
    lang: 'python',
    components: {
      pre: (props) => <span {...props} {...rest} className={cn(rest.className, props.className)} />,
    },
  });
}

export function PyFunction(props: { name: string; type: string; children?: ReactNode }) {
  return (
    <figure id={props.name} className={card}>
      <div className="flex gap-2 items-center font-mono flex-wrap mb-4">
        <Badge kind="func" />
        {props.name}
        <InlineCode
          className="not-prose text-xs text-fd-muted-foreground"
          code={props.type}
        />
      </div>
      <div className="text-fd-muted-foreground prose-no-margin">{props.children}</div>
    </figure>
  );
}

export function PyAttribute(props: {
  name: string;
  type?: string;
  value?: string;
  children?: ReactNode;
}) {
  return (
    <figure id={props.name} className={card}>
      <div className="flex gap-2 items-center flex-wrap font-mono mb-4">
        <Badge kind="attribute" />
        {props.name}
        {props.type && (
          <InlineCode
            className="not-prose text-fd-muted-foreground text-xs"
            code={props.type}
          />
        )}
      </div>
      <div className="text-fd-muted-foreground prose-no-margin">
        {props.value && (
          <InlineCode className="not-prose text-xs" code={`= ${props.value}`} />
        )}
        {props.children}
      </div>
    </figure>
  );
}

export function PyParameter(props: {
  name: string;
  type?: string;
  value?: string;
  children?: ReactNode;
}) {
  return (
    // Square middle corners, rounded only at the ends, so the parameters of one
    // function stack into a single seamless list.
    <div
      data-parameter=""
      className="bg-fd-secondary text-sm p-3 border shadow-md rounded-none first:rounded-t-lg last:rounded-b-lg"
    >
      <div className="flex flex-wrap gap-2 items-center font-mono text-fd-foreground">
        <Badge kind="param" />
        {props.name}
        {props.type && (
          <InlineCode
            className="ms-auto text-fd-muted-foreground not-prose text-xs"
            code={props.type}
          />
        )}
      </div>
      <div className="text-fd-muted-foreground prose-no-margin mt-4 empty:hidden">
        {props.value ? (
          <InlineCode code={`= ${props.value}`} className="not-prose text-xs" />
        ) : null}
        {props.children}
      </div>
    </div>
  );
}

export function PyFunctionReturn({ type, children }: { type?: string; children?: ReactNode }) {
  return (
    <div className="border bg-fd-secondary rounded-lg p-3 mt-2">
      <div className="flex flex-wrap gap-2 not-prose">
        <p className="font-medium me-auto">Returns</p>
        <InlineCode code={type ?? 'None'} className="text-xs" />
      </div>
      {children}
    </div>
  );
}

export function PySourceCode({ children }: { children?: ReactNode }) {
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
