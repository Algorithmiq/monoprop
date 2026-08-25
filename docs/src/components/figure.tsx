// Fumadocs ships no figure/caption pair, so the benchmarks page (and any other
// page with images) rolled its own caption as a plain paragraph after the
// image. This gives that pattern a real `<figure>`/`<figcaption>` and matches
// it visually to the muted, small-text captions already used for API cards.

import type { ReactNode } from 'react';

export function Figure({ children }: { children?: ReactNode }) {
  return <figure className="my-6">{children}</figure>;
}

export function Caption({ children }: { children?: ReactNode }) {
  return (
    <figcaption className="text-fd-muted-foreground text-sm mt-3 prose-no-margin">
      {children}
    </figcaption>
  );
}
