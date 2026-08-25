// Fumadocs ships no figure/caption pair, so the benchmarks page (and any other
// page with images) rolled its own caption as a plain paragraph after the
// image. This gives that pattern a real `<figure>`/`<figcaption>` and matches
// it visually to the muted, small-text captions already used for API cards.

import type { ReactNode } from 'react';

export function Figure({ children }: { children?: ReactNode }) {
  // Markdown wraps a lone image in a `<p>`, and prose typography puts its own
  // margin on both: `mb-0` on the `<p>` and `my-0` on the `<img>` (a block
  // element with no border/padding around it, so its margin would otherwise
  // collapse straight through the `<p>` and swamp `Caption`'s margin-top).
  return <figure className="my-6 [&>p]:mb-0 [&_img]:my-0">{children}</figure>;
}

export function Caption({ children }: { children?: ReactNode }) {
  return (
    <figcaption className="text-fd-muted-foreground text-sm mt-2 prose-no-margin">
      {children}
    </figcaption>
  );
}
