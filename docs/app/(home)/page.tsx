import type { Metadata } from 'next';
import Link from 'next/link';
import { redirect } from 'next/navigation';

export const metadata: Metadata = {
  // Static export renders a meta-refresh fallback from this redirect.
  title: 'monoprop',
};

export default function HomePage() {
  // The site is documentation-first: send the root straight to the docs.
  redirect('/docs');

  // Rendered only if a client lands here without following the redirect.
  return (
    <div className="flex flex-col justify-center text-center flex-1">
      <p>
        Redirecting to the{' '}
        <Link href="/docs" className="font-medium underline">
          documentation
        </Link>
        …
      </p>
    </div>
  );
}
