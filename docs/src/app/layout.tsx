import { RootProvider } from 'fumadocs-ui/provider/next';
import { gtPlanar } from '@/lib/fonts';
import './global.css';
// KaTeX styles for math rendered by rehype-katex.
import 'katex/dist/katex.min.css';

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={gtPlanar.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
