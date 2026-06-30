import { Inter } from 'next/font/google';
import { Provider } from '@/components/provider';
import './global.css';
// KaTeX styles for math rendered by rehype-katex.
import 'katex/dist/katex.min.css';

const inter = Inter({
  subsets: ['latin'],
});

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={inter.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <Provider>{children}</Provider>
      </body>
    </html>
  );
}
