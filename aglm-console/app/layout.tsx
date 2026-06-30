import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'aGLM // Console',
  description: 'AGLM participant console — clean house on the Vercel AI SDK',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
