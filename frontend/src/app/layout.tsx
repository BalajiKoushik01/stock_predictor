import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Project Apex | Institutional Quantitative Equities Engine',
  description: 'An institutional-grade, zero-cost quantitative forecasting engine for Indian Equities. Leverages fractional differencing, Empirical Mode Decomposition, option chains microstructure, and FinBERT news sentiment.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body id="apex-root-body">
        {children}
      </body>
    </html>
  );
}
