import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: '🤖 Automatic Facial Image Labelling System',
  description: 'AI-powered facial image analysis for age, gender, ethnicity, and emotion detection',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

