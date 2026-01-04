/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: [],
    unoptimized: true
  },
  experimental: {
    serverActions: true
  }
}

module.exports = nextConfig

