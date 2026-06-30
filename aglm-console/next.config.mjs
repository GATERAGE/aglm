/** @type {import('next').NextConfig} */
const FLASK = process.env.AGLM_FLASK_URL || 'http://localhost:5000';

const nextConfig = {
  reactStrictMode: true,
  // Proxy the AGLM faculty/brain API (Flask) under /aglm/* so the console can
  // reach every original faculty without CORS or a second origin in the UI.
  async rewrites() {
    return [{ source: '/aglm/:path*', destination: `${FLASK}/api/:path*` }];
  },
};

export default nextConfig;
