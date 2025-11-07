const path = require("path");

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    optimizePackageImports: ["framer-motion", "lucide-react"],
  },
  transpilePackages: ["three", "three-stdlib", "web-ifc", "web-ifc-three"],
  async rewrites() {
    return [
      {
        source: "/_next/static/chunks/wasm/web-ifc/:path*",
        destination: "/wasm/web-ifc/:path*",
      },
    ];
  },
  webpack: (config) => {
    config.experiments = {
      ...(config.experiments || {}),
      asyncWebAssembly: true,
    };
    config.resolve = config.resolve || {};
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      "three/examples/jsm/utils/BufferGeometryUtils": path.resolve(__dirname, "lib/three-buffer-geometry-utils.js"),
      "three/examples/jsm/utils/BufferGeometryUtils.js": path.resolve(__dirname, "lib/three-buffer-geometry-utils.js"),
    };
    return config;
  },
};

module.exports = nextConfig;


