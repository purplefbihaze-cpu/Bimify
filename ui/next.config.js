const path = require("path");

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    optimizePackageImports: ["framer-motion", "lucide-react"],
  },
  transpilePackages: ["three", "three-stdlib", "web-ifc", "web-ifc-three"],
  // Optimize for hot reload - ensure fast refresh works properly
  onDemandEntries: {
    // Period (in ms) where the server will keep pages in the buffer
    maxInactiveAge: 25 * 1000,
    // Number of pages that should be kept simultaneously without being disposed
    pagesBufferLength: 2,
  },
  async rewrites() {
    return [
      {
        source: "/_next/static/chunks/wasm/web-ifc/:path*",
        destination: "/wasm/web-ifc/:path*",
      },
    ];
  },
  webpack: (config, { dev, isServer }) => {
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
    
    // Optimize hot reload in development
    if (dev && !isServer) {
      config.optimization = {
        ...config.optimization,
        // Ensure fast refresh works properly
        runtimeChunk: 'single',
      };
      
      // Improve HMR performance
      config.watchOptions = {
        ...config.watchOptions,
        poll: false, // Use native file system events (faster)
        aggregateTimeout: 300, // Delay before rebuilding once the first file changed
        ignored: ['**/node_modules', '**/.git', '**/data'],
      };
    }
    
    return config;
  },
};

module.exports = nextConfig;


